//! Implementation of the `wasmtime replay` command

use crate::commands::run::{Replaying, RunCommand};
use crate::common::RunTarget;
use anyhow::{Context, Result, bail};
use clap::Parser;
use std::path::PathBuf;
use std::{fs, io};
use tokio::time::error::Elapsed;
use wasmtime::{Engine, ReplayEnvironment, ReplaySettings};

#[derive(Parser)]
/// Replay-specific options for CLI.
pub struct ReplayOptions {
    /// The path of the recorded trace.
    ///
    /// Execution traces can be obtained with the -R option on other Wasmtime commands
    /// (e.g. `wasmtime run` or `wasmtime serve`). See `wasmtime run -R help` for
    /// relevant information on recording execution.
    ///
    /// Note: The module used for replay must exactly match that used during recording.
    #[arg(short, long, required = true, value_name = "RECORDED TRACE")]
    pub trace: PathBuf,

    /// Dynamic checks of record signatures to validate replay consistency.
    ///
    /// Requires record traces to be generated with `validation_metadata` enabled.
    /// This resembles an internal "safety" assert and verifies extra non-essential events
    /// (e.g. return args from Wasm function calls or entry args into host function calls) also
    /// match during replay. A failed validation will abort the replay run with an error.
    #[arg(short, long, default_value_t = false)]
    pub validate: bool,

    /// Size of static buffer needed to deserialized variable-length types like String. This is not
    /// not important for basic functional recording/replaying, but may be required to replay traces where
    /// `validate` was enabled for recording.
    #[arg(short, long, default_value_t = 64)]
    pub deserialize_buffer_size: usize,
}

/// Execute a deterministic, embedding-agnostic replay of a Wasm modules given its associated recorded trace.
#[derive(Parser)]
pub struct ReplayCommand {
    #[command(flatten)]
    replay_opts: ReplayOptions,

    #[command(flatten)]
    run_cmd: RunCommand,
}

impl ReplayCommand {
    /// Executes the command.
    pub fn execute(mut self) -> Result<()> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_time()
            .enable_io()
            .build()?;

        runtime.block_on(async {
            self.run_cmd.run.common.init_logging()?;

            let engine = self.run_cmd.new_engine(Replaying::Yes)?;
            let main = self
                .run_cmd
                .run
                .load_module(&engine, self.run_cmd.module_and_args[0].as_ref())?;

            self.run_replay(&engine, &main).await?;
            Ok(())
        })
    }

    /// Execute the store with the replay settings.
    ///
    /// Applies similar configurations to `instantiate_and_run`.
    async fn run_replay(self, engine: &Engine, main: &RunTarget) -> Result<()> {
        let opts = self.replay_opts;

        // Validate coredump-on-trap argument
        if let Some(path) = &self.run_cmd.run.common.debug.coredump {
            if path.contains("%") {
                bail!("the coredump-on-trap path does not support patterns yet.")
            }
        }

        // In general, replays will need an "almost exact" superset of
        // the run configurations, but with potentially certain different options (e.g. fuel consumption).
        let settings = ReplaySettings {
            validate: opts.validate,
            deserialize_buffer_size: opts.deserialize_buffer_size,
            ..Default::default()
        };

        let mut renv = ReplayEnvironment::new(&engine, settings);
        match &main {
            RunTarget::Core(m) => {
                renv.add_module(m.clone());
            }
            #[cfg(feature = "component-model")]
            RunTarget::Component(c) => {
                renv.add_component(c.clone());
            }
        }

        let allow_unknown_exports = self.run_cmd.run.common.wasm.unknown_exports_allow;
        let mut replay_instance = renv.instantiate_with(
            io::BufReader::new(fs::File::open(opts.trace)?),
            |store| {
                // If fuel has been configured, we want to add the configured
                // fuel amount to this store.
                if let Some(fuel) = self.run_cmd.run.common.wasm.fuel {
                    store.set_fuel(fuel)?;
                }
                Ok(())
            },
            |module_linker| {
                if let Some(enable) = allow_unknown_exports {
                    module_linker.allow_unknown_exports(enable);
                }
                Ok(())
            },
            |_component_linker| {
                if allow_unknown_exports.is_some() {
                    bail!("--allow-unknown-exports not supported with components");
                }
                Ok(())
            },
        )?;

        let dur = self
            .run_cmd
            .run
            .common
            .wasm
            .timeout
            .unwrap_or(std::time::Duration::MAX);

        let result: Result<Result<()>, Elapsed> = tokio::time::timeout(dur, async {
            replay_instance.run_to_completion_async().await
        })
        .await;

        // This is basically the same finish logic as `instantiate_and_run`.
        match result.unwrap_or_else(|elapsed| {
            Err(anyhow::Error::from(wasmtime::Trap::Interrupt))
                .with_context(|| format!("timed out after {elapsed}"))
        }) {
            Ok(_) => Ok(()),
            Err(e) => {
                if e.is::<wasmtime::Trap>() {
                    eprintln!("Error returned from replay: {e:?}");
                    cfg_if::cfg_if! {
                        if #[cfg(unix)] {
                            std::process::exit(rustix::process::EXIT_SIGNALED_SIGABRT);
                        } else if #[cfg(windows)] {
                            // https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/abort?view=vs-2019
                            std::process::exit(3);
                        }
                    }
                }
                Err(e)
            }
        }
    }
}

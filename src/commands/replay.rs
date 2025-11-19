//! Implementation of the `wasmtime replay` command

use crate::commands::run::{Host, RunCommand};
use crate::common::RunTarget;
use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use std::{fs, io};
use tokio::time::error::Elapsed;
use wasmtime::{Engine, ReplayEnvironment, ReplaySettings, Store};

#[derive(Parser)]
/// Replay-specific options for CLI
pub struct ReplayOptions {
    /// The path of the recorded trace
    ///
    /// Execution traces can be obtained for most modes of Wasmtime execution with -R.
    /// See `wasmtime run -R help` for relevant information on recording execution
    ///
    /// Note: The module used for replay must exactly match that used during recording
    #[arg(short, long, required = true, value_name = "RECORDED TRACE")]
    pub trace: PathBuf,

    /// Dynamic checks of record signatures to validate replay consistency.
    ///
    /// Requires record traces to be generated with `validation_metadata` enabled.
    #[arg(short, long, default_value_t = false)]
    pub validate: bool,

    /// Size of static buffer needed to deserialized variable-length types like String. This is not
    /// not important for basic functional recording/replaying, but may be required to replay traces where
    /// `validate` was enabled for recording
    #[arg(short, long, default_value_t = 64)]
    pub deser_buffer_size: usize,
}

/// Execute a deterministic, embedding-agnostic replay of a Wasm modules given its associated recorded trace
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

            let engine = self.run_cmd.new_engine(true)?;
            let main = self
                .run_cmd
                .run
                .load_module(&engine, self.run_cmd.module_and_args[0].as_ref())?;
            let (store, _) = self.run_cmd.new_store_and_linker(&engine, &main)?;

            self.instantiate_and_run_replay(&engine, &main, store)
                .await?;
            Ok(())
        })
    }

    /// Execute the store with the replay settings
    ///
    /// Applies similar configurations to `instantiate_and_run`
    async fn instantiate_and_run_replay(
        self,
        engine: &Engine,
        main: &RunTarget,
        store: Store<Host>,
    ) -> Result<()> {
        let opts = self.replay_opts;
        // In general, replays will need an "almost exact" superset of
        // the run configurations, but with potentially certain different options (e.g. fuel consumption).
        let settings = ReplaySettings {
            validate: opts.validate,
            deser_buffer_size: opts.deser_buffer_size,
            ..Default::default()
        };

        let mut renv = ReplayEnvironment::new(&engine, settings);
        match &main {
            RunTarget::Core(m) => {
                renv.add_module(m.clone());
            }
            #[cfg(feature = "component-model")]
            RunTarget::Component(_c) => {
                #[cfg(feature = "rr-component")]
                renv.add_component(_c.clone());
            }
        }
        let mut replay_instance =
            renv.instantiate_with_store(|| store, io::BufReader::new(fs::File::open(opts.trace)?))?;

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

        // Extract the store for error handling below
        let store = replay_instance.extract_store();

        // This is basically the same finish logic as `instantiate_and_run`
        match result.unwrap_or_else(|elapsed| {
            Err(anyhow::Error::from(wasmtime::Trap::Interrupt))
                .with_context(|| format!("timed out after {elapsed}"))
        }) {
            Ok(_) => Ok(()),
            Err(e) => {
                // Exit the process if Wasmtime understands the error;
                // otherwise, fall back on Rust's default error printing/return
                // code.
                if store.data().legacy_p1_ctx.is_some() {
                    return Err(wasi_common::maybe_exit_on_error(e));
                } else if store.data().wasip1_ctx.is_some() {
                    if let Some(exit) = e.downcast_ref::<wasmtime_wasi::I32Exit>() {
                        std::process::exit(exit.0);
                    }
                }
                if e.is::<wasmtime::Trap>() {
                    eprintln!("Error: {e:?}");
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

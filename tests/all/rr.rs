//! This module does NOT run tests with `call_async` if `component-model-async`
//! feature is disabled since async ABI is not supported for RR builds yet.

use anyhow::Result;
use std::future::Future;
use std::io::Cursor;
use std::pin::Pin;
use wasmtime::component::{Component, HasSelf, Linker as ComponentLinker, bindgen};
use wasmtime::{
    Config, Engine, Linker, Module, OptLevel, RRConfig, RecordSettings, ReplayEnvironment,
    ReplaySettings, Store,
};

struct TestState;

impl TestState {
    fn new() -> Self {
        TestState
    }
}

fn init_logger() {
    let _ = env_logger::try_init();
}

fn create_recording_engine(is_async: bool) -> Result<Engine> {
    let mut config = Config::new();
    config
        .debug_info(true)
        .cranelift_opt_level(OptLevel::None)
        .rr(RRConfig::Recording);
    #[cfg(feature = "component-model-async")]
    if is_async {
        config.async_support(true);
    }
    #[cfg(not(feature = "component-model-async"))]
    {
        let _ = is_async;
    }
    Engine::new(&config)
}

fn create_replay_engine(is_async: bool) -> Result<Engine> {
    let mut config = Config::new();
    config
        .debug_info(true)
        .cranelift_opt_level(OptLevel::None)
        .rr(RRConfig::Replaying);
    #[cfg(feature = "component-model-async")]
    if is_async {
        config.async_support(true);
    }
    #[cfg(not(feature = "component-model-async"))]
    {
        let _ = is_async;
    }
    Engine::new(&config)
}

/// Run a core module test with recording and replay
fn run_core_module_test<F, R>(module_wat: &str, setup_linker: F, test_fn: R) -> Result<()>
where
    F: Fn(&mut Linker<TestState>, bool) -> Result<()>,
    R: for<'a> Fn(
        &'a mut Store<TestState>,
        &'a wasmtime::Instance,
        bool,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>,
{
    init_logger();

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    #[cfg(feature = "component-model-async")]
    let async_modes = [false];
    #[cfg(not(feature = "component-model-async"))]
    let async_modes = [false, true];
    // Run with in sync/async mode with/without validation
    for is_async in async_modes {
        for validation in [true, false] {
            let run = async {
                run_core_module_test_with_validation(
                    module_wat,
                    &setup_linker,
                    &test_fn,
                    validation,
                    is_async,
                )
                .await?;
                Ok::<(), anyhow::Error>(())
            };

            rt.block_on(run)?;
        }
    }

    Ok(())
}

async fn run_core_module_test_with_validation<F, R>(
    module_wat: &str,
    setup_linker: &F,
    test_fn: &R,
    validate: bool,
    is_async: bool,
) -> Result<()>
where
    F: Fn(&mut Linker<TestState>, bool) -> Result<()>,
    R: for<'a> Fn(
        &'a mut Store<TestState>,
        &'a wasmtime::Instance,
        bool,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>,
{
    // === RECORDING PHASE ===
    let engine = create_recording_engine(is_async)?;
    let module = Module::new(&engine, module_wat)?;

    let mut linker = Linker::new(&engine);
    setup_linker(&mut linker, is_async)?;

    let writer: Cursor<Vec<u8>> = Cursor::new(Vec::new());
    let mut store = Store::new(&engine, TestState::new());
    let record_settings = RecordSettings {
        add_validation: validate,
        ..Default::default()
    };
    store.record(writer, record_settings)?;

    let instance = if is_async {
        linker.instantiate_async(&mut store, &module).await?
    } else {
        linker.instantiate(&mut store, &module)?
    };

    test_fn(&mut store, &instance, is_async).await?;

    // Extract the recording
    let trace_box = store.into_record_writer()?;
    let any_box: Box<dyn std::any::Any> = trace_box;
    let mut trace_reader = any_box.downcast::<Cursor<Vec<u8>>>().unwrap();
    trace_reader.set_position(0);

    // === REPLAY PHASE ===
    let engine = create_replay_engine(is_async)?;
    let module = Module::new(&engine, module_wat)?;

    let replay_settings = ReplaySettings {
        validate,
        ..Default::default()
    };
    let mut renv = ReplayEnvironment::new(&engine, replay_settings);
    renv.add_module(module);

    let mut replay_instance = renv.instantiate(*trace_reader)?;
    if is_async {
        replay_instance.run_to_completion_async().await?;
    } else {
        replay_instance.run_to_completion()?;
    }

    Ok(())
}

/// Run a component test with recording and replay, testing both with and without validation
fn run_component_test<F, R>(component_wat: &str, setup_linker: F, test_fn: R) -> Result<()>
where
    F: Fn(&mut ComponentLinker<TestState>) -> Result<()> + Clone,
    R: for<'a> Fn(
            &'a mut Store<TestState>,
            &'a wasmtime::component::Instance,
            bool,
        ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>
        + Clone,
{
    init_logger();

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    #[cfg(feature = "component-model-async")]
    let async_modes = [false];
    #[cfg(not(feature = "component-model-async"))]
    let async_modes = [false, true];

    // Run with in sync/async mode with/without validation
    for is_async in async_modes {
        for validation in [true, false] {
            let run = async {
                run_component_test_with_validation(
                    component_wat,
                    setup_linker.clone(),
                    test_fn.clone(),
                    validation,
                    is_async,
                )
                .await?;
                Ok::<(), anyhow::Error>(())
            };

            rt.block_on(run)?;
        }
    }

    Ok(())
}

/// Run a component test with recording and replay with specified validation setting
async fn run_component_test_with_validation<F, R>(
    component_wat: &str,
    setup_linker: F,
    test_fn: R,
    validate: bool,
    is_async: bool,
) -> Result<()>
where
    F: Fn(&mut ComponentLinker<TestState>) -> Result<()>,
    R: for<'a> Fn(
        &'a mut Store<TestState>,
        &'a wasmtime::component::Instance,
        bool,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>,
{
    // === RECORDING PHASE ===
    log::info!("Recording | Validate: {validate}, Async: {is_async}");
    let engine = create_recording_engine(is_async)?;
    let component = Component::new(&engine, component_wat)?;

    let mut linker = ComponentLinker::new(&engine);
    setup_linker(&mut linker)?;

    let writer: Cursor<Vec<u8>> = Cursor::new(Vec::new());
    let mut store = Store::new(&engine, TestState::new());
    let record_settings = RecordSettings {
        add_validation: validate,
        ..Default::default()
    };
    store.record(writer, record_settings)?;

    let instance = if is_async {
        linker.instantiate_async(&mut store, &component).await?
    } else {
        linker.instantiate(&mut store, &component)?
    };

    test_fn(&mut store, &instance, is_async).await?;

    // Extract the recording
    let trace_box = store.into_record_writer()?;
    let any_box: Box<dyn std::any::Any> = trace_box;
    let mut trace_reader = any_box.downcast::<Cursor<Vec<u8>>>().unwrap();
    trace_reader.set_position(0);

    // === REPLAY PHASE ===
    log::info!("Replaying | Validate: {validate}, Async: {is_async}");
    let engine = create_replay_engine(is_async)?;
    let component = Component::new(&engine, component_wat)?;
    let replay_settings = ReplaySettings {
        validate,
        ..Default::default()
    };
    let mut renv = ReplayEnvironment::new(&engine, replay_settings);
    renv.add_component(component);

    let mut replay_instance = renv.instantiate(*trace_reader)?;
    if is_async {
        replay_instance.run_to_completion_async().await?;
    } else {
        replay_instance.run_to_completion()?;
    }

    Ok(())
}

// ============================================================================
// Core Module Tests
// ============================================================================

#[test]
fn test_core_module_with_host_double() -> Result<()> {
    let module_wat = r#"
        (module
            (import "env" "double" (func $double (param i32) (result i32)))
            (func (export "main") (param i32) (result i32)
                local.get 0
                call $double
            )
        )
    "#;

    run_core_module_test(
        module_wat,
        |linker, _| {
            linker.func_wrap("env", "double", |param: i32| param * 2)?;
            Ok(())
        },
        |store, instance, is_async| {
            Box::pin(async move {
                let run = instance.get_typed_func::<i32, i32>(&mut *store, "main")?;
                let result = if is_async {
                    let result = run.call_async(&mut *store, 42).await?;
                    run.call_async(&mut *store, result).await?
                } else {
                    let result = run.call(&mut *store, 42)?;
                    run.call(&mut *store, result)?
                };
                assert_eq!(result, 168);
                Ok(())
            })
        },
    )
}

#[test]
fn test_core_module_with_multiple_host_imports() -> Result<()> {
    let module_wat = r#"
        (module
            (import "env" "double" (func $double (param i32) (result i32)))
            (import "env" "complex" (func $complex (param i32 i64) (result i32 i64 f32)))
            (func (export "main") (param i32) (result i32)
                local.get 0
                call $double
                call $double
                i64.const 10
                call $complex
                drop
                drop
                i64.const 5
                call $complex
                drop
                drop
            )
        )
    "#;

    run_core_module_test(
        module_wat,
        |linker, _| {
            linker.func_wrap("env", "double", |param: i32| param * 2)?;
            linker.func_wrap("env", "complex", |p1: i32, p2: i64| -> (i32, i64, f32) {
                ((p1 as f32).sqrt() as i32, (p1 * p1) as i64 * p2, 8.66)
            })?;
            Ok(())
        },
        |store, instance, is_async| {
            Box::pin(async move {
                let run = instance.get_typed_func::<i32, i32>(&mut *store, "main")?;
                let result = if is_async {
                    run.call_async(&mut *store, 42).await?
                } else {
                    run.call(&mut *store, 42)?
                };
                assert_eq!(result, 3); // sqrt(sqrt(42*2*2)) = sqrt(12) = 3
                Ok(())
            })
        },
    )
}

#[test]
fn test_core_module_reentrancy() -> Result<()> {
    let module_wat = r#"
        (module
            (import "env" "host_call" (func $host_call (param i32) (result i32)))
            (func (export "main") (param i32) (result i32)
                local.get 0
                call $host_call
            )
            (func (export "wasm_callback") (param i32) (result i32)
                local.get 0
                i32.const 1
                i32.add
            )
        )
    "#;

    run_core_module_test(
        module_wat,
        |linker, is_async| {
            if is_async {
                linker.func_wrap_async(
                    "env",
                    "host_call",
                    |mut caller: wasmtime::Caller<'_, TestState>, (param,): (i32,)| {
                        Box::new(async move {
                            let func = caller
                                .get_export("wasm_callback")
                                .unwrap()
                                .into_func()
                                .unwrap();
                            let typed = func.typed::<i32, i32>(&caller)?;
                            typed.call_async(&mut caller, param).await
                        })
                    },
                )?;
            } else {
                linker.func_wrap(
                    "env",
                    "host_call",
                    |mut caller: wasmtime::Caller<'_, TestState>,
                     param: i32|
                     -> wasmtime::Result<i32> {
                        let func = caller
                            .get_export("wasm_callback")
                            .unwrap()
                            .into_func()
                            .unwrap();
                        let typed = func.typed::<i32, i32>(&caller)?;
                        typed.call(&mut caller, param)
                    },
                )?;
            }
            Ok(())
        },
        |store, instance, is_async| {
            Box::pin(async move {
                let run = instance.get_typed_func::<i32, i32>(&mut *store, "main")?;
                let result = if is_async {
                    run.call_async(&mut *store, 42).await?
                } else {
                    run.call(&mut *store, 42)?
                };
                assert_eq!(result, 43);
                Ok(())
            })
        },
    )
}

#[test]
#[should_panic]
fn test_recording_panics_for_core_module_memory_export() {
    let module_wat = r#"
        (module
            (memory (export "memory") 1)
        )
    "#;

    run_core_module_test(
        module_wat,
        |_, _| Ok(()),
        |_, _, _| Box::pin(async { Ok(()) }),
    )
    .unwrap();
}

// ============================================================================
// Component Model Tests with Host Imports
// ============================================================================

// Few Parameters and Few Results (not exceeding MAX_FLAT_PARAMS=16 and
// MAX_FLAT_RESULTS=1)
#[test]
fn test_component_under_max_params_results() -> Result<()> {
    mod test {
        use super::*;

        pub fn run() -> Result<()> {
            bindgen!({
                inline: r#"
                    package component:test-package;
                    
                    interface env {
                        add: func(a: u32, b: u32) -> u32;
                        multiply: func(x: s64, y: s64, z: s64) -> s64;
                    }
                    
                    world my-world {
                        import env;
                        export calculate: func(a: u32, b: u32, c: s64) -> s64;
                    }
                "#,
                world: "my-world",
            });

            impl component::test_package::env::Host for TestState {
                fn add(&mut self, a: u32, b: u32) -> u32 {
                    a + b
                }

                fn multiply(&mut self, x: i64, y: i64, z: i64) -> i64 {
                    x * y * z
                }
            }

            let component_wat = r#"
                (component
                    (import "component:test-package/env" (instance $env
                        (export "add" (func (param "a" u32) (param "b" u32) (result u32)))
                        (export "multiply" (func (param "x" s64) (param "y" s64) (param "z" s64) (result s64)))
                    ))
                    
                    (core module $m
                        (import "host" "add" (func $add (param i32 i32) (result i32)))
                        (import "host" "multiply" (func $multiply (param i64 i64 i64) (result i64)))
                        
                        (func (export "calculate") (param i32 i32 i64) (result i64)
                            local.get 0
                            local.get 1
                            call $add
                            i64.extend_i32_u
                            local.get 2
                            i64.const 2
                            call $multiply
                        )
                    )
                    
                    (core func $add (canon lower (func $env "add")))
                    (core func $multiply (canon lower (func $env "multiply")))
                    (core instance $m_inst (instantiate $m
                        (with "host" (instance
                            (export "add" (func $add))
                            (export "multiply" (func $multiply))
                        ))
                    ))
                    
                    (func (export "calculate") (param "a" u32) (param "b" u32) (param "c" s64) (result s64)
                        (canon lift (core func $m_inst "calculate"))
                    )
                )
            "#;

            run_component_test(
                component_wat,
                |linker| {
                    MyWorld::add_to_linker::<_, HasSelf<_>>(linker, |state: &mut TestState| state)?;
                    Ok(())
                },
                |store, instance, is_async| {
                    Box::pin(async move {
                        let func = instance
                            .get_typed_func::<(u32, u32, i64), (i64,)>(&mut *store, "calculate")?;

                        let result = if is_async {
                            let (res,) = func.call_async(&mut *store, (10, 20, 3)).await?;
                            func.post_return_async(&mut *store).await?;
                            let (res,) = func
                                .call_async(&mut *store, (res.try_into().unwrap(), 0, 3))
                                .await?;
                            func.post_return_async(&mut *store).await?;
                            res
                        } else {
                            let (res,) = func.call(&mut *store, (10, 20, 3))?;
                            func.post_return(&mut *store)?;
                            let (res,) = func.call(&mut *store, (res as u32, 0, 3))?;
                            func.post_return(&mut *store)?;
                            res
                        };
                        assert_eq!(result, 1080);
                        Ok(())
                    })
                },
            )
        }
    }

    test::run()
}

// Large Record (exceeding MAX_FLAT_PARAMS=16 and MAX_FLAT_RESULTS=1)
#[test]
fn test_component_over_max_params_results() -> Result<()> {
    mod test {
        use super::*;

        pub fn run() -> Result<()> {
            bindgen!({
                inline: r#"
                    package component:test-package;
                    
                    interface env {
                        record big-data {
                            f1: u32, f2: u32, f3: u32, f4: u32,
                            f5: u32, f6: u32, f7: u32, f8: u32,
                            f9: u32, f10: u32, f11: u32, f12: u32,
                            f13: u32, f14: u32, f15: u32, f16: u32,
                            f17: u32, f18: u32, f19: u32, f20: u32,
                        }
                        
                        process-record: func(data: big-data) -> big-data;
                    }
                    
                    world my-world {
                        import env;
                    }
                "#,
                world: "my-world",
            });

            use component::test_package::env::{BigData, Host};

            impl Host for TestState {
                fn process_record(&mut self, mut data: BigData) -> BigData {
                    // Double all the fields
                    data.f1 *= 2;
                    data.f2 *= 2;
                    data.f3 *= 2;
                    data.f4 *= 2;
                    data.f5 *= 2;
                    data.f6 *= 2;
                    data.f7 *= 2;
                    data.f8 *= 2;
                    data.f9 *= 2;
                    data.f10 *= 2;
                    data.f11 *= 2;
                    data.f12 *= 2;
                    data.f13 *= 2;
                    data.f14 *= 2;
                    data.f15 *= 2;
                    data.f16 *= 2;
                    data.f17 *= 2;
                    data.f18 *= 2;
                    data.f19 *= 2;
                    data.f20 *= 2;
                    data
                }
            }

            let component_wat = format!(
                r#"
                (component
                (type (;0;)
                    (instance
                    (type (;0;) (record (field "f1" u32) (field "f2" u32) (field "f3" u32) (field "f4" u32) (field "f5" u32) (field "f6" u32) (field "f7" u32) (field "f8" u32) (field "f9" u32) (field "f10" u32) (field "f11" u32) (field "f12" u32) (field "f13" u32) (field "f14" u32) (field "f15" u32) (field "f16" u32) (field "f17" u32) (field "f18" u32) (field "f19" u32) (field "f20" u32)))
                    (export (;1;) "big-data" (type (eq 0)))
                    (type (;2;) (func (param "data" 1) (result 1)))
                    (export (;0;) "process-record" (func (type 2)))
                    )
                )
                (import "component:test-package/env" (instance (;0;) (type 0)))
                (alias export 0 "big-data" (type (;1;)))
                (alias export 0 "process-record" (func $host))
                (import "big-data" (type (;2;) (eq 1)))
                (core module $m
                    (type (;0;) (func (param i32 i32)))
                    (type (;2;) (func (param i32) (result i32)))
                    (import "env" "process-record" (func $process_record (type 0)))
                    (memory (export "memory") 1)
                    {realloc}
                    (func (export "main") (param $input_ptr i32) (result i32)
                        (local $output_ptr i32)
                        i32.const 116
                        local.set $output_ptr
                        local.get $input_ptr
                        local.get $output_ptr
                        call $process_record
                        local.get $output_ptr
                    )
                    (data (;0;) (i32.const 16) "\01\00\00\00\02\00\00\00\03\00\00\00\04\00\00\00\05\00\00\00\06\00\00\00\07\00\00\00\08\00\00\00\09\00\00\00\0a\00\00\00\0b\00\00\00\0c\00\00\00\0d\00\00\00\0e\00\00\00\0f\00\00\00\10\00\00\00\11\00\00\00\12\00\00\00\13\00\00\00\14\00\00\00")
                )
                {shims}
                {instantiation}
                )
            "#,
                realloc = cabi_realloc_wat(),
                shims = shims_wat("i32 i32"),
                instantiation =
                    instantiation_wat("process-record", "(param \"data\" 2) (result 2)")
            );

            run_component_test(
                &component_wat,
                |linker| {
                    component::test_package::env::add_to_linker::<_, HasSelf<_>>(
                        linker,
                        |state: &mut TestState| state,
                    )?;
                    Ok(())
                },
                |store, instance, is_async| {
                    Box::pin(async move {
                        // Call the main export with test data
                        let main_func = instance
                            .get_typed_func::<(BigData,), (BigData,)>(&mut *store, "run")?;

                        let test_data = BigData {
                            f1: 1,
                            f2: 2,
                            f3: 3,
                            f4: 4,
                            f5: 5,
                            f6: 6,
                            f7: 7,
                            f8: 8,
                            f9: 9,
                            f10: 10,
                            f11: 11,
                            f12: 12,
                            f13: 13,
                            f14: 14,
                            f15: 15,
                            f16: 16,
                            f17: 17,
                            f18: 18,
                            f19: 19,
                            f20: 20,
                        };

                        let (result,) = if is_async {
                            let res = main_func.call_async(&mut *store, (test_data,)).await?;
                            main_func.post_return_async(&mut *store).await?;
                            res
                        } else {
                            let res = main_func.call(&mut *store, (test_data,))?;
                            main_func.post_return(&mut *store)?;
                            res
                        };

                        // All fields should be doubled
                        assert_eq!(result.f1, 2);
                        assert_eq!(result.f10, 20);
                        assert_eq!(result.f20, 40);

                        Ok(())
                    })
                },
            )
        }
    }

    test::run()
}

#[test]
fn test_component_tuple() -> Result<()> {
    mod test {
        use super::*;

        pub fn run() -> Result<()> {
            bindgen!({
                inline: r#"
                    package component:test;
                    
                    interface env {
                        swap-tuple: func(val: tuple<u32, u32>) -> tuple<u32, u32>;
                    }
                    
                    world my-world {
                        import env;
                    }
                "#,
                world: "my-world",
            });

            use component::test::env::Host;

            impl Host for TestState {
                fn swap_tuple(&mut self, val: (u32, u32)) -> (u32, u32) {
                    (val.1, val.0)
                }
            }

            let component_wat = format!(
                r#"
                (component
                  (import "component:test/env" (instance $env
                    (export "swap-tuple" (func (param "val" (tuple u32 u32)) (result (tuple u32 u32))))
                  ))
                  (alias export $env "swap-tuple" (func $host))
                  (core module $m
                    (import "env" "swap" (func $swap (param i32 i32 i32)))
                    (memory (export "memory") 1)
                    {realloc}
                    (func (export "main") (result i32)
                      (local $retptr i32)
                      i32.const 100
                      local.set $retptr
                      
                      i32.const 10
                      i32.const 20
                      local.get $retptr
                      call $swap
                      
                      local.get $retptr
                    )
                  )
                  {shims}
                  {instantiation}
                )
            "#,
                realloc = cabi_realloc_wat(),
                shims = shims_wat("i32 i32 i32"),
                instantiation = instantiation_wat("swap", "(result (tuple u32 u32))")
            );

            run_component_test(
                &component_wat,
                |linker| {
                    component::test::env::add_to_linker::<_, HasSelf<_>>(
                        linker,
                        |state: &mut TestState| state,
                    )?;
                    Ok(())
                },
                |store, instance, is_async| {
                    Box::pin(async move {
                        let func =
                            instance.get_typed_func::<(), ((u32, u32),)>(&mut *store, "run")?;
                        let (result,) = if is_async {
                            let res = func.call_async(&mut *store, ()).await?;
                            func.post_return_async(&mut *store).await?;
                            res
                        } else {
                            let res = func.call(&mut *store, ())?;
                            func.post_return(&mut *store)?;
                            res
                        };
                        assert_eq!(result, (20, 10));
                        Ok(())
                    })
                },
            )
        }
    }

    test::run()
}

#[test]
fn test_component_string() -> Result<()> {
    mod test {
        use super::*;

        pub fn run() -> Result<()> {
            bindgen!({
                inline: r#"
                    package component:test;
                    
                    interface env {
                        reverse-string: func(s: string) -> string;
                    }
                    
                    world my-world {
                        import env;
                    }
                "#,
                world: "my-world",
            });

            use component::test::env::Host;

            impl Host for TestState {
                fn reverse_string(&mut self, s: String) -> String {
                    s.chars().rev().collect()
                }
            }

            let component_wat = format!(
                r#"
                (component
                  (import "component:test/env" (instance $env
                    (export "reverse-string" (func (param "s" string) (result string)))
                  ))
                  (alias export $env "reverse-string" (func $host))
                  (core module $m
                    (import "env" "reverse" (func $reverse (param i32 i32 i32)))
                    (memory (export "memory") 1)
                    {realloc}
                    (func (export "main") (result i32)
                      (local $retptr i32)
                      i32.const 100
                      local.set $retptr
                      
                      ;; Call reverse("hello")
                      ;; "hello" is at offset 16, len 5
                      i32.const 16
                      i32.const 5
                      local.get $retptr
                      call $reverse
                      
                      ;; Return retptr which points to (ptr, len)
                      local.get $retptr
                    )
                    (data (i32.const 16) "hello")
                  )
                  {shims}
                  {instantiation}
                )
            "#,
                realloc = cabi_realloc_wat(),
                shims = shims_wat("i32 i32 i32"),
                instantiation = instantiation_wat("reverse", "(result string)")
            );

            run_component_test(
                &component_wat,
                |linker| {
                    component::test::env::add_to_linker::<_, HasSelf<_>>(
                        linker,
                        |state: &mut TestState| state,
                    )?;
                    Ok(())
                },
                |store, instance, is_async| {
                    Box::pin(async move {
                        let func = instance.get_typed_func::<(), (String,)>(&mut *store, "run")?;
                        let (result,) = if is_async {
                            let res = func.call_async(&mut *store, ()).await?;
                            func.post_return_async(&mut *store).await?;
                            res
                        } else {
                            let res = func.call(&mut *store, ())?;
                            func.post_return(&mut *store)?;
                            res
                        };
                        assert_eq!(result, "olleh");
                        Ok(())
                    })
                },
            )
        }
    }

    test::run()
}

#[test]
fn test_component_variant() -> Result<()> {
    mod test {
        use super::*;

        pub fn run() -> Result<()> {
            bindgen!({
                inline: r#"
                    package component:test;
                    
                    interface env {
                        variant shape {
                            circle(f32),
                            rectangle(tuple<f32, f32>)
                        }
                        transform: func(s: shape) -> shape;
                    }
                    
                    world my-world {
                        import env;
                    }
                "#,
                world: "my-world",
            });

            use component::test::env::{Host, Shape};

            impl Host for TestState {
                fn transform(&mut self, s: Shape) -> Shape {
                    match s {
                        Shape::Circle(r) => Shape::Circle(r * 2.0),
                        Shape::Rectangle((w, h)) => Shape::Rectangle((h, w)),
                    }
                }
            }

            let component_wat = format!(
                r#"
                (component
                  (type (;0;)
                    (instance
                      (type (;0;) (tuple f32 f32))
                      (type (;1;) (variant (case "circle" f32) (case "rectangle" 0)))
                      (export (;2;) "shape" (type (eq 1)))
                      (type (;3;) (func (param "s" 2) (result 2)))
                      (export (;0;) "transform" (func (type 3)))
                    )
                  )
                  (import "component:test/env" (instance (;0;) (type 0)))
                  (alias export 0 "shape" (type (;1;)))
                  (alias export 0 "transform" (func $host))
                  (core module $m
                    (import "env" "transform" (func $transform (param i32 f32 f32 i32)))
                    (memory (export "memory") 1)
                    {realloc}
                    (func (export "main") (result i32)
                      (local $retptr i32)
                      i32.const 100
                      local.set $retptr
                      
                      i32.const 0       ;; discriminant = Circle
                      f32.const 10.0    ;; payload 1
                      f32.const 0.0     ;; payload 2
                      local.get $retptr
                      call $transform
                      
                      local.get $retptr
                    )
                  )
                  {shims}
                  {instantiation}
                )
            "#,
                realloc = cabi_realloc_wat(),
                shims = shims_wat("i32 f32 f32 i32"),
                instantiation = instantiation_wat("transform", "(result 1)")
            );

            run_component_test(
                &component_wat,
                |linker| {
                    component::test::env::add_to_linker::<_, HasSelf<_>>(
                        linker,
                        |state: &mut TestState| state,
                    )?;
                    Ok(())
                },
                |store, instance, is_async| {
                    Box::pin(async move {
                        let func = instance.get_typed_func::<(), (Shape,)>(&mut *store, "run")?;
                        let (result,) = if is_async {
                            let res = func.call_async(&mut *store, ()).await?;
                            func.post_return_async(&mut *store).await?;
                            res
                        } else {
                            let res = func.call(&mut *store, ())?;
                            func.post_return(&mut *store)?;
                            res
                        };

                        match result {
                            Shape::Circle(r) => assert_eq!(r, 20.0),
                            _ => panic!("Expected Circle"),
                        }
                        Ok(())
                    })
                },
            )
        }
    }

    test::run()
}

#[test]
fn test_component_result() -> Result<()> {
    mod test {
        use super::*;

        pub fn run() -> Result<()> {
            bindgen!({
                inline: r#"
                    package component:test;
                    
                    interface env {
                        convert: func(r: result<u32, string>) -> result<string, u32>;
                    }
                    
                    world my-world {
                        import env;
                    }
                "#,
                world: "my-world",
            });

            use component::test::env::Host;

            impl Host for TestState {
                fn convert(&mut self, r: Result<u32, String>) -> Result<String, u32> {
                    match r {
                        Ok(val) => Ok(val.to_string()),
                        Err(msg) => Err(msg.len() as u32),
                    }
                }
            }

            let component_wat = format!(
                r#"
                (component
                  (type (;0;)
                    (instance
                      (type (;0;) (result u32 (error string)))
                      (type (;1;) (result string (error u32)))
                      (type (;2;) (func (param "r" 0) (result 1)))
                      (export (;0;) "convert" (func (type 2)))
                    )
                  )
                  (import "component:test/env" (instance (;0;) (type 0)))
                  (alias export 0 "convert" (func $host))
                  (core module $m
                    (import "env" "convert" (func $convert (param i32 i32 i32 i32)))
                    (memory (export "memory") 1)
                    {realloc}
                    (func (export "main") (result i32)
                      (local $retptr i32)
                      i32.const 100
                      local.set $retptr
                      i32.const 0
                      i32.const 42
                      i32.const 0
                      local.get $retptr
                      call $convert
                      local.get $retptr
                      i32.load
                      i32.const 0
                      i32.ne
                      if
                        unreachable
                      end
                      local.get $retptr
                      i32.load offset=8
                      i32.const 2
                      i32.ne
                      if
                        unreachable
                      end
                      i32.const 1
                      i32.const 16
                      i32.const 5
                      local.get $retptr
                      call $convert
                      local.get $retptr
                      i32.load
                      i32.const 1
                      i32.ne
                      if
                        unreachable
                      end
                      local.get $retptr
                      i32.load offset=4
                      i32.const 5
                      i32.ne
                      if
                        unreachable
                      end
                      i32.const 1
                    )
                    (data (i32.const 16) "hello")
                  )
                  {shims}
                  {instantiation}
                )
            "#,
                realloc = cabi_realloc_wat(),
                shims = shims_wat("i32 i32 i32 i32"),
                instantiation = instantiation_wat("convert", "(result u32)")
            );

            run_component_test(
                &component_wat,
                |linker| {
                    component::test::env::add_to_linker::<_, HasSelf<_>>(
                        linker,
                        |state: &mut TestState| state,
                    )?;
                    Ok(())
                },
                |store, instance, is_async| {
                    Box::pin(async move {
                        let func = instance.get_typed_func::<(), (u32,)>(&mut *store, "run")?;
                        let (result,) = if is_async {
                            let res = func.call_async(&mut *store, ()).await?;
                            func.post_return_async(&mut *store).await?;
                            res
                        } else {
                            let res = func.call(&mut *store, ())?;
                            func.post_return(&mut *store)?;
                            res
                        };
                        assert_eq!(result, 1);
                        Ok(())
                    })
                },
            )
        }
    }

    test::run()
}

#[test]
fn test_component_list() -> Result<()> {
    mod test {
        use super::*;

        pub fn run() -> Result<()> {
            bindgen!({
                inline: r#"
                    package component:test;
                    
                    interface env {
                        reverse-list: func(l: list<u32>) -> list<u32>;
                    }
                    
                    world my-world {
                        import env;
                    }
                "#,
                world: "my-world",
            });

            use component::test::env::Host;

            impl Host for TestState {
                fn reverse_list(&mut self, l: Vec<u32>) -> Vec<u32> {
                    l.into_iter().rev().collect()
                }
            }

            let component_wat = format!(
                r#"
                (component
                  (type (;0;)
                    (instance
                      (type (;0;) (list u32))
                      (type (;1;) (func (param "l" 0) (result 0)))
                      (export (;0;) "reverse-list" (func (type 1)))
                    )
                  )
                  (import "component:test/env" (instance (;0;) (type 0)))
                  (alias export 0 "reverse-list" (func $host))
                  (core module $m
                    (import "env" "reverse" (func $reverse (param i32 i32 i32)))
                    (memory (export "memory") 1)
                    {realloc}
                    (func (export "main") (result i32)
                      (local $retptr i32)
                      i32.const 100
                      local.set $retptr
                      i32.const 16
                      i32.const 3
                      local.get $retptr
                      call $reverse
                      local.get $retptr
                      i32.load offset=4
                      i32.const 3
                      i32.ne
                      if
                        unreachable
                      end
                      local.get $retptr
                      i32.load
                      local.set $retptr
                      local.get $retptr
                      i32.load
                      i32.const 3
                      i32.ne
                      if
                        unreachable
                      end
                      local.get $retptr
                      i32.load offset=4
                      i32.const 2
                      i32.ne
                      if
                        unreachable
                      end
                      local.get $retptr
                      i32.load offset=8
                      i32.const 1
                      i32.ne
                      if
                        unreachable
                      end
                      i32.const 1
                    )
                    (data (i32.const 16) "\01\00\00\00\02\00\00\00\03\00\00\00")
                  )
                  {shims}
                  {instantiation}
                )
            "#,
                realloc = cabi_realloc_wat(),
                shims = shims_wat("i32 i32 i32"),
                instantiation = instantiation_wat("reverse", "(result u32)")
            );

            run_component_test(
                &component_wat,
                |linker| {
                    component::test::env::add_to_linker::<_, HasSelf<_>>(
                        linker,
                        |state: &mut TestState| state,
                    )?;
                    Ok(())
                },
                |store, instance, is_async| {
                    Box::pin(async move {
                        let func = instance.get_typed_func::<(), (u32,)>(&mut *store, "run")?;
                        let (result,) = if is_async {
                            let res = func.call_async(&mut *store, ()).await?;
                            func.post_return_async(&mut *store).await?;
                            res
                        } else {
                            let res = func.call(&mut *store, ())?;
                            func.post_return(&mut *store)?;
                            res
                        };
                        assert_eq!(result, 1);
                        Ok(())
                    })
                },
            )
        }
    }

    test::run()
}

#[test]
fn test_component_option() -> Result<()> {
    mod test {
        use super::*;

        pub fn run() -> Result<()> {
            bindgen!({
                inline: r#"
                    package component:test;
                    
                    interface env {
                        inc-option: func(o: option<u32>) -> option<u32>;
                    }
                    
                    world my-world {
                        import env;
                    }
                "#,
                world: "my-world",
            });

            use component::test::env::Host;

            impl Host for TestState {
                fn inc_option(&mut self, o: Option<u32>) -> Option<u32> {
                    o.map(|x| x + 1)
                }
            }

            let component_wat = format!(
                r#"
                (component
                  (type (;0;)
                    (instance
                      (type (;0;) (option u32))
                      (type (;1;) (func (param "o" 0) (result 0)))
                      (export (;0;) "inc-option" (func (type 1)))
                    )
                  )
                  (import "component:test/env" (instance (;0;) (type 0)))
                  (alias export 0 "inc-option" (func $host))
                  (core module $m
                    (import "env" "inc" (func $inc (param i32 i32 i32)))
                    (memory (export "memory") 1)
                    {realloc}
                    (func (export "main") (result i32)
                      (local $retptr i32)
                      i32.const 100
                      local.set $retptr
                      i32.const 1
                      i32.const 10
                      local.get $retptr
                      call $inc
                      local.get $retptr
                      i32.load
                      i32.const 1
                      i32.ne
                      if
                        unreachable
                      end
                      local.get $retptr
                      i32.load offset=4
                      i32.const 11
                      i32.ne
                      if
                        unreachable
                      end
                      i32.const 0
                      i32.const 0
                      local.get $retptr
                      call $inc
                      local.get $retptr
                      i32.load
                      if
                        unreachable
                      end
                      i32.const 1
                    )
                  )
                  {shims}
                  {instantiation}
                )
            "#,
                realloc = cabi_realloc_wat(),
                shims = shims_wat("i32 i32 i32"),
                instantiation = instantiation_wat("inc", "(result u32)")
            );

            run_component_test(
                &component_wat,
                |linker| {
                    component::test::env::add_to_linker::<_, HasSelf<_>>(
                        linker,
                        |state: &mut TestState| state,
                    )?;
                    Ok(())
                },
                |store, instance, is_async| {
                    Box::pin(async move {
                        let func = instance.get_typed_func::<(), (u32,)>(&mut *store, "run")?;
                        let (result,) = if is_async {
                            let res = func.call_async(&mut *store, ()).await?;
                            func.post_return_async(&mut *store).await?;
                            res
                        } else {
                            let res = func.call(&mut *store, ())?;
                            func.post_return(&mut *store)?;
                            res
                        };
                        assert_eq!(result, 1);
                        Ok(())
                    })
                },
            )
        }
    }

    test::run()
}

#[test]
fn test_component_builtins() -> Result<()> {
    run_component_test(
        r#"
            (component
                (type $r (resource (rep i32)))
                (core func $rep (canon resource.rep $r))
                (core func $new (canon resource.new $r))
                (core func $drop (canon resource.drop $r))

                (import "host-double" (func $host_double (param "v" s32) (result s32)))
                (core func $host_double_core (canon lower (func $host_double)))

                (core module $m
                    (import "" "rep" (func $rep (param i32) (result i32)))
                    (import "" "new" (func $new (param i32) (result i32)))
                    (import "" "drop" (func $drop (param i32)))
                    (import "" "host_double" (func $host_double (param i32) (result i32)))

                    (func $start
                        (local $r1 i32)
                        (local $r2 i32)

                        ;; resources assigned sequentially
                        (local.set $r1 (call $new (i32.const 100)))
                        (if (i32.ne (local.get $r1) (i32.const 1)) (then (unreachable)))

                        (local.set $r2 (call $new (i32.const 200)))
                        (if (i32.ne (local.get $r2) (i32.const 2)) (then (unreachable)))

                        ;; representations all look good
                        (if (i32.ne (call $rep (local.get $r1)) (i32.const 100)) (then (unreachable)))
                        (if (i32.ne (call $rep (local.get $r2)) (i32.const 200)) (then (unreachable)))

                        ;; reallocate r2
                        (call $drop (local.get $r2))
                        (local.set $r2 (call $new (i32.const 400)))

                        ;; should have reused index 1
                        (if (i32.ne (local.get $r2) (i32.const 2)) (then (unreachable)))

                        ;; representations all look good
                        (if (i32.ne (call $rep (local.get $r1)) (i32.const 100)) (then (unreachable)))
                        (if (i32.ne (call $rep (local.get $r2)) (i32.const 400)) (then (unreachable)))

                        ;; deallocate everything
                        (call $drop (local.get $r1))
                        (call $drop (local.get $r2))
                    )
                    (start $start)

                    (func $run (result i32)
                        (local $r1 i32)
                        (local $val i32)

                        ;; Create a new resource
                        (local.set $r1 (call $new (i32.const 500)))
                        
                        ;; Get its representation
                        (local.set $val (call $rep (local.get $r1)))
                        
                        ;; Double it using host function
                        (local.set $val (call $host_double (local.get $val)))
                        
                        ;; Drop the resource
                        (call $drop (local.get $r1))
                        
                        local.get $val
                    )
                    (export "run" (func $run))
                )
                (core instance $i (instantiate $m
                    (with "" (instance
                        (export "rep" (func $rep))
                        (export "new" (func $new))
                        (export "drop" (func $drop))
                        (export "host_double" (func $host_double_core))
                    ))
                ))
                (func $run_comp (result s32) (canon lift (core func $i "run")))
                (export "run" (func $run_comp))
            )
        "#,
        |linker| {
            linker
                .root()
                .func_wrap("host-double", |_, (v,): (i32,)| Ok((v * 2,)))?;
            Ok(())
        },
        |store, instance, is_async| {
            Box::pin(async move {
                let run = instance.get_typed_func::<(), (i32,)>(&mut *store, "run")?;
                let (result,) = if is_async {
                    run.call_async(&mut *store, ()).await?
                } else {
                    run.call(&mut *store, ())?
                };
                assert_eq!(result, 1000);
                Ok(())
            })
        },
    )
}

fn cabi_realloc_wat() -> String {
    r#"
    (global $bump (mut i32) (i32.const 256))
    (export "cabi_realloc" (func $realloc))
    (func $realloc (param $old_ptr i32) (param $old_size i32) (param $align i32) (param $new_size i32) (result i32)
      (local $result i32)
      global.get $bump
      local.get $align
      i32.const 1
      i32.sub
      i32.add
      local.get $align
      i32.const 1
      i32.sub
      i32.const -1
      i32.xor
      i32.and
      local.set $result
      local.get $result
      local.get $new_size
      i32.add
      global.set $bump
      local.get $result
    )
    "#.to_string()
}

fn shims_wat(params: &str) -> String {
    let count = params.split_whitespace().count();
    let locals_get = (0..count)
        .map(|i| format!("local.get {i}"))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        r#"
      (core module $shim1
        (table (export "$imports") 1 funcref)
        (func (export "0") (param {params})
          {locals_get}
          i32.const 0
          call_indirect (param {params})
        )
      )
      (core module $shim2
        (import "" "0" (func (param {params})))
        (import "" "$imports" (table 1 funcref))
        (elem (i32.const 0) func 0)
      )
    "#
    )
}

fn instantiation_wat(core_name: &str, lift_sig: &str) -> String {
    format!(
        r#"
      (core instance $s1 (instantiate $shim1))
      (alias core export $s1 "0" (core func $indirect))
      (core instance $env_inst (export "{core_name}" (func $indirect)))
      (core instance $inst (instantiate $m (with "env" (instance $env_inst))))
      (alias core export $inst "memory" (core memory $mem))
      (alias core export $inst "cabi_realloc" (core func $realloc))
      (alias core export $s1 "$imports" (core table $tbl))
      (core func $lowered (canon lower (func $host) (memory $mem) (realloc $realloc)))
      (core instance $tbl_inst (export "$imports" (table $tbl)) (export "0" (func $lowered)))
      (core instance (instantiate $shim2 (with "" (instance $tbl_inst))))
      (alias core export $inst "main" (core func $run))
      (func (export "run") {lift_sig} (canon lift (core func $run) (memory $mem) (realloc $realloc)))
    "#
    )
}

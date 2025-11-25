use crate::rr::Validate;
use crate::rr::{RREvent, ReplayError, core_events};
use crate::store::InstanceId;
use crate::{AsContextMut, Engine, Module, ReplayReader, ReplaySettings, Store, prelude::*};
#[cfg(rr_component)]
use crate::{
    ValRaw, component, component::Component, component::ComponentInstanceId, rr::component_events,
    rr::component_hooks,
};
use alloc::{collections::BTreeMap, sync::Arc};
#[cfg(not(rr_component))]
use anyhow::bail;
#[cfg(rr_component)]
use core::mem::MaybeUninit;
#[cfg(rr_component)]
use wasmtime_environ::component::{MAX_FLAT_PARAMS, MAX_FLAT_RESULTS};
use wasmtime_environ::{EntityIndex, WasmChecksum};

/// The environment necessary to produce a [`ReplayInstance`]
#[derive(Clone)]
pub struct ReplayEnvironment {
    engine: Engine,
    modules: BTreeMap<WasmChecksum, Module>,
    #[cfg(rr_component)]
    components: BTreeMap<WasmChecksum, Component>,
    settings: ReplaySettings,
}

impl ReplayEnvironment {
    /// Construct a new [`ReplayEnvironment`] from scratch
    pub fn new(engine: &Engine, settings: ReplaySettings) -> Self {
        Self {
            engine: engine.clone(),
            modules: BTreeMap::new(),
            #[cfg(rr_component)]
            components: BTreeMap::new(),
            settings,
        }
    }

    /// Add a [`Module`] to the replay environment
    pub fn add_module(&mut self, module: Module) -> &mut Self {
        self.modules.insert(*module.checksum(), module);
        self
    }

    /// Add a [`Component`] to the replay environment
    #[cfg(rr_component)]
    pub fn add_component(&mut self, component: Component) -> &mut Self {
        self.components.insert(*component.checksum(), component);
        self
    }

    /// Instantiate a new [`ReplayInstance`] using a [`ReplayReader`] in context of this environment
    pub fn instantiate(&self, reader: impl ReplayReader + 'static) -> Result<ReplayInstance<()>> {
        let store = Store::new(&self.engine, ());
        ReplayInstance::<()>::from_environment_and_store(self.clone(), store, reader)
    }

    /// Like [`Self::instantiate`] but allows providing a custom [`Store`] generator
    pub fn instantiate_with_store<T>(
        &self,
        store_gen: impl FnOnce() -> Store<T>,
        reader: impl ReplayReader + 'static,
    ) -> Result<ReplayInstance<T>> {
        ReplayInstance::from_environment_and_store(self.clone(), store_gen(), reader)
    }
}

/// A [`ReplayInstance`] is an object providing a opaquely managed, replayable [`Store`]
///
/// Debugger capabilities in the future will interact with this object for
/// inserting breakpoints, snapshotting, and restoring state
///
/// # Example
///
/// ```
/// use wasmtime::*;
/// use wasmtime::component::Component;
/// # use std::io::Cursor;
/// # use wasmtime::component;
/// # use core::any::Any;
/// # fn main() -> Result<()> {
/// let component_str: &str = r#"
///     (component
///         (core module $m
///             (func (export "main") (result i32)
///                 i32.const 42
///             )
///         )
///         (core instance $i (instantiate $m))
///
///         (func (export "main") (result u32)
///             (canon lift (core func $i "main"))
///         )
///     )
/// "#;
///
/// # let record_settings = RecordSettings::default();
/// # let mut config = Config::new();
/// # config.rr(RRConfig::Recording);
///
/// # let engine = Engine::new(&config)?;
/// # let component = Component::new(&engine, component_str)?;
/// # let mut linker = component::Linker::new(&engine);
///
/// # let writer: Cursor<Vec<u8>> = Cursor::new(Vec::new());
/// # let mut store = Store::new(&engine, ());
/// # store.record(writer, record_settings)?;
///
/// # let instance = linker.instantiate(&mut store, &component)?;
/// # let func = instance.get_typed_func::<(), (u32,)>(&mut store, "main")?;
/// # let _ = func.call(&mut store, ());
///
/// # let trace_box = store.into_record_writer()?;
/// # let any_box: Box<dyn Any> = trace_box;
/// # let mut trace_reader = any_box.downcast::<Cursor<Vec<u8>>>().unwrap();
/// # trace_reader.set_position(0);
///
/// // let trace_reader = ... (obtain a ReplayReader over the recorded trace from somewhere)
///
/// let mut config = Config::new();
/// config.rr(RRConfig::Replaying);
/// let engine = Engine::new(&config)?;
/// let mut renv = ReplayEnvironment::new(&engine, ReplaySettings::default());
/// renv.add_component(Component::new(&engine, component_str)?);
/// // You can add more components, or modules with renv.add_module(module);
/// // ....
/// let mut instance = renv.instantiate(trace_reader)?;
/// instance.run_to_completion()?;
/// # Ok(())
/// # }
/// ```
pub struct ReplayInstance<T: 'static> {
    env: Arc<ReplayEnvironment>,
    store: Store<T>,
    #[cfg(rr_component)]
    component_linker: component::Linker<T>,
    module_linker: crate::Linker<T>,
    module_instances: BTreeMap<InstanceId, crate::Instance>,
    #[cfg(rr_component)]
    component_instances: BTreeMap<ComponentInstanceId, component::Instance>,
}

impl<T: 'static> ReplayInstance<T> {
    fn from_environment_and_store(
        env: ReplayEnvironment,
        mut store: Store<T>,
        reader: impl ReplayReader + 'static,
    ) -> Result<Self> {
        let env = Arc::new(env);
        store.init_replaying(reader, env.settings.clone())?;
        let mut module_linker = crate::Linker::<T>::new(&env.engine);
        // Replays shouldn't use any imports, so stub them all out as traps
        for module in env.modules.values() {
            module_linker.define_unknown_imports_as_traps(module)?;
        }
        #[cfg(rr_component)]
        let mut component_linker = component::Linker::<T>::new(&env.engine);
        #[cfg(rr_component)]
        for component in env.components.values() {
            component_linker.define_unknown_imports_as_traps(component)?;
        }
        Ok(Self {
            env,
            store,
            #[cfg(rr_component)]
            component_linker,
            module_linker,
            module_instances: BTreeMap::new(),
            #[cfg(rr_component)]
            component_instances: BTreeMap::new(),
        })
    }

    /// Obtain a reference to the internal [`Store`]
    pub fn store(&self) -> &Store<T> {
        &self.store
    }

    /// Consume the [`ReplayInstance`] and extract the internal [`Store`]
    pub fn extract_store(self) -> Store<T> {
        self.store
    }

    /// Run a single top-level event from the instance
    ///
    /// "Top-level" events are those explicitly invoked events, namely:
    /// * Instantiation events (component/module)
    /// * Wasm function begin events (`ComponentWasmFuncBegin` for components and `CoreWasmFuncEntry` for core)
    ///
    /// All other events are transparently dispatched under the context of these top-level events
    fn run_single_top_level_event(&mut self, rr_event: RREvent) -> Result<()> {
        match rr_event {
            RREvent::ComponentInstantiation(event) => {
                #[cfg(rr_component)]
                {
                    // Find matching component from environment to instantiate
                    let component = self
                        .env
                        .components
                        .get(&event.component)
                        .ok_or(ReplayError::MissingComponent(event.component))?;

                    let instance = self
                        .component_linker
                        .instantiate(self.store.as_context_mut(), component)?;
                    // Validate the instantiation event
                    event.validate(&component_events::InstantiationEvent {
                        component: *component.checksum(),
                        instance: instance.id().instance(),
                    })?;

                    self.component_instances
                        .insert(instance.id().instance(), instance);
                }
                #[cfg(not(rr_component))]
                {
                    let _ = event;
                    bail!(
                        "Cannot parse ComponentInstantation replay event without rr and component-model feature enabled"
                    );
                }
            }
            RREvent::ComponentWasmFuncBegin(event) => {
                #[cfg(rr_component)]
                {
                    // Grab the correct component instance
                    let key = event.instance;
                    let instance = self
                        .component_instances
                        .get_mut(&key)
                        .ok_or(ReplayError::MissingComponentInstance(key.as_u32()))?;

                    // Replay lowering steps and obtain raw value arguments to raw function call
                    let func = component::Func::from_lifted_func(*instance, event.func_idx);
                    let store = self.store.as_context_mut();

                    // Call the function
                    //
                    // This is almost a mirror of the usage in [`component::Func::call_impl`]
                    let mut results_storage = [component::Val::U64(0); MAX_FLAT_RESULTS];
                    let mut num_results = 0;
                    let results = &mut results_storage;
                    let _return = unsafe {
                        func.call_raw(
                                store,
                                |cx, _, dst: &mut MaybeUninit<[MaybeUninit<ValRaw>; MAX_FLAT_PARAMS]>| {
                                    // For lowering, use replay instead of actual lowering
                                    let dst: &mut [MaybeUninit<ValRaw>] = dst.assume_init_mut();
                                    cx.replay_lowering(Some(dst), component_hooks::ReplayLoweringPhase::WasmFuncEntry)
                                },
                                |cx, results_ty, src: &[ValRaw; MAX_FLAT_RESULTS]| {
                                    // Lifting can proceed exactly as normal
                                    for (result, slot) in
                                        component::Func::lift_results(cx, results_ty, src, MAX_FLAT_RESULTS)?.zip(results)
                                    {
                                        *slot = result?;
                                        num_results += 1;
                                    }
                                    Ok(())
                                },
                            )?
                    };

                    log::info!(
                        "Returned {:?} for calling {:?}",
                        &results_storage[..num_results],
                        func
                    );
                }
                #[cfg(not(rr_component))]
                {
                    let _ = event;
                    bail!(
                        "Cannot parse ComponentWasmFuncBegin replay event without rr and component-model feature enabled"
                    );
                }
            }
            RREvent::ComponentPostReturn(event) => {
                #[cfg(rr_component)]
                {
                    // Grab the correct component instance
                    let key = event.instance;
                    let instance = self
                        .component_instances
                        .get_mut(&key)
                        .ok_or(ReplayError::MissingComponentInstance(key.as_u32()))?;

                    let func = component::Func::from_lifted_func(*instance, event.func_idx);
                    let mut store = self.store.as_context_mut();

                    func.post_return(&mut store)?;
                }
                #[cfg(not(rr_component))]
                {
                    let _ = event;
                    bail!(
                        "Cannot parse ComponentPostReturn replay event without rr and component-model feature enabled"
                    );
                }
            }
            RREvent::CoreWasmInstantiation(event) => {
                // Find matching module from environment to instantiate
                let module = self
                    .env
                    .modules
                    .get(&event.module)
                    .ok_or(ReplayError::MissingModule(event.module))?;

                let instance = self
                    .module_linker
                    .instantiate(self.store.as_context_mut(), module)?;

                // Validate the instantiation event
                event.validate(&core_events::InstantiationEvent {
                    module: *module.checksum(),
                    instance: instance.id(),
                })?;

                self.module_instances.insert(instance.id(), instance);
            }
            RREvent::CoreWasmFuncEntry(event) => {
                // Grab the correct module instance
                let key = event.origin.instance;
                let instance = self
                    .module_instances
                    .get_mut(&key)
                    .ok_or(ReplayError::MissingModuleInstance(key.as_u32()))?;

                let entity = EntityIndex::from(event.origin.index);
                let mut store = self.store.as_context_mut();
                let func = instance
                    ._get_export(store.0, entity)
                    .into_func()
                    .ok_or(ReplayError::InvalidCoreFuncIndex(entity))?;

                let params_ty = func.ty(&store).params().collect::<Vec<_>>();

                // Obtain the argument values for function call
                let mut results = vec![crate::Val::I64(0); func.ty(&store).results().len()];
                let params = event.args.to_val_vec(&mut store, params_ty);

                // Call the function
                //
                // This is almost a mirror of the usage in [`crate::Func::call_impl`]
                func.call_impl_check_args(&mut store, &params, &mut results)?;
                unsafe {
                    func.call_impl_do_call(&mut store, params.as_slice(), results.as_mut_slice())?;
                }
            }

            _ => {
                log::error!("Unexpected top-level RR event: {:?}", rr_event);
                Err(ReplayError::IncorrectEventVariant)?
            }
        }
        Ok(())
    }

    /// Exactly like [`Self::run_single_top_level_event`] but uses async stores and calls
    #[cfg(feature = "async")]
    async fn run_single_top_level_event_async(&mut self, rr_event: RREvent) -> Result<()>
    where
        T: Send,
    {
        match rr_event {
            RREvent::ComponentInstantiation(event) => {
                #[cfg(rr_component)]
                {
                    // Find matching component from environment to instantiate
                    let component = self
                        .env
                        .components
                        .get(&event.component)
                        .ok_or(ReplayError::MissingComponent(event.component))?;

                    let instance = self
                        .component_linker
                        .instantiate_async(self.store.as_context_mut(), component)
                        .await?;
                    // Validate the instantiation event
                    event.validate(&component_events::InstantiationEvent {
                        component: *component.checksum(),
                        instance: instance.id().instance(),
                    })?;

                    self.component_instances
                        .insert(instance.id().instance(), instance);
                }
                #[cfg(not(rr_component))]
                {
                    let _ = event;
                    bail!(
                        "Cannot parse ComponentInstantation replay event without rr and component-model feature enabled"
                    );
                }
            }
            RREvent::ComponentWasmFuncBegin(event) => {
                #[cfg(rr_component)]
                {
                    // Grab the correct component instance
                    let key = event.instance;
                    let instance = self
                        .component_instances
                        .get_mut(&key)
                        .ok_or(ReplayError::MissingComponentInstance(key.as_u32()))?;

                    // Replay lowering steps and obtain raw value arguments to raw function call
                    let func = component::Func::from_lifted_func(*instance, event.func_idx);
                    let mut store = self.store.as_context_mut();

                    // Call the function
                    //
                    // This is almost a mirror of the usage in [`component::Func::call_impl`]
                    let mut results_storage = [component::Val::U64(0); MAX_FLAT_RESULTS];
                    let mut num_results = 0;
                    let results = &mut results_storage;
                    let _return = store
                        .on_fiber(|store| unsafe {
                            func.call_raw(
                                store.as_context_mut(),
                                    |cx,
                                     _,
                                     dst: &mut MaybeUninit<
                                        [MaybeUninit<ValRaw>; MAX_FLAT_PARAMS],
                                    >| {
                                        // For lowering, use replay instead of actual lowering
                                        let dst: &mut [MaybeUninit<ValRaw>] = dst.assume_init_mut();
                                        cx.replay_lowering(
                                            Some(dst),
                                            component_hooks::ReplayLoweringPhase::WasmFuncEntry,
                                        )
                                    },
                                    |cx, results_ty, src: &[ValRaw; MAX_FLAT_RESULTS]| {
                                        // Lifting can proceed exactly as normal
                                        for (result, slot) in component::Func::lift_results(
                                            cx,
                                            results_ty,
                                            src,
                                            MAX_FLAT_RESULTS,
                                        )?
                                        .zip(results)
                                        {
                                            *slot = result?;
                                            num_results += 1;
                                        }
                                        Ok(())
                                    },
                            )
                        })
                        .await??;

                    log::info!(
                        "Returned {:?} for calling {:?}",
                        &results_storage[..num_results],
                        func
                    );
                }
                #[cfg(not(rr_component))]
                {
                    let _ = event;
                    bail!(
                        "Cannot parse ComponentWasmFuncBegin replay event without rr and component-model feature enabled"
                    );
                }
            }
            RREvent::ComponentPostReturn(event) => {
                #[cfg(rr_component)]
                {
                    // Grab the correct component instance
                    let key = event.instance;
                    let instance = self
                        .component_instances
                        .get_mut(&key)
                        .ok_or(ReplayError::MissingComponentInstance(key.as_u32()))?;

                    let func = component::Func::from_lifted_func(*instance, event.func_idx);
                    let mut store = self.store.as_context_mut();

                    func.post_return_async(&mut store).await?;
                }
                #[cfg(not(rr_component))]
                {
                    let _ = event;
                    bail!(
                        "Cannot parse ComponentPostReturn replay event without rr and component-model feature enabled"
                    );
                }
            }
            RREvent::CoreWasmInstantiation(event) => {
                // Find matching module from environment to instantiate
                let module = self
                    .env
                    .modules
                    .get(&event.module)
                    .ok_or(ReplayError::MissingModule(event.module))?;

                let instance = self
                    .module_linker
                    .instantiate_async(self.store.as_context_mut(), module)
                    .await?;

                // Validate the instantiation event
                event.validate(&core_events::InstantiationEvent {
                    module: *module.checksum(),
                    instance: instance.id(),
                })?;

                self.module_instances.insert(instance.id(), instance);
            }
            RREvent::CoreWasmFuncEntry(event) => {
                // Grab the correct module instance
                let key = event.origin.instance;
                let instance = self
                    .module_instances
                    .get_mut(&key)
                    .ok_or(ReplayError::MissingModuleInstance(key.as_u32()))?;

                let entity = EntityIndex::from(event.origin.index);
                let mut store = self.store.as_context_mut();
                let func = instance
                    ._get_export(store.0, entity)
                    .into_func()
                    .ok_or(ReplayError::InvalidCoreFuncIndex(entity))?;

                let params_ty = func.ty(&store).params().collect::<Vec<_>>();

                // Obtain the argument values for function call
                let mut results = vec![crate::Val::I64(0); func.ty(&store).results().len()];
                let params = event.args.to_val_vec(&mut store, params_ty);

                // Call the function
                //
                // This is almost a mirror of the usage in [`crate::Func::call_impl`]
                func.call_impl_check_args(&mut store, &params, &mut results)?;
                store
                    .on_fiber(|store| unsafe {
                        let mut ctx = store.as_context_mut();
                        func.call_impl_do_call(&mut ctx, params.as_slice(), results.as_mut_slice())
                    })
                    .await??;
            }

            _ => {
                log::error!("Unexpected top-level RR event: {:?}", rr_event);
                Err(ReplayError::IncorrectEventVariant)?
            }
        }
        Ok(())
    }

    /// Run this replay instance to completion
    pub fn run_to_completion(&mut self) -> Result<()> {
        while let Some(rr_event) = self
            .store
            .as_context_mut()
            .0
            .replay_buffer_mut()
            .expect("unexpected; replay buffer must be initialized within an instance")
            .next()
        {
            self.run_single_top_level_event(rr_event?)?;
        }
        Ok(())
    }

    /// Exactly like [`Self::run_to_completion`] but uses async stores and calls
    #[cfg(feature = "async")]
    pub async fn run_to_completion_async(&mut self) -> Result<()>
    where
        T: Send,
    {
        while let Some(rr_event) = self
            .store
            .as_context_mut()
            .0
            .replay_buffer_mut()
            .expect("unexpected; replay buffer must be initialized within an instance")
            .next()
        {
            self.run_single_top_level_event_async(rr_event?).await?;
        }
        Ok(())
    }
}

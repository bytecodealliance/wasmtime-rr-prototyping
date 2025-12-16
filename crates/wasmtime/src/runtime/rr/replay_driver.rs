use crate::rr::{
    RREvent, RRFuncArgValsConvertable, ReplayError,
    component_events::{self, RRComponentInstanceId},
    core_events::{self, RRModuleInstanceId},
};
use crate::{AsContextMut, Engine, Module, ReplayReader, ReplaySettings, Store, prelude::*};
use crate::{ValRaw, component, component::Component, rr::component_hooks};
use alloc::{collections::BTreeMap, sync::Arc};
use core::mem::MaybeUninit;
use wasmtime_environ::component::{MAX_FLAT_PARAMS, MAX_FLAT_RESULTS};
use wasmtime_environ::{EntityIndex, WasmChecksum};

/// The environment necessary to produce a [`ReplayInstance`]
#[derive(Clone)]
pub struct ReplayEnvironment {
    engine: Engine,
    modules: BTreeMap<WasmChecksum, Module>,
    components: BTreeMap<WasmChecksum, Component>,
    settings: ReplaySettings,
}

impl ReplayEnvironment {
    /// Construct a new [`ReplayEnvironment`] from scratch
    pub fn new(engine: &Engine, settings: ReplaySettings) -> Self {
        Self {
            engine: engine.clone(),
            modules: BTreeMap::new(),
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
    pub fn add_component(&mut self, component: Component) -> &mut Self {
        self.components.insert(*component.checksum(), component);
        self
    }

    fn get_component(&self, checksum: WasmChecksum) -> Result<&Component, ReplayError> {
        self.components
            .get(&checksum)
            .ok_or(ReplayError::MissingComponent(checksum))
    }

    fn get_module(&self, checksum: WasmChecksum) -> Result<&Module, ReplayError> {
        self.modules
            .get(&checksum)
            .ok_or(ReplayError::MissingModule(checksum))
    }

    /// Instantiate a new [`ReplayInstance`] using a [`ReplayReader`] in context of this environment
    pub fn instantiate(&self, reader: impl ReplayReader + 'static) -> Result<ReplayInstance> {
        self.instantiate_with(reader, |_| Ok(()), |_| Ok(()), |_| Ok(()))
    }

    /// Like [`Self::instantiate`] but allows providing a custom modifier functions for
    /// [`Store`], [`crate::Linker`], and [`component::Linker`] within the replay
    pub fn instantiate_with(
        &self,
        reader: impl ReplayReader + 'static,
        store_fn: impl FnOnce(&mut Store<ReplayHostContext>) -> Result<()>,
        module_linker_fn: impl FnOnce(&mut crate::Linker<ReplayHostContext>) -> Result<()>,
        component_linker_fn: impl FnOnce(&mut component::Linker<ReplayHostContext>) -> Result<()>,
    ) -> Result<ReplayInstance> {
        let mut store = Store::new(
            &self.engine,
            ReplayHostContext {
                module_instances: BTreeMap::new(),
                current_module_instantiation: None,
                current_component_instantiation: None,
            },
        );
        store_fn(&mut store)?;
        store.init_replaying(reader, self.settings.clone())?;

        ReplayInstance::from_environment_and_store(
            self.clone(),
            store,
            module_linker_fn,
            component_linker_fn,
        )
    }
}

/// The host context tied to the store during replay.
///
/// This context encapsulates the state from the replay environment that are
/// required to be accessible within the Store. This is an opaque type from the
/// public API perspective.
pub struct ReplayHostContext {
    /// A tracker of instantiated modules.
    ///
    /// Core wasm modules can be re-entrant and invoke methods from other instances, and this
    /// needs to be accessible within host functions.
    module_instances: BTreeMap<RRModuleInstanceId, crate::Instance>,
    /// The currently executing module instantiation event.
    ///
    /// This must be set by the driver prior to instantiation and cleared after
    /// used internally for validation.
    current_module_instantiation: Option<core_events::InstantiationEvent>,
    /// The currently executing component instantiation event.
    ///
    /// This must be set by the driver prior to instantiation and cleared after
    /// used internally for validation.
    current_component_instantiation: Option<component_events::InstantiationEvent>,
}

impl ReplayHostContext {
    /// Get a module instance from the context's tracking map
    ///
    /// This is necessary for core wasm to identify re-entrant calls during replay.
    pub(crate) fn get_module_instance(
        &self,
        id: RRModuleInstanceId,
    ) -> Result<&crate::Instance, ReplayError> {
        self.module_instances
            .get(&id)
            .ok_or(ReplayError::MissingModuleInstance(id))
    }

    /// Take the current module instantiation event from the context, leaving
    /// `None` in its place.
    pub(crate) fn take_current_module_instantiation(
        &mut self,
    ) -> Option<core_events::InstantiationEvent> {
        self.current_module_instantiation.take()
    }

    /// Take the current component instantiation event from the context, leaving
    /// `None` in its place.
    pub(crate) fn take_current_component_instantiation(
        &mut self,
    ) -> Option<component_events::InstantiationEvent> {
        self.current_component_instantiation.take()
    }
}

/// A [`ReplayInstance`] is an object providing a opaquely managed, replayable [`Store`].
///
/// Debugger capabilities in the future will interact with this object for
/// inserting breakpoints, snapshotting, and restoring state.
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
pub struct ReplayInstance {
    env: Arc<ReplayEnvironment>,
    store: Store<ReplayHostContext>,
    component_linker: component::Linker<ReplayHostContext>,
    module_linker: crate::Linker<ReplayHostContext>,
    module_instances: ModuleInstanceMap,
    component_instances: ComponentInstanceMap,
}

struct ComponentInstanceMap(BTreeMap<RRComponentInstanceId, component::Instance>);

impl ComponentInstanceMap {
    fn new() -> Self {
        Self(BTreeMap::new())
    }

    fn get_mut(
        &mut self,
        id: RRComponentInstanceId,
    ) -> Result<&mut component::Instance, ReplayError> {
        self.0
            .get_mut(&id)
            .ok_or(ReplayError::MissingComponentInstance(id))
    }
}

struct ModuleInstanceMap(BTreeMap<RRModuleInstanceId, crate::Instance>);
impl ModuleInstanceMap {
    fn new() -> Self {
        Self(BTreeMap::new())
    }

    fn get_mut(&mut self, id: RRModuleInstanceId) -> Result<&mut crate::Instance, ReplayError> {
        self.0
            .get_mut(&id)
            .ok_or(ReplayError::MissingModuleInstance(id))
    }
}

impl ReplayInstance {
    fn from_environment_and_store(
        env: ReplayEnvironment,
        store: Store<ReplayHostContext>,
        module_linker_fn: impl FnOnce(&mut crate::Linker<ReplayHostContext>) -> Result<()>,
        component_linker_fn: impl FnOnce(&mut component::Linker<ReplayHostContext>) -> Result<()>,
    ) -> Result<Self> {
        let env = Arc::new(env);
        let mut module_linker = crate::Linker::<ReplayHostContext>::new(&env.engine);
        // Replays shouldn't use any imports, so stub them all out as traps
        for module in env.modules.values() {
            module_linker.define_unknown_imports_as_traps(module)?;
        }
        module_linker_fn(&mut module_linker)?;

        let mut component_linker = component::Linker::<ReplayHostContext>::new(&env.engine);
        for component in env.components.values() {
            component_linker.define_unknown_imports_as_traps(component)?;
        }
        component_linker_fn(&mut component_linker)?;

        Ok(Self {
            env,
            store,
            component_linker,
            module_linker,
            module_instances: ModuleInstanceMap::new(),
            component_instances: ComponentInstanceMap::new(),
        })
    }

    /// Obtain a reference to the internal [`Store`].
    pub fn store(&self) -> &Store<ReplayHostContext> {
        &self.store
    }

    /// Consume the [`ReplayInstance`] and extract the internal [`Store`].
    pub fn extract_store(self) -> Store<ReplayHostContext> {
        self.store
    }

    fn insert_component_instance(&mut self, instance: component::Instance) {
        self.component_instances
            .0
            .insert(instance.id().instance().into(), instance);
    }

    fn insert_module_instance(&mut self, instance: crate::Instance) {
        self.module_instances
            .0
            .insert(instance.id().into(), instance);
        // Insert into host context tracking as well, for re-entrancy calls
        self.store
            .as_context_mut()
            .data_mut()
            .module_instances
            .insert(instance.id().into(), instance);
    }

    /// Run a single top-level event from the instance.
    ///
    /// "Top-level" events are those explicitly invoked events, namely:
    /// * Instantiation events (component/module)
    /// * Wasm function begin events (`ComponentWasmFuncBegin` for components and `CoreWasmFuncEntry` for core)
    ///
    /// All other events are transparently dispatched under the context of these top-level events.
    fn run_single_top_level_event(&mut self, rr_event: RREvent) -> Result<()> {
        match rr_event {
            RREvent::ComponentInstantiation(event) => {
                let component = self.env.get_component(event.component)?;
                // Set current instantiation event for validation
                self.store.data_mut().current_component_instantiation = Some(event);
                let instance = self
                    .component_linker
                    .instantiate(self.store.as_context_mut(), component)?;
                self.insert_component_instance(instance);
            }
            RREvent::ComponentWasmFuncBegin(event) => {
                let instance = self.component_instances.get_mut(event.instance)?;

                // Replay lowering steps and obtain raw value arguments to raw function call
                let func = component::Func::from_lifted_func(*instance, event.func_index);
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
                    )?
                };

                log::info!(
                    "Returned {:?} for calling {:?}",
                    &results_storage[..num_results],
                    func
                );
            }
            RREvent::ComponentPostReturn(event) => {
                let instance = self.component_instances.get_mut(event.instance)?;
                let func = component::Func::from_lifted_func(*instance, event.func_index);
                let mut store = self.store.as_context_mut();
                func.post_return(&mut store)?;
            }
            RREvent::CoreWasmInstantiation(event) => {
                let module = self.env.get_module(event.module)?;
                // Set current instantiation event for validation
                self.store.data_mut().current_module_instantiation = Some(event);
                let instance = self
                    .module_linker
                    .instantiate(self.store.as_context_mut(), module)?;
                self.insert_module_instance(instance);
            }
            RREvent::CoreWasmFuncEntry(event) => {
                let instance = self.module_instances.get_mut(event.instance)?;
                let entity = EntityIndex::from(event.func_index);
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
                log::error!("Unexpected top-level RR event: {rr_event:?}");
                Err(ReplayError::IncorrectEventVariant)?
            }
        }
        Ok(())
    }

    /// Exactly like [`Self::run_single_top_level_event`] but uses async stores and calls.
    #[cfg(feature = "async")]
    async fn run_single_top_level_event_async(&mut self, rr_event: RREvent) -> Result<()> {
        match rr_event {
            RREvent::ComponentInstantiation(event) => {
                let component = self.env.get_component(event.component)?;
                // Set current instantiation event for validation
                self.store.data_mut().current_component_instantiation = Some(event);
                let instance = self
                    .component_linker
                    .instantiate_async(self.store.as_context_mut(), component)
                    .await?;
                self.insert_component_instance(instance);
            }
            RREvent::ComponentWasmFuncBegin(event) => {
                let instance = self.component_instances.get_mut(event.instance)?;

                // Replay lowering steps and obtain raw value arguments to raw function call
                let func = component::Func::from_lifted_func(*instance, event.func_index);
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
            RREvent::ComponentPostReturn(event) => {
                let instance = self.component_instances.get_mut(event.instance)?;
                let func = component::Func::from_lifted_func(*instance, event.func_index);
                let mut store = self.store.as_context_mut();
                func.post_return_async(&mut store).await?;
            }
            RREvent::CoreWasmInstantiation(event) => {
                let module = self.env.get_module(event.module)?;
                // Set current instantiation event for validation
                self.store.data_mut().current_module_instantiation = Some(event);
                let instance = self
                    .module_linker
                    .instantiate_async(self.store.as_context_mut(), module)
                    .await?;
                self.insert_module_instance(instance);
            }
            RREvent::CoreWasmFuncEntry(event) => {
                let instance = self.module_instances.get_mut(event.instance)?;
                let entity = EntityIndex::from(event.func_index);
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
                log::error!("Unexpected top-level RR event: {rr_event:?}");
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
    pub async fn run_to_completion_async(&mut self) -> Result<()> {
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

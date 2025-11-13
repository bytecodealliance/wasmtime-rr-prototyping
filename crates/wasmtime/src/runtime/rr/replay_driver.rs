#[cfg(feature = "rr-component")]
use crate::component::{self, Component};
#[cfg(feature = "rr-component")]
use crate::rr::component_events;
use crate::rr::{
    RREvent, ReplayError, Validate, component_hooks::ReplayLoweringPhase, core_events,
};
use crate::{
    AsContextMut, Engine, Module, ReplayReader, ReplaySettings, Store, ValRaw, ValType, prelude::*,
};
use alloc::collections::BTreeMap;
use core::mem::MaybeUninit;
use wasmtime_environ::EntityIndex;
#[cfg(feature = "rr-component")]
use wasmtime_environ::component::{MAX_FLAT_PARAMS, MAX_FLAT_RESULTS};

/// The environment necessary to produce a [`ReplayInstance`]
#[derive(Clone)]
pub struct ReplayEnvironment {
    engine: Engine,
    modules: BTreeMap<[u8; 32], Module>,
    components: BTreeMap<[u8; 32], Component>,
    settings: ReplaySettings,
}

impl ReplayEnvironment {
    /// Create a new [`ReplayEnvironment`]
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

    /// Instantiate a new [`ReplayInstance`] using a [`ReplayReader`] in context of this environment
    pub fn instantiate(&self, reader: impl ReplayReader + 'static) -> Result<ReplayInstance<'_>> {
        ReplayInstance::from_environment(self, reader)
    }
}

/// A [`ReplayInstance`] is an object providing a opaquely managed, replayable [`Store`]
///
/// Debugger capabilities in the future will interact with this object for
/// inserting breakpoints, snapshotting, and restoring state
pub struct ReplayInstance<'a> {
    /// The store doesn't need any host data because the trace format and
    /// replay is designed to be embedding-agnostic
    store: Store<()>,
    component_linker: component::Linker<()>,
    module_linker: crate::Linker<()>,
    modules: &'a BTreeMap<[u8; 32], Module>,
    components: &'a BTreeMap<[u8; 32], Component>,
    module_instances: BTreeMap<core_events::InstantiationEvent, crate::Instance>,
    component_instances: BTreeMap<component_events::InstantiationEvent, component::Instance>,
}

impl<'a> ReplayInstance<'a> {
    fn from_environment(
        env: &'a ReplayEnvironment,
        reader: impl ReplayReader + 'static,
    ) -> Result<Self> {
        let mut store = Store::new(&env.engine, ());
        store.init_replaying(reader, env.settings.clone())?;
        let mut component_linker = component::Linker::<()>::new(&env.engine);
        let mut module_linker = crate::Linker::<()>::new(&env.engine);
        // Replays shouldn't use any imports, so stub them all out as traps
        for module in env.modules.values() {
            // Defining unknown imports as trap seems to not actually trigger the entrypoint?
            // Use default values instead for now
            module_linker.define_unknown_imports_as_traps(module)?;
        }
        for component in env.components.values() {
            component_linker.define_unknown_imports_as_traps(component)?;
        }
        Ok(Self {
            store,
            component_linker,
            module_linker,
            modules: &env.modules,
            components: &env.components,
            module_instances: BTreeMap::new(),
            component_instances: BTreeMap::new(),
        })
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
            // The only valid "top-level" events are:
            // * Instantiation events (component/module)
            // * Wasm function begin events (`ComponentWasmFuncBegin` for components and `CoreWasmFuncEntry` for core)
            //
            // All other events are transparently dispatched under the context of these top-level events
            match rr_event? {
                RREvent::ComponentInstantiation(event) => {
                    #[cfg(feature = "rr-component")]
                    {
                        // Find matching component from environment to instantiate
                        let component = self
                            .components
                            .get(&event.0)
                            .ok_or(ReplayError::MissingComponentOrModule)?;

                        let instance = self
                            .component_linker
                            .instantiate(self.store.as_context_mut(), component)?;
                        // Validate the instantiation event
                        event.validate(&component_events::InstantiationEvent(
                            *component.checksum(),
                            instance.id().instance(),
                        ))?;

                        let ret = self.component_instances.insert(event, instance);
                        // Ensures that an already-instantiated configuration is not re-instantiated
                        assert!(ret.is_none());
                    }
                    #[cfg(not(feature = "rr-component"))]
                    {
                        anyhow!(
                            "Cannot parse ComponentInstantation replay event without rr-component feature enabled"
                        );
                    }
                }
                RREvent::ComponentWasmFuncBegin(event) => {
                    #[cfg(feature = "rr-component")]
                    {
                        // Grab the correct component instance
                        let key =
                            component_events::InstantiationEvent(event.component, event.instance);
                        let instance = self
                            .component_instances
                            .get_mut(&key)
                            .ok_or(ReplayError::MissingComponentOrModuleInstance)?;

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
                                    cx.replay_lowering(Some(dst), ReplayLoweringPhase::WasmFuncEntry)
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
                    #[cfg(not(feature = "rr-component"))]
                    {
                        anyhow!(
                            "Cannot parse ComponentWasmFuncBegin replay event without rr-component feature enabled"
                        );
                    }
                }
                RREvent::CoreWasmInstantiation(event) => {
                    // Find matching module from environment to instantiate
                    let module = self
                        .modules
                        .get(&event.0)
                        .ok_or(ReplayError::MissingComponentOrModule)?;

                    let instance = self
                        .module_linker
                        .instantiate(self.store.as_context_mut(), module)?;

                    // Validate the instantiation event
                    event.validate(&core_events::InstantiationEvent(
                        *module.checksum(),
                        instance.id(),
                    ))?;

                    let ret = self.module_instances.insert(event, instance);
                    // Ensures that an already-instantiated configuration is not re-instantiated
                    assert!(ret.is_none());
                }
                RREvent::CoreWasmFuncEntry(event) => {
                    // Grab the correct module instance
                    let key = core_events::InstantiationEvent(event.module, event.origin.instance);
                    let instance = self
                        .module_instances
                        .get_mut(&key)
                        .ok_or(ReplayError::MissingComponentOrModule)?;

                    let entity = EntityIndex::from(event.origin.index);
                    let mut store = self.store.as_context_mut();
                    let func = instance
                        ._get_export(store.0, entity)
                        .into_func()
                        .ok_or(ReplayError::IncorrectCoreFuncIndex)?;

                    let params_ty = func.ty(&store).params().collect::<Vec<ValType>>();

                    // Obtain the argument values for function call
                    let mut results = vec![crate::Val::I64(0); func.ty(&store).results().len()];
                    let params = event.to_val_vec(&mut store, params_ty);

                    // Call the function
                    //
                    // This is almost a mirror of the usage in [`crate::Func::call`] or [`crate::Func::call_async`]
                    func.call_impl_check_args(&mut store, &params, &mut results)?;
                    unsafe {
                        func.call_impl_do_call(
                            &mut store,
                            params.as_slice(),
                            results.as_mut_slice(),
                        )?;
                    }
                }

                _ => Err(ReplayError::IncorrectEventVariant)?,
            }
        }
        Ok(())
    }
}

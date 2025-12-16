#[cfg(feature = "rr")]
use super::{replay_data_from_store, replay_data_from_store_mut};
use crate::rr::FlatBytes;
#[cfg(feature = "rr")]
use crate::rr::{
    RREvent, RRFuncArgVals, RRFuncArgValsConvertable, ReplayError, Replayer, ResultEvent, Validate,
    common_events::{HostFuncEntryEvent, HostFuncReturnEvent, WasmFuncReturnEvent},
    core_events::{InstantiationEvent, WasmFuncEntryEvent},
};
use crate::store::{InstanceId, StoreOpaque};
use crate::{Caller, FuncType, Module, StoreContextMut, ValRaw, WasmFuncOrigin, prelude::*};
#[cfg(feature = "rr")]
use wasmtime_environ::EntityIndex;
use wasmtime_environ::WasmChecksum;

/// Record and replay hook operation for core wasm function entry events
///
/// Recording/replay validation DOES NOT happen if origin is `None`
#[inline]
pub fn record_and_replay_validate_wasm_func<F, T>(
    wasm_call: F,
    args: &[ValRaw],
    ty: &FuncType,
    origin: Option<WasmFuncOrigin>,
    store: &mut StoreContextMut<'_, T>,
) -> Result<()>
where
    F: FnOnce(&mut StoreContextMut<'_, T>) -> Result<()>,
{
    let _ = (args, ty, origin);
    #[cfg(feature = "rr")]
    {
        if let Some(origin) = origin {
            store.0.record_event(|| {
                let flat = ty.params().map(|t| t.to_wasm_type().byte_size());
                WasmFuncEntryEvent {
                    instance: origin.instance.into(),
                    func_index: origin.index.into(),
                    args: RRFuncArgVals::from_flat_iter(args, flat),
                }
            })?;
        }
    }
    let result = wasm_call(store);
    #[cfg(feature = "rr")]
    {
        if origin.is_some() {
            if let Err(e) = &result {
                log::warn!("Wasm function call exited with error: {e:?}");
            }
            let flat = ty.results().map(|t| t.to_wasm_type().byte_size());
            let result = result.map(|_| RRFuncArgVals::from_flat_iter(args, flat));
            store.0.record_event_validation(|| {
                WasmFuncReturnEvent(ResultEvent::from_anyhow_result(&result))
            })?;
            store
                .0
                .next_replay_event_validation::<WasmFuncReturnEvent, _, &Result<RRFuncArgVals>>(
                    || &result,
                )?;
            result?;
            Ok(())
        } else {
            result
        }
    }
    #[cfg(not(feature = "rr"))]
    {
        result
    }
}

/// Record hook operation for host function entry events
#[inline]
pub fn record_validate_host_func_entry<T>(
    args: &[T],
    flat: impl Iterator<Item = u8>,
    store: &mut StoreOpaque,
) -> Result<()>
where
    T: FlatBytes,
{
    let _ = (args, &flat, &store);
    #[cfg(feature = "rr")]
    store.record_event_validation(|| HostFuncEntryEvent {
        args: RRFuncArgVals::from_flat_iter(args, flat),
    })?;
    Ok(())
}

/// Record hook operation for host function return events
#[inline]
pub fn record_host_func_return<T>(
    args: &[T],
    flat: impl Iterator<Item = u8>,
    store: &mut StoreOpaque,
) -> Result<()>
where
    T: FlatBytes,
{
    let _ = (args, &flat, &store);
    // Record the return values
    #[cfg(feature = "rr")]
    store.record_event(|| HostFuncReturnEvent {
        args: RRFuncArgVals::from_flat_iter(args, flat),
    })?;
    Ok(())
}

/// Replay hook operation for host function entry events
#[inline]
pub fn replay_validate_host_func_entry<T>(
    args: &[T],
    flat: impl Iterator<Item = u8>,
    store: &mut StoreOpaque,
) -> Result<()>
where
    T: FlatBytes,
{
    let _ = (args, &flat, &store);
    #[cfg(feature = "rr")]
    store.next_replay_event_validation::<HostFuncEntryEvent, _, _>(|| HostFuncEntryEvent {
        args: RRFuncArgVals::from_flat_iter(args, flat),
    })?;
    Ok(())
}

/// Replay hook operation for host function return events.
#[inline]
pub fn replay_host_func_return<T, U: 'static>(
    args: &mut [T],
    caller: &mut Caller<'_, U>,
) -> Result<()>
where
    T: FlatBytes,
{
    #[cfg(feature = "rr")]
    {
        // Core wasm can be re-entrant, so we need to check for this
        let mut complete = false;
        while !complete {
            let buf = caller.store.0.replay_buffer_mut().unwrap();
            let event = buf.next_event()?;
            match event {
                RREvent::HostFuncReturn(event) => {
                    event.args.into_raw_slice(args);
                    complete = true;
                }
                // Re-entrant call into wasm function: this resembles the implementation in [`ReplayInstance`]
                RREvent::CoreWasmFuncEntry(event) => {
                    let entity = EntityIndex::from(event.func_index);

                    // Unwrapping the `replay_buffer_mut()` above ensures that we are in replay mode
                    // passing the safety contract for `replay_data_from_store`
                    let replay_data = unsafe { replay_data_from_store(&caller.store) };

                    // Grab the correct module instance
                    let instance = replay_data.get_module_instance(event.instance)?;

                    let mut store = &mut caller.store;
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
                        func.call_impl_do_call(
                            &mut store,
                            params.as_slice(),
                            results.as_mut_slice(),
                        )?;
                    }
                }
                _ => {
                    bail!("Unexpected event during core wasm host function replay: {event:?}",);
                }
            }
        }
    }
    let _ = (args, caller);
    Ok(())
}

/// Hook for recording a module instantiation event and validating the
/// instantiation during replay.
pub fn record_and_replay_validate_instantiation<T: 'static>(
    store: &mut StoreContextMut<'_, T>,
    module: WasmChecksum,
    instance: InstanceId,
) -> Result<()> {
    #[cfg(feature = "rr")]
    {
        store.0.record_event(|| InstantiationEvent {
            module,
            instance: instance.into(),
        })?;
        if store.0.replay_enabled() {
            let replay_data = unsafe { replay_data_from_store_mut(store) };
            replay_data.take_current_module_instantiation().expect(
                "replay driver should have set module instantiate data before trying to validate it",
            ).validate(&InstantiationEvent { module, instance: instance.into() })?;
        }
    }
    let _ = (store, module, instance);
    Ok(())
}

/// Ensure that memories are not exported memories in Core wasm modules when
/// recording is enabled.
pub fn rr_validate_module_unexported_memory(module: &Module) -> Result<()> {
    // Check for exported memories when recording is enabled.
    #[cfg(feature = "rr")]
    {
        if module.engine().is_recording()
            && module.exports().any(|export| {
                if let crate::ExternType::Memory(_) = export.ty() {
                    true
                } else {
                    false
                }
            })
        {
            bail!("Cannot support recording for core wasm modules when a memory is exported");
        }
    }
    let _ = module;
    Ok(())
}

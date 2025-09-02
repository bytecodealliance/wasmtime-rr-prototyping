use crate::ValRaw;
use crate::prelude::*;
#[cfg(feature = "rr")]
use crate::rr::core_events::HostFuncReturnEvent;
use crate::store::StoreOpaque;
use core::mem::MaybeUninit;
use wasmtime_environ::WasmFuncType;

#[inline]
/// Record and replay hook operation for host function entry events
pub fn record_replay_host_func_entry(
    args: &[MaybeUninit<ValRaw>],
    wasm_func_type: &WasmFuncType,
    store: &mut StoreOpaque,
) -> Result<()> {
    #[cfg(all(feature = "rr", feature = "rr-validate"))]
    {
        // Record/replay the raw parameter args
        use crate::rr::core_events::HostFuncEntryEvent;
        store.record_event_validation(|| {
            let num_params = wasm_func_type.params().len();
            HostFuncEntryEvent::new(&args[..num_params], wasm_func_type.clone())
        })?;
        store.next_replay_event_validation::<HostFuncEntryEvent, _>(wasm_func_type)?;
    }
    let _ = (args, wasm_func_type, store);
    Ok(())
}

#[inline]
/// Record hook operation for host function return events
pub fn record_host_func_return(
    args: &[MaybeUninit<ValRaw>],
    wasm_func_type: &WasmFuncType,
    store: &mut StoreOpaque,
) -> Result<()> {
    // Record the return values
    #[cfg(feature = "rr")]
    store.record_event(|| {
        let func_type = wasm_func_type;
        let num_results = func_type.params().len();
        HostFuncReturnEvent::new(&args[..num_results])
    })?;
    let _ = (args, wasm_func_type, store);
    Ok(())
}

#[inline]
/// Replay hook operation for host function return events
pub fn replay_host_func_return(
    args: &mut [MaybeUninit<ValRaw>],
    wasm_func_type: &WasmFuncType,
    store: &mut StoreOpaque,
) -> Result<()> {
    #[cfg(feature = "rr")]
    store.next_replay_event_and(|event: HostFuncReturnEvent| {
        event.move_into_slice(args);
        Ok(())
    })?;
    let _ = (args, wasm_func_type, store);
    Ok(())
}

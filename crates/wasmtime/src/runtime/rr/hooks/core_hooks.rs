use crate::rr::FlatBytes;
#[cfg(feature = "rr")]
use crate::rr::core_events::{HostFuncReturnEvent, WasmFuncEntryEvent};
#[cfg(all(feature = "rr", feature = "rr-validate"))]
use crate::rr::{
    RRFuncArgVals, ResultEvent, core_events::HostFuncEntryEvent, core_events::WasmFuncReturnEvent,
};
#[cfg(feature = "rr")]
use crate::store::StoreInstanceId;
use crate::store::StoreOpaque;
use crate::{FuncType, StoreContextMut, ValRaw, WasmFuncOrigin, prelude::*};

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
            let checksum = *store
                .0
                .module_for_instance(StoreInstanceId::new(store.0.id(), origin.instance))
                .unwrap()
                .checksum();
            store.0.record_event(|| {
                let flat = ty
                    .params()
                    .map(|t| t.to_wasm_type().byte_size())
                    .collect::<Vec<u8>>();
                WasmFuncEntryEvent {
                    module: checksum,
                    origin,
                    args: RRFuncArgVals::from_flat_u8(args, &flat),
                }
            })?;
        }
    }
    let result = wasm_call(store);
    #[cfg(all(feature = "rr", feature = "rr-validate"))]
    {
        if origin.is_some() {
            let flat = ty
                .results()
                .map(|t| t.to_wasm_type().byte_size())
                .collect::<Vec<u8>>();
            let result = result.map(|_| RRFuncArgVals::from_raw_slice(args, flat.iter().copied()));
            store.0.record_event_validation(|| {
                WasmFuncReturnEvent(ResultEvent::from_anyhow_result(&result))
            })?;
            store
                .0
                .next_replay_event_validation::<WasmFuncReturnEvent, _, &Result<RRFuncArgVals>>(
                    || &result,
                )?;
            result?;
        }
        return Ok(());
    }
    #[cfg(not(all(feature = "rr", feature = "rr-validate")))]
    return result;
}

/// Record and replay hook operation for host function entry events
#[inline]
pub fn record_and_replay_validate_host_func_entry<T>(
    args: &[T],
    flat: &[u8],
    store: &mut StoreOpaque,
) -> Result<()>
where
    T: FlatBytes,
{
    #[cfg(all(feature = "rr", feature = "rr-validate"))]
    {
        // Record/replay the raw parameter args
        store.record_event_validation(|| HostFuncEntryEvent {
            args: RRFuncArgVals::from_flat_u8(args, flat),
        })?;
        store.next_replay_event_validation::<HostFuncEntryEvent, _, _>(|| HostFuncEntryEvent {
            args: RRFuncArgVals::from_flat_u8(args, flat),
        })?;
    }
    let _ = (args, flat, store);
    Ok(())
}

/// Record hook operation for host function return events
#[inline]
pub fn record_host_func_return<T>(args: &[T], flat: &[u8], store: &mut StoreOpaque) -> Result<()>
where
    T: FlatBytes,
{
    // Record the return values
    #[cfg(feature = "rr")]
    store.record_event(|| HostFuncReturnEvent {
        args: RRFuncArgVals::from_flat_u8(args, flat),
    })?;
    let _ = (args, flat, store);
    Ok(())
}

/// Replay hook operation for host function return events
#[inline]
pub fn replay_host_func_return<T>(args: &mut [T], store: &mut StoreOpaque) -> Result<()>
where
    T: FlatBytes,
{
    #[cfg(feature = "rr")]
    store.next_replay_event_and(|event: HostFuncReturnEvent| {
        event.args.into_raw_slice(args);
        Ok(())
    })?;
    let _ = (args, store);
    Ok(())
}

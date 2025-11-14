use crate::rr::FlatBytes;
#[cfg(feature = "rr")]
use crate::rr::{
    RRFuncArgVals, ResultEvent, common_events::HostFuncReturnEvent,
    common_events::WasmFuncReturnEvent, core_events::HostFuncEntryEvent,
    core_events::WasmFuncEntryEvent,
};
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
            store.0.record_event(|| {
                let flat = ty.params().map(|t| t.to_wasm_type().byte_size());
                WasmFuncEntryEvent {
                    origin,
                    args: RRFuncArgVals::from_flat_iter(args, flat),
                }
            })?;
        }
    }
    let result = wasm_call(store);
    #[cfg(feature = "rr")]
    {
        if origin.is_some() {
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
        }
        return Ok(());
    }
    #[cfg(not(feature = "rr"))]
    return result;
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

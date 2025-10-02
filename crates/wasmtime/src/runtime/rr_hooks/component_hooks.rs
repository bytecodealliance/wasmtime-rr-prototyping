use crate::ValRaw;
#[cfg(feature = "component-model")]
use crate::component::func::LowerContext;
#[cfg(feature = "rr-component")]
use crate::rr::component_events::{
    HostFuncReturnEvent, LowerFlatReturnEvent, LowerMemoryReturnEvent, WasmFuncEntryEvent,
};
#[cfg(all(feature = "rr-component", feature = "rr-validate"))]
use crate::rr::{
    RRFuncArgVals, component_events::WasmFuncReturnEvent, func_argvals_from_raw_slice,
};
use crate::store::StoreOpaque;
use crate::{StoreContextMut, prelude::*};
use core::mem::MaybeUninit;
#[cfg(feature = "component-model")]
use wasmtime_environ::component::{ExportIndex, InterfaceType, TypeFuncIndex};

/// Indicator type signalling the context during lowering
#[cfg(feature = "rr-component")]
pub enum ReplayLoweringPhase {
    WasmFuncEntry,
    HostFuncReturn,
}

/// Record hook wrapping a wasm component export function invocation and replay
/// validation of return value
#[inline]
pub fn record_replay_wasm_func<F, T>(
    wasm_call: F,
    args: &[ValRaw],
    func_idx: ExportIndex,
    component: [u8; 32],
    store: &mut StoreContextMut<'_, T>,
) -> Result<()>
where
    F: FnOnce(&mut StoreContextMut<'_, T>) -> Result<()>,
{
    let _ = (args, component, func_idx);
    #[cfg(feature = "rr-component")]
    store
        .0
        .record_event(|| WasmFuncEntryEvent::new(args, component, func_idx))?;
    let result = wasm_call(store);
    #[cfg(all(feature = "rr-component", feature = "rr-validate"))]
    {
        let result = result.map(|_| func_argvals_from_raw_slice(args));
        store
            .0
            .record_event_validation(|| WasmFuncReturnEvent::from_anyhow_result(&result))?;
        store
            .0
            .next_replay_event_validation::<WasmFuncReturnEvent, Result<RRFuncArgVals>>(&result)?;
        result?;
        return Ok(());
    }
    #[cfg(not(all(feature = "rr-component", feature = "rr-validate")))]
    return result;
}

/// Record/replay hook operation for host function entry events
#[inline]
pub fn record_replay_host_func_entry(
    args: &mut [MaybeUninit<ValRaw>],
    func_idx: &TypeFuncIndex,
    store: &mut StoreOpaque,
) -> Result<()> {
    #[cfg(all(feature = "rr-component", feature = "rr-validate"))]
    {
        use crate::rr::component_events::HostFuncEntryEvent;
        store.record_event_validation(|| HostFuncEntryEvent::new(args, func_idx.clone()))?;
        store.next_replay_event_validation::<HostFuncEntryEvent, _>(func_idx)?;
    }
    let _ = (args, func_idx, store);
    Ok(())
}

/// Record hook operation for host function return events
#[inline]
pub fn record_host_func_return(
    args: &[MaybeUninit<ValRaw>],
    store: &mut StoreOpaque,
) -> Result<()> {
    #[cfg(feature = "rr-component")]
    store.record_event(|| HostFuncReturnEvent::new(args))?;
    let _ = (args, store);
    Ok(())
}

/// Record hook wrapping a memory lowering call of component types
#[inline]
pub fn record_lower_memory<F, T>(
    lower_store: F,
    cx: &mut LowerContext<'_, T>,
    ty: InterfaceType,
    offset: usize,
) -> Result<()>
where
    F: FnOnce(&mut LowerContext<'_, T>, InterfaceType, usize) -> Result<()>,
{
    #[cfg(all(feature = "rr-component", feature = "rr-validate"))]
    {
        use crate::rr::component_events::LowerMemoryEntryEvent;
        cx.store
            .0
            .record_event_validation(|| LowerMemoryEntryEvent { ty, offset })?;
    }
    let store_result = lower_store(cx, ty, offset);
    #[cfg(feature = "rr-component")]
    cx.store
        .0
        .record_event(|| LowerMemoryReturnEvent::from_anyhow_result(&store_result))?;
    store_result
}

/// Record hook wrapping a flat lowering call of component types
#[inline]
pub fn record_lower_flat<F, T>(
    lower: F,
    cx: &mut LowerContext<'_, T>,
    ty: InterfaceType,
) -> Result<()>
where
    F: FnOnce(&mut LowerContext<'_, T>, InterfaceType) -> Result<()>,
{
    #[cfg(all(feature = "rr-component", feature = "rr-validate"))]
    {
        use crate::rr::component_events::LowerFlatEntryEvent;
        cx.store
            .0
            .record_event_validation(|| LowerFlatEntryEvent { ty })?;
    }
    let lower_result = lower(cx, ty);
    #[cfg(feature = "rr-component")]
    cx.store
        .0
        .record_event(|| LowerFlatReturnEvent::from_anyhow_result(&lower_result))?;
    lower_result
}

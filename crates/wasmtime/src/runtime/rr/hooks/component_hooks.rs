use crate::ValRaw;
#[cfg(feature = "component-model")]
use crate::component::func::LowerContext;
use crate::component::store::StoreComponentInstanceId;
#[cfg(feature = "rr-component")]
use crate::rr::component_events::{
    HostFuncReturnEvent, LowerFlatReturnEvent, LowerMemoryReturnEvent, ResultEvent,
    WasmFuncEntryEvent,
};
#[cfg(all(feature = "rr-component", feature = "rr-validate"))]
use crate::rr::{RRFuncArgVals, component_events::WasmFuncReturnEvent};
use crate::store::StoreOpaque;
use crate::{StoreContextMut, prelude::*};
use alloc::sync::Arc;
use core::mem::MaybeUninit;
#[cfg(feature = "component-model")]
use wasmtime_environ::component::{ComponentTypes, ExportIndex, InterfaceType, TypeFuncIndex};
#[cfg(all(feature = "rr-component"))]
use wasmtime_environ::component::{MAX_FLAT_PARAMS, MAX_FLAT_RESULTS};

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
    type_idx: TypeFuncIndex,
    id: StoreComponentInstanceId,
    store: &mut StoreContextMut<'_, T>,
) -> Result<()>
where
    F: FnOnce(&mut StoreContextMut<'_, T>) -> Result<()>,
{
    let _ = (args, id, func_idx, type_idx);
    #[cfg(feature = "rr-component")]
    {
        let component = id.get(store.0).component();
        let types = component.types();
        let checksum = *component.checksum();
        let flat_params = types.flat_types_storage(
            &InterfaceType::Tuple(types[type_idx].params),
            MAX_FLAT_PARAMS,
        );
        store
            .0
            .record_event(|| WasmFuncEntryEvent::new(args, flat_params, checksum, func_idx))?;
    }
    let result = wasm_call(store);
    #[cfg(all(feature = "rr-component", feature = "rr-validate"))]
    {
        let component = id.get(store.0).component();
        let types = component.types();
        let flat_results = types.flat_types_storage(
            &InterfaceType::Tuple(types[type_idx].results),
            MAX_FLAT_RESULTS,
        );
        let result = result.map(|_| RRFuncArgVals::from_raw_slice(args, flat_results.iter32()));
        store.0.record_event_validation(|| {
            WasmFuncReturnEvent(ResultEvent::from_anyhow_result(&result))
        })?;
        store
            .0
            .next_replay_event_validation::<WasmFuncReturnEvent, _, &Result<RRFuncArgVals>>(
                || &result,
            )?;
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
    types: &Arc<ComponentTypes>,
    type_idx: &TypeFuncIndex,
    store: &mut StoreOpaque,
) -> Result<()> {
    #[cfg(all(feature = "rr-component", feature = "rr-validate"))]
    {
        use crate::rr::component_events::HostFuncEntryEvent;
        let event = || {
            let flat_params = types.flat_types_storage(
                &InterfaceType::Tuple(types[*type_idx].params),
                MAX_FLAT_PARAMS,
            );
            HostFuncEntryEvent::new(args, flat_params, type_idx.clone())
        };
        store.record_event_validation(|| event())?;
        store.next_replay_event_validation::<HostFuncEntryEvent, _, _>(|| event())?;
    }
    let _ = (args, types, type_idx, store);
    Ok(())
}

/// Record hook operation for host function return events
#[inline]
pub fn record_host_func_return(
    args: &[MaybeUninit<ValRaw>],
    types: &ComponentTypes,
    ty: &InterfaceType,
    store: &mut StoreOpaque,
) -> Result<()> {
    #[cfg(feature = "rr-component")]
    store.record_event(|| {
        let flat_results = types.flat_types_storage(&ty, MAX_FLAT_RESULTS);
        HostFuncReturnEvent::new_from_flat_storage(args, flat_results)
    })?;
    let _ = (args, types, ty, store);
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
        .record_event(|| LowerMemoryReturnEvent(ResultEvent::from_anyhow_result(&store_result)))?;
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
        .record_event(|| LowerFlatReturnEvent(ResultEvent::from_anyhow_result(&lower_result)))?;
    lower_result
}

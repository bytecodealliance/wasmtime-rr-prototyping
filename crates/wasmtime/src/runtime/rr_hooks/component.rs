use crate::ValRaw;
#[cfg(feature = "component-model")]
use crate::component::func::LowerContext;
use crate::prelude::*;
use crate::store::StoreOpaque;
use core::mem::MaybeUninit;
#[cfg(feature = "component-model")]
use wasmtime_environ::component::{InterfaceType, TypeFuncIndex};

#[cfg(feature = "rr-component")]
use crate::rr::component_events::{HostFuncReturnEvent, LowerReturnEvent, LowerStoreReturnEvent};

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

/// Record hook wrapping a lowering `store` call of component types
#[inline]
pub fn record_lower_store<F, T>(
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
        use crate::rr::component_events::LowerStoreEntryEvent;
        cx.store
            .0
            .record_event_validation(|| LowerStoreEntryEvent { ty, offset })?;
    }
    let store_result = lower_store(cx, ty, offset);
    #[cfg(feature = "rr-component")]
    cx.store
        .0
        .record_event(|| LowerStoreReturnEvent::from_anyhow_result(&store_result))?;
    store_result
}

/// Record hook wrapping a lowering `lower` call of component types
#[inline]
pub fn record_lower<F, T>(lower: F, cx: &mut LowerContext<'_, T>, ty: InterfaceType) -> Result<()>
where
    F: FnOnce(&mut LowerContext<'_, T>, InterfaceType) -> Result<()>,
{
    #[cfg(all(feature = "rr-component", feature = "rr-validate"))]
    {
        use crate::rr::component_events::LowerEntryEvent;
        cx.store
            .0
            .record_event_validation(|| LowerEntryEvent { ty })?;
    }
    let lower_result = lower(cx, ty);
    #[cfg(feature = "rr-component")]
    cx.store
        .0
        .record_event(|| LowerReturnEvent::from_anyhow_result(&lower_result))?;
    lower_result
}

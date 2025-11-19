use crate::ValRaw;
#[cfg(feature = "rr-component")]
use crate::rr::{RecordBuffer, Recorder, component_events::MemorySliceWriteEvent};

use core::ops::{Deref, DerefMut};

use crate::component::func::LowerContext;
#[cfg(feature = "rr-component")]
use crate::rr::common_events::{HostFuncEntryEvent, WasmFuncReturnEvent};
#[cfg(feature = "rr-component")]
use crate::rr::component_events::{
    LowerFlatEntryEvent, LowerFlatReturnEvent, LowerMemoryEntryEvent, LowerMemoryReturnEvent,
    WasmFuncEntryEvent,
};
#[cfg(feature = "rr-component")]
use crate::rr::{RRFuncArgVals, ResultEvent, common_events::HostFuncReturnEvent};
use crate::store::StoreOpaque;
use crate::{StoreContextMut, prelude::*};
use alloc::sync::Arc;
use core::mem::MaybeUninit;
use wasmtime_environ::component::{ComponentTypes, InterfaceType, TypeFuncIndex};
#[cfg(all(feature = "rr-component"))]
use wasmtime_environ::component::{MAX_FLAT_PARAMS, MAX_FLAT_RESULTS};

/// Indicator type signalling the context during lowering
#[cfg(feature = "rr-component")]
#[derive(Debug)]
pub enum ReplayLoweringPhase {
    WasmFuncEntry,
    HostFuncReturn,
}

/// Record hook wrapping a wasm component export function invocation and replay
/// validation of return value
#[inline]
pub fn record_and_replay_validate_wasm_func<F, T>(
    wasm_call: F,
    args: &[ValRaw],
    type_idx: TypeFuncIndex,
    types: Arc<ComponentTypes>,
    store: &mut StoreContextMut<'_, T>,
) -> Result<()>
where
    F: FnOnce(&mut StoreContextMut<'_, T>) -> Result<()>,
{
    let _ = (args, type_idx, &types);
    #[cfg(feature = "rr-component")]
    store.0.record_event(|| {
        let flat_params = types.flat_types_storage_or_pointer(
            &InterfaceType::Tuple(types[type_idx].params),
            MAX_FLAT_PARAMS,
        );
        WasmFuncEntryEvent {
            args: RRFuncArgVals::from_flat_storage(args, flat_params),
        }
    })?;
    let result = wasm_call(store);
    #[cfg(feature = "rr-component")]
    {
        let flat_results = types.flat_types_storage_or_pointer(
            &InterfaceType::Tuple(types[type_idx].results),
            MAX_FLAT_RESULTS,
        );
        let result = result.map(|_| RRFuncArgVals::from_flat_iter(args, flat_results.iter32()));
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
    }
    #[cfg(not(feature = "rr-component"))]
    {
        result
    }
}

/// Record hook operation for host function entry events
#[inline]
pub fn record_validate_host_func_entry(
    args: &mut [MaybeUninit<ValRaw>],
    types: &Arc<ComponentTypes>,
    param_tys: &InterfaceType,
    store: &mut StoreOpaque,
) -> Result<()> {
    #[cfg(feature = "rr-component")]
    store.record_event_validation(|| create_host_func_entry_event(args, types, param_tys))?;
    let _ = (args, types, param_tys, store);
    Ok(())
}

/// Replay hook operation for host function entry events
#[inline]
pub fn replay_validate_host_func_entry(
    args: &mut [MaybeUninit<ValRaw>],
    types: &Arc<ComponentTypes>,
    param_tys: &InterfaceType,
    store: &mut StoreOpaque,
) -> Result<()> {
    #[cfg(feature = "rr-component")]
    store.next_replay_event_validation::<HostFuncEntryEvent, _, _>(|| {
        create_host_func_entry_event(args, types, param_tys)
    })?;
    let _ = (args, types, param_tys, store);
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
        let flat_results = types.flat_types_storage_or_pointer(&ty, MAX_FLAT_RESULTS);
        HostFuncReturnEvent {
            args: RRFuncArgVals::from_flat_storage(args, flat_results),
        }
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
    #[cfg(feature = "rr-component")]
    cx.store
        .0
        .record_event_validation(|| LowerMemoryEntryEvent { ty, offset })?;
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
    #[cfg(feature = "rr-component")]
    cx.store
        .0
        .record_event_validation(|| LowerFlatEntryEvent { ty })?;
    let lower_result = lower(cx, ty);
    #[cfg(feature = "rr-component")]
    cx.store
        .0
        .record_event(|| LowerFlatReturnEvent(ResultEvent::from_anyhow_result(&lower_result)))?;
    lower_result
}

#[cfg(feature = "rr-component")]
#[inline(always)]
fn create_host_func_entry_event(
    args: &mut [MaybeUninit<ValRaw>],
    types: &Arc<ComponentTypes>,
    param_tys: &InterfaceType,
) -> HostFuncEntryEvent {
    let flat_params = types.flat_types_storage_or_pointer(param_tys, MAX_FLAT_PARAMS);
    HostFuncEntryEvent {
        args: RRFuncArgVals::from_flat_storage(args, flat_params),
    }
}

/// Same as [`ConstMemorySliceCell`] except allows for dynamically sized slices.
///
/// Prefer the above for efficiency if slice size is known statically.
///
/// **Note**: The correct operation of this type relies of several invariants.
/// See [`ConstMemorySliceCell`] for detailed description on the role
/// of these types.
pub struct MemorySliceCell<'a> {
    pub bytes: &'a mut [u8],
    #[cfg(feature = "rr-component")]
    pub offset: usize,
    #[cfg(feature = "rr-component")]
    pub recorder: Option<&'a mut RecordBuffer>,
}
impl<'a> Deref for MemorySliceCell<'a> {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        self.bytes
    }
}
impl DerefMut for MemorySliceCell<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.bytes
    }
}
impl Drop for MemorySliceCell<'_> {
    /// Drop serves as a recording hook for stores to the memory slice
    fn drop(&mut self) {
        #[cfg(feature = "rr-component")]
        if let Some(buf) = &mut self.recorder {
            buf.record_event(|| MemorySliceWriteEvent {
                offset: self.offset,
                bytes: self.bytes.to_vec(),
            })
            .unwrap();
        }
    }
}

/// Zero-cost encapsulation type for a statically sized slice of mutable memory
///
/// # Purpose and Usage (Read Carefully!)
///
/// This type (and its dynamic counterpart [`MemorySliceCell`]) are critical to
/// record/replay (RR) support in Wasmtime. In practice, all lowering operations utilize
/// a [`LowerContext`], which provides a capability to modify guest Wasm module state in
/// the following ways:
///
/// 1. Write to slices of memory with [`get`](LowerContext::get)/[`get_dyn`](LowerContext::get_dyn)
/// 2. Movement of memory with [`realloc`](LowerContext::realloc)
///
/// The above are intended to be the narrow waists for recording changes to guest state, and
/// should be the **only** interfaces used during lowerng. In particular,
/// [`get`](LowerContext::get)/[`get_dyn`](LowerContext::get_dyn) return
/// ([`ConstMemorySliceCell`]/[`MemorySliceCell`]), which implement [`Drop`]
/// allowing us a hook to just capture the final aggregate changes made to guest memory by the host.
///
/// ## Critical Invariants
///
/// Typically recording would need to know both when the slice was borrowed AND when it was
/// dropped, since memory movement with [`realloc`](LowerContext::realloc) can be interleaved between
/// borrows and drops, and replays would have to be aware of this. **However**, with this abstraction,
/// we can be more efficient and get away with **only** recording drops, because of the implicit interaction between
/// [`realloc`](LowerContext::realloc) and [`get`](LowerContext::get)/[`get_dyn`](LowerContext::get_dyn),
/// which both take a `&mut self`. Since the latter implements [`Drop`], which also takes a `&mut self`,
/// the compiler will automatically enforce that drops of this type need to be triggered before a
/// [`realloc`](LowerContext::realloc), preventing any interleavings in between the borrow and drop of the slice.
pub struct ConstMemorySliceCell<'a, const N: usize> {
    pub bytes: &'a mut [u8; N],
    #[cfg(feature = "rr-component")]
    pub offset: usize,
    #[cfg(feature = "rr-component")]
    pub recorder: Option<&'a mut RecordBuffer>,
}
impl<'a, const N: usize> Deref for ConstMemorySliceCell<'a, N> {
    type Target = [u8; N];
    fn deref(&self) -> &Self::Target {
        self.bytes
    }
}
impl<'a, const N: usize> DerefMut for ConstMemorySliceCell<'a, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.bytes
    }
}
impl<'a, const N: usize> Drop for ConstMemorySliceCell<'a, N> {
    /// Drops serves as a recording hook for stores to the memory slice
    fn drop(&mut self) {
        #[cfg(feature = "rr-component")]
        if let Some(buf) = &mut self.recorder {
            buf.record_event(|| MemorySliceWriteEvent {
                offset: self.offset,
                bytes: self.bytes.to_vec(),
            })
            .unwrap();
        }
    }
}

use crate::{FuncType, WasmFuncOrigin};
#[cfg(feature = "rr")]
use crate::{StoreContextMut, rr::ReplayHostContext};
#[cfg(feature = "component-model")]
use alloc::sync::Arc;
#[cfg(feature = "component-model")]
use wasmtime_environ::component::{ComponentTypes, TypeFuncIndex};

/// Component specific RR hooks that use `component-model` feature gating
#[cfg(feature = "component-model")]
pub mod component_hooks;
/// Core RR hooks
pub mod core_hooks;

/// Wasm function type information for RR hooks
pub enum RRWasmFuncType<'a> {
    /// Core RR hooks to be performed
    Core {
        ty: &'a FuncType,
        origin: Option<WasmFuncOrigin>,
    },
    /// Component RR hooks to be performed
    #[cfg(feature = "component-model")]
    Component {
        type_idx: TypeFuncIndex,
        types: Arc<ComponentTypes>,
    },
    /// No RR hooks to be performed
    #[cfg(feature = "component-model")]
    None,
}

/// Obtain the replay host context from the store.
///
/// SAFETY: The store's data is always of type `ReplayHostContext` when created by
/// the replay driver. As an additional guarantee, we assert that replay is indeed
/// truly enabled.
#[cfg(feature = "rr")]
unsafe fn replay_data_from_store<'a, T: 'static>(
    store: &StoreContextMut<'a, T>,
) -> &'a ReplayHostContext {
    assert!(store.0.replay_enabled());
    let raw_ptr: *const T = store.data();
    unsafe { &*(raw_ptr as *const ReplayHostContext) }
}

/// Same as [replay_data_from_store], but mutable
///
/// SAFETY: See [replay_data_from_store]
#[cfg(feature = "rr")]
unsafe fn replay_data_from_store_mut<'a, T: 'static>(
    store: &mut StoreContextMut<'a, T>,
) -> &'a mut ReplayHostContext {
    assert!(store.0.replay_enabled());
    let raw_ptr: *mut T = store.data_mut();
    unsafe { &mut *(raw_ptr as *mut ReplayHostContext) }
}

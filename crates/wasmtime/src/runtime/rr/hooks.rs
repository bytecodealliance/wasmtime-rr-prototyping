/// Component specific RR hooks that use `component-model` feature gating
#[cfg(feature = "component-model")]
pub mod component_hooks;
/// Core RR hooks
pub mod core_hooks;

use crate::{FuncType, WasmFuncOrigin};
#[cfg(feature = "component-model")]
use alloc::sync::Arc;
#[cfg(feature = "component-model")]
use wasmtime_environ::component::{ComponentTypes, TypeFuncIndex};

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

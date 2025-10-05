//! Wasmtime's Record and Replay support.
//!
//! This feature is currently not optimized and under development

/// Convenience method hooks for injecting event recording/replaying in the rest of the engine
mod hooks;
pub use hooks::{ConstMemorySliceCell, MemorySliceCell, component_hooks, core_hooks};

#[cfg(feature = "rr")]
/// Core infrastructure for RR support
mod core;
#[cfg(feature = "rr")]
pub use core::*;

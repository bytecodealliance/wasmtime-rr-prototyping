//! Wasmtime's Record and Replay support.
//!
//! This feature is currently not optimized and under development

/// Convenience method hooks for injecting event recording/replaying in the rest of the engine
mod hooks;
pub(crate) use hooks::{ConstMemorySliceCell, MemorySliceCell, component_hooks, core_hooks};

/// Core infrastructure for RR support
#[cfg(feature = "rr")]
mod core;
#[cfg(feature = "rr")]
pub use core::*;

/// Driver capabilities for executing replays
#[cfg(feature = "rr")]
mod replay_driver;
#[cfg(feature = "rr")]
pub use replay_driver::*;

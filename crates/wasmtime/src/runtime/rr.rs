//! Wasmtime's Record and Replay support.
//!
//! This feature is currently not optimized and under development
use crate::ValRaw;
use ::core::{mem::MaybeUninit, slice};

/// Component-async-ABI is not supported for record/replay yet; add a feature gate
//const _: () = {
//    #[cfg(all(feature = "rr", feature = "component-model-async"))]
//    compile_error!(
//        "The `component-model-async` feature is not supported with the `rr` feature yet"
//    );
//};

/// Types that can be serialized/deserialized into/from
/// flat types for record and replay
#[allow(
    unused,
    reason = "trait used as a bound for hooks despite not calling methods directly"
)]
pub trait FlatBytes {
    unsafe fn bytes(&self, size: u8) -> &[u8];
    fn from_bytes(value: &[u8]) -> Self;
}

impl FlatBytes for ValRaw {
    #[inline]
    unsafe fn bytes(&self, size: u8) -> &[u8] {
        &self.get_bytes()[..size as usize]
    }
    #[inline]
    fn from_bytes(value: &[u8]) -> Self {
        ValRaw::from_bytes(value)
    }
}

impl FlatBytes for MaybeUninit<ValRaw> {
    #[inline]
    /// SAFETY: the caller must ensure that 'size' number of bytes provided
    /// are initialized for the underlying ValRaw.
    /// When serializing for record/replay, uninitialized parts of the ValRaw
    /// are not relevant, so this only accesses initialized values as long as
    /// the size contract is upheld.
    unsafe fn bytes(&self, size: u8) -> &[u8] {
        // The cleanest way for this would use MaybeUninit::as_bytes and an assume_init(),
        // but that is currently only available in nightly.
        let ptr = self.as_ptr().cast::<MaybeUninit<u8>>();
        // SAFETY: the caller must ensure that 'size' bytes are initialized
        unsafe {
            let s = slice::from_raw_parts(ptr, size as usize);
            &*(s as *const [MaybeUninit<u8>] as *const [u8])
        }
    }
    #[inline]
    fn from_bytes(value: &[u8]) -> Self {
        MaybeUninit::new(ValRaw::from_bytes(value))
    }
}

/// Convenience method hooks for injecting event recording/replaying in the rest of the engine
mod hooks;
pub(crate) use hooks::{RRWasmFuncType, core_hooks};
#[cfg(feature = "component-model")]
pub(crate) use hooks::{
    component_hooks, component_hooks::DynamicMemorySlice, component_hooks::FixedMemorySlice,
};

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

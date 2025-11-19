//! Wasmtime's Record and Replay support.
//!
//! This feature is currently not optimized and under development
use crate::ValRaw;
use ::core::mem::MaybeUninit;

/// Types that can be serialized/deserialized into/from
/// flat types for record and replay
#[allow(
    unused,
    reason = "trait used as a bound for hooks despite not calling methods directly"
)]
pub trait FlatBytes {
    fn bytes_ref(&self, size: u8) -> &[u8];
    fn from_bytes(value: &[u8]) -> Self;
}

impl FlatBytes for ValRaw {
    #[inline]
    fn bytes_ref(&self, size: u8) -> &[u8] {
        &self.get_bytes()[..size as usize]
    }
    #[inline]
    fn from_bytes(value: &[u8]) -> Self {
        ValRaw::bytes(value)
    }
}

impl FlatBytes for MaybeUninit<ValRaw> {
    #[inline]
    fn bytes_ref(&self, size: u8) -> &[u8] {
        // Uninitialized data is assumed and serialized, so hence
        // may contain some undefined values. But these are irrelevant
        // when serializing to `RRFuncArgVals`
        let val = unsafe { self.assume_init_ref() };
        val.bytes_ref(size)
    }
    #[inline]
    fn from_bytes(value: &[u8]) -> Self {
        MaybeUninit::new(ValRaw::bytes(value))
    }
}

/// Convenience method hooks for injecting event recording/replaying in the rest of the engine
mod hooks;
pub(crate) use hooks::core_hooks;
#[cfg(feature = "component-model")]
pub(crate) use hooks::{
    component_hooks, component_hooks::ConstMemorySliceCell, component_hooks::MemorySliceCell,
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

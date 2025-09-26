//! Convenience methods for hooking in RR event recording/replaying to the rest of the engine
#[cfg(feature = "rr-component")]
use crate::rr::{RecordBuffer, Recorder, component_events::MemorySliceWriteEvent};

use core::ops::{Deref, DerefMut};

/// Component specific RR hooks that use `component-model` feature gating
pub mod component_hooks;
/// Core RR hooks
pub mod core_hooks;

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

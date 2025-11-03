//! Module comprising of event descriptions common to both core wasm and components
//!
//! When using these events, prefer using the re-exported links in [`component_events`]
//! or [`core_events`]

use super::*;
use serde::{Deserialize, Serialize};
use wasmtime_environ::component::FlatTypesStorage;

/// A return event after a host call for a core OR component Wasm
///
/// Matches with either [`component_events::HostFuncEntryEvent`] or
/// [`core_events::HostFuncEntryEvent`]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostFuncReturnEvent {
    /// Raw values passed across the call/return boundary
    args: RRFuncArgVals,
}
impl HostFuncReturnEvent {
    // Record
    pub fn new_from_u8(args: &[MaybeUninit<ValRaw>], flat: &[u8]) -> Self {
        Self {
            args: RRFuncArgVals::from_raw_slice(args, flat.iter().copied()),
        }
    }

    #[cfg(feature = "rr-component")]
    pub fn new_from_flat_storage(args: &[MaybeUninit<ValRaw>], flat: FlatTypesStorage) -> Self {
        Self {
            args: RRFuncArgVals::from_raw_slice(args, flat.iter32()),
        }
    }

    // Replay
    /// Consume the caller event and encode it back into the slice
    pub fn move_into_slice(self, args: &mut [MaybeUninit<ValRaw>]) {
        self.args.into_raw_slice(args);
    }
}

//! Module comprising of event descriptions common to both core wasm and components
//!
//! When using these events, prefer using the re-exported links in [`component_events`]
//! or [`core_events`]

use super::*;
use serde::{Deserialize, Serialize};

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
    pub fn new(args: &[MaybeUninit<ValRaw>]) -> Self {
        Self {
            args: func_argvals_from_raw_slice(args),
        }
    }
    // Replay
    /// Consume the caller event and encode it back into the slice
    pub fn move_into_slice(self, args: &mut [MaybeUninit<ValRaw>]) {
        func_argvals_into_raw_slice(self.args, args);
    }
}

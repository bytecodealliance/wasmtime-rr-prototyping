//! Module comprising of core wasm events
use super::*;
use wasmtime_environ::VMSharedTypeIndex;
// Re-export common events from this module
pub use common_events::*;

/// A call event from a Core Wasm module into the host
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HostFuncEntryEvent {
    /// Raw values passed across the call/return boundary
    args: RRFuncArgVals,
    /// Param/return types (required to support replay validation)
    types: VMSharedTypeIndex,
}
impl HostFuncEntryEvent {
    // Record
    pub fn new(args: &[MaybeUninit<ValRaw>], flat: &[u8], types: VMSharedTypeIndex) -> Self {
        Self {
            args: RRFuncArgVals::from_raw_slice(args, flat.iter().copied()),
            types: types,
        }
    }
}

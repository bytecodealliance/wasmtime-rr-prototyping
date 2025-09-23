//! Module comprising of core wasm events
use super::*;
use wasmtime_environ::VMSharedTypeIndex;
// Re-export common events from this module
pub use common_events::*;

/// A call event from a Core Wasm module into the host
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostFuncEntryEvent {
    /// Raw values passed across the call/return boundary
    args: RRFuncArgVals,
    /// Param/return types (required to support replay validation)
    types: VMSharedTypeIndex,
}
impl HostFuncEntryEvent {
    // Record
    pub fn new(args: &[MaybeUninit<ValRaw>], types: VMSharedTypeIndex) -> Self {
        Self {
            args: func_argvals_from_raw_slice(args),
            types: types,
        }
    }
}
#[cfg(feature = "rr-validate")]
impl Validate<VMSharedTypeIndex> for HostFuncEntryEvent {
    fn validate(&self, expect_types: &VMSharedTypeIndex) -> Result<(), ReplayError> {
        self.log();
        if &self.types == expect_types {
            Ok(())
        } else {
            Err(ReplayError::FailedValidation)
        }
    }
}

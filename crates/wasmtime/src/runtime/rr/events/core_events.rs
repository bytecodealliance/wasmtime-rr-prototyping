//! Module comprising of core wasm events
use super::*;
#[expect(unused_imports, reason = "used for doc-links")]
use wasmtime_environ::{WasmFuncType, WasmValType};
// Re-export common events from this module
pub use common_events::*;

/// Note: Switch [`CoreFuncArgTypes`] to use [`Vec<WasmValType>`] for better efficiency
type CoreFuncArgTypes = WasmFuncType;

/// A call event from a Core Wasm module into the host
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostFuncEntryEvent {
    /// Raw values passed across the call/return boundary
    args: RRFuncArgVals,
    /// Param/return types (required to support replay validation)
    types: CoreFuncArgTypes,
}
impl HostFuncEntryEvent {
    // Record
    pub fn new(args: &[MaybeUninit<ValRaw>], types: WasmFuncType) -> Self {
        Self {
            args: func_argvals_from_raw_slice(args),
            types: types,
        }
    }
}
#[cfg(feature = "rr-validate")]
impl Validate<CoreFuncArgTypes> for HostFuncEntryEvent {
    fn validate(&self, expect_types: &CoreFuncArgTypes) -> Result<(), ReplayError> {
        self.log();
        if &self.types == expect_types {
            Ok(())
        } else {
            Err(ReplayError::FailedValidation)
        }
    }
}

//! Module comprising of core wasm events
use super::*;
use crate::AsContextMut;
use crate::{Val, ValType, WasmFuncOrigin, store::InstanceId};
// Re-export common events from this module
pub use common_events::*;

/// A core Wasm instantiatation event
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Ord, PartialOrd)]
pub struct InstantiationEvent(
    /// Checksum of the bytecode used to instantiate the module
    pub [u8; 32],
    pub InstanceId,
);

/// A call event from Host into a core Wasm function
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WasmFuncEntryEvent {
    /// Checksum of module containing function
    pub module: [u8; 32],
    /// Origin (instance + function index) for this function
    pub origin: WasmFuncOrigin,
    /// Raw values passed across call boundary
    pub args: RRFuncArgVals,
}

impl WasmFuncEntryEvent {
    /// Record
    pub fn new(module: [u8; 32], origin: WasmFuncOrigin, args: &[ValRaw], flat: &[u8]) -> Self {
        Self {
            module,
            origin,
            args: RRFuncArgVals::from_raw_slice(args, flat.iter().copied()),
        }
    }

    // Replay
    /// Consume the caller event and encode it back into the slice
    pub fn to_val_vec(self, store: impl AsContextMut, vals: Vec<ValType>) -> Vec<Val> {
        self.args.to_val_vec(store, vals)
    }
}

/// A call event from a Core Wasm module into the host
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HostFuncEntryEvent {
    /// Raw values passed across the call/return boundary
    args: RRFuncArgVals,
}
impl HostFuncEntryEvent {
    // Record
    pub fn new<T>(args: &[T], flat: &[u8]) -> Self
    where
        T: FlatBytes,
    {
        Self {
            args: RRFuncArgVals::from_raw_slice(args, flat.iter().copied()),
        }
    }
}

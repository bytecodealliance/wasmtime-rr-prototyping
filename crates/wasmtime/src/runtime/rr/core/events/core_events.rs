//! Module comprising of core wasm events
use super::*;
use crate::{WasmFuncOrigin, store::InstanceId};
// Re-export common events from this module
pub use common_events::*;

/// A core Wasm instantiatation event
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Ord, PartialOrd)]
pub struct InstantiationEvent {
    /// Checksum of the bytecode used to instantiate the module
    pub module: [u8; 32],
    /// Instance ID for the instantiated module
    pub instance: InstanceId,
}

/// A call event from Host into a core Wasm function
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WasmFuncEntryEvent {
    /// Origin (instance + function index) for this function
    pub origin: WasmFuncOrigin,
    /// Raw values passed across call boundary
    pub args: RRFuncArgVals,
}

/// A call event from a Core Wasm module into the host
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HostFuncEntryEvent {
    /// Raw values passed across the call/return boundary
    pub args: RRFuncArgVals,
}

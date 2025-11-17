//! Module comprising of core wasm events
use super::*;
use crate::{WasmFuncOrigin, store::InstanceId};

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

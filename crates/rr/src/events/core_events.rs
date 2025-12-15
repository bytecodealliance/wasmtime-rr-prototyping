//! Module comprising of core wasm events
use super::*;
use wasmtime_environ::{EntityIndex, FuncIndex, WasmChecksum};

/// Representation of a Wasm module instance identifier during record/replay.
///
/// This ID is tied to module instantiation events in the trace, and used
/// during replay to refer to specific module instances.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Ord, PartialOrd)]
pub struct RRModuleInstanceId(pub u32);

/// Representation of a Wasm module function index during record/replay.
///
/// This index is used to identify target call functions within a module during replay.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Ord, PartialOrd)]
pub struct RRModuleFuncIndex(pub u32);

impl Into<FuncIndex> for RRModuleFuncIndex {
    fn into(self) -> FuncIndex {
        FuncIndex::from_u32(self.0)
    }
}

impl Into<EntityIndex> for RRModuleFuncIndex {
    fn into(self) -> EntityIndex {
        EntityIndex::from(Into::<FuncIndex>::into(self))
    }
}

impl From<FuncIndex> for RRModuleFuncIndex {
    fn from(f: FuncIndex) -> Self {
        RRModuleFuncIndex(f.as_u32())
    }
}

/// A core Wasm module instantiatation event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Ord, PartialOrd)]
pub struct InstantiationEvent {
    /// Checksum of the bytecode used to instantiate the module.
    pub module: WasmChecksum,
    /// Instance ID for the instantiated module.
    pub instance: RRModuleInstanceId,
}

/// A call event from Host into a core Wasm function.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WasmFuncEntryEvent {
    /// Instance ID for the instantiated module.
    pub instance: RRModuleInstanceId,
    /// Function index of callee within the module.
    pub func_index: RRModuleFuncIndex,
    /// Raw values passed across call boundary
    pub args: RRFuncArgVals,
}

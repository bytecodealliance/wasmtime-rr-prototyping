//! Module comprising of event descriptions common to both core wasm and components
//!
//! When using these events, prefer using the re-exported links in [`component_events`]
//! or [`core_events`]

use super::*;
use serde::{Deserialize, Serialize};

/// A return event after a host call to Wasm (core or component)
///
/// Matches with either [`component_events::HostFuncEntryEvent`] or
/// [`core_events::HostFuncEntryEvent`]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostFuncReturnEvent {
    /// Raw values passed across the call/return boundary
    pub args: RRFuncArgVals,
}

/// A return event from a Wasm (core or component) function to host
///
/// Matches with either [`component_events::WasmFuncEntryEvent`] or
/// [`core_events::WasmFuncEntryEvent`]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmFuncReturnEvent(pub ResultEvent<RRFuncArgVals, WasmFuncReturnError>);

impl Validate<&Result<RRFuncArgVals>> for WasmFuncReturnEvent {
    fn validate(&self, expect: &&Result<RRFuncArgVals>) -> Result<(), ReplayError> {
        self.0.validate(*expect)
    }
}

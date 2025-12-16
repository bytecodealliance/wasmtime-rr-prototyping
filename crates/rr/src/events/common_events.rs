//! Module comprising of event descriptions common to both core wasm and components
//!
//! When using these events, prefer using the re-exported links in [`component_events`]
//! or [`core_events`]

use super::*;
use crate::RecordSettings;
use serde::{Deserialize, Serialize};

/// A call event from Wasm (core or component) into the host
///
/// Matches with [`HostFuncReturnEvent`]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HostFuncEntryEvent {
    /// Raw values passed across the call/return boundary
    pub args: RRFuncArgVals,
}

/// A return event after a host call to Wasm (core or component)
///
/// Matches with [`HostFuncEntryEvent`]
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

event_error_types! {
    pub struct WasmFuncReturnError(..)
}

/// Signature of recorded trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceSignatureEvent {
    /// Checksum of the trace contents.
    ///
    /// This can be used to verify integrity of the trace during replay.
    pub checksum: String,
    /// Settings used during trace recording.
    pub settings: RecordSettings,
}

impl Validate<str> for TraceSignatureEvent {
    fn validate(&self, expect: &str) -> Result<(), ReplayError> {
        self.log();
        if self.checksum == expect {
            Ok(())
        } else {
            Err(ReplayError::FailedValidation)
        }
    }
}

/// A diagnostic event for custom String messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomMessageEvent(pub String);
impl<T> From<T> for CustomMessageEvent
where
    T: Into<String>,
{
    fn from(v: T) -> Self {
        Self(v.into())
    }
}

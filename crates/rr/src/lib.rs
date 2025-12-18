#![no_std]

pub(crate) mod prelude {
    pub use anyhow::{self, Result};
    pub use serde::{Deserialize, Serialize};
    pub use wasmtime_environ::prelude::*;
}

use crate::prelude::*;
pub use core::fmt;
pub use events::{
    EventError, RRFuncArgVals, ResultEvent, Validate, common_events,
    component_events::{self, RRComponentInstanceId},
    core_events::{self, RRModuleInstanceId},
};
pub use io::{RecordWriter, ReplayReader, from_replay_reader, to_record_writer};
// Export necessary environ types for interactions with the crate
pub use wasmtime_environ::{
    EntityIndex, FuncIndex, WasmChecksum,
    component::{ExportIndex, InterfaceType, ResourceDropRet},
};

/// Encapsulation of event types comprising an [`RREvent`] sum type
mod events;
/// I/O support for reading and writing traces
mod io;

/// Settings for execution recording.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordSettings {
    /// Flag to include additional signatures for replay validation.
    pub add_validation: bool,
    /// Maximum window size of internal event buffer.
    pub event_window_size: usize,
}

impl Default for RecordSettings {
    fn default() -> Self {
        Self {
            add_validation: false,
            event_window_size: 16,
        }
    }
}

/// Settings for execution replay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplaySettings {
    /// Flag to include additional signatures for replay validation.
    pub validate: bool,
    /// Static buffer size for deserialization of variable-length types (like [String]).
    pub deserialize_buffer_size: usize,
}

impl Default for ReplaySettings {
    fn default() -> Self {
        Self {
            validate: false,
            deserialize_buffer_size: 64,
        }
    }
}

/// Macro template for [`RREvent`] and its conversion to/from specific
/// event types
macro_rules! rr_event {
    (
        $(
            $(#[doc = $doc_np:literal])*
            $variant_no_payload:ident
        ),*
        ;
        $(
            $(#[doc = $doc:literal])*
            $variant:ident($event:ty)
        ),*
    ) => (
        /// A single, unified, low-level recording/replay event
        ///
        /// This type is the narrow waist for serialization/deserialization.
        /// Higher-level events (e.g. import calls consisting of lifts and lowers
        /// of parameter/return types) may drop down to one or more [`RREvent`]s
        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub enum RREvent {
            $(
                $(#[doc = $doc_np])*
                $variant_no_payload,
            )*
            $(
                $(#[doc = $doc])*
                $variant($event),
            )*
        }

        impl fmt::Display for RREvent {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self {
                    $(
                        Self::$variant_no_payload => write!(f, "{}", stringify!($variant_no_payload)),
                    )*
                    $(
                        Self::$variant(payload) => write!(f, "{:?}", payload),
                    )*
                }
            }
        }

        $(
            impl From<$event> for RREvent {
                fn from(value: $event) -> Self {
                    RREvent::$variant(value)
                }
            }
            impl TryFrom<RREvent> for $event {
                type Error = ReplayError;
                fn try_from(value: RREvent) -> Result<Self, Self::Error> {
                    if let RREvent::$variant(x) = value {
                        Ok(x)
                    } else {
                        log::error!("Expected {}; got {}", stringify!($event), value);
                        Err(ReplayError::IncorrectEventVariant)
                    }
                }
            }
        )*
    );

}

// Set of supported record/replay events
rr_event! {
    /// Nop Event
    Nop,
    /// Event signalling the end of a trace
    Eof
    ;
    /// The signature of the trace, enabling trace integrity during replay.
    ///
    /// This is always at the start of any valid trace.
    TraceSignature(common_events::TraceSignatureEvent),
    /// A custom message in the trace, useful for diagnostics.
    ///
    /// Does not affect trace replay functionality
    CustomMessage(common_events::CustomMessageEvent),

    // Common events for both core or component wasm
    // REQUIRED events
    /// Return from host function (core or component) to host
    HostFuncReturn(common_events::HostFuncReturnEvent),
    // OPTIONAL events
    /// Call into host function from Wasm (core or component)
    HostFuncEntry(common_events::HostFuncEntryEvent),
    /// Return from Wasm function (core or component) to host
    WasmFuncReturn(common_events::WasmFuncReturnEvent),

    // REQUIRED events for replay (Core)
    /// Instantiation of a core Wasm module
    CoreWasmInstantiation(core_events::InstantiationEvent),
    /// Entry from host into a core Wasm function
    CoreWasmFuncEntry(core_events::WasmFuncEntryEvent),

    // REQUIRED events for replay (Component)

    /// Starting marker for a Wasm component function call from host
    ///
    /// This is distinguished from `ComponentWasmFuncEntry` as there may
    /// be multiple lowering steps before actually entering the Wasm function
    ComponentWasmFuncBegin(component_events::WasmFuncBeginEvent),
    /// Entry from the host into the Wasm component function
    ComponentWasmFuncEntry(component_events::WasmFuncEntryEvent),
    /// Instantiation of a component
    ComponentInstantiation(component_events::InstantiationEvent),
    /// Component ABI realloc call in linear wasm memory
    ComponentReallocEntry(component_events::ReallocEntryEvent),
    /// Return from a type lowering operation
    ComponentLowerFlatReturn(component_events::LowerFlatReturnEvent),
    /// Return from a store during a type lowering operation
    ComponentLowerMemoryReturn(component_events::LowerMemoryReturnEvent),
    /// An attempt to obtain a mutable slice into Wasm linear memory
    ComponentMemorySliceWrite(component_events::MemorySliceWriteEvent),
    /// Return from a component builtin
    ComponentBuiltinReturn(component_events::BuiltinReturnEvent),
    /// Call to `post_return` (after the function call)
    ComponentPostReturn(component_events::PostReturnEvent),

    // OPTIONAL events for replay validation (Component)

    /// Return from Component ABI realloc call
    ///
    /// Since realloc is deterministic, ReallocReturn is optional.
    /// Any error is subsumed by the containing LowerReturn/LowerStoreReturn
    /// that triggered realloc
    ComponentReallocReturn(component_events::ReallocReturnEvent),
    /// Call into type lowering for flat destination
    ComponentLowerFlatEntry(component_events::LowerFlatEntryEvent),
    /// Call into type lowering for memory destination
    ComponentLowerMemoryEntry(component_events::LowerMemoryEntryEvent),
    /// Call into a component builtin
    ComponentBuiltinEntry(component_events::BuiltinEntryEvent)
}

impl RREvent {
    /// Indicates whether current event is a diagnostic event
    #[inline]
    pub fn is_diagnostic(&self) -> bool {
        match self {
            Self::Nop | Self::CustomMessage(_) => true,
            _ => false,
        }
    }
}

/// Error type signalling failures during a replay run
#[derive(Debug)]
pub enum ReplayError {
    EmptyBuffer,
    FailedValidation,
    IncorrectEventVariant,
    InvalidEventPosition,
    FailedRead(anyhow::Error),
    EventError(Box<dyn EventError>),
    MissingComponent(WasmChecksum),
    MissingModule(WasmChecksum),
    MissingComponentInstance(RRComponentInstanceId),
    MissingModuleInstance(RRModuleInstanceId),
    InvalidCoreFuncIndex(EntityIndex),
}

impl fmt::Display for ReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyBuffer => {
                write!(f, "replay buffer is empty")
            }
            Self::FailedValidation => {
                write!(
                    f,
                    "failed validation check during replay; see wasmtime log for error"
                )
            }
            Self::IncorrectEventVariant => {
                write!(f, "event type mismatch during replay")
            }
            Self::EventError(e) => {
                write!(f, "{e:?}")
            }
            Self::FailedRead(e) => {
                write!(f, "{e}")?;
                f.write_str("Note: Ensure sufficient `deserialization-buffer-size` in replay settings if you included `validation-metadata` during recording")
            }
            Self::InvalidEventPosition => {
                write!(f, "event occured at an invalid position in the trace")
            }
            Self::MissingComponent(checksum) => {
                write!(
                    f,
                    "missing component binary with checksum 0x{} during replay",
                    checksum
                        .iter()
                        .map(|b| format!("{b:02x}"))
                        .collect::<String>()
                )
            }
            Self::MissingModule(checksum) => {
                write!(
                    f,
                    "missing module binary with checksum {:02x?} during replay",
                    checksum
                        .iter()
                        .map(|b| format!("{b:02x}"))
                        .collect::<String>()
                )
            }
            Self::MissingComponentInstance(id) => {
                write!(f, "missing component instance ID {id:?} during replay")
            }
            Self::MissingModuleInstance(id) => {
                write!(f, "missing module instance ID {id:?} during replay")
            }
            Self::InvalidCoreFuncIndex(index) => {
                write!(f, "replay core func ({index:?}) during replay is invalid")
            }
        }
    }
}

impl core::error::Error for ReplayError {}

impl<T: EventError> From<T> for ReplayError {
    fn from(value: T) -> Self {
        Self::EventError(Box::new(value))
    }
}

/// This trait provides the interface for a FIFO recorder
pub trait Recorder {
    /// Construct a recorder with the writer backend
    fn new_recorder(writer: impl RecordWriter, settings: RecordSettings) -> Result<Self>
    where
        Self: Sized;

    /// Record the event generated by `f`
    ///
    /// ## Error
    ///
    /// Propogates from underlying writer
    fn record_event<T, F>(&mut self, f: F) -> Result<()>
    where
        T: Into<RREvent>,
        F: FnOnce() -> T;

    /// Consumes this [`Recorder`] and returns its underlying writer
    fn into_writer(self) -> Result<Box<dyn RecordWriter>>;

    /// Trigger an explicit flush of any buffered data to the writer
    ///
    /// Buffer should be emptied during this process
    fn flush(&mut self) -> Result<()>;

    /// Get settings associated with the recording process
    fn settings(&self) -> &RecordSettings;

    // Provided methods

    /// Record a event only when validation is requested
    #[inline]
    fn record_event_validation<T, F>(&mut self, f: F) -> Result<()>
    where
        T: Into<RREvent>,
        F: FnOnce() -> T,
    {
        let settings = self.settings();
        if settings.add_validation {
            self.record_event(f)?;
        }
        Ok(())
    }
}

/// This trait provides the interface for a FIFO replayer that
/// essentially operates as an iterator over the recorded events
pub trait Replayer: Iterator<Item = Result<RREvent, ReplayError>> {
    /// Constructs a reader on buffer
    fn new_replayer(reader: impl ReplayReader + 'static, settings: ReplaySettings) -> Result<Self>
    where
        Self: Sized;

    /// Get settings associated with the replay process
    fn settings(&self) -> &ReplaySettings;

    /// Get the settings (embedded within the trace) during recording
    fn trace_settings(&self) -> &RecordSettings;

    // Provided Methods

    /// Get the next functional replay event (skips past all non-marker events)
    #[inline]
    fn next_event(&mut self) -> Result<RREvent, ReplayError> {
        self.next().ok_or(ReplayError::EmptyBuffer)?
    }

    /// Pop the next replay event with an attemped type conversion to expected
    /// event type
    ///
    /// ## Errors
    ///
    /// Returns a  [`ReplayError::IncorrectEventVariant`] if it failed to convert typecheck event safely
    #[inline]
    fn next_event_typed<T>(&mut self) -> Result<T, ReplayError>
    where
        T: TryFrom<RREvent>,
        ReplayError: From<<T as TryFrom<RREvent>>::Error>,
    {
        T::try_from(self.next_event()?).map_err(|e| e.into())
    }

    /// Conditionally process the next validation recorded event and if
    /// replay validation is enabled, run the validation check
    ///
    /// ## Errors
    ///
    /// In addition to errors in [`next_event_typed`](Replayer::next_event_typed),
    /// validation errors can be thrown
    #[inline]
    fn next_event_validation<T, Y>(&mut self, expect: &Y) -> Result<(), ReplayError>
    where
        T: TryFrom<RREvent> + Validate<Y>,
        ReplayError: From<<T as TryFrom<RREvent>>::Error>,
    {
        if self.trace_settings().add_validation {
            let event = self.next_event_typed::<T>()?;
            if self.settings().validate {
                event.validate(expect)
            } else {
                Ok(())
            }
        } else {
            Ok(())
        }
    }
}

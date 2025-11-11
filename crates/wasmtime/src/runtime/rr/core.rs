use crate::config::ModuleVersionStrategy;
use crate::prelude::*;
use core::fmt;
use events::EventError;
use serde::{Deserialize, Serialize};
// Use component events internally even without feature flags enabled
// so that [`RREvent`] has a well-defined serialization format, but export
// it for other modules only when enabled
#[cfg(all(feature = "rr-validate", feature = "rr-component"))]
pub use events::RRFuncArgVals;
#[cfg(any(feature = "rr-validate", feature = "rr-component"))]
pub use events::Validate;
#[cfg(feature = "rr-component")]
pub use events::component_events;
use events::{common_events, component_events as __component_events};
pub use events::{core_events, marker_events};
pub use io::{IOError, RecordWriter, ReplayReader};

/// Settings for execution recording.
#[cfg(feature = "rr")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordSettings {
    /// Flag to include additional signatures for replay validation.
    pub add_validation: bool,
    /// Maximum window size of internal event buffer.
    pub event_window_size: usize,
}

#[cfg(feature = "rr")]
impl Default for RecordSettings {
    fn default() -> Self {
        Self {
            add_validation: false,
            event_window_size: 16,
        }
    }
}

/// Settings for execution replay.
#[cfg(feature = "rr")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplaySettings {
    /// Flag to include additional signatures for replay validation.
    pub validate: bool,
    /// Static buffer size for deserialization of variable-length types (like [String]).
    pub deser_buffer_size: usize,
}

#[cfg(feature = "rr")]
impl Default for ReplaySettings {
    fn default() -> Self {
        Self {
            validate: false,
            deser_buffer_size: 64,
        }
    }
}

/// Encapsulation of event types comprising an [`RREvent`] sum type
mod events;
/// I/O support for reading and writing traces
mod io;

/// Macro template for [`RREvent`] and its conversion to/from specific
/// event types
macro_rules! rr_event {
        (
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
            /// Event signalling the end of a trace
            Eof,
            $(
                $(#[doc = $doc])*
                $variant($event),
            )*
        }

        impl fmt::Display for RREvent {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self {
                    Self::Eof => write!(f, "Eof event"),
                    $(
                    Self::$variant(e) => write!(f, "{:?}", e),
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
    // Marker events
    /// Nop Event
    Nop(marker_events::NopEvent),
    /// A custom message
    CustomMessage(marker_events::CustomMessageEvent),

    // Common events for both core or component wasm
    /// Return from host function to either core Wasm or component
    HostFuncReturn(common_events::HostFuncReturnEvent),

    /// Call into host function from core Wasm
    CoreHostFuncEntry(core_events::HostFuncEntryEvent),

    // REQUIRED events for replay

    /// Starting marker for a Wasm component function call from host
    ///
    /// This is distinguished from `ComponentWasmFuncEntry` as there may
    /// be multiple lowering steps before actually entering the Wasm function
    ComponentWasmFuncBegin(__component_events::WasmFuncBeginEvent),
    /// Entry from the host into the Wasm component function
    ComponentWasmFuncEntry(__component_events::WasmFuncEntryEvent),
    /// Instantiation of a component
    ComponentInstantiation(__component_events::InstantiationEvent),
    /// Component ABI realloc call in linear wasm memory
    ComponentReallocEntry(__component_events::ReallocEntryEvent),
    /// Return from a type lowering operation
    ComponentLowerFlatReturn(__component_events::LowerFlatReturnEvent),
    /// Return from a store during a type lowering operation
    ComponentLowerMemoryReturn(__component_events::LowerMemoryReturnEvent),
    /// An attempt to obtain a mutable slice into Wasm linear memory
    ComponentMemorySliceWrite(__component_events::MemorySliceWriteEvent),
    /// Return from a component builtin
    ComponentBuiltinReturn(__component_events::BuiltinReturnEvent),

    // OPTIONAL events for replay validation

    /// Return from a Wasm component function back to host
    ComponentWasmFuncReturn(__component_events::WasmFuncReturnEvent),
    /// Return from Component ABI realloc call
    ///
    /// Since realloc is deterministic, ReallocReturn is optional.
    /// Any error is subsumed by the containing LowerReturn/LowerStoreReturn
    /// that triggered realloc
    ComponentReallocReturn(__component_events::ReallocReturnEvent),
    /// Call into host function from component
    ComponentHostFuncEntry(__component_events::HostFuncEntryEvent),
    /// Call into type lowering for flat destination
    ComponentLowerFlatEntry(__component_events::LowerFlatEntryEvent),
    /// Call into type lowering for memory destination
    ComponentLowerMemoryEntry(__component_events::LowerMemoryEntryEvent),
    /// Call into a component builtin
    ComponentBuiltinEntry(__component_events::BuiltinEntryEvent)
}

impl RREvent {
    /// Indicates whether current event is a marker event
    #[inline]
    fn is_marker(&self) -> bool {
        match self {
            Self::Nop(_) | Self::CustomMessage(_) => true,
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
    InvalidOrdering,
    FailedRead(IOError),
    EventError(Box<dyn EventError>),
    MissingComponentOrModule,
    MissingComponentOrModuleInstance,
}

impl fmt::Display for ReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyBuffer => {
                write!(f, "replay buffer is empty")
            }
            Self::FailedValidation => {
                write!(f, "replay event validation failed")
            }
            Self::IncorrectEventVariant => {
                write!(f, "event method invoked on incorrect variant")
            }
            Self::EventError(e) => {
                write!(f, "{:?}", e)
            }
            Self::FailedRead(e) => {
                write!(f, "{}", e)?;
                f.write_str("Note: Ensure sufficient `deserialization-buffer-size` in replay settings if you included `validation-metadata` during recording")
            }
            Self::InvalidOrdering => {
                write!(f, "event occured at an invalid position in the trace")
            }
            Self::MissingComponentOrModule => {
                write!(f, "missing component or module for replay")
            }
            Self::MissingComponentOrModuleInstance => {
                write!(f, "missing component or module instance for replay")
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
    fn new_recorder(writer: impl RecordWriter + 'static, settings: RecordSettings) -> Result<Self>
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

    /// Trigger an explicit flush of any buffered data to the writer
    ///
    /// Buffer should be emptied during this process
    fn flush(&mut self) -> Result<()>;

    /// Get settings associated with the recording process
    fn settings(&self) -> &RecordSettings;

    // Provided methods

    /// Record a event only when validation is requested
    #[inline]
    #[cfg(feature = "rr-validate")]
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

    ///// Peek at the next event without consuming it
    //fn peek(&mut self) -> Option<Result<&RREvent, ReplayError>>;

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

    /// Pop the next replay event and calls `f` with a desired type conversion
    ///
    /// ## Errors
    ///
    /// See [`next_event_typed`](Replayer::next_event_typed)
    #[inline]
    fn next_event_and<T, F>(&mut self, f: F) -> Result<(), ReplayError>
    where
        T: TryFrom<RREvent>,
        ReplayError: From<<T as TryFrom<RREvent>>::Error>,
        F: FnOnce(T) -> Result<(), ReplayError>,
    {
        let call_event = self.next_event_typed()?;
        Ok(f(call_event)?)
    }

    /// Conditionally process the next validation recorded event and if
    /// replay validation is enabled, run the validation check
    ///
    /// ## Errors
    ///
    /// In addition to errors in [`next_event_typed`](Replayer::next_event_typed),
    /// validation errors can be thrown
    #[inline]
    #[cfg(feature = "rr-validate")]
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

/// Buffer to write recording data.
///
/// This type can be optimized for [`RREvent`] data configurations.
pub struct RecordBuffer {
    /// In-memory event buffer to enable windows for coalescing
    buf: Vec<RREvent>,
    /// Writer to store data into
    writer: Box<dyn RecordWriter>,
    /// Settings in record configuration
    settings: RecordSettings,
}

impl RecordBuffer {
    /// Push a new record event [`RREvent`] to the buffer
    fn push_event(&mut self, event: RREvent) -> Result<()> {
        self.buf.push(event);
        if self.buf.len() >= self.settings().event_window_size {
            self.flush()?;
        }
        Ok(())
    }
}

impl Drop for RecordBuffer {
    fn drop(&mut self) {
        // Insert End of trace delimiter
        self.push_event(RREvent::Eof).unwrap();
        self.flush().unwrap();
    }
}

impl Recorder for RecordBuffer {
    fn new_recorder(
        mut writer: impl RecordWriter + 'static,
        settings: RecordSettings,
    ) -> Result<Self> {
        // Replay requires the Module version and record settings
        io::to_record_writer(ModuleVersionStrategy::WasmtimeVersion.as_str(), &mut writer)?;
        io::to_record_writer(&settings, &mut writer)?;
        Ok(RecordBuffer {
            buf: Vec::new(),
            writer: Box::new(writer),
            settings: settings,
        })
    }

    #[inline]
    fn record_event<T, F>(&mut self, f: F) -> Result<()>
    where
        T: Into<RREvent>,
        F: FnOnce() -> T,
    {
        let event = f().into();
        log::debug!("Recording event => {}", &event);
        self.push_event(event)
    }

    fn flush(&mut self) -> Result<()> {
        log::debug!("Flushing record buffer...");
        for e in self.buf.drain(..) {
            io::to_record_writer(&e, &mut self.writer)?;
        }
        return Ok(());
    }

    #[inline]
    fn settings(&self) -> &RecordSettings {
        &self.settings
    }
}

/// Buffer to read replay data
pub struct ReplayBuffer {
    /// Reader to read replay trace from
    reader: Box<dyn ReplayReader>,
    /// Settings in replay configuration
    settings: ReplaySettings,
    /// Settings for record configuration (encoded in the trace)
    trace_settings: RecordSettings,
    /// Intermediate static buffer for deserialization
    deser_buffer: Vec<u8>,
    /// Whether buffer has been completely read
    eof_encountered: bool,
    /// Peeked event for lookahead
    peeked: Option<RREvent>,
}

impl Iterator for ReplayBuffer {
    type Item = Result<RREvent, ReplayError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.eof_encountered {
            return None;
        }
        if self.peeked.is_some() {
            return self.peeked.take().map(Ok);
        }
        let ret = 'event_loop: loop {
            let result = io::from_replay_reader(&mut self.reader, &mut self.deser_buffer);
            match result {
                Err(e) => {
                    break 'event_loop Some(Err(ReplayError::FailedRead(e)));
                }
                Ok(event) => {
                    if let RREvent::Eof = &event {
                        break 'event_loop None;
                    } else if event.is_marker() {
                        continue 'event_loop;
                    } else {
                        log::debug!("Read replay event => {}", event);
                        break 'event_loop Some(Ok(event));
                    }
                }
            }
        };
        ret
    }
}

impl Drop for ReplayBuffer {
    fn drop(&mut self) {
        let mut remaining_events = 0;
        // Cannot use count() in iterator because IO error may loop indefinitely
        while let Some(event) = self.next() {
            if let Ok(e) = event {
                println!("Remaining event: {:?}", e);
            } else {
                event.unwrap();
            };
            remaining_events += 1;
        }
        if remaining_events > 0 {
            log::warn!(
                "Replay buffer is dropped with {} remaining events, 
                and is likely an invalid/incomplete execution",
                remaining_events
            );
        }
    }
}

impl Replayer for ReplayBuffer {
    fn new_replayer(
        mut reader: impl ReplayReader + 'static,
        settings: ReplaySettings,
    ) -> Result<Self> {
        let mut scratch = [0u8; 12];
        // Ensure module versions match
        let version = io::from_replay_reader::<&str, _>(&mut reader, &mut scratch)?;
        assert_eq!(
            version,
            ModuleVersionStrategy::WasmtimeVersion.as_str(),
            "Wasmtime version mismatch between engine used for record and replay"
        );

        // Read the recording settings
        let trace_settings: RecordSettings = io::from_replay_reader(&mut reader, &mut scratch)?;

        if settings.validate && !trace_settings.add_validation {
            log::warn!(
                "Replay validation will be omitted since the recorded trace has no validation metadata..."
            );
        }

        let deser_buffer = vec![0; settings.deser_buffer_size];
        let reader = Box::new(reader);

        Ok(ReplayBuffer {
            reader,
            settings,
            trace_settings,
            deser_buffer,
            eof_encountered: false,
            peeked: None,
        })
    }

    #[inline]
    fn settings(&self) -> &ReplaySettings {
        &self.settings
    }

    #[inline]
    fn trace_settings(&self) -> &RecordSettings {
        &self.trace_settings
    }

    //#[inline]
    //fn peek(&mut self) -> Option<Result<&RREvent, ReplayError>> {
    //    if self.peeked.is_none() {
    //        self.peeked = self.next();
    //    }
    //    self.peeked.as_ref().map(|r| Ok(r))
    //}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ValRaw;
    use core::mem::MaybeUninit;
    use std::fs::File;
    use std::path::Path;
    use tempfile::{NamedTempFile, TempPath};

    #[test]
    #[cfg(all(feature = "rr", feature = "rr-component"))]
    fn rr_buffers() -> Result<()> {
        use wasmtime_environ::component::FlatTypesStorage;

        let record_settings = RecordSettings::default();
        let tmp = NamedTempFile::new()?;
        let tmppath = tmp.path().to_str().expect("Filename should be UTF-8");

        let values = vec![ValRaw::i32(1), ValRaw::f32(2), ValRaw::i64(3)];
        let flat = FlatTypesStorage::new();
        flat.push(FlatType::I32, FlatType::I32);
        flat.push(FlatType::F32, FlatType::F32);
        flat.push(FlatType::I64, FlatType::I64);

        // Record values
        let mut recorder =
            RecordBuffer::new_recorder(Box::new(File::create(tmppath)?), record_settings)?;
        recorder.record_event(|| {
            __component_events::HostFuncReturnEvent::new(values.as_slice(), flat)
        })?;
        recorder.flush()?;

        let tmp = tmp.into_temp_path();
        let tmppath = <TempPath as AsRef<Path>>::as_ref(&tmp)
            .to_str()
            .expect("Filename should be UTF-8");
        let replay_settings = ReplaySettings::default();

        // Assert that replayed values are identical
        let mut replayer =
            ReplayBuffer::new_replayer(Box::new(File::open(tmppath)?), replay_settings)?;
        let mut result_values = values.clone();
        replayer.next_event_and(|event: __component_events::HostFuncReturnEvent| {
            event.move_into_slice(result_values.as_mut_slice());

            // Check replay `values` matches record `values`
            for (a, b) in values.iter().zip(result_values.iter()) {
                unsafe {
                    assert!(a.assume_init().as_bytes() == b.assume_init().as_bytes());
                }
            }
            Ok(())
        })?;

        // Check queue is empty
        assert!(replayer.next().is_none());

        Ok(())
    }
}

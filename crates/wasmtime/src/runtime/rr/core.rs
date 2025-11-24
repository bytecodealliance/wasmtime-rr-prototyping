use crate::config::ModuleVersionStrategy;
use crate::prelude::*;
use core::fmt;
use events::EventError;
use serde::{Deserialize, Serialize};
use wasmtime_environ::EntityIndex;
// Use component events internally even without feature flags enabled
// so that [`RREvent`] has a well-defined serialization format, but export
// it for other modules only when enabled
pub use events::Validate;
#[cfg(feature = "rr-component")]
pub use events::component_events;
use events::component_events as __component_events;
pub use events::{RRFuncArgVals, ResultEvent, common_events, core_events, marker_events};
pub use io::{IOError, RecordWriter, ReplayReader};

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
    pub deser_buffer_size: usize,
}

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
    /// Call to `post_return` (after the function call)
    ComponentPostReturn(__component_events::PostReturnEvent),

    // OPTIONAL events for replay validation (Component)

    /// Return from Component ABI realloc call
    ///
    /// Since realloc is deterministic, ReallocReturn is optional.
    /// Any error is subsumed by the containing LowerReturn/LowerStoreReturn
    /// that triggered realloc
    ComponentReallocReturn(__component_events::ReallocReturnEvent),
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
    InvalidEventPosition,
    FailedRead(IOError),
    EventError(Box<dyn EventError>),
    MissingComponent([u8; 32]),
    MissingModule([u8; 32]),
    MissingComponentInstance(u32),
    MissingModuleInstance(u32),
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
                write!(f, "{:?}", e)
            }
            Self::FailedRead(e) => {
                write!(f, "{}", e)?;
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
                        .map(|b| format!("{:02x}", b))
                        .collect::<String>()
                )
            }
            Self::MissingModule(checksum) => {
                write!(
                    f,
                    "missing module binary with checksum {:02x?} during replay",
                    checksum
                        .iter()
                        .map(|b| format!("{:02x}", b))
                        .collect::<String>()
                )
            }
            Self::MissingComponentInstance(id) => {
                write!(f, "missing component instance ID {:?} during replay", id)
            }
            Self::MissingModuleInstance(id) => {
                write!(f, "missing module instance ID {:?} during replay", id)
            }
            Self::InvalidCoreFuncIndex(index) => {
                write!(f, "replay core func ({:?}) during replay is invalid", index)
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

    /// End the trace and flush any remaining data
    pub fn finish(&mut self) -> Result<()> {
        // Insert End of trace delimiter
        self.push_event(RREvent::Eof)?;
        self.flush()
    }
}

impl Recorder for RecordBuffer {
    fn new_recorder(mut writer: impl RecordWriter, settings: RecordSettings) -> Result<Self> {
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

    #[inline]
    fn into_writer(mut self) -> Result<Box<dyn RecordWriter>> {
        self.finish()?;
        Ok(self.writer)
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
}

impl Iterator for ReplayBuffer {
    type Item = Result<RREvent, ReplayError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.eof_encountered {
            return None;
        }
        let ret = 'event_loop: loop {
            let result = io::from_replay_reader(&mut self.reader, &mut self.deser_buffer);
            match result {
                Err(e) => {
                    break 'event_loop Some(Err(ReplayError::FailedRead(e)));
                }
                Ok(event) => {
                    if let RREvent::Eof = &event {
                        self.eof_encountered = true;
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
        let mut remaining = false;
        log::debug!("Replay buffer is being dropped; checking for remaining replay events...");
        // Cannot use count() in iterator because IO error may loop indefinitely
        while let Some(e) = self.next() {
            e.unwrap();
            remaining = true;
            break;
        }
        if remaining {
            log::warn!(
                "Some events were not used in the replay buffer. This is likely the result of an erroneous/incomplete execution",
            );
        } else {
            log::debug!("All replay events were successfully processed.");
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
        })
    }

    #[inline]
    #[allow(
        unused,
        reason = "method only used for gated validation, but will be extended in the future"
    )]
    fn settings(&self) -> &ReplaySettings {
        &self.settings
    }

    #[inline]
    #[allow(
        unused,
        reason = "method only used for gated validation, but will be extended in the future"
    )]
    fn trace_settings(&self) -> &RecordSettings {
        &self.trace_settings
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ValRaw;
    use crate::WasmFuncOrigin;
    use crate::store::InstanceId;
    use crate::vm::component::libcalls::ResourceDropRet;
    use std::fs::File;
    use std::path::Path;
    use tempfile::{NamedTempFile, TempPath};
    use wasmtime_environ::FuncIndex;

    fn rr_harness<S, T>(record_fn: S, replay_fn: T) -> Result<()>
    where
        S: FnOnce(&mut RecordBuffer) -> Result<()>,
        T: FnOnce(&mut ReplayBuffer) -> Result<()>,
    {
        // Record information
        let record_settings = RecordSettings::default();
        let tmp = NamedTempFile::new()?;
        let tmppath = tmp.path().to_str().expect("Filename should be UTF-8");

        // Record values
        let mut recorder =
            RecordBuffer::new_recorder(Box::new(File::create(tmppath)?), record_settings)?;

        record_fn(&mut recorder)?;
        recorder.finish()?;

        let tmp = tmp.into_temp_path();
        let tmppath = <TempPath as AsRef<Path>>::as_ref(&tmp)
            .to_str()
            .expect("Filename should be UTF-8");
        let replay_settings = ReplaySettings::default();

        // Assert that replayed values are identical
        let mut replayer =
            ReplayBuffer::new_replayer(Box::new(File::open(tmppath)?), replay_settings)?;

        replay_fn(&mut replayer)?;

        // Check queue is empty
        assert!(replayer.next().is_none());
        Ok(())
    }

    fn verify_equal_slices(
        record_vals: &[ValRaw],
        replay_vals: &[ValRaw],
        flat_sizes: &[u8],
    ) -> Result<()> {
        for ((a, b), sz) in record_vals
            .iter()
            .zip(replay_vals.iter())
            .zip(flat_sizes.iter())
        {
            let a_slice: &[u8] = &a.get_bytes()[..*sz as usize];
            let b_slice: &[u8] = &b.get_bytes()[..*sz as usize];
            assert!(
                a_slice == b_slice,
                "Recorded values {:?} and replayed values {:?} do not match",
                a_slice,
                b_slice
            );
        }
        Ok(())
    }

    #[test]
    fn host_func() -> Result<()> {
        let values = vec![ValRaw::f64(20), ValRaw::i32(10), ValRaw::i64(30)];
        let flat_sizes: Vec<u8> = vec![8, 4, 8];

        let return_values = vec![ValRaw::i32(1), ValRaw::f32(2), ValRaw::i64(3)];
        let return_flat_sizes: Vec<u8> = vec![4, 4, 8];
        let mut return_replay_values = values.clone();

        rr_harness(
            |recorder| {
                recorder.record_event(|| common_events::HostFuncEntryEvent {
                    args: RRFuncArgVals::from_flat_iter(&values, flat_sizes.iter().copied()),
                })?;
                recorder.record_event(|| common_events::HostFuncReturnEvent {
                    args: RRFuncArgVals::from_flat_iter(
                        &return_values,
                        return_flat_sizes.iter().copied(),
                    ),
                })
            },
            |replayer| {
                replayer.next_event_and(|event: common_events::HostFuncEntryEvent| {
                    event.validate(&common_events::HostFuncEntryEvent {
                        args: RRFuncArgVals::from_flat_iter(&values, flat_sizes.iter().copied()),
                    })
                })?;
                replayer.next_event_and(|event: common_events::HostFuncReturnEvent| {
                    event.args.into_raw_slice(&mut return_replay_values);
                    Ok(())
                })?;
                verify_equal_slices(&return_values, &return_replay_values, &return_flat_sizes)
            },
        )
    }

    #[test]
    fn wasm_func_entry() -> Result<()> {
        let values = vec![ValRaw::i32(42), ValRaw::f64(314), ValRaw::i64(84)];
        let flat_sizes: Vec<u8> = vec![4, 8, 8];
        let origin = WasmFuncOrigin {
            instance: InstanceId::from_u32(15),
            index: FuncIndex::from_u32(7),
        };
        let mut replay_values = values.clone();
        let mut replay_origin = None;

        let return_values = vec![ValRaw::f32(7), ValRaw::f32(8), ValRaw::v128(21)];
        let return_flat_sizes: Vec<u8> = vec![4, 4, 16];
        let mut return_replay_values = values.clone();

        rr_harness(
            |recorder| {
                recorder.record_event(|| core_events::WasmFuncEntryEvent {
                    origin: origin.clone(),
                    args: RRFuncArgVals::from_flat_iter(&values, flat_sizes.iter().copied()),
                })?;
                recorder.record_event(|| __component_events::WasmFuncEntryEvent {
                    args: RRFuncArgVals::from_flat_iter(
                        &return_values,
                        return_flat_sizes.iter().copied(),
                    ),
                })
            },
            |replayer| {
                replayer.next_event_and(|event: core_events::WasmFuncEntryEvent| {
                    replay_origin = Some(event.origin);
                    event.args.into_raw_slice(&mut replay_values);
                    Ok(())
                })?;
                assert!(origin == replay_origin.unwrap());
                verify_equal_slices(&values, &replay_values, &flat_sizes)?;

                replayer.next_event_and(|event: __component_events::WasmFuncEntryEvent| {
                    event.args.into_raw_slice(&mut return_replay_values);
                    Ok(())
                })?;
                verify_equal_slices(&return_values, &return_replay_values, &return_flat_sizes)
            },
        )
    }

    #[test]
    fn builtin_event_entry() -> Result<()> {
        use __component_events::{
            BuiltinEntryEvent, ResourceDropEntryEvent, ResourceEnterCallEntryEvent,
            ResourceExitCallEntryEvent, ResourceTransferBorrowEntryEvent,
            ResourceTransferOwnEntryEvent,
        };
        let events: Vec<BuiltinEntryEvent> = vec![
            BuiltinEntryEvent::ResourceDrop(ResourceDropEntryEvent {
                caller_instance: 3,
                resource: 42,
                idx: 10,
            }),
            BuiltinEntryEvent::ResourceTransferOwn(ResourceTransferOwnEntryEvent {
                src_idx: 5,
                src_table: 1,
                dst_table: 2,
            }),
            BuiltinEntryEvent::ResourceTransferBorrow(ResourceTransferBorrowEntryEvent {
                src_idx: 7,
                src_table: 3,
                dst_table: 4,
            }),
            BuiltinEntryEvent::ResourceEnterCall(ResourceEnterCallEntryEvent {}),
            BuiltinEntryEvent::ResourceExitCall(ResourceExitCallEntryEvent {}),
        ];

        rr_harness(
            |recorder| {
                for event in &events {
                    recorder.record_event(|| event.clone())?;
                }
                Ok(())
            },
            |replayer| {
                for event in &events {
                    replayer.next_event_and(|replay_event: BuiltinEntryEvent| {
                        assert!(*event == replay_event);
                        Ok(())
                    })?;
                }
                Ok(())
            },
        )
    }

    #[test]
    fn builtin_event_return() -> Result<()> {
        use __component_events::{
            BuiltinError, BuiltinReturnEvent, ResourceDropReturnEvent, ResourceExitCallReturnEvent,
            ResourceRep32ReturnEvent, ResourceTransferBorrowReturnEvent,
            ResourceTransferOwnReturnEvent,
        };
        let events: Vec<BuiltinReturnEvent> = vec![
            BuiltinReturnEvent::ResourceDrop(ResourceDropReturnEvent(
                ResultEvent::from_anyhow_result(&Ok(ResourceDropRet::default())),
            )),
            BuiltinReturnEvent::ResourceRep32(ResourceRep32ReturnEvent(
                ResultEvent::from_anyhow_result(&Ok(123)),
            )),
            BuiltinReturnEvent::ResourceTransferOwn(ResourceTransferOwnReturnEvent(
                ResultEvent::from_anyhow_result(&Ok(42)),
            )),
            BuiltinReturnEvent::ResourceTransferBorrow(ResourceTransferBorrowReturnEvent(
                ResultEvent::from_anyhow_result(&Ok(17)),
            )),
            BuiltinReturnEvent::ResourceExitCall(ResourceExitCallReturnEvent(
                ResultEvent::from_anyhow_result(&Err(anyhow::anyhow!("Exit call failed!"))),
            )),
        ];

        rr_harness(
            |recorder| {
                for event in &events {
                    recorder.record_event(|| event.clone())?;
                }
                Ok(())
            },
            |replayer| {
                for event in &events {
                    replayer.next_event_and(|replay_event: BuiltinReturnEvent| {
                        match (replay_event, event) {
                            (
                                BuiltinReturnEvent::ResourceDrop(e),
                                BuiltinReturnEvent::ResourceDrop(expected),
                            ) => {
                                assert_eq!(e.ret().unwrap(), expected.clone().ret().unwrap());
                            }
                            (
                                BuiltinReturnEvent::ResourceRep32(e),
                                BuiltinReturnEvent::ResourceRep32(expected),
                            ) => {
                                assert_eq!(e.ret().unwrap(), expected.clone().ret().unwrap());
                            }
                            (
                                BuiltinReturnEvent::ResourceTransferOwn(e),
                                BuiltinReturnEvent::ResourceTransferOwn(expected),
                            ) => {
                                assert_eq!(e.ret().unwrap(), expected.clone().ret().unwrap());
                            }
                            (
                                BuiltinReturnEvent::ResourceTransferBorrow(e),
                                BuiltinReturnEvent::ResourceTransferBorrow(expected),
                            ) => {
                                assert_eq!(e.ret().unwrap(), expected.clone().ret().unwrap());
                            }
                            (
                                BuiltinReturnEvent::ResourceExitCall(e),
                                BuiltinReturnEvent::ResourceExitCall(expected),
                            ) => {
                                assert_eq!(
                                    e.ret()
                                        .unwrap_err()
                                        .downcast_ref::<BuiltinError>()
                                        .unwrap()
                                        .get(),
                                    expected
                                        .clone()
                                        .ret()
                                        .unwrap_err()
                                        .downcast_ref::<BuiltinError>()
                                        .unwrap()
                                        .get()
                                );
                            }
                            _ => unreachable!(),
                        };
                        Ok(())
                    })?;
                }
                Ok(())
            },
        )
    }

    #[test]
    fn lower_flat_events() -> Result<()> {
        use __component_events::{LowerFlatEntryEvent, LowerFlatReturnEvent};
        use wasmtime_environ::component::InterfaceType;

        let entry = LowerFlatEntryEvent {
            ty: InterfaceType::U32,
        };
        let return_event = LowerFlatReturnEvent(ResultEvent::from_anyhow_result(&Ok(())));

        rr_harness(
            |recorder| {
                recorder.record_event(|| entry.clone())?;
                recorder.record_event(|| return_event.clone())?;
                Ok(())
            },
            |replayer| {
                replayer.next_event_and(|e: LowerFlatEntryEvent| {
                    assert_eq!(e.ty, InterfaceType::U32);
                    Ok(())
                })?;
                replayer.next_event_and(|e: LowerFlatReturnEvent| {
                    assert!(e.0.ret().is_ok());
                    Ok(())
                })?;
                Ok(())
            },
        )
    }

    #[test]
    fn lower_memory_events() -> Result<()> {
        use __component_events::{LowerMemoryEntryEvent, LowerMemoryReturnEvent};
        use wasmtime_environ::component::InterfaceType;

        let entry = LowerMemoryEntryEvent {
            ty: InterfaceType::String,
            offset: 1024,
        };
        let return_event = LowerMemoryReturnEvent(ResultEvent::from_anyhow_result(&Ok(())));

        rr_harness(
            |recorder| {
                recorder.record_event(|| entry.clone())?;
                recorder.record_event(|| return_event.clone())?;
                Ok(())
            },
            |replayer| {
                replayer.next_event_and(|e: LowerMemoryEntryEvent| {
                    assert_eq!(e.ty, InterfaceType::String);
                    assert_eq!(e.offset, 1024);
                    Ok(())
                })?;
                replayer.next_event_and(|e: LowerMemoryReturnEvent| {
                    assert!(e.0.ret().is_ok());
                    Ok(())
                })?;
                Ok(())
            },
        )
    }

    #[test]
    fn realloc_events() -> Result<()> {
        use __component_events::{ReallocEntryEvent, ReallocReturnEvent};

        let entry = ReallocEntryEvent {
            old_addr: 0x1000,
            old_size: 64,
            old_align: 8,
            new_size: 128,
        };
        let return_event = ReallocReturnEvent(ResultEvent::from_anyhow_result(&Ok(0x2000)));

        rr_harness(
            |recorder| {
                recorder.record_event(|| entry.clone())?;
                recorder.record_event(|| return_event.clone())?;
                Ok(())
            },
            |replayer| {
                replayer.next_event_and(|e: ReallocEntryEvent| {
                    assert_eq!(e.old_addr, 0x1000);
                    assert_eq!(e.old_size, 64);
                    assert_eq!(e.old_align, 8);
                    assert_eq!(e.new_size, 128);
                    Ok(())
                })?;
                replayer.next_event_and(|e: ReallocReturnEvent| {
                    assert_eq!(e.0.ret().unwrap(), 0x2000);
                    Ok(())
                })?;
                Ok(())
            },
        )
    }

    #[test]
    fn memory_slice_write_event() -> Result<()> {
        use __component_events::MemorySliceWriteEvent;

        let event = MemorySliceWriteEvent {
            offset: 512,
            bytes: vec![0x01, 0x02, 0x03, 0x04, 0xFF],
        };

        rr_harness(
            |recorder| {
                recorder.record_event(|| event.clone())?;
                Ok(())
            },
            |replayer| {
                replayer.next_event_and(|e: MemorySliceWriteEvent| {
                    assert_eq!(e.offset, 512);
                    assert_eq!(e.bytes, vec![0x01, 0x02, 0x03, 0x04, 0xFF]);
                    Ok(())
                })?;
                Ok(())
            },
        )
    }

    #[test]
    fn instantiation_event() -> Result<()> {
        use crate::component::ComponentInstanceId;
        use crate::store::InstanceId;
        use __component_events::InstantiationEvent as ComponentInstantiationEvent;
        use core_events::InstantiationEvent as CoreInstantiationEvent;

        let component_event = ComponentInstantiationEvent {
            component: [0xAB; 32],
            instance: ComponentInstanceId::from_u32(42),
        };

        let core_event = CoreInstantiationEvent {
            module: [0xCD; 32],
            instance: InstanceId::from_u32(17),
        };

        rr_harness(
            |recorder| {
                recorder.record_event(|| component_event.clone())?;
                recorder.record_event(|| core_event.clone())?;
                Ok(())
            },
            |replayer| {
                replayer.next_event_and(|e: ComponentInstantiationEvent| {
                    e.validate(&component_event)?;
                    Ok(())
                })?;
                replayer.next_event_and(|e: CoreInstantiationEvent| {
                    e.validate(&core_event)?;
                    Ok(())
                })?;
                Ok(())
            },
        )
    }
}

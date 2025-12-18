use super::FlatBytes;
use crate::component::ComponentInstanceId;
use crate::prelude::*;
use crate::store::InstanceId;
use crate::{AsContextMut, ModuleVersionStrategy, Val, ValRaw, ValType};
use wasmtime_environ::component::FlatTypesStorage;

// Public Re-exports
pub use wasm_crimp::{RecordSettings, RecordWriter, ReplayError, ReplayReader, ReplaySettings};
// Crate-internal re-exports
pub(crate) use wasm_crimp::{
    RREvent, RRFuncArgVals, Recorder, Replayer, ResultEvent, Validate, common_events,
    component_events::{self, RRComponentInstanceId},
    core_events::{self, RRModuleInstanceId},
    from_replay_reader, to_record_writer,
};

pub trait RRFuncArgValsConvertable {
    /// Construct [`RRFuncArgVals`] from raw value buffer and a flat size iterator
    fn from_flat_iter<T>(args: &[T], flat: impl Iterator<Item = u8>) -> Self
    where
        T: FlatBytes;

    /// Construct [`RRFuncArgVals`] from raw value buffer and a [`FlatTypesStorage`]
    fn from_flat_storage<T>(args: &[T], flat: FlatTypesStorage) -> RRFuncArgVals
    where
        T: FlatBytes;

    /// Encode [`RRFuncArgVals`] back into raw value buffer
    fn into_raw_slice<T>(self, raw_args: &mut [T])
    where
        T: FlatBytes;

    /// Generate a vector of [`crate::Val`] from [`RRFuncArgVals`] and [`ValType`]s
    fn to_val_vec(self, store: impl AsContextMut, val_types: Vec<ValType>) -> Vec<Val>;
}

impl RRFuncArgValsConvertable for RRFuncArgVals {
    #[inline]
    fn from_flat_iter<T>(args: &[T], flat: impl Iterator<Item = u8>) -> RRFuncArgVals
    where
        T: FlatBytes,
    {
        let mut bytes = Vec::new();
        let mut sizes = Vec::new();
        for (flat_size, arg) in flat.zip(args.iter()) {
            bytes.extend_from_slice(unsafe { &arg.bytes(flat_size) });
            sizes.push(flat_size);
        }
        RRFuncArgVals { bytes, sizes }
    }

    /// Construct [`RRFuncArgVals`] from raw value buffer and a [`FlatTypesStorage`]
    #[inline]
    fn from_flat_storage<T>(args: &[T], flat: FlatTypesStorage) -> RRFuncArgVals
    where
        T: FlatBytes,
    {
        RRFuncArgVals::from_flat_iter(args, flat.iter32())
    }

    /// Encode [`RRFuncArgVals`] back into raw value buffer
    #[inline]
    fn into_raw_slice<T>(self, raw_args: &mut [T])
    where
        T: FlatBytes,
    {
        let mut pos = 0;
        for (flat_size, dst) in self.sizes.into_iter().zip(raw_args.iter_mut()) {
            *dst = T::from_bytes(&self.bytes[pos..pos + flat_size as usize]);
            pos += flat_size as usize;
        }
    }

    /// Generate a vector of [`crate::Val`] from [`RRFuncArgVals`] and [`ValType`]s
    #[inline]
    fn to_val_vec(self, mut store: impl AsContextMut, val_types: Vec<ValType>) -> Vec<Val> {
        let mut pos = 0;
        let mut vals = Vec::new();
        for (flat_size, val_type) in self.sizes.into_iter().zip(val_types.into_iter()) {
            let raw = ValRaw::from_bytes(&self.bytes[pos..pos + flat_size as usize]);
            // SAFETY: The safety contract here is the same as that of [`Val::from_raw`].
            // The caller must ensure that raw has the type provided.
            vals.push(unsafe { Val::from_raw(&mut store, raw, val_type) });
            pos += flat_size as usize;
        }
        vals
    }
}

// Conversions from Wasmtime types to RR types
impl From<ComponentInstanceId> for RRComponentInstanceId {
    fn from(id: ComponentInstanceId) -> Self {
        RRComponentInstanceId(id.as_u32())
    }
}

impl From<InstanceId> for RRModuleInstanceId {
    fn from(id: InstanceId) -> Self {
        RRModuleInstanceId(id.as_u32())
    }
}

impl From<RRModuleInstanceId> for InstanceId {
    fn from(rr_id: RRModuleInstanceId) -> Self {
        InstanceId::from_u32(rr_id.0)
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
    fn new_recorder(writer: impl RecordWriter, settings: RecordSettings) -> Result<Self> {
        let settings_local = settings.clone();
        let mut buf = RecordBuffer {
            buf: Vec::new(),
            writer: Box::new(writer),
            settings,
        };
        buf.record_event(|| common_events::TraceSignatureEvent {
            checksum: ModuleVersionStrategy::WasmtimeVersion.as_str().to_string(),
            settings: settings_local,
        })?;
        Ok(buf)
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
            to_record_writer(&e, &mut *self.writer)?;
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
            let result = from_replay_reader(&mut *self.reader, &mut self.deser_buffer);
            match result {
                Err(e) => {
                    break 'event_loop Some(Err(ReplayError::FailedRead(e)));
                }
                Ok(event) => {
                    if let RREvent::Eof = &event {
                        self.eof_encountered = true;
                        break 'event_loop None;
                    } else if event.is_diagnostic() {
                        continue 'event_loop;
                    } else {
                        log::debug!("Read replay event => {event}");
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
    fn new_replayer(reader: impl ReplayReader + 'static, settings: ReplaySettings) -> Result<Self> {
        let mut buf = ReplayBuffer {
            reader: Box::new(reader),
            deser_buffer: vec![0; settings.deserialize_buffer_size],
            settings,
            // This doesn't matter now; will override after reading header
            trace_settings: RecordSettings::default(),
            eof_encountered: false,
        };

        let signature: common_events::TraceSignatureEvent = buf.next_event_typed()?;
        // Ensure the trace integrity
        signature.validate(ModuleVersionStrategy::WasmtimeVersion.as_str())?;
        // Update the trace settings
        buf.trace_settings = signature.settings;

        if buf.settings.validate && !buf.trace_settings.add_validation {
            log::warn!(
                "Replay validation will be omitted since the recorded trace has no validation metadata..."
            );
        }

        Ok(buf)
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
    use std::fs::File;
    use std::path::Path;
    use tempfile::{NamedTempFile, TempPath};
    use wasm_crimp::EventError;
    use wasmtime_environ::{FuncIndex, component::ResourceDropRet};

    impl ReplayBuffer {
        /// Pop the next replay event and calls `f` with a expected event type
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
    }

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
                "Recorded values {a_slice:?} and replayed values {b_slice:?} do not match"
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
                    instance: origin.instance.into(),
                    func_index: origin.index.into(),
                    args: RRFuncArgVals::from_flat_iter(&values, flat_sizes.iter().copied()),
                })?;
                recorder.record_event(|| component_events::WasmFuncEntryEvent {
                    args: RRFuncArgVals::from_flat_iter(
                        &return_values,
                        return_flat_sizes.iter().copied(),
                    ),
                })
            },
            |replayer| {
                replayer.next_event_and(|event: core_events::WasmFuncEntryEvent| {
                    replay_origin = Some(WasmFuncOrigin {
                        instance: event.instance.into(),
                        index: event.func_index.into(),
                    });
                    event.args.into_raw_slice(&mut replay_values);
                    Ok(())
                })?;
                assert!(origin == replay_origin.unwrap());
                verify_equal_slices(&values, &replay_values, &flat_sizes)?;

                replayer.next_event_and(|event: component_events::WasmFuncEntryEvent| {
                    event.args.into_raw_slice(&mut return_replay_values);
                    Ok(())
                })?;
                verify_equal_slices(&return_values, &return_replay_values, &return_flat_sizes)
            },
        )
    }

    #[test]
    fn builtin_event_entry() -> Result<()> {
        use component_events::{
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
        use component_events::{
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
        use component_events::{LowerFlatEntryEvent, LowerFlatReturnEvent};
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
        use component_events::{LowerMemoryEntryEvent, LowerMemoryReturnEvent};
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
        use component_events::{ReallocEntryEvent, ReallocReturnEvent};

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
        use component_events::MemorySliceWriteEvent;

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
        use component_events::InstantiationEvent as ComponentInstantiationEvent;
        use core_events::InstantiationEvent as CoreInstantiationEvent;
        use wasmtime_environ::WasmChecksum;

        let component_event = ComponentInstantiationEvent {
            component: WasmChecksum::from_binary(&[0xAB; 256]),
            instance: ComponentInstanceId::from_u32(42).into(),
        };

        let core_event = CoreInstantiationEvent {
            module: WasmChecksum::from_binary(&[0xCD; 256]),
            instance: InstanceId::from_u32(17).into(),
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

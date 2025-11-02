#[cfg(any(feature = "rr-component", feature = "rr-validate"))]
use super::ReplayError;
use crate::ValRaw;
use crate::prelude::*;
use core::fmt;
use core::mem::MaybeUninit;
use serde::{Deserialize, Serialize};

/// A serde compatible representation of errors produced by actions during
/// initial recording for specific events
///
/// We need this since the [anyhow::Error] trait object cannot be used. This
/// type just encapsulates the corresponding display messages during recording
/// so that it can be re-thrown during replay
///
/// Unforunately since we cannot serialize [anyhow::Error], there's no good
/// way to equate errors across record/replay boundary without creating a
/// common error format. Perhaps this is future work
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventActionError {
    ReallocError(String),
    LowerFlatError(String),
    LowerMemoryError(String),
    BuiltinError(String),
    WasmFuncReturnError(String),
}

impl fmt::Display for EventActionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReallocError(s)
            | Self::LowerFlatError(s)
            | Self::LowerMemoryError(s)
            | Self::BuiltinError(s)
            | Self::WasmFuncReturnError(s) => {
                write!(f, "{}", s)
            }
        }
    }
}

impl core::error::Error for EventActionError {}

/// Types that can be serialized/deserialized into/from
/// flat types for record and replay
pub trait FlatBytes {
    fn bytes_ref(&self, size: u8) -> &[u8];
    fn from_bytes(value: &[u8]) -> Self;
}

impl FlatBytes for ValRaw {
    #[inline]
    fn bytes_ref(&self, size: u8) -> &[u8] {
        &self.get_bytes()[..size as usize]
    }
    #[inline]
    fn from_bytes(value: &[u8]) -> Self {
        ValRaw::bytes(value)
    }
}

impl FlatBytes for MaybeUninit<ValRaw> {
    #[inline]
    fn bytes_ref(&self, size: u8) -> &[u8] {
        // Uninitialized data is assumed and serialized, so hence
        // may contain some undefined values
        let val = unsafe { self.assume_init_ref() };
        val.bytes_ref(size)
    }
    #[inline]
    fn from_bytes(value: &[u8]) -> Self {
        MaybeUninit::new(ValRaw::bytes(value))
    }
}

/// Representation of flat arguments for function entry/return
#[derive(Serialize, Deserialize, Clone, PartialEq)]
pub struct RRFuncArgVals {
    /// Flat data vector of bytes
    bytes: Vec<u8>,
    /// Descriptor vector of sizes of each flat types
    ///
    /// The length of this vector equals the number of flat types,
    /// and the sum of this vector equals the length of `bytes`
    sizes: Vec<u8>,
}

impl fmt::Debug for RRFuncArgVals {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RRFuncArgVals ")?;
        let mut pos: usize = 0;
        let mut list = f.debug_list();
        let hex_fmt = |bytes: &[u8]| {
            let hex_string = bytes
                .iter()
                .rev()
                .map(|b| format!("{:02x}", b))
                .collect::<String>();
            format!("0x{}", hex_string)
        };
        for flat_size in self.sizes.iter() {
            list.entry(&(
                flat_size,
                hex_fmt(&self.bytes[pos..pos + *flat_size as usize]),
            ));
            pos += *flat_size as usize;
        }
        list.finish()
    }
}

impl RRFuncArgVals {
    /// Construct [`RRFuncArgVals`] from raw value buffer and flat sizes
    pub fn from_raw_slice<T>(args: &[T], flat: impl Iterator<Item = u8>) -> RRFuncArgVals
    where
        T: FlatBytes,
    {
        let mut bytes = Vec::<u8>::new();
        let mut sizes = Vec::<u8>::new();
        for (flat_size, arg) in flat.zip(args.iter()) {
            bytes.extend_from_slice(&arg.bytes_ref(flat_size));
            sizes.push(flat_size);
        }
        RRFuncArgVals { bytes, sizes }
    }

    /// Encode [`RRFuncArgVals`] back into raw value buffer
    pub fn into_raw_slice<T>(self, raw_args: &mut [T])
    where
        T: FlatBytes,
    {
        let mut pos = 0;
        for (flat_size, dst) in self.sizes.into_iter().zip(raw_args.iter_mut()) {
            *dst = T::from_bytes(&self.bytes[pos..pos + flat_size as usize]);
            pos += flat_size as usize;
        }
    }
}

/// Trait signifying types that can be validated on replay
///
/// All `PartialEq` types are directly validatable with themselves.
/// Note however that some [`Validate`] implementations are present even
/// when feature `rr-validate` is disabled, when validation is needed
/// for a faithful replay (e.g. [`component_events::InstantiationEvent`]).
///
/// In terms of usage, an event that implements `Validate` can call
/// any RR validation methods on a `Store`
#[cfg(any(feature = "rr-component", feature = "rr-validate"))]
pub trait Validate<T: ?Sized> {
    /// Perform a validation of the event to ensure replay consistency
    fn validate(&self, expect: &T) -> Result<(), ReplayError>;

    /// Write a log message
    fn log(&self)
    where
        Self: fmt::Debug,
    {
        log::debug!("Validating => {:?}", self);
    }
}

#[cfg(any(feature = "rr-component", feature = "rr-validate"))]
impl<T> Validate<T> for T
where
    T: PartialEq + fmt::Debug,
{
    /// All types that are [`PartialEq`] are directly validatable with themselves
    fn validate(&self, expect: &T) -> Result<(), ReplayError> {
        self.log();
        if self == expect {
            Ok(())
        } else {
            log::error!("Validation against {:?} failed!", expect);
            Err(ReplayError::FailedValidation)
        }
    }
}

/// Events used as markers for debugging/testing in traces
///
/// Marker events should be injectable at any point in a record
/// trace without impacting functional correctness of replay
pub mod marker_events {
    use crate::prelude::*;
    use serde::{Deserialize, Serialize};

    /// A Nop event
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct NopEvent;

    /// An event for custom String messages
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
}

pub mod common_events;
pub mod component_events;
pub mod core_events;

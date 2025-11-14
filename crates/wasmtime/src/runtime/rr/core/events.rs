#[cfg(any(feature = "rr-component", feature = "rr-validate"))]
use super::ReplayError;
use crate::rr::FlatBytes;
use crate::{AsContextMut, Val, prelude::*};
use crate::{ValRaw, ValType};
use core::fmt;
use serde::{Deserialize, Serialize};
#[cfg(feature = "rr-component")]
use wasmtime_environ::component::FlatTypesStorage;

/// A serde compatible representation of errors produced during execution
/// of certain events
///
/// We need this since the [anyhow::Error] trait object cannot be used. This
/// type just encapsulates the corresponding display messages during recording
/// so that it can be re-thrown during replay. Unforunately since we cannot
/// serialize [anyhow::Error], there's no good way to equate errors across
/// record/replay boundary without creating a common error format.
/// Perhaps this is future work
pub trait EventError: core::error::Error + Send + Sync + 'static {
    fn new(t: String) -> Self
    where
        Self: Sized;
    fn get(&self) -> &String;
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
    /// Construct [`RRFuncArgVals`] from raw value buffer and a flat size iterator
    #[inline]
    pub fn from_flat_iter<T>(args: &[T], flat: impl Iterator<Item = u8>) -> RRFuncArgVals
    where
        T: FlatBytes,
    {
        let mut bytes = Vec::new();
        let mut sizes = Vec::new();
        for (flat_size, arg) in flat.zip(args.iter()) {
            bytes.extend_from_slice(&arg.bytes_ref(flat_size));
            sizes.push(flat_size);
        }
        RRFuncArgVals { bytes, sizes }
    }

    /// Construct [`RRFuncArgVals`] from raw value buffer and a [`FlatTypesStorage`]
    #[cfg(feature = "rr-component")]
    #[inline]
    pub fn from_flat_storage<T>(args: &[T], flat: FlatTypesStorage) -> RRFuncArgVals
    where
        T: FlatBytes,
    {
        RRFuncArgVals::from_flat_iter(args, flat.iter32())
    }

    /// Encode [`RRFuncArgVals`] back into raw value buffer
    #[inline]
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

    /// Generate a vector of [`crate::Val`] from [`RRFuncArgVals`] and [`ValType`]s
    #[inline]
    pub fn to_val_vec(self, mut store: impl AsContextMut, val_types: Vec<ValType>) -> Vec<Val> {
        let mut pos = 0;
        let mut vals = Vec::new();
        for (flat_size, val_type) in self.sizes.into_iter().zip(val_types.into_iter()) {
            let raw = ValRaw::bytes(&self.bytes[pos..pos + flat_size as usize]);
            vals.push(unsafe { Val::from_raw(&mut store, raw, val_type) });
            pos += flat_size as usize;
        }
        vals
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

/// Result newtype for events that can be serialized/deserialized for record/replay.
///
/// Anyhow result types cannot use blanket PartialEq implementations since
/// anyhow results are not serialized directly. They need to specifically check
/// for divergence between recorded and replayed effects with [EventError]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultEvent<T, E: EventError>(Result<T, E>);

impl<T, E> ResultEvent<T, E>
where
    T: Clone,
    E: EventError,
{
    pub fn from_anyhow_result(ret: &Result<T>) -> Self {
        Self(
            ret.as_ref()
                .map(|t| (*t).clone())
                .map_err(|e| E::new(e.to_string())),
        )
    }
    pub fn ret(self) -> Result<T, E> {
        self.0
    }
}

impl<T, E> Validate<Result<T>> for ResultEvent<T, E>
where
    T: fmt::Debug + PartialEq,
    E: EventError,
{
    fn validate(&self, expect_ret: &Result<T>) -> Result<(), ReplayError> {
        self.log();
        // Cannot just use eq since anyhow::Error and EventError cannot be compared
        match (self.0.as_ref(), expect_ret.as_ref()) {
            (Ok(r), Ok(s)) => {
                if r == s {
                    Ok(())
                } else {
                    Err(ReplayError::FailedValidation)
                }
            }
            // Return the recorded error
            (Err(e), Err(f)) => Err(ReplayError::from(E::new(format!(
                "Error on execution: {} | Error from recording: {}",
                f,
                e.get()
            )))),
            // Diverging errors.. Report as a failed validation
            (Ok(_), Err(_)) => Err(ReplayError::FailedValidation),
            (Err(_), Ok(_)) => Err(ReplayError::FailedValidation),
        }
    }
}

macro_rules! event_error_types {
    (
        $(
            $( #[cfg($attr:meta)] )?
            pub struct $ee:ident(..)
        ),*
    ) => (
        $(
            /// Return from a reallocation call (needed only for validation)
            #[derive(Debug, Serialize, Deserialize, Clone)]
            pub struct $ee(String);

            impl core::error::Error for $ee {}
            impl fmt::Display for $ee {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    write!(f, "{}", &self.0)
                }
            }
            impl EventError for $ee {
                fn new(t: String) -> Self where Self: Sized { Self(t) }
                fn get(&self) -> &String { &self.0 }
            }
        )*
    );
}

event_error_types! {
    pub struct WasmFuncReturnError(..)
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

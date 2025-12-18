use crate::{ReplayError, prelude::*};
use core::fmt;

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
    pub bytes: Vec<u8>,
    /// Descriptor vector of sizes of each flat types
    ///
    /// The length of this vector equals the number of flat types,
    /// and the sum of this vector equals the length of `bytes`
    pub sizes: Vec<u8>,
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
                .map(|b| format!("{b:02x}"))
                .collect::<String>();
            format!("0x{hex_string}")
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

/// Trait signifying types that can be validated on replay
///
/// All `PartialEq` types are directly validatable with themselves.
/// Note however that some [`Validate`] implementations are present and
/// required for a faithful replay (e.g. [`component_events::InstantiationEvent`]).
///
/// In terms of usage, an event that implements `Validate` can call
/// any RR validation methods on a `Store`
pub trait Validate<T: ?Sized> {
    /// Perform a validation of the event to ensure replay consistency
    fn validate(&self, expect: &T) -> Result<(), ReplayError>;

    /// Write a log message
    fn log(&self)
    where
        Self: fmt::Debug,
    {
        log::debug!("Validating => {self:?}");
    }
}

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
            log::error!("Validation against {expect:?} failed!");
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

pub mod common_events;
pub mod component_events;
pub mod core_events;

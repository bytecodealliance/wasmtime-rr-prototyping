//! Module comprising of component model wasm events

use super::*;
use crate::component::Component;
use crate::vm::component::libcalls::ResourceDropRet;
// Re-export common events from this module
pub use common_events::*;
use wasmtime_environ::{
    self,
    component::{ExportIndex, FlatTypesStorage, InterfaceType, TypeFuncIndex},
};

/// A [`Component`] instantiatation event
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstantiationEvent {
    /// A checksum of the component bytecode
    component: [u8; 32],
}

impl InstantiationEvent {
    pub fn from_component(component: &Component) -> Self {
        Self {
            component: *component.checksum(),
        }
    }
}

/// A call event from Host into a Wasm component function
///
/// Note: Could potential merge with [`HostFuncReturnEvent`] as [`WasmToHostEvent`]?
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmFuncEntryEvent {
    /// Raw values passed across call boundary
    args: RRFuncArgVals,
    /// Checksum of component containing function
    component: [u8; 32],
    func_idx: ExportIndex,
}
impl WasmFuncEntryEvent {
    // Record
    pub fn new(
        args: &[ValRaw],
        flat: FlatTypesStorage,
        component: [u8; 32],
        func_idx: ExportIndex,
    ) -> Self {
        Self {
            args: RRFuncArgVals::from_raw_slice(args, flat.iter32()),
            component,
            func_idx,
        }
    }

    // Replay
    /// Consume the caller event and encode it back into the slice
    pub fn move_into_slice(self, args: &mut [MaybeUninit<ValRaw>]) {
        self.args.into_raw_slice(args);
    }
}

/// A call event from a Wasm component into the host
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HostFuncEntryEvent {
    /// Raw values passed across the call entry boundary
    args: RRFuncArgVals,

    /// Function index (required to support replay validation).
    ///
    /// Note: This relies on the invariant that [InterfaceType] will always be
    /// deterministic. Currently, the type indices into various [ComponentTypes]
    /// maintain this, allowing for quick type-checking.
    ty: TypeFuncIndex,
}
impl HostFuncEntryEvent {
    // Record
    pub fn new(args: &[MaybeUninit<ValRaw>], flat: FlatTypesStorage, ty: TypeFuncIndex) -> Self {
        Self {
            args: RRFuncArgVals::from_raw_slice(args, flat.iter32()),
            ty: ty,
        }
    }
}

/// A reallocation call event in the Component Model canonical ABI
///
/// Usually performed during lowering of complex [`ComponentType`]s to Wasm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReallocEntryEvent {
    pub old_addr: usize,
    pub old_size: usize,
    pub old_align: u32,
    pub new_size: usize,
}

/// Entry to a type lowering invocation to flat destination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LowerFlatEntryEvent {
    pub ty: InterfaceType,
}

/// Entry to type lowering invocation to destination in memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LowerMemoryEntryEvent {
    pub ty: InterfaceType,
    pub offset: usize,
}

/// A write to a mutable slice of Wasm linear memory by the host. This is the
/// fundamental representation of host-written data to Wasm and is usually
/// performed during lowering of a [`ComponentType`].
/// Note that this currently signifies a single mutable operation at the smallest granularity
/// on a given linear memory slice. These can be optimized and coalesced into
/// larger granularity operations in the future at either the recording or the replay level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySliceWriteEvent {
    pub offset: usize,
    pub bytes: Vec<u8>,
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
    pub struct ReallocError(..),
    pub struct LowerFlatError(..),
    pub struct LowerMemoryError(..),
    pub struct WasmFuncReturnError(..),
    pub struct BuiltinError(..)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReallocReturnEvent(pub ResultEvent<usize, ReallocError>);

/// Return from type lowering to flat destination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LowerFlatReturnEvent(pub ResultEvent<(), LowerFlatError>);

/// Return from type lowering to destination in memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LowerMemoryReturnEvent(pub ResultEvent<(), LowerMemoryError>);

/// A return event from a Wasm component function to Host
///
/// Matches 1:1 with [`WasmFuncEntryEvent`].
///
/// Note: Could potential merge with [`HostFuncReturnEvent`] as [`HostToWasmEvent`]?
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmFuncReturnEvent(pub ResultEvent<RRFuncArgVals, WasmFuncReturnError>);

impl Validate<Result<RRFuncArgVals>> for WasmFuncReturnEvent {
    fn validate(&self, expect: &Result<RRFuncArgVals>) -> Result<(), ReplayError> {
        self.0.validate(expect)
    }
}

// Macro to generate RR events from the builtin descriptions
macro_rules! builtin_events {
    // Main rule matching component function definitions
    (
        $(
            $( #[cfg($attr:meta)] )?
            $( #[rr_builtin(variant = $rr_var:ident, entry = $rr_entry:ident $(, exit = $rr_return:ident)? $(, success_ty = $rr_succ:tt)?)] )?
            $name:ident( vmctx: vmctx $(, $pname:ident: $param:ident )* ) $( -> $result:ident )?;
        )*
    ) => (
        builtin_events!(@gen_return_enum $($($($rr_var $rr_return)?)?)*);
        builtin_events!(@gen_entry_enum $($($rr_var $rr_entry)?)*);
        // Prioitize ret_succ if provided
        $(
            builtin_events!(@gen_entry_events $($rr_entry)? $($pname, $param)*);
            builtin_events!(@gen_return_events $($($rr_return)?)? -> $($($rr_succ)?)? $($result)?);
        )*
    );

    // All things related to BuiltinReturnEvent enum
    (@gen_return_enum $($rr_var:ident $event:ident)*) => {
        #[derive(Clone, Serialize, Deserialize)]
        pub enum BuiltinReturnEvent {
            $($rr_var($event),)*
        }
        builtin_events!(@from_impls BuiltinReturnEvent $($rr_var $event)*);
    };

    // All things related to BuiltinEntryEvent enum
    (@gen_entry_enum $($rr_var:ident $event:ident)*) => {
        // PartialEq gives all these events `Validate`
        #[derive(Clone, PartialEq, Serialize, Deserialize)]
        pub enum BuiltinEntryEvent {
            $($rr_var($event),)*
        }
        builtin_events!(@from_impls BuiltinEntryEvent $($rr_var $event)*);
    };


    (@gen_entry_events $rr_entry:ident $($pname:ident, $param:ident)*) => {
        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
        pub struct $rr_entry {
            $(pub $pname: $param),*
        }
    };
    // Stubbed if `rr_builtin` not provided
    (@gen_entry_events $($pname:ident, $param:ident)*) => {};

    (@gen_return_events $rr_return:ident -> $($result_opts:tt)*) => {
        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct $rr_return(pub ResultEvent<builtin_events!(@ret_first $($result_opts)*), BuiltinError>);

        impl $rr_return {
            pub fn ret(self) -> Result<builtin_events!(@ret_first $($result_opts)*)> {
                self.0.0.map_err(|e| e.into())
            }
        }
    };
    // Stubbed if `rr_builtin` not provided
    (@gen_return_events -> $($result_opts:tt)*) => {};

    // Debug traits for $enum (BuiltinReturnEvent/BuiltinEntryEvent) and
    // conversion to/from specific `$event` to `$enum`
    (@from_impls $enum:ident $($rr_var:ident $event:ident)*) => {
        $(
            impl From<$event> for $enum {
                fn from(value: $event) -> Self {
                    Self::$rr_var(value)
                }
            }

            impl TryFrom<$enum> for $event {
                type Error = ReplayError;

                fn try_from(value: $enum) -> Result<Self, Self::Error> {
                    #[allow(irrefutable_let_patterns)]
                    if let $enum::$rr_var(x) = value {
                        Ok(x)
                    } else {
                        Err(ReplayError::IncorrectEventVariant)
                    }
                }
            }
        )*

        impl fmt::Debug for $enum {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let mut res = f.debug_tuple(stringify!($enum));
                match self {
                    $(Self::$rr_var(e) => res.field(e),)*
                }.finish()
            }
        }
    };

    // Return first value
    (@ret_first $first:tt $($rest:tt)*) => ($first);
}

// Entry/return events for each builtin function
wasmtime_environ::foreach_builtin_component_function!(builtin_events);

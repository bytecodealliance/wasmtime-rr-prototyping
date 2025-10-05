//! Module comprising of component model wasm events

use super::*;
use crate::component::Component;
use crate::vm::component::libcalls::ResourceDropRet;
// Re-export common events from this module
pub use common_events::*;
use wasmtime_environ::{
    self,
    component::{ExportIndex, InterfaceType, TypeFuncIndex},
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
    pub fn new(args: &[ValRaw], component: [u8; 32], func_idx: ExportIndex) -> Self {
        Self {
            args: func_argvals_from_raw_slice(args),
            component,
            func_idx,
        }
    }

    // Replay
    /// Consume the caller event and encode it back into the slice
    pub fn move_into_slice(self, args: &mut [MaybeUninit<ValRaw>]) {
        func_argvals_into_raw_slice(self.args, args);
    }
}

/// A call event from a Wasm component into the host
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    pub fn new(args: &[MaybeUninit<ValRaw>], ty: TypeFuncIndex) -> Self {
        Self {
            args: func_argvals_from_raw_slice(args),
            ty: ty,
        }
    }
}
#[cfg(feature = "rr-validate")]
impl Validate<TypeFuncIndex> for HostFuncEntryEvent {
    fn validate(&self, expect_ty: &TypeFuncIndex) -> Result<(), ReplayError> {
        self.log();
        if &self.ty == expect_ty {
            Ok(())
        } else {
            Err(ReplayError::FailedValidation)
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

macro_rules! generic_new_result_events {
    (
        $(
            $(#[$meta:meta])*
            $event:ident -> ($ok_ty:ty,$err_variant:path)
        ),*
    ) => (
        $(
            $(#[$meta])*
            #[derive(Debug, Clone, Serialize, Deserialize)]
            pub struct $event(Result<$ok_ty, EventActionError>);

            impl $event {
                pub fn from_anyhow_result(ret: &Result<$ok_ty>) -> Self {
                    Self(ret.as_ref().map(|t| (*t).clone()).map_err(|e| $err_variant(e.to_string())))
                }
                pub fn ret(self) -> Result<$ok_ty, EventActionError> { self.0 }
            }

            generic_new_result_events!(@validate_impl $event,$ok_ty,$err_variant);
        )*
    );

    (@validate_impl $event:ident,$ok_ty:ty,$err_variant:path) => {
        #[cfg(feature = "rr-validate")]
        /// Note: Anyhow result types cannot use blanket PartialEq implementations since
        /// anyhow results are not serialized directly. They need to specifically check
        /// for divergence between recorded and replayed effects with [EventActionError]
        impl Validate<Result<$ok_ty>> for $event {
            fn validate(&self, expect_ret: &Result<$ok_ty>) -> Result<(), ReplayError> {
                self.log();
                // Cannot just use eq since anyhow::Error and EventActionError cannot be compared
                match (self.0.as_ref(), expect_ret.as_ref()) {
                    (Ok(r), Ok(s)) => {
                        if r == s {
                            Ok(())
                        } else {
                            Err(ReplayError::FailedValidation)
                        }
                    }
                    // Return the recorded error
                    (Err(e), Err(f)) => Err(ReplayError::from($err_variant(format!(
                        "Replayed Error for {}: {} \nRecorded Error for {}: {}",
                        stringify!($event), e, stringify!($event), f
                    )))),
                    // Diverging errors.. Report as a failed validation
                    (Ok(_), Err(_)) => Err(ReplayError::FailedValidation),
                    (Err(_), Ok(_)) => Err(ReplayError::FailedValidation),
                }
            }
        }
    };
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
        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub enum BuiltinReturnEvent {
            $($rr_var($event),)*
        }
        builtin_events!(@from_impls BuiltinReturnEvent $($rr_var $event)*);
    };

    // All things related to BuiltinEntryEvent enum
    (@gen_entry_enum $($rr_var:ident $event:ident)*) => {
        // PartialEq gives all these events `Validate`
        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
        pub struct $rr_return(Result<builtin_events!(@ret_first $($result_opts)*), EventActionError>);

        impl $rr_return {
            pub fn from_anyhow_result(ret: &Result<builtin_events!(@ret_first $($result_opts)*)>) -> Self {
                Self(
                    ret.as_ref()
                        .map(|t| t.clone())
                        .map_err(|e| EventActionError::BuiltinError(e.to_string())),
                )
            }
            pub fn ret(self) -> Result<builtin_events!(@ret_first $($result_opts)*)> {
                self.0.map_err(|e| e.into())
            }
        }
    };
    // Stubbed if `rr_builtin` not provided
    (@gen_return_events -> $($result_opts:tt)*) => {};

    // Conversion to/from specific return `$event` and `BuiltinEntryEvent`
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
    };

    // Return first value
    (@ret_first $first:tt $($rest:tt)*) => ($first);
}

// Return events with anyhow error conversion to EventActionError
generic_new_result_events! {
    /// Return from a reallocation call (needed only for validation)
    ReallocReturnEvent -> (usize, EventActionError::ReallocError),
    /// Return from type lowering to flat destination
    LowerFlatReturnEvent -> ((), EventActionError::LowerFlatError),
    /// Return from type lowering to destination in memory
    LowerMemoryReturnEvent -> ((), EventActionError::LowerMemoryError),
    /// A return event from a Wasm component function to Host
    ///
    /// Matches 1:1 with [`WasmFuncEntryEvent`].
    ///
    /// Note: Could potential merge with [`HostFuncEntryEvent`] as [`HostToWasmEvent`]?
    WasmFuncReturnEvent -> (RRFuncArgVals, EventActionError::WasmFuncReturnError)
}

// Entry/return events for each builtin function
wasmtime_environ::foreach_builtin_component_function!(builtin_events);

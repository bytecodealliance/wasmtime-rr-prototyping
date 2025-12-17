//! Module comprising of component model wasm events

use super::*;
use crate::{ExportIndex, InterfaceType, ResourceDropRet, WasmChecksum};

/// Representation of a component instance identifier during record/replay.
///
/// This ID is tied to component instantiation events in the trace, and used
/// during replay to refer to specific component instances.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Ord, PartialOrd)]
pub struct RRComponentInstanceId(pub u32);

/// Beginning marker for a Wasm component function call from host.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WasmFuncBeginEvent {
    /// Instance ID for the component instance.
    pub instance: RRComponentInstanceId,
    /// Export index for the invoked function.
    pub func_index: ExportIndex,
}

/// A instantiatation event for a Wasm component.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Ord, PartialOrd)]
pub struct InstantiationEvent {
    /// Checksum of the bytecode used to instantiate the component
    pub component: WasmChecksum,
    /// Instance ID for the instantiated component.
    pub instance: RRComponentInstanceId,
}

/// A call to `post_return` (after the function call).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostReturnEvent {
    /// Instance ID for the component instance.
    pub instance: RRComponentInstanceId,
    /// Export index for the function on which post_return is invoked.
    pub func_index: ExportIndex,
}

/// A call event from Host into a Wasm component function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmFuncEntryEvent {
    /// Raw values passed across call boundary.
    pub args: RRFuncArgVals,
}

/// A reallocation call event in the Wasm Component Model canonical ABI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReallocEntryEvent {
    pub old_addr: usize,
    pub old_size: usize,
    pub old_align: u32,
    pub new_size: usize,
}

/// Entry to a type lowering invocation to flat destination.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LowerFlatEntryEvent {
    pub ty: InterfaceType,
}

/// Entry to type lowering invocation to destination in memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LowerMemoryEntryEvent {
    pub ty: InterfaceType,
    pub offset: usize,
}

/// A write to a mutable slice of Wasm linear memory by the host. This is the
/// fundamental representation of host-written data to Wasm and is usually
/// performed during lowering of a [`ComponentType`].
///
/// Note that this currently signifies a single mutable operation at the smallest granularity
/// on a given linear memory slice. These can be optimized and coalesced into
/// larger granularity operations in the future at either the recording or the replay level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySliceWriteEvent {
    pub offset: usize,
    pub bytes: Vec<u8>,
}

event_error_types! {
    pub struct ReallocError(..),
    pub struct LowerFlatError(..),
    pub struct LowerMemoryError(..),
    pub struct BuiltinError(..)
}

/// Return from a reallocation call in the Component Model canonical ABI, providing
/// the address of allocation if successful.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReallocReturnEvent(pub ResultEvent<usize, ReallocError>);

/// Return from type lowering to flat destination.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LowerFlatReturnEvent(pub ResultEvent<(), LowerFlatError>);

/// Return from type lowering to destination in memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LowerMemoryReturnEvent(pub ResultEvent<(), LowerMemoryError>);

// Macro to generate RR events from the builtin descriptions.
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

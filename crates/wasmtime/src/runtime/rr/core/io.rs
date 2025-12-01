use crate::prelude::*;
use core::any::Any;
use postcard;
use serde::{Deserialize, Serialize};

cfg_if::cfg_if! {
    if #[cfg(feature = "std")] {
        use std::io::{Write, Seek, Read};
        /// A writer for recording in RR.
        pub trait RecordWriter: Write + Send + Sync + Any {}
        impl<T: Write + Send + Sync + Any> RecordWriter for T {}

        /// A reader for replaying in RR.
        pub trait ReplayReader: Read + Seek + Send + Sync {}
        impl<T: Read + Seek + Send + Sync> ReplayReader for T {}

    } else {
        use core::{convert::AsRef, iter::Extend};

        /// A writer for recording in RR.
        pub trait RecordWriter: Extend<u8> + Send + Sync + Any {}
        impl<T: Extend<u8> + Send + Sync + Any> RecordWriter for T {}

        /// A reader for replaying in RR.
        ///
        /// In `no_std`, types must provide explicit read/seek capabilities
        /// to a underlying byte slice through these methods.
        pub trait ReplayReader: AsRef<[u8]> + Send + Sync {
            /// Advance the reader's internal cursor by `cnt` bytes
            fn advance(&mut self, cnt: usize);
            /// Seek to an absolute position `pos` in the reader
            fn seek(&mut self, pos: usize);
        }

    }
}

/// Serialize and write `value` to a `RecordWriter`
///
/// Currently uses `postcard` serializer
pub(super) fn to_record_writer<T, W>(value: &T, writer: &mut W) -> Result<()>
where
    T: Serialize + ?Sized,
    W: RecordWriter,
{
    #[cfg(feature = "std")]
    {
        postcard::to_io(value, writer)?;
    }
    #[cfg(not(feature = "std"))]
    {
        postcard::to_extend(value, writer)?;
    }
    Ok(())
}

/// Read and deserialize a `value` from a `ReplayReader`.
///
/// Currently uses `postcard` deserializer, with optional scratch
/// buffer to deserialize into
pub(super) fn from_replay_reader<'a, T, R>(reader: &'a mut R, scratch: &'a mut [u8]) -> Result<T>
where
    T: Deserialize<'a>,
    R: ReplayReader,
{
    #[cfg(feature = "std")]
    {
        Ok(postcard::from_io((reader, scratch))?.0)
    }
    #[cfg(not(feature = "std"))]
    {
        let bytes = reader.as_ref();
        let original_len = bytes.len();
        let (value, new) = postcard::take_from_bytes(bytes)?;
        reader.advance(new.len() - original_len);
        Ok(value)
    }
}

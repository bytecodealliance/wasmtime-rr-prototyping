use crate::{RREvent, prelude::*};
use core::any::Any;
use postcard;

cfg_if::cfg_if! {
    if #[cfg(feature = "std")] {
        extern crate std;
        use std::io::{Write, Seek, Read};
        /// A writer for recording in RR.
        pub trait RecordWriter: Write + Send + Sync + Any {}
        impl<T: Write + Send + Sync + Any> RecordWriter for T {}

        /// A reader for replaying in RR.
        pub trait ReplayReader: Read + Seek + Send + Sync {}
        impl<T: Read + Seek + Send + Sync> ReplayReader for T {}

    } else {
        use core::iter::Extend;
        use postcard::{ser_flavors, de_flavors};

        type PcError = postcard::Error;
        type PcResult<T> = postcard::Result<T>;

        /// A writer for recording in RR.
        ///
        /// In `no_std`, types must provide explicit write capabilities.
        pub trait RecordWriter: Send + Sync + Any {
            /// Write all the bytes from `buf` to the writer
            fn write(&mut self, buf: &[u8]) -> Result<usize>;
            /// Flush the writer
            fn flush(&mut self) -> Result<()>;
        }
        impl <T: Send + Sync + Any + Extend<u8>> RecordWriter for T {
            fn write(&mut self, buf: &[u8]) -> Result<usize> {
                self.extend(buf.iter().copied());
                Ok(buf.len())
            }
            fn flush(&mut self) -> Result<()> {
                Ok(())
            }
        }

        /// A reader for replaying in RR.
        ///
        /// In `no_std`, types must provide explicit read/seek capabilities.
        pub trait ReplayReader: Send + Sync {
            /// Read bytes into `buf`, returning number of bytes read.
            fn read(&mut self, buf: &mut [u8]) -> Result<usize>;
            /// Seek to an absolute position `pos` in the reader.
            fn seek(&mut self, pos: usize);
        }

        /// Resembles a `WriteFlavor` in postcard
        struct RecordWriterFlavor<'a, W: RecordWriter + ?Sized> {
            writer: &'a mut W,
        }

        impl<'a, W: RecordWriter + ?Sized> ser_flavors::Flavor for RecordWriterFlavor<'a, W> {
            type Output = ();

            #[inline]
            fn try_push(&mut self, data: u8) -> PcResult<()> {
                self.writer.write(&[data]).map_err(|_| PcError::SerializeBufferFull)?;
                Ok(())
            }
            #[inline]
            fn try_extend(&mut self, data: &[u8]) -> PcResult<()> {
                self.writer.write(data).map_err(|_| PcError::SerializeBufferFull)?;
                Ok(())
            }
            #[inline]
            fn finalize(self) -> PcResult<Self::Output> {
                self.writer.flush().map_err(|_| PcError::SerializeBufferFull)?;
                Ok(())
            }
        }

        struct ReplayReaderFlavor<'a, 'b, R: ReplayReader + ?Sized> {
            reader: &'a mut R,
            scratch: &'b mut [u8],
        }

        impl<'de, 'a, 'b, R: ReplayReader + ?Sized> de_flavors::Flavor<'de> for ReplayReaderFlavor<'a, 'b, R>
        where 'b: 'de, 'a: 'de
        {
            type Remainder = ();
            type Source = ();

            #[inline]
            fn pop(&mut self) -> PcResult<u8> {
                let scratch = core::mem::replace(&mut self.scratch, &mut []);
                if scratch.is_empty() {
                    return PcResult::Err(PcError::DeserializeUnexpectedEnd);
                }
                let (slice, rest) = scratch.split_at_mut(1);
                self.scratch = rest;

                match self.reader.read(slice) {
                    Ok(1) => Ok(slice[0]),
                    _ => PcResult::Err(PcError::DeserializeUnexpectedEnd),
                }
            }

            #[inline]
            fn try_take_n(&mut self, ct: usize) -> postcard::Result<&'de [u8]> {
                let scratch = core::mem::replace(&mut self.scratch, &mut []);
                if scratch.len() < ct {
                    return PcResult::Err(PcError::DeserializeUnexpectedEnd);
                }
                let (slice, rest) = scratch.split_at_mut(ct);
                self.scratch = rest;

                let mut total_read = 0;
                while total_read < ct {
                    match self.reader.read(&mut slice[total_read..]) {
                        Ok(0) => return PcResult::Err(PcError::DeserializeUnexpectedEnd),
                        Ok(n) => total_read += n,
                        Err(_) => return PcResult::Err(PcError::DeserializeUnexpectedEnd),
                    }
                }

                Ok(slice)
            }

            #[inline]
            fn finalize(self) -> PcResult<Self::Remainder> {
                Ok(())
            }
        }
    }
}

/// Serialize and write an `RREvent` to a `RecordWriter`
///
/// This is the lowest-level underlying writer function for RR events,
/// helpful for implementing `Recorder`s. Currently uses [`postcard`] serializer.
pub fn to_record_writer<W>(value: &RREvent, writer: &mut W) -> Result<()>
where
    W: RecordWriter + ?Sized,
{
    #[cfg(feature = "std")]
    {
        postcard::to_io(value, writer)?;
    }
    #[cfg(not(feature = "std"))]
    {
        let flavor = RecordWriterFlavor { writer };
        postcard::serialize_with_flavor(value, flavor)?;
    }
    Ok(())
}

/// Read and deserialize an `RREvent` from a `ReplayReader`.
///
/// This is the lowest-level underlying reader function for RR events,
/// helpful for implementing `Replayer`s. Currently uses [`postcard`] deserializer.
pub fn from_replay_reader<'a, R>(reader: &'a mut R, scratch: &'a mut [u8]) -> Result<RREvent>
where
    R: ReplayReader + ?Sized,
{
    #[cfg(feature = "std")]
    {
        Ok(postcard::from_io((reader, scratch))?.0)
    }
    #[cfg(not(feature = "std"))]
    {
        let flavor = ReplayReaderFlavor { reader, scratch };
        let mut deserializer = postcard::Deserializer::from_flavor(flavor);
        let t = serde::Deserialize::deserialize(&mut deserializer)?;
        deserializer.finalize()?;
        Ok(t)
    }
}

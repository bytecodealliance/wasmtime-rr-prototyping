//! A threaded writer that decouples event serialization from file I/O.
//!
//! [`ThreadedWriter`] implements [`std::io::Write`] (and thus [`RecordWriter`])
//! by buffering writes in memory and shipping full buffers to a background
//! thread over a bounded channel. The background thread performs the actual
//! I/O to the underlying writer (typically a [`std::io::BufWriter<std::fs::File>`]).

use super::RecordWriter;
use crate::prelude::*;
use alloc::sync::Arc;
use core::sync::atomic::{AtomicBool, Ordering};
use std::io::{self, Write};
use std::sync::Mutex;
use std::sync::mpsc::{self, SyncSender};
use std::thread::{self, JoinHandle};

/// Messages sent from the foreground to the background writer thread.
enum WriterMsg {
    /// A buffer of serialized bytes to write.
    Buffer(Vec<u8>),
    /// Request the background thread to flush the underlying writer.
    Flush,
    /// Shut down: flush the writer and exit the thread.
    Shutdown,
}

/// Shared error slot for propagating I/O errors from the background thread.
///
/// Uses an [`AtomicBool`] fast path so that the common (no-error) case avoids
/// locking the mutex entirely — just a single relaxed atomic load.
struct SharedError {
    has_error: AtomicBool,
    error: Mutex<Option<io::Error>>,
}

impl SharedError {
    fn new() -> Arc<Self> {
        Arc::new(SharedError {
            has_error: AtomicBool::new(false),
            error: Mutex::new(None),
        })
    }

    /// Store an error (only the first error is kept).
    fn set(&self, err: io::Error) {
        let mut guard = self.error.lock().unwrap();
        if guard.is_none() {
            *guard = Some(err);
            self.has_error.store(true, Ordering::Relaxed);
        }
    }

    /// Take the stored error, if any. Fast path: returns `None` without
    /// locking the mutex when no error has been set.
    fn take(&self) -> Option<io::Error> {
        if !self.has_error.load(Ordering::Relaxed) {
            return None;
        }
        let err = self.error.lock().unwrap().take();
        if err.is_some() {
            self.has_error.store(false, Ordering::Relaxed);
        }
        err
    }
}

/// Configuration for [`ThreadedWriter`].
pub struct ThreadedWriterConfig {
    /// Size (in bytes) of each local buffer before it is shipped to the
    /// background thread.
    pub buffer_capacity: usize,
    /// Maximum number of buffers that can be queued in the channel before the
    /// foreground blocks. Defaults to 8.
    pub channels: usize,
}

/// A writer that buffers serialized bytes in memory and ships them to a
/// background thread for async I/O to the underlying writer.
///
/// Implements [`Write`] (and therefore [`RecordWriter`] under `std`).
///
/// # Backpressure
///
/// Uses a bounded channel (`sync_channel`). When the channel is full, writes
/// block until the background thread drains a slot. With defaults of 8 slots
/// at 64 KiB each, at most 512 KiB of data is queued.
pub struct ThreadedWriter {
    /// Channel sender, wrapped in Mutex for the `Sync` bound required by
    /// `RecordWriter`. Never contended since we always have `&mut self`.
    sender: Mutex<SyncSender<WriterMsg>>,
    /// Return channel for recycling drained buffers from the background thread.
    /// Wrapped in Mutex for the `Sync` bound; only locked in `ship_buffer()`
    /// (once per ~64 KiB), not on every small `write()` call.
    recycler: Mutex<mpsc::Receiver<Vec<u8>>>,
    /// Local accumulation buffer.
    buffer: Vec<u8>,
    /// Threshold at which the buffer is shipped.
    buffer_capacity: usize,
    /// Shared error from the background thread.
    shared_error: Arc<SharedError>,
    /// Background thread handle; taken and joined on drop.
    join_handle: Option<JoinHandle<()>>,
}

impl ThreadedWriter {
    /// Create a new `ThreadedWriter` that wraps the given writer.
    ///
    /// Spawns a background thread named `"rr-record-writer"` that receives
    /// buffers over a bounded channel and writes them to `writer`.
    pub fn new(writer: Box<dyn RecordWriter>, config: ThreadedWriterConfig) -> Self {
        let (tx, rx) = mpsc::sync_channel::<WriterMsg>(config.channels);
        let (recycle_tx, recycle_rx) = mpsc::channel::<Vec<u8>>();
        let shared_error = SharedError::new();
        let bg_error = Arc::clone(&shared_error);

        let handle = thread::Builder::new()
            .name("rr-record-writer".into())
            .spawn(move || {
                Self::bg_thread_main(writer, rx, bg_error, recycle_tx);
            })
            .expect("failed to spawn rr-record-writer thread");

        ThreadedWriter {
            sender: Mutex::new(tx),
            recycler: Mutex::new(recycle_rx),
            buffer: Vec::with_capacity(config.buffer_capacity),
            buffer_capacity: config.buffer_capacity,
            shared_error,
            join_handle: Some(handle),
        }
    }

    /// Background thread entry point.
    fn bg_thread_main(
        mut writer: Box<dyn RecordWriter>,
        rx: mpsc::Receiver<WriterMsg>,
        shared_error: Arc<SharedError>,
        recycle_tx: mpsc::Sender<Vec<u8>>,
    ) {
        for msg in rx.iter() {
            let result = match msg {
                WriterMsg::Buffer(mut data) => {
                    let r = writer.write_all(&data);
                    // Recycle the buffer back to the foreground (ignore if fg dropped).
                    data.clear();
                    let _ = recycle_tx.send(data);
                    r
                }
                WriterMsg::Flush => writer.flush(),
                WriterMsg::Shutdown => {
                    let r = writer.flush();
                    if let Err(e) = r {
                        shared_error.set(e);
                    }
                    return;
                }
            };
            if let Err(e) = result {
                shared_error.set(e);
                return;
            }
        }
    }

    /// Check for a background thread error and return it.
    fn check_error(&self) -> io::Result<()> {
        if let Some(e) = self.shared_error.take() {
            Err(e)
        } else {
            Ok(())
        }
    }

    /// Send a message to the background thread.
    fn send(&self, msg: WriterMsg) -> io::Result<()> {
        self.sender
            .lock()
            .unwrap()
            .send(msg)
            .map_err(|_| match self.shared_error.take() {
                Some(e) => e,
                None => io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "rr-record-writer thread exited unexpectedly",
                ),
            })
    }

    /// Ship the current buffer to the background thread, replacing it with a
    /// recycled buffer if available (avoids heap allocation in the common case).
    fn ship_buffer(&mut self) -> io::Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        let replacement = self
            .recycler
            .lock()
            .unwrap()
            .try_recv()
            .unwrap_or_else(|_| Vec::with_capacity(self.buffer_capacity));
        let data = core::mem::replace(&mut self.buffer, replacement);
        self.send(WriterMsg::Buffer(data))
    }
}

impl Write for ThreadedWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.check_error()?;
        self.buffer.extend_from_slice(buf);
        if self.buffer.len() >= self.buffer_capacity {
            self.ship_buffer()?;
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.check_error()?;
        self.ship_buffer()?;
        self.send(WriterMsg::Flush)
    }
}

impl Drop for ThreadedWriter {
    fn drop(&mut self) {
        // Best-effort: ship remaining buffered data and signal shutdown.
        let _ = self.ship_buffer();
        let _ = self.send(WriterMsg::Shutdown);
        if let Some(handle) = self.join_handle.take()
            && let Err(e) = handle.join()
        {
            log::error!("rr-record-writer thread panicked: {e:?}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A writer backed by a shared Vec for verifying output.
    struct SharedVecWriter(Arc<Mutex<Vec<u8>>>);

    impl Write for SharedVecWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn basic_round_trip() {
        let shared = Arc::new(Mutex::new(Vec::new()));
        let writer = SharedVecWriter(Arc::clone(&shared));
        let config = ThreadedWriterConfig {
            buffer_capacity: 32,
            channels: 4,
        };
        let mut tw = ThreadedWriter::new(Box::new(writer), config);

        let data = b"hello, threaded writer!";
        tw.write_all(data).unwrap();
        tw.flush().unwrap();
        drop(tw);

        assert_eq!(&*shared.lock().unwrap(), data);
    }

    #[test]
    fn large_data_multiple_buffers() {
        let shared = Arc::new(Mutex::new(Vec::new()));
        let writer = SharedVecWriter(Arc::clone(&shared));
        let config = ThreadedWriterConfig {
            buffer_capacity: 64,
            channels: 2,
        };
        let mut tw = ThreadedWriter::new(Box::new(writer), config);

        // Write more data than buffer_capacity * channels
        let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        tw.write_all(&data).unwrap();
        tw.flush().unwrap();
        drop(tw);

        assert_eq!(&*shared.lock().unwrap(), &data);
    }

    #[test]
    fn error_propagation() {
        /// A writer that fails after a certain number of bytes.
        struct FailingWriter {
            remaining: usize,
        }
        impl Write for FailingWriter {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                if self.remaining == 0 {
                    return Err(io::Error::new(io::ErrorKind::Other, "disk full"));
                }
                let n = buf.len().min(self.remaining);
                self.remaining -= n;
                Ok(n)
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }

        let writer = FailingWriter { remaining: 16 };
        let config = ThreadedWriterConfig {
            buffer_capacity: 32,
            channels: 2,
        };
        let mut tw = ThreadedWriter::new(Box::new(writer), config);

        // Write enough data to trigger the background error
        let data = vec![0xABu8; 256];
        // The error may not appear immediately (it depends on scheduling),
        // but eventually writes or flush should fail.
        let mut errored = false;
        for _ in 0..100 {
            if tw.write_all(&data).is_err() || tw.flush().is_err() {
                errored = true;
                break;
            }
        }
        assert!(errored, "expected an error from the failing writer");
    }

    #[test]
    fn empty_write() {
        let shared = Arc::new(Mutex::new(Vec::new()));
        let writer = SharedVecWriter(Arc::clone(&shared));
        let config = ThreadedWriterConfig {
            buffer_capacity: 64,
            channels: 4,
        };
        let mut tw = ThreadedWriter::new(Box::new(writer), config);

        tw.flush().unwrap();
        drop(tw);

        assert!(shared.lock().unwrap().is_empty());
    }
}

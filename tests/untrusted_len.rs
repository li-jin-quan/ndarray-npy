//! Regression tests for allocations driven by untrusted values in the file.
//!
//! The header of an `.npy` file declares both the header length and the array
//! shape. These values are controlled by whoever wrote the file, so they must
//! not be used to size an allocation before checking that the file actually
//! contains that much data. Otherwise, a very small file could trigger a huge
//! allocation (CWE-770); when such an allocation fails, the process aborts.
//!
//! These tests assert that the largest single allocation is bounded by the
//! amount of data actually present in the file, rather than by the declared
//! values.
//!
//! Note: all the checks live in a single `#[test]` function because the
//! allocation tracker is process-global, so the checks must not run
//! concurrently with each other.

use ndarray::prelude::*;
use ndarray_npy::{ReadNpyExt, WriteNpyExt};
use std::alloc::{GlobalAlloc, Layout, System};
use std::io::Cursor;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Records the size of the largest single allocation request so that the tests
/// can assert the allocation is bounded by the real input size.
struct TrackingAlloc;

static MAX_REQUEST: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for TrackingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        MAX_REQUEST.fetch_max(layout.size(), Ordering::SeqCst);
        System.alloc(layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        MAX_REQUEST.fetch_max(layout.size(), Ordering::SeqCst);
        System.alloc_zeroed(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        MAX_REQUEST.fetch_max(new_size, Ordering::SeqCst);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static ALLOC: TrackingAlloc = TrackingAlloc;

fn max_request() -> usize {
    MAX_REQUEST.load(Ordering::SeqCst)
}

fn reset_max_request() {
    MAX_REQUEST.store(0, Ordering::SeqCst);
}

/// Builds a version 2.0 file whose declared header length (which may be up to
/// 4 GiB) is far larger than the number of bytes actually present.
fn file_with_declared_header_len(declared_len: u32) -> Vec<u8> {
    let mut v = Vec::new();
    v.extend_from_slice(b"\x93NUMPY");
    v.extend_from_slice(&[2, 0]);
    v.extend_from_slice(&declared_len.to_le_bytes());
    v.extend_from_slice(b"xxxxxxxx"); // far fewer bytes than declared
    v
}

/// Builds a version 1.0 file with a valid header whose declared shape implies
/// far more data than the file contains.
fn file_with_declared_shape(shape: &str) -> Vec<u8> {
    let header = format!(
        "{{'descr': '<f8', 'fortran_order': False, 'shape': {}, }}",
        shape
    );
    // The header is padded so that the total length is a multiple of 64.
    let prefix_len = 6 + 2 + 2; // magic string + version + `u16` header length
    let unpadded = prefix_len + header.len() + 1;
    let pad = 64 - (unpadded % 64);
    let header_len = header.len() + pad + 1; // including the trailing newline

    let mut v = Vec::new();
    v.extend_from_slice(b"\x93NUMPY");
    v.extend_from_slice(&[1, 0]);
    v.extend_from_slice(&(header_len as u16).to_le_bytes());
    v.extend_from_slice(header.as_bytes());
    v.extend(std::iter::repeat(b' ').take(pad));
    v.push(b'\n');
    v
}

/// The declared header length and shape must not drive allocation.
///
/// The sizes below (64 MiB and 8,000,000 elements) are chosen to be large
/// enough to detect the vulnerability but small enough that the allocation
/// succeeds, so that a regression is reported as an assertion failure rather
/// than as an aborted test process.
#[test]
fn allocation_is_bounded_by_data_actually_present() {
    // A file declaring a 64 MiB header, but containing only 8 bytes of it.
    let file = file_with_declared_header_len(64 * 1024 * 1024);
    reset_max_request();
    let res = Array1::<f64>::read_npy(Cursor::new(&file));
    assert!(res.is_err(), "a truncated file should fail to parse");
    assert!(
        max_request() < 1024 * 1024,
        "the declared header length should not drive allocation: a file of \
         {} bytes caused a single allocation of {} bytes",
        file.len(),
        max_request()
    );

    // A file declaring 8,000,000 `f64`s (64 MB of data), but containing none.
    let file = file_with_declared_shape("(8000000,)");
    reset_max_request();
    let res = Array1::<f64>::read_npy(Cursor::new(&file));
    assert!(res.is_err(), "a truncated file should fail to parse");
    assert!(
        max_request() < 1024 * 1024,
        "the declared shape should not drive allocation: a file of {} bytes \
         caused a single allocation of {} bytes",
        file.len(),
        max_request()
    );

    // Sanity check: a well-formed file still round-trips, and its allocation is
    // on the order of the real data size.
    let arr = Array1::<f64>::from_vec((0..1000).map(|i| i as f64).collect());
    let mut buf = Vec::new();
    arr.write_npy(&mut buf).unwrap();

    reset_max_request();
    let read: Array1<f64> = Array1::<f64>::read_npy(Cursor::new(&buf)).unwrap();
    assert_eq!(read, arr);
    // 1000 `f64`s are 8000 bytes; allow generous slack for the header and for
    // `Vec` growth.
    assert!(
        max_request() < 1024 * 1024,
        "unexpectedly large allocation of {} bytes for a {} byte file",
        max_request(),
        buf.len()
    );
}

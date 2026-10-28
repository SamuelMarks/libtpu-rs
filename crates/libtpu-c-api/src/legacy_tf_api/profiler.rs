//! Legacy TensorFlow TPU profiler (`TpuProfiler_*`) C API entrypoints.

use super::*;

/// Creates a [`TpuProfiler`] session (`TpuProfiler_Create`).
///
/// # Arguments
///
/// * `profiler_out` - Output pointer for the created [`TpuProfiler`].
/// * `status` - Optional [`TpuStatus`] pointer.
///
/// # Safety
///
/// `profiler_out` must be null or a valid mutable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuProfiler_Create(
    profiler_out: *mut *mut TpuProfiler,
    status: *mut TpuStatus,
) {
    if profiler_out.is_null() {
        // SAFETY: `status` may be null; `TpuStatus_Set` checks for null.
        unsafe {
            TpuStatus_Set(status, 3, c"profiler_out is null".as_ptr(), 20);
        }
        return;
    }
    let handle = Box::into_raw(Box::new(TpuProfiler {
        inner: TpuCoreProfiler::new(),
    }));
    // SAFETY: `profiler_out` is non-null.
    unsafe {
        *profiler_out = handle;
    }
}

/// Destroys a [`TpuProfiler`] session (`TpuProfiler_Destroy`).
///
/// # Arguments
///
/// * `profiler` - Pointer to [`TpuProfiler`].
///
/// # Safety
///
/// `profiler` must be null or allocated by `TpuProfiler_Create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuProfiler_Destroy(profiler: *mut TpuProfiler) {
    if !profiler.is_null() {
        // SAFETY: `profiler` is non-null.
        unsafe {
            drop(Box::from_raw(profiler));
        }
    }
}

/// Starts a [`TpuProfiler`] session (`TpuProfiler_Start`).
///
/// # Arguments
///
/// * `profiler` - Pointer to [`TpuProfiler`].
/// * `status` - Optional [`TpuStatus`] pointer.
///
/// # Safety
///
/// `profiler` and `status` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuProfiler_Start(profiler: *mut TpuProfiler, status: *mut TpuStatus) {
    if profiler.is_null() {
        // SAFETY: `TpuStatus_Set` handles null `status`.
        unsafe {
            TpuStatus_Set(status, 3, c"profiler is null".as_ptr(), 16);
        }
        return;
    }
    // SAFETY: `profiler` is non-null.
    if let Err(err) = unsafe { (*profiler).inner.start() } {
        let msg = to_c_string(&err.to_string());
        // SAFETY: `TpuStatus_Set` handles null `status`.
        unsafe {
            TpuStatus_Set(status, err.error_code().as_raw(), msg.as_ptr(), 0);
        }
    }
}

/// Stops a [`TpuProfiler`] session (`TpuProfiler_Stop`).
///
/// # Arguments
///
/// * `profiler` - Pointer to [`TpuProfiler`].
/// * `status` - Optional [`TpuStatus`] pointer.
///
/// # Safety
///
/// `profiler` and `status` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuProfiler_Stop(profiler: *mut TpuProfiler, status: *mut TpuStatus) {
    if profiler.is_null() {
        // SAFETY: `TpuStatus_Set` handles null `status`.
        unsafe {
            TpuStatus_Set(status, 3, c"profiler is null".as_ptr(), 16);
        }
        return;
    }
    // SAFETY: `profiler` is non-null.
    if let Err(err) = unsafe { (*profiler).inner.stop() } {
        let msg = to_c_string(&err.to_string());
        // SAFETY: `TpuStatus_Set` handles null `status`.
        unsafe {
            TpuStatus_Set(status, err.error_code().as_raw(), msg.as_ptr(), 0);
        }
    }
}

/// Collects serialized `XSpace` profiling data from `profiler` (`TpuProfiler_CollectData`).
///
/// # Arguments
///
/// * `profiler` - Pointer to [`TpuProfiler`].
/// * `status` - Optional [`TpuStatus`] pointer.
/// * `buffer` - Optional destination byte buffer (`NULL` to query required size).
/// * `size_in_bytes` - In/out buffer byte size pointer.
///
/// # Safety
///
/// All pointers must be null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuProfiler_CollectData(
    profiler: *mut TpuProfiler,
    status: *mut TpuStatus,
    buffer: *mut u8,
    size_in_bytes: *mut usize,
) {
    if any_null(&[profiler as usize, size_in_bytes as usize]) {
        // SAFETY: `TpuStatus_Set` handles null `status`.
        unsafe {
            TpuStatus_Set(status, 3, c"null profiler or size".as_ptr(), 21);
        }
        return;
    }
    // SAFETY: `profiler` is non-null.
    let bytes = unsafe { (*profiler).inner.collect_xspace_bytes() };
    // SAFETY: `size_in_bytes` is non-null.
    unsafe {
        *size_in_bytes = bytes.len();
    }
    if !buffer.is_null() {
        // SAFETY: Caller guarantees `buffer` has at least `bytes.len()` capacity.
        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr(), buffer, bytes.len());
        }
    }
}

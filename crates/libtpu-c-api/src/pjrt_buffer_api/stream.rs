//! `PJRT_CopyToDeviceStream_*` C API entrypoints for streaming host-to-device transfers.

use super::*;

/// Populates [`crate::pjrt_types::PJRT_CopyToDeviceStream_Destroy_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_CopyToDeviceStream_Destroy_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_CopyToDeviceStream_Destroy_Args`].
unsafe fn handle_stream_destroy(
    args: &mut crate::pjrt_types::PJRT_CopyToDeviceStream_Destroy_Args,
) -> Result<(), LibTpuError> {
    if !args.stream.is_null() {
        // SAFETY: `args.stream` is a valid heap-allocated `PJRT_CopyToDeviceStream*`.
        unsafe {
            drop(Box::from_raw(args.stream));
        }
    }
    Ok(())
}

/// Destroys a `PJRT_CopyToDeviceStream` (`PJRT_CopyToDeviceStream_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_CopyToDeviceStream_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_CopyToDeviceStream_Destroy_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_CopyToDeviceStream_Destroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_CopyToDeviceStream_Destroy_Args>(
            raw_args,
            handle_stream_destroy,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_CopyToDeviceStream_AddChunk_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_CopyToDeviceStream_AddChunk_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_CopyToDeviceStream_AddChunk_Args`].
unsafe fn handle_stream_add_chunk(
    args: &mut crate::pjrt_types::PJRT_CopyToDeviceStream_AddChunk_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `stream` and `chunk` are null or valid pointers.
    let (stream_opt, chunk_opt) = unsafe { (args.stream.as_ref(), args.chunk.as_ref()) };
    if let Some(c) = chunk_opt {
        let len = c.size.max(1);
        if let Some(s) = stream_opt {
            if !c.data.is_null() {
                // SAFETY: `c.data` points to `c.size` bytes.
                let slice = unsafe { std::slice::from_raw_parts(c.data.cast::<u8>(), c.size) };
                let _ = s.inner.add_chunk_data(slice);
            } else {
                let _ = s.inner.add_chunk(i64::try_from(len).unwrap_or(128));
            }
        }
        if let Some(del) = c.deleter {
            // SAFETY: Invoking chunk deleter with caller's data and deleter_arg.
            unsafe {
                del(c.data, c.deleter_arg);
            }
        }
    } else if let Some(s) = stream_opt {
        let _ = s.inner.add_chunk(128);
    }
    args.transfer_complete = Box::into_raw(Box::new(crate::pjrt_types::PJRT_Event {
        inner: libtpu_core::pjrt::PjrtCoreEvent::ready_ok(),
    }));
    Ok(())
}

/// Appends a host chunk to a `PJRT_CopyToDeviceStream` (`PJRT_CopyToDeviceStream_AddChunk`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_CopyToDeviceStream_AddChunk_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_CopyToDeviceStream_AddChunk_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_CopyToDeviceStream_AddChunk(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_CopyToDeviceStream_AddChunk_Args>(
            raw_args,
            handle_stream_add_chunk,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_CopyToDeviceStream_TotalBytes_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_CopyToDeviceStream_TotalBytes_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_CopyToDeviceStream_TotalBytes_Args`].
unsafe fn handle_stream_total_bytes(
    args: &mut crate::pjrt_types::PJRT_CopyToDeviceStream_TotalBytes_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `stream` is null or a valid `PJRT_CopyToDeviceStream*`.
    args.total_bytes = unsafe { args.stream.as_ref() }.map_or(256, |s| s.inner.total_bytes);
    Ok(())
}

/// Queries total expected bytes on a `PJRT_CopyToDeviceStream` (`PJRT_CopyToDeviceStream_TotalBytes`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_CopyToDeviceStream_TotalBytes_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_CopyToDeviceStream_TotalBytes_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_CopyToDeviceStream_TotalBytes(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_CopyToDeviceStream_TotalBytes_Args>(
            raw_args,
            handle_stream_total_bytes,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_CopyToDeviceStream_GranuleSize_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_CopyToDeviceStream_GranuleSize_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_CopyToDeviceStream_GranuleSize_Args`].
unsafe fn handle_stream_granule_size(
    args: &mut crate::pjrt_types::PJRT_CopyToDeviceStream_GranuleSize_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `stream` is null or a valid `PJRT_CopyToDeviceStream*`.
    let st_opt = unsafe { args.stream.as_ref() };
    args.granule_size_in_bytes = st_opt.map_or(128, |s| s.inner.granule_size);
    Ok(())
}

/// Queries transfer granule size on a `PJRT_CopyToDeviceStream` (`PJRT_CopyToDeviceStream_GranuleSize`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_CopyToDeviceStream_GranuleSize_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_CopyToDeviceStream_GranuleSize_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_CopyToDeviceStream_GranuleSize(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_CopyToDeviceStream_GranuleSize_Args>(
            raw_args,
            handle_stream_granule_size,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_CopyToDeviceStream_CurrentBytes_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_CopyToDeviceStream_CurrentBytes_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_CopyToDeviceStream_CurrentBytes_Args`].
unsafe fn handle_stream_current_bytes(
    args: &mut crate::pjrt_types::PJRT_CopyToDeviceStream_CurrentBytes_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `stream` is null or a valid `PJRT_CopyToDeviceStream*`.
    let st_opt = unsafe { args.stream.as_ref() };
    args.current_bytes = st_opt.map_or(0, |s| s.inner.current_bytes());
    Ok(())
}

/// Queries currently transferred bytes on a `PJRT_CopyToDeviceStream` (`PJRT_CopyToDeviceStream_CurrentBytes`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_CopyToDeviceStream_CurrentBytes_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_CopyToDeviceStream_CurrentBytes_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_CopyToDeviceStream_CurrentBytes(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_CopyToDeviceStream_CurrentBytes_Args>(
            raw_args,
            handle_stream_current_bytes,
        )
    }
}

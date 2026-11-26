//! PJRT Profiler (`PJRT_Profiler_Extension`) and `PLUGIN_Profiler_Api` (`104`-byte `0.0.53.1`) entrypoints.

use super::args::*;
use super::{dispatch_typed_extension_args, dispatch_typed_fallible_extension_args};
use crate::pjrt_helpers::{borrow_live_handle, box_registered_handle, take_live_handle};
use crate::pjrt_types::{
    PJRT_Error, PLUGIN_Profiler_ConsumeResult, PLUGIN_Profiler_Instance, PjrtFn,
};
use libtpu_core::driver::TpuCoreProfiler;
use libtpu_core::error::LibTpuError;
use std::ptr;

/// Strongly-typed handler for [`PLUGIN_Profiler_Error_Destroy`].
///
/// # Arguments
///
/// * `args` - Validated [`PLUGIN_Profiler_Error_Destroy_Args`] structure.
///
/// # Safety
///
/// `args.error` must be null or a valid [`PJRT_Error`] pointer.
unsafe fn handle_profiler_error_destroy(args: &mut PLUGIN_Profiler_Error_Destroy_Args) {
    // SAFETY: Delegated to `PJRT_Error_Destroy`.
    unsafe {
        let mut del_args = crate::pjrt_types::PJRT_Error_Destroy_Args {
            struct_size: size_of::<crate::pjrt_types::PJRT_Error_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            error: args.error,
        };
        crate::pjrt_error_api::PJRT_Error_Destroy(ptr::addr_of_mut!(del_args).cast());
    }
    args.error = ptr::null_mut();
}

/// Destroys a `PLUGIN_Profiler_Error` (`PLUGIN_Profiler_Error_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PLUGIN_Profiler_Error_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PLUGIN_Profiler_Error_Destroy_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PLUGIN_Profiler_Error_Destroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PLUGIN_Profiler_Error_Destroy_Args>(
            raw_args,
            handle_profiler_error_destroy,
        )
    }
}

/// Strongly-typed handler for [`PLUGIN_Profiler_Error_Message`].
///
/// # Arguments
///
/// * `args` - Validated [`PLUGIN_Profiler_Error_Message_Args`] structure.
///
/// # Safety
///
/// `args.error` must be null or a valid [`PJRT_Error`] pointer.
unsafe fn handle_profiler_error_message(args: &mut PLUGIN_Profiler_Error_Message_Args) {
    // SAFETY: Delegated to `PJRT_Error_Message`.
    unsafe {
        let mut msg_args = crate::pjrt_types::PJRT_Error_Message_Args {
            struct_size: size_of::<crate::pjrt_types::PJRT_Error_Message_Args>(),
            extension_start: ptr::null_mut(),
            error: args.error,
            message: ptr::null(),
            message_size: 0,
        };
        crate::pjrt_error_api::PJRT_Error_Message(ptr::addr_of_mut!(msg_args).cast());
        args.message = msg_args.message;
        args.message_size = msg_args.message_size;
    }
}

/// Queries the message of a `PLUGIN_Profiler_Error` (`PLUGIN_Profiler_Error_Message`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PLUGIN_Profiler_Error_Message_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PLUGIN_Profiler_Error_Message_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PLUGIN_Profiler_Error_Message(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PLUGIN_Profiler_Error_Message_Args>(
            raw_args,
            handle_profiler_error_message,
        )
    }
}

/// Strongly-typed handler for [`PLUGIN_Profiler_Error_GetCode`].
///
/// # Arguments
///
/// * `args` - Validated [`PLUGIN_Profiler_Error_GetCode_Args`] structure.
///
/// # Safety
///
/// `args.error` must be null or a valid [`PJRT_Error`] pointer.
unsafe fn handle_profiler_error_get_code(args: &mut PLUGIN_Profiler_Error_GetCode_Args) {
    // SAFETY: Delegated to `PJRT_Error_GetCode`.
    unsafe {
        let mut code_args = crate::pjrt_types::PJRT_Error_GetCode_Args {
            struct_size: size_of::<crate::pjrt_types::PJRT_Error_GetCode_Args>(),
            extension_start: ptr::null_mut(),
            error: args.error,
            code: 0,
        };
        let _ = crate::pjrt_error_api::PJRT_Error_GetCode(ptr::addr_of_mut!(code_args).cast());
        args.code = code_args.code;
    }
}

/// Queries the status code of a `PLUGIN_Profiler_Error` (`PLUGIN_Profiler_Error_GetCode`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PLUGIN_Profiler_Error_GetCode_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PLUGIN_Profiler_Error_GetCode_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PLUGIN_Profiler_Error_GetCode(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PLUGIN_Profiler_Error_GetCode_Args>(
            raw_args,
            handle_profiler_error_get_code,
        )
    }
}

/// Strongly-typed handler for [`PLUGIN_Profiler_Create`].
///
/// # Arguments
///
/// * `args` - Validated [`PLUGIN_Profiler_Create_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PLUGIN_Profiler_Create_Args`].
unsafe fn handle_profiler_create(args: &mut PLUGIN_Profiler_Create_Args) {
    args.profiler = box_registered_handle(PLUGIN_Profiler_Instance {
        inner: TpuCoreProfiler::new(),
        last_xspace: parking_lot::Mutex::new(Vec::new().into_boxed_slice()),
    });
}

/// Creates a `PLUGIN_Profiler` instance (`PLUGIN_Profiler_Create`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PLUGIN_Profiler_Create_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PLUGIN_Profiler_Create_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PLUGIN_Profiler_Create(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PLUGIN_Profiler_Create_Args>(
            raw_args,
            handle_profiler_create,
        )
    }
}

/// Strongly-typed handler for [`PLUGIN_Profiler_Destroy`].
///
/// # Arguments
///
/// * `args` - Validated [`PLUGIN_Profiler_Destroy_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PLUGIN_Profiler_Destroy_Args`].
unsafe fn handle_profiler_destroy(args: &mut PLUGIN_Profiler_Destroy_Args) {
    drop(take_live_handle(args.profiler));
    args.profiler = ptr::null_mut();
}

/// Destroys a `PLUGIN_Profiler` instance (`PLUGIN_Profiler_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PLUGIN_Profiler_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PLUGIN_Profiler_Destroy_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PLUGIN_Profiler_Destroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PLUGIN_Profiler_Destroy_Args>(
            raw_args,
            handle_profiler_destroy,
        )
    }
}

/// Strongly-typed handler for [`PLUGIN_Profiler_Start`].
///
/// # Arguments
///
/// * `args` - Validated [`PLUGIN_Profiler_Start_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PLUGIN_Profiler_Start_Args`].
unsafe fn handle_profiler_start(args: &mut PLUGIN_Profiler_Start_Args) {
    if let Some(inst) = borrow_live_handle(args.profiler.cast_const()) {
        let _ = inst.inner.start();
    }
}

/// Starts a `PLUGIN_Profiler` session (`PLUGIN_Profiler_Start`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PLUGIN_Profiler_Start_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PLUGIN_Profiler_Start_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PLUGIN_Profiler_Start(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PLUGIN_Profiler_Start_Args>(raw_args, handle_profiler_start)
    }
}

/// Strongly-typed handler for [`PLUGIN_Profiler_Stop`].
///
/// # Arguments
///
/// * `args` - Validated [`PLUGIN_Profiler_Stop_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PLUGIN_Profiler_Stop_Args`].
unsafe fn handle_profiler_stop(args: &mut PLUGIN_Profiler_Stop_Args) {
    if let Some(inst) = borrow_live_handle(args.profiler.cast_const()) {
        let _ = inst.inner.stop();
    }
}

/// Stops a `PLUGIN_Profiler` session (`PLUGIN_Profiler_Stop`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PLUGIN_Profiler_Stop_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PLUGIN_Profiler_Stop_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PLUGIN_Profiler_Stop(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PLUGIN_Profiler_Stop_Args>(raw_args, handle_profiler_stop)
    }
}

/// Strongly-typed handler for [`PLUGIN_Profiler_CollectData`].
///
/// # Arguments
///
/// * `args` - Validated [`PLUGIN_Profiler_CollectData_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PLUGIN_Profiler_CollectData_Args`].
unsafe fn handle_profiler_collect_data(args: &mut PLUGIN_Profiler_CollectData_Args) {
    let xspace_bytes = borrow_live_handle(args.profiler.cast_const()).map_or_else(
        || TpuCoreProfiler::new().collect_xspace_bytes(),
        |inst| {
            let _ = inst.inner.stop();
            inst.inner.collect_xspace_bytes()
        },
    );
    let xspace_len = xspace_bytes.len();
    if !args.size_in_bytes.is_null() {
        // SAFETY: `args.size_in_bytes` is non-null.
        unsafe {
            *args.size_in_bytes = xspace_len;
        }
    }
    if !args.buffer.is_null() {
        // SAFETY: `args.buffer` points to a caller buffer of at least `xspace_len` bytes.
        unsafe {
            ptr::copy_nonoverlapping(xspace_bytes.as_ptr(), args.buffer, xspace_len);
        }
    }
}

/// Collects serialized `XSpace` protobuf data from a `PLUGIN_Profiler` (`PLUGIN_Profiler_CollectData`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PLUGIN_Profiler_CollectData_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PLUGIN_Profiler_CollectData_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PLUGIN_Profiler_CollectData(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PLUGIN_Profiler_CollectData_Args>(
            raw_args,
            handle_profiler_collect_data,
        )
    }
}

/// Strongly-typed handler for [`PLUGIN_Profiler_Consume`] (`0.0.53.1`).
///
/// # Arguments
///
/// * `args` - Validated [`PLUGIN_Profiler_Consume_Args`] structure.
///
/// # Errors
///
/// Returns [`LibTpuError::InvalidArgument`] when `args.profiler` is non-null and unregistered.
///
/// # Safety
///
/// `args` must be a valid [`PLUGIN_Profiler_Consume_Args`].
unsafe fn handle_profiler_consume(
    args: &mut PLUGIN_Profiler_Consume_Args,
) -> Result<(), LibTpuError> {
    let xspace_bytes = if args.profiler.is_null() {
        TpuCoreProfiler::new().collect_xspace_bytes()
    } else {
        let inst = borrow_live_handle(args.profiler.cast_const()).ok_or_else(|| {
            LibTpuError::InvalidArgument {
                parameter: "profiler",
                reason: String::from("unregistered PLUGIN_Profiler_Instance handle"),
            }
        })?;
        let _ = inst.inner.stop();
        inst.inner.collect_xspace_bytes()
    };
    args.result = box_registered_handle(PLUGIN_Profiler_ConsumeResult {
        xspace_bytes: xspace_bytes.into_boxed_slice(),
    });
    Ok(())
}

/// Consumes collected trace data into an opaque [`PLUGIN_Profiler_ConsumeResult`] (`PLUGIN_Profiler_Consume`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PLUGIN_Profiler_Consume_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PLUGIN_Profiler_Consume_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PLUGIN_Profiler_Consume(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PLUGIN_Profiler_Consume_Args>(
            raw_args,
            handle_profiler_consume,
        )
    }
}

/// Strongly-typed handler for [`PLUGIN_Profiler_ConsumeResult_Destroy`] (`0.0.53.1`).
///
/// # Arguments
///
/// * `args` - Validated [`PLUGIN_Profiler_ConsumeResult_Destroy_Args`] structure.
///
/// # Errors
///
/// Returns [`LibTpuError::InvalidArgument`] when `args.result` is non-null and unregistered.
///
/// # Safety
///
/// `args` must be a valid [`PLUGIN_Profiler_ConsumeResult_Destroy_Args`].
unsafe fn handle_profiler_consume_result_destroy(
    args: &mut PLUGIN_Profiler_ConsumeResult_Destroy_Args,
) -> Result<(), LibTpuError> {
    if !args.result.is_null() && take_live_handle(args.result).is_none() {
        return Err(LibTpuError::InvalidArgument {
            parameter: "result",
            reason: String::from("unregistered PLUGIN_Profiler_ConsumeResult handle"),
        });
    }
    args.result = ptr::null_mut();
    Ok(())
}

/// Destroys a consumed trace snapshot handle (`PLUGIN_Profiler_ConsumeResult_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PLUGIN_Profiler_ConsumeResult_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PLUGIN_Profiler_ConsumeResult_Destroy_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PLUGIN_Profiler_ConsumeResult_Destroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PLUGIN_Profiler_ConsumeResult_Destroy_Args>(
            raw_args,
            handle_profiler_consume_result_destroy,
        )
    }
}

/// Strongly-typed handler for [`PLUGIN_Profiler_Serialize`] (`0.0.53.1`).
///
/// # Arguments
///
/// * `args` - Validated [`PLUGIN_Profiler_Serialize_Args`] structure.
///
/// # Errors
///
/// Returns [`LibTpuError::InvalidArgument`] when `args.result` is null or unregistered.
///
/// # Safety
///
/// `args` must be a valid [`PLUGIN_Profiler_Serialize_Args`].
unsafe fn handle_profiler_serialize(
    args: &mut PLUGIN_Profiler_Serialize_Args,
) -> Result<(), LibTpuError> {
    let res = borrow_live_handle(args.result).ok_or_else(|| LibTpuError::InvalidArgument {
        parameter: "result",
        reason: String::from("null or unregistered PLUGIN_Profiler_ConsumeResult handle"),
    })?;
    args.buffer = res.xspace_bytes.as_ptr();
    args.size_in_bytes = res.xspace_bytes.len();
    Ok(())
}

/// Exposes zero-copy serialized `XSpace` bytes from a [`PLUGIN_Profiler_ConsumeResult`] (`PLUGIN_Profiler_Serialize`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PLUGIN_Profiler_Serialize_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PLUGIN_Profiler_Serialize_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PLUGIN_Profiler_Serialize(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PLUGIN_Profiler_Serialize_Args>(
            raw_args,
            handle_profiler_serialize,
        )
    }
}

/// Plugin profiler vtable structure (`PLUGIN_Profiler_Api`, 104 bytes in `0.0.53.1`) pointed to by `EXT_18_PROFILER.profiler_api`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PLUGIN_Profiler_Api {
    /// Size of this struct (`104` / `0x68`).
    pub struct_size: usize,
    /// Reserved private pointer (`NULL`).
    pub priv_data: *mut std::ffi::c_void,
    /// Destroys a profiler error (`PLUGIN_Profiler_Error_Destroy`).
    pub error_destroy: PjrtFn,
    /// Queries a profiler error message (`PLUGIN_Profiler_Error_Message`).
    pub error_message: PjrtFn,
    /// Queries a profiler error code (`PLUGIN_Profiler_Error_GetCode`).
    pub error_get_code: PjrtFn,
    /// Creates a profiler session (`PLUGIN_Profiler_Create`).
    pub create: PjrtFn,
    /// Destroys a profiler session (`PLUGIN_Profiler_Destroy`).
    pub destroy: PjrtFn,
    /// Starts profiling (`PLUGIN_Profiler_Start`).
    pub start: PjrtFn,
    /// Stops profiling (`PLUGIN_Profiler_Stop`).
    pub stop: PjrtFn,
    /// Collects serialized `XSpace` profiling data (`PLUGIN_Profiler_CollectData`).
    pub collect_data: PjrtFn,
    /// Consumes collected trace data into an opaque snapshot (`PLUGIN_Profiler_Consume`).
    pub consume: PjrtFn,
    /// Destroys a consumed trace snapshot (`PLUGIN_Profiler_ConsumeResult_Destroy`).
    pub consume_result_destroy: PjrtFn,
    /// Serializes a consumed trace snapshot (`PLUGIN_Profiler_Serialize`).
    pub serialize: PjrtFn,
}

// SAFETY: Immutable static vtable structure.
unsafe impl Sync for PLUGIN_Profiler_Api {}

const _: () = assert!(size_of::<PLUGIN_Profiler_Api>() == 104);

/// Static `PLUGIN_Profiler_Api` vtable referenced by [`super::EXT_18_PROFILER`].
pub static GLOBAL_PLUGIN_PROFILER_API: PLUGIN_Profiler_Api = PLUGIN_Profiler_Api {
    struct_size: size_of::<PLUGIN_Profiler_Api>(),
    priv_data: ptr::null_mut(),
    error_destroy: PLUGIN_Profiler_Error_Destroy,
    error_message: PLUGIN_Profiler_Error_Message,
    error_get_code: PLUGIN_Profiler_Error_GetCode,
    create: PLUGIN_Profiler_Create,
    destroy: PLUGIN_Profiler_Destroy,
    start: PLUGIN_Profiler_Start,
    stop: PLUGIN_Profiler_Stop,
    collect_data: PLUGIN_Profiler_CollectData,
    consume: PLUGIN_Profiler_Consume,
    consume_result_destroy: PLUGIN_Profiler_ConsumeResult_Destroy,
    serialize: PLUGIN_Profiler_Serialize,
};

/// `profiler_c_api.h` v0.1 (`jaxlib` 0.4.x) `Create` trampoline (`32`-byte args without `extension_start`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to `{ struct_size, options, options_size, profiler_out }`.
///
/// # Safety
///
/// `raw_args` must be null or point to 4 pointer-sized words.
unsafe extern "C" fn negotiated_profiler_create(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    crate::ffi_catch_unwind(std::ptr::null_mut(), || {
        if !raw_args.is_null() {
            let inst = box_registered_handle(PLUGIN_Profiler_Instance {
                inner: TpuCoreProfiler::new(),
                last_xspace: parking_lot::Mutex::new(Vec::new().into_boxed_slice()),
            });
            // SAFETY: Offset `+0x18` (word 3) is `PLUGIN_Profiler** profiler` in 32-byte `Create_Args`.
            unsafe {
                *raw_args.cast::<*mut PLUGIN_Profiler_Instance>().add(3) = inst;
            }
        }
        std::ptr::null_mut()
    })
}

/// `profiler_c_api.h` v0.1 (`jaxlib` 0.4.x) `Destroy` trampoline (`16`-byte args `{ struct_size, profiler }`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to `{ struct_size, profiler }`.
///
/// # Safety
///
/// `raw_args` must be null or point to 2 pointer-sized words.
unsafe extern "C" fn negotiated_profiler_destroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    crate::ffi_catch_unwind(std::ptr::null_mut(), || {
        if !raw_args.is_null() {
            // SAFETY: Word 1 (`+0x08`) is `PLUGIN_Profiler*` in 16-byte `Destroy_Args`.
            let word1_ptr = unsafe { *raw_args.cast::<*mut PLUGIN_Profiler_Instance>().add(1) };
            drop(take_live_handle(word1_ptr));
        }
        std::ptr::null_mut()
    })
}

/// `profiler_c_api.h` v0.1 (`jaxlib` 0.4.x) `Start` trampoline (`16`-byte args `{ struct_size, profiler }`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to `{ struct_size, profiler }`.
///
/// # Safety
///
/// `raw_args` must be null or point to 2 pointer-sized words.
unsafe extern "C" fn negotiated_profiler_start(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    crate::ffi_catch_unwind(std::ptr::null_mut(), || {
        if !raw_args.is_null() {
            // SAFETY: Word 1 (`+0x08`) is `PLUGIN_Profiler*`.
            let word1_ptr = unsafe { *raw_args.cast::<*const PLUGIN_Profiler_Instance>().add(1) };
            if let Some(inst) = borrow_live_handle(word1_ptr) {
                let _ = inst.inner.start();
            }
        }
        std::ptr::null_mut()
    })
}

/// `profiler_c_api.h` v0.1 (`jaxlib` 0.4.x) `Stop` trampoline (`16`-byte args `{ struct_size, profiler }`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to `{ struct_size, profiler }`.
///
/// # Safety
///
/// `raw_args` must be null or point to 2 pointer-sized words.
unsafe extern "C" fn negotiated_profiler_stop(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    crate::ffi_catch_unwind(std::ptr::null_mut(), || {
        if !raw_args.is_null() {
            // SAFETY: Word 1 (`+0x08`) is `PLUGIN_Profiler*`.
            let word1_ptr = unsafe { *raw_args.cast::<*const PLUGIN_Profiler_Instance>().add(1) };
            if let Some(inst) = borrow_live_handle(word1_ptr) {
                let _ = inst.inner.stop();
            }
        }
        std::ptr::null_mut()
    })
}

/// `profiler_c_api.h` v0.1 (`jaxlib` 0.4.x) `CollectData` trampoline (`32`-byte args `{ struct_size, profiler, buffer, buffer_size_in_bytes }`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to `{ struct_size, profiler, buffer, buffer_size_in_bytes }`.
///
/// # Safety
///
/// `raw_args` must be null or point to 4 pointer-sized words.
unsafe extern "C" fn negotiated_profiler_collect_data(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    crate::ffi_catch_unwind(std::ptr::null_mut(), || {
        if !raw_args.is_null() {
            // SAFETY: Word 1 (`+0x08`) is `PLUGIN_Profiler*`.
            let word1_inst = unsafe { *raw_args.cast::<*const PLUGIN_Profiler_Instance>().add(1) };
            if let Some(inst) = borrow_live_handle(word1_inst) {
                let xspace = inst.inner.collect_xspace_bytes().into_boxed_slice();
                let len = xspace.len();
                let mut guard = inst.last_xspace.lock();
                *guard = xspace;
                let data_ptr = guard.as_ptr();
                drop(guard);
                // SAFETY: Writing `buffer` (`+0x10`) and `buffer_size_in_bytes` (`+0x18`) into 32-byte `CollectData_Args`.
                unsafe {
                    *raw_args.cast::<*const u8>().add(2) = data_ptr;
                    *raw_args.cast::<usize>().add(3) = len;
                }
            }
        }
        std::ptr::null_mut()
    })
}

/// Negotiated `PLUGIN_Profiler_Api` vtable compatible with `jaxlib` 0.4.x `PluginTracer` (`80` bytes through `collect_data`).
pub static NEGOTIATED_PLUGIN_PROFILER_API: PLUGIN_Profiler_Api = PLUGIN_Profiler_Api {
    struct_size: core::mem::offset_of!(PLUGIN_Profiler_Api, consume),
    priv_data: std::ptr::null_mut(),
    error_destroy: PLUGIN_Profiler_Error_Destroy,
    error_message: PLUGIN_Profiler_Error_Message,
    error_get_code: PLUGIN_Profiler_Error_GetCode,
    create: negotiated_profiler_create,
    destroy: negotiated_profiler_destroy,
    start: negotiated_profiler_start,
    stop: negotiated_profiler_stop,
    collect_data: negotiated_profiler_collect_data,
    consume: PLUGIN_Profiler_Consume,
    consume_result_destroy: PLUGIN_Profiler_ConsumeResult_Destroy,
    serialize: PLUGIN_Profiler_Serialize,
};

/// Negotiated `PJRT_Profiler_Extension` node (`ext_type = 1`) exposed when `LIBTPU_PJRT_API_MINOR_VERSION` < `PJRT_API_MINOR`.
pub static NEGOTIATED_EXT_PROFILER: super::PJRT_Profiler_Extension =
    super::PJRT_Profiler_Extension {
        base: crate::pjrt_types::PJRT_Extension_Base {
            struct_size: size_of::<super::PJRT_Profiler_Extension>(),
            ext_type: 1,
            pad: 0,
            next: std::ptr::null_mut(),
        },
        profiler_api: std::ptr::addr_of!(NEGOTIATED_PLUGIN_PROFILER_API),
        traceme_context_id: 0x0054_5055_5F58_5350,
    };

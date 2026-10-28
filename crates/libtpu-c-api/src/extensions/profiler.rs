//! PJRT Profiler (`PJRT_Profiler_Extension`) and `PLUGIN_Profiler_Api` entrypoints.

use super::args::*;
use super::{dispatch_typed_extension_args, extension_profiler};
use crate::pjrt_types::{PJRT_Error, PLUGIN_Profiler_Instance, PjrtFn};
use libtpu_core::driver::TpuCoreProfiler;
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
            error: args.error.cast::<PJRT_Error>(),
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
            error: args.error.cast::<PJRT_Error>(),
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
            error: args.error.cast::<PJRT_Error>(),
            code: 0,
        };
        let _ = crate::pjrt_error_api::PJRT_Error_GetCode(ptr::addr_of_mut!(code_args).cast());
        args.code = usize::try_from(code_args.code).unwrap_or(2);
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
    let _ = extension_profiler();
    let instance = Box::into_raw(Box::new(PLUGIN_Profiler_Instance {
        inner: TpuCoreProfiler::new(),
    }));
    args.profiler = instance.cast::<std::ffi::c_void>();
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
    if !args.profiler.is_null() {
        // SAFETY: `profiler` was allocated by `PLUGIN_Profiler_Create`.
        unsafe {
            drop(Box::from_raw(
                args.profiler.cast::<PLUGIN_Profiler_Instance>(),
            ));
        }
    }
    let _ = extension_profiler().lock().stop();
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
    if !args.profiler.is_null() {
        // SAFETY: `profiler` is a valid `PLUGIN_Profiler_Instance*`.
        let inst = unsafe { &*args.profiler.cast::<PLUGIN_Profiler_Instance>() };
        let _ = inst.inner.start();
    }
    let _ = extension_profiler().lock().start();
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
    if !args.profiler.is_null() {
        // SAFETY: `profiler` is a valid `PLUGIN_Profiler_Instance*`.
        let inst = unsafe { &*args.profiler.cast::<PLUGIN_Profiler_Instance>() };
        let _ = inst.inner.stop();
    }
    let _ = extension_profiler().lock().stop();
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
    let xspace_bytes = if args.profiler != 0 {
        // SAFETY: `profiler` is a valid `PLUGIN_Profiler_Instance*`.
        let inst = unsafe { &*(args.profiler as *const PLUGIN_Profiler_Instance) };
        let _ = inst.inner.stop();
        inst.inner.collect_xspace_bytes()
    } else {
        let prof = extension_profiler().lock();
        let _ = prof.stop();
        prof.collect_xspace_bytes()
    };
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

/// Plugin profiler vtable structure (`PLUGIN_Profiler_Api`, 80 bytes) pointed to by `EXT_17_PROFILER.profiler_api`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PLUGIN_Profiler_Api {
    /// Size of this struct (`80`).
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
}

// SAFETY: Immutable static vtable structure.
unsafe impl Sync for PLUGIN_Profiler_Api {}

const _: () = assert!(size_of::<PLUGIN_Profiler_Api>() == 80);

/// Static `PLUGIN_Profiler_Api` vtable referenced by [`super::EXT_17_PROFILER`].
pub static GLOBAL_PLUGIN_PROFILER_API: PLUGIN_Profiler_Api = PLUGIN_Profiler_Api {
    struct_size: 80,
    priv_data: ptr::null_mut(),
    error_destroy: PLUGIN_Profiler_Error_Destroy,
    error_message: PLUGIN_Profiler_Error_Message,
    error_get_code: PLUGIN_Profiler_Error_GetCode,
    create: PLUGIN_Profiler_Create,
    destroy: PLUGIN_Profiler_Destroy,
    start: PLUGIN_Profiler_Start,
    stop: PLUGIN_Profiler_Stop,
    collect_data: PLUGIN_Profiler_CollectData,
};

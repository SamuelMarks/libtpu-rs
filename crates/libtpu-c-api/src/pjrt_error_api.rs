//! PJRT error, plugin initialization, and event C API entrypoints (`PJRT_Error_*`, `PJRT_Plugin_*`, `PJRT_Event_*`).

use super::*;

/// Strongly-typed handler for [`PJRT_Error_Destroy`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Error_Destroy_Args`] structure.
///
/// # Safety
///
/// `args.error` must be null or a valid heap-allocated [`PJRT_Error`].
unsafe fn handle_error_destroy(args: &mut PJRT_Error_Destroy_Args) {
    if !args.error.is_null() {
        // SAFETY: `args.error` was allocated by `box_pjrt_error`.
        unsafe {
            drop(Box::from_raw(args.error));
        }
    }
}

/// Destroys a [`PJRT_Error`] handle (`PJRT_Error_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Error_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must be either null or a valid pointer to [`PJRT_Error_Destroy_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Error_Destroy(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_typed_pjrt_void_args`.
    unsafe {
        dispatch_typed_pjrt_void_args::<PJRT_Error_Destroy_Args>(raw_args, handle_error_destroy);
    }
}

/// Strongly-typed handler for [`PJRT_Error_Message`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Error_Message_Args`] structure.
///
/// # Safety
///
/// `args.error` must be null or a valid [`PJRT_Error`].
unsafe fn handle_error_message(args: &mut PJRT_Error_Message_Args) {
    // SAFETY: `args.error` is null or a valid `PJRT_Error*`.
    if let Some(err_ref) = unsafe { args.error.as_ref() } {
        let bytes = err_ref.c_message.as_bytes();
        args.message = err_ref.c_message.as_ptr();
        args.message_size = bytes.len();
    }
}

/// Populates the error message string of a [`PJRT_Error`] (`PJRT_Error_Message`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Error_Message_Args`].
///
/// # Safety
///
/// `raw_args` must be either null or a valid pointer to [`PJRT_Error_Message_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Error_Message(raw_args: *mut std::ffi::c_void) {
    // SAFETY: Validated by `dispatch_typed_pjrt_void_args`.
    unsafe {
        dispatch_typed_pjrt_void_args::<PJRT_Error_Message_Args>(raw_args, handle_error_message);
    }
}

/// Strongly-typed handler for [`PJRT_Error_GetCode`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Error_GetCode_Args`] structure.
///
/// # Safety
///
/// `args.error` must be null or a valid [`PJRT_Error`].
unsafe fn handle_error_get_code(args: &mut PJRT_Error_GetCode_Args) -> Result<(), LibTpuError> {
    // SAFETY: `args.error` is null or a valid `PJRT_Error*`.
    let err_opt = unsafe { args.error.as_ref() };
    let err_ref = err_opt.ok_or(LibTpuError::NullPointer { parameter: "error" })?;
    args.code = err_ref.error.error_code().as_raw();
    Ok(())
}

/// Queries the [`libtpu_core::error::PjrtErrorCode`] of a [`PJRT_Error`] (`PJRT_Error_GetCode`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Error_GetCode_Args`].
///
/// # Safety
///
/// `raw_args` must be either null or a valid pointer to [`PJRT_Error_GetCode_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Error_GetCode(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe { dispatch_typed_pjrt_args::<PJRT_Error_GetCode_Args>(raw_args, handle_error_get_code) }
}

/// Strongly-typed handler for [`PJRT_Plugin_Initialize`].
///
/// # Arguments
///
/// * `_args` - Validated [`PJRT_Plugin_Initialize_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Plugin_Initialize_Args`].
unsafe fn handle_plugin_initialize(
    _args: &mut PJRT_Plugin_Initialize_Args,
) -> Result<(), LibTpuError> {
    Ok(())
}

/// Initializes the PJRT TPU plugin (`PJRT_Plugin_Initialize`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Plugin_Initialize_Args`].
///
/// # Safety
///
/// `raw_args` must be either null or a valid pointer to [`PJRT_Plugin_Initialize_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Plugin_Initialize(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<PJRT_Plugin_Initialize_Args>(raw_args, handle_plugin_initialize)
    }
}

/// Populates [`crate::pjrt_types::PJRT_Plugin_Attributes_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Plugin_Attributes_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Plugin_Attributes_Args`].
unsafe fn handle_plugin_attributes(
    args: &mut crate::pjrt_types::PJRT_Plugin_Attributes_Args,
) -> Result<(), LibTpuError> {
    args.attributes = GLOBAL_PLUGIN_ATTRIBUTES.as_ptr();
    args.num_attributes = GLOBAL_PLUGIN_ATTRIBUTES.len();
    Ok(())
}

/// Queries plugin attributes (`PJRT_Plugin_Attributes`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Plugin_Attributes_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Plugin_Attributes_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Plugin_Attributes(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Plugin_Attributes_Args>(
            raw_args,
            handle_plugin_attributes,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Event_Destroy`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Event_Destroy_Args`] structure.
///
/// # Safety
///
/// `args.event` must be null or a heap-allocated `PJRT_Event*`.
unsafe fn handle_event_destroy(
    args: &mut crate::pjrt_types::PJRT_Event_Destroy_Args,
) -> Result<(), LibTpuError> {
    if !args.event.is_null() {
        // SAFETY: `args.event` was allocated as `Box<PJRT_Event>`.
        unsafe {
            drop(Box::from_raw(args.event));
        }
    }
    Ok(())
}

/// Destroys a `PJRT_Event` (`PJRT_Event_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Event_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Event_Destroy_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Event_Destroy(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Event_Destroy_Args>(
            raw_args,
            handle_event_destroy,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Event_IsReady_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Event_IsReady_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Event_IsReady_Args`].
unsafe fn handle_event_is_ready(
    args: &mut crate::pjrt_types::PJRT_Event_IsReady_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `args.event` is null or a valid `PJRT_Event*`.
    let ev_opt = unsafe { args.event.as_ref() };
    args.is_ready = ev_opt.is_none_or(|ev| ev.inner.is_ready());
    Ok(())
}

/// Checks whether a `PJRT_Event` is ready (`PJRT_Event_IsReady`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Event_IsReady_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Event_IsReady_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Event_IsReady(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Event_IsReady_Args>(
            raw_args,
            handle_event_is_ready,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Event_Error`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Event_Error_Args`] structure.
///
/// # Safety
///
/// `args.event` must be null or a valid `PJRT_Event*`.
unsafe fn handle_event_error(
    args: &mut crate::pjrt_types::PJRT_Event_Error_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `args.event` is null or a valid `PJRT_Event*`.
    if let Some(ev) = unsafe { args.event.as_ref() } {
        if let Some(ev_err) = ev.inner.get_error() {
            return Err(ev_err);
        }
    }
    Ok(())
}

/// Queries the error status of a `PJRT_Event` (`PJRT_Event_Error`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Event_Error_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Event_Error_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Event_Error(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Event_Error_Args>(
            raw_args,
            handle_event_error,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Event_Await`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Event_Await_Args`] structure.
///
/// # Safety
///
/// `args.event` must be null or a valid `PJRT_Event*`.
unsafe fn handle_event_await(
    args: &mut crate::pjrt_types::PJRT_Event_Await_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `args.event` is null or a valid `PJRT_Event*`.
    if let Some(ev) = unsafe { args.event.as_ref() } {
        ev.inner.await_ready()?;
    }
    Ok(())
}

/// Blocks until a `PJRT_Event` completes (`PJRT_Event_Await`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Event_Await_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Event_Await_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Event_Await(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Event_Await_Args>(
            raw_args,
            handle_event_await,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Event_OnReady`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Event_OnReady_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Event_OnReady_Args`].
unsafe fn handle_event_on_ready(
    args: &mut crate::pjrt_types::PJRT_Event_OnReady_Args,
) -> Result<(), LibTpuError> {
    if let Some(cb_fn) = args.callback {
        let user_arg_addr = args.user_arg as usize;
        // SAFETY: `args.event` is null or a valid `PJRT_Event*`.
        if let Some(ev) = unsafe { args.event.as_ref() } {
            ev.inner.on_ready(move |status| {
                let err_ptr = status.map_or(ptr::null_mut(), crate::box_pjrt_error);
                // SAFETY: Invoking caller's registered readiness callback with `user_arg`.
                unsafe {
                    cb_fn(err_ptr, user_arg_addr as *mut std::ffi::c_void);
                }
            });
        } else {
            // SAFETY: Invoking registered readiness callback when `args.event` is null.
            unsafe {
                cb_fn(ptr::null_mut(), args.user_arg);
            }
        }
    }
    Ok(())
}

/// Registers a completion callback on a `PJRT_Event` (`PJRT_Event_OnReady`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Event_OnReady_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Event_OnReady_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Event_OnReady(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Event_OnReady_Args>(
            raw_args,
            handle_event_on_ready,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Event_Create`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Event_Create_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Event_Create_Args`].
unsafe fn handle_event_create(
    args: &mut crate::pjrt_types::PJRT_Event_Create_Args,
) -> Result<(), LibTpuError> {
    args.event = Box::into_raw(Box::new(crate::pjrt_types::PJRT_Event {
        inner: libtpu_core::pjrt::PjrtCoreEvent::pending(),
    }));
    Ok(())
}

/// Creates a new user-controlled `PJRT_Event` (`PJRT_Event_Create`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Event_Create_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Event_Create_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Event_Create(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Event_Create_Args>(
            raw_args,
            handle_event_create,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Event_Set`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Event_Set_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Event_Set_Args`].
unsafe fn handle_event_set(
    args: &mut crate::pjrt_types::PJRT_Event_Set_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `args.event` is null or a valid `PJRT_Event*`.
    if let Some(ev) = unsafe { args.event.as_ref() } {
        let status = if args.error_code == 0 {
            None
        } else {
            let reason = if args.error_message.is_null() {
                format!("event error code {}", args.error_code)
            } else {
                // SAFETY: `args.error_message` is a null-terminated C-string when non-null.
                unsafe { CStr::from_ptr(args.error_message) }
                    .to_string_lossy()
                    .into_owned()
            };
            Some(LibTpuError::InvalidArgument {
                parameter: "event",
                reason,
            })
        };
        ev.inner.set(status);
    }
    Ok(())
}

/// Marks a user-controlled `PJRT_Event` as complete (`PJRT_Event_Set`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Event_Set_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Event_Set_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Event_Set(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Event_Set_Args>(
            raw_args,
            handle_event_set,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Error_ForEachPayload`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Error_ForEachPayload_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Error_ForEachPayload_Args`].
unsafe fn handle_error_for_each_payload(
    args: &mut crate::pjrt_types::PJRT_Error_ForEachPayload_Args,
) -> Result<(), LibTpuError> {
    if let (false, Some(visitor)) = (args.error.is_null(), args.visitor) {
        // SAFETY: `args.error` is a valid `PJRT_Error*`.
        let payloads = unsafe { (*args.error).error.payloads() };
        for payload in payloads {
            // SAFETY: Invoking caller's payload visitor callback.
            unsafe {
                visitor(
                    payload.key.as_ptr().cast::<c_char>(),
                    payload.key.len(),
                    payload.value.as_ptr().cast::<c_char>(),
                    payload.value.len(),
                    args.user_arg,
                );
            }
        }
    }
    Ok(())
}

/// Iterates over structured payloads attached to a `PJRT_Error` (`PJRT_Error_ForEachPayload`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Error_ForEachPayload_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Error_ForEachPayload_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Error_ForEachPayload(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Error_ForEachPayload_Args>(
            raw_args,
            handle_error_for_each_payload,
        )
    }
}

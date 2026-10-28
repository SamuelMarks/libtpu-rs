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

/// Iterates over key-value metadata payloads attached to a [`PJRT_Error`] (`PJRT_Error_ForEachPayload`).
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

#[cfg(test)]
#[allow(
    clippy::cognitive_complexity,
    clippy::too_many_lines,
    clippy::items_after_statements
)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::pjrt_types::{
        PJRT_Event_Await_Args, PJRT_Event_Create_Args, PJRT_Event_Destroy_Args,
        PJRT_Event_Error_Args, PJRT_Event_IsReady_Args, PJRT_Event_OnReady_Args,
        PJRT_Event_Set_Args, PJRT_Plugin_Attributes_Args,
    };

    #[test]
    fn test_pjrt_error_plugin_and_event_api() {
        // SAFETY: Testing PJRT error, plugin, and event entrypoints with null and valid arguments.
        unsafe {
            PJRT_Error_Destroy(ptr::null_mut());
            let mut destroy_null_err = PJRT_Error_Destroy_Args {
                struct_size: 16,
                extension_start: ptr::null_mut(),
                error: ptr::null_mut(),
            };
            assert_ne!(format!("{destroy_null_err:?}"), "");
            PJRT_Error_Destroy(ptr::addr_of_mut!(destroy_null_err).cast());

            PJRT_Error_Message(ptr::null_mut());
            let mut msg_null_err = PJRT_Error_Message_Args {
                struct_size: 32,
                extension_start: ptr::null_mut(),
                error: ptr::null(),
                message: ptr::null(),
                message_size: 0,
            };
            assert_ne!(format!("{msg_null_err:?}"), "");
            PJRT_Error_Message(ptr::addr_of_mut!(msg_null_err).cast());

            let null_code_err = PJRT_Error_GetCode(ptr::null_mut());
            assert!(!null_code_err.is_null());
            let mut msg_args = PJRT_Error_Message_Args {
                struct_size: 32,
                extension_start: ptr::null_mut(),
                error: null_code_err,
                message: ptr::null(),
                message_size: 0,
            };
            PJRT_Error_Message(ptr::addr_of_mut!(msg_args).cast());
            assert!(msg_args.message_size > 0);
            assert_ne!(format!("{:?}", *null_code_err), "");

            let mut code_args = PJRT_Error_GetCode_Args {
                struct_size: size_of::<PJRT_Error_GetCode_Args>(),
                extension_start: ptr::null_mut(),
                error: null_code_err,
                code: 0,
            };
            assert_ne!(format!("{code_args:?}"), "");
            assert!(PJRT_Error_GetCode(ptr::addr_of_mut!(code_args).cast()).is_null());
            assert_eq!(code_args.code, 3);

            let mut code_null_inner = PJRT_Error_GetCode_Args {
                struct_size: size_of::<PJRT_Error_GetCode_Args>(),
                extension_start: ptr::null_mut(),
                error: ptr::null(),
                code: 0,
            };
            let err2 = PJRT_Error_GetCode(ptr::addr_of_mut!(code_null_inner).cast());
            let mut d2 = PJRT_Error_Destroy_Args {
                struct_size: 16,
                extension_start: ptr::null_mut(),
                error: err2,
            };
            PJRT_Error_Destroy(ptr::addr_of_mut!(d2).cast());

            let mut d1 = PJRT_Error_Destroy_Args {
                struct_size: 16,
                extension_start: ptr::null_mut(),
                error: null_code_err,
            };
            PJRT_Error_Destroy(ptr::addr_of_mut!(d1).cast());

            let e_init = PJRT_Plugin_Initialize(ptr::null_mut());
            drop(Box::from_raw(e_init));
            let mut init_args = PJRT_Plugin_Initialize_Args {
                struct_size: 16,
                extension_start: ptr::null_mut(),
            };
            assert_ne!(format!("{init_args:?}"), "");
            assert!(PJRT_Plugin_Initialize(ptr::addr_of_mut!(init_args).cast()).is_null());

            let mut plug_attr_args = PJRT_Plugin_Attributes_Args {
                struct_size: 16,
                extension_start: ptr::null_mut(),
                attributes: ptr::null(),
                num_attributes: 0,
            };
            drop(Box::from_raw(PJRT_Plugin_Attributes(
                ptr::addr_of_mut!(plug_attr_args).cast(),
            )));
            plug_attr_args.struct_size = size_of::<PJRT_Plugin_Attributes_Args>();
            assert!(PJRT_Plugin_Attributes(ptr::addr_of_mut!(plug_attr_args).cast()).is_null());
            assert_eq!(plug_attr_args.num_attributes, 3);
            assert!(!plug_attr_args.attributes.is_null());
            assert_ne!(format!("{:?}", *plug_attr_args.attributes), "");

            drop(Box::from_raw(PJRT_Event_Create(ptr::null_mut())));
            drop(Box::from_raw(PJRT_Event_Set(ptr::null_mut())));
            drop(Box::from_raw(PJRT_Event_IsReady(ptr::null_mut())));
            drop(Box::from_raw(PJRT_Event_Error(ptr::null_mut())));
            drop(Box::from_raw(PJRT_Event_Await(ptr::null_mut())));
            drop(Box::from_raw(PJRT_Event_OnReady(ptr::null_mut())));
            drop(Box::from_raw(PJRT_Event_Destroy(ptr::null_mut())));
            drop(Box::from_raw(PJRT_Error_ForEachPayload(ptr::null_mut())));

            let mut ev_create_args = PJRT_Event_Create_Args {
                struct_size: size_of::<PJRT_Event_Create_Args>(),
                extension_start: ptr::null_mut(),
                event: ptr::null_mut(),
            };
            assert!(PJRT_Event_Create(ptr::addr_of_mut!(ev_create_args).cast()).is_null());
            let created_ev = ev_create_args.event;
            assert!(!created_ev.is_null());

            let mut ev_ready_args = PJRT_Event_IsReady_Args {
                struct_size: size_of::<PJRT_Event_IsReady_Args>(),
                extension_start: ptr::null_mut(),
                event: created_ev,
                is_ready: true,
                pad: [0; 7],
            };
            assert!(PJRT_Event_IsReady(ptr::addr_of_mut!(ev_ready_args).cast()).is_null());
            assert!(!ev_ready_args.is_ready);

            let mut ev_set_args = PJRT_Event_Set_Args {
                struct_size: size_of::<PJRT_Event_Set_Args>(),
                extension_start: ptr::null_mut(),
                event: created_ev,
                error_code: 0,
                pad: 0,
                error_message: ptr::null(),
                error_message_size: 0,
            };
            assert!(PJRT_Event_Set(ptr::addr_of_mut!(ev_set_args).cast()).is_null());
            assert!(PJRT_Event_IsReady(ptr::addr_of_mut!(ev_ready_args).cast()).is_null());
            assert!(ev_ready_args.is_ready);

            let mut ev_err_args = PJRT_Event_Error_Args {
                struct_size: size_of::<PJRT_Event_Error_Args>(),
                extension_start: ptr::null_mut(),
                event: created_ev,
            };
            assert!(PJRT_Event_Error(ptr::addr_of_mut!(ev_err_args).cast()).is_null());
            let mut ev_await_args = PJRT_Event_Await_Args {
                struct_size: size_of::<PJRT_Event_Await_Args>(),
                extension_start: ptr::null_mut(),
                event: created_ev,
            };
            assert!(PJRT_Event_Await(ptr::addr_of_mut!(ev_await_args).cast()).is_null());

            ev_set_args.error_code = 3;
            ev_set_args.error_message = ptr::null();
            assert!(PJRT_Event_Set(ptr::addr_of_mut!(ev_set_args).cast()).is_null());
            drop(Box::from_raw(PJRT_Event_Error(
                ptr::addr_of_mut!(ev_err_args).cast(),
            )));
            drop(Box::from_raw(PJRT_Event_Await(
                ptr::addr_of_mut!(ev_await_args).cast(),
            )));
            ev_set_args.error_message = c"custom event failure".as_ptr();
            assert!(PJRT_Event_Set(ptr::addr_of_mut!(ev_set_args).cast()).is_null());

            unsafe extern "C" fn test_on_ready_cb(
                err: *mut PJRT_Error,
                user_arg: *mut std::ffi::c_void,
            ) {
                if !err.is_null() {
                    // SAFETY: `err` was boxed by `box_pjrt_error`.
                    unsafe {
                        drop(Box::from_raw(err));
                    }
                }
                // SAFETY: `user_arg` points to a mutable `u32` counter in this test.
                unsafe {
                    *user_arg.cast::<u32>() += 1;
                }
            }
            let mut cb_hits: u32 = 0;
            let mut on_ready_args = PJRT_Event_OnReady_Args {
                struct_size: size_of::<PJRT_Event_OnReady_Args>(),
                extension_start: ptr::null_mut(),
                event: created_ev,
                callback: Some(test_on_ready_cb),
                user_arg: ptr::addr_of_mut!(cb_hits).cast(),
            };
            assert!(PJRT_Event_OnReady(ptr::addr_of_mut!(on_ready_args).cast()).is_null());
            assert_eq!(cb_hits, 1);
            on_ready_args.event = ptr::null_mut();
            assert!(PJRT_Event_OnReady(ptr::addr_of_mut!(on_ready_args).cast()).is_null());
            assert_eq!(cb_hits, 2);

            let mut ev_create2 = PJRT_Event_Create_Args {
                struct_size: size_of::<PJRT_Event_Create_Args>(),
                extension_start: ptr::null_mut(),
                event: ptr::null_mut(),
            };
            assert!(PJRT_Event_Create(ptr::addr_of_mut!(ev_create2).cast()).is_null());
            on_ready_args.event = ev_create2.event;
            assert!(PJRT_Event_OnReady(ptr::addr_of_mut!(on_ready_args).cast()).is_null());
            assert_eq!(cb_hits, 2);
            ev_set_args.event = ev_create2.event;
            ev_set_args.error_code = 0;
            ev_set_args.error_message = ptr::null();
            assert!(PJRT_Event_Set(ptr::addr_of_mut!(ev_set_args).cast()).is_null());
            assert_eq!(cb_hits, 3);
            let mut ev_del2 = PJRT_Event_Destroy_Args {
                struct_size: size_of::<PJRT_Event_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                event: ev_create2.event,
            };
            assert!(PJRT_Event_Destroy(ptr::addr_of_mut!(ev_del2).cast()).is_null());

            let mut ev_del_args = PJRT_Event_Destroy_Args {
                struct_size: size_of::<PJRT_Event_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                event: created_ev,
            };
            assert!(PJRT_Event_Destroy(ptr::addr_of_mut!(ev_del_args).cast()).is_null());

            unsafe extern "C" fn test_payload_visitor(
                _k: *const c_char,
                _k_len: usize,
                _v: *const c_char,
                _v_len: usize,
                user_arg: *mut std::ffi::c_void,
            ) {
                // SAFETY: `user_arg` points to a mutable `u32` counter in this test.
                unsafe {
                    *user_arg.cast::<u32>() += 1;
                }
            }
            let sample_err = box_pjrt_error(LibTpuError::NullPointer { parameter: "test" });
            let mut visited_count: u32 = 0;
            let mut fep_args = crate::pjrt_types::PJRT_Error_ForEachPayload_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Error_ForEachPayload_Args>(),
                extension_start: ptr::null_mut(),
                error: sample_err,
                visitor: Some(test_payload_visitor),
                user_arg: ptr::addr_of_mut!(visited_count).cast(),
            };
            assert!(PJRT_Error_ForEachPayload(ptr::addr_of_mut!(fep_args).cast()).is_null());
            assert!(visited_count >= 2);
            drop(Box::from_raw(sample_err));
        }
    }
}

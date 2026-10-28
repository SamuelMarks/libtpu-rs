//! `PJRT_ExecuteContext_*` C API entrypoints for execution context creation and destruction.

use super::*;

/// Populates [`crate::pjrt_types::PJRT_ExecuteContext_Create_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_ExecuteContext_Create_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_ExecuteContext_Create_Args`].
unsafe fn handle_execute_context_create(
    args: &mut crate::pjrt_types::PJRT_ExecuteContext_Create_Args,
) -> Result<(), LibTpuError> {
    args.context = Box::into_raw(Box::new(crate::pjrt_types::PJRT_ExecuteContext {
        context_id: 1,
    }));
    Ok(())
}

/// Creates a `PJRT_ExecuteContext` (`PJRT_ExecuteContext_Create`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_ExecuteContext_Create_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_ExecuteContext_Create_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_ExecuteContext_Create(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_ExecuteContext_Create_Args>(
            raw_args,
            handle_execute_context_create,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_ExecuteContext_Destroy_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_ExecuteContext_Destroy_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_ExecuteContext_Destroy_Args`].
unsafe fn handle_execute_context_destroy(
    args: &mut crate::pjrt_types::PJRT_ExecuteContext_Destroy_Args,
) -> Result<(), LibTpuError> {
    if !args.context.is_null() {
        // SAFETY: `args.context` was allocated by `PJRT_ExecuteContext_Create`.
        unsafe {
            drop(Box::from_raw(args.context));
        }
    }
    Ok(())
}

/// Destroys a `PJRT_ExecuteContext` (`PJRT_ExecuteContext_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_ExecuteContext_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_ExecuteContext_Destroy_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_ExecuteContext_Destroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_ExecuteContext_Destroy_Args>(
            raw_args,
            handle_execute_context_destroy,
        )
    }
}

//! `LibtpuSdk_Api` (`tpumonitoring`, `tpuz`, `debug`, `slice`) C API entrypoints.

use super::*;

/// Slot 1 of `LibtpuSdk_Api`: populates the message of a [`LibtpuSdk_Error`] (`LibtpuSdk_Error_GetMessage`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`LibtpuSdk_Error_GetMessage_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`LibtpuSdk_Error_GetMessage_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LibtpuSdk_Error_GetMessage(raw_args: *mut std::ffi::c_void) {
    let args_ptr = raw_args.cast::<LibtpuSdk_Error_GetMessage_Args>();
    if args_ptr.is_null() {
        return;
    }
    // SAFETY: `args_ptr` is non-null.
    let err_ptr = unsafe { (*args_ptr).error };
    if err_ptr.is_null() {
        return;
    }
    // SAFETY: `err_ptr` is non-null.
    let err_ref = unsafe { &*err_ptr };
    let bytes = err_ref.c_message.as_bytes();
    // SAFETY: Writing output message fields.
    unsafe {
        (*args_ptr).message = err_ref.c_message.as_ptr();
        (*args_ptr).message_size = bytes.len();
    }
}

/// Slot 2 of `LibtpuSdk_Api`: destroys a [`LibtpuSdk_Error`] (`LibtpuSdk_Error_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`LibtpuSdk_Error_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`LibtpuSdk_Error_Destroy_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LibtpuSdk_Error_Destroy(raw_args: *mut std::ffi::c_void) {
    let args_ptr = raw_args.cast::<LibtpuSdk_Error_Destroy_Args>();
    if args_ptr.is_null() {
        return;
    }
    // SAFETY: `args_ptr` is non-null.
    let err_ptr = unsafe { (*args_ptr).error };
    if !err_ptr.is_null() {
        // SAFETY: `err_ptr` was allocated by `box_sdk_error`.
        unsafe {
            drop(Box::from_raw(err_ptr));
        }
    }
}

/// Slot 3 of `LibtpuSdk_Api`: queries the error code of a [`LibtpuSdk_Error`] (`LibtpuSdk_Error_GetCode`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`LibtpuSdk_Error_GetCode_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`LibtpuSdk_Error_GetCode_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LibtpuSdk_Error_GetCode(
    raw_args: *mut std::ffi::c_void,
) -> *mut LibtpuSdk_Error {
    let args_ptr = raw_args.cast::<LibtpuSdk_Error_GetCode_Args>();
    if args_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer { parameter: "args" });
    }
    // SAFETY: `args_ptr` is non-null.
    let err_ptr = unsafe { (*args_ptr).error };
    if err_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer { parameter: "error" });
    }
    // SAFETY: `err_ptr` is non-null.
    unsafe {
        (*args_ptr).code = (*err_ptr).code;
    }
    ptr::null_mut()
}

/// Slot 4 of `LibtpuSdk_Api`: creates a new [`LibtpuSdk_Client`] (`LibtpuSdk_Client_Create`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`LibtpuSdk_Client_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`LibtpuSdk_Client_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LibtpuSdk_Client_Create(
    raw_args: *mut std::ffi::c_void,
) -> *mut LibtpuSdk_Error {
    let args_ptr = raw_args.cast::<LibtpuSdk_Client_Args>();
    if args_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer { parameter: "args" });
    }
    let inner = LibtpuSdkCoreClient::new();
    let mut host_names = Vec::with_capacity(inner.chip_coordinates.len());
    let mut coords_entries = Vec::with_capacity(inner.chip_coordinates.len());
    for (idx, c) in inner.chip_coordinates.iter().enumerate() {
        let cs = to_c_string(&inner.hostname);
        let host_name_ptr = cs.as_ptr();
        let host_name_len = cs.as_bytes().len();
        host_names.push(cs);
        coords_entries.push(LibtpuSdk_ChipLocationToCoordinates {
            host_name_ptr,
            host_name_len,
            chip_index: i32::try_from(idx).unwrap_or(0),
            pad: 0,
            coordinates: [i64::from(c.x), i64::from(c.y), i64::from(c.z)],
        });
    }
    let client = Box::into_raw(Box::new(LibtpuSdk_Client {
        inner,
        host_names,
        coords_entries,
    }));
    // SAFETY: `args_ptr` is non-null.
    unsafe {
        (*args_ptr).client = client;
    }
    ptr::null_mut()
}

/// Slot 5 of `LibtpuSdk_Api`: destroys a [`LibtpuSdk_Client`] (`LibtpuSdk_Client_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`LibtpuSdk_Client_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`LibtpuSdk_Client_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LibtpuSdk_Client_Destroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut LibtpuSdk_Error {
    let args_ptr = raw_args.cast::<LibtpuSdk_Client_Args>();
    if args_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer { parameter: "args" });
    }
    // SAFETY: `args_ptr` is non-null.
    let client_ptr = unsafe { (*args_ptr).client };
    if client_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer {
            parameter: "client",
        });
    }
    // SAFETY: `client_ptr` was allocated in `LibtpuSdk_Client_Create`.
    unsafe {
        drop(Box::from_raw(client_ptr));
    }
    ptr::null_mut()
}

/// Slot 6 of `LibtpuSdk_Api`: queries all chip coordinate records (`LibtpuSdk_GetChipCoordinates`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`LibtpuSdk_GetChipCoordinates_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`LibtpuSdk_GetChipCoordinates_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LibtpuSdk_GetChipCoordinates(
    raw_args: *mut std::ffi::c_void,
) -> *mut LibtpuSdk_Error {
    let args_ptr = raw_args.cast::<LibtpuSdk_GetChipCoordinates_Args>();
    if args_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer { parameter: "args" });
    }
    // SAFETY: `args_ptr` is non-null.
    let client_ptr = unsafe { (*args_ptr).client };
    if client_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer {
            parameter: "client",
        });
    }
    // SAFETY: `client_ptr` is non-null.
    let client_ref = unsafe { &*client_ptr };
    // SAFETY: Writing output array pointer and length.
    unsafe {
        (*args_ptr).coords = client_ref.coords_entries.as_ptr();
        (*args_ptr).num_coords = client_ref.coords_entries.len();
    }
    ptr::null_mut()
}

/// Slot 7 of `LibtpuSdk_Api`: queries the hostname of a chip coordinate entry (`LibtpuSdk_GetHostName`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`LibtpuSdk_GetHostName_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`LibtpuSdk_GetHostName_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LibtpuSdk_GetHostName(
    raw_args: *mut std::ffi::c_void,
) -> *mut LibtpuSdk_Error {
    let args_ptr = raw_args.cast::<LibtpuSdk_GetHostName_Args>();
    if args_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer { parameter: "args" });
    }
    // SAFETY: `args_ptr` is non-null.
    let coord_ptr = unsafe { (*args_ptr).coord };
    if coord_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer { parameter: "coord" });
    }
    // SAFETY: `coord_ptr` is non-null.
    let coord_ref = unsafe { &*coord_ptr };
    // SAFETY: Writing output hostname fields.
    unsafe {
        (*args_ptr).host_name = coord_ref.host_name_ptr;
        (*args_ptr).host_name_size = coord_ref.host_name_len;
    }
    ptr::null_mut()
}

/// Slot 8 of `LibtpuSdk_Api`: queries the chip index of a chip coordinate entry (`LibtpuSdk_GetChipIndex`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`LibtpuSdk_GetChipIndex_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`LibtpuSdk_GetChipIndex_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LibtpuSdk_GetChipIndex(
    raw_args: *mut std::ffi::c_void,
) -> *mut LibtpuSdk_Error {
    let args_ptr = raw_args.cast::<LibtpuSdk_GetChipIndex_Args>();
    if args_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer { parameter: "args" });
    }
    // SAFETY: `args_ptr` is non-null.
    let coord_ptr = unsafe { (*args_ptr).coord };
    if coord_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer { parameter: "coord" });
    }
    // SAFETY: `coord_ptr` is non-null.
    unsafe {
        (*args_ptr).chip_index = (*coord_ptr).chip_index;
    }
    ptr::null_mut()
}

/// Slot 9 of `LibtpuSdk_Api`: queries the Cartesian `[x, y, z]` coordinates (`LibtpuSdk_GetCartesianCoordinates`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`LibtpuSdk_GetCartesianCoordinates_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`LibtpuSdk_GetCartesianCoordinates_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LibtpuSdk_GetCartesianCoordinates(
    raw_args: *mut std::ffi::c_void,
) -> *mut LibtpuSdk_Error {
    let args_ptr = raw_args.cast::<LibtpuSdk_GetCartesianCoordinates_Args>();
    if args_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer { parameter: "args" });
    }
    // SAFETY: `args_ptr` is non-null.
    let coord_ptr = unsafe { (*args_ptr).coord };
    if coord_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer { parameter: "coord" });
    }
    // SAFETY: `coord_ptr` is non-null.
    let coord_ref = unsafe { &*coord_ptr };
    // SAFETY: Writing output coordinates pointer and length.
    unsafe {
        (*args_ptr).coordinates = coord_ref.coordinates.as_ptr();
        (*args_ptr).num_coordinates = coord_ref.coordinates.len();
    }
    ptr::null_mut()
}

/// Slot 10 of `LibtpuSdk_Api`: queries a telemetry metric by name (`LibtpuSdk_GetMetric`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`LibtpuSdk_GetMetric_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`LibtpuSdk_GetMetric_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LibtpuSdk_GetMetric(
    raw_args: *mut std::ffi::c_void,
) -> *mut LibtpuSdk_Error {
    let args_ptr = raw_args.cast::<LibtpuSdk_GetMetric_Args>();
    if args_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer { parameter: "args" });
    }
    // SAFETY: `args_ptr` is non-null.
    let (client_ptr, name_ptr) = unsafe { ((*args_ptr).client, (*args_ptr).metric_name) };
    if any_null(&[client_ptr as usize, name_ptr as usize]) {
        return box_sdk_error(&LibTpuError::NullPointer {
            parameter: "client_or_metric_name",
        });
    }
    // SAFETY: `name_ptr` is non-null.
    let metric_name = unsafe { CStr::from_ptr(name_ptr).to_string_lossy() };
    // SAFETY: `client_ptr` is non-null.
    let client_ref = unsafe { &*client_ptr };
    match client_ref.inner.get_metric(&metric_name) {
        Ok((desc, vals)) => {
            let description = to_c_string(&desc);
            let value_cstrings: Vec<CString> =
                vals.iter().map(|v| to_c_string(&v.to_string())).collect();
            let value_ptrs: Vec<*const c_char> =
                value_cstrings.iter().map(|cs| cs.as_ptr()).collect();
            let metric_handle = Box::into_raw(Box::new(LibtpuSdk_Metric {
                description,
                value_cstrings,
                value_ptrs,
            }));
            // SAFETY: Writing output metric pointer.
            unsafe {
                (*args_ptr).metric = metric_handle;
            }
            ptr::null_mut()
        }
        Err(e) => box_sdk_error(&e),
    }
}

/// Slot 11 of `LibtpuSdk_Api`: queries the description of a [`LibtpuSdk_Metric`] (`LibtpuSdk_GetMetricDescription`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`LibtpuSdk_GetMetricDescription_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`LibtpuSdk_GetMetricDescription_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LibtpuSdk_GetMetricDescription(
    raw_args: *mut std::ffi::c_void,
) -> *mut LibtpuSdk_Error {
    let args_ptr = raw_args.cast::<LibtpuSdk_GetMetricDescription_Args>();
    if args_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer { parameter: "args" });
    }
    // SAFETY: `args_ptr` is non-null.
    let metric_ptr = unsafe { (*args_ptr).metric };
    if metric_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer {
            parameter: "metric",
        });
    }
    // SAFETY: `metric_ptr` is non-null.
    let metric_ref = unsafe { &*metric_ptr };
    let bytes = metric_ref.description.as_bytes();
    // SAFETY: Writing output description fields.
    unsafe {
        (*args_ptr).description = metric_ref.description.as_ptr();
        (*args_ptr).description_size = bytes.len();
    }
    ptr::null_mut()
}

/// Slot 12 of `LibtpuSdk_Api`: queries the value strings of a [`LibtpuSdk_Metric`] (`LibtpuSdk_GetMetricValues`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`LibtpuSdk_GetMetricValues_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`LibtpuSdk_GetMetricValues_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LibtpuSdk_GetMetricValues(
    raw_args: *mut std::ffi::c_void,
) -> *mut LibtpuSdk_Error {
    let args_ptr = raw_args.cast::<LibtpuSdk_GetMetricValues_Args>();
    if args_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer { parameter: "args" });
    }
    // SAFETY: `args_ptr` is non-null.
    let metric_ptr = unsafe { (*args_ptr).metric };
    if metric_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer {
            parameter: "metric",
        });
    }
    // SAFETY: `metric_ptr` is non-null.
    let metric_ref = unsafe { &*metric_ptr };
    // SAFETY: Writing output values pointer and count.
    unsafe {
        (*args_ptr).values = metric_ref.value_ptrs.as_ptr();
        (*args_ptr).num_values = metric_ref.value_ptrs.len();
    }
    ptr::null_mut()
}

/// Slot 13 of `LibtpuSdk_Api`: queries runtime core state (`LibtpuSdk_GetRuntimeStatus`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`LibtpuSdk_GetRuntimeStatus_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`LibtpuSdk_GetRuntimeStatus_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LibtpuSdk_GetRuntimeStatus(
    raw_args: *mut std::ffi::c_void,
) -> *mut LibtpuSdk_Error {
    let args_ptr = raw_args.cast::<LibtpuSdk_GetRuntimeStatus_Args>();
    if args_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer { parameter: "args" });
    }
    // SAFETY: `args_ptr` is non-null.
    let client_ptr = unsafe { (*args_ptr).client };
    if client_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer {
            parameter: "client",
        });
    }
    // SAFETY: `client_ptr` is non-null.
    let summary_text = unsafe { (*client_ptr).inner.core_state_summary() };
    let status_handle = Box::into_raw(Box::new(LibtpuSdk_RuntimeStatus {
        summary: to_c_string(&summary_text),
    }));
    // SAFETY: Writing output status pointer.
    unsafe {
        (*args_ptr).status = status_handle;
    }
    ptr::null_mut()
}

/// Slot 14 of `LibtpuSdk_Api`: extracts the summary string from [`LibtpuSdk_RuntimeStatus`]
/// (`LibtpuSdk_RuntimeStatus_GetCoreStateSummary`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`LibtpuSdk_RuntimeStatus_GetCoreStateSummary_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`LibtpuSdk_RuntimeStatus_GetCoreStateSummary_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LibtpuSdk_RuntimeStatus_GetCoreStateSummary(
    raw_args: *mut std::ffi::c_void,
) -> *mut LibtpuSdk_Error {
    let args_ptr = raw_args.cast::<LibtpuSdk_RuntimeStatus_GetCoreStateSummary_Args>();
    if args_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer { parameter: "args" });
    }
    // SAFETY: `args_ptr` is non-null.
    let status_ptr = unsafe { (*args_ptr).status };
    if status_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer {
            parameter: "status",
        });
    }
    // SAFETY: `status_ptr` is non-null.
    let status_ref = unsafe { &*status_ptr };
    let bytes = status_ref.summary.as_bytes();
    // SAFETY: Writing output summary fields.
    unsafe {
        (*args_ptr).summary = status_ref.summary.as_ptr();
        (*args_ptr).summary_size = bytes.len();
    }
    ptr::null_mut()
}

/// Slot 15 of `LibtpuSdk_Api`: destroys a [`LibtpuSdk_RuntimeStatus`] (`LibtpuSdk_RuntimeStatus_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`LibtpuSdk_RuntimeStatus_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`LibtpuSdk_RuntimeStatus_Destroy_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LibtpuSdk_RuntimeStatus_Destroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut LibtpuSdk_Error {
    let args_ptr = raw_args.cast::<LibtpuSdk_RuntimeStatus_Destroy_Args>();
    if args_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer { parameter: "args" });
    }
    // SAFETY: `args_ptr` is non-null.
    let status_ptr = unsafe { (*args_ptr).status };
    if !status_ptr.is_null() {
        // SAFETY: `status_ptr` was allocated in `LibtpuSdk_GetRuntimeStatus`.
        unsafe {
            drop(Box::from_raw(status_ptr));
        }
    }
    ptr::null_mut()
}

/// Slot 16 of `LibtpuSdk_Api`: registers a debug callback (`LibtpuSdk_RegisterDebugCallback`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`LibtpuSdk_RegisterDebugCallback_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`LibtpuSdk_RegisterDebugCallback_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LibtpuSdk_RegisterDebugCallback(
    raw_args: *mut std::ffi::c_void,
) -> *mut LibtpuSdk_Error {
    let args_ptr = raw_args.cast::<LibtpuSdk_RegisterDebugCallback_Args>();
    if args_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer { parameter: "args" });
    }
    // SAFETY: `args_ptr` is non-null.
    let client_ptr = unsafe { (*args_ptr).client };
    if client_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer {
            parameter: "client",
        });
    }
    // SAFETY: `client_ptr` is non-null.
    let id = unsafe {
        (*client_ptr).inner.register_debug_callback(
            (*args_ptr).callback as usize,
            (*args_ptr).user_data as usize,
        )
    };
    // SAFETY: Writing output callback_id.
    unsafe {
        (*args_ptr).callback_id = i64::try_from(id.0).unwrap_or(0);
    }
    ptr::null_mut()
}

/// Slot 17 of `LibtpuSdk_Api`: unregisters a debug callback (`LibtpuSdk_UnregisterDebugCallback`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`LibtpuSdk_UnregisterDebugCallback_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`LibtpuSdk_UnregisterDebugCallback_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LibtpuSdk_UnregisterDebugCallback(
    raw_args: *mut std::ffi::c_void,
) -> *mut LibtpuSdk_Error {
    let args_ptr = raw_args.cast::<LibtpuSdk_UnregisterDebugCallback_Args>();
    if args_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer { parameter: "args" });
    }
    // SAFETY: `args_ptr` is non-null.
    let (client_ptr, cb_id) = unsafe { ((*args_ptr).client, (*args_ptr).callback_id) };
    if client_ptr.is_null() {
        return box_sdk_error(&LibTpuError::NullPointer {
            parameter: "client",
        });
    }
    // SAFETY: `client_ptr` is non-null.
    if let Err(e) = unsafe {
        (*client_ptr)
            .inner
            .unregister_debug_callback(CallbackId(u64::try_from(cb_id).unwrap_or(u64::MAX)))
    } {
        return box_sdk_error(&e);
    }
    ptr::null_mut()
}

/// Invokes all registered C debug callbacks (`void (*)(void*)`) on `client`
/// (`LibtpuSdk_TriggerDebugCallbacks`).
///
/// # Arguments
///
/// * `client` - Pointer to [`LibtpuSdk_Client`] (or null).
///
/// # Safety
///
/// `client` must be null or a valid pointer to [`LibtpuSdk_Client`], and any registered
/// non-null callback pointers with `fn_addr >= 0x10000` must be valid `extern "C" fn(*mut c_void)`
/// function pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LibtpuSdk_TriggerDebugCallbacks(client: *const LibtpuSdk_Client) -> usize {
    if client.is_null() {
        return 0;
    }
    // SAFETY: `client` is non-null.
    let callbacks = unsafe { (*client).inner.registered_debug_callbacks() };
    let mut invoked = 0_usize;
    for (fn_addr, user_data) in callbacks {
        if fn_addr >= 0x10000 {
            // SAFETY: Caller registered a valid `extern "C" fn(*mut c_void)` function pointer.
            let cb: unsafe extern "C" fn(*mut std::ffi::c_void) =
                unsafe { std::mem::transmute(fn_addr) };
            // SAFETY: Invoking registered C callback with its opaque `user_data` pointer.
            unsafe {
                cb(user_data as *mut std::ffi::c_void);
            }
        }
        invoked = invoked.saturating_add(1);
    }
    invoked
}

/// Static 144-byte `LibtpuSdk_Api` vtable returned by [`GetLibtpuSdkApi`].
static GLOBAL_SDK_API: LibtpuSdk_Api = LibtpuSdk_Api {
    version_header: 0x0000_0001_0000_0000,
    error_get_message: LibtpuSdk_Error_GetMessage,
    error_destroy: LibtpuSdk_Error_Destroy,
    error_get_code: LibtpuSdk_Error_GetCode,
    client_create: LibtpuSdk_Client_Create,
    client_destroy: LibtpuSdk_Client_Destroy,
    get_chip_coordinates: LibtpuSdk_GetChipCoordinates,
    get_host_name: LibtpuSdk_GetHostName,
    get_chip_index: LibtpuSdk_GetChipIndex,
    get_cartesian_coordinates: LibtpuSdk_GetCartesianCoordinates,
    get_metric: LibtpuSdk_GetMetric,
    get_metric_description: LibtpuSdk_GetMetricDescription,
    get_metric_values: LibtpuSdk_GetMetricValues,
    get_runtime_status: LibtpuSdk_GetRuntimeStatus,
    runtime_status_get_core_state_summary: LibtpuSdk_RuntimeStatus_GetCoreStateSummary,
    runtime_status_destroy: LibtpuSdk_RuntimeStatus_Destroy,
    register_debug_callback: LibtpuSdk_RegisterDebugCallback,
    unregister_debug_callback: LibtpuSdk_UnregisterDebugCallback,
};

/// Secondary SDK C API entrypoint exported by `libtpu.so` (`GetLibtpuSdkApi`).
#[unsafe(no_mangle)]
pub extern "C" fn GetLibtpuSdkApi() -> *const LibtpuSdk_Api {
    ptr::addr_of!(GLOBAL_SDK_API)
}

/// Native `CPython` extension entrypoint exported by `sdk.so` (`PyInit_sdk`).
#[unsafe(no_mangle)]
pub extern "C" fn PyInit_sdk() -> *const LibtpuSdk_Api {
    ptr::addr_of!(GLOBAL_SDK_API)
}

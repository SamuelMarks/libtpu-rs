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
        metrics: Vec::new(),
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
    let client_ref = unsafe { &mut *client_ptr };
    match client_ref.inner.get_metric(&metric_name) {
        Ok((desc, vals)) => {
            let description = to_c_string(&desc);
            let value_cstrings: Vec<CString> =
                vals.iter().map(|v| to_c_string(&v.to_string())).collect();
            let value_ptrs: Vec<*const c_char> =
                value_cstrings.iter().map(|cs| cs.as_ptr()).collect();
            let mut boxed_metric = Box::new(LibtpuSdk_Metric {
                description,
                value_cstrings,
                value_ptrs,
            });
            let metric_handle: *mut LibtpuSdk_Metric = ptr::from_mut(&mut *boxed_metric);
            client_ref.metrics.push(boxed_metric);
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

#[cfg(test)]
#[allow(
    clippy::cognitive_complexity,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::items_after_statements
)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_libtpu_sdk_api_vtable_and_lifecycle() {
        let sdk_ptr = GetLibtpuSdkApi();
        assert!(!sdk_ptr.is_null());
        assert_eq!(PyInit_sdk(), sdk_ptr);
        // SAFETY: Testing all 17 LibtpuSdk_Api entrypoints with null and valid arguments.
        unsafe {
            let sdk = *sdk_ptr;
            assert_eq!(sdk.version_header, 0x0000_0001_0000_0000);

            LibtpuSdk_Error_GetMessage(ptr::null_mut());
            let mut sdk_msg_args = LibtpuSdk_Error_GetMessage_Args {
                error: ptr::null(),
                message: ptr::null(),
                message_size: 0,
            };
            assert_ne!(format!("{sdk_msg_args:?}"), "");
            LibtpuSdk_Error_GetMessage(ptr::addr_of_mut!(sdk_msg_args).cast());
            let sdk_err = LibtpuSdk_Error_GetCode(ptr::null_mut());
            assert!(!sdk_err.is_null());
            assert_ne!(format!("{:?}", *sdk_err), "");
            sdk_msg_args.error = sdk_err;
            LibtpuSdk_Error_GetMessage(ptr::addr_of_mut!(sdk_msg_args).cast());
            assert!(sdk_msg_args.message_size > 0);

            let mut sdk_code_args = LibtpuSdk_Error_GetCode_Args {
                error: ptr::null(),
                code: 0,
                pad: 0,
            };
            assert_ne!(format!("{sdk_code_args:?}"), "");
            drop(Box::from_raw(LibtpuSdk_Error_GetCode(
                ptr::addr_of_mut!(sdk_code_args).cast(),
            )));
            sdk_code_args.error = sdk_err;
            assert!(LibtpuSdk_Error_GetCode(ptr::addr_of_mut!(sdk_code_args).cast()).is_null());
            assert_eq!(sdk_code_args.code, 3);

            LibtpuSdk_Error_Destroy(ptr::null_mut());
            let mut sdk_del_args = LibtpuSdk_Error_Destroy_Args {
                error: ptr::null_mut(),
            };
            assert_ne!(format!("{sdk_del_args:?}"), "");
            LibtpuSdk_Error_Destroy(ptr::addr_of_mut!(sdk_del_args).cast());
            sdk_del_args.error = sdk_err;
            LibtpuSdk_Error_Destroy(ptr::addr_of_mut!(sdk_del_args).cast());

            drop(Box::from_raw(LibtpuSdk_Client_Create(ptr::null_mut())));
            let mut sdk_cli_args = LibtpuSdk_Client_Args {
                client: ptr::null_mut(),
            };
            assert_ne!(format!("{sdk_cli_args:?}"), "");
            assert!(LibtpuSdk_Client_Create(ptr::addr_of_mut!(sdk_cli_args).cast()).is_null());
            let sdk_client = sdk_cli_args.client;
            assert!(!sdk_client.is_null());

            drop(Box::from_raw(LibtpuSdk_GetChipCoordinates(ptr::null_mut())));
            let mut coords_args = LibtpuSdk_GetChipCoordinates_Args {
                client: ptr::null_mut(),
                coords: ptr::null(),
                num_coords: 0,
            };
            assert_ne!(format!("{coords_args:?}"), "");
            drop(Box::from_raw(LibtpuSdk_GetChipCoordinates(
                ptr::addr_of_mut!(coords_args).cast(),
            )));
            coords_args.client = sdk_client;
            assert!(LibtpuSdk_GetChipCoordinates(ptr::addr_of_mut!(coords_args).cast()).is_null());
            assert_eq!(coords_args.num_coords, 4);
            assert_ne!(format!("{:?}", *coords_args.coords), "");

            drop(Box::from_raw(LibtpuSdk_GetHostName(ptr::null_mut())));
            let mut host_args = LibtpuSdk_GetHostName_Args {
                coord: ptr::null(),
                host_name: ptr::null(),
                host_name_size: 0,
            };
            assert_ne!(format!("{host_args:?}"), "");
            drop(Box::from_raw(LibtpuSdk_GetHostName(
                ptr::addr_of_mut!(host_args).cast(),
            )));
            host_args.coord = coords_args.coords;
            assert!(LibtpuSdk_GetHostName(ptr::addr_of_mut!(host_args).cast()).is_null());
            assert!(host_args.host_name_size > 0);

            drop(Box::from_raw(LibtpuSdk_GetChipIndex(ptr::null_mut())));
            let mut cidx_args = LibtpuSdk_GetChipIndex_Args {
                coord: ptr::null(),
                chip_index: -1,
                pad: 0,
            };
            assert_ne!(format!("{cidx_args:?}"), "");
            drop(Box::from_raw(LibtpuSdk_GetChipIndex(
                ptr::addr_of_mut!(cidx_args).cast(),
            )));
            cidx_args.coord = coords_args.coords;
            assert!(LibtpuSdk_GetChipIndex(ptr::addr_of_mut!(cidx_args).cast()).is_null());
            assert_eq!(cidx_args.chip_index, 0);

            drop(Box::from_raw(LibtpuSdk_GetCartesianCoordinates(
                ptr::null_mut(),
            )));
            let mut cart_args = LibtpuSdk_GetCartesianCoordinates_Args {
                coord: ptr::null(),
                coordinates: ptr::null(),
                num_coordinates: 0,
            };
            assert_ne!(format!("{cart_args:?}"), "");
            drop(Box::from_raw(LibtpuSdk_GetCartesianCoordinates(
                ptr::addr_of_mut!(cart_args).cast(),
            )));
            cart_args.coord = coords_args.coords;
            assert!(
                LibtpuSdk_GetCartesianCoordinates(ptr::addr_of_mut!(cart_args).cast()).is_null()
            );
            assert_eq!(cart_args.num_coordinates, 3);

            drop(Box::from_raw(LibtpuSdk_GetMetric(ptr::null_mut())));
            let mut met_args = LibtpuSdk_GetMetric_Args {
                client: ptr::null_mut(),
                metric_name: ptr::null(),
                metric: ptr::null_mut(),
            };
            assert_ne!(format!("{met_args:?}"), "");
            drop(Box::from_raw(LibtpuSdk_GetMetric(
                ptr::addr_of_mut!(met_args).cast(),
            )));
            met_args.client = sdk_client;
            met_args.metric_name = c"unknown_metric".as_ptr();
            drop(Box::from_raw(LibtpuSdk_GetMetric(
                ptr::addr_of_mut!(met_args).cast(),
            )));
            met_args.metric_name = c"tpu.runtime.hbm.usage_bytes".as_ptr();
            assert!(LibtpuSdk_GetMetric(ptr::addr_of_mut!(met_args).cast()).is_null());
            assert!(!met_args.metric.is_null());
            assert_ne!(format!("{:?}", *met_args.metric), "");

            drop(Box::from_raw(LibtpuSdk_GetMetricDescription(
                ptr::null_mut(),
            )));
            let mut mdesc_args = LibtpuSdk_GetMetricDescription_Args {
                metric: ptr::null(),
                description: ptr::null(),
                description_size: 0,
            };
            assert_ne!(format!("{mdesc_args:?}"), "");
            drop(Box::from_raw(LibtpuSdk_GetMetricDescription(
                ptr::addr_of_mut!(mdesc_args).cast(),
            )));
            mdesc_args.metric = met_args.metric;
            assert!(LibtpuSdk_GetMetricDescription(ptr::addr_of_mut!(mdesc_args).cast()).is_null());
            assert!(mdesc_args.description_size > 0);

            drop(Box::from_raw(LibtpuSdk_GetMetricValues(ptr::null_mut())));
            let mut mvals_args = LibtpuSdk_GetMetricValues_Args {
                metric: ptr::null(),
                values: ptr::null(),
                num_values: 0,
            };
            assert_ne!(format!("{mvals_args:?}"), "");
            drop(Box::from_raw(LibtpuSdk_GetMetricValues(
                ptr::addr_of_mut!(mvals_args).cast(),
            )));
            mvals_args.metric = met_args.metric;
            assert!(LibtpuSdk_GetMetricValues(ptr::addr_of_mut!(mvals_args).cast()).is_null());
            assert_eq!(mvals_args.num_values, 4);

            drop(Box::from_raw(LibtpuSdk_GetRuntimeStatus(ptr::null_mut())));
            let mut rstat_args = LibtpuSdk_GetRuntimeStatus_Args {
                client: ptr::null_mut(),
                include_hlo_info: true,
                pad: [0; 7],
                status: ptr::null_mut(),
            };
            assert_ne!(format!("{rstat_args:?}"), "");
            drop(Box::from_raw(LibtpuSdk_GetRuntimeStatus(
                ptr::addr_of_mut!(rstat_args).cast(),
            )));
            rstat_args.client = sdk_client;
            assert!(LibtpuSdk_GetRuntimeStatus(ptr::addr_of_mut!(rstat_args).cast()).is_null());
            assert!(!rstat_args.status.is_null());
            assert_ne!(format!("{:?}", *rstat_args.status), "");

            drop(Box::from_raw(LibtpuSdk_RuntimeStatus_GetCoreStateSummary(
                ptr::null_mut(),
            )));
            let mut sum_args = LibtpuSdk_RuntimeStatus_GetCoreStateSummary_Args {
                status: ptr::null(),
                summary: ptr::null(),
                summary_size: 0,
            };
            assert_ne!(format!("{sum_args:?}"), "");
            drop(Box::from_raw(LibtpuSdk_RuntimeStatus_GetCoreStateSummary(
                ptr::addr_of_mut!(sum_args).cast(),
            )));
            sum_args.status = rstat_args.status;
            assert!(
                LibtpuSdk_RuntimeStatus_GetCoreStateSummary(ptr::addr_of_mut!(sum_args).cast())
                    .is_null()
            );
            assert!(sum_args.summary_size > 0);

            drop(Box::from_raw(LibtpuSdk_RuntimeStatus_Destroy(
                ptr::null_mut(),
            )));
            let mut rdel_args = LibtpuSdk_RuntimeStatus_Destroy_Args {
                status: ptr::null_mut(),
            };
            assert_ne!(format!("{rdel_args:?}"), "");
            assert!(LibtpuSdk_RuntimeStatus_Destroy(ptr::addr_of_mut!(rdel_args).cast()).is_null());
            rdel_args.status = rstat_args.status;
            assert!(LibtpuSdk_RuntimeStatus_Destroy(ptr::addr_of_mut!(rdel_args).cast()).is_null());

            drop(Box::from_raw(LibtpuSdk_RegisterDebugCallback(
                ptr::null_mut(),
            )));
            let mut reg_cb_args = LibtpuSdk_RegisterDebugCallback_Args {
                client: ptr::null_mut(),
                callback: ptr::null(),
                user_data: ptr::null_mut(),
                callback_id: 0,
            };
            assert_ne!(format!("{reg_cb_args:?}"), "");
            drop(Box::from_raw(LibtpuSdk_RegisterDebugCallback(
                ptr::addr_of_mut!(reg_cb_args).cast(),
            )));
            reg_cb_args.client = sdk_client;
            assert!(
                LibtpuSdk_RegisterDebugCallback(ptr::addr_of_mut!(reg_cb_args).cast()).is_null()
            );
            assert!(reg_cb_args.callback_id > 0);
            assert_eq!(LibtpuSdk_TriggerDebugCallbacks(ptr::null()), 0);
            assert_eq!(LibtpuSdk_TriggerDebugCallbacks(sdk_client), 1);

            unsafe extern "C" fn rust_debug_cb(user_data: *mut std::ffi::c_void) {
                // SAFETY: `user_data` points to `cb_counter`.
                unsafe {
                    *user_data.cast::<u32>() += 1;
                }
            }
            let mut cb_counter = 0_u32;
            let mut real_cb_args = LibtpuSdk_RegisterDebugCallback_Args {
                client: sdk_client,
                callback: (rust_debug_cb as *const () as usize) as *const std::ffi::c_void,
                user_data: ptr::addr_of_mut!(cb_counter).cast(),
                callback_id: 0,
            };
            assert!(
                LibtpuSdk_RegisterDebugCallback(ptr::addr_of_mut!(real_cb_args).cast()).is_null()
            );
            assert_eq!(LibtpuSdk_TriggerDebugCallbacks(sdk_client), 2);
            assert_eq!(cb_counter, 1);
            let mut unreg_real = LibtpuSdk_UnregisterDebugCallback_Args {
                client: sdk_client,
                callback_id: real_cb_args.callback_id,
            };
            assert!(
                LibtpuSdk_UnregisterDebugCallback(ptr::addr_of_mut!(unreg_real).cast()).is_null()
            );

            drop(Box::from_raw(LibtpuSdk_UnregisterDebugCallback(
                ptr::null_mut(),
            )));
            let mut unreg_cb_args = LibtpuSdk_UnregisterDebugCallback_Args {
                client: ptr::null_mut(),
                callback_id: reg_cb_args.callback_id,
            };
            assert_ne!(format!("{unreg_cb_args:?}"), "");
            drop(Box::from_raw(LibtpuSdk_UnregisterDebugCallback(
                ptr::addr_of_mut!(unreg_cb_args).cast(),
            )));
            unreg_cb_args.client = sdk_client;
            assert!(
                LibtpuSdk_UnregisterDebugCallback(ptr::addr_of_mut!(unreg_cb_args).cast())
                    .is_null()
            );
            drop(Box::from_raw(LibtpuSdk_UnregisterDebugCallback(
                ptr::addr_of_mut!(unreg_cb_args).cast(),
            )));

            drop(Box::from_raw(LibtpuSdk_Client_Destroy(ptr::null_mut())));
            let mut null_cli_args = LibtpuSdk_Client_Args {
                client: ptr::null_mut(),
            };
            drop(Box::from_raw(LibtpuSdk_Client_Destroy(
                ptr::addr_of_mut!(null_cli_args).cast(),
            )));
            assert!(LibtpuSdk_Client_Destroy(ptr::addr_of_mut!(sdk_cli_args).cast()).is_null());
        }
    }
}

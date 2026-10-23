//! Ergonomic `libtpu_rs_*` C/C++/Python FFI helper entrypoints.

use super::*;

// ============================================================================
// Clean C / C++20 FFI Entrypoints (`libtpu_rs_*`)
// ============================================================================

/// Creates a success [`LibTpuCStatus`].
pub(crate) fn c_status_ok() -> LibTpuCStatus {
    LibTpuCStatus {
        code: 0,
        message: ptr::null_mut(),
    }
}

/// Creates an error [`LibTpuCStatus`] from a [`LibTpuError`].
///
/// # Arguments
///
/// * `err` - Domain error to convert.
fn c_status_err(err: &LibTpuError) -> LibTpuCStatus {
    let msg = to_c_string(&err.to_string());
    LibTpuCStatus {
        code: err.error_code().as_raw(),
        message: msg.into_raw(),
    }
}

/// Frees the error message inside a [`LibTpuCStatus`] (`libtpu_rs_status_free`).
///
/// # Arguments
///
/// * `status` - Status struct whose `message` pointer should be freed.
///
/// # Safety
///
/// `status.message` must be null or allocated by `c_status_err`.
#[no_mangle]
pub unsafe extern "C" fn libtpu_rs_status_free(status: LibTpuCStatus) {
    if !status.message.is_null() {
        // SAFETY: `status.message` was allocated via `CString::into_raw`.
        unsafe {
            drop(CString::from_raw(status.message));
        }
    }
}

/// Creates a new [`PJRT_Client`] for C/C++ consumers (`libtpu_rs_client_create`).
///
/// # Arguments
///
/// * `client_out` - Output pointer to receive the created [`PJRT_Client`].
///
/// # Safety
///
/// `client_out` must be null or a valid mutable pointer.
#[no_mangle]
pub unsafe extern "C" fn libtpu_rs_client_create(
    client_out: *mut *mut PJRT_Client,
) -> LibTpuCStatus {
    if client_out.is_null() {
        return c_status_err(&LibTpuError::NullPointer {
            parameter: "client_out",
        });
    }
    let mut args = PJRT_Client_Create_Args {
        struct_size: std::mem::size_of::<PJRT_Client_Create_Args>(),
        extension_start: ptr::null_mut(),
        create_options: ptr::null(),
        num_options: 0,
        kv_get_callback: ptr::null(),
        kv_get_user_arg: ptr::null_mut(),
        kv_put_callback: ptr::null(),
        kv_put_user_arg: ptr::null_mut(),
        client: ptr::null_mut(),
    };
    // SAFETY: Passing valid stack `args`.
    let _ = unsafe { PJRT_Client_Create(ptr::addr_of_mut!(args).cast()) };
    // SAFETY: `client_out` is non-null.
    unsafe {
        *client_out = args.client;
    }
    c_status_ok()
}

/// Destroys a [`PJRT_Client`] created by `libtpu_rs_client_create` (`libtpu_rs_client_destroy`).
///
/// # Arguments
///
/// * `client` - Client handle to destroy.
///
/// # Safety
///
/// `client` must be null or allocated by `libtpu_rs_client_create`.
#[no_mangle]
pub unsafe extern "C" fn libtpu_rs_client_destroy(client: *mut PJRT_Client) {
    if client.is_null() {
        return;
    }
    let mut args = PJRT_Client_Destroy_Args {
        struct_size: std::mem::size_of::<PJRT_Client_Destroy_Args>(),
        extension_start: ptr::null_mut(),
        client,
    };
    // SAFETY: `args` is valid.
    let _ = unsafe { PJRT_Client_Destroy(ptr::addr_of_mut!(args).cast()) };
}

/// Returns the number of addressable devices on `client` (`libtpu_rs_client_device_count`).
///
/// # Arguments
///
/// * `client` - Pointer to [`PJRT_Client`].
///
/// # Safety
///
/// `client` must be null or a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn libtpu_rs_client_device_count(client: *const PJRT_Client) -> i32 {
    if client.is_null() {
        return -1;
    }
    // SAFETY: `client` is non-null.
    unsafe { i32::try_from((*client).addressable_device_ptrs.len()).unwrap_or(0) }
}

/// Allocates a [`PJRT_Buffer`] on `device_id` and copies `host_data` into HBM (`libtpu_rs_buffer_from_host`).
///
/// # Arguments
///
/// * `client` - Pointer to [`PJRT_Client`].
/// * `device_id` - Target logical TPU device ID.
/// * `element_type` - Raw `PJRT_Buffer_Type` integer (`11` = `F32`, `4` = `S32`).
/// * `dims` - Pointer to dimension array.
/// * `num_dims` - Number of dimensions.
/// * `host_data` - Pointer to source host bytes.
/// * `host_data_len` - Number of bytes in `host_data`.
/// * `buffer_out` - Output pointer to receive the created [`PJRT_Buffer`].
///
/// # Safety
///
/// All pointers must be null or valid for their respective lengths.
#[no_mangle]
pub unsafe extern "C" fn libtpu_rs_buffer_from_host(
    client: *mut PJRT_Client,
    device_id: u32,
    element_type: i32,
    dims: *const i64,
    num_dims: usize,
    host_data: *const u8,
    host_data_len: usize,
    buffer_out: *mut *mut PJRT_Buffer,
) -> LibTpuCStatus {
    if any_null(&[client as usize, buffer_out as usize]) {
        return c_status_err(&LibTpuError::NullPointer {
            parameter: "client_or_buffer_out",
        });
    }
    let elem = match ElementType::from_raw(element_type) {
        Ok(t) => t,
        Err(e) => return c_status_err(&e),
    };
    let dims_vec = if dims.is_null() {
        Vec::new()
    } else {
        // SAFETY: `dims` is non-null with `num_dims` elements.
        unsafe { std::slice::from_raw_parts(dims, num_dims).to_vec() }
    };
    let shape = match Shape::new(elem, dims_vec, None, None) {
        Ok(s) => s,
        Err(e) => return c_status_err(&e),
    };
    let data_slice = if host_data.is_null() {
        &[][..]
    } else {
        // SAFETY: `host_data` is non-null with `host_data_len` bytes.
        unsafe { std::slice::from_raw_parts(host_data, host_data_len) }
    };

    // SAFETY: `client` is non-null.
    let client_ref = unsafe { &*client };
    match client_ref
        .inner
        .buffer_from_host(DeviceId(device_id), shape, data_slice)
    {
        Ok(core_buf) => {
            let dev_ptr = client_ref
                .device_ptrs
                .get(device_id as usize)
                .copied()
                .unwrap_or(ptr::null_mut());
            let mem_ptr = client_ref
                .memory_ptrs
                .first()
                .copied()
                .unwrap_or(ptr::null_mut());
            let handle = Box::into_raw(Box::new(PJRT_Buffer {
                inner: core_buf,
                device_ptr: dev_ptr,
                memory_ptr: mem_ptr,
            }));
            // SAFETY: `buffer_out` is non-null.
            unsafe {
                *buffer_out = handle;
            }
            c_status_ok()
        }
        Err(e) => c_status_err(&e),
    }
}

/// Copies a [`PJRT_Buffer`] from device HBM into `host_out` (`libtpu_rs_buffer_to_host`).
///
/// # Arguments
///
/// * `buffer` - Pointer to [`PJRT_Buffer`].
/// * `host_out` - Destination host buffer.
/// * `host_out_len` - Number of bytes to copy into `host_out`.
///
/// # Safety
///
/// `buffer` and `host_out` must be null or valid pointers.
#[no_mangle]
pub unsafe extern "C" fn libtpu_rs_buffer_to_host(
    buffer: *const PJRT_Buffer,
    host_out: *mut u8,
    host_out_len: usize,
) -> LibTpuCStatus {
    if any_null(&[buffer as usize, host_out as usize]) {
        return c_status_err(&LibTpuError::NullPointer {
            parameter: "buffer_or_host_out",
        });
    }
    // SAFETY: `buffer` is non-null.
    let buf_ref = unsafe { &(*buffer).inner };
    match buf_ref.to_host_bytes() {
        Ok(bytes) => {
            if host_out_len > bytes.len() {
                return c_status_err(&LibTpuError::TransferFailed {
                    direction: "D2H",
                    reason: format!(
                        "Requested {host_out_len} bytes from {}-byte buffer",
                        bytes.len()
                    ),
                });
            }
            // SAFETY: `host_out` is valid for `host_out_len` bytes.
            unsafe {
                ptr::copy_nonoverlapping(bytes.as_ptr(), host_out, host_out_len);
            }
            c_status_ok()
        }
        Err(e) => c_status_err(&e),
    }
}

/// Deletes and frees a [`PJRT_Buffer`] (`libtpu_rs_buffer_destroy`).
///
/// # Arguments
///
/// * `buffer` - Pointer to [`PJRT_Buffer`] to destroy.
///
/// # Safety
///
/// `buffer` must be null or allocated by `libtpu_rs_buffer_from_host` / `libtpu_rs_execute`.
#[no_mangle]
pub unsafe extern "C" fn libtpu_rs_buffer_destroy(buffer: *mut PJRT_Buffer) {
    if !buffer.is_null() {
        // SAFETY: `buffer` is non-null.
        unsafe {
            let b = Box::from_raw(buffer);
            let _ = b.inner.delete();
        }
    }
}

/// Compiles and loads an HLO or `StableHLO` program (`libtpu_rs_compile`).
///
/// # Arguments
///
/// * `client` - Pointer to [`PJRT_Client`].
/// * `format` - Null-terminated format string (`"mlir"` or `"hlo"`).
/// * `code` - Pointer to program bytes.
/// * `code_len` - Number of bytes in `code`.
/// * `exec_out` - Output pointer to receive the [`PJRT_LoadedExecutable`].
///
/// # Safety
///
/// All pointers must be null or valid.
#[no_mangle]
pub unsafe extern "C" fn libtpu_rs_compile(
    client: *mut PJRT_Client,
    format: *const c_char,
    code: *const u8,
    code_len: usize,
    exec_out: *mut *mut PJRT_LoadedExecutable,
) -> LibTpuCStatus {
    if any_null(&[
        client as usize,
        format as usize,
        code as usize,
        exec_out as usize,
    ]) {
        return c_status_err(&LibTpuError::NullPointer {
            parameter: "compile_args",
        });
    }
    // SAFETY: `format` is non-null.
    let fmt_str = unsafe { CStr::from_ptr(format).to_string_lossy() };
    // SAFETY: `code` is non-null.
    let code_slice = unsafe { std::slice::from_raw_parts(code, code_len) };
    // SAFETY: `client` is non-null.
    let client_ref = unsafe { &*client };
    match client_ref.inner.compile_and_load(&fmt_str, code_slice, b"") {
        Ok(loaded) => {
            let exec_ptr = Box::into_raw(Box::new(PJRT_Executable {
                inner: loaded.executable.clone(),
            }));
            let loaded_ptr = Box::into_raw(Box::new(PJRT_LoadedExecutable {
                inner: loaded,
                executable_ptr: exec_ptr,
                device_ptrs: client_ref.addressable_device_ptrs.clone(),
            }));
            // SAFETY: `exec_out` is non-null.
            unsafe {
                *exec_out = loaded_ptr;
            }
            c_status_ok()
        }
        Err(e) => c_status_err(&e),
    }
}

/// Executes a [`PJRT_LoadedExecutable`] on `device_id` (`libtpu_rs_execute`).
///
/// # Arguments
///
/// * `executable` - Pointer to [`PJRT_LoadedExecutable`].
/// * `device_id` - Target logical TPU device ID.
/// * `inputs` - Array of input `*const PJRT_Buffer` pointers.
/// * `num_inputs` - Number of input buffers.
/// * `output_out` - Output pointer to receive the primary output [`PJRT_Buffer`].
///
/// # Safety
///
/// All pointers must be null or valid.
#[no_mangle]
pub unsafe extern "C" fn libtpu_rs_execute(
    executable: *const PJRT_LoadedExecutable,
    device_id: u32,
    inputs: *const *const PJRT_Buffer,
    num_inputs: usize,
    output_out: *mut *mut PJRT_Buffer,
) -> LibTpuCStatus {
    if any_null(&[executable as usize, output_out as usize]) {
        return c_status_err(&LibTpuError::NullPointer {
            parameter: "execute_args",
        });
    }
    let mut core_inputs = Vec::with_capacity(num_inputs);
    if !inputs.is_null() {
        // SAFETY: `inputs` is non-null with `num_inputs` elements.
        let slice = unsafe { std::slice::from_raw_parts(inputs, num_inputs) };
        for &ptr_item in slice {
            if ptr_item.is_null() {
                return c_status_err(&LibTpuError::NullPointer {
                    parameter: "input_buffer",
                });
            }
            // SAFETY: `ptr_item` is non-null.
            core_inputs.push(unsafe { &(*ptr_item).inner });
        }
    }
    // SAFETY: `executable` is non-null.
    let exec_ref = unsafe { &(*executable).inner };
    match exec_ref.execute_on_device(
        DeviceId(device_id),
        libtpu_core::types::MemorySpaceId(0),
        &core_inputs,
    ) {
        Ok(outs) => {
            for first in outs {
                let handle = Box::into_raw(Box::new(PJRT_Buffer {
                    inner: first,
                    device_ptr: ptr::null_mut(),
                    memory_ptr: ptr::null_mut(),
                }));
                // SAFETY: `output_out` is non-null.
                unsafe {
                    *output_out = handle;
                }
            }
            c_status_ok()
        }
        Err(e) => c_status_err(&e),
    }
}

/// Destroys a [`PJRT_LoadedExecutable`] (`libtpu_rs_executable_destroy`).
///
/// # Arguments
///
/// * `executable` - Pointer to [`PJRT_LoadedExecutable`].
///
/// # Safety
///
/// `executable` must be null or allocated by `libtpu_rs_compile` / `PJRT_Client_Compile`.
#[no_mangle]
pub unsafe extern "C" fn libtpu_rs_executable_destroy(executable: *mut PJRT_LoadedExecutable) {
    if !executable.is_null() {
        // SAFETY: `executable` is non-null.
        unsafe {
            let loaded = Box::from_raw(executable);
            drop(Box::from_raw(loaded.executable_ptr));
        }
    }
}

/// Queries the dimensions of a [`PJRT_Buffer`] (`libtpu_rs_buffer_dimensions`).
///
/// # Arguments
///
/// * `buffer` - Pointer to [`PJRT_Buffer`].
/// * `out_dims` - Optional output buffer for `i64` dimensions.
/// * `max_dims` - Maximum number of dimensions writable to `out_dims`.
/// * `out_num_dims` - Output pointer to receive the rank of `buffer`.
///
/// # Safety
///
/// `buffer`, `out_dims`, and `out_num_dims` must be null or valid pointers.
#[no_mangle]
pub unsafe extern "C" fn libtpu_rs_buffer_dimensions(
    buffer: *const PJRT_Buffer,
    out_dims: *mut i64,
    max_dims: usize,
    out_num_dims: *mut usize,
) -> LibTpuCStatus {
    if any_null(&[buffer as usize, out_num_dims as usize]) {
        return c_status_err(&LibTpuError::NullPointer {
            parameter: "buffer_or_out_num_dims",
        });
    }
    // SAFETY: `buffer` is non-null.
    let dims = unsafe { &(*buffer).inner.shape.dims };
    // SAFETY: `out_num_dims` is non-null.
    unsafe {
        *out_num_dims = dims.len();
    }
    if !out_dims.is_null() {
        let copy_count = max_dims.min(dims.len());
        // SAFETY: `out_dims` has capacity for `copy_count` elements.
        unsafe {
            ptr::copy_nonoverlapping(dims.as_ptr(), out_dims, copy_count);
        }
    }
    c_status_ok()
}

/// Returns the raw `PJRT_Buffer_Type` integer of a [`PJRT_Buffer`] (`libtpu_rs_buffer_element_type`).
///
/// # Arguments
///
/// * `buffer` - Pointer to [`PJRT_Buffer`].
///
/// # Safety
///
/// `buffer` must be null or a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn libtpu_rs_buffer_element_type(buffer: *const PJRT_Buffer) -> i32 {
    if buffer.is_null() {
        return 0;
    }
    // SAFETY: `buffer` is non-null.
    unsafe { (*buffer).inner.shape.element_type.as_raw() }
}

/// Copies a [`PJRT_Buffer`] to `dst_device_id` on `client` (`libtpu_rs_buffer_copy_to_device`).
///
/// # Arguments
///
/// * `client` - Pointer to [`PJRT_Client`].
/// * `buffer` - Source [`PJRT_Buffer`].
/// * `dst_device_id` - Destination logical TPU device ID.
/// * `out_buffer` - Output pointer to receive the copied [`PJRT_Buffer`].
///
/// # Safety
///
/// All pointers must be null or valid.
#[no_mangle]
pub unsafe extern "C" fn libtpu_rs_buffer_copy_to_device(
    client: *mut PJRT_Client,
    buffer: *const PJRT_Buffer,
    dst_device_id: u32,
    out_buffer: *mut *mut PJRT_Buffer,
) -> LibTpuCStatus {
    if any_null(&[client as usize, buffer as usize, out_buffer as usize]) {
        return c_status_err(&LibTpuError::NullPointer {
            parameter: "copy_to_device_args",
        });
    }
    // SAFETY: `buffer` and `client` are non-null.
    let (buf_ref, client_ref) = unsafe { (&(*buffer).inner, &*client) };
    match buf_ref.copy_to_device(
        DeviceId(dst_device_id),
        libtpu_core::types::MemorySpaceId(0),
    ) {
        Ok(copied) => {
            let dev_ptr = client_ref
                .device_ptrs
                .get(dst_device_id as usize)
                .copied()
                .unwrap_or(ptr::null_mut());
            let mem_ptr = client_ref
                .memory_ptrs
                .first()
                .copied()
                .unwrap_or(ptr::null_mut());
            let handle = Box::into_raw(Box::new(PJRT_Buffer {
                inner: copied,
                device_ptr: dev_ptr,
                memory_ptr: mem_ptr,
            }));
            // SAFETY: `out_buffer` is non-null.
            unsafe {
                *out_buffer = handle;
            }
            c_status_ok()
        }
        Err(e) => c_status_err(&e),
    }
}

/// Executes a [`PJRT_LoadedExecutable`] with arbitrary `num_inputs` and `num_outputs` (`libtpu_rs_execute_multi`).
///
/// # Arguments
///
/// * `executable` - Pointer to [`PJRT_LoadedExecutable`].
/// * `device_id` - Target logical TPU device ID.
/// * `inputs` - Array of input `*const PJRT_Buffer` pointers.
/// * `num_inputs` - Number of input buffers.
/// * `outputs_out` - Caller-allocated array of `*mut PJRT_Buffer` of length `num_outputs`.
/// * `num_outputs` - Number of output buffer slots in `outputs_out`.
///
/// # Safety
///
/// All pointers must be null or valid for their respective counts.
#[no_mangle]
pub unsafe extern "C" fn libtpu_rs_execute_multi(
    executable: *const PJRT_LoadedExecutable,
    device_id: u32,
    inputs: *const *const PJRT_Buffer,
    num_inputs: usize,
    outputs_out: *mut *mut PJRT_Buffer,
    num_outputs: usize,
) -> LibTpuCStatus {
    if any_null(&[executable as usize, outputs_out as usize]) {
        return c_status_err(&LibTpuError::NullPointer {
            parameter: "execute_multi_args",
        });
    }
    let mut core_inputs = Vec::with_capacity(num_inputs);
    if !inputs.is_null() {
        // SAFETY: `inputs` is non-null with `num_inputs` elements.
        let slice = unsafe { std::slice::from_raw_parts(inputs, num_inputs) };
        for &ptr_item in slice {
            if ptr_item.is_null() {
                return c_status_err(&LibTpuError::NullPointer {
                    parameter: "input_buffer",
                });
            }
            // SAFETY: `ptr_item` is non-null.
            core_inputs.push(unsafe { &(*ptr_item).inner });
        }
    }
    // SAFETY: `executable` is non-null.
    let exec_ref = unsafe { &(*executable).inner };
    match exec_ref.execute_on_device(
        DeviceId(device_id),
        libtpu_core::types::MemorySpaceId(0),
        &core_inputs,
    ) {
        Ok(outs) => {
            for (idx, out_buf) in outs.into_iter().take(num_outputs).enumerate() {
                let handle = Box::into_raw(Box::new(PJRT_Buffer {
                    inner: out_buf,
                    device_ptr: ptr::null_mut(),
                    memory_ptr: ptr::null_mut(),
                }));
                // SAFETY: `outputs_out` has `num_outputs` capacity.
                unsafe {
                    *outputs_out.add(idx) = handle;
                }
            }
            c_status_ok()
        }
        Err(e) => c_status_err(&e),
    }
}

/// Serializes a [`PJRT_LoadedExecutable`] into `out_bytes` (`libtpu_rs_executable_serialize`).
///
/// # Arguments
///
/// * `executable` - Pointer to [`PJRT_LoadedExecutable`].
/// * `out_bytes` - Optional output byte buffer (pass `NULL` to query size).
/// * `max_len` - Capacity of `out_bytes` in bytes.
/// * `out_len` - Output pointer receiving the total serialized byte length.
///
/// # Safety
///
/// All pointers must be null or valid.
#[no_mangle]
pub unsafe extern "C" fn libtpu_rs_executable_serialize(
    executable: *const PJRT_LoadedExecutable,
    out_bytes: *mut u8,
    max_len: usize,
    out_len: *mut usize,
) -> LibTpuCStatus {
    if any_null(&[executable as usize, out_len as usize]) {
        return c_status_err(&LibTpuError::NullPointer {
            parameter: "executable_or_out_len",
        });
    }
    // SAFETY: `executable` is non-null.
    let bytes = unsafe { (*executable).inner.executable.program.serialize() };
    // SAFETY: `out_len` is non-null.
    unsafe {
        *out_len = bytes.len();
    }
    if !out_bytes.is_null() {
        let copy_len = max_len.min(bytes.len());
        // SAFETY: `out_bytes` has `copy_len` capacity.
        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr(), out_bytes, copy_len);
        }
    }
    c_status_ok()
}

/// Deserializes and loads a [`PJRT_LoadedExecutable`] on `client` (`libtpu_rs_executable_deserialize`).
///
/// # Arguments
///
/// * `client` - Pointer to [`PJRT_Client`].
/// * `bytes` - Pointer to serialized executable bytes.
/// * `bytes_len` - Length of `bytes` in bytes.
/// * `exec_out` - Output pointer to receive the loaded [`PJRT_LoadedExecutable`].
///
/// # Safety
///
/// All pointers must be null or valid.
#[no_mangle]
pub unsafe extern "C" fn libtpu_rs_executable_deserialize(
    client: *mut PJRT_Client,
    bytes: *const u8,
    bytes_len: usize,
    exec_out: *mut *mut PJRT_LoadedExecutable,
) -> LibTpuCStatus {
    if any_null(&[client as usize, bytes as usize, exec_out as usize]) {
        return c_status_err(&LibTpuError::NullPointer {
            parameter: "deserialize_args",
        });
    }
    // SAFETY: `bytes` and `client` are non-null.
    let (slice, client_ref) = unsafe { (std::slice::from_raw_parts(bytes, bytes_len), &*client) };
    match client_ref.inner.deserialize_and_load(slice) {
        Ok(loaded) => {
            let exec_ptr = Box::into_raw(Box::new(PJRT_Executable {
                inner: loaded.executable.clone(),
            }));
            let loaded_ptr = Box::into_raw(Box::new(PJRT_LoadedExecutable {
                inner: loaded,
                executable_ptr: exec_ptr,
                device_ptrs: client_ref.addressable_device_ptrs.clone(),
            }));
            // SAFETY: `exec_out` is non-null.
            unsafe {
                *exec_out = loaded_ptr;
            }
            c_status_ok()
        }
        Err(e) => c_status_err(&e),
    }
}

/// Copies the 32-character hex fingerprint of `executable` into `out_buf` (`libtpu_rs_executable_fingerprint`).
///
/// # Arguments
///
/// * `executable` - Pointer to [`PJRT_LoadedExecutable`].
/// * `out_buf` - Optional destination character buffer.
/// * `buf_len` - Capacity of `out_buf` in bytes.
///
/// # Safety
///
/// `executable` and `out_buf` must be null or valid pointers.
#[no_mangle]
pub unsafe extern "C" fn libtpu_rs_executable_fingerprint(
    executable: *const PJRT_LoadedExecutable,
    out_buf: *mut c_char,
    buf_len: usize,
) -> usize {
    if executable.is_null() {
        return 0;
    }
    // SAFETY: `executable` is non-null.
    let fp = unsafe {
        (*executable)
            .inner
            .executable
            .program
            .fingerprint
            .as_bytes()
    };
    if !out_buf.is_null() && buf_len > 0 {
        let copy_len = fp.len().min(buf_len.saturating_sub(1));
        // SAFETY: `out_buf` has at least `copy_len + 1` bytes.
        unsafe {
            ptr::copy_nonoverlapping(fp.as_ptr().cast::<c_char>(), out_buf, copy_len);
            *out_buf.add(copy_len) = 0;
        }
    }
    fp.len()
}

/// Returns the estimated FLOPs of `executable` (`libtpu_rs_executable_flops`).
///
/// # Arguments
///
/// * `executable` - Pointer to [`PJRT_LoadedExecutable`].
///
/// # Safety
///
/// `executable` must be null or a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn libtpu_rs_executable_flops(
    executable: *const PJRT_LoadedExecutable,
) -> f32 {
    if executable.is_null() {
        return 0.0;
    }
    // SAFETY: `executable` is non-null.
    unsafe { (*executable).inner.executable.program.cost_analysis.flops }
}

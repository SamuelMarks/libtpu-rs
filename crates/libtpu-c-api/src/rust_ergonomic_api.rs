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
#[unsafe(no_mangle)]
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn libtpu_rs_client_create(
    client_out: *mut *mut PJRT_Client,
) -> LibTpuCStatus {
    if client_out.is_null() {
        return c_status_err(&LibTpuError::NullPointer {
            parameter: "client_out",
        });
    }
    let mut args = PJRT_Client_Create_Args {
        struct_size: size_of::<PJRT_Client_Create_Args>(),
        extension_start: ptr::null_mut(),
        create_options: ptr::null(),
        num_options: 0,
        kv_get_callback: ptr::null(),
        kv_get_user_arg: ptr::null_mut(),
        kv_put_callback: ptr::null(),
        kv_put_user_arg: ptr::null_mut(),
        kv_try_get_callback: ptr::null(),
        kv_try_get_user_arg: ptr::null_mut(),
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn libtpu_rs_client_destroy(client: *mut PJRT_Client) {
    if client.is_null() {
        return;
    }
    let mut args = PJRT_Client_Destroy_Args {
        struct_size: size_of::<PJRT_Client_Destroy_Args>(),
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
#[unsafe(no_mangle)]
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
#[unsafe(no_mangle)]
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
            let handle = Box::into_raw(Box::new(PJRT_Buffer::new(core_buf, dev_ptr, mem_ptr)));
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
#[unsafe(no_mangle)]
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
#[unsafe(no_mangle)]
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
#[unsafe(no_mangle)]
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
            let exec_ptr = Box::into_raw(Box::new(PJRT_Executable::new(loaded.executable.clone())));
            let loaded_ptr = Box::into_raw(Box::new(PJRT_LoadedExecutable::new(
                loaded,
                exec_ptr,
                client_ref.addressable_device_ptrs.clone(),
            )));
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn libtpu_rs_execute(
    executable: *const PJRT_LoadedExecutable,
    device_id: u32,
    inputs: *const *const PJRT_Buffer,
    num_inputs: usize,
    output_out: *mut *mut PJRT_Buffer,
) -> LibTpuCStatus {
    // SAFETY: Delegates directly to `libtpu_rs_execute_multi` with `num_outputs = 1`.
    unsafe { libtpu_rs_execute_multi(executable, device_id, inputs, num_inputs, output_out, 1) }
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
#[unsafe(no_mangle)]
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
#[unsafe(no_mangle)]
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
#[unsafe(no_mangle)]
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
#[unsafe(no_mangle)]
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
            let handle = Box::into_raw(Box::new(PJRT_Buffer::new(copied, dev_ptr, mem_ptr)));
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
#[unsafe(no_mangle)]
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
    let loaded_wrapper = unsafe { &*executable };
    let exec_ref = &loaded_wrapper.inner;
    let dev_ptr = loaded_wrapper
        .device_ptrs
        .get(device_id as usize)
        .copied()
        .unwrap_or(ptr::null_mut());
    // SAFETY: `dev_ptr` is null or a valid `PJRT_Device*` owned by the client.
    let mem_ptr = unsafe { dev_ptr.as_ref() }.map_or(ptr::null_mut(), |d| d.default_memory_ptr);
    match exec_ref.execute_on_device(
        DeviceId(device_id),
        libtpu_core::types::MemorySpaceId(0),
        &core_inputs,
    ) {
        Ok(outs) => {
            for (idx, out_buf) in outs.into_iter().take(num_outputs).enumerate() {
                let handle = Box::into_raw(Box::new(PJRT_Buffer::new(out_buf, dev_ptr, mem_ptr)));
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
#[unsafe(no_mangle)]
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
#[unsafe(no_mangle)]
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
            let exec_ptr = Box::into_raw(Box::new(PJRT_Executable::new(loaded.executable.clone())));
            let loaded_ptr = Box::into_raw(Box::new(PJRT_LoadedExecutable::new(
                loaded,
                exec_ptr,
                client_ref.addressable_device_ptrs.clone(),
            )));
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
#[unsafe(no_mangle)]
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn libtpu_rs_executable_flops(
    executable: *const PJRT_LoadedExecutable,
) -> f32 {
    if executable.is_null() {
        return 0.0;
    }
    // SAFETY: `executable` is non-null.
    unsafe { (*executable).inner.executable.program.cost_analysis.flops }
}

#[cfg(test)]
#[allow(
    clippy::cognitive_complexity,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::float_cmp
)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::pjrt_buffer_api::{PJRT_Buffer_DynamicDimensionIndices, PJRT_Buffer_ReadyEvent};
    use crate::pjrt_client_api::{
        PJRT_Client_CreateAliasBuffer, PJRT_Client_CreateErrorBuffer,
        PJRT_Client_FulfillAliasBuffer, PJRT_Client_UpdateGlobalProcessInfo,
        PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace,
    };
    use crate::pjrt_error_api::{PJRT_Error_Destroy, PJRT_Event_Destroy, PJRT_Event_Error};

    #[test]
    fn test_libtpu_rs_ergonomic_api_and_advanced_lifecycle() {
        // SAFETY: Testing all libtpu_rs_* entrypoints and advanced buffer/client/extension paths.
        unsafe {
            let bad_st = libtpu_rs_client_create(ptr::null_mut());
            assert_ne!(format!("{bad_st:?}"), "");
            assert_ne!(bad_st.code, 0);
            libtpu_rs_status_free(bad_st);
            libtpu_rs_status_free(c_status_ok());

            let mut rs_client: *mut PJRT_Client = ptr::null_mut();
            let st_ok = libtpu_rs_client_create(ptr::addr_of_mut!(rs_client));
            assert_eq!(st_ok.code, 0);
            assert_eq!(libtpu_rs_client_device_count(ptr::null()), -1);
            assert_eq!(libtpu_rs_client_device_count(rs_client), 4);

            let dims = [4_i64];
            let mut in0 = Vec::new();
            let mut in1 = Vec::new();
            for f in [2.0_f32, 4.0, 6.0, 8.0] {
                in0.extend_from_slice(&f.to_le_bytes());
            }
            for f in [1.0_f32, 2.0, 3.0, 4.0] {
                in1.extend_from_slice(&f.to_le_bytes());
            }

            let mut b0: *mut PJRT_Buffer = ptr::null_mut();
            let mut b1: *mut PJRT_Buffer = ptr::null_mut();
            let mut b_scalar: *mut PJRT_Buffer = ptr::null_mut();

            libtpu_rs_status_free(libtpu_rs_buffer_from_host(
                ptr::null_mut(),
                0,
                11,
                dims.as_ptr(),
                1,
                in0.as_ptr(),
                in0.len(),
                ptr::addr_of_mut!(b0),
            ));
            libtpu_rs_status_free(libtpu_rs_buffer_from_host(
                rs_client,
                0,
                999,
                dims.as_ptr(),
                1,
                in0.as_ptr(),
                in0.len(),
                ptr::addr_of_mut!(b0),
            ));
            let neg_dims = [-1_i64];
            libtpu_rs_status_free(libtpu_rs_buffer_from_host(
                rs_client,
                0,
                11,
                neg_dims.as_ptr(),
                1,
                in0.as_ptr(),
                in0.len(),
                ptr::addr_of_mut!(b0),
            ));
            libtpu_rs_status_free(libtpu_rs_buffer_from_host(
                rs_client,
                99,
                11,
                dims.as_ptr(),
                1,
                in0.as_ptr(),
                in0.len(),
                ptr::addr_of_mut!(b0),
            ));

            assert_eq!(
                libtpu_rs_buffer_from_host(
                    rs_client,
                    0,
                    11,
                    ptr::null(),
                    0,
                    ptr::null(),
                    0,
                    ptr::addr_of_mut!(b_scalar),
                )
                .code,
                0
            );
            assert_ne!(format!("{:?}", *b_scalar), "");
            libtpu_rs_buffer_destroy(b_scalar);

            assert_eq!(
                libtpu_rs_buffer_from_host(
                    rs_client,
                    0,
                    11,
                    dims.as_ptr(),
                    1,
                    in0.as_ptr(),
                    in0.len(),
                    ptr::addr_of_mut!(b0),
                )
                .code,
                0
            );
            assert_eq!(
                libtpu_rs_buffer_from_host(
                    rs_client,
                    0,
                    11,
                    dims.as_ptr(),
                    1,
                    in1.as_ptr(),
                    in1.len(),
                    ptr::addr_of_mut!(b1),
                )
                .code,
                0
            );

            let mut exec: *mut PJRT_LoadedExecutable = ptr::null_mut();
            let code_src = b"module @add { stablehlo.add }";
            libtpu_rs_status_free(libtpu_rs_compile(
                ptr::null_mut(),
                c"mlir".as_ptr(),
                code_src.as_ptr(),
                code_src.len(),
                ptr::addr_of_mut!(exec),
            ));
            libtpu_rs_status_free(libtpu_rs_compile(
                rs_client,
                c"bad".as_ptr(),
                code_src.as_ptr(),
                code_src.len(),
                ptr::addr_of_mut!(exec),
            ));
            assert_eq!(
                libtpu_rs_compile(
                    rs_client,
                    c"mlir".as_ptr(),
                    code_src.as_ptr(),
                    code_src.len(),
                    ptr::addr_of_mut!(exec),
                )
                .code,
                0
            );

            let inputs = [b0.cast_const(), b1.cast_const()];
            let mut out_buf: *mut PJRT_Buffer = ptr::null_mut();
            libtpu_rs_status_free(libtpu_rs_execute(
                ptr::null(),
                0,
                inputs.as_ptr(),
                2,
                ptr::addr_of_mut!(out_buf),
            ));
            let null_inputs: [*const PJRT_Buffer; 1] = [ptr::null()];
            libtpu_rs_status_free(libtpu_rs_execute(
                exec,
                0,
                null_inputs.as_ptr(),
                1,
                ptr::addr_of_mut!(out_buf),
            ));
            libtpu_rs_status_free(libtpu_rs_execute(
                exec,
                99,
                ptr::null(),
                0,
                ptr::addr_of_mut!(out_buf),
            ));
            assert_eq!(
                libtpu_rs_execute(exec, 0, inputs.as_ptr(), 2, ptr::addr_of_mut!(out_buf)).code,
                0
            );

            let mut host_out = [0_u8; 16];
            libtpu_rs_status_free(libtpu_rs_buffer_to_host(
                ptr::null(),
                host_out.as_mut_ptr(),
                16,
            ));
            libtpu_rs_status_free(libtpu_rs_buffer_to_host(
                out_buf,
                host_out.as_mut_ptr(),
                1024,
            ));
            assert_eq!(
                libtpu_rs_buffer_to_host(out_buf, host_out.as_mut_ptr(), 16).code,
                0
            );
            let f0 = f32::from_le_bytes([host_out[0], host_out[1], host_out[2], host_out[3]]);
            assert!((f0 - 3.0).abs() < 1e-5);

            let mut rank_out: usize = 0;
            let mut queried_dims = [0_i64; 4];
            libtpu_rs_status_free(libtpu_rs_buffer_dimensions(
                ptr::null(),
                ptr::null_mut(),
                0,
                ptr::addr_of_mut!(rank_out),
            ));
            assert_eq!(
                libtpu_rs_buffer_dimensions(b0, ptr::null_mut(), 0, ptr::addr_of_mut!(rank_out))
                    .code,
                0
            );
            assert_eq!(rank_out, 1);
            assert_eq!(
                libtpu_rs_buffer_dimensions(
                    b0,
                    queried_dims.as_mut_ptr(),
                    4,
                    ptr::addr_of_mut!(rank_out),
                )
                .code,
                0
            );
            assert_eq!(queried_dims[0], 4);
            assert_eq!(libtpu_rs_buffer_element_type(ptr::null()), 0);
            assert_eq!(libtpu_rs_buffer_element_type(b0), 11);

            let mut copied_buf: *mut PJRT_Buffer = ptr::null_mut();
            libtpu_rs_status_free(libtpu_rs_buffer_copy_to_device(
                ptr::null_mut(),
                b0,
                1,
                ptr::addr_of_mut!(copied_buf),
            ));
            libtpu_rs_status_free(libtpu_rs_buffer_copy_to_device(
                rs_client,
                b0,
                99,
                ptr::addr_of_mut!(copied_buf),
            ));
            assert_eq!(
                libtpu_rs_buffer_copy_to_device(rs_client, b0, 1, ptr::addr_of_mut!(copied_buf))
                    .code,
                0
            );
            libtpu_rs_buffer_destroy(copied_buf);

            let mut multi_outs: [*mut PJRT_Buffer; 1] = [ptr::null_mut()];
            libtpu_rs_status_free(libtpu_rs_execute_multi(
                ptr::null(),
                0,
                inputs.as_ptr(),
                2,
                multi_outs.as_mut_ptr(),
                1,
            ));
            libtpu_rs_status_free(libtpu_rs_execute_multi(
                exec,
                0,
                null_inputs.as_ptr(),
                1,
                multi_outs.as_mut_ptr(),
                1,
            ));
            libtpu_rs_status_free(libtpu_rs_execute_multi(
                exec,
                99,
                ptr::null(),
                0,
                multi_outs.as_mut_ptr(),
                1,
            ));
            assert_eq!(
                libtpu_rs_execute_multi(exec, 0, inputs.as_ptr(), 2, multi_outs.as_mut_ptr(), 1)
                    .code,
                0
            );
            libtpu_rs_buffer_destroy(multi_outs[0]);

            let mut ser_len: usize = 0;
            libtpu_rs_status_free(libtpu_rs_executable_serialize(
                ptr::null(),
                ptr::null_mut(),
                0,
                ptr::addr_of_mut!(ser_len),
            ));
            assert_eq!(
                libtpu_rs_executable_serialize(
                    exec,
                    ptr::null_mut(),
                    0,
                    ptr::addr_of_mut!(ser_len)
                )
                .code,
                0
            );
            assert!(ser_len > 0);
            let mut ser_bytes = vec![0_u8; ser_len];
            assert_eq!(
                libtpu_rs_executable_serialize(
                    exec,
                    ser_bytes.as_mut_ptr(),
                    ser_len,
                    ptr::addr_of_mut!(ser_len),
                )
                .code,
                0
            );

            let mut deser_exec: *mut PJRT_LoadedExecutable = ptr::null_mut();
            libtpu_rs_status_free(libtpu_rs_executable_deserialize(
                ptr::null_mut(),
                ser_bytes.as_ptr(),
                ser_len,
                ptr::addr_of_mut!(deser_exec),
            ));
            libtpu_rs_status_free(libtpu_rs_executable_deserialize(
                rs_client,
                b"bad".as_ptr(),
                3,
                ptr::addr_of_mut!(deser_exec),
            ));
            assert_eq!(
                libtpu_rs_executable_deserialize(
                    rs_client,
                    ser_bytes.as_ptr(),
                    ser_len,
                    ptr::addr_of_mut!(deser_exec),
                )
                .code,
                0
            );
            libtpu_rs_executable_destroy(deser_exec);

            assert_eq!(
                libtpu_rs_executable_fingerprint(ptr::null(), ptr::null_mut(), 0),
                0
            );
            assert_eq!(
                libtpu_rs_executable_fingerprint(exec, ptr::null_mut(), 0),
                32
            );
            let mut fp_buf = [0 as c_char; 40];
            assert_eq!(
                libtpu_rs_executable_fingerprint(exec, fp_buf.as_mut_ptr(), 0),
                32
            );
            assert_eq!(
                libtpu_rs_executable_fingerprint(exec, fp_buf.as_mut_ptr(), fp_buf.len()),
                32
            );
            assert_eq!(libtpu_rs_executable_flops(ptr::null()), 0.0);
            assert!(libtpu_rs_executable_flops(exec) > 0.0);

            (*out_buf).inner.delete().ok();
            libtpu_rs_status_free(libtpu_rs_buffer_to_host(out_buf, host_out.as_mut_ptr(), 16));

            if let Ok(dyn_shape) =
                Shape::new(ElementType::F32, vec![4, 8], Some(vec![false, true]), None)
            {
                let dyn_core_buf = libtpu_core::pjrt::PjrtCoreBuffer::new(
                    DeviceId(0),
                    libtpu_core::types::MemorySpaceId(0),
                    dyn_shape,
                    libtpu_core::types::DeviceAddress(0x1000),
                    std::sync::Arc::clone(&(*rs_client).inner.hal),
                );
                let dyn_buf = Box::into_raw(Box::new(PJRT_Buffer::new(
                    dyn_core_buf,
                    ptr::null_mut(),
                    ptr::null_mut(),
                )));
                let mut dyn_args = crate::pjrt_types::PJRT_Buffer_DynamicDimensionIndices_Args {
                    struct_size: size_of::<
                        crate::pjrt_types::PJRT_Buffer_DynamicDimensionIndices_Args,
                    >(),
                    extension_start: ptr::null_mut(),
                    buffer: dyn_buf,
                    dynamic_dimension_indices: ptr::null(),
                    num_dynamic_dimensions: 0,
                };
                assert!(
                    PJRT_Buffer_DynamicDimensionIndices(ptr::addr_of_mut!(dyn_args).cast())
                        .is_null()
                );
                assert_eq!(dyn_args.num_dynamic_dimensions, 1);
                assert!(!dyn_args.dynamic_dimension_indices.is_null());
                libtpu_rs_buffer_destroy(dyn_buf);
            }

            let mem0 = (*rs_client)
                .memory_ptrs
                .first()
                .copied()
                .unwrap_or(ptr::null_mut());
            let dims_arr: [i64; 2] = [2, 2];
            let err_msg = b"custom buffer error";
            let mut err_buf_args = crate::pjrt_types::PJRT_Client_CreateErrorBuffer_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Client_CreateErrorBuffer_Args>(),
                extension_start: ptr::null_mut(),
                client: rs_client,
                dims: dims_arr.as_ptr(),
                num_dims: 2,
                element_type: 11,
                error_code: 13,
                error_message: err_msg.as_ptr().cast::<c_char>(),
                error_message_size: err_msg.len(),
                memory: mem0,
                layout: ptr::null(),
                device: ptr::null_mut(),
                reserved: 0,
                reserved1: 0,
                buffer: ptr::null_mut(),
            };
            assert!(
                PJRT_Client_CreateErrorBuffer(ptr::addr_of_mut!(err_buf_args).cast()).is_null()
            );
            let mut ready_ev_args = crate::pjrt_types::PJRT_Buffer_ReadyEvent_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_ReadyEvent_Args>(),
                extension_start: ptr::null_mut(),
                buffer: err_buf_args.buffer,
                event: ptr::null_mut(),
            };
            assert!(PJRT_Buffer_ReadyEvent(ptr::addr_of_mut!(ready_ev_args).cast()).is_null());
            let mut ev_err_args = crate::pjrt_types::PJRT_Event_Error_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Event_Error_Args>(),
                extension_start: ptr::null_mut(),
                event: ready_ev_args.event,
            };
            let ev_err = PJRT_Event_Error(ptr::addr_of_mut!(ev_err_args).cast());
            assert!(!ev_err.is_null());
            let mut err_destroy = crate::pjrt_types::PJRT_Error_Destroy_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Error_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                error: ev_err,
            };
            PJRT_Error_Destroy(ptr::addr_of_mut!(err_destroy).cast());
            let mut ev_destroy = crate::pjrt_types::PJRT_Event_Destroy_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Event_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                event: ready_ev_args.event,
            };
            assert!(PJRT_Event_Destroy(ptr::addr_of_mut!(ev_destroy).cast()).is_null());
            libtpu_rs_buffer_destroy(err_buf_args.buffer);

            let mut alias_args = crate::pjrt_types::PJRT_Client_CreateAliasBuffer_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Client_CreateAliasBuffer_Args>(),
                extension_start: ptr::null_mut(),
                client: rs_client,
                memory: ptr::null_mut(),
                dims: dims_arr.as_ptr(),
                num_dims: 2,
                element_type: 11,
                pad: 0,
                layout: ptr::null(),
                alias_buffer: ptr::null_mut(),
                fulfill_alias_buffer_cb: ptr::null_mut(),
            };
            assert!(PJRT_Client_CreateAliasBuffer(ptr::addr_of_mut!(alias_args).cast()).is_null());
            let mut fulfill_ok = crate::pjrt_types::PJRT_Client_FulfillAliasBuffer_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Client_FulfillAliasBuffer_Args>(),
                extension_start: ptr::null_mut(),
                client: rs_client,
                buffer: b0,
                status_code: 0,
                pad: 0,
                error_message: ptr::null(),
                error_message_size: 0,
                fulfill_alias_buffer_cb: alias_args.fulfill_alias_buffer_cb,
            };
            assert!(PJRT_Client_FulfillAliasBuffer(ptr::addr_of_mut!(fulfill_ok).cast()).is_null());
            fulfill_ok.buffer = ptr::null_mut();
            assert!(PJRT_Client_FulfillAliasBuffer(ptr::addr_of_mut!(fulfill_ok).cast()).is_null());
            fulfill_ok.status_code = 13;
            assert!(PJRT_Client_FulfillAliasBuffer(ptr::addr_of_mut!(fulfill_ok).cast()).is_null());
            fulfill_ok.error_message = err_msg.as_ptr().cast::<c_char>();
            fulfill_ok.error_message_size = err_msg.len();
            assert!(PJRT_Client_FulfillAliasBuffer(ptr::addr_of_mut!(fulfill_ok).cast()).is_null());
            libtpu_rs_buffer_destroy(alias_args.alias_buffer);

            let proc_info = crate::pjrt_types::PJRT_ProcessInfo {
                struct_size: size_of::<crate::pjrt_types::PJRT_ProcessInfo>(),
                task_id: 0,
                state: 1,
                incarnation_id: 7,
                error_code: 0,
            };
            let mut proc_args = crate::pjrt_types::PJRT_Client_UpdateGlobalProcessInfo_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Client_UpdateGlobalProcessInfo_Args>(
                ),
                extension_start: ptr::null_mut(),
                client: rs_client,
                processes: ptr::null(),
                num_processes: 0,
            };
            assert!(
                PJRT_Client_UpdateGlobalProcessInfo(ptr::addr_of_mut!(proc_args).cast()).is_null()
            );
            proc_args.processes = ptr::addr_of!(proc_info);
            proc_args.num_processes = 1;
            assert!(
                PJRT_Client_UpdateGlobalProcessInfo(ptr::addr_of_mut!(proc_args).cast()).is_null()
            );

            let mut out_layout = crate::pjrt_types::PJRT_Buffer_MemoryLayout {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_MemoryLayout>(),
                extension_start: ptr::null_mut(),
                minor_to_major: ptr::null(),
                minor_to_major_size: 0,
                tile_dims: ptr::null(),
                tile_dim_sizes: ptr::null(),
            };
            let mut canon_args =
                crate::pjrt_types::PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args {
                    struct_size: size_of::<
                        crate::pjrt_types::PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args,
                    >(),
                    extension_start: ptr::null_mut(),
                    topology: (*rs_client).topology_ptr,
                    dims: dims_arr.as_ptr(),
                    num_dims: 2,
                    element_type: 11,
                    memory_space_kind: 0,
                    layout: ptr::addr_of_mut!(out_layout),
                    reserved: 0,
                    reserved1: 0,
                    reserved2: 0,
                    reserved3: 0,
                };
            assert!(
                PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace(
                    ptr::addr_of_mut!(canon_args).cast()
                )
                .is_null()
            );
            assert!(canon_args.reserved > 0);
            assert_eq!(out_layout.minor_to_major_size, 2);

            let mut raw_q = crate::extensions::PJRT_RawBuffer_Query_Args {
                struct_size: size_of::<crate::extensions::PJRT_RawBuffer_Query_Args>(),
                extension_start: ptr::null_mut(),
                buffer: b0.cast(),
                out_value: 0,
            };
            assert!(
                crate::extensions::PJRT_RawBuffer_GetHostPointer(ptr::addr_of_mut!(raw_q).cast())
                    .is_null()
            );
            assert!(
                crate::extensions::PJRT_RawBuffer_GetOnDeviceSizeInBytes(
                    ptr::addr_of_mut!(raw_q).cast()
                )
                .is_null()
            );
            assert!(
                crate::extensions::PJRT_RawBuffer_GetMemorySpace(ptr::addr_of_mut!(raw_q).cast())
                    .is_null()
            );
            assert!(
                crate::extensions::PJRT_RawBuffer_CreateRawAliasOfBuffer(
                    ptr::addr_of_mut!(raw_q).cast()
                )
                .is_null()
            );
            assert!(raw_q.out_value > 0);
            let mut raw_del_q = crate::extensions::PJRT_RawBuffer_Destroy_Args {
                struct_size: size_of::<crate::extensions::PJRT_RawBuffer_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                raw_buffer: raw_q.out_value as *mut std::ffi::c_void,
            };
            assert!(
                crate::extensions::PJRT_RawBuffer_Destroy(ptr::addr_of_mut!(raw_del_q).cast())
                    .is_null()
            );

            let mut buf_layout_q = crate::extensions::PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args {
                struct_size: size_of::<crate::extensions::PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args>(
                ),
                extension_start: ptr::null_mut(),
                buffer: b0.cast(),
                layout: ptr::null_mut(),
            };
            assert!(
                crate::extensions::PJRT_Layouts_Buffer_GetMemoryLayout(
                    ptr::addr_of_mut!(buf_layout_q).cast()
                )
                .is_null()
            );
            assert!(!buf_layout_q.layout.is_null());
            let mut buf_layout_del = crate::extensions::PJRT_Layouts_MemoryLayout_Destroy_Args {
                struct_size: size_of::<crate::extensions::PJRT_Layouts_MemoryLayout_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                layout: buf_layout_q.layout,
            };
            assert!(
                crate::extensions::PJRT_Layouts_MemoryLayout_Destroy(
                    ptr::addr_of_mut!(buf_layout_del).cast()
                )
                .is_null()
            );

            let phase_prog = b"module @phase { stablehlo.add }";
            let mut phase_run = crate::extensions::PJRT_PhaseCompile_RunPhase_Args {
                struct_size: size_of::<crate::extensions::PJRT_PhaseCompile_RunPhase_Args>(),
                extension_start: ptr::null_mut(),
                phase_compiler: ptr::null_mut(),
                input_bytes: phase_prog.as_ptr().cast(),
                input_size: phase_prog.len(),
                phase_name: c"hlo_verification".as_ptr(),
                output_status: 0,
            };
            assert!(
                crate::extensions::PJRT_PhaseCompile_RunPhase(ptr::addr_of_mut!(phase_run).cast())
                    .is_null()
            );
            assert!(phase_run.output_status >= 6);

            let mut ag_args = crate::extensions::PJRT_Communicator_AllGather_Args {
                struct_size: size_of::<crate::extensions::PJRT_Communicator_AllGather_Args>(),
                extension_start: ptr::null_mut(),
                communicator: ptr::null_mut(),
                send_buffer: b0.cast(),
                recv_buffer: b1.cast(),
                count: 4,
                event: ptr::null_mut(),
            };
            assert!(
                crate::extensions::PJRT_Communicator_AllGather(ptr::addr_of_mut!(ag_args).cast())
                    .is_null()
            );
            let mut cp_args = crate::extensions::PJRT_Communicator_CollectivePermute_Args {
                struct_size: size_of::<crate::extensions::PJRT_Communicator_CollectivePermute_Args>(
                ),
                extension_start: ptr::null_mut(),
                communicator: ptr::null_mut(),
                send_buffer: b0.cast(),
                recv_buffer: b1.cast(),
                target_ranks: ptr::null(),
                count: 4,
                event: ptr::null_mut(),
            };
            assert!(
                crate::extensions::PJRT_Communicator_AllToAll(ptr::addr_of_mut!(cp_args).cast())
                    .is_null()
            );
            assert!(
                crate::extensions::PJRT_Communicator_CollectivePermute(
                    ptr::addr_of_mut!(cp_args).cast()
                )
                .is_null()
            );

            let mut xh_args = crate::extensions::PJRT_CrossHostTransfers_Args {
                struct_size: size_of::<crate::extensions::PJRT_CrossHostTransfers_Args>(),
                extension_start: ptr::null_mut(),
                client: rs_client.cast(),
                buffers: b0 as usize,
                num_buffers: 1,
                remote_device: b1 as usize,
                out_event: 0,
            };
            assert!(
                crate::extensions::PJRT_CrossHostTransfers_CopyToRemoteDevice(
                    ptr::addr_of_mut!(xh_args).cast()
                )
                .is_null()
            );
            assert!(xh_args.out_event > 0);

            let exec_inner = (*exec).executable_ptr;
            let mut meta_q = crate::extensions::PJRT_Metadata_Get_Args {
                struct_size: size_of::<crate::extensions::PJRT_Metadata_Get_Args>(),
                extension_start: ptr::null_mut(),
                handle: exec_inner.cast(),
                metadata_out: 0,
            };
            assert!(
                crate::extensions::PJRT_ExecutableMetadata_Get(ptr::addr_of_mut!(meta_q).cast())
                    .is_null()
            );
            assert!(meta_q.metadata_out > 0);
            assert!(
                crate::extensions::PJRT_Layouts_Executable_GetOutputLayouts(
                    ptr::addr_of_mut!(meta_q).cast()
                )
                .is_null()
            );
            assert_eq!(meta_q.metadata_out, 1);

            let mut sh_args = crate::extensions::PJRT_Shardings_PJRT_Executable_Shardings_Args {
                struct_size: size_of::<
                    crate::extensions::PJRT_Shardings_PJRT_Executable_Shardings_Args,
                >(),
                extension_start: ptr::null_mut(),
                executable: exec_inner.cast(),
                num_items: 0,
                shardings: ptr::null(),
                sharding_sizes: ptr::null(),
            };
            assert!(
                crate::extensions::PJRT_Shardings_ParameterShardings(
                    ptr::addr_of_mut!(sh_args).cast()
                )
                .is_null()
            );
            assert_eq!(sh_args.num_items, 1);

            let mut hlo_q = crate::extensions::PJRT_TpuExecutable_ProtoQuery_Args {
                struct_size: size_of::<crate::extensions::PJRT_TpuExecutable_ProtoQuery_Args>(),
                extension_start: ptr::null_mut(),
                executable: exec_inner.cast(),
                serialized_bytes: 0,
                serialized_bytes_size: 0,
                holder: ptr::null_mut(),
            };
            assert!(
                crate::extensions::PJRT_TpuExecutable_GetHloModuleWithConfig(
                    ptr::addr_of_mut!(hlo_q).cast()
                )
                .is_null()
            );
            assert!(hlo_q.serialized_bytes_size > 0);

            let mut cost_q = crate::extensions::PJRT_Megascale_Query_Args {
                struct_size: size_of::<crate::extensions::PJRT_Megascale_Query_Args>(),
                extension_start: ptr::null_mut(),
                handle: exec_inner as usize,
                out_value: 0,
            };
            assert!(
                crate::extensions::PJRT_TpuExecutable_RunHloCostAnalysis(
                    ptr::addr_of_mut!(cost_q).cast()
                )
                .is_null()
            );
            assert!(cost_q.out_value >= 1);

            libtpu_rs_buffer_destroy(out_buf);
            libtpu_rs_buffer_destroy(b0);
            libtpu_rs_buffer_destroy(b1);
            libtpu_rs_buffer_destroy(ptr::null_mut());
            libtpu_rs_executable_destroy(exec);
            libtpu_rs_executable_destroy(ptr::null_mut());
            libtpu_rs_client_destroy(rs_client);
            libtpu_rs_client_destroy(ptr::null_mut());
        }
    }
}

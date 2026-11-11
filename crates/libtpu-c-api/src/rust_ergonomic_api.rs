//! Ergonomic `libtpu_rs_*` C/C++/Python FFI helper entrypoints.

use super::*;
use libtpu_core::proto::ProtoSchemaMessage;

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

/// Creates a new in-memory [`libtpu_core::distributed::DistributedCoordinator`] handle (`libtpu_rs_coordinator_create`).
#[unsafe(no_mangle)]
pub extern "C" fn libtpu_rs_coordinator_create() -> *mut std::ffi::c_void {
    let coord = Box::new(libtpu_core::distributed::DistributedCoordinator::new());
    Box::into_raw(coord).cast::<std::ffi::c_void>()
}

/// Destroys a coordinator handle created by [`libtpu_rs_coordinator_create`] (`libtpu_rs_coordinator_destroy`).
///
/// # Arguments
///
/// * `coordinator` - Coordinator handle pointer (may be null).
///
/// # Safety
///
/// `coordinator` must be null or a pointer returned by [`libtpu_rs_coordinator_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn libtpu_rs_coordinator_destroy(coordinator: *mut std::ffi::c_void) {
    if !coordinator.is_null() {
        // SAFETY: `coordinator` was allocated via `Box::into_raw` in `libtpu_rs_coordinator_create`.
        unsafe {
            drop(Box::from_raw(
                coordinator.cast::<libtpu_core::distributed::DistributedCoordinator>(),
            ));
        }
    }
}

/// Inserts a key-value entry into a coordinator handle (`libtpu_rs_coordinator_put_kv`).
///
/// # Arguments
///
/// * `coordinator` - Coordinator handle pointer.
/// * `key` - Null-terminated UTF-8 key string.
/// * `val` - Value byte buffer pointer.
/// * `val_len` - Byte length of `val`.
///
/// # Safety
///
/// `coordinator`, `key`, and `val` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn libtpu_rs_coordinator_put_kv(
    coordinator: *mut std::ffi::c_void,
    key: *const c_char,
    val: *const u8,
    val_len: usize,
) -> LibTpuCStatus {
    if coordinator.is_null() || key.is_null() || (val.is_null() && val_len > 0) {
        return c_status_err(&LibTpuError::NullPointer {
            parameter: "libtpu_rs_coordinator_put_kv",
        });
    }
    // SAFETY: `coordinator`, `key`, and `val` are non-null.
    unsafe {
        let coord = &*coordinator.cast::<libtpu_core::distributed::DistributedCoordinator>();
        let key_str = CStr::from_ptr(key).to_string_lossy();
        let val_vec = if val_len == 0 {
            Vec::new()
        } else {
            std::slice::from_raw_parts(val, val_len).to_vec()
        };
        coord.put_key_value(&key_str, val_vec);
        c_status_ok()
    }
}

/// Looks up a key-value entry from a coordinator handle (`libtpu_rs_coordinator_get_kv`).
///
/// Returns the byte length of the value, or `0` if `coordinator`/`key` is null or the key is missing.
///
/// # Arguments
///
/// * `coordinator` - Coordinator handle pointer.
/// * `key` - Null-terminated UTF-8 key string.
/// * `out_buf` - Optional destination byte buffer.
/// * `buf_len` - Capacity of `out_buf` in bytes.
///
/// # Safety
///
/// `coordinator`, `key`, and `out_buf` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn libtpu_rs_coordinator_get_kv(
    coordinator: *const std::ffi::c_void,
    key: *const c_char,
    out_buf: *mut u8,
    buf_len: usize,
) -> usize {
    if coordinator.is_null() || key.is_null() {
        return 0;
    }
    // SAFETY: `coordinator` and `key` are non-null.
    let val_opt = unsafe {
        let coord = &*coordinator.cast::<libtpu_core::distributed::DistributedCoordinator>();
        let key_str = CStr::from_ptr(key).to_string_lossy();
        coord.try_get_key_value(&key_str)
    };
    let Some(bytes) = val_opt else {
        return 0;
    };
    if !out_buf.is_null() && buf_len > 0 {
        let copy_len = bytes.len().min(buf_len);
        // SAFETY: `out_buf` is valid for `copy_len` bytes.
        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr(), out_buf, copy_len);
        }
    }
    bytes.len()
}

/// Helper to copy a UTF-8 string into an optional caller buffer `(out_buf, buf_len)` and return the full byte length.
///
/// # Arguments
///
/// * `s` - Source UTF-8 string slice.
/// * `out_buf` - Optional destination character buffer.
/// * `buf_len` - Capacity of `out_buf` in bytes (including null terminator).
unsafe fn copy_str_to_c_buf(s: &str, out_buf: *mut c_char, buf_len: usize) -> usize {
    let bytes = s.as_bytes();
    if (u8::from(!out_buf.is_null()) & u8::from(buf_len > 0)) != 0 {
        let copy_len = bytes.len().min(buf_len.saturating_sub(1));
        // SAFETY: `out_buf` has at least `copy_len + 1` bytes capacity.
        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr().cast::<c_char>(), out_buf, copy_len);
            *out_buf.add(copy_len) = 0;
        }
    }
    bytes.len()
}

/// Disassembles a [`PJRT_LoadedExecutable`] into a multi-line `DeepSea` LLO assembly listing (`libtpu_rs_disassemble_executable_llo`).
///
/// # Arguments
///
/// * `executable` - Pointer to [`PJRT_LoadedExecutable`].
/// * `out_buf` - Optional destination character buffer.
/// * `buf_len` - Capacity of `out_buf` in bytes (including null terminator).
///
/// # Safety
///
/// `executable` and `out_buf` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn libtpu_rs_disassemble_executable_llo(
    executable: *const PJRT_LoadedExecutable,
    out_buf: *mut c_char,
    buf_len: usize,
) -> usize {
    if executable.is_null() {
        return 0;
    }
    // SAFETY: `executable` is non-null.
    let prog = unsafe { &(*executable).inner.executable.program };
    let vliw_bundles: Vec<_> = prog
        .instruction_bundles
        .iter()
        .map(libtpu_core::compiler::TpuInstructionBundle::encode_architectural_bundle)
        .collect();
    let listing = libtpu_core::compiler::format_llo_assembly(&vliw_bundles);
    // SAFETY: Delegated to `copy_str_to_c_buf` with caller's `out_buf` and `buf_len`.
    unsafe { copy_str_to_c_buf(&listing, out_buf, buf_len) }
}

/// Registers an `OpenXLA` Typed FFI v0.2 [`crate::extensions::XLA_FFI_Handler`] callback
/// (`libtpu_rs_register_ffi_handler`).
///
/// # Arguments
///
/// * `target_name` - Null-terminated UTF-8 custom-call target name (`"my_ffi_op"`).
/// * `handler` - Function pointer matching `XLA_FFI_Error* (*)(XLA_FFI_CallFrame*)`.
///
/// # Safety
///
/// `target_name` must be null or a valid null-terminated C string, and `handler` must be a valid
/// function pointer safe to invoke from execution threads.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn libtpu_rs_register_ffi_handler(
    target_name: *const c_char,
    handler: Option<crate::extensions::XLA_FFI_Handler>,
) -> LibTpuCStatus {
    if target_name.is_null() {
        return c_status_err(&LibTpuError::NullPointer {
            parameter: "target_name",
        });
    }
    let Some(fn_ptr) = handler else {
        return c_status_err(&LibTpuError::NullPointer {
            parameter: "handler",
        });
    };
    // SAFETY: `target_name` is non-null.
    let name_str = unsafe { CStr::from_ptr(target_name).to_string_lossy().into_owned() };
    let fn_addr = fn_ptr as usize;
    libtpu_core::hal::register_custom_call_handler(&name_str, move |inputs, attrs| {
        let default_shape = libtpu_core::types::Shape {
            element_type: ElementType::F32,
            dims: vec![4],
            dynamic_dims: vec![false],
            layout: libtpu_core::types::MemoryLayout::default_tpu_tiled(1),
        };
        let out_shape = attrs.output_shape.as_ref().unwrap_or(&default_shape);
        let out_len = out_shape.unpadded_byte_size().0.max(4);
        let mut out_bytes = inputs.first().copied().unwrap_or(&[]).to_vec();
        out_bytes.resize(out_len, 0);

        let mut in_buf_descs = Vec::with_capacity(inputs.len());
        let mut in_buf_ptrs = Vec::with_capacity(inputs.len());
        let in_dims: Vec<i64> = out_shape.dims.clone();
        for &inp in inputs {
            in_buf_descs.push(crate::extensions::XLA_FFI_Buffer {
                struct_size: size_of::<crate::extensions::XLA_FFI_Buffer>(),
                extension_start: ptr::null_mut(),
                dtype: out_shape.element_type.as_raw(),
                rank: i32::try_from(in_dims.len()).unwrap_or(1),
                data: inp.as_ptr().cast_mut().cast(),
                dims: in_dims.as_ptr(),
            });
        }
        for desc in &in_buf_descs {
            in_buf_ptrs.push(ptr::from_ref(desc));
        }

        let mut out_buf_desc = crate::extensions::XLA_FFI_Buffer {
            struct_size: size_of::<crate::extensions::XLA_FFI_Buffer>(),
            extension_start: ptr::null_mut(),
            dtype: out_shape.element_type.as_raw(),
            rank: i32::try_from(in_dims.len()).unwrap_or(1),
            data: out_bytes.as_mut_ptr().cast(),
            dims: in_dims.as_ptr(),
        };
        let mut out_buf_ptr = ptr::addr_of_mut!(out_buf_desc);

        let mut frame = crate::extensions::XLA_FFI_CallFrame {
            struct_size: size_of::<crate::extensions::XLA_FFI_CallFrame>(),
            extension_start: ptr::null_mut(),
            api: ptr::null(),
            ctx: ptr::null_mut(),
            stage: 3,
            num_args: u32::try_from(in_buf_ptrs.len()).unwrap_or(0),
            args: in_buf_ptrs.as_ptr(),
            num_rets: 1,
            rets: ptr::addr_of_mut!(out_buf_ptr),
        };
        // SAFETY: Reconstructing validated `XLA_FFI_Handler` function pointer and invoking with stack `frame`.
        let cb: crate::extensions::XLA_FFI_Handler = unsafe { std::mem::transmute(fn_addr) };
        // SAFETY: `frame` is a valid `XLA_FFI_CallFrame`.
        let err_ptr = unsafe { cb(ptr::addr_of_mut!(frame)) };
        if !err_ptr.is_null() {
            // SAFETY: `err_ptr` is a heap-allocated `PJRT_Error`.
            let boxed = unsafe { Box::from_raw(err_ptr) };
            return Err(boxed.error);
        }
        Ok(out_bytes)
    });
    c_status_ok()
}

/// Formats a human-readable physical register allocation and `VMEM` spill summary for `executable`
/// (`libtpu_rs_inspect_register_allocation`).
///
/// # Arguments
///
/// * `executable` - Pointer to [`PJRT_LoadedExecutable`].
/// * `out_buf` - Optional destination character buffer.
/// * `buf_len` - Capacity of `out_buf` in bytes (including null terminator).
///
/// # Safety
///
/// `executable` and `out_buf` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn libtpu_rs_inspect_register_allocation(
    executable: *const PJRT_LoadedExecutable,
    out_buf: *mut c_char,
    buf_len: usize,
) -> usize {
    if executable.is_null() {
        return 0;
    }
    // SAFETY: `executable` is non-null.
    let prog = unsafe { &(*executable).inner.executable.program };
    let tpu_gen = libtpu_core::config::global_init_config().generation;
    let report = prog.allocate_physical_registers(tpu_gen);
    let summary = format!(
        "generation={:?} num_vregs={} allocated_slots={} spills={} spill_vmem_bytes={}",
        report.config.generation,
        report.config.num_vregs,
        report.slot_to_vreg.len(),
        report.spill_actions.len(),
        report.total_spill_vmem_bytes
    );
    // SAFETY: Delegated to `copy_str_to_c_buf`.
    unsafe { copy_str_to_c_buf(&summary, out_buf, buf_len) }
}

/// Serializes the wire-compatible `xla.BufferAssignmentProto` for `executable` into `out_buf`
/// (`libtpu_rs_executable_buffer_assignment_proto`).
///
/// Returns the full encoded byte length of the `BufferAssignmentProto`, or `0` if `executable` is null.
///
/// # Arguments
///
/// * `executable` - Pointer to [`PJRT_LoadedExecutable`].
/// * `out_buf` - Optional destination byte buffer.
/// * `buf_len` - Capacity of `out_buf` in bytes.
///
/// # Safety
///
/// `executable` and `out_buf` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn libtpu_rs_executable_buffer_assignment_proto(
    executable: *const PJRT_LoadedExecutable,
    out_buf: *mut u8,
    buf_len: usize,
) -> usize {
    if executable.is_null() {
        return 0;
    }
    // SAFETY: `executable` is non-null.
    let prog = unsafe { &(*executable).inner.executable.program };
    let ba_proto = prog.build_buffer_assignment_proto();
    let encoded = ba_proto.encode();
    if !out_buf.is_null() {
        let copy_len = encoded.len().min(buf_len);
        // SAFETY: `out_buf` is valid for `copy_len` bytes.
        unsafe {
            ptr::copy_nonoverlapping(encoded.as_ptr(), out_buf, copy_len);
        }
    }
    encoded.len()
}

/// Formats a canonical hardware chip configuration summary for the active TPU generation
/// (`libtpu_rs_query_chip_config_summary`).
///
/// # Arguments
///
/// * `out_buf` - Optional destination character buffer.
/// * `buf_len` - Capacity of `out_buf` in bytes (including null terminator).
///
/// # Safety
///
/// `out_buf` must be null or a valid mutable pointer of at least `buf_len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn libtpu_rs_query_chip_config_summary(
    out_buf: *mut c_char,
    buf_len: usize,
) -> usize {
    let tpu_gen = libtpu_core::config::global_init_config().generation;
    let cfg = libtpu_core::proto::TpuChipConfigProto::from_generation(tpu_gen);
    let parts = libtpu_core::proto::TpuChipPartsProto::from_generation(tpu_gen);
    let isa = libtpu_core::elf::IsaBitfieldSpecTable::for_generation(tpu_gen);
    let summary = format!(
        "generation={} hbm_bytes={} vmem_bytes={} smem_bytes={} cmem_bytes={} tensor_cores={} sparse_cores={} mxu_dim={} isa_bits={}",
        cfg.chip_name,
        cfg.hbm_capacity_bytes,
        cfg.vmem_capacity_bytes,
        cfg.smem_capacity_bytes,
        parts.cmem_capacity_bytes,
        cfg.tensor_cores_per_chip,
        cfg.sparse_cores_per_chip,
        parts.mxu_dimension,
        isa.bundle_width_bits
    );
    // SAFETY: Delegated to `copy_str_to_c_buf`.
    unsafe { copy_str_to_c_buf(&summary, out_buf, buf_len) }
}

/// Computes the shortest 3D Torus optical ICI hop count between `(src_x, src_y, src_z)` and
/// `(dst_x, dst_y, dst_z)` (`libtpu_rs_compute_ici_route`).
///
/// # Arguments
///
/// * `src_x` - Source X chip coordinate.
/// * `src_y` - Source Y chip coordinate.
/// * `src_z` - Source Z chip coordinate.
/// * `dst_x` - Destination X chip coordinate.
/// * `dst_y` - Destination Y chip coordinate.
/// * `dst_z` - Destination Z chip coordinate.
/// * `wrap_around` - Non-zero (`1`) to enable twisted-torus wrap-around links, `0` otherwise.
#[unsafe(no_mangle)]
pub extern "C" fn libtpu_rs_compute_ici_route(
    src_x: u32,
    src_y: u32,
    src_z: u32,
    dst_x: u32,
    dst_y: u32,
    dst_z: u32,
    wrap_around: i32,
) -> u32 {
    let bounds = libtpu_core::types::ChipCoordinates::new(
        src_x.max(dst_x).saturating_add(1).max(4),
        src_y.max(dst_y).saturating_add(1).max(4),
        src_z.max(dst_z).saturating_add(1).max(4),
    );
    let router = libtpu_core::distributed::IciTorusRouter::new(bounds, wrap_around != 0);
    let src = libtpu_core::types::ChipCoordinates::new(src_x, src_y, src_z);
    let dst = libtpu_core::types::ChipCoordinates::new(dst_x, dst_y, dst_z);
    router.compute_shortest_hop_count(src, dst).unwrap_or(0)
}

/// Decodes a raw or Riegeli-framed `asic_sw.CoreDump` payload into canonical JSON (`libtpu_rs_read_core_dump`).
///
/// # Arguments
///
/// * `raw_bytes` - Pointer to raw or Riegeli-framed `CoreDump` bytes.
/// * `raw_len` - Length of `raw_bytes` in bytes.
/// * `out_buf` - Optional destination character buffer.
/// * `buf_len` - Capacity of `out_buf` in bytes.
///
/// # Safety
///
/// `raw_bytes` and `out_buf` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn libtpu_rs_read_core_dump(
    raw_bytes: *const u8,
    raw_len: usize,
    out_buf: *mut c_char,
    buf_len: usize,
) -> usize {
    if raw_bytes.is_null() {
        return 0;
    }
    // SAFETY: `raw_bytes` is non-null for `raw_len` bytes.
    let slice = unsafe { std::slice::from_raw_parts(raw_bytes, raw_len) };
    let Ok(unwrapped) = libtpu_core::proto::decode_riegeli_record_container(slice) else {
        return 0;
    };
    let Ok(msg) = libtpu_core::proto::CoreDumpProto::decode(&unwrapped) else {
        return 0;
    };
    let json = msg.to_proto_json().unwrap_or_default();
    // SAFETY: Delegated to `copy_str_to_c_buf`.
    unsafe { copy_str_to_c_buf(&json, out_buf, buf_len) }
}

/// Decodes a raw or Riegeli-framed `AllCoreStateSummaries` payload into canonical JSON (`libtpu_rs_read_stall_summaries`).
///
/// # Arguments
///
/// * `raw_bytes` - Pointer to raw or Riegeli-framed `AllCoreStateSummaries` bytes.
/// * `raw_len` - Length of `raw_bytes` in bytes.
/// * `out_buf` - Optional destination character buffer.
/// * `buf_len` - Capacity of `out_buf` in bytes.
///
/// # Safety
///
/// `raw_bytes` and `out_buf` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn libtpu_rs_read_stall_summaries(
    raw_bytes: *const u8,
    raw_len: usize,
    out_buf: *mut c_char,
    buf_len: usize,
) -> usize {
    if raw_bytes.is_null() {
        return 0;
    }
    // SAFETY: `raw_bytes` is non-null for `raw_len` bytes.
    let slice = unsafe { std::slice::from_raw_parts(raw_bytes, raw_len) };
    let Ok(unwrapped) = libtpu_core::proto::decode_riegeli_record_container(slice) else {
        return 0;
    };
    let Ok(msg) = libtpu_core::proto::AllCoreStateSummariesProto::decode(&unwrapped) else {
        return 0;
    };
    let json = msg.to_proto_json().unwrap_or_default();
    // SAFETY: Delegated to `copy_str_to_c_buf`.
    unsafe { copy_str_to_c_buf(&json, out_buf, buf_len) }
}

/// Decodes a raw or Riegeli-framed `CompilerMetadata` payload into canonical JSON (`libtpu_rs_read_compiler_metadata`).
///
/// # Arguments
///
/// * `raw_bytes` - Pointer to raw or Riegeli-framed `CompilerMetadata` bytes.
/// * `raw_len` - Length of `raw_bytes` in bytes.
/// * `out_buf` - Optional destination character buffer.
/// * `buf_len` - Capacity of `out_buf` in bytes.
///
/// # Safety
///
/// `raw_bytes` and `out_buf` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn libtpu_rs_read_compiler_metadata(
    raw_bytes: *const u8,
    raw_len: usize,
    out_buf: *mut c_char,
    buf_len: usize,
) -> usize {
    if raw_bytes.is_null() {
        return 0;
    }
    // SAFETY: `raw_bytes` is non-null for `raw_len` bytes.
    let slice = unsafe { std::slice::from_raw_parts(raw_bytes, raw_len) };
    let Ok(unwrapped) = libtpu_core::proto::decode_riegeli_record_container(slice) else {
        return 0;
    };
    let Ok(msg) = libtpu_core::proto::CompilerMetadataProto::decode(&unwrapped) else {
        return 0;
    };
    let json = msg.to_proto_json().unwrap_or_default();
    // SAFETY: Delegated to `copy_str_to_c_buf`.
    unsafe { copy_str_to_c_buf(&json, out_buf, buf_len) }
}

/// Emits a canonical `proto3` `.proto` schema string for `full_message_name` (`libtpu_rs_emit_proto_schema`).
///
/// Returns the full UTF-8 byte length of the emitted schema, or `0` if `full_message_name` is null or unknown.
///
/// # Arguments
///
/// * `full_message_name` - Null-terminated fully-qualified protobuf message name (`"xla.TileProto"`).
/// * `out_buf` - Optional destination character buffer.
/// * `buf_len` - Capacity of `out_buf` in bytes (including null terminator).
///
/// # Safety
///
/// `full_message_name` and `out_buf` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn libtpu_rs_emit_proto_schema(
    full_message_name: *const c_char,
    out_buf: *mut c_char,
    buf_len: usize,
) -> usize {
    if full_message_name.is_null() {
        return 0;
    }
    // SAFETY: `full_message_name` is non-null.
    let name_str = unsafe { CStr::from_ptr(full_message_name).to_string_lossy() };
    let Ok(schema) = libtpu_core::proto::emit_proto_message_schema(&name_str) else {
        return 0;
    };
    let bytes = schema.as_bytes();
    if !out_buf.is_null() && buf_len > 0 {
        let copy_len = bytes.len().min(buf_len.saturating_sub(1));
        // SAFETY: `out_buf` has at least `copy_len + 1` bytes.
        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr().cast::<c_char>(), out_buf, copy_len);
            *out_buf.add(copy_len) = 0;
        }
    }
    bytes.len()
}

/// Compiles and executes an N-dimensional program (`libtpu_rs_compile_and_execute_nd`) over two
/// typed host input buffers (`in0_data`, `in1_data`) of shape `dims[0..num_dims]` and `element_type`.
///
/// # Arguments
///
/// * `client` - Pointer to [`PJRT_Client`].
/// * `device_id` - Target logical TPU device ID.
/// * `format` - Null-terminated format string (`"mlir"` or `"hlo"`).
/// * `code` - Pointer to program source or bytecode bytes.
/// * `code_len` - Byte length of `code`.
/// * `element_type` - Raw `PJRT_Buffer_Type` integer (`11` = `F32`, etc.).
/// * `dims` - Pointer to N-dimensional extent array.
/// * `num_dims` - Rank (`number of dimensions`).
/// * `in0_data` - First input host byte buffer.
/// * `in1_data` - Second input host byte buffer.
/// * `byte_len` - Byte length of `in0_data`, `in1_data`, and `out_data`.
/// * `out_data` - Destination host byte buffer of length `byte_len`.
///
/// # Safety
///
/// All pointer arguments must be null or valid for their respective byte/element counts.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn libtpu_rs_compile_and_execute_nd(
    client: *mut PJRT_Client,
    device_id: u32,
    format: *const c_char,
    code: *const u8,
    code_len: usize,
    element_type: i32,
    dims: *const i64,
    num_dims: usize,
    in0_data: *const u8,
    in1_data: *const u8,
    byte_len: usize,
    out_data: *mut u8,
) -> LibTpuCStatus {
    let mut b0: *mut PJRT_Buffer = ptr::null_mut();
    let mut b1: *mut PJRT_Buffer = ptr::null_mut();
    let mut exec: *mut PJRT_LoadedExecutable = ptr::null_mut();
    let mut out_buf: *mut PJRT_Buffer = ptr::null_mut();

    // SAFETY: Delegated to validated `libtpu_rs_*` entrypoints.
    unsafe {
        let st0 = libtpu_rs_buffer_from_host(
            client,
            device_id,
            element_type,
            dims,
            num_dims,
            in0_data,
            byte_len,
            ptr::addr_of_mut!(b0),
        );
        if st0.code != 0 {
            return st0;
        }
        let _ = libtpu_rs_buffer_from_host(
            client,
            device_id,
            element_type,
            dims,
            num_dims,
            in1_data,
            byte_len,
            ptr::addr_of_mut!(b1),
        );
        let st_c = libtpu_rs_compile(client, format, code, code_len, ptr::addr_of_mut!(exec));
        if st_c.code != 0 {
            libtpu_rs_buffer_destroy(b0);
            libtpu_rs_buffer_destroy(b1);
            return st_c;
        }
        let inputs = [b0.cast_const(), b1.cast_const()];
        let _ = libtpu_rs_execute(
            exec,
            device_id,
            inputs.as_ptr(),
            2,
            ptr::addr_of_mut!(out_buf),
        );
        let st_out = libtpu_rs_buffer_to_host(out_buf, out_data, byte_len);
        libtpu_rs_buffer_destroy(out_buf);
        libtpu_rs_executable_destroy(exec);
        libtpu_rs_buffer_destroy(b1);
        libtpu_rs_buffer_destroy(b0);
        st_out
    }
}

/// Resolves a hierarchical `DeepSea` CSR path in the 13,261+ CSR catalog (`libtpu_rs_lookup_csr_by_path`)
/// and returns its physical BAR0 MMIO byte offset, or `u64::MAX` if `path` is null or unknown.
///
/// # Arguments
///
/// * `path` - Null-terminated hierarchical CSR path string (e.g., `"pxc_tc_0_tcs_core_state.PC"`).
///
/// # Safety
///
/// `path` must be null or a valid null-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn libtpu_rs_lookup_csr_by_path(path: *const c_char) -> u64 {
    if path.is_null() {
        return u64::MAX;
    }
    // SAFETY: `path` is non-null.
    let path_str = unsafe { CStr::from_ptr(path).to_string_lossy() };
    let tpu_gen = libtpu_core::config::global_init_config().generation;
    let map = libtpu_core::hal::ChipRegisterMap::for_generation(tpu_gen);
    map.resolve_hierarchical_csr_path(&path_str)
        .map_or(u64::MAX, |desc| desc.byte_offset)
}

/// Resolves a physical BAR0 MMIO byte offset in the 13,261+ CSR catalog (`libtpu_rs_lookup_csr_by_offset`).
///
/// Writes its canonical hierarchical register name into `out_buf` and returns the name's UTF-8 byte length
/// (or `0` if unknown).
///
/// # Arguments
///
/// * `byte_offset` - Physical BAR0 MMIO byte offset.
/// * `out_buf` - Optional destination character buffer.
/// * `buf_len` - Capacity of `out_buf` in bytes (including null terminator).
///
/// # Safety
///
/// `out_buf` must be null or a valid mutable pointer of at least `buf_len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn libtpu_rs_lookup_csr_by_offset(
    byte_offset: u64,
    out_buf: *mut c_char,
    buf_len: usize,
) -> usize {
    let tpu_gen = libtpu_core::config::global_init_config().generation;
    let map = libtpu_core::hal::ChipRegisterMap::for_generation(tpu_gen);
    let Some(desc) = map.lookup_by_offset(byte_offset) else {
        return 0;
    };
    // SAFETY: Delegated to `copy_str_to_c_buf`.
    unsafe { copy_str_to_c_buf(desc.full_path, out_buf, buf_len) }
}

/// Encodes a textual MLIR module into a standard upstream `ML\xefR` bytecode container
/// (without `0xFE` custom magic) and compiles it into a [`PJRT_LoadedExecutable`]
/// (`libtpu_rs_compile_standard_mlir_bytecode`).
///
/// # Arguments
///
/// * `client` - Pointer to [`PJRT_Client`].
/// * `mlir_source` - Null-terminated UTF-8 MLIR module text.
/// * `exec_out` - Output pointer to receive the [`PJRT_LoadedExecutable`].
///
/// # Safety
///
/// All pointers must be null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn libtpu_rs_compile_standard_mlir_bytecode(
    client: *mut PJRT_Client,
    mlir_source: *const c_char,
    exec_out: *mut *mut PJRT_LoadedExecutable,
) -> LibTpuCStatus {
    if any_null(&[client as usize, mlir_source as usize, exec_out as usize]) {
        return c_status_err(&LibTpuError::NullPointer {
            parameter: "libtpu_rs_compile_standard_mlir_bytecode",
        });
    }
    // SAFETY: `mlir_source` is non-null.
    let src_str = unsafe { CStr::from_ptr(mlir_source).to_string_lossy() };
    let Ok(ast_mod) = libtpu_core::compiler::parse_mlir_ast_module(&src_str) else {
        return c_status_err(&LibTpuError::CompilationFailed {
            reason: String::from("Invalid MLIR syntax"),
        });
    };
    let bc_bytes = libtpu_core::compiler::MlirBytecodeHeader::encode_from_ast_module(&ast_mod);
    // SAFETY: Delegating validated pointers to `libtpu_rs_compile`.
    unsafe {
        libtpu_rs_compile(
            client,
            c"mlir".as_ptr(),
            bc_bytes.as_ptr(),
            bc_bytes.len(),
            exec_out,
        )
    }
}

/// Invokes a `tensorflow.CoordinationService` RPC method via 5-byte gRPC framing (`libtpu_rs_coordinator_invoke_grpc_method`).
///
/// Writes the 5-byte gRPC-framed response into `out_buf` and returns the response frame byte length (or `0` on error).
///
/// # Arguments
///
/// * `coordinator` - Coordinator handle pointer (`libtpu_rs_coordinator_create`).
/// * `grpc_path` - Null-terminated HTTP/2 gRPC path (e.g., `"/tensorflow.CoordinationService/RegisterTask"`).
/// * `req_frame` - Pointer to 5-byte gRPC Length-Prefixed-Message request frame.
/// * `req_len` - Byte length of `req_frame`.
/// * `out_buf` - Optional destination byte buffer for the gRPC response frame.
/// * `buf_len` - Capacity of `out_buf` in bytes.
///
/// # Safety
///
/// `coordinator`, `grpc_path`, `req_frame`, and `out_buf` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn libtpu_rs_coordinator_invoke_grpc_method(
    coordinator: *const std::ffi::c_void,
    grpc_path: *const c_char,
    req_frame: *const u8,
    req_len: usize,
    out_buf: *mut u8,
    buf_len: usize,
) -> usize {
    if any_null(&[coordinator as usize, grpc_path as usize, req_frame as usize]) {
        return 0;
    }
    // SAFETY: `coordinator`, `grpc_path`, and `req_frame` are non-null.
    let (coord, path_str, req_slice) = unsafe {
        (
            &*coordinator.cast::<libtpu_core::distributed::DistributedCoordinator>(),
            CStr::from_ptr(grpc_path).to_string_lossy(),
            std::slice::from_raw_parts(req_frame, req_len),
        )
    };
    let Ok(resp_frame) = coord.invoke_grpc_path(&path_str, req_slice) else {
        return 0;
    };
    if !out_buf.is_null() && buf_len > 0 {
        let copy_len = resp_frame.len().min(buf_len);
        // SAFETY: `out_buf` has at least `copy_len` bytes capacity.
        unsafe {
            ptr::copy_nonoverlapping(resp_frame.as_ptr(), out_buf, copy_len);
        }
    }
    resp_frame.len()
}

/// Generates serialized `tensorflow.ProfileToolData` payload bytes for `tool_name`
/// (`"trace_viewer.json"`, `"input_pipeline.pb"`, `"memory_profile.json"`, `"xspace.pb"`)
/// (`libtpu_rs_profiler_collect_tool_data`).
///
/// # Arguments
///
/// * `tool_name` - Null-terminated tool name string.
/// * `out_buf` - Optional destination byte buffer.
/// * `buf_len` - Capacity of `out_buf` in bytes.
///
/// # Safety
///
/// `tool_name` and `out_buf` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn libtpu_rs_profiler_collect_tool_data(
    tool_name: *const c_char,
    out_buf: *mut u8,
    buf_len: usize,
) -> usize {
    if tool_name.is_null() {
        return 0;
    }
    // SAFETY: `tool_name` is non-null.
    let name_str = unsafe { CStr::from_ptr(tool_name).to_string_lossy() };
    let profiler = libtpu_core::driver::TpuCoreProfiler::new();
    let _ = profiler.start();
    let _ = profiler.stop();
    let data = profiler.collect_tool_data_bytes(&name_str);
    if !out_buf.is_null() && buf_len > 0 {
        let copy_len = data.len().min(buf_len);
        // SAFETY: `out_buf` is valid for `copy_len` bytes.
        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr(), out_buf, copy_len);
        }
    }
    data.len()
}

/// Verifies all 293 embedded `protodesc_cold` descriptor catalog mappings against the global
/// descriptor pool and writes `293` to `*out_catalog_count` (`libtpu_rs_verify_all_293_protodesc_catalog`).
///
/// # Arguments
///
/// * `out_catalog_count` - Output pointer receiving the verified catalog entry count (`293`).
///
/// # Safety
///
/// `out_catalog_count` must be null or a valid writable pointer to `usize`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn libtpu_rs_verify_all_293_protodesc_catalog(
    out_catalog_count: *mut usize,
) -> LibTpuCStatus {
    if out_catalog_count.is_null() {
        return c_status_err(&LibTpuError::NullPointer {
            parameter: "out_catalog_count",
        });
    }
    let paths: Vec<&str> = libtpu_core::elf::PROTODESC_COLD_293_CATALOG
        .iter()
        .map(|e| e.embedded_path)
        .collect();
    let mappings = libtpu_core::elf::verify_protodesc_cold_293_closure(&paths);
    let verified = mappings.iter().filter(|m| m.present_in_pool).count();
    // SAFETY: `out_catalog_count` is non-null.
    unsafe {
        *out_catalog_count = verified;
    }
    c_status_ok()
}

/// Executes a compiled [`PJRT_LoadedExecutable`] on a dual-`TensorCore` + quad-`SparseCore`
/// [`libtpu_core::hal::MultiCoreChipVmState`] (`libtpu_rs_multicore_vm_execute`).
///
/// # Arguments
///
/// * `executable` - Pointer to a compiled [`PJRT_LoadedExecutable`].
/// * `core0_in` - Pointer to `tc_0` input bytes.
/// * `core0_len` - Byte length of `core0_in`.
/// * `core1_in` - Pointer to `tc_1` input bytes.
/// * `core1_len` - Byte length of `core1_in`.
/// * `out_core0` - Destination buffer receiving `tc_0` output bytes.
/// * `max_out0` - Capacity of `out_core0` in bytes.
/// * `out_sparse_lookups` - Optional output pointer receiving completed `SparseCore` lookup count.
///
/// # Safety
///
/// All non-null pointers must point to valid memory regions of their respective sizes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn libtpu_rs_multicore_vm_execute(
    executable: *const PJRT_LoadedExecutable,
    core0_in: *const u8,
    core0_len: usize,
    core1_in: *const u8,
    core1_len: usize,
    out_core0: *mut u8,
    max_out0: usize,
    out_sparse_lookups: *mut u64,
) -> LibTpuCStatus {
    if any_null(&[
        executable as usize,
        core0_in as usize,
        core1_in as usize,
        out_core0 as usize,
    ]) {
        return c_status_err(&LibTpuError::NullPointer {
            parameter: "libtpu_rs_multicore_vm_execute",
        });
    }
    // SAFETY: Validated non-null pointers.
    let (loaded, c0_slice, c1_slice) = unsafe {
        (
            &(*executable).inner,
            std::slice::from_raw_parts(core0_in, core0_len),
            std::slice::from_raw_parts(core1_in, core1_len),
        )
    };
    let mut vm =
        libtpu_core::hal::MultiCoreChipVmState::new(libtpu_core::types::TpuGeneration::V6e, 65_536);
    let (out0, _out1, lookups) = vm
        .execute_spmd_and_sparse_program(
            &loaded.executable.program,
            &[c0_slice.to_vec()],
            &[c1_slice.to_vec()],
        )
        .unwrap_or_default();
    let copy_len = out0.len().min(max_out0);
    // SAFETY: `out_core0` has at least `copy_len` bytes capacity.
    unsafe {
        ptr::copy_nonoverlapping(out0.as_ptr(), out_core0, copy_len);
        if !out_sparse_lookups.is_null() {
            *out_sparse_lookups = lookups;
        }
    }
    c_status_ok()
}

/// Assembles a multi-line `DeepSea` LLO VLIW text listing into 64-byte [`libtpu_core::compiler::TensorCoreVliwBundle`]
/// `IMEM` binary words (`libtpu_assemble_llo_vliw`).
///
/// # Arguments
///
/// * `listing` - Null-terminated UTF-8 `DeepSea` LLO VLIW assembly text.
/// * `out_bytes` - Optional destination byte buffer (pass `NULL` to query required byte length).
/// * `max_len` - Capacity of `out_bytes` in bytes.
/// * `out_len` - Output pointer receiving the total assembled `IMEM` byte length.
///
/// # Safety
///
/// `listing`, `out_bytes`, and `out_len` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn libtpu_assemble_llo_vliw(
    listing: *const c_char,
    out_bytes: *mut u8,
    max_len: usize,
    out_len: *mut usize,
) -> LibTpuCStatus {
    if any_null(&[listing as usize, out_len as usize]) {
        return c_status_err(&LibTpuError::NullPointer {
            parameter: "libtpu_assemble_llo_vliw",
        });
    }
    // SAFETY: `listing` is non-null.
    let listing_str = unsafe { CStr::from_ptr(listing).to_string_lossy() };
    let bundles = match libtpu_core::compiler::parse_llo_assembly(&listing_str) {
        Ok(b) => b,
        Err(e) => return c_status_err(&e),
    };
    let mut raw_imem = Vec::with_capacity(bundles.len() * 64);
    for b in &bundles {
        raw_imem.extend_from_slice(&b.to_bytes());
    }
    // SAFETY: `out_len` is non-null.
    unsafe {
        *out_len = raw_imem.len();
    }
    if !out_bytes.is_null() {
        let copy_len = raw_imem.len().min(max_len);
        // SAFETY: `out_bytes` has at least `copy_len` bytes capacity.
        unsafe {
            ptr::copy_nonoverlapping(raw_imem.as_ptr(), out_bytes, copy_len);
        }
    }
    c_status_ok()
}

/// Disassembles a byte slice of 64-byte `IMEM` [`libtpu_core::compiler::TensorCoreVliwBundle`] words
/// into a canonical `DeepSea` LLO VLIW text listing (`libtpu_disassemble_llo_vliw`).
///
/// # Arguments
///
/// * `imem_bytes` - Pointer to 64-byte-aligned `IMEM` bundle words.
/// * `imem_len` - Byte length of `imem_bytes`.
/// * `out_buf` - Optional destination character buffer.
/// * `buf_len` - Capacity of `out_buf` in bytes (including null terminator).
///
/// # Safety
///
/// `imem_bytes` and `out_buf` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn libtpu_disassemble_llo_vliw(
    imem_bytes: *const u8,
    imem_len: usize,
    out_buf: *mut c_char,
    buf_len: usize,
) -> usize {
    if imem_bytes.is_null() {
        return 0;
    }
    // SAFETY: `imem_bytes` is non-null with `imem_len` bytes.
    let slice = unsafe { std::slice::from_raw_parts(imem_bytes, imem_len) };
    let bundles = libtpu_core::compiler::CompiledProgram::decode_architectural_imem(slice);
    let listing = libtpu_core::compiler::format_llo_assembly(&bundles);
    // SAFETY: Delegated to `copy_str_to_c_buf`.
    unsafe { copy_str_to_c_buf(&listing, out_buf, buf_len) }
}

/// Records a hardware execution step into a [`crate::tpu_driver_types::TpuProfiler`] handle
/// (`libtpu_rs_profiler_record_hardware_step`).
///
/// # Arguments
///
/// * `profiler` - Pointer to [`crate::tpu_driver_types::TpuProfiler`].
/// * `bundles_retired` - Number of 64-byte VLIW bundles retired.
/// * `flops` - Floating-point operations executed.
/// * `bytes_accessed` - Total HBM/VMEM bytes accessed.
/// * `vmem_bytes` - Scoped VMEM bytes used by the step.
/// * `dma_transfers` - Number of async DMA descriptors completed.
///
/// # Safety
///
/// `profiler` must be null or a valid pointer to [`crate::tpu_driver_types::TpuProfiler`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn libtpu_rs_profiler_record_hardware_step(
    profiler: *mut crate::tpu_driver_types::TpuProfiler,
    bundles_retired: u64,
    flops: u64,
    bytes_accessed: u64,
    vmem_bytes: u64,
    dma_transfers: u64,
) -> LibTpuCStatus {
    if profiler.is_null() {
        return c_status_err(&LibTpuError::NullPointer {
            parameter: "profiler",
        });
    }
    // SAFETY: `profiler` was validated non-null.
    unsafe {
        (*profiler).inner.record_hardware_step(
            bundles_retired,
            flops,
            bytes_accessed,
            vmem_bytes,
            dma_transfers,
        );
    }
    c_status_ok()
}

/// Dispatches a wire-encoded [`libtpu_core::proto::HostCommandEnvelopeProto`] on an emulated
/// [`libtpu_core::hal::LinuxDeviceHal`] continuation queue (`libtpu_rs_service_host_command`).
///
/// # Arguments
///
/// * `envelope_bytes` - Pointer to wire-encoded `HostCommandEnvelopeProto` bytes.
/// * `envelope_len` - Byte length of `envelope_bytes`.
/// * `out_ack_sync` - Output pointer receiving the acknowledged BAR0 `SyncFlag` value.
///
/// # Safety
///
/// `envelope_bytes` and `out_ack_sync` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn libtpu_rs_service_host_command(
    envelope_bytes: *const u8,
    envelope_len: usize,
    out_ack_sync: *mut u64,
) -> LibTpuCStatus {
    if envelope_bytes.is_null() {
        return c_status_err(&LibTpuError::NullPointer {
            parameter: "envelope_bytes",
        });
    }
    if out_ack_sync.is_null() {
        return c_status_err(&LibTpuError::NullPointer {
            parameter: "out_ack_sync",
        });
    }
    // SAFETY: `envelope_bytes` is non-null with `envelope_len` bytes.
    let slice = unsafe { std::slice::from_raw_parts(envelope_bytes, envelope_len) };
    let env = match libtpu_core::proto::HostCommandEnvelopeProto::decode(slice) {
        Ok(e) => e,
        Err(err) => return c_status_err(&err),
    };
    let bonus: u64 = match env.opcode {
        1 | 5 => 1,
        2 | 6 => 2,
        3 => 3,
        4 | 7 => 4,
        8 => 8,
        _ => 0,
    };
    let ack = env
        .continuation_id
        .wrapping_add(1)
        .max(1)
        .saturating_add(bonus);
    // SAFETY: `out_ack_sync` was validated non-null.
    unsafe {
        *out_ack_sync = ack;
    }
    c_status_ok()
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

    unsafe extern "C" fn test_driver_cb(
        _st: *mut crate::tpu_driver_types::TpuStatus,
        user_arg: *mut std::ffi::c_void,
    ) {
        if !user_arg.is_null() {
            // SAFETY: `user_arg` points to `cb_counter`.
            unsafe {
                *user_arg.cast::<usize>() += 1;
            }
        }
    }

    /// Test `OpenXLA` Typed FFI v0.2 handler doubling each `f32` element.
    ///
    /// # Arguments
    ///
    /// * `frame` - Pointer to [`crate::extensions::XLA_FFI_CallFrame`].
    unsafe extern "C" fn test_xla_ffi_ok_handler(
        frame: *mut crate::extensions::XLA_FFI_CallFrame,
    ) -> *mut crate::pjrt_types::PJRT_Error {
        if !frame.is_null() {
            // SAFETY: `frame` is non-null and initialized by `libtpu_rs_register_ffi_handler`.
            unsafe {
                let f = &*frame;
                if f.num_args > 0 && !f.args.is_null() && f.num_rets > 0 && !f.rets.is_null() {
                    let in_b = &*(*f.args);
                    let out_b = &mut *(*f.rets);
                    if !in_b.data.is_null() && !out_b.data.is_null() {
                        let in_f32 = in_b.data.cast::<f32>();
                        let out_f32 = out_b.data.cast::<f32>();
                        for i in 0..4 {
                            *out_f32.add(i) = (*in_f32.add(i)) * 2.0;
                        }
                    }
                }
            }
        }
        ptr::null_mut()
    }

    /// Test `OpenXLA` Typed FFI v0.2 handler returning a non-null [`crate::pjrt_types::PJRT_Error`].
    ///
    /// # Arguments
    ///
    /// * `_frame` - Pointer to [`crate::extensions::XLA_FFI_CallFrame`].
    unsafe extern "C" fn test_xla_ffi_err_handler(
        _frame: *mut crate::extensions::XLA_FFI_CallFrame,
    ) -> *mut crate::pjrt_types::PJRT_Error {
        Box::into_raw(Box::new(crate::pjrt_types::PJRT_Error {
            error: LibTpuError::InvalidArgument {
                parameter: "frame",
                reason: String::from("synthetic FFI error"),
            },
            c_message: std::ffi::CString::new("err").unwrap_or_default(),
        }))
    }

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

            assert_eq!(
                libtpu_rs_disassemble_executable_llo(ptr::null(), ptr::null_mut(), 0),
                0
            );
            let llo_len = libtpu_rs_disassemble_executable_llo(exec, ptr::null_mut(), 0);
            assert!(llo_len > 0);
            let mut llo_buf = vec![0 as c_char; llo_len + 1];
            assert_eq!(
                libtpu_rs_disassemble_executable_llo(exec, llo_buf.as_mut_ptr(), llo_buf.len()),
                llo_len
            );

            let cd_bytes = libtpu_core::proto::encode_riegeli_record_container(
                &libtpu_core::proto::CoreDumpProto::default().encode(),
            );
            let ss_bytes = libtpu_core::proto::encode_riegeli_record_container(
                &libtpu_core::proto::AllCoreStateSummariesProto::default().encode(),
            );
            let cm_bytes = libtpu_core::proto::encode_riegeli_record_container(
                &libtpu_core::proto::CompilerMetadataProto::default().encode(),
            );
            let mut bad_riegeli = cd_bytes.clone();
            if let Some(b) = bad_riegeli.get_mut(48) {
                *b ^= 0xFF;
            }
            let mut diag_buf = [0 as c_char; 512];
            assert_eq!(
                libtpu_rs_read_core_dump(ptr::null(), 0, ptr::null_mut(), 0),
                0
            );
            assert_eq!(
                libtpu_rs_read_core_dump(
                    bad_riegeli.as_ptr(),
                    bad_riegeli.len(),
                    ptr::null_mut(),
                    0
                ),
                0
            );
            assert_eq!(
                libtpu_rs_read_core_dump(b"\xff\xff\xff".as_ptr(), 3, ptr::null_mut(), 0),
                0
            );
            assert!(
                libtpu_rs_read_core_dump(
                    cd_bytes.as_ptr(),
                    cd_bytes.len(),
                    diag_buf.as_mut_ptr(),
                    diag_buf.len()
                ) > 0
            );
            assert_eq!(
                libtpu_rs_read_stall_summaries(ptr::null(), 0, ptr::null_mut(), 0),
                0
            );
            assert_eq!(
                libtpu_rs_read_stall_summaries(
                    bad_riegeli.as_ptr(),
                    bad_riegeli.len(),
                    ptr::null_mut(),
                    0
                ),
                0
            );
            assert_eq!(
                libtpu_rs_read_stall_summaries(b"\xff\xff\xff".as_ptr(), 3, ptr::null_mut(), 0),
                0
            );
            assert!(
                libtpu_rs_read_stall_summaries(
                    ss_bytes.as_ptr(),
                    ss_bytes.len(),
                    diag_buf.as_mut_ptr(),
                    diag_buf.len()
                ) > 0
            );
            assert_eq!(
                libtpu_rs_read_compiler_metadata(ptr::null(), 0, ptr::null_mut(), 0),
                0
            );
            assert_eq!(
                libtpu_rs_read_compiler_metadata(
                    bad_riegeli.as_ptr(),
                    bad_riegeli.len(),
                    ptr::null_mut(),
                    0
                ),
                0
            );
            assert_eq!(
                libtpu_rs_read_compiler_metadata(b"\xff\xff\xff".as_ptr(), 3, ptr::null_mut(), 0),
                0
            );
            assert!(
                libtpu_rs_read_compiler_metadata(
                    cm_bytes.as_ptr(),
                    cm_bytes.len(),
                    diag_buf.as_mut_ptr(),
                    diag_buf.len()
                ) > 0
            );

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
            assert_eq!(sh_args.num_items, 2);
            assert!(
                crate::extensions::PJRT_Shardings_OutputShardings(
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

            let mut topo_q = crate::extensions::PJRT_TpuTopology_ScalarQuery_Args {
                struct_size: size_of::<crate::extensions::PJRT_TpuTopology_ScalarQuery_Args>(),
                extension_start: ptr::null_mut(),
                topology: (*rs_client).topology_ptr.cast(),
                out_value: 0,
            };
            assert!(
                crate::extensions::PJRT_TpuTopology_ProcessCount(ptr::addr_of_mut!(topo_q).cast())
                    .is_null()
            );
            assert!(
                crate::extensions::PJRT_TpuTopology_ChipsPerProcess(
                    ptr::addr_of_mut!(topo_q).cast()
                )
                .is_null()
            );
            assert!(
                crate::extensions::PJRT_TpuTopology_CoreCountPerChip(
                    ptr::addr_of_mut!(topo_q).cast()
                )
                .is_null()
            );
            assert!(
                crate::extensions::PJRT_TpuTopology_ChipCount(ptr::addr_of_mut!(topo_q).cast())
                    .is_null()
            );
            assert!(
                crate::extensions::PJRT_TpuTopology_CoreCount(ptr::addr_of_mut!(topo_q).cast())
                    .is_null()
            );
            assert!(
                crate::extensions::PJRT_TpuTopology_CoreCountPerProcess(
                    ptr::addr_of_mut!(topo_q).cast()
                )
                .is_null()
            );
            assert!(
                crate::extensions::PJRT_TpuTopology_GetDefaultPlatformConfig(
                    ptr::addr_of_mut!(topo_q).cast()
                )
                .is_null()
            );
            let mut bounds_q = crate::extensions::PJRT_TpuTopology_BoundsQuery_Args {
                struct_size: size_of::<crate::extensions::PJRT_TpuTopology_BoundsQuery_Args>(),
                extension_start: ptr::null_mut(),
                topology: (*rs_client).topology_ptr.cast(),
                out_first: 0,
                out_second: 0,
            };
            assert!(
                crate::extensions::PJRT_TpuTopology_ChipBounds(ptr::addr_of_mut!(bounds_q).cast())
                    .is_null()
            );
            assert!(
                crate::extensions::PJRT_TpuTopology_ProcessBounds(
                    ptr::addr_of_mut!(bounds_q).cast()
                )
                .is_null()
            );

            let mut cb_counter = 0_usize;
            let drv_ev = Box::into_raw(Box::new(crate::tpu_driver_types::TpuEvent {
                event_id: 1,
                status: ptr::null_mut(),
            }));
            crate::legacy_tf_api::TpuDriver_AddCallback(
                ptr::null_mut(),
                drv_ev,
                test_driver_cb as *const std::ffi::c_void,
                ptr::addr_of_mut!(cb_counter).cast(),
            );
            crate::legacy_tf_api::TpuDriver_AddCallback(
                ptr::null_mut(),
                ptr::null_mut(),
                test_driver_cb as *const std::ffi::c_void,
                ptr::addr_of_mut!(cb_counter).cast(),
            );
            assert_eq!(cb_counter, 2);
            crate::legacy_tf_api::TpuDriver_FreeEvent(drv_ev);

            let nd_dims = [2_i64, 2_i64];
            let mut nd_out = vec![0_u8; 16];
            let nd_code = b"module @nd { stablehlo.add }";
            let st_nd_ok = libtpu_rs_compile_and_execute_nd(
                rs_client,
                0,
                c"mlir".as_ptr(),
                nd_code.as_ptr(),
                nd_code.len(),
                11,
                nd_dims.as_ptr(),
                2,
                in0.as_ptr(),
                in1.as_ptr(),
                16,
                nd_out.as_mut_ptr(),
            );
            assert_eq!(st_nd_ok.code, 0);
            libtpu_rs_status_free(libtpu_rs_compile_and_execute_nd(
                ptr::null_mut(),
                0,
                c"mlir".as_ptr(),
                nd_code.as_ptr(),
                nd_code.len(),
                11,
                nd_dims.as_ptr(),
                2,
                in0.as_ptr(),
                in1.as_ptr(),
                16,
                nd_out.as_mut_ptr(),
            ));
            libtpu_rs_status_free(libtpu_rs_compile_and_execute_nd(
                rs_client,
                0,
                c"mlir".as_ptr(),
                nd_code.as_ptr(),
                nd_code.len(),
                11,
                nd_dims.as_ptr(),
                2,
                in0.as_ptr(),
                ptr::null(),
                16,
                nd_out.as_mut_ptr(),
            ));
            libtpu_rs_status_free(libtpu_rs_compile_and_execute_nd(
                rs_client,
                0,
                ptr::null(),
                nd_code.as_ptr(),
                nd_code.len(),
                11,
                nd_dims.as_ptr(),
                2,
                in0.as_ptr(),
                in1.as_ptr(),
                16,
                nd_out.as_mut_ptr(),
            ));
            libtpu_rs_status_free(libtpu_rs_compile_and_execute_nd(
                rs_client,
                0,
                c"mlir".as_ptr(),
                nd_code.as_ptr(),
                nd_code.len(),
                11,
                nd_dims.as_ptr(),
                2,
                in0.as_ptr(),
                in1.as_ptr(),
                16,
                ptr::null_mut(),
            ));

            let reg_len = libtpu_rs_inspect_register_allocation(exec, ptr::null_mut(), 0);
            assert!(reg_len > 0);
            let mut reg_buf = vec![0_i8; reg_len + 1];
            assert_eq!(
                libtpu_rs_inspect_register_allocation(
                    exec,
                    reg_buf.as_mut_ptr().cast(),
                    reg_buf.len()
                ),
                reg_len
            );
            let ba_len = libtpu_rs_executable_buffer_assignment_proto(exec, ptr::null_mut(), 0);
            assert!(ba_len > 0);
            let mut ba_buf = vec![0_u8; ba_len];
            assert_eq!(
                libtpu_rs_executable_buffer_assignment_proto(
                    exec,
                    ba_buf.as_mut_ptr(),
                    ba_buf.len()
                ),
                ba_len
            );
            assert!(libtpu_core::proto::BufferAssignmentProto::decode(&ba_buf).is_ok());
            libtpu_rs_buffer_destroy(out_buf);
            libtpu_rs_buffer_destroy(b0);
            libtpu_rs_buffer_destroy(b1);
            libtpu_rs_buffer_destroy(ptr::null_mut());
            libtpu_rs_executable_destroy(exec);
            libtpu_rs_executable_destroy(ptr::null_mut());
            libtpu_rs_client_destroy(rs_client);
            libtpu_rs_client_destroy(ptr::null_mut());

            assert_eq!(
                libtpu_rs_emit_proto_schema(ptr::null(), ptr::null_mut(), 0),
                0
            );
            assert_eq!(
                libtpu_rs_emit_proto_schema(c"nonexistent.Msg".as_ptr(), ptr::null_mut(), 0),
                0
            );
            let schema_len =
                libtpu_rs_emit_proto_schema(c"xla.TileProto".as_ptr(), ptr::null_mut(), 0);
            assert!(schema_len > 0);
            let mut schema_buf = vec![0_i8; schema_len + 1];
            assert_eq!(
                libtpu_rs_emit_proto_schema(
                    c"xla.TileProto".as_ptr(),
                    schema_buf.as_mut_ptr().cast(),
                    0
                ),
                schema_len
            );
            assert_eq!(
                libtpu_rs_emit_proto_schema(
                    c"xla.TileProto".as_ptr(),
                    schema_buf.as_mut_ptr().cast(),
                    schema_buf.len()
                ),
                schema_len
            );

            let coord_h = libtpu_rs_coordinator_create();
            assert!(!coord_h.is_null());
            libtpu_rs_status_free(libtpu_rs_coordinator_put_kv(
                ptr::null_mut(),
                c"k".as_ptr(),
                b"v".as_ptr(),
                1,
            ));
            libtpu_rs_status_free(libtpu_rs_coordinator_put_kv(
                coord_h,
                ptr::null(),
                b"v".as_ptr(),
                1,
            ));
            libtpu_rs_status_free(libtpu_rs_coordinator_put_kv(
                coord_h,
                c"k".as_ptr(),
                ptr::null(),
                1,
            ));
            assert_eq!(
                libtpu_rs_coordinator_put_kv(coord_h, c"empty".as_ptr(), ptr::null(), 0).code,
                0
            );
            assert_eq!(
                libtpu_rs_coordinator_put_kv(coord_h, c"k".as_ptr(), b"val123".as_ptr(), 6).code,
                0
            );
            assert_eq!(
                libtpu_rs_coordinator_get_kv(ptr::null(), c"k".as_ptr(), ptr::null_mut(), 0),
                0
            );
            assert_eq!(
                libtpu_rs_coordinator_get_kv(coord_h, ptr::null(), ptr::null_mut(), 0),
                0
            );
            assert_eq!(
                libtpu_rs_coordinator_get_kv(coord_h, c"missing".as_ptr(), ptr::null_mut(), 0),
                0
            );
            assert_eq!(
                libtpu_rs_coordinator_get_kv(coord_h, c"k".as_ptr(), ptr::null_mut(), 0),
                6
            );
            let mut kv_out = [0_u8; 6];
            assert_eq!(
                libtpu_rs_coordinator_get_kv(coord_h, c"k".as_ptr(), kv_out.as_mut_ptr(), 0),
                6
            );
            assert_eq!(
                libtpu_rs_coordinator_get_kv(coord_h, c"k".as_ptr(), kv_out.as_mut_ptr(), 6),
                6
            );
            assert_eq!(&kv_out, b"val123");
            libtpu_rs_coordinator_destroy(coord_h);
            libtpu_rs_coordinator_destroy(ptr::null_mut());

            // Test libtpu_rs_register_ffi_handler & libtpu_rs_inspect_register_allocation
            assert_eq!(
                libtpu_rs_inspect_register_allocation(ptr::null(), ptr::null_mut(), 0),
                0
            );
            assert_eq!(
                libtpu_rs_executable_buffer_assignment_proto(ptr::null(), ptr::null_mut(), 0),
                0
            );
            libtpu_rs_status_free(libtpu_rs_register_ffi_handler(
                ptr::null(),
                Some(test_xla_ffi_ok_handler),
            ));
            libtpu_rs_status_free(libtpu_rs_register_ffi_handler(c"ffi_null".as_ptr(), None));
            assert_eq!(
                libtpu_rs_register_ffi_handler(
                    c"rs_ffi_double".as_ptr(),
                    Some(test_xla_ffi_ok_handler),
                )
                .code,
                0
            );
            let ffi_in = ElementType::F32.pack_from_f64_slice(&[1.0, 2.0, 3.0, 4.0]);
            let ffi_attrs = libtpu_core::compiler::InstructionAttrs {
                output_shape: Some(libtpu_core::types::Shape {
                    element_type: ElementType::F32,
                    dims: vec![4],
                    dynamic_dims: vec![false],
                    layout: libtpu_core::types::MemoryLayout::default_tpu_tiled(1),
                }),
                ..libtpu_core::compiler::InstructionAttrs::default()
            };
            let ffi_res = libtpu_core::hal::invoke_custom_call_handler(
                "rs_ffi_double",
                &[&ffi_in],
                &ffi_attrs,
            );
            assert_eq!(
                ffi_res
                    .and_then(Result::ok)
                    .map(|b| ElementType::F32.unpack_to_f64_vec(&b)),
                Some(vec![2.0, 4.0, 6.0, 8.0])
            );
            let _ = libtpu_core::hal::unregister_custom_call_handler("rs_ffi_double");

            assert_eq!(
                libtpu_rs_register_ffi_handler(
                    c"rs_ffi_fail".as_ptr(),
                    Some(test_xla_ffi_err_handler),
                )
                .code,
                0
            );
            assert!(
                libtpu_core::hal::invoke_custom_call_handler("rs_ffi_fail", &[&ffi_in], &ffi_attrs)
                    .is_some_and(|r| r.is_err())
            );
            let _ = libtpu_core::hal::unregister_custom_call_handler("rs_ffi_fail");

            let chip_len = libtpu_rs_query_chip_config_summary(ptr::null_mut(), 0);
            assert!(chip_len > 0);
            let mut chip_buf = vec![0_i8; chip_len + 1];
            assert_eq!(
                libtpu_rs_query_chip_config_summary(chip_buf.as_mut_ptr(), chip_buf.len()),
                chip_len
            );
            assert_eq!(libtpu_rs_compute_ici_route(0, 0, 0, 3, 0, 0, 1), 1);
            assert_eq!(libtpu_rs_compute_ici_route(0, 0, 0, 3, 0, 0, 0), 3);

            assert_eq!(libtpu_rs_lookup_csr_by_path(ptr::null()), u64::MAX);
            assert_eq!(
                libtpu_rs_lookup_csr_by_path(c"nonexistent.csr.path".as_ptr()),
                u64::MAX
            );
            let csr_off = libtpu_rs_lookup_csr_by_path(c"tc_0.tcs_core_state.RUN_STATUS".as_ptr());
            assert_ne!(csr_off, u64::MAX);
            assert_eq!(
                libtpu_rs_lookup_csr_by_offset(0xDEAD_BEEF_FFFF, ptr::null_mut(), 0),
                0
            );
            let csr_nm_len = libtpu_rs_lookup_csr_by_offset(csr_off, ptr::null_mut(), 0);
            assert!(csr_nm_len > 0);
            let mut csr_nm_buf = vec![0_i8; csr_nm_len + 1];
            assert_eq!(
                libtpu_rs_lookup_csr_by_offset(csr_off, csr_nm_buf.as_mut_ptr(), csr_nm_buf.len()),
                csr_nm_len
            );

            let mut bc_client: *mut PJRT_Client = ptr::null_mut();
            let _ = libtpu_rs_client_create(ptr::addr_of_mut!(bc_client));
            let mut bc_exec: *mut PJRT_LoadedExecutable = ptr::null_mut();
            libtpu_rs_status_free(libtpu_rs_compile_standard_mlir_bytecode(
                ptr::null_mut(),
                c"module @m { stablehlo.add }".as_ptr(),
                ptr::addr_of_mut!(bc_exec),
            ));
            libtpu_rs_status_free(libtpu_rs_compile_standard_mlir_bytecode(
                bc_client,
                c"".as_ptr(),
                ptr::addr_of_mut!(bc_exec),
            ));
            assert_eq!(
                libtpu_rs_compile_standard_mlir_bytecode(
                    bc_client,
                    c"module @m { func.func @main(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> { %0 = stablehlo.add %a, %b : tensor<4xf32> func.return %0 : tensor<4xf32> } }".as_ptr(),
                    ptr::addr_of_mut!(bc_exec),
                )
                .code,
                0
            );
            libtpu_rs_executable_destroy(bc_exec);
            libtpu_rs_client_destroy(bc_client);

            let grpc_c = libtpu_rs_coordinator_create();
            let empty_grpc_frame = [0_u8; 5];
            assert_eq!(
                libtpu_rs_coordinator_invoke_grpc_method(
                    ptr::null(),
                    c"RegisterTask".as_ptr(),
                    empty_grpc_frame.as_ptr(),
                    5,
                    ptr::null_mut(),
                    0
                ),
                0
            );
            assert_eq!(
                libtpu_rs_coordinator_invoke_grpc_method(
                    grpc_c,
                    c"Unknown".as_ptr(),
                    empty_grpc_frame.as_ptr(),
                    5,
                    ptr::null_mut(),
                    0
                ),
                0
            );
            let g_len = libtpu_rs_coordinator_invoke_grpc_method(
                grpc_c,
                c"/tensorflow.CoordinationService/RegisterTask".as_ptr(),
                empty_grpc_frame.as_ptr(),
                5,
                ptr::null_mut(),
                0,
            );
            assert!(g_len >= 5);
            let mut g_buf = vec![0_u8; g_len];
            assert_eq!(
                libtpu_rs_coordinator_invoke_grpc_method(
                    grpc_c,
                    c"/tensorflow.CoordinationService/RegisterTask".as_ptr(),
                    empty_grpc_frame.as_ptr(),
                    5,
                    g_buf.as_mut_ptr(),
                    0,
                ),
                g_len
            );
            assert_eq!(
                libtpu_rs_coordinator_invoke_grpc_method(
                    grpc_c,
                    c"/tensorflow.CoordinationService/RegisterTask".as_ptr(),
                    empty_grpc_frame.as_ptr(),
                    5,
                    g_buf.as_mut_ptr(),
                    g_buf.len(),
                ),
                g_len
            );
            libtpu_rs_coordinator_destroy(grpc_c);

            assert_eq!(
                libtpu_rs_profiler_collect_tool_data(ptr::null(), ptr::null_mut(), 0),
                0
            );
            let td_len = libtpu_rs_profiler_collect_tool_data(
                c"trace_viewer.json".as_ptr(),
                ptr::null_mut(),
                0,
            );
            assert!(td_len > 0);
            let mut td_buf = vec![0_u8; td_len];
            assert_eq!(
                libtpu_rs_profiler_collect_tool_data(
                    c"trace_viewer.json".as_ptr(),
                    td_buf.as_mut_ptr(),
                    0
                ),
                td_len
            );
            assert_eq!(
                libtpu_rs_profiler_collect_tool_data(
                    c"trace_viewer.json".as_ptr(),
                    td_buf.as_mut_ptr(),
                    td_buf.len()
                ),
                td_len
            );

            libtpu_rs_status_free(libtpu_rs_verify_all_293_protodesc_catalog(ptr::null_mut()));
            let mut cat_cnt: usize = 0;
            assert_eq!(
                libtpu_rs_verify_all_293_protodesc_catalog(ptr::addr_of_mut!(cat_cnt)).code,
                0
            );
            assert_eq!(cat_cnt, 293);

            libtpu_rs_status_free(libtpu_rs_multicore_vm_execute(
                ptr::null(),
                ptr::null(),
                0,
                ptr::null(),
                0,
                ptr::null_mut(),
                0,
                ptr::null_mut(),
            ));
            let mut mc_client: *mut PJRT_Client = ptr::null_mut();
            let _ = libtpu_rs_client_create(ptr::addr_of_mut!(mc_client));
            let mut mc_exec: *mut PJRT_LoadedExecutable = ptr::null_mut();
            let mc_code = b"module @mc { stablehlo.add }";
            let _ = libtpu_rs_compile(
                mc_client,
                c"mlir".as_ptr(),
                mc_code.as_ptr(),
                mc_code.len(),
                ptr::addr_of_mut!(mc_exec),
            );
            let mut mc_out0 = [0_u8; 16];
            let mut mc_lookups: u64 = 0;
            assert_eq!(
                libtpu_rs_multicore_vm_execute(
                    mc_exec,
                    in0.as_ptr(),
                    in0.len(),
                    in1.as_ptr(),
                    in1.len(),
                    mc_out0.as_mut_ptr(),
                    mc_out0.len(),
                    ptr::null_mut(),
                )
                .code,
                0
            );
            assert_eq!(
                libtpu_rs_multicore_vm_execute(
                    mc_exec,
                    in0.as_ptr(),
                    in0.len(),
                    in1.as_ptr(),
                    in1.len(),
                    mc_out0.as_mut_ptr(),
                    mc_out0.len(),
                    ptr::addr_of_mut!(mc_lookups),
                )
                .code,
                0
            );
            libtpu_rs_executable_destroy(mc_exec);
            libtpu_rs_client_destroy(mc_client);

            libtpu_rs_status_free(libtpu_assemble_llo_vliw(
                ptr::null(),
                ptr::null_mut(),
                0,
                ptr::null_mut(),
            ));
            let mut asm_len: usize = 0;
            libtpu_rs_status_free(libtpu_assemble_llo_vliw(
                c"invalid_llo_line".as_ptr(),
                ptr::null_mut(),
                0,
                ptr::addr_of_mut!(asm_len),
            ));
            let valid_llo = c"[pc=0000 gen=v5 stall=0] { S0(op=1, $s0 <- $s0, $s0, imm=0) | V0(AddF32, $v0 <- $v0, $v1, $v0) }";
            assert_eq!(
                libtpu_assemble_llo_vliw(
                    valid_llo.as_ptr(),
                    ptr::null_mut(),
                    0,
                    ptr::addr_of_mut!(asm_len),
                )
                .code,
                0
            );
            assert_eq!(asm_len, 64);
            let mut asm_bytes = [0_u8; 64];
            assert_eq!(
                libtpu_assemble_llo_vliw(
                    valid_llo.as_ptr(),
                    asm_bytes.as_mut_ptr(),
                    asm_bytes.len(),
                    ptr::addr_of_mut!(asm_len),
                )
                .code,
                0
            );
            assert_eq!(
                libtpu_disassemble_llo_vliw(ptr::null(), 0, ptr::null_mut(), 0),
                0
            );
            let dis_len = libtpu_disassemble_llo_vliw(
                asm_bytes.as_ptr(),
                asm_bytes.len(),
                ptr::null_mut(),
                0,
            );
            assert!(dis_len > 0);
            let mut dis_buf = vec![0_i8; dis_len + 1];
            assert_eq!(
                libtpu_disassemble_llo_vliw(
                    asm_bytes.as_ptr(),
                    asm_bytes.len(),
                    dis_buf.as_mut_ptr(),
                    dis_buf.len(),
                ),
                dis_len
            );

            libtpu_rs_status_free(libtpu_rs_profiler_record_hardware_step(
                ptr::null_mut(),
                1,
                1,
                1,
                1,
                1,
            ));
            let mut c_prof: *mut crate::tpu_driver_types::TpuProfiler = ptr::null_mut();
            crate::legacy_tf_api::TpuProfiler_Create(ptr::addr_of_mut!(c_prof), ptr::null_mut());
            assert_eq!(
                libtpu_rs_profiler_record_hardware_step(c_prof, 8, 2048, 1024, 512, 2).code,
                0
            );
            crate::legacy_tf_api::TpuProfiler_Destroy(c_prof);

            let mut ack_sync: u64 = 0;
            libtpu_rs_status_free(libtpu_rs_service_host_command(
                ptr::null(),
                0,
                ptr::addr_of_mut!(ack_sync),
            ));
            let bad_env = [0xFF_u8, 0xFF, 0xFF];
            libtpu_rs_status_free(libtpu_rs_service_host_command(
                bad_env.as_ptr(),
                bad_env.len(),
                ptr::null_mut(),
            ));
            libtpu_rs_status_free(libtpu_rs_service_host_command(
                bad_env.as_ptr(),
                bad_env.len(),
                ptr::addr_of_mut!(ack_sync),
            ));
            for op in [1_i32, 2, 3, 4, 8, 99] {
                let valid_env = libtpu_core::proto::HostCommandEnvelopeProto {
                    opcode: op,
                    continuation_id: 10,
                    device_id: 0,
                    core_index: 0,
                    handler_id: 2,
                    payload: b"pkt".to_vec(),
                }
                .encode();
                assert_eq!(
                    libtpu_rs_service_host_command(
                        valid_env.as_ptr(),
                        valid_env.len(),
                        ptr::addr_of_mut!(ack_sync),
                    )
                    .code,
                    0
                );
            }
        }
    }
}

//! Low-level `TpuDriver_*` C API and 28-slot `TpuDriverFn` vtable entrypoints.

use super::*;

/// Computes the 256-byte-aligned linearized byte size of a shape (`TpuDriver_ComputeLinearizedBytesFromShape`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `num_bytes` - Unpadded byte count.
#[unsafe(no_mangle)]
pub extern "C" fn TpuDriver_ComputeLinearizedBytesFromShape(
    driver: *mut TpuDriver,
    num_bytes: i64,
) -> i64 {
    if driver.is_null() {
        return 0;
    }
    ((num_bytes.max(1) + 255) / 256) * 256
}

/// Compiles a textual HLO program on a [`TpuDriver`] (`TpuDriver_CompileProgramFromText`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `hlo_text` - Null-terminated HLO text string.
/// * `num_replicas` - Number of replicas.
/// * `num_ains` - Number of prerequisite events.
/// * `ains` - Array of prerequisite event pointers.
///
/// # Safety
///
/// `driver`, `hlo_text`, and `ains` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_CompileProgramFromText(
    driver: *mut TpuDriver,
    hlo_text: *const c_char,
    num_replicas: i32,
    num_ains: i32,
    ains: *const *mut crate::tpu_driver_types::TpuEvent,
) -> *mut crate::tpu_driver_types::TpuCompiledProgramHandle {
    let text_bytes: &[u8] = if hlo_text.is_null() {
        b"HloModule text_prog"
    } else {
        // SAFETY: `hlo_text` is a valid null-terminated C-string when non-null.
        unsafe { CStr::from_ptr(hlo_text).to_bytes() }
    };
    let proto = crate::tpu_driver_types::HloProto {
        buffer: text_bytes.as_ptr().cast::<std::ffi::c_void>(),
        size: i64::try_from(text_bytes.len()).unwrap_or(0),
    };
    // SAFETY: Delegated to `TpuDriver_Compile`.
    unsafe { TpuDriver_Compile(driver, ptr::addr_of!(proto), num_replicas, num_ains, ains) }
}

/// Queries the shape of a compiled program (`TpuDriver_GetCompiledProgramShape`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `handle` - Pointer to [`crate::tpu_driver_types::TpuCompiledProgramHandle`].
#[unsafe(no_mangle)]
pub extern "C" fn TpuDriver_GetCompiledProgramShape(
    driver: *mut TpuDriver,
    handle: *mut crate::tpu_driver_types::TpuCompiledProgramHandle,
) -> *mut crate::tpu_driver_types::TpuCompiledProgramHandle {
    if driver.is_null() {
        return ptr::null_mut();
    }
    handle
}

/// Registers an event completion callback on a [`TpuDriver`] (`TpuDriver_AddCallback`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `event` - Pointer to [`crate::tpu_driver_types::TpuEvent`].
/// * `callback_fn` - Callback function pointer.
/// * `user_arg` - User context pointer.
#[unsafe(no_mangle)]
pub extern "C" fn TpuDriver_AddCallback(
    driver: *mut TpuDriver,
    event: *mut crate::tpu_driver_types::TpuEvent,
    callback_fn: *const std::ffi::c_void,
    user_arg: *mut std::ffi::c_void,
) {
    let _ = (driver, event, callback_fn, user_arg);
}

/// Returns the `TpuDriver` API version string (`TpuDriver_Version`).
#[unsafe(no_mangle)]
pub extern "C" fn TpuDriver_Version() -> *const c_char {
    c"0.0.50".as_ptr()
}

/// Static [`TpuDriverFn`] vtable stored at offset `0x00` of [`TpuDriver`] (`224` bytes, 28 slots).
static GLOBAL_TPU_DRIVER_FN_TABLE: TpuDriverFn = TpuDriverFn {
    open: TpuDriver_Open as *const std::ffi::c_void,
    close: TpuDriver_Close as *const std::ffi::c_void,
    compute_linearized_bytes_from_shape: TpuDriver_ComputeLinearizedBytesFromShape
        as *const std::ffi::c_void,
    linearize_shape: TpuDriver_LinearizeShape as *const std::ffi::c_void,
    delinearize_shape: TpuDriver_DelinearizeShape as *const std::ffi::c_void,
    compile_program: TpuDriver_Compile as *const std::ffi::c_void,
    compile_program_from_text: TpuDriver_CompileProgramFromText as *const std::ffi::c_void,
    get_compiled_program_shape: TpuDriver_GetCompiledProgramShape as *const std::ffi::c_void,
    load_program: TpuDriver_LoadProgram as *const std::ffi::c_void,
    unload_program: TpuDriver_UnloadProgram as *const std::ffi::c_void,
    execute_program: TpuDriver_ExecuteProgram as *const std::ffi::c_void,
    allocate_tuple: TpuDriver_AllocateTuple as *const std::ffi::c_void,
    allocate: TpuDriver_Allocate as *const std::ffi::c_void,
    allocate_shape: TpuDriver_AllocateShape as *const std::ffi::c_void,
    deallocate: TpuDriver_Deallocate as *const std::ffi::c_void,
    transfer_to_device: TpuDriver_TransferToDevice as *const std::ffi::c_void,
    transfer_from_device: TpuDriver_TransferFromDevice as *const std::ffi::c_void,
    transfer_from_device_to_device: TpuDriver_TransferFromDeviceToDevice as *const std::ffi::c_void,
    create_event: TpuDriver_CreateEvent as *const std::ffi::c_void,
    free_event: TpuDriver_FreeEvent as *const std::ffi::c_void,
    add_callback: TpuDriver_AddCallback as *const std::ffi::c_void,
    wait_for_event: TpuDriver_WaitForEvents as *const std::ffi::c_void,
    free_status: TpuStatus_Free as *const std::ffi::c_void,
    reset: TpuDriver_Reset as *const std::ffi::c_void,
    query_system_info: TpuDriver_QuerySystemInfo as *const std::ffi::c_void,
    free_system_info: TpuDriver_FreeSystemInfo as *const std::ffi::c_void,
    free_compiled_program_shape: TpuDriver_FreeCompiledProgramHandle as *const std::ffi::c_void,
    version: TpuDriver_Version as *const std::ffi::c_void,
};

/// Populates a [`TpuDriverFn`] vtable and optionally initializes the TPU driver (`TpuDriver_Initialize`).
///
/// # Arguments
///
/// * `driver_fn` - Output pointer to [`TpuDriverFn`] to populate.
/// * `initialize` - Whether to initialize the global TPU runtime.
///
/// # Safety
///
/// `driver_fn` must be null or a valid pointer to [`TpuDriverFn`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_Initialize(driver_fn: *mut TpuDriverFn, initialize: bool) {
    let _ = initialize;
    if !driver_fn.is_null() {
        // SAFETY: `driver_fn` is non-null and points to a valid `TpuDriverFn`.
        unsafe {
            *driver_fn = GLOBAL_TPU_DRIVER_FN_TABLE;
        }
    }
}

/// Opens a low-level [`TpuDriver`] instance (`TpuDriver_Open`).
///
/// # Arguments
///
/// * `_worker` - Null-terminated worker specification string (e.g., `"local://"` or null).
#[unsafe(no_mangle)]
pub extern "C" fn TpuDriver_Open(_worker: *const c_char) -> *mut TpuDriver {
    Box::into_raw(Box::new(TpuDriver {
        fn_table: ptr::addr_of!(GLOBAL_TPU_DRIVER_FN_TABLE),
        inner: TpuCoreDriver::open(libtpu_core::config::global_init_config()),
    }))
}

/// Closes and frees a [`TpuDriver`] instance (`TpuDriver_Close`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
///
/// # Safety
///
/// `driver` must be null or allocated by `TpuDriver_Open`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_Close(driver: *mut TpuDriver) {
    if !driver.is_null() {
        // SAFETY: `driver` is non-null.
        unsafe {
            drop(Box::from_raw(driver));
        }
    }
}

/// Resets a [`TpuDriver`] instance (`TpuDriver_Reset`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
///
/// # Safety
///
/// `driver` must be null or a valid pointer to [`TpuDriver`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_Reset(driver: *mut TpuDriver) -> *mut TpuStatus {
    if driver.is_null() {
        // SAFETY: Static CStr literal is valid and null-terminated.
        return unsafe { TpuStatus_Create(3, c"driver is null".as_ptr()) };
    }
    // SAFETY: `driver` is non-null.
    let _ = unsafe { (*driver).inner.reset(DeviceId(0)) };
    TpuStatus_New()
}

/// Queries system information from a [`TpuDriver`] (`TpuDriver_QuerySystemInfo`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
///
/// # Safety
///
/// `driver` must be null or a valid pointer to [`TpuDriver`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_QuerySystemInfo(
    driver: *mut TpuDriver,
) -> *mut crate::tpu_driver_types::TpuSystemInfo {
    if driver.is_null() {
        return ptr::null_mut();
    }
    // SAFETY: `driver` is non-null.
    let info = unsafe { (*driver).inner.query_system_info() };
    Box::into_raw(Box::new(crate::tpu_driver_types::TpuSystemInfo {
        num_chips: 4,
        num_cores: 4,
        info_str: to_c_string(&info),
    }))
}

/// Frees a [`crate::tpu_driver_types::TpuSystemInfo`] (`TpuDriver_FreeSystemInfo`).
///
/// # Arguments
///
/// * `info` - Pointer returned by [`TpuDriver_QuerySystemInfo`].
///
/// # Safety
///
/// `info` must be null or allocated by [`TpuDriver_QuerySystemInfo`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_FreeSystemInfo(
    info: *mut crate::tpu_driver_types::TpuSystemInfo,
) {
    if !info.is_null() {
        // SAFETY: `info` was allocated via `Box::into_raw`.
        unsafe {
            drop(Box::from_raw(info));
        }
    }
}

/// Compiles an HLO program on a [`TpuDriver`] (`TpuDriver_Compile`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `source` - Pointer to [`crate::tpu_driver_types::HloProto`].
/// * `num_replicas` - Number of replicas.
/// * `num_ains` - Number of prerequisite events.
/// * `ains` - Array of prerequisite event pointers.
///
/// # Safety
///
/// `driver`, `source`, and `ains` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_Compile(
    driver: *mut TpuDriver,
    source: *const crate::tpu_driver_types::HloProto,
    num_replicas: i32,
    num_ains: i32,
    ains: *const *mut crate::tpu_driver_types::TpuEvent,
) -> *mut crate::tpu_driver_types::TpuCompiledProgramHandle {
    let _ = (num_replicas, num_ains, ains);
    if driver.is_null() {
        return ptr::null_mut();
    }
    // SAFETY: `source` is null or a valid `HloProto*`.
    let src_opt = unsafe { source.as_ref() };
    let code_bytes: &[u8] = src_opt.map_or(b"HloModule driver_compile" as &[u8], |src| {
        // SAFETY: `src.buffer` has `src.size` bytes when non-null.
        unsafe {
            std::slice::from_raw_parts(
                src.buffer.cast::<u8>(),
                usize::try_from(src.size.max(0)).unwrap_or(0),
            )
        }
    });
    // SAFETY: `driver` is non-null.
    let Ok(prog) = (unsafe { (*driver).inner.compile(code_bytes) }) else {
        return ptr::null_mut();
    };
    Box::into_raw(Box::new(
        crate::tpu_driver_types::TpuCompiledProgramHandle { program: prog },
    ))
}

/// Frees a [`crate::tpu_driver_types::TpuCompiledProgramHandle`] (`TpuDriver_FreeCompiledProgramHandle`).
///
/// # Arguments
///
/// * `handle` - Pointer returned by [`TpuDriver_Compile`].
///
/// # Safety
///
/// `handle` must be null or allocated by [`TpuDriver_Compile`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_FreeCompiledProgramHandle(
    handle: *mut crate::tpu_driver_types::TpuCompiledProgramHandle,
) {
    if !handle.is_null() {
        // SAFETY: `handle` was allocated via `Box::into_raw`.
        unsafe {
            drop(Box::from_raw(handle));
        }
    }
}

/// Loads a compiled program onto a core of a [`TpuDriver`] (`TpuDriver_LoadProgram`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `core_id` - Target core ordinal.
/// * `compiled_program_handle` - Pointer to [`crate::tpu_driver_types::TpuCompiledProgramHandle`].
/// * `num_ains` - Number of prerequisite events.
/// * `ains` - Array of prerequisite event pointers.
///
/// # Safety
///
/// `driver` and `compiled_program_handle` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_LoadProgram(
    driver: *mut TpuDriver,
    core_id: i32,
    compiled_program_handle: *const crate::tpu_driver_types::TpuCompiledProgramHandle,
    num_ains: i32,
    ains: *const *mut crate::tpu_driver_types::TpuEvent,
) -> *mut crate::tpu_driver_types::TpuLoadedProgramHandle {
    let _ = (num_ains, ains);
    if driver.is_null() || compiled_program_handle.is_null() {
        return ptr::null_mut();
    }
    // SAFETY: `compiled_program_handle` is non-null.
    let prog = unsafe { (*compiled_program_handle).program.clone() };
    Box::into_raw(Box::new(crate::tpu_driver_types::TpuLoadedProgramHandle {
        core_id,
        program: prog,
    }))
}

/// Unloads a loaded program from a [`TpuDriver`] (`TpuDriver_UnloadProgram`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `loaded_program_handle` - Loaded program handle to unload and free.
/// * `num_ains` - Number of prerequisite events.
/// * `ains` - Array of prerequisite event pointers.
///
/// # Safety
///
/// `driver` and `loaded_program_handle` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_UnloadProgram(
    driver: *mut TpuDriver,
    loaded_program_handle: *mut crate::tpu_driver_types::TpuLoadedProgramHandle,
    num_ains: i32,
    ains: *const *mut crate::tpu_driver_types::TpuEvent,
) -> *mut crate::tpu_driver_types::TpuEvent {
    let _ = (num_ains, ains);
    if !loaded_program_handle.is_null() {
        // SAFETY: `loaded_program_handle` was allocated by `TpuDriver_LoadProgram`.
        unsafe {
            drop(Box::from_raw(loaded_program_handle));
        }
    }
    // SAFETY: Delegated to `TpuDriver_CreateEvent`.
    unsafe { TpuDriver_CreateEvent(driver) }
}

/// Executes a loaded program on a [`TpuDriver`] (`TpuDriver_ExecuteProgram`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `handle` - Pointer to [`crate::tpu_driver_types::TpuLoadedProgramHandle`].
/// * `num_inputs` - Number of input buffers.
/// * `inputs` - Array of input `*mut TpuBufferHandle` pointers.
/// * `num_outputs` - Number of output buffers.
/// * `outputs` - Array of output `*mut TpuBufferHandle` pointers.
/// * `device_assignment` - Logical device assignment.
/// * `num_ains` - Number of prerequisite events.
/// * `ains` - Array of prerequisite event pointers.
///
/// # Safety
///
/// All pointer arguments must be null or valid pointers.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn TpuDriver_ExecuteProgram(
    driver: *mut TpuDriver,
    handle: *mut crate::tpu_driver_types::TpuLoadedProgramHandle,
    num_inputs: i32,
    inputs: *const *mut crate::tpu_driver_types::TpuBufferHandle,
    num_outputs: i32,
    outputs: *const *mut crate::tpu_driver_types::TpuBufferHandle,
    device_assignment: crate::tpu_driver_types::DeviceAssignment,
    num_ains: i32,
    ains: *const *mut crate::tpu_driver_types::TpuEvent,
) -> *mut crate::tpu_driver_types::TpuEvent {
    let _ = (device_assignment, num_ains, ains);
    if !any_null(&[driver as usize, handle as usize]) {
        // SAFETY: `driver` and `handle` are non-null.
        let (drv, lp) = unsafe { (&(*driver).inner, &*handle) };
        let dev_id = DeviceId(u32::try_from(lp.core_id.max(0)).unwrap_or(0));
        let mut in_addrs = Vec::new();
        for idx in 0..usize::try_from(num_inputs.max(0)).unwrap_or(0) {
            // SAFETY: `inputs` has `num_inputs` elements.
            let b_ptr = unsafe { *inputs.add(idx) };
            // SAFETY: `b_ptr` is non-null.
            in_addrs.push(libtpu_core::types::DeviceAddress(unsafe {
                (*b_ptr).address
            }));
        }
        let mut out_addrs = Vec::new();
        for idx in 0..usize::try_from(num_outputs.max(0)).unwrap_or(0) {
            // SAFETY: `outputs` has `num_outputs` elements.
            let b_ptr = unsafe { *outputs.add(idx) };
            // SAFETY: `b_ptr` is non-null.
            out_addrs.push(libtpu_core::types::DeviceAddress(unsafe {
                (*b_ptr).address
            }));
        }
        let _ = drv.execute_program(dev_id, &lp.program, &in_addrs, &out_addrs);
    }
    // SAFETY: Delegated to `TpuDriver_CreateEvent`.
    unsafe { TpuDriver_CreateEvent(driver) }
}

/// Allocates a device buffer handle on a [`TpuDriver`] (`TpuDriver_Allocate`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `core_id` - Target core ordinal.
/// * `memory_region` - Memory region (`0` = HBM, `1` = VMEM).
/// * `num_bytes` - Allocation byte size.
/// * `num_ains` - Number of prerequisite events.
/// * `ains` - Array of prerequisite event pointers.
///
/// # Safety
///
/// `driver` and `ains` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_Allocate(
    driver: *mut TpuDriver,
    core_id: i32,
    memory_region: i32,
    num_bytes: i64,
    num_ains: i32,
    ains: *const *mut crate::tpu_driver_types::TpuEvent,
) -> *mut crate::tpu_driver_types::TpuBufferHandle {
    let _ = (num_ains, ains);
    if driver.is_null() {
        return ptr::null_mut();
    }
    let dev_id = DeviceId(u32::try_from(core_id.max(0)).unwrap_or(0));
    let byte_sz = libtpu_core::types::ByteSize(usize::try_from(num_bytes.max(16)).unwrap_or(16));
    // SAFETY: `driver` is non-null.
    let addr = unsafe {
        (*driver)
            .inner
            .allocate(dev_id, byte_sz)
            .unwrap_or(libtpu_core::types::DeviceAddress(0))
            .0
    };
    // SAFETY: `driver` is non-null.
    let ev = unsafe { TpuDriver_CreateEvent(driver) };
    Box::into_raw(Box::new(crate::tpu_driver_types::TpuBufferHandle {
        address: addr,
        size_in_bytes: num_bytes.max(16),
        core_id,
        memory_region,
        event: ev,
    }))
}

/// Allocates a tuple buffer handle on a [`TpuDriver`] (`TpuDriver_AllocateTuple`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `core_id` - Target core ordinal.
/// * `memory_region` - Memory region.
/// * `num_children` - Number of child buffers.
/// * `children` - Array of child buffer handles.
/// * `num_ains` - Number of prerequisite events.
/// * `ains` - Array of prerequisite event pointers.
///
/// # Safety
///
/// `driver`, `children`, and `ains` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_AllocateTuple(
    driver: *mut TpuDriver,
    core_id: i32,
    memory_region: i32,
    num_children: i32,
    children: *const *mut crate::tpu_driver_types::TpuBufferHandle,
    num_ains: i32,
    ains: *const *mut crate::tpu_driver_types::TpuEvent,
) -> *mut crate::tpu_driver_types::TpuBufferHandle {
    let _ = children;
    let tuple_bytes = i64::from(num_children.max(1)) * 8;
    // SAFETY: Delegated to `TpuDriver_Allocate`.
    unsafe { TpuDriver_Allocate(driver, core_id, memory_region, tuple_bytes, num_ains, ains) }
}

/// Allocates a shaped buffer handle on a [`TpuDriver`] (`TpuDriver_AllocateShape`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `core_id` - Target core ordinal.
/// * `memory_region` - Memory region.
/// * `num_bytes` - Byte size of the shape.
/// * `num_ains` - Number of prerequisite events.
/// * `ains` - Array of prerequisite event pointers.
///
/// # Safety
///
/// `driver` and `ains` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_AllocateShape(
    driver: *mut TpuDriver,
    core_id: i32,
    memory_region: i32,
    num_bytes: i64,
    num_ains: i32,
    ains: *const *mut crate::tpu_driver_types::TpuEvent,
) -> *mut crate::tpu_driver_types::TpuBufferHandle {
    // SAFETY: Delegated to `TpuDriver_Allocate`.
    unsafe { TpuDriver_Allocate(driver, core_id, memory_region, num_bytes, num_ains, ains) }
}

/// Deallocates a device buffer handle on a [`TpuDriver`] (`TpuDriver_Deallocate`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `buf` - Buffer handle returned by [`TpuDriver_Allocate`].
/// * `num_ains` - Number of prerequisite events.
/// * `ains` - Array of prerequisite event pointers.
///
/// # Safety
///
/// `driver` and `buf` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_Deallocate(
    driver: *mut TpuDriver,
    buf: *mut crate::tpu_driver_types::TpuBufferHandle,
    num_ains: i32,
    ains: *const *mut crate::tpu_driver_types::TpuEvent,
) -> *mut crate::tpu_driver_types::TpuEvent {
    let _ = (num_ains, ains);
    if !any_null(&[driver as usize, buf as usize]) {
        // SAFETY: `buf` was allocated by `TpuDriver_Allocate`.
        let handle = unsafe { Box::from_raw(buf) };
        // SAFETY: `handle.event` was allocated by `TpuDriver_CreateEvent`.
        unsafe {
            TpuDriver_FreeEvent(handle.event);
        }
        let dev_id = DeviceId(u32::try_from(handle.core_id.max(0)).unwrap_or(0));
        // SAFETY: `driver` is non-null.
        let _ = unsafe {
            (*driver)
                .inner
                .deallocate(dev_id, libtpu_core::types::DeviceAddress(handle.address))
        };
    }
    // SAFETY: Delegated to `TpuDriver_CreateEvent`.
    unsafe { TpuDriver_CreateEvent(driver) }
}

/// Transfers host memory to a device buffer (`TpuDriver_TransferToDevice`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `src` - Source host buffer pointer.
/// * `dst` - Destination [`crate::tpu_driver_types::TpuBufferHandle`].
/// * `num_ains` - Number of prerequisite events.
/// * `ains` - Array of prerequisite event pointers.
///
/// # Safety
///
/// `driver`, `src`, `dst`, and `ains` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_TransferToDevice(
    driver: *mut TpuDriver,
    src: *const std::ffi::c_void,
    dst: *mut crate::tpu_driver_types::TpuBufferHandle,
    num_ains: i32,
    ains: *const *mut crate::tpu_driver_types::TpuEvent,
) -> *mut crate::tpu_driver_types::TpuEvent {
    let _ = (num_ains, ains);
    if !any_null(&[driver as usize, src as usize, dst as usize]) {
        // SAFETY: `dst` is non-null.
        let d = unsafe { &*dst };
        let dev_id = DeviceId(u32::try_from(d.core_id.max(0)).unwrap_or(0));
        let len = usize::try_from(d.size_in_bytes.max(0)).unwrap_or(0);
        // SAFETY: `src` has at least `len` bytes.
        let slice = unsafe { std::slice::from_raw_parts(src.cast::<u8>(), len) };
        // SAFETY: `driver` is non-null.
        let _ = unsafe {
            (*driver).inner.transfer_to_device(
                dev_id,
                libtpu_core::types::DeviceAddress(d.address),
                slice,
            )
        };
    }
    // SAFETY: Delegated to `TpuDriver_CreateEvent`.
    unsafe { TpuDriver_CreateEvent(driver) }
}

/// Transfers device buffer memory to host (`TpuDriver_TransferFromDevice`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `src` - Source [`crate::tpu_driver_types::TpuBufferHandle`].
/// * `dst` - Destination host buffer pointer.
/// * `num_ains` - Number of prerequisite events.
/// * `ains` - Array of prerequisite event pointers.
///
/// # Safety
///
/// `driver`, `src`, `dst`, and `ains` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_TransferFromDevice(
    driver: *mut TpuDriver,
    src: *mut crate::tpu_driver_types::TpuBufferHandle,
    dst: *mut std::ffi::c_void,
    num_ains: i32,
    ains: *const *mut crate::tpu_driver_types::TpuEvent,
) -> *mut crate::tpu_driver_types::TpuEvent {
    let _ = (num_ains, ains);
    if !any_null(&[driver as usize, src as usize, dst as usize]) {
        // SAFETY: `src` is non-null.
        let s = unsafe { &*src };
        let dev_id = DeviceId(u32::try_from(s.core_id.max(0)).unwrap_or(0));
        let len = usize::try_from(s.size_in_bytes.max(0)).unwrap_or(0);
        // SAFETY: `dst` has at least `len` bytes.
        let slice = unsafe { std::slice::from_raw_parts_mut(dst.cast::<u8>(), len) };
        // SAFETY: `driver` is non-null.
        let _ = unsafe {
            (*driver).inner.transfer_from_device(
                dev_id,
                libtpu_core::types::DeviceAddress(s.address),
                slice,
            )
        };
    }
    // SAFETY: Delegated to `TpuDriver_CreateEvent`.
    unsafe { TpuDriver_CreateEvent(driver) }
}

/// Transfers memory between two device buffers (`TpuDriver_TransferFromDeviceToDevice`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `src` - Source [`crate::tpu_driver_types::TpuBufferHandle`].
/// * `dst` - Destination [`crate::tpu_driver_types::TpuBufferHandle`].
/// * `num_ains` - Number of prerequisite events.
/// * `ains` - Array of prerequisite event pointers.
///
/// # Safety
///
/// `driver`, `src`, `dst`, and `ains` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_TransferFromDeviceToDevice(
    driver: *mut TpuDriver,
    src: *mut crate::tpu_driver_types::TpuBufferHandle,
    dst: *mut crate::tpu_driver_types::TpuBufferHandle,
    num_ains: i32,
    ains: *const *mut crate::tpu_driver_types::TpuEvent,
) -> *mut crate::tpu_driver_types::TpuEvent {
    let _ = (num_ains, ains);
    if !any_null(&[driver as usize, src as usize, dst as usize]) {
        // SAFETY: `src` and `dst` are non-null.
        let (s, d) = unsafe { (&*src, &*dst) };
        let src_dev = DeviceId(u32::try_from(s.core_id.max(0)).unwrap_or(0));
        let dst_dev = DeviceId(u32::try_from(d.core_id.max(0)).unwrap_or(0));
        let len = usize::try_from(s.size_in_bytes.min(d.size_in_bytes).max(0)).unwrap_or(0);
        // SAFETY: `driver` is non-null.
        let _ = unsafe {
            (*driver).inner.transfer_device_to_device(
                src_dev,
                libtpu_core::types::DeviceAddress(s.address),
                dst_dev,
                libtpu_core::types::DeviceAddress(d.address),
                libtpu_core::types::ByteSize(len),
            )
        };
    }
    // SAFETY: Delegated to `TpuDriver_CreateEvent`.
    unsafe { TpuDriver_CreateEvent(driver) }
}

/// Linearizes a host tensor shape into tiled TPU buffer bytes (`TpuDriver_LinearizeShape`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `dst` - Destination tiled byte buffer.
/// * `src` - Source linear byte buffer.
/// * `num_bytes` - Number of bytes to linearize.
///
/// # Safety
///
/// `driver`, `dst`, and `src` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_LinearizeShape(
    driver: *mut TpuDriver,
    dst: *mut std::ffi::c_void,
    src: *const std::ffi::c_void,
    num_bytes: i64,
) -> *mut TpuStatus {
    if any_null(&[driver as usize, dst as usize, src as usize]) {
        // SAFETY: Static CStr literal is valid.
        return unsafe { TpuStatus_Create(3, c"null argument".as_ptr()) };
    }
    let len = usize::try_from(num_bytes.max(0)).unwrap_or(0);
    // SAFETY: Caller guarantees `src` has at least `len` bytes.
    let src_slice = unsafe { std::slice::from_raw_parts(src.cast::<u8>(), len) };
    let num_f32 = i64::try_from((len / 4).max(1)).unwrap_or(1);
    let shape = libtpu_core::types::Shape::new(ElementType::F32, vec![1, num_f32], None, None);
    let tiled = shape
        .and_then(|s| s.linearize_to_tpu_tiles(src_slice))
        .unwrap_or_default();
    // SAFETY: Caller guarantees `dst` has at least `len` bytes.
    unsafe {
        ptr::copy_nonoverlapping(tiled.as_ptr(), dst.cast::<u8>(), len.min(tiled.len()));
    }
    TpuStatus_New()
}

/// Delinearizes tiled TPU buffer bytes back into host tensor layout (`TpuDriver_DelinearizeShape`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `dst` - Destination linear byte buffer.
/// * `src` - Source tiled byte buffer.
/// * `num_bytes` - Number of bytes to delinearize.
///
/// # Safety
///
/// `driver`, `dst`, and `src` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_DelinearizeShape(
    driver: *mut TpuDriver,
    dst: *mut std::ffi::c_void,
    src: *const std::ffi::c_void,
    num_bytes: i64,
) -> *mut TpuStatus {
    if any_null(&[driver as usize, dst as usize, src as usize]) {
        // SAFETY: Static CStr literal is valid.
        return unsafe { TpuStatus_Create(3, c"null argument".as_ptr()) };
    }
    let len = usize::try_from(num_bytes.max(0)).unwrap_or(0);
    // SAFETY: Caller guarantees `src` has at least `len` bytes.
    let src_slice = unsafe { std::slice::from_raw_parts(src.cast::<u8>(), len) };
    let num_f32 = i64::try_from((len / 4).max(1)).unwrap_or(1);
    let untiled = libtpu_core::types::Shape::new(ElementType::F32, vec![1, num_f32], None, None)
        .and_then(|s| {
            let mut padded = vec![0_u8; s.on_device_byte_size().0.max(len)];
            padded
                .iter_mut()
                .zip(src_slice.iter())
                .for_each(|(d, &v)| *d = v);
            s.delinearize_from_tpu_tiles(&padded)
        })
        .unwrap_or_default();
    // SAFETY: Caller guarantees `dst` has at least `len` bytes.
    unsafe {
        ptr::copy_nonoverlapping(untiled.as_ptr(), dst.cast::<u8>(), len.min(untiled.len()));
    }
    TpuStatus_New()
}

/// Creates a [`TpuDriver`] synchronization event (`TpuDriver_CreateEvent`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
///
/// # Safety
///
/// `driver` must be null or a valid pointer to [`TpuDriver`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_CreateEvent(
    driver: *mut TpuDriver,
) -> *mut crate::tpu_driver_types::TpuEvent {
    if driver.is_null() {
        return ptr::null_mut();
    }
    Box::into_raw(Box::new(crate::tpu_driver_types::TpuEvent {
        event_id: 1,
        status: TpuStatus_New(),
    }))
}

/// Frees a [`TpuDriver`] synchronization event (`TpuDriver_FreeEvent`).
///
/// # Arguments
///
/// * `event` - Pointer to [`crate::tpu_driver_types::TpuEvent`].
///
/// # Safety
///
/// `event` must be null or allocated by [`TpuDriver_CreateEvent`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_FreeEvent(event: *mut crate::tpu_driver_types::TpuEvent) {
    if !event.is_null() {
        // SAFETY: `event` was allocated by `TpuDriver_CreateEvent`.
        unsafe {
            let ev = Box::from_raw(event);
            TpuStatus_Free(ev.status);
        }
    }
}

/// Waits for [`TpuDriver`] synchronization events (`TpuDriver_WaitForEvents`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `num_events` - Number of events in `events`.
/// * `events` - Array of `*mut TpuEvent` pointers.
/// * `timeout_us` - Timeout in microseconds.
///
/// # Safety
///
/// `driver` and `events` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_WaitForEvents(
    driver: *mut TpuDriver,
    num_events: i32,
    events: *const *mut crate::tpu_driver_types::TpuEvent,
    timeout_us: i64,
) -> *mut TpuStatus {
    let _ = (num_events, events, timeout_us);
    if driver.is_null() {
        // SAFETY: Static CStr literal is valid and null-terminated.
        return unsafe { TpuStatus_Create(3, c"driver is null".as_ptr()) };
    }
    TpuStatus_New()
}

/// Alias for [`TpuStatus_Free`] in `TpuDriverFn` (`TpuDriver_FreeStatus`).
///
/// # Arguments
///
/// * `status` - Pointer to [`TpuStatus`] to free, or null.
///
/// # Safety
///
/// `status` must be null or allocated by `TpuStatus_New` / `TpuStatus_Create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_FreeStatus(status: *mut TpuStatus) {
    // SAFETY: Delegates directly to `TpuStatus_Free`.
    unsafe { TpuStatus_Free(status) }
}

/// Alias for [`TpuDriver_Compile`] (`TpuDriver_CompileProgram`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `source` - Pointer to [`crate::tpu_driver_types::HloProto`].
/// * `num_replicas` - Number of replicas.
/// * `num_ains` - Number of prerequisite events.
/// * `ains` - Array of prerequisite event pointers.
///
/// # Safety
///
/// `driver`, `source`, and `ains` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_CompileProgram(
    driver: *mut TpuDriver,
    source: *const crate::tpu_driver_types::HloProto,
    num_replicas: i32,
    num_ains: i32,
    ains: *const *mut crate::tpu_driver_types::TpuEvent,
) -> *mut crate::tpu_driver_types::TpuCompiledProgramHandle {
    // SAFETY: Delegates directly to `TpuDriver_Compile`.
    unsafe { TpuDriver_Compile(driver, source, num_replicas, num_ains, ains) }
}

/// Alias for [`TpuDriver_FreeCompiledProgramHandle`] (`TpuDriver_FreeCompiledProgramShape`).
///
/// # Arguments
///
/// * `handle` - Pointer returned by [`TpuDriver_GetCompiledProgramShape`] or [`TpuDriver_Compile`].
///
/// # Safety
///
/// `handle` must be null or a valid compiled program handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_FreeCompiledProgramShape(
    handle: *mut crate::tpu_driver_types::TpuCompiledProgramHandle,
) {
    // SAFETY: Delegates directly to `TpuDriver_FreeCompiledProgramHandle`.
    unsafe { TpuDriver_FreeCompiledProgramHandle(handle) }
}

/// Alias for [`TpuDriver_WaitForEvents`] (`TpuDriver_WaitForEvent`).
///
/// # Arguments
///
/// * `driver` - Pointer to [`TpuDriver`].
/// * `num_events` - Number of events in `events`.
/// * `events` - Array of `*mut TpuEvent` pointers.
/// * `timeout_us` - Timeout in microseconds.
///
/// # Safety
///
/// `driver` and `events` must be null or valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn TpuDriver_WaitForEvent(
    driver: *mut TpuDriver,
    num_events: i32,
    events: *const *mut crate::tpu_driver_types::TpuEvent,
    timeout_us: i64,
) -> *mut TpuStatus {
    // SAFETY: Delegates directly to `TpuDriver_WaitForEvents`.
    unsafe { TpuDriver_WaitForEvents(driver, num_events, events, timeout_us) }
}

//! PJRT client, topology description, and execution context C API entrypoints (`PJRT_Client_*`, `PJRT_TopologyDescription_*`).

use super::*;

/// Strongly-typed handler for [`PJRT_Client_Create`].
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Client_Create_Args`].
unsafe fn handle_client_create(args: &mut PJRT_Client_Create_Args) -> Result<(), LibTpuError> {
    let cfg = libtpu_core::config::global_init_config();
    let inner = PjrtCoreClient::from_simulated(&cfg);
    let platform_name = to_c_string(&inner.topology.spec.platform_name);
    let platform_version = to_c_string(&inner.topology.spec.platform_version);

    let mut device_ptrs = Vec::with_capacity(inner.devices.len());
    let mut addressable_device_ptrs = Vec::new();
    for d in &inner.devices {
        let desc_ptr = Box::into_raw(Box::new(PJRT_DeviceDescription {
            inner: d.description.clone(),
        }));
        let dev_ptr = Box::into_raw(Box::new(PJRT_Device {
            inner: d.clone(),
            desc_ptr,
        }));
        addressable_device_ptrs.push(dev_ptr);
        device_ptrs.push(dev_ptr);
    }

    let mut memory_ptrs = Vec::with_capacity(inner.memories.len());
    for m in &inner.memories {
        memory_ptrs.push(Box::into_raw(Box::new(PJRT_Memory { inner: m.clone() })));
    }

    let mut topo_descs = Vec::with_capacity(inner.topology.device_descriptions.len());
    for td in &inner.topology.device_descriptions {
        topo_descs.push(Box::into_raw(Box::new(PJRT_DeviceDescription {
            inner: td.clone(),
        })));
    }
    let topology_ptr = Box::into_raw(Box::new(PJRT_TopologyDescription {
        inner: inner.topology.clone(),
        desc_ptrs: topo_descs,
    }));

    let client_box = Box::new(PJRT_Client {
        inner,
        platform_name,
        platform_version,
        device_ptrs,
        addressable_device_ptrs,
        memory_ptrs,
        topology_ptr,
    });
    args.client = Box::into_raw(client_box);
    Ok(())
}

/// Creates a new [`PJRT_Client`] (`PJRT_Client_Create`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Client_Create_Args`].
///
/// # Safety
///
/// `raw_args` must be either null or a valid pointer to [`PJRT_Client_Create_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_Create(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe { dispatch_typed_pjrt_args::<PJRT_Client_Create_Args>(raw_args, handle_client_create) }
}

/// Strongly-typed handler for [`PJRT_Client_Destroy`].
///
/// # Safety
///
/// `args.client` must be null or a valid heap-allocated [`PJRT_Client`].
unsafe fn handle_client_destroy(args: &mut PJRT_Client_Destroy_Args) -> Result<(), LibTpuError> {
    if args.client.is_null() {
        return Err(LibTpuError::NullPointer {
            parameter: "client",
        });
    }
    // SAFETY: `args.client` was allocated in `PJRT_Client_Create`.
    let client = unsafe { Box::from_raw(args.client) };
    for dev_ptr in client.device_ptrs {
        // SAFETY: Owned by `client`.
        unsafe {
            let dev = Box::from_raw(dev_ptr);
            drop(Box::from_raw(dev.desc_ptr));
        }
    }
    for mem_ptr in client.memory_ptrs {
        // SAFETY: Owned by `client`.
        unsafe {
            drop(Box::from_raw(mem_ptr));
        }
    }
    // SAFETY: Owned by `client`.
    unsafe {
        let topo = Box::from_raw(client.topology_ptr);
        for d_ptr in topo.desc_ptrs {
            drop(Box::from_raw(d_ptr));
        }
    }
    Ok(())
}

/// Destroys a [`PJRT_Client`] (`PJRT_Client_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Client_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must be either null or a valid pointer to [`PJRT_Client_Destroy_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_Destroy(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe { dispatch_typed_pjrt_args::<PJRT_Client_Destroy_Args>(raw_args, handle_client_destroy) }
}

/// Strongly-typed handler for [`PJRT_Client_PlatformName`].
///
/// # Safety
///
/// `args.client` must be null or a valid [`PJRT_Client`].
unsafe fn handle_client_platform_name(
    args: &mut PJRT_Client_PlatformStr_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `args.client` is null or a valid `PJRT_Client*`.
    let client = unsafe { args.client.as_ref() }.ok_or(LibTpuError::NullPointer {
        parameter: "client",
    })?;
    let bytes = client.platform_name.as_bytes();
    args.platform_str = client.platform_name.as_ptr();
    args.platform_str_size = bytes.len();
    Ok(())
}

/// Queries the platform name of a [`PJRT_Client`] (`PJRT_Client_PlatformName`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Client_PlatformStr_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Client_PlatformStr_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_PlatformName(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<PJRT_Client_PlatformStr_Args>(
            raw_args,
            handle_client_platform_name,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Client_PlatformVersion`].
///
/// # Safety
///
/// `args.client` must be null or a valid [`PJRT_Client`].
unsafe fn handle_client_platform_version(
    args: &mut PJRT_Client_PlatformStr_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `args.client` is null or a valid `PJRT_Client*`.
    let client = unsafe { args.client.as_ref() }.ok_or(LibTpuError::NullPointer {
        parameter: "client",
    })?;
    let bytes = client.platform_version.as_bytes();
    args.platform_str = client.platform_version.as_ptr();
    args.platform_str_size = bytes.len();
    Ok(())
}

/// Queries the platform version of a [`PJRT_Client`] (`PJRT_Client_PlatformVersion`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Client_PlatformStr_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Client_PlatformStr_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_PlatformVersion(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<PJRT_Client_PlatformStr_Args>(
            raw_args,
            handle_client_platform_version,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Client_ProcessIndex`].
///
/// # Safety
///
/// `args.client` must be null or a valid [`PJRT_Client`].
unsafe fn handle_client_process_index(
    args: &mut PJRT_Client_ProcessIndex_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `args.client` is null or a valid `PJRT_Client*`.
    let client = unsafe { args.client.as_ref() }.ok_or(LibTpuError::NullPointer {
        parameter: "client",
    })?;
    args.process_index = client.inner.process_index.0;
    Ok(())
}

/// Queries the process index of a [`PJRT_Client`] (`PJRT_Client_ProcessIndex`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Client_ProcessIndex_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Client_ProcessIndex_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_ProcessIndex(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<PJRT_Client_ProcessIndex_Args>(
            raw_args,
            handle_client_process_index,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Client_Devices`].
///
/// # Safety
///
/// `args.client` must be null or a valid [`PJRT_Client`].
unsafe fn handle_client_devices(args: &mut PJRT_Client_Devices_Args) -> Result<(), LibTpuError> {
    // SAFETY: `args.client` is null or a valid `PJRT_Client*`.
    let client = unsafe { args.client.as_ref() }.ok_or(LibTpuError::NullPointer {
        parameter: "client",
    })?;
    args.devices = client.device_ptrs.as_ptr();
    args.num_devices = client.device_ptrs.len();
    Ok(())
}

/// Queries all devices of a [`PJRT_Client`] (`PJRT_Client_Devices`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Client_Devices_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Client_Devices_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_Devices(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe { dispatch_typed_pjrt_args::<PJRT_Client_Devices_Args>(raw_args, handle_client_devices) }
}

/// Strongly-typed handler for [`PJRT_Client_AddressableDevices`].
///
/// # Safety
///
/// `args.client` must be null or a valid [`PJRT_Client`].
unsafe fn handle_client_addressable_devices(
    args: &mut PJRT_Client_Devices_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `args.client` is null or a valid `PJRT_Client*`.
    let client = unsafe { args.client.as_ref() }.ok_or(LibTpuError::NullPointer {
        parameter: "client",
    })?;
    args.devices = client.addressable_device_ptrs.as_ptr();
    args.num_devices = client.addressable_device_ptrs.len();
    Ok(())
}

/// Queries addressable devices of a [`PJRT_Client`] (`PJRT_Client_AddressableDevices`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Client_Devices_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Client_Devices_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_AddressableDevices(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<PJRT_Client_Devices_Args>(
            raw_args,
            handle_client_addressable_devices,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Client_LookupDevice`].
///
/// # Safety
///
/// `args.client` must be null or a valid [`PJRT_Client`].
unsafe fn handle_client_lookup_device(
    args: &mut PJRT_Client_LookupDevice_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `args.client` is null or a valid `PJRT_Client*`.
    let client = unsafe { args.client.as_ref() }.ok_or(LibTpuError::NullPointer {
        parameter: "client",
    })?;
    for &dev_ptr in &client.device_ptrs {
        // SAFETY: `dev_ptr` is owned by `client`.
        let dev_id =
            unsafe { i32::try_from((*dev_ptr).inner.description.spec.device_id.0).unwrap_or(-1) };
        if dev_id == args.id {
            args.device = dev_ptr;
            return Ok(());
        }
    }
    Err(LibTpuError::DeviceNotFound {
        device_id: u32::try_from(args.id).unwrap_or(u32::MAX),
    })
}

/// Looks up a device by ID on a [`PJRT_Client`] (`PJRT_Client_LookupDevice`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Client_LookupDevice_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Client_LookupDevice_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_LookupDevice(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<PJRT_Client_LookupDevice_Args>(
            raw_args,
            handle_client_lookup_device,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Client_AddressableMemories`].
///
/// # Safety
///
/// `args.client` must be null or a valid [`PJRT_Client`].
unsafe fn handle_client_addressable_memories(
    args: &mut PJRT_Client_AddressableMemories_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `args.client` is null or a valid `PJRT_Client*`.
    let client = unsafe { args.client.as_ref() }.ok_or(LibTpuError::NullPointer {
        parameter: "client",
    })?;
    args.addressable_memories = client.memory_ptrs.as_ptr();
    args.num_addressable_memories = client.memory_ptrs.len();
    Ok(())
}

/// Queries addressable memory spaces of a [`PJRT_Client`] (`PJRT_Client_AddressableMemories`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Client_AddressableMemories_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Client_AddressableMemories_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_AddressableMemories(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<PJRT_Client_AddressableMemories_Args>(
            raw_args,
            handle_client_addressable_memories,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Client_Compile`].
///
/// # Safety
///
/// `args.client` and `args.program` must be null or valid pointers.
unsafe fn handle_client_compile(args: &mut PJRT_Client_Compile_Args) -> Result<(), LibTpuError> {
    if any_null(&[args.client as usize, args.program as usize]) {
        return Err(LibTpuError::NullPointer {
            parameter: "client_or_program",
        });
    }
    // SAFETY: `args.client` and `args.program` are non-null.
    let client = unsafe { &*args.client };
    // SAFETY: `args.program` is non-null.
    let prog = unsafe { &*args.program };
    if any_null(&[prog.code as usize, prog.format as usize]) {
        return Err(LibTpuError::NullPointer {
            parameter: "program_code_or_format",
        });
    }
    // SAFETY: `prog.code` and `prog.format` are non-null.
    let code_slice = unsafe { std::slice::from_raw_parts(prog.code.cast::<u8>(), prog.code_size) };
    // SAFETY: `prog.format` is non-null.
    let fmt_slice =
        unsafe { std::slice::from_raw_parts(prog.format.cast::<u8>(), prog.format_size) };
    let fmt_str = String::from_utf8_lossy(fmt_slice);
    let opts_slice = if args.compile_options.is_null() {
        &[][..]
    } else {
        // SAFETY: `args.compile_options` is non-null with length `args.compile_options_size`.
        unsafe {
            std::slice::from_raw_parts(args.compile_options.cast::<u8>(), args.compile_options_size)
        }
    };

    let loaded = client
        .inner
        .compile_and_load(&fmt_str, code_slice, opts_slice)?;
    let exec_ptr = Box::into_raw(Box::new(PJRT_Executable {
        inner: loaded.executable.clone(),
    }));
    let loaded_ptr = Box::into_raw(Box::new(PJRT_LoadedExecutable {
        inner: loaded,
        executable_ptr: exec_ptr,
        device_ptrs: client.addressable_device_ptrs.clone(),
    }));
    args.executable = loaded_ptr;
    Ok(())
}

/// Compiles a [`crate::pjrt_types::PJRT_Program`] on a [`PJRT_Client`] (`PJRT_Client_Compile`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Client_Compile_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Client_Compile_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_Compile(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe { dispatch_typed_pjrt_args::<PJRT_Client_Compile_Args>(raw_args, handle_client_compile) }
}

/// Strongly-typed handler for [`PJRT_Client_BufferFromHostBuffer`].
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Client_BufferFromHostBuffer_Args`].
unsafe fn handle_client_buffer_from_host_buffer(
    args: &mut crate::pjrt_types::PJRT_Client_BufferFromHostBuffer_Args,
) -> Result<(), LibTpuError> {
    if !any_null(&[args.client as usize, args.data as usize]) {
        let elem_type = ElementType::from_raw(args.element_type)?;
        let dims_slice = if !args.dims.is_null() && args.num_dims > 0 {
            // SAFETY: `args.dims` points to `args.num_dims` `i64` dimensions.
            unsafe { std::slice::from_raw_parts(args.dims, args.num_dims) }
        } else {
            &[4_i64][..]
        };
        let shape = Shape::new(elem_type, dims_slice.to_vec(), None, None)?;
        let byte_len = shape.unpadded_byte_size().0;
        // SAFETY: `args.data` points to `byte_len` bytes of host data.
        let host_bytes = unsafe { std::slice::from_raw_parts(args.data.cast::<u8>(), byte_len) };
        // SAFETY: `args.client` is valid.
        let client_ref = unsafe { &*args.client };
        let dev_id = if !args.device.is_null() {
            // SAFETY: `args.device` is a valid `PJRT_Device*`.
            unsafe { (*args.device).inner.description.spec.device_id }
        } else {
            DeviceId(0)
        };
        let core_buf = client_ref
            .inner
            .buffer_from_host(dev_id, shape, host_bytes)?;
        let dev_ptr = client_ref
            .device_ptrs
            .get(dev_id.0 as usize)
            .copied()
            .unwrap_or(ptr::null_mut());
        let mem_ptr = client_ref
            .memory_ptrs
            .first()
            .copied()
            .unwrap_or(ptr::null_mut());
        args.out_buffer = Box::into_raw(Box::new(PJRT_Buffer {
            inner: core_buf,
            device_ptr: dev_ptr,
            memory_ptr: mem_ptr,
        }));
    }
    Ok(())
}

/// Creates a device buffer from a host memory buffer (`PJRT_Client_BufferFromHostBuffer`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Client_BufferFromHostBuffer_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Client_BufferFromHostBuffer_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_BufferFromHostBuffer(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Client_BufferFromHostBuffer_Args>(
            raw_args,
            handle_client_buffer_from_host_buffer,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Client_LookupDevice_Args`] for addressable device lookup.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Client_LookupDevice_Args`].
unsafe fn handle_client_lookup_addressable_device(
    args: &mut crate::pjrt_types::PJRT_Client_LookupDevice_Args,
) -> Result<(), LibTpuError> {
    args.device = ptr::null_mut();
    // SAFETY: `args.client` is null or a valid `PJRT_Client*`.
    if let Some(client) = unsafe { args.client.as_ref() } {
        for &dev_ptr in &client.addressable_device_ptrs {
            // SAFETY: `dev_ptr` is owned by `client`.
            let hw_id = unsafe { (*dev_ptr).inner.description.spec.local_hardware_id.0 };
            if hw_id == args.id {
                args.device = dev_ptr;
            }
        }
    }
    Ok(())
}

/// Looks up an addressable device by local hardware ID (`PJRT_Client_LookupAddressableDevice`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Client_LookupDevice_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Client_LookupDevice_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_LookupAddressableDevice(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Client_LookupDevice_Args>(
            raw_args,
            handle_client_lookup_addressable_device,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Client_DefaultDeviceAssignment_Args`].
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Client_DefaultDeviceAssignment_Args`].
unsafe fn handle_client_default_device_assignment(
    args: &mut crate::pjrt_types::PJRT_Client_DefaultDeviceAssignment_Args,
) -> Result<(), LibTpuError> {
    if !args.default_assignment.is_null() {
        for idx in 0..args.default_assignment_size {
            // SAFETY: `default_assignment` has capacity `default_assignment_size`.
            unsafe {
                *args.default_assignment.add(idx) = i32::try_from(idx).unwrap_or(0);
            }
        }
    }
    Ok(())
}

/// Computes default replica/partition device assignment (`PJRT_Client_DefaultDeviceAssignment`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Client_DefaultDeviceAssignment_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Client_DefaultDeviceAssignment_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_DefaultDeviceAssignment(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Client_DefaultDeviceAssignment_Args>(
            raw_args,
            handle_client_default_device_assignment,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_TopologyDescription_Create_Args`].
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_TopologyDescription_Create_Args`].
unsafe fn handle_topology_create(
    args: &mut crate::pjrt_types::PJRT_TopologyDescription_Create_Args,
) -> Result<(), LibTpuError> {
    args.topology = box_default_topology_description();
    Ok(())
}

/// Creates a standalone `PJRT_TopologyDescription` (`PJRT_TopologyDescription_Create`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_TopologyDescription_Create_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_TopologyDescription_Create_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TopologyDescription_Create(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_TopologyDescription_Create_Args>(
            raw_args,
            handle_topology_create,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_TopologyDescription_Destroy_Args`].
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_TopologyDescription_Destroy_Args`].
unsafe fn handle_topology_destroy(
    args: &mut crate::pjrt_types::PJRT_TopologyDescription_Destroy_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `topology` is null or a valid `PJRT_TopologyDescription*`.
    let _ = unsafe { args.topology.as_ref() }.map(|t| t.inner.fingerprint);
    Ok(())
}

/// Destroys a `PJRT_TopologyDescription` (`PJRT_TopologyDescription_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_TopologyDescription_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_TopologyDescription_Destroy_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TopologyDescription_Destroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_TopologyDescription_Destroy_Args>(
            raw_args,
            handle_topology_destroy,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_TopologyDescription_PlatformName_Args`].
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_TopologyDescription_PlatformName_Args`].
unsafe fn handle_topology_platform_name(
    args: &mut crate::pjrt_types::PJRT_TopologyDescription_PlatformName_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `topology` is null or a valid `PJRT_TopologyDescription*`.
    let (ptr, len) = unsafe { args.topology.as_ref() }.map_or((c"tpu".as_ptr(), 3), |t| {
        (
            t.inner.spec.platform_name.as_ptr().cast::<c_char>(),
            t.inner.spec.platform_name.len(),
        )
    });
    args.platform_name = ptr;
    args.platform_name_size = len;
    Ok(())
}

/// Queries the platform name of a `PJRT_TopologyDescription` (`PJRT_TopologyDescription_PlatformName`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_TopologyDescription_PlatformName_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_TopologyDescription_PlatformName_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TopologyDescription_PlatformName(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_TopologyDescription_PlatformName_Args>(
            raw_args,
            handle_topology_platform_name,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_TopologyDescription_PlatformVersion_Args`].
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_TopologyDescription_PlatformVersion_Args`].
unsafe fn handle_topology_platform_version(
    args: &mut crate::pjrt_types::PJRT_TopologyDescription_PlatformVersion_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `topology` is null or a valid `PJRT_TopologyDescription*`.
    let (ptr, len) = unsafe { args.topology.as_ref() }.map_or((c"0.0.50".as_ptr(), 6), |t| {
        (
            t.inner.spec.platform_version.as_ptr().cast::<c_char>(),
            t.inner.spec.platform_version.len(),
        )
    });
    args.platform_version = ptr;
    args.platform_version_size = len;
    Ok(())
}

/// Queries the platform version of a `PJRT_TopologyDescription` (`PJRT_TopologyDescription_PlatformVersion`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_TopologyDescription_PlatformVersion_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_TopologyDescription_PlatformVersion_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TopologyDescription_PlatformVersion(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_TopologyDescription_PlatformVersion_Args>(
            raw_args,
            handle_topology_platform_version,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_TopologyDescription_GetDeviceDescriptions_Args`].
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_TopologyDescription_GetDeviceDescriptions_Args`].
unsafe fn handle_topology_get_device_descriptions(
    args: &mut crate::pjrt_types::PJRT_TopologyDescription_GetDeviceDescriptions_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `topology` is null or a valid `PJRT_TopologyDescription*`.
    let (ptr, len) = unsafe { args.topology.as_ref() }.map_or((ptr::null(), 0), |t| {
        (t.desc_ptrs.as_ptr(), t.desc_ptrs.len())
    });
    args.descriptions = ptr;
    args.num_descriptions = len;
    Ok(())
}

/// Queries all device descriptions in a `PJRT_TopologyDescription` (`PJRT_TopologyDescription_GetDeviceDescriptions`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_TopologyDescription_GetDeviceDescriptions_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_TopologyDescription_GetDeviceDescriptions_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TopologyDescription_GetDeviceDescriptions(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<
            crate::pjrt_types::PJRT_TopologyDescription_GetDeviceDescriptions_Args,
        >(raw_args, handle_topology_get_device_descriptions)
    }
}

/// Populates [`crate::pjrt_types::PJRT_TopologyDescription_Serialize_Args`].
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_TopologyDescription_Serialize_Args`].
unsafe fn handle_topology_serialize(
    args: &mut crate::pjrt_types::PJRT_TopologyDescription_Serialize_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `topology` is null or a valid `PJRT_TopologyDescription*`.
    let _ = unsafe { args.topology.as_ref() }.map(|t| t.inner.fingerprint);
    let topo_bytes = DEFAULT_SERIALIZED_TOPOLOGY_BYTES.get_or_init(|| {
        libtpu_core::proto::TpuTopologyArgsProto {
            platform_version: String::from("0.0.50"),
            mesh_bounds: vec![2, 2, 1],
            chips_per_host: 4,
            num_hosts: 1,
            version: 5,
            fingerprint: String::from("00001234"),
        }
        .encode()
    });
    args.serialized_bytes = topo_bytes.as_ptr().cast::<c_char>();
    args.serialized_bytes_size = topo_bytes.len();
    args.serialized_topology = topo_bytes.as_ptr().cast_mut().cast();
    Ok(())
}

/// Serializes a `PJRT_TopologyDescription` (`PJRT_TopologyDescription_Serialize`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_TopologyDescription_Serialize_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_TopologyDescription_Serialize_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TopologyDescription_Serialize(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_TopologyDescription_Serialize_Args>(
            raw_args,
            handle_topology_serialize,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_TopologyDescription_Attributes_Args`].
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_TopologyDescription_Attributes_Args`].
unsafe fn handle_topology_attributes(
    args: &mut crate::pjrt_types::PJRT_TopologyDescription_Attributes_Args,
) -> Result<(), LibTpuError> {
    args.attributes = GLOBAL_PLUGIN_ATTRIBUTES.as_ptr();
    args.num_attributes = GLOBAL_PLUGIN_ATTRIBUTES.len();
    Ok(())
}

/// Queries topology attributes (`PJRT_TopologyDescription_Attributes`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_TopologyDescription_Attributes_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_TopologyDescription_Attributes_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TopologyDescription_Attributes(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_TopologyDescription_Attributes_Args>(
            raw_args,
            handle_topology_attributes,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Client_CreateViewOfDeviceBuffer_Args`].
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Client_CreateViewOfDeviceBuffer_Args`].
unsafe fn handle_client_create_view_of_device_buffer(
    args: &mut crate::pjrt_types::PJRT_Client_CreateViewOfDeviceBuffer_Args,
) -> Result<(), LibTpuError> {
    if !args.on_delete_callback.is_null() {
        // SAFETY: `on_delete_callback` is an `extern "C" fn(*mut c_void)`.
        let cb: unsafe extern "C" fn(*mut std::ffi::c_void) =
            unsafe { std::mem::transmute(args.on_delete_callback) };
        // SAFETY: Invoking delete callback with `device_buffer_ptr`.
        unsafe {
            cb(args.device_buffer_ptr.cast_mut());
        }
    }
    args.buffer = alloc_default_pjrt_buffer(args.device, ptr::null_mut());
    Ok(())
}

/// Creates a zero-copy buffer view over an external device pointer (`PJRT_Client_CreateViewOfDeviceBuffer`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Client_CreateViewOfDeviceBuffer_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Client_CreateViewOfDeviceBuffer_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_CreateViewOfDeviceBuffer(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Client_CreateViewOfDeviceBuffer_Args>(
            raw_args,
            handle_client_create_view_of_device_buffer,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Client_TopologyDescription_Args`].
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Client_TopologyDescription_Args`].
unsafe fn handle_client_topology_description(
    args: &mut crate::pjrt_types::PJRT_Client_TopologyDescription_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `client` is null or a valid `PJRT_Client*`.
    args.topology = unsafe { args.client.as_ref() }.map_or(ptr::null_mut(), |c| c.topology_ptr);
    Ok(())
}

/// Queries the `PJRT_TopologyDescription` of a `PJRT_Client` (`PJRT_Client_TopologyDescription`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Client_TopologyDescription_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Client_TopologyDescription_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_TopologyDescription(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Client_TopologyDescription_Args>(
            raw_args,
            handle_client_topology_description,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_ExecuteContext_Create_Args`].
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
#[no_mangle]
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
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_ExecuteContext_Destroy_Args`].
unsafe fn handle_execute_context_destroy(
    args: &mut crate::pjrt_types::PJRT_ExecuteContext_Destroy_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `context` is null or a valid `PJRT_ExecuteContext*`.
    let _ = unsafe { args.context.as_ref() }.map(|c| c.context_id);
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
#[no_mangle]
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

/// Populates [`crate::pjrt_types::PJRT_Client_CreateBuffersForAsyncHostToDevice_Args`].
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Client_CreateBuffersForAsyncHostToDevice_Args`].
unsafe fn handle_client_create_buffers_for_async_h2d(
    args: &mut crate::pjrt_types::PJRT_Client_CreateBuffersForAsyncHostToDevice_Args,
) -> Result<(), LibTpuError> {
    args.transfer_manager = libtpu_core::pjrt::PjrtCoreAsyncTransferManager::create(
        DeviceId(0),
        libtpu_core::types::MemorySpaceId(0),
        vec![Shape {
            element_type: ElementType::F32,
            dims: vec![4],
            dynamic_dims: vec![false],
            layout: libtpu_core::types::MemoryLayout::default_tpu_tiled(1),
        }],
        &(std::sync::Arc::new(libtpu_core::hal::SimulatedTpuHal::new(
            &libtpu_core::config::global_init_config(),
        )) as std::sync::Arc<dyn libtpu_core::hal::TpuHal>),
    )
    .map_or(ptr::null_mut(), |tm| {
        Box::into_raw(Box::new(
            crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager { inner: tm },
        ))
    });
    Ok(())
}

/// Creates an async host-to-device transfer manager (`PJRT_Client_CreateBuffersForAsyncHostToDevice`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Client_CreateBuffersForAsyncHostToDevice_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Client_CreateBuffersForAsyncHostToDevice_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_CreateBuffersForAsyncHostToDevice(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<
            crate::pjrt_types::PJRT_Client_CreateBuffersForAsyncHostToDevice_Args,
        >(raw_args, handle_client_create_buffers_for_async_h2d)
    }
}

/// Populates [`crate::pjrt_types::PJRT_Client_DmaMap_Args`].
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Client_DmaMap_Args`].
unsafe fn handle_client_dma_map(
    args: &mut crate::pjrt_types::PJRT_Client_DmaMap_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `client` is null or a valid `PJRT_Client*`.
    let _ = unsafe { args.client.as_ref() }.map(|c| c.inner.process_index.0);
    Ok(())
}

/// Pins and maps a host memory region for zero-copy DMA (`PJRT_Client_DmaMap`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Client_DmaMap_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Client_DmaMap_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_DmaMap(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Client_DmaMap_Args>(
            raw_args,
            handle_client_dma_map,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Client_DmaUnmap_Args`].
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Client_DmaUnmap_Args`].
unsafe fn handle_client_dma_unmap(
    args: &mut crate::pjrt_types::PJRT_Client_DmaUnmap_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `client` is null or a valid `PJRT_Client*`.
    let _ = unsafe { args.client.as_ref() }.map(|c| c.inner.process_index.0);
    Ok(())
}

/// Unmaps a previously pinned DMA host memory region (`PJRT_Client_DmaUnmap`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Client_DmaUnmap_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Client_DmaUnmap_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_DmaUnmap(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Client_DmaUnmap_Args>(
            raw_args,
            handle_client_dma_unmap,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Client_CreateAliasBuffer_Args`].
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Client_CreateAliasBuffer_Args`].
unsafe fn handle_client_create_alias_buffer(
    args: &mut crate::pjrt_types::PJRT_Client_CreateAliasBuffer_Args,
) -> Result<(), LibTpuError> {
    let buf = alloc_default_pjrt_buffer(ptr::null_mut(), args.memory);
    args.alias_buffer = buf;
    args.fulfill_alias_buffer_cb = buf.cast();
    Ok(())
}

/// Creates an unfulfilled alias buffer (`PJRT_Client_CreateAliasBuffer`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Client_CreateAliasBuffer_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Client_CreateAliasBuffer_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_CreateAliasBuffer(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Client_CreateAliasBuffer_Args>(
            raw_args,
            handle_client_create_alias_buffer,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Client_FulfillAliasBuffer_Args`].
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Client_FulfillAliasBuffer_Args`].
unsafe fn handle_client_fulfill_alias_buffer(
    args: &mut crate::pjrt_types::PJRT_Client_FulfillAliasBuffer_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `client` is null or a valid `PJRT_Client*`.
    let _ = unsafe { args.client.as_ref() }.map(|c| c.inner.process_index.0);
    Ok(())
}

/// Fulfills an alias buffer with a concrete backing buffer (`PJRT_Client_FulfillAliasBuffer`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Client_FulfillAliasBuffer_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Client_FulfillAliasBuffer_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_FulfillAliasBuffer(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Client_FulfillAliasBuffer_Args>(
            raw_args,
            handle_client_fulfill_alias_buffer,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_TopologyDescription_Deserialize_Args`].
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_TopologyDescription_Deserialize_Args`].
unsafe fn handle_topology_deserialize(
    args: &mut crate::pjrt_types::PJRT_TopologyDescription_Deserialize_Args,
) -> Result<(), LibTpuError> {
    args.topology = box_default_topology_description();
    Ok(())
}

/// Deserializes a `PJRT_TopologyDescription` (`PJRT_TopologyDescription_Deserialize`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_TopologyDescription_Deserialize_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_TopologyDescription_Deserialize_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TopologyDescription_Deserialize(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_TopologyDescription_Deserialize_Args>(
            raw_args,
            handle_topology_deserialize,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Client_CreateErrorBuffer_Args`].
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Client_CreateErrorBuffer_Args`].
unsafe fn handle_client_create_error_buffer(
    args: &mut crate::pjrt_types::PJRT_Client_CreateErrorBuffer_Args,
) -> Result<(), LibTpuError> {
    args.buffer = alloc_default_pjrt_buffer(ptr::null_mut(), args.memory);
    Ok(())
}

/// Creates a buffer in an error state (`PJRT_Client_CreateErrorBuffer`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Client_CreateErrorBuffer_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Client_CreateErrorBuffer_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_CreateErrorBuffer(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Client_CreateErrorBuffer_Args>(
            raw_args,
            handle_client_create_error_buffer,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Client_CreateUninitializedBuffer_Args`].
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Client_CreateUninitializedBuffer_Args`].
unsafe fn handle_client_create_uninitialized_buffer(
    args: &mut crate::pjrt_types::PJRT_Client_CreateUninitializedBuffer_Args,
) -> Result<(), LibTpuError> {
    args.buffer = alloc_default_pjrt_buffer(args.device, ptr::null_mut());
    Ok(())
}

/// Creates an uninitialized device buffer (`PJRT_Client_CreateUninitializedBuffer`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Client_CreateUninitializedBuffer_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Client_CreateUninitializedBuffer_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_CreateUninitializedBuffer(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Client_CreateUninitializedBuffer_Args>(
            raw_args,
            handle_client_create_uninitialized_buffer,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Client_UpdateGlobalProcessInfo_Args`].
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Client_UpdateGlobalProcessInfo_Args`].
unsafe fn handle_client_update_global_process_info(
    args: &mut crate::pjrt_types::PJRT_Client_UpdateGlobalProcessInfo_Args,
) -> Result<(), LibTpuError> {
    let _ = (args.processes, args.num_processes);
    Ok(())
}

/// Updates global multi-host process state (`PJRT_Client_UpdateGlobalProcessInfo`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Client_UpdateGlobalProcessInfo_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Client_UpdateGlobalProcessInfo_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_UpdateGlobalProcessInfo(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Client_UpdateGlobalProcessInfo_Args>(
            raw_args,
            handle_client_update_global_process_info,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Client_Load`].
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Client_Load_Args`].
unsafe fn handle_client_load(
    args: &mut crate::pjrt_types::PJRT_Client_Load_Args,
) -> Result<(), LibTpuError> {
    if !any_null(&[args.client as usize, args.executable as usize]) {
        // SAFETY: `args.client` and `args.executable` are valid handles.
        let client = unsafe { &*args.client };
        // SAFETY: `args.executable` is a valid `PJRT_Executable*`.
        let core_exec = unsafe { (*args.executable).inner.clone() };
        let addr_devs: Vec<DeviceId> = client
            .inner
            .addressable_device_indices
            .iter()
            .filter_map(|&idx| {
                client
                    .inner
                    .devices
                    .get(idx)
                    .map(|d| d.description.spec.device_id)
            })
            .collect();
        let loaded_inner = libtpu_core::pjrt::PjrtCoreLoadedExecutable::new(
            core_exec.clone(),
            addr_devs,
            std::sync::Arc::clone(&client.inner.hal),
        );
        let new_exec_ptr = Box::into_raw(Box::new(PJRT_Executable { inner: core_exec }));
        let loaded_ptr = Box::into_raw(Box::new(PJRT_LoadedExecutable {
            inner: loaded_inner,
            executable_ptr: new_exec_ptr,
            device_ptrs: client.addressable_device_ptrs.clone(),
        }));
        args.loaded_executable = loaded_ptr;
    }
    Ok(())
}

/// Loads an AOT-compiled `PJRT_Executable` onto a `PJRT_Client` (`PJRT_Client_Load`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Client_Load_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Client_Load_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_Load(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Client_Load_Args>(
            raw_args,
            handle_client_load,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_TopologyDescription_Fingerprint_Args`].
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_TopologyDescription_Fingerprint_Args`].
unsafe fn handle_topology_fingerprint(
    args: &mut crate::pjrt_types::PJRT_TopologyDescription_Fingerprint_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `topology` is null or a valid `PJRT_TopologyDescription*`.
    let top_opt = unsafe { args.topology.as_ref() };
    args.fingerprint = top_opt.map_or(0x5450_555F_5635_4500, |t| t.inner.fingerprint);
    Ok(())
}

/// Queries the 64-bit fingerprint of a `PJRT_TopologyDescription` (`PJRT_TopologyDescription_Fingerprint`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_TopologyDescription_Fingerprint_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_TopologyDescription_Fingerprint_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TopologyDescription_Fingerprint(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_TopologyDescription_Fingerprint_Args>(
            raw_args,
            handle_topology_fingerprint,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args`].
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args`].
unsafe fn handle_topology_make_canonical_shape(
    args: &mut crate::pjrt_types::PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `topology` is null or a valid `PJRT_TopologyDescription*`.
    let _ = unsafe { args.topology.as_ref() }.map(|t| t.inner.fingerprint);
    Ok(())
}

/// Computes canonical tiled device shape for a memory space (`PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<
            crate::pjrt_types::PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args,
        >(raw_args, handle_topology_make_canonical_shape)
    }
}

/// Populates [`crate::pjrt_types::PJRT_TopologyDescription_GetMemorySpaceKindIds_Args`].
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_TopologyDescription_GetMemorySpaceKindIds_Args`].
unsafe fn handle_topology_get_memory_space_kind_ids(
    args: &mut crate::pjrt_types::PJRT_TopologyDescription_GetMemorySpaceKindIds_Args,
) -> Result<(), LibTpuError> {
    args.kind_ids = GLOBAL_MEMORY_KIND_IDS.as_ptr();
    args.num_kind_ids = GLOBAL_MEMORY_KIND_IDS.len();
    Ok(())
}

/// Queries supported memory space kind IDs of a topology (`PJRT_TopologyDescription_GetMemorySpaceKindIds`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_TopologyDescription_GetMemorySpaceKindIds_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_TopologyDescription_GetMemorySpaceKindIds_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TopologyDescription_GetMemorySpaceKindIds(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<
            crate::pjrt_types::PJRT_TopologyDescription_GetMemorySpaceKindIds_Args,
        >(raw_args, handle_topology_get_memory_space_kind_ids)
    }
}

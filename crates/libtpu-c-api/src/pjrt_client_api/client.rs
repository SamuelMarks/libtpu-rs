//! `PJRT_Client_*` C API entrypoints for client lifecycle, device/memory queries, and compilation.

use super::*;

/// Strongly-typed handler for [`PJRT_Client_Create`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Client_Create_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Client_Create_Args`].
unsafe fn handle_client_create(args: &mut PJRT_Client_Create_Args) -> Result<(), LibTpuError> {
    let mut cfg = libtpu_core::config::global_init_config();
    if !args.create_options.is_null() {
        // SAFETY: `args.create_options` points to `args.num_options` elements of `PJRT_NamedValue`.
        let opts = unsafe {
            std::slice::from_raw_parts(
                args.create_options
                    .cast::<crate::pjrt_types::PJRT_NamedValue>(),
                args.num_options,
            )
        };
        for opt in opts {
            if !opt.name.is_null() {
                // SAFETY: `opt.name` points to `opt.name_size` UTF-8 bytes.
                let name_bytes =
                    unsafe { std::slice::from_raw_parts(opt.name.cast::<u8>(), opt.name_size) };
                let name = String::from_utf8_lossy(name_bytes);
                if matches!(name.as_ref(), "node_id" | "process_index") {
                    let val = i64::from_ne_bytes(opt.value_payload.to_ne_bytes());
                    cfg.process_index =
                        libtpu_core::types::ProcessIndex(i32::try_from(val).unwrap_or(0));
                }
            }
        }
    }
    let inner = PjrtCoreClient::from_config(&cfg);
    let platform_name = to_c_string(&inner.topology.spec.platform_name);
    let platform_version = to_c_string(&inner.topology.spec.platform_version);

    let mut memory_ptrs = Vec::with_capacity(inner.memories.len());
    for m in &inner.memories {
        memory_ptrs.push(Box::into_raw(Box::new(PJRT_Memory {
            inner: m.clone(),
            addressable_by_device_ptrs: Vec::new(),
        })));
    }

    let mut device_ptrs = Vec::with_capacity(inner.devices.len());
    let mut addressable_device_ptrs = Vec::new();
    for (d, dev_mems) in inner.devices.iter().zip(memory_ptrs.chunks(3)) {
        let desc_ptr = Box::into_raw(Box::new(PJRT_DeviceDescription::new(d.description.clone())));
        let default_memory_ptr = dev_mems.first().copied().unwrap_or(ptr::null_mut());
        let addressable_memory_ptrs = dev_mems.to_vec();
        let dev_ptr = Box::into_raw(Box::new(PJRT_Device {
            inner: d.clone(),
            desc_ptr,
            default_memory_ptr,
            addressable_memory_ptrs,
            hal: std::sync::Arc::clone(&inner.hal),
        }));
        for &m_ptr in dev_mems {
            // SAFETY: `m_ptr` was allocated in `memory_ptrs` above.
            unsafe {
                (*m_ptr).addressable_by_device_ptrs.push(dev_ptr);
            }
        }
        addressable_device_ptrs.push(dev_ptr);
        device_ptrs.push(dev_ptr);
    }

    let mut topo_descs = Vec::with_capacity(inner.topology.device_descriptions.len());
    for td in &inner.topology.device_descriptions {
        topo_descs.push(Box::into_raw(Box::new(PJRT_DeviceDescription::new(
            td.clone(),
        ))));
    }
    let topology_ptr = Box::into_raw(Box::new(PJRT_TopologyDescription {
        inner: inner.topology.clone(),
        desc_ptrs: topo_descs,
    }));

    // SAFETY: `args.kv_put_callback` is null or a valid `PJRT_KeyValuePutCallback` pointer.
    unsafe {
        notify_kv_put_topology_ready(args.kv_put_callback, args.kv_put_user_arg);
    }

    let client_box = Box::new(PJRT_Client {
        inner,
        platform_name,
        platform_version,
        device_ptrs,
        addressable_device_ptrs,
        memory_ptrs,
        topology_ptr,
        kv_get_callback: args.kv_get_callback,
        kv_get_user_arg: args.kv_get_user_arg,
        kv_put_callback: args.kv_put_callback,
        kv_put_user_arg: args.kv_put_user_arg,
        kv_try_get_callback: args.kv_try_get_callback,
        kv_try_get_user_arg: args.kv_try_get_user_arg,
    });
    args.client = Box::into_raw(client_box);
    Ok(())
}

/// Invokes an optional `PJRT_KeyValuePutCallback` to publish `"tpu_topology_ready"` = `"1"`.
///
/// # Arguments
///
/// * `kv_put_callback` - Optional C callback pointer.
/// * `kv_put_user_arg` - User argument pointer passed to `kv_put_callback`.
///
/// # Safety
///
/// If non-null, `kv_put_callback` must be a valid C function pointer matching `PJRT_KeyValuePutCallback`.
unsafe fn notify_kv_put_topology_ready(
    kv_put_callback: *const std::ffi::c_void,
    kv_put_user_arg: *mut std::ffi::c_void,
) {
    /// C callback type for `PJRT_KeyValuePutCallback`.
    type KvPutCallbackFn = unsafe extern "C" fn(
        *const std::ffi::c_char,
        usize,
        *const std::ffi::c_char,
        usize,
        *mut std::ffi::c_void,
    ) -> *mut PJRT_Error;
    if kv_put_callback.is_null() {
        return;
    }
    // SAFETY: `kv_put_callback` is a valid C function pointer supplied by the caller.
    let kv_put: KvPutCallbackFn = unsafe { std::mem::transmute(kv_put_callback) };
    let key = b"tpu_topology_ready";
    let val = b"1";
    // SAFETY: Invoking caller's KV put callback with static byte slices and `kv_put_user_arg`.
    let kv_err = unsafe {
        kv_put(
            key.as_ptr().cast::<std::ffi::c_char>(),
            key.len(),
            val.as_ptr().cast::<std::ffi::c_char>(),
            val.len(),
            kv_put_user_arg,
        )
    };
    if !kv_err.is_null() {
        // SAFETY: `kv_err` is a heap-allocated `PJRT_Error` returned by the callback.
        unsafe {
            drop(Box::from_raw(kv_err));
        }
    }
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Client_Create(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe { dispatch_typed_pjrt_args::<PJRT_Client_Create_Args>(raw_args, handle_client_create) }
}

/// Strongly-typed handler for [`PJRT_Client_Destroy`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Client_Destroy_Args`] structure.
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Client_Destroy(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe { dispatch_typed_pjrt_args::<PJRT_Client_Destroy_Args>(raw_args, handle_client_destroy) }
}

/// Strongly-typed handler for [`PJRT_Client_PlatformName`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Client_PlatformStr_Args`] structure.
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
#[unsafe(no_mangle)]
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
/// # Arguments
///
/// * `args` - Validated [`PJRT_Client_PlatformStr_Args`] structure.
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
#[unsafe(no_mangle)]
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
/// # Arguments
///
/// * `args` - Validated [`PJRT_Client_ProcessIndex_Args`] structure.
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
#[unsafe(no_mangle)]
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
/// # Arguments
///
/// * `args` - Validated [`PJRT_Client_Devices_Args`] structure.
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Client_Devices(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe { dispatch_typed_pjrt_args::<PJRT_Client_Devices_Args>(raw_args, handle_client_devices) }
}

/// Strongly-typed handler for [`PJRT_Client_AddressableDevices`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Client_Devices_Args`] structure.
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
#[unsafe(no_mangle)]
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
/// # Arguments
///
/// * `args` - Validated [`PJRT_Client_LookupDevice_Args`] structure.
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
#[unsafe(no_mangle)]
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
/// # Arguments
///
/// * `args` - Validated [`PJRT_Client_AddressableMemories_Args`] structure.
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
#[unsafe(no_mangle)]
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
/// # Arguments
///
/// * `args` - Validated [`PJRT_Client_Compile_Args`] structure.
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
    let exec_ptr = Box::into_raw(Box::new(PJRT_Executable::new(loaded.executable.clone())));
    let loaded_ptr = Box::into_raw(Box::new(PJRT_LoadedExecutable::new(
        loaded,
        exec_ptr,
        client.addressable_device_ptrs.clone(),
    )));
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Client_Compile(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe { dispatch_typed_pjrt_args::<PJRT_Client_Compile_Args>(raw_args, handle_client_compile) }
}

/// Populates [`crate::pjrt_types::PJRT_Client_LookupDevice_Args`] for addressable device lookup.
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Client_LookupDevice_Args`] structure.
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
#[unsafe(no_mangle)]
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
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Client_DefaultDeviceAssignment_Args`] structure.
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
#[unsafe(no_mangle)]
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

/// Populates [`crate::pjrt_types::PJRT_Client_TopologyDescription_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Client_TopologyDescription_Args`] structure.
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
#[unsafe(no_mangle)]
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

/// Populates [`crate::pjrt_types::PJRT_Client_UpdateGlobalProcessInfo_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Client_UpdateGlobalProcessInfo_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Client_UpdateGlobalProcessInfo_Args`].
unsafe fn handle_client_update_global_process_info(
    args: &mut crate::pjrt_types::PJRT_Client_UpdateGlobalProcessInfo_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `args.client` is null or a valid `PJRT_Client*`.
    if let Some(client) = unsafe { args.client.as_ref() } {
        let infos = if args.processes.is_null() {
            Vec::new()
        } else {
            // SAFETY: `args.processes` points to `args.num_processes` `PJRT_ProcessInfo` elements.
            let slice = unsafe { std::slice::from_raw_parts(args.processes, args.num_processes) };
            slice
                .iter()
                .map(|p| libtpu_core::pjrt::ProcessInfoRecord {
                    process_id: p.task_id,
                    incarnation_id: p.incarnation_id,
                    state: p.state,
                    error_code: i32::try_from(p.error_code).unwrap_or(0),
                    error_message: String::new(),
                })
                .collect()
        };
        client.inner.update_global_process_info(infos);
    }
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
#[unsafe(no_mangle)]
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
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Client_Load_Args`] structure.
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
        let new_exec_ptr = Box::into_raw(Box::new(PJRT_Executable::new(core_exec)));
        let loaded_ptr = Box::into_raw(Box::new(PJRT_LoadedExecutable::new(
            loaded_inner,
            new_exec_ptr,
            client.addressable_device_ptrs.clone(),
        )));
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Client_Load(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Client_Load_Args>(
            raw_args,
            handle_client_load,
        )
    }
}

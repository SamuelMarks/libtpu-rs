//! PJRT device, device description, and memory space C API entrypoints (`PJRT_Device_*`, `PJRT_DeviceDescription_*`, `PJRT_Memory_*`).

use super::*;

/// Populates [`crate::pjrt_types::PJRT_DeviceDescription_Id_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_DeviceDescription_Id_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_DeviceDescription_Id_Args`].
unsafe fn handle_device_description_id(
    args: &mut crate::pjrt_types::PJRT_DeviceDescription_Id_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `device_description` is null or a valid `PJRT_DeviceDescription*`.
    args.id = unsafe { args.device_description.as_ref() }
        .map_or(0, |d| i32::try_from(d.inner.spec.device_id.0).unwrap_or(0));
    Ok(())
}

/// Queries the global ID of a `PJRT_DeviceDescription` (`PJRT_DeviceDescription_Id`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_DeviceDescription_Id_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_DeviceDescription_Id_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_DeviceDescription_Id(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_DeviceDescription_Id_Args>(
            raw_args,
            handle_device_description_id,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_DeviceDescription_ProcessIndex_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_DeviceDescription_ProcessIndex_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_DeviceDescription_ProcessIndex_Args`].
unsafe fn handle_device_description_process_index(
    args: &mut crate::pjrt_types::PJRT_DeviceDescription_ProcessIndex_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `device_description` is null or a valid `PJRT_DeviceDescription*`.
    let desc_opt = unsafe { args.device_description.as_ref() };
    args.process_index = desc_opt.map_or(0, |d| d.inner.spec.process_index.0);
    Ok(())
}

/// Queries the process index of a `PJRT_DeviceDescription` (`PJRT_DeviceDescription_ProcessIndex`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_DeviceDescription_ProcessIndex_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_DeviceDescription_ProcessIndex_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_DeviceDescription_ProcessIndex(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_DeviceDescription_ProcessIndex_Args>(
            raw_args,
            handle_device_description_process_index,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_DeviceDescription_Attributes_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_DeviceDescription_Attributes_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_DeviceDescription_Attributes_Args`].
unsafe fn handle_device_description_attributes(
    args: &mut crate::pjrt_types::PJRT_DeviceDescription_Attributes_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `device_description` is null or a valid `PJRT_DeviceDescription*`.
    let (ptr, len) = unsafe { args.device_description.as_ref() }.map_or(
        (
            GLOBAL_PLUGIN_ATTRIBUTES.as_ptr(),
            GLOBAL_PLUGIN_ATTRIBUTES.len(),
        ),
        |d| (d.attributes.as_ptr(), d.attributes.len()),
    );
    args.attributes = ptr;
    args.num_attributes = len;
    Ok(())
}

/// Queries device-specific attributes (`PJRT_DeviceDescription_Attributes`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_DeviceDescription_Attributes_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_DeviceDescription_Attributes_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_DeviceDescription_Attributes(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_DeviceDescription_Attributes_Args>(
            raw_args,
            handle_device_description_attributes,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_DeviceDescription_Kind_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_DeviceDescription_Kind_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_DeviceDescription_Kind_Args`].
unsafe fn handle_device_description_kind(
    args: &mut crate::pjrt_types::PJRT_DeviceDescription_Kind_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `device_description` is null or a valid `PJRT_DeviceDescription*`.
    let desc_opt = unsafe { args.device_description.as_ref() };
    let (ptr, len) = desc_opt.map_or((c"TPU v5 lite".as_ptr(), 11), |d| {
        (
            d.inner.kind_str.as_ptr().cast::<c_char>(),
            d.inner.kind_str.len(),
        )
    });
    args.device_kind = ptr;
    args.device_kind_size = len;
    Ok(())
}

/// Queries the device kind string (`PJRT_DeviceDescription_Kind`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_DeviceDescription_Kind_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_DeviceDescription_Kind_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_DeviceDescription_Kind(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_DeviceDescription_Kind_Args>(
            raw_args,
            handle_device_description_kind,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_DeviceDescription_DebugString_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_DeviceDescription_DebugString_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_DeviceDescription_DebugString_Args`].
unsafe fn handle_device_description_debug_string(
    args: &mut crate::pjrt_types::PJRT_DeviceDescription_DebugString_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `device_description` is null or a valid `PJRT_DeviceDescription*`.
    let desc_opt = unsafe { args.device_description.as_ref() };
    let (ptr, len) = desc_opt.map_or((c"TPU:0".as_ptr(), 5), |d| {
        (
            d.inner.debug_string.as_ptr().cast::<c_char>(),
            d.inner.debug_string.len(),
        )
    });
    args.debug_string = ptr;
    args.debug_string_size = len;
    Ok(())
}

/// Queries the debug string of a device description (`PJRT_DeviceDescription_DebugString`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_DeviceDescription_DebugString_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_DeviceDescription_DebugString_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_DeviceDescription_DebugString(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_DeviceDescription_DebugString_Args>(
            raw_args,
            handle_device_description_debug_string,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_DeviceDescription_ToString_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_DeviceDescription_ToString_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_DeviceDescription_ToString_Args`].
unsafe fn handle_device_description_to_string(
    args: &mut crate::pjrt_types::PJRT_DeviceDescription_ToString_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `device_description` is null or a valid `PJRT_DeviceDescription*`.
    let desc_opt = unsafe { args.device_description.as_ref() };
    let (ptr, len) = desc_opt.map_or((c"TPU:0".as_ptr(), 5), |d| {
        (
            d.inner.to_string_val.as_ptr().cast::<c_char>(),
            d.inner.to_string_val.len(),
        )
    });
    args.to_string = ptr;
    args.to_string_size = len;
    Ok(())
}

/// Queries the display string of a device description (`PJRT_DeviceDescription_ToString`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_DeviceDescription_ToString_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_DeviceDescription_ToString_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_DeviceDescription_ToString(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_DeviceDescription_ToString_Args>(
            raw_args,
            handle_device_description_to_string,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Device_GetDescription_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Device_GetDescription_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Device_GetDescription_Args`].
unsafe fn handle_device_get_description(
    args: &mut crate::pjrt_types::PJRT_Device_GetDescription_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `device` is null or a valid `PJRT_Device*`.
    let dev_opt = unsafe { args.device.as_ref() };
    args.device_description = dev_opt.map_or(ptr::null_mut(), |d| d.desc_ptr);
    Ok(())
}

/// Queries the `PJRT_DeviceDescription` of a `PJRT_Device` (`PJRT_Device_GetDescription`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Device_GetDescription_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Device_GetDescription_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Device_GetDescription(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Device_GetDescription_Args>(
            raw_args,
            handle_device_get_description,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Device_IsAddressable_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Device_IsAddressable_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Device_IsAddressable_Args`].
unsafe fn handle_device_is_addressable(
    args: &mut crate::pjrt_types::PJRT_Device_IsAddressable_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `device` is null or a valid `PJRT_Device*`.
    let dev_opt = unsafe { args.device.as_ref() };
    args.is_addressable = dev_opt.is_none_or(|d| d.inner.description.spec.is_addressable);
    Ok(())
}

/// Checks whether a `PJRT_Device` is addressable by this process (`PJRT_Device_IsAddressable`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Device_IsAddressable_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Device_IsAddressable_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Device_IsAddressable(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Device_IsAddressable_Args>(
            raw_args,
            handle_device_is_addressable,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Device_LocalHardwareId_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Device_LocalHardwareId_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Device_LocalHardwareId_Args`].
unsafe fn handle_device_local_hardware_id(
    args: &mut crate::pjrt_types::PJRT_Device_LocalHardwareId_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `device` is null or a valid `PJRT_Device*`.
    let dev_opt = unsafe { args.device.as_ref() };
    args.local_hardware_id = dev_opt.map_or(0, |d| d.inner.description.spec.local_hardware_id.0);
    Ok(())
}

/// Queries the local hardware ID of a `PJRT_Device` (`PJRT_Device_LocalHardwareId`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Device_LocalHardwareId_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Device_LocalHardwareId_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Device_LocalHardwareId(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Device_LocalHardwareId_Args>(
            raw_args,
            handle_device_local_hardware_id,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Device_AddressableMemories_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Device_AddressableMemories_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Device_AddressableMemories_Args`].
unsafe fn handle_device_addressable_memories(
    args: &mut crate::pjrt_types::PJRT_Device_AddressableMemories_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `device` is null or a valid `PJRT_Device*`.
    let (ptr, len) = unsafe { args.device.as_ref() }.map_or((ptr::null(), 0), |d| {
        (
            d.addressable_memory_ptrs.as_ptr(),
            d.addressable_memory_ptrs.len(),
        )
    });
    args.memories = ptr;
    args.num_memories = len;
    Ok(())
}

/// Queries the addressable memory spaces of a `PJRT_Device` (`PJRT_Device_AddressableMemories`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Device_AddressableMemories_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Device_AddressableMemories_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Device_AddressableMemories(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Device_AddressableMemories_Args>(
            raw_args,
            handle_device_addressable_memories,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Device_DefaultMemory_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Device_DefaultMemory_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Device_DefaultMemory_Args`].
unsafe fn handle_device_default_memory(
    args: &mut crate::pjrt_types::PJRT_Device_DefaultMemory_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `device` is null or a valid `PJRT_Device*`.
    args.memory = unsafe { args.device.as_ref() }.map_or(ptr::null_mut(), |d| d.default_memory_ptr);
    Ok(())
}

/// Queries the default memory space of a `PJRT_Device` (`PJRT_Device_DefaultMemory`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Device_DefaultMemory_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Device_DefaultMemory_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Device_DefaultMemory(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Device_DefaultMemory_Args>(
            raw_args,
            handle_device_default_memory,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Device_MemoryStats_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Device_MemoryStats_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Device_MemoryStats_Args`].
unsafe fn handle_device_memory_stats(
    args: &mut crate::pjrt_types::PJRT_Device_MemoryStats_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `device` is null or a valid `PJRT_Device*`.
    let stats = unsafe { args.device.as_ref() }
        .and_then(|d| d.hal.memory_stats(d.inner.description.spec.device_id).ok());
    let used = stats.as_ref().map_or(0_i64, |s| s.bytes_in_use);
    let peak = stats.as_ref().map_or(0_i64, |s| s.peak_bytes_in_use);
    let num_allocs = stats.as_ref().map_or(0_i64, |s| s.num_allocs);
    let largest_alloc = stats.as_ref().map_or(0_i64, |s| s.largest_alloc_size);
    let hbm_limit = stats.as_ref().map_or(17_179_869_184_i64, |s| s.bytes_limit);
    let largest_free = stats
        .as_ref()
        .map_or(17_179_869_184_i64, |s| s.largest_free_block_bytes);
    args.bytes_in_use = used;
    args.peak_bytes_in_use = peak;
    args.peak_bytes_in_use_is_set = true;
    args.num_allocs = num_allocs;
    args.num_allocs_is_set = true;
    args.largest_alloc_size = largest_alloc;
    args.largest_alloc_size_is_set = true;
    args.bytes_limit = hbm_limit;
    args.bytes_limit_is_set = true;
    args.bytes_reserved = used;
    args.bytes_reserved_is_set = true;
    args.peak_bytes_reserved = peak;
    args.peak_bytes_reserved_is_set = true;
    args.bytes_reservable_limit = hbm_limit;
    args.bytes_reservable_limit_is_set = true;
    args.largest_free_block_bytes = largest_free;
    args.largest_free_block_bytes_is_set = true;
    args.pool_bytes = used;
    args.pool_bytes_is_set = true;
    args.peak_pool_bytes = peak;
    args.peak_pool_bytes_is_set = true;
    Ok(())
}

/// Queries HBM allocation statistics of a `PJRT_Device` (`PJRT_Device_MemoryStats`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Device_MemoryStats_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Device_MemoryStats_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Device_MemoryStats(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Device_MemoryStats_Args>(
            raw_args,
            handle_device_memory_stats,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Memory_Id_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Memory_Id_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Memory_Id_Args`].
unsafe fn handle_memory_id(
    args: &mut crate::pjrt_types::PJRT_Memory_Id_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `memory` is null or a valid `PJRT_Memory*`.
    args.id = unsafe { args.memory.as_ref() }.map_or(0, |m| m.inner.id.0);
    Ok(())
}

/// Queries the unique ID of a `PJRT_Memory` (`PJRT_Memory_Id`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Memory_Id_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Memory_Id_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Memory_Id(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Memory_Id_Args>(
            raw_args,
            handle_memory_id,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Memory_Kind_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Memory_Kind_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Memory_Kind_Args`].
unsafe fn handle_memory_kind(
    args: &mut crate::pjrt_types::PJRT_Memory_Kind_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `memory` is null or a valid `PJRT_Memory*`.
    let (ptr, len) = unsafe { args.memory.as_ref() }.map_or((c"device".as_ptr(), 6), |m| {
        (
            m.inner.kind.as_str().as_ptr().cast::<c_char>(),
            m.inner.kind.as_str().len(),
        )
    });
    args.kind = ptr;
    args.kind_size = len;
    Ok(())
}

/// Queries the memory kind string of a `PJRT_Memory` (`PJRT_Memory_Kind`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Memory_Kind_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Memory_Kind_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Memory_Kind(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Memory_Kind_Args>(
            raw_args,
            handle_memory_kind,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Memory_DebugString_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Memory_DebugString_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Memory_DebugString_Args`].
unsafe fn handle_memory_debug_string(
    args: &mut crate::pjrt_types::PJRT_Memory_DebugString_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `memory` is null or a valid `PJRT_Memory*`.
    let (ptr, len) = unsafe { args.memory.as_ref() }.map_or((c"TPU_HBM".as_ptr(), 7), |m| {
        (
            m.inner.debug_string.as_ptr().cast::<c_char>(),
            m.inner.debug_string.len(),
        )
    });
    args.debug_string = ptr;
    args.debug_string_size = len;
    Ok(())
}

/// Queries the debug string of a `PJRT_Memory` (`PJRT_Memory_DebugString`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Memory_DebugString_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Memory_DebugString_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Memory_DebugString(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Memory_DebugString_Args>(
            raw_args,
            handle_memory_debug_string,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Memory_ToString_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Memory_ToString_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Memory_ToString_Args`].
unsafe fn handle_memory_to_string(
    args: &mut crate::pjrt_types::PJRT_Memory_ToString_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `memory` is null or a valid `PJRT_Memory*`.
    let (ptr, len) = unsafe { args.memory.as_ref() }.map_or((c"TPU_HBM".as_ptr(), 7), |m| {
        (
            m.inner.to_string_val.as_ptr().cast::<c_char>(),
            m.inner.to_string_val.len(),
        )
    });
    args.to_string = ptr;
    args.to_string_size = len;
    Ok(())
}

/// Queries the display string of a `PJRT_Memory` (`PJRT_Memory_ToString`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Memory_ToString_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Memory_ToString_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Memory_ToString(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Memory_ToString_Args>(
            raw_args,
            handle_memory_to_string,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Memory_AddressableByDevices_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Memory_AddressableByDevices_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Memory_AddressableByDevices_Args`].
unsafe fn handle_memory_addressable_by_devices(
    args: &mut crate::pjrt_types::PJRT_Memory_AddressableByDevices_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `memory` is null or a valid `PJRT_Memory*`.
    let (ptr, len) = unsafe { args.memory.as_ref() }.map_or((ptr::null(), 0), |m| {
        (
            m.addressable_by_device_ptrs.as_ptr(),
            m.addressable_by_device_ptrs.len(),
        )
    });
    args.devices = ptr;
    args.num_devices = len;
    Ok(())
}

/// Queries the devices that can address a `PJRT_Memory` (`PJRT_Memory_AddressableByDevices`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Memory_AddressableByDevices_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Memory_AddressableByDevices_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Memory_AddressableByDevices(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Memory_AddressableByDevices_Args>(
            raw_args,
            handle_memory_addressable_by_devices,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Memory_Kind_Id_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Memory_Kind_Id_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Memory_Kind_Id_Args`].
unsafe fn handle_memory_kind_id(
    args: &mut crate::pjrt_types::PJRT_Memory_Kind_Id_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `memory` is null or a valid `PJRT_Memory*`.
    args.kind_id = unsafe { args.memory.as_ref() }.map_or(0, |m| m.inner.kind.kind_id());
    Ok(())
}

/// Queries the numeric kind ID of a `PJRT_Memory` (`PJRT_Memory_Kind_Id`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Memory_Kind_Id_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Memory_Kind_Id_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Memory_Kind_Id(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Memory_Kind_Id_Args>(
            raw_args,
            handle_memory_kind_id,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Device_PoisonExecution_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Device_PoisonExecution_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Device_PoisonExecution_Args`].
unsafe fn handle_device_poison_execution(
    args: &mut crate::pjrt_types::PJRT_Device_PoisonExecution_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `device` is null or a valid `PJRT_Device*`.
    if let Some(d) = unsafe { args.device.as_ref() } {
        if args.launch_id < 0 {
            let _ = d.hal.poison_device(
                d.inner.description.spec.device_id,
                format!("launch_id={}", args.launch_id),
            );
        }
    }
    args.poisoned = true;
    Ok(())
}

/// Poisons a device execution for a given launch ID (`PJRT_Device_PoisonExecution`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Device_PoisonExecution_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Device_PoisonExecution_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Device_PoisonExecution(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Device_PoisonExecution_Args>(
            raw_args,
            handle_device_poison_execution,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Device_CreateAsyncTrackingEvent_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Device_CreateAsyncTrackingEvent_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Device_CreateAsyncTrackingEvent_Args`].
unsafe fn handle_device_create_async_tracking_event(
    args: &mut crate::pjrt_types::PJRT_Device_CreateAsyncTrackingEvent_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `device` is null or a valid `PJRT_Device*`.
    let _ = unsafe { args.device.as_ref() }.map(|d| d.inner.description.spec.device_id.0);
    args.event = Box::into_raw(Box::new(crate::pjrt_types::PJRT_AsyncTrackingEvent {
        event_id: 1,
    }));
    Ok(())
}

/// Creates an async tracking event on a device (`PJRT_Device_CreateAsyncTrackingEvent`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Device_CreateAsyncTrackingEvent_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Device_CreateAsyncTrackingEvent_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Device_CreateAsyncTrackingEvent(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Device_CreateAsyncTrackingEvent_Args>(
            raw_args,
            handle_device_create_async_tracking_event,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_AsyncTrackingEvent_Destroy_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_AsyncTrackingEvent_Destroy_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_AsyncTrackingEvent_Destroy_Args`].
unsafe fn handle_async_tracking_event_destroy(
    args: &mut crate::pjrt_types::PJRT_AsyncTrackingEvent_Destroy_Args,
) -> Result<(), LibTpuError> {
    if !args.event.is_null() {
        // SAFETY: `args.event` was allocated by `PJRT_Device_CreateAsyncTrackingEvent`.
        unsafe {
            drop(Box::from_raw(args.event));
        }
    }
    Ok(())
}

/// Destroys an async tracking event (`PJRT_AsyncTrackingEvent_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_AsyncTrackingEvent_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_AsyncTrackingEvent_Destroy_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_AsyncTrackingEvent_Destroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_AsyncTrackingEvent_Destroy_Args>(
            raw_args,
            handle_async_tracking_event_destroy,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Device_GetAttributes_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Device_GetAttributes_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Device_GetAttributes_Args`].
unsafe fn handle_device_get_attributes(
    args: &mut crate::pjrt_types::PJRT_Device_GetAttributes_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `device` and `d.desc_ptr` are null or valid pointers.
    let (ptr, len) = unsafe { args.device.as_ref().and_then(|d| d.desc_ptr.as_ref()) }.map_or(
        (
            GLOBAL_PLUGIN_ATTRIBUTES.as_ptr(),
            GLOBAL_PLUGIN_ATTRIBUTES.len(),
        ),
        |desc| (desc.attributes.as_ptr(), desc.attributes.len()),
    );
    args.attributes = ptr;
    args.num_attributes = len;
    Ok(())
}

/// Queries attributes of a `PJRT_Device` (`PJRT_Device_GetAttributes`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Device_GetAttributes_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Device_GetAttributes_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Device_GetAttributes(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Device_GetAttributes_Args>(
            raw_args,
            handle_device_get_attributes,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Device_ClearMemoryStats_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Device_ClearMemoryStats_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Device_ClearMemoryStats_Args`].
unsafe fn handle_device_clear_memory_stats(
    args: &mut crate::pjrt_types::PJRT_Device_ClearMemoryStats_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `device` is null or a valid `PJRT_Device*`.
    let _ = unsafe { args.device.as_ref() }
        .map(|d| d.hal.clear_memory_stats(d.inner.description.spec.device_id));
    Ok(())
}

/// Resets peak and cumulative HBM statistics on a `PJRT_Device` (`PJRT_Device_ClearMemoryStats`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Device_ClearMemoryStats_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Device_ClearMemoryStats_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Device_ClearMemoryStats(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Device_ClearMemoryStats_Args>(
            raw_args,
            handle_device_clear_memory_stats,
        )
    }
}

#[cfg(test)]
#[allow(
    for_loops_over_fallibles,
    clippy::cognitive_complexity,
    clippy::too_many_lines,
    clippy::similar_names
)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::pjrt_client_api::{PJRT_Client_Create, PJRT_Client_Destroy};
    use crate::pjrt_types::{PJRT_Client_Create_Args, PJRT_Client_Destroy_Args};

    #[test]
    fn test_pjrt_device_description_device_and_memory_api() {
        // SAFETY: Testing PJRT device description, device, and memory entrypoints with a valid client.
        unsafe {
            let mut create_args = PJRT_Client_Create_Args {
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
            assert!(PJRT_Client_Create(ptr::addr_of_mut!(create_args).cast()).is_null());
            let client = create_args.client;

            drop(Box::from_raw(PJRT_Device_GetAttributes(ptr::null_mut())));
            for &dev0 in (*client).device_ptrs.first() {
                assert_ne!(format!("{:?}{:?}", *dev0, *(*dev0).desc_ptr), "");
                let desc0 = (*dev0).desc_ptr;
                let mut dd_id = crate::pjrt_types::PJRT_DeviceDescription_Id_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_DeviceDescription_Id_Args>(),
                    extension_start: ptr::null_mut(),
                    device_description: desc0,
                    id: -1,
                    pad: 0,
                };
                assert!(PJRT_DeviceDescription_Id(ptr::addr_of_mut!(dd_id).cast()).is_null());
                assert_eq!(dd_id.id, 0);

                let mut dd_proc = crate::pjrt_types::PJRT_DeviceDescription_ProcessIndex_Args {
                    struct_size: size_of::<
                        crate::pjrt_types::PJRT_DeviceDescription_ProcessIndex_Args,
                    >(),
                    extension_start: ptr::null_mut(),
                    device_description: desc0,
                    process_index: -1,
                    pad: 0,
                };
                assert!(
                    PJRT_DeviceDescription_ProcessIndex(ptr::addr_of_mut!(dd_proc).cast())
                        .is_null()
                );
                assert_eq!(dd_proc.process_index, 0);

                let mut dd_attrs = crate::pjrt_types::PJRT_DeviceDescription_Attributes_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_DeviceDescription_Attributes_Args>(
                    ),
                    extension_start: ptr::null_mut(),
                    device_description: desc0,
                    num_attributes: 0,
                    attributes: ptr::null(),
                };
                assert!(
                    PJRT_DeviceDescription_Attributes(ptr::addr_of_mut!(dd_attrs).cast()).is_null()
                );

                let mut dd_kind = crate::pjrt_types::PJRT_DeviceDescription_Kind_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_DeviceDescription_Kind_Args>(),
                    extension_start: ptr::null_mut(),
                    device_description: desc0,
                    device_kind: ptr::null(),
                    device_kind_size: 0,
                };
                assert!(PJRT_DeviceDescription_Kind(ptr::addr_of_mut!(dd_kind).cast()).is_null());
                assert!(dd_kind.device_kind_size > 0);

                let mut dd_dbg = crate::pjrt_types::PJRT_DeviceDescription_DebugString_Args {
                    struct_size: size_of::<
                        crate::pjrt_types::PJRT_DeviceDescription_DebugString_Args,
                    >(),
                    extension_start: ptr::null_mut(),
                    device_description: desc0,
                    debug_string: ptr::null(),
                    debug_string_size: 0,
                };
                assert!(
                    PJRT_DeviceDescription_DebugString(ptr::addr_of_mut!(dd_dbg).cast()).is_null()
                );
                assert!(dd_dbg.debug_string_size > 0);

                let mut dd_str = crate::pjrt_types::PJRT_DeviceDescription_ToString_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_DeviceDescription_ToString_Args>(
                    ),
                    extension_start: ptr::null_mut(),
                    device_description: desc0,
                    to_string: ptr::null(),
                    to_string_size: 0,
                };
                assert!(
                    PJRT_DeviceDescription_ToString(ptr::addr_of_mut!(dd_str).cast()).is_null()
                );
                assert!(dd_str.to_string_size > 0);

                let mut dv_desc = crate::pjrt_types::PJRT_Device_GetDescription_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_Device_GetDescription_Args>(),
                    extension_start: ptr::null_mut(),
                    device: dev0,
                    device_description: ptr::null_mut(),
                };
                assert!(PJRT_Device_GetDescription(ptr::addr_of_mut!(dv_desc).cast()).is_null());
                assert_eq!(dv_desc.device_description, desc0);

                let mut dv_addr = crate::pjrt_types::PJRT_Device_IsAddressable_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_Device_IsAddressable_Args>(),
                    extension_start: ptr::null_mut(),
                    device: dev0,
                    is_addressable: false,
                    pad: [0; 7],
                };
                assert!(PJRT_Device_IsAddressable(ptr::addr_of_mut!(dv_addr).cast()).is_null());
                assert!(dv_addr.is_addressable);

                let mut dv_hw = crate::pjrt_types::PJRT_Device_LocalHardwareId_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_Device_LocalHardwareId_Args>(),
                    extension_start: ptr::null_mut(),
                    device: dev0,
                    local_hardware_id: -1,
                    pad: 0,
                };
                assert!(PJRT_Device_LocalHardwareId(ptr::addr_of_mut!(dv_hw).cast()).is_null());
                assert_eq!(dv_hw.local_hardware_id, 0);

                let mut dv_mems = crate::pjrt_types::PJRT_Device_AddressableMemories_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_Device_AddressableMemories_Args>(
                    ),
                    extension_start: ptr::null_mut(),
                    device: dev0,
                    memories: ptr::null(),
                    num_memories: 0,
                };
                assert!(
                    PJRT_Device_AddressableMemories(ptr::addr_of_mut!(dv_mems).cast()).is_null()
                );

                let mut dv_defm = crate::pjrt_types::PJRT_Device_DefaultMemory_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_Device_DefaultMemory_Args>(),
                    extension_start: ptr::null_mut(),
                    device: dev0,
                    memory: ptr::null_mut(),
                };
                assert!(PJRT_Device_DefaultMemory(ptr::addr_of_mut!(dv_defm).cast()).is_null());

                let host_floats = [1.0_f32, 2.0, 3.0, 4.0];
                let dims_arr = [4_i64];
                let mut h2d_args = crate::pjrt_types::PJRT_Client_BufferFromHostBuffer_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_Client_BufferFromHostBuffer_Args>(
                    ),
                    extension_start: ptr::null_mut(),
                    client,
                    data: host_floats.as_ptr().cast(),
                    element_type: 11,
                    pad0: 0,
                    dims: dims_arr.as_ptr(),
                    num_dims: 1,
                    byte_strides: ptr::null(),
                    num_byte_strides: 0,
                    host_buffer_semantics: 0,
                    pad1: 0,
                    device: dev0,
                    memory: ptr::null_mut(),
                    device_layout: ptr::null(),
                    done_with_host_buffer: ptr::null_mut(),
                    out_buffer: ptr::null_mut(),
                };
                assert!(
                    crate::pjrt_client_api::PJRT_Client_BufferFromHostBuffer(
                        ptr::addr_of_mut!(h2d_args).cast()
                    )
                    .is_null()
                );

                let mut dv_mstat = crate::pjrt_types::PJRT_Device_MemoryStats_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_Device_MemoryStats_Args>(),
                    extension_start: ptr::null_mut(),
                    device: dev0,
                    bytes_in_use: 0,
                    peak_bytes_in_use: 0,
                    peak_bytes_in_use_is_set: false,
                    pad0: [0; 7],
                    num_allocs: 0,
                    num_allocs_is_set: false,
                    pad1: [0; 7],
                    largest_alloc_size: 0,
                    largest_alloc_size_is_set: false,
                    pad2: [0; 7],
                    bytes_limit: 0,
                    bytes_limit_is_set: false,
                    pad3: [0; 7],
                    bytes_reserved: 0,
                    bytes_reserved_is_set: false,
                    pad4: [0; 7],
                    peak_bytes_reserved: 0,
                    peak_bytes_reserved_is_set: false,
                    pad5: [0; 7],
                    bytes_reservable_limit: 0,
                    bytes_reservable_limit_is_set: false,
                    pad6: [0; 7],
                    largest_free_block_bytes: 0,
                    largest_free_block_bytes_is_set: false,
                    pad7: [0; 7],
                    pool_bytes: 0,
                    pool_bytes_is_set: false,
                    pad8: [0; 7],
                    peak_pool_bytes: 0,
                    peak_pool_bytes_is_set: false,
                    pad9: [0; 7],
                };
                assert!(PJRT_Device_MemoryStats(ptr::addr_of_mut!(dv_mstat).cast()).is_null());
                assert!(dv_mstat.bytes_in_use > 0);
                assert!(dv_mstat.peak_bytes_in_use_is_set);
                crate::rust_ergonomic_api::libtpu_rs_buffer_destroy(h2d_args.out_buffer);

                let mut dv_clr = crate::pjrt_types::PJRT_Device_ClearMemoryStats_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_Device_ClearMemoryStats_Args>(),
                    extension_start: ptr::null_mut(),
                    device: dev0,
                };
                assert!(PJRT_Device_ClearMemoryStats(ptr::addr_of_mut!(dv_clr).cast()).is_null());

                let mut dv_pois = crate::pjrt_types::PJRT_Device_PoisonExecution_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_Device_PoisonExecution_Args>(),
                    extension_start: ptr::null_mut(),
                    device: (*client).device_ptrs.last().copied().unwrap_or(dev0),
                    launch_id: 1,
                    error_code: 3,
                    error_message: ptr::null(),
                    error_message_size: 0,
                    reserved0: 0,
                    reserved1: 0,
                    poisoned: false,
                    pad: [0; 7],
                };
                assert!(PJRT_Device_PoisonExecution(ptr::addr_of_mut!(dv_pois).cast()).is_null());
                assert!(dv_pois.poisoned);
                dv_pois.launch_id = -1;
                assert!(PJRT_Device_PoisonExecution(ptr::addr_of_mut!(dv_pois).cast()).is_null());

                let mut dev_attr_args = crate::pjrt_types::PJRT_Device_GetAttributes_Args {
                    struct_size: 16,
                    extension_start: ptr::null_mut(),
                    device: dev0,
                    attributes: ptr::null(),
                    num_attributes: 0,
                    reserved0: 0,
                    reserved1: 0,
                };
                drop(Box::from_raw(PJRT_Device_GetAttributes(
                    ptr::addr_of_mut!(dev_attr_args).cast(),
                )));
                dev_attr_args.struct_size =
                    size_of::<crate::pjrt_types::PJRT_Device_GetAttributes_Args>();
                assert!(
                    PJRT_Device_GetAttributes(ptr::addr_of_mut!(dev_attr_args).cast()).is_null()
                );
                assert_eq!(dev_attr_args.num_attributes, 3);

                let mut ate_create = crate::pjrt_types::PJRT_Device_CreateAsyncTrackingEvent_Args {
                    struct_size: size_of::<
                        crate::pjrt_types::PJRT_Device_CreateAsyncTrackingEvent_Args,
                    >(),
                    extension_start: ptr::null_mut(),
                    device: dev0,
                    description: c"trace".as_ptr(),
                    description_size: 5,
                    event: ptr::null_mut(),
                };
                assert!(
                    PJRT_Device_CreateAsyncTrackingEvent(ptr::addr_of_mut!(ate_create).cast())
                        .is_null()
                );
                let mut ate_destroy = crate::pjrt_types::PJRT_AsyncTrackingEvent_Destroy_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_AsyncTrackingEvent_Destroy_Args>(
                    ),
                    extension_start: ptr::null_mut(),
                    event: ate_create.event,
                };
                assert!(
                    PJRT_AsyncTrackingEvent_Destroy(ptr::addr_of_mut!(ate_destroy).cast())
                        .is_null()
                );
            }

            for &mem0 in (*client).memory_ptrs.first() {
                assert_ne!(format!("{:?}", *mem0), "");
                let mut mm_id = crate::pjrt_types::PJRT_Memory_Id_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_Memory_Id_Args>(),
                    extension_start: ptr::null_mut(),
                    memory: mem0,
                    id: -1,
                    pad: 0,
                };
                assert!(PJRT_Memory_Id(ptr::addr_of_mut!(mm_id).cast()).is_null());
                assert_eq!(mm_id.id, 0);

                let mut mm_kind = crate::pjrt_types::PJRT_Memory_Kind_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_Memory_Kind_Args>(),
                    extension_start: ptr::null_mut(),
                    memory: mem0,
                    kind: ptr::null(),
                    kind_size: 0,
                };
                assert!(PJRT_Memory_Kind(ptr::addr_of_mut!(mm_kind).cast()).is_null());
                assert!(mm_kind.kind_size > 0);

                let mut mm_dbg = crate::pjrt_types::PJRT_Memory_DebugString_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_Memory_DebugString_Args>(),
                    extension_start: ptr::null_mut(),
                    memory: mem0,
                    debug_string: ptr::null(),
                    debug_string_size: 0,
                };
                assert!(PJRT_Memory_DebugString(ptr::addr_of_mut!(mm_dbg).cast()).is_null());
                assert!(mm_dbg.debug_string_size > 0);

                let mut mm_str = crate::pjrt_types::PJRT_Memory_ToString_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_Memory_ToString_Args>(),
                    extension_start: ptr::null_mut(),
                    memory: mem0,
                    to_string: ptr::null(),
                    to_string_size: 0,
                };
                assert!(PJRT_Memory_ToString(ptr::addr_of_mut!(mm_str).cast()).is_null());
                assert!(mm_str.to_string_size > 0);

                let mut mm_devs = crate::pjrt_types::PJRT_Memory_AddressableByDevices_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_Memory_AddressableByDevices_Args>(
                    ),
                    extension_start: ptr::null_mut(),
                    memory: mem0,
                    devices: ptr::null(),
                    num_devices: 0,
                };
                assert!(
                    PJRT_Memory_AddressableByDevices(ptr::addr_of_mut!(mm_devs).cast()).is_null()
                );

                let mut mm_kid = crate::pjrt_types::PJRT_Memory_Kind_Id_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_Memory_Kind_Id_Args>(),
                    extension_start: ptr::null_mut(),
                    memory: mem0,
                    kind_id: -1,
                    pad: 0,
                };
                assert!(PJRT_Memory_Kind_Id(ptr::addr_of_mut!(mm_kid).cast()).is_null());
                assert!(mm_kid.kind_id >= 0);
            }

            let mut destroy_client_args = PJRT_Client_Destroy_Args {
                struct_size: size_of::<PJRT_Client_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                client,
            };
            assert!(PJRT_Client_Destroy(ptr::addr_of_mut!(destroy_client_args).cast()).is_null());
        }
    }
}

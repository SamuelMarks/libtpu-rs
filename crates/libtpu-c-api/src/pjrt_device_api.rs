//! PJRT device, device description, and memory space C API entrypoints (`PJRT_Device_*`, `PJRT_DeviceDescription_*`, `PJRT_Memory_*`).

use super::*;

/// Populates [`crate::pjrt_types::PJRT_DeviceDescription_Id_Args`].
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

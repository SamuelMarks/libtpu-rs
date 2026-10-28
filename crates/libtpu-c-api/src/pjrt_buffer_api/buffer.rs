//! `PJRT_Buffer_*` C API entrypoints for buffer metadata, host/device/memory copies, and lifecycle.

use super::*;

/// Strongly-typed handler for [`PJRT_Buffer_ToHostBuffer`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Buffer_ToHostBuffer_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Buffer_ToHostBuffer_Args`].
unsafe fn handle_buffer_to_host_buffer(
    args: &mut crate::pjrt_types::PJRT_Buffer_ToHostBuffer_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `args.src` is null or a valid `PJRT_Buffer*`.
    if let Some(buf) = unsafe { args.src.as_ref() } {
        let bytes = buf.inner.to_host_bytes()?;
        if args.dst.is_null() {
            args.dst_size = bytes.len();
        } else {
            let copy_len = args.dst_size.min(bytes.len());
            // SAFETY: Copying `copy_len` bytes to caller's `dst`.
            unsafe {
                ptr::copy_nonoverlapping(bytes.as_ptr(), args.dst.cast::<u8>(), copy_len);
            }
            args.event = ptr::null_mut();
        }
    }
    Ok(())
}

/// Copies a `PJRT_Buffer` from device HBM to host memory (`PJRT_Buffer_ToHostBuffer`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_ToHostBuffer_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_ToHostBuffer_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Buffer_ToHostBuffer(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Buffer_ToHostBuffer_Args>(
            raw_args,
            handle_buffer_to_host_buffer,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Buffer_Dimensions`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Buffer_Dimensions_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Buffer_Dimensions_Args`].
unsafe fn handle_buffer_dimensions(
    args: &mut crate::pjrt_types::PJRT_Buffer_Dimensions_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `args.buffer` is null or a valid `PJRT_Buffer*`.
    if let Some(buf) = unsafe { args.buffer.as_ref() } {
        args.dims = buf.inner.shape.dims.as_ptr();
        args.num_dims = buf.inner.shape.dims.len();
    }
    Ok(())
}

/// Queries the static dimensions of a `PJRT_Buffer` (`PJRT_Buffer_Dimensions`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_Dimensions_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_Dimensions_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Buffer_Dimensions(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Buffer_Dimensions_Args>(
            raw_args,
            handle_buffer_dimensions,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Buffer_ElementType`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Buffer_ElementType_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Buffer_ElementType_Args`].
unsafe fn handle_buffer_element_type(
    args: &mut crate::pjrt_types::PJRT_Buffer_ElementType_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `args.buffer` is null or a valid `PJRT_Buffer*`.
    if let Some(buf) = unsafe { args.buffer.as_ref() } {
        args.element_type = buf.inner.shape.element_type.as_raw();
    }
    Ok(())
}

/// Queries the element type of a `PJRT_Buffer` (`PJRT_Buffer_ElementType`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_ElementType_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_ElementType_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Buffer_ElementType(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Buffer_ElementType_Args>(
            raw_args,
            handle_buffer_element_type,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Buffer_OnDeviceSizeInBytes`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Buffer_OnDeviceSizeInBytes_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Buffer_OnDeviceSizeInBytes_Args`].
unsafe fn handle_buffer_on_device_size_in_bytes(
    args: &mut crate::pjrt_types::PJRT_Buffer_OnDeviceSizeInBytes_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `args.buffer` is null or a valid `PJRT_Buffer*`.
    if let Some(buf) = unsafe { args.buffer.as_ref() } {
        args.on_device_size_in_bytes = buf.inner.shape.on_device_byte_size().0;
    }
    Ok(())
}

/// Queries the on-device byte size of a `PJRT_Buffer` (`PJRT_Buffer_OnDeviceSizeInBytes`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_OnDeviceSizeInBytes_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_OnDeviceSizeInBytes_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Buffer_OnDeviceSizeInBytes(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Buffer_OnDeviceSizeInBytes_Args>(
            raw_args,
            handle_buffer_on_device_size_in_bytes,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Buffer_Destroy`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Buffer_Destroy_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Buffer_Destroy_Args`].
unsafe fn handle_buffer_destroy(
    args: &mut crate::pjrt_types::PJRT_Buffer_Destroy_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `args.buffer` is a heap-allocated `PJRT_Buffer*` or null.
    unsafe {
        libtpu_rs_buffer_destroy(args.buffer);
    }
    Ok(())
}

/// Destroys a `PJRT_Buffer` (`PJRT_Buffer_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_Destroy_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Buffer_Destroy(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Buffer_Destroy_Args>(
            raw_args,
            handle_buffer_destroy,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Buffer_UnpaddedDimensions_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Buffer_UnpaddedDimensions_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Buffer_UnpaddedDimensions_Args`].
unsafe fn handle_buffer_unpadded_dimensions(
    args: &mut crate::pjrt_types::PJRT_Buffer_UnpaddedDimensions_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `buffer` is null or a valid `PJRT_Buffer*`.
    let (ptr, len) = unsafe { args.buffer.as_ref() }.map_or((ptr::null(), 0), |b| {
        (b.inner.shape.dims.as_ptr(), b.inner.shape.dims.len())
    });
    args.unpadded_dims = ptr;
    args.num_dims = len;
    Ok(())
}

/// Queries the unpadded dimensions of a `PJRT_Buffer` (`PJRT_Buffer_UnpaddedDimensions`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_UnpaddedDimensions_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_UnpaddedDimensions_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Buffer_UnpaddedDimensions(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Buffer_UnpaddedDimensions_Args>(
            raw_args,
            handle_buffer_unpadded_dimensions,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Buffer_DynamicDimensionIndices_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Buffer_DynamicDimensionIndices_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Buffer_DynamicDimensionIndices_Args`].
unsafe fn handle_buffer_dynamic_dimension_indices(
    args: &mut crate::pjrt_types::PJRT_Buffer_DynamicDimensionIndices_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `args.buffer` is null or a valid `PJRT_Buffer*`.
    let (ptr, len) = unsafe { args.buffer.as_ref() }.map_or((ptr::null(), 0), |b| {
        (b.dynamic_dim_indices.as_ptr(), b.dynamic_dim_indices.len())
    });
    args.dynamic_dimension_indices = ptr;
    args.num_dynamic_dimensions = len;
    Ok(())
}

/// Queries dynamic dimension indices of a `PJRT_Buffer` (`PJRT_Buffer_DynamicDimensionIndices`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_DynamicDimensionIndices_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_DynamicDimensionIndices_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Buffer_DynamicDimensionIndices(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Buffer_DynamicDimensionIndices_Args>(
            raw_args,
            handle_buffer_dynamic_dimension_indices,
        )
    }
}

/// Default static tile dimensions (`[8, 128]`) for `PJRT_Buffer_GetMemoryLayout`.
static DEFAULT_TILE_DIMS: [i64; 2] = [8, 128];

/// Default static tile dimension count (`[2]`) for `PJRT_Buffer_GetMemoryLayout`.
static DEFAULT_TILE_DIM_SIZES: [usize; 1] = [2];

/// Populates [`crate::pjrt_types::PJRT_Buffer_GetMemoryLayout_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Buffer_GetMemoryLayout_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Buffer_GetMemoryLayout_Args`].
unsafe fn handle_buffer_get_memory_layout(
    args: &mut crate::pjrt_types::PJRT_Buffer_GetMemoryLayout_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `buffer` is null or a valid `PJRT_Buffer*`.
    let (m2m_ptr, m2m_len, tile_dims_ptr, tile_sizes_ptr, num_tiles) =
        unsafe { args.buffer.as_ref() }.map_or(
            (
                ptr::null(),
                0,
                DEFAULT_TILE_DIMS.as_ptr(),
                DEFAULT_TILE_DIM_SIZES.as_ptr(),
                1_usize,
            ),
            |b| match &b.inner.shape.layout {
                libtpu_core::types::MemoryLayout::Tiled { minor_to_major, .. } => (
                    minor_to_major.as_ptr(),
                    minor_to_major.len(),
                    b.tile_dims_flat.as_ptr(),
                    b.tile_dim_sizes.as_ptr(),
                    b.tile_dim_sizes.len(),
                ),
                libtpu_core::types::MemoryLayout::Strides { byte_strides } => (
                    byte_strides.as_ptr(),
                    byte_strides.len(),
                    b.tile_dims_flat.as_ptr(),
                    b.tile_dim_sizes.as_ptr(),
                    b.tile_dim_sizes.len(),
                ),
            },
        );
    args.layout_words = [
        80,
        0,
        m2m_ptr as usize,
        m2m_len,
        tile_dims_ptr as usize,
        tile_sizes_ptr as usize,
        num_tiles,
        0,
        0,
        0,
    ];
    Ok(())
}

/// Queries the tiled memory layout of a `PJRT_Buffer` (`PJRT_Buffer_GetMemoryLayout`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_GetMemoryLayout_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_GetMemoryLayout_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Buffer_GetMemoryLayout(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Buffer_GetMemoryLayout_Args>(
            raw_args,
            handle_buffer_get_memory_layout,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Buffer_Device_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Buffer_Device_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Buffer_Device_Args`].
unsafe fn handle_buffer_device(
    args: &mut crate::pjrt_types::PJRT_Buffer_Device_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `buffer` is null or a valid `PJRT_Buffer*`.
    args.device = unsafe { args.buffer.as_ref() }.map_or(ptr::null_mut(), |b| b.device_ptr);
    Ok(())
}

/// Queries the owning `PJRT_Device` of a `PJRT_Buffer` (`PJRT_Buffer_Device`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_Device_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_Device_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Buffer_Device(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Buffer_Device_Args>(
            raw_args,
            handle_buffer_device,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Buffer_Memory_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Buffer_Memory_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Buffer_Memory_Args`].
unsafe fn handle_buffer_memory(
    args: &mut crate::pjrt_types::PJRT_Buffer_Memory_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `buffer` is null or a valid `PJRT_Buffer*`.
    args.memory = unsafe { args.buffer.as_ref() }.map_or(ptr::null_mut(), |b| b.memory_ptr);
    Ok(())
}

/// Queries the owning `PJRT_Memory` of a `PJRT_Buffer` (`PJRT_Buffer_Memory`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_Memory_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_Memory_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Buffer_Memory(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Buffer_Memory_Args>(
            raw_args,
            handle_buffer_memory,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Buffer_Delete_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Buffer_Delete_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Buffer_Delete_Args`].
unsafe fn handle_buffer_delete(
    args: &mut crate::pjrt_types::PJRT_Buffer_Delete_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `buffer` is null or a valid `PJRT_Buffer*`.
    if let Some(b) = unsafe { args.buffer.as_ref() } {
        let _ = b.inner.delete();
    }
    Ok(())
}

/// Deletes the underlying storage of a `PJRT_Buffer` (`PJRT_Buffer_Delete`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_Delete_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_Delete_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Buffer_Delete(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Buffer_Delete_Args>(
            raw_args,
            handle_buffer_delete,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Buffer_IsDeleted_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Buffer_IsDeleted_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Buffer_IsDeleted_Args`].
unsafe fn handle_buffer_is_deleted(
    args: &mut crate::pjrt_types::PJRT_Buffer_IsDeleted_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `buffer` is null or a valid `PJRT_Buffer*`.
    let buf_opt = unsafe { args.buffer.as_ref() };
    args.is_deleted = buf_opt.is_some_and(|b| b.inner.is_deleted());
    Ok(())
}

/// Checks whether a `PJRT_Buffer` is deleted (`PJRT_Buffer_IsDeleted`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_IsDeleted_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_IsDeleted_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Buffer_IsDeleted(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Buffer_IsDeleted_Args>(
            raw_args,
            handle_buffer_is_deleted,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Buffer_CopyToDevice_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Buffer_CopyToDevice_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Buffer_CopyToDevice_Args`].
unsafe fn handle_buffer_copy_to_device(
    args: &mut crate::pjrt_types::PJRT_Buffer_CopyToDevice_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `buffer` and `dst_device` are null or valid pointers.
    let (buf_opt, dev_opt) = unsafe { (args.buffer.as_ref(), args.dst_device.as_ref()) };
    args.dst_buffer = match (buf_opt, dev_opt) {
        (Some(b), Some(d)) => b
            .inner
            .copy_to_device(d.inner.description.spec.device_id, b.inner.memory_id)
            .map_or(ptr::null_mut(), |copied| {
                Box::into_raw(Box::new(PJRT_Buffer::new(
                    copied,
                    args.dst_device,
                    b.memory_ptr,
                )))
            }),
        _ => ptr::null_mut(),
    };
    Ok(())
}

/// Copies a `PJRT_Buffer` to another `PJRT_Device` (`PJRT_Buffer_CopyToDevice`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_CopyToDevice_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_CopyToDevice_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Buffer_CopyToDevice(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Buffer_CopyToDevice_Args>(
            raw_args,
            handle_buffer_copy_to_device,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Buffer_IsOnCpu_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Buffer_IsOnCpu_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Buffer_IsOnCpu_Args`].
unsafe fn handle_buffer_is_on_cpu(
    args: &mut crate::pjrt_types::PJRT_Buffer_IsOnCpu_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `buffer` is null or a valid `PJRT_Buffer*`.
    args.is_on_cpu = unsafe { args.buffer.as_ref() }.is_some_and(|b| b.inner.memory_id.0 % 3 == 2);
    Ok(())
}

/// Checks whether a `PJRT_Buffer` resides in CPU memory (`PJRT_Buffer_IsOnCpu`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_IsOnCpu_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_IsOnCpu_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Buffer_IsOnCpu(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Buffer_IsOnCpu_Args>(
            raw_args,
            handle_buffer_is_on_cpu,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Buffer_ReadyEvent_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Buffer_ReadyEvent_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Buffer_ReadyEvent_Args`].
unsafe fn handle_buffer_ready_event(
    args: &mut crate::pjrt_types::PJRT_Buffer_ReadyEvent_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `buffer` is null or a valid `PJRT_Buffer*`.
    let inner_event = unsafe { args.buffer.as_ref() }
        .map_or_else(libtpu_core::pjrt::PjrtCoreEvent::ready_ok, |b| {
            b.inner.ready_event()
        });
    args.event = Box::into_raw(Box::new(crate::pjrt_types::PJRT_Event {
        inner: inner_event,
    }));
    Ok(())
}

/// Queries the readiness event of a `PJRT_Buffer` (`PJRT_Buffer_ReadyEvent`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_ReadyEvent_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_ReadyEvent_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Buffer_ReadyEvent(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Buffer_ReadyEvent_Args>(
            raw_args,
            handle_buffer_ready_event,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Buffer_UnsafePointer_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Buffer_UnsafePointer_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Buffer_UnsafePointer_Args`].
unsafe fn handle_buffer_unsafe_pointer(
    args: &mut crate::pjrt_types::PJRT_Buffer_UnsafePointer_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `buffer` is null or a valid `PJRT_Buffer*`.
    args.buffer_pointer = unsafe { args.buffer.as_ref() }.map_or(0x1000, |b| {
        usize::try_from(b.inner.address.0).unwrap_or(0x1000)
    });
    Ok(())
}

/// Queries the raw device address of a `PJRT_Buffer` (`PJRT_Buffer_UnsafePointer`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_UnsafePointer_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_UnsafePointer_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Buffer_UnsafePointer(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Buffer_UnsafePointer_Args>(
            raw_args,
            handle_buffer_unsafe_pointer,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Buffer_IncreaseExternalReferenceCount_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Buffer_IncreaseExternalReferenceCount_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Buffer_IncreaseExternalReferenceCount_Args`].
unsafe fn handle_buffer_inc_ref(
    args: &mut crate::pjrt_types::PJRT_Buffer_IncreaseExternalReferenceCount_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `buffer` is null or a valid `PJRT_Buffer*`.
    let _ = unsafe { args.buffer.as_ref() }.map(|b| b.inner.inc_external_ref());
    Ok(())
}

/// Increments external reference count on a `PJRT_Buffer` (`PJRT_Buffer_IncreaseExternalReferenceCount`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_IncreaseExternalReferenceCount_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_IncreaseExternalReferenceCount_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Buffer_IncreaseExternalReferenceCount(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Buffer_IncreaseExternalReferenceCount_Args>(
            raw_args,
            handle_buffer_inc_ref,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Buffer_DecreaseExternalReferenceCount_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Buffer_DecreaseExternalReferenceCount_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Buffer_DecreaseExternalReferenceCount_Args`].
unsafe fn handle_buffer_dec_ref(
    args: &mut crate::pjrt_types::PJRT_Buffer_DecreaseExternalReferenceCount_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `buffer` is null or a valid `PJRT_Buffer*`.
    let _ = unsafe { args.buffer.as_ref() }.map(|b| b.inner.dec_external_ref());
    Ok(())
}

/// Decrements external reference count on a `PJRT_Buffer` (`PJRT_Buffer_DecreaseExternalReferenceCount`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_DecreaseExternalReferenceCount_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_DecreaseExternalReferenceCount_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Buffer_DecreaseExternalReferenceCount(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Buffer_DecreaseExternalReferenceCount_Args>(
            raw_args,
            handle_buffer_dec_ref,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args`].
unsafe fn handle_buffer_opaque_ptr(
    args: &mut crate::pjrt_types::PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `buffer` is null or a valid `PJRT_Buffer*`.
    args.device_memory_ptr = unsafe { args.buffer.as_ref() }
        .map_or(0x1000 as *mut std::ffi::c_void, |b| {
            usize::try_from(b.inner.address.0).unwrap_or(0x1000) as *mut std::ffi::c_void
        });
    Ok(())
}

/// Queries the opaque device memory data pointer (`PJRT_Buffer_OpaqueDeviceMemoryDataPointer`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Buffer_OpaqueDeviceMemoryDataPointer(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args>(
            raw_args,
            handle_buffer_opaque_ptr,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Buffer_CopyToMemory_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Buffer_CopyToMemory_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Buffer_CopyToMemory_Args`].
unsafe fn handle_buffer_copy_to_memory(
    args: &mut crate::pjrt_types::PJRT_Buffer_CopyToMemory_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `buffer` and `dst_memory` are null or valid pointers.
    let (buf_opt, mem_opt) = unsafe { (args.buffer.as_ref(), args.dst_memory.as_ref()) };
    args.dst_buffer = match (buf_opt, mem_opt) {
        (Some(b), Some(m)) => b
            .inner
            .copy_to_device(b.inner.device_id, m.inner.id)
            .map_or(ptr::null_mut(), |copied| {
                Box::into_raw(Box::new(PJRT_Buffer::new(
                    copied,
                    b.device_ptr,
                    args.dst_memory,
                )))
            }),
        _ => ptr::null_mut(),
    };
    Ok(())
}

/// Copies a `PJRT_Buffer` to another `PJRT_Memory` space (`PJRT_Buffer_CopyToMemory`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_CopyToMemory_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_CopyToMemory_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Buffer_CopyToMemory(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Buffer_CopyToMemory_Args>(
            raw_args,
            handle_buffer_copy_to_memory,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Buffer_CopyRawToHost_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Buffer_CopyRawToHost_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Buffer_CopyRawToHost_Args`].
unsafe fn handle_buffer_copy_raw_to_host(
    args: &mut crate::pjrt_types::PJRT_Buffer_CopyRawToHost_Args,
) -> Result<(), LibTpuError> {
    if !any_null(&[args.buffer as usize, args.dst as usize]) {
        // SAFETY: `args.buffer` is non-null.
        let b = unsafe { &*args.buffer };
        let bytes = b.inner.to_host_bytes().unwrap_or_default();
        let off = usize::try_from(args.offset.max(0)).unwrap_or(0);
        let req = usize::try_from(args.transfer_size.max(0)).unwrap_or(0);
        let slice = bytes.get(off..).unwrap_or(&[]);
        let n = req.min(slice.len());
        // SAFETY: Caller's `dst` has capacity for at least `n` bytes.
        unsafe {
            ptr::copy_nonoverlapping(slice.as_ptr(), args.dst.cast::<u8>(), n);
        }
    }
    args.event = Box::into_raw(Box::new(crate::pjrt_types::PJRT_Event {
        inner: libtpu_core::pjrt::PjrtCoreEvent::ready_ok(),
    }));
    Ok(())
}

/// Copies a raw byte slice of a `PJRT_Buffer` to host memory (`PJRT_Buffer_CopyRawToHost`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_CopyRawToHost_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_CopyRawToHost_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Buffer_CopyRawToHost(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Buffer_CopyRawToHost_Args>(
            raw_args,
            handle_buffer_copy_raw_to_host,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Buffer_CopyRawToHostFuture_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Buffer_CopyRawToHostFuture_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Buffer_CopyRawToHostFuture_Args`].
unsafe fn handle_buffer_copy_raw_to_host_future(
    args: &mut crate::pjrt_types::PJRT_Buffer_CopyRawToHostFuture_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `buffer` is null or a valid `PJRT_Buffer*`.
    let _ = unsafe { args.buffer.as_ref() }.map(|b| b.inner.shape.unpadded_byte_size().0);
    args.event = Box::into_raw(Box::new(crate::pjrt_types::PJRT_Event {
        inner: libtpu_core::pjrt::PjrtCoreEvent::ready_ok(),
    }));
    Ok(())
}

/// Copies raw buffer bytes to host with a future readiness callback (`PJRT_Buffer_CopyRawToHostFuture`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_CopyRawToHostFuture_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_CopyRawToHostFuture_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Buffer_CopyRawToHostFuture(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Buffer_CopyRawToHostFuture_Args>(
            raw_args,
            handle_buffer_copy_raw_to_host_future,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_Buffer_DonateWithControlDependency_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Buffer_DonateWithControlDependency_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Buffer_DonateWithControlDependency_Args`].
unsafe fn handle_buffer_donate_with_control_dependency(
    args: &mut crate::pjrt_types::PJRT_Buffer_DonateWithControlDependency_Args,
) -> Result<(), LibTpuError> {
    args.out_buffer = args.buffer;
    Ok(())
}

/// Donates a buffer with an explicit control dependency (`PJRT_Buffer_DonateWithControlDependency`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_DonateWithControlDependency_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_DonateWithControlDependency_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Buffer_DonateWithControlDependency(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Buffer_DonateWithControlDependency_Args>(
            raw_args,
            handle_buffer_donate_with_control_dependency,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Buffer_Bitcast`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Buffer_Bitcast_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Buffer_Bitcast_Args`].
unsafe fn handle_buffer_bitcast(
    args: &mut crate::pjrt_types::PJRT_Buffer_Bitcast_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `args.buffer` is null or a valid `PJRT_Buffer*`.
    if let Some(buf_ref) = unsafe { args.buffer.as_ref() } {
        let elem_type =
            ElementType::from_raw(args.element_type).unwrap_or(buf_ref.inner.shape.element_type);
        let dims = if !args.dims.is_null() {
            // SAFETY: `args.dims` points to `args.num_dims` `i64` dimensions.
            unsafe { std::slice::from_raw_parts(args.dims, args.num_dims).to_vec() }
        } else {
            buf_ref.inner.shape.dims.clone()
        };
        if let Ok(bitcast_core) = buf_ref.inner.bitcast(elem_type, dims) {
            args.out_buffer = Box::into_raw(Box::new(PJRT_Buffer::new(
                bitcast_core,
                buf_ref.device_ptr,
                buf_ref.memory_ptr,
            )));
        }
    }
    Ok(())
}

/// Zero-copy bitcasts a `PJRT_Buffer` to a new element type and shape (`PJRT_Buffer_Bitcast`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_Buffer_Bitcast_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_Buffer_Bitcast_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Buffer_Bitcast(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_Buffer_Bitcast_Args>(
            raw_args,
            handle_buffer_bitcast,
        )
    }
}

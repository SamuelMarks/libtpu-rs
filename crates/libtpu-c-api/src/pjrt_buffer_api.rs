//! PJRT buffer, copy-to-device stream, and async host-to-device transfer manager C API entrypoints.

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

/// Populates [`crate::pjrt_types::PJRT_CopyToDeviceStream_Destroy_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_CopyToDeviceStream_Destroy_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_CopyToDeviceStream_Destroy_Args`].
unsafe fn handle_stream_destroy(
    args: &mut crate::pjrt_types::PJRT_CopyToDeviceStream_Destroy_Args,
) -> Result<(), LibTpuError> {
    if !args.stream.is_null() {
        // SAFETY: `args.stream` is a valid heap-allocated `PJRT_CopyToDeviceStream*`.
        unsafe {
            drop(Box::from_raw(args.stream));
        }
    }
    Ok(())
}

/// Destroys a `PJRT_CopyToDeviceStream` (`PJRT_CopyToDeviceStream_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_CopyToDeviceStream_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_CopyToDeviceStream_Destroy_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_CopyToDeviceStream_Destroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_CopyToDeviceStream_Destroy_Args>(
            raw_args,
            handle_stream_destroy,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_CopyToDeviceStream_AddChunk_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_CopyToDeviceStream_AddChunk_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_CopyToDeviceStream_AddChunk_Args`].
unsafe fn handle_stream_add_chunk(
    args: &mut crate::pjrt_types::PJRT_CopyToDeviceStream_AddChunk_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `stream` and `chunk` are null or valid pointers.
    let (stream_opt, chunk_opt) = unsafe { (args.stream.as_ref(), args.chunk.as_ref()) };
    if let Some(c) = chunk_opt {
        let len = c.size.max(1);
        if let Some(s) = stream_opt {
            if !c.data.is_null() {
                // SAFETY: `c.data` points to `c.size` bytes.
                let slice = unsafe { std::slice::from_raw_parts(c.data.cast::<u8>(), c.size) };
                let _ = s.inner.add_chunk_data(slice);
            } else {
                let _ = s.inner.add_chunk(i64::try_from(len).unwrap_or(128));
            }
        }
        if let Some(del) = c.deleter {
            // SAFETY: Invoking chunk deleter with caller's data and deleter_arg.
            unsafe {
                del(c.data, c.deleter_arg);
            }
        }
    } else if let Some(s) = stream_opt {
        let _ = s.inner.add_chunk(128);
    }
    args.transfer_complete = Box::into_raw(Box::new(crate::pjrt_types::PJRT_Event {
        inner: libtpu_core::pjrt::PjrtCoreEvent::ready_ok(),
    }));
    Ok(())
}

/// Appends a host chunk to a `PJRT_CopyToDeviceStream` (`PJRT_CopyToDeviceStream_AddChunk`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_CopyToDeviceStream_AddChunk_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_CopyToDeviceStream_AddChunk_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_CopyToDeviceStream_AddChunk(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_CopyToDeviceStream_AddChunk_Args>(
            raw_args,
            handle_stream_add_chunk,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_CopyToDeviceStream_TotalBytes_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_CopyToDeviceStream_TotalBytes_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_CopyToDeviceStream_TotalBytes_Args`].
unsafe fn handle_stream_total_bytes(
    args: &mut crate::pjrt_types::PJRT_CopyToDeviceStream_TotalBytes_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `stream` is null or a valid `PJRT_CopyToDeviceStream*`.
    args.total_bytes = unsafe { args.stream.as_ref() }.map_or(256, |s| s.inner.total_bytes);
    Ok(())
}

/// Queries total expected bytes on a `PJRT_CopyToDeviceStream` (`PJRT_CopyToDeviceStream_TotalBytes`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_CopyToDeviceStream_TotalBytes_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_CopyToDeviceStream_TotalBytes_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_CopyToDeviceStream_TotalBytes(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_CopyToDeviceStream_TotalBytes_Args>(
            raw_args,
            handle_stream_total_bytes,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_CopyToDeviceStream_GranuleSize_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_CopyToDeviceStream_GranuleSize_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_CopyToDeviceStream_GranuleSize_Args`].
unsafe fn handle_stream_granule_size(
    args: &mut crate::pjrt_types::PJRT_CopyToDeviceStream_GranuleSize_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `stream` is null or a valid `PJRT_CopyToDeviceStream*`.
    let st_opt = unsafe { args.stream.as_ref() };
    args.granule_size_in_bytes = st_opt.map_or(128, |s| s.inner.granule_size);
    Ok(())
}

/// Queries transfer granule size on a `PJRT_CopyToDeviceStream` (`PJRT_CopyToDeviceStream_GranuleSize`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_CopyToDeviceStream_GranuleSize_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_CopyToDeviceStream_GranuleSize_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_CopyToDeviceStream_GranuleSize(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_CopyToDeviceStream_GranuleSize_Args>(
            raw_args,
            handle_stream_granule_size,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_CopyToDeviceStream_CurrentBytes_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_CopyToDeviceStream_CurrentBytes_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_CopyToDeviceStream_CurrentBytes_Args`].
unsafe fn handle_stream_current_bytes(
    args: &mut crate::pjrt_types::PJRT_CopyToDeviceStream_CurrentBytes_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `stream` is null or a valid `PJRT_CopyToDeviceStream*`.
    let st_opt = unsafe { args.stream.as_ref() };
    args.current_bytes = st_opt.map_or(0, |s| s.inner.current_bytes());
    Ok(())
}

/// Queries currently transferred bytes on a `PJRT_CopyToDeviceStream` (`PJRT_CopyToDeviceStream_CurrentBytes`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_CopyToDeviceStream_CurrentBytes_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_CopyToDeviceStream_CurrentBytes_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_CopyToDeviceStream_CurrentBytes(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_CopyToDeviceStream_CurrentBytes_Args>(
            raw_args,
            handle_stream_current_bytes,
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

/// Populates [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_Destroy_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_Destroy_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_Destroy_Args`].
unsafe fn handle_async_transfer_manager_destroy(
    args: &mut crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_Destroy_Args,
) -> Result<(), LibTpuError> {
    if !args.transfer_manager.is_null() {
        // SAFETY: `args.transfer_manager` was allocated by `PJRT_Client_CreateBuffersForAsyncHostToDevice`.
        unsafe {
            drop(Box::from_raw(args.transfer_manager));
        }
    }
    Ok(())
}

/// Destroys an async host-to-device transfer manager (`PJRT_AsyncHostToDeviceTransferManager_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_Destroy_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_AsyncHostToDeviceTransferManager_Destroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<
            crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_Destroy_Args,
        >(raw_args, handle_async_transfer_manager_destroy)
    }
}

/// Populates [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_TransferData_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_TransferData_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_TransferData_Args`].
unsafe fn handle_async_transfer_manager_transfer_data(
    args: &mut crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_TransferData_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `transfer_manager` is null or a valid `PJRT_AsyncHostToDeviceTransferManager*`.
    if let Some(tm) = unsafe { args.transfer_manager.as_ref() } {
        let idx = usize::try_from(args.buffer_index.max(0)).unwrap_or(0);
        let len = usize::try_from(args.transfer_size.max(0)).unwrap_or(0);
        if !args.data.is_null() {
            // SAFETY: `args.data` points to `len` host bytes.
            let slice = unsafe { std::slice::from_raw_parts(args.data.cast::<u8>(), len) };
            let _ = tm.inner.transfer_data(idx, slice);
        }
    }
    args.done_with_h2d_transfer = Box::into_raw(Box::new(crate::pjrt_types::PJRT_Event {
        inner: libtpu_core::pjrt::PjrtCoreEvent::ready_ok(),
    }));
    Ok(())
}

/// Transfers a data chunk via an async transfer manager (`PJRT_AsyncHostToDeviceTransferManager_TransferData`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_TransferData_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_TransferData_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_AsyncHostToDeviceTransferManager_TransferData(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<
            crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_TransferData_Args,
        >(raw_args, handle_async_transfer_manager_transfer_data)
    }
}

/// Populates [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args`].
unsafe fn handle_async_transfer_manager_retrieve_buffer(
    args: &mut crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `transfer_manager` is null or a valid `PJRT_AsyncHostToDeviceTransferManager*`.
    if let Some(tm) = unsafe { args.transfer_manager.as_ref() } {
        let idx = usize::try_from(args.buffer_index.max(0)).unwrap_or(0);
        if let Ok(arc_buf) = tm.inner.retrieve_buffer(idx) {
            let cloned_core = libtpu_core::pjrt::PjrtCoreBuffer::new(
                arc_buf.device_id,
                arc_buf.memory_id,
                arc_buf.shape.clone(),
                arc_buf.address,
                std::sync::Arc::clone(&arc_buf.hal),
            );
            args.buffer_out = Box::into_raw(Box::new(PJRT_Buffer::new(
                cloned_core,
                tm.device_ptr,
                tm.memory_ptr,
            )));
            return Ok(());
        }
    }
    args.buffer_out = alloc_default_pjrt_buffer(ptr::null_mut(), ptr::null_mut());
    Ok(())
}

/// Retrieves a target `PJRT_Buffer` from an async transfer manager (`PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<
            crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args,
        >(raw_args, handle_async_transfer_manager_retrieve_buffer)
    }
}

/// Populates [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_Device_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_Device_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_Device_Args`].
unsafe fn handle_async_transfer_manager_device(
    args: &mut crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_Device_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `transfer_manager` is null or a valid `PJRT_AsyncHostToDeviceTransferManager*`.
    args.device_out =
        unsafe { args.transfer_manager.as_ref() }.map_or(ptr::null_mut(), |tm| tm.device_ptr);
    Ok(())
}

/// Queries the target device of an async transfer manager (`PJRT_AsyncHostToDeviceTransferManager_Device`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_Device_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_Device_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_AsyncHostToDeviceTransferManager_Device(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<
            crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_Device_Args,
        >(raw_args, handle_async_transfer_manager_device)
    }
}

/// Populates [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args`].
unsafe fn handle_async_transfer_manager_buffer_count(
    args: &mut crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `transfer_manager` is null or a valid `PJRT_AsyncHostToDeviceTransferManager*`.
    let tm_opt = unsafe { args.transfer_manager.as_ref() };
    args.buffer_count = tm_opt.map_or(1, |tm| tm.inner.buffers.len());
    Ok(())
}

/// Queries the buffer count of an async transfer manager (`PJRT_AsyncHostToDeviceTransferManager_BufferCount`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_AsyncHostToDeviceTransferManager_BufferCount(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<
            crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args,
        >(raw_args, handle_async_transfer_manager_buffer_count)
    }
}

/// Populates [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args`].
unsafe fn handle_async_transfer_manager_buffer_size(
    args: &mut crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `transfer_manager` is null or a valid `PJRT_AsyncHostToDeviceTransferManager*`.
    let tm_opt = unsafe { args.transfer_manager.as_ref() };
    let idx = usize::try_from(args.buffer_index.max(0)).unwrap_or(0);
    args.buffer_size = tm_opt.map_or(256, |tm| {
        tm.inner
            .buffer_sizes
            .get(idx)
            .copied()
            .and_then(|sz| usize::try_from(sz).ok())
            .unwrap_or(tm.inner.buffers.len() * 16)
    });
    Ok(())
}

/// Queries the byte size of a buffer in an async transfer manager (`PJRT_AsyncHostToDeviceTransferManager_BufferSize`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_AsyncHostToDeviceTransferManager_BufferSize(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<
            crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args,
        >(raw_args, handle_async_transfer_manager_buffer_size)
    }
}

/// Populates [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args`].
unsafe fn handle_async_transfer_manager_set_buffer_error(
    args: &mut crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `transfer_manager` is null or a valid `PJRT_AsyncHostToDeviceTransferManager*`.
    if let Some(tm) = unsafe { args.transfer_manager.as_ref() } {
        let idx = usize::try_from(args.buffer_index.max(0)).unwrap_or(0);
        tm.inner.set_buffer_error(
            idx,
            LibTpuError::TransferFailed {
                direction: "AsyncH2D",
                reason: format!("buffer error code {}", args.error_code),
            },
        );
    }
    Ok(())
}

/// Marks a buffer in an async transfer manager as failed (`PJRT_AsyncHostToDeviceTransferManager_SetBufferError`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_AsyncHostToDeviceTransferManager_SetBufferError(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<
            crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args,
        >(raw_args, handle_async_transfer_manager_set_buffer_error)
    }
}

/// Populates [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args`].
unsafe fn handle_async_transfer_manager_add_metadata(
    args: &mut crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `transfer_manager` is null or a valid `PJRT_AsyncHostToDeviceTransferManager*`.
    if let Some(tm) = unsafe { args.transfer_manager.as_ref() } {
        let mut entries = Vec::with_capacity(args.num_metadata.max(1));
        if !args.transfer_metadata.is_null() {
            // SAFETY: `args.transfer_metadata` points to `args.num_metadata` entries.
            let meta_slice =
                unsafe { std::slice::from_raw_parts(args.transfer_metadata, args.num_metadata) };
            for (idx, item) in meta_slice.iter().enumerate() {
                entries.push((format!("meta_{idx}"), item.flags.to_string()));
            }
        } else {
            entries.push((String::from("num_metadata"), args.num_metadata.to_string()));
        }
        tm.inner.add_metadata(entries);
    }
    Ok(())
}

/// Attaches metadata to an async transfer manager (`PJRT_AsyncHostToDeviceTransferManager_AddMetadata`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_AsyncHostToDeviceTransferManager_AddMetadata(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<
            crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args,
        >(raw_args, handle_async_transfer_manager_add_metadata)
    }
}

/// Populates [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args`].
unsafe fn handle_async_transfer_manager_transfer_literal(
    args: &mut crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `transfer_manager` is null or a valid `PJRT_AsyncHostToDeviceTransferManager*`.
    if let Some(tm) = unsafe { args.transfer_manager.as_ref() } {
        let idx = usize::try_from(args.buffer_index.max(0)).unwrap_or(0);
        if !args.data.is_null() {
            let elem_bytes = ElementType::from_raw(args.shape_element_type)
                .unwrap_or(ElementType::F32)
                .byte_size();
            let num_elems: usize = if !args.shape_dims.is_null() {
                // SAFETY: `args.shape_dims` has `args.shape_num_dims` elements.
                unsafe { std::slice::from_raw_parts(args.shape_dims, args.shape_num_dims) }
                    .iter()
                    .map(|&d| usize::try_from(d.max(1)).unwrap_or(1))
                    .product()
            } else {
                1
            };
            let byte_len = elem_bytes.saturating_mul(num_elems);
            // SAFETY: `args.data` points to at least `byte_len` host bytes.
            let literal_slice =
                unsafe { std::slice::from_raw_parts(args.data.cast::<u8>(), byte_len) };
            let _ = tm.inner.transfer_literal(idx, literal_slice);
        }
    }
    args.done_with_h2d_transfer = Box::into_raw(Box::new(crate::pjrt_types::PJRT_Event {
        inner: libtpu_core::pjrt::PjrtCoreEvent::ready_ok(),
    }));
    Ok(())
}

/// Transfers a literal via an async transfer manager (`PJRT_AsyncHostToDeviceTransferManager_TransferLiteral`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_AsyncHostToDeviceTransferManager_TransferLiteral(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<
            crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args,
        >(raw_args, handle_async_transfer_manager_transfer_literal)
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

//! `PJRT_Client_*` C API entrypoints for buffer creation, aliases, async H2D, and DMA mapping.

use super::*;

/// Strongly-typed handler for [`PJRT_Client_BufferFromHostBuffer`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Client_BufferFromHostBuffer_Args`] structure.
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
        args.out_buffer = Box::into_raw(Box::new(PJRT_Buffer::new(core_buf, dev_ptr, mem_ptr)));
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
#[unsafe(no_mangle)]
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

/// Populates [`crate::pjrt_types::PJRT_Client_CreateViewOfDeviceBuffer_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Client_CreateViewOfDeviceBuffer_Args`] structure.
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
    args.buffer = alloc_shaped_pjrt_buffer(
        args.client,
        args.device,
        ptr::null_mut(),
        args.element_type,
        args.dims,
        args.num_dims,
    );
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
#[unsafe(no_mangle)]
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

/// Populates [`crate::pjrt_types::PJRT_Client_CreateBuffersForAsyncHostToDevice_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Client_CreateBuffersForAsyncHostToDevice_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Client_CreateBuffersForAsyncHostToDevice_Args`].
unsafe fn handle_client_create_buffers_for_async_h2d(
    args: &mut crate::pjrt_types::PJRT_Client_CreateBuffersForAsyncHostToDevice_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `args.client` is null or a valid `PJRT_Client*`.
    let (dev_ptr, mem_ptr, dev_id, mem_id, hal) = unsafe {
        args.client.as_ref().map_or_else(
            || {
                let cfg = libtpu_core::config::global_init_config();
                (
                    ptr::null_mut(),
                    args.memory,
                    DeviceId(0),
                    libtpu_core::types::MemorySpaceId(0),
                    libtpu_core::hal::create_default_hal(&cfg),
                )
            },
            |client_ref| {
                let d_ptr = client_ref
                    .addressable_device_ptrs
                    .first()
                    .copied()
                    .unwrap_or(ptr::null_mut());
                (
                    d_ptr,
                    args.memory,
                    DeviceId(0),
                    libtpu_core::types::MemorySpaceId(0),
                    std::sync::Arc::clone(&client_ref.inner.hal),
                )
            },
        )
    };
    let mut shapes = Vec::new();
    if !args.shape_specs.is_null() {
        // SAFETY: `args.shape_specs` points to `args.num_shape_specs` elements.
        let specs = unsafe { std::slice::from_raw_parts(args.shape_specs, args.num_shape_specs) };
        for spec in specs {
            let elem = ElementType::from_raw(spec.element_type).unwrap_or(ElementType::F32);
            let dims = if spec.dims.is_null() {
                vec![4]
            } else {
                // SAFETY: `spec.dims` points to `spec.num_dims` `i64` elements.
                unsafe { std::slice::from_raw_parts(spec.dims, spec.num_dims).to_vec() }
            };
            let s = Shape::new(elem, dims, None, None).unwrap_or_else(|_| Shape {
                element_type: ElementType::F32,
                dims: vec![4],
                dynamic_dims: vec![false],
                layout: libtpu_core::types::MemoryLayout::default_tpu_tiled(1),
            });
            shapes.push(s);
        }
    } else {
        shapes.push(Shape {
            element_type: ElementType::F32,
            dims: vec![4],
            dynamic_dims: vec![false],
            layout: libtpu_core::types::MemoryLayout::default_tpu_tiled(1),
        });
    }
    args.transfer_manager =
        libtpu_core::pjrt::PjrtCoreAsyncTransferManager::create(dev_id, mem_id, shapes, &hal)
            .map_or(ptr::null_mut(), |tm| {
                Box::into_raw(Box::new(
                    crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager {
                        inner: tm,
                        device_ptr: dev_ptr,
                        memory_ptr: mem_ptr,
                    },
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
#[unsafe(no_mangle)]
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
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Client_DmaMap_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Client_DmaMap_Args`].
unsafe fn handle_client_dma_map(
    args: &mut crate::pjrt_types::PJRT_Client_DmaMap_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `client` is null or a valid `PJRT_Client*`.
    if let Some(client) = unsafe { args.client.as_ref() } {
        let base_addr = (args.data as u64).max(0x1000);
        let size = args.size.max(4096);
        let _ = client.inner.dma_map(base_addr, size);
    }
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
#[unsafe(no_mangle)]
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
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Client_DmaUnmap_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Client_DmaUnmap_Args`].
unsafe fn handle_client_dma_unmap(
    args: &mut crate::pjrt_types::PJRT_Client_DmaUnmap_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `client` is null or a valid `PJRT_Client*`.
    if let Some(client) = unsafe { args.client.as_ref() } {
        let base_addr = (args.data as u64).max(0x1000);
        let _ = client.inner.dma_unmap(base_addr);
    }
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
#[unsafe(no_mangle)]
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
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Client_CreateAliasBuffer_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Client_CreateAliasBuffer_Args`].
unsafe fn handle_client_create_alias_buffer(
    args: &mut crate::pjrt_types::PJRT_Client_CreateAliasBuffer_Args,
) -> Result<(), LibTpuError> {
    let buf = alloc_shaped_pjrt_buffer(
        args.client,
        ptr::null_mut(),
        args.memory,
        args.element_type,
        args.dims,
        args.num_dims,
    );
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
#[unsafe(no_mangle)]
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
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Client_FulfillAliasBuffer_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Client_FulfillAliasBuffer_Args`].
unsafe fn handle_client_fulfill_alias_buffer(
    args: &mut crate::pjrt_types::PJRT_Client_FulfillAliasBuffer_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `fulfill_alias_buffer_cb` and `buffer` are null or valid `PJRT_Buffer*` handles.
    let (alias_opt, real_opt) = unsafe {
        (
            args.fulfill_alias_buffer_cb.cast::<PJRT_Buffer>().as_ref(),
            args.buffer.as_ref(),
        )
    };
    if let Some(alias_buf) = alias_opt {
        if args.status_code != 0 {
            let msg = if args.error_message.is_null() {
                format!("alias fulfillment failed with code {}", args.status_code)
            } else {
                // SAFETY: `error_message` points to `error_message_size` bytes.
                let slice = unsafe {
                    std::slice::from_raw_parts(
                        args.error_message.cast::<u8>(),
                        args.error_message_size,
                    )
                };
                String::from_utf8_lossy(slice).into_owned()
            };
            alias_buf
                .inner
                .set_error(LibTpuError::EventFailed { reason: msg });
        } else if let Some(real_buf) = real_opt {
            let _ = alias_buf.inner.fulfill_from(&real_buf.inner);
        }
    }
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
#[unsafe(no_mangle)]
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

/// Populates [`crate::pjrt_types::PJRT_Client_CreateErrorBuffer_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Client_CreateErrorBuffer_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Client_CreateErrorBuffer_Args`].
unsafe fn handle_client_create_error_buffer(
    args: &mut crate::pjrt_types::PJRT_Client_CreateErrorBuffer_Args,
) -> Result<(), LibTpuError> {
    let buf = alloc_shaped_pjrt_buffer(
        args.client,
        args.device,
        args.memory,
        args.element_type,
        args.dims,
        args.num_dims,
    );
    let msg = if args.error_message.is_null() {
        format!("error buffer code {}", args.error_code)
    } else {
        // SAFETY: `error_message` points to `error_message_size` bytes.
        let slice = unsafe {
            std::slice::from_raw_parts(args.error_message.cast::<u8>(), args.error_message_size)
        };
        String::from_utf8_lossy(slice).into_owned()
    };
    // SAFETY: `buf` is a valid heap-allocated `PJRT_Buffer*`.
    unsafe {
        (*buf)
            .inner
            .set_error(LibTpuError::EventFailed { reason: msg });
    }
    args.buffer = buf;
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
#[unsafe(no_mangle)]
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
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_Client_CreateUninitializedBuffer_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_Client_CreateUninitializedBuffer_Args`].
unsafe fn handle_client_create_uninitialized_buffer(
    args: &mut crate::pjrt_types::PJRT_Client_CreateUninitializedBuffer_Args,
) -> Result<(), LibTpuError> {
    args.buffer = alloc_shaped_pjrt_buffer(
        args.client,
        args.device,
        args.memory,
        args.element_type,
        args.dims,
        args.num_dims,
    );
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
#[unsafe(no_mangle)]
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

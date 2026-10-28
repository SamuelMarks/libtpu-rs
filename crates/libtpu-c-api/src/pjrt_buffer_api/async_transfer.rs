//! `PJRT_AsyncHostToDeviceTransferManager_*` C API entrypoints for batched asynchronous H2D transfers.

use super::*;

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

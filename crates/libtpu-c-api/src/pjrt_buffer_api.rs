//! PJRT buffer, copy-to-device stream, and async host-to-device transfer manager C API entrypoints.

/// `PJRT_AsyncHostToDeviceTransferManager_*` C API entrypoints for batched asynchronous H2D transfers.
pub mod async_transfer;
/// `PJRT_Buffer_*` C API entrypoints for buffer metadata, host/device/memory copies, and lifecycle.
pub mod buffer;
/// `PJRT_CopyToDeviceStream_*` C API entrypoints for streaming host-to-device transfers.
pub mod stream;

use super::*;

pub use async_transfer::*;
pub use buffer::*;
pub use stream::*;

#[cfg(test)]
#[allow(
    for_loops_over_fallibles,
    clippy::cognitive_complexity,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::items_after_statements,
    clippy::float_cmp
)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::pjrt_client_api::{
        PJRT_Client_BufferFromHostBuffer, PJRT_Client_Create, PJRT_Client_CreateAliasBuffer,
        PJRT_Client_CreateErrorBuffer, PJRT_Client_CreateUninitializedBuffer, PJRT_Client_Destroy,
        PJRT_Client_LookupDevice,
    };
    use crate::pjrt_types::{
        PJRT_AsyncHostToDeviceTransferManager, PJRT_Client_BufferFromHostBuffer_Args,
        PJRT_Client_Create_Args, PJRT_Client_Destroy_Args, PJRT_Client_LookupDevice_Args,
    };
    use crate::rust_ergonomic_api::libtpu_rs_buffer_destroy;
    use libtpu_core::pjrt::PjrtCoreAsyncTransferManager;

    #[test]
    fn test_pjrt_buffer_stream_and_async_transfer_manager_api() {
        // SAFETY: Testing PJRT buffer, copy-to-device stream, and async transfer manager entrypoints.
        unsafe {
            drop(Box::from_raw(PJRT_Buffer_ToHostBuffer(ptr::null_mut())));
            drop(Box::from_raw(PJRT_Buffer_Dimensions(ptr::null_mut())));
            drop(Box::from_raw(PJRT_Buffer_ElementType(ptr::null_mut())));
            drop(Box::from_raw(PJRT_Buffer_OnDeviceSizeInBytes(
                ptr::null_mut(),
            )));
            drop(Box::from_raw(PJRT_Buffer_Destroy(ptr::null_mut())));
            drop(Box::from_raw(PJRT_Buffer_Bitcast(ptr::null_mut())));

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

            for mgr in PjrtCoreAsyncTransferManager::create(
                DeviceId(0),
                libtpu_core::types::MemorySpaceId(0),
                Vec::new(),
                &(*client).inner.hal,
            ) {
                let c_mgr = PJRT_AsyncHostToDeviceTransferManager {
                    inner: mgr,
                    device_ptr: ptr::null_mut(),
                    memory_ptr: ptr::null_mut(),
                };
                assert_ne!(format!("{c_mgr:?}"), "");
            }

            let mut look_args = PJRT_Client_LookupDevice_Args {
                struct_size: size_of::<PJRT_Client_LookupDevice_Args>(),
                extension_start: ptr::null_mut(),
                client,
                id: 0,
                pad: 0,
                device: ptr::null_mut(),
            };
            assert!(PJRT_Client_LookupDevice(ptr::addr_of_mut!(look_args).cast()).is_null());

            let host_floats = [1.0_f32, 2.0, 3.0, 4.0];
            let dims_arr = [4_i64];
            let mut h2d_args = PJRT_Client_BufferFromHostBuffer_Args {
                struct_size: size_of::<PJRT_Client_BufferFromHostBuffer_Args>(),
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
                device: look_args.device,
                memory: ptr::null_mut(),
                device_layout: ptr::null(),
                done_with_host_buffer: ptr::null_mut(),
                out_buffer: ptr::null_mut(),
            };
            assert!(PJRT_Client_BufferFromHostBuffer(ptr::addr_of_mut!(h2d_args).cast()).is_null());
            let buf_in = h2d_args.out_buffer;
            assert!(!buf_in.is_null());
            assert_ne!(format!("{:?}", *buf_in), "");

            let mut dim_args = crate::pjrt_types::PJRT_Buffer_Dimensions_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_Dimensions_Args>(),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
                dims: ptr::null(),
                num_dims: 0,
            };
            assert!(PJRT_Buffer_Dimensions(ptr::addr_of_mut!(dim_args).cast()).is_null());
            assert_eq!(dim_args.num_dims, 1);

            let mut bf_udim = crate::pjrt_types::PJRT_Buffer_UnpaddedDimensions_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_UnpaddedDimensions_Args>(),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
                unpadded_dims: ptr::null(),
                num_dims: 0,
            };
            assert!(PJRT_Buffer_UnpaddedDimensions(ptr::addr_of_mut!(bf_udim).cast()).is_null());
            assert_eq!(bf_udim.num_dims, 1);

            let mut bf_ddim = crate::pjrt_types::PJRT_Buffer_DynamicDimensionIndices_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_DynamicDimensionIndices_Args>(
                ),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
                dynamic_dimension_indices: ptr::null(),
                num_dynamic_dimensions: 0,
            };
            assert!(
                PJRT_Buffer_DynamicDimensionIndices(ptr::addr_of_mut!(bf_ddim).cast()).is_null()
            );

            let mut bf_gml = crate::pjrt_types::PJRT_Buffer_GetMemoryLayout_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_GetMemoryLayout_Args>(),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
                layout_words: [0; 10],
            };
            assert!(PJRT_Buffer_GetMemoryLayout(ptr::addr_of_mut!(bf_gml).cast()).is_null());
            assert_eq!(bf_gml.layout_words[0], 80);
            assert_eq!(bf_gml.layout_words[3], 1);
            assert_eq!(bf_gml.layout_words[6], 1);
            (*buf_in).inner.shape.layout = libtpu_core::types::MemoryLayout::Strides {
                byte_strides: vec![4],
            };
            let strided_buf = PJRT_Buffer::new(
                libtpu_core::pjrt::PjrtCoreBuffer::new(
                    (*buf_in).inner.device_id,
                    (*buf_in).inner.memory_id,
                    (*buf_in).inner.shape.clone(),
                    (*buf_in).inner.address,
                    std::sync::Arc::clone(&(*buf_in).inner.hal),
                ),
                (*buf_in).device_ptr,
                (*buf_in).memory_ptr,
            );
            assert_eq!(strided_buf.tile_dims_flat, vec![4]);
            assert!(PJRT_Buffer_GetMemoryLayout(ptr::addr_of_mut!(bf_gml).cast()).is_null());
            (*buf_in).inner.shape.layout = libtpu_core::types::MemoryLayout::default_tpu_tiled(1);

            let mut bf_dev = crate::pjrt_types::PJRT_Buffer_Device_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_Device_Args>(),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
                device: ptr::null_mut(),
            };
            assert!(PJRT_Buffer_Device(ptr::addr_of_mut!(bf_dev).cast()).is_null());
            assert!(!bf_dev.device.is_null());

            let mut bf_mem = crate::pjrt_types::PJRT_Buffer_Memory_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_Memory_Args>(),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
                memory: ptr::null_mut(),
            };
            assert!(PJRT_Buffer_Memory(ptr::addr_of_mut!(bf_mem).cast()).is_null());
            assert!(!bf_mem.memory.is_null());

            let mut bf_isdel = crate::pjrt_types::PJRT_Buffer_IsDeleted_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_IsDeleted_Args>(),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
                is_deleted: true,
                pad: [0; 7],
            };
            assert!(PJRT_Buffer_IsDeleted(ptr::addr_of_mut!(bf_isdel).cast()).is_null());
            assert!(!bf_isdel.is_deleted);

            let mut bf_cpu = crate::pjrt_types::PJRT_Buffer_IsOnCpu_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_IsOnCpu_Args>(),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
                is_on_cpu: true,
                pad: [0; 7],
            };
            assert!(PJRT_Buffer_IsOnCpu(ptr::addr_of_mut!(bf_cpu).cast()).is_null());
            assert!(!bf_cpu.is_on_cpu);

            let mut bf_rev = crate::pjrt_types::PJRT_Buffer_ReadyEvent_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_ReadyEvent_Args>(),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
                event: ptr::null_mut(),
            };
            assert!(PJRT_Buffer_ReadyEvent(ptr::addr_of_mut!(bf_rev).cast()).is_null());

            let mut bf_uptr = crate::pjrt_types::PJRT_Buffer_UnsafePointer_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_UnsafePointer_Args>(),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
                buffer_pointer: 0,
            };
            assert!(PJRT_Buffer_UnsafePointer(ptr::addr_of_mut!(bf_uptr).cast()).is_null());
            assert_ne!(bf_uptr.buffer_pointer, 0);

            let mut bf_inc = crate::pjrt_types::PJRT_Buffer_IncreaseExternalReferenceCount_Args {
                struct_size: size_of::<
                    crate::pjrt_types::PJRT_Buffer_IncreaseExternalReferenceCount_Args,
                >(),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
            };
            assert!(
                PJRT_Buffer_IncreaseExternalReferenceCount(ptr::addr_of_mut!(bf_inc).cast())
                    .is_null()
            );

            let mut bf_dec = crate::pjrt_types::PJRT_Buffer_DecreaseExternalReferenceCount_Args {
                struct_size: size_of::<
                    crate::pjrt_types::PJRT_Buffer_DecreaseExternalReferenceCount_Args,
                >(),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
            };
            assert!(
                PJRT_Buffer_DecreaseExternalReferenceCount(ptr::addr_of_mut!(bf_dec).cast())
                    .is_null()
            );

            let mut bf_optr = crate::pjrt_types::PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args {
                struct_size: size_of::<
                    crate::pjrt_types::PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args,
                >(),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
                device_memory_ptr: ptr::null_mut(),
            };
            assert!(
                PJRT_Buffer_OpaqueDeviceMemoryDataPointer(ptr::addr_of_mut!(bf_optr).cast())
                    .is_null()
            );
            assert!(!bf_optr.device_memory_ptr.is_null());

            let mut bf_crhf = crate::pjrt_types::PJRT_Buffer_CopyRawToHostFuture_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_CopyRawToHostFuture_Args>(),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
                offset: 0,
                transfer_size: 16,
                event: ptr::null_mut(),
                callback_data: ptr::null_mut(),
                future_ready_callback: ptr::null(),
            };
            assert!(PJRT_Buffer_CopyRawToHostFuture(ptr::addr_of_mut!(bf_crhf).cast()).is_null());

            let mut bf_don = crate::pjrt_types::PJRT_Buffer_DonateWithControlDependency_Args {
                struct_size: size_of::<
                    crate::pjrt_types::PJRT_Buffer_DonateWithControlDependency_Args,
                >(),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
                callback_data: ptr::null_mut(),
                dependency_callback: ptr::null(),
                out_buffer: ptr::null_mut(),
            };
            assert!(
                PJRT_Buffer_DonateWithControlDependency(ptr::addr_of_mut!(bf_don).cast()).is_null()
            );
            assert_eq!(bf_don.out_buffer, buf_in);

            let mut et_args = crate::pjrt_types::PJRT_Buffer_ElementType_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_ElementType_Args>(),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
                element_type: 0,
                pad: 0,
            };
            assert!(PJRT_Buffer_ElementType(ptr::addr_of_mut!(et_args).cast()).is_null());
            assert_eq!(et_args.element_type, 11);

            let mut sz_args = crate::pjrt_types::PJRT_Buffer_OnDeviceSizeInBytes_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_OnDeviceSizeInBytes_Args>(),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
                on_device_size_in_bytes: 0,
            };
            assert!(PJRT_Buffer_OnDeviceSizeInBytes(ptr::addr_of_mut!(sz_args).cast()).is_null());
            assert!(sz_args.on_device_size_in_bytes >= 16);

            let mut c2d_args = crate::pjrt_types::PJRT_Buffer_CopyToDevice_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_CopyToDevice_Args>(),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
                dst_device: look_args.device,
                dst_buffer: ptr::null_mut(),
            };
            assert!(PJRT_Buffer_CopyToDevice(ptr::addr_of_mut!(c2d_args).cast()).is_null());
            assert!(!c2d_args.dst_buffer.is_null());
            libtpu_rs_buffer_destroy(c2d_args.dst_buffer);

            for &mem0 in (*client).memory_ptrs.first() {
                let mut c2m_args = crate::pjrt_types::PJRT_Buffer_CopyToMemory_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_CopyToMemory_Args>(),
                    extension_start: ptr::null_mut(),
                    buffer: buf_in,
                    dst_memory: mem0,
                    dst_buffer: ptr::null_mut(),
                };
                assert!(PJRT_Buffer_CopyToMemory(ptr::addr_of_mut!(c2m_args).cast()).is_null());
                assert!(!c2m_args.dst_buffer.is_null());
                libtpu_rs_buffer_destroy(c2m_args.dst_buffer);

                let mut uninit_args =
                    crate::pjrt_types::PJRT_Client_CreateUninitializedBuffer_Args {
                        struct_size: size_of::<
                            crate::pjrt_types::PJRT_Client_CreateUninitializedBuffer_Args,
                        >(),
                        extension_start: ptr::null_mut(),
                        client,
                        dims: dims_arr.as_ptr(),
                        num_dims: 1,
                        element_type: 11,
                        pad: 0,
                        layout: ptr::null(),
                        device: look_args.device,
                        memory: mem0,
                        buffer: ptr::null_mut(),
                    };
                assert!(
                    PJRT_Client_CreateUninitializedBuffer(ptr::addr_of_mut!(uninit_args).cast())
                        .is_null()
                );
                assert!(!uninit_args.buffer.is_null());
                libtpu_rs_buffer_destroy(uninit_args.buffer);

                let mut alias_args = crate::pjrt_types::PJRT_Client_CreateAliasBuffer_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_Client_CreateAliasBuffer_Args>(),
                    extension_start: ptr::null_mut(),
                    client,
                    memory: mem0,
                    dims: dims_arr.as_ptr(),
                    num_dims: 1,
                    element_type: 11,
                    pad: 0,
                    layout: ptr::null(),
                    alias_buffer: ptr::null_mut(),
                    fulfill_alias_buffer_cb: ptr::null_mut(),
                };
                assert!(
                    PJRT_Client_CreateAliasBuffer(ptr::addr_of_mut!(alias_args).cast()).is_null()
                );
                assert!(!alias_args.alias_buffer.is_null());
                libtpu_rs_buffer_destroy(alias_args.alias_buffer);

                let mut err_buf_args = crate::pjrt_types::PJRT_Client_CreateErrorBuffer_Args {
                    struct_size: size_of::<crate::pjrt_types::PJRT_Client_CreateErrorBuffer_Args>(),
                    extension_start: ptr::null_mut(),
                    client,
                    dims: dims_arr.as_ptr(),
                    num_dims: 1,
                    element_type: 11,
                    error_code: 3,
                    error_message: c"err".as_ptr(),
                    error_message_size: 3,
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
                assert!(!err_buf_args.buffer.is_null());
                libtpu_rs_buffer_destroy(err_buf_args.buffer);
            }

            unsafe extern "C" fn test_chunk_deleter(
                _data: *mut std::ffi::c_void,
                arg: *mut std::ffi::c_void,
            ) {
                // SAFETY: `arg` points to a mutable `u32` counter.
                unsafe {
                    *arg.cast::<u32>() += 1;
                }
            }
            let sample_stream =
                Box::into_raw(Box::new(crate::pjrt_types::PJRT_CopyToDeviceStream {
                    inner: libtpu_core::pjrt::PjrtCoreCopyToDeviceStream::new(256, 128),
                }));
            for st_fn in [
                PJRT_CopyToDeviceStream_AddChunk
                    as unsafe extern "C" fn(*mut std::ffi::c_void) -> *mut PJRT_Error,
                PJRT_CopyToDeviceStream_TotalBytes,
                PJRT_CopyToDeviceStream_GranuleSize,
                PJRT_CopyToDeviceStream_CurrentBytes,
            ] {
                let mut st_args = [0_usize; 8];
                st_args[0] = 64;
                st_args[2] = sample_stream as usize;
                assert!(st_fn(st_args.as_mut_ptr().cast()).is_null());
            }
            let mut chunk_del_hits: u32 = 0;
            let mut chunk_payload = [42_u8; 64];
            let mut chunk_val = crate::pjrt_types::PJRT_Chunk {
                data: chunk_payload.as_mut_ptr().cast(),
                size: 64,
                deleter: Some(test_chunk_deleter),
                deleter_arg: ptr::addr_of_mut!(chunk_del_hits).cast(),
            };
            let mut add_chunk_args = crate::pjrt_types::PJRT_CopyToDeviceStream_AddChunk_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_CopyToDeviceStream_AddChunk_Args>(),
                extension_start: ptr::null_mut(),
                stream: sample_stream,
                chunk: ptr::addr_of_mut!(chunk_val),
                transfer_complete: ptr::null_mut(),
            };
            assert!(
                PJRT_CopyToDeviceStream_AddChunk(ptr::addr_of_mut!(add_chunk_args).cast())
                    .is_null()
            );
            assert_eq!(chunk_del_hits, 1);
            let mut chunk_val_no_del = crate::pjrt_types::PJRT_Chunk {
                data: ptr::null_mut(),
                size: 64,
                deleter: None,
                deleter_arg: ptr::null_mut(),
            };
            add_chunk_args.chunk = ptr::addr_of_mut!(chunk_val_no_del);
            assert!(
                PJRT_CopyToDeviceStream_AddChunk(ptr::addr_of_mut!(add_chunk_args).cast())
                    .is_null()
            );
            add_chunk_args.stream = ptr::null_mut();
            assert!(
                PJRT_CopyToDeviceStream_AddChunk(ptr::addr_of_mut!(add_chunk_args).cast())
                    .is_null()
            );
            let mut st_del_args = crate::pjrt_types::PJRT_CopyToDeviceStream_Destroy_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_CopyToDeviceStream_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                stream: sample_stream,
            };
            assert!(
                PJRT_CopyToDeviceStream_Destroy(ptr::addr_of_mut!(st_del_args).cast()).is_null()
            );

            let mut raw_dst = [0_u8; 16];
            let mut raw_copy_h = crate::pjrt_types::PJRT_Buffer_CopyRawToHost_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_CopyRawToHost_Args>(),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
                dst: raw_dst.as_mut_ptr().cast(),
                offset: 0,
                transfer_size: 16,
                event: ptr::null_mut(),
            };
            assert!(PJRT_Buffer_CopyRawToHost(ptr::addr_of_mut!(raw_copy_h).cast()).is_null());
            raw_copy_h.offset = 9999;
            assert!(PJRT_Buffer_CopyRawToHost(ptr::addr_of_mut!(raw_copy_h).cast()).is_null());

            let raw_host_bytes: Vec<u8> = [1.0_f32, 2.0, 3.0, 4.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect();
            for tm_inner in libtpu_core::pjrt::PjrtCoreAsyncTransferManager::create(
                DeviceId(0),
                libtpu_core::types::MemorySpaceId(0),
                vec![(*buf_in).inner.shape.clone()],
                &(*client).inner.hal,
            ) {
                let sample_tm = Box::into_raw(Box::new(
                    crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager {
                        inner: tm_inner,
                        device_ptr: look_args.device,
                        memory_ptr: (*look_args.device).default_memory_ptr,
                    },
                ));
                let mut tm_bc =
                    crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args {
                        struct_size: size_of::<
                            crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args,
                        >(),
                        extension_start: ptr::null_mut(),
                        transfer_manager: sample_tm,
                        buffer_count: 0,
                    };
                assert!(
                    PJRT_AsyncHostToDeviceTransferManager_BufferCount(
                        ptr::addr_of_mut!(tm_bc).cast()
                    )
                    .is_null()
                );
                assert_eq!(tm_bc.buffer_count, 1);

                let mut tm_bs =
                    crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args {
                        struct_size: size_of::<
                            crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args,
                        >(),
                        extension_start: ptr::null_mut(),
                        transfer_manager: sample_tm,
                        buffer_index: 0,
                        pad: 0,
                        buffer_size: 0,
                    };
                assert!(
                    PJRT_AsyncHostToDeviceTransferManager_BufferSize(
                        ptr::addr_of_mut!(tm_bs).cast()
                    )
                    .is_null()
                );
                assert!(tm_bs.buffer_size >= 16);
                tm_bs.buffer_index = 99;
                assert!(
                    PJRT_AsyncHostToDeviceTransferManager_BufferSize(
                        ptr::addr_of_mut!(tm_bs).cast()
                    )
                    .is_null()
                );

                let mut tm_dev =
                    crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_Device_Args {
                        struct_size: size_of::<
                            crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_Device_Args,
                        >(),
                        extension_start: ptr::null_mut(),
                        transfer_manager: sample_tm,
                        device_out: ptr::null_mut(),
                    };
                assert!(
                    PJRT_AsyncHostToDeviceTransferManager_Device(ptr::addr_of_mut!(tm_dev).cast())
                        .is_null()
                );

                let mut tm_tx =
                    crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_TransferData_Args {
                        struct_size: size_of::<
                            crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_TransferData_Args,
                        >(),
                        extension_start: ptr::null_mut(),
                        transfer_manager: sample_tm,
                        buffer_index: 0,
                        pad: 0,
                        data: raw_host_bytes.as_ptr().cast(),
                        offset: 0,
                        transfer_size: 16,
                        is_last_transfer: true,
                        pad1: [0; 7],
                        done_with_h2d_transfer: ptr::null_mut(),
                    };
                assert!(
                    PJRT_AsyncHostToDeviceTransferManager_TransferData(
                        ptr::addr_of_mut!(tm_tx).cast()
                    )
                    .is_null()
                );
                tm_tx.data = ptr::null();
                assert!(
                    PJRT_AsyncHostToDeviceTransferManager_TransferData(
                        ptr::addr_of_mut!(tm_tx).cast()
                    )
                    .is_null()
                );

                let mut tm_ret =
                    crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args {
                        struct_size: size_of::<
                            crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args,
                        >(),
                        extension_start: ptr::null_mut(),
                        transfer_manager: sample_tm,
                        buffer_index: 0,
                        pad: 0,
                        buffer_out: ptr::null_mut(),
                    };
                assert!(
                    PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer(
                        ptr::addr_of_mut!(tm_ret).cast()
                    )
                    .is_null()
                );
                assert!(!tm_ret.buffer_out.is_null());
                libtpu_rs_buffer_destroy(tm_ret.buffer_out);
                tm_ret.buffer_index = 99;
                tm_ret.buffer_out = ptr::null_mut();
                assert!(
                    PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer(
                        ptr::addr_of_mut!(tm_ret).cast()
                    )
                    .is_null()
                );
                libtpu_rs_buffer_destroy(tm_ret.buffer_out);

                let mut tm_sbe =
                    crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args {
                        struct_size: size_of::<
                            crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args,
                        >(),
                        extension_start: ptr::null_mut(),
                        transfer_manager: sample_tm,
                        buffer_index: 0,
                        pad0: 0,
                        error_code: 3,
                        pad1: 0,
                        error_message: c"transfer failed".as_ptr(),
                    };
                assert!(
                    PJRT_AsyncHostToDeviceTransferManager_SetBufferError(
                        ptr::addr_of_mut!(tm_sbe).cast()
                    )
                    .is_null()
                );

                let sample_meta = crate::pjrt_types::PJRT_TransferMetadata {
                    struct_size: size_of::<crate::pjrt_types::PJRT_TransferMetadata>(),
                    extension_start: ptr::null_mut(),
                    flags: 7,
                };
                let mut tm_meta =
                    crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args {
                        struct_size: size_of::<
                            crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args,
                        >(),
                        extension_start: ptr::null_mut(),
                        transfer_manager: sample_tm,
                        transfer_metadata: ptr::null(),
                        num_metadata: 0,
                    };
                assert!(
                    PJRT_AsyncHostToDeviceTransferManager_AddMetadata(
                        ptr::addr_of_mut!(tm_meta).cast()
                    )
                    .is_null()
                );
                tm_meta.transfer_metadata = ptr::addr_of!(sample_meta);
                tm_meta.num_metadata = 1;
                assert!(
                    PJRT_AsyncHostToDeviceTransferManager_AddMetadata(
                        ptr::addr_of_mut!(tm_meta).cast()
                    )
                    .is_null()
                );

                let mut tm_lit =
                    crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args {
                        struct_size: size_of::<
                            crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args,
                        >(),
                        extension_start: ptr::null_mut(),
                        transfer_manager: sample_tm,
                        buffer_index: 0,
                        pad: 0,
                        data: ptr::null(),
                        shape_dims: ptr::null(),
                        shape_num_dims: 0,
                        shape_element_type: 11,
                        pad1: 0,
                        shape_layout: ptr::null(),
                        done_with_h2d_transfer: ptr::null_mut(),
                    };
                assert!(
                    PJRT_AsyncHostToDeviceTransferManager_TransferLiteral(
                        ptr::addr_of_mut!(tm_lit).cast()
                    )
                    .is_null()
                );
                tm_lit.data = raw_host_bytes.as_ptr().cast();
                tm_lit.shape_dims = dims_arr.as_ptr();
                tm_lit.shape_num_dims = 1;
                assert!(
                    PJRT_AsyncHostToDeviceTransferManager_TransferLiteral(
                        ptr::addr_of_mut!(tm_lit).cast()
                    )
                    .is_null()
                );
                tm_lit.shape_dims = ptr::null();
                tm_lit.shape_num_dims = 0;
                tm_lit.shape_element_type = 999;
                assert!(
                    PJRT_AsyncHostToDeviceTransferManager_TransferLiteral(
                        ptr::addr_of_mut!(tm_lit).cast()
                    )
                    .is_null()
                );

                let mut tm_del =
                    crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_Destroy_Args {
                        struct_size: size_of::<
                            crate::pjrt_types::PJRT_AsyncHostToDeviceTransferManager_Destroy_Args,
                        >(),
                        extension_start: ptr::null_mut(),
                        transfer_manager: sample_tm,
                    };
                assert!(
                    PJRT_AsyncHostToDeviceTransferManager_Destroy(ptr::addr_of_mut!(tm_del).cast())
                        .is_null()
                );
            }

            let bc_dims = [2_i64, 2];
            let mut bc_args = crate::pjrt_types::PJRT_Buffer_Bitcast_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_Bitcast_Args>(),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
                element_type: 11,
                pad: 0,
                dims: bc_dims.as_ptr(),
                num_dims: bc_dims.len(),
                device_layout: ptr::null(),
                out_buffer: ptr::null_mut(),
            };
            assert!(PJRT_Buffer_Bitcast(ptr::addr_of_mut!(bc_args).cast()).is_null());
            assert!(!bc_args.out_buffer.is_null());
            libtpu_rs_buffer_destroy(bc_args.out_buffer);

            bc_args.dims = ptr::null();
            bc_args.num_dims = 0;
            bc_args.element_type = 999;
            bc_args.out_buffer = ptr::null_mut();
            assert!(PJRT_Buffer_Bitcast(ptr::addr_of_mut!(bc_args).cast()).is_null());
            libtpu_rs_buffer_destroy(bc_args.out_buffer);

            let bad_bc_dims = [-1_i64];
            bc_args.dims = bad_bc_dims.as_ptr();
            bc_args.num_dims = 1;
            bc_args.out_buffer = ptr::null_mut();
            assert!(PJRT_Buffer_Bitcast(ptr::addr_of_mut!(bc_args).cast()).is_null());
            assert!(bc_args.out_buffer.is_null());

            let mut del_b_args = crate::pjrt_types::PJRT_Buffer_Destroy_Args {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
            };
            assert!(PJRT_Buffer_Destroy(ptr::addr_of_mut!(del_b_args).cast()).is_null());

            let mut destroy_client_args = PJRT_Client_Destroy_Args {
                struct_size: size_of::<PJRT_Client_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                client,
            };
            assert!(PJRT_Client_Destroy(ptr::addr_of_mut!(destroy_client_args).cast()).is_null());
        }
    }
}

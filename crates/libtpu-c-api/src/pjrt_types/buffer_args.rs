//! `#[repr(C)]` argument structures for buffer, stream, and async transfer entrypoints.
//!
//! Covers `PJRT_Buffer_*`, `PJRT_CopyToDeviceStream_*`, and `PJRT_AsyncHostToDeviceTransferManager_*`.

use crate::pjrt_types::{
    PJRT_AsyncHostToDeviceTransferManager, PJRT_Buffer, PJRT_Buffer_MemoryLayout, PJRT_Chunk,
    PJRT_CopyToDeviceStream, PJRT_Device, PJRT_Event, PJRT_Extension_Base, PJRT_Memory,
    PJRT_TransferMetadata, PjrtArgsStruct,
};
use std::os::raw::c_char;

/// Argument structure for `PJRT_Buffer_Destroy` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Buffer handle to destroy.
    pub buffer: *mut PJRT_Buffer,
}

/// Argument structure for `PJRT_Buffer_ElementType` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_ElementType_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Buffer handle.
    pub buffer: *const PJRT_Buffer,
    /// Output element type (`i32`).
    pub element_type: i32,
    /// Explicit padding.
    pub pad: i32,
}

/// Argument structure for `PJRT_Buffer_Dimensions` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_Dimensions_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Buffer handle.
    pub buffer: *const PJRT_Buffer,
    /// Output dimensions pointer (`const int64_t*`).
    pub dims: *const i64,
    /// Output number of dimensions.
    pub num_dims: usize,
}

/// Argument structure for `PJRT_Buffer_UnpaddedDimensions` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_UnpaddedDimensions_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Buffer handle.
    pub buffer: *const PJRT_Buffer,
    /// Output unpadded dimensions pointer (`const int64_t*`).
    pub unpadded_dims: *const i64,
    /// Output number of dimensions.
    pub num_dims: usize,
}

/// Argument structure for `PJRT_Buffer_DynamicDimensionIndices` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_DynamicDimensionIndices_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Buffer handle.
    pub buffer: *const PJRT_Buffer,
    /// Output dynamic dimension indices pointer.
    pub dynamic_dimension_indices: *const usize,
    /// Output number of dynamic dimensions.
    pub num_dynamic_dimensions: usize,
}

/// Argument structure for `PJRT_Buffer_GetMemoryLayout` (`104` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_GetMemoryLayout_Args {
    /// Size of this struct (`104`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Buffer handle.
    pub buffer: *const PJRT_Buffer,
    /// Output inline `PJRT_Buffer_MemoryLayout` (`80` bytes = `10` words).
    pub layout_words: [usize; 10],
}

/// Argument structure for `PJRT_Buffer_OnDeviceSizeInBytes` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_OnDeviceSizeInBytes_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Buffer handle.
    pub buffer: *const PJRT_Buffer,
    /// Output `on_device_size_in_bytes`.
    pub on_device_size_in_bytes: usize,
}

/// Argument structure for `PJRT_Buffer_Device` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_Device_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Buffer handle.
    pub buffer: *const PJRT_Buffer,
    /// Output `PJRT_Device*`.
    pub device: *mut PJRT_Device,
}

/// Argument structure for `PJRT_Buffer_Memory` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_Memory_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Buffer handle.
    pub buffer: *const PJRT_Buffer,
    /// Output `PJRT_Memory*`.
    pub memory: *mut PJRT_Memory,
}

/// Argument structure for `PJRT_Buffer_Delete` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_Delete_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Buffer handle.
    pub buffer: *mut PJRT_Buffer,
}

/// Argument structure for `PJRT_Buffer_IsDeleted` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_IsDeleted_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Buffer handle.
    pub buffer: *const PJRT_Buffer,
    /// Output `is_deleted` flag.
    pub is_deleted: bool,
    /// Explicit padding.
    pub pad: [u8; 7],
}

/// Argument structure for `PJRT_Buffer_CopyToDevice` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_CopyToDevice_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Source buffer handle.
    pub buffer: *const PJRT_Buffer,
    /// Destination `PJRT_Device*`.
    pub dst_device: *mut PJRT_Device,
    /// Output destination `PJRT_Buffer*`.
    pub dst_buffer: *mut PJRT_Buffer,
}

/// Argument structure for `PJRT_Buffer_ToHostBuffer` (`56` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_ToHostBuffer_Args {
    /// Size of this struct (`56`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Source `PJRT_Buffer*`.
    pub src: *const PJRT_Buffer,
    /// Optional host layout pointer.
    pub host_layout: *const PJRT_Buffer_MemoryLayout,
    /// Destination host buffer pointer (or `NULL` to query size).
    pub dst: *mut std::ffi::c_void,
    /// Destination byte size.
    pub dst_size: usize,
    /// Output completion `PJRT_Event*`.
    pub event: *mut PJRT_Event,
}

/// Argument structure for `PJRT_Buffer_IsOnCpu` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_IsOnCpu_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Buffer handle.
    pub buffer: *const PJRT_Buffer,
    /// Output `is_on_cpu` flag.
    pub is_on_cpu: bool,
    /// Explicit padding.
    pub pad: [u8; 7],
}

/// Argument structure for `PJRT_Buffer_ReadyEvent` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_ReadyEvent_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Buffer handle.
    pub buffer: *const PJRT_Buffer,
    /// Output `PJRT_Event*`.
    pub event: *mut PJRT_Event,
}

/// Argument structure for `PJRT_Buffer_UnsafePointer` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_UnsafePointer_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Buffer handle.
    pub buffer: *const PJRT_Buffer,
    /// Output raw device pointer (`usize`).
    pub buffer_pointer: usize,
}

/// Argument structure for `PJRT_Buffer_IncreaseExternalReferenceCount` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_IncreaseExternalReferenceCount_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Buffer handle.
    pub buffer: *mut PJRT_Buffer,
}

/// Argument structure for `PJRT_Buffer_DecreaseExternalReferenceCount` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_DecreaseExternalReferenceCount_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Buffer handle.
    pub buffer: *mut PJRT_Buffer,
}

/// Argument structure for `PJRT_Buffer_OpaqueDeviceMemoryDataPointer` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Buffer handle.
    pub buffer: *const PJRT_Buffer,
    /// Output opaque device memory pointer.
    pub device_memory_ptr: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_Buffer_CopyToMemory` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_CopyToMemory_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Source buffer handle.
    pub buffer: *const PJRT_Buffer,
    /// Destination `PJRT_Memory*`.
    pub dst_memory: *mut PJRT_Memory,
    /// Output destination `PJRT_Buffer*`.
    pub dst_buffer: *mut PJRT_Buffer,
}

/// Argument structure for `PJRT_Buffer_CopyRawToHost` (`56` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_CopyRawToHost_Args {
    /// Size of this struct (`56`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Source buffer handle.
    pub buffer: *const PJRT_Buffer,
    /// Destination host pointer.
    pub dst: *mut std::ffi::c_void,
    /// Byte offset into source buffer.
    pub offset: i64,
    /// Number of bytes to copy.
    pub transfer_size: i64,
    /// Output completion `PJRT_Event*`.
    pub event: *mut PJRT_Event,
}

/// Argument structure for `PJRT_Buffer_CopyRawToHostFuture` (`64` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_CopyRawToHostFuture_Args {
    /// Size of this struct (`64`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Buffer handle.
    pub buffer: *const PJRT_Buffer,
    /// Byte offset.
    pub offset: i64,
    /// Transfer size in bytes.
    pub transfer_size: i64,
    /// Output completion `PJRT_Event*`.
    pub event: *mut PJRT_Event,
    /// Output callback data.
    pub callback_data: *mut std::ffi::c_void,
    /// Output future ready callback.
    pub future_ready_callback: *const std::ffi::c_void,
}

/// Argument structure for `PJRT_Buffer_DonateWithControlDependency` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_DonateWithControlDependency_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Buffer handle.
    pub buffer: *mut PJRT_Buffer,
    /// Dependency callback data.
    pub callback_data: *mut std::ffi::c_void,
    /// Dependency callback pointer.
    pub dependency_callback: *const std::ffi::c_void,
    /// Output donated `PJRT_Buffer*`.
    pub out_buffer: *mut PJRT_Buffer,
}

/// Argument structure for `PJRT_Buffer_Bitcast` (`64` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_Bitcast_Args {
    /// Size of this struct (`64`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Source buffer handle.
    pub buffer: *mut PJRT_Buffer,
    /// New element type (`i32`).
    pub element_type: i32,
    /// Explicit padding.
    pub pad: i32,
    /// New dimensions pointer.
    pub dims: *const i64,
    /// Number of dimensions.
    pub num_dims: usize,
    /// New device layout pointer.
    pub device_layout: *const PJRT_Buffer_MemoryLayout,
    /// Output bitcast `PJRT_Buffer*`.
    pub out_buffer: *mut PJRT_Buffer,
}

/// Argument structure for `PJRT_CopyToDeviceStream_Destroy` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_CopyToDeviceStream_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Stream handle to destroy.
    pub stream: *mut PJRT_CopyToDeviceStream,
}

/// Argument structure for `PJRT_CopyToDeviceStream_AddChunk` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_CopyToDeviceStream_AddChunk_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Stream handle.
    pub stream: *mut PJRT_CopyToDeviceStream,
    /// Input `PJRT_Chunk*`.
    pub chunk: *const PJRT_Chunk,
    /// Output `transfer_complete` (`PJRT_Event*`).
    pub transfer_complete: *mut PJRT_Event,
}

/// Argument structure for `PJRT_CopyToDeviceStream_TotalBytes` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_CopyToDeviceStream_TotalBytes_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Stream handle.
    pub stream: *const PJRT_CopyToDeviceStream,
    /// Output `total_bytes`.
    pub total_bytes: i64,
}

/// Argument structure for `PJRT_CopyToDeviceStream_GranuleSize` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_CopyToDeviceStream_GranuleSize_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Stream handle.
    pub stream: *const PJRT_CopyToDeviceStream,
    /// Output `granule_size_in_bytes`.
    pub granule_size_in_bytes: i64,
}

/// Argument structure for `PJRT_CopyToDeviceStream_CurrentBytes` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_CopyToDeviceStream_CurrentBytes_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Stream handle.
    pub stream: *const PJRT_CopyToDeviceStream,
    /// Output `current_bytes`.
    pub current_bytes: i64,
}

/// Argument structure for `PJRT_AsyncHostToDeviceTransferManager_Destroy` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_AsyncHostToDeviceTransferManager_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Transfer manager handle to destroy.
    pub transfer_manager: *mut PJRT_AsyncHostToDeviceTransferManager,
}

/// Argument structure for `PJRT_AsyncHostToDeviceTransferManager_TransferData` (`72` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_AsyncHostToDeviceTransferManager_TransferData_Args {
    /// Size of this struct (`72`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Transfer manager handle.
    pub transfer_manager: *mut PJRT_AsyncHostToDeviceTransferManager,
    /// Target buffer index.
    pub buffer_index: i32,
    /// Explicit padding.
    pub pad: i32,
    /// Source host data pointer.
    pub data: *const std::ffi::c_void,
    /// Byte offset into target buffer.
    pub offset: i64,
    /// Number of bytes to transfer.
    pub transfer_size: i64,
    /// Whether this chunk completes the transfer for `buffer_index`.
    pub is_last_transfer: bool,
    /// Explicit 7-byte padding.
    pub pad1: [u8; 7],
    /// Output `done_with_h2d_transfer` (`PJRT_Event*`).
    pub done_with_h2d_transfer: *mut PJRT_Event,
}

/// Argument structure for `PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Transfer manager handle.
    pub transfer_manager: *mut PJRT_AsyncHostToDeviceTransferManager,
    /// Buffer index.
    pub buffer_index: i32,
    /// Explicit padding.
    pub pad: i32,
    /// Output `PJRT_Buffer*`.
    pub buffer_out: *mut PJRT_Buffer,
}

/// Argument structure for `PJRT_AsyncHostToDeviceTransferManager_Device` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_AsyncHostToDeviceTransferManager_Device_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Transfer manager handle.
    pub transfer_manager: *const PJRT_AsyncHostToDeviceTransferManager,
    /// Output `PJRT_Device*`.
    pub device_out: *mut PJRT_Device,
}

/// Argument structure for `PJRT_AsyncHostToDeviceTransferManager_BufferCount` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Transfer manager handle.
    pub transfer_manager: *const PJRT_AsyncHostToDeviceTransferManager,
    /// Output `buffer_count`.
    pub buffer_count: usize,
}

/// Argument structure for `PJRT_AsyncHostToDeviceTransferManager_BufferSize` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Transfer manager handle.
    pub transfer_manager: *const PJRT_AsyncHostToDeviceTransferManager,
    /// Buffer index.
    pub buffer_index: i32,
    /// Explicit padding.
    pub pad: i32,
    /// Output `buffer_size`.
    pub buffer_size: usize,
}

/// Argument structure for `PJRT_AsyncHostToDeviceTransferManager_SetBufferError` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Transfer manager handle.
    pub transfer_manager: *mut PJRT_AsyncHostToDeviceTransferManager,
    /// Buffer index.
    pub buffer_index: i32,
    /// Explicit padding.
    pub pad0: i32,
    /// Error code (`i32`).
    pub error_code: i32,
    /// Explicit padding.
    pub pad1: i32,
    /// Error message pointer.
    pub error_message: *const c_char,
}

/// Argument structure for `PJRT_AsyncHostToDeviceTransferManager_AddMetadata` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Transfer manager handle.
    pub transfer_manager: *mut PJRT_AsyncHostToDeviceTransferManager,
    /// Metadata array pointer.
    pub transfer_metadata: *const PJRT_TransferMetadata,
    /// Number of metadata entries.
    pub num_metadata: usize,
}

/// Argument structure for `PJRT_AsyncHostToDeviceTransferManager_TransferLiteral` (`80` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args {
    /// Size of this struct (`80`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Transfer manager handle.
    pub transfer_manager: *mut PJRT_AsyncHostToDeviceTransferManager,
    /// Buffer index.
    pub buffer_index: i32,
    /// Explicit padding.
    pub pad: i32,
    /// Host literal data pointer.
    pub data: *const std::ffi::c_void,
    /// Shape dimensions pointer.
    pub shape_dims: *const i64,
    /// Number of shape dimensions.
    pub shape_num_dims: usize,
    /// Shape element type (`i32`).
    pub shape_element_type: i32,
    /// Explicit padding.
    pub pad1: i32,
    /// Optional shape memory layout pointer.
    pub shape_layout: *const PJRT_Buffer_MemoryLayout,
    /// Output `done_with_h2d_transfer` (`PJRT_Event*`).
    pub done_with_h2d_transfer: *mut PJRT_Event,
}

impl PjrtArgsStruct for PJRT_Buffer_Destroy_Args {}
impl PjrtArgsStruct for PJRT_Buffer_ElementType_Args {
    const MIN_STRUCT_SIZE: usize = 28;
}
impl PjrtArgsStruct for PJRT_Buffer_Dimensions_Args {}
impl PjrtArgsStruct for PJRT_Buffer_UnpaddedDimensions_Args {}
impl PjrtArgsStruct for PJRT_Buffer_DynamicDimensionIndices_Args {}
impl PjrtArgsStruct for PJRT_Buffer_GetMemoryLayout_Args {}
impl PjrtArgsStruct for PJRT_Buffer_OnDeviceSizeInBytes_Args {}
impl PjrtArgsStruct for PJRT_Buffer_Device_Args {}
impl PjrtArgsStruct for PJRT_Buffer_Memory_Args {}
impl PjrtArgsStruct for PJRT_Buffer_Delete_Args {}
impl PjrtArgsStruct for PJRT_Buffer_IsDeleted_Args {
    const MIN_STRUCT_SIZE: usize = 25;
}
impl PjrtArgsStruct for PJRT_Buffer_CopyToDevice_Args {}
impl PjrtArgsStruct for PJRT_Buffer_ToHostBuffer_Args {}
impl PjrtArgsStruct for PJRT_Buffer_IsOnCpu_Args {
    const MIN_STRUCT_SIZE: usize = 25;
}
impl PjrtArgsStruct for PJRT_Buffer_ReadyEvent_Args {}
impl PjrtArgsStruct for PJRT_Buffer_UnsafePointer_Args {}
impl PjrtArgsStruct for PJRT_Buffer_IncreaseExternalReferenceCount_Args {}
impl PjrtArgsStruct for PJRT_Buffer_DecreaseExternalReferenceCount_Args {}
impl PjrtArgsStruct for PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args {}
impl PjrtArgsStruct for PJRT_Buffer_CopyToMemory_Args {}
impl PjrtArgsStruct for PJRT_Buffer_CopyRawToHost_Args {}
impl PjrtArgsStruct for PJRT_Buffer_CopyRawToHostFuture_Args {}
impl PjrtArgsStruct for PJRT_Buffer_DonateWithControlDependency_Args {}
impl PjrtArgsStruct for PJRT_Buffer_Bitcast_Args {}
impl PjrtArgsStruct for PJRT_CopyToDeviceStream_Destroy_Args {}
impl PjrtArgsStruct for PJRT_CopyToDeviceStream_AddChunk_Args {}
impl PjrtArgsStruct for PJRT_CopyToDeviceStream_TotalBytes_Args {}
impl PjrtArgsStruct for PJRT_CopyToDeviceStream_GranuleSize_Args {}
impl PjrtArgsStruct for PJRT_CopyToDeviceStream_CurrentBytes_Args {}
impl PjrtArgsStruct for PJRT_AsyncHostToDeviceTransferManager_Destroy_Args {}
impl PjrtArgsStruct for PJRT_AsyncHostToDeviceTransferManager_TransferData_Args {}
impl PjrtArgsStruct for PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args {}
impl PjrtArgsStruct for PJRT_AsyncHostToDeviceTransferManager_Device_Args {}
impl PjrtArgsStruct for PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args {}
impl PjrtArgsStruct for PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args {}
impl PjrtArgsStruct for PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args {}
impl PjrtArgsStruct for PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args {}
impl PjrtArgsStruct for PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args {}

const _: () = assert!(std::mem::offset_of!(PJRT_Buffer_ToHostBuffer_Args, src) == 0x10);
const _: () = assert!(std::mem::offset_of!(PJRT_Buffer_ToHostBuffer_Args, host_layout) == 0x18);
const _: () = assert!(std::mem::offset_of!(PJRT_Buffer_ToHostBuffer_Args, dst) == 0x20);
const _: () = assert!(std::mem::offset_of!(PJRT_Buffer_ToHostBuffer_Args, dst_size) == 0x28);
const _: () = assert!(std::mem::offset_of!(PJRT_Buffer_ToHostBuffer_Args, event) == 0x30);

const _: () = assert!(size_of::<PJRT_Buffer_Destroy_Args>() == 24);
const _: () = assert!(size_of::<PJRT_Buffer_ElementType_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Buffer_Dimensions_Args>() == 40);
const _: () = assert!(size_of::<PJRT_Buffer_UnpaddedDimensions_Args>() == 40);
const _: () = assert!(size_of::<PJRT_Buffer_DynamicDimensionIndices_Args>() == 40);
const _: () = assert!(size_of::<PJRT_Buffer_GetMemoryLayout_Args>() == 104);
const _: () = assert!(size_of::<PJRT_Buffer_OnDeviceSizeInBytes_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Buffer_Device_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Buffer_Memory_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Buffer_Delete_Args>() == 24);
const _: () = assert!(size_of::<PJRT_Buffer_IsDeleted_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Buffer_CopyToDevice_Args>() == 40);
const _: () = assert!(size_of::<PJRT_Buffer_ToHostBuffer_Args>() == 56);
const _: () = assert!(size_of::<PJRT_Buffer_IsOnCpu_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Buffer_ReadyEvent_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Buffer_UnsafePointer_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Buffer_IncreaseExternalReferenceCount_Args>() == 24);
const _: () = assert!(size_of::<PJRT_Buffer_DecreaseExternalReferenceCount_Args>() == 24);
const _: () = assert!(size_of::<PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Buffer_CopyToMemory_Args>() == 40);
const _: () = assert!(size_of::<PJRT_Buffer_CopyRawToHost_Args>() == 56);
const _: () = assert!(size_of::<PJRT_Buffer_CopyRawToHostFuture_Args>() == 64);
const _: () = assert!(size_of::<PJRT_Buffer_DonateWithControlDependency_Args>() == 48);
const _: () = assert!(size_of::<PJRT_Buffer_Bitcast_Args>() == 64);
const _: () = assert!(size_of::<PJRT_CopyToDeviceStream_Destroy_Args>() == 24);
const _: () = assert!(size_of::<PJRT_CopyToDeviceStream_AddChunk_Args>() == 40);
const _: () = assert!(size_of::<PJRT_CopyToDeviceStream_TotalBytes_Args>() == 32);
const _: () = assert!(size_of::<PJRT_CopyToDeviceStream_GranuleSize_Args>() == 32);
const _: () = assert!(size_of::<PJRT_CopyToDeviceStream_CurrentBytes_Args>() == 32);
const _: () = assert!(size_of::<PJRT_AsyncHostToDeviceTransferManager_Destroy_Args>() == 24);
const _: () = assert!(size_of::<PJRT_AsyncHostToDeviceTransferManager_TransferData_Args>() == 72);
const _: () = assert!(size_of::<PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args>() == 40);
const _: () = assert!(size_of::<PJRT_AsyncHostToDeviceTransferManager_Device_Args>() == 32);
const _: () = assert!(size_of::<PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args>() == 32);
const _: () = assert!(size_of::<PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args>() == 40);
const _: () = assert!(size_of::<PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args>() == 48);
const _: () = assert!(size_of::<PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args>() == 40);
const _: () =
    assert!(size_of::<PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args>() == 80);

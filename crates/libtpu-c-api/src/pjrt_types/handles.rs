//! Opaque PJRT C API handles, common `#[repr(C)]` descriptors, and the [`PjrtArgsStruct`] trait.

use crate::pjrt_types::PJRT_Extension_Base;
use libtpu_core::error::LibTpuError;
use libtpu_core::pjrt::{
    PjrtCoreAsyncTransferManager, PjrtCoreBuffer, PjrtCoreClient, PjrtCoreCopyToDeviceStream,
    PjrtCoreDevice, PjrtCoreDeviceDescription, PjrtCoreEvent, PjrtCoreExecutable,
    PjrtCoreLoadedExecutable, PjrtCoreMemory, PjrtCoreTopology,
};
use std::ffi::CString;
use std::os::raw::c_char;

/// Opaque C handle wrapping a [`LibTpuError`] and its cached null-terminated message.
#[derive(Debug)]
pub struct PJRT_Error {
    /// Underlying core error.
    pub error: LibTpuError,
    /// Cached C-compatible error message string.
    pub c_message: CString,
}

/// Opaque C handle wrapping a [`PjrtCoreClient`].
#[derive(Debug)]
pub struct PJRT_Client {
    /// Underlying core client state.
    pub inner: PjrtCoreClient,
    /// Cached platform name (`"tpu"`).
    pub platform_name: CString,
    /// Cached platform version string.
    pub platform_version: CString,
    /// Heap-allocated device wrappers owned by this client.
    pub device_ptrs: Vec<*mut PJRT_Device>,
    /// Heap-allocated addressable device pointers.
    pub addressable_device_ptrs: Vec<*mut PJRT_Device>,
    /// Heap-allocated memory space wrappers owned by this client.
    pub memory_ptrs: Vec<*mut PJRT_Memory>,
    /// Heap-allocated topology wrapper owned by this client.
    pub topology_ptr: *mut PJRT_TopologyDescription,
    /// Optional distributed key-value store `kv_get` callback pointer.
    pub kv_get_callback: *const std::ffi::c_void,
    /// Optional distributed key-value store `kv_get` user argument.
    pub kv_get_user_arg: *mut std::ffi::c_void,
    /// Optional distributed key-value store `kv_put` callback pointer.
    pub kv_put_callback: *const std::ffi::c_void,
    /// Optional distributed key-value store `kv_put` user argument.
    pub kv_put_user_arg: *mut std::ffi::c_void,
    /// Optional distributed key-value store `kv_try_get` callback pointer.
    pub kv_try_get_callback: *const std::ffi::c_void,
    /// Optional distributed key-value store `kv_try_get` user argument.
    pub kv_try_get_user_arg: *mut std::ffi::c_void,
}

/// Opaque C handle wrapping a [`PjrtCoreDeviceDescription`].
#[derive(Debug)]
pub struct PJRT_DeviceDescription {
    /// Underlying core device description.
    pub inner: PjrtCoreDeviceDescription,
    /// Per-device `PJRT_NamedValue` attributes (`coords`, `core_on_chip`, `device_kind`).
    pub attributes: [PJRT_NamedValue; 3],
}

impl PJRT_DeviceDescription {
    /// Constructs a [`PJRT_DeviceDescription`] with precomputed per-device `PJRT_NamedValue` attributes.
    ///
    /// # Arguments
    ///
    /// * `inner` - Core device description.
    #[must_use]
    pub fn new(inner: PjrtCoreDeviceDescription) -> Self {
        let mut s = Self {
            inner,
            attributes: [PJRT_NamedValue {
                struct_size: size_of::<PJRT_NamedValue>(),
                extension_start: std::ptr::null_mut(),
                name: std::ptr::null(),
                name_size: 0,
                value_type_and_pad: 0,
                value_payload: 0,
                value_size: 0,
            }; 3],
        };
        let core_val = usize::from_ne_bytes(i64::from(s.inner.spec.core_on_chip.0).to_ne_bytes());
        s.attributes = [
            PJRT_NamedValue {
                struct_size: size_of::<PJRT_NamedValue>(),
                extension_start: std::ptr::null_mut(),
                name: c"device_kind".as_ptr(),
                name_size: 11,
                value_type_and_pad: 0,
                value_payload: s.inner.kind_str.as_ptr() as usize,
                value_size: s.inner.kind_str.len(),
            },
            PJRT_NamedValue {
                struct_size: size_of::<PJRT_NamedValue>(),
                extension_start: std::ptr::null_mut(),
                name: c"core_on_chip".as_ptr(),
                name_size: 12,
                value_type_and_pad: 1,
                value_payload: core_val,
                value_size: 1,
            },
            PJRT_NamedValue {
                struct_size: size_of::<PJRT_NamedValue>(),
                extension_start: std::ptr::null_mut(),
                name: c"coords".as_ptr(),
                name_size: 6,
                value_type_and_pad: 3,
                value_payload: s.inner.coords_attr.as_ptr() as usize,
                value_size: 3,
            },
        ];
        s
    }
}

/// Opaque C handle wrapping a [`PjrtCoreDevice`].
#[derive(Debug)]
pub struct PJRT_Device {
    /// Underlying core device.
    pub inner: PjrtCoreDevice,
    /// Owned pointer to the device description wrapper.
    pub desc_ptr: *mut PJRT_DeviceDescription,
    /// Default HBM memory space wrapper pointer for this device.
    pub default_memory_ptr: *mut PJRT_Memory,
    /// Memory space wrapper pointers addressable by this device.
    pub addressable_memory_ptrs: Vec<*mut PJRT_Memory>,
    /// Shared reference to the underlying hardware abstraction layer.
    pub hal: std::sync::Arc<dyn libtpu_core::hal::TpuHal>,
}

/// Opaque C handle wrapping a [`PjrtCoreMemory`].
#[derive(Debug)]
pub struct PJRT_Memory {
    /// Underlying core memory space.
    pub inner: PjrtCoreMemory,
    /// Device wrapper pointers that can address this memory space.
    pub addressable_by_device_ptrs: Vec<*mut PJRT_Device>,
}

/// Opaque C handle wrapping a [`PjrtCoreBuffer`].
#[derive(Debug)]
pub struct PJRT_Buffer {
    /// Underlying core buffer.
    pub inner: PjrtCoreBuffer,
    /// Owning device wrapper pointer.
    pub device_ptr: *mut PJRT_Device,
    /// Owning memory wrapper pointer.
    pub memory_ptr: *mut PJRT_Memory,
    /// Flattened tile dimensions for `PJRT_Buffer_GetMemoryLayout`.
    pub tile_dims_flat: Vec<i64>,
    /// Number of dimensions per tile for `PJRT_Buffer_GetMemoryLayout`.
    pub tile_dim_sizes: Vec<usize>,
    /// Precomputed dynamic dimension indices for `PJRT_Buffer_DynamicDimensionIndices`.
    pub dynamic_dim_indices: Vec<usize>,
}

impl PJRT_Buffer {
    /// Constructs a [`PJRT_Buffer`] with precomputed C ABI tile dimension vectors from `inner.shape.layout`.
    ///
    /// # Arguments
    ///
    /// * `inner` - Core buffer instance.
    /// * `device_ptr` - Owning device wrapper pointer.
    /// * `memory_ptr` - Owning memory wrapper pointer.
    #[must_use]
    pub fn new(
        inner: PjrtCoreBuffer,
        device_ptr: *mut PJRT_Device,
        memory_ptr: *mut PJRT_Memory,
    ) -> Self {
        let tile_dims_flat = match &inner.shape.layout {
            libtpu_core::types::MemoryLayout::Tiled { tile_dims, .. } => tile_dims.clone(),
            libtpu_core::types::MemoryLayout::Strides { byte_strides } => byte_strides.clone(),
        };
        let tile_dim_sizes = vec![tile_dims_flat.len()];
        let dynamic_dim_indices = inner
            .shape
            .dynamic_dims
            .iter()
            .enumerate()
            .filter_map(|(idx, &is_dyn)| is_dyn.then_some(idx))
            .collect();
        Self {
            inner,
            device_ptr,
            memory_ptr,
            tile_dims_flat,
            tile_dim_sizes,
            dynamic_dim_indices,
        }
    }
}

/// Opaque C handle wrapping a [`PjrtCoreExecutable`].
#[derive(Debug)]
pub struct PJRT_Executable {
    /// Underlying core executable.
    pub inner: PjrtCoreExecutable,
    /// Precomputed output element type codes (`PJRT_Buffer_Type`).
    pub output_element_types: Vec<i32>,
    /// Flattened output dimensions across all outputs.
    pub output_dims_flat: Vec<i64>,
    /// Number of dimensions for each output.
    pub output_dim_sizes: Vec<usize>,
    /// Per-output memory kind C string pointers (`"device"`).
    pub output_memory_kind_ptrs: Vec<*const c_char>,
    /// Per-output memory kind byte lengths (`6`).
    pub output_memory_kind_sizes: Vec<usize>,
    /// Per-parameter memory kind C string pointers (`"device"`).
    pub param_memory_kind_ptrs: Vec<*const c_char>,
    /// Per-parameter memory kind byte lengths (`6`).
    pub param_memory_kind_sizes: Vec<usize>,
    /// Precomputed cost analysis `PJRT_NamedValue` properties (`flops`, `transcendentals`, `bytes_accessed`).
    pub cost_properties: [PJRT_NamedValue; 3],
    /// Whether this executable is owned by a `PJRT_LoadedExecutable` (`true`) or standalone (`false`).
    pub owned_by_loaded: std::sync::atomic::AtomicBool,
}

impl PJRT_Executable {
    /// Constructs a [`PJRT_Executable`] with precomputed C ABI output type, dimension, memory kind, and cost arrays.
    ///
    /// # Arguments
    ///
    /// * `inner` - Core executable instance.
    #[must_use]
    pub fn new(inner: PjrtCoreExecutable) -> Self {
        let num_outputs = inner.program.output_shapes.len().max(1);
        let num_params = inner.program.parameter_shapes.len().max(1);
        let mut output_element_types = Vec::with_capacity(inner.program.output_shapes.len());
        let mut output_dims_flat = Vec::new();
        let mut output_dim_sizes = Vec::with_capacity(inner.program.output_shapes.len());
        for shape in &inner.program.output_shapes {
            output_element_types.push(shape.element_type.as_raw());
            output_dims_flat.extend_from_slice(&shape.dims);
            output_dim_sizes.push(shape.dims.len());
        }
        let output_memory_kind_ptrs = vec![c"device".as_ptr(); num_outputs];
        let output_memory_kind_sizes = vec![6_usize; num_outputs];
        let param_memory_kind_ptrs = vec![c"device".as_ptr(); num_params];
        let param_memory_kind_sizes = vec![6_usize; num_params];
        let cost = inner.cost_analysis();
        let cost_properties = [
            PJRT_NamedValue {
                struct_size: size_of::<PJRT_NamedValue>(),
                extension_start: std::ptr::null_mut(),
                name: c"flops".as_ptr(),
                name_size: 5,
                value_type_and_pad: 2,
                value_payload: usize::try_from(cost.flops.to_bits()).unwrap_or(0),
                value_size: 1,
            },
            PJRT_NamedValue {
                struct_size: size_of::<PJRT_NamedValue>(),
                extension_start: std::ptr::null_mut(),
                name: c"transcendentals".as_ptr(),
                name_size: 15,
                value_type_and_pad: 2,
                value_payload: usize::try_from(cost.transcendentals.to_bits()).unwrap_or(0),
                value_size: 1,
            },
            PJRT_NamedValue {
                struct_size: size_of::<PJRT_NamedValue>(),
                extension_start: std::ptr::null_mut(),
                name: c"bytes_accessed".as_ptr(),
                name_size: 14,
                value_type_and_pad: 2,
                value_payload: usize::try_from(cost.bytes_accessed.to_bits()).unwrap_or(0),
                value_size: 1,
            },
        ];
        Self {
            inner,
            output_element_types,
            output_dims_flat,
            output_dim_sizes,
            output_memory_kind_ptrs,
            output_memory_kind_sizes,
            param_memory_kind_ptrs,
            param_memory_kind_sizes,
            cost_properties,
            owned_by_loaded: std::sync::atomic::AtomicBool::new(false),
        }
    }
}

/// Opaque C handle wrapping a [`PjrtCoreLoadedExecutable`].
#[derive(Debug)]
pub struct PJRT_LoadedExecutable {
    /// Underlying core loaded executable.
    pub inner: PjrtCoreLoadedExecutable,
    /// Owned executable wrapper pointer.
    pub executable_ptr: *mut PJRT_Executable,
    /// Addressable device wrapper pointers owned by the parent client.
    pub device_ptrs: Vec<*mut PJRT_Device>,
    /// Precomputed logical replica/partition IDs for addressable devices.
    pub logical_device_ids: Vec<PJRT_LogicalDeviceIds>,
}

impl PJRT_LoadedExecutable {
    /// Constructs a [`PJRT_LoadedExecutable`] with precomputed logical device IDs.
    ///
    /// # Arguments
    ///
    /// * `inner` - Core loaded executable instance.
    /// * `executable_ptr` - Owned [`PJRT_Executable`] wrapper pointer.
    /// * `device_ptrs` - Addressable [`PJRT_Device`] wrapper pointers.
    #[must_use]
    pub fn new(
        inner: PjrtCoreLoadedExecutable,
        executable_ptr: *mut PJRT_Executable,
        device_ptrs: Vec<*mut PJRT_Device>,
    ) -> Self {
        // SAFETY: `executable_ptr` is a valid heap-allocated `PJRT_Executable`.
        unsafe {
            (*executable_ptr)
                .owned_by_loaded
                .store(true, std::sync::atomic::Ordering::Release);
        }
        let mut logical_device_ids = Vec::with_capacity(inner.addressable_devices.len());
        for d in &inner.addressable_devices {
            logical_device_ids.push(PJRT_LogicalDeviceIds {
                replica: i32::try_from(d.0).unwrap_or(0),
                partition: 0,
            });
        }
        Self {
            inner,
            executable_ptr,
            device_ptrs,
            logical_device_ids,
        }
    }
}

/// Opaque C handle wrapping a [`PjrtCoreEvent`].
#[derive(Debug)]
pub struct PJRT_Event {
    /// Underlying core event.
    pub inner: PjrtCoreEvent,
}

/// Opaque C handle wrapping a [`PjrtCoreTopology`].
#[derive(Debug)]
pub struct PJRT_TopologyDescription {
    /// Underlying core topology.
    pub inner: PjrtCoreTopology,
    /// Owned device description pointers.
    pub desc_ptrs: Vec<*mut PJRT_DeviceDescription>,
}

/// Opaque C handle representing a `PJRT_ExecuteContext`.
#[derive(Debug)]
pub struct PJRT_ExecuteContext {
    /// Unique context identifier.
    pub context_id: u64,
}

/// Opaque C handle wrapping a [`PjrtCoreCopyToDeviceStream`].
#[derive(Debug)]
pub struct PJRT_CopyToDeviceStream {
    /// Underlying stream state.
    pub inner: PjrtCoreCopyToDeviceStream,
}

/// Opaque C handle wrapping a [`PjrtCoreAsyncTransferManager`].
#[derive(Debug)]
pub struct PJRT_AsyncHostToDeviceTransferManager {
    /// Underlying async transfer manager.
    pub inner: PjrtCoreAsyncTransferManager,
    /// Target `PJRT_Device*` handle.
    pub device_ptr: *mut PJRT_Device,
    /// Target `PJRT_Memory*` handle.
    pub memory_ptr: *mut PJRT_Memory,
}

/// Heap-allocated serialized artifact holder backing `PJRT_SerializedExecutable`,
/// `PJRT_SerializedTopology`, `PJRT_SerializedDeviceAssignment`, and `PJRT_SerializedCompileOptions`.
#[derive(Debug)]
pub struct PjrtSerializedArtifact {
    /// Owned serialized wire bytes.
    pub bytes: Vec<u8>,
}

/// Heap-allocated communicator handle for `PJRT_Collectives_Extension` (`PJRT_Collectives_Communicator`).
#[repr(C)]
#[derive(Debug)]
pub struct PJRT_Collectives_Communicator {
    /// Unique communicator identifier.
    pub communicator_id: libtpu_core::types::CommunicatorId,
    /// Participating device IDs in the clique.
    pub device_ids: Vec<i32>,
    /// Cached C-string descriptor for `PJRT_Communicator_ToString`.
    pub descriptor: CString,
}

/// Heap-allocated memory layout handle for `PJRT_Layouts_Extension` (`PJRT_Layouts_MemoryLayout`).
#[repr(C)]
#[derive(Debug)]
pub struct PJRT_Layouts_MemoryLayout {
    /// Strongly-typed core [`libtpu_core::types::MemoryLayout`].
    pub layout: libtpu_core::types::MemoryLayout,
    /// Sublane tile dimension (`8`).
    pub sublane_tile: i64,
    /// Lane tile dimension (`128` or `256`).
    pub lane_tile: i64,
}

/// Static/heap-backed memory description structure for `PJRT_MemoryDescriptions_Extension` (`PJRT_MemoryDescription`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_MemoryDescription {
    /// Strongly-typed memory space classification (`Hbm`, `Vmem`, `Smem`, `HostPinned`).
    pub kind: libtpu_core::types::MemorySpaceKind,
    /// Numeric kind ID (`0` = HBM, `1` = VMEM, `2` = SMEM, `3` = `HostPinned`).
    pub kind_id: i32,
}

/// Heap-allocated executable metadata handle for `PJRT_ExecutableMetadata_Extension`.
#[repr(C)]
#[derive(Debug)]
pub struct PJRT_ExecutableMetadata_Handle {
    /// C-compatible 16-byte `PJRT_ExecutableMetadata` header (`serialized_metadata`, `serialized_metadata_size`).
    pub raw: crate::extensions::PJRT_ExecutableMetadata,
    /// Owned serialized `PjRtExecutableMetadataProto` wire bytes.
    pub serialized_bytes: Vec<u8>,
}

/// Heap-allocated raw buffer alias handle for `PJRT_RawBuffer_Extension` (`PJRT_RawBuffer`).
#[repr(C)]
#[derive(Debug)]
pub struct PJRT_RawBuffer {
    /// Target `PJRT_Buffer*` being aliased.
    pub buffer_ptr: *mut PJRT_Buffer,
    /// Device HBM/VMEM base address.
    pub device_address: libtpu_core::types::DeviceAddress,
    /// On-device size in bytes.
    pub byte_size: libtpu_core::types::ByteSize,
    /// Owning `PJRT_Memory*` handle.
    pub memory_ptr: *mut PJRT_Memory,
}

/// Heap-allocated ABI version handle for `PJRT_AbiVersion_Extension` (`PJRT_AbiVersion_Holder`).
#[repr(C)]
#[derive(Debug)]
pub struct PJRT_AbiVersion_Holder {
    /// Numeric 64-bit platform ID (`TPU_PLATFORM_ID`).
    pub platform_id: u64,
    /// Runtime or executable ABI version number.
    pub version: i32,
    /// Owned serialized protobuf bytes (`RuntimeAbiVersionProto` or `ExecutableAbiVersionProto`).
    pub proto_bytes: Vec<u8>,
}

/// Heap-allocated phase compiler handle for `PJRT_PhaseCompile_Extension` (`PJRT_PhaseCompile_Compiler`).
#[repr(C)]
#[derive(Debug)]
pub struct PJRT_PhaseCompile_Compiler {
    /// Unique compiler instance ID.
    pub compiler_id: u64,
    /// Number of phases executed on this compiler instance.
    pub phases_executed: usize,
}

/// Heap-allocated profiler instance handle for `PLUGIN_Profiler_Api` (`PLUGIN_Profiler_Instance`).
#[repr(C)]
#[derive(Debug)]
pub struct PLUGIN_Profiler_Instance {
    /// Independent core profiler state.
    pub inner: libtpu_core::driver::TpuCoreProfiler,
}

/// Program descriptor passed to `PJRT_Client_Compile` (`PJRT_Program`).
#[repr(C)]
#[derive(Debug)]
pub struct PJRT_Program {
    /// Size of this struct.
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Pointer to program code bytes.
    pub code: *const c_char,
    /// Length of `code` in bytes.
    pub code_size: usize,
    /// Pointer to format string (`"mlir"` or `"hlo"`).
    pub format: *const c_char,
    /// Length of `format` in bytes.
    pub format_size: usize,
}

/// Opaque C handle representing a `PJRT_Stream`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Stream {
    /// Stream identifier.
    pub stream_id: u64,
}

/// Opaque C handle representing a `PJRT_AsyncTrackingEvent`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_AsyncTrackingEvent {
    /// Event identifier.
    pub event_id: u64,
}

/// Opaque C handle representing a `PJRT_SerializedExecutable`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_SerializedExecutable {
    /// Pointer to serialized bytes.
    pub ptr: *mut u8,
    /// Number of serialized bytes.
    pub len: usize,
}

/// Opaque C handle representing a `PJRT_SerializedTopology`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_SerializedTopology {
    /// Pointer to serialized bytes.
    pub ptr: *mut u8,
    /// Number of serialized bytes.
    pub len: usize,
}

/// Key-value attribute structure (`PJRT_NamedValue`, `40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_NamedValue {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Attribute name pointer.
    pub name: *const c_char,
    /// Attribute name byte length.
    pub name_size: usize,
    /// Value type discriminant (`i32`) and padding (`i32`).
    pub value_type_and_pad: u64,
    /// Union value payload (`int64`, `float`, `bool`, or string/array pointer).
    pub value_payload: usize,
    /// Value array size.
    pub value_size: usize,
}

/// Send callback descriptor (`PJRT_SendCallbackInfo`, `32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_SendCallbackInfo {
    /// Channel identifier.
    pub channel_id: i64,
    /// User callback context pointer.
    pub user_arg: *mut std::ffi::c_void,
    /// Send callback function pointer.
    pub send_callback: *const std::ffi::c_void,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
}

/// Receive callback descriptor (`PJRT_RecvCallbackInfo`, `32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_RecvCallbackInfo {
    /// Channel identifier.
    pub channel_id: i64,
    /// User callback context pointer.
    pub user_arg: *mut std::ffi::c_void,
    /// Receive callback function pointer.
    pub recv_callback: *const std::ffi::c_void,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
}

/// Execution options passed to `PJRT_LoadedExecutable_Execute` (`PJRT_ExecuteOptions`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_ExecuteOptions {
    /// Size of this struct.
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Send callbacks per device.
    pub send_callbacks: *const *const PJRT_SendCallbackInfo,
    /// Receive callbacks per device.
    pub recv_callbacks: *const *const PJRT_RecvCallbackInfo,
    /// Number of send callbacks per device.
    pub num_send_ops: usize,
    /// Number of receive callbacks per device.
    pub num_recv_ops: usize,
    /// Launch identifier (`i32`) and pad (`i32`).
    pub launch_id: i32,
    /// Explicit padding.
    pub pad: i32,
    /// Non-donatable input indices pointer.
    pub non_donatable_input_indices: *const i64,
    /// Number of non-donatable input indices.
    pub num_non_donatable_input_indices: usize,
    /// Optional execution context handle.
    pub context: *mut PJRT_ExecuteContext,
}

/// Memory layout descriptor (`PJRT_Buffer_MemoryLayout`, `48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Buffer_MemoryLayout {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Minor-to-major dimension ordering or byte strides pointer.
    pub minor_to_major: *const i64,
    /// Number of minor-to-major dimensions.
    pub minor_to_major_size: usize,
    /// Tile dimensions pointer.
    pub tile_dims: *const i64,
    /// Number of tile dimensions (`usize`) and layout type (`i32`).
    pub tile_dim_sizes: *const usize,
}

/// Transfer metadata structure (`PJRT_TransferMetadata`, `24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TransferMetadata {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Metadata flags.
    pub flags: u64,
}

/// Logical `(replica, partition)` device ID pair (`PJRT_LogicalDeviceIds`, `8` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_LogicalDeviceIds {
    /// Replica index.
    pub replica: i32,
    /// Partition index.
    pub partition: i32,
}

/// Shape specification for async H2D transfer (`PJRT_ShapeSpec`, `40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_ShapeSpec {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Dimensions array pointer.
    pub dims: *const i64,
    /// Number of dimensions.
    pub num_dims: usize,
    /// Element type (`i32`) and padding (`i32`).
    pub element_type: i32,
    /// Explicit padding.
    pub pad: i32,
}

/// Host chunk passed to `PJRT_CopyToDeviceStream_AddChunk` (`PJRT_Chunk`, `32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Chunk {
    /// Pointer to chunk data bytes.
    pub data: *mut std::ffi::c_void,
    /// Size of chunk in bytes.
    pub size: usize,
    /// Optional deleter callback (`void (*)(void* data, void* deleter_arg)`).
    pub deleter: Option<unsafe extern "C" fn(*mut std::ffi::c_void, *mut std::ffi::c_void)>,
    /// Optional deleter user argument.
    pub deleter_arg: *mut std::ffi::c_void,
}

/// Multi-host process status descriptor (`PJRT_ProcessInfo`, `32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_ProcessInfo {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Task index.
    pub task_id: i32,
    /// Process state discriminant.
    pub state: i32,
    /// Incarnation ID.
    pub incarnation_id: u64,
    /// Error code (`i32`) and pad (`i32`).
    pub error_code: i64,
}

/// Trait implemented by every `PJRT_*_Args` C ABI struct providing its exact minimum valid
/// `struct_size` (`PJRT_STRUCT_SIZE(type, last_field)`) as disassembled from `libtpu.so`.
pub trait PjrtArgsStruct: Sized {
    /// Minimum valid `struct_size` accepted by `PJRT_CheckMatchingStructSizes` in `libtpu.so`.
    const MIN_STRUCT_SIZE: usize = size_of::<Self>();
}

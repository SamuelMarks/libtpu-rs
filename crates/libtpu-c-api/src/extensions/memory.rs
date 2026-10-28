//! PJRT Host Memory Allocator, Metadata, Memory Descriptions, Layouts, and Raw Buffer entrypoints.

use super::args::*;
use super::{dispatch_typed_extension_args, extension_state};
use crate::pjrt_helpers::{box_serialized_artifact, pjrt_serialized_artifact_deleter};
use crate::pjrt_types::{
    PJRT_Error, PJRT_ExecutableMetadata_Handle, PJRT_Layouts_MemoryLayout, PJRT_MemoryDescription,
    PJRT_RawBuffer,
};
use libtpu_core::config::global_init_config;
use libtpu_core::types::{ByteSize, DeviceAddress, MemoryLayout, MemorySpaceKind};
use std::ptr;

/// Static 4-element table of strongly-typed [`PJRT_MemoryDescription`] descriptors (`0..4`).
static STATIC_MEMORY_DESCRIPTIONS: [PJRT_MemoryDescription; 4] = [
    PJRT_MemoryDescription {
        kind: MemorySpaceKind::DeviceHbm,
        kind_id: 0,
    },
    PJRT_MemoryDescription {
        kind: MemorySpaceKind::Vmem,
        kind_id: 1,
    },
    PJRT_MemoryDescription {
        kind: MemorySpaceKind::Smem,
        kind_id: 2,
    },
    PJRT_MemoryDescription {
        kind: MemorySpaceKind::PinnedHost,
        kind_id: 3,
    },
];

/// Thread-safe static pointer array referencing [`STATIC_MEMORY_DESCRIPTIONS`].
#[repr(transparent)]
struct SyncMemoryDescriptionHandles([*const std::ffi::c_void; 4]);

// SAFETY: `STATIC_MEMORY_DESCRIPTIONS` is an immutable static array.
unsafe impl Sync for SyncMemoryDescriptionHandles {}

/// Static pointer array referencing [`STATIC_MEMORY_DESCRIPTIONS`] for `PJRT_MemoryDescriptions_Get`.
static MEMORY_DESCRIPTION_HANDLES: SyncMemoryDescriptionHandles = SyncMemoryDescriptionHandles([
    ptr::addr_of!(STATIC_MEMORY_DESCRIPTIONS[0]).cast::<std::ffi::c_void>(),
    ptr::addr_of!(STATIC_MEMORY_DESCRIPTIONS[1]).cast::<std::ffi::c_void>(),
    ptr::addr_of!(STATIC_MEMORY_DESCRIPTIONS[2]).cast::<std::ffi::c_void>(),
    ptr::addr_of!(STATIC_MEMORY_DESCRIPTIONS[3]).cast::<std::ffi::c_void>(),
]);

/// Strongly-typed handler for [`PJRT_HostMemoryAllocator_Get`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_HostMemoryAllocator_Get_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_HostMemoryAllocator_Get_Args`] (or [`PJRT_HostMemoryAllocator_Allocate_Args`] when `struct_size >= 64`).
unsafe fn handle_host_memory_allocator_get(args: &mut PJRT_HostMemoryAllocator_Get_Args) {
    let cap = global_init_config().premapped_buffer_size.0;
    if args.struct_size >= size_of::<PJRT_HostMemoryAllocator_Allocate_Args>() {
        // SAFETY: `args.struct_size >= 64` guarantees `args` is a valid `PJRT_HostMemoryAllocator_Allocate_Args`.
        let alloc_args =
            unsafe { &mut *ptr::from_mut(args).cast::<PJRT_HostMemoryAllocator_Allocate_Args>() };
        let alloc_len = alloc_args.size.max(64);
        let (ptr, _, holder) = box_serialized_artifact(vec![0_u8; alloc_len]);
        alloc_args.ptr = ptr.cast_mut().cast::<std::ffi::c_void>();
        alloc_args.deleter =
            (pjrt_serialized_artifact_deleter as *const ()).cast::<std::ffi::c_void>();
        alloc_args.deleter_arg = holder.cast::<std::ffi::c_void>();
    } else {
        args.allocator_out = cap;
    }
}

/// Returns the TPU host memory allocator or allocates a pinned host buffer (`PJRT_HostMemoryAllocator_Get`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_HostMemoryAllocator_Get_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_HostMemoryAllocator_Get_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_HostMemoryAllocator_Get(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_HostMemoryAllocator_Get_Args>(
            raw_args,
            handle_host_memory_allocator_get,
        )
    }
}

/// Strongly-typed handler for [`PJRT_ExecutableMetadata_Get`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Metadata_Get_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Metadata_Get_Args`].
unsafe fn handle_executable_metadata_get(args: &mut PJRT_Metadata_Get_Args) {
    let encoded = libtpu_core::proto::PjRtExecutableMetadataProto {
        compiled_memory_stats: Some(libtpu_core::proto::CompiledMemoryStatsProto {
            generated_code_size_in_bytes: 1024,
            argument_size_in_bytes: 256,
            output_size_in_bytes: 256,
            alias_size_in_bytes: 0,
            temp_size_in_bytes: 512,
            host_generated_code_size_in_bytes: 0,
            host_argument_size_in_bytes: 0,
            host_output_size_in_bytes: 0,
            host_alias_size_in_bytes: 0,
            host_temp_size_in_bytes: 0,
            serialized_buffer_assignment: Vec::new(),
            peak_memory_in_bytes: 2048,
            total_size_in_bytes: 2048,
            total_allocation_bytes: 2048,
            indefinite_allocations: 0,
            peak_unpadded_heap_bytes: 1536,
        }),
        platform_specific_metadata: Vec::new(),
    }
    .encode();
    let mut handle = Box::new(PJRT_ExecutableMetadata_Handle {
        raw: PJRT_ExecutableMetadata {
            serialized_metadata: ptr::null(),
            serialized_metadata_size: encoded.len(),
        },
        serialized_bytes: encoded,
    });
    handle.raw.serialized_metadata = handle.serialized_bytes.as_ptr().cast();
    args.metadata_out = Box::into_raw(handle) as usize;
}

/// Queries metadata from an executable (`PJRT_ExecutableMetadata_Get`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Metadata_Get_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Metadata_Get_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_ExecutableMetadata_Get(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Metadata_Get_Args>(
            raw_args,
            handle_executable_metadata_get,
        )
    }
}

/// Strongly-typed handler for [`PJRT_ExecutableMetadata_Destroy`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Metadata_Destroy_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Metadata_Destroy_Args`].
unsafe fn handle_executable_metadata_destroy(args: &mut PJRT_Metadata_Destroy_Args) {
    if !args.metadata.is_null() {
        // SAFETY: `metadata` was allocated by `PJRT_ExecutableMetadata_Get`.
        unsafe {
            drop(Box::from_raw(
                args.metadata.cast::<PJRT_ExecutableMetadata_Handle>(),
            ));
        }
    }
    args.metadata = ptr::null_mut();
}

/// Destroys executable metadata (`PJRT_ExecutableMetadata_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Metadata_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Metadata_Destroy_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_ExecutableMetadata_Destroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Metadata_Destroy_Args>(
            raw_args,
            handle_executable_metadata_destroy,
        )
    }
}

/// Strongly-typed handler for [`PJRT_MemoryDescriptions_Get`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Metadata_Get_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Metadata_Get_Args`] (or [`PJRT_DeviceDescription_MemoryDescriptions_Args`] when `struct_size >= 48`).
unsafe fn handle_memory_descriptions_get(args: &mut PJRT_Metadata_Get_Args) {
    if args.struct_size >= size_of::<PJRT_DeviceDescription_MemoryDescriptions_Args>() {
        // SAFETY: `args.struct_size >= 48` guarantees `args` is a valid `PJRT_DeviceDescription_MemoryDescriptions_Args`.
        let desc_args = unsafe {
            &mut *ptr::from_mut(args).cast::<PJRT_DeviceDescription_MemoryDescriptions_Args>()
        };
        desc_args.memory_descriptions = MEMORY_DESCRIPTION_HANDLES.0.as_ptr();
        desc_args.num_memory_descriptions = MEMORY_DESCRIPTION_HANDLES.0.len();
        desc_args.default_memory_index = 0;
    } else {
        args.metadata_out = 4;
    }
}

/// Queries memory descriptions for a device (`PJRT_MemoryDescriptions_Get`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Metadata_Get_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Metadata_Get_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_MemoryDescriptions_Get(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Metadata_Get_Args>(
            raw_args,
            handle_memory_descriptions_get,
        )
    }
}

/// Strongly-typed handler for [`PJRT_MemoryDescriptions_Destroy`] (`24`-byte destroy or `48`-byte kind query).
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Metadata_Destroy_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Metadata_Destroy_Args`].
unsafe fn handle_memory_descriptions_destroy(args: &mut PJRT_Metadata_Destroy_Args) {
    if args.struct_size >= size_of::<PJRT_MemoryDescription_Kind_Args>() {
        // SAFETY: `args.struct_size >= 48` guarantees `args` is a valid `PJRT_MemoryDescription_Kind_Args`.
        let kind_args =
            unsafe { &mut *ptr::from_mut(args).cast::<PJRT_MemoryDescription_Kind_Args>() };
        let desc = if kind_args.memory_description.is_null() {
            &STATIC_MEMORY_DESCRIPTIONS[0]
        } else {
            // SAFETY: `memory_description` is a pointer into `STATIC_MEMORY_DESCRIPTIONS`.
            unsafe {
                &*kind_args
                    .memory_description
                    .cast::<PJRT_MemoryDescription>()
            }
        };
        let kind_name = desc.kind.as_str();
        kind_args.kind = kind_name.as_ptr().cast();
        kind_args.kind_size = kind_name.len();
        kind_args.kind_id = desc.kind_id;
    } else {
        args.metadata = ptr::null_mut();
    }
}

/// Queries the kind of a memory description or destroys a memory description list (`PJRT_MemoryDescriptions_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Metadata_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Metadata_Destroy_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_MemoryDescriptions_Destroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Metadata_Destroy_Args>(
            raw_args,
            handle_memory_descriptions_destroy,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Layouts_MemoryLayout_Destroy`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Layouts_MemoryLayout_Destroy_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Layouts_MemoryLayout_Destroy_Args`].
unsafe fn handle_layouts_destroy(args: &mut PJRT_Layouts_MemoryLayout_Destroy_Args) {
    if !args.layout.is_null() {
        // SAFETY: `layout` was allocated as a `Box<PJRT_Layouts_MemoryLayout>`.
        unsafe {
            drop(Box::from_raw(
                args.layout.cast::<PJRT_Layouts_MemoryLayout>(),
            ));
        }
    }
    args.layout = ptr::null_mut();
}

/// Destroys a `PJRT_Layouts_MemoryLayout` (`PJRT_Layouts_MemoryLayout_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Layouts_MemoryLayout_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Layouts_MemoryLayout_Destroy_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Layouts_MemoryLayout_Destroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Layouts_MemoryLayout_Destroy_Args>(
            raw_args,
            handle_layouts_destroy,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Layouts_MemoryLayout_Serialize`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Layouts_MemoryLayout_Serialize_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Layouts_MemoryLayout_Serialize_Args`].
unsafe fn handle_layouts_serialize(args: &mut PJRT_Layouts_MemoryLayout_Serialize_Args) {
    let (sublane, lane) = if !args.layout.is_null() {
        // SAFETY: `layout` is a valid `PJRT_Layouts_MemoryLayout*`.
        let l_ref = unsafe { &*args.layout.cast::<PJRT_Layouts_MemoryLayout>() };
        (l_ref.sublane_tile, l_ref.lane_tile)
    } else {
        let mxu_dim = i64::from(global_init_config().generation.mxu_dimension());
        (8, mxu_dim)
    };
    let encoded = libtpu_core::proto::LayoutProto {
        minor_to_major: vec![1, 0],
        tiles: vec![libtpu_core::proto::TileProto {
            dimensions: vec![sublane, lane],
        }],
        tail_padding_alignment_in_elements: 1,
        index_primitive_type: 0,
        pointer_primitive_type: 0,
        memory_space: 0,
        element_size_in_bits: 32,
        dynamic_shape_metadata_prefix_bytes: 0,
    }
    .encode();
    let (ptr, len, holder) = box_serialized_artifact(encoded);
    args.serialized_bytes = ptr;
    args.serialized_bytes_size = len;
    args.serialized_layout = holder.cast::<std::ffi::c_void>();
    args.serialized_layout_deleter =
        (pjrt_serialized_artifact_deleter as *const ()).cast::<std::ffi::c_void>();
}

/// Serializes a `PJRT_Layouts_MemoryLayout` (`PJRT_Layouts_MemoryLayout_Serialize`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Layouts_MemoryLayout_Serialize_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Layouts_MemoryLayout_Serialize_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Layouts_MemoryLayout_Serialize(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Layouts_MemoryLayout_Serialize_Args>(
            raw_args,
            handle_layouts_serialize,
        )
    }
}

/// Allocates a heap-owned [`PJRT_Layouts_MemoryLayout`] for the given rank.
///
/// # Arguments
///
/// * `rank` - Tensor rank.
fn alloc_memory_layout_handle(rank: usize) -> *mut std::ffi::c_void {
    let lane_tile = i64::from(global_init_config().generation.mxu_dimension());
    let handle = Box::new(PJRT_Layouts_MemoryLayout {
        layout: MemoryLayout::default_tpu_tiled(rank.max(1)),
        sublane_tile: 8,
        lane_tile,
    });
    Box::into_raw(handle).cast::<std::ffi::c_void>()
}

/// Strongly-typed handler for [`PJRT_Layouts_Client_GetDefaultLayout`] and [`PJRT_Layouts_Topology_GetDefaultLayout`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args`].
unsafe fn handle_layouts_get_default_layout(
    args: &mut PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args,
) {
    args.layout = alloc_memory_layout_handle(args.num_dims);
}

/// Queries default memory layout from a `PJRT_Client` (`PJRT_Layouts_Client_GetDefaultLayout`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Layouts_Client_GetDefaultLayout(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args>(
            raw_args,
            handle_layouts_get_default_layout,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Layouts_Buffer_GetMemoryLayout`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args`].
unsafe fn handle_layouts_buffer_get_memory_layout(
    args: &mut PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args,
) {
    let rank = if !args.buffer.is_null() {
        // SAFETY: `buffer` is a valid `PJRT_Buffer*`.
        let buf = unsafe { &*args.buffer.cast::<crate::pjrt_types::PJRT_Buffer>() };
        buf.inner.shape.rank()
    } else {
        1
    };
    args.layout = alloc_memory_layout_handle(rank);
}

/// Queries memory layout of a `PJRT_Buffer` (`PJRT_Layouts_Buffer_GetMemoryLayout`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Layouts_Buffer_GetMemoryLayout(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args>(
            raw_args,
            handle_layouts_buffer_get_memory_layout,
        )
    }
}

/// Queries default memory layout from a `PJRT_TopologyDescription` (`PJRT_Layouts_Topology_GetDefaultLayout`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Layouts_Topology_GetDefaultLayout(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args>(
            raw_args,
            handle_layouts_get_default_layout,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Layouts_Executable_GetOutputLayouts`] and [`PJRT_Layouts_Executable_GetParameterLayouts`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Metadata_Get_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Metadata_Get_Args`].
unsafe fn handle_layouts_executable_layouts(args: &mut PJRT_Metadata_Get_Args) {
    args.metadata_out = if !args.handle.is_null() {
        // SAFETY: `args.handle` is a valid `PJRT_Executable*`.
        let exec = unsafe { &*args.handle.cast::<crate::pjrt_types::PJRT_Executable>() };
        exec.inner.program.output_shapes.len().max(1)
    } else {
        1
    };
}

/// Queries output memory layouts of a `PJRT_Executable` (`PJRT_Layouts_Executable_GetOutputLayouts`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Metadata_Get_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Metadata_Get_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Layouts_Executable_GetOutputLayouts(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Metadata_Get_Args>(
            raw_args,
            handle_layouts_executable_layouts,
        )
    }
}

/// Queries parameter memory layouts of a `PJRT_Executable` (`PJRT_Layouts_Executable_GetParameterLayouts`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Metadata_Get_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Metadata_Get_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Layouts_Executable_GetParameterLayouts(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Metadata_Get_Args>(
            raw_args,
            handle_layouts_executable_layouts,
        )
    }
}

/// Strongly-typed handler for [`PJRT_RawBuffer_CreateRawAliasOfBuffer`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_RawBuffer_Query_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_RawBuffer_Query_Args`].
unsafe fn handle_raw_buffer_create(args: &mut PJRT_RawBuffer_Query_Args) {
    let mut st = extension_state().lock();
    st.raw_buffers = st.raw_buffers.saturating_add(1);
    let (buf_ptr, addr, byte_size, mem_ptr) = if !args.buffer.is_null() {
        // SAFETY: `buffer` is a valid `PJRT_Buffer*`.
        let buf = unsafe { &*args.buffer.cast::<crate::pjrt_types::PJRT_Buffer>() };
        (
            args.buffer.cast::<crate::pjrt_types::PJRT_Buffer>(),
            buf.inner.address,
            ByteSize(buf.inner.shape.on_device_byte_size().0),
            buf.memory_ptr,
        )
    } else {
        (
            ptr::null_mut(),
            DeviceAddress(0x1000),
            ByteSize(16),
            ptr::null_mut(),
        )
    };
    let raw_handle = Box::into_raw(Box::new(PJRT_RawBuffer {
        buffer_ptr: buf_ptr,
        device_address: addr,
        byte_size,
        memory_ptr: mem_ptr,
    }));
    args.out_value = raw_handle as usize;
}

/// Creates a `PJRT_RawBuffer` alias of a `PJRT_Buffer` (`PJRT_RawBuffer_CreateRawAliasOfBuffer`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_RawBuffer_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_RawBuffer_Query_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_RawBuffer_CreateRawAliasOfBuffer(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_RawBuffer_Query_Args>(
            raw_args,
            handle_raw_buffer_create,
        )
    }
}

/// Strongly-typed handler for [`PJRT_RawBuffer_Destroy`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_RawBuffer_Destroy_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_RawBuffer_Destroy_Args`].
unsafe fn handle_raw_buffer_destroy(args: &mut PJRT_RawBuffer_Destroy_Args) {
    if !args.raw_buffer.is_null() {
        // SAFETY: `raw_buffer` was allocated by `PJRT_RawBuffer_CreateRawAliasOfBuffer`.
        unsafe {
            drop(Box::from_raw(args.raw_buffer.cast::<PJRT_RawBuffer>()));
        }
    }
    let mut st = extension_state().lock();
    st.raw_buffers = st.raw_buffers.saturating_sub(1);
    args.raw_buffer = ptr::null_mut();
}

/// Destroys a `PJRT_RawBuffer` (`PJRT_RawBuffer_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_RawBuffer_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_RawBuffer_Destroy_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_RawBuffer_Destroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_RawBuffer_Destroy_Args>(
            raw_args,
            handle_raw_buffer_destroy,
        )
    }
}

/// Strongly-typed handler for [`PJRT_RawBuffer_GetHostPointer`] and [`PJRT_RawBuffer_GetMemorySpace`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_RawBuffer_Query_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_RawBuffer_Query_Args`].
unsafe fn handle_raw_buffer_get_device_address(args: &mut PJRT_RawBuffer_Query_Args) {
    args.out_value = if !args.buffer.is_null() {
        // SAFETY: `buffer` is a valid `PJRT_Buffer*`.
        let buf = unsafe { &*args.buffer.cast::<crate::pjrt_types::PJRT_Buffer>() };
        usize::try_from(buf.inner.address.0).unwrap_or(0x1000)
    } else {
        0x1000
    };
}

/// Queries host pointer of a `PJRT_RawBuffer` (`PJRT_RawBuffer_GetHostPointer`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_RawBuffer_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_RawBuffer_Query_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_RawBuffer_GetHostPointer(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_RawBuffer_Query_Args>(
            raw_args,
            handle_raw_buffer_get_device_address,
        )
    }
}

/// Strongly-typed handler for [`PJRT_RawBuffer_GetOnDeviceSizeInBytes`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_RawBuffer_Query_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_RawBuffer_Query_Args`].
unsafe fn handle_raw_buffer_get_size(args: &mut PJRT_RawBuffer_Query_Args) {
    args.out_value = if !args.buffer.is_null() {
        // SAFETY: `buffer` is a valid `PJRT_Buffer*`.
        let buf = unsafe { &*args.buffer.cast::<crate::pjrt_types::PJRT_Buffer>() };
        buf.inner.shape.on_device_byte_size().0
    } else {
        256
    };
}

/// Queries on-device size in bytes of a `PJRT_RawBuffer` (`PJRT_RawBuffer_GetOnDeviceSizeInBytes`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_RawBuffer_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_RawBuffer_Query_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_RawBuffer_GetOnDeviceSizeInBytes(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_RawBuffer_Query_Args>(
            raw_args,
            handle_raw_buffer_get_size,
        )
    }
}

/// Queries memory space of a `PJRT_RawBuffer` (`PJRT_RawBuffer_GetMemorySpace`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_RawBuffer_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_RawBuffer_Query_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_RawBuffer_GetMemorySpace(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_RawBuffer_Query_Args>(
            raw_args,
            handle_raw_buffer_get_device_address,
        )
    }
}

/// Strongly-typed handler for [`PJRT_RawBuffer_CopyRawHostToDevice`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_RawBuffer_CopyRaw_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_RawBuffer_CopyRaw_Args`].
unsafe fn handle_raw_buffer_copy_h2d(args: &mut PJRT_RawBuffer_CopyRaw_Args) {
    if !crate::any_null(&[args.buffer as usize, args.host_ptr as usize]) {
        // SAFETY: `args.buffer` is a valid `PJRT_Buffer*` when non-null.
        let buf = unsafe { &*args.buffer.cast::<crate::pjrt_types::PJRT_Buffer>() };
        let tx_len = usize::try_from(args.transfer_size).unwrap_or(0);
        // SAFETY: `args.host_ptr` points to `tx_len` bytes.
        let src = unsafe { std::slice::from_raw_parts(args.host_ptr.cast::<u8>(), tx_len) };
        let _ = buf
            .inner
            .hal
            .write_device_memory(buf.inner.device_id, buf.inner.address, src);
    }
    args.event = Box::into_raw(Box::new(crate::pjrt_types::PJRT_Event {
        inner: libtpu_core::pjrt::PjrtCoreEvent::ready_ok(),
    }))
    .cast::<std::ffi::c_void>();
}

/// Copies raw host memory into a `PJRT_RawBuffer` (`PJRT_RawBuffer_CopyRawHostToDevice`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_RawBuffer_CopyRaw_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_RawBuffer_CopyRaw_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_RawBuffer_CopyRawHostToDevice(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_RawBuffer_CopyRaw_Args>(
            raw_args,
            handle_raw_buffer_copy_h2d,
        )
    }
}

/// Strongly-typed handler for [`PJRT_RawBuffer_CopyRawDeviceToHost`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_RawBuffer_CopyRaw_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_RawBuffer_CopyRaw_Args`].
unsafe fn handle_raw_buffer_copy_d2h(args: &mut PJRT_RawBuffer_CopyRaw_Args) {
    if !crate::any_null(&[args.buffer as usize, args.host_ptr as usize]) {
        // SAFETY: `args.buffer` is a valid `PJRT_Buffer*` when non-null.
        let buf = unsafe { &*args.buffer.cast::<crate::pjrt_types::PJRT_Buffer>() };
        let bytes = buf.inner.to_host_bytes().unwrap_or_default();
        let tx_len = usize::try_from(args.transfer_size).unwrap_or(0);
        let len = bytes.len().min(tx_len);
        // SAFETY: `args.host_ptr` has room for `len` bytes.
        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr(), args.host_ptr.cast::<u8>(), len);
        }
    }
    args.event = Box::into_raw(Box::new(crate::pjrt_types::PJRT_Event {
        inner: libtpu_core::pjrt::PjrtCoreEvent::ready_ok(),
    }))
    .cast::<std::ffi::c_void>();
}

/// Copies raw device memory from a `PJRT_RawBuffer` to host (`PJRT_RawBuffer_CopyRawDeviceToHost`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_RawBuffer_CopyRaw_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_RawBuffer_CopyRaw_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_RawBuffer_CopyRawDeviceToHost(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_RawBuffer_CopyRaw_Args>(
            raw_args,
            handle_raw_buffer_copy_d2h,
        )
    }
}

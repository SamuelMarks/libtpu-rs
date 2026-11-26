//! PJRT HostMemoryAllocator, ExecutableMetadata, MemoryDescriptions, Layouts, and RawBuffer entrypoints.

use super::args::*;
use super::{dispatch_typed_extension_args, dispatch_typed_fallible_extension_args};
use crate::pjrt_helpers::{
    borrow_live_handle, box_ready_event, box_registered_handle, box_serialized_artifact,
    pjrt_serialized_artifact_deleter, take_live_handle,
};
use crate::pjrt_types::{
    PJRT_Error, PJRT_Layouts_MemoryLayout, PJRT_MemoryDescription, PJRT_RawBuffer,
};
use libtpu_core::config::global_init_config;
use libtpu_core::error::LibTpuError;
use libtpu_core::proto::ProtoSchemaMessage;
use libtpu_core::types::{
    DeviceAddress, MemoryLayout, MemorySpaceKind, TensorShape, TpuGeneration,
};
use std::collections::BTreeMap;
use std::ptr;
use std::sync::OnceLock;

/// 64-byte cache-line-aligned host DMA block used by [`PJRT_HostMemoryAllocator_Allocate`].
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct AlignedHostBlock([u8; 64]);

/// Registry mapping host-allocated pointers from [`PJRT_HostMemoryAllocator_Allocate`] to owned aligned blocks.
static HOST_MEMORY_ALLOCS: OnceLock<parking_lot::Mutex<BTreeMap<usize, Box<[AlignedHostBlock]>>>> =
    OnceLock::new();

/// Returns a reference to [`HOST_MEMORY_ALLOCS`].
fn host_memory_allocs() -> &'static parking_lot::Mutex<BTreeMap<usize, Box<[AlignedHostBlock]>>> {
    HOST_MEMORY_ALLOCS.get_or_init(|| parking_lot::Mutex::new(BTreeMap::new()))
}

/// Static memory description descriptors (`HBM`, `VMEM`, `SMEM`, `PinnedHost`, `CMEM`).
static STATIC_MEMORY_DESCRIPTIONS: [PJRT_MemoryDescription; 5] = [
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
    PJRT_MemoryDescription {
        kind: MemorySpaceKind::Cmem,
        kind_id: 4,
    },
];

/// Thread-safe static pointer array wrapper for [`PJRT_MemoryDescription`] tables.
#[repr(transparent)]
struct SyncMemDescPtrs<const N: usize>([*const PJRT_MemoryDescription; N]);

// SAFETY: Points only to immutable static `STATIC_MEMORY_DESCRIPTIONS` entries.
unsafe impl<const N: usize> Sync for SyncMemDescPtrs<N> {}

/// Memory descriptions for SparseCore-equipped generations (`V5p`, `V6e`, `V7x`: 4 spaces).
static MEM_DESC_SPARSECORE: SyncMemDescPtrs<4> = SyncMemDescPtrs([
    ptr::addr_of!(STATIC_MEMORY_DESCRIPTIONS[0]),
    ptr::addr_of!(STATIC_MEMORY_DESCRIPTIONS[1]),
    ptr::addr_of!(STATIC_MEMORY_DESCRIPTIONS[2]),
    ptr::addr_of!(STATIC_MEMORY_DESCRIPTIONS[3]),
]);

/// Memory descriptions for TPU v4 (`HBM`, `VMEM`, `SMEM`, `PinnedHost`, `CMEM`: 5 spaces).
static MEM_DESC_V4_CMEM: SyncMemDescPtrs<5> = SyncMemDescPtrs([
    ptr::addr_of!(STATIC_MEMORY_DESCRIPTIONS[0]),
    ptr::addr_of!(STATIC_MEMORY_DESCRIPTIONS[1]),
    ptr::addr_of!(STATIC_MEMORY_DESCRIPTIONS[2]),
    ptr::addr_of!(STATIC_MEMORY_DESCRIPTIONS[3]),
    ptr::addr_of!(STATIC_MEMORY_DESCRIPTIONS[4]),
]);

/// Memory descriptions for compact TensorCore-only generations (`V5e`: `HBM`, `VMEM`, `PinnedHost`: 3 spaces).
static MEM_DESC_COMPACT: SyncMemDescPtrs<3> = SyncMemDescPtrs([
    ptr::addr_of!(STATIC_MEMORY_DESCRIPTIONS[0]),
    ptr::addr_of!(STATIC_MEMORY_DESCRIPTIONS[1]),
    ptr::addr_of!(STATIC_MEMORY_DESCRIPTIONS[3]),
]);

/// C deleter callback freeing aligned host memory allocated by [`PJRT_HostMemoryAllocator_Allocate`].
///
/// # Arguments
///
/// * `deleter_arg` - Host pointer returned by [`PJRT_HostMemoryAllocator_Allocate`].
///
/// # Safety
///
/// `deleter_arg` must be null or a pointer previously registered in `HOST_MEMORY_ALLOCS`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pjrt_host_memory_deleter(deleter_arg: *mut std::ffi::c_void) {
    crate::ffi_catch_unwind((), || {
        if !deleter_arg.is_null() {
            drop(host_memory_allocs().lock().remove(&(deleter_arg as usize)));
        }
    });
}

/// Strongly-typed handler for [`PJRT_HostMemoryAllocator_Get`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_HostMemoryAllocator_Get_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_HostMemoryAllocator_Get_Args`].
unsafe fn handle_host_memory_allocator_get(args: &mut PJRT_HostMemoryAllocator_Get_Args) {
    args.allocator_out = global_init_config().generation.hbm_capacity_bytes().0;
}

/// Queries the host memory allocator from a `PJRT_Client` (`PJRT_HostMemoryAllocator_Get`).
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

/// Strongly-typed handler for [`PJRT_HostMemoryAllocator_Allocate`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_HostMemoryAllocator_Allocate_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_HostMemoryAllocator_Allocate_Args`].
unsafe fn handle_host_memory_allocator_allocate(args: &mut PJRT_HostMemoryAllocator_Allocate_Args) {
    let max_bytes = usize::try_from(isize::MAX).unwrap_or(usize::MAX / 2) - 64;
    if args.size <= max_bytes {
        let blocks = args.size.max(64).div_ceil(64);
        let mut owned = vec![AlignedHostBlock([0_u8; 64]); blocks].into_boxed_slice();
        let raw_ptr = owned.as_mut_ptr().cast::<std::ffi::c_void>();
        host_memory_allocs().lock().insert(raw_ptr as usize, owned);
        args.ptr = raw_ptr;
        args.deleter = Some(pjrt_host_memory_deleter);
        args.deleter_arg = raw_ptr;
    }
}

/// Allocates 64-byte aligned DMA-ready host memory (`PJRT_HostMemoryAllocator_Allocate`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_HostMemoryAllocator_Allocate_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_HostMemoryAllocator_Allocate_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_HostMemoryAllocator_Allocate(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_HostMemoryAllocator_Allocate_Args>(
            raw_args,
            handle_host_memory_allocator_allocate,
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
/// If `args.executable` is non-null, it must point to a valid [`crate::pjrt_types::PJRT_Executable`].
unsafe fn handle_executable_metadata_get(args: &mut PJRT_Metadata_Get_Args) {
    // SAFETY: `args.executable` is null or a valid `PJRT_Executable*`.
    let (stats_proto, platform_meta) = unsafe { args.executable.as_ref() }.map_or_else(
        || {
            (
                libtpu_core::proto::CompiledMemoryStatsProto {
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
                },
                Vec::new(),
            )
        },
        |exec_ref| {
            let ms = exec_ref.inner.compiled_memory_stats();
            let ba_bytes = exec_ref
                .inner
                .program
                .buffer_assignment_proto
                .encode_proto();
            let target_args = {
                use libtpu_core::proto::TpuGenerationProtoExt as _;
                libtpu_core::proto::TargetArgumentsProto::from_generation(
                    global_init_config().generation,
                )
                .encode_proto()
            };
            let peak_mem = ms
                .argument_size_in_bytes
                .saturating_add(ms.output_size_in_bytes)
                .saturating_add(ms.temp_size_in_bytes);
            let total_size = peak_mem.saturating_add(ms.generated_code_size_in_bytes);
            (
                libtpu_core::proto::CompiledMemoryStatsProto {
                    generated_code_size_in_bytes: ms.generated_code_size_in_bytes,
                    argument_size_in_bytes: ms.argument_size_in_bytes,
                    output_size_in_bytes: ms.output_size_in_bytes,
                    alias_size_in_bytes: ms.alias_size_in_bytes,
                    temp_size_in_bytes: ms.temp_size_in_bytes,
                    host_generated_code_size_in_bytes: 0,
                    host_argument_size_in_bytes: 0,
                    host_output_size_in_bytes: 0,
                    host_alias_size_in_bytes: 0,
                    host_temp_size_in_bytes: 0,
                    serialized_buffer_assignment: ba_bytes,
                    peak_memory_in_bytes: peak_mem,
                    total_size_in_bytes: total_size,
                    total_allocation_bytes: total_size,
                    indefinite_allocations: ms.argument_size_in_bytes,
                    peak_unpadded_heap_bytes: peak_mem,
                },
                target_args,
            )
        },
    );
    let encoded = libtpu_core::proto::PjRtExecutableMetadataProto {
        compiled_memory_stats: Some(stats_proto),
        platform_specific_metadata: (!platform_meta.is_empty()).then(|| {
            libtpu_core::proto::ProtobufAny {
                type_url: String::from("type.googleapis.com/xla.TargetArgumentsProto"),
                value: platform_meta,
            }
        }),
    }
    .encode_proto();
    let (_, _, holder) = box_serialized_artifact(encoded);
    args.metadata_out = holder;
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
    pjrt_serialized_artifact_deleter(args.metadata.cast::<std::ffi::c_void>());
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

/// Selects the supported [`PJRT_MemoryDescription`] slice for `generation`.
///
/// # Arguments
///
/// * `generation` - Target [`TpuGeneration`].
const fn select_memory_descriptions_for_generation(
    generation: TpuGeneration,
) -> (*const *const PJRT_MemoryDescription, usize) {
    match generation {
        TpuGeneration::V4 => (MEM_DESC_V4_CMEM.0.as_ptr(), MEM_DESC_V4_CMEM.0.len()),
        TpuGeneration::V2 | TpuGeneration::V3 | TpuGeneration::V4Lite | TpuGeneration::V5e => {
            (MEM_DESC_COMPACT.0.as_ptr(), MEM_DESC_COMPACT.0.len())
        }
        TpuGeneration::V5p | TpuGeneration::V6e | TpuGeneration::V7x => {
            (MEM_DESC_SPARSECORE.0.as_ptr(), MEM_DESC_SPARSECORE.0.len())
        }
    }
}

/// Queries memory descriptions for a device (`PJRT_MemoryDescriptions_Get`), filtering returned
/// [`PJRT_MemoryDescription`] handles according to `args.device_description`'s [`TpuGeneration`].
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_DeviceDescription_MemoryDescriptions_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to at least 32 bytes (`PJRT_DeviceDescription_MemoryDescriptions_Args`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_MemoryDescriptions_Get(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    crate::ffi_catch_unwind(ptr::null_mut(), || {
        if raw_args.is_null() {
            return crate::pjrt_helpers::box_pjrt_error(LibTpuError::NullPointer {
                parameter: "args",
            });
        }
        // SAFETY: `raw_args` is non-null; reading leading `usize` `struct_size`.
        let actual_size = unsafe { *raw_args.cast::<usize>() };
        let expected_min = PJRT_DeviceDescription_MemoryDescriptions_Args::MIN_STRUCT_SIZE;
        if actual_size < expected_min {
            return crate::pjrt_helpers::box_pjrt_error(LibTpuError::InvalidStructSize {
                struct_name: "PJRT_DeviceDescription_MemoryDescriptions_Args",
                expected_min,
                actual: actual_size,
            });
        }
        // SAFETY: `actual_size >= 32` covers offsets `0..32` (`struct_size`, `extension_start`, `device_description`, `memory_descriptions`).
        let dev_desc_ptr = unsafe {
            *raw_args
                .byte_add(16)
                .cast::<*const crate::pjrt_types::PJRT_DeviceDescription>()
        };
        // SAFETY: `dev_desc_ptr` is null or a valid `PJRT_DeviceDescription*`.
        let generation = unsafe { dev_desc_ptr.as_ref() }.map_or_else(
            || global_init_config().generation,
            |d| d.inner.spec.generation,
        );
        let (ptrs, count) = select_memory_descriptions_for_generation(generation);
        if actual_size >= 48 {
            // SAFETY: `actual_size >= 48` covers full `PJRT_DeviceDescription_MemoryDescriptions_Args`.
            let full =
                unsafe { &mut *raw_args.cast::<PJRT_DeviceDescription_MemoryDescriptions_Args>() };
            full.memory_descriptions = ptrs;
            full.num_memory_descriptions = count;
            full.default_memory_index = 0;
        } else {
            // SAFETY: Legacy 32-byte query stores count at offset 24.
            unsafe {
                *raw_args.byte_add(24).cast::<usize>() = count;
            }
        }
        ptr::null_mut()
    })
}

/// Strongly-typed handler for [`PJRT_MemoryDescription_Kind`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_MemoryDescription_Kind_Args`] structure.
///
/// # Safety
///
/// If `args.memory_description` is non-null, it must point to a valid [`PJRT_MemoryDescription`].
unsafe fn handle_memory_description_kind(args: &mut PJRT_MemoryDescription_Kind_Args) {
    // SAFETY: `memory_description` is null or a valid `PJRT_MemoryDescription*`.
    let desc =
        unsafe { args.memory_description.as_ref() }.unwrap_or(&STATIC_MEMORY_DESCRIPTIONS[0]);
    let kind_name = desc.kind.as_str();
    args.kind = kind_name.as_ptr().cast();
    args.kind_size = kind_name.len();
    args.kind_id = desc.kind_id;
}

/// Queries the kind and kind ID of a `PJRT_MemoryDescription` (`PJRT_MemoryDescription_Kind`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_MemoryDescription_Kind_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_MemoryDescription_Kind_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_MemoryDescription_Kind(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_MemoryDescription_Kind_Args>(
            raw_args,
            handle_memory_description_kind,
        )
    }
}

/// Queries the kind of a memory description or destroys a memory description list (`PJRT_MemoryDescriptions_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Metadata_Destroy_Args`] (`24` bytes) or [`PJRT_MemoryDescription_Kind_Args`] (`48` bytes).
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to at least 24 bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_MemoryDescriptions_Destroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    crate::ffi_catch_unwind(ptr::null_mut(), || {
        if raw_args.is_null() {
            return crate::pjrt_helpers::box_pjrt_error(LibTpuError::NullPointer {
                parameter: "args",
            });
        }
        // SAFETY: `raw_args` is non-null.
        let actual_size = unsafe { *raw_args.cast::<usize>() };
        if actual_size < size_of::<PJRT_Metadata_Destroy_Args>() {
            return crate::pjrt_helpers::box_pjrt_error(LibTpuError::InvalidStructSize {
                struct_name: "PJRT_Metadata_Destroy_Args",
                expected_min: size_of::<PJRT_Metadata_Destroy_Args>(),
                actual: actual_size,
            });
        }
        if actual_size >= size_of::<PJRT_MemoryDescription_Kind_Args>() {
            // SAFETY: `actual_size >= 48` guarantees `raw_args` covers `PJRT_MemoryDescription_Kind_Args`.
            unsafe {
                handle_memory_description_kind(
                    &mut *raw_args.cast::<PJRT_MemoryDescription_Kind_Args>(),
                );
            }
        } else {
            // SAFETY: `actual_size >= 24` covers `PJRT_Metadata_Destroy_Args`.
            unsafe {
                (*raw_args.cast::<PJRT_Metadata_Destroy_Args>()).metadata = ptr::null_mut();
            }
        }
        ptr::null_mut()
    })
}

/// Allocates a registered heap [`PJRT_Layouts_MemoryLayout`] for a [`TensorShape`] or rank.
///
/// # Arguments
///
/// * `layout` - Strongly-typed [`MemoryLayout`].
pub(crate) fn alloc_registered_memory_layout(
    layout: MemoryLayout,
) -> *mut PJRT_Layouts_MemoryLayout {
    let lane_tile = i64::from(global_init_config().generation.mxu_dimension());
    box_registered_handle(PJRT_Layouts_MemoryLayout {
        layout,
        sublane_tile: 8,
        lane_tile,
    })
}

/// Allocates a registered heap [`PJRT_Layouts_MemoryLayout`] for the given tensor `rank`.
///
/// # Arguments
///
/// * `rank` - Tensor rank.
fn alloc_memory_layout_handle(rank: usize) -> *mut PJRT_Layouts_MemoryLayout {
    alloc_registered_memory_layout(MemoryLayout::default_tpu_tiled(rank))
}

/// Formats a [`MemoryLayout`] into canonical `OpenXLA` HLO layout syntax (`xla::Layout::ToString`).
///
/// Produces strings parseable by `xla::ParseLayout` (`{...}`), such as `"{}"` for 0D scalars,
/// `"{0:T(128)}"` for 1D vectors, `"{1,0:T(8,128)}"` for >=2D tiled tensors, and `"{1,0}"`
/// for untiled strided host layouts.
///
/// # Arguments
///
/// * `layout` - Strongly-typed [`MemoryLayout`] descriptor.
/// * `sublane_tile` - Sublane tile extent (typically `8`).
/// * `lane_tile` - Lane tile extent (typically `128`).
fn format_hlo_memory_layout(layout: &MemoryLayout, sublane_tile: i64, lane_tile: i64) -> String {
    match layout {
        MemoryLayout::Tiled { minor_to_major, .. } => {
            let m2m_str = minor_to_major
                .iter()
                .map(i64::to_string)
                .collect::<Vec<_>>()
                .join(",");
            match minor_to_major.len() {
                0 => String::from("{}"),
                1 => format!("{{{m2m_str}:T({lane_tile})}}"),
                _ => format!("{{{m2m_str}:T({sublane_tile},{lane_tile})}}"),
            }
        }
        MemoryLayout::Strides { byte_strides } => {
            let m2m_str = (0..i64::try_from(byte_strides.len()).unwrap_or(0))
                .rev()
                .map(|idx| idx.to_string())
                .collect::<Vec<_>>()
                .join(",");
            format!("{{{m2m_str}}}")
        }
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
    drop(take_live_handle(args.layout));
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
    let encoded = borrow_live_handle(args.layout)
        .map_or_else(
            || {
                let mxu_dim = i64::from(global_init_config().generation.mxu_dimension());
                format!("{{1,0:T(8,{mxu_dim})}}")
            },
            |l_ref| format_hlo_memory_layout(&l_ref.layout, l_ref.sublane_tile, l_ref.lane_tile),
        )
        .into_bytes();
    let (ptr, len, holder) = box_serialized_artifact(encoded);
    args.serialized_bytes = ptr;
    args.serialized_bytes_size = len;
    args.serialized_layout = holder.cast::<std::ffi::c_void>();
    args.serialized_layout_deleter = Some(pjrt_serialized_artifact_deleter);
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
/// If `args.buffer` is non-null, it must point to a valid [`crate::pjrt_types::PJRT_Buffer`].
unsafe fn handle_layouts_buffer_get_memory_layout(
    args: &mut PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args,
) {
    // SAFETY: `args.buffer` is null or a valid `PJRT_Buffer*`.
    let layout = unsafe { args.buffer.as_ref() }.map_or_else(
        || MemoryLayout::default_tpu_tiled(2),
        |buf| buf.inner.shape.layout.clone(),
    );
    args.layout = alloc_registered_memory_layout(layout);
}

/// Queries the memory layout of a `PJRT_Buffer` (`PJRT_Layouts_Buffer_GetMemoryLayout`).
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

/// Builds registered [`PJRT_Layouts_MemoryLayout`] handles for a slice of [`TensorShape`]s.
///
/// # Arguments
///
/// * `shapes` - Input or output tensor shapes from a compiled program.
pub(crate) fn build_layout_handles_for_shapes(
    shapes: &[TensorShape],
) -> Vec<*mut PJRT_Layouts_MemoryLayout> {
    if shapes.is_empty() {
        vec![alloc_memory_layout_handle(1)]
    } else {
        shapes
            .iter()
            .map(|s| alloc_registered_memory_layout(s.layout.clone()))
            .collect()
    }
}

/// Populates executable parameter or output layouts from `PJRT_Executable`'s compiled program.
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Layouts_PJRT_Executable_Layouts_Args`] (`32` or `40` bytes).
/// * `is_output` - `true` for output shapes, `false` for input parameter shapes.
///
/// # Safety
///
/// `raw_args` must be null or point to at least 32 writable bytes.
unsafe fn populate_executable_layouts(
    raw_args: *mut std::ffi::c_void,
    is_output: bool,
) -> *mut PJRT_Error {
    crate::ffi_catch_unwind(ptr::null_mut(), || {
        if raw_args.is_null() {
            return crate::pjrt_helpers::box_pjrt_error(LibTpuError::NullPointer {
                parameter: "args",
            });
        }
        // SAFETY: `raw_args` is non-null.
        let actual_size = unsafe { *raw_args.cast::<usize>() };
        let expected_min = PJRT_Layouts_PJRT_Executable_Layouts_Args::MIN_STRUCT_SIZE;
        if actual_size < expected_min {
            return crate::pjrt_helpers::box_pjrt_error(LibTpuError::InvalidStructSize {
                struct_name: "PJRT_Layouts_PJRT_Executable_Layouts_Args",
                expected_min,
                actual: actual_size,
            });
        }
        // SAFETY: `actual_size >= 32` covers `executable` at offset 16 and `num_layouts` at offset 24.
        let exec_ptr = unsafe {
            *raw_args
                .byte_add(16)
                .cast::<*mut crate::pjrt_types::PJRT_Executable>()
        };
        // SAFETY: `exec_ptr` is null or a valid `PJRT_Executable*`.
        let (count, layouts_ptr) = unsafe { exec_ptr.as_ref() }.map_or((0, ptr::null()), |exec| {
            let mut arena = exec.ext_arena.lock();
            let vec_ref = if is_output {
                if arena.output_layout_ptrs.is_empty() {
                    arena.output_layout_ptrs =
                        build_layout_handles_for_shapes(&exec.inner.program.output_shapes);
                }
                &arena.output_layout_ptrs
            } else {
                if arena.parameter_layout_ptrs.is_empty() {
                    arena.parameter_layout_ptrs =
                        build_layout_handles_for_shapes(&exec.inner.program.parameter_shapes);
                }
                &arena.parameter_layout_ptrs
            };
            (vec_ref.len(), vec_ref.as_ptr())
        });
        // SAFETY: Writing `num_layouts` at offset 24 and optional `layouts` at offset 32.
        unsafe {
            *raw_args.byte_add(24).cast::<usize>() = count;
            if actual_size >= 40 {
                *raw_args
                    .byte_add(32)
                    .cast::<*const *mut PJRT_Layouts_MemoryLayout>() = layouts_ptr;
            }
        }
        ptr::null_mut()
    })
}

/// Queries output memory layouts from a `PJRT_Executable` (`PJRT_Layouts_Executable_GetOutputLayouts`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Layouts_PJRT_Executable_Layouts_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Layouts_PJRT_Executable_Layouts_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Layouts_Executable_GetOutputLayouts(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `populate_executable_layouts`.
    unsafe { populate_executable_layouts(raw_args, true) }
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

/// Queries parameter memory layouts from a `PJRT_Executable` (`PJRT_Layouts_Executable_GetParameterLayouts`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Layouts_PJRT_Executable_Layouts_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Layouts_PJRT_Executable_Layouts_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Layouts_Executable_GetParameterLayouts(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `populate_executable_layouts`.
    unsafe { populate_executable_layouts(raw_args, false) }
}

/// Strongly-typed handler for [`PJRT_RawBuffer_CreateRawAliasOfBuffer`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_RawBuffer_CreateViewOfDeviceBuffer_Args`] structure.
///
/// # Errors
///
/// Returns [`LibTpuError::NullPointer`] if `args.buffer` is null.
///
/// # Safety
///
/// If non-null, `args.buffer` must point to a valid [`crate::pjrt_types::PJRT_Buffer`].
unsafe fn handle_raw_buffer_create_alias(
    args: &mut PJRT_RawBuffer_CreateViewOfDeviceBuffer_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `args.buffer` is null or a valid `PJRT_Buffer*`.
    let buf = unsafe { args.buffer.as_ref() }.ok_or(LibTpuError::NullPointer {
        parameter: "buffer",
    })?;
    args.raw_buffer_out = box_registered_handle(PJRT_RawBuffer {
        buffer_ptr: args.buffer,
        device_id: buf.inner.device_id,
        device_address: buf.inner.address,
        byte_size: buf.inner.shape.on_device_byte_size(),
        memory_ptr: buf.memory_ptr,
        hal: std::sync::Arc::clone(&buf.inner.hal),
    });
    Ok(())
}

/// Creates a raw alias of a `PJRT_Buffer` (`PJRT_RawBuffer_CreateRawAliasOfBuffer`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_RawBuffer_CreateViewOfDeviceBuffer_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_RawBuffer_CreateViewOfDeviceBuffer_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_RawBuffer_CreateRawAliasOfBuffer(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PJRT_RawBuffer_CreateViewOfDeviceBuffer_Args>(
            raw_args,
            handle_raw_buffer_create_alias,
        )
    }
}

/// Strongly-typed handler for [`PJRT_RawBuffer_Destroy`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_RawBuffer_Destroy_Args`] structure.
///
/// # Errors
///
/// Returns [`LibTpuError::InvalidArgument`] if `args.raw_buffer` is null or not a live [`PJRT_RawBuffer`].
///
/// # Safety
///
/// `args` must be a valid [`PJRT_RawBuffer_Destroy_Args`].
unsafe fn handle_raw_buffer_destroy(
    args: &mut PJRT_RawBuffer_Destroy_Args,
) -> Result<(), LibTpuError> {
    if take_live_handle(args.raw_buffer).is_none() {
        return Err(LibTpuError::InvalidArgument {
            parameter: "raw_buffer",
            reason: String::from("null or unregistered PJRT_RawBuffer handle"),
        });
    }
    args.raw_buffer = ptr::null_mut();
    Ok(())
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
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PJRT_RawBuffer_Destroy_Args>(
            raw_args,
            handle_raw_buffer_destroy,
        )
    }
}

/// Strongly-typed handler for [`PJRT_RawBuffer_GetHostPointer`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_RawBuffer_GetHostPointer_Args`] structure.
///
/// # Errors
///
/// Returns [`LibTpuError::NullPointer`] on null `raw_buffer` or [`LibTpuError::InvalidArgument`] on invalid handle.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_RawBuffer_GetHostPointer_Args`].
unsafe fn handle_raw_buffer_get_host_ptr(
    args: &mut PJRT_RawBuffer_GetHostPointer_Args,
) -> Result<(), LibTpuError> {
    if args.raw_buffer.is_null() {
        return Err(LibTpuError::NullPointer {
            parameter: "raw_buffer",
        });
    }
    let raw_buf =
        borrow_live_handle(args.raw_buffer).ok_or_else(|| LibTpuError::InvalidArgument {
            parameter: "raw_buffer",
            reason: String::from("unregistered PJRT_RawBuffer handle"),
        })?;
    let addr = usize::try_from(raw_buf.device_address.0).unwrap_or(0);
    args.host_pointer_out = addr as *mut std::ffi::c_void;
    Ok(())
}

/// Queries the device/host pointer of a `PJRT_RawBuffer` (`PJRT_RawBuffer_GetHostPointer`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_RawBuffer_GetHostPointer_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_RawBuffer_GetHostPointer_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_RawBuffer_GetHostPointer(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PJRT_RawBuffer_GetHostPointer_Args>(
            raw_args,
            handle_raw_buffer_get_host_ptr,
        )
    }
}

/// Strongly-typed handler for [`PJRT_RawBuffer_GetOnDeviceSizeInBytes`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_RawBuffer_GetOnDeviceSizeInBytes_Args`] structure.
///
/// # Errors
///
/// Returns [`LibTpuError::NullPointer`] on null `raw_buffer` or [`LibTpuError::InvalidArgument`] on invalid handle.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_RawBuffer_GetOnDeviceSizeInBytes_Args`].
unsafe fn handle_raw_buffer_get_size(
    args: &mut PJRT_RawBuffer_GetOnDeviceSizeInBytes_Args,
) -> Result<(), LibTpuError> {
    if args.raw_buffer.is_null() {
        return Err(LibTpuError::NullPointer {
            parameter: "raw_buffer",
        });
    }
    let raw_buf =
        borrow_live_handle(args.raw_buffer).ok_or_else(|| LibTpuError::InvalidArgument {
            parameter: "raw_buffer",
            reason: String::from("unregistered PJRT_RawBuffer handle"),
        })?;
    args.on_device_size_in_bytes = raw_buf.byte_size.0;
    Ok(())
}

/// Queries the on-device byte size of a `PJRT_RawBuffer` (`PJRT_RawBuffer_GetOnDeviceSizeInBytes`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_RawBuffer_GetOnDeviceSizeInBytes_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_RawBuffer_GetOnDeviceSizeInBytes_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_RawBuffer_GetOnDeviceSizeInBytes(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PJRT_RawBuffer_GetOnDeviceSizeInBytes_Args>(
            raw_args,
            handle_raw_buffer_get_size,
        )
    }
}

/// Strongly-typed handler for [`PJRT_RawBuffer_GetMemorySpace`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_RawBuffer_GetMemorySpace_Args`] structure.
///
/// # Errors
///
/// Returns [`LibTpuError::NullPointer`] on null `raw_buffer` or [`LibTpuError::InvalidArgument`] on invalid handle.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_RawBuffer_GetMemorySpace_Args`].
unsafe fn handle_raw_buffer_get_mem_space(
    args: &mut PJRT_RawBuffer_GetMemorySpace_Args,
) -> Result<(), LibTpuError> {
    if args.raw_buffer.is_null() {
        return Err(LibTpuError::NullPointer {
            parameter: "raw_buffer",
        });
    }
    let raw_buf =
        borrow_live_handle(args.raw_buffer).ok_or_else(|| LibTpuError::InvalidArgument {
            parameter: "raw_buffer",
            reason: String::from("unregistered PJRT_RawBuffer handle"),
        })?;
    args.memory_space_out = raw_buf.memory_ptr;
    Ok(())
}

/// Queries the memory space of a `PJRT_RawBuffer` (`PJRT_RawBuffer_GetMemorySpace`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_RawBuffer_GetMemorySpace_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_RawBuffer_GetMemorySpace_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_RawBuffer_GetMemorySpace(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PJRT_RawBuffer_GetMemorySpace_Args>(
            raw_args,
            handle_raw_buffer_get_mem_space,
        )
    }
}

/// Validates raw buffer transfer bounds and returns `(&PJRT_RawBuffer, DeviceAddress, usize)`.
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_RawBuffer_CopyRaw_Args`].
/// * `ptr_name` - Parameter name for `host_ptr` (`"src"` or `"dst"`).
///
/// # Errors
///
/// Returns [`LibTpuError::NullPointer`] or [`LibTpuError::InvalidArgument`] on invalid arguments or bounds.
fn validate_raw_buffer_transfer<'a>(
    args: &'a PJRT_RawBuffer_CopyRaw_Args,
    ptr_name: &'static str,
) -> Result<(&'a PJRT_RawBuffer, DeviceAddress, usize), LibTpuError> {
    if args.buffer.is_null() {
        return Err(LibTpuError::NullPointer {
            parameter: "raw_buffer",
        });
    }
    let raw_buf = borrow_live_handle(args.buffer.cast_const()).ok_or_else(|| {
        LibTpuError::InvalidArgument {
            parameter: "raw_buffer",
            reason: String::from("unregistered PJRT_RawBuffer handle"),
        }
    })?;
    if args.host_ptr.is_null() {
        return Err(LibTpuError::NullPointer {
            parameter: ptr_name,
        });
    }
    let offset_u64 = u64::try_from(args.offset).map_err(|_| LibTpuError::InvalidArgument {
        parameter: "offset",
        reason: format!("negative raw buffer offset {}", args.offset),
    })?;
    let transfer_u64 =
        u64::try_from(args.transfer_size).map_err(|_| LibTpuError::InvalidArgument {
            parameter: "transfer_size",
            reason: format!("negative raw buffer transfer_size {}", args.transfer_size),
        })?;
    let end_u64 = offset_u64 + transfer_u64;
    let cap_u64 = u64::try_from(raw_buf.byte_size.0).unwrap_or(u64::MAX);
    if end_u64 > cap_u64 {
        return Err(LibTpuError::InvalidArgument {
            parameter: "transfer_size",
            reason: format!(
                "offset ({offset_u64}) + transfer_size ({transfer_u64}) exceeds raw_buffer size ({})",
                raw_buf.byte_size.0
            ),
        });
    }
    let target_addr = DeviceAddress(raw_buf.device_address.0.saturating_add(offset_u64));
    let len = usize::try_from(transfer_u64).unwrap_or(0);
    Ok((raw_buf, target_addr, len))
}

/// Strongly-typed handler for [`PJRT_RawBuffer_CopyRawHostToDevice`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_RawBuffer_CopyRaw_Args`] structure.
///
/// # Errors
///
/// Returns [`LibTpuError`] if `buffer`/`src` is null, unregistered, or `offset + transfer_size > byte_size`.
///
/// # Safety
///
/// `args.host_ptr` must reference at least `args.transfer_size` readable bytes when non-null.
unsafe fn handle_raw_buffer_copy_h2d(
    args: &mut PJRT_RawBuffer_CopyRaw_Args,
) -> Result<(), LibTpuError> {
    let (raw_buf, target_addr, len) = validate_raw_buffer_transfer(args, "src")?;
    if len > 0 {
        // SAFETY: `args.host_ptr` is non-null and caller guarantees `len` readable bytes.
        let src_slice = unsafe { std::slice::from_raw_parts(args.host_ptr.cast::<u8>(), len) };
        raw_buf
            .hal
            .write_device_memory(raw_buf.device_id, target_addr, src_slice)?;
    }
    args.event = box_ready_event();
    Ok(())
}

/// Copies raw host bytes into a `PJRT_RawBuffer` (`PJRT_RawBuffer_CopyRawHostToDevice`).
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
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PJRT_RawBuffer_CopyRaw_Args>(
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
/// # Errors
///
/// Returns [`LibTpuError`] if `buffer`/`dst` is null, unregistered, or `offset + transfer_size > byte_size`.
///
/// # Safety
///
/// `args.host_ptr` must reference at least `args.transfer_size` writable bytes when non-null.
unsafe fn handle_raw_buffer_copy_d2h(
    args: &mut PJRT_RawBuffer_CopyRaw_Args,
) -> Result<(), LibTpuError> {
    let (raw_buf, target_addr, len) = validate_raw_buffer_transfer(args, "dst")?;
    if len > 0 {
        // SAFETY: `args.host_ptr` is non-null and caller guarantees `len` writable bytes.
        let dst_slice = unsafe { std::slice::from_raw_parts_mut(args.host_ptr.cast::<u8>(), len) };
        raw_buf
            .hal
            .read_device_memory(raw_buf.device_id, target_addr, dst_slice)?;
    }
    args.event = box_ready_event();
    Ok(())
}

/// Copies raw device bytes from a `PJRT_RawBuffer` to host (`PJRT_RawBuffer_CopyRawDeviceToHost`).
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
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PJRT_RawBuffer_CopyRaw_Args>(
            raw_args,
            handle_raw_buffer_copy_d2h,
        )
    }
}

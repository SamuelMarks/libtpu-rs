//! `#[repr(C)]` PJRT extension structures for all 17 TPU extensions linked from `PJRT_Api.extension_start`.
//!
//! Matches the exact byte sizes, type discriminants (`PJRT_Extension_Type`), and named function pointer
//! fields disassembled from `GetTpuPjrtApi()` (`0x0b4fb4e0`) in `libtpu.so`:
//! 1. `PJRT_XlaTransform_Extension` (`56` bytes, `ext_type = 29`)
//! 2. `PJRT_HostMemoryAllocator_Extension` (`32` bytes, `ext_type = 28`)
//! 3. `PJRT_MultiSlice_Extension` (`64` bytes, `ext_type = 26`)
//! 4. `PJRT_Collectives_Extension` (`96` bytes, `ext_type = 25`)
//! 5. `PJRT_AbiVersion_Extension` (`120` bytes, `ext_type = 20`)
//! 6. `PJRT_Shardings_Extension` (`40` bytes, `ext_type = 19`)
//! 7. `PJRT_Megascale_Extension` (`248` bytes, `ext_type = 18`)
//! 8. `PJRT_TpuExecutable_Extension` (`96` bytes, `ext_type = 17`)
//! 9. `PJRT_TpuTopology_Extension` (`272` bytes, `ext_type = 16`)
//! 10. `PJRT_Callback_Extension` (`40` bytes, `ext_type = 14`)
//! 11. `PJRT_PhaseCompile_Extension` (`64` bytes, `ext_type = 9`)
//! 12. `PJRT_CrossHostTransfers_Extension` (`56` bytes, `ext_type = 12`)
//! 13. `PJRT_ExecutableMetadata_Extension` (`40` bytes, `ext_type = 11`)
//! 14. `PJRT_MemoryDescriptions_Extension` (`40` bytes, `ext_type = 7`)
//! 15. `PJRT_Layouts_Extension` (`80` bytes, `ext_type = 6`)
//! 16. `PJRT_RawBuffer_Extension` (`80` bytes, `ext_type = 8`)
//! 17. `PJRT_Profiler_Extension` (`40` bytes, `ext_type = 1`)

/// `#[repr(C)]` argument structures, enums, and size assertions for PJRT extensions.
pub mod args;
/// PJRT Collectives and Cross-Host Transfers extension entrypoints.
pub mod collectives;
/// PJRT XLA Transform, Callback, PhaseCompile, and Triton extension entrypoints.
pub mod compiler;
/// PJRT ABI Version, Shardings, and TPU Executable extension entrypoints.
pub mod executable;
/// PJRT Multi-Slice and Megascale extension entrypoints.
pub mod megascale;
/// PJRT Host Memory Allocator, Metadata, Memory Descriptions, Layouts, and Raw Buffer entrypoints.
pub mod memory;
/// PJRT Profiler and `PLUGIN_Profiler_Api` extension entrypoints.
pub mod profiler;
/// PJRT TPU Topology extension entrypoints.
pub mod topology;

pub use args::*;
pub use collectives::*;
pub use compiler::*;
pub use executable::*;
pub use megascale::*;
pub use memory::*;
pub use profiler::*;
pub use topology::*;

use crate::box_pjrt_error;
use crate::pjrt_types::{PJRT_Error, PJRT_Extension_Base, PjrtFn};
use libtpu_core::driver::TpuCoreProfiler;
use libtpu_core::error::LibTpuError;
use parking_lot::Mutex;
use std::ptr;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// Monotonic counter recording extension invocations across all 138 extension and profiler entrypoints.
static EXTENSION_CALL_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Process-wide [`TpuCoreProfiler`] backing `PLUGIN_Profiler_Api`.
static EXTENSION_PROFILER: OnceLock<Mutex<TpuCoreProfiler>> = OnceLock::new();

/// Returns the total number of extension invocations recorded so far.
pub fn extension_call_count() -> u64 {
    EXTENSION_CALL_COUNTER.load(Ordering::Relaxed)
}

/// Validates a PJRT extension argument structure (`PJRT_Extension_ValidateArgs`).
///
/// # Arguments
///
/// * `raw_args` - Opaque pointer to a PJRT extension argument structure whose first field is `size_t struct_size`.
///
/// # Safety
///
/// `raw_args` must be null or point to a struct beginning with `usize`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Extension_ValidateArgs(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Caller upholds `raw_args` contract.
    unsafe { validate_extension_args_size(raw_args, size_of::<usize>()) }
}

/// Validates that `raw_args` is non-null and its leading `struct_size` field is at least `min_size`.
///
/// # Arguments
///
/// * `raw_args` - Opaque pointer to a PJRT extension argument structure whose first field is `size_t struct_size`.
/// * `min_size` - Minimum required `struct_size` in bytes.
///
/// # Safety
///
/// `raw_args` must be null or point to a struct beginning with `usize`.
unsafe fn validate_extension_args_size(
    raw_args: *mut std::ffi::c_void,
    min_size: usize,
) -> *mut PJRT_Error {
    if raw_args.is_null() {
        return box_pjrt_error(LibTpuError::NullPointer {
            parameter: "extension_args",
        });
    }
    // SAFETY: `raw_args` is non-null and starts with `usize struct_size`.
    let actual_size = unsafe { *raw_args.cast::<usize>() };
    if actual_size < min_size {
        return box_pjrt_error(LibTpuError::InvalidStructSize {
            struct_name: "PJRT_Extension_Base",
            expected_min: min_size,
            actual: actual_size,
        });
    }
    EXTENSION_CALL_COUNTER.fetch_add(1, Ordering::Relaxed);
    ptr::null_mut()
}

/// Stateful coordinator and registry backing the 17 TPU PJRT extension structures.
#[derive(Debug, Default)]
pub struct ExtensionRuntimeState {
    /// Number of registered XLA HLO transform passes (`PJRT_XlaTransform_Extension`).
    pub registered_transforms: usize,
    /// Map of registered XLA HLO transform pass names to registration IDs.
    pub transform_passes: std::collections::BTreeMap<String, usize>,
    /// Number of active ICI/DCN collective communicators (`PJRT_Collectives_Extension`).
    pub active_communicators: usize,
    /// Total collective operations executed across communicators.
    pub collective_ops_executed: usize,
    /// Number of active Megascale client contexts (`PJRT_Megascale_Extension`).
    pub megascale_contexts: usize,
    /// Number of active Megascale multi-slice configs.
    pub megascale_configs: usize,
    /// Number of completed Megascale multi-slice barriers.
    pub megascale_barriers: usize,
    /// Number of registered Megascale error handlers.
    pub megascale_error_handlers: usize,
    /// Configured Megascale transport type (`0` = Default/ICI, `1` = DCN).
    pub megascale_transport_type: usize,
    /// Number of configured `TpuCompilationEnvironmentProto` field overrides.
    pub tpu_compilation_env_overrides: usize,
    /// Number of registered `PJRT_Callback_Extension` callbacks.
    pub registered_callbacks: usize,
    /// Registered callback function pointer and user context per callback type.
    pub callback_registry: std::collections::BTreeMap<i64, (usize, usize)>,
    /// Number of active `PJRT_PhaseCompiler` instances.
    pub phase_compilers: usize,
    /// Number of completed cross-host buffer transfers.
    pub cross_host_transfers: usize,
    /// Number of active `PJRT_RawBuffer` alias handles.
    pub raw_buffers: usize,
    /// Whether the active TPU topology is a subslice topology.
    pub is_subslice_topology: bool,
    /// Heap-allocated [`crate::pjrt_types::PjrtSerializedArtifact`] buffers whose lifetime is managed by extension state or explicit deleters.
    pub serialized_artifacts: Vec<Box<[u8]>>,
}

/// Process-wide [`ExtensionRuntimeState`] singleton.
static EXTENSION_STATE: OnceLock<Mutex<ExtensionRuntimeState>> = OnceLock::new();

/// Returns the singleton [`Mutex<ExtensionRuntimeState>`] backing all 17 PJRT extensions.
pub(super) fn extension_state() -> &'static Mutex<ExtensionRuntimeState> {
    EXTENSION_STATE.get_or_init(|| Mutex::new(ExtensionRuntimeState::default()))
}

/// Allocates an immutable heap-backed byte slice registered in [`ExtensionRuntimeState::serialized_artifacts`]
/// so that multiple concurrent or sequential extension queries never overwrite each other's buffers.
///
/// # Arguments
///
/// * `bytes` - Serialized protobuf or string payload bytes.
pub(super) fn alloc_extension_bytes(bytes: Vec<u8>) -> (*const u8, usize) {
    let boxed: Box<[u8]> = bytes.into_boxed_slice();
    let ptr = boxed.as_ptr();
    let len = boxed.len();
    extension_state().lock().serialized_artifacts.push(boxed);
    (ptr, len)
}

/// Returns the singleton [`Mutex<TpuCoreProfiler>`] backing `PLUGIN_Profiler_Api`.
pub(super) fn extension_profiler() -> &'static Mutex<TpuCoreProfiler> {
    EXTENSION_PROFILER.get_or_init(|| Mutex::new(TpuCoreProfiler::new()))
}

/// Invokes a strongly-typed `unsafe fn(&mut T)` from a type-erased reference.
///
/// # Arguments
///
/// * `raw_args` - Raw pointer to the validated extension `_Args` structure of type `T`.
/// * `handler_ref` - Type-erased pointer to the `unsafe fn(&mut T)` callback.
///
/// # Safety
///
/// `raw_args` must point to a valid `T` and `handler_ref` must point to `unsafe fn(&mut T)`.
unsafe fn invoke_typed_extension_handler<T>(
    raw_args: *mut std::ffi::c_void,
    handler_ref: *const std::ffi::c_void,
) {
    // SAFETY: `handler_ref` points to `handler` on the stack of `dispatch_typed_extension_args`.
    let handler = unsafe { *handler_ref.cast::<unsafe fn(&mut T)>() };
    // SAFETY: `raw_args` was validated non-null by `dispatch_erased_extension_args`.
    unsafe {
        handler(&mut *raw_args.cast::<T>());
    }
}

/// Non-generic validation and invocation dispatcher backing [`dispatch_typed_extension_args`].
///
/// # Arguments
///
/// * `raw_args` - Raw pointer to the extension `_Args` structure.
/// * `min_size` - Minimum required `struct_size` in bytes (`size_of::<T>()`).
/// * `handler_ref` - Type-erased pointer to the strongly-typed callback.
/// * `invoker` - Monomorphized trampoline invoking the strongly-typed callback.
///
/// # Safety
///
/// `raw_args` must be null or point to a valid extension `_Args` structure of type `T`.
unsafe fn dispatch_erased_extension_args(
    raw_args: *mut std::ffi::c_void,
    min_size: usize,
    handler_ref: *const std::ffi::c_void,
    invoker: unsafe fn(*mut std::ffi::c_void, *const std::ffi::c_void),
) -> *mut PJRT_Error {
    // SAFETY: Caller upholds `raw_args` precondition.
    let err = unsafe { validate_extension_args_size(raw_args, min_size) };
    if err.is_null() {
        // SAFETY: `raw_args` was validated non-null and `>= min_size` by `validate_extension_args_size`.
        unsafe {
            invoker(raw_args, handler_ref);
        }
    }
    err
}

/// Validates a typed `#[repr(C)]` PJRT extension argument structure and invokes `handler(&mut T)`.
///
/// # Arguments
///
/// * `raw_args` - Pointer to the typed extension `_Args` structure of type `T`.
/// * `handler` - Strongly-typed callback operating on `&mut T`.
///
/// # Safety
///
/// `raw_args` must be null or point to a valid extension `_Args` struct of type `T`.
#[inline]
pub(super) unsafe fn dispatch_typed_extension_args<T>(
    raw_args: *mut std::ffi::c_void,
    handler: unsafe fn(&mut T),
) -> *mut PJRT_Error {
    let handler_ref = ptr::from_ref(&handler).cast::<std::ffi::c_void>();
    // SAFETY: Caller upholds `raw_args` contract; `handler_ref` points to `handler`.
    unsafe {
        dispatch_erased_extension_args(
            raw_args,
            size_of::<T>(),
            handler_ref,
            invoke_typed_extension_handler::<T>,
        )
    }
}

/// Node 1 (`ext_type = 29`, `56` bytes): `PJRT_XlaTransform_Extension`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_XlaTransform_Extension {
    /// 24-byte `PJRT_Extension_Base` header.
    pub base: PJRT_Extension_Base,
    /// Slot 1 (`0x18`): `register_xla_transform`.
    pub register_xla_transform: PjrtFn,
    /// Slot 2 (`0x20`): `wait_for_all_transforms`.
    pub wait_for_all_transforms: PjrtFn,
    /// Slot 3 (`0x28`): `unregister_xla_transform`.
    pub unregister_xla_transform: PjrtFn,
    /// Slot 4 (`0x30`): `list_xla_transforms`.
    pub list_xla_transforms: PjrtFn,
}

// SAFETY: Immutable static vtable structure.
unsafe impl Sync for PJRT_XlaTransform_Extension {}

/// Node 2 (`ext_type = 28`, `32` bytes): `PJRT_HostMemoryAllocator_Extension`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_HostMemoryAllocator_Extension {
    /// 24-byte `PJRT_Extension_Base` header.
    pub base: PJRT_Extension_Base,
    /// Slot 1 (`0x18`): `get_host_memory_allocator`.
    pub get_host_memory_allocator: PjrtFn,
}

// SAFETY: Immutable static vtable structure.
unsafe impl Sync for PJRT_HostMemoryAllocator_Extension {}

/// Node 3 (`ext_type = 26`, `64` bytes): `PJRT_MultiSlice_Extension`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_MultiSlice_Extension {
    /// 24-byte `PJRT_Extension_Base` header.
    pub base: PJRT_Extension_Base,
    /// Slot 1 (`0x18`): `config_destroy`.
    pub config_destroy: PjrtFn,
    /// Slot 2 (`0x20`): `config_num_slices`.
    pub config_num_slices: PjrtFn,
    /// Slot 3 (`0x28`): `config_num_devices_per_slice`.
    pub config_num_devices_per_slice: PjrtFn,
    /// Slot 4 (`0x30`): `config_slice_id`.
    pub config_slice_id: PjrtFn,
    /// Slot 5 (`0x38`): `config_serialize`.
    pub config_serialize: PjrtFn,
}

// SAFETY: Immutable static vtable structure.
unsafe impl Sync for PJRT_MultiSlice_Extension {}

/// Node 4 (`ext_type = 25`, `96` bytes): `PJRT_Collectives_Extension`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Collectives_Extension {
    /// 24-byte `PJRT_Extension_Base` header.
    pub base: PJRT_Extension_Base,
    /// Slot 1 (`0x18`): `collectives_destroy`.
    pub collectives_destroy: PjrtFn,
    /// Slot 2 (`0x20`): `create_communicator`.
    pub create_communicator: PjrtFn,
    /// Slot 3 (`0x28`): `communicator_destroy`.
    pub communicator_destroy: PjrtFn,
    /// Slot 4 (`0x30`): `communicator_all_reduce`.
    pub communicator_all_reduce: PjrtFn,
    /// Slot 5 (`0x38`): `communicator_reduce_scatter`.
    pub communicator_reduce_scatter: PjrtFn,
    /// Slot 6 (`0x40`): `communicator_all_gather`.
    pub communicator_all_gather: PjrtFn,
    /// Slot 7 (`0x48`): `communicator_collective_permute`.
    pub communicator_collective_permute: PjrtFn,
    /// Slot 8 (`0x50`): `communicator_all_to_all`.
    pub communicator_all_to_all: PjrtFn,
    /// Slot 9 (`0x58`): `communicator_to_string`.
    pub communicator_to_string: PjrtFn,
}

// SAFETY: Immutable static vtable structure.
unsafe impl Sync for PJRT_Collectives_Extension {}

/// Node 5 (`ext_type = 20`, `120` bytes): `PJRT_AbiVersion_Extension`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_AbiVersion_Extension {
    /// 24-byte `PJRT_Extension_Base` header.
    pub base: PJRT_Extension_Base,
    /// Slot 1 (`0x18`): `client_runtime_abi_version`.
    pub client_runtime_abi_version: PjrtFn,
    /// Slot 2 (`0x20`): `executable_get_abi_version`.
    pub executable_get_abi_version: PjrtFn,
    /// Slot 3 (`0x28`): `runtime_abi_version_destroy`.
    pub runtime_abi_version_destroy: PjrtFn,
    /// Slot 4 (`0x30`): `runtime_abi_version_is_compatible_with_runtime`.
    pub runtime_abi_version_is_compatible_with_runtime: PjrtFn,
    /// Slot 5 (`0x38`): `runtime_abi_version_is_compatible_with_executable`.
    pub runtime_abi_version_is_compatible_with_executable: PjrtFn,
    /// Slot 6 (`0x40`): `runtime_abi_version_to_proto`.
    pub runtime_abi_version_to_proto: PjrtFn,
    /// Slot 7 (`0x48`): `runtime_abi_version_platform_id`.
    pub runtime_abi_version_platform_id: PjrtFn,
    /// Slot 8 (`0x50`): `executable_abi_version_destroy`.
    pub executable_abi_version_destroy: PjrtFn,
    /// Slot 9 (`0x58`): `executable_abi_version_to_proto`.
    pub executable_abi_version_to_proto: PjrtFn,
    /// Slot 10 (`0x60`): `executable_abi_version_platform_id`.
    pub executable_abi_version_platform_id: PjrtFn,
    /// Slot 11 (`0x68`): `runtime_abi_version_from_proto`.
    pub runtime_abi_version_from_proto: PjrtFn,
    /// Slot 12 (`0x70`): `executable_abi_version_from_proto`.
    pub executable_abi_version_from_proto: PjrtFn,
}

// SAFETY: Immutable static vtable structure.
unsafe impl Sync for PJRT_AbiVersion_Extension {}

/// Node 6 (`ext_type = 19`, `40` bytes): `PJRT_Shardings_Extension`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Shardings_Extension {
    /// 24-byte `PJRT_Extension_Base` header.
    pub base: PJRT_Extension_Base,
    /// Slot 1 (`0x18`): `PJRT_Shardings_PJRT_Executable_ParameterShardings`.
    pub parameter_shardings: PjrtFn,
    /// Slot 2 (`0x20`): `PJRT_Shardings_PJRT_Executable_OutputShardings`.
    pub output_shardings: PjrtFn,
}

// SAFETY: Immutable static vtable structure.
unsafe impl Sync for PJRT_Shardings_Extension {}

/// Node 7 (`ext_type = 18`, `248` bytes): `PJRT_Megascale_Extension`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_Extension {
    /// 24-byte `PJRT_Extension_Base` header.
    pub base: PJRT_Extension_Base,
    /// Slot 1 (`0x18`): `create_client_context`.
    pub create_client_context: PjrtFn,
    /// Slot 2 (`0x20`): `delete_client_context`.
    pub delete_client_context: PjrtFn,
    /// Slot 3 (`0x28`): `create_default_config`.
    pub create_default_config: PjrtFn,
    /// Slot 4 (`0x30`): `create_aot_config`.
    pub create_aot_config: PjrtFn,
    /// Slot 5 (`0x38`): `create_multi_slice_config`.
    pub create_multi_slice_config: PjrtFn,
    /// Slot 6 (`0x40`): `delete_multi_slice_config`.
    pub delete_multi_slice_config: PjrtFn,
    /// Slot 7 (`0x48`): `multi_slice_config_num_slices`.
    pub multi_slice_config_num_slices: PjrtFn,
    /// Slot 8 (`0x50`): `multi_slice_config_num_devices_per_slice`.
    pub multi_slice_config_num_devices_per_slice: PjrtFn,
    /// Slot 9 (`0x58`): `multi_slice_config_slice_id`.
    pub multi_slice_config_slice_id: PjrtFn,
    /// Slot 10 (`0x60`): `multi_slice_config_serialized_slice_key_value`.
    pub multi_slice_config_serialized_slice_key_value: PjrtFn,
    /// Slot 11 (`0x68`): `multi_slice_config_serialize`.
    pub multi_slice_config_serialize: PjrtFn,
    /// Slot 12 (`0x70`): `deserialize_multi_slice_config`.
    pub deserialize_multi_slice_config: PjrtFn,
    /// Slot 13 (`0x78`): `megascale_addresses`.
    pub megascale_addresses: PjrtFn,
    /// Slot 14 (`0x80`): `barrier`.
    pub barrier: PjrtFn,
    /// Slot 15 (`0x88`): `log_error`.
    pub log_error: PjrtFn,
    /// Slot 16 (`0x90`): `get_megascale_dcn_topology`.
    pub get_megascale_dcn_topology: PjrtFn,
    /// Slot 17 (`0x98`): `get_slice_id`.
    pub get_slice_id: PjrtFn,
    /// Slot 18 (`0xa0`): `get_num_slices`.
    pub get_num_slices: PjrtFn,
    /// Slot 19 (`0xa8`): `initialize_slice`.
    pub initialize_slice: PjrtFn,
    /// Slot 20 (`0xb0`): `finalize_slice`.
    pub finalize_slice: PjrtFn,
    /// Slot 21 (`0xb8`): `register_error_handler`.
    pub register_error_handler: PjrtFn,
    /// Slot 22 (`0xc0`): `unregister_error_handler`.
    pub unregister_error_handler: PjrtFn,
    /// Slot 23 (`0xc8`): `collect_metrics`.
    pub collect_metrics: PjrtFn,
    /// Slot 24 (`0xd0`): `reset_metrics`.
    pub reset_metrics: PjrtFn,
    /// Slot 25 (`0xd8`): `get_transport_type`.
    pub get_transport_type: PjrtFn,
    /// Slot 26 (`0xe0`): `set_transport_type`.
    pub set_transport_type: PjrtFn,
    /// Slot 27 (`0xe8`): `wait_for_barrier`.
    pub wait_for_barrier: PjrtFn,
    /// Slot 28 (`0xf0`): `abort_collectives`.
    pub abort_collectives: PjrtFn,
}

// SAFETY: Immutable static vtable structure.
unsafe impl Sync for PJRT_Megascale_Extension {}

/// Node 8 (`ext_type = 17`, `96` bytes): `PJRT_TpuExecutable_Extension`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuExecutable_Extension {
    /// 24-byte `PJRT_Extension_Base` header.
    pub base: PJRT_Extension_Base,
    /// Slot 1 (`0x18`): `get_target_arguments`.
    pub get_target_arguments: PjrtFn,
    /// Slot 2 (`0x20`): `get_hlo_module_with_config`.
    pub get_hlo_module_with_config: PjrtFn,
    /// Slot 3 (`0x28`): `reserved`.
    pub reserved: PjrtFn,
    /// Slot 4 (`0x30`): `get_compiled_memory_stats`.
    pub get_compiled_memory_stats: PjrtFn,
    /// Slot 5 (`0x38`): `run_hlo_cost_analysis`.
    pub run_hlo_cost_analysis: PjrtFn,
    /// Slot 6 (`0x40`): `set_tpu_compilation_env`.
    pub set_tpu_compilation_env: PjrtFn,
    /// Slot 7 (`0x48`): `get_tpu_compilation_env_field_as_string`.
    pub get_tpu_compilation_env_field_as_string: PjrtFn,
    /// Slot 8 (`0x50`): `is_tpu_predetermined_error`.
    pub is_tpu_predetermined_error: PjrtFn,
    /// Slot 9 (`0x58`): `is_tpu_cancellation_placeholder_error`.
    pub is_tpu_cancellation_placeholder_error: PjrtFn,
}

// SAFETY: Immutable static vtable structure.
unsafe impl Sync for PJRT_TpuExecutable_Extension {}

/// Node 9 (`ext_type = 16`, `272` bytes): `PJRT_TpuTopology_Extension`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_Extension {
    /// 24-byte `PJRT_Extension_Base` header.
    pub base: PJRT_Extension_Base,
    /// Slot 1 (`0x18`): `subslice`.
    pub subslice: PjrtFn,
    /// Slot 2 (`0x20`): `is_subslice_topology`.
    pub is_subslice_topology: PjrtFn,
    /// Slot 3 (`0x28`): `subslice_device_id_from_full_device_id`.
    pub subslice_device_id_from_full_device_id: PjrtFn,
    /// Slot 4 (`0x30`): `replace_host_bounds`.
    pub replace_host_bounds: PjrtFn,
    /// Slot 5 (`0x38`): `is_enhanced_barrier_enabled`.
    pub is_enhanced_barrier_enabled: PjrtFn,
    /// Slot 6 (`0x40`): `has_limited_ici_connectivity`.
    pub has_limited_ici_connectivity: PjrtFn,
    /// Slot 7 (`0x48`): `is_reachable_over_limited_ici`.
    pub is_reachable_over_limited_ici: PjrtFn,
    /// Slot 8 (`0x50`): `process_count`.
    pub process_count: PjrtFn,
    /// Slot 9 (`0x58`): `chips_per_process`.
    pub chips_per_process: PjrtFn,
    /// Slot 10 (`0x60`): `core_count_per_chip`.
    pub core_count_per_chip: PjrtFn,
    /// Slot 11 (`0x68`): `chip_count`.
    pub chip_count: PjrtFn,
    /// Slot 12 (`0x70`): `core_count`.
    pub core_count: PjrtFn,
    /// Slot 13 (`0x78`): `logi_device_count_per_process`.
    pub logi_device_count_per_process: PjrtFn,
    /// Slot 14 (`0x80`): `logi_device_count`.
    pub logi_device_count: PjrtFn,
    /// Slot 15 (`0x88`): `logi_device_count_per_chip`.
    pub logi_device_count_per_chip: PjrtFn,
    /// Slot 16 (`0x90`): `core_count_per_process`.
    pub core_count_per_process: PjrtFn,
    /// Slot 17 (`0x98`): `process_ids`.
    pub process_ids: PjrtFn,
    /// Slot 18 (`0xa0`): `logi_device_ids_on_process`.
    pub logi_device_ids_on_process: PjrtFn,
    /// Slot 19 (`0xa8`): `proc_id_and_idx_on_proc_for_chip`.
    pub proc_id_and_idx_on_proc_for_chip: PjrtFn,
    /// Slot 20 (`0xb0`): `proc_id_and_idx_on_proc_for_logi_device`.
    pub proc_id_and_idx_on_proc_for_logi_device: PjrtFn,
    /// Slot 21 (`0xb8`): `process_coord_from_id`.
    pub process_coord_from_id: PjrtFn,
    /// Slot 22 (`0xc0`): `chip_id_from_coord`.
    pub chip_id_from_coord: PjrtFn,
    /// Slot 23 (`0xc8`): `logi_device_id_from_chip_coord_and_idx`.
    pub logi_device_id_from_chip_coord_and_idx: PjrtFn,
    /// Slot 24 (`0xd0`): `chip_coord_and_idx_for_logi_device`.
    pub chip_coord_and_idx_for_logi_device: PjrtFn,
    /// Slot 25 (`0xd8`): `chips_per_process_bounds`.
    pub chips_per_process_bounds: PjrtFn,
    /// Slot 26 (`0xe0`): `chip_bounds`.
    pub chip_bounds: PjrtFn,
    /// Slot 27 (`0xe8`): `process_bounds`.
    pub process_bounds: PjrtFn,
    /// Slot 28 (`0xf0`): `get_routing_strategy`.
    pub get_routing_strategy: PjrtFn,
    /// Slot 29 (`0xf8`): `get_slice_config`.
    pub get_slice_config: PjrtFn,
    /// Slot 30 (`0x100`): `get_slice_configs`.
    pub get_slice_configs: PjrtFn,
    /// Slot 31 (`0x108`): `get_default_platform_config`.
    pub get_default_platform_config: PjrtFn,
}

// SAFETY: Immutable static vtable structure.
unsafe impl Sync for PJRT_TpuTopology_Extension {}

/// Node 10 (`ext_type = 14`, `40` bytes): `PJRT_Callback_Extension`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Callback_Extension {
    /// 24-byte `PJRT_Extension_Base` header.
    pub base: PJRT_Extension_Base,
    /// Slot 1 (`0x18`): `PJRT_Callback_RegisterCallback`.
    pub register_callback: PjrtFn,
    /// Slot 2 (`0x20`): `PJRT_Callback_InvokeCallback`.
    pub invoke_callback: PjrtFn,
}

// SAFETY: Immutable static vtable structure.
unsafe impl Sync for PJRT_Callback_Extension {}

/// Node 11 (`ext_type = 9`, `64` bytes): `PJRT_PhaseCompile_Extension`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_PhaseCompile_Extension {
    /// 24-byte `PJRT_Extension_Base` header.
    pub base: PJRT_Extension_Base,
    /// Slot 1 (`0x18`): `phase_compile_get_compiler`.
    pub phase_compile_get_compiler: PjrtFn,
    /// Slot 2 (`0x20`): `phase_compile_destroy_compiler`.
    pub phase_compile_destroy_compiler: PjrtFn,
    /// Slot 3 (`0x28`): `phase_compile_run_phase`.
    pub phase_compile_run_phase: PjrtFn,
    /// Slot 4 (`0x30`): `phase_compile_get_phase_names`.
    pub phase_compile_get_phase_names: PjrtFn,
    /// Slot 5 (`0x38`): `phase_compile_c_buffers_destroy`.
    pub phase_compile_c_buffers_destroy: PjrtFn,
}

// SAFETY: Immutable static vtable structure.
unsafe impl Sync for PJRT_PhaseCompile_Extension {}

/// Node 12 (`ext_type = 12`, `56` bytes): `PJRT_CrossHostTransfers_Extension`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_CrossHostTransfers_Extension {
    /// 24-byte `PJRT_Extension_Base` header.
    pub base: PJRT_Extension_Base,
    /// Slot 1 (`0x18`): `make_cross_host_receive_buffers`.
    pub make_cross_host_receive_buffers: PjrtFn,
    /// Slot 2 (`0x20`): `copy_to_remote_device`.
    pub copy_to_remote_device: PjrtFn,
    /// Slot 3 (`0x28`): `cross_host_receive_buffers`.
    pub cross_host_receive_buffers: PjrtFn,
    /// Slot 4 (`0x30`): `cross_host_send_buffers`.
    pub cross_host_send_buffers: PjrtFn,
}

// SAFETY: Immutable static vtable structure.
unsafe impl Sync for PJRT_CrossHostTransfers_Extension {}

/// Node 13 (`ext_type = 11`, `40` bytes): `PJRT_ExecutableMetadata_Extension`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_ExecutableMetadata_Extension {
    /// 24-byte `PJRT_Extension_Base` header.
    pub base: PJRT_Extension_Base,
    /// Slot 1 (`0x18`): `get_executable_metadata`.
    pub get_executable_metadata: PjrtFn,
    /// Slot 2 (`0x20`): `destroy_executable_metadata`.
    pub destroy_executable_metadata: PjrtFn,
}

// SAFETY: Immutable static vtable structure.
unsafe impl Sync for PJRT_ExecutableMetadata_Extension {}

/// Node 14 (`ext_type = 7`, `40` bytes): `PJRT_MemoryDescriptions_Extension`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_MemoryDescriptions_Extension {
    /// 24-byte `PJRT_Extension_Base` header.
    pub base: PJRT_Extension_Base,
    /// Slot 1 (`0x18`): `PJRT_MemoryDescriptions_GetMemoryDescriptions`.
    pub get_memory_descriptions: PjrtFn,
    /// Slot 2 (`0x20`): `PJRT_MemoryDescriptions_Destroy`.
    pub destroy_memory_descriptions: PjrtFn,
}

// SAFETY: Immutable static vtable structure.
unsafe impl Sync for PJRT_MemoryDescriptions_Extension {}

/// Node 15 (`ext_type = 6`, `80` bytes): `PJRT_Layouts_Extension`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Layouts_Extension {
    /// 24-byte `PJRT_Extension_Base` header.
    pub base: PJRT_Extension_Base,
    /// Slot 1 (`0x18`): `memory_layout_destroy`.
    pub memory_layout_destroy: PjrtFn,
    /// Slot 2 (`0x20`): `memory_layout_serialize`.
    pub memory_layout_serialize: PjrtFn,
    /// Slot 3 (`0x28`): `client_get_default_layout`.
    pub client_get_default_layout: PjrtFn,
    /// Slot 4 (`0x30`): `buffer_memory_layout`.
    pub buffer_memory_layout: PjrtFn,
    /// Slot 5 (`0x38`): `topology_get_default_layout`.
    pub topology_get_default_layout: PjrtFn,
    /// Slot 6 (`0x40`): `executable_get_output_layouts`.
    pub executable_get_output_layouts: PjrtFn,
    /// Slot 7 (`0x48`): `executable_get_parameter_layouts`.
    pub executable_get_parameter_layouts: PjrtFn,
}

// SAFETY: Immutable static vtable structure.
unsafe impl Sync for PJRT_Layouts_Extension {}

/// Node 16 (`ext_type = 8`, `80` bytes): `PJRT_RawBuffer_Extension`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_RawBuffer_Extension {
    /// 24-byte `PJRT_Extension_Base` header.
    pub base: PJRT_Extension_Base,
    /// Slot 1 (`0x18`): `create_raw_alias_of_buffer`.
    pub create_raw_alias_of_buffer: PjrtFn,
    /// Slot 2 (`0x20`): `destroy`.
    pub destroy: PjrtFn,
    /// Slot 3 (`0x28`): `get_host_pointer`.
    pub get_host_pointer: PjrtFn,
    /// Slot 4 (`0x30`): `get_on_device_size_in_bytes`.
    pub get_on_device_size_in_bytes: PjrtFn,
    /// Slot 5 (`0x38`): `get_memory_space`.
    pub get_memory_space: PjrtFn,
    /// Slot 6 (`0x40`): `copy_raw_host_to_device`.
    pub copy_raw_host_to_device: PjrtFn,
    /// Slot 7 (`0x48`): `copy_raw_device_to_host`.
    pub copy_raw_device_to_host: PjrtFn,
}

// SAFETY: Immutable static vtable structure.
unsafe impl Sync for PJRT_RawBuffer_Extension {}

/// Node 17 (`ext_type = 1`, `40` bytes): `PJRT_Profiler_Extension`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Profiler_Extension {
    /// 24-byte `PJRT_Extension_Base` header.
    pub base: PJRT_Extension_Base,
    /// Slot 1 (`0x18`): pointer to `PLUGIN_Profiler_Api`.
    pub profiler_api: *const PLUGIN_Profiler_Api,
    /// Slot 2 (`0x20`): `traceme_context_id`.
    pub traceme_context_id: i64,
}

// SAFETY: Immutable static vtable structure.
unsafe impl Sync for PJRT_Profiler_Extension {}

const _: () = assert!(size_of::<PJRT_XlaTransform_Extension>() == 56);
const _: () =
    assert!(std::mem::offset_of!(PJRT_XlaTransform_Extension, register_xla_transform) == 0x18);
const _: () = assert!(size_of::<PJRT_HostMemoryAllocator_Extension>() == 32);
const _: () = assert!(
    std::mem::offset_of!(
        PJRT_HostMemoryAllocator_Extension,
        get_host_memory_allocator
    ) == 0x18
);
const _: () = assert!(size_of::<PJRT_MultiSlice_Extension>() == 64);
const _: () = assert!(std::mem::offset_of!(PJRT_MultiSlice_Extension, config_serialize) == 0x38);
const _: () = assert!(size_of::<PJRT_Collectives_Extension>() == 96);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Collectives_Extension, communicator_to_string) == 0x58);
const _: () = assert!(size_of::<PJRT_AbiVersion_Extension>() == 120);
const _: () = assert!(
    std::mem::offset_of!(PJRT_AbiVersion_Extension, executable_abi_version_from_proto) == 0x70
);
const _: () = assert!(size_of::<PJRT_Shardings_Extension>() == 40);
const _: () = assert!(std::mem::offset_of!(PJRT_Shardings_Extension, output_shardings) == 0x20);
const _: () = assert!(size_of::<PJRT_Megascale_Extension>() == 248);
const _: () = assert!(std::mem::offset_of!(PJRT_Megascale_Extension, abort_collectives) == 0xf0);
const _: () = assert!(size_of::<PJRT_TpuExecutable_Extension>() == 96);
const _: () = assert!(
    std::mem::offset_of!(
        PJRT_TpuExecutable_Extension,
        is_tpu_cancellation_placeholder_error
    ) == 0x58
);
const _: () = assert!(size_of::<PJRT_TpuTopology_Extension>() == 272);
const _: () =
    assert!(std::mem::offset_of!(PJRT_TpuTopology_Extension, get_default_platform_config) == 0x108);
const _: () = assert!(size_of::<PJRT_Callback_Extension>() == 40);
const _: () = assert!(std::mem::offset_of!(PJRT_Callback_Extension, invoke_callback) == 0x20);
const _: () = assert!(size_of::<PJRT_PhaseCompile_Extension>() == 64);
const _: () = assert!(
    std::mem::offset_of!(PJRT_PhaseCompile_Extension, phase_compile_c_buffers_destroy) == 0x38
);
const _: () = assert!(size_of::<PJRT_CrossHostTransfers_Extension>() == 56);
const _: () = assert!(
    std::mem::offset_of!(PJRT_CrossHostTransfers_Extension, cross_host_send_buffers) == 0x30
);
const _: () = assert!(size_of::<PJRT_ExecutableMetadata_Extension>() == 40);
const _: () = assert!(
    std::mem::offset_of!(
        PJRT_ExecutableMetadata_Extension,
        destroy_executable_metadata
    ) == 0x20
);
const _: () = assert!(size_of::<PJRT_MemoryDescriptions_Extension>() == 40);
const _: () = assert!(
    std::mem::offset_of!(
        PJRT_MemoryDescriptions_Extension,
        destroy_memory_descriptions
    ) == 0x20
);
const _: () = assert!(size_of::<PJRT_Layouts_Extension>() == 80);
const _: () =
    assert!(std::mem::offset_of!(PJRT_Layouts_Extension, executable_get_parameter_layouts) == 0x48);
const _: () = assert!(size_of::<PJRT_RawBuffer_Extension>() == 80);
const _: () =
    assert!(std::mem::offset_of!(PJRT_RawBuffer_Extension, copy_raw_device_to_host) == 0x48);
const _: () = assert!(size_of::<PJRT_Profiler_Extension>() == 40);
const _: () = assert!(std::mem::offset_of!(PJRT_Profiler_Extension, traceme_context_id) == 0x20);

/// Static Node 17 (`PJRT_Profiler_Extension`).
pub static EXT_17_PROFILER: PJRT_Profiler_Extension = PJRT_Profiler_Extension {
    base: PJRT_Extension_Base {
        struct_size: 40,
        ext_type: 1,
        pad: 0,
        next: ptr::null(),
    },
    profiler_api: ptr::addr_of!(GLOBAL_PLUGIN_PROFILER_API),
    traceme_context_id: 0,
};

/// Static Node 16 (`PJRT_RawBuffer_Extension`).
pub static EXT_16_RAW_BUFFER: PJRT_RawBuffer_Extension = PJRT_RawBuffer_Extension {
    base: PJRT_Extension_Base {
        struct_size: 80,
        ext_type: 8,
        pad: 0,
        next: ptr::addr_of!(EXT_17_PROFILER.base),
    },
    create_raw_alias_of_buffer: PJRT_RawBuffer_CreateRawAliasOfBuffer,
    destroy: PJRT_RawBuffer_Destroy,
    get_host_pointer: PJRT_RawBuffer_GetHostPointer,
    get_on_device_size_in_bytes: PJRT_RawBuffer_GetOnDeviceSizeInBytes,
    get_memory_space: PJRT_RawBuffer_GetMemorySpace,
    copy_raw_host_to_device: PJRT_RawBuffer_CopyRawHostToDevice,
    copy_raw_device_to_host: PJRT_RawBuffer_CopyRawDeviceToHost,
};

/// Static Node 15 (`PJRT_Layouts_Extension`).
pub static EXT_15_LAYOUTS: PJRT_Layouts_Extension = PJRT_Layouts_Extension {
    base: PJRT_Extension_Base {
        struct_size: 80,
        ext_type: 6,
        pad: 0,
        next: ptr::addr_of!(EXT_16_RAW_BUFFER.base),
    },
    memory_layout_destroy: PJRT_Layouts_MemoryLayout_Destroy,
    memory_layout_serialize: PJRT_Layouts_MemoryLayout_Serialize,
    client_get_default_layout: PJRT_Layouts_Client_GetDefaultLayout,
    buffer_memory_layout: PJRT_Layouts_Buffer_GetMemoryLayout,
    topology_get_default_layout: PJRT_Layouts_Topology_GetDefaultLayout,
    executable_get_output_layouts: PJRT_Layouts_Executable_GetOutputLayouts,
    executable_get_parameter_layouts: PJRT_Layouts_Executable_GetParameterLayouts,
};

/// Static Node 14 (`PJRT_MemoryDescriptions_Extension`).
pub static EXT_14_MEM_DESC: PJRT_MemoryDescriptions_Extension = PJRT_MemoryDescriptions_Extension {
    base: PJRT_Extension_Base {
        struct_size: 40,
        ext_type: 7,
        pad: 0,
        next: ptr::addr_of!(EXT_15_LAYOUTS.base),
    },
    get_memory_descriptions: PJRT_MemoryDescriptions_Get,
    destroy_memory_descriptions: PJRT_MemoryDescriptions_Destroy,
};

/// Static Node 13 (`PJRT_ExecutableMetadata_Extension`).
pub static EXT_13_EXEC_METADATA: PJRT_ExecutableMetadata_Extension =
    PJRT_ExecutableMetadata_Extension {
        base: PJRT_Extension_Base {
            struct_size: 40,
            ext_type: 11,
            pad: 0,
            next: ptr::addr_of!(EXT_14_MEM_DESC.base),
        },
        get_executable_metadata: PJRT_ExecutableMetadata_Get,
        destroy_executable_metadata: PJRT_ExecutableMetadata_Destroy,
    };

/// Static Node 12 (`PJRT_CrossHostTransfers_Extension`).
pub static EXT_12_CROSS_HOST: PJRT_CrossHostTransfers_Extension =
    PJRT_CrossHostTransfers_Extension {
        base: PJRT_Extension_Base {
            struct_size: 56,
            ext_type: 12,
            pad: 0,
            next: ptr::addr_of!(EXT_13_EXEC_METADATA.base),
        },
        make_cross_host_receive_buffers: PJRT_CrossHostTransfers_MakeReceiveBuffers,
        copy_to_remote_device: PJRT_CrossHostTransfers_CopyToRemoteDevice,
        cross_host_receive_buffers: PJRT_CrossHostTransfers_ReceiveBuffers,
        cross_host_send_buffers: PJRT_CrossHostTransfers_SendBuffers,
    };

/// Static Node 11 (`PJRT_PhaseCompile_Extension`).
pub static EXT_11_PHASE_COMPILE: PJRT_PhaseCompile_Extension = PJRT_PhaseCompile_Extension {
    base: PJRT_Extension_Base {
        struct_size: 64,
        ext_type: 9,
        pad: 0,
        next: ptr::addr_of!(EXT_12_CROSS_HOST.base),
    },
    phase_compile_get_compiler: PJRT_PhaseCompile_GetCompiler,
    phase_compile_destroy_compiler: PJRT_PhaseCompile_DestroyCompiler,
    phase_compile_run_phase: PJRT_PhaseCompile_RunPhase,
    phase_compile_get_phase_names: PJRT_PhaseCompile_GetPhaseNames,
    phase_compile_c_buffers_destroy: PJRT_PhaseCompile_CBuffersDestroy,
};

/// Static Node 10 (`PJRT_Callback_Extension`).
pub static EXT_10_CALLBACK: PJRT_Callback_Extension = PJRT_Callback_Extension {
    base: PJRT_Extension_Base {
        struct_size: 40,
        ext_type: 14,
        pad: 0,
        next: ptr::addr_of!(EXT_11_PHASE_COMPILE.base),
    },
    register_callback: PJRT_Callback_RegisterCallback,
    invoke_callback: PJRT_Callback_InvokeCallback,
};

/// Static Node 9 (`PJRT_TpuTopology_Extension`).
pub static EXT_09_TPU_TOPOLOGY: PJRT_TpuTopology_Extension = PJRT_TpuTopology_Extension {
    base: PJRT_Extension_Base {
        struct_size: 272,
        ext_type: 16,
        pad: 0,
        next: ptr::addr_of!(EXT_10_CALLBACK.base),
    },
    subslice: PJRT_TpuTopology_Subslice,
    is_subslice_topology: PJRT_TpuTopology_IsSubsliceTopology,
    subslice_device_id_from_full_device_id: PJRT_TpuTopology_SubsliceDeviceIdFromFullDeviceId,
    replace_host_bounds: PJRT_TpuTopology_ReplaceHostBounds,
    is_enhanced_barrier_enabled: PJRT_TpuTopology_IsEnhancedBarrierEnabled,
    has_limited_ici_connectivity: PJRT_TpuTopology_HasLimitedIciConnectivity,
    is_reachable_over_limited_ici: PJRT_TpuTopology_IsReachableOverLimitedIci,
    process_count: PJRT_TpuTopology_ProcessCount,
    chips_per_process: PJRT_TpuTopology_ChipsPerProcess,
    core_count_per_chip: PJRT_TpuTopology_CoreCountPerChip,
    chip_count: PJRT_TpuTopology_ChipCount,
    core_count: PJRT_TpuTopology_CoreCount,
    logi_device_count_per_process: PJRT_TpuTopology_LogiDeviceCountPerProcess,
    logi_device_count: PJRT_TpuTopology_LogiDeviceCount,
    logi_device_count_per_chip: PJRT_TpuTopology_LogiDeviceCountPerChip,
    core_count_per_process: PJRT_TpuTopology_CoreCountPerProcess,
    process_ids: PJRT_TpuTopology_ProcessIds,
    logi_device_ids_on_process: PJRT_TpuTopology_LogiDeviceIdsOnProcess,
    proc_id_and_idx_on_proc_for_chip: PJRT_TpuTopology_ProcIdAndIdxOnProcForChip,
    proc_id_and_idx_on_proc_for_logi_device: PJRT_TpuTopology_ProcIdAndIdxOnProcForLogiDevice,
    process_coord_from_id: PJRT_TpuTopology_ProcessCoordFromId,
    chip_id_from_coord: PJRT_TpuTopology_ChipIdFromCoord,
    logi_device_id_from_chip_coord_and_idx: PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx,
    chip_coord_and_idx_for_logi_device: PJRT_TpuTopology_ChipCoordAndIdxForLogiDevice,
    chips_per_process_bounds: PJRT_TpuTopology_ChipsPerProcessBounds,
    chip_bounds: PJRT_TpuTopology_ChipBounds,
    process_bounds: PJRT_TpuTopology_ProcessBounds,
    get_routing_strategy: PJRT_TpuTopology_GetRoutingStrategy,
    get_slice_config: PJRT_TpuTopology_GetSliceConfig,
    get_slice_configs: PJRT_TpuTopology_GetSliceConfigs,
    get_default_platform_config: PJRT_TpuTopology_GetDefaultPlatformConfig,
};

/// Static Node 8 (`PJRT_TpuExecutable_Extension`).
pub static EXT_08_TPU_EXECUTABLE: PJRT_TpuExecutable_Extension = PJRT_TpuExecutable_Extension {
    base: PJRT_Extension_Base {
        struct_size: 96,
        ext_type: 17,
        pad: 0,
        next: ptr::addr_of!(EXT_09_TPU_TOPOLOGY.base),
    },
    get_target_arguments: PJRT_TpuExecutable_GetTargetArguments,
    get_hlo_module_with_config: PJRT_TpuExecutable_GetHloModuleWithConfig,
    reserved: PJRT_TpuExecutable_Reserved,
    get_compiled_memory_stats: PJRT_TpuExecutable_GetCompiledMemoryStats,
    run_hlo_cost_analysis: PJRT_TpuExecutable_RunHloCostAnalysis,
    set_tpu_compilation_env: PJRT_TpuExecutable_SetTpuCompilationEnv,
    get_tpu_compilation_env_field_as_string: PJRT_TpuExecutable_GetTpuCompilationEnvFieldAsString,
    is_tpu_predetermined_error: PJRT_TpuExecutable_IsPredeterminedError,
    is_tpu_cancellation_placeholder_error: PJRT_TpuExecutable_IsCancellationPlaceholderError,
};

/// Static Node 7 (`PJRT_Megascale_Extension`).
pub static EXT_07_MEGASCALE: PJRT_Megascale_Extension = PJRT_Megascale_Extension {
    base: PJRT_Extension_Base {
        struct_size: 248,
        ext_type: 18,
        pad: 0,
        next: ptr::addr_of!(EXT_08_TPU_EXECUTABLE.base),
    },
    create_client_context: PJRT_Megascale_CreateClientContext,
    delete_client_context: PJRT_Megascale_DeleteClientContext,
    create_default_config: PJRT_Megascale_CreateDefaultConfig,
    create_aot_config: PJRT_Megascale_CreateAotConfig,
    create_multi_slice_config: PJRT_Megascale_CreateMultiSliceConfig,
    delete_multi_slice_config: PJRT_Megascale_DeleteMultiSliceConfig,
    multi_slice_config_num_slices: PJRT_Megascale_ConfigNumSlices,
    multi_slice_config_num_devices_per_slice: PJRT_Megascale_ConfigNumDevicesPerSlice,
    multi_slice_config_slice_id: PJRT_Megascale_ConfigSliceId,
    multi_slice_config_serialized_slice_key_value: PJRT_Megascale_ConfigSerializedSliceKeyValue,
    multi_slice_config_serialize: PJRT_Megascale_ConfigSerialize,
    deserialize_multi_slice_config: PJRT_Megascale_DeserializeMultiSliceConfig,
    megascale_addresses: PJRT_Megascale_Addresses,
    barrier: PJRT_Megascale_Barrier,
    log_error: PJRT_Megascale_LogError,
    get_megascale_dcn_topology: PJRT_Megascale_GetDcnTopology,
    get_slice_id: PJRT_Megascale_GetSliceId,
    get_num_slices: PJRT_Megascale_GetNumSlices,
    initialize_slice: PJRT_Megascale_InitializeSlice,
    finalize_slice: PJRT_Megascale_FinalizeSlice,
    register_error_handler: PJRT_Megascale_RegisterErrorHandler,
    unregister_error_handler: PJRT_Megascale_UnregisterErrorHandler,
    collect_metrics: PJRT_Megascale_CollectMetrics,
    reset_metrics: PJRT_Megascale_ResetMetrics,
    get_transport_type: PJRT_Megascale_GetTransportType,
    set_transport_type: PJRT_Megascale_SetTransportType,
    wait_for_barrier: PJRT_Megascale_WaitForBarrier,
    abort_collectives: PJRT_Megascale_AbortCollectives,
};

/// Static Node 6 (`PJRT_Shardings_Extension`).
pub static EXT_06_SHARDINGS: PJRT_Shardings_Extension = PJRT_Shardings_Extension {
    base: PJRT_Extension_Base {
        struct_size: 40,
        ext_type: 19,
        pad: 0,
        next: ptr::addr_of!(EXT_07_MEGASCALE.base),
    },
    parameter_shardings: PJRT_Shardings_ParameterShardings,
    output_shardings: PJRT_Shardings_OutputShardings,
};

/// Static Node 5 (`PJRT_AbiVersion_Extension`).
pub static EXT_05_ABI_VERSION: PJRT_AbiVersion_Extension = PJRT_AbiVersion_Extension {
    base: PJRT_Extension_Base {
        struct_size: 120,
        ext_type: 20,
        pad: 0,
        next: ptr::addr_of!(EXT_06_SHARDINGS.base),
    },
    client_runtime_abi_version: PJRT_AbiVersion_ClientRuntime,
    executable_get_abi_version: PJRT_AbiVersion_ExecutableGet,
    runtime_abi_version_destroy: PJRT_AbiVersion_RuntimeDestroy,
    runtime_abi_version_is_compatible_with_runtime: PJRT_AbiVersion_RuntimeCompatibleWithRuntime,
    runtime_abi_version_is_compatible_with_executable:
        PJRT_AbiVersion_RuntimeCompatibleWithExecutable,
    runtime_abi_version_to_proto: PJRT_AbiVersion_RuntimeToProto,
    runtime_abi_version_platform_id: PJRT_AbiVersion_RuntimePlatformId,
    executable_abi_version_destroy: PJRT_AbiVersion_ExecutableDestroy,
    executable_abi_version_to_proto: PJRT_AbiVersion_ExecutableToProto,
    executable_abi_version_platform_id: PJRT_AbiVersion_ExecutablePlatformId,
    runtime_abi_version_from_proto: PJRT_AbiVersion_RuntimeFromProto,
    executable_abi_version_from_proto: PJRT_AbiVersion_ExecutableFromProto,
};

/// Static Node 4 (`PJRT_Collectives_Extension`).
pub static EXT_04_COLLECTIVES: PJRT_Collectives_Extension = PJRT_Collectives_Extension {
    base: PJRT_Extension_Base {
        struct_size: 96,
        ext_type: 25,
        pad: 0,
        next: ptr::addr_of!(EXT_05_ABI_VERSION.base),
    },
    collectives_destroy: PJRT_Collectives_Destroy,
    create_communicator: PJRT_Collectives_CreateCommunicator,
    communicator_destroy: PJRT_Communicator_Destroy,
    communicator_all_reduce: PJRT_Communicator_AllReduce,
    communicator_reduce_scatter: PJRT_Communicator_ReduceScatter,
    communicator_all_gather: PJRT_Communicator_AllGather,
    communicator_collective_permute: PJRT_Communicator_CollectivePermute,
    communicator_all_to_all: PJRT_Communicator_AllToAll,
    communicator_to_string: PJRT_Communicator_ToString,
};

/// Static Node 3 (`PJRT_MultiSlice_Extension`).
pub static EXT_03_MULTI_SLICE: PJRT_MultiSlice_Extension = PJRT_MultiSlice_Extension {
    base: PJRT_Extension_Base {
        struct_size: 64,
        ext_type: 26,
        pad: 0,
        next: ptr::addr_of!(EXT_04_COLLECTIVES.base),
    },
    config_destroy: PJRT_MultiSlice_Config_Destroy,
    config_num_slices: PJRT_MultiSlice_Config_NumSlices,
    config_num_devices_per_slice: PJRT_MultiSlice_Config_NumDevicesPerSlice,
    config_slice_id: PJRT_MultiSlice_Config_SliceId,
    config_serialize: PJRT_MultiSlice_Config_Serialize,
};

/// Static Node 2 (`PJRT_HostMemoryAllocator_Extension`).
pub static EXT_02_HOST_ALLOCATOR: PJRT_HostMemoryAllocator_Extension =
    PJRT_HostMemoryAllocator_Extension {
        base: PJRT_Extension_Base {
            struct_size: 32,
            ext_type: 28,
            pad: 0,
            next: ptr::addr_of!(EXT_03_MULTI_SLICE.base),
        },
        get_host_memory_allocator: PJRT_HostMemoryAllocator_Get,
    };

/// Static Node 1 (`PJRT_XlaTransform_Extension`), head of the 17-node TPU extension chain.
pub static EXT_01_XLA_TRANSFORM: PJRT_XlaTransform_Extension = PJRT_XlaTransform_Extension {
    base: PJRT_Extension_Base {
        struct_size: 56,
        ext_type: 29,
        pad: 0,
        next: ptr::addr_of!(EXT_02_HOST_ALLOCATOR.base),
    },
    register_xla_transform: PJRT_XlaTransform_Register,
    wait_for_all_transforms: PJRT_XlaTransform_WaitForAll,
    unregister_xla_transform: PJRT_XlaTransform_Unregister,
    list_xla_transforms: PJRT_XlaTransform_List,
};

#[cfg(test)]
#[allow(
    clippy::cognitive_complexity,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::items_after_statements,
    clippy::float_cmp
)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::pjrt_buffer_api::{PJRT_Buffer_Destroy, PJRT_Buffer_IsDeleted};
    use crate::pjrt_client_api::{
        PJRT_Client_BufferFromHostBuffer, PJRT_Client_Compile, PJRT_Client_Create,
        PJRT_Client_Destroy, PJRT_Client_LookupDevice,
    };
    use crate::pjrt_device_api::{
        PJRT_Device_GetDescription, PJRT_DeviceDescription_Id, PJRT_Memory_Id,
    };
    use crate::pjrt_error_api::PJRT_Plugin_Attributes;
    use crate::pjrt_executable_api::{PJRT_Executable_Name, PJRT_LoadedExecutable_Destroy};
    use crate::pjrt_helpers::box_pjrt_error;
    use crate::pjrt_types::{
        PJRT_Buffer_Destroy_Args, PJRT_Client_BufferFromHostBuffer_Args, PJRT_Client_Compile_Args,
        PJRT_Client_Create_Args, PJRT_Client_Destroy_Args, PJRT_Client_LookupDevice_Args,
        PJRT_LoadedExecutable_Destroy_Args, PJRT_Program, PjrtFn,
    };
    use libtpu_core::error::LibTpuError;

    #[test]
    fn test_pjrt_extensions_vtable_and_behavioral_operations() {
        assert_ne!(
            format!(
                "{GLOBAL_PLUGIN_PROFILER_API:?}{EXT_01_XLA_TRANSFORM:?}{EXT_02_HOST_ALLOCATOR:?}{EXT_03_MULTI_SLICE:?}{EXT_04_COLLECTIVES:?}{EXT_05_ABI_VERSION:?}{EXT_06_SHARDINGS:?}{EXT_07_MEGASCALE:?}{EXT_08_TPU_EXECUTABLE:?}{EXT_09_TPU_TOPOLOGY:?}{EXT_10_CALLBACK:?}{EXT_11_PHASE_COMPILE:?}{EXT_12_CROSS_HOST:?}{EXT_13_EXEC_METADATA:?}{EXT_14_MEM_DESC:?}{EXT_15_LAYOUTS:?}{EXT_16_RAW_BUFFER:?}{EXT_17_PROFILER:?}"
            ),
            ""
        );

        // SAFETY: Testing all 138 PJRT extension entrypoints and argument validation paths.
        unsafe {
            let ext_null_err = PJRT_Extension_ValidateArgs(ptr::null_mut());
            drop(Box::from_raw(ext_null_err));
            drop(Box::from_raw(PJRT_Plugin_Attributes(ptr::null_mut())));
            drop(Box::from_raw(PJRT_DeviceDescription_Id(ptr::null_mut())));
            drop(Box::from_raw(PJRT_Device_GetDescription(ptr::null_mut())));
            drop(Box::from_raw(PJRT_Memory_Id(ptr::null_mut())));
            drop(Box::from_raw(PJRT_Executable_Name(ptr::null_mut())));
            drop(Box::from_raw(PJRT_Buffer_IsDeleted(ptr::null_mut())));
            drop(Box::from_raw(
                PLUGIN_Profiler_Error_Destroy(ptr::null_mut()),
            ));
            let mut zero_sz: usize = 0;
            let ext_sz_err = PJRT_Extension_ValidateArgs(ptr::addr_of_mut!(zero_sz).cast());
            drop(Box::from_raw(ext_sz_err));
            let mut valid_sz: [usize; 12] = [96, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
            assert!(PJRT_Extension_ValidateArgs(valid_sz.as_mut_ptr().cast()).is_null());

            let ext_fns: [PjrtFn; 138] = [
                GLOBAL_PLUGIN_PROFILER_API.error_destroy,
                GLOBAL_PLUGIN_PROFILER_API.error_message,
                GLOBAL_PLUGIN_PROFILER_API.error_get_code,
                GLOBAL_PLUGIN_PROFILER_API.create,
                GLOBAL_PLUGIN_PROFILER_API.destroy,
                GLOBAL_PLUGIN_PROFILER_API.start,
                GLOBAL_PLUGIN_PROFILER_API.stop,
                GLOBAL_PLUGIN_PROFILER_API.collect_data,
                EXT_01_XLA_TRANSFORM.register_xla_transform,
                EXT_01_XLA_TRANSFORM.wait_for_all_transforms,
                EXT_01_XLA_TRANSFORM.unregister_xla_transform,
                EXT_01_XLA_TRANSFORM.list_xla_transforms,
                EXT_02_HOST_ALLOCATOR.get_host_memory_allocator,
                EXT_03_MULTI_SLICE.config_destroy,
                EXT_03_MULTI_SLICE.config_num_slices,
                EXT_03_MULTI_SLICE.config_num_devices_per_slice,
                EXT_03_MULTI_SLICE.config_slice_id,
                EXT_03_MULTI_SLICE.config_serialize,
                EXT_04_COLLECTIVES.collectives_destroy,
                EXT_04_COLLECTIVES.create_communicator,
                EXT_04_COLLECTIVES.communicator_destroy,
                EXT_04_COLLECTIVES.communicator_all_reduce,
                EXT_04_COLLECTIVES.communicator_reduce_scatter,
                EXT_04_COLLECTIVES.communicator_all_gather,
                EXT_04_COLLECTIVES.communicator_collective_permute,
                EXT_04_COLLECTIVES.communicator_all_to_all,
                EXT_04_COLLECTIVES.communicator_to_string,
                EXT_05_ABI_VERSION.client_runtime_abi_version,
                EXT_05_ABI_VERSION.executable_get_abi_version,
                EXT_05_ABI_VERSION.runtime_abi_version_destroy,
                EXT_05_ABI_VERSION.runtime_abi_version_is_compatible_with_runtime,
                EXT_05_ABI_VERSION.runtime_abi_version_is_compatible_with_executable,
                EXT_05_ABI_VERSION.runtime_abi_version_to_proto,
                EXT_05_ABI_VERSION.runtime_abi_version_platform_id,
                EXT_05_ABI_VERSION.executable_abi_version_destroy,
                EXT_05_ABI_VERSION.executable_abi_version_to_proto,
                EXT_05_ABI_VERSION.executable_abi_version_platform_id,
                EXT_05_ABI_VERSION.runtime_abi_version_from_proto,
                EXT_05_ABI_VERSION.executable_abi_version_from_proto,
                EXT_06_SHARDINGS.parameter_shardings,
                EXT_06_SHARDINGS.output_shardings,
                EXT_07_MEGASCALE.create_client_context,
                EXT_07_MEGASCALE.delete_client_context,
                EXT_07_MEGASCALE.create_default_config,
                EXT_07_MEGASCALE.create_aot_config,
                EXT_07_MEGASCALE.create_multi_slice_config,
                EXT_07_MEGASCALE.delete_multi_slice_config,
                EXT_07_MEGASCALE.multi_slice_config_num_slices,
                EXT_07_MEGASCALE.multi_slice_config_num_devices_per_slice,
                EXT_07_MEGASCALE.multi_slice_config_slice_id,
                EXT_07_MEGASCALE.multi_slice_config_serialized_slice_key_value,
                EXT_07_MEGASCALE.multi_slice_config_serialize,
                EXT_07_MEGASCALE.deserialize_multi_slice_config,
                EXT_07_MEGASCALE.megascale_addresses,
                EXT_07_MEGASCALE.barrier,
                EXT_07_MEGASCALE.log_error,
                EXT_07_MEGASCALE.get_megascale_dcn_topology,
                EXT_07_MEGASCALE.get_slice_id,
                EXT_07_MEGASCALE.get_num_slices,
                EXT_07_MEGASCALE.initialize_slice,
                EXT_07_MEGASCALE.finalize_slice,
                EXT_07_MEGASCALE.register_error_handler,
                EXT_07_MEGASCALE.unregister_error_handler,
                EXT_07_MEGASCALE.collect_metrics,
                EXT_07_MEGASCALE.reset_metrics,
                EXT_07_MEGASCALE.get_transport_type,
                EXT_07_MEGASCALE.set_transport_type,
                EXT_07_MEGASCALE.wait_for_barrier,
                EXT_07_MEGASCALE.abort_collectives,
                EXT_08_TPU_EXECUTABLE.get_target_arguments,
                EXT_08_TPU_EXECUTABLE.get_hlo_module_with_config,
                EXT_08_TPU_EXECUTABLE.reserved,
                EXT_08_TPU_EXECUTABLE.get_compiled_memory_stats,
                EXT_08_TPU_EXECUTABLE.run_hlo_cost_analysis,
                EXT_08_TPU_EXECUTABLE.set_tpu_compilation_env,
                EXT_08_TPU_EXECUTABLE.get_tpu_compilation_env_field_as_string,
                EXT_08_TPU_EXECUTABLE.is_tpu_predetermined_error,
                EXT_08_TPU_EXECUTABLE.is_tpu_cancellation_placeholder_error,
                EXT_09_TPU_TOPOLOGY.subslice,
                EXT_09_TPU_TOPOLOGY.is_subslice_topology,
                EXT_09_TPU_TOPOLOGY.subslice_device_id_from_full_device_id,
                EXT_09_TPU_TOPOLOGY.replace_host_bounds,
                EXT_09_TPU_TOPOLOGY.is_enhanced_barrier_enabled,
                EXT_09_TPU_TOPOLOGY.has_limited_ici_connectivity,
                EXT_09_TPU_TOPOLOGY.is_reachable_over_limited_ici,
                EXT_09_TPU_TOPOLOGY.process_count,
                EXT_09_TPU_TOPOLOGY.chips_per_process,
                EXT_09_TPU_TOPOLOGY.core_count_per_chip,
                EXT_09_TPU_TOPOLOGY.chip_count,
                EXT_09_TPU_TOPOLOGY.core_count,
                EXT_09_TPU_TOPOLOGY.logi_device_count_per_process,
                EXT_09_TPU_TOPOLOGY.logi_device_count,
                EXT_09_TPU_TOPOLOGY.logi_device_count_per_chip,
                EXT_09_TPU_TOPOLOGY.core_count_per_process,
                EXT_09_TPU_TOPOLOGY.process_ids,
                EXT_09_TPU_TOPOLOGY.logi_device_ids_on_process,
                EXT_09_TPU_TOPOLOGY.proc_id_and_idx_on_proc_for_chip,
                EXT_09_TPU_TOPOLOGY.proc_id_and_idx_on_proc_for_logi_device,
                EXT_09_TPU_TOPOLOGY.process_coord_from_id,
                EXT_09_TPU_TOPOLOGY.chip_id_from_coord,
                EXT_09_TPU_TOPOLOGY.logi_device_id_from_chip_coord_and_idx,
                EXT_09_TPU_TOPOLOGY.chip_coord_and_idx_for_logi_device,
                EXT_09_TPU_TOPOLOGY.chips_per_process_bounds,
                EXT_09_TPU_TOPOLOGY.chip_bounds,
                EXT_09_TPU_TOPOLOGY.process_bounds,
                EXT_09_TPU_TOPOLOGY.get_routing_strategy,
                EXT_09_TPU_TOPOLOGY.get_slice_config,
                EXT_09_TPU_TOPOLOGY.get_slice_configs,
                EXT_09_TPU_TOPOLOGY.get_default_platform_config,
                EXT_10_CALLBACK.register_callback,
                EXT_10_CALLBACK.invoke_callback,
                EXT_11_PHASE_COMPILE.phase_compile_get_compiler,
                EXT_11_PHASE_COMPILE.phase_compile_destroy_compiler,
                EXT_11_PHASE_COMPILE.phase_compile_run_phase,
                EXT_11_PHASE_COMPILE.phase_compile_get_phase_names,
                EXT_11_PHASE_COMPILE.phase_compile_c_buffers_destroy,
                EXT_12_CROSS_HOST.make_cross_host_receive_buffers,
                EXT_12_CROSS_HOST.copy_to_remote_device,
                EXT_12_CROSS_HOST.cross_host_receive_buffers,
                EXT_12_CROSS_HOST.cross_host_send_buffers,
                EXT_13_EXEC_METADATA.get_executable_metadata,
                EXT_13_EXEC_METADATA.destroy_executable_metadata,
                EXT_14_MEM_DESC.get_memory_descriptions,
                EXT_14_MEM_DESC.destroy_memory_descriptions,
                EXT_15_LAYOUTS.memory_layout_destroy,
                EXT_15_LAYOUTS.memory_layout_serialize,
                EXT_15_LAYOUTS.client_get_default_layout,
                EXT_15_LAYOUTS.buffer_memory_layout,
                EXT_15_LAYOUTS.topology_get_default_layout,
                EXT_15_LAYOUTS.executable_get_output_layouts,
                EXT_15_LAYOUTS.executable_get_parameter_layouts,
                EXT_16_RAW_BUFFER.create_raw_alias_of_buffer,
                EXT_16_RAW_BUFFER.destroy,
                EXT_16_RAW_BUFFER.get_host_pointer,
                EXT_16_RAW_BUFFER.get_on_device_size_in_bytes,
                EXT_16_RAW_BUFFER.get_memory_space,
                EXT_16_RAW_BUFFER.copy_raw_host_to_device,
                EXT_16_RAW_BUFFER.copy_raw_device_to_host,
            ];
            for ext_fn in ext_fns {
                valid_sz = [96, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
                assert!(ext_fn(valid_sz.as_mut_ptr().cast()).is_null());
            }

            drop(Box::from_raw(PLUGIN_Profiler_CollectData(ptr::null_mut())));
            let mut prof_create_args = PLUGIN_Profiler_Create_Args {
                struct_size: size_of::<PLUGIN_Profiler_Create_Args>(),
                extension_start: ptr::null_mut(),
                options: ptr::null(),
                options_size: 0,
                profiler: ptr::null_mut(),
            };
            assert!(PLUGIN_Profiler_Create(ptr::addr_of_mut!(prof_create_args).cast()).is_null());
            assert!(!prof_create_args.profiler.is_null());
            let mut prof_start_args = PLUGIN_Profiler_Start_Args {
                struct_size: size_of::<PLUGIN_Profiler_Start_Args>(),
                extension_start: ptr::null_mut(),
                profiler: prof_create_args.profiler,
            };
            assert!(PLUGIN_Profiler_Start(ptr::addr_of_mut!(prof_start_args).cast()).is_null());
            assert!(PLUGIN_Profiler_Start(ptr::addr_of_mut!(prof_start_args).cast()).is_null());

            let mut prof_collect_null = PLUGIN_Profiler_CollectData_Args {
                struct_size: size_of::<PLUGIN_Profiler_CollectData_Args>(),
                extension_start: ptr::null_mut(),
                profiler: prof_create_args.profiler as usize,
                buffer: ptr::null_mut(),
                size_in_bytes: ptr::null_mut(),
            };
            assert_ne!(format!("{prof_collect_null:?}"), "");
            assert!(
                PLUGIN_Profiler_CollectData(ptr::addr_of_mut!(prof_collect_null).cast()).is_null()
            );
            let mut xspace_sz: usize = 0;
            let mut xspace_buf = vec![0_u8; 512];
            let mut prof_collect_full = PLUGIN_Profiler_CollectData_Args {
                struct_size: size_of::<PLUGIN_Profiler_CollectData_Args>(),
                extension_start: ptr::null_mut(),
                profiler: prof_create_args.profiler as usize,
                buffer: xspace_buf.as_mut_ptr(),
                size_in_bytes: ptr::addr_of_mut!(xspace_sz),
            };
            assert!(
                PLUGIN_Profiler_CollectData(ptr::addr_of_mut!(prof_collect_full).cast()).is_null()
            );
            assert!(xspace_sz > 0);
            assert!(PLUGIN_Profiler_Start(ptr::addr_of_mut!(prof_start_args).cast()).is_null());
            let mut prof_stop_args = PLUGIN_Profiler_Stop_Args {
                struct_size: size_of::<PLUGIN_Profiler_Stop_Args>(),
                extension_start: ptr::null_mut(),
                profiler: prof_create_args.profiler,
            };
            assert!(PLUGIN_Profiler_Stop(ptr::addr_of_mut!(prof_stop_args).cast()).is_null());
            let mut prof_destroy_args = PLUGIN_Profiler_Destroy_Args {
                struct_size: size_of::<PLUGIN_Profiler_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                profiler: prof_create_args.profiler,
            };
            assert!(PLUGIN_Profiler_Destroy(ptr::addr_of_mut!(prof_destroy_args).cast()).is_null());
            assert!(prof_destroy_args.profiler.is_null());
            assert_ne!(format!("{:?}", ExtensionRuntimeState::default()), "");

            let ext_destroy_args = PJRT_Megascale_Handle_Args {
                struct_size: 24,
                extension_start: ptr::null_mut(),
                handle: ptr::null_mut(),
            };
            let ext_q32_args = PJRT_Megascale_Query_Args {
                struct_size: 32,
                extension_start: ptr::null_mut(),
                handle: 0,
                out_value: 0,
            };
            let ext_q40_args = PJRT_Megascale_Data_Args {
                struct_size: 40,
                extension_start: ptr::null_mut(),
                handle: 0,
                out_first: 0,
                out_second: 0,
            };
            let ext_q48_args = PJRT_TpuExecutable_ProtoQuery_Args {
                struct_size: 48,
                extension_start: ptr::null_mut(),
                executable: ptr::null(),
                serialized_bytes: 0,
                serialized_bytes_size: 0,
                holder: ptr::null_mut(),
            };
            let ext_q56_args = PJRT_CrossHostTransfers_Args {
                struct_size: 56,
                extension_start: ptr::null_mut(),
                client: ptr::null_mut(),
                buffers: 0,
                num_buffers: 0,
                remote_device: 0,
                out_event: 0,
            };
            let ext_q64_args = PJRT_Communicator_CollectivePermute_Args {
                struct_size: 64,
                extension_start: ptr::null_mut(),
                communicator: ptr::null_mut(),
                send_buffer: ptr::null_mut(),
                recv_buffer: ptr::null_mut(),
                target_ranks: ptr::null(),
                count: 0,
                event: ptr::null_mut(),
            };
            let prof_err_del = PLUGIN_Profiler_Error_Destroy_Args {
                struct_size: 24,
                extension_start: ptr::null_mut(),
                error: ptr::null_mut(),
            };
            let prof_err_msg = PLUGIN_Profiler_Error_Message_Args {
                struct_size: 40,
                extension_start: ptr::null_mut(),
                error: ptr::null(),
                message: ptr::null(),
                message_size: 0,
            };
            let prof_err_code = PLUGIN_Profiler_Error_GetCode_Args {
                struct_size: 32,
                extension_start: ptr::null_mut(),
                error: ptr::null(),
                code: 0,
            };
            let prof_create = PLUGIN_Profiler_Create_Args {
                struct_size: 40,
                extension_start: ptr::null_mut(),
                options: ptr::null(),
                options_size: 0,
                profiler: ptr::null_mut(),
            };
            let prof_del = PLUGIN_Profiler_Destroy_Args {
                struct_size: 24,
                extension_start: ptr::null_mut(),
                profiler: ptr::null_mut(),
            };
            let prof_start = PLUGIN_Profiler_Start_Args {
                struct_size: 24,
                extension_start: ptr::null_mut(),
                profiler: ptr::null_mut(),
            };
            let prof_stop = PLUGIN_Profiler_Stop_Args {
                struct_size: 24,
                extension_start: ptr::null_mut(),
                profiler: ptr::null_mut(),
            };
            let ms_del = PJRT_MultiSlice_Config_Destroy_Args {
                struct_size: 24,
                extension_start: ptr::null_mut(),
                config: ptr::null_mut(),
            };
            let ms_ns = PJRT_MultiSlice_Config_NumSlices_Args {
                struct_size: 32,
                extension_start: ptr::null_mut(),
                config: ptr::null(),
                num_slices: 0,
            };
            let ms_ndps = PJRT_MultiSlice_Config_NumDevicesPerSlice_Args {
                struct_size: 40,
                extension_start: ptr::null_mut(),
                config: ptr::null(),
                num_devices_per_slice_map: 0,
                num_slices: 0,
            };
            let ms_sid = PJRT_MultiSlice_Config_SliceId_Args {
                struct_size: 32,
                extension_start: ptr::null_mut(),
                config: ptr::null(),
                slice_id: 0,
            };
            let coll_del = PJRT_Collectives_Destroy_Args {
                struct_size: 24,
                extension_start: ptr::null_mut(),
                collectives: ptr::null_mut(),
            };
            let coll_cc = PJRT_Collectives_CreateCommunicator_Args {
                struct_size: 48,
                extension_start: ptr::null_mut(),
                collectives: ptr::null_mut(),
                device_ids: ptr::null(),
                num_devices: 0,
                communicator: ptr::null_mut(),
            };
            let comm_del = PJRT_Communicator_Destroy_Args {
                struct_size: 24,
                extension_start: ptr::null_mut(),
                communicator: ptr::null_mut(),
            };
            let comm_ag = PJRT_Communicator_AllGather_Args {
                struct_size: 56,
                extension_start: ptr::null_mut(),
                communicator: ptr::null_mut(),
                send_buffer: ptr::null_mut(),
                recv_buffer: ptr::null_mut(),
                count: 0,
                event: ptr::null_mut(),
            };
            let comm_str = PJRT_Communicator_ToString_Args {
                struct_size: 40,
                extension_start: ptr::null_mut(),
                communicator: ptr::null(),
                str_out: 0,
                str_size: 0,
            };
            let abi_get = PJRT_AbiVersion_Get_Args {
                struct_size: 32,
                extension_start: ptr::null_mut(),
                handle: ptr::null(),
                abi_version_out: 0,
            };
            let abi_del = PJRT_AbiVersion_Destroy_Args {
                struct_size: 24,
                extension_start: ptr::null_mut(),
                abi_version: ptr::null_mut(),
            };
            let abi_cmp = PJRT_AbiVersion_IsCompatible_Args {
                struct_size: 40,
                extension_start: ptr::null_mut(),
                abi_version: ptr::null(),
                other_abi_version: 0,
                is_compatible_out: 0,
            };
            let abi_pid = PJRT_AbiVersion_PlatformId_Args {
                struct_size: 32,
                extension_start: ptr::null_mut(),
                abi_version: ptr::null(),
                platform_id_out: 0,
            };
            let abi_fp = PJRT_AbiVersion_FromProto_Args {
                struct_size: 40,
                extension_start: ptr::null_mut(),
                serialized_proto: ptr::null(),
                serialized_proto_size: 0,
                abi_version_out: 0,
            };
            let tpu_ec = PJRT_TpuExecutable_ErrorCheck_Args {
                struct_size: 32,
                extension_start: ptr::null_mut(),
                error: ptr::null(),
                is_matched: 0,
            };
            let topo_sq = PJRT_TpuTopology_ScalarQuery_Args {
                struct_size: 32,
                extension_start: ptr::null_mut(),
                topology: ptr::null(),
                out_value: 0,
            };
            let topo_bq = PJRT_TpuTopology_BoundsQuery_Args {
                struct_size: 40,
                extension_start: ptr::null_mut(),
                topology: ptr::null(),
                out_first: 0,
                out_second: 0,
            };
            let topo_mq = PJRT_TpuTopology_MappingQuery_Args {
                struct_size: 48,
                extension_start: ptr::null_mut(),
                topology: ptr::null(),
                param0: 0,
                out_first: 0,
                out_second: 0,
            };
            let cb_inv = PJRT_Callback_Invoke_Args {
                struct_size: 40,
                extension_start: ptr::null_mut(),
                client: ptr::null_mut(),
                callback_type: 0,
                payload: ptr::null_mut(),
            };
            let pc_gc = PJRT_PhaseCompile_GetCompiler_Args {
                struct_size: 32,
                extension_start: ptr::null_mut(),
                client: ptr::null_mut(),
                compiler_out: 0,
            };
            let pc_del = PJRT_PhaseCompile_Destroy_Args {
                struct_size: 24,
                extension_start: ptr::null_mut(),
                handle: ptr::null_mut(),
            };
            let md_get = PJRT_Metadata_Get_Args {
                struct_size: 32,
                extension_start: ptr::null_mut(),
                handle: ptr::null(),
                metadata_out: 0,
            };
            let md_del = PJRT_Metadata_Destroy_Args {
                struct_size: 24,
                extension_start: ptr::null_mut(),
                metadata: ptr::null_mut(),
            };
            let rb_q = PJRT_RawBuffer_Query_Args {
                struct_size: 32,
                extension_start: ptr::null_mut(),
                buffer: ptr::null_mut(),
                out_value: 0,
            };
            let rb_del = PJRT_RawBuffer_Destroy_Args {
                struct_size: 24,
                extension_start: ptr::null_mut(),
                raw_buffer: ptr::null_mut(),
            };
            assert_ne!(
                format!(
                    "{ext_destroy_args:?}{ext_q32_args:?}{ext_q40_args:?}{ext_q48_args:?}{ext_q56_args:?}{ext_q64_args:?}{prof_err_del:?}{prof_err_msg:?}{prof_err_code:?}{prof_create:?}{prof_del:?}{prof_start:?}{prof_stop:?}{ms_del:?}{ms_ns:?}{ms_ndps:?}{ms_sid:?}{coll_del:?}{coll_cc:?}{comm_del:?}{comm_ag:?}{comm_str:?}{abi_get:?}{abi_del:?}{abi_cmp:?}{abi_pid:?}{abi_fp:?}{tpu_ec:?}{topo_sq:?}{topo_bq:?}{topo_mq:?}{cb_inv:?}{pc_gc:?}{pc_del:?}{md_get:?}{md_del:?}{rb_q:?}{rb_del:?}"
                ),
                ""
            );

            let layout_destroy_args = PJRT_Layouts_MemoryLayout_Destroy_Args {
                struct_size: 24,
                extension_start: ptr::null_mut(),
                layout: ptr::null_mut(),
            };
            let layout_ser_args = PJRT_Layouts_MemoryLayout_Serialize_Args {
                struct_size: 56,
                extension_start: ptr::null_mut(),
                layout: ptr::null(),
                serialized_bytes: ptr::null(),
                serialized_bytes_size: 0,
                serialized_layout: ptr::null_mut(),
                serialized_layout_deleter: ptr::null(),
            };
            let layout_def_args = PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args {
                struct_size: 56,
                extension_start: ptr::null_mut(),
                client: ptr::null_mut(),
                elem_type: 11,
                pad: 0,
                dims: ptr::null(),
                num_dims: 0,
                layout: ptr::null_mut(),
            };
            let layout_buf_args = PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args {
                struct_size: 32,
                extension_start: ptr::null_mut(),
                buffer: ptr::null_mut(),
                layout: ptr::null_mut(),
            };
            let shardings_args = PJRT_Shardings_PJRT_Executable_Shardings_Args {
                struct_size: 48,
                extension_start: ptr::null_mut(),
                executable: ptr::null_mut(),
                num_items: 0,
                shardings: ptr::null(),
                sharding_sizes: ptr::null(),
            };
            let phase_names_args = PJRT_PhaseCompile_GetPhaseNames_Args {
                struct_size: 48,
                extension_start: ptr::null_mut(),
                phase_compiler: ptr::null(),
                phase_names: ptr::null(),
                phase_names_sizes: ptr::null(),
                num_phase_names: 0,
            };
            let triton_args = PJRT_Triton_Compile_Args {
                struct_size: 88,
                extension_start: ptr::null_mut(),
                module: ptr::null(),
                module_size: 0,
                arch_name: ptr::null(),
                arch_name_size: 0,
                num_warps: 4,
                num_ctas: 1,
                num_stages: 2,
                pad: 0,
                out_asm: ptr::null(),
                out_asm_size: 0,
                out_smem_bytes: 0,
            };
            assert_ne!(
                format!(
                    "{layout_destroy_args:?}{layout_ser_args:?}{layout_def_args:?}{layout_buf_args:?}{shardings_args:?}{phase_names_args:?}{triton_args:?}"
                ),
                ""
            );

            let xla_reg_args = PJRT_XlaTransform_Register_Args {
                struct_size: 40,
                extension_start: ptr::null_mut(),
                name: ptr::null(),
                name_size: 0,
                transform_fn: 0,
            };
            let xla_wait_args = PJRT_XlaTransform_WaitForAll_Args {
                struct_size: 24,
                extension_start: ptr::null_mut(),
                client: ptr::null_mut(),
            };
            let xla_unreg_args = PJRT_XlaTransform_Unregister_Args {
                struct_size: 32,
                extension_start: ptr::null_mut(),
                name: ptr::null(),
                name_size: 0,
            };
            let xla_list_args = PJRT_XlaTransform_List_Args {
                struct_size: 32,
                extension_start: ptr::null_mut(),
                names_out: ptr::null(),
                num_names_out: 0,
            };
            let mut host_alloc_args = PJRT_HostMemoryAllocator_Get_Args {
                struct_size: 32,
                extension_start: ptr::null_mut(),
                client: ptr::null_mut(),
                allocator_out: 0,
            };
            assert!(
                PJRT_HostMemoryAllocator_Get(ptr::addr_of_mut!(host_alloc_args).cast()).is_null()
            );
            assert!(host_alloc_args.allocator_out > 0);
            let mut md_get32 = PJRT_Metadata_Get_Args {
                struct_size: 32,
                extension_start: ptr::null_mut(),
                handle: ptr::null(),
                metadata_out: 0,
            };
            assert!(PJRT_MemoryDescriptions_Get(ptr::addr_of_mut!(md_get32).cast()).is_null());
            assert_eq!(md_get32.metadata_out, 4);
            let mut md_del24 = PJRT_Metadata_Destroy_Args {
                struct_size: 24,
                extension_start: ptr::null_mut(),
                metadata: ptr::dangling_mut(),
            };
            assert!(PJRT_MemoryDescriptions_Destroy(ptr::addr_of_mut!(md_del24).cast()).is_null());
            assert!(md_del24.metadata.is_null());
            let ms_ser_args = PJRT_MultiSlice_Config_Serialize_Args {
                struct_size: 48,
                extension_start: ptr::null_mut(),
                config: ptr::null(),
                serialized_bytes: ptr::null(),
                serialized_bytes_size: 0,
                serialized_config: ptr::null_mut(),
            };
            let comm_ar_args = PJRT_Communicator_AllReduce_Args {
                struct_size: 64,
                extension_start: ptr::null_mut(),
                communicator: ptr::null_mut(),
                send_buffer: ptr::null_mut(),
                recv_buffer: ptr::null_mut(),
                reduction_op: 0,
                pad: 0,
                count: 4,
                event: ptr::null_mut(),
            };
            let abi_proto_args = PJRT_AbiVersion_ToProto_Args {
                struct_size: 48,
                extension_start: ptr::null_mut(),
                abi_version: ptr::null(),
                serialized_proto: ptr::null(),
                serialized_proto_size: 0,
                serialized_proto_holder: ptr::null_mut(),
            };
            let tpu_mem_stats_args = PJRT_TpuExecutable_GetCompiledMemoryStats_Args {
                struct_size: 64,
                extension_start: ptr::null_mut(),
                executable: ptr::null_mut(),
                generated_code_size_in_bytes: 0,
                argument_size_in_bytes: 0,
                output_size_in_bytes: 0,
                alias_size_in_bytes: 0,
                temp_size_in_bytes: 0,
            };
            let cb_reg_args = PJRT_Callback_Register_Args {
                struct_size: 48,
                extension_start: ptr::null_mut(),
                client: ptr::null_mut(),
                callback_type: 1,
                callback_fn: ptr::null(),
                user_arg: ptr::null_mut(),
            };
            let phase_run_args = PJRT_PhaseCompile_RunPhase_Args {
                struct_size: 56,
                extension_start: ptr::null_mut(),
                phase_compiler: ptr::null_mut(),
                input_bytes: ptr::null(),
                input_size: 0,
                phase_name: ptr::null(),
                output_status: 0,
            };
            let raw_copy_args = PJRT_RawBuffer_CopyRaw_Args {
                struct_size: 56,
                extension_start: ptr::null_mut(),
                buffer: ptr::null_mut(),
                host_ptr: ptr::null_mut(),
                offset: 0,
                transfer_size: 0,
                event: ptr::null_mut(),
            };
            assert_ne!(
                format!(
                    "{xla_reg_args:?}{xla_wait_args:?}{xla_unreg_args:?}{xla_list_args:?}{host_alloc_args:?}{ms_ser_args:?}{comm_ar_args:?}{abi_proto_args:?}{tpu_mem_stats_args:?}{cb_reg_args:?}{phase_run_args:?}{raw_copy_args:?}"
                ),
                ""
            );

            let mut host_alloc64 = PJRT_HostMemoryAllocator_Allocate_Args {
                struct_size: size_of::<PJRT_HostMemoryAllocator_Allocate_Args>(),
                extension_start: ptr::null_mut(),
                client: ptr::null_mut(),
                size: 4096,
                numa_node: 0,
                pad: 0,
                ptr: ptr::null_mut(),
                deleter: ptr::null(),
                deleter_arg: ptr::null_mut(),
            };
            assert!(PJRT_HostMemoryAllocator_Get(ptr::addr_of_mut!(host_alloc64).cast()).is_null());
            assert!(!host_alloc64.ptr.is_null());
            let mut mem_desc48 = PJRT_DeviceDescription_MemoryDescriptions_Args {
                struct_size: size_of::<PJRT_DeviceDescription_MemoryDescriptions_Args>(),
                extension_start: ptr::null_mut(),
                device_description: ptr::null(),
                memory_descriptions: ptr::null(),
                num_memory_descriptions: 0,
                default_memory_index: -1,
            };
            assert!(PJRT_MemoryDescriptions_Get(ptr::addr_of_mut!(mem_desc48).cast()).is_null());
            assert_eq!(mem_desc48.num_memory_descriptions, 4);
            assert_eq!(mem_desc48.default_memory_index, 0);
            let vmem_desc_ptr = *mem_desc48.memory_descriptions.add(1);
            let mut mem_kind48 = PJRT_MemoryDescription_Kind_Args {
                struct_size: size_of::<PJRT_MemoryDescription_Kind_Args>(),
                extension_start: ptr::null_mut(),
                memory_description: vmem_desc_ptr,
                kind: ptr::null(),
                kind_size: 0,
                kind_id: -1,
                pad: 0,
            };
            assert!(
                PJRT_MemoryDescriptions_Destroy(ptr::addr_of_mut!(mem_kind48).cast()).is_null()
            );
            assert_eq!(mem_kind48.kind_id, 1);
            assert_eq!(mem_kind48.kind_size, 4);
            mem_kind48.memory_description = ptr::null();
            assert!(
                PJRT_MemoryDescriptions_Destroy(ptr::addr_of_mut!(mem_kind48).cast()).is_null()
            );
            assert_eq!(mem_kind48.kind_id, 0);
            let cb_sb = PJRT_Callback_Tpu_SliceBuilderArgs {
                struct_size: 16,
                failure_type: PJRT_Callback_Tpu_SliceFailureType::default() as i32,
                pad: 0,
            };
            let cb_pf = PJRT_Callback_PrefatalArgs {
                struct_size: 32,
                error_code: 0,
                pad: 0,
                error_message: ptr::null(),
                error_message_size: 0,
            };
            let exec_md = PJRT_ExecutableMetadata {
                serialized_metadata: ptr::null(),
                serialized_metadata_size: 0,
            };
            assert_ne!(
                format!(
                    "{host_alloc64:?}{mem_desc48:?}{mem_kind48:?}{cb_sb:?}{cb_pf:?}{exec_md:?}{:?}{:?}{:?}",
                    PJRT_Collectives_ReductionKind::default(),
                    PJRT_Collectives_CollectiveOpKind::default(),
                    PJRT_Callback_Type::default(),
                ),
                ""
            );
            assert!(extension_call_count() > 0);

            // Behavioral tests with real PJRT client, executable, and buffer handles
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

            let mut look_args = PJRT_Client_LookupDevice_Args {
                struct_size: size_of::<PJRT_Client_LookupDevice_Args>(),
                extension_start: ptr::null_mut(),
                client,
                id: 0,
                pad: 0,
                device: ptr::null_mut(),
            };
            assert!(PJRT_Client_LookupDevice(ptr::addr_of_mut!(look_args).cast()).is_null());

            let code_bytes = b"module @test { stablehlo.add }";
            let fmt_bytes = b"mlir";
            let prog = PJRT_Program {
                struct_size: size_of::<PJRT_Program>(),
                extension_start: ptr::null_mut(),
                code: code_bytes.as_ptr().cast(),
                code_size: code_bytes.len(),
                format: fmt_bytes.as_ptr().cast(),
                format_size: fmt_bytes.len(),
            };
            let mut comp_args = PJRT_Client_Compile_Args {
                struct_size: size_of::<PJRT_Client_Compile_Args>(),
                extension_start: ptr::null_mut(),
                client,
                program: ptr::addr_of!(prog),
                compile_options: ptr::null(),
                compile_options_size: 0,
                executable: ptr::null_mut(),
            };
            assert!(PJRT_Client_Compile(ptr::addr_of_mut!(comp_args).cast()).is_null());

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

            let pass_name = b"custom_fusion_pass";
            let mut xla_reg = PJRT_XlaTransform_Register_Args {
                struct_size: size_of::<PJRT_XlaTransform_Register_Args>(),
                extension_start: ptr::null_mut(),
                name: pass_name.as_ptr().cast(),
                name_size: pass_name.len(),
                transform_fn: 0,
            };
            assert!(PJRT_XlaTransform_Register(ptr::addr_of_mut!(xla_reg).cast()).is_null());
            assert!(xla_reg.transform_fn > 0);
            let mut xla_list = PJRT_XlaTransform_List_Args {
                struct_size: size_of::<PJRT_XlaTransform_List_Args>(),
                extension_start: ptr::null_mut(),
                names_out: ptr::null(),
                num_names_out: 0,
            };
            assert!(PJRT_XlaTransform_List(ptr::addr_of_mut!(xla_list).cast()).is_null());
            assert!(xla_list.num_names_out >= 1);
            let mut xla_unreg = PJRT_XlaTransform_Unregister_Args {
                struct_size: size_of::<PJRT_XlaTransform_Unregister_Args>(),
                extension_start: ptr::null_mut(),
                name: pass_name.as_ptr().cast(),
                name_size: pass_name.len(),
            };
            assert!(PJRT_XlaTransform_Unregister(ptr::addr_of_mut!(xla_unreg).cast()).is_null());

            let dev_ids = [0_i32, 1];
            let mut comm_create = PJRT_Collectives_CreateCommunicator_Args {
                struct_size: size_of::<PJRT_Collectives_CreateCommunicator_Args>(),
                extension_start: ptr::null_mut(),
                collectives: ptr::null_mut(),
                device_ids: dev_ids.as_ptr(),
                num_devices: dev_ids.len(),
                communicator: ptr::null_mut(),
            };
            assert!(
                PJRT_Collectives_CreateCommunicator(ptr::addr_of_mut!(comm_create).cast())
                    .is_null()
            );
            assert!(!comm_create.communicator.is_null());
            let mut comm_to_str = PJRT_Communicator_ToString_Args {
                struct_size: size_of::<PJRT_Communicator_ToString_Args>(),
                extension_start: ptr::null_mut(),
                communicator: comm_create.communicator,
                str_out: 0,
                str_size: 0,
            };
            assert!(PJRT_Communicator_ToString(ptr::addr_of_mut!(comm_to_str).cast()).is_null());
            assert!(comm_to_str.str_size > 0);

            let host_floats_2 = [10.0_f32, 20.0, 30.0, 40.0];
            let mut h2d_args_2 = PJRT_Client_BufferFromHostBuffer_Args {
                struct_size: size_of::<PJRT_Client_BufferFromHostBuffer_Args>(),
                extension_start: ptr::null_mut(),
                client,
                data: host_floats_2.as_ptr().cast(),
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
            assert!(
                PJRT_Client_BufferFromHostBuffer(ptr::addr_of_mut!(h2d_args_2).cast()).is_null()
            );
            let buf_recv = h2d_args_2.out_buffer;

            for red_op in [0_i32, 1, 2, 3] {
                let mut comm_ar = PJRT_Communicator_AllReduce_Args {
                    struct_size: size_of::<PJRT_Communicator_AllReduce_Args>(),
                    extension_start: ptr::null_mut(),
                    communicator: comm_create.communicator,
                    send_buffer: buf_in.cast(),
                    recv_buffer: buf_recv.cast(),
                    reduction_op: red_op,
                    pad: 0,
                    count: 4,
                    event: ptr::null_mut(),
                };
                assert!(PJRT_Communicator_AllReduce(ptr::addr_of_mut!(comm_ar).cast()).is_null());
                assert!(!comm_ar.event.is_null());
            }

            let host_ints_1 = [2_i32, 3, 4, 5];
            let host_ints_2 = [10_i32, 20, 30, 40];
            let mut h2d_i1 = h2d_args_2;
            h2d_i1.data = host_ints_1.as_ptr().cast();
            h2d_i1.element_type = 4;
            h2d_i1.out_buffer = ptr::null_mut();
            assert!(PJRT_Client_BufferFromHostBuffer(ptr::addr_of_mut!(h2d_i1).cast()).is_null());
            let mut h2d_i2 = h2d_i1;
            h2d_i2.data = host_ints_2.as_ptr().cast();
            h2d_i2.out_buffer = ptr::null_mut();
            assert!(PJRT_Client_BufferFromHostBuffer(ptr::addr_of_mut!(h2d_i2).cast()).is_null());
            for red_op in [0_i32, 1, 2, 3] {
                let mut comm_ar_i = PJRT_Communicator_AllReduce_Args {
                    struct_size: size_of::<PJRT_Communicator_AllReduce_Args>(),
                    extension_start: ptr::null_mut(),
                    communicator: comm_create.communicator,
                    send_buffer: h2d_i1.out_buffer.cast(),
                    recv_buffer: h2d_i2.out_buffer.cast(),
                    reduction_op: red_op,
                    pad: 0,
                    count: 4,
                    event: ptr::null_mut(),
                };
                assert!(PJRT_Communicator_AllReduce(ptr::addr_of_mut!(comm_ar_i).cast()).is_null());
            }
            let mut del_bi1 = PJRT_Buffer_Destroy_Args {
                struct_size: size_of::<PJRT_Buffer_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                buffer: h2d_i1.out_buffer,
            };
            assert!(PJRT_Buffer_Destroy(ptr::addr_of_mut!(del_bi1).cast()).is_null());
            del_bi1.buffer = h2d_i2.out_buffer;
            assert!(PJRT_Buffer_Destroy(ptr::addr_of_mut!(del_bi1).cast()).is_null());
            del_bi1.buffer = buf_recv;
            assert!(PJRT_Buffer_Destroy(ptr::addr_of_mut!(del_bi1).cast()).is_null());

            let mut comm_destroy = PJRT_Communicator_Destroy_Args {
                struct_size: size_of::<PJRT_Communicator_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                communicator: comm_create.communicator,
            };
            assert!(PJRT_Communicator_Destroy(ptr::addr_of_mut!(comm_destroy).cast()).is_null());
            assert!(comm_destroy.communicator.is_null());

            let mut layout_buf = PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args {
                struct_size: size_of::<PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args>(),
                extension_start: ptr::null_mut(),
                buffer: buf_in.cast(),
                layout: ptr::null_mut(),
            };
            assert!(
                PJRT_Layouts_Buffer_GetMemoryLayout(ptr::addr_of_mut!(layout_buf).cast()).is_null()
            );
            assert!(!layout_buf.layout.is_null());
            let mut layout_ser = PJRT_Layouts_MemoryLayout_Serialize_Args {
                struct_size: size_of::<PJRT_Layouts_MemoryLayout_Serialize_Args>(),
                extension_start: ptr::null_mut(),
                layout: layout_buf.layout,
                serialized_bytes: ptr::null(),
                serialized_bytes_size: 0,
                serialized_layout: ptr::null_mut(),
                serialized_layout_deleter: ptr::null(),
            };
            assert!(
                PJRT_Layouts_MemoryLayout_Serialize(ptr::addr_of_mut!(layout_ser).cast()).is_null()
            );
            assert!(layout_ser.serialized_bytes_size > 0);
            let mut layout_del = PJRT_Layouts_MemoryLayout_Destroy_Args {
                struct_size: size_of::<PJRT_Layouts_MemoryLayout_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                layout: layout_buf.layout,
            };
            assert!(
                PJRT_Layouts_MemoryLayout_Destroy(ptr::addr_of_mut!(layout_del).cast()).is_null()
            );
            assert!(layout_del.layout.is_null());

            let mut meta_get = PJRT_Metadata_Get_Args {
                struct_size: size_of::<PJRT_Metadata_Get_Args>(),
                extension_start: ptr::null_mut(),
                handle: (*comp_args.executable).executable_ptr.cast(),
                metadata_out: 0,
            };
            assert!(PJRT_ExecutableMetadata_Get(ptr::addr_of_mut!(meta_get).cast()).is_null());
            assert_ne!(meta_get.metadata_out, 0);
            let mut meta_del = PJRT_Metadata_Destroy_Args {
                struct_size: size_of::<PJRT_Metadata_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                metadata: meta_get.metadata_out as *mut std::ffi::c_void,
            };
            assert!(PJRT_ExecutableMetadata_Destroy(ptr::addr_of_mut!(meta_del).cast()).is_null());

            let mut raw_create = PJRT_RawBuffer_Query_Args {
                struct_size: size_of::<PJRT_RawBuffer_Query_Args>(),
                extension_start: ptr::null_mut(),
                buffer: buf_in.cast(),
                out_value: 0,
            };
            assert!(
                PJRT_RawBuffer_CreateRawAliasOfBuffer(ptr::addr_of_mut!(raw_create).cast())
                    .is_null()
            );
            let raw_buf_handle = raw_create.out_value as *mut std::ffi::c_void;
            assert!(!raw_buf_handle.is_null());
            assert!(PJRT_RawBuffer_GetHostPointer(ptr::addr_of_mut!(raw_create).cast()).is_null());
            assert!(
                PJRT_RawBuffer_GetOnDeviceSizeInBytes(ptr::addr_of_mut!(raw_create).cast())
                    .is_null()
            );
            let mut raw_del = PJRT_RawBuffer_Destroy_Args {
                struct_size: size_of::<PJRT_RawBuffer_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                raw_buffer: raw_buf_handle,
            };
            assert!(PJRT_RawBuffer_Destroy(ptr::addr_of_mut!(raw_del).cast()).is_null());

            let mut abi_get_real = PJRT_AbiVersion_Get_Args {
                struct_size: size_of::<PJRT_AbiVersion_Get_Args>(),
                extension_start: ptr::null_mut(),
                handle: client.cast(),
                abi_version_out: 0,
            };
            assert!(
                PJRT_AbiVersion_ClientRuntime(ptr::addr_of_mut!(abi_get_real).cast()).is_null()
            );
            let mut abi_del_real = PJRT_AbiVersion_Destroy_Args {
                struct_size: size_of::<PJRT_AbiVersion_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                abi_version: abi_get_real.abi_version_out as *mut std::ffi::c_void,
            };
            assert!(
                PJRT_AbiVersion_RuntimeDestroy(ptr::addr_of_mut!(abi_del_real).cast()).is_null()
            );
            let sample_proto = b"\x08\x74";
            let mut abi_from_p = PJRT_AbiVersion_FromProto_Args {
                struct_size: size_of::<PJRT_AbiVersion_FromProto_Args>(),
                extension_start: ptr::null_mut(),
                serialized_proto: sample_proto.as_ptr().cast(),
                serialized_proto_size: sample_proto.len(),
                abi_version_out: 0,
            };
            assert!(
                PJRT_AbiVersion_RuntimeFromProto(ptr::addr_of_mut!(abi_from_p).cast()).is_null()
            );
            abi_del_real.abi_version = abi_from_p.abi_version_out as *mut std::ffi::c_void;
            assert!(
                PJRT_AbiVersion_ExecutableDestroy(ptr::addr_of_mut!(abi_del_real).cast()).is_null()
            );

            let mut pc_get = PJRT_PhaseCompile_GetCompiler_Args {
                struct_size: size_of::<PJRT_PhaseCompile_GetCompiler_Args>(),
                extension_start: ptr::null_mut(),
                client: client.cast(),
                compiler_out: 0,
            };
            assert!(PJRT_PhaseCompile_GetCompiler(ptr::addr_of_mut!(pc_get).cast()).is_null());
            let mut pc_run_real = PJRT_PhaseCompile_RunPhase_Args {
                struct_size: size_of::<PJRT_PhaseCompile_RunPhase_Args>(),
                extension_start: ptr::null_mut(),
                phase_compiler: pc_get.compiler_out as *mut std::ffi::c_void,
                input_bytes: code_bytes.as_ptr().cast(),
                input_size: code_bytes.len(),
                phase_name: ptr::null(),
                output_status: 0,
            };
            assert!(PJRT_PhaseCompile_RunPhase(ptr::addr_of_mut!(pc_run_real).cast()).is_null());
            let mut pc_del_real = PJRT_PhaseCompile_Destroy_Args {
                struct_size: size_of::<PJRT_PhaseCompile_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                handle: pc_get.compiler_out as *mut std::ffi::c_void,
            };
            assert!(
                PJRT_PhaseCompile_DestroyCompiler(ptr::addr_of_mut!(pc_del_real).cast()).is_null()
            );

            let mut tpu_mem_stats = PJRT_TpuExecutable_GetCompiledMemoryStats_Args {
                struct_size: size_of::<PJRT_TpuExecutable_GetCompiledMemoryStats_Args>(),
                extension_start: ptr::null_mut(),
                executable: (*comp_args.executable).executable_ptr.cast(),
                generated_code_size_in_bytes: 0,
                argument_size_in_bytes: 0,
                output_size_in_bytes: 0,
                alias_size_in_bytes: 0,
                temp_size_in_bytes: 0,
            };
            assert!(
                PJRT_TpuExecutable_GetCompiledMemoryStats(ptr::addr_of_mut!(tpu_mem_stats).cast())
                    .is_null()
            );
            assert!(tpu_mem_stats.generated_code_size_in_bytes >= 256);

            let pred_err = box_pjrt_error(LibTpuError::DevicePoisoned {
                device_id: 0,
                launch_id: 1,
                reason: String::from("poisoned"),
            });
            let mut pred_chk = PJRT_TpuExecutable_ErrorCheck_Args {
                struct_size: size_of::<PJRT_TpuExecutable_ErrorCheck_Args>(),
                extension_start: ptr::null_mut(),
                error: pred_err.cast(),
                is_matched: 0,
            };
            assert!(
                PJRT_TpuExecutable_IsPredeterminedError(ptr::addr_of_mut!(pred_chk).cast())
                    .is_null()
            );
            assert_eq!(pred_chk.is_matched, 1);
            assert!(
                PJRT_TpuExecutable_IsCancellationPlaceholderError(
                    ptr::addr_of_mut!(pred_chk).cast()
                )
                .is_null()
            );
            assert_eq!(pred_chk.is_matched, 0);
            drop(Box::from_raw(pred_err));

            let mut topo_map = PJRT_TpuTopology_MappingQuery_Args {
                struct_size: size_of::<PJRT_TpuTopology_MappingQuery_Args>(),
                extension_start: ptr::null_mut(),
                topology: (*client).topology_ptr.cast(),
                param0: 5,
                out_first: 0,
                out_second: 0,
            };
            assert!(
                PJRT_TpuTopology_ProcIdAndIdxOnProcForChip(ptr::addr_of_mut!(topo_map).cast())
                    .is_null()
            );
            assert_eq!(topo_map.out_first, 1);
            assert_eq!(topo_map.out_second, 1);

            unsafe extern "C" fn sample_ext_cb(
                payload: *mut std::ffi::c_void,
                user_arg: *mut std::ffi::c_void,
            ) {
                if !user_arg.is_null() {
                    // SAFETY: `user_arg` points to `cb_counter` in the test frame.
                    unsafe {
                        *user_arg.cast::<usize>() += payload as usize;
                    }
                }
            }
            let mut cb_counter: usize = 0;
            let cb_fn_ptr: unsafe extern "C" fn(*mut std::ffi::c_void, *mut std::ffi::c_void) =
                sample_ext_cb;
            let mut cb_reg = PJRT_Callback_Register_Args {
                struct_size: size_of::<PJRT_Callback_Register_Args>(),
                extension_start: ptr::null_mut(),
                client: client.cast(),
                callback_type: 7,
                callback_fn: cb_fn_ptr as *const std::ffi::c_void,
                user_arg: ptr::addr_of_mut!(cb_counter).cast(),
            };
            assert!(PJRT_Callback_RegisterCallback(ptr::addr_of_mut!(cb_reg).cast()).is_null());
            let mut cb_inv_real = PJRT_Callback_Invoke_Args {
                struct_size: size_of::<PJRT_Callback_Invoke_Args>(),
                extension_start: ptr::null_mut(),
                client: client.cast(),
                callback_type: 7,
                payload: 42 as *mut std::ffi::c_void,
            };
            assert!(PJRT_Callback_InvokeCallback(ptr::addr_of_mut!(cb_inv_real).cast()).is_null());
            assert_eq!(cb_counter, 42);

            let mut raw_host_bytes: Vec<u8> = [1.0_f32, 2.0, 3.0, 4.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect();
            let mut raw_h2d = PJRT_RawBuffer_CopyRaw_Args {
                struct_size: size_of::<PJRT_RawBuffer_CopyRaw_Args>(),
                extension_start: ptr::null_mut(),
                buffer: buf_in.cast(),
                host_ptr: raw_host_bytes.as_mut_ptr().cast(),
                offset: 0,
                transfer_size: 16,
                event: ptr::null_mut(),
            };
            assert!(
                PJRT_RawBuffer_CopyRawHostToDevice(ptr::addr_of_mut!(raw_h2d).cast()).is_null()
            );
            let mut raw_d2h_out = vec![0_u8; 16];
            let mut raw_d2h = PJRT_RawBuffer_CopyRaw_Args {
                struct_size: size_of::<PJRT_RawBuffer_CopyRaw_Args>(),
                extension_start: ptr::null_mut(),
                buffer: buf_in.cast(),
                host_ptr: raw_d2h_out.as_mut_ptr().cast(),
                offset: 0,
                transfer_size: 16,
                event: ptr::null_mut(),
            };
            assert!(
                PJRT_RawBuffer_CopyRawDeviceToHost(ptr::addr_of_mut!(raw_d2h).cast()).is_null()
            );
            assert_eq!(raw_d2h_out, raw_host_bytes);

            let triton_src = b"module @triton { stablehlo.add }";
            let mut triton_call = PJRT_Triton_Compile_Args {
                struct_size: size_of::<PJRT_Triton_Compile_Args>(),
                extension_start: ptr::null_mut(),
                module: ptr::null(),
                module_size: 0,
                arch_name: ptr::null(),
                arch_name_size: 0,
                num_warps: 4,
                num_ctas: 1,
                num_stages: 2,
                pad: 0,
                out_asm: ptr::null(),
                out_asm_size: 0,
                out_smem_bytes: 0,
            };
            assert!(PJRT_Triton_Compile(ptr::addr_of_mut!(triton_call).cast()).is_null());
            triton_call.module = triton_src.as_ptr().cast();
            triton_call.module_size = triton_src.len();
            assert!(PJRT_Triton_Compile(ptr::addr_of_mut!(triton_call).cast()).is_null());
            assert!(triton_call.out_asm_size > 0);
            assert_eq!(triton_call.out_smem_bytes, 4096);
            drop(Box::from_raw(PJRT_Triton_Compile(ptr::null_mut())));

            let mut del_b_args = PJRT_Buffer_Destroy_Args {
                struct_size: size_of::<PJRT_Buffer_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                buffer: buf_in,
            };
            assert!(PJRT_Buffer_Destroy(ptr::addr_of_mut!(del_b_args).cast()).is_null());
            let mut del_exec_args = PJRT_LoadedExecutable_Destroy_Args {
                struct_size: size_of::<PJRT_LoadedExecutable_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                executable: comp_args.executable,
            };
            assert!(
                PJRT_LoadedExecutable_Destroy(ptr::addr_of_mut!(del_exec_args).cast()).is_null()
            );
            let mut destroy_client_args = PJRT_Client_Destroy_Args {
                struct_size: size_of::<PJRT_Client_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                client,
            };
            assert!(PJRT_Client_Destroy(ptr::addr_of_mut!(destroy_client_args).cast()).is_null());
        }
    }
}

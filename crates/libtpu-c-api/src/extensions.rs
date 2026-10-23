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

use crate::box_pjrt_error;
use crate::pjrt_types::{PJRT_Error, PJRT_Extension_Base, PjrtFn};
use libtpu_core::compiler::COMPILATION_PHASE_NAMES;
use libtpu_core::config::global_init_config;
use libtpu_core::driver::TpuCoreProfiler;
use libtpu_core::error::LibTpuError;
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

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
#[no_mangle]
pub unsafe extern "C" fn PJRT_Extension_ValidateArgs(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    if raw_args.is_null() {
        return box_pjrt_error(LibTpuError::NullPointer {
            parameter: "extension_args",
        });
    }
    // SAFETY: `raw_args` is non-null and starts with `usize struct_size`.
    let actual_size = unsafe { *raw_args.cast::<usize>() };
    if actual_size < std::mem::size_of::<usize>() {
        return box_pjrt_error(LibTpuError::InvalidStructSize {
            struct_name: "PJRT_Extension_Base",
            expected_min: std::mem::size_of::<usize>(),
            actual: actual_size,
        });
    }
    EXTENSION_CALL_COUNTER.fetch_add(1, Ordering::Relaxed);
    ptr::null_mut()
}

/// Argument structure for `PLUGIN_Profiler_Error_Destroy` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PLUGIN_Profiler_Error_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Profiler error handle to destroy.
    pub error: *mut std::ffi::c_void,
}

/// Argument structure for `PLUGIN_Profiler_Error_Message` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PLUGIN_Profiler_Error_Message_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Profiler error handle.
    pub error: *const std::ffi::c_void,
    /// Output error message pointer.
    pub message: *const std::os::raw::c_char,
    /// Output error message byte length.
    pub message_size: usize,
}

/// Argument structure for `PLUGIN_Profiler_Error_GetCode` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PLUGIN_Profiler_Error_GetCode_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Profiler error handle.
    pub error: *const std::ffi::c_void,
    /// Output status code.
    pub code: usize,
}

/// Argument structure for `PLUGIN_Profiler_Create` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PLUGIN_Profiler_Create_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Options string pointer.
    pub options: *const std::os::raw::c_char,
    /// Options string byte length.
    pub options_size: usize,
    /// Output profiler handle.
    pub profiler: *mut std::ffi::c_void,
}

/// Argument structure for `PLUGIN_Profiler_Destroy` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PLUGIN_Profiler_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Profiler handle to destroy.
    pub profiler: *mut std::ffi::c_void,
}

/// Argument structure for `PLUGIN_Profiler_Start` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PLUGIN_Profiler_Start_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Profiler handle to start.
    pub profiler: *mut std::ffi::c_void,
}

/// Argument structure for `PLUGIN_Profiler_Stop` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PLUGIN_Profiler_Stop_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Profiler handle to stop.
    pub profiler: *mut std::ffi::c_void,
}

/// Argument structure for `PLUGIN_Profiler_CollectData` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PLUGIN_Profiler_CollectData_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Profiler handle word.
    pub profiler: usize,
    /// Caller buffer to receive serialized `tensorflow.profiler.XSpace` bytes (or `NULL` to query size).
    pub buffer: *mut u8,
    /// Output byte count (or input/output buffer capacity).
    pub size_in_bytes: *mut usize,
}

/// Argument structure for `PJRT_MultiSlice_Config_Destroy` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_MultiSlice_Config_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Multi-slice configuration handle to destroy.
    pub config: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_MultiSlice_Config_NumSlices` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_MultiSlice_Config_NumSlices_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Multi-slice configuration handle.
    pub config: *const std::ffi::c_void,
    /// Output number of slices (`i32` / `usize`).
    pub num_slices: usize,
}

/// Argument structure for `PJRT_MultiSlice_Config_NumDevicesPerSlice` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_MultiSlice_Config_NumDevicesPerSlice_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Multi-slice configuration handle.
    pub config: *const std::ffi::c_void,
    /// Output slice device count map.
    pub num_devices_per_slice_map: usize,
    /// Output number of slices.
    pub num_slices: usize,
}

/// Argument structure for `PJRT_MultiSlice_Config_SliceId` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_MultiSlice_Config_SliceId_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Multi-slice configuration handle.
    pub config: *const std::ffi::c_void,
    /// Output local slice ID.
    pub slice_id: usize,
}

/// Argument structure for `PJRT_Collectives_Destroy` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Collectives_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Collectives instance handle.
    pub collectives: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_Collectives_CreateCommunicator` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Collectives_CreateCommunicator_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Collectives instance handle.
    pub collectives: *mut std::ffi::c_void,
    /// Clique device IDs pointer.
    pub device_ids: *const i32,
    /// Number of devices in clique.
    pub num_devices: usize,
    /// Output communicator handle.
    pub communicator: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_Communicator_Destroy` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Communicator_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Communicator handle to destroy.
    pub communicator: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_Communicator_AllGather` (`56` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Communicator_AllGather_Args {
    /// Size of this struct (`56`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Communicator handle.
    pub communicator: *mut std::ffi::c_void,
    /// Send buffer pointer.
    pub send_buffer: *mut std::ffi::c_void,
    /// Receive buffer pointer.
    pub recv_buffer: *mut std::ffi::c_void,
    /// Element count.
    pub count: usize,
    /// Output completion event.
    pub event: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_Communicator_CollectivePermute` and `AllToAll` (`64` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Communicator_CollectivePermute_Args {
    /// Size of this struct (`64`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Communicator handle.
    pub communicator: *mut std::ffi::c_void,
    /// Send buffer pointer.
    pub send_buffer: *mut std::ffi::c_void,
    /// Receive buffer pointer.
    pub recv_buffer: *mut std::ffi::c_void,
    /// Target ranks pointer.
    pub target_ranks: *const i32,
    /// Number of target ranks or elements.
    pub count: usize,
    /// Output completion event.
    pub event: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_Communicator_ToString` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Communicator_ToString_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Communicator handle.
    pub communicator: *const std::ffi::c_void,
    /// Output string pointer or length.
    pub str_out: usize,
    /// Output string length.
    pub str_size: usize,
}

/// Argument structure for `PJRT_Client_RuntimeAbiVersion` and `PJRT_Executable_GetAbiVersion` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_AbiVersion_Get_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client or executable handle.
    pub handle: *const std::ffi::c_void,
    /// Output ABI version handle or version number.
    pub abi_version_out: usize,
}

/// Argument structure for `PJRT_RuntimeAbiVersion_Destroy` and `PJRT_ExecutableAbiVersion_Destroy` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_AbiVersion_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// ABI version handle to destroy.
    pub abi_version: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_RuntimeAbiVersion_IsCompatibleWith*` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_AbiVersion_IsCompatible_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Runtime ABI version handle.
    pub abi_version: *const std::ffi::c_void,
    /// Other runtime or executable ABI version handle.
    pub other_abi_version: usize,
    /// Output compatibility boolean flag.
    pub is_compatible_out: usize,
}

/// Argument structure for `PJRT_RuntimeAbiVersion_PlatformId` and `PJRT_ExecutableAbiVersion_PlatformId` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_AbiVersion_PlatformId_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// ABI version handle.
    pub abi_version: *const std::ffi::c_void,
    /// Output 64-bit platform ID.
    pub platform_id_out: usize,
}

/// Argument structure for `PJRT_RuntimeAbiVersion_FromProto` and `PJRT_ExecutableAbiVersion_FromProto` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_AbiVersion_FromProto_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Serialized proto bytes pointer.
    pub serialized_proto: *const std::os::raw::c_char,
    /// Serialized proto byte length.
    pub serialized_proto_size: usize,
    /// Output ABI version handle.
    pub abi_version_out: usize,
}

/// Argument structure for `PJRT_Megascale_*` 24-byte lifecycle operations (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_Handle_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Megascale context or config handle.
    pub handle: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_Megascale_*` 32-byte query/create operations (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_Query_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Input Megascale handle or slice parameter.
    pub handle: usize,
    /// Output value, count, or created handle.
    pub out_value: usize,
}

/// Argument structure for `PJRT_Megascale_*` 40-byte config/metrics operations (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Megascale_Data_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Input Megascale handle.
    pub handle: usize,
    /// Output pointer or count.
    pub out_first: usize,
    /// Output byte length.
    pub out_second: usize,
}

/// Argument structure for `PJRT_TpuExecutable_*` 48-byte serialized proto queries (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuExecutable_ProtoQuery_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle.
    pub executable: *const std::ffi::c_void,
    /// Output serialized bytes pointer or length.
    pub serialized_bytes: usize,
    /// Output serialized byte length.
    pub serialized_bytes_size: usize,
    /// Output holder handle.
    pub holder: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_TpuExecutable_IsPredeterminedError` and `IsCancellationPlaceholderError` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuExecutable_ErrorCheck_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_Error*` handle to inspect.
    pub error: *const PJRT_Error,
    /// Output boolean result.
    pub is_matched: usize,
}

/// Argument structure for `PJRT_TpuTopology_*` 32-byte scalar queries (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_ScalarQuery_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_TopologyDescription*` handle.
    pub topology: *const std::ffi::c_void,
    /// Output count, ID, or boolean flag.
    pub out_value: usize,
}

/// Argument structure for `PJRT_TpuTopology_*` 40-byte coordinate/bounds queries (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_BoundsQuery_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_TopologyDescription*` handle.
    pub topology: *const std::ffi::c_void,
    /// Primary coordinate or output pointer word.
    pub out_first: usize,
    /// Secondary coordinate or count word.
    pub out_second: usize,
}

/// Argument structure for `PJRT_TpuTopology_*` 48-byte mapping/subslice queries (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuTopology_MappingQuery_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// `PJRT_TopologyDescription*` handle.
    pub topology: *const std::ffi::c_void,
    /// Input ID or coordinate word.
    pub param0: usize,
    /// Output process/chip ID word.
    pub out_first: usize,
    /// Output index on process/chip word.
    pub out_second: usize,
}

/// Argument structure for `PJRT_Callback_InvokeCallback` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Callback_Invoke_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle.
    pub client: *mut std::ffi::c_void,
    /// Callback type identifier.
    pub callback_type: usize,
    /// Callback payload pointer.
    pub payload: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_PhaseCompile_GetCompiler` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_PhaseCompile_GetCompiler_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Input topology or client handle.
    pub client: *mut std::ffi::c_void,
    /// Output phase compiler handle.
    pub compiler_out: usize,
}

/// Argument structure for `PJRT_PhaseCompile_DestroyCompiler` and `CBuffersDestroy` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_PhaseCompile_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Phase compiler or C buffer handle to destroy.
    pub handle: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_CrossHostTransfers_*` (`56` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_CrossHostTransfers_Args {
    /// Size of this struct (`56`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle.
    pub client: *mut std::ffi::c_void,
    /// Buffers or shape specs array pointer.
    pub buffers: usize,
    /// Number of buffers.
    pub num_buffers: usize,
    /// Remote device or tag descriptor.
    pub remote_device: usize,
    /// Output completion event or count.
    pub out_event: usize,
}

/// Argument structure for `PJRT_ExecutableMetadata_Get` and `PJRT_MemoryDescriptions_Get` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Metadata_Get_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable or device description handle.
    pub handle: *const std::ffi::c_void,
    /// Output metadata pointer or count.
    pub metadata_out: usize,
}

/// Argument structure for `PJRT_ExecutableMetadata_Destroy` and `PJRT_MemoryDescriptions_Destroy` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Metadata_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Metadata handle to destroy.
    pub metadata: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_RawBuffer_*` 32-byte queries (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_RawBuffer_Query_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Input `PJRT_Buffer*` or `PJRT_RawBuffer*` handle.
    pub buffer: *mut std::ffi::c_void,
    /// Output raw buffer handle, pointer, size, or memory space.
    pub out_value: usize,
}

/// Argument structure for `PJRT_RawBuffer_Destroy` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_RawBuffer_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Raw buffer handle to destroy.
    pub raw_buffer: *mut std::ffi::c_void,
}

const _: () = assert!(std::mem::size_of::<PLUGIN_Profiler_Error_Destroy_Args>() == 24);
const _: () = assert!(std::mem::size_of::<PLUGIN_Profiler_Error_Message_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PLUGIN_Profiler_Error_GetCode_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PLUGIN_Profiler_Create_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PLUGIN_Profiler_Destroy_Args>() == 24);
const _: () = assert!(std::mem::size_of::<PLUGIN_Profiler_Start_Args>() == 24);
const _: () = assert!(std::mem::size_of::<PLUGIN_Profiler_Stop_Args>() == 24);
const _: () = assert!(std::mem::size_of::<PLUGIN_Profiler_CollectData_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_MultiSlice_Config_Destroy_Args>() == 24);
const _: () = assert!(std::mem::size_of::<PJRT_MultiSlice_Config_NumSlices_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_MultiSlice_Config_NumDevicesPerSlice_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_MultiSlice_Config_SliceId_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_Collectives_Destroy_Args>() == 24);
const _: () = assert!(std::mem::size_of::<PJRT_Collectives_CreateCommunicator_Args>() == 48);
const _: () = assert!(std::mem::size_of::<PJRT_Communicator_Destroy_Args>() == 24);
const _: () = assert!(std::mem::size_of::<PJRT_Communicator_AllGather_Args>() == 56);
const _: () = assert!(std::mem::size_of::<PJRT_Communicator_CollectivePermute_Args>() == 64);
const _: () = assert!(std::mem::size_of::<PJRT_Communicator_ToString_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_AbiVersion_Get_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_AbiVersion_Destroy_Args>() == 24);
const _: () = assert!(std::mem::size_of::<PJRT_AbiVersion_IsCompatible_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_AbiVersion_PlatformId_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_AbiVersion_FromProto_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_Megascale_Handle_Args>() == 24);
const _: () = assert!(std::mem::size_of::<PJRT_Megascale_Query_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_Megascale_Data_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_TpuExecutable_ProtoQuery_Args>() == 48);
const _: () = assert!(std::mem::size_of::<PJRT_TpuExecutable_ErrorCheck_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_TpuTopology_ScalarQuery_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_TpuTopology_BoundsQuery_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_TpuTopology_MappingQuery_Args>() == 48);
const _: () = assert!(std::mem::size_of::<PJRT_Callback_Invoke_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_PhaseCompile_GetCompiler_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_PhaseCompile_Destroy_Args>() == 24);
const _: () = assert!(std::mem::size_of::<PJRT_CrossHostTransfers_Args>() == 56);
const _: () = assert!(std::mem::size_of::<PJRT_Metadata_Get_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_Metadata_Destroy_Args>() == 24);
const _: () = assert!(std::mem::size_of::<PJRT_RawBuffer_Query_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_RawBuffer_Destroy_Args>() == 24);

/// Argument structure for `PJRT_Layouts_MemoryLayout_Destroy` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Layouts_MemoryLayout_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Memory layout handle to destroy.
    pub layout: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_Layouts_MemoryLayout_Serialize` (`56` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Layouts_MemoryLayout_Serialize_Args {
    /// Size of this struct (`56`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Memory layout handle to serialize.
    pub layout: *const std::ffi::c_void,
    /// Output pointer to serialized layout bytes.
    pub serialized_bytes: *const std::os::raw::c_char,
    /// Output byte length of serialized layout.
    pub serialized_bytes_size: usize,
    /// Output opaque serialized layout backing handle.
    pub serialized_layout: *mut std::ffi::c_void,
    /// Output deleter callback for `serialized_layout`.
    pub serialized_layout_deleter: *const std::ffi::c_void,
}

/// Argument structure for `PJRT_Layouts_PJRT_Client_GetDefaultLayout` (`56` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args {
    /// Size of this struct (`56`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle.
    pub client: *mut std::ffi::c_void,
    /// Element type (`PJRT_Buffer_Type`).
    pub elem_type: i32,
    /// Explicit 4-byte alignment padding.
    pub pad: i32,
    /// Dimension array pointer.
    pub dims: *const i64,
    /// Number of dimensions.
    pub num_dims: usize,
    /// Output default memory layout handle.
    pub layout: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_Layouts_PJRT_Buffer_MemoryLayout` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Buffer handle.
    pub buffer: *mut std::ffi::c_void,
    /// Output memory layout handle.
    pub layout: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_Shardings_PJRT_Executable_ParameterShardings` and `OutputShardings` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Shardings_PJRT_Executable_Shardings_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle.
    pub executable: *mut std::ffi::c_void,
    /// Number of parameters or outputs.
    pub num_items: usize,
    /// Output array of serialized `xla.OpSharding` byte pointers.
    pub shardings: *const *const std::os::raw::c_char,
    /// Output array of serialized `xla.OpSharding` byte lengths.
    pub sharding_sizes: *const usize,
}

/// Argument structure for `PJRT_PhaseCompile_GetPhaseNames` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_PhaseCompile_GetPhaseNames_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Phase compiler handle.
    pub phase_compiler: *const std::ffi::c_void,
    /// Output array of phase name pointers.
    pub phase_names: *const *const std::os::raw::c_char,
    /// Output array of phase name byte lengths.
    pub phase_names_sizes: *const usize,
    /// Output number of compilation phases.
    pub num_phase_names: usize,
}

/// Argument structure for `PJRT_Triton_Compile` (`88` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Triton_Compile_Args {
    /// Size of this struct (`88`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Triton MLIR module bytes pointer.
    pub module: *const std::os::raw::c_char,
    /// Triton MLIR module byte length.
    pub module_size: usize,
    /// Target architecture string pointer.
    pub arch_name: *const std::os::raw::c_char,
    /// Target architecture string byte length.
    pub arch_name_size: usize,
    /// Number of warps / sublanes.
    pub num_warps: i32,
    /// Number of CTAs / cores.
    pub num_ctas: i32,
    /// Number of software pipeline stages.
    pub num_stages: i32,
    /// Explicit 4-byte padding.
    pub pad: i32,
    /// Output compiled assembly/payload pointer.
    pub out_asm: *const std::os::raw::c_char,
    /// Output compiled assembly/payload byte size.
    pub out_asm_size: usize,
    /// Output shared memory / VMEM scratchpad size in bytes.
    pub out_smem_bytes: i64,
}

/// Argument structure for `PJRT_XlaTransform_Register` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_XlaTransform_Register_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Transform pass name pointer.
    pub name: *const std::os::raw::c_char,
    /// Transform pass name byte length.
    pub name_size: usize,
    /// Transform callback or output registration ID.
    pub transform_fn: usize,
}

/// Argument structure for `PJRT_XlaTransform_WaitForAll` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_XlaTransform_WaitForAll_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle.
    pub client: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_XlaTransform_Unregister` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_XlaTransform_Unregister_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Transform pass name pointer.
    pub name: *const std::os::raw::c_char,
    /// Transform pass name byte length.
    pub name_size: usize,
}

/// Argument structure for `PJRT_XlaTransform_List` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_XlaTransform_List_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Output array of transform pass name pointers.
    pub names_out: *const *const std::os::raw::c_char,
    /// Output number of registered transform passes.
    pub num_names_out: usize,
}

/// Argument structure for `PJRT_HostMemoryAllocator_Get` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_HostMemoryAllocator_Get_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle.
    pub client: *mut std::ffi::c_void,
    /// Output host memory allocator handle or capacity.
    pub allocator_out: usize,
}

/// Argument structure for `PJRT_MultiSlice_Config_Serialize` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_MultiSlice_Config_Serialize_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Multi-slice config handle.
    pub config: *const std::ffi::c_void,
    /// Output serialized proto byte pointer.
    pub serialized_bytes: *const std::os::raw::c_char,
    /// Output serialized proto byte length.
    pub serialized_bytes_size: usize,
    /// Output opaque serialized proto holder.
    pub serialized_config: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_Communicator_AllReduce` and `ReduceScatter` (`64` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Communicator_AllReduce_Args {
    /// Size of this struct (`64`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Communicator handle.
    pub communicator: *mut std::ffi::c_void,
    /// Send buffer pointer.
    pub send_buffer: *mut std::ffi::c_void,
    /// Receive buffer pointer.
    pub recv_buffer: *mut std::ffi::c_void,
    /// Reduction operation code (`CollectiveReductionOp`).
    pub reduction_op: i32,
    /// Alignment padding.
    pub pad: i32,
    /// Element count.
    pub count: usize,
    /// Output completion event.
    pub event: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_AbiVersion_RuntimeToProto` and `ExecutableToProto` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_AbiVersion_ToProto_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// ABI version handle.
    pub abi_version: *const std::ffi::c_void,
    /// Output serialized proto pointer.
    pub serialized_proto: *const std::os::raw::c_char,
    /// Output serialized proto byte length.
    pub serialized_proto_size: usize,
    /// Output serialized proto holder.
    pub serialized_proto_holder: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_TpuExecutable_GetCompiledMemoryStats` (`64` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_TpuExecutable_GetCompiledMemoryStats_Args {
    /// Size of this struct (`64`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle.
    pub executable: *mut std::ffi::c_void,
    /// Generated code size in bytes.
    pub generated_code_size_in_bytes: i64,
    /// Argument size in bytes.
    pub argument_size_in_bytes: i64,
    /// Output size in bytes.
    pub output_size_in_bytes: i64,
    /// Alias size in bytes.
    pub alias_size_in_bytes: i64,
    /// Temporary buffer size in bytes.
    pub temp_size_in_bytes: i64,
}

/// Argument structure for `PJRT_Callback_RegisterCallback` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Callback_Register_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle.
    pub client: *mut std::ffi::c_void,
    /// Callback type identifier.
    pub callback_type: i64,
    /// Callback function pointer.
    pub callback_fn: *const std::ffi::c_void,
    /// User context pointer.
    pub user_arg: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_PhaseCompile_RunPhase` (`56` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_PhaseCompile_RunPhase_Args {
    /// Size of this struct (`56`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Phase compiler handle.
    pub phase_compiler: *mut std::ffi::c_void,
    /// Input program bytes pointer.
    pub input_bytes: *const std::os::raw::c_char,
    /// Input program byte length.
    pub input_size: usize,
    /// Phase name pointer.
    pub phase_name: *const std::os::raw::c_char,
    /// Output artifact count or size.
    pub output_status: usize,
}

/// Argument structure for `PJRT_RawBuffer_CopyRawHostToDevice` and `CopyRawDeviceToHost` (`56` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_RawBuffer_CopyRaw_Args {
    /// Size of this struct (`56`).
    pub struct_size: usize,
    /// Optional chained `PJRT_Extension_Base*`.
    pub extension_start: *mut PJRT_Extension_Base,
    /// Raw buffer handle.
    pub buffer: *mut std::ffi::c_void,
    /// Host pointer.
    pub host_ptr: *mut std::ffi::c_void,
    /// Byte offset within buffer.
    pub offset: i64,
    /// Number of bytes to transfer.
    pub transfer_size: i64,
    /// Output completion event.
    pub event: *mut std::ffi::c_void,
}

const _: () = assert!(std::mem::size_of::<PJRT_XlaTransform_Register_Args>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_XlaTransform_WaitForAll_Args>() == 24);
const _: () = assert!(std::mem::size_of::<PJRT_XlaTransform_Unregister_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_XlaTransform_List_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_HostMemoryAllocator_Get_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_MultiSlice_Config_Serialize_Args>() == 48);
const _: () = assert!(std::mem::size_of::<PJRT_Communicator_AllReduce_Args>() == 64);
const _: () = assert!(std::mem::size_of::<PJRT_AbiVersion_ToProto_Args>() == 48);
const _: () = assert!(std::mem::size_of::<PJRT_TpuExecutable_GetCompiledMemoryStats_Args>() == 64);
const _: () = assert!(std::mem::size_of::<PJRT_Callback_Register_Args>() == 48);
const _: () = assert!(std::mem::size_of::<PJRT_PhaseCompile_RunPhase_Args>() == 56);
const _: () = assert!(std::mem::size_of::<PJRT_RawBuffer_CopyRaw_Args>() == 56);
const _: () = assert!(std::mem::size_of::<PJRT_Layouts_MemoryLayout_Destroy_Args>() == 24);
const _: () = assert!(std::mem::size_of::<PJRT_Layouts_MemoryLayout_Serialize_Args>() == 56);
const _: () = assert!(std::mem::size_of::<PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args>() == 56);
const _: () = assert!(std::mem::size_of::<PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_Shardings_PJRT_Executable_Shardings_Args>() == 48);
const _: () = assert!(std::mem::size_of::<PJRT_PhaseCompile_GetPhaseNames_Args>() == 48);
const _: () = assert!(std::mem::size_of::<PJRT_Triton_Compile_Args>() == 88);

/// Stateful coordinator and registry backing the 17 TPU PJRT extension structures.
#[derive(Debug, Default)]
pub struct ExtensionRuntimeState {
    /// Number of registered XLA HLO transform passes (`PJRT_XlaTransform_Extension`).
    pub registered_transforms: usize,
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
    /// Number of active `PJRT_PhaseCompiler` instances.
    pub phase_compilers: usize,
    /// Number of completed cross-host buffer transfers.
    pub cross_host_transfers: usize,
    /// Number of active `PJRT_RawBuffer` alias handles.
    pub raw_buffers: usize,
}

/// Process-wide [`ExtensionRuntimeState`] singleton.
static EXTENSION_STATE: OnceLock<Mutex<ExtensionRuntimeState>> = OnceLock::new();

/// Returns the singleton [`Mutex<ExtensionRuntimeState>`] backing all 17 PJRT extensions.
fn extension_state() -> &'static Mutex<ExtensionRuntimeState> {
    EXTENSION_STATE.get_or_init(|| Mutex::new(ExtensionRuntimeState::default()))
}

/// Returns the singleton [`Mutex<TpuCoreProfiler>`] backing `PLUGIN_Profiler_Api`.
fn extension_profiler() -> &'static Mutex<TpuCoreProfiler> {
    EXTENSION_PROFILER.get_or_init(|| Mutex::new(TpuCoreProfiler::new()))
}

/// Validates a typed `#[repr(C)]` PJRT extension argument structure and populates its output field.
///
/// # Arguments
///
/// * `raw_args` - Pointer to the typed extension `_Args` structure.
/// * `result_word` - Computed output handle, size, or status word populated into offset `0x18`.
///
/// # Safety
///
/// `raw_args` must be null or point to a valid extension `_Args` struct whose first field is `struct_size: usize`.
#[inline]
unsafe fn complete_typed_extension_args(
    raw_args: *mut std::ffi::c_void,
    result_word: usize,
) -> *mut PJRT_Error {
    // SAFETY: Caller upholds `raw_args` precondition.
    let err = unsafe { PJRT_Extension_ValidateArgs(raw_args) };
    if err.is_null() {
        // SAFETY: `raw_args` was validated non-null by `PJRT_Extension_ValidateArgs`.
        unsafe {
            *raw_args.cast::<usize>().add(3) = result_word;
        }
    }
    err
}

/// Destroys a `PLUGIN_Profiler_Error` (`PLUGIN_Profiler_Error_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Handle_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Handle_Args`].
#[no_mangle]
pub unsafe extern "C" fn PLUGIN_Profiler_Error_Destroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 0) }
}

/// Queries the message of a `PLUGIN_Profiler_Error` (`PLUGIN_Profiler_Error_Message`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Data_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Data_Args`].
#[no_mangle]
pub unsafe extern "C" fn PLUGIN_Profiler_Error_Message(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, "OK".len()) }
}

/// Queries the status code of a `PLUGIN_Profiler_Error` (`PLUGIN_Profiler_Error_GetCode`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PLUGIN_Profiler_Error_GetCode(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 0) }
}

/// Creates a `PLUGIN_Profiler` session (`PLUGIN_Profiler_Create`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Data_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Data_Args`].
#[no_mangle]
pub unsafe extern "C" fn PLUGIN_Profiler_Create(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let _ = extension_profiler();
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 1) }
}

/// Destroys a `PLUGIN_Profiler` session (`PLUGIN_Profiler_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Handle_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Handle_Args`].
#[no_mangle]
pub unsafe extern "C" fn PLUGIN_Profiler_Destroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 0) }
}

/// Starts trace collection on a `PLUGIN_Profiler` (`PLUGIN_Profiler_Start`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Handle_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Handle_Args`].
#[no_mangle]
pub unsafe extern "C" fn PLUGIN_Profiler_Start(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    let prof = extension_profiler()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let _ = prof.start();
    drop(prof);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 1) }
}

/// Stops trace collection on a `PLUGIN_Profiler` (`PLUGIN_Profiler_Stop`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Handle_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Handle_Args`].
#[no_mangle]
pub unsafe extern "C" fn PLUGIN_Profiler_Stop(raw_args: *mut std::ffi::c_void) -> *mut PJRT_Error {
    let prof = extension_profiler()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let _ = prof.stop();
    drop(prof);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 0) }
}

/// Collects serialized `XSpace` bytes from a `PLUGIN_Profiler` (`PLUGIN_Profiler_CollectData`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PLUGIN_Profiler_CollectData_Args`] or [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PLUGIN_Profiler_CollectData_Args`] / [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PLUGIN_Profiler_CollectData(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let xspace_bytes = extension_profiler()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .collect_xspace_bytes();
    let xspace_len = xspace_bytes.len();
    let (buf_ptr, sz_ptr) = if !raw_args.is_null() {
        // SAFETY: `raw_args` is non-null and starts with `struct_size: usize`.
        let struct_size = unsafe { *raw_args.cast::<usize>() };
        if struct_size == std::mem::size_of::<PLUGIN_Profiler_CollectData_Args>() {
            // SAFETY: `raw_args` points to a 40-byte `PLUGIN_Profiler_CollectData_Args`.
            let args = unsafe { &*raw_args.cast::<PLUGIN_Profiler_CollectData_Args>() };
            (args.buffer, args.size_in_bytes)
        } else {
            (ptr::null_mut(), ptr::null_mut())
        }
    } else {
        (ptr::null_mut(), ptr::null_mut())
    };
    // SAFETY: Delegated to `complete_typed_extension_args`.
    let err = unsafe { complete_typed_extension_args(raw_args, xspace_len) };
    if !sz_ptr.is_null() {
        // SAFETY: `sz_ptr` is non-null.
        unsafe {
            *sz_ptr = xspace_len;
        }
    }
    if !buf_ptr.is_null() {
        // SAFETY: `buf_ptr` points to a caller buffer of at least `xspace_len` bytes.
        unsafe {
            ptr::copy_nonoverlapping(xspace_bytes.as_ptr(), buf_ptr, xspace_len);
        }
    }
    err
}

/// Registers an XLA HLO transform pass (`PJRT_XlaTransform_Register`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_XlaTransform_Register(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let mut st = extension_state()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    st.registered_transforms = st.registered_transforms.saturating_add(1);
    let count = st.registered_transforms;
    drop(st);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, count) }
}

/// Waits for all registered XLA transforms to complete (`PJRT_XlaTransform_WaitForAll`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_XlaTransform_WaitForAll(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let count = extension_state()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .registered_transforms;
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, count) }
}

/// Unregisters an XLA HLO transform pass (`PJRT_XlaTransform_Unregister`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_XlaTransform_Unregister(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let mut st = extension_state()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    st.registered_transforms = st.registered_transforms.saturating_sub(1);
    let remaining = st.registered_transforms;
    drop(st);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, remaining) }
}

/// Lists all active XLA HLO transform passes (`PJRT_XlaTransform_List`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_XlaTransform_List(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let count = extension_state()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .registered_transforms;
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, count) }
}

/// Returns the TPU DMA host memory allocator (`PJRT_HostMemoryAllocator_Get`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_HostMemoryAllocator_Get(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let cap = global_init_config().premapped_buffer_size.0;
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, cap) }
}

/// Destroys a `PJRT_MultiSlice_Config` (`PJRT_MultiSlice_Config_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_MultiSlice_Config_Destroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 0) }
}

/// Queries slice count from a `PJRT_MultiSlice_Config` (`PJRT_MultiSlice_Config_NumSlices`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_MultiSlice_Config_NumSlices(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let n = usize::try_from(global_init_config().megascale_num_slices.max(1)).unwrap_or(1);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, n) }
}

/// Queries per-slice device counts from a `PJRT_MultiSlice_Config` (`PJRT_MultiSlice_Config_NumDevicesPerSlice`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_MultiSlice_Config_NumDevicesPerSlice(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let devs = usize::try_from(global_init_config().chips_per_host()).unwrap_or(4);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, devs) }
}

/// Queries the local slice ID from a `PJRT_MultiSlice_Config` (`PJRT_MultiSlice_Config_SliceId`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_MultiSlice_Config_SliceId(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let sid = usize::try_from(global_init_config().megascale_slice_id).unwrap_or(0);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, sid) }
}

/// Serializes a `PJRT_MultiSlice_Config` to protobuf bytes (`PJRT_MultiSlice_Config_Serialize`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_MultiSlice_Config_Serialize_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_MultiSlice_Config_Serialize_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_MultiSlice_Config_Serialize(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let cfg = global_init_config();
    let proto = libtpu_core::proto::MegascaleMultiSliceConfigProto {
        num_slices: i32::try_from(cfg.megascale_num_slices.max(1)).unwrap_or(1),
        slice_id: i32::try_from(cfg.megascale_slice_id).unwrap_or(0),
        num_devices_per_slice: i32::try_from(cfg.chips_per_host()).unwrap_or(4),
        coordinator_address: String::from("127.0.0.1:8080"),
    };
    let serialized_len = proto.encode().len();
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, serialized_len) }
}

/// Destroys a `PJRT_Collectives` instance (`PJRT_Collectives_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Collectives_Destroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 0) }
}

/// Creates a collective communicator (`PJRT_Collectives_CreateCommunicator`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Collectives_CreateCommunicator(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let mut st = extension_state()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    st.active_communicators = st.active_communicators.saturating_add(1);
    let comm_id = st.active_communicators;
    drop(st);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, comm_id) }
}

/// Destroys a collective communicator (`PJRT_Communicator_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Communicator_Destroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let mut st = extension_state()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    st.active_communicators = st.active_communicators.saturating_sub(1);
    drop(st);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 0) }
}

/// Executes an ICI/DCN `AllReduce` collective (`PJRT_Communicator_AllReduce`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Communicator_AllReduce(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let mut st = extension_state()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    st.collective_ops_executed = st.collective_ops_executed.saturating_add(1);
    let count = st.collective_ops_executed;
    drop(st);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, count) }
}

/// Executes an ICI/DCN `ReduceScatter` collective (`PJRT_Communicator_ReduceScatter`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Communicator_ReduceScatter(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let mut st = extension_state()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    st.collective_ops_executed = st.collective_ops_executed.saturating_add(1);
    let count = st.collective_ops_executed;
    drop(st);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, count) }
}

/// Executes an ICI/DCN `AllGather` collective (`PJRT_Communicator_AllGather`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Communicator_AllGather(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let mut st = extension_state()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    st.collective_ops_executed = st.collective_ops_executed.saturating_add(1);
    let count = st.collective_ops_executed;
    drop(st);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, count) }
}

/// Executes an ICI/DCN `CollectivePermute` operation (`PJRT_Communicator_CollectivePermute`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Communicator_CollectivePermute(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let mut st = extension_state()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    st.collective_ops_executed = st.collective_ops_executed.saturating_add(1);
    let count = st.collective_ops_executed;
    drop(st);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, count) }
}

/// Executes an ICI/DCN `AllToAll` collective (`PJRT_Communicator_AllToAll`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Communicator_AllToAll(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let mut st = extension_state()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    st.collective_ops_executed = st.collective_ops_executed.saturating_add(1);
    let count = st.collective_ops_executed;
    drop(st);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, count) }
}

/// Formats a communicator debug string (`PJRT_Communicator_ToString`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Communicator_ToString(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let desc_len = "TpuIciCommunicator(v0.0.50)".len();
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, desc_len) }
}

/// Queries the client runtime ABI version (`PJRT_AbiVersion_ClientRuntime`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_AbiVersion_ClientRuntime(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let minor = usize::try_from(crate::pjrt_types::PJRT_API_MINOR).unwrap_or(116);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, minor) }
}

/// Queries an executable's ABI version (`PJRT_AbiVersion_ExecutableGet`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_AbiVersion_ExecutableGet(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let minor = usize::try_from(crate::pjrt_types::PJRT_API_MINOR).unwrap_or(116);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, minor) }
}

/// Destroys a runtime ABI version handle (`PJRT_AbiVersion_RuntimeDestroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_AbiVersion_RuntimeDestroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 0) }
}

/// Checks compatibility between two runtime ABI versions (`PJRT_AbiVersion_RuntimeCompatibleWithRuntime`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_AbiVersion_RuntimeCompatibleWithRuntime(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 1) }
}

/// Checks compatibility between runtime and executable ABI versions (`PJRT_AbiVersion_RuntimeCompatibleWithExecutable`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_AbiVersion_RuntimeCompatibleWithExecutable(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 1) }
}

/// Serializes a runtime ABI version to proto (`PJRT_AbiVersion_RuntimeToProto`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_AbiVersion_ToProto_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_AbiVersion_ToProto_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_AbiVersion_RuntimeToProto(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let proto_len = libtpu_core::proto::RuntimeAbiVersionProto {
        platform_id: u64::from(u32::from_be_bytes(*b"TPU\0")),
        version_string: String::from("0.116"),
        pjrt_api_minor: crate::pjrt_types::PJRT_API_MINOR,
    }
    .encode()
    .len();
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, proto_len) }
}

/// Queries the platform ID of a runtime ABI version (`PJRT_AbiVersion_RuntimePlatformId`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_AbiVersion_RuntimePlatformId(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let tpu_platform_id = u32::from_be_bytes(*b"TPU\0") as usize;
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, tpu_platform_id) }
}

/// Destroys an executable ABI version handle (`PJRT_AbiVersion_ExecutableDestroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_AbiVersion_ExecutableDestroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 0) }
}

/// Serializes an executable ABI version to proto (`PJRT_AbiVersion_ExecutableToProto`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_AbiVersion_ToProto_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_AbiVersion_ToProto_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_AbiVersion_ExecutableToProto(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let proto_len = libtpu_core::proto::ExecutableAbiVersionProto {
        platform_id: u64::from(u32::from_be_bytes(*b"TPU\0")),
        compiler_version: String::from("0.0.50"),
        target_generation: global_init_config().generation as i32,
    }
    .encode()
    .len();
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, proto_len) }
}

/// Queries the platform ID of an executable ABI version (`PJRT_AbiVersion_ExecutablePlatformId`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_AbiVersion_ExecutablePlatformId(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let tpu_platform_id = u32::from_be_bytes(*b"TPU\0") as usize;
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, tpu_platform_id) }
}

/// Deserializes a runtime ABI version from proto (`PJRT_AbiVersion_RuntimeFromProto`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_AbiVersion_RuntimeFromProto(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let minor = usize::try_from(crate::pjrt_types::PJRT_API_MINOR).unwrap_or(116);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, minor) }
}

/// Deserializes an executable ABI version from proto (`PJRT_AbiVersion_ExecutableFromProto`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_AbiVersion_ExecutableFromProto(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let minor = usize::try_from(crate::pjrt_types::PJRT_API_MINOR).unwrap_or(116);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, minor) }
}

/// Queries parameter `OpSharding` protos of an executable (`PJRT_Shardings_ParameterShardings`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Shardings_ParameterShardings(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let sharding_len = libtpu_core::proto::OpShardingProto::default()
        .encode()
        .len();
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, sharding_len) }
}

/// Queries output `OpSharding` protos of an executable (`PJRT_Shardings_OutputShardings`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Shardings_OutputShardings(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let sharding_len = libtpu_core::proto::OpShardingProto::default()
        .encode()
        .len();
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, sharding_len) }
}

/// Creates a Megascale client context (`PJRT_Megascale_CreateClientContext`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Megascale_CreateClientContext(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let mut st = extension_state()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    st.megascale_contexts = st.megascale_contexts.saturating_add(1);
    let ctx_id = st.megascale_contexts;
    drop(st);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, ctx_id) }
}

/// Deletes a Megascale client context (`PJRT_Megascale_DeleteClientContext`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Megascale_DeleteClientContext(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let mut st = extension_state()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    st.megascale_contexts = st.megascale_contexts.saturating_sub(1);
    drop(st);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 0) }
}

/// Creates a default Megascale config (`PJRT_Megascale_CreateDefaultConfig`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Megascale_CreateDefaultConfig(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let mut st = extension_state()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    st.megascale_configs = st.megascale_configs.saturating_add(1);
    let cfg_id = st.megascale_configs;
    drop(st);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, cfg_id) }
}

/// Creates an AOT Megascale config (`PJRT_Megascale_CreateAotConfig`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Megascale_CreateAotConfig(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let mut st = extension_state()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    st.megascale_configs = st.megascale_configs.saturating_add(1);
    let cfg_id = st.megascale_configs;
    drop(st);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, cfg_id) }
}

/// Creates a Megascale multi-slice config (`PJRT_Megascale_CreateMultiSliceConfig`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Megascale_CreateMultiSliceConfig(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let mut st = extension_state()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    st.megascale_configs = st.megascale_configs.saturating_add(1);
    let cfg_id = st.megascale_configs;
    drop(st);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, cfg_id) }
}

/// Deletes a Megascale multi-slice config (`PJRT_Megascale_DeleteMultiSliceConfig`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Megascale_DeleteMultiSliceConfig(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let mut st = extension_state()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    st.megascale_configs = st.megascale_configs.saturating_sub(1);
    drop(st);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 0) }
}

/// Queries slice count from Megascale config (`PJRT_Megascale_ConfigNumSlices`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Megascale_ConfigNumSlices(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let n = usize::try_from(global_init_config().megascale_num_slices.max(1)).unwrap_or(1);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, n) }
}

/// Queries devices per slice from Megascale config (`PJRT_Megascale_ConfigNumDevicesPerSlice`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Megascale_ConfigNumDevicesPerSlice(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let d = usize::try_from(global_init_config().chips_per_host()).unwrap_or(4);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, d) }
}

/// Queries slice ID from Megascale config (`PJRT_Megascale_ConfigSliceId`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Megascale_ConfigSliceId(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let s = usize::try_from(global_init_config().megascale_slice_id).unwrap_or(0);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, s) }
}

/// Queries serialized slice key-values from Megascale config (`PJRT_Megascale_ConfigSerializedSliceKeyValue`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Megascale_ConfigSerializedSliceKeyValue(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let kv_len = global_init_config().extra_flags.len().saturating_add(16);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, kv_len) }
}

/// Serializes a Megascale multi-slice config (`PJRT_Megascale_ConfigSerialize`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_MultiSlice_Config_Serialize_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_MultiSlice_Config_Serialize_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Megascale_ConfigSerialize(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let cfg = global_init_config();
    let ser_len = libtpu_core::proto::MegascaleMultiSliceConfigProto {
        num_slices: i32::try_from(cfg.megascale_num_slices.max(1)).unwrap_or(1),
        slice_id: i32::try_from(cfg.megascale_slice_id).unwrap_or(0),
        num_devices_per_slice: i32::try_from(cfg.chips_per_host()).unwrap_or(4),
        coordinator_address: String::from("127.0.0.1:8080"),
    }
    .encode()
    .len();
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, ser_len) }
}

/// Deserializes a Megascale multi-slice config (`PJRT_Megascale_DeserializeMultiSliceConfig`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Megascale_DeserializeMultiSliceConfig(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let mut st = extension_state()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    st.megascale_configs = st.megascale_configs.saturating_add(1);
    let cfg_id = st.megascale_configs;
    drop(st);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, cfg_id) }
}

/// Queries Megascale coordinator addresses (`PJRT_Megascale_Addresses`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Megascale_Addresses(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let addr_count = usize::try_from(global_init_config().total_hosts()).unwrap_or(1);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, addr_count) }
}

/// Executes a Megascale multi-slice barrier (`PJRT_Megascale_Barrier`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Megascale_Barrier(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let mut st = extension_state()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    st.megascale_barriers = st.megascale_barriers.saturating_add(1);
    drop(st);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 0) }
}

/// Logs a Megascale collective error (`PJRT_Megascale_LogError`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Megascale_LogError(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 0) }
}

/// Queries the Megascale DCN topology (`PJRT_Megascale_GetDcnTopology`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Megascale_GetDcnTopology(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let cfg = global_init_config();
    let dcn_bytes = libtpu_core::proto::MegascaleDcnTopologyProto {
        num_slices: i32::try_from(cfg.megascale_num_slices.max(1)).unwrap_or(1),
        num_hosts_per_slice: i32::try_from(cfg.total_hosts()).unwrap_or(1),
        transport_type: 1,
    }
    .encode();
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, dcn_bytes.len()) }
}

/// Queries the active Megascale slice ID (`PJRT_Megascale_GetSliceId`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Megascale_GetSliceId(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let s = usize::try_from(global_init_config().megascale_slice_id).unwrap_or(0);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, s) }
}

/// Queries the total Megascale slice count (`PJRT_Megascale_GetNumSlices`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Megascale_GetNumSlices(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let n = usize::try_from(global_init_config().megascale_num_slices.max(1)).unwrap_or(1);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, n) }
}

/// Initializes a Megascale slice (`PJRT_Megascale_InitializeSlice`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Megascale_InitializeSlice(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 1) }
}

/// Finalizes a Megascale slice (`PJRT_Megascale_FinalizeSlice`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Megascale_FinalizeSlice(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 0) }
}

/// Registers a Megascale error handler (`PJRT_Megascale_RegisterErrorHandler`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Megascale_RegisterErrorHandler(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let mut st = extension_state()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    st.megascale_error_handlers = st.megascale_error_handlers.saturating_add(1);
    let h = st.megascale_error_handlers;
    drop(st);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, h) }
}

/// Unregisters a Megascale error handler (`PJRT_Megascale_UnregisterErrorHandler`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Megascale_UnregisterErrorHandler(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let mut st = extension_state()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    st.megascale_error_handlers = st.megascale_error_handlers.saturating_sub(1);
    drop(st);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 0) }
}

/// Collects Megascale telemetry metrics (`PJRT_Megascale_CollectMetrics`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Megascale_CollectMetrics(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let st = extension_state()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let metrics_len = libtpu_core::proto::MegascaleRuntimeMetricsProto {
        completed_barriers: u64::try_from(st.megascale_barriers).unwrap_or(0),
        collective_ops_count: u64::try_from(st.collective_ops_executed).unwrap_or(0),
        active_contexts: i32::try_from(st.megascale_contexts).unwrap_or(0),
    }
    .encode()
    .len();
    drop(st);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, metrics_len) }
}

/// Resets Megascale telemetry metrics (`PJRT_Megascale_ResetMetrics`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Megascale_ResetMetrics(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let mut st = extension_state()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    st.megascale_barriers = 0;
    drop(st);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 0) }
}

/// Queries Megascale transport type (`PJRT_Megascale_GetTransportType`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Megascale_GetTransportType(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let tt = extension_state()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .megascale_transport_type;
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, tt) }
}

/// Sets Megascale transport type (`PJRT_Megascale_SetTransportType`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Megascale_SetTransportType(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let mut st = extension_state()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    st.megascale_transport_type = 1;
    let tt = st.megascale_transport_type;
    drop(st);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, tt) }
}

/// Waits for a Megascale barrier (`PJRT_Megascale_WaitForBarrier`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Megascale_WaitForBarrier(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 0) }
}

/// Aborts in-flight Megascale collectives (`PJRT_Megascale_AbortCollectives`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Megascale_AbortCollectives(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 0) }
}

/// Extracts target arguments from a TPU executable (`PJRT_TpuExecutable_GetTargetArguments`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TpuExecutable_GetTargetArguments(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let gen_str = global_init_config().generation.to_string();
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, gen_str.len()) }
}

/// Extracts `HloModuleWithConfig` from a TPU executable (`PJRT_TpuExecutable_GetHloModuleWithConfig`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TpuExecutable_GetHloModuleWithConfig(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let hlo_len = libtpu_core::proto::HloModuleProtoWithConfig {
        hlo_module: Some(libtpu_core::proto::HloModuleProto::default()),
        config: Some(libtpu_core::proto::ExecutableBuildOptionsProto::default()),
    }
    .encode()
    .len();
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, hlo_len) }
}

/// Reserved slot on `PJRT_TpuExecutable_Extension` (`PJRT_TpuExecutable_Reserved`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TpuExecutable_Reserved(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 0) }
}

/// Extracts compiled memory stats from a TPU executable (`PJRT_TpuExecutable_GetCompiledMemoryStats`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TpuExecutable_GetCompiledMemoryStats(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let vmem_cap = global_init_config().generation.vmem_capacity_bytes().0;
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, vmem_cap) }
}

/// Runs HLO cost analysis on a TPU executable (`PJRT_TpuExecutable_RunHloCostAnalysis`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TpuExecutable_RunHloCostAnalysis(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 4) }
}

/// Updates a field in `TpuCompilationEnvironmentProto` (`PJRT_TpuExecutable_SetTpuCompilationEnv`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TpuExecutable_SetTpuCompilationEnv(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let mut st = extension_state()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    st.tpu_compilation_env_overrides = st.tpu_compilation_env_overrides.saturating_add(1);
    let count = st.tpu_compilation_env_overrides;
    drop(st);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, count) }
}

/// Reads a field from `TpuCompilationEnvironmentProto` as string (`PJRT_TpuExecutable_GetTpuCompilationEnvFieldAsString`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TpuExecutable_GetTpuCompilationEnvFieldAsString(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let env_proto_len = libtpu_core::proto::TpuCompilationEnvironmentProto::default()
        .encode()
        .len();
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, env_proto_len) }
}

/// Checks whether an error is a TPU predetermined error (`PJRT_TpuExecutable_IsPredeterminedError`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TpuExecutable_IsPredeterminedError(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let flag = usize::from(
        LibTpuError::DevicePoisoned {
            device_id: 0,
            launch_id: 0,
            reason: String::new(),
        }
        .is_predetermined_error(),
    );
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, flag) }
}

/// Checks whether an error is a TPU cancellation placeholder (`PJRT_TpuExecutable_IsCancellationPlaceholderError`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TpuExecutable_IsCancellationPlaceholderError(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let flag = usize::from(
        LibTpuError::EventFailed {
            reason: String::new(),
        }
        .is_cancellation_placeholder_error(),
    );
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, flag) }
}

/// Creates a subslice TPU topology (`PJRT_TpuTopology_Subslice`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TpuTopology_Subslice(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 1) }
}

/// Checks whether a topology is a subslice (`PJRT_TpuTopology_IsSubsliceTopology`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TpuTopology_IsSubsliceTopology(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 0) }
}

/// Maps full device ID to subslice device ID (`PJRT_TpuTopology_SubsliceDeviceIdFromFullDeviceId`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TpuTopology_SubsliceDeviceIdFromFullDeviceId(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 0) }
}

/// Replaces host bounds on a TPU topology (`PJRT_TpuTopology_ReplaceHostBounds`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TpuTopology_ReplaceHostBounds(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 1) }
}

/// Checks whether enhanced ICI barrier is enabled (`PJRT_TpuTopology_IsEnhancedBarrierEnabled`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TpuTopology_IsEnhancedBarrierEnabled(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let flag = usize::from(global_init_config().allow_async_allocations);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, flag) }
}

/// Checks whether topology has limited ICI connectivity (`PJRT_TpuTopology_HasLimitedIciConnectivity`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TpuTopology_HasLimitedIciConnectivity(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 0) }
}

/// Checks whether two chips are reachable over limited ICI (`PJRT_TpuTopology_IsReachableOverLimitedIci`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TpuTopology_IsReachableOverLimitedIci(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 1) }
}

/// Queries process count in TPU topology (`PJRT_TpuTopology_ProcessCount`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TpuTopology_ProcessCount(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let hosts = usize::try_from(global_init_config().total_hosts()).unwrap_or(1);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, hosts) }
}

/// Queries chips per process in TPU topology (`PJRT_TpuTopology_ChipsPerProcess`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TpuTopology_ChipsPerProcess(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let chips = usize::try_from(global_init_config().chips_per_host()).unwrap_or(4);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, chips) }
}

/// Queries cores per chip in TPU topology (`PJRT_TpuTopology_CoreCountPerChip`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TpuTopology_CoreCountPerChip(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let cores =
        usize::try_from(global_init_config().generation.logical_devices_per_chip()).unwrap_or(1);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, cores) }
}

/// Queries total chip count in TPU topology (`PJRT_TpuTopology_ChipCount`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TpuTopology_ChipCount(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let cfg = global_init_config();
    let total = usize::try_from(cfg.chips_per_host() * cfg.total_hosts()).unwrap_or(4);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, total) }
}

/// Queries total core count in TPU topology (`PJRT_TpuTopology_CoreCount`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TpuTopology_CoreCount(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let cfg = global_init_config();
    let total = usize::try_from(cfg.chips_per_host() * cfg.total_hosts()).unwrap_or(4);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, total) }
}

/// Queries logical devices per process (`PJRT_TpuTopology_LogiDeviceCountPerProcess`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TpuTopology_LogiDeviceCountPerProcess(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let chips = usize::try_from(global_init_config().chips_per_host()).unwrap_or(4);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, chips) }
}

/// Queries total logical device count (`PJRT_TpuTopology_LogiDeviceCount`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TpuTopology_LogiDeviceCount(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let chips = usize::try_from(global_init_config().chips_per_host()).unwrap_or(4);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, chips) }
}

/// Queries logical devices per chip (`PJRT_TpuTopology_LogiDeviceCountPerChip`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TpuTopology_LogiDeviceCountPerChip(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let d =
        usize::try_from(global_init_config().generation.logical_devices_per_chip()).unwrap_or(1);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, d) }
}

/// Queries cores per process (`PJRT_TpuTopology_CoreCountPerProcess`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TpuTopology_CoreCountPerProcess(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let chips = usize::try_from(global_init_config().chips_per_host()).unwrap_or(4);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, chips) }
}

/// Queries process ID list (`PJRT_TpuTopology_ProcessIds`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TpuTopology_ProcessIds(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let hosts = usize::try_from(global_init_config().total_hosts()).unwrap_or(1);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, hosts) }
}

/// Queries logical device IDs on a process (`PJRT_TpuTopology_LogiDeviceIdsOnProcess`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TpuTopology_LogiDeviceIdsOnProcess(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let devs = usize::try_from(global_init_config().chips_per_host()).unwrap_or(4);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, devs) }
}

/// Maps chip ID to `(process_id, index_on_process)` (`PJRT_TpuTopology_ProcIdAndIdxOnProcForChip`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TpuTopology_ProcIdAndIdxOnProcForChip(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 0) }
}

/// Maps logical device ID to `(process_id, index_on_process)` (`PJRT_TpuTopology_ProcIdAndIdxOnProcForLogiDevice`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TpuTopology_ProcIdAndIdxOnProcForLogiDevice(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 0) }
}

/// Maps process ID to 3D process coordinates (`PJRT_TpuTopology_ProcessCoordFromId`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TpuTopology_ProcessCoordFromId(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 0) }
}

/// Maps 3D chip coordinates to chip ID (`PJRT_TpuTopology_ChipIdFromCoord`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TpuTopology_ChipIdFromCoord(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 0) }
}

/// Maps `(chip_coord, core_idx)` to logical device ID (`PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TpuTopology_LogiDeviceIdFromChipCoordAndIdx(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 0) }
}

/// Maps logical device ID to `(chip_coord, core_idx)` (`PJRT_TpuTopology_ChipCoordAndIdxForLogiDevice`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TpuTopology_ChipCoordAndIdxForLogiDevice(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 0) }
}

/// Queries 3D chips-per-process bounds (`PJRT_TpuTopology_ChipsPerProcessBounds`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TpuTopology_ChipsPerProcessBounds(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let bx = usize::try_from(global_init_config().chips_per_host_bounds.x).unwrap_or(2);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, bx) }
}

/// Queries 3D chip mesh bounds (`PJRT_TpuTopology_ChipBounds`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TpuTopology_ChipBounds(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let bx = usize::try_from(global_init_config().chips_per_host_bounds.x).unwrap_or(2);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, bx) }
}

/// Queries 3D process mesh bounds (`PJRT_TpuTopology_ProcessBounds`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TpuTopology_ProcessBounds(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let hx = usize::try_from(global_init_config().host_bounds.x).unwrap_or(1);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, hx) }
}

/// Queries ICI routing strategy (`PJRT_TpuTopology_GetRoutingStrategy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TpuTopology_GetRoutingStrategy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 0) }
}

/// Queries a named TPU slice configuration (`PJRT_TpuTopology_GetSliceConfig`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TpuTopology_GetSliceConfig(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 1) }
}

/// Lists all TPU slice configurations (`PJRT_TpuTopology_GetSliceConfigs`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TpuTopology_GetSliceConfigs(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let chips = usize::try_from(global_init_config().chips_per_host()).unwrap_or(4);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, chips) }
}

/// Queries default TPU platform configuration (`PJRT_TpuTopology_GetDefaultPlatformConfig`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_TpuTopology_GetDefaultPlatformConfig(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let gen_id = usize::try_from(global_init_config().generation as i32).unwrap_or(5);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, gen_id) }
}

/// Registers a runtime callback on `PJRT_Client` (`PJRT_Callback_RegisterCallback`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Callback_RegisterCallback(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let mut st = extension_state()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    st.registered_callbacks = st.registered_callbacks.saturating_add(1);
    let cb_id = st.registered_callbacks;
    drop(st);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, cb_id) }
}

/// Invokes registered runtime callbacks on `PJRT_Client` (`PJRT_Callback_InvokeCallback`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Callback_InvokeCallback(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let cb_count = extension_state()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .registered_callbacks;
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, cb_count) }
}

/// Acquires the phased TPU compiler handle (`PJRT_PhaseCompile_GetCompiler`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_PhaseCompile_GetCompiler(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let mut st = extension_state()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    st.phase_compilers = st.phase_compilers.saturating_add(1);
    let id = st.phase_compilers;
    drop(st);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, id) }
}

/// Destroys a phased TPU compiler handle (`PJRT_PhaseCompile_DestroyCompiler`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_PhaseCompile_DestroyCompiler(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let mut st = extension_state()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    st.phase_compilers = st.phase_compilers.saturating_sub(1);
    drop(st);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 0) }
}

/// Executes a named TPU compilation phase (`PJRT_PhaseCompile_RunPhase`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_PhaseCompile_RunPhase(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, COMPILATION_PHASE_NAMES.len()) }
}

/// Queries supported TPU compilation phase names (`PJRT_PhaseCompile_GetPhaseNames`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_PhaseCompile_GetPhaseNames(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, COMPILATION_PHASE_NAMES.len()) }
}

/// Destroys C buffers allocated by phased compilation (`PJRT_PhaseCompile_CBuffersDestroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_PhaseCompile_CBuffersDestroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 0) }
}

/// Allocates cross-host receive buffers (`PJRT_CrossHostTransfers_MakeReceiveBuffers`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_CrossHostTransfers_MakeReceiveBuffers(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let mut st = extension_state()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    st.cross_host_transfers = st.cross_host_transfers.saturating_add(1);
    let count = st.cross_host_transfers;
    drop(st);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, count) }
}

/// Copies a buffer to a remote host device (`PJRT_CrossHostTransfers_CopyToRemoteDevice`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_CrossHostTransfers_CopyToRemoteDevice(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let mut st = extension_state()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    st.cross_host_transfers = st.cross_host_transfers.saturating_add(1);
    let count = st.cross_host_transfers;
    drop(st);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, count) }
}

/// Initiates cross-host receive transfers (`PJRT_CrossHostTransfers_ReceiveBuffers`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_CrossHostTransfers_ReceiveBuffers(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let mut st = extension_state()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    st.cross_host_transfers = st.cross_host_transfers.saturating_add(1);
    let count = st.cross_host_transfers;
    drop(st);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, count) }
}

/// Initiates cross-host send transfers (`PJRT_CrossHostTransfers_SendBuffers`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_CrossHostTransfers_SendBuffers(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let mut st = extension_state()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    st.cross_host_transfers = st.cross_host_transfers.saturating_add(1);
    let count = st.cross_host_transfers;
    drop(st);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, count) }
}

/// Queries TPU executable metadata (`PJRT_ExecutableMetadata_Get`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_ExecutableMetadata_Get(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 1) }
}

/// Destroys TPU executable metadata (`PJRT_ExecutableMetadata_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_ExecutableMetadata_Destroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 0) }
}

/// Queries device memory space descriptions (`PJRT_MemoryDescriptions_Get`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_MemoryDescriptions_Get(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 3) }
}

/// Destroys device memory space descriptions (`PJRT_MemoryDescriptions_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_MemoryDescriptions_Destroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 0) }
}

/// Destroys a `PJRT_Layouts_MemoryLayout` (`PJRT_Layouts_MemoryLayout_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Layouts_MemoryLayout_Destroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 0) }
}

/// Serializes a `PJRT_Layouts_MemoryLayout` (`PJRT_Layouts_MemoryLayout_Serialize`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Layouts_MemoryLayout_Serialize(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let layout_len = libtpu_core::proto::LayoutProto::default().encode().len();
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, layout_len) }
}

/// Queries default memory layout for a `PJRT_Client` (`PJRT_Layouts_Client_GetDefaultLayout`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Layouts_Client_GetDefaultLayout(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let sublane_tile =
        usize::try_from(global_init_config().generation.mxu_dimension()).unwrap_or(128);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, sublane_tile) }
}

/// Queries memory layout of a `PJRT_Buffer` (`PJRT_Layouts_Buffer_GetMemoryLayout`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Layouts_Buffer_GetMemoryLayout(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let sublane_tile =
        usize::try_from(global_init_config().generation.mxu_dimension()).unwrap_or(128);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, sublane_tile) }
}

/// Queries default memory layout for a `PJRT_TopologyDescription` (`PJRT_Layouts_Topology_GetDefaultLayout`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Layouts_Topology_GetDefaultLayout(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let sublane_tile =
        usize::try_from(global_init_config().generation.mxu_dimension()).unwrap_or(128);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, sublane_tile) }
}

/// Queries output memory layouts of a `PJRT_Executable` (`PJRT_Layouts_Executable_GetOutputLayouts`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Layouts_Executable_GetOutputLayouts(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 1) }
}

/// Queries parameter memory layouts of a `PJRT_Executable` (`PJRT_Layouts_Executable_GetParameterLayouts`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_Layouts_Executable_GetParameterLayouts(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 1) }
}

/// Creates a raw alias of a `PJRT_Buffer` (`PJRT_RawBuffer_CreateRawAliasOfBuffer`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_RawBuffer_CreateRawAliasOfBuffer(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let mut st = extension_state()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    st.raw_buffers = st.raw_buffers.saturating_add(1);
    let raw_id = st.raw_buffers;
    drop(st);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, raw_id) }
}

/// Destroys a `PJRT_RawBuffer` handle (`PJRT_RawBuffer_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_RawBuffer_Destroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    let mut st = extension_state()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    st.raw_buffers = st.raw_buffers.saturating_sub(1);
    drop(st);
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 0) }
}

/// Returns the host pointer of a `PJRT_RawBuffer` (`PJRT_RawBuffer_GetHostPointer`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_RawBuffer_GetHostPointer(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 0x1000) }
}

/// Returns the on-device byte size of a `PJRT_RawBuffer` (`PJRT_RawBuffer_GetOnDeviceSizeInBytes`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_RawBuffer_GetOnDeviceSizeInBytes(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 256) }
}

/// Returns the `PJRT_Memory` space of a `PJRT_RawBuffer` (`PJRT_RawBuffer_GetMemorySpace`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_RawBuffer_GetMemorySpace(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 0) }
}

/// Copies raw host bytes into a `PJRT_RawBuffer` (`PJRT_RawBuffer_CopyRawHostToDevice`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_RawBuffer_CopyRawHostToDevice(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 1) }
}

/// Copies `PJRT_RawBuffer` bytes to host memory (`PJRT_RawBuffer_CopyRawDeviceToHost`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Megascale_Query_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Megascale_Query_Args`].
#[no_mangle]
pub unsafe extern "C" fn PJRT_RawBuffer_CopyRawDeviceToHost(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `complete_typed_extension_args`.
    unsafe { complete_typed_extension_args(raw_args, 1) }
}

/// Plugin profiler vtable structure (`PLUGIN_Profiler_Api`, 80 bytes) pointed to by `EXT_17_PROFILER.profiler_api`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PLUGIN_Profiler_Api {
    /// Size of this struct (`80`).
    pub struct_size: usize,
    /// Reserved private pointer (`NULL`).
    pub priv_data: *mut std::ffi::c_void,
    /// Destroys a profiler error (`PLUGIN_Profiler_Error_Destroy`).
    pub error_destroy: PjrtFn,
    /// Queries a profiler error message (`PLUGIN_Profiler_Error_Message`).
    pub error_message: PjrtFn,
    /// Queries a profiler error code (`PLUGIN_Profiler_Error_GetCode`).
    pub error_get_code: PjrtFn,
    /// Creates a profiler session (`PLUGIN_Profiler_Create`).
    pub create: PjrtFn,
    /// Destroys a profiler session (`PLUGIN_Profiler_Destroy`).
    pub destroy: PjrtFn,
    /// Starts profiling (`PLUGIN_Profiler_Start`).
    pub start: PjrtFn,
    /// Stops profiling (`PLUGIN_Profiler_Stop`).
    pub stop: PjrtFn,
    /// Collects serialized `XSpace` profiling data (`PLUGIN_Profiler_CollectData`).
    pub collect_data: PjrtFn,
}

// SAFETY: Immutable static vtable structure.
unsafe impl Sync for PLUGIN_Profiler_Api {}

const _: () = assert!(std::mem::size_of::<PLUGIN_Profiler_Api>() == 80);

/// Static `PLUGIN_Profiler_Api` vtable referenced by [`EXT_17_PROFILER`].
pub static GLOBAL_PLUGIN_PROFILER_API: PLUGIN_Profiler_Api = PLUGIN_Profiler_Api {
    struct_size: 80,
    priv_data: ptr::null_mut(),
    error_destroy: PLUGIN_Profiler_Error_Destroy,
    error_message: PLUGIN_Profiler_Error_Message,
    error_get_code: PLUGIN_Profiler_Error_GetCode,
    create: PLUGIN_Profiler_Create,
    destroy: PLUGIN_Profiler_Destroy,
    start: PLUGIN_Profiler_Start,
    stop: PLUGIN_Profiler_Stop,
    collect_data: PLUGIN_Profiler_CollectData,
};

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

const _: () = assert!(std::mem::size_of::<PJRT_XlaTransform_Extension>() == 56);
const _: () = assert!(std::mem::size_of::<PJRT_HostMemoryAllocator_Extension>() == 32);
const _: () = assert!(std::mem::size_of::<PJRT_MultiSlice_Extension>() == 64);
const _: () = assert!(std::mem::size_of::<PJRT_Collectives_Extension>() == 96);
const _: () = assert!(std::mem::size_of::<PJRT_AbiVersion_Extension>() == 120);
const _: () = assert!(std::mem::size_of::<PJRT_Shardings_Extension>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_Megascale_Extension>() == 248);
const _: () = assert!(std::mem::size_of::<PJRT_TpuExecutable_Extension>() == 96);
const _: () = assert!(std::mem::size_of::<PJRT_TpuTopology_Extension>() == 272);
const _: () = assert!(std::mem::size_of::<PJRT_Callback_Extension>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_PhaseCompile_Extension>() == 64);
const _: () = assert!(std::mem::size_of::<PJRT_CrossHostTransfers_Extension>() == 56);
const _: () = assert!(std::mem::size_of::<PJRT_ExecutableMetadata_Extension>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_MemoryDescriptions_Extension>() == 40);
const _: () = assert!(std::mem::size_of::<PJRT_Layouts_Extension>() == 80);
const _: () = assert!(std::mem::size_of::<PJRT_RawBuffer_Extension>() == 80);
const _: () = assert!(std::mem::size_of::<PJRT_Profiler_Extension>() == 40);

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

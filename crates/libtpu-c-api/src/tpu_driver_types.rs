//! `#[repr(C)]` structures for `GetLibtpuSdkApi`, `TpuStatus_*`, `TpuTopology_*`,
//! `TpuProfiler_*`, `TfTpuOrdinalSelector_*`, `TpuEmbeddingEngine_*`, and `TpuDriver_*`.

use libtpu_core::driver::{
    LibtpuSdkCoreClient, TfTpuCoreOrdinalSelector, TpuCoreDriver, TpuCoreEmbeddingEngine,
    TpuCoreProfiler, TpuCoreStatus,
};
use libtpu_core::hal::SimulatedTpuHal;
use std::ffi::CString;
use std::os::raw::c_char;

/// Exact byte size of `LibtpuSdk_Api` (`0x90` = 144 bytes).
pub const LIBTPU_SDK_API_STRUCT_SIZE: usize = 144;

/// Opaque C handle wrapping [`TpuCoreStatus`] (`TpuStatus`).
#[derive(Debug)]
pub struct TpuStatus {
    /// Underlying status state.
    pub inner: TpuCoreStatus,
    /// Cached null-terminated error message.
    pub c_message: CString,
}

/// Opaque C handle wrapping [`TfTpuCoreOrdinalSelector`].
#[derive(Debug)]
pub struct TfTpuOrdinalSelector {
    /// Underlying ordinal selector.
    pub inner: TfTpuCoreOrdinalSelector,
}

/// Opaque C handle wrapping [`TpuCoreProfiler`].
#[derive(Debug)]
pub struct TpuProfiler {
    /// Underlying profiler state.
    pub inner: TpuCoreProfiler,
}

/// Opaque C handle wrapping [`TpuCoreEmbeddingEngine`].
#[derive(Debug)]
pub struct TpuEmbeddingEngineState {
    /// Underlying embedding engine state.
    pub inner: TpuCoreEmbeddingEngine,
}

/// Opaque C handle representing a `TpuMeshState`.
#[derive(Debug)]
pub struct TpuMeshState {
    /// Common mesh state identifier.
    pub mesh_common_state: usize,
}

/// Opaque C handle representing a `TpuCoreLocation`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TpuCoreLocation {
    /// Global core identifier.
    pub core_id: i32,
    /// Core index on chip.
    pub index_on_chip: i32,
    /// Chip X coordinate.
    pub chip_x: i32,
    /// Chip Y coordinate.
    pub chip_y: i32,
    /// Chip Z coordinate.
    pub chip_z: i32,
}

/// Opaque C handle representing a `TpuHostLocation`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TpuHostLocation {
    /// Host identifier (`0..num_hosts`).
    pub host_id: i32,
    /// Host 3D coordinates `[x, y, z]`.
    pub host_coords: [i32; 3],
    /// Number of cores on this host.
    pub num_cores: i32,
}

/// Opaque C handle representing the global `TpuTopology`.
#[derive(Debug)]
pub struct TpuTopology {
    /// Underlying simulated HAL storing topology specs.
    pub hal: SimulatedTpuHal,
    /// Pre-built core locations.
    pub core_locations: Vec<TpuCoreLocation>,
}

/// Function pointer table stored at offset `0x00` of [`TpuDriver`] (`TpuDriverFn`, `224` bytes, 28 slots).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TpuDriverFn {
    /// Slot 0 (`0x00`): `TpuDriver_Open`.
    pub open: *const std::ffi::c_void,
    /// Slot 1 (`0x08`): `TpuDriver_Close`.
    pub close: *const std::ffi::c_void,
    /// Slot 2 (`0x10`): `TpuDriver_ComputeLinearizedBytesFromShape`.
    pub compute_linearized_bytes_from_shape: *const std::ffi::c_void,
    /// Slot 3 (`0x18`): `TpuDriver_LinearizeShape`.
    pub linearize_shape: *const std::ffi::c_void,
    /// Slot 4 (`0x20`): `TpuDriver_DelinearizeShape`.
    pub delinearize_shape: *const std::ffi::c_void,
    /// Slot 5 (`0x28`): `TpuDriver_CompileProgram`.
    pub compile_program: *const std::ffi::c_void,
    /// Slot 6 (`0x30`): `TpuDriver_CompileProgramFromText`.
    pub compile_program_from_text: *const std::ffi::c_void,
    /// Slot 7 (`0x38`): `TpuDriver_GetCompiledProgramShape`.
    pub get_compiled_program_shape: *const std::ffi::c_void,
    /// Slot 8 (`0x40`): `TpuDriver_LoadProgram`.
    pub load_program: *const std::ffi::c_void,
    /// Slot 9 (`0x48`): `TpuDriver_UnloadProgram`.
    pub unload_program: *const std::ffi::c_void,
    /// Slot 10 (`0x50`): `TpuDriver_ExecuteProgram`.
    pub execute_program: *const std::ffi::c_void,
    /// Slot 11 (`0x58`): `TpuDriver_AllocateTuple`.
    pub allocate_tuple: *const std::ffi::c_void,
    /// Slot 12 (`0x60`): `TpuDriver_Allocate`.
    pub allocate: *const std::ffi::c_void,
    /// Slot 13 (`0x68`): `TpuDriver_AllocateShape`.
    pub allocate_shape: *const std::ffi::c_void,
    /// Slot 14 (`0x70`): `TpuDriver_Deallocate`.
    pub deallocate: *const std::ffi::c_void,
    /// Slot 15 (`0x78`): `TpuDriver_TransferToDevice`.
    pub transfer_to_device: *const std::ffi::c_void,
    /// Slot 16 (`0x80`): `TpuDriver_TransferFromDevice`.
    pub transfer_from_device: *const std::ffi::c_void,
    /// Slot 17 (`0x88`): `TpuDriver_TransferFromDeviceToDevice`.
    pub transfer_from_device_to_device: *const std::ffi::c_void,
    /// Slot 18 (`0x90`): `TpuDriver_CreateEvent`.
    pub create_event: *const std::ffi::c_void,
    /// Slot 19 (`0x98`): `TpuDriver_FreeEvent`.
    pub free_event: *const std::ffi::c_void,
    /// Slot 20 (`0xa0`): `TpuDriver_AddCallback`.
    pub add_callback: *const std::ffi::c_void,
    /// Slot 21 (`0xa8`): `TpuDriver_WaitForEvent`.
    pub wait_for_event: *const std::ffi::c_void,
    /// Slot 22 (`0xb0`): `TpuDriver_FreeStatus`.
    pub free_status: *const std::ffi::c_void,
    /// Slot 23 (`0xb8`): `TpuDriver_Reset`.
    pub reset: *const std::ffi::c_void,
    /// Slot 24 (`0xc0`): `TpuDriver_QuerySystemInfo`.
    pub query_system_info: *const std::ffi::c_void,
    /// Slot 25 (`0xc8`): `TpuDriver_FreeSystemInfo`.
    pub free_system_info: *const std::ffi::c_void,
    /// Slot 26 (`0xd0`): `TpuDriver_FreeCompiledProgramShape`.
    pub free_compiled_program_shape: *const std::ffi::c_void,
    /// Slot 27 (`0xd8`): `TpuDriver_Version`.
    pub version: *const std::ffi::c_void,
}

const _: () = assert!(size_of::<TpuDriverFn>() == 224);

// SAFETY: `TpuDriverFn` contains immutable static function pointers.
unsafe impl Sync for TpuDriverFn {}

/// Opaque C handle wrapping [`TpuCoreDriver`] with `fn_table` at offset `0x00`.
#[repr(C)]
#[derive(Debug)]
pub struct TpuDriver {
    /// Pointer to the static [`TpuDriverFn`] vtable.
    pub fn_table: *const TpuDriverFn,
    /// Underlying driver instance.
    pub inner: TpuCoreDriver,
}

/// Serialized HLO program descriptor passed to `TpuDriver_Compile` (`16` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HloProto {
    /// Pointer to serialized `HloModuleProto` or HLO text bytes.
    pub buffer: *const std::ffi::c_void,
    /// Byte length of `buffer`.
    pub size: i64,
}

/// Logical device assignment passed to `TpuDriver_ExecuteProgram` (`8` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DeviceAssignment {
    /// Number of replicas.
    pub replica_count: i32,
    /// Number of computations (partitions).
    pub computation_count: i32,
}

/// Synchronization event handle used by `TpuDriver_*` (`16` bytes).
#[repr(C)]
#[derive(Debug)]
pub struct TpuEvent {
    /// Event identifier (`> 0`).
    pub event_id: u64,
    /// Owned status associated with the event.
    pub status: *mut TpuStatus,
}

/// System topology information returned by `TpuDriver_QuerySystemInfo` (`24` bytes).
#[repr(C)]
#[derive(Debug)]
pub struct TpuSystemInfo {
    /// Number of physical TPU chips.
    pub num_chips: i32,
    /// Number of `TensorCores` across all chips.
    pub num_cores: i32,
    /// Owned summary string C buffer.
    pub info_str: CString,
}

/// Compiled program handle returned by `TpuDriver_Compile` (`16` bytes).
#[repr(C)]
#[derive(Debug)]
pub struct TpuCompiledProgramHandle {
    /// Underlying compiled program.
    pub program: libtpu_core::compiler::CompiledProgram,
}

/// Loaded program handle returned by `TpuDriver_LoadProgram` (`16` bytes).
#[repr(C)]
#[derive(Debug)]
pub struct TpuLoadedProgramHandle {
    /// Target core ordinal on which the program is loaded.
    pub core_id: i32,
    /// Underlying compiled program.
    pub program: libtpu_core::compiler::CompiledProgram,
}

/// Device buffer handle returned by `TpuDriver_Allocate` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TpuBufferHandle {
    /// On-device HBM address.
    pub address: u64,
    /// Size of the allocation in bytes.
    pub size_in_bytes: i64,
    /// Core ordinal owning the buffer.
    pub core_id: i32,
    /// Memory region (`0` = HBM, `1` = VMEM).
    pub memory_region: i32,
    /// Optional completion event pointer.
    pub event: *mut TpuEvent,
}

/// C-ABI-safe structure representing a chip coordinate entry in `LibtpuSdk_GetChipCoordinates` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LibtpuSdk_ChipLocationToCoordinates {
    /// Pointer to null-terminated UTF-8 hostname string owned by [`LibtpuSdk_Client`].
    pub host_name_ptr: *const c_char,
    /// Byte length of the hostname string (excluding null terminator).
    pub host_name_len: usize,
    /// Zero-based chip index on the host tray.
    pub chip_index: i32,
    /// Explicit 4-byte padding for 8-byte alignment.
    pub pad: i32,
    /// 3D Cartesian coordinates `[x, y, z]`.
    pub coordinates: [i64; 3],
}

const _: () = assert!(size_of::<LibtpuSdk_ChipLocationToCoordinates>() == 48);

/// Opaque C handle wrapping [`LibtpuSdkCoreClient`].
#[derive(Debug)]
pub struct LibtpuSdk_Client {
    /// Underlying SDK client.
    pub inner: LibtpuSdkCoreClient,
    /// Owned hostname C strings backing `coords_entries`.
    pub host_names: Vec<CString>,
    /// Pre-allocated coordinate entries returned by `LibtpuSdk_GetChipCoordinates`.
    pub coords_entries: Vec<LibtpuSdk_ChipLocationToCoordinates>,
}

/// Opaque C handle representing a `LibtpuSdk_Error`.
#[derive(Debug)]
pub struct LibtpuSdk_Error {
    /// Error status code.
    pub code: i32,
    /// Cached null-terminated message.
    pub c_message: CString,
}

/// Opaque C handle representing a queried `LibtpuSdk_Metric`.
#[derive(Debug)]
pub struct LibtpuSdk_Metric {
    /// Cached C metric description string.
    pub description: CString,
    /// Owned C strings for each metric value.
    pub value_cstrings: Vec<CString>,
    /// Array of raw `*const c_char` pointers into `value_cstrings`.
    pub value_ptrs: Vec<*const c_char>,
}

/// Opaque C handle representing a `LibtpuSdk_RuntimeStatus`.
#[derive(Debug)]
pub struct LibtpuSdk_RuntimeStatus {
    /// Cached C core state summary string.
    pub summary: CString,
}

/// Argument structure for `LibtpuSdk_Error_GetMessage` (`24` bytes).
#[repr(C)]
#[derive(Debug)]
pub struct LibtpuSdk_Error_GetMessage_Args {
    /// Input `LibtpuSdk_Error*`.
    pub error: *const LibtpuSdk_Error,
    /// Output UTF-8 message pointer.
    pub message: *const c_char,
    /// Output message length in bytes.
    pub message_size: usize,
}

/// Argument structure for `LibtpuSdk_Error_Destroy` (`8` bytes).
#[repr(C)]
#[derive(Debug)]
pub struct LibtpuSdk_Error_Destroy_Args {
    /// Input `LibtpuSdk_Error*` to destroy.
    pub error: *mut LibtpuSdk_Error,
}

/// Argument structure for `LibtpuSdk_Error_GetCode` (`16` bytes).
#[repr(C)]
#[derive(Debug)]
pub struct LibtpuSdk_Error_GetCode_Args {
    /// Input `LibtpuSdk_Error*`.
    pub error: *const LibtpuSdk_Error,
    /// Output status code.
    pub code: i32,
    /// Explicit 4-byte padding.
    pub pad: i32,
}

/// Argument structure for `LibtpuSdk_Client_Create` and `LibtpuSdk_Client_Destroy` (`8` bytes).
#[repr(C)]
#[derive(Debug)]
pub struct LibtpuSdk_Client_Args {
    /// In/out `LibtpuSdk_Client*`.
    pub client: *mut LibtpuSdk_Client,
}

/// Argument structure for `LibtpuSdk_GetChipCoordinates` (`24` bytes).
#[repr(C)]
#[derive(Debug)]
pub struct LibtpuSdk_GetChipCoordinates_Args {
    /// Input `LibtpuSdk_Client*`.
    pub client: *mut LibtpuSdk_Client,
    /// Output pointer to array of `LibtpuSdk_ChipLocationToCoordinates`.
    pub coords: *const LibtpuSdk_ChipLocationToCoordinates,
    /// Output number of coordinate entries.
    pub num_coords: usize,
}

/// Argument structure for `LibtpuSdk_GetHostName` (`24` bytes).
#[repr(C)]
#[derive(Debug)]
pub struct LibtpuSdk_GetHostName_Args {
    /// Input `LibtpuSdk_ChipLocationToCoordinates*`.
    pub coord: *const LibtpuSdk_ChipLocationToCoordinates,
    /// Output hostname pointer.
    pub host_name: *const c_char,
    /// Output hostname length in bytes.
    pub host_name_size: usize,
}

/// Argument structure for `LibtpuSdk_GetChipIndex` (`16` bytes).
#[repr(C)]
#[derive(Debug)]
pub struct LibtpuSdk_GetChipIndex_Args {
    /// Input `LibtpuSdk_ChipLocationToCoordinates*`.
    pub coord: *const LibtpuSdk_ChipLocationToCoordinates,
    /// Output chip index.
    pub chip_index: i32,
    /// Explicit 4-byte padding.
    pub pad: i32,
}

/// Argument structure for `LibtpuSdk_GetCartesianCoordinates` (`24` bytes).
#[repr(C)]
#[derive(Debug)]
pub struct LibtpuSdk_GetCartesianCoordinates_Args {
    /// Input `LibtpuSdk_ChipLocationToCoordinates*`.
    pub coord: *const LibtpuSdk_ChipLocationToCoordinates,
    /// Output pointer to `[x, y, z]` `i64` coordinates array.
    pub coordinates: *const i64,
    /// Output number of coordinates (`3`).
    pub num_coordinates: usize,
}

/// Argument structure for `LibtpuSdk_GetMetric` (`24` bytes).
#[repr(C)]
#[derive(Debug)]
pub struct LibtpuSdk_GetMetric_Args {
    /// Input `LibtpuSdk_Client*`.
    pub client: *mut LibtpuSdk_Client,
    /// Input null-terminated metric name C string.
    pub metric_name: *const c_char,
    /// Output `LibtpuSdk_Metric*` (allocated on heap, freed when next queried or destroyed).
    pub metric: *mut LibtpuSdk_Metric,
}

/// Argument structure for `LibtpuSdk_GetMetricDescription` (`24` bytes).
#[repr(C)]
#[derive(Debug)]
pub struct LibtpuSdk_GetMetricDescription_Args {
    /// Input `LibtpuSdk_Metric*`.
    pub metric: *const LibtpuSdk_Metric,
    /// Output metric description string pointer.
    pub description: *const c_char,
    /// Output metric description length in bytes.
    pub description_size: usize,
}

/// Argument structure for `LibtpuSdk_GetMetricValues` (`24` bytes).
#[repr(C)]
#[derive(Debug)]
pub struct LibtpuSdk_GetMetricValues_Args {
    /// Input `LibtpuSdk_Metric*`.
    pub metric: *const LibtpuSdk_Metric,
    /// Output array of `*const c_char` value strings.
    pub values: *const *const c_char,
    /// Output number of values.
    pub num_values: usize,
}

/// Argument structure for `LibtpuSdk_GetRuntimeStatus` (`24` bytes).
#[repr(C)]
#[derive(Debug)]
pub struct LibtpuSdk_GetRuntimeStatus_Args {
    /// Input `LibtpuSdk_Client*`.
    pub client: *mut LibtpuSdk_Client,
    /// Whether to include active HLO execution info.
    pub include_hlo_info: bool,
    /// Explicit 7-byte alignment padding.
    pub pad: [u8; 7],
    /// Output `LibtpuSdk_RuntimeStatus*`.
    pub status: *mut LibtpuSdk_RuntimeStatus,
}

/// Argument structure for `LibtpuSdk_RuntimeStatus_GetCoreStateSummary` (`24` bytes).
#[repr(C)]
#[derive(Debug)]
pub struct LibtpuSdk_RuntimeStatus_GetCoreStateSummary_Args {
    /// Input `LibtpuSdk_RuntimeStatus*`.
    pub status: *const LibtpuSdk_RuntimeStatus,
    /// Output summary string pointer.
    pub summary: *const c_char,
    /// Output summary length in bytes.
    pub summary_size: usize,
}

/// Argument structure for `LibtpuSdk_RuntimeStatus_Destroy` (`8` bytes).
#[repr(C)]
#[derive(Debug)]
pub struct LibtpuSdk_RuntimeStatus_Destroy_Args {
    /// Input `LibtpuSdk_RuntimeStatus*` to destroy.
    pub status: *mut LibtpuSdk_RuntimeStatus,
}

/// Argument structure for `LibtpuSdk_RegisterDebugCallback` (`32` bytes).
#[repr(C)]
#[derive(Debug)]
pub struct LibtpuSdk_RegisterDebugCallback_Args {
    /// Input `LibtpuSdk_Client*`.
    pub client: *mut LibtpuSdk_Client,
    /// Callback function pointer.
    pub callback: *const std::ffi::c_void,
    /// Opaque user context pointer.
    pub user_data: *mut std::ffi::c_void,
    /// Output registered callback ID (`i64`).
    pub callback_id: i64,
}

/// Argument structure for `LibtpuSdk_UnregisterDebugCallback` (`16` bytes).
#[repr(C)]
#[derive(Debug)]
pub struct LibtpuSdk_UnregisterDebugCallback_Args {
    /// Input `LibtpuSdk_Client*`.
    pub client: *mut LibtpuSdk_Client,
    /// Callback ID to unregister.
    pub callback_id: i64,
}

/// Void-returning function pointer slot in `LibtpuSdk_Api` (slots 1 and 2).
pub type SdkVoidFn = unsafe extern "C" fn(*mut std::ffi::c_void);

/// Error-returning function pointer slot in `LibtpuSdk_Api` (slots 3..17).
pub type SdkFn = unsafe extern "C" fn(*mut std::ffi::c_void) -> *mut LibtpuSdk_Error;

/// Exact 144-byte (`18 * 8`) `LibtpuSdk_Api` vtable returned by `GetLibtpuSdkApi`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LibtpuSdk_Api {
    /// Version header (`major = 0`, `minor = 1` packed as `0x0000_0001_0000_0000`).
    pub version_header: u64,
    /// Slot 1 (`0x08`): `LibtpuSdk_Error_GetMessage` (returns `void`).
    pub error_get_message: SdkVoidFn,
    /// Slot 2 (`0x10`): `LibtpuSdk_Error_Destroy` (returns `void`).
    pub error_destroy: SdkVoidFn,
    /// Slot 3 (`0x18`): `LibtpuSdk_Error_GetCode`.
    pub error_get_code: SdkFn,
    /// Slot 4 (`0x20`): `LibtpuSdk_Client_Create`.
    pub client_create: SdkFn,
    /// Slot 5 (`0x28`): `LibtpuSdk_Client_Destroy`.
    pub client_destroy: SdkFn,
    /// Slot 6 (`0x30`): `LibtpuSdk_GetChipCoordinates`.
    pub get_chip_coordinates: SdkFn,
    /// Slot 7 (`0x38`): `LibtpuSdk_GetHostName`.
    pub get_host_name: SdkFn,
    /// Slot 8 (`0x40`): `LibtpuSdk_GetChipIndex`.
    pub get_chip_index: SdkFn,
    /// Slot 9 (`0x48`): `LibtpuSdk_GetCartesianCoordinates`.
    pub get_cartesian_coordinates: SdkFn,
    /// Slot 10 (`0x50`): `LibtpuSdk_GetMetric`.
    pub get_metric: SdkFn,
    /// Slot 11 (`0x58`): `LibtpuSdk_GetMetricDescription`.
    pub get_metric_description: SdkFn,
    /// Slot 12 (`0x60`): `LibtpuSdk_GetMetricValues`.
    pub get_metric_values: SdkFn,
    /// Slot 13 (`0x68`): `LibtpuSdk_GetRuntimeStatus`.
    pub get_runtime_status: SdkFn,
    /// Slot 14 (`0x70`): `LibtpuSdk_RuntimeStatus_GetCoreStateSummary`.
    pub runtime_status_get_core_state_summary: SdkFn,
    /// Slot 15 (`0x78`): `LibtpuSdk_RuntimeStatus_Destroy`.
    pub runtime_status_destroy: SdkFn,
    /// Slot 16 (`0x80`): `LibtpuSdk_RegisterDebugCallback`.
    pub register_debug_callback: SdkFn,
    /// Slot 17 (`0x88`): `LibtpuSdk_UnregisterDebugCallback`.
    pub unregister_debug_callback: SdkFn,
}

// SAFETY: `LibtpuSdk_Api` contains only static function pointers and version constant.
unsafe impl Sync for LibtpuSdk_Api {}

const _: () = assert!(size_of::<LibtpuSdk_Api>() == LIBTPU_SDK_API_STRUCT_SIZE);
const _: () = assert!(size_of::<LibtpuSdk_Error_GetMessage_Args>() == 24);
const _: () = assert!(size_of::<LibtpuSdk_Error_Destroy_Args>() == 8);
const _: () = assert!(size_of::<LibtpuSdk_Error_GetCode_Args>() == 16);
const _: () = assert!(size_of::<LibtpuSdk_Client_Args>() == 8);
const _: () = assert!(size_of::<LibtpuSdk_GetChipCoordinates_Args>() == 24);
const _: () = assert!(size_of::<LibtpuSdk_GetHostName_Args>() == 24);
const _: () = assert!(size_of::<LibtpuSdk_GetChipIndex_Args>() == 16);
const _: () = assert!(size_of::<LibtpuSdk_GetCartesianCoordinates_Args>() == 24);
const _: () = assert!(size_of::<LibtpuSdk_GetMetric_Args>() == 24);
const _: () = assert!(size_of::<LibtpuSdk_GetMetricDescription_Args>() == 24);
const _: () = assert!(size_of::<LibtpuSdk_GetMetricValues_Args>() == 24);
const _: () = assert!(size_of::<LibtpuSdk_GetRuntimeStatus_Args>() == 24);
const _: () = assert!(size_of::<LibtpuSdk_RuntimeStatus_GetCoreStateSummary_Args>() == 24);
const _: () = assert!(size_of::<LibtpuSdk_RuntimeStatus_Destroy_Args>() == 8);
const _: () = assert!(size_of::<LibtpuSdk_RegisterDebugCallback_Args>() == 32);
const _: () = assert!(size_of::<LibtpuSdk_UnregisterDebugCallback_Args>() == 16);

/// Clean C ABI status structure returned by `libtpu_rs_*` functions for C and C++20 consumers.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LibTpuCStatus {
    /// Status code (`0` = OK, non-zero = `PjrtErrorCode`).
    pub code: i32,
    /// Null-terminated error message pointer (or null when `code == 0`).
    pub message: *mut c_char,
}

/// Parameter structure for `TFNPD_InitPlugin` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TFNPD_PluginParams {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Private pointer (`NULL`).
    pub priv_data: *mut std::ffi::c_void,
    /// Output device type C string (`"TPU"`).
    pub device_type: *const c_char,
    /// Output compilation device name C string (`"TPU_REPLICATED_CORE"`).
    pub compilation_device_name: *const c_char,
    /// Output priority (`55`).
    pub priority: i32,
    /// Output pluggable device flag (`true`).
    pub is_pluggable_device: bool,
    /// Output on-demand compile flag (`true`).
    pub use_pjrt_on_demand_compile: bool,
    /// Explicit 2-byte padding.
    pub pad: u16,
}

/// Parameter structure for `TfTpu_GetTpuPartitionedCallParams` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TpuPartitionedCall_Params {
    /// Input shape optimization flag.
    pub input_shape_opt: bool,
    /// Tensor packing flag.
    pub group_tensors_for_packing: bool,
    /// Explicit 2-byte padding.
    pub pad0: u16,
    /// Minimum input tensors for packing.
    pub minimum_input_tensors_packing: i32,
    /// Minimum output tensors for packing.
    pub minimum_output_tensors_packing: i32,
    /// Automatic XLA input sharding flag.
    pub enable_auto_xla_input_sharding: bool,
    /// Explicit 3-byte padding.
    pub pad1: [u8; 3],
    /// Automatic XLA input sharding dimension.
    pub auto_xla_input_sharding_dim: i32,
    /// Scoped allocator / variable deduplication flag (`enable_variable_deduplication` in `xla/tpu/tpu_ops_c_api.h`).
    pub scoped_allocator_for_tpu_anned_call: bool,
    /// Explicit 3-byte padding.
    pub pad2: [u8; 3],
}

impl TpuPartitionedCall_Params {
    /// Returns the upstream `enable_variable_deduplication` flag (`xla/tpu/tpu_ops_c_api.h`).
    #[must_use]
    pub const fn enable_variable_deduplication(&self) -> bool {
        self.scoped_allocator_for_tpu_anned_call
    }
}

/// XLA shape structure for `TpuTransferManager_GetInfeedLayout` (`96` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct XLA_Shape {
    /// Element type (`i32`).
    pub element_type: i32,
    /// Rank (`i32`).
    pub rank: i32,
    /// Dimensions array pointer.
    pub dimensions: *const i64,
    /// Minor-to-major ordering (`[i64; 8]`).
    pub minor_to_major: [i64; 8],
    /// Tile dimensions (`[i64; 2]`).
    pub tile_dims: [i64; 2],
}

/// Serialized protobuf buffer descriptor (`TpuSerializedProto`, `16` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TpuSerializedProto {
    /// Heap-allocated protobuf bytes pointer.
    pub bytes: *mut c_char,
    /// Number of bytes.
    pub size: usize,
}

/// Parameter structure for `TpuEmbeddingEngine_ExecutePartitioner` (`56` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TpuEmbeddingEngine_ExecutePartitioner_Params {
    /// Size of this struct (`56`).
    pub struct_size: usize,
    /// Reserved private pointer (`NULL`).
    pub priv_data: *mut std::ffi::c_void,
    /// Input `TpuEmbeddingConfigurationProto` bytes pointer.
    pub tpu_embedding_config_bytes: *const c_char,
    /// Input `TpuEmbeddingConfigurationProto` byte length.
    pub tpu_embedding_config_size: usize,
    /// Output byte length pointer.
    pub common_config_string_size: *mut usize,
    /// Output heap-allocated `char*` pointer.
    pub common_config_string: *mut *mut c_char,
    /// Optional status pointer.
    pub status: *mut TpuStatus,
}

/// Type alias preserving compatibility with callers referencing `TpuEmbeddingEngine_Params`.
pub type TpuEmbeddingEngine_Params = TpuEmbeddingEngine_ExecutePartitioner_Params;

/// Parameter structure for `TpuEmbeddingEngine_ConfigureMemory` (`64` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TpuEmbeddingEngine_ConfigureMemory_Params {
    /// Size of this struct (`64`).
    pub struct_size: usize,
    /// Reserved private pointer (`NULL`).
    pub priv_data: *mut std::ffi::c_void,
    /// Number of embedding inputs.
    pub num_inputs: i32,
    /// Explicit 4-byte padding.
    pub pad: i32,
    /// Input configuration string pointer.
    pub common_config_string: *const c_char,
    /// Input configuration string size.
    pub common_config_string_size: usize,
    /// Output size pointer.
    pub task_host_config_size: *mut usize,
    /// Output buffer pointer.
    pub task_host_config: *mut *mut c_char,
    /// Optional status pointer.
    pub status: *mut TpuStatus,
}

/// Parameter structure for `TpuEmbeddingEngine_CollateMemory` (`56` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TpuEmbeddingEngine_CollateMemory_Params {
    /// Size of this struct (`56`).
    pub struct_size: usize,
    /// Reserved private pointer (`NULL`).
    pub priv_data: *mut std::ffi::c_void,
    /// Number of memory configurations (`memory_configs_count`).
    pub memory_configs_size: usize,
    /// Array of `TpuSerializedProto` per-host memory configurations.
    pub memory_configs: *const TpuSerializedProto,
    /// Output merged memory configuration size pointer (`merged_memory_config_size`).
    pub collated_size: *mut usize,
    /// Output merged memory configuration buffer pointer (`merged_memory_config`).
    pub merged_memory_config: *mut *mut c_char,
    /// Optional status pointer.
    pub status: *mut TpuStatus,
}

/// Parameter structure for `TpuEmbeddingEngine_ConfigureHost` (`96` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TpuEmbeddingEngine_ConfigureHost_Params {
    /// Size of this struct (`96`).
    pub struct_size: usize,
    /// Reserved private pointer (`NULL`).
    pub priv_data: *mut std::ffi::c_void,
    /// Number of inputs.
    pub num_inputs: i32,
    /// Explicit 4-byte padding.
    pub pad: i32,
    /// Common configuration string pointer.
    pub common_config_string: *const c_char,
    /// Common configuration string byte length.
    pub common_config_string_size: usize,
    /// Task host memory configuration bytes pointer.
    pub task_host_config: *const c_char,
    /// Task host memory configuration byte length.
    pub task_host_config_size: usize,
    /// `TpuEmbeddingConfigurationProto` bytes pointer.
    pub tpu_embedding_config_bytes: *const c_char,
    /// `TpuEmbeddingConfigurationProto` byte length.
    pub tpu_embedding_config_size: usize,
    /// Output network configuration size pointer.
    pub network_config_size: *mut usize,
    /// Output network configuration buffer pointer.
    pub network_config: *mut *mut c_char,
    /// Optional status pointer.
    pub status: *mut TpuStatus,
}

/// Parameter structure for `TpuEmbeddingEngine_ConnectHosts` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TpuEmbeddingEngine_ConnectHosts_Params {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Reserved private pointer (`NULL`).
    pub priv_data: *mut std::ffi::c_void,
    /// Number of network configurations.
    pub network_configs_size: usize,
    /// Array of `TpuSerializedProto` network configurations.
    pub network_configs: *const TpuSerializedProto,
    /// Optional status pointer.
    pub status: *mut TpuStatus,
}

/// Parameter structure for `TpuEmbeddingEngine_Finalize` (`64` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TpuEmbeddingEngine_Finalize_Params {
    /// Size of this struct (`64`).
    pub struct_size: usize,
    /// Reserved private pointer (`NULL`).
    pub priv_data: *mut std::ffi::c_void,
    /// Mesh common state pointer.
    pub mesh_common_state: *const std::ffi::c_void,
    /// Common configuration string pointer.
    pub common_config_string: *const c_char,
    /// Common configuration string byte length.
    pub common_config_string_size: usize,
    /// Task host configuration pointer.
    pub task_host_config: *const c_char,
    /// Task host configuration byte length.
    pub task_host_config_size: usize,
    /// Optional status pointer.
    pub status: *mut TpuStatus,
}

/// Parameter structure for `TpuEmbeddingEngine_IsInitialized` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TpuEmbeddingEngine_IsInitialized_Params {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Reserved private pointer (`NULL`).
    pub priv_data: *mut std::ffi::c_void,
    /// Configuration string pointer.
    pub config_string: *const c_char,
    /// Configuration string byte length.
    pub config_string_size: usize,
    /// Output initialization flag pointer (`bool*`).
    pub is_embedding_initialized: *mut bool,
    /// Optional status pointer.
    pub status: *mut TpuStatus,
}

/// Parameter structure for `TpuEmbeddingEngine_WriteParameters` and `ReadParameters` (`72` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TpuEmbeddingEngine_WriteParameters_Params {
    /// Size of this struct (`72`).
    pub struct_size: usize,
    /// Reserved private pointer (`NULL`).
    pub priv_data: *mut std::ffi::c_void,
    /// Embedding table name pointer (or null for `"default_table"`).
    pub table_name: *const c_char,
    /// Embedding table name byte length.
    pub table_name_len: usize,
    /// Embedding table input float parameters pointer (`parameters_in` / `parameters`).
    pub parameters: *const std::ffi::c_void,
    /// Embedding table output float parameters pointer (`parameters_out`).
    pub parameters_out: *mut f32,
    /// Number of `f32` parameters to write or read.
    pub num_parameters: usize,
    /// Optional output pointer receiving number of `f32` parameters written or read.
    pub num_parameters_written_or_read: *mut usize,
    /// Optional status pointer.
    pub status: *mut TpuStatus,
}

/// Type alias for `TpuEmbeddingEngine_ReadParameters` parameter structure (`72` bytes).
pub type TpuEmbeddingEngine_ReadParameters_Params = TpuEmbeddingEngine_WriteParameters_Params;

/// Parameter structure for `TpuEmbeddingEngine_EnqueueTensorBatch` (`56` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TpuEmbeddingEngine_EnqueueTensorBatch_Params {
    /// Size of this struct (`56`).
    pub struct_size: usize,
    /// Reserved private pointer (`NULL`).
    pub priv_data: *mut std::ffi::c_void,
    /// Sample indices tensors pointer (`const int32_t*` or `const void*`).
    pub sample_indices: *const std::ffi::c_void,
    /// Embedding indices tensors pointer.
    pub embedding_indices: *const std::ffi::c_void,
    /// Aggregation weights tensors pointer.
    pub aggregation_weights: *const std::ffi::c_void,
    /// Packed mode override / fixed batch state / sample count metadata (`0` defaults to `4` or `8`).
    pub num_samples: usize,
    /// Optional status pointer (with local device ordinal encoded in `priv_data` or `0`).
    pub status: *mut TpuStatus,
}

/// Parameter structure for `TpuEmbeddingEngine_*Computation` entrypoints (`96` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TpuEmbeddingEngine_Computation_Params {
    /// Size of this struct (`96`).
    pub struct_size: usize,
    /// Reserved private pointer (`NULL`).
    pub priv_data: *mut std::ffi::c_void,
    /// `TpuEmbeddingConfigurationProto` bytes pointer.
    pub tpu_embedding_config_bytes: *const c_char,
    /// `TpuEmbeddingConfigurationProto` byte length.
    pub tpu_embedding_config_size: usize,
    /// Deduplication tuple mask bytes pointer.
    pub deduplication_tuple_mask_bytes: *const c_char,
    /// Deduplication tuple mask byte length.
    pub deduplication_tuple_mask_size: usize,
    /// HBM buffers config bytes pointer.
    pub hbm_buffers_config_bytes: *const c_char,
    /// HBM buffers config byte length.
    pub hbm_buffers_config_size: usize,
    /// TPU topology bytes pointer.
    pub tpu_topology_bytes: *const c_char,
    /// TPU topology byte length.
    pub tpu_topology_size: usize,
    /// Output serialized XLA computation proto pointer.
    pub xla_computation_out: *mut TpuSerializedProto,
    /// Optional status pointer.
    pub status: *mut TpuStatus,
}

const _: () = assert!(size_of::<TFNPD_PluginParams>() == 40);
const _: () = assert!(std::mem::offset_of!(TFNPD_PluginParams, device_type) == 0x10);
const _: () = assert!(std::mem::offset_of!(TFNPD_PluginParams, priority) == 0x20);
const _: () = assert!(std::mem::offset_of!(TFNPD_PluginParams, is_pluggable_device) == 0x24);
const _: () = assert!(std::mem::offset_of!(TFNPD_PluginParams, use_pjrt_on_demand_compile) == 0x25);
const _: () = assert!(size_of::<TpuPartitionedCall_Params>() == 24);
const _: () = assert!(size_of::<XLA_Shape>() == 96);
const _: () = assert!(std::mem::offset_of!(XLA_Shape, minor_to_major) == 0x10);
const _: () = assert!(std::mem::offset_of!(XLA_Shape, tile_dims) == 0x50);
const _: () = assert!(size_of::<TpuSerializedProto>() == 16);
const _: () = assert!(size_of::<TpuEmbeddingEngine_ExecutePartitioner_Params>() == 56);
const _: () =
    assert!(std::mem::offset_of!(TpuEmbeddingEngine_ExecutePartitioner_Params, status) == 0x30);
const _: () = assert!(size_of::<TpuEmbeddingEngine_ConfigureMemory_Params>() == 64);
const _: () =
    assert!(std::mem::offset_of!(TpuEmbeddingEngine_ConfigureMemory_Params, status) == 0x38);
const _: () = assert!(size_of::<TpuEmbeddingEngine_CollateMemory_Params>() == 56);
const _: () =
    assert!(std::mem::offset_of!(TpuEmbeddingEngine_CollateMemory_Params, status) == 0x30);
const _: () = assert!(size_of::<TpuEmbeddingEngine_ConfigureHost_Params>() == 96);
const _: () =
    assert!(std::mem::offset_of!(TpuEmbeddingEngine_ConfigureHost_Params, status) == 0x58);
const _: () = assert!(size_of::<TpuEmbeddingEngine_ConnectHosts_Params>() == 40);
const _: () = assert!(std::mem::offset_of!(TpuEmbeddingEngine_ConnectHosts_Params, status) == 0x20);
const _: () = assert!(size_of::<TpuEmbeddingEngine_Finalize_Params>() == 64);
const _: () = assert!(std::mem::offset_of!(TpuEmbeddingEngine_Finalize_Params, status) == 0x38);
const _: () = assert!(size_of::<TpuEmbeddingEngine_IsInitialized_Params>() == 48);
const _: () =
    assert!(std::mem::offset_of!(TpuEmbeddingEngine_IsInitialized_Params, status) == 0x28);
const _: () = assert!(size_of::<TpuEmbeddingEngine_WriteParameters_Params>() == 72);
const _: () =
    assert!(std::mem::offset_of!(TpuEmbeddingEngine_WriteParameters_Params, status) == 0x40);
const _: () = assert!(size_of::<TpuEmbeddingEngine_EnqueueTensorBatch_Params>() == 56);
const _: () =
    assert!(std::mem::offset_of!(TpuEmbeddingEngine_EnqueueTensorBatch_Params, status) == 0x30);
const _: () = assert!(size_of::<TpuEmbeddingEngine_Computation_Params>() == 96);
const _: () = assert!(std::mem::offset_of!(TpuEmbeddingEngine_Computation_Params, status) == 0x58);

//! `#[repr(C)]` argument structures for `PJRT_Executable_*` and `PJRT_LoadedExecutable_*` entrypoints.

use crate::pjrt_types::{
    PJRT_Buffer, PJRT_Client, PJRT_Device, PJRT_Event, PJRT_Executable, PJRT_ExecuteOptions,
    PJRT_Extension_Base, PJRT_LoadedExecutable, PJRT_LogicalDeviceIds, PJRT_NamedValue,
    PJRT_Program, PJRT_SerializedExecutable, PjrtArgsStruct,
};
use std::os::raw::c_char;

/// Argument structure for `PJRT_Executable_Destroy` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Executable_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle to destroy.
    pub executable: *mut PJRT_Executable,
}

/// Argument structure for `PJRT_Executable_Name` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Executable_Name_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle.
    pub executable: *const PJRT_Executable,
    /// Output name string pointer.
    pub executable_name: *const c_char,
    /// Output name string length.
    pub executable_name_size: usize,
}

/// Argument structure for `PJRT_Executable_NumReplicas` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Executable_NumReplicas_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle.
    pub executable: *const PJRT_Executable,
    /// Output `num_replicas`.
    pub num_replicas: usize,
}

/// Argument structure for `PJRT_Executable_NumPartitions` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Executable_NumPartitions_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle.
    pub executable: *const PJRT_Executable,
    /// Output `num_partitions`.
    pub num_partitions: usize,
}

/// Argument structure for `PJRT_Executable_NumOutputs` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Executable_NumOutputs_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle.
    pub executable: *const PJRT_Executable,
    /// Output `num_outputs`.
    pub num_outputs: usize,
}

/// Argument structure for `PJRT_Executable_SizeOfGeneratedCodeInBytes` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Executable_SizeOfGeneratedCodeInBytes_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle.
    pub executable: *const PJRT_Executable,
    /// Output `size_in_bytes`.
    pub size_in_bytes: i64,
}

/// Argument structure for `PJRT_Executable_GetCostAnalysis` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Executable_GetCostAnalysis_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle.
    pub executable: *const PJRT_Executable,
    /// Output `PJRT_NamedValue` array pointer.
    pub properties: *const PJRT_NamedValue,
    /// Output number of properties.
    pub num_properties: usize,
}

/// Argument structure for `PJRT_Executable_OutputMemoryKinds` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Executable_OutputMemoryKinds_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle.
    pub executable: *const PJRT_Executable,
    /// Output `num_outputs`.
    pub num_outputs: usize,
    /// Output `memory_kinds` string pointer array.
    pub memory_kinds: *const *const c_char,
    /// Output `memory_kind_sizes` array.
    pub memory_kind_sizes: *const usize,
}

/// Argument structure for `PJRT_Executable_OptimizedProgram` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Executable_OptimizedProgram_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle.
    pub executable: *const PJRT_Executable,
    /// Output `PJRT_Program*`.
    pub program: *mut PJRT_Program,
}

/// Argument structure for `PJRT_Executable_Serialize` (`56` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Executable_Serialize_Args {
    /// Size of this struct (`56`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle.
    pub executable: *const PJRT_Executable,
    /// Output serialized bytes pointer.
    pub serialized_bytes: *const c_char,
    /// Output serialized byte count.
    pub serialized_bytes_size: usize,
    /// Output `PJRT_SerializedExecutable*`.
    pub serialized_executable: *mut PJRT_SerializedExecutable,
    /// Output deleter callback (`void (*)(PJRT_SerializedExecutable*)`).
    pub serialized_executable_deleter: Option<extern "C" fn(*mut std::ffi::c_void)>,
}

/// Argument structure for `PJRT_Executable_DeserializeAndLoad` (`64` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Executable_DeserializeAndLoad_Args {
    /// Size of this struct (`64`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Client handle.
    pub client: *mut PJRT_Client,
    /// Serialized executable bytes pointer.
    pub serialized_executable: *const c_char,
    /// Serialized executable byte count.
    pub serialized_executable_size: usize,
    /// Output `PJRT_LoadedExecutable*`.
    pub loaded_executable: *mut PJRT_LoadedExecutable,
    /// Optional overridden compile options pointer.
    pub overridden_serialized_compile_options: *const c_char,
    /// Byte length of `overridden_serialized_compile_options`.
    pub overridden_serialized_compile_options_size: usize,
}

/// Argument structure for `PJRT_Executable_OutputElementTypes` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Executable_OutputElementTypes_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle.
    pub executable: *const PJRT_Executable,
    /// Output `int32_t*` element types array.
    pub output_types: *const i32,
    /// Output number of output types.
    pub num_output_types: usize,
}

/// Argument structure for `PJRT_Executable_OutputDimensions` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Executable_OutputDimensions_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle.
    pub executable: *const PJRT_Executable,
    /// Output number of outputs.
    pub num_outputs: usize,
    /// Output flattened `const int64_t*` dimensions.
    pub dims: *const i64,
    /// Output `const size_t*` dimension sizes per output.
    pub dim_sizes: *const usize,
}

/// Argument structure for `PJRT_Executable_Fingerprint` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Executable_Fingerprint_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle.
    pub executable: *const PJRT_Executable,
    /// Output fingerprint string pointer.
    pub executable_fingerprint: *const c_char,
    /// Output fingerprint string length.
    pub executable_fingerprint_size: usize,
}

/// Argument structure for `PJRT_Executable_GetCompiledMemoryStats` (`112` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Executable_GetCompiledMemoryStats_Args {
    /// Size of this struct (`112`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle.
    pub executable: *const PJRT_Executable,
    /// Output `generated_code_size_in_bytes`.
    pub generated_code_size_in_bytes: i64,
    /// Output `argument_size_in_bytes`.
    pub argument_size_in_bytes: i64,
    /// Output `output_size_in_bytes`.
    pub output_size_in_bytes: i64,
    /// Output `alias_size_in_bytes`.
    pub alias_size_in_bytes: i64,
    /// Output `temp_size_in_bytes`.
    pub temp_size_in_bytes: i64,
    /// Output `host_generated_code_size_in_bytes`.
    pub host_generated_code_size_in_bytes: i64,
    /// Output `host_argument_size_in_bytes`.
    pub host_argument_size_in_bytes: i64,
    /// Output `host_output_size_in_bytes`.
    pub host_output_size_in_bytes: i64,
    /// Output `host_alias_size_in_bytes`.
    pub host_alias_size_in_bytes: i64,
    /// Output `host_temp_size_in_bytes`.
    pub host_temp_size_in_bytes: i64,
    /// Output `peak_memory_in_bytes`.
    pub peak_memory_in_bytes: i64,
}

/// Argument structure for `PJRT_Executable_GetCompileOptions` (`56` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Executable_GetCompileOptions_Args {
    /// Size of this struct (`56`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle.
    pub executable: *const PJRT_Executable,
    /// Output serialized compile options bytes pointer.
    pub serialized_bytes: *const c_char,
    /// Output serialized byte count.
    pub serialized_bytes_size: usize,
    /// Output serialized compile options handle.
    pub serialized_compile_options: *mut std::ffi::c_void,
    /// Output deleter callback (`void (*)(PJRT_SerializedCompileOptions*)`).
    pub serialized_compile_options_deleter: Option<extern "C" fn(*mut std::ffi::c_void)>,
}

/// Argument structure for `PJRT_Executable_ParameterMemoryKinds` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Executable_ParameterMemoryKinds_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Executable handle.
    pub executable: *const PJRT_Executable,
    /// Output `num_parameters`.
    pub num_parameters: usize,
    /// Output `memory_kinds` string array pointer.
    pub memory_kinds: *const *const c_char,
    /// Output `memory_kind_sizes` array pointer.
    pub memory_kind_sizes: *const usize,
}

/// Argument structure for `PJRT_LoadedExecutable_Destroy` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_LoadedExecutable_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Loaded executable handle to destroy.
    pub executable: *mut PJRT_LoadedExecutable,
}

/// Argument structure for `PJRT_LoadedExecutable_GetExecutable` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_LoadedExecutable_GetExecutable_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Loaded executable handle.
    pub loaded_executable: *const PJRT_LoadedExecutable,
    /// Output `PJRT_Executable*`.
    pub executable: *mut PJRT_Executable,
}

/// Argument structure for `PJRT_LoadedExecutable_AddressableDevices` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_LoadedExecutable_AddressableDevices_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Loaded executable handle.
    pub executable: *const PJRT_LoadedExecutable,
    /// Output `PJRT_Device*` array pointer.
    pub addressable_devices: *const *mut PJRT_Device,
    /// Output number of addressable devices.
    pub num_addressable_devices: usize,
}

/// Argument structure for `PJRT_LoadedExecutable_Delete` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_LoadedExecutable_Delete_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Loaded executable handle.
    pub executable: *mut PJRT_LoadedExecutable,
}

/// Argument structure for `PJRT_LoadedExecutable_IsDeleted` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_LoadedExecutable_IsDeleted_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Loaded executable handle.
    pub executable: *const PJRT_LoadedExecutable,
    /// Output `is_deleted` flag.
    pub is_deleted: bool,
    /// Explicit padding.
    pub pad: [u8; 7],
}

/// Argument structure for `PJRT_LoadedExecutable_Execute` (`80` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_LoadedExecutable_Execute_Args {
    /// Size of this struct (`80`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Loaded executable handle.
    pub executable: *mut PJRT_LoadedExecutable,
    /// Optional `PJRT_ExecuteOptions*`.
    pub options: *const PJRT_ExecuteOptions,
    /// Input `PJRT_Buffer* const* const*` argument lists.
    pub argument_lists: *const *const *const PJRT_Buffer,
    /// Number of devices (`usize`).
    pub num_devices: usize,
    /// Number of input arguments per device (`usize`).
    pub num_args: usize,
    /// Output `PJRT_Buffer** const*` lists.
    pub output_lists: *const *mut *mut PJRT_Buffer,
    /// Optional output `PJRT_Event**` device completion events array.
    pub device_complete_events: *mut *mut PJRT_Event,
    /// Optional single target `PJRT_Device*` for portable execution.
    pub execute_device: *mut PJRT_Device,
}

/// Argument structure for `PJRT_LoadedExecutable_Fingerprint` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_LoadedExecutable_Fingerprint_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Loaded executable handle.
    pub executable: *const PJRT_LoadedExecutable,
    /// Output fingerprint string pointer.
    pub executable_fingerprint: *const c_char,
    /// Output fingerprint string length.
    pub executable_fingerprint_size: usize,
}

/// Argument structure for `PJRT_LoadedExecutable_GetDeviceAssignment` (`56` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_LoadedExecutable_GetDeviceAssignment_Args {
    /// Size of this struct (`56`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Loaded executable handle.
    pub executable: *const PJRT_LoadedExecutable,
    /// Output serialized bytes pointer.
    pub serialized_bytes: *const c_char,
    /// Output serialized byte count.
    pub serialized_bytes_size: usize,
    /// Output serialized device assignment handle.
    pub serialized_device_assignment: *mut std::ffi::c_void,
    /// Output deleter callback (`void (*)(PJRT_SerializedDeviceAssignment*)`).
    pub serialized_device_assignment_deleter: Option<extern "C" fn(*mut std::ffi::c_void)>,
}

/// Argument structure for `PJRT_LoadedExecutable_AddressableDeviceLogicalIds` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_LoadedExecutable_AddressableDeviceLogicalIds_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Loaded executable handle.
    pub executable: *const PJRT_LoadedExecutable,
    /// Output `PJRT_LogicalDeviceIds*` array.
    pub addressable_device_logical_ids: *const PJRT_LogicalDeviceIds,
    /// Output number of logical IDs.
    pub num_addressable_device_logical_ids: usize,
}

impl PjrtArgsStruct for PJRT_Executable_Destroy_Args {}
impl PjrtArgsStruct for PJRT_Executable_Name_Args {}
impl PjrtArgsStruct for PJRT_Executable_NumReplicas_Args {}
impl PjrtArgsStruct for PJRT_Executable_NumPartitions_Args {}
impl PjrtArgsStruct for PJRT_Executable_NumOutputs_Args {}
impl PjrtArgsStruct for PJRT_Executable_SizeOfGeneratedCodeInBytes_Args {}
impl PjrtArgsStruct for PJRT_Executable_GetCostAnalysis_Args {}
impl PjrtArgsStruct for PJRT_Executable_OutputMemoryKinds_Args {}
impl PjrtArgsStruct for PJRT_Executable_OptimizedProgram_Args {}
impl PjrtArgsStruct for PJRT_Executable_Serialize_Args {}
impl PjrtArgsStruct for PJRT_Executable_DeserializeAndLoad_Args {}
impl PjrtArgsStruct for PJRT_Executable_OutputElementTypes_Args {}
impl PjrtArgsStruct for PJRT_Executable_OutputDimensions_Args {}
impl PjrtArgsStruct for PJRT_Executable_Fingerprint_Args {}
impl PjrtArgsStruct for PJRT_Executable_GetCompiledMemoryStats_Args {}
impl PjrtArgsStruct for PJRT_Executable_GetCompileOptions_Args {}
impl PjrtArgsStruct for PJRT_Executable_ParameterMemoryKinds_Args {}
impl PjrtArgsStruct for PJRT_LoadedExecutable_Destroy_Args {}
impl PjrtArgsStruct for PJRT_LoadedExecutable_GetExecutable_Args {}
impl PjrtArgsStruct for PJRT_LoadedExecutable_AddressableDevices_Args {}
impl PjrtArgsStruct for PJRT_LoadedExecutable_Delete_Args {}
impl PjrtArgsStruct for PJRT_LoadedExecutable_IsDeleted_Args {
    const MIN_STRUCT_SIZE: usize = 25;
}
impl PjrtArgsStruct for PJRT_LoadedExecutable_Execute_Args {}
impl PjrtArgsStruct for PJRT_LoadedExecutable_Fingerprint_Args {}
impl PjrtArgsStruct for PJRT_LoadedExecutable_GetDeviceAssignment_Args {}
impl PjrtArgsStruct for PJRT_LoadedExecutable_AddressableDeviceLogicalIds_Args {}

const _: () = assert!(std::mem::offset_of!(PJRT_LoadedExecutable_Execute_Args, executable) == 0x10);
const _: () = assert!(std::mem::offset_of!(PJRT_LoadedExecutable_Execute_Args, options) == 0x18);
const _: () =
    assert!(std::mem::offset_of!(PJRT_LoadedExecutable_Execute_Args, argument_lists) == 0x20);
const _: () =
    assert!(std::mem::offset_of!(PJRT_LoadedExecutable_Execute_Args, num_devices) == 0x28);
const _: () = assert!(std::mem::offset_of!(PJRT_LoadedExecutable_Execute_Args, num_args) == 0x30);
const _: () =
    assert!(std::mem::offset_of!(PJRT_LoadedExecutable_Execute_Args, output_lists) == 0x38);
const _: () = assert!(
    std::mem::offset_of!(PJRT_LoadedExecutable_Execute_Args, device_complete_events) == 0x40
);
const _: () =
    assert!(std::mem::offset_of!(PJRT_LoadedExecutable_Execute_Args, execute_device) == 0x48);

const _: () = assert!(size_of::<PJRT_Executable_Destroy_Args>() == 24);
const _: () = assert!(size_of::<PJRT_Executable_Name_Args>() == 40);
const _: () = assert!(size_of::<PJRT_Executable_NumReplicas_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Executable_NumPartitions_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Executable_NumOutputs_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Executable_SizeOfGeneratedCodeInBytes_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Executable_GetCostAnalysis_Args>() == 40);
const _: () = assert!(size_of::<PJRT_Executable_OutputMemoryKinds_Args>() == 48);
const _: () = assert!(size_of::<PJRT_Executable_OptimizedProgram_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Executable_Serialize_Args>() == 56);
const _: () = assert!(size_of::<PJRT_Executable_DeserializeAndLoad_Args>() == 64);
const _: () = assert!(size_of::<PJRT_Executable_OutputElementTypes_Args>() == 40);
const _: () = assert!(size_of::<PJRT_Executable_OutputDimensions_Args>() == 48);
const _: () = assert!(size_of::<PJRT_Executable_Fingerprint_Args>() == 40);
const _: () = assert!(size_of::<PJRT_Executable_GetCompiledMemoryStats_Args>() == 112);
const _: () = assert!(size_of::<PJRT_Executable_GetCompileOptions_Args>() == 56);
const _: () = assert!(size_of::<PJRT_Executable_ParameterMemoryKinds_Args>() == 48);
const _: () = assert!(size_of::<PJRT_LoadedExecutable_Destroy_Args>() == 24);
const _: () = assert!(size_of::<PJRT_LoadedExecutable_GetExecutable_Args>() == 32);
const _: () = assert!(size_of::<PJRT_LoadedExecutable_AddressableDevices_Args>() == 40);
const _: () = assert!(size_of::<PJRT_LoadedExecutable_Delete_Args>() == 24);
const _: () = assert!(size_of::<PJRT_LoadedExecutable_IsDeleted_Args>() == 32);
const _: () = assert!(size_of::<PJRT_LoadedExecutable_Execute_Args>() == 80);
const _: () = assert!(size_of::<PJRT_LoadedExecutable_Fingerprint_Args>() == 40);
const _: () = assert!(size_of::<PJRT_LoadedExecutable_GetDeviceAssignment_Args>() == 56);
const _: () = assert!(size_of::<PJRT_LoadedExecutable_AddressableDeviceLogicalIds_Args>() == 40);

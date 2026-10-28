//! C ABI drop-in replacement for `libtpu.so`.
//!
//! Exports `GetPjrtApi`, `GetLibtpuSdkApi`, `TfTpu_Initialize`, `TpuStatus_*`,
//! `TpuTopology_*`, `TpuCoreLocation_*`, `TpuHostLocation_*`, `TpuMeshState_*`,
//! `TpuProfiler_*`, `TfTpuOrdinalSelector_*`, `TpuEmbeddingEngine_*`, `TpuDriver_*`,
//! and clean `libtpu_rs_*` entrypoints for C and C++20 consumers.

#![cfg_attr(all(test, coverage_nightly), feature(coverage_attribute))]
#![allow(
    non_camel_case_types,
    non_snake_case,
    unused_qualifications,
    clippy::not_unsafe_ptr_arg_deref,
    clippy::unnecessary_wraps,
    clippy::wildcard_imports,
    clippy::missing_const_for_fn,
    clippy::significant_drop_tightening,
    clippy::if_not_else
)]

pub mod extensions;
/// Legacy TensorFlow TPU C API entrypoints.
pub mod legacy_tf_api;
/// PJRT buffer, copy-to-device stream, and async transfer manager C API entrypoints.
pub mod pjrt_buffer_api;
/// PJRT client, topology description, and execution context C API entrypoints.
pub mod pjrt_client_api;
/// PJRT device, device description, and memory space C API entrypoints.
pub mod pjrt_device_api;
/// PJRT error, plugin initialization, and event C API entrypoints.
pub mod pjrt_error_api;
/// PJRT executable and loaded executable C API entrypoints.
pub mod pjrt_executable_api;
/// Shared PJRT and `LibtpuSdk` FFI dispatchers and helpers.
pub mod pjrt_helpers;
pub mod pjrt_types;
/// `PJRT_Api` static vtable assembly and `GetPjrtApi` entrypoint.
pub mod pjrt_vtable;
/// Ergonomic `libtpu_rs_*` C/C++/Python FFI helper entrypoints.
pub mod rust_ergonomic_api;
/// `LibtpuSdk_Api` (`tpumonitoring`, `tpuz`, `debug`, `slice`) C API entrypoints.
pub mod sdk_api;
pub mod tpu_driver_types;

use crate::pjrt_types::{
    PJRT_Buffer, PJRT_Client, PJRT_Client_AddressableMemories_Args, PJRT_Client_Compile_Args,
    PJRT_Client_Create_Args, PJRT_Client_Destroy_Args, PJRT_Client_Devices_Args,
    PJRT_Client_LookupDevice_Args, PJRT_Client_PlatformStr_Args, PJRT_Client_ProcessIndex_Args,
    PJRT_Device, PJRT_DeviceDescription, PJRT_Error, PJRT_Error_Destroy_Args,
    PJRT_Error_GetCode_Args, PJRT_Error_Message_Args, PJRT_Executable, PJRT_LoadedExecutable,
    PJRT_Memory, PJRT_Plugin_Initialize_Args, PJRT_TopologyDescription,
};
use crate::tpu_driver_types::{
    LibTpuCStatus, LibtpuSdk_Api, LibtpuSdk_ChipLocationToCoordinates, LibtpuSdk_Client,
    LibtpuSdk_Client_Args, LibtpuSdk_Error, LibtpuSdk_Error_Destroy_Args,
    LibtpuSdk_Error_GetCode_Args, LibtpuSdk_Error_GetMessage_Args,
    LibtpuSdk_GetCartesianCoordinates_Args, LibtpuSdk_GetChipCoordinates_Args,
    LibtpuSdk_GetChipIndex_Args, LibtpuSdk_GetHostName_Args, LibtpuSdk_GetMetric_Args,
    LibtpuSdk_GetMetricDescription_Args, LibtpuSdk_GetMetricValues_Args,
    LibtpuSdk_GetRuntimeStatus_Args, LibtpuSdk_Metric, LibtpuSdk_RegisterDebugCallback_Args,
    LibtpuSdk_RuntimeStatus, LibtpuSdk_RuntimeStatus_Destroy_Args,
    LibtpuSdk_RuntimeStatus_GetCoreStateSummary_Args, LibtpuSdk_UnregisterDebugCallback_Args,
    TFNPD_PluginParams, TfTpuOrdinalSelector, TpuCoreLocation, TpuDriver, TpuDriverFn,
    TpuEmbeddingEngine_CollateMemory_Params, TpuEmbeddingEngine_Computation_Params,
    TpuEmbeddingEngine_ConfigureHost_Params, TpuEmbeddingEngine_ConfigureMemory_Params,
    TpuEmbeddingEngine_ConnectHosts_Params, TpuEmbeddingEngine_EnqueueTensorBatch_Params,
    TpuEmbeddingEngine_ExecutePartitioner_Params, TpuEmbeddingEngine_Finalize_Params,
    TpuEmbeddingEngine_IsInitialized_Params, TpuEmbeddingEngine_WriteParameters_Params,
    TpuEmbeddingEngineState, TpuHostLocation, TpuMeshState, TpuPartitionedCall_Params, TpuProfiler,
    TpuSerializedProto, TpuStatus, TpuTopology, XLA_Shape,
};
use libtpu_core::config::TpuInitConfig;
use libtpu_core::driver::{
    LibtpuSdkCoreClient, TfTpuCoreOrdinalSelector, TpuCoreDriver, TpuCoreEmbeddingEngine,
    TpuCoreProfiler, TpuCoreStatus,
};
use libtpu_core::error::LibTpuError;
use libtpu_core::hal::SimulatedTpuHal;
use libtpu_core::pjrt::PjrtCoreClient;
use libtpu_core::types::{CallbackId, DeviceId, ElementType, Shape};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;
use std::sync::OnceLock;

pub use legacy_tf_api::*;
pub use pjrt_buffer_api::*;
pub use pjrt_client_api::*;
pub use pjrt_device_api::*;
pub use pjrt_error_api::*;
pub use pjrt_executable_api::*;
pub use pjrt_helpers::*;
pub use pjrt_vtable::*;
pub use rust_ergonomic_api::*;
pub use sdk_api::*;

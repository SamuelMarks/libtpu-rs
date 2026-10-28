//! Unified error types and status codes for `libtpu-rs`.
//!
//! This module defines the single workspace-wide error enumeration [`LibTpuError`]
//! powered by `derive_more`, along with strongly-typed PJRT, Abseil, and `LibtpuSdk`
//! status codes ([`PjrtErrorCode`], [`AbseilStatusCode`], and [`LibtpuSdkErrorCode`]).

use derive_more::{Display, Error, From};
use num_enum::{FromPrimitive, IntoPrimitive};

/// Standard PJRT and Abseil status error codes matching `PJRT_Error_Code` and `TpuStatus`.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Display, FromPrimitive, IntoPrimitive)]
pub enum PjrtErrorCode {
    /// Operation completed successfully (`0`).
    #[display("OK")]
    Ok = 0,
    /// The operation was cancelled (`1`).
    #[display("CANCELLED")]
    Cancelled = 1,
    /// Unknown error (`2`).
    #[display("UNKNOWN")]
    #[num_enum(default)]
    Unknown = 2,
    /// Client specified an invalid argument (`3`).
    #[display("INVALID_ARGUMENT")]
    InvalidArgument = 3,
    /// Deadline expired before operation could complete (`4`).
    #[display("DEADLINE_EXCEEDED")]
    DeadlineExceeded = 4,
    /// Some requested entity (e.g., device or memory space) was not found (`5`).
    #[display("NOT_FOUND")]
    NotFound = 5,
    /// Some entity that we attempted to create already exists (`6`).
    #[display("ALREADY_EXISTS")]
    AlreadyExists = 6,
    /// The caller does not have permission to execute the specified operation (`7`).
    #[display("PERMISSION_DENIED")]
    PermissionDenied = 7,
    /// Some resource has been exhausted, such as HBM device memory (`8`).
    #[display("RESOURCE_EXHAUSTED")]
    ResourceExhausted = 8,
    /// Operation was rejected because the system is not in a required state (`9`).
    #[display("FAILED_PRECONDITION")]
    FailedPrecondition = 9,
    /// The operation was aborted (`10`).
    #[display("ABORTED")]
    Aborted = 10,
    /// Operation was attempted past the valid range (`11`).
    #[display("OUT_OF_RANGE")]
    OutOfRange = 11,
    /// Operation is not implemented or not supported (`12`).
    #[display("UNIMPLEMENTED")]
    Unimplemented = 12,
    /// Internal invariant violation (`13`).
    #[display("INTERNAL")]
    Internal = 13,
    /// The service or hardware device is currently unavailable (`14`).
    #[display("UNAVAILABLE")]
    Unavailable = 14,
    /// Unrecoverable data loss or corruption (`15`).
    #[display("DATA_LOSS")]
    DataLoss = 15,
    /// The request does not have valid authentication credentials (`16`).
    #[display("UNAUTHENTICATED")]
    Unauthenticated = 16,
}

impl PjrtErrorCode {
    /// Converts a raw integer code into a strongly-typed [`PjrtErrorCode`].
    ///
    /// # Arguments
    ///
    /// * `raw` - The raw 32-bit signed integer status code from C FFI.
    #[must_use]
    pub const fn from_raw(raw: i32) -> Self {
        match raw {
            0 => Self::Ok,
            1 => Self::Cancelled,
            3 => Self::InvalidArgument,
            4 => Self::DeadlineExceeded,
            5 => Self::NotFound,
            6 => Self::AlreadyExists,
            7 => Self::PermissionDenied,
            8 => Self::ResourceExhausted,
            9 => Self::FailedPrecondition,
            10 => Self::Aborted,
            11 => Self::OutOfRange,
            12 => Self::Unimplemented,
            13 => Self::Internal,
            14 => Self::Unavailable,
            15 => Self::DataLoss,
            16 => Self::Unauthenticated,
            _ => Self::Unknown,
        }
    }

    /// Returns the raw 32-bit signed integer representation of this [`PjrtErrorCode`].
    #[must_use]
    pub const fn as_raw(self) -> i32 {
        self as i32
    }
}

/// Abseil status code type alias (`absl::StatusCode`), matching `PjrtErrorCode` integer discriminants.
pub type AbseilStatusCode = PjrtErrorCode;

/// Strongly-typed `LibtpuSdk` status code returned by `LibtpuSdk_Error_GetCode`.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Display, FromPrimitive, IntoPrimitive)]
pub enum LibtpuSdkErrorCode {
    /// Operation succeeded (`0`).
    #[display("SDK_OK")]
    Ok = 0,
    /// Invalid argument passed to SDK function (`3`).
    #[display("SDK_INVALID_ARGUMENT")]
    InvalidArgument = 3,
    /// Requested telemetry metric or callback ID was not found (`5`).
    #[display("SDK_NOT_FOUND")]
    NotFound = 5,
    /// SDK operation failed precondition (`9`).
    #[display("SDK_FAILED_PRECONDITION")]
    FailedPrecondition = 9,
    /// Internal SDK or telemetry error (`13`).
    #[display("SDK_INTERNAL")]
    #[num_enum(default)]
    Internal = 13,
    /// Hardware telemetry service is unavailable (`14`).
    #[display("SDK_UNAVAILABLE")]
    Unavailable = 14,
}

impl LibtpuSdkErrorCode {
    /// Converts a [`PjrtErrorCode`] into the corresponding [`LibtpuSdkErrorCode`].
    ///
    /// # Arguments
    ///
    /// * `code` - The [`PjrtErrorCode`] to convert.
    #[must_use]
    pub const fn from_pjrt_code(code: PjrtErrorCode) -> Self {
        match code {
            PjrtErrorCode::Ok => Self::Ok,
            PjrtErrorCode::InvalidArgument | PjrtErrorCode::OutOfRange => Self::InvalidArgument,
            PjrtErrorCode::NotFound => Self::NotFound,
            PjrtErrorCode::FailedPrecondition | PjrtErrorCode::AlreadyExists => {
                Self::FailedPrecondition
            }
            PjrtErrorCode::Unavailable => Self::Unavailable,
            _ => Self::Internal,
        }
    }

    /// Returns the raw 32-bit integer discriminant for `LibtpuSdk_Error_GetCode`.
    #[must_use]
    pub const fn as_raw(self) -> i32 {
        self as i32
    }
}

/// Type alias for [`LibtpuSdkErrorCode`] (`SdkStatusCode`).
pub type SdkStatusCode = LibtpuSdkErrorCode;

/// Type alias for [`LibtpuSdkErrorCode`] (`SdkErrorCode`).
pub type SdkErrorCode = LibtpuSdkErrorCode;

/// Wrapper struct for UTF-8 conversion errors to enable automatic `From` conversion into [`LibTpuError`].
#[derive(Debug, Clone, PartialEq, Eq, Display, Error, From)]
#[display("UTF-8 conversion failed: {_0}")]
pub struct Utf8ErrorWrapper(#[error(not(source))] pub std::str::Utf8Error);

/// Wrapper struct for `prost` protobuf wire decode errors to enable `From` conversion into [`LibTpuError`].
#[derive(Debug, Clone, PartialEq, Eq, Display, Error, From)]
#[display("Prost decode error: {_0}")]
pub struct ProstDecodeErrorWrapper(#[error(not(source))] pub prost::DecodeError);

/// Wrapper struct for `prost` protobuf wire encode errors to enable `From` conversion into [`LibTpuError`].
#[derive(Debug, Clone, PartialEq, Eq, Display, Error, From)]
#[display("Prost encode error: {_0}")]
pub struct ProstEncodeErrorWrapper(#[error(not(source))] pub prost::EncodeError);

/// Wrapper struct for `prost_reflect::DescriptorError` reflection errors to enable `From` conversion into [`LibTpuError`].
#[derive(Debug, Clone, PartialEq, Eq, Display, Error)]
#[display("Prost reflect descriptor error: {_0}")]
pub struct ProstReflectErrorWrapper(#[error(not(source))] pub String);

impl From<prost_reflect::DescriptorError> for ProstReflectErrorWrapper {
    fn from(err: prost_reflect::DescriptorError) -> Self {
        Self(err.to_string())
    }
}

/// Wrapper struct for `goblin` binary object parsing errors to enable `From` conversion into [`LibTpuError`].
#[derive(Debug, Clone, PartialEq, Eq, Display, Error)]
#[display("Goblin binary parse error: {_0}")]
pub struct GoblinErrorWrapper(#[error(not(source))] pub String);

impl From<goblin::error::Error> for GoblinErrorWrapper {
    fn from(err: goblin::error::Error) -> Self {
        Self(err.to_string())
    }
}

/// Wrapper struct for `lexopt` CLI flag parsing errors to enable `From` conversion into [`LibTpuError`].
#[derive(Debug, Clone, PartialEq, Eq, Display, Error)]
#[display("CLI flag parse error: {_0}")]
pub struct LexoptErrorWrapper(#[error(not(source))] pub String);

impl From<lexopt::Error> for LexoptErrorWrapper {
    fn from(err: lexopt::Error) -> Self {
        Self(err.to_string())
    }
}

/// Wrapper struct for `object::read::Error` binary parsing errors to enable `From` conversion into [`LibTpuError`].
#[derive(Debug, Clone, PartialEq, Eq, Display, Error)]
#[display("Object binary parse error: {_0}")]
pub struct ObjectErrorWrapper(#[error(not(source))] pub String);

impl From<object::read::Error> for ObjectErrorWrapper {
    fn from(err: object::read::Error) -> Self {
        Self(err.to_string())
    }
}

/// Wrapper struct for `gimli::Error` DWARF `.eh_frame` parsing errors to enable `From` conversion into [`LibTpuError`].
#[derive(Debug, Clone, PartialEq, Eq, Display, Error, From)]
#[display("Gimli DWARF parse error: {_0}")]
pub struct GimliErrorWrapper(#[error(not(source))] pub gimli::Error);

/// Wrapper struct for `cpp_demangle::error::Error` C++ symbol demangling errors to enable `From` conversion into [`LibTpuError`].
#[derive(Debug, Clone, PartialEq, Eq, Display, Error, From)]
#[display("C++ symbol demangle error: {_0}")]
pub struct CppDemangleErrorWrapper(#[error(not(source))] pub cpp_demangle::error::Error);

/// Wrapper struct for `std::io::Error` I/O errors to enable `From` conversion into [`LibTpuError`].
#[derive(Debug, Clone, PartialEq, Eq, Display, Error)]
#[display("I/O error (errno={errno:?}): {message}")]
pub struct IoErrorWrapper {
    /// Optional OS error code (`errno`).
    pub errno: Option<i32>,
    /// Human-readable I/O error message.
    pub message: String,
}

impl From<std::io::Error> for IoErrorWrapper {
    fn from(err: std::io::Error) -> Self {
        Self {
            errno: err.raw_os_error(),
            message: err.to_string(),
        }
    }
}

/// Wrapper struct for `rustix::io::Errno` POSIX syscall errors to enable `From` conversion into [`LibTpuError`].
#[derive(Debug, Clone, PartialEq, Eq, Display, Error, From)]
#[display("POSIX syscall error: {_0}")]
pub struct RustixErrnoWrapper(#[error(not(source))] pub rustix::io::Errno);

/// Structured diagnostic payload attached to a [`LibTpuError`] for `PJRT_Error_ForEachPayload`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorPayload {
    /// Diagnostic payload key (typically a type URL such as `"type.googleapis.com/xla.TpuErrorPayload"`).
    pub key: String,
    /// Raw serialized diagnostic payload bytes.
    pub value: Vec<u8>,
}

impl ErrorPayload {
    /// Creates a new [`ErrorPayload`] key-value entry.
    ///
    /// # Arguments
    ///
    /// * `key` - Diagnostic payload key string.
    /// * `value` - Serialized payload byte vector.
    #[must_use]
    pub const fn new(key: String, value: Vec<u8>) -> Self {
        Self { key, value }
    }
}

/// Single unified error enum for the entire `libtpu-rs` workspace.
///
/// Uses `derive_more` (`Display`, `Error`, `From`) without `anyhow` or `thiserror`.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Display, Error, From)]
pub enum LibTpuError {
    /// A null pointer was passed across an FFI boundary.
    #[display("Null pointer passed for parameter '{parameter}'")]
    #[from(ignore)]
    NullPointer {
        /// Name of the parameter that was null.
        parameter: &'static str,
    },

    /// A PJRT or SDK argument struct had a `struct_size` smaller than the minimum required size.
    #[display(
        "Invalid struct_size for {struct_name}: expected at least {expected_min}, got {actual}"
    )]
    #[from(ignore)]
    InvalidStructSize {
        /// Name of the C ABI structure.
        struct_name: &'static str,
        /// Minimum required byte size.
        expected_min: usize,
        /// Actual byte size provided by caller.
        actual: usize,
    },

    /// A C ABI struct size did not match the expected layout size (`StructSizeMismatch`).
    #[display("Struct size mismatch for {struct_name}: expected {expected}, got {actual}")]
    #[from(ignore)]
    StructSizeMismatch {
        /// Name of the C ABI structure.
        struct_name: &'static str,
        /// Expected byte size.
        expected: usize,
        /// Actual byte size provided by caller.
        actual: usize,
    },

    /// An invalid argument was supplied to a function or FFI entrypoint.
    #[display("Invalid argument '{parameter}': {reason}")]
    #[from(ignore)]
    InvalidArgument {
        /// Name of the invalid parameter.
        parameter: &'static str,
        /// Detailed explanation of why the argument is invalid.
        reason: String,
    },

    /// Element type integer does not map to a valid `PJRT_Buffer_Type` (`InvalidElementType`).
    #[display("Invalid element type code: {raw_type}")]
    #[from(ignore)]
    InvalidElementType {
        /// Raw integer element type code.
        raw_type: i32,
    },

    /// A tensor dimension extent was negative or out of bounds (`InvalidDimension`).
    #[display("Invalid dimension at axis {index}: {value}")]
    #[from(ignore)]
    InvalidDimension {
        /// Axis index.
        index: usize,
        /// Invalid dimension value.
        value: i64,
    },

    /// Memory space kind string is not recognized by the TPU runtime (`InvalidMemoryKind`).
    #[display("Invalid TPU memory kind '{kind}'")]
    #[from(ignore)]
    InvalidMemoryKind {
        /// Unrecognized memory kind string.
        kind: String,
    },

    /// Host DMA buffer address is not aligned to the required hardware page boundary (`DmaAlignmentError`).
    #[display(
        "DMA address {address:#x} is not aligned to required {required_alignment}-byte boundary"
    )]
    #[from(ignore)]
    DmaAlignmentError {
        /// Unaligned host address.
        address: u64,
        /// Required alignment in bytes.
        required_alignment: usize,
    },

    /// Requested host DMA mapping was not found in the active IOMMU table (`DmaMappingNotFound`).
    #[display("DMA mapping not found for host address {host_address:#x}")]
    #[from(ignore)]
    DmaMappingNotFound {
        /// Host virtual address that was not mapped.
        host_address: u64,
    },

    /// Attempted to fulfill a `PJRT_Buffer` alias that was already fulfilled (`AliasAlreadyFulfilled`).
    #[display("Alias buffer {buffer_id} has already been fulfilled")]
    #[from(ignore)]
    AliasAlreadyFulfilled {
        /// Identifier of the alias buffer.
        buffer_id: u64,
    },

    /// Linux VFIO IOMMU container or group operation failed (`VfioIommuError`).
    #[display("VFIO IOMMU error on group {group_id}: {reason}")]
    #[from(ignore)]
    VfioIommuError {
        /// IOMMU group number.
        group_id: u32,
        /// Detailed failure reason.
        reason: String,
    },

    /// Environment variable or `LIBTPU_INIT_ARGS` flag failed to parse (`EnvironmentParseError`).
    #[display("Failed to parse environment variable {variable}='{value}': {reason}")]
    #[from(ignore)]
    EnvironmentParseError {
        /// Environment variable or flag name.
        variable: String,
        /// Raw value string.
        value: String,
        /// Parse error explanation.
        reason: String,
    },

    /// Cross-replica or cross-slice collective operation failed (`CollectiveError`).
    #[display(
        "Collective operation '{op_name}' failed on communicator {communicator_id}: {reason}"
    )]
    #[from(ignore)]
    CollectiveError {
        /// Communicator identifier.
        communicator_id: u64,
        /// Name of the collective operation.
        op_name: &'static str,
        /// Failure explanation.
        reason: String,
    },

    /// Megascale multi-slice barrier timed out (`MegascaleBarrierTimeout`).
    #[display("Megascale barrier {barrier_id} timed out on slice {slice_id}")]
    #[from(ignore)]
    MegascaleBarrierTimeout {
        /// Slice identifier.
        slice_id: i32,
        /// Barrier identifier.
        barrier_id: u64,
    },

    /// Profiler session was in an unexpected state (`ProfilerStateError`).
    #[display("Profiler state error: expected {expected_state}, actual {actual_state}")]
    #[from(ignore)]
    ProfilerStateError {
        /// Expected state name.
        expected_state: &'static str,
        /// Actual state name.
        actual_state: &'static str,
    },

    /// Internal runtime invariant violation (`InternalError`).
    #[display("Internal TPU runtime error: {context}")]
    #[from(ignore)]
    InternalError {
        /// Internal error context description.
        context: String,
    },

    /// Integer conversion or checked arithmetic overflowed.
    #[display("Numeric overflow in {context}")]
    #[from(ignore)]
    NumericOverflow {
        /// Context where the numeric overflow occurred.
        context: &'static str,
    },

    /// The requested PJRT API version is incompatible with the runtime.
    #[display(
        "Unsupported API version {requested_major}.{requested_minor} (supported: {supported_major}.{supported_minor})"
    )]
    #[from(ignore)]
    UnsupportedApiVersion {
        /// Requested major version.
        requested_major: i32,
        /// Requested minor version.
        requested_minor: i32,
        /// Supported major version.
        supported_major: i32,
        /// Supported minor version.
        supported_minor: i32,
    },

    /// A requested TPU device ID was not found in the topology.
    #[display("TPU device {device_id} not found")]
    #[from(ignore)]
    DeviceNotFound {
        /// Identifier of the missing device.
        device_id: u32,
    },

    /// A TPU device is not addressable by the current host process.
    #[display("TPU device {device_id} is not addressable by process {process_index}")]
    #[from(ignore)]
    DeviceNotAddressable {
        /// Identifier of the non-addressable device.
        device_id: u32,
        /// Process index of the current client.
        process_index: i32,
    },

    /// Execution on a TPU device was poisoned by `PJRT_Device_PoisonExecution`.
    #[display("TPU device {device_id} execution poisoned at launch {launch_id}: {reason}")]
    #[from(ignore)]
    DevicePoisoned {
        /// Identifier of the poisoned device.
        device_id: u32,
        /// Launch ID that was poisoned.
        launch_id: i32,
        /// Poisoning reason message.
        reason: String,
    },

    /// A requested TPU chip at `(x, y, z)` was not found in the mesh topology.
    #[display("TPU chip at ({x}, {y}, {z}) not found in topology")]
    #[from(ignore)]
    ChipNotFound {
        /// X coordinate.
        x: u32,
        /// Y coordinate.
        y: u32,
        /// Z coordinate.
        z: u32,
    },

    /// A requested TPU host ID was not found in the topology.
    #[display("TPU host {host_id} not found in topology")]
    #[from(ignore)]
    HostNotFound {
        /// Identifier of the missing host.
        host_id: u32,
    },

    /// A requested memory space ID was not found.
    #[display("Memory space {memory_id} not found")]
    #[from(ignore)]
    MemorySpaceNotFound {
        /// Identifier of the missing memory space.
        memory_id: i32,
    },

    /// Device HBM or pinned host memory allocation exceeded available capacity.
    #[display(
        "Out of TPU memory: requested {requested_bytes} bytes, available {available_bytes} bytes"
    )]
    #[from(ignore)]
    OutOfMemory {
        /// Number of bytes requested.
        requested_bytes: usize,
        /// Number of bytes currently available.
        available_bytes: usize,
    },

    /// Host virtual memory DMA mapping failed.
    #[display("DMA mapping failed for host address {host_address:#x} (size={size}): {reason}")]
    #[from(ignore)]
    DmaMappingFailed {
        /// Host virtual address.
        host_address: u64,
        /// Mapping byte size.
        size: usize,
        /// Failure reason.
        reason: String,
    },

    /// Operation attempted on a buffer that has already been deleted or donated.
    #[display("Buffer {buffer_id} has been deleted")]
    #[from(ignore)]
    BufferDeleted {
        /// Identifier of the deleted buffer.
        buffer_id: u64,
    },

    /// Operation attempted on an executable that has already been deleted.
    #[display("Executable {executable_id} has been deleted")]
    #[from(ignore)]
    ExecutableDeleted {
        /// Identifier of the deleted executable.
        executable_id: u64,
    },

    /// Tensor shape or layout specification is invalid.
    #[display("Invalid tensor shape: {reason}")]
    #[from(ignore)]
    InvalidShape {
        /// Detailed reason for the shape invalidity.
        reason: String,
    },

    /// Element type integer does not map to a valid `PJRT_Buffer_Type`.
    #[display("Unsupported element type code: {raw_type}")]
    #[from(ignore)]
    UnsupportedElementType {
        /// Raw integer element type code.
        raw_type: i32,
    },

    /// Binary protobuf wire decoding failed.
    #[display("Protobuf decode error in {message_type}: {reason}")]
    #[from(ignore)]
    ProtobufDecodeError {
        /// Name of the protobuf message type.
        message_type: &'static str,
        /// Detailed wire decoding failure reason.
        reason: String,
    },

    /// Binary protobuf wire encoding failed.
    #[display("Protobuf encode error in {message_type}: {reason}")]
    #[from(ignore)]
    ProtobufEncodeError {
        /// Name of the protobuf message type.
        message_type: &'static str,
        /// Detailed wire encoding failure reason.
        reason: String,
    },

    /// Parsing of an XLA `HloModuleProto` or textual HLO module failed.
    #[display("HLO parse error: {reason}")]
    #[from(ignore)]
    HloParseError {
        /// Detailed HLO parser diagnostic.
        reason: String,
    },

    /// Parsing of an MLIR `StableHLO` module failed.
    #[display("StableHLO parse error: {reason}")]
    #[from(ignore)]
    StableHloParseError {
        /// Detailed `StableHLO` parser diagnostic.
        reason: String,
    },

    /// HLO or `StableHLO` compilation failed.
    #[display("TPU compilation failed: {reason}")]
    #[from(ignore)]
    CompilationFailed {
        /// Compiler error message.
        reason: String,
    },

    /// A specific phased compilation step (`PJRT_PhaseCompile_Extension`) failed.
    #[display("Phase compilation failed in phase '{phase_name}': {reason}")]
    #[from(ignore)]
    PhaseCompilationFailed {
        /// Name of the compilation phase.
        phase_name: String,
        /// Failure diagnostic.
        reason: String,
    },

    /// Serialization of an executable or topology failed.
    #[display("Serialization failed: {reason}")]
    #[from(ignore)]
    SerializationFailed {
        /// Explanation of the serialization error.
        reason: String,
    },

    /// Deserialization of an executable or topology failed due to corrupt or incompatible bytes.
    #[display("Deserialization failed: {reason}")]
    #[from(ignore)]
    DeserializationFailed {
        /// Explanation of the deserialization error.
        reason: String,
    },

    /// Program execution on a TPU core failed.
    #[display("Execution failed on device {device_id}: {reason}")]
    #[from(ignore)]
    ExecutionFailed {
        /// Device ID where execution failed.
        device_id: u32,
        /// Description of the execution fault.
        reason: String,
    },

    /// Host-to-device, device-to-host, or device-to-device DMA transfer failed.
    #[display("DMA transfer ({direction}) failed: {reason}")]
    #[from(ignore)]
    TransferFailed {
        /// Transfer direction description.
        direction: &'static str,
        /// Reason for the transfer failure.
        reason: String,
    },

    /// A `CopyToDeviceStream` chunk exceeded the total stream byte size.
    #[display(
        "CopyToDeviceStream overflow: attempted {attempted_bytes} bytes with {remaining_bytes} bytes remaining"
    )]
    #[from(ignore)]
    StreamOverflow {
        /// Number of bytes in the chunk being added.
        attempted_bytes: i64,
        /// Number of bytes remaining in the stream.
        remaining_bytes: i64,
    },

    /// An asynchronous host-to-device transfer buffer encountered an error.
    #[display("Async transfer error on buffer index {buffer_index}: {reason}")]
    #[from(ignore)]
    AsyncTransferError {
        /// Index of the transfer buffer in the manager.
        buffer_index: usize,
        /// Transfer failure description.
        reason: String,
    },

    /// Attempted to fulfill a `PJRT_Buffer` alias that was already fulfilled.
    #[display("Alias buffer {buffer_id} has already been fulfilled")]
    #[from(ignore)]
    AliasBufferAlreadyFulfilled {
        /// Identifier of the alias buffer.
        buffer_id: u64,
    },

    /// An asynchronous event completed with an error.
    #[display("Event {event_id} failed: {reason}")]
    #[from(ignore)]
    EventError {
        /// Identifier of the failed event.
        event_id: u64,
        /// Error reason attached to the event.
        reason: String,
    },

    /// A synchronization or tracking event failed without a specific numeric event ID.
    #[display("Event failed: {reason}")]
    #[from(ignore)]
    EventFailed {
        /// Error reason attached to the event.
        reason: String,
    },

    /// Low-level operating system or device file I/O failure.
    #[display("Hardware I/O error during {operation} (errno={errno}): {message}")]
    #[from(ignore)]
    HardwareIoError {
        /// Name of the syscall or hardware operation.
        operation: &'static str,
        /// OS error number.
        errno: i32,
        /// Human-readable OS error message.
        message: String,
    },

    /// Kernel driver `ioctl` call to `/dev/accel*` or `/dev/vfio/*` failed.
    #[display("Kernel ioctl {request:#x} failed (errno={errno}): {message}")]
    #[from(ignore)]
    IoctlFailed {
        /// Raw `ioctl` request command number.
        request: u64,
        /// OS error number.
        errno: i32,
        /// Human-readable error details.
        message: String,
    },

    /// `PCIe` BAR MMIO register read or write was out of bounds.
    #[display("PCIe BAR{bar_index} MMIO access out of bounds at offset {offset:#x} (size={size})")]
    #[from(ignore)]
    MmioOutOfBounds {
        /// `PCIe` BAR index (e.g., `0` or `2`).
        bar_index: u8,
        /// Byte offset into the BAR aperture.
        offset: u64,
        /// Access width in bytes.
        size: usize,
    },

    /// Attempted to stop or collect from a profiler session that is not active.
    #[display("TPU profiler session is not currently active")]
    #[from(ignore)]
    ProfilerNotActive,

    /// Attempted to start a profiler session that is already active.
    #[display("TPU profiler session is already active")]
    #[from(ignore)]
    ProfilerAlreadyActive,

    /// `TpuEmbeddingEngine` operation was called before `Finalize`.
    #[display("TPU embedding engine is not initialized")]
    #[from(ignore)]
    EmbeddingEngineNotInitialized,

    /// Requested `SparseCore` embedding table name was not found.
    #[display("TPU embedding table '{table_name}' not found")]
    #[from(ignore)]
    EmbeddingTableNotFound {
        /// Name of the missing embedding table.
        table_name: String,
    },

    /// Requested `LibtpuSdk` telemetry metric name was not found.
    #[display("TPU telemetry metric '{metric_name}' not found")]
    #[from(ignore)]
    MetricNotFound {
        /// Name of the unknown telemetry metric.
        metric_name: String,
    },

    /// Requested debug or host callback ID was not found.
    #[display("Callback ID {callback_id} not found")]
    #[from(ignore)]
    CallbackNotFound {
        /// Identifier of the missing callback.
        callback_id: u64,
    },

    /// Requested XLA transform pass name was not found in the transform registry.
    #[display("XLA transform '{transform_name}' not found")]
    #[from(ignore)]
    TransformNotFound {
        /// Name of the missing XLA transform pass.
        transform_name: String,
    },

    /// Requested PJRT extension type is not registered.
    #[display("PJRT extension type {extension_type} not found")]
    #[from(ignore)]
    ExtensionNotFound {
        /// Raw extension type identifier.
        extension_type: i32,
    },

    /// A synchronization primitive (`Mutex` or `RwLock`) was poisoned.
    #[display("Synchronization lock poisoned for {resource}")]
    #[from(ignore)]
    LockPoisoned {
        /// Name of the locked resource.
        resource: &'static str,
    },

    /// A C string or byte slice failed UTF-8 validation.
    #[display("{_0}")]
    Utf8ConversionFailed(#[error(not(source))] Utf8ErrorWrapper),

    /// A named parameter failed UTF-8 validation across an FFI boundary.
    #[display("Parameter '{parameter}' is not valid UTF-8")]
    #[from(ignore)]
    Utf8ConversionError {
        /// Name of the parameter containing invalid UTF-8 bytes.
        parameter: &'static str,
    },

    /// A `prost` protobuf wire decoding operation failed.
    #[display("{_0}")]
    ProstDecodeFailed(#[error(not(source))] ProstDecodeErrorWrapper),

    /// A `prost` protobuf wire encoding operation failed.
    #[display("{_0}")]
    ProstEncodeFailed(#[error(not(source))] ProstEncodeErrorWrapper),

    /// A `prost_reflect` dynamic descriptor pool operation failed.
    #[display("{_0}")]
    ProstReflectDescriptorFailed(#[error(not(source))] ProstReflectErrorWrapper),

    /// A hardware `SyncFlag` wait barrier timed out before reaching the expected threshold.
    #[display(
        "SyncFlag {sync_flag_id} timed out on device {device_id}: expected >= {expected_value}, actual {actual_value}"
    )]
    #[from(ignore)]
    SyncFlagTimeout {
        /// Logical device ID.
        device_id: u32,
        /// Hardware `SyncFlag` register index.
        sync_flag_id: u32,
        /// Minimum expected counter value.
        expected_value: u32,
        /// Observed counter value at timeout.
        actual_value: u32,
    },

    /// A host command continuation (`HostCommandOpcodeProto`) failed during dispatch.
    #[display("Host command (opcode={opcode}, continuation_id={continuation_id}) failed: {reason}")]
    #[from(ignore)]
    HostCommandError {
        /// Raw `HostCommandOpcodeProto` discriminant.
        opcode: i32,
        /// Host continuation identifier.
        continuation_id: u64,
        /// Failure diagnostic.
        reason: String,
    },

    /// Low-level optimizer (`LLO`) physical register allocation exceeded hardware register file capacity.
    #[display(
        "Register allocation failed for {reg_class}: requested {requested} registers, capacity {capacity}"
    )]
    #[from(ignore)]
    RegisterAllocationFailed {
        /// Hardware register file class (`SREG`, `PREG`, `VREG`, `VMASK`).
        reg_class: &'static str,
        /// Number of live registers requested.
        requested: usize,
        /// Hardware register file capacity.
        capacity: usize,
    },

    /// An unexpected panic was caught at an `extern "C"` FFI boundary.
    #[display("FFI boundary panic caught in '{symbol}': {detail}")]
    #[from(ignore)]
    FfiBoundaryPanic {
        /// Exported C symbol name where the unwind was intercepted.
        symbol: &'static str,
        /// Extracted panic payload description.
        detail: String,
    },

    /// An ELF, Mach-O, or PE binary object failed to parse via `goblin`.
    #[display("{_0}")]
    GoblinParseFailed(#[error(not(source))] GoblinErrorWrapper),

    /// A CLI flag failed to parse via `lexopt`.
    #[display("{_0}")]
    LexoptParseFailed(#[error(not(source))] LexoptErrorWrapper),

    /// An ELF, Mach-O, or PE binary failed to parse via `object`.
    #[display("{_0}")]
    ObjectParseFailed(#[error(not(source))] ObjectErrorWrapper),

    /// A DWARF `.eh_frame` unwind section failed to parse via `gimli`.
    #[display("{_0}")]
    GimliParseFailed(#[error(not(source))] GimliErrorWrapper),

    /// An Itanium C++ ABI symbol failed to demangle via `cpp_demangle`.
    #[display("{_0}")]
    CppDemangleFailed(#[error(not(source))] CppDemangleErrorWrapper),

    /// A standard library I/O operation failed.
    #[display("{_0}")]
    IoFailed(#[error(not(source))] IoErrorWrapper),

    /// A low-level POSIX syscall failed via `rustix`.
    #[display("{_0}")]
    RustixSyscallFailed(#[error(not(source))] RustixErrnoWrapper),

    /// An executable or shared library binary had an invalid or unsupported format.
    #[display("Binary format error: {reason}")]
    #[from(ignore)]
    BinaryFormatError {
        /// Detailed explanation of the binary format failure.
        reason: String,
    },

    /// An operation is recognized by the ABI but intentionally unimplemented.
    #[display("Operation '{operation}' is not implemented")]
    #[from(ignore)]
    UnimplementedOperation {
        /// Name of the unimplemented operation.
        operation: &'static str,
    },

    /// Unrecognized TPU hardware generation identifier (`UnsupportedGeneration`).
    #[display("Unsupported TPU generation '{raw_value}'")]
    #[from(ignore)]
    UnsupportedGeneration {
        /// Raw generation string that could not be parsed.
        raw_value: String,
    },

    /// Host-to-device or device-to-host byte length mismatch (`BufferShapeMismatch`).
    #[display("Buffer shape byte mismatch: expected {expected_bytes}, got {actual_bytes}")]
    #[from(ignore)]
    BufferShapeMismatch {
        /// Expected byte length for the tensor shape.
        expected_bytes: usize,
        /// Actual byte length provided by caller.
        actual_bytes: usize,
    },

    /// HBM or VMEM region exhaustion in `HbmRegionAllocator` (`OutOfDeviceMemory`).
    #[display(
        "Out of device memory in {memory_kind}: requested {requested}, available {available}"
    )]
    #[from(ignore)]
    OutOfDeviceMemory {
        /// Number of bytes requested.
        requested: usize,
        /// Number of bytes available.
        available: usize,
        /// Memory space kind name.
        memory_kind: &'static str,
    },

    /// Unmapped or misaligned device physical/IOVA address (`InvalidDeviceAddress`).
    #[display("Invalid device address {address:#x}")]
    #[from(ignore)]
    InvalidDeviceAddress {
        /// Raw device physical or IOVA address.
        address: u64,
    },

    /// Linux `PCIe` `/sys/bus/pci/devices` or `/sys/class/accel` enumeration error (`SysfsDiscoveryFailed`).
    #[display("Sysfs device discovery failed at '{path}': {reason}")]
    #[from(ignore)]
    SysfsDiscoveryFailed {
        /// Sysfs directory path.
        path: String,
        /// Failure diagnostic.
        reason: String,
    },

    /// `TensorCore` or `SparseCore` ECC, MXU overflow, or `SyncFlag` timeout fault (`HardwareFault`).
    #[display("Hardware fault on device {device_id} (CSR={status_csr:#x}): {fault_description}")]
    #[from(ignore)]
    HardwareFault {
        /// Logical device ID where the fault occurred.
        device_id: u32,
        /// Raw status CSR register value.
        status_csr: u64,
        /// Static description of the hardware fault bit.
        fault_description: &'static str,
    },

    /// Textual MLIR / `StableHLO` / HLO syntax or semantic error (`MlirParseError`).
    #[display("MLIR parse error at {line}:{column}: {reason}")]
    #[from(ignore)]
    MlirParseError {
        /// 1-based line number.
        line: usize,
        /// 1-based column number.
        column: usize,
        /// Diagnostic message.
        reason: String,
    },

    /// Binary `ML\xefR` container or bitstream decoding error (`MlirBytecodeDecodeError`).
    #[display("MLIR bytecode decode error at offset {offset} (section {section_id}): {reason}")]
    #[from(ignore)]
    MlirBytecodeDecodeError {
        /// Byte offset in the `ML\xefR` payload.
        offset: usize,
        /// MLIR bytecode section identifier (`0..=5`).
        section_id: u8,
        /// Static failure description.
        reason: &'static str,
    },

    /// SSA dominance, operand arity, or shape inference violation (`HloVerificationError`).
    #[display("HLO verification error in '{instruction_name}': {reason}")]
    #[from(ignore)]
    HloVerificationError {
        /// Name of the failing HLO instruction.
        instruction_name: String,
        /// Verification failure explanation.
        reason: String,
    },

    /// Invalid minor-to-major permutation or tile specification (`LayoutAssignmentError`).
    #[display("Layout assignment error for {shape_summary}: {reason}")]
    #[from(ignore)]
    LayoutAssignmentError {
        /// Summary of the tensor shape.
        shape_summary: String,
        /// Static layout assignment failure reason.
        reason: &'static str,
    },

    /// Binary ELF/Mach-O/PE header, symbol, relocation, or DWARF `.eh_frame` parse error (`ElfInspectionError`).
    #[display("ELF inspection error: {reason}")]
    #[from(ignore)]
    ElfInspectionError {
        /// Detailed inspection failure reason.
        reason: String,
    },

    /// Embedded `filewrapper_toc` CSR decompression failure (`BrotliDecompressionError`).
    #[display("Brotli decompression error for '{resource_name}': {reason}")]
    #[from(ignore)]
    BrotliDecompressionError {
        /// Embedded TOC resource name.
        resource_name: String,
        /// Decompression failure explanation.
        reason: String,
    },

    /// Host filesystem or memory-map I/O failure (`IoError`).
    #[display("I/O error during {operation}: {reason}")]
    #[from(ignore)]
    IoError {
        /// Operation name.
        operation: &'static str,
        /// Failure reason string.
        reason: String,
    },
}

impl From<prost::DecodeError> for LibTpuError {
    fn from(err: prost::DecodeError) -> Self {
        Self::ProstDecodeFailed(ProstDecodeErrorWrapper::from(err))
    }
}

impl From<prost::EncodeError> for LibTpuError {
    fn from(err: prost::EncodeError) -> Self {
        Self::ProstEncodeFailed(ProstEncodeErrorWrapper::from(err))
    }
}

impl From<prost_reflect::DescriptorError> for LibTpuError {
    fn from(err: prost_reflect::DescriptorError) -> Self {
        Self::ProstReflectDescriptorFailed(ProstReflectErrorWrapper::from(err))
    }
}

impl From<goblin::error::Error> for LibTpuError {
    fn from(err: goblin::error::Error) -> Self {
        Self::GoblinParseFailed(GoblinErrorWrapper::from(err))
    }
}

impl From<lexopt::Error> for LibTpuError {
    fn from(err: lexopt::Error) -> Self {
        Self::LexoptParseFailed(LexoptErrorWrapper::from(err))
    }
}

impl From<object::read::Error> for LibTpuError {
    fn from(err: object::read::Error) -> Self {
        Self::ObjectParseFailed(ObjectErrorWrapper::from(err))
    }
}

impl From<gimli::Error> for LibTpuError {
    fn from(err: gimli::Error) -> Self {
        Self::GimliParseFailed(GimliErrorWrapper::from(err))
    }
}

impl From<cpp_demangle::error::Error> for LibTpuError {
    fn from(err: cpp_demangle::error::Error) -> Self {
        Self::CppDemangleFailed(CppDemangleErrorWrapper::from(err))
    }
}

impl From<std::io::Error> for LibTpuError {
    fn from(err: std::io::Error) -> Self {
        Self::IoFailed(IoErrorWrapper::from(err))
    }
}

impl From<rustix::io::Errno> for LibTpuError {
    fn from(err: rustix::io::Errno) -> Self {
        Self::RustixSyscallFailed(RustixErrnoWrapper::from(err))
    }
}

impl LibTpuError {
    /// Maps this [`LibTpuError`] to its corresponding [`PjrtErrorCode`].
    #[must_use]
    pub const fn error_code(&self) -> PjrtErrorCode {
        match self {
            Self::NullPointer { .. }
            | Self::InvalidStructSize { .. }
            | Self::StructSizeMismatch { .. }
            | Self::InvalidArgument { .. }
            | Self::InvalidElementType { .. }
            | Self::InvalidDimension { .. }
            | Self::InvalidMemoryKind { .. }
            | Self::DmaAlignmentError { .. }
            | Self::EnvironmentParseError { .. }
            | Self::InvalidShape { .. }
            | Self::UnsupportedElementType { .. }
            | Self::ProtobufDecodeError { .. }
            | Self::ProstDecodeFailed(_)
            | Self::ProstReflectDescriptorFailed(_)
            | Self::GoblinParseFailed(_)
            | Self::LexoptParseFailed(_)
            | Self::ObjectParseFailed(_)
            | Self::GimliParseFailed(_)
            | Self::CppDemangleFailed(_)
            | Self::BinaryFormatError { .. }
            | Self::HloParseError { .. }
            | Self::StableHloParseError { .. }
            | Self::UnsupportedGeneration { .. }
            | Self::BufferShapeMismatch { .. }
            | Self::InvalidDeviceAddress { .. }
            | Self::MlirParseError { .. }
            | Self::MlirBytecodeDecodeError { .. }
            | Self::HloVerificationError { .. }
            | Self::LayoutAssignmentError { .. }
            | Self::ElfInspectionError { .. }
            | Self::BrotliDecompressionError { .. }
            | Self::Utf8ConversionFailed(_)
            | Self::Utf8ConversionError { .. } => PjrtErrorCode::InvalidArgument,
            Self::UnsupportedApiVersion { .. } | Self::UnimplementedOperation { .. } => {
                PjrtErrorCode::Unimplemented
            }
            Self::DeviceNotFound { .. }
            | Self::ChipNotFound { .. }
            | Self::HostNotFound { .. }
            | Self::MemorySpaceNotFound { .. }
            | Self::DmaMappingNotFound { .. }
            | Self::EmbeddingTableNotFound { .. }
            | Self::MetricNotFound { .. }
            | Self::CallbackNotFound { .. }
            | Self::TransformNotFound { .. }
            | Self::ExtensionNotFound { .. } => PjrtErrorCode::NotFound,
            Self::DeviceNotAddressable { .. }
            | Self::BufferDeleted { .. }
            | Self::ExecutableDeleted { .. }
            | Self::AliasAlreadyFulfilled { .. }
            | Self::AliasBufferAlreadyFulfilled { .. }
            | Self::ProfilerStateError { .. }
            | Self::ProfilerNotActive
            | Self::EmbeddingEngineNotInitialized => PjrtErrorCode::FailedPrecondition,
            Self::ProfilerAlreadyActive => PjrtErrorCode::AlreadyExists,
            Self::OutOfMemory { .. }
            | Self::OutOfDeviceMemory { .. }
            | Self::RegisterAllocationFailed { .. }
            | Self::DmaMappingFailed { .. } => PjrtErrorCode::ResourceExhausted,
            Self::NumericOverflow { .. }
            | Self::StreamOverflow { .. }
            | Self::MmioOutOfBounds { .. } => PjrtErrorCode::OutOfRange,
            Self::MegascaleBarrierTimeout { .. } | Self::SyncFlagTimeout { .. } => {
                PjrtErrorCode::DeadlineExceeded
            }
            Self::DevicePoisoned { .. }
            | Self::HardwareFault { .. }
            | Self::ProtobufEncodeError { .. }
            | Self::ProstEncodeFailed(_)
            | Self::CompilationFailed { .. }
            | Self::PhaseCompilationFailed { .. }
            | Self::SerializationFailed { .. }
            | Self::DeserializationFailed { .. }
            | Self::ExecutionFailed { .. }
            | Self::TransferFailed { .. }
            | Self::AsyncTransferError { .. }
            | Self::CollectiveError { .. }
            | Self::HostCommandError { .. }
            | Self::FfiBoundaryPanic { .. }
            | Self::InternalError { .. }
            | Self::EventError { .. }
            | Self::EventFailed { .. }
            | Self::LockPoisoned { .. } => PjrtErrorCode::Internal,
            Self::HardwareIoError { .. }
            | Self::SysfsDiscoveryFailed { .. }
            | Self::IoctlFailed { .. }
            | Self::VfioIommuError { .. }
            | Self::IoError { .. }
            | Self::IoFailed(_)
            | Self::RustixSyscallFailed(_) => PjrtErrorCode::Unavailable,
        }
    }

    /// Maps this [`LibTpuError`] to its corresponding [`PjrtErrorCode`].
    #[must_use]
    pub const fn pjrt_code(&self) -> PjrtErrorCode {
        self.error_code()
    }

    /// Maps this [`LibTpuError`] to its corresponding [`PjrtErrorCode`].
    #[must_use]
    pub const fn pjrt_error_code(&self) -> PjrtErrorCode {
        self.error_code()
    }

    /// Maps this [`LibTpuError`] to its corresponding [`AbseilStatusCode`].
    #[must_use]
    pub const fn abseil_code(&self) -> AbseilStatusCode {
        self.error_code()
    }

    /// Maps this [`LibTpuError`] to its corresponding [`AbseilStatusCode`].
    #[must_use]
    pub const fn abseil_status_code(&self) -> AbseilStatusCode {
        self.error_code()
    }

    /// Maps this [`LibTpuError`] to its corresponding [`LibtpuSdkErrorCode`].
    #[must_use]
    pub const fn sdk_code(&self) -> LibtpuSdkErrorCode {
        self.sdk_error_code()
    }

    /// Maps this [`LibTpuError`] to its corresponding [`LibtpuSdkErrorCode`].
    #[must_use]
    pub const fn sdk_error_code(&self) -> LibtpuSdkErrorCode {
        LibtpuSdkErrorCode::from_pjrt_code(self.error_code())
    }

    /// Returns structured diagnostic [`ErrorPayload`] entries for `PJRT_Error_ForEachPayload`.
    #[must_use]
    pub fn payloads(&self) -> Vec<ErrorPayload> {
        let mut out = vec![
            ErrorPayload::new(
                String::from("type.googleapis.com/xla.TpuErrorPayload"),
                self.to_string().into_bytes(),
            ),
            ErrorPayload::new(
                String::from("type.googleapis.com/xla.StatusCode"),
                self.error_code().to_string().into_bytes(),
            ),
        ];
        match self {
            Self::MegascaleBarrierTimeout {
                slice_id,
                barrier_id,
            } => {
                out.push(ErrorPayload::new(
                    String::from(
                        "type.googleapis.com/xla.megascale.runtime.external.MegaScaleRuntimeError",
                    ),
                    format!("slice_id={slice_id},barrier_id={barrier_id}").into_bytes(),
                ));
            }
            Self::CollectiveError {
                communicator_id,
                op_name,
                reason,
            } => {
                out.push(ErrorPayload::new(
                    String::from(
                        "type.googleapis.com/xla.megascale.runtime.external.MegaScaleRuntimeError",
                    ),
                    format!("communicator_id={communicator_id},op={op_name},reason={reason}")
                        .into_bytes(),
                ));
            }
            Self::CompilationFailed { reason }
            | Self::HloParseError { reason }
            | Self::StableHloParseError { reason } => {
                out.push(ErrorPayload::new(
                    String::from("type.googleapis.com/tensorflow.tpu.CompilationResultProto"),
                    reason.as_bytes().to_vec(),
                ));
            }
            Self::PhaseCompilationFailed { phase_name, reason } => {
                out.push(ErrorPayload::new(
                    String::from("type.googleapis.com/tensorflow.tpu.CompilationResultProto"),
                    format!("phase={phase_name}:{reason}").into_bytes(),
                ));
            }
            Self::HloVerificationError {
                instruction_name,
                reason,
            } => {
                out.push(ErrorPayload::new(
                    String::from("type.googleapis.com/tensorflow.tpu.CompilationResultProto"),
                    format!("instruction={instruction_name}:{reason}").into_bytes(),
                ));
            }
            Self::HardwareFault {
                device_id,
                status_csr,
                fault_description,
            } => {
                out.push(ErrorPayload::new(
                    String::from("type.googleapis.com/xla.TpuHardwareFaultPayload"),
                    format!("device={device_id},csr={status_csr:#x},fault={fault_description}")
                        .into_bytes(),
                ));
            }
            Self::DmaAlignmentError {
                address,
                required_alignment,
            } => {
                out.push(ErrorPayload::new(
                    String::from("type.googleapis.com/xla.TpuDmaErrorPayload"),
                    format!("address={address:#x},align={required_alignment}").into_bytes(),
                ));
            }
            _ => {}
        }
        out
    }

    /// Returns `true` if this error represents a predetermined device execution error
    /// (`PJRT_TpuExecutable_IsPredeterminedError`).
    #[must_use]
    pub const fn is_predetermined_error(&self) -> bool {
        matches!(
            self,
            Self::DevicePoisoned { .. }
                | Self::BufferDeleted { .. }
                | Self::ExecutableDeleted { .. }
        )
    }

    /// Returns `true` if this error represents a collective or transfer cancellation placeholder
    /// (`PJRT_TpuExecutable_IsCancellationPlaceholderError`).
    #[must_use]
    pub const fn is_cancellation_placeholder_error(&self) -> bool {
        matches!(
            self,
            Self::AsyncTransferError { .. } | Self::EventError { .. } | Self::EventFailed { .. }
        )
    }
}

/// Alias for `Result<T, LibTpuError>` used throughout `libtpu-rs`.
pub type LibTpuResult<T> = Result<T, LibTpuError>;

#[cfg(test)]
#[allow(for_loops_over_fallibles, clippy::too_many_lines)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn test_pjrt_and_sdk_error_code_roundtrip() {
        let cases = [
            (0, PjrtErrorCode::Ok, "OK", LibtpuSdkErrorCode::Ok),
            (
                1,
                PjrtErrorCode::Cancelled,
                "CANCELLED",
                LibtpuSdkErrorCode::Internal,
            ),
            (
                2,
                PjrtErrorCode::Unknown,
                "UNKNOWN",
                LibtpuSdkErrorCode::Internal,
            ),
            (
                3,
                PjrtErrorCode::InvalidArgument,
                "INVALID_ARGUMENT",
                LibtpuSdkErrorCode::InvalidArgument,
            ),
            (
                4,
                PjrtErrorCode::DeadlineExceeded,
                "DEADLINE_EXCEEDED",
                LibtpuSdkErrorCode::Internal,
            ),
            (
                5,
                PjrtErrorCode::NotFound,
                "NOT_FOUND",
                LibtpuSdkErrorCode::NotFound,
            ),
            (
                6,
                PjrtErrorCode::AlreadyExists,
                "ALREADY_EXISTS",
                LibtpuSdkErrorCode::FailedPrecondition,
            ),
            (
                7,
                PjrtErrorCode::PermissionDenied,
                "PERMISSION_DENIED",
                LibtpuSdkErrorCode::Internal,
            ),
            (
                8,
                PjrtErrorCode::ResourceExhausted,
                "RESOURCE_EXHAUSTED",
                LibtpuSdkErrorCode::Internal,
            ),
            (
                9,
                PjrtErrorCode::FailedPrecondition,
                "FAILED_PRECONDITION",
                LibtpuSdkErrorCode::FailedPrecondition,
            ),
            (
                10,
                PjrtErrorCode::Aborted,
                "ABORTED",
                LibtpuSdkErrorCode::Internal,
            ),
            (
                11,
                PjrtErrorCode::OutOfRange,
                "OUT_OF_RANGE",
                LibtpuSdkErrorCode::InvalidArgument,
            ),
            (
                12,
                PjrtErrorCode::Unimplemented,
                "UNIMPLEMENTED",
                LibtpuSdkErrorCode::Internal,
            ),
            (
                13,
                PjrtErrorCode::Internal,
                "INTERNAL",
                LibtpuSdkErrorCode::Internal,
            ),
            (
                14,
                PjrtErrorCode::Unavailable,
                "UNAVAILABLE",
                LibtpuSdkErrorCode::Unavailable,
            ),
            (
                15,
                PjrtErrorCode::DataLoss,
                "DATA_LOSS",
                LibtpuSdkErrorCode::Internal,
            ),
            (
                16,
                PjrtErrorCode::Unauthenticated,
                "UNAUTHENTICATED",
                LibtpuSdkErrorCode::Internal,
            ),
        ];
        for (raw, expected, text, expected_sdk) in cases {
            let code = PjrtErrorCode::from_raw(raw);
            assert_eq!(code, expected);
            assert_eq!(code.as_raw(), raw);
            assert_eq!(code.to_string(), text);
            assert_ne!(format!("{code:?}"), "");
            let sdk = LibtpuSdkErrorCode::from_pjrt_code(code);
            assert_eq!(sdk, expected_sdk);
            assert_eq!(sdk.as_raw(), expected_sdk as i32);
            assert_ne!(sdk.to_string(), "");
            assert_ne!(format!("{sdk:?}"), "");
        }
        assert_eq!(PjrtErrorCode::from_raw(999), PjrtErrorCode::Unknown);
    }

    #[test]
    fn test_all_libtpu_error_variants_display_and_codes() {
        let invalid_utf8 = vec![0xff_u8, 0xfe_u8];
        for utf8_err in std::str::from_utf8(invalid_utf8.as_slice()).err() {
            let wrapper = Utf8ErrorWrapper::from(utf8_err);
            assert!(wrapper.source().is_none());
            let cloned_wrapper = wrapper.clone();
            assert_ne!(format!("{cloned_wrapper:?}"), "");
            let err_from_utf8 = LibTpuError::from(wrapper);
            assert_eq!(err_from_utf8.error_code(), PjrtErrorCode::InvalidArgument);
        }

        for prost_dec_err in <u64 as prost::Message>::decode(&[0x80_u8][..]).err() {
            let err = LibTpuError::from(prost_dec_err);
            assert_eq!(err.error_code(), PjrtErrorCode::InvalidArgument);
            assert_ne!(err.to_string(), "");
        }

        let mut zero_buf = [0_u8; 0];
        let mut zero_slice: &mut [u8] = &mut zero_buf;
        for prost_enc_err in <u64 as prost::Message>::encode(&1_u64, &mut zero_slice).err() {
            let err = LibTpuError::from(prost_enc_err);
            assert_eq!(err.error_code(), PjrtErrorCode::Internal);
            assert_ne!(err.to_string(), "");
        }

        assert_eq!(
            LibtpuSdkErrorCode::from_primitive(999),
            LibtpuSdkErrorCode::Internal
        );
        assert_eq!(i32::from(LibtpuSdkErrorCode::Ok), 0);
        assert_eq!(i32::from(PjrtErrorCode::Ok), 0);

        let variants: Vec<(LibTpuError, PjrtErrorCode)> = vec![
            (
                LibTpuError::NullPointer { parameter: "args" },
                PjrtErrorCode::InvalidArgument,
            ),
            (
                LibTpuError::InvalidStructSize {
                    struct_name: "PJRT_Client_Create_Args",
                    expected_min: 32,
                    actual: 8,
                },
                PjrtErrorCode::InvalidArgument,
            ),
            (
                LibTpuError::StructSizeMismatch {
                    struct_name: "PJRT_Client_Create_Args",
                    expected: 72,
                    actual: 8,
                },
                PjrtErrorCode::InvalidArgument,
            ),
            (
                LibTpuError::InvalidArgument {
                    parameter: "device_id",
                    reason: String::from("negative"),
                },
                PjrtErrorCode::InvalidArgument,
            ),
            (
                LibTpuError::InvalidElementType { raw_type: -1 },
                PjrtErrorCode::InvalidArgument,
            ),
            (
                LibTpuError::InvalidDimension {
                    index: 0,
                    value: -1,
                },
                PjrtErrorCode::InvalidArgument,
            ),
            (
                LibTpuError::InvalidMemoryKind {
                    kind: String::from("bad_kind"),
                },
                PjrtErrorCode::InvalidArgument,
            ),
            (
                LibTpuError::DmaAlignmentError {
                    address: 0x1003,
                    required_alignment: 4096,
                },
                PjrtErrorCode::InvalidArgument,
            ),
            (
                LibTpuError::DmaMappingNotFound {
                    host_address: 0x2000,
                },
                PjrtErrorCode::NotFound,
            ),
            (
                LibTpuError::AliasAlreadyFulfilled { buffer_id: 99 },
                PjrtErrorCode::FailedPrecondition,
            ),
            (
                LibTpuError::VfioIommuError {
                    group_id: 1,
                    reason: String::from("iommu fault"),
                },
                PjrtErrorCode::Unavailable,
            ),
            (
                LibTpuError::EnvironmentParseError {
                    variable: String::from("TPU_HOST_BOUNDS"),
                    value: String::from("bad"),
                    reason: String::from("expected 3 ints"),
                },
                PjrtErrorCode::InvalidArgument,
            ),
            (
                LibTpuError::CollectiveError {
                    communicator_id: 1,
                    op_name: "AllReduce",
                    reason: String::from("timeout"),
                },
                PjrtErrorCode::Internal,
            ),
            (
                LibTpuError::MegascaleBarrierTimeout {
                    slice_id: 0,
                    barrier_id: 42,
                },
                PjrtErrorCode::DeadlineExceeded,
            ),
            (
                LibTpuError::ProfilerStateError {
                    expected_state: "Active",
                    actual_state: "Inactive",
                },
                PjrtErrorCode::FailedPrecondition,
            ),
            (
                LibTpuError::InternalError {
                    context: String::from("invariant"),
                },
                PjrtErrorCode::Internal,
            ),
            (
                LibTpuError::NumericOverflow {
                    context: "shape_multiply",
                },
                PjrtErrorCode::OutOfRange,
            ),
            (
                LibTpuError::UnsupportedApiVersion {
                    requested_major: 1,
                    requested_minor: 0,
                    supported_major: 0,
                    supported_minor: 116,
                },
                PjrtErrorCode::Unimplemented,
            ),
            (
                LibTpuError::DeviceNotFound { device_id: 42 },
                PjrtErrorCode::NotFound,
            ),
            (
                LibTpuError::DeviceNotAddressable {
                    device_id: 7,
                    process_index: 1,
                },
                PjrtErrorCode::FailedPrecondition,
            ),
            (
                LibTpuError::DevicePoisoned {
                    device_id: 0,
                    launch_id: 7,
                    reason: String::from("poisoned"),
                },
                PjrtErrorCode::Internal,
            ),
            (
                LibTpuError::ChipNotFound { x: 1, y: 2, z: 3 },
                PjrtErrorCode::NotFound,
            ),
            (
                LibTpuError::HostNotFound { host_id: 2 },
                PjrtErrorCode::NotFound,
            ),
            (
                LibTpuError::MemorySpaceNotFound { memory_id: 99 },
                PjrtErrorCode::NotFound,
            ),
            (
                LibTpuError::OutOfMemory {
                    requested_bytes: 1024,
                    available_bytes: 512,
                },
                PjrtErrorCode::ResourceExhausted,
            ),
            (
                LibTpuError::DmaMappingFailed {
                    host_address: 0x1000,
                    size: 4096,
                    reason: String::from("unaligned"),
                },
                PjrtErrorCode::ResourceExhausted,
            ),
            (
                LibTpuError::BufferDeleted { buffer_id: 10 },
                PjrtErrorCode::FailedPrecondition,
            ),
            (
                LibTpuError::ExecutableDeleted { executable_id: 20 },
                PjrtErrorCode::FailedPrecondition,
            ),
            (
                LibTpuError::InvalidShape {
                    reason: String::from("bad rank"),
                },
                PjrtErrorCode::InvalidArgument,
            ),
            (
                LibTpuError::UnsupportedElementType { raw_type: -1 },
                PjrtErrorCode::InvalidArgument,
            ),
            (
                LibTpuError::ProtobufDecodeError {
                    message_type: "HloModuleProto",
                    reason: String::from("truncated varint"),
                },
                PjrtErrorCode::InvalidArgument,
            ),
            (
                LibTpuError::ProtobufEncodeError {
                    message_type: "XSpace",
                    reason: String::from("size overflow"),
                },
                PjrtErrorCode::Internal,
            ),
            (
                LibTpuError::HloParseError {
                    reason: String::from("missing ENTRY"),
                },
                PjrtErrorCode::InvalidArgument,
            ),
            (
                LibTpuError::StableHloParseError {
                    reason: String::from("invalid mlir op"),
                },
                PjrtErrorCode::InvalidArgument,
            ),
            (
                LibTpuError::CompilationFailed {
                    reason: String::from("syntax error"),
                },
                PjrtErrorCode::Internal,
            ),
            (
                LibTpuError::PhaseCompilationFailed {
                    phase_name: String::from("layout_assignment"),
                    reason: String::from("incompatible tile"),
                },
                PjrtErrorCode::Internal,
            ),
            (
                LibTpuError::SerializationFailed {
                    reason: String::from("io error"),
                },
                PjrtErrorCode::Internal,
            ),
            (
                LibTpuError::DeserializationFailed {
                    reason: String::from("bad magic"),
                },
                PjrtErrorCode::Internal,
            ),
            (
                LibTpuError::ExecutionFailed {
                    device_id: 0,
                    reason: String::from("halt"),
                },
                PjrtErrorCode::Internal,
            ),
            (
                LibTpuError::TransferFailed {
                    direction: "H2D",
                    reason: String::from("dma timeout"),
                },
                PjrtErrorCode::Internal,
            ),
            (
                LibTpuError::StreamOverflow {
                    attempted_bytes: 128,
                    remaining_bytes: 64,
                },
                PjrtErrorCode::OutOfRange,
            ),
            (
                LibTpuError::AsyncTransferError {
                    buffer_index: 2,
                    reason: String::from("aborted"),
                },
                PjrtErrorCode::Internal,
            ),
            (
                LibTpuError::AliasBufferAlreadyFulfilled { buffer_id: 88 },
                PjrtErrorCode::FailedPrecondition,
            ),
            (
                LibTpuError::EventError {
                    event_id: 5,
                    reason: String::from("aborted"),
                },
                PjrtErrorCode::Internal,
            ),
            (
                LibTpuError::EventFailed {
                    reason: String::from("failed"),
                },
                PjrtErrorCode::Internal,
            ),
            (
                LibTpuError::HardwareIoError {
                    operation: "open",
                    errno: 2,
                    message: String::from("ENOENT"),
                },
                PjrtErrorCode::Unavailable,
            ),
            (
                LibTpuError::IoctlFailed {
                    request: 0x4020_dc08,
                    errno: 22,
                    message: String::from("EINVAL"),
                },
                PjrtErrorCode::Unavailable,
            ),
            (
                LibTpuError::MmioOutOfBounds {
                    bar_index: 0,
                    offset: 0x10_0000,
                    size: 8,
                },
                PjrtErrorCode::OutOfRange,
            ),
            (
                LibTpuError::ProfilerNotActive,
                PjrtErrorCode::FailedPrecondition,
            ),
            (
                LibTpuError::ProfilerAlreadyActive,
                PjrtErrorCode::AlreadyExists,
            ),
            (
                LibTpuError::EmbeddingEngineNotInitialized,
                PjrtErrorCode::FailedPrecondition,
            ),
            (
                LibTpuError::EmbeddingTableNotFound {
                    table_name: String::from("vocab"),
                },
                PjrtErrorCode::NotFound,
            ),
            (
                LibTpuError::MetricNotFound {
                    metric_name: String::from("bad_metric"),
                },
                PjrtErrorCode::NotFound,
            ),
            (
                LibTpuError::CallbackNotFound { callback_id: 404 },
                PjrtErrorCode::NotFound,
            ),
            (
                LibTpuError::TransformNotFound {
                    transform_name: String::from("custom_pass"),
                },
                PjrtErrorCode::NotFound,
            ),
            (
                LibTpuError::ExtensionNotFound { extension_type: 99 },
                PjrtErrorCode::NotFound,
            ),
            (
                LibTpuError::LockPoisoned { resource: "client" },
                PjrtErrorCode::Internal,
            ),
            (
                LibTpuError::Utf8ConversionError {
                    parameter: "metric_name",
                },
                PjrtErrorCode::InvalidArgument,
            ),
            (
                LibTpuError::from(goblin::error::Error::Malformed(String::from("bad elf"))),
                PjrtErrorCode::InvalidArgument,
            ),
            (
                LibTpuError::from(lexopt::Error::MissingValue {
                    option: Some(String::from("--flag")),
                }),
                PjrtErrorCode::InvalidArgument,
            ),
            (
                object::File::parse(&b"bad"[..]).map_or_else(LibTpuError::from, |_| {
                    LibTpuError::ObjectParseFailed(ObjectErrorWrapper(String::from("bad")))
                }),
                PjrtErrorCode::InvalidArgument,
            ),
            (
                LibTpuError::from(gimli::Error::Io),
                PjrtErrorCode::InvalidArgument,
            ),
            (
                cpp_demangle::Symbol::new(&b"_Z"[..]).map_or_else(LibTpuError::from, |_| {
                    LibTpuError::CppDemangleFailed(CppDemangleErrorWrapper(
                        cpp_demangle::error::Error::UnexpectedEnd,
                    ))
                }),
                PjrtErrorCode::InvalidArgument,
            ),
            (
                LibTpuError::from(std::io::Error::from_raw_os_error(2)),
                PjrtErrorCode::Unavailable,
            ),
            (
                LibTpuError::from(rustix::io::Errno::INVAL),
                PjrtErrorCode::Unavailable,
            ),
            (
                LibTpuError::BinaryFormatError {
                    reason: String::from("not an ELF64 binary"),
                },
                PjrtErrorCode::InvalidArgument,
            ),
            (
                LibTpuError::UnimplementedOperation {
                    operation: "custom",
                },
                PjrtErrorCode::Unimplemented,
            ),
            (
                LibTpuError::UnsupportedGeneration {
                    raw_value: String::from("tpu_v99"),
                },
                PjrtErrorCode::InvalidArgument,
            ),
            (
                LibTpuError::BufferShapeMismatch {
                    expected_bytes: 256,
                    actual_bytes: 128,
                },
                PjrtErrorCode::InvalidArgument,
            ),
            (
                LibTpuError::OutOfDeviceMemory {
                    requested: 1024,
                    available: 512,
                    memory_kind: "vmem",
                },
                PjrtErrorCode::ResourceExhausted,
            ),
            (
                LibTpuError::InvalidDeviceAddress {
                    address: 0xDEAD_BEEF,
                },
                PjrtErrorCode::InvalidArgument,
            ),
            (
                LibTpuError::SysfsDiscoveryFailed {
                    path: String::from("/sys/class/accel"),
                    reason: String::from("ENOENT"),
                },
                PjrtErrorCode::Unavailable,
            ),
            (
                LibTpuError::HardwareFault {
                    device_id: 0,
                    status_csr: 0x04,
                    fault_description: "HBM_ECC_ERROR",
                },
                PjrtErrorCode::Internal,
            ),
            (
                LibTpuError::MlirParseError {
                    line: 10,
                    column: 4,
                    reason: String::from("unexpected token"),
                },
                PjrtErrorCode::InvalidArgument,
            ),
            (
                LibTpuError::MlirBytecodeDecodeError {
                    offset: 64,
                    section_id: 4,
                    reason: "truncated section",
                },
                PjrtErrorCode::InvalidArgument,
            ),
            (
                LibTpuError::HloVerificationError {
                    instruction_name: String::from("dot.1"),
                    reason: String::from("contracting dim mismatch"),
                },
                PjrtErrorCode::InvalidArgument,
            ),
            (
                LibTpuError::LayoutAssignmentError {
                    shape_summary: String::from("f32[8,128]"),
                    reason: "invalid minor_to_major permutation",
                },
                PjrtErrorCode::InvalidArgument,
            ),
            (
                LibTpuError::ElfInspectionError {
                    reason: String::from("missing .dynsym"),
                },
                PjrtErrorCode::InvalidArgument,
            ),
            (
                LibTpuError::BrotliDecompressionError {
                    resource_name: String::from("vfc_pf_csrs.br"),
                    reason: String::from("corrupt stream"),
                },
                PjrtErrorCode::InvalidArgument,
            ),
            (
                LibTpuError::IoError {
                    operation: "mmap",
                    reason: String::from("EPERM"),
                },
                PjrtErrorCode::Unavailable,
            ),
            (
                prost_reflect::DescriptorPool::decode(&b"\x80"[..]).map_or_else(
                    LibTpuError::from,
                    |_| {
                        LibTpuError::ProstReflectDescriptorFailed(ProstReflectErrorWrapper(
                            String::from("bad pool"),
                        ))
                    },
                ),
                PjrtErrorCode::InvalidArgument,
            ),
            (
                LibTpuError::SyncFlagTimeout {
                    device_id: 0,
                    sync_flag_id: 3,
                    expected_value: 2,
                    actual_value: 1,
                },
                PjrtErrorCode::DeadlineExceeded,
            ),
            (
                LibTpuError::HostCommandError {
                    opcode: 3,
                    continuation_id: 99,
                    reason: String::from("callback failed"),
                },
                PjrtErrorCode::Internal,
            ),
            (
                LibTpuError::RegisterAllocationFailed {
                    reg_class: "VREG",
                    requested: 128,
                    capacity: 64,
                },
                PjrtErrorCode::ResourceExhausted,
            ),
            (
                LibTpuError::FfiBoundaryPanic {
                    symbol: "GetPjrtApi",
                    detail: String::from("unexpected panic"),
                },
                PjrtErrorCode::Internal,
            ),
        ];

        for (err, expected_code) in variants {
            let cloned = err.clone();
            assert_eq!(cloned.error_code(), expected_code);
            assert_eq!(cloned.pjrt_code(), expected_code);
            assert_eq!(cloned.pjrt_error_code(), expected_code);
            assert_eq!(cloned.abseil_code(), expected_code);
            assert_eq!(cloned.abseil_status_code(), expected_code);
            assert_eq!(
                cloned.sdk_error_code(),
                LibtpuSdkErrorCode::from_pjrt_code(expected_code)
            );
            assert_eq!(
                cloned.sdk_code(),
                LibtpuSdkErrorCode::from_pjrt_code(expected_code)
            );
            let payloads = cloned.payloads();
            assert!(payloads.len() >= 2);
            assert_eq!(
                payloads.first().map(|p| p.key.as_str()),
                Some("type.googleapis.com/xla.TpuErrorPayload")
            );
            assert_eq!(
                payloads.get(1).map(|p| p.key.as_str()),
                Some("type.googleapis.com/xla.StatusCode")
            );
            let _ = cloned.is_predetermined_error();
            let _ = cloned.is_cancellation_placeholder_error();
            let cloned_payload = payloads.clone();
            assert_eq!(payloads, cloned_payload);
            assert_ne!(format!("{payloads:?}"), "");
            assert_ne!(cloned.to_string(), "");
            assert_ne!(format!("{cloned:?}"), "");
            assert!(cloned.source().is_none());
        }
    }
}

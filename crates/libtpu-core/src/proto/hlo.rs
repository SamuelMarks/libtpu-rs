//! XLA High-Level Optimizer (HLO) IR and buffer assignment protobuf messages.
//!
//! Synchronized with `xla/service/hlo.proto`, `xla/xla.proto`, and `xla/service/buffer_assignment.proto`.

use crate::proto::descriptor::impl_proto_schema_message;
use crate::proto::xla_data::{ProgramShapeProto, ShapeProto};
use prost::Message;

/// Wire-compatible `xla.HloInstructionProto` (`xla/service/hlo.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct HloInstructionProto {
    /// Instruction name (field `1`).
    #[prost(string, tag = "1")]
    pub name: String,
    /// Instruction opcode string (e.g., `"add"`, `"multiply"`, `"dot"`, `"parameter"`, field `2`).
    #[prost(string, tag = "2")]
    pub opcode: String,
    /// Output shape of this instruction (field `3`).
    #[prost(message, optional, tag = "3")]
    pub shape: Option<ShapeProto>,
    /// Parameter number for `opcode == "parameter"` (field `9`).
    #[prost(int64, tag = "9")]
    pub parameter_number: i64,
    /// Custom call target name (field `28`).
    #[prost(string, tag = "28")]
    pub custom_call_target: String,
    /// Unique instruction ID within the module (field `35`).
    #[prost(int64, tag = "35")]
    pub id: i64,
    /// Operand instruction IDs (field `36`).
    #[prost(int64, repeated, packed = "true", tag = "36")]
    pub operand_ids: Vec<i64>,
}

impl_proto_schema_message!(HloInstructionProto, "xla.HloInstructionProto");

/// Wire-compatible `xla.HloComputationProto` (`xla/service/hlo.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct HloComputationProto {
    /// Computation name (field `1`).
    #[prost(string, tag = "1")]
    pub name: String,
    /// Instructions in topological order (field `2`).
    #[prost(message, repeated, tag = "2")]
    pub instructions: Vec<HloInstructionProto>,
    /// Optional program shape of this computation (field `4`).
    #[prost(message, optional, tag = "4")]
    pub program_shape: Option<ProgramShapeProto>,
    /// Computation ID (field `5`).
    #[prost(int64, tag = "5")]
    pub id: i64,
    /// Root instruction ID (field `6`).
    #[prost(int64, tag = "6")]
    pub root_id: i64,
}

impl_proto_schema_message!(HloComputationProto, "xla.HloComputationProto");

/// Wire-compatible `xla.HloModuleProto` (`xla/service/hlo.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct HloModuleProto {
    /// Module name (field `1`).
    #[prost(string, tag = "1")]
    pub name: String,
    /// Entry computation name (field `2`).
    #[prost(string, tag = "2")]
    pub entry_computation_name: String,
    /// Computations belonging to this module (field `3`).
    #[prost(message, repeated, tag = "3")]
    pub computations: Vec<HloComputationProto>,
    /// Optional host program shape (field `4`).
    #[prost(message, optional, tag = "4")]
    pub host_program_shape: Option<ProgramShapeProto>,
    /// Module ID (field `5`).
    #[prost(int64, tag = "5")]
    pub id: i64,
    /// Entry computation ID (field `6`).
    #[prost(int64, tag = "6")]
    pub entry_computation_id: i64,
}

impl_proto_schema_message!(HloModuleProto, "xla.HloModuleProto");

/// Wire-compatible `xla.HloModuleConfigProto` (`xla/xla.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct HloModuleConfigProto {
    /// Entry computation parameter and result layout (field `1`).
    #[prost(message, optional, tag = "1")]
    pub entry_computation_layout: Option<ProgramShapeProto>,
    /// Deterministic PRNG compilation seed (field `2`).
    #[prost(uint64, tag = "2")]
    pub seed: u64,
    /// Number of data-parallel replicas (field `4`).
    #[prost(int64, tag = "4")]
    pub replica_count: i64,
    /// Number of SPMD partitions (field `5`).
    #[prost(int64, tag = "5")]
    pub num_partitions: i64,
    /// Whether SPMD partitioning is enabled (field `7`).
    #[prost(bool, tag = "7")]
    pub use_spmd_partitioning: bool,
}

impl_proto_schema_message!(HloModuleConfigProto, "xla.HloModuleConfigProto");

/// Wire-compatible `xla.HloModuleProtoWithConfig` (`xla/xla.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct HloModuleProtoWithConfig {
    /// Embedded `xla.HloModuleProto` (field `1`).
    #[prost(message, optional, tag = "1")]
    pub hlo_module: Option<HloModuleProto>,
    /// Embedded `xla.HloModuleConfigProto` config (field `2`).
    #[prost(message, optional, tag = "2")]
    pub config: Option<HloModuleConfigProto>,
}

impl_proto_schema_message!(HloModuleProtoWithConfig, "xla.HloModuleProtoWithConfig");

/// Wire-compatible `xla.buffer_assignment.BufferAllocationSliceProto` (`xla/service/buffer_assignment.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct BufferAllocationSliceProto {
    /// Byte offset within the parent `BufferAllocationProto` (field `1`).
    #[prost(int64, tag = "1")]
    pub offset: i64,
    /// Size of the slice in bytes (field `2`).
    #[prost(int64, tag = "2")]
    pub size: i64,
    /// Parent allocation index (field `3`).
    #[prost(int64, tag = "3")]
    pub buffer_allocation_index: i64,
    /// Primitive element type (`xla.PrimitiveType`, field `4`).
    #[prost(int32, tag = "4")]
    pub element_type: i32,
}

impl_proto_schema_message!(
    BufferAllocationSliceProto,
    "xla.buffer_assignment.BufferAllocationSliceProto"
);

/// Wire-compatible `xla.BufferAllocationProto` (`xla/service/hlo.proto`).
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, PartialEq, Eq, Message)]
pub struct BufferAllocationProto {
    /// Allocation index (field `1`).
    #[prost(int64, tag = "1")]
    pub index: i64,
    /// Total size in bytes (field `2`).
    #[prost(int64, tag = "2")]
    pub size: i64,
    /// Whether this allocation is a thread-local buffer (field `3`).
    #[prost(bool, tag = "3")]
    pub is_thread_local: bool,
    /// Whether this allocation represents an entry computation parameter (field `5`).
    #[prost(bool, tag = "5")]
    pub is_entry_computation_parameter: bool,
    /// Parameter number if `is_entry_computation_parameter` is true (field `6`).
    #[prost(int64, tag = "6")]
    pub parameter_number: i64,
    /// Whether this allocation is live out of the module (output, field `7`).
    #[prost(bool, tag = "7")]
    pub maybe_live_out: bool,
    /// Memory space color (`0` = HBM, `1` = VMEM, `2` = SMEM, `3` = CMEM, `4` = `HostPinned`, field `8`).
    #[prost(int64, tag = "8")]
    pub color: i64,
    /// Assigned slices within this allocation (field `9`).
    #[prost(message, repeated, tag = "9")]
    pub assigned: Vec<BufferAllocationSliceProto>,
    /// Whether this allocation is a tuple buffer (field `11`).
    #[prost(bool, tag = "11")]
    pub is_tuple: bool,
    /// Whether this allocation is a constant buffer (field `12`).
    #[prost(bool, tag = "12")]
    pub is_constant: bool,
}

impl_proto_schema_message!(BufferAllocationProto, "xla.BufferAllocationProto");

/// Wire-compatible `xla.BufferAssignmentProto` (`xla/service/hlo.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct BufferAssignmentProto {
    /// Physical buffer allocations across HBM, VMEM, SMEM, and CMEM (field `3`).
    #[prost(message, repeated, tag = "3")]
    pub buffer_allocations: Vec<BufferAllocationProto>,
}

impl_proto_schema_message!(BufferAssignmentProto, "xla.BufferAssignmentProto");

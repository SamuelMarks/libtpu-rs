//! Pure-Rust, zero-panic Protobuf wire-format encoder and decoder for `libtpu-rs` powered by `prost`.
//!
//! Implements wire-compatible binary serialization and deserialization matching the
//! canonical `OpenXLA`, TensorFlow, and embedded `libtpu.so` protobuf schemas:
//! - `xla.ShapeProto`, `xla.LayoutProto`, `xla.TileProto`, `xla.ProgramShapeProto`, `xla.OpSharding`
//! - `xla.HloInstructionProto`, `xla.HloComputationProto`, `xla.HloModuleProto`, `xla.HloModuleConfigProto`, `xla.HloModuleProtoWithConfig`
//! - `xla.OptionOverrideProto`, `xla.ExecutableBuildOptionsProto`, `xla.CompileOptionsProto`, `xla.ExecuteOptionsProto`
//! - `xla.CompiledMemoryStatsProto`, `xla.PjRtExecutableMetadata`, `xla.PjRtPartialProgramProto`, `xla.PjRtTopologyDescriptionProto`
//! - `xla.PjRtRuntimeAbiVersionProto`, `xla.PjRtExecutableAbiVersionProto`
//! - `tpu.TpuTopologyArgsProto`, `tpu.TpuCompilationEnvironmentProto`, `tpu.TpuSerializedExecutableProto`, `tpu.CompilationResultProto`, `tensorflow.tpu.TopologyProto`, `tensorflow.tpu.TPUCompileMetadataProto`, `tensorflow.tpu.SparseCoreTableLayouts`
//! - `tensorflow.profiler.XSpace`, `XPlane`, `XLine`, `XEvent`, `XEventMetadata`, `XStat`, `XStatMetadata`
//! - `tensorflow.tpu.TPUEmbeddingConfiguration`, `TableDescriptor`, `FeatureDescriptor`, `SpmdSharding`

use crate::error::{LibTpuError, LibTpuResult};
use num_enum::{FromPrimitive, IntoPrimitive};
use prost::Message;
use prost_reflect::{DescriptorPool, DynamicMessage};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::sync::OnceLock;

/// Raw wire-encoded `google.protobuf.FileDescriptorSet` compiled at build time by `protox` + `prost-build`
/// from `docs/reverse_engineering/protos/*.proto`.
pub const EMBEDDED_FILE_DESCRIPTOR_SET_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/libtpu_file_descriptor_set.bin"));

/// Global cached [`DescriptorPool`] decoded from [`EMBEDDED_FILE_DESCRIPTOR_SET_BYTES`].
static GLOBAL_DESCRIPTOR_POOL: OnceLock<DescriptorPool> = OnceLock::new();

/// Returns a reference to the global [`DescriptorPool`] compiled from `docs/reverse_engineering/protos/*.proto`.
///
/// # Errors
///
/// Returns [`LibTpuError::ProstReflectDescriptorFailed`] if the descriptor pool fails to decode.
pub fn global_descriptor_pool() -> LibTpuResult<&'static DescriptorPool> {
    if let Some(pool) = GLOBAL_DESCRIPTOR_POOL.get() {
        return Ok(pool);
    }
    let pool = DescriptorPool::decode(EMBEDDED_FILE_DESCRIPTOR_SET_BYTES)?;
    Ok(GLOBAL_DESCRIPTOR_POOL.get_or_init(|| pool))
}

/// Decodes a dynamic protobuf message by its fully-qualified protobuf type name
/// (e.g., `"platforms.deepsea.jellyfish.isa.IsaProgramProto"` or `"xla.TargetArgumentsProto"`)
/// using [`global_descriptor_pool`].
///
/// # Arguments
///
/// * `full_message_name` - Fully-qualified protobuf message name (`package.MessageName`).
/// * `bytes` - Wire-encoded protobuf message payload.
///
/// # Errors
///
/// Returns [`LibTpuError::ProtobufDecodeError`] if the message descriptor is not found or wire decoding fails.
pub fn decode_dynamic_proto_message(
    full_message_name: &'static str,
    bytes: &[u8],
) -> LibTpuResult<DynamicMessage> {
    let pool = global_descriptor_pool()?;
    let Some(msg_desc) = pool.get_message_by_name(full_message_name) else {
        return Err(LibTpuError::ProtobufDecodeError {
            message_type: full_message_name,
            reason: format!("message descriptor '{full_message_name}' not found in DescriptorPool"),
        });
    };
    DynamicMessage::decode(msg_desc, bytes).map_err(|err| LibTpuError::ProtobufDecodeError {
        message_type: full_message_name,
        reason: err.to_string(),
    })
}

/// Transcodes a wire-encoded protobuf message into canonical Protobuf JSON via `prost_reflect` and `serde_json`.
///
/// # Arguments
///
/// * `full_message_name` - Fully-qualified protobuf message name (`package.MessageName`).
/// * `bytes` - Wire-encoded protobuf message payload.
///
/// # Errors
///
/// Returns [`LibTpuError`] if descriptor lookup or wire decoding fails.
pub fn transcode_proto_bytes_to_json(
    full_message_name: &'static str,
    bytes: &[u8],
) -> LibTpuResult<String> {
    let dyn_msg = decode_dynamic_proto_message(full_message_name, bytes)?;
    Ok(serde_json::to_string(&dyn_msg).unwrap_or_default())
}

/// Decodes a `prost::Message` type `M` from `bytes`, mapping any wire error into
/// [`LibTpuError::ProtobufDecodeError`].
///
/// # Arguments
///
/// * `bytes` - Raw binary protobuf slice.
/// * `message_type` - Canonical protobuf message name for diagnostics.
///
/// # Errors
///
/// Returns [`LibTpuError::ProtobufDecodeError`] if `prost::Message::decode` fails.
pub fn decode_prost_message<M: Message + Default>(
    bytes: &[u8],
    message_type: &'static str,
) -> LibTpuResult<M> {
    M::decode(bytes).map_err(|err| LibTpuError::ProtobufDecodeError {
        message_type,
        reason: err.to_string(),
    })
}

/// Computes a canonical 128-bit truncated SHA-256 fingerprint (32 lowercase hex characters)
/// over a serialized protobuf or executable byte slice using `sha2::Sha256`.
///
/// # Arguments
///
/// * `payload` - Input byte slice to hash.
#[must_use]
pub fn sha256_fingerprint_hex(payload: &[u8]) -> String {
    let digest = Sha256::digest(payload);
    let mut hex = String::with_capacity(32);
    for &b in digest.iter().take(16) {
        let hi = char::from_digit(u32::from(b >> 4), 16).unwrap_or('0');
        let lo = char::from_digit(u32::from(b & 0x0F), 16).unwrap_or('0');
        hex.push(hi);
        hex.push(lo);
    }
    hex
}

/// Computes a deterministic 64-bit unsigned integer fingerprint over `payload` using `sha2::Sha256`.
///
/// # Arguments
///
/// * `payload` - Input byte slice to hash.
#[must_use]
pub fn sha256_fingerprint_u64(payload: &[u8]) -> u64 {
    let digest = Sha256::digest(payload);
    let mut first8 = [0_u8; 8];
    for (dst, &src) in first8.iter_mut().zip(digest.iter().take(8)) {
        *dst = src;
    }
    u64::from_le_bytes(first8)
}

/// Wire-compatible `xla.TileProto` (`xla/xla_data.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TileProto {
    /// Tile dimensions (e.g., `[8, 128]`), field `1`.
    #[prost(int64, repeated, packed = "true", tag = "1")]
    pub dimensions: Vec<i64>,
}

impl TileProto {
    /// Encodes this [`TileProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`TileProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `TileProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "xla.TileProto")
    }
}

/// Wire-compatible `xla.LayoutProto` (`xla/xla_data.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct LayoutProto {
    /// Minor-to-major dimension ordering (field `1`).
    #[prost(int64, repeated, packed = "true", tag = "1")]
    pub minor_to_major: Vec<i64>,
    /// Tiles applied to the layout (field `2`).
    #[prost(message, repeated, tag = "2")]
    pub tiles: Vec<TileProto>,
    /// Alignment of tail padding in elements (field `3`).
    #[prost(int64, tag = "3")]
    pub tail_padding_alignment_in_elements: i64,
    /// Primitive type used for sparse indices (field `4`).
    #[prost(int32, tag = "4")]
    pub index_primitive_type: i32,
    /// Primitive type used for sparse pointers (field `5`).
    #[prost(int32, tag = "5")]
    pub pointer_primitive_type: i32,
    /// Memory space ID (`0` = default HBM, `1` = VMEM, `3` = pinned host, field `6`).
    #[prost(int64, tag = "6")]
    pub memory_space: i64,
    /// Element size in bits (field `7`).
    #[prost(int64, tag = "7")]
    pub element_size_in_bits: i64,
    /// Dynamic shape metadata prefix size in bytes (field `9`).
    #[prost(int64, tag = "9")]
    pub dynamic_shape_metadata_prefix_bytes: i64,
}

impl LayoutProto {
    /// Encodes this [`LayoutProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`LayoutProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `LayoutProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "xla.LayoutProto")
    }
}

/// Wire-compatible `xla.ShapeProto` (`xla/xla_data.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ShapeProto {
    /// Primitive element type (`PrimitiveType`, field `2`).
    #[prost(int32, tag = "2")]
    pub element_type: i32,
    /// Dimension extents (field `3`).
    #[prost(int64, repeated, packed = "true", tag = "3")]
    pub dimensions: Vec<i64>,
    /// Tuple subshapes if `element_type == TUPLE` (field `4`).
    #[prost(message, repeated, tag = "4")]
    pub tuple_shapes: Vec<Self>,
    /// Optional physical layout (field `5`).
    #[prost(message, optional, tag = "5")]
    pub layout: Option<LayoutProto>,
    /// Dynamic dimension flags (field `6`).
    #[prost(bool, repeated, packed = "false", tag = "6")]
    pub is_dynamic_dimension: Vec<bool>,
}

impl ShapeProto {
    /// Encodes this [`ShapeProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`ShapeProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `ShapeProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "xla.ShapeProto")
    }
}

/// Wire-compatible `xla.ProgramShapeProto` (`xla/xla_data.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ProgramShapeProto {
    /// Parameter shapes of the computation (field `1`).
    #[prost(message, repeated, tag = "1")]
    pub parameters: Vec<ShapeProto>,
    /// Result shape of the computation (field `2`).
    #[prost(message, optional, tag = "2")]
    pub result: Option<ShapeProto>,
    /// Parameter names (field `3`).
    #[prost(string, repeated, tag = "3")]
    pub parameter_names: Vec<String>,
}

impl ProgramShapeProto {
    /// Encodes this [`ProgramShapeProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`ProgramShapeProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `ProgramShapeProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "xla.ProgramShapeProto")
    }
}

/// Wire-compatible `xla.OpSharding` (`xla/xla_data.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct OpShardingProto {
    /// Sharding type (`0` = Replicated, `1` = Maximal, `2` = Tuple, `3` = Other/Tiled, `4` = Manual, field `1`).
    #[prost(int32, tag = "1")]
    pub sharding_type: i32,
    /// Shape for tuple shardings (field `2`).
    #[prost(message, optional, tag = "2")]
    pub tuple_shardings_shape: Option<ShapeProto>,
    /// Tile assignment dimensions (field `3`).
    #[prost(int64, repeated, packed = "true", tag = "3")]
    pub tile_assignment_dimensions: Vec<i64>,
    /// Tile assignment devices (field `4`).
    #[prost(int64, repeated, packed = "true", tag = "4")]
    pub tile_assignment_devices: Vec<i64>,
    /// Nested tuple shardings (field `5`).
    #[prost(message, repeated, tag = "5")]
    pub tuple_shardings: Vec<Self>,
    /// Whether the last tile dimension is replicated across devices (field `6`).
    #[prost(bool, tag = "6")]
    pub replicate_on_last_tile_dim: bool,
    /// Compact iota reshape dimensions (field `8`).
    #[prost(int64, repeated, packed = "true", tag = "8")]
    pub iota_reshape_dims: Vec<i64>,
    /// Compact iota transpose permutation (field `9`).
    #[prost(int32, repeated, packed = "true", tag = "9")]
    pub iota_transpose_perm: Vec<i32>,
}

impl OpShardingProto {
    /// Encodes this [`OpShardingProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes an [`OpShardingProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `OpSharding` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "xla.OpSharding")
    }
}

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
    /// Unique instruction ID within the module (field `7`).
    #[prost(int64, tag = "7")]
    pub id: i64,
    /// Operand instruction IDs (field `8`).
    #[prost(int64, repeated, packed = "true", tag = "8")]
    pub operand_ids: Vec<i64>,
    /// Parameter number for `opcode == "parameter"` (field `9`).
    #[prost(int64, tag = "9")]
    pub parameter_number: i64,
    /// Custom call target name (field `14`).
    #[prost(string, tag = "14")]
    pub custom_call_target: String,
}

impl HloInstructionProto {
    /// Encodes this [`HloInstructionProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes an [`HloInstructionProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `HloInstructionProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "xla.HloInstructionProto")
    }
}

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

impl HloComputationProto {
    /// Encodes this [`HloComputationProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes an [`HloComputationProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `HloComputationProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "xla.HloComputationProto")
    }
}

/// Wire-compatible `xla.HloModuleProto` (`xla/service/hlo.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct HloModuleProto {
    /// Module name (field `1`).
    #[prost(string, tag = "1")]
    pub name: String,
    /// Entry computation name (field `2`).
    #[prost(string, tag = "2")]
    pub entry_computation_name: String,
    /// Entry computation ID (field `3`).
    #[prost(int64, tag = "3")]
    pub entry_computation_id: i64,
    /// Computations belonging to this module (field `4`).
    #[prost(message, repeated, tag = "4")]
    pub computations: Vec<HloComputationProto>,
    /// Optional host program shape (field `5`).
    #[prost(message, optional, tag = "5")]
    pub host_program_shape: Option<ProgramShapeProto>,
    /// Module ID (field `6`).
    #[prost(int64, tag = "6")]
    pub id: i64,
}

impl HloModuleProto {
    /// Encodes this [`HloModuleProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes an [`HloModuleProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `HloModuleProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "xla.HloModuleProto")
    }
}

/// Wire-compatible `xla.HloModuleConfigProto` (`xla/service/hlo.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct HloModuleConfigProto {
    /// Entry computation parameter and result layout (field `1`).
    #[prost(message, optional, tag = "1")]
    pub entry_computation_layout: Option<ProgramShapeProto>,
    /// Deterministic PRNG compilation seed (field `2`).
    #[prost(uint64, tag = "2")]
    pub seed: u64,
    /// Number of data-parallel replicas (field `3`).
    #[prost(int64, tag = "3")]
    pub replica_count: i64,
    /// Number of SPMD partitions (field `4`).
    #[prost(int64, tag = "4")]
    pub num_partitions: i64,
    /// Whether SPMD partitioning is enabled (field `5`).
    #[prost(bool, tag = "5")]
    pub use_spmd_partitioning: bool,
}

impl HloModuleConfigProto {
    /// Encodes this [`HloModuleConfigProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes an [`HloModuleConfigProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `HloModuleConfigProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "xla.HloModuleConfigProto")
    }
}

/// Wire-compatible `xla.OptionOverrideProto` (`xla/pjrt/proto/compile_options.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct OptionOverrideProto {
    /// String option override (field `1`).
    #[prost(string, tag = "1")]
    pub string_field: String,
    /// Boolean option override (field `2`).
    #[prost(bool, tag = "2")]
    pub bool_field: bool,
    /// Integer option override (field `3`).
    #[prost(int64, tag = "3")]
    pub int_field: i64,
    /// Floating-point option override (field `4`).
    #[prost(double, tag = "4")]
    pub double_field: f64,
}

impl OptionOverrideProto {
    /// Encodes this [`OptionOverrideProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes an [`OptionOverrideProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `OptionOverrideProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "xla.OptionOverrideProto")
    }
}

/// Wire-compatible `xla.ExecutableBuildOptionsProto` (`xla/pjrt/proto/compile_options.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ExecutableBuildOptionsProto {
    /// Device ordinal (`-1` if unspecified, field `1`).
    #[prost(int64, tag = "1")]
    pub device_ordinal: i64,
    /// Result layout (field `2`).
    #[prost(message, optional, tag = "2")]
    pub result_layout: Option<ShapeProto>,
    /// Number of replicas (field `4`).
    #[prost(int64, tag = "4")]
    pub num_replicas: i64,
    /// Number of SPMD partitions (field `5`).
    #[prost(int64, tag = "5")]
    pub num_partitions: i64,
    /// Whether SPMD partitioning is enabled (field `6`).
    #[prost(bool, tag = "6")]
    pub use_spmd_partitioning: bool,
    /// Whether auto-SPMD partitioning is enabled (field `7`).
    #[prost(bool, tag = "7")]
    pub use_auto_spmd_partitioning: bool,
    /// Static device assignment (field `9`).
    #[prost(message, optional, tag = "9")]
    pub device_assignment: Option<DeviceAssignmentProto>,
    /// Per-output SPMD sharding propagation flags (field `12`).
    #[prost(bool, repeated, packed = "true", tag = "12")]
    pub allow_spmd_sharding_propagation_to_output: Vec<bool>,
    /// Per-parameter SPMD sharding propagation flags (field `18`).
    #[prost(bool, repeated, packed = "true", tag = "18")]
    pub allow_spmd_sharding_propagation_to_parameters: Vec<bool>,
    /// Process index in multi-host jobs (field `19`).
    #[prost(int64, tag = "19")]
    pub process_index: i64,
    /// Total process count in multi-host jobs (field `20`).
    #[prost(int64, tag = "20")]
    pub process_count: i64,
    /// Whether the Shardy partitioner is enabled (field `21`).
    #[prost(bool, tag = "21")]
    pub use_shardy_partitioner: bool,
}

impl ExecutableBuildOptionsProto {
    /// Encodes this [`ExecutableBuildOptionsProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes an [`ExecutableBuildOptionsProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `ExecutableBuildOptionsProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "xla.ExecutableBuildOptionsProto")
    }
}

/// Wire-compatible `xla.CompileOptionsProto` (`xla/pjrt/proto/compile_options.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct CompileOptionsProto {
    /// Argument layouts (field `1`).
    #[prost(message, repeated, tag = "1")]
    pub argument_layouts: Vec<ShapeProto>,
    /// Whether parameters are tupled (field `2`).
    #[prost(bool, tag = "2")]
    pub parameter_is_tupled_arguments: bool,
    /// Build options (field `3`).
    #[prost(message, optional, tag = "3")]
    pub executable_build_options: Option<ExecutableBuildOptionsProto>,
    /// Compile portable executable flag (field `4`).
    #[prost(bool, tag = "4")]
    pub compile_portable_executable: bool,
    /// Environment option overrides keyed by flag name (field `5`).
    #[prost(btree_map = "string, message", tag = "5")]
    pub env_option_overrides: BTreeMap<String, OptionOverrideProto>,
    /// Profile version (field `6`).
    #[prost(int64, tag = "6")]
    pub profile_version: i64,
    /// Serialized multi-slice configuration bytes (field `7`).
    #[prost(bytes = "vec", tag = "7")]
    pub serialized_multi_slice_config: Vec<u8>,
}

impl CompileOptionsProto {
    /// Encodes this [`CompileOptionsProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`CompileOptionsProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `CompileOptionsProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "xla.CompileOptionsProto")
    }
}

/// Execution mode enum (`xla.ExecutionModeProto`) for [`ExecuteOptionsProto`].
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, FromPrimitive, IntoPrimitive)]
pub enum ExecutionModeProto {
    /// Unspecified execution mode (`0`).
    #[default]
    Unspecified = 0,
    /// Default runtime execution mode (`1`).
    Default = 1,
    /// Synchronous host-blocking execution (`2`).
    Synchronous = 2,
    /// Asynchronous non-blocking execution (`3`).
    Asynchronous = 3,
}

/// Wire-compatible `xla.ExecuteOptionsProto` (`xla/pjrt/proto/execute_options.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ExecuteOptionsProto {
    /// Whether arguments are passed as a single tuple (field `1`).
    #[prost(bool, tag = "1")]
    pub arguments_are_tupled: bool,
    /// Whether tuple results should be untupled into individual buffers (field `2`).
    #[prost(bool, tag = "2")]
    pub untuple_result: bool,
    /// Launch identifier for multi-host coordination (field `3`).
    #[prost(int32, tag = "3")]
    pub launch_id: i32,
    /// Whether strict input shape checking is enforced (field `4`).
    #[prost(bool, tag = "4")]
    pub strict_shape_checking: bool,
    /// Execution mode (`ExecutionModeProto`, field `6`).
    #[prost(int32, tag = "6")]
    pub execution_mode: i32,
    /// Input parameter indices that must not be donated (field `7`).
    #[prost(int32, repeated, packed = "true", tag = "7")]
    pub non_donatable_input_indices: Vec<i32>,
    /// PRNG execution seed (field `9`).
    #[prost(int64, tag = "9")]
    pub seed: i64,
}

impl ExecuteOptionsProto {
    /// Encodes this [`ExecuteOptionsProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes an [`ExecuteOptionsProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `ExecuteOptionsProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "xla.ExecuteOptionsProto")
    }
}

/// Wire-compatible `xla.CompiledMemoryStatsProto` (`xla/pjrt/proto/executable_metadata.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct CompiledMemoryStatsProto {
    /// Device generated code size in bytes (field `1`).
    #[prost(int64, tag = "1")]
    pub generated_code_size_in_bytes: i64,
    /// Device argument buffer size in bytes (field `2`).
    #[prost(int64, tag = "2")]
    pub argument_size_in_bytes: i64,
    /// Device output buffer size in bytes (field `3`).
    #[prost(int64, tag = "3")]
    pub output_size_in_bytes: i64,
    /// Device aliased buffer size in bytes (field `4`).
    #[prost(int64, tag = "4")]
    pub alias_size_in_bytes: i64,
    /// Device temporary scratchpad size in bytes (field `5`).
    #[prost(int64, tag = "5")]
    pub temp_size_in_bytes: i64,
    /// Host generated code size in bytes (field `7`).
    #[prost(int64, tag = "7")]
    pub host_generated_code_size_in_bytes: i64,
    /// Host argument buffer size in bytes (field `8`).
    #[prost(int64, tag = "8")]
    pub host_argument_size_in_bytes: i64,
    /// Host output buffer size in bytes (field `9`).
    #[prost(int64, tag = "9")]
    pub host_output_size_in_bytes: i64,
    /// Host aliased buffer size in bytes (field `10`).
    #[prost(int64, tag = "10")]
    pub host_alias_size_in_bytes: i64,
    /// Host temporary buffer size in bytes (field `11`).
    #[prost(int64, tag = "11")]
    pub host_temp_size_in_bytes: i64,
    /// Serialized `BufferAssignmentProto` bytes (field `13`).
    #[prost(bytes = "vec", tag = "13")]
    pub serialized_buffer_assignment: Vec<u8>,
    /// Peak device HBM usage in bytes (field `14`).
    #[prost(int64, tag = "14")]
    pub peak_memory_in_bytes: i64,
    /// Total device footprint in bytes (field `15`).
    #[prost(int64, tag = "15")]
    pub total_size_in_bytes: i64,
    /// Total cumulative allocation bytes (field `16`).
    #[prost(int64, tag = "16")]
    pub total_allocation_bytes: i64,
    /// Indefinite/live allocation bytes (field `17`).
    #[prost(int64, tag = "17")]
    pub indefinite_allocations: i64,
    /// Peak unpadded heap bytes (field `18`).
    #[prost(int64, tag = "18")]
    pub peak_unpadded_heap_bytes: i64,
}

impl CompiledMemoryStatsProto {
    /// Encodes this [`CompiledMemoryStatsProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`CompiledMemoryStatsProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `CompiledMemoryStatsProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "xla.CompiledMemoryStatsProto")
    }
}

/// Wire-compatible `xla.PjRtExecutableMetadata` (`xla/pjrt/proto/executable_metadata.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct PjRtExecutableMetadataProto {
    /// Compiled memory footprint statistics (field `1`).
    #[prost(message, optional, tag = "1")]
    pub compiled_memory_stats: Option<CompiledMemoryStatsProto>,
    /// Platform-specific metadata payload (field `2`).
    #[prost(bytes = "vec", tag = "2")]
    pub platform_specific_metadata: Vec<u8>,
}

impl PjRtExecutableMetadataProto {
    /// Encodes this [`PjRtExecutableMetadataProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`PjRtExecutableMetadataProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `PjRtExecutableMetadata` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "xla.PjRtExecutableMetadata")
    }
}

/// Wire-compatible `xla.PjRtPartialProgramProto` (`xla/pjrt/proto/pjrt_partial_program.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct PjRtPartialProgramProto {
    /// Intermediate program payload bytes (field `1`).
    #[prost(bytes = "vec", tag = "1")]
    pub program: Vec<u8>,
    /// Program format identifier (e.g., `"mlir"`, `"hlo"`, `"llo"`, field `2`).
    #[prost(string, tag = "2")]
    pub program_format: String,
    /// Name of the compilation phase that produced this artifact (field `3`).
    #[prost(string, tag = "3")]
    pub producer_phase: String,
    /// Valid next consumer compilation phase names (field `4`).
    #[prost(string, repeated, tag = "4")]
    pub consumer_phases: Vec<String>,
    /// Schema/compiler version string (field `5`).
    #[prost(string, tag = "5")]
    pub version: String,
    /// Program or module name (field `6`).
    #[prost(string, tag = "6")]
    pub program_name: String,
}

impl PjRtPartialProgramProto {
    /// Encodes this [`PjRtPartialProgramProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`PjRtPartialProgramProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `PjRtPartialProgramProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "xla.PjRtPartialProgramProto")
    }
}

/// Wire-compatible `xla.PjRtTopologyDescriptionProto` (`xla/pjrt/proto/topology_description.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct PjRtTopologyDescriptionProto {
    /// Platform ID (field `1`).
    #[prost(uint64, tag = "1")]
    pub platform_id: u64,
    /// Platform name (e.g., `"tpu"`, field `2`).
    #[prost(string, tag = "2")]
    pub platform_name: String,
    /// Platform version string (field `3`).
    #[prost(string, tag = "3")]
    pub platform_version: String,
    /// Whether this topology represents a subslice (field `4`).
    #[prost(bool, tag = "4")]
    pub is_subslice_topology: bool,
    /// Serialized platform-specific topology proto (`TpuTopologyArgsProto`, field `9`).
    #[prost(bytes = "vec", tag = "9")]
    pub platform_specific_topology: Vec<u8>,
}

impl PjRtTopologyDescriptionProto {
    /// Encodes this [`PjRtTopologyDescriptionProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`PjRtTopologyDescriptionProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `PjRtTopologyDescriptionProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "xla.PjRtTopologyDescriptionProto")
    }
}

/// Wire-compatible `xla.DeviceAssignmentProto.ComputationDevice` (`xla/xla_data.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ComputationDeviceProto {
    /// Device IDs assigned to each replica of this computation (field `1`).
    #[prost(int64, repeated, packed = "true", tag = "1")]
    pub replica_device_ids: Vec<i64>,
}

impl ComputationDeviceProto {
    /// Encodes this [`ComputationDeviceProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`ComputationDeviceProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `ComputationDeviceProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "xla.ComputationDeviceProto")
    }
}

/// Wire-compatible `xla.DeviceAssignmentProto` (`xla/xla_data.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct DeviceAssignmentProto {
    /// Number of data-parallel replicas (field `1`).
    #[prost(int32, tag = "1")]
    pub replica_count: i32,
    /// Number of model-parallel computations/partitions (field `2`).
    #[prost(int32, tag = "2")]
    pub computation_count: i32,
    /// Per-computation replica device assignments (field `3`).
    #[prost(message, repeated, tag = "3")]
    pub computation_devices: Vec<ComputationDeviceProto>,
}

impl DeviceAssignmentProto {
    /// Encodes this [`DeviceAssignmentProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`DeviceAssignmentProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `DeviceAssignmentProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "xla.DeviceAssignmentProto")
    }
}

/// Wire-compatible `tpu.TpuCompilationEnvironmentProto` (`tpu_compilation_environment.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuCompilationEnvironmentProto {
    /// XLA TPU Megacore fusion toggle (field `1`).
    #[prost(bool, tag = "1")]
    pub xla_tpu_enable_megacore_fusion: bool,
    /// XLA TPU async collective fusion toggle (field `2`).
    #[prost(bool, tag = "2")]
    pub xla_tpu_enable_async_collective_fusion: bool,
    /// XLA TPU VMEM limit in KiB (field `3`).
    #[prost(int64, tag = "3")]
    pub xla_tpu_scoped_vmem_limit_kib: i64,
    /// Raw custom flag string overrides (field `4`).
    #[prost(string, tag = "4")]
    pub custom_flags: String,
}

impl TpuCompilationEnvironmentProto {
    /// Encodes this [`TpuCompilationEnvironmentProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`TpuCompilationEnvironmentProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `TpuCompilationEnvironmentProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "tpu.TpuCompilationEnvironmentProto")
    }
}

/// Wire-compatible `tensorflow.profiler.XStatMetadata` (`tsl/profiler/protobuf/xplane.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct XStatMetadataProto {
    /// Metadata ID (field `1`).
    #[prost(int64, tag = "1")]
    pub id: i64,
    /// Stat name (field `2`).
    #[prost(string, tag = "2")]
    pub name: String,
    /// Stat description (field `3`).
    #[prost(string, tag = "3")]
    pub description: String,
}

impl XStatMetadataProto {
    /// Encodes this [`XStatMetadataProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes an [`XStatMetadataProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `XStatMetadata` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "tensorflow.profiler.XStatMetadata")
    }
}

/// Wire-compatible `tensorflow.profiler.XEventMetadata` (`tsl/profiler/protobuf/xplane.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct XEventMetadataProto {
    /// Metadata ID (field `1`).
    #[prost(int64, tag = "1")]
    pub id: i64,
    /// Event name (field `2`).
    #[prost(string, tag = "2")]
    pub name: String,
    /// Display name (field `3`).
    #[prost(string, tag = "3")]
    pub display_name: String,
    /// Additional metadata string (field `4`).
    #[prost(string, tag = "4")]
    pub metadata: String,
}

impl XEventMetadataProto {
    /// Encodes this [`XEventMetadataProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes an [`XEventMetadataProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `XEventMetadata` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "tensorflow.profiler.XEventMetadata")
    }
}

/// Wire-compatible `tpu.TpuTopologyArgsProto` (`tpu_topology.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuTopologyArgsProto {
    /// TPU platform generation enum (`TpuVersionEnum`, field `1`).
    #[prost(int32, tag = "1")]
    pub version: i32,
    /// Number of hosts in the pod slice (field `2`).
    #[prost(int32, tag = "2")]
    pub num_hosts: i32,
    /// Number of chips per host (field `3`).
    #[prost(int32, tag = "3")]
    pub chips_per_host: i32,
    /// Mesh dimensions `[x, y, z]` (field `4`).
    #[prost(int64, repeated, packed = "true", tag = "4")]
    pub mesh_bounds: Vec<i64>,
    /// Platform version string (field `5`).
    #[prost(string, tag = "5")]
    pub platform_version: String,
    /// Deterministic topology fingerprint (field `6`).
    #[prost(string, tag = "6")]
    pub fingerprint: String,
}

impl TpuTopologyArgsProto {
    /// Encodes this [`TpuTopologyArgsProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`TpuTopologyArgsProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `TpuTopologyArgsProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "tpu.TpuTopologyArgsProto")
    }
}

/// Wire-compatible `tensorflow.tpu.TPUHardwareFeature` (`tensorflow/core/protobuf/tpu/topology.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuHardwareFeatureProto {
    /// Embedding feature version enum (field `1`).
    #[prost(int32, tag = "1")]
    pub embedding_feature: i32,
    /// Number of embedding (`SparseCore`) devices per chip (field `2`).
    #[prost(int32, tag = "2")]
    pub num_embedding_devices_per_chip: i32,
}

impl TpuHardwareFeatureProto {
    /// Encodes this [`TpuHardwareFeatureProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`TpuHardwareFeatureProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `TPUHardwareFeature` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "tensorflow.tpu.TPUHardwareFeature")
    }
}

/// Wire-compatible `tensorflow.tpu.TopologyProto` (`tensorflow/core/protobuf/tpu/topology.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TopologyProto {
    /// Mesh shape `[x, y, z, core]` (field `1`).
    #[prost(int32, repeated, packed = "true", tag = "1")]
    pub mesh_shape: Vec<i32>,
    /// Number of TensorFlow tasks/hosts in the cluster (field `2`).
    #[prost(int32, tag = "2")]
    pub num_tasks: i32,
    /// Number of TPU devices per task (field `3`).
    #[prost(int32, tag = "3")]
    pub num_tpu_devices_per_task: i32,
    /// Flattened `(x, y, z, core)` device coordinates (field `4`).
    #[prost(int32, repeated, packed = "true", tag = "4")]
    pub device_coordinates: Vec<i32>,
    /// Hardware feature capabilities (field `5`).
    #[prost(message, optional, tag = "5")]
    pub tpu_hardware_feature: Option<TpuHardwareFeatureProto>,
}

impl TopologyProto {
    /// Encodes this [`TopologyProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`TopologyProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `TopologyProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "tensorflow.tpu.TopologyProto")
    }
}

/// Wire-compatible `tpu.TpuSerializedExecutableProto` (`tpu_executable.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuSerializedExecutableProto {
    /// Executable name (field `1`).
    #[prost(string, tag = "1")]
    pub name: String,
    /// SHA-256 fingerprint hex string (field `2`).
    #[prost(string, tag = "2")]
    pub fingerprint: String,
    /// Number of replicas (field `3`).
    #[prost(int32, tag = "3")]
    pub num_replicas: i32,
    /// Number of partitions (field `4`).
    #[prost(int32, tag = "4")]
    pub num_partitions: i32,
    /// Encoded VLIW instruction bundle opcodes (field `5`).
    #[prost(int64, repeated, packed = "true", tag = "5")]
    pub instruction_opcodes: Vec<i64>,
    /// Output shape protos (field `6`).
    #[prost(message, repeated, tag = "6")]
    pub output_shapes: Vec<ShapeProto>,
    /// Serialized `CompileOptionsProto` (field `7`).
    #[prost(bytes = "vec", tag = "7")]
    pub compile_options_bytes: Vec<u8>,
    /// Serialized HLO / `StableHLO` module text or proto (field `8`).
    #[prost(bytes = "vec", tag = "8")]
    pub hlo_module_bytes: Vec<u8>,
}

/// Type alias for [`TpuSerializedExecutableProto`] (`TpuExecutableSerializedProto`).
pub type TpuExecutableSerializedProto = TpuSerializedExecutableProto;

impl TpuSerializedExecutableProto {
    /// Encodes this [`TpuSerializedExecutableProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`TpuSerializedExecutableProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `TpuSerializedExecutableProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "tpu.TpuSerializedExecutableProto")
    }
}

/// Wire-compatible `tensorflow.profiler.XStat` (`tsl/profiler/protobuf/xplane.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct XStatProto {
    /// Metadata ID (field `1`).
    #[prost(int64, tag = "1")]
    pub metadata_id: i64,
    /// Double stat value (field `2`).
    #[prost(double, tag = "2")]
    pub double_value: f64,
    /// Unsigned 64-bit integer stat value (field `3`).
    #[prost(uint64, tag = "3")]
    pub uint64_value: u64,
    /// Signed 64-bit integer stat value (field `4`).
    #[prost(int64, tag = "4")]
    pub int64_value: i64,
    /// String stat value (field `5`).
    #[prost(string, tag = "5")]
    pub str_value: String,
    /// Raw bytes stat value (field `6`).
    #[prost(bytes = "vec", tag = "6")]
    pub bytes_value: Vec<u8>,
    /// Referenced string metadata ID (field `7`).
    #[prost(uint64, tag = "7")]
    pub ref_value: u64,
}

impl XStatProto {
    /// Encodes this [`XStatProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes an [`XStatProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `XStat` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "tensorflow.profiler.XStat")
    }
}

/// Wire-compatible `tensorflow.profiler.XEvent` (`tsl/profiler/protobuf/xplane.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct XEventProto {
    /// Event metadata ID (field `1`).
    #[prost(int64, tag = "1")]
    pub metadata_id: i64,
    /// Offset in picoseconds from the line start timestamp (field `2`).
    #[prost(int64, tag = "2")]
    pub offset_ps: i64,
    /// Duration in picoseconds (field `3`).
    #[prost(int64, tag = "3")]
    pub duration_ps: i64,
    /// Number of aggregated event occurrences (field `4`).
    #[prost(int64, tag = "4")]
    pub num_occurrences: i64,
    /// Attached event statistics (field `5`).
    #[prost(message, repeated, tag = "5")]
    pub stats: Vec<XStatProto>,
}

impl XEventProto {
    /// Encodes this [`XEventProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes an [`XEventProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `XEvent` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "tensorflow.profiler.XEvent")
    }
}

/// Wire-compatible `tensorflow.profiler.XLine` (`tsl/profiler/protobuf/xplane.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct XLineProto {
    /// Line ID (field `1`).
    #[prost(int64, tag = "1")]
    pub id: i64,
    /// Display ID (field `2`).
    #[prost(int64, tag = "2")]
    pub display_id: i64,
    /// Line name (e.g., `"TensorCore 0"`, field `3`).
    #[prost(string, tag = "3")]
    pub name: String,
    /// Display name (field `4`).
    #[prost(string, tag = "4")]
    pub display_name: String,
    /// Start timestamp in nanoseconds (field `5`).
    #[prost(int64, tag = "5")]
    pub timestamp_ns: i64,
    /// Line duration in picoseconds (field `6`).
    #[prost(int64, tag = "6")]
    pub duration_ps: i64,
    /// Events recorded on this timeline (field `7`).
    #[prost(message, repeated, tag = "7")]
    pub events: Vec<XEventProto>,
}

impl XLineProto {
    /// Encodes this [`XLineProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes an [`XLineProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `XLine` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "tensorflow.profiler.XLine")
    }
}

/// Wire-compatible `tensorflow.profiler.XPlane` (`tsl/profiler/protobuf/xplane.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct XPlaneProto {
    /// Plane ID (field `1`).
    #[prost(int64, tag = "1")]
    pub id: i64,
    /// Plane name (e.g., `"/device:TPU:0"`, field `2`).
    #[prost(string, tag = "2")]
    pub name: String,
    /// Timelines within this plane (field `3`).
    #[prost(message, repeated, tag = "3")]
    pub lines: Vec<XLineProto>,
    /// Event metadata dictionary keyed by metadata ID (field `4`).
    #[prost(btree_map = "int64, message", tag = "4")]
    pub event_metadata: BTreeMap<i64, XEventMetadataProto>,
    /// Stat metadata dictionary keyed by metadata ID (field `5`).
    #[prost(btree_map = "int64, message", tag = "5")]
    pub stat_metadata: BTreeMap<i64, XStatMetadataProto>,
    /// Plane-level statistics (field `6`).
    #[prost(message, repeated, tag = "6")]
    pub stats: Vec<XStatProto>,
}

impl XPlaneProto {
    /// Encodes this [`XPlaneProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes an [`XPlaneProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `XPlane` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "tensorflow.profiler.XPlane")
    }
}

/// Wire-compatible `tensorflow.profiler.XSpace` (`tsl/profiler/protobuf/xplane.proto`) returned by `TpuProfiler_CollectData`.
#[derive(Clone, PartialEq, Message)]
pub struct XSpaceProto {
    /// Planes in the trace (field `1`).
    #[prost(message, repeated, tag = "1")]
    pub planes: Vec<XPlaneProto>,
    /// Errors recorded during profiling (field `2`).
    #[prost(string, repeated, tag = "2")]
    pub errors: Vec<String>,
    /// Warnings recorded during profiling (field `3`).
    #[prost(string, repeated, tag = "3")]
    pub warnings: Vec<String>,
    /// Hostnames participating in the trace (field `4`).
    #[prost(string, repeated, tag = "4")]
    pub hostnames: Vec<String>,
}

impl XSpaceProto {
    /// Encodes this [`XSpaceProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes an [`XSpaceProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `XSpace` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "tensorflow.profiler.XSpace")
    }
}

/// Wire-compatible `tensorflow.tpu.TPUEmbeddingConfiguration.TableDescriptor` (`xla/tpu/protobuf/tpu_embedding_configuration.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct EmbeddingTableDescriptorProto {
    /// Table name (field `1`).
    #[prost(string, tag = "1")]
    pub name: String,
    /// Vocabulary size (number of rows, field `2`).
    #[prost(int64, tag = "2")]
    pub vocabulary_size: i64,
    /// Embedding dimension width (field `3`).
    #[prost(int64, tag = "3")]
    pub dimension: i64,
    /// Number of features referencing this table (field `4`).
    #[prost(int64, tag = "4")]
    pub num_features: i64,
}

impl EmbeddingTableDescriptorProto {
    /// Encodes this [`EmbeddingTableDescriptorProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes an [`EmbeddingTableDescriptorProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `TableDescriptor` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "tensorflow.tpu.TableDescriptor")
    }
}

/// Wire-compatible `tensorflow.tpu.TPUEmbeddingConfiguration.FeatureDescriptor` (`xla/tpu/protobuf/tpu_embedding_configuration.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct EmbeddingFeatureDescriptorProto {
    /// Feature name (field `1`).
    #[prost(string, tag = "1")]
    pub name: String,
    /// Index of the corresponding table in `table_descriptor` (field `2`).
    #[prost(int32, tag = "2")]
    pub table_id: i32,
    /// Input activation shape dimensions (field `3`).
    #[prost(int32, repeated, packed = "true", tag = "3")]
    pub input_shape: Vec<i32>,
}

impl EmbeddingFeatureDescriptorProto {
    /// Encodes this [`EmbeddingFeatureDescriptorProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes an [`EmbeddingFeatureDescriptorProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `FeatureDescriptor` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "tensorflow.tpu.FeatureDescriptor")
    }
}

/// Wire-compatible `tensorflow.tpu.TPUEmbeddingConfiguration.SpmdSharding` (`xla/tpu/protobuf/tpu_embedding_configuration.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct EmbeddingSpmdShardingProto {
    /// Whether SPMD sharding is enabled (field `1`).
    #[prost(bool, tag = "1")]
    pub enabled: bool,
    /// Number of cores per replica in SPMD mode (field `2`).
    #[prost(int32, tag = "2")]
    pub num_cores_per_replica: i32,
}

impl EmbeddingSpmdShardingProto {
    /// Encodes this [`EmbeddingSpmdShardingProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes an [`EmbeddingSpmdShardingProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `SpmdSharding` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "tensorflow.tpu.SpmdSharding")
    }
}

/// Wire-compatible `tensorflow.tpu.TPUEmbeddingConfiguration` (`xla/tpu/protobuf/tpu_embedding_configuration.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuEmbeddingConfigurationProto {
    /// Table descriptors (field `1`).
    #[prost(message, repeated, tag = "1")]
    pub table_descriptor: Vec<EmbeddingTableDescriptorProto>,
    /// Mode (`1` = Inference, `2` = Training, `3` = `BackwardPassOnly`, field `2`).
    #[prost(int32, tag = "2")]
    pub mode: i32,
    /// Batch size per `TensorCore` (field `3`).
    #[prost(int32, tag = "3")]
    pub batch_size_per_tensor_core: i32,
    /// Number of hosts (field `4`).
    #[prost(int32, tag = "4")]
    pub num_hosts: i32,
    /// Number of `TensorCores` (field `5`).
    #[prost(int32, tag = "5")]
    pub num_tensor_cores: i32,
    /// Table sharding strategy (`0` = Unspecified, `1` = Mod, `2` = Div, field `6`).
    #[prost(int32, tag = "6")]
    pub sharding_strategy: i32,
    /// Whether `SparseCore` execution is pipelined with `TensorCore` (field `7`).
    #[prost(bool, tag = "7")]
    pub pipeline_execution_with_tensor_core: bool,
    /// Directory for embedding profile data dumps (field `9`).
    #[prost(string, tag = "9")]
    pub profile_data_directory: String,
    /// Feature descriptors (field `10`).
    #[prost(message, repeated, tag = "10")]
    pub feature_descriptor: Vec<EmbeddingFeatureDescriptorProto>,
    /// Optional SPMD sharding configuration (field `11`).
    #[prost(message, optional, tag = "11")]
    pub spmd_sharding: Option<EmbeddingSpmdShardingProto>,
}

impl TpuEmbeddingConfigurationProto {
    /// Encodes this [`TpuEmbeddingConfigurationProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`TpuEmbeddingConfigurationProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `TPUEmbeddingConfiguration` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "tensorflow.tpu.TPUEmbeddingConfiguration")
    }
}

/// Wire-compatible `tensorflow.tpu.SparseCoreTableLayout` (`tensorflow/core/tpu/kernels/sparse_core_layout.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct SparseCoreTableLayoutProto {
    /// Table name (field `1`).
    #[prost(string, tag = "1")]
    pub table_name: String,
    /// Logical shard dimensions `[vocab, dim]` (field `2`).
    #[prost(int64, repeated, packed = "true", tag = "2")]
    pub unsharded_shape: Vec<i64>,
    /// Physical padded shape `[padded_vocab, padded_dim]` (field `3`).
    #[prost(int64, repeated, packed = "true", tag = "3")]
    pub unsharded_padded_shape: Vec<i64>,
    /// Number of `SparseCore` shards across the slice (field `4`).
    #[prost(int32, tag = "4")]
    pub num_sparse_cores: i32,
    /// Number of partitions per chip (field `5`).
    #[prost(int32, tag = "5")]
    pub num_partitions: i32,
    /// Total rows per `SparseCore` shard (field `6`).
    #[prost(int32, tag = "6")]
    pub total_rows_per_sparse_core_shard: i32,
}

impl SparseCoreTableLayoutProto {
    /// Encodes this [`SparseCoreTableLayoutProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`SparseCoreTableLayoutProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `SparseCoreTableLayout` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "tensorflow.tpu.SparseCoreTableLayout")
    }
}

/// Wire-compatible `tensorflow.tpu.SparseCoreTableLayouts` (`tensorflow/core/tpu/kernels/sparse_core_layout.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct SparseCoreTableLayoutsProto {
    /// Per-table `SparseCore` layout specifications (field `1`).
    #[prost(message, repeated, tag = "1")]
    pub tables: Vec<SparseCoreTableLayoutProto>,
}

impl SparseCoreTableLayoutsProto {
    /// Encodes this [`SparseCoreTableLayoutsProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`SparseCoreTableLayoutsProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `SparseCoreTableLayouts` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "tensorflow.tpu.SparseCoreTableLayouts")
    }
}

/// Wire-compatible `tpu.CompilationResultProto` (`xla/tpu/protobuf/compilation_result.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct CompilationResultProto {
    /// Abseil status code (`0` = OK, field `1`).
    #[prost(int32, tag = "1")]
    pub status_code: i32,
    /// Error message when `status_code != 0` (field `2`).
    #[prost(string, tag = "2")]
    pub status_error_message: String,
    /// HLO protocols produced by compilation (field `3`).
    #[prost(message, repeated, tag = "3")]
    pub hlo_protos: Vec<HloModuleProtoWithConfig>,
}

impl CompilationResultProto {
    /// Encodes this [`CompilationResultProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`CompilationResultProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `CompilationResultProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "tpu.CompilationResultProto")
    }
}

/// Wire-compatible `tensorflow.tpu.TPUCompileMetadataProto` (`tensorflow/core/protobuf/tpu/compile_metadata.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuCompileMetadataProto {
    /// Number of replicas (field `3`).
    #[prost(int32, tag = "3")]
    pub num_replicas: i32,
    /// Number of cores per replica (field `4`).
    #[prost(int32, tag = "4")]
    pub num_cores_per_replica: i32,
    /// Static device assignment (field `6`).
    #[prost(message, optional, tag = "6")]
    pub device_assignment: Option<DeviceAssignmentProto>,
    /// Whether SPMD is enabled (field `15`).
    #[prost(bool, tag = "15")]
    pub use_spmd_for_xla_partitioning: bool,
}

impl TpuCompileMetadataProto {
    /// Encodes this [`TpuCompileMetadataProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`TpuCompileMetadataProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `TPUCompileMetadataProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "tensorflow.tpu.TPUCompileMetadataProto")
    }
}

/// Wire-compatible `xla.HloModuleProtoWithConfig` (`xla/service/hlo.proto`) pairing an [`HloModuleProto`] with compilation flags.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct HloModuleProtoWithConfig {
    /// Embedded `xla.HloModuleProto` (field `1`).
    #[prost(message, optional, tag = "1")]
    pub hlo_module: Option<HloModuleProto>,
    /// Embedded `xla.HloModuleConfigProto` config (field `2`).
    #[prost(message, optional, tag = "2")]
    pub config: Option<HloModuleConfigProto>,
}

impl HloModuleProtoWithConfig {
    /// Encodes this [`HloModuleProtoWithConfig`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes an [`HloModuleProtoWithConfig`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `HloModuleProtoWithConfig` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "xla.HloModuleProtoWithConfig")
    }
}

/// Wire-compatible `xla.PjRtRuntimeAbiVersionProto` (`xla/pjrt/proto/pjrt_abi_version.proto`) for `PJRT_AbiVersion_Extension`.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct RuntimeAbiVersionProto {
    /// Platform ID (`xla::PjRtRuntimeAbiVersionProto::platform`, field `1`).
    #[prost(uint64, tag = "1")]
    pub platform_id: u64,
    /// Runtime ABI version payload (`xla::PjRtRuntimeAbiVersionProto::version`, field `2`).
    #[prost(string, tag = "2")]
    pub version_string: String,
    /// PJRT API minor version (field `3`).
    #[prost(int32, tag = "3")]
    pub pjrt_api_minor: i32,
}

/// Type alias for [`RuntimeAbiVersionProto`] matching upstream `PjRtRuntimeAbiVersionProto`.
pub type PjRtRuntimeAbiVersionProto = RuntimeAbiVersionProto;

impl RuntimeAbiVersionProto {
    /// Encodes this [`RuntimeAbiVersionProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`RuntimeAbiVersionProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `RuntimeAbiVersionProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "xla.PjRtRuntimeAbiVersionProto")
    }
}

/// Wire-compatible `xla.PjRtExecutableAbiVersionProto` (`xla/pjrt/proto/pjrt_abi_version.proto`) for `PJRT_AbiVersion_Extension`.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ExecutableAbiVersionProto {
    /// Platform ID (`xla::PjRtExecutableAbiVersionProto::platform`, field `1`).
    #[prost(uint64, tag = "1")]
    pub platform_id: u64,
    /// Compiler ABI version payload (`xla::PjRtExecutableAbiVersionProto::version`, field `2`).
    #[prost(string, tag = "2")]
    pub compiler_version: String,
    /// Target TPU generation code (field `3`).
    #[prost(int32, tag = "3")]
    pub target_generation: i32,
}

/// Type alias for [`ExecutableAbiVersionProto`] matching upstream `PjRtExecutableAbiVersionProto`.
pub type PjRtExecutableAbiVersionProto = ExecutableAbiVersionProto;

impl ExecutableAbiVersionProto {
    /// Encodes this [`ExecutableAbiVersionProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes an [`ExecutableAbiVersionProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `ExecutableAbiVersionProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "xla.PjRtExecutableAbiVersionProto")
    }
}

/// Wire-compatible `tpu.TpuEmbeddingPartitionedConfigProto` produced by `TpuEmbeddingEngine_ExecutePartitioner`.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuEmbeddingPartitionedConfigProto {
    /// Number of `SparseCore` partitions across the slice (field `1`).
    #[prost(int32, tag = "1")]
    pub num_partitions: i32,
    /// Partitioned table names (field `2`).
    #[prost(string, repeated, tag = "2")]
    pub table_names: Vec<String>,
    /// Original serialized `TpuEmbeddingConfigurationProto` bytes (field `3`).
    #[prost(bytes = "vec", tag = "3")]
    pub raw_config_bytes: Vec<u8>,
}

impl TpuEmbeddingPartitionedConfigProto {
    /// Encodes this [`TpuEmbeddingPartitionedConfigProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`TpuEmbeddingPartitionedConfigProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `TpuEmbeddingPartitionedConfigProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "tpu.TpuEmbeddingPartitionedConfigProto")
    }
}

/// Wire-compatible `tpu.TpuEmbeddingMemoryConfigProto` used by `TpuEmbeddingEngine_ConfigureMemory`
/// and `TpuEmbeddingEngine_CollateMemory`.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuEmbeddingMemoryConfigProto {
    /// Total HBM bytes reserved for `SparseCore` lookup tables (field `1`).
    #[prost(uint64, tag = "1")]
    pub hbm_buffers_bytes: u64,
    /// Number of collated host memory segments (field `2`).
    #[prost(int32, tag = "2")]
    pub num_segments: i32,
    /// Collated per-host memory configuration blobs (field `3`).
    #[prost(bytes = "vec", repeated, tag = "3")]
    pub segment_blobs: Vec<Vec<u8>>,
}

impl TpuEmbeddingMemoryConfigProto {
    /// Encodes this [`TpuEmbeddingMemoryConfigProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`TpuEmbeddingMemoryConfigProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `TpuEmbeddingMemoryConfigProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "tpu.TpuEmbeddingMemoryConfigProto")
    }
}

/// Wire-compatible `tpu.TpuEmbeddingHostNetworkConfigProto` used by `TpuEmbeddingEngine_ConfigureHost`.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuEmbeddingHostNetworkConfigProto {
    /// Host ordinal index (field `1`).
    #[prost(int32, tag = "1")]
    pub host_id: i32,
    /// Network endpoint address (field `2`).
    #[prost(string, tag = "2")]
    pub endpoint_address: String,
    /// Raw partition configuration payload (field `3`).
    #[prost(bytes = "vec", tag = "3")]
    pub partition_payload: Vec<u8>,
}

impl TpuEmbeddingHostNetworkConfigProto {
    /// Encodes this [`TpuEmbeddingHostNetworkConfigProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`TpuEmbeddingHostNetworkConfigProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `TpuEmbeddingHostNetworkConfigProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "tpu.TpuEmbeddingHostNetworkConfigProto")
    }
}

/// Wire-compatible `megascale.MultiSliceConfigProto` for `PJRT_MultiSlice_Extension` and `PJRT_Megascale_Extension`.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct MegascaleMultiSliceConfigProto {
    /// Total number of slices in the multi-slice training job (field `1`).
    #[prost(int32, tag = "1")]
    pub num_slices: i32,
    /// Local slice identifier (`0..num_slices`, field `2`).
    #[prost(int32, tag = "2")]
    pub slice_id: i32,
    /// Number of logical TPU devices per slice (field `3`).
    #[prost(int32, tag = "3")]
    pub num_devices_per_slice: i32,
    /// Coordinator network endpoint address (field `4`).
    #[prost(string, tag = "4")]
    pub coordinator_address: String,
}

impl MegascaleMultiSliceConfigProto {
    /// Encodes this [`MegascaleMultiSliceConfigProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`MegascaleMultiSliceConfigProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `MultiSliceConfigProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "megascale.MultiSliceConfigProto")
    }
}

/// Wire-compatible `megascale.DCNTopologyProto` for `PJRT_Megascale_Extension`.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct MegascaleDcnTopologyProto {
    /// Number of DCN slices (field `1`).
    #[prost(int32, tag = "1")]
    pub num_slices: i32,
    /// Number of hosts per slice (field `2`).
    #[prost(int32, tag = "2")]
    pub num_hosts_per_slice: i32,
    /// Transport type (`0` = Default/TCP, `1` = `RoCEv2`, `2` = ICI, field `3`).
    #[prost(int32, tag = "3")]
    pub transport_type: i32,
}

impl MegascaleDcnTopologyProto {
    /// Encodes this [`MegascaleDcnTopologyProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`MegascaleDcnTopologyProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `DCNTopologyProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "megascale.DCNTopologyProto")
    }
}

/// Wire-compatible `megascale.RuntimeMetricsProto` for `PJRT_Megascale_Extension`.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct MegascaleRuntimeMetricsProto {
    /// Number of completed cross-slice barriers (field `1`).
    #[prost(uint64, tag = "1")]
    pub completed_barriers: u64,
    /// Total collective operations executed (field `2`).
    #[prost(uint64, tag = "2")]
    pub collective_ops_count: u64,
    /// Active Megascale contexts (field `3`).
    #[prost(int32, tag = "3")]
    pub active_contexts: i32,
}

impl MegascaleRuntimeMetricsProto {
    /// Encodes this [`MegascaleRuntimeMetricsProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`MegascaleRuntimeMetricsProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `RuntimeMetricsProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "megascale.RuntimeMetricsProto")
    }
}

/// Wire-compatible `xla.IsaBundleSlotProto` (`platforms/xla/service/jellyfish/isa_program.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct IsaBundleSlotProto {
    /// Execution unit kind (`1` = Scalar, `2` = Vector, `3` = Matrix, `4` = Unary, `5` = DMA, field `1`).
    #[prost(uint32, tag = "1")]
    pub unit_kind: u32,
    /// Architectural instruction opcode (`TpuOpCode`, field `2`).
    #[prost(uint32, tag = "2")]
    pub opcode: u32,
    /// Destination register index (field `3`).
    #[prost(uint32, tag = "3")]
    pub dst_reg: u32,
    /// First source register index (field `4`).
    #[prost(uint32, tag = "4")]
    pub src0_reg: u32,
    /// Second source register index (field `5`).
    #[prost(uint32, tag = "5")]
    pub src1_reg: u32,
    /// Predicate register index (`PREG[0..15]`, field `6`).
    #[prost(uint32, tag = "6")]
    pub predicate_reg: u32,
    /// Signed 64-bit immediate operand (field `7`).
    #[prost(int64, tag = "7")]
    pub immediate: i64,
}

impl IsaBundleSlotProto {
    /// Encodes this [`IsaBundleSlotProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes an [`IsaBundleSlotProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `IsaBundleSlotProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "xla.IsaBundleSlotProto")
    }
}

/// Wire-compatible `xla.IsaVliwBundleProto` (`platforms/xla/service/jellyfish/isa_program.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct IsaVliwBundleProto {
    /// Program counter bundle index (field `1`).
    #[prost(uint32, tag = "1")]
    pub pc_index: u32,
    /// Scalar slot 0 (`S0`, field `2`).
    #[prost(message, optional, tag = "2")]
    pub scalar_s0: Option<IsaBundleSlotProto>,
    /// Scalar slot 1 (`S1`, field `3`).
    #[prost(message, optional, tag = "3")]
    pub scalar_s1: Option<IsaBundleSlotProto>,
    /// Vector slot 0 (`V0`, field `4`).
    #[prost(message, optional, tag = "4")]
    pub vector_v0: Option<IsaBundleSlotProto>,
    /// Vector slot 1 (`V1`, field `5`).
    #[prost(message, optional, tag = "5")]
    pub vector_v1: Option<IsaBundleSlotProto>,
    /// Matrix unit slot 0 (`MXU0`, field `6`).
    #[prost(message, optional, tag = "6")]
    pub matrix_m0: Option<IsaBundleSlotProto>,
    /// Matrix unit slot 1 (`MXU1`, field `7`).
    #[prost(message, optional, tag = "7")]
    pub matrix_m1: Option<IsaBundleSlotProto>,
    /// Extended unary unit slot (`XLU` / `ERU`, field `8`).
    #[prost(message, optional, tag = "8")]
    pub unary_xlu: Option<IsaBundleSlotProto>,
    /// DMA / `SyncFlag` / ICI slot (`D0`, field `9`).
    #[prost(message, optional, tag = "9")]
    pub dma_d0: Option<IsaBundleSlotProto>,
    /// LLO instruction debug annotation (field `10`).
    #[prost(string, tag = "10")]
    pub llo_annotation: String,
}

impl IsaVliwBundleProto {
    /// Encodes this [`IsaVliwBundleProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes an [`IsaVliwBundleProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `IsaVliwBundleProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "xla.IsaVliwBundleProto")
    }
}

/// Wire-compatible `xla.IsaCoreProgramHeaderProto` (`platforms/xla/service/jellyfish/isa_program.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct IsaCoreProgramHeaderProto {
    /// Core index within the chip (field `1`).
    #[prost(uint32, tag = "1")]
    pub core_index: u32,
    /// Core type (`0` = `TensorCore`, `1` = `SparseCore`, field `2`).
    #[prost(uint32, tag = "2")]
    pub core_type: u32,
    /// Entrypoint program counter offset (field `3`).
    #[prost(uint64, tag = "3")]
    pub entry_pc: u64,
    /// Number of 64-byte VLIW instruction bundles (field `4`).
    #[prost(uint64, tag = "4")]
    pub bundle_count: u64,
    /// Raw packed 64-byte `TensorCoreVliwBundle` stream (field `5`).
    #[prost(bytes = "vec", tag = "5")]
    pub raw_vliw_stream: Vec<u8>,
    /// Structured per-slot bundle descriptors (field `6`).
    #[prost(message, repeated, tag = "6")]
    pub structured_bundles: Vec<IsaVliwBundleProto>,
}

impl IsaCoreProgramHeaderProto {
    /// Encodes this [`IsaCoreProgramHeaderProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes an [`IsaCoreProgramHeaderProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `IsaCoreProgramHeaderProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "xla.IsaCoreProgramHeaderProto")
    }
}

/// Wire-compatible `xla.IsaSymbolTableEntryProto` (`platforms/xla/service/jellyfish/isa_program.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct IsaSymbolTableEntryProto {
    /// Symbol name (field `1`).
    #[prost(string, tag = "1")]
    pub name: String,
    /// Program counter byte offset in IMEM (field `2`).
    #[prost(uint64, tag = "2")]
    pub pc_offset: u64,
    /// Symbol body size in bytes (field `3`).
    #[prost(uint64, tag = "3")]
    pub size_bytes: u64,
}

impl IsaSymbolTableEntryProto {
    /// Encodes this [`IsaSymbolTableEntryProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes an [`IsaSymbolTableEntryProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `IsaSymbolTableEntryProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "xla.IsaSymbolTableEntryProto")
    }
}

/// Wire-compatible `xla.IsaProgramProto` (`platforms/xla/service/jellyfish/isa_program.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct IsaProgramProto {
    /// ISA architecture family (`jxc`, `pxc`, `vfc`, `gfc`, `zfc`, field `1`).
    #[prost(string, tag = "1")]
    pub isa_architecture: String,
    /// ISA ABI version (`163` in `libtpu-0.0.50`, field `2`).
    #[prost(int64, tag = "2")]
    pub isa_abi_version: i64,
    /// Per-core compiled instruction programs (field `3`).
    #[prost(message, repeated, tag = "3")]
    pub core_programs: Vec<IsaCoreProgramHeaderProto>,
    /// IMEM symbol table entries (field `4`).
    #[prost(message, repeated, tag = "4")]
    pub symbols: Vec<IsaSymbolTableEntryProto>,
    /// Deterministic 64-bit ISA program fingerprint (field `5`).
    #[prost(uint64, tag = "5")]
    pub fingerprint: u64,
}

impl IsaProgramProto {
    /// Encodes this [`IsaProgramProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes an [`IsaProgramProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `IsaProgramProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "xla.IsaProgramProto")
    }
}

/// Wire-compatible `xla.TargetArgumentsProto` (`platforms/xla/service/jellyfish/target_arguments.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TargetArgumentsProto {
    /// Target TPU version discriminant (`tpu.TpuVersionProto`, field `1`).
    #[prost(int32, tag = "1")]
    pub tpu_version: i32,
    /// Chip variant codename (`jellyfish`, `pufferfish`, `viperfish`, `ghostfish`, `zebrafish`, field `2`).
    #[prost(string, tag = "2")]
    pub chip_variant: String,
    /// Core clock frequency in MHz (field `3`).
    #[prost(int64, tag = "3")]
    pub clock_rate_mhz: i64,
    /// Per-chip HBM capacity in bytes (field `4`).
    #[prost(int64, tag = "4")]
    pub hbm_capacity_bytes: i64,
    /// Per-core VMEM capacity in bytes (field `5`).
    #[prost(int64, tag = "5")]
    pub vmem_capacity_bytes: i64,
    /// Per-core SMEM capacity in bytes (field `6`).
    #[prost(int64, tag = "6")]
    pub smem_capacity_bytes: i64,
    /// Per-chip CMEM capacity in bytes (field `7`).
    #[prost(int64, tag = "7")]
    pub cmem_capacity_bytes: i64,
    /// Number of `TensorCores` per chip (field `8`).
    #[prost(int32, tag = "8")]
    pub tensor_cores_per_chip: i32,
    /// Number of `SparseCores` per chip (field `9`).
    #[prost(int32, tag = "9")]
    pub sparse_cores_per_chip: i32,
    /// Systolic array dimension (`128` or `256`, field `10`).
    #[prost(int32, tag = "10")]
    pub mxu_dimension: i32,
    /// Inter-Chip Interconnect bandwidth in bits per second (field `11`).
    #[prost(int64, tag = "11")]
    pub ici_bandwidth_bps: i64,
}

impl TargetArgumentsProto {
    /// Encodes this [`TargetArgumentsProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`TargetArgumentsProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `TargetArgumentsProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "xla.TargetArgumentsProto")
    }
}

/// Wire-compatible `xla.DeepseaOverlaySectionProto` (`platforms/xla/service/jellyfish/deepsea_executable.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct DeepseaOverlaySectionProto {
    /// Overlay identifier (`OverlayId`, field `1`).
    #[prost(uint32, tag = "1")]
    pub overlay_id: u32,
    /// Overlay symbol name (field `2`).
    #[prost(string, tag = "2")]
    pub symbol_name: String,
    /// Target IMEM byte offset (field `3`).
    #[prost(uint64, tag = "3")]
    pub imem_offset_bytes: u64,
    /// Overlay size in bytes (field `4`).
    #[prost(uint64, tag = "4")]
    pub byte_size: u64,
    /// Packed VLIW instruction bundle bytes (field `5`).
    #[prost(bytes = "vec", tag = "5")]
    pub encoded_bundles: Vec<u8>,
    /// Target core type (`0` = `TensorCore`, `1` = `SparseCore`, field `6`).
    #[prost(uint32, tag = "6")]
    pub core_type: u32,
}

impl DeepseaOverlaySectionProto {
    /// Encodes this [`DeepseaOverlaySectionProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`DeepseaOverlaySectionProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `DeepseaOverlaySectionProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "xla.DeepseaOverlaySectionProto")
    }
}

/// Wire-compatible `xla.DeepseaRelocationEntryProto` (`platforms/xla/service/jellyfish/deepsea_executable.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct DeepseaRelocationEntryProto {
    /// Byte offset of the VLIW bundle in IMEM (field `1`).
    #[prost(uint64, tag = "1")]
    pub bundle_offset_bytes: u64,
    /// VLIW slot index (`0..8`, field `2`).
    #[prost(uint32, tag = "2")]
    pub slot_index: u32,
    /// Buffer allocation index in `BufferAssignmentProto` (field `3`).
    #[prost(uint32, tag = "3")]
    pub buffer_index: u32,
    /// Signed byte addend within the target buffer (field `4`).
    #[prost(int64, tag = "4")]
    pub addend_bytes: i64,
    /// Target memory space (`0` = HBM, `1` = VMEM, `2` = SMEM, `4` = CMEM, field `5`).
    #[prost(uint32, tag = "5")]
    pub memory_space: u32,
}

impl DeepseaRelocationEntryProto {
    /// Encodes this [`DeepseaRelocationEntryProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`DeepseaRelocationEntryProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `DeepseaRelocationEntryProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "xla.DeepseaRelocationEntryProto")
    }
}

/// Wire-compatible `xla.DeepseaConstantPoolEntryProto` (`platforms/xla/service/jellyfish/deepsea_executable.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct DeepseaConstantPoolEntryProto {
    /// Constant pool identifier (field `1`).
    #[prost(uint32, tag = "1")]
    pub constant_id: u32,
    /// Target memory space (`0` = HBM, `1` = VMEM, `2` = SMEM, field `2`).
    #[prost(uint32, tag = "2")]
    pub memory_space: u32,
    /// Byte offset in the constant pool segment (field `3`).
    #[prost(uint64, tag = "3")]
    pub offset_bytes: u64,
    /// Raw constant literal payload (field `4`).
    #[prost(bytes = "vec", tag = "4")]
    pub payload: Vec<u8>,
}

impl DeepseaConstantPoolEntryProto {
    /// Encodes this [`DeepseaConstantPoolEntryProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`DeepseaConstantPoolEntryProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `DeepseaConstantPoolEntryProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "xla.DeepseaConstantPoolEntryProto")
    }
}

/// Wire-compatible `xla.DeepseaSyncFlagInitProto` (`platforms/xla/service/jellyfish/deepsea_executable.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct DeepseaSyncFlagInitProto {
    /// Hardware `SyncFlag` register index (`0..255`, field `1`).
    #[prost(uint32, tag = "1")]
    pub sync_flag_id: u32,
    /// Initial counter value before program launch (field `2`).
    #[prost(uint32, tag = "2")]
    pub initial_value: u32,
    /// Human-readable debug label (field `3`).
    #[prost(string, tag = "3")]
    pub debug_label: String,
}

impl DeepseaSyncFlagInitProto {
    /// Encodes this [`DeepseaSyncFlagInitProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`DeepseaSyncFlagInitProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `DeepseaSyncFlagInitProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "xla.DeepseaSyncFlagInitProto")
    }
}

/// Wire-compatible `tpu.TpuCoreProgramAbiProto` (`learning/45eac/tpu/runtime/tpu_version.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuCoreProgramAbiProto {
    /// Target `tpu.TpuVersionProto` enum value (field `1`).
    #[prost(int32, tag = "1")]
    pub version: i32,
    /// Hardware variant string (field `2`).
    #[prost(string, tag = "2")]
    pub variant: String,
    /// Chip configuration profile name (`"default"` or `"inference"`, field `3`).
    #[prost(string, tag = "3")]
    pub chip_config: String,
    /// Base runtime ABI version (`163` in `libtpu-0.0.50`, field `4`).
    #[prost(int64, tag = "4")]
    pub base_abi_version: i64,
    /// Hardware ABI version (`65537` or `65539`, field `5`).
    #[prost(int64, tag = "5")]
    pub hardware_abi_version: i64,
    /// Required `DeepSea` compiler SHA-256 hash (`00ef0b06...`, field `6`).
    #[prost(string, tag = "6")]
    pub required_compiler_hash: String,
}

impl TpuCoreProgramAbiProto {
    /// Encodes this [`TpuCoreProgramAbiProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`TpuCoreProgramAbiProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `TpuCoreProgramAbiProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "tpu.TpuCoreProgramAbiProto")
    }
}

/// Wire-compatible `tpu.TpuMemorySegmentProto.Range` (`learning/45eac/tpu/runtime/hal/tpu_core_program.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuMemorySegmentRangeProto {
    /// Byte offset within the initialized data blob (field `1`).
    #[prost(int64, tag = "1")]
    pub offset: i64,
    /// Size in bytes within the initialized data blob (field `2`).
    #[prost(int64, tag = "2")]
    pub size: i64,
}

impl TpuMemorySegmentRangeProto {
    /// Encodes this [`TpuMemorySegmentRangeProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`TpuMemorySegmentRangeProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `TpuMemorySegmentRangeProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "xla.TpuMemorySegmentRangeProto")
    }
}

/// Wire-compatible `tpu.TpuMemorySegmentProto` (`learning/45eac/tpu/runtime/hal/tpu_core_program.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuMemorySegmentProto {
    /// Segment type (`1` = `CODE`, `2` = `DATA`, `3` = `NOBITS`, field `1`).
    #[prost(int32, tag = "1")]
    pub segment_type: i32,
    /// Target memory space (`1` = `HBM`, `2` = `VMEM`, `3` = `SMEM`, `4` = `CMEM`, `5` = `IMEM`, field `2`).
    #[prost(int32, tag = "2")]
    pub memory_type: i32,
    /// Range within the parent segment set's `initialized_data` buffer (field `3`).
    #[prost(message, optional, tag = "3")]
    pub initialized_data_range: Option<TpuMemorySegmentRangeProto>,
    /// Total allocated segment size in bytes (field `4`).
    #[prost(int64, tag = "4")]
    pub segment_size_in_bytes: i64,
}

impl TpuMemorySegmentProto {
    /// Encodes this [`TpuMemorySegmentProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`TpuMemorySegmentProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `TpuMemorySegmentProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "xla.TpuMemorySegmentProto")
    }
}

/// Wire-compatible `tpu.TpuMemorySegmentSetProto` (`learning/45eac/tpu/runtime/hal/tpu_core_program.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuMemorySegmentSetProto {
    /// Target memory space (`1` = `HBM`, `2` = `VMEM`, `3` = `SMEM`, `4` = `CMEM`, `5` = `IMEM`, field `1`).
    #[prost(int32, tag = "1")]
    pub memory_type: i32,
    /// Individual code/data/nobits segments in this memory space (field `2`).
    #[prost(message, repeated, tag = "2")]
    pub segments: Vec<TpuMemorySegmentProto>,
    /// Concatenated initialized segment bytes (field `3`).
    #[prost(bytes = "vec", tag = "3")]
    pub initialized_data: Vec<u8>,
    /// Content hash digest of `initialized_data` (field `4`).
    #[prost(bytes = "vec", tag = "4")]
    pub hash: Vec<u8>,
    /// Compression algorithm (`0` = uncompressed, `1` = Brotli, field `5`).
    #[prost(int32, tag = "5")]
    pub compression: i32,
}

impl TpuMemorySegmentSetProto {
    /// Encodes this [`TpuMemorySegmentSetProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`TpuMemorySegmentSetProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `TpuMemorySegmentSetProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "xla.TpuMemorySegmentSetProto")
    }
}

/// Wire-compatible `tpu.TpuCoreProgramProto` (`learning/45eac/tpu/runtime/hal/tpu_core_program.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuCoreProgramProto {
    /// Target `tpu.TpuPlatformTypeProto` enum value (field `2`).
    #[prost(int32, tag = "2")]
    pub platform_type: i32,
    /// Binary program fingerprint (field `3`).
    #[prost(bytes = "vec", tag = "3")]
    pub fingerprint: Vec<u8>,
    /// Monotonic compilation identifier (field `4`).
    #[prost(uint64, tag = "4")]
    pub compilation_id: u64,
    /// Loadable HBM/VMEM/IMEM memory segment sets (field `8`).
    #[prost(message, repeated, tag = "8")]
    pub memory_segments: Vec<TpuMemorySegmentSetProto>,
    /// Concurrent program HBM stack size in bytes (field `9`).
    #[prost(int64, tag = "9")]
    pub concurrent_program_hbm_stack_size: i64,
    /// Whether this program includes a concurrent continuation program (field `10`).
    #[prost(bool, tag = "10")]
    pub has_concurrent_program: bool,
    /// Embedded `TpuCoreProgramAbiProto` descriptor (field `12`).
    #[prost(message, optional, tag = "12")]
    pub abi: Option<TpuCoreProgramAbiProto>,
    /// Whether an out-of-memory condition occurred during compilation (field `14`).
    #[prost(bool, tag = "14")]
    pub oom_during_compilation: bool,
}

impl TpuCoreProgramProto {
    /// Encodes this [`TpuCoreProgramProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`TpuCoreProgramProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `TpuCoreProgramProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "xla.TpuCoreProgramProto")
    }
}

/// Wire-compatible `xla.DeepseaExecutableProto` (`platforms/xla/service/jellyfish/deepsea_executable.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct DeepseaExecutableProto {
    /// Module name (field `1`).
    #[prost(string, tag = "1")]
    pub module_name: String,
    /// Compiled ISA program (`xla.IsaProgramProto`, field `2`).
    #[prost(message, optional, tag = "2")]
    pub isa_program: Option<IsaProgramProto>,
    /// IMEM and `SparseCore` instruction overlays (field `3`).
    #[prost(message, repeated, tag = "3")]
    pub overlays: Vec<DeepseaOverlaySectionProto>,
    /// HBM/VMEM/SMEM address relocations (field `4`).
    #[prost(message, repeated, tag = "4")]
    pub relocations: Vec<DeepseaRelocationEntryProto>,
    /// Constant pool literals (field `5`).
    #[prost(message, repeated, tag = "5")]
    pub constants: Vec<DeepseaConstantPoolEntryProto>,
    /// Hardware `SyncFlag` initializers (field `6`).
    #[prost(message, repeated, tag = "6")]
    pub sync_flags: Vec<DeepseaSyncFlagInitProto>,
    /// Hardware target arguments (`xla.TargetArgumentsProto`, field `7`).
    #[prost(message, optional, tag = "7")]
    pub target_arguments: Option<TargetArgumentsProto>,
    /// Scoped VMEM bytes allocated (field `8`).
    #[prost(int64, tag = "8")]
    pub vmem_allocation_bytes: i64,
    /// Scoped SMEM bytes allocated (field `9`).
    #[prost(int64, tag = "9")]
    pub smem_allocation_bytes: i64,
    /// Scoped CMEM bytes allocated (field `10`).
    #[prost(int64, tag = "10")]
    pub cmem_allocation_bytes: i64,
    /// HBM scratchpad bytes allocated (field `11`).
    #[prost(int64, tag = "11")]
    pub hbm_scratchpad_bytes: i64,
    /// Authentic `tpu.TpuCoreProgramProto` for `TensorCore` (field `12`).
    #[prost(message, optional, tag = "12")]
    pub core_program: Option<TpuCoreProgramProto>,
    /// Authentic `tpu.TpuCoreProgramProto` for `SparseCore` (field `13`).
    #[prost(message, optional, tag = "13")]
    pub sparse_core_program: Option<TpuCoreProgramProto>,
}

impl DeepseaExecutableProto {
    /// Encodes this [`DeepseaExecutableProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`DeepseaExecutableProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `DeepseaExecutableProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "xla.DeepseaExecutableProto")
    }
}

/// Wire-compatible `tpu.HostCommandEnvelopeProto` (`learning/45eac/tpu/runtime/host_commands/host_command_opcodes.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct HostCommandEnvelopeProto {
    /// Host command opcode (`tpu.HostCommandOpcodeProto`, field `1`).
    #[prost(int32, tag = "1")]
    pub opcode: i32,
    /// Continuation queue sequence ID (`ContinuationId`, field `2`).
    #[prost(uint64, tag = "2")]
    pub continuation_id: u64,
    /// Source device ID (field `3`).
    #[prost(uint32, tag = "3")]
    pub device_id: u32,
    /// Source core index (field `4`).
    #[prost(uint32, tag = "4")]
    pub core_index: u32,
    /// Registered callback or handler ID (field `5`).
    #[prost(uint64, tag = "5")]
    pub handler_id: u64,
    /// Serialized command payload bytes (field `6`).
    #[prost(bytes = "vec", tag = "6")]
    pub payload: Vec<u8>,
}

impl HostCommandEnvelopeProto {
    /// Encodes this [`HostCommandEnvelopeProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`HostCommandEnvelopeProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `HostCommandEnvelopeProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "tpu.HostCommandEnvelopeProto")
    }
}

/// Wire-compatible `xla.TpuExecutableProto` (`learning/45eac/research/pjrt/tpu_executable.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct TpuExecutableProto {
    /// Embedded `DeepseaExecutableProto` (field `1`).
    #[prost(message, optional, tag = "1")]
    pub ds_executable: Option<DeepseaExecutableProto>,
    /// Embedded `HloModuleProtoWithConfig` (field `3`).
    #[prost(message, optional, tag = "3")]
    pub hlo_module_with_config: Option<HloModuleProtoWithConfig>,
    /// Embedded `CompileOptionsProto` (field `6`).
    #[prost(message, optional, tag = "6")]
    pub compile_options: Option<CompileOptionsProto>,
    /// Embedded `TargetArgumentsProto` (field `7`).
    #[prost(message, optional, tag = "7")]
    pub target_arguments: Option<TargetArgumentsProto>,
    /// Legacy compatibility `TpuSerializedExecutableProto` payload (field `8`).
    #[prost(message, optional, tag = "8")]
    pub legacy_executable: Option<TpuSerializedExecutableProto>,
    /// Source URI of the compiled module (field `9`).
    #[prost(string, tag = "9")]
    pub source_uri: String,
}

impl TpuExecutableProto {
    /// Encodes this [`TpuExecutableProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`TpuExecutableProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `TpuExecutableProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "xla.TpuExecutableProto")
    }
}

/// Wire-compatible `xla.DeviceProto` (`xla/pjrt/distributed/protocol.proto`) describing a local or global device.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct DistributedDeviceProto {
    /// Local device ordinal on the host (field `1`).
    #[prost(int32, tag = "1")]
    pub local_device_ordinal: i32,
    /// Device name string (field `2`).
    #[prost(string, tag = "2")]
    pub name: String,
    /// Vendor string (e.g., `"Google"`, field `3`).
    #[prost(string, tag = "3")]
    pub vendor: String,
    /// Globally unique device ID assigned by the coordinator (field `4`).
    #[prost(int32, tag = "4")]
    pub global_device_id: i32,
    /// Fast-interconnect partition index (field `5`).
    #[prost(int32, tag = "5")]
    pub partition_index: i32,
    /// Compute capability string (field `6`).
    #[prost(string, tag = "6")]
    pub compute_capability: String,
    /// Core count per device (field `7`).
    #[prost(int32, tag = "7")]
    pub core_count: i32,
    /// Device kind string (e.g., `"Cloud TPU v5e"`, field `8`).
    #[prost(string, tag = "8")]
    pub device_kind: String,
    /// Human-readable `to_string` representation (field `9`).
    #[prost(string, tag = "9")]
    pub to_string: String,
    /// Human-readable `debug_string` representation (field `10`).
    #[prost(string, tag = "10")]
    pub debug_string: String,
    /// Fabric UUID (`"slice_uuid/clique_id"`, field `12`).
    #[prost(string, tag = "12")]
    pub fabric_uuid: String,
    /// NUMA node index (`-1` if unpinned, field `14`).
    #[prost(int32, tag = "14")]
    pub numa_node: i32,
    /// Device HBM memory limit in bytes (field `15`).
    #[prost(int64, tag = "15")]
    pub device_memory_bytes_limit: i64,
}

impl DistributedDeviceProto {
    /// Encodes this [`DistributedDeviceProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`DistributedDeviceProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `xla.DeviceProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "xla.DeviceProto")
    }
}

/// Wire-compatible `xla.LocalTopologyProto` (`xla/pjrt/distributed/protocol.proto`) published by each host.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct LocalTopologyProto {
    /// Communicating process index in `[0, num_processes)` (field `1`).
    #[prost(int32, tag = "1")]
    pub process_id: i32,
    /// Kernel boot UUID (`/proc/sys/kernel/random/boot_id`, field `2`).
    #[prost(string, tag = "2")]
    pub boot_id: String,
    /// Local devices controlled by `process_id` (field `3`).
    #[prost(message, repeated, tag = "3")]
    pub devices: Vec<DistributedDeviceProto>,
    /// Partition index (field `4`).
    #[prost(int32, tag = "4")]
    pub partition_index: i32,
    /// Ordered list of DCN switch hierarchy node IDs (field `5`).
    #[prost(string, repeated, tag = "5")]
    pub network_nodes: Vec<String>,
}

impl LocalTopologyProto {
    /// Encodes this [`LocalTopologyProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`LocalTopologyProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `xla.LocalTopologyProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "xla.LocalTopologyProto")
    }
}

/// Wire-compatible `xla.GlobalTopologyProto` (`xla/pjrt/distributed/protocol.proto`) broadcast by the coordinator.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct GlobalTopologyProto {
    /// Per-process topologies with globally assigned `global_device_id` ordinals (field `1`).
    #[prost(message, repeated, tag = "1")]
    pub processes: Vec<LocalTopologyProto>,
}

impl GlobalTopologyProto {
    /// Encodes this [`GlobalTopologyProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`GlobalTopologyProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `xla.GlobalTopologyProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "xla.GlobalTopologyProto")
    }
}

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

impl BufferAllocationSliceProto {
    /// Encodes this [`BufferAllocationSliceProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`BufferAllocationSliceProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `BufferAllocationSliceProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "xla.buffer_assignment.BufferAllocationSliceProto")
    }
}

/// Wire-compatible `xla.BufferAllocationProto` (`xla/service/buffer_assignment.proto`).
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
    /// Whether this allocation is a tuple buffer (field `4`).
    #[prost(bool, tag = "4")]
    pub is_tuple: bool,
    /// Whether this allocation represents an entry computation parameter (field `5`).
    #[prost(bool, tag = "5")]
    pub is_entry_computation_parameter: bool,
    /// Parameter number if `is_entry_computation_parameter` is true (field `6`).
    #[prost(int64, tag = "6")]
    pub parameter_number: i64,
    /// Whether this allocation is a constant buffer (field `7`).
    #[prost(bool, tag = "7")]
    pub is_constant: bool,
    /// Whether this allocation is live out of the module (output, field `8`).
    #[prost(bool, tag = "8")]
    pub maybe_live_out: bool,
    /// Memory space color (`0` = HBM, `1` = VMEM, `2` = SMEM, `3` = CMEM, `4` = `HostPinned`, field `9`).
    #[prost(int64, tag = "9")]
    pub color: i64,
    /// Assigned slices within this allocation (field `10`).
    #[prost(message, repeated, tag = "10")]
    pub assigned: Vec<BufferAllocationSliceProto>,
}

impl BufferAllocationProto {
    /// Encodes this [`BufferAllocationProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`BufferAllocationProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `BufferAllocationProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "xla.BufferAllocationProto")
    }
}

/// Wire-compatible `xla.BufferAssignmentProto` (`xla/service/buffer_assignment.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct BufferAssignmentProto {
    /// Physical buffer allocations across HBM, VMEM, SMEM, and CMEM (field `1`).
    #[prost(message, repeated, tag = "1")]
    pub buffer_allocations: Vec<BufferAllocationProto>,
}

impl BufferAssignmentProto {
    /// Encodes this [`BufferAssignmentProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`BufferAssignmentProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `BufferAssignmentProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, "xla.BufferAssignmentProto")
    }
}

/// Wire-compatible `xla.megascale.runtime.external.MegaScaleRuntimeError` (`xla/megascale/megascale_runtime_error_overlay.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct MegaScaleRuntimeErrorProto {
    /// Error type (`0` = `NO_ERROR`, `1` = `HANG_DETECTED`, `2` = `UNRECOVERABLE_ERROR`, `3` = `CANCELLED`, field `1`).
    #[prost(int32, tag = "1")]
    pub error_type: i32,
    /// Human-readable error message (field `2`).
    #[prost(string, tag = "2")]
    pub error_message: String,
    /// Estimated start time in milliseconds since epoch (field `3`).
    #[prost(uint64, tag = "3")]
    pub estimated_start_time_ms: u64,
    /// Triggering launch ID (field `5`).
    #[prost(int32, tag = "5")]
    pub triggering_launch_id: i32,
    /// Reporting hostname (field `6`).
    #[prost(string, tag = "6")]
    pub hostname: String,
    /// Reporting job name (field `8`).
    #[prost(string, tag = "8")]
    pub job_name: String,
    /// Task ordinal index (field `9`).
    #[prost(int32, tag = "9")]
    pub task_id: i32,
    /// Event identifier (field `11`).
    #[prost(fixed64, tag = "11")]
    pub event_id: u64,
    /// Unrecoverable error sub-type (`0` = Unclassified, `1` = H2D, `2` = D2H, field `12`).
    #[prost(int32, tag = "12")]
    pub unrecoverable_error_type: i32,
    /// Runtime error identifier (field `13`).
    #[prost(fixed64, tag = "13")]
    pub runtime_error_id: u64,
    /// Hang error sub-type (`0` = Unspecified, `1` = Timeout, `2` = `NoProgress`, field `14`).
    #[prost(int32, tag = "14")]
    pub hang_error_type: i32,
}

impl MegaScaleRuntimeErrorProto {
    /// Encodes this [`MegaScaleRuntimeErrorProto`] to protobuf wire bytes via `prost`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes a [`MegaScaleRuntimeErrorProto`] from protobuf wire bytes via `prost`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `MegaScaleRuntimeError` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(
            bytes,
            "xla.megascale.runtime.external.MegaScaleRuntimeError",
        )
    }
}

#[cfg(test)]
#[allow(clippy::cognitive_complexity, clippy::too_many_lines)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_sha256_fingerprint_helpers() {
        let fp_hex = sha256_fingerprint_hex(b"tpu_executable_payload");
        assert_eq!(fp_hex.len(), 32);
        assert!(fp_hex.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(fp_hex, sha256_fingerprint_hex(b"tpu_executable_payload"));
        assert_ne!(fp_hex, sha256_fingerprint_hex(b"different_payload"));

        let fp_u64 = sha256_fingerprint_u64(b"tpu_topology_v4");
        assert_ne!(fp_u64, 0);
        assert_eq!(fp_u64, sha256_fingerprint_u64(b"tpu_topology_v4"));
    }

    #[test]
    fn test_all_protobuf_messages_roundtrip_and_unknown_tags() {
        let extra = [0xF8_u8, 0x06, 0x01];

        let tile = TileProto {
            dimensions: vec![8, 128],
        };
        let mut tile_bytes = tile.encode();
        tile_bytes.extend_from_slice(&extra);
        assert_eq!(TileProto::decode(&tile_bytes).ok(), Some(tile.clone()));

        let layout = LayoutProto {
            minor_to_major: vec![1, 0],
            tiles: vec![tile],
            tail_padding_alignment_in_elements: 1,
            index_primitive_type: 0,
            pointer_primitive_type: 0,
            memory_space: 1,
            element_size_in_bits: 32,
            dynamic_shape_metadata_prefix_bytes: 0,
        };
        let mut layout_bytes = layout.encode();
        layout_bytes.extend_from_slice(&extra);
        assert_eq!(
            LayoutProto::decode(&layout_bytes).ok(),
            Some(layout.clone())
        );

        let sub_shape = ShapeProto {
            element_type: 11,
            dimensions: vec![4],
            tuple_shapes: vec![],
            layout: None,
            is_dynamic_dimension: vec![false],
        };
        let shape = ShapeProto {
            element_type: 11,
            dimensions: vec![8, 128],
            tuple_shapes: vec![sub_shape],
            layout: Some(layout),
            is_dynamic_dimension: vec![false, true],
        };
        let mut shape_bytes = shape.encode();
        shape_bytes.extend_from_slice(&extra);
        assert_eq!(ShapeProto::decode(&shape_bytes).ok(), Some(shape.clone()));

        let prog_shape = ProgramShapeProto {
            parameters: vec![shape.clone()],
            result: Some(shape.clone()),
            parameter_names: vec![String::from("p0")],
        };
        let mut prog_shape_bytes = prog_shape.encode();
        prog_shape_bytes.extend_from_slice(&extra);
        assert_eq!(
            ProgramShapeProto::decode(&prog_shape_bytes).ok(),
            Some(prog_shape.clone())
        );

        let sharding = OpShardingProto {
            sharding_type: 3,
            tuple_shardings_shape: None,
            tile_assignment_dimensions: vec![2, 2],
            tile_assignment_devices: vec![0, 1, 2, 3],
            tuple_shardings: vec![],
            replicate_on_last_tile_dim: false,
            iota_reshape_dims: vec![4],
            iota_transpose_perm: vec![0],
        };
        let mut sharding_bytes = sharding.encode();
        sharding_bytes.extend_from_slice(&extra);
        assert_eq!(
            OpShardingProto::decode(&sharding_bytes).ok(),
            Some(sharding)
        );

        let inst = HloInstructionProto {
            name: String::from("add.1"),
            opcode: String::from("add"),
            shape: Some(shape.clone()),
            id: 3,
            operand_ids: vec![1, 2],
            parameter_number: 1,
            custom_call_target: String::from("TpuCustomCall"),
        };
        let mut inst_bytes = inst.encode();
        inst_bytes.extend_from_slice(&extra);
        assert_eq!(
            HloInstructionProto::decode(&inst_bytes).ok(),
            Some(inst.clone())
        );

        let comp = HloComputationProto {
            name: String::from("main"),
            instructions: vec![inst],
            program_shape: Some(prog_shape.clone()),
            id: 1,
            root_id: 3,
        };
        let mut comp_bytes = comp.encode();
        comp_bytes.extend_from_slice(&extra);
        assert_eq!(
            HloComputationProto::decode(&comp_bytes).ok(),
            Some(comp.clone())
        );

        let module = HloModuleProto {
            name: String::from("test_module"),
            entry_computation_name: String::from("main"),
            entry_computation_id: 1,
            computations: vec![comp],
            host_program_shape: Some(prog_shape.clone()),
            id: 42,
        };
        let mut mod_bytes = module.encode();
        mod_bytes.extend_from_slice(&extra);
        assert_eq!(HloModuleProto::decode(&mod_bytes).ok(), Some(module));

        let mod_cfg = HloModuleConfigProto {
            entry_computation_layout: Some(prog_shape),
            seed: 42,
            replica_count: 1,
            num_partitions: 2,
            use_spmd_partitioning: true,
        };
        let mut mod_cfg_bytes = mod_cfg.encode();
        mod_cfg_bytes.extend_from_slice(&extra);
        assert_eq!(
            HloModuleConfigProto::decode(&mod_cfg_bytes).ok(),
            Some(mod_cfg.clone())
        );

        let opt_override = OptionOverrideProto {
            string_field: String::from("true"),
            bool_field: true,
            int_field: 42,
            double_field: 1.5,
        };
        let mut opt_override_bytes = opt_override.encode();
        opt_override_bytes.extend_from_slice(&extra);
        assert_eq!(
            OptionOverrideProto::decode(&opt_override_bytes).ok(),
            Some(opt_override.clone())
        );

        let build_opts = ExecutableBuildOptionsProto {
            device_ordinal: 1,
            result_layout: Some(shape.clone()),
            num_replicas: 1,
            num_partitions: 2,
            use_spmd_partitioning: true,
            use_auto_spmd_partitioning: false,
            device_assignment: None,
            allow_spmd_sharding_propagation_to_output: vec![true],
            allow_spmd_sharding_propagation_to_parameters: vec![false],
            process_index: 0,
            process_count: 1,
            use_shardy_partitioner: true,
        };
        let mut bopts_bytes = build_opts.encode();
        bopts_bytes.extend_from_slice(&extra);
        assert_eq!(
            ExecutableBuildOptionsProto::decode(&bopts_bytes).ok(),
            Some(build_opts.clone())
        );

        let empty_copts = CompileOptionsProto::default();
        assert_eq!(
            CompileOptionsProto::decode(&empty_copts.encode()).ok(),
            Some(empty_copts)
        );

        let mut env_overrides = BTreeMap::new();
        env_overrides.insert(String::from("xla_tpu_flag"), opt_override);
        let compile_opts = CompileOptionsProto {
            argument_layouts: vec![shape.clone()],
            parameter_is_tupled_arguments: true,
            executable_build_options: Some(build_opts),
            compile_portable_executable: true,
            env_option_overrides: env_overrides,
            profile_version: 7,
            serialized_multi_slice_config: vec![1, 2],
        };
        let mut copts_bytes = compile_opts.encode();
        copts_bytes.extend_from_slice(&extra);
        assert_eq!(
            CompileOptionsProto::decode(&copts_bytes).ok(),
            Some(compile_opts)
        );

        assert_eq!(
            ExecutionModeProto::from(i32::from(ExecutionModeProto::Asynchronous)),
            ExecutionModeProto::Asynchronous
        );
        let exec_opts = ExecuteOptionsProto {
            arguments_are_tupled: false,
            untuple_result: true,
            launch_id: 9,
            strict_shape_checking: true,
            execution_mode: i32::from(ExecutionModeProto::Asynchronous),
            non_donatable_input_indices: vec![0],
            seed: 99,
        };
        let mut exec_opts_bytes = exec_opts.encode();
        exec_opts_bytes.extend_from_slice(&extra);
        assert_eq!(
            ExecuteOptionsProto::decode(&exec_opts_bytes).ok(),
            Some(exec_opts)
        );

        let mem_stats = CompiledMemoryStatsProto {
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
            serialized_buffer_assignment: vec![1, 2],
            peak_memory_in_bytes: 2048,
            total_size_in_bytes: 2048,
            total_allocation_bytes: 2048,
            indefinite_allocations: 0,
            peak_unpadded_heap_bytes: 1536,
        };
        let mut mem_stats_bytes = mem_stats.encode();
        mem_stats_bytes.extend_from_slice(&extra);
        assert_eq!(
            CompiledMemoryStatsProto::decode(&mem_stats_bytes).ok(),
            Some(mem_stats.clone())
        );

        let exec_meta = PjRtExecutableMetadataProto {
            compiled_memory_stats: Some(mem_stats),
            platform_specific_metadata: vec![9, 9],
        };
        let mut exec_meta_bytes = exec_meta.encode();
        exec_meta_bytes.extend_from_slice(&extra);
        assert_eq!(
            PjRtExecutableMetadataProto::decode(&exec_meta_bytes).ok(),
            Some(exec_meta)
        );

        let partial_prog = PjRtPartialProgramProto {
            program: b"module @m {}".to_vec(),
            program_format: String::from("mlir"),
            producer_phase: String::from("hlo_verification"),
            consumer_phases: vec![String::from("spmd_partitioning")],
            version: String::from("1.0"),
            program_name: String::from("m"),
        };
        let mut partial_prog_bytes = partial_prog.encode();
        partial_prog_bytes.extend_from_slice(&extra);
        assert_eq!(
            PjRtPartialProgramProto::decode(&partial_prog_bytes).ok(),
            Some(partial_prog)
        );

        let topo_proto = TpuTopologyArgsProto {
            version: 5,
            num_hosts: 1,
            chips_per_host: 4,
            mesh_bounds: vec![2, 2, 1],
            platform_version: String::from("0.0.50"),
            fingerprint: sha256_fingerprint_hex(b"topo"),
        };
        let mut topo_bytes = topo_proto.encode();
        topo_bytes.extend_from_slice(&extra);
        assert_eq!(
            TpuTopologyArgsProto::decode(&topo_bytes).ok(),
            Some(topo_proto.clone())
        );

        let pjrt_topo_desc = PjRtTopologyDescriptionProto {
            platform_id: 0x7470_7500,
            platform_name: String::from("tpu"),
            platform_version: String::from("0.0.50"),
            is_subslice_topology: false,
            platform_specific_topology: topo_proto.encode(),
        };
        let mut pjrt_topo_bytes = pjrt_topo_desc.encode();
        pjrt_topo_bytes.extend_from_slice(&extra);
        assert_eq!(
            PjRtTopologyDescriptionProto::decode(&pjrt_topo_bytes).ok(),
            Some(pjrt_topo_desc)
        );

        let hw_feat = TpuHardwareFeatureProto {
            embedding_feature: 2,
            num_embedding_devices_per_chip: 2,
        };
        let mut hw_feat_bytes = hw_feat.encode();
        hw_feat_bytes.extend_from_slice(&extra);
        assert_eq!(
            TpuHardwareFeatureProto::decode(&hw_feat_bytes).ok(),
            Some(hw_feat.clone())
        );

        let tf_topo = TopologyProto {
            mesh_shape: vec![2, 2, 1, 2],
            num_tasks: 1,
            num_tpu_devices_per_task: 8,
            device_coordinates: vec![0, 0, 0, 0],
            tpu_hardware_feature: Some(hw_feat),
        };
        let mut tf_topo_bytes = tf_topo.encode();
        tf_topo_bytes.extend_from_slice(&extra);
        assert_eq!(TopologyProto::decode(&tf_topo_bytes).ok(), Some(tf_topo));

        let exec_proto = TpuSerializedExecutableProto {
            name: String::from("exec_0"),
            fingerprint: sha256_fingerprint_hex(b"exec_0"),
            num_replicas: 1,
            num_partitions: 1,
            instruction_opcodes: vec![1, 2],
            output_shapes: vec![shape],
            compile_options_bytes: vec![1, 2, 3],
            hlo_module_bytes: vec![4, 5, 6],
        };
        let mut exec_bytes = exec_proto.encode();
        exec_bytes.extend_from_slice(&extra);
        assert_eq!(
            TpuSerializedExecutableProto::decode(&exec_bytes).ok(),
            Some(exec_proto)
        );

        let stat_meta = XStatMetadataProto {
            id: 1,
            name: String::from("flops"),
            description: String::from("FLOP count"),
        };
        let mut stat_meta_bytes = stat_meta.encode();
        stat_meta_bytes.extend_from_slice(&extra);
        assert_eq!(
            XStatMetadataProto::decode(&stat_meta_bytes).ok(),
            Some(stat_meta.clone())
        );

        let ev_meta = XEventMetadataProto {
            id: 2,
            name: String::from("execute"),
            display_name: String::from("TPU Execute"),
            metadata: String::from("vliw"),
        };
        let mut ev_meta_bytes = ev_meta.encode();
        ev_meta_bytes.extend_from_slice(&extra);
        assert_eq!(
            XEventMetadataProto::decode(&ev_meta_bytes).ok(),
            Some(ev_meta.clone())
        );

        let stat = XStatProto {
            metadata_id: 1,
            double_value: 2.5,
            uint64_value: 10,
            int64_value: 100,
            str_value: String::from("ok"),
            bytes_value: vec![1],
            ref_value: 0,
        };
        let mut stat_bytes = stat.encode();
        stat_bytes.extend_from_slice(&extra);
        assert_eq!(XStatProto::decode(&stat_bytes).ok(), Some(stat.clone()));

        let ev = XEventProto {
            metadata_id: 2,
            offset_ps: 1000,
            duration_ps: 5000,
            num_occurrences: 1,
            stats: vec![stat.clone()],
        };
        let mut ev_bytes = ev.encode();
        ev_bytes.extend_from_slice(&extra);
        assert_eq!(XEventProto::decode(&ev_bytes).ok(), Some(ev.clone()));

        let line = XLineProto {
            id: 1,
            display_id: 1,
            name: String::from("TensorCore 0"),
            display_name: String::from("TensorCore 0"),
            timestamp_ns: 123_456,
            duration_ps: 5000,
            events: vec![ev],
        };
        let mut line_bytes = line.encode();
        line_bytes.extend_from_slice(&extra);
        assert_eq!(XLineProto::decode(&line_bytes).ok(), Some(line.clone()));

        let mut ev_map = BTreeMap::new();
        ev_map.insert(2, ev_meta);
        let mut st_map = BTreeMap::new();
        st_map.insert(1, stat_meta);
        let plane = XPlaneProto {
            id: 1,
            name: String::from("/device:TPU:0"),
            lines: vec![line],
            event_metadata: ev_map,
            stat_metadata: st_map,
            stats: vec![stat],
        };
        let mut plane_bytes = plane.encode();
        plane_bytes.extend_from_slice(&extra);
        assert_eq!(XPlaneProto::decode(&plane_bytes).ok(), Some(plane.clone()));

        let xspace = XSpaceProto {
            planes: vec![plane],
            errors: vec![],
            warnings: vec![],
            hostnames: vec![String::from("localhost")],
        };
        let mut xspace_bytes = xspace.encode();
        xspace_bytes.extend_from_slice(&extra);
        assert_eq!(XSpaceProto::decode(&xspace_bytes).ok(), Some(xspace));

        let tbl = EmbeddingTableDescriptorProto {
            name: String::from("user_table"),
            vocabulary_size: 1024,
            dimension: 64,
            num_features: 2,
        };
        let mut tbl_bytes = tbl.encode();
        tbl_bytes.extend_from_slice(&extra);
        assert_eq!(
            EmbeddingTableDescriptorProto::decode(&tbl_bytes).ok(),
            Some(tbl.clone())
        );

        let feat = EmbeddingFeatureDescriptorProto {
            name: String::from("user_id"),
            table_id: 0,
            input_shape: vec![32, 1],
        };
        let mut feat_bytes = feat.encode();
        feat_bytes.extend_from_slice(&extra);
        assert_eq!(
            EmbeddingFeatureDescriptorProto::decode(&feat_bytes).ok(),
            Some(feat.clone())
        );

        let spmd_shard = EmbeddingSpmdShardingProto {
            enabled: true,
            num_cores_per_replica: 4,
        };
        let mut spmd_shard_bytes = spmd_shard.encode();
        spmd_shard_bytes.extend_from_slice(&extra);
        assert_eq!(
            EmbeddingSpmdShardingProto::decode(&spmd_shard_bytes).ok(),
            Some(spmd_shard.clone())
        );

        let emb_cfg = TpuEmbeddingConfigurationProto {
            table_descriptor: vec![tbl],
            mode: 2,
            batch_size_per_tensor_core: 32,
            num_hosts: 1,
            num_tensor_cores: 4,
            sharding_strategy: 1,
            pipeline_execution_with_tensor_core: true,
            profile_data_directory: String::new(),
            feature_descriptor: vec![feat],
            spmd_sharding: Some(spmd_shard),
        };
        let mut emb_cfg_bytes = emb_cfg.encode();
        emb_cfg_bytes.extend_from_slice(&extra);
        assert_eq!(
            TpuEmbeddingConfigurationProto::decode(&emb_cfg_bytes).ok(),
            Some(emb_cfg)
        );

        let sc_layout = SparseCoreTableLayoutProto {
            table_name: String::from("user_table"),
            unsharded_shape: vec![1024, 64],
            unsharded_padded_shape: vec![1024, 64],
            num_sparse_cores: 4,
            num_partitions: 1,
            total_rows_per_sparse_core_shard: 256,
        };
        let mut sc_layout_bytes = sc_layout.encode();
        sc_layout_bytes.extend_from_slice(&extra);
        assert_eq!(
            SparseCoreTableLayoutProto::decode(&sc_layout_bytes).ok(),
            Some(sc_layout.clone())
        );

        let sc_multi = SparseCoreTableLayoutsProto {
            tables: vec![sc_layout],
        };
        let mut sc_multi_bytes = sc_multi.encode();
        sc_multi_bytes.extend_from_slice(&extra);
        assert_eq!(
            SparseCoreTableLayoutsProto::decode(&sc_multi_bytes).ok(),
            Some(sc_multi)
        );

        let comp_dev = ComputationDeviceProto {
            replica_device_ids: vec![0, 1],
        };
        let mut cd_bytes = comp_dev.encode();
        cd_bytes.extend_from_slice(&extra);
        assert_eq!(
            ComputationDeviceProto::decode(&cd_bytes).ok(),
            Some(comp_dev.clone())
        );

        let dev_assign = DeviceAssignmentProto {
            replica_count: 2,
            computation_count: 1,
            computation_devices: vec![comp_dev],
        };
        let mut da_bytes = dev_assign.encode();
        da_bytes.extend_from_slice(&extra);
        assert_eq!(
            DeviceAssignmentProto::decode(&da_bytes).ok(),
            Some(dev_assign.clone())
        );

        let compile_meta = TpuCompileMetadataProto {
            num_replicas: 2,
            num_cores_per_replica: 1,
            device_assignment: Some(dev_assign),
            use_spmd_for_xla_partitioning: true,
        };
        let mut compile_meta_bytes = compile_meta.encode();
        compile_meta_bytes.extend_from_slice(&extra);
        assert_eq!(
            TpuCompileMetadataProto::decode(&compile_meta_bytes).ok(),
            Some(compile_meta)
        );

        let comp_env = TpuCompilationEnvironmentProto {
            xla_tpu_enable_megacore_fusion: true,
            xla_tpu_enable_async_collective_fusion: true,
            xla_tpu_scoped_vmem_limit_kib: 16384,
            custom_flags: String::from("--xla_tpu_flag=true"),
        };
        let mut comp_env_bytes = comp_env.encode();
        comp_env_bytes.extend_from_slice(&extra);
        assert_eq!(
            TpuCompilationEnvironmentProto::decode(&comp_env_bytes).ok(),
            Some(comp_env)
        );

        let empty_hmc = HloModuleProtoWithConfig::default();
        assert_eq!(
            HloModuleProtoWithConfig::decode(&empty_hmc.encode()).ok(),
            Some(empty_hmc)
        );
        let hmc = HloModuleProtoWithConfig {
            hlo_module: Some(HloModuleProto::decode(&mod_bytes).unwrap_or_default()),
            config: Some(mod_cfg),
        };
        let mut hmc_bytes = hmc.encode();
        hmc_bytes.extend_from_slice(&extra);
        assert_eq!(
            HloModuleProtoWithConfig::decode(&hmc_bytes).ok(),
            Some(hmc.clone())
        );

        let comp_res = CompilationResultProto {
            status_code: 0,
            status_error_message: String::new(),
            hlo_protos: vec![hmc],
        };
        let mut comp_res_bytes = comp_res.encode();
        comp_res_bytes.extend_from_slice(&extra);
        assert_eq!(
            CompilationResultProto::decode(&comp_res_bytes).ok(),
            Some(comp_res)
        );

        let r_abi = RuntimeAbiVersionProto {
            platform_id: 0x7470_7500,
            version_string: String::from("0.0.50"),
            pjrt_api_minor: 116,
        };
        let mut r_abi_bytes = r_abi.encode();
        r_abi_bytes.extend_from_slice(&extra);
        assert_eq!(
            RuntimeAbiVersionProto::decode(&r_abi_bytes).ok(),
            Some(r_abi)
        );

        let e_abi = ExecutableAbiVersionProto {
            platform_id: 0x7470_7500,
            compiler_version: String::from("0.0.50"),
            target_generation: 5,
        };
        let mut e_abi_bytes = e_abi.encode();
        e_abi_bytes.extend_from_slice(&extra);
        assert_eq!(
            ExecutableAbiVersionProto::decode(&e_abi_bytes).ok(),
            Some(e_abi)
        );

        let part_cfg = TpuEmbeddingPartitionedConfigProto {
            num_partitions: 2,
            table_names: vec![String::from("user_table")],
            raw_config_bytes: vec![1, 2, 3],
        };
        let mut part_cfg_bytes = part_cfg.encode();
        part_cfg_bytes.extend_from_slice(&extra);
        assert_eq!(
            TpuEmbeddingPartitionedConfigProto::decode(&part_cfg_bytes).ok(),
            Some(part_cfg)
        );

        let mem_cfg = TpuEmbeddingMemoryConfigProto {
            hbm_buffers_bytes: 4096,
            num_segments: 1,
            segment_blobs: vec![vec![10, 20]],
        };
        let mut mem_cfg_bytes = mem_cfg.encode();
        mem_cfg_bytes.extend_from_slice(&extra);
        assert_eq!(
            TpuEmbeddingMemoryConfigProto::decode(&mem_cfg_bytes).ok(),
            Some(mem_cfg)
        );

        let host_net_cfg = TpuEmbeddingHostNetworkConfigProto {
            host_id: 1,
            endpoint_address: String::from("127.0.0.1:8471"),
            partition_payload: vec![5, 6],
        };
        let mut host_net_bytes = host_net_cfg.encode();
        host_net_bytes.extend_from_slice(&extra);
        assert_eq!(
            TpuEmbeddingHostNetworkConfigProto::decode(&host_net_bytes).ok(),
            Some(host_net_cfg)
        );

        let ms_cfg = MegascaleMultiSliceConfigProto {
            num_slices: 2,
            slice_id: 1,
            num_devices_per_slice: 4,
            coordinator_address: String::from("127.0.0.1:8080"),
        };
        let mut ms_cfg_bytes = ms_cfg.encode();
        ms_cfg_bytes.extend_from_slice(&extra);
        assert_eq!(
            MegascaleMultiSliceConfigProto::decode(&ms_cfg_bytes).ok(),
            Some(ms_cfg)
        );

        let dcn_topo = MegascaleDcnTopologyProto {
            num_slices: 2,
            num_hosts_per_slice: 1,
            transport_type: 1,
        };
        let mut dcn_bytes = dcn_topo.encode();
        dcn_bytes.extend_from_slice(&extra);
        assert_eq!(
            MegascaleDcnTopologyProto::decode(&dcn_bytes).ok(),
            Some(dcn_topo)
        );

        let ms_metrics = MegascaleRuntimeMetricsProto {
            completed_barriers: 4,
            collective_ops_count: 8,
            active_contexts: 1,
        };
        let mut ms_metrics_bytes = ms_metrics.encode();
        ms_metrics_bytes.extend_from_slice(&extra);
        assert_eq!(
            MegascaleRuntimeMetricsProto::decode(&ms_metrics_bytes).ok(),
            Some(ms_metrics)
        );

        let isa_slot = IsaBundleSlotProto {
            unit_kind: 2,
            opcode: 1,
            dst_reg: 4,
            src0_reg: 0,
            src1_reg: 1,
            predicate_reg: 0,
            immediate: 0,
        };
        assert_eq!(
            IsaBundleSlotProto::decode(&isa_slot.encode()).ok(),
            Some(isa_slot.clone())
        );

        let isa_bundle = IsaVliwBundleProto {
            pc_index: 0,
            scalar_s0: Some(isa_slot.clone()),
            scalar_s1: None,
            vector_v0: Some(isa_slot.clone()),
            vector_v1: None,
            matrix_m0: Some(isa_slot.clone()),
            matrix_m1: None,
            unary_xlu: Some(isa_slot.clone()),
            dma_d0: Some(isa_slot),
            llo_annotation: String::from("llo.vadd"),
        };
        assert_eq!(
            IsaVliwBundleProto::decode(&isa_bundle.encode()).ok(),
            Some(isa_bundle.clone())
        );

        let isa_core = IsaCoreProgramHeaderProto {
            core_index: 0,
            core_type: 0,
            entry_pc: 0,
            bundle_count: 1,
            raw_vliw_stream: vec![0_u8; 64],
            structured_bundles: vec![isa_bundle],
        };
        assert_eq!(
            IsaCoreProgramHeaderProto::decode(&isa_core.encode()).ok(),
            Some(isa_core.clone())
        );

        let isa_sym = IsaSymbolTableEntryProto {
            name: String::from("main"),
            pc_offset: 0,
            size_bytes: 64,
        };
        assert_eq!(
            IsaSymbolTableEntryProto::decode(&isa_sym.encode()).ok(),
            Some(isa_sym.clone())
        );

        let isa_prog = IsaProgramProto {
            isa_architecture: String::from("vfc"),
            isa_abi_version: 163,
            core_programs: vec![isa_core],
            symbols: vec![isa_sym],
            fingerprint: 0x1234,
        };
        assert_eq!(
            IsaProgramProto::decode(&isa_prog.encode()).ok(),
            Some(isa_prog.clone())
        );

        let target_args = TargetArgumentsProto {
            tpu_version: 6,
            chip_variant: String::from("ghostfish"),
            clock_rate_mhz: 1750,
            hbm_capacity_bytes: 95 * 1024 * 1024 * 1024,
            vmem_capacity_bytes: 64 * 1024 * 1024,
            smem_capacity_bytes: 2 * 1024 * 1024,
            cmem_capacity_bytes: 128 * 1024 * 1024,
            tensor_cores_per_chip: 2,
            sparse_cores_per_chip: 4,
            mxu_dimension: 128,
            ici_bandwidth_bps: 4_800_000_000_000,
        };
        assert_eq!(
            TargetArgumentsProto::decode(&target_args.encode()).ok(),
            Some(target_args.clone())
        );

        let ds_overlay = DeepseaOverlaySectionProto {
            overlay_id: 0,
            symbol_name: String::from(".text.main"),
            imem_offset_bytes: 0,
            byte_size: 64,
            encoded_bundles: vec![0_u8; 64],
            core_type: 0,
        };
        assert_eq!(
            DeepseaOverlaySectionProto::decode(&ds_overlay.encode()).ok(),
            Some(ds_overlay.clone())
        );

        let ds_reloc = DeepseaRelocationEntryProto {
            bundle_offset_bytes: 0,
            slot_index: 8,
            buffer_index: 0,
            addend_bytes: 0,
            memory_space: 0,
        };
        assert_eq!(
            DeepseaRelocationEntryProto::decode(&ds_reloc.encode()).ok(),
            Some(ds_reloc.clone())
        );

        let ds_const = DeepseaConstantPoolEntryProto {
            constant_id: 0,
            memory_space: 1,
            offset_bytes: 0,
            payload: vec![1, 2, 3, 4],
        };
        assert_eq!(
            DeepseaConstantPoolEntryProto::decode(&ds_const.encode()).ok(),
            Some(ds_const.clone())
        );

        let ds_sflag = DeepseaSyncFlagInitProto {
            sync_flag_id: 1,
            initial_value: 0,
            debug_label: String::from("dma_done"),
        };
        assert_eq!(
            DeepseaSyncFlagInitProto::decode(&ds_sflag.encode()).ok(),
            Some(ds_sflag.clone())
        );

        let core_abi = TpuCoreProgramAbiProto {
            version: 6,
            variant: String::from("default"),
            chip_config: String::from("default"),
            base_abi_version: 163,
            hardware_abi_version: 65_539,
            required_compiler_hash: String::from(
                "00ef0b06b559d6911774f1854abee45cdcb62dc50b6c9535921e52231188d011",
            ),
        };
        assert_eq!(
            TpuCoreProgramAbiProto::decode(&core_abi.encode()).ok(),
            Some(core_abi.clone())
        );

        let seg_range = TpuMemorySegmentRangeProto {
            offset: 0,
            size: 64,
        };
        assert_eq!(
            TpuMemorySegmentRangeProto::decode(&seg_range.encode()).ok(),
            Some(seg_range.clone())
        );

        let seg_proto = TpuMemorySegmentProto {
            segment_type: 1,
            memory_type: 1,
            initialized_data_range: Some(seg_range),
            segment_size_in_bytes: 64,
        };
        assert_eq!(
            TpuMemorySegmentProto::decode(&seg_proto.encode()).ok(),
            Some(seg_proto.clone())
        );

        let seg_set = TpuMemorySegmentSetProto {
            memory_type: 1,
            segments: vec![seg_proto],
            initialized_data: vec![0_u8; 64],
            hash: vec![1, 2, 3, 4],
            compression: 0,
        };
        assert_eq!(
            TpuMemorySegmentSetProto::decode(&seg_set.encode()).ok(),
            Some(seg_set.clone())
        );

        let core_prog = TpuCoreProgramProto {
            platform_type: 1,
            fingerprint: vec![5, 6, 7, 8],
            compilation_id: 1,
            memory_segments: vec![seg_set],
            concurrent_program_hbm_stack_size: 0,
            has_concurrent_program: false,
            abi: Some(core_abi),
            oom_during_compilation: false,
        };
        assert_eq!(
            TpuCoreProgramProto::decode(&core_prog.encode()).ok(),
            Some(core_prog.clone())
        );

        let ds_exec = DeepseaExecutableProto {
            module_name: String::from("main"),
            isa_program: Some(isa_prog),
            overlays: vec![ds_overlay],
            relocations: vec![ds_reloc],
            constants: vec![ds_const],
            sync_flags: vec![ds_sflag],
            target_arguments: Some(target_args.clone()),
            vmem_allocation_bytes: 4096,
            smem_allocation_bytes: 256,
            cmem_allocation_bytes: 0,
            hbm_scratchpad_bytes: 4096,
            core_program: Some(core_prog.clone()),
            sparse_core_program: Some(core_prog),
        };
        assert_eq!(
            DeepseaExecutableProto::decode(&ds_exec.encode()).ok(),
            Some(ds_exec.clone())
        );

        let host_cmd = HostCommandEnvelopeProto {
            opcode: 1,
            continuation_id: 42,
            device_id: 0,
            core_index: 0,
            handler_id: 7,
            payload: vec![9, 9],
        };
        assert_eq!(
            HostCommandEnvelopeProto::decode(&host_cmd.encode()).ok(),
            Some(host_cmd)
        );

        let tpu_exec = TpuExecutableProto {
            ds_executable: Some(ds_exec),
            hlo_module_with_config: None,
            compile_options: None,
            target_arguments: Some(target_args),
            legacy_executable: None,
            source_uri: String::from("embed://main"),
        };
        assert_eq!(
            TpuExecutableProto::decode(&tpu_exec.encode()).ok(),
            Some(tpu_exec)
        );

        let dist_dev = DistributedDeviceProto {
            local_device_ordinal: 0,
            name: String::from("TPU:0"),
            vendor: String::from("Google"),
            global_device_id: 0,
            partition_index: 0,
            compute_capability: String::from("v5e"),
            core_count: 1,
            device_kind: String::from("Cloud TPU v5e"),
            to_string: String::from("TpuDevice(id=0)"),
            debug_string: String::from("TpuDevice(id=0, coords=(0,0,0))"),
            fabric_uuid: String::from("slice0/0"),
            numa_node: 0,
            device_memory_bytes_limit: 16 * 1024 * 1024 * 1024,
        };
        assert_eq!(
            DistributedDeviceProto::decode(&dist_dev.encode()).ok(),
            Some(dist_dev.clone())
        );

        let local_topo = LocalTopologyProto {
            process_id: 0,
            boot_id: String::from("boot-0"),
            devices: vec![dist_dev],
            partition_index: 0,
            network_nodes: vec![String::from("spine-0"), String::from("rack-0")],
        };
        assert_eq!(
            LocalTopologyProto::decode(&local_topo.encode()).ok(),
            Some(local_topo.clone())
        );

        let global_topo = GlobalTopologyProto {
            processes: vec![local_topo],
        };
        assert_eq!(
            GlobalTopologyProto::decode(&global_topo.encode()).ok(),
            Some(global_topo)
        );

        let buf_slice = BufferAllocationSliceProto {
            offset: 0,
            size: 4096,
            buffer_allocation_index: 0,
            element_type: 11,
        };
        assert_eq!(
            BufferAllocationSliceProto::decode(&buf_slice.encode()).ok(),
            Some(buf_slice.clone())
        );

        let buf_alloc = BufferAllocationProto {
            index: 0,
            size: 4096,
            is_thread_local: false,
            is_tuple: false,
            is_entry_computation_parameter: true,
            parameter_number: 0,
            is_constant: false,
            maybe_live_out: true,
            color: 0,
            assigned: vec![buf_slice],
        };
        assert_eq!(
            BufferAllocationProto::decode(&buf_alloc.encode()).ok(),
            Some(buf_alloc.clone())
        );

        let buf_assign = BufferAssignmentProto {
            buffer_allocations: vec![buf_alloc],
        };
        assert_eq!(
            BufferAssignmentProto::decode(&buf_assign.encode()).ok(),
            Some(buf_assign)
        );

        let ms_err = MegaScaleRuntimeErrorProto {
            error_type: 1,
            error_message: String::from("timeout"),
            estimated_start_time_ms: 1000,
            triggering_launch_id: 7,
            hostname: String::from("tpu-worker-0"),
            job_name: String::from("trainer"),
            task_id: 0,
            event_id: 11,
            unrecoverable_error_type: 0,
            runtime_error_id: 22,
            hang_error_type: 1,
        };
        assert_eq!(
            MegaScaleRuntimeErrorProto::decode(&ms_err.encode()).ok(),
            Some(ms_err)
        );

        // Corrupt wire bytes to exercise error branches on all message types
        let corrupt = [0x80_u8];
        assert!(TileProto::decode(&corrupt).is_err());
        assert!(LayoutProto::decode(&corrupt).is_err());
        assert!(ShapeProto::decode(&corrupt).is_err());
        assert!(ProgramShapeProto::decode(&corrupt).is_err());
        assert!(OpShardingProto::decode(&corrupt).is_err());
        assert!(HloInstructionProto::decode(&corrupt).is_err());
        assert!(HloComputationProto::decode(&corrupt).is_err());
        assert!(HloModuleProto::decode(&corrupt).is_err());
        assert!(HloModuleConfigProto::decode(&corrupt).is_err());
        assert!(HloModuleProtoWithConfig::decode(&corrupt).is_err());
        assert!(OptionOverrideProto::decode(&corrupt).is_err());
        assert!(ExecutableBuildOptionsProto::decode(&corrupt).is_err());
        assert!(CompileOptionsProto::decode(&corrupt).is_err());
        assert!(ExecuteOptionsProto::decode(&corrupt).is_err());
        assert!(CompiledMemoryStatsProto::decode(&corrupt).is_err());
        assert!(PjRtExecutableMetadataProto::decode(&corrupt).is_err());
        assert!(PjRtPartialProgramProto::decode(&corrupt).is_err());
        assert!(PjRtTopologyDescriptionProto::decode(&corrupt).is_err());
        assert!(ComputationDeviceProto::decode(&corrupt).is_err());
        assert!(DeviceAssignmentProto::decode(&corrupt).is_err());
        assert!(TpuCompilationEnvironmentProto::decode(&corrupt).is_err());
        assert!(XStatMetadataProto::decode(&corrupt).is_err());
        assert!(XEventMetadataProto::decode(&corrupt).is_err());
        assert!(TpuTopologyArgsProto::decode(&corrupt).is_err());
        assert!(TpuHardwareFeatureProto::decode(&corrupt).is_err());
        assert!(TopologyProto::decode(&corrupt).is_err());
        assert!(TpuSerializedExecutableProto::decode(&corrupt).is_err());
        assert!(XStatProto::decode(&corrupt).is_err());
        assert!(XEventProto::decode(&corrupt).is_err());
        assert!(XLineProto::decode(&corrupt).is_err());
        assert!(XPlaneProto::decode(&corrupt).is_err());
        assert!(XSpaceProto::decode(&corrupt).is_err());
        assert!(EmbeddingTableDescriptorProto::decode(&corrupt).is_err());
        assert!(EmbeddingFeatureDescriptorProto::decode(&corrupt).is_err());
        assert!(EmbeddingSpmdShardingProto::decode(&corrupt).is_err());
        assert!(TpuEmbeddingConfigurationProto::decode(&corrupt).is_err());
        assert!(SparseCoreTableLayoutProto::decode(&corrupt).is_err());
        assert!(SparseCoreTableLayoutsProto::decode(&corrupt).is_err());
        assert!(CompilationResultProto::decode(&corrupt).is_err());
        assert!(TpuCompileMetadataProto::decode(&corrupt).is_err());
        assert!(RuntimeAbiVersionProto::decode(&corrupt).is_err());
        assert!(ExecutableAbiVersionProto::decode(&corrupt).is_err());
        assert!(TpuEmbeddingPartitionedConfigProto::decode(&corrupt).is_err());
        assert!(TpuEmbeddingMemoryConfigProto::decode(&corrupt).is_err());
        assert!(TpuEmbeddingHostNetworkConfigProto::decode(&corrupt).is_err());
        assert!(MegascaleMultiSliceConfigProto::decode(&corrupt).is_err());
        assert!(MegascaleDcnTopologyProto::decode(&corrupt).is_err());
        assert!(MegascaleRuntimeMetricsProto::decode(&corrupt).is_err());
        assert!(IsaBundleSlotProto::decode(&corrupt).is_err());
        assert!(IsaVliwBundleProto::decode(&corrupt).is_err());
        assert!(IsaCoreProgramHeaderProto::decode(&corrupt).is_err());
        assert!(IsaSymbolTableEntryProto::decode(&corrupt).is_err());
        assert!(IsaProgramProto::decode(&corrupt).is_err());
        assert!(TargetArgumentsProto::decode(&corrupt).is_err());
        assert!(DeepseaOverlaySectionProto::decode(&corrupt).is_err());
        assert!(DeepseaRelocationEntryProto::decode(&corrupt).is_err());
        assert!(DeepseaConstantPoolEntryProto::decode(&corrupt).is_err());
        assert!(DeepseaSyncFlagInitProto::decode(&corrupt).is_err());
        assert!(TpuCoreProgramAbiProto::decode(&corrupt).is_err());
        assert!(TpuMemorySegmentRangeProto::decode(&corrupt).is_err());
        assert!(TpuMemorySegmentProto::decode(&corrupt).is_err());
        assert!(TpuMemorySegmentSetProto::decode(&corrupt).is_err());
        assert!(TpuCoreProgramProto::decode(&corrupt).is_err());
        assert!(DeepseaExecutableProto::decode(&corrupt).is_err());
        assert!(HostCommandEnvelopeProto::decode(&corrupt).is_err());
        assert!(TpuExecutableProto::decode(&corrupt).is_err());
        assert!(DistributedDeviceProto::decode(&corrupt).is_err());
        assert!(LocalTopologyProto::decode(&corrupt).is_err());
        assert!(GlobalTopologyProto::decode(&corrupt).is_err());
        assert!(BufferAllocationSliceProto::decode(&corrupt).is_err());
        assert!(BufferAllocationProto::decode(&corrupt).is_err());
        assert!(BufferAssignmentProto::decode(&corrupt).is_err());
        assert!(MegaScaleRuntimeErrorProto::decode(&corrupt).is_err());

        let pool_res = global_descriptor_pool();
        assert!(pool_res.is_ok());
        let pool_res_cached = global_descriptor_pool();
        assert!(pool_res_cached.is_ok());

        let isa_prog = IsaProgramProto {
            isa_architecture: String::from("deepsea_v6e"),
            isa_abi_version: 6,
            core_programs: Vec::new(),
            symbols: Vec::new(),
            fingerprint: 0x1234,
        };
        let isa_bytes = isa_prog.encode();
        let dyn_isa = decode_dynamic_proto_message("xla.IsaProgramProto", &isa_bytes);
        assert!(dyn_isa.is_ok());
        let json_isa = transcode_proto_bytes_to_json("xla.IsaProgramProto", &isa_bytes);
        assert!(json_isa.unwrap_or_default().contains("deepsea_v6e"));

        assert!(decode_dynamic_proto_message("nonexistent.MessageProto", &isa_bytes).is_err());
        assert!(decode_dynamic_proto_message("xla.IsaProgramProto", &corrupt).is_err());
    }

    proptest! {
        #[test]
        fn proptest_shape_and_topology_roundtrip(
            elem_type in 0_i32..28,
            dims in prop::collection::vec(0_i64..1024, 0..5),
            num_hosts in 1_i32..16,
            chips_per_host in 1_i32..8,
            platform_version in "[a-zA-Z0-9._-]{1,16}"
        ) {
            let rank_i64 = i64::try_from(dims.len()).unwrap_or(0);
            let shape = ShapeProto {
                element_type: elem_type,
                dimensions: dims.clone(),
                tuple_shapes: Vec::new(),
                layout: Some(LayoutProto {
                    minor_to_major: (0..rank_i64).rev().collect(),
                    tiles: vec![TileProto { dimensions: vec![8, 128] }],
                    tail_padding_alignment_in_elements: 1,
                    index_primitive_type: 0,
                    pointer_primitive_type: 0,
                    memory_space: 0,
                    element_size_in_bits: 32,
                    dynamic_shape_metadata_prefix_bytes: 0,
                }),
                is_dynamic_dimension: vec![false; dims.len()],
            };
            let encoded = shape.encode();
            prop_assert_eq!(ShapeProto::decode(&encoded).ok(), Some(shape));

            let topo = TpuTopologyArgsProto {
                version: 6,
                num_hosts,
                chips_per_host,
                mesh_bounds: vec![2, 2, 1],
                platform_version: platform_version.clone(),
                fingerprint: sha256_fingerprint_hex(platform_version.as_bytes()),
            };
            let topo_encoded = topo.encode();
            prop_assert_eq!(TpuTopologyArgsProto::decode(&topo_encoded).ok(), Some(topo));
        }
    }
}

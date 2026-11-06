//! Protobuf descriptor reflection, dynamic transcoding, and `.proto` schema verification.
//!
//! Powered by `prost` and `prost-reflect` over the embedded `FileDescriptorSet`.

use crate::error::{LibTpuError, LibTpuResult};
use prost::Message;
use prost_reflect::{DescriptorPool, DynamicMessage, MessageDescriptor};
use sha2::{Digest, Sha256};
use std::sync::OnceLock;

/// Raw wire-encoded `google.protobuf.FileDescriptorSet` compiled at build time by `protox` + `prost-build`
/// from `docs/reverse_engineering/protos/*.proto`.
pub const EMBEDDED_FILE_DESCRIPTOR_SET_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/libtpu_file_descriptor_set.bin"));

/// Canonical list of all strongly-typed `libtpu-rs` protobuf message names synchronized with
/// [`global_descriptor_pool`].
pub const ALL_REGISTERED_PROTO_MESSAGE_NAMES: &[&str] = &[
    "xla.TileProto",
    "xla.SplitConfigProto",
    "xla.LayoutProto",
    "xla.ShapeProto",
    "xla.ProgramShapeProto",
    "xla.OpMetadata",
    "xla.ExecutionProfile",
    "xla.ExecutionHandle",
    "xla.GlobalDataHandle",
    "xla.DeviceHandle",
    "xla.ChannelHandle",
    "xla.LiteralProto",
    "xla.WindowDimension",
    "xla.Window",
    "xla.GatherDimensionNumbers",
    "xla.ScatterDimensionNumbers",
    "xla.ConvolutionDimensionNumbers",
    "xla.DotDimensionNumbers",
    "xla.PaddingConfig.PaddingConfigDimension",
    "xla.PaddingConfig",
    "xla.ReplicaGroup",
    "xla.SourceTarget",
    "xla.PrecisionConfig",
    "xla.FrontendAttributes",
    "xla.Statistic",
    "xla.StatisticsViz",
    "xla.ParameterReplication",
    "xla.OpSharding",
    "xla.DeviceAssignmentProto.ComputationDevice",
    "xla.DeviceAssignmentProto",
    "xla.HloInstructionProto.SliceDimensions",
    "xla.HloInputOutputAliasProto.AliasEntryProto",
    "xla.HloInputOutputAliasProto",
    "xla.HloBufferDonorProto.BufferDonorEntryProto",
    "xla.HloBufferDonorProto",
    "xla.CrossProgramPrefetch",
    "xla.HloInstructionProto",
    "xla.HloComputationProto",
    "xla.HloModuleProto",
    "xla.HloModuleConfigProto",
    "xla.HloModuleProtoWithConfig",
    "xla.buffer_assignment.BufferAllocationSliceProto",
    "xla.BufferAllocationProto",
    "xla.BufferAssignmentProto",
    "xla.OptionOverrideProto",
    "xla.ExecutableBuildOptionsProto",
    "xla.CompileOptionsProto",
    "xla.ExecuteOptionsProto",
    "xla.CompiledMemoryStatsProto",
    "xla.PjRtExecutableMetadata",
    "xla.PjRtPartialProgramProto",
    "xla.PjRtTopologyDescriptionProto",
    "xla.PjRtRuntimeAbiVersionProto",
    "xla.PjRtExecutableAbiVersionProto",
    "xla.DeviceProto",
    "xla.LocalTopologyProto",
    "xla.GlobalTopologyProto",
    "tpu.TpuCompilationEnvironmentProto",
    "tpu.TpuTopologyArgsProto",
    "tpu.RegisterTranslatorEntryProto",
    "tpu.RegisterTranslatorDataProto",
    "tpu.TpuChipConfigProto",
    "tpu.TpuChipPartsProto",
    "tpu.TpuRuntimeAbiProto",
    "tensorflow.tpu.TPUHardwareFeature",
    "tensorflow.tpu.TopologyProto",
    "tensorflow.tpu.TPUEmbeddingConfiguration.TableDescriptor",
    "tensorflow.tpu.TPUEmbeddingConfiguration.FeatureDescriptor",
    "tensorflow.tpu.TPUEmbeddingConfiguration.SpmdSharding",
    "tensorflow.tpu.TPUEmbeddingConfiguration",
    "tensorflow.tpu.SparseCoreTableLayout",
    "tensorflow.tpu.SparseCoreTableLayouts",
    "tensorflow.tpu.CompilationResultProto",
    "tensorflow.tpu.TPUCompileMetadataProto",
    "tpu.TpuEmbeddingPartitionedConfigProto",
    "tpu.TpuEmbeddingMemoryConfigProto",
    "tpu.TpuEmbeddingHostNetworkConfigProto",
    "xla.IsaBundleSlotProto",
    "xla.IsaVliwBundleProto",
    "xla.IsaCoreProgramHeaderProto",
    "xla.IsaSymbolTableEntryProto",
    "xla.IsaProgramProto",
    "xla.TargetArgumentsProto",
    "xla.DeepseaOverlaySectionProto",
    "xla.DeepseaRelocationEntryProto",
    "xla.DeepseaConstantPoolEntryProto",
    "xla.DeepseaSyncFlagInitProto",
    "tpu.TpuCoreProgramAbiProto",
    "xla.TpuMemorySegmentRangeProto",
    "xla.TpuMemorySegmentProto",
    "xla.TpuMemorySegmentSetProto",
    "xla.TpuCoreProgramProto",
    "xla.DeepseaExecutableProto",
    "xla.HostOffloadingExecutableProto",
    "xla.ffi.ExecutionStateProto",
    "xla.ffi.FlatAttributeProto",
    "xla.ffi.FlatAttributesMapProto",
    "tpu.HostCommandEnvelopeProto",
    "tpu.TpuSerializedExecutableProto",
    "xla.TpuExecutableProto",
    "megascale.MultiSliceConfigProto",
    "megascale.DCNTopologyProto",
    "megascale.RuntimeMetricsProto",
    "xla.megascale.runtime.HostNetworkAddress",
    "xla.megascale.runtime.NetworkAddressMapping",
    "xla.megascale.runtime.EndpointAddresses",
    "xla.megascale.runtime.MegascaleTransportConfigProto",
    "xla.megascale.runtime.external.MegaScaleRuntimeError",
    "tensorflow.profiler.XStatMetadata",
    "tensorflow.profiler.XEventMetadata",
    "tensorflow.profiler.XStat",
    "tensorflow.profiler.XEvent",
    "tensorflow.profiler.XLine",
    "tensorflow.profiler.XPlane",
    "tensorflow.profiler.XSpace",
];

/// Global cached [`DescriptorPool`] decoded from [`EMBEDDED_FILE_DESCRIPTOR_SET_BYTES`].
static GLOBAL_DESCRIPTOR_POOL: OnceLock<DescriptorPool> = OnceLock::new();

/// Decodes a [`DescriptorPool`] from a wire-encoded `google.protobuf.FileDescriptorSet` slice.
///
/// # Arguments
///
/// * `bytes` - Wire-encoded `FileDescriptorSet` byte slice.
///
/// # Errors
///
/// Returns [`LibTpuError::ProstReflectDescriptorFailed`] if the descriptor pool fails to decode.
pub fn decode_descriptor_pool(bytes: &[u8]) -> LibTpuResult<DescriptorPool> {
    Ok(DescriptorPool::decode(bytes)?)
}

/// Returns a fallible reference to the global [`DescriptorPool`] compiled from `docs/reverse_engineering/protos/*.proto`.
///
/// # Errors
///
/// Returns [`LibTpuError::ProstReflectDescriptorFailed`] if the embedded descriptor set is malformed.
pub fn try_global_descriptor_pool() -> LibTpuResult<&'static DescriptorPool> {
    let _ = decode_descriptor_pool(EMBEDDED_FILE_DESCRIPTOR_SET_BYTES)?;
    Ok(global_descriptor_pool())
}

/// Returns a reference to the global [`DescriptorPool`] compiled from `docs/reverse_engineering/protos/*.proto`.
#[must_use]
pub fn global_descriptor_pool() -> &'static DescriptorPool {
    GLOBAL_DESCRIPTOR_POOL.get_or_init(|| {
        let mut pool = DescriptorPool::new();
        let _ = pool.decode_file_descriptor_set(EMBEDDED_FILE_DESCRIPTOR_SET_BYTES);
        pool
    })
}

/// Looks up a [`MessageDescriptor`] by its fully-qualified protobuf type name in [`global_descriptor_pool`].
///
/// # Arguments
///
/// * `full_message_name` - Fully-qualified protobuf message name (`package.MessageName`).
///
/// # Errors
///
/// Returns [`LibTpuError::ProtobufDecodeError`] if the message descriptor is not found.
pub fn lookup_message_descriptor(
    full_message_name: &'static str,
) -> LibTpuResult<MessageDescriptor> {
    global_descriptor_pool()
        .get_message_by_name(full_message_name)
        .ok_or_else(|| LibTpuError::ProtobufDecodeError {
            message_type: full_message_name,
            reason: format!("message descriptor '{full_message_name}' not found in DescriptorPool"),
        })
}

/// Decodes a dynamic protobuf message by its fully-qualified protobuf type name
/// (e.g., `"xla.IsaProgramProto"` or `"xla.TargetArgumentsProto"`) using [`global_descriptor_pool`].
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
    let msg_desc = lookup_message_descriptor(full_message_name)?;
    DynamicMessage::decode(msg_desc, bytes).map_err(|err| LibTpuError::ProtobufDecodeError {
        message_type: full_message_name,
        reason: err.to_string(),
    })
}

/// Decodes a dynamic protobuf message from canonical Protobuf JSON (`proto3` JSON mapping)
/// using [`global_descriptor_pool`] and `serde_json`.
///
/// # Arguments
///
/// * `full_message_name` - Fully-qualified protobuf message name (`package.MessageName`).
/// * `json` - Canonical Protobuf JSON string.
///
/// # Errors
///
/// Returns [`LibTpuError::ProtobufDecodeError`] if the message descriptor is not found or JSON parsing fails.
pub fn decode_dynamic_proto_json(
    full_message_name: &'static str,
    json: &str,
) -> LibTpuResult<DynamicMessage> {
    let msg_desc = lookup_message_descriptor(full_message_name)?;
    let mut deserializer = serde_json::Deserializer::from_str(json);
    DynamicMessage::deserialize(msg_desc, &mut deserializer).map_err(|err| {
        LibTpuError::ProtobufDecodeError {
            message_type: full_message_name,
            reason: err.to_string(),
        }
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
    Ok(serde_json::to_string(&dyn_msg)?)
}

/// Decodes a dynamic protobuf message from canonical Protobuf text-format (`textproto`)
/// using [`global_descriptor_pool`] and `prost_reflect`.
///
/// # Arguments
///
/// * `full_message_name` - Fully-qualified protobuf message name (`package.MessageName`).
/// * `text` - Canonical Protobuf text-format (`textproto`) string.
///
/// # Errors
///
/// Returns [`LibTpuError::ProtobufDecodeError`] if the message descriptor is not found or textproto parsing fails.
pub fn decode_dynamic_proto_text(
    full_message_name: &'static str,
    text: &str,
) -> LibTpuResult<DynamicMessage> {
    let msg_desc = lookup_message_descriptor(full_message_name)?;
    DynamicMessage::parse_text_format(msg_desc, text).map_err(|err| {
        LibTpuError::ProtobufDecodeError {
            message_type: full_message_name,
            reason: err.to_string(),
        }
    })
}

/// Transcodes a wire-encoded protobuf message into canonical Protobuf text-format (`textproto`) via `prost_reflect`.
///
/// # Arguments
///
/// * `full_message_name` - Fully-qualified protobuf message name (`package.MessageName`).
/// * `bytes` - Wire-encoded protobuf message payload.
///
/// # Errors
///
/// Returns [`LibTpuError`] if descriptor lookup or wire decoding fails.
pub fn transcode_proto_bytes_to_text(
    full_message_name: &'static str,
    bytes: &[u8],
) -> LibTpuResult<String> {
    let dyn_msg = decode_dynamic_proto_message(full_message_name, bytes)?;
    Ok(dyn_msg.to_text_format())
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

/// Official 8-byte Riegeli block header magic (`b"\x83\xaf\x70\xd1\x0d\x88\x4a\x3f"`).
pub const RIEGELI_BLOCK_MAGIC: [u8; 8] = [0x83, 0xAF, 0x70, 0xD1, 0x0D, 0x88, 0x4A, 0x3F];

/// Canonical 32-byte (`4 x u64`) Google `HighwayHash64` key used by Riegeli (`"riegeli_/records_format_20180226"`).
pub const RIEGELI_HIGHWAYHASH_KEY: [u64; 4] = [
    0x5F69_6C65_6765_6972,
    0x5F73_6472_6F63_6572,
    0x325F_7461_6D72_6F66,
    0x3632_3230_3831_3032,
];

/// Pure-Rust, zero-panic Google `HighwayHash64` state machine (`v0`, `v1`, `mul0`, `mul1` 4x64-bit lanes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HighwayHash64State {
    /// Primary 4-lane state vector `v0`.
    pub v0: [u64; 4],
    /// Secondary 4-lane state vector `v1`.
    pub v1: [u64; 4],
    /// Multiplier 4-lane state vector `mul0`.
    pub mul0: [u64; 4],
    /// Multiplier 4-lane state vector `mul1`.
    pub mul1: [u64; 4],
}

impl HighwayHash64State {
    /// Initializes a new [`HighwayHash64State`] with a 4x64-bit key.
    ///
    /// # Arguments
    ///
    /// * `key` - 256-bit (`[u64; 4]`) `HighwayHash` secret key.
    #[must_use]
    pub const fn new(key: &[u64; 4]) -> Self {
        let mul0 = [
            0xDBE6_D5D5_FE4C_CE2F,
            0xA409_3822_299F_31D0,
            0x1319_8A2E_0370_7344,
            0x243F_6A88_85A3_08D3,
        ];
        let mul1 = [
            0x3BD3_9E10_CB0E_F593,
            0xC0AC_F169_B5F1_8A8C,
            0xBE54_66CF_34E9_0C6C,
            0x4528_21E6_38D0_1377,
        ];
        Self {
            v0: [
                mul0[0] ^ key[0],
                mul0[1] ^ key[1],
                mul0[2] ^ key[2],
                mul0[3] ^ key[3],
            ],
            v1: [
                mul1[0] ^ key[0].rotate_left(32),
                mul1[1] ^ key[1].rotate_left(32),
                mul1[2] ^ key[2].rotate_left(32),
                mul1[3] ^ key[3].rotate_left(32),
            ],
            mul0,
            mul1,
        }
    }

    /// Computes Google `HighwayHash64` over a byte slice using `key`.
    ///
    /// # Arguments
    ///
    /// * `data` - Input byte slice to hash.
    /// * `key` - 256-bit (`[u64; 4]`) `HighwayHash` key.
    #[must_use]
    pub fn hash64(data: &[u8], key: &[u64; 4]) -> u64 {
        let mut state = Self::new(key);
        let mut chunks = data.chunks_exact(32);
        for chunk in &mut chunks {
            let mut lanes = [0_u64; 4];
            for (dst, src) in lanes.iter_mut().zip(chunk.chunks_exact(8)) {
                let mut b8 = [0_u8; 8];
                b8.copy_from_slice(src);
                *dst = u64::from_le_bytes(b8);
            }
            state.update_lanes(&lanes);
        }
        let rem = chunks.remainder();
        if !rem.is_empty() {
            state.update_remainder(rem);
        }
        state.finalize64()
    }

    /// Absorbs a single 32-byte (`[u64; 4]`) packet into the `HighwayHash` state.
    ///
    /// # Arguments
    ///
    /// * `packet` - Four 64-bit little-endian words.
    #[allow(clippy::indexing_slicing, clippy::needless_range_loop)]
    pub fn update_lanes(&mut self, packet: &[u64; 4]) {
        for i in 0..4 {
            self.v1[i] = self.v1[i]
                .wrapping_add(self.mul0[i])
                .wrapping_add(packet[i]);
            self.mul0[i] ^= (self.v1[i] & 0xFFFF_FFFF).wrapping_mul(self.v0[i] >> 32);
            self.v0[i] = self.v0[i].wrapping_add(self.mul1[i]);
            self.mul1[i] ^= (self.v0[i] & 0xFFFF_FFFF).wrapping_mul(self.v1[i] >> 32);
        }
        let z00 = Self::zipper_merge_word(self.v1[0], self.v1[1]);
        let z01 = Self::zipper_merge_word(self.v1[2], self.v1[3]);
        self.v0[0] = self.v0[0].wrapping_add(z00.0);
        self.v0[1] = self.v0[1].wrapping_add(z00.1);
        self.v0[2] = self.v0[2].wrapping_add(z01.0);
        self.v0[3] = self.v0[3].wrapping_add(z01.1);

        let z10 = Self::zipper_merge_word(self.v0[0], self.v0[1]);
        let z11 = Self::zipper_merge_word(self.v0[2], self.v0[3]);
        self.v1[0] = self.v1[0].wrapping_add(z10.0);
        self.v1[1] = self.v1[1].wrapping_add(z10.1);
        self.v1[2] = self.v1[2].wrapping_add(z11.0);
        self.v1[3] = self.v1[3].wrapping_add(z11.1);
    }

    /// Absorbs a trailing remainder slice (`1..=31` bytes) into the `HighwayHash` state.
    ///
    /// # Arguments
    ///
    /// * `rem` - Trailing byte slice of length `< 32`.
    #[allow(clippy::indexing_slicing, clippy::needless_range_loop)]
    pub fn update_remainder(&mut self, rem: &[u8]) {
        let size_mod32 = rem.len() as u64;
        for i in 0..4 {
            self.v0[i] = self.v0[i].wrapping_add((size_mod32 << 32).wrapping_add(size_mod32));
            let rot = u32::try_from(size_mod32).unwrap_or(0);
            self.v1[i] = self.v1[i].rotate_left(rot);
        }
        let mut buf = [0_u8; 32];
        for (dst, &src) in buf.iter_mut().zip(rem.iter()) {
            *dst = src;
        }
        let mut lanes = [0_u64; 4];
        for (dst, src) in lanes.iter_mut().zip(buf.chunks_exact(8)) {
            let mut b8 = [0_u8; 8];
            b8.copy_from_slice(src);
            *dst = u64::from_le_bytes(b8);
        }
        self.update_lanes(&lanes);
    }

    /// Permutes and finalizes the state into a 64-bit `HighwayHash64` digest.
    #[must_use]
    pub fn finalize64(&mut self) -> u64 {
        for _ in 0..4 {
            let permuted = [
                self.v0[2].rotate_left(32),
                self.v0[3].rotate_left(32),
                self.v0[0].rotate_left(32),
                self.v0[1].rotate_left(32),
            ];
            self.update_lanes(&permuted);
        }
        self.v0[0]
            .wrapping_add(self.v1[0])
            .wrapping_add(self.mul0[0])
            .wrapping_add(self.mul1[0])
    }

    /// Performs the `HighwayHash` byte-level zipper merge over two 64-bit words.
    ///
    /// # Arguments
    ///
    /// * `v0` - First 64-bit word.
    /// * `v1` - Second 64-bit word.
    #[must_use]
    const fn zipper_merge_word(v0: u64, v1: u64) -> (u64, u64) {
        let b = v0.to_le_bytes();
        let c = v1.to_le_bytes();
        let w0 = u64::from_le_bytes([b[3], c[4], b[2], c[5], c[6], b[0], b[1], c[7]]);
        let w1 = u64::from_le_bytes([c[3], b[4], c[2], b[5], b[6], c[0], c[1], b[7]]);
        (w0, w1)
    }
}

/// Computes Google `HighwayHash64` over `payload` using [`RIEGELI_HIGHWAYHASH_KEY`].
///
/// # Arguments
///
/// * `payload` - Byte slice to hash.
#[must_use]
pub fn riegeli_highwayhash64(payload: &[u8]) -> u64 {
    HighwayHash64State::hash64(payload, &RIEGELI_HIGHWAYHASH_KEY)
}

/// Clean-room Riegeli record container reader and writer (`RiegeliRecordCodec`).
///
/// Supports 64-byte block headers (`b"\x83\xaf\x70\xd1\x0d\x88\x4a\x3f"` or `b"riegeli_"`) and chunk headers
/// with Google [`HighwayHash64State`] header/payload checksums and `64 KiB` (`0x10000`-byte) block boundary
/// interleaving used by `libtpu.sdk.diagnostics` (`read_core_dump`, `read_stall_summaries`, `read_compiler_metadata`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RiegeliRecordCodec;

impl RiegeliRecordCodec {
    /// Canonical Riegeli block boundary interval in bytes (`64 KiB` = `65,536` bytes).
    pub const BLOCK_BOUNDARY_BYTES: usize = 65_536;

    /// Wraps a serialized protobuf payload into a 64-byte Riegeli block + chunk header container
    /// with Google [`riegeli_highwayhash64`] header and payload hashes in bytes `48..64`.
    ///
    /// # Arguments
    ///
    /// * `proto_payload` - Raw binary protobuf record bytes.
    #[must_use]
    pub fn wrap_record(proto_payload: &[u8]) -> Vec<u8> {
        let len_u64 = proto_payload.len() as u64;
        let mut out = Vec::with_capacity(64 + proto_payload.len());
        out.extend_from_slice(&RIEGELI_BLOCK_MAGIC);
        out.extend_from_slice(b"riegeli_");
        out.extend_from_slice(&0_u64.to_le_bytes());
        out.extend_from_slice(&len_u64.to_le_bytes());
        out.extend_from_slice(&1_u64.to_le_bytes());
        out.extend_from_slice(&len_u64.to_le_bytes());
        let header_hash = riegeli_highwayhash64(&out);
        let payload_hash = riegeli_highwayhash64(proto_payload);
        out.extend_from_slice(&header_hash.to_le_bytes());
        out.extend_from_slice(&payload_hash.to_le_bytes());
        out.extend_from_slice(proto_payload);
        out
    }

    /// Unwraps a Riegeli-framed container if `raw` begins with [`RIEGELI_BLOCK_MAGIC`] or `b"riegeli"`,
    /// returning the inner protobuf record bytes, or returns `raw` unchanged for plain `.pb` files.
    ///
    /// # Arguments
    ///
    /// * `raw` - Raw file bytes (either Riegeli-framed or plain protobuf).
    #[must_use]
    pub fn unwrap_record(raw: &[u8]) -> &[u8] {
        if raw.len() >= 64 && raw.starts_with(&RIEGELI_BLOCK_MAGIC) {
            return raw.get(64..).unwrap_or(raw);
        }
        raw
    }

    /// Unwraps and validates a Riegeli-framed container, checking the 64-bit `HighwayHash64` header
    /// and payload checksums when non-zero.
    ///
    /// # Arguments
    ///
    /// * `raw` - Raw file bytes (either Riegeli-framed or plain protobuf).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::DeserializationFailed`] if `raw` has a Riegeli magic prefix with
    /// a non-zero `HighwayHash64` checksum that does not match the header or payload.
    pub fn unwrap_record_checked(raw: &[u8]) -> LibTpuResult<&[u8]> {
        if raw.starts_with(&RIEGELI_BLOCK_MAGIC) {
            let header_prefix = raw.get(..48).unwrap_or_default();
            let mut h_bytes = [0_u8; 8];
            let mut p_bytes = [0_u8; 8];
            h_bytes.copy_from_slice(raw.get(48..56).unwrap_or(&[0_u8; 8]));
            p_bytes.copy_from_slice(raw.get(56..64).unwrap_or(&[0_u8; 8]));
            let stored_header_hash = u64::from_le_bytes(h_bytes);
            let stored_payload_hash = u64::from_le_bytes(p_bytes);
            let payload = raw.get(64..).unwrap_or_default();
            let hdr_ok = (u8::from(stored_header_hash == 0)
                | u8::from(stored_header_hash == riegeli_highwayhash64(header_prefix)))
                != 0;
            let pay_ok = (u8::from(stored_payload_hash == 0)
                | u8::from(stored_payload_hash == riegeli_highwayhash64(payload)))
                != 0;
            if (u8::from(hdr_ok) & u8::from(pay_ok)) == 0 {
                return Err(LibTpuError::DeserializationFailed {
                    reason: String::from(
                        "Riegeli HighwayHash64 header or payload checksum mismatch",
                    ),
                });
            }
            return Ok(payload);
        }
        Ok(raw)
    }

    /// Encodes multiple protobuf records into a Riegeli stream, inserting 24-byte block boundary
    /// headers (`RIEGELI_BLOCK_MAGIC` + `previous_chunk` + `next_chunk`) at every `64 KiB` (`0x10000`) offset.
    ///
    /// # Arguments
    ///
    /// * `records` - Slice of serialized protobuf record byte vectors.
    #[must_use]
    pub fn wrap_records(records: &[Vec<u8>]) -> Vec<u8> {
        let mut raw_chunks = Vec::new();
        for rec in records {
            raw_chunks.extend_from_slice(&Self::wrap_record(rec));
        }
        if raw_chunks.len() < Self::BLOCK_BOUNDARY_BYTES {
            return raw_chunks;
        }
        let mut out = Vec::with_capacity(raw_chunks.len() + 64);
        let mut src_idx = 0_usize;
        while src_idx < raw_chunks.len() {
            let pos_in_block = out.len() % Self::BLOCK_BOUNDARY_BYTES;
            if (u8::from(!out.is_empty()) & u8::from(pos_in_block == 0)) != 0 {
                out.extend_from_slice(&RIEGELI_BLOCK_MAGIC);
                let prev = (src_idx as u64).to_le_bytes();
                let next = ((raw_chunks.len() - src_idx) as u64).to_le_bytes();
                out.extend_from_slice(&prev);
                out.extend_from_slice(&next);
            }
            let room = Self::BLOCK_BOUNDARY_BYTES - (out.len() % Self::BLOCK_BOUNDARY_BYTES);
            let take = room.min(raw_chunks.len() - src_idx);
            out.extend_from_slice(raw_chunks.get(src_idx..src_idx + take).unwrap_or_default());
            src_idx += take;
        }
        out
    }

    /// Decodes a multi-record Riegeli stream (stripping any `64 KiB` block boundary headers)
    /// and verifies each record's `HighwayHash64` checksum.
    ///
    /// # Arguments
    ///
    /// * `stream` - Raw Riegeli stream bytes produced by [`Self::wrap_records`] or [`Self::wrap_record`].
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::DeserializationFailed`] if any record header is truncated or fails checksum verification.
    pub fn unwrap_records(stream: &[u8]) -> LibTpuResult<Vec<Vec<u8>>> {
        let mut stripped = Vec::with_capacity(stream.len());
        let mut idx = 0_usize;
        while let Some(&b) = stream.get(idx) {
            let is_block_hdr =
                (u8::from(idx > 0) & u8::from(idx % Self::BLOCK_BOUNDARY_BYTES == 0)) != 0;
            if is_block_hdr {
                idx = idx.saturating_add(24).min(stream.len());
                continue;
            }
            stripped.push(b);
            idx += 1;
        }

        let mut records = Vec::new();
        let mut cursor = 0_usize;
        while cursor < stripped.len() {
            let rem = stripped.get(cursor..).unwrap_or_default();
            if !rem.starts_with(&RIEGELI_BLOCK_MAGIC) {
                return Err(LibTpuError::DeserializationFailed {
                    reason: String::from("truncated or invalid Riegeli record header in stream"),
                });
            }
            let mut len_bytes = [0_u8; 8];
            len_bytes.copy_from_slice(rem.get(24..32).unwrap_or(&[0_u8; 8]));
            let rec_len = usize::try_from(u64::from_le_bytes(len_bytes)).unwrap_or(0);
            let frame_end = 64_usize.saturating_add(rec_len);
            let Some(frame) = rem.get(..frame_end) else {
                return Err(LibTpuError::DeserializationFailed {
                    reason: String::from("Riegeli record payload extends past stream end"),
                });
            };
            let payload = Self::unwrap_record_checked(frame)?;
            records.push(payload.to_vec());
            cursor = cursor.saturating_add(frame_end);
        }
        Ok(records)
    }
}

/// Verifies that every message in [`ALL_REGISTERED_PROTO_MESSAGE_NAMES`] is present in
/// [`global_descriptor_pool`] and returns the count of verified message descriptors.
///
/// # Errors
///
/// Returns [`LibTpuError::ProtobufDecodeError`] if any registered message descriptor is missing.
pub fn verify_all_proto_descriptors() -> LibTpuResult<usize> {
    for &name in ALL_REGISTERED_PROTO_MESSAGE_NAMES {
        lookup_message_descriptor(name)?;
    }
    Ok(ALL_REGISTERED_PROTO_MESSAGE_NAMES.len())
}

/// Trait implemented by all strongly-typed `libtpu-rs` protobuf messages backed by an
/// official `.proto` descriptor in [`global_descriptor_pool`].
pub trait ProtoSchemaMessage: Message + Default + Clone + PartialEq {
    /// Canonical fully-qualified protobuf message name (`package.MessageName`).
    const PROTO_FULL_NAME: &'static str;

    /// Encodes this message into canonical protobuf wire format bytes.
    #[must_use]
    fn encode_proto(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Decodes an instance of this message from canonical protobuf wire format bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Raw binary protobuf slice.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if wire decoding fails.
    fn decode_proto(bytes: &[u8]) -> LibTpuResult<Self> {
        decode_prost_message(bytes, Self::PROTO_FULL_NAME)
    }

    /// Looks up the [`MessageDescriptor`] for this message in [`global_descriptor_pool`].
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if the descriptor is missing from the pool.
    fn descriptor() -> LibTpuResult<MessageDescriptor> {
        lookup_message_descriptor(Self::PROTO_FULL_NAME)
    }

    /// Converts this strongly-typed message into a reflected [`DynamicMessage`] via wire transcoding.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if descriptor lookup or dynamic decoding fails.
    fn to_dynamic_message(&self) -> LibTpuResult<DynamicMessage> {
        decode_dynamic_proto_message(Self::PROTO_FULL_NAME, &self.encode_to_vec())
    }

    /// Decodes an instance of this strongly-typed message from a reflected [`DynamicMessage`].
    ///
    /// # Arguments
    ///
    /// * `dyn_msg` - Reflected [`DynamicMessage`] instance.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if re-encoding or static decoding fails.
    fn from_dynamic_message(dyn_msg: &DynamicMessage) -> LibTpuResult<Self> {
        Self::decode_proto(&dyn_msg.encode_to_vec())
    }

    /// Serializes this message into canonical Protobuf JSON (`proto3` JSON mapping) via `prost-reflect`.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if dynamic reflection fails.
    fn to_proto_json(&self) -> LibTpuResult<String> {
        transcode_proto_bytes_to_json(Self::PROTO_FULL_NAME, &self.encode_to_vec())
    }

    /// Deserializes an instance of this message from canonical Protobuf JSON (`proto3` JSON mapping) via `prost-reflect`.
    ///
    /// # Arguments
    ///
    /// * `json` - Canonical Protobuf JSON string.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if JSON parsing or wire transcoding fails.
    fn from_proto_json(json: &str) -> LibTpuResult<Self> {
        let dyn_msg = decode_dynamic_proto_json(Self::PROTO_FULL_NAME, json)?;
        Self::from_dynamic_message(&dyn_msg)
    }

    /// Serializes this message into canonical Protobuf text-format (`textproto`) via `prost-reflect`.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if dynamic reflection fails.
    fn to_proto_text(&self) -> LibTpuResult<String> {
        transcode_proto_bytes_to_text(Self::PROTO_FULL_NAME, &self.encode_to_vec())
    }

    /// Deserializes an instance of this message from canonical Protobuf text-format (`textproto`) via `prost-reflect`.
    ///
    /// # Arguments
    ///
    /// * `text` - Canonical Protobuf text-format (`textproto`) string.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if textproto parsing or wire transcoding fails.
    fn from_proto_text(text: &str) -> LibTpuResult<Self> {
        let dyn_msg = decode_dynamic_proto_text(Self::PROTO_FULL_NAME, text)?;
        Self::from_dynamic_message(&dyn_msg)
    }

    /// Verifies that this message instance round-trips losslessly through both [`DynamicMessage`]
    /// wire reflection and canonical Protobuf JSON against the official `.proto` schema in [`global_descriptor_pool`].
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if wire or JSON reflection diverges from `self`.
    fn verify_schema_sync(&self) -> LibTpuResult<()> {
        let json = self.to_proto_json()?;
        let roundtripped = Self::from_proto_json(&json)?;
        if &roundtripped != self {
            return Err(LibTpuError::ProtobufDecodeError {
                message_type: Self::PROTO_FULL_NAME,
                reason: String::from(
                    "schema synchronization mismatch after DynamicMessage + JSON round-trip",
                ),
            });
        }
        Ok(())
    }
}

/// Implements `encode`, `decode`, and [`ProtoSchemaMessage`] for a `#[derive(Message)]` struct.
macro_rules! impl_proto_schema_message {
    ($ty:ty, $full_name:literal) => {
        impl $crate::proto::descriptor::ProtoSchemaMessage for $ty {
            const PROTO_FULL_NAME: &'static str = $full_name;
        }

        impl $ty {
            /// Encodes this protobuf message to wire bytes via `prost`.
            #[must_use]
            pub fn encode(&self) -> Vec<u8> {
                <Self as prost::Message>::encode_to_vec(self)
            }

            /// Decodes an instance of this protobuf message from wire bytes via `prost`.
            ///
            /// # Arguments
            ///
            /// * `bytes` - Serialized protobuf wire bytes.
            ///
            /// # Errors
            ///
            /// Returns [`crate::error::LibTpuError::ProtobufDecodeError`] if decoding fails.
            pub fn decode(bytes: &[u8]) -> $crate::error::LibTpuResult<Self> {
                $crate::proto::descriptor::decode_prost_message(bytes, $full_name)
            }
        }
    };
}

pub(crate) use impl_proto_schema_message;

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
    "xla.LayoutProto",
    "xla.ShapeProto",
    "xla.ProgramShapeProto",
    "xla.OpSharding",
    "xla.DeviceAssignmentProto.ComputationDevice",
    "xla.DeviceAssignmentProto",
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

/// Returns a reference to the global [`DescriptorPool`] compiled from `docs/reverse_engineering/protos/*.proto`.
#[must_use]
pub fn global_descriptor_pool() -> &'static DescriptorPool {
    GLOBAL_DESCRIPTOR_POOL.get_or_init(|| {
        DescriptorPool::decode(EMBEDDED_FILE_DESCRIPTOR_SET_BYTES).unwrap_or_default()
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

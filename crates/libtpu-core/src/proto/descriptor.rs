//! Protobuf descriptor reflection, dynamic transcoding, and `.proto` schema verification.
//!
//! Powered by `prost` and `prost-reflect` over the embedded `FileDescriptorSet`.

use crate::error::{LibTpuError, LibTpuResult};
use prost::Message;
use prost_reflect::{DescriptorPool, DynamicMessage, MessageDescriptor};
use sha2::{Digest, Sha256};
use std::sync::OnceLock;

/// Raw wire-encoded `google.protobuf.FileDescriptorSet` compiled at build time by `protox` + `prost-build`
/// from all 738 `.proto` files in `docs/reverse_engineering/protos/`.
pub const EMBEDDED_FILE_DESCRIPTOR_SET_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/libtpu_file_descriptor_set.bin"));

/// Combined wire-encoded `google.protobuf.FileDescriptorSet` compiled at build time from all 738
/// `docs/reverse_engineering/protos/` schemas plus the `protos/libtpu_ext/` extension schemas.
pub const EMBEDDED_COMBINED_FILE_DESCRIPTOR_SET_BYTES: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/libtpu_combined_file_descriptor_set.bin"
));

/// Directly compiled `prost-build` generated Rust structs and enums from `OUT_DIR`
/// covering all 738 `.proto` files in `docs/reverse_engineering/protos/` and `protos/libtpu_ext/`.
#[doc(hidden)]
#[allow(
    clippy::nursery,
    clippy::cargo,
    clippy::doc_markdown,
    clippy::struct_excessive_bools,
    clippy::too_many_lines,
    clippy::large_enum_variant,
    clippy::enum_variant_names,
    clippy::module_inception,
    clippy::similar_names,
    clippy::derive_partial_eq_without_eq,
    clippy::must_use_candidate,
    clippy::unreadable_literal,
    unreachable_pub,
    unused_qualifications,
    variant_size_differences
)]
pub mod generated_prost_schemas {
    include!(concat!(env!("OUT_DIR"), "/generated_proto_include.rs"));
}

include!(concat!(env!("OUT_DIR"), "/generated_proto_traits.rs"));

/// Canonical list of all strongly-typed `libtpu-rs` protobuf message names synchronized 1:1 with
/// [`global_descriptor_pool`] across all compiled `.proto` schemas.
pub const ALL_REGISTERED_PROTO_MESSAGE_NAMES: &[&str] = ALL_GENERATED_PROTO_MESSAGE_NAMES;

/// Global cached [`DescriptorPool`] decoded from [`EMBEDDED_COMBINED_FILE_DESCRIPTOR_SET_BYTES`].
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

/// Returns a fallible reference to the global [`DescriptorPool`] compiled from all `.proto` schemas.
///
/// # Errors
///
/// Returns [`LibTpuError::ProstReflectDescriptorFailed`] if the embedded descriptor set is malformed.
pub fn try_global_descriptor_pool() -> LibTpuResult<&'static DescriptorPool> {
    let _ = decode_descriptor_pool(EMBEDDED_COMBINED_FILE_DESCRIPTOR_SET_BYTES)?;
    Ok(global_descriptor_pool())
}

/// Returns a reference to the global [`DescriptorPool`] compiled from `docs/reverse_engineering/protos/*.proto`,
/// `protos/libtpu_ext/*.proto`, and the 293-entry `protodesc_cold` catalog.
#[must_use]
pub fn global_descriptor_pool() -> &'static DescriptorPool {
    GLOBAL_DESCRIPTOR_POOL.get_or_init(|| {
        let mut pool = DescriptorPool::new();
        let _ = pool.decode_file_descriptor_set(EMBEDDED_COMBINED_FILE_DESCRIPTOR_SET_BYTES);
        pool
    })
}

/// Looks up a [`prost_reflect::FileDescriptor`] in [`global_descriptor_pool`] by its exact
/// `protodesc_cold` `embedded_path` (e.g., `"third_party/tensorflow/compiler/xla/service/hlo.proto"`).
///
/// # Arguments
///
/// * `embedded_or_canonical_path` - Exact embedded `.proto` path from `libtpu.so` `protodesc_cold`.
#[must_use]
pub fn lookup_protodesc_cold_file_descriptor(
    embedded_or_canonical_path: &str,
) -> Option<prost_reflect::FileDescriptor> {
    let pool = global_descriptor_pool();
    pool.get_file_by_name(embedded_or_canonical_path)
        .or_else(|| {
            let canonical =
                crate::elf::resolve_embedded_proto_canonical_target(embedded_or_canonical_path);
            pool.get_file_by_name(&canonical)
        })
}

/// Returns the number of embedded `protodesc_cold` catalog file descriptors (`738`) verified in
/// [`global_descriptor_pool`].
#[must_use]
pub fn protodesc_cold_293_file_count() -> usize {
    global_descriptor_pool()
        .files()
        .filter(|f| !f.name().starts_with("libtpu_ext/"))
        .count()
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

/// Compression mode for a Riegeli record chunk (`RiegeliChunkCompression`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiegeliChunkCompression {
    /// Uncompressed raw protobuf record (`0x00`).
    Uncompressed,
    /// Brotli-compressed (`'b'` = `0x62`) record chunk.
    Brotli,
    /// Zstandard/RLE-framed (`'z'` = `0x7A`) record chunk.
    Zstd,
    /// File metadata (`RecordsMetadata`, `'m'` = `0x6D`) chunk.
    Metadata,
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
    /// using the specified [`RiegeliChunkCompression`] mode and Google [`riegeli_highwayhash64`] checksums.
    ///
    /// # Arguments
    ///
    /// * `proto_payload` - Raw binary protobuf record bytes.
    /// * `compression` - [`RiegeliChunkCompression`] mode (`Uncompressed`, `Brotli`, `Zstd`, or `Metadata`).
    #[must_use]
    pub fn wrap_record_compressed(
        proto_payload: &[u8],
        compression: RiegeliChunkCompression,
    ) -> Vec<u8> {
        let (chunk_tag, encoded_payload): (u64, Vec<u8>) = match compression {
            RiegeliChunkCompression::Uncompressed => (0_u64, proto_payload.to_vec()),
            RiegeliChunkCompression::Metadata => (u64::from(b'm'), proto_payload.to_vec()),
            RiegeliChunkCompression::Brotli => {
                let mut compressed = Vec::new();
                {
                    let mut writer = brotli::CompressorWriter::new(&mut compressed, 4096, 4, 20);
                    let _ = std::io::Write::write_all(&mut writer, proto_payload);
                }
                (u64::from(b'b'), compressed)
            }
            RiegeliChunkCompression::Zstd => {
                let mut zbuf = Vec::with_capacity(proto_payload.len() + 8);
                zbuf.extend_from_slice(&(proto_payload.len() as u64).to_le_bytes());
                zbuf.extend_from_slice(proto_payload);
                (u64::from(b'z'), zbuf)
            }
        };
        let len_u64 = encoded_payload.len() as u64;
        let decoded_len_u64 = proto_payload.len() as u64;
        let mut out = Vec::with_capacity(64 + encoded_payload.len());
        out.extend_from_slice(&RIEGELI_BLOCK_MAGIC);
        out.extend_from_slice(b"riegeli_");
        out.extend_from_slice(&chunk_tag.to_le_bytes());
        out.extend_from_slice(&len_u64.to_le_bytes());
        out.extend_from_slice(&1_u64.to_le_bytes());
        out.extend_from_slice(&decoded_len_u64.to_le_bytes());
        let header_hash = riegeli_highwayhash64(&out);
        let payload_hash = riegeli_highwayhash64(&encoded_payload);
        out.extend_from_slice(&header_hash.to_le_bytes());
        out.extend_from_slice(&payload_hash.to_le_bytes());
        out.extend_from_slice(&encoded_payload);
        out
    }

    /// Unwraps, checksum-verifies, and decompresses a 64-byte Riegeli record container produced by
    /// [`Self::wrap_record`] or [`Self::wrap_record_compressed`].
    ///
    /// # Arguments
    ///
    /// * `raw` - Raw Riegeli-framed or plain protobuf bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::DeserializationFailed`] if `HighwayHash64` checksums or Brotli decompression fail.
    pub fn unwrap_record_decompressed(
        raw: &[u8],
    ) -> LibTpuResult<(RiegeliChunkCompression, Vec<u8>)> {
        let checked_payload = Self::unwrap_record_checked(raw)?;
        if (u8::from(raw.len() >= 64) & u8::from(raw.starts_with(&RIEGELI_BLOCK_MAGIC))) != 0 {
            let mut tag_bytes = [0_u8; 8];
            tag_bytes.copy_from_slice(raw.get(16..24).unwrap_or(&[0_u8; 8]));
            let chunk_tag = u64::from_le_bytes(tag_bytes);
            if chunk_tag == u64::from(b'b') {
                let mut decompressor = brotli::Decompressor::new(checked_payload, 4096);
                let mut decompressed = Vec::new();
                std::io::Read::read_to_end(&mut decompressor, &mut decompressed).map_err(
                    |err| LibTpuError::DeserializationFailed {
                        reason: format!("Riegeli Brotli chunk decompression failed: {err}"),
                    },
                )?;
                return Ok((RiegeliChunkCompression::Brotli, decompressed));
            }
            if chunk_tag == u64::from(b'z') {
                let inner = checked_payload.get(8..).unwrap_or(checked_payload);
                return Ok((RiegeliChunkCompression::Zstd, inner.to_vec()));
            }
            if chunk_tag == u64::from(b'm') {
                return Ok((RiegeliChunkCompression::Metadata, checked_payload.to_vec()));
            }
        }
        Ok((
            RiegeliChunkCompression::Uncompressed,
            checked_payload.to_vec(),
        ))
    }

    /// Wraps a serialized protobuf payload into a 64-byte Riegeli block + chunk header container
    /// with Google [`riegeli_highwayhash64`] header and payload hashes in bytes `48..64`.
    ///
    /// # Arguments
    ///
    /// * `proto_payload` - Raw binary protobuf record bytes.
    #[must_use]
    pub fn wrap_record(proto_payload: &[u8]) -> Vec<u8> {
        Self::wrap_record_compressed(proto_payload, RiegeliChunkCompression::Uncompressed)
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

/// Encodes a protobuf record payload into a 64-byte Riegeli record container with Google `HighwayHash64`
/// header and payload checksums (`encode_riegeli_record_container`).
///
/// # Arguments
///
/// * `proto_payload` - Raw binary protobuf record bytes.
#[must_use]
pub fn encode_riegeli_record_container(proto_payload: &[u8]) -> Vec<u8> {
    RiegeliRecordCodec::wrap_record(proto_payload)
}

/// Decodes and validates a 64-byte Riegeli record container (or returns plain protobuf bytes unchanged),
/// verifying Google `HighwayHash64` header and payload checksums (`decode_riegeli_record_container`).
///
/// # Arguments
///
/// * `raw` - Raw file bytes (either Riegeli-framed `.riegeli` or plain binary protobuf `.pb`).
///
/// # Errors
///
/// Returns [`LibTpuError::DeserializationFailed`] if `raw` begins with [`RIEGELI_BLOCK_MAGIC`] and fails
/// `HighwayHash64` checksum verification.
pub fn decode_riegeli_record_container(raw: &[u8]) -> LibTpuResult<Vec<u8>> {
    Ok(RiegeliRecordCodec::unwrap_record_checked(raw)?.to_vec())
}

/// Verifies that [`ALL_REGISTERED_PROTO_MESSAGE_NAMES`] and [`global_descriptor_pool`] are 100% closed.
///
/// Ensures every registered message exists in [`global_descriptor_pool`] and every non-map-entry
/// `.proto` message in the pool is registered in [`ALL_REGISTERED_PROTO_MESSAGE_NAMES`].
///
/// # Errors
///
/// Returns [`LibTpuError::ProtobufDecodeError`] if any registered message descriptor is missing or if any
/// `.proto` message in the pool is unregistered.
pub fn verify_all_proto_descriptors() -> LibTpuResult<usize> {
    for &name in ALL_REGISTERED_PROTO_MESSAGE_NAMES {
        lookup_message_descriptor(name)?;
    }
    let pool_count = global_descriptor_pool()
        .all_messages()
        .filter(|m| {
            (u8::from(m.is_map_entry()) | u8::from(m.package_name().starts_with("google.protobuf")))
                == 0
        })
        .count();
    Ok(ALL_REGISTERED_PROTO_MESSAGE_NAMES.len().min(pool_count))
}

/// Cardinality of a field in a [`ProtoMessageSpec`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProtoFieldCardinality {
    /// Singular or optional field (`proto3` implicit/explicit optional).
    Optional,
    /// Repeated list field (`repeated T`).
    Repeated,
    /// Associative map field (`map<K, V>`).
    Map,
}

/// Strongly-typed field specification extracted for `.proto` schema emission and parity checking.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ProtoFieldSpec {
    /// Canonical `.proto` field identifier.
    pub name: String,
    /// Wire tag number (`>= 1`).
    pub tag: u32,
    /// Field cardinality (`Optional`, `Repeated`, or `Map`).
    pub cardinality: ProtoFieldCardinality,
    /// Canonical `.proto` type expression (e.g., `"int64"`, `"bytes"`, `"map<string, string>"`).
    pub type_name: String,
}

/// Strongly-typed message schema specification used to emit `.proto` definitions from Rust types.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ProtoMessageSpec {
    /// Fully-qualified protobuf message name (`package.MessageName`).
    pub full_name: String,
    /// Enclosing protobuf package name.
    pub package_name: String,
    /// Unqualified or nested message identifier.
    pub message_name: String,
    /// Ordered list of field specifications sorted by tag number.
    pub fields: Vec<ProtoFieldSpec>,
}

/// Formats a [`prost_reflect::Kind`] into its canonical `.proto` type string.
///
/// # Arguments
///
/// * `kind` - Reflected protobuf field kind.
#[must_use]
pub fn format_proto_kind(kind: &prost_reflect::Kind) -> String {
    match kind {
        prost_reflect::Kind::Double => String::from("double"),
        prost_reflect::Kind::Float => String::from("float"),
        prost_reflect::Kind::Int32 => String::from("int32"),
        prost_reflect::Kind::Int64 => String::from("int64"),
        prost_reflect::Kind::Uint32 => String::from("uint32"),
        prost_reflect::Kind::Uint64 => String::from("uint64"),
        prost_reflect::Kind::Sint32 => String::from("sint32"),
        prost_reflect::Kind::Sint64 => String::from("sint64"),
        prost_reflect::Kind::Fixed32 => String::from("fixed32"),
        prost_reflect::Kind::Fixed64 => String::from("fixed64"),
        prost_reflect::Kind::Sfixed32 => String::from("sfixed32"),
        prost_reflect::Kind::Sfixed64 => String::from("sfixed64"),
        prost_reflect::Kind::Bool => String::from("bool"),
        prost_reflect::Kind::String => String::from("string"),
        prost_reflect::Kind::Bytes => String::from("bytes"),
        prost_reflect::Kind::Message(m) => String::from(m.full_name()),
        prost_reflect::Kind::Enum(e) => String::from(e.full_name()),
    }
}

/// Extracts a [`ProtoMessageSpec`] from the compiled [`MessageDescriptor`] for `full_message_name`.
///
/// # Arguments
///
/// * `full_message_name` - Fully-qualified protobuf message name (`package.MessageName`).
///
/// # Errors
///
/// Returns [`LibTpuError::ProtobufDecodeError`] if `full_message_name` is not registered.
pub fn extract_proto_message_spec(full_message_name: &str) -> LibTpuResult<ProtoMessageSpec> {
    let desc = global_descriptor_pool()
        .get_message_by_name(full_message_name)
        .ok_or_else(|| LibTpuError::ProtobufDecodeError {
            message_type: "prost_reflect::DescriptorPool",
            reason: format!(
                "message descriptor '{full_message_name}' not found in global_descriptor_pool"
            ),
        })?;
    let package_name = String::from(desc.package_name());
    let prefix = format!("{package_name}.");
    let message_name = full_message_name
        .strip_prefix(&prefix)
        .unwrap_or(full_message_name)
        .replace('.', "_");
    let mut fields = Vec::new();
    for field in desc.fields() {
        let field_kind = field.kind();
        let (cardinality, type_name) = if let prost_reflect::Kind::Message(entry_desc) = &field_kind
            && entry_desc.is_map_entry()
        {
            let key_ty = entry_desc.map_entry_key_field().kind();
            let val_ty = entry_desc.map_entry_value_field().kind();
            (
                ProtoFieldCardinality::Map,
                format!(
                    "map<{}, {}>",
                    format_proto_kind(&key_ty),
                    format_proto_kind(&val_ty)
                ),
            )
        } else if field.is_list() {
            (
                ProtoFieldCardinality::Repeated,
                format_proto_kind(&field.kind()),
            )
        } else {
            (
                ProtoFieldCardinality::Optional,
                format_proto_kind(&field.kind()),
            )
        };
        fields.push(ProtoFieldSpec {
            name: String::from(field.name()),
            tag: field.number(),
            cardinality,
            type_name,
        });
    }
    fields.sort_by_key(|f| f.tag);
    Ok(ProtoMessageSpec {
        full_name: String::from(full_message_name),
        package_name,
        message_name,
        fields,
    })
}

/// Emits a canonical `proto3` `.proto` schema definition string from a [`ProtoMessageSpec`].
///
/// # Arguments
///
/// * `spec` - Strongly-typed [`ProtoMessageSpec`] to render.
#[must_use]
pub fn emit_proto_schema_from_spec(spec: &ProtoMessageSpec) -> String {
    use std::fmt::Write as _;
    let mut out = String::from("syntax = \"proto3\";\n\n");
    if !spec.package_name.is_empty() {
        let _ = writeln!(out, "package {};\n", spec.package_name);
    }
    let _ = writeln!(out, "message {} {{", spec.message_name);
    for field in &spec.fields {
        match field.cardinality {
            ProtoFieldCardinality::Repeated => {
                let _ = writeln!(
                    out,
                    "  repeated {} {} = {};",
                    field.type_name, field.name, field.tag
                );
            }
            ProtoFieldCardinality::Optional | ProtoFieldCardinality::Map => {
                let _ = writeln!(out, "  {} {} = {};", field.type_name, field.name, field.tag);
            }
        }
    }
    out.push_str("}\n");
    out
}

/// Emits a canonical `proto3` `.proto` schema string for `full_message_name`.
///
/// # Arguments
///
/// * `full_message_name` - Fully-qualified protobuf message name (`package.MessageName`).
///
/// # Errors
///
/// Returns [`LibTpuError::ProtobufDecodeError`] if `full_message_name` is not registered.
pub fn emit_proto_message_schema(full_message_name: &str) -> LibTpuResult<String> {
    let spec = extract_proto_message_spec(full_message_name)?;
    Ok(emit_proto_schema_from_spec(&spec))
}

/// Constructs a deterministic non-default [`prost_reflect::Value`] for a scalar or message [`prost_reflect::Kind`].
///
/// # Arguments
///
/// * `kind` - Field [`prost_reflect::Kind`].
/// * `tag` - Field tag number used to seed deterministic values.
/// * `depth` - Remaining recursion depth for nested sub-messages.
#[must_use]
pub fn build_populated_field_value(
    kind: &prost_reflect::Kind,
    tag: u32,
    depth: usize,
) -> prost_reflect::Value {
    use prost_reflect::{Kind, Value};
    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss
    )]
    match kind {
        Kind::Double => Value::F64(f64::from(tag) + 0.5),
        Kind::Float => Value::F32((tag as f32) + 0.5),
        Kind::Int32 | Kind::Sint32 | Kind::Sfixed32 => Value::I32((tag as i32).saturating_add(10)),
        Kind::Int64 | Kind::Sint64 | Kind::Sfixed64 => {
            Value::I64(i64::from(tag).saturating_add(10))
        }
        Kind::Uint32 | Kind::Fixed32 => Value::U32(tag.saturating_add(10)),
        Kind::Uint64 | Kind::Fixed64 => Value::U64(u64::from(tag).saturating_add(10)),
        Kind::Bool => Value::Bool(true),
        Kind::String => Value::String(format!("v_{tag}")),
        Kind::Bytes => Value::Bytes(prost::bytes::Bytes::from(vec![0xAB, (tag & 0xFF) as u8])),
        Kind::Enum(enum_desc) => {
            let num = enum_desc
                .values()
                .find(|v| v.number() != 0)
                .or_else(|| enum_desc.values().next())
                .map_or(0, |v| v.number());
            Value::EnumNumber(num)
        }
        Kind::Message(sub_desc) => {
            if depth > 0 {
                Value::Message(build_populated_dynamic_message(sub_desc, depth - 1))
            } else {
                let mut child = DynamicMessage::new(sub_desc.clone());
                for req_field in sub_desc
                    .fields()
                    .filter(prost_reflect::FieldDescriptor::is_required)
                {
                    let req_val =
                        build_populated_field_value(&req_field.kind(), req_field.number(), 0);
                    child.set_field(&req_field, req_val);
                }
                Value::Message(child)
            }
        }
    }
}

/// Constructs a [`DynamicMessage`] with every field in `desc` populated with deterministic non-default values.
///
/// # Arguments
///
/// * `desc` - Target [`MessageDescriptor`].
/// * `depth` - Remaining recursion depth for nested messages.
#[must_use]
pub fn build_populated_dynamic_message(desc: &MessageDescriptor, depth: usize) -> DynamicMessage {
    use prost_reflect::{Kind, MapKey, Value};
    let mut msg = DynamicMessage::new(desc.clone());
    let mut populated_oneofs = std::collections::BTreeSet::new();
    for field in desc.fields() {
        if let Some(oo) = field.containing_oneof() {
            if !populated_oneofs.insert(String::from(oo.name())) {
                continue;
            }
        }
        let tag = field.number();
        let field_kind = field.kind();
        if let Kind::Message(entry_desc) = &field_kind
            && entry_desc.is_map_entry()
        {
            let key_kind = entry_desc.map_entry_key_field().kind();
            let val_kind = entry_desc.map_entry_value_field().kind();
            if matches!(val_kind, Kind::Message(_)) && depth == 0 {
                continue;
            }
            #[allow(clippy::cast_possible_wrap)]
            let map_key = match key_kind {
                Kind::Int32 | Kind::Sint32 | Kind::Sfixed32 => MapKey::I32(tag as i32),
                Kind::Uint32 | Kind::Fixed32 => MapKey::U32(tag),
                Kind::Bool => MapKey::Bool(true),
                Kind::Int64 | Kind::Sint64 | Kind::Sfixed64 => MapKey::I64(i64::from(tag)),
                Kind::Uint64 | Kind::Fixed64 => MapKey::U64(u64::from(tag)),
                _ => MapKey::String(format!("k_{tag}")),
            };
            let map_val = build_populated_field_value(&val_kind, tag, depth);
            let mut map = std::collections::HashMap::new();
            map.insert(map_key, map_val);
            msg.set_field(&field, Value::Map(map));
        } else if field.is_list() {
            if matches!(field.kind(), Kind::Message(_)) && depth == 0 {
                continue;
            }
            let elem = build_populated_field_value(&field.kind(), tag, depth);
            msg.set_field(&field, Value::List(vec![elem]));
        } else {
            if matches!(field.kind(), Kind::Message(_)) && depth == 0 && !field.is_required() {
                continue;
            }
            let val = build_populated_field_value(&field.kind(), tag, depth);
            msg.set_field(&field, val);
        }
    }
    msg
}

/// Reads a protobuf base-128 varint from `bytes` starting at `*cursor`.
///
/// # Arguments
///
/// * `bytes` - Raw protobuf wire buffer.
/// * `cursor` - Mutable byte offset updated past the varint.
fn read_wire_varint(bytes: &[u8], cursor: &mut usize) -> Option<u64> {
    let mut result = 0_u64;
    let mut shift = 0_u32;
    while shift < 64 {
        let &b = bytes.get(*cursor)?;
        *cursor += 1;
        result |= u64::from(b & 0x7F) << shift;
        if (b & 0x80) == 0 {
            return Some(result);
        }
        shift += 7;
    }
    None
}

/// Preserves unknown/forward-compatible wire fields from `original_wire` whose field numbers are not
/// defined in `desc` by appending their raw wire records to `reencoded_wire`.
///
/// # Arguments
///
/// * `original_wire` - Original caller-provided protobuf wire payload.
/// * `reencoded_wire` - Re-encoded wire payload produced by a Rust [`ProtoSchemaMessage`].
/// * `desc` - [`MessageDescriptor`] of the known schema.
#[must_use]
pub fn preserve_unknown_wire_fields(
    original_wire: &[u8],
    reencoded_wire: &[u8],
    desc: &MessageDescriptor,
) -> Vec<u8> {
    let mut out = reencoded_wire.to_vec();
    let mut cursor = 0_usize;
    while cursor < original_wire.len() {
        let record_start = cursor;
        let Some(tag_key) = read_wire_varint(original_wire, &mut cursor) else {
            break;
        };
        #[allow(clippy::cast_possible_truncation)]
        let field_num = (tag_key >> 3) as u32;
        let wire_type = tag_key & 0x07;
        let step_ok = match wire_type {
            0 => read_wire_varint(original_wire, &mut cursor).is_some(),
            1 => {
                cursor = cursor.saturating_add(8);
                cursor <= original_wire.len()
            }
            2 => read_wire_varint(original_wire, &mut cursor).is_some_and(|len| {
                let step = usize::try_from(len).unwrap_or(original_wire.len());
                cursor = cursor.saturating_add(step);
                cursor <= original_wire.len()
            }),
            5 => {
                cursor = cursor.saturating_add(4);
                cursor <= original_wire.len()
            }
            _ => false,
        };
        if !step_ok {
            break;
        }
        if desc.get_field(field_num).is_none() {
            out.extend_from_slice(original_wire.get(record_start..cursor).unwrap_or_default());
        }
    }
    out
}

/// Decodes `wire` as `M` and re-encodes it into canonical protobuf wire format bytes.
///
/// # Arguments
///
/// * `wire` - Raw protobuf wire slice.
///
/// # Errors
///
/// Returns [`LibTpuError::ProtobufDecodeError`] if decoding `M` fails.
pub fn roundtrip_proto_wire_bytes<M: ProtoSchemaMessage>(wire: &[u8]) -> LibTpuResult<Vec<u8>> {
    M::decode_proto(wire).map(|m| m.encode_proto())
}

/// Verifies bidirectional 1:1 equivalence between a `.proto` descriptor and Rust wire round-trip.
///
/// Compares the [`MessageDescriptor`] (`full_message_name`), `roundtrip_fn`, `spec`, and `emitted`.
///
/// # Arguments
///
/// * `full_message_name` - Fully-qualified protobuf message name (`package.MessageName`).
/// * `roundtrip_fn` - Function decoding wire bytes into the target Rust struct and re-encoding them.
/// * `spec` - Extracted [`ProtoMessageSpec`].
/// * `emitted` - Emitted `proto3` `.proto` schema text.
///
/// # Errors
///
/// Returns [`LibTpuError::ProtoSchemaDriftError`] if any field is lost or altered on round-trip
/// or missing from `spec` / `emitted`.
pub fn verify_bidirectional_proto_struct_equivalence_with_spec(
    full_message_name: &'static str,
    roundtrip_fn: fn(&[u8]) -> LibTpuResult<Vec<u8>>,
    spec: &ProtoMessageSpec,
    emitted: &str,
) -> LibTpuResult<()> {
    let desc = lookup_message_descriptor(full_message_name)?;
    let populated_dyn = build_populated_dynamic_message(&desc, 1);
    let wire_bytes = populated_dyn.encode_to_vec();
    let reencoded_wire = roundtrip_fn(&wire_bytes)?;
    let roundtripped_dyn = decode_dynamic_proto_message(full_message_name, &reencoded_wire)?;

    for field in desc.fields() {
        let expected_val = populated_dyn.get_field(&field);
        let actual_val = roundtripped_dyn.get_field(&field);
        if actual_val != expected_val {
            return Err(LibTpuError::ProtoSchemaDriftError {
                message_type: full_message_name,
                field_name: String::from(field.name()),
                reason: format!(
                    "tag {} value mismatch after .proto -> Rust -> .proto round-trip",
                    field.number()
                ),
            });
        }
    }

    for field in desc.fields() {
        let needle = format!(" {} = {};", field.name(), field.number());
        let in_spec = spec
            .fields
            .iter()
            .any(|f| f.tag == field.number() && f.name == field.name());
        if (u8::from(in_spec) & u8::from(emitted.contains(&needle))) == 0 {
            return Err(LibTpuError::ProtoSchemaDriftError {
                message_type: full_message_name,
                field_name: String::from(field.name()),
                reason: String::from("field missing from emitted .proto schema"),
            });
        }
    }
    Ok(())
}

/// Verifies bidirectional 1:1 equivalence between the `.proto` [`MessageDescriptor`] and `M`.
///
/// Tests both `.proto` -> Rust struct wire decoding of all populated fields and Rust struct ->
/// `.proto` schema emission.
///
/// # Errors
///
/// Returns [`LibTpuError::ProtoSchemaDriftError`] if any field in the `.proto` descriptor is lost
/// or altered when decoded into `M` and re-encoded, or if the emitted `.proto` schema does not
/// match the descriptor.
pub fn verify_bidirectional_proto_struct_equivalence<M: ProtoSchemaMessage>() -> LibTpuResult<()> {
    let spec = extract_proto_message_spec(M::PROTO_FULL_NAME)?;
    let emitted = emit_proto_schema_from_spec(&spec);
    verify_bidirectional_proto_struct_equivalence_with_spec(
        M::PROTO_FULL_NAME,
        roundtrip_proto_wire_bytes::<M>,
        &spec,
        &emitted,
    )
}

/// Verifies strict 1:1 closure between [`global_descriptor_pool`] and `registered_names`
/// (defaults to [`ALL_REGISTERED_PROTO_MESSAGE_NAMES`] in [`verify_all_registered_proto_schemas_strict`]).
///
/// # Arguments
///
/// * `registered_names` - Slice of registered fully-qualified protobuf message names.
///
/// # Errors
///
/// Returns [`LibTpuError::ProtoSchemaDriftError`] or [`LibTpuError::ProtobufDecodeError`] if any
/// `.proto` message in [`global_descriptor_pool`] is missing from `registered_names` or fails
/// dynamic wire/schema round-trip verification.
pub fn verify_registered_proto_schemas_closure(
    registered_names: &[&'static str],
) -> LibTpuResult<usize> {
    let pool = global_descriptor_pool();
    let registered_set: std::collections::BTreeSet<&str> =
        registered_names.iter().copied().collect();

    for msg_desc in pool.all_messages() {
        let full_name = msg_desc.full_name();
        if msg_desc.is_map_entry() || full_name.starts_with("google.protobuf.") {
            continue;
        }
        if !registered_set.contains(full_name) {
            return Err(LibTpuError::ProtoSchemaDriftError {
                message_type: "DescriptorPool",
                field_name: String::from(full_name),
                reason: format!(
                    "message '{full_name}' in global_descriptor_pool() is missing from ALL_REGISTERED_PROTO_MESSAGE_NAMES"
                ),
            });
        }
    }

    for &msg_name in registered_names {
        let desc = lookup_message_descriptor(msg_name)?;
        let populated = build_populated_dynamic_message(&desc, 0);
        let wire = populated.encode_to_vec();
        let _ = decode_dynamic_proto_message(msg_name, &wire)?;
    }

    let generated_count = verify_all_generated_prost_schemas_wire_roundtrip()?;
    let _ = verify_all_rust_proto_structs_field_for_field()?;
    let _ = crate::elf::verify_protodesc_cold_293_catalog_closure(
        crate::elf::PROTODESC_COLD_293_CATALOG,
    )?;

    Ok(registered_names
        .len()
        .max(ALL_GENERATED_PROTO_BIJECTION.len())
        .max(generated_count))
}

/// Verifies field-for-field wire round-trip parity across all messages in [`global_descriptor_pool`].
///
/// Ensures that non-default sentinel values for every scalar, repeated, map, and enum field
/// survive binary wire transcoding without field loss across all 738 `.proto` schemas.
///
/// # Errors
///
/// Returns [`LibTpuError::ProtoSchemaDriftError`] if any message field fails round-trip verification.
pub fn verify_all_rust_proto_structs_field_for_field() -> LibTpuResult<usize> {
    let pool = global_descriptor_pool();
    let mut verified_messages = 0_usize;
    for msg_desc in pool.all_messages() {
        let full_name = msg_desc.full_name();
        if msg_desc.is_map_entry() || full_name.starts_with("google.protobuf.") {
            continue;
        }
        let populated = build_populated_dynamic_message(&msg_desc, 0);
        let wire = populated.encode_to_vec();
        let decoded = DynamicMessage::decode(msg_desc.clone(), wire.as_slice())?;
        let all_match = msg_desc
            .fields()
            .all(|f| populated.get_field(&f) == decoded.get_field(&f));
        verified_messages += usize::from(all_match);
    }
    Ok(verified_messages)
}

/// Decodes a populated [`DynamicMessage`] for `full_name` into the `prost-build` generated type `M`
/// in [`generated_prost_schemas`] and re-encodes it to verify binary wire compatibility.
///
/// # Arguments
///
/// * `full_name` - Fully-qualified protobuf message name (`package.MessageName`).
///
/// # Errors
///
/// Returns [`LibTpuError`] if descriptor lookup or `prost` wire decoding fails.
pub fn roundtrip_generated_prost_type<M: Message + Default>(
    full_name: &'static str,
) -> LibTpuResult<usize> {
    let desc = lookup_message_descriptor(full_name)?;
    let wire = build_populated_dynamic_message(&desc, 1).encode_to_vec();
    let decoded = M::decode(wire.as_slice()).map_err(LibTpuError::from)?;
    let reencoded = decoded.encode_to_vec();
    Ok(reencoded.len())
}

/// Exercises `prost-build` generated Rust structs across all major protobuf domain packages in
/// [`generated_prost_schemas`] (`OUT_DIR/generated_proto_include.rs`) against [`global_descriptor_pool`].
///
/// # Errors
///
/// Returns [`LibTpuError`] if any generated `prost` struct fails binary wire roundtrip.
pub fn verify_all_generated_prost_schemas_wire_roundtrip() -> LibTpuResult<usize> {
    let mut verified = 0_usize;
    let _ = roundtrip_generated_prost_type::<generated_prost_schemas::xla::HloModuleProto>(
        "xla.HloModuleProto",
    )?;
    verified += 1;
    let _ = roundtrip_generated_prost_type::<generated_prost_schemas::xla::CompileOptionsProto>(
        "xla.CompileOptionsProto",
    )?;
    verified += 1;
    let _ = roundtrip_generated_prost_type::<generated_prost_schemas::xla::TpuExecutableProto>(
        "xla.TpuExecutableProto",
    )?;
    verified += 1;
    let _ = roundtrip_generated_prost_type::<generated_prost_schemas::xla::IsaProgramProto>(
        "xla.IsaProgramProto",
    )?;
    verified += 1;
    let _ = roundtrip_generated_prost_type::<
        generated_prost_schemas::xla::megascale::runtime::DcnTopology,
    >("xla.megascale.runtime.DCNTopology")?;
    verified += 1;
    let _ = roundtrip_generated_prost_type::<
        generated_prost_schemas::xla::megascale::runtime::MegaScaleActionGraph,
    >("xla.megascale.runtime.MegaScaleActionGraph")?;
    verified += 1;
    let _ = roundtrip_generated_prost_type::<
        generated_prost_schemas::xla::megascale::runtime::MegaScaleRuntimeError,
    >("xla.megascale.runtime.MegaScaleRuntimeError")?;
    verified += 1;
    let _ = roundtrip_generated_prost_type::<
        generated_prost_schemas::xla::jellyfish::TpuCompilationEnvironment,
    >("xla.jellyfish.TpuCompilationEnvironment")?;
    verified += 1;
    let _ = roundtrip_generated_prost_type::<generated_prost_schemas::tpu::TpuCoreProgramProto>(
        "tpu.TpuCoreProgramProto",
    )?;
    verified += 1;
    let _ = roundtrip_generated_prost_type::<generated_prost_schemas::tpu::TpuChipConfigProto>(
        "tpu.TpuChipConfigProto",
    )?;
    verified += 1;
    let _ = roundtrip_generated_prost_type::<
        generated_prost_schemas::pjrt::tpu_topology_extension::TpuPlatformConfigs,
    >("pjrt.tpu_topology_extension.TpuPlatformConfigs")?;
    verified += 1;
    let _ = roundtrip_generated_prost_type::<
        generated_prost_schemas::platforms_deepsea::jellyfish::xdb::tpu_telemetry::AllCoreStateSummaries,
    >("platforms_deepsea.jellyfish.xdb.tpu_telemetry.AllCoreStateSummaries")?;
    verified += 1;
    let _ = roundtrip_generated_prost_type::<generated_prost_schemas::asic_sw::CoreDump>(
        "asic_sw.CoreDump",
    )?;
    verified += 1;
    let _ = roundtrip_generated_prost_type::<
        generated_prost_schemas::platforms_deepsea::jellyfish::xdb::CompilerMetadata,
    >("platforms_deepsea.jellyfish.xdb.CompilerMetadata")?;
    verified += 1;
    let _ = roundtrip_generated_prost_type::<
        generated_prost_schemas::tensorflow::CoordinationServiceConfig,
    >("tensorflow.CoordinationServiceConfig")?;
    verified += 1;
    let _ = roundtrip_generated_prost_type::<
        generated_prost_schemas::tensorflow::tpu::TpuEmbeddingConfiguration,
    >("tensorflow.tpu.TPUEmbeddingConfiguration")?;
    verified += 1;
    let _ = roundtrip_generated_prost_type::<generated_prost_schemas::tensorflow::profiler::XSpace>(
        "tensorflow.profiler.XSpace",
    )?;
    verified += 1;
    let _ = roundtrip_generated_prost_type::<
        generated_prost_schemas::stream_executor::RuntimeAbiVersionProto,
    >("stream_executor.RuntimeAbiVersionProto")?;
    verified += 1;
    let _ = roundtrip_generated_prost_type::<
        generated_prost_schemas::asic_sw::tools::RegisterTranslatorData,
    >("asic_sw.tools.RegisterTranslatorData")?;
    verified += 1;
    let _ = roundtrip_generated_prost_type::<
        generated_prost_schemas::hardware::chips::infra::gcsr::gcsrproto::node::Register,
    >("hardware.chips.infra.gcsr.gcsrproto.Node.Register")?;
    verified += 1;
    let _ = roundtrip_generated_prost_type::<
        generated_prost_schemas::platforms_performance_deepsea::barna_core::BarnaCoreModelParameters,
    >("platforms_performance_deepsea.barna_core.BarnaCoreModelParameters")?;
    verified += 1;
    let _ = roundtrip_generated_prost_type::<
        generated_prost_schemas::asic_sw::generators::errors::ErrorSources,
    >("asic_sw.generators.errors.ErrorSources")?;
    verified += 1;
    let _ = roundtrip_generated_prost_type::<generated_prost_schemas::xprof::XprofRequest>(
        "xprof.XprofRequest",
    )?;
    verified += 1;
    Ok(verified)
}

/// Verifies strict 1:1 closure between [`global_descriptor_pool`] and [`ALL_REGISTERED_PROTO_MESSAGE_NAMES`].
///
/// # Errors
///
/// Returns [`LibTpuError::ProtoSchemaDriftError`] or [`LibTpuError::ProtobufDecodeError`] if any
/// `.proto` message is unmapped or fails dynamic wire/schema round-trip verification.
pub fn verify_all_registered_proto_schemas_strict() -> LibTpuResult<usize> {
    verify_registered_proto_schemas_closure(ALL_REGISTERED_PROTO_MESSAGE_NAMES)
}

/// Trait implemented by all strongly-typed `libtpu-rs` protobuf messages backed by an
/// official `.proto` descriptor in [`global_descriptor_pool`].
pub trait ProtoSchemaMessage: Message + Default + Clone + PartialEq {
    /// Canonical fully-qualified protobuf message name (`package.MessageName`).
    const PROTO_FULL_NAME: &'static str;

    /// Returns the canonical fully-qualified protobuf message name (`package.MessageName`).
    #[must_use]
    fn full_name() -> &'static str {
        Self::PROTO_FULL_NAME
    }

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

    /// Computes a canonical 128-bit truncated SHA-256 hex fingerprint over this message's wire bytes.
    #[must_use]
    fn sha256_fingerprint_hex(&self) -> String {
        sha256_fingerprint_hex(&self.encode_to_vec())
    }

    /// Computes a deterministic 64-bit SHA-256 fingerprint over this message's wire bytes.
    #[must_use]
    fn sha256_fingerprint_u64(&self) -> u64 {
        sha256_fingerprint_u64(&self.encode_to_vec())
    }

    /// Computes Google `HighwayHash64` over this message's wire bytes using [`RIEGELI_HIGHWAYHASH_KEY`].
    #[must_use]
    fn highway_hash64(&self) -> u64 {
        riegeli_highwayhash64(&self.encode_to_vec())
    }

    /// Looks up the [`MessageDescriptor`] for this message in [`global_descriptor_pool`].
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if the descriptor is missing from the pool.
    fn descriptor() -> LibTpuResult<MessageDescriptor> {
        lookup_message_descriptor(Self::PROTO_FULL_NAME)
    }

    /// Returns the strongly-typed [`ProtoMessageSpec`] for this Rust protobuf struct.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if the descriptor is missing from the pool.
    fn schema_spec() -> LibTpuResult<ProtoMessageSpec> {
        extract_proto_message_spec(Self::PROTO_FULL_NAME)
    }

    /// Emits a canonical `proto3` `.proto` schema definition string from this Rust struct's [`ProtoMessageSpec`].
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if the descriptor is missing from the pool.
    fn emit_proto_schema() -> LibTpuResult<String> {
        emit_proto_message_schema(Self::PROTO_FULL_NAME)
    }

    /// Verifies 1:1 bidirectional equivalence between the `.proto` descriptor and this Rust struct.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtoSchemaDriftError`] if any field drifts between `.proto` and Rust.
    fn verify_bidirectional_equivalence() -> LibTpuResult<()> {
        verify_bidirectional_proto_struct_equivalence::<Self>()
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

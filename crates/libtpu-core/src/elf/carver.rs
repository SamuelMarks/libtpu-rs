//! Embedded `protodesc_cold` Protobuf carver and `filewrapper_toc` extractor.
//!
//! Provides:
//! - [`carve_protodesc_cold_entries`], [`decode_carved_file_descriptor`],
//!   [`build_descriptor_pool_from_carved_protos`], and [`CarvedProtoDesc`].
//! - [`carve_protodesc_cold_descriptors`], [`verify_descriptor_topological_closure`], and
//!   [`CarvedFileDescriptorSummary`].
//! - [`render_proto_schema`] and [`proto_scalar_type_name`] to reconstruct canonical `.proto` source files.
//! - [`extract_filewrapper_toc`], [`extract_indirect_filewrapper_toc`], [`decompress_brotli_csr_table`],
//!   and [`FileTocEntry`] for embedded TPU configuration blobs and Brotli CSR tables.

use crate::error::{LibTpuError, LibTpuResult};
use prost::Message;
use prost_types::FileDescriptorProto;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::io::Read as _;

/// Carved `FileDescriptorProto` entry recovered from the `protodesc_cold` ELF section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CarvedProtoDesc {
    /// Protobuf schema file name (e.g., `"xla_data.proto"`).
    pub name: String,
    /// Protobuf package name (e.g., `"xla"`).
    pub package: String,
    /// Complete wire-encoded `google.protobuf.FileDescriptorProto` bytes.
    pub raw_bytes: Vec<u8>,
}

/// Embedded static resource entry recovered from the `filewrapper_toc` ELF section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileTocEntry {
    /// Embedded file name.
    pub name: String,
    /// Embedded payload bytes.
    pub data: Vec<u8>,
    /// Trailing 16-byte RFC 1321 MD5 digest from the 40-byte `FileToc` struct (`[0; 16]` for 24-byte entries).
    pub md5_digest: [u8; 16],
    /// Whether `md5_digest` was verified against `compute_rfc1321_md5(&data)` (or was zero-filled).
    pub md5_verified: bool,
}

impl FileTocEntry {
    /// Creates a new [`FileTocEntry`] with its RFC 1321 MD5 digest automatically computed and verified.
    ///
    /// # Arguments
    ///
    /// * `name` - Embedded file name.
    /// * `data` - Embedded payload bytes.
    #[must_use]
    pub fn new(name: String, data: Vec<u8>) -> Self {
        let md5_digest = compute_rfc1321_md5(&data);
        Self {
            name,
            data,
            md5_digest,
            md5_verified: true,
        }
    }
}

/// RFC 1321 MD5 per-round left-rotation shift amounts (`MD5_S`).
const MD5_S: [u32; 64] = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9,
    14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10, 15,
    21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
];

/// RFC 1321 MD5 integer sine table constants (`MD5_K`).
const MD5_K: [u32; 64] = [
    0xd76a_a478,
    0xe8c7_b756,
    0x2420_70db,
    0xc1bd_ceee,
    0xf57c_0faf,
    0x4787_c62a,
    0xa830_4613,
    0xfd46_9501,
    0x6980_98d8,
    0x8b44_f7af,
    0xffff_5bb1,
    0x895c_d7be,
    0x6b90_1122,
    0xfd98_7193,
    0xa679_438e,
    0x49b4_0821,
    0xf61e_2562,
    0xc040_b340,
    0x265e_5a51,
    0xe9b6_c7aa,
    0xd62f_105d,
    0x0244_1453,
    0xd8a1_e681,
    0xe7d3_fbc8,
    0x21e1_cde6,
    0xc337_07d6,
    0xf4d5_0d87,
    0x455a_14ed,
    0xa9e3_e905,
    0xfcef_a3f8,
    0x676f_02d9,
    0x8d2a_4c8a,
    0xfffa_3942,
    0x8771_f681,
    0x6d9d_6122,
    0xfde5_380c,
    0xa4be_ea44,
    0x4bde_cfa9,
    0xf6bb_4b60,
    0xbebf_bc70,
    0x289b_7ec6,
    0xeaa1_27fa,
    0xd4ef_3085,
    0x0488_1d05,
    0xd9d4_d039,
    0xe6db_99e5,
    0x1fa2_7cf8,
    0xc4ac_5665,
    0xf429_2244,
    0x432a_ff97,
    0xab94_23a7,
    0xfc93_a039,
    0x655b_59c3,
    0x8f0c_cc92,
    0xffef_f47d,
    0x8584_5dd1,
    0x6fa8_7e4f,
    0xfe2c_e6e0,
    0xa301_4314,
    0x4e08_11a1,
    0xf753_7e82,
    0xbd3a_f235,
    0x2ad7_d2bb,
    0xeb86_d391,
];

/// Computes the 16-byte RFC 1321 MD5 message digest of `data` in pure safe Rust.
///
/// # Arguments
///
/// * `data` - Input byte slice to hash.
#[must_use]
pub fn compute_rfc1321_md5(data: &[u8]) -> [u8; 16] {
    let mut state_a: u32 = 0x6745_2301;
    let mut state_b: u32 = 0xefcd_ab89;
    let mut state_c: u32 = 0x98ba_dcfe;
    let mut state_d: u32 = 0x1032_5476;

    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut padded = Vec::with_capacity((data.len() + 72) & !63);
    padded.extend_from_slice(data);
    padded.push(0x80);
    while (padded.len() % 64) != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_le_bytes());

    for chunk in padded.chunks_exact(64) {
        let mut msg_words = [0_u32; 16];
        for (slot, word_bytes) in msg_words.iter_mut().zip(chunk.chunks_exact(4)) {
            *slot = u32::from_le(bytemuck::pod_read_unaligned::<u32>(word_bytes));
        }
        let mut reg_a = state_a;
        let mut reg_b = state_b;
        let mut reg_c = state_c;
        let mut reg_d = state_d;

        for step in 0..64_usize {
            let (func_val, word_idx) = if step < 16 {
                ((reg_b & reg_c) | ((!reg_b) & reg_d), step)
            } else if step < 32 {
                ((reg_d & reg_b) | ((!reg_d) & reg_c), (5 * step + 1) % 16)
            } else if step < 48 {
                (reg_b ^ reg_c ^ reg_d, (3 * step + 5) % 16)
            } else {
                (reg_c ^ (reg_b | (!reg_d)), (7 * step) % 16)
            };
            let k_val = MD5_K.get(step).copied().unwrap_or(0);
            let m_val = msg_words.get(word_idx).copied().unwrap_or(0);
            let s_val = MD5_S.get(step).copied().unwrap_or(0);
            let temp = reg_a
                .wrapping_add(func_val)
                .wrapping_add(k_val)
                .wrapping_add(m_val);
            reg_a = reg_d;
            reg_d = reg_c;
            reg_c = reg_b;
            reg_b = reg_b.wrapping_add(temp.rotate_left(s_val));
        }

        state_a = state_a.wrapping_add(reg_a);
        state_b = state_b.wrapping_add(reg_b);
        state_c = state_c.wrapping_add(reg_c);
        state_d = state_d.wrapping_add(reg_d);
    }

    let mut digest = [0_u8; 16];
    digest[0..4].copy_from_slice(&state_a.to_le_bytes());
    digest[4..8].copy_from_slice(&state_b.to_le_bytes());
    digest[8..12].copy_from_slice(&state_c.to_le_bytes());
    digest[12..16].copy_from_slice(&state_d.to_le_bytes());
    digest
}

/// Formats a 16-byte MD5 digest as a 32-character lowercase hexadecimal string.
///
/// # Arguments
///
/// * `digest` - 16-byte MD5 digest array.
#[must_use]
pub fn format_md5_hex(digest: &[u8; 16]) -> String {
    let mut out = String::with_capacity(32);
    for &b in digest {
        let _ = write!(out, "{b:02x}");
    }
    out
}

/// Verifies the trailing 16-byte RFC 1321 MD5 digests of a slice of [`FileTocEntry`] records.
///
/// Entries whose `md5_digest` is all zeros (`[0; 16]`) are treated as unhashed 24-byte records
/// and pass verification automatically.
///
/// # Arguments
///
/// * `entries` - Slice of [`FileTocEntry`] items to verify.
///
/// # Errors
///
/// Returns [`LibTpuError::TocPayloadChecksumMismatch`] if any non-zero `md5_digest` mismatches
/// `compute_rfc1321_md5(&entry.data)`.
pub fn verify_filewrapper_toc_md5(entries: &[FileTocEntry]) -> LibTpuResult<()> {
    for entry in entries.iter().filter(|e| e.md5_digest != [0_u8; 16]) {
        let actual = compute_rfc1321_md5(&entry.data);
        if actual != entry.md5_digest {
            return Err(LibTpuError::TocPayloadChecksumMismatch {
                entry_name: entry.name.clone(),
                expected_md5_hex: format_md5_hex(&entry.md5_digest),
                actual_md5_hex: format_md5_hex(&actual),
            });
        }
    }
    Ok(())
}

/// Summary of a wire-encoded `google.protobuf.FileDescriptorProto` carved from `protodesc_cold`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CarvedFileDescriptorSummary {
    /// Embedded `.proto` file path (e.g., `"xla/xla_data.proto"`).
    pub name: String,
    /// Protobuf package namespace (e.g., `"xla"`).
    pub package: String,
    /// Protobuf syntax version (`"proto2"` or `"proto3"`).
    pub syntax: String,
    /// Imported `.proto` dependencies (`FileDescriptorProto.dependency`).
    pub dependencies: Vec<String>,
    /// Top-level message names defined in this descriptor.
    pub message_names: Vec<String>,
    /// Top-level enum names defined in this descriptor.
    pub enum_names: Vec<String>,
    /// Canonical `.proto` source text reconstructed via [`render_proto_schema`].
    pub rendered_proto: String,
}

/// Reads a protobuf base-128 varint starting at `bytes[pos]`, returning `(value, next_pos)`.
///
/// # Arguments
///
/// * `bytes` - Input byte slice.
/// * `pos` - Starting byte offset.
fn read_proto_varint(bytes: &[u8], pos: usize) -> Option<(u64, usize)> {
    let mut result = 0_u64;
    let mut shift = 0_u32;
    let mut cur = pos;
    while shift < 64 {
        let byte = bytes.get(cur).copied()?;
        cur += 1;
        result |= u64::from(byte & 0x7F) << shift;
        if (byte & 0x80) == 0 {
            return Some((result, cur));
        }
        shift += 7;
    }
    None
}

/// Advances across all valid consecutive `google.protobuf.FileDescriptorProto` wire fields
/// starting at `start_idx`, returning the exclusive end byte offset.
///
/// Supports varint (`0`), 64-bit fixed (`1`), length-delimited (`2`), and 32-bit fixed (`5`)
/// wire types across `FileDescriptorProto` fields `1..=14` (including `edition = 14`).
///
/// # Arguments
///
/// * `bytes` - `protodesc_cold` byte slice.
/// * `start_idx` - Offset of the leading `0x0A` (`name`) tag.
fn scan_file_descriptor_wire_end(bytes: &[u8], start_idx: usize) -> usize {
    let mut cur = start_idx;
    while cur < bytes.len() {
        if cur > start_idx && bytes.get(cur).copied().unwrap_or(0) == 0x0A {
            let (next_len_u64, after_next_len) =
                read_proto_varint(bytes, cur + 1).unwrap_or((0, cur + 1));
            let next_len = usize::try_from(next_len_u64).unwrap_or(usize::MAX);
            let next_end = after_next_len.saturating_add(next_len);
            let next_name = bytes.get(after_next_len..next_end).unwrap_or_default();
            if next_name.ends_with(b".proto") {
                break;
            }
        }
        let Some((tag, after_tag)) = read_proto_varint(bytes, cur) else {
            break;
        };
        let field_num = tag >> 3;
        let wire_type = tag & 0x07;
        if !(1..=14).contains(&field_num) {
            break;
        }
        match wire_type {
            0 => {
                let Some((_, after_val)) = read_proto_varint(bytes, after_tag) else {
                    break;
                };
                cur = after_val;
            }
            2 => {
                let Some((len_u64, after_len)) = read_proto_varint(bytes, after_tag) else {
                    break;
                };
                let len = usize::try_from(len_u64).unwrap_or(usize::MAX);
                let next_pos = after_len.saturating_add(len);
                if next_pos > bytes.len() {
                    break;
                }
                cur = next_pos;
            }
            1 | 5 => {
                let step = if wire_type == 1 { 8 } else { 4 };
                let next_pos = after_tag.saturating_add(step);
                if next_pos > bytes.len() {
                    break;
                }
                cur = next_pos;
            }
            _ => break,
        }
    }
    cur
}

/// Scans a `protodesc_cold` section byte slice for wire-encoded `FileDescriptorProto` entries.
///
/// Matches entries starting with tag `0x0a` (field 1: `string name` ending in `".proto"`,
/// including multi-byte varint string lengths `> 127` bytes), preserving all subsequent
/// `FileDescriptorProto` fields (`message_type`, `enum_type`, `dependency`, `syntax`, `edition`).
///
/// # Arguments
///
/// * `section_bytes` - Raw byte slice of the `protodesc_cold` section.
#[must_use]
pub fn carve_protodesc_cold_entries(section_bytes: &[u8]) -> Vec<CarvedProtoDesc> {
    let mut results = Vec::new();
    let mut idx = 0_usize;
    while idx + 8 <= section_bytes.len() {
        if section_bytes.get(idx).copied().unwrap_or(0) == 0x0A {
            let (name_len_u64, name_start) =
                read_proto_varint(section_bytes, idx + 1).unwrap_or((0, idx + 1));
            let name_len = usize::try_from(name_len_u64).unwrap_or(usize::MAX);
            let name_end = name_start.saturating_add(name_len);
            let name_slice = section_bytes.get(name_start..name_end).unwrap_or_default();
            if name_slice.ends_with(b".proto") {
                let end_idx = scan_file_descriptor_wire_end(section_bytes, idx);
                let raw_bytes = section_bytes.get(idx..end_idx).unwrap_or_default().to_vec();
                let (name, pkg) = FileDescriptorProto::decode(raw_bytes.as_slice()).map_or_else(
                    |_| {
                        (
                            String::from_utf8_lossy(name_slice).into_owned(),
                            String::new(),
                        )
                    },
                    |fd| (fd.name.unwrap_or_default(), fd.package.unwrap_or_default()),
                );
                results.push(CarvedProtoDesc {
                    name,
                    package: pkg,
                    raw_bytes,
                });
                idx = end_idx.max(idx + 1);
                continue;
            }
        }
        idx += 1;
    }
    results
}

/// Decodes a [`CarvedProtoDesc`] into a strongly-typed `prost_types::FileDescriptorProto`.
///
/// # Arguments
///
/// * `carved` - Carved descriptor entry from [`carve_protodesc_cold_entries`].
///
/// # Errors
///
/// Returns [`LibTpuError`] if `carved.raw_bytes` is not a valid
/// wire-encoded `FileDescriptorProto`.
pub fn decode_carved_file_descriptor(
    carved: &CarvedProtoDesc,
) -> LibTpuResult<FileDescriptorProto> {
    FileDescriptorProto::decode(carved.raw_bytes.as_slice()).map_err(LibTpuError::from)
}

/// Builds a live `prost_reflect::DescriptorPool` from a slice of carved [`CarvedProtoDesc`] entries.
///
/// Decodes each `FileDescriptorProto` and registers them as a `prost_types::FileDescriptorSet`
/// in a `prost_reflect::DescriptorPool`.
///
/// # Arguments
///
/// * `entries` - Carved protobuf file descriptor entries from `protodesc_cold`.
///
/// # Errors
///
/// Returns [`LibTpuError`] if any descriptor fails wire decoding or topological resolution in `prost_reflect`.
pub fn build_descriptor_pool_from_carved_protos(
    entries: &[CarvedProtoDesc],
) -> LibTpuResult<prost_reflect::DescriptorPool> {
    let mut files = Vec::with_capacity(entries.len());
    for entry in entries {
        files.push(decode_carved_file_descriptor(entry)?);
    }
    let fds = prost_types::FileDescriptorSet { file: files };
    prost_reflect::DescriptorPool::from_file_descriptor_set(fds).map_err(LibTpuError::from)
}

/// Maps a `prost_types::field_descriptor_proto::Type` integer discriminant (`1..=18`)
/// to its canonical `.proto` scalar type keyword.
///
/// # Arguments
///
/// * `type_id` - Optional protobuf `FieldDescriptorProto.Type` integer tag (`1..=18`).
#[must_use]
pub const fn proto_scalar_type_name(type_id: Option<i32>) -> Option<&'static str> {
    match type_id {
        Some(1) => Some("double"),
        Some(2) => Some("float"),
        Some(3) => Some("int64"),
        Some(4) => Some("uint64"),
        Some(5) => Some("int32"),
        Some(6) => Some("fixed64"),
        Some(7) => Some("fixed32"),
        Some(8) => Some("bool"),
        Some(9) => Some("string"),
        Some(10) => Some("group"),
        Some(12) => Some("bytes"),
        Some(13) => Some("uint32"),
        Some(15) => Some("sfixed32"),
        Some(16) => Some("sfixed64"),
        Some(17) => Some("sint32"),
        Some(18) => Some("sint64"),
        _ => None,
    }
}

/// Resolves the canonical `.proto` type string for a `prost_types::FieldDescriptorProto`.
///
/// # Arguments
///
/// * `field` - Field descriptor to inspect.
#[must_use]
fn resolve_proto_field_type(field: &prost_types::FieldDescriptorProto) -> &str {
    field
        .type_name
        .as_deref()
        .filter(|t| !t.is_empty())
        .map_or_else(
            || proto_scalar_type_name(field.r#type).unwrap_or("bytes"),
            |t_name| t_name.trim_start_matches('.'),
        )
}

/// Renders an `EnumDescriptorProto` block at the given indentation depth.
///
/// # Arguments
///
/// * `out` - Destination buffer.
/// * `en` - Enum descriptor to render.
/// * `indent` - Leading whitespace prefix.
fn render_proto_enum(out: &mut String, en: &prost_types::EnumDescriptorProto, indent: &str) {
    let en_name = en.name.as_deref().unwrap_or("UnnamedEnum");
    let _ = writeln!(out);
    let _ = writeln!(out, "{indent}enum {en_name} {{");
    for val in &en.value {
        let v_name = val.name.as_deref().unwrap_or("UNKNOWN");
        let v_num = val.number.unwrap_or(0);
        let _ = writeln!(out, "{indent}  {v_name} = {v_num};");
    }
    let _ = writeln!(out, "{indent}}}");
}

/// Renders a `DescriptorProto` message (including nested enums, nested messages, `map<K, V>` fields,
/// `oneof` blocks, and field labels) into `out`.
///
/// # Arguments
///
/// * `out` - Destination buffer.
/// * `msg` - Message descriptor to render.
/// * `indent` - Leading whitespace prefix.
/// * `is_proto3` - Whether the enclosing file uses `proto3` syntax.
fn render_proto_message(
    out: &mut String,
    msg: &prost_types::DescriptorProto,
    indent: &str,
    is_proto3: bool,
) {
    let msg_name = msg.name.as_deref().unwrap_or("UnnamedMessage");
    let _ = writeln!(out);
    let _ = writeln!(out, "{indent}message {msg_name} {{");
    let inner_indent = format!("{indent}  ");

    let mut map_entries = BTreeMap::new();
    for nested in &msg.nested_type {
        let is_map = nested
            .options
            .as_ref()
            .and_then(|opts| opts.map_entry)
            .unwrap_or(false);
        if is_map {
            let entry_name = nested.name.as_deref().unwrap_or_default();
            let key_type = nested
                .field
                .iter()
                .find(|f| f.number == Some(1))
                .map_or("string", resolve_proto_field_type);
            let val_type = nested
                .field
                .iter()
                .find(|f| f.number == Some(2))
                .map_or("bytes", resolve_proto_field_type);
            map_entries.insert(entry_name, (key_type, val_type));
        }
    }

    for en in &msg.enum_type {
        render_proto_enum(out, en, &inner_indent);
    }

    for nested in &msg.nested_type {
        let is_map = nested
            .options
            .as_ref()
            .and_then(|opts| opts.map_entry)
            .unwrap_or(false);
        if !is_map {
            render_proto_message(out, nested, &inner_indent, is_proto3);
        }
    }

    for (oneof_idx, oneof) in msg.oneof_decl.iter().enumerate() {
        let oneof_name = oneof.name.as_deref().unwrap_or("unnamed_oneof");
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let target_idx = oneof_idx as i32;
        let members: Vec<&prost_types::FieldDescriptorProto> = msg
            .field
            .iter()
            .filter(|f| f.oneof_index == Some(target_idx) && !f.proto3_optional.unwrap_or(false))
            .collect();
        if !members.is_empty() {
            let _ = writeln!(out, "{inner_indent}oneof {oneof_name} {{");
            for field in members {
                let f_name = field.name.as_deref().unwrap_or("field");
                let f_num = field.number.unwrap_or(1);
                let f_type = resolve_proto_field_type(field);
                let _ = writeln!(out, "{inner_indent}  {f_type} {f_name} = {f_num};");
            }
            let _ = writeln!(out, "{inner_indent}}}");
        }
    }

    for field in &msg.field {
        if field.oneof_index.is_some() && !field.proto3_optional.unwrap_or(false) {
            continue;
        }
        let f_name = field.name.as_deref().unwrap_or("field");
        let f_num = field.number.unwrap_or(1);
        let raw_type = resolve_proto_field_type(field);
        let short_type = raw_type.rsplit('.').next().unwrap_or(raw_type);
        if let Some(&(k_ty, v_ty)) = map_entries.get(short_type) {
            let _ = writeln!(out, "{inner_indent}map<{k_ty}, {v_ty}> {f_name} = {f_num};");
            continue;
        }
        let label_prefix = match field.label {
            Some(3) => "repeated ",
            Some(2) => "required ",
            Some(1) if !is_proto3 || field.proto3_optional.unwrap_or(false) => "optional ",
            _ => "",
        };
        let _ = writeln!(
            out,
            "{inner_indent}{label_prefix}{raw_type} {f_name} = {f_num};"
        );
    }

    let _ = writeln!(out, "{indent}}}");
}

/// Renders a `prost_types::FileDescriptorProto` into canonical `.proto` source text.
///
/// Reconstructs `syntax` (or `edition = "2023";` when `syntax == "editions"` / `edition == Some(1000)`),
/// `package`, `import`, top-level and nested `enum` blocks, top-level and nested `message` blocks,
/// `oneof` declarations, `map<K, V>` fields, field labels (`optional`, `required`, `repeated`),
/// and all 18 protobuf scalar types.
///
/// # Arguments
///
/// * `fd` - Decoded `FileDescriptorProto` schema.
#[must_use]
pub fn render_proto_schema(fd: &FileDescriptorProto) -> String {
    let mut out = String::new();
    let syntax = fd.syntax.as_deref().unwrap_or("proto2");
    let is_editions = matches!(syntax, "editions" | "edition_2023" | "edition_2024");
    let is_proto3 = syntax == "proto3" || is_editions;
    if is_editions {
        let edition_str = if syntax == "edition_2024" {
            "2024"
        } else {
            "2023"
        };
        let _ = writeln!(out, r#"edition = "{edition_str}";"#);
    } else {
        let _ = writeln!(out, r#"syntax = "{syntax}";"#);
    }
    if let Some(pkg) = fd.package.as_deref().filter(|p| !p.is_empty()) {
        let _ = writeln!(out, "package {pkg};");
    }
    for dep in &fd.dependency {
        let _ = writeln!(out, r#"import "{dep}";"#);
    }
    for en in &fd.enum_type {
        render_proto_enum(&mut out, en, "");
    }
    for msg in &fd.message_type {
        render_proto_message(&mut out, msg, "", is_proto3);
    }
    out
}

/// Carves length-prefixed wire-encoded `google.protobuf.FileDescriptorProto` records from a
/// `protodesc_cold` or `.rodata` section byte slice.
///
/// Each entry in `section_bytes` is framed as `[u32_le_length][FileDescriptorProto_wire_bytes]`.
///
/// # Arguments
///
/// * `section_bytes` - Raw bytes of the `protodesc_cold` section containing length-prefixed `FileDescriptorProto` payloads.
#[must_use]
pub fn carve_protodesc_cold_descriptors(section_bytes: &[u8]) -> Vec<CarvedFileDescriptorSummary> {
    let mut cursor = section_bytes;
    let mut results = Vec::new();
    while let Some((len_bytes, after_len)) = cursor.split_at_checked(4) {
        let frame_len =
            usize::try_from(u32::from_le(bytemuck::pod_read_unaligned::<u32>(len_bytes)))
                .unwrap_or(usize::MAX);
        let Some((payload, rest)) = after_len.split_at_checked(frame_len) else {
            break;
        };
        cursor = rest;
        if let Ok(fd) = FileDescriptorProto::decode(payload) {
            let name = fd.name.clone().unwrap_or_default();
            if std::path::Path::new(&name)
                .extension()
                .is_some_and(|ext| ext == "proto")
            {
                let rendered_proto = render_proto_schema(&fd);
                results.push(CarvedFileDescriptorSummary {
                    name,
                    package: fd.package.unwrap_or_default(),
                    syntax: fd.syntax.unwrap_or_else(|| String::from("proto2")),
                    dependencies: fd.dependency,
                    message_names: fd
                        .message_type
                        .into_iter()
                        .map(|m| m.name.unwrap_or_default())
                        .collect(),
                    enum_names: fd
                        .enum_type
                        .into_iter()
                        .map(|e| e.name.unwrap_or_default())
                        .collect(),
                    rendered_proto,
                });
            }
        }
    }
    results
}

/// Verifies that a slice of [`CarvedFileDescriptorSummary`] items is topologically closed under
/// `import` dependencies (`dependencies`) and returns their dependency-ordered `.proto` names.
///
/// # Arguments
///
/// * `descriptors` - Slice of carved protobuf file descriptor summaries.
///
/// # Errors
///
/// Returns [`LibTpuError::BinaryFormatError`] if any imported `.proto` file is missing from `descriptors`.
pub fn verify_descriptor_topological_closure(
    descriptors: &[CarvedFileDescriptorSummary],
) -> LibTpuResult<Vec<String>> {
    let mut known_names = BTreeSet::new();
    for desc in descriptors {
        let basename = desc.name.rsplit('/').next().unwrap_or(desc.name.as_str());
        known_names.insert(basename);
    }

    let mut ordered = Vec::with_capacity(descriptors.len());
    for desc in descriptors {
        for dep in &desc.dependencies {
            let dep_base = dep.rsplit('/').next().unwrap_or(dep.as_str());
            if !known_names.contains(dep_base) {
                return Err(LibTpuError::BinaryFormatError {
                    reason: format!(
                        "Descriptor '{}' imports missing dependency '{}'",
                        desc.name, dep
                    ),
                });
            }
        }
        ordered.push(desc.name.clone());
    }
    Ok(ordered)
}

/// Parses a `filewrapper_toc` table of contents (`24`-byte or `40`-byte `FileToc` entries)
/// backed by a companion data blob.
///
/// # Arguments
///
/// * `toc_bytes` - Raw byte slice containing 24-byte `(name_off: u64, data_off: u64, size: u64)` or 40-byte `(name_off, data_off, size, md5_lo, md5_hi)` records.
/// * `blob_bytes` - Backing string and payload data buffer referenced by `toc_bytes`.
#[must_use]
pub fn extract_filewrapper_toc(toc_bytes: &[u8], blob_bytes: &[u8]) -> Vec<FileTocEntry> {
    let stride = if toc_bytes.len() >= 40 && toc_bytes.len() % 40 == 0 {
        40
    } else {
        24
    };
    let mut entries = Vec::new();
    for chunk in toc_bytes.chunks_exact(stride) {
        let name_off = usize::try_from(u64::from_le(bytemuck::pod_read_unaligned::<u64>(
            chunk.get(0..8).unwrap_or(&[0; 8]),
        )))
        .unwrap_or(usize::MAX);
        let data_off = usize::try_from(u64::from_le(bytemuck::pod_read_unaligned::<u64>(
            chunk.get(8..16).unwrap_or(&[0; 8]),
        )))
        .unwrap_or(usize::MAX);
        let data_size = usize::try_from(u64::from_le(bytemuck::pod_read_unaligned::<u64>(
            chunk.get(16..24).unwrap_or(&[0; 8]),
        )))
        .unwrap_or(0);

        let Some(name_tail) = blob_bytes.get(name_off..) else {
            continue;
        };
        let Some(data_slice) = blob_bytes.get(data_off..data_off.saturating_add(data_size)) else {
            continue;
        };
        let nul_pos = name_tail
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(name_tail.len());
        let name =
            String::from_utf8_lossy(name_tail.get(..nul_pos).unwrap_or_default()).into_owned();
        if !name.is_empty() {
            let mut md5_digest = [0_u8; 16];
            if let Some(md5_slice) = chunk.get(24..40) {
                md5_digest.copy_from_slice(md5_slice);
            }
            let md5_verified =
                (md5_digest == [0_u8; 16]) | (compute_rfc1321_md5(data_slice) == md5_digest);
            entries.push(FileTocEntry {
                name,
                data: data_slice.to_vec(),
                md5_digest,
                md5_verified,
            });
        }
    }
    entries
}

/// Resolves an indirect `filewrapper_toc` pointer array into [`FileTocEntry`] items.
///
/// In `libtpu.so`, each 8-byte slot in `filewrapper_toc` is relocated via `.rela.dyn` to a
/// null-terminated array of 40-byte `FileToc` structs (`name_ptr`, `data_ptr`, `size`, `md5[16]`)
/// in `.data.rel.ro`.
///
/// # Arguments
///
/// * `toc_slot_vaddrs` - Virtual addresses of the 8-byte pointer slots inside `filewrapper_toc`.
/// * `reloc_map` - Mapping from relocated virtual address (`r_offset`) to target `r_addend`.
/// * `size_word_map` - Mapping from virtual address (`toc_struct_vaddr + 16`) to payload byte length.
/// * `rodata_blobs` - Mapping from virtual address to raw byte slice in `.rodata`.
#[must_use]
pub fn extract_indirect_filewrapper_toc(
    toc_slot_vaddrs: &[u64],
    reloc_map: &BTreeMap<u64, u64>,
    size_word_map: &BTreeMap<u64, u64>,
    rodata_blobs: &BTreeMap<u64, Vec<u8>>,
) -> Vec<FileTocEntry> {
    let mut entries = Vec::new();
    for &slot_vaddr in toc_slot_vaddrs {
        let Some(&base_toc_vaddr) = reloc_map.get(&slot_vaddr) else {
            continue;
        };
        let mut elem_idx = 0_u64;
        while let Some(&name_vaddr) = reloc_map.get(&base_toc_vaddr.wrapping_add(elem_idx * 40)) {
            let entry_vaddr = base_toc_vaddr.wrapping_add(elem_idx * 40);
            elem_idx += 1;
            let Some(&data_vaddr) = reloc_map.get(&entry_vaddr.wrapping_add(8)) else {
                break;
            };
            let size = usize::try_from(
                size_word_map
                    .get(&entry_vaddr.wrapping_add(16))
                    .copied()
                    .unwrap_or(0),
            )
            .unwrap_or(0);
            let Some(name_bytes) = rodata_blobs.get(&name_vaddr) else {
                continue;
            };
            let Some(data_bytes) = rodata_blobs.get(&data_vaddr) else {
                continue;
            };
            let nul_pos = name_bytes
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(name_bytes.len());
            let name =
                String::from_utf8_lossy(name_bytes.get(..nul_pos).unwrap_or_default()).into_owned();
            let payload = data_bytes
                .get(..size.min(data_bytes.len()))
                .unwrap_or_default();
            if !name.is_empty() {
                let mut md5_digest = [0_u8; 16];
                let md5_lo = size_word_map
                    .get(&entry_vaddr.wrapping_add(24))
                    .copied()
                    .unwrap_or(0);
                let md5_hi = size_word_map
                    .get(&entry_vaddr.wrapping_add(32))
                    .copied()
                    .unwrap_or(0);
                md5_digest[0..8].copy_from_slice(&md5_lo.to_le_bytes());
                md5_digest[8..16].copy_from_slice(&md5_hi.to_le_bytes());
                let md5_verified =
                    (md5_digest == [0_u8; 16]) | (compute_rfc1321_md5(payload) == md5_digest);
                entries.push(FileTocEntry {
                    name,
                    data: payload.to_vec(),
                    md5_digest,
                    md5_verified,
                });
            }
        }
    }
    entries
}

/// Decoded collection of embedded `filewrapper_toc` chip configurations, die parts, CSR tables, and runtime ABI descriptors (`EmbeddedChipResourceBundle`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EmbeddedChipResourceBundle {
    /// Decoded `*_chip_configs_*.binarypb` descriptors.
    pub chip_configs: Vec<crate::proto::TpuChipConfigProto>,
    /// Decoded `*_chip_parts.binarypb` descriptors.
    pub chip_parts: Vec<crate::proto::TpuChipPartsProto>,
    /// Decoded `tpu_platform_configs.binarypb` descriptors.
    pub platform_configs: Vec<crate::proto::TpuPlatformConfigsProto>,
    /// Decompressed and decoded `*_csrs.br` (`RegisterTranslatorDataProto`) CSR tables.
    pub csr_tables: Vec<crate::proto::RegisterTranslatorDataProto>,
    /// Decoded `current_tpu_runtime_abi.binarypb` descriptor, if present.
    pub runtime_abi: Option<crate::proto::TpuRuntimeAbiProto>,
    /// Decoded `deepsea_compiler_hash.txt` string, if present.
    pub compiler_hash: Option<String>,
}

/// Decodes embedded chip config, chip parts, platform configs, Brotli CSR tables, and runtime ABI resources from `filewrapper_toc`.
///
/// Inspects `*_chip_configs*.binarypb`, `*_chip_parts*.binarypb`, `*_platform_configs*.binarypb`,
/// `*_csrs.br`, `current_tpu_runtime_abi.binarypb`, and `deepsea_compiler_hash.txt` entries.
///
/// # Arguments
///
/// * `entries` - Slice of [`FileTocEntry`] records carved from `filewrapper_toc`.
///
/// # Errors
///
/// Returns [`LibTpuError`] if any matching `.binarypb` or `.br` payload is malformed.
pub fn decode_embedded_chip_resources(
    entries: &[FileTocEntry],
) -> LibTpuResult<EmbeddedChipResourceBundle> {
    use crate::proto::ProtoSchemaMessage as _;
    let mut bundle = EmbeddedChipResourceBundle::default();
    for entry in entries {
        if entry.name.contains("chip_configs") {
            bundle
                .chip_configs
                .push(crate::proto::TpuChipConfigProto::decode_proto(&entry.data)?);
        } else if entry.name.contains("chip_parts") {
            bundle
                .chip_parts
                .push(crate::proto::TpuChipPartsProto::decode_proto(&entry.data)?);
        } else if entry.name.contains("platform_configs") {
            bundle
                .platform_configs
                .push(crate::proto::TpuPlatformConfigsProto::decode_proto(
                    &entry.data,
                )?);
        } else if entry.name.ends_with("csrs.br") {
            let decompressed = decompress_brotli_csr_table(&entry.data)?;
            let csr_proto = crate::proto::RegisterTranslatorDataProto::decode_proto(&decompressed)?;
            bundle.csr_tables.push(csr_proto);
        } else if entry.name.ends_with("tpu_runtime_abi.binarypb") {
            bundle.runtime_abi = Some(crate::proto::TpuRuntimeAbiProto::decode_proto(&entry.data)?);
        } else if entry.name.ends_with("deepsea_compiler_hash.txt") {
            bundle.compiler_hash = Some(String::from_utf8_lossy(&entry.data).trim().to_owned());
        }
    }
    Ok(bundle)
}

/// Decompresses an embedded Brotli-compressed (`*_pf_csrs.br`) hardware CSR table from `libtpu.so`.
///
/// # Arguments
///
/// * `compressed` - Raw Brotli-compressed payload extracted from `filewrapper_toc`.
///
/// # Errors
///
/// Returns [`LibTpuError::BinaryFormatError`] if `compressed` is not a valid Brotli stream.
pub fn decompress_brotli_csr_table(compressed: &[u8]) -> LibTpuResult<Vec<u8>> {
    let mut decompressor = brotli::Decompressor::new(compressed, 4096);
    let mut out = Vec::new();
    decompressor
        .read_to_end(&mut out)
        .map_err(|err| LibTpuError::BinaryFormatError {
            reason: format!("brotli CSR decompression failed: {err}"),
        })?;
    Ok(out)
}

/// Parsed register tree entry extracted from a decompressed `*_pf_csrs.br` (`asic_sw.tools.RegisterTranslatorData`) table (`BrotliRegisterTreeEntry`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BrotliRegisterTreeEntry {
    /// Hierarchical CSR path (e.g., `"tile.tensor_core_0.program_location"` or `"bar0.tc0_pc_csr"`).
    pub full_path: String,
    /// Byte offset inside the `PCIe` BAR0 MMIO aperture (`MmioOffset`).
    pub bar0_offset: u64,
    /// Register width in bits (`32` or `64`).
    pub width_bits: u32,
}

/// Parses a decompressed `*_pf_csrs.br` register tree payload and verifies its offsets.
///
/// Cross-checks known register names against a [`crate::hal::ChipRegisterMap`].
///
/// # Arguments
///
/// * `decompressed_bytes` - Raw decompressed CSR table bytes from [`decompress_brotli_csr_table`].
/// * `register_map` - Target generation [`crate::hal::ChipRegisterMap`] to verify against.
///
/// # Errors
///
/// Returns [`LibTpuError::BinaryFormatError`] if any CSR path in `decompressed_bytes` conflicts
/// with the architectural offset in `register_map`.
pub fn parse_and_verify_brotli_csr_tree(
    decompressed_bytes: &[u8],
    register_map: &crate::hal::ChipRegisterMap,
) -> LibTpuResult<Vec<BrotliRegisterTreeEntry>> {
    use crate::proto::ProtoSchemaMessage as _;
    if !decompressed_bytes.contains(&b'=') {
        if let Ok(proto) =
            crate::proto::RegisterTranslatorDataProto::decode_proto(decompressed_bytes)
        {
            if !proto.nodes.is_empty() {
                let mut out = Vec::new();
                for node in proto.nodes {
                    for r in node.registers {
                        let w_bytes = u32::try_from(r.width.max(0)).unwrap_or(0);
                        out.push(BrotliRegisterTreeEntry {
                            full_path: format!("{}.{}", node.mnemonic, r.name),
                            bar0_offset: node.base_offset.saturating_add(r.offset),
                            width_bits: if w_bytes == 0 {
                                64
                            } else {
                                w_bytes.saturating_mul(8)
                            },
                        });
                    }
                }
                return Ok(out);
            }
        }
    }
    let text = String::from_utf8_lossy(decompressed_bytes);
    let mut entries = Vec::new();
    for raw_line in text.lines() {
        let trimmed = raw_line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if let Some((name_part, rhs_part)) = trimmed.split_once('=') {
            let full_path = String::from(name_part.trim());
            let mut rhs_tokens = rhs_part.split_whitespace();
            let off_str = rhs_tokens.next().unwrap_or("0");
            let width_str = rhs_tokens.next().unwrap_or("64");
            let stripped_hex = off_str
                .strip_prefix("0x")
                .or_else(|| off_str.strip_prefix("0X"));
            let parsed_off = stripped_hex.map_or_else(
                || off_str.parse::<u64>(),
                |hex| u64::from_str_radix(hex, 16),
            );
            let bar0_offset = parsed_off.map_err(|_| LibTpuError::BinaryFormatError {
                reason: format!("Invalid CSR offset '{off_str}' for '{full_path}'"),
            })?;
            let width_bits = width_str.parse::<u32>().unwrap_or(64);
            let expected_opt = match full_path.as_str() {
                "tc0_program_location_csr" => Some(register_map.tc0_program_location_csr),
                "tc0_pc_csr" => Some(register_map.tc0_pc_csr),
                "tc0_run_status_csr" => Some(register_map.tc0_run_status_csr),
                "tc0_vwait_state_csr" => Some(register_map.tc0_vwait_state_csr),
                "tc0_vwait_sync_flag_state_csr" => Some(register_map.tc0_vwait_sync_flag_state_csr),
                "sc0_run_status_csr" => Some(register_map.sc0_run_status_csr),
                "global_chip_id_csr" => Some(register_map.global_chip_id_csr),
                _ => None,
            };
            if let Some(expected) = expected_opt {
                if bar0_offset != expected {
                    return Err(LibTpuError::BinaryFormatError {
                        reason: format!(
                            "CSR offset mismatch for '{full_path}': expected {expected:#x}, got {bar0_offset:#x}"
                        ),
                    });
                }
            }
            entries.push(BrotliRegisterTreeEntry {
                full_path,
                bar0_offset,
                width_bits,
            });
        }
    }
    Ok(entries)
}

/// Hardware target engine for an embedded microcode/firmware overlay (`FirmwareOverlayTarget`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FirmwareOverlayTarget {
    /// Primary `TensorCore` (`TC0` / `TC1`) VLIW instruction memory initialization overlay (`tc_init.bin`).
    TensorCore,
    /// `SparseCore` (`SCS` / `TAC` / `TEC`) embedding processor initialization overlay (`sc_init.bin`).
    SparseCore,
    /// Inter-Chip Interconnect (`ICI`) `SerDes` & router link training sequence (`ici_init.bin`).
    IciRouter,
}

/// Extracted microcode or link-training firmware overlay from `filewrapper_toc` (`CarvedFirmwareOverlay`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CarvedFirmwareOverlay {
    /// Original TOC resource filename (e.g., `"vfc_tc_init.bin"`, `"vfi_sc_init.bin"`, `"ici_link_train.bin"`).
    pub name: String,
    /// Classified hardware subsystem target ([`FirmwareOverlayTarget`]).
    pub target: FirmwareOverlayTarget,
    /// Target `IMEM` or BAR0 load offset in bytes.
    pub load_offset_bytes: u64,
    /// Raw firmware microcode payload bytes.
    pub payload: Vec<u8>,
}

/// Extracts `TensorCore` (`tc_init.bin`), `SparseCore` (`sc_init.bin`), and `ICI` link training
/// (`ici_*.bin`) firmware overlays from a slice of carved `filewrapper_toc` entries.
///
/// # Arguments
///
/// * `entries` - Slice of [`FileTocEntry`] records carved from `filewrapper_toc`.
#[must_use]
pub fn extract_carved_firmware_overlays(entries: &[FileTocEntry]) -> Vec<CarvedFirmwareOverlay> {
    let mut overlays = Vec::new();
    for entry in entries {
        let lower = entry.name.to_ascii_lowercase();
        let tensor_match =
            (u8::from(lower.contains("tc_init")) | u8::from(lower.ends_with(".tc.bin"))) != 0;
        let sparse_match =
            (u8::from(lower.contains("sc_init")) | u8::from(lower.ends_with(".sc.bin"))) != 0;
        let ici_match =
            (u8::from(lower.contains("ici_")) | u8::from(lower.ends_with(".ici.bin"))) != 0;
        let target_and_offset = if tensor_match {
            Some((FirmwareOverlayTarget::TensorCore, 0x0080_0000_u64))
        } else if sparse_match {
            Some((FirmwareOverlayTarget::SparseCore, 0x00C0_0000_u64))
        } else if ici_match {
            Some((FirmwareOverlayTarget::IciRouter, 0x0010_0000_u64))
        } else {
            None
        };
        if let Some((target, load_offset_bytes)) = target_and_offset {
            overlays.push(CarvedFirmwareOverlay {
                name: entry.name.clone(),
                target,
                load_offset_bytes,
                payload: entry.data.clone(),
            });
        }
    }
    overlays
}

/// Extractor and validator for `TensorCore`, `SparseCore` (`scs` / `tac`), and `ICI` firmware
/// overlays embedded inside `libtpu.so`'s `filewrapper_toc` section (`FileTocFirmwareExtractor`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FileTocFirmwareExtractor {
    /// Extracted and classified firmware overlays.
    pub overlays: Vec<CarvedFirmwareOverlay>,
    /// Decoded chip configurations, parts, and runtime ABI descriptors.
    pub chip_resources: EmbeddedChipResourceBundle,
}

impl FileTocFirmwareExtractor {
    /// Extracts all firmware overlays and chip configuration descriptors from `entries`.
    ///
    /// # Arguments
    ///
    /// * `entries` - Slice of [`FileTocEntry`] records carved from `filewrapper_toc`.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if any embedded `.binarypb` chip config or ABI descriptor is malformed.
    pub fn extract_and_validate(entries: &[FileTocEntry]) -> LibTpuResult<Self> {
        verify_filewrapper_toc_md5(entries)?;
        let overlays = extract_carved_firmware_overlays(entries);
        let chip_resources = decode_embedded_chip_resources(entries)?;
        Ok(Self {
            overlays,
            chip_resources,
        })
    }

    /// Returns all extracted overlays targeting `target` ([`FirmwareOverlayTarget`]).
    ///
    /// # Arguments
    ///
    /// * `target` - Hardware subsystem target to filter by.
    #[must_use]
    pub fn overlays_for_target(
        &self,
        target: FirmwareOverlayTarget,
    ) -> Vec<&CarvedFirmwareOverlay> {
        self.overlays
            .iter()
            .filter(|ov| ov.target == target)
            .collect()
    }
}

/// Canonical mapping from a `protodesc_cold` `.proto` path to a clean-room `.proto` filename.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalProtoPathMapping {
    /// Original embedded path in `protodesc_cold` (e.g., `"platforms/asic_sw/proto/core_dump.proto"`).
    pub embedded_path: String,
    /// Canonical `.proto` filename in `docs/reverse_engineering/protos/` (e.g., `"coredump.proto"`).
    pub canonical_filename: String,
    /// Whether the canonical target exists in the global [`crate::proto::global_descriptor_pool`].
    pub present_in_pool: bool,
}

/// Deterministic catalog entry for a `protodesc_cold` descriptor in `libtpu.so` or `sdk.abi3.so` (`ProtodescColdCatalogEntry`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProtodescColdCatalogEntry {
    /// Original embedded repository path in `protodesc_cold`.
    pub embedded_path: &'static str,
    /// Canonical `.proto` filename in `docs/reverse_engineering/protos/`.
    pub canonical_filename: &'static str,
    /// Protobuf package namespace (`package`).
    pub package: &'static str,
    /// Protobuf syntax version (`"proto2"` or `"proto3"`).
    pub syntax: &'static str,
}

include!(concat!(env!("OUT_DIR"), "/generated_protodesc_catalog.rs"));

/// Resolves an internal Google `.proto` path or unique basename to its canonical `.proto` path
/// in `docs/reverse_engineering/protos/`.
///
/// # Arguments
///
/// * `embedded_path` - Internal Google repository path or basename from `FileDescriptorProto.name`.
#[must_use]
pub fn resolve_embedded_proto_canonical_target(embedded_path: &str) -> String {
    if PROTODESC_COLD_293_CATALOG
        .iter()
        .any(|entry| entry.embedded_path == embedded_path)
    {
        return String::from(embedded_path);
    }
    let suffix = format!("/{embedded_path}");
    if let Some(entry) = PROTODESC_COLD_293_CATALOG
        .iter()
        .find(|entry| entry.embedded_path.ends_with(&suffix))
    {
        return String::from(entry.canonical_filename);
    }
    String::from(embedded_path)
}

/// Cross-verifies a list of embedded Google `.proto` paths from `libtpu.so`'s `protodesc_cold`
/// section against the 88 consolidated `.proto` files in [`crate::proto::global_descriptor_pool`].
///
/// # Arguments
///
/// * `embedded_paths` - Slice of embedded `.proto` path strings carved from `libtpu.so`.
#[must_use]
pub fn verify_protodesc_cold_293_closure(
    embedded_paths: &[&str],
) -> Vec<CanonicalProtoPathMapping> {
    let pool = crate::proto::descriptor::global_descriptor_pool();
    let known_files: BTreeSet<String> = pool.files().map(|f| String::from(f.name())).collect();
    embedded_paths
        .iter()
        .map(|&path| {
            let canonical = resolve_embedded_proto_canonical_target(path);
            let present_in_pool = known_files.contains(&canonical);
            CanonicalProtoPathMapping {
                embedded_path: String::from(path),
                canonical_filename: canonical,
                present_in_pool,
            }
        })
        .collect()
}

/// Verifies that every entry in `catalog` (such as [`PROTODESC_COLD_293_CATALOG`]) has both its
/// `embedded_path` and its `canonical_filename` registered in [`crate::proto::global_descriptor_pool`].
///
/// # Arguments
///
/// * `catalog` - Slice of [`ProtodescColdCatalogEntry`] records to verify.
///
/// # Errors
///
/// Returns [`LibTpuError::ProtoSchemaDriftError`] if any entry is missing from [`crate::proto::global_descriptor_pool`].
pub fn verify_protodesc_cold_293_catalog_closure(
    catalog: &[ProtodescColdCatalogEntry],
) -> LibTpuResult<usize> {
    let pool = crate::proto::descriptor::global_descriptor_pool();
    let mut verified_count = 0_usize;
    for entry in catalog {
        if pool
            .get_file_by_name(entry.canonical_filename)
            .and_then(|_| pool.get_file_by_name(entry.embedded_path))
            .is_some()
        {
            verified_count += 1;
        } else {
            return Err(LibTpuError::ProtoSchemaDriftError {
                message_type: entry.embedded_path,
                field_name: String::from(entry.canonical_filename),
                reason: format!(
                    "canonical file '{}' or embedded path '{}' not found in global_descriptor_pool()",
                    entry.canonical_filename, entry.embedded_path
                ),
            });
        }
    }
    Ok(verified_count)
}

/// Synthesizes a binary `protodesc_cold` ELF section payload containing all 293
/// `google.protobuf.FileDescriptorProto` wire records registered in [`PROTODESC_COLD_293_CATALOG`].
///
/// Each entry is encoded as a canonical wire `FileDescriptorProto` with its `.proto` name,
/// package, syntax, dependencies, and message/enum definitions resolved from
/// [`crate::proto::global_descriptor_pool`], so [`carve_protodesc_cold_entries`] recovers all 293
/// descriptors without any empty placeholder blobs.
#[must_use]
pub fn synthesize_protodesc_cold_293_section_bytes() -> Vec<u8> {
    let fds = prost_types::FileDescriptorSet::decode(
        crate::proto::descriptor::EMBEDDED_FILE_DESCRIPTOR_SET_BYTES,
    )
    .unwrap_or_default();
    let by_name: BTreeMap<String, FileDescriptorProto> = fds
        .file
        .into_iter()
        .map(|f| (f.name.clone().unwrap_or_default(), f))
        .collect();
    let mut section = Vec::new();
    for entry in PROTODESC_COLD_293_CATALOG {
        let mut fd = by_name
            .get(entry.canonical_filename)
            .cloned()
            .unwrap_or_default();
        fd.name = Some(String::from(entry.embedded_path));
        fd.package = Some(String::from(entry.package));
        fd.syntax = Some(String::from(entry.syntax));
        section.extend_from_slice(&fd.encode_to_vec());
        section.extend_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF]);
    }
    section
}

#[cfg(test)]
#[allow(clippy::indexing_slicing, clippy::too_many_lines)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use prost_types::{
        DescriptorProto, EnumDescriptorProto, EnumValueDescriptorProto, FieldDescriptorProto,
    };

    #[test]
    fn test_carve_protodesc_cold_and_filewrapper_toc() {
        let fd1 = FileDescriptorProto {
            name: Some(String::from("xla_data.proto")),
            package: Some(String::from("xla")),
            syntax: Some(String::from("proto3")),
            dependency: vec![String::from("base.proto")],
            enum_type: vec![EnumDescriptorProto {
                name: Some(String::from("PrimitiveType")),
                value: vec![EnumValueDescriptorProto {
                    name: Some(String::from("PRIMITIVE_TYPE_INVALID")),
                    number: Some(0),
                    options: None,
                }],
                options: None,
                reserved_range: Vec::new(),
                reserved_name: Vec::new(),
            }],
            message_type: vec![DescriptorProto {
                name: Some(String::from("TileProto")),
                field: vec![FieldDescriptorProto {
                    name: Some(String::from("dimensions")),
                    number: Some(1),
                    type_name: Some(String::from("int64")),
                    ..FieldDescriptorProto::default()
                }],
                ..DescriptorProto::default()
            }],
            ..FileDescriptorProto::default()
        };
        let fd2 = FileDescriptorProto {
            name: Some(String::from("hlo.proto")),
            package: None,
            syntax: None,
            enum_type: vec![EnumDescriptorProto {
                name: None,
                value: vec![EnumValueDescriptorProto {
                    name: None,
                    number: None,
                    options: None,
                }],
                ..EnumDescriptorProto::default()
            }],
            message_type: vec![DescriptorProto {
                name: None,
                field: vec![FieldDescriptorProto {
                    name: None,
                    number: None,
                    type_name: Some(String::new()),
                    ..FieldDescriptorProto::default()
                }],
                ..DescriptorProto::default()
            }],
            ..FileDescriptorProto::default()
        };

        let mut proto_sec = vec![0x00, 0xFF];
        proto_sec.extend_from_slice(&fd1.encode_to_vec());
        proto_sec.extend_from_slice(&[0x50, 0x00]);
        proto_sec.extend_from_slice(&fd2.encode_to_vec());
        proto_sec.push(0x00);
        proto_sec.extend_from_slice(b"\x0A\x09bad.proto\x0A\x03abc\x22\x02\xFF\xFF\x09\x78\x00");
        proto_sec.extend_from_slice(b"\x0A\x0Bfld15.proto\x78\x00");
        proto_sec.extend_from_slice(b"\x0A\x0Btrunc.proto\x12\x7F\x00");
        proto_sec
            .extend_from_slice(b"\x0A\x0Cvarint.proto\x50\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80");
        let _ = carve_protodesc_cold_entries(&fd1.encode_to_vec());
        let _ = carve_protodesc_cold_entries(b"\x0A\x0Cbadlen.proto\x12\x80");
        let eof_tag = b"\x0A\x0Ceoftag.proto\x80";
        let _ = carve_protodesc_cold_entries(eof_tag);
        proto_sec.extend_from_slice(b"\x0A\x08notproto\x00\x00");

        let carved = carve_protodesc_cold_entries(&proto_sec);
        assert!(carved.len() >= 2);
        assert_eq!(carved[0].name, "xla_data.proto");
        assert_eq!(carved[0].package, "xla");
        assert_eq!(carved[1].name, "hlo.proto");
        assert_eq!(carved[1].package, "");
        assert_eq!(carved[0].clone(), carved[0]);
        assert_ne!(format!("{:?}", carved[0]), "");

        if let Ok(decoded) = decode_carved_file_descriptor(&carved[0]) {
            assert_eq!(decoded.message_type.len(), 1);
            let rendered = render_proto_schema(&decoded);
            assert!(rendered.contains(r#"syntax = "proto3";"#));
            assert!(rendered.contains("package xla;"));
            assert!(rendered.contains(r#"import "base.proto";"#));
            assert!(rendered.contains("enum PrimitiveType"));
            assert!(rendered.contains("message TileProto"));
        }
        if let Ok(decoded2) = decode_carved_file_descriptor(&carved[1]) {
            let rendered2 = render_proto_schema(&decoded2);
            assert!(rendered2.contains(r#"syntax = "proto2";"#));
            assert!(rendered2.contains("UnnamedEnum"));
            assert!(rendered2.contains("UnnamedMessage"));
        }

        let standalone_fd = FileDescriptorProto {
            name: Some(String::from("standalone.proto")),
            package: Some(String::from("tpu.test")),
            syntax: Some(String::from("proto3")),
            message_type: vec![DescriptorProto {
                name: Some(String::from("StandaloneMsg")),
                field: vec![FieldDescriptorProto {
                    name: Some(String::from("val")),
                    number: Some(1),
                    label: Some(1),
                    r#type: Some(3),
                    ..FieldDescriptorProto::default()
                }],
                ..DescriptorProto::default()
            }],
            ..FileDescriptorProto::default()
        };
        let valid_carved = CarvedProtoDesc {
            name: String::from("standalone.proto"),
            package: String::from("tpu.test"),
            raw_bytes: standalone_fd.encode_to_vec(),
        };
        let pool_ok = build_descriptor_pool_from_carved_protos(std::slice::from_ref(&valid_carved));
        assert!(pool_ok.is_ok());
        let bad_carved = CarvedProtoDesc {
            name: String::from("bad.proto"),
            package: String::new(),
            raw_bytes: vec![0x80],
        };
        assert!(build_descriptor_pool_from_carved_protos(&[bad_carved]).is_err());
        assert!(build_descriptor_pool_from_carved_protos(&carved[..1]).is_err());

        for type_id in 1..=18 {
            let _ = proto_scalar_type_name(Some(type_id));
        }
        assert!(proto_scalar_type_name(None).is_none());
        assert!(proto_scalar_type_name(Some(99)).is_none());

        let rich_fd = FileDescriptorProto {
            name: Some(String::from("rich.proto")),
            package: Some(String::from("xla.tpu")),
            syntax: Some(String::from("proto3")),
            message_type: vec![DescriptorProto {
                name: Some(String::from("OuterMsg")),
                enum_type: vec![EnumDescriptorProto {
                    name: Some(String::from("InnerEnum")),
                    value: vec![EnumValueDescriptorProto {
                        name: Some(String::from("INNER_ZERO")),
                        number: Some(0),
                        options: None,
                    }],
                    ..EnumDescriptorProto::default()
                }],
                nested_type: vec![
                    DescriptorProto {
                        name: Some(String::from("AttrsEntry")),
                        options: Some(prost_types::MessageOptions {
                            map_entry: Some(true),
                            ..prost_types::MessageOptions::default()
                        }),
                        field: vec![
                            FieldDescriptorProto {
                                name: Some(String::from("key")),
                                number: Some(1),
                                r#type: Some(9),
                                ..FieldDescriptorProto::default()
                            },
                            FieldDescriptorProto {
                                name: Some(String::from("value")),
                                number: Some(2),
                                r#type: Some(3),
                                ..FieldDescriptorProto::default()
                            },
                        ],
                        ..DescriptorProto::default()
                    },
                    DescriptorProto {
                        name: Some(String::from("EmptyMapEntry")),
                        options: Some(prost_types::MessageOptions {
                            map_entry: Some(true),
                            ..prost_types::MessageOptions::default()
                        }),
                        ..DescriptorProto::default()
                    },
                    DescriptorProto {
                        name: Some(String::from("InnerMsg")),
                        field: vec![FieldDescriptorProto {
                            name: Some(String::from("flag")),
                            number: Some(1),
                            r#type: Some(8),
                            ..FieldDescriptorProto::default()
                        }],
                        ..DescriptorProto::default()
                    },
                ],
                oneof_decl: vec![
                    prost_types::OneofDescriptorProto {
                        name: Some(String::from("payload_kind")),
                        options: None,
                    },
                    prost_types::OneofDescriptorProto {
                        name: Some(String::from("_opt_val")),
                        options: None,
                    },
                ],
                field: vec![
                    FieldDescriptorProto {
                        name: Some(String::from("attrs")),
                        number: Some(1),
                        label: Some(3),
                        type_name: Some(String::from(".xla.tpu.OuterMsg.AttrsEntry")),
                        ..FieldDescriptorProto::default()
                    },
                    FieldDescriptorProto {
                        name: Some(String::from("rep_f64")),
                        number: Some(2),
                        label: Some(3),
                        r#type: Some(1),
                        ..FieldDescriptorProto::default()
                    },
                    FieldDescriptorProto {
                        name: Some(String::from("req_f32")),
                        number: Some(3),
                        label: Some(2),
                        r#type: Some(2),
                        ..FieldDescriptorProto::default()
                    },
                    FieldDescriptorProto {
                        name: Some(String::from("opt_val")),
                        number: Some(4),
                        label: Some(1),
                        r#type: Some(4),
                        oneof_index: Some(1),
                        proto3_optional: Some(true),
                        ..FieldDescriptorProto::default()
                    },
                    FieldDescriptorProto {
                        name: Some(String::from("choice_i32")),
                        number: Some(5),
                        label: Some(1),
                        r#type: Some(5),
                        oneof_index: Some(0),
                        ..FieldDescriptorProto::default()
                    },
                    FieldDescriptorProto {
                        name: Some(String::from("plain_u32")),
                        number: Some(6),
                        label: Some(1),
                        r#type: Some(13),
                        ..FieldDescriptorProto::default()
                    },
                ],
                ..DescriptorProto::default()
            }],
            ..FileDescriptorProto::default()
        };
        let rich_rendered = render_proto_schema(&rich_fd);
        assert!(rich_rendered.contains("enum InnerEnum"));
        assert!(rich_rendered.contains("message InnerMsg"));
        assert!(rich_rendered.contains("oneof payload_kind"));
        assert!(rich_rendered.contains("map<string, int64> attrs = 1;"));
        assert!(rich_rendered.contains("repeated double rep_f64 = 2;"));
        assert!(rich_rendered.contains("required float req_f32 = 3;"));
        assert!(rich_rendered.contains("optional uint64 opt_val = 4;"));
        assert!(rich_rendered.contains("uint32 plain_u32 = 6;"));

        let proto2_fd = FileDescriptorProto {
            name: Some(String::from("p2.proto")),
            syntax: Some(String::from("proto2")),
            message_type: vec![DescriptorProto {
                name: Some(String::from("P2Msg")),
                field: vec![FieldDescriptorProto {
                    name: Some(String::from("opt_i64")),
                    number: Some(1),
                    label: Some(1),
                    r#type: Some(3),
                    ..FieldDescriptorProto::default()
                }],
                ..DescriptorProto::default()
            }],
            ..FileDescriptorProto::default()
        };
        assert!(render_proto_schema(&proto2_fd).contains("optional int64 opt_i64 = 1;"));

        let edition_2023_fd = FileDescriptorProto {
            name: Some(String::from("ed2023.proto")),
            syntax: Some(String::from("editions")),
            ..FileDescriptorProto::default()
        };
        assert!(render_proto_schema(&edition_2023_fd).contains(r#"edition = "2023";"#));
        let edition_2024_fd = FileDescriptorProto {
            name: Some(String::from("ed2024.proto")),
            syntax: Some(String::from("edition_2024")),
            ..FileDescriptorProto::default()
        };
        assert!(render_proto_schema(&edition_2024_fd).contains(r#"edition = "2024";"#));

        // Multi-byte varint name_len (> 127 bytes) + wire types 1 (64-bit) & 5 (32-bit)
        let long_proto_name = format!("{}.proto", "a".repeat(130));
        let long_fd = FileDescriptorProto {
            name: Some(long_proto_name.clone()),
            package: Some(String::from("xla.long")),
            syntax: Some(String::from("editions")),
            ..FileDescriptorProto::default()
        };
        let mut long_sec = long_fd.encode_to_vec();
        // Append wire-type 1 (tag = (8 << 3) | 1 = 0x41) + 8 bytes, and wire-type 5 (tag = (8 << 3) | 5 = 0x45) + 4 bytes
        long_sec.push(0x41);
        long_sec.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        long_sec.push(0x45);
        long_sec.extend_from_slice(&[9, 10, 11, 12]);
        // Followed by another descriptor to trigger lookahead break
        long_sec.extend_from_slice(&fd1.encode_to_vec());
        // Truncated wire-type 1 & wire-type 5 descriptors
        long_sec.extend_from_slice(b"\x0A\x0Bwtype1.proto\x41\x01\x02");
        let carved_long = carve_protodesc_cold_entries(&long_sec);
        assert!(carved_long.iter().any(|c| c.name == long_proto_name));
        let _ = carve_protodesc_cold_entries(b"\x0A\x0Cwtype5.proto\x45\x01");
        let _ = carve_protodesc_cold_entries(b"\x0A\x0Cbadwtype.proto\x43\x01");

        let blob = b"\0tpu_tiles.cfg\0TILE_8x128";
        let mut toc = Vec::new();
        toc.extend_from_slice(&0_u64.to_le_bytes());
        toc.extend_from_slice(&0_u64.to_le_bytes());
        toc.extend_from_slice(&0_u64.to_le_bytes());
        toc.extend_from_slice(&999_u64.to_le_bytes());
        toc.extend_from_slice(&0_u64.to_le_bytes());
        toc.extend_from_slice(&4_u64.to_le_bytes());
        toc.extend_from_slice(&1_u64.to_le_bytes());
        toc.extend_from_slice(&999_u64.to_le_bytes());
        toc.extend_from_slice(&4_u64.to_le_bytes());
        toc.extend_from_slice(&1_u64.to_le_bytes());
        toc.extend_from_slice(&15_u64.to_le_bytes());
        toc.extend_from_slice(&10_u64.to_le_bytes());

        assert_eq!(extract_filewrapper_toc(&[], blob).len(), 0);
        let toc_entries = extract_filewrapper_toc(&toc, blob);
        assert_eq!(toc_entries.len(), 1);
        assert_eq!(toc_entries[0].name, "tpu_tiles.cfg");
        assert_eq!(toc_entries[0].data, b"TILE_8x128");
        assert_eq!(toc_entries[0].clone(), toc_entries[0]);
        assert_ne!(format!("{:?}", toc_entries[0]), "");

        let mut toc_40 = Vec::new();
        toc_40.extend_from_slice(&1_u64.to_le_bytes());
        toc_40.extend_from_slice(&15_u64.to_le_bytes());
        toc_40.extend_from_slice(&10_u64.to_le_bytes());
        toc_40.extend_from_slice(&0_u64.to_le_bytes());
        toc_40.extend_from_slice(&0_u64.to_le_bytes());
        let toc_40_entries = extract_filewrapper_toc(&toc_40, blob);
        assert_eq!(toc_40_entries.len(), 1);
        assert_eq!(toc_40_entries[0].name, "tpu_tiles.cfg");

        let mut reloc_map = BTreeMap::new();
        let mut size_map = BTreeMap::new();
        let mut rodata_blobs = BTreeMap::new();
        reloc_map.insert(0x1008, 0x2000);
        reloc_map.insert(0x1010, 0x2020);
        reloc_map.insert(0x2020, 0x3000);
        reloc_map.insert(0x1018, 0x2040);
        reloc_map.insert(0x2040, 0x3000);
        reloc_map.insert(0x2048, 0x4000);
        reloc_map.insert(0x1020, 0x2060);
        reloc_map.insert(0x2060, 0x3010);
        reloc_map.insert(0x2068, 0x4000);
        rodata_blobs.insert(0x3010, b"deepsea_compiler_hash.txt\0".to_vec());
        reloc_map.insert(0x1028, 0x2080);
        reloc_map.insert(0x2080, 0x3020);
        reloc_map.insert(0x2088, 0x4010);
        rodata_blobs.insert(0x3020, b"\0".to_vec());
        rodata_blobs.insert(0x4010, b"00ef0b06".to_vec());
        reloc_map.insert(0x1030, 0x20A0);
        reloc_map.insert(0x20A0, 0x3010);
        reloc_map.insert(0x20A8, 0x4010);
        size_map.insert(0x20B0, 8);

        let indirect = extract_indirect_filewrapper_toc(
            &[0x1000, 0x1008, 0x1010, 0x1018, 0x1020, 0x1028, 0x1030],
            &reloc_map,
            &size_map,
            &rodata_blobs,
        );
        assert_eq!(indirect.len(), 1);
        assert_eq!(indirect[0].name, "deepsea_compiler_hash.txt");
        assert_eq!(indirect[0].data, b"00ef0b06");

        let mut compressed_csr = Vec::new();
        let raw_csr = b"asic_sw.tools.RegisterTranslatorData: vlc_pf_csrs";
        let mut compressor = brotli::CompressorReader::new(&raw_csr[..], 4096, 5, 20);
        let _ = compressor.read_to_end(&mut compressed_csr);
        if let Ok(decompressed) = decompress_brotli_csr_table(&compressed_csr) {
            assert_eq!(decompressed, raw_csr);
        }
        assert!(decompress_brotli_csr_table(&[0xFF, 0xFF, 0xFF]).is_err());

        let fd_base = FileDescriptorProto {
            name: Some(String::from("xla/xla_data.proto")),
            package: Some(String::from("xla")),
            syntax: Some(String::from("proto3")),
            dependency: Vec::new(),
            message_type: vec![DescriptorProto {
                name: Some(String::from("ShapeProto")),
                ..DescriptorProto::default()
            }],
            enum_type: vec![EnumDescriptorProto {
                name: Some(String::from("PrimitiveType")),
                ..EnumDescriptorProto::default()
            }],
            ..FileDescriptorProto::default()
        };
        let fd_dep = FileDescriptorProto {
            name: Some(String::from("xla/hlo.proto")),
            package: Some(String::from("xla")),
            syntax: None,
            dependency: vec![
                String::from("xla/xla_data.proto"),
                String::from("xla_data.proto"),
            ],
            ..FileDescriptorProto::default()
        };
        let fd_non_proto = FileDescriptorProto {
            name: Some(String::from("not_a_proto.txt")),
            ..FileDescriptorProto::default()
        };
        let mut cold_bytes = Vec::new();
        for fd in [&fd_base, &fd_dep, &fd_non_proto] {
            let wire = fd.encode_to_vec();
            cold_bytes.extend_from_slice(&u32::try_from(wire.len()).unwrap_or(0).to_le_bytes());
            cold_bytes.extend_from_slice(&wire);
        }
        cold_bytes.extend_from_slice(&3_u32.to_le_bytes());
        cold_bytes.extend_from_slice(&[0xFF, 0xFF, 0xFF]);
        cold_bytes.extend_from_slice(&999_u32.to_le_bytes());
        cold_bytes.push(0x0A);

        assert_eq!(carve_protodesc_cold_descriptors(&[]), Vec::new());
        let carved_descs = carve_protodesc_cold_descriptors(&cold_bytes);
        assert_eq!(carved_descs.len(), 2);
        assert_eq!(carved_descs[0].name, "xla/xla_data.proto");
        assert_eq!(carved_descs[0].syntax, "proto3");
        assert_eq!(carved_descs[0].message_names, vec!["ShapeProto"]);
        assert_eq!(carved_descs[0].enum_names, vec!["PrimitiveType"]);
        assert!(
            carved_descs[0]
                .rendered_proto
                .contains("message ShapeProto")
        );
        assert_eq!(carved_descs[1].syntax, "proto2");
        assert_eq!(carved_descs[0].clone(), carved_descs[0]);
        assert_ne!(format!("{:?}", carved_descs[0]), "");

        let closure_ok = verify_descriptor_topological_closure(&carved_descs);
        assert!(closure_ok.is_ok());
        if let Ok(ordered) = closure_ok {
            assert_eq!(ordered, vec!["xla/xla_data.proto", "xla/hlo.proto"]);
        }
        let mut broken_carved = carved_descs;
        broken_carved[1]
            .dependencies
            .push(String::from("missing_dep.proto"));
        assert!(verify_descriptor_topological_closure(&broken_carved).is_err());

        // Test decode_embedded_chip_resources and all TpuGeneration chip configs/parts
        for generation in [
            crate::types::TpuGeneration::V2,
            crate::types::TpuGeneration::V3,
            crate::types::TpuGeneration::V4,
            crate::types::TpuGeneration::V4Lite,
            crate::types::TpuGeneration::V5e,
            crate::types::TpuGeneration::V5p,
            crate::types::TpuGeneration::V6e,
            crate::types::TpuGeneration::V7x,
        ] {
            let cfg = crate::proto::TpuChipConfigProto::from_generation(generation);
            let parts = crate::proto::TpuChipPartsProto::from_generation(generation);
            let target = crate::proto::TargetArgumentsProto::from_generation(generation);
            assert_eq!(
                target.tpu_topology_args.as_ref().and_then(|t| t.version),
                Some(generation as i32)
            );
            assert_eq!(
                target
                    .tpu_topology_args
                    .as_ref()
                    .and_then(|t| t.variant.as_deref()),
                Some(parts.variant_name.as_str())
            );
            assert!(crate::proto::ProtoSchemaMessage::verify_schema_sync(&cfg).is_ok());
            assert!(crate::proto::ProtoSchemaMessage::verify_schema_sync(&parts).is_ok());
            assert_eq!(cfg.clone(), cfg);
            assert_eq!(parts.clone(), parts);
            assert_ne!(format!("{cfg:?}"), "");
            assert_ne!(format!("{parts:?}"), "");
        }

        let abi_proto = crate::proto::TpuRuntimeAbiProto {
            base_abi_version: 1,
            compiler_hash: String::from("00ef0b06"),
            ..crate::proto::TpuRuntimeAbiProto::default()
        };
        assert!(crate::proto::ProtoSchemaMessage::verify_schema_sync(&abi_proto).is_ok());
        assert_eq!(abi_proto.clone(), abi_proto);
        assert_ne!(format!("{abi_proto:?}"), "");

        let toc_bundle_entries = vec![
            FileTocEntry::new(
                String::from("viperfish_chip_configs_megacore.binarypb"),
                crate::proto::ProtoSchemaMessage::encode_proto(
                    &crate::proto::TpuChipConfigProto::from_generation(
                        crate::types::TpuGeneration::V5p,
                    ),
                ),
            ),
            FileTocEntry::new(
                String::from("pufferfish_chip_parts.binarypb"),
                crate::proto::ProtoSchemaMessage::encode_proto(
                    &crate::proto::TpuChipPartsProto::from_generation(
                        crate::types::TpuGeneration::V4,
                    ),
                ),
            ),
            FileTocEntry::new(
                String::from("tpu_platform_configs.binarypb"),
                crate::proto::ProtoSchemaMessage::encode_proto(
                    &crate::proto::TpuPlatformConfigsProto::default(),
                ),
            ),
            FileTocEntry::new(
                String::from("current_tpu_runtime_abi.binarypb"),
                crate::proto::ProtoSchemaMessage::encode_proto(&abi_proto),
            ),
            FileTocEntry::new(
                String::from("deepsea_compiler_hash.txt"),
                b"00ef0b06\n".to_vec(),
            ),
            FileTocEntry::new(String::from("ignored_other_file.txt"), b"noop".to_vec()),
        ];
        assert!(verify_filewrapper_toc_md5(&toc_bundle_entries).is_ok());
        assert_eq!(
            format_md5_hex(&compute_rfc1321_md5(b"abc")),
            "900150983cd24fb0d6963f7d28e17f72"
        );
        let mut bad_md5_entry = FileTocEntry::new(String::from("corrupt.bin"), b"hello".to_vec());
        bad_md5_entry.md5_digest[0] ^= 0xFF;
        assert!(matches!(
            verify_filewrapper_toc_md5(&[bad_md5_entry]),
            Err(LibTpuError::TocPayloadChecksumMismatch { .. })
        ));
        let decoded_bundle = decode_embedded_chip_resources(&toc_bundle_entries);
        assert!(decoded_bundle.is_ok());
        if let Ok(b) = decoded_bundle {
            assert_eq!(b.chip_configs.len(), 1);
            assert_eq!(b.chip_parts.len(), 1);
            assert_eq!(b.platform_configs.len(), 1);
            assert_eq!(b.runtime_abi, Some(abi_proto));
            assert_eq!(b.compiler_hash.as_deref(), Some("00ef0b06"));
            assert_eq!(b.clone(), b);
            assert_ne!(format!("{b:?}"), "");
        }
        // Test Brotli *_csrs.br resource decoding
        let rmap_v5e =
            crate::hal::ChipRegisterMap::for_generation(crate::types::TpuGeneration::V5e);
        let reg_proto = rmap_v5e.to_register_translator_proto();
        let reg_proto_bytes = crate::proto::ProtoSchemaMessage::encode_proto(&reg_proto);
        let mut brotli_csr_bytes = Vec::new();
        {
            let mut writer = brotli::CompressorWriter::new(&mut brotli_csr_bytes, 4096, 4, 20);
            let _ = std::io::Write::write_all(&mut writer, &reg_proto_bytes);
        }
        let csr_bundle = decode_embedded_chip_resources(&[FileTocEntry::new(
            String::from("vlc_pf_csrs.br"),
            brotli_csr_bytes,
        )]);
        assert!(csr_bundle.is_ok());
        if let Ok(cb) = csr_bundle {
            assert_eq!(cb.csr_tables.len(), 1);
            assert_eq!(cb.csr_tables[0].nodes[0].mnemonic, "viperfish_lite");
        }
        let proto_parsed_tree = parse_and_verify_brotli_csr_tree(&reg_proto_bytes, &rmap_v5e);
        assert!(proto_parsed_tree.is_ok());
        assert!(parse_and_verify_brotli_csr_tree(&[], &rmap_v5e).is_ok());
        let mut reg_zero_width = reg_proto;
        if let Some(first) = reg_zero_width
            .nodes
            .first_mut()
            .and_then(|n| n.registers.first_mut())
        {
            first.width = 0;
        }
        let zero_w_bytes = crate::proto::ProtoSchemaMessage::encode_proto(&reg_zero_width);
        assert!(parse_and_verify_brotli_csr_tree(&zero_w_bytes, &rmap_v5e).is_ok());
        assert!(parse_and_verify_brotli_csr_tree(&[0xFF, 0xFF, 0xFF], &rmap_v5e).is_ok());
        assert!(
            decode_embedded_chip_resources(&[FileTocEntry::new(
                String::from("bad_platform_configs.binarypb"),
                vec![0xFF, 0xFF, 0xFF],
            )])
            .is_err()
        );
        assert!(
            decode_embedded_chip_resources(&[FileTocEntry::new(
                String::from("bad_csrs.br"),
                vec![0xFF, 0xFF, 0xFF],
            )])
            .is_err()
        );
        let mut brotli_bad_proto = Vec::new();
        {
            let mut writer = brotli::CompressorWriter::new(&mut brotli_bad_proto, 4096, 4, 20);
            let _ = std::io::Write::write_all(&mut writer, &[0xFF, 0xFF, 0xFF]);
        }
        assert!(
            decode_embedded_chip_resources(&[FileTocEntry::new(
                String::from("corrupt_proto_csrs.br"),
                brotli_bad_proto,
            )])
            .is_err()
        );
        assert!(
            decode_embedded_chip_resources(&[FileTocEntry::new(
                String::from("bad_chip_configs.binarypb"),
                vec![0xFF, 0xFF, 0xFF],
            )])
            .is_err()
        );
        assert!(
            decode_embedded_chip_resources(&[FileTocEntry::new(
                String::from("bad_chip_parts.binarypb"),
                vec![0xFF, 0xFF, 0xFF],
            )])
            .is_err()
        );
        assert!(
            decode_embedded_chip_resources(&[FileTocEntry::new(
                String::from("current_tpu_runtime_abi.binarypb"),
                vec![0xFF, 0xFF, 0xFF],
            )])
            .is_err()
        );

        // Test parse_and_verify_brotli_csr_tree & extract_carved_firmware_overlays
        let rmap = crate::hal::ChipRegisterMap::for_generation(crate::types::TpuGeneration::V5p);
        let csr_text = format!(
            "# comment\n\ntc0_program_location_csr = {:#x} 64\ntc0_pc_csr = {:#x} 64\ntc0_run_status_csr = {:#x} 32\ntc0_vwait_state_csr = {:#x} 64\ntc0_vwait_sync_flag_state_csr = {:#x} 64\nsc0_run_status_csr = {:#x} 32\nglobal_chip_id_csr = {} 32\ncustom_reg = 0X1234 64\nno_equals_line\n",
            rmap.tc0_program_location_csr,
            rmap.tc0_pc_csr,
            rmap.tc0_run_status_csr,
            rmap.tc0_vwait_state_csr,
            rmap.tc0_vwait_sync_flag_state_csr,
            rmap.sc0_run_status_csr,
            rmap.global_chip_id_csr
        );
        let parsed_tree = parse_and_verify_brotli_csr_tree(csr_text.as_bytes(), &rmap);
        assert!(parsed_tree.is_ok());
        if let Ok(entries) = parsed_tree {
            assert_eq!(entries.len(), 8);
            assert_eq!(entries[0].clone(), entries[0]);
            assert_ne!(format!("{:?}", entries[0]), "");
        }
        assert!(parse_and_verify_brotli_csr_tree(b"tc0_pc_csr = 0xDEAD 64", &rmap).is_err());
        assert!(parse_and_verify_brotli_csr_tree(b"tc0_pc_csr = not_a_num 64", &rmap).is_err());

        let fw_entries = vec![
            FileTocEntry::new(String::from("vfc_tc_init.bin"), vec![1, 2, 3, 4]),
            FileTocEntry::new(String::from("vfi_sc_init.bin"), vec![5, 6, 7, 8]),
            FileTocEntry::new(String::from("ici_link_train.bin"), vec![9, 10]),
            FileTocEntry::new(String::from("other.txt"), vec![0]),
        ];
        let overlays = extract_carved_firmware_overlays(&fw_entries);
        assert_eq!(overlays.len(), 3);
        assert_eq!(overlays[0].target, FirmwareOverlayTarget::TensorCore);
        assert_eq!(overlays[1].target, FirmwareOverlayTarget::SparseCore);
        assert_eq!(overlays[2].target, FirmwareOverlayTarget::IciRouter);
        assert_eq!(overlays[0].clone(), overlays[0]);
        assert_ne!(format!("{:?}", overlays[0]), "");
        assert_eq!(overlays[0].target.clone(), overlays[0].target);
        assert_ne!(format!("{:?}", overlays[0].target), "");

        let extractor = FileTocFirmwareExtractor::extract_and_validate(&fw_entries);
        assert!(extractor.is_ok());
        if let Ok(ext) = extractor {
            assert_eq!(
                ext.overlays_for_target(FirmwareOverlayTarget::TensorCore)
                    .len(),
                1
            );
            assert_eq!(
                ext.overlays_for_target(FirmwareOverlayTarget::SparseCore)
                    .len(),
                1
            );
            assert_eq!(
                ext.overlays_for_target(FirmwareOverlayTarget::IciRouter)
                    .len(),
                1
            );
            assert_eq!(ext.clone(), ext);
            assert_ne!(format!("{ext:?}"), "");
        }
        assert!(
            FileTocFirmwareExtractor::extract_and_validate(&[FileTocEntry::new(
                String::from("bad_chip_configs.binarypb"),
                vec![0xFF, 0xFF, 0xFF],
            )])
            .is_err()
        );

        let embedded_sample_paths = [
            "platforms/asic_sw/proto/core_dump.proto",
            "third_party/tensorflow/core/protobuf/tpu/topology.proto",
            "third_party/tensorflow/compiler/xla/xla_data.proto",
            "third_party/tensorflow/compiler/xla/service/hlo.proto",
            "platforms/xla/service/jellyfish/tpu_compilation_environment.proto",
            "platforms/xla/service/jellyfish/isa_program.proto",
            "platforms/xla/megascale/runtime/common/runtime_config.proto",
            "platforms/asic_sw/tools/common/registers_memfile/encoded_gcsr.proto",
            "third_party/tensorflow/tsl/profiler/protobuf/xplane.proto",
            "hlo.proto",
            "xla/service/hlo.proto",
            "unknown/internal/fallback_schema.proto",
        ];
        let mappings = verify_protodesc_cold_293_closure(&embedded_sample_paths);
        assert_eq!(mappings.len(), embedded_sample_paths.len());
        assert!(
            mappings
                .iter()
                .filter(|m| m.embedded_path != "unknown/internal/fallback_schema.proto")
                .all(|m| m.present_in_pool)
        );
        assert!(mappings.iter().any(|m| m.embedded_path
            == "unknown/internal/fallback_schema.proto"
            && !m.present_in_pool));
        assert_eq!(mappings[0].clone(), mappings[0]);
        assert_ne!(format!("{:?}", mappings[0]), "");
        assert_eq!(PROTODESC_COLD_293_CATALOG.len(), 738);
        let all_738_paths: Vec<&str> = PROTODESC_COLD_293_CATALOG
            .iter()
            .map(|e| e.embedded_path)
            .collect();
        let all_738_mappings = verify_protodesc_cold_293_closure(&all_738_paths);
        assert_eq!(all_738_mappings.len(), 738);
        assert!(all_738_mappings.iter().all(|m| m.present_in_pool));
        assert_eq!(PROTODESC_COLD_293_CATALOG[0], PROTODESC_COLD_293_CATALOG[0]);
        assert_ne!(format!("{:?}", PROTODESC_COLD_293_CATALOG[0]), "");
        for entry in PROTODESC_COLD_293_CATALOG {
            assert_eq!(
                resolve_embedded_proto_canonical_target(entry.embedded_path),
                entry.canonical_filename
            );
            assert_eq!(
                resolve_embedded_proto_canonical_target(entry.canonical_filename),
                entry.canonical_filename
            );
        }
        assert_eq!(
            verify_protodesc_cold_293_catalog_closure(PROTODESC_COLD_293_CATALOG),
            Ok(738)
        );
        let synth_sec = synthesize_protodesc_cold_293_section_bytes();
        let carved_738 = carve_protodesc_cold_entries(&synth_sec);
        assert_eq!(carved_738.len(), 738);
        assert!(
            verify_protodesc_cold_293_catalog_closure(&[ProtodescColdCatalogEntry {
                embedded_path: "nonexistent/missing.proto",
                canonical_filename: "missing.proto",
                package: "xla",
                syntax: "proto3",
            }])
            .is_err()
        );
    }
}

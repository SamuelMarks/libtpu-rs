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
/// # Arguments
///
/// * `bytes` - `protodesc_cold` byte slice.
/// * `start_idx` - Offset of the leading `0x0A` (`name`) tag.
fn scan_file_descriptor_wire_end(bytes: &[u8], start_idx: usize) -> usize {
    let mut cur = start_idx;
    while cur < bytes.len() {
        if cur > start_idx && bytes.get(cur).copied().unwrap_or(0) == 0x0A {
            let next_len = usize::from(bytes.get(cur + 1).copied().unwrap_or(0));
            let next_name = bytes.get(cur + 2..cur + 2 + next_len).unwrap_or_default();
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
            _ => break,
        }
    }
    cur
}

/// Scans a `protodesc_cold` section byte slice for wire-encoded `FileDescriptorProto` entries.
///
/// Matches entries starting with tag `0x0a` (field 1: `string name` ending in `".proto"`),
/// preserving all subsequent `FileDescriptorProto` fields (`message_type`, `enum_type`,
/// `dependency`, `syntax`).
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
            let name_len = usize::from(section_bytes.get(idx + 1).copied().unwrap_or(0));
            let name_slice = section_bytes
                .get(idx + 2..idx + 2 + name_len)
                .unwrap_or_default();
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
/// Reconstructs `syntax`, `package`, `import`, top-level and nested `enum` blocks,
/// top-level and nested `message` blocks, `oneof` declarations, `map<K, V>` fields,
/// field labels (`optional`, `required`, `repeated`), and all 18 protobuf scalar types.
///
/// # Arguments
///
/// * `fd` - Decoded `FileDescriptorProto` schema.
#[must_use]
pub fn render_proto_schema(fd: &FileDescriptorProto) -> String {
    let mut out = String::new();
    let syntax = fd.syntax.as_deref().unwrap_or("proto2");
    let is_proto3 = syntax == "proto3";
    let _ = writeln!(out, r#"syntax = "{syntax}";"#);
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
            entries.push(FileTocEntry {
                name,
                data: data_slice.to_vec(),
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
                entries.push(FileTocEntry {
                    name,
                    data: payload.to_vec(),
                });
            }
        }
    }
    entries
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

#[cfg(test)]
#[allow(
    for_loops_over_fallibles,
    clippy::indexing_slicing,
    clippy::too_many_lines
)]
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

        for decoded in decode_carved_file_descriptor(&carved[0]) {
            assert_eq!(decoded.message_type.len(), 1);
            let rendered = render_proto_schema(&decoded);
            assert!(rendered.contains(r#"syntax = "proto3";"#));
            assert!(rendered.contains("package xla;"));
            assert!(rendered.contains(r#"import "base.proto";"#));
            assert!(rendered.contains("enum PrimitiveType"));
            assert!(rendered.contains("message TileProto"));
        }
        for decoded2 in decode_carved_file_descriptor(&carved[1]) {
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
        for decompressed in decompress_brotli_csr_table(&compressed_csr) {
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
        for ordered in closure_ok {
            assert_eq!(ordered, vec!["xla/xla_data.proto", "xla/hlo.proto"]);
        }
        let mut broken_carved = carved_descs;
        broken_carved[1]
            .dependencies
            .push(String::from("missing_dep.proto"));
        assert!(verify_descriptor_topological_closure(&broken_carved).is_err());
    }
}

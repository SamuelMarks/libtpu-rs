//! Ground-truth ELF64 `protodesc_cold`, `filewrapper_toc`, and `encoded_gcsr` binary forensics.
//!
//! Extracts `DescriptorTable` protobuf descriptors, decompiles `.proto` schemas, verifies
//! `FileToc` MD5 digests, and decodes Brotli-compressed GCSR hardware register trees.

use crate::elf::carver::{
    BrotliRegisterTreeEntry, FileTocEntry, compute_rfc1321_md5, decompress_brotli_csr_table,
    proto_scalar_type_name,
};
use crate::error::{LibTpuError, LibTpuResult};
use crate::proto::{
    BarnaCoreConstantParameters, BarnaCoreModelParameters, EncodedRegister, ErrorSources,
    GcsrRegister, ProtoSchemaMessage as _, RegisterTranslatorData,
};
use goblin::elf::Elf;
use goblin::elf::section_header::SHT_NOBITS;
use prost::Message as _;
use prost_types::FileDescriptorProto;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt::Write as _;

/// Decoded `google::protobuf::internal::DescriptorTable` entry recovered from an ELF64 binary.
#[derive(Debug, Clone, PartialEq)]
pub struct DescriptorTableRecord {
    /// Virtual address of the 32/48-byte `DescriptorTable` structure in `.data` / `.data.rel.ro`.
    pub table_vaddr: u64,
    /// Virtual address of the wire-encoded `FileDescriptorProto` inside `protodesc_cold`.
    pub descriptor_vaddr: u64,
    /// Exact serialized byte length from `DescriptorTable::size`.
    pub descriptor_size: usize,
    /// Decoded `google.protobuf.FileDescriptorProto` message.
    pub file_descriptor: FileDescriptorProto,
    /// Verbatim wire-encoded `FileDescriptorProto` byte slice.
    pub raw_wire_bytes: Vec<u8>,
}

/// File-backed virtual-address span for an allocated ELF64 section.
#[derive(Debug, Clone, Copy)]
struct AllocSectionSpan {
    /// Section base virtual address (`sh_addr`).
    addr: u64,
    /// Section file byte offset (`sh_offset`).
    offset: u64,
    /// Section byte length (`sh_size`).
    size: u64,
}

/// Translates an ELF64 virtual address `vaddr` into a file byte offset using `spans`.
///
/// # Arguments
///
/// * `spans` - Slice of allocated file-backed ELF64 section spans.
/// * `vaddr` - Target virtual address to translate.
#[must_use]
fn vaddr_to_file_offset(spans: &[AllocSectionSpan], vaddr: u64) -> Option<usize> {
    for span in spans {
        let end = span.addr.saturating_add(span.size);
        if vaddr >= span.addr && vaddr < end {
            let delta = vaddr - span.addr;
            return usize::try_from(span.offset.saturating_add(delta)).ok();
        }
    }
    None
}

/// Collects allocated non-`SHT_NOBITS` section spans from a parsed [`Elf`] binary.
///
/// # Arguments
///
/// * `elf` - Parsed `goblin::elf::Elf` binary descriptor.
#[must_use]
fn collect_alloc_spans(elf: &Elf<'_>) -> Vec<AllocSectionSpan> {
    elf.section_headers
        .iter()
        .filter(|sh| sh.sh_addr > 0 && sh.sh_size > 0 && sh.sh_type != SHT_NOBITS)
        .map(|sh| AllocSectionSpan {
            addr: sh.sh_addr,
            offset: sh.sh_offset,
            size: sh.sh_size,
        })
        .collect()
}

/// Reads a NUL-terminated UTF-8 C string starting at virtual address `vaddr` in `elf_bytes`.
///
/// # Arguments
///
/// * `elf_bytes` - Full raw ELF64 binary slice.
/// * `spans` - Allocated section spans for virtual-address translation.
/// * `vaddr` - Starting virtual address of the C string.
#[must_use]
fn read_elf_cstring(elf_bytes: &[u8], spans: &[AllocSectionSpan], vaddr: u64) -> Option<String> {
    let start = vaddr_to_file_offset(spans, vaddr)?;
    let tail = elf_bytes.get(start..)?;
    let nul_pos = tail.iter().position(|&b| b == 0)?;
    let name_bytes = tail.get(..nul_pos)?;
    if name_bytes.is_empty() {
        return None;
    }
    Some(String::from_utf8_lossy(name_bytes).into_owned())
}

/// Extracts wire-encoded `FileDescriptorProto` records via `.rela.dyn` `DescriptorTable` relocations.
///
/// Scans dynamic relocations targeting `protodesc_cold` without requiring `.symtab`.
///
/// # Arguments
///
/// * `elf_bytes` - Raw byte slice of the 64-bit ELF shared object (`libtpu.so`).
///
/// # Errors
///
/// Returns [`LibTpuError`] if `elf_bytes` is not a valid 64-bit ELF or lacks a `protodesc_cold` section.
pub fn extract_elf64_descriptor_table_protos(
    elf_bytes: &[u8],
) -> LibTpuResult<Vec<DescriptorTableRecord>> {
    let elf = Elf::parse(elf_bytes)?;
    if !elf.is_64 {
        return Err(LibTpuError::BinaryFormatError {
            reason: String::from("expected 64-bit ELF shared object"),
        });
    }
    let pdesc_sh = elf
        .section_headers
        .iter()
        .find(|sh| elf.shdr_strtab.get_at(sh.sh_name) == Some("protodesc_cold"))
        .ok_or_else(|| LibTpuError::BinaryFormatError {
            reason: String::from("missing protodesc_cold section in ELF64 binary"),
        })?;

    let pdesc_start = pdesc_sh.sh_addr;
    let pdesc_end = pdesc_start.saturating_add(pdesc_sh.sh_size);
    let spans = collect_alloc_spans(&elf);
    let mut by_name: BTreeMap<String, DescriptorTableRecord> = BTreeMap::new();

    for rela in &elf.dynrelas {
        let Ok(target_vaddr) = u64::try_from(rela.r_addend.unwrap_or(-1)) else {
            continue;
        };
        if target_vaddr < pdesc_start || target_vaddr >= pdesc_end || rela.r_offset < 8 {
            continue;
        }
        let table_vaddr = rela.r_offset - 8;
        let Some(hdr_bytes) = vaddr_to_file_offset(&spans, table_vaddr)
            .and_then(|off| elf_bytes.get(off..off.saturating_add(8)))
        else {
            continue;
        };
        let flags_u32 = u32::from_le(bytemuck::pod_read_unaligned::<u32>(
            hdr_bytes.get(0..4).unwrap_or(&[1; 4]),
        ));
        let desc_size = u32::from_le(bytemuck::pod_read_unaligned::<u32>(
            hdr_bytes.get(4..8).unwrap_or(&[0; 4]),
        )) as usize;
        if flags_u32 != 0 || desc_size < 8 {
            continue;
        }
        let Some(wire_slice) = vaddr_to_file_offset(&spans, target_vaddr)
            .and_then(|off| elf_bytes.get(off..off.saturating_add(desc_size)))
        else {
            continue;
        };
        if let Ok(fd) = FileDescriptorProto::decode(wire_slice) {
            let name = fd.name.clone().unwrap_or_default();
            if std::path::Path::new(&name)
                .extension()
                .is_some_and(|ext| ext == "proto")
            {
                by_name.insert(
                    name,
                    DescriptorTableRecord {
                        table_vaddr,
                        descriptor_vaddr: target_vaddr,
                        descriptor_size: desc_size,
                        file_descriptor: fd,
                        raw_wire_bytes: wire_slice.to_vec(),
                    },
                );
            }
        }
    }

    Ok(by_name.into_values().collect())
}

/// Sanitizes dotted protobuf package or type names whose segments start with an ASCII digit,
/// optionally preserving a leading `.` for root-anchored `.proto` type resolution.
///
/// # Arguments
///
/// * `raw_name` - Raw protobuf package or qualified type name.
/// * `preserve_leading_dot` - Whether to keep a leading `.` when present on `raw_name`.
#[must_use]
pub fn sanitize_proto_ident(raw_name: &str, preserve_leading_dot: bool) -> String {
    let has_dot = raw_name.starts_with('.');
    let trimmed = raw_name.trim_start_matches('.');
    if trimmed.is_empty() {
        return String::new();
    }
    let mut parts = Vec::new();
    for seg in trimmed.split('.') {
        if seg.as_bytes().first().is_some_and(u8::is_ascii_digit) {
            parts.push(format!("c_{seg}"));
        } else {
            parts.push(String::from(seg));
        }
    }
    let joined = parts.join(".");
    if preserve_leading_dot && has_dot {
        format!(".{joined}")
    } else {
        joined
    }
}

/// Sanitizes dotted protobuf package or type names whose segments start with an ASCII digit
/// (e.g., Google codename-scrubbed `.xla.tpu.6acc60406` -> `xla.tpu.c_6acc60406`).
///
/// # Arguments
///
/// * `raw_name` - Raw protobuf package or qualified type name.
#[must_use]
pub fn sanitize_proto_qualified_name(raw_name: &str) -> String {
    sanitize_proto_ident(raw_name, false)
}

/// Checks whether an `EnumDescriptorProto` requires `proto2` closed-enum semantics because its
/// first enum discriminant is non-zero.
///
/// # Arguments
///
/// * `en` - Enum descriptor to inspect.
#[must_use]
fn enum_requires_proto2(en: &prost_types::EnumDescriptorProto) -> bool {
    en.value.first().and_then(|v| v.number).unwrap_or(0) != 0
}

/// Recursively checks whether a `DescriptorProto` requires `proto2` syntax due to closed enums,
/// `extensions`, `extend`, `required` fields, or explicit `default_value` annotations.
///
/// # Arguments
///
/// * `msg` - Message descriptor to inspect.
#[must_use]
fn message_requires_proto2(msg: &prost_types::DescriptorProto) -> bool {
    msg.enum_type.iter().any(enum_requires_proto2)
        || !msg.extension_range.is_empty()
        || !msg.extension.is_empty()
        || msg.field.iter().any(|f| {
            f.label == Some(2)
                || f.default_value
                    .as_deref()
                    .is_some_and(|d| !d.trim().is_empty())
        })
        || msg.nested_type.iter().any(message_requires_proto2)
}

/// Checks whether a `FileDescriptorProto` must be emitted as `syntax = "proto2";` to preserve
/// `proto2` or `EDITION_2023` features (closed enums, extensions, `required`, or defaults).
///
/// # Arguments
///
/// * `fd` - File descriptor to inspect.
#[must_use]
fn file_requires_proto2(fd: &FileDescriptorProto) -> bool {
    let raw_syntax = fd.syntax.as_deref().unwrap_or("");
    if raw_syntax.is_empty() || raw_syntax == "proto2" {
        return true;
    }
    !fd.extension.is_empty()
        || fd.enum_type.iter().any(enum_requires_proto2)
        || fd.message_type.iter().any(message_requires_proto2)
}

/// Resolves the sanitized, root-anchored `.proto` field type token for a `FieldDescriptorProto`.
///
/// # Arguments
///
/// * `field` - Protobuf field descriptor to inspect.
#[must_use]
fn resolve_decompiled_field_type(field: &prost_types::FieldDescriptorProto) -> String {
    field
        .type_name
        .as_deref()
        .filter(|t| !t.is_empty())
        .map_or_else(
            || String::from(proto_scalar_type_name(field.r#type).unwrap_or("bytes")),
            |t| sanitize_proto_ident(t, true),
        )
}

/// Formats a `proto2` `default = ...` field option clause when `field.default_value` is non-empty.
///
/// # Arguments
///
/// * `field` - Protobuf field descriptor carrying an optional `default_value`.
/// * `enum_val_renames` - Map of `(enum_fqn:orig_name) -> disambiguated_name` for scrubbed enums.
#[must_use]
fn format_decompiled_default_option(
    field: &prost_types::FieldDescriptorProto,
    enum_val_renames: &BTreeMap<String, String>,
) -> Option<String> {
    let raw_def = field.default_value.as_deref()?.trim();
    if raw_def.is_empty() {
        return None;
    }
    match field.r#type.unwrap_or(9) {
        9 => {
            let mut escaped = String::from("\"");
            for ch in raw_def.chars() {
                match ch {
                    '\\' => escaped.push_str("\\\\"),
                    '"' => escaped.push_str("\\\""),
                    '\n' => escaped.push_str("\\n"),
                    '\r' => escaped.push_str("\\r"),
                    '\t' => escaped.push_str("\\t"),
                    c => escaped.push(c),
                }
            }
            escaped.push('"');
            Some(format!("default = {escaped}"))
        }
        12 => Some(format!("default = \"{raw_def}\"")),
        14 => {
            let clean = sanitize_proto_ident(raw_def, false);
            let target_ty = sanitize_proto_ident(field.type_name.as_deref().unwrap_or(""), true);
            let key = format!("{target_ty}:{clean}");
            let final_ident = enum_val_renames.get(&key).cloned().unwrap_or(clean);
            Some(format!("default = {final_ident}"))
        }
        _ => Some(format!("default = {raw_def}")),
    }
}

/// Formats bracketed `[default = ..., packed = ...]` field options for a decompiled field.
///
/// # Arguments
///
/// * `field` - Protobuf field descriptor to inspect.
/// * `is_proto3` - Whether the enclosing file uses `proto3` syntax.
/// * `enum_val_renames` - Map of `(enum_fqn:orig_name) -> disambiguated_name` for scrubbed enums.
#[must_use]
fn format_decompiled_field_options(
    field: &prost_types::FieldDescriptorProto,
    is_proto3: bool,
    enum_val_renames: &BTreeMap<String, String>,
) -> String {
    let mut opts = Vec::new();
    if !is_proto3 {
        if let Some(def_opt) = format_decompiled_default_option(field, enum_val_renames) {
            opts.push(def_opt);
        }
    }
    if field.label == Some(3) {
        if let Some(packed) = field.options.as_ref().and_then(|o| o.packed) {
            opts.push(format!("packed = {packed}"));
        }
    }
    if opts.is_empty() {
        String::new()
    } else {
        format!(" [{}]", opts.join(", "))
    }
}

/// Renders an `EnumDescriptorProto` (including `option allow_alias = true;` and sibling-scope
/// collision disambiguation for scrubbed codenames) into `out`.
///
/// # Arguments
///
/// * `out` - Destination `.proto` text buffer.
/// * `en` - Enum descriptor to render.
/// * `parent_fqn` - Root-anchored fully-qualified parent scope (e.g., `".pkg.Msg"`).
/// * `indent` - Indentation prefix string.
/// * `sibling_names` - Mutable set of names already claimed in the enclosing scope.
/// * `enum_val_renames` - Output map recording any disambiguated enum constant names.
fn render_decompiled_enum(
    out: &mut String,
    en: &prost_types::EnumDescriptorProto,
    parent_fqn: &str,
    indent: &str,
    sibling_names: &mut BTreeSet<String>,
    enum_val_renames: &mut BTreeMap<String, String>,
) {
    let en_name = sanitize_proto_ident(en.name.as_deref().unwrap_or("UnnamedEnum"), false);
    let enum_fqn = format!("{parent_fqn}.{en_name}");
    let _ = writeln!(out, "{indent}enum {en_name} {{");
    let mut seen_numbers = BTreeSet::new();
    let has_alias = en
        .value
        .iter()
        .any(|v| !seen_numbers.insert(v.number.unwrap_or(0)));
    if has_alias {
        let _ = writeln!(out, "{indent}  option allow_alias = true;");
    }
    for val in &en.value {
        let orig_clean = sanitize_proto_ident(val.name.as_deref().unwrap_or("UNKNOWN"), false);
        let mut v_name = orig_clean.clone();
        if sibling_names.contains(&v_name) {
            let upper = v_name.to_ascii_uppercase();
            v_name = if sibling_names.contains(&upper) {
                format!("{en_name}_{v_name}")
            } else {
                upper
            };
            enum_val_renames.insert(format!("{enum_fqn}:{orig_clean}"), v_name.clone());
        }
        sibling_names.insert(v_name.clone());
        let v_num = val.number.unwrap_or(0);
        let _ = writeln!(out, "{indent}  {v_name} = {v_num};");
    }
    let _ = writeln!(out, "{indent}}}");
}

/// Renders grouped `extend .target { ... }` blocks for a slice of extension `FieldDescriptorProto`s.
///
/// # Arguments
///
/// * `out` - Destination `.proto` text buffer.
/// * `extensions` - Extension field descriptors to group by `extendee`.
/// * `indent` - Indentation prefix string.
/// * `is_proto3` - Whether the enclosing file uses `proto3` syntax.
/// * `enum_val_renames` - Map of disambiguated enum value identifiers.
fn render_decompiled_extensions(
    out: &mut String,
    extensions: &[prost_types::FieldDescriptorProto],
    indent: &str,
    is_proto3: bool,
    enum_val_renames: &BTreeMap<String, String>,
) {
    let mut by_extendee: BTreeMap<String, Vec<&prost_types::FieldDescriptorProto>> =
        BTreeMap::new();
    for ext in extensions {
        if let Some(target) = ext.extendee.as_deref().filter(|t| !t.is_empty()) {
            by_extendee
                .entry(sanitize_proto_ident(target, true))
                .or_default()
                .push(ext);
        }
    }
    for (target, fields) in by_extendee {
        let _ = writeln!(out, "{indent}extend {target} {{");
        for field in fields {
            let f_name = sanitize_proto_ident(field.name.as_deref().unwrap_or("ext_field"), false);
            let f_num = field.number.unwrap_or(1);
            let f_ty = resolve_decompiled_field_type(field);
            let prefix = if field.label == Some(3) {
                "repeated "
            } else {
                "optional "
            };
            let opts = format_decompiled_field_options(field, is_proto3, enum_val_renames);
            let _ = writeln!(out, "{indent}  {prefix}{f_ty} {f_name} = {f_num}{opts};");
        }
        let _ = writeln!(out, "{indent}}}");
    }
}

/// Renders a `DescriptorProto` message (including `extensions`, `reserved`, nested enums/messages,
/// direct-child `map<K, V>`, `oneof`, `proto3_optional`, defaults, and nested `extend`) into `out`.
///
/// # Arguments
///
/// * `out` - Destination `.proto` text buffer.
/// * `msg` - Message descriptor to render.
/// * `parent_fqn` - Root-anchored fully-qualified parent scope (e.g., `".pkg"`).
/// * `indent` - Indentation prefix string.
/// * `is_proto3` - Whether enclosing syntax uses `proto3` implicit-presence defaults.
/// * `enum_val_renames` - Output map recording any disambiguated enum constant names.
fn render_decompiled_message(
    out: &mut String,
    msg: &prost_types::DescriptorProto,
    parent_fqn: &str,
    indent: &str,
    is_proto3: bool,
    enum_val_renames: &mut BTreeMap<String, String>,
) {
    let raw_msg_name = msg.name.as_deref().unwrap_or("UnnamedMessage");
    let msg_name = sanitize_proto_ident(raw_msg_name, false);
    let msg_fqn = format!("{parent_fqn}.{msg_name}");
    let _ = writeln!(out, "{indent}message {msg_name} {{");
    let inner = format!("{indent}  ");

    for er in &msg.extension_range {
        let start = er.start.unwrap_or(1);
        let end_excl = er.end.unwrap_or(start + 1);
        if end_excl >= 536_870_912 {
            let _ = writeln!(out, "{inner}extensions {start} to max;");
        } else {
            let end_incl = end_excl - 1;
            if start == end_incl {
                let _ = writeln!(out, "{inner}extensions {start};");
            } else {
                let _ = writeln!(out, "{inner}extensions {start} to {end_incl};");
            }
        }
    }

    let mut map_entries: BTreeMap<&str, (String, String)> = BTreeMap::new();
    for nested in &msg.nested_type {
        let is_map = nested
            .options
            .as_ref()
            .and_then(|o| o.map_entry)
            .unwrap_or(false);
        if is_map {
            let entry_name = nested.name.as_deref().unwrap_or_default();
            let k_opt = nested
                .field
                .iter()
                .find(|f| f.number == Some(1))
                .map(resolve_decompiled_field_type);
            let v_opt = nested
                .field
                .iter()
                .find(|f| f.number == Some(2))
                .map(resolve_decompiled_field_type);
            let k_ty = String::from(k_opt.as_deref().unwrap_or("string"));
            let v_ty = String::from(v_opt.as_deref().unwrap_or("bytes"));
            map_entries.insert(entry_name, (k_ty, v_ty));
        }
    }

    for rr in &msg.reserved_range {
        let start = rr.start.unwrap_or(1);
        let end_incl = rr.end.unwrap_or(start + 1) - 1;
        if start == end_incl {
            let _ = writeln!(out, "{inner}reserved {start};");
        } else {
            let _ = writeln!(out, "{inner}reserved {start} to {end_incl};");
        }
    }
    for rname in &msg.reserved_name {
        let _ = writeln!(out, "{inner}reserved {rname:?};");
    }

    let mut scope_names: BTreeSet<String> = msg
        .field
        .iter()
        .map(|f| sanitize_proto_ident(f.name.as_deref().unwrap_or("field"), false))
        .collect();
    for nested in &msg.nested_type {
        if let Some(n) = nested.name.as_deref() {
            scope_names.insert(sanitize_proto_ident(n, false));
        }
    }

    for en in &msg.enum_type {
        render_decompiled_enum(
            out,
            en,
            &msg_fqn,
            &inner,
            &mut scope_names,
            enum_val_renames,
        );
    }
    for nested in &msg.nested_type {
        let is_map = nested
            .options
            .as_ref()
            .and_then(|o| o.map_entry)
            .unwrap_or(false);
        if !is_map {
            render_decompiled_message(out, nested, &msg_fqn, &inner, is_proto3, enum_val_renames);
        }
    }

    render_decompiled_message_fields(out, msg, &inner, is_proto3, &map_entries, enum_val_renames);
    render_decompiled_extensions(out, &msg.extension, &inner, is_proto3, enum_val_renames);
    let _ = writeln!(out, "{indent}}}");
}

/// Renders `oneof` declarations and non-`oneof` fields (including direct-child `map<K, V>`) for a `DescriptorProto`.
///
/// # Arguments
///
/// * `out` - Destination `.proto` text buffer.
/// * `msg` - Message descriptor whose `oneof`s and fields are emitted.
/// * `inner` - Indentation prefix string for message body members.
/// * `is_proto3` - Whether enclosing syntax uses `proto3` implicit-presence defaults.
/// * `map_entries` - Direct-child synthetic `map_entry` types keyed by short entry message name.
/// * `enum_val_renames` - Map of disambiguated enum constant names for default options.
fn render_decompiled_message_fields(
    out: &mut String,
    msg: &prost_types::DescriptorProto,
    inner: &str,
    is_proto3: bool,
    map_entries: &BTreeMap<&str, (String, String)>,
    enum_val_renames: &BTreeMap<String, String>,
) {
    let raw_msg_name = msg.name.as_deref().unwrap_or("UnnamedMessage");
    for (idx, oneof) in msg.oneof_decl.iter().enumerate() {
        let idx_i32 = i32::try_from(idx).unwrap_or(-1);
        let members: Vec<&prost_types::FieldDescriptorProto> = msg
            .field
            .iter()
            .filter(|f| f.oneof_index == Some(idx_i32) && !f.proto3_optional.unwrap_or(false))
            .collect();
        if !members.is_empty() {
            let oo_name =
                sanitize_proto_ident(oneof.name.as_deref().unwrap_or("unnamed_oneof"), false);
            let _ = writeln!(out, "{inner}oneof {oo_name} {{");
            for field in members {
                let f_name = sanitize_proto_ident(field.name.as_deref().unwrap_or("field"), false);
                let f_num = field.number.unwrap_or(1);
                let f_ty = resolve_decompiled_field_type(field);
                let opts = format_decompiled_field_options(field, is_proto3, enum_val_renames);
                let _ = writeln!(out, "{inner}  {f_ty} {f_name} = {f_num}{opts};");
            }
            let _ = writeln!(out, "{inner}}}");
        }
    }

    for field in &msg.field {
        if field.oneof_index.is_some() && !field.proto3_optional.unwrap_or(false) {
            continue;
        }
        let f_name = sanitize_proto_ident(field.name.as_deref().unwrap_or("field"), false);
        let f_num = field.number.unwrap_or(1);
        let f_ty = resolve_decompiled_field_type(field);
        let direct_map = field
            .type_name
            .as_deref()
            .and_then(|full| {
                let short = full.rsplit('.').next().unwrap_or(full);
                let expected = format!(".{raw_msg_name}.{short}");
                full.ends_with(&expected).then_some(short)
            })
            .and_then(|short| map_entries.get(short));
        if let Some((k_ty, v_ty)) = direct_map {
            let _ = writeln!(out, "{inner}map<{k_ty}, {v_ty}> {f_name} = {f_num};");
            continue;
        }
        let prefix = match field.label {
            Some(3) => "repeated ",
            Some(2) => "required ",
            Some(1) if !is_proto3 || field.proto3_optional.unwrap_or(false) => "optional ",
            _ => "",
        };
        let opts = format_decompiled_field_options(field, is_proto3, enum_val_renames);
        let _ = writeln!(out, "{inner}{prefix}{f_ty} {f_name} = {f_num}{opts};");
    }
}

/// Renders a `ServiceDescriptorProto` (`rpc Name(stream? Req) returns (stream? Resp);`) into `out`.
///
/// # Arguments
///
/// * `out` - Destination `.proto` text buffer.
/// * `svc` - Service descriptor to render.
fn render_decompiled_service(out: &mut String, svc: &prost_types::ServiceDescriptorProto) {
    let svc_name = sanitize_proto_ident(svc.name.as_deref().unwrap_or("UnnamedService"), false);
    let _ = writeln!(out, "service {svc_name} {{");
    for method in &svc.method {
        let m_name = sanitize_proto_ident(method.name.as_deref().unwrap_or("Call"), false);
        let in_ty = sanitize_proto_ident(
            method
                .input_type
                .as_deref()
                .unwrap_or(".google.protobuf.Empty"),
            true,
        );
        let out_ty = sanitize_proto_ident(
            method
                .output_type
                .as_deref()
                .unwrap_or(".google.protobuf.Empty"),
            true,
        );
        let c_stream = if method.client_streaming.unwrap_or(false) {
            "stream "
        } else {
            ""
        };
        let s_stream = if method.server_streaming.unwrap_or(false) {
            "stream "
        } else {
            ""
        };
        let _ = writeln!(
            out,
            "  rpc {m_name}({c_stream}{in_ty}) returns ({s_stream}{out_ty});"
        );
    }
    let _ = writeln!(out, "}}");
}

/// Checks whether a `FileDescriptorProto` declares extensions against `.google.protobuf.*Options`
/// so `import "google/protobuf/descriptor.proto";` is emitted when not explicitly listed.
///
/// # Arguments
///
/// * `fd` - File descriptor to inspect.
#[must_use]
fn file_extends_google_descriptor(fd: &FileDescriptorProto) -> bool {
    /// Checks whether `m` or any of its nested messages declare `.google.protobuf.*` extensions.
    ///
    /// # Arguments
    ///
    /// * `m` - Message descriptor to inspect recursively.
    fn msg_extends_desc(m: &prost_types::DescriptorProto) -> bool {
        m.extension.iter().any(|e| {
            e.extendee
                .as_deref()
                .is_some_and(|t| t.starts_with(".google.protobuf."))
        }) || m.nested_type.iter().any(msg_extends_desc)
    }
    fd.extension.iter().any(|e| {
        e.extendee
            .as_deref()
            .is_some_and(|t| t.starts_with(".google.protobuf."))
    }) || fd.message_type.iter().any(msg_extends_desc)
}

/// Decompiles a wire-decoded `FileDescriptorProto` into lossless, `protox`-compilable `.proto` text.
///
/// Preserves root-anchored `.pkg.Type` references, `import public`, `option allow_alias = true;`,
/// `extensions`, `extend`, `service`, `default = ...`, `packed = ...`, direct-child `map<K, V>`,
/// and disambiguates Google codename-scrubbed enum/field scope collisions.
///
/// # Arguments
///
/// * `fd` - Decoded `google.protobuf.FileDescriptorProto` to decompile.
#[must_use]
pub fn decompile_file_descriptor_to_proto(fd: &FileDescriptorProto) -> String {
    let mut out = String::new();
    let is_proto3 = !file_requires_proto2(fd);
    let emitted_syntax = if is_proto3 { "proto3" } else { "proto2" };
    let _ = writeln!(out, r#"syntax = "{emitted_syntax}";"#);

    let pkg_fqn = fd
        .package
        .as_deref()
        .filter(|p| !p.is_empty())
        .map_or_else(String::new, |pkg| {
            let clean_pkg = sanitize_proto_ident(pkg, false);
            let _ = writeln!(out, "package {clean_pkg};");
            format!(".{clean_pkg}")
        });

    let public_indices: BTreeSet<i32> = fd.public_dependency.iter().copied().collect();
    let mut has_descriptor_dep = false;
    for (idx, dep) in fd.dependency.iter().enumerate() {
        if dep == "google/protobuf/descriptor.proto" || dep == "net/proto2/proto/descriptor.proto" {
            has_descriptor_dep = true;
        }
        let is_pub = i32::try_from(idx).is_ok_and(|i| public_indices.contains(&i));
        if is_pub {
            let _ = writeln!(out, r#"import public "{dep}";"#);
        } else {
            let _ = writeln!(out, r#"import "{dep}";"#);
        }
    }
    if !has_descriptor_dep && file_extends_google_descriptor(fd) {
        let _ = writeln!(out, r#"import "google/protobuf/descriptor.proto";"#);
    }

    let mut enum_val_renames = BTreeMap::new();
    let mut top_names: BTreeSet<String> = fd
        .message_type
        .iter()
        .filter_map(|m| m.name.as_deref())
        .map(|n| sanitize_proto_ident(n, false))
        .collect();
    for en in &fd.enum_type {
        render_decompiled_enum(
            &mut out,
            en,
            &pkg_fqn,
            "",
            &mut top_names,
            &mut enum_val_renames,
        );
    }
    for msg in &fd.message_type {
        render_decompiled_message(
            &mut out,
            msg,
            &pkg_fqn,
            "",
            is_proto3,
            &mut enum_val_renames,
        );
    }
    render_decompiled_extensions(&mut out, &fd.extension, "", is_proto3, &enum_val_renames);
    for svc in &fd.service {
        render_decompiled_service(&mut out, svc);
    }
    out
}

/// Topologically sorts a slice of `FileDescriptorProto` schemas so all `dependency` imports
/// precede the files that import them.
///
/// # Arguments
///
/// * `descriptors` - Unordered slice of `FileDescriptorProto` schemas.
///
/// # Errors
///
/// Returns [`LibTpuError::BinaryFormatError`] if a cyclic `import` graph is detected.
pub fn topological_sort_file_descriptors(
    descriptors: &[FileDescriptorProto],
) -> LibTpuResult<Vec<FileDescriptorProto>> {
    let mut by_name: BTreeMap<String, FileDescriptorProto> = BTreeMap::new();
    for fd in descriptors {
        if let Some(name) = fd.name.clone() {
            by_name.insert(name, fd.clone());
        }
    }

    let mut in_degree: BTreeMap<String, usize> = BTreeMap::new();
    let mut dependents: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (name, fd) in &by_name {
        let mut deg = 0_usize;
        for dep in &fd.dependency {
            if by_name.contains_key(dep) {
                deg += 1;
                dependents
                    .entry(dep.clone())
                    .or_default()
                    .push(name.clone());
            }
        }
        in_degree.insert(name.clone(), deg);
    }

    let mut ready: VecDeque<String> = in_degree
        .iter()
        .filter(|&(_, &deg)| deg == 0)
        .map(|(k, _)| k.clone())
        .collect();
    let mut sorted = Vec::with_capacity(by_name.len());

    while let Some(cur) = ready.pop_front() {
        sorted.extend(by_name.get(&cur).cloned());
        if let Some(children) = dependents.get(&cur) {
            for child in children {
                let deg = in_degree.entry(child.clone()).or_default();
                *deg = deg.saturating_sub(1);
                if *deg == 0 {
                    ready.push_back(child.clone());
                }
            }
        }
    }

    if sorted.len() != by_name.len() {
        return Err(LibTpuError::BinaryFormatError {
            reason: String::from("cyclic import dependency detected in FileDescriptorSet"),
        });
    }
    Ok(sorted)
}

/// Extracts all embedded `FileToc` resources from an ELF64 `filewrapper_toc` section.
///
/// Resolves `.rela.dyn` pointer chains and sets `md5_verified` per entry using RFC 1321 MD5.
///
/// # Arguments
///
/// * `elf_bytes` - Raw byte slice of the 64-bit ELF shared object (`libtpu.so`).
///
/// # Errors
///
/// Returns [`LibTpuError`] if `elf_bytes` is malformed or lacks a `filewrapper_toc` section.
pub fn extract_elf64_filewrapper_toc_entries(elf_bytes: &[u8]) -> LibTpuResult<Vec<FileTocEntry>> {
    let elf = Elf::parse(elf_bytes)?;
    let ftoc_sh = elf
        .section_headers
        .iter()
        .find(|sh| elf.shdr_strtab.get_at(sh.sh_name) == Some("filewrapper_toc"))
        .ok_or_else(|| LibTpuError::BinaryFormatError {
            reason: String::from("missing filewrapper_toc section in ELF64 binary"),
        })?;

    let spans = collect_alloc_spans(&elf);
    let mut relocs: BTreeMap<u64, u64> = BTreeMap::new();
    for rela in &elf.dynrelas {
        if let Ok(target) = u64::try_from(rela.r_addend.unwrap_or(-1)) {
            relocs.insert(rela.r_offset, target);
        }
    }

    let mut entries = Vec::new();
    let mut slot_off = 0_u64;
    while slot_off.saturating_add(8) <= ftoc_sh.sh_size {
        let slot_vaddr = ftoc_sh.sh_addr.saturating_add(slot_off);
        slot_off = slot_off.saturating_add(8);
        let Some(&first_toc_vaddr) = relocs.get(&slot_vaddr) else {
            continue;
        };
        let mut cur_vaddr = first_toc_vaddr;
        while let Some(&name_vaddr) = relocs.get(&cur_vaddr) {
            let entry_vaddr = cur_vaddr;
            cur_vaddr = cur_vaddr.saturating_add(40);
            let Some(&data_vaddr) = relocs.get(&entry_vaddr.saturating_add(8)) else {
                break;
            };
            let Some(entry_raw) = vaddr_to_file_offset(&spans, entry_vaddr)
                .and_then(|off| elf_bytes.get(off..off.saturating_add(40)))
            else {
                break;
            };
            let size_u64 = u64::from_le(bytemuck::pod_read_unaligned::<u64>(
                entry_raw.get(16..24).unwrap_or(&[0; 8]),
            ));
            let data_len = usize::try_from(size_u64).unwrap_or(usize::MAX);
            let Some(name) = read_elf_cstring(elf_bytes, &spans, name_vaddr) else {
                continue;
            };
            let Some(payload) = vaddr_to_file_offset(&spans, data_vaddr)
                .and_then(|off| elf_bytes.get(off..off.saturating_add(data_len)))
            else {
                continue;
            };
            let md5_digest =
                bytemuck::pod_read_unaligned::<[u8; 16]>(entry_raw.get(24..40).unwrap_or(&[0; 16]));
            let md5_verified =
                (md5_digest == [0_u8; 16]) || (compute_rfc1321_md5(payload) == md5_digest);
            entries.push(FileTocEntry {
                name,
                data: payload.to_vec(),
                md5_digest,
                md5_verified,
            });
        }
    }

    Ok(entries)
}

/// Decompresses a Brotli `*_pf_csrs.br` blob and decodes its `RegisterTranslatorData` CSR map.
///
/// Resolves both inline `EncodedNode` registers and shared `node_layout_table` templates.
///
/// # Arguments
///
/// * `compressed_br` - Raw Brotli-compressed `RegisterTranslatorData` bytes from `filewrapper_toc`.
///
/// # Errors
///
/// Returns [`LibTpuError`] if Brotli decompression or protobuf wire decoding fails.
pub fn decode_encoded_gcsr_brotli_csr_map(
    compressed_br: &[u8],
) -> LibTpuResult<Vec<BrotliRegisterTreeEntry>> {
    let decompressed = decompress_brotli_csr_table(compressed_br)?;
    let reg_data = RegisterTranslatorData::decode_proto(&decompressed)?;
    let mut flattened = Vec::new();

    for node in &reg_data.nodes {
        let active_regs: &[EncodedRegister] = if !node.registers.is_empty() {
            &node.registers
        } else if let Ok(layout_idx) = usize::try_from(node.layout_index)
            && let Some(template) = reg_data.node_layout_table.get(layout_idx)
        {
            &template.registers
        } else {
            &[]
        };

        for reg in active_regs {
            let width_bits = u32::try_from(reg.width)
                .ok()
                .filter(|&w| w > 0)
                .map_or(64, |w| w.saturating_mul(8));
            flattened.push(BrotliRegisterTreeEntry {
                full_path: format!("{}.{}", node.mnemonic, reg.name),
                bar0_offset: node.base_offset.saturating_add(reg.offset),
                width_bits,
            });
        }
    }

    Ok(flattened)
}

/// Hardware ASIC family identifier for static `.lrodata` CSR range shards in `libtpu.so`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StaticCsrFamily {
    /// Cloud TPU v2 (`jf` / Jellyfish) and v3 (`df` / Dragonfish) shared `jxc` ASIC family.
    Jxc,
    /// Cloud TPU v4 (`pf` / Pufferfish) `pfc` ASIC family.
    Pfc,
    /// Cloud TPU v4lite (`pf_lite` / Pufferfish Lite) `plc` ASIC family.
    Plc,
    /// Cloud TPU v5e (`vf_lite` / Viperfish Lite) `vlc` ASIC family.
    Vlc,
    /// Cloud TPU v5p (`vf` / Viperfish) `vfc` ASIC family.
    Vfc,
    /// Cloud TPU v6e (`gf` / Ghostlite / Trillium) `glc` ASIC family.
    Glc,
    /// Cloud TPU v7x (`iw` / Ironwood / `6acc60406`) `gfc` ASIC family.
    Gfc,
}

impl StaticCsrFamily {
    /// Classifies an Itanium-mangled `PerformanceCounterNameToString{0..N}` symbol into its [`StaticCsrFamily`].
    ///
    /// Returns `None` for non-`PerformanceCounterNameToString` symbols, top-level dispatchers
    /// lacking a numeric shard suffix before `ENS`, or unrecognized namespaces.
    ///
    /// # Arguments
    ///
    /// * `sym_name` - Mangled C++ symbol name from `.symtab`.
    #[must_use]
    pub fn from_mangled_shard_symbol(sym_name: &str) -> Option<Self> {
        if !sym_name.contains("PerformanceCounterNameToString") {
            return None;
        }
        let ens_pos = sym_name.find("ENS")?;
        if !sym_name
            .as_bytes()
            .get(ens_pos.wrapping_sub(1))
            .is_some_and(u8::is_ascii_digit)
        {
            return None;
        }
        if sym_name.contains("3jxc") {
            Some(Self::Jxc)
        } else if sym_name.contains("3pfc") {
            Some(Self::Pfc)
        } else if sym_name.contains("3plc") {
            Some(Self::Plc)
        } else if sym_name.contains("3vlc") {
            Some(Self::Vlc)
        } else if sym_name.contains("3vfc") {
            Some(Self::Vfc)
        } else if sym_name.contains("3glc") {
            Some(Self::Glc)
        } else if sym_name.contains("3gfc") {
            Some(Self::Gfc)
        } else {
            None
        }
    }

    /// Returns the `PCIe` BAR aperture index (`0` or `2`) for `byte_offset` on this ASIC family.
    ///
    /// # Arguments
    ///
    /// * `byte_offset` - Physical CSR byte offset within the BAR window.
    #[must_use]
    pub const fn default_bar_index(self, byte_offset: u64) -> u8 {
        match self {
            Self::Jxc | Self::Pfc | Self::Plc => 2,
            Self::Vlc | Self::Vfc | Self::Glc | Self::Gfc => {
                if byte_offset < 0x8000_0000 {
                    0
                } else {
                    2
                }
            }
        }
    }
}

/// Decoded architectural bitfield inside an [`ExtractedHardwareCsrEntry`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractedCsrBitfield {
    /// Bitfield mnemonic identifier (e.g., `"run_status"`, `"pc"`, `"tracemark"`).
    pub mnemonic: String,
    /// Least-significant bit index (`0..63`).
    pub lsb: u32,
    /// Most-significant bit index (`lsb..63`).
    pub msb: u32,
    /// Software access mode string (`"RO"`, `"RW"`, `"W1C"`, `"WO"`).
    pub access: String,
    /// Hardware power-on reset value of this bitfield.
    pub reset_value: u64,
}

/// Complete ground-truth hardware CSR specification extracted from `libtpu.so` Brotli or `.lrodata` tables.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractedHardwareCsrEntry {
    /// Hierarchical GCSR or `asic_sw` register name.
    pub full_path: String,
    /// `PCIe` BAR aperture index (`0` = BAR0 VF/Gasket aperture, `2` = BAR2 `DeepSea` CSR aperture).
    pub bar_index: u8,
    /// Physical MMIO byte offset within `bar_index`.
    pub byte_offset: u64,
    /// Register width in bits (`32` or `64`).
    pub width_bits: u32,
    /// Software access mode (`"RO"`, `"RW"`, `"W1C"`, `"WO"`).
    pub access: String,
    /// Composite 64-bit hardware power-on reset value.
    pub reset_value: u64,
    /// Ordered bitfield specifications inside this register.
    pub bitfields: Vec<ExtractedCsrBitfield>,
}

/// Maps a `gcsr.proto` `SWAccessType` enum integer (or `0` when unspecified) and `reg_name`
/// to a canonical access mode code (`"RO"`, `"RW"`, `"W1C"`, `"WO"`).
///
/// # Arguments
///
/// * `swaccess` - `GcsrSwAccessType` integer discriminant from `RegField`.
/// * `reg_name` - Register or bitfield name used when `swaccess == 0`.
#[must_use]
pub fn gcsr_swaccess_to_code(swaccess: i32, reg_name: &str) -> &'static str {
    match swaccess {
        1 | 3 | 4 => "RO",
        2 | 5..=10 => "RW",
        11..=20 => "W1C",
        21..=25 => "WO",
        _ => {
            let upper = reg_name.to_ascii_uppercase();
            if upper.contains("CORE_RESET") || upper.contains("DOORBELL") {
                "WO"
            } else if upper.contains("FAULT")
                || upper.contains("PARITY")
                || upper.contains("ERROR_LOG")
                || upper.contains("INTERRUPT_STATUS")
            {
                "W1C"
            } else if upper.contains("STATUS")
                || upper.contains("STATE")
                || upper.contains("COUNTER")
                || upper.ends_with("_COUNT")
                || upper.contains("OCCUPANCY")
                || upper.contains("TIMESTAMP")
            {
                "RO"
            } else {
                "RW"
            }
        }
    }
}

/// Decompresses a Brotli `*_pf_csrs.br` blob and decodes full [`ExtractedHardwareCsrEntry`]
/// descriptors including `gcsr.proto` `RegField` bitfields, access modes, BAR apertures, and reset values.
///
/// # Arguments
///
/// * `compressed_br` - Raw Brotli-compressed `RegisterTranslatorData` bytes from `filewrapper_toc`.
///
/// # Errors
///
/// Returns [`LibTpuError`] if Brotli decompression or protobuf wire decoding fails.
pub fn decode_encoded_gcsr_brotli_detailed(
    compressed_br: &[u8],
) -> LibTpuResult<Vec<ExtractedHardwareCsrEntry>> {
    let decompressed = decompress_brotli_csr_table(compressed_br)?;
    let reg_data = RegisterTranslatorData::decode_proto(&decompressed)?;
    let mut entries = Vec::new();

    for node in &reg_data.nodes {
        let active_regs: &[EncodedRegister] = if !node.registers.is_empty() {
            &node.registers
        } else if let Ok(layout_idx) = usize::try_from(node.layout_index)
            && let Some(template) = reg_data.node_layout_table.get(layout_idx)
        {
            &template.registers
        } else {
            &[]
        };

        for reg in active_regs {
            let full_path = format!("{}.{}", node.mnemonic, reg.name);
            let byte_offset = node.base_offset.saturating_add(reg.offset);
            let bar_index = if byte_offset < 0x8000_0000 { 0 } else { 2 };
            let default_access = gcsr_swaccess_to_code(0, &full_path);

            let raw_meta: &[u8] = if !reg.register_data.is_empty() {
                &reg.register_data
            } else if let Ok(data_idx) = usize::try_from(reg.register_data_index)
                && let Some(shared_meta) = reg_data.register_data_table.get(data_idx)
            {
                shared_meta
            } else {
                &[]
            };

            let mut width_bits = u32::try_from(reg.width)
                .ok()
                .filter(|&w| w > 0)
                .map_or(64, |w| w.saturating_mul(8));
            let mut bitfields = Vec::new();
            let mut reset_value = 0_u64;
            let mut reg_access = String::from(default_access);

            if !raw_meta.is_empty() {
                let gcsr_reg = GcsrRegister::decode_proto(raw_meta)?;
                if let Some(gw) = gcsr_reg.width.filter(|&w| w > 0) {
                    width_bits = gw;
                }
                for rf in gcsr_reg.field {
                    let bf_acc = gcsr_swaccess_to_code(rf.swaccess.unwrap_or(0), &full_path);
                    if bitfields.is_empty() {
                        reg_access = String::from(bf_acc);
                    }
                    let lsb = rf.lsb.unwrap_or(0);
                    let msb = rf.msb.unwrap_or(lsb);
                    let init_val = rf.init.unwrap_or(0);
                    let bit_span = msb.saturating_sub(lsb).saturating_add(1);
                    let mask = 1_u64
                        .checked_shl(bit_span)
                        .map_or(u64::MAX, |v| v.wrapping_sub(1));
                    reset_value |= (init_val & mask).checked_shl(lsb).unwrap_or(0);
                    bitfields.push(ExtractedCsrBitfield {
                        mnemonic: rf.mnemonic.unwrap_or_default(),
                        lsb,
                        msb,
                        access: String::from(bf_acc),
                        reset_value: init_val,
                    });
                }
            }

            if bitfields.is_empty() {
                bitfields.push(ExtractedCsrBitfield {
                    mnemonic: String::from("value"),
                    lsb: 0,
                    msb: width_bits.saturating_sub(1),
                    access: reg_access.clone(),
                    reset_value: 0,
                });
            }

            entries.push(ExtractedHardwareCsrEntry {
                full_path,
                bar_index,
                byte_offset,
                width_bits,
                access: reg_access,
                reset_value,
                bitfields,
            });
        }
    }

    Ok(entries)
}

/// Decompresses a Brotli `*_error_collector_error_sources.br` blob and decodes its
/// `asic_sw.generators.errors.ErrorSources` hardware fault and SRAM/HBM ECC CSR map.
///
/// # Arguments
///
/// * `compressed_br` - Raw Brotli-compressed `ErrorSources` bytes from `filewrapper_toc`.
///
/// # Errors
///
/// Returns [`LibTpuError`] if Brotli decompression or protobuf wire decoding fails.
pub fn decode_brotli_error_sources(compressed_br: &[u8]) -> LibTpuResult<ErrorSources> {
    let decompressed = decompress_brotli_csr_table(compressed_br)?;
    ErrorSources::decode_proto(&decompressed)
}

/// Decodes a `*_model_parameters.binarypb` (`jfc`, `dfc`, `pfc`) cost-model blob from `filewrapper_toc`.
///
/// # Arguments
///
/// * `binarypb` - Raw wire-encoded `BarnaCoreModelParameters` bytes.
///
/// # Errors
///
/// Returns [`LibTpuError`] if protobuf wire decoding fails.
pub fn decode_barna_core_model_parameters(
    binarypb: &[u8],
) -> LibTpuResult<BarnaCoreModelParameters> {
    BarnaCoreModelParameters::decode_proto(binarypb)
}

/// Decodes a `barna_core_constant_param_*.binarypb` (`jfc`, `dfc`, `pfc`) cost-model blob from `filewrapper_toc`.
///
/// # Arguments
///
/// * `binarypb` - Raw wire-encoded `BarnaCoreConstantParameters` bytes.
///
/// # Errors
///
/// Returns [`LibTpuError`] if protobuf wire decoding fails.
pub fn decode_barna_core_constant_parameters(
    binarypb: &[u8],
) -> LibTpuResult<BarnaCoreConstantParameters> {
    BarnaCoreConstantParameters::decode_proto(binarypb)
}

/// Reads an unaligned little-endian `u16` at `offset` in `bytes` (`0` if out of bounds).
///
/// # Arguments
///
/// * `bytes` - Source byte slice.
/// * `offset` - Byte offset of the 2-byte integer.
#[inline]
fn read_le_u16(bytes: &[u8], offset: usize) -> u16 {
    bytes
        .get(offset..offset.saturating_add(2))
        .map_or(0, |s| u16::from_le(bytemuck::pod_read_unaligned::<u16>(s)))
}

/// Reads an unaligned little-endian `u32` at `offset` in `bytes` (`0` if out of bounds).
///
/// # Arguments
///
/// * `bytes` - Source byte slice.
/// * `offset` - Byte offset of the 4-byte integer.
#[inline]
fn read_le_u32(bytes: &[u8], offset: usize) -> u32 {
    bytes
        .get(offset..offset.saturating_add(4))
        .map_or(0, |s| u32::from_le(bytemuck::pod_read_unaligned::<u32>(s)))
}

/// Reads an unaligned little-endian `i32` at `offset` in `bytes` (`0` if out of bounds).
///
/// # Arguments
///
/// * `bytes` - Source byte slice.
/// * `offset` - Byte offset of the 4-byte signed integer.
#[inline]
fn read_le_i32(bytes: &[u8], offset: usize) -> i32 {
    bytes
        .get(offset..offset.saturating_add(4))
        .map_or(0, |s| i32::from_le(bytemuck::pod_read_unaligned::<i32>(s)))
}

/// Reads an unaligned little-endian `u64` at `offset` in `bytes` (`0` if out of bounds).
///
/// # Arguments
///
/// * `bytes` - Source byte slice.
/// * `offset` - Byte offset of the 8-byte integer.
#[inline]
fn read_le_u64(bytes: &[u8], offset: usize) -> u64 {
    bytes
        .get(offset..offset.saturating_add(8))
        .map_or(0, |s| u64::from_le(bytemuck::pod_read_unaligned::<u64>(s)))
}

/// Reads an unaligned little-endian `i64` at `offset` in `bytes` (`0` if out of bounds).
///
/// # Arguments
///
/// * `bytes` - Source byte slice.
/// * `offset` - Byte offset of the 8-byte signed integer.
#[inline]
fn read_le_i64(bytes: &[u8], offset: usize) -> i64 {
    bytes
        .get(offset..offset.saturating_add(8))
        .map_or(0, |s| i64::from_le(bytemuck::pod_read_unaligned::<i64>(s)))
}

/// Decoded RIP-relative / GOT-relative table pointers inside a `PerformanceCounterNameToString` shard.
#[derive(Debug, Default, Clone, Copy)]
struct ShardTablePointers {
    /// Virtual address of `kCounterRanges` (`24`-byte descriptors), if multi-range.
    ranges_va: Option<u64>,
    /// Number of `24`-byte `CounterRange` entries.
    num_ranges: usize,
    /// Virtual address of `kRelStrOffsets` (`u16[]`).
    rel_str_va: Option<u64>,
    /// Virtual address of the concatenated CSR string pool.
    pool_va: Option<u64>,
    /// Base CSR byte offset for single-range shards.
    single_base: Option<u64>,
    /// Number of 8-byte CSRs in a single-range shard.
    single_count: usize,
}

/// Scans x86-64 machine `code` at `fn_vaddr` for `kCounterRanges`, `kRelStrOffsets`, and string-pool pointers.
///
/// # Arguments
///
/// * `fn_vaddr` - Function start virtual address (`st_value`).
/// * `code` - Function machine-code byte slice.
fn scan_shard_table_pointers(fn_vaddr: u64, code: &[u8]) -> ShardTablePointers {
    let mut got_va = 0_u64;
    let mut ptrs = ShardTablePointers::default();
    let mut i = 0_usize;
    while i.saturating_add(12) <= code.len() {
        let cur_va = fn_vaddr.saturating_add(i as u64);
        let pfx3 = code.get(i..i + 3);
        let pfx2 = code.get(i..i + 2);
        if pfx3 == Some(&[0x48, 0x8d, 0x35]) {
            let disp = i64::from(read_le_i32(code, i + 3));
            got_va = cur_va.saturating_add(7).wrapping_add_signed(disp);
        } else if pfx3 == Some(&[0x4c, 0x8d, 0x05]) {
            let disp = i64::from(read_le_i32(code, i + 3));
            ptrs.ranges_va = Some(cur_va.saturating_add(7).wrapping_add_signed(disp));
            if code.get(i + 7) == Some(&0xba) {
                ptrs.num_ranges = usize::try_from(read_le_u32(code, i + 8)).unwrap_or(0);
            }
        } else if pfx3 == Some(&[0x48, 0x8d, 0x15]) {
            let disp = i64::from(read_le_i32(code, i + 3));
            ptrs.rel_str_va = Some(cur_va.saturating_add(7).wrapping_add_signed(disp));
        } else if pfx2 == Some(&[0x48, 0xba]) {
            ptrs.rel_str_va = Some(got_va.wrapping_add_signed(read_le_i64(code, i + 2)));
        } else if pfx2 == Some(&[0x48, 0xb9]) || pfx2 == Some(&[0x48, 0xb8]) {
            ptrs.pool_va = Some(got_va.wrapping_add_signed(read_le_i64(code, i + 2)));
        } else if pfx3 == Some(&[0x48, 0x8d, 0x87]) {
            let neg_disp = i64::from(read_le_i32(code, i + 3));
            ptrs.single_base = u64::try_from(neg_disp.wrapping_neg()).ok();
        } else if pfx2 == Some(&[0x48, 0x3d]) && ptrs.single_count == 0 {
            let span_plus_1 = read_le_u32(code, i + 2);
            ptrs.single_count = usize::try_from(span_plus_1.saturating_sub(1) / 8 + 1).unwrap_or(0);
        }
        i += 1;
    }
    ptrs
}

/// Decodes a single `PerformanceCounterNameToString` x86-64 function shard into `out_entries`.
///
/// # Arguments
///
/// * `elf_bytes` - Complete ELF64 file byte slice.
/// * `spans` - Allocated section spans for virtual-address translation.
/// * `family` - Target [`StaticCsrFamily`].
/// * `fn_vaddr` - Function start virtual address (`st_value`).
/// * `code` - Function machine-code byte slice.
/// * `out_entries` - Output vector receiving decoded [`ExtractedHardwareCsrEntry`] records.
fn decode_static_csr_shard_x86_64(
    elf_bytes: &[u8],
    spans: &[AllocSectionSpan],
    family: StaticCsrFamily,
    fn_vaddr: u64,
    code: &[u8],
    out_entries: &mut Vec<ExtractedHardwareCsrEntry>,
) {
    let ptrs = scan_shard_table_pointers(fn_vaddr, code);
    let push_csr = |out: &mut Vec<ExtractedHardwareCsrEntry>, csr_off: u64, cname: String| {
        let access = String::from(gcsr_swaccess_to_code(0, &cname));
        let bar_index = family.default_bar_index(csr_off);
        out.push(ExtractedHardwareCsrEntry {
            full_path: cname,
            bar_index,
            byte_offset: csr_off,
            width_bits: 64,
            access: access.clone(),
            reset_value: 0,
            bitfields: vec![ExtractedCsrBitfield {
                mnemonic: String::from("value"),
                lsb: 0,
                msb: 63,
                access,
                reset_value: 0,
            }],
        });
    };

    let rel_off_opt = ptrs
        .rel_str_va
        .and_then(|va| vaddr_to_file_offset(spans, va));
    let pool_off_opt = ptrs.pool_va.and_then(|va| vaddr_to_file_offset(spans, va));

    if let (Some(ranges_file_off), Some(rel_file_off), Some(pool_file_off)) = (
        ptrs.ranges_va
            .and_then(|va| vaddr_to_file_offset(spans, va)),
        rel_off_opt,
        pool_off_opt,
    ) {
        for r_idx in 0..ptrs.num_ranges {
            let rec_off = ranges_file_off.saturating_add(r_idx.saturating_mul(24));
            let base_off = read_le_u64(elf_bytes, rec_off);
            let stride = u64::from(read_le_u32(elf_bytes, rec_off.saturating_add(8)));
            let str_base =
                usize::try_from(read_le_u32(elf_bytes, rec_off.saturating_add(12))).unwrap_or(0);
            let cnt = usize::from(read_le_u16(elf_bytes, rec_off.saturating_add(16)));
            let rel_idx = usize::from(read_le_u16(elf_bytes, rec_off.saturating_add(18)));

            for k in 0..cnt {
                let u16_off =
                    rel_file_off.saturating_add(rel_idx.saturating_add(k).saturating_mul(2));
                let s0 = usize::from(read_le_u16(elf_bytes, u16_off));
                let s1 = usize::from(read_le_u16(elf_bytes, u16_off.saturating_add(2)));
                let start = pool_file_off.saturating_add(str_base).saturating_add(s0);
                let end = pool_file_off.saturating_add(str_base).saturating_add(s1);
                if let Some(name_slice) = elf_bytes.get(start..end) {
                    let cname = String::from_utf8_lossy(name_slice).into_owned();
                    push_csr(
                        out_entries,
                        base_off.saturating_add((k as u64).saturating_mul(stride)),
                        cname,
                    );
                }
            }
        }
    } else if let (Some(base_off), Some(rel_file_off), Some(pool_file_off)) =
        (ptrs.single_base, rel_off_opt, pool_off_opt)
    {
        for k in 0..ptrs.single_count {
            let u16_off = rel_file_off.saturating_add(k.saturating_mul(2));
            let s0 = usize::from(read_le_u16(elf_bytes, u16_off));
            let s1 = usize::from(read_le_u16(elf_bytes, u16_off.saturating_add(2)));
            let start = pool_file_off.saturating_add(s0);
            let end = pool_file_off.saturating_add(s1);
            if let Some(name_slice) = elf_bytes.get(start..end) {
                let cname = String::from_utf8_lossy(name_slice).into_owned();
                push_csr(
                    out_entries,
                    base_off.saturating_add((k as u64).saturating_mul(8)),
                    cname,
                );
            }
        }
    }
}

/// Extracts all static `.lrodata` / `.rodata` hardware CSR register maps across `jxc` (v2/v3),
/// `pfc` (v4), `plc` (v4lite), `vlc` (v5e), `vfc` (v5p), `glc` (v6e), and `gfc` (v7x) from an ELF64 `libtpu.so`.
///
/// # Arguments
///
/// * `elf_bytes` - Complete ELF64 binary image of `libtpu.so`.
///
/// # Errors
///
/// Returns [`LibTpuError::BinaryFormatError`] if `elf_bytes` is not a valid ELF64 object.
pub fn extract_elf64_static_csr_maps(
    elf_bytes: &[u8],
) -> LibTpuResult<BTreeMap<StaticCsrFamily, Vec<ExtractedHardwareCsrEntry>>> {
    let elf = Elf::parse(elf_bytes).map_err(|err| LibTpuError::BinaryFormatError {
        reason: format!("ELF64 static CSR parse failed: {err}"),
    })?;

    let spans = collect_alloc_spans(&elf);

    let mut shards_by_family: BTreeMap<StaticCsrFamily, Vec<(u64, usize)>> = BTreeMap::new();
    for sym in &elf.syms {
        if sym.is_function()
            && sym.st_value > 0
            && sym.st_size > 0
            && let Some(sym_name) = elf.strtab.get_at(sym.st_name)
            && let Some(family) = StaticCsrFamily::from_mangled_shard_symbol(sym_name)
        {
            shards_by_family
                .entry(family)
                .or_default()
                .push((sym.st_value, usize::try_from(sym.st_size).unwrap_or(0)));
        }
    }

    let mut result: BTreeMap<StaticCsrFamily, Vec<ExtractedHardwareCsrEntry>> = BTreeMap::new();
    for (family, mut shards) in shards_by_family {
        shards.sort_unstable_by_key(|&(va, _)| va);
        let mut entries = Vec::new();
        for (fn_va, fn_sz) in shards {
            if let Some(code) = vaddr_to_file_offset(&spans, fn_va)
                .and_then(|off| elf_bytes.get(off..off.saturating_add(fn_sz)))
            {
                decode_static_csr_shard_x86_64(
                    elf_bytes,
                    &spans,
                    family,
                    fn_va,
                    code,
                    &mut entries,
                );
            }
        }
        entries.sort_by(|a, b| (a.byte_offset, &a.full_path).cmp(&(b.byte_offset, &b.full_path)));
        result.insert(family, entries);
    }

    Ok(result)
}

#[cfg(test)]
#[allow(clippy::cognitive_complexity, clippy::too_many_lines)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::elf::carver::verify_filewrapper_toc_md5;
    use std::io::Write as _;

    /// Builds a minimal synthetic 64-bit ELF shared object containing `.rela.dyn`,
    /// `.rodata`, `.data.rel.ro`, `protodesc_cold`, and `filewrapper_toc` for unit testing.
    ///
    /// # Arguments
    ///
    /// * `fd_bytes` - Wire-encoded `FileDescriptorProto` payload.
    /// * `toc_name` - Raw C-string bytes for the `FileToc` resource name.
    /// * `toc_payload` - Raw binary payload for the `FileToc` resource.
    /// * `corrupt_toc_md5` - Whether to flip a bit in the expected MD5 digest.
    fn build_synthetic_forensics_elf64(
        fd_bytes: &[u8],
        toc_name: &[u8],
        toc_payload: &[u8],
        corrupt_toc_md5: bool,
    ) -> Vec<u8> {
        let non_proto_fd = FileDescriptorProto {
            name: Some(String::from("ignored.txt")),
            ..FileDescriptorProto::default()
        }
        .encode_to_vec();
        build_one_pt_load_elf(
            fd_bytes,
            &non_proto_fd,
            toc_name,
            toc_payload,
            corrupt_toc_md5,
        )
    }

    /// Builds a 1:1 `PT_LOAD` mapped ELF64 (`vaddr = 0x100000 + file_offset`) with `PT_DYNAMIC`.
    ///
    /// # Arguments
    ///
    /// * `fd_bytes` - Valid `.proto` wire-encoded `FileDescriptorProto` slice.
    /// * `non_proto_fd` - Non-`.proto` wire-encoded `FileDescriptorProto` slice.
    /// * `toc_name` - Resource name bytes for `filewrapper_toc`.
    /// * `toc_payload` - Resource data bytes for `filewrapper_toc`.
    /// * `corrupt_toc_md5` - Whether to intentionally corrupt the `FileToc` MD5 checksum.
    fn build_one_pt_load_elf(
        fd_bytes: &[u8],
        non_proto_fd: &[u8],
        toc_name: &[u8],
        toc_payload: &[u8],
        corrupt_toc_md5: bool,
    ) -> Vec<u8> {
        let base_va = 0x10_0000_u64;
        let shstrtab =
            b"\0.shstrtab\0.dynstr\0.dynsym\0.rela.dyn\0.rodata\0.data.rel.ro\0protodesc_cold\0filewrapper_toc\0";
        let ehdr_size = 64_usize;
        let phdr_size = 56_usize;
        let num_phdrs = 2_usize;
        let header_bytes = ehdr_size + num_phdrs * phdr_size;

        let mut buf = vec![0_u8; header_bytes];
        let pad8 = |b: &mut Vec<u8>| {
            while !b.len().is_multiple_of(8) {
                b.push(0);
            }
        };

        let shstr_off = buf.len();
        buf.extend_from_slice(shstrtab);
        pad8(&mut buf);

        let dynstr_off = buf.len();
        buf.push(0);
        pad8(&mut buf);

        let dynsym_off = buf.len();
        buf.extend_from_slice(&[0_u8; 24]);
        pad8(&mut buf);

        let rodata_off = buf.len();
        let name_vaddr = base_va + (buf.len() as u64);
        buf.extend_from_slice(toc_name);
        buf.push(0);
        let payload_vaddr = base_va + (buf.len() as u64);
        buf.extend_from_slice(toc_payload);
        let empty_cstr_vaddr = base_va + (buf.len() as u64);
        buf.push(0);
        // Unterminated bytes at end of buffer check
        pad8(&mut buf);

        let pdesc_off = buf.len();
        let fd_vaddr = base_va + (buf.len() as u64);
        buf.extend_from_slice(fd_bytes);
        let non_proto_vaddr = base_va + (buf.len() as u64);
        buf.extend_from_slice(non_proto_fd);
        let bad_wire_vaddr = base_va + (buf.len() as u64);
        buf.extend_from_slice(&[0xFF; 12]);
        let pdesc_len = buf.len() - pdesc_off;
        pad8(&mut buf);

        let drel_off = buf.len();
        let tbl0_vaddr = base_va + (buf.len() as u64);
        buf.extend_from_slice(&0_u32.to_le_bytes());
        buf.extend_from_slice(&u32::try_from(fd_bytes.len()).unwrap_or(0).to_le_bytes());
        buf.extend_from_slice(&[0_u8; 24]);

        let tbl1_vaddr = base_va + (buf.len() as u64);
        buf.extend_from_slice(&0_u32.to_le_bytes());
        buf.extend_from_slice(&u32::try_from(non_proto_fd.len()).unwrap_or(0).to_le_bytes());
        buf.extend_from_slice(&[0_u8; 24]);

        let tbl2_vaddr = base_va + (buf.len() as u64);
        buf.extend_from_slice(&1_u32.to_le_bytes());
        buf.extend_from_slice(&16_u32.to_le_bytes());
        buf.extend_from_slice(&[0_u8; 24]);

        let tbl3_vaddr = base_va + (buf.len() as u64);
        buf.extend_from_slice(&0_u32.to_le_bytes());
        buf.extend_from_slice(&12_u32.to_le_bytes());
        buf.extend_from_slice(&[0_u8; 24]);

        let tbl4_vaddr = base_va + (buf.len() as u64);
        buf.extend_from_slice(&0_u32.to_le_bytes());
        buf.extend_from_slice(&9_999_999_u32.to_le_bytes());
        buf.extend_from_slice(&[0_u8; 24]);

        let tbl5_vaddr = base_va + (buf.len() as u64);
        buf.extend_from_slice(&0_u32.to_le_bytes());
        buf.extend_from_slice(&4_u32.to_le_bytes());
        buf.extend_from_slice(&[0_u8; 24]);

        let toc_a_vaddr = base_va + (buf.len() as u64);
        let mut md5 = compute_rfc1321_md5(toc_payload);
        if corrupt_toc_md5 && let Some(first_byte) = md5.first_mut() {
            *first_byte ^= 0xFF;
        }
        buf.extend_from_slice(&0_u64.to_le_bytes());
        buf.extend_from_slice(&0_u64.to_le_bytes());
        buf.extend_from_slice(&(toc_payload.len() as u64).to_le_bytes());
        buf.extend_from_slice(&md5);

        let toc_empty_vaddr = base_va + (buf.len() as u64);
        buf.extend_from_slice(&0_u64.to_le_bytes());
        buf.extend_from_slice(&0_u64.to_le_bytes());
        buf.extend_from_slice(&1_u64.to_le_bytes());
        buf.extend_from_slice(&[0_u8; 16]);

        let toc_baddata_vaddr = base_va + (buf.len() as u64);
        buf.extend_from_slice(&0_u64.to_le_bytes());
        buf.extend_from_slice(&0_u64.to_le_bytes());
        buf.extend_from_slice(&1_u64.to_le_bytes());
        buf.extend_from_slice(&[0_u8; 16]);

        let toc_oversize_vaddr = base_va + (buf.len() as u64);
        buf.extend_from_slice(&0_u64.to_le_bytes());
        buf.extend_from_slice(&0_u64.to_le_bytes());
        buf.extend_from_slice(&9_999_999_u64.to_le_bytes());
        buf.extend_from_slice(&[0_u8; 16]);

        let toc_zeromd5_vaddr = base_va + (buf.len() as u64);
        buf.extend_from_slice(&0_u64.to_le_bytes());
        buf.extend_from_slice(&0_u64.to_le_bytes());
        buf.extend_from_slice(&(toc_payload.len() as u64).to_le_bytes());
        buf.extend_from_slice(&[0_u8; 16]);

        let toc_nodata_vaddr = base_va + (buf.len() as u64);
        buf.extend_from_slice(&[0_u8; 40]);

        let toc_single_vaddr = base_va + (buf.len() as u64);
        buf.extend_from_slice(&0_u64.to_le_bytes());
        buf.extend_from_slice(&0_u64.to_le_bytes());
        buf.extend_from_slice(&(toc_payload.len() as u64).to_le_bytes());
        buf.extend_from_slice(&[0_u8; 16]);
        buf.extend_from_slice(&[0_u8; 40]);
        let drel_len = buf.len() - drel_off;
        pad8(&mut buf);

        let ftoc_off = buf.len();
        let ftoc_vaddr = base_va + (buf.len() as u64);
        buf.extend_from_slice(&[0_u8; 32]);
        let ftoc_len = 32_usize;
        pad8(&mut buf);

        let push_rela = |b: &mut Vec<u8>, r_off: u64, r_add: i64| {
            b.extend_from_slice(&r_off.to_le_bytes());
            b.extend_from_slice(&8_u64.to_le_bytes());
            b.extend_from_slice(&r_add.to_le_bytes());
        };
        let rela_off = buf.len();
        let rela_vaddr = base_va + (buf.len() as u64);
        let mut rela_bytes = Vec::new();
        push_rela(&mut rela_bytes, tbl0_vaddr + 8, -1);
        push_rela(&mut rela_bytes, tbl0_vaddr + 8, 0x100);
        push_rela(&mut rela_bytes, 4, i64::try_from(fd_vaddr).unwrap_or(0));
        push_rela(
            &mut rela_bytes,
            0x999_008,
            i64::try_from(fd_vaddr).unwrap_or(0),
        );
        push_rela(
            &mut rela_bytes,
            tbl0_vaddr + 8,
            i64::try_from(fd_vaddr).unwrap_or(0),
        );
        push_rela(
            &mut rela_bytes,
            tbl1_vaddr + 8,
            i64::try_from(non_proto_vaddr).unwrap_or(0),
        );
        push_rela(
            &mut rela_bytes,
            tbl2_vaddr + 8,
            i64::try_from(fd_vaddr).unwrap_or(0),
        );
        push_rela(
            &mut rela_bytes,
            tbl3_vaddr + 8,
            i64::try_from(bad_wire_vaddr).unwrap_or(0),
        );
        push_rela(
            &mut rela_bytes,
            tbl4_vaddr + 8,
            i64::try_from(fd_vaddr).unwrap_or(0),
        );
        push_rela(
            &mut rela_bytes,
            tbl5_vaddr + 8,
            i64::try_from(fd_vaddr).unwrap_or(0),
        );

        push_rela(
            &mut rela_bytes,
            ftoc_vaddr,
            i64::try_from(toc_a_vaddr).unwrap_or(0),
        );
        push_rela(&mut rela_bytes, ftoc_vaddr + 16, 0x999_000);
        push_rela(
            &mut rela_bytes,
            ftoc_vaddr + 24,
            i64::try_from(toc_single_vaddr).unwrap_or(0),
        );
        push_rela(
            &mut rela_bytes,
            0x999_000,
            i64::try_from(name_vaddr).unwrap_or(0),
        );
        push_rela(
            &mut rela_bytes,
            0x999_008,
            i64::try_from(payload_vaddr).unwrap_or(0),
        );

        push_rela(
            &mut rela_bytes,
            toc_a_vaddr,
            i64::try_from(name_vaddr).unwrap_or(0),
        );
        push_rela(
            &mut rela_bytes,
            toc_a_vaddr + 8,
            i64::try_from(payload_vaddr).unwrap_or(0),
        );
        push_rela(
            &mut rela_bytes,
            toc_empty_vaddr,
            i64::try_from(empty_cstr_vaddr).unwrap_or(0),
        );
        push_rela(
            &mut rela_bytes,
            toc_empty_vaddr + 8,
            i64::try_from(payload_vaddr).unwrap_or(0),
        );
        push_rela(
            &mut rela_bytes,
            toc_baddata_vaddr,
            i64::try_from(name_vaddr).unwrap_or(0),
        );
        push_rela(&mut rela_bytes, toc_baddata_vaddr + 8, 0x888_000);
        push_rela(
            &mut rela_bytes,
            toc_oversize_vaddr,
            i64::try_from(name_vaddr).unwrap_or(0),
        );
        push_rela(
            &mut rela_bytes,
            toc_oversize_vaddr + 8,
            i64::try_from(payload_vaddr).unwrap_or(0),
        );
        push_rela(
            &mut rela_bytes,
            toc_zeromd5_vaddr,
            i64::try_from(name_vaddr).unwrap_or(0),
        );
        push_rela(
            &mut rela_bytes,
            toc_zeromd5_vaddr + 8,
            i64::try_from(payload_vaddr).unwrap_or(0),
        );
        push_rela(
            &mut rela_bytes,
            toc_nodata_vaddr,
            i64::try_from(name_vaddr).unwrap_or(0),
        );
        push_rela(
            &mut rela_bytes,
            toc_single_vaddr,
            i64::try_from(name_vaddr).unwrap_or(0),
        );
        push_rela(
            &mut rela_bytes,
            toc_single_vaddr + 8,
            i64::try_from(payload_vaddr).unwrap_or(0),
        );

        buf.extend_from_slice(&rela_bytes);
        pad8(&mut buf);

        let dyn_off = buf.len();
        let dyn_vaddr = base_va + (buf.len() as u64);
        let mut dyn_bytes = Vec::new();
        for (tag, val) in [
            (5_u64, base_va + (dynstr_off as u64)),
            (10_u64, 1_u64),
            (6_u64, base_va + (dynsym_off as u64)),
            (11_u64, 24_u64),
            (7_u64, rela_vaddr),
            (8_u64, rela_bytes.len() as u64),
            (9_u64, 24_u64),
            (0_u64, 0_u64),
        ] {
            dyn_bytes.extend_from_slice(&tag.to_le_bytes());
            dyn_bytes.extend_from_slice(&val.to_le_bytes());
        }
        buf.extend_from_slice(&dyn_bytes);
        pad8(&mut buf);

        let load_filesz = buf.len();
        let shoff = buf.len();

        let pack_shdr = |name_idx: u32,
                         sh_type: u32,
                         flags: u64,
                         addr: u64,
                         off: usize,
                         sz: usize,
                         link: u32,
                         entsz: u64| {
            let mut v = Vec::with_capacity(64);
            v.extend_from_slice(&name_idx.to_le_bytes());
            v.extend_from_slice(&sh_type.to_le_bytes());
            v.extend_from_slice(&flags.to_le_bytes());
            v.extend_from_slice(&addr.to_le_bytes());
            v.extend_from_slice(&(off as u64).to_le_bytes());
            v.extend_from_slice(&(sz as u64).to_le_bytes());
            v.extend_from_slice(&link.to_le_bytes());
            v.extend_from_slice(&0_u32.to_le_bytes());
            v.extend_from_slice(&8_u64.to_le_bytes());
            v.extend_from_slice(&entsz.to_le_bytes());
            v
        };

        buf.extend_from_slice(&pack_shdr(0, 0, 0, 0, 0, 0, 0, 0));
        buf.extend_from_slice(&pack_shdr(1, 3, 0, 0, shstr_off, shstrtab.len(), 0, 0));
        buf.extend_from_slice(&pack_shdr(
            11,
            3,
            2,
            base_va + (dynstr_off as u64),
            dynstr_off,
            1,
            0,
            0,
        ));
        buf.extend_from_slice(&pack_shdr(
            19,
            11,
            2,
            base_va + (dynsym_off as u64),
            dynsym_off,
            24,
            2,
            24,
        ));
        buf.extend_from_slice(&pack_shdr(
            27,
            4,
            2,
            rela_vaddr,
            rela_off,
            rela_bytes.len(),
            3,
            24,
        ));
        buf.extend_from_slice(&pack_shdr(
            0,
            6,
            3,
            dyn_vaddr,
            dyn_off,
            dyn_bytes.len(),
            2,
            16,
        ));
        buf.extend_from_slice(&pack_shdr(
            37,
            1,
            2,
            base_va + (rodata_off as u64),
            rodata_off,
            pdesc_off - rodata_off,
            0,
            0,
        ));
        buf.extend_from_slice(&pack_shdr(
            45,
            1,
            3,
            base_va + (drel_off as u64),
            drel_off,
            drel_len,
            0,
            0,
        ));
        buf.extend_from_slice(&pack_shdr(
            58,
            1,
            2,
            base_va + (pdesc_off as u64),
            pdesc_off,
            pdesc_len,
            0,
            0,
        ));
        buf.extend_from_slice(&pack_shdr(73, 1, 3, ftoc_vaddr, ftoc_off, ftoc_len, 0, 0));
        buf.extend_from_slice(&pack_shdr(
            0,
            SHT_NOBITS,
            3,
            base_va + 0x80_0000,
            load_filesz,
            64,
            0,
            0,
        ));
        buf.extend_from_slice(&pack_shdr(
            0,
            1,
            2,
            base_va + 0x70_0000,
            load_filesz,
            0,
            0,
            0,
        ));

        let mut hdr = Vec::with_capacity(header_bytes);
        hdr.extend_from_slice(b"\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00");
        hdr.extend_from_slice(&3_u16.to_le_bytes());
        hdr.extend_from_slice(&62_u16.to_le_bytes());
        hdr.extend_from_slice(&1_u32.to_le_bytes());
        hdr.extend_from_slice(&0_u64.to_le_bytes());
        hdr.extend_from_slice(&(ehdr_size as u64).to_le_bytes());
        hdr.extend_from_slice(&(shoff as u64).to_le_bytes());
        hdr.extend_from_slice(&0_u32.to_le_bytes());
        hdr.extend_from_slice(&u16::try_from(ehdr_size).unwrap_or(0).to_le_bytes());
        hdr.extend_from_slice(&u16::try_from(phdr_size).unwrap_or(0).to_le_bytes());
        hdr.extend_from_slice(&u16::try_from(num_phdrs).unwrap_or(0).to_le_bytes());
        hdr.extend_from_slice(&64_u16.to_le_bytes());
        hdr.extend_from_slice(&12_u16.to_le_bytes());
        hdr.extend_from_slice(&1_u16.to_le_bytes());

        // PT_LOAD (type=1, flags=6) covering [0 .. load_filesz) at base_va
        hdr.extend_from_slice(&1_u32.to_le_bytes());
        hdr.extend_from_slice(&6_u32.to_le_bytes());
        hdr.extend_from_slice(&0_u64.to_le_bytes());
        hdr.extend_from_slice(&base_va.to_le_bytes());
        hdr.extend_from_slice(&base_va.to_le_bytes());
        hdr.extend_from_slice(&(load_filesz as u64).to_le_bytes());
        hdr.extend_from_slice(&(load_filesz as u64).to_le_bytes());
        hdr.extend_from_slice(&0x1000_u64.to_le_bytes());

        // PT_DYNAMIC (type=2, flags=6)
        hdr.extend_from_slice(&2_u32.to_le_bytes());
        hdr.extend_from_slice(&6_u32.to_le_bytes());
        hdr.extend_from_slice(&(dyn_off as u64).to_le_bytes());
        hdr.extend_from_slice(&dyn_vaddr.to_le_bytes());
        hdr.extend_from_slice(&dyn_vaddr.to_le_bytes());
        hdr.extend_from_slice(&(dyn_bytes.len() as u64).to_le_bytes());
        hdr.extend_from_slice(&(dyn_bytes.len() as u64).to_le_bytes());
        hdr.extend_from_slice(&8_u64.to_le_bytes());

        if let Some(dst) = buf.get_mut(..header_bytes) {
            dst.copy_from_slice(&hdr);
        }
        buf
    }

    #[test]
    fn test_descriptor_table_carver_and_filewrapper_toc_synthetic() -> LibTpuResult<()> {
        let sample_fd = FileDescriptorProto {
            name: Some(String::from("platforms/deepsea/6acc60406/test_isa.proto")),
            package: Some(String::from("asic_sw.6acc60406")),
            syntax: Some(String::from("editions")),
            dependency: vec![String::from("base_a.proto"), String::from("base_b.proto")],
            public_dependency: vec![0],
            enum_type: vec![prost_types::EnumDescriptorProto {
                name: Some(String::from("SlotKind")),
                value: vec![
                    prost_types::EnumValueDescriptorProto {
                        name: Some(String::from("SLOT_NONE")),
                        number: Some(0),
                        options: None,
                    },
                    prost_types::EnumValueDescriptorProto {
                        name: Some(String::from("SLOT_ALIAS")),
                        number: Some(0),
                        options: None,
                    },
                ],
                options: None,
                reserved_range: vec![],
                reserved_name: vec![],
            }],
            message_type: vec![prost_types::DescriptorProto {
                name: Some(String::from("BundleSpec")),
                field: vec![
                    prost_types::FieldDescriptorProto {
                        name: Some(String::from("attrs")),
                        number: Some(1),
                        label: Some(3),
                        r#type: Some(11),
                        type_name: Some(String::from(".asic_sw.6acc60406.BundleSpec.AttrsEntry")),
                        ..prost_types::FieldDescriptorProto::default()
                    },
                    prost_types::FieldDescriptorProto {
                        name: Some(String::from("choice_i32")),
                        number: Some(2),
                        label: Some(1),
                        r#type: Some(5),
                        oneof_index: Some(0),
                        ..prost_types::FieldDescriptorProto::default()
                    },
                    prost_types::FieldDescriptorProto {
                        name: Some(String::from("opt_flag")),
                        number: Some(3),
                        label: Some(1),
                        r#type: Some(8),
                        oneof_index: Some(1),
                        proto3_optional: Some(true),
                        ..prost_types::FieldDescriptorProto::default()
                    },
                    prost_types::FieldDescriptorProto {
                        name: Some(String::from("tags")),
                        number: Some(4),
                        label: Some(3),
                        r#type: Some(9),
                        ..prost_types::FieldDescriptorProto::default()
                    },
                    prost_types::FieldDescriptorProto {
                        name: Some(String::from("plain_u32")),
                        number: Some(5),
                        label: Some(1),
                        r#type: Some(13),
                        ..prost_types::FieldDescriptorProto::default()
                    },
                    prost_types::FieldDescriptorProto {
                        name: Some(String::from("editions_legacy_req")),
                        number: Some(6),
                        label: Some(2),
                        r#type: Some(13),
                        ..prost_types::FieldDescriptorProto::default()
                    },
                ],
                nested_type: vec![
                    prost_types::DescriptorProto {
                        name: Some(String::from("AttrsEntry")),
                        field: vec![
                            prost_types::FieldDescriptorProto {
                                name: Some(String::from("key")),
                                number: Some(1),
                                label: Some(1),
                                r#type: Some(9),
                                ..prost_types::FieldDescriptorProto::default()
                            },
                            prost_types::FieldDescriptorProto {
                                name: Some(String::from("value")),
                                number: Some(2),
                                label: Some(1),
                                r#type: Some(4),
                                ..prost_types::FieldDescriptorProto::default()
                            },
                        ],
                        options: Some(prost_types::MessageOptions {
                            map_entry: Some(true),
                            ..prost_types::MessageOptions::default()
                        }),
                        ..prost_types::DescriptorProto::default()
                    },
                    prost_types::DescriptorProto {
                        name: Some(String::from("NestedSub")),
                        ..prost_types::DescriptorProto::default()
                    },
                ],
                enum_type: vec![prost_types::EnumDescriptorProto {
                    name: Some(String::from("InnerMode")),
                    value: vec![prost_types::EnumValueDescriptorProto {
                        name: Some(String::from("MODE_A")),
                        number: Some(0),
                        options: None,
                    }],
                    options: None,
                    reserved_range: vec![],
                    reserved_name: vec![],
                }],
                oneof_decl: vec![
                    prost_types::OneofDescriptorProto {
                        name: Some(String::from("payload_kind")),
                        options: None,
                    },
                    prost_types::OneofDescriptorProto {
                        name: Some(String::from("_opt_flag")),
                        options: None,
                    },
                ],
                reserved_range: vec![
                    prost_types::descriptor_proto::ReservedRange {
                        start: Some(10),
                        end: Some(11),
                    },
                    prost_types::descriptor_proto::ReservedRange {
                        start: Some(20),
                        end: Some(25),
                    },
                ],
                reserved_name: vec![String::from("legacy_slot")],
                ..prost_types::DescriptorProto::default()
            }],
            ..FileDescriptorProto::default()
        };

        let proto_text = decompile_file_descriptor_to_proto(&sample_fd);
        assert!(proto_text.contains("package asic_sw.c_6acc60406;"));
        assert!(proto_text.contains(r#"import public "base_a.proto";"#));
        assert!(proto_text.contains(r#"import "base_b.proto";"#));
        assert!(proto_text.contains("option allow_alias = true;"));
        assert!(proto_text.contains("map<string, uint64> attrs = 1;"));
        assert!(proto_text.contains("oneof payload_kind {"));
        assert!(proto_text.contains("optional bool opt_flag = 3;"));
        assert!(proto_text.contains("reserved 10;"));
        assert!(proto_text.contains("reserved 20 to 24;"));
        assert!(proto_text.contains(r#"reserved "legacy_slot";"#));

        // Also test proto2 required + empty package + empty sanitize
        assert_eq!(sanitize_proto_qualified_name(""), "");
        let proto2_fd = FileDescriptorProto {
            name: Some(String::from("base_a.proto")),
            syntax: Some(String::from("proto2")),
            dependency: vec![String::from("google/protobuf/any.proto")],
            message_type: vec![prost_types::DescriptorProto {
                name: Some(String::from("ReqMsg")),
                field: vec![
                    prost_types::FieldDescriptorProto {
                        name: Some(String::from("req_id")),
                        number: Some(1),
                        label: Some(2),
                        r#type: Some(5),
                        ..prost_types::FieldDescriptorProto::default()
                    },
                    prost_types::FieldDescriptorProto {
                        name: Some(String::from("opt_id")),
                        number: Some(2),
                        label: Some(1),
                        r#type: Some(5),
                        ..prost_types::FieldDescriptorProto::default()
                    },
                    prost_types::FieldDescriptorProto {
                        name: Some(String::from("unlabeled_id")),
                        number: Some(3),
                        label: None,
                        r#type: Some(5),
                        ..prost_types::FieldDescriptorProto::default()
                    },
                ],
                ..prost_types::DescriptorProto::default()
            }],
            ..FileDescriptorProto::default()
        };
        let p2_text = decompile_file_descriptor_to_proto(&proto2_fd);
        assert!(p2_text.contains(r#"syntax = "proto2";"#));
        assert!(p2_text.contains("required int32 req_id = 1;"));
        assert!(p2_text.contains("optional int32 opt_id = 2;"));

        let base_b_fd = FileDescriptorProto {
            name: Some(String::from("base_b.proto")),
            dependency: vec![String::from("base_a.proto")],
            ..FileDescriptorProto::default()
        };
        let sorted = topological_sort_file_descriptors(&[
            FileDescriptorProto::default(),
            sample_fd.clone(),
            base_b_fd,
            proto2_fd,
        ])?;
        assert_eq!(sorted.len(), 3);
        assert_eq!(
            sorted.first().and_then(|f| f.name.as_deref()),
            Some("base_a.proto")
        );

        // Cyclic dependency detection
        let cyc1 = FileDescriptorProto {
            name: Some(String::from("c1.proto")),
            dependency: vec![String::from("c2.proto")],
            ..FileDescriptorProto::default()
        };
        let cyc2 = FileDescriptorProto {
            name: Some(String::from("c2.proto")),
            dependency: vec![String::from("c1.proto")],
            ..FileDescriptorProto::default()
        };
        assert!(topological_sort_file_descriptors(&[cyc1, cyc2]).is_err());

        // Synthetic ELF64 extraction tests
        let fd_wire = sample_fd.encode_to_vec();
        let elf_ok =
            build_synthetic_forensics_elf64(&fd_wire, b"sample_blob.bin", b"tpu-data", false);
        let carved = extract_elf64_descriptor_table_protos(&elf_ok)?;
        assert_eq!(carved.len(), 1);
        if let Some(first_rec) = carved.first() {
            let cloned_rec = first_rec.clone();
            assert_eq!(&cloned_rec, first_rec);
            assert_ne!(format!("{cloned_rec:?}"), "");
            assert_eq!(first_rec.raw_wire_bytes.as_slice(), fd_wire.as_slice());
        }

        let toc_entries = extract_elf64_filewrapper_toc_entries(&elf_ok)?;
        assert_eq!(toc_entries.len(), 3);
        assert_eq!(
            toc_entries.first().map(|e| e.name.as_str()),
            Some("sample_blob.bin")
        );
        verify_filewrapper_toc_md5(&toc_entries)?;

        // Corrupt MD5 flagged as md5_verified == false and rejected by verify_filewrapper_toc_md5
        let elf_bad_md5 =
            build_synthetic_forensics_elf64(&fd_wire, b"sample_blob.bin", b"tpu-data", true);
        let bad_entries = extract_elf64_filewrapper_toc_entries(&elf_bad_md5)?;
        assert_eq!(bad_entries.first().map(|e| e.md5_verified), Some(false));
        assert!(verify_filewrapper_toc_md5(&bad_entries).is_err());

        // Minimal valid 32-bit & 64-bit ELF headers without sections
        let mut min_elf32 = Vec::with_capacity(52);
        min_elf32.extend_from_slice(b"\x7fELF\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00");
        min_elf32.extend_from_slice(&3_u16.to_le_bytes());
        min_elf32.extend_from_slice(&3_u16.to_le_bytes());
        min_elf32.extend_from_slice(&1_u32.to_le_bytes());
        min_elf32.extend_from_slice(&[0_u8; 16]);
        min_elf32.extend_from_slice(&52_u16.to_le_bytes());
        min_elf32.extend_from_slice(&[0_u8; 10]);
        assert!(extract_elf64_descriptor_table_protos(&min_elf32).is_err());

        let mut min_elf64 = Vec::with_capacity(64);
        min_elf64.extend_from_slice(b"\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00");
        min_elf64.extend_from_slice(&3_u16.to_le_bytes());
        min_elf64.extend_from_slice(&62_u16.to_le_bytes());
        min_elf64.extend_from_slice(&1_u32.to_le_bytes());
        min_elf64.extend_from_slice(&[0_u8; 28]);
        min_elf64.extend_from_slice(&64_u16.to_le_bytes());
        min_elf64.extend_from_slice(&[0_u8; 10]);
        assert!(extract_elf64_descriptor_table_protos(&min_elf64).is_err());
        assert!(extract_elf64_filewrapper_toc_entries(&min_elf64).is_err());
        assert!(extract_elf64_descriptor_table_protos(b"not-an-elf").is_err());
        assert!(extract_elf64_filewrapper_toc_entries(b"not-an-elf").is_err());

        // Helper edge cases
        let spans = [AllocSectionSpan {
            addr: 0x1000,
            offset: 0,
            size: 4,
        }];
        if let Some(&span_copy) = spans.first() {
            assert_ne!(format!("{span_copy:?}"), "");
        }
        assert!(read_elf_cstring(b"abcd", &spans, 0x1000).is_none());

        Ok(())
    }

    #[test]
    fn test_decode_encoded_gcsr_brotli_csr_map_and_cached_libtpu() -> LibTpuResult<()> {
        let gcsr = RegisterTranslatorData {
            nodes: vec![
                crate::proto::EncodedNode {
                    mnemonic: String::from("vlc_tc_core_state"),
                    registers: vec![EncodedRegister {
                        name: String::from("RUN_STATUS"),
                        offset: 0x18,
                        register_data: vec![],
                        width: 8,
                        register_data_index: 0,
                    }],
                    memories: vec![],
                    tag: Vec::new(),
                    base_offset: 0xa842_2000,
                    layout_index: 0,
                },
                crate::proto::EncodedNode {
                    mnemonic: String::from("shared_node"),
                    registers: vec![],
                    memories: vec![],
                    tag: Vec::new(),
                    base_offset: 0xb034_0000,
                    layout_index: 0,
                },
                crate::proto::EncodedNode {
                    mnemonic: String::from("unmapped_layout_node"),
                    registers: vec![],
                    memories: vec![],
                    tag: Vec::new(),
                    base_offset: 0xc000_0000,
                    layout_index: -1,
                },
                crate::proto::EncodedNode {
                    mnemonic: String::from("oob_layout_node"),
                    registers: vec![],
                    memories: vec![],
                    tag: Vec::new(),
                    base_offset: 0xd000_0000,
                    layout_index: 99,
                },
            ],
            register_data_table: vec![],
            node_layout_table: vec![crate::proto::EncodedNode {
                mnemonic: String::new(),
                registers: vec![EncodedRegister {
                    name: String::from("VIRT_FISH_ADDR"),
                    offset: 0x0,
                    register_data: vec![],
                    width: 0,
                    register_data_index: 0,
                }],
                memories: vec![],
                tag: Vec::new(),
                base_offset: 0,
                layout_index: 0,
            }],
        };

        let wire = gcsr.encode_proto();
        let mut br_bytes = Vec::new();
        {
            let mut writer = brotli::CompressorWriter::new(&mut br_bytes, 4096, 4, 20);
            writer
                .write_all(&wire)
                .map_err(|e| LibTpuError::BinaryFormatError {
                    reason: format!("brotli compress failed: {e}"),
                })?;
        }

        let entries = decode_encoded_gcsr_brotli_csr_map(&br_bytes)?;
        assert_eq!(entries.len(), 2);
        assert_eq!(
            entries.first(),
            Some(&BrotliRegisterTreeEntry {
                full_path: String::from("vlc_tc_core_state.RUN_STATUS"),
                bar0_offset: 0xa842_2018,
                width_bits: 64,
            })
        );
        assert_eq!(
            entries.get(1),
            Some(&BrotliRegisterTreeEntry {
                full_path: String::from("shared_node.VIRT_FISH_ADDR"),
                bar0_offset: 0xb034_0000,
                width_bits: 64,
            })
        );
        assert!(decode_encoded_gcsr_brotli_csr_map(b"invalid-brotli").is_err());

        // Verify against the real cached 467.5 MB official libtpu.so when present on disk
        let real_so = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../.cache/wheels/extracted/0.0.52/cp311/manylinux_2_31_x86_64/libtpu.so");
        if real_so.exists() {
            let raw_elf = std::fs::read(&real_so).map_err(|e| LibTpuError::BinaryFormatError {
                reason: format!("read real libtpu.so failed: {e}"),
            })?;
            let protos = extract_elf64_descriptor_table_protos(&raw_elf)?;
            assert_eq!(protos.len(), 738);
            let fds: Vec<FileDescriptorProto> =
                protos.into_iter().map(|r| r.file_descriptor).collect();
            let topo = topological_sort_file_descriptors(&fds)?;
            assert_eq!(topo.len(), 738);

            let toc = extract_elf64_filewrapper_toc_entries(&raw_elf)?;
            assert_eq!(toc.len(), 229);
            assert_eq!(toc.iter().filter(|e| e.md5_verified).count(), 227);
            let csr_entries: Vec<&FileTocEntry> = toc
                .iter()
                .filter(|e| e.name.ends_with("_pf_csrs.br"))
                .collect();
            assert_eq!(csr_entries.len(), 4);
            assert!(csr_entries.iter().all(|e| e.md5_verified));
            let csr_count = csr_entries.iter().try_fold(0_usize, |acc, e| {
                let regs = decode_encoded_gcsr_brotli_csr_map(&e.data)?;
                Ok::<usize, LibTpuError>(acc + regs.len())
            })?;
            assert_eq!(csr_count, 2161);
        }

        Ok(())
    }

    #[test]
    fn test_phase1_2_hardware_csr_and_cost_model_forensics() -> LibTpuResult<()> {
        use crate::proto::{
            BarnaCoreAddressHandlerConfig, EncodedNode, GeneralErrorSource, RegField,
        };

        // 1. StaticCsrFamily & gcsr_swaccess_to_code branch coverage
        assert_eq!(
            StaticCsrFamily::from_mangled_shard_symbol("unrelated"),
            None
        );
        assert_eq!(
            StaticCsrFamily::from_mangled_shard_symbol("PerformanceCounterNameToString_no_ens"),
            None
        );
        assert_eq!(
            StaticCsrFamily::from_mangled_shard_symbol(
                "_ZN7asic_sw6driver7deepsea3jxc30PerformanceCounterNameToStringENS2_"
            ),
            None
        );
        assert_eq!(
            StaticCsrFamily::from_mangled_shard_symbol(
                "_ZN7asic_sw6driver7deepsea3unk31PerformanceCounterNameToString0ENS2_"
            ),
            None
        );
        for (sym, expected) in [
            (
                "_ZN3jxc31PerformanceCounterNameToString0ENS_",
                StaticCsrFamily::Jxc,
            ),
            (
                "_ZN3pfc31PerformanceCounterNameToString0ENS_",
                StaticCsrFamily::Pfc,
            ),
            (
                "_ZN3plc31PerformanceCounterNameToString0ENS_",
                StaticCsrFamily::Plc,
            ),
            (
                "_ZN3vlc31PerformanceCounterNameToString0ENS_",
                StaticCsrFamily::Vlc,
            ),
            (
                "_ZN3vfc31PerformanceCounterNameToString0ENS_",
                StaticCsrFamily::Vfc,
            ),
            (
                "_ZN3glc31PerformanceCounterNameToString0ENS_",
                StaticCsrFamily::Glc,
            ),
            (
                "_ZN3gfc31PerformanceCounterNameToString0ENS_",
                StaticCsrFamily::Gfc,
            ),
        ] {
            assert_eq!(
                StaticCsrFamily::from_mangled_shard_symbol(sym),
                Some(expected)
            );
        }
        assert_eq!(StaticCsrFamily::Jxc.default_bar_index(0x100), 2);
        assert_eq!(StaticCsrFamily::Vlc.default_bar_index(0x10500), 0);
        assert_eq!(StaticCsrFamily::Vlc.default_bar_index(0xa205_4008), 2);

        assert_eq!(gcsr_swaccess_to_code(1, "X"), "RO");
        assert_eq!(gcsr_swaccess_to_code(2, "X"), "RW");
        assert_eq!(gcsr_swaccess_to_code(11, "X"), "W1C");
        assert_eq!(gcsr_swaccess_to_code(21, "X"), "WO");
        assert_eq!(gcsr_swaccess_to_code(0, "tc.CORE_RESET"), "WO");
        assert_eq!(gcsr_swaccess_to_code(0, "dma.DOORBELL"), "WO");
        assert_eq!(gcsr_swaccess_to_code(0, "mem.ECC_FAULT"), "W1C");
        assert_eq!(gcsr_swaccess_to_code(0, "mem.PARITY"), "W1C");
        assert_eq!(gcsr_swaccess_to_code(0, "vat.ERROR_LOG"), "W1C");
        assert_eq!(gcsr_swaccess_to_code(0, "irq.INTERRUPT_STATUS"), "W1C");
        assert_eq!(gcsr_swaccess_to_code(0, "tc.RUN_STATUS"), "RO");
        assert_eq!(gcsr_swaccess_to_code(0, "tc.VWAIT_STATE"), "RO");
        assert_eq!(gcsr_swaccess_to_code(0, "clk.CYCLE_COUNTER"), "RO");
        assert_eq!(gcsr_swaccess_to_code(0, "fifo.INSERTION_COUNT"), "RO");
        assert_eq!(gcsr_swaccess_to_code(0, "fifo.TOTAL_OCCUPANCY"), "RO");
        assert_eq!(gcsr_swaccess_to_code(0, "icr.IDLE_TIMESTAMP"), "RO");
        assert_eq!(gcsr_swaccess_to_code(0, "tc.PC"), "RW");

        assert_eq!(read_le_u16(&[1], 0), 0);
        assert_eq!(read_le_u32(&[1, 2], 0), 0);
        assert_eq!(read_le_i32(&[1, 2], 0), 0);
        assert_eq!(read_le_u64(&[1, 2], 0), 0);
        assert_eq!(read_le_i64(&[1, 2], 0), 0);

        // 2. decode_encoded_gcsr_brotli_detailed branch coverage
        let gcsr_meta_inline = GcsrRegister {
            width: Some(64),
            field: vec![
                RegField {
                    mnemonic: Some(String::from("run_status")),
                    swaccess: Some(1),
                    init: Some(3),
                    msb: Some(3),
                    lsb: Some(0),
                    ..RegField::default()
                },
                RegField {
                    mnemonic: Some(String::from("full_word")),
                    swaccess: Some(2),
                    init: Some(5),
                    msb: Some(63),
                    lsb: Some(0),
                    ..RegField::default()
                },
            ],
            ..GcsrRegister::default()
        }
        .encode_proto();
        let gcsr_meta_empty_fields = GcsrRegister {
            mnemonic: Some(String::from("pc_reg")),
            width: Some(0),
            field: Vec::new(),
            ..GcsrRegister::default()
        }
        .encode_proto();

        let rtd = RegisterTranslatorData {
            nodes: vec![
                EncodedNode {
                    mnemonic: String::from("vf_node"),
                    registers: vec![
                        EncodedRegister {
                            name: String::from("RUN_STATUS"),
                            offset: 0x18,
                            register_data: gcsr_meta_inline,
                            width: 0,
                            register_data_index: -1,
                        },
                        EncodedRegister {
                            name: String::from("PC"),
                            offset: 0x20,
                            register_data: Vec::new(),
                            width: 4,
                            register_data_index: 0,
                        },
                        EncodedRegister {
                            name: String::from("SCRATCH_NEG"),
                            offset: 0x28,
                            register_data: Vec::new(),
                            width: -1,
                            register_data_index: -1,
                        },
                        EncodedRegister {
                            name: String::from("SCRATCH_OOB"),
                            offset: 0x30,
                            register_data: Vec::new(),
                            width: 8,
                            register_data_index: 99,
                        },
                    ],
                    memories: Vec::new(),
                    tag: Vec::new(),
                    base_offset: 0x0001_0000,
                    layout_index: -1,
                },
                EncodedNode {
                    mnemonic: String::from("pf_shared"),
                    registers: Vec::new(),
                    memories: Vec::new(),
                    tag: Vec::new(),
                    base_offset: 0xa842_0000,
                    layout_index: 0,
                },
                EncodedNode {
                    mnemonic: String::from("neg_layout"),
                    registers: Vec::new(),
                    memories: Vec::new(),
                    tag: Vec::new(),
                    base_offset: 0,
                    layout_index: -1,
                },
                EncodedNode {
                    mnemonic: String::from("oob_layout"),
                    registers: Vec::new(),
                    memories: Vec::new(),
                    tag: Vec::new(),
                    base_offset: 0,
                    layout_index: 99,
                },
            ],
            register_data_table: vec![gcsr_meta_empty_fields],
            node_layout_table: vec![EncodedNode {
                mnemonic: String::new(),
                registers: vec![EncodedRegister {
                    name: String::from("CORE_RESET"),
                    offset: 0,
                    register_data: Vec::new(),
                    width: 8,
                    register_data_index: -1,
                }],
                memories: Vec::new(),
                tag: Vec::new(),
                base_offset: 0,
                layout_index: -1,
            }],
        };

        let compress_br = |payload: &[u8]| -> LibTpuResult<Vec<u8>> {
            let mut out = Vec::new();
            let mut writer = brotli::CompressorWriter::new(&mut out, 4096, 4, 20);
            writer
                .write_all(payload)
                .map_err(|e| LibTpuError::BinaryFormatError {
                    reason: format!("brotli compress failed: {e}"),
                })?;
            drop(writer);
            Ok(out)
        };

        let detailed = decode_encoded_gcsr_brotli_detailed(&compress_br(&rtd.encode_proto())?)?;
        assert_eq!(detailed.len(), 5);
        assert_eq!(
            detailed.first().map(|e| (e.bar_index, e.reset_value)),
            Some((0, 7))
        );
        assert_eq!(detailed.get(1).map(|e| e.width_bits), Some(32));
        assert_eq!(
            detailed.get(4).map(|e| (e.bar_index, e.access.as_str())),
            Some((2, "WO"))
        );

        assert!(decode_encoded_gcsr_brotli_detailed(b"bad-brotli").is_err());
        assert!(decode_encoded_gcsr_brotli_detailed(&compress_br(&[0xFF])?).is_err());
        let bad_gcsr_rtd = RegisterTranslatorData {
            nodes: vec![EncodedNode {
                mnemonic: String::from("n"),
                registers: vec![EncodedRegister {
                    name: String::from("R"),
                    offset: 0,
                    register_data: vec![0xFF],
                    width: 8,
                    register_data_index: -1,
                }],
                memories: Vec::new(),
                tag: Vec::new(),
                base_offset: 0,
                layout_index: -1,
            }],
            register_data_table: Vec::new(),
            node_layout_table: Vec::new(),
        };
        assert!(
            decode_encoded_gcsr_brotli_detailed(&compress_br(&bad_gcsr_rtd.encode_proto())?)
                .is_err()
        );

        // 3. ErrorSources & BarnaCore cost-model decoders
        let err_src = ErrorSources {
            general_error_sources: vec![GeneralErrorSource {
                status_offset: 0xa800_1000,
                location: String::from("tcs"),
                ..GeneralErrorSource::default()
            }],
            memory_ecc_error_sources: Vec::new(),
        };
        let dec_err = decode_brotli_error_sources(&compress_br(&err_src.encode_proto())?)?;
        assert_eq!(dec_err.general_error_sources.len(), 1);
        assert!(decode_brotli_error_sources(b"bad").is_err());

        let model_params = BarnaCoreModelParameters {
            core_frequency_ghz: Some(0.7),
            hbm_bandwidth_per_stack_gbps: Some(180.0),
            ..BarnaCoreModelParameters::default()
        };
        let decoded_model = decode_barna_core_model_parameters(&model_params.encode_proto())?;
        assert!((decoded_model.core_frequency_ghz.unwrap_or(0.0) - 0.7).abs() < 1e-5);
        assert!(decode_barna_core_model_parameters(&[0xFF]).is_err());

        let const_params = BarnaCoreConstantParameters {
            address_handler_config: vec![BarnaCoreAddressHandlerConfig {
                name: Some(1),
                cycles_base: Some(12),
                ..BarnaCoreAddressHandlerConfig::default()
            }],
            ..BarnaCoreConstantParameters::default()
        };
        let decoded_consts = decode_barna_core_constant_parameters(&const_params.encode_proto())?;
        assert_eq!(decoded_consts.address_handler_config.len(), 1);
        assert!(decode_barna_core_constant_parameters(&[0xFF]).is_err());

        // 4. Synthetic ELF64 with .symtab + .strtab + .rodata + .text + .bss for extract_elf64_static_csr_maps
        assert!(extract_elf64_static_csr_maps(b"invalid-elf").is_err());
        let synth_csr_elf = build_synthetic_static_csr_elf64();
        let synth_maps = extract_elf64_static_csr_maps(&synth_csr_elf)?;
        assert_eq!(synth_maps.get(&StaticCsrFamily::Jxc).map(Vec::len), Some(2));
        assert_eq!(synth_maps.get(&StaticCsrFamily::Vlc).map(Vec::len), Some(1));
        assert_eq!(synth_maps.get(&StaticCsrFamily::Pfc).map(Vec::len), Some(0));

        // 5. Real cached 467.5 MB libtpu.so verification
        let real_so = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../.cache/wheels/extracted/0.0.52/cp311/manylinux_2_31_x86_64/libtpu.so");
        if real_so.exists() {
            let raw_elf = std::fs::read(&real_so).map_err(|e| LibTpuError::BinaryFormatError {
                reason: format!("read real libtpu.so failed: {e}"),
            })?;
            let static_maps = extract_elf64_static_csr_maps(&raw_elf)?;
            assert_eq!(
                static_maps.get(&StaticCsrFamily::Jxc).map(Vec::len),
                Some(3808)
            );
            assert_eq!(
                static_maps.get(&StaticCsrFamily::Pfc).map(Vec::len),
                Some(25388)
            );
            assert_eq!(
                static_maps.get(&StaticCsrFamily::Plc).map(Vec::len),
                Some(11619)
            );
            assert_eq!(
                static_maps.get(&StaticCsrFamily::Vlc).map(Vec::len),
                Some(7964)
            );
            assert_eq!(
                static_maps.get(&StaticCsrFamily::Vfc).map(Vec::len),
                Some(52644)
            );
            assert_eq!(
                static_maps.get(&StaticCsrFamily::Glc).map(Vec::len),
                Some(37094)
            );
            assert_eq!(
                static_maps.get(&StaticCsrFamily::Gfc).map(Vec::len),
                Some(56412)
            );

            let toc = extract_elf64_filewrapper_toc_entries(&raw_elf)?;
            let mut detailed_total = 0_usize;
            let mut err_sources_total = 0_usize;
            let mut model_params_count = 0_usize;
            for entry in &toc {
                if entry.name.ends_with("_pf_csrs.br") {
                    detailed_total += decode_encoded_gcsr_brotli_detailed(&entry.data)?.len();
                } else if entry.name.ends_with("_error_collector_error_sources.br") {
                    let es = decode_brotli_error_sources(&entry.data)?;
                    err_sources_total +=
                        es.general_error_sources.len() + es.memory_ecc_error_sources.len();
                } else if entry.name.ends_with("_model_parameters.binarypb") {
                    let mp = decode_barna_core_model_parameters(&entry.data)?;
                    assert!(mp.core_frequency_ghz.unwrap_or(0.0) > 0.5);
                    model_params_count += 1;
                }
            }
            assert_eq!(detailed_total, 2161);
            assert_eq!(err_sources_total, 17378);
            assert_eq!(model_params_count, 3);
        }

        Ok(())
    }

    /// Builds a synthetic ELF64 containing `.rodata`, `.text`, `.bss`, `.symtab`, and `.strtab`
    /// with multi-range, single-range, and no-op `PerformanceCounterNameToString` shards.
    fn build_synthetic_static_csr_elf64() -> Vec<u8> {
        let base_va = 0x20_0000_u64;
        let shstrtab = b"\0.shstrtab\0.rodata\0.text\0.bss\0.strtab\0.symtab\0";
        let mut buf = vec![0_u8; 64];
        let pad8 = |b: &mut Vec<u8>| {
            while !b.len().is_multiple_of(8) {
                b.push(0);
            }
        };

        let shstr_off = buf.len();
        buf.extend_from_slice(shstrtab);
        pad8(&mut buf);

        // .rodata: GOT placeholder (8B), string pool, kRelStrOffsets, kCounterRanges
        let rodata_off = buf.len();
        let got_va = base_va + rodata_off as u64;
        buf.extend_from_slice(&0_u64.to_le_bytes());

        let pool_va = base_va + buf.len() as u64;
        buf.extend_from_slice(b"MGT_USER_PCIE_CLK_COUNTERTN_0_USER_RUN_STATUSVF_VLC_DOORBELL");
        pad8(&mut buf);

        let rel_multi_va = base_va + buf.len() as u64;
        // 3 valid strings: [0..25], [25..45], and 1 inverted [10..5] to exercise get(start..end) == None
        for u in [0_u16, 25_u16, 45_u16, 10_u16, 5_u16] {
            buf.extend_from_slice(&u.to_le_bytes());
        }
        pad8(&mut buf);

        let rel_single_va = base_va + buf.len() as u64;
        // [45..60] ("VF_VLC_DOORBELL") and inverted [8..2]
        for u in [45_u16, 60_u16, 8_u16, 2_u16] {
            buf.extend_from_slice(&u.to_le_bytes());
        }
        pad8(&mut buf);

        let ranges_va = base_va + buf.len() as u64;
        // 1 CounterRange (24B): base=0x402008, stride=8, str_base=0, cnt=3 (2 valid + 1 inverted), rel_idx=0
        buf.extend_from_slice(&0x0040_2008_u64.to_le_bytes());
        buf.extend_from_slice(&8_u32.to_le_bytes());
        buf.extend_from_slice(&0_u32.to_le_bytes());
        buf.extend_from_slice(&3_u16.to_le_bytes());
        buf.extend_from_slice(&0_u16.to_le_bytes());
        buf.extend_from_slice(&0_u32.to_le_bytes());
        let rodata_len = buf.len() - rodata_off;

        // .text: Fn 0 (multi-range 3jxc), Fn 1 (single-range 3vlc), Fn 2 (noop 3pfc)
        let to_i64 = |v: u64| i64::try_from(v).unwrap_or(0);
        let len_i64 = |v: usize| i64::try_from(v).unwrap_or(0);
        let text_off = buf.len();
        let fn0_va = base_va + buf.len() as u64;
        let mut fn0 = Vec::new();
        // lea rsi, [rip + disp32] -> got_va
        fn0.extend_from_slice(&[0x48, 0x8d, 0x35]);
        let disp_got0 = i32::try_from(to_i64(got_va) - (to_i64(fn0_va) + 7)).unwrap_or(0);
        fn0.extend_from_slice(&disp_got0.to_le_bytes());
        // First 4c 8d 05 WITHOUT 0xba to cover the false branch of `code.get(i + 7) == Some(&0xba)`
        fn0.extend_from_slice(&[0x4c, 0x8d, 0x05, 0, 0, 0, 0, 0x90]);
        // Second 4c 8d 05 WITH 0xba 1
        let r_ip = to_i64(fn0_va) + len_i64(fn0.len()) + 7;
        fn0.extend_from_slice(&[0x4c, 0x8d, 0x05]);
        let disp_r = i32::try_from(to_i64(ranges_va) - r_ip).unwrap_or(0);
        fn0.extend_from_slice(&disp_r.to_le_bytes());
        fn0.push(0xba);
        fn0.extend_from_slice(&1_u32.to_le_bytes());
        // movabs rdx, rel_multi_va - got_va
        fn0.extend_from_slice(&[0x48, 0xba]);
        fn0.extend_from_slice(&(to_i64(rel_multi_va) - to_i64(got_va)).to_le_bytes());
        // movabs rcx, pool_va - got_va
        fn0.extend_from_slice(&[0x48, 0xb9]);
        fn0.extend_from_slice(&(to_i64(pool_va) - to_i64(got_va)).to_le_bytes());
        fn0.extend_from_slice(&[0xc3; 12]);
        buf.extend_from_slice(&fn0);
        pad8(&mut buf);

        let fn1_va = base_va + buf.len() as u64;
        let mut fn1 = Vec::new();
        // leaq -0x10500(%rdi), %rax
        fn1.extend_from_slice(&[0x48, 0x8d, 0x87]);
        fn1.extend_from_slice(&(-0x10500_i32).to_le_bytes());
        // cmpq $0x09, %rax -> count = (8 / 8) + 1 = 2 (1 valid + 1 inverted)
        fn1.extend_from_slice(&[0x48, 0x3d]);
        fn1.extend_from_slice(&9_u32.to_le_bytes());
        // Second 48 3d to cover `single_count == 0` false branch
        fn1.extend_from_slice(&[0x48, 0x3d, 1, 0, 0, 0]);
        // lea rdx, [rip + disp32] -> rel_single_va
        let rel1_ip = to_i64(fn1_va) + len_i64(fn1.len()) + 7;
        fn1.extend_from_slice(&[0x48, 0x8d, 0x15]);
        let disp_rel1 = i32::try_from(to_i64(rel_single_va) - rel1_ip).unwrap_or(0);
        fn1.extend_from_slice(&disp_rel1.to_le_bytes());
        // lea rsi, [rip + disp32] -> got_va
        let got1_ip = to_i64(fn1_va) + len_i64(fn1.len()) + 7;
        fn1.extend_from_slice(&[0x48, 0x8d, 0x35]);
        let disp_got1 = i32::try_from(to_i64(got_va) - got1_ip).unwrap_or(0);
        fn1.extend_from_slice(&disp_got1.to_le_bytes());
        // movabs rax, pool_va - got_va
        fn1.extend_from_slice(&[0x48, 0xb8]);
        fn1.extend_from_slice(&(to_i64(pool_va) - to_i64(got_va)).to_le_bytes());
        fn1.extend_from_slice(&[0xc3; 12]);
        buf.extend_from_slice(&fn1);
        pad8(&mut buf);

        let fn2_va = base_va + buf.len() as u64;
        let fn2 = [0x90_u8; 16];
        buf.extend_from_slice(&fn2);
        let text_len = buf.len() - text_off;
        pad8(&mut buf);

        // .strtab
        let strtab_off = buf.len();
        let mut strtab = vec![0_u8];
        let push_str = |st: &mut Vec<u8>, s: &[u8]| -> u32 {
            let off = u32::try_from(st.len()).unwrap_or(0);
            st.extend_from_slice(s);
            st.push(0);
            off
        };
        let s_obj = push_str(&mut strtab, b"kSomeObject");
        let s_zero_va = push_str(&mut strtab, b"_ZN3jxc31PerformanceCounterNameToString9ENS_");
        let s_zero_sz = push_str(&mut strtab, b"_ZN3jxc31PerformanceCounterNameToString8ENS_");
        let s_unmapped = push_str(&mut strtab, b"_ZN3jxc31PerformanceCounterNameToString7ENS_");
        let s_other_fn = push_str(&mut strtab, b"helper_fn");
        let s_fn0 = push_str(&mut strtab, b"_ZN3jxc31PerformanceCounterNameToString0ENS_");
        let s_fn1 = push_str(&mut strtab, b"_ZN3vlc31PerformanceCounterNameToString1ENS_");
        let s_fn2 = push_str(&mut strtab, b"_ZN3pfc31PerformanceCounterNameToString2ENS_");
        buf.extend_from_slice(&strtab);
        let strtab_len = strtab.len();
        pad8(&mut buf);

        // .symtab (24B Elf64_Sym entries)
        let symtab_off = buf.len();
        let push_sym = |b: &mut Vec<u8>, name: u32, info: u8, shndx: u16, val: u64, sz: u64| {
            b.extend_from_slice(&name.to_le_bytes());
            b.push(info);
            b.push(0);
            b.extend_from_slice(&shndx.to_le_bytes());
            b.extend_from_slice(&val.to_le_bytes());
            b.extend_from_slice(&sz.to_le_bytes());
        };
        push_sym(&mut buf, 0, 0, 0, 0, 0);
        push_sym(&mut buf, s_obj, 0x11, 2, got_va, 8);
        push_sym(&mut buf, s_zero_va, 0x12, 3, 0, 16);
        push_sym(&mut buf, s_zero_sz, 0x12, 3, fn0_va, 0);
        push_sym(&mut buf, s_unmapped, 0x12, 3, 0x9999_0000, 16);
        push_sym(&mut buf, s_other_fn, 0x12, 3, fn2_va, 16);
        push_sym(&mut buf, s_fn0, 0x12, 3, fn0_va, fn0.len() as u64);
        push_sym(&mut buf, s_fn1, 0x12, 3, fn1_va, fn1.len() as u64);
        push_sym(&mut buf, s_fn2, 0x12, 3, fn2_va, fn2.len() as u64);
        push_sym(&mut buf, u32::MAX / 2, 0x12, 3, fn2_va, 16);
        let symtab_len = buf.len() - symtab_off;
        pad8(&mut buf);

        // Section headers: 0=NULL, 1=.shstrtab, 2=.rodata, 3=.text, 4=.bss (SHT_NOBITS), 5=.strtab, 6=.symtab
        let shoff = buf.len();
        let push_shdr = |b: &mut Vec<u8>,
                         name: u32,
                         ty: u32,
                         flags: u64,
                         addr: u64,
                         off: usize,
                         sz: usize,
                         link: u32,
                         entsize: u64| {
            b.extend_from_slice(&name.to_le_bytes());
            b.extend_from_slice(&ty.to_le_bytes());
            b.extend_from_slice(&flags.to_le_bytes());
            b.extend_from_slice(&addr.to_le_bytes());
            b.extend_from_slice(&(off as u64).to_le_bytes());
            b.extend_from_slice(&(sz as u64).to_le_bytes());
            b.extend_from_slice(&link.to_le_bytes());
            b.extend_from_slice(&0_u32.to_le_bytes());
            b.extend_from_slice(&8_u64.to_le_bytes());
            b.extend_from_slice(&entsize.to_le_bytes());
        };
        push_shdr(&mut buf, 0, 0, 0, 0, 0, 0, 0, 0);
        push_shdr(&mut buf, 1, 3, 0, 0, shstr_off, shstrtab.len(), 0, 0);
        push_shdr(
            &mut buf,
            11,
            1,
            2,
            base_va + rodata_off as u64,
            rodata_off,
            rodata_len,
            0,
            0,
        );
        push_shdr(
            &mut buf,
            19,
            1,
            6,
            base_va + text_off as u64,
            text_off,
            text_len,
            0,
            0,
        );
        push_shdr(&mut buf, 25, 8, 3, base_va + 0x80000, 0, 64, 0, 0);
        push_shdr(&mut buf, 30, 3, 0, 0, strtab_off, strtab_len, 0, 0);
        push_shdr(&mut buf, 38, 2, 0, 0, symtab_off, symtab_len, 5, 24);

        // ELF64 header
        let mut eh = [0_u8; 64];
        eh[0..4].copy_from_slice(b"\x7fELF");
        eh[4] = 2;
        eh[5] = 1;
        eh[6] = 1;
        eh[16..18].copy_from_slice(&3_u16.to_le_bytes());
        eh[18..20].copy_from_slice(&62_u16.to_le_bytes());
        eh[20..24].copy_from_slice(&1_u32.to_le_bytes());
        eh[40..48].copy_from_slice(&(shoff as u64).to_le_bytes());
        eh[52..54].copy_from_slice(&64_u16.to_le_bytes());
        eh[58..60].copy_from_slice(&64_u16.to_le_bytes());
        eh[60..62].copy_from_slice(&7_u16.to_le_bytes());
        eh[62..64].copy_from_slice(&1_u16.to_le_bytes());
        if let Some(hdr) = buf.get_mut(0..64) {
            hdr.copy_from_slice(&eh);
        }
        buf
    }

    #[test]
    fn test_decompile_all_738_file_descriptors_and_editions_edge_cases() -> LibTpuResult<()> {
        let embedded = prost_types::FileDescriptorSet::decode(
            crate::proto::EMBEDDED_FILE_DESCRIPTOR_SET_BYTES,
        )
        .map_err(|e| LibTpuError::BinaryFormatError {
            reason: e.to_string(),
        })?;
        let sorted = topological_sort_file_descriptors(&embedded.file)?;
        assert_eq!(sorted.len(), 738);
        for fd in &sorted {
            let text = decompile_file_descriptor_to_proto(fd);
            assert!(text.starts_with("syntax = "));
        }

        // Cover editions syntax check, escaped string/bytes/blank defaults, double enum collision,
        // single-tag extension range, streaming service, and implicit descriptor.proto import.
        assert_eq!(sanitize_proto_qualified_name(".xla.6acc"), "xla.c_6acc");
        let pure_editions = FileDescriptorProto {
            name: Some(String::from("pure_ed.proto")),
            syntax: Some(String::from("editions")),
            enum_type: vec![prost_types::EnumDescriptorProto {
                name: Some(String::from("OpenEnum")),
                value: vec![prost_types::EnumValueDescriptorProto {
                    name: Some(String::from("ZERO")),
                    number: Some(0),
                    options: None,
                }],
                options: None,
                reserved_range: Vec::new(),
                reserved_name: Vec::new(),
            }],
            message_type: vec![
                prost_types::DescriptorProto {
                    name: None,
                    nested_type: vec![prost_types::DescriptorProto {
                        name: None,
                        ..prost_types::DescriptorProto::default()
                    }],
                    ..prost_types::DescriptorProto::default()
                },
                prost_types::DescriptorProto {
                    name: Some(String::from("MapCollisionMsg")),
                    nested_type: vec![prost_types::DescriptorProto {
                        name: Some(String::from("ItemsEntry")),
                        field: vec![
                            prost_types::FieldDescriptorProto {
                                name: Some(String::from("key")),
                                number: Some(1),
                                r#type: Some(9),
                                ..prost_types::FieldDescriptorProto::default()
                            },
                            prost_types::FieldDescriptorProto {
                                name: Some(String::from("value")),
                                number: Some(2),
                                r#type: Some(9),
                                ..prost_types::FieldDescriptorProto::default()
                            },
                        ],
                        options: Some(prost_types::MessageOptions {
                            map_entry: Some(true),
                            ..prost_types::MessageOptions::default()
                        }),
                        ..prost_types::DescriptorProto::default()
                    }],
                    field: vec![prost_types::FieldDescriptorProto {
                        name: Some(String::from("other_entry")),
                        number: Some(1),
                        r#type: Some(11),
                        type_name: Some(String::from("other.ItemsEntry")),
                        ..prost_types::FieldDescriptorProto::default()
                    }],
                    ..prost_types::DescriptorProto::default()
                },
            ],
            ..FileDescriptorProto::default()
        };
        assert!(
            decompile_file_descriptor_to_proto(&pure_editions).contains("syntax = \"proto3\";")
        );

        for p2_msg in [
            prost_types::DescriptorProto {
                name: Some(String::from("M1")),
                enum_type: vec![prost_types::EnumDescriptorProto {
                    name: Some(String::from("E")),
                    value: vec![prost_types::EnumValueDescriptorProto {
                        name: Some(String::from("ONE")),
                        number: Some(1),
                        options: None,
                    }],
                    ..prost_types::EnumDescriptorProto::default()
                }],
                ..prost_types::DescriptorProto::default()
            },
            prost_types::DescriptorProto {
                name: Some(String::from("M2")),
                extension: vec![prost_types::FieldDescriptorProto {
                    name: Some(String::from("nested_opt")),
                    extendee: Some(String::from(".google.protobuf.FieldOptions")),
                    number: Some(50002),
                    r#type: Some(5),
                    ..prost_types::FieldDescriptorProto::default()
                }],
                ..prost_types::DescriptorProto::default()
            },
            prost_types::DescriptorProto {
                name: Some(String::from("M3")),
                field: vec![prost_types::FieldDescriptorProto {
                    name: Some(String::from("f_def")),
                    number: Some(1),
                    label: Some(1),
                    r#type: Some(5),
                    default_value: Some(String::from("7")),
                    ..prost_types::FieldDescriptorProto::default()
                }],
                ..prost_types::DescriptorProto::default()
            },
            prost_types::DescriptorProto {
                name: Some(String::from("M4")),
                extension_range: vec![prost_types::descriptor_proto::ExtensionRange {
                    start: Some(10),
                    end: Some(20),
                    options: None,
                }],
                ..prost_types::DescriptorProto::default()
            },
        ] {
            let m_fd = FileDescriptorProto {
                name: Some(String::from("m_p2.proto")),
                syntax: Some(String::from("editions")),
                message_type: vec![p2_msg],
                ..FileDescriptorProto::default()
            };
            assert!(decompile_file_descriptor_to_proto(&m_fd).contains("syntax = \"proto2\";"));
        }

        let explicit_desc_dep = FileDescriptorProto {
            name: Some(String::from("has_desc.proto")),
            syntax: Some(String::from("proto2")),
            dependency: vec![
                String::from("google/protobuf/descriptor.proto"),
                String::from("net/proto2/proto/descriptor.proto"),
            ],
            extension: vec![prost_types::FieldDescriptorProto {
                name: Some(String::from("no_target")),
                extendee: None,
                number: Some(50003),
                r#type: Some(5),
                ..prost_types::FieldDescriptorProto::default()
            }],
            ..FileDescriptorProto::default()
        };
        assert!(
            decompile_file_descriptor_to_proto(&explicit_desc_dep)
                .contains("import \"google/protobuf/descriptor.proto\";")
        );

        let ed_proto2_enum = FileDescriptorProto {
            name: Some(String::from("ed_closed_enum.proto")),
            syntax: Some(String::from("editions")),
            enum_type: vec![prost_types::EnumDescriptorProto {
                name: Some(String::from("ClosedEnum")),
                value: vec![prost_types::EnumValueDescriptorProto {
                    name: Some(String::from("ONE")),
                    number: Some(1),
                    options: None,
                }],
                options: None,
                reserved_range: Vec::new(),
                reserved_name: Vec::new(),
            }],
            ..FileDescriptorProto::default()
        };
        assert!(
            decompile_file_descriptor_to_proto(&ed_proto2_enum).contains("syntax = \"proto2\";")
        );

        let rich_fd = FileDescriptorProto {
            name: Some(String::from("rich.proto")),
            package: Some(String::from("pkg")),
            syntax: Some(String::from("editions")),
            dependency: vec![String::from("dep_a.proto"), String::from("dep_b.proto")],
            public_dependency: vec![0],
            message_type: vec![prost_types::DescriptorProto {
                name: Some(String::from("RichMsg")),
                extension_range: vec![
                    prost_types::descriptor_proto::ExtensionRange {
                        start: Some(5),
                        end: Some(6),
                        options: None,
                    },
                    prost_types::descriptor_proto::ExtensionRange {
                        start: Some(10),
                        end: Some(20),
                        options: None,
                    },
                ],
                reserved_range: vec![prost_types::descriptor_proto::ReservedRange {
                    start: Some(50),
                    end: Some(55),
                }],
                reserved_name: vec![String::from("old_field")],
                enum_type: vec![prost_types::EnumDescriptorProto {
                    name: Some(String::from("Kind")),
                    value: vec![
                        prost_types::EnumValueDescriptorProto {
                            name: Some(String::from("dup")),
                            number: Some(0),
                            options: None,
                        },
                        prost_types::EnumValueDescriptorProto {
                            name: Some(String::from("dup")),
                            number: Some(0),
                            options: None,
                        },
                        prost_types::EnumValueDescriptorProto {
                            name: Some(String::from("blank_def")),
                            number: Some(1),
                            options: None,
                        },
                    ],
                    options: None,
                    reserved_range: Vec::new(),
                    reserved_name: Vec::new(),
                }],
                field: vec![
                    prost_types::FieldDescriptorProto {
                        name: Some(String::from("dup")),
                        number: Some(1),
                        label: Some(1),
                        r#type: Some(9),
                        default_value: Some(String::from("a\\b\"c\nd\re\tf")),
                        ..prost_types::FieldDescriptorProto::default()
                    },
                    prost_types::FieldDescriptorProto {
                        name: Some(String::from("DUP")),
                        number: Some(2),
                        label: Some(1),
                        r#type: Some(12),
                        default_value: Some(String::from("raw")),
                        ..prost_types::FieldDescriptorProto::default()
                    },
                    prost_types::FieldDescriptorProto {
                        name: Some(String::from("blank_def")),
                        number: Some(3),
                        label: Some(1),
                        r#type: Some(9),
                        default_value: Some(String::from("   ")),
                        ..prost_types::FieldDescriptorProto::default()
                    },
                    prost_types::FieldDescriptorProto {
                        name: Some(String::from("enum_def")),
                        number: Some(4),
                        label: Some(1),
                        r#type: Some(14),
                        type_name: Some(String::from(".pkg.RichMsg.Kind")),
                        default_value: Some(String::from("dup")),
                        ..prost_types::FieldDescriptorProto::default()
                    },
                ],
                ..prost_types::DescriptorProto::default()
            }],
            extension: vec![prost_types::FieldDescriptorProto {
                name: Some(String::from("my_msg_opt")),
                extendee: Some(String::from(".google.protobuf.MessageOptions")),
                number: Some(50001),
                label: Some(3),
                r#type: Some(5),
                options: Some(prost_types::FieldOptions {
                    packed: Some(true),
                    ..prost_types::FieldOptions::default()
                }),
                ..prost_types::FieldDescriptorProto::default()
            }],
            service: vec![prost_types::ServiceDescriptorProto {
                name: Some(String::from("StreamSvc")),
                method: vec![prost_types::MethodDescriptorProto {
                    name: Some(String::from("BiStream")),
                    input_type: Some(String::from(".pkg.RichMsg")),
                    output_type: Some(String::from(".pkg.RichMsg")),
                    client_streaming: Some(true),
                    server_streaming: Some(true),
                    options: None,
                }],
                options: None,
            }],
            ..FileDescriptorProto::default()
        };
        let rich_text = decompile_file_descriptor_to_proto(&rich_fd);
        assert!(rich_text.contains("import public \"dep_a.proto\";"));
        assert!(rich_text.contains("import \"google/protobuf/descriptor.proto\";"));
        assert!(rich_text.contains("option allow_alias = true;"));
        assert!(rich_text.contains("Kind_dup = 0;"));
        assert!(rich_text.contains("extensions 5;"));
        assert!(rich_text.contains("extensions 10 to 19;"));
        assert!(
            rich_text.contains("rpc BiStream(stream .pkg.RichMsg) returns (stream .pkg.RichMsg);")
        );
        Ok(())
    }
}

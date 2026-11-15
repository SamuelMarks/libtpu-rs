//! Build script compiling all 738 reconstructed `libtpu.so` `protodesc_cold` `.proto` schemas
//! via `protox` and `prost-build` into a binary `FileDescriptorSet`, strongly-typed Rust
//! `prost::Message` structs/enums, and auto-generated `ProtoSchemaMessage` implementations.

use heck::{ToSnakeCase, ToUpperCamelCase};
use prost::Message as _;
use prost_types::{DescriptorProto, FileDescriptorSet};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// Sanitizes a Rust identifier following `prost-build`'s exact keyword and leading-digit rules.
///
/// # Arguments
///
/// * `raw` - Candidate Rust identifier produced by `heck` case conversion.
fn prost_sanitize_identifier(raw: &str) -> String {
    match raw {
        "as" | "break" | "const" | "continue" | "else" | "enum" | "false" | "fn" | "for" | "if"
        | "impl" | "in" | "let" | "loop" | "match" | "mod" | "move" | "mut" | "pub" | "ref"
        | "return" | "static" | "struct" | "trait" | "true" | "type" | "unsafe" | "use"
        | "where" | "while" | "dyn" | "abstract" | "become" | "box" | "do" | "final" | "macro"
        | "override" | "priv" | "typeof" | "unsized" | "virtual" | "yield" | "async" | "await"
        | "try" | "gen" => format!("r#{raw}"),
        "_" | "super" | "self" | "Self" | "extern" | "crate" => format!("{raw}_"),
        s if s.chars().next().is_some_and(char::is_numeric) => format!("_{raw}"),
        _ => String::from(raw),
    }
}

/// Converts a Protobuf identifier to `prost-build`'s `snake_case` Rust module/field identifier.
///
/// # Arguments
///
/// * `ident` - Raw Protobuf package segment or message name.
fn prost_to_snake(ident: &str) -> String {
    prost_sanitize_identifier(&ident.to_snake_case())
}

/// Converts a Protobuf identifier to `prost-build`'s `UpperCamelCase` Rust type identifier.
///
/// # Arguments
///
/// * `ident` - Raw Protobuf message or enum name.
fn prost_to_upper_camel(ident: &str) -> String {
    prost_sanitize_identifier(&ident.to_upper_camel_case())
}

/// Recursively collects all `.proto` file paths under `dir` and registers `cargo:rerun-if-changed`.
///
/// # Arguments
///
/// * `dir` - Directory to traverse recursively.
/// * `out_files` - Output vector receiving discovered `.proto` paths.
///
/// # Errors
///
/// Returns [`std::io::Error`] if reading `dir` or any subdirectory fails.
fn collect_proto_files_recursively(
    dir: &Path,
    out_files: &mut Vec<PathBuf>,
) -> std::io::Result<()> {
    println!("cargo:rerun-if-changed={}", dir.display());
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|res| res.ok().map(|e| e.path()))
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            collect_proto_files_recursively(&path, out_files)?;
        } else if path.extension().and_then(|s| s.to_str()) == Some("proto") {
            println!("cargo:rerun-if-changed={}", path.display());
            out_files.push(path);
        }
    }
    Ok(())
}

/// Removes stale `.rs` files in `out_dir` prior to `prost-build` code generation.
///
/// # Arguments
///
/// * `out_dir` - Cargo `OUT_DIR` directory containing generated Rust artifacts.
///
/// # Errors
///
/// Returns [`std::io::Error`] if enumerating or removing stale `.rs` files in `out_dir` fails.
fn clean_stale_generated_rs(out_dir: &Path) -> std::io::Result<()> {
    if !out_dir.exists() {
        return Ok(());
    }
    for entry in std::fs::read_dir(out_dir)? {
        let path = entry?.path();
        if path.extension().and_then(|s| s.to_str()) == Some("rs") {
            std::fs::remove_file(path)?;
        }
    }
    Ok(())
}

/// Post-processes `prost-build` `.rs` files in `out_dir` so every generated struct, enum,
/// nested module, field, and variant carries rustdoc comments (`#![forbid(missing_docs)]` and
/// 100% `rustdoc --coverage` compliant, with broken intra-doc link brackets neutralized).
///
/// # Arguments
///
/// * `out_dir` - Cargo `OUT_DIR` directory containing `prost-build` generated `.rs` files.
///
/// # Errors
///
/// Returns [`std::io::Error`] if reading or writing generated `.rs` files in `out_dir` fails.
fn ensure_doc_comments_in_generated_rs(out_dir: &Path) -> std::io::Result<()> {
    for entry in std::fs::read_dir(out_dir)? {
        let path = entry?.path();
        if path.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        if path.file_name().and_then(|s| s.to_str()) == Some("generated_proto_include.rs") {
            let raw_inc = std::fs::read_to_string(&path)?;
            let mut doc_inc = String::with_capacity(raw_inc.len() + 4096);
            for line in raw_inc.lines() {
                if line.trim_start().starts_with("pub mod ") {
                    let indent: String = line.chars().take_while(|c| c.is_whitespace()).collect();
                    let _ = writeln!(doc_inc, "{indent}/// Generated protobuf package module.");
                }
                let _ = writeln!(doc_inc, "{line}");
            }
            std::fs::write(&path, doc_inc)?;
            continue;
        }

        let raw = std::fs::read_to_string(&path)?;
        let mut out = String::with_capacity(raw.len() + 4096);
        let mut pending_docs: Vec<String> = Vec::new();
        let mut pending_attrs: Vec<String> = Vec::new();
        for line in raw.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("///") || trimmed.starts_with("//!") {
                pending_docs.push(String::from(line));
                continue;
            }
            if trimmed.starts_with("#[") {
                pending_attrs.push(String::from(line));
                continue;
            }
            let is_item = trimmed.starts_with("pub struct ")
                || trimmed.starts_with("pub enum ")
                || trimmed.starts_with("pub mod ")
                || trimmed.starts_with("pub type ")
                || (trimmed.starts_with("pub ") && trimmed.contains(": "))
                || (!pending_attrs.is_empty()
                    && !trimmed.is_empty()
                    && !trimmed.starts_with('}')
                    && !trimmed.starts_with("//"))
                || (trimmed.contains(" = ") && trimmed.ends_with(','));

            if is_item {
                let indent: String = line.chars().take_while(|c| c.is_whitespace()).collect();
                let _ = writeln!(out, "{indent}/// Generated protobuf schema item.");
                pending_docs.clear();
            }
            for doc_line in std::mem::take(&mut pending_docs) {
                let safe_doc = doc_line.replace('[', "(").replace(']', ")");
                let _ = writeln!(out, "{safe_doc}");
            }
            for attr_line in std::mem::take(&mut pending_attrs) {
                let _ = writeln!(out, "{attr_line}");
            }
            let _ = writeln!(out, "{line}");
        }
        for doc_line in pending_docs {
            let safe_doc = doc_line.replace('[', "(").replace(']', ")");
            let _ = writeln!(out, "{safe_doc}");
        }
        for attr_line in pending_attrs {
            let _ = writeln!(out, "{attr_line}");
        }
        std::fs::write(&path, out)?;
    }
    Ok(())
}

/// Recursively collects `(full_proto_name, rust_type_path)` entries for a message and its nested
/// non-map messages.
///
/// # Arguments
///
/// * `proto_prefix` - Enclosing Protobuf scope (e.g., `"xla"` or `"xla.OuterMsg"`).
/// * `rust_mod_prefix` - Enclosing Rust module path inside `generated_prost_schemas` (e.g., `"xla"` or `"xla::outer_msg"`).
/// * `msg` - Protobuf `DescriptorProto` to traverse.
/// * `entries` - Output map from canonical `full_proto_name` to `rust_type_path`.
fn collect_message_bijection_entries(
    proto_prefix: &str,
    rust_mod_prefix: &str,
    msg: &DescriptorProto,
    entries: &mut BTreeMap<String, String>,
) {
    if msg
        .options
        .as_ref()
        .and_then(|o| o.map_entry)
        .unwrap_or(false)
    {
        return;
    }
    let Some(msg_name) = msg.name.as_deref().filter(|s| !s.is_empty()) else {
        return;
    };
    let full_proto_name = if proto_prefix.is_empty() {
        String::from(msg_name)
    } else {
        format!("{proto_prefix}.{msg_name}")
    };
    let struct_ident = prost_to_upper_camel(msg_name);
    let rust_type_path = if rust_mod_prefix.is_empty() {
        format!("generated_prost_schemas::{struct_ident}")
    } else {
        format!("generated_prost_schemas::{rust_mod_prefix}::{struct_ident}")
    };
    entries.insert(full_proto_name.clone(), rust_type_path);

    let child_mod = prost_to_snake(msg_name);
    let next_rust_mod = if rust_mod_prefix.is_empty() {
        child_mod
    } else {
        format!("{rust_mod_prefix}::{child_mod}")
    };
    for nested in &msg.nested_type {
        collect_message_bijection_entries(&full_proto_name, &next_rust_mod, nested, entries);
    }
}

/// Generates `OUT_DIR/generated_proto_traits.rs` implementing `ProtoSchemaMessage` for every
/// compiled Protobuf message across all 738 `.proto` files (including `.google.protobuf.*` WKTs)
/// and emitting `ALL_GENERATED_PROTO_MESSAGE_NAMES` and `ALL_GENERATED_PROTO_BIJECTION`.
///
/// # Arguments
///
/// * `fds` - Compiled `FileDescriptorSet` covering all 738 `.proto` files.
/// * `out_dir` - Cargo `OUT_DIR` directory to write `generated_proto_traits.rs` into.
///
/// # Errors
///
/// Returns [`std::io::Error`] if writing `generated_proto_traits.rs` or `generated_protodesc_catalog.rs` fails.
fn generate_proto_schema_traits(fds: &FileDescriptorSet, out_dir: &Path) -> std::io::Result<()> {
    let mut entries: BTreeMap<String, String> = BTreeMap::new();

    let mut pkg_reps: BTreeMap<String, (String, String)> = BTreeMap::new();
    for file in &fds.file {
        let pkg = file.package.as_deref().unwrap_or("");
        if pkg == "google.protobuf" {
            continue;
        }
        let rust_pkg_mod = if pkg.is_empty() {
            String::new()
        } else {
            pkg.split('.')
                .map(prost_to_snake)
                .collect::<Vec<_>>()
                .join("::")
        };
        for msg in &file.message_type {
            collect_message_bijection_entries(pkg, &rust_pkg_mod, msg, &mut entries);
            if !pkg_reps.contains_key(pkg)
                && !msg
                    .options
                    .as_ref()
                    .and_then(|o| o.map_entry)
                    .unwrap_or(false)
            {
                let msg_name = msg.name.as_deref().unwrap_or("");
                let full_name = if pkg.is_empty() {
                    String::from(msg_name)
                } else {
                    format!("{pkg}.{msg_name}")
                };
                if let Some(rust_ty) = entries.get(&full_name) {
                    pkg_reps.insert(String::from(pkg), (full_name, rust_ty.clone()));
                }
            }
        }
    }

    let mut code = String::with_capacity(entries.len() * 140);
    for (full_name, rust_ty) in &entries {
        let _ = writeln!(code, "impl ProtoSchemaMessage for {rust_ty} {{");
        let _ = writeln!(
            code,
            "    const PROTO_FULL_NAME: &'static str = {full_name:?};"
        );
        let _ = writeln!(code, "}}");
    }

    let _ = writeln!(
        code,
        "/// Sorted slice of all generated Protobuf message full names across `protodesc_cold`."
    );
    let _ = writeln!(
        code,
        "pub const ALL_GENERATED_PROTO_MESSAGE_NAMES: &[&str] = &["
    );
    for full_name in entries.keys() {
        let _ = writeln!(code, "    {full_name:?},");
    }
    let _ = writeln!(code, "];");

    let _ = writeln!(
        code,
        "/// Sorted `(full_proto_name, rust_type_path)` bijection across all generated messages."
    );
    let _ = writeln!(
        code,
        "pub const ALL_GENERATED_PROTO_BIJECTION: &[(&str, &str)] = &["
    );
    for (full_name, rust_ty) in &entries {
        let _ = writeln!(code, "    ({full_name:?}, {rust_ty:?}),");
    }
    let _ = writeln!(code, "];");

    let _ = writeln!(
        code,
        "/// Tuple type for a generated `.proto` package test representative."
    );
    let _ = writeln!(code, "#[cfg(test)]");
    let _ = writeln!(
        code,
        "pub type GeneratedPackageRepresentative = (&'static str, &'static str, fn() -> LibTpuResult<()>, fn(&[u8]) -> LibTpuResult<Vec<u8>>);"
    );
    let _ = writeln!(
        code,
        "/// Representative `(package, full_message_name, verify_equiv_fn, typed_wire_roundtrip_fn)` for every generated `.proto` package."
    );
    let _ = writeln!(code, "#[cfg(test)]");
    let _ = writeln!(
        code,
        "pub const GENERATED_PACKAGE_REPRESENTATIVES: &[GeneratedPackageRepresentative] = &["
    );
    for (pkg, (full_name, rust_ty)) in &pkg_reps {
        let _ = writeln!(
            code,
            "    ({pkg:?}, {full_name:?}, verify_bidirectional_proto_struct_equivalence::<{rust_ty}>, roundtrip_proto_wire_bytes::<{rust_ty}>),"
        );
    }
    let _ = writeln!(code, "];");

    std::fs::write(out_dir.join("generated_proto_traits.rs"), code)
}

/// Emits `generated_protodesc_catalog.rs` listing all 738 embedded `protodesc_cold` files.
///
/// # Arguments
///
/// * `fds` - Compiled `FileDescriptorSet` covering all 738 `.proto` files.
/// * `out_dir` - Cargo `OUT_DIR` directory to write `generated_protodesc_catalog.rs` into.
///
/// # Errors
///
/// Returns [`std::io::Error`] if writing `generated_protodesc_catalog.rs` fails.
fn write_protodesc_catalog(fds: &FileDescriptorSet, out_dir: &Path) -> std::io::Result<()> {
    let mut cat = String::with_capacity(fds.file.len() * 220);
    let _ = writeln!(
        cat,
        "/// Deterministic catalog of all 738 embedded `protodesc_cold` descriptor schemas in `libtpu.so`."
    );
    let _ = writeln!(
        cat,
        "pub const PROTODESC_COLD_293_CATALOG: &[ProtodescColdCatalogEntry] = &["
    );
    for file in &fds.file {
        let name = file.name.as_deref().unwrap_or("");
        let pkg = file.package.as_deref().unwrap_or("");
        let syn = match file.syntax.as_deref().unwrap_or("") {
            "proto3" => "proto3",
            _ => "proto2",
        };
        let _ = writeln!(cat, "    ProtodescColdCatalogEntry {{");
        let _ = writeln!(cat, "        embedded_path: {name:?},");
        let _ = writeln!(cat, "        canonical_filename: {name:?},");
        let _ = writeln!(cat, "        package: {pkg:?},");
        let _ = writeln!(cat, "        syntax: {syn:?},");
        let _ = writeln!(cat, "    }},");
    }
    let _ = writeln!(cat, "];");
    std::fs::write(out_dir.join("generated_protodesc_catalog.rs"), cat)
}

/// Entrypoint for `cargo` build script execution.
///
/// # Errors
///
/// Returns [`std::io::Error`] if `protox` or `prost-build` fails to compile the `.proto` schemas.
fn main() -> std::io::Result<()> {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default());
    let proto_dir = manifest_dir
        .join("../../docs/reverse_engineering/protos")
        .canonicalize()
        .unwrap_or_else(|_| manifest_dir.join("../../docs/reverse_engineering/protos"));
    let ext_proto_dir = manifest_dir
        .join("../../protos")
        .canonicalize()
        .unwrap_or_else(|_| manifest_dir.join("../../protos"));

    let mut golden_proto_paths = Vec::new();
    collect_proto_files_recursively(&proto_dir, &mut golden_proto_paths)?;
    golden_proto_paths.sort();

    let mut ext_proto_paths = Vec::new();
    if ext_proto_dir.exists() {
        collect_proto_files_recursively(&ext_proto_dir, &mut ext_proto_paths)?;
        ext_proto_paths.sort();
    }

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap_or_default());
    clean_stale_generated_rs(&out_dir)?;

    let golden_fds =
        protox::compile(&golden_proto_paths, [&proto_dir]).map_err(std::io::Error::other)?;
    let fds_path = out_dir.join("libtpu_file_descriptor_set.bin");
    std::fs::write(&fds_path, golden_fds.encode_to_vec())?;
    write_protodesc_catalog(&golden_fds, &out_dir)?;

    let mut all_proto_paths = golden_proto_paths;
    all_proto_paths.extend(ext_proto_paths);
    let combined_fds = protox::compile(&all_proto_paths, [&proto_dir, &ext_proto_dir])
        .map_err(std::io::Error::other)?;
    let combined_fds_path = out_dir.join("libtpu_combined_file_descriptor_set.bin");
    std::fs::write(&combined_fds_path, combined_fds.encode_to_vec())?;

    let mut config = prost_build::Config::new();
    config.btree_map(["."]);
    for eq_ty in [
        ".libtpu.ext.deepsea.LloBundleScheduleProto",
        ".libtpu.ext.deepsea.LloSoftwarePipelineScheduleProto",
        ".libtpu.ext.megascale.MegascaleCongestionControlConfigProto",
    ] {
        config.type_attribute(eq_ty, "#[derive(Eq)]");
    }
    config.include_file("generated_proto_include.rs");
    config.compile_fds(combined_fds.clone())?;

    ensure_doc_comments_in_generated_rs(&out_dir)?;
    generate_proto_schema_traits(&combined_fds, &out_dir)?;
    Ok(())
}

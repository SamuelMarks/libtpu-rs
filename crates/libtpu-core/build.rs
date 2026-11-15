//! Build script compiling the 738-file `protodesc_cold` catalog, embedded binary
//! `FileDescriptorSet` artifacts, 1:1 `OUT_DIR` `.rs` verification files, and unified
//! `generated_prost_schemas` re-exports over `libtpu-proto-s0`..`s5`.

use prost::Message as _;
use prost_types::FileDescriptorSet;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// Shared topological Protobuf shard planner and `prost-build` driver.
#[path = "../proto_codegen_shared.rs"]
pub mod proto_codegen;

/// Appends `ProtoSchemaId` implementations for `libtpu.ext.*` messages compiled inside `libtpu-core`.
///
/// # Arguments
///
/// * `fds` - Combined `FileDescriptorSet` covering all 744 `.proto` files.
/// * `code` - Output Rust source buffer.
fn append_local_proto_id_impls(fds: &FileDescriptorSet, code: &mut String) {
    let mut local_entries: BTreeMap<String, String> = BTreeMap::new();
    for file in &fds.file {
        let pkg = file.package.as_deref().unwrap_or("");
        if !proto_codegen::is_core_local_package(pkg) {
            continue;
        }
        let rust_pkg_mod = proto_codegen::package_to_rust_mod_path(pkg);
        for msg in &file.message_type {
            proto_codegen::collect_message_bijection_entries(
                pkg,
                &rust_pkg_mod,
                "generated_prost_schemas",
                msg,
                &mut local_entries,
            );
        }
    }
    for (full_name, rust_ty) in &local_entries {
        let _ = writeln!(
            code,
            "impl ::libtpu_proto_s0::ProtoSchemaId for {rust_ty} {{"
        );
        let _ = writeln!(
            code,
            "    const PROTO_FULL_NAME: &'static str = {full_name:?};"
        );
        let _ = writeln!(code, "}}");
    }
}

/// Generates `OUT_DIR/generated_proto_traits.rs` for `ProtoSchemaMessage` and schema catalogs.
///
/// # Arguments
///
/// * `fds` - Combined `FileDescriptorSet` covering all 744 `.proto` files.
/// * `out_dir` - Cargo `OUT_DIR` directory to write `generated_proto_traits.rs` into.
///
/// # Errors
///
/// Returns [`std::io::Error`] if writing `generated_proto_traits.rs` fails.
fn generate_proto_schema_traits(fds: &FileDescriptorSet, out_dir: &Path) -> std::io::Result<()> {
    let mut entries: BTreeMap<String, String> = BTreeMap::new();
    let mut pkg_reps: BTreeMap<String, (String, String)> = BTreeMap::new();

    for file in &fds.file {
        let pkg = file.package.as_deref().unwrap_or("");
        if pkg == "google.protobuf" {
            continue;
        }
        let rust_pkg_mod = proto_codegen::package_to_rust_mod_path(pkg);
        for msg in &file.message_type {
            proto_codegen::collect_message_bijection_entries(
                pkg,
                &rust_pkg_mod,
                "generated_prost_schemas",
                msg,
                &mut entries,
            );
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

    let mut code = String::with_capacity(entries.len() * 96 + 4096);
    append_local_proto_id_impls(fds, &mut code);
    let _ = writeln!(
        code,
        "impl<T: ::libtpu_proto_s0::ProtoSchemaId> ProtoSchemaMessage for T {{"
    );
    let _ = writeln!(
        code,
        "    const PROTO_FULL_NAME: &'static str = <T as ::libtpu_proto_s0::ProtoSchemaId>::PROTO_FULL_NAME;"
    );
    let _ = writeln!(code, "}}");

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

/// Entrypoint for `libtpu-core` `cargo` build script execution.
///
/// # Errors
///
/// Returns [`std::io::Error`] if `protox` or `prost-build` fails to compile the `.proto` schemas.
fn main() -> std::io::Result<()> {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default());
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap_or_default());
    proto_codegen::clean_stale_generated_rs(&out_dir)?;

    let (golden_fds, combined_fds) = proto_codegen::load_golden_and_combined_fds(&manifest_dir)?;
    std::fs::write(
        out_dir.join("libtpu_file_descriptor_set.bin"),
        golden_fds.encode_to_vec(),
    )?;
    write_protodesc_catalog(&golden_fds, &out_dir)?;

    std::fs::write(
        out_dir.join("libtpu_combined_file_descriptor_set.bin"),
        combined_fds.encode_to_vec(),
    )?;

    // Emit canonical `<pkg>.rs` files into `libtpu-core` OUT_DIR for 1:1 `proto-sync` auditing,
    // then overwrite `generated_proto_include.rs` with zero-cost re-exports from `libtpu-proto-s0..s5`.
    let mut config = prost_build::Config::new();
    config.btree_map(["."]);
    for eq_ty in proto_codegen::EXTRA_EQ_TYPES {
        config.type_attribute(eq_ty, "#[derive(Eq)]");
    }
    config.include_file("generated_proto_include.rs");
    config.compile_fds(combined_fds.clone())?;
    proto_codegen::ensure_doc_comments_in_generated_rs(&out_dir, "generated_proto_include.rs")?;

    let pkg_to_shard = proto_codegen::compute_package_shards(&combined_fds);
    proto_codegen::write_core_reexport_include(&combined_fds, &pkg_to_shard, &out_dir)?;
    generate_proto_schema_traits(&combined_fds, &out_dir)?;
    Ok(())
}

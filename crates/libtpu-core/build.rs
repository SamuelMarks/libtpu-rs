//! Build script for `libtpu-core` compiling Clean-Room `.proto` schemas via `protox`
//! and `prost-build` into a binary `FileDescriptorSet` and generated Rust module.

use prost::Message;
use std::fmt::Write as _;
use std::path::PathBuf;

/// Normalizes a `.proto` file text into single-header `proto3` syntax for `protox`
/// when a file uses `edition = "2023";` or concatenates multiple embedded sub-descriptors.
///
/// # Arguments
///
/// * `raw` - Raw `.proto` source text.
fn normalize_proto_for_protox(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut seen_syntax = false;
    let mut seen_package = false;
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("edition =") {
            if !seen_syntax {
                let _ = writeln!(out, r#"syntax = "proto3";"#);
                seen_syntax = true;
            }
            continue;
        }
        if trimmed.starts_with("option features.") {
            continue;
        }
        if trimmed.starts_with("syntax =") {
            if !seen_syntax {
                let _ = writeln!(out, r#"syntax = "proto3";"#);
                seen_syntax = true;
            }
            continue;
        }
        if trimmed.starts_with("package ") {
            if !seen_package {
                let _ = writeln!(out, "{line}");
                seen_package = true;
            }
            continue;
        }
        let _ = writeln!(out, "{line}");
    }
    out
}

/// Entrypoint for `cargo` build script execution.
///
/// # Errors
///
/// Returns an error if `protox` or `prost-build` fails to compile the staged schemas.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default());
    let proto_dir = manifest_dir
        .join("../../docs/reverse_engineering/protos")
        .canonicalize()
        .unwrap_or_else(|_| manifest_dir.join("../../docs/reverse_engineering/protos"));

    let compile_protos = [
        "tpu_version.proto",
        "target_arguments.proto",
        "isa_program.proto",
        "deepsea_executable.proto",
        "host_command_opcodes.proto",
        "runtime_config.proto",
        "memory_scheduler.proto",
        "memory_space_assignment.proto",
        "emitter_learned_cost_model_options.proto",
        "reduction_kind.proto",
        "error_codes.proto",
        "pjrt_abi_version.proto",
        "pjrt_device_dimensions.proto",
        "pjrt_partial_program.proto",
        "pjrt_value_type.proto",
        "py_host_callback.proto",
        "pytree.proto",
        "ffi_execution_state.proto",
        "ffi_attribute_map.proto",
        "dynamic_padding.proto",
        "runtime_abi_version.proto",
        "executable_abi_version.proto",
        "sparse_core_layout.proto",
        "topology_description.proto",
    ];

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap_or_default());
    let staged_proto_dir = out_dir.join("staged_protos");
    std::fs::create_dir_all(&staged_proto_dir)?;

    let mut staged_paths = Vec::with_capacity(compile_protos.len());
    for name in compile_protos {
        let src_path = proto_dir.join(name);
        println!("cargo:rerun-if-changed={}", src_path.display());
        let raw = std::fs::read_to_string(&src_path)?;
        let normalized = normalize_proto_for_protox(&raw);
        let dst_path = staged_proto_dir.join(name);
        std::fs::write(&dst_path, normalized)?;
        staged_paths.push(dst_path);
    }

    let file_descriptor_set = protox::compile(&staged_paths, [&staged_proto_dir])?;
    let fds_path = out_dir.join("libtpu_file_descriptor_set.bin");
    std::fs::write(&fds_path, file_descriptor_set.encode_to_vec())?;

    let mut config = prost_build::Config::new();
    config.file_descriptor_set_path(&fds_path);
    config.compile_fds(file_descriptor_set)?;
    Ok(())
}

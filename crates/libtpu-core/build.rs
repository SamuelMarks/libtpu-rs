//! Build script for `libtpu-core` compiling Clean-Room `.proto` schemas via `protox`
//! and `prost-build` into a binary `FileDescriptorSet` and generated Rust module.

use prost::Message;
use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::PathBuf;

/// Rewrites an `import` path so local schemas in `docs/reverse_engineering/protos/`
/// resolve against the flat `staged_proto_dir` directory.
///
/// # Arguments
///
/// * `import_path` - Raw quoted path from an `import` statement.
/// * `available_files` - Set of staged `.proto` basenames.
fn rewrite_import_path(import_path: &str, available_files: &BTreeSet<String>) -> String {
    if import_path.starts_with("google/protobuf/") {
        return import_path.to_owned();
    }
    let basename = import_path.rsplit('/').next().unwrap_or(import_path);
    if available_files.contains(basename) {
        return basename.to_owned();
    }
    import_path.to_owned()
}

/// Normalizes a `.proto` file text into single-header `proto3` syntax for `protox`
/// when a file uses `edition = "2023";` or nested directory import paths.
///
/// # Arguments
///
/// * `raw` - Raw `.proto` source text.
/// * `available_files` - Set of staged `.proto` basenames.
fn normalize_proto_for_protox(raw: &str, available_files: &BTreeSet<String>) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut seen_syntax = false;
    let mut seen_package = false;
    let mut seen_imports = BTreeSet::new();
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
        if trimmed.starts_with("import ")
            && let Some(first_quote) = trimmed.find('"')
            && let Some(last_quote) = trimmed.rfind('"')
            && first_quote < last_quote
        {
            let raw_import = &trimmed[first_quote + 1..last_quote];
            let rewritten = rewrite_import_path(raw_import, available_files);
            if seen_imports.insert(rewritten.clone()) {
                let _ = writeln!(out, r#"import "{rewritten}";"#);
            }
            continue;
        }
        let _ = writeln!(out, "{line}");
    }
    out
}

/// Unified build-script error enum for `libtpu-core` schema staging and compilation.
#[derive(Debug)]
enum BuildError {
    /// Filesystem I/O failure during `.proto` staging or `FileDescriptorSet` emission.
    Io(std::io::Error),
    /// `protox` schema compilation failure.
    Protox(protox::Error),
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Protox(e) => write!(f, "protox error: {e}"),
        }
    }
}

impl std::error::Error for BuildError {}

impl From<std::io::Error> for BuildError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<protox::Error> for BuildError {
    fn from(err: protox::Error) -> Self {
        Self::Protox(err)
    }
}

/// Post-processes `prost-build` generated `.rs` files in `out_dir` so every generated struct,
/// enum, module, field, and variant has a rustdoc comment satisfying `forbid(missing_docs)`.
///
/// # Arguments
///
/// * `out_dir` - Directory containing `.rs` files emitted by `prost-build`.
fn ensure_doc_comments_in_generated_rs(out_dir: &std::path::Path) -> std::io::Result<()> {
    for entry in std::fs::read_dir(out_dir)? {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let raw = std::fs::read_to_string(&path)?;
        let mut out = String::with_capacity(raw.len() + 4096);
        let mut pending_docs: Vec<&str> = Vec::new();
        let mut pending_attrs: Vec<&str> = Vec::new();
        for line in raw.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("///") || trimmed.starts_with("//!") {
                pending_docs.push(line);
                continue;
            }
            if trimmed.starts_with("#[") {
                pending_attrs.push(line);
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
            if is_item && pending_docs.is_empty() {
                let indent: String = line.chars().take_while(|c| c.is_whitespace()).collect();
                let _ = writeln!(out, "{indent}/// Generated protobuf schema item.");
            }
            for doc in std::mem::take(&mut pending_docs) {
                let _ = writeln!(out, "{doc}");
            }
            for attr in std::mem::take(&mut pending_attrs) {
                let _ = writeln!(out, "{attr}");
            }
            let _ = writeln!(out, "{line}");
        }
        for doc in pending_docs {
            let _ = writeln!(out, "{doc}");
        }
        for attr in pending_attrs {
            let _ = writeln!(out, "{attr}");
        }
        std::fs::write(&path, out)?;
    }
    Ok(())
}

/// Ordered list of `.proto` schemas in `docs/reverse_engineering/protos/` compiled by `protox`.
const COMPILE_PROTOS: &[&str] = &[
    "gpu_topology.proto",
    "backends.proto",
    "dnn.proto",
    "cuda_compute_capability.proto",
    "oneapi_compute_capability.proto",
    "tensor_shape.proto",
    "types.proto",
    "tensor.proto",
    "tpu_version.proto",
    "register_translator.proto",
    "tpu_chip_config.proto",
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
    "status.proto",
    "pjrt_abi_version.proto",
    "pjrt_device_dimensions.proto",
    "pjrt_partial_program.proto",
    "pjrt_value_type.proto",
    "pjrt_distributed_protocol.proto",
    "py_host_callback.proto",
    "pytree.proto",
    "ffi_execution_state.proto",
    "ffi_attribute_map.proto",
    "dynamic_padding.proto",
    "runtime_abi_version.proto",
    "executable_abi_version.proto",
    "sparse_core_layout.proto",
    "topology_description.proto",
    "xla_data.proto",
    "metrics.proto",
    "hlo.proto",
    "buffer_assignment.proto",
    "shaped_slice.proto",
    "execute_options.proto",
    "executable_metadata.proto",
    "xplane.proto",
    "profiler_options.proto",
    "profiler_service.proto",
    "profiler_topology.proto",
    "tf_tpu_topology.proto",
    "optimization_parameters.proto",
    "tpu_embedding_configuration.proto",
    "tpu_compilation_environment.proto",
    "tpu_topology.proto",
    "tpu_runtime.proto",
    "tpu_executable.proto",
    "tpu_input_pipeline.proto",
    "host_offloading_executable.proto",
    "megascale_dcn_topology.proto",
    "megascale_addresses.proto",
    "megascale_runtime_error_overlay.proto",
    "megascale.proto",
    "coordination_config.proto",
    "coordination_service.proto",
    "xla_compile_result.proto",
    "autotuning.proto",
    "autotune_results.proto",
    "device_description.proto",
    "xla.proto",
    "compile_options.proto",
    "compile_metadata.proto",
    "compilation_result.proto",
    "host_compute_metadata.proto",
    "tf2xla.proto",
    "xla_argument.proto",
    "tpu_executable_info.proto",
    "coredump.proto",
    "stall_summaries.proto",
    "compiler_metadata.proto",
    "mosaic_tpu.proto",
    "hlo_profile_printer_data.proto",
    "jellyfish_executable.proto",
    "barnacore_executable.proto",
    "tpu_fingerprint.proto",
    "xla_ffi.proto",
    "power_telemetry.proto",
];

/// Entrypoint for `cargo` build script execution.
///
/// # Errors
///
/// Returns [`BuildError`] if `protox` or `prost-build` fails to compile the staged schemas.
fn main() -> Result<(), BuildError> {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default());
    let proto_dir = manifest_dir
        .join("../../docs/reverse_engineering/protos")
        .canonicalize()
        .unwrap_or_else(|_| manifest_dir.join("../../docs/reverse_engineering/protos"));

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap_or_default());
    let staged_proto_dir = out_dir.join("staged_protos");
    std::fs::create_dir_all(&staged_proto_dir)?;

    let available_files: BTreeSet<String> =
        COMPILE_PROTOS.iter().map(|s| (*s).to_owned()).collect();
    let mut staged_paths = Vec::with_capacity(COMPILE_PROTOS.len());

    for &name in COMPILE_PROTOS {
        let src_path = proto_dir.join(name);
        println!("cargo:rerun-if-changed={}", src_path.display());
        let raw = std::fs::read_to_string(&src_path)?;
        let normalized = normalize_proto_for_protox(&raw, &available_files);
        let dst_path = staged_proto_dir.join(name);
        std::fs::write(&dst_path, normalized)?;
        staged_paths.push(dst_path);
    }

    let file_descriptor_set = protox::compile(&staged_paths, [&staged_proto_dir])?;
    let fds_path = out_dir.join("libtpu_file_descriptor_set.bin");
    std::fs::write(&fds_path, file_descriptor_set.encode_to_vec())?;

    let mut config = prost_build::Config::new();
    config.btree_map(["."]);
    config.file_descriptor_set_path(&fds_path);
    config.compile_fds(file_descriptor_set)?;
    ensure_doc_comments_in_generated_rs(&out_dir)?;
    Ok(())
}

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

/// Synthetic leaf dependency and runtime bridge `.proto` schemas staged alongside
/// `docs/reverse_engineering/protos/*.proto`.
const SYNTHETIC_DEPENDENCY_PROTOS: &[(&str, &str)] = &[
    (
        "gpu_topology.proto",
        r#"syntax = "proto3";
package xla;
message GpuTopologyProto {
  int32 num_partitions = 1;
  int32 num_hosts_per_partition = 2;
  int32 num_devices_per_host = 3;
}
"#,
    ),
    (
        "backends.proto",
        r#"syntax = "proto3";
package xla.autotuner;
enum Backend {
  BACKEND_UNSPECIFIED = 0;
  CPU = 1;
  GPU = 2;
  TPU = 3;
}
"#,
    ),
    (
        "dnn.proto",
        r#"syntax = "proto3";
package stream_executor.dnn;
message AlgorithmProto {
  int64 algo_id = 1;
  bool math_type = 2;
}
"#,
    ),
    (
        "cuda_compute_capability.proto",
        r#"syntax = "proto3";
package stream_executor;
message CudaComputeCapabilityProto {
  int32 major = 1;
  int32 minor = 2;
}
"#,
    ),
    (
        "oneapi_compute_capability.proto",
        r#"syntax = "proto3";
package stream_executor;
message OneAPIComputeCapabilityProto {
  int32 generation = 1;
}
"#,
    ),
    (
        "tensor_shape.proto",
        r#"syntax = "proto3";
package tensorflow;
message TensorShapeProto {
  message Dim {
    int64 size = 1;
    string name = 2;
  }
  repeated Dim dim = 2;
  bool unknown_rank = 3;
}
"#,
    ),
    (
        "types.proto",
        r#"syntax = "proto3";
package tensorflow;
enum DataType {
  DT_INVALID = 0;
  DT_FLOAT = 1;
  DT_DOUBLE = 2;
  DT_INT32 = 3;
  DT_UINT8 = 4;
  DT_INT16 = 5;
  DT_INT8 = 6;
  DT_STRING = 7;
  DT_COMPLEX64 = 8;
  DT_INT64 = 9;
  DT_BOOL = 10;
  DT_BFLOAT16 = 14;
  DT_UINT32 = 22;
  DT_UINT64 = 23;
}
"#,
    ),
    (
        "tensor.proto",
        r#"syntax = "proto3";
package tensorflow;
import "tensor_shape.proto";
import "types.proto";
message TensorProto {
  DataType dtype = 1;
  TensorShapeProto tensor_shape = 2;
  int32 version_number = 3;
  bytes tensor_content = 4;
}
"#,
    ),
    (
        "libtpu_runtime_tpu_bridge.proto",
        r#"syntax = "proto3";
package tpu;
import "xla_data.proto";

message TpuCompilationEnvironmentProto {
  bool xla_tpu_enable_megacore_fusion = 1;
  bool xla_tpu_enable_async_collective_fusion = 2;
  int64 xla_tpu_scoped_vmem_limit_kib = 3;
  string custom_flags = 4;
}

message TpuTopologyArgsProto {
  int32 version = 1;
  int32 num_hosts = 2;
  int32 chips_per_host = 3;
  repeated int64 mesh_bounds = 4;
  string platform_version = 5;
  string fingerprint = 6;
}

message TpuSerializedExecutableProto {
  string name = 1;
  string fingerprint = 2;
  int32 num_replicas = 3;
  int32 num_partitions = 4;
  repeated int64 instruction_opcodes = 5;
  repeated xla.ShapeProto output_shapes = 6;
  bytes compile_options_bytes = 7;
  bytes hlo_module_bytes = 8;
}

message TpuEmbeddingPartitionedConfigProto {
  int32 num_partitions = 1;
  repeated string table_names = 2;
  bytes raw_config_bytes = 3;
}

message TpuEmbeddingMemoryConfigProto {
  uint64 hbm_buffers_bytes = 1;
  int32 num_segments = 2;
  repeated bytes segment_blobs = 3;
}

message TpuEmbeddingHostNetworkConfigProto {
  int32 host_id = 1;
  string endpoint_address = 2;
  bytes partition_payload = 3;
}
"#,
    ),
    (
        "libtpu_runtime_xla_bridge.proto",
        r#"syntax = "proto3";
package xla;
import "deepsea_executable.proto";
import "xla.proto";
import "compile_options.proto";
import "target_arguments.proto";
import "libtpu_runtime_tpu_bridge.proto";

message TpuExecutableProto {
  DeepseaExecutableProto ds_executable = 1;
  HloModuleProtoWithConfig hlo_module_with_config = 3;
  CompileOptionsProto compile_options = 6;
  TargetArgumentsProto target_arguments = 7;
  tpu.TpuSerializedExecutableProto legacy_executable = 8;
  string source_uri = 9;
}
"#,
    ),
    (
        "libtpu_runtime_megascale_bridge.proto",
        r#"syntax = "proto3";
package megascale;

message MultiSliceConfigProto {
  int32 num_slices = 1;
  int32 slice_id = 2;
  int32 num_devices_per_slice = 3;
  string coordinator_address = 4;
}

message DCNTopologyProto {
  int32 num_slices = 1;
  int32 num_hosts_per_slice = 2;
  int32 transport_type = 3;
}

message RuntimeMetricsProto {
  uint64 completed_barriers = 1;
  uint64 collective_ops_count = 2;
  int32 active_contexts = 3;
}
"#,
    ),
];

/// Writes synthetic leaf dependency `.proto` schemas required by top-level `OpenXLA`
/// and TensorFlow `.proto` files into `staged_proto_dir`.
///
/// # Arguments
///
/// * `staged_proto_dir` - Output staging directory for `protox`.
/// * `staged_paths` - Vector of staged `.proto` paths to append to.
/// * `available_files` - Set of staged `.proto` basenames.
fn write_synthetic_dependency_protos(
    staged_proto_dir: &std::path::Path,
    staged_paths: &mut Vec<PathBuf>,
    available_files: &mut BTreeSet<String>,
) -> std::io::Result<()> {
    for &(name, content) in SYNTHETIC_DEPENDENCY_PROTOS {
        available_files.insert(name.to_owned());
        let dst = staged_proto_dir.join(name);
        std::fs::write(&dst, content)?;
        staged_paths.push(dst);
    }
    Ok(())
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
        "tf_tpu_topology.proto",
        "optimization_parameters.proto",
        "tpu_embedding_configuration.proto",
        "megascale_dcn_topology.proto",
        "megascale_addresses.proto",
        "megascale_runtime_error_overlay.proto",
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
    ];

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap_or_default());
    let staged_proto_dir = out_dir.join("staged_protos");
    std::fs::create_dir_all(&staged_proto_dir)?;

    let mut available_files: BTreeSet<String> =
        compile_protos.iter().map(|s| (*s).to_owned()).collect();
    let mut staged_paths = Vec::with_capacity(compile_protos.len() + 12);

    write_synthetic_dependency_protos(&staged_proto_dir, &mut staged_paths, &mut available_files)?;

    for name in compile_protos {
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
    config.file_descriptor_set_path(&fds_path);
    config.compile_fds(file_descriptor_set)?;
    Ok(())
}

//! TPU topology, compilation environment, and `SparseCore` embedding protobuf messages.
//!
//! Synchronized with `tpu_topology.proto`, `tpu_embedding_configuration.proto`, and `sparse_core_layout.proto`.

use crate::proto::descriptor::impl_proto_schema_message;
use crate::proto::hlo::{HloModuleProto, HloModuleProtoWithConfig};
use crate::proto::xla_data::{DeviceAssignmentProto, OpShardingProto, ShapeProto};
use prost::Message;

/// Wire-compatible `xla.jellyfish.MsaTensorOverrides` (`memory_space_assignment.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct MsaTensorOverridesProto {
    /// HLO instruction names pinned to alternate memory (VMEM, field `1`).
    #[prost(string, repeated, tag = "1")]
    pub pinned_hlo_instruction_names: Vec<String>,
    /// Maximum alternate memory (VMEM) bytes override (field `2`).
    #[prost(int64, tag = "2")]
    pub max_alternate_memory_bytes: i64,
}

impl_proto_schema_message!(MsaTensorOverridesProto, "xla.jellyfish.MsaTensorOverrides");

/// Wire-compatible `xla.jellyfish.MemoryBoundLoopOptimizerOptions` (`memory_space_assignment.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct MemoryBoundLoopOptimizerOptionsProto {
    /// Whether the memory-bound loop optimizer is enabled (field `1`).
    #[prost(bool, tag = "1")]
    pub enabled: bool,
    /// Desired async copy-to-compute ratio (field `2`).
    #[prost(int32, tag = "2")]
    pub desired_copy_ratio: i32,
    /// Whether unsatisfied prefetches are allowed (field `3`).
    #[prost(bool, tag = "3")]
    pub allow_unsatisfied_prefetch: bool,
}

impl_proto_schema_message!(
    MemoryBoundLoopOptimizerOptionsProto,
    "xla.jellyfish.MemoryBoundLoopOptimizerOptions"
);

/// Wire-compatible `xla.jellyfish.PreferredPrefetchOverrides` (`memory_space_assignment.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct PreferredPrefetchOverridesProto {
    /// Minimum asynchronous DMA copy interval (field `1`).
    #[prost(double, tag = "1")]
    pub min_async_copy_interval: f64,
    /// Maximum asynchronous DMA copy interval (field `2`).
    #[prost(double, tag = "2")]
    pub max_async_copy_interval: f64,
}

impl_proto_schema_message!(
    PreferredPrefetchOverridesProto,
    "xla.jellyfish.PreferredPrefetchOverrides"
);

/// Wire-compatible `xla.jellyfish.MsaSortOrderOverrides` (`memory_space_assignment.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct MsaSortOrderOverridesProto {
    /// Prioritize larger tensors during VMEM allocation (field `1`).
    #[prost(bool, tag = "1")]
    pub prioritize_larger_tensors: bool,
    /// Prioritize longer live ranges during VMEM allocation (field `2`).
    #[prost(bool, tag = "2")]
    pub prioritize_longer_live_ranges: bool,
}

impl_proto_schema_message!(
    MsaSortOrderOverridesProto,
    "xla.jellyfish.MsaSortOrderOverrides"
);

/// Wire-compatible `xla.jellyfish.MemorySpaceAssignmentOptions` (`memory_space_assignment.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct MemorySpaceAssignmentOptionsProto {
    /// Scoped alternate memory mode enum (`ExpandedScopedAlternateMemoryMode`, field `1`).
    #[prost(int32, tag = "1")]
    pub expanded_scoped_alternate_memory_mode: i32,
    /// Per-tensor VMEM pinning overrides (field `2`).
    #[prost(message, optional, tag = "2")]
    pub tensor_overrides: Option<MsaTensorOverridesProto>,
    /// Memory-bound loop optimizer options (field `3`).
    #[prost(message, optional, tag = "3")]
    pub memory_bound_loop_optimizer_options: Option<MemoryBoundLoopOptimizerOptionsProto>,
    /// Preferred prefetch interval overrides (field `4`).
    #[prost(message, optional, tag = "4")]
    pub preferred_prefetch_overrides: Option<PreferredPrefetchOverridesProto>,
    /// Allocation candidate sort order overrides (field `5`).
    #[prost(message, optional, tag = "5")]
    pub sort_order_overrides: Option<MsaSortOrderOverridesProto>,
}

impl_proto_schema_message!(
    MemorySpaceAssignmentOptionsProto,
    "xla.jellyfish.MemorySpaceAssignmentOptions"
);

/// Wire-compatible `xla.MemorySchedulerProto` (`memory_scheduler.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct MemorySchedulerProto {
    /// Scheduler algorithm enum (`MemorySchedulerAlgorithmProto`, field `1`).
    #[prost(int32, tag = "1")]
    pub algorithm: i32,
    /// Memory limit in bytes (field `2`).
    #[prost(int64, tag = "2")]
    pub memory_limit_bytes: i64,
}

impl_proto_schema_message!(MemorySchedulerProto, "xla.MemorySchedulerProto");

/// Wire-compatible `xla.jellyfish.EmitterLearnedCostModelOptions` (`emitter_learned_cost_model_options.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct EmitterLearnedCostModelOptionsProto {
    /// Whether the learned cost model is enabled (field `1`).
    #[prost(bool, tag = "1")]
    pub enabled: bool,
    /// Cost model version string (field `2`).
    #[prost(string, tag = "2")]
    pub model_version: String,
    /// Confidence threshold in `[0.0, 1.0]` (field `3`).
    #[prost(double, tag = "3")]
    pub confidence_threshold: f64,
}

impl_proto_schema_message!(
    EmitterLearnedCostModelOptionsProto,
    "xla.jellyfish.EmitterLearnedCostModelOptions"
);

/// Static descriptor specification for a single `TpuCompilationEnvironmentProto` compiler flag (`JellyfishCompilationEnvFieldSpec`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JellyfishCompilationEnvFieldSpec {
    /// Protobuf field tag number (`1..=256`).
    pub tag: u32,
    /// Canonical compiler flag name without leading `--` (`"xla_tpu_*"`).
    pub name: &'static str,
    /// Protobuf wire type name (`"bool"`, `"int32"`, `"int64"`, `"double"`, `"string"`).
    pub wire_type: &'static str,
    /// Canonical default value string in `libtpu.so` (`v0.0.52`).
    pub default_value: &'static str,
}

/// Canonical table of extracted `tpu::TpuCompilationEnvironmentProto` compiler flags (`JELLYFISH_COMPILATION_ENV_FIELD_SPECS`).
pub static JELLYFISH_COMPILATION_ENV_FIELD_SPECS: [JellyfishCompilationEnvFieldSpec; 40] = [
    JellyfishCompilationEnvFieldSpec {
        tag: 1,
        name: "xla_tpu_enable_megacore_fusion",
        wire_type: "bool",
        default_value: "true",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 2,
        name: "xla_tpu_enable_async_collective_fusion",
        wire_type: "bool",
        default_value: "true",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 3,
        name: "xla_tpu_scoped_vmem_limit_kib",
        wire_type: "int64",
        default_value: "16384",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 4,
        name: "custom_flags",
        wire_type: "string",
        default_value: "",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 5,
        name: "xla_tpu_enable_latency_hiding_scheduler",
        wire_type: "bool",
        default_value: "true",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 6,
        name: "xla_tpu_overlap_compute_collective_tc",
        wire_type: "bool",
        default_value: "true",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 7,
        name: "xla_tpu_use_tc_device_shape_on_sc",
        wire_type: "bool",
        default_value: "false",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 8,
        name: "xla_tpu_enable_flash_attention",
        wire_type: "bool",
        default_value: "true",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 9,
        name: "xla_tpu_spmd_rng_bit_generator_unsafe",
        wire_type: "bool",
        default_value: "false",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 13,
        name: "xla_tpu_use_shardy_partitioner",
        wire_type: "bool",
        default_value: "false",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 14,
        name: "xla_tpu_enable_sparse_core_collective_offload",
        wire_type: "bool",
        default_value: "false",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 15,
        name: "xla_tpu_enable_async_collective_fusion_multiple_steps",
        wire_type: "bool",
        default_value: "false",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 16,
        name: "xla_tpu_overlap_compute_collective_sc",
        wire_type: "bool",
        default_value: "false",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 17,
        name: "xla_tpu_enable_cross_program_prefetch",
        wire_type: "bool",
        default_value: "false",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 18,
        name: "xla_tpu_enable_rematerialization",
        wire_type: "bool",
        default_value: "false",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 19,
        name: "xla_tpu_mosaic_enable_auto_pipeline",
        wire_type: "bool",
        default_value: "false",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 20,
        name: "xla_tpu_nd_short_transfer_max_chunks",
        wire_type: "int32",
        default_value: "8",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 21,
        name: "xla_tpu_ici_all_reduce_combine_threshold_bytes",
        wire_type: "int64",
        default_value: "33554432",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 22,
        name: "xla_tpu_dcn_all_reduce_combine_threshold_bytes",
        wire_type: "int64",
        default_value: "67108864",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 23,
        name: "xla_tpu_spmd_threshold_for_windowed_einsum_mib",
        wire_type: "int64",
        default_value: "256",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 24,
        name: "xla_tpu_bf16_emission_mode",
        wire_type: "string",
        default_value: "DEFAULT",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 25,
        name: "xla_tpu_autotune_level",
        wire_type: "int32",
        default_value: "2",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 26,
        name: "xla_tpu_xplane_trace_level",
        wire_type: "int32",
        default_value: "1",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 27,
        name: "xla_tpu_enable_windowed_einsum",
        wire_type: "bool",
        default_value: "true",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 28,
        name: "xla_tpu_enable_ag_backward_pipelining",
        wire_type: "bool",
        default_value: "true",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 29,
        name: "xla_tpu_sparse_core_max_ids_per_partition",
        wire_type: "int32",
        default_value: "1024",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 30,
        name: "xla_tpu_sparse_core_max_unique_ids_per_partition",
        wire_type: "int32",
        default_value: "512",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 31,
        name: "xla_tpu_enable_megascale_pjrt_collectives",
        wire_type: "bool",
        default_value: "true",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 32,
        name: "xla_tpu_enable_ici_compression",
        wire_type: "bool",
        default_value: "false",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 33,
        name: "xla_tpu_cmem_limit_kib",
        wire_type: "int64",
        default_value: "0",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 34,
        name: "xla_tpu_enable_llo_software_pipelining",
        wire_type: "bool",
        default_value: "true",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 35,
        name: "xla_tpu_llo_modulo_scheduler_ii_limit",
        wire_type: "int32",
        default_value: "16",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 36,
        name: "xla_tpu_enable_2d_all_gather_collective_matmul",
        wire_type: "bool",
        default_value: "false",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 37,
        name: "xla_tpu_enable_rs_forward_pipelining",
        wire_type: "bool",
        default_value: "true",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 38,
        name: "xla_tpu_enable_dcn_zstd_compression",
        wire_type: "bool",
        default_value: "false",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 39,
        name: "xla_tpu_fp8_quantization_mode",
        wire_type: "string",
        default_value: "E4M3FN",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 40,
        name: "xla_tpu_int4_weight_only_quantization",
        wire_type: "bool",
        default_value: "true",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 100,
        name: "xla_jf_auto_cross_replica_sharding",
        wire_type: "bool",
        default_value: "true",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 200,
        name: "xla_tpu_megascale_hang_detection_timeout_ms",
        wire_type: "int64",
        default_value: "30000",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 256,
        name: "xla_tpu_runtime_abi_enforcement_strict",
        wire_type: "bool",
        default_value: "true",
    },
];

/// Wire-compatible `tpu.TpuCompilationEnvironmentProto` (`tpu_compilation_environment.proto`).
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, PartialEq, Message)]
pub struct TpuCompilationEnvironmentProto {
    /// XLA TPU Megacore fusion toggle (field `1`).
    #[prost(bool, tag = "1")]
    pub xla_tpu_enable_megacore_fusion: bool,
    /// XLA TPU async collective fusion toggle (field `2`).
    #[prost(bool, tag = "2")]
    pub xla_tpu_enable_async_collective_fusion: bool,
    /// XLA TPU VMEM limit in KiB (field `3`).
    #[prost(int64, tag = "3")]
    pub xla_tpu_scoped_vmem_limit_kib: i64,
    /// Raw custom flag string overrides (field `4`).
    #[prost(string, tag = "4")]
    pub custom_flags: String,
    /// XLA TPU latency-hiding scheduler toggle (field `5`).
    #[prost(bool, tag = "5")]
    pub xla_tpu_enable_latency_hiding_scheduler: bool,
    /// Overlap `TensorCore` compute with ICI/DCN collectives (field `6`).
    #[prost(bool, tag = "6")]
    pub xla_tpu_overlap_compute_collective_tc: bool,
    /// Use `TensorCore` device shape on `SparseCore` (field `7`).
    #[prost(bool, tag = "7")]
    pub xla_tpu_use_tc_device_shape_on_sc: bool,
    /// Enable hardware `FlashAttention` fusion pass (field `8`).
    #[prost(bool, tag = "8")]
    pub xla_tpu_enable_flash_attention: bool,
    /// Allow unsafe SPMD RNG bit generator partitioning (field `9`).
    #[prost(bool, tag = "9")]
    pub xla_tpu_spmd_rng_bit_generator_unsafe: bool,
    /// Memory Space Assignment (`MSA`) options (field `10`).
    #[prost(message, optional, tag = "10")]
    pub memory_space_assignment_options: Option<MemorySpaceAssignmentOptionsProto>,
    /// HLO memory scheduler options (field `11`).
    #[prost(message, optional, tag = "11")]
    pub memory_scheduler_options: Option<MemorySchedulerProto>,
    /// Learned cost model options for LLO emitter (field `12`).
    #[prost(message, optional, tag = "12")]
    pub learned_cost_model_options: Option<EmitterLearnedCostModelOptionsProto>,
    /// Enable `OpenXLA` Shardy (`sdy`) SPMD partitioner (field `13`).
    #[prost(bool, tag = "13")]
    pub xla_tpu_use_shardy_partitioner: bool,
    /// Offload collective operations onto `SparseCore` engines (field `14`).
    #[prost(bool, tag = "14")]
    pub xla_tpu_enable_sparse_core_collective_offload: bool,
    /// Allow multi-step async collective fusion across loop iterations (field `15`).
    #[prost(bool, tag = "15")]
    pub xla_tpu_enable_async_collective_fusion_multiple_steps: bool,
    /// Overlap `SparseCore` compute with collective transfers (field `16`).
    #[prost(bool, tag = "16")]
    pub xla_tpu_overlap_compute_collective_sc: bool,
    /// Enable cross-program HBM -> VMEM prefetching (field `17`).
    #[prost(bool, tag = "17")]
    pub xla_tpu_enable_cross_program_prefetch: bool,
    /// Enable HLO rematerialization under VMEM/HBM pressure (field `18`).
    #[prost(bool, tag = "18")]
    pub xla_tpu_enable_rematerialization: bool,
    /// Enable automatic DMA/compute software pipelining in Mosaic kernels (field `19`).
    #[prost(bool, tag = "19")]
    pub xla_tpu_mosaic_enable_auto_pipeline: bool,
    /// Maximum chunk count for N-D short DMA transfers (field `20`).
    #[prost(int32, tag = "20")]
    pub xla_tpu_nd_short_transfer_max_chunks: i32,
    /// ICI `AllReduce` combiner threshold in bytes (field `21`).
    #[prost(int64, tag = "21")]
    pub xla_tpu_ici_all_reduce_combine_threshold_bytes: i64,
    /// DCN `AllReduce` combiner threshold in bytes (field `22`).
    #[prost(int64, tag = "22")]
    pub xla_tpu_dcn_all_reduce_combine_threshold_bytes: i64,
    /// Activation threshold in MiB for windowed einsum collective pipelining (field `23`).
    #[prost(int64, tag = "23")]
    pub xla_tpu_spmd_threshold_for_windowed_einsum_mib: i64,
    /// BF16 systolic emission mode string (e.g., `"DEFAULT"`, `"HIGH"`, `"HIGHEST"`, field `24`).
    #[prost(string, tag = "24")]
    pub xla_tpu_bf16_emission_mode: String,
    /// Compiler autotuning effort level (`0..=3`, field `25`).
    #[prost(int32, tag = "25")]
    pub xla_tpu_autotune_level: i32,
    /// `XPlane` hardware profiler trace verbosity level (`0..=3`, field `26`).
    #[prost(int32, tag = "26")]
    pub xla_tpu_xplane_trace_level: i32,
    /// Enable windowed einsum collective matmul pipelining (field `27`).
    #[prost(bool, tag = "27")]
    pub xla_tpu_enable_windowed_einsum: bool,
    /// Enable `AllGather` backward collective pipelining (field `28`).
    #[prost(bool, tag = "28")]
    pub xla_tpu_enable_ag_backward_pipelining: bool,
    /// Maximum sparse IDs per `SparseCore` partition (field `29`).
    #[prost(int32, tag = "29")]
    pub xla_tpu_sparse_core_max_ids_per_partition: i32,
    /// Maximum unique sparse IDs per `SparseCore` partition (field `30`).
    #[prost(int32, tag = "30")]
    pub xla_tpu_sparse_core_max_unique_ids_per_partition: i32,
    /// Enable `MegaScale` PJRT multi-slice collectives (field `31`).
    #[prost(bool, tag = "31")]
    pub xla_tpu_enable_megascale_pjrt_collectives: bool,
    /// Enable optical ICI payload compression (field `32`).
    #[prost(bool, tag = "32")]
    pub xla_tpu_enable_ici_compression: bool,
    /// Scoped CMEM limit in KiB (field `33`).
    #[prost(int64, tag = "33")]
    pub xla_tpu_cmem_limit_kib: i64,
    /// Enable LLO modulo scheduler software pipelining (field `34`).
    #[prost(bool, tag = "34")]
    pub xla_tpu_enable_llo_software_pipelining: bool,
    /// Maximum initiation interval (`II`) limit for LLO modulo scheduling (field `35`).
    #[prost(int32, tag = "35")]
    pub xla_tpu_llo_modulo_scheduler_ii_limit: i32,
    /// Enable 2D `AllGather` collective matmul decomposition (field `36`).
    #[prost(bool, tag = "36")]
    pub xla_tpu_enable_2d_all_gather_collective_matmul: bool,
}

impl_proto_schema_message!(
    TpuCompilationEnvironmentProto,
    "tpu.TpuCompilationEnvironmentProto"
);

impl TpuCompilationEnvironmentProto {
    /// Looks up a [`JellyfishCompilationEnvFieldSpec`] by flag name or numeric field tag (`1..=256`).
    ///
    /// # Arguments
    ///
    /// * `query` - Flag name (with or without leading `--`) or numeric tag string.
    #[must_use]
    pub fn lookup_jellyfish_field_spec(
        query: &str,
    ) -> Option<&'static JellyfishCompilationEnvFieldSpec> {
        let normalized = query.trim().trim_start_matches('-');
        let tag_opt = normalized.parse::<u32>().ok();
        JELLYFISH_COMPILATION_ENV_FIELD_SPECS
            .iter()
            .find(|spec| spec.name == normalized || tag_opt == Some(spec.tag))
    }

    /// Looks up a compiler flag field by name or numeric tag (`1..=26`) on this [`TpuCompilationEnvironmentProto`]
    /// via `prost_reflect::DynamicMessage` reflection or in the `xla.jellyfish` descriptor schemas,
    /// returning its string representation (matching `PJRT_TpuExecutable_GetTpuCompilationEnvFieldAsString`).
    ///
    /// # Arguments
    ///
    /// * `field_query` - Field name (e.g., `"xla_tpu_enable_megacore_fusion"` or `"--xla_tpu_use_shardy_partitioner"`) or field tag (`"1"`..`"256"`).
    #[must_use]
    pub fn get_field_as_string(&self, field_query: &str) -> String {
        let normalized = field_query.trim().trim_start_matches('-');
        let pool = crate::proto::descriptor::global_descriptor_pool();
        let field_opt = pool
            .get_message_by_name("tpu.TpuCompilationEnvironmentProto")
            .and_then(|desc| {
                normalized
                    .parse::<u32>()
                    .ok()
                    .and_then(|tag| desc.get_field(tag))
                    .or_else(|| desc.get_field_by_name(normalized))
            });
        if let Some(field_desc) = field_opt {
            let parent = field_desc.parent_message().clone();
            let mut dyn_msg = prost_reflect::DynamicMessage::new(parent.clone());
            let _ = prost_reflect::DynamicMessage::decode(parent, self.encode_to_vec().as_slice())
                .map(|decoded| dyn_msg = decoded);
            let val = dyn_msg.get_field(&field_desc);
            return match val.as_ref() {
                prost_reflect::Value::Bool(b) => b.to_string(),
                prost_reflect::Value::I32(i) => i.to_string(),
                prost_reflect::Value::I64(i) => i.to_string(),
                prost_reflect::Value::String(s) => s.clone(),
                other => format!("{other:?}"),
            };
        }
        if let Some(val) = self
            .custom_flags
            .split(',')
            .filter_map(|entry| entry.split_once('='))
            .find_map(|(k, v)| {
                (k.trim().trim_start_matches('-') == normalized).then(|| String::from(v.trim()))
            })
        {
            return val;
        }
        if let Some(spec) = Self::lookup_jellyfish_field_spec(normalized) {
            return String::from(spec.default_value);
        }
        for msg_name in [
            "xla.tpu.TpuCompilationEnvironment",
            "tpu.TpuCompilationEnvironmentProto",
            "xla.jellyfish.EmitterLearnedCostModelOptions",
            "xla.jellyfish.MemorySpaceAssignmentOptions",
        ] {
            if let Some(msg_desc) = pool.get_message_by_name(msg_name)
                && let Some(field_desc) = msg_desc.get_field_by_name(normalized)
            {
                let default_msg = prost_reflect::DynamicMessage::new(msg_desc);
                let default_val = default_msg.get_field(&field_desc);
                return format!("{default_val:?}");
            }
        }
        String::from("default")
    }

    /// Alias for [`Self::get_field_as_string`] that resolves any `--xla_tpu_*` or `--xla_jf_*`
    /// compiler flag by name or field number.
    ///
    /// # Arguments
    ///
    /// * `flag_name` - Flag name (with or without leading `--`) or field tag.
    #[must_use]
    pub fn get_flag_by_name(&self, flag_name: &str) -> String {
        self.get_field_as_string(flag_name)
    }

    /// Alias for [`Self::set_field_from_str`] that updates or records any `--xla_tpu_*` or `--xla_jf_*`
    /// compiler flag override by name or field number.
    ///
    /// # Arguments
    ///
    /// * `flag_name` - Flag name (with or without leading `--`) or field tag.
    /// * `value_str` - String representation of the new flag value.
    pub fn set_flag_by_name(&mut self, flag_name: &str, value_str: &str) {
        self.set_field_from_str(flag_name, value_str);
    }

    /// Updates a compiler flag field by name or numeric tag (`1..=26`) from a string value via
    /// `prost_reflect::DynamicMessage` reflection, or appends a `key=value` override to `custom_flags`
    /// for unrecognized custom flags.
    ///
    /// # Arguments
    ///
    /// * `field_name` - Flag name or numeric field tag.
    /// * `value_str` - String representation of the new flag value.
    pub fn set_field_from_str(&mut self, field_name: &str, value_str: &str) {
        let normalized = field_name.trim().trim_start_matches('-');
        let val = value_str.trim();
        let is_bool_true = matches!(val, "true" | "1" | "TRUE" | "yes");
        let pool = crate::proto::descriptor::global_descriptor_pool();
        let field_opt = pool
            .get_message_by_name("tpu.TpuCompilationEnvironmentProto")
            .and_then(|desc| {
                normalized
                    .parse::<u32>()
                    .ok()
                    .and_then(|tag| desc.get_field(tag))
                    .or_else(|| desc.get_field_by_name(normalized))
            });
        if let Some(field_desc) = field_opt {
            let parent = field_desc.parent_message().clone();
            let mut dyn_msg = prost_reflect::DynamicMessage::new(parent.clone());
            let _ = prost_reflect::DynamicMessage::decode(parent, self.encode_to_vec().as_slice())
                .map(|decoded| dyn_msg = decoded);
            let updated = match field_desc.kind() {
                prost_reflect::Kind::Bool => {
                    dyn_msg.set_field(&field_desc, prost_reflect::Value::Bool(is_bool_true));
                    true
                }
                prost_reflect::Kind::Int32 => {
                    let parsed = val.parse::<i32>().unwrap_or(0);
                    dyn_msg.set_field(&field_desc, prost_reflect::Value::I32(parsed));
                    true
                }
                prost_reflect::Kind::Int64 => {
                    let default_i64 = if field_desc.number() == 3 { 16_384 } else { 0 };
                    let parsed = val.parse::<i64>().unwrap_or(default_i64);
                    dyn_msg.set_field(&field_desc, prost_reflect::Value::I64(parsed));
                    true
                }
                prost_reflect::Kind::String => {
                    dyn_msg.set_field(&field_desc, prost_reflect::Value::String(String::from(val)));
                    true
                }
                _ => false,
            };
            if updated {
                *self = Self::decode(dyn_msg.encode_to_vec().as_slice()).unwrap_or_default();
                return;
            }
        }
        if !self.custom_flags.is_empty() {
            self.custom_flags.push(',');
        }
        self.custom_flags.push_str(normalized);
        self.custom_flags.push('=');
        self.custom_flags.push_str(val);
    }

    /// Parses a whitespace- or comma-delimited CLI flag string (e.g., `"--xla_tpu_enable_megacore_fusion=true --xla_tpu_scoped_vmem_limit_kib=32768"`)
    /// into a [`TpuCompilationEnvironmentProto`], resolving numeric tags or field names against
    /// `tpu.TpuCompilationEnvironmentProto` in [`crate::proto::descriptor::global_descriptor_pool`].
    ///
    /// # Arguments
    ///
    /// * `cli_flags` - Raw CLI flag string.
    #[must_use]
    pub fn from_cli_flags(cli_flags: &str) -> Self {
        let mut env = Self {
            xla_tpu_enable_megacore_fusion: true,
            xla_tpu_enable_async_collective_fusion: true,
            xla_tpu_scoped_vmem_limit_kib: 16_384,
            custom_flags: String::new(),
            xla_tpu_enable_latency_hiding_scheduler: true,
            xla_tpu_overlap_compute_collective_tc: true,
            xla_tpu_use_tc_device_shape_on_sc: false,
            xla_tpu_enable_flash_attention: true,
            xla_tpu_spmd_rng_bit_generator_unsafe: false,
            memory_space_assignment_options: None,
            memory_scheduler_options: None,
            learned_cost_model_options: None,
            xla_tpu_use_shardy_partitioner: false,
            xla_tpu_enable_sparse_core_collective_offload: false,
            xla_tpu_enable_async_collective_fusion_multiple_steps: false,
            xla_tpu_overlap_compute_collective_sc: false,
            xla_tpu_enable_cross_program_prefetch: false,
            xla_tpu_enable_rematerialization: false,
            xla_tpu_mosaic_enable_auto_pipeline: false,
            xla_tpu_nd_short_transfer_max_chunks: 8,
            xla_tpu_ici_all_reduce_combine_threshold_bytes: 33_554_432,
            xla_tpu_dcn_all_reduce_combine_threshold_bytes: 67_108_864,
            xla_tpu_spmd_threshold_for_windowed_einsum_mib: 256,
            xla_tpu_bf16_emission_mode: String::from("DEFAULT"),
            xla_tpu_autotune_level: 2,
            xla_tpu_xplane_trace_level: 1,
            xla_tpu_enable_windowed_einsum: true,
            xla_tpu_enable_ag_backward_pipelining: true,
            xla_tpu_sparse_core_max_ids_per_partition: 1024,
            xla_tpu_sparse_core_max_unique_ids_per_partition: 512,
            xla_tpu_enable_megascale_pjrt_collectives: true,
            xla_tpu_enable_ici_compression: false,
            xla_tpu_cmem_limit_kib: 0,
            xla_tpu_enable_llo_software_pipelining: true,
            xla_tpu_llo_modulo_scheduler_ii_limit: 16,
            xla_tpu_enable_2d_all_gather_collective_matmul: false,
        };
        for token in cli_flags.split(|c: char| c.is_whitespace() || c == ',') {
            let trimmed = token.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Some((k, v)) = trimmed.split_once('=') {
                env.set_field_from_str(k, v);
            } else {
                env.set_field_from_str(trimmed, "true");
            }
        }
        env
    }

    /// Applies this [`TpuCompilationEnvironmentProto`] onto a mutable [`crate::config::TpuInitConfig`].
    ///
    /// # Arguments
    ///
    /// * `config` - Target [`crate::config::TpuInitConfig`] to update.
    pub fn apply_to_init_config(&self, config: &mut crate::config::TpuInitConfig) {
        config.extra_flags.insert(
            String::from("xla_tpu_enable_megacore_fusion"),
            self.xla_tpu_enable_megacore_fusion.to_string(),
        );
        config.extra_flags.insert(
            String::from("xla_tpu_enable_async_collective_fusion"),
            self.xla_tpu_enable_async_collective_fusion.to_string(),
        );
        config.extra_flags.insert(
            String::from("xla_tpu_scoped_vmem_limit_kib"),
            self.xla_tpu_scoped_vmem_limit_kib.to_string(),
        );
        for (k, v) in self
            .custom_flags
            .split(',')
            .filter_map(|entry| entry.split_once('='))
        {
            config
                .extra_flags
                .insert(String::from(k.trim()), String::from(v.trim()));
        }
    }

    /// Lists all field names registered in the `TpuCompilationEnvironmentProto` and `xla.jellyfish` descriptors.
    #[must_use]
    pub fn list_jellyfish_schema_field_names() -> Vec<String> {
        let pool = crate::proto::descriptor::global_descriptor_pool();
        let mut fields = Vec::new();
        for msg_name in [
            "xla.tpu.TpuCompilationEnvironment",
            "tpu.TpuCompilationEnvironmentProto",
            "xla.jellyfish.EmitterLearnedCostModelOptions",
            "xla.jellyfish.MemorySpaceAssignmentOptions",
        ] {
            if let Some(desc) = pool.get_message_by_name(msg_name) {
                for f in desc.fields() {
                    fields.push(String::from(f.name()));
                }
            }
        }
        fields
    }
}

/// Wire-compatible `tpu.TpuTopologyArgsProto` (`tpu_topology.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuTopologyArgsProto {
    /// TPU platform generation enum (`TpuVersionEnum`, field `1`).
    #[prost(int32, tag = "1")]
    pub version: i32,
    /// Number of hosts in the pod slice (field `2`).
    #[prost(int32, tag = "2")]
    pub num_hosts: i32,
    /// Number of chips per host (field `3`).
    #[prost(int32, tag = "3")]
    pub chips_per_host: i32,
    /// Mesh dimensions `[x, y, z]` (field `4`).
    #[prost(int64, repeated, packed = "true", tag = "4")]
    pub mesh_bounds: Vec<i64>,
    /// Platform version string (field `5`).
    #[prost(string, tag = "5")]
    pub platform_version: String,
    /// Deterministic topology fingerprint (field `6`).
    #[prost(string, tag = "6")]
    pub fingerprint: String,
}

impl_proto_schema_message!(TpuTopologyArgsProto, "tpu.TpuTopologyArgsProto");

/// Wire-compatible `tensorflow.tpu.TPUHardwareFeature` (`tensorflow/core/protobuf/tpu/topology.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuHardwareFeatureProto {
    /// Embedding feature version enum (field `1`).
    #[prost(int32, tag = "1")]
    pub embedding_feature: i32,
    /// Number of embedding (`SparseCore`) devices per chip (field `2`).
    #[prost(int32, tag = "2")]
    pub num_embedding_devices_per_chip: i32,
}

impl_proto_schema_message!(TpuHardwareFeatureProto, "tensorflow.tpu.TPUHardwareFeature");

/// Wire-compatible `tensorflow.tpu.TopologyProto` (`tensorflow/core/protobuf/tpu/topology.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TopologyProto {
    /// Mesh shape `[x, y, z, core]` (field `1`).
    #[prost(int32, repeated, packed = "true", tag = "1")]
    pub mesh_shape: Vec<i32>,
    /// Number of TensorFlow tasks/hosts in the cluster (field `2`).
    #[prost(int32, tag = "2")]
    pub num_tasks: i32,
    /// Number of TPU devices per task (field `3`).
    #[prost(int32, tag = "3")]
    pub num_tpu_devices_per_task: i32,
    /// Flattened `(x, y, z, core)` device coordinates (field `4`).
    #[prost(int32, repeated, packed = "true", tag = "4")]
    pub device_coordinates: Vec<i32>,
    /// Hardware feature capabilities (field `5`).
    #[prost(message, optional, tag = "5")]
    pub tpu_hardware_feature: Option<TpuHardwareFeatureProto>,
}

impl_proto_schema_message!(TopologyProto, "tensorflow.tpu.TopologyProto");

/// Wire-compatible `tensorflow.tpu.TPUEmbeddingConfiguration.TableDescriptor`.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct EmbeddingTableDescriptorProto {
    /// Table name (field `1`).
    #[prost(string, tag = "1")]
    pub name: String,
    /// Vocabulary size (number of rows, field `2`).
    #[prost(int64, tag = "2")]
    pub vocabulary_size: i64,
    /// Embedding dimension width (field `3`).
    #[prost(int64, tag = "3")]
    pub dimension: i64,
    /// Number of features referencing this table (field `4`).
    #[prost(int64, tag = "4")]
    pub num_features: i64,
}

impl_proto_schema_message!(
    EmbeddingTableDescriptorProto,
    "tensorflow.tpu.TPUEmbeddingConfiguration.TableDescriptor"
);

/// Wire-compatible `tensorflow.tpu.TPUEmbeddingConfiguration.FeatureDescriptor`.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct EmbeddingFeatureDescriptorProto {
    /// Feature name (field `1`).
    #[prost(string, tag = "1")]
    pub name: String,
    /// Index of the corresponding table in `table_descriptor` (field `2`).
    #[prost(int32, tag = "2")]
    pub table_id: i32,
    /// Input activation shape dimensions (field `3`).
    #[prost(int32, repeated, packed = "true", tag = "3")]
    pub input_shape: Vec<i32>,
}

impl_proto_schema_message!(
    EmbeddingFeatureDescriptorProto,
    "tensorflow.tpu.TPUEmbeddingConfiguration.FeatureDescriptor"
);

/// Wire-compatible `tensorflow.tpu.TPUEmbeddingConfiguration.SpmdSharding`.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct EmbeddingSpmdShardingProto {
    /// Whether SPMD sharding is enabled (field `1`).
    #[prost(bool, tag = "1")]
    pub enabled: bool,
    /// Number of cores per replica in SPMD mode (field `2`).
    #[prost(int32, tag = "2")]
    pub num_cores_per_replica: i32,
    /// Whether manual SPMD partitioning is enabled (field `3`).
    #[prost(bool, tag = "3")]
    pub use_manual_partitioning: bool,
    /// Whether the Shardy partitioner is enabled (field `4`).
    #[prost(bool, tag = "4")]
    pub use_shardy_partitioner: bool,
}

impl_proto_schema_message!(
    EmbeddingSpmdShardingProto,
    "tensorflow.tpu.TPUEmbeddingConfiguration.SpmdSharding"
);

/// Wire-compatible `tensorflow.tpu.TPUEmbeddingConfiguration` (`xla/tpu/protobuf/tpu_embedding_configuration.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuEmbeddingConfigurationProto {
    /// Table descriptors (field `1`).
    #[prost(message, repeated, tag = "1")]
    pub table_descriptor: Vec<EmbeddingTableDescriptorProto>,
    /// Mode (`1` = Inference, `2` = Training, `3` = `BackwardPassOnly`, field `2`).
    #[prost(int32, tag = "2")]
    pub mode: i32,
    /// Batch size per `TensorCore` (field `3`).
    #[prost(int32, tag = "3")]
    pub batch_size_per_tensor_core: i32,
    /// Number of hosts (field `4`).
    #[prost(int32, tag = "4")]
    pub num_hosts: i32,
    /// Number of `TensorCores` (field `5`).
    #[prost(int32, tag = "5")]
    pub num_tensor_cores: i32,
    /// Table sharding strategy (`0` = Unspecified, `1` = Mod, `2` = Div, field `6`).
    #[prost(int32, tag = "6")]
    pub sharding_strategy: i32,
    /// Whether `SparseCore` execution is pipelined with `TensorCore` (field `7`).
    #[prost(bool, tag = "7")]
    pub pipeline_execution_with_tensor_core: bool,
    /// Directory for embedding profile data dumps (field `9`).
    #[prost(string, tag = "9")]
    pub profile_data_directory: String,
    /// Feature descriptors (field `10`).
    #[prost(message, repeated, tag = "10")]
    pub feature_descriptor: Vec<EmbeddingFeatureDescriptorProto>,
    /// Optional SPMD sharding configuration (field `11`).
    #[prost(message, optional, tag = "11")]
    pub spmd_sharding: Option<EmbeddingSpmdShardingProto>,
}

impl_proto_schema_message!(
    TpuEmbeddingConfigurationProto,
    "tensorflow.tpu.TPUEmbeddingConfiguration"
);

/// Wire-compatible `tensorflow.tpu.SparseCoreTableLayout` (`tensorflow/core/tpu/kernels/sparse_core_layout.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct SparseCoreTableLayoutProto {
    /// Table name (field `1`).
    #[prost(string, tag = "1")]
    pub table_name: String,
    /// Stacked table group name (field `2`).
    #[prost(string, tag = "2")]
    pub stacked_table_name: String,
    /// Number of `SparseCore` shards across the slice (field `3`).
    #[prost(int32, tag = "3")]
    pub num_sparse_cores: i32,
    /// Number of partitions per chip (field `4`).
    #[prost(int32, tag = "4")]
    pub num_partitions: i32,
    /// Total rows per `SparseCore` shard (field `5`).
    #[prost(int32, tag = "5")]
    pub total_rows_per_sparse_core_shard: i32,
    /// Logical shard dimensions `[vocab, dim]` (field `6`).
    #[prost(int64, repeated, packed = "true", tag = "6")]
    pub unsharded_shape: Vec<i64>,
    /// Physical padded shape `[padded_vocab, padded_dim]` (field `7`).
    #[prost(int64, repeated, packed = "true", tag = "7")]
    pub unsharded_padded_shape: Vec<i64>,
    /// Row offset for `SparseCore` shard placement (field `8`).
    #[prost(int64, tag = "8")]
    pub sparse_core_shard_row_offset: i64,
    /// Rotation index across `SparseCore` shards (field `9`).
    #[prost(int64, tag = "9")]
    pub sparse_core_shard_rotation: i64,
    /// Per-`SparseCore` micro-batch size (field `10`).
    #[prost(int64, tag = "10")]
    pub per_sparse_core_batch_size: i64,
    /// Number of features bound to this table (field `11`).
    #[prost(int64, tag = "11")]
    pub num_features: i64,
    /// Feedback-directed optimization lookup key (field `12`).
    #[prost(string, tag = "12")]
    pub fdo_key: String,
    /// Spanner feedback-directed optimization key (field `13`).
    #[prost(string, tag = "13")]
    pub spanner_fdo_key: String,
}

impl_proto_schema_message!(
    SparseCoreTableLayoutProto,
    "tensorflow.tpu.SparseCoreTableLayout"
);

/// Wire-compatible `tensorflow.tpu.SparseCoreTableLayouts` (`tensorflow/core/tpu/kernels/sparse_core_layout.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct SparseCoreTableLayoutsProto {
    /// Per-table `SparseCore` layout specifications (field `1`).
    #[prost(message, repeated, tag = "1")]
    pub tables: Vec<SparseCoreTableLayoutProto>,
}

impl_proto_schema_message!(
    SparseCoreTableLayoutsProto,
    "tensorflow.tpu.SparseCoreTableLayouts"
);

/// Wire-compatible `tensorflow.tpu.CompilationResultProto` (`tensorflow/core/protobuf/tpu/compilation_result.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct CompilationResultProto {
    /// Abseil status code (`0` = OK, field `1`).
    #[prost(int32, tag = "1")]
    pub status_code: i32,
    /// Error message when `status_code != 0` (field `2`).
    #[prost(string, tag = "2")]
    pub status_error_message: String,
    /// HLO protocols produced by compilation (field `3`).
    #[prost(message, repeated, tag = "3")]
    pub hlo_protos: Vec<HloModuleProtoWithConfig>,
    /// Detailed compilation error code enum (`CompilationResultProto.ErrorCode`, field `4`).
    #[prost(int32, tag = "4")]
    pub error_code: i32,
}

impl_proto_schema_message!(
    CompilationResultProto,
    "tensorflow.tpu.CompilationResultProto"
);

/// Wire-compatible `tensorflow.TensorShapeProto.Dim` (`tensor_shape.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TensorShapeDimProto {
    /// Dimension extent (`-1` for dynamic/unknown, field `1`).
    #[prost(int64, tag = "1")]
    pub size: i64,
    /// Optional dimension name (field `2`).
    #[prost(string, tag = "2")]
    pub name: String,
}

impl_proto_schema_message!(TensorShapeDimProto, "tensorflow.TensorShapeProto.Dim");

/// Wire-compatible `tensorflow.TensorShapeProto` (`tensor_shape.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TensorShapeProto {
    /// Ordered dimensions of the tensor (field `2`).
    #[prost(message, repeated, tag = "2")]
    pub dim: Vec<TensorShapeDimProto>,
    /// Whether the rank itself is unknown (field `3`).
    #[prost(bool, tag = "3")]
    pub unknown_rank: bool,
}

impl_proto_schema_message!(TensorShapeProto, "tensorflow.TensorShapeProto");

/// Wire-compatible `tensorflow.TensorProto` (`tensor.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct TensorProto {
    /// TensorFlow `DataType` enum (field `1`).
    #[prost(int32, tag = "1")]
    pub dtype: i32,
    /// Tensor shape (field `2`).
    #[prost(message, optional, tag = "2")]
    pub tensor_shape: Option<TensorShapeProto>,
    /// Version number (field `3`).
    #[prost(int32, tag = "3")]
    pub version_number: i32,
    /// Raw serialized tensor content bytes (field `4`).
    #[prost(bytes = "vec", tag = "4")]
    pub tensor_content: Vec<u8>,
    /// `DT_FLOAT` values (field `5`).
    #[prost(float, repeated, packed = "true", tag = "5")]
    pub float_val: Vec<f32>,
    /// `DT_DOUBLE` values (field `6`).
    #[prost(double, repeated, packed = "true", tag = "6")]
    pub double_val: Vec<f64>,
    /// `DT_INT32` / `DT_INT16` / `DT_INT8` / `DT_UINT8` values (field `7`).
    #[prost(int32, repeated, packed = "true", tag = "7")]
    pub int_val: Vec<i32>,
    /// `DT_STRING` values (field `8`).
    #[prost(bytes = "vec", repeated, tag = "8")]
    pub string_val: Vec<Vec<u8>>,
    /// `DT_COMPLEX64` interleaved `(real, imag)` values (field `9`).
    #[prost(float, repeated, packed = "true", tag = "9")]
    pub scomplex_val: Vec<f32>,
    /// `DT_INT64` values (field `10`).
    #[prost(int64, repeated, packed = "true", tag = "10")]
    pub int64_val: Vec<i64>,
    /// `DT_BOOL` values (field `11`).
    #[prost(bool, repeated, packed = "true", tag = "11")]
    pub bool_val: Vec<bool>,
    /// `DT_COMPLEX128` interleaved `(real, imag)` values (field `12`).
    #[prost(double, repeated, packed = "true", tag = "12")]
    pub dcomplex_val: Vec<f64>,
    /// `DT_HALF` / `DT_BFLOAT16` bit-cast values (field `13`).
    #[prost(int32, repeated, packed = "true", tag = "13")]
    pub half_val: Vec<i32>,
    /// `DT_UINT32` values (field `16`).
    #[prost(uint32, repeated, packed = "true", tag = "16")]
    pub uint32_val: Vec<u32>,
    /// `DT_UINT64` values (field `17`).
    #[prost(uint64, repeated, packed = "true", tag = "17")]
    pub uint64_val: Vec<u64>,
}

impl_proto_schema_message!(TensorProto, "tensorflow.TensorProto");

/// Wire-compatible `tensorflow.tpu.PaddingMap` (`dynamic_padding.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct PaddingMapProto {
    /// Argument index that has dynamic shapes (field `1`).
    #[prost(int32, tag = "1")]
    pub arg_index: i32,
    /// Dynamic shape dimension index (field `2`).
    #[prost(int32, tag = "2")]
    pub shape_index: i32,
    /// Argument index providing the unpadded dynamic bound (field `3`).
    #[prost(int32, tag = "3")]
    pub padding_arg_index: i32,
}

impl_proto_schema_message!(PaddingMapProto, "tensorflow.tpu.PaddingMap");

/// Wire-compatible `tensorflow.tpu.TPUCompileMetadataProto.Arg` (`compile_metadata.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuCompileMetadataArgProto {
    /// Argument `DataType` enum (field `1`).
    #[prost(int32, tag = "1")]
    pub dtype: i32,
    /// Argument tensor shape (field `2`).
    #[prost(message, optional, tag = "2")]
    pub shape: Option<TensorShapeProto>,
    /// Argument kind (`0` = Invalid, `1` = Parameter, `2` = Variable, `3` = `GuaranteedConstant`, field `3`).
    #[prost(int32, tag = "3")]
    pub kind: i32,
    /// SPMD sharding specification (field `4`).
    #[prost(message, optional, tag = "4")]
    pub sharding: Option<OpShardingProto>,
    /// Whether this argument receives identical data across replicas (field `5`).
    #[prost(bool, tag = "5")]
    pub is_same_data_across_replicas: bool,
    /// Whether this variable resides in fast memory (field `7`).
    #[prost(bool, tag = "7")]
    pub fast_mem: bool,
    /// Whether this argument requires cross-replica broadcast (field `9`).
    #[prost(bool, tag = "9")]
    pub requires_broadcast: bool,
}

impl_proto_schema_message!(
    TpuCompileMetadataArgProto,
    "tensorflow.tpu.TPUCompileMetadataProto.Arg"
);

/// Wire-compatible `tensorflow.tpu.TPUCompileMetadataProto.Retval` (`compile_metadata.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuCompileMetadataRetvalProto {
    /// Return value SPMD sharding specification (field `1`).
    #[prost(message, optional, tag = "1")]
    pub sharding: Option<OpShardingProto>,
}

impl_proto_schema_message!(
    TpuCompileMetadataRetvalProto,
    "tensorflow.tpu.TPUCompileMetadataProto.Retval"
);

/// Wire-compatible `tensorflow.tpu.TPUCompileMetadataProto` (`tensorflow/core/protobuf/tpu/compile_metadata.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuCompileMetadataProto {
    /// Input argument specifications (field `1`).
    #[prost(message, repeated, tag = "1")]
    pub args: Vec<TpuCompileMetadataArgProto>,
    /// Output return value specifications (field `2`).
    #[prost(message, repeated, tag = "2")]
    pub retvals: Vec<TpuCompileMetadataRetvalProto>,
    /// Number of replicas (field `3`).
    #[prost(int32, tag = "3")]
    pub num_replicas: i32,
    /// Number of cores per replica (field `4`).
    #[prost(int32, tag = "4")]
    pub num_cores_per_replica: i32,
    /// Static device assignment (field `8`).
    #[prost(message, optional, tag = "8")]
    pub device_assignment: Option<DeviceAssignmentProto>,
    /// Dynamic shape padding mappings (field `11`).
    #[prost(message, repeated, tag = "11")]
    pub padding_map: Vec<PaddingMapProto>,
    /// Session handle identifier (field `12`).
    #[prost(int64, tag = "12")]
    pub session_handle: i64,
    /// Function library fingerprint string (field `13`).
    #[prost(string, tag = "13")]
    pub function_library_fingerprint: String,
    /// Whether SPMD is enabled (field `15`).
    #[prost(bool, tag = "15")]
    pub use_spmd_for_xla_partitioning: bool,
    /// Whether separate sharding/unsharding programs are allowed (field `16`).
    #[prost(bool, tag = "16")]
    pub allow_separate_sharding_programs: bool,
}

impl_proto_schema_message!(
    TpuCompileMetadataProto,
    "tensorflow.tpu.TPUCompileMetadataProto"
);

/// Wire-compatible `tensorflow.tf2xla.TensorId` (`tf2xla.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct Tf2XlaTensorIdProto {
    /// Graph node name (field `1`).
    #[prost(string, tag = "1")]
    pub node_name: String,
    /// Output slot index on the node (field `2`).
    #[prost(int64, tag = "2")]
    pub output_index: i64,
}

impl_proto_schema_message!(Tf2XlaTensorIdProto, "tensorflow.tf2xla.TensorId");

/// Wire-compatible `tensorflow.tf2xla.Feed` (`tf2xla.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct Tf2XlaFeedProto {
    /// Source tensor ID (field `1`).
    #[prost(message, optional, tag = "1")]
    pub id: Option<Tf2XlaTensorIdProto>,
    /// Input tensor shape (field `2`).
    #[prost(message, optional, tag = "2")]
    pub shape: Option<TensorShapeProto>,
    /// Optional generated argument name (field `3`).
    #[prost(string, tag = "3")]
    pub name: String,
    /// Optional `DataType` enum (field `4`).
    #[prost(int32, tag = "4")]
    pub r#type: i32,
}

impl_proto_schema_message!(Tf2XlaFeedProto, "tensorflow.tf2xla.Feed");

/// Wire-compatible `tensorflow.tf2xla.Fetch` (`tf2xla.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct Tf2XlaFetchProto {
    /// Target tensor ID (field `1`).
    #[prost(message, optional, tag = "1")]
    pub id: Option<Tf2XlaTensorIdProto>,
    /// Optional generated result name (field `2`).
    #[prost(string, tag = "2")]
    pub name: String,
    /// Optional expected shape (field `3`).
    #[prost(message, optional, tag = "3")]
    pub shape: Option<TensorShapeProto>,
    /// Optional expected `DataType` enum (field `4`).
    #[prost(int32, tag = "4")]
    pub r#type: i32,
}

impl_proto_schema_message!(Tf2XlaFetchProto, "tensorflow.tf2xla.Fetch");

/// Wire-compatible `tensorflow.tf2xla.Variable` (`tf2xla.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct Tf2XlaVariableProto {
    /// Graph resource variable node name (field `1`).
    #[prost(string, tag = "1")]
    pub node_name: String,
    /// Optional generated variable name (field `2`).
    #[prost(string, tag = "2")]
    pub name: String,
    /// Variable tensor shape (field `3`).
    #[prost(message, optional, tag = "3")]
    pub shape: Option<TensorShapeProto>,
    /// Variable `DataType` enum (field `4`).
    #[prost(int32, tag = "4")]
    pub r#type: i32,
    /// Whether this variable is read-only (field `5`).
    #[prost(bool, tag = "5")]
    pub readonly: bool,
}

impl_proto_schema_message!(Tf2XlaVariableProto, "tensorflow.tf2xla.Variable");

/// Wire-compatible `tensorflow.tf2xla.Config` (`tf2xla.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct Tf2XlaConfigProto {
    /// Positional input feeds (field `1`).
    #[prost(message, repeated, tag = "1")]
    pub feed: Vec<Tf2XlaFeedProto>,
    /// Positional output fetches (field `2`).
    #[prost(message, repeated, tag = "2")]
    pub fetch: Vec<Tf2XlaFetchProto>,
    /// Resource variables (field `3`).
    #[prost(message, repeated, tag = "3")]
    pub variable: Vec<Tf2XlaVariableProto>,
}

impl_proto_schema_message!(Tf2XlaConfigProto, "tensorflow.tf2xla.Config");

/// Wire-compatible `tensorflow.tf2xla.TensorMetadata` (`host_compute_metadata.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct HostTransferTensorMetadataProto {
    /// Tensor `DataType` enum (field `1`).
    #[prost(int32, tag = "1")]
    pub r#type: i32,
    /// Tensor shape (field `2`).
    #[prost(message, optional, tag = "2")]
    pub shape: Option<TensorShapeProto>,
    /// XLA channel ID used for transfer (field `3`).
    #[prost(int64, tag = "3")]
    pub channel_id: i64,
}

impl_proto_schema_message!(
    HostTransferTensorMetadataProto,
    "tensorflow.tf2xla.TensorMetadata"
);

/// Wire-compatible `tensorflow.tf2xla.HostTransferMetadata` (`host_compute_metadata.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct HostTransferMetadataProto {
    /// Unique transfer key within the computation (field `1`).
    #[prost(string, tag = "1")]
    pub key: String,
    /// Transferred tensor metadata entries (field `2`).
    #[prost(message, repeated, tag = "2")]
    pub metadata: Vec<HostTransferTensorMetadataProto>,
}

impl_proto_schema_message!(
    HostTransferMetadataProto,
    "tensorflow.tf2xla.HostTransferMetadata"
);

/// Wire-compatible `tensorflow.tf2xla.HostComputeMetadata` (`host_compute_metadata.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct HostComputeMetadataProto {
    /// Device-to-host transfer descriptors (field `1`).
    #[prost(message, repeated, tag = "1")]
    pub device_to_host: Vec<HostTransferMetadataProto>,
    /// Host-to-device transfer descriptors (field `2`).
    #[prost(message, repeated, tag = "2")]
    pub host_to_device: Vec<HostTransferMetadataProto>,
}

impl_proto_schema_message!(
    HostComputeMetadataProto,
    "tensorflow.tf2xla.HostComputeMetadata"
);

/// Wire-compatible `tensorflow.tf2xla.XlaArgumentProto.Shape` (`xla_argument.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct XlaArgumentShapeProto {
    /// TensorFlow tensor shape variant (field `1`).
    #[prost(message, optional, tag = "1")]
    pub tensor_shape: Option<TensorShapeProto>,
    /// XLA shape variant (field `2`).
    #[prost(message, optional, tag = "2")]
    pub xla_shape: Option<ShapeProto>,
}

impl_proto_schema_message!(
    XlaArgumentShapeProto,
    "tensorflow.tf2xla.XlaArgumentProto.Shape"
);

/// Wire-compatible `tensorflow.tf2xla.XlaArgumentProto` (`xla_argument.proto`).
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, PartialEq, Message)]
pub struct XlaArgumentProto {
    /// Argument kind enum (field `1`).
    #[prost(int32, tag = "1")]
    pub kind: i32,
    /// Argument `DataType` enum (field `2`).
    #[prost(int32, tag = "2")]
    pub r#type: i32,
    /// Argument shape union (field `3`).
    #[prost(message, optional, tag = "3")]
    pub shape: Option<XlaArgumentShapeProto>,
    /// Compile-time constant value tensor (field `4`).
    #[prost(message, optional, tag = "4")]
    pub constant_value: Option<TensorProto>,
    /// Upper bound tensor for dynamic dimensions (field `5`).
    #[prost(message, optional, tag = "5")]
    pub value_bound: Option<TensorProto>,
    /// Dynamism predicate tensor (field `6`).
    #[prost(message, optional, tag = "6")]
    pub value_dynamism: Option<TensorProto>,
    /// Argument name (field `7`).
    #[prost(string, tag = "7")]
    pub name: String,
    /// Graph node name (field `8`).
    #[prost(string, tag = "8")]
    pub node_name: String,
    /// Resource kind enum (field `9`).
    #[prost(int32, tag = "9")]
    pub resource_kind: i32,
    /// Whether resource variable is initialized (field `10`).
    #[prost(bool, tag = "10")]
    pub initialized: bool,
    /// Fast memory placement hint (field `11`).
    #[prost(bool, tag = "11")]
    pub fast_mem: bool,
    /// Maximum tensor array size (field `12`).
    #[prost(int64, tag = "12")]
    pub max_array_size: i64,
    /// Tensor array gradient names (field `13`).
    #[prost(string, repeated, tag = "13")]
    pub tensor_array_gradients: Vec<String>,
    /// Identical data across replicas flag (field `14`).
    #[prost(bool, tag = "14")]
    pub is_same_data_across_replicas: bool,
    /// Requires cross-replica broadcast flag (field `15`).
    #[prost(bool, tag = "15")]
    pub requires_broadcast: bool,
}

impl_proto_schema_message!(XlaArgumentProto, "tensorflow.tf2xla.XlaArgumentProto");

/// Wire-compatible `tensorflow.TPUExecutableInfoProto.UpdateIndexPair` (`tpu_executable_info.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuUpdateIndexPairProto {
    /// Input resource variable index (field `1`).
    #[prost(int32, tag = "1")]
    pub index: i32,
    /// Whether the variable was modified by the computation (field `2`).
    #[prost(bool, tag = "2")]
    pub updated: bool,
}

impl_proto_schema_message!(
    TpuUpdateIndexPairProto,
    "tensorflow.TPUExecutableInfoProto.UpdateIndexPair"
);

/// Wire-compatible `tensorflow.TPUExecutableInfoProto.ShapeIndex` (`tpu_executable_info.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuShapeIndexProto {
    /// Tuple shape index path (field `1`).
    #[prost(int32, repeated, packed = "true", tag = "1")]
    pub index: Vec<i32>,
}

impl_proto_schema_message!(
    TpuShapeIndexProto,
    "tensorflow.TPUExecutableInfoProto.ShapeIndex"
);

/// Wire-compatible `tensorflow.TPUExecutableInfoProto` (`tpu_executable_info.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct TpuExecutableInfoProto {
    /// Input parameter shapes (field `2`).
    #[prost(message, repeated, tag = "2")]
    pub input_shapes: Vec<ShapeProto>,
    /// Output tuple shape (field `3`).
    #[prost(message, optional, tag = "3")]
    pub output_shape: Option<ShapeProto>,
    /// Session module snapshot (field `5`).
    #[prost(message, optional, tag = "5")]
    pub session_module: Option<crate::proto::hlo::HloSnapshotProto>,
    /// Static device assignment (field `6`).
    #[prost(message, optional, tag = "6")]
    pub device_assignment: Option<DeviceAssignmentProto>,
    /// Output tensor shapes (field `8`).
    #[prost(message, repeated, tag = "8")]
    pub output_tensor_shapes: Vec<TensorShapeProto>,
    /// Resource variable update index pairs (field `10`).
    #[prost(message, repeated, tag = "10")]
    pub variable_indices: Vec<TpuUpdateIndexPairProto>,
    /// Dynamic output shape indices (field `11`).
    #[prost(message, repeated, tag = "11")]
    pub dynamic_output_indices: Vec<TpuShapeIndexProto>,
}

impl_proto_schema_message!(TpuExecutableInfoProto, "tensorflow.TPUExecutableInfoProto");

/// Wire-compatible `tensorflow.TPUHostTransferInfoProto` (`tpu_executable_info.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuHostTransferInfoProto {
    /// Serialized transfer metadata blobs (field `2`).
    #[prost(bytes = "vec", repeated, tag = "2")]
    pub serialized_transfers: Vec<Vec<u8>>,
}

impl_proto_schema_message!(
    TpuHostTransferInfoProto,
    "tensorflow.TPUHostTransferInfoProto"
);

/// Wire-compatible `tensorflow.tpu.ClippingLimits` (`optimization_parameters.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct ClippingLimitsProto {
    /// Optional lower clipping bound (`google.protobuf.FloatValue`, `-inf` if unset, field `1`).
    #[prost(message, optional, tag = "1")]
    pub lower: Option<f32>,
    /// Optional upper clipping bound (`google.protobuf.FloatValue`, `+inf` if unset, field `2`).
    #[prost(message, optional, tag = "2")]
    pub upper: Option<f32>,
}

impl_proto_schema_message!(ClippingLimitsProto, "tensorflow.tpu.ClippingLimits");

/// Wire-compatible `tensorflow.tpu.SimulatedQuantization` (`optimization_parameters.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct SimulatedQuantizationProto {
    /// Whether simulated quantization is enabled (field `1`).
    #[prost(bool, tag = "1")]
    pub enabled: bool,
    /// Quantization clipping range (field `2`).
    #[prost(message, optional, tag = "2")]
    pub clipping_limits: Option<ClippingLimitsProto>,
    /// Number of quantization buckets (field `3`).
    #[prost(int32, tag = "3")]
    pub num_buckets: i32,
}

impl_proto_schema_message!(
    SimulatedQuantizationProto,
    "tensorflow.tpu.SimulatedQuantization"
);

/// Wire-compatible `tensorflow.tpu.OptimizerDynamicInput` (`optimization_parameters.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct OptimizerDynamicInputProto {
    /// Non-negative dynamic hyperparameter tag (field `1`).
    #[prost(int32, tag = "1")]
    pub tag: i32,
}

impl_proto_schema_message!(
    OptimizerDynamicInputProto,
    "tensorflow.tpu.OptimizerDynamicInput"
);

/// Wire-compatible `tensorflow.tpu.LearningRate` (`optimization_parameters.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct LearningRateProto {
    /// Constant learning rate scalar (field `1`).
    #[prost(float, tag = "1")]
    pub constant: f32,
    /// Dynamic learning rate input tag (field `2`).
    #[prost(message, optional, tag = "2")]
    pub dynamic: Option<OptimizerDynamicInputProto>,
}

impl_proto_schema_message!(LearningRateProto, "tensorflow.tpu.LearningRate");

/// Wire-compatible `tensorflow.tpu.AdagradParameters` (`optimization_parameters.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct AdagradParametersProto {}

impl_proto_schema_message!(AdagradParametersProto, "tensorflow.tpu.AdagradParameters");

/// Wire-compatible `tensorflow.tpu.AdagradMomentumParameters` (`optimization_parameters.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct AdagradMomentumParametersProto {
    /// Momentum moving average decay (field `1`).
    #[prost(float, tag = "1")]
    pub momentum: f32,
    /// Whether Nesterov momentum is enabled (field `2`).
    #[prost(bool, tag = "2")]
    pub use_nesterov: bool,
    /// Exponent for the squared gradient accumulator (field `3`).
    #[prost(float, tag = "3")]
    pub exponent: f32,
    /// Second-moment moving average parameter (field `4`).
    #[prost(float, tag = "4")]
    pub beta2: f32,
    /// Numerical stability epsilon (field `5`).
    #[prost(float, tag = "5")]
    pub epsilon: f32,
}

impl_proto_schema_message!(
    AdagradMomentumParametersProto,
    "tensorflow.tpu.AdagradMomentumParameters"
);

/// Wire-compatible `tensorflow.tpu.BoundedAdagradParameters` (`optimization_parameters.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct BoundedAdagradParametersProto {
    /// Whether to update accumulator before computing effective learning rate (field `1`).
    #[prost(bool, tag = "1")]
    pub update_accumulator_first: bool,
    /// Maximum variable update clipping bound (field `2`).
    #[prost(float, tag = "2")]
    pub max_var_update: f32,
    /// Maximum accumulator value bound (field `3`).
    #[prost(float, tag = "3")]
    pub max_accumulator: f32,
}

impl_proto_schema_message!(
    BoundedAdagradParametersProto,
    "tensorflow.tpu.BoundedAdagradParameters"
);

/// Wire-compatible `tensorflow.tpu.FrequencyAwareAdagradParameters` (`optimization_parameters.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct FrequencyAwareAdagradParametersProto {
    /// L1 regularization strength (field `1`).
    #[prost(float, tag = "1")]
    pub l1_regularization_strength: f32,
    /// L2 regularization strength (field `2`).
    #[prost(float, tag = "2")]
    pub l2_regularization_strength: f32,
    /// Maximum learning rate multiplier (field `3`).
    #[prost(float, tag = "3")]
    pub max_lr_multiplier: f32,
    /// Sparsity probability exponent (field `4`).
    #[prost(float, tag = "4")]
    pub probability_exponent: f32,
    /// Adagrad accumulator decay factor (field `5`).
    #[prost(float, tag = "5")]
    pub accumulator_decay: f32,
    /// Initial and minimum Adagrad accumulator value (field `6`).
    #[prost(float, tag = "6")]
    pub initial_accumulator_value: f32,
    /// Dynamic step counter input tag (field `7`).
    #[prost(message, optional, tag = "7")]
    pub step_counter: Option<OptimizerDynamicInputProto>,
}

impl_proto_schema_message!(
    FrequencyAwareAdagradParametersProto,
    "tensorflow.tpu.FrequencyAwareAdagradParameters"
);

/// Wire-compatible `tensorflow.tpu.StochasticGradientDescentParameters` (`optimization_parameters.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct StochasticGradientDescentParametersProto {}

impl_proto_schema_message!(
    StochasticGradientDescentParametersProto,
    "tensorflow.tpu.StochasticGradientDescentParameters"
);

/// Wire-compatible `tensorflow.tpu.FtrlParameters` (`optimization_parameters.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct FtrlParametersProto {
    /// L1 regularization parameter (field `1`).
    #[prost(float, tag = "1")]
    pub l1: f32,
    /// L2 regularization parameter (field `2`).
    #[prost(float, tag = "2")]
    pub l2: f32,
    /// Learning rate power exponent (field `3`).
    #[prost(float, tag = "3")]
    pub lr_power: f32,
    /// Whether linear accumulator is pre-multiplied by learning rate (field `6`).
    #[prost(bool, tag = "6")]
    pub multiply_linear_by_lr: bool,
    /// Beta parameter (field `7`).
    #[prost(float, tag = "7")]
    pub beta: f32,
    /// Deprecated allow zero accumulator flag (field `8`).
    #[prost(bool, tag = "8")]
    pub allow_zero_accumulator: bool,
}

impl_proto_schema_message!(FtrlParametersProto, "tensorflow.tpu.FtrlParameters");

/// Wire-compatible `tensorflow.tpu.AdamParameters` (`optimization_parameters.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct AdamParametersProto {
    /// First-moment exponential decay rate (field `3`).
    #[prost(float, tag = "3")]
    pub beta1: f32,
    /// Second-moment exponential decay rate (field `4`).
    #[prost(float, tag = "4")]
    pub beta2: f32,
    /// Numerical stability epsilon (field `5`).
    #[prost(float, tag = "5")]
    pub epsilon: f32,
    /// Use non-lazy Adam update rule (field `8`).
    #[prost(bool, tag = "8")]
    pub use_non_lazy_adam: bool,
    /// Place `epsilon^2` inside `sqrt(v + epsilon^2)` (field `10`).
    #[prost(bool, tag = "10")]
    pub use_sum_inside_sqrt: bool,
}

impl_proto_schema_message!(AdamParametersProto, "tensorflow.tpu.AdamParameters");

/// Wire-compatible `tensorflow.tpu.MomentumParameters` (`optimization_parameters.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct MomentumParametersProto {
    /// Momentum factor (field `1`).
    #[prost(float, tag = "1")]
    pub momentum: f32,
    /// Whether Nesterov momentum is enabled (field `2`).
    #[prost(bool, tag = "2")]
    pub use_nesterov: bool,
}

impl_proto_schema_message!(MomentumParametersProto, "tensorflow.tpu.MomentumParameters");

/// Wire-compatible `tensorflow.tpu.LionParameters` (`optimization_parameters.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct LionParametersProto {
    /// Interpolation decay rate `beta1` (field `1`).
    #[prost(float, tag = "1")]
    pub beta1: f32,
    /// Momentum update decay rate `beta2` (field `2`).
    #[prost(float, tag = "2")]
    pub beta2: f32,
    /// Whether non-lazy Lion updates are enabled (field `3`).
    #[prost(bool, tag = "3")]
    pub use_non_lazy_lion: bool,
}

impl_proto_schema_message!(LionParametersProto, "tensorflow.tpu.LionParameters");

/// Wire-compatible `tensorflow.tpu.RmsPropParameters` (`optimization_parameters.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct RmsPropParametersProto {
    /// Discounting factor `rho` (field `1`).
    #[prost(float, tag = "1")]
    pub rho: f32,
    /// Momentum factor (field `2`).
    #[prost(float, tag = "2")]
    pub momentum: f32,
    /// Numerical stability epsilon (field `3`).
    #[prost(float, tag = "3")]
    pub epsilon: f32,
}

impl_proto_schema_message!(RmsPropParametersProto, "tensorflow.tpu.RmsPropParameters");

/// Wire-compatible `tensorflow.tpu.CenteredRmsPropParameters` (`optimization_parameters.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct CenteredRmsPropParametersProto {
    /// Discounting factor `rho` (field `1`).
    #[prost(float, tag = "1")]
    pub rho: f32,
    /// Momentum factor (field `2`).
    #[prost(float, tag = "2")]
    pub momentum: f32,
    /// Numerical stability epsilon (field `3`).
    #[prost(float, tag = "3")]
    pub epsilon: f32,
}

impl_proto_schema_message!(
    CenteredRmsPropParametersProto,
    "tensorflow.tpu.CenteredRmsPropParameters"
);

/// Wire-compatible `tensorflow.tpu.MdlAdagradLightParameters` (`optimization_parameters.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct MdlAdagradLightParametersProto {
    /// L2 regularization factor (field `1`).
    #[prost(float, tag = "1")]
    pub l2: f32,
    /// Learning rate power (field `2`).
    #[prost(float, tag = "2")]
    pub lr_power: f32,
    /// Minimum servable MDL benefit (field `3`).
    #[prost(float, tag = "3")]
    pub min_servable_mdl_benefit: f32,
    /// MDL mix-in margin (field `4`).
    #[prost(float, tag = "4")]
    pub mdl_mix_in_margin: f32,
    /// MDL benefit ramp-up coefficient (field `5`).
    #[prost(float, tag = "5")]
    pub mdl_benefit_rampup_coeff: f32,
    /// MDL minimum weight (field `6`).
    #[prost(float, tag = "6")]
    pub mdl_min_weight: f32,
    /// Benefit revisit scale (field `7`).
    #[prost(float, tag = "7")]
    pub benefit_revisit_scale: f32,
    /// Maximum event benefit (field `8`).
    #[prost(float, tag = "8")]
    pub max_event_benefit: f32,
    /// Maximum total benefit (field `9`).
    #[prost(float, tag = "9")]
    pub max_total_benefit: f32,
    /// MDL hard limit (field `10`).
    #[prost(float, tag = "10")]
    pub mdl_hard_limit: f32,
    /// Hard limit minimum benefit flag (field `11`).
    #[prost(bool, tag = "11")]
    pub hard_limit_min_benefit: bool,
    /// MDL regularize flag (field `12`).
    #[prost(bool, tag = "12")]
    pub mdl_regularize: bool,
}

impl_proto_schema_message!(
    MdlAdagradLightParametersProto,
    "tensorflow.tpu.MdlAdagradLightParameters"
);

/// Wire-compatible `tensorflow.tpu.AdadeltaParameters` (`optimization_parameters.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct AdadeltaParametersProto {
    /// Decay factor `rho` (field `1`).
    #[prost(float, tag = "1")]
    pub rho: f32,
    /// Numerical stability epsilon (field `2`).
    #[prost(float, tag = "2")]
    pub epsilon: f32,
}

impl_proto_schema_message!(AdadeltaParametersProto, "tensorflow.tpu.AdadeltaParameters");

/// Wire-compatible `tensorflow.tpu.ProximalAdagradParameters` (`optimization_parameters.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct ProximalAdagradParametersProto {
    /// L1 regularization strength (field `1`).
    #[prost(float, tag = "1")]
    pub l1: f32,
    /// L2 regularization strength (field `2`).
    #[prost(float, tag = "2")]
    pub l2: f32,
}

impl_proto_schema_message!(
    ProximalAdagradParametersProto,
    "tensorflow.tpu.ProximalAdagradParameters"
);

/// Wire-compatible `tensorflow.tpu.OnlineYogiParameters` (`optimization_parameters.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct OnlineYogiParametersProto {
    /// L1 regularization parameter (field `1`).
    #[prost(float, tag = "1")]
    pub l1: f32,
    /// L2 regularization parameter (field `2`).
    #[prost(float, tag = "2")]
    pub l2: f32,
    /// Second-moment decay parameter `beta2` (field `3`).
    #[prost(float, tag = "3")]
    pub beta2: f32,
}

impl_proto_schema_message!(
    OnlineYogiParametersProto,
    "tensorflow.tpu.OnlineYogiParameters"
);

/// Wire-compatible `tensorflow.tpu.ProximalYogiParameters` (`optimization_parameters.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct ProximalYogiParametersProto {
    /// L1 regularization parameter (field `1`).
    #[prost(float, tag = "1")]
    pub l1: f32,
    /// L2 regularization parameter (field `2`).
    #[prost(float, tag = "2")]
    pub l2: f32,
    /// First-moment decay parameter `beta1` (field `3`).
    #[prost(float, tag = "3")]
    pub beta1: f32,
    /// Second-moment decay parameter `beta2` (field `4`).
    #[prost(float, tag = "4")]
    pub beta2: f32,
    /// Numerical stability epsilon (field `5`).
    #[prost(float, tag = "5")]
    pub epsilon: f32,
}

impl_proto_schema_message!(
    ProximalYogiParametersProto,
    "tensorflow.tpu.ProximalYogiParameters"
);

/// Wire-compatible `tensorflow.tpu.FrequencyEstimatorParameters` (`optimization_parameters.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct FrequencyEstimatorParametersProto {
    /// Moving average learning rate `tau` in `(0, 1)` (field `1`).
    #[prost(float, tag = "1")]
    pub tau: f32,
    /// Maximum step delta cap (field `2`).
    #[prost(float, tag = "2")]
    pub max_delta: f32,
    /// Outlier detection threshold (field `3`).
    #[prost(float, tag = "3")]
    pub outlier_threshold: f32,
    /// Weight transformation exponent (field `4`).
    #[prost(float, tag = "4")]
    pub weight_exponent: f32,
}

impl_proto_schema_message!(
    FrequencyEstimatorParametersProto,
    "tensorflow.tpu.FrequencyEstimatorParameters"
);

/// Wire-compatible `tensorflow.tpu.UserDefinedProgramParameters` (`optimization_parameters.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct UserDefinedProgramParametersProto {
    /// Custom elementwise `HloModuleProto` optimizer program (field `1`).
    #[prost(message, optional, tag = "1")]
    pub program: Option<HloModuleProto>,
}

impl_proto_schema_message!(
    UserDefinedProgramParametersProto,
    "tensorflow.tpu.UserDefinedProgramParameters"
);

/// Wire-compatible `tensorflow.tpu.AssignParameters` (`optimization_parameters.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct AssignParametersProto {}

impl_proto_schema_message!(AssignParametersProto, "tensorflow.tpu.AssignParameters");

/// Wire-compatible `tensorflow.tpu.GradientAccumulationStatus` (`optimization_parameters.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct GradientAccumulationStatusProto {}

impl_proto_schema_message!(
    GradientAccumulationStatusProto,
    "tensorflow.tpu.GradientAccumulationStatus"
);

/// Wire-compatible `tensorflow.tpu.LowDimensionalPackingStatus` (`optimization_parameters.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct LowDimensionalPackingStatusProto {}

impl_proto_schema_message!(
    LowDimensionalPackingStatusProto,
    "tensorflow.tpu.LowDimensionalPackingStatus"
);

/// Wire-compatible `tensorflow.tpu.HotIdReplicationConfiguration` (`optimization_parameters.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct HotIdReplicationConfigurationProto {
    /// Hot ID replication status enum (`Status`, field `1`).
    #[prost(int32, tag = "1")]
    pub status: i32,
}

impl_proto_schema_message!(
    HotIdReplicationConfigurationProto,
    "tensorflow.tpu.HotIdReplicationConfiguration"
);

/// Wire-compatible `tensorflow.tpu.OptimizationParameters` (`optimization_parameters.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct OptimizationParametersProto {
    /// Post-update weight clipping limits (field `2`).
    #[prost(message, optional, tag = "2")]
    pub clipping_limits: Option<ClippingLimitsProto>,
    /// Adagrad optimizer parameters (field `3`).
    #[prost(message, optional, tag = "3")]
    pub adagrad: Option<AdagradParametersProto>,
    /// SGD optimizer parameters (field `4`).
    #[prost(message, optional, tag = "4")]
    pub stochastic_gradient_descent: Option<StochasticGradientDescentParametersProto>,
    /// FTRL optimizer parameters (field `5`).
    #[prost(message, optional, tag = "5")]
    pub ftrl: Option<FtrlParametersProto>,
    /// Adam optimizer parameters (field `6`).
    #[prost(message, optional, tag = "6")]
    pub adam: Option<AdamParametersProto>,
    /// Gradient clipping limits (field `7`).
    #[prost(message, optional, tag = "7")]
    pub gradient_clipping_limits: Option<ClippingLimitsProto>,
    /// Momentum optimizer parameters (field `8`).
    #[prost(message, optional, tag = "8")]
    pub momentum: Option<MomentumParametersProto>,
    /// `RMSProp` optimizer parameters (field `9`).
    #[prost(message, optional, tag = "9")]
    pub rms_prop: Option<RmsPropParametersProto>,
    /// Centered `RMSProp` optimizer parameters (field `10`).
    #[prost(message, optional, tag = "10")]
    pub centered_rms_prop: Option<CenteredRmsPropParametersProto>,
    /// MDL Adagrad Light optimizer parameters (field `11`).
    #[prost(message, optional, tag = "11")]
    pub mdl_adagrad_light: Option<MdlAdagradLightParametersProto>,
    /// Adadelta optimizer parameters (field `12`).
    #[prost(message, optional, tag = "12")]
    pub adadelta: Option<AdadeltaParametersProto>,
    /// Learning rate specification (field `13`).
    #[prost(message, optional, tag = "13")]
    pub learning_rate: Option<LearningRateProto>,
    /// Proximal Adagrad optimizer parameters (field `14`).
    #[prost(message, optional, tag = "14")]
    pub proximal_adagrad: Option<ProximalAdagradParametersProto>,
    /// Weight decay factor (field `16`).
    #[prost(float, tag = "16")]
    pub weight_decay_factor: f32,
    /// Gradient accumulation status enum (field `17`).
    #[prost(int32, tag = "17")]
    pub gradient_accumulation_status: i32,
    /// Hot ID replication configuration (field `18`).
    #[prost(message, optional, tag = "18")]
    pub hot_id_replication_configuration: Option<HotIdReplicationConfigurationProto>,
    /// Bounded Adagrad optimizer parameters (field `19`).
    #[prost(message, optional, tag = "19")]
    pub bounded_adagrad: Option<BoundedAdagradParametersProto>,
    /// Online Yogi optimizer parameters (field `20`).
    #[prost(message, optional, tag = "20")]
    pub online_yogi: Option<OnlineYogiParametersProto>,
    /// Proximal Yogi optimizer parameters (field `21`).
    #[prost(message, optional, tag = "21")]
    pub proximal_yogi: Option<ProximalYogiParametersProto>,
    /// Multiply weight decay factor by learning rate flag (field `22`).
    #[prost(bool, tag = "22")]
    pub multiply_weight_decay_factor_by_learning_rate: bool,
    /// Frequency estimator parameters (field `23`).
    #[prost(message, optional, tag = "23")]
    pub frequency_estimator: Option<FrequencyEstimatorParametersProto>,
    /// User-defined HLO optimizer program parameters (field `24`).
    #[prost(message, optional, tag = "24")]
    pub user_defined_program: Option<UserDefinedProgramParametersProto>,
    /// Assign optimizer parameters (field `25`).
    #[prost(message, optional, tag = "25")]
    pub assign: Option<AssignParametersProto>,
    /// Adagrad-Momentum optimizer parameters (field `26`).
    #[prost(message, optional, tag = "26")]
    pub adagrad_momentum: Option<AdagradMomentumParametersProto>,
    /// Simulated quantization configuration (field `27`).
    #[prost(message, optional, tag = "27")]
    pub simulated_quantization: Option<SimulatedQuantizationProto>,
    /// Low-dimensional HBM packing status enum (field `28`).
    #[prost(int32, tag = "28")]
    pub low_dimensional_packing_status: i32,
    /// Lion optimizer parameters (field `29`).
    #[prost(message, optional, tag = "29")]
    pub lion: Option<LionParametersProto>,
    /// Frequency-aware Adagrad optimizer parameters (field `30`).
    #[prost(message, optional, tag = "30")]
    pub frequency_aware_adagrad: Option<FrequencyAwareAdagradParametersProto>,
}

impl_proto_schema_message!(
    OptimizationParametersProto,
    "tensorflow.tpu.OptimizationParameters"
);

/// Wire-compatible `tensorflow.tpu.StateVariableSpecification.UserDefined` (`optimization_parameters.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct StateVariableUserDefinedProto {}

impl_proto_schema_message!(
    StateVariableUserDefinedProto,
    "tensorflow.tpu.StateVariableSpecification.UserDefined"
);

/// Wire-compatible `tensorflow.tpu.StateVariableSpecification.FillWithConstant` (`optimization_parameters.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct StateVariableFillWithConstantProto {
    /// Initial constant fill value (field `1`).
    #[prost(double, tag = "1")]
    pub initial_value: f64,
}

impl_proto_schema_message!(
    StateVariableFillWithConstantProto,
    "tensorflow.tpu.StateVariableSpecification.FillWithConstant"
);

/// Wire-compatible `tensorflow.tpu.StateVariableSpecification` (`optimization_parameters.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct StateVariableSpecificationProto {
    /// State variable parameter name (field `1`).
    #[prost(string, tag = "1")]
    pub name: String,
    /// User-defined checkpointed state variable usage (field `2`).
    #[prost(message, optional, tag = "2")]
    pub user_defined: Option<StateVariableUserDefinedProto>,
    /// Constant-filled hidden state variable usage (field `3`).
    #[prost(message, optional, tag = "3")]
    pub fill_with_constant: Option<StateVariableFillWithConstantProto>,
}

impl_proto_schema_message!(
    StateVariableSpecificationProto,
    "tensorflow.tpu.StateVariableSpecification"
);

/// Wire-compatible `tpu.TpuEmbeddingPartitionedConfigProto` produced by `TpuEmbeddingEngine_ExecutePartitioner`.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuEmbeddingPartitionedConfigProto {
    /// Number of `SparseCore` partitions across the slice (field `1`).
    #[prost(int32, tag = "1")]
    pub num_partitions: i32,
    /// Partitioned table names (field `2`).
    #[prost(string, repeated, tag = "2")]
    pub table_names: Vec<String>,
    /// Original serialized `TpuEmbeddingConfigurationProto` bytes (field `3`).
    #[prost(bytes = "vec", tag = "3")]
    pub raw_config_bytes: Vec<u8>,
}

impl_proto_schema_message!(
    TpuEmbeddingPartitionedConfigProto,
    "tpu.TpuEmbeddingPartitionedConfigProto"
);

/// Wire-compatible `tpu.TpuEmbeddingMemoryConfigProto` for `SparseCore` memory configuration.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuEmbeddingMemoryConfigProto {
    /// Total HBM bytes reserved for `SparseCore` lookup tables (field `1`).
    #[prost(uint64, tag = "1")]
    pub hbm_buffers_bytes: u64,
    /// Number of collated host memory segments (field `2`).
    #[prost(int32, tag = "2")]
    pub num_segments: i32,
    /// Collated per-host memory configuration blobs (field `3`).
    #[prost(bytes = "vec", repeated, tag = "3")]
    pub segment_blobs: Vec<Vec<u8>>,
}

impl_proto_schema_message!(
    TpuEmbeddingMemoryConfigProto,
    "tpu.TpuEmbeddingMemoryConfigProto"
);

/// Wire-compatible `tpu.TpuEmbeddingHostNetworkConfigProto` used by `TpuEmbeddingEngine_ConfigureHost`.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuEmbeddingHostNetworkConfigProto {
    /// Host ordinal index (field `1`).
    #[prost(int32, tag = "1")]
    pub host_id: i32,
    /// Network endpoint address (field `2`).
    #[prost(string, tag = "2")]
    pub endpoint_address: String,
    /// Raw partition configuration payload (field `3`).
    #[prost(bytes = "vec", tag = "3")]
    pub partition_payload: Vec<u8>,
}

impl_proto_schema_message!(
    TpuEmbeddingHostNetworkConfigProto,
    "tpu.TpuEmbeddingHostNetworkConfigProto"
);

/// Wire-compatible `tpu.SliceConfig` (`tpu_platform_configs.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct SliceConfigProto {
    /// Slice configuration name (e.g., `"2x2"` or `"2x2x1"`, field `1`).
    #[prost(string, tag = "1")]
    pub name: String,
    /// Mesh dimensions along each axis (field `2`).
    #[prost(int32, repeated, packed = "true", tag = "2")]
    pub dimensions: Vec<i32>,
    /// Torus wrap-around flags along each axis (field `3`).
    #[prost(bool, repeated, packed = "true", tag = "3")]
    pub wraps: Vec<bool>,
    /// Whether twisted torus routing is active (field `4`).
    #[prost(bool, tag = "4")]
    pub twist: bool,
}

impl_proto_schema_message!(SliceConfigProto, "tpu.SliceConfig");

/// Wire-compatible `tpu.TpuPlatformConfig` (`tpu_platform_configs.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuPlatformConfigProto {
    /// Canonical TPU platform type name (e.g., `"tpuv5e"`, field `1`).
    #[prost(string, tag = "1")]
    pub platform_type_name: String,
    /// Supported slice configurations for this platform (field `2`).
    #[prost(message, repeated, tag = "2")]
    pub slice_configs: Vec<SliceConfigProto>,
    /// Number of TPU chips per physical tray (field `3`).
    #[prost(int64, tag = "3")]
    pub num_chips_per_tray: i64,
    /// Number of physical trays per host (field `4`).
    #[prost(int64, tag = "4")]
    pub num_trays: i64,
}

impl_proto_schema_message!(TpuPlatformConfigProto, "tpu.TpuPlatformConfig");

/// Wire-compatible `tpu.TpuPlatformConfigs` (`tpu_platform_configs.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuPlatformConfigsProto {
    /// Embedded platform configuration entries across TPU generations (field `1`).
    #[prost(message, repeated, tag = "1")]
    pub platform_configs: Vec<TpuPlatformConfigProto>,
}

impl_proto_schema_message!(TpuPlatformConfigsProto, "tpu.TpuPlatformConfigs");

/// Wire-compatible `asic_sw.MachineInfo` (`coredump.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct CoreDumpMachineInfoProto {
    /// Host machine name where the core dump was collected (field `1`).
    #[prost(string, tag = "1")]
    pub machine_name: String,
    /// Host Linux kernel version string (field `2`).
    #[prost(string, tag = "2")]
    pub kernel_version: String,
    /// Diagnostic reason for core dump collection (field `3`).
    #[prost(string, tag = "3")]
    pub reason_for_collection: String,
}

impl_proto_schema_message!(CoreDumpMachineInfoProto, "asic_sw.MachineInfo");

/// Wire-compatible `asic_sw.TimestampInfo` (`coredump.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct CoreDumpTimestampInfoProto {
    /// Collection start time in seconds since Unix epoch (field `1`).
    #[prost(int64, tag = "1")]
    pub collection_start_sec: i64,
    /// Collection end time in seconds since Unix epoch (field `2`).
    #[prost(int64, tag = "2")]
    pub collection_end_sec: i64,
    /// Host boot time in seconds since Unix epoch (field `3`).
    #[prost(int64, tag = "3")]
    pub boot_time_sec: i64,
    /// Device initialization time in seconds since Unix epoch (field `4`).
    #[prost(int64, tag = "4")]
    pub device_start_sec: i64,
}

impl_proto_schema_message!(CoreDumpTimestampInfoProto, "asic_sw.TimestampInfo");

/// Wire-compatible `asic_sw.HardwareInfo` (`coredump.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct CoreDumpHardwareInfoProto {
    /// Per-chip power performance states (`pstate`, field `1`).
    #[prost(int32, repeated, packed = "true", tag = "1")]
    pub pstate: Vec<i32>,
}

impl_proto_schema_message!(CoreDumpHardwareInfoProto, "asic_sw.HardwareInfo");

/// Wire-compatible `asic_sw.RegisterDump` (`coredump.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct CoreDumpRegisterDumpProto {
    /// Architectural CSR or register name (field `1`).
    #[prost(string, tag = "1")]
    pub register_name: String,
    /// BAR0 byte offset of the register (field `2`).
    #[prost(uint64, tag = "2")]
    pub register_offset: u64,
    /// Captured 64-bit register value (field `3`).
    #[prost(uint64, tag = "3")]
    pub register_value: u64,
}

impl_proto_schema_message!(CoreDumpRegisterDumpProto, "asic_sw.RegisterDump");

/// Wire-compatible `asic_sw.CoreStateDump` (`coredump.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct CoreStateDumpProto {
    /// Core engine type (`"TC"` or `"SC"`, field `1`).
    #[prost(string, tag = "1")]
    pub core_type: String,
    /// Logical core index on the chip (field `2`).
    #[prost(int32, tag = "2")]
    pub core_id: i32,
    /// Captured scalar instruction pointer (`PC`, field `3`).
    #[prost(uint64, tag = "3")]
    pub pc: u64,
    /// Active HLO instruction name at `pc` (field `4`).
    #[prost(string, tag = "4")]
    pub hlo_name: String,
    /// Captured hardware register snapshots (field `5`).
    #[prost(message, repeated, tag = "5")]
    pub registers: Vec<CoreDumpRegisterDumpProto>,
}

impl_proto_schema_message!(CoreStateDumpProto, "asic_sw.CoreStateDump");

/// Wire-compatible `asic_sw.AnalyzerReport` (`coredump.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct CoreDumpAnalyzerReportProto {
    /// Diagnostic analyzer name (e.g., `"GeneralInfo"` or `"Anomaly"`, field `1`).
    #[prost(string, tag = "1")]
    pub analyzer_name: String,
    /// Diagnostic category name (e.g., `"RUN_STATUS"` or `"ERROR_STATUS"`, field `2`).
    #[prost(string, tag = "2")]
    pub category: String,
    /// Short diagnostic tag (field `3`).
    #[prost(string, tag = "3")]
    pub tag: String,
    /// Sanitized human-readable analyzer message (field `4`).
    #[prost(string, tag = "4")]
    pub message: String,
}

impl_proto_schema_message!(CoreDumpAnalyzerReportProto, "asic_sw.AnalyzerReport");

/// Wire-compatible `asic_sw.CoreDump` (`coredump.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct CoreDumpProto {
    /// Sanitized host machine and kernel metadata (field `1`).
    #[prost(message, optional, tag = "1")]
    pub machine_info: Option<CoreDumpMachineInfoProto>,
    /// Collection and boot timestamps (field `2`).
    #[prost(message, optional, tag = "2")]
    pub timestamps: Option<CoreDumpTimestampInfoProto>,
    /// Hardware power state summary (field `3`).
    #[prost(message, optional, tag = "3")]
    pub hardware: Option<CoreDumpHardwareInfoProto>,
    /// Per-core execution state snapshots (field `4`).
    #[prost(message, repeated, tag = "4")]
    pub core_states: Vec<CoreStateDumpProto>,
    /// Sanitized diagnostic analyzer reports (field `5`).
    #[prost(message, repeated, tag = "5")]
    pub analyzer_reports: Vec<CoreDumpAnalyzerReportProto>,
}

impl_proto_schema_message!(CoreDumpProto, "asic_sw.CoreDump");

/// Wire-compatible `libtpu.sdk.tpu_telemetry.ActiveSyncFlagWait` (`stall_summaries.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ActiveSyncFlagWaitProto {
    /// Hardware `SyncFlag` ID being waited on (field `1`).
    #[prost(uint32, tag = "1")]
    pub sync_flag_id: u32,
    /// Expected threshold count required to unblock `vwait` (field `2`).
    #[prost(uint32, tag = "2")]
    pub expected_value: u32,
    /// Current hardware `SyncFlag` counter value (field `3`).
    #[prost(uint32, tag = "3")]
    pub current_value: u32,
}

impl_proto_schema_message!(
    ActiveSyncFlagWaitProto,
    "libtpu.sdk.tpu_telemetry.ActiveSyncFlagWait"
);

/// Wire-compatible `libtpu.sdk.tpu_telemetry.CoreStallSummary` (`stall_summaries.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct CoreStallSummaryProto {
    /// Core dictionary key (e.g., `"core_0"`, field `1`).
    #[prost(string, tag = "1")]
    pub core_key: String,
    /// Global logical core ID across the slice (field `2`).
    #[prost(int32, tag = "2")]
    pub global_core_id: i32,
    /// Physical chip ID on the host (field `3`).
    #[prost(int32, tag = "3")]
    pub chip_id: i32,
    /// Core type string (`"TPU_CORE_TYPE_TENSOR_CORE"` or `"TPU_CORE_TYPE_SPARSE_CORE"`, field `4`).
    #[prost(string, tag = "4")]
    pub core_on_chip_type: String,
    /// Intra-chip core ordinal index (field `5`).
    #[prost(int32, tag = "5")]
    pub core_on_chip_index: i32,
    /// Whether the XDB debug server is active on this core (field `6`).
    #[prost(bool, tag = "6")]
    pub xdb_server_running: bool,
    /// Current instruction pointer (`PC`, field `7`).
    #[prost(uint64, tag = "7")]
    pub pc: u64,
    /// Active HLO source location string (field `8`).
    #[prost(string, tag = "8")]
    pub hlo_location: String,
    /// Diagnostic stall or fault message (field `9`).
    #[prost(string, tag = "9")]
    pub error_message: String,
    /// Unsatisfied `SyncFlag` wait descriptors (field `10`).
    #[prost(message, repeated, tag = "10")]
    pub active_sync_waits: Vec<ActiveSyncFlagWaitProto>,
}

impl_proto_schema_message!(
    CoreStallSummaryProto,
    "libtpu.sdk.tpu_telemetry.CoreStallSummary"
);

/// Wire-compatible `libtpu.sdk.tpu_telemetry.AllCoreStateSummaries` (`stall_summaries.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct AllCoreStateSummariesProto {
    /// Per-core stall and sequencer summaries (field `1`).
    #[prost(message, repeated, tag = "1")]
    pub core_stall_summaries: Vec<CoreStallSummaryProto>,
    /// Hostname where the stall summary was harvested (field `2`).
    #[prost(string, tag = "2")]
    pub hostname: String,
    /// Harvest start timestamp in Unix microseconds (field `3`).
    #[prost(int64, tag = "3")]
    pub harvest_start_time_unix_micros: i64,
    /// Harvest end timestamp in Unix microseconds (field `4`).
    #[prost(int64, tag = "4")]
    pub harvest_end_time_unix_micros: i64,
    /// `libtpu` build identifier string (field `5`).
    #[prost(string, tag = "5")]
    pub libtpu_build: String,
}

impl_proto_schema_message!(
    AllCoreStateSummariesProto,
    "libtpu.sdk.tpu_telemetry.AllCoreStateSummaries"
);

/// Wire-compatible `platforms_deepsea.jellyfish.xdb.SectionAllocation` (`compiler_metadata.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct CompilerSectionAllocationProto {
    /// Section name (e.g., `".text.imem"`, `".rodata.cmem"`, `".bss.vmem"`, field `1`).
    #[prost(string, tag = "1")]
    pub section_name: String,
    /// Base byte offset within the target memory space (field `2`).
    #[prost(uint64, tag = "2")]
    pub base_offset: u64,
    /// Section size in bytes (field `3`).
    #[prost(uint64, tag = "3")]
    pub byte_size: u64,
    /// Memory space ordinal (`0 = HBM`, `1 = VMEM`, `2 = SMEM`, `3 = CMEM`, field `4`).
    #[prost(int32, tag = "4")]
    pub memory_space: i32,
}

impl_proto_schema_message!(
    CompilerSectionAllocationProto,
    "platforms_deepsea.jellyfish.xdb.SectionAllocation"
);

/// Wire-compatible `platforms_deepsea.jellyfish.xdb.SymbolTableEntry` (`compiler_metadata.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct CompilerSymbolTableEntryProto {
    /// Symbol identifier (field `1`).
    #[prost(string, tag = "1")]
    pub symbol_name: String,
    /// Byte offset in `IMEM` (field `2`).
    #[prost(uint64, tag = "2")]
    pub imem_offset_bytes: u64,
    /// Symbol size in bytes (field `3`).
    #[prost(uint64, tag = "3")]
    pub byte_size: u64,
}

impl_proto_schema_message!(
    CompilerSymbolTableEntryProto,
    "platforms_deepsea.jellyfish.xdb.SymbolTableEntry"
);

/// Wire-compatible `platforms_deepsea.jellyfish.xdb.HloToLloMapping` (`compiler_metadata.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct HloToLloMappingProto {
    /// Originating HLO instruction name (field `1`).
    #[prost(string, tag = "1")]
    pub hlo_op_name: String,
    /// Inclusive starting VLIW bundle index (field `2`).
    #[prost(uint32, tag = "2")]
    pub start_bundle_index: u32,
    /// Exclusive ending VLIW bundle index (field `3`).
    #[prost(uint32, tag = "3")]
    pub end_bundle_index: u32,
}

impl_proto_schema_message!(
    HloToLloMappingProto,
    "platforms_deepsea.jellyfish.xdb.HloToLloMapping"
);

/// Wire-compatible `platforms_deepsea.jellyfish.xdb.CompilerMetadata` (`compiler_metadata.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct CompilerMetadataProto {
    /// Unique 64-bit program compilation ID (field `1`).
    #[prost(int64, tag = "1")]
    pub program_id: i64,
    /// Number of replicas (field `2`).
    #[prost(int32, tag = "2")]
    pub replica_count: i32,
    /// Number of cores per replica (field `3`).
    #[prost(int32, tag = "3")]
    pub num_cores_per_replica: i32,
    /// Whether haltless execution mode is enabled (field `4`).
    #[prost(bool, tag = "4")]
    pub is_haltless: bool,
    /// Total compiled executable size in bytes (field `5`).
    #[prost(int64, tag = "5")]
    pub executable_size: i64,
    /// Post-optimization HLO instruction count (field `6`).
    #[prost(int64, tag = "6")]
    pub optimized_hlo_instruction_count: i64,
    /// HLO module name (field `7`).
    #[prost(string, tag = "7")]
    pub module_name: String,
    /// 32-character SHA-256 executable fingerprint (field `8`).
    #[prost(string, tag = "8")]
    pub executable_fingerprint: String,
    /// Memory section allocations (field `9`).
    #[prost(message, repeated, tag = "9")]
    pub section_allocations: Vec<CompilerSectionAllocationProto>,
    /// LLO symbol table entries (field `10`).
    #[prost(message, repeated, tag = "10")]
    pub symbol_table: Vec<CompilerSymbolTableEntryProto>,
    /// HLO-to-LLO VLIW bundle range mappings (field `11`).
    #[prost(message, repeated, tag = "11")]
    pub hlo_to_llo_mappings: Vec<HloToLloMappingProto>,
}

impl_proto_schema_message!(
    CompilerMetadataProto,
    "platforms_deepsea.jellyfish.xdb.CompilerMetadata"
);

/// Wire-compatible `mlir.tpu.MosaicDmaDescriptorProto` (`mosaic_tpu.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct MosaicDmaDescriptorProto {
    /// Logical DMA descriptor identifier (field `1`).
    #[prost(uint32, tag = "1")]
    pub dma_id: u32,
    /// Source memory space ordinal (`0 = HBM`, `1 = VMEM`, `2 = SMEM`, `3 = CMEM`, field `2`).
    #[prost(uint32, tag = "2")]
    pub src_memory_space: u32,
    /// Destination memory space ordinal (`0 = HBM`, `1 = VMEM`, `2 = SMEM`, `3 = CMEM`, field `3`).
    #[prost(uint32, tag = "3")]
    pub dst_memory_space: u32,
    /// Transfer byte length (field `4`).
    #[prost(uint64, tag = "4")]
    pub transfer_bytes: u64,
    /// Associated Mosaic hardware semaphore ID (field `5`).
    #[prost(uint32, tag = "5")]
    pub semaphore_id: u32,
}

impl_proto_schema_message!(
    MosaicDmaDescriptorProto,
    "mlir.tpu.MosaicDmaDescriptorProto"
);

/// Wire-compatible `mlir.tpu.MosaicScratchpadLayoutProto` (`mosaic_tpu.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct MosaicScratchpadLayoutProto {
    /// VMEM scratchpad allocation in bytes (field `1`).
    #[prost(uint64, tag = "1")]
    pub vmem_bytes: u64,
    /// SMEM scratchpad allocation in bytes (field `2`).
    #[prost(uint64, tag = "2")]
    pub smem_bytes: u64,
    /// CMEM scratchpad allocation in bytes (field `3`).
    #[prost(uint64, tag = "3")]
    pub cmem_bytes: u64,
    /// Tile stride dimensions (field `4`).
    #[prost(int64, repeated, packed = "true", tag = "4")]
    pub tile_strides: Vec<i64>,
}

impl_proto_schema_message!(
    MosaicScratchpadLayoutProto,
    "mlir.tpu.MosaicScratchpadLayoutProto"
);

/// Wire-compatible `mlir.tpu.MosaicBackendConfigProto` (`mosaic_tpu.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct MosaicBackendConfigProto {
    /// Custom Mosaic / Pallas kernel symbol name (field `1`).
    #[prost(string, tag = "1")]
    pub kernel_name: String,
    /// Target core type (`0 = TensorCore`, `1 = SparseCore`, field `2`).
    #[prost(int32, tag = "2")]
    pub core_type: i32,
    /// Polyhedral iteration grid bounds (field `3`).
    #[prost(int64, repeated, packed = "true", tag = "3")]
    pub grid_bounds: Vec<i64>,
    /// Scratchpad memory layout (field `4`).
    #[prost(message, optional, tag = "4")]
    pub scratchpad_layout: Option<MosaicScratchpadLayoutProto>,
    /// Asynchronous DMA descriptors used by the kernel (field `5`).
    #[prost(message, repeated, tag = "5")]
    pub dma_descriptors: Vec<MosaicDmaDescriptorProto>,
}

impl_proto_schema_message!(
    MosaicBackendConfigProto,
    "mlir.tpu.MosaicBackendConfigProto"
);

/// Wire-compatible `tpu.HloModuleFingerprintProto` (`tpu_fingerprint.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct HloModuleFingerprintProto {
    /// HLO module name (field `1`).
    #[prost(string, tag = "1")]
    pub module_name: String,
    /// Structural instruction DAG hash (field `2`).
    #[prost(uint64, tag = "2")]
    pub structural_hash: u64,
    /// SPMD sharding annotation hash (field `3`).
    #[prost(uint64, tag = "3")]
    pub sharding_hash: u64,
    /// Minor-to-major layout & tile hash (field `4`).
    #[prost(uint64, tag = "4")]
    pub layout_hash: u64,
}

impl_proto_schema_message!(HloModuleFingerprintProto, "tpu.HloModuleFingerprintProto");

/// Wire-compatible `tpu.TpuCompilerEnvironmentFingerprintProto` (`tpu_fingerprint.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuCompilerEnvironmentFingerprintProto {
    /// Target Cloud TPU version ordinal (field `1`).
    #[prost(int32, tag = "1")]
    pub tpu_version: i32,
    /// Compiler ABI hash string (field `2`).
    #[prost(string, tag = "2")]
    pub compiler_abi_hash: String,
    /// 64-bit digest over `--xla_tpu_*` compiler flags (field `3`).
    #[prost(uint64, tag = "3")]
    pub flags_digest: u64,
    /// Whether `MegaCore` mode is enabled (field `4`).
    #[prost(bool, tag = "4")]
    pub megacore_enabled: bool,
}

impl_proto_schema_message!(
    TpuCompilerEnvironmentFingerprintProto,
    "tpu.TpuCompilerEnvironmentFingerprintProto"
);

/// Wire-compatible `tpu.TpuCompilationFingerprintProto` (`tpu_fingerprint.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuCompilationFingerprintProto {
    /// 32-character lowercase hex SHA-256 compilation fingerprint (field `1`).
    #[prost(string, tag = "1")]
    pub hex_sha256: String,
    /// HLO module fingerprint components (field `2`).
    #[prost(message, optional, tag = "2")]
    pub hlo_fingerprint: Option<HloModuleFingerprintProto>,
    /// Compiler environment fingerprint components (field `3`).
    #[prost(message, optional, tag = "3")]
    pub env_fingerprint: Option<TpuCompilerEnvironmentFingerprintProto>,
    /// Target mesh topology fingerprint (field `4`).
    #[prost(uint64, tag = "4")]
    pub topology_fingerprint: u64,
}

impl_proto_schema_message!(
    TpuCompilationFingerprintProto,
    "tpu.TpuCompilationFingerprintProto"
);

/// Wire-compatible `tpu.telemetry.TpuPowerStateProto` (`power_telemetry.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct TpuPowerStateProto {
    /// Physical chip ID (field `1`).
    #[prost(uint32, tag = "1")]
    pub chip_id: u32,
    /// Power state level (`0 = P0 Max`, `1 = P1 Nominal`, `2 = P2 Idle`, field `2`).
    #[prost(int32, tag = "2")]
    pub power_state_level: i32,
    /// Instantaneous power draw in Watts (field `3`).
    #[prost(double, tag = "3")]
    pub current_watts: f64,
    /// Maximum thermal design power (`TDP`) in Watts (field `4`).
    #[prost(double, tag = "4")]
    pub max_tdp_watts: f64,
    /// Die junction temperature in degrees Celsius (field `5`).
    #[prost(double, tag = "5")]
    pub junction_temp_celsius: f64,
}

impl_proto_schema_message!(TpuPowerStateProto, "tpu.telemetry.TpuPowerStateProto");

/// Wire-compatible `tpu.telemetry.ChipThermalThrottleSummaryProto` (`power_telemetry.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct ChipThermalThrottleSummaryProto {
    /// Physical chip ID (field `1`).
    #[prost(uint32, tag = "1")]
    pub chip_id: u32,
    /// Cumulative thermal throttle duration in microseconds (field `2`).
    #[prost(uint64, tag = "2")]
    pub throttle_duration_us: u64,
    /// Total number of thermal throttle events (field `3`).
    #[prost(uint32, tag = "3")]
    pub throttle_event_count: u32,
    /// Peak observed die temperature in degrees Celsius (field `4`).
    #[prost(double, tag = "4")]
    pub peak_temp_celsius: f64,
}

impl_proto_schema_message!(
    ChipThermalThrottleSummaryProto,
    "tpu.telemetry.ChipThermalThrottleSummaryProto"
);

/// Wire-compatible `tpu.telemetry.FrequencyScalingTelemetryProto` (`power_telemetry.proto`).
#[derive(Clone, PartialEq, Message)]
pub struct FrequencyScalingTelemetryProto {
    /// Physical chip ID (field `1`).
    #[prost(uint32, tag = "1")]
    pub chip_id: u32,
    /// Nominal `TensorCore` clock frequency in MHz (field `2`).
    #[prost(uint32, tag = "2")]
    pub nominal_clock_mhz: u32,
    /// Active `TensorCore` clock frequency in MHz (field `3`).
    #[prost(uint32, tag = "3")]
    pub active_clock_mhz: u32,
    /// Active HBM memory clock frequency in MHz (field `4`).
    #[prost(uint32, tag = "4")]
    pub hbm_clock_mhz: u32,
    /// Per-chip power state samples (field `5`).
    #[prost(message, repeated, tag = "5")]
    pub power_states: Vec<TpuPowerStateProto>,
    /// Per-chip thermal throttle summaries (field `6`).
    #[prost(message, repeated, tag = "6")]
    pub thermal_summaries: Vec<ChipThermalThrottleSummaryProto>,
}

impl_proto_schema_message!(
    FrequencyScalingTelemetryProto,
    "tpu.telemetry.FrequencyScalingTelemetryProto"
);

/// Wire-compatible `libtpu.sdk.tpu_telemetry.TpuzActiveHloExecutionTraceProto` (`tpu_telemetry_snapshot.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuzActiveHloExecutionTraceProto {
    /// Global or intra-chip core ID (field `1`).
    #[prost(uint32, tag = "1")]
    pub core_id: u32,
    /// Currently executing HLO module name (field `2`).
    #[prost(string, tag = "2")]
    pub hlo_module_name: String,
    /// Currently executing HLO instruction name (field `3`).
    #[prost(string, tag = "3")]
    pub active_hlo_op_name: String,
    /// HLO instruction category (e.g., `"dot"`, `"all-reduce"`, field `4`).
    #[prost(string, tag = "4")]
    pub active_hlo_category: String,
    /// Current hardware instruction pointer (`PC`, field `5`).
    #[prost(uint64, tag = "5")]
    pub program_counter: u64,
    /// Elapsed cycles spent in the current instruction (field `6`).
    #[prost(uint64, tag = "6")]
    pub elapsed_cycles: u64,
}

impl_proto_schema_message!(
    TpuzActiveHloExecutionTraceProto,
    "libtpu.sdk.tpu_telemetry.TpuzActiveHloExecutionTraceProto"
);

/// Wire-compatible `libtpu.sdk.tpu_telemetry.TpuzLiveBufferAllocationProto` (`tpu_telemetry_snapshot.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuzLiveBufferAllocationProto {
    /// Unique buffer allocation identifier (field `1`).
    #[prost(uint64, tag = "1")]
    pub buffer_id: u64,
    /// Memory space (`0 = HBM`, `1 = VMEM`, `2 = SMEM`, `3 = CMEM`, field `2`).
    #[prost(uint32, tag = "2")]
    pub memory_space: u32,
    /// Base device address in bytes (field `3`).
    #[prost(uint64, tag = "3")]
    pub base_address: u64,
    /// Allocation size in bytes (field `4`).
    #[prost(uint64, tag = "4")]
    pub size_bytes: u64,
    /// Formatted tensor shape and layout string (field `5`).
    #[prost(string, tag = "5")]
    pub shape_description: String,
    /// Producing HLO instruction name (field `6`).
    #[prost(string, tag = "6")]
    pub owner_hlo_name: String,
}

impl_proto_schema_message!(
    TpuzLiveBufferAllocationProto,
    "libtpu.sdk.tpu_telemetry.TpuzLiveBufferAllocationProto"
);

/// Wire-compatible `libtpu.sdk.tpu_telemetry.TpuzCoreTelemetrySnapshotProto` (`tpu_telemetry_snapshot.proto`).
#[derive(Clone, PartialEq, Eq, Message)]
pub struct TpuzCoreTelemetrySnapshotProto {
    /// Physical chip ordinal ID (field `1`).
    #[prost(uint32, tag = "1")]
    pub chip_id: u32,
    /// Core ordinal ID (field `2`).
    #[prost(uint32, tag = "2")]
    pub core_id: u32,
    /// Core type (`0 = TensorCore`, `1 = SparseCore`, field `3`).
    #[prost(uint32, tag = "3")]
    pub core_type: u32,
    /// Active HLO execution trace on this core (field `4`).
    #[prost(message, optional, tag = "4")]
    pub active_trace: Option<TpuzActiveHloExecutionTraceProto>,
    /// Live device buffer allocations on this core (field `5`).
    #[prost(message, repeated, tag = "5")]
    pub live_buffers: Vec<TpuzLiveBufferAllocationProto>,
    /// Total VMEM bytes currently allocated on this core (field `6`).
    #[prost(uint64, tag = "6")]
    pub vmem_bytes_in_use: u64,
    /// Total HBM bytes currently allocated on this core (field `7`).
    #[prost(uint64, tag = "7")]
    pub hbm_bytes_in_use: u64,
}

impl_proto_schema_message!(
    TpuzCoreTelemetrySnapshotProto,
    "libtpu.sdk.tpu_telemetry.TpuzCoreTelemetrySnapshotProto"
);

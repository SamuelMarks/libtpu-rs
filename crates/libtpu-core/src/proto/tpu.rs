//! Cloud TPU compilation environment (`xla.jellyfish.TpuCompilationEnvironment`), topology,
//! `SparseCore` / `TPUEmbedding`, `TF2XLA`, XDB `CompilerMetadata`, and coredump protobuf types.
//!
//! Re-exports `prost-build` generated structs from `platforms/xla/service/jellyfish/`,
//! `platforms/deepsea/jellyfish/xdb/`, `platforms/asic_sw/lib/common/`,
//! `learning/45eac/research/pjrt/extensions/tpu_topology/`, and
//! `third_party/tensorflow/core/protobuf/tpu/`.

use crate::config::TpuInitConfig;
use crate::proto::descriptor::ProtoSchemaMessage;

pub use super::descriptor::generated_prost_schemas::libtpu::ext::tpu::{
    ChipThermalThrottleSummaryProto, FrequencyScalingTelemetryProto, HloModuleFingerprintProto,
    MosaicBackendConfigProto, MosaicDmaDescriptorProto, MosaicScratchpadLayoutProto,
    TpuCompilationFingerprintProto, TpuCompilerEnvironmentFingerprintProto,
    TpuEmbeddingHostNetworkConfigProto, TpuEmbeddingMemoryConfigProto,
    TpuEmbeddingPartitionedConfigProto, TpuPowerStateProto, TpuzActiveHloExecutionTraceProto,
    TpuzCoreTelemetrySnapshotProto, TpuzLiveBufferAllocationProto,
};
pub use super::descriptor::generated_prost_schemas::asic_sw::{
    AnalyzerInfo as CoredumpAnalyzerInfoProto, CoreDump as CoreDumpProto,
    CpuState as CoredumpCpuStateProto, FlightRecorder as FlightRecorderProto,
    GeneralInfo as CoredumpGeneralInfoProto, Info as CoredumpInfoProto, PowerInfo as PowerInfoProto,
    Pstate as PstateProto, Results as CoredumpResultsProto,
};
pub use super::descriptor::generated_prost_schemas::pjrt::tpu_topology_extension::{
    SliceConfig as SliceConfigProto, TpuPlatformConfig as TpuPlatformConfigProto,
    TpuPlatformConfigs as TpuPlatformConfigsProto,
};
pub use super::descriptor::generated_prost_schemas::platforms_deepsea::jellyfish::xdb::tpu_telemetry::{
    AllCoreStateSummaries as AllCoreStateSummariesProto,
    CurrentCoreStateSummary as CoreStallSummaryProto,
    StallArtifactMetadata as StallArtifactMetadataProto,
    TpuCoreIdentifier as TelemetryCoreIdentifierProto,
};
pub use super::descriptor::generated_prost_schemas::platforms_deepsea::jellyfish::xdb::{
    AllocationMetadata as XdbAllocationMetadataProto, CompilerMetadata as CompilerMetadataProto,
    SymbolTable as CompilerSymbolTableProto,
};
pub use super::descriptor::generated_prost_schemas::tensorflow::tf2xla::{
    Config as Tf2XlaConfigProto, Feed as Tf2XlaFeedProto, Fetch as Tf2XlaFetchProto,
    HostComputeMetadata as HostComputeMetadataProto,
    HostTransferMetadata as HostTransferMetadataProto,
    TensorId as Tf2XlaTensorIdProto, TensorMetadata as HostTransferTensorMetadataProto,
    Variable as Tf2XlaVariableProto, XlaArgumentProto,
};
pub use super::descriptor::generated_prost_schemas::tensorflow::tpu::{
    AdadeltaParameters as AdadeltaParametersProto,
    AdagradMomentumParameters as AdagradMomentumParametersProto,
    AdagradParameters as AdagradParametersProto, AdamParameters as AdamParametersProto,
    AssignParameters as AssignParametersProto,
    BoundedAdagradParameters as BoundedAdagradParametersProto,
    CenteredRmsPropParameters as CenteredRmsPropParametersProto,
    ClippingLimits as ClippingLimitsProto, CompilationResultProto,
    FrequencyAwareAdagradParameters as FrequencyAwareAdagradParametersProto,
    FrequencyEstimatorParameters as FrequencyEstimatorParametersProto,
    FtrlParameters as FtrlParametersProto,
    GradientAccumulationStatus as GradientAccumulationStatusProto,
    HotIdReplicationConfiguration as HotIdReplicationConfigurationProto,
    LearningRate as LearningRateProto, LionParameters as LionParametersProto,
    LowDimensionalPackingStatus as LowDimensionalPackingStatusProto,
    MdlAdagradLightParameters as MdlAdagradLightParametersProto,
    MomentumParameters as MomentumParametersProto,
    OnlineYogiParameters as OnlineYogiParametersProto,
    OptimizationParameters as OptimizationParametersProto,
    OptimizerDynamicInput as OptimizerDynamicInputProto, PaddingMap as PaddingMapProto,
    ProximalAdagradParameters as ProximalAdagradParametersProto,
    ProximalYogiParameters as ProximalYogiParametersProto,
    RmsPropParameters as RmsPropParametersProto,
    SimulatedQuantization as SimulatedQuantizationProto,
    SparseCoreTableLayout as SparseCoreTableLayoutProto,
    SparseCoreTableLayouts as SparseCoreTableLayoutsProto,
    StateVariableSpecification as StateVariableSpecificationProto,
    StochasticGradientDescentParameters as StochasticGradientDescentParametersProto,
    TopologyProto, TpuCompileMetadataProto,
    TpuEmbeddingConfiguration as TpuEmbeddingConfigurationProto,
    TpuHardwareFeature as TpuHardwareFeatureProto,
    UserDefinedProgramParameters as UserDefinedProgramParametersProto,
};
pub use super::descriptor::generated_prost_schemas::tensorflow::{
    ResourceHandleProto, TensorProto, TensorShapeProto, VariantTensorDataProto,
};
pub use super::descriptor::generated_prost_schemas::tpu::TpuTopologyArgsProto;
pub use super::descriptor::generated_prost_schemas::xla::jellyfish::{
    AutoProto, DbQueryType as JellyfishDbQueryTypeProto,
    EmitterLearnedCostModelOptions as EmitterLearnedCostModelOptionsProto,
    LayoutProfileProto as JellyfishLayoutProfileProto,
    LearnedCostModelClientOptions as JellyfishLearnedCostModelClientOptionsProto,
    LearnedCostModelMode as JellyfishLearnedCostModelModeProto,
    MemorySpaceProto as JellyfishMemorySpaceProto,
    MlOutputValidationStrategy as JellyfishMlOutputValidationStrategyProto,
    OpLayoutProposedProto as JellyfishOpLayoutProposedProto,
    ShardyOptions as JellyfishShardyOptionsProto,
    TpuCompilationEnvironment as TpuCompilationEnvironmentProto, auto_proto,
};
pub use super::descriptor::generated_prost_schemas::xla::memory_space_assignment::{
    ExpandedScopedAlternateMemoryMode, HloOperandFilter as MsaHloOperandFilterProto,
    HloPositionMatcher as MsaHloPositionMatcherProto,
    KeepInDefaultMemoryAction as MsaKeepInDefaultMemoryActionProto,
    MemoryBoundLoopOptimizerOptions as MemoryBoundLoopOptimizerOptionsProto,
    MsaSortOrderOverride as MsaSortOrderOverrideProto,
    MsaSortOrderOverrideOptions as MsaSortOrderOverrideOptionsProto,
    MsaSortOrderOverrides as MsaSortOrderOverridesProto,
    MsaTensorOverride as MsaTensorOverrideProto, MsaTensorOverrides as MsaTensorOverridesProto,
    PinInAlternateMemoryAction as MsaPinInAlternateMemoryActionProto,
    PreferredPrefetchOverride as PreferredPrefetchOverrideProto,
    PreferredPrefetchOverrideOptions as PreferredPrefetchOverrideOptionsProto,
    PreferredPrefetchOverrides as PreferredPrefetchOverridesProto,
    SlicedPrefetchOptions as SlicedPrefetchOptionsProto,
    msa_sort_order_override_options, msa_tensor_override, preferred_prefetch_override_options,
};

/// Alias for `tensorflow.TensorShapeProto.Dim`.
pub type TensorShapeDimProto =
    super::descriptor::generated_prost_schemas::tensorflow::tensor_shape_proto::Dim;

/// Alias for `tensorflow.ResourceHandleProto.DtypeAndShape`.
pub type ResourceDtypeAndShapeProto =
    super::descriptor::generated_prost_schemas::tensorflow::resource_handle_proto::DtypeAndShape;

/// Alias for `tensorflow.tf2xla.XlaArgumentProto.Shape`.
pub type XlaArgumentShapeProto =
    super::descriptor::generated_prost_schemas::tensorflow::tf2xla::xla_argument_proto::Shape;

/// Alias for `tensorflow.tpu.TPUCompileMetadataProto.Arg`.
pub type TpuCompileMetadataArgProto =
    super::descriptor::generated_prost_schemas::tensorflow::tpu::tpu_compile_metadata_proto::Arg;

/// Alias for `tensorflow.tpu.TPUCompileMetadataProto.Retval`.
pub type TpuCompileMetadataRetvalProto =
    super::descriptor::generated_prost_schemas::tensorflow::tpu::tpu_compile_metadata_proto::Retval;

/// Alias for `tensorflow.tpu.TPUEmbeddingConfiguration.TableDescriptor`.
pub type EmbeddingTableDescriptorProto = super::descriptor::generated_prost_schemas::tensorflow::tpu::tpu_embedding_configuration::TableDescriptor;

/// Alias for `tensorflow.tpu.TPUEmbeddingConfiguration.FeatureDescriptor`.
pub type EmbeddingFeatureDescriptorProto = super::descriptor::generated_prost_schemas::tensorflow::tpu::tpu_embedding_configuration::FeatureDescriptor;

/// Alias for `tensorflow.tpu.TPUEmbeddingConfiguration.SpmdSharding`.
pub type EmbeddingSpmdShardingProto = super::descriptor::generated_prost_schemas::tensorflow::tpu::tpu_embedding_configuration::SpmdSharding;

/// Alias for `tensorflow.tpu.StateVariableSpecification.UserDefined`.
pub type StateVariableUserDefinedProto = super::descriptor::generated_prost_schemas::tensorflow::tpu::state_variable_specification::UserDefined;

/// Alias for `tensorflow.tpu.StateVariableSpecification.FillWithConstant`.
pub type StateVariableFillWithConstantProto = super::descriptor::generated_prost_schemas::tensorflow::tpu::state_variable_specification::FillWithConstant;

/// Static descriptor specification for a `TpuCompilationEnvironment` compiler flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JellyfishCompilationEnvFieldSpec {
    /// Protobuf field tag number.
    pub tag: u32,
    /// Canonical `--xla_tpu_*` flag name without leading dashes.
    pub name: &'static str,
    /// Protobuf scalar/enum wire type name (`"bool"`, `"int32"`, `"int64"`, `"double"`, `"string"`).
    pub wire_type: &'static str,
    /// Canonical default value literal string.
    pub default_value: &'static str,
}

/// Validated table of core `xla.jellyfish.TpuCompilationEnvironment` and runtime CLI flags.
pub static JELLYFISH_COMPILATION_ENV_FIELD_SPECS: [JellyfishCompilationEnvFieldSpec; 18] = [
    JellyfishCompilationEnvFieldSpec {
        tag: 2,
        name: "xla_tpu_sdc_checker_instrument_megacore_fusion",
        wire_type: "bool",
        default_value: "false",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 4,
        name: "xla_tpu_allocate_scoped_vmem_at_same_offset",
        wire_type: "bool",
        default_value: "true",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 14,
        name: "xla_tpu_max_cmem_used_by_memory_space_assignment",
        wire_type: "int64",
        default_value: "-1",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 81,
        name: "xla_tpu_autotune_windows",
        wire_type: "bool",
        default_value: "true",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 116,
        name: "xla_jf_use_rng_bit_generator_emitter",
        wire_type: "bool",
        default_value: "true",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 117,
        name: "xla_tpu_expand_rng_bit_generator",
        wire_type: "bool",
        default_value: "true",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 150,
        name: "xla_tpu_enable_latency_hiding_scheduler",
        wire_type: "bool",
        default_value: "true",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 153,
        name: "xla_tpu_spmd_auto_partitioning",
        wire_type: "bool",
        default_value: "true",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 241,
        name: "xla_jf_enable_pipelining",
        wire_type: "bool",
        default_value: "true",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 418,
        name: "xla_tpu_scoped_vmem_limit_kib",
        wire_type: "int64",
        default_value: "16384",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 629,
        name: "xla_tpu_msa_reduce_scoped_vmem_limit",
        wire_type: "bool",
        default_value: "false",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 10001,
        name: "xla_tpu_enable_megacore_fusion",
        wire_type: "bool",
        default_value: "true",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 10002,
        name: "xla_tpu_enable_async_collective_fusion",
        wire_type: "bool",
        default_value: "true",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 10003,
        name: "xla_tpu_spmd_rng_bit_generator_unsafe",
        wire_type: "bool",
        default_value: "false",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 10004,
        name: "xla_tpu_use_shardy_partitioner",
        wire_type: "bool",
        default_value: "true",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 10005,
        name: "xla_tpu_autotune_level",
        wire_type: "int32",
        default_value: "2",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 10006,
        name: "xla_tpu_xplane_trace_level",
        wire_type: "int32",
        default_value: "1",
    },
    JellyfishCompilationEnvFieldSpec {
        tag: 10007,
        name: "xla_tpu_enable_windowed_einsum",
        wire_type: "bool",
        default_value: "true",
    },
];

/// Extension helpers on [`TpuCompilationEnvironmentProto`] for dynamic descriptor reflection and CLI flag sync.
pub trait TpuCompilationEnvironmentProtoExt: Sized {
    /// Looks up a [`JellyfishCompilationEnvFieldSpec`] by flag name (with or without `--`) or numeric tag.
    ///
    /// # Arguments
    ///
    /// * `name_or_tag` - Flag name or numeric field tag string.
    #[must_use]
    fn lookup_jellyfish_field_spec(
        name_or_tag: &str,
    ) -> Option<&'static JellyfishCompilationEnvFieldSpec>;

    /// Returns all real field names defined on `xla.jellyfish.TpuCompilationEnvironment` in `libtpu.so`.
    #[must_use]
    fn list_jellyfish_schema_field_names() -> Vec<&'static str>;

    /// Resolves a [`prost_reflect::MessageDescriptor`] and [`prost_reflect::DynamicMessage`]
    /// for `full_name` using the wire encoding of `self`.
    ///
    /// # Arguments
    ///
    /// * `full_name` - Fully-qualified protobuf message name in the global descriptor pool.
    #[must_use]
    fn reflect_as_message_by_name(
        &self,
        full_name: &str,
    ) -> Option<(
        prost_reflect::MessageDescriptor,
        prost_reflect::DynamicMessage,
    )>;

    /// Reads any field on `xla.jellyfish.TpuCompilationEnvironment` (by name or numeric tag) as a string,
    /// falling back to [`Self::lookup_jellyfish_field_spec`] defaults or `"default"`.
    ///
    /// # Arguments
    ///
    /// * `field_name_or_tag` - Field name (with or without `--`) or decimal tag string.
    #[must_use]
    fn get_field_as_string(&self, field_name_or_tag: &str) -> String;

    /// Reads `field_name_or_tag` against `schema_name` in the global descriptor pool.
    ///
    /// # Arguments
    ///
    /// * `schema_name` - Fully-qualified protobuf message name.
    /// * `field_name_or_tag` - Field name or numeric tag string.
    #[must_use]
    fn get_field_as_string_in_schema(&self, schema_name: &str, field_name_or_tag: &str) -> String;

    /// Sets a field on `xla.jellyfish.TpuCompilationEnvironment` (by name or numeric tag) from a
    /// CLI string value, storing unrecognized custom key/value pairs in the sweep profile annotation.
    ///
    /// # Arguments
    ///
    /// * `field_name_or_tag` - Field name (with or without `--`) or decimal tag string.
    /// * `raw_value` - String value to parse and assign.
    fn set_field_from_str(&mut self, field_name_or_tag: &str, raw_value: &str);

    /// Sets `field_name_or_tag` against `schema_name` in the global descriptor pool.
    ///
    /// # Arguments
    ///
    /// * `schema_name` - Fully-qualified protobuf message name.
    /// * `field_name_or_tag` - Field name or numeric tag string.
    /// * `raw_value` - String value to parse and assign.
    fn set_field_from_str_in_schema(
        &mut self,
        schema_name: &str,
        field_name_or_tag: &str,
        raw_value: &str,
    );

    /// Parses a whitespace-delimited CLI flag string (`--xla_tpu_foo=bar --xla_tpu_bool`) into a
    /// [`TpuCompilationEnvironmentProto`].
    ///
    /// # Arguments
    ///
    /// * `cli_flags` - Whitespace-separated `--key=value` or `--bool_flag` arguments.
    #[must_use]
    fn from_cli_flags(cli_flags: &str) -> Self;

    /// Exports all explicitly set fields on this [`TpuCompilationEnvironmentProto`] into `config`.
    ///
    /// # Arguments
    ///
    /// * `config` - Target [`TpuInitConfig`] to update.
    fn apply_to_init_config(&self, config: &mut TpuInitConfig);

    /// Exports set fields from `schema_name` on `self` into `config`.
    ///
    /// # Arguments
    ///
    /// * `schema_name` - Fully-qualified protobuf message name.
    /// * `config` - Target [`TpuInitConfig`] to update.
    fn apply_to_init_config_in_schema(&self, schema_name: &str, config: &mut TpuInitConfig);
}

impl TpuCompilationEnvironmentProtoExt for TpuCompilationEnvironmentProto {
    fn lookup_jellyfish_field_spec(
        name_or_tag: &str,
    ) -> Option<&'static JellyfishCompilationEnvFieldSpec> {
        let key = name_or_tag.trim_start_matches('-');
        key.parse::<u32>().map_or_else(
            |_| {
                JELLYFISH_COMPILATION_ENV_FIELD_SPECS
                    .iter()
                    .find(|s| s.name == key)
            },
            |tag_num| {
                JELLYFISH_COMPILATION_ENV_FIELD_SPECS
                    .iter()
                    .find(|s| s.tag == tag_num)
            },
        )
    }

    fn list_jellyfish_schema_field_names() -> Vec<&'static str> {
        let pool = crate::proto::descriptor::global_descriptor_pool();
        pool.get_message_by_name(Self::PROTO_FULL_NAME)
            .into_iter()
            .flat_map(|desc| {
                desc.fields()
                    .map(|fd| {
                        JELLYFISH_COMPILATION_ENV_FIELD_SPECS
                            .iter()
                            .find(|s| s.name == fd.name())
                            .map_or("xla_tpu_scoped_vmem_limit_kib", |s| s.name)
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn reflect_as_message_by_name(
        &self,
        full_name: &str,
    ) -> Option<(
        prost_reflect::MessageDescriptor,
        prost_reflect::DynamicMessage,
    )> {
        use prost::Message as _;
        let desc =
            crate::proto::descriptor::global_descriptor_pool().get_message_by_name(full_name)?;
        let mut dyn_msg = prost_reflect::DynamicMessage::new(desc.clone());
        let _ = dyn_msg.merge(self.encode_proto().as_slice());
        Some((desc, dyn_msg))
    }

    fn get_field_as_string(&self, field_name_or_tag: &str) -> String {
        self.get_field_as_string_in_schema(Self::PROTO_FULL_NAME, field_name_or_tag)
    }

    fn get_field_as_string_in_schema(&self, schema_name: &str, field_name_or_tag: &str) -> String {
        use prost_reflect::{Kind, ReflectMessage as _, Value};
        let clean = field_name_or_tag.trim_start_matches('-');
        if let Some((desc, dyn_msg)) = self.reflect_as_message_by_name(schema_name) {
            let field_opt = clean
                .parse::<u32>()
                .ok()
                .and_then(|tag| desc.get_field(tag))
                .or_else(|| desc.get_field_by_name(clean));
            if let Some(field) = field_opt {
                if dyn_msg.has_field(&field) {
                    let val = dyn_msg.get_field(&field);
                    return match val.as_ref() {
                        Value::Bool(v) => v.to_string(),
                        Value::I32(v) | Value::EnumNumber(v) => v.to_string(),
                        Value::I64(v) => v.to_string(),
                        Value::U32(v) => v.to_string(),
                        Value::U64(v) => v.to_string(),
                        Value::F32(v) => v.to_string(),
                        Value::F64(v) => v.to_string(),
                        Value::String(s) => s.clone(),
                        Value::Message(sub)
                            if sub.descriptor().full_name() == "xla.jellyfish.AutoProto" =>
                        {
                            for af in sub.descriptor().fields() {
                                if sub.has_field(&af) {
                                    return match sub.get_field(&af).as_ref() {
                                        Value::Bool(v) => v.to_string(),
                                        Value::I64(v) => v.to_string(),
                                        Value::U64(v) => v.to_string(),
                                        Value::I32(v) => v.to_string(),
                                        Value::U32(v) => v.to_string(),
                                        Value::F64(v) => v.to_string(),
                                        Value::F32(v) => v.to_string(),
                                        Value::String(s) => s.clone(),
                                        _ => String::from("set"),
                                    };
                                }
                            }
                            String::from("auto")
                        }
                        _ => String::from("set"),
                    };
                }
                if field.kind() == Kind::Bool {
                    return String::from("false");
                }
            }
        }
        if let Some(custom_path) = &self.xla_tpu_scoped_vmem_limit_sweep_profile_path {
            for kv in custom_path.split(';') {
                if let Some((k, v)) = kv.split_once('=') {
                    if k == clean {
                        return String::from(v);
                    }
                }
            }
        }
        if let Some(spec) = Self::lookup_jellyfish_field_spec(clean) {
            return String::from(spec.default_value);
        }
        String::from("default")
    }

    fn set_field_from_str(&mut self, field_name_or_tag: &str, raw_value: &str) {
        self.set_field_from_str_in_schema(Self::PROTO_FULL_NAME, field_name_or_tag, raw_value);
    }

    fn set_field_from_str_in_schema(
        &mut self,
        schema_name: &str,
        field_name_or_tag: &str,
        raw_value: &str,
    ) {
        use prost::Message as _;
        use prost_reflect::{DynamicMessage, Kind, Value};
        let clean = field_name_or_tag.trim_start_matches('-');
        if let Some((desc, mut dyn_msg)) = self.reflect_as_message_by_name(schema_name) {
            let field_opt = clean
                .parse::<u32>()
                .ok()
                .and_then(|tag| desc.get_field(tag))
                .or_else(|| desc.get_field_by_name(clean));
            if let Some(field) = field_opt {
                let parsed_val = match field.kind() {
                    Kind::Bool => Some(Value::Bool(matches!(
                        raw_value.trim(),
                        "true" | "1" | "yes" | "True" | "TRUE"
                    ))),
                    Kind::Int32 | Kind::Sint32 | Kind::Sfixed32 => {
                        raw_value.trim().parse::<i32>().ok().map(Value::I32)
                    }
                    Kind::Int64 | Kind::Sint64 | Kind::Sfixed64 => {
                        raw_value.trim().parse::<i64>().ok().map(Value::I64)
                    }
                    Kind::Uint32 | Kind::Fixed32 => {
                        raw_value.trim().parse::<u32>().ok().map(Value::U32)
                    }
                    Kind::Uint64 | Kind::Fixed64 => {
                        raw_value.trim().parse::<u64>().ok().map(Value::U64)
                    }
                    Kind::Float => raw_value.trim().parse::<f32>().ok().map(Value::F32),
                    Kind::Double => raw_value.trim().parse::<f64>().ok().map(Value::F64),
                    Kind::String => Some(Value::String(String::from(raw_value))),
                    Kind::Enum(en_desc) => raw_value
                        .trim()
                        .parse::<i32>()
                        .ok()
                        .or_else(|| {
                            en_desc
                                .get_value_by_name(raw_value.trim())
                                .map(|v| v.number())
                        })
                        .map(Value::EnumNumber),
                    Kind::Message(sub_desc)
                        if sub_desc.full_name() == "xla.jellyfish.AutoProto" =>
                    {
                        let mut auto_msg = DynamicMessage::new(sub_desc.clone());
                        let trimmed = raw_value.trim();
                        let (target_name, target_val) = if matches!(trimmed, "true" | "false") {
                            ("b", Value::Bool(trimmed == "true"))
                        } else if let Ok(iv) = trimmed.parse::<i64>() {
                            ("i64", Value::I64(iv))
                        } else if let Ok(dv) = trimmed.parse::<f64>() {
                            ("d", Value::F64(dv))
                        } else {
                            ("s", Value::String(String::from(trimmed)))
                        };
                        for af in sub_desc.fields() {
                            if af.name() == target_name {
                                auto_msg.set_field(&af, target_val.clone());
                            }
                        }
                        Some(Value::Message(auto_msg))
                    }
                    _ => None,
                };
                if let Some(val) = parsed_val {
                    dyn_msg.set_field(&field, val);
                    *self = Self::decode_proto(&dyn_msg.encode_to_vec()).unwrap_or_default();
                    return;
                }
            }
        }
        let entry = if clean == "custom_flags" {
            String::from(raw_value)
        } else {
            format!("{clean}={raw_value}")
        };
        let path = self
            .xla_tpu_scoped_vmem_limit_sweep_profile_path
            .get_or_insert_with(String::new);
        if !path.is_empty() {
            path.push(';');
        }
        path.push_str(&entry);
    }

    fn from_cli_flags(cli_flags: &str) -> Self {
        let mut env = Self::default();
        for token in cli_flags.split_whitespace() {
            let stripped = token.trim_start_matches('-');
            if stripped.is_empty() {
                continue;
            }
            if let Some((k, v)) = stripped.split_once('=') {
                env.set_field_from_str(k, v);
            } else {
                env.set_field_from_str(stripped, "true");
            }
        }
        env
    }

    fn apply_to_init_config(&self, config: &mut TpuInitConfig) {
        self.apply_to_init_config_in_schema(Self::PROTO_FULL_NAME, config);
    }

    fn apply_to_init_config_in_schema(&self, schema_name: &str, config: &mut TpuInitConfig) {
        if let Some(vmem_kib) = self.xla_tpu_scoped_vmem_limit_kib.filter(|&v| v > 0) {
            let kib_u64 = u64::try_from(vmem_kib).unwrap_or(16384);
            config.xla_tpu_scoped_vmem_limit_kib = kib_u64;
            config.extra_flags.insert(
                String::from("xla_tpu_scoped_vmem_limit_kib"),
                kib_u64.to_string(),
            );
        }
        if let Some((desc, dyn_msg)) = self.reflect_as_message_by_name(schema_name) {
            for field in desc.fields() {
                if field.name() == "xla_tpu_scoped_vmem_limit_sweep_profile_path" {
                    continue;
                }
                if dyn_msg.has_field(&field) {
                    let val_str = self.get_field_as_string(field.name());
                    config
                        .extra_flags
                        .insert(String::from(field.name()), val_str);
                }
            }
        }
        if let Some(custom_kv) = &self.xla_tpu_scoped_vmem_limit_sweep_profile_path {
            for pair in custom_kv.split(';') {
                if let Some((k, v)) = pair.split_once('=') {
                    if k == "xla_tpu_scoped_vmem_limit_kib" {
                        if let Ok(parsed) = v.parse::<u64>() {
                            config.xla_tpu_scoped_vmem_limit_kib = parsed;
                        }
                    }
                    config.extra_flags.insert(String::from(k), String::from(v));
                }
            }
        }
    }
}

impl TpuEmbeddingPartitionedConfigProto {
    /// Serializes this [`TpuEmbeddingPartitionedConfigProto`] into Protobuf wire bytes.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_proto()
    }

    /// Deserializes a [`TpuEmbeddingPartitionedConfigProto`] from Protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `TpuEmbeddingPartitionedConfigProto` payload.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> crate::error::LibTpuResult<Self> {
        Self::decode_proto(bytes)
    }
}

impl TpuEmbeddingMemoryConfigProto {
    /// Serializes this [`TpuEmbeddingMemoryConfigProto`] into Protobuf wire bytes.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_proto()
    }

    /// Deserializes a [`TpuEmbeddingMemoryConfigProto`] from Protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `TpuEmbeddingMemoryConfigProto` payload.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> crate::error::LibTpuResult<Self> {
        Self::decode_proto(bytes)
    }
}

impl TpuEmbeddingHostNetworkConfigProto {
    /// Serializes this [`TpuEmbeddingHostNetworkConfigProto`] into Protobuf wire bytes.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_proto()
    }

    /// Deserializes a [`TpuEmbeddingHostNetworkConfigProto`] from Protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `TpuEmbeddingHostNetworkConfigProto` payload.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> crate::error::LibTpuResult<Self> {
        Self::decode_proto(bytes)
    }
}

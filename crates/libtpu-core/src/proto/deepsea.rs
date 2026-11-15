//! `DeepSea` / Jellyfish / Pufferfish / Viperfish / Ghostfish / Ironwood ISA, core programs,
//! host commands, TPU executables, GCSR hardware registers, `BarnaCore` models, and error sources.
//!
//! Re-exports `prost-build` generated structs from `platforms/xla/service/jellyfish/`,
//! `learning/45eac/research/pjrt/tpu_executable.proto`, `learning/45eac/tpu/runtime/`,
//! `hardware/chips/infra/gcsr/lib/gcsr.proto`, `platforms/asic_sw/`, and
//! `platforms/performance/deepsea/barna_core_simulator/barna_core_info.proto`.

use super::descriptor::ProtoSchemaMessage as _;
use crate::error::LibTpuResult;

pub use super::descriptor::generated_prost_schemas::asic_sw::deepsea::gxc::gfc::isa::{
    SparseCoreScsProgram as IronwoodSparseCoreScsProgramProto,
    SparseCoreTecBundle as IronwoodSparseCoreTecBundleProto,
    SparseCoreTecProgram as IronwoodSparseCoreTecProgramProto,
    TensorCoreBundle as IronwoodTensorCoreBundleProto,
    TensorCoreProgram as IronwoodTensorCoreProgramProto,
};
pub use super::descriptor::generated_prost_schemas::asic_sw::deepsea::gxc::glc::isa::{
    SparseCoreScsProgram as GhostliteSparseCoreScsProgramProto,
    SparseCoreTacProgram as GhostliteSparseCoreTacProgramProto,
    SparseCoreTecBundle as GhostliteSparseCoreTecBundleProto,
    SparseCoreTecProgram as GhostliteSparseCoreTecProgramProto,
    TensorCoreBundle as GhostliteTensorCoreBundleProto,
    TensorCoreProgram as GhostliteTensorCoreProgramProto,
};
pub use super::descriptor::generated_prost_schemas::asic_sw::deepsea::pxc::isa::{
    TensorCoreBundle as PufferfishTensorCoreBundleProto,
    TensorCoreProgram as PufferfishTensorCoreProgramProto,
};
pub use super::descriptor::generated_prost_schemas::asic_sw::deepsea::pxc::pfc::isa::{
    BarnaCoreChannelBundle as PufferfishBarnaCoreChannelBundleProto,
    BarnaCoreChannelProgram as PufferfishBarnaCoreChannelProgramProto,
    BarnaCoreSequencerProgram as PufferfishBarnaCoreSequencerProgramProto,
};
pub use super::descriptor::generated_prost_schemas::asic_sw::deepsea::vxc::isa::{
    TensorCoreBundle as ViperfishTensorCoreBundleProto,
    TensorCoreProgram as ViperfishTensorCoreProgramProto,
};
pub use super::descriptor::generated_prost_schemas::asic_sw::deepsea::vxc::vfc::isa::{
    SparseCoreScsProgram as ViperfishSparseCoreScsProgramProto,
    SparseCoreTacProgram as ViperfishSparseCoreTacProgramProto,
    SparseCoreTecBundle as ViperfishSparseCoreTecBundleProto,
    SparseCoreTecProgram as ViperfishSparseCoreTecProgramProto,
};
pub use super::descriptor::generated_prost_schemas::asic_sw::generators::errors::{
    ErrorSources, GeneralErrorSource, MemoryEccErrorSource, general_error_source,
    memory_ecc_error_source,
};
pub use super::descriptor::generated_prost_schemas::asic_sw::tools::{
    EncodedMemory, EncodedNode, EncodedRegister, RegisterTranslatorData,
    RegisterTranslatorData as RegisterTranslatorDataProto,
};
pub use super::descriptor::generated_prost_schemas::hardware::chips::infra::gcsr::gcsrproto::{
    Block as GcsrBlock, FieldConstraint, MemField as GcsrMemField, Node as GcsrNode, RegField,
};
pub use super::descriptor::generated_prost_schemas::libtpu::ext::deepsea::{
    BarnaCoreChannelConfigProto, BarnaCoreExecutableProto, DmaDescriptorRingConfigProto,
    HostCommandEnvelopeProto, IciRouteHopProto, IciRoutingTableProto, LloBundleScheduleProto,
    LloLiveIntervalProto, LloModuloStageProto, LloRegisterAllocationProto,
    LloScheduledInstructionProto, LloSoftwarePipelineScheduleProto, LloSpillSlotProto,
    SparseCoreDmaDescriptorProto, SparseCoreProgramProto, SparseCoreTileExecutionScheduleProto,
    SparseCoreV3ExecutableConfigProto, TpuSerializedExecutableProto, VliwBundleEncodingSpecProto,
};
pub use super::descriptor::generated_prost_schemas::platforms_deepsea::jellyfish::isa::{
    BarnaCoreAddressHandlerBundle as JellyfishBarnaCoreBundleProto,
    BarnaCoreAddressHandlerProgram as JellyfishBarnaCoreProgramProto,
    BarnaCoreFsmProgram as JellyfishBarnaCoreFsmProgramProto, Bundle as JellyfishIsaBundleProto,
    Program as JellyfishIsaProgramProto,
};
pub use super::descriptor::generated_prost_schemas::platforms_performance_deepsea::barna_core::{
    BarnaCoreConstantParameters, BarnaCoreModelParameters,
};
pub use super::descriptor::generated_prost_schemas::riegeli::RecordsMetadata as RiegeliRecordsMetadataProto;
pub use super::descriptor::generated_prost_schemas::tpu::{
    HostCommandOpcodeProto, TpuChipConfigProto, TpuChipPartsProto, TpuConfiguredPropertiesProto,
    TpuCoreProgramAbiOptionsProto, TpuCoreProgramAbiProto, TpuCoreProgramProto, TpuCoreTypeProto,
    TpuDimensionsProto, TpuMemorySegmentProto, TpuMemorySegmentSetProto, TpuRuntimeAbiProto,
    TpuSequencerProgramProto, TpuVersionProto, tpu_memory_segment_proto,
};
pub use super::descriptor::generated_prost_schemas::xla::ffi::{
    ArrayProto as FfiArrayProto, ExecutionStateProto as FfiExecutionStateProto,
    FlatAttributeProto as FfiFlatAttributeProto,
    FlatAttributesMapProto as FfiFlatAttributesMapProto, ScalarProto as FfiScalarProto,
    flat_attribute_proto,
};
pub use super::descriptor::generated_prost_schemas::xla::jellyfish::{
    BundlePackerInfoProto, LloAllocationProto, LloConstantProto, LloInstructionProto, LloLoopProto,
    LloModuleProto, LloOpcodeProto, LloPredicatedRegionProto, LloRegionProto,
    MatmulInstructionDataProto, RegisterTypeProto as LloRegisterTypeProto,
    StaticPerBundleUtilizationProto, TransposeMetadataProto, llo_instruction_proto, llo_loop_proto,
};
pub use super::descriptor::generated_prost_schemas::xla::tpu::sparse_core::isa_emitter::{
    c_6acc60406::ProgramWrapper as IronwoodSparseCoreProgramWrapperProto,
    ghostlite::ProgramWrapper as GhostliteSparseCoreProgramWrapperProto,
    viperfish::ProgramWrapper as ViperfishSparseCoreProgramWrapperProto,
};
pub use super::descriptor::generated_prost_schemas::xla::{
    CallTargetInfo as IsaCallTargetInfoProto, DeepseaExecutableProto, HostExecutionProto,
    HostOffloadingExecutableProto, HostTransferProto, IsaProgramProto, LloAddressProto,
    LoadableSegmentProto, MemorySpaceProto as IsaMemorySpaceProto,
    ProgramRelocation as ProgramRelocationProto, TargetArgumentsProto, TpuExecutableProto,
    TpuRelocatableProto, UninitializedSegmentProto, isa_program_proto, tpu_relocatable_proto,
};

/// Alias for `asic_sw.generators.errors.GeneralErrorSource.GeneralErrorClass`.
pub type GeneralErrorClass = general_error_source::GeneralErrorClass;

/// Alias for `asic_sw.generators.errors.GeneralErrorSource.EccUncorrectableErrorClass`.
pub type EccUncorrectableErrorClass = general_error_source::EccUncorrectableErrorClass;

/// Alias for `asic_sw.generators.errors.MemoryEccErrorSource.Offsets`.
pub type MemoryEccOffsets = memory_ecc_error_source::Offsets;

/// Alias for `hardware.chips.infra.gcsr.gcsrproto.Node.Register` (`*_pf_csrs.br` register payload).
pub type GcsrRegister =
    super::descriptor::generated_prost_schemas::hardware::chips::infra::gcsr::gcsrproto::node::Register;

/// Alias for `platforms_performance_deepsea.barna_core.BarnaCoreConstantParameters.AddressHandlerConfig`.
pub type BarnaCoreAddressHandlerConfig = super::descriptor::generated_prost_schemas::platforms_performance_deepsea::barna_core::barna_core_constant_parameters::AddressHandlerConfig;

/// Alias for `platforms_performance_deepsea.barna_core.BarnaCoreConstantParameters.HbmUtilization`.
pub type BarnaCoreHbmUtilization = super::descriptor::generated_prost_schemas::platforms_performance_deepsea::barna_core::barna_core_constant_parameters::HbmUtilization;

/// Alias for `platforms_performance_deepsea.barna_core.BarnaCoreConstantParameters.SparseReduceHeaderOverhead`.
pub type BarnaCoreSparseReduceHeaderOverhead = super::descriptor::generated_prost_schemas::platforms_performance_deepsea::barna_core::barna_core_constant_parameters::SparseReduceHeaderOverhead;

/// Alias for `tpu.TpuMemorySegmentProto.Range`.
pub type TpuMemorySegmentRangeProto = tpu_memory_segment_proto::Range;

/// Extension helpers on [`ProgramRelocationProto`] for live `IMEM` IOVA patching.
pub trait ProgramRelocationProtoExt {
    /// Patches live parameter and output IOVA addresses (`param_iovas`, `output_iovas`) into `imem_bytes`
    /// at the 64-byte bundle `self.bundle` and 4-byte slot displacement `self.slot`.
    ///
    /// # Arguments
    ///
    /// * `imem_bytes` - Mutable `IMEM` VLIW instruction stream.
    /// * `param_iovas` - Live device HBM/IOVA addresses for program input parameters (`kind == 0`).
    /// * `output_iovas` - Live device HBM/IOVA addresses for program output buffers (`kind != 0`).
    fn apply_to_imem(&self, imem_bytes: &mut [u8], param_iovas: &[u64], output_iovas: &[u64]);
}

impl ProgramRelocationProtoExt for ProgramRelocationProto {
    fn apply_to_imem(&self, imem_bytes: &mut [u8], param_iovas: &[u64], output_iovas: &[u64]) {
        let idx = usize::try_from(self.memory_sub_space.max(0)).unwrap_or(0);
        let base_addr = if self.kind == 0 {
            param_iovas.get(idx).copied().unwrap_or(0)
        } else {
            output_iovas.get(idx).copied().unwrap_or(0)
        };
        #[allow(clippy::cast_possible_truncation)]
        let patched_u32 = (base_addr & 0xFFFF_FFFF) as u32;
        let bundle_off = usize::try_from(self.bundle.max(0))
            .unwrap_or(0)
            .saturating_mul(64);
        let slot_off = usize::try_from(self.slot.max(0))
            .unwrap_or(0)
            .saturating_mul(4);
        let byte_off = bundle_off.saturating_add(slot_off);
        imem_bytes
            .get_mut(byte_off..byte_off.saturating_add(4))
            .into_iter()
            .for_each(|dst| dst.copy_from_slice(&patched_u32.to_le_bytes()));
        if let Some(b_slice) = imem_bytes.get_mut(bundle_off..bundle_off.saturating_add(64))
            && matches!(b_slice.first().copied(), Some(0x66 | 0x77))
        {
            let tag = crate::compiler::isa::compute_bundle_highway_tag8(b_slice);
            b_slice.get_mut(7).into_iter().for_each(|dst| *dst = tag);
        }
    }
}

/// Extension factory constructing hardware descriptor protos from a [`crate::types::TpuGeneration`].
pub trait TpuGenerationProtoExt: Sized {
    /// Constructs a canonical descriptor populated for `generation`.
    ///
    /// # Arguments
    ///
    /// * `generation` - Cloud TPU hardware generation.
    #[must_use]
    fn from_generation(generation: crate::types::TpuGeneration) -> Self;
}

impl TpuGenerationProtoExt for TpuChipConfigProto {
    fn from_generation(generation: crate::types::TpuGeneration) -> Self {
        use crate::types::TpuGeneration;
        let megacore = matches!(
            generation,
            TpuGeneration::V4 | TpuGeneration::V5p | TpuGeneration::V7x
        );
        Self {
            version: generation as i32,
            chip_config_abi_version: 1,
            megacore,
            ..Self::default()
        }
    }
}

impl TpuGenerationProtoExt for TpuChipPartsProto {
    fn from_generation(generation: crate::types::TpuGeneration) -> Self {
        use crate::types::TpuGeneration;
        let variant_name = match generation {
            TpuGeneration::V2 => "jellyfish",
            TpuGeneration::V3 => "dragonfish",
            TpuGeneration::V4 => "pufferfish",
            TpuGeneration::V4Lite => "pufferfish_lite",
            TpuGeneration::V5e => "viperfish_lite",
            TpuGeneration::V5p => "viperfish",
            TpuGeneration::V6e => "ghostlite",
            TpuGeneration::V7x => "6acc60406",
        };
        let tc_count = i32::try_from(generation.tensor_cores_per_chip()).unwrap_or(1);
        let sc_count = i32::try_from(generation.sparse_cores_per_chip()).unwrap_or(0);
        let mut cores = vec![
            super::descriptor::generated_prost_schemas::tpu::tpu_chip_parts_proto::Core {
                r#type: TpuCoreTypeProto::TpuCoreTypeTensorCore as i32,
                parts: None,
                count: tc_count,
            },
        ];
        if sc_count > 0 {
            cores.push(
                super::descriptor::generated_prost_schemas::tpu::tpu_chip_parts_proto::Core {
                    r#type: TpuCoreTypeProto::TpuCoreTypeSparseCore as i32,
                    parts: None,
                    count: sc_count,
                },
            );
        }
        Self {
            version: generation as i32,
            driver_abi_version: 1,
            cores,
            variant_name: String::from(variant_name),
            ..Self::default()
        }
    }
}

/// Extension factory constructing [`TargetArgumentsProto`] from chip configuration and parts.
pub trait TargetArgumentsProtoExt: Sized {
    /// Constructs a wire-compatible [`TargetArgumentsProto`] from [`TpuChipConfigProto`] and [`TpuChipPartsProto`].
    ///
    /// # Arguments
    ///
    /// * `config` - Chip configuration descriptor.
    /// * `parts` - Physical chip parts descriptor.
    #[must_use]
    fn from_chip_config_and_parts(config: &TpuChipConfigProto, parts: &TpuChipPartsProto) -> Self;
}

impl TargetArgumentsProtoExt for TargetArgumentsProto {
    fn from_chip_config_and_parts(config: &TpuChipConfigProto, parts: &TpuChipPartsProto) -> Self {
        Self {
            tpu_topology_args: Some(super::tpu::TpuTopologyArgsProto {
                version: Some(config.version),
                variant: Some(parts.variant_name.clone()),
                chip_config_name: Some(if config.megacore {
                    String::from("megacore")
                } else {
                    String::from("default")
                }),
                ..super::tpu::TpuTopologyArgsProto::default()
            }),
            configured_properties: Some(TpuConfiguredPropertiesProto {
                degraded_axes: None,
                is_nhop_source_relative: config.megacore,
                routing_strategy: 0,
            }),
            replica_count: Some(1),
            reserved_hbm_usage_bytes: Some(0),
            multislice_topology: None,
        }
    }
}

impl TpuGenerationProtoExt for TargetArgumentsProto {
    fn from_generation(generation: crate::types::TpuGeneration) -> Self {
        let config = TpuChipConfigProto::from_generation(generation);
        let parts = TpuChipPartsProto::from_generation(generation);
        Self::from_chip_config_and_parts(&config, &parts)
    }
}

/// Alias for [`TpuSerializedExecutableProto`].
pub type TpuExecutableSerializedProto = TpuSerializedExecutableProto;

impl TpuSerializedExecutableProto {
    /// Encodes this [`TpuSerializedExecutableProto`] into deterministic Protobuf wire bytes.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_proto()
    }

    /// Decodes a [`TpuSerializedExecutableProto`] from Protobuf wire bytes produced by [`Self::encode`].
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `TpuSerializedExecutableProto` payload.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::LibTpuError::ProtobufDecodeError`] if `bytes` is malformed.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        Self::decode_proto(bytes)
    }
}

impl HostCommandEnvelopeProto {
    /// Encodes this [`HostCommandEnvelopeProto`] into Protobuf wire bytes.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        self.encode_proto()
    }

    /// Decodes a [`HostCommandEnvelopeProto`] from Protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `HostCommandEnvelopeProto` payload.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::LibTpuError::ProtobufDecodeError`] if `bytes` is malformed.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        Self::decode_proto(bytes)
    }
}

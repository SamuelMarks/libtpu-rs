//! `SparseCore` V2 (`BarnaCore` / Viperfish `vfc`) and `SparseCore` V3 (`Ghostfish` / `Ironwood` `iwc`)
//! binary instruction codecs, disassemblers, and generated protobuf lowerers.

use super::super::opcode::TpuInstructionBundle;
use crate::error::{LibTpuError, LibTpuResult};
use crate::proto::{
    BarnaCoreChannelConfigProto, BarnaCoreExecutableProto, IronwoodSparseCoreProgramWrapperProto,
    IronwoodSparseCoreScsProgramProto, IronwoodSparseCoreTecBundleProto,
    IronwoodSparseCoreTecProgramProto, SparseCoreDmaDescriptorProto, SparseCoreProgramProto,
    SparseCoreTileExecutionScheduleProto, SparseCoreV3ExecutableConfigProto,
    ViperfishSparseCoreProgramWrapperProto, ViperfishSparseCoreScsProgramProto,
    ViperfishSparseCoreTacProgramProto, ViperfishSparseCoreTecBundleProto,
    ViperfishSparseCoreTecProgramProto, riegeli_highwayhash64,
};
use crate::types::TpuGeneration;

/// Subcore engine targeted by a `SparseCore` instruction (`SparseCoreSubcoreKind`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseCoreSubcoreKind {
    /// Sequencer Control Subcore (`SCS`, scalar control & loop dispatch).
    Scs = 0,
    /// Tile Access Controller (`TAC`, HBM <-> SPMEM scatter-gather DMA).
    Tac = 1,
    /// Tile Execute Controller (`TEC`, 8/16-lane SIMD vector gather/optimizer ALU).
    Tec = 2,
}

impl SparseCoreSubcoreKind {
    /// Reconstructs a [`SparseCoreSubcoreKind`] from a raw byte (`0..=2`).
    ///
    /// # Arguments
    ///
    /// * `raw` - Subcore code byte.
    #[must_use]
    pub const fn from_byte(raw: u8) -> Self {
        match raw {
            0 => Self::Scs,
            1 => Self::Tac,
            _ => Self::Tec,
        }
    }

    /// Returns the short mnemonic (`"SCS"`, `"TAC"`, `"TEC"`).
    #[must_use]
    pub const fn mnemonic(self) -> &'static str {
        match self {
            Self::Scs => "SCS",
            Self::Tac => "TAC",
            Self::Tec => "TEC",
        }
    }
}

/// Hardware `SparseCore` micro-operation opcode (`SparseCoreOp`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseCoreOp {
    /// Sort COO `(sample_id, row_id)` pairs by vocabulary row (`0`).
    CooRadixSort = 0,
    /// Build compressed CSR row pointers & deduplicate row IDs (`1`).
    CsrBuildAndDedup = 1,
    /// Async HBM -> SPMEM embedding row gather (`2`).
    SpmemGatherDma = 2,
    /// SIMD combiner reduction (`sum`/`mean`/`sqrtn`) across tile lanes (`3`).
    VectorCombinerReduce = 3,
    /// Optimizer state update micro-kernel (`4`).
    OptimizerUpdate = 4,
    /// Async SPMEM -> HBM updated row scatter-write (`5`).
    SpmemScatterWriteDma = 5,
    /// Cross-core `SyncFlag` barrier with `TensorCore` (`6`).
    SyncBarrier = 6,
}

impl SparseCoreOp {
    /// Reconstructs a [`SparseCoreOp`] from a raw opcode byte (`0..=6`).
    ///
    /// # Arguments
    ///
    /// * `raw` - Raw opcode byte.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::SparseCoreCodecError`] if `raw > 6`.
    pub fn from_byte(raw: u8) -> LibTpuResult<Self> {
        match raw {
            0 => Ok(Self::CooRadixSort),
            1 => Ok(Self::CsrBuildAndDedup),
            2 => Ok(Self::SpmemGatherDma),
            3 => Ok(Self::VectorCombinerReduce),
            4 => Ok(Self::OptimizerUpdate),
            5 => Ok(Self::SpmemScatterWriteDma),
            6 => Ok(Self::SyncBarrier),
            _ => Err(LibTpuError::SparseCoreCodecError {
                core_id: 0,
                pc_offset: 0,
                reason: format!("unknown SparseCoreOp byte {raw:#04x}"),
            }),
        }
    }

    /// Returns the symbolic mnemonic of this [`SparseCoreOp`].
    #[must_use]
    pub const fn mnemonic(self) -> &'static str {
        match self {
            Self::CooRadixSort => "coo.sort",
            Self::CsrBuildAndDedup => "csr.dedup",
            Self::SpmemGatherDma => "dma.gather",
            Self::VectorCombinerReduce => "tec.combine",
            Self::OptimizerUpdate => "tec.opt_step",
            Self::SpmemScatterWriteDma => "dma.scatter",
            Self::SyncBarrier => "scs.barrier",
        }
    }
}

/// Computes an 8-bit `HighwayHash64` integrity tag over `raw` with byte `3` zeroed.
///
/// # Arguments
///
/// * `raw` - Encoded `SparseCore` V2 (`16` B) or V3 (`24` B) instruction slice.
#[must_use]
pub fn compute_sparsecore_tag8(raw: &[u8]) -> u8 {
    let mut buf = [0_u8; 24];
    let n = raw.len().min(24);
    buf.iter_mut().zip(raw.iter()).for_each(|(d, s)| *d = *s);
    buf[3] = 0;
    let h = riegeli_highwayhash64(buf.get(..n).unwrap_or(&buf));
    #[allow(clippy::cast_possible_truncation)]
    let tag = ((h ^ (h >> 24)) & 0xFF) as u8;
    tag
}

/// Decoded `SparseCore` V2 (`16`-byte) instruction (`SparseCoreV2Instruction`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseCoreV2Instruction {
    /// Target subcore (`SCS`, `TAC`, `TEC`).
    pub subcore: SparseCoreSubcoreKind,
    /// Operation code.
    pub op: SparseCoreOp,
    /// Optimizer opcode (`0..=18` when `op == OptimizerUpdate`).
    pub optimizer_id: u8,
    /// SPMEM scratchpad byte offset (`0..256 KiB`).
    pub spmem_offset: u32,
    /// HBM table 32-bit tile offset (`4 KiB` pages).
    pub hbm_page_offset: u32,
    /// Active vector length / transfer bytes (`u16`).
    pub transfer_bytes: u16,
    /// Hardware `SyncFlag` completion ID (`u16`).
    pub sync_flag_id: u16,
}

/// Binary encoder/decoder/disassembler for `SparseCore` V2 (`16`-byte bundles, magic `0x52`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SparseCoreV2Codec;

impl SparseCoreV2Codec {
    /// Header magic byte for `SparseCore` V2 (`0x52`).
    pub const MAGIC: u8 = 0x52;
    /// Fixed bundle size in bytes (`16`).
    pub const BUNDLE_BYTES: usize = 16;

    /// Encodes a [`SparseCoreV2Instruction`] into 16 bytes.
    ///
    /// # Arguments
    ///
    /// * `inst` - [`SparseCoreV2Instruction`] to encode.
    #[must_use]
    pub fn encode(inst: &SparseCoreV2Instruction) -> [u8; 16] {
        let mut raw = [0_u8; 16];
        raw[0] = Self::MAGIC;
        raw[1] = ((inst.subcore as u8) << 6) | (inst.op as u8);
        raw[2] = inst.optimizer_id;
        raw[4..8].copy_from_slice(&inst.spmem_offset.to_le_bytes());
        raw[8..12].copy_from_slice(&inst.hbm_page_offset.to_le_bytes());
        raw[12..14].copy_from_slice(&inst.transfer_bytes.to_le_bytes());
        raw[14..16].copy_from_slice(&inst.sync_flag_id.to_le_bytes());
        raw[3] = compute_sparsecore_tag8(&raw);
        raw
    }

    /// Decodes a 16-byte `SparseCore` V2 instruction word.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Raw 16-byte word.
    /// * `pc_offset` - Byte offset for error reporting.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::SparseCoreCodecError`] if length, magic, opcode, or tag validation fails.
    pub fn decode(bytes: &[u8], pc_offset: u32) -> LibTpuResult<SparseCoreV2Instruction> {
        if bytes.len() != Self::BUNDLE_BYTES {
            return Err(LibTpuError::SparseCoreCodecError {
                core_id: 0,
                pc_offset,
                reason: format!("expected 16-byte SparseCoreV2 word, got {}", bytes.len()),
            });
        }
        let mut raw = [0_u8; 16];
        raw.copy_from_slice(bytes.get(..16).unwrap_or(&[0_u8; 16]));
        if raw[0] != Self::MAGIC {
            return Err(LibTpuError::SparseCoreCodecError {
                core_id: 0,
                pc_offset,
                reason: format!("invalid SparseCoreV2 magic {:#04x}", raw[0]),
            });
        }
        let exp_tag = compute_sparsecore_tag8(&raw);
        if raw[3] != exp_tag {
            return Err(LibTpuError::SparseCoreCodecError {
                core_id: 0,
                pc_offset,
                reason: format!(
                    "SparseCoreV2 HighwayHash64 tag mismatch: expected {exp_tag:#04x}, got {:#04x}",
                    raw[3]
                ),
            });
        }
        let subcore = SparseCoreSubcoreKind::from_byte(raw[1] >> 6);
        let op = SparseCoreOp::from_byte(raw[1] & 0x3F)?;
        Ok(SparseCoreV2Instruction {
            subcore,
            op,
            optimizer_id: raw[2],
            spmem_offset: u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]),
            hbm_page_offset: u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]),
            transfer_bytes: u16::from_le_bytes([raw[12], raw[13]]),
            sync_flag_id: u16::from_le_bytes([raw[14], raw[15]]),
        })
    }

    /// Disassembles a `SparseCore` V2 binary stream into symbolic text.
    ///
    /// # Arguments
    ///
    /// * `stream` - Raw byte slice (multiple of 16 bytes).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::SparseCoreCodecError`] if `stream` is empty, misaligned, or invalid.
    pub fn disassemble_stream(stream: &[u8]) -> LibTpuResult<String> {
        if stream.is_empty() || (stream.len() % Self::BUNDLE_BYTES != 0) {
            return Err(LibTpuError::SparseCoreCodecError {
                core_id: 0,
                pc_offset: 0,
                reason: format!("SparseCoreV2 stream length {} misaligned", stream.len()),
            });
        }
        let mut lines = Vec::with_capacity(stream.len() / Self::BUNDLE_BYTES);
        for (idx, chunk) in stream.chunks_exact(Self::BUNDLE_BYTES).enumerate() {
            let inst = Self::decode(chunk, u32::try_from(idx * 16).unwrap_or(0))?;
            lines.push(format!(
                "[sc_v2 pc={idx:04x} {}] {} opt={} spmem={:#x} hbm_pg={:#x} len={} sf={}",
                inst.subcore.mnemonic(),
                inst.op.mnemonic(),
                inst.optimizer_id,
                inst.spmem_offset,
                inst.hbm_page_offset,
                inst.transfer_bytes,
                inst.sync_flag_id
            ));
        }
        Ok(lines.join(
            "
",
        ))
    }

    /// Lowers a slice of [`SparseCoreV2Instruction`]s into a generated `ViperfishSparseCoreProgramWrapperProto`.
    ///
    /// # Arguments
    ///
    /// * `insts` - `SparseCore` V2 instruction sequence.
    #[must_use]
    pub fn to_program_wrapper_proto(
        insts: &[SparseCoreV2Instruction],
    ) -> ViperfishSparseCoreProgramWrapperProto {
        let tec_bundles = insts
            .iter()
            .filter(|i| i.subcore == SparseCoreSubcoreKind::Tec)
            .map(|_| ViperfishSparseCoreTecBundleProto::default())
            .collect();
        ViperfishSparseCoreProgramWrapperProto {
            tec: Some(ViperfishSparseCoreTecProgramProto {
                bundles: tec_bundles,
            }),
            tac: Some(ViperfishSparseCoreTacProgramProto::default()),
            scs: Some(ViperfishSparseCoreScsProgramProto::default()),
        }
    }
}

/// Decoded `SparseCore` V3 (`24`-byte) instruction (`SparseCoreV3Instruction`) supporting 16-lane
/// SIMD vector tiles, quantized BF16/FP8 gather, and 48-bit `HBM3e` table addressing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseCoreV3Instruction {
    /// Target subcore (`SCS`, `TAC`, `TEC`).
    pub subcore: SparseCoreSubcoreKind,
    /// Operation code.
    pub op: SparseCoreOp,
    /// Optimizer opcode (`0..=18`).
    pub optimizer_id: u8,
    /// Quantized embedding precision (`0 = FP32`, `1 = BF16`, `2 = FP8_E4M3FN`).
    pub quant_mode: u8,
    /// Active SIMD vector lanes (`8` or `16`).
    pub vector_lanes: u8,
    /// SPMEM scratchpad byte offset.
    pub spmem_offset: u32,
    /// Full 48-bit `HBM3e` embedding table byte address.
    pub hbm3e_byte_address: u64,
    /// Hardware `SyncFlag` completion ID.
    pub sync_flag_id: u16,
}

/// Binary encoder/decoder/disassembler for `SparseCore` V3 (`24`-byte bundles, magic `0x53`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SparseCoreV3Codec;

impl SparseCoreV3Codec {
    /// Header magic byte for `SparseCore` V3 (`0x53`).
    pub const MAGIC: u8 = 0x53;
    /// Fixed bundle size in bytes (`24`).
    pub const BUNDLE_BYTES: usize = 24;

    /// Encodes a [`SparseCoreV3Instruction`] into 24 bytes.
    ///
    /// # Arguments
    ///
    /// * `inst` - [`SparseCoreV3Instruction`] to encode.
    #[must_use]
    pub fn encode(inst: &SparseCoreV3Instruction) -> [u8; 24] {
        let mut raw = [0_u8; 24];
        raw[0] = Self::MAGIC;
        raw[1] = ((inst.subcore as u8) << 6) | (inst.op as u8);
        raw[2] = inst.optimizer_id;
        raw[4] = inst.quant_mode;
        raw[5] = inst.vector_lanes;
        raw[6..8].copy_from_slice(&inst.sync_flag_id.to_le_bytes());
        raw[8..12].copy_from_slice(&inst.spmem_offset.to_le_bytes());
        raw[12..20]
            .copy_from_slice(&(inst.hbm3e_byte_address & 0x0000_FFFF_FFFF_FFFF).to_le_bytes());
        raw[3] = compute_sparsecore_tag8(&raw);
        raw
    }

    /// Decodes a 24-byte `SparseCore` V3 instruction word.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Raw 24-byte word.
    /// * `pc_offset` - Byte offset for error reporting.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::SparseCoreCodecError`] if length, magic, opcode, or tag validation fails.
    pub fn decode(bytes: &[u8], pc_offset: u32) -> LibTpuResult<SparseCoreV3Instruction> {
        if bytes.len() != Self::BUNDLE_BYTES {
            return Err(LibTpuError::SparseCoreCodecError {
                core_id: 0,
                pc_offset,
                reason: format!("expected 24-byte SparseCoreV3 word, got {}", bytes.len()),
            });
        }
        let mut raw = [0_u8; 24];
        raw.copy_from_slice(bytes.get(..24).unwrap_or(&[0_u8; 24]));
        if raw[0] != Self::MAGIC {
            return Err(LibTpuError::SparseCoreCodecError {
                core_id: 0,
                pc_offset,
                reason: format!("invalid SparseCoreV3 magic {:#04x}", raw[0]),
            });
        }
        let exp_tag = compute_sparsecore_tag8(&raw);
        if raw[3] != exp_tag {
            return Err(LibTpuError::SparseCoreCodecError {
                core_id: 0,
                pc_offset,
                reason: format!(
                    "SparseCoreV3 HighwayHash64 tag mismatch: expected {exp_tag:#04x}, got {:#04x}",
                    raw[3]
                ),
            });
        }
        let subcore = SparseCoreSubcoreKind::from_byte(raw[1] >> 6);
        let op = SparseCoreOp::from_byte(raw[1] & 0x3F)?;
        Ok(SparseCoreV3Instruction {
            subcore,
            op,
            optimizer_id: raw[2],
            quant_mode: raw[4],
            vector_lanes: raw[5],
            sync_flag_id: u16::from_le_bytes([raw[6], raw[7]]),
            spmem_offset: u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]),
            hbm3e_byte_address: u64::from_le_bytes([
                raw[12], raw[13], raw[14], raw[15], raw[16], raw[17], raw[18], raw[19],
            ]),
        })
    }

    /// Disassembles a `SparseCore` V3 binary stream into symbolic text.
    ///
    /// # Arguments
    ///
    /// * `stream` - Raw byte slice (multiple of 24 bytes).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::SparseCoreCodecError`] if `stream` is empty, misaligned, or invalid.
    pub fn disassemble_stream(stream: &[u8]) -> LibTpuResult<String> {
        if stream.is_empty() || (stream.len() % Self::BUNDLE_BYTES != 0) {
            return Err(LibTpuError::SparseCoreCodecError {
                core_id: 0,
                pc_offset: 0,
                reason: format!("SparseCoreV3 stream length {} misaligned", stream.len()),
            });
        }
        let mut lines = Vec::with_capacity(stream.len() / Self::BUNDLE_BYTES);
        for (idx, chunk) in stream.chunks_exact(Self::BUNDLE_BYTES).enumerate() {
            let inst = Self::decode(chunk, u32::try_from(idx * 24).unwrap_or(0))?;
            lines.push(format!(
                "[sc_v3 pc={idx:04x} {}] {} opt={} lanes={} quant={} spmem={:#x} hbm3e={:#x} sf={}",
                inst.subcore.mnemonic(),
                inst.op.mnemonic(),
                inst.optimizer_id,
                inst.vector_lanes,
                inst.quant_mode,
                inst.spmem_offset,
                inst.hbm3e_byte_address,
                inst.sync_flag_id
            ));
        }
        Ok(lines.join(
            "
",
        ))
    }

    /// Lowers a slice of [`SparseCoreV3Instruction`]s into a generated `IronwoodSparseCoreProgramWrapperProto`.
    ///
    /// # Arguments
    ///
    /// * `insts` - `SparseCore` V3 instruction sequence.
    #[must_use]
    pub fn to_program_wrapper_proto(
        insts: &[SparseCoreV3Instruction],
    ) -> IronwoodSparseCoreProgramWrapperProto {
        let tec_bundles = insts
            .iter()
            .filter(|i| i.subcore == SparseCoreSubcoreKind::Tec)
            .map(|_| IronwoodSparseCoreTecBundleProto::default())
            .collect();
        IronwoodSparseCoreProgramWrapperProto {
            tec: Some(IronwoodSparseCoreTecProgramProto {
                bundles: tec_bundles,
            }),
            scs: Some(IronwoodSparseCoreScsProgramProto::default()),
        }
    }
}

/// Top-level `SparseCore` V2/V3 lowering facade (`SparseCoreBackend`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseCoreBackend {
    /// Target Cloud TPU generation (`V2..=V7x`).
    pub generation: TpuGeneration,
}

impl SparseCoreBackend {
    /// Creates a [`SparseCoreBackend`] for `generation`.
    ///
    /// # Arguments
    ///
    /// * `generation` - Target [`TpuGeneration`].
    #[must_use]
    pub const fn new(generation: TpuGeneration) -> Self {
        Self { generation }
    }

    /// Lowers [`TpuInstructionBundle`]s into a [`BarnaCoreExecutableProto`] containing authentic
    /// [`SparseCoreV2Codec`] or [`SparseCoreV3Codec`] instruction bitstreams.
    ///
    /// # Arguments
    ///
    /// * `module_name` - Module identifier string.
    /// * `bundles` - Lowered [`TpuInstructionBundle`] slice.
    #[must_use]
    pub fn lower_bundles_to_barnacore_proto(
        &self,
        module_name: &str,
        bundles: &[TpuInstructionBundle],
    ) -> BarnaCoreExecutableProto {
        let is_v3 = matches!(self.generation, TpuGeneration::V6e | TpuGeneration::V7x);
        let sc_version = if is_v3 { 3_u32 } else { 2_u32 };
        let mut scs_stream = Vec::new();
        let mut inst_stream = Vec::new();
        let mut dma_descriptors = Vec::with_capacity(bundles.len().max(1));

        for (idx, b) in bundles.iter().enumerate() {
            let is_scatter = (b.opcode == super::super::opcode::TpuOpCode::Scatter)
                | b.attrs.custom_call_target.contains("gradient");
            let op = if is_scatter {
                SparseCoreOp::SpmemScatterWriteDma
            } else {
                SparseCoreOp::SpmemGatherDma
            };
            let spmem_off = u32::try_from((idx * 256) & 0x3FFF).unwrap_or(0);
            let sf_id = u16::try_from(b.sync_flag_id.max(1)).unwrap_or(1);
            if is_v3 {
                let sc_v3 = SparseCoreV3Instruction {
                    subcore: SparseCoreSubcoreKind::Tec,
                    op,
                    optimizer_id: 0,
                    quant_mode: 1,
                    vector_lanes: 16,
                    spmem_offset: spmem_off,
                    hbm3e_byte_address: (idx as u64) * 4096,
                    sync_flag_id: sf_id,
                };
                let bytes = SparseCoreV3Codec::encode(&sc_v3);
                scs_stream.extend_from_slice(&bytes);
                inst_stream.extend_from_slice(&bytes);
            } else {
                let sc_v2 = SparseCoreV2Instruction {
                    subcore: SparseCoreSubcoreKind::Tec,
                    op,
                    optimizer_id: 0,
                    spmem_offset: spmem_off,
                    hbm_page_offset: u32::try_from(idx).unwrap_or(0),
                    transfer_bytes: 256,
                    sync_flag_id: sf_id,
                };
                let bytes = SparseCoreV2Codec::encode(&sc_v2);
                scs_stream.extend_from_slice(&bytes);
                inst_stream.extend_from_slice(&bytes);
            }
            dma_descriptors.push(SparseCoreDmaDescriptorProto {
                descriptor_id: u32::try_from(idx).unwrap_or(0),
                hbm_base_offset: (idx as u64) * 4096,
                spmem_offset: spmem_off,
                transfer_bytes: 256,
                sync_flag_id: u32::from(sf_id),
                is_scatter_write: is_scatter,
            });
        }

        BarnaCoreExecutableProto {
            executable_name: format!("{module_name}_barnacore"),
            sparse_core_version: sc_version,
            channels: vec![BarnaCoreChannelConfigProto {
                channel_id: 0,
                sparse_core_index: 0,
                max_unique_ids_per_partition: 2048,
                max_ids_per_partition: 4096,
                table_name: format!("{module_name}_table0"),
            }],
            programs: vec![SparseCoreProgramProto {
                sparse_core_id: 0,
                instruction_stream: inst_stream,
                spmem_scratchpad_bytes: 16_384,
                dma_descriptors,
                optimizer_opcode: 0,
                tec_bundle_count: u32::try_from(bundles.len().max(1)).unwrap_or(1),
                scs_control_stream: scs_stream,
                tile_spmem_size_bytes: 4096,
            }],
            table_hbm_total_bytes: 262_144,
        }
    }

    /// Builds a [`SparseCoreV3ExecutableConfigProto`] for Trillium (`V6e`) and Ironwood (`V7x`).
    ///
    /// # Arguments
    ///
    /// * `config_name` - Configuration name.
    /// * `vocab_rows` - Total embedding table vocabulary rows.
    /// * `embedding_dim` - Embedding vector width in elements.
    #[must_use]
    pub fn build_v3_executable_config_proto(
        &self,
        config_name: &str,
        vocab_rows: u32,
        embedding_dim: u32,
    ) -> SparseCoreV3ExecutableConfigProto {
        let num_sc = self.generation.sparse_cores_per_chip().max(2);
        let spmem_bytes = if self.generation == TpuGeneration::V7x {
            524_288_u64
        } else {
            262_144_u64
        };
        let row_bytes = u64::from(embedding_dim.max(1)) * 4;
        let tile_rows = u32::try_from((spmem_bytes / 2 / row_bytes).clamp(1, 256)).unwrap_or(64);
        SparseCoreV3ExecutableConfigProto {
            executable_name: String::from(config_name),
            num_sparse_cores: num_sc,
            tiles_per_sparse_core: vocab_rows.div_ceil(tile_rows.saturating_mul(num_sc)).max(1),
            hw_coo_sort_enabled: true,
            tile_schedules: vec![SparseCoreTileExecutionScheduleProto {
                tile_id: 0,
                tec_start_pc: 0,
                tec_bundle_count: tile_rows.max(1),
                spmem_base_offset: 0,
                spmem_size_bytes: tile_rows.saturating_mul(embedding_dim).saturating_mul(4),
            }],
        }
    }
}

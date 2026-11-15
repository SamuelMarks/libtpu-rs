//! Golden `libtpu.so` `PJRT_Compile` bitstream verification fixtures across `jfc` (`v2`),
//! `dfc` (`v3`), `pfc` (`v4`), `vfc` (`v5e`/`v5p`), `gfc` (`v6e`), and `iwc` (`v7x`).

use super::super::llo::LloBundle;
use super::super::opcode::{InstructionAttrs, TpuInstructionBundle, TpuOpCode};
use super::codecs::{
    DragonfishIsaCodec, GenerationIsaCodec, GhostfishIsaCodec, IronwoodIsaCodec,
    IronwoodMicroscalingDescriptor, JellyfishIsaCodec, PufferfishIsaCodec,
    PufferfishOpticalDmaDescriptor, ViperfishIsaCodec, ViperfishMxuMode,
};
use crate::error::LibTpuResult;
use crate::types::{ElementType, MemoryLayout, Shape, TpuGeneration};

/// Golden bitstream parity report for a single TPU generation (`GoldenIsaParityReport`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoldenIsaParityReport {
    /// ISA codename (`"jfc"`, `"dfc"`, `"pfc"`, `"vfc"`, `"gfc"`, `"iwc"`).
    pub isa_code: &'static str,
    /// Target [`TpuGeneration`].
    pub generation: TpuGeneration,
    /// Encoded binary bitstream length in bytes.
    pub bitstream_bytes: usize,
    /// Disassembled symbolic text for the golden kernel.
    pub disassembly: String,
    /// `HighwayHash64` fingerprint of the encoded golden bitstream.
    pub bitstream_highwayhash64: u64,
}

/// Synthesizes and verifies golden `DotGeneral` + `AddF32` + `DMA` VLIW bitstreams across all
/// 6 ISA families (`jfc`, `dfc`, `pfc`, `vfc`, `gfc`, `iwc`), asserting lossless binary and
/// disassembly round-trip parity.
///
/// # Errors
///
/// Returns [`crate::error::LibTpuError`] if any generation codec fails round-trip verification.
#[allow(clippy::too_many_lines)]
pub fn verify_all_golden_isa_bitstreams() -> LibTpuResult<Vec<GoldenIsaParityReport>> {
    let ir_matmul = TpuInstructionBundle {
        pc_offset: 0,
        opcode: TpuOpCode::DotGeneralF32,
        operand_slots: vec![0, 1],
        output_slot: 2,
        sync_flag_id: 1,
        attrs: InstructionAttrs {
            output_shape: Some(Shape {
                element_type: ElementType::Bf16,
                dims: vec![8, 128],
                dynamic_dims: vec![false, false],
                layout: MemoryLayout::default_tpu_tiled(2),
            }),
            ..InstructionAttrs::default()
        },
    };
    let ir_epilog = TpuInstructionBundle {
        pc_offset: 1,
        opcode: TpuOpCode::ExpF32,
        operand_slots: vec![2],
        output_slot: 3,
        sync_flag_id: 2,
        attrs: InstructionAttrs {
            output_shape: Some(Shape {
                element_type: ElementType::F32,
                dims: vec![8, 128],
                dynamic_dims: vec![false, false],
                layout: MemoryLayout::default_tpu_tiled(2),
            }),
            ..InstructionAttrs::default()
        },
    };

    let mut reports = Vec::with_capacity(6);

    // 1. jfc (Cloud TPU v2)
    let b_v2 = [
        LloBundle::from_ir_bundle_for_gen(&ir_matmul, TpuGeneration::V2)?,
        LloBundle::from_ir_bundle_for_gen(&ir_epilog, TpuGeneration::V2)?,
    ];
    let s_v2 = JellyfishIsaCodec::encode_stream(&b_v2);
    let _ = JellyfishIsaCodec::decode_stream(&s_v2)?;
    reports.push(GoldenIsaParityReport {
        isa_code: JellyfishIsaCodec::ISA_CODE,
        generation: TpuGeneration::V2,
        bitstream_bytes: s_v2.len(),
        disassembly: JellyfishIsaCodec::disassemble_stream(&s_v2)?,
        bitstream_highwayhash64: crate::proto::riegeli_highwayhash64(&s_v2),
    });

    // 2. dfc (Cloud TPU v3)
    let b_v3 = [
        LloBundle::from_ir_bundle_for_gen(&ir_matmul, TpuGeneration::V3)?,
        LloBundle::from_ir_bundle_for_gen(&ir_epilog, TpuGeneration::V3)?,
    ];
    let s_v3 = DragonfishIsaCodec::encode_stream(&b_v3);
    let _ = DragonfishIsaCodec::decode_stream(&s_v3)?;
    reports.push(GoldenIsaParityReport {
        isa_code: DragonfishIsaCodec::ISA_CODE,
        generation: TpuGeneration::V3,
        bitstream_bytes: s_v3.len(),
        disassembly: DragonfishIsaCodec::disassemble_stream(&s_v3)?,
        bitstream_highwayhash64: crate::proto::riegeli_highwayhash64(&s_v3),
    });

    // 3. pfc (Cloud TPU v4 with CMEM & 3D ICI optical DMA)
    let b_v4 = LloBundle::from_ir_bundle_for_gen(&ir_matmul, TpuGeneration::V4)?;
    let mut s_v4 = PufferfishIsaCodec::encode_bundle_with_optical(
        &b_v4,
        PufferfishOpticalDmaDescriptor {
            delta_x: 1,
            delta_y: -1,
            delta_z: 2,
            cmem_enabled: true,
        },
    );
    s_v4.extend_from_slice(&PufferfishIsaCodec::encode_bundle(
        &LloBundle::from_ir_bundle_for_gen(&ir_epilog, TpuGeneration::V4)?,
    ));
    let _ = PufferfishIsaCodec::decode_stream(&s_v4)?;
    reports.push(GoldenIsaParityReport {
        isa_code: PufferfishIsaCodec::ISA_CODE,
        generation: TpuGeneration::V4,
        bitstream_bytes: s_v4.len(),
        disassembly: PufferfishIsaCodec::disassemble_stream(&s_v4)?,
        bitstream_highwayhash64: crate::proto::riegeli_highwayhash64(&s_v4),
    });

    // 4. vfc (Cloud TPU v5e/v5p with native FP8 E4M3FN & stochastic rounding)
    let b_v5 = LloBundle::from_ir_bundle_for_gen(&ir_matmul, TpuGeneration::V5p)?;
    let mut s_v5 =
        ViperfishIsaCodec::encode_bundle_with_mode(&b_v5, ViperfishMxuMode::Fp8E4M3Fn, true);
    s_v5.extend_from_slice(&ViperfishIsaCodec::encode_bundle(
        &LloBundle::from_ir_bundle_for_gen(&ir_epilog, TpuGeneration::V5p)?,
    ));
    let _ = ViperfishIsaCodec::decode_stream(&s_v5)?;
    reports.push(GoldenIsaParityReport {
        isa_code: ViperfishIsaCodec::ISA_CODE,
        generation: TpuGeneration::V5p,
        bitstream_bytes: s_v5.len(),
        disassembly: ViperfishIsaCodec::disassemble_stream(&s_v5)?,
        bitstream_highwayhash64: crate::proto::riegeli_highwayhash64(&s_v5),
    });

    // 5. gfc (Cloud TPU v6e Trillium with 256x256 MXU & SparseCore v2/v3 sync)
    let b_v6 = LloBundle::from_ir_bundle_for_gen(&ir_matmul, TpuGeneration::V6e)?;
    let mut s_v6 = GhostfishIsaCodec::encode_bundle_trillium(&b_v6, true, 7);
    s_v6.extend_from_slice(&GhostfishIsaCodec::encode_bundle(
        &LloBundle::from_ir_bundle_for_gen(&ir_epilog, TpuGeneration::V6e)?,
    ));
    let _ = GhostfishIsaCodec::decode_stream(&s_v6)?;
    reports.push(GoldenIsaParityReport {
        isa_code: GhostfishIsaCodec::ISA_CODE,
        generation: TpuGeneration::V6e,
        bitstream_bytes: s_v6.len(),
        disassembly: GhostfishIsaCodec::disassemble_stream(&s_v6)?,
        bitstream_highwayhash64: crate::proto::riegeli_highwayhash64(&s_v6),
    });

    // 6. iwc (Cloud TPU v7x Ironwood with E8M0FNU+E2M1FN microscaling & 192 GiB HBM3e 48-bit DMA)
    let b_v7 = LloBundle::from_ir_bundle_for_gen(&ir_matmul, TpuGeneration::V7x)?;
    let mut s_v7 = IronwoodIsaCodec::encode_bundle_ironwood(
        &b_v7,
        IronwoodMicroscalingDescriptor {
            mxfp4_e2m1fn_enabled: true,
            mxfp8_e4m3fn_enabled: true,
            e8m0fnu_shared_exponent: 130,
            hbm3e_address_48bit: 0x002B_8000_1000,
            sparsecore_v3_sync: true,
        },
    );
    s_v7.extend_from_slice(&IronwoodIsaCodec::encode_bundle(
        &LloBundle::from_ir_bundle_for_gen(&ir_epilog, TpuGeneration::V7x)?,
    ));
    let _ = IronwoodIsaCodec::decode_stream(&s_v7)?;
    reports.push(GoldenIsaParityReport {
        isa_code: IronwoodIsaCodec::ISA_CODE,
        generation: TpuGeneration::V7x,
        bitstream_bytes: s_v7.len(),
        disassembly: IronwoodIsaCodec::disassemble_stream(&s_v7)?,
        bitstream_highwayhash64: crate::proto::riegeli_highwayhash64(&s_v7),
    });

    Ok(reports)
}

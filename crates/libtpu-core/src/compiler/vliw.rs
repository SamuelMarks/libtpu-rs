//! Functional-unit slot descriptors, scoreboard, polyhedral tiling, modulo pipeline helpers,
//! and LLO assembly formatting/parsing over [`LloBundle`] and generation-exact [`super::isa`] codecs.
//!
//! All synthetic `TensorCoreVliwBundle`, `VliwBundleHeader` (`pc_offset: u32`), and `0x5A` CRC-8
//! placeholders have been eliminated in favor of positional [`LloBundle`]s and `HighwayHash64`-tagged
//! [`super::isa::GenerationIsaCodec`] bitstreams.

use bytemuck::{Pod, Zeroable};

use super::isa::{
    GenerationIsaCodec, GhostfishIsaCodec, JellyfishIsaCodec, PufferfishIsaCodec,
    ViperfishIsaCodec, decode_bundle_for_generation, encode_bundle_for_generation,
    encode_canonical_64b_bundle,
};
use super::llo::LloBundle;
use super::opcode::{TpuInstructionBundle, TpuOpCode};
use crate::error::{LibTpuError, LibTpuResult};
use crate::types::TpuGeneration;

/// Compact 16-byte VLIW instruction word summary (`VliwInstructionWord`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Zeroable, Pod)]
pub struct VliwInstructionWord {
    /// Hardware [`TpuOpCode`] discriminant.
    pub opcode: u16,
    /// Functional unit execution bitmask (`S0`, `V0`, `M0`, `XLU`, `D0`).
    pub unit_flags: u16,
    /// Destination vector or scalar register index.
    pub dst_reg: u32,
    /// First source operand register index.
    pub src0_reg: u32,
    /// Second source operand register or immediate slot.
    pub src1_reg: u32,
}

const _: () = assert!(size_of::<VliwInstructionWord>() == 16);

impl VliwInstructionWord {
    /// Serializes this 16-byte [`VliwInstructionWord`] into little-endian bytes.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; 16] {
        let mut out = [0_u8; 16];
        out.copy_from_slice(bytemuck::bytes_of(self));
        out
    }

    /// Reconstructs a [`VliwInstructionWord`] from 16 little-endian bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Raw 16-byte word.
    #[must_use]
    pub fn from_bytes(bytes: &[u8; 16]) -> Self {
        bytemuck::pod_read_unaligned(bytes)
    }
}

/// Scalar ALU & Branch Unit slot (`S0` / `S1`, 8 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Zeroable, Pod)]
pub struct ScalarAluSlot {
    /// Scalar ALU or control-flow opcode (`0 = NOP`, `1 = S_ADD`, `2 = S_MOV_IMM`, `3 = S_BRANCH_REL`, `4 = S_BRANCH_COND`, `5 = S_SYNC_WAIT`).
    pub opcode: u8,
    /// Destination scalar register (`$s0..$s63`).
    pub dst_sreg: u8,
    /// First source scalar register (`$s0..$s63`).
    pub src0_sreg: u8,
    /// Second source scalar register (`$s0..$s63`).
    pub src1_sreg: u8,
    /// 32-bit immediate constant or branch target bundle index.
    pub imm32: u32,
}

/// 2D Vector Processing Unit (`VPU`) slot (`V0` / `V1`, 20 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Zeroable, Pod)]
pub struct VectorAluSlot {
    /// Vector ALU [`TpuOpCode`] discriminant.
    pub opcode: u16,
    /// Encoded [`crate::types::ElementType`] discriminant.
    pub element_type: u8,
    /// 8-bit sublane execution mask (`0xFF` = all 8 sublanes active).
    pub lane_mask: u8,
    /// Destination `[8, 128]` vector register (`$v0..$v63`).
    pub dst_vreg: u32,
    /// First source vector register.
    pub src0_vreg: u32,
    /// Second source vector register.
    pub src1_vreg: u32,
    /// Third source vector register (ternary `Select` / `Clamp` / `FMA`).
    pub src2_vreg: u32,
}

/// Systolic Matrix Multiply Unit (`MXU`) slot (`M0..M3`, 12 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Zeroable, Pod)]
pub struct MatrixMxuSlot {
    /// Matrix unit opcode (`0 = NOP`, non-zero = systolic contraction).
    pub opcode: u8,
    /// Precision mode (`0 = BF16`, `1 = FP32`, `2 = INT8`, `3 = FP8_E4M3FN`, `4 = FP8_E5M2`).
    pub precision_mode: u8,
    /// Systolic array control flags (`transpose_lhs`, `clear_acc`).
    pub flags: u16,
    /// Accumulator register (`$gmr0..$gmr7`).
    pub acc_vreg: u16,
    /// Left-hand activation vector register.
    pub lhs_vreg: u16,
    /// Right-hand weight vector register.
    pub rhs_vreg: u16,
    /// Contracting tile count (`k_tiles`).
    pub k_tiles: u16,
}

/// Extended Unary (`EUP`) & Cross-Lane (`XLU`) slot (`X0` / `X1`, 8 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Zeroable, Pod)]
pub struct ExtendedUnarySlot {
    /// EUP/XLU opcode (`0 = NOP`).
    pub opcode: u8,
    /// Cross-lane permutation / transpose mode.
    pub perm_mode: u8,
    /// Destination vector register.
    pub dst_vreg: u16,
    /// Source vector register.
    pub src_vreg: u16,
    /// Auxiliary 16-bit permutation / rotation / `HBM3e` upper-address immediate.
    pub imm16: u16,
}

/// Asynchronous DMA & ICI Collective Engine slot (`D0`, 8 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Zeroable, Pod)]
pub struct AsyncDmaSlot {
    /// DMA opcode (`0 = NOP`, `1 = HBM_TO_VMEM`, `2 = VMEM_TO_HBM`, `>2 = ICI_COLLECTIVE`).
    pub dma_opcode: u8,
    /// Source memory space (`0 = HBM`, `1 = VMEM`, `2 = SMEM`, `3 = CMEM`).
    pub src_space: u8,
    /// Destination memory space (`0 = HBM`, `1 = VMEM`, `2 = SMEM`, `3 = CMEM`).
    pub dst_space: u8,
    /// ICI collective reduction opcode (`0` for local DMA).
    pub collective_op: u8,
    /// Hardware `SyncFlag` completion counter index.
    pub sync_flag_id: u32,
}

/// Fixed-size 32-byte (`256`-bit) `jfc`/`dfc` (`V2`/`V3`) hardware bundle word (`JfcV2V3BundleWord`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JfcV2V3BundleWord {
    /// Raw 32-byte (`256`-bit) `jfc` VLIW bundle bitstream.
    pub raw_bytes: [u8; 32],
}

const _: () = assert!(size_of::<JfcV2V3BundleWord>() == 32);

impl JfcV2V3BundleWord {
    /// Packs an [`LloBundle`] into a 32-byte `jfc` [`JfcV2V3BundleWord`].
    ///
    /// # Arguments
    ///
    /// * `bundle` - Source [`LloBundle`].
    #[must_use]
    pub fn from_bundle(bundle: &LloBundle) -> Self {
        let vec = JellyfishIsaCodec::encode_bundle(bundle);
        let mut raw_bytes = [0_u8; 32];
        raw_bytes.copy_from_slice(vec.get(..32).unwrap_or(&[0_u8; 32]));
        Self { raw_bytes }
    }

    /// Unpacks and verifies this 32-byte [`JfcV2V3BundleWord`] into an [`LloBundle`].
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::VliwIsaCodecError`] if `HighwayHash64` tag verification fails.
    pub fn to_bundle(&self) -> LibTpuResult<LloBundle> {
        JellyfishIsaCodec::decode_bundle(&self.raw_bytes, 0)
    }
}

/// Fixed-size 40-byte (`320`-bit) `pfc` (`V4`/`V4Lite`) hardware bundle word (`PfcV4BundleWord`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PfcV4BundleWord {
    /// Raw 40-byte (`320`-bit) `pfc` VLIW bundle bitstream.
    pub raw_bytes: [u8; 40],
}

const _: () = assert!(size_of::<PfcV4BundleWord>() == 40);

impl PfcV4BundleWord {
    /// Packs an [`LloBundle`] into a 40-byte `pfc` [`PfcV4BundleWord`].
    ///
    /// # Arguments
    ///
    /// * `bundle` - Source [`LloBundle`].
    #[must_use]
    pub fn from_bundle(bundle: &LloBundle) -> Self {
        let vec = PufferfishIsaCodec::encode_bundle(bundle);
        let mut raw_bytes = [0_u8; 40];
        raw_bytes.copy_from_slice(vec.get(..40).unwrap_or(&[0_u8; 40]));
        Self { raw_bytes }
    }

    /// Unpacks and verifies this 40-byte [`PfcV4BundleWord`] into an [`LloBundle`].
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::VliwIsaCodecError`] if `HighwayHash64` tag verification fails.
    pub fn to_bundle(&self) -> LibTpuResult<LloBundle> {
        PufferfishIsaCodec::decode_bundle(&self.raw_bytes, 0)
    }
}

/// Fixed-size 52-byte (`416`-bit) `vfc` (`V5e`/`V5p`) hardware bundle word (`VfcV5BundleWord`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VfcV5BundleWord {
    /// Raw 52-byte (`416`-bit) `vfc` VLIW bundle bitstream.
    pub raw_bytes: [u8; 52],
}

const _: () = assert!(size_of::<VfcV5BundleWord>() == 52);

impl VfcV5BundleWord {
    /// Packs an [`LloBundle`] into a 52-byte `vfc` [`VfcV5BundleWord`].
    ///
    /// # Arguments
    ///
    /// * `bundle` - Source [`LloBundle`].
    #[must_use]
    pub fn from_bundle(bundle: &LloBundle) -> Self {
        let vec = ViperfishIsaCodec::encode_bundle(bundle);
        let mut raw_bytes = [0_u8; 52];
        raw_bytes.copy_from_slice(vec.get(..52).unwrap_or(&[0_u8; 52]));
        Self { raw_bytes }
    }

    /// Unpacks and verifies this 52-byte [`VfcV5BundleWord`] into an [`LloBundle`].
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::VliwIsaCodecError`] if `HighwayHash64` tag verification fails.
    pub fn to_bundle(&self) -> LibTpuResult<LloBundle> {
        ViperfishIsaCodec::decode_bundle(&self.raw_bytes, 0)
    }
}

/// Fixed-size 64-byte (`512`-bit) `gfc`/`iwc` (`V6e`/`V7x`) hardware bundle word (`VfkV6V7BundleWord`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VfkV6V7BundleWord {
    /// Raw 64-byte (`512`-bit) `gfc`/`iwc` VLIW bundle bitstream.
    pub raw_bytes: [u8; 64],
}

const _: () = assert!(size_of::<VfkV6V7BundleWord>() == 64);

impl VfkV6V7BundleWord {
    /// Packs an [`LloBundle`] into a 64-byte `gfc`/`iwc` [`VfkV6V7BundleWord`].
    ///
    /// # Arguments
    ///
    /// * `bundle` - Source [`LloBundle`].
    #[must_use]
    pub fn from_bundle(bundle: &LloBundle) -> Self {
        Self {
            raw_bytes: encode_canonical_64b_bundle(bundle),
        }
    }

    /// Unpacks and verifies this 64-byte [`VfkV6V7BundleWord`] into an [`LloBundle`].
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::VliwIsaCodecError`] if `HighwayHash64` tag verification fails.
    pub fn to_bundle(&self) -> LibTpuResult<LloBundle> {
        GhostfishIsaCodec::decode_bundle(&self.raw_bytes, 0)
    }
}

/// Extracts a bitfield slice from `bitstream` for `slot_kind` using `IsaBitfieldSpecTable`.
///
/// # Arguments
///
/// * `generation` - Target [`TpuGeneration`].
/// * `bitstream` - Generation-encoded bundle bytes.
/// * `slot_kind` - Target [`crate::elf::IsaSlotKind`].
#[must_use]
pub fn extract_slot_bitfield(
    generation: TpuGeneration,
    bitstream: &[u8],
    slot_kind: crate::elf::IsaSlotKind,
) -> Option<u64> {
    let table = crate::elf::IsaBitfieldSpecTable::for_generation(generation);
    let mut acc = 0_u64;
    for spec in table.slots.iter().filter(|s| s.slot == slot_kind) {
        let width = spec.bit_width.min(64);
        for bit_idx in 0..width {
            let abs_bit = usize::from(spec.bit_offset.saturating_add(bit_idx));
            let byte_idx = abs_bit >> 3;
            let bit_in_byte = abs_bit & 7;
            let byte_val = *bitstream.get(byte_idx)?;
            if ((byte_val >> bit_in_byte) & 1) != 0 {
                acc |= 1_u64 << bit_idx;
            }
        }
    }
    Some(acc)
}

/// Packs `field_value` into `bitstream` for `slot_kind` using `IsaBitfieldSpecTable`.
///
/// # Arguments
///
/// * `generation` - Target [`TpuGeneration`].
/// * `bitstream` - Mutable generation-encoded bundle bytes.
/// * `slot_kind` - Target [`crate::elf::IsaSlotKind`].
/// * `field_value` - 64-bit value to pack.
pub fn pack_slot_bitfield(
    generation: TpuGeneration,
    bitstream: &mut [u8],
    slot_kind: crate::elf::IsaSlotKind,
    field_value: u64,
) {
    let table = crate::elf::IsaBitfieldSpecTable::for_generation(generation);
    for spec in table.slots.iter().filter(|s| s.slot == slot_kind) {
        let width = spec.bit_width.min(64);
        for bit_idx in 0..width {
            let abs_bit = usize::from(spec.bit_offset.saturating_add(bit_idx));
            let byte_idx = abs_bit >> 3;
            let bit_in_byte = abs_bit & 7;
            bitstream.get_mut(byte_idx).into_iter().for_each(|b| {
                if ((field_value >> bit_idx) & 1) != 0 {
                    *b |= 1_u8 << bit_in_byte;
                } else {
                    *b &= !(1_u8 << bit_in_byte);
                }
            });
        }
    }
}

/// Disassembles a single [`LloBundle`] at positional `pc` into canonical LLO text.
///
/// # Arguments
///
/// * `bundle` - Architectural [`LloBundle`].
/// * `pc` - Positional bundle index in `IMEM`.
#[must_use]
pub fn disassemble_llo_bundle(bundle: &LloBundle, pc: usize) -> String {
    let stalls = bundle.stall_cycles;
    let gen_ord = match bundle.generation {
        TpuGeneration::V2 => 2_u8,
        TpuGeneration::V3 => 3_u8,
        TpuGeneration::V4 | TpuGeneration::V4Lite => 4_u8,
        TpuGeneration::V5e | TpuGeneration::V5p => 5_u8,
        TpuGeneration::V6e => 6_u8,
        TpuGeneration::V7x => 7_u8,
    };
    let s0 = bundle.scalar_slot();
    let v0 = bundle.vector_slot();
    let m0 = bundle.matrix_slot();
    let x0 = bundle.xunary_slot();
    let v_op = u8::try_from(v0.opcode)
        .ok()
        .and_then(|b| TpuOpCode::from_byte(b).ok())
        .map_or_else(|| String::from("AddF32"), |op| format!("{op:?}"));
    let s_str = format!(
        "S0(op={}, $s{} <- $s{}, $s{}, imm={})",
        s0.opcode, s0.dst_sreg, s0.src0_sreg, s0.src1_sreg, s0.imm32
    );
    let v_str = format!(
        "V0({v_op}, $v{} <- $v{}, $v{}, $v{})",
        v0.dst_vreg, v0.src0_vreg, v0.src1_vreg, v0.src2_vreg
    );
    let m_str = if m0.opcode != 0 {
        format!(
            " | M0(acc=$v{}, lhs=$v{}, rhs=$v{}, k={})",
            m0.acc_vreg, m0.lhs_vreg, m0.rhs_vreg, m0.k_tiles
        )
    } else {
        String::new()
    };
    let x_str = if x0.opcode != 0 {
        format!(
            " | XLU(op={}, $v{} <- $v{})",
            x0.opcode, x0.dst_vreg, x0.src_vreg
        )
    } else {
        String::new()
    };
    let d_str = if bundle.dma_slot.dma_opcode != 0 {
        format!(
            " | D0(op={}, flag={})",
            bundle.dma_slot.dma_opcode, bundle.dma_slot.sync_flag_id
        )
    } else {
        String::new()
    };
    format!(
        "[pc={pc:04} gen=v{gen_ord} stall={stalls}] {{ {s_str} | {v_str}{m_str}{x_str}{d_str} }}"
    )
}

/// Assembles a single canonical LLO bundle line (`"[pc=0000 gen=v5 stall=0] { S0(...) | V0(...) }"`) into an [`LloBundle`].
///
/// # Arguments
///
/// * `mnemonic_line` - Single-line `DeepSea` LLO assembly string.
///
/// # Errors
///
/// Returns [`LibTpuError::CompilationFailed`] if syntax or opcode mnemonic parsing fails.
#[allow(clippy::too_many_lines)]
pub fn assemble_llo_bundle(mnemonic_line: &str) -> LibTpuResult<LloBundle> {
    let trimmed = mnemonic_line.trim();
    let Some(rest_pc) = trimmed.strip_prefix("[pc=") else {
        return Err(LibTpuError::CompilationFailed {
            reason: format!("Missing '[pc=' header in VLIW assembly line: '{trimmed}'"),
        });
    };
    let extract_between = |haystack: &str, prefix: &str, suffix: &str| -> Option<String> {
        let after = haystack.split_once(prefix)?.1;
        Some(String::from(after.split_once(suffix)?.0.trim()))
    };
    let default_imm = rest_pc
        .split_once(' ')
        .and_then(|(s, _)| s.parse::<u32>().ok())
        .unwrap_or(0);
    let gen_ord = extract_between(trimmed, "gen=v", " ")
        .and_then(|s| s.parse::<u8>().ok())
        .unwrap_or(5);
    let generation = match gen_ord {
        2 => TpuGeneration::V2,
        3 => TpuGeneration::V3,
        4 => TpuGeneration::V4,
        6 => TpuGeneration::V6e,
        7 => TpuGeneration::V7x,
        _ => TpuGeneration::V5e,
    };
    let stall_cycles = extract_between(trimmed, "stall=", "]")
        .and_then(|s| s.parse::<u8>().ok())
        .unwrap_or(0)
        & 0x0F;

    let s0_body =
        extract_between(trimmed, "S0(", ")").ok_or_else(|| LibTpuError::CompilationFailed {
            reason: String::from("Missing S0(...) slot in VLIW assembly line"),
        })?;
    let s_op = extract_between(&s0_body, "op=", ",")
        .and_then(|s| s.parse::<u8>().ok())
        .unwrap_or(1);
    let scalar_dst = extract_between(&s0_body, "$s", "<-")
        .and_then(|s| s.parse::<u8>().ok())
        .unwrap_or(0);
    let after_arrow_s = s0_body.split_once("<-").map_or("", |(_, r)| r);
    let s_parts: Vec<&str> = after_arrow_s.split(',').map(str::trim).collect();
    let scalar_lhs = s_parts
        .first()
        .and_then(|s| s.strip_prefix("$s"))
        .and_then(|s| s.parse::<u8>().ok())
        .unwrap_or(0);
    let scalar_rhs = s_parts
        .get(1)
        .and_then(|s| s.strip_prefix("$s"))
        .and_then(|s| s.parse::<u8>().ok())
        .unwrap_or(0);
    let imm32 = s_parts
        .get(2)
        .and_then(|s| s.strip_prefix("imm="))
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(default_imm);

    let v0_body =
        extract_between(trimmed, "V0(", ")").ok_or_else(|| LibTpuError::CompilationFailed {
            reason: String::from("Missing V0(...) slot in VLIW assembly line"),
        })?;
    let (v_op_name, v_regs_part) =
        v0_body
            .split_once(',')
            .ok_or_else(|| LibTpuError::CompilationFailed {
                reason: String::from("Malformed V0(...) slot in VLIW assembly line"),
            })?;
    let resolved_op = (0_u8..=101_u8).find_map(|byte| {
        TpuOpCode::from_byte(byte)
            .ok()
            .filter(|candidate| format!("{candidate:?}") == v_op_name.trim())
    });
    let v_opcode = resolved_op.ok_or_else(|| LibTpuError::CompilationFailed {
        reason: format!("Unknown V0 opcode mnemonic '{}'", v_op_name.trim()),
    })?;
    let vec_out = extract_between(v_regs_part, "$v", "<-")
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0);
    let after_arrow_v = v_regs_part.split_once("<-").map_or("", |(_, r)| r);
    let v_parts: Vec<&str> = after_arrow_v.split(',').map(str::trim).collect();
    let vec_arg_a = v_parts
        .first()
        .and_then(|s| s.strip_prefix("$v"))
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0);
    let vec_arg_b = v_parts
        .get(1)
        .and_then(|s| s.strip_prefix("$v"))
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0);
    let vec_arg_c = v_parts
        .get(2)
        .and_then(|s| s.strip_prefix("$v"))
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0);

    let mut matrix_slot = MatrixMxuSlot::zeroed();
    let mut is_mxu = false;
    if let Some(m0_body) = extract_between(trimmed, "M0(", ")") {
        is_mxu = true;
        matrix_slot.opcode = 1;
        matrix_slot.flags = 1;
        matrix_slot.acc_vreg = extract_between(&m0_body, "acc=$v", ",")
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(0);
        matrix_slot.lhs_vreg = extract_between(&m0_body, "lhs=$v", ",")
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(0);
        matrix_slot.rhs_vreg = extract_between(&m0_body, "rhs=$v", ",")
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(0);
        matrix_slot.k_tiles = m0_body
            .rsplit_once("k=")
            .and_then(|(_, s)| s.trim().parse::<u16>().ok())
            .unwrap_or(1);
    }

    let mut xunary_slot = ExtendedUnarySlot::zeroed();
    let mut is_xunary = false;
    if let Some(xlu_body) = extract_between(trimmed, "XLU(", ")") {
        is_xunary = true;
        xunary_slot.opcode = extract_between(&xlu_body, "op=", ",")
            .and_then(|s| s.parse::<u8>().ok())
            .unwrap_or(v_opcode as u8);
        xunary_slot.dst_vreg = extract_between(&xlu_body, "$v", "<-")
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(0);
        xunary_slot.src_vreg = xlu_body
            .rsplit_once("$v")
            .and_then(|(_, s)| s.trim().parse::<u16>().ok())
            .unwrap_or(0);
    }

    let mut dma_slot = AsyncDmaSlot::zeroed();
    let mut is_dma = false;
    if let Some(d0_body) = extract_between(trimmed, "D0(", ")") {
        is_dma = true;
        dma_slot.dma_opcode = extract_between(&d0_body, "op=", ",")
            .and_then(|s| s.parse::<u8>().ok())
            .unwrap_or(1);
        dma_slot.src_space = 1;
        dma_slot.dst_space = 0;
        dma_slot.collective_op = v_opcode as u8;
        dma_slot.sync_flag_id = d0_body
            .rsplit_once("flag=")
            .and_then(|(_, s)| s.trim().parse::<u32>().ok())
            .unwrap_or(0);
    }

    let bundle_flags: u16 =
        0x0003 | (u16::from(is_mxu) << 2) | (u16::from(is_xunary) << 3) | (u16::from(is_dma) << 4);

    let s0 = ScalarAluSlot {
        opcode: s_op,
        dst_sreg: scalar_dst,
        src0_sreg: scalar_lhs,
        src1_sreg: scalar_rhs,
        imm32,
    };
    let v0 = VectorAluSlot {
        opcode: u16::from(v_opcode as u8),
        element_type: 11,
        lane_mask: 0xFF,
        dst_vreg: vec_out,
        src0_vreg: vec_arg_a,
        src1_vreg: vec_arg_b,
        src2_vreg: vec_arg_c,
    };

    Ok(LloBundle {
        generation,
        bundle_flags,
        pred_mask: 0xFFFF,
        stall_cycles,
        scalar_slots: [s0, ScalarAluSlot::default()],
        vector_slots: [v0, VectorAluSlot::default()],
        matrix_slots: [
            matrix_slot,
            MatrixMxuSlot::default(),
            MatrixMxuSlot::default(),
            MatrixMxuSlot::default(),
        ],
        xlu_slots: [xunary_slot, ExtendedUnarySlot::default()],
        dma_slot,
        immediates: [imm32, 0, 0, 0, 0, 0],
    })
}

/// Formats a sequence of [`LloBundle`]s into a multi-line `DeepSea` LLO assembly listing.
///
/// # Arguments
///
/// * `bundles` - Slice of [`LloBundle`]s to format.
#[must_use]
pub fn format_llo_assembly(bundles: &[LloBundle]) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let mut cycle = 0_u32;
    for (pc, bundle) in bundles.iter().enumerate() {
        let stall = u32::from(bundle.stall_cycles);
        cycle = cycle.saturating_add(stall);
        let pred_idx = bundle.pred_mask.trailing_zeros().min(15);
        let dis = disassemble_llo_bundle(bundle, pc);
        let _ = writeln!(out, "[c={cycle:04}, stall={stall}] @p{pred_idx} {dis}");
        cycle = cycle.saturating_add(1);
    }
    out
}

/// Parses a multi-line `DeepSea` LLO assembly listing into a vector of [`LloBundle`]s.
///
/// # Arguments
///
/// * `listing` - Multi-line `DeepSea` LLO assembly text.
///
/// # Errors
///
/// Returns [`LibTpuError::CompilationFailed`] if any non-empty, non-comment line fails to assemble.
pub fn parse_llo_assembly(listing: &str) -> LibTpuResult<Vec<LloBundle>> {
    let mut bundles = Vec::new();
    for raw_line in listing.lines() {
        let trimmed = raw_line.trim();
        if trimmed.is_empty() | trimmed.starts_with("//") | trimmed.starts_with('#') {
            continue;
        }
        let pc_pos = trimmed
            .find("[pc=")
            .ok_or_else(|| LibTpuError::CompilationFailed {
                reason: format!("Missing '[pc=' marker in LLO assembly line: '{trimmed}'"),
            })?;
        let canonical = trimmed.get(pc_pos..).unwrap_or(trimmed);
        bundles.push(assemble_llo_bundle(canonical)?);
    }
    Ok(bundles)
}

/// Cycle-accurate architectural register & functional-unit hazard scoreboard (`VliwScoreboard`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VliwScoreboard {
    /// Current schedule cycle counter.
    pub current_cycle: u32,
    /// Cycle at which each scalar register `SREG[0..64]` becomes available.
    pub sreg_ready: [u32; 64],
    /// Cycle at which each predicate register `PREG[0..16]` becomes available.
    pub preg_ready: [u32; 16],
    /// Cycle at which each vector register `VREG[0..64]` becomes available.
    pub vreg_ready: [u32; 64],
    /// Cycle at which each vector mask register `VMASK[0..8]` becomes available.
    pub vmask_ready: [u32; 8],
    /// Cycle at which `MXU0` and `MXU1` systolic accumulators become available.
    pub mxu_accum_ready: [u32; 2],
    /// Cycle at which each of the 16 `VMEM` scratchpad banks becomes conflict-free.
    pub vmem_bank_ready: [u32; 16],
    /// Cycle at which each hardware `SyncFlag[0..256]` DMA barrier completes.
    pub sync_flag_ready: Vec<u32>,
}

impl Default for VliwScoreboard {
    fn default() -> Self {
        Self::new()
    }
}

impl VliwScoreboard {
    /// Creates a zero-initialized [`VliwScoreboard`] at cycle `0`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            current_cycle: 0,
            sreg_ready: [0; 64],
            preg_ready: [0; 16],
            vreg_ready: [0; 64],
            vmask_ready: [0; 8],
            mxu_accum_ready: [0; 2],
            vmem_bank_ready: [0; 16],
            sync_flag_ready: vec![0; 256],
        }
    }

    /// Computes `(bank_index, conflict_stall_cycles)` for concurrent `V0`/`D0` VMEM accesses.
    ///
    /// # Arguments
    ///
    /// * `v0_vmem_offset` - Byte offset in `VMEM` accessed by `V0`.
    /// * `d0_vmem_offset` - Byte offset in `VMEM` accessed by `D0`.
    #[must_use]
    #[allow(clippy::indexing_slicing, clippy::cast_possible_truncation)]
    pub fn check_vmem_bank_conflict(
        &self,
        v0_vmem_offset: Option<u64>,
        d0_vmem_offset: Option<u64>,
    ) -> (usize, u8) {
        let v_bank = v0_vmem_offset.map(|off| ((off / 256) as usize) & 15);
        let d_bank = d0_vmem_offset.map(|off| ((off / 256) as usize) & 15);
        let primary_bank = v_bank.or(d_bank).unwrap_or(0);
        let bank_stall = self.vmem_bank_ready[primary_bank]
            .saturating_sub(self.current_cycle)
            .min(15);
        let same_bank_penalty = u32::from(v_bank.is_some() && (v_bank == d_bank));
        let total_stall = bank_stall.saturating_add(same_bank_penalty).min(15) as u8;
        (primary_bank, total_stall)
    }

    /// Records the issue of `bundle` with `latency` and returns required stall cycles (`0..=15`).
    ///
    /// # Arguments
    ///
    /// * `bundle` - Logical [`TpuInstructionBundle`] being scheduled.
    /// * `latency` - Functional unit latency in cycles.
    #[allow(clippy::indexing_slicing, clippy::cast_possible_truncation)]
    pub fn record_issue(&mut self, bundle: &TpuInstructionBundle, latency: u8) -> u8 {
        let mut required_cycle = self.current_cycle;
        for &src_slot in &bundle.operand_slots {
            required_cycle = required_cycle.max(self.vreg_ready[src_slot & 63]);
        }
        required_cycle = required_cycle.max(self.vreg_ready[bundle.output_slot & 63]);
        let v0_bank = bundle.output_slot & 15;
        required_cycle = required_cycle.max(self.vmem_bank_ready[v0_bank]);
        if bundle.opcode == TpuOpCode::DotGeneralF32 {
            required_cycle = required_cycle.max(self.mxu_accum_ready[0]);
        }
        if bundle.opcode.has_side_effects() {
            required_cycle =
                required_cycle.max(self.sync_flag_ready[(bundle.sync_flag_id as usize) & 255]);
        }
        let stall_cycles = required_cycle.saturating_sub(self.current_cycle).min(15) as u8;
        self.current_cycle = required_cycle.saturating_add(1);
        let ready_at = self.current_cycle.saturating_add(u32::from(latency));
        self.vreg_ready[bundle.output_slot & 63] = ready_at;
        self.vmem_bank_ready[v0_bank] = self.current_cycle.saturating_add(1);
        self.sreg_ready[bundle.attrs.iota_dim & 63] = self.current_cycle.saturating_add(1);
        if bundle.opcode == TpuOpCode::DotGeneralF32 {
            self.mxu_accum_ready[0] = ready_at;
        }
        if bundle.opcode.has_side_effects() {
            self.sync_flag_ready[(bundle.sync_flag_id as usize) & 255] = ready_at;
        }
        stall_cycles
    }
}

/// Polyhedral loop tile and double-buffered VMEM DMA prefetch descriptor (`PolyhedralTileSchedule`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PolyhedralTileSchedule {
    /// Batch dimension tile factor (`>= 1`).
    pub batch_tiles: u32,
    /// Non-contracting row dimension (`M`) tile count.
    pub m_tiles: u32,
    /// Contracting dimension (`K`) tile count.
    pub k_tiles: u32,
    /// Non-contracting column dimension (`N`) tile count.
    pub n_tiles: u32,
    /// Ping-buffer byte offset in `VMEM` scratchpad.
    pub ping_vmem_offset: u64,
    /// Pong-buffer byte offset in `VMEM` scratchpad.
    pub pong_vmem_offset: u64,
    /// Architectural `SyncFlag` ID used to synchronize `HBM -> VMEM` prefetch completion.
    pub prefetch_sync_flag_id: u32,
}

impl PolyhedralTileSchedule {
    /// Computes the polyhedral tile schedule and double-buffered `VMEM` ping/pong offsets.
    ///
    /// # Arguments
    ///
    /// * `shape` - Logical tensor [`crate::types::Shape`] to tile.
    /// * `generation` - Target Cloud TPU generation.
    /// * `base_vmem_offset` - Starting byte offset in `VMEM`.
    /// * `sync_flag_id` - Base `SyncFlag` ID for DMA completion.
    #[must_use]
    pub fn from_shape(
        shape: &crate::types::Shape,
        generation: TpuGeneration,
        base_vmem_offset: u64,
        sync_flag_id: u32,
    ) -> Self {
        let mxu_dim = u64::from(generation.mxu_dimension().max(128));
        let rank = shape.dims.len();
        let d_n = u64::try_from(shape.dims.last().copied().unwrap_or(1).max(1)).unwrap_or(1);
        let d_m = if rank >= 2 {
            u64::try_from(shape.dims.get(rank - 2).copied().unwrap_or(1).max(1)).unwrap_or(1)
        } else {
            8
        };
        let d_b: u64 = if rank >= 3 {
            shape.dims.iter().take(rank - 2).fold(1_u64, |acc, &d| {
                acc.saturating_mul(u64::try_from(d.max(1)).unwrap_or(1))
            })
        } else {
            1
        };
        let m_tiles = u32::try_from(d_m.div_ceil(8).max(1)).unwrap_or(1);
        let k_tiles = u32::try_from(d_n.div_ceil(mxu_dim).max(1)).unwrap_or(1);
        let n_tiles = u32::try_from(d_n.div_ceil(mxu_dim).max(1)).unwrap_or(1);
        let tile_bytes = (mxu_dim * 8 * 4).max(4096);
        Self {
            batch_tiles: u32::try_from(d_b).unwrap_or(1),
            m_tiles,
            k_tiles,
            n_tiles,
            ping_vmem_offset: base_vmem_offset,
            pong_vmem_offset: base_vmem_offset.saturating_add(tile_bytes),
            prefetch_sync_flag_id: sync_flag_id.max(1),
        }
    }
}

/// Software-pipelined stage classification for [`ModuloPipelineScheduler`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModuloPipelineStage {
    /// Stage 0 (`Prologue`): Issues initial `HBM -> VMEM_ping` prefetch DMA (`D0`).
    Prologue,
    /// Stage 1 (`SteadyState`): Simultaneously prefetches tile `k + 1`, executes `MXU`, and runs `VPU` epilogue.
    SteadyState,
    /// Stage 2 (`Epilogue`): Drains final `MXU` accumulator tiles and writes results `VMEM -> HBM`.
    Epilogue,
}

/// Software-pipelined schedule produced by [`ModuloPipelineScheduler`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModuloPipelinePlan {
    /// Initiation interval (`II`) in clock cycles between successive tile iterations.
    pub initiation_interval_cycles: u32,
    /// Total number of software-pipelined loop iterations (`k_tiles`).
    pub total_iterations: u32,
    /// Ordered `(ModuloPipelineStage, LloBundle)` sequence across prologue, steady-state, and epilogue.
    pub staged_bundles: Vec<(ModuloPipelineStage, LloBundle)>,
}

/// Modulo Software Pipeline Scheduler (`ModuloPipelineScheduler`) over [`LloBundle`]s.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModuloPipelineScheduler {
    /// Target Cloud TPU generation.
    pub generation: TpuGeneration,
}

impl ModuloPipelineScheduler {
    /// Creates a new [`ModuloPipelineScheduler`] for `generation`.
    ///
    /// # Arguments
    ///
    /// * `generation` - Target [`TpuGeneration`].
    #[must_use]
    pub const fn new(generation: TpuGeneration) -> Self {
        Self { generation }
    }

    /// Computes `(resource_mii, recurrence_mii, mii)` for `loop_bundles`.
    ///
    /// # Arguments
    ///
    /// * `loop_bundles` - Inner loop [`TpuInstructionBundle`] slice.
    #[must_use]
    pub fn compute_minimum_initiation_interval(
        &self,
        loop_bundles: &[TpuInstructionBundle],
    ) -> (u32, u32, u32) {
        let llo = LloVliwScheduler::new(self.generation);
        let mut scalar_count = 0_u32;
        let mut vector_count = 0_u32;
        let mut matrix_count = 0_u32;
        let mut xlu_count = 0_u32;
        let mut dma_count = 0_u32;
        let mut recurrence_mii = 1_u32;
        let mut def_latencies: std::collections::BTreeMap<usize, u32> =
            std::collections::BTreeMap::new();

        for bundle in loop_bundles {
            scalar_count = scalar_count.saturating_add(1);
            if bundle.opcode == TpuOpCode::DotGeneralF32 {
                matrix_count = matrix_count.saturating_add(1);
            } else if bundle.opcode.is_transcendental() {
                xlu_count = xlu_count.saturating_add(1);
            } else if bundle.opcode.has_side_effects() {
                dma_count = dma_count.saturating_add(1);
            } else {
                vector_count = vector_count.saturating_add(1);
            }
            let lat = u32::from(llo.instruction_latency_cycles(bundle.opcode));
            for &op_slot in &bundle.operand_slots {
                if let Some(&prev_lat) = def_latencies.get(&op_slot) {
                    recurrence_mii = recurrence_mii.max(prev_lat.saturating_add(lat));
                }
            }
            def_latencies.insert(bundle.output_slot, lat);
        }
        let mxu_units = self.generation.mxu_count_per_core().max(1);
        let resource_mii = scalar_count
            .div_ceil(2)
            .max(vector_count.div_ceil(2))
            .max(matrix_count.div_ceil(mxu_units))
            .max(xlu_count)
            .max(dma_count)
            .max(1);
        let mii = resource_mii.max(recurrence_mii);
        (resource_mii, recurrence_mii, mii)
    }

    /// Lowers `SparseCore` [`TpuInstructionBundle`]s into a [`crate::proto::BarnaCoreExecutableProto`].
    ///
    /// # Arguments
    ///
    /// * `module_name` - Executable module name.
    /// * `bundles` - Slice of lowered [`TpuInstructionBundle`]s.
    #[must_use]
    pub fn lower_sparse_core_to_barnacore_proto(
        &self,
        module_name: &str,
        bundles: &[TpuInstructionBundle],
    ) -> crate::proto::BarnaCoreExecutableProto {
        crate::compiler::sparsecore::SparseCoreBackend::new(self.generation)
            .lower_bundles_to_barnacore_proto(module_name, bundles)
    }

    /// Generates a 3-stage software-pipelined [`ModuloPipelinePlan`] (`Prologue`, `SteadyState`, `Epilogue`).
    ///
    /// # Arguments
    ///
    /// * `contraction_bundle` - Logical contraction or loop body [`TpuInstructionBundle`].
    /// * `tile_schedule` - [`PolyhedralTileSchedule`] specifying tile counts and `ping`/`pong` VMEM offsets.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn schedule_tile_loop(
        &self,
        contraction_bundle: &TpuInstructionBundle,
        tile_schedule: &PolyhedralTileSchedule,
    ) -> ModuloPipelinePlan {
        let llo = LloVliwScheduler::new(self.generation);
        let ii = u32::from(llo.instruction_latency_cycles(contraction_bundle.opcode));
        let k_iters = tile_schedule.k_tiles.max(1);
        let mut staged = Vec::with_capacity((k_iters as usize) + 2);

        let mut prologue = contraction_bundle.encode_for_generation(self.generation, 0);
        prologue.bundle_flags |= 1 << 4;
        prologue.pred_mask = 0x0001;
        prologue.scalar_slots[0].imm32 = tile_schedule.ping_vmem_offset as u32;
        prologue.matrix_slots[0].opcode = 0;
        prologue.dma_slot.dma_opcode = 1;
        prologue.dma_slot.src_space = 0;
        prologue.dma_slot.dst_space = 1;
        prologue.dma_slot.sync_flag_id = tile_schedule.prefetch_sync_flag_id;
        staged.push((ModuloPipelineStage::Prologue, prologue));

        for step in 0..k_iters {
            let mut steady = contraction_bundle.encode_for_generation(self.generation, 1);
            steady.bundle_flags |= (1 << 2) | (1 << 4);
            let rot_bit = 1_u16 << ((step % 6) + 1);
            steady.pred_mask = rot_bit | 0x0001;
            let next_vmem = if (step & 1) == 0 {
                tile_schedule.pong_vmem_offset
            } else {
                tile_schedule.ping_vmem_offset
            };
            steady.scalar_slots[0].imm32 = next_vmem as u32;
            steady.matrix_slots[0].opcode = 1;
            steady.matrix_slots[0].k_tiles = u16::try_from(k_iters).unwrap_or(1);
            steady.dma_slot.dma_opcode = 1;
            steady.dma_slot.src_space = 0;
            steady.dma_slot.dst_space = 1;
            steady.dma_slot.sync_flag_id =
                tile_schedule.prefetch_sync_flag_id.saturating_add(step & 1);
            staged.push((ModuloPipelineStage::SteadyState, steady));
        }

        let mut epilogue = contraction_bundle.encode_for_generation(self.generation, 2);
        epilogue.bundle_flags |= 1 << 4;
        epilogue.pred_mask = 0x0080;
        epilogue.matrix_slots[0].opcode = 0;
        epilogue.dma_slot.dma_opcode = 2;
        epilogue.dma_slot.src_space = 1;
        epilogue.dma_slot.dst_space = 0;
        epilogue.dma_slot.sync_flag_id = tile_schedule.prefetch_sync_flag_id.saturating_add(2);
        staged.push((ModuloPipelineStage::Epilogue, epilogue));

        ModuloPipelinePlan {
            initiation_interval_cycles: ii,
            total_iterations: k_iters,
            staged_bundles: staged,
        }
    }

    /// Exports a [`crate::proto::LloSoftwarePipelineScheduleProto`] from `loop_bundles` and `plan`.
    ///
    /// # Arguments
    ///
    /// * `loop_name` - Name of the software-pipelined tile loop.
    /// * `loop_bundles` - Inner loop instruction bundles.
    /// * `plan` - Scheduled [`ModuloPipelinePlan`].
    #[must_use]
    pub fn to_software_pipeline_schedule_proto(
        &self,
        loop_name: &str,
        loop_bundles: &[TpuInstructionBundle],
        plan: &ModuloPipelinePlan,
    ) -> crate::proto::LloSoftwarePipelineScheduleProto {
        let (resource_mii, recurrence_mii, mii) =
            self.compute_minimum_initiation_interval(loop_bundles);
        let stages = plan
            .staged_bundles
            .iter()
            .enumerate()
            .map(|(idx, (stage, b))| {
                let stage_idx = match stage {
                    ModuloPipelineStage::Prologue => 0,
                    ModuloPipelineStage::SteadyState => 1,
                    ModuloPipelineStage::Epilogue => 2,
                };
                let start_c = u32::try_from(idx).unwrap_or(0).saturating_mul(mii);
                crate::proto::LloModuloStageProto {
                    stage_index: stage_idx,
                    rotating_predicate_reg: b.pred_mask.trailing_zeros().min(15),
                    start_cycle: start_c,
                    end_cycle: start_c.saturating_add(mii),
                }
            })
            .collect();
        crate::proto::LloSoftwarePipelineScheduleProto {
            loop_name: String::from(loop_name),
            resource_mii,
            recurrence_mii,
            achieved_ii: plan.initiation_interval_cycles.max(mii),
            stages,
        }
    }

    /// Exports a [`crate::proto::LloBundleScheduleProto`] from `plan`.
    ///
    /// # Arguments
    ///
    /// * `schedule_name` - Schedule identifier string.
    /// * `plan` - Scheduled [`ModuloPipelinePlan`].
    #[must_use]
    pub fn to_bundle_schedule_proto(
        &self,
        schedule_name: &str,
        plan: &ModuloPipelinePlan,
    ) -> crate::proto::LloBundleScheduleProto {
        let scheduled_instructions = plan
            .staged_bundles
            .iter()
            .enumerate()
            .map(|(pc, (_, b))| {
                let fu = if b.matrix_slots[0].opcode != 0 { 2 } else { 4 };
                let pc_u32 = u32::try_from(pc).unwrap_or(0);
                crate::proto::LloScheduledInstructionProto {
                    bundle_pc: pc_u32,
                    functional_unit: fu,
                    opcode: u32::from(b.vector_slots[0].opcode),
                    predicate_reg: b.pred_mask.trailing_zeros().min(15),
                    stall_cycles: u32::from(b.stall_cycles),
                    hlo_instruction_name: format!("bundle_pc_{pc_u32}"),
                }
            })
            .collect();
        crate::proto::LloBundleScheduleProto {
            schedule_name: String::from(schedule_name),
            total_cycles: plan
                .initiation_interval_cycles
                .saturating_mul(u32::try_from(plan.staged_bundles.len()).unwrap_or(1)),
            initiation_interval_mii: plan.initiation_interval_cycles,
            scheduled_instructions,
        }
    }
}

/// Latency-aware `DeepSea` VLIW instruction list scheduler (`LloVliwScheduler`) producing [`LloBundle`]s.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LloVliwScheduler {
    /// Target Cloud TPU hardware generation.
    pub generation: TpuGeneration,
}

impl LloVliwScheduler {
    /// Creates a new [`LloVliwScheduler`] for `generation`.
    ///
    /// # Arguments
    ///
    /// * `generation` - Target [`TpuGeneration`].
    #[must_use]
    pub const fn new(generation: TpuGeneration) -> Self {
        Self { generation }
    }

    /// Returns the architectural execution latency in cycles for `opcode` on `self.generation`.
    ///
    /// # Arguments
    ///
    /// * `opcode` - Hardware [`TpuOpCode`] to query.
    #[must_use]
    pub const fn instruction_latency_cycles(&self, opcode: TpuOpCode) -> u8 {
        if matches!(opcode, TpuOpCode::DotGeneralF32) {
            match self.generation {
                TpuGeneration::V6e | TpuGeneration::V7x => 12,
                _ => 8,
            }
        } else if opcode.is_transcendental() {
            4
        } else if opcode.has_side_effects() {
            6
        } else if matches!(opcode, TpuOpCode::Constant | TpuOpCode::Identity) {
            1
        } else {
            2
        }
    }

    /// Schedules [`TpuInstructionBundle`]s into hazard-annotated [`LloBundle`]s using [`VliwScoreboard`].
    ///
    /// # Arguments
    ///
    /// * `bundles` - Slice of logical instruction bundles to schedule.
    #[must_use]
    pub fn schedule_bundles(&self, bundles: &[TpuInstructionBundle]) -> Vec<LloBundle> {
        let mut scoreboard = VliwScoreboard::new();
        let mut out = Vec::with_capacity(bundles.len());
        for bundle in bundles {
            let latency = self.instruction_latency_cycles(bundle.opcode);
            let stall_cycles = scoreboard.record_issue(bundle, latency);
            out.push(bundle.encode_for_generation(self.generation, stall_cycles));
        }
        out
    }

    /// Co-issues independent [`TpuInstructionBundle`]s across `(S0, V0, M0, XLU, D0)` slots into [`LloBundle`]s.
    ///
    /// # Arguments
    ///
    /// * `bundles` - Slice of logical instruction bundles to schedule and pack.
    #[must_use]
    pub fn pack_independent_bundles(&self, bundles: &[TpuInstructionBundle]) -> Vec<LloBundle> {
        let scheduled = self.schedule_bundles(bundles);
        let mut packed = Vec::with_capacity(scheduled.len());
        let mut idx = 0_usize;
        while let Some(&first) = scheduled.get(idx) {
            let mut merged = first;
            if let Some(&second) = scheduled.get(idx + 1) {
                let first_dst = first.vector_slots[0].dst_vreg;
                let second_reads_first = second.vector_slots[0].src0_vreg == first_dst;
                let can_merge_mxu =
                    (second.matrix_slots[0].opcode != 0) && (first.matrix_slots[0].opcode == 0);
                let can_merge_xlu =
                    (second.xlu_slots[0].opcode != 0) && (first.xlu_slots[0].opcode == 0);
                let can_merge_dma =
                    (second.dma_slot.dma_opcode != 0) && (first.dma_slot.dma_opcode == 0);
                if (!second_reads_first) && (can_merge_mxu | can_merge_xlu | can_merge_dma) {
                    if can_merge_mxu {
                        merged.matrix_slots[0] = second.matrix_slots[0];
                        merged.bundle_flags |= 1 << 2;
                    }
                    if can_merge_xlu {
                        merged.xlu_slots[0] = second.xlu_slots[0];
                        merged.bundle_flags |= 1 << 3;
                    }
                    if can_merge_dma {
                        merged.dma_slot = second.dma_slot;
                        merged.bundle_flags |= 1 << 4;
                    }
                    idx += 2;
                    packed.push(merged);
                    continue;
                }
            }
            packed.push(merged);
            idx += 1;
        }
        packed
    }
}

impl TpuInstructionBundle {
    /// Encodes this [`TpuInstructionBundle`] into a 16-byte binary [`VliwInstructionWord`].
    #[must_use]
    pub fn encode_vliw_word(&self) -> VliwInstructionWord {
        let mut flags: u16 = 0;
        if self.opcode.is_binary() {
            flags |= 1 << 0;
        }
        if self.opcode.is_transcendental() {
            flags |= 1 << 1;
        }
        let s0 = self.operand_slots.first().copied().unwrap_or(0);
        let s1 = self
            .operand_slots
            .get(1)
            .copied()
            .unwrap_or(self.sync_flag_id as usize);
        VliwInstructionWord {
            opcode: u16::from(self.opcode as u8),
            unit_flags: flags,
            dst_reg: u32::try_from(self.output_slot).unwrap_or(0),
            src0_reg: u32::try_from(s0).unwrap_or(0),
            src1_reg: u32::try_from(s1).unwrap_or(0),
        }
    }

    /// Encodes this [`TpuInstructionBundle`] into an architectural [`LloBundle`].
    #[must_use]
    pub fn encode_architectural_bundle(&self) -> LloBundle {
        let tpu_gen = crate::config::global_init_config().generation;
        self.encode_for_generation(tpu_gen, 0)
    }

    /// Encodes this [`TpuInstructionBundle`] into an [`LloBundle`] for `generation` and `stall_cycles`.
    ///
    /// # Arguments
    ///
    /// * `generation` - Target [`TpuGeneration`].
    /// * `stall_cycles` - Inter-bundle hazard stall cycles (`0..=15`).
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn encode_for_generation(&self, generation: TpuGeneration, stall_cycles: u8) -> LloBundle {
        let s0 = u32::try_from(self.operand_slots.first().copied().unwrap_or(0)).unwrap_or(0);
        let s1 = u32::try_from(self.operand_slots.get(1).copied().unwrap_or(0)).unwrap_or(0);
        let s2 = u32::try_from(self.operand_slots.get(2).copied().unwrap_or(0)).unwrap_or(0);
        let dst = u32::try_from(self.output_slot).unwrap_or(0);
        let et_raw = self.attrs.output_shape.as_ref().map_or(11_u8, |sh| {
            u8::try_from(sh.element_type.as_raw()).unwrap_or(11)
        });
        let iota_u8 = u8::try_from(self.attrs.iota_dim).unwrap_or(0);
        let concat_u8 = u8::try_from(self.attrs.concat_dim).unwrap_or(0);
        let red_mode = self
            .attrs
            .reduce_dims
            .first()
            .map_or(0_u8, |&d| u8::try_from(d.saturating_add(1)).unwrap_or(0));

        let is_mxu = self.opcode == TpuOpCode::DotGeneralF32;
        let is_xunary = self.opcode.is_transcendental()
            | matches!(
                self.opcode,
                TpuOpCode::Transpose
                    | TpuOpCode::BroadcastInDim
                    | TpuOpCode::Reverse
                    | TpuOpCode::Iota
                    | TpuOpCode::DynamicSlice
                    | TpuOpCode::Concatenate
                    | TpuOpCode::Slice
                    | TpuOpCode::Pad
                    | TpuOpCode::Gather
                    | TpuOpCode::Scatter
            );
        let is_collective = matches!(
            self.opcode,
            TpuOpCode::AllReduceSumF32 | TpuOpCode::AllGather | TpuOpCode::ReduceScatter
        );
        let is_dma = is_collective || (self.sync_flag_id != 0);
        let bundle_flags: u16 = 0x0003
            | (u16::from(is_mxu) << 2)
            | (u16::from(is_xunary) << 3)
            | (u16::from(is_dma) << 4);

        let k_tiles = match generation {
            TpuGeneration::V6e | TpuGeneration::V7x => 2,
            _ => 1,
        };

        let s_slot = ScalarAluSlot {
            opcode: 1,
            dst_sreg: 0,
            src0_sreg: iota_u8,
            src1_sreg: concat_u8,
            imm32: self.pc_offset,
        };
        let v_slot = VectorAluSlot {
            opcode: u16::from(self.opcode as u8),
            element_type: et_raw,
            lane_mask: 0xFF,
            dst_vreg: dst,
            src0_vreg: s0,
            src1_vreg: s1,
            src2_vreg: s2,
        };
        let m_slot = MatrixMxuSlot {
            opcode: u8::from(is_mxu),
            precision_mode: 0,
            flags: u16::from(is_mxu),
            acc_vreg: u16::try_from(dst).unwrap_or(0),
            lhs_vreg: u16::try_from(s0).unwrap_or(0),
            rhs_vreg: u16::try_from(s1).unwrap_or(0),
            k_tiles,
        };
        let x_slot = ExtendedUnarySlot {
            opcode: if is_xunary { self.opcode as u8 } else { 0 },
            perm_mode: red_mode,
            dst_vreg: u16::try_from(dst).unwrap_or(0),
            src_vreg: u16::try_from(s0).unwrap_or(0),
            imm16: 0,
        };
        let dma_slot = AsyncDmaSlot {
            dma_opcode: if is_collective {
                self.opcode as u8
            } else {
                u8::from(self.sync_flag_id != 0)
            },
            src_space: 1,
            dst_space: 0,
            collective_op: if is_collective { self.opcode as u8 } else { 0 },
            sync_flag_id: self.sync_flag_id,
        };

        LloBundle {
            generation,
            bundle_flags,
            pred_mask: 0xFFFF,
            stall_cycles: stall_cycles & 0x0F,
            scalar_slots: [s_slot, ScalarAluSlot::default()],
            vector_slots: [v_slot, VectorAluSlot::default()],
            matrix_slots: [
                m_slot,
                MatrixMxuSlot::default(),
                MatrixMxuSlot::default(),
                MatrixMxuSlot::default(),
            ],
            xlu_slots: [x_slot, ExtendedUnarySlot::default()],
            dma_slot,
            immediates: [self.pc_offset, 0, 0, 0, 0, 0],
        }
    }

    /// Reconstructs a [`TpuInstructionBundle`] from an [`LloBundle`] at `pc_offset`.
    ///
    /// # Arguments
    ///
    /// * `bundle` - Decoded [`LloBundle`].
    /// * `pc_offset` - Positional bundle index in `IMEM`.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if the encoded opcode is invalid.
    pub fn from_llo_bundle(bundle: &LloBundle, pc_offset: u32) -> LibTpuResult<Self> {
        let mut ir = bundle.to_ir_bundle(pc_offset)?;
        ir.attrs.iota_dim = usize::from(bundle.scalar_slots[0].src0_sreg);
        ir.attrs.concat_dim = usize::from(bundle.scalar_slots[0].src1_sreg);
        if bundle.xlu_slots[0].perm_mode != 0 {
            ir.attrs
                .reduce_dims
                .push(usize::from(bundle.xlu_slots[0].perm_mode.saturating_sub(1)));
        }
        Ok(ir)
    }
}

/// Encodes `bundle` using [`encode_bundle_for_generation`] and verifies round-trip decoding.
///
/// # Arguments
///
/// * `generation` - Target [`TpuGeneration`].
/// * `bundle` - Source [`LloBundle`].
///
/// # Errors
///
/// Returns [`LibTpuError`] if round-trip decoding fails.
pub fn roundtrip_generation_bundle(
    generation: TpuGeneration,
    bundle: &LloBundle,
) -> LibTpuResult<LloBundle> {
    let encoded = encode_bundle_for_generation(generation, bundle);
    decode_bundle_for_generation(generation, &encoded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::opcode::InstructionAttrs;

    #[test]
    fn vliw_word_and_bundle_words_round_trip() {
        let word = VliwInstructionWord {
            opcode: 3,
            unit_flags: 1,
            dst_reg: 2,
            src0_reg: 0,
            src1_reg: 1,
        };
        let raw_w = word.to_bytes();
        assert_eq!(VliwInstructionWord::from_bytes(&raw_w), word);

        let ir = TpuInstructionBundle {
            pc_offset: 0,
            opcode: TpuOpCode::DotGeneralF32,
            operand_slots: vec![0, 1],
            output_slot: 2,
            sync_flag_id: 3,
            attrs: InstructionAttrs {
                iota_dim: 1,
                concat_dim: 2,
                reduce_dims: vec![0],
                ..InstructionAttrs::default()
            },
        };
        assert_eq!(
            ir.encode_vliw_word().opcode,
            u16::from(TpuOpCode::DotGeneralF32 as u8)
        );
        let mut llo_b = ir.encode_architectural_bundle();
        llo_b.set_dual_issue_slots(1, 3, 2);
        assert_eq!(llo_b.dual_issue_slots(), (1, 3, 2));

        let jfc = JfcV2V3BundleWord::from_bundle(&llo_b);
        assert!(jfc.to_bundle().is_ok());
        let pfc = PfcV4BundleWord::from_bundle(&llo_b);
        assert!(pfc.to_bundle().is_ok());
        let vfc = VfcV5BundleWord::from_bundle(&llo_b);
        assert!(vfc.to_bundle().is_ok());
        let gfc = VfkV6V7BundleWord::from_bundle(&llo_b);
        assert!(gfc.to_bundle().is_ok());

        for g in [
            TpuGeneration::V2,
            TpuGeneration::V3,
            TpuGeneration::V4,
            TpuGeneration::V4Lite,
            TpuGeneration::V5e,
            TpuGeneration::V5p,
            TpuGeneration::V6e,
            TpuGeneration::V7x,
        ] {
            assert!(roundtrip_generation_bundle(g, &llo_b).is_ok());
            let mut bs = llo_b.encode_generation_bitstream(g);
            pack_slot_bitfield(g, &mut bs, crate::elf::IsaSlotKind::S0, 0x05);
            assert!(extract_slot_bitfield(g, &bs, crate::elf::IsaSlotKind::S0).is_some());
        }

        let listing = format_llo_assembly(&[llo_b]);
        assert!(listing.contains("pc=0000"));
        let parsed = parse_llo_assembly(&listing);
        assert!(parsed.is_ok());
        assert!(TpuInstructionBundle::from_llo_bundle(&llo_b, 0).is_ok());
    }
}

//! Architectural 64-byte (`512`-bit) `TensorCore` VLIW instruction bundle layout and encoders.

use super::opcode::{InstructionAttrs, TpuInstructionBundle, TpuOpCode};
use crate::error::{LibTpuError, LibTpuResult};
use bytemuck::Zeroable;

/// Binary 16-byte VLIW instruction word emitted by [`TpuInstructionBundle::encode_vliw_word`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VliwInstructionWord {
    /// Hardware [`TpuOpCode`] discriminant (`u16`).
    pub opcode: u16,
    /// Execution unit and transcendental/collective flags (`u16`).
    pub unit_flags: u16,
    /// Destination VMEM/HBM register slot index (`u32`).
    pub dst_reg: u32,
    /// First source operand register slot index (`u32`).
    pub src0_reg: u32,
    /// Second source operand register slot index or `SyncFlag` ID (`u32`).
    pub src1_reg: u32,
}

const _: () = assert!(size_of::<VliwInstructionWord>() == 16);

/// 8-byte header of an architectural 64-byte [`TensorCoreVliwBundle`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VliwBundleHeader {
    /// Program counter bundle index (`u32`).
    pub pc_offset: u32,
    /// Active functional unit slot bitmask (`u16`).
    pub bundle_flags: u16,
    /// Predicate execution mask (`u16`).
    pub pred_mask: u16,
}

/// 8-byte Scalar ALU (`S0`) slot inside a [`TensorCoreVliwBundle`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ScalarAluSlot {
    /// Scalar opcode (`u8`).
    pub opcode: u8,
    /// Destination scalar register (`u8`).
    pub dst_sreg: u8,
    /// First source scalar register (`u8`).
    pub src0_sreg: u8,
    /// Second source scalar register (`u8`).
    pub src1_sreg: u8,
    /// 32-bit immediate value (`u32`).
    pub imm32: u32,
}

/// 20-byte Vector Processing Unit (`V0`) slot inside a [`TensorCoreVliwBundle`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VectorAluSlot {
    /// Vector ALU opcode (`u16`).
    pub opcode: u16,
    /// Element type code (`u8`).
    pub element_type: u8,
    /// Sublane mask (`u8`).
    pub lane_mask: u8,
    /// Destination VMEM register (`u32`).
    pub dst_vreg: u32,
    /// First source VMEM register (`u32`).
    pub src0_vreg: u32,
    /// Second source VMEM register (`u32`).
    pub src1_vreg: u32,
    /// Third source VMEM register or auxiliary parameter (`u32`).
    pub src2_vreg: u32,
}

/// 12-byte Matrix Multiply Unit (`M0`, `128x128` MXU systolic array) slot inside a [`TensorCoreVliwBundle`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MatrixMxuSlot {
    /// MXU opcode (`u8`).
    pub opcode: u8,
    /// Precision mode (`0` = `BF16`, `1` = `F32`, `2` = `INT8`) (`u8`).
    pub precision_mode: u8,
    /// MXU control flags (`u16`).
    pub flags: u16,
    /// Accumulator/destination register (`u16`).
    pub acc_vreg: u16,
    /// Left-hand activation register (`u16`).
    pub lhs_vreg: u16,
    /// Right-hand weight register (`u16`).
    pub rhs_vreg: u16,
    /// Contracting tile count (`u16`).
    pub k_tiles: u16,
}

/// 8-byte Extended Unary / Transcendental (`X0`) slot inside a [`TensorCoreVliwBundle`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ExtendedUnarySlot {
    /// Extended unary / cross-lane permute opcode (`u8`).
    pub opcode: u8,
    /// Permutation mode (`u8`).
    pub perm_mode: u8,
    /// Destination vector register (`u16`).
    pub dst_vreg: u16,
    /// Source vector register (`u16`).
    pub src_vreg: u16,
    /// 16-bit immediate parameter (`u16`).
    pub imm16: u16,
}

/// 8-byte Asynchronous DMA / ICI Collective (`D0`) slot inside a [`TensorCoreVliwBundle`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AsyncDmaSlot {
    /// DMA / collective opcode (`u8`).
    pub dma_opcode: u8,
    /// Source memory space (`0` = HBM, `1` = VMEM, `2` = SMEM, `3` = CMEM) (`u8`).
    pub src_space: u8,
    /// Destination memory space (`u8`).
    pub dst_space: u8,
    /// Collective reduction mode (`u8`).
    pub collective_op: u8,
    /// Architectural `SyncFlag` ID incremented upon completion (`u32`).
    pub sync_flag_id: u32,
}

/// Architectural 64-byte (`512`-bit) `TensorCore` VLIW instruction bundle (`TensorCoreVliwBundle`).
///
/// Combines parallel issue slots for the Scalar ALU (`S0`), Vector Processing Unit (`V0`),
/// Matrix Multiply Unit (`M0`), Extended Unary/Transcendental Unit (`X0`), and Async DMA/ICI Engine (`D0`).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TensorCoreVliwBundle {
    /// 8-byte bundle header (`0x00..0x08`).
    pub header: VliwBundleHeader,
    /// 8-byte Scalar ALU slot (`0x08..0x10`).
    pub scalar_slot: ScalarAluSlot,
    /// 20-byte Vector ALU slot (`0x10..0x24`).
    pub vector_slot: VectorAluSlot,
    /// 12-byte Matrix MXU slot (`0x24..0x30`).
    pub matrix_slot: MatrixMxuSlot,
    /// 8-byte Extended Unary / Transcendental slot (`0x30..0x38`).
    pub xunary_slot: ExtendedUnarySlot,
    /// 8-byte Async DMA / Collective slot (`0x38..0x40`).
    pub dma_slot: AsyncDmaSlot,
}

const _: () = assert!(size_of::<TensorCoreVliwBundle>() == 64);

/// Computes an 8-bit CRC-8 (`poly = 0x07`) parity digest over a byte slice for hardware VLIW bundle integrity.
///
/// # Arguments
///
/// * `bytes` - Raw instruction bundle byte slice to checksum.
#[must_use]
pub const fn compute_vliw_bundle_crc8(bytes: &[u8]) -> u8 {
    let mut crc: u8 = 0x5A;
    let mut idx = 0_usize;
    while idx < bytes.len() {
        #[allow(clippy::indexing_slicing)]
        let b = bytes[idx];
        crc ^= b;
        let mut bit = 0_u8;
        while bit < 8 {
            if (crc & 0x80) != 0 {
                crc = (crc << 1) ^ 0x07;
            } else {
                crc <<= 1;
            }
            bit += 1;
        }
        idx += 1;
    }
    crc
}

/// 40-byte (`320`-bit) bit-packed `DeepSea` VLIW instruction bundle for `TpuGeneration::V4` (`pufferfish`, `asic_sw::deepsea::pxc::pfc`).
///
/// Layout:
/// - `0x00..0x08` (`64` bits): Header (`pc_offset: u24`, `crc8: u8`, `bundle_flags: u16`, `pred_mask: u16`)
/// - `0x08..0x0E` (`48` bits): Scalar slots `S0` / `S1` (`opcode`, `dst_sreg`, `src0_sreg`, `src1_sreg`, `imm16`)
/// - `0x0E..0x1A` (`96` bits): Vector slots `V0` / `V1` (`opcode: u16`, `element_type: u8`, `lane_mask: u8`, `dst_vreg: u16`, `src0_vreg: u16`, `src1_vreg: u16`, `src2_vreg: u16`)
/// - `0x1A..0x22` (`64` bits): Matrix slot `M0` (`opcode: u8`, `precision_mode: u8`, `acc_vreg: u16`, `lhs_vreg: u16`, `rhs_vreg: u16`)
/// - `0x22..0x28` (`48` bits): Multiplexed `XLU` / `D0` slot (`xlu_op: u8`, `dma_op: u8`, `dst_or_space: u16`, `src_or_flag: u16`)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PfcV4BundleWord {
    /// Raw 40-byte (`320`-bit) `pfc` VLIW bundle bitstream.
    pub raw_bytes: [u8; 40],
}

const _: () = assert!(size_of::<PfcV4BundleWord>() == 40);

impl PfcV4BundleWord {
    /// Packs an architectural [`TensorCoreVliwBundle`] into a 40-byte (`320`-bit) `V4` [`PfcV4BundleWord`].
    ///
    /// # Arguments
    ///
    /// * `bundle` - Canonical 64-byte [`TensorCoreVliwBundle`] to compress.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn from_bundle(bundle: &TensorCoreVliwBundle) -> Self {
        let mut raw = [0_u8; 40];
        let pc_bytes = bundle.header.pc_offset.to_le_bytes();
        raw[0] = pc_bytes[0];
        raw[1] = pc_bytes[1];
        raw[2] = pc_bytes[2];
        let flags_bytes = bundle.header.bundle_flags.to_le_bytes();
        raw[4] = flags_bytes[0];
        raw[5] = flags_bytes[1];
        let pred_bytes = bundle.header.pred_mask.to_le_bytes();
        raw[6] = pred_bytes[0];
        raw[7] = pred_bytes[1];
        // 48-bit S0/S1 (bytes 8..14)
        raw[8] = bundle.scalar_slot.opcode;
        raw[9] = bundle.scalar_slot.dst_sreg;
        raw[10] = bundle.scalar_slot.src0_sreg;
        raw[11] = bundle.scalar_slot.src1_sreg;
        let imm16 = (bundle.scalar_slot.imm32 as u16).to_le_bytes();
        raw[12] = imm16[0];
        raw[13] = imm16[1];
        // 96-bit V0/V1 (bytes 14..26)
        let v_op = bundle.vector_slot.opcode.to_le_bytes();
        raw[14] = v_op[0];
        raw[15] = v_op[1];
        raw[16] = bundle.vector_slot.element_type;
        raw[17] = bundle.vector_slot.lane_mask;
        let v_dst = (bundle.vector_slot.dst_vreg as u16).to_le_bytes();
        let v_s0 = (bundle.vector_slot.src0_vreg as u16).to_le_bytes();
        let v_s1 = (bundle.vector_slot.src1_vreg as u16).to_le_bytes();
        let v_s2 = (bundle.vector_slot.src2_vreg as u16).to_le_bytes();
        raw[18] = v_dst[0];
        raw[19] = v_dst[1];
        raw[20] = v_s0[0];
        raw[21] = v_s0[1];
        raw[22] = v_s1[0];
        raw[23] = v_s1[1];
        raw[24] = v_s2[0];
        raw[25] = v_s2[1];
        // 64-bit M0 (bytes 26..34)
        raw[26] = bundle.matrix_slot.opcode;
        raw[27] = bundle.matrix_slot.precision_mode;
        let m_acc = bundle.matrix_slot.acc_vreg.to_le_bytes();
        let m_lhs = bundle.matrix_slot.lhs_vreg.to_le_bytes();
        let m_rhs = bundle.matrix_slot.rhs_vreg.to_le_bytes();
        raw[28] = m_acc[0];
        raw[29] = m_acc[1];
        raw[30] = m_lhs[0];
        raw[31] = m_lhs[1];
        raw[32] = m_rhs[0];
        raw[33] = m_rhs[1];
        // 48-bit XLU / D0 (bytes 34..40)
        raw[34] = bundle.xunary_slot.opcode;
        raw[35] = bundle.dma_slot.dma_opcode;
        raw[36] = bundle.xunary_slot.perm_mode;
        raw[37] = bundle.dma_slot.collective_op;
        let flag16 = (bundle.dma_slot.sync_flag_id as u16).to_le_bytes();
        raw[38] = flag16[0];
        raw[39] = flag16[1];
        // Compute CRC-8 over bytes 4..40 and store in raw[3]
        raw[3] = compute_vliw_bundle_crc8(&raw[4..40]);
        Self { raw_bytes: raw }
    }

    /// Unpacks and validates this 40-byte [`PfcV4BundleWord`] into a 64-byte [`TensorCoreVliwBundle`].
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidExecutableFormat`] if the embedded CRC-8 parity byte or opcode is invalid.
    pub fn to_bundle(&self) -> LibTpuResult<TensorCoreVliwBundle> {
        let raw = &self.raw_bytes;
        let expected_crc = compute_vliw_bundle_crc8(&raw[4..40]);
        if raw[3] != expected_crc {
            return Err(LibTpuError::InvalidExecutableFormat {
                reason: format!(
                    "PfcV4BundleWord CRC-8 mismatch: expected {expected_crc:#04x}, got {:#04x}",
                    raw[3]
                ),
            });
        }
        let pc_offset = u32::from_le_bytes([raw[0], raw[1], raw[2], 0]);
        let bundle_flags = u16::from_le_bytes([raw[4], raw[5]]);
        let pred_mask = u16::from_le_bytes([raw[6], raw[7]]);
        let v_opcode = u16::from_le_bytes([raw[14], raw[15]]);
        let raw_op = u8::try_from(v_opcode).map_err(|_| LibTpuError::InvalidExecutableFormat {
            reason: format!("Invalid PfcV4 vector opcode {v_opcode}"),
        })?;
        let _ = TpuOpCode::from_byte(raw_op).map_err(|_| LibTpuError::InvalidExecutableFormat {
            reason: format!("Unknown PfcV4 vector TpuOpCode {raw_op}"),
        })?;
        let v_dst = u32::from(u16::from_le_bytes([raw[18], raw[19]]));
        let v_s0 = u32::from(u16::from_le_bytes([raw[20], raw[21]]));
        let v_s1 = u32::from(u16::from_le_bytes([raw[22], raw[23]]));
        let v_s2 = u32::from(u16::from_le_bytes([raw[24], raw[25]]));
        let m_op = raw[26];
        let x_op = raw[34];
        let d_op = raw[35];
        Ok(TensorCoreVliwBundle {
            header: VliwBundleHeader {
                pc_offset,
                bundle_flags,
                pred_mask,
            },
            scalar_slot: ScalarAluSlot {
                opcode: raw[8],
                dst_sreg: raw[9],
                src0_sreg: raw[10],
                src1_sreg: raw[11],
                imm32: u32::from(u16::from_le_bytes([raw[12], raw[13]])),
            },
            vector_slot: VectorAluSlot {
                opcode: v_opcode,
                element_type: raw[16],
                lane_mask: raw[17],
                dst_vreg: v_dst,
                src0_vreg: v_s0,
                src1_vreg: v_s1,
                src2_vreg: v_s2,
            },
            matrix_slot: MatrixMxuSlot {
                opcode: m_op,
                precision_mode: raw[27],
                flags: u16::from(m_op != 0),
                acc_vreg: u16::from_le_bytes([raw[28], raw[29]]),
                lhs_vreg: u16::from_le_bytes([raw[30], raw[31]]),
                rhs_vreg: u16::from_le_bytes([raw[32], raw[33]]),
                k_tiles: u16::from(m_op != 0),
            },
            xunary_slot: ExtendedUnarySlot {
                opcode: x_op,
                perm_mode: raw[36],
                dst_vreg: if x_op != 0 {
                    u16::try_from(v_dst).unwrap_or(0)
                } else {
                    0
                },
                src_vreg: if x_op != 0 {
                    u16::try_from(v_s0).unwrap_or(0)
                } else {
                    0
                },
                imm16: 0,
            },
            dma_slot: AsyncDmaSlot {
                dma_opcode: d_op,
                src_space: u8::from(d_op != 0),
                dst_space: 0,
                collective_op: raw[37],
                sync_flag_id: u32::from(u16::from_le_bytes([raw[38], raw[39]])),
            },
        })
    }
}

/// 52-byte (`416`-bit) bit-packed `DeepSea` VLIW instruction bundle for `TpuGeneration::V5e` / `TpuGeneration::V5p` (`viperfish`, `asic_sw::deepsea::pxc::{vfc,vfi}`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VfcV5BundleWord {
    /// Raw 52-byte (`416`-bit) `vfc` / `vfi` VLIW bundle bitstream.
    pub raw_bytes: [u8; 52],
}

const _: () = assert!(size_of::<VfcV5BundleWord>() == 52);

impl VfcV5BundleWord {
    /// Packs an architectural [`TensorCoreVliwBundle`] into a 52-byte (`416`-bit) `V5e`/`V5p` [`VfcV5BundleWord`].
    ///
    /// # Arguments
    ///
    /// * `bundle` - Canonical 64-byte [`TensorCoreVliwBundle`] to compress.
    #[must_use]
    pub fn from_bundle(bundle: &TensorCoreVliwBundle) -> Self {
        let mut raw = [0_u8; 52];
        let pfc = PfcV4BundleWord::from_bundle(bundle);
        raw[..40].copy_from_slice(&pfc.raw_bytes);
        // Bytes 40..52 store full 32-bit scalar immediate, k_tiles, xlu dst/src, and 32-bit sync_flag_id
        raw[40..44].copy_from_slice(&bundle.scalar_slot.imm32.to_le_bytes());
        raw[44..46].copy_from_slice(&bundle.matrix_slot.k_tiles.to_le_bytes());
        raw[46..48].copy_from_slice(&bundle.xunary_slot.dst_vreg.to_le_bytes());
        raw[48..52].copy_from_slice(&bundle.dma_slot.sync_flag_id.to_le_bytes());
        raw[3] = compute_vliw_bundle_crc8(&raw[4..52]);
        Self { raw_bytes: raw }
    }

    /// Unpacks and validates this 52-byte [`VfcV5BundleWord`] into a 64-byte [`TensorCoreVliwBundle`].
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidExecutableFormat`] if the embedded CRC-8 parity byte or opcode is invalid.
    pub fn to_bundle(&self) -> LibTpuResult<TensorCoreVliwBundle> {
        let raw = &self.raw_bytes;
        let expected_crc = compute_vliw_bundle_crc8(&raw[4..52]);
        if raw[3] != expected_crc {
            return Err(LibTpuError::InvalidExecutableFormat {
                reason: format!(
                    "VfcV5BundleWord CRC-8 mismatch: expected {expected_crc:#04x}, got {:#04x}",
                    raw[3]
                ),
            });
        }
        let mut pfc_bytes = [0_u8; 40];
        pfc_bytes.copy_from_slice(&raw[..40]);
        pfc_bytes[3] = compute_vliw_bundle_crc8(&pfc_bytes[4..40]);
        let mut bundle = (PfcV4BundleWord {
            raw_bytes: pfc_bytes,
        })
        .to_bundle()?;
        bundle.scalar_slot.imm32 = u32::from_le_bytes([raw[40], raw[41], raw[42], raw[43]]);
        bundle.matrix_slot.k_tiles = u16::from_le_bytes([raw[44], raw[45]]);
        bundle.xunary_slot.dst_vreg = u16::from_le_bytes([raw[46], raw[47]]);
        bundle.dma_slot.sync_flag_id = u32::from_le_bytes([raw[48], raw[49], raw[50], raw[51]]);
        Ok(bundle)
    }
}

/// 64-byte (`512`-bit) bit-packed `DeepSea` VLIW instruction bundle for `TpuGeneration::V6e` / `TpuGeneration::V7x` (`trillium` / `ironwood`, `asic_sw::deepsea::pxc::{vfk,vfl}`).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VfkV6V7BundleWord {
    /// Raw 64-byte (`512`-bit) `vfk` / `vfl` VLIW bundle bitstream with CRC-8 parity in `raw_bytes[7]`.
    pub raw_bytes: [u8; 64],
}

const _: () = assert!(size_of::<VfkV6V7BundleWord>() == 64);

impl VfkV6V7BundleWord {
    /// Packs an architectural [`TensorCoreVliwBundle`] into a 64-byte (`512`-bit) `V6e`/`V7x` [`VfkV6V7BundleWord`].
    ///
    /// # Arguments
    ///
    /// * `bundle` - Canonical 64-byte [`TensorCoreVliwBundle`] to encode.
    #[must_use]
    pub fn from_bundle(bundle: &TensorCoreVliwBundle) -> Self {
        let mut raw: [u8; 64] = bytemuck::cast(*bundle);
        // Store CRC-8 parity in raw[7] (upper byte of pred_mask when bit 7 parity is enabled)
        raw[7] = 0;
        let crc = compute_vliw_bundle_crc8(&raw);
        raw[7] = crc;
        Self { raw_bytes: raw }
    }

    /// Unpacks and validates this 64-byte [`VfkV6V7BundleWord`] into a 64-byte [`TensorCoreVliwBundle`].
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidExecutableFormat`] if the embedded CRC-8 parity byte or opcode is invalid.
    pub fn to_bundle(&self) -> LibTpuResult<TensorCoreVliwBundle> {
        let mut raw = self.raw_bytes;
        let actual_crc = raw[7];
        raw[7] = 0;
        let expected_crc = compute_vliw_bundle_crc8(&raw);
        if actual_crc != expected_crc {
            return Err(LibTpuError::InvalidExecutableFormat {
                reason: format!(
                    "VfkV6V7BundleWord CRC-8 mismatch: expected {expected_crc:#04x}, got {actual_crc:#04x}"
                ),
            });
        }
        raw[7] = raw[6];
        TensorCoreVliwBundle::decode(&raw).map_err(|err| LibTpuError::InvalidExecutableFormat {
            reason: err.to_string(),
        })
    }
}

impl TensorCoreVliwBundle {
    /// Encodes this [`TensorCoreVliwBundle`] into its generation-specific hardware bitstream
    /// (`40` bytes for `V4`, `52` bytes for `V5e`/`V5p`, `64` bytes for `V6e`/`V7x`).
    ///
    /// # Arguments
    ///
    /// * `generation` - Target [`crate::types::TpuGeneration`] (`V4`, `V5e`, `V5p`, `V6e`, `V7x`).
    #[must_use]
    pub fn encode_generation_bitstream(&self, generation: crate::types::TpuGeneration) -> Vec<u8> {
        match generation {
            crate::types::TpuGeneration::V2
            | crate::types::TpuGeneration::V3
            | crate::types::TpuGeneration::V4
            | crate::types::TpuGeneration::V4Lite => {
                PfcV4BundleWord::from_bundle(self).raw_bytes.to_vec()
            }
            crate::types::TpuGeneration::V5e | crate::types::TpuGeneration::V5p => {
                VfcV5BundleWord::from_bundle(self).raw_bytes.to_vec()
            }
            crate::types::TpuGeneration::V6e | crate::types::TpuGeneration::V7x => {
                VfkV6V7BundleWord::from_bundle(self).raw_bytes.to_vec()
            }
        }
    }

    /// Decodes a generation-specific hardware VLIW bundle bitstream (`40`, `52`, or `64` bytes)
    /// back into a canonical 64-byte [`TensorCoreVliwBundle`], verifying CRC-8 integrity.
    ///
    /// # Arguments
    ///
    /// * `generation` - Target [`crate::types::TpuGeneration`].
    /// * `bytes` - Raw encoded bundle bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidExecutableFormat`] if `bytes.len()` does not match the generation's
    /// bundle size or if CRC-8 / opcode validation fails.
    pub fn decode_generation_bitstream(
        generation: crate::types::TpuGeneration,
        bytes: &[u8],
    ) -> LibTpuResult<Self> {
        match generation {
            crate::types::TpuGeneration::V2
            | crate::types::TpuGeneration::V3
            | crate::types::TpuGeneration::V4
            | crate::types::TpuGeneration::V4Lite => {
                let Ok(arr) = <[u8; 40]>::try_from(bytes) else {
                    return Err(LibTpuError::InvalidExecutableFormat {
                        reason: format!(
                            "Expected 40-byte PfcV4BundleWord bitstream, got {} bytes",
                            bytes.len()
                        ),
                    });
                };
                (PfcV4BundleWord { raw_bytes: arr }).to_bundle()
            }
            crate::types::TpuGeneration::V5e | crate::types::TpuGeneration::V5p => {
                let Ok(arr) = <[u8; 52]>::try_from(bytes) else {
                    return Err(LibTpuError::InvalidExecutableFormat {
                        reason: format!(
                            "Expected 52-byte VfcV5BundleWord bitstream, got {} bytes",
                            bytes.len()
                        ),
                    });
                };
                (VfcV5BundleWord { raw_bytes: arr }).to_bundle()
            }
            crate::types::TpuGeneration::V6e | crate::types::TpuGeneration::V7x => {
                let Ok(arr) = <[u8; 64]>::try_from(bytes) else {
                    return Err(LibTpuError::InvalidExecutableFormat {
                        reason: format!(
                            "Expected 64-byte VfkV6V7BundleWord bitstream, got {} bytes",
                            bytes.len()
                        ),
                    });
                };
                (VfkV6V7BundleWord { raw_bytes: arr }).to_bundle()
            }
        }
    }
}

/// Formats a sequence of [`TensorCoreVliwBundle`]s into a multi-line `DeepSea` LLO assembly listing
/// with cycle annotations (`[c=0000, stall=0]`), predicate guards (`@p0`), and functional-unit mnemonics.
///
/// # Arguments
///
/// * `bundles` - Slice of 64-byte [`TensorCoreVliwBundle`]s to format.
#[must_use]
pub fn format_llo_assembly(bundles: &[TensorCoreVliwBundle]) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let mut cycle = 0_u32;
    for bundle in bundles {
        let stall = u32::from(bundle.stall_cycles());
        cycle = cycle.saturating_add(stall);
        let pred_idx = bundle.header.pred_mask.trailing_zeros().min(15);
        let dis = bundle.disassemble_vliw_bundle();
        let _ = writeln!(out, "[c={cycle:04}, stall={stall}] @p{pred_idx} {dis}");
        cycle = cycle.saturating_add(1);
    }
    out
}

/// Parses a multi-line `DeepSea` LLO assembly listing produced by [`format_llo_assembly`]
/// or [`TensorCoreVliwBundle::disassemble_vliw_bundle`] back into a vector of [`TensorCoreVliwBundle`]s.
///
/// # Arguments
///
/// * `listing` - Multi-line `DeepSea` LLO assembly text.
///
/// # Errors
///
/// Returns [`LibTpuError::CompilationFailed`] if any non-empty, non-comment line fails to assemble.
pub fn parse_llo_assembly(listing: &str) -> LibTpuResult<Vec<TensorCoreVliwBundle>> {
    let mut bundles = Vec::new();
    for raw_line in listing.lines() {
        let trimmed = raw_line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") || trimmed.starts_with('#') {
            continue;
        }
        let pc_pos = trimmed
            .find("[pc=")
            .ok_or_else(|| LibTpuError::CompilationFailed {
                reason: format!("Missing '[pc=' marker in LLO assembly line: '{trimmed}'"),
            })?;
        let canonical = trimmed.get(pc_pos..).unwrap_or(trimmed);
        bundles.push(TensorCoreVliwBundle::assemble_vliw_bundle(canonical)?);
    }
    Ok(bundles)
}

impl TensorCoreVliwBundle {
    /// Decodes a 64-byte architectural VLIW instruction bundle from raw `IMEM` bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - 64-byte raw `IMEM` instruction word.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::DeserializationFailed`] if the vector slot opcode byte is invalid.
    pub fn decode(bytes: &[u8; 64]) -> LibTpuResult<Self> {
        let bundle: Self = bytemuck::pod_read_unaligned(bytes);
        let raw_op = u8::try_from(bundle.vector_slot.opcode).map_err(|_| {
            LibTpuError::DeserializationFailed {
                reason: format!(
                    "Invalid VLIW vector slot opcode {}",
                    bundle.vector_slot.opcode
                ),
            }
        })?;
        let _ = TpuOpCode::from_byte(raw_op)?;
        Ok(bundle)
    }

    /// Returns the 5-bit active functional unit mask (`S0=1`, `M0=2`, `XLU=4`, `D0=8`, `V0=16`).
    #[must_use]
    pub const fn active_slot_mask(&self) -> u8 {
        #[allow(clippy::cast_possible_truncation)]
        let low = (self.header.bundle_flags & 0x1F) as u8;
        low
    }

    /// Returns the hardware stall cycle count (`0..=15`) encoded in bits `8..=11` of `header.bundle_flags`.
    #[must_use]
    pub const fn stall_cycles(&self) -> u8 {
        #[allow(clippy::cast_possible_truncation)]
        let stalls = ((self.header.bundle_flags >> 8) & 0x0F) as u8;
        stalls
    }

    /// Returns the target [`crate::types::TpuGeneration`] ordinal (`2..=7`) encoded in bits `12..=15` of `header.bundle_flags`.
    #[must_use]
    pub const fn generation_ordinal(&self) -> u8 {
        #[allow(clippy::cast_possible_truncation)]
        let gen_ord = ((self.header.bundle_flags >> 12) & 0x0F) as u8;
        if gen_ord == 0 { 5 } else { gen_ord }
    }

    /// Disassembles this 64-byte [`TensorCoreVliwBundle`] into a human-readable `DeepSea` LLO assembly mnemonic string.
    #[must_use]
    pub fn disassemble_vliw_bundle(&self) -> String {
        let pc = self.header.pc_offset;
        let stalls = self.stall_cycles();
        let gen_ord = self.generation_ordinal();
        let v_op = u8::try_from(self.vector_slot.opcode)
            .ok()
            .and_then(|b| TpuOpCode::from_byte(b).ok())
            .map_or_else(|| String::from("UNKNOWN"), |op| format!("{op:?}"));
        let s_str = format!(
            "S0(op={}, $s{} <- $s{}, $s{}, imm={})",
            self.scalar_slot.opcode,
            self.scalar_slot.dst_sreg,
            self.scalar_slot.src0_sreg,
            self.scalar_slot.src1_sreg,
            self.scalar_slot.imm32
        );
        let v_str = format!(
            "V0({v_op}, $v{} <- $v{}, $v{}, $v{})",
            self.vector_slot.dst_vreg,
            self.vector_slot.src0_vreg,
            self.vector_slot.src1_vreg,
            self.vector_slot.src2_vreg
        );
        let m_str = if self.matrix_slot.opcode != 0 {
            format!(
                " | M0(acc=$v{}, lhs=$v{}, rhs=$v{}, k={})",
                self.matrix_slot.acc_vreg,
                self.matrix_slot.lhs_vreg,
                self.matrix_slot.rhs_vreg,
                self.matrix_slot.k_tiles
            )
        } else {
            String::new()
        };
        let x_str = if self.xunary_slot.opcode != 0 {
            format!(
                " | XLU(op={}, $v{} <- $v{})",
                self.xunary_slot.opcode, self.xunary_slot.dst_vreg, self.xunary_slot.src_vreg
            )
        } else {
            String::new()
        };
        let d_str = if self.dma_slot.dma_opcode != 0 {
            format!(
                " | D0(op={}, flag={})",
                self.dma_slot.dma_opcode, self.dma_slot.sync_flag_id
            )
        } else {
            String::new()
        };
        format!(
            "[pc={pc:04} gen=v{gen_ord} stall={stalls}] {{ {s_str} | {v_str}{m_str}{x_str}{d_str} }}"
        )
    }

    /// Parses a mnemonic string emitted by [`Self::disassemble_vliw_bundle`] back into a
    /// 64-byte [`TensorCoreVliwBundle`], verifying round-trip assembler/disassembler fidelity.
    ///
    /// # Arguments
    ///
    /// * `mnemonic_line` - Single-line `DeepSea` LLO assembly string (e.g., `"[pc=0000 gen=v5 stall=0] { S0(...) | V0(...) }"`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::CompilationFailed`] if the assembly syntax or opcode mnemonic is invalid.
    #[allow(clippy::too_many_lines)]
    pub fn assemble_vliw_bundle(mnemonic_line: &str) -> LibTpuResult<Self> {
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
        let pc_offset = rest_pc
            .split_once(' ')
            .and_then(|(s, _)| s.parse::<u32>().ok())
            .unwrap_or(0);
        let gen_ord = extract_between(trimmed, "gen=v", " ")
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(5)
            & 0x0F;
        let stall_cycles = extract_between(trimmed, "stall=", "]")
            .and_then(|s| s.parse::<u16>().ok())
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
            .unwrap_or(pc_offset);

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

        let bundle_flags: u16 = 1
            | (u16::from(is_mxu) << 1)
            | (u16::from(is_xunary) << 2)
            | (u16::from(is_dma) << 3)
            | (1 << 4)
            | (stall_cycles << 8)
            | (gen_ord << 12);

        Ok(Self {
            header: VliwBundleHeader {
                pc_offset,
                bundle_flags,
                pred_mask: 0xFFFF,
            },
            scalar_slot: ScalarAluSlot {
                opcode: s_op,
                dst_sreg: scalar_dst,
                src0_sreg: scalar_lhs,
                src1_sreg: scalar_rhs,
                imm32,
            },
            vector_slot: VectorAluSlot {
                opcode: u16::from(v_opcode as u8),
                element_type: 11,
                lane_mask: 0xFF,
                dst_vreg: vec_out,
                src0_vreg: vec_arg_a,
                src1_vreg: vec_arg_b,
                src2_vreg: vec_arg_c,
            },
            matrix_slot,
            xunary_slot,
            dma_slot,
        })
    }

    /// Converts a logical [`TpuInstructionBundle`] into an architectural 64-byte [`TensorCoreVliwBundle`],
    /// validating that all register slot indices fit within the 8-bit physical register bank (`0..256`).
    ///
    /// # Arguments
    ///
    /// * `bundle` - Logical instruction bundle to encode.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::RegisterIndexOverflow`] if `output_slot` or any operand slot index is `>= 256`.
    pub fn from_ir_bundle(bundle: &TpuInstructionBundle) -> LibTpuResult<Self> {
        const MAX_PHYSICAL_REGISTERS: usize = 255;
        if bundle.output_slot > MAX_PHYSICAL_REGISTERS {
            return Err(LibTpuError::RegisterIndexOverflow {
                slot_index: bundle.output_slot,
                max_registers: MAX_PHYSICAL_REGISTERS,
            });
        }
        for &op_slot in &bundle.operand_slots {
            if op_slot > MAX_PHYSICAL_REGISTERS {
                return Err(LibTpuError::RegisterIndexOverflow {
                    slot_index: op_slot,
                    max_registers: MAX_PHYSICAL_REGISTERS,
                });
            }
        }
        Ok(bundle.encode_architectural_bundle())
    }

    /// Reconstructs a logical [`TpuInstructionBundle`] from this 64-byte [`TensorCoreVliwBundle`],
    /// preserving `vector_slot.element_type`, `matrix_slot`, `xunary_slot`, and `dma_slot` fields.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::DeserializationFailed`] if the functional unit slot opcode is unknown.
    pub fn to_instruction_bundle(&self) -> LibTpuResult<TpuInstructionBundle> {
        let raw_op = if self.matrix_slot.opcode != 0 && self.vector_slot.opcode == 0 {
            TpuOpCode::DotGeneralF32 as u8
        } else if self.xunary_slot.opcode != 0 && self.vector_slot.opcode == 0 {
            self.xunary_slot.opcode
        } else if self.dma_slot.collective_op != 0 && self.vector_slot.opcode == 0 {
            self.dma_slot.collective_op
        } else {
            u8::try_from(self.vector_slot.opcode).map_err(|_| {
                LibTpuError::DeserializationFailed {
                    reason: format!(
                        "Invalid VLIW vector slot opcode {}",
                        self.vector_slot.opcode
                    ),
                }
            })?
        };
        let opcode = TpuOpCode::from_byte(raw_op)?;
        let arity = opcode.arity();
        let mut operand_slots = Vec::with_capacity(arity);
        if self.matrix_slot.opcode != 0 && self.vector_slot.opcode == 0 {
            operand_slots.push(usize::from(self.matrix_slot.lhs_vreg));
            operand_slots.push(usize::from(self.matrix_slot.rhs_vreg));
        } else if self.xunary_slot.opcode != 0 && self.vector_slot.opcode == 0 {
            operand_slots.push(usize::from(self.xunary_slot.src_vreg));
        } else {
            if arity >= 1 {
                operand_slots.push(self.vector_slot.src0_vreg as usize);
            }
            if arity >= 2 {
                operand_slots.push(self.vector_slot.src1_vreg as usize);
            }
            if arity >= 3 {
                operand_slots.push(self.vector_slot.src2_vreg as usize);
            }
        }
        let (input_shapes, output_shape) = if self.vector_slot.element_type != 0 {
            let elem_type =
                crate::types::ElementType::from_raw(i32::from(self.vector_slot.element_type))
                    .unwrap_or(crate::types::ElementType::F32);
            let default_shape = crate::types::Shape {
                element_type: elem_type,
                dims: vec![4],
                dynamic_dims: vec![false],
                layout: crate::types::MemoryLayout::default_tpu_tiled_for_type(1, elem_type),
            };
            (
                vec![default_shape.clone(); operand_slots.len()],
                Some(default_shape),
            )
        } else {
            (Vec::new(), None)
        };
        let mut attrs = InstructionAttrs {
            iota_dim: usize::from(self.scalar_slot.src0_sreg),
            concat_dim: usize::from(self.scalar_slot.src1_sreg),
            input_shapes,
            output_shape,
            ..InstructionAttrs::default()
        };
        if self.xunary_slot.perm_mode != 0 {
            attrs
                .reduce_dims
                .push(usize::from(self.xunary_slot.perm_mode.saturating_sub(1)));
        }
        let output_slot = if self.matrix_slot.opcode != 0 && self.vector_slot.dst_vreg == 0 {
            usize::from(self.matrix_slot.acc_vreg)
        } else if self.xunary_slot.opcode != 0 && self.vector_slot.dst_vreg == 0 {
            usize::from(self.xunary_slot.dst_vreg)
        } else {
            self.vector_slot.dst_vreg as usize
        };
        Ok(TpuInstructionBundle {
            pc_offset: self.header.pc_offset,
            opcode,
            operand_slots,
            output_slot,
            sync_flag_id: self.dma_slot.sync_flag_id,
            attrs,
        })
    }
}

/// Cycle-accurate architectural register & functional-unit hazard scoreboard (`VliwScoreboard`).
///
/// Tracks per-register readiness cycles across `SREG[0..64]`, `PREG[0..16]`, `VREG[0..64]`,
/// `VMASK[0..8]`, `MXU0_ACCUM`, `MXU1_ACCUM`, and `SYNC_FLAGS[0..256]` to compute exact
/// RAW/WAW hazard stall counts during multi-issue VLIW bundle scheduling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VliwScoreboard {
    /// Current schedule cycle counter.
    pub current_cycle: u32,
    /// Cycle at which each scalar register `SREG[0..64]` becomes available for reading.
    pub sreg_ready: [u32; 64],
    /// Cycle at which each predicate register `PREG[0..16]` becomes available for reading.
    pub preg_ready: [u32; 16],
    /// Cycle at which each vector register `VREG[0..64]` becomes available for reading.
    pub vreg_ready: [u32; 64],
    /// Cycle at which each vector mask register `VMASK[0..8]` becomes available for reading.
    pub vmask_ready: [u32; 8],
    /// Cycle at which `MXU0` and `MXU1` systolic accumulators become available.
    pub mxu_accum_ready: [u32; 2],
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
            sync_flag_ready: vec![0; 256],
        }
    }

    /// Computes the number of stall cycles (`0..=15`) required before `bundle` can issue
    /// without RAW/WAW hazards, advances `self.current_cycle`, and records output register
    /// readiness using `latency`.
    ///
    /// # Arguments
    ///
    /// * `bundle` - Logical [`TpuInstructionBundle`] being scheduled.
    /// * `latency` - Functional unit latency in cycles (`1..=16`).
    #[allow(clippy::indexing_slicing)]
    pub fn record_issue(&mut self, bundle: &TpuInstructionBundle, latency: u8) -> u8 {
        let mut required_cycle = self.current_cycle;
        for &src_slot in &bundle.operand_slots {
            required_cycle = required_cycle.max(self.vreg_ready[src_slot & 63]);
        }
        required_cycle = required_cycle.max(self.vreg_ready[bundle.output_slot & 63]);
        if bundle.opcode == TpuOpCode::DotGeneralF32 {
            required_cycle = required_cycle.max(self.mxu_accum_ready[0]);
        }
        if bundle.opcode.has_side_effects() {
            required_cycle =
                required_cycle.max(self.sync_flag_ready[(bundle.sync_flag_id as usize) & 255]);
        }
        let stall_u32 = required_cycle.saturating_sub(self.current_cycle).min(15);
        #[allow(clippy::cast_possible_truncation)]
        let stall_cycles = stall_u32 as u8;
        self.current_cycle = required_cycle.saturating_add(1);
        let ready_at = self.current_cycle.saturating_add(u32::from(latency));
        self.vreg_ready[bundle.output_slot & 63] = ready_at;
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
    /// Non-contracting row dimension (`M`) tile count in `MXU` sublanes (`128` or `256`).
    pub m_tiles: u32,
    /// Contracting dimension (`K`) tile count in `MXU` lanes (`128` or `256`).
    pub k_tiles: u32,
    /// Non-contracting column dimension (`N`) tile count in `MXU` lanes (`128` or `256`).
    pub n_tiles: u32,
    /// Ping-buffer byte offset in `VMEM` scratchpad.
    pub ping_vmem_offset: u64,
    /// Pong-buffer byte offset in `VMEM` scratchpad for overlapped `D0` DMA prefetching.
    pub pong_vmem_offset: u64,
    /// Architectural `SyncFlag` ID used to synchronize `HBM -> VMEM` prefetch completion.
    pub prefetch_sync_flag_id: u32,
}

impl PolyhedralTileSchedule {
    /// Computes the polyhedral tile schedule and double-buffered `VMEM` ping/pong offsets
    /// for a tensor shape on `generation`.
    ///
    /// # Arguments
    ///
    /// * `shape` - Logical tensor [`crate::types::Shape`] to tile.
    /// * `generation` - Target Cloud TPU generation (`MXU` `128x128` vs `256x256`).
    /// * `base_vmem_offset` - Starting byte offset in `VMEM`.
    /// * `sync_flag_id` - Base `SyncFlag` ID for DMA completion.
    #[must_use]
    pub fn from_shape(
        shape: &crate::types::Shape,
        generation: crate::types::TpuGeneration,
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
    /// Stage 1 (`SteadyState`): Simultaneously prefetches tile `k + 1` into `VMEM_pong` (`D0`),
    /// executes `MXU` systolic matmul on tile `k` (`M0`), and runs `VPU`/`XLU` epilogue (`V0`/`X0`).
    SteadyState,
    /// Stage 2 (`Epilogue`): Drains final `MXU` accumulator tiles and writes results `VMEM -> HBM` (`D0`).
    Epilogue,
}

/// Software-pipelined schedule produced by [`ModuloPipelineScheduler`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModuloPipelinePlan {
    /// Initiation interval (`II`) in hardware clock cycles between successive tile iterations.
    pub initiation_interval_cycles: u32,
    /// Total number of software-pipelined loop iterations (`k_tiles`).
    pub total_iterations: u32,
    /// Ordered `(ModuloPipelineStage, TensorCoreVliwBundle)` sequence across prologue, steady-state, and epilogue.
    pub staged_bundles: Vec<(ModuloPipelineStage, TensorCoreVliwBundle)>,
}

/// Swing Modulo Software Pipeline Scheduler (`ModuloPipelineScheduler`) overlapping `D0` DMA
/// prefetching, `M0` systolic array execution, and `V0`/`X0` vector epilogues across `ping`/`pong` VMEM buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModuloPipelineScheduler {
    /// Target Cloud TPU generation.
    pub generation: crate::types::TpuGeneration,
}

impl ModuloPipelineScheduler {
    /// Creates a new [`ModuloPipelineScheduler`] for `generation`.
    ///
    /// # Arguments
    ///
    /// * `generation` - Target Cloud TPU generation (`V4`, `V5e`, `V5p`, `V6e`, `V7x`).
    #[must_use]
    pub const fn new(generation: crate::types::TpuGeneration) -> Self {
        Self { generation }
    }

    /// Generates a 3-stage software-pipelined [`ModuloPipelinePlan`] (`Prologue`, `SteadyState`, `Epilogue`)
    /// for a tiled contraction bundle using `tile_schedule`.
    ///
    /// # Arguments
    ///
    /// * `contraction_bundle` - Logical contraction or loop body [`TpuInstructionBundle`].
    /// * `tile_schedule` - [`PolyhedralTileSchedule`] specifying tile counts and `ping`/`pong` VMEM offsets.
    #[must_use]
    pub fn schedule_tile_loop(
        &self,
        contraction_bundle: &TpuInstructionBundle,
        tile_schedule: &PolyhedralTileSchedule,
    ) -> ModuloPipelinePlan {
        let llo = LloVliwScheduler::new(self.generation);
        let ii = u32::from(llo.instruction_latency_cycles(contraction_bundle.opcode));
        let k_iters = tile_schedule.k_tiles.max(1);
        let mut staged = Vec::with_capacity((k_iters as usize) + 2);

        // Stage 0: Prologue — async HBM -> VMEM_ping DMA prefetch for tile k = 0
        let mut prologue = contraction_bundle.encode_for_generation(self.generation, 0);
        prologue.header.pc_offset = contraction_bundle.pc_offset;
        prologue.header.bundle_flags |= 1 << 3;
        #[allow(clippy::cast_possible_truncation)]
        {
            prologue.scalar_slot.imm32 = tile_schedule.ping_vmem_offset as u32;
        }
        prologue.matrix_slot.opcode = 0;
        prologue.dma_slot.dma_opcode = 1;
        prologue.dma_slot.src_space = 0;
        prologue.dma_slot.dst_space = 1;
        prologue.dma_slot.sync_flag_id = tile_schedule.prefetch_sync_flag_id;
        staged.push((ModuloPipelineStage::Prologue, prologue));

        // Stage 1: SteadyState — overlap D0 prefetch (ping/pong), M0 MXU compute, and V0/X0 epilogue
        for step in 0..k_iters {
            let mut steady = contraction_bundle.encode_for_generation(self.generation, 1);
            steady.header.pc_offset = contraction_bundle
                .pc_offset
                .saturating_add(step.saturating_add(1));
            steady.header.bundle_flags |= (1 << 1) | (1 << 3);
            let next_vmem = if (step & 1) == 0 {
                tile_schedule.pong_vmem_offset
            } else {
                tile_schedule.ping_vmem_offset
            };
            #[allow(clippy::cast_possible_truncation)]
            {
                steady.scalar_slot.imm32 = next_vmem as u32;
            }
            steady.matrix_slot.opcode = 1;
            steady.matrix_slot.k_tiles = u16::try_from(k_iters).unwrap_or(1);
            steady.dma_slot.dma_opcode = 1;
            steady.dma_slot.src_space = 0;
            steady.dma_slot.dst_space = 1;
            steady.dma_slot.sync_flag_id =
                tile_schedule.prefetch_sync_flag_id.saturating_add(step & 1);
            staged.push((ModuloPipelineStage::SteadyState, steady));
        }

        // Stage 2: Epilogue — drain accumulator through V0 and issue VMEM -> HBM writeback DMA
        let mut epilogue = contraction_bundle.encode_for_generation(self.generation, 2);
        epilogue.header.pc_offset = contraction_bundle
            .pc_offset
            .saturating_add(k_iters.saturating_add(1));
        epilogue.header.bundle_flags |= 1 << 3;
        epilogue.matrix_slot.opcode = 0;
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
}

/// Latency-aware `DeepSea` VLIW instruction list scheduler (`LloVliwScheduler`).
///
/// Models functional unit latencies (Scalar = 1 cycle, Vector ALU = 2 cycles,
/// EUP/XLU = 4 cycles, MXU systolic array = 8 cycles, Async DMA = `SyncFlag` barrier)
/// and packs independent instructions across functional unit slots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LloVliwScheduler {
    /// Target Cloud TPU hardware generation.
    pub generation: crate::types::TpuGeneration,
}

impl LloVliwScheduler {
    /// Creates a new [`LloVliwScheduler`] for the specified [`crate::types::TpuGeneration`].
    ///
    /// # Arguments
    ///
    /// * `generation` - Target Cloud TPU generation (`V4`, `V5e`, `V5p`, `V6e`, `V7x`).
    #[must_use]
    pub const fn new(generation: crate::types::TpuGeneration) -> Self {
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
                crate::types::TpuGeneration::V6e | crate::types::TpuGeneration::V7x => 12,
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

    /// Schedules a sequence of logical [`TpuInstructionBundle`]s into hazard-annotated,
    /// generation-parameterized 64-byte [`TensorCoreVliwBundle`]s using [`VliwScoreboard`].
    ///
    /// # Arguments
    ///
    /// * `bundles` - Slice of logical instruction bundles to schedule.
    #[must_use]
    pub fn schedule_bundles(&self, bundles: &[TpuInstructionBundle]) -> Vec<TensorCoreVliwBundle> {
        let mut scoreboard = VliwScoreboard::new();
        let mut out = Vec::with_capacity(bundles.len());

        for bundle in bundles {
            let latency = self.instruction_latency_cycles(bundle.opcode);
            let stall_cycles = scoreboard.record_issue(bundle, latency);
            out.push(bundle.encode_for_generation(self.generation, stall_cycles));
        }

        out
    }

    /// Schedules and co-issues independent [`TpuInstructionBundle`]s across functional unit slots
    /// (`S0`, `V0`, `M0`, `XLU`, `D0`) into consolidated 64-byte [`TensorCoreVliwBundle`]s when
    /// no RAW or WAW register hazard exists between adjacent instructions.
    ///
    /// # Arguments
    ///
    /// * `bundles` - Slice of logical instruction bundles to schedule and pack.
    #[must_use]
    pub fn pack_independent_bundles(
        &self,
        bundles: &[TpuInstructionBundle],
    ) -> Vec<TensorCoreVliwBundle> {
        let scheduled = self.schedule_bundles(bundles);
        let mut packed = Vec::with_capacity(scheduled.len());
        let mut idx = 0_usize;
        while let Some(&first) = scheduled.get(idx) {
            let mut merged = first;
            if let Some(&second) = scheduled.get(idx + 1) {
                let first_dst = first.vector_slot.dst_vreg;
                let second_reads_first = second.vector_slot.src0_vreg == first_dst;
                let can_merge_mxu = second.matrix_slot.opcode != 0 && first.matrix_slot.opcode == 0;
                let can_merge_xlu = second.xunary_slot.opcode != 0 && first.xunary_slot.opcode == 0;
                let can_merge_dma =
                    second.dma_slot.dma_opcode != 0 && first.dma_slot.dma_opcode == 0;
                if !second_reads_first && (can_merge_mxu || can_merge_xlu || can_merge_dma) {
                    if can_merge_mxu {
                        merged.matrix_slot = second.matrix_slot;
                        merged.header.bundle_flags |= 1 << 1;
                    }
                    if can_merge_xlu {
                        merged.xunary_slot = second.xunary_slot;
                        merged.header.bundle_flags |= 1 << 2;
                    }
                    if can_merge_dma {
                        merged.dma_slot = second.dma_slot;
                        merged.header.bundle_flags |= 1 << 3;
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

    /// Encodes this [`TpuInstructionBundle`] into a 64-byte architectural [`TensorCoreVliwBundle`].
    #[must_use]
    pub fn encode_architectural_bundle(&self) -> TensorCoreVliwBundle {
        let tpu_gen = crate::config::global_init_config().generation;
        self.encode_for_generation(tpu_gen, 0)
    }

    /// Encodes this [`TpuInstructionBundle`] into a 64-byte architectural [`TensorCoreVliwBundle`]
    /// parameterized by target [`crate::types::TpuGeneration`] and hardware `stall_cycles` (`0..=15`).
    ///
    /// # Arguments
    ///
    /// * `generation` - Target Cloud TPU generation (`V4`, `V5e`, `V5p`, `V6e`, `V7x`).
    /// * `stall_cycles` - Number of inter-bundle hazard stall cycles (`0..=15`).
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn encode_for_generation(
        &self,
        generation: crate::types::TpuGeneration,
        stall_cycles: u8,
    ) -> TensorCoreVliwBundle {
        let s0 = u32::try_from(self.operand_slots.first().copied().unwrap_or(0)).unwrap_or(0);
        let s1 = u32::try_from(self.operand_slots.get(1).copied().unwrap_or(0)).unwrap_or(0);
        let s2 = u32::try_from(self.operand_slots.get(2).copied().unwrap_or(0)).unwrap_or(0);
        let dst = u32::try_from(self.output_slot).unwrap_or(0);
        let et_raw = self.attrs.output_shape.as_ref().map_or(0_u8, |sh| {
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
            || matches!(
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
        let gen_ord = u16::try_from(i32::from(generation)).unwrap_or(5) & 0x0F;
        let stalls_u16 = u16::from(stall_cycles & 0x0F);
        let bundle_flags: u16 = 1
            | (u16::from(is_mxu) << 1)
            | (u16::from(is_xunary) << 2)
            | (u16::from(is_collective) << 3)
            | (1 << 4)
            | (stalls_u16 << 8)
            | (gen_ord << 12);

        let k_tiles = match generation {
            crate::types::TpuGeneration::V6e | crate::types::TpuGeneration::V7x => 2,
            _ => 1,
        };

        TensorCoreVliwBundle {
            header: VliwBundleHeader {
                pc_offset: self.pc_offset,
                bundle_flags,
                pred_mask: 0xFFFF,
            },
            scalar_slot: ScalarAluSlot {
                opcode: 1,
                dst_sreg: 0,
                src0_sreg: iota_u8,
                src1_sreg: concat_u8,
                imm32: self.pc_offset,
            },
            vector_slot: VectorAluSlot {
                opcode: u16::from(self.opcode as u8),
                element_type: et_raw,
                lane_mask: 0xFF,
                dst_vreg: dst,
                src0_vreg: s0,
                src1_vreg: s1,
                src2_vreg: s2,
            },
            matrix_slot: MatrixMxuSlot {
                opcode: u8::from(is_mxu),
                precision_mode: 0,
                flags: u16::from(is_mxu),
                acc_vreg: u16::try_from(dst).unwrap_or(0),
                lhs_vreg: u16::try_from(s0).unwrap_or(0),
                rhs_vreg: u16::try_from(s1).unwrap_or(0),
                k_tiles,
            },
            xunary_slot: ExtendedUnarySlot {
                opcode: if is_xunary { self.opcode as u8 } else { 0 },
                perm_mode: red_mode,
                dst_vreg: u16::try_from(dst).unwrap_or(0),
                src_vreg: u16::try_from(s0).unwrap_or(0),
                imm16: 0,
            },
            dma_slot: AsyncDmaSlot {
                dma_opcode: u8::from(is_collective),
                src_space: 1,
                dst_space: 0,
                collective_op: if is_collective { self.opcode as u8 } else { 0 },
                sync_flag_id: self.sync_flag_id,
            },
        }
    }
}

#[cfg(test)]
#[allow(clippy::indexing_slicing, clippy::too_many_lines)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::types::TpuGeneration;

    #[test]
    fn test_per_generation_vliw_bitstreams_and_llo_assembly_roundtrip() {
        let mut bundle = TpuInstructionBundle {
            pc_offset: 64,
            opcode: TpuOpCode::DotGeneralF32,
            operand_slots: vec![1, 2],
            output_slot: 3,
            sync_flag_id: 7,
            attrs: InstructionAttrs::default(),
        }
        .encode_for_generation(TpuGeneration::V5p, 2);
        bundle.xunary_slot.opcode = TpuOpCode::ExpF32 as u8;
        bundle.dma_slot.dma_opcode = 1;

        for (tpu_gen, expected_len) in [
            (TpuGeneration::V2, 40),
            (TpuGeneration::V3, 40),
            (TpuGeneration::V4, 40),
            (TpuGeneration::V4Lite, 40),
            (TpuGeneration::V5e, 52),
            (TpuGeneration::V5p, 52),
            (TpuGeneration::V6e, 64),
            (TpuGeneration::V7x, 64),
        ] {
            let encoded = bundle.encode_generation_bitstream(tpu_gen);
            assert_eq!(encoded.len(), expected_len);
            let decoded_res = TensorCoreVliwBundle::decode_generation_bitstream(tpu_gen, &encoded);
            assert!(decoded_res.is_ok());
            if let Ok(decoded) = decoded_res {
                assert_eq!(decoded.header.pc_offset, bundle.header.pc_offset);
                assert_eq!(decoded.vector_slot.opcode, bundle.vector_slot.opcode);
                assert_eq!(decoded.vector_slot.dst_vreg, bundle.vector_slot.dst_vreg);
                assert_eq!(decoded.matrix_slot.opcode, bundle.matrix_slot.opcode);
                assert_eq!(decoded.xunary_slot.opcode, bundle.xunary_slot.opcode);
                assert_eq!(decoded.dma_slot.dma_opcode, bundle.dma_slot.dma_opcode);
                assert_eq!(decoded.dma_slot.sync_flag_id, bundle.dma_slot.sync_flag_id);
            }
            // Corrupt CRC byte and verify error
            let mut corrupted = encoded.clone();
            if expected_len == 64 {
                corrupted[7] ^= 0xFF;
            } else {
                corrupted[3] ^= 0xFF;
            }
            assert!(
                TensorCoreVliwBundle::decode_generation_bitstream(tpu_gen, &corrupted).is_err()
            );
            // Wrong length error
            assert!(
                TensorCoreVliwBundle::decode_generation_bitstream(tpu_gen, &[0_u8; 12]).is_err()
            );
        }

        // Test PfcV4BundleWord with xunary_slot.opcode == 0 and invalid vector opcode
        let plain_bundle = TpuInstructionBundle {
            pc_offset: 0,
            opcode: TpuOpCode::AddF32,
            operand_slots: vec![0, 1],
            output_slot: 2,
            sync_flag_id: 1,
            attrs: InstructionAttrs::default(),
        }
        .encode_for_generation(TpuGeneration::V4, 0);
        let pfc_plain = PfcV4BundleWord::from_bundle(&plain_bundle);
        assert_eq!(pfc_plain.clone(), pfc_plain);
        assert_ne!(format!("{pfc_plain:?}"), "");
        let unpacked_plain = pfc_plain.to_bundle();
        assert!(unpacked_plain.is_ok());
        if let Ok(up) = unpacked_plain {
            assert_eq!(up.xunary_slot.dst_vreg, 0);
            assert_eq!(up.xunary_slot.src_vreg, 0);
        }

        // Invalid vector opcode (> 255 and unknown byte 250) in PfcV4BundleWord
        let mut bad_pfc = pfc_plain;
        bad_pfc.raw_bytes[14] = 0x00;
        bad_pfc.raw_bytes[15] = 0x02; // 512 > u8::MAX
        bad_pfc.raw_bytes[3] = compute_vliw_bundle_crc8(&bad_pfc.raw_bytes[4..40]);
        assert!(bad_pfc.to_bundle().is_err());

        bad_pfc.raw_bytes[14] = 250; // > 100 unknown TpuOpCode
        bad_pfc.raw_bytes[15] = 0;
        bad_pfc.raw_bytes[3] = compute_vliw_bundle_crc8(&bad_pfc.raw_bytes[4..40]);
        assert!(bad_pfc.to_bundle().is_err());

        // Invalid vector opcode in VfkV6V7BundleWord
        let vfk = VfkV6V7BundleWord::from_bundle(&plain_bundle);
        assert_eq!(vfk.clone(), vfk);
        assert_ne!(format!("{vfk:?}"), "");
        let vfc = VfcV5BundleWord::from_bundle(&plain_bundle);
        assert_eq!(vfc.clone(), vfc);
        assert_ne!(format!("{vfc:?}"), "");
        let mut bad_vfk = vfk;
        bad_vfk.raw_bytes[16] = 250;
        bad_vfk.raw_bytes[17] = 0;
        bad_vfk.raw_bytes[7] = 0;
        bad_vfk.raw_bytes[7] = compute_vliw_bundle_crc8(&bad_vfk.raw_bytes);
        assert!(bad_vfk.to_bundle().is_err());

        // Test format_llo_assembly & parse_llo_assembly
        let listing = format_llo_assembly(&[plain_bundle, bundle]);
        assert!(listing.contains("[c=0000, stall=0] @p0 [pc="));
        let with_comments = format!("// header comment\n# hash comment\n\n{listing}");
        let parsed_bundles = parse_llo_assembly(&with_comments);
        assert!(parsed_bundles.is_ok());
        if let Ok(pb) = parsed_bundles {
            assert_eq!(pb.len(), 2);
            assert_eq!(pb[0].vector_slot.opcode, plain_bundle.vector_slot.opcode);
            assert_eq!(pb[1].vector_slot.opcode, bundle.vector_slot.opcode);
        }
        assert!(parse_llo_assembly("invalid line without pc marker").is_err());

        // Test ModuloPipelineScheduler across Prologue, SteadyState, and Epilogue stages
        let tile_shape = crate::types::Shape {
            element_type: crate::types::ElementType::F32,
            dims: vec![1, 512, 512],
            dynamic_dims: vec![false, false, false],
            layout: crate::types::MemoryLayout::default_tpu_tiled(3),
        };
        let tile_sched =
            PolyhedralTileSchedule::from_shape(&tile_shape, TpuGeneration::V6e, 0x1000, 5);
        let mod_sched = ModuloPipelineScheduler::new(TpuGeneration::V6e);
        assert_eq!(mod_sched.clone(), mod_sched);
        assert_ne!(format!("{mod_sched:?}"), "");
        let dot_bundle = TpuInstructionBundle {
            pc_offset: 10,
            opcode: TpuOpCode::DotGeneralF32,
            operand_slots: vec![0, 1],
            output_slot: 2,
            sync_flag_id: 5,
            attrs: InstructionAttrs::default(),
        };
        let mod_plan = mod_sched.schedule_tile_loop(&dot_bundle, &tile_sched);
        assert_eq!(mod_plan.clone(), mod_plan);
        assert_ne!(format!("{mod_plan:?}"), "");
        assert_eq!(mod_plan.total_iterations, 2);
        assert_eq!(mod_plan.staged_bundles.len(), 4);
        assert_eq!(mod_plan.staged_bundles[0].0, ModuloPipelineStage::Prologue);
        assert_eq!(
            mod_plan.staged_bundles[1].0,
            ModuloPipelineStage::SteadyState
        );
        assert_eq!(mod_plan.staged_bundles[3].0, ModuloPipelineStage::Epilogue);
        assert_eq!(
            mod_plan.staged_bundles[0].0.clone(),
            ModuloPipelineStage::Prologue
        );
        assert_ne!(format!("{:?}", mod_plan.staged_bundles[0].0), "");

        // Test HardwareRegisterFileConfig and LinearScanRegisterAllocator with and without VMEM spills
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
            let cfg = crate::compiler::HardwareRegisterFileConfig::for_generation(g);
            assert_eq!(cfg.clone(), cfg);
            assert_ne!(format!("{cfg:?}"), "");
        }
        let alloc_tight =
            crate::compiler::LinearScanRegisterAllocator::with_max_vregs(TpuGeneration::V4, 2);
        assert_eq!(alloc_tight.config.num_vregs, 2);
        assert_eq!(alloc_tight.clone(), alloc_tight);
        assert_ne!(format!("{alloc_tight:?}"), "");
        let intervals = vec![
            crate::compiler::RegisterLivenessInterval {
                slot_index: 0,
                first_def_pc: 0,
                last_use_pc: 10,
                size_bytes: 256,
                assigned_vmem_offset: 0,
            },
            crate::compiler::RegisterLivenessInterval {
                slot_index: 1,
                first_def_pc: 1,
                last_use_pc: 12,
                size_bytes: 256,
                assigned_vmem_offset: 256,
            },
            crate::compiler::RegisterLivenessInterval {
                slot_index: 2,
                first_def_pc: 2,
                last_use_pc: 5,
                size_bytes: 256,
                assigned_vmem_offset: 512,
            },
            crate::compiler::RegisterLivenessInterval {
                slot_index: 3,
                first_def_pc: 8,
                last_use_pc: 14,
                size_bytes: 256,
                assigned_vmem_offset: 768,
            },
        ];
        let spill_rep = alloc_tight.allocate(&intervals, 1024);
        assert_eq!(spill_rep.clone(), spill_rep);
        assert_ne!(format!("{spill_rep:?}"), "");
        assert_eq!(spill_rep.spill_actions.len(), 2);
        assert!(spill_rep.total_spill_vmem_bytes >= 256);
        assert_eq!(
            spill_rep.spill_actions[0].clone(),
            spill_rep.spill_actions[0]
        );
        assert_ne!(format!("{:?}", spill_rep.spill_actions[0]), "");
    }
}

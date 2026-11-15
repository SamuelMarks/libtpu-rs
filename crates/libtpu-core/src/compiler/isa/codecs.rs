//! Generation-exact `TensorCore` binary VLIW bundle encoders, decoders, protobuf lowerers,
//! and symbolic disassemblers (`jfc`, `dfc`, `pfc`, `vfc`, `gfc`, `iwc`).
//!
//! Uses `riegeli_highwayhash64` 8-bit bundle integrity tags and `IsaBitfieldSpecTable` slot layouts,
//! with zero synthetic `pc_offset` fields and zero `0x5A` CRC-8 placeholders.

use super::super::llo::LloBundle;
use super::super::opcode::TpuOpCode;
use super::super::vliw::{
    AsyncDmaSlot, ExtendedUnarySlot, MatrixMxuSlot, ScalarAluSlot, VectorAluSlot,
};
use crate::error::{LibTpuError, LibTpuResult};
use crate::proto::{
    GhostliteTensorCoreBundleProto, GhostliteTensorCoreProgramProto, IronwoodTensorCoreBundleProto,
    IronwoodTensorCoreProgramProto, JellyfishIsaBundleProto, JellyfishIsaProgramProto,
    PufferfishTensorCoreBundleProto, PufferfishTensorCoreProgramProto,
    ViperfishTensorCoreBundleProto, ViperfishTensorCoreProgramProto, riegeli_highwayhash64,
};
use crate::types::TpuGeneration;

/// Computes the 8-bit `HighwayHash64` bundle integrity tag over `raw` with byte `7` zeroed.
///
/// # Arguments
///
/// * `raw` - Generation-encoded bundle byte slice (`32`, `40`, `52`, or `64` bytes).
#[must_use]
pub fn compute_bundle_highway_tag8(raw: &[u8]) -> u8 {
    let mut scratch = [0_u8; 64];
    let n = raw.len().min(64);
    scratch
        .iter_mut()
        .zip(raw.iter())
        .for_each(|(d, s)| *d = *s);
    scratch[7] = 0;
    let h = riegeli_highwayhash64(scratch.get(..n).unwrap_or(&scratch));
    #[allow(clippy::cast_possible_truncation)]
    let tag = ((h ^ (h >> 32)) & 0xFF) as u8;
    tag
}

/// Common trait implemented by all six Cloud TPU generation VLIW binary ISA codecs (`GenerationIsaCodec`).
pub trait GenerationIsaCodec {
    /// Short ISA codename (`"jfc"`, `"dfc"`, `"pfc"`, `"vfc"`, `"gfc"`, `"iwc"`).
    const ISA_CODE: &'static str;
    /// Single-byte hardware bundle header magic (`0x22`, `0x33`, `0x44`, `0x55`, `0x66`, `0x77`).
    const HEADER_MAGIC: u8;
    /// Fixed byte width of one encoded VLIW bundle (`32`, `40`, `52`, or `64` bytes).
    const BUNDLE_BYTES: usize;

    /// Encodes a single [`LloBundle`] into `Self::BUNDLE_BYTES` raw hardware bytes.
    ///
    /// # Arguments
    ///
    /// * `bundle` - Architectural [`LloBundle`] to encode.
    fn encode_bundle(bundle: &LloBundle) -> Vec<u8>;

    /// Decodes a single [`LloBundle`] from `Self::BUNDLE_BYTES` raw hardware bytes, validating
    /// `Self::HEADER_MAGIC` and [`compute_bundle_highway_tag8`].
    ///
    /// # Arguments
    ///
    /// * `bytes` - Raw bundle slice of length `Self::BUNDLE_BYTES`.
    /// * `bundle_offset` - Byte offset within `IMEM` for error reporting.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::VliwIsaCodecError`] if length, magic, or `HighwayHash64` tag validation fails.
    fn decode_bundle(bytes: &[u8], bundle_offset: usize) -> LibTpuResult<LloBundle>;

    /// Disassembles a single encoded bundle at positional `bundle_pc` into canonical LLO text.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Raw bundle slice of length `Self::BUNDLE_BYTES`.
    /// * `bundle_pc` - Positional bundle index in `IMEM`.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::VliwIsaCodecError`] if `bytes` fails [`Self::decode_bundle`].
    fn disassemble_bundle(bytes: &[u8], bundle_pc: usize) -> LibTpuResult<String> {
        let b = Self::decode_bundle(bytes, bundle_pc.saturating_mul(Self::BUNDLE_BYTES))?;
        Ok(format_generation_bundle_line(Self::ISA_CODE, bundle_pc, &b))
    }

    /// Encodes a slice of [`LloBundle`]s into a contiguous `IMEM` bitstream.
    ///
    /// # Arguments
    ///
    /// * `bundles` - Slice of [`LloBundle`]s.
    #[must_use]
    fn encode_stream(bundles: &[LloBundle]) -> Vec<u8> {
        let mut out = Vec::with_capacity(bundles.len().saturating_mul(Self::BUNDLE_BYTES));
        for b in bundles {
            out.extend_from_slice(&Self::encode_bundle(b));
        }
        out
    }

    /// Decodes a contiguous `IMEM` bitstream into [`LloBundle`]s.
    ///
    /// # Arguments
    ///
    /// * `stream` - Raw `IMEM` byte slice (must be a non-empty multiple of `Self::BUNDLE_BYTES`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::VliwIsaCodecError`] if `stream` is empty, misaligned, or corrupt.
    fn decode_stream(stream: &[u8]) -> LibTpuResult<Vec<LloBundle>> {
        if stream.is_empty() || (stream.len() % Self::BUNDLE_BYTES != 0) {
            return Err(LibTpuError::VliwIsaCodecError {
                generation_code: Self::ISA_CODE,
                bundle_offset: 0,
                reason: format!(
                    "IMEM stream length {} is not a positive multiple of {} bytes",
                    stream.len(),
                    Self::BUNDLE_BYTES
                ),
            });
        }
        let mut out = Vec::with_capacity(stream.len() / Self::BUNDLE_BYTES);
        for (idx, chunk) in stream.chunks_exact(Self::BUNDLE_BYTES).enumerate() {
            out.push(Self::decode_bundle(chunk, idx * Self::BUNDLE_BYTES)?);
        }
        Ok(out)
    }

    /// Disassembles a contiguous `IMEM` bitstream into a multi-line symbolic listing.
    ///
    /// # Arguments
    ///
    /// * `stream` - Raw `IMEM` byte slice.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::VliwIsaCodecError`] if `stream` fails decoding.
    fn disassemble_stream(stream: &[u8]) -> LibTpuResult<String> {
        let bundles = Self::decode_stream(stream)?;
        let mut lines = Vec::with_capacity(bundles.len());
        for (pc, b) in bundles.iter().enumerate() {
            lines.push(format_generation_bundle_line(Self::ISA_CODE, pc, b));
        }
        Ok(lines.join(
            "
",
        ))
    }
}

/// Formats a single [`LloBundle`] at positional `pc` with its generation `isa_code`.
///
/// # Arguments
///
/// * `isa_code` - Generation ISA mnemonic (`"jfc"`, `"dfc"`, `"pfc"`, `"vfc"`, `"gfc"`, `"iwc"`).
/// * `pc` - Positional bundle index (`0..`).
/// * `b` - Decoded [`LloBundle`].
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn format_generation_bundle_line(isa_code: &str, pc: usize, b: &LloBundle) -> String {
    let v0 = &b.vector_slots[0];
    let s0 = &b.scalar_slots[0];
    let m0 = &b.matrix_slots[0];
    let x0 = &b.xlu_slots[0];
    let v_mnem = TpuOpCode::from_byte(v0.opcode as u8)
        .map_or_else(|_| String::from("Nop"), |op| format!("{op:?}"));
    let mut parts = vec![
        format!(
            "S0(op={}, $s{} <- $s{}, $s{}, imm={})",
            s0.opcode, s0.dst_sreg, s0.src0_sreg, s0.src1_sreg, s0.imm32
        ),
        format!(
            "V0({v_mnem}, $v{} <- $v{}, $v{}, $v{})",
            v0.dst_vreg, v0.src0_vreg, v0.src1_vreg, v0.src2_vreg
        ),
    ];
    if m0.opcode != 0 {
        parts.push(format!(
            "M0(prec={}, $gmr{} <- $v{} * $v{}, k={})",
            m0.precision_mode, m0.acc_vreg, m0.lhs_vreg, m0.rhs_vreg, m0.k_tiles
        ));
    }
    if x0.opcode != 0 {
        parts.push(format!(
            "X0(op={}, mode={}, $v{} <- $v{}, imm={})",
            x0.opcode, x0.perm_mode, x0.dst_vreg, x0.src_vreg, x0.imm16
        ));
    }
    if b.dma_slot.dma_opcode != 0 {
        parts.push(format!(
            "D0(op={}, {}->{}, sf={})",
            b.dma_slot.dma_opcode,
            b.dma_slot.src_space,
            b.dma_slot.dst_space,
            b.dma_slot.sync_flag_id
        ));
    }
    format!(
        "[pc={pc:04x} isa={isa_code} stall={} pred={:#06x}] {{ {} }}",
        b.stall_cycles,
        b.pred_mask,
        parts.join(" | ")
    )
}

/// Validates bundle slice length, `expected_magic`, and `HighwayHash64` tag byte `7`.
///
/// # Arguments
///
/// * `isa_code` - ISA codename for diagnostics.
/// * `expected_magic` - Expected byte `0` magic tag.
/// * `expected_len` - Expected bundle byte width.
/// * `bytes` - Candidate bundle slice.
/// * `offset` - Byte offset in `IMEM`.
fn verify_header_and_tag(
    isa_code: &'static str,
    expected_magic: u8,
    expected_len: usize,
    bytes: &[u8],
    offset: usize,
) -> LibTpuResult<()> {
    if bytes.len() != expected_len {
        return Err(LibTpuError::VliwIsaCodecError {
            generation_code: isa_code,
            bundle_offset: offset,
            reason: format!("expected {expected_len}-byte bundle, got {}", bytes.len()),
        });
    }
    let got_magic = bytes.first().copied().unwrap_or(0);
    if got_magic != expected_magic {
        return Err(LibTpuError::VliwIsaCodecError {
            generation_code: isa_code,
            bundle_offset: offset,
            reason: format!("invalid magic {got_magic:#04x}, expected {expected_magic:#04x}"),
        });
    }
    let expected_tag = compute_bundle_highway_tag8(bytes);
    let got_tag = bytes.get(7).copied().unwrap_or(0);
    if got_tag != expected_tag {
        return Err(LibTpuError::VliwIsaCodecError {
            generation_code: isa_code,
            bundle_offset: offset,
            reason: format!(
                "HighwayHash64 bundle tag mismatch: expected {expected_tag:#04x}, got {got_tag:#04x}"
            ),
        });
    }
    Ok(())
}

/// Cloud TPU v2 (`jfc` / Jellyfish) 32-byte (`256`-bit) binary VLIW codec (`JellyfishIsaCodec`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct JellyfishIsaCodec;

impl JellyfishIsaCodec {
    /// Lowers a slice of [`LloBundle`]s into a `platforms_deepsea.jellyfish.isa.Program` protobuf.
    ///
    /// # Arguments
    ///
    /// * `bundles` - Architectural [`LloBundle`] slice.
    #[must_use]
    pub fn to_isa_program_proto(bundles: &[LloBundle]) -> JellyfishIsaProgramProto {
        let pb_bundles = bundles
            .iter()
            .map(|b| JellyfishIsaBundleProto {
                imm_0: Some(b.scalar_slots[0].imm32),
                imm_1: Some(u32::from(b.vector_slots[0].opcode)),
                imm_2: Some(b.vector_slots[0].dst_vreg),
                imm_3: Some(b.vector_slots[0].src0_vreg),
                imm_4: Some(u32::from(b.matrix_slots[0].acc_vreg)),
                imm_5: Some(b.dma_slot.sync_flag_id),
                ..JellyfishIsaBundleProto::default()
            })
            .collect();
        JellyfishIsaProgramProto {
            bundles: pb_bundles,
            annotations: None,
        }
    }
}

impl GenerationIsaCodec for JellyfishIsaCodec {
    const ISA_CODE: &'static str = "jfc";
    const HEADER_MAGIC: u8 = 0x22;
    const BUNDLE_BYTES: usize = 32;

    #[allow(clippy::cast_possible_truncation)]
    fn encode_bundle(bundle: &LloBundle) -> Vec<u8> {
        let mut raw = [0_u8; 32];
        raw[0] = Self::HEADER_MAGIC;
        raw[1] = bundle.stall_cycles & 0x0F;
        raw[2..4].copy_from_slice(&bundle.bundle_flags.to_le_bytes());
        raw[4..6].copy_from_slice(&bundle.pred_mask.to_le_bytes());
        raw[6] = (bundle.xlu_slots[0].imm16 & 0xFF) as u8;

        let s0 = &bundle.scalar_slots[0];
        raw[8] = s0.opcode;
        raw[9] = s0.dst_sreg;
        raw[10] = s0.src0_sreg;
        raw[11] = s0.src1_sreg;
        raw[12..14].copy_from_slice(&((s0.imm32 >> 8) as u16).to_le_bytes());

        let v0 = &bundle.vector_slots[0];
        raw[14..16].copy_from_slice(&v0.opcode.to_le_bytes());
        raw[16] = v0.element_type;
        raw[17] = v0.lane_mask;
        raw[18] = v0.dst_vreg as u8;
        raw[19] = v0.src0_vreg as u8;
        raw[20] = v0.src1_vreg as u8;
        raw[21] = v0.src2_vreg as u8;
        raw[22] = bundle.xlu_slots[0].opcode;

        let m0 = &bundle.matrix_slots[0];
        raw[23] = m0.opcode;
        raw[24] = m0.precision_mode;
        raw[25] = m0.acc_vreg as u8;
        raw[26] = m0.lhs_vreg as u8;
        raw[27] = m0.rhs_vreg as u8;
        raw[28] = m0.k_tiles as u8;

        raw[29] = bundle.dma_slot.dma_opcode;
        raw[30] = (bundle.dma_slot.src_space & 0x0F) | ((bundle.dma_slot.dst_space & 0x0F) << 4);
        raw[31] = bundle.dma_slot.sync_flag_id as u8;

        raw[7] = compute_bundle_highway_tag8(&raw);
        raw.to_vec()
    }

    fn decode_bundle(bytes: &[u8], bundle_offset: usize) -> LibTpuResult<LloBundle> {
        verify_header_and_tag(
            Self::ISA_CODE,
            Self::HEADER_MAGIC,
            Self::BUNDLE_BYTES,
            bytes,
            bundle_offset,
        )?;
        Ok(decode_32byte_jfc_dfc(bytes, TpuGeneration::V2))
    }
}

/// Cloud TPU v3 (`dfc` / Dragonfish) 32-byte (`256`-bit) binary VLIW codec (`DragonfishIsaCodec`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DragonfishIsaCodec;

impl DragonfishIsaCodec {
    /// Lowers a slice of [`LloBundle`]s into a `platforms_deepsea.jellyfish.isa.Program` protobuf.
    ///
    /// # Arguments
    ///
    /// * `bundles` - Architectural [`LloBundle`] slice.
    #[must_use]
    pub fn to_isa_program_proto(bundles: &[LloBundle]) -> JellyfishIsaProgramProto {
        JellyfishIsaCodec::to_isa_program_proto(bundles)
    }
}

impl GenerationIsaCodec for DragonfishIsaCodec {
    const ISA_CODE: &'static str = "dfc";
    const HEADER_MAGIC: u8 = 0x33;
    const BUNDLE_BYTES: usize = 32;

    #[allow(clippy::cast_possible_truncation)]
    fn encode_bundle(bundle: &LloBundle) -> Vec<u8> {
        let mut v = JellyfishIsaCodec::encode_bundle(bundle);
        v.get_mut(0)
            .into_iter()
            .for_each(|b0| *b0 = Self::HEADER_MAGIC);
        let bf16_mode = (bundle.matrix_slots[0].precision_mode & 0x0F) << 4;
        v.get_mut(1)
            .into_iter()
            .for_each(|b1| *b1 = (bundle.stall_cycles & 0x0F) | bf16_mode);
        let tag = compute_bundle_highway_tag8(&v);
        v.get_mut(7).into_iter().for_each(|b7| *b7 = tag);
        v
    }

    fn decode_bundle(bytes: &[u8], bundle_offset: usize) -> LibTpuResult<LloBundle> {
        verify_header_and_tag(
            Self::ISA_CODE,
            Self::HEADER_MAGIC,
            Self::BUNDLE_BYTES,
            bytes,
            bundle_offset,
        )?;
        Ok(decode_32byte_jfc_dfc(bytes, TpuGeneration::V3))
    }
}

/// Shared 32-byte `jfc`/`dfc` slot decoder after magic and `HighwayHash64` verification.
///
/// # Arguments
///
/// * `b` - Verified 32-byte bundle slice.
/// * `generation` - Target [`TpuGeneration`] (`V2` or `V3`).
fn decode_32byte_jfc_dfc(b: &[u8], generation: TpuGeneration) -> LloBundle {
    let mut raw = [0_u8; 32];
    raw.copy_from_slice(b.get(..32).unwrap_or(&[0_u8; 32]));
    let stall_cycles = raw[1] & 0x0F;
    let bundle_flags = u16::from_le_bytes([raw[2], raw[3]]);
    let pred_mask = u16::from_le_bytes([raw[4], raw[5]]);
    let imm_scaled = u32::from(u16::from_le_bytes([raw[12], raw[13]])) << 8;
    let s0 = ScalarAluSlot {
        opcode: raw[8],
        dst_sreg: raw[9],
        src0_sreg: raw[10],
        src1_sreg: raw[11],
        imm32: imm_scaled,
    };
    let v0 = VectorAluSlot {
        opcode: u16::from_le_bytes([raw[14], raw[15]]),
        element_type: raw[16],
        lane_mask: raw[17],
        dst_vreg: u32::from(raw[18]),
        src0_vreg: u32::from(raw[19]),
        src1_vreg: u32::from(raw[20]),
        src2_vreg: u32::from(raw[21]),
    };
    let m0 = MatrixMxuSlot {
        opcode: raw[23],
        precision_mode: raw[24],
        flags: u16::from(raw[23] != 0),
        acc_vreg: u16::from(raw[25]),
        lhs_vreg: u16::from(raw[26]),
        rhs_vreg: u16::from(raw[27]),
        k_tiles: u16::from(raw[28]),
    };
    let x0 = ExtendedUnarySlot {
        opcode: raw[22],
        perm_mode: u8::from(raw[22] != 0),
        dst_vreg: u16::from(raw[18]),
        src_vreg: u16::from(raw[19]),
        imm16: u16::from(raw[6]),
    };
    let dma_slot = AsyncDmaSlot {
        dma_opcode: raw[29],
        src_space: raw[30] & 0x0F,
        dst_space: (raw[30] >> 4) & 0x0F,
        collective_op: if raw[29] > 1 { raw[29] } else { 0 },
        sync_flag_id: u32::from(raw[31]),
    };
    LloBundle {
        generation,
        bundle_flags,
        pred_mask,
        stall_cycles,
        scalar_slots: [s0, ScalarAluSlot::default()],
        vector_slots: [v0, VectorAluSlot::default()],
        matrix_slots: [
            m0,
            MatrixMxuSlot::default(),
            MatrixMxuSlot::default(),
            MatrixMxuSlot::default(),
        ],
        xlu_slots: [x0, ExtendedUnarySlot::default()],
        dma_slot,
        immediates: [imm_scaled, 0, 0, 0, 0, 0],
    }
}

/// 3D twisted-torus ICI optical & `CMEM` DMA descriptor for Cloud TPU v4 (`PufferfishOpticalDmaDescriptor`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PufferfishOpticalDmaDescriptor {
    /// Signed 2-bit X-axis optical link hop (`-1..=2`).
    pub delta_x: i8,
    /// Signed 2-bit Y-axis optical link hop (`-1..=2`).
    pub delta_y: i8,
    /// Signed 2-bit Z-axis optical OCS hop (`-1..=2`).
    pub delta_z: i8,
    /// Whether `CMEM` (`32 MiB` on-chip memory space `3`) is addressed.
    pub cmem_enabled: bool,
}

impl PufferfishOpticalDmaDescriptor {
    /// Packs the 3D optical coordinate + `CMEM` flag into a single header descriptor byte.
    #[must_use]
    #[allow(clippy::cast_sign_loss)]
    pub const fn pack_byte(self) -> u8 {
        let x = ((self.delta_x + 1) as u8) & 0x03;
        let y = ((self.delta_y + 1) as u8) & 0x03;
        let z = ((self.delta_z + 1) as u8) & 0x03;
        x | (y << 2) | (z << 4) | (if self.cmem_enabled { 0x40 } else { 0 })
    }

    /// Unpacks a [`PufferfishOpticalDmaDescriptor`] from its packed byte.
    ///
    /// # Arguments
    ///
    /// * `byte` - Packed optical/CMEM descriptor byte.
    #[must_use]
    #[allow(clippy::cast_possible_wrap)]
    pub const fn unpack_byte(byte: u8) -> Self {
        Self {
            delta_x: ((byte & 0x03) as i8) - 1,
            delta_y: (((byte >> 2) & 0x03) as i8) - 1,
            delta_z: (((byte >> 4) & 0x03) as i8) - 1,
            cmem_enabled: (byte & 0x40) != 0,
        }
    }
}

/// Cloud TPU v4 & v4lite (`pfc` / Pufferfish) 40-byte (`320`-bit) binary VLIW codec (`PufferfishIsaCodec`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PufferfishIsaCodec;

impl PufferfishIsaCodec {
    /// Encodes an [`LloBundle`] with explicit `CMEM` and 3D ICI optical DMA metadata into 40 bytes.
    ///
    /// # Arguments
    ///
    /// * `bundle` - Architectural [`LloBundle`].
    /// * `optical` - 3D ICI optical & `CMEM` descriptor.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn encode_bundle_with_optical(
        bundle: &LloBundle,
        optical: PufferfishOpticalDmaDescriptor,
    ) -> Vec<u8> {
        let mut raw = [0_u8; 40];
        let cmem_active = optical.cmem_enabled
            || (bundle.dma_slot.src_space == 3)
            || (bundle.dma_slot.dst_space == 3);
        raw[0] = Self::HEADER_MAGIC;
        raw[1] = (bundle.stall_cycles & 0x0F) | (u8::from(cmem_active) << 4);
        raw[2..4].copy_from_slice(&bundle.bundle_flags.to_le_bytes());
        raw[4..6].copy_from_slice(&bundle.pred_mask.to_le_bytes());
        raw[6] = PufferfishOpticalDmaDescriptor {
            cmem_enabled: cmem_active,
            ..optical
        }
        .pack_byte();

        // S0 (bytes 8..11) + S1 (bytes 11..14) + IMM0 (bytes 14..18)
        let s0 = &bundle.scalar_slots[0];
        let s1 = &bundle.scalar_slots[1];
        raw[8] = s0.opcode;
        raw[9] = s0.dst_sreg;
        raw[10] = (s0.src0_sreg & 0x0F) | ((s0.src1_sreg & 0x0F) << 4);
        raw[11] = s1.opcode;
        raw[12] = s1.dst_sreg;
        raw[13] = (s1.src0_sreg & 0x0F) | ((s1.src1_sreg & 0x0F) << 4);
        raw[14..18].copy_from_slice(&s0.imm32.to_le_bytes());

        // V0 (bytes 18..24) + V1 (bytes 24..30)
        let v0 = &bundle.vector_slots[0];
        let v1 = &bundle.vector_slots[1];
        raw[18] = v0.opcode as u8;
        raw[19] = v0.element_type;
        raw[20] = v0.dst_vreg as u8;
        raw[21] = v0.src0_vreg as u8;
        raw[22] = v0.src1_vreg as u8;
        raw[23] = v0.src2_vreg as u8;

        raw[24] = v1.opcode as u8;
        raw[25] = v1.element_type;
        raw[26] = v1.dst_vreg as u8;
        raw[27] = v1.src0_vreg as u8;
        raw[28] = v1.src1_vreg as u8;
        raw[29] = v1.src2_vreg as u8;

        // M0 (bytes 30..36)
        let m0 = &bundle.matrix_slots[0];
        raw[30] = m0.opcode;
        raw[31] = m0.precision_mode;
        raw[32] = m0.acc_vreg as u8;
        raw[33] = m0.lhs_vreg as u8;
        raw[34] = m0.rhs_vreg as u8;
        raw[35] = m0.k_tiles as u8;

        // X0 + D0 (bytes 36..40)
        raw[36] = bundle.xlu_slots[0].opcode;
        raw[37] = (bundle.dma_slot.dma_opcode & 0x0F)
            | ((bundle.dma_slot.src_space & 0x03) << 4)
            | ((bundle.dma_slot.dst_space & 0x03) << 6);
        raw[38..40].copy_from_slice(&(bundle.dma_slot.sync_flag_id as u16).to_le_bytes());

        raw[7] = compute_bundle_highway_tag8(&raw);
        raw.to_vec()
    }

    /// Extracts the [`PufferfishOpticalDmaDescriptor`] from an encoded 40-byte `pfc` bundle.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Raw 40-byte `pfc` bundle slice.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::VliwIsaCodecError`] if `bytes` fails `pfc` verification.
    pub fn decode_optical_descriptor(bytes: &[u8]) -> LibTpuResult<PufferfishOpticalDmaDescriptor> {
        verify_header_and_tag(
            Self::ISA_CODE,
            Self::HEADER_MAGIC,
            Self::BUNDLE_BYTES,
            bytes,
            0,
        )?;
        Ok(PufferfishOpticalDmaDescriptor::unpack_byte(
            bytes.get(6).copied().unwrap_or(0),
        ))
    }

    /// Lowers [`LloBundle`]s into an `asic_sw.deepsea.pxc.isa.TensorCoreProgram` protobuf, populating
    /// `cmem_load` when `CMEM` space (`3`) is active.
    ///
    /// # Arguments
    ///
    /// * `bundles` - Architectural [`LloBundle`] slice.
    #[must_use]
    pub fn to_isa_program_proto(bundles: &[LloBundle]) -> PufferfishTensorCoreProgramProto {
        use crate::proto::generated_prost_schemas::asic_sw::deepsea::pxc::isa::TensorCoreCmemLoad;
        let pb_bundles = bundles
            .iter()
            .map(|b| {
                let cmem_load = if (b.dma_slot.src_space == 3) || (b.dma_slot.dst_space == 3) {
                    Some(TensorCoreCmemLoad {
                        latency: Some(4),
                        bit_width: Some(320),
                        ..TensorCoreCmemLoad::default()
                    })
                } else {
                    None
                };
                PufferfishTensorCoreBundleProto {
                    cmem_load,
                    ..PufferfishTensorCoreBundleProto::default()
                }
            })
            .collect();
        PufferfishTensorCoreProgramProto {
            bundles: pb_bundles,
        }
    }
}

impl GenerationIsaCodec for PufferfishIsaCodec {
    const ISA_CODE: &'static str = "pfc";
    const HEADER_MAGIC: u8 = 0x44;
    const BUNDLE_BYTES: usize = 40;

    fn encode_bundle(bundle: &LloBundle) -> Vec<u8> {
        Self::encode_bundle_with_optical(bundle, PufferfishOpticalDmaDescriptor::default())
    }

    fn decode_bundle(bytes: &[u8], bundle_offset: usize) -> LibTpuResult<LloBundle> {
        verify_header_and_tag(
            Self::ISA_CODE,
            Self::HEADER_MAGIC,
            Self::BUNDLE_BYTES,
            bytes,
            bundle_offset,
        )?;
        let mut raw = [0_u8; 40];
        raw.copy_from_slice(bytes.get(..40).unwrap_or(&[0_u8; 40]));
        let stall_cycles = raw[1] & 0x0F;
        let bundle_flags = u16::from_le_bytes([raw[2], raw[3]]);
        let pred_mask = u16::from_le_bytes([raw[4], raw[5]]);
        let imm0 = u32::from_le_bytes([raw[14], raw[15], raw[16], raw[17]]);
        let s0 = ScalarAluSlot {
            opcode: raw[8],
            dst_sreg: raw[9],
            src0_sreg: raw[10] & 0x0F,
            src1_sreg: (raw[10] >> 4) & 0x0F,
            imm32: imm0,
        };
        let s1 = ScalarAluSlot {
            opcode: raw[11],
            dst_sreg: raw[12],
            src0_sreg: raw[13] & 0x0F,
            src1_sreg: (raw[13] >> 4) & 0x0F,
            imm32: 0,
        };
        let v0 = VectorAluSlot {
            opcode: u16::from(raw[18]),
            element_type: raw[19],
            lane_mask: 0xFF,
            dst_vreg: u32::from(raw[20]),
            src0_vreg: u32::from(raw[21]),
            src1_vreg: u32::from(raw[22]),
            src2_vreg: u32::from(raw[23]),
        };
        let v1 = VectorAluSlot {
            opcode: u16::from(raw[24]),
            element_type: raw[25],
            lane_mask: 0xFF,
            dst_vreg: u32::from(raw[26]),
            src0_vreg: u32::from(raw[27]),
            src1_vreg: u32::from(raw[28]),
            src2_vreg: u32::from(raw[29]),
        };
        let m0 = MatrixMxuSlot {
            opcode: raw[30],
            precision_mode: raw[31],
            flags: u16::from(raw[30] != 0),
            acc_vreg: u16::from(raw[32]),
            lhs_vreg: u16::from(raw[33]),
            rhs_vreg: u16::from(raw[34]),
            k_tiles: u16::from(raw[35]),
        };
        let x0 = ExtendedUnarySlot {
            opcode: raw[36],
            perm_mode: u8::from(raw[36] != 0),
            dst_vreg: u16::from(raw[20]),
            src_vreg: u16::from(raw[21]),
            imm16: 0,
        };
        let d_op = raw[37] & 0x0F;
        let dma_slot = AsyncDmaSlot {
            dma_opcode: d_op,
            src_space: (raw[37] >> 4) & 0x03,
            dst_space: (raw[37] >> 6) & 0x03,
            collective_op: if d_op > 1 { d_op } else { 0 },
            sync_flag_id: u32::from(u16::from_le_bytes([raw[38], raw[39]])),
        };
        Ok(LloBundle {
            generation: TpuGeneration::V4,
            bundle_flags,
            pred_mask,
            stall_cycles,
            scalar_slots: [s0, s1],
            vector_slots: [v0, v1],
            matrix_slots: [
                m0,
                MatrixMxuSlot::default(),
                MatrixMxuSlot::default(),
                MatrixMxuSlot::default(),
            ],
            xlu_slots: [x0, ExtendedUnarySlot::default()],
            dma_slot,
            immediates: [imm0, 0, 0, 0, 0, 0],
        })
    }
}

/// Native MXU numeric precision mode for Cloud TPU v5e & v5p (`ViperfishMxuMode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViperfishMxuMode {
    /// IEEE FP32 accumulation (`0`).
    Fp32 = 0,
    /// Native BF16 systolic contraction (`1`).
    Bf16 = 1,
    /// Native INT8 (`S8`/`U8`) 2x-throughput systolic contraction (`2`).
    Int8Native = 2,
    /// Native FP8 `E4M3FN` systolic contraction (`3`).
    Fp8E4M3Fn = 3,
    /// Native FP8 `E5M2` systolic contraction (`4`).
    Fp8E5M2 = 4,
}

impl ViperfishMxuMode {
    /// Reconstructs a [`ViperfishMxuMode`] from a 4-bit nibble.
    ///
    /// # Arguments
    ///
    /// * `nibble` - Encoded 4-bit MXU mode nibble.
    #[must_use]
    pub const fn from_nibble(nibble: u8) -> Self {
        match nibble & 0x0F {
            1 => Self::Bf16,
            2 => Self::Int8Native,
            3 => Self::Fp8E4M3Fn,
            4 => Self::Fp8E5M2,
            _ => Self::Fp32,
        }
    }
}

/// Cloud TPU v5e & v5p (`vfc` / Viperfish) 52-byte (`416`-bit) binary VLIW codec (`ViperfishIsaCodec`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ViperfishIsaCodec;

impl ViperfishIsaCodec {
    /// Encodes an [`LloBundle`] with explicit native INT8/FP8 [`ViperfishMxuMode`] and stochastic rounding mask.
    ///
    /// # Arguments
    ///
    /// * `bundle` - Architectural [`LloBundle`].
    /// * `mxu_mode` - Native [`ViperfishMxuMode`].
    /// * `stochastic_round` - Whether LFSR stochastic rounding is enabled on vector pack/convert ops.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn encode_bundle_with_mode(
        bundle: &LloBundle,
        mxu_mode: ViperfishMxuMode,
        stochastic_round: bool,
    ) -> Vec<u8> {
        let mut raw = [0_u8; 52];
        raw[0] = Self::HEADER_MAGIC;
        raw[1] = (bundle.stall_cycles & 0x0F) | ((mxu_mode as u8) << 4);
        raw[2..4].copy_from_slice(&bundle.bundle_flags.to_le_bytes());
        raw[4..6].copy_from_slice(&bundle.pred_mask.to_le_bytes());
        raw[6] = u8::from(stochastic_round);

        let s0 = &bundle.scalar_slots[0];
        let s1 = &bundle.scalar_slots[1];
        raw[8] = s0.opcode;
        raw[9] = s0.dst_sreg;
        raw[10] = s0.src0_sreg;
        raw[11] = s0.src1_sreg;
        raw[12] = s1.opcode;
        raw[13] = s1.dst_sreg;
        raw[14] = s1.src0_sreg;
        raw[15] = s1.src1_sreg;
        raw[16..20].copy_from_slice(&s0.imm32.to_le_bytes());

        let v0 = &bundle.vector_slots[0];
        let v1 = &bundle.vector_slots[1];
        raw[20..22].copy_from_slice(&v0.opcode.to_le_bytes());
        raw[22] = v0.element_type;
        raw[23] = v0.lane_mask;
        raw[24] = v0.dst_vreg as u8;
        raw[25] = v0.src0_vreg as u8;
        raw[26] = v0.src1_vreg as u8;
        raw[27] = v0.src2_vreg as u8;
        raw[28..30].copy_from_slice(&v1.opcode.to_le_bytes());
        raw[30] = v1.element_type;
        raw[31] = v1.lane_mask;
        raw[32] = v1.dst_vreg as u8;
        raw[33] = v1.src0_vreg as u8;
        raw[34] = v1.src1_vreg as u8;
        raw[35] = v1.src2_vreg as u8;

        let m0 = &bundle.matrix_slots[0];
        let m1 = &bundle.matrix_slots[1];
        raw[36] = m0.opcode;
        raw[37] = mxu_mode as u8;
        raw[38] = m0.acc_vreg as u8;
        raw[39] = m0.lhs_vreg as u8;
        raw[40] = m0.rhs_vreg as u8;
        raw[41] = m0.k_tiles as u8;

        raw[42] = m1.opcode;
        raw[43] = m1.precision_mode;
        raw[44] = m1.acc_vreg as u8;
        raw[45] = m1.lhs_vreg as u8;
        raw[46] = m1.rhs_vreg as u8;
        raw[47] = m1.k_tiles as u8;

        raw[48] = bundle.xlu_slots[0].opcode;
        raw[49] = bundle.dma_slot.dma_opcode;
        raw[50..52].copy_from_slice(&(bundle.dma_slot.sync_flag_id as u16).to_le_bytes());

        raw[7] = compute_bundle_highway_tag8(&raw);
        raw.to_vec()
    }

    /// Decodes the native [`ViperfishMxuMode`] and stochastic rounding flag from an encoded 52-byte `vfc` bundle.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Raw 52-byte `vfc` bundle slice.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::VliwIsaCodecError`] if `bytes` fails `vfc` verification.
    pub fn decode_mxu_mode(bytes: &[u8]) -> LibTpuResult<(ViperfishMxuMode, bool)> {
        verify_header_and_tag(
            Self::ISA_CODE,
            Self::HEADER_MAGIC,
            Self::BUNDLE_BYTES,
            bytes,
            0,
        )?;
        let mode = ViperfishMxuMode::from_nibble(bytes.get(1).copied().unwrap_or(0) >> 4);
        let sr = bytes.get(6).copied().unwrap_or(0) != 0;
        Ok((mode, sr))
    }

    /// Lowers [`LloBundle`]s into an `asic_sw.deepsea.vxc.isa.TensorCoreProgram` protobuf.
    ///
    /// # Arguments
    ///
    /// * `bundles` - Architectural [`LloBundle`] slice.
    #[must_use]
    pub fn to_isa_program_proto(bundles: &[LloBundle]) -> ViperfishTensorCoreProgramProto {
        use crate::proto::generated_prost_schemas::asic_sw::deepsea::vxc::isa::TensorCoreImmediates;
        let pb_bundles = bundles
            .iter()
            .map(|b| ViperfishTensorCoreBundleProto {
                immediates: Some(TensorCoreImmediates {
                    imm_0: Some(b.scalar_slots[0].imm32),
                    imm_1: Some(u32::from(b.vector_slots[0].opcode)),
                    imm_2: Some(b.vector_slots[0].dst_vreg),
                    imm_3: Some(b.dma_slot.sync_flag_id),
                    ..TensorCoreImmediates::default()
                }),
                ..ViperfishTensorCoreBundleProto::default()
            })
            .collect();
        ViperfishTensorCoreProgramProto {
            bundles: pb_bundles,
        }
    }
}

impl GenerationIsaCodec for ViperfishIsaCodec {
    const ISA_CODE: &'static str = "vfc";
    const HEADER_MAGIC: u8 = 0x55;
    const BUNDLE_BYTES: usize = 52;

    fn encode_bundle(bundle: &LloBundle) -> Vec<u8> {
        let mode = ViperfishMxuMode::from_nibble(bundle.matrix_slots[0].precision_mode);
        Self::encode_bundle_with_mode(bundle, mode, false)
    }

    fn decode_bundle(bytes: &[u8], bundle_offset: usize) -> LibTpuResult<LloBundle> {
        verify_header_and_tag(
            Self::ISA_CODE,
            Self::HEADER_MAGIC,
            Self::BUNDLE_BYTES,
            bytes,
            bundle_offset,
        )?;
        let mut raw = [0_u8; 52];
        raw.copy_from_slice(bytes.get(..52).unwrap_or(&[0_u8; 52]));
        let stall_cycles = raw[1] & 0x0F;
        let bundle_flags = u16::from_le_bytes([raw[2], raw[3]]);
        let pred_mask = u16::from_le_bytes([raw[4], raw[5]]);
        let imm0 = u32::from_le_bytes([raw[16], raw[17], raw[18], raw[19]]);
        let s0 = ScalarAluSlot {
            opcode: raw[8],
            dst_sreg: raw[9],
            src0_sreg: raw[10],
            src1_sreg: raw[11],
            imm32: imm0,
        };
        let s1 = ScalarAluSlot {
            opcode: raw[12],
            dst_sreg: raw[13],
            src0_sreg: raw[14],
            src1_sreg: raw[15],
            imm32: 0,
        };
        let v_slots = [
            VectorAluSlot {
                opcode: u16::from_le_bytes([raw[20], raw[21]]),
                element_type: raw[22],
                lane_mask: raw[23],
                dst_vreg: u32::from(raw[24]),
                src0_vreg: u32::from(raw[25]),
                src1_vreg: u32::from(raw[26]),
                src2_vreg: u32::from(raw[27]),
            },
            VectorAluSlot {
                opcode: u16::from_le_bytes([raw[28], raw[29]]),
                element_type: raw[30],
                lane_mask: raw[31],
                dst_vreg: u32::from(raw[32]),
                src0_vreg: u32::from(raw[33]),
                src1_vreg: u32::from(raw[34]),
                src2_vreg: u32::from(raw[35]),
            },
        ];
        let m0 = MatrixMxuSlot {
            opcode: raw[36],
            precision_mode: raw[37],
            flags: u16::from(raw[36] != 0),
            acc_vreg: u16::from(raw[38]),
            lhs_vreg: u16::from(raw[39]),
            rhs_vreg: u16::from(raw[40]),
            k_tiles: u16::from(raw[41]),
        };
        let m1 = MatrixMxuSlot {
            opcode: raw[42],
            precision_mode: raw[43],
            flags: u16::from(raw[42] != 0),
            acc_vreg: u16::from(raw[44]),
            lhs_vreg: u16::from(raw[45]),
            rhs_vreg: u16::from(raw[46]),
            k_tiles: u16::from(raw[47]),
        };
        let x0 = ExtendedUnarySlot {
            opcode: raw[48],
            perm_mode: u8::from(raw[48] != 0),
            dst_vreg: u16::from(raw[24]),
            src_vreg: u16::from(raw[25]),
            imm16: 0,
        };
        let dma_slot = AsyncDmaSlot {
            dma_opcode: raw[49],
            src_space: 1,
            dst_space: u8::from(raw[49] <= 1),
            collective_op: if raw[49] > 1 { raw[49] } else { 0 },
            sync_flag_id: u32::from(u16::from_le_bytes([raw[50], raw[51]])),
        };
        Ok(LloBundle {
            generation: TpuGeneration::V5e,
            bundle_flags,
            pred_mask,
            stall_cycles,
            scalar_slots: [s0, s1],
            vector_slots: v_slots,
            matrix_slots: [m0, m1, MatrixMxuSlot::default(), MatrixMxuSlot::default()],
            xlu_slots: [x0, ExtendedUnarySlot::default()],
            dma_slot,
            immediates: [imm0, 0, 0, 0, 0, 0],
        })
    }
}

/// Shared 64-byte (`512`-bit) encoder for Trillium (`gfc`) and Ironwood (`iwc`) keeping `S0.imm32`
/// at bytes `12..16` (`ProgramRelocationProto` `slot = 3`) and `HighwayHash64` tag at byte `7`.
///
/// # Arguments
///
/// * `magic` - Generation magic byte (`0x66` for `gfc`, `0x77` for `iwc`).
/// * `mode_nibble` - High nibble of byte `1` (`256x256` MXU / microscaling mode).
/// * `aux_byte` - Auxiliary descriptor byte `6` (`SparseCore` barrier ID or `E8M0FNU` scale exponent).
/// * `bundle` - Architectural [`LloBundle`].
#[must_use]
#[allow(clippy::cast_possible_truncation)]
fn encode_64byte_gfc_iwc(magic: u8, mode_nibble: u8, aux_byte: u8, bundle: &LloBundle) -> Vec<u8> {
    let mut raw = [0_u8; 64];
    raw[0] = magic;
    raw[1] = (bundle.stall_cycles & 0x0F) | ((mode_nibble & 0x0F) << 4);
    raw[2..4].copy_from_slice(&bundle.bundle_flags.to_le_bytes());
    raw[4..6].copy_from_slice(&bundle.pred_mask.to_le_bytes());
    raw[6] = aux_byte;

    // S0 (bytes 8..12) + S0.imm32 (bytes 12..16 -> relocation slot 3) + S1 (bytes 16..20)
    let s0 = &bundle.scalar_slots[0];
    let s1 = &bundle.scalar_slots[1];
    raw[8] = s0.opcode;
    raw[9] = s0.dst_sreg;
    raw[10] = s0.src0_sreg;
    raw[11] = s0.src1_sreg;
    raw[12..16].copy_from_slice(&s0.imm32.to_le_bytes());
    raw[16] = s1.opcode;
    raw[17] = s1.dst_sreg;
    raw[18] = s1.src0_sreg;
    raw[19] = s1.src1_sreg;

    // V0 (bytes 20..30) + V1 (bytes 30..40)
    let v0 = &bundle.vector_slots[0];
    let v1 = &bundle.vector_slots[1];
    raw[20..22].copy_from_slice(&v0.opcode.to_le_bytes());
    raw[22] = v0.element_type;
    raw[23] = v0.lane_mask;
    raw[24..26].copy_from_slice(&(v0.dst_vreg as u16).to_le_bytes());
    raw[26..28].copy_from_slice(&(v0.src0_vreg as u16).to_le_bytes());
    raw[28] = v0.src1_vreg as u8;
    raw[29] = v0.src2_vreg as u8;

    raw[30..32].copy_from_slice(&v1.opcode.to_le_bytes());
    raw[32] = v1.element_type;
    raw[33] = v1.lane_mask;
    raw[34..36].copy_from_slice(&(v1.dst_vreg as u16).to_le_bytes());
    raw[36..38].copy_from_slice(&(v1.src0_vreg as u16).to_le_bytes());
    raw[38] = v1.src1_vreg as u8;
    raw[39] = v1.src2_vreg as u8;

    // M0 (bytes 40..48) + M1 (bytes 48..56)
    let m0 = &bundle.matrix_slots[0];
    let m1 = &bundle.matrix_slots[1];
    raw[40] = m0.opcode;
    raw[41] = m0.precision_mode;
    raw[42..44].copy_from_slice(&m0.flags.to_le_bytes());
    raw[44] = m0.acc_vreg as u8;
    raw[45] = m0.lhs_vreg as u8;
    raw[46] = m0.rhs_vreg as u8;
    raw[47] = m0.k_tiles as u8;

    raw[48] = m1.opcode;
    raw[49] = m1.precision_mode;
    raw[50..52].copy_from_slice(&m1.flags.to_le_bytes());
    raw[52] = m1.acc_vreg as u8;
    raw[53] = m1.lhs_vreg as u8;
    raw[54] = m1.rhs_vreg as u8;
    raw[55] = m1.k_tiles as u8;

    // X0 (bytes 56..60) + D0 & X0.imm16 / HBM3e upper bits (bytes 60..64)
    let x0 = &bundle.xlu_slots[0];
    raw[56] = x0.opcode;
    raw[57] = x0.perm_mode;
    raw[58] = x0.dst_vreg as u8;
    raw[59] = x0.src_vreg as u8;

    raw[60] = bundle.dma_slot.dma_opcode;
    raw[61] = (bundle.dma_slot.src_space & 0x07)
        | ((bundle.dma_slot.dst_space & 0x07) << 3)
        | (u8::from(bundle.dma_slot.collective_op != 0) << 6);
    if bundle.dma_slot.dma_opcode != 0 {
        raw[62..64].copy_from_slice(&(bundle.dma_slot.sync_flag_id as u16).to_le_bytes());
    } else {
        raw[62..64].copy_from_slice(&x0.imm16.to_le_bytes());
    }

    raw[7] = compute_bundle_highway_tag8(&raw);
    raw.to_vec()
}

/// Shared 64-byte (`512`-bit) decoder for Trillium (`gfc`) and Ironwood (`iwc`).
///
/// # Arguments
///
/// * `b` - Verified 64-byte bundle slice.
/// * `generation` - Target [`TpuGeneration`] (`V6e` or `V7x`).
fn decode_64byte_gfc_iwc(b: &[u8], generation: TpuGeneration) -> LloBundle {
    let mut raw = [0_u8; 64];
    raw.copy_from_slice(b.get(..64).unwrap_or(&[0_u8; 64]));
    let stall_cycles = raw[1] & 0x0F;
    let bundle_flags = u16::from_le_bytes([raw[2], raw[3]]);
    let pred_mask = u16::from_le_bytes([raw[4], raw[5]]);
    let imm0 = u32::from_le_bytes([raw[12], raw[13], raw[14], raw[15]]);
    let s0 = ScalarAluSlot {
        opcode: raw[8],
        dst_sreg: raw[9],
        src0_sreg: raw[10],
        src1_sreg: raw[11],
        imm32: imm0,
    };
    let s1 = ScalarAluSlot {
        opcode: raw[16],
        dst_sreg: raw[17],
        src0_sreg: raw[18],
        src1_sreg: raw[19],
        imm32: 0,
    };
    let v_slots = [
        VectorAluSlot {
            opcode: u16::from_le_bytes([raw[20], raw[21]]),
            element_type: raw[22],
            lane_mask: raw[23],
            dst_vreg: u32::from(u16::from_le_bytes([raw[24], raw[25]])),
            src0_vreg: u32::from(u16::from_le_bytes([raw[26], raw[27]])),
            src1_vreg: u32::from(raw[28]),
            src2_vreg: u32::from(raw[29]),
        },
        VectorAluSlot {
            opcode: u16::from_le_bytes([raw[30], raw[31]]),
            element_type: raw[32],
            lane_mask: raw[33],
            dst_vreg: u32::from(u16::from_le_bytes([raw[34], raw[35]])),
            src0_vreg: u32::from(u16::from_le_bytes([raw[36], raw[37]])),
            src1_vreg: u32::from(raw[38]),
            src2_vreg: u32::from(raw[39]),
        },
    ];
    let m_slots = [
        MatrixMxuSlot {
            opcode: raw[40],
            precision_mode: raw[41],
            flags: u16::from_le_bytes([raw[42], raw[43]]),
            acc_vreg: u16::from(raw[44]),
            lhs_vreg: u16::from(raw[45]),
            rhs_vreg: u16::from(raw[46]),
            k_tiles: u16::from(raw[47]),
        },
        MatrixMxuSlot {
            opcode: raw[48],
            precision_mode: raw[49],
            flags: u16::from_le_bytes([raw[50], raw[51]]),
            acc_vreg: u16::from(raw[52]),
            lhs_vreg: u16::from(raw[53]),
            rhs_vreg: u16::from(raw[54]),
            k_tiles: u16::from(raw[55]),
        },
        MatrixMxuSlot::default(),
        MatrixMxuSlot::default(),
    ];
    let tail_u16 = u16::from_le_bytes([raw[62], raw[63]]);
    let d_op = raw[60];
    let x0 = ExtendedUnarySlot {
        opcode: raw[56],
        perm_mode: raw[57],
        dst_vreg: u16::from(raw[58]),
        src_vreg: u16::from(raw[59]),
        imm16: if d_op == 0 { tail_u16 } else { 0 },
    };
    let is_coll = (raw[61] & 0x40) != 0;
    let dma_slot = AsyncDmaSlot {
        dma_opcode: d_op,
        src_space: raw[61] & 0x07,
        dst_space: (raw[61] >> 3) & 0x07,
        collective_op: if is_coll { d_op } else { 0 },
        sync_flag_id: if d_op != 0 { u32::from(tail_u16) } else { 0 },
    };
    LloBundle {
        generation,
        bundle_flags,
        pred_mask,
        stall_cycles,
        scalar_slots: [s0, s1],
        vector_slots: v_slots,
        matrix_slots: m_slots,
        xlu_slots: [x0, ExtendedUnarySlot::default()],
        dma_slot,
        immediates: [imm0, 0, 0, 0, 0, 0],
    }
}

/// Cloud TPU v6e Trillium (`gfc` / Ghostfish) 64-byte (`512`-bit) binary VLIW codec (`GhostfishIsaCodec`).
///
/// Supports `256x256` systolic MXU tiles and `SparseCore` v2/v3 cross-core `SyncFlag` barriers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GhostfishIsaCodec;

impl GhostfishIsaCodec {
    /// Encodes an [`LloBundle`] with `256x256` MXU mode and `SparseCore` v2/v3 barrier `SyncFlag` ID.
    ///
    /// # Arguments
    ///
    /// * `bundle` - Architectural [`LloBundle`].
    /// * `mxu_256x256` - Whether `256x256` systolic array mode is active.
    /// * `sparsecore_sync_flag` - Cross-core `SparseCore` v2/v3 `SyncFlag` barrier ID (`0` if none).
    #[must_use]
    pub fn encode_bundle_trillium(
        bundle: &LloBundle,
        mxu_256x256: bool,
        sparsecore_sync_flag: u8,
    ) -> Vec<u8> {
        let mode = u8::from(mxu_256x256) | (u8::from(sparsecore_sync_flag != 0) << 1);
        encode_64byte_gfc_iwc(Self::HEADER_MAGIC, mode, sparsecore_sync_flag, bundle)
    }

    /// Extracts `(mxu_256x256, sparsecore_sync_flag)` from a 64-byte `gfc` bundle.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Raw 64-byte `gfc` bundle slice.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::VliwIsaCodecError`] if `bytes` fails `gfc` verification.
    pub fn decode_trillium_descriptors(bytes: &[u8]) -> LibTpuResult<(bool, u8)> {
        verify_header_and_tag(
            Self::ISA_CODE,
            Self::HEADER_MAGIC,
            Self::BUNDLE_BYTES,
            bytes,
            0,
        )?;
        let mode = (bytes.get(1).copied().unwrap_or(0) >> 4) & 0x0F;
        let sc_flag = bytes.get(6).copied().unwrap_or(0);
        Ok(((mode & 0x01) != 0, sc_flag))
    }

    /// Lowers [`LloBundle`]s into an `asic_sw.deepsea.gxc.glc.isa.TensorCoreProgram` protobuf.
    ///
    /// # Arguments
    ///
    /// * `bundles` - Architectural [`LloBundle`] slice.
    #[must_use]
    pub fn to_isa_program_proto(bundles: &[LloBundle]) -> GhostliteTensorCoreProgramProto {
        use crate::proto::generated_prost_schemas::asic_sw::deepsea::gxc::glc::isa::TensorCoreImmediates;
        let pb_bundles = bundles
            .iter()
            .map(|b| GhostliteTensorCoreBundleProto {
                immediates: Some(TensorCoreImmediates {
                    imm_0: Some(b.scalar_slots[0].imm32),
                    imm_1: Some(u32::from(b.vector_slots[0].opcode)),
                    imm_2: Some(b.vector_slots[0].dst_vreg),
                    imm_3: Some(b.dma_slot.sync_flag_id),
                    ..TensorCoreImmediates::default()
                }),
                ..GhostliteTensorCoreBundleProto::default()
            })
            .collect();
        GhostliteTensorCoreProgramProto {
            bundles: pb_bundles,
        }
    }
}

impl GenerationIsaCodec for GhostfishIsaCodec {
    const ISA_CODE: &'static str = "gfc";
    const HEADER_MAGIC: u8 = 0x66;
    const BUNDLE_BYTES: usize = 64;

    fn encode_bundle(bundle: &LloBundle) -> Vec<u8> {
        Self::encode_bundle_trillium(bundle, true, 0)
    }

    fn decode_bundle(bytes: &[u8], bundle_offset: usize) -> LibTpuResult<LloBundle> {
        verify_header_and_tag(
            Self::ISA_CODE,
            Self::HEADER_MAGIC,
            Self::BUNDLE_BYTES,
            bytes,
            bundle_offset,
        )?;
        Ok(decode_64byte_gfc_iwc(bytes, TpuGeneration::V6e))
    }
}

/// FP8/FP4 microscaling (`E8M0FNU` + `E2M1FN`) & 192 GiB `HBM3e` descriptor for Cloud TPU v7x (`IronwoodMicroscalingDescriptor`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IronwoodMicroscalingDescriptor {
    /// Whether `MXFP4` (`E8M0FNU` block scale + `E2M1FN` mantissa) microscaling is active.
    pub mxfp4_e2m1fn_enabled: bool,
    /// Whether `MXFP8` (`E8M0FNU` block scale + `F8E4M3FN`) microscaling is active.
    pub mxfp8_e4m3fn_enabled: bool,
    /// Shared `E8M0FNU` unsigned block scale exponent byte (`0..=254`, `127 = 2^0`).
    pub e8m0fnu_shared_exponent: u8,
    /// Full 48-bit `HBM3e` physical byte address (`0..192 GiB`).
    pub hbm3e_address_48bit: u64,
    /// Whether `SparseCore` v3 barrier synchronization is active.
    pub sparsecore_v3_sync: bool,
}

/// Cloud TPU v7x Ironwood (`iwc`) 64-byte (`512`-bit) binary VLIW codec (`IronwoodIsaCodec`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IronwoodIsaCodec;

impl IronwoodIsaCodec {
    /// Encodes an [`LloBundle`] with `E8M0FNU`+`E2M1FN` microscaling, 192 GiB `HBM3e` 48-bit addressing,
    /// and `SparseCore` v3 sync into 64 bytes.
    ///
    /// # Arguments
    ///
    /// * `bundle` - Architectural [`LloBundle`].
    /// * `desc` - [`IronwoodMicroscalingDescriptor`] parameters.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn encode_bundle_ironwood(
        bundle: &LloBundle,
        desc: IronwoodMicroscalingDescriptor,
    ) -> Vec<u8> {
        let mode = u8::from(desc.mxfp4_e2m1fn_enabled)
            | (u8::from(desc.mxfp8_e4m3fn_enabled) << 1)
            | (u8::from(desc.sparsecore_v3_sync) << 2)
            | (u8::from(desc.hbm3e_address_48bit > 0xFFFF_FFFF) << 3);
        let mut adjusted = *bundle;
        if desc.hbm3e_address_48bit != 0 {
            adjusted.scalar_slots[0].imm32 = (desc.hbm3e_address_48bit & 0xFFFF_FFFF) as u32;
            adjusted.xlu_slots[0].imm16 = ((desc.hbm3e_address_48bit >> 32) & 0xFFFF) as u16;
            adjusted.dma_slot.dma_opcode = 0;
        }
        encode_64byte_gfc_iwc(
            Self::HEADER_MAGIC,
            mode,
            desc.e8m0fnu_shared_exponent,
            &adjusted,
        )
    }

    /// Decodes the [`IronwoodMicroscalingDescriptor`] (including 48-bit `HBM3e` address reconstruction)
    /// from an encoded 64-byte `iwc` bundle.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Raw 64-byte `iwc` bundle slice.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::VliwIsaCodecError`] if `bytes` fails `iwc` verification.
    pub fn decode_ironwood_descriptor(
        bytes: &[u8],
    ) -> LibTpuResult<IronwoodMicroscalingDescriptor> {
        let decoded = Self::decode_bundle(bytes, 0)?;
        let mode = (bytes.get(1).copied().unwrap_or(0) >> 4) & 0x0F;
        let e8m0 = bytes.get(6).copied().unwrap_or(0);
        let lo32 = u64::from(decoded.scalar_slots[0].imm32);
        let hi16 = u64::from(decoded.xlu_slots[0].imm16);
        let addr48 = lo32 | (hi16 << 32);
        Ok(IronwoodMicroscalingDescriptor {
            mxfp4_e2m1fn_enabled: (mode & 0x01) != 0,
            mxfp8_e4m3fn_enabled: (mode & 0x02) != 0,
            sparsecore_v3_sync: (mode & 0x04) != 0,
            e8m0fnu_shared_exponent: e8m0,
            hbm3e_address_48bit: addr48,
        })
    }

    /// Lowers [`LloBundle`]s into an `asic_sw.deepsea.gxc.gfc.isa.TensorCoreProgram` protobuf.
    ///
    /// # Arguments
    ///
    /// * `bundles` - Architectural [`LloBundle`] slice.
    #[must_use]
    pub fn to_isa_program_proto(bundles: &[LloBundle]) -> IronwoodTensorCoreProgramProto {
        use crate::proto::generated_prost_schemas::asic_sw::deepsea::gxc::gfc::isa::TensorCoreImmediates;
        let pb_bundles = bundles
            .iter()
            .map(|b| IronwoodTensorCoreBundleProto {
                immediates: Some(TensorCoreImmediates {
                    imm_0: Some(b.scalar_slots[0].imm32),
                    imm_1: Some(u32::from(b.vector_slots[0].opcode)),
                    imm_2: Some(b.vector_slots[0].dst_vreg),
                    imm_3: Some(b.dma_slot.sync_flag_id),
                    ..TensorCoreImmediates::default()
                }),
                ..IronwoodTensorCoreBundleProto::default()
            })
            .collect();
        IronwoodTensorCoreProgramProto {
            bundles: pb_bundles,
        }
    }
}

impl GenerationIsaCodec for IronwoodIsaCodec {
    const ISA_CODE: &'static str = "iwc";
    const HEADER_MAGIC: u8 = 0x77;
    const BUNDLE_BYTES: usize = 64;

    fn encode_bundle(bundle: &LloBundle) -> Vec<u8> {
        encode_64byte_gfc_iwc(Self::HEADER_MAGIC, 0x03, 127, bundle)
    }

    fn decode_bundle(bytes: &[u8], bundle_offset: usize) -> LibTpuResult<LloBundle> {
        verify_header_and_tag(
            Self::ISA_CODE,
            Self::HEADER_MAGIC,
            Self::BUNDLE_BYTES,
            bytes,
            bundle_offset,
        )?;
        Ok(decode_64byte_gfc_iwc(bytes, TpuGeneration::V7x))
    }
}

/// Returns the encoded VLIW bundle byte width for `generation` (`32`, `40`, `52`, or `64` bytes).
///
/// # Arguments
///
/// * `generation` - Target [`TpuGeneration`].
#[must_use]
pub const fn generation_bundle_byte_width(generation: TpuGeneration) -> usize {
    match generation {
        TpuGeneration::V2 | TpuGeneration::V3 => 32,
        TpuGeneration::V4 | TpuGeneration::V4Lite => 40,
        TpuGeneration::V5e | TpuGeneration::V5p => 52,
        TpuGeneration::V6e | TpuGeneration::V7x => 64,
    }
}

/// Encodes a single [`LloBundle`] using the exact codec for `generation`.
///
/// # Arguments
///
/// * `generation` - Target [`TpuGeneration`].
/// * `bundle` - Architectural [`LloBundle`] to encode.
#[must_use]
pub fn encode_bundle_for_generation(generation: TpuGeneration, bundle: &LloBundle) -> Vec<u8> {
    match generation {
        TpuGeneration::V2 => JellyfishIsaCodec::encode_bundle(bundle),
        TpuGeneration::V3 => DragonfishIsaCodec::encode_bundle(bundle),
        TpuGeneration::V4 | TpuGeneration::V4Lite => PufferfishIsaCodec::encode_bundle(bundle),
        TpuGeneration::V5e | TpuGeneration::V5p => ViperfishIsaCodec::encode_bundle(bundle),
        TpuGeneration::V6e => GhostfishIsaCodec::encode_bundle(bundle),
        TpuGeneration::V7x => IronwoodIsaCodec::encode_bundle(bundle),
    }
}

/// Decodes a single [`LloBundle`] from `bitstream` using the exact codec for `generation`.
///
/// # Arguments
///
/// * `generation` - Target [`TpuGeneration`].
/// * `bitstream` - Raw generation-encoded bundle bytes (`32`, `40`, `52`, or `64` bytes).
///
/// # Errors
///
/// Returns [`LibTpuError::VliwIsaCodecError`] if length, magic, or `HighwayHash64` tag verification fails.
pub fn decode_bundle_for_generation(
    generation: TpuGeneration,
    bitstream: &[u8],
) -> LibTpuResult<LloBundle> {
    match generation {
        TpuGeneration::V2 => JellyfishIsaCodec::decode_bundle(bitstream, 0),
        TpuGeneration::V3 => DragonfishIsaCodec::decode_bundle(bitstream, 0),
        TpuGeneration::V4 | TpuGeneration::V4Lite => {
            PufferfishIsaCodec::decode_bundle(bitstream, 0)
        }
        TpuGeneration::V5e | TpuGeneration::V5p => ViperfishIsaCodec::decode_bundle(bitstream, 0),
        TpuGeneration::V6e => GhostfishIsaCodec::decode_bundle(bitstream, 0),
        TpuGeneration::V7x => IronwoodIsaCodec::decode_bundle(bitstream, 0),
    }
}

/// Encodes a canonical 64-byte (`512`-bit) `IMEM` word (allowing either `0x66` `gfc` or `0x77` `iwc`).
///
/// # Arguments
///
/// * `bundle` - Architectural [`LloBundle`].
#[must_use]
pub fn encode_canonical_64b_bundle(bundle: &LloBundle) -> [u8; 64] {
    let gen_byte = bundle.generation as u8;
    let vec = if bundle.generation == TpuGeneration::V7x {
        encode_64byte_gfc_iwc(IronwoodIsaCodec::HEADER_MAGIC, 0x03, gen_byte, bundle)
    } else {
        encode_64byte_gfc_iwc(GhostfishIsaCodec::HEADER_MAGIC, 0x01, gen_byte, bundle)
    };
    let mut out = [0_u8; 64];
    out.copy_from_slice(vec.get(..64).unwrap_or(&[0_u8; 64]));
    out
}

/// Decodes a canonical 64-byte (`512`-bit) `IMEM` word (`0x66` `gfc` or `0x77` `iwc`).
///
/// # Arguments
///
/// * `bytes` - Raw 64-byte `IMEM` word.
/// * `offset` - Byte offset in `IMEM`.
///
/// # Errors
///
/// Returns [`LibTpuError::VliwIsaCodecError`] if `bytes` is not a valid 64-byte `gfc`/`iwc` word.
pub fn decode_canonical_64b_bundle(bytes: &[u8], offset: usize) -> LibTpuResult<LloBundle> {
    let mut decoded = if bytes.first().copied() == Some(IronwoodIsaCodec::HEADER_MAGIC) {
        IronwoodIsaCodec::decode_bundle(bytes, offset)?
    } else {
        GhostfishIsaCodec::decode_bundle(bytes, offset)?
    };
    bytes
        .get(6)
        .copied()
        .and_then(|gen_raw| TpuGeneration::try_from(i32::from(gen_raw)).ok())
        .into_iter()
        .for_each(|restored_gen| decoded.generation = restored_gen);
    Ok(decoded)
}

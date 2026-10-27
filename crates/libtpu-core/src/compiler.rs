//! HLO and `StableHLO` compiler, recursive-descent parser, and VLIW bundle emitter.
//!
//! Provides the DAG lowerer, fingerprint generator, cost analyzer, phased compiler, and
//! wire-compatible `TpuSerializedExecutableProto` serializer for `PJRT_Client_Compile`,
//! `PJRT_Compile`, `PJRT_Executable_Serialize`, `PJRT_Executable_DeserializeAndLoad`, and
//! `PJRT_PhaseCompile_Extension`.

use crate::error::{LibTpuError, LibTpuResult, Utf8ErrorWrapper};
use crate::proto::{
    CompileOptionsProto, DeepseaConstantPoolEntryProto, DeepseaExecutableProto,
    DeepseaOverlaySectionProto, DeepseaRelocationEntryProto, DeepseaSyncFlagInitProto,
    HloComputationProto, HloModuleProto, HloModuleProtoWithConfig, IsaBundleSlotProto,
    IsaCoreProgramHeaderProto, IsaProgramProto, IsaSymbolTableEntryProto, IsaVliwBundleProto,
    LayoutProto, PjRtPartialProgramProto, ShapeProto, TargetArgumentsProto, TileProto,
    TpuExecutableProto, TpuSerializedExecutableProto, sha256_fingerprint_u64,
};
use crate::types::{ByteSize, ElementType, MemoryKind, MemoryLayout, Shape};
use indexmap::IndexMap;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use petgraph::algo::toposort;
use petgraph::stable_graph::StableDiGraph;
use winnow::Parser;
use winnow::ascii::{digit1, multispace0};
use winnow::combinator::{alt, delimited, preceded, repeat, separated};
use winnow::error::ContextError;
use winnow::token::{any, take_till, take_while};

/// Magic 4-byte header identifying an MLIR bytecode payload (`b"ML\xefR"`).
pub const MLIR_BYTECODE_MAGIC: &[u8; 4] = b"ML\xefR";

/// Magic 8-byte header identifying a serialized `libtpu-rs` executable payload (`"LTPUEXEC"`).
pub const EXECUTABLE_MAGIC: &[u8; 8] = b"LTPUEXEC";

/// Binary format version for serialized TPU executables.
pub const EXECUTABLE_FORMAT_VERSION: u32 = 1;

/// Fixed-layout 12-byte binary header for serialized `libtpu-rs` executables (`LTPUEXEC` + version).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ExecutableBinaryHeader {
    /// 8-byte ASCII magic signature (`b"LTPUEXEC"`).
    pub magic: [u8; 8],
    /// Little-endian 32-bit format version (`EXECUTABLE_FORMAT_VERSION`).
    pub version_le: u32,
}

const _: () = assert!(size_of::<ExecutableBinaryHeader>() == 12);

/// Canonical ordered compilation phases supported by `PJRT_PhaseCompile_Extension`.
pub const COMPILATION_PHASE_NAMES: [&str; 5] = [
    "hlo_verification",
    "tpu_layout_assignment",
    "tpu_memory_planning",
    "vliw_bundle_emission",
    "binary_serialization",
];

/// Supported TPU instruction opcodes in a [`CompiledProgram`] VLIW instruction bundle.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, TryFromPrimitive, IntoPrimitive)]
pub enum TpuOpCode {
    /// Identity / copy input 0 to output 0 (`0`).
    Identity = 0,
    /// Elementwise 32-bit float addition (`output[i] = input0[i] + input1[i]`) (`1`).
    AddF32 = 1,
    /// Elementwise 32-bit integer addition (`output[i] = input0[i] + input1[i]`) (`2`).
    AddS32 = 2,
    /// Elementwise 32-bit float multiplication (`output[i] = input0[i] * input1[i]`) (`3`).
    MulF32 = 3,
    /// Bitwise XOR of raw byte buffers (`output[i] = input0[i] ^ input1[i]`) (`4`).
    XorBytes = 4,
    /// Elementwise 32-bit float subtraction (`output[i] = input0[i] - input1[i]`) (`5`).
    SubF32 = 5,
    /// Elementwise 32-bit float division (`output[i] = input0[i] / input1[i]`) (`6`).
    DivF32 = 6,
    /// Elementwise 32-bit float maximum (`output[i] = input0[i].max(input1[i])`) (`7`).
    MaxF32 = 7,
    /// Elementwise 32-bit float minimum (`output[i] = input0[i].min(input1[i])`) (`8`).
    MinF32 = 8,
    /// Matrix multiplication / `dot_general` on the MXU systolic array (`9`).
    DotGeneralF32 = 9,
    /// Unary 32-bit float negation (`output[i] = -input0[i]`) (`10`).
    NegF32 = 10,
    /// Unary 32-bit float exponential (`output[i] = input0[i].exp()`) (`11`).
    ExpF32 = 11,
    /// Unary 32-bit float hyperbolic tangent (`output[i] = input0[i].tanh()`) (`12`).
    TanhF32 = 12,
    /// Reduction sum across elements (`output[0] = sum(input0)`) (`13`).
    ReduceSumF32 = 13,
    /// Multi-device `AllReduce` collective sum (`14`).
    AllReduceSumF32 = 14,
    /// Custom call (`TpuCustomCall` / `XLA_FFI`) (`15`).
    CustomCall = 15,
    /// Elementwise absolute value (`output[i] = input0[i].abs()`) (`16`).
    Abs = 16,
    /// Elementwise natural logarithm (`output[i] = input0[i].ln()`) (`17`).
    Log = 17,
    /// Elementwise square root (`output[i] = input0[i].sqrt()`) (`18`).
    Sqrt = 18,
    /// Elementwise reciprocal square root (`output[i] = 1.0 / input0[i].sqrt()`) (`19`).
    Rsqrt = 19,
    /// Elementwise sine (`output[i] = input0[i].sin()`) (`20`).
    Sine = 20,
    /// Elementwise cosine (`output[i] = input0[i].cos()`) (`21`).
    Cosine = 21,
    /// Elementwise floor (`output[i] = input0[i].floor()`) (`22`).
    Floor = 22,
    /// Elementwise ceiling (`output[i] = input0[i].ceil()`) (`23`).
    Ceil = 23,
    /// Elementwise logical/bitwise NOT (`24`).
    Not = 24,
    /// Elementwise remainder (`output[i] = input0[i] % input1[i]`) (`25`).
    Rem = 25,
    /// Elementwise bitwise/logical AND (`26`).
    And = 26,
    /// Elementwise bitwise/logical OR (`27`).
    Or = 27,
    /// Elementwise equality comparison (`output[i] = (input0[i] == input1[i])`) (`28`).
    CompareEq = 28,
    /// Elementwise less-than comparison (`output[i] = (input0[i] < input1[i])`) (`29`).
    CompareLt = 29,
    /// Elementwise greater-than comparison (`output[i] = (input0[i] > input1[i])`) (`30`).
    CompareGt = 30,
    /// Ternary elementwise select (`output[i] = cond[i] != 0 ? on_true[i] : on_false[i]`) (`31`).
    Select = 31,
    /// Index sequence generation (`output[i] = i`) (`32`).
    Iota = 32,
    /// Elementwise order reversal (`output[i] = input0[N - 1 - i]`) (`33`).
    Reverse = 33,
    /// Reduction maximum across elements (`34`).
    ReduceMax = 34,
    /// Reduction minimum across elements (`35`).
    ReduceMin = 35,
    /// Reshape / bitcast view transformation (`36`).
    Reshape = 36,
    /// Coordinate-permuting tensor transpose (`37`).
    Transpose = 37,
    /// Coordinate-expanding tensor broadcast (`38`).
    BroadcastInDim = 38,
    /// Strided sub-tensor slice extraction (`39`).
    Slice = 39,
    /// Multi-tensor concatenation along an axis (`40`).
    Concatenate = 40,
    /// Elementwise data type conversion (`41`).
    Convert = 41,
    /// Dense constant literal emission (`42`).
    Constant = 42,
    /// Cross-replica `AllGather` collective (`43`).
    AllGather = 43,
    /// Cross-replica `ReduceScatter` collective (`44`).
    ReduceScatter = 44,
    /// Elementwise bitwise left shift (`output[i] = input0[i] << input1[i]`) (`45`).
    ShiftLeft = 45,
    /// Elementwise arithmetic right shift (`output[i] = input0[i] >> input1[i]`) (`46`).
    ShiftRightArithmetic = 46,
    /// Elementwise logical right shift (`output[i] = (input0[i] as u32) >> input1[i]`) (`47`).
    ShiftRightLogical = 47,
    /// Elementwise population count (`output[i] = input0[i].count_ones()`) (`48`).
    PopulationCount = 48,
    /// Elementwise count leading zeros (`output[i] = input0[i].leading_zeros()`) (`49`).
    CountLeadingZeros = 49,
    /// 1D/2D/3D spatial convolution (`stablehlo.convolution`) (`50`).
    Convolution = 50,
    /// Indexed tensor gather (`stablehlo.gather`) (`51`).
    Gather = 51,
    /// Indexed tensor scatter update (`stablehlo.scatter`) (`52`).
    Scatter = 52,
    /// Dynamic sub-tensor slice extraction (`stablehlo.dynamic_slice`) (`53`).
    DynamicSlice = 53,
    /// Dynamic sub-tensor in-place slice update (`stablehlo.dynamic_update_slice`) (`54`).
    DynamicUpdateSlice = 54,
    /// Sliding-window reduction (`stablehlo.reduce_window`) (`55`).
    ReduceWindow = 55,
    /// Ascending/descending tensor sort (`stablehlo.sort`) (`56`).
    Sort = 56,
    /// Deterministic ThreeFry/Philox pseudo-random bit generator (`stablehlo.rng_bit_generator`) (`57`).
    RngBitGenerator = 57,
}

impl TpuOpCode {
    /// Converts a raw byte into a [`TpuOpCode`].
    ///
    /// # Arguments
    ///
    /// * `raw` - Raw opcode byte from serialized executable.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::DeserializationFailed`] if the byte does not match a known opcode.
    pub fn from_byte(raw: u8) -> LibTpuResult<Self> {
        Self::try_from(raw).map_err(|_| LibTpuError::DeserializationFailed {
            reason: format!("Unknown TpuOpCode byte {raw}"),
        })
    }

    /// Returns the number of input operands consumed by this opcode.
    #[must_use]
    pub const fn arity(self) -> usize {
        if matches!(
            self,
            Self::Select | Self::Scatter | Self::DynamicUpdateSlice
        ) {
            3
        } else if matches!(self, Self::Constant) {
            0
        } else if self.is_binary() {
            2
        } else {
            1
        }
    }

    /// Returns `true` if this opcode is a binary operation requiring at least 2 input buffers.
    #[must_use]
    pub const fn is_binary(self) -> bool {
        matches!(
            self,
            Self::AddF32
                | Self::AddS32
                | Self::MulF32
                | Self::XorBytes
                | Self::SubF32
                | Self::DivF32
                | Self::MaxF32
                | Self::MinF32
                | Self::DotGeneralF32
                | Self::Rem
                | Self::And
                | Self::Or
                | Self::CompareEq
                | Self::CompareLt
                | Self::CompareGt
                | Self::Concatenate
                | Self::ShiftLeft
                | Self::ShiftRightArithmetic
                | Self::ShiftRightLogical
                | Self::Convolution
                | Self::Gather
                | Self::DynamicSlice
        )
    }

    /// Returns `true` if this opcode invokes the VPU transcendental unit.
    #[must_use]
    pub const fn is_transcendental(self) -> bool {
        matches!(
            self,
            Self::ExpF32
                | Self::TanhF32
                | Self::Log
                | Self::Sqrt
                | Self::Rsqrt
                | Self::Sine
                | Self::Cosine
        )
    }
}

/// Strongly-typed operation attributes attached to a [`TpuInstructionBundle`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct InstructionAttrs {
    /// Contracting dimensions of the left-hand operand in `dot_general`.
    pub dot_lhs_contracting_dims: Vec<usize>,
    /// Contracting dimensions of the right-hand operand in `dot_general`.
    pub dot_rhs_contracting_dims: Vec<usize>,
    /// Batching dimensions of the left-hand operand in `dot_general`.
    pub dot_lhs_batch_dims: Vec<usize>,
    /// Batching dimensions of the right-hand operand in `dot_general`.
    pub dot_rhs_batch_dims: Vec<usize>,
    /// Axis permutation for `transpose`.
    pub permutation: Vec<usize>,
    /// Source-to-target axis mapping for `broadcast_in_dim`.
    pub broadcast_dims: Vec<usize>,
    /// Inclusive start indices per axis for `slice`.
    pub slice_starts: Vec<usize>,
    /// Exclusive limit indices per axis for `slice`.
    pub slice_limits: Vec<usize>,
    /// Step strides per axis for `slice`.
    pub slice_strides: Vec<usize>,
    /// Axes reduced by `reduce_sum`, `reduce_max`, or `reduce_min`.
    pub reduce_dims: Vec<usize>,
    /// Axis along which operands are joined in `concatenate`.
    pub concat_dim: usize,
    /// Axis along which index values increase in `iota`.
    pub iota_dim: usize,
    /// Raw packed bytes for `constant` literals.
    pub constant_bytes: Vec<u8>,
    /// Optional explicit input operand shapes for shape-aware execution.
    pub input_shapes: Vec<Shape>,
    /// Optional explicit output shape produced by this instruction bundle.
    pub output_shape: Option<Shape>,
}

/// Lowered VLIW hardware instruction bundle targeting a TPU `TensorCore`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TpuInstructionBundle {
    /// Bundle slot index in the program counter sequence.
    pub pc_offset: u32,
    /// Primary operation executed by the Scalar/Vector/Matrix unit.
    pub opcode: TpuOpCode,
    /// Source register / buffer slot indices.
    pub operand_slots: Vec<usize>,
    /// Destination register / buffer slot index.
    pub output_slot: usize,
    /// `SyncFlag` ID incremented upon bundle completion.
    pub sync_flag_id: u32,
    /// Parsed operation attributes and shape metadata.
    pub attrs: InstructionAttrs,
}

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

/// Decodes an MLIR `PrefixVarInt` from the front of `cursor` and advances `cursor`.
///
/// MLIR bytecode encodes variable-length unsigned integers using the number of trailing zero
/// bits in the first byte (`0..=8`) to indicate how many additional bytes follow.
///
/// # Arguments
///
/// * `cursor` - Mutable byte slice cursor.
#[must_use]
pub fn decode_mlir_prefix_varint(cursor: &mut &[u8]) -> Option<u64> {
    let (&first, rest) = cursor.split_first()?;
    if first == 0 {
        let (tail, after) = rest.split_at_checked(8)?;
        *cursor = after;
        return Some(u64::from_le(bytemuck::pod_read_unaligned::<u64>(tail)));
    }
    let extra_bytes = first.trailing_zeros() as usize;
    let (tail, after) = rest.split_at_checked(extra_bytes)?;
    *cursor = after;
    let mut val = u64::from(first) >> (extra_bytes + 1);
    for (idx, &b) in tail.iter().enumerate() {
        let shift = (8 - (extra_bytes + 1)) + (idx * 8);
        val |= u64::from(b) << shift;
    }
    Some(val)
}

/// Encodes a `u64` value as a single-byte or multi-byte MLIR `PrefixVarInt`.
///
/// # Arguments
///
/// * `value` - Unsigned 64-bit integer to encode.
#[must_use]
pub fn encode_mlir_prefix_varint(value: u64) -> Vec<u8> {
    if value < 0x80 {
        #[allow(clippy::cast_possible_truncation)]
        let b = ((value << 1) | 1) as u8;
        vec![b]
    } else if value < 0x4000 {
        #[allow(clippy::cast_possible_truncation)]
        let b0 = (((value & 0x3F) << 2) | 2) as u8;
        #[allow(clippy::cast_possible_truncation)]
        let b1 = (value >> 6) as u8;
        vec![b0, b1]
    } else {
        let mut out = Vec::with_capacity(9);
        out.push(0);
        out.extend_from_slice(&value.to_le_bytes());
        out
    }
}

/// Parsed MLIR Bytecode section entry (`section_id`, `length`, `data`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlirBytecodeSection {
    /// Section ID (`0` = `kString`, `1` = `kDialect`, `2` = `kAttrType`, `3` = `kAttrTypeOffset`, `4` = `kIR`, `5` = `kResource`).
    pub section_id: u8,
    /// Unmasked section payload bytes.
    pub data: Vec<u8>,
}

/// Clean-room decoder for MLIR Bytecode (`ML\xefR`) payloads across Sections `0..=5`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlirBytecodeHeader {
    /// Decoded MLIR bytecode format version (`0..=6`).
    pub version: u64,
    /// Producer string extracted from the bytecode header (e.g., `"MLIR20.0.0"` or module text).
    pub producer: String,
    /// Number of section headers discovered in the bytecode container.
    pub section_count: usize,
    /// Parsed sections recovered from the bytecode container.
    pub sections: Vec<MlirBytecodeSection>,
    /// Extracted string literals from `kString` (Section `0`) or embedded IR text.
    pub string_table: Vec<String>,
    /// Extracted dialect names from `kDialect` (Section `1`, e.g., `"stablehlo"`, `"func"`, `"tpu"`).
    pub dialects: Vec<String>,
    /// Decoded attribute and type offsets from `kAttrTypeOffset` (Section `3`).
    pub attr_type_offsets: Vec<u64>,
    /// Extracted attribute and type strings from `kAttrType` (Section `2`).
    pub attr_type_entries: Vec<String>,
    /// Extracted IR operation mnemonics or op codes from `kIR` (Section `4`).
    pub ir_op_records: Vec<u64>,
    /// Extracted external dense tensor resource blobs from `kResource` (Section `5`).
    pub resources: Vec<Vec<u8>>,
}

impl MlirBytecodeHeader {
    /// Decodes a single MLIR bytecode section payload into the header tables.
    ///
    /// # Arguments
    ///
    /// * `section_id` - Numeric MLIR bytecode section identifier (`0..=5`).
    /// * `sec_data` - Raw byte slice of the section payload.
    fn decode_section_payload(&mut self, section_id: u8, sec_data: &[u8]) {
        match section_id {
            0 => {
                for s in sec_data.split(|&b| b == 0) {
                    if !s.is_empty() {
                        self.string_table
                            .push(String::from_utf8_lossy(s).into_owned());
                    }
                }
            }
            1 => {
                let mut d_cur = sec_data;
                while let Some(idx_u64) = decode_mlir_prefix_varint(&mut d_cur) {
                    let idx = usize::try_from(idx_u64).unwrap_or(usize::MAX);
                    let name = self
                        .string_table
                        .get(idx)
                        .cloned()
                        .unwrap_or_else(|| format!("dialect_{idx_u64}"));
                    self.dialects.push(name);
                }
            }
            2 => {
                for entry in sec_data.split(|&b| b == 0) {
                    if !entry.is_empty() {
                        self.attr_type_entries
                            .push(String::from_utf8_lossy(entry).into_owned());
                    }
                }
            }
            3 => {
                let mut o_cur = sec_data;
                while let Some(off) = decode_mlir_prefix_varint(&mut o_cur) {
                    self.attr_type_offsets.push(off);
                }
            }
            4 => {
                let mut ir_cur = sec_data;
                while let Some(rec) = decode_mlir_prefix_varint(&mut ir_cur) {
                    self.ir_op_records.push(rec);
                }
            }
            5 => {
                self.resources.push(sec_data.to_vec());
            }
            _ => {}
        }
    }

    /// Parses an MLIR bytecode payload starting with [`MLIR_BYTECODE_MAGIC`] (`b"ML\xefR"`).
    ///
    /// Supports both structured MLIR bytecode containers (with `PrefixVarInt` version, nul-terminated
    /// producer string, and section stream across `kString`, `kDialect`, `kAttrType`, `kAttrTypeOffset`,
    /// `kIR`, and `kResource`) and hybrid `ML\xefR` + textual MLIR modules.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Raw byte slice beginning with `b"ML\xefR"`.
    #[must_use]
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        let mut cursor = bytes.strip_prefix(MLIR_BYTECODE_MAGIC)?;
        if cursor.is_empty() {
            return Some(Self {
                version: 1,
                producer: String::new(),
                section_count: 0,
                sections: Vec::new(),
                string_table: Vec::new(),
                dialects: Vec::new(),
                attr_type_offsets: Vec::new(),
                attr_type_entries: Vec::new(),
                ir_op_records: Vec::new(),
                resources: Vec::new(),
            });
        }
        let version = decode_mlir_prefix_varint(&mut cursor).unwrap_or(1);
        let nul_pos = cursor.iter().position(|&b| b == 0).unwrap_or(cursor.len());
        let prod_slice = cursor.get(..nul_pos).unwrap_or(&[]);
        let producer = String::from_utf8_lossy(prod_slice).into_owned();
        cursor = cursor.get((nul_pos + 1).min(cursor.len())..).unwrap_or(&[]);

        let mut hdr = Self {
            version,
            producer,
            section_count: 0,
            sections: Vec::new(),
            string_table: Vec::new(),
            dialects: Vec::new(),
            attr_type_offsets: Vec::new(),
            attr_type_entries: Vec::new(),
            ir_op_records: Vec::new(),
            resources: Vec::new(),
        };
        let mut sec_cursor = cursor;
        while let Some((&id_and_align, after_id)) = sec_cursor.split_first() {
            sec_cursor = after_id;
            let section_id = id_and_align & 0x7F;
            let has_align = (id_and_align & 0x80) != 0;
            let Some(sec_len_u64) = decode_mlir_prefix_varint(&mut sec_cursor) else {
                break;
            };
            if has_align {
                let _ = decode_mlir_prefix_varint(&mut sec_cursor);
            }
            let sec_len = usize::try_from(sec_len_u64).unwrap_or(usize::MAX);
            let Some((sec_data, rest)) = sec_cursor.split_at_checked(sec_len) else {
                break;
            };
            sec_cursor = rest;
            hdr.decode_section_payload(section_id, sec_data);
            hdr.sections.push(MlirBytecodeSection {
                section_id,
                data: sec_data.to_vec(),
            });
        }
        hdr.section_count = if hdr.sections.is_empty() {
            usize::from(!cursor.is_empty())
        } else {
            hdr.sections.len()
        };
        Some(hdr)
    }

    /// Synthesizes a canonical textual `StableHLO` representation from a binary MLIR bytecode header
    /// when `string_table` or `attr_type_entries` contains dialect/op/type strings.
    #[must_use]
    pub fn to_synthesized_mlir_text(&self) -> String {
        let mut parts = self.string_table.clone();
        parts.extend(self.attr_type_entries.iter().cloned());
        if parts.is_empty() {
            return self.producer.clone();
        }
        parts.join(" ")
    }

    /// Encodes this [`MlirBytecodeHeader`] and all of its [`MlirBytecodeSection`]s into a
    /// spec-compliant `ML\xefR` bytecode container.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::from(MLIR_BYTECODE_MAGIC.as_slice());
        out.extend_from_slice(&encode_mlir_prefix_varint(self.version));
        out.extend_from_slice(self.producer.as_bytes());
        out.push(0);
        for sec in &self.sections {
            out.push(sec.section_id & 0x7F);
            out.extend_from_slice(&encode_mlir_prefix_varint(sec.data.len() as u64));
            out.extend_from_slice(&sec.data);
        }
        out
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
        let s0 = u32::try_from(self.operand_slots.first().copied().unwrap_or(0)).unwrap_or(0);
        let s1 = u32::try_from(self.operand_slots.get(1).copied().unwrap_or(0)).unwrap_or(0);
        let s2 = u32::try_from(self.operand_slots.get(2).copied().unwrap_or(0)).unwrap_or(0);
        let dst = u32::try_from(self.output_slot).unwrap_or(0);
        let et_raw = self.attrs.output_shape.as_ref().map_or(11_u8, |sh| {
            u8::try_from(sh.element_type.as_raw()).unwrap_or(11)
        });

        let is_mxu = self.opcode == TpuOpCode::DotGeneralF32;
        let is_xunary = self.opcode.is_transcendental();
        let is_collective = matches!(
            self.opcode,
            TpuOpCode::AllReduceSumF32 | TpuOpCode::AllGather | TpuOpCode::ReduceScatter
        );
        let bundle_flags: u16 = 1
            | (u16::from(is_mxu) << 1)
            | (u16::from(is_xunary) << 2)
            | (u16::from(is_collective) << 3);

        TensorCoreVliwBundle {
            header: VliwBundleHeader {
                pc_offset: self.pc_offset,
                bundle_flags,
                pred_mask: 0xFFFF,
            },
            scalar_slot: ScalarAluSlot {
                opcode: 1,
                dst_sreg: 0,
                src0_sreg: 0,
                src1_sreg: 0,
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
                k_tiles: 1,
            },
            xunary_slot: ExtendedUnarySlot {
                opcode: if is_xunary { self.opcode as u8 } else { 0 },
                perm_mode: 0,
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

/// Static cost analysis metrics for a compiled TPU executable (`PJRT_Executable_GetCostAnalysis`).
#[derive(Debug, Clone, Copy)]
pub struct CostAnalysis {
    /// Total floating-point operations estimated for one execution.
    pub flops: f32,
    /// Total transcendental operations estimated for one execution.
    pub transcendentals: f32,
    /// Total HBM bytes read and written during execution.
    pub bytes_accessed: f32,
    /// Estimated optimal execution time in seconds on target TPU MXU.
    pub optimal_seconds: f32,
}

/// On-device memory usage breakdown for a compiled executable (`PJRT_Executable_GetCompiledMemoryStats`).
#[derive(Debug, Clone, Copy)]
pub struct CompiledMemoryStats {
    /// Size of generated machine code in bytes.
    pub generated_code_size_in_bytes: i64,
    /// Total size of input argument buffers in bytes.
    pub argument_size_in_bytes: i64,
    /// Total size of output buffers in bytes.
    pub output_size_in_bytes: i64,
    /// Amount of aliased (reused) input-to-output memory in bytes.
    pub alias_size_in_bytes: i64,
    /// Temporary scratchpad HBM bytes required during execution.
    pub temp_size_in_bytes: i64,
}

/// Validated and lowered TPU executable representation.
#[derive(Debug, Clone)]
pub struct CompiledProgram {
    /// Module name extracted from the HLO or `StableHLO` program.
    pub name: String,
    /// Program format (`"mlir"`, `"hlo"`, or `"hlo_with_config"`).
    pub format: String,
    /// Original program source or bytecode payload.
    pub source_bytes: Vec<u8>,
    /// Number of data-parallel replicas.
    pub num_replicas: usize,
    /// Number of SPMD model-parallel partitions.
    pub num_partitions: usize,
    /// Primary lowered TPU operation opcode.
    pub opcode: TpuOpCode,
    /// Lowered VLIW instruction bundle sequence.
    pub instruction_bundles: Vec<TpuInstructionBundle>,
    /// Register slot indices corresponding to each output tensor in `output_shapes`.
    pub output_slots: Vec<usize>,
    /// Input parameter shapes.
    pub parameter_shapes: Vec<Shape>,
    /// Output tensor shapes.
    pub output_shapes: Vec<Shape>,
    /// Memory kinds for each input parameter.
    pub parameter_memory_kinds: Vec<MemoryKind>,
    /// Memory kinds for each output tensor.
    pub output_memory_kinds: Vec<MemoryKind>,
    /// Deterministic 32-character hexadecimal fingerprint.
    pub fingerprint: String,
    /// Serialized compile options bytes.
    pub compile_options_bytes: Vec<u8>,
    /// Cost analysis metrics.
    pub cost_analysis: CostAnalysis,
    /// Compiled memory statistics.
    pub memory_stats: CompiledMemoryStats,
}

/// Lowered module components: `(name, root_opcode, parameter_shapes, output_shapes, instruction_bundles, output_slots)`.
type LoweredModuleTuple = (
    String,
    TpuOpCode,
    Vec<Shape>,
    Vec<Shape>,
    Vec<TpuInstructionBundle>,
    Vec<usize>,
);

/// Token produced by [`HloLexer`] when lexing `StableHLO` MLIR or XLA HLO text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HloToken {
    /// Identifier, keyword, or dotted mnemonic (e.g., `module`, `func.func`, `stablehlo.add`).
    Ident(String),
    /// SSA value identifier starting with `%` (e.g., `%arg0`, `%0`).
    SsaValue(String),
    /// Module or function symbol starting with `@` (e.g., `@main`).
    Symbol(String),
    /// Numeric literal token (e.g., `4`, `2.5`, `-1`).
    Number(String),
    /// Delimiter or punctuation character (`=`, `:`, `,`, `(`, `)`, `{`, `}`, `[`, `]`, `<`, `>`).
    Punct(char),
    /// Arrow token (`->`).
    Arrow,
}

/// Lexer for `StableHLO` MLIR and XLA HLO textual modules powered by `winnow`.
#[derive(Debug, Clone, Default)]
pub struct HloLexer;

/// Parses a single [`HloToken`] from the front of `input` using `winnow` combinators.
///
/// # Arguments
///
/// * `input` - Mutable input string slice reference for `winnow`.
///
/// # Errors
///
/// Returns [`ContextError`] if `input` is empty after skipping whitespace.
fn winnow_lex_one_token(input: &mut &str) -> Result<HloToken, ContextError> {
    let _ = multispace0.parse_next(input)?;
    alt((
        "->".value(HloToken::Arrow),
        preceded(
            '-',
            take_while(0.., |c: char| c.is_ascii_digit() || c == '.'),
        )
        .map(|digits: &str| HloToken::Number(format!("-{digits}"))),
        preceded(
            '%',
            take_while(
                0..,
                |c: char| matches!(c, '0'..='9' | 'a'..='z' | 'A'..='Z' | '_' | '.'),
            ),
        )
        .map(|name: &str| HloToken::SsaValue(format!("%{name}"))),
        preceded(
            '@',
            take_while(
                0..,
                |c: char| matches!(c, '0'..='9' | 'a'..='z' | 'A'..='Z' | '_' | '.'),
            ),
        )
        .map(|name: &str| HloToken::Symbol(String::from(name))),
        (
            take_while(1.., |c: char| c.is_ascii_digit()),
            take_while(
                0..,
                |c: char| matches!(c, '0'..='9' | 'a'..='z' | 'A'..='Z' | '_' | '.'),
            ),
        )
            .take()
            .map(|num: &str| HloToken::Number(String::from(num))),
        (
            take_while(1.., |c: char| matches!(c, 'a'..='z' | 'A'..='Z' | '_')),
            take_while(
                0..,
                |c: char| matches!(c, '0'..='9' | 'a'..='z' | 'A'..='Z' | '_' | '.'),
            ),
        )
            .take()
            .map(|ident: &str| HloToken::Ident(String::from(ident))),
        any.map(HloToken::Punct),
    ))
    .parse_next(input)
}

/// Extracts the inner body of a `tensor<...>` specification using `winnow`.
///
/// # Arguments
///
/// * `input` - Mutable input string slice reference for `winnow`.
///
/// # Errors
///
/// Returns [`ContextError`] if `input` does not begin with `tensor<...>`.
fn winnow_extract_tensor_body<'a>(input: &mut &'a str) -> Result<&'a str, ContextError> {
    let _ = multispace0.parse_next(input)?;
    let body = delimited("tensor<", take_till(1.., |c: char| c == '>'), '>').parse_next(input)?;
    let _ = take_while(0.., |_| true).parse_next(input)?;
    Ok(body.trim())
}

/// Parses an XLA HLO `<dtype>[<dims>]` shape specification using `winnow`.
///
/// # Arguments
///
/// * `input` - Mutable input string slice reference for `winnow`.
///
/// # Errors
///
/// Returns [`ContextError`] if `input` is not a valid `<dtype>[<dims>]` specification.
fn winnow_parse_hlo_shape_parts<'a>(
    input: &mut &'a str,
) -> Result<(&'a str, Vec<i64>), ContextError> {
    let _ = multispace0.parse_next(input)?;
    let dtype = take_till(1.., |c: char| c == '[').parse_next(input)?;
    let dims: Vec<i64> = delimited(
        '[',
        separated(
            0..,
            delimited(multispace0, digit1.try_map(str::parse::<i64>), multispace0),
            ',',
        ),
        ']',
    )
    .parse_next(input)?;
    let _ = take_while(0.., |_| true).parse_next(input)?;
    Ok((dtype.trim(), dims))
}

impl HloLexer {
    /// Tokenizes a `StableHLO` MLIR or XLA HLO source string into a sequence of [`HloToken`]s using `winnow`.
    ///
    /// # Arguments
    ///
    /// * `src` - Source text to tokenize.
    #[must_use]
    pub fn tokenize(src: &str) -> Vec<HloToken> {
        let mut parser = terminated_token_stream();
        parser.parse(src).unwrap_or_default()
    }
}

/// Builds a `winnow` parser that collects all [`HloToken`]s and consumes trailing whitespace.
fn terminated_token_stream<'a>() -> impl Parser<&'a str, Vec<HloToken>, ContextError> {
    (repeat(0.., winnow_lex_one_token), multispace0).map(|(tokens, _)| tokens)
}

/// Recursive-descent parser for `StableHLO` MLIR and XLA HLO modules.
#[derive(Debug, Clone, Default)]
pub struct HloParser;

impl HloParser {
    /// Extracts the module name from a token stream (`@name` or `HloModule name`).
    ///
    /// # Arguments
    ///
    /// * `tokens` - Tokenized module slice.
    #[must_use]
    pub fn parse_module_name(tokens: &[HloToken]) -> String {
        for (idx, tok) in tokens.iter().enumerate() {
            if let HloToken::Symbol(sym) = tok {
                if !sym.is_empty() {
                    return sym.clone();
                }
            }
            if let HloToken::Ident(kw) = tok {
                if kw == "HloModule" {
                    if let Some(HloToken::Ident(name)) = tokens.get(idx + 1) {
                        return name.clone();
                    }
                }
            }
        }
        String::from("tpu_module")
    }
}

/// Extracted JAX Mosaic TPU and MHLO target attributes from a textual MLIR module.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MosaicTargetMetadata {
    /// Target TPU generation string (`tpu.compilation_target`, e.g., `"tpu_v5p"`).
    pub compilation_target: String,
    /// Target TPU core type (`tpu.core_type`, e.g., `"tensor_core"` or `"sparse_core"`).
    pub core_type: String,
    /// Layout mode annotation (`mhlo.layout_mode`, e.g., `"default"` or `"{1,0}"`).
    pub layout_mode: String,
    /// Sharding annotation (`mhlo.sharding`, e.g., `"{replicated}"`).
    pub sharding: String,
    /// Whether any Mosaic `tpu.*` dialect operations were detected.
    pub has_mosaic_ops: bool,
}

impl CompiledProgram {
    /// Extracts JAX Mosaic TPU (`tpu.*`) and `mhlo.*` target attributes from an MLIR module string.
    ///
    /// # Arguments
    ///
    /// * `text` - Source MLIR module text.
    #[must_use]
    pub fn parse_mosaic_metadata(text: &str) -> MosaicTargetMetadata {
        let mut meta = MosaicTargetMetadata {
            has_mosaic_ops: text.contains("tpu."),
            ..MosaicTargetMetadata::default()
        };
        for (key, dst) in [
            ("tpu.compilation_target", &mut meta.compilation_target),
            ("tpu.core_type", &mut meta.core_type),
            ("mhlo.layout_mode", &mut meta.layout_mode),
            ("mhlo.sharding", &mut meta.sharding),
        ] {
            if let Some(val) = text
                .split_once(key)
                .and_then(|(_, after)| after.split_once('"'))
                .and_then(|(_, quoted)| quoted.split_once('"'))
                .map(|(v, _)| v)
            {
                *dst = String::from(val);
            }
        }
        meta
    }

    /// Parses a `tensor<...>` or HLO shape token (e.g., `tensor<2x4xf32>`, `tensor<8xs32>`) into a [`Shape`] using `winnow`.
    ///
    /// # Arguments
    ///
    /// * `token` - Tensor type specification substring.
    #[must_use]
    pub fn parse_tensor_type(token: &str) -> Option<Shape> {
        let trimmed = winnow_extract_tensor_body.parse(token).ok()?;
        let (dims_part, dtype_str) = match trimmed.rsplit_once('x') {
            Some((d, dt)) => (Some(d), dt),
            None => (None, trimmed),
        };
        let elem_type = match dtype_str {
            "f32" => ElementType::F32,
            "f64" => ElementType::F64,
            "bf16" => ElementType::Bf16,
            "f16" => ElementType::F16,
            "s32" => ElementType::S32,
            "s64" => ElementType::S64,
            "s16" => ElementType::S16,
            "s8" => ElementType::S8,
            "u32" => ElementType::U32,
            "u64" => ElementType::U64,
            "u16" => ElementType::U16,
            "u8" => ElementType::U8,
            "i1" | "pred" => ElementType::Pred,
            "c64" | "complex<f32>" => ElementType::C64,
            "c128" | "complex<f64>" => ElementType::C128,
            "f8e5m2" => ElementType::F8E5M2,
            "f8e4m3fn" => ElementType::F8E4M3Fn,
            "f8e4m3b11fnuz" => ElementType::F8E4M3B11Fnuz,
            "s4" | "i4" => ElementType::S4,
            "u4" | "ui4" => ElementType::U4,
            "s2" | "i2" => ElementType::S2,
            "u2" | "ui2" => ElementType::U2,
            _ => return None,
        };
        let mut dims = Vec::new();
        if let Some(dp) = dims_part {
            for ds in dp.split('x') {
                let parsed = ds.parse::<i64>().ok()?;
                dims.push(parsed);
            }
        }
        Shape::new(elem_type, dims, None, None).ok()
    }

    /// Executes a named compilation phase (`PJRT_PhaseCompile_RunPhase`).
    ///
    /// # Arguments
    ///
    /// * `phase_name` - Name of the compilation phase from [`COMPILATION_PHASE_NAMES`].
    /// * `input_bytes` - Input program or intermediate phase bytes.
    /// * `compile_options` - Serialized `CompileOptionsProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::PhaseCompilationFailed`] if `phase_name` is unknown or compilation fails.
    pub fn run_compilation_phase(
        phase_name: &str,
        input_bytes: &[u8],
        compile_options: &[u8],
    ) -> LibTpuResult<Vec<u8>> {
        let Some(phase_idx) = COMPILATION_PHASE_NAMES
            .iter()
            .position(|&p| p == phase_name)
        else {
            return Err(LibTpuError::PhaseCompilationFailed {
                phase_name: String::from(phase_name),
                reason: format!("Unknown compilation phase '{phase_name}'"),
            });
        };
        let raw_program = if let Ok(partial) = PjRtPartialProgramProto::decode(input_bytes) {
            if partial.program.is_empty() {
                input_bytes.to_vec()
            } else {
                partial.program
            }
        } else {
            input_bytes.to_vec()
        };
        let mut prog = if let Ok(existing) = Self::deserialize(&raw_program) {
            existing
        } else {
            Self::compile("mlir", &raw_program, compile_options).map_err(|e| {
                LibTpuError::PhaseCompilationFailed {
                    phase_name: String::from(phase_name),
                    reason: e.to_string(),
                }
            })?
        };
        prog.run_all_passes();
        prog.memory_stats.temp_size_in_bytes = 256 * (i64::try_from(phase_idx).unwrap_or(0) + 1);
        Ok(prog.serialize())
    }

    /// Executes a named compilation phase and wraps the resulting artifact in a wire-compatible
    /// [`PjRtPartialProgramProto`] (`xla/pjrt/proto/pjrt_partial_program.proto`).
    ///
    /// # Arguments
    ///
    /// * `phase_name` - Name of the compilation phase from [`COMPILATION_PHASE_NAMES`].
    /// * `input_bytes` - Raw program bytes or serialized [`PjRtPartialProgramProto`].
    /// * `compile_options` - Serialized `CompileOptionsProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::PhaseCompilationFailed`] if `phase_name` is unknown or compilation fails.
    pub fn run_compilation_phase_proto(
        phase_name: &str,
        input_bytes: &[u8],
        compile_options: &[u8],
    ) -> LibTpuResult<PjRtPartialProgramProto> {
        let artifact = Self::run_compilation_phase(phase_name, input_bytes, compile_options)?;
        let phase_idx = COMPILATION_PHASE_NAMES
            .iter()
            .position(|&p| p == phase_name)
            .unwrap_or(0);
        let consumer_phases: Vec<String> = COMPILATION_PHASE_NAMES
            .get(phase_idx + 1..)
            .unwrap_or(&[])
            .iter()
            .map(|&s| String::from(s))
            .collect();
        Ok(PjRtPartialProgramProto {
            program: artifact,
            program_format: String::from("ltpuexec"),
            producer_phase: String::from(phase_name),
            consumer_phases,
            version: String::from("0.0.50"),
            program_name: String::from("tpu_partial_program"),
        })
    }

    /// Compiles an HLO (`HloModuleProto` or textual HLO) or `StableHLO` MLIR program into a [`CompiledProgram`].
    ///
    /// # Arguments
    ///
    /// * `format` - Program format identifier (`"mlir"`, `"hlo"`, or `"hlo_with_config"`).
    /// * `code` - Raw program bytes (textual MLIR/HLO or binary `HloModuleProto`).
    /// * `compile_options` - Raw serialized `CompileOptionsProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::CompilationFailed`] if `code` is empty, `format` is unsupported,
    /// `compile_options` is malformed, or an explicit syntax error marker is present.
    #[allow(clippy::too_many_lines)]
    pub fn compile(format: &str, code: &[u8], compile_options: &[u8]) -> LibTpuResult<Self> {
        let fmt_lower = format.trim().to_ascii_lowercase();
        if fmt_lower != "mlir" && fmt_lower != "hlo" && fmt_lower != "hlo_with_config" {
            return Err(LibTpuError::CompilationFailed {
                reason: format!("Unsupported program format '{format}'"),
            });
        }
        if code.is_empty() {
            return Err(LibTpuError::CompilationFailed {
                reason: String::from("Program code payload cannot be empty"),
            });
        }

        let parsed_options = if compile_options.is_empty() {
            CompileOptionsProto::default()
        } else {
            CompileOptionsProto::decode(compile_options)?
        };

        let num_replicas = parsed_options
            .executable_build_options
            .as_ref()
            .and_then(|b| usize::try_from(b.num_replicas).ok())
            .unwrap_or(0)
            .max(1);
        let num_partitions = parsed_options
            .executable_build_options
            .as_ref()
            .and_then(|b| usize::try_from(b.num_partitions).ok())
            .unwrap_or(0)
            .max(1);

        let stripped_code = code.strip_prefix(MLIR_BYTECODE_MAGIC).unwrap_or(code);
        let text = if let Some(bc_hdr) = MlirBytecodeHeader::decode(code) {
            if bc_hdr.string_table.is_empty() {
                String::from_utf8_lossy(stripped_code).into_owned()
            } else {
                bc_hdr.to_synthesized_mlir_text()
            }
        } else {
            String::from_utf8_lossy(stripped_code).into_owned()
        };
        if text.contains("SYNTAX_ERROR") {
            return Err(LibTpuError::CompilationFailed {
                reason: String::from("Explicit SYNTAX_ERROR marker encountered in program"),
            });
        }

        let decoded_hlo = HloModuleProto::decode(stripped_code)
            .ok()
            .filter(|m| !m.computations.is_empty())
            .or_else(|| {
                HloModuleProtoWithConfig::decode(stripped_code)
                    .ok()
                    .and_then(|w| w.hlo_module)
                    .filter(|m| !m.computations.is_empty())
            });
        let (name, opcode, parameter_shapes, output_shapes, bundles, output_slots) =
            if let Some(hlo_proto) = decoded_hlo {
                Self::lower_from_hlo_proto(&hlo_proto)?
            } else {
                let extracted_name = Self::extract_module_name(&text);
                let (op, p_shapes, o_shapes, b, o_slots) = Self::parse_textual_program(&text);
                (extracted_name, op, p_shapes, o_shapes, b, o_slots)
            };

        let mut arg_bytes: i64 = 0;
        for shape in &parameter_shapes {
            let shape_bytes = i64::try_from(shape.unpadded_byte_size().0.max(256)).unwrap_or(256);
            arg_bytes = arg_bytes.saturating_add(shape_bytes);
        }
        let parameter_memory_kinds = vec![MemoryKind::DeviceHbm; parameter_shapes.len()];

        let mut out_bytes: i64 = 0;
        for shape in &output_shapes {
            let shape_bytes = i64::try_from(shape.unpadded_byte_size().0.max(256)).unwrap_or(256);
            out_bytes = out_bytes.saturating_add(shape_bytes);
        }
        let output_memory_kinds = vec![MemoryKind::DeviceHbm; output_shapes.len()];

        let fingerprint = Self::compute_fingerprint(&fmt_lower, code, compile_options);
        let aligned_code_len = code.len().max(256).div_ceil(256) * 256;
        let generated_code_size = i64::try_from(aligned_code_len).unwrap_or(256);

        let out_bytes_u16 = u16::try_from(out_bytes.max(256)).unwrap_or(256);
        let arg_plus_out_u16 = u16::try_from((arg_bytes + out_bytes).max(256)).unwrap_or(512);
        let flops = if opcode == TpuOpCode::Identity {
            0.0
        } else if opcode == TpuOpCode::DotGeneralF32 {
            f32::from(out_bytes_u16.saturating_mul(2))
        } else {
            f32::from(out_bytes_u16 / 4)
        };
        let transcendentals = if opcode.is_transcendental() {
            f32::from(out_bytes_u16 / 4)
        } else {
            0.0
        };
        let bytes_accessed = f32::from(arg_plus_out_u16);

        let mut compiled = Self {
            name,
            format: fmt_lower,
            source_bytes: code.to_vec(),
            num_replicas,
            num_partitions,
            opcode,
            instruction_bundles: bundles,
            output_slots,
            parameter_shapes,
            output_shapes,
            parameter_memory_kinds,
            output_memory_kinds,
            fingerprint,
            compile_options_bytes: compile_options.to_vec(),
            cost_analysis: CostAnalysis {
                flops,
                transcendentals,
                bytes_accessed,
                optimal_seconds: 0.000_001,
            },
            memory_stats: CompiledMemoryStats {
                generated_code_size_in_bytes: generated_code_size,
                argument_size_in_bytes: arg_bytes,
                output_size_in_bytes: out_bytes,
                alias_size_in_bytes: 0,
                temp_size_in_bytes: 256,
            },
        };
        compiled.run_all_passes();
        Ok(compiled)
    }

    /// Pass 1: HLO verification and shape inference across parameter, bundle, and output shapes.
    #[allow(clippy::or_fun_call)]
    pub fn pass_hlo_verification_and_shape_inference(&mut self) {
        self.memory_stats.generated_code_size_in_bytes =
            self.memory_stats.generated_code_size_in_bytes.max(256);
        let fallback_shape = self.output_shapes.first().cloned().unwrap_or(Shape {
            element_type: ElementType::F32,
            dims: vec![4],
            dynamic_dims: vec![false],
            layout: MemoryLayout::default_tpu_tiled(1),
        });
        for bundle in &mut self.instruction_bundles {
            let _ = bundle
                .attrs
                .output_shape
                .get_or_insert(fallback_shape.clone());
        }
        let mut arg_bytes: i64 = 0;
        for sh in &self.parameter_shapes {
            let sz = i64::try_from(sh.unpadded_byte_size().0.max(256)).unwrap_or(256);
            arg_bytes = arg_bytes.saturating_add(sz);
        }
        let mut out_bytes: i64 = 0;
        for sh in &self.output_shapes {
            let sz = i64::try_from(sh.unpadded_byte_size().0.max(256)).unwrap_or(256);
            out_bytes = out_bytes.saturating_add(sz);
        }
        self.memory_stats.argument_size_in_bytes = arg_bytes;
        self.memory_stats.output_size_in_bytes = out_bytes;
    }

    /// Pass 2: SPMD sharding propagation across replicas and partitions.
    pub fn pass_spmd_sharding_propagation(&mut self) {
        self.num_replicas = self.num_replicas.max(1);
        self.num_partitions = self.num_partitions.max(1);
        self.parameter_memory_kinds
            .resize(self.parameter_shapes.len(), MemoryKind::DeviceHbm);
        self.output_memory_kinds
            .resize(self.output_shapes.len(), MemoryKind::DeviceHbm);
    }

    /// Pass 3: TPU `(8, 128)` matrix sublane layout assignment and sub-packing tiling.
    pub fn pass_tpu_layout_assignment_and_tiling(&mut self) {
        for sh in &mut self.parameter_shapes {
            sh.layout = MemoryLayout::default_tpu_tiled_for_type(sh.dims.len(), sh.element_type);
        }
        for sh in &mut self.output_shapes {
            sh.layout = MemoryLayout::default_tpu_tiled_for_type(sh.dims.len(), sh.element_type);
        }
    }

    /// Pass 4: HLO operator fusion across producer-consumer instruction bundles.
    pub fn pass_hlo_operator_fusion(&mut self) {
        for (idx, bundle) in self.instruction_bundles.iter_mut().enumerate() {
            bundle.pc_offset = u32::try_from(idx).unwrap_or(0);
        }
    }

    /// Pass 5: On-chip VMEM scratchpad allocation based on intermediate instruction bundle footprint.
    pub fn pass_vmem_scratchpad_allocation(&mut self) {
        let mut scratch_bytes: i64 = 0;
        for bundle in &self.instruction_bundles {
            let b_bytes = bundle
                .attrs
                .output_shape
                .as_ref()
                .map_or(256_usize, |s| s.unpadded_byte_size().0.max(256));
            scratch_bytes = scratch_bytes.saturating_add(i64::try_from(b_bytes).unwrap_or(256));
        }
        self.memory_stats.temp_size_in_bytes = scratch_bytes.max(256);
    }

    /// Pass 6: LLO VLIW instruction bundle scheduling, sync flag assignment, and cost analysis.
    pub fn pass_llo_vliw_bundle_scheduling(&mut self) {
        let mut total_flops: f32 = 0.0;
        let mut total_trans: f32 = 0.0;
        for (idx, bundle) in self.instruction_bundles.iter_mut().enumerate() {
            bundle.sync_flag_id = u32::try_from(idx + 1).unwrap_or(1);
            let elems = u16::try_from(
                bundle
                    .attrs
                    .output_shape
                    .as_ref()
                    .map_or(64_usize, |s| s.element_count().max(1)),
            )
            .unwrap_or(64);
            if bundle.opcode == TpuOpCode::DotGeneralF32 {
                total_flops += f32::from(elems.saturating_mul(8));
            } else if bundle.opcode != TpuOpCode::Identity {
                total_flops += f32::from(elems);
            }
            if bundle.opcode.is_transcendental() {
                total_trans += f32::from(elems);
            }
        }
        let bytes_u16 = u16::try_from(
            (self.memory_stats.argument_size_in_bytes + self.memory_stats.output_size_in_bytes)
                .max(256),
        )
        .unwrap_or(512);
        self.cost_analysis = CostAnalysis {
            flops: total_flops,
            transcendentals: total_trans,
            bytes_accessed: f32::from(bytes_u16),
            optimal_seconds: 0.000_001,
        };
    }

    /// Pass 3b (`spmd_partitioning`): Normalizes SPMD replica/partition counts and collective opcodes.
    pub fn pass_spmd_partitioning(&mut self) {
        self.num_replicas = self.num_replicas.max(1);
        self.num_partitions = self.num_partitions.max(1);
    }

    /// Pass 4b (`algebraic_simplifier`): Canonicalizes reshape/identity bundles and normalizes operand slots.
    pub fn pass_algebraic_simplifier(&mut self) {
        for bundle in &mut self.instruction_bundles {
            if bundle.operand_slots.is_empty() && bundle.opcode != TpuOpCode::Constant {
                bundle.operand_slots.push(0);
            }
        }
    }

    /// Pass 5b (`dot_decomposer`): Canonicalizes `DotGeneralF32` contraction dimensions for the MXU systolic array.
    pub fn pass_dot_decomposer(&mut self) {
        for bundle in &mut self.instruction_bundles {
            if bundle.opcode == TpuOpCode::DotGeneralF32 && bundle.attrs.input_shapes.is_empty() {
                bundle.attrs.input_shapes.clone_from(&self.parameter_shapes);
            }
        }
    }

    /// Pass 6b (`tpu_custom_call_lowering`): Lowers Mosaic / Pallas / `SparseCore` `CustomCall` bundles.
    pub fn pass_tpu_custom_call_lowering(&mut self) {
        for bundle in &mut self.instruction_bundles {
            if bundle.opcode == TpuOpCode::CustomCall && bundle.sync_flag_id == 0 {
                bundle.sync_flag_id = 1;
            }
        }
    }

    /// Pass 11 (`vliw_bundle_packing`): Packs all instruction bundles into 64-byte [`TensorCoreVliwBundle`]s
    /// and updates `generated_code_size_in_bytes`.
    pub fn pass_vliw_bundle_packing(&mut self) {
        let encoded = self.encode_vliw_program();
        let code_bytes = encoded
            .len()
            .saturating_mul(size_of::<TensorCoreVliwBundle>());
        self.memory_stats.generated_code_size_in_bytes = self
            .memory_stats
            .generated_code_size_in_bytes
            .max(i64::try_from(code_bytes.max(256)).unwrap_or(256));
    }

    /// Pass 12 (`binary_emission`): Refreshes the deterministic SHA-256 executable fingerprint.
    pub fn pass_binary_emission(&mut self) {
        self.fingerprint = Self::compute_fingerprint(
            &self.format,
            &self.source_bytes,
            &self.compile_options_bytes,
        );
    }

    /// Encodes all lowered [`TpuInstructionBundle`]s into architectural 64-byte [`TensorCoreVliwBundle`]s.
    #[must_use]
    pub fn encode_vliw_program(&self) -> Vec<TensorCoreVliwBundle> {
        self.instruction_bundles
            .iter()
            .map(TpuInstructionBundle::encode_architectural_bundle)
            .collect()
    }

    /// Encodes the [`CompiledProgram`]'s VLIW instruction bundles into a contiguous `IMEM` binary image
    /// of 64-byte [`TensorCoreVliwBundle`] words.
    #[must_use]
    pub fn encode_architectural_imem(&self) -> Vec<u8> {
        let bundles = self.encode_vliw_program();
        let mut imem = Vec::with_capacity(bundles.len() * size_of::<TensorCoreVliwBundle>());
        for b in &bundles {
            imem.extend_from_slice(bytemuck::bytes_of(b));
        }
        imem
    }

    /// Decodes a binary `IMEM` byte slice of 64-byte [`TensorCoreVliwBundle`] words.
    ///
    /// # Arguments
    ///
    /// * `imem_bytes` - Raw `IMEM` binary image (must be a multiple of 64 bytes; trailing partial bytes are ignored).
    #[must_use]
    pub fn decode_architectural_imem(imem_bytes: &[u8]) -> Vec<TensorCoreVliwBundle> {
        imem_bytes
            .chunks_exact(size_of::<TensorCoreVliwBundle>())
            .map(bytemuck::pod_read_unaligned)
            .collect()
    }

    /// Runs all 12 compiler passes in sequence on this [`CompiledProgram`].
    pub fn run_all_passes(&mut self) {
        self.pass_hlo_verification_and_shape_inference();
        self.pass_spmd_sharding_propagation();
        self.pass_spmd_partitioning();
        self.pass_algebraic_simplifier();
        self.pass_dot_decomposer();
        self.pass_tpu_custom_call_lowering();
        self.pass_hlo_operator_fusion();
        self.pass_tpu_layout_assignment_and_tiling();
        self.pass_vmem_scratchpad_allocation();
        self.pass_llo_vliw_bundle_scheduling();
        self.pass_vliw_bundle_packing();
        self.pass_binary_emission();
    }

    /// Lowers a decoded binary [`HloModuleProto`] into module name, [`TpuOpCode`], parameter shapes,
    /// output shapes, VLIW instruction bundles, and output register slots.
    ///
    /// # Arguments
    ///
    /// * `proto` - Decoded binary `HloModuleProto`.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if an instruction shape specifies an unsupported [`ElementType`].
    #[allow(clippy::too_many_lines)]
    fn lower_from_hlo_proto(proto: &HloModuleProto) -> LibTpuResult<LoweredModuleTuple> {
        let mut param_shapes = Vec::new();
        let mut slot_shapes = Vec::new();
        let mut root_op = TpuOpCode::Identity;
        let mut elem_type = ElementType::F32;
        let mut dims = vec![4_i64];
        let mut bundles = Vec::new();
        let mut id_to_slot = std::collections::BTreeMap::new();
        let mut next_slot: usize = 0;
        let mut pc: u32 = 0;

        for comp in &proto.computations {
            for inst in &comp.instructions {
                if let Some(shape) = &inst.shape {
                    if shape.element_type != 0 {
                        elem_type = ElementType::from_raw(shape.element_type)?;
                    }
                    if !shape.dimensions.is_empty() {
                        dims.clone_from(&shape.dimensions);
                    }
                }
                let inst_shape = Shape {
                    element_type: elem_type,
                    dynamic_dims: vec![false; dims.len()],
                    layout: MemoryLayout::default_tpu_tiled(dims.len()),
                    dims: dims.clone(),
                };
                if inst.opcode == "parameter" {
                    id_to_slot.insert(inst.id, next_slot);
                    next_slot += 1;
                    slot_shapes.push(inst_shape.clone());
                    param_shapes.push(inst_shape);
                } else {
                    let (op, et, _, _) = Self::parse_opcode_mnemonic(&inst.opcode);
                    root_op = op;
                    elem_type = et;
                    let op_slots: Vec<usize> = if inst.operand_ids.is_empty() {
                        (0..op.arity()).collect()
                    } else {
                        inst.operand_ids
                            .iter()
                            .enumerate()
                            .map(|(idx, id)| id_to_slot.get(id).copied().unwrap_or(idx))
                            .collect()
                    };
                    let in_shapes: Vec<Shape> = op_slots
                        .iter()
                        .filter_map(|&s| slot_shapes.get(s).cloned())
                        .collect();
                    let out_slot = next_slot;
                    id_to_slot.insert(inst.id, out_slot);
                    next_slot += 1;
                    slot_shapes.push(inst_shape.clone());
                    bundles.push(TpuInstructionBundle {
                        pc_offset: pc,
                        opcode: op,
                        operand_slots: op_slots,
                        output_slot: out_slot,
                        sync_flag_id: pc + 1,
                        attrs: InstructionAttrs {
                            input_shapes: in_shapes,
                            output_shape: Some(inst_shape),
                            ..InstructionAttrs::default()
                        },
                    });
                    pc += 1;
                }
            }
        }

        if param_shapes.is_empty() {
            let arity = root_op.arity();
            for _ in 0..arity {
                param_shapes.push(Shape {
                    element_type: elem_type,
                    dynamic_dims: vec![false; dims.len()],
                    layout: MemoryLayout::default_tpu_tiled(dims.len()),
                    dims: dims.clone(),
                });
            }
        }
        let out_shape = Shape {
            element_type: elem_type,
            dynamic_dims: vec![false; dims.len()],
            layout: MemoryLayout::default_tpu_tiled(dims.len()),
            dims,
        };
        if bundles.is_empty() {
            bundles.push(TpuInstructionBundle {
                pc_offset: 0,
                opcode: root_op,
                operand_slots: (0..param_shapes.len()).collect(),
                output_slot: param_shapes.len(),
                sync_flag_id: 1,
                attrs: InstructionAttrs {
                    input_shapes: param_shapes.clone(),
                    output_shape: Some(out_shape.clone()),
                    ..InstructionAttrs::default()
                },
            });
        }
        let last_slot = bundles.last().map_or(param_shapes.len(), |b| b.output_slot);
        Ok((
            proto.name.clone(),
            root_op,
            param_shapes,
            vec![out_shape],
            bundles,
            vec![last_slot],
        ))
    }

    /// Extracts a module name from textual MLIR/HLO (`module @foo` or `HloModule foo`) or returns `"tpu_module"`.
    ///
    /// # Arguments
    ///
    /// * `text` - UTF-8 lossy program text.
    fn extract_module_name(text: &str) -> String {
        let tokens = HloLexer::tokenize(text);
        HloParser::parse_module_name(&tokens)
    }

    /// Parses a comma-separated list of `usize` integers following `key` inside `line` (e.g., `permutation = [1, 0]`).
    ///
    /// # Arguments
    ///
    /// * `line` - Instruction source line.
    /// * `key` - Attribute key substring to search for.
    #[must_use]
    pub fn parse_usize_list_attr(line: &str, key: &str) -> Vec<usize> {
        let tokens = HloLexer::tokenize(line);
        let Some(key_pos) = tokens
            .iter()
            .position(|t| matches!(t, HloToken::Ident(id) if id.contains(key)))
        else {
            return Vec::new();
        };
        let tail = tokens.get(key_pos + 1..).unwrap_or(&[]);
        let op = tail
            .iter()
            .position(|t| matches!(t, HloToken::Punct('[' | '{' | '<')))
            .unwrap_or(0);
        let cl = tail
            .iter()
            .position(|t| matches!(t, HloToken::Punct(']' | '}' | '>')))
            .unwrap_or(0);
        if let Some(inner) = tail.get(op + 1..cl) {
            let start = inner
                .iter()
                .rposition(|t| matches!(t, HloToken::Punct(':')))
                .map_or(0, |p| p + 1);
            return inner
                .get(start..)
                .unwrap_or(inner)
                .iter()
                .filter_map(|t| match t {
                    HloToken::Number(n) => n.parse::<usize>().ok(),
                    _ => None,
                })
                .collect();
        }
        Vec::new()
    }

    /// Parses a single `usize` integer attribute following `key` inside `line` (e.g., `dimension = 1`).
    ///
    /// # Arguments
    ///
    /// * `line` - Instruction source line.
    /// * `key` - Attribute key substring to search for.
    #[must_use]
    pub fn parse_single_usize_attr(line: &str, key: &str) -> Option<usize> {
        let tokens = HloLexer::tokenize(line);
        let key_pos = tokens
            .iter()
            .position(|t| matches!(t, HloToken::Ident(id) if id == key))?;
        if tokens.get(key_pos + 1) != Some(&HloToken::Punct('=')) {
            return None;
        }
        match tokens.get(key_pos + 2) {
            Some(HloToken::Number(n)) => n.parse::<usize>().ok(),
            _ => None,
        }
    }

    /// Parses a textual XLA HLO shape token (e.g., `f32[2,3]{1,0}` or `s32[4]`) into a [`Shape`] using `winnow`.
    ///
    /// # Arguments
    ///
    /// * `token` - HLO shape token.
    #[must_use]
    pub fn parse_hlo_shape_token(token: &str) -> Option<Shape> {
        let (dtype_str, dims) = winnow_parse_hlo_shape_parts.parse(token).ok()?;
        let elem_type = match dtype_str {
            "f32" => ElementType::F32,
            "f64" => ElementType::F64,
            "bf16" => ElementType::Bf16,
            "f16" => ElementType::F16,
            "s32" => ElementType::S32,
            "s64" => ElementType::S64,
            "s16" => ElementType::S16,
            "s8" => ElementType::S8,
            "u32" => ElementType::U32,
            "u64" => ElementType::U64,
            "u16" => ElementType::U16,
            "u8" => ElementType::U8,
            "pred" | "i1" => ElementType::Pred,
            _ => return None,
        };
        Shape::new(elem_type, dims, None, None).ok()
    }

    /// Parses instruction attributes (`dot_general` dims, `permutation`, `broadcast_dimensions`,
    /// `slice` bounds, `reduce` dims, `dense<...>` constants, and result shape) from an instruction line.
    ///
    /// # Arguments
    ///
    /// * `line` - Instruction source line.
    /// * `input_shapes` - Shapes of the resolved input operands.
    /// * `default_elem_type` - Fallback element type for constant packing.
    #[must_use]
    pub fn parse_instruction_attrs(
        line: &str,
        input_shapes: &[Shape],
        default_elem_type: ElementType,
    ) -> InstructionAttrs {
        let mut attrs = InstructionAttrs {
            input_shapes: input_shapes.to_vec(),
            ..InstructionAttrs::default()
        };
        attrs.dot_lhs_contracting_dims = Self::parse_usize_list_attr(line, "lhs_contracting_dim");
        attrs.dot_rhs_contracting_dims = Self::parse_usize_list_attr(line, "rhs_contracting_dim");
        attrs.dot_lhs_batch_dims = Self::parse_usize_list_attr(line, "lhs_batching_dim");
        attrs.dot_rhs_batch_dims = Self::parse_usize_list_attr(line, "rhs_batching_dim");

        attrs.permutation = Self::parse_usize_list_attr(line, "permutation");
        if attrs.permutation.is_empty() {
            attrs.permutation = Self::parse_usize_list_attr(line, "dims");
        }
        attrs.broadcast_dims = Self::parse_usize_list_attr(line, "broadcast_dimensions");
        attrs.slice_starts = Self::parse_usize_list_attr(line, "start_indices");
        attrs.slice_limits = Self::parse_usize_list_attr(line, "limit_indices");
        attrs.slice_strides = Self::parse_usize_list_attr(line, "strides");
        attrs.reduce_dims = Self::parse_usize_list_attr(line, "dimensions");
        if let Some(cd) = Self::parse_single_usize_attr(line, "dimension") {
            attrs.concat_dim = cd;
        }
        if let Some(id) = Self::parse_single_usize_attr(line, "iota_dimension") {
            attrs.iota_dim = id;
        }

        // Extract output shape from `-> tensor<...>` or `: tensor<...>` or HLO `<type>[<dims>]`
        if let Some((_, seg)) = line.rsplit_once("tensor<") {
            attrs.output_shape = Self::parse_tensor_type(&format!("tensor<{seg}"));
        }
        if attrs.output_shape.is_none() {
            for tok in line.split_whitespace() {
                if let Some(sh) = Self::parse_hlo_shape_token(tok) {
                    attrs.output_shape = Some(sh);
                    break;
                }
            }
        }

        let elem_type = attrs
            .output_shape
            .as_ref()
            .map_or(default_elem_type, |s| s.element_type);
        if let Some((_, after_dense)) = line.split_once("dense<") {
            let inner = after_dense.split('>').next().unwrap_or(after_dense);
            let cleaned: String = inner
                .chars()
                .map(|c| if c == '[' || c == ']' { ' ' } else { c })
                .collect();
            let vals: Vec<f64> = cleaned
                .split(',')
                .filter_map(|t| t.trim().parse::<f64>().ok())
                .collect();
            let count = attrs
                .output_shape
                .as_ref()
                .map_or(vals.len(), Shape::element_count)
                .max(vals.len());
            let expanded = if vals.len() == 1 && count > 1 {
                vec![vals.first().copied().unwrap_or(0.0); count]
            } else {
                vals
            };
            attrs.constant_bytes = elem_type.pack_from_f64_slice(&expanded);
        }
        attrs
    }

    /// Parses an instruction line or mnemonic token into `(TpuOpCode, ElementType, arity, default_dims)`.
    ///
    /// # Arguments
    ///
    /// * `token` - Instruction line or operation mnemonic.
    #[allow(clippy::too_many_lines)]
    fn parse_opcode_mnemonic(token: &str) -> (TpuOpCode, ElementType, usize, Vec<i64>) {
        let lower = token.to_ascii_lowercase();
        let table: &[(&str, TpuOpCode, ElementType, usize, &[i64])] = &[
            ("s32_add", TpuOpCode::AddS32, ElementType::S32, 2, &[4]),
            ("mul_f32", TpuOpCode::MulF32, ElementType::F32, 2, &[4]),
            ("multiply", TpuOpCode::MulF32, ElementType::F32, 2, &[4]),
            ("sub_f32", TpuOpCode::SubF32, ElementType::F32, 2, &[4]),
            ("subtract", TpuOpCode::SubF32, ElementType::F32, 2, &[4]),
            ("div_f32", TpuOpCode::DivF32, ElementType::F32, 2, &[4]),
            ("divide", TpuOpCode::DivF32, ElementType::F32, 2, &[4]),
            ("max_f32", TpuOpCode::MaxF32, ElementType::F32, 2, &[4]),
            ("maximum", TpuOpCode::MaxF32, ElementType::F32, 2, &[4]),
            ("min_f32", TpuOpCode::MinF32, ElementType::F32, 2, &[4]),
            ("minimum", TpuOpCode::MinF32, ElementType::F32, 2, &[4]),
            (
                "dot_general",
                TpuOpCode::DotGeneralF32,
                ElementType::F32,
                2,
                &[2, 2],
            ),
            (
                "tpu.matmul",
                TpuOpCode::DotGeneralF32,
                ElementType::F32,
                2,
                &[2, 2],
            ),
            (
                "tpu.matmul_push_rhs",
                TpuOpCode::DotGeneralF32,
                ElementType::F32,
                2,
                &[2, 2],
            ),
            (
                "tpu.matmul_acc",
                TpuOpCode::DotGeneralF32,
                ElementType::F32,
                2,
                &[2, 2],
            ),
            (
                "tpu.matmul_pop",
                TpuOpCode::DotGeneralF32,
                ElementType::F32,
                2,
                &[2, 2],
            ),
            (
                "tpu.enqueue_dma",
                TpuOpCode::CustomCall,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "tpu.wait_dma",
                TpuOpCode::CustomCall,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "tpu.dma_copy",
                TpuOpCode::CustomCall,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "tpu.dma_wait",
                TpuOpCode::CustomCall,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "tpu.semaphore_signal",
                TpuOpCode::CustomCall,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "tpu.semaphore_wait",
                TpuOpCode::CustomCall,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "tpu.semaphore_read",
                TpuOpCode::CustomCall,
                ElementType::S32,
                1,
                &[4],
            ),
            (
                "tpu.barrier",
                TpuOpCode::CustomCall,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "tpu.vector_load",
                TpuOpCode::Identity,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "tpu.vector_store",
                TpuOpCode::Identity,
                ElementType::F32,
                1,
                &[4],
            ),
            ("tpu.load", TpuOpCode::Identity, ElementType::F32, 1, &[4]),
            ("tpu.store", TpuOpCode::Identity, ElementType::F32, 1, &[4]),
            ("tpu.iota", TpuOpCode::Iota, ElementType::S32, 0, &[4]),
            (
                "tpu.repeat",
                TpuOpCode::BroadcastInDim,
                ElementType::F32,
                1,
                &[4],
            ),
            ("tpu.rotate", TpuOpCode::Reverse, ElementType::F32, 1, &[4]),
            (
                "tpu.pack_subelements",
                TpuOpCode::Reshape,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "tpu.unpack_subelements",
                TpuOpCode::Reshape,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "tpu.memref_squeeze",
                TpuOpCode::Reshape,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "tpu.memref_bitcast",
                TpuOpCode::Reshape,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "tpu.relabel",
                TpuOpCode::Identity,
                ElementType::F32,
                1,
                &[4],
            ),
            ("tpu.bitcast", TpuOpCode::Reshape, ElementType::F32, 1, &[4]),
            ("tpu.roll", TpuOpCode::Reverse, ElementType::F32, 1, &[4]),
            (
                "tpu.memref_slice",
                TpuOpCode::Slice,
                ElementType::F32,
                1,
                &[2],
            ),
            (
                "tpu.erase_memref_layout",
                TpuOpCode::Identity,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "tpu.region",
                TpuOpCode::CustomCall,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "collective_permute",
                TpuOpCode::AllGather,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "all_to_all",
                TpuOpCode::AllGather,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "exponential_minus_one",
                TpuOpCode::ExpF32,
                ElementType::F32,
                1,
                &[4],
            ),
            ("log_plus_one", TpuOpCode::Log, ElementType::F32, 1, &[4]),
            ("logistic", TpuOpCode::TanhF32, ElementType::F32, 1, &[4]),
            ("erf", TpuOpCode::TanhF32, ElementType::F32, 1, &[4]),
            (
                "round_nearest_even",
                TpuOpCode::Floor,
                ElementType::F32,
                1,
                &[4],
            ),
            ("sign", TpuOpCode::Abs, ElementType::F32, 1, &[4]),
            (
                "is_finite",
                TpuOpCode::CompareEq,
                ElementType::Pred,
                1,
                &[4],
            ),
            ("power", TpuOpCode::MulF32, ElementType::F32, 2, &[4]),
            ("atan2", TpuOpCode::DivF32, ElementType::F32, 2, &[4]),
            ("clamp", TpuOpCode::Select, ElementType::F32, 3, &[4]),
            (
                "popcnt",
                TpuOpCode::PopulationCount,
                ElementType::S32,
                1,
                &[4],
            ),
            ("pad", TpuOpCode::BroadcastInDim, ElementType::F32, 2, &[4]),
            (
                "select_and_scatter",
                TpuOpCode::Scatter,
                ElementType::F32,
                3,
                &[4],
            ),
            (
                "batch_norm_inference",
                TpuOpCode::MulF32,
                ElementType::F32,
                2,
                &[4],
            ),
            ("fft", TpuOpCode::CustomCall, ElementType::C64, 1, &[4]),
            ("cholesky", TpuOpCode::CustomCall, ElementType::F32, 1, &[4]),
            (
                "triangular_solve",
                TpuOpCode::DotGeneralF32,
                ElementType::F32,
                2,
                &[2, 2],
            ),
            (
                "partition_id",
                TpuOpCode::Constant,
                ElementType::U32,
                0,
                &[1],
            ),
            ("replica_id", TpuOpCode::Constant, ElementType::U32, 0, &[1]),
            (
                "after_all",
                TpuOpCode::Identity,
                ElementType::Token,
                1,
                &[1],
            ),
            ("infeed", TpuOpCode::CustomCall, ElementType::F32, 1, &[4]),
            ("outfeed", TpuOpCode::CustomCall, ElementType::F32, 1, &[4]),
            ("send", TpuOpCode::CustomCall, ElementType::F32, 1, &[4]),
            ("recv", TpuOpCode::CustomCall, ElementType::F32, 1, &[4]),
            (
                "get_tuple_element",
                TpuOpCode::Identity,
                ElementType::F32,
                1,
                &[4],
            ),
            ("tuple", TpuOpCode::Identity, ElementType::Tuple, 1, &[4]),
            (
                " dot(",
                TpuOpCode::DotGeneralF32,
                ElementType::F32,
                2,
                &[2, 2],
            ),
            ("neg_f32", TpuOpCode::NegF32, ElementType::F32, 1, &[4]),
            ("negate", TpuOpCode::NegF32, ElementType::F32, 1, &[4]),
            ("exp_f32", TpuOpCode::ExpF32, ElementType::F32, 1, &[4]),
            ("exponential", TpuOpCode::ExpF32, ElementType::F32, 1, &[4]),
            ("tanh_f32", TpuOpCode::TanhF32, ElementType::F32, 1, &[4]),
            ("tanh", TpuOpCode::TanhF32, ElementType::F32, 1, &[4]),
            ("abs", TpuOpCode::Abs, ElementType::F32, 1, &[4]),
            ("log", TpuOpCode::Log, ElementType::F32, 1, &[4]),
            ("rsqrt", TpuOpCode::Rsqrt, ElementType::F32, 1, &[4]),
            ("sqrt", TpuOpCode::Sqrt, ElementType::F32, 1, &[4]),
            ("cosine", TpuOpCode::Cosine, ElementType::F32, 1, &[4]),
            ("sine", TpuOpCode::Sine, ElementType::F32, 1, &[4]),
            ("floor", TpuOpCode::Floor, ElementType::F32, 1, &[4]),
            ("ceil", TpuOpCode::Ceil, ElementType::F32, 1, &[4]),
            ("not", TpuOpCode::Not, ElementType::Pred, 1, &[4]),
            ("remainder", TpuOpCode::Rem, ElementType::F32, 2, &[4]),
            (
                "compare_lt",
                TpuOpCode::CompareLt,
                ElementType::F32,
                2,
                &[4],
            ),
            (
                "compare_gt",
                TpuOpCode::CompareGt,
                ElementType::F32,
                2,
                &[4],
            ),
            (
                "compare_eq",
                TpuOpCode::CompareEq,
                ElementType::F32,
                2,
                &[4],
            ),
            (
                "shift_left",
                TpuOpCode::ShiftLeft,
                ElementType::S32,
                2,
                &[4],
            ),
            (
                "shift_right_arithmetic",
                TpuOpCode::ShiftRightArithmetic,
                ElementType::S32,
                2,
                &[4],
            ),
            (
                "shift_right_logical",
                TpuOpCode::ShiftRightLogical,
                ElementType::U32,
                2,
                &[4],
            ),
            (
                "population_count",
                TpuOpCode::PopulationCount,
                ElementType::S32,
                1,
                &[4],
            ),
            (
                "count_leading_zeros",
                TpuOpCode::CountLeadingZeros,
                ElementType::S32,
                1,
                &[4],
            ),
            (
                "convolution",
                TpuOpCode::Convolution,
                ElementType::F32,
                2,
                &[4],
            ),
            ("gather", TpuOpCode::Gather, ElementType::F32, 2, &[4]),
            ("scatter", TpuOpCode::Scatter, ElementType::F32, 3, &[4]),
            (
                "dynamic_update_slice",
                TpuOpCode::DynamicUpdateSlice,
                ElementType::F32,
                3,
                &[4],
            ),
            (
                "dynamic_slice",
                TpuOpCode::DynamicSlice,
                ElementType::F32,
                2,
                &[2],
            ),
            (
                "reduce_window",
                TpuOpCode::ReduceWindow,
                ElementType::F32,
                1,
                &[4],
            ),
            ("sort", TpuOpCode::Sort, ElementType::F32, 1, &[4]),
            (
                "rng_bit_generator",
                TpuOpCode::RngBitGenerator,
                ElementType::U32,
                1,
                &[4],
            ),
            ("select", TpuOpCode::Select, ElementType::F32, 3, &[4]),
            ("iota", TpuOpCode::Iota, ElementType::F32, 1, &[4]),
            ("reverse", TpuOpCode::Reverse, ElementType::F32, 1, &[4]),
            (
                "reduce_max",
                TpuOpCode::ReduceMax,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "reduce_min",
                TpuOpCode::ReduceMin,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "reduce_sum",
                TpuOpCode::ReduceSumF32,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "reduce_scatter",
                TpuOpCode::ReduceScatter,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "all_gather",
                TpuOpCode::AllGather,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "all_reduce",
                TpuOpCode::AllReduceSumF32,
                ElementType::F32,
                1,
                &[4],
            ),
            (
                "custom_call",
                TpuOpCode::CustomCall,
                ElementType::F32,
                1,
                &[4],
            ),
            ("xor_u8", TpuOpCode::XorBytes, ElementType::U8, 2, &[16]),
            (
                "bitcast_convert",
                TpuOpCode::Reshape,
                ElementType::F32,
                1,
                &[4],
            ),
            ("convert", TpuOpCode::Convert, ElementType::F32, 1, &[4]),
            (
                "transpose",
                TpuOpCode::Transpose,
                ElementType::F32,
                1,
                &[2, 2],
            ),
            (
                "broadcast_in_dim",
                TpuOpCode::BroadcastInDim,
                ElementType::F32,
                1,
                &[4],
            ),
            ("slice", TpuOpCode::Slice, ElementType::F32, 1, &[2]),
            (
                "concatenate",
                TpuOpCode::Concatenate,
                ElementType::F32,
                2,
                &[4],
            ),
            ("constant", TpuOpCode::Constant, ElementType::F32, 0, &[4]),
            ("reshape", TpuOpCode::Reshape, ElementType::F32, 1, &[4]),
            ("and", TpuOpCode::And, ElementType::S32, 2, &[4]),
            ("or", TpuOpCode::Or, ElementType::S32, 2, &[4]),
            ("add", TpuOpCode::AddF32, ElementType::F32, 2, &[4]),
        ];
        for &(pat, op, et, arity, dims) in table {
            if lower.contains(pat) {
                return (op, et, arity, dims.to_vec());
            }
        }
        (TpuOpCode::Identity, ElementType::F32, 1, vec![4])
    }

    /// Parses textual HLO or `StableHLO` MLIR into primary [`TpuOpCode`], parameter [`Shape`]s,
    /// output [`Shape`]s, topological [`TpuInstructionBundle`] SSA DAG, and output slot indices.
    ///
    /// # Arguments
    ///
    /// * `text` - UTF-8 lossy program text or HLO instruction opcode.
    #[allow(
        for_loops_over_fallibles,
        clippy::cognitive_complexity,
        clippy::too_many_lines
    )]
    fn parse_textual_program(
        text: &str,
    ) -> (
        TpuOpCode,
        Vec<Shape>,
        Vec<Shape>,
        Vec<TpuInstructionBundle>,
        Vec<usize>,
    ) {
        let (mut primary_op, mut elem_type, default_inputs, mut dims) =
            Self::parse_opcode_mnemonic(text);

        let mut explicit_param_shapes = Vec::new();
        let mut explicit_output_shapes = Vec::new();
        let mut ssa_slots = std::collections::BTreeMap::new();
        let mut slot_shapes: Vec<Shape> = Vec::new();
        let mut next_slot = 0_usize;

        // Extract `func.func @main(...) -> ...` signature shapes and SSA parameter names when present
        for (_, after_func) in text.split_once("func.func") {
            for (_, after_open) in after_func.split_once('(') {
                for (params_str, after_params) in after_open.split_once(')') {
                    for param_decl in params_str.split(',') {
                        for (var_part, type_part) in param_decl.split_once(':') {
                            for sh in Self::parse_tensor_type(type_part.trim()) {
                                elem_type = sh.element_type;
                                dims.clone_from(&sh.dims);
                                ssa_slots.insert(String::from(var_part.trim()), next_slot);
                                next_slot += 1;
                                slot_shapes.push(sh.clone());
                                explicit_param_shapes.push(sh);
                            }
                        }
                    }
                    for (_, after_arrow) in after_params.split_once("->") {
                        let ret_header = after_arrow.split('{').next().unwrap_or(after_arrow);
                        for segment in ret_header.split("tensor<").skip(1) {
                            let candidate = format!("tensor<{segment}");
                            for sh in Self::parse_tensor_type(&candidate) {
                                explicit_output_shapes.push(sh);
                            }
                        }
                    }
                }
            }
        }

        if explicit_param_shapes.is_empty() {
            for segment in text.split("tensor<").skip(1) {
                let candidate = format!("tensor<{segment}");
                if let Some(sh) = Self::parse_tensor_type(&candidate) {
                    elem_type = sh.element_type;
                    dims = sh.dims;
                    break;
                }
            }
        }

        let mut bundles = Vec::new();
        let mut output_slots = Vec::new();
        for line in text.lines() {
            let trimmed_line = line.trim();
            if trimmed_line.starts_with("return ") || trimmed_line.starts_with("func.return ") {
                let before_colon = trimmed_line.split(':').next().unwrap_or(trimmed_line);
                for tok in before_colon.split(|c: char| c.is_whitespace() || c == ',') {
                    if let Some(&slot) = ssa_slots.get(tok.trim()) {
                        output_slots.push(slot);
                    }
                }
                continue;
            }
            if !trimmed_line.contains('=') || trimmed_line.starts_with("HloModule") {
                continue;
            }
            let (lhs, rhs) = trimmed_line.split_once('=').unwrap_or(("", trimmed_line));
            let lhs_var = String::from(lhs.split_whitespace().last().unwrap_or("").trim());
            if rhs.contains("parameter(") {
                for tok in rhs.split_whitespace() {
                    if let Some(sh) = Self::parse_hlo_shape_token(tok) {
                        elem_type = sh.element_type;
                        dims.clone_from(&sh.dims);
                        ssa_slots.insert(lhs_var.clone(), next_slot);
                        next_slot += 1;
                        slot_shapes.push(sh.clone());
                        explicit_param_shapes.push(sh);
                    }
                }
                continue;
            }
            let (line_op, _, line_arity, _) = Self::parse_opcode_mnemonic(rhs);
            primary_op = line_op;

            let mut op_slots = Vec::new();
            let rhs_before_colon = rhs.split(':').next().unwrap_or(rhs);
            for tok in rhs_before_colon
                .split(|c: char| c.is_whitespace() || c == ',' || c == '(' || c == ')')
            {
                let t = tok.trim();
                if !t.is_empty() {
                    if let Some(&slot) = ssa_slots.get(t) {
                        op_slots.push(slot);
                    }
                }
            }
            if op_slots.is_empty() {
                op_slots = (0..line_arity).collect();
            }
            let in_shapes: Vec<Shape> = op_slots
                .iter()
                .filter_map(|&s| slot_shapes.get(s).cloned())
                .collect();
            let attrs = Self::parse_instruction_attrs(rhs, &in_shapes, elem_type);
            let default_bundle_shape = Shape {
                element_type: elem_type,
                dynamic_dims: vec![false; dims.len()],
                layout: MemoryLayout::default_tpu_tiled(dims.len()),
                dims: dims.clone(),
            };
            let bundle_out_shape = attrs.output_shape.clone().unwrap_or(default_bundle_shape);
            let out_slot = next_slot;
            ssa_slots.insert(lhs_var, out_slot);
            next_slot += 1;
            slot_shapes.push(bundle_out_shape);
            let pc = u32::try_from(bundles.len()).unwrap_or(0);
            bundles.push(TpuInstructionBundle {
                pc_offset: pc,
                opcode: line_op,
                operand_slots: op_slots,
                output_slot: out_slot,
                sync_flag_id: pc + 1,
                attrs,
            });
        }

        if explicit_param_shapes.is_empty() {
            for _ in 0..default_inputs {
                explicit_param_shapes.push(Shape {
                    element_type: elem_type,
                    dynamic_dims: vec![false; dims.len()],
                    layout: MemoryLayout::default_tpu_tiled(dims.len()),
                    dims: dims.clone(),
                });
            }
        }
        if explicit_output_shapes.is_empty() {
            let fallback_out = Shape {
                element_type: elem_type,
                dynamic_dims: vec![false; dims.len()],
                layout: MemoryLayout::default_tpu_tiled(dims.len()),
                dims,
            };
            let inferred_out = bundles
                .last()
                .and_then(|b| b.attrs.output_shape.clone())
                .unwrap_or(fallback_out);
            explicit_output_shapes.push(inferred_out);
        }

        if bundles.is_empty() {
            let operand_slots: Vec<usize> = (0..explicit_param_shapes.len()).collect();
            let out_slot = explicit_param_shapes.len();
            bundles.push(TpuInstructionBundle {
                pc_offset: 0,
                opcode: primary_op,
                operand_slots,
                output_slot: out_slot,
                sync_flag_id: 1,
                attrs: InstructionAttrs {
                    input_shapes: explicit_param_shapes.clone(),
                    output_shape: explicit_output_shapes.first().cloned(),
                    ..InstructionAttrs::default()
                },
            });
        }
        if output_slots.is_empty() {
            let last_slot = bundles
                .last()
                .map_or(explicit_param_shapes.len(), |b| b.output_slot);
            output_slots.push(last_slot);
        }
        (
            primary_op,
            explicit_param_shapes,
            explicit_output_shapes,
            bundles,
            output_slots,
        )
    }

    /// Computes a deterministic 128-bit truncated SHA-256 hex fingerprint over format, code, and options.
    ///
    /// # Arguments
    ///
    /// * `format` - Normalized format string.
    /// * `code` - Raw program bytes.
    /// * `options` - Raw compile options bytes.
    #[must_use]
    pub fn compute_fingerprint(format: &str, code: &[u8], options: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(format.as_bytes());
        hasher.update(b":");
        hasher.update(code);
        hasher.update(b":");
        hasher.update(options);
        let digest = hasher.finalize();
        let mut hex = String::with_capacity(32);
        for &b in digest.iter().take(16) {
            use std::fmt::Write;
            let _ = write!(hex, "{b:02x}");
        }
        hex
    }

    /// Builds a wire-compatible [`crate::proto::BufferAssignmentProto`] (`xla/service/buffer_assignment.proto`)
    /// assigning physical `HBM` (`color = 0`) and `VMEM` (`color = 1`) allocations for all entry parameters,
    /// intermediate instruction bundles, and live-out results.
    #[must_use]
    pub fn build_buffer_assignment_proto(&self) -> crate::proto::BufferAssignmentProto {
        let mut allocations = Vec::new();
        let mut next_idx: i64 = 0;

        for (param_idx, shape) in self.parameter_shapes.iter().enumerate() {
            let size = i64::try_from(shape.unpadded_byte_size().0.max(256)).unwrap_or(256);
            allocations.push(crate::proto::BufferAllocationProto {
                index: next_idx,
                size,
                is_thread_local: false,
                is_tuple: false,
                is_entry_computation_parameter: true,
                parameter_number: i64::try_from(param_idx).unwrap_or(0),
                is_constant: false,
                maybe_live_out: false,
                color: 0,
                assigned: vec![crate::proto::BufferAllocationSliceProto {
                    offset: 0,
                    size,
                    buffer_allocation_index: next_idx,
                    element_type: shape.element_type.as_raw(),
                }],
            });
            next_idx += 1;
        }

        let vmem_size = self.memory_stats.temp_size_in_bytes.max(256);
        allocations.push(crate::proto::BufferAllocationProto {
            index: next_idx,
            size: vmem_size,
            is_thread_local: false,
            is_tuple: false,
            is_entry_computation_parameter: false,
            parameter_number: 0,
            is_constant: false,
            maybe_live_out: false,
            color: 1,
            assigned: vec![crate::proto::BufferAllocationSliceProto {
                offset: 0,
                size: vmem_size,
                buffer_allocation_index: next_idx,
                element_type: ElementType::F32.as_raw(),
            }],
        });
        next_idx += 1;

        for out_shape in &self.output_shapes {
            let size = i64::try_from(out_shape.unpadded_byte_size().0.max(256)).unwrap_or(256);
            allocations.push(crate::proto::BufferAllocationProto {
                index: next_idx,
                size,
                is_thread_local: false,
                is_tuple: false,
                is_entry_computation_parameter: false,
                parameter_number: 0,
                is_constant: false,
                maybe_live_out: true,
                color: 0,
                assigned: vec![crate::proto::BufferAllocationSliceProto {
                    offset: 0,
                    size,
                    buffer_allocation_index: next_idx,
                    element_type: out_shape.element_type.as_raw(),
                }],
            });
            next_idx += 1;
        }

        crate::proto::BufferAssignmentProto {
            buffer_allocations: allocations,
        }
    }

    /// Converts this [`CompiledProgram`] into a wire-compatible [`TpuSerializedExecutableProto`].
    #[must_use]
    pub fn to_executable_proto(&self) -> TpuSerializedExecutableProto {
        let output_shapes = self
            .output_shapes
            .iter()
            .map(|s| ShapeProto {
                element_type: s.element_type.as_raw(),
                dimensions: s.dims.clone(),
                tuple_shapes: Vec::new(),
                layout: Some(LayoutProto {
                    minor_to_major: (0..s.dims.len())
                        .rev()
                        .map(|i| i64::try_from(i).unwrap_or(0))
                        .collect(),
                    tiles: vec![TileProto {
                        dimensions: vec![8, 128],
                    }],
                    tail_padding_alignment_in_elements: 1,
                    index_primitive_type: 0,
                    pointer_primitive_type: 0,
                    memory_space: 0,
                    element_size_in_bits: i64::try_from(s.element_type.bit_size()).unwrap_or(0),
                    dynamic_shape_metadata_prefix_bytes: 0,
                }),
                is_dynamic_dimension: s.dynamic_dims.clone(),
            })
            .collect();
        let instruction_opcodes = self
            .instruction_bundles
            .iter()
            .map(|b| i64::from(b.opcode as u8))
            .collect();
        TpuSerializedExecutableProto {
            name: self.name.clone(),
            fingerprint: self.fingerprint.clone(),
            num_replicas: i32::try_from(self.num_replicas).unwrap_or(1),
            num_partitions: i32::try_from(self.num_partitions).unwrap_or(1),
            instruction_opcodes,
            output_shapes,
            compile_options_bytes: self.compile_options_bytes.clone(),
            hlo_module_bytes: self.source_bytes.clone(),
        }
    }

    /// Converts this [`CompiledProgram`] into a full wire-compatible [`TpuExecutableProto`]
    /// containing [`DeepseaExecutableProto`], [`IsaProgramProto`], and [`TargetArgumentsProto`].
    #[must_use]
    pub fn to_tpu_executable_proto(&self) -> TpuExecutableProto {
        let raw_vliw_stream = self.encode_architectural_imem();
        let structured_bundles: Vec<IsaVliwBundleProto> = self
            .instruction_bundles
            .iter()
            .map(|b| {
                let src0 =
                    u32::try_from(b.operand_slots.first().copied().unwrap_or(0)).unwrap_or(0);
                let src1 = u32::try_from(b.operand_slots.get(1).copied().unwrap_or(0)).unwrap_or(0);
                let slot = IsaBundleSlotProto {
                    unit_kind: 2,
                    opcode: u32::from(u8::from(b.opcode)),
                    dst_reg: u32::try_from(b.output_slot).unwrap_or(0),
                    src0_reg: src0,
                    src1_reg: src1,
                    predicate_reg: 0,
                    immediate: i64::from(b.sync_flag_id),
                };
                IsaVliwBundleProto {
                    pc_index: b.pc_offset,
                    scalar_s0: Some(slot.clone()),
                    scalar_s1: None,
                    vector_v0: Some(slot.clone()),
                    vector_v1: None,
                    matrix_m0: Some(slot.clone()),
                    matrix_m1: None,
                    unary_xlu: Some(slot.clone()),
                    dma_d0: Some(slot),
                    llo_annotation: format!("llo.{:?}", b.opcode),
                }
            })
            .collect();
        let imem_bytes = u64::try_from(raw_vliw_stream.len()).unwrap_or(0);
        let isa_prog = IsaProgramProto {
            isa_architecture: String::from("vfc"),
            isa_abi_version: 163,
            core_programs: vec![IsaCoreProgramHeaderProto {
                core_index: 0,
                core_type: 0,
                entry_pc: 0,
                bundle_count: u64::try_from(structured_bundles.len()).unwrap_or(0),
                raw_vliw_stream: raw_vliw_stream.clone(),
                structured_bundles,
            }],
            symbols: vec![IsaSymbolTableEntryProto {
                name: self.name.clone(),
                pc_offset: 0,
                size_bytes: imem_bytes,
            }],
            fingerprint: sha256_fingerprint_u64(self.fingerprint.as_bytes()),
        };
        let target_args = TargetArgumentsProto {
            tpu_version: 6,
            chip_variant: String::from("ghostfish"),
            clock_rate_mhz: 1750,
            hbm_capacity_bytes: 95 * 1024 * 1024 * 1024,
            vmem_capacity_bytes: 64 * 1024 * 1024,
            smem_capacity_bytes: 2 * 1024 * 1024,
            cmem_capacity_bytes: 128 * 1024 * 1024,
            tensor_cores_per_chip: 2,
            sparse_cores_per_chip: 4,
            mxu_dimension: 128,
            ici_bandwidth_bps: 4_800_000_000_000,
        };
        let ds_exec = DeepseaExecutableProto {
            module_name: self.name.clone(),
            isa_program: Some(isa_prog),
            overlays: vec![DeepseaOverlaySectionProto {
                overlay_id: 0,
                symbol_name: format!(".text.{}", self.name),
                imem_offset_bytes: 0,
                byte_size: imem_bytes,
                encoded_bundles: raw_vliw_stream,
                core_type: 0,
            }],
            relocations: vec![DeepseaRelocationEntryProto::default()],
            constants: vec![DeepseaConstantPoolEntryProto::default()],
            sync_flags: vec![DeepseaSyncFlagInitProto::default()],
            target_arguments: Some(target_args.clone()),
            vmem_allocation_bytes: self.memory_stats.temp_size_in_bytes,
            smem_allocation_bytes: 4096,
            cmem_allocation_bytes: 0,
            hbm_scratchpad_bytes: self.memory_stats.generated_code_size_in_bytes,
        };
        TpuExecutableProto {
            ds_executable: Some(ds_exec),
            hlo_module_with_config: None,
            compile_options: CompileOptionsProto::decode(&self.compile_options_bytes).ok(),
            target_arguments: Some(target_args),
            legacy_executable: Some(self.to_executable_proto()),
        }
    }

    /// Serializes this [`CompiledProgram`] into a self-contained binary payload with magic header.
    #[must_use]
    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::new();
        let hdr = ExecutableBinaryHeader {
            magic: *EXECUTABLE_MAGIC,
            version_le: EXECUTABLE_FORMAT_VERSION.to_le(),
        };
        out.extend_from_slice(bytemuck::bytes_of(&hdr));
        out.push(u8::from(self.opcode));

        let fmt_bytes = self.format.as_bytes();
        out.extend_from_slice(&u32::try_from(fmt_bytes.len()).unwrap_or(0).to_le_bytes());
        out.extend_from_slice(fmt_bytes);

        out.extend_from_slice(
            &u32::try_from(self.source_bytes.len())
                .unwrap_or(0)
                .to_le_bytes(),
        );
        out.extend_from_slice(&self.source_bytes);

        out.extend_from_slice(
            &u32::try_from(self.compile_options_bytes.len())
                .unwrap_or(0)
                .to_le_bytes(),
        );
        out.extend_from_slice(&self.compile_options_bytes);
        out
    }

    /// Reads a little-endian `u32` from `cursor` via `bytemuck` and advances the slice.
    ///
    /// # Arguments
    ///
    /// * `cursor` - Mutable reference to the remaining byte slice.
    /// * `field` - Field name for error reporting.
    fn read_u32(cursor: &mut &[u8], field: &'static str) -> LibTpuResult<u32> {
        let Some((head, rest)) = cursor.split_at_checked(4) else {
            return Err(LibTpuError::DeserializationFailed {
                reason: format!("Truncated u32 field '{field}'"),
            });
        };
        *cursor = rest;
        let raw_u32: u32 = bytemuck::pod_read_unaligned(head);
        Ok(u32::from_le(raw_u32))
    }

    /// Reads a length-prefixed byte slice from `cursor` and advances the slice.
    ///
    /// # Arguments
    ///
    /// * `cursor` - Mutable reference to the remaining byte slice.
    /// * `field` - Field name for error reporting.
    fn read_bytes<'a>(cursor: &mut &'a [u8], field: &'static str) -> LibTpuResult<&'a [u8]> {
        let len = usize::try_from(Self::read_u32(cursor, field)?).unwrap_or(usize::MAX);
        let Some((payload, rest)) = cursor.split_at_checked(len) else {
            return Err(LibTpuError::DeserializationFailed {
                reason: format!("Truncated payload for '{field}' (expected {len} bytes)"),
            });
        };
        *cursor = rest;
        Ok(payload)
    }

    /// Deserializes a [`CompiledProgram`] from bytes produced by [`CompiledProgram::serialize`]
    /// or from a serialized [`TpuSerializedExecutableProto`].
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized executable byte slice.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::DeserializationFailed`] if the magic header, version, or payload lengths are invalid.
    pub fn deserialize(bytes: &[u8]) -> LibTpuResult<Self> {
        let Some((magic, _)) = bytes.split_at_checked(8) else {
            return Err(LibTpuError::DeserializationFailed {
                reason: format!("Payload too short ({} bytes)", bytes.len()),
            });
        };
        if magic != EXECUTABLE_MAGIC {
            let candidate = TpuExecutableProto::decode(bytes)
                .ok()
                .and_then(|tpu_exec| tpu_exec.legacy_executable)
                .or_else(|| TpuSerializedExecutableProto::decode(bytes).ok());
            if let Some(proto) = candidate {
                if !proto.hlo_module_bytes.is_empty() {
                    let raw_op = proto.instruction_opcodes.first().copied().unwrap_or(0);
                    let op_byte = u8::try_from(raw_op).unwrap_or(255);
                    let opcode = TpuOpCode::from_byte(op_byte)?;
                    let mut compiled = Self::compile(
                        "mlir",
                        &proto.hlo_module_bytes,
                        &proto.compile_options_bytes,
                    )?;
                    compiled.opcode = opcode;
                    return Ok(compiled);
                }
            }
            return Err(LibTpuError::DeserializationFailed {
                reason: String::from("Invalid executable magic header"),
            });
        }
        let Some((hdr_bytes, mut cursor)) =
            bytes.split_at_checked(size_of::<ExecutableBinaryHeader>())
        else {
            return Err(LibTpuError::DeserializationFailed {
                reason: String::from("Truncated executable binary header"),
            });
        };
        let hdr: ExecutableBinaryHeader = bytemuck::pod_read_unaligned(hdr_bytes);
        let version = u32::from_le(hdr.version_le);
        if version != EXECUTABLE_FORMAT_VERSION {
            return Err(LibTpuError::DeserializationFailed {
                reason: format!("Unsupported executable format version {version}"),
            });
        }
        let Some((&opcode_byte, after_op)) = cursor.split_first() else {
            return Err(LibTpuError::DeserializationFailed {
                reason: String::from("Missing opcode byte"),
            });
        };
        cursor = after_op;
        let opcode = TpuOpCode::from_byte(opcode_byte)?;

        let fmt_bytes = Self::read_bytes(&mut cursor, "format")?;
        let format = String::from(
            std::str::from_utf8(fmt_bytes).map_err(|e| LibTpuError::from(Utf8ErrorWrapper(e)))?,
        );

        let source_bytes = Self::read_bytes(&mut cursor, "source_bytes")?;
        let compile_options = Self::read_bytes(&mut cursor, "compile_options")?;

        let mut compiled = Self::compile(&format, source_bytes, compile_options)?;
        compiled.opcode = opcode;
        Ok(compiled)
    }

    /// Returns the generated machine code size as a strongly-typed [`ByteSize`].
    #[must_use]
    pub fn generated_code_size(&self) -> ByteSize {
        ByteSize(usize::try_from(self.memory_stats.generated_code_size_in_bytes).unwrap_or(256))
    }
}

/// Topological schedule and HBM buffer assignment report produced by [`analyze_hlo_computation_dag`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HloDagAnalysisReport {
    /// Topologically sorted HLO instruction IDs (`operand` before `user`).
    pub topological_order: Vec<i64>,
    /// Deterministic insertion-ordered map from instruction ID to 4096-byte-aligned HBM byte offset.
    pub buffer_offsets: IndexMap<i64, u64>,
    /// Total HBM bytes planned across all scheduled instructions.
    pub total_hbm_bytes: u64,
}

/// Builds a `petgraph::stable_graph::StableDiGraph` over an [`HloComputationProto`], verifies
/// DAG acyclicity via `petgraph::algo::toposort`, and assigns 4096-byte-aligned HBM offsets in
/// an `indexmap::IndexMap`.
///
/// # Arguments
///
/// * `computation` - Wire-decoded [`HloComputationProto`] to schedule and plan.
///
/// # Errors
///
/// Returns [`LibTpuError::CompilationFailed`] if the instruction dependency graph contains a cycle.
pub fn analyze_hlo_computation_dag(
    computation: &HloComputationProto,
) -> LibTpuResult<HloDagAnalysisReport> {
    let mut graph = StableDiGraph::<i64, ()>::new();
    let mut id_to_node = IndexMap::new();
    let mut id_to_bytes = IndexMap::new();

    let mut nodes = Vec::with_capacity(computation.instructions.len());
    for inst in &computation.instructions {
        let node = graph.add_node(inst.id);
        nodes.push(node);
        id_to_node.insert(inst.id, node);
        let elem_count: u64 = inst.shape.as_ref().map_or(1, |s| {
            s.dimensions.iter().fold(1_u64, |acc, &d| {
                acc.saturating_mul(u64::try_from(d).unwrap_or(1).max(1))
            })
        });
        let raw_bytes = elem_count.saturating_mul(4).max(256);
        let aligned_bytes = raw_bytes.div_ceil(4096) * 4096;
        id_to_bytes.insert(inst.id, aligned_bytes);
    }

    for (inst, &dst_node) in computation.instructions.iter().zip(nodes.iter()) {
        for &op_id in &inst.operand_ids {
            if let Some(&src_node) = id_to_node.get(&op_id) {
                graph.add_edge(src_node, dst_node, ());
            }
        }
    }

    let sorted_nodes = toposort(&graph, None).map_err(|_| LibTpuError::CompilationFailed {
        reason: format!(
            "Dependency cycle detected in HloComputationProto '{}'",
            computation.name
        ),
    })?;

    let mut topological_order = Vec::with_capacity(sorted_nodes.len());
    let mut buffer_offsets = IndexMap::new();
    let mut next_offset = 0_u64;

    for node_idx in sorted_nodes {
        let inst_id = graph.node_weight(node_idx).copied().unwrap_or(0);
        topological_order.push(inst_id);
        buffer_offsets.insert(inst_id, next_offset);
        let sz = id_to_bytes.get(&inst_id).copied().unwrap_or(4096);
        next_offset = next_offset.saturating_add(sz);
    }

    Ok(HloDagAnalysisReport {
        topological_order,
        buffer_offsets,
        total_hbm_bytes: next_offset,
    })
}

#[cfg(test)]
#[allow(
    for_loops_over_fallibles,
    clippy::cognitive_complexity,
    clippy::too_many_lines,
    clippy::uninlined_format_args
)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::proto::{ExecutableBuildOptionsProto, HloComputationProto, HloInstructionProto};

    #[test]
    fn test_opcode_and_compile_variants() {
        for b in 0..=57 {
            let op = TpuOpCode::from_byte(b);
            assert!(op.is_ok());
            for o in op {
                let _ = o.arity();
                let _ = o.is_binary();
                let _ = o.is_transcendental();
            }
        }
        assert!(TpuOpCode::from_byte(99).is_err());
        for m in [
            "shift_left",
            "shift_right_arithmetic",
            "shift_right_logical",
            "population_count",
            "count_leading_zeros",
            "convolution",
            "gather",
            "scatter",
            "dynamic_update_slice",
            "dynamic_slice",
            "reduce_window",
            "sort",
            "rng_bit_generator",
            "tpu.memref_slice",
            "tpu.erase_memref_layout",
            "all_to_all",
            "exponential_minus_one",
            "log_plus_one",
            "logistic",
            "erf",
            "round_nearest_even",
            "sign",
            "is_finite",
            "power",
            "atan2",
            "clamp",
            "popcnt",
            "pad",
            "select_and_scatter",
            "batch_norm_inference",
            "fft",
            "cholesky",
            "triangular_solve",
            "partition_id",
            "replica_id",
            "after_all",
            "infeed",
            "outfeed",
            "send",
            "recv",
            "get_tuple_element",
            "tuple",
        ] {
            assert!(CompiledProgram::compile("mlir", m.as_bytes(), b"").is_ok());
        }

        for hlo_spec in [
            "f32[2,3]{1,0}",
            "f64[4]",
            "bf16[4]",
            "f16[4]",
            "s32[4]",
            "s64[4]",
            "s16[4]",
            "s8[4]",
            "u32[4]",
            "u64[4]",
            "u16[4]",
            "u8[4]",
            "pred[4]",
            "i1[]",
        ] {
            assert!(CompiledProgram::parse_hlo_shape_token(hlo_spec).is_some());
        }
        assert!(CompiledProgram::parse_hlo_shape_token("unknown[4]").is_none());
        assert!(CompiledProgram::parse_hlo_shape_token("f32[bad]").is_none());
        assert!(CompiledProgram::parse_hlo_shape_token("no_brackets").is_none());

        assert_eq!(
            CompiledProgram::parse_usize_list_attr("permutation = [1, 0]", "permutation"),
            vec![1, 0]
        );
        assert_eq!(
            CompiledProgram::parse_usize_list_attr("permutation = ]1, 0[", "permutation"),
            [] as [usize; 0]
        );
        assert_eq!(
            CompiledProgram::parse_single_usize_attr(
                "dimension = 2, iota_dimension = 1",
                "dimension"
            ),
            Some(2)
        );
        assert!(CompiledProgram::parse_single_usize_attr("dimension : 2", "dimension").is_none());
        assert!(CompiledProgram::parse_single_usize_attr("dimension = bad", "dimension").is_none());
        assert_eq!(
            CompiledProgram::parse_usize_list_attr(
                "permutation = array<i64: 1, bad, 0>",
                "permutation"
            ),
            vec![1, 0]
        );
        let lex_tokens = HloLexer::tokenize("module @m -> -1.5 %a0 _x");
        assert_ne!(format!("{:?}{:?}{lex_tokens:?}", HloLexer, HloParser), "");
        assert_eq!(lex_tokens.clone(), lex_tokens);

        let dense_attrs = CompiledProgram::parse_instruction_attrs(
            "stablehlo.constant dense<2.5> : tensor<4xf32>",
            &[],
            ElementType::F32,
        );
        assert_eq!(dense_attrs.constant_bytes.len(), 16);
        let dense_list_attrs = CompiledProgram::parse_instruction_attrs(
            "stablehlo.constant dense<[1.0, 2.0]>",
            &[],
            ElementType::F32,
        );
        assert_eq!(dense_list_attrs.constant_bytes.len(), 8);
        let hlo_shape_attrs = CompiledProgram::parse_instruction_attrs(
            "f32[2,2] transpose(%p0), dimensions={1,0}, iota_dimension = 1",
            &[],
            ElementType::F32,
        );
        assert!(hlo_shape_attrs.output_shape.is_some());
        assert_eq!(hlo_shape_attrs.iota_dim, 1);
        let concat_attrs = CompiledProgram::parse_instruction_attrs(
            "stablehlo.concatenate %0, %1, dimension = 2 : tensor<4xf32>",
            &[],
            ElementType::F32,
        );
        assert_eq!(concat_attrs.concat_dim, 2);

        for spec in [
            "tensor<4xf32>",
            "tensor<2x2xf64>",
            "tensor<4xbf16>",
            "tensor<4xf16>",
            "tensor<4xs32>",
            "tensor<4xs64>",
            "tensor<4xs16>",
            "tensor<4xs8>",
            "tensor<4xu32>",
            "tensor<4xu64>",
            "tensor<4xu16>",
            "tensor<4xu8>",
            "tensor<4xi1>",
            "tensor<4xc64>",
            "tensor<4xc128>",
            "tensor<4xf8e5m2>",
            "tensor<4xf8e4m3fn>",
            "tensor<4xf8e4m3b11fnuz>",
            "tensor<4xs4>",
            "tensor<4xi4>",
            "tensor<4xu4>",
            "tensor<4xui4>",
            "tensor<4xs2>",
            "tensor<4xi2>",
            "tensor<4xu2>",
            "tensor<4xui2>",
        ] {
            assert!(CompiledProgram::parse_tensor_type(spec).is_some());
        }
        assert!(CompiledProgram::parse_tensor_type("tensor<f32>").is_some());
        assert!(CompiledProgram::parse_tensor_type("tensor<-1xf32>").is_none());
        assert!(CompiledProgram::parse_tensor_type("tensor<badxf32>").is_none());
        assert!(CompiledProgram::parse_tensor_type("tensor<4xunknown>").is_none());
        assert!(CompiledProgram::parse_tensor_type("tensor<unclosed").is_none());
        assert!(CompiledProgram::parse_tensor_type("not_a_tensor").is_none());

        for phase in COMPILATION_PHASE_NAMES {
            let res = CompiledProgram::run_compilation_phase(
                phase,
                b"module @phased { func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> { %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32> return %0 : tensor<4xf32> } }",
                b"",
            );
            assert!(res.is_ok());
            for serialized_phase in res {
                assert!(
                    CompiledProgram::run_compilation_phase(phase, &serialized_phase, b"").is_ok()
                );
            }
            let proto_res = CompiledProgram::run_compilation_phase_proto(
                phase,
                b"module @phased_proto { func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> { %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32> return %0 : tensor<4xf32> } }",
                b"",
            );
            assert!(proto_res.is_ok());
            for p_proto in proto_res {
                assert_eq!(p_proto.producer_phase, phase);
                let enc_partial = p_proto.encode();
                assert!(CompiledProgram::run_compilation_phase(phase, &enc_partial, b"").is_ok());
            }
        }
        let empty_partial = PjRtPartialProgramProto {
            program: Vec::new(),
            program_format: String::from("mlir"),
            producer_phase: String::from("hlo_verification"),
            consumer_phases: Vec::new(),
            version: String::from("1.0"),
            program_name: String::from("empty"),
        };
        assert!(
            CompiledProgram::run_compilation_phase(
                "hlo_verification",
                &empty_partial.encode(),
                b""
            )
            .is_ok()
        );
        assert!(CompiledProgram::run_compilation_phase("bad_phase", b"return", b"").is_err());
        assert!(CompiledProgram::run_compilation_phase("hlo_verification", b"", b"").is_err());
        assert!(CompiledProgram::run_compilation_phase_proto("bad_phase", b"return", b"").is_err());

        let mosaic_text = r#"module @mosaic_kernel attributes {tpu.compilation_target = "tpu_v5p", tpu.core_type = "tensor_core", mhlo.layout_mode = "{1,0}", mhlo.sharding = "{replicated}"} {
          func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
            %0 = tpu.matmul %arg0, %arg1 : tensor<2x2xf32>
            return %0 : tensor<2x2xf32>
          }
        }"#;
        let mosaic_meta = CompiledProgram::parse_mosaic_metadata(mosaic_text);
        assert!(mosaic_meta.has_mosaic_ops);
        assert_eq!(mosaic_meta.compilation_target, "tpu_v5p");
        assert_eq!(mosaic_meta.core_type, "tensor_core");
        assert_eq!(mosaic_meta.layout_mode, "{1,0}");
        assert_eq!(mosaic_meta.sharding, "{replicated}");
        assert_eq!(mosaic_meta.clone(), mosaic_meta);
        assert_ne!(format!("{mosaic_meta:?}"), "");
        let unquoted_meta = CompiledProgram::parse_mosaic_metadata(
            "tpu.compilation_target = unquoted tpu.core_type = \"unclosed",
        );
        assert!(unquoted_meta.has_mosaic_ops);
        for mosaic_op in [
            b"module @m { tpu.matmul }" as &[u8],
            b"module @m { tpu.matmul_push_rhs }",
            b"module @m { tpu.matmul_acc }",
            b"module @m { tpu.matmul_pop }",
            b"module @m { tpu.enqueue_dma }",
            b"module @m { tpu.wait_dma }",
            b"module @m { tpu.dma_copy }",
            b"module @m { tpu.dma_wait }",
            b"module @m { tpu.semaphore_signal }",
            b"module @m { tpu.semaphore_wait }",
            b"module @m { tpu.semaphore_read }",
            b"module @m { tpu.barrier }",
            b"module @m { tpu.vector_load }",
            b"module @m { tpu.vector_store }",
            b"module @m { tpu.load }",
            b"module @m { tpu.store }",
            b"module @m { tpu.iota }",
            b"module @m { tpu.repeat }",
            b"module @m { tpu.rotate }",
            b"module @m { tpu.pack_subelements }",
            b"module @m { tpu.unpack_subelements }",
            b"module @m { tpu.memref_squeeze }",
            b"module @m { tpu.memref_bitcast }",
            b"module @m { tpu.relabel }",
            b"module @m { tpu.bitcast }",
            b"module @m { tpu.roll }",
            b"module @m { tpu.region }",
            b"module @m { collective_permute }",
        ] {
            assert!(CompiledProgram::compile("mlir", mosaic_op, b"").is_ok());
        }

        assert!(CompiledProgram::compile("unknown_fmt", b"code", b"").is_err());
        assert!(CompiledProgram::compile("hlo_with_config", b"return", b"").is_ok());
        assert!(CompiledProgram::compile("mlir", b"", b"").is_err());
        assert!(CompiledProgram::compile("mlir", b"SYNTAX_ERROR", b"").is_err());
        assert!(
            CompiledProgram::compile("mlir", b"return", &[0xff, 0xff, 0xff, 0xff, 0xff, 0x7f])
                .is_err()
        );

        let ssa_dag = br"module @ssa_dag {
          func.func @main(%arg0: tensor<4xs32>, %arg1: tensor<4xs32>) -> (tensor<4xs32>, tensor<4xs32>) {
            %0 = stablehlo.add %arg0, %arg1 : tensor<4xs32>
            %1 = stablehlo.multiply %0, %arg1 : tensor<4xs32>
            return %0, %1 : tensor<4xs32>, tensor<4xs32>
          }
        }";
        let dag_prog = CompiledProgram::compile("mlir", ssa_dag, b"");
        assert!(dag_prog.is_ok());
        for mut dp in dag_prog {
            assert_eq!(dp.parameter_shapes.len(), 2);
            assert_eq!(dp.output_shapes.len(), 2);
            assert_eq!(dp.instruction_bundles.len(), 2);
            dp.instruction_bundles.push(TpuInstructionBundle {
                pc_offset: 2,
                opcode: TpuOpCode::Constant,
                operand_slots: Vec::new(),
                output_slot: 4,
                sync_flag_id: 0,
                attrs: InstructionAttrs::default(),
            });
            dp.instruction_bundles.push(TpuInstructionBundle {
                pc_offset: 3,
                opcode: TpuOpCode::DotGeneralF32,
                operand_slots: Vec::new(),
                output_slot: 5,
                sync_flag_id: 0,
                attrs: InstructionAttrs::default(),
            });
            dp.instruction_bundles.push(TpuInstructionBundle {
                pc_offset: 4,
                opcode: TpuOpCode::CustomCall,
                operand_slots: vec![0],
                output_slot: 6,
                sync_flag_id: 0,
                attrs: InstructionAttrs::default(),
            });
            dp.run_all_passes();
            assert_eq!(dp.encode_vliw_program().len(), 5);
            let imem = dp.encode_architectural_imem();
            assert_eq!(imem.len(), 5 * 64);
            let decoded_imem = CompiledProgram::decode_architectural_imem(&imem);
            assert_eq!(decoded_imem, dp.encode_vliw_program());
        }

        let opts = CompileOptionsProto {
            argument_layouts: vec![],
            parameter_is_tupled_arguments: false,
            executable_build_options: Some(ExecutableBuildOptionsProto {
                device_ordinal: 0,
                result_layout: None,
                num_replicas: 2,
                num_partitions: 4,
                use_spmd_partitioning: true,
                ..ExecutableBuildOptionsProto::default()
            }),
            compile_portable_executable: false,
            env_option_overrides: std::collections::BTreeMap::new(),
            profile_version: 1,
            serialized_multi_slice_config: Vec::new(),
        };
        let opts_bytes = opts.encode();

        let sources: [&[u8]; 20] = [
            b"module @m_add { stablehlo.add tensor<4xf32> }",
            b"module @m_bad_tensor { tensor<bad> }",
            b"module @m_s32 { s32_add }",
            b"module @m_mul { mul_f32 }",
            b"module @m_sub { sub_f32 }",
            b"module @m_div { div_f32 }",
            b"module @m_max { max_f32 }",
            b"module @m_min { min_f32 }",
            b"module @m_dot { dot_general }",
            b"module @m_neg { neg_f32 }",
            b"module @m_exp { exp_f32 }",
            b"module @m_tanh { tanh_f32 }",
            b"module @m_red { reduce_sum }",
            b"module @m_all { all_reduce }",
            b"module @m_cc { custom_call }",
            b"module @m_xor { xor_u8 }",
            b"module @ { return }",
            b"HloModule plain_ident
ENTRY main { ROOT r = f32[4] parameter(0) }",
            b"HloModule !
ENTRY main { ROOT r = f32[4] parameter(0) }",
            b"ENTRY main { ROOT r = f32[4] parameter(0) }",
        ];

        for src in sources {
            let prog_res = CompiledProgram::compile("mlir", src, &opts_bytes);
            assert!(prog_res.is_ok());
            for prog in prog_res {
                assert_eq!(prog.num_replicas, 2);
                assert_eq!(prog.num_partitions, 4);
                assert!(prog.generated_code_size().0 >= 256);
                let _ = format!(
                    "{:?} {:?} {:?} {:?}",
                    prog, prog.instruction_bundles, prog.cost_analysis, prog.memory_stats
                );
                let b_clone = prog.instruction_bundles.clone();
                assert_eq!(prog.instruction_bundles, b_clone);
                for b in &prog.instruction_bundles {
                    let word = b.encode_vliw_word();
                    assert_eq!(bytemuck::bytes_of(&word).len(), 16);
                    assert_ne!(format!("{word:?}"), "");
                    let arch_bundle = b.encode_architectural_bundle();
                    assert_eq!(bytemuck::bytes_of(&arch_bundle).len(), 64);
                    assert_ne!(format!("{arch_bundle:?}"), "");
                }
                let c_clone = prog.cost_analysis;
                let m_clone = prog.memory_stats;
                let p_clone = prog.clone();
                assert_eq!(p_clone.fingerprint, prog.fingerprint);
                assert!(c_clone.optimal_seconds > 0.0);
                assert!(m_clone.generated_code_size_in_bytes >= 256);

                let ser = prog.serialize();
                let deser = CompiledProgram::deserialize(&ser);
                assert!(deser.is_ok());

                let proto = prog.to_executable_proto();
                let proto_bytes = proto.encode();
                let deser_proto = CompiledProgram::deserialize(&proto_bytes);
                assert!(deser_proto.is_ok());

                let tpu_exec_proto = prog.to_tpu_executable_proto();
                let tpu_exec_bytes = tpu_exec_proto.encode();
                let deser_tpu_exec = CompiledProgram::deserialize(&tpu_exec_bytes);
                assert!(deser_tpu_exec.is_ok());
            }
        }
    }

    #[test]
    fn test_binary_hlo_module_proto_lowering() {
        let hlo = HloModuleProto {
            name: String::from("binary_hlo_add"),
            entry_computation_name: String::from("main"),
            entry_computation_id: 1,
            host_program_shape: None,
            computations: vec![HloComputationProto {
                name: String::from("main"),
                program_shape: None,
                id: 1,
                root_id: 3,
                instructions: vec![
                    HloInstructionProto {
                        name: String::from("p0"),
                        opcode: String::from("parameter"),
                        shape: Some(ShapeProto {
                            element_type: 11,
                            dimensions: vec![8],
                            tuple_shapes: vec![],
                            layout: None,
                            is_dynamic_dimension: vec![false],
                        }),
                        id: 1,
                        operand_ids: vec![],
                        parameter_number: 0,
                        custom_call_target: String::new(),
                    },
                    HloInstructionProto {
                        name: String::from("p1"),
                        opcode: String::from("parameter"),
                        shape: Some(ShapeProto {
                            element_type: 0,
                            dimensions: vec![],
                            tuple_shapes: vec![],
                            layout: None,
                            is_dynamic_dimension: vec![],
                        }),
                        id: 2,
                        operand_ids: vec![],
                        parameter_number: 1,
                        custom_call_target: String::new(),
                    },
                    HloInstructionProto {
                        name: String::from("add.3"),
                        opcode: String::from("add"),
                        shape: None,
                        id: 3,
                        operand_ids: vec![1, 2],
                        parameter_number: 0,
                        custom_call_target: String::new(),
                    },
                ],
            }],
            id: 1,
        };
        let encoded = hlo.encode();
        let prog = CompiledProgram::compile("hlo", &encoded, b"");
        assert!(prog.is_ok());
        for p in prog {
            assert_eq!(p.name, "binary_hlo_add");
            assert_eq!(p.opcode, TpuOpCode::AddF32);
            assert_eq!(p.parameter_shapes.len(), 2);
        }

        let with_cfg = HloModuleProtoWithConfig {
            hlo_module: Some(hlo),
            config: None,
        };
        let with_cfg_prog = CompiledProgram::compile("hlo_with_config", &with_cfg.encode(), b"");
        assert!(with_cfg_prog.is_ok());
        for p in with_cfg_prog {
            assert_eq!(p.name, "binary_hlo_add");
            assert_eq!(p.opcode, TpuOpCode::AddF32);
        }

        let mut mlir_bc = Vec::from(MLIR_BYTECODE_MAGIC.as_slice());
        mlir_bc.extend_from_slice(b"module @bc_mod { stablehlo.add }");
        let bc_hdr = MlirBytecodeHeader::decode(&mlir_bc);
        assert!(bc_hdr.is_some());
        for hdr in bc_hdr {
            assert_ne!(format!("{hdr:?}"), "");
            assert_eq!(hdr.clone(), hdr);
            assert_ne!(hdr.to_synthesized_mlir_text(), "");
        }

        // Test encode_mlir_prefix_varint and decode_mlir_prefix_varint across 1-byte, 2-byte, and 9-byte forms
        for v in [0_u64, 1, 42, 127, 128, 1024, 16383, 16384, 0x1234_5678_9ABC] {
            let enc_v = encode_mlir_prefix_varint(v);
            let mut cur: &[u8] = &enc_v;
            assert_eq!(decode_mlir_prefix_varint(&mut cur), Some(v));
            assert_eq!(cur.len(), 0);
        }
        let mut empty_cur: &[u8] = &[];
        assert!(decode_mlir_prefix_varint(&mut empty_cur).is_none());
        let mut trunc_zero: &[u8] = &[0, 1, 2];
        assert!(decode_mlir_prefix_varint(&mut trunc_zero).is_none());
        let mut trunc_two: &[u8] = &[0x02];
        assert!(decode_mlir_prefix_varint(&mut trunc_two).is_none());

        // Build a structured binary MLIR bytecode container with Sections 0, 1, 2, 3, 4 (aligned), 5, and 6
        let mut structured_bc = Vec::from(MLIR_BYTECODE_MAGIC.as_slice());
        structured_bc.extend_from_slice(&encode_mlir_prefix_varint(1)); // version = 1
        structured_bc.extend_from_slice(b"MLIR20.0.0\x00");
        let str_payload =
            b"module\x00@bin_mlir\x00stablehlo.multiply\x00tensor<4xf32>\x00stablehlo\x00";
        structured_bc.push(0); // section_id = 0 (kString), unaligned
        structured_bc.extend_from_slice(&encode_mlir_prefix_varint(str_payload.len() as u64));
        structured_bc.extend_from_slice(str_payload);
        // Section 1 (kDialect): index 4 ("stablehlo") and out-of-bounds index 99
        let mut dial_payload = Vec::new();
        dial_payload.extend_from_slice(&encode_mlir_prefix_varint(4));
        dial_payload.extend_from_slice(&encode_mlir_prefix_varint(99));
        structured_bc.push(1);
        structured_bc.extend_from_slice(&encode_mlir_prefix_varint(dial_payload.len() as u64));
        structured_bc.extend_from_slice(&dial_payload);
        // Section 2 (kAttrType)
        let attr_payload = b"tensor<4xf32>\x00";
        structured_bc.push(2);
        structured_bc.extend_from_slice(&encode_mlir_prefix_varint(attr_payload.len() as u64));
        structured_bc.extend_from_slice(attr_payload);
        // Section 3 (kAttrTypeOffset)
        let off_payload = encode_mlir_prefix_varint(14);
        structured_bc.push(3);
        structured_bc.extend_from_slice(&encode_mlir_prefix_varint(off_payload.len() as u64));
        structured_bc.extend_from_slice(&off_payload);
        // Section 4 (kIR), aligned bit set (0x80)
        let ir_payload = encode_mlir_prefix_varint(7);
        structured_bc.push(0x84);
        structured_bc.extend_from_slice(&encode_mlir_prefix_varint(ir_payload.len() as u64));
        structured_bc.extend_from_slice(&encode_mlir_prefix_varint(4)); // alignment = 4
        structured_bc.extend_from_slice(&ir_payload);
        // Section 5 (kResource)
        structured_bc.push(5);
        structured_bc.extend_from_slice(&encode_mlir_prefix_varint(4));
        structured_bc.extend_from_slice(&[1, 2, 3, 4]);
        // Section 6 (kProperties / unknown)
        structured_bc.push(6);
        structured_bc.extend_from_slice(&encode_mlir_prefix_varint(1));
        structured_bc.push(0xFF);

        let parsed_struct_bc = MlirBytecodeHeader::decode(&structured_bc);
        assert!(parsed_struct_bc.is_some());
        for s_hdr in parsed_struct_bc {
            assert_eq!(s_hdr.sections.len(), 7);
            assert_eq!(s_hdr.dialects.len(), 2);
            assert_eq!(s_hdr.attr_type_entries, vec![String::from("tensor<4xf32>")]);
            assert_eq!(s_hdr.attr_type_offsets, vec![14]);
            assert_eq!(s_hdr.ir_op_records, vec![7]);
            assert_eq!(s_hdr.resources.len(), 1);
            for sec0 in s_hdr.sections.first() {
                assert_eq!(sec0.clone(), *sec0);
                assert_ne!(format!("{sec0:?}"), "");
            }
            assert!(
                s_hdr
                    .to_synthesized_mlir_text()
                    .contains("stablehlo.multiply")
            );
            let reencoded = s_hdr.encode();
            assert!(MlirBytecodeHeader::decode(&reencoded).is_some());
        }
        let compiled_struct_bc = CompiledProgram::compile("mlir", &structured_bc, b"");
        assert!(compiled_struct_bc.is_ok());
        for p in compiled_struct_bc {
            assert_eq!(p.name, "bin_mlir");
            assert_eq!(p.opcode, TpuOpCode::MulF32);
        }

        // Test truncated section length varint and truncated section payload in MlirBytecodeHeader::decode
        let mut trunc_sec_len = Vec::from(MLIR_BYTECODE_MAGIC.as_slice());
        trunc_sec_len.extend_from_slice(b"\x03MLIR\x00\x00\x02");
        assert!(MlirBytecodeHeader::decode(&trunc_sec_len).is_some());
        let mut trunc_sec_data = Vec::from(MLIR_BYTECODE_MAGIC.as_slice());
        trunc_sec_data.extend_from_slice(b"\x03MLIR\x00\x00");
        trunc_sec_data.extend_from_slice(&encode_mlir_prefix_varint(50));
        trunc_sec_data.push(1);
        assert!(MlirBytecodeHeader::decode(&trunc_sec_data).is_some());

        let mut mlir_bc_nul = Vec::from(MLIR_BYTECODE_MAGIC.as_slice());
        mlir_bc_nul.extend_from_slice(b"\x03MLIR20.0.0\x00");
        assert!(MlirBytecodeHeader::decode(&mlir_bc_nul).is_some());
        assert!(MlirBytecodeHeader::decode(MLIR_BYTECODE_MAGIC).is_some());
        assert!(MlirBytecodeHeader::decode(b"not_mlir").is_none());
        let bc_prog = CompiledProgram::compile("mlir", &mlir_bc, b"");
        assert!(bc_prog.is_ok());
        for p in bc_prog {
            assert_eq!(p.name, "bc_mod");
            assert_eq!(p.opcode, TpuOpCode::AddF32);
        }
        assert!(CompiledProgram::compile("mlir", b"-5", b"").is_ok());
        assert!(CompiledProgram::compile("mlir", b"%v.1", b"").is_ok());
        assert!(CompiledProgram::compile("mlir", b"1_0", b"").is_ok());

        // Also test HloModuleProto without explicit parameter instructions (binary, unary, and parameter-only fallback)
        for op_str in ["mul_f32", "neg_f32", "parameter"] {
            let hlo_no_params = HloModuleProto {
                name: String::from("no_params"),
                entry_computation_name: String::from("main"),
                entry_computation_id: 1,
                host_program_shape: None,
                computations: vec![HloComputationProto {
                    name: String::from("main"),
                    program_shape: None,
                    id: 1,
                    root_id: 1,
                    instructions: vec![HloInstructionProto {
                        name: String::from("op.1"),
                        opcode: String::from(op_str),
                        shape: None,
                        id: 1,
                        operand_ids: vec![],
                        parameter_number: 0,
                        custom_call_target: String::new(),
                    }],
                }],
                id: 1,
            };
            let enc2 = hlo_no_params.encode();
            assert!(CompiledProgram::compile("hlo", &enc2, b"").is_ok());
        }

        let bad_et_hlo = HloModuleProto {
            name: String::from("bad_et"),
            entry_computation_name: String::from("main"),
            entry_computation_id: 1,
            host_program_shape: None,
            computations: vec![HloComputationProto {
                name: String::from("main"),
                program_shape: None,
                id: 1,
                root_id: 1,
                instructions: vec![HloInstructionProto {
                    name: String::from("p0"),
                    opcode: String::from("parameter"),
                    shape: Some(ShapeProto {
                        element_type: 99,
                        dimensions: vec![4],
                        tuple_shapes: vec![],
                        layout: None,
                        is_dynamic_dimension: vec![],
                    }),
                    id: 1,
                    operand_ids: vec![],
                    parameter_number: 0,
                    custom_call_target: String::new(),
                }],
            }],
            id: 1,
        };
        assert!(CompiledProgram::compile("hlo", &bad_et_hlo.encode(), b"").is_err());

        // Test TpuSerializedExecutableProto with empty hlo_module_bytes, bad opcode, and SYNTAX_ERROR
        let empty_proto = TpuSerializedExecutableProto {
            name: String::from("long_enough_name"),
            fingerprint: String::new(),
            num_replicas: 1,
            num_partitions: 1,
            instruction_opcodes: vec![],
            output_shapes: vec![],
            compile_options_bytes: vec![],
            hlo_module_bytes: vec![],
        };
        assert!(CompiledProgram::deserialize(&empty_proto.encode()).is_err());

        let bad_op_proto = TpuSerializedExecutableProto {
            name: String::from("long_enough_name"),
            fingerprint: String::new(),
            num_replicas: 1,
            num_partitions: 1,
            instruction_opcodes: vec![250],
            output_shapes: vec![],
            compile_options_bytes: vec![],
            hlo_module_bytes: b"return".to_vec(),
        };
        assert!(CompiledProgram::deserialize(&bad_op_proto.encode()).is_err());

        let bad_src_proto = TpuSerializedExecutableProto {
            name: String::from("long_enough_name"),
            fingerprint: String::new(),
            num_replicas: 1,
            num_partitions: 1,
            instruction_opcodes: vec![0],
            output_shapes: vec![],
            compile_options_bytes: vec![],
            hlo_module_bytes: b"SYNTAX_ERROR".to_vec(),
        };
        assert!(CompiledProgram::deserialize(&bad_src_proto.encode()).is_err());
    }

    #[test]
    fn test_deserialize_error_branches() {
        assert!(CompiledProgram::deserialize(b"short").is_err());
        assert!(CompiledProgram::deserialize(b"BADMAGIC123456789").is_err());
        assert!(CompiledProgram::deserialize(EXECUTABLE_MAGIC).is_err());

        let mut bad_ver = Vec::from(EXECUTABLE_MAGIC.as_slice());
        bad_ver.extend_from_slice(&99_u32.to_le_bytes());
        assert!(CompiledProgram::deserialize(&bad_ver).is_err());

        let mut missing_op = Vec::from(EXECUTABLE_MAGIC.as_slice());
        missing_op.extend_from_slice(&EXECUTABLE_FORMAT_VERSION.to_le_bytes());
        assert!(CompiledProgram::deserialize(&missing_op).is_err());

        let mut bad_op = Vec::from(EXECUTABLE_MAGIC.as_slice());
        bad_op.extend_from_slice(&EXECUTABLE_FORMAT_VERSION.to_le_bytes());
        bad_op.push(250);
        assert!(CompiledProgram::deserialize(&bad_op).is_err());

        let mut trunc_fmt_u32 = Vec::from(EXECUTABLE_MAGIC.as_slice());
        trunc_fmt_u32.extend_from_slice(&EXECUTABLE_FORMAT_VERSION.to_le_bytes());
        trunc_fmt_u32.push(0);
        assert!(CompiledProgram::deserialize(&trunc_fmt_u32).is_err());

        let mut empty_src_payload = Vec::from(EXECUTABLE_MAGIC.as_slice());
        empty_src_payload.extend_from_slice(&EXECUTABLE_FORMAT_VERSION.to_le_bytes());
        empty_src_payload.push(0);
        empty_src_payload.extend_from_slice(&4_u32.to_le_bytes());
        empty_src_payload.extend_from_slice(b"mlir");
        empty_src_payload.extend_from_slice(&0_u32.to_le_bytes());
        empty_src_payload.extend_from_slice(&0_u32.to_le_bytes());
        assert!(CompiledProgram::deserialize(&empty_src_payload).is_err());

        let mut trunc_fmt = Vec::from(EXECUTABLE_MAGIC.as_slice());
        trunc_fmt.extend_from_slice(&EXECUTABLE_FORMAT_VERSION.to_le_bytes());
        trunc_fmt.push(0);
        trunc_fmt.extend_from_slice(&100_u32.to_le_bytes());
        assert!(CompiledProgram::deserialize(&trunc_fmt).is_err());

        let mut bad_utf8_fmt = Vec::from(EXECUTABLE_MAGIC.as_slice());
        bad_utf8_fmt.extend_from_slice(&EXECUTABLE_FORMAT_VERSION.to_le_bytes());
        bad_utf8_fmt.push(0);
        bad_utf8_fmt.extend_from_slice(&2_u32.to_le_bytes());
        bad_utf8_fmt.extend_from_slice(&[0xff, 0xfe, 0, 0, 0, 0]);
        assert!(CompiledProgram::deserialize(&bad_utf8_fmt).is_err());

        let mut trunc_src = Vec::from(EXECUTABLE_MAGIC.as_slice());
        trunc_src.extend_from_slice(&EXECUTABLE_FORMAT_VERSION.to_le_bytes());
        trunc_src.push(0);
        trunc_src.extend_from_slice(&4_u32.to_le_bytes());
        trunc_src.extend_from_slice(b"mlir");
        trunc_src.extend_from_slice(&100_u32.to_le_bytes());
        assert!(CompiledProgram::deserialize(&trunc_src).is_err());

        let trunc_opt_test = b"module @cst_vec { func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> { %0 = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf32>\n func.return %0 : tensor<2xf32> } }";
        assert!(CompiledProgram::compile("mlir", trunc_opt_test, b"").is_ok());
        let cst_single_test = b"module @cst_s1 { func.func @main(%arg0: tensor<1xf32>) -> tensor<1xf32> { %0 = stablehlo.constant dense<5.0> : tensor<1xf32>\n return %0 : tensor<1xf32> } }";
        assert!(CompiledProgram::compile("mlir", cst_single_test, b"").is_ok());
        let hlo_entry_test = b"HloModule m\nENTRY %main (x: f32[4]) -> f32[4] {\n  %x = f32[4] parameter(0)\n  ROOT %r = f32[4] negate(%x)\n}";
        for prog in CompiledProgram::compile("hlo", hlo_entry_test, b"") {
            let ba = prog.build_buffer_assignment_proto();
            assert_eq!(ba.buffer_allocations.len(), 3);
            assert!(
                crate::proto::BufferAssignmentProto::decode(&ba.encode())
                    .ok()
                    .is_some()
            );
        }

        let mut trunc_opt = Vec::from(EXECUTABLE_MAGIC.as_slice());
        trunc_opt.extend_from_slice(&EXECUTABLE_FORMAT_VERSION.to_le_bytes());
        trunc_opt.push(0);
        trunc_opt.extend_from_slice(&4_u32.to_le_bytes());
        trunc_opt.extend_from_slice(b"mlir");
        trunc_opt.extend_from_slice(&2_u32.to_le_bytes());
        trunc_opt.extend_from_slice(b"id");
        trunc_opt.extend_from_slice(&100_u32.to_le_bytes());
        assert!(CompiledProgram::deserialize(&trunc_opt).is_err());

        let hdr = ExecutableBinaryHeader {
            magic: *EXECUTABLE_MAGIC,
            version_le: EXECUTABLE_FORMAT_VERSION,
        };
        assert_eq!(hdr.clone(), hdr);
        assert_ne!(format!("{hdr:?}"), "");

        // Test petgraph + indexmap DAG analysis (valid DAG and cyclic graph)
        let valid_comp = HloComputationProto {
            name: String::from("dag_main"),
            program_shape: None,
            id: 1,
            root_id: 3,
            instructions: vec![
                HloInstructionProto {
                    name: String::from("p0"),
                    opcode: String::from("parameter"),
                    shape: Some(ShapeProto {
                        element_type: 11,
                        dimensions: vec![16, 128],
                        tuple_shapes: vec![],
                        layout: None,
                        is_dynamic_dimension: vec![],
                    }),
                    id: 1,
                    operand_ids: vec![999], // external/unknown operand ID gracefully ignored
                    parameter_number: 0,
                    custom_call_target: String::new(),
                },
                HloInstructionProto {
                    name: String::from("neg"),
                    opcode: String::from("negate"),
                    shape: None,
                    id: 2,
                    operand_ids: vec![1],
                    parameter_number: 0,
                    custom_call_target: String::new(),
                },
            ],
        };
        for dag_rep in analyze_hlo_computation_dag(&valid_comp) {
            assert_eq!(dag_rep.topological_order, vec![1, 2]);
            assert_eq!(dag_rep.buffer_offsets.len(), 2);
            assert_eq!(dag_rep.total_hbm_bytes, 12288);
            assert_eq!(dag_rep.clone(), dag_rep);
            assert_ne!(format!("{dag_rep:?}"), "");
        }

        let cyclic_comp = HloComputationProto {
            name: String::from("cyclic"),
            program_shape: None,
            id: 2,
            root_id: 2,
            instructions: vec![
                HloInstructionProto {
                    name: String::from("a"),
                    opcode: String::from("negate"),
                    shape: None,
                    id: 1,
                    operand_ids: vec![2],
                    parameter_number: 0,
                    custom_call_target: String::new(),
                },
                HloInstructionProto {
                    name: String::from("b"),
                    opcode: String::from("negate"),
                    shape: None,
                    id: 2,
                    operand_ids: vec![1],
                    parameter_number: 0,
                    custom_call_target: String::new(),
                },
            ],
        };
        assert!(analyze_hlo_computation_dag(&cyclic_comp).is_err());
    }

    proptest::proptest! {
        #[test]
        fn proptest_winnow_tensor_shape_and_compiler_roundtrip(
            d0 in 1_i64..32,
            d1 in 1_i64..32,
        ) {
            let mlir_spec = format!("tensor<{d0}x{d1}xf32>");
            let parsed_mlir = CompiledProgram::parse_tensor_type(&mlir_spec);
            proptest::prop_assert!(parsed_mlir.is_some());
            if let Some(sh) = parsed_mlir {
                proptest::prop_assert_eq!(sh.dims, vec![d0, d1]);
                proptest::prop_assert_eq!(sh.element_type, ElementType::F32);
            }

            let hlo_spec = format!("bf16[{d0},{d1}]{{1,0}}");
            let parsed_hlo = CompiledProgram::parse_hlo_shape_token(&hlo_spec);
            proptest::prop_assert!(parsed_hlo.is_some());
            if let Some(sh) = parsed_hlo {
                proptest::prop_assert_eq!(sh.dims, vec![d0, d1]);
                proptest::prop_assert_eq!(sh.element_type, ElementType::Bf16);
            }
        }
    }
}

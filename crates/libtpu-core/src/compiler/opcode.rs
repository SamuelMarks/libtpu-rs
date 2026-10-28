//! TPU instruction opcode definitions and operation classification (`TpuOpCode`, `InstructionAttrs`, `TpuInstructionBundle`).

use crate::error::{LibTpuError, LibTpuResult};
use crate::types::Shape;
use num_enum::{IntoPrimitive, TryFromPrimitive};

/// Supported TPU instruction opcodes in a [`super::CompiledProgram`] VLIW instruction bundle.
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
    /// Elementwise exponential minus one (`output[i] = input0[i].exp_m1()`) (`58`).
    Expm1 = 58,
    /// Elementwise natural logarithm of one plus input (`output[i] = input0[i].ln_1p()`) (`59`).
    Log1p = 59,
    /// Elementwise logistic sigmoid (`output[i] = 1.0 / (1.0 + (-input0[i]).exp())`) (`60`).
    Logistic = 60,
    /// Elementwise signum (`-1.0`, `0.0`, or `1.0`) (`61`).
    Sign = 61,
    /// Elementwise finiteness predicate (`output[i] = input0[i].is_finite()`) (`62`).
    IsFinite = 62,
    /// Elementwise tangent (`output[i] = input0[i].tan()`) (`63`).
    Tan = 63,
    /// Elementwise cube root (`output[i] = input0[i].cbrt()`) (`64`).
    Cbrt = 64,
    /// Elementwise round to nearest even integer (`output[i] = input0[i].round_ties_even()`) (`65`).
    RoundNearestEven = 65,
    /// Elementwise round to nearest integer away from zero (`output[i] = input0[i].round()`) (`66`).
    RoundNearestAfz = 66,
    /// Elementwise four-quadrant arc tangent (`output[i] = input0[i].atan2(input1[i])`) (`67`).
    Atan2 = 67,
    /// Elementwise exponentiation (`output[i] = input0[i].powf(input1[i])`) (`68`).
    Pow = 68,
    /// Ternary elementwise clamp (`output[i] = operand[i].clamp(min[i], max[i])`) (`69`).
    Clamp = 69,
    /// Edge and interior tensor padding (`stablehlo.pad`) (`70`).
    Pad = 70,
    /// Lower-triangular Cholesky matrix factorization (`stablehlo.cholesky`) (`71`).
    Cholesky = 71,
    /// Triangular linear system solver (`stablehlo.triangular_solve`) (`72`).
    TriangularSolve = 72,
    /// Fast Fourier Transform magnitude/real projection (`stablehlo.fft`) (`73`).
    Fft = 73,
    /// Top-K largest elements selection (`chlo.top_k` / `ApproxTopK`) (`74`).
    TopK = 74,
    /// Uniform affine quantization (`stablehlo.uniform_quantize`) (`75`).
    UniformQuantize = 75,
    /// Uniform affine dequantization (`stablehlo.uniform_dequantize`) (`76`).
    UniformDequantize = 76,
    /// Cross-replica ICI `CollectivePermute` (`stablehlo.collective_permute`) (`77`).
    CollectivePermute = 77,
    /// Cross-replica ICI `AllToAll` (`stablehlo.all_to_all`) (`78`).
    AllToAll = 78,
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
            Self::Select | Self::Scatter | Self::DynamicUpdateSlice | Self::Clamp
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
                | Self::Atan2
                | Self::Pow
                | Self::Pad
                | Self::TriangularSolve
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
                | Self::Expm1
                | Self::Log1p
                | Self::Logistic
                | Self::Tan
                | Self::Cbrt
                | Self::Atan2
                | Self::Pow
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
    /// Low-edge padding elements per axis for `pad` (`edge_padding_low`).
    pub pad_edge_low: Vec<usize>,
    /// High-edge padding elements per axis for `pad` (`edge_padding_high`).
    pub pad_edge_high: Vec<usize>,
    /// Interior padding elements between adjacent elements per axis for `pad` (`interior_padding`).
    pub pad_interior: Vec<usize>,
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

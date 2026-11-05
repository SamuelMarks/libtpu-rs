//! Strongly-typed Abstract Syntax Tree (`AST`), type system (`MlirType`), and attribute hierarchy (`MlirAttribute`).
//!
//! Supports textual MLIR (`stablehlo`, `mhlo`, `chlo`, `sdy`, `tpu` Mosaic, `func`, `arith`, `scf`, `vector`, `memref`)
//! and textual XLA HLO (`HloModuleAst`).

use super::lexer::TokenSpan;
use super::opcode::TpuOpCode;
use crate::types::{ElementType, MemoryLayout, Shape};
use std::collections::BTreeMap;

/// Supported MLIR and XLA HLO dialect namespaces recognized by the recursive-descent AST parser.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DialectKind {
    /// `OpenXLA` `stablehlo` portable MLIR dialect (`stablehlo.*`).
    StableHlo,
    /// `OpenXLA` versioned `vhlo` dialect (`vhlo.*`).
    Vhlo,
    /// Legacy XLA Meta-HLO dialect (`mhlo.*`).
    Mhlo,
    /// Client HLO high-level decomposition dialect (`chlo.*`).
    Chlo,
    /// `OpenXLA` Shardy SPMD partitioning dialect (`sdy.*`).
    Sdy,
    /// JAX/Pallas TPU Mosaic low-level kernel dialect (`tpu.*`).
    TpuMosaic,
    /// Standard MLIR function dialect (`func.*`).
    Func,
    /// Standard MLIR arithmetic dialect (`arith.*`).
    Arith,
    /// Standard MLIR structured control-flow dialect (`scf.*`).
    Scf,
    /// Standard MLIR hardware vector dialect (`vector.*`).
    Vector,
    /// Standard MLIR memory-reference dialect (`memref.*`).
    MemRef,
    /// Standard MLIR builtin dialect (`builtin.*`, `module`).
    Builtin,
    /// Native XLA HLO textual representation (`HloModule`).
    XlaHlo,
}

impl DialectKind {
    /// Resolves a dialect prefix string (e.g., `"stablehlo"`, `"sdy"`, `"tpu"`) into a [`DialectKind`].
    ///
    /// # Arguments
    ///
    /// * `prefix` - Dialect namespace prefix before `.`.
    #[must_use]
    pub fn from_prefix(prefix: &str) -> Self {
        match prefix.trim().to_ascii_lowercase().as_str() {
            "stablehlo" => Self::StableHlo,
            "vhlo" => Self::Vhlo,
            "mhlo" => Self::Mhlo,
            "chlo" => Self::Chlo,
            "sdy" => Self::Sdy,
            "tpu" => Self::TpuMosaic,
            "func" => Self::Func,
            "arith" => Self::Arith,
            "scf" => Self::Scf,
            "vector" => Self::Vector,
            "memref" => Self::MemRef,
            "builtin" | "module" => Self::Builtin,
            _ => Self::XlaHlo,
        }
    }

    /// Returns the canonical lowercase prefix string for this [`DialectKind`].
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StableHlo => "stablehlo",
            Self::Vhlo => "vhlo",
            Self::Mhlo => "mhlo",
            Self::Chlo => "chlo",
            Self::Sdy => "sdy",
            Self::TpuMosaic => "tpu",
            Self::Func => "func",
            Self::Arith => "arith",
            Self::Scf => "scf",
            Self::Vector => "vector",
            Self::MemRef => "memref",
            Self::Builtin => "builtin",
            Self::XlaHlo => "hlo",
        }
    }
}

/// Function or symbol visibility modifier in MLIR (`public`, `private`, `nested`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Visibility {
    /// Publicly exported symbol (`public`, default for entry functions).
    #[default]
    Public,
    /// Module-private symbol (`private`).
    Private,
    /// Nested symbol visible to parent modules (`nested`).
    Nested,
}

/// Dimension extent inside a [`RankedTensorType`], [`MemRefType`], or [`VectorType`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DimensionSize {
    /// Statically known non-negative dimension extent (e.g., `128`).
    Static(i64),
    /// Unbounded dynamic dimension (`?`).
    Dynamic,
    /// Bounded dynamic dimension with an inclusive upper bound (`<=N` or `?, <=N`).
    Bounded(i64),
}

impl DimensionSize {
    /// Returns the concrete `i64` bound or static extent (`1` for unbounded [`Self::Dynamic`]).
    #[must_use]
    pub const fn concrete_or_bound(self) -> i64 {
        match self {
            Self::Static(d) | Self::Bounded(d) => d,
            Self::Dynamic => 1,
        }
    }

    /// Returns `true` if this dimension is [`Self::Dynamic`] or [`Self::Bounded`].
    #[must_use]
    pub const fn is_dynamic(self) -> bool {
        matches!(self, Self::Dynamic | Self::Bounded(_))
    }
}

/// Physical TPU memory space annotation on a [`MemRefType`] or XLA HLO layout (`S(k)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TpuMemorySpace {
    /// High-Bandwidth Memory (`hbm`, default space `0`).
    #[default]
    Hbm,
    /// Vector Memory scratchpad (`vmem`, space `1`).
    Vmem,
    /// Scalar Memory scratchpad (`smem`, space `2`).
    Smem,
    /// Collector Memory (`cmem`, space `3`).
    Cmem,
    /// Hardware semaphore memory (`semaphore_mem`, space `4`).
    SemaphoreMem,
}

impl TpuMemorySpace {
    /// Parses a Mosaic memory space identifier (`"vmem"`, `"smem"`, `"cmem"`, `"hbm"`, `"semaphore_mem"`).
    ///
    /// # Arguments
    ///
    /// * `raw` - Memory space identifier string.
    #[must_use]
    pub fn from_ident(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "hbm" | "0" => Some(Self::Hbm),
            "vmem" | "1" => Some(Self::Vmem),
            "smem" | "2" => Some(Self::Smem),
            "cmem" | "3" => Some(Self::Cmem),
            "semaphore_mem" | "4" => Some(Self::SemaphoreMem),
            _ => None,
        }
    }
}

/// Hardware semaphore type kind in the `tpu` Mosaic dialect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TpuSemaphoreKind {
    /// Standard core-to-core or barrier synchronization semaphore (`!tpu.semaphore`).
    Regular,
    /// Asynchronous ICI/HBM/VMEM DMA completion semaphore (`!tpu.dma_semaphore`).
    Dma,
}

/// Memory layout specification attached to an MLIR [`MemRefType`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemRefLayoutAttr {
    /// Strided layout (`strided<[s0, s1], offset: k>`).
    Strided {
        /// Per-dimension strides in elements.
        strides: Vec<i64>,
        /// Base element offset (`None` if dynamic `?`).
        offset: Option<i64>,
    },
    /// Affine map layout (`affine_map<(d0, d1) -> (d0 * 128 + d1)>`).
    AffineMap(AffineMapExpr),
    /// TPU Mosaic tiled layout (`#tpu.tiled<(8, 128), ...>`).
    Tiled {
        /// Tile dimensions (e.g., `[8, 128]`).
        tile_dims: Vec<i64>,
    },
}

/// Optional encoding attribute attached to a [`RankedTensorType`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorEncodingAttr {
    /// Bounded dynamic dimension sizes (`#stablehlo.bounds<16, ?>`).
    Bounds(Vec<Option<i64>>),
    /// Shardy tensor sharding annotation (`#sdy.sharding<...>`).
    SdySharding(SdyTensorShardingAttr),
    /// Raw or custom dialect encoding string.
    Custom(String),
}

/// Ranked tensor type (`tensor<d0xd1xElementType, encoding>`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankedTensorType {
    /// Per-axis dimension extents (`Static`, `Dynamic`, or `Bounded`).
    pub dims: Vec<DimensionSize>,
    /// Scalar element type.
    pub element_type: ElementType,
    /// Optional tensor encoding attribute (bounds, sharding, etc.).
    pub encoding: Option<TensorEncodingAttr>,
    /// Optional physical layout (when parsed from XLA HLO `f32[8,128]{1,0:T(8,128)}`).
    pub layout: Option<MemoryLayout>,
}

impl RankedTensorType {
    /// Converts this [`RankedTensorType`] into a runtime [`Shape`].
    #[must_use]
    pub fn to_shape(&self) -> Shape {
        let dims: Vec<i64> = self
            .dims
            .iter()
            .map(|d| d.concrete_or_bound().max(1))
            .collect();
        let mut dynamic_dims: Vec<bool> = self.dims.iter().map(|d| d.is_dynamic()).collect();
        if let Some(TensorEncodingAttr::Bounds(bounds)) = &self.encoding {
            for (slot, b) in dynamic_dims.iter_mut().zip(bounds.iter()) {
                *slot = *slot || b.is_some();
            }
        }
        let layout = self
            .layout
            .clone()
            .unwrap_or_else(|| MemoryLayout::default_tpu_tiled(dims.len()));
        Shape {
            element_type: self.element_type,
            dims,
            dynamic_dims,
            layout,
        }
    }
}

/// MLIR `memref<...>` buffer type used in `tpu` Mosaic and `memref` dialects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemRefType {
    /// Per-axis dimension extents.
    pub dims: Vec<DimensionSize>,
    /// Scalar element type.
    pub element_type: ElementType,
    /// Optional strided, affine, or tiled layout attribute.
    pub layout: Option<MemRefLayoutAttr>,
    /// Optional target TPU memory space (`Vmem`, `Smem`, `Hbm`, `Cmem`, `SemaphoreMem`).
    pub memory_space: Option<TpuMemorySpace>,
}

impl MemRefType {
    /// Converts this [`MemRefType`] into a runtime [`Shape`].
    #[must_use]
    pub fn to_shape(&self) -> Shape {
        let dims: Vec<i64> = self
            .dims
            .iter()
            .map(|d| d.concrete_or_bound().max(1))
            .collect();
        let dynamic_dims: Vec<bool> = self.dims.iter().map(|d| d.is_dynamic()).collect();
        Shape {
            element_type: self.element_type,
            layout: MemoryLayout::default_tpu_tiled(dims.len()),
            dims,
            dynamic_dims,
        }
    }
}

/// MLIR hardware `vector<...>` register type used in `tpu` Mosaic and `vector` dialects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorType {
    /// Per-axis vector lane extents.
    pub dims: Vec<DimensionSize>,
    /// Flags indicating which dimensions are scalable (`[8]x128`).
    pub scalable_dims: Vec<bool>,
    /// Scalar element type.
    pub element_type: ElementType,
}

impl VectorType {
    /// Converts this [`VectorType`] into a runtime [`Shape`].
    #[must_use]
    pub fn to_shape(&self) -> Shape {
        let dims: Vec<i64> = self
            .dims
            .iter()
            .map(|d| d.concrete_or_bound().max(1))
            .collect();
        let dynamic_dims: Vec<bool> = self.dims.iter().map(|d| d.is_dynamic()).collect();
        Shape {
            element_type: self.element_type,
            layout: MemoryLayout::default_tpu_tiled(dims.len()),
            dims,
            dynamic_dims,
        }
    }
}

/// Uniform affine quantized element type (`!quant.uniform<StorageType:ExpressedType, scale:zero_point>`).
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedElementType {
    /// Underlying integer storage type (e.g., `S8`, `U8`, `S4`).
    pub storage_type: ElementType,
    /// Expressed floating-point type (e.g., `F32`, `Bf16`).
    pub expressed_type: ElementType,
    /// Per-tensor or per-channel quantization scales.
    pub scales: Vec<f64>,
    /// Per-tensor or per-channel quantization zero points.
    pub zero_points: Vec<i64>,
    /// Optional quantized axis index for per-channel quantization.
    pub quantized_dimension: Option<usize>,
}

impl Eq for QuantizedElementType {}

/// Function or operation type signature `(inputs...) -> (outputs...)`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AstFunctionType {
    /// Ordered input parameter or operand types.
    pub inputs: Vec<MlirType>,
    /// Ordered output result types.
    pub results: Vec<MlirType>,
}

/// Complete strongly-typed MLIR and XLA HLO type hierarchy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MlirType {
    /// Ranked tensor (`tensor<4x8xf32>`).
    RankedTensor(RankedTensorType),
    /// Unranked tensor (`tensor<*xf32>`).
    UnrankedTensor(ElementType),
    /// Memory reference buffer (`memref<8x128xbf16, #tpu.memory_space<vmem>>`).
    MemRef(MemRefType),
    /// Hardware vector register (`vector<8x128xf32>`).
    Vector(VectorType),
    /// Heterogeneous tuple of types (`tuple<tensor<4xf32>, tensor<4xs32>>`).
    Tuple(Vec<Self>),
    /// Ordering token (`!stablehlo.token`, `!mhlo.token`, HLO `token[]`).
    Token,
    /// Uniform quantized element or tensor type (`!quant.uniform<...>`).
    Quantized(QuantizedElementType),
    /// TPU Mosaic hardware semaphore (`!tpu.semaphore` or `!tpu.dma_semaphore`).
    Semaphore(TpuSemaphoreKind),
    /// Bare scalar or index type (`i32`, `f32`, `index`, `i1`).
    Scalar(ElementType),
    /// Functional mapping type (`(T1, T2) -> R1`).
    Function(AstFunctionType),
}

impl MlirType {
    /// Extracts the primary [`ElementType`] of this [`MlirType`].
    #[must_use]
    pub fn element_type(&self) -> ElementType {
        match self {
            Self::RankedTensor(rt) => rt.element_type,
            Self::UnrankedTensor(et) | Self::Scalar(et) => *et,
            Self::MemRef(mr) => mr.element_type,
            Self::Vector(vt) => vt.element_type,
            Self::Quantized(qt) => qt.storage_type,
            Self::Tuple(elems) => elems.first().map_or(ElementType::F32, Self::element_type),
            Self::Token | Self::Semaphore(_) => ElementType::U32,
            Self::Function(ft) => ft
                .results
                .iter()
                .chain(&ft.inputs)
                .next()
                .map_or(ElementType::F32, Self::element_type),
        }
    }

    /// Converts this [`MlirType`] into a runtime [`Shape`] if applicable.
    #[must_use]
    pub fn to_shape(&self) -> Option<Shape> {
        match self {
            Self::RankedTensor(rt) => Some(rt.to_shape()),
            Self::UnrankedTensor(et) | Self::Scalar(et) => Some(Shape {
                element_type: *et,
                dims: vec![1],
                dynamic_dims: vec![false],
                layout: MemoryLayout::default_tpu_tiled(1),
            }),
            Self::MemRef(mr) => Some(mr.to_shape()),
            Self::Vector(vt) => Some(vt.to_shape()),
            Self::Quantized(qt) => Some(Shape {
                element_type: qt.storage_type,
                dims: vec![1],
                dynamic_dims: vec![false],
                layout: MemoryLayout::default_tpu_tiled(1),
            }),
            Self::Token | Self::Semaphore(_) => Some(Shape {
                element_type: ElementType::U32,
                dims: vec![1],
                dynamic_dims: vec![false],
                layout: MemoryLayout::default_tpu_tiled(1),
            }),
            Self::Tuple(items) => items.first().and_then(Self::to_shape),
            Self::Function(ft) => ft
                .results
                .iter()
                .chain(&ft.inputs)
                .next()
                .and_then(Self::to_shape),
        }
    }

    /// Flattens a [`Self::Tuple`] into its leaf [`Shape`]s, or returns a single-element vector for non-tuples.
    #[must_use]
    pub fn flatten_shapes(&self) -> Vec<Shape> {
        match self {
            Self::Tuple(items) => items.iter().flat_map(Self::flatten_shapes).collect(),
            other => other.to_shape().into_iter().collect(),
        }
    }
}

/// Comparison direction predicate (`EQ`, `NE`, `GE`, `GT`, `LE`, `LT`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonDirection {
    /// Equal (`EQ`).
    Eq,
    /// Not equal (`NE`).
    Ne,
    /// Greater than or equal (`GE`).
    Ge,
    /// Greater than (`GT`).
    Gt,
    /// Less than or equal (`LE`).
    Le,
    /// Less than (`LT`).
    Lt,
}

impl ComparisonDirection {
    /// Parses a comparison direction identifier (`"EQ"`, `"NE"`, `"GE"`, `"GT"`, `"LE"`, `"LT"`).
    ///
    /// # Arguments
    ///
    /// * `raw` - Direction string.
    #[must_use]
    pub fn from_ident(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_uppercase().as_str() {
            "EQ" => Some(Self::Eq),
            "NE" => Some(Self::Ne),
            "GE" => Some(Self::Ge),
            "GT" => Some(Self::Gt),
            "LE" => Some(Self::Le),
            "LT" => Some(Self::Lt),
            _ => None,
        }
    }

    /// Returns the canonical uppercase string representation (`"EQ"`, `"LT"`, etc.).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Eq => "EQ",
            Self::Ne => "NE",
            Self::Ge => "GE",
            Self::Gt => "GT",
            Self::Le => "LE",
            Self::Lt => "LT",
        }
    }
}

/// Comparison semantics category (`FLOAT`, `TOTALORDER`, `SIGNED`, `UNSIGNED`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonType {
    /// IEEE-754 floating-point comparison (`FLOAT`).
    Float,
    /// IEEE-754 totalOrder comparison (`TOTALORDER`).
    TotalOrder,
    /// Two's-complement signed integer comparison (`SIGNED`).
    Signed,
    /// Unsigned integer comparison (`UNSIGNED`).
    Unsigned,
}

impl ComparisonType {
    /// Parses a comparison type identifier (`"FLOAT"`, `"TOTALORDER"`, `"SIGNED"`, `"UNSIGNED"`).
    ///
    /// # Arguments
    ///
    /// * `raw` - Comparison type string.
    #[must_use]
    pub fn from_ident(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_uppercase().as_str() {
            "FLOAT" => Some(Self::Float),
            "TOTALORDER" => Some(Self::TotalOrder),
            "SIGNED" => Some(Self::Signed),
            "UNSIGNED" => Some(Self::Unsigned),
            _ => None,
        }
    }
}

/// MXU systolic array operand precision level (`DEFAULT`, `HIGH`, `HIGHEST`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecisionLevel {
    /// Default (`bf16` pass on TPU MXU).
    Default,
    /// High (`3x bf16` pass on TPU MXU).
    High,
    /// Highest (`6x bf16` / `f32` accumulation pass on TPU MXU).
    Highest,
}

/// Fast Fourier Transform operation kind (`FFT`, `IFFT`, `RFFT`, `IRFFT`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftKind {
    /// Forward complex-to-complex FFT (`FFT`).
    Fft,
    /// Inverse complex-to-complex FFT (`IFFT`).
    Ifft,
    /// Real-to-complex forward FFT (`RFFT`).
    Rfft,
    /// Complex-to-real inverse FFT (`IRFFT`).
    Irfft,
}

/// Pseudo-random number generator algorithm (`DEFAULT`, `THREE_FRY`, `PHILOX`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RngAlgorithmKind {
    /// Backend default PRNG (`DEFAULT`).
    Default,
    /// Counter-based `ThreeFry` PRNG (`THREE_FRY`).
    ThreeFry,
    /// Counter-based Philox PRNG (`PHILOX`).
    Philox,
}

/// Triangular solve matrix transpose mode (`NO_TRANSPOSE`, `TRANSPOSE`, `ADJOINT`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransposeMode {
    /// Do not transpose (`NO_TRANSPOSE`).
    NoTranspose,
    /// Transpose matrix (`TRANSPOSE`).
    Transpose,
    /// Conjugate transpose matrix (`ADJOINT`).
    Adjoint,
}

/// Contraction and batching dimensions for `stablehlo.dot_general` (`#stablehlo.dot<...>`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DotDimensionNumbersAttr {
    /// Left-hand side batching axes.
    pub lhs_batching_dimensions: Vec<usize>,
    /// Right-hand side batching axes.
    pub rhs_batching_dimensions: Vec<usize>,
    /// Left-hand side contracting axes.
    pub lhs_contracting_dimensions: Vec<usize>,
    /// Right-hand side contracting axes.
    pub rhs_contracting_dimensions: Vec<usize>,
}

/// Spatial dimension numbers for `stablehlo.convolution` (`[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ConvDimensionNumbersAttr {
    /// Input batch axis index.
    pub input_batch_dimension: usize,
    /// Input feature axis index.
    pub input_feature_dimension: usize,
    /// Input spatial axis indices.
    pub input_spatial_dimensions: Vec<usize>,
    /// Kernel input feature axis index.
    pub kernel_input_feature_dimension: usize,
    /// Kernel output feature axis index.
    pub kernel_output_feature_dimension: usize,
    /// Kernel spatial axis indices.
    pub kernel_spatial_dimensions: Vec<usize>,
    /// Output batch axis index.
    pub output_batch_dimension: usize,
    /// Output feature axis index.
    pub output_feature_dimension: usize,
    /// Output spatial axis indices.
    pub output_spatial_dimensions: Vec<usize>,
}

/// Dimension numbers for `stablehlo.gather` (`#stablehlo.gather<...>`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct GatherDimensionNumbersAttr {
    /// Axes in the output tensor that correspond to slice dimensions.
    pub offset_dims: Vec<usize>,
    /// Axes in the operand tensor that are collapsed out of the slice.
    pub collapsed_slice_dims: Vec<usize>,
    /// Batching axes in the operand tensor.
    pub operand_batching_dims: Vec<usize>,
    /// Batching axes in the `start_indices` tensor.
    pub start_indices_batching_dims: Vec<usize>,
    /// Map from index vector component to operand axis.
    pub start_index_map: Vec<usize>,
    /// Axis of `start_indices` containing the coordinate vector.
    pub index_vector_dim: usize,
}

/// Dimension numbers for `stablehlo.scatter` (`#stablehlo.scatter<...>`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ScatterDimensionNumbersAttr {
    /// Axes in the `updates` tensor that correspond to window dimensions.
    pub update_window_dims: Vec<usize>,
    /// Axes in the operand tensor where scalar updates are inserted.
    pub inserted_window_dims: Vec<usize>,
    /// Batching axes in the operand tensor.
    pub input_batching_dims: Vec<usize>,
    /// Batching axes in the `scatter_indices` tensor.
    pub scatter_indices_batching_dims: Vec<usize>,
    /// Map from scatter index vector component to operand axis.
    pub scatter_dims_to_operand_dims: Vec<usize>,
    /// Axis of `scatter_indices` containing the coordinate vector.
    pub index_vector_dim: usize,
}

/// Dense multi-dimensional literal payload (`dense<...>`).
#[derive(Debug, Clone, PartialEq)]
pub struct DenseElementsAttr {
    /// Flattened `f64` scalar values (complex numbers stored as interleaved `[re, im]`).
    pub values_f64: Vec<f64>,
    /// Optional raw hex-decoded byte payload (`dense<"0x...">`).
    pub raw_bytes: Option<Vec<u8>>,
    /// Whether the literal was written as a single scalar splat (`dense<2.5>`).
    pub is_splat: bool,
    /// Optional tensor type attached to the dense attribute.
    pub tensor_type: Option<RankedTensorType>,
}

impl Eq for DenseElementsAttr {}

/// Shardy (`sdy`) device mesh declaration (`sdy.mesh @mesh = <["x"=2, "y"=4], device_ids=[...]>`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SdyMeshAttr {
    /// Symbolic mesh name (without `@`).
    pub name: String,
    /// Ordered `(axis_name, axis_size)` pairs.
    pub axes: Vec<(String, usize)>,
    /// Optional explicit logical device ID permutation list.
    pub device_ids: Vec<usize>,
}

impl SdyMeshAttr {
    /// Computes the total number of devices in the mesh (`product(axis_size)`).
    #[must_use]
    pub fn total_devices(&self) -> usize {
        if self.axes.is_empty() {
            return self.device_ids.len().max(1);
        }
        self.axes
            .iter()
            .fold(1_usize, |acc, &(_, sz)| acc.saturating_mul(sz.max(1)))
    }
}

/// Per-dimension axis sharding specification inside an [`SdyTensorShardingAttr`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SdyDimensionSharding {
    /// Ordered mesh axis names partitioning this tensor dimension (e.g., `["x", "y"]`).
    pub axes: Vec<String>,
    /// Whether this dimension is open (`?`) to further Shardy propagation.
    pub is_open: bool,
    /// Optional sharding priority (`p0`, `p1`, etc.).
    pub priority: Option<i64>,
}

/// Shardy (`#sdy.sharding<@mesh, [...], replicated={...}, unreduced={...}>`) attribute.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SdyTensorShardingAttr {
    /// Referenced mesh symbol name (e.g., `"mesh"`) or inline mesh definition.
    pub mesh_name: String,
    /// Per-tensor-dimension sharding specifiers.
    pub dim_shardings: Vec<SdyDimensionSharding>,
    /// Explicitly replicated mesh axes.
    pub replicated_axes: Vec<String>,
    /// Explicitly unreduced mesh axes.
    pub unreduced_axes: Vec<String>,
}

/// TPU Mosaic (`#tpu.*`) dialect-specific attribute value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MosaicDialectAttr {
    /// Grid iteration dimension semantics (`#tpu.dimension_semantics<parallel>` or `arbitrary`).
    DimensionSemantics(String),
    /// Memory space attribute (`#tpu.memory_space<vmem>`).
    MemorySpace(TpuMemorySpace),
    /// Core type attribute (`#tpu.core_type<tensor_core>`).
    CoreType(String),
    /// Pipeline schedule / window parameters dictionary.
    WindowParams(BTreeMap<String, MlirAttribute>),
}

/// Affine map expression (`affine_map<(d0, d1)[s0] -> (d0 + s0, d1)>`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AffineMapExpr {
    /// Dimension parameter identifiers (`d0`, `d1`, ...).
    pub dims: Vec<String>,
    /// Symbol parameter identifiers (`s0`, `s1`, ...).
    pub symbols: Vec<String>,
    /// Result expression strings.
    pub results: Vec<String>,
}

/// Source location metadata (`loc(...)`) attached to an MLIR operation or block argument.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MlirLocation {
    /// Unknown location (`loc(unknown)`).
    Unknown,
    /// File/line/column location (`loc("file.py":10:4)`).
    FileLineCol {
        /// Source file path.
        file: String,
        /// 1-based line number.
        line: usize,
        /// 1-based column number.
        column: usize,
    },
    /// Reference to a location alias (`loc(#loc0)`).
    Alias(String),
    /// Named location (`loc("op_name"(#loc0))`).
    Named {
        /// Location label.
        name: String,
        /// Optional child location.
        child: Option<Box<Self>>,
    },
    /// Callsite location (`loc(callsite(#callee at #caller))`).
    CallSite {
        /// Callee location.
        callee: Box<Self>,
        /// Caller location.
        caller: Box<Self>,
    },
    /// Fused location (`loc(fused[#loc1, #loc2])`).
    Fused(Vec<Self>),
}

/// Strongly-typed MLIR attribute hierarchy (`MlirAttribute`).
#[derive(Debug, Clone, PartialEq)]
pub enum MlirAttribute {
    /// Boolean literal (`true` or `false`).
    Bool(bool),
    /// Signed or unsigned integer literal (`42 : i64`).
    Integer {
        /// Numeric integer value.
        value: i128,
        /// Optional integer type annotation.
        ty: Option<ElementType>,
    },
    /// Floating-point literal (`3.14 : f32`).
    Float {
        /// Numeric float value.
        value: f64,
        /// Optional float type annotation.
        ty: Option<ElementType>,
    },
    /// Decoded string literal (`"tpu_v5p"`).
    String(String),
    /// Type value attribute (`tensor<4xf32>`).
    Type(MlirType),
    /// Flat or nested symbol reference (`@main`, `@mesh::@x`).
    SymbolRef(Vec<String>),
    /// Homogeneous or heterogeneous attribute array (`[...]`).
    Array(Vec<Self>),
    /// Dense 64-bit integer array (`array<i64: 0, 1, 2>`).
    DenseI64Array(Vec<i64>),
    /// Dense boolean array (`array<i1: true, false>`).
    DenseBoolArray(Vec<bool>),
    /// Dense multi-dimensional tensor literal (`dense<...>`).
    DenseElements(DenseElementsAttr),
    /// External dense resource handle (`dense_resource<__elided__>`).
    DenseResource(String),
    /// Nested attribute dictionary (`{key = val, ...}`).
    Dictionary(BTreeMap<String, Self>),
    /// `stablehlo.dot_general` contraction dimension numbers (`#stablehlo.dot<...>`).
    DotDimensionNumbers(DotDimensionNumbersAttr),
    /// `stablehlo.convolution` spatial dimension numbers (`#stablehlo.conv<...>`).
    ConvDimensionNumbers(ConvDimensionNumbersAttr),
    /// `stablehlo.gather` dimension numbers (`#stablehlo.gather<...>`).
    GatherDimensionNumbers(GatherDimensionNumbersAttr),
    /// `stablehlo.scatter` dimension numbers (`#stablehlo.scatter<...>`).
    ScatterDimensionNumbers(ScatterDimensionNumbersAttr),
    /// Comparison direction (`#stablehlo<comparison_direction LT>`).
    ComparisonDirection(ComparisonDirection),
    /// Comparison type (`#stablehlo<comparison_type FLOAT>`).
    ComparisonType(ComparisonType),
    /// MXU precision configuration (`[#stablehlo<precision HIGH>, ...]`).
    PrecisionConfig(Vec<PrecisionLevel>),
    /// FFT kind (`#stablehlo<fft_type FFT>`).
    FftType(FftKind),
    /// PRNG algorithm (`#stablehlo<rng_algorithm THREE_FRY>`).
    RngAlgorithm(RngAlgorithmKind),
    /// Triangular solve transpose mode (`#stablehlo<transpose NO_TRANSPOSE>`).
    TransposeKind(TransposeMode),
    /// Cross-device communication channel handle (`#stablehlo.channel_handle<handle = 1, type = 2>`).
    ChannelHandle {
        /// Channel identifier.
        handle: i64,
        /// Channel type.
        channel_type: i64,
    },
    /// Shardy mesh definition (`#sdy.mesh<...>`).
    SdyMesh(SdyMeshAttr),
    /// Shardy tensor sharding (`#sdy.sharding<...>`).
    SdyTensorSharding(SdyTensorShardingAttr),
    /// Shardy per-value tensor sharding list (`#sdy.sharding_per_value<[...]>`).
    SdyTensorShardingPerValue(Vec<SdyTensorShardingAttr>),
    /// TPU Mosaic dialect attribute (`#tpu.*`).
    MosaicAttr(MosaicDialectAttr),
    /// Affine map attribute (`affine_map<...>`).
    AffineMap(AffineMapExpr),
    /// Source location attribute (`loc(...)`).
    Location(MlirLocation),
}

impl Eq for MlirAttribute {}

impl MlirAttribute {
    /// Extracts a `Vec<usize>` from an integer [`Self::Array`], [`Self::DenseI64Array`], or [`Self::DenseElements`].
    #[must_use]
    pub fn as_usize_vec(&self) -> Vec<usize> {
        match self {
            Self::DenseI64Array(vals) => vals
                .iter()
                .filter_map(|&v| usize::try_from(v).ok())
                .collect(),
            Self::Array(items) => items
                .iter()
                .filter_map(|item| match item {
                    Self::Integer { value, .. } => usize::try_from(*value).ok(),
                    _ => None,
                })
                .collect(),
            Self::DenseElements(de) => de
                .values_f64
                .iter()
                .filter_map(|&v| {
                    if v >= 0.0 {
                        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                        Some(v as usize)
                    } else {
                        None
                    }
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Extracts a single `usize` from an [`Self::Integer`] attribute.
    #[must_use]
    pub fn as_usize(&self) -> Option<usize> {
        match self {
            Self::Integer { value, .. } => usize::try_from(*value).ok(),
            _ => None,
        }
    }

    /// Extracts a string slice from a [`Self::String`] or [`Self::SymbolRef`] attribute.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            Self::SymbolRef(parts) => parts.last().map(String::as_str),
            _ => None,
        }
    }
}

/// Strongly-typed operation classification across all supported dialects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AstOpKind {
    /// Direct hardware/evaluator [`TpuOpCode`] operation.
    HardwareOp(TpuOpCode),
    /// Function return or region terminator (`func.return`, `stablehlo.return`, `sdy.return`, `scf.yield`).
    Return,
    /// Tuple construction (`stablehlo.tuple`, `mhlo.tuple`, HLO `tuple`).
    Tuple,
    /// Tuple element extraction (`stablehlo.get_tuple_element`, HLO `get-tuple-element`).
    GetTupleElement {
        /// Zero-based tuple element index.
        index: usize,
    },
    /// Parameter input instruction in XLA HLO (`parameter(k)`).
    Parameter {
        /// Zero-based parameter index.
        index: usize,
    },
    /// Reduction operation with nested combiner region (`stablehlo.reduce`, `mhlo.reduce`, HLO `reduce`).
    Reduce,
    /// `AllReduce` collective with nested combiner region (`stablehlo.all_reduce`).
    AllReduce,
    /// `ReduceScatter` collective with nested combiner region (`stablehlo.reduce_scatter`).
    ReduceScatter,
    /// Structured `while` loop with condition and body regions (`stablehlo.while`, `scf.while`, HLO `while`).
    While,
    /// Conditional `if` / `case` branch operation with nested regions (`stablehlo.if`, `stablehlo.case`, `scf.if`).
    Conditional,
    /// Shardy SPMD `sdy.manual_computation` region wrapper.
    SdyManualComputation,
    /// Shardy `sdy.sharding_constraint` or `sdy.reshard` annotation op.
    SdyShardingConstraint,
    /// Shardy `sdy.named_computation` or `sdy.data_flow_edge`.
    SdyDataFlow,
    /// Client HLO (`chlo.broadcast_*`, `chlo.erf`, `chlo.lgamma`, etc.) op requiring decomposition.
    ChloDecomposable(String),
    /// TPU Mosaic asynchronous DMA initiation (`tpu.enqueue_dma`, `tpu.dma_copy`).
    MosaicDmaStart,
    /// TPU Mosaic asynchronous DMA wait (`tpu.wait_dma`, `tpu.dma_wait`).
    MosaicDmaWait,
    /// TPU Mosaic hardware semaphore operation (`tpu.semaphore_signal`, `tpu.semaphore_wait`, `tpu.semaphore_read`, `tpu.barrier`).
    MosaicSemaphore(String),
    /// TPU Mosaic VMEM/SMEM load or store (`tpu.vector_load`, `tpu.vector_store`, `tpu.load`, `tpu.store`, `memref.load`, `memref.store`).
    MosaicMemoryOp(String),
    /// Structured `scf.for` loop inside a Mosaic kernel.
    ScfFor,
    /// Call to another function or HLO computation (`func.call`, HLO `call`).
    Call(String),
    /// Metadata-only or no-op barrier (`optimization_barrier`, `after_all`, `create_token`, `sdy.mesh`).
    BarrierOrToken,
}

/// SSA result defined on the left-hand side of an [`AstOperation`] (`%0`, `%res:2`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AstOpResult {
    /// Base SSA value name including `%` (e.g., `"%0"`, `"%res"`).
    pub name: String,
    /// Number of packed results in this binding (`1` for `%0`, `2` for `%res:2`).
    pub count: usize,
}

/// SSA operand referenced by an [`AstOperation`] (`%arg0`, `%0#1`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AstOperand {
    /// Base SSA value name including `%` (e.g., `"%arg0"`, `"%0"`).
    pub name: String,
    /// Optional multi-result index (`Some(1)` for `%0#1`, `None` for `%0`).
    pub result_index: Option<usize>,
    /// Source span of the operand reference.
    pub span: TokenSpan,
}

/// Block or function argument declaration (`%arg0: tensor<4xf32> {sdy.sharding = ...} loc(...)`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AstBlockArg {
    /// SSA variable name including `%` (e.g., `"%arg0"`).
    pub ssa_name: String,
    /// Argument type.
    pub ty: MlirType,
    /// Optional per-argument attribute dictionary.
    pub attrs: BTreeMap<String, MlirAttribute>,
    /// Optional source location.
    pub location: Option<MlirLocation>,
}

/// Function return type entry (`tensor<4xf32> {sdy.sharding = ...}`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AstResultType {
    /// Result type.
    pub ty: MlirType,
    /// Optional per-result attribute dictionary.
    pub attrs: BTreeMap<String, MlirAttribute>,
}

/// Single operation node inside an [`AstBlock`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AstOperation {
    /// Defined SSA output bindings on the LHS of `=`.
    pub results: Vec<AstOpResult>,
    /// Dialect classification (`StableHlo`, `Mhlo`, `Chlo`, `Sdy`, `TpuMosaic`, etc.).
    pub dialect: DialectKind,
    /// Unqualified operation mnemonic (e.g., `"add"`, `"dot_general"`, `"manual_computation"`).
    pub mnemonic: String,
    /// Strongly-typed operation classification.
    pub op_kind: AstOpKind,
    /// Input SSA operands.
    pub operands: Vec<AstOperand>,
    /// Nested regions attached to this operation (`reduce`, `while`, `if`, `sdy.manual_computation`, `scf.for`).
    pub regions: Vec<AstRegion>,
    /// Successor block labels (`^bb1`, `^bb2`).
    pub successors: Vec<String>,
    /// Named attributes attached to this operation.
    pub attributes: BTreeMap<String, MlirAttribute>,
    /// Functional type signature `(operand_types) -> (result_types)`.
    pub signature: AstFunctionType,
    /// Optional trailing `loc(...)` annotation.
    pub location: Option<MlirLocation>,
    /// Source span of the operation.
    pub span: TokenSpan,
}

/// Basic block (`^bb0(%arg0: T): ...`) inside an [`AstRegion`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AstBlock {
    /// Optional caret label (e.g., `Some("^bb0")`, `None` for implicit entry block).
    pub label: Option<String>,
    /// Block header SSA arguments.
    pub arguments: Vec<AstBlockArg>,
    /// Sequence of operations inside the block.
    pub operations: Vec<AstOperation>,
    /// Source span of the block.
    pub span: TokenSpan,
}

/// Lexical region (`{ ^bb0(...): ... }`) attached to an [`AstFunction`] or [`AstOperation`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AstRegion {
    /// Ordered basic blocks inside the region.
    pub blocks: Vec<AstBlock>,
    /// Source span of the region.
    pub span: TokenSpan,
}

/// Parsed MLIR `func.func` declaration or definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AstFunction {
    /// Function symbol name (without `@`, e.g., `"main"`).
    pub name: String,
    /// Function visibility (`Public`, `Private`, `Nested`).
    pub visibility: Visibility,
    /// Function input arguments.
    pub arguments: Vec<AstBlockArg>,
    /// Function output result types and attributes.
    pub results: Vec<AstResultType>,
    /// Function-level attributes (`attributes { ... }`).
    pub function_attrs: BTreeMap<String, MlirAttribute>,
    /// Function body region.
    pub body: AstRegion,
    /// Source span of the function.
    pub span: TokenSpan,
}

/// Window dimension descriptor in textual XLA HLO (`window={size=3x3 stride=1x1 pad=1_1x1_1}`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct HloWindowSpec {
    /// Window kernel dimensions (`size=3x3`).
    pub size: Vec<usize>,
    /// Window step strides (`stride=1x1`).
    pub stride: Vec<usize>,
    /// Low-edge padding per spatial dimension (`pad=1_1x1_1`).
    pub pad_low: Vec<usize>,
    /// High-edge padding per spatial dimension (`pad=1_1x1_1`).
    pub pad_high: Vec<usize>,
    /// Base operand dilation per spatial dimension (`lhs_dilate=1x1`).
    pub lhs_dilate: Vec<usize>,
    /// Window kernel dilation per spatial dimension (`rhs_dilate=1x1`).
    pub rhs_dilate: Vec<usize>,
}

/// Single instruction inside a textual XLA HLO computation (`[ROOT] %name = f32[4] add(%a, %b), ...`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HloInstructionAst {
    /// Instruction name (normalized without leading `%`, e.g., `"add.1"`).
    pub name: String,
    /// Whether this instruction is marked with the `ROOT` keyword.
    pub is_root: bool,
    /// Parsed instruction output type (`RankedTensor`, `Tuple`, or `Token`).
    pub result_type: MlirType,
    /// XLA HLO opcode string (e.g., `"parameter"`, `"add"`, `"dot"`, `"reduce"`, `"tuple"`).
    pub opcode: String,
    /// Strongly-typed operation kind.
    pub op_kind: AstOpKind,
    /// Operand instruction names (normalized without leading `%`).
    pub operands: Vec<String>,
    /// Parameter index for `parameter(k)` instructions.
    pub parameter_index: Option<usize>,
    /// Tuple element index for `get-tuple-element(%t), index=k`.
    pub tuple_index: Option<usize>,
    /// Referenced sub-computations (`to_apply=%reducer`, `calls=%sub`, `condition=%cond`, `body=%body`).
    pub called_computations: BTreeMap<String, String>,
    /// Parsed `window={...}` specification for `convolution` and `reduce-window`.
    pub window: Option<HloWindowSpec>,
    /// Parsed `dim_labels=b01f_01io->b01f` for `convolution`.
    pub conv_dim_labels: Option<ConvDimensionNumbersAttr>,
    /// Parsed dot contraction and batch dimensions (`lhs_contracting_dims={...}`, etc.).
    pub dot_dims: DotDimensionNumbersAttr,
    /// Parsed dimension list (`dimensions={1,0}`).
    pub dimensions: Vec<usize>,
    /// Parsed slice bounds (`slice={[0:2:1], [1:4:2]}`).
    pub slice_bounds: Vec<(usize, usize, usize)>,
    /// Parsed padding config (`padding=1_1_0x2_2_0`).
    pub padding_config: Vec<(usize, usize, usize)>,
    /// Parsed HLO sharding string (`sharding={devices=[2,2]0,1,2,3}`).
    pub sharding: Option<String>,
    /// Parsed `custom_call_target="..."`.
    pub custom_call_target: Option<String>,
    /// Parsed `backend_config="..."`.
    pub backend_config: Option<String>,
    /// Parsed `channel_id=k`.
    pub channel_id: Option<i64>,
    /// Parsed `replica_groups={{0,1},{2,3}}`.
    pub replica_groups: Vec<Vec<usize>>,
    /// Control predecessor instruction names (`control-predecessors={%a, %b}`).
    pub control_predecessors: Vec<String>,
    /// Additional key-value attributes (`iota_dimension=0`, `direction=LT`, etc.).
    pub extra_attrs: BTreeMap<String, MlirAttribute>,
    /// Source span of the instruction.
    pub span: TokenSpan,
}

/// Parsed textual XLA HLO computation (`[ENTRY] %main (p0: f32[4]) -> f32[4] { ... }`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HloComputationAst {
    /// Computation name (normalized without leading `%`).
    pub name: String,
    /// Whether this computation is prefixed with `ENTRY`.
    pub is_entry: bool,
    /// Optional explicit parameter `(name, MlirType)` declarations in the computation header.
    pub header_params: Vec<(String, MlirType)>,
    /// Optional explicit return type in the computation header.
    pub header_return_type: Option<MlirType>,
    /// Instructions in lexical order.
    pub instructions: Vec<HloInstructionAst>,
    /// Source span of the computation.
    pub span: TokenSpan,
}

/// Top-level unified AST for a parsed MLIR (`stablehlo`, `mhlo`, `chlo`, `sdy`, `tpu` Mosaic) or XLA HLO module.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AstModule {
    /// Optional module symbol name (e.g., `Some("jit_step")` from `module @jit_step` or `HloModule jit_step`).
    pub name: Option<String>,
    /// Module-level attribute dictionary (`module @m attributes { ... }`).
    pub attributes: BTreeMap<String, MlirAttribute>,
    /// Top-level `!alias = type` definitions.
    pub type_aliases: BTreeMap<String, MlirType>,
    /// Top-level `#alias = attr` definitions.
    pub attr_aliases: BTreeMap<String, MlirAttribute>,
    /// Top-level Shardy `sdy.mesh @name = <...>` declarations.
    pub sdy_meshes: BTreeMap<String, SdyMeshAttr>,
    /// Top-level `#loc = loc(...)` location alias definitions.
    pub locations: BTreeMap<String, MlirLocation>,
    /// Parsed MLIR functions (`func.func`).
    pub functions: Vec<AstFunction>,
    /// Top-level module body operations outside of `func.func` (for bare MLIR snippets).
    pub top_level_operations: Vec<AstOperation>,
    /// Parsed textual XLA HLO computations (`ENTRY %main { ... }`, `%reducer { ... }`).
    pub hlo_computations: Vec<HloComputationAst>,
    /// Source span of the module.
    pub span: TokenSpan,
}

//! Complete `OpenXLA` `vhlo` (`v0.9.0`–`v1.11.0+`) bytecode codecs and version passes.
//!
//! Implements all 41 `VhloTypeCode`s, 20 `VhloAttrCode`s, 140 versioned `VHLO_*OpV*` specs,
//! `vhlo -> stablehlo` upgrade, and `stablehlo -> vhlo` portable artifact serialization.

use super::MlirBytecodeHeader;
use super::stream::{DialectBytecodeReader, DialectBytecodeWriter, MlirSectionId};
use crate::compiler::ast::{
    AstFunctionType, AstModule, AstOperation, AstRegion, ComparisonDirection, ComparisonType,
    DenseElementsAttr, DialectKind, DimensionSize, FftKind, MlirAttribute, MlirType,
    PrecisionLevel, QuantizedElementType, RankedTensorType, RngAlgorithmKind, TransposeMode,
};
use crate::compiler::parser::{classify_ast_operation, legalize_chlo_in_ast_module};
use crate::error::{LibTpuError, LibTpuResult};
use crate::types::ElementType;
use std::collections::BTreeMap;

/// Semantic version `(major, minor, patch)` of the `OpenXLA` `vhlo` portable dialect (`v0.9.0`–`v1.11.0+`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VhloVersion {
    /// Major version component (`0` or `1`).
    pub major: u32,
    /// Minor version component (`9` for `0.9.0`; `0..=11+` for `1.x.y`).
    pub minor: u32,
    /// Patch version component.
    pub patch: u32,
}

impl VhloVersion {
    /// Minimum supported portable artifact version (`v0.9.0`).
    pub const MIN_SUPPORTED: Self = Self::new(0, 9, 0);
    /// Canonical `v1.0.0` `StableHLO` 1.0 baseline.
    pub const V1_0_0: Self = Self::new(1, 0, 0);
    /// `v1.1.0` introducing `dynamic_gather_v2` and `all_to_all_v2` / `all_gather_v2` / `all_reduce_v2`.
    pub const V1_1_0: Self = Self::new(1, 1, 0);
    /// `v1.3.0` introducing `collective_permute_v2`, `reduce_scatter_v2`, and `custom_call_v2`.
    pub const V1_3_0: Self = Self::new(1, 3, 0);
    /// `v1.6.0` introducing `dot_general_v2` and `gather_v2`.
    pub const V1_6_0: Self = Self::new(1, 6, 0);
    /// `v1.9.0` introducing `result_accuracy_v1` transcendental `*_v2` ops (`exp_v2`, `log_v2`, `tanh_v2`, etc.).
    pub const V1_9_0: Self = Self::new(1, 9, 0);
    /// Current target `vhlo` version (`v1.11.0`).
    pub const CURRENT: Self = Self::new(1, 11, 0);

    /// Constructs a new [`VhloVersion`] from `(major, minor, patch)`.
    ///
    /// # Arguments
    ///
    /// * `major` - Major version number.
    /// * `minor` - Minor version number.
    /// * `patch` - Patch version number.
    #[must_use]
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Parses a `"major.minor.patch"` string (optionally prefixed with `'v'`) into a [`VhloVersion`].
    ///
    /// # Arguments
    ///
    /// * `raw` - Version string such as `"0.9.0"`, `"v1.9.3"`, or `"1.11.0"`.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::CompilationFailed`] if `raw` does not contain three numeric components.
    pub fn parse_str(raw: &str) -> LibTpuResult<Self> {
        let trimmed = raw.trim().trim_start_matches('v');
        let mut parts = trimmed.split('.');
        let parse_part = |opt: Option<&str>| -> Option<u32> { opt?.parse::<u32>().ok() };
        let (Some(major), Some(minor), Some(patch), None) = (
            parse_part(parts.next()),
            parse_part(parts.next()),
            parse_part(parts.next()),
            parts.next(),
        ) else {
            return Err(LibTpuError::CompilationFailed {
                reason: format!("Invalid VHLO version string '{raw}' (expected X.Y.Z)"),
            });
        };
        Ok(Self {
            major,
            minor,
            patch,
        })
    }

    /// Returns `true` if `self >= Self::MIN_SUPPORTED` (`v0.9.0`).
    #[must_use]
    pub fn is_supported(self) -> bool {
        self >= Self::MIN_SUPPORTED
    }

    /// Encodes this [`VhloVersion`] as a 3-varint `(major, minor, patch)` payload for MLIR Section `7` (`kDialectVersions`).
    #[must_use]
    pub fn encode_dialect_version(self) -> Vec<u8> {
        let mut writer = DialectBytecodeWriter::with_capacity(6);
        writer.write_varint(u64::from(self.major));
        writer.write_varint(u64::from(self.minor));
        writer.write_varint(u64::from(self.patch));
        writer.into_bytes()
    }

    /// Decodes a [`VhloVersion`] from a Section `7` (`kDialectVersions`) byte slice.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Raw `VhloBytecodeVersion` payload (`major`, `minor`, `patch` varints).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MlirBytecodeDecodeError`] if fewer than three varints are present.
    #[allow(clippy::cast_possible_truncation)]
    pub fn decode_dialect_version(bytes: &[u8]) -> LibTpuResult<Self> {
        let mut reader = DialectBytecodeReader::new(bytes, MlirSectionId::DialectVersions.as_u8());
        let major = reader.read_varint()? as u32;
        let minor = reader.read_varint()? as u32;
        let patch = reader.read_varint()? as u32;
        Ok(Self {
            major,
            minor,
            patch,
        })
    }
}

impl std::fmt::Display for VhloVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Complete `vhlo_encoding::TypeCode` (`0..=40`) matching `openxla/stablehlo/dialect/VhloBytecode.cpp`.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VhloTypeCode {
    /// `!vhlo.bool_v1` (`0`).
    BooleanV1 = 0,
    /// `!vhlo.complex_v1<elem>` (`1`).
    ComplexV1 = 1,
    /// `!vhlo.bf16_v1` (`2`).
    FloatBF16V1 = 2,
    /// `!vhlo.f16_v1` (`3`).
    FloatF16V1 = 3,
    /// `!vhlo.f32_v1` (`4`).
    FloatF32V1 = 4,
    /// `!vhlo.f64_v1` (`5`).
    FloatF64V1 = 5,
    /// `!vhlo.f8E4M3B11FNUZ_v1` (`6`).
    FloatF8E4M3B11FNUZV1 = 6,
    /// `!vhlo.f8E4M3FN_v1` (`7`).
    FloatF8E4M3FNV1 = 7,
    /// `!vhlo.f8E5M2_v1` (`8`).
    FloatF8E5M2V1 = 8,
    /// `!vhlo.func_v1` (`9`).
    FunctionV1 = 9,
    /// `!vhlo.index_v1` (`10`).
    IndexV1 = 10,
    /// `!vhlo.i4_v1` (`11`).
    IntegerSI4V1 = 11,
    /// `!vhlo.i8_v1` (`12`).
    IntegerSI8V1 = 12,
    /// `!vhlo.i16_v1` (`13`).
    IntegerSI16V1 = 13,
    /// `!vhlo.i32_v1` (`14`).
    IntegerSI32V1 = 14,
    /// `!vhlo.i64_v1` (`15`).
    IntegerSI64V1 = 15,
    /// `!vhlo.ui4_v1` (`16`).
    IntegerUI4V1 = 16,
    /// `!vhlo.ui8_v1` (`17`).
    IntegerUI8V1 = 17,
    /// `!vhlo.ui16_v1` (`18`).
    IntegerUI16V1 = 18,
    /// `!vhlo.ui32_v1` (`19`).
    IntegerUI32V1 = 19,
    /// `!vhlo.ui64_v1` (`20`).
    IntegerUI64V1 = 20,
    /// `!vhlo.tensor_v1` without encoding (`21`).
    RankedTensorV1 = 21,
    /// `!vhlo.tensor_v1` with `#vhlo.type_extensions_v1` encoding (`22`).
    RankedTensorV1WithEncoding = 22,
    /// `!vhlo.token_v1` (`23`).
    TokenV1 = 23,
    /// `!vhlo.tuple_v1` (`24`).
    TupleV1 = 24,
    /// `!vhlo.uniform_quantized_v1` (`25`).
    UniformQuantizedV1 = 25,
    /// `!vhlo.unranked_tensor_v1` (`26`).
    UnrankedTensorV1 = 26,
    /// `!vhlo.witness_v1` (`27`).
    WitnessV1 = 27,
    /// `!vhlo.f8E4M3FNUZ_v1` (`28`).
    FloatF8E4M3FNUZV1 = 28,
    /// `!vhlo.f8E5M2FNUZ_v1` (`29`).
    FloatF8E5M2FNUZV1 = 29,
    /// `!vhlo.uniform_quantized_per_axis_v1` (`30`).
    UniformQuantizedPerAxisV1 = 30,
    /// `!vhlo.tf32_v1` (`31`).
    FloatTF32V1 = 31,
    /// `!vhlo.i2_v1` (`32`).
    IntegerSI2V1 = 32,
    /// `!vhlo.ui2_v1` (`33`).
    IntegerUI2V1 = 33,
    /// `!vhlo.f4E2M1FN_v1` (`34`).
    FloatF4E2M1FNV1 = 34,
    /// `!vhlo.f6E2M3FN_v1` (`35`).
    FloatF6E2M3FNV1 = 35,
    /// `!vhlo.f6E3M2FN_v1` (`36`).
    FloatF6E3M2FNV1 = 36,
    /// `!vhlo.f8E3M4_v1` (`37`).
    FloatF8E3M4V1 = 37,
    /// `!vhlo.f8E4M3_v1` (`38`).
    FloatF8E4M3V1 = 38,
    /// `!vhlo.f8E8M0FNU_v1` (`39`).
    FloatF8E8M0FNUV1 = 39,
    /// `!vhlo.none_v1` (`40`).
    NoneV1 = 40,
}

impl VhloTypeCode {
    /// Converts a raw varint tag (`0..=40`) into a [`VhloTypeCode`].
    ///
    /// # Arguments
    ///
    /// * `tag` - Raw `PrefixVarInt` tag from Section `2` (`kAttrType`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MlirBytecodeDecodeError`] if `tag > 40`.
    pub fn from_tag(tag: u64) -> LibTpuResult<Self> {
        const ALL: [VhloTypeCode; 41] = [
            VhloTypeCode::BooleanV1,
            VhloTypeCode::ComplexV1,
            VhloTypeCode::FloatBF16V1,
            VhloTypeCode::FloatF16V1,
            VhloTypeCode::FloatF32V1,
            VhloTypeCode::FloatF64V1,
            VhloTypeCode::FloatF8E4M3B11FNUZV1,
            VhloTypeCode::FloatF8E4M3FNV1,
            VhloTypeCode::FloatF8E5M2V1,
            VhloTypeCode::FunctionV1,
            VhloTypeCode::IndexV1,
            VhloTypeCode::IntegerSI4V1,
            VhloTypeCode::IntegerSI8V1,
            VhloTypeCode::IntegerSI16V1,
            VhloTypeCode::IntegerSI32V1,
            VhloTypeCode::IntegerSI64V1,
            VhloTypeCode::IntegerUI4V1,
            VhloTypeCode::IntegerUI8V1,
            VhloTypeCode::IntegerUI16V1,
            VhloTypeCode::IntegerUI32V1,
            VhloTypeCode::IntegerUI64V1,
            VhloTypeCode::RankedTensorV1,
            VhloTypeCode::RankedTensorV1WithEncoding,
            VhloTypeCode::TokenV1,
            VhloTypeCode::TupleV1,
            VhloTypeCode::UniformQuantizedV1,
            VhloTypeCode::UnrankedTensorV1,
            VhloTypeCode::WitnessV1,
            VhloTypeCode::FloatF8E4M3FNUZV1,
            VhloTypeCode::FloatF8E5M2FNUZV1,
            VhloTypeCode::UniformQuantizedPerAxisV1,
            VhloTypeCode::FloatTF32V1,
            VhloTypeCode::IntegerSI2V1,
            VhloTypeCode::IntegerUI2V1,
            VhloTypeCode::FloatF4E2M1FNV1,
            VhloTypeCode::FloatF6E2M3FNV1,
            VhloTypeCode::FloatF6E3M2FNV1,
            VhloTypeCode::FloatF8E3M4V1,
            VhloTypeCode::FloatF8E4M3V1,
            VhloTypeCode::FloatF8E8M0FNUV1,
            VhloTypeCode::NoneV1,
        ];
        usize::try_from(tag)
            .ok()
            .and_then(|idx| ALL.get(idx).copied())
            .ok_or(LibTpuError::MlirBytecodeDecodeError {
                section_id: MlirSectionId::AttrType.as_u8(),
                offset: 0,
                reason: "Unknown VHLO type tag (> 40)",
            })
    }

    /// Maps a scalar [`ElementType`] to its canonical [`VhloTypeCode`].
    ///
    /// # Arguments
    ///
    /// * `et` - Source [`ElementType`].
    #[must_use]
    pub const fn from_scalar_element_type(et: ElementType) -> Self {
        match et {
            ElementType::Pred => Self::BooleanV1,
            ElementType::S2 => Self::IntegerSI2V1,
            ElementType::U2 => Self::IntegerUI2V1,
            ElementType::S4 => Self::IntegerSI4V1,
            ElementType::U4 => Self::IntegerUI4V1,
            ElementType::S8 => Self::IntegerSI8V1,
            ElementType::U8 => Self::IntegerUI8V1,
            ElementType::S16 => Self::IntegerSI16V1,
            ElementType::U16 => Self::IntegerUI16V1,
            ElementType::S32 => Self::IntegerSI32V1,
            ElementType::U32 => Self::IntegerUI32V1,
            ElementType::S64 => Self::IntegerSI64V1,
            ElementType::U64 => Self::IntegerUI64V1,
            ElementType::Bf16 => Self::FloatBF16V1,
            ElementType::F16 => Self::FloatF16V1,
            ElementType::F32 => Self::FloatF32V1,
            ElementType::F64 => Self::FloatF64V1,
            ElementType::F4E2M1Fn => Self::FloatF4E2M1FNV1,
            ElementType::F8E4M3Fn => Self::FloatF8E4M3FNV1,
            ElementType::F8E5M2 => Self::FloatF8E5M2V1,
            ElementType::F8E4M3B11Fnuz => Self::FloatF8E4M3B11FNUZV1,
            ElementType::F8E4M3Fnuz => Self::FloatF8E4M3FNUZV1,
            ElementType::F8E5M2Fnuz => Self::FloatF8E5M2FNUZV1,
            ElementType::F8E8M0Fnu => Self::FloatF8E8M0FNUV1,
            ElementType::Token => Self::TokenV1,
            ElementType::Tuple => Self::TupleV1,
            ElementType::Invalid => Self::NoneV1,
            ElementType::C64 | ElementType::C128 => Self::ComplexV1,
        }
    }

    /// Maps a unit scalar [`VhloTypeCode`] back to [`ElementType`].
    #[must_use]
    pub const fn to_scalar_element_type(self) -> Option<ElementType> {
        match self {
            Self::BooleanV1 => Some(ElementType::Pred),
            Self::FloatBF16V1 => Some(ElementType::Bf16),
            Self::FloatF16V1 => Some(ElementType::F16),
            Self::FloatF32V1 | Self::FloatTF32V1 => Some(ElementType::F32),
            Self::FloatF64V1 => Some(ElementType::F64),
            Self::FloatF4E2M1FNV1 | Self::FloatF6E2M3FNV1 | Self::FloatF6E3M2FNV1 => {
                Some(ElementType::F4E2M1Fn)
            }
            Self::FloatF8E4M3B11FNUZV1 => Some(ElementType::F8E4M3B11Fnuz),
            Self::FloatF8E4M3FNV1 | Self::FloatF8E4M3V1 => Some(ElementType::F8E4M3Fn),
            Self::FloatF8E5M2V1 | Self::FloatF8E3M4V1 => Some(ElementType::F8E5M2),
            Self::FloatF8E4M3FNUZV1 => Some(ElementType::F8E4M3Fnuz),
            Self::FloatF8E5M2FNUZV1 => Some(ElementType::F8E5M2Fnuz),
            Self::FloatF8E8M0FNUV1 => Some(ElementType::F8E8M0Fnu),
            Self::IndexV1 | Self::IntegerSI64V1 => Some(ElementType::S64),
            Self::IntegerSI2V1 => Some(ElementType::S2),
            Self::IntegerUI2V1 => Some(ElementType::U2),
            Self::IntegerSI4V1 => Some(ElementType::S4),
            Self::IntegerUI4V1 => Some(ElementType::U4),
            Self::IntegerSI8V1 => Some(ElementType::S8),
            Self::IntegerUI8V1 => Some(ElementType::U8),
            Self::IntegerSI16V1 => Some(ElementType::S16),
            Self::IntegerUI16V1 => Some(ElementType::U16),
            Self::IntegerSI32V1 => Some(ElementType::S32),
            Self::IntegerUI32V1 => Some(ElementType::U32),
            Self::IntegerUI64V1 => Some(ElementType::U64),
            Self::TokenV1 | Self::WitnessV1 => Some(ElementType::Token),
            _ => None,
        }
    }
}

/// Decoded `vhlo` custom type record from Section `2` (`kAttrType`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VhloCustomType {
    /// Unit scalar, token, witness, or none type (`!vhlo.f32_v1`, `!vhlo.token_v1`, `!vhlo.none_v1`, etc.).
    Scalar(VhloTypeCode),
    /// `!vhlo.complex_v1<element_type_idx>`.
    ComplexV1 {
        /// Inner float type index.
        element_type_idx: u32,
    },
    /// `!vhlo.func_v1<(inputs...) -> (outputs...)>`.
    FunctionV1 {
        /// Input type indices.
        inputs: Vec<u32>,
        /// Output type indices.
        outputs: Vec<u32>,
    },
    /// `!vhlo.tensor_v1<shape, elem, encoding?>`.
    RankedTensorV1 {
        /// Optional `#vhlo.type_extensions_v1` attribute index.
        encoding_attr_idx: Option<u32>,
        /// Signed shape extents (`< 0` denotes dynamic `?`).
        shape: Vec<i64>,
        /// Element type index.
        element_type_idx: u32,
    },
    /// `!vhlo.unranked_tensor_v1<elem>`.
    UnrankedTensorV1 {
        /// Element type index.
        element_type_idx: u32,
    },
    /// `!vhlo.tuple_v1<T0, T1, ...>`.
    TupleV1 {
        /// Element type indices.
        elements: Vec<u32>,
    },
    /// `!vhlo.uniform_quantized_v1` or `!vhlo.uniform_quantized_per_axis_v1`.
    UniformQuantized {
        /// Quantization flags (`1` = signed).
        flags: u32,
        /// Storage integer type index.
        storage_type_idx: u32,
        /// Expressed float type index.
        expressed_type_idx: u32,
        /// Optional quantized axis (`None` for per-tensor).
        quantized_dimension: Option<i32>,
        /// Per-tensor or per-axis scale factors (`f64::to_bits()`).
        scale_bits: Vec<u64>,
        /// Per-tensor or per-axis zero points.
        zero_points: Vec<i64>,
        /// Minimum representable storage integer.
        storage_min: i64,
        /// Maximum representable storage integer.
        storage_max: i64,
    },
}

impl VhloCustomType {
    /// Encodes this [`VhloCustomType`] into `writer` following `VhloBytecode.cpp`.
    ///
    /// # Arguments
    ///
    /// * `writer` - Destination [`DialectBytecodeWriter`].
    pub fn encode(&self, writer: &mut DialectBytecodeWriter) {
        match self {
            Self::Scalar(code) => {
                writer.write_varint(*code as u64);
            }
            Self::ComplexV1 { element_type_idx } => {
                writer.write_varint(VhloTypeCode::ComplexV1 as u64);
                writer.write_varint(u64::from(*element_type_idx));
            }
            Self::FunctionV1 { inputs, outputs } => {
                writer.write_varint(VhloTypeCode::FunctionV1 as u64);
                writer.write_list(inputs, |w, &i| w.write_varint(u64::from(i)));
                writer.write_list(outputs, |w, &o| w.write_varint(u64::from(o)));
            }
            Self::RankedTensorV1 {
                encoding_attr_idx,
                shape,
                element_type_idx,
            } => {
                if let Some(enc) = encoding_attr_idx {
                    writer.write_varint(VhloTypeCode::RankedTensorV1WithEncoding as u64);
                    writer.write_varint(u64::from(*enc));
                } else {
                    writer.write_varint(VhloTypeCode::RankedTensorV1 as u64);
                }
                writer.write_varint(u64::from(*element_type_idx));
                writer.write_list(shape, |w, &d| w.write_signed_varint(d));
            }
            Self::UnrankedTensorV1 { element_type_idx } => {
                writer.write_varint(VhloTypeCode::UnrankedTensorV1 as u64);
                writer.write_varint(u64::from(*element_type_idx));
            }
            Self::TupleV1 { elements } => {
                writer.write_varint(VhloTypeCode::TupleV1 as u64);
                writer.write_list(elements, |w, &e| w.write_varint(u64::from(e)));
            }
            Self::UniformQuantized {
                flags,
                storage_type_idx,
                expressed_type_idx,
                quantized_dimension,
                scale_bits,
                zero_points,
                storage_min,
                storage_max,
            } => {
                if let Some(q_dim) = quantized_dimension {
                    writer.write_varint(VhloTypeCode::UniformQuantizedPerAxisV1 as u64);
                    writer.write_varint(u64::from(*flags));
                    writer.write_varint(u64::from(*storage_type_idx));
                    writer.write_varint(u64::from(*expressed_type_idx));
                    writer.write_signed_varint(i64::from(*q_dim));
                    writer.write_signed_varint(*storage_min);
                    writer.write_signed_varint(*storage_max);
                    writer.write_list(scale_bits, |w, &b| {
                        w.write_apfloat_f64_bits(ElementType::F64, b);
                    });
                    writer.write_list(zero_points, |w, &zp| w.write_signed_varint(zp));
                } else {
                    writer.write_varint(VhloTypeCode::UniformQuantizedV1 as u64);
                    writer.write_varint(u64::from(*flags));
                    writer.write_varint(u64::from(*storage_type_idx));
                    writer.write_varint(u64::from(*expressed_type_idx));
                    writer.write_apfloat_f64_bits(
                        ElementType::F64,
                        scale_bits.first().copied().unwrap_or(1.0_f64.to_bits()),
                    );
                    writer.write_signed_varint(zero_points.first().copied().unwrap_or(0));
                    writer.write_signed_varint(*storage_min);
                    writer.write_signed_varint(*storage_max);
                }
            }
        }
    }

    /// Decodes a [`VhloCustomType`] from `reader` following `VhloBytecode.cpp`.
    ///
    /// # Arguments
    ///
    /// * `reader` - Source [`DialectBytecodeReader`].
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MlirBytecodeDecodeError`] if the tag is unknown or payload is truncated.
    #[allow(clippy::cast_possible_truncation)]
    pub fn decode(reader: &mut DialectBytecodeReader<'_>) -> LibTpuResult<Self> {
        let code = VhloTypeCode::from_tag(reader.read_varint()?)?;
        match code {
            VhloTypeCode::ComplexV1 => Ok(Self::ComplexV1 {
                element_type_idx: reader.read_varint()? as u32,
            }),
            VhloTypeCode::FunctionV1 => {
                let inputs = reader.read_list(|r| Ok(r.read_varint()? as u32))?;
                let outputs = reader.read_list(|r| Ok(r.read_varint()? as u32))?;
                Ok(Self::FunctionV1 { inputs, outputs })
            }
            VhloTypeCode::RankedTensorV1 | VhloTypeCode::RankedTensorV1WithEncoding => {
                let encoding_attr_idx = if code == VhloTypeCode::RankedTensorV1WithEncoding {
                    Some(reader.read_varint()? as u32)
                } else {
                    None
                };
                let element_type_idx = reader.read_varint()? as u32;
                let shape = reader.read_list(DialectBytecodeReader::read_signed_varint)?;
                Ok(Self::RankedTensorV1 {
                    encoding_attr_idx,
                    shape,
                    element_type_idx,
                })
            }
            VhloTypeCode::UnrankedTensorV1 => Ok(Self::UnrankedTensorV1 {
                element_type_idx: reader.read_varint()? as u32,
            }),
            VhloTypeCode::TupleV1 => {
                let elements = reader.read_list(|r| Ok(r.read_varint()? as u32))?;
                Ok(Self::TupleV1 { elements })
            }
            VhloTypeCode::UniformQuantizedV1 => {
                let flags = reader.read_varint()? as u32;
                let storage_type_idx = reader.read_varint()? as u32;
                let expressed_type_idx = reader.read_varint()? as u32;
                let scale = reader.read_apfloat_f64_bits(ElementType::F64)?;
                let zp = reader.read_signed_varint()?;
                let storage_min = reader.read_signed_varint()?;
                let storage_max = reader.read_signed_varint()?;
                Ok(Self::UniformQuantized {
                    flags,
                    storage_type_idx,
                    expressed_type_idx,
                    quantized_dimension: None,
                    scale_bits: vec![scale],
                    zero_points: vec![zp],
                    storage_min,
                    storage_max,
                })
            }
            VhloTypeCode::UniformQuantizedPerAxisV1 => {
                let flags = reader.read_varint()? as u32;
                let storage_type_idx = reader.read_varint()? as u32;
                let expressed_type_idx = reader.read_varint()? as u32;
                let q_dim = reader.read_signed_varint()? as i32;
                let storage_min = reader.read_signed_varint()?;
                let storage_max = reader.read_signed_varint()?;
                let scale_bits = reader.read_list(|r| r.read_apfloat_f64_bits(ElementType::F64))?;
                let zero_points = reader.read_list(DialectBytecodeReader::read_signed_varint)?;
                Ok(Self::UniformQuantized {
                    flags,
                    storage_type_idx,
                    expressed_type_idx,
                    quantized_dimension: Some(q_dim),
                    scale_bits,
                    zero_points,
                    storage_min,
                    storage_max,
                })
            }
            other => Ok(Self::Scalar(other)),
        }
    }

    /// Converts this [`VhloCustomType`] into an [`MlirType`] using resolved type and attribute tables.
    ///
    /// # Arguments
    ///
    /// * `resolved_types` - Previously-resolved [`MlirType`] table.
    /// * `resolved_attrs` - Previously-resolved [`MlirAttribute`] table.
    #[must_use]
    pub fn to_mlir_type(
        &self,
        resolved_types: &[MlirType],
        resolved_attrs: &[MlirAttribute],
    ) -> MlirType {
        let lookup_elem = |idx: u32| -> ElementType {
            resolved_types
                .get(idx as usize)
                .map_or(ElementType::F32, MlirType::element_type)
        };
        match self {
            Self::Scalar(VhloTypeCode::TokenV1 | VhloTypeCode::WitnessV1) => MlirType::Token,
            Self::Scalar(VhloTypeCode::NoneV1) => MlirType::Tuple(Vec::new()),
            Self::Scalar(code) => {
                MlirType::Scalar(code.to_scalar_element_type().unwrap_or(ElementType::F32))
            }
            Self::ComplexV1 { element_type_idx } => {
                let et = if lookup_elem(*element_type_idx) == ElementType::F64 {
                    ElementType::C128
                } else {
                    ElementType::C64
                };
                MlirType::Scalar(et)
            }
            Self::FunctionV1 { inputs, outputs } => MlirType::Function(AstFunctionType {
                inputs: inputs
                    .iter()
                    .filter_map(|&i| resolved_types.get(i as usize).cloned())
                    .collect(),
                results: outputs
                    .iter()
                    .filter_map(|&o| resolved_types.get(o as usize).cloned())
                    .collect(),
            }),
            Self::RankedTensorV1 {
                encoding_attr_idx,
                shape,
                element_type_idx,
            } => {
                let mut dims: Vec<DimensionSize> = shape
                    .iter()
                    .map(|&d| {
                        if d < 0 {
                            DimensionSize::Dynamic
                        } else {
                            DimensionSize::Static(d)
                        }
                    })
                    .collect();
                if let Some(enc_idx) = encoding_attr_idx
                    && let Some(MlirAttribute::DenseI64Array(bounds)) =
                        resolved_attrs.get(*enc_idx as usize)
                {
                    for (slot, &b) in dims.iter_mut().zip(bounds.iter()) {
                        if *slot == DimensionSize::Dynamic && b > 0 {
                            *slot = DimensionSize::Bounded(b);
                        }
                    }
                }
                MlirType::RankedTensor(RankedTensorType {
                    dims,
                    element_type: lookup_elem(*element_type_idx),
                    encoding: None,
                    layout: None,
                })
            }
            Self::UnrankedTensorV1 { element_type_idx } => {
                MlirType::UnrankedTensor(lookup_elem(*element_type_idx))
            }
            Self::TupleV1 { elements } => MlirType::Tuple(
                elements
                    .iter()
                    .filter_map(|&e| resolved_types.get(e as usize).cloned())
                    .collect(),
            ),
            Self::UniformQuantized {
                storage_type_idx,
                expressed_type_idx,
                quantized_dimension,
                scale_bits,
                zero_points,
                ..
            } => MlirType::Quantized(QuantizedElementType {
                storage_type: lookup_elem(*storage_type_idx),
                expressed_type: lookup_elem(*expressed_type_idx),
                scales: scale_bits.iter().map(|&b| f64::from_bits(b)).collect(),
                zero_points: zero_points.clone(),
                quantized_dimension: quantized_dimension.and_then(|d| usize::try_from(d).ok()),
            }),
        }
    }
}

/// Complete `vhlo_encoding::AttributeCode` (`1..=20`) matching `openxla/stablehlo/dialect/VhloBytecode.cpp`.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VhloAttrCode {
    /// `#vhlo.array_v1` (`1`).
    ArrayV1 = 1,
    /// `#vhlo.bool_v1` (`2`).
    BooleanV1 = 2,
    /// `#vhlo.comparison_direction_v1` (`3`).
    ComparisonDirectionV1 = 3,
    /// `#vhlo.comparison_type_v1` (`4`).
    ComparisonTypeV1 = 4,
    /// `#vhlo.api_version_v1` (`5`).
    CustomCallApiVersionV1 = 5,
    /// `#vhlo.dict_v1` (`6`).
    DictionaryV1 = 6,
    /// `#vhlo.fft_type_v1` (`7`).
    FftTypeV1 = 7,
    /// `#vhlo.float_v1` (`8`).
    FloatV1 = 8,
    /// `#vhlo.integer_v1` (`9`).
    IntegerV1 = 9,
    /// `#vhlo.output_operand_alias_v1` (`10`).
    OutputOperandAliasV1 = 10,
    /// `#vhlo.precision_v1` (`11`).
    PrecisionV1 = 11,
    /// `#vhlo.rng_algorithm_v1` (`12`).
    RngAlgorithmV1 = 12,
    /// `#vhlo.rng_distribution_v1` (`13`).
    RngDistributionV1 = 13,
    /// `#vhlo.string_v1` (`14`).
    StringV1 = 14,
    /// `#vhlo.tensor_v1` (`15`).
    TensorV1 = 15,
    /// `#vhlo.transpose_v1` (`16`).
    TransposeV1 = 16,
    /// `#vhlo.type_v1` (`17`).
    TypeV1 = 17,
    /// `#vhlo.type_extensions_v1<bounds = [...]>` (`18`).
    TypeExtensionsV1 = 18,
    /// `#vhlo.result_accuracy_mode_v1` (`19`).
    ResultAccuracyModeV1 = 19,
    /// `#vhlo.result_accuracy_v1` (`20`).
    ResultAccuracyV1 = 20,
}

impl VhloAttrCode {
    /// Converts a raw varint tag (`1..=20`) into a [`VhloAttrCode`].
    ///
    /// # Arguments
    ///
    /// * `tag` - Raw `PrefixVarInt` tag from Section `2` (`kAttrType`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MlirBytecodeDecodeError`] if `tag == 0` or `tag > 20`.
    pub fn from_tag(tag: u64) -> LibTpuResult<Self> {
        const ALL: [VhloAttrCode; 20] = [
            VhloAttrCode::ArrayV1,
            VhloAttrCode::BooleanV1,
            VhloAttrCode::ComparisonDirectionV1,
            VhloAttrCode::ComparisonTypeV1,
            VhloAttrCode::CustomCallApiVersionV1,
            VhloAttrCode::DictionaryV1,
            VhloAttrCode::FftTypeV1,
            VhloAttrCode::FloatV1,
            VhloAttrCode::IntegerV1,
            VhloAttrCode::OutputOperandAliasV1,
            VhloAttrCode::PrecisionV1,
            VhloAttrCode::RngAlgorithmV1,
            VhloAttrCode::RngDistributionV1,
            VhloAttrCode::StringV1,
            VhloAttrCode::TensorV1,
            VhloAttrCode::TransposeV1,
            VhloAttrCode::TypeV1,
            VhloAttrCode::TypeExtensionsV1,
            VhloAttrCode::ResultAccuracyModeV1,
            VhloAttrCode::ResultAccuracyV1,
        ];
        tag.checked_sub(1)
            .and_then(|z| usize::try_from(z).ok())
            .and_then(|idx| ALL.get(idx).copied())
            .ok_or(LibTpuError::MlirBytecodeDecodeError {
                section_id: MlirSectionId::AttrType.as_u8(),
                offset: 0,
                reason: "Unknown VHLO attribute tag (expected 1..=20)",
            })
    }
}

/// Decoded `vhlo` custom attribute record from Section `2` (`kAttrType`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VhloCustomAttr {
    /// `#vhlo.array_v1<[attr0, attr1, ...]>`.
    ArrayV1(Vec<u32>),
    /// `#vhlo.bool_v1<true|false>`.
    BooleanV1(bool),
    /// `#vhlo.comparison_direction_v1<EQ|NE|GE|GT|LE|LT>`.
    ComparisonDirectionV1(ComparisonDirection),
    /// `#vhlo.comparison_type_v1<NOTYPE|FLOAT|TOTALORDER|SIGNED|UNSIGNED>`.
    ComparisonTypeV1(ComparisonType),
    /// `#vhlo.api_version_v1<version>`.
    CustomCallApiVersionV1(u32),
    /// `#vhlo.dict_v1<{ key_attr_idx = val_attr_idx }>`.
    DictionaryV1(Vec<(u32, u32)>),
    /// `#vhlo.fft_type_v1<FFT|IFFT|RFFT|IRFFT>`.
    FftTypeV1(FftKind),
    /// `#vhlo.float_v1<type_idx, element_type, f64_bits>`.
    FloatV1 {
        /// Scalar float type index.
        type_idx: u32,
        /// Floating-point [`ElementType`].
        element_type: ElementType,
        /// IEEE-754 `f64::to_bits()`.
        f64_bits: u64,
    },
    /// `#vhlo.integer_v1<type_idx, bit_width, value>`.
    IntegerV1 {
        /// Scalar integer type index.
        type_idx: u32,
        /// Integer bit width.
        bit_width: u32,
        /// Signed 128-bit integer value.
        value: i128,
    },
    /// `#vhlo.output_operand_alias_v1`.
    OutputOperandAliasV1 {
        /// Output tuple path indices.
        output_tuple_indices: Vec<i64>,
        /// Aliased operand index.
        operand_index: i64,
        /// Operand tuple path indices.
        operand_tuple_indices: Vec<i64>,
    },
    /// `#vhlo.precision_v1<DEFAULT|HIGH|HIGHEST>`.
    PrecisionV1(PrecisionLevel),
    /// `#vhlo.rng_algorithm_v1<DEFAULT|THREE_FRY|PHILOX>`.
    RngAlgorithmV1(RngAlgorithmKind),
    /// `#vhlo.rng_distribution_v1<1=UNIFORM, 2=NORMAL>`.
    RngDistributionV1(u8),
    /// `#vhlo.string_v1<"bytes">`.
    StringV1(Vec<u8>),
    /// `#vhlo.tensor_v1<type_idx, raw_bytes>`.
    TensorV1 {
        /// Ranked tensor type index.
        type_idx: u32,
        /// Raw little-endian tensor constant bytes.
        raw_bytes: Vec<u8>,
    },
    /// `#vhlo.transpose_v1<NO_TRANSPOSE|TRANSPOSE|ADJOINT>`.
    TransposeV1(TransposeMode),
    /// `#vhlo.type_v1<type_idx>`.
    TypeV1(u32),
    /// `#vhlo.type_extensions_v1<bounds = [...]>`.
    TypeExtensionsV1(Vec<i64>),
    /// `#vhlo.result_accuracy_mode_v1<mode>`.
    ResultAccuracyModeV1(u8),
    /// `#vhlo.result_accuracy_v1<atol_bits, rtol_bits, ulps, mode>`.
    ResultAccuracyV1 {
        /// Absolute tolerance (`f64::to_bits()`).
        atol_bits: u64,
        /// Relative tolerance (`f64::to_bits()`).
        rtol_bits: u64,
        /// Maximum ULPs error.
        ulps: i64,
        /// Accuracy mode code (`0=DEFAULT`, `1=HIGHEST`, `2=TOLERANCE`).
        mode: u8,
    },
}

impl VhloCustomAttr {
    /// Serializes this [`VhloCustomAttr`] into `writer` following `VhloBytecode.cpp`.
    ///
    /// # Arguments
    ///
    /// * `writer` - Destination [`DialectBytecodeWriter`].
    #[allow(clippy::too_many_lines)]
    pub fn encode(&self, writer: &mut DialectBytecodeWriter) {
        match self {
            Self::ArrayV1(items) => {
                writer.write_varint(VhloAttrCode::ArrayV1 as u64);
                writer.write_list(items, |w, &i| w.write_varint(u64::from(i)));
            }
            Self::BooleanV1(b) => {
                writer.write_varint(VhloAttrCode::BooleanV1 as u64);
                writer.write_bool(*b);
            }
            Self::ComparisonDirectionV1(dir) => {
                writer.write_varint(VhloAttrCode::ComparisonDirectionV1 as u64);
                let v = match dir {
                    ComparisonDirection::Eq => 0,
                    ComparisonDirection::Ne => 1,
                    ComparisonDirection::Ge => 2,
                    ComparisonDirection::Gt => 3,
                    ComparisonDirection::Le => 4,
                    ComparisonDirection::Lt => 5,
                };
                writer.write_varint(v);
            }
            Self::ComparisonTypeV1(ct) => {
                writer.write_varint(VhloAttrCode::ComparisonTypeV1 as u64);
                let v = match ct {
                    ComparisonType::Float => 1,
                    ComparisonType::TotalOrder => 2,
                    ComparisonType::Signed => 3,
                    ComparisonType::Unsigned => 4,
                };
                writer.write_varint(v);
            }
            Self::CustomCallApiVersionV1(ver) => {
                writer.write_varint(VhloAttrCode::CustomCallApiVersionV1 as u64);
                writer.write_varint(u64::from(*ver));
            }
            Self::DictionaryV1(entries) => {
                writer.write_varint(VhloAttrCode::DictionaryV1 as u64);
                writer.write_list(entries, |w, &(k, v)| {
                    w.write_varint(u64::from(k));
                    w.write_varint(u64::from(v));
                });
            }
            Self::FftTypeV1(fft) => {
                writer.write_varint(VhloAttrCode::FftTypeV1 as u64);
                let v = match fft {
                    FftKind::Fft => 0,
                    FftKind::Ifft => 1,
                    FftKind::Rfft => 2,
                    FftKind::Irfft => 3,
                };
                writer.write_varint(v);
            }
            Self::FloatV1 {
                type_idx,
                element_type,
                f64_bits,
            } => {
                writer.write_varint(VhloAttrCode::FloatV1 as u64);
                writer.write_varint(u64::from(*type_idx));
                writer.write_byte(VhloTypeCode::from_scalar_element_type(*element_type) as u8);
                writer.write_apfloat_f64_bits(*element_type, *f64_bits);
            }
            Self::IntegerV1 {
                type_idx,
                bit_width,
                value,
            } => {
                writer.write_varint(VhloAttrCode::IntegerV1 as u64);
                writer.write_varint(u64::from(*type_idx));
                writer.write_varint(u64::from(*bit_width));
                writer.write_apint(*bit_width, *value);
            }
            Self::OutputOperandAliasV1 {
                output_tuple_indices,
                operand_index,
                operand_tuple_indices,
            } => {
                writer.write_varint(VhloAttrCode::OutputOperandAliasV1 as u64);
                writer.write_list(output_tuple_indices, |w, &d| w.write_signed_varint(d));
                writer.write_signed_varint(*operand_index);
                writer.write_list(operand_tuple_indices, |w, &d| w.write_signed_varint(d));
            }
            Self::PrecisionV1(lvl) => {
                writer.write_varint(VhloAttrCode::PrecisionV1 as u64);
                let v = match lvl {
                    PrecisionLevel::Default => 0,
                    PrecisionLevel::High => 1,
                    PrecisionLevel::Highest => 2,
                };
                writer.write_varint(v);
            }
            Self::RngAlgorithmV1(alg) => {
                writer.write_varint(VhloAttrCode::RngAlgorithmV1 as u64);
                let v = match alg {
                    RngAlgorithmKind::Default => 0,
                    RngAlgorithmKind::ThreeFry => 1,
                    RngAlgorithmKind::Philox => 2,
                };
                writer.write_varint(v);
            }
            Self::RngDistributionV1(dist) => {
                writer.write_varint(VhloAttrCode::RngDistributionV1 as u64);
                writer.write_varint(u64::from(*dist));
            }
            Self::StringV1(bytes) => {
                writer.write_varint(VhloAttrCode::StringV1 as u64);
                writer.write_blob(bytes);
            }
            Self::TensorV1 {
                type_idx,
                raw_bytes,
            } => {
                writer.write_varint(VhloAttrCode::TensorV1 as u64);
                writer.write_varint(u64::from(*type_idx));
                writer.write_blob(raw_bytes);
            }
            Self::TransposeV1(mode) => {
                writer.write_varint(VhloAttrCode::TransposeV1 as u64);
                let v = match mode {
                    TransposeMode::NoTranspose => 1,
                    TransposeMode::Transpose => 2,
                    TransposeMode::Adjoint => 3,
                };
                writer.write_varint(v);
            }
            Self::TypeV1(ty_idx) => {
                writer.write_varint(VhloAttrCode::TypeV1 as u64);
                writer.write_varint(u64::from(*ty_idx));
            }
            Self::TypeExtensionsV1(bounds) => {
                writer.write_varint(VhloAttrCode::TypeExtensionsV1 as u64);
                writer.write_list(bounds, |w, &b| w.write_signed_varint(b));
            }
            Self::ResultAccuracyModeV1(mode) => {
                writer.write_varint(VhloAttrCode::ResultAccuracyModeV1 as u64);
                writer.write_varint(u64::from(*mode));
            }
            Self::ResultAccuracyV1 {
                atol_bits,
                rtol_bits,
                ulps,
                mode,
            } => {
                writer.write_varint(VhloAttrCode::ResultAccuracyV1 as u64);
                writer.write_apfloat_f64_bits(ElementType::F64, *atol_bits);
                writer.write_apfloat_f64_bits(ElementType::F64, *rtol_bits);
                writer.write_signed_varint(*ulps);
                writer.write_varint(u64::from(*mode));
            }
        }
    }

    /// Decodes a [`VhloCustomAttr`] from `reader` following `VhloBytecode.cpp`.
    ///
    /// # Arguments
    ///
    /// * `reader` - Source [`DialectBytecodeReader`].
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MlirBytecodeDecodeError`] if the tag is invalid or payload is truncated.
    #[allow(clippy::cast_possible_truncation, clippy::too_many_lines)]
    pub fn decode(reader: &mut DialectBytecodeReader<'_>) -> LibTpuResult<Self> {
        let code = VhloAttrCode::from_tag(reader.read_varint()?)?;
        match code {
            VhloAttrCode::ArrayV1 => {
                let items = reader.read_list(|r| Ok(r.read_varint()? as u32))?;
                Ok(Self::ArrayV1(items))
            }
            VhloAttrCode::BooleanV1 => Ok(Self::BooleanV1(reader.read_bool()?)),
            VhloAttrCode::ComparisonDirectionV1 => {
                let dir = match reader.read_varint()? {
                    1 => ComparisonDirection::Ne,
                    2 => ComparisonDirection::Ge,
                    3 => ComparisonDirection::Gt,
                    4 => ComparisonDirection::Le,
                    5 => ComparisonDirection::Lt,
                    _ => ComparisonDirection::Eq,
                };
                Ok(Self::ComparisonDirectionV1(dir))
            }
            VhloAttrCode::ComparisonTypeV1 => {
                let ct = match reader.read_varint()? {
                    2 => ComparisonType::TotalOrder,
                    3 => ComparisonType::Signed,
                    4 => ComparisonType::Unsigned,
                    _ => ComparisonType::Float,
                };
                Ok(Self::ComparisonTypeV1(ct))
            }
            VhloAttrCode::CustomCallApiVersionV1 => {
                Ok(Self::CustomCallApiVersionV1(reader.read_varint()? as u32))
            }
            VhloAttrCode::DictionaryV1 => {
                let entries = reader.read_list(|r| {
                    let k = r.read_varint()? as u32;
                    let v = r.read_varint()? as u32;
                    Ok((k, v))
                })?;
                Ok(Self::DictionaryV1(entries))
            }
            VhloAttrCode::FftTypeV1 => {
                let fft = match reader.read_varint()? {
                    1 => FftKind::Ifft,
                    2 => FftKind::Rfft,
                    3 => FftKind::Irfft,
                    _ => FftKind::Fft,
                };
                Ok(Self::FftTypeV1(fft))
            }
            VhloAttrCode::FloatV1 => {
                let type_idx = reader.read_varint()? as u32;
                let tc = VhloTypeCode::from_tag(u64::from(reader.read_byte()?))?;
                let element_type = tc.to_scalar_element_type().unwrap_or(ElementType::F32);
                let f64_bits = reader.read_apfloat_f64_bits(element_type)?;
                Ok(Self::FloatV1 {
                    type_idx,
                    element_type,
                    f64_bits,
                })
            }
            VhloAttrCode::IntegerV1 => {
                let type_idx = reader.read_varint()? as u32;
                let bit_width = reader.read_varint()? as u32;
                let value = reader.read_apint(bit_width)?;
                Ok(Self::IntegerV1 {
                    type_idx,
                    bit_width,
                    value,
                })
            }
            VhloAttrCode::OutputOperandAliasV1 => {
                let output_tuple_indices =
                    reader.read_list(DialectBytecodeReader::read_signed_varint)?;
                let operand_index = reader.read_signed_varint()?;
                let operand_tuple_indices =
                    reader.read_list(DialectBytecodeReader::read_signed_varint)?;
                Ok(Self::OutputOperandAliasV1 {
                    output_tuple_indices,
                    operand_index,
                    operand_tuple_indices,
                })
            }
            VhloAttrCode::PrecisionV1 => {
                let lvl = match reader.read_varint()? {
                    1 => PrecisionLevel::High,
                    2 => PrecisionLevel::Highest,
                    _ => PrecisionLevel::Default,
                };
                Ok(Self::PrecisionV1(lvl))
            }
            VhloAttrCode::RngAlgorithmV1 => {
                let alg = match reader.read_varint()? {
                    1 => RngAlgorithmKind::ThreeFry,
                    2 => RngAlgorithmKind::Philox,
                    _ => RngAlgorithmKind::Default,
                };
                Ok(Self::RngAlgorithmV1(alg))
            }
            VhloAttrCode::RngDistributionV1 => {
                Ok(Self::RngDistributionV1(reader.read_varint()? as u8))
            }
            VhloAttrCode::StringV1 => Ok(Self::StringV1(reader.read_blob()?.to_vec())),
            VhloAttrCode::TensorV1 => {
                let type_idx = reader.read_varint()? as u32;
                let raw_bytes = reader.read_blob()?.to_vec();
                Ok(Self::TensorV1 {
                    type_idx,
                    raw_bytes,
                })
            }
            VhloAttrCode::TransposeV1 => {
                let mode = match reader.read_varint()? {
                    2 => TransposeMode::Transpose,
                    3 => TransposeMode::Adjoint,
                    _ => TransposeMode::NoTranspose,
                };
                Ok(Self::TransposeV1(mode))
            }
            VhloAttrCode::TypeV1 => Ok(Self::TypeV1(reader.read_varint()? as u32)),
            VhloAttrCode::TypeExtensionsV1 => {
                let bounds = reader.read_list(DialectBytecodeReader::read_signed_varint)?;
                Ok(Self::TypeExtensionsV1(bounds))
            }
            VhloAttrCode::ResultAccuracyModeV1 => {
                Ok(Self::ResultAccuracyModeV1(reader.read_varint()? as u8))
            }
            VhloAttrCode::ResultAccuracyV1 => {
                let atol_bits = reader.read_apfloat_f64_bits(ElementType::F64)?;
                let rtol_bits = reader.read_apfloat_f64_bits(ElementType::F64)?;
                let ulps = reader.read_signed_varint()?;
                let mode = reader.read_varint()? as u8;
                Ok(Self::ResultAccuracyV1 {
                    atol_bits,
                    rtol_bits,
                    ulps,
                    mode,
                })
            }
        }
    }

    /// Converts this [`VhloCustomAttr`] into a canonical [`MlirAttribute`] using resolved type and attribute tables.
    ///
    /// # Arguments
    ///
    /// * `resolved_types` - Previously-resolved [`MlirType`] table.
    /// * `resolved_attrs` - Previously-resolved [`MlirAttribute`] table.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn to_mlir_attribute(
        &self,
        resolved_types: &[MlirType],
        resolved_attrs: &[MlirAttribute],
    ) -> MlirAttribute {
        match self {
            Self::ArrayV1(indices) => {
                let items: Vec<MlirAttribute> = indices
                    .iter()
                    .filter_map(|&i| resolved_attrs.get(i as usize).cloned())
                    .collect();
                let precs: Vec<PrecisionLevel> = items
                    .iter()
                    .filter_map(|a| match a {
                        MlirAttribute::PrecisionConfig(v) => Some(v.as_slice()),
                        _ => None,
                    })
                    .flatten()
                    .copied()
                    .collect();
                if !items.is_empty() && precs.len() >= items.len() {
                    return MlirAttribute::PrecisionConfig(precs);
                }
                MlirAttribute::Array(items)
            }
            Self::BooleanV1(b) => MlirAttribute::Bool(*b),
            Self::ComparisonDirectionV1(dir) => MlirAttribute::ComparisonDirection(*dir),
            Self::ComparisonTypeV1(ct) => MlirAttribute::ComparisonType(*ct),
            Self::CustomCallApiVersionV1(ver) => MlirAttribute::Integer {
                value: i128::from(*ver),
                ty: Some(ElementType::S32),
            },
            Self::DictionaryV1(entries) => {
                let mut map = BTreeMap::new();
                for &(k_idx, v_idx) in entries {
                    if let (Some(k_str), Some(val)) = (
                        resolved_attrs
                            .get(k_idx as usize)
                            .and_then(MlirAttribute::as_str),
                        resolved_attrs.get(v_idx as usize),
                    ) {
                        map.insert(String::from(k_str), val.clone());
                    }
                }
                MlirAttribute::Dictionary(map)
            }
            Self::FftTypeV1(fft) => MlirAttribute::FftType(*fft),
            Self::FloatV1 {
                element_type,
                f64_bits,
                ..
            } => MlirAttribute::Float {
                value: f64::from_bits(*f64_bits),
                ty: Some(*element_type),
            },
            Self::IntegerV1 {
                type_idx,
                bit_width,
                value,
            } => {
                let et = resolved_types
                    .get(*type_idx as usize)
                    .map(MlirType::element_type);
                if *bit_width == 1 {
                    MlirAttribute::Bool(*value != 0)
                } else {
                    MlirAttribute::Integer {
                        value: *value,
                        ty: et,
                    }
                }
            }
            Self::OutputOperandAliasV1 {
                output_tuple_indices,
                operand_index,
                operand_tuple_indices,
            } => MlirAttribute::String(format!(
                "#vhlo.output_operand_alias_v1<output_tuple_indices = {output_tuple_indices:?}, operand_index = {operand_index}, operand_tuple_indices = {operand_tuple_indices:?}>"
            )),
            Self::PrecisionV1(lvl) => MlirAttribute::PrecisionConfig(vec![*lvl]),
            Self::RngAlgorithmV1(alg) => MlirAttribute::RngAlgorithm(*alg),
            Self::RngDistributionV1(dist) => MlirAttribute::Integer {
                value: i128::from(*dist),
                ty: Some(ElementType::S32),
            },
            Self::StringV1(bytes) => {
                MlirAttribute::String(String::from_utf8_lossy(bytes).into_owned())
            }
            Self::TensorV1 {
                type_idx,
                raw_bytes,
            } => {
                let ranked_ty = match resolved_types.get(*type_idx as usize) {
                    Some(MlirType::RankedTensor(rt)) => Some(rt.clone()),
                    _ => None,
                };
                let et = ranked_ty
                    .as_ref()
                    .map_or(ElementType::F32, |rt| rt.element_type);
                let unpacked = et.unpack_to_f64_vec(raw_bytes);
                MlirAttribute::DenseElements(DenseElementsAttr {
                    values_f64: if unpacked.is_empty() {
                        vec![0.0]
                    } else {
                        unpacked
                    },
                    raw_bytes: Some(raw_bytes.clone()),
                    is_splat: false,
                    tensor_type: ranked_ty,
                })
            }
            Self::TransposeV1(mode) => MlirAttribute::TransposeKind(*mode),
            Self::TypeV1(ty_idx) => resolved_types.get(*ty_idx as usize).map_or(
                MlirAttribute::Type(MlirType::Scalar(ElementType::F32)),
                |t| MlirAttribute::Type(t.clone()),
            ),
            Self::TypeExtensionsV1(bounds) => MlirAttribute::DenseI64Array(bounds.clone()),
            Self::ResultAccuracyModeV1(mode) => MlirAttribute::Integer {
                value: i128::from(*mode),
                ty: Some(ElementType::U8),
            },
            Self::ResultAccuracyV1 {
                atol_bits,
                rtol_bits,
                ulps,
                mode,
            } => {
                let mut dict = BTreeMap::new();
                dict.insert(
                    String::from("atol"),
                    MlirAttribute::Float {
                        value: f64::from_bits(*atol_bits),
                        ty: Some(ElementType::F64),
                    },
                );
                dict.insert(
                    String::from("rtol"),
                    MlirAttribute::Float {
                        value: f64::from_bits(*rtol_bits),
                        ty: Some(ElementType::F64),
                    },
                );
                dict.insert(
                    String::from("ulps"),
                    MlirAttribute::Integer {
                        value: i128::from(*ulps),
                        ty: Some(ElementType::S64),
                    },
                );
                dict.insert(
                    String::from("mode"),
                    MlirAttribute::Integer {
                        value: i128::from(*mode),
                        ty: Some(ElementType::U8),
                    },
                );
                MlirAttribute::Dictionary(dict)
            }
        }
    }
}

/// Specification entry for one of the 126 upstream `VHLO_*OpV*` operations in `VhloOps.td`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VhloOpSpec {
    /// Fully-qualified `vhlo` operation name (e.g., `"vhlo.dot_general_v2"`).
    pub vhlo_name: &'static str,
    /// Target dialect after upgrading (`DialectKind::Func` for `func_v1`/`call_v1`, else `DialectKind::StableHlo`).
    pub target_dialect: DialectKind,
    /// Canonical `stablehlo` or `func` unqualified mnemonic (e.g., `"dot_general"`).
    pub stablehlo_mnemonic: &'static str,
    /// Operation version suffix (`1` for `_v1`, `2` for `_v2`).
    pub op_version: u8,
    /// Minimum `VhloVersion` supporting this exact versioned operation.
    pub min_version: VhloVersion,
}

/// Constructs a `v1` [`VhloOpSpec`] valid from [`VhloVersion::MIN_SUPPORTED`] (`0.9.0`).
macro_rules! op_v1 {
    ($vhlo_name:expr, $mnemonic:expr) => {
        VhloOpSpec {
            vhlo_name: $vhlo_name,
            target_dialect: DialectKind::StableHlo,
            stablehlo_mnemonic: $mnemonic,
            op_version: 1,
            min_version: VhloVersion::MIN_SUPPORTED,
        }
    };
}

/// Constructs a `v2` [`VhloOpSpec`] valid starting at `$min_version`.
macro_rules! op_v2 {
    ($vhlo_name:expr, $mnemonic:expr, $min_version:expr) => {
        VhloOpSpec {
            vhlo_name: $vhlo_name,
            target_dialect: DialectKind::StableHlo,
            stablehlo_mnemonic: $mnemonic,
            op_version: 2,
            min_version: $min_version,
        }
    };
}

/// Complete 140-entry versioned table covering all `VHLO_*OpV*` operations (`v1` + `v2` evolutions) from `openxla/stablehlo/dialect/VhloOps.td`.
pub const VHLO_OP_SPECS: &[VhloOpSpec] = &[
    op_v1!("vhlo.abs_v1", "abs"),
    op_v1!("vhlo.add_v1", "add"),
    op_v1!("vhlo.after_all_v1", "after_all"),
    op_v1!("vhlo.all_gather_v1", "all_gather"),
    op_v2!("vhlo.all_gather_v2", "all_gather", VhloVersion::V1_1_0),
    op_v1!("vhlo.all_reduce_v1", "all_reduce"),
    op_v2!("vhlo.all_reduce_v2", "all_reduce", VhloVersion::V1_1_0),
    op_v1!("vhlo.all_to_all_v1", "all_to_all"),
    op_v2!("vhlo.all_to_all_v2", "all_to_all", VhloVersion::V1_1_0),
    op_v1!("vhlo.and_v1", "and"),
    op_v1!("vhlo.atan2_v1", "atan2"),
    op_v1!("vhlo.batch_norm_grad_v1", "batch_norm_grad"),
    op_v1!("vhlo.batch_norm_inference_v1", "batch_norm_inference"),
    op_v1!("vhlo.batch_norm_training_v1", "batch_norm_training"),
    op_v1!("vhlo.bitcast_convert_v1", "bitcast_convert"),
    op_v1!("vhlo.broadcast_in_dim_v1", "broadcast_in_dim"),
    op_v1!("vhlo.broadcast_v1", "broadcast"),
    VhloOpSpec {
        vhlo_name: "vhlo.call_v1",
        target_dialect: DialectKind::Func,
        stablehlo_mnemonic: "call",
        op_version: 1,
        min_version: VhloVersion::MIN_SUPPORTED,
    },
    op_v1!("vhlo.case_v1", "case"),
    op_v1!("vhlo.cbrt_v1", "cbrt"),
    op_v2!("vhlo.cbrt_v2", "cbrt", VhloVersion::V1_9_0),
    op_v1!("vhlo.ceil_v1", "ceil"),
    op_v1!("vhlo.cholesky_v1", "cholesky"),
    op_v1!("vhlo.clamp_v1", "clamp"),
    op_v1!("vhlo.collective_broadcast_v1", "collective_broadcast"),
    op_v1!("vhlo.collective_permute_v1", "collective_permute"),
    op_v2!(
        "vhlo.collective_permute_v2",
        "collective_permute",
        VhloVersion::V1_3_0
    ),
    op_v1!("vhlo.compare_v1", "compare"),
    op_v1!("vhlo.complex_v1", "complex"),
    op_v1!("vhlo.composite_v1", "composite"),
    op_v1!("vhlo.compute_reshape_shape_v1", "compute_reshape_shape"),
    op_v1!("vhlo.concatenate_v1", "concatenate"),
    op_v1!("vhlo.constant_v1", "constant"),
    op_v1!("vhlo.convert_v1", "convert"),
    op_v1!("vhlo.convolution_v1", "convolution"),
    op_v1!("vhlo.cosine_v1", "cosine"),
    op_v2!("vhlo.cosine_v2", "cosine", VhloVersion::V1_9_0),
    op_v1!("vhlo.count_leading_zeros_v1", "count_leading_zeros"),
    op_v1!("vhlo.create_token_v1", "create_token"),
    op_v1!("vhlo.cross-replica-sum_v1", "cross-replica-sum"),
    op_v1!("vhlo.cstr_reshapable_v1", "cstr_reshapable"),
    op_v1!("vhlo.custom_call_v1", "custom_call"),
    op_v2!("vhlo.custom_call_v2", "custom_call", VhloVersion::V1_3_0),
    op_v1!("vhlo.divide_v1", "divide"),
    op_v1!("vhlo.dot_general_v1", "dot_general"),
    op_v2!("vhlo.dot_general_v2", "dot_general", VhloVersion::V1_6_0),
    op_v1!("vhlo.dot_v1", "dot"),
    op_v1!(
        "vhlo.dynamic_broadcast_in_dim_v1",
        "dynamic_broadcast_in_dim"
    ),
    op_v1!("vhlo.dynamic_conv_v1", "dynamic_conv"),
    op_v1!("vhlo.dynamic_gather_v1", "dynamic_gather"),
    op_v2!(
        "vhlo.dynamic_gather_v2",
        "dynamic_gather",
        VhloVersion::V1_1_0
    ),
    op_v1!("vhlo.dynamic_iota_v1", "dynamic_iota"),
    op_v1!("vhlo.dynamic_pad_v1", "dynamic_pad"),
    op_v1!("vhlo.dynamic_reshape_v1", "dynamic_reshape"),
    op_v1!("vhlo.dynamic_slice_v1", "dynamic_slice"),
    op_v1!("vhlo.dynamic_update_slice_v1", "dynamic_update_slice"),
    op_v1!("vhlo.einsum_v1", "einsum"),
    op_v1!("vhlo.exponential_minus_one_v1", "exponential_minus_one"),
    op_v2!(
        "vhlo.exponential_minus_one_v2",
        "exponential_minus_one",
        VhloVersion::V1_9_0
    ),
    op_v1!("vhlo.exponential_v1", "exponential"),
    op_v2!("vhlo.exponential_v2", "exponential", VhloVersion::V1_9_0),
    op_v1!("vhlo.fft_v1", "fft"),
    op_v1!("vhlo.floor_v1", "floor"),
    VhloOpSpec {
        vhlo_name: "vhlo.func_v1",
        target_dialect: DialectKind::Func,
        stablehlo_mnemonic: "func",
        op_version: 1,
        min_version: VhloVersion::MIN_SUPPORTED,
    },
    op_v1!("vhlo.gather_v1", "gather"),
    op_v2!("vhlo.gather_v2", "gather", VhloVersion::V1_6_0),
    op_v1!("vhlo.get_dimension_size_v1", "get_dimension_size"),
    op_v1!("vhlo.get_tuple_element_v1", "get_tuple_element"),
    op_v1!("vhlo.if_v1", "if"),
    op_v1!("vhlo.imag_v1", "imag"),
    op_v1!("vhlo.infeed_v1", "infeed"),
    op_v1!("vhlo.iota_v1", "iota"),
    op_v1!("vhlo.is_finite_v1", "is_finite"),
    op_v1!("vhlo.log_plus_one_v1", "log_plus_one"),
    op_v2!("vhlo.log_plus_one_v2", "log_plus_one", VhloVersion::V1_9_0),
    op_v1!("vhlo.log_v1", "log"),
    op_v2!("vhlo.log_v2", "log", VhloVersion::V1_9_0),
    op_v1!("vhlo.logistic_v1", "logistic"),
    op_v2!("vhlo.logistic_v2", "logistic", VhloVersion::V1_9_0),
    op_v1!("vhlo.map_v1", "map"),
    op_v1!("vhlo.maximum_v1", "maximum"),
    op_v1!("vhlo.minimum_v1", "minimum"),
    op_v1!("vhlo.multiply_v1", "multiply"),
    op_v1!("vhlo.negate_v1", "negate"),
    op_v1!("vhlo.not_v1", "not"),
    op_v1!("vhlo.optimization_barrier_v1", "optimization_barrier"),
    op_v1!("vhlo.or_v1", "or"),
    op_v1!("vhlo.outfeed_v1", "outfeed"),
    op_v1!("vhlo.pad_v1", "pad"),
    op_v1!("vhlo.partition_id_v1", "partition_id"),
    op_v1!("vhlo.population_count_v1", "population_count"),
    op_v1!("vhlo.power_v1", "power"),
    op_v1!("vhlo.real_dynamic_slice_v1", "real_dynamic_slice"),
    op_v1!("vhlo.real_v1", "real"),
    op_v1!("vhlo.recv_v1", "recv"),
    op_v1!("vhlo.reduce_precision_v1", "reduce_precision"),
    op_v1!("vhlo.reduce_scatter_v1", "reduce_scatter"),
    op_v2!(
        "vhlo.reduce_scatter_v2",
        "reduce_scatter",
        VhloVersion::V1_3_0
    ),
    op_v1!("vhlo.reduce_v1", "reduce"),
    op_v1!("vhlo.reduce_window_v1", "reduce_window"),
    op_v1!("vhlo.remainder_v1", "remainder"),
    op_v1!("vhlo.replica_id_v1", "replica_id"),
    op_v1!("vhlo.reshape_v1", "reshape"),
    VhloOpSpec {
        vhlo_name: "vhlo.return_v1",
        target_dialect: DialectKind::Func,
        stablehlo_mnemonic: "return",
        op_version: 1,
        min_version: VhloVersion::MIN_SUPPORTED,
    },
    op_v1!("vhlo.reverse_v1", "reverse"),
    op_v1!("vhlo.rng_bit_generator_v1", "rng_bit_generator"),
    op_v1!("vhlo.rng_v1", "rng"),
    op_v1!("vhlo.round_nearest_afz_v1", "round_nearest_afz"),
    op_v1!("vhlo.round_nearest_even_v1", "round_nearest_even"),
    op_v1!("vhlo.rsqrt_v1", "rsqrt"),
    op_v2!("vhlo.rsqrt_v2", "rsqrt", VhloVersion::V1_9_0),
    op_v1!("vhlo.scatter_v1", "scatter"),
    op_v1!("vhlo.select_and_scatter_v1", "select_and_scatter"),
    op_v1!("vhlo.select_v1", "select"),
    op_v1!("vhlo.send_v1", "send"),
    op_v1!("vhlo.set_dimension_size_v1", "set_dimension_size"),
    op_v1!("vhlo.shift_left_v1", "shift_left"),
    op_v1!("vhlo.shift_right_arithmetic_v1", "shift_right_arithmetic"),
    op_v1!("vhlo.shift_right_logical_v1", "shift_right_logical"),
    op_v1!("vhlo.sign_v1", "sign"),
    op_v1!("vhlo.sine_v1", "sine"),
    op_v2!("vhlo.sine_v2", "sine", VhloVersion::V1_9_0),
    op_v1!("vhlo.slice_v1", "slice"),
    op_v1!("vhlo.sort_v1", "sort"),
    op_v1!("vhlo.sqrt_v1", "sqrt"),
    op_v2!("vhlo.sqrt_v2", "sqrt", VhloVersion::V1_9_0),
    op_v1!("vhlo.subtract_v1", "subtract"),
    op_v1!("vhlo.tan_v1", "tan"),
    op_v2!("vhlo.tan_v2", "tan", VhloVersion::V1_9_0),
    op_v1!("vhlo.tanh_v1", "tanh"),
    op_v2!("vhlo.tanh_v2", "tanh", VhloVersion::V1_9_0),
    op_v1!("vhlo.torch_index_select_v1", "torch_index_select"),
    op_v1!("vhlo.transpose_v1", "transpose"),
    op_v1!("vhlo.triangular_solve_v1", "triangular_solve"),
    op_v1!("vhlo.tuple_v1", "tuple"),
    op_v1!("vhlo.unary_einsum_v1", "unary_einsum"),
    op_v1!("vhlo.uniform_dequantize_v1", "uniform_dequantize"),
    op_v1!("vhlo.uniform_quantize_v1", "uniform_quantize"),
    op_v1!("vhlo.while_v1", "while"),
    op_v1!("vhlo.xor_v1", "xor"),
];

/// Looks up a [`VhloOpSpec`] by its exact `"vhlo.<mnemonic>_v<N>"` name across all versioned entries.
///
/// # Arguments
///
/// * `vhlo_name` - Fully-qualified VHLO operation name (e.g., `"vhlo.dot_general_v2"`).
#[must_use]
pub fn lookup_vhlo_op_by_name(vhlo_name: &str) -> Option<&'static VhloOpSpec> {
    VHLO_OP_SPECS
        .iter()
        .find(|spec| spec.vhlo_name == vhlo_name)
}

/// Selects the highest-versioned [`VhloOpSpec`] compatible with `target_version` for a `stablehlo` / `func` mnemonic.
///
/// # Arguments
///
/// * `mnemonic` - Canonical `stablehlo` or `func` mnemonic (e.g., `"dot_general"`, `"add"`, `"return"`).
/// * `target_version` - Requested portable artifact [`VhloVersion`].
///
/// # Errors
///
/// Returns [`LibTpuError::CompilationFailed`] if `target_version < VhloVersion::MIN_SUPPORTED` or
/// `mnemonic` has no legal `vhlo` counterpart at `target_version`.
pub fn select_vhlo_op_for_stablehlo(
    mnemonic: &str,
    target_version: VhloVersion,
) -> LibTpuResult<&'static VhloOpSpec> {
    if !target_version.is_supported() {
        return Err(LibTpuError::CompilationFailed {
            reason: format!(
                "Unsupported target VHLO version {target_version} (minimum is {})",
                VhloVersion::MIN_SUPPORTED
            ),
        });
    }
    let clean = mnemonic
        .strip_suffix("_v1")
        .or_else(|| mnemonic.strip_suffix("_v2"))
        .unwrap_or(mnemonic);
    let best = VHLO_OP_SPECS
        .iter()
        .filter(|spec| spec.stablehlo_mnemonic == clean && spec.min_version <= target_version)
        .max_by_key(|spec| spec.op_version);
    best.ok_or_else(|| LibTpuError::CompilationFailed {
        reason: format!("Operation '{mnemonic}' cannot be downgraded to VHLO v{target_version}"),
    })
}

/// Recursively upgrades `vhlo` operations inside an [`AstRegion`] to canonical `stablehlo` / `func` operations.
///
/// # Arguments
///
/// * `region` - Mutable [`AstRegion`] to upgrade.
fn upgrade_vhlo_region(region: &mut AstRegion) {
    for block in &mut region.blocks {
        for op in &mut block.operations {
            upgrade_vhlo_op(op);
        }
    }
}

/// Upgrades a single [`AstOperation`] from `DialectKind::Vhlo` (or `_v1`/`_v2` mnemonics) to `stablehlo` / `func`.
///
/// # Arguments
///
/// * `op` - Mutable [`AstOperation`] to upgrade.
fn upgrade_vhlo_op(op: &mut AstOperation) {
    for region in &mut op.regions {
        upgrade_vhlo_region(region);
    }
    if op.dialect == DialectKind::Vhlo {
        let candidate = format!("vhlo.{}", op.mnemonic);
        if let Some(spec) = lookup_vhlo_op_by_name(&candidate) {
            op.dialect = spec.target_dialect;
            op.mnemonic = String::from(spec.stablehlo_mnemonic);
        } else {
            let stripped = op
                .mnemonic
                .strip_suffix("_v1")
                .or_else(|| op.mnemonic.strip_suffix("_v2"))
                .unwrap_or(&op.mnemonic);
            op.dialect = if matches!(stripped, "return" | "call" | "func") {
                DialectKind::Func
            } else {
                DialectKind::StableHlo
            };
            op.mnemonic = String::from(stripped);
        }
        let elem_ty = op
            .signature
            .results
            .iter()
            .chain(&op.signature.inputs)
            .next()
            .map_or(ElementType::F32, MlirType::element_type);
        op.op_kind = classify_ast_operation(
            op.dialect,
            &op.mnemonic,
            elem_ty,
            &op.attributes,
            &op.regions,
        );
    }
}

/// Upgrades all `vhlo` (`v0.9.0`–`v1.11.0+`) operations, regions, and CHLO ops in `module` into canonical `stablehlo` + `func`.
///
/// # Arguments
///
/// * `module` - Mutable [`AstModule`] to upgrade in place.
///
/// # Errors
///
/// Returns [`LibTpuError::CompilationFailed`] if module-level `vhlo.version` metadata specifies an unsupported version `< 0.9.0`.
pub fn upgrade_vhlo_to_stablehlo(module: &mut AstModule) -> LibTpuResult<()> {
    if let Some(ver_str) = module
        .attributes
        .get("vhlo.version")
        .and_then(MlirAttribute::as_str)
    {
        let ver = VhloVersion::parse_str(ver_str)?;
        if !ver.is_supported() {
            return Err(LibTpuError::CompilationFailed {
                reason: format!("VHLO module version {ver} is older than minimum 0.9.0"),
            });
        }
    }
    module.attributes.remove("vhlo.version");
    for func in &mut module.functions {
        upgrade_vhlo_region(&mut func.body);
    }
    for op in &mut module.top_level_operations {
        upgrade_vhlo_op(op);
    }
    legalize_chlo_in_ast_module(module);
    Ok(())
}

/// Recursively downgrades `stablehlo` / `func` operations inside an [`AstRegion`] to version-checked `vhlo` operations.
///
/// # Arguments
///
/// * `region` - Mutable [`AstRegion`] to downgrade.
/// * `target_version` - Target [`VhloVersion`].
fn downgrade_region_to_vhlo(
    region: &mut AstRegion,
    target_version: VhloVersion,
) -> LibTpuResult<()> {
    for block in &mut region.blocks {
        for op in &mut block.operations {
            downgrade_op_to_vhlo(op, target_version)?;
        }
    }
    Ok(())
}

/// Downgrades a single [`AstOperation`] to `DialectKind::Vhlo` at `target_version`.
///
/// # Arguments
///
/// * `op` - Mutable [`AstOperation`] to downgrade.
/// * `target_version` - Target [`VhloVersion`].
fn downgrade_op_to_vhlo(op: &mut AstOperation, target_version: VhloVersion) -> LibTpuResult<()> {
    for region in &mut op.regions {
        downgrade_region_to_vhlo(region, target_version)?;
    }
    if matches!(
        op.dialect,
        DialectKind::StableHlo | DialectKind::Mhlo | DialectKind::Func | DialectKind::Vhlo
    ) {
        let spec = select_vhlo_op_for_stablehlo(&op.mnemonic, target_version)?;
        op.dialect = DialectKind::Vhlo;
        op.mnemonic = String::from(
            spec.vhlo_name
                .strip_prefix("vhlo.")
                .unwrap_or(spec.vhlo_name),
        );
    }
    Ok(())
}

/// Downgrades a canonical `stablehlo` [`AstModule`] into a version-pinned `vhlo` [`AstModule`] (`v0.9.0`–`v1.11.0+`).
///
/// # Arguments
///
/// * `module` - Source [`AstModule`] containing `stablehlo` / `chlo` / `func` operations.
/// * `target_version` - Target portable [`VhloVersion`].
///
/// # Errors
///
/// Returns [`LibTpuError::CompilationFailed`] if `target_version < 0.9.0` or any operation cannot be represented in `target_version`.
pub fn downgrade_stablehlo_to_vhlo(
    module: &AstModule,
    target_version: VhloVersion,
) -> LibTpuResult<AstModule> {
    if !target_version.is_supported() {
        return Err(LibTpuError::CompilationFailed {
            reason: format!("Target VHLO version {target_version} is below minimum 0.9.0"),
        });
    }
    let mut cloned = module.clone();
    legalize_chlo_in_ast_module(&mut cloned);
    for func in &mut cloned.functions {
        downgrade_region_to_vhlo(&mut func.body, target_version)?;
    }
    for op in &mut cloned.top_level_operations {
        downgrade_op_to_vhlo(op, target_version)?;
    }
    cloned.attributes.insert(
        String::from("vhlo.version"),
        MlirAttribute::String(target_version.to_string()),
    );
    Ok(cloned)
}

/// Serializes a `stablehlo` [`AstModule`] into an upstream `ML\xefR` v6 portable `vhlo` bytecode artifact
/// embedding `target_version` inside Section `7` (`kDialectVersions`).
///
/// # Arguments
///
/// * `module` - Source [`AstModule`] to downgrade and encode.
/// * `target_version` - Target portable [`VhloVersion`] (`v0.9.0`–`v1.11.0+`).
///
/// # Errors
///
/// Returns [`LibTpuError`] if `module` cannot be downgraded to `target_version`.
pub fn serialize_portable_vhlo_artifact(
    module: &AstModule,
    target_version: VhloVersion,
) -> LibTpuResult<Vec<u8>> {
    let vhlo_mod = downgrade_stablehlo_to_vhlo(module, target_version)?;
    Ok(MlirBytecodeHeader::encode_portable_vhlo_ast(
        &vhlo_mod,
        target_version,
    ))
}

/// Deserializes an `ML\xefR` portable `vhlo` bytecode artifact, extracting its `VhloVersion` from Section `7`
/// (`kDialectVersions`) and upgrading the module to canonical `stablehlo` + `func`.
///
/// # Arguments
///
/// * `bytes` - Raw `ML\xefR` portable artifact bytes.
///
/// # Errors
///
/// Returns [`LibTpuError`] if `bytes` is not a valid MLIR bytecode container or has an unsupported `VhloVersion`.
pub fn deserialize_portable_vhlo_artifact(bytes: &[u8]) -> LibTpuResult<(VhloVersion, AstModule)> {
    let hdr = MlirBytecodeHeader::decode(bytes)?;
    let version = if let Some(sec7) = hdr
        .sections
        .iter()
        .rfind(|s| s.section_id == MlirSectionId::DialectVersions.as_u8())
    {
        VhloVersion::decode_dialect_version(&sec7.data)?
    } else {
        VhloVersion::CURRENT
    };
    if !version.is_supported() {
        return Err(LibTpuError::CompilationFailed {
            reason: format!("Unsupported portable artifact VHLO version {version}"),
        });
    }
    let mut ast_mod = hdr.to_ast_module();
    upgrade_vhlo_to_stablehlo(&mut ast_mod)?;
    Ok((version, ast_mod))
}

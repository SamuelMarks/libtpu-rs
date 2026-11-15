//! Upstream MLIR `builtin` dialect custom bytecode encodings (`BuiltinDialectBytecode.td`).
//!
//! Implements exact tag discriminants and `DialectBytecodeReader` / `DialectBytecodeWriter` wire
//! codecs for all `builtin` types (`0..=20`) and `builtin` attributes (`0..=23`).

use super::stream::{DialectBytecodeReader, DialectBytecodeWriter, MlirSectionId};
use crate::compiler::ast::{
    AffineMapExpr, AstFunctionType, DenseElementsAttr, DimensionSize, MemRefLayoutAttr, MemRefType,
    MlirAttribute, MlirLocation, MlirType, RankedTensorType, TensorEncodingAttr, TpuMemorySpace,
    VectorType,
};
use crate::error::{LibTpuError, LibTpuResult};
use crate::types::ElementType;
use std::collections::BTreeMap;

/// Upstream MLIR `builtin` dialect custom type tags from `BuiltinDialectBytecode.td` (`0..=20`).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinTypeCode {
    /// `IntegerType` (`0`): `(width << 2) | signedness`.
    IntegerType = 0,
    /// `IndexType` (`1`).
    IndexType = 1,
    /// `FunctionType` (`2`): `inputs` list + `results` list.
    FunctionType = 2,
    /// `BFloat16Type` (`3`).
    BFloat16Type = 3,
    /// `Float16Type` (`4`).
    Float16Type = 4,
    /// `Float32Type` (`5`).
    Float32Type = 5,
    /// `Float64Type` (`6`).
    Float64Type = 6,
    /// `Float80Type` (`7`).
    Float80Type = 7,
    /// `Float128Type` (`8`).
    Float128Type = 8,
    /// `ComplexType` (`9`): `elementTypeIdx`.
    ComplexType = 9,
    /// `MemRefType` (`10`): `shape`, `elementTypeIdx`, `layoutAttrIdx`.
    MemRefType = 10,
    /// `MemRefTypeWithMemSpace` (`11`): `memorySpaceAttrIdx`, `shape`, `elementTypeIdx`, `layoutAttrIdx`.
    MemRefTypeWithMemSpace = 11,
    /// `NoneType` (`12`).
    NoneType = 12,
    /// `RankedTensorType` (`13`): `shape` (signed varints, `i64::MIN` or `-1` for dynamic), `elementTypeIdx`.
    RankedTensorType = 13,
    /// `RankedTensorTypeWithEncoding` (`14`): `encodingAttrIdx`, `shape`, `elementTypeIdx`.
    RankedTensorTypeWithEncoding = 14,
    /// `TupleType` (`15`): `elementTypeIndices` list.
    TupleType = 15,
    /// `UnrankedMemRefType` (`16`): `elementTypeIdx`.
    UnrankedMemRefType = 16,
    /// `UnrankedMemRefTypeWithMemSpace` (`17`): `memorySpaceAttrIdx`, `elementTypeIdx`.
    UnrankedMemRefTypeWithMemSpace = 17,
    /// `UnrankedTensorType` (`18`): `elementTypeIdx`.
    UnrankedTensorType = 18,
    /// `VectorType` (`19`): `shape`, `elementTypeIdx`.
    VectorType = 19,
    /// `VectorTypeWithScalableDims` (`20`): `shape`, `scalableDims`, `elementTypeIdx`.
    VectorTypeWithScalableDims = 20,
}

impl BuiltinTypeCode {
    /// Parses a raw varint tag into a [`BuiltinTypeCode`].
    ///
    /// # Arguments
    ///
    /// * `tag` - Raw `PrefixVarInt` tag from Section `2` (`kAttrType`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MlirBytecodeDecodeError`] if `tag > 20`.
    pub const fn from_tag(tag: u64) -> LibTpuResult<Self> {
        match tag {
            0 => Ok(Self::IntegerType),
            1 => Ok(Self::IndexType),
            2 => Ok(Self::FunctionType),
            3 => Ok(Self::BFloat16Type),
            4 => Ok(Self::Float16Type),
            5 => Ok(Self::Float32Type),
            6 => Ok(Self::Float64Type),
            7 => Ok(Self::Float80Type),
            8 => Ok(Self::Float128Type),
            9 => Ok(Self::ComplexType),
            10 => Ok(Self::MemRefType),
            11 => Ok(Self::MemRefTypeWithMemSpace),
            12 => Ok(Self::NoneType),
            13 => Ok(Self::RankedTensorType),
            14 => Ok(Self::RankedTensorTypeWithEncoding),
            15 => Ok(Self::TupleType),
            16 => Ok(Self::UnrankedMemRefType),
            17 => Ok(Self::UnrankedMemRefTypeWithMemSpace),
            18 => Ok(Self::UnrankedTensorType),
            19 => Ok(Self::VectorType),
            20 => Ok(Self::VectorTypeWithScalableDims),
            _ => Err(LibTpuError::MlirBytecodeDecodeError {
                section_id: MlirSectionId::AttrType.as_u8(),
                offset: 0,
                reason: "Unknown builtin dialect type tag (> 20)",
            }),
        }
    }
}

/// Upstream MLIR `builtin` dialect custom attribute tags from `BuiltinDialectBytecode.td` (`0..=23`).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinAttrCode {
    /// `ArrayAttr` (`0`): list of `attrIdx`.
    ArrayAttr = 0,
    /// `DictionaryAttr` (`1`): list of `(nameAttrIdx, valueAttrIdx)`.
    DictionaryAttr = 1,
    /// `StringAttr` (`2`): `stringIdx`.
    StringAttr = 2,
    /// `StringAttrWithType` (`3`): `stringIdx`, `typeIdx`.
    StringAttrWithType = 3,
    /// `FlatSymbolRefAttr` (`4`): `stringIdx`.
    FlatSymbolRefAttr = 4,
    /// `SymbolRefAttr` (`5`): `rootNameAttrIdx`, list of nested `FlatSymbolRefAttr` indices.
    SymbolRefAttr = 5,
    /// `TypeAttr` (`6`): `typeIdx`.
    TypeAttr = 6,
    /// `UnitAttr` (`7`).
    UnitAttr = 7,
    /// `IntegerAttr` (`8`): `typeIdx`, `APInt`.
    IntegerAttr = 8,
    /// `FloatAttr` (`9`): `typeIdx`, `APFloat`.
    FloatAttr = 9,
    /// `CallSiteLoc` (`10`): `calleeLocIdx`, `callerLocIdx`.
    CallSiteLoc = 10,
    /// `FileLineColLoc` (`11`): `fileStringAttrIdx`, `line`, `column`.
    FileLineColLoc = 11,
    /// `FusedLoc` (`12`): list of `locIdx`.
    FusedLoc = 12,
    /// `FusedLocWithMetadata` (`13`): list of `locIdx`, `metadataAttrIdx`.
    FusedLocWithMetadata = 13,
    /// `NameLoc` (`14`): `nameStringAttrIdx`, `childLocIdx`.
    NameLoc = 14,
    /// `UnknownLoc` (`15`).
    UnknownLoc = 15,
    /// `DenseResourceElementsAttr` (`16`): `shapedTypeIdx`, `resourceHandleIdx`.
    DenseResourceElementsAttr = 16,
    /// `DenseArrayAttr` (`17`): `elementBitWidthOrCode`, `size`, `rawBlob`.
    DenseArrayAttr = 17,
    /// `DenseIntOrFPElementsAttr` (`18`): `shapedTypeIdx`, `rawBlob`.
    DenseIntOrFPElementsAttr = 18,
    /// `DenseStringElementsAttr` (`19`): `shapedTypeIdx`, `isSplat`, `stringIndices`.
    DenseStringElementsAttr = 19,
    /// `SparseElementsAttr` (`20`): `shapedTypeIdx`, `indicesAttrIdx`, `valuesAttrIdx`.
    SparseElementsAttr = 20,
    /// `DistinctAttr` (`21`): `referencedAttrIdx`.
    DistinctAttr = 21,
    /// `FileLineColRange` (`22`): `fileStringIdx`, `startLine`, `startCol`, `endLine`, `endCol`.
    FileLineColRange = 22,
    /// `AffineMapAttr` (`23`): `numDims`, `numSymbols`, `results`.
    AffineMapAttr = 23,
}

impl BuiltinAttrCode {
    /// Parses a raw varint tag into a [`BuiltinAttrCode`].
    ///
    /// # Arguments
    ///
    /// * `tag` - Raw `PrefixVarInt` tag from Section `2` (`kAttrType`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MlirBytecodeDecodeError`] if `tag > 23`.
    pub const fn from_tag(tag: u64) -> LibTpuResult<Self> {
        match tag {
            0 => Ok(Self::ArrayAttr),
            1 => Ok(Self::DictionaryAttr),
            2 => Ok(Self::StringAttr),
            3 => Ok(Self::StringAttrWithType),
            4 => Ok(Self::FlatSymbolRefAttr),
            5 => Ok(Self::SymbolRefAttr),
            6 => Ok(Self::TypeAttr),
            7 => Ok(Self::UnitAttr),
            8 => Ok(Self::IntegerAttr),
            9 => Ok(Self::FloatAttr),
            10 => Ok(Self::CallSiteLoc),
            11 => Ok(Self::FileLineColLoc),
            12 => Ok(Self::FusedLoc),
            13 => Ok(Self::FusedLocWithMetadata),
            14 => Ok(Self::NameLoc),
            15 => Ok(Self::UnknownLoc),
            16 => Ok(Self::DenseResourceElementsAttr),
            17 => Ok(Self::DenseArrayAttr),
            18 => Ok(Self::DenseIntOrFPElementsAttr),
            19 => Ok(Self::DenseStringElementsAttr),
            20 => Ok(Self::SparseElementsAttr),
            21 => Ok(Self::DistinctAttr),
            22 => Ok(Self::FileLineColRange),
            23 => Ok(Self::AffineMapAttr),
            _ => Err(LibTpuError::MlirBytecodeDecodeError {
                section_id: MlirSectionId::AttrType.as_u8(),
                offset: 0,
                reason: "Unknown builtin dialect attribute tag (> 23)",
            }),
        }
    }
}

/// Decoded upstream `builtin` dialect custom type record from Section `2` (`kAttrType`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuiltinCustomType {
    /// `IntegerType` (`width` in bits; `signedness`: `0 = signless`, `1 = signed`, `2 = unsigned`).
    Integer {
        /// Bit width (`1`, `2`, `4`, `8`, `16`, `32`, `64`, ...).
        width: u32,
        /// Signedness flag (`0 = signless`, `1 = signed`, `2 = unsigned`).
        signedness: u8,
    },
    /// `IndexType` (`index`).
    Index,
    /// `FunctionType` referencing previously-decoded type indices.
    Function {
        /// Input type indices in the Section `2` type table.
        inputs: Vec<u32>,
        /// Result type indices in the Section `2` type table.
        results: Vec<u32>,
    },
    /// Standard IEEE / brain floating-point scalar type (`bf16`, `f16`, `f32`, `f64`, `f80`, `f128`).
    Float(BuiltinTypeCode),
    /// `ComplexType` (`complex<elem>`).
    Complex {
        /// Element type index in the Section `2` type table.
        element_type_idx: u32,
    },
    /// `MemRefType` (with optional `memory_space_attr_idx`).
    MemRef {
        /// Optional memory space attribute index (`None` for default HBM).
        memory_space_attr_idx: Option<u32>,
        /// Signed dimension extents (`< 0` denotes dynamic `?`).
        shape: Vec<i64>,
        /// Element type index in the Section `2` type table.
        element_type_idx: u32,
        /// Optional layout attribute index (`u32::MAX` when absent).
        layout_attr_idx: Option<u32>,
    },
    /// `NoneType` (`none`).
    NoneType,
    /// `RankedTensorType` (with optional `encoding_attr_idx`).
    RankedTensor {
        /// Optional encoding attribute index (`#sdy.sharding`, `#stablehlo.bounds`, etc.).
        encoding_attr_idx: Option<u32>,
        /// Signed dimension extents (`< 0` denotes dynamic `?`).
        shape: Vec<i64>,
        /// Element type index in the Section `2` type table.
        element_type_idx: u32,
    },
    /// `TupleType` (`tuple<T0, T1, ...>`).
    Tuple {
        /// Element type indices in the Section `2` type table.
        elements: Vec<u32>,
    },
    /// `UnrankedMemRefType` (`memref<*xElem>`).
    UnrankedMemRef {
        /// Optional memory space attribute index.
        memory_space_attr_idx: Option<u32>,
        /// Element type index in the Section `2` type table.
        element_type_idx: u32,
    },
    /// `UnrankedTensorType` (`tensor<*xElem>`).
    UnrankedTensor {
        /// Element type index in the Section `2` type table.
        element_type_idx: u32,
    },
    /// `VectorType` (`vector<...xElem>` with optional scalable dimension flags).
    Vector {
        /// Static vector dimension extents.
        shape: Vec<i64>,
        /// Per-dimension scalable flags (`[4]` vs `4`).
        scalable_dims: Vec<bool>,
        /// Element type index in the Section `2` type table.
        element_type_idx: u32,
    },
}

impl BuiltinCustomType {
    /// Serializes this [`BuiltinCustomType`] into `writer` following `BuiltinDialectBytecode.td`.
    ///
    /// # Arguments
    ///
    /// * `writer` - Destination [`DialectBytecodeWriter`].
    pub fn encode(&self, writer: &mut DialectBytecodeWriter) {
        match self {
            Self::Integer { width, signedness } => {
                writer.write_varint(BuiltinTypeCode::IntegerType as u64);
                let packed = (u64::from(*width) << 2) | u64::from(*signedness & 0x03);
                writer.write_varint(packed);
            }
            Self::Index => {
                writer.write_varint(BuiltinTypeCode::IndexType as u64);
            }
            Self::Function { inputs, results } => {
                writer.write_varint(BuiltinTypeCode::FunctionType as u64);
                writer.write_list(inputs, |w, &idx| w.write_varint(u64::from(idx)));
                writer.write_list(results, |w, &idx| w.write_varint(u64::from(idx)));
            }
            Self::Float(code) => {
                writer.write_varint(*code as u64);
            }
            Self::Complex { element_type_idx } => {
                writer.write_varint(BuiltinTypeCode::ComplexType as u64);
                writer.write_varint(u64::from(*element_type_idx));
            }
            Self::MemRef {
                memory_space_attr_idx,
                shape,
                element_type_idx,
                layout_attr_idx,
            } => {
                if let Some(ms_idx) = memory_space_attr_idx {
                    writer.write_varint(BuiltinTypeCode::MemRefTypeWithMemSpace as u64);
                    writer.write_varint(u64::from(*ms_idx));
                } else {
                    writer.write_varint(BuiltinTypeCode::MemRefType as u64);
                }
                writer.write_list(shape, |w, &d| w.write_signed_varint(d));
                writer.write_varint(u64::from(*element_type_idx));
                writer.write_varint(layout_attr_idx.map_or(0, |idx| u64::from(idx) + 1));
            }
            Self::NoneType => {
                writer.write_varint(BuiltinTypeCode::NoneType as u64);
            }
            Self::RankedTensor {
                encoding_attr_idx,
                shape,
                element_type_idx,
            } => {
                if let Some(enc_idx) = encoding_attr_idx {
                    writer.write_varint(BuiltinTypeCode::RankedTensorTypeWithEncoding as u64);
                    writer.write_varint(u64::from(*enc_idx));
                } else {
                    writer.write_varint(BuiltinTypeCode::RankedTensorType as u64);
                }
                writer.write_list(shape, |w, &d| w.write_signed_varint(d));
                writer.write_varint(u64::from(*element_type_idx));
            }
            Self::Tuple { elements } => {
                writer.write_varint(BuiltinTypeCode::TupleType as u64);
                writer.write_list(elements, |w, &idx| w.write_varint(u64::from(idx)));
            }
            Self::UnrankedMemRef {
                memory_space_attr_idx,
                element_type_idx,
            } => {
                if let Some(ms_idx) = memory_space_attr_idx {
                    writer.write_varint(BuiltinTypeCode::UnrankedMemRefTypeWithMemSpace as u64);
                    writer.write_varint(u64::from(*ms_idx));
                } else {
                    writer.write_varint(BuiltinTypeCode::UnrankedMemRefType as u64);
                }
                writer.write_varint(u64::from(*element_type_idx));
            }
            Self::UnrankedTensor { element_type_idx } => {
                writer.write_varint(BuiltinTypeCode::UnrankedTensorType as u64);
                writer.write_varint(u64::from(*element_type_idx));
            }
            Self::Vector {
                shape,
                scalable_dims,
                element_type_idx,
            } => {
                if scalable_dims.iter().any(|&b| b) {
                    writer.write_varint(BuiltinTypeCode::VectorTypeWithScalableDims as u64);
                    writer.write_list(shape, |w, &d| w.write_signed_varint(d));
                    writer.write_list(scalable_dims, |w, &s| w.write_bool(s));
                } else {
                    writer.write_varint(BuiltinTypeCode::VectorType as u64);
                    writer.write_list(shape, |w, &d| w.write_signed_varint(d));
                }
                writer.write_varint(u64::from(*element_type_idx));
            }
        }
    }

    /// Decodes a [`BuiltinCustomType`] from `reader` following `BuiltinDialectBytecode.td`.
    ///
    /// # Arguments
    ///
    /// * `reader` - Source [`DialectBytecodeReader`].
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MlirBytecodeDecodeError`] if the tag is invalid or payload is truncated.
    #[allow(clippy::cast_possible_truncation)]
    pub fn decode(reader: &mut DialectBytecodeReader<'_>) -> LibTpuResult<Self> {
        let code = BuiltinTypeCode::from_tag(reader.read_varint()?)?;
        match code {
            BuiltinTypeCode::IntegerType => {
                let ws = reader.read_varint()?;
                Ok(Self::Integer {
                    width: (ws >> 2) as u32,
                    signedness: (ws & 0x03) as u8,
                })
            }
            BuiltinTypeCode::IndexType => Ok(Self::Index),
            BuiltinTypeCode::FunctionType => {
                let inputs = reader.read_list(|r| Ok(r.read_varint()? as u32))?;
                let results = reader.read_list(|r| Ok(r.read_varint()? as u32))?;
                Ok(Self::Function { inputs, results })
            }
            BuiltinTypeCode::BFloat16Type
            | BuiltinTypeCode::Float16Type
            | BuiltinTypeCode::Float32Type
            | BuiltinTypeCode::Float64Type
            | BuiltinTypeCode::Float80Type
            | BuiltinTypeCode::Float128Type => Ok(Self::Float(code)),
            BuiltinTypeCode::ComplexType => Ok(Self::Complex {
                element_type_idx: reader.read_varint()? as u32,
            }),
            BuiltinTypeCode::MemRefType | BuiltinTypeCode::MemRefTypeWithMemSpace => {
                let memory_space_attr_idx = if code == BuiltinTypeCode::MemRefTypeWithMemSpace {
                    Some(reader.read_varint()? as u32)
                } else {
                    None
                };
                let shape = reader.read_list(DialectBytecodeReader::read_signed_varint)?;
                let element_type_idx = reader.read_varint()? as u32;
                let layout_raw = reader.read_varint()?;
                let layout_attr_idx = layout_raw.checked_sub(1).map(|v| v as u32);
                Ok(Self::MemRef {
                    memory_space_attr_idx,
                    shape,
                    element_type_idx,
                    layout_attr_idx,
                })
            }
            BuiltinTypeCode::NoneType => Ok(Self::NoneType),
            BuiltinTypeCode::RankedTensorType | BuiltinTypeCode::RankedTensorTypeWithEncoding => {
                let encoding_attr_idx = if code == BuiltinTypeCode::RankedTensorTypeWithEncoding {
                    Some(reader.read_varint()? as u32)
                } else {
                    None
                };
                let shape = reader.read_list(DialectBytecodeReader::read_signed_varint)?;
                let element_type_idx = reader.read_varint()? as u32;
                Ok(Self::RankedTensor {
                    encoding_attr_idx,
                    shape,
                    element_type_idx,
                })
            }
            BuiltinTypeCode::TupleType => {
                let elements = reader.read_list(|r| Ok(r.read_varint()? as u32))?;
                Ok(Self::Tuple { elements })
            }
            BuiltinTypeCode::UnrankedMemRefType
            | BuiltinTypeCode::UnrankedMemRefTypeWithMemSpace => {
                let memory_space_attr_idx =
                    if code == BuiltinTypeCode::UnrankedMemRefTypeWithMemSpace {
                        Some(reader.read_varint()? as u32)
                    } else {
                        None
                    };
                let element_type_idx = reader.read_varint()? as u32;
                Ok(Self::UnrankedMemRef {
                    memory_space_attr_idx,
                    element_type_idx,
                })
            }
            BuiltinTypeCode::UnrankedTensorType => Ok(Self::UnrankedTensor {
                element_type_idx: reader.read_varint()? as u32,
            }),
            BuiltinTypeCode::VectorType | BuiltinTypeCode::VectorTypeWithScalableDims => {
                let shape = reader.read_list(DialectBytecodeReader::read_signed_varint)?;
                let scalable_dims = if code == BuiltinTypeCode::VectorTypeWithScalableDims {
                    reader.read_list(DialectBytecodeReader::read_bool)?
                } else {
                    vec![false; shape.len()]
                };
                let element_type_idx = reader.read_varint()? as u32;
                Ok(Self::Vector {
                    shape,
                    scalable_dims,
                    element_type_idx,
                })
            }
        }
    }

    /// Resolves this [`BuiltinCustomType`] into a high-level [`MlirType`] using previously resolved types and attributes.
    ///
    /// # Arguments
    ///
    /// * `resolved_types` - Already-resolved [`MlirType`] table.
    /// * `resolved_attrs` - Already-resolved [`MlirAttribute`] table.
    #[must_use]
    #[allow(clippy::too_many_lines)]
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
        let map_shape = |dims: &[i64]| -> Vec<DimensionSize> {
            dims.iter()
                .map(|&d| {
                    if d < 0 {
                        DimensionSize::Dynamic
                    } else {
                        DimensionSize::Static(d)
                    }
                })
                .collect()
        };

        match self {
            Self::Integer { width, signedness } => {
                let et = match (*width, *signedness) {
                    (1, _) => ElementType::Pred,
                    (2, 2) => ElementType::U2,
                    (2, _) => ElementType::S2,
                    (4, 2) => ElementType::U4,
                    (4, _) => ElementType::S4,
                    (8, 2) => ElementType::U8,
                    (8, _) => ElementType::S8,
                    (16, 2) => ElementType::U16,
                    (16, _) => ElementType::S16,
                    (64, 2) => ElementType::U64,
                    (64, _) => ElementType::S64,
                    (_, 2) => ElementType::U32,
                    _ => ElementType::S32,
                };
                MlirType::Scalar(et)
            }
            Self::Index => MlirType::Scalar(ElementType::S64),
            Self::Function { inputs, results } => MlirType::Function(AstFunctionType {
                inputs: inputs
                    .iter()
                    .filter_map(|&i| resolved_types.get(i as usize).cloned())
                    .collect(),
                results: results
                    .iter()
                    .filter_map(|&i| resolved_types.get(i as usize).cloned())
                    .collect(),
            }),
            Self::Float(code) => {
                let et = match code {
                    BuiltinTypeCode::BFloat16Type => ElementType::Bf16,
                    BuiltinTypeCode::Float16Type => ElementType::F16,
                    BuiltinTypeCode::Float64Type
                    | BuiltinTypeCode::Float80Type
                    | BuiltinTypeCode::Float128Type => ElementType::F64,
                    _ => ElementType::F32,
                };
                MlirType::Scalar(et)
            }
            Self::Complex { element_type_idx } => {
                let inner = lookup_elem(*element_type_idx);
                let et = if inner == ElementType::F64 {
                    ElementType::C128
                } else {
                    ElementType::C64
                };
                MlirType::Scalar(et)
            }
            Self::MemRef {
                memory_space_attr_idx,
                shape,
                element_type_idx,
                layout_attr_idx,
            } => {
                let memory_space = memory_space_attr_idx
                    .and_then(|idx| resolved_attrs.get(idx as usize))
                    .and_then(|attr| match attr {
                        MlirAttribute::MosaicAttr(
                            crate::compiler::ast::MosaicDialectAttr::MemorySpace(s),
                        ) => Some(*s),
                        MlirAttribute::Integer { value, .. } => Some(match *value {
                            1 => TpuMemorySpace::Vmem,
                            2 => TpuMemorySpace::Smem,
                            3 => TpuMemorySpace::Cmem,
                            4 => TpuMemorySpace::SemaphoreMem,
                            _ => TpuMemorySpace::Hbm,
                        }),
                        MlirAttribute::String(s) => TpuMemorySpace::from_ident(s),
                        _ => None,
                    });
                let layout = layout_attr_idx
                    .and_then(|idx| resolved_attrs.get(idx as usize))
                    .and_then(|attr| match attr {
                        MlirAttribute::AffineMap(expr) => {
                            Some(MemRefLayoutAttr::AffineMap(expr.clone()))
                        }
                        _ => None,
                    });
                MlirType::MemRef(MemRefType {
                    dims: map_shape(shape),
                    element_type: lookup_elem(*element_type_idx),
                    layout,
                    memory_space,
                })
            }
            Self::NoneType => MlirType::Tuple(Vec::new()),
            Self::RankedTensor {
                encoding_attr_idx,
                shape,
                element_type_idx,
            } => {
                let mut dims = map_shape(shape);
                let mut encoding = None;
                if let Some(enc_idx) = encoding_attr_idx
                    && let Some(attr) = resolved_attrs.get(*enc_idx as usize)
                {
                    match attr {
                        MlirAttribute::SdyTensorSharding(sh) => {
                            encoding = Some(TensorEncodingAttr::SdySharding(sh.clone()));
                        }
                        MlirAttribute::Array(bounds_arr) => {
                            for (dim_slot, b_attr) in dims.iter_mut().zip(bounds_arr.iter()) {
                                if *dim_slot == DimensionSize::Dynamic
                                    && let Some(bound_val) = b_attr.as_i64()
                                    && bound_val > 0
                                {
                                    *dim_slot = DimensionSize::Bounded(bound_val);
                                }
                            }
                        }
                        MlirAttribute::String(s) => {
                            encoding = Some(TensorEncodingAttr::Custom(s.clone()));
                        }
                        _ => {}
                    }
                }
                MlirType::RankedTensor(RankedTensorType {
                    dims,
                    element_type: lookup_elem(*element_type_idx),
                    encoding,
                    layout: None,
                })
            }
            Self::Tuple { elements } => MlirType::Tuple(
                elements
                    .iter()
                    .filter_map(|&i| resolved_types.get(i as usize).cloned())
                    .collect(),
            ),
            Self::UnrankedMemRef {
                memory_space_attr_idx,
                element_type_idx,
            } => {
                let memory_space = memory_space_attr_idx
                    .and_then(|idx| resolved_attrs.get(idx as usize))
                    .and_then(MlirAttribute::as_i64)
                    .map(|v| match v {
                        1 => TpuMemorySpace::Vmem,
                        2 => TpuMemorySpace::Smem,
                        3 => TpuMemorySpace::Cmem,
                        _ => TpuMemorySpace::Hbm,
                    });
                MlirType::MemRef(MemRefType {
                    dims: vec![DimensionSize::Dynamic],
                    element_type: lookup_elem(*element_type_idx),
                    layout: None,
                    memory_space,
                })
            }
            Self::UnrankedTensor { element_type_idx } => {
                MlirType::UnrankedTensor(lookup_elem(*element_type_idx))
            }
            Self::Vector {
                shape,
                scalable_dims,
                element_type_idx,
            } => MlirType::Vector(VectorType {
                dims: map_shape(shape),
                scalable_dims: scalable_dims.clone(),
                element_type: lookup_elem(*element_type_idx),
            }),
        }
    }
}

/// Decoded upstream `builtin` dialect custom attribute record from Section `2` (`kAttrType`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuiltinCustomAttr {
    /// `ArrayAttr` (`[attr0, attr1, ...]`).
    Array(Vec<u32>),
    /// `DictionaryAttr` (`{ key_str_or_attr_idx = val_attr_idx, ... }`).
    Dictionary(Vec<(u32, u32)>),
    /// `StringAttr` (`"..."` referencing Section `0` `kString`).
    String {
        /// String index in Section `0` (`kString`).
        string_idx: u32,
        /// Optional typed string `type_idx`.
        type_idx: Option<u32>,
    },
    /// `FlatSymbolRefAttr` (`@sym`).
    FlatSymbolRef(u32),
    /// `SymbolRefAttr` (`@root::@nested0::@nested1`).
    SymbolRef {
        /// Root string or `StringAttr` index.
        root_idx: u32,
        /// Nested flat symbol attribute or string indices.
        nested_indices: Vec<u32>,
    },
    /// `TypeAttr` (`type_idx` in Section `2`).
    TypeAttr(u32),
    /// `UnitAttr` (`unit`).
    Unit,
    /// `IntegerAttr` (`type_idx`, `bit_width`, `value`).
    Integer {
        /// Scalar integer type index in Section `2`.
        type_idx: u32,
        /// Bit width used for `APInt` encoding (`1`, `8`, `16`, `32`, `64`, `128`).
        bit_width: u32,
        /// Signed 128-bit integer value.
        value: i128,
    },
    /// `FloatAttr` (`type_idx`, `element_type`, `f64_bits`).
    Float {
        /// Scalar float type index in Section `2`.
        type_idx: u32,
        /// Floating-point [`ElementType`] semantics.
        element_type: ElementType,
        /// IEEE-754 `f64::to_bits()` payload.
        f64_bits: u64,
    },
    /// `CallSiteLoc` (`callsite(callee at caller)`).
    CallSiteLoc {
        /// Callee location attribute index.
        callee_loc_idx: u32,
        /// Caller location attribute index.
        caller_loc_idx: u32,
    },
    /// `FileLineColLoc` (`loc("file":line:col)`).
    FileLineColLoc {
        /// Filename string index in Section `0` (`kString`).
        file_string_idx: u32,
        /// 1-based source line number.
        line: u32,
        /// 1-based source column number.
        column: u32,
    },
    /// `FusedLoc` (`loc(fused<metadata>[loc0, loc1, ...])`).
    FusedLoc {
        /// Component location attribute indices.
        loc_indices: Vec<u32>,
        /// Optional metadata attribute index.
        metadata_attr_idx: Option<u32>,
    },
    /// `NameLoc` (`loc("name"(child_loc))`).
    NameLoc {
        /// Name string index in Section `0` (`kString`).
        name_string_idx: u32,
        /// Child location attribute index.
        child_loc_idx: u32,
    },
    /// `UnknownLoc` (`loc(unknown)`).
    UnknownLoc,
    /// `DenseResourceElementsAttr` (`dense_resource<handle> : tensor<...>`).
    DenseResourceElements {
        /// Shaped type index in Section `2`.
        shaped_type_idx: u32,
        /// Resource handle index in Section `6` (`kResourceOffset`).
        resource_handle_idx: u32,
    },
    /// `DenseArrayAttr` (`array<i64: ...>`).
    DenseArray {
        /// Scalar [`ElementType`] of the dense array.
        element_type: ElementType,
        /// Element count.
        size: u32,
        /// Little-endian packed element bytes.
        raw_bytes: Vec<u8>,
    },
    /// `DenseIntOrFPElementsAttr` (`dense<...> : tensor<...>`).
    DenseIntOrFpElements {
        /// Shaped type index in Section `2`.
        shaped_type_idx: u32,
        /// Raw packed element bytes (single element when splat).
        raw_bytes: Vec<u8>,
    },
    /// `DenseStringElementsAttr` (`dense<["a", "b"]> : tensor<...>`).
    DenseStringElements {
        /// Shaped type index in Section `2`.
        shaped_type_idx: u32,
        /// Whether the string tensor is a single splat string.
        is_splat: bool,
        /// String table indices in Section `0` (`kString`).
        string_indices: Vec<u32>,
    },
    /// `SparseElementsAttr` (`sparse<indices, values> : tensor<...>`).
    SparseElements {
        /// Shaped type index in Section `2`.
        shaped_type_idx: u32,
        /// Coordinate indices `DenseElementsAttr` index.
        indices_attr_idx: u32,
        /// Non-zero values `DenseElementsAttr` index.
        values_attr_idx: u32,
    },
    /// `DistinctAttr` (`distinct[id]<attr>`).
    Distinct {
        /// Referenced attribute index.
        referenced_attr_idx: u32,
    },
    /// `FileLineColRange` (`loc("file":start_line:start_col to end_line:end_col)`).
    FileLineColRange {
        /// Filename string index in Section `0` (`kString`).
        file_string_idx: u32,
        /// Starting line number.
        start_line: u32,
        /// Starting column number.
        start_col: u32,
        /// Ending line number.
        end_line: u32,
        /// Ending column number.
        end_col: u32,
    },
    /// `AffineMapAttr` (`affine_map<(d0, d1)[s0] -> (...)>`).
    AffineMap(AffineMapExpr),
}

/// Encodes an [`ElementType`] into its canonical 1-byte `PrimitiveType` discriminant (`0..=28`).
///
/// # Arguments
///
/// * `et` - Source [`ElementType`].
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn element_type_to_code(et: ElementType) -> u8 {
    i32::from(et) as u8
}

/// Decodes a 1-byte `PrimitiveType` discriminant (`0..=28`) back into an [`ElementType`].
///
/// # Arguments
///
/// * `code` - Raw 1-byte code.
#[must_use]
fn code_to_element_type(code: u8) -> ElementType {
    ElementType::from_raw(i32::from(code)).unwrap_or(ElementType::S32)
}

/// Encodes an [`AffineMapExpr`] into `writer`.
///
/// # Arguments
///
/// * `writer` - Destination [`DialectBytecodeWriter`].
/// * `map` - Source [`AffineMapExpr`].
fn encode_affine_map(writer: &mut DialectBytecodeWriter, map: &AffineMapExpr) {
    writer.write_list(&map.dims, |w, s| w.write_nul_string(s));
    writer.write_list(&map.symbols, |w, s| w.write_nul_string(s));
    writer.write_list(&map.results, |w, s| w.write_nul_string(s));
}

/// Decodes an [`AffineMapExpr`] from `reader`.
///
/// # Arguments
///
/// * `reader` - Source [`DialectBytecodeReader`].
fn decode_affine_map(reader: &mut DialectBytecodeReader<'_>) -> LibTpuResult<AffineMapExpr> {
    let dims = reader.read_list(|r| Ok(String::from(r.read_nul_string()?)))?;
    let symbols = reader.read_list(|r| Ok(String::from(r.read_nul_string()?)))?;
    let results = reader.read_list(|r| Ok(String::from(r.read_nul_string()?)))?;
    Ok(AffineMapExpr {
        dims,
        symbols,
        results,
    })
}

impl BuiltinCustomAttr {
    /// Serializes this [`BuiltinCustomAttr`] into `writer` following `BuiltinDialectBytecode.td`.
    ///
    /// # Arguments
    ///
    /// * `writer` - Destination [`DialectBytecodeWriter`].
    #[allow(clippy::too_many_lines)]
    pub fn encode(&self, writer: &mut DialectBytecodeWriter) {
        match self {
            Self::Array(indices) => {
                writer.write_varint(BuiltinAttrCode::ArrayAttr as u64);
                writer.write_list(indices, |w, &idx| w.write_varint(u64::from(idx)));
            }
            Self::Dictionary(entries) => {
                writer.write_varint(BuiltinAttrCode::DictionaryAttr as u64);
                writer.write_list(entries, |w, &(k, v)| {
                    w.write_varint(u64::from(k));
                    w.write_varint(u64::from(v));
                });
            }
            Self::String {
                string_idx,
                type_idx: None,
            } => {
                writer.write_varint(BuiltinAttrCode::StringAttr as u64);
                writer.write_varint(u64::from(*string_idx));
            }
            Self::String {
                string_idx,
                type_idx: Some(ty_idx),
            } => {
                writer.write_varint(BuiltinAttrCode::StringAttrWithType as u64);
                writer.write_varint(u64::from(*string_idx));
                writer.write_varint(u64::from(*ty_idx));
            }
            Self::FlatSymbolRef(str_idx) => {
                writer.write_varint(BuiltinAttrCode::FlatSymbolRefAttr as u64);
                writer.write_varint(u64::from(*str_idx));
            }
            Self::SymbolRef {
                root_idx,
                nested_indices,
            } => {
                writer.write_varint(BuiltinAttrCode::SymbolRefAttr as u64);
                writer.write_varint(u64::from(*root_idx));
                writer.write_list(nested_indices, |w, &idx| w.write_varint(u64::from(idx)));
            }
            Self::TypeAttr(ty_idx) => {
                writer.write_varint(BuiltinAttrCode::TypeAttr as u64);
                writer.write_varint(u64::from(*ty_idx));
            }
            Self::Unit => {
                writer.write_varint(BuiltinAttrCode::UnitAttr as u64);
            }
            Self::Integer {
                type_idx,
                bit_width,
                value,
            } => {
                writer.write_varint(BuiltinAttrCode::IntegerAttr as u64);
                writer.write_varint(u64::from(*type_idx));
                writer.write_varint(u64::from(*bit_width));
                writer.write_apint(*bit_width, *value);
            }
            Self::Float {
                type_idx,
                element_type,
                f64_bits,
            } => {
                writer.write_varint(BuiltinAttrCode::FloatAttr as u64);
                writer.write_varint(u64::from(*type_idx));
                writer.write_byte(element_type_to_code(*element_type));
                writer.write_apfloat_f64_bits(*element_type, *f64_bits);
            }
            Self::CallSiteLoc {
                callee_loc_idx,
                caller_loc_idx,
            } => {
                writer.write_varint(BuiltinAttrCode::CallSiteLoc as u64);
                writer.write_varint(u64::from(*callee_loc_idx));
                writer.write_varint(u64::from(*caller_loc_idx));
            }
            Self::FileLineColLoc {
                file_string_idx,
                line,
                column,
            } => {
                writer.write_varint(BuiltinAttrCode::FileLineColLoc as u64);
                writer.write_varint(u64::from(*file_string_idx));
                writer.write_varint(u64::from(*line));
                writer.write_varint(u64::from(*column));
            }
            Self::FusedLoc {
                loc_indices,
                metadata_attr_idx: None,
            } => {
                writer.write_varint(BuiltinAttrCode::FusedLoc as u64);
                writer.write_list(loc_indices, |w, &idx| w.write_varint(u64::from(idx)));
            }
            Self::FusedLoc {
                loc_indices,
                metadata_attr_idx: Some(meta_idx),
            } => {
                writer.write_varint(BuiltinAttrCode::FusedLocWithMetadata as u64);
                writer.write_list(loc_indices, |w, &idx| w.write_varint(u64::from(idx)));
                writer.write_varint(u64::from(*meta_idx));
            }
            Self::NameLoc {
                name_string_idx,
                child_loc_idx,
            } => {
                writer.write_varint(BuiltinAttrCode::NameLoc as u64);
                writer.write_varint(u64::from(*name_string_idx));
                writer.write_varint(u64::from(*child_loc_idx));
            }
            Self::UnknownLoc => {
                writer.write_varint(BuiltinAttrCode::UnknownLoc as u64);
            }
            Self::DenseResourceElements {
                shaped_type_idx,
                resource_handle_idx,
            } => {
                writer.write_varint(BuiltinAttrCode::DenseResourceElementsAttr as u64);
                writer.write_varint(u64::from(*shaped_type_idx));
                writer.write_varint(u64::from(*resource_handle_idx));
            }
            Self::DenseArray {
                element_type,
                size,
                raw_bytes,
            } => {
                writer.write_varint(BuiltinAttrCode::DenseArrayAttr as u64);
                writer.write_byte(element_type_to_code(*element_type));
                writer.write_varint(u64::from(*size));
                writer.write_blob(raw_bytes);
            }
            Self::DenseIntOrFpElements {
                shaped_type_idx,
                raw_bytes,
            } => {
                writer.write_varint(BuiltinAttrCode::DenseIntOrFPElementsAttr as u64);
                writer.write_varint(u64::from(*shaped_type_idx));
                writer.write_blob(raw_bytes);
            }
            Self::DenseStringElements {
                shaped_type_idx,
                is_splat,
                string_indices,
            } => {
                writer.write_varint(BuiltinAttrCode::DenseStringElementsAttr as u64);
                writer.write_varint(u64::from(*shaped_type_idx));
                writer.write_bool(*is_splat);
                writer.write_list(string_indices, |w, &idx| w.write_varint(u64::from(idx)));
            }
            Self::SparseElements {
                shaped_type_idx,
                indices_attr_idx,
                values_attr_idx,
            } => {
                writer.write_varint(BuiltinAttrCode::SparseElementsAttr as u64);
                writer.write_varint(u64::from(*shaped_type_idx));
                writer.write_varint(u64::from(*indices_attr_idx));
                writer.write_varint(u64::from(*values_attr_idx));
            }
            Self::Distinct {
                referenced_attr_idx,
            } => {
                writer.write_varint(BuiltinAttrCode::DistinctAttr as u64);
                writer.write_varint(u64::from(*referenced_attr_idx));
            }
            Self::FileLineColRange {
                file_string_idx,
                start_line,
                start_col,
                end_line,
                end_col,
            } => {
                writer.write_varint(BuiltinAttrCode::FileLineColRange as u64);
                writer.write_varint(u64::from(*file_string_idx));
                writer.write_varint(u64::from(*start_line));
                writer.write_varint(u64::from(*start_col));
                writer.write_varint(u64::from(*end_line));
                writer.write_varint(u64::from(*end_col));
            }
            Self::AffineMap(map) => {
                writer.write_varint(BuiltinAttrCode::AffineMapAttr as u64);
                encode_affine_map(writer, map);
            }
        }
    }

    /// Decodes a [`BuiltinCustomAttr`] from `reader` following `BuiltinDialectBytecode.td`.
    ///
    /// # Arguments
    ///
    /// * `reader` - Source [`DialectBytecodeReader`].
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MlirBytecodeDecodeError`] if the tag is unknown or payload is truncated.
    #[allow(clippy::cast_possible_truncation, clippy::too_many_lines)]
    pub fn decode(reader: &mut DialectBytecodeReader<'_>) -> LibTpuResult<Self> {
        let code = BuiltinAttrCode::from_tag(reader.read_varint()?)?;
        match code {
            BuiltinAttrCode::ArrayAttr => {
                let indices = reader.read_list(|r| Ok(r.read_varint()? as u32))?;
                Ok(Self::Array(indices))
            }
            BuiltinAttrCode::DictionaryAttr => {
                let entries = reader.read_list(|r| {
                    let k = r.read_varint()? as u32;
                    let v = r.read_varint()? as u32;
                    Ok((k, v))
                })?;
                Ok(Self::Dictionary(entries))
            }
            BuiltinAttrCode::StringAttr => Ok(Self::String {
                string_idx: reader.read_varint()? as u32,
                type_idx: None,
            }),
            BuiltinAttrCode::StringAttrWithType => Ok(Self::String {
                string_idx: reader.read_varint()? as u32,
                type_idx: Some(reader.read_varint()? as u32),
            }),
            BuiltinAttrCode::FlatSymbolRefAttr => {
                Ok(Self::FlatSymbolRef(reader.read_varint()? as u32))
            }
            BuiltinAttrCode::SymbolRefAttr => {
                let root_idx = reader.read_varint()? as u32;
                let nested_indices = reader.read_list(|r| Ok(r.read_varint()? as u32))?;
                Ok(Self::SymbolRef {
                    root_idx,
                    nested_indices,
                })
            }
            BuiltinAttrCode::TypeAttr => Ok(Self::TypeAttr(reader.read_varint()? as u32)),
            BuiltinAttrCode::UnitAttr => Ok(Self::Unit),
            BuiltinAttrCode::IntegerAttr => {
                let type_idx = reader.read_varint()? as u32;
                let bit_width = reader.read_varint()? as u32;
                let value = reader.read_apint(bit_width)?;
                Ok(Self::Integer {
                    type_idx,
                    bit_width,
                    value,
                })
            }
            BuiltinAttrCode::FloatAttr => {
                let type_idx = reader.read_varint()? as u32;
                let element_type = code_to_element_type(reader.read_byte()?);
                let f64_bits = reader.read_apfloat_f64_bits(element_type)?;
                Ok(Self::Float {
                    type_idx,
                    element_type,
                    f64_bits,
                })
            }
            BuiltinAttrCode::CallSiteLoc => Ok(Self::CallSiteLoc {
                callee_loc_idx: reader.read_varint()? as u32,
                caller_loc_idx: reader.read_varint()? as u32,
            }),
            BuiltinAttrCode::FileLineColLoc => Ok(Self::FileLineColLoc {
                file_string_idx: reader.read_varint()? as u32,
                line: reader.read_varint()? as u32,
                column: reader.read_varint()? as u32,
            }),
            BuiltinAttrCode::FusedLoc => {
                let loc_indices = reader.read_list(|r| Ok(r.read_varint()? as u32))?;
                Ok(Self::FusedLoc {
                    loc_indices,
                    metadata_attr_idx: None,
                })
            }
            BuiltinAttrCode::FusedLocWithMetadata => {
                let loc_indices = reader.read_list(|r| Ok(r.read_varint()? as u32))?;
                let meta_idx = reader.read_varint()? as u32;
                Ok(Self::FusedLoc {
                    loc_indices,
                    metadata_attr_idx: Some(meta_idx),
                })
            }
            BuiltinAttrCode::NameLoc => Ok(Self::NameLoc {
                name_string_idx: reader.read_varint()? as u32,
                child_loc_idx: reader.read_varint()? as u32,
            }),
            BuiltinAttrCode::UnknownLoc => Ok(Self::UnknownLoc),
            BuiltinAttrCode::DenseResourceElementsAttr => Ok(Self::DenseResourceElements {
                shaped_type_idx: reader.read_varint()? as u32,
                resource_handle_idx: reader.read_varint()? as u32,
            }),
            BuiltinAttrCode::DenseArrayAttr => {
                let element_type = code_to_element_type(reader.read_byte()?);
                let size = reader.read_varint()? as u32;
                let raw_bytes = reader.read_blob()?.to_vec();
                Ok(Self::DenseArray {
                    element_type,
                    size,
                    raw_bytes,
                })
            }
            BuiltinAttrCode::DenseIntOrFPElementsAttr => {
                let shaped_type_idx = reader.read_varint()? as u32;
                let raw_bytes = reader.read_blob()?.to_vec();
                Ok(Self::DenseIntOrFpElements {
                    shaped_type_idx,
                    raw_bytes,
                })
            }
            BuiltinAttrCode::DenseStringElementsAttr => {
                let shaped_type_idx = reader.read_varint()? as u32;
                let is_splat = reader.read_bool()?;
                let string_indices = reader.read_list(|r| Ok(r.read_varint()? as u32))?;
                Ok(Self::DenseStringElements {
                    shaped_type_idx,
                    is_splat,
                    string_indices,
                })
            }
            BuiltinAttrCode::SparseElementsAttr => Ok(Self::SparseElements {
                shaped_type_idx: reader.read_varint()? as u32,
                indices_attr_idx: reader.read_varint()? as u32,
                values_attr_idx: reader.read_varint()? as u32,
            }),
            BuiltinAttrCode::DistinctAttr => Ok(Self::Distinct {
                referenced_attr_idx: reader.read_varint()? as u32,
            }),
            BuiltinAttrCode::FileLineColRange => Ok(Self::FileLineColRange {
                file_string_idx: reader.read_varint()? as u32,
                start_line: reader.read_varint()? as u32,
                start_col: reader.read_varint()? as u32,
                end_line: reader.read_varint()? as u32,
                end_col: reader.read_varint()? as u32,
            }),
            BuiltinAttrCode::AffineMapAttr => Ok(Self::AffineMap(decode_affine_map(reader)?)),
        }
    }

    /// Resolves this [`BuiltinCustomAttr`] into a high-level [`MlirAttribute`] using resolved string, type, attribute, and resource tables.
    ///
    /// # Arguments
    ///
    /// * `strings` - Section `0` (`kString`) table.
    /// * `resolved_types` - Previously-resolved [`MlirType`] table.
    /// * `resolved_attrs` - Previously-resolved [`MlirAttribute`] table.
    /// * `resource_keys` - Optional Section `6` (`kResourceOffset`) resource key names.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn to_mlir_attribute(
        &self,
        strings: &[String],
        resolved_types: &[MlirType],
        resolved_attrs: &[MlirAttribute],
        resource_keys: &[String],
    ) -> MlirAttribute {
        let lookup_str = |idx: u32| -> String {
            strings
                .get(idx as usize)
                .cloned()
                .unwrap_or_else(|| format!("str_{idx}"))
        };
        let lookup_loc = |idx: u32| -> MlirLocation {
            match resolved_attrs.get(idx as usize) {
                Some(MlirAttribute::Location(loc)) => loc.clone(),
                _ => MlirLocation::Alias(format!("#loc{idx}")),
            }
        };

        match self {
            Self::Array(indices) => MlirAttribute::Array(
                indices
                    .iter()
                    .filter_map(|&i| resolved_attrs.get(i as usize).cloned())
                    .collect(),
            ),
            Self::Dictionary(entries) => {
                let mut map = BTreeMap::new();
                for &(k_idx, v_idx) in entries {
                    let key = resolved_attrs
                        .get(k_idx as usize)
                        .and_then(MlirAttribute::as_str)
                        .map_or_else(|| lookup_str(k_idx), String::from);
                    if let Some(val) = resolved_attrs.get(v_idx as usize).cloned() {
                        map.insert(key, val);
                    }
                }
                MlirAttribute::Dictionary(map)
            }
            Self::String { string_idx, .. } => MlirAttribute::String(lookup_str(*string_idx)),
            Self::FlatSymbolRef(str_idx) => MlirAttribute::SymbolRef(vec![lookup_str(*str_idx)]),
            Self::SymbolRef {
                root_idx,
                nested_indices,
            } => {
                let mut chain = vec![lookup_str(*root_idx)];
                for &n_idx in nested_indices {
                    let sym = match resolved_attrs.get(n_idx as usize) {
                        Some(MlirAttribute::SymbolRef(v)) => v.first().cloned().unwrap_or_default(),
                        _ => lookup_str(n_idx),
                    };
                    chain.push(sym);
                }
                MlirAttribute::SymbolRef(chain)
            }
            Self::TypeAttr(ty_idx) => resolved_types.get(*ty_idx as usize).map_or(
                MlirAttribute::Type(MlirType::Scalar(ElementType::F32)),
                |t| MlirAttribute::Type(t.clone()),
            ),
            Self::Unit => MlirAttribute::Bool(true),
            Self::Integer {
                type_idx,
                bit_width,
                value,
            } => {
                let et = resolved_types
                    .get(*type_idx as usize)
                    .map(MlirType::element_type);
                if *bit_width == 1 || et == Some(ElementType::Pred) {
                    MlirAttribute::Bool(*value != 0)
                } else {
                    MlirAttribute::Integer {
                        value: *value,
                        ty: et,
                    }
                }
            }
            Self::Float {
                element_type,
                f64_bits,
                ..
            } => MlirAttribute::Float {
                value: f64::from_bits(*f64_bits),
                ty: Some(*element_type),
            },
            Self::CallSiteLoc {
                callee_loc_idx,
                caller_loc_idx,
            } => MlirAttribute::Location(MlirLocation::CallSite {
                callee: Box::new(lookup_loc(*callee_loc_idx)),
                caller: Box::new(lookup_loc(*caller_loc_idx)),
            }),
            Self::FileLineColLoc {
                file_string_idx,
                line,
                column,
            } => MlirAttribute::Location(MlirLocation::FileLineCol {
                file: lookup_str(*file_string_idx),
                line: *line as usize,
                column: *column as usize,
            }),
            Self::FusedLoc { loc_indices, .. } => MlirAttribute::Location(MlirLocation::Fused(
                loc_indices.iter().map(|&idx| lookup_loc(idx)).collect(),
            )),
            Self::NameLoc {
                name_string_idx,
                child_loc_idx,
            } => MlirAttribute::Location(MlirLocation::Named {
                name: lookup_str(*name_string_idx),
                child: Some(Box::new(lookup_loc(*child_loc_idx))),
            }),
            Self::UnknownLoc => MlirAttribute::Location(MlirLocation::Unknown),
            Self::DenseResourceElements {
                resource_handle_idx,
                ..
            } => {
                let handle = resource_keys
                    .get(*resource_handle_idx as usize)
                    .cloned()
                    .unwrap_or_else(|| format!("res_{resource_handle_idx}"));
                MlirAttribute::DenseResource(handle)
            }
            Self::DenseArray {
                element_type,
                raw_bytes,
                ..
            } => {
                let vals = element_type.unpack_to_f64_vec(raw_bytes);
                let int_vals: Vec<i64> = vals
                    .into_iter()
                    .map(|v| {
                        #[allow(clippy::cast_possible_truncation)]
                        {
                            v as i64
                        }
                    })
                    .collect();
                MlirAttribute::DenseI64Array(int_vals)
            }
            Self::DenseIntOrFpElements {
                shaped_type_idx,
                raw_bytes,
            } => {
                let ranked_ty = match resolved_types.get(*shaped_type_idx as usize) {
                    Some(MlirType::RankedTensor(rt)) => Some(rt.clone()),
                    _ => None,
                };
                let elem_ty = ranked_ty
                    .as_ref()
                    .map_or(ElementType::F32, |rt| rt.element_type);
                let elem_count = ranked_ty.as_ref().map_or(1, |rt| {
                    rt.dims
                        .iter()
                        .map(|d| usize::try_from(d.concrete_or_bound().max(1)).unwrap_or(1))
                        .product::<usize>()
                });
                let unpacked = elem_ty.unpack_to_f64_vec(raw_bytes);
                let is_splat = elem_count > 1 && unpacked.len() == 1;
                MlirAttribute::DenseElements(DenseElementsAttr {
                    values_f64: if unpacked.is_empty() {
                        vec![0.0]
                    } else {
                        unpacked
                    },
                    raw_bytes: Some(raw_bytes.clone()),
                    is_splat,
                    tensor_type: ranked_ty,
                })
            }
            Self::DenseStringElements { string_indices, .. } => MlirAttribute::Array(
                string_indices
                    .iter()
                    .map(|&idx| MlirAttribute::String(lookup_str(idx)))
                    .collect(),
            ),
            Self::SparseElements {
                indices_attr_idx,
                values_attr_idx,
                ..
            } => {
                let mut items = Vec::new();
                if let Some(ia) = resolved_attrs.get(*indices_attr_idx as usize) {
                    items.push(ia.clone());
                }
                if let Some(va) = resolved_attrs.get(*values_attr_idx as usize) {
                    items.push(va.clone());
                }
                MlirAttribute::Array(items)
            }
            Self::Distinct {
                referenced_attr_idx,
            } => resolved_attrs
                .get(*referenced_attr_idx as usize)
                .cloned()
                .unwrap_or(MlirAttribute::Bool(true)),
            Self::FileLineColRange {
                file_string_idx,
                start_line,
                start_col,
                ..
            } => MlirAttribute::Location(MlirLocation::FileLineCol {
                file: lookup_str(*file_string_idx),
                line: *start_line as usize,
                column: *start_col as usize,
            }),
            Self::AffineMap(map) => MlirAttribute::AffineMap(map.clone()),
        }
    }
}

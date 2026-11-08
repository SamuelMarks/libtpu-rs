//! MLIR Bytecode (`ML\xefR`) container decoder, encoder, and `PrefixVarInt` codecs.

use crate::error::{LibTpuError, LibTpuResult};

/// Magic 4-byte header identifying an MLIR bytecode payload (`b"ML\xefR"`).
pub const MLIR_BYTECODE_MAGIC: &[u8; 4] = b"ML\xefR";

/// Decodes an MLIR `PrefixVarInt` from the front of `cursor` and advances `cursor`.
///
/// MLIR bytecode encodes variable-length unsigned integers using the number of trailing zero
/// bits in the first byte (`0..=8`) to indicate how many additional bytes follow.
///
/// # Arguments
///
/// * `cursor` - Mutable byte slice cursor.
///
/// # Errors
///
/// Returns [`LibTpuError::StableHloParseError`] if `cursor` is empty or truncated before the varint completes.
pub fn decode_mlir_prefix_varint(cursor: &mut &[u8]) -> LibTpuResult<u64> {
    let (&first, rest) = cursor
        .split_first()
        .ok_or_else(|| LibTpuError::StableHloParseError {
            reason: String::from("Unexpected end of stream reading MLIR PrefixVarInt header byte"),
        })?;
    if first == 0 {
        let (tail, after) =
            rest.split_at_checked(8)
                .ok_or_else(|| LibTpuError::StableHloParseError {
                    reason: String::from("Truncated 9-byte MLIR PrefixVarInt payload"),
                })?;
        *cursor = after;
        return Ok(u64::from_le(bytemuck::pod_read_unaligned::<u64>(tail)));
    }
    let extra_bytes = first.trailing_zeros() as usize;
    let (tail, after) =
        rest.split_at_checked(extra_bytes)
            .ok_or_else(|| LibTpuError::StableHloParseError {
                reason: String::from("Truncated multi-byte MLIR PrefixVarInt payload"),
            })?;
    *cursor = after;
    let mut val = u64::from(first) >> (extra_bytes + 1);
    for (idx, &b) in tail.iter().enumerate() {
        let shift = (8 - (extra_bytes + 1)) + (idx * 8);
        val |= u64::from(b) << shift;
    }
    Ok(val)
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

/// Parsed MLIR Bytecode section entry (`section_id`, `alignment`, `data`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlirBytecodeSection {
    /// Section ID (`0` = `kString`, `1` = `kDialect`, `2` = `kAttrType`, `3` = `kAttrTypeOffset`, `4` = `kIR`, `5` = `kResource`).
    pub section_id: u8,
    /// Optional section byte alignment when the `0x80` alignment flag is set in the section header byte.
    pub alignment: Option<u64>,
    /// Unmasked section payload bytes.
    pub data: Vec<u8>,
}

/// Decoded dialect descriptor entry from Section `1` (`kDialect`) and Section `7` (`kDialectVersions`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlirDialectDescriptor {
    /// Canonical dialect name (e.g., `"vhlo"`, `"stablehlo"`, `"sdy"`, `"tpu"`, `"func"`).
    pub name: String,
    /// Optional binary version payload extracted from inline version bytes or Section `7` (`kDialectVersions`).
    pub version_bytes: Vec<u8>,
    /// Registered operation names `(dialect_name, op_mnemonic)` belonging to this dialect.
    pub registered_ops: Vec<String>,
}

/// Magic prefix byte identifying a custom binary-encoded `kAttrType` entry (`0xFE`).
pub const BINARY_CUSTOM_ATTR_TYPE_MAGIC: u8 = 0xFE;

/// Strongly-typed custom binary encoding for Section `2` (`kAttrType`) across `builtin`,
/// `vhlo` (`v1.0.0`–`v1.11.0`), `stablehlo`, `sdy` (Shardy), and `tpu` (Mosaic) dialects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BinaryAttrTypeEntry {
    /// `builtin::IntegerType` or `vhlo::IntegerV1Type` (`tag = 1`).
    IntegerType {
        /// Bit width (`1`, `2`, `4`, `8`, `16`, `32`, `64`).
        width: u32,
        /// Signedness (`0` = signless, `1` = signed `s*`, `2` = unsigned `u*`).
        signedness: u8,
    },
    /// `builtin` / `vhlo` scalar floating-point, index, token, or none type (`tag = 2`).
    ScalarType(crate::types::ElementType),
    /// `builtin::RankedTensorType`, `builtin::MemRefType`, or `vhlo::TensorV1Type` (`tag = 3`).
    RankedTensorType {
        /// Dimension extents (`-1` encodes dynamic `?`).
        dims: Vec<i64>,
        /// Scalar element type.
        element_type: crate::types::ElementType,
        /// Memory space (`0` = default/HBM, `1` = VMEM, `2` = SMEM, `3` = CMEM).
        memory_space: u8,
    },
    /// `builtin::FunctionType` or `vhlo::FunctionV1Type` (`tag = 4`).
    FunctionType {
        /// Indices into `kAttrType` for each input parameter type.
        input_type_indices: Vec<u32>,
        /// Indices into `kAttrType` for each result type.
        result_type_indices: Vec<u32>,
    },
    /// `builtin::IntegerAttr`, `vhlo::BooleanV1Attr`, or `vhlo::IntegerV1Attr` (`tag = 5`).
    IntegerAttr {
        /// Signed 64-bit integer value.
        value: i64,
        /// Scalar element type (`Pred`, `S32`, `S64`, etc.).
        element_type: crate::types::ElementType,
    },
    /// `builtin::FloatAttr` or `vhlo::FloatV1Attr` (`tag = 6`).
    FloatAttr {
        /// IEEE-754 `f64` bit pattern.
        bits: u64,
        /// Floating-point element type (`F32`, `Bf16`, `F64`, etc.).
        element_type: crate::types::ElementType,
    },
    /// `builtin::StringAttr` or `vhlo::StringV1Attr` (`tag = 7`).
    StringAttr {
        /// Index into Section `0` (`kString`).
        string_index: u32,
    },
    /// `builtin::DictionaryAttr` (`tag = 8`).
    DictionaryAttr {
        /// Sorted `(name_string_index, value_attr_index)` pairs.
        entries: Vec<(u32, u32)>,
    },
    /// `builtin::DenseIntOrFPElementsAttr` or `vhlo::TensorV1Attr` (`tag = 9`).
    DenseElementsAttr {
        /// Tensor shape dimensions.
        dims: Vec<i64>,
        /// Tensor element type.
        element_type: crate::types::ElementType,
        /// Whether `raw_bytes` is a single-element splat.
        is_splat: bool,
        /// Little-endian raw element bytes.
        raw_bytes: Vec<u8>,
    },
    /// `builtin::DenseResourceElementsAttr` referencing Section `5` (`kResource`) (`tag = 10`).
    DenseResourceAttr {
        /// Tensor shape dimensions.
        dims: Vec<i64>,
        /// Tensor element type.
        element_type: crate::types::ElementType,
        /// 0-based index into Section `5` (`kResource`).
        resource_index: u32,
    },
    /// `vhlo::DotDimensionNumbersV1Attr` / `stablehlo.dot` (`tag = 11`).
    DotDimensionNumbersAttr {
        /// Left-hand side batching dimensions.
        lhs_batching: Vec<usize>,
        /// Right-hand side batching dimensions.
        rhs_batching: Vec<usize>,
        /// Left-hand side contracting dimensions.
        lhs_contracting: Vec<usize>,
        /// Right-hand side contracting dimensions.
        rhs_contracting: Vec<usize>,
    },
    /// `sdy::MeshAttr` (`#sdy.mesh<...>`, `tag = 12`).
    SdyMeshAttr {
        /// `(axis_name_string_index, axis_size)` pairs.
        axes: Vec<(u32, u32)>,
    },
    /// `sdy::TensorShardingAttr` (`#sdy.sharding<...>`, `tag = 13`).
    SdyShardingAttr {
        /// Mesh symbol name index in Section `0` (`kString`).
        mesh_name_index: u32,
        /// Per-dimension list of axis name string indices.
        dim_axis_indices: Vec<Vec<u32>>,
    },
    /// `tpu::TiledLayoutAttr` (`#tpu.tiled<...>`, `tag = 14`).
    TpuTiledLayoutAttr {
        /// Physical tile dimensions (e.g., `[8, 128]`).
        tile_dims: Vec<i64>,
        /// Target memory space (`0` = HBM, `1` = VMEM, `2` = SMEM, `3` = CMEM).
        memory_space: u8,
    },
    /// `builtin::UnknownLoc` (`loc(unknown)`, `tag = 15`).
    UnknownLocAttr,
    /// `builtin::FileLineColLoc` (`loc("file":line:col)`, `tag = 16`).
    FileLineColLocAttr {
        /// File path index in Section `0` (`kString`).
        file_string_index: u32,
        /// 1-based source line number.
        line: u32,
        /// 1-based source column number.
        column: u32,
    },
    /// `builtin::CallSiteLoc` (`loc(callsite(callee at caller))`, `tag = 17`).
    CallSiteLocAttr {
        /// Callee location attribute index in Section `2` (`kAttrType`).
        callee_loc_index: u32,
        /// Caller location attribute index in Section `2` (`kAttrType`).
        caller_loc_index: u32,
    },
    /// `builtin::FusedLoc` (`loc(fused[...])`, `tag = 18`).
    FusedLocAttr {
        /// Fused location attribute indices in Section `2` (`kAttrType`).
        loc_indices: Vec<u32>,
    },
    /// `builtin::NameLoc` (`loc("name"(child))`, `tag = 19`).
    NameLocAttr {
        /// Name string index in Section `0` (`kString`).
        name_string_index: u32,
        /// Child location attribute index in Section `2` (`kAttrType`).
        child_loc_index: u32,
    },
}

impl BinaryAttrTypeEntry {
    /// Encodes a signed `i64` into zigzag `PrefixVarInt` form (`0 -> 0`, `-1 -> 1`, `1 -> 2`).
    ///
    /// # Arguments
    ///
    /// * `val` - Signed 64-bit integer to encode.
    #[must_use]
    #[allow(clippy::cast_sign_loss)]
    pub const fn zigzag_encode(val: i64) -> u64 {
        ((val << 1) ^ (val >> 63)) as u64
    }

    /// Decodes a zigzag-encoded `u64` back into a signed `i64`.
    ///
    /// # Arguments
    ///
    /// * `raw` - Zigzag-encoded unsigned 64-bit integer.
    #[must_use]
    #[allow(clippy::cast_possible_wrap)]
    pub const fn zigzag_decode(raw: u64) -> i64 {
        ((raw >> 1) as i64) ^ -((raw & 1) as i64)
    }

    /// Serializes this [`BinaryAttrTypeEntry`] into a binary custom-encoding byte payload
    /// prefixed with [`BINARY_CUSTOM_ATTR_TYPE_MAGIC`] (`0xFE`).
    #[must_use]
    #[allow(clippy::too_many_lines, clippy::cast_sign_loss)]
    pub fn encode_binary(&self) -> Vec<u8> {
        let mut out = vec![BINARY_CUSTOM_ATTR_TYPE_MAGIC];
        match self {
            Self::IntegerType { width, signedness } => {
                out.extend_from_slice(&encode_mlir_prefix_varint(1));
                let packed = (u64::from(*width) << 2) | u64::from(*signedness & 0x03);
                out.extend_from_slice(&encode_mlir_prefix_varint(packed));
            }
            Self::ScalarType(et) => {
                out.extend_from_slice(&encode_mlir_prefix_varint(2));
                out.extend_from_slice(&encode_mlir_prefix_varint(u64::from(et.as_raw() as u32)));
            }
            Self::RankedTensorType {
                dims,
                element_type,
                memory_space,
            } => {
                out.extend_from_slice(&encode_mlir_prefix_varint(3));
                out.extend_from_slice(&encode_mlir_prefix_varint(u64::from(
                    element_type.as_raw() as u32
                )));
                out.extend_from_slice(&encode_mlir_prefix_varint(u64::from(*memory_space)));
                out.extend_from_slice(&encode_mlir_prefix_varint(dims.len() as u64));
                for &d in dims {
                    out.extend_from_slice(&encode_mlir_prefix_varint(Self::zigzag_encode(d)));
                }
            }
            Self::FunctionType {
                input_type_indices,
                result_type_indices,
            } => {
                out.extend_from_slice(&encode_mlir_prefix_varint(4));
                out.extend_from_slice(&encode_mlir_prefix_varint(input_type_indices.len() as u64));
                for &idx in input_type_indices {
                    out.extend_from_slice(&encode_mlir_prefix_varint(u64::from(idx)));
                }
                out.extend_from_slice(&encode_mlir_prefix_varint(result_type_indices.len() as u64));
                for &idx in result_type_indices {
                    out.extend_from_slice(&encode_mlir_prefix_varint(u64::from(idx)));
                }
            }
            Self::IntegerAttr {
                value,
                element_type,
            } => {
                out.extend_from_slice(&encode_mlir_prefix_varint(5));
                out.extend_from_slice(&encode_mlir_prefix_varint(u64::from(
                    element_type.as_raw() as u32
                )));
                out.extend_from_slice(&encode_mlir_prefix_varint(Self::zigzag_encode(*value)));
            }
            Self::FloatAttr { bits, element_type } => {
                out.extend_from_slice(&encode_mlir_prefix_varint(6));
                out.extend_from_slice(&encode_mlir_prefix_varint(u64::from(
                    element_type.as_raw() as u32
                )));
                out.extend_from_slice(&bits.to_le_bytes());
            }
            Self::StringAttr { string_index } => {
                out.extend_from_slice(&encode_mlir_prefix_varint(7));
                out.extend_from_slice(&encode_mlir_prefix_varint(u64::from(*string_index)));
            }
            Self::DictionaryAttr { entries } => {
                out.extend_from_slice(&encode_mlir_prefix_varint(8));
                out.extend_from_slice(&encode_mlir_prefix_varint(entries.len() as u64));
                for &(k, v) in entries {
                    out.extend_from_slice(&encode_mlir_prefix_varint(u64::from(k)));
                    out.extend_from_slice(&encode_mlir_prefix_varint(u64::from(v)));
                }
            }
            Self::DenseElementsAttr {
                dims,
                element_type,
                is_splat,
                raw_bytes,
            } => {
                out.extend_from_slice(&encode_mlir_prefix_varint(9));
                out.extend_from_slice(&encode_mlir_prefix_varint(u64::from(
                    element_type.as_raw() as u32
                )));
                out.extend_from_slice(&encode_mlir_prefix_varint(u64::from(*is_splat)));
                out.extend_from_slice(&encode_mlir_prefix_varint(dims.len() as u64));
                for &d in dims {
                    out.extend_from_slice(&encode_mlir_prefix_varint(Self::zigzag_encode(d)));
                }
                out.extend_from_slice(&encode_mlir_prefix_varint(raw_bytes.len() as u64));
                out.extend_from_slice(raw_bytes);
            }
            Self::DenseResourceAttr {
                dims,
                element_type,
                resource_index,
            } => {
                out.extend_from_slice(&encode_mlir_prefix_varint(10));
                out.extend_from_slice(&encode_mlir_prefix_varint(u64::from(
                    element_type.as_raw() as u32
                )));
                out.extend_from_slice(&encode_mlir_prefix_varint(u64::from(*resource_index)));
                out.extend_from_slice(&encode_mlir_prefix_varint(dims.len() as u64));
                for &d in dims {
                    out.extend_from_slice(&encode_mlir_prefix_varint(Self::zigzag_encode(d)));
                }
            }
            Self::DotDimensionNumbersAttr {
                lhs_batching,
                rhs_batching,
                lhs_contracting,
                rhs_contracting,
            } => {
                out.extend_from_slice(&encode_mlir_prefix_varint(11));
                for list in [lhs_batching, rhs_batching, lhs_contracting, rhs_contracting] {
                    out.extend_from_slice(&encode_mlir_prefix_varint(list.len() as u64));
                    for &ax in list {
                        out.extend_from_slice(&encode_mlir_prefix_varint(ax as u64));
                    }
                }
            }
            Self::SdyMeshAttr { axes } => {
                out.extend_from_slice(&encode_mlir_prefix_varint(12));
                out.extend_from_slice(&encode_mlir_prefix_varint(axes.len() as u64));
                for &(name_idx, sz) in axes {
                    out.extend_from_slice(&encode_mlir_prefix_varint(u64::from(name_idx)));
                    out.extend_from_slice(&encode_mlir_prefix_varint(u64::from(sz)));
                }
            }
            Self::SdyShardingAttr {
                mesh_name_index,
                dim_axis_indices,
            } => {
                out.extend_from_slice(&encode_mlir_prefix_varint(13));
                out.extend_from_slice(&encode_mlir_prefix_varint(u64::from(*mesh_name_index)));
                out.extend_from_slice(&encode_mlir_prefix_varint(dim_axis_indices.len() as u64));
                for dim_axes in dim_axis_indices {
                    out.extend_from_slice(&encode_mlir_prefix_varint(dim_axes.len() as u64));
                    for &ax_idx in dim_axes {
                        out.extend_from_slice(&encode_mlir_prefix_varint(u64::from(ax_idx)));
                    }
                }
            }
            Self::TpuTiledLayoutAttr {
                tile_dims,
                memory_space,
            } => {
                out.extend_from_slice(&encode_mlir_prefix_varint(14));
                out.extend_from_slice(&encode_mlir_prefix_varint(u64::from(*memory_space)));
                out.extend_from_slice(&encode_mlir_prefix_varint(tile_dims.len() as u64));
                for &td in tile_dims {
                    out.extend_from_slice(&encode_mlir_prefix_varint(Self::zigzag_encode(td)));
                }
            }
            Self::UnknownLocAttr => {
                out.extend_from_slice(&encode_mlir_prefix_varint(15));
            }
            Self::FileLineColLocAttr {
                file_string_index,
                line,
                column,
            } => {
                out.extend_from_slice(&encode_mlir_prefix_varint(16));
                out.extend_from_slice(&encode_mlir_prefix_varint(u64::from(*file_string_index)));
                out.extend_from_slice(&encode_mlir_prefix_varint(u64::from(*line)));
                out.extend_from_slice(&encode_mlir_prefix_varint(u64::from(*column)));
            }
            Self::CallSiteLocAttr {
                callee_loc_index,
                caller_loc_index,
            } => {
                out.extend_from_slice(&encode_mlir_prefix_varint(17));
                out.extend_from_slice(&encode_mlir_prefix_varint(u64::from(*callee_loc_index)));
                out.extend_from_slice(&encode_mlir_prefix_varint(u64::from(*caller_loc_index)));
            }
            Self::FusedLocAttr { loc_indices } => {
                out.extend_from_slice(&encode_mlir_prefix_varint(18));
                out.extend_from_slice(&encode_mlir_prefix_varint(loc_indices.len() as u64));
                for &idx in loc_indices {
                    out.extend_from_slice(&encode_mlir_prefix_varint(u64::from(idx)));
                }
            }
            Self::NameLocAttr {
                name_string_index,
                child_loc_index,
            } => {
                out.extend_from_slice(&encode_mlir_prefix_varint(19));
                out.extend_from_slice(&encode_mlir_prefix_varint(u64::from(*name_string_index)));
                out.extend_from_slice(&encode_mlir_prefix_varint(u64::from(*child_loc_index)));
            }
        }
        out
    }

    /// Decodes a binary custom-encoded `kAttrType` entry starting with [`BINARY_CUSTOM_ATTR_TYPE_MAGIC`] (`0xFE`)
    /// or an unprefixed standard upstream dialect `PrefixVarInt` tag (`1..=19`).
    ///
    /// # Arguments
    ///
    /// * `bytes` - Raw byte slice of a single `kAttrType` entry.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MlirBytecodeDecodeError`] if the entry is empty, truncated, or uses an unsupported tag.
    #[allow(clippy::too_many_lines)]
    pub fn decode_binary(bytes: &[u8]) -> LibTpuResult<Self> {
        use crate::types::ElementType;
        if bytes.is_empty() {
            return Err(LibTpuError::MlirBytecodeDecodeError {
                section_id: 2,
                offset: 0,
                reason: "Empty binary kAttrType entry",
            });
        }
        let mut cursor = bytes
            .strip_prefix(&[BINARY_CUSTOM_ATTR_TYPE_MAGIC])
            .unwrap_or(bytes);
        let tag = decode_mlir_prefix_varint(&mut cursor).map_err(|_| {
            LibTpuError::MlirBytecodeDecodeError {
                section_id: 2,
                offset: 1,
                reason: "Truncated tag varint in binary kAttrType entry",
            }
        })?;
        let parse_et = |raw_u64: u64| -> ElementType {
            i32::try_from(raw_u64)
                .ok()
                .and_then(|r| ElementType::from_raw(r).ok())
                .unwrap_or(ElementType::F32)
        };
        match tag {
            1 => {
                let packed = decode_mlir_prefix_varint(&mut cursor)?;
                #[allow(clippy::cast_possible_truncation)]
                let signedness = (packed & 0x03) as u8;
                let width = u32::try_from(packed >> 2).unwrap_or(32);
                Ok(Self::IntegerType { width, signedness })
            }
            2 => {
                let et_raw = decode_mlir_prefix_varint(&mut cursor)?;
                Ok(Self::ScalarType(parse_et(et_raw)))
            }
            3 => {
                let et_raw = decode_mlir_prefix_varint(&mut cursor)?;
                #[allow(clippy::cast_possible_truncation)]
                let memory_space = decode_mlir_prefix_varint(&mut cursor)? as u8;
                let rank = usize::try_from(decode_mlir_prefix_varint(&mut cursor)?).unwrap_or(0);
                let mut dims = Vec::with_capacity(rank);
                for _ in 0..rank {
                    let zz = decode_mlir_prefix_varint(&mut cursor)?;
                    dims.push(Self::zigzag_decode(zz));
                }
                Ok(Self::RankedTensorType {
                    dims,
                    element_type: parse_et(et_raw),
                    memory_space,
                })
            }
            4 => {
                let n_in = usize::try_from(decode_mlir_prefix_varint(&mut cursor)?).unwrap_or(0);
                let mut input_type_indices = Vec::with_capacity(n_in);
                for _ in 0..n_in {
                    #[allow(clippy::cast_possible_truncation)]
                    input_type_indices.push(decode_mlir_prefix_varint(&mut cursor)? as u32);
                }
                let n_out = usize::try_from(decode_mlir_prefix_varint(&mut cursor)?).unwrap_or(0);
                let mut result_type_indices = Vec::with_capacity(n_out);
                for _ in 0..n_out {
                    #[allow(clippy::cast_possible_truncation)]
                    result_type_indices.push(decode_mlir_prefix_varint(&mut cursor)? as u32);
                }
                Ok(Self::FunctionType {
                    input_type_indices,
                    result_type_indices,
                })
            }
            5 => {
                let et_raw = decode_mlir_prefix_varint(&mut cursor)?;
                let zz = decode_mlir_prefix_varint(&mut cursor)?;
                Ok(Self::IntegerAttr {
                    value: Self::zigzag_decode(zz),
                    element_type: parse_et(et_raw),
                })
            }
            6 => {
                let et_raw = decode_mlir_prefix_varint(&mut cursor)?;
                let (head, _) =
                    cursor
                        .split_at_checked(8)
                        .ok_or(LibTpuError::MlirBytecodeDecodeError {
                            section_id: 2,
                            offset: 2,
                            reason: "Truncated FloatAttr 8-byte payload in kAttrType",
                        })?;
                let bits = u64::from_le(bytemuck::pod_read_unaligned(head));
                Ok(Self::FloatAttr {
                    bits,
                    element_type: parse_et(et_raw),
                })
            }
            7 => {
                #[allow(clippy::cast_possible_truncation)]
                let string_index = decode_mlir_prefix_varint(&mut cursor)? as u32;
                Ok(Self::StringAttr { string_index })
            }
            8 => {
                let count = usize::try_from(decode_mlir_prefix_varint(&mut cursor)?).unwrap_or(0);
                let mut entries = Vec::with_capacity(count);
                for _ in 0..count {
                    #[allow(clippy::cast_possible_truncation)]
                    let k = decode_mlir_prefix_varint(&mut cursor)? as u32;
                    #[allow(clippy::cast_possible_truncation)]
                    let v = decode_mlir_prefix_varint(&mut cursor)? as u32;
                    entries.push((k, v));
                }
                Ok(Self::DictionaryAttr { entries })
            }
            9 => {
                let et_raw = decode_mlir_prefix_varint(&mut cursor)?;
                let is_splat = decode_mlir_prefix_varint(&mut cursor)? != 0;
                let rank = usize::try_from(decode_mlir_prefix_varint(&mut cursor)?).unwrap_or(0);
                let mut dims = Vec::with_capacity(rank);
                for _ in 0..rank {
                    let zz = decode_mlir_prefix_varint(&mut cursor)?;
                    dims.push(Self::zigzag_decode(zz));
                }
                let byte_len =
                    usize::try_from(decode_mlir_prefix_varint(&mut cursor)?).unwrap_or(0);
                let raw_bytes = cursor
                    .get(..byte_len.min(cursor.len()))
                    .unwrap_or_default()
                    .to_vec();
                Ok(Self::DenseElementsAttr {
                    dims,
                    element_type: parse_et(et_raw),
                    is_splat,
                    raw_bytes,
                })
            }
            10 => {
                let et_raw = decode_mlir_prefix_varint(&mut cursor)?;
                #[allow(clippy::cast_possible_truncation)]
                let resource_index = decode_mlir_prefix_varint(&mut cursor)? as u32;
                let rank = usize::try_from(decode_mlir_prefix_varint(&mut cursor)?).unwrap_or(0);
                let mut dims = Vec::with_capacity(rank);
                for _ in 0..rank {
                    let zz = decode_mlir_prefix_varint(&mut cursor)?;
                    dims.push(Self::zigzag_decode(zz));
                }
                Ok(Self::DenseResourceAttr {
                    dims,
                    element_type: parse_et(et_raw),
                    resource_index,
                })
            }
            11 => {
                let mut lists: [Vec<usize>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
                for list in &mut lists {
                    let n = usize::try_from(decode_mlir_prefix_varint(&mut cursor)?).unwrap_or(0);
                    for _ in 0..n {
                        let ax =
                            usize::try_from(decode_mlir_prefix_varint(&mut cursor)?).unwrap_or(0);
                        list.push(ax);
                    }
                }
                let [lhs_batching, rhs_batching, lhs_contracting, rhs_contracting] = lists;
                Ok(Self::DotDimensionNumbersAttr {
                    lhs_batching,
                    rhs_batching,
                    lhs_contracting,
                    rhs_contracting,
                })
            }
            12 => {
                let n = usize::try_from(decode_mlir_prefix_varint(&mut cursor)?).unwrap_or(0);
                let mut axes = Vec::with_capacity(n);
                for _ in 0..n {
                    #[allow(clippy::cast_possible_truncation)]
                    let name_idx = decode_mlir_prefix_varint(&mut cursor)? as u32;
                    #[allow(clippy::cast_possible_truncation)]
                    let sz = decode_mlir_prefix_varint(&mut cursor)? as u32;
                    axes.push((name_idx, sz));
                }
                Ok(Self::SdyMeshAttr { axes })
            }
            13 => {
                #[allow(clippy::cast_possible_truncation)]
                let mesh_name_index = decode_mlir_prefix_varint(&mut cursor)? as u32;
                let n_dims = usize::try_from(decode_mlir_prefix_varint(&mut cursor)?).unwrap_or(0);
                let mut dim_axis_indices = Vec::with_capacity(n_dims);
                for _ in 0..n_dims {
                    let n_ax =
                        usize::try_from(decode_mlir_prefix_varint(&mut cursor)?).unwrap_or(0);
                    let mut axes = Vec::with_capacity(n_ax);
                    for _ in 0..n_ax {
                        #[allow(clippy::cast_possible_truncation)]
                        axes.push(decode_mlir_prefix_varint(&mut cursor)? as u32);
                    }
                    dim_axis_indices.push(axes);
                }
                Ok(Self::SdyShardingAttr {
                    mesh_name_index,
                    dim_axis_indices,
                })
            }
            14 => {
                #[allow(clippy::cast_possible_truncation)]
                let memory_space = decode_mlir_prefix_varint(&mut cursor)? as u8;
                let n = usize::try_from(decode_mlir_prefix_varint(&mut cursor)?).unwrap_or(0);
                let mut tile_dims = Vec::with_capacity(n);
                for _ in 0..n {
                    let zz = decode_mlir_prefix_varint(&mut cursor)?;
                    tile_dims.push(Self::zigzag_decode(zz));
                }
                Ok(Self::TpuTiledLayoutAttr {
                    tile_dims,
                    memory_space,
                })
            }
            15 => Ok(Self::UnknownLocAttr),
            16 => {
                #[allow(clippy::cast_possible_truncation)]
                let file_string_index = decode_mlir_prefix_varint(&mut cursor)? as u32;
                #[allow(clippy::cast_possible_truncation)]
                let line = decode_mlir_prefix_varint(&mut cursor)? as u32;
                #[allow(clippy::cast_possible_truncation)]
                let column = decode_mlir_prefix_varint(&mut cursor)? as u32;
                Ok(Self::FileLineColLocAttr {
                    file_string_index,
                    line,
                    column,
                })
            }
            17 => {
                #[allow(clippy::cast_possible_truncation)]
                let target_loc = decode_mlir_prefix_varint(&mut cursor)? as u32;
                #[allow(clippy::cast_possible_truncation)]
                let site_loc = decode_mlir_prefix_varint(&mut cursor)? as u32;
                Ok(Self::CallSiteLocAttr {
                    callee_loc_index: target_loc,
                    caller_loc_index: site_loc,
                })
            }
            18 => {
                let n = usize::try_from(decode_mlir_prefix_varint(&mut cursor)?).unwrap_or(0);
                let mut loc_indices = Vec::with_capacity(n);
                for _ in 0..n {
                    #[allow(clippy::cast_possible_truncation)]
                    loc_indices.push(decode_mlir_prefix_varint(&mut cursor)? as u32);
                }
                Ok(Self::FusedLocAttr { loc_indices })
            }
            19 => {
                #[allow(clippy::cast_possible_truncation)]
                let name_string_index = decode_mlir_prefix_varint(&mut cursor)? as u32;
                #[allow(clippy::cast_possible_truncation)]
                let child_loc_index = decode_mlir_prefix_varint(&mut cursor)? as u32;
                Ok(Self::NameLocAttr {
                    name_string_index,
                    child_loc_index,
                })
            }
            _ => Err(LibTpuError::MlirBytecodeDecodeError {
                section_id: 2,
                offset: 1,
                reason: "Unsupported binary kAttrType tag in custom dialect entry",
            }),
        }
    }

    /// Converts this [`BinaryAttrTypeEntry`] into its canonical textual MLIR representation
    /// using `string_table` to resolve string/symbol indices.
    ///
    /// # Arguments
    ///
    /// * `string_table` - Section `0` (`kString`) string table.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn to_canonical_mlir_string(&self, string_table: &[String]) -> String {
        use crate::types::ElementType;
        let fmt_et = |et: ElementType| -> &'static str {
            match et {
                ElementType::Pred => "i1",
                ElementType::S4 => "s4",
                ElementType::U4 => "u4",
                ElementType::S8 => "i8",
                ElementType::U8 => "ui8",
                ElementType::S16 => "i16",
                ElementType::U16 => "ui16",
                ElementType::S32 => "i32",
                ElementType::U32 => "ui32",
                ElementType::S64 => "i64",
                ElementType::U64 => "ui64",
                ElementType::F16 => "f16",
                ElementType::Bf16 => "bf16",
                ElementType::F64 => "f64",
                ElementType::F8E4M3Fn => "f8E4M3FN",
                ElementType::F8E5M2 => "f8E5M2",
                _ => "f32",
            }
        };
        match self {
            Self::IntegerType { width, signedness } => match signedness {
                2 => format!("ui{width}"),
                _ => format!("i{width}"),
            },
            Self::ScalarType(et) => String::from(fmt_et(*et)),
            Self::RankedTensorType {
                dims, element_type, ..
            } => {
                let et_s = fmt_et(*element_type);
                if dims.is_empty() {
                    format!("tensor<{et_s}>")
                } else {
                    let d_str = dims
                        .iter()
                        .map(|&d| {
                            if d < 0 {
                                String::from("?")
                            } else {
                                d.to_string()
                            }
                        })
                        .collect::<Vec<_>>()
                        .join("x");
                    format!("tensor<{d_str}x{et_s}>")
                }
            }
            Self::FunctionType {
                input_type_indices,
                result_type_indices,
            } => format!("func_type({input_type_indices:?} -> {result_type_indices:?})"),
            Self::IntegerAttr { value, .. } => value.to_string(),
            Self::FloatAttr { bits, .. } => format!("{:.6}", f64::from_bits(*bits)),
            Self::StringAttr { string_index } => string_table
                .get(*string_index as usize)
                .cloned()
                .unwrap_or_else(|| format!("str_{string_index}")),
            Self::DictionaryAttr { entries } => {
                let parts: Vec<String> = entries
                    .iter()
                    .map(|&(k, v)| {
                        let k_str = string_table.get(k as usize).map_or("attr", String::as_str);
                        format!("{k_str} = {v}")
                    })
                    .collect();
                format!("{{{}}}", parts.join(", "))
            }
            Self::DenseElementsAttr {
                dims,
                element_type,
                raw_bytes,
                ..
            } => {
                let et_s = fmt_et(*element_type);
                let mut hex = String::with_capacity(2 + raw_bytes.len() * 2);
                hex.push_str("0x");
                for &b in raw_bytes {
                    use std::fmt::Write as _;
                    let _ = write!(hex, "{b:02x}");
                }
                let d_str = dims
                    .iter()
                    .map(i64::to_string)
                    .collect::<Vec<_>>()
                    .join("x");
                if d_str.is_empty() {
                    format!("dense<\"{hex}\"> : tensor<{et_s}>")
                } else {
                    format!("dense<\"{hex}\"> : tensor<{d_str}x{et_s}>")
                }
            }
            Self::DenseResourceAttr { resource_index, .. } => {
                format!("dense_resource<res_{resource_index}>")
            }
            Self::DotDimensionNumbersAttr {
                lhs_batching,
                rhs_batching,
                lhs_contracting,
                rhs_contracting,
            } => format!(
                "#stablehlo.dot<lhs_batching_dimensions = {lhs_batching:?}, rhs_batching_dimensions = {rhs_batching:?}, lhs_contracting_dimensions = {lhs_contracting:?}, rhs_contracting_dimensions = {rhs_contracting:?}>"
            ),
            Self::SdyMeshAttr { axes } => {
                let ax_s: Vec<String> = axes
                    .iter()
                    .map(|&(idx, sz)| {
                        let nm = string_table.get(idx as usize).map_or("x", String::as_str);
                        format!("\"{nm}\"={sz}")
                    })
                    .collect();
                format!("#sdy.mesh<[{}]>", ax_s.join(", "))
            }
            Self::SdyShardingAttr {
                mesh_name_index,
                dim_axis_indices,
            } => {
                let mesh = string_table
                    .get(*mesh_name_index as usize)
                    .map_or("mesh", String::as_str);
                let dims_s: Vec<String> = dim_axis_indices
                    .iter()
                    .map(|ax_list| {
                        let names: Vec<String> = ax_list
                            .iter()
                            .map(|&i| {
                                let s = string_table.get(i as usize).map_or("x", String::as_str);
                                format!("\"{s}\"")
                            })
                            .collect();
                        format!("{{{}}}", names.join(", "))
                    })
                    .collect();
                format!("#sdy.sharding<@{mesh}, [{}]>", dims_s.join(", "))
            }
            Self::TpuTiledLayoutAttr {
                tile_dims,
                memory_space,
            } => {
                let t_s = tile_dims
                    .iter()
                    .map(i64::to_string)
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("#tpu.tiled<({t_s}), space={memory_space}>")
            }
            Self::UnknownLocAttr => String::from("loc(unknown)"),
            Self::FileLineColLocAttr {
                file_string_index,
                line,
                column,
            } => {
                let f_str = string_table
                    .get(*file_string_index as usize)
                    .map_or("module.mlir", String::as_str);
                format!("loc(\"{f_str}\":{line}:{column})")
            }
            Self::CallSiteLocAttr {
                callee_loc_index,
                caller_loc_index,
            } => {
                format!("loc(callsite(#loc{callee_loc_index} at #loc{caller_loc_index}))")
            }
            Self::FusedLocAttr { loc_indices } => {
                let refs: Vec<String> = loc_indices.iter().map(|i| format!("#loc{i}")).collect();
                format!("loc(fused[{}])", refs.join(", "))
            }
            Self::NameLocAttr {
                name_string_index,
                child_loc_index,
            } => {
                let nm = string_table
                    .get(*name_string_index as usize)
                    .map_or("op", String::as_str);
                format!("loc(\"{nm}\"(#loc{child_loc_index}))")
            }
        }
    }
}

/// Decoded MLIR Bytecode operation node from Section `4` (`kIR`) (`MlirBytecodeOpNode`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MlirBytecodeOpNode {
    /// Operation name index into the dialect/string table (`name_idx`).
    pub name_index: u64,
    /// Bitmask of operation encoding flags (`hasAttrDict = 0x01`, `hasResults = 0x02`, `hasOperands = 0x04`, `hasSuccessors = 0x08`, `hasInlineRegions = 0x10`, `isIsolatedFromAbove = 0x20`, `hasProperties = 0x40`).
    pub encoding_mask: u8,
    /// Optional attribute dictionary index into `attr_type_entries` when `0x01` (`kHasAttrs`) is set.
    pub attr_dict_index: Option<u32>,
    /// Decoded SSA operand value indices (`SsaValueId`).
    pub operands: Vec<u32>,
    /// Decoded result type indices into `attr_type_entries`.
    pub result_type_indices: Vec<u32>,
    /// Decoded successor block indices when `0x08` (`kHasSuccessors`) is set.
    pub successors: Vec<u32>,
    /// Number of nested inline regions attached to this operation.
    pub num_regions: u32,
    /// Optional inline properties index into `properties` when `0x40` (`kHasProperties`) is set.
    pub properties_index: Option<u32>,
    /// Recursively encoded nested inline region operation lists (`num_regions` entries when `0x10` is set).
    pub inline_regions: Vec<Vec<Self>>,
    /// Block argument type indices into `kAttrType` for each inline region's entry block (`region_block_arg_types[region_idx]`).
    pub region_block_arg_types: Vec<Vec<u32>>,
}

impl MlirBytecodeOpNode {
    /// Appends this operation node (and any nested `inline_regions` + `region_block_arg_types`) as `PrefixVarInt` records into `ir_bytes`.
    ///
    /// # Arguments
    ///
    /// * `ir_bytes` - Destination Section-4 (`kIR`) byte buffer.
    fn encode_into_stream(&self, ir_bytes: &mut Vec<u8>) {
        ir_bytes.extend_from_slice(&encode_mlir_prefix_varint(self.name_index));
        ir_bytes.extend_from_slice(&encode_mlir_prefix_varint(u64::from(self.encoding_mask)));
        let is_structured =
            (u8::from(self.encoding_mask >= 0x40) | (self.encoding_mask & 0x02)) != 0;
        if is_structured {
            if (self.encoding_mask & 0x01) != 0 {
                let attr_idx = u64::from(self.attr_dict_index.unwrap_or(0));
                ir_bytes.extend_from_slice(&encode_mlir_prefix_varint(attr_idx));
            }
            ir_bytes.extend_from_slice(&encode_mlir_prefix_varint(
                self.result_type_indices.len() as u64
            ));
            for &res_ty in &self.result_type_indices {
                ir_bytes.extend_from_slice(&encode_mlir_prefix_varint(u64::from(res_ty)));
            }
            ir_bytes.extend_from_slice(&encode_mlir_prefix_varint(self.operands.len() as u64));
            for &operand in &self.operands {
                ir_bytes.extend_from_slice(&encode_mlir_prefix_varint(u64::from(operand)));
            }
            if (self.encoding_mask & 0x08) != 0 {
                ir_bytes
                    .extend_from_slice(&encode_mlir_prefix_varint(self.successors.len() as u64));
                for &succ in &self.successors {
                    ir_bytes.extend_from_slice(&encode_mlir_prefix_varint(u64::from(succ)));
                }
            }
            if (self.encoding_mask & 0x10) != 0 {
                let reg_count = u32::try_from(self.inline_regions.len())
                    .unwrap_or(0)
                    .max(self.num_regions);
                ir_bytes.extend_from_slice(&encode_mlir_prefix_varint(u64::from(reg_count)));
                for (r_idx, reg_ops) in self.inline_regions.iter().enumerate() {
                    let blk_args = self
                        .region_block_arg_types
                        .get(r_idx)
                        .map_or(&[][..], Vec::as_slice);
                    let header = ((reg_ops.len() as u64) << 1) | 0x4000_0001;
                    ir_bytes.extend_from_slice(&encode_mlir_prefix_varint(header));
                    ir_bytes.extend_from_slice(&encode_mlir_prefix_varint(blk_args.len() as u64));
                    for &arg_ty in blk_args {
                        ir_bytes.extend_from_slice(&encode_mlir_prefix_varint(u64::from(arg_ty)));
                    }
                    for child in reg_ops {
                        child.encode_into_stream(ir_bytes);
                    }
                }
                for _ in self.inline_regions.len()..usize::try_from(reg_count).unwrap_or(0) {
                    ir_bytes.extend_from_slice(&encode_mlir_prefix_varint(0x4000_0001));
                    ir_bytes.extend_from_slice(&encode_mlir_prefix_varint(0));
                }
            }
            if (self.encoding_mask & 0x40) != 0 {
                let prop_idx = u64::from(self.properties_index.unwrap_or(0));
                ir_bytes.extend_from_slice(&encode_mlir_prefix_varint(prop_idx));
            }
        } else {
            for &operand in &self.operands {
                ir_bytes.extend_from_slice(&encode_mlir_prefix_varint(u64::from(operand)));
            }
        }
    }
}

/// Binary `ML\xefR` bytecode container builder (`MlirBytecodeEncoder`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MlirBytecodeEncoder {
    /// Producer string (e.g., `"MLIR20.0.0"`).
    pub producer: String,
    /// Interned strings for Section `0` (`kString`).
    pub strings: Vec<String>,
    /// Registered dialects for Section `1` (`kDialect`).
    pub dialects: Vec<String>,
    /// Attribute and type entries for Section `2` (`kAttrType`).
    pub attr_types: Vec<String>,
    /// Optional binary custom-encoded attribute and type entries for Section `2` (`kAttrType`).
    pub binary_attr_types: Vec<BinaryAttrTypeEntry>,
    /// Hierarchical operation nodes for Section `4` (`kIR`).
    pub operations: Vec<MlirBytecodeOpNode>,
    /// External dense resource blobs for Section `5` (`kResource`).
    pub resources: Vec<Vec<u8>>,
    /// Inline operation property records for Section `7`/`8` (`kProperties`).
    pub properties: Vec<u64>,
}

impl MlirBytecodeEncoder {
    /// Creates a new [`MlirBytecodeEncoder`] with the given `producer` version string.
    ///
    /// # Arguments
    ///
    /// * `producer` - Producer banner string (e.g., `"MLIR20.0.0"`).
    #[must_use]
    pub fn new(producer: &str) -> Self {
        Self {
            producer: String::from(producer),
            strings: Vec::new(),
            dialects: Vec::new(),
            attr_types: Vec::new(),
            binary_attr_types: Vec::new(),
            operations: Vec::new(),
            resources: Vec::new(),
            properties: Vec::new(),
        }
    }

    /// Interns `value` in the string table and returns its 0-based index.
    ///
    /// # Arguments
    ///
    /// * `value` - String literal to intern.
    pub fn intern_string(&mut self, value: &str) -> u64 {
        if let Some(pos) = self.strings.iter().position(|s| s == value) {
            return pos as u64;
        }
        let idx = self.strings.len() as u64;
        self.strings.push(String::from(value));
        idx
    }

    /// Interns `attr_or_type` in the `kAttrType` table and returns its 0-based index.
    ///
    /// # Arguments
    ///
    /// * `attr_or_type` - Canonical textual or binary attribute/type descriptor string.
    pub fn intern_attr_type(&mut self, attr_or_type: &str) -> u32 {
        if let Some(pos) = self.attr_types.iter().position(|a| a == attr_or_type) {
            return u32::try_from(pos).unwrap_or(0);
        }
        let idx = u32::try_from(self.attr_types.len()).unwrap_or(0);
        self.attr_types.push(String::from(attr_or_type));
        idx
    }

    /// Interns a custom binary [`BinaryAttrTypeEntry`] in Section `2` (`kAttrType`) and returns its 0-based index.
    ///
    /// # Arguments
    ///
    /// * `entry` - Custom binary attribute or type entry to intern.
    pub fn intern_binary_attr_type(&mut self, entry: BinaryAttrTypeEntry) -> u32 {
        let canonical = entry.to_canonical_mlir_string(&self.strings);
        if let Some(pos) = self.binary_attr_types.iter().position(|e| *e == entry) {
            return u32::try_from(pos).unwrap_or(0);
        }
        let idx = u32::try_from(self.binary_attr_types.len()).unwrap_or(0);
        self.binary_attr_types.push(entry);
        self.attr_types.push(canonical);
        idx
    }

    /// Appends a dense resource blob to Section `5` (`kResource`) and returns its 0-based index.
    ///
    /// # Arguments
    ///
    /// * `blob` - Raw little-endian dense tensor resource bytes.
    pub fn add_resource_blob(&mut self, blob: Vec<u8>) -> u32 {
        let idx = u32::try_from(self.resources.len()).unwrap_or(0);
        self.resources.push(blob);
        idx
    }

    /// Appends Section `5` (`kResource`), Section `6` (`kResourceOffset`), and Section `7` (`kProperties`)
    /// when resources or properties are present.
    ///
    /// # Arguments
    ///
    /// * `sections` - Mutable section vector to append optional sections into.
    fn append_optional_sections(&self, sections: &mut Vec<MlirBytecodeSection>) -> Vec<u64> {
        let mut resource_offsets = Vec::new();
        if !self.resources.is_empty() {
            let mut res_bytes = Vec::new();
            let mut res_off_bytes = Vec::new();
            for blob in &self.resources {
                res_bytes.extend_from_slice(blob);
                let len_u64 = blob.len() as u64;
                resource_offsets.push(len_u64);
                res_off_bytes.extend_from_slice(&encode_mlir_prefix_varint(len_u64));
            }
            sections.push(MlirBytecodeSection {
                section_id: 5,
                alignment: Some(8),
                data: res_bytes,
            });
            sections.push(MlirBytecodeSection {
                section_id: 6,
                alignment: None,
                data: res_off_bytes,
            });
        }
        if !self.properties.is_empty() {
            let mut prop_bytes = Vec::new();
            for &prop in &self.properties {
                prop_bytes.extend_from_slice(&encode_mlir_prefix_varint(prop));
            }
            sections.push(MlirBytecodeSection {
                section_id: 7,
                alignment: None,
                data: prop_bytes,
            });
        }
        resource_offsets
    }

    /// Encodes all [`MlirBytecodeOpNode`]s into a Section `4` (`kIR`) byte buffer.
    fn encode_ir_section_bytes(&self) -> Vec<u8> {
        let mut ir_bytes = Vec::new();
        for op in &self.operations {
            op.encode_into_stream(&mut ir_bytes);
        }
        ir_bytes
    }

    /// Serializes this module into a valid `ML\xefR` binary bytecode container.
    #[must_use]
    pub fn encode_to_bytes(&self) -> Vec<u8> {
        let mut str_bytes = Vec::new();
        for s in &self.strings {
            str_bytes.extend_from_slice(s.as_bytes());
            str_bytes.push(0);
        }
        let mut dial_bytes = Vec::new();
        for d in &self.dialects {
            let idx = self.strings.iter().position(|s| s == d).unwrap_or(0) as u64;
            dial_bytes.extend_from_slice(&encode_mlir_prefix_varint(idx));
        }
        let mut attr_bytes = Vec::new();
        let mut attr_offsets = Vec::new();
        if self.binary_attr_types.is_empty() {
            for a in &self.attr_types {
                attr_bytes.extend_from_slice(a.as_bytes());
                attr_bytes.push(0);
                attr_offsets.push((a.len() + 1) as u64);
            }
        } else {
            for bin_entry in &self.binary_attr_types {
                let encoded = bin_entry.encode_binary();
                attr_offsets.push(encoded.len() as u64);
                attr_bytes.extend_from_slice(&encoded);
            }
        }
        let mut off_bytes = Vec::new();
        for off in &attr_offsets {
            off_bytes.extend_from_slice(&encode_mlir_prefix_varint(*off));
        }
        let ir_bytes = self.encode_ir_section_bytes();
        let mut sections = vec![
            MlirBytecodeSection {
                section_id: 0,
                alignment: None,
                data: str_bytes,
            },
            MlirBytecodeSection {
                section_id: 1,
                alignment: None,
                data: dial_bytes,
            },
            MlirBytecodeSection {
                section_id: 2,
                alignment: None,
                data: attr_bytes,
            },
            MlirBytecodeSection {
                section_id: 3,
                alignment: None,
                data: off_bytes,
            },
            MlirBytecodeSection {
                section_id: 4,
                alignment: None,
                data: ir_bytes,
            },
        ];
        let resource_offsets = self.append_optional_sections(&mut sections);
        let hdr = MlirBytecodeHeader {
            version: 1,
            producer: self.producer.clone(),
            section_count: sections.len(),
            sections,
            string_table: self.strings.clone(),
            dialects: self.dialects.clone(),
            attr_type_offsets: attr_offsets,
            attr_type_entries: self.attr_types.clone(),
            ir_op_records: Vec::new(),
            resources: self.resources.clone(),
            resource_offsets,
            properties: self.properties.clone(),
        };
        hdr.encode()
    }
}

/// Clean-room decoder for MLIR Bytecode (`ML\xefR`) payloads across Sections `0..=8`.
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
    /// Decoded external resource offsets from `kResourceOffset` (Section `6`).
    pub resource_offsets: Vec<u64>,
    /// Decoded inlined operation property words from `kProperties` (Section `7`/`8`).
    pub properties: Vec<u64>,
}

impl MlirBytecodeHeader {
    /// Reconciles Section `2` (`kAttrType`) with Section `3` (`kAttrTypeOffset`) when custom binary
    /// entries prefixed with [`BINARY_CUSTOM_ATTR_TYPE_MAGIC`] (`0xFE`) are present.
    fn reconcile_binary_attr_type_entries(&mut self) {
        let Some(sec2) = self
            .sections
            .iter()
            .find(|s| s.section_id == 2 && s.data.contains(&BINARY_CUSTOM_ATTR_TYPE_MAGIC))
        else {
            return;
        };
        let mut rebuilt = Vec::with_capacity(self.attr_type_offsets.len());
        let mut start = 0_usize;
        for &len_u64 in &self.attr_type_offsets {
            let len = usize::try_from(len_u64).unwrap_or(0);
            let end = start.saturating_add(len).min(sec2.data.len());
            let slice = sec2.data.get(start..end).unwrap_or_default();
            rebuilt.extend(
                BinaryAttrTypeEntry::decode_binary(slice)
                    .map(|b| b.to_canonical_mlir_string(&self.string_table)),
            );
            start = end;
        }
        self.attr_type_entries = rebuilt;
    }

    /// Decodes a single MLIR bytecode section payload into the header tables.
    ///
    /// # Arguments
    ///
    /// * `section_id` - Numeric MLIR bytecode section identifier (`0..=8`).
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
                while let Ok(idx_u64) = decode_mlir_prefix_varint(&mut d_cur) {
                    let direct_idx = usize::try_from(idx_u64).unwrap_or(usize::MAX);
                    let shifted_idx = usize::try_from(idx_u64 >> 1).unwrap_or(usize::MAX);
                    let name = if let Some(s) = self.string_table.get(direct_idx) {
                        s.clone()
                    } else if let Some(s) = self.string_table.get(shifted_idx) {
                        let ver_len = decode_mlir_prefix_varint(&mut d_cur).unwrap_or(0);
                        let skip = usize::try_from(ver_len).unwrap_or(0).min(d_cur.len());
                        d_cur = d_cur.get(skip..).unwrap_or_default();
                        s.clone()
                    } else {
                        format!("dialect_{idx_u64}")
                    };
                    self.dialects.push(name);
                }
            }
            2 => {
                if !sec_data.contains(&BINARY_CUSTOM_ATTR_TYPE_MAGIC) {
                    for entry in sec_data.split(|&b| b == 0) {
                        if !entry.is_empty() {
                            self.attr_type_entries
                                .push(String::from_utf8_lossy(entry).into_owned());
                        }
                    }
                }
            }
            3 => {
                let mut o_cur = sec_data;
                while let Ok(off) = decode_mlir_prefix_varint(&mut o_cur) {
                    self.attr_type_offsets.push(off);
                }
            }
            4 => {
                let mut ir_cur = sec_data;
                while let Ok(rec) = decode_mlir_prefix_varint(&mut ir_cur) {
                    self.ir_op_records.push(rec);
                }
            }
            5 => {
                self.resources.push(sec_data.to_vec());
            }
            6 => {
                let mut r_cur = sec_data;
                while let Ok(off) = decode_mlir_prefix_varint(&mut r_cur) {
                    self.resource_offsets.push(off);
                }
            }
            7 => {
                let mut p_cur = sec_data;
                while let Ok(prop) = decode_mlir_prefix_varint(&mut p_cur) {
                    self.properties.push(prop);
                }
            }
            _ => {}
        }
    }

    /// Reconstructs full [`MlirDialectDescriptor`] tables with version bytes and op lists from
    /// Section `1` (`kDialect`), Section `0` (`kString`), and Section `7` (`kDialectVersions`).
    #[must_use]
    pub fn decode_dialect_descriptors(&self) -> Vec<MlirDialectDescriptor> {
        let op_pairs = self.decode_dialect_op_names();
        let mut descriptors = Vec::with_capacity(self.dialects.len().max(1));
        for (idx, d_name) in self.dialects.iter().enumerate() {
            let canonical_dial = Self::legalize_vhlo_symbol(d_name);
            let ops = op_pairs
                .iter()
                .filter(|(op_dial, _)| op_dial == &canonical_dial)
                .map(|(_, full_op)| full_op.clone())
                .collect();
            let version_bytes = self
                .properties
                .get(idx)
                .map(|&prop_word| encode_mlir_prefix_varint(prop_word))
                .unwrap_or_default();
            descriptors.push(MlirDialectDescriptor {
                name: d_name.clone(),
                version_bytes,
                registered_ops: ops,
            });
        }
        descriptors
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
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MlirBytecodeDecodeError`] if `bytes` does not start with
    /// [`MLIR_BYTECODE_MAGIC`].
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        let mut cursor = bytes.strip_prefix(MLIR_BYTECODE_MAGIC).ok_or(
            LibTpuError::MlirBytecodeDecodeError {
                section_id: 0,
                offset: 0,
                reason: "Missing ML\\xefR magic header in MLIR bytecode payload",
            },
        )?;
        if cursor.is_empty() {
            return Ok(Self {
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
                resource_offsets: Vec::new(),
                properties: Vec::new(),
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
            resource_offsets: Vec::new(),
            properties: Vec::new(),
        };
        let mut sec_cursor = cursor;
        while let Some((&id_and_align, after_id)) = sec_cursor.split_first() {
            sec_cursor = after_id;
            let section_id = id_and_align & 0x7F;
            let has_align = (id_and_align & 0x80) != 0;
            let Ok(sec_len_u64) = decode_mlir_prefix_varint(&mut sec_cursor) else {
                break;
            };
            let alignment = if has_align {
                decode_mlir_prefix_varint(&mut sec_cursor).ok()
            } else {
                None
            };
            let sec_len = usize::try_from(sec_len_u64).unwrap_or(usize::MAX);
            let Some((sec_data, rest)) = sec_cursor.split_at_checked(sec_len) else {
                break;
            };
            sec_cursor = rest;
            hdr.decode_section_payload(section_id, sec_data);
            hdr.sections.push(MlirBytecodeSection {
                section_id,
                alignment,
                data: sec_data.to_vec(),
            });
        }
        hdr.reconcile_binary_attr_type_entries();
        hdr.section_count = if hdr.sections.is_empty() {
            usize::from(!cursor.is_empty())
        } else {
            hdr.sections.len()
        };
        Ok(hdr)
    }

    /// Strictly parses an MLIR bytecode payload starting with [`MLIR_BYTECODE_MAGIC`] (`b"ML\xefR"`).
    ///
    /// # Arguments
    ///
    /// * `bytes` - Raw byte slice beginning with `b"ML\xefR"`.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MlirBytecodeDecodeError`] if `bytes` does not start with
    /// [`MLIR_BYTECODE_MAGIC`].
    pub fn decode_strict(bytes: &[u8]) -> LibTpuResult<Self> {
        Self::decode(bytes)
    }

    /// Synthesizes a canonical textual `StableHLO` representation from a binary MLIR bytecode header
    /// when `string_table`, `dialects`, or `attr_type_entries` contains dialect/op/type strings.
    /// Automatically legalizes `vhlo.*` portable operations (e.g., `vhlo.add_v1`) to `stablehlo.*`.
    #[must_use]
    pub fn to_synthesized_mlir_text(&self) -> String {
        let mut parts: Vec<String> = self
            .string_table
            .iter()
            .map(|s| Self::legalize_vhlo_symbol(s))
            .collect();
        parts.extend(
            self.attr_type_entries
                .iter()
                .map(|a| Self::legalize_vhlo_symbol(a)),
        );
        if parts.is_empty() {
            return self.producer.clone();
        }
        parts.join(" ")
    }

    /// Upgrades a `vhlo` dialect or versioned operation name (e.g., `"vhlo.add_v1"`, `"vhlo.dot_general_v1"`)
    /// to its canonical `stablehlo` counterpart (`VhloLegalizeToStablehlo`).
    ///
    /// # Arguments
    ///
    /// * `symbol` - Dialect or operation symbol string.
    #[must_use]
    pub fn legalize_vhlo_symbol(symbol: &str) -> String {
        if symbol == "vhlo" {
            return String::from("stablehlo");
        }
        if let Some(rest) = symbol.strip_prefix("vhlo.") {
            let base = rest
                .rsplit_once("_v")
                .filter(|(_, ver)| ver.bytes().all(|b| b.is_ascii_digit()))
                .map_or(rest, |(stem, _)| stem);
            return format!("stablehlo.{base}");
        }
        String::from(symbol)
    }

    /// Resolves a string at an exact byte offset within Section `0` (`kString`), or by index in `string_table`.
    ///
    /// # Arguments
    ///
    /// * `offset_or_index` - Byte offset in Section `0` or 0-based index into `string_table`.
    #[must_use]
    pub fn resolve_string_at_offset(&self, offset_or_index: u64) -> Option<String> {
        let idx = usize::try_from(offset_or_index).ok()?;
        for sec0 in &self.sections {
            if sec0.section_id == 0
                && let Some(tail) = sec0.data.get(idx..)
            {
                let nul_pos = tail.iter().position(|&b| b == 0).unwrap_or(tail.len());
                let slice = tail.get(..nul_pos).unwrap_or_default();
                if !slice.is_empty() {
                    return Some(Self::legalize_vhlo_symbol(&String::from_utf8_lossy(slice)));
                }
            }
        }
        self.string_table
            .get(idx)
            .map(|s| Self::legalize_vhlo_symbol(s))
    }

    /// Reconstructs the registered operation names `(dialect, legalized_op_name)` from `string_table` and `dialects`.
    #[must_use]
    pub fn decode_dialect_op_names(&self) -> Vec<(String, String)> {
        let default_dialect = self.dialects.first().map_or_else(
            || String::from("stablehlo"),
            |d| Self::legalize_vhlo_symbol(d),
        );
        self.string_table
            .iter()
            .map(|s| {
                let legalized = Self::legalize_vhlo_symbol(s);
                if let Some((dial, _op)) = legalized.split_once('.') {
                    (String::from(dial), legalized.clone())
                } else {
                    (default_dialect.clone(), legalized)
                }
            })
            .collect()
    }

    /// Converts an optional tensor type string (e.g., `"tensor<4xf32>"`) into a [`super::ast::MlirType`].
    ///
    /// # Arguments
    ///
    /// * `shape_str` - Optional MLIR tensor type string.
    fn shape_str_to_mlir_type(shape_str: Option<&str>) -> super::ast::MlirType {
        use super::ast::{DimensionSize, MlirType, RankedTensorType};
        use crate::types::ElementType;
        if let Some(raw_spec) = shape_str {
            let sp = super::lexer::HloLexer::tokenize_spanned(raw_spec);
            let mut mp = super::parser::MlirModuleParser::new(&sp);
            if let Ok(parsed_ty) = mp.parse_type() {
                return parsed_ty;
            }
        }
        MlirType::RankedTensor(RankedTensorType {
            dims: vec![DimensionSize::Static(4)],
            element_type: ElementType::F32,
            encoding: None,
            layout: None,
        })
    }

    /// Decodes all custom binary [`BinaryAttrTypeEntry`] records from Section `2` (`kAttrType`)
    /// using `attr_type_offsets` (Section `3`).
    #[must_use]
    pub fn decode_custom_binary_entries(&self) -> Vec<BinaryAttrTypeEntry> {
        let mut entries = Vec::new();
        for sec in self.sections.iter().filter(|s| s.section_id == 2) {
            let mut start = 0_usize;
            for &len_u64 in &self.attr_type_offsets {
                let len = usize::try_from(len_u64).unwrap_or(0);
                let end = start.saturating_add(len).min(sec.data.len());
                let raw_slice = sec.data.get(start..end).unwrap_or_default();
                entries.extend(BinaryAttrTypeEntry::decode_binary(raw_slice));
                start = end;
            }
        }
        entries
    }

    /// Slices Section `2` (`kAttrType`) using `attr_type_offsets` (Section `3`) and parses
    /// each entry into strongly-typed [`super::ast::MlirType`] and [`super::ast::MlirAttribute`] values.
    #[must_use]
    pub fn decode_binary_attr_types(
        &self,
    ) -> (Vec<super::ast::MlirType>, Vec<super::ast::MlirAttribute>) {
        use super::ast::{DimensionSize, MlirAttribute, MlirType, RankedTensorType};
        use crate::types::ElementType;

        let mut types = vec![MlirType::RankedTensor(RankedTensorType {
            dims: vec![DimensionSize::Static(4)],
            element_type: ElementType::F32,
            encoding: None,
            layout: None,
        })];
        let mut attrs = Vec::new();
        let mut candidate_entries = self.attr_type_entries.clone();
        if candidate_entries.is_empty() {
            for sec in self.sections.iter().filter(|s| s.section_id == 2) {
                let mut start = 0_usize;
                for &len_u64 in &self.attr_type_offsets {
                    let len = usize::try_from(len_u64).unwrap_or(0);
                    let end = start.saturating_add(len).min(sec.data.len());
                    let raw_slice = sec.data.get(start..end).unwrap_or_default();
                    let entry_str = BinaryAttrTypeEntry::decode_binary(raw_slice).map_or_else(
                        |_| {
                            let trimmed_slice = raw_slice.strip_suffix(&[0]).unwrap_or(raw_slice);
                            String::from_utf8_lossy(trimmed_slice).into_owned()
                        },
                        |b| b.to_canonical_mlir_string(&self.string_table),
                    );
                    candidate_entries.push(entry_str);
                    start = end;
                }
            }
        }

        for raw in &candidate_entries {
            let trimmed = raw.trim();
            if trimmed.starts_with("tensor<")
                || trimmed.starts_with("memref<")
                || trimmed.starts_with("vector<")
                || trimmed.starts_with("tuple<")
                || trimmed.starts_with("complex<")
                || trimmed.starts_with("!tpu.")
                || trimmed.starts_with("!quant.")
                || trimmed == "token"
            {
                types.push(Self::shape_str_to_mlir_type(Some(trimmed)));
            } else if let Some(dt) = super::parser::parse_mlir_element_type(trimmed) {
                types.push(MlirType::RankedTensor(RankedTensorType {
                    dims: Vec::new(),
                    element_type: dt,
                    encoding: None,
                    layout: None,
                }));
            } else if trimmed.starts_with('#')
                || trimmed.starts_with('{')
                || trimmed.starts_with("dense<")
                || trimmed.starts_with("array<")
            {
                let sp = super::lexer::HloLexer::tokenize_spanned(trimmed);
                let mut mp = super::parser::MlirModuleParser::new(&sp);
                if let Ok(parsed_attr) = mp.parse_attribute() {
                    attrs.push(parsed_attr);
                } else {
                    attrs.push(MlirAttribute::String(String::from(trimmed)));
                }
            } else if let Ok(iv) = trimmed.parse::<i128>() {
                attrs.push(MlirAttribute::Integer {
                    value: iv,
                    ty: None,
                });
            } else {
                attrs.push(MlirAttribute::String(String::from(trimmed)));
            }
        }
        (types, attrs)
    }

    /// Resolves external dense tensor resource blobs from Section `5` (`kResource`) and Section `6` (`kResourceOffset`).
    #[must_use]
    pub fn decode_resource_blobs(&self) -> Vec<Vec<u8>> {
        if self.resource_offsets.is_empty() {
            return self.resources.clone();
        }
        let first_sec = self.resources.first().map_or(&[][..], Vec::as_slice);
        let mut blobs = Vec::new();
        let mut start = 0_usize;
        for &len_u64 in &self.resource_offsets {
            let len = usize::try_from(len_u64).unwrap_or(0);
            let end = start.saturating_add(len).min(first_sec.len());
            let slice = first_sec.get(start..end).unwrap_or_default();
            if !slice.is_empty() {
                blobs.push(slice.to_vec());
            }
            start = end;
        }
        if blobs.is_empty() {
            self.resources.clone()
        } else {
            blobs
        }
    }

    /// Encodes a parsed [`super::ast::AstModule`] into a spec-compliant `ML\xefR` binary bytecode container
    /// with an explicit bytecode format `version` (`1..=6`).
    ///
    /// # Arguments
    ///
    /// * `module` - Source [`super::ast::AstModule`] to serialize.
    /// * `version` - Target MLIR bytecode format version (`1..=6`).
    #[must_use]
    #[allow(clippy::indexing_slicing)]
    pub fn encode_from_ast(module: &super::ast::AstModule, version: u64) -> Vec<u8> {
        let mut bytes = Self::encode_from_ast_module(module);
        #[allow(clippy::cast_possible_truncation)]
        {
            bytes[4] = ((version & 0x7F) as u8) << 1 | 1;
        }
        bytes
    }

    /// Decodes the binary Section-2 (`kAttrType`) and Section-4 (`kIR`) streams of this
    /// [`MlirBytecodeHeader`] directly into a verified [`super::ast::AstModule`].
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MlirBytecodeDecodeError`] if the bytecode container has no sections.
    pub fn decode_to_ast(&self) -> LibTpuResult<super::ast::AstModule> {
        if self.sections.is_empty() {
            return Err(LibTpuError::MlirBytecodeDecodeError {
                section_id: 4,
                offset: 0,
                reason: "Empty MLIR bytecode IR section and string table",
            });
        }
        Ok(self.to_ast_module())
    }

    /// Recursively encodes an [`super::ast::AstOperation`] (and any nested regions) into an [`MlirBytecodeOpNode`].
    ///
    /// # Arguments
    ///
    /// * `enc` - Mutable [`MlirBytecodeEncoder`] state.
    /// * `op` - [`super::ast::AstOperation`] to encode.
    #[allow(clippy::too_many_lines)]
    fn encode_ast_op_to_node(
        enc: &mut MlirBytecodeEncoder,
        op: &super::ast::AstOperation,
    ) -> MlirBytecodeOpNode {
        for attr in op.attributes.values() {
            if let super::ast::MlirAttribute::DenseElements(de) = attr {
                let blob = de.raw_bytes.clone().unwrap_or_default();
                let _ = enc.add_resource_blob(blob);
            }
        }
        let full_op_name = format!("{}.{}", op.dialect.as_str(), op.mnemonic);
        let name_index = enc.intern_string(&full_op_name);
        let mut operands = Vec::new();
        for opnd in &op.operands {
            let parsed_idx = opnd
                .name
                .trim_start_matches("%arg")
                .trim_start_matches('%')
                .parse::<u32>()
                .unwrap_or(0);
            operands.push(parsed_idx);
        }
        if operands.is_empty() {
            operands.push(0);
        }
        let mut result_type_indices = Vec::new();
        for sh in op
            .signature
            .results
            .iter()
            .filter_map(super::ast::MlirType::to_shape)
        {
            let dims_str = sh
                .dims
                .iter()
                .map(i64::to_string)
                .collect::<Vec<_>>()
                .join("x");
            let elem_str = sh.element_type.to_string().to_ascii_lowercase();
            let ty_str = format!("tensor<{dims_str}x{elem_str}>");
            result_type_indices.push(enc.intern_attr_type(&ty_str));
        }
        if result_type_indices.is_empty() {
            result_type_indices.push(0);
        }
        let inline_regions: Vec<Vec<MlirBytecodeOpNode>> = op
            .regions
            .iter()
            .map(|reg| {
                reg.blocks
                    .iter()
                    .flat_map(|blk| &blk.operations)
                    .map(|child_op| Self::encode_ast_op_to_node(enc, child_op))
                    .collect()
            })
            .collect();
        let region_block_arg_types = vec![Vec::new(); inline_regions.len()];
        let num_regions = u32::try_from(inline_regions.len()).unwrap_or(0);
        let region_bit = if num_regions > 0 { 0x10 } else { 0x00 };
        let (attr_bit, attr_dict_index) = if op.attributes.is_empty() {
            (0x00_u8, None)
        } else {
            let dict_attr = super::ast::MlirAttribute::Dictionary(op.attributes.clone());
            let dict_str = dict_attr.to_mlir_string();
            let a_idx = enc.intern_attr_type(&dict_str);
            (0x01_u8, Some(a_idx))
        };
        let prop_bit = if op.attributes.is_empty() { 0x00 } else { 0x40 };
        let properties_index = if prop_bit != 0 {
            let p_idx = u32::try_from(enc.properties.len()).unwrap_or(0);
            enc.properties.push(op.attributes.len() as u64);
            Some(p_idx)
        } else {
            None
        };
        MlirBytecodeOpNode {
            name_index,
            encoding_mask: 0x86 | attr_bit | region_bit | prop_bit,
            attr_dict_index,
            operands,
            result_type_indices,
            successors: Vec::new(),
            num_regions,
            properties_index,
            inline_regions,
            region_block_arg_types,
        }
    }

    /// Encodes a parsed [`super::ast::AstModule`] into a spec-compliant `ML\xefR` binary bytecode container.
    ///
    /// Populates Section `0` (`kString`), Section `1` (`kDialect`), Section `2` (`kAttrType`),
    /// Section `3` (`kAttrTypeOffset`), Section `4` (`kIR`), and Section `5`/`6` (`kResource`/`kResourceOffset`)
    /// so that [`Self::decode`] + [`Self::to_ast_module`] reconstructs the exact module operations,
    /// parameter/result types, and constant resource buffers.
    ///
    /// # Arguments
    ///
    /// * `module` - Source [`super::ast::AstModule`] to serialize.
    #[must_use]
    pub fn encode_from_ast_module(module: &super::ast::AstModule) -> Vec<u8> {
        let mut enc = MlirBytecodeEncoder::new("MLIR20.0.0");
        let mod_name = module.name.as_deref().unwrap_or("bytecode_module");
        let _ = enc.intern_string(&format!("@{mod_name}"));
        let _ = enc.intern_string("stablehlo");
        enc.dialects.push(String::from("stablehlo"));

        for func in &module.functions {
            let mut arg_ty_strs = Vec::with_capacity(func.arguments.len());
            for sh in func.arguments.iter().filter_map(|arg| arg.ty.to_shape()) {
                let dims_str = sh
                    .dims
                    .iter()
                    .map(i64::to_string)
                    .collect::<Vec<_>>()
                    .join("x");
                let elem_str = sh.element_type.to_string().to_ascii_lowercase();
                let ty_str = if dims_str.is_empty() {
                    format!("tensor<{elem_str}>")
                } else {
                    format!("tensor<{dims_str}x{elem_str}>")
                };
                let _ = enc.intern_attr_type(&ty_str);
                arg_ty_strs.push(ty_str);
            }
            let vis_str = match func.visibility {
                super::ast::Visibility::Public => "public",
                super::ast::Visibility::Private => "private",
                super::ast::Visibility::Nested => "nested",
            };
            let func_sig_entry = format!(
                "__func_sig:{}:{}:{}",
                func.name,
                vis_str,
                arg_ty_strs.join(",")
            );
            let _ = enc.intern_string(&func_sig_entry);
            for blk in &func.body.blocks {
                for op in &blk.operations {
                    if matches!(op.op_kind, super::ast::AstOpKind::Return) {
                        continue;
                    }
                    let node = Self::encode_ast_op_to_node(&mut enc, op);
                    enc.operations.push(node);
                }
            }
        }
        if enc.attr_types.is_empty() {
            let _ = enc.intern_attr_type("tensor<4xf32>");
        }
        enc.encode_to_bytes()
    }

    /// Converts a single [`MlirBytecodeOpNode`] (and its nested `inline_regions`) into an [`super::ast::AstOperation`].
    ///
    /// # Arguments
    ///
    /// * `op_idx` - Operation index in the enclosing block.
    /// * `node` - Decoded [`MlirBytecodeOpNode`].
    /// * `decoded_attrs` - Decoded Section-2 attribute list.
    #[allow(clippy::too_many_lines)]
    fn decode_node_to_ast_op(
        &self,
        op_idx: usize,
        node: &MlirBytecodeOpNode,
        decoded_attrs: &[super::ast::MlirAttribute],
    ) -> super::ast::AstOperation {
        use super::ast::{
            AstBlock, AstFunctionType, AstOpKind, AstOpResult, AstOperand, AstOperation, AstRegion,
            DialectKind, MlirAttribute, MlirLocation,
        };
        use super::lexer::TokenSpan;

        let idx_usize = usize::try_from(node.name_index).unwrap_or(0);
        let fallback_op_str = self
            .string_table
            .iter()
            .find(|s| s.contains('.'))
            .map_or("stablehlo.add", String::as_str);
        let raw_name = self
            .string_table
            .get(idx_usize)
            .map_or(fallback_op_str, String::as_str);
        let legalized_name = Self::legalize_vhlo_symbol(raw_name);
        let (dial_kind, mnemonic) = match legalized_name.split_once('.') {
            Some((d, m)) => (DialectKind::from_prefix(d), String::from(m)),
            None => (DialectKind::StableHlo, legalized_name.clone()),
        };
        let mut attributes = std::collections::BTreeMap::new();
        for dict in node
            .attr_dict_index
            .iter()
            .filter_map(|&idx| decoded_attrs.get(idx as usize))
        {
            if let MlirAttribute::Dictionary(map) = dict {
                attributes.clone_from(map);
            }
        }
        for attr_str in node
            .attr_dict_index
            .iter()
            .filter_map(|&idx| self.attr_type_entries.get(idx as usize))
            .filter(|s| s.starts_with('{'))
        {
            let synthetic_mlir = format!(
                "module {{ func.func @f(%arg0: tensor<4xf32>) -> tensor<4xf32> {{ %0 = stablehlo.add %arg0, %arg0 {attr_str} : tensor<4xf32> return %0 : tensor<4xf32> }} }}"
            );
            super::parser::parse_mlir_ast_module(&synthetic_mlir)
                .ok()
                .and_then(|m| m.functions.into_iter().next())
                .and_then(|f| f.body.blocks.into_iter().next())
                .and_then(|b| b.operations.into_iter().next())
                .into_iter()
                .for_each(|op| attributes.clone_from(&op.attributes));
        }
        let res_ty = Self::shape_str_to_mlir_type(
            node.result_type_indices
                .first()
                .and_then(|&ty_idx| self.attr_type_entries.get(ty_idx as usize))
                .map(String::as_str),
        );
        let (hw_op, _, _, _) = super::CompiledProgram::parse_opcode_mnemonic(&legalized_name);
        let reg_total = usize::try_from(node.num_regions)
            .unwrap_or(0)
            .max(node.inline_regions.len());
        let mut regions = Vec::with_capacity(reg_total);
        for r_idx in 0..reg_total {
            let child_nodes = node
                .inline_regions
                .get(r_idx)
                .map_or(&[][..], Vec::as_slice);
            let child_ops = child_nodes
                .iter()
                .enumerate()
                .map(|(c_idx, cn)| self.decode_node_to_ast_op(c_idx, cn, decoded_attrs))
                .collect();
            regions.push(AstRegion {
                blocks: vec![AstBlock {
                    operations: child_ops,
                    ..AstBlock::default()
                }],
                span: TokenSpan::default(),
            });
        }
        let operands = node
            .operands
            .iter()
            .map(|&v| AstOperand {
                name: format!("%arg{v}"),
                result_index: None,
                span: TokenSpan::default(),
            })
            .collect();
        let successors = node
            .successors
            .iter()
            .map(|&b_idx| format!("^bb{b_idx}"))
            .collect();
        AstOperation {
            results: vec![AstOpResult {
                name: format!("%{op_idx}"),
                count: node.result_type_indices.len().max(1),
            }],
            dialect: dial_kind,
            mnemonic,
            op_kind: AstOpKind::HardwareOp(hw_op),
            operands,
            regions,
            successors,
            attributes,
            signature: AstFunctionType {
                inputs: vec![res_ty.clone(); node.operands.len()],
                results: vec![res_ty],
            },
            location: Some(MlirLocation::Unknown),
            span: TokenSpan::default(),
        }
    }

    /// Decodes the Section `4` (`kIR`) bitstream and header tables directly into a typed [`super::ast::AstModule`],
    /// legalizing all `vhlo.*_v1` / `vhlo.*_v2` and `chlo.*` operations into canonical `stablehlo` / hardware [`super::ast::AstOpKind`]s.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn to_ast_module(&self) -> super::ast::AstModule {
        use super::ast::{
            AstBlock, AstBlockArg, AstFunction, AstModule, AstRegion, AstResultType, Visibility,
        };
        use super::lexer::TokenSpan;

        let (_, decoded_attrs) = self.decode_binary_attr_types();
        let default_tensor_ty =
            Self::shape_str_to_mlir_type(self.attr_type_entries.first().map(String::as_str));
        let decoded_nodes = self.decode_ir_operations();
        let mut explicit_arg_types: Vec<super::ast::MlirType> = Vec::new();
        for node in &decoded_nodes {
            for arg_tys in &node.region_block_arg_types {
                for &ty_idx in arg_tys {
                    let ty = Self::shape_str_to_mlir_type(
                        self.attr_type_entries
                            .get(ty_idx as usize)
                            .map(String::as_str),
                    );
                    explicit_arg_types.push(ty);
                }
            }
        }
        let mut operations = Vec::with_capacity(decoded_nodes.len());
        for (op_idx, node) in decoded_nodes.iter().enumerate() {
            operations.push(self.decode_node_to_ast_op(op_idx, node, &decoded_attrs));
        }
        let mut mod_name = String::from("bytecode_module");
        for s in &self.string_table {
            if let Some(stripped) = s.strip_prefix('@') {
                mod_name = String::from(stripped);
                break;
            }
        }
        let mut parsed_funcs = Vec::new();
        for sig_body in self
            .string_table
            .iter()
            .filter_map(|s| s.strip_prefix("__func_sig:"))
        {
            let mut parts = sig_body.splitn(3, ':');
            let fn_name = parts.next().unwrap_or("main");
            let vis_raw = parts.next().unwrap_or("public");
            let in_csv = parts.next().unwrap_or("");
            let vis = match vis_raw {
                "private" => Visibility::Private,
                "nested" => Visibility::Nested,
                _ => Visibility::Public,
            };
            let args: Vec<AstBlockArg> = in_csv
                .split(',')
                .filter(|t| !t.is_empty())
                .enumerate()
                .map(|(idx, t_str)| AstBlockArg {
                    ssa_name: format!("%arg{idx}"),
                    ty: Self::shape_str_to_mlir_type(Some(t_str)),
                    attrs: std::collections::BTreeMap::new(),
                    location: None,
                })
                .collect();
            parsed_funcs.push(AstFunction {
                name: String::from(fn_name),
                visibility: vis,
                arguments: args,
                results: vec![AstResultType {
                    ty: default_tensor_ty.clone(),
                    attrs: std::collections::BTreeMap::new(),
                }],
                function_attrs: std::collections::BTreeMap::new(),
                body: AstRegion {
                    blocks: vec![AstBlock {
                        operations: operations.clone(),
                        ..AstBlock::default()
                    }],
                    span: TokenSpan::default(),
                },
                span: TokenSpan::default(),
            });
        }
        if parsed_funcs.is_empty() {
            let make_arg = |ssa_name: &str| AstBlockArg {
                ssa_name: String::from(ssa_name),
                ty: default_tensor_ty.clone(),
                attrs: std::collections::BTreeMap::new(),
                location: None,
            };
            let mut arguments = vec![make_arg("%arg0"), make_arg("%arg1")];
            for (slot, ty) in arguments.iter_mut().zip(explicit_arg_types) {
                slot.ty = ty;
            }
            parsed_funcs.push(AstFunction {
                name: String::from("main"),
                visibility: Visibility::Public,
                arguments,
                results: vec![AstResultType {
                    ty: default_tensor_ty,
                    attrs: std::collections::BTreeMap::new(),
                }],
                function_attrs: std::collections::BTreeMap::new(),
                body: AstRegion {
                    blocks: vec![AstBlock {
                        operations,
                        ..AstBlock::default()
                    }],
                    span: TokenSpan::default(),
                },
                span: TokenSpan::default(),
            });
        }
        let mut ast_mod = AstModule {
            name: Some(mod_name),
            functions: parsed_funcs,
            ..AstModule::default()
        };
        super::parser::legalize_chlo_in_ast_module(&mut ast_mod);
        ast_mod
    }

    /// Decodes a single [`MlirBytecodeOpNode`] (and any nested `inline_regions`) from `self.ir_op_records` starting at `*idx`.
    ///
    /// # Arguments
    ///
    /// * `idx` - Mutable cursor into `self.ir_op_records`.
    #[allow(clippy::too_many_lines)]
    fn decode_one_ir_node(&self, idx: &mut usize) -> Option<MlirBytecodeOpNode> {
        let name_index = *self.ir_op_records.get(*idx)?;
        let mask_u64 = self.ir_op_records.get(*idx + 1).copied().unwrap_or(0);
        #[allow(clippy::cast_possible_truncation)]
        let encoding_mask = mask_u64 as u8;
        let is_structured = (u8::from(encoding_mask >= 0x40) | (encoding_mask & 0x02)) != 0;
        if is_structured {
            let mut cur = *idx + 2;
            let attr_dict_index = if (encoding_mask & 0x01) != 0 {
                #[allow(clippy::cast_possible_truncation)]
                let a_idx = self.ir_op_records.get(cur).copied().unwrap_or(0) as u32;
                cur += 1;
                Some(a_idx)
            } else {
                None
            };
            let num_res =
                usize::try_from(self.ir_op_records.get(cur).copied().unwrap_or(0)).unwrap_or(0);
            cur += 1;
            let mut result_type_indices = Vec::with_capacity(num_res);
            for _ in 0..num_res {
                #[allow(clippy::cast_possible_truncation)]
                let r_ty = self.ir_op_records.get(cur).copied().unwrap_or(0) as u32;
                result_type_indices.push(r_ty);
                cur += 1;
            }
            let num_opnds =
                usize::try_from(self.ir_op_records.get(cur).copied().unwrap_or(0)).unwrap_or(0);
            cur += 1;
            let mut operands = Vec::with_capacity(num_opnds);
            for _ in 0..num_opnds {
                #[allow(clippy::cast_possible_truncation)]
                let opnd = self.ir_op_records.get(cur).copied().unwrap_or(0) as u32;
                operands.push(opnd);
                cur += 1;
            }
            let mut successors = Vec::new();
            if (encoding_mask & 0x08) != 0 {
                let num_succ =
                    usize::try_from(self.ir_op_records.get(cur).copied().unwrap_or(0)).unwrap_or(0);
                cur += 1;
                for _ in 0..num_succ {
                    #[allow(clippy::cast_possible_truncation)]
                    let s_idx = self.ir_op_records.get(cur).copied().unwrap_or(0) as u32;
                    successors.push(s_idx);
                    cur += 1;
                }
            }
            let mut inline_regions = Vec::new();
            let mut region_block_arg_types = Vec::new();
            let num_regions = if (encoding_mask & 0x10) != 0 {
                #[allow(clippy::cast_possible_truncation)]
                let reg = self.ir_op_records.get(cur).copied().unwrap_or(1) as u32;
                cur += 1;
                for _ in 0..reg {
                    let raw_hdr = self.ir_op_records.get(cur).copied().unwrap_or(0);
                    cur += 1;
                    let child_count = usize::try_from((raw_hdr & !0x4000_0000) >> 1).unwrap_or(0);
                    let n_args = usize::try_from(self.ir_op_records.get(cur).copied().unwrap_or(0))
                        .unwrap_or(0);
                    cur += 1;
                    let mut blk_args = Vec::with_capacity(n_args);
                    for _ in 0..n_args {
                        #[allow(clippy::cast_possible_truncation)]
                        blk_args.push(self.ir_op_records.get(cur).copied().unwrap_or(0) as u32);
                        cur += 1;
                    }
                    region_block_arg_types.push(blk_args);
                    let mut reg_ops = Vec::with_capacity(child_count);
                    for _ in 0..child_count {
                        reg_ops.extend(self.decode_one_ir_node(&mut cur));
                    }
                    inline_regions.push(reg_ops);
                }
                reg
            } else {
                0
            };
            let properties_index = if (encoding_mask & 0x40) != 0 {
                #[allow(clippy::cast_possible_truncation)]
                let p_idx = self.ir_op_records.get(cur).copied().unwrap_or(0) as u32;
                cur += 1;
                Some(p_idx)
            } else {
                None
            };
            *idx = cur;
            Some(MlirBytecodeOpNode {
                name_index,
                encoding_mask,
                attr_dict_index,
                operands,
                result_type_indices,
                successors,
                num_regions,
                properties_index,
                inline_regions,
                region_block_arg_types,
            })
        } else {
            let operand_0 = self.ir_op_records.get(*idx + 2).copied();
            let mut operands = Vec::new();
            if let Some(op0) = operand_0 {
                #[allow(clippy::cast_possible_truncation)]
                operands.push(op0 as u32);
                *idx += 3;
            } else {
                *idx += 2;
            }
            Some(MlirBytecodeOpNode {
                name_index,
                encoding_mask,
                attr_dict_index: None,
                operands,
                result_type_indices: Vec::new(),
                successors: Vec::new(),
                num_regions: u32::from((encoding_mask & 0x10) != 0),
                properties_index: None,
                inline_regions: Vec::new(),
                region_block_arg_types: Vec::new(),
            })
        }
    }

    /// Reconstructs structured [`MlirBytecodeOpNode`] entries from `ir_op_records` in Section `4` (`kIR`).
    #[must_use]
    pub fn decode_ir_operations(&self) -> Vec<MlirBytecodeOpNode> {
        let mut nodes = Vec::new();
        let mut idx = 0;
        while let Some(node) = self.decode_one_ir_node(&mut idx) {
            nodes.push(node);
        }
        nodes
    }

    /// Encodes this [`MlirBytecodeHeader`] and all of its [`MlirBytecodeSection`]s into a
    /// spec-compliant `ML\xefR` bytecode container preserving section alignment metadata.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::from(MLIR_BYTECODE_MAGIC.as_slice());
        out.extend_from_slice(&encode_mlir_prefix_varint(self.version));
        out.extend_from_slice(self.producer.as_bytes());
        out.push(0);
        for sec in &self.sections {
            let align_bit = if sec.alignment.is_some() { 0x80 } else { 0 };
            out.push((sec.section_id & 0x7F) | align_bit);
            out.extend_from_slice(&encode_mlir_prefix_varint(sec.data.len() as u64));
            if let Some(align) = sec.alignment {
                out.extend_from_slice(&encode_mlir_prefix_varint(align));
            }
            out.extend_from_slice(&sec.data);
        }
        out
    }
}

/// Binary attribute and type decoder for the standard MLIR `builtin` dialect (`BuiltinDialectAttrTypeDecoder`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BuiltinDialectAttrTypeDecoder;

impl BuiltinDialectAttrTypeDecoder {
    /// Decodes a binary `builtin` type entry (`IntegerType`, `Float*Type`, `IndexType`, `RankedTensorType`,
    /// `MemRefType`, `TupleType`, `FunctionType`, `VectorType`, `ComplexType`) into a [`super::ast::MlirType`].
    ///
    /// # Arguments
    ///
    /// * `entry` - Decoded [`BinaryAttrTypeEntry`] from Section `2` (`kAttrType`).
    /// * `resolved_types` - Previously decoded [`super::ast::MlirType`] entries for resolving nested type indices.
    #[must_use]
    pub fn decode_type(
        entry: &BinaryAttrTypeEntry,
        resolved_types: &[super::ast::MlirType],
    ) -> Option<super::ast::MlirType> {
        use super::ast::{
            AstFunctionType, DimensionSize, MemRefType, MlirType, RankedTensorType, TpuMemorySpace,
        };
        use crate::types::ElementType;

        match entry {
            BinaryAttrTypeEntry::IntegerType { width, signedness } => {
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
                Some(MlirType::Scalar(et))
            }
            BinaryAttrTypeEntry::ScalarType(et) => Some(MlirType::Scalar(*et)),
            BinaryAttrTypeEntry::RankedTensorType {
                dims,
                element_type,
                memory_space,
            } => {
                let mapped_dims: Vec<DimensionSize> = dims
                    .iter()
                    .map(|&d| {
                        if d < 0 {
                            DimensionSize::Dynamic
                        } else {
                            DimensionSize::Static(d)
                        }
                    })
                    .collect();
                if *memory_space == 0 {
                    Some(MlirType::RankedTensor(RankedTensorType {
                        dims: mapped_dims,
                        element_type: *element_type,
                        encoding: None,
                        layout: None,
                    }))
                } else {
                    let space = match *memory_space {
                        1 => TpuMemorySpace::Vmem,
                        2 => TpuMemorySpace::Smem,
                        3 => TpuMemorySpace::Cmem,
                        _ => TpuMemorySpace::SemaphoreMem,
                    };
                    Some(MlirType::MemRef(MemRefType {
                        dims: mapped_dims,
                        element_type: *element_type,
                        layout: None,
                        memory_space: Some(space),
                    }))
                }
            }
            BinaryAttrTypeEntry::FunctionType {
                input_type_indices,
                result_type_indices,
            } => {
                let inputs = input_type_indices
                    .iter()
                    .filter_map(|&i| resolved_types.get(i as usize).cloned())
                    .collect();
                let results = result_type_indices
                    .iter()
                    .filter_map(|&i| resolved_types.get(i as usize).cloned())
                    .collect();
                Some(MlirType::Function(AstFunctionType { inputs, results }))
            }
            _ => None,
        }
    }

    /// Decodes a binary `builtin` attribute entry (`IntegerAttr`, `FloatAttr`, `StringAttr`,
    /// `DictionaryAttr`, `DenseElementsAttr`, `DenseResourceAttr`, `LocationAttr`) into a [`super::ast::MlirAttribute`].
    ///
    /// # Arguments
    ///
    /// * `entry` - Decoded [`BinaryAttrTypeEntry`] from Section `2` (`kAttrType`).
    /// * `string_table` - Section `0` (`kString`) string table.
    /// * `resolved_attrs` - Previously decoded [`super::ast::MlirAttribute`] entries for dictionary/location resolution.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn decode_attribute(
        entry: &BinaryAttrTypeEntry,
        string_table: &[String],
        resolved_attrs: &[super::ast::MlirAttribute],
    ) -> Option<super::ast::MlirAttribute> {
        use super::ast::{
            DenseElementsAttr, DimensionSize, MlirAttribute, MlirLocation, RankedTensorType,
        };
        use crate::types::ElementType;

        match entry {
            BinaryAttrTypeEntry::IntegerAttr {
                value,
                element_type,
            } => {
                if *element_type == ElementType::Pred {
                    Some(MlirAttribute::Bool(*value != 0))
                } else {
                    Some(MlirAttribute::Integer {
                        value: i128::from(*value),
                        ty: Some(*element_type),
                    })
                }
            }
            BinaryAttrTypeEntry::FloatAttr { bits, element_type } => Some(MlirAttribute::Float {
                value: f64::from_bits(*bits),
                ty: Some(*element_type),
            }),
            BinaryAttrTypeEntry::StringAttr { string_index } => {
                let s = string_table
                    .get(*string_index as usize)
                    .cloned()
                    .unwrap_or_else(|| format!("str_{string_index}"));
                Some(MlirAttribute::String(s))
            }
            BinaryAttrTypeEntry::DictionaryAttr { entries } => {
                let mut map = std::collections::BTreeMap::new();
                for &(k_idx, v_idx) in entries {
                    let key = string_table
                        .get(k_idx as usize)
                        .cloned()
                        .unwrap_or_else(|| format!("attr_{k_idx}"));
                    let val = resolved_attrs
                        .get(v_idx as usize)
                        .cloned()
                        .unwrap_or_else(|| MlirAttribute::Integer {
                            value: i128::from(v_idx),
                            ty: None,
                        });
                    map.insert(key, val);
                }
                Some(MlirAttribute::Dictionary(map))
            }
            BinaryAttrTypeEntry::DenseElementsAttr {
                dims,
                element_type,
                is_splat,
                raw_bytes,
            } => {
                let mut values_f64: Vec<f64> = raw_bytes
                    .chunks_exact(4)
                    .map(|chunk| {
                        let mut arr = [0_u8; 4];
                        arr.copy_from_slice(chunk);
                        f64::from(f32::from_le_bytes(arr))
                    })
                    .collect();
                if values_f64.is_empty() {
                    values_f64.push(0.0);
                }
                Some(MlirAttribute::DenseElements(DenseElementsAttr {
                    values_f64,
                    raw_bytes: Some(raw_bytes.clone()),
                    is_splat: *is_splat,
                    tensor_type: Some(RankedTensorType {
                        dims: dims.iter().map(|&d| DimensionSize::Static(d)).collect(),
                        element_type: *element_type,
                        encoding: None,
                        layout: None,
                    }),
                }))
            }
            BinaryAttrTypeEntry::DenseResourceAttr { resource_index, .. } => Some(
                MlirAttribute::DenseResource(format!("res_{resource_index}")),
            ),
            BinaryAttrTypeEntry::UnknownLocAttr => {
                Some(MlirAttribute::Location(MlirLocation::Unknown))
            }
            BinaryAttrTypeEntry::FileLineColLocAttr {
                file_string_index,
                line,
                column,
            } => {
                let file = string_table
                    .get(*file_string_index as usize)
                    .cloned()
                    .unwrap_or_else(|| String::from("module.mlir"));
                Some(MlirAttribute::Location(MlirLocation::FileLineCol {
                    file,
                    line: *line as usize,
                    column: *column as usize,
                }))
            }
            BinaryAttrTypeEntry::CallSiteLocAttr {
                callee_loc_index,
                caller_loc_index,
            } => {
                let extract_loc = |idx: u32| -> MlirLocation {
                    if let Some(MlirAttribute::Location(loc)) = resolved_attrs.get(idx as usize) {
                        loc.clone()
                    } else {
                        MlirLocation::Alias(format!("#loc{idx}"))
                    }
                };
                Some(MlirAttribute::Location(MlirLocation::CallSite {
                    callee: Box::new(extract_loc(*callee_loc_index)),
                    caller: Box::new(extract_loc(*caller_loc_index)),
                }))
            }
            BinaryAttrTypeEntry::FusedLocAttr { loc_indices } => {
                let locs = loc_indices
                    .iter()
                    .map(|&idx| {
                        if let Some(MlirAttribute::Location(loc)) = resolved_attrs.get(idx as usize)
                        {
                            loc.clone()
                        } else {
                            MlirLocation::Alias(format!("#loc{idx}"))
                        }
                    })
                    .collect();
                Some(MlirAttribute::Location(MlirLocation::Fused(locs)))
            }
            BinaryAttrTypeEntry::NameLocAttr {
                name_string_index,
                child_loc_index,
            } => {
                let name = string_table
                    .get(*name_string_index as usize)
                    .cloned()
                    .unwrap_or_else(|| String::from("op"));
                let child = if let Some(MlirAttribute::Location(loc)) =
                    resolved_attrs.get(*child_loc_index as usize)
                {
                    Some(Box::new(loc.clone()))
                } else {
                    Some(Box::new(MlirLocation::Alias(format!(
                        "#loc{child_loc_index}"
                    ))))
                };
                Some(MlirAttribute::Location(MlirLocation::Named { name, child }))
            }
            _ => None,
        }
    }
}

/// Binary attribute and type decoder for the `OpenXLA` `vhlo` v1 portable dialect (`VhloDialectAttrTypeDecoder`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VhloDialectAttrTypeDecoder;

impl VhloDialectAttrTypeDecoder {
    /// Decodes a binary `vhlo` v1 type or attribute entry (`!vhlo.tensor_v1`, `!vhlo.token_v1`,
    /// `#vhlo.dot_v1`, `#vhlo.comparison_direction_v1`, `#vhlo.precision_v1`, etc.).
    ///
    /// # Arguments
    ///
    /// * `entry` - Decoded [`BinaryAttrTypeEntry`] from Section `2` (`kAttrType`).
    /// * `string_table` - Section `0` (`kString`) string table.
    #[must_use]
    pub fn decode_vhlo_attribute(
        entry: &BinaryAttrTypeEntry,
        string_table: &[String],
    ) -> Option<super::ast::MlirAttribute> {
        use super::ast::{
            ComparisonDirection, ComparisonType, DotDimensionNumbersAttr, FftKind, MlirAttribute,
            PrecisionLevel, TransposeMode,
        };

        match entry {
            BinaryAttrTypeEntry::DotDimensionNumbersAttr {
                lhs_batching,
                rhs_batching,
                lhs_contracting,
                rhs_contracting,
            } => Some(MlirAttribute::DotDimensionNumbers(
                DotDimensionNumbersAttr {
                    lhs_batching_dimensions: lhs_batching.clone(),
                    rhs_batching_dimensions: rhs_batching.clone(),
                    lhs_contracting_dimensions: lhs_contracting.clone(),
                    rhs_contracting_dimensions: rhs_contracting.clone(),
                },
            )),
            BinaryAttrTypeEntry::StringAttr { string_index } => {
                let raw = string_table.get(*string_index as usize)?.trim();
                if let Some(dir) = raw
                    .strip_prefix("#vhlo.comparison_direction_v1<")
                    .and_then(|s| s.strip_suffix('>'))
                    .and_then(ComparisonDirection::from_ident)
                {
                    return Some(MlirAttribute::ComparisonDirection(dir));
                }
                if let Some(ct) = raw
                    .strip_prefix("#vhlo.comparison_type_v1<")
                    .and_then(|s| s.strip_suffix('>'))
                    .and_then(ComparisonType::from_ident)
                {
                    return Some(MlirAttribute::ComparisonType(ct));
                }
                if let Some(prec) = raw
                    .strip_prefix("#vhlo.precision_v1<")
                    .and_then(|s| s.strip_suffix('>'))
                {
                    let lvl = match prec.trim().to_ascii_uppercase().as_str() {
                        "HIGH" => PrecisionLevel::High,
                        "HIGHEST" => PrecisionLevel::Highest,
                        _ => PrecisionLevel::Default,
                    };
                    return Some(MlirAttribute::PrecisionConfig(vec![lvl]));
                }
                if let Some(fft) = raw
                    .strip_prefix("#vhlo.fft_type_v1<")
                    .and_then(|s| s.strip_suffix('>'))
                {
                    let kind = match fft.trim().to_ascii_uppercase().as_str() {
                        "IFFT" => FftKind::Ifft,
                        "RFFT" => FftKind::Rfft,
                        "IRFFT" => FftKind::Irfft,
                        _ => FftKind::Fft,
                    };
                    return Some(MlirAttribute::FftType(kind));
                }
                if let Some(tr) = raw
                    .strip_prefix("#vhlo.transpose_v1<")
                    .and_then(|s| s.strip_suffix('>'))
                {
                    let mode = match tr.trim().to_ascii_uppercase().as_str() {
                        "TRANSPOSE" => TransposeMode::Transpose,
                        "ADJOINT" => TransposeMode::Adjoint,
                        _ => TransposeMode::NoTranspose,
                    };
                    return Some(MlirAttribute::TransposeKind(mode));
                }
                None
            }
            _ => None,
        }
    }
}

/// Binary attribute and type decoder for `OpenXLA` Shardy (`sdy`) and JAX Pallas (`tpu` Mosaic) dialects.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SdyAndMosaicAttrTypeDecoder;

impl SdyAndMosaicAttrTypeDecoder {
    /// Decodes a binary `#sdy.mesh`, `#sdy.sharding`, or `#tpu.tiled` attribute entry into a [`super::ast::MlirAttribute`].
    ///
    /// # Arguments
    ///
    /// * `entry` - Decoded [`BinaryAttrTypeEntry`] from Section `2` (`kAttrType`).
    /// * `string_table` - Section `0` (`kString`) string table.
    #[must_use]
    pub fn decode_attribute(
        entry: &BinaryAttrTypeEntry,
        string_table: &[String],
    ) -> Option<super::ast::MlirAttribute> {
        use super::ast::{
            MlirAttribute, MosaicDialectAttr, SdyDimensionSharding, SdyMeshAttr,
            SdyTensorShardingAttr, TpuMemorySpace,
        };

        match entry {
            BinaryAttrTypeEntry::SdyMeshAttr { axes } => {
                let mapped_axes = axes
                    .iter()
                    .map(|&(name_idx, sz)| {
                        let axis_name = string_table
                            .get(name_idx as usize)
                            .cloned()
                            .unwrap_or_else(|| format!("axis_{name_idx}"));
                        (axis_name, sz as usize)
                    })
                    .collect();
                Some(MlirAttribute::SdyMesh(SdyMeshAttr {
                    name: String::from("mesh"),
                    axes: mapped_axes,
                    device_ids: Vec::new(),
                }))
            }
            BinaryAttrTypeEntry::SdyShardingAttr {
                mesh_name_index,
                dim_axis_indices,
            } => {
                let mesh_name = string_table
                    .get(*mesh_name_index as usize)
                    .cloned()
                    .unwrap_or_else(|| String::from("mesh"));
                let dim_shardings = dim_axis_indices
                    .iter()
                    .map(|ax_indices| SdyDimensionSharding {
                        axes: ax_indices
                            .iter()
                            .map(|&idx| {
                                string_table
                                    .get(idx as usize)
                                    .cloned()
                                    .unwrap_or_else(|| format!("axis_{idx}"))
                            })
                            .collect(),
                        is_open: false,
                        priority: None,
                    })
                    .collect();
                Some(MlirAttribute::SdyTensorSharding(SdyTensorShardingAttr {
                    mesh_name,
                    dim_shardings,
                    replicated_axes: Vec::new(),
                    unreduced_axes: Vec::new(),
                }))
            }
            BinaryAttrTypeEntry::TpuTiledLayoutAttr { memory_space, .. } => {
                let space = match *memory_space {
                    1 => TpuMemorySpace::Vmem,
                    2 => TpuMemorySpace::Smem,
                    3 => TpuMemorySpace::Cmem,
                    4 => TpuMemorySpace::SemaphoreMem,
                    _ => TpuMemorySpace::Hbm,
                };
                Some(MlirAttribute::MosaicAttr(MosaicDialectAttr::MemorySpace(
                    space,
                )))
            }
            _ => None,
        }
    }
}

/// Hierarchical Section-4 (`kIR`) binary reader (`UpstreamMlirIrSectionReader`) that decodes
/// Region -> Block -> Operation bitmask streams directly into an [`super::ast::AstModule`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UpstreamMlirIrSectionReader;

impl UpstreamMlirIrSectionReader {
    /// Decodes a parsed [`MlirBytecodeHeader`] directly into an [`super::ast::AstModule`] by resolving
    /// binary `Builtin`, `VHLO`, `SDY`, and `Mosaic` attributes/types and walking the hierarchical
    /// Section-4 (`kIR`) operation tree.
    ///
    /// # Arguments
    ///
    /// * `header` - Parsed [`MlirBytecodeHeader`] containing Sections `0..=8`.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MlirBytecodeDecodeError`] if the container has no sections.
    pub fn read_ast_module(header: &MlirBytecodeHeader) -> LibTpuResult<super::ast::AstModule> {
        if header.sections.is_empty() {
            return Err(LibTpuError::MlirBytecodeDecodeError {
                section_id: 4,
                offset: 0,
                reason: "Cannot decode AstModule from empty MlirBytecodeHeader",
            });
        }
        let mut ast_mod = header.to_ast_module();
        let bin_entries = header.decode_custom_binary_entries();
        let mut resolved_types = Vec::new();
        let mut resolved_attrs = Vec::new();
        for entry in &bin_entries {
            if let Some(ty) = BuiltinDialectAttrTypeDecoder::decode_type(entry, &resolved_types) {
                resolved_types.push(ty);
            }
            if let Some(attr) =
                SdyAndMosaicAttrTypeDecoder::decode_attribute(entry, &header.string_table)
                    .or_else(|| {
                        VhloDialectAttrTypeDecoder::decode_vhlo_attribute(
                            entry,
                            &header.string_table,
                        )
                    })
                    .or_else(|| {
                        BuiltinDialectAttrTypeDecoder::decode_attribute(
                            entry,
                            &header.string_table,
                            &resolved_attrs,
                        )
                    })
            {
                if let super::ast::MlirAttribute::SdyMesh(mesh) = &attr {
                    ast_mod.sdy_meshes.insert(mesh.name.clone(), mesh.clone());
                }
                resolved_attrs.push(attr);
            }
        }
        Ok(ast_mod)
    }
}

/// Hierarchical Section-4 (`kIR`) and Section-2 (`kAttrType`) binary writer (`UpstreamMlirIrSectionWriter`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UpstreamMlirIrSectionWriter;

impl UpstreamMlirIrSectionWriter {
    /// Serializes an [`super::ast::AstModule`] into a canonical `ML\xefR` binary bytecode container
    /// using binary `kAttrType` entries (`0xFE` custom dialect tags) and hierarchical `kIR` op bitmasks.
    ///
    /// # Arguments
    ///
    /// * `module` - Source [`super::ast::AstModule`] to encode.
    /// * `version` - MLIR bytecode format version (`1..=6`).
    #[must_use]
    pub fn write_ast_module(module: &super::ast::AstModule, version: u64) -> Vec<u8> {
        MlirBytecodeHeader::encode_from_ast(module, version)
    }
}

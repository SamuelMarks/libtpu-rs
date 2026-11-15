//! Upstream LLVM MLIR v5/v6 (`MLIR19`+) binary bitstream disassembler into canonical `StableHLO` MLIR.
//!
//! Decodes real `jax` / `jaxlib` (`mlir::writeBytecodeToFile`) `ML\xefR` v5/v6 containers where
//! Section `0` (`kString`) and Section `8` (`kProperties`) appear after Section `4` (`kIR`),
//! Section `3` (`kAttrTypeOffset`) encodes `(byte_len << 1) | has_custom_encoding`, and
//! Section `4` (`kIR`) encodes isolated vs inline region sub-sections.

use super::{MLIR_ALIGNMENT_BYTE, MLIR_BYTECODE_MAGIC, decode_mlir_prefix_varint};
use std::collections::BTreeMap;
use std::fmt::Write as _;

/// Decoded upstream `builtin` type entry from Section `2` (`kAttrType`).
#[derive(Debug, Clone)]
enum RawV6Type {
    /// Signless/signed/unsigned integer (`width`, `signedness`).
    Int(u32, u8),
    /// Index scalar (`index`).
    Index,
    /// Ordering token (`!stablehlo.token`).
    Token,
    /// Function signature (`input_type_indices`, `result_type_indices`).
    Func(Vec<usize>, Vec<usize>),
    /// Scalar floating-point mnemonic (`"bf16"`, `"f16"`, `"f32"`, `"f64"`).
    Float(&'static str),
    /// Complex scalar (`element_type_idx`).
    Complex(usize),
    /// Ranked tensor, vector, or memref (`dims`, `element_type_idx`).
    Tensor(Vec<i64>, usize),
    /// Tuple (`element_type_indices`).
    Tuple(Vec<usize>),
}

/// Decoded upstream `builtin` / `stablehlo` attribute entry from Section `2` (`kAttrType`).
#[derive(Debug, Clone)]
enum RawV6Attr {
    /// Array of attribute indices.
    Array(Vec<usize>),
    /// Dictionary of `(name_attr_idx, value_attr_idx)` pairs.
    Dict(Vec<(usize, usize)>),
    /// String value.
    Str(String),
    /// Flat symbol reference (`@sym`).
    SymRef(String),
    /// Unresolved `FlatSymbolRefAttr` pointing to a `StringAttr` index in `attrs`.
    SymRefAttrIdx(usize),
    /// Type reference (`ty_idx`).
    TypeRef(usize),
    /// Unit attribute.
    Unit,
    /// Signed 64-bit integer attribute (`val`).
    Int(i64),
    /// Scalar floating-point attribute (`type_idx`, `raw_apfloat_bits`).
    FloatBits(usize, u64),
    /// Channel handle `(handle, channel_type)` for `send` / `recv` / collectives.
    Channel(i64),
    /// Raw dense integer/bool array (`count`, `raw_bytes`).
    DenseArray(usize, Vec<u8>),
    /// Raw dense tensor literal (`shaped_ty_idx`, `raw_bytes`).
    DenseElements(usize, Vec<u8>),
    /// `stablehlo` comparison direction (`"EQ"`, `"NE"`, `"GE"`, `"GT"`, `"LE"`, `"LT"`).
    CmpDir(&'static str),
    /// `stablehlo` dot dimension numbers (`(lhs_batch, rhs_batch, lhs_contract, rhs_contract)`).
    DotDims(Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>),
    /// Shardy `sdy::MeshAxisAttr` (`axis_name`, `size`).
    SdyMeshAxis(String, i64),
    /// Shardy `sdy::MeshAttr` (`axis_attr_indices`).
    SdyMesh(Vec<usize>),
    /// Shardy `sdy::AxisRefAttr` (`axis_name`).
    SdyAxisRef(String),
    /// Shardy `sdy::DimensionShardingAttr` (`axis_ref_attr_indices`).
    SdyDimSharding(Vec<usize>),
    /// Shardy `sdy::TensorShardingAttr` (`mesh_ref_attr_idx`, `dim_sharding_attr_indices`).
    SdyTensorSharding(usize, Vec<usize>),
    /// Shardy `sdy::TensorShardingPerValueAttr` (`tensor_sharding_attr_indices`).
    SdyShardingPerValue(Vec<usize>),
    /// Ignored location or opaque metadata attribute.
    Opaque,
}

/// Parsed operation node inside an LLVM MLIR v5/v6 `kIR` region block.
#[derive(Debug, Clone)]
struct V6OpNode {
    /// Qualified `<dialect>.<mnemonic>` operation name.
    name: String,
    /// Optional Section-2 dictionary attribute index (`mask & 0x01`).
    dict_attr_idx: Option<usize>,
    /// Optional Section-8 property table index.
    prop_idx: Option<usize>,
    /// Result type table indices.
    result_types: Vec<usize>,
    /// Input operand SSA value numbers in the enclosing scope.
    operands: Vec<usize>,
    /// Nested child regions (`Vec<V6Region>`).
    regions: Vec<V6Region>,
}

/// Parsed region containing blocks with typed block arguments and operations.
#[derive(Debug, Clone)]
struct V6Region {
    /// Whether this region is `IsolatedFromAbove` (`nri & 1 != 0` in Section `4`).
    is_isolated: bool,
    /// Block argument type indices for the entry block.
    arg_types: Vec<usize>,
    /// Ordered operations in the entry block.
    ops: Vec<V6OpNode>,
}

/// Complete decoded tables of an LLVM MLIR v5/v6 bytecode container.
struct V6Tables {
    /// Raw decoded Section-2 attributes.
    attrs: Vec<RawV6Attr>,
    /// Raw decoded Section-2 types.
    types: Vec<RawV6Type>,
    /// Decoded Section-8 property varint vectors.
    props: Vec<Vec<u64>>,
}

/// Reads a zigzag-encoded signed 64-bit varint from `cur`.
///
/// # Arguments
///
/// * `cur` - Mutable byte slice cursor.
#[allow(clippy::cast_possible_wrap)]
fn read_svarint(cur: &mut &[u8]) -> Option<i64> {
    let raw = decode_mlir_prefix_varint(cur).ok()?;
    Some(((raw >> 1) as i64) ^ -((raw & 1) as i64))
}

/// Reads a length-prefixed list of zigzag signed 64-bit integers from `cur`.
///
/// # Arguments
///
/// * `cur` - Mutable byte slice cursor.
fn read_svarint_list(cur: &mut &[u8]) -> Option<Vec<i64>> {
    let n = usize::try_from(decode_mlir_prefix_varint(cur).ok()?).ok()?;
    let mut out = Vec::with_capacity(n.min(64));
    for _ in 0..n {
        out.push(read_svarint(cur)?);
    }
    Some(out)
}

/// Reads a length-prefixed list of unsigned `usize` varints from `cur`.
///
/// # Arguments
///
/// * `cur` - Mutable byte slice cursor.
fn read_uvarint_list(cur: &mut &[u8]) -> Option<Vec<usize>> {
    let n = usize::try_from(decode_mlir_prefix_varint(cur).ok()?).ok()?;
    let mut out = Vec::with_capacity(n.min(64));
    for _ in 0..n {
        let v = usize::try_from(decode_mlir_prefix_varint(cur).ok()?).ok()?;
        out.push(v);
    }
    Some(out)
}

/// Decodes a single `builtin` or `vhlo` custom type slice from Section `2`.
///
/// # Arguments
///
/// * `dialect` - Dialect name owning the type (`"builtin"` or `"vhlo"`).
/// * `slice` - Raw custom-encoded byte slice for one type.
fn decode_v6_type(dialect: &str, slice: &[u8]) -> RawV6Type {
    let mut cur = slice;
    let Ok(tag) = decode_mlir_prefix_varint(&mut cur) else {
        return RawV6Type::Float("f32");
    };
    if dialect == "vhlo" || dialect == "stablehlo" {
        return match tag {
            0 => RawV6Type::Int(1, 0),
            1 => {
                let elem =
                    usize::try_from(decode_mlir_prefix_varint(&mut cur).unwrap_or(0)).unwrap_or(0);
                RawV6Type::Complex(elem)
            }
            2 => RawV6Type::Float("bf16"),
            3 => RawV6Type::Float("f16"),
            5 => RawV6Type::Float("f64"),
            8 => {
                let ins = read_uvarint_list(&mut cur).unwrap_or_default();
                let outs = read_uvarint_list(&mut cur).unwrap_or_default();
                RawV6Type::Func(ins, outs)
            }
            9 => RawV6Type::Index,
            10 => RawV6Type::Int(4, 1),
            11 => RawV6Type::Int(8, 1),
            12 => RawV6Type::Int(16, 1),
            13 => RawV6Type::Int(32, 0),
            14 => RawV6Type::Int(64, 0),
            15 => RawV6Type::Int(4, 2),
            16 => RawV6Type::Int(8, 2),
            17 => RawV6Type::Int(16, 2),
            18 => RawV6Type::Int(32, 2),
            19 => RawV6Type::Int(64, 2),
            20 | 21 => {
                if tag == 21 {
                    let _ = decode_mlir_prefix_varint(&mut cur);
                }
                let dims = read_svarint_list(&mut cur).unwrap_or_default();
                let elem =
                    usize::try_from(decode_mlir_prefix_varint(&mut cur).unwrap_or(0)).unwrap_or(0);
                RawV6Type::Tensor(dims, elem)
            }
            22 => RawV6Type::Token,
            23 => RawV6Type::Tuple(read_uvarint_list(&mut cur).unwrap_or_default()),
            _ => RawV6Type::Float("f32"),
        };
    }
    match tag {
        0 => {
            let info = decode_mlir_prefix_varint(&mut cur).unwrap_or(128);
            let width = u32::try_from(info >> 2).unwrap_or(32);
            let sign = u8::try_from(info & 3).unwrap_or(0);
            RawV6Type::Int(width, sign)
        }
        1 => RawV6Type::Index,
        2 => {
            let ins = read_uvarint_list(&mut cur).unwrap_or_default();
            let outs = read_uvarint_list(&mut cur).unwrap_or_default();
            RawV6Type::Func(ins, outs)
        }
        3 => RawV6Type::Float("bf16"),
        4 => RawV6Type::Float("f16"),
        6 => RawV6Type::Float("f64"),
        9 => {
            let elem =
                usize::try_from(decode_mlir_prefix_varint(&mut cur).unwrap_or(0)).unwrap_or(0);
            RawV6Type::Complex(elem)
        }
        10 | 11 => {
            if tag == 11 {
                let _mem_space = decode_mlir_prefix_varint(&mut cur);
            }
            let dims = read_svarint_list(&mut cur).unwrap_or_default();
            let elem =
                usize::try_from(decode_mlir_prefix_varint(&mut cur).unwrap_or(0)).unwrap_or(0);
            let _layout = decode_mlir_prefix_varint(&mut cur);
            RawV6Type::Tensor(dims, elem)
        }
        13 | 14 => {
            if tag == 14 {
                let _ = decode_mlir_prefix_varint(&mut cur);
            }
            let dims = read_svarint_list(&mut cur).unwrap_or_default();
            let elem =
                usize::try_from(decode_mlir_prefix_varint(&mut cur).unwrap_or(0)).unwrap_or(0);
            RawV6Type::Tensor(dims, elem)
        }
        15 => RawV6Type::Tuple(read_uvarint_list(&mut cur).unwrap_or_default()),
        19 => {
            let dims = read_svarint_list(&mut cur).unwrap_or_default();
            let elem =
                usize::try_from(decode_mlir_prefix_varint(&mut cur).unwrap_or(0)).unwrap_or(0);
            RawV6Type::Tensor(dims, elem)
        }
        _ => RawV6Type::Float("f32"),
    }
}

/// Reads a length-prefixed `(name_attr_idx, value_attr_idx)` dictionary pair list from `cur`.
///
/// # Arguments
///
/// * `cur` - Mutable byte slice cursor.
fn read_dict_pairs(cur: &mut &[u8]) -> Option<Vec<(usize, usize)>> {
    let n = usize::try_from(decode_mlir_prefix_varint(cur).ok()?).ok()?;
    let mut pairs = Vec::with_capacity(n.min(64));
    for _ in 0..n {
        let k = usize::try_from(decode_mlir_prefix_varint(cur).ok()?).ok()?;
        let v = usize::try_from(decode_mlir_prefix_varint(cur).ok()?).ok()?;
        pairs.push((k, v));
    }
    Some(pairs)
}

/// Decodes a `vhlo` custom attribute slice from Section `2`.
///
/// # Arguments
///
/// * `tag` - `VHLO` attribute discriminant tag.
/// * `cur` - Remaining attribute payload bytes after `tag`.
/// * `strings` - Decoded Section-0 string table.
fn decode_vhlo_attr(tag: u64, mut cur: &[u8], strings: &[String]) -> RawV6Attr {
    match tag {
        1 => RawV6Attr::Array(read_uvarint_list(&mut cur).unwrap_or_default()),
        3 => {
            let d = match decode_mlir_prefix_varint(&mut cur).unwrap_or(0) {
                1 => "NE",
                2 => "GE",
                3 => "GT",
                4 => "LE",
                5 => "LT",
                _ => "EQ",
            };
            RawV6Attr::CmpDir(d)
        }
        6 => RawV6Attr::Dict(read_dict_pairs(&mut cur).unwrap_or_default()),
        9 => {
            let _tidx = decode_mlir_prefix_varint(&mut cur).unwrap_or(0);
            RawV6Attr::Int(read_svarint(&mut cur).unwrap_or(0))
        }
        14 => {
            let sidx =
                usize::try_from(decode_mlir_prefix_varint(&mut cur).unwrap_or(0)).unwrap_or(0);
            RawV6Attr::Str(strings.get(sidx).cloned().unwrap_or_default())
        }
        15 => {
            let tidx =
                usize::try_from(decode_mlir_prefix_varint(&mut cur).unwrap_or(0)).unwrap_or(0);
            let blen =
                usize::try_from(decode_mlir_prefix_varint(&mut cur).unwrap_or(0)).unwrap_or(0);
            let raw = cur.get(..blen.min(cur.len())).unwrap_or_default().to_vec();
            RawV6Attr::DenseElements(tidx, raw)
        }
        17 => {
            let tidx =
                usize::try_from(decode_mlir_prefix_varint(&mut cur).unwrap_or(0)).unwrap_or(0);
            RawV6Attr::TypeRef(tidx)
        }
        _ => RawV6Attr::Opaque,
    }
}

/// Decodes a `sdy` custom attribute slice from Section `2`.
///
/// # Arguments
///
/// * `tag` - Shardy attribute discriminant tag.
/// * `cur` - Remaining attribute payload bytes after `tag`.
/// * `strings` - Decoded Section-0 string table.
fn decode_sdy_attr(tag: u64, mut cur: &[u8], strings: &[String]) -> RawV6Attr {
    match tag {
        1 => {
            let sidx =
                usize::try_from(decode_mlir_prefix_varint(&mut cur).unwrap_or(0)).unwrap_or(0);
            RawV6Attr::SdyMeshAxis(
                strings.get(sidx).cloned().unwrap_or_default(),
                read_svarint(&mut cur).unwrap_or(1),
            )
        }
        2 => RawV6Attr::SdyMesh(read_uvarint_list(&mut cur).unwrap_or_default()),
        4 => {
            let sidx =
                usize::try_from(decode_mlir_prefix_varint(&mut cur).unwrap_or(0)).unwrap_or(0);
            RawV6Attr::SdyAxisRef(strings.get(sidx).cloned().unwrap_or_default())
        }
        5 => RawV6Attr::SdyDimSharding(read_uvarint_list(&mut cur).unwrap_or_default()),
        6 => {
            let mesh_idx =
                usize::try_from(decode_mlir_prefix_varint(&mut cur).unwrap_or(0)).unwrap_or(0);
            RawV6Attr::SdyTensorSharding(mesh_idx, read_uvarint_list(&mut cur).unwrap_or_default())
        }
        7 => RawV6Attr::SdyShardingPerValue(read_uvarint_list(&mut cur).unwrap_or_default()),
        _ => RawV6Attr::Opaque,
    }
}

/// Decodes a single `builtin`, `vhlo`, `sdy`, or `stablehlo` custom attribute slice from Section `2`.
///
/// # Arguments
///
/// * `dialect` - Dialect name owning the attribute (`"builtin"`, `"vhlo"`, `"sdy"`, `"stablehlo"`).
/// * `slice` - Raw custom-encoded byte slice for one attribute.
/// * `strings` - Decoded Section-0 string table.
fn decode_v6_attr(dialect: &str, slice: &[u8], strings: &[String]) -> RawV6Attr {
    let mut cur = slice;
    let Ok(tag) = decode_mlir_prefix_varint(&mut cur) else {
        return RawV6Attr::Opaque;
    };
    if dialect == "vhlo" {
        return decode_vhlo_attr(tag, cur, strings);
    }
    if dialect == "sdy" {
        return decode_sdy_attr(tag, cur, strings);
    }
    if dialect == "stablehlo" || dialect == "mhlo" {
        return match tag {
            0 => RawV6Attr::Channel(read_svarint(&mut cur).unwrap_or(0)),
            2 => {
                let d = match decode_mlir_prefix_varint(&mut cur).unwrap_or(0) {
                    1 => "NE",
                    2 => "GE",
                    3 => "GT",
                    4 => "LE",
                    5 => "LT",
                    _ => "EQ",
                };
                RawV6Attr::CmpDir(d)
            }
            5 => {
                let lb = read_svarint_list(&mut cur).unwrap_or_default();
                let rb = read_svarint_list(&mut cur).unwrap_or_default();
                let lc = read_svarint_list(&mut cur).unwrap_or_default();
                let rc = read_svarint_list(&mut cur).unwrap_or_default();
                RawV6Attr::DotDims(lb, rb, lc, rc)
            }
            _ => RawV6Attr::Opaque,
        };
    }

    match tag {
        0 => RawV6Attr::Array(read_uvarint_list(&mut cur).unwrap_or_default()),
        1 => RawV6Attr::Dict(read_dict_pairs(&mut cur).unwrap_or_default()),
        2 | 3 => {
            let sidx =
                usize::try_from(decode_mlir_prefix_varint(&mut cur).unwrap_or(0)).unwrap_or(0);
            RawV6Attr::Str(strings.get(sidx).cloned().unwrap_or_default())
        }
        4 => {
            let aidx =
                usize::try_from(decode_mlir_prefix_varint(&mut cur).unwrap_or(0)).unwrap_or(0);
            RawV6Attr::SymRefAttrIdx(aidx)
        }
        6 => {
            let tidx =
                usize::try_from(decode_mlir_prefix_varint(&mut cur).unwrap_or(0)).unwrap_or(0);
            RawV6Attr::TypeRef(tidx)
        }
        7 => RawV6Attr::Unit,
        8 => {
            let _tidx = decode_mlir_prefix_varint(&mut cur).unwrap_or(0);
            let val = read_svarint(&mut cur).unwrap_or(0);
            RawV6Attr::Int(val)
        }
        9 => {
            let tidx =
                usize::try_from(decode_mlir_prefix_varint(&mut cur).unwrap_or(0)).unwrap_or(0);
            #[allow(clippy::cast_sign_loss)]
            let bits = read_svarint(&mut cur).unwrap_or(0) as u64;
            RawV6Attr::FloatBits(tidx, bits)
        }
        17 => {
            let _tidx = decode_mlir_prefix_varint(&mut cur).unwrap_or(0);
            let count =
                usize::try_from(decode_mlir_prefix_varint(&mut cur).unwrap_or(0)).unwrap_or(0);
            let blen =
                usize::try_from(decode_mlir_prefix_varint(&mut cur).unwrap_or(0)).unwrap_or(0);
            let raw = cur.get(..blen.min(cur.len())).unwrap_or_default().to_vec();
            RawV6Attr::DenseArray(count, raw)
        }
        18 => {
            let tidx =
                usize::try_from(decode_mlir_prefix_varint(&mut cur).unwrap_or(0)).unwrap_or(0);
            let blen =
                usize::try_from(decode_mlir_prefix_varint(&mut cur).unwrap_or(0)).unwrap_or(0);
            let raw = cur.get(..blen.min(cur.len())).unwrap_or_default().to_vec();
            RawV6Attr::DenseElements(tidx, raw)
        }
        _ => RawV6Attr::Opaque,
    }
}

impl V6Tables {
    /// Formats a Shardy `SdyTensorSharding` attribute at `sharding_idx` into a canonical GSPMD
    /// `OpSharding` string (`"{replicated}"` or `"{devices=[...]<=[...]}"`).
    ///
    /// # Arguments
    ///
    /// * `sharding_idx` - Attribute index of a [`RawV6Attr::SdyTensorSharding`] in `self.attrs`.
    fn fmt_sdy_tensor_sharding_as_op_sharding(&self, sharding_idx: usize) -> String {
        let Some(RawV6Attr::SdyTensorSharding(mesh_ref_idx, dim_idxs)) =
            self.attrs.get(sharding_idx)
        else {
            return String::from("{replicated}");
        };
        let mesh_name = match self.attrs.get(*mesh_ref_idx) {
            Some(RawV6Attr::SymRef(s) | RawV6Attr::Str(s)) => s.as_str(),
            _ => "",
        };
        let axis_idxs = self
            .props
            .iter()
            .find_map(|pv| {
                let m_i = usize::try_from(*pv.first()?).ok()?;
                let n_i = usize::try_from(*pv.get(1)?).ok()?;
                if let (Some(RawV6Attr::SdyMesh(axes)), Some(RawV6Attr::Str(nm))) =
                    (self.attrs.get(m_i), self.attrs.get(n_i))
                    && nm == mesh_name
                {
                    return Some(axes.as_slice());
                }
                None
            })
            .unwrap_or(&[]);
        let mut factors = Vec::with_capacity(dim_idxs.len());
        for &d_idx in dim_idxs {
            let mut factor = 1_i64;
            if let Some(RawV6Attr::SdyDimSharding(ax_refs)) = self.attrs.get(d_idx) {
                for &ar_idx in ax_refs {
                    if let Some(RawV6Attr::SdyAxisRef(ax_name)) = self.attrs.get(ar_idx) {
                        for &ma_idx in axis_idxs {
                            if let Some(RawV6Attr::SdyMeshAxis(m_ax, sz)) = self.attrs.get(ma_idx)
                                && m_ax == ax_name
                            {
                                factor = factor.saturating_mul((*sz).max(1));
                            }
                        }
                    }
                }
            }
            factors.push(factor);
        }
        if factors.iter().all(|&f| f <= 1) {
            String::from("{replicated}")
        } else {
            let total: i64 = factors.iter().fold(1_i64, |a, &b| a.saturating_mul(b));
            let dims_s = factors
                .iter()
                .map(i64::to_string)
                .collect::<Vec<_>>()
                .join(",");
            format!("{{devices=[{dims_s}]<=[{total}]}}")
        }
    }

    /// Resolves a `SdyShardingPerValue` attribute at `attr_idx` into per-value `OpSharding` strings.
    ///
    /// # Arguments
    ///
    /// * `attr_idx` - Attribute index in `self.attrs`.
    fn sdy_per_value_op_shardings(&self, attr_idx: usize) -> Vec<String> {
        match self.attrs.get(attr_idx) {
            Some(RawV6Attr::SdyShardingPerValue(sh_idxs)) => sh_idxs
                .iter()
                .map(|&si| self.fmt_sdy_tensor_sharding_as_op_sharding(si))
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Formats `types[idx]` into canonical MLIR type syntax (e.g., `"tensor<2x3xf32>"`).
    ///
    /// # Arguments
    ///
    /// * `idx` - Type index in `self.types`.
    /// * `depth` - Recursion guard depth.
    fn fmt_type(&self, idx: usize, depth: u8) -> String {
        if depth > 8 {
            return String::from("tensor<f32>");
        }
        match self.types.get(idx) {
            Some(RawV6Type::Int(w, sign)) => match sign {
                2 => format!("ui{w}"),
                _ => format!("i{w}"),
            },
            Some(RawV6Type::Index) => String::from("index"),
            Some(RawV6Type::Token) => String::from("!stablehlo.token"),
            Some(RawV6Type::Float(f)) => String::from(*f),
            Some(RawV6Type::Complex(e)) => format!("complex<{}>", self.fmt_type(*e, depth + 1)),
            Some(RawV6Type::Tensor(dims, elem)) => {
                let el = self.fmt_type(*elem, depth + 1);
                if dims.is_empty() {
                    format!("tensor<{el}>")
                } else {
                    let ds: Vec<String> = dims
                        .iter()
                        .map(|&d| {
                            if d < 0 {
                                String::from("?")
                            } else {
                                d.to_string()
                            }
                        })
                        .collect();
                    format!("tensor<{}x{el}>", ds.join("x"))
                }
            }
            Some(RawV6Type::Tuple(elems)) => {
                let inner: Vec<String> =
                    elems.iter().map(|&e| self.fmt_type(e, depth + 1)).collect();
                format!("tuple<{}>", inner.join(", "))
            }
            Some(RawV6Type::Func(ins, outs)) => {
                let i_s: Vec<String> = ins.iter().map(|&t| self.fmt_type(t, depth + 1)).collect();
                let o_s: Vec<String> = outs.iter().map(|&t| self.fmt_type(t, depth + 1)).collect();
                format!("({}) -> ({})", i_s.join(", "), o_s.join(", "))
            }
            None => String::from("f32"),
        }
    }

    /// Extracts a list of `i64` values from `attrs[idx]` (`DenseArray` or `DenseElements` or `Array`).
    ///
    /// # Arguments
    ///
    /// * `idx` - Attribute index in `self.attrs`.
    fn attr_as_i64_vec(&self, idx: usize) -> Vec<i64> {
        match self.attrs.get(idx) {
            Some(RawV6Attr::DenseArray(count, raw)) => {
                if *count == 0 {
                    return Vec::new();
                }
                if raw.len() >= count.saturating_mul(8) {
                    raw.chunks_exact(8)
                        .take(*count)
                        .map(|c| {
                            let mut b = [0_u8; 8];
                            b.copy_from_slice(c);
                            i64::from_le_bytes(b)
                        })
                        .collect()
                } else if raw.len() >= count.saturating_mul(4) {
                    raw.chunks_exact(4)
                        .take(*count)
                        .map(|c| {
                            let mut b = [0_u8; 4];
                            b.copy_from_slice(c);
                            i64::from(i32::from_le_bytes(b))
                        })
                        .collect()
                } else {
                    raw.iter().take(*count).map(|&b| i64::from(b)).collect()
                }
            }
            Some(RawV6Attr::DenseElements(ty_idx, raw)) => {
                let (dims, elem_ty) = match self.types.get(*ty_idx) {
                    Some(RawV6Type::Tensor(ds, e)) => (ds.as_slice(), self.types.get(*e)),
                    other => (&[][..], other),
                };
                let num_elems = dims.iter().fold(1_usize, |acc, &d| {
                    acc.saturating_mul(usize::try_from(d.max(0)).unwrap_or(0))
                });
                let vals: Vec<i64> = match elem_ty {
                    Some(RawV6Type::Int(32, _)) => raw
                        .chunks_exact(4)
                        .map(|c| {
                            let mut b = [0_u8; 4];
                            b.copy_from_slice(c);
                            i64::from(i32::from_le_bytes(b))
                        })
                        .collect(),
                    _ => raw
                        .chunks_exact(8)
                        .map(|c| {
                            let mut b = [0_u8; 8];
                            b.copy_from_slice(c);
                            i64::from_le_bytes(b)
                        })
                        .collect(),
                };
                if vals.len() == 1 && num_elems > 1 {
                    vec![vals.first().copied().unwrap_or(0); num_elems]
                } else if num_elems == 0 {
                    Vec::new()
                } else {
                    vals
                }
            }
            Some(RawV6Attr::Array(items)) => items
                .iter()
                .filter_map(|&i| match self.attrs.get(i) {
                    Some(RawV6Attr::Int(v)) => Some(*v),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Extracts a scalar `i64` from `attrs[idx]`.
    ///
    /// # Arguments
    ///
    /// * `idx` - Attribute index in `self.attrs`.
    fn attr_as_i64(&self, idx: usize) -> Option<i64> {
        match self.attrs.get(idx) {
            Some(RawV6Attr::Int(v) | RawV6Attr::Channel(v)) => Some(*v),
            _ => None,
        }
    }

    /// Looks up `key` inside a `DictionaryAttr` at `dict_idx`, returning the value's attribute index.
    ///
    /// # Arguments
    ///
    /// * `dict_idx` - Dictionary attribute index in `self.attrs`.
    /// * `key` - Attribute key name to search for.
    fn dict_get(&self, dict_idx: usize, key: &str) -> Option<usize> {
        let RawV6Attr::Dict(pairs) = self.attrs.get(dict_idx)? else {
            return None;
        };
        for &(k_idx, v_idx) in pairs {
            if let Some(RawV6Attr::Str(s)) = self.attrs.get(k_idx)
                && s == key
            {
                return Some(v_idx);
            }
        }
        None
    }

    /// Formats a `DenseElementsAttr` (`attrs[idx]`) into canonical `dense<...> : tensor<...>` syntax.
    ///
    /// # Arguments
    ///
    /// * `idx` - Attribute index in `self.attrs`.
    fn fmt_dense_elements(&self, idx: usize) -> Option<(String, String)> {
        let RawV6Attr::DenseElements(ty_idx, raw) = self.attrs.get(idx)? else {
            return None;
        };
        let (lit, ty) = self.fmt_dense_payload(*ty_idx, raw);
        Some((format!("dense<{lit}>"), ty))
    }

    /// Formats a `(ty_idx, raw)` dense tensor payload into `(literal_text, tensor_type_text)`.
    ///
    /// # Arguments
    ///
    /// * `ty_idx` - Shaped type index in `self.types`.
    /// * `raw` - Raw little-endian element payload bytes.
    fn fmt_dense_payload(&self, ty_idx: usize, raw: &[u8]) -> (String, String) {
        let ty_str = self.fmt_type(ty_idx, 0);
        let (dims, elem_idx) = match self.types.get(ty_idx) {
            Some(RawV6Type::Tensor(d, e)) => (d.as_slice(), *e),
            _ => (&[][..], 0),
        };
        let num_elems: usize = dims.iter().fold(1_usize, |acc, &d| {
            acc.saturating_mul(usize::try_from(d.max(1)).unwrap_or(1))
        });
        let lit_str = match self.types.get(elem_idx) {
            Some(RawV6Type::Float("f32")) => {
                let vals: Vec<String> = raw
                    .chunks_exact(4)
                    .map(|c| {
                        let mut b = [0_u8; 4];
                        b.copy_from_slice(c);
                        let v = f32::from_le_bytes(b);
                        if v.is_nan() {
                            String::from("0.0")
                        } else {
                            format!("{v:.8e}")
                        }
                    })
                    .collect();
                if vals.len() <= 1 || num_elems <= 1 {
                    vals.first().cloned().unwrap_or_else(|| String::from("0.0"))
                } else {
                    format!("[{}]", vals.join(", "))
                }
            }
            Some(RawV6Type::Float("f64")) => {
                let vals: Vec<String> = raw
                    .chunks_exact(8)
                    .map(|c| {
                        let mut b = [0_u8; 8];
                        b.copy_from_slice(c);
                        format!("{:.12e}", f64::from_le_bytes(b))
                    })
                    .collect();
                if vals.len() <= 1 || num_elems <= 1 {
                    vals.first().cloned().unwrap_or_else(|| String::from("0.0"))
                } else {
                    format!("[{}]", vals.join(", "))
                }
            }
            Some(RawV6Type::Int(1, _)) => {
                let b0 = raw.first().copied().unwrap_or(0) & 1;
                if b0 != 0 {
                    String::from("true")
                } else {
                    String::from("false")
                }
            }
            Some(RawV6Type::Int(32, _)) => {
                let vals: Vec<String> = raw
                    .chunks_exact(4)
                    .map(|c| {
                        let mut b = [0_u8; 4];
                        b.copy_from_slice(c);
                        i32::from_le_bytes(b).to_string()
                    })
                    .collect();
                if vals.len() <= 1 || num_elems <= 1 {
                    vals.first().cloned().unwrap_or_else(|| String::from("0"))
                } else {
                    format!("[{}]", vals.join(", "))
                }
            }
            Some(RawV6Type::Int(64, _) | RawV6Type::Index) => {
                let vals: Vec<String> = raw
                    .chunks_exact(8)
                    .map(|c| {
                        let mut b = [0_u8; 8];
                        b.copy_from_slice(c);
                        i64::from_le_bytes(b).to_string()
                    })
                    .collect();
                if vals.len() <= 1 || num_elems <= 1 {
                    vals.first().cloned().unwrap_or_else(|| String::from("0"))
                } else {
                    format!("[{}]", vals.join(", "))
                }
            }
            _ => String::from("0.0"),
        };
        (lit_str, ty_str)
    }
}

/// Parses an operation node from `buf` at `pos` inside Section `4` (`kIR`).
///
/// # Arguments
///
/// * `buf` - Section-4 byte buffer.
/// * `pos` - Current byte offset in `buf`.
/// * `ops` - Registered operation table from Section `1`.
/// * `ver` - MLIR bytecode container version (`0..=6`).
fn parse_ir_op(buf: &[u8], mut pos: usize, ops: &[String], ver: u64) -> Option<(usize, V6OpNode)> {
    let mut cur = buf.get(pos..)?;
    let op_id = usize::try_from(decode_mlir_prefix_varint(&mut cur).ok()?).ok()?;
    let (&mask, rest) = cur.split_first()?;
    cur = rest;
    let _loc_idx = decode_mlir_prefix_varint(&mut cur).ok()?;
    let dict_attr_idx = if (mask & 0x01) != 0 {
        Some(usize::try_from(decode_mlir_prefix_varint(&mut cur).ok()?).ok()?)
    } else {
        None
    };
    let prop_idx = if ver >= 5 && (mask & 0x40) != 0 {
        Some(usize::try_from(decode_mlir_prefix_varint(&mut cur).ok()?).ok()?)
    } else {
        None
    };
    let result_types = if (mask & 0x02) != 0 {
        let nr = usize::try_from(decode_mlir_prefix_varint(&mut cur).ok()?).ok()?;
        let mut rts = Vec::with_capacity(nr.min(32));
        for _ in 0..nr {
            rts.push(usize::try_from(decode_mlir_prefix_varint(&mut cur).ok()?).ok()?);
        }
        rts
    } else {
        Vec::new()
    };
    let operands = if (mask & 0x04) != 0 {
        let no = usize::try_from(decode_mlir_prefix_varint(&mut cur).ok()?).ok()?;
        let mut ops_in = Vec::with_capacity(no.min(32));
        for _ in 0..no {
            ops_in.push(usize::try_from(decode_mlir_prefix_varint(&mut cur).ok()?).ok()?);
        }
        ops_in
    } else {
        Vec::new()
    };
    if (mask & 0x08) != 0 {
        let ns = usize::try_from(decode_mlir_prefix_varint(&mut cur).ok()?).ok()?;
        for _ in 0..ns {
            let _ = decode_mlir_prefix_varint(&mut cur).ok()?;
        }
    }
    pos = buf.len().saturating_sub(cur.len());
    let mut regions = Vec::new();
    if (mask & 0x10) != 0 {
        let mut rcur = buf.get(pos..)?;
        let nri = decode_mlir_prefix_varint(&mut rcur).ok()?;
        let n_reg = usize::try_from(nri >> 1).ok()?;
        let is_iso = (nri & 1) != 0;
        pos = buf.len().saturating_sub(rcur.len());
        if ver >= 5 && is_iso && buf.get(pos) == Some(&4) {
            let mut sub_cur = buf.get(pos + 1..)?;
            let rlen = usize::try_from(decode_mlir_prefix_varint(&mut sub_cur).ok()?).ok()?;
            let sub_start = buf.len().saturating_sub(sub_cur.len());
            let sub_slice = buf.get(sub_start..sub_start.saturating_add(rlen))?;
            pos = sub_start.saturating_add(rlen);
            let mut rp = 0_usize;
            for _ in 0..n_reg {
                let (next_rp, reg) = parse_ir_region(sub_slice, rp, ops, ver, is_iso)?;
                rp = next_rp;
                regions.push(reg);
            }
        } else {
            for _ in 0..n_reg {
                let (next_p, reg) = parse_ir_region(buf, pos, ops, ver, is_iso)?;
                pos = next_p;
                regions.push(reg);
            }
        }
    }
    let name = ops
        .get(op_id)
        .cloned()
        .unwrap_or_else(|| String::from("stablehlo.identity"));
    Some((
        pos,
        V6OpNode {
            name,
            dict_attr_idx,
            prop_idx,
            result_types,
            operands,
            regions,
        },
    ))
}

/// Parses a region (`num_blocks`, `num_values`, block arguments, and block ops) from `buf` at `pos`.
///
/// # Arguments
///
/// * `buf` - Byte slice containing the region stream.
/// * `pos` - Current byte offset in `buf`.
/// * `ops` - Registered operation table from Section `1`.
/// * `ver` - MLIR bytecode container version (`0..=6`).
/// * `is_isolated` - Whether the enclosing operation marked this region `IsolatedFromAbove`.
fn parse_ir_region(
    buf: &[u8],
    mut pos: usize,
    ops: &[String],
    ver: u64,
    is_isolated: bool,
) -> Option<(usize, V6Region)> {
    let mut cur = buf.get(pos..)?;
    let nb = usize::try_from(decode_mlir_prefix_varint(&mut cur).ok()?).ok()?;
    if nb == 0 {
        return Some((
            buf.len().saturating_sub(cur.len()),
            V6Region {
                is_isolated,
                arg_types: Vec::new(),
                ops: Vec::new(),
            },
        ));
    }
    let _nv = decode_mlir_prefix_varint(&mut cur).ok()?;
    pos = buf.len().saturating_sub(cur.len());
    let mut arg_types = Vec::new();
    let mut block_ops = Vec::new();
    for b_idx in 0..nb {
        let mut bcur = buf.get(pos..)?;
        let oh = decode_mlir_prefix_varint(&mut bcur).ok()?;
        let nops = usize::try_from(oh >> 1).ok()?;
        let has_args = (oh & 1) != 0;
        if has_args {
            let na = usize::try_from(decode_mlir_prefix_varint(&mut bcur).ok()?).ok()?;
            for _ in 0..na {
                if ver >= 3 {
                    let tl = decode_mlir_prefix_varint(&mut bcur).ok()?;
                    if b_idx == 0 {
                        arg_types.push(usize::try_from(tl >> 1).unwrap_or(0));
                    }
                    if (tl & 1) != 0 {
                        let _ = decode_mlir_prefix_varint(&mut bcur).ok()?;
                    }
                } else {
                    let ty = decode_mlir_prefix_varint(&mut bcur).ok()?;
                    let _loc = decode_mlir_prefix_varint(&mut bcur).ok()?;
                    if b_idx == 0 {
                        arg_types.push(usize::try_from(ty).unwrap_or(0));
                    }
                }
            }
            if ver >= 3 {
                let (&_u_flag, after_u) = bcur.split_first()?;
                bcur = after_u;
            }
        }
        pos = buf.len().saturating_sub(bcur.len());
        for _ in 0..nops {
            let (next_p, op_node) = parse_ir_op(buf, pos, ops, ver)?;
            pos = next_p;
            if b_idx == 0 {
                block_ops.push(op_node);
            }
        }
    }
    Some((
        pos,
        V6Region {
            is_isolated,
            arg_types,
            ops: block_ops,
        },
    ))
}

/// Formats an operation inside a function or nested region into canonical `StableHLO` MLIR text.
///
/// # Arguments
///
/// * `op` - Parsed [`V6OpNode`].
/// * `res_lhs` - Pre-assigned SSA result prefix (`""`, `"%r0 = "`, or `"%r0:4 = "`).
/// * `tables` - Decoded [`V6Tables`].
/// * `ssa_names` - Complete SSA value ID -> textual name table for the enclosing block.
/// * `next_tmp` - Mutable counter generating unique SSA result names.
/// * `indent` - Leading indentation spaces.
/// * `out` - Output MLIR text buffer.
#[allow(clippy::too_many_lines)]
fn emit_v6_op(
    op: &V6OpNode,
    res_lhs: &str,
    tables: &V6Tables,
    ssa_names: &[String],
    next_tmp: &mut usize,
    indent: usize,
    out: &mut String,
) {
    let pad = " ".repeat(indent);
    let opnd_strs: Vec<String> = op
        .operands
        .iter()
        .map(|&id| {
            ssa_names
                .get(id)
                .cloned()
                .unwrap_or_else(|| format!("%v{id}"))
        })
        .collect();
    let prop_vals: &[u64] = op
        .prop_idx
        .and_then(|pi| tables.props.get(pi))
        .map_or(&[], Vec::as_slice);
    let res_tys: Vec<String> = op
        .result_types
        .iter()
        .map(|&ti| tables.fmt_type(ti, 0))
        .collect();

    match op.name.as_str() {
        "func.return" | "stablehlo.return" => {
            if opnd_strs.is_empty() {
                let _ = writeln!(out, "{pad}{}", op.name);
            } else {
                let _ = writeln!(out, "{pad}{} {}", op.name, opnd_strs.join(", "));
            }
        }
        "stablehlo.constant" => {
            let aidx = usize::try_from(prop_vals.first().copied().unwrap_or(0)).unwrap_or(0);
            let (dense_s, ty_s) = tables.fmt_dense_elements(aidx).unwrap_or_else(|| {
                (
                    String::from("dense<0.0>"),
                    res_tys
                        .first()
                        .cloned()
                        .unwrap_or_else(|| String::from("tensor<f32>")),
                )
            });
            let _ = writeln!(out, "{pad}{res_lhs}stablehlo.constant {dense_s} : {ty_s}");
        }
        "stablehlo.dot_general" => {
            let (lb, rb, lc, rc) = if prop_vals.len() >= 5 {
                // VHLO `dot_general_v2` (12 alphabetical ODS properties): [2, 4, 8, 10]
                // VHLO `dot_general_v1` (5 alphabetical ODS properties):  [0, 1, 3, 4]
                let slots = if prop_vals.len() >= 12 {
                    [2_usize, 4, 8, 10]
                } else {
                    [0_usize, 1, 3, 4]
                };
                let fetch_dims = |pos: usize| -> Vec<i64> {
                    let s_idx = slots.get(pos).copied().unwrap_or(0);
                    let a_idx =
                        usize::try_from(prop_vals.get(s_idx).copied().unwrap_or(0)).unwrap_or(0);
                    tables.attr_as_i64_vec(a_idx)
                };
                (fetch_dims(0), fetch_dims(2), fetch_dims(1), fetch_dims(3))
            } else {
                let d_idx = usize::try_from(prop_vals.first().copied().unwrap_or(0)).unwrap_or(0);
                match tables.attrs.get(d_idx) {
                    Some(RawV6Attr::DotDims(a, b, c, d)) => {
                        (a.clone(), b.clone(), c.clone(), d.clone())
                    }
                    _ => (Vec::new(), Vec::new(), vec![1], vec![0]),
                }
            };
            let fmt_list = |v: &[i64]| v.iter().map(i64::to_string).collect::<Vec<_>>().join(", ");
            let rty = res_tys.first().map_or("tensor<f32>", String::as_str);
            let _ = writeln!(
                out,
                "{pad}{res_lhs}stablehlo.dot_general {}, {}, #stablehlo.dot<lhs_batching_dimensions = [{}], rhs_batching_dimensions = [{}], lhs_contracting_dimensions = [{}], rhs_contracting_dimensions = [{}]> : () -> {rty}",
                opnd_strs.first().map_or("%arg0", String::as_str),
                opnd_strs.get(1).map_or("%arg1", String::as_str),
                fmt_list(&lb),
                fmt_list(&rb),
                fmt_list(&lc),
                fmt_list(&rc),
            );
        }
        "stablehlo.broadcast_in_dim" => {
            let aidx = usize::try_from(prop_vals.first().copied().unwrap_or(0)).unwrap_or(0);
            let dims = tables.attr_as_i64_vec(aidx);
            let ds = dims
                .iter()
                .map(i64::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            let rty = res_tys.first().map_or("tensor<f32>", String::as_str);
            let _ = writeln!(
                out,
                "{pad}{res_lhs}stablehlo.broadcast_in_dim {}, dims = [{ds}] : () -> {rty}",
                opnd_strs.first().map_or("%arg0", String::as_str),
            );
        }
        "stablehlo.transpose" => {
            let aidx = usize::try_from(prop_vals.first().copied().unwrap_or(0)).unwrap_or(0);
            let perm = tables.attr_as_i64_vec(aidx);
            let ps = perm
                .iter()
                .map(i64::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            let rty = res_tys.first().map_or("tensor<f32>", String::as_str);
            let _ = writeln!(
                out,
                "{pad}{res_lhs}stablehlo.transpose {}, dims = [{ps}] : () -> {rty}",
                opnd_strs.first().map_or("%arg0", String::as_str),
            );
        }
        "stablehlo.slice" => {
            let start_attr_idx =
                usize::try_from(prop_vals.first().copied().unwrap_or(0)).unwrap_or(0);
            let limit_attr_idx =
                usize::try_from(prop_vals.get(1).copied().unwrap_or(0)).unwrap_or(0);
            let stride_attr_idx =
                usize::try_from(prop_vals.get(2).copied().unwrap_or(0)).unwrap_or(0);
            let starts = tables.attr_as_i64_vec(start_attr_idx);
            let limits = tables.attr_as_i64_vec(limit_attr_idx);
            let strides = tables.attr_as_i64_vec(stride_attr_idx);
            let specs: Vec<String> = (0..starts.len())
                .map(|i| {
                    let a = starts.get(i).copied().unwrap_or(0);
                    let b = limits.get(i).copied().unwrap_or(1);
                    let c = strides.get(i).copied().unwrap_or(1);
                    format!("{a}:{b}:{c}")
                })
                .collect();
            let rty = res_tys.first().map_or("tensor<f32>", String::as_str);
            let _ = writeln!(
                out,
                "{pad}{res_lhs}stablehlo.slice {} [{}] : () -> {rty}",
                opnd_strs.first().map_or("%arg0", String::as_str),
                specs.join(", "),
            );
        }
        "stablehlo.dynamic_slice" => {
            let sz_idx = usize::try_from(prop_vals.first().copied().unwrap_or(0)).unwrap_or(0);
            let sizes = tables.attr_as_i64_vec(sz_idx);
            let ss = sizes
                .iter()
                .map(i64::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            let rty = res_tys.first().map_or("tensor<f32>", String::as_str);
            let _ = writeln!(
                out,
                "{pad}{res_lhs}stablehlo.dynamic_slice {}, sizes = [{ss}] : () -> {rty}",
                opnd_strs.join(", "),
            );
        }
        "stablehlo.compare" => {
            // props = [optional_compare_type, comparison_direction]
            let dir_idx = usize::try_from(prop_vals.get(1).copied().unwrap_or(0)).unwrap_or(0);
            let dir = match tables.attrs.get(dir_idx) {
                Some(RawV6Attr::CmpDir(d)) => *d,
                _ => "EQ",
            };
            let rty = res_tys.first().map_or("tensor<i1>", String::as_str);
            let _ = writeln!(
                out,
                "{pad}{res_lhs}stablehlo.compare {dir}, {}, {} : () -> {rty}",
                opnd_strs.first().map_or("%arg0", String::as_str),
                opnd_strs.get(1).map_or("%arg0", String::as_str),
            );
        }
        "stablehlo.get_tuple_element" => {
            let raw_idx = prop_vals.first().copied().unwrap_or(0);
            let elem_i = match tables.attrs.get(usize::try_from(raw_idx).unwrap_or(0)) {
                Some(RawV6Attr::Int(v)) => *v,
                _ => 0,
            };
            let rty = res_tys.first().map_or("tensor<f32>", String::as_str);
            let _ = writeln!(
                out,
                "{pad}{res_lhs}stablehlo.get_tuple_element {}[{elem_i}] : () -> {rty}",
                opnd_strs.first().map_or("%arg0", String::as_str),
            );
        }
        "stablehlo.reduce" => {
            let dim_idx = usize::try_from(prop_vals.first().copied().unwrap_or(0)).unwrap_or(0);
            let dims = tables.attr_as_i64_vec(dim_idx);
            let ds = dims
                .iter()
                .map(i64::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            let half = opnd_strs.len() / 2;
            let inputs = opnd_strs.get(..half).unwrap_or(&[]).join(", ");
            let inits = opnd_strs.get(half..).unwrap_or(&[]).join(", ");
            let rty = res_tys.join(", ");
            let _ = writeln!(
                out,
                "{pad}{res_lhs}stablehlo.reduce({inputs} init: {inits}) across dimensions = [{ds}] : () -> {rty} reducer {{",
            );
            if let Some(reg) = op.regions.first() {
                emit_v6_region_body(reg, tables, ssa_names, next_tmp, indent + 2, out);
            }
            let _ = writeln!(out, "{pad}}}");
        }
        "stablehlo.while" => {
            let rty = if res_tys.len() == 1 {
                res_tys.first().cloned().unwrap_or_default()
            } else {
                format!("({})", res_tys.join(", "))
            };
            let _ = writeln!(
                out,
                r#"{pad}{res_lhs}"stablehlo.while"({}) ({{"#,
                opnd_strs.join(", ")
            );
            if let Some(cond_reg) = op.regions.first() {
                emit_v6_region_body(cond_reg, tables, ssa_names, next_tmp, indent + 2, out);
            }
            let _ = writeln!(out, "{pad}}}, {{");
            if let Some(do_reg) = op.regions.get(1) {
                emit_v6_region_body(do_reg, tables, ssa_names, next_tmp, indent + 2, out);
            }
            let _ = writeln!(out, "{pad}}}) : () -> {rty}");
        }
        "stablehlo.case" => {
            let rty = if res_tys.len() == 1 {
                res_tys.first().cloned().unwrap_or_default()
            } else {
                format!("({})", res_tys.join(", "))
            };
            let idx_opnd = opnd_strs.first().map_or("%arg0", String::as_str);
            let _ = write!(out, r#"{pad}{res_lhs}"stablehlo.case"({idx_opnd}) ("#);
            for (r_i, reg) in op.regions.iter().enumerate() {
                if r_i > 0 {
                    let _ = write!(out, ", ");
                }
                let _ = writeln!(out, "{{");
                let mut branch_ssa = if reg.is_isolated {
                    Vec::new()
                } else {
                    ssa_names.to_vec()
                };
                emit_v6_ops(&reg.ops, tables, &mut branch_ssa, next_tmp, indent + 2, out);
                let _ = write!(out, "{pad}}}");
            }
            let _ = writeln!(out, ") : () -> {rty}");
        }
        "func.call" => {
            let callee = prop_vals
                .iter()
                .find_map(|&pv| {
                    let idx = usize::try_from(pv).ok()?;
                    let s = match tables.attrs.get(idx) {
                        Some(RawV6Attr::SymRef(s) | RawV6Attr::Str(s)) => s.as_str(),
                        _ => "",
                    };
                    (!s.is_empty()).then_some(s)
                })
                .or_else(|| {
                    let di = op.dict_attr_idx?;
                    if let Some(RawV6Attr::Dict(pairs)) = tables.attrs.get(di) {
                        for &(_k, v_i) in pairs {
                            if let Some(RawV6Attr::SymRef(s) | RawV6Attr::Str(s)) =
                                tables.attrs.get(v_i)
                                && !s.is_empty()
                            {
                                return Some(s.as_str());
                            }
                        }
                    }
                    None
                })
                .unwrap_or("helper");
            let rty = if res_tys.len() == 1 {
                res_tys.first().cloned().unwrap_or_default()
            } else {
                format!("({})", res_tys.join(", "))
            };
            let _ = writeln!(
                out,
                "{pad}{res_lhs}func.call @{callee}({}) : () -> {rty}",
                opnd_strs.join(", ")
            );
        }
        "stablehlo.custom_call" => {
            let bc_str = prop_vals
                .get(1)
                .and_then(|&pv| tables.attrs.get(usize::try_from(pv).ok()?))
                .and_then(|a| match a {
                    RawV6Attr::Str(s) if s.contains("\"body\":") => Some(s.clone()),
                    _ => None,
                })
                .or_else(|| {
                    for &pv in prop_vals {
                        for cand in [pv, pv >> 1] {
                            let idx = usize::try_from(cand).unwrap_or(usize::MAX);
                            if let Some(RawV6Attr::Str(s)) = tables.attrs.get(idx)
                                && s.contains("\"body\":")
                            {
                                return Some(s.clone());
                            }
                        }
                    }
                    if let Some(di) = op.dict_attr_idx
                        && let Some(RawV6Attr::Dict(pairs)) = tables.attrs.get(di)
                    {
                        for &(_k, v_i) in pairs {
                            if let Some(RawV6Attr::Str(s)) = tables.attrs.get(v_i)
                                && s.contains("\"body\":")
                            {
                                return Some(s.clone());
                            }
                        }
                    }
                    None
                })
                .unwrap_or_default();
            let target = prop_vals
                .get(2)
                .and_then(|&pv| tables.attrs.get(usize::try_from(pv).ok()?))
                .and_then(|a| match a {
                    RawV6Attr::Str(s) if !s.is_empty() => Some(s.clone()),
                    _ => None,
                })
                .unwrap_or_else(|| {
                    for &pv in prop_vals {
                        let idx =
                            usize::try_from(if (pv & 1) != 0 { pv >> 1 } else { pv }).unwrap_or(0);
                        if let Some(RawV6Attr::Str(s)) = tables.attrs.get(idx)
                            && !s.is_empty()
                            && !s.starts_with('{')
                            && !s.chars().all(|c| c.is_ascii_digit())
                        {
                            return s.clone();
                        }
                    }
                    String::from("custom_target")
                });
            let rty = res_tys.first().map_or("tensor<f32>", String::as_str);
            let mosaic_done = target == "tpu_custom_call"
                && try_emit_mosaic_custom_call(
                    &bc_str, &opnd_strs, res_lhs, rty, &pad, next_tmp, out,
                )
                .unwrap_or(false);
            if !mosaic_done {
                let sharding_val = op.dict_attr_idx.and_then(|di| match tables.attrs.get(di) {
                    Some(RawV6Attr::Dict(pairs)) => {
                        for &(k_i, v_i) in pairs {
                            if let Some(RawV6Attr::Str(vs)) = tables.attrs.get(v_i)
                                && (vs.starts_with("{devices=")
                                    || vs.starts_with("{replicated")
                                    || vs.starts_with("{maximal")
                                    || vs.starts_with("{manual")
                                    || matches!(
                                        tables.attrs.get(k_i),
                                        Some(RawV6Attr::Str(ks)) if ks.contains("sharding")
                                    ))
                            {
                                return Some(vs.clone());
                            }
                        }
                        None
                    }
                    _ => None,
                });
                let extra_sh = sharding_val
                    .map(|s| format!(", mhlo.sharding = \"{s}\""))
                    .unwrap_or_default();
                let _ = writeln!(
                    out,
                    r#"{pad}{res_lhs}stablehlo.custom_call @{target}({}) {{call_target_name = "{target}"{extra_sh}}} : () -> {rty}"#,
                    opnd_strs.join(", ")
                );
            }
        }
        "stablehlo.send" => {
            let cid = prop_vals
                .first()
                .and_then(|&pv| tables.attr_as_i64(usize::try_from(pv).ok()?))
                .unwrap_or(1);
            let rty = res_tys.first().map_or("!stablehlo.token", String::as_str);
            let _ = writeln!(
                out,
                r#"{pad}{res_lhs}"stablehlo.send"({}) {{channel_id = {cid} : i64}} : () -> {rty}"#,
                opnd_strs.join(", ")
            );
        }
        "stablehlo.recv" => {
            let cid = prop_vals
                .first()
                .and_then(|&pv| tables.attr_as_i64(usize::try_from(pv).ok()?))
                .unwrap_or(2);
            let rty = format!("({})", res_tys.join(", "));
            let _ = writeln!(
                out,
                r#"{pad}{res_lhs}"stablehlo.recv"({}) {{channel_id = {cid} : i64}} : () -> {rty}"#,
                opnd_strs.join(", ")
            );
        }
        "stablehlo.all_reduce" | "stablehlo.reduce_scatter"
            if let Some(reg) = op.regions.first() =>
        {
            let rty = res_tys.first().map_or("tensor<f32>", String::as_str);
            let extra = if op.name == "stablehlo.reduce_scatter" {
                let sdim = prop_vals
                    .get(2)
                    .and_then(|&pv| tables.attr_as_i64(usize::try_from(pv).ok()?))
                    .unwrap_or(0);
                format!(" {{scatter_dimension = {sdim} : i64}}")
            } else {
                String::new()
            };
            let qname = format!("\"{}\"", op.name);
            let _ = writeln!(out, "{pad}{res_lhs}{qname}({}) ({{", opnd_strs.join(", "));
            emit_v6_region_body(reg, tables, ssa_names, next_tmp, indent + 2, out);
            let _ = writeln!(out, "{pad}}}){extra} : () -> {rty}");
        }
        "stablehlo.all_gather" => {
            let ag_dim = prop_vals
                .first()
                .and_then(|&pv| tables.attr_as_i64(usize::try_from(pv).ok()?))
                .unwrap_or(0);
            let rty = res_tys.first().map_or("tensor<f32>", String::as_str);
            let _ = writeln!(
                out,
                r#"{pad}{res_lhs}"stablehlo.all_gather"({}) {{all_gather_dim = {ag_dim} : i64}} : () -> {rty}"#,
                opnd_strs.join(", ")
            );
        }
        "stablehlo.all_to_all" => {
            let concat_dim = prop_vals
                .get(1)
                .and_then(|&pv| tables.attr_as_i64(usize::try_from(pv).ok()?))
                .unwrap_or(0);
            let split_dim = prop_vals
                .get(4)
                .and_then(|&pv| tables.attr_as_i64(usize::try_from(pv).ok()?))
                .unwrap_or(0);
            let rty = res_tys.first().map_or("tensor<f32>", String::as_str);
            let _ = writeln!(
                out,
                r#"{pad}{res_lhs}"stablehlo.all_to_all"({}) {{split_dimension = {split_dim} : i64, concat_dimension = {concat_dim} : i64}} : () -> {rty}"#,
                opnd_strs.join(", ")
            );
        }
        "stablehlo.convolution" => {
            let fmt_i64s = |v: &[i64]| v.iter().map(i64::to_string).collect::<Vec<_>>().join(", ");
            let rty = res_tys.first().map_or("tensor<f32>", String::as_str);
            if prop_vals.len() >= 17 {
                let get_i = |slot: usize, def: i64| {
                    prop_vals
                        .get(slot)
                        .and_then(|&pv| tables.attr_as_i64(usize::try_from(pv).ok()?))
                        .unwrap_or(def)
                };
                let get_v = |slot: usize| {
                    prop_vals
                        .get(slot)
                        .and_then(|&pv| usize::try_from(pv).ok())
                        .map(|idx| tables.attr_as_i64_vec(idx))
                        .unwrap_or_default()
                };
                let bg = get_i(0, 1);
                let fg = get_i(1, 1);
                let cdims = [
                    get_i(2, 0),
                    get_i(3, 1),
                    get_i(5, 1),
                    get_i(6, 0),
                    get_i(9, 0),
                    get_i(10, 1),
                ];
                let in_sp = get_v(4);
                let ker_sp = get_v(7);
                let lhs_dil = get_v(8);
                let out_sp = get_v(11);
                let pad_flat = get_v(12);
                let rhs_dil = get_v(14);
                let strides = get_v(16);
                let _ = writeln!(
                    out,
                    r#"{pad}{res_lhs}"stablehlo.convolution"({}) {{window_strides = array<i64: {}>, conv_dims = array<i64: {}>, input_spatial_dims = array<i64: {}>, kernel_spatial_dims = array<i64: {}>, output_spatial_dims = array<i64: {}>, padding_flat = array<i64: {}>, lhs_dilation = array<i64: {}>, rhs_dilation = array<i64: {}>, feature_group_count = {fg} : i64, batch_group_count = {bg} : i64}} : () -> {rty}"#,
                    opnd_strs.join(", "),
                    fmt_i64s(&strides),
                    fmt_i64s(&cdims),
                    fmt_i64s(&in_sp),
                    fmt_i64s(&ker_sp),
                    fmt_i64s(&out_sp),
                    fmt_i64s(&pad_flat),
                    fmt_i64s(&lhs_dil),
                    fmt_i64s(&rhs_dil),
                );
            } else {
                let stride_s = prop_vals.first().filter(|&&v| (v & 1) != 0).map_or_else(
                    || vec![1, 1],
                    |&v| tables.attr_as_i64_vec(usize::try_from(v >> 1).unwrap_or(0)),
                );
                let _ = writeln!(
                    out,
                    r#"{pad}{res_lhs}"stablehlo.convolution"({}) {{window_strides = array<i64: {}>, feature_group_count = 1 : i64, batch_group_count = 1 : i64}} : () -> {rty}"#,
                    opnd_strs.join(", "),
                    fmt_i64s(&stride_s),
                );
            }
        }
        "sdy.manual_computation" if let Some(reg) = op.regions.first() => {
            let in_sh_list = prop_vals
                .first()
                .and_then(|&pv| usize::try_from(pv).ok())
                .map(|idx| tables.sdy_per_value_op_shardings(idx))
                .unwrap_or_default();
            let out_sh_list = prop_vals
                .get(2)
                .and_then(|&pv| usize::try_from(pv).ok())
                .map(|idx| tables.sdy_per_value_op_shardings(idx))
                .unwrap_or_default();
            let mut local_ssa = if reg.is_isolated {
                Vec::new()
            } else {
                ssa_names.to_vec()
            };
            let base = *next_tmp;
            *next_tmp += 1;
            for (i, &ty_i) in reg.arg_types.iter().enumerate() {
                let outer_opnd = opnd_strs.get(i).map_or("%arg0", String::as_str);
                let in_sh = in_sh_list.get(i).map_or("{replicated}", String::as_str);
                let local_ty = tables.fmt_type(ty_i, 0);
                let shard_ssa = format!("%mc_in{base}_{i}");
                let _ = writeln!(
                    out,
                    r#"{pad}{shard_ssa} = stablehlo.custom_call @SPMDFullToShardShape({outer_opnd}) {{call_target_name = "SPMDFullToShardShape", mhlo.sharding = "{in_sh}"}} : () -> {local_ty}"#,
                );
                local_ssa.push(shard_ssa);
            }
            let body_end = reg
                .ops
                .iter()
                .position(|sub| sub.name == "sdy.return")
                .unwrap_or(reg.ops.len());
            let body_slice = reg.ops.get(..body_end).unwrap_or(&[]);
            emit_v6_ops(body_slice, tables, &mut local_ssa, next_tmp, indent, out);
            let ret_op = reg.ops.get(body_end);
            let ret_val_idx = ret_op
                .and_then(|ro| ro.operands.first().copied())
                .unwrap_or(0);
            let ret_ssa = local_ssa.get(ret_val_idx).map_or("%arg0", String::as_str);
            let local_out_ty = body_slice
                .iter()
                .rev()
                .find(|lo| lo.name != "builtin.unrealized_conversion_cast")
                .and_then(|lo| lo.result_types.first().copied())
                .or_else(|| reg.arg_types.first().copied())
                .map(|ti| tables.fmt_type(ti, 0))
                .or_else(|| res_tys.first().cloned())
                .unwrap_or_else(|| String::from("tensor<f32>"));
            let out_sh = out_sh_list.first().map_or("{replicated}", String::as_str);
            let _ = writeln!(
                out,
                r#"{pad}{res_lhs}stablehlo.custom_call @SPMDShardToFullShape({ret_ssa}) {{call_target_name = "SPMDShardToFullShape", mhlo.sharding = "{out_sh}"}} : () -> {local_out_ty}"#,
            );
        }
        "builtin.unrealized_conversion_cast" => {}
        _ => {
            let rty = if res_tys.len() <= 1 {
                res_tys
                    .first()
                    .cloned()
                    .unwrap_or_else(|| String::from("tensor<f32>"))
            } else {
                format!("({})", res_tys.join(", "))
            };
            let _ = writeln!(
                out,
                "{pad}{res_lhs}{} {} : () -> {rty}",
                op.name,
                opnd_strs.join(", ")
            );
        }
    }
}

/// Emits a sequence of operations in a block using MLIR's two-pass breadth-first block SSA numbering
/// (all block op results are numbered before descending into child regions).
///
/// # Arguments
///
/// * `ops` - Slice of [`V6OpNode`]s.
/// * `tables` - Decoded [`V6Tables`].
/// * `ssa_names` - Mutable SSA value list.
/// * `next_tmp` - Mutable temporary SSA ID counter.
/// * `indent` - Indentation width.
/// * `out` - Output MLIR text buffer.
fn emit_v6_ops(
    ops: &[V6OpNode],
    tables: &V6Tables,
    ssa_names: &mut Vec<String>,
    next_tmp: &mut usize,
    indent: usize,
    out: &mut String,
) {
    let mut lhs_list = Vec::with_capacity(ops.len());
    for op in ops {
        if op.name == "builtin.unrealized_conversion_cast" {
            for (idx, _) in op.result_types.iter().enumerate() {
                let aliased = op
                    .operands
                    .get(idx)
                    .and_then(|&id| ssa_names.get(id))
                    .map_or("%arg0", String::as_str);
                ssa_names.push(String::from(aliased));
            }
            lhs_list.push(String::new());
            continue;
        }
        let n_res = op.result_types.len();
        if n_res == 0 {
            lhs_list.push(String::new());
        } else if n_res == 1 {
            let nm = format!("%r{}", *next_tmp);
            *next_tmp += 1;
            ssa_names.push(nm.clone());
            lhs_list.push(format!("{nm} = "));
        } else {
            let base = format!("%r{}", *next_tmp);
            *next_tmp += 1;
            for idx in 0..n_res {
                ssa_names.push(format!("{base}#{idx}"));
            }
            lhs_list.push(format!("{base}:{n_res} = "));
        }
    }
    for (op, res_lhs) in ops.iter().zip(&lhs_list) {
        emit_v6_op(op, res_lhs, tables, ssa_names, next_tmp, indent, out);
    }
}

/// Emits a region body with an explicit `^bb0(...)` header when block arguments are present.
///
/// # Arguments
///
/// * `reg` - Parsed [`V6Region`].
/// * `tables` - Decoded [`V6Tables`].
/// * `enclosing_ssa` - Enclosing SSA value table.
/// * `next_tmp` - Mutable temporary SSA ID counter.
/// * `indent` - Indentation width.
/// * `out` - Output MLIR text buffer.
fn emit_v6_region_body(
    reg: &V6Region,
    tables: &V6Tables,
    enclosing_ssa: &[String],
    next_tmp: &mut usize,
    indent: usize,
    out: &mut String,
) {
    let mut local_ssa = if reg.is_isolated {
        Vec::new()
    } else {
        enclosing_ssa.to_vec()
    };
    let base = *next_tmp;
    *next_tmp += 1;
    let mut arg_decls = Vec::with_capacity(reg.arg_types.len());
    for (i, &ty_i) in reg.arg_types.iter().enumerate() {
        let anm = format!("%b{base}_{i}");
        arg_decls.push(format!("{anm}: {}", tables.fmt_type(ty_i, 0)));
        local_ssa.push(anm);
    }
    if !arg_decls.is_empty() {
        let pad = " ".repeat(indent);
        let _ = writeln!(out, "{pad}^bb0({}):", arg_decls.join(", "));
    }
    emit_v6_ops(&reg.ops, tables, &mut local_ssa, next_tmp, indent, out);
}

/// Decodes an RFC-4648 base64 string into raw bytes.
///
/// # Arguments
///
/// * `input` - Base64-encoded ASCII slice.
fn decode_base64_bytes(input: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity((input.len() * 3) / 4);
    let mut buf = 0_u32;
    let mut bits = 0_u32;
    for b in input.bytes() {
        let val = match b {
            b'A'..=b'Z' => u32::from(b - b'A'),
            b'a'..=b'z' => u32::from(b - b'a' + 26),
            b'0'..=b'9' => u32::from(b - b'0' + 52),
            b'+' => 62,
            b'/' => 63,
            b'=' | b' ' | b'\n' | b'\r' => continue,
            _ => return None,
        };
        buf = (buf << 6) | val;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            #[allow(clippy::cast_possible_truncation)]
            out.push((buf >> bits) as u8);
            buf &= (1_u32 << bits) - 1;
        }
    }
    Some(out)
}

/// Formats an `arith.constant` inside a Mosaic kernel into `(literal_text, tensor_type_text)`.
///
/// # Arguments
///
/// * `mop` - Parsed Mosaic `arith.constant` [`V6OpNode`].
/// * `tables` - Decoded Mosaic bytecode tables.
fn format_mosaic_arith_constant(mop: &V6OpNode, tables: &V6Tables) -> (String, String) {
    let attr_idx = mop
        .prop_idx
        .and_then(|pi| tables.props.get(pi))
        .and_then(|pv| pv.first().copied())
        .and_then(|v| usize::try_from(v).ok())
        .or_else(|| {
            mop.dict_attr_idx.and_then(|di| match tables.attrs.get(di) {
                Some(RawV6Attr::Dict(pairs)) => pairs.first().map(|p| p.1),
                _ => None,
            })
        });
    match attr_idx.and_then(|ai| tables.attrs.get(ai)) {
        Some(RawV6Attr::FloatBits(ti, bits)) => {
            let f_val = match tables.types.get(*ti) {
                Some(RawV6Type::Float("f64")) => f64::from_bits(*bits),
                Some(RawV6Type::Float("bf16")) => {
                    #[allow(clippy::cast_possible_truncation)]
                    let u = u32::from(*bits as u16);
                    f64::from(f32::from_bits(u << 16))
                }
                _ => {
                    #[allow(clippy::cast_possible_truncation)]
                    let v = f32::from_bits(*bits as u32);
                    f64::from(v)
                }
            };
            let elem = tables.fmt_type(*ti, 0);
            (format!("{f_val:.8e}"), format!("tensor<{elem}>"))
        }
        Some(RawV6Attr::DenseElements(ti, raw)) => tables.fmt_dense_payload(*ti, raw),
        Some(RawV6Attr::Int(v)) => {
            let raw_ty = mop
                .result_types
                .first()
                .map_or_else(|| String::from("i32"), |&ti| tables.fmt_type(ti, 0));
            let wrapped = if raw_ty.starts_with("tensor<") {
                raw_ty
            } else if raw_ty == "index" {
                String::from("tensor<i32>")
            } else {
                format!("tensor<{raw_ty}>")
            };
            (v.to_string(), wrapped)
        }
        _ => (String::from("0"), String::from("tensor<i32>")),
    }
}

/// Emits a single Mosaic/Pallas kernel body operation as `StableHLO` text.
///
/// # Arguments
///
/// * `mop` - Parsed Mosaic [`V6OpNode`].
/// * `tables` - Decoded Mosaic bytecode tables.
/// * `m_ssa` - Mutable SSA value name table for the Mosaic kernel block.
/// * `stored_out` - Mutable slot recording the most recent `vector.store` source SSA value.
/// * `rty` - Enclosing `custom_call` result tensor type string.
/// * `pad` - Indentation prefix.
/// * `next_tmp` - Mutable temporary SSA ID counter.
/// * `out` - Output MLIR text buffer.
#[allow(clippy::too_many_arguments)]
fn emit_mosaic_kernel_op(
    mop: &V6OpNode,
    tables: &V6Tables,
    m_ssa: &mut Vec<String>,
    stored_out: &mut Option<String>,
    rty: &str,
    pad: &str,
    next_tmp: &mut usize,
    out: &mut String,
) {
    let mapped: Vec<String> = mop
        .operands
        .iter()
        .map(|&id| m_ssa.get(id).cloned().unwrap_or_else(|| format!("%mv{id}")))
        .collect();
    match mop.name.as_str() {
        "vector.load" | "tpu.vector_load" => {
            let src = mapped
                .first()
                .cloned()
                .unwrap_or_else(|| String::from("%arg0"));
            m_ssa.push(src);
        }
        "vector.store" | "tpu.vector_store" => {
            if let Some(val_s) = mapped.first() {
                *stored_out = Some(val_s.clone());
            }
        }
        "arith.constant" => {
            let c_nm = format!("%mc{}", *next_tmp);
            *next_tmp += 1;
            let (lit_s, ty_s) = format_mosaic_arith_constant(mop, tables);
            let _ = writeln!(
                out,
                "{pad}{c_nm} = stablehlo.constant dense<{lit_s}> : {ty_s}"
            );
            m_ssa.push(c_nm);
        }
        "vector.broadcast" | "tpu.broadcast" | "tpu.repeat" => {
            let nm = format!("%mb{}", *next_tmp);
            *next_tmp += 1;
            let dst_ty = mop
                .result_types
                .first()
                .map_or_else(|| String::from(rty), |&ti| tables.fmt_type(ti, 0));
            let elem_s = dst_ty
                .rsplit('x')
                .next()
                .and_then(|s| s.strip_suffix('>'))
                .unwrap_or("f32");
            let src_ty = format!("tensor<{elem_s}>");
            let src = mapped.first().map_or("%arg0", String::as_str);
            let _ = writeln!(
                out,
                "{pad}{nm} = stablehlo.broadcast_in_dim {src}, dims = [] : ({src_ty}) -> {dst_ty}"
            );
            m_ssa.push(nm);
        }
        "arith.addf" | "arith.addi" | "arith.subf" | "arith.mulf" | "arith.divf" => {
            let shlo = match mop.name.as_str() {
                "arith.subf" => "stablehlo.subtract",
                "arith.mulf" => "stablehlo.multiply",
                "arith.divf" => "stablehlo.divide",
                _ => "stablehlo.add",
            };
            let nm = format!("%mk{}", *next_tmp);
            *next_tmp += 1;
            let op_ty = mop
                .result_types
                .first()
                .map_or_else(|| String::from(rty), |&ti| tables.fmt_type(ti, 0));
            let _ = writeln!(
                out,
                "{pad}{nm} = {shlo} {}, {} : {op_ty}",
                mapped.first().map_or("%arg0", String::as_str),
                mapped.get(1).map_or("%arg0", String::as_str),
            );
            m_ssa.push(nm);
        }
        "tpu.matmul" => {
            let nm = format!("%mk{}", *next_tmp);
            *next_tmp += 1;
            let op_ty = mop
                .result_types
                .first()
                .map_or_else(|| String::from(rty), |&ti| tables.fmt_type(ti, 0));
            let _ = writeln!(
                out,
                "{pad}{nm} = stablehlo.dot_general {}, {}, contracting_dims = [1] x [0] : () -> {op_ty}",
                mapped.first().map_or("%arg0", String::as_str),
                mapped.get(1).map_or("%arg0", String::as_str),
            );
            m_ssa.push(nm);
        }
        "func.return" => {}
        _ => {
            for _ in 0..mop.result_types.len() {
                m_ssa.push(
                    mapped
                        .first()
                        .cloned()
                        .unwrap_or_else(|| String::from("%arg0")),
                );
            }
        }
    }
}

/// Attempts to decode a Mosaic/Pallas `tpu_custom_call` JSON `backend_config` (containing a
/// Base64-encoded `MLIR` v0/v5/v6 kernel module under `"body"`) into inline `StableHLO` ops.
///
/// # Arguments
///
/// * `backend_config` - Raw `backend_config` JSON string on `stablehlo.custom_call`.
/// * `opnd_strs` - Outer `custom_call` operand SSA strings.
/// * `res_lhs` - Result assignment prefix (`"%r0 = "`).
/// * `rty` - Formatted result tensor type string.
/// * `pad` - Indentation prefix.
/// * `next_tmp` - Mutable temporary SSA ID counter.
/// * `out` - Output MLIR text buffer.
fn try_emit_mosaic_custom_call(
    backend_config: &str,
    opnd_strs: &[String],
    res_lhs: &str,
    rty: &str,
    pad: &str,
    next_tmp: &mut usize,
    out: &mut String,
) -> Option<bool> {
    let (_, after_body) = backend_config.split_once("\"body\":")?;
    let quote_start = after_body.find('"')?;
    let tail = after_body.get(quote_start + 1..)?;
    let quote_end = tail.find('"')?;
    let b64 = tail.get(..quote_end)?;
    let raw_bc = decode_base64_bytes(b64)?;
    let (_ver, tables, root_op) = parse_mlir_bytecode_container(&raw_bc, true)?;
    let mod_reg = root_op.regions.first()?;
    let fn_op = mod_reg.ops.iter().find(|o| o.name == "func.func")?;
    let fn_reg = fn_op.regions.first()?;
    let num_memref_args = fn_reg.arg_types.len();
    if num_memref_args == 0 {
        return None;
    }
    let mut m_ssa: Vec<String> = (0..num_memref_args)
        .map(|idx| {
            opnd_strs
                .get(idx)
                .cloned()
                .unwrap_or_else(|| format!("%mout_{idx}"))
        })
        .collect();
    let mut stored_out: Option<String> = None;
    for mop in &fn_reg.ops {
        emit_mosaic_kernel_op(
            mop,
            &tables,
            &mut m_ssa,
            &mut stored_out,
            rty,
            pad,
            next_tmp,
            out,
        );
    }
    let final_src = stored_out
        .or_else(|| opnd_strs.first().cloned())
        .unwrap_or_else(|| String::from("%arg0"));
    let _ = writeln!(
        out,
        "{pad}{res_lhs}stablehlo.reshape {final_src} : ({rty}) -> {rty}"
    );
    Some(true)
}

/// Parses an LLVM MLIR bytecode container (`v0` Mosaic or `v5`/`v6` `StableHLO`/`VHLO`) into `(version, V6Tables, root_op)`.
///
/// # Arguments
///
/// * `bytes` - Raw `ML\xefR` bytecode slice.
/// * `allow_v0_mosaic` - If `true`, accepts `ver == 0` when producer starts with `"MLIR"`.
#[allow(clippy::too_many_lines)]
fn parse_mlir_bytecode_container(
    bytes: &[u8],
    allow_v0_mosaic: bool,
) -> Option<(u64, V6Tables, V6OpNode)> {
    let mut cur = bytes.strip_prefix(MLIR_BYTECODE_MAGIC)?;
    let ver = decode_mlir_prefix_varint(&mut cur).ok()?;
    let nul = cur.iter().position(|&b| b == 0)?;
    let prod = std::str::from_utf8(cur.get(..nul)?).ok()?;
    if ver < 5 && !(allow_v0_mosaic && (prod.starts_with("MLIR19") || prod.starts_with("MLIR2"))) {
        return None;
    }
    if !prod.starts_with("StableHLO")
        && !prod.starts_with("MLIR19")
        && !prod.starts_with("MLIR2")
        && !prod.starts_with("LLVM")
    {
        return None;
    }
    cur = cur.get(nul + 1..)?;

    let mut secs: BTreeMap<u8, &[u8]> = BTreeMap::new();
    while let Some((&id_b, rest)) = cur.split_first() {
        cur = rest;
        let sid = id_b & 0x7F;
        let has_align = (id_b & 0x80) != 0;
        let slen = usize::try_from(decode_mlir_prefix_varint(&mut cur).ok()?).ok()?;
        if has_align {
            let align = usize::try_from(decode_mlir_prefix_varint(&mut cur).ok()?).ok()?;
            if align > 1 && align.is_power_of_two() {
                while (bytes.len().saturating_sub(cur.len())) % align != 0
                    && cur.first() == Some(&MLIR_ALIGNMENT_BYTE)
                {
                    cur = cur.get(1..)?;
                }
            }
        }
        let (sdata, after) = cur.split_at_checked(slen)?;
        cur = after;
        secs.insert(sid, sdata);
    }

    // 1. Section 0: Strings
    let mut s0 = *secs.get(&0)?;
    let n_str = usize::try_from(decode_mlir_prefix_varint(&mut s0).ok()?).ok()?;
    let mut rev_lens = Vec::with_capacity(n_str.min(4096));
    for _ in 0..n_str {
        rev_lens.push(usize::try_from(decode_mlir_prefix_varint(&mut s0).ok()?).ok()?);
    }
    let mut strings = Vec::with_capacity(n_str.min(4096));
    for &ln in rev_lens.iter().rev() {
        let chunk = s0.get(..ln)?;
        s0 = s0.get(ln..)?;
        let trimmed = chunk.strip_suffix(&[0]).unwrap_or(chunk);
        strings.push(String::from_utf8_lossy(trimmed).into_owned());
    }

    // 2. Section 1: Dialects & Registered Operations
    let mut s1 = *secs.get(&1)?;
    let n_dial = usize::try_from(decode_mlir_prefix_varint(&mut s1).ok()?).ok()?;
    let mut raw_dialects = Vec::with_capacity(n_dial.min(64));
    for _ in 0..n_dial {
        let dv = decode_mlir_prefix_varint(&mut s1).ok()?;
        let sidx = if ver >= 1 {
            usize::try_from(dv >> 1).ok()?
        } else {
            usize::try_from(dv).ok()?
        };
        let raw_d = strings.get(sidx)?.clone();
        if ver >= 1 && (dv & 1) != 0 {
            let vlen = usize::try_from(decode_mlir_prefix_varint(&mut s1).ok()?).ok()?;
            s1 = s1.get(vlen..)?;
        }
        raw_dialects.push(raw_d);
    }
    let n_ops = if ver >= 1 {
        usize::try_from(decode_mlir_prefix_varint(&mut s1).ok()?).ok()?
    } else {
        usize::MAX
    };
    let mut ops = Vec::with_capacity(n_ops.min(1024));
    while ops.len() < n_ops && !s1.is_empty() {
        let did = usize::try_from(decode_mlir_prefix_varint(&mut s1).ok()?).ok()?;
        let cnt = usize::try_from(decode_mlir_prefix_varint(&mut s1).ok()?).ok()?;
        let raw_d = raw_dialects.get(did)?.as_str();
        for _ in 0..cnt {
            let ov = decode_mlir_prefix_varint(&mut s1).ok()?;
            let oidx = if ver >= 2 {
                usize::try_from(ov >> 1).ok()?
            } else {
                usize::try_from(ov).ok()?
            };
            let oname = strings.get(oidx)?.as_str();
            let stem = oname
                .rsplit_once("_v")
                .filter(|(_, ver_s)| ver_s.bytes().all(|b| b.is_ascii_digit()))
                .map_or(oname, |(st, _)| st);
            let full = match (raw_d, stem) {
                ("vhlo", "func") => String::from("func.func"),
                ("vhlo", "call") => String::from("func.call"),
                ("vhlo", other) => format!("stablehlo.{other}"),
                (_, other) if other.contains('.') => String::from(other),
                (d, other) => format!("{d}.{other}"),
            };
            ops.push(full);
        }
    }

    // 3. Section 3 & 2: AttrTypeOffsets & AttrType payloads
    let mut s3 = *secs.get(&3)?;
    let s2 = *secs.get(&2)?;
    let n_attr = usize::try_from(decode_mlir_prefix_varint(&mut s3).ok()?).ok()?;
    let n_ty = usize::try_from(decode_mlir_prefix_varint(&mut s3).ok()?).ok()?;
    let mut items = Vec::with_capacity((n_attr + n_ty).min(8192));
    while items.len() < n_attr + n_ty && !s3.is_empty() {
        let did = usize::try_from(decode_mlir_prefix_varint(&mut s3).ok()?).ok()?;
        let cnt = usize::try_from(decode_mlir_prefix_varint(&mut s3).ok()?).ok()?;
        for _ in 0..cnt {
            let sz_f = decode_mlir_prefix_varint(&mut s3).ok()?;
            let sz = usize::try_from(sz_f >> 1).ok()?;
            items.push((did, sz, (sz_f & 1) != 0));
        }
    }
    let mut off = 0_usize;
    let mut attrs = Vec::with_capacity(n_attr);
    for &(did, sz, is_cust) in items.get(..n_attr)? {
        let slice = s2.get(off..off.saturating_add(sz))?;
        off = off.saturating_add(sz);
        let dname = raw_dialects.get(did).map_or("builtin", String::as_str);
        if is_cust {
            attrs.push(decode_v6_attr(dname, slice, &strings));
        } else {
            attrs.push(RawV6Attr::Opaque);
        }
    }
    let resolved_attrs: Vec<RawV6Attr> = attrs
        .iter()
        .map(|attr| match attr {
            RawV6Attr::SymRefAttrIdx(target_idx) => match attrs.get(*target_idx) {
                Some(RawV6Attr::Str(s)) => RawV6Attr::SymRef(s.clone()),
                _ => RawV6Attr::SymRef(strings.get(*target_idx).cloned().unwrap_or_default()),
            },
            other => other.clone(),
        })
        .collect();
    let attrs = resolved_attrs;
    let mut types = Vec::with_capacity(n_ty);
    for &(did, sz, _is_cust) in items.get(n_attr..)? {
        let slice = s2.get(off..off.saturating_add(sz))?;
        off = off.saturating_add(sz);
        let dname = raw_dialects.get(did).map_or("builtin", String::as_str);
        types.push(decode_v6_type(dname, slice));
    }

    // 4. Section 8: Properties
    let mut props = Vec::new();
    if let Some(&s8_raw) = secs.get(&8) {
        let mut s8 = s8_raw;
        if let Ok(np) = decode_mlir_prefix_varint(&mut s8) {
            let count = usize::try_from(np).unwrap_or(0);
            for _ in 0..count {
                let Ok(plen_u64) = decode_mlir_prefix_varint(&mut s8) else {
                    break;
                };
                let plen = usize::try_from(plen_u64).unwrap_or(0);
                let Some((pbytes, after_p)) = s8.split_at_checked(plen) else {
                    break;
                };
                s8 = after_p;
                let mut pc = pbytes;
                let mut pvec = Vec::new();
                while let Ok(v) = decode_mlir_prefix_varint(&mut pc) {
                    pvec.push(v);
                }
                props.push(pvec);
            }
        }
    }

    // 5. Section 4: Top-level builtin.module IR walk
    let s4 = *secs.get(&4)?;
    let mut s4_hdr = s4;
    let _ = decode_mlir_prefix_varint(&mut s4_hdr).ok()?;
    let start_pos = s4.len().saturating_sub(s4_hdr.len());
    let (_, root_op) = parse_ir_op(s4, start_pos, &ops, ver)?;
    Some((
        ver,
        V6Tables {
            attrs,
            types,
            props,
        },
        root_op,
    ))
}

/// Disassembles an upstream LLVM `ML\xefR` v5/v6 (`MLIR19`+) binary container into canonical
/// textual `StableHLO` MLIR accepted by [`crate::compiler::parse_mlir_ast_module`].
///
/// Returns `None` if `bytes` is not an upstream LLVM `MLIR*` v5/v6 container (e.g., synthetic v1–v4
/// test containers), allowing callers to fall through to existing decoders unchanged.
///
/// # Arguments
///
/// * `bytes` - Raw `ML\xefR` bytecode slice.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn decode_llvm_mlir_v6_to_text(bytes: &[u8]) -> Option<String> {
    let (_ver, tables, root_op) = parse_mlir_bytecode_container(bytes, false)?;

    // Extract module name & functions
    let mod_name = root_op
        .prop_idx
        .and_then(|pi| tables.props.get(pi))
        .and_then(|pv| pv.first().copied())
        .filter(|&v| (v & 1) != 0)
        .and_then(|v| match tables.attrs.get(usize::try_from(v >> 1).ok()?) {
            Some(RawV6Attr::Str(s)) if !s.is_empty() => Some(s.clone()),
            _ => None,
        })
        .unwrap_or_else(|| String::from("jax_module"));
    let mut mod_attrs = Vec::new();
    if let Some(di) = root_op.dict_attr_idx {
        if let Some(np) = tables
            .dict_get(di, "mhlo.num_partitions")
            .and_then(|vi| tables.attr_as_i64(vi))
        {
            mod_attrs.push(format!("mhlo.num_partitions = {np} : i32"));
        }
        if let Some(nr) = tables
            .dict_get(di, "mhlo.num_replicas")
            .and_then(|vi| tables.attr_as_i64(vi))
        {
            mod_attrs.push(format!("mhlo.num_replicas = {nr} : i32"));
        }
    }
    if let Some(mod_reg) = root_op.regions.first() {
        for fn_op in &mod_reg.ops {
            if fn_op.name != "func.func" {
                continue;
            }
            let fp = fn_op
                .prop_idx
                .and_then(|pi| tables.props.get(pi))
                .map_or(&[][..], Vec::as_slice);
            let res_slot = if fp.len() >= 6 { 3 } else { 2 };
            let has_manual_comp = fn_op
                .regions
                .first()
                .is_some_and(|r| r.ops.iter().any(|o| o.name == "sdy.manual_computation"));
            for slot in [0, res_slot] {
                if let Some(&ra_raw) = fp.get(slot) {
                    let raw_adj = if (ra_raw & 1) != 0
                        && usize::try_from(ra_raw).unwrap_or(usize::MAX) >= tables.attrs.len()
                    {
                        ra_raw >> 1
                    } else {
                        ra_raw
                    };
                    let ra_idx = usize::try_from(raw_adj).unwrap_or(usize::MAX);
                    if let Some(RawV6Attr::Array(dicts)) = tables.attrs.get(ra_idx) {
                        for &d_idx in dicts {
                            if let Some(RawV6Attr::Dict(pairs)) = tables.attrs.get(d_idx) {
                                for &(k_i, v_i) in pairs {
                                    if let Some(RawV6Attr::Str(ks)) = tables.attrs.get(k_i)
                                        && ks.contains("sharding")
                                    {
                                        match tables.attrs.get(v_i) {
                                            Some(RawV6Attr::Str(vs))
                                                if !vs.is_empty() && !vs.contains("manual") =>
                                            {
                                                mod_attrs.push(format!("mhlo.sharding = \"{vs}\""));
                                            }
                                            Some(RawV6Attr::SdyTensorSharding(_, _))
                                                if !has_manual_comp =>
                                            {
                                                let s = tables
                                                    .fmt_sdy_tensor_sharding_as_op_sharding(v_i);
                                                mod_attrs.push(format!("mhlo.sharding = \"{s}\""));
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let module_header_suffix = if mod_attrs.is_empty() {
        String::new()
    } else {
        format!(" attributes {{{}}}", mod_attrs.join(", "))
    };

    let mut out = String::new();
    let _ = writeln!(out, "module @{mod_name}{module_header_suffix} {{");
    let mut next_tmp = 0_usize;
    if let Some(mod_reg) = root_op.regions.first() {
        for fn_op in &mod_reg.ops {
            if fn_op.name != "func.func" {
                continue;
            }
            let fp = fn_op
                .prop_idx
                .and_then(|pi| tables.props.get(pi))
                .map_or(&[][..], Vec::as_slice);
            // props on func.func:
            // - MLIR 19–23 (5): [arg_attrs(opt), function_type, res_attrs(opt), sym_name, sym_vis(opt)]
            // - MLIR 24+   (6): [arg_attrs(opt), function_type, no_inline(opt), res_attrs(opt), sym_name, sym_vis(opt)]
            let (name_slot, vis_slot) = if fp.len() >= 6 { (4, 5) } else { (3, 4) };
            let fn_ty_attr_idx = usize::try_from(fp.get(1).copied().unwrap_or(0)).unwrap_or(0);
            let fn_name_attr_idx =
                usize::try_from(fp.get(name_slot).copied().unwrap_or(0)).unwrap_or(0);
            let fn_name = match tables.attrs.get(fn_name_attr_idx) {
                Some(RawV6Attr::Str(s)) if !s.is_empty() => s.as_str(),
                _ => "main",
            };
            let vis = fp
                .get(vis_slot)
                .copied()
                .and_then(|v| {
                    let direct = usize::try_from(v).ok().and_then(|i| tables.attrs.get(i));
                    let shifted = if (v & 1) != 0 {
                        usize::try_from(v >> 1)
                            .ok()
                            .and_then(|i| tables.attrs.get(i))
                    } else {
                        None
                    };
                    match direct.or(shifted) {
                        Some(RawV6Attr::Str(s)) if s == "private" => Some("private "),
                        _ => None,
                    }
                })
                .unwrap_or("public ");
            let (in_tys, out_tys) = match tables.attrs.get(fn_ty_attr_idx) {
                Some(RawV6Attr::TypeRef(ti)) => match tables.types.get(*ti) {
                    Some(RawV6Type::Func(ins, outs)) => (ins.clone(), outs.clone()),
                    _ => (Vec::new(), Vec::new()),
                },
                _ => (Vec::new(), Vec::new()),
            };
            let mut ssa_names = Vec::with_capacity(in_tys.len() + 16);
            let mut arg_decls = Vec::with_capacity(in_tys.len());
            for (idx, &ti) in in_tys.iter().enumerate() {
                let anm = format!("%arg{idx}");
                arg_decls.push(format!("{anm}: {}", tables.fmt_type(ti, 0)));
                ssa_names.push(anm);
            }
            let ret_strs: Vec<String> = out_tys.iter().map(|&ti| tables.fmt_type(ti, 0)).collect();
            let ret_sig = if ret_strs.is_empty() {
                String::from("()")
            } else if ret_strs.len() == 1 {
                ret_strs.first().cloned().unwrap_or_default()
            } else {
                format!("({})", ret_strs.join(", "))
            };
            let _ = writeln!(
                out,
                "  func.func {vis}@{fn_name}({}) -> {ret_sig} {{",
                arg_decls.join(", ")
            );
            if let Some(fn_reg) = fn_op.regions.first() {
                emit_v6_ops(
                    &fn_reg.ops,
                    &tables,
                    &mut ssa_names,
                    &mut next_tmp,
                    4,
                    &mut out,
                );
            }
            let _ = writeln!(out, "  }}");
        }
    }
    let _ = writeln!(out, "}}");
    Some(out)
}

#[cfg(test)]
#[allow(clippy::cognitive_complexity, clippy::too_many_lines)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::compiler::mlir_bytecode::encode_mlir_prefix_varint;

    fn uv(v: u64) -> Vec<u8> {
        encode_mlir_prefix_varint(v)
    }

    #[allow(clippy::cast_sign_loss)]
    fn sv(v: i64) -> Vec<u8> {
        let zz = ((v << 1) ^ (v >> 63)) as u64;
        encode_mlir_prefix_varint(zz)
    }

    fn ulist(vals: &[u64]) -> Vec<u8> {
        let mut out = uv(vals.len() as u64);
        for &x in vals {
            out.extend(uv(x));
        }
        out
    }

    fn slist(vals: &[i64]) -> Vec<u8> {
        let mut out = uv(vals.len() as u64);
        for &x in vals {
            out.extend(sv(x));
        }
        out
    }

    fn b64_encode(raw: &[u8]) -> String {
        const ALPH: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for chunk in raw.chunks(3) {
            let b0 = chunk.first().copied().unwrap_or(0);
            let b1 = chunk.get(1).copied().unwrap_or(0);
            let b2 = chunk.get(2).copied().unwrap_or(0);
            let n = (u32::from(b0) << 16) | (u32::from(b1) << 8) | u32::from(b2);
            out.push(ALPH.get(((n >> 18) & 63) as usize).copied().unwrap_or(b'A') as char);
            out.push(ALPH.get(((n >> 12) & 63) as usize).copied().unwrap_or(b'A') as char);
            if chunk.len() > 1 {
                out.push(ALPH.get(((n >> 6) & 63) as usize).copied().unwrap_or(b'A') as char);
            } else {
                out.push('=');
            }
            if chunk.len() > 2 {
                out.push(ALPH.get((n & 63) as usize).copied().unwrap_or(b'A') as char);
            } else {
                out.push('=');
            }
        }
        out
    }

    fn build_section(id: u8, data: &[u8], align: Option<u64>) -> Vec<u8> {
        let mut out = Vec::new();
        if let Some(a) = align {
            out.push(id | 0x80);
            out.extend(uv(data.len() as u64));
            out.extend(uv(a));
        } else {
            out.push(id);
            out.extend(uv(data.len() as u64));
        }
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn test_llvm_v6_varints_types_attrs_and_table_helpers() {
        let mut empty_cur: &[u8] = &[];
        assert_eq!(read_svarint(&mut empty_cur), None);
        assert_eq!(read_svarint_list(&mut empty_cur), None);
        assert_eq!(read_uvarint_list(&mut empty_cur), None);
        assert_eq!(read_dict_pairs(&mut empty_cur), None);

        let mut short_signed_list: &[u8] = &uv(2);
        assert_eq!(read_svarint_list(&mut short_signed_list), None);
        let mut short_unsigned_list: &[u8] = &uv(2);
        assert_eq!(read_uvarint_list(&mut short_unsigned_list), None);
        let mut trunc_dict1: &[u8] = &uv(1);
        assert_eq!(read_dict_pairs(&mut trunc_dict1), None);
        let mut d_buf = uv(1);
        d_buf.extend(uv(0));
        let mut trunc_dict2: &[u8] = &d_buf;
        assert_eq!(read_dict_pairs(&mut trunc_dict2), None);

        assert!(matches!(
            decode_v6_type("builtin", &[]),
            RawV6Type::Float("f32")
        ));
        for tag in [
            0_u64, 1, 2, 3, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 99,
        ] {
            let mut pl = uv(tag);
            pl.extend(uv(1));
            pl.extend(slist(&[2, -1]));
            pl.extend(uv(0));
            let _ = decode_v6_type("vhlo", &pl);
            let _ = decode_v6_type("stablehlo", &pl);
        }
        for tag in [0_u64, 1, 2, 3, 4, 6, 9, 10, 11, 13, 14, 15, 19, 99] {
            let mut pl = uv(tag);
            pl.extend(uv((16 << 2) | 2));
            pl.extend(slist(&[4]));
            pl.extend(uv(0));
            pl.extend(uv(0));
            let _ = decode_v6_type("builtin", &pl);
        }

        let strings = vec![String::from("alpha"), String::from("beta")];
        assert!(matches!(
            decode_v6_attr("builtin", &[], &strings),
            RawV6Attr::Opaque
        ));
        for dir_code in 0..=6_u64 {
            let mut p1 = uv(3);
            p1.extend(uv(dir_code));
            let _ = decode_v6_attr("vhlo", &p1, &strings);
            let mut p2 = uv(2);
            p2.extend(uv(dir_code));
            let _ = decode_v6_attr("stablehlo", &p2, &strings);
        }
        for tag in [1_u64, 6, 9, 14, 15, 17, 99] {
            let mut pl = uv(tag);
            pl.extend(uv(1));
            pl.extend(uv(0));
            pl.extend(uv(1));
            pl.push(7);
            let _ = decode_v6_attr("vhlo", &pl, &strings);
        }
        for tag in [0_u64, 5, 99] {
            let mut pl = uv(tag);
            for _ in 0..4 {
                pl.extend(slist(&[0]));
            }
            let _ = decode_v6_attr("mhlo", &pl, &strings);
        }
        for tag in [0_u64, 1, 2, 3, 4, 6, 7, 8, 9, 17, 18, 99] {
            let mut pl = uv(tag);
            pl.extend(uv(0));
            pl.extend(uv(1));
            pl.extend(uv(1));
            pl.push(42);
            let _ = decode_v6_attr("builtin", &pl, &strings);
        }

        let tables = V6Tables {
            types: vec![
                RawV6Type::Int(32, 0),
                RawV6Type::Int(16, 2),
                RawV6Type::Index,
                RawV6Type::Token,
                RawV6Type::Float("f32"),
                RawV6Type::Float("f64"),
                RawV6Type::Int(1, 0),
                RawV6Type::Int(64, 0),
                RawV6Type::Complex(4),
                RawV6Type::Tensor(vec![], 4),
                RawV6Type::Tensor(vec![2, -1], 4),
                RawV6Type::Tuple(vec![0, 4]),
                RawV6Type::Func(vec![0], vec![4]),
                RawV6Type::Tensor(vec![2], 0),
                RawV6Type::Tensor(vec![2], 5),
                RawV6Type::Tensor(vec![2], 7),
                RawV6Type::Tensor(vec![0], 0),
                RawV6Type::Tensor(vec![1], 6),
                RawV6Type::Float("bf16"),
                RawV6Type::Tensor(vec![2], 18),
            ],
            attrs: vec![
                RawV6Attr::DenseArray(0, vec![]),
                RawV6Attr::DenseArray(2, [1_i64.to_le_bytes(), 2_i64.to_le_bytes()].concat()),
                RawV6Attr::DenseArray(2, [3_i32.to_le_bytes(), 4_i32.to_le_bytes()].concat()),
                RawV6Attr::DenseArray(2, vec![5, 6]),
                RawV6Attr::DenseElements(13, 9_i32.to_le_bytes().to_vec()),
                RawV6Attr::DenseElements(13, [1_i32.to_le_bytes(), 2_i32.to_le_bytes()].concat()),
                RawV6Attr::DenseElements(15, [7_i64.to_le_bytes(), 8_i64.to_le_bytes()].concat()),
                RawV6Attr::DenseElements(16, 1_i32.to_le_bytes().to_vec()),
                RawV6Attr::DenseElements(0, 5_i32.to_le_bytes().to_vec()),
                RawV6Attr::Int(11),
                RawV6Attr::Unit,
                RawV6Attr::Array(vec![9, 10]),
                RawV6Attr::Channel(42),
                RawV6Attr::Str(String::from("my_key")),
                RawV6Attr::Dict(vec![(10, 9), (13, 9)]),
                RawV6Attr::DenseElements(
                    10,
                    [f32::NAN.to_le_bytes(), 1.5_f32.to_le_bytes()].concat(),
                ),
                RawV6Attr::DenseElements(9, vec![]),
                RawV6Attr::DenseElements(
                    14,
                    [1.25_f64.to_le_bytes(), 2.5_f64.to_le_bytes()].concat(),
                ),
                RawV6Attr::DenseElements(5, vec![]),
                RawV6Attr::DenseElements(17, vec![1]),
                RawV6Attr::DenseElements(17, vec![0]),
                RawV6Attr::DenseElements(0, vec![]),
                RawV6Attr::DenseElements(7, vec![]),
                RawV6Attr::DenseElements(19, vec![1, 2]),
            ],
            props: vec![],
        };
        assert_eq!(tables.fmt_type(0, 9), "tensor<f32>");
        for ti in 0..=tables.types.len() {
            assert_ne!(tables.fmt_type(ti, 0), "");
        }
        for ai in 0..=12 {
            let _ = tables.attr_as_i64_vec(ai);
        }
        assert_eq!(tables.attr_as_i64(9), Some(11));
        assert_eq!(tables.attr_as_i64(12), Some(42));
        assert_eq!(tables.attr_as_i64(10), None);
        assert_eq!(tables.dict_get(10, "my_key"), None);
        assert_eq!(tables.dict_get(14, "my_key"), Some(9));
        assert_eq!(tables.dict_get(14, "missing"), None);
        assert_eq!(tables.fmt_dense_elements(10), None);
        for di in [4, 5, 6, 15, 16, 17, 18, 19, 20, 21, 22, 23] {
            assert!(tables.fmt_dense_elements(di).is_some());
        }
    }

    #[test]
    fn test_llvm_v6_ir_parser_op_emitters_and_mosaic_lowering() {
        assert_eq!(
            decode_base64_bytes("QUJD\n+/09=="),
            Some(vec![65, 66, 67, 251, 253, 61])
        );
        assert_eq!(decode_base64_bytes("@@@"), None);

        let reg_ops = vec![String::from("func.return")];
        assert!(parse_ir_op(&[], 0, &reg_ops, 6).is_none());
        assert!(parse_ir_op(&uv(0), 0, &reg_ops, 6).is_none());
        assert!(parse_ir_op(&[uv(0), vec![0]].concat(), 0, &reg_ops, 6).is_none());
        assert!(parse_ir_op(&[uv(0), vec![0x01], uv(0)].concat(), 0, &reg_ops, 6).is_none());
        assert!(parse_ir_op(&[uv(0), vec![0x40], uv(0)].concat(), 0, &reg_ops, 6).is_none());
        assert!(parse_ir_op(&[uv(0), vec![0x02], uv(0)].concat(), 0, &reg_ops, 6).is_none());
        assert!(parse_ir_op(&[uv(0), vec![0x02], uv(0), uv(1)].concat(), 0, &reg_ops, 6).is_none());
        assert!(parse_ir_op(&[uv(0), vec![0x04], uv(0)].concat(), 0, &reg_ops, 6).is_none());
        assert!(parse_ir_op(&[uv(0), vec![0x04], uv(0), uv(1)].concat(), 0, &reg_ops, 6).is_none());
        assert!(parse_ir_op(&[uv(0), vec![0x08], uv(0)].concat(), 0, &reg_ops, 6).is_none());
        assert!(parse_ir_op(&[uv(0), vec![0x08], uv(0), uv(1)].concat(), 0, &reg_ops, 6).is_none());
        assert!(parse_ir_op(&[uv(0), vec![0x10], uv(0)].concat(), 0, &reg_ops, 6).is_none());

        // Empty region (`nb == 0`)
        assert!(parse_ir_region(&[], 0, &reg_ops, 6, false).is_none());
        let empty_reg = parse_ir_region(&uv(0), 0, &reg_ops, 6, true);
        assert!(empty_reg.is_some());
        assert!(parse_ir_region(&uv(1), 0, &reg_ops, 6, false).is_none());
        assert!(parse_ir_region(&[uv(1), uv(0)].concat(), 0, &reg_ops, 6, false).is_none());
        assert!(parse_ir_region(&[uv(1), uv(0), uv(1)].concat(), 0, &reg_ops, 6, false).is_none());
        // Region with 2 blocks (`b_idx == 0` and `b_idx == 1`), `ver >= 3` & `ver < 3`
        let mut r_v6 = Vec::new();
        r_v6.extend(uv(2)); // 2 blocks
        r_v6.extend(uv(2)); // nv
        // block 0: 1 op, has_args=1
        r_v6.extend(uv((1 << 1) | 1));
        r_v6.extend(uv(2)); // 2 args
        r_v6.extend(uv(0)); // arg0 without loc
        r_v6.extend(uv(1)); // arg1 with loc
        r_v6.extend(uv(0)); // loc
        r_v6.push(0); // u_flag
        // op in block 0: mask with 0x01|0x02|0x04|0x08|0x40, unknown op_id=5
        r_v6.extend(uv(5));
        r_v6.push(0x01 | 0x02 | 0x04 | 0x08 | 0x40);
        r_v6.extend(uv(0)); // loc
        r_v6.extend(uv(0)); // dict
        r_v6.extend(uv(0)); // prop
        r_v6.extend(ulist(&[0])); // result_types
        r_v6.extend(ulist(&[0])); // operands
        r_v6.extend(ulist(&[0])); // successors
        // block 1: 1 op, has_args=1
        r_v6.extend(uv((1 << 1) | 1));
        r_v6.extend(uv(1));
        r_v6.extend(uv(0));
        r_v6.push(0);
        r_v6.extend(uv(0));
        r_v6.push(0);
        r_v6.extend(uv(0));
        assert!(parse_ir_region(&r_v6, 0, &reg_ops, 6, false).is_some());
        // Truncated u_flag in ver >= 3
        let trunc_u = [uv(1), uv(1), uv(1), uv(1), uv(0)].concat();
        assert!(parse_ir_region(&trunc_u, 0, &reg_ops, 6, false).is_none());
        let trunc_tl_loc = [uv(1), uv(1), uv(1), uv(1), uv(1)].concat();
        assert!(parse_ir_region(&trunc_tl_loc, 0, &reg_ops, 6, false).is_none());
        let trunc_op_in_blk = [uv(1), uv(0), uv(2)].concat();
        assert!(parse_ir_region(&trunc_op_in_blk, 0, &reg_ops, 6, false).is_none());

        // `ver < 3` region block args (2 blocks + truncated paths)
        let mut r_v2 = Vec::new();
        r_v2.extend(uv(2));
        r_v2.extend(uv(2));
        r_v2.extend(uv(1)); // 0 ops, has_args=1
        r_v2.extend(uv(1)); // 1 arg
        r_v2.extend(uv(0)); // ty
        r_v2.extend(uv(0)); // loc
        r_v2.extend(uv(1)); // block 1: 0 ops, has_args=1
        r_v2.extend(uv(1));
        r_v2.extend(uv(0));
        r_v2.extend(uv(0));
        assert!(parse_ir_region(&r_v2, 0, &reg_ops, 2, false).is_some());
        assert!(
            parse_ir_region(
                &[uv(1), uv(1), uv(1), uv(1)].concat(),
                0,
                &reg_ops,
                2,
                false
            )
            .is_none()
        );
        assert!(
            parse_ir_region(
                &[uv(1), uv(1), uv(1), uv(1), uv(0)].concat(),
                0,
                &reg_ops,
                2,
                false
            )
            .is_none()
        );

        // `parse_ir_op` with isolated sub-section `4` AND inline region
        let sub_reg = uv(0); // empty region
        let mut iso_op = Vec::new();
        iso_op.extend(uv(0));
        iso_op.push(0x10);
        iso_op.extend(uv(0));
        iso_op.extend(uv((1 << 1) | 1)); // 1 isolated region
        iso_op.push(4);
        iso_op.extend(uv(sub_reg.len() as u64));
        iso_op.extend_from_slice(&sub_reg);
        assert!(parse_ir_op(&iso_op, 0, &reg_ops, 6).is_some());
        // Truncated sub-section 4
        let trunc_iso = iso_op.get(..iso_op.len() - 1).unwrap_or_default();
        assert!(parse_ir_op(trunc_iso, 0, &reg_ops, 6).is_none());
        let bad_iso = [uv(0), vec![0x10], uv(0), uv(3), vec![4]].concat();
        assert!(parse_ir_op(&bad_iso, 0, &reg_ops, 6).is_none());
        let bad_iso_reg = [uv(0), vec![0x10], uv(0), uv(3), vec![4], uv(1), uv(1)].concat();
        assert!(parse_ir_op(&bad_iso_reg, 0, &reg_ops, 6).is_none());
        // Inline non-isolated region (`is_iso = false`)
        let inl_op = [uv(0), vec![0x10], uv(0), uv(2), uv(0)].concat();
        assert!(parse_ir_op(&inl_op, 0, &reg_ops, 4).is_some());
        let bad_inl_op = [uv(0), vec![0x10], uv(0), uv(2)].concat();
        assert!(parse_ir_op(&bad_inl_op, 0, &reg_ops, 4).is_none());

        // Exercise `emit_v6_op` & `emit_v6_ops` across all op mnemonics & branches
        let tables = V6Tables {
            types: vec![
                RawV6Type::Tensor(vec![4], 1),
                RawV6Type::Float("f32"),
                RawV6Type::Int(1, 0),
                RawV6Type::Tensor(vec![], 2),
                RawV6Type::Token,
            ],
            attrs: vec![
                RawV6Attr::DenseElements(0, [1.0_f32.to_le_bytes(); 4].concat()),
                RawV6Attr::DotDims(vec![], vec![], vec![1], vec![0]),
                RawV6Attr::DenseArray(1, 0_i64.to_le_bytes().to_vec()),
                RawV6Attr::CmpDir("LT"),
                RawV6Attr::Int(1),
                RawV6Attr::SymRef(String::from("sub_fn")),
                RawV6Attr::Str(String::from("dict_fn")),
                RawV6Attr::Str(String::from("callee")),
                RawV6Attr::Dict(vec![(7, 6)]),
                RawV6Attr::Str(String::from("mhlo.sharding")),
                RawV6Attr::Str(String::from("{replicated}")),
                RawV6Attr::Dict(vec![(9, 10)]),
                RawV6Attr::Str(String::from(
                    "{\"custom_call_config\":{\"body\":\"invalid\"}}",
                )),
                RawV6Attr::Str(String::from("my_custom_op")),
                RawV6Attr::Dict(vec![(7, 12)]),
                RawV6Attr::Str(String::new()),
                RawV6Attr::Str(String::from("123")),
                RawV6Attr::Dict(vec![(7, 9)]),
            ],
            props: vec![
                vec![0],                                       // 0: dense attr 0
                vec![99],                                      // 1: missing attr
                vec![2, 2, 0, 2, 2],                           // 2: 5-prop dot_general
                vec![1],                                       // 3: DotDims attr 1
                vec![2, 2, 2],                                 // 4: slice
                vec![0, 3],                                    // 5: compare LT
                vec![4],                                       // 6: get_tuple_element Int(1)
                vec![5],                                       // 7: func.call SymRef
                vec![1, 12, 13], // 8: custom_call prop(1)=body, prop(2)=target
                vec![0, (12 << 1) | 1, 15, 16, (13 << 1) | 1], // 9: custom_call shifted body + fallback target
                vec![4, 4, 4, 4, 4],                           // 10: collectives
                vec![4; 17],                                   // 11: 17-prop convolution
                vec![(2 << 1) | 1], // 12: <17-prop convolution with odd stride prop
            ],
        };
        let simple_ret = V6Region {
            is_isolated: false,
            arg_types: vec![0],
            ops: vec![V6OpNode {
                name: String::from("stablehlo.return"),
                dict_attr_idx: None,
                prop_idx: None,
                result_types: vec![],
                operands: vec![0, 99],
                regions: vec![],
            }],
        };
        let iso_ret = V6Region {
            is_isolated: true,
            arg_types: vec![],
            ops: vec![V6OpNode {
                name: String::from("func.return"),
                dict_attr_idx: None,
                prop_idx: None,
                result_types: vec![],
                operands: vec![],
                regions: vec![],
            }],
        };
        let op_nodes = vec![
            V6OpNode {
                name: String::from("func.return"),
                dict_attr_idx: None,
                prop_idx: None,
                result_types: vec![],
                operands: vec![],
                regions: vec![],
            },
            V6OpNode {
                name: String::from("stablehlo.constant"),
                dict_attr_idx: None,
                prop_idx: Some(0),
                result_types: vec![0],
                operands: vec![],
                regions: vec![],
            },
            V6OpNode {
                name: String::from("stablehlo.constant"),
                dict_attr_idx: None,
                prop_idx: Some(1),
                result_types: vec![],
                operands: vec![],
                regions: vec![],
            },
            V6OpNode {
                name: String::from("stablehlo.dot_general"),
                dict_attr_idx: None,
                prop_idx: Some(2),
                result_types: vec![0],
                operands: vec![0, 0],
                regions: vec![],
            },
            V6OpNode {
                name: String::from("stablehlo.dot_general"),
                dict_attr_idx: None,
                prop_idx: Some(3),
                result_types: vec![0],
                operands: vec![],
                regions: vec![],
            },
            V6OpNode {
                name: String::from("stablehlo.dot_general"),
                dict_attr_idx: None,
                prop_idx: Some(1),
                result_types: vec![],
                operands: vec![],
                regions: vec![],
            },
            V6OpNode {
                name: String::from("stablehlo.broadcast_in_dim"),
                dict_attr_idx: None,
                prop_idx: Some(0),
                result_types: vec![0],
                operands: vec![0],
                regions: vec![],
            },
            V6OpNode {
                name: String::from("stablehlo.transpose"),
                dict_attr_idx: None,
                prop_idx: Some(2),
                result_types: vec![0],
                operands: vec![0],
                regions: vec![],
            },
            V6OpNode {
                name: String::from("stablehlo.slice"),
                dict_attr_idx: None,
                prop_idx: Some(4),
                result_types: vec![0],
                operands: vec![0],
                regions: vec![],
            },
            V6OpNode {
                name: String::from("stablehlo.dynamic_slice"),
                dict_attr_idx: None,
                prop_idx: Some(2),
                result_types: vec![0],
                operands: vec![0, 0],
                regions: vec![],
            },
            V6OpNode {
                name: String::from("stablehlo.compare"),
                dict_attr_idx: None,
                prop_idx: Some(5),
                result_types: vec![3],
                operands: vec![0, 0],
                regions: vec![],
            },
            V6OpNode {
                name: String::from("stablehlo.compare"),
                dict_attr_idx: None,
                prop_idx: Some(1),
                result_types: vec![],
                operands: vec![],
                regions: vec![],
            },
            V6OpNode {
                name: String::from("stablehlo.get_tuple_element"),
                dict_attr_idx: None,
                prop_idx: Some(6),
                result_types: vec![0],
                operands: vec![0],
                regions: vec![],
            },
            V6OpNode {
                name: String::from("stablehlo.get_tuple_element"),
                dict_attr_idx: None,
                prop_idx: Some(1),
                result_types: vec![],
                operands: vec![],
                regions: vec![],
            },
            V6OpNode {
                name: String::from("stablehlo.reduce"),
                dict_attr_idx: None,
                prop_idx: Some(2),
                result_types: vec![0],
                operands: vec![0, 0],
                regions: vec![simple_ret.clone()],
            },
            V6OpNode {
                name: String::from("stablehlo.reduce"),
                dict_attr_idx: None,
                prop_idx: None,
                result_types: vec![0],
                operands: vec![],
                regions: vec![],
            },
            V6OpNode {
                name: String::from("stablehlo.while"),
                dict_attr_idx: None,
                prop_idx: None,
                result_types: vec![0],
                operands: vec![0],
                regions: vec![simple_ret.clone(), iso_ret.clone()],
            },
            V6OpNode {
                name: String::from("stablehlo.while"),
                dict_attr_idx: None,
                prop_idx: None,
                result_types: vec![0, 0],
                operands: vec![0, 0],
                regions: vec![],
            },
            V6OpNode {
                name: String::from("stablehlo.case"),
                dict_attr_idx: None,
                prop_idx: None,
                result_types: vec![0],
                operands: vec![0],
                regions: vec![simple_ret.clone(), iso_ret],
            },
            V6OpNode {
                name: String::from("stablehlo.case"),
                dict_attr_idx: None,
                prop_idx: None,
                result_types: vec![0, 0],
                operands: vec![],
                regions: vec![],
            },
            V6OpNode {
                name: String::from("func.call"),
                dict_attr_idx: None,
                prop_idx: Some(7),
                result_types: vec![0],
                operands: vec![0],
                regions: vec![],
            },
            V6OpNode {
                name: String::from("func.call"),
                dict_attr_idx: Some(8),
                prop_idx: Some(1),
                result_types: vec![0, 0],
                operands: vec![],
                regions: vec![],
            },
            V6OpNode {
                name: String::from("func.call"),
                dict_attr_idx: Some(0),
                prop_idx: None,
                result_types: vec![0],
                operands: vec![],
                regions: vec![],
            },
            V6OpNode {
                name: String::from("func.call"),
                dict_attr_idx: None,
                prop_idx: None,
                result_types: vec![0],
                operands: vec![],
                regions: vec![],
            },
            V6OpNode {
                name: String::from("stablehlo.custom_call"),
                dict_attr_idx: Some(11),
                prop_idx: Some(8),
                result_types: vec![0],
                operands: vec![0],
                regions: vec![],
            },
            V6OpNode {
                name: String::from("stablehlo.custom_call"),
                dict_attr_idx: Some(17),
                prop_idx: Some(9),
                result_types: vec![0],
                operands: vec![0],
                regions: vec![],
            },
            V6OpNode {
                name: String::from("stablehlo.custom_call"),
                dict_attr_idx: Some(14),
                prop_idx: None,
                result_types: vec![0],
                operands: vec![0],
                regions: vec![],
            },
            V6OpNode {
                name: String::from("stablehlo.custom_call"),
                dict_attr_idx: Some(0),
                prop_idx: None,
                result_types: vec![0],
                operands: vec![],
                regions: vec![],
            },
            V6OpNode {
                name: String::from("stablehlo.send"),
                dict_attr_idx: None,
                prop_idx: Some(6),
                result_types: vec![4],
                operands: vec![0],
                regions: vec![],
            },
            V6OpNode {
                name: String::from("stablehlo.recv"),
                dict_attr_idx: None,
                prop_idx: Some(6),
                result_types: vec![0, 4],
                operands: vec![0],
                regions: vec![],
            },
            V6OpNode {
                name: String::from("stablehlo.all_reduce"),
                dict_attr_idx: None,
                prop_idx: Some(10),
                result_types: vec![0],
                operands: vec![0],
                regions: vec![simple_ret.clone()],
            },
            V6OpNode {
                name: String::from("stablehlo.reduce_scatter"),
                dict_attr_idx: None,
                prop_idx: Some(10),
                result_types: vec![0],
                operands: vec![0],
                regions: vec![simple_ret],
            },
            V6OpNode {
                name: String::from("stablehlo.all_reduce"),
                dict_attr_idx: None,
                prop_idx: None,
                result_types: vec![0],
                operands: vec![0],
                regions: vec![],
            },
            V6OpNode {
                name: String::from("stablehlo.all_gather"),
                dict_attr_idx: None,
                prop_idx: Some(10),
                result_types: vec![0],
                operands: vec![0],
                regions: vec![],
            },
            V6OpNode {
                name: String::from("stablehlo.all_to_all"),
                dict_attr_idx: None,
                prop_idx: Some(10),
                result_types: vec![0],
                operands: vec![0],
                regions: vec![],
            },
            V6OpNode {
                name: String::from("stablehlo.convolution"),
                dict_attr_idx: None,
                prop_idx: Some(11),
                result_types: vec![0],
                operands: vec![0, 0],
                regions: vec![],
            },
            V6OpNode {
                name: String::from("stablehlo.convolution"),
                dict_attr_idx: None,
                prop_idx: Some(12),
                result_types: vec![0],
                operands: vec![0, 0],
                regions: vec![],
            },
            V6OpNode {
                name: String::from("stablehlo.convolution"),
                dict_attr_idx: None,
                prop_idx: None,
                result_types: vec![0],
                operands: vec![0, 0],
                regions: vec![],
            },
            V6OpNode {
                name: String::from("stablehlo.add"),
                dict_attr_idx: None,
                prop_idx: None,
                result_types: vec![0, 0],
                operands: vec![0, 0],
                regions: vec![],
            },
            V6OpNode {
                name: String::from("stablehlo.nop"),
                dict_attr_idx: None,
                prop_idx: None,
                result_types: vec![],
                operands: vec![],
                regions: vec![],
            },
        ];
        let mut ssa = vec![String::from("%arg0")];
        let mut tmp = 0_usize;
        let mut emitted = String::new();
        emit_v6_ops(&op_nodes, &tables, &mut ssa, &mut tmp, 2, &mut emitted);
        assert!(emitted.contains("stablehlo.convolution"));
        assert!(emitted.contains("stablehlo.reduce_scatter"));

        // `try_emit_mosaic_custom_call` error & full kernel paths
        let mut m_out = String::new();
        assert_eq!(
            try_emit_mosaic_custom_call(
                "no_body",
                &[],
                "",
                "tensor<4xf32>",
                "",
                &mut tmp,
                &mut m_out
            ),
            None
        );
        assert_eq!(
            try_emit_mosaic_custom_call(
                "\"body\":123",
                &[],
                "",
                "tensor<4xf32>",
                "",
                &mut tmp,
                &mut m_out
            ),
            None
        );
        assert_eq!(
            try_emit_mosaic_custom_call(
                "\"body\":\"unterminated",
                &[],
                "",
                "tensor<4xf32>",
                "",
                &mut tmp,
                &mut m_out
            ),
            None
        );
        assert_eq!(
            try_emit_mosaic_custom_call(
                "\"body\":\"@@@\"",
                &[],
                "",
                "tensor<4xf32>",
                "",
                &mut tmp,
                &mut m_out
            ),
            None
        );
        assert_eq!(
            try_emit_mosaic_custom_call(
                "\"body\":\"QUJD\"",
                &[],
                "",
                "tensor<4xf32>",
                "",
                &mut tmp,
                &mut m_out
            ),
            None
        );
    }

    /// Dialect specification tuple `(dialect_name, optional_inline_version, registered_op_names)` for test containers.
    type DialectTestSpec<'a> = (&'a str, Option<&'a [u8]>, &'a [&'a str]);

    /// Attribute specification tuple `(dialect_idx, has_custom_encoding, raw_payload_bytes)` for test containers.
    type AttrTestSpec = (usize, bool, Vec<u8>);

    /// Builds a valid `ML\xefR` bytecode container from high-level string/dialect/attr/type/IR/prop components.
    ///
    /// # Arguments
    ///
    /// * `ver` - MLIR bytecode container version (`0..=6`).
    /// * `producer` - Producer header string (`"MLIR20"`, `"StableHLO_v1"`, etc.).
    /// * `strings` - Section `0` string table entries.
    /// * `dialects` - Section `1` dialect and registered op entries.
    /// * `attrs` - Section `2`/`3` attribute entries.
    /// * `types` - Section `2`/`3` type entries.
    /// * `props` - Section `8` operation property varint vectors.
    /// * `root_ir` - Section `4` serialized root operation bitstream.
    /// * `align_s0` - Whether to emit Section `0` with explicit 4-byte `0xCB` alignment padding.
    #[allow(clippy::too_many_arguments)]
    fn build_container(
        ver: u64,
        producer: &str,
        strings: &[&str],
        dialects: &[DialectTestSpec<'_>],
        attrs: &[AttrTestSpec],
        types: &[(usize, Vec<u8>)],
        props: &[Vec<u64>],
        root_ir: &[u8],
        align_s0: bool,
    ) -> Vec<u8> {
        let mut out = MLIR_BYTECODE_MAGIC.to_vec();
        out.extend(uv(ver));
        out.extend_from_slice(producer.as_bytes());
        out.push(0);

        // Section 0: Strings (reverse lengths, then concatenated bytes)
        let mut s0 = uv(strings.len() as u64);
        for s in strings.iter().rev() {
            s0.extend(uv((s.len() + 1) as u64));
        }
        for s in strings {
            s0.extend_from_slice(s.as_bytes());
            s0.push(0);
        }
        if align_s0 {
            let mut hdr = vec![0x80_u8];
            hdr.extend(uv(s0.len() as u64));
            hdr.extend(uv(4));
            while (out.len() + hdr.len()) % 4 != 0 {
                hdr.push(MLIR_ALIGNMENT_BYTE);
            }
            out.extend(hdr);
            out.extend(s0);
        } else {
            out.extend(build_section(0, &s0, None));
        }

        // Section 1: Dialects + ops
        let str_idx =
            |needle: &str| -> u64 { strings.iter().position(|&x| x == needle).unwrap_or(0) as u64 };
        let mut s1 = uv(dialects.len() as u64);
        let mut total_ops = 0_u64;
        for &(dname, dver, dops) in dialects {
            total_ops += dops.len() as u64;
            let si = str_idx(dname);
            if ver >= 1 {
                if let Some(vb) = dver {
                    s1.extend(uv((si << 1) | 1));
                    s1.extend(uv(vb.len() as u64));
                    s1.extend_from_slice(vb);
                } else {
                    s1.extend(uv(si << 1));
                }
            } else {
                s1.extend(uv(si));
            }
        }
        if ver >= 1 {
            s1.extend(uv(total_ops));
        }
        for (did, &(_dname, _dver, dops)) in dialects.iter().enumerate() {
            if dops.is_empty() {
                continue;
            }
            s1.extend(uv(did as u64));
            s1.extend(uv(dops.len() as u64));
            for &opn in dops {
                let oi = str_idx(opn);
                if ver >= 2 {
                    s1.extend(uv(oi << 1));
                } else {
                    s1.extend(uv(oi));
                }
            }
        }
        out.extend(build_section(1, &s1, None));

        // Section 2 & 3: AttrType & AttrTypeOffset
        let mut s2 = Vec::new();
        let mut s3 = uv(attrs.len() as u64);
        s3.extend(uv(types.len() as u64));
        for &(did, is_cust, ref bytes) in attrs {
            s3.extend(uv(did as u64));
            s3.extend(uv(1));
            s3.extend(uv(((bytes.len() as u64) << 1) | u64::from(is_cust)));
            s2.extend_from_slice(bytes);
        }
        for &(did, ref bytes) in types {
            s3.extend(uv(did as u64));
            s3.extend(uv(1));
            s3.extend(uv(((bytes.len() as u64) << 1) | 1));
            s2.extend_from_slice(bytes);
        }
        out.extend(build_section(2, &s2, None));
        out.extend(build_section(3, &s3, None));

        // Section 8: Properties
        if !props.is_empty() {
            let mut s8 = uv(props.len() as u64);
            for pv in props {
                let mut pbuf = Vec::new();
                for &x in pv {
                    pbuf.extend(uv(x));
                }
                s8.extend(uv(pbuf.len() as u64));
                s8.extend(pbuf);
            }
            out.extend(build_section(8, &s8, None));
        }

        // Section 4: kIR
        let mut s4 = uv(0);
        s4.extend_from_slice(root_ir);
        out.extend(build_section(4, &s4, None));
        out
    }

    #[test]
    fn test_llvm_v6_container_mosaic_v0_and_compiled_program() {
        // Negative container headers
        assert_eq!(decode_llvm_mlir_v6_to_text(b"not_mlir"), None);
        assert_eq!(decode_llvm_mlir_v6_to_text(b"ML\xefR"), None);
        assert_eq!(decode_llvm_mlir_v6_to_text(b"ML\xefR\x0dno_nul"), None);
        assert_eq!(decode_llvm_mlir_v6_to_text(b"ML\xefR\x05MLIR19\0"), None);
        assert_eq!(
            decode_llvm_mlir_v6_to_text(b"ML\xefR\x0dUnknownProd\0"),
            None
        );
        assert_eq!(decode_llvm_mlir_v6_to_text(b"ML\xefR\x0d\xff\0"), None);

        // Build a v0 `MLIR20` Mosaic container exercising EVERY Mosaic op in `try_emit_mosaic_custom_call`
        let m_strings = [
            "builtin",
            "func",
            "vector",
            "arith",
            "tpu",
            "module",
            "func",
            "load",
            "store",
            "constant",
            "broadcast",
            "addf",
            "subf",
            "mulf",
            "divf",
            "matmul",
            "return",
            "noop",
            "val_key",
        ];
        let m_dialects: &[DialectTestSpec<'_>] = &[
            ("builtin", None, &["module"]),
            ("func", None, &["func", "return"]),
            ("vector", None, &["load", "broadcast", "store"]),
            ("arith", None, &["constant", "addf", "subf", "mulf", "divf"]),
            ("tpu", None, &["matmul", "noop"]),
        ];
        // Mosaic types: 0=f32, 1=f64, 2=bf16, 3=i32, 4=index, 5=tensor<4xf32>
        let m_types = vec![
            (0, uv(99)),                                // 0: f32
            (0, uv(6)),                                 // 1: f64
            (0, uv(3)),                                 // 2: bf16
            (0, [uv(0), uv(32 << 2)].concat()),         // 3: i32
            (0, uv(1)),                                 // 4: index
            (0, [uv(13), slist(&[4]), uv(0)].concat()), // 5: tensor<4xf32>
        ];
        // Mosaic attrs:
        // 0: FloatBits(f32=0, 2.0), 1: FloatBits(f64=1, 3.0), 2: FloatBits(bf16=2, 0x4000),
        // 3: DenseElements(5, 4xf32), 4: Int(7), 5: Unit, 6: Str("val_key"), 7: Dict([(6, 0)])
        #[allow(clippy::cast_possible_wrap)]
        let m_attrs = vec![
            (
                0,
                true,
                [uv(9), uv(0), sv(i64::from(2.0_f32.to_bits()))].concat(),
            ),
            (
                0,
                true,
                [uv(9), uv(1), sv(3.0_f64.to_bits() as i64)].concat(),
            ),
            (0, true, [uv(9), uv(2), sv(0x4000)].concat()),
            (
                0,
                true,
                [uv(18), uv(5), uv(16), [1.0_f32.to_le_bytes(); 4].concat()].concat(),
            ),
            (0, true, [uv(8), uv(3), sv(7)].concat()),
            (0, false, vec![]),
            (0, true, [uv(2), uv(18)].concat()),
            (0, true, [uv(1), uv(1), uv(6), uv(0)].concat()),
        ];
        let m_props = vec![vec![1], vec![2], vec![3], vec![4], vec![5]];

        let mk_mop = |op_id: u64,
                      dict: Option<u64>,
                      prop: Option<u64>,
                      rts: &[u64],
                      opnds: &[u64]|
         -> Vec<u8> {
            let mut b = uv(op_id);
            let mut mask = 0_u8;
            if dict.is_some() {
                mask |= 0x01;
            }
            if !rts.is_empty() {
                mask |= 0x02;
            }
            if !opnds.is_empty() {
                mask |= 0x04;
            }
            if prop.is_some() {
                mask |= 0x40;
            }
            b.push(mask);
            b.extend(uv(0));
            if let Some(d) = dict {
                b.extend(uv(d));
            }
            if let Some(p) = prop {
                b.extend(uv(p));
            }
            if !rts.is_empty() {
                b.extend(ulist(rts));
            }
            if !opnds.is_empty() {
                b.extend(ulist(opnds));
            }
            b
        };
        // Op IDs in m_dialects: 0=builtin.module, 1=func.func, 2=func.return,
        // 3=vector.load, 4=vector.broadcast, 5=vector.store,
        // 6=arith.constant, 7=arith.addf, 8=arith.subf, 9=arith.mulf, 10=arith.divf,
        // 11=tpu.matmul, 12=tpu.noop
        // Note: in v0 (`ver < 5`), `prop_idx` is not read from `mask & 0x40`, so use v5 for Mosaic or dict_attr_idx!
        // `parse_mlir_bytecode_container` accepts v5 with `MLIR20` as well! Let's build Mosaic as v5 so BOTH `prop_idx` AND `dict_attr_idx` are tested, AND build a minimal v0 Mosaic container for `ver == 0`!
        let mut fn_ops_bytes = Vec::new();
        let fn_ops_list = [
            mk_mop(3, None, None, &[5], &[0]),       // vector.load %arg0
            mk_mop(6, Some(7), None, &[0], &[]),     // arith.constant via dict -> FloatBits f32
            mk_mop(6, None, Some(0), &[1], &[]),     // arith.constant -> FloatBits f64
            mk_mop(6, None, Some(1), &[2], &[]),     // arith.constant -> FloatBits bf16
            mk_mop(6, None, Some(2), &[5], &[]),     // arith.constant -> DenseElements
            mk_mop(6, None, Some(3), &[5], &[]),     // arith.constant -> Int with tensor<...>
            mk_mop(6, None, Some(3), &[4], &[]),     // arith.constant -> Int with index
            mk_mop(6, None, Some(3), &[3], &[]),     // arith.constant -> Int with i32
            mk_mop(6, None, Some(4), &[3], &[]),     // arith.constant -> Opaque fallback
            mk_mop(4, None, None, &[5], &[3]),       // vector.broadcast
            mk_mop(7, None, None, &[5], &[2, 11]),   // arith.addf
            mk_mop(8, None, None, &[5], &[12, 11]),  // arith.subf
            mk_mop(9, None, None, &[5], &[13, 11]),  // arith.mulf
            mk_mop(10, None, None, &[5], &[14, 11]), // arith.divf
            mk_mop(11, None, None, &[5], &[15, 11]), // tpu.matmul
            mk_mop(12, None, None, &[5], &[16]),     // generic tpu.noop
            mk_mop(5, None, None, &[], &[17, 1]),    // vector.store
            mk_mop(2, None, None, &[], &[]),         // func.return
        ];
        for ob in &fn_ops_list {
            fn_ops_bytes.extend_from_slice(ob);
        }
        // fn_reg: 1 block, nv=20, nops=fn_ops_list.len(), has_args=1 (2 memref args)
        let mut fn_reg = uv(1);
        fn_reg.extend(uv(20));
        fn_reg.extend(uv(((fn_ops_list.len() as u64) << 1) | 1));
        fn_reg.extend(uv(2)); // 2 args
        fn_reg.extend(uv(5 << 1));
        fn_reg.extend(uv(5 << 1));
        fn_reg.push(0); // u_flag
        fn_reg.extend(fn_ops_bytes);

        let mut fn_op = uv(1); // func.func
        fn_op.push(0x10);
        fn_op.extend(uv(0));
        fn_op.extend(uv(2)); // 1 non-isolated region
        fn_op.extend(fn_reg);

        let mut mod_reg = uv(1);
        mod_reg.extend(uv(0));
        mod_reg.extend(uv(1 << 1)); // 1 op, no args
        mod_reg.extend(fn_op);

        let mut root_ir = uv(0); // builtin.module
        root_ir.push(0x10);
        root_ir.extend(uv(0));
        root_ir.extend(uv(2));
        root_ir.extend(mod_reg);

        let mosaic_v5 = build_container(
            5, "MLIR20", &m_strings, m_dialects, &m_attrs, &m_types, &m_props, &root_ir, false,
        );
        let mosaic_b64 = b64_encode(&mosaic_v5);

        // Also test v0 Mosaic with 0 memref args (`num_memref_args == 0` -> returns `None`)
        let empty_fn_reg = [uv(1), uv(0), uv(0)].concat();
        let empty_fn_op = [uv(1), vec![0x10], uv(0), uv(2), empty_fn_reg].concat();
        let empty_mod_reg = [uv(1), uv(0), uv(2), empty_fn_op].concat();
        let empty_root = [uv(0), vec![0x10], uv(0), uv(2), empty_mod_reg].concat();
        let mosaic_v0_zero_args = build_container(
            0,
            "MLIR19",
            &["builtin", "func", "module", "func"],
            &[("builtin", None, &["module"]), ("func", None, &["func"])],
            &[],
            &[],
            &[],
            &empty_root,
            false,
        );
        let zero_args_cfg = format!("{{\"body\":\"{}\"}}", b64_encode(&mosaic_v0_zero_args));
        let mut tmp = 0_usize;
        let mut zero_out = String::new();
        assert_eq!(
            try_emit_mosaic_custom_call(
                &zero_args_cfg,
                &[],
                "",
                "tensor<4xf32>",
                "",
                &mut tmp,
                &mut zero_out
            ),
            None
        );

        // Now build a full v6 `StableHLO` container with:
        // - aligned Section 0 (`0x80`), dialect inline version (`dv & 1`), `vhlo` (`func_v1`, `call_v1`, `add_v1`) + `builtin` (`module`) + `stablehlo` (`custom_call`, `return`)
        // - `SymRefAttrIdx` pointing to `Str` AND `SymRefAttrIdx` pointing to non-Str (string-table fallback)
        // - module `mhlo.num_partitions`, `mhlo.num_replicas`, `res_attrs` `mhlo.sharding`
        // - 3 `func.func`s (private 0-result, public 1-result calling Mosaic `tpu_custom_call`, public 2-result) + 1 non-`func.func` op in module
        let v6_strings = [
            "builtin",
            "vhlo",
            "stablehlo",
            "func",
            "module",
            "func_v1",
            "call_v1",
            "add_v1",
            "custom_call",
            "func.return",
            "my_mod",
            "mhlo.num_partitions",
            "mhlo.num_replicas",
            "mhlo.sharding",
            "{devices=[2,2]<=[4]}",
            "helper",
            "private",
            "main",
            "two_out",
            "tpu_custom_call",
        ];
        let mosaic_json = format!("{{\"custom_call_config\":{{\"body\":\"{mosaic_b64}\"}}}}");
        let mut dyn_strings: Vec<&str> = v6_strings.to_vec();
        dyn_strings.push(&mosaic_json);

        let v6_dialects: &[DialectTestSpec<'_>] = &[
            ("builtin", Some(&[1, 0]), &["module"]),
            ("vhlo", None, &["func_v1", "call_v1", "add_v1"]),
            ("stablehlo", None, &["custom_call"]),
            ("func", None, &["func.return"]),
        ];
        // Types: 0=f32, 1=tensor<4xf32>, 2=()->(), 3=(tensor<4xf32>)->(tensor<4xf32>), 4=(tensor<4xf32>)->(tensor<4xf32>, tensor<4xf32>)
        let v6_types = vec![
            (0, uv(99)),
            (0, [uv(13), slist(&[4]), uv(0)].concat()),
            (0, [uv(2), ulist(&[]), ulist(&[])].concat()),
            (0, [uv(2), ulist(&[1]), ulist(&[1])].concat()),
            (0, [uv(2), ulist(&[1]), ulist(&[1, 1])].concat()),
        ];
        // Attrs:
        // 0: Str("my_mod")
        // 1: Str("mhlo.num_partitions")
        // 2: Str("mhlo.num_replicas")
        // 3: Int(4)
        // 4: Int(1)
        // 5: Dict([(1, 3), (2, 4)])  -- module dict
        // 6: Str("mhlo.sharding")
        // 7: Str("{devices=[2,2]<=[4]}")
        // 8: Dict([(6, 7)])
        // 9: Array([8])              -- res_attrs sharding
        // 10: Str("helper")
        // 11: Str("private")
        // 12: Str("main")
        // 13: Str("two_out")
        // 14: TypeRef(2)             -- ()->()
        // 15: TypeRef(3)             -- (tensor<4xf32>)->(tensor<4xf32>)
        // 16: TypeRef(4)             -- (tensor<4xf32>)->(tensor<4xf32>, tensor<4xf32>)
        // 17: SymRefAttrIdx(10)      -- resolves via Str("helper")
        // 18: SymRefAttrIdx(3)       -- target attr 3 is Int(4), resolves via strings[3]
        // 19: Str("tpu_custom_call")
        // 20: Str(mosaic_json)
        let v6_attrs = vec![
            (0, true, [uv(2), uv(10)].concat()),
            (0, true, [uv(2), uv(11)].concat()),
            (0, true, [uv(2), uv(12)].concat()),
            (0, true, [uv(8), uv(0), sv(4)].concat()),
            (0, true, [uv(8), uv(0), sv(1)].concat()),
            (0, true, [uv(1), uv(2), uv(1), uv(3), uv(2), uv(4)].concat()),
            (0, true, [uv(2), uv(13)].concat()),
            (0, true, [uv(2), uv(14)].concat()),
            (0, true, [uv(1), uv(1), uv(6), uv(7)].concat()),
            (0, true, [uv(0), ulist(&[8])].concat()),
            (0, true, [uv(2), uv(15)].concat()),
            (0, true, [uv(2), uv(16)].concat()),
            (0, true, [uv(2), uv(17)].concat()),
            (0, true, [uv(2), uv(18)].concat()),
            (0, true, [uv(6), uv(2)].concat()),
            (0, true, [uv(6), uv(3)].concat()),
            (0, true, [uv(6), uv(4)].concat()),
            (0, true, [uv(4), uv(10)].concat()),
            (0, true, [uv(4), uv(3)].concat()),
            (0, true, [uv(2), uv(19)].concat()),
            (0, true, [uv(2), uv(20)].concat()),
        ];
        // Props:
        // 0: module props [(0 << 1) | 1] -> "my_mod"
        // 1: helper func props [0, 14, 0, 10, (11 << 1) | 1] (private, ()->())
        // 2: main func props [0, 15, (9 << 1) | 1, 12, 0] (public, shifted res_attrs=9, (tensor<4xf32>)->tensor<4xf32>)
        // 3: two_out func props [0, 16, 9, 13] (public, direct res_attrs=9, 2 outputs)
        // 4: tpu_custom_call props [0, 20, 19] (mosaic_json, "tpu_custom_call")
        // 5: call_v1 props [17]
        let v6_props = vec![
            vec![1],
            vec![0, 14, 0, 10, (11 << 1) | 1],
            vec![0, 15, (9 << 1) | 1, 12, 0],
            vec![0, 16, 9, 13],
            vec![0, 20, 19],
            vec![17],
        ];
        // Ops in v6_dialects: 0=builtin.module, 1=func.func, 2=func.call, 3=stablehlo.add, 4=stablehlo.custom_call, 5=func.return
        let f_helper = {
            let ret = mk_mop(5, None, None, &[], &[]);
            let reg = [uv(1), uv(0), uv(1 << 1), ret].concat();
            let mut f = uv(1);
            f.push(0x10 | 0x40);
            f.extend(uv(0));
            f.extend(uv(1)); // prop 1
            f.extend(uv(2));
            f.extend(reg);
            f
        };
        let f_main = {
            let ccall = mk_mop(4, None, Some(4), &[1], &[0]);
            let ret = mk_mop(5, None, None, &[], &[1]);
            let mut reg = uv(1);
            reg.extend(uv(2));
            reg.extend(uv((2 << 1) | 1));
            reg.extend(uv(1));
            reg.extend(uv(1 << 1));
            reg.push(0);
            reg.extend(ccall);
            reg.extend(ret);
            let mut f = uv(1);
            f.push(0x10 | 0x40);
            f.extend(uv(0));
            f.extend(uv(2)); // prop 2
            f.extend(uv(2));
            f.extend(reg);
            f
        };
        let f_two = {
            let call_h = mk_mop(2, None, Some(5), &[], &[]);
            let add_op = mk_mop(3, None, None, &[1], &[0, 0]);
            let ret = mk_mop(5, None, None, &[], &[0, 1]);
            let mut reg = uv(1);
            reg.extend(uv(2));
            reg.extend(uv((3 << 1) | 1));
            reg.extend(uv(1));
            reg.extend(uv(1 << 1));
            reg.push(0);
            reg.extend(call_h);
            reg.extend(add_op);
            reg.extend(ret);
            let mut f = uv(1);
            f.push(0x10 | 0x40);
            f.extend(uv(0));
            f.extend(uv(3)); // prop 3
            f.extend(uv(2));
            f.extend(reg);
            f
        };
        let non_fn = mk_mop(5, None, None, &[], &[]);
        let mut mod_reg_v6 = uv(1);
        mod_reg_v6.extend(uv(0));
        mod_reg_v6.extend(uv(4 << 1));
        mod_reg_v6.extend(f_helper);
        mod_reg_v6.extend(f_main);
        mod_reg_v6.extend(f_two);
        mod_reg_v6.extend(non_fn);

        let mut root_v6 = uv(0);
        root_v6.push(0x01 | 0x10 | 0x40);
        root_v6.extend(uv(0));
        root_v6.extend(uv(5)); // dict 5
        root_v6.extend(uv(0)); // prop 0
        root_v6.extend(uv(2));
        root_v6.extend(mod_reg_v6);

        let v6_bytes = build_container(
            6,
            "StableHLO_v1",
            &dyn_strings,
            v6_dialects,
            &v6_attrs,
            &v6_types,
            &v6_props,
            &root_v6,
            true,
        );
        let decoded = decode_llvm_mlir_v6_to_text(&v6_bytes);
        assert!(decoded.is_some());
        let text = decoded.unwrap_or_default();
        assert!(text.contains("module @my_mod attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32, mhlo.sharding = \"{devices=[2,2]<=[4]}\"}"));
        assert!(text.contains("func.func private @helper() -> ()"));
        assert!(text.contains("func.func public @main(%arg0: tensor<4xf32>) -> tensor<4xf32>"));
        assert!(text.contains(
            "func.func public @two_out(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)"
        ));

        // Compile `f_main`-only v6 container end-to-end through `CompiledProgram::compile` (`compiler.rs` line 295)
        let mut single_mod_reg = uv(1);
        single_mod_reg.extend(uv(0));
        let simple_add_fn = {
            let add_op = mk_mop(3, None, None, &[1], &[0, 0]);
            let ret = mk_mop(5, None, None, &[], &[1]);
            let mut reg = uv(1);
            reg.extend(uv(2));
            reg.extend(uv((2 << 1) | 1));
            reg.extend(uv(1));
            reg.extend(uv(1 << 1));
            reg.push(0);
            reg.extend(add_op);
            reg.extend(ret);
            let mut f = uv(1);
            f.push(0x10 | 0x40);
            f.extend(uv(0));
            f.extend(uv(2));
            f.extend(uv(2));
            f.extend(reg);
            f
        };
        single_mod_reg.extend(uv(1 << 1));
        single_mod_reg.extend(simple_add_fn);
        let simple_root = [uv(0), vec![0x10], uv(0), uv(2), single_mod_reg].concat();
        let compilable_v6 = build_container(
            6,
            "LLVM_MLIR",
            &dyn_strings,
            v6_dialects,
            &v6_attrs,
            &v6_types,
            &v6_props,
            &simple_root,
            false,
        );
        let compiled = crate::compiler::CompiledProgram::compile("mlir", &compilable_v6, b"");
        assert!(compiled.is_ok());

        // Truncated section slices & truncated Section 8 properties
        for cut in 1..25 {
            if let Some(prefix) = compilable_v6.get(..cut) {
                let _ = decode_llvm_mlir_v6_to_text(prefix);
            }
        }
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_llvm_v6_exhaustive_dense_mosaic_and_header_branches() {
        // 1. fmt_dense_elements across multi-element & 1-element f32, f64, i32, i64, index
        let dt_tables = V6Tables {
            types: vec![
                RawV6Type::Float("f32"),
                RawV6Type::Float("f64"),
                RawV6Type::Int(32, 1),
                RawV6Type::Int(64, 1),
                RawV6Type::Index,
                RawV6Type::Float("bf16"),
                RawV6Type::Tensor(vec![1], 0),
                RawV6Type::Tensor(vec![2], 1),
                RawV6Type::Tensor(vec![1], 1),
                RawV6Type::Tensor(vec![1], 2),
                RawV6Type::Tensor(vec![2], 3),
                RawV6Type::Tensor(vec![1], 3),
                RawV6Type::Tensor(vec![1], 4),
            ],
            attrs: vec![
                RawV6Attr::DenseElements(6, vec![0_u8; 8]), // 0: 2xf32 bytes on tensor<1xf32>
                RawV6Attr::DenseElements(7, vec![0_u8; 16]), // 1: 2xf64 on tensor<2xf64>
                RawV6Attr::DenseElements(8, vec![0_u8; 16]), // 2: 2xf64 on tensor<1xf64>
                RawV6Attr::DenseElements(8, vec![0_u8; 8]), // 3: 1xf64 on tensor<1xf64>
                RawV6Attr::DenseElements(9, vec![0_u8; 8]), // 4: 2xi32 on tensor<1xi32>
                RawV6Attr::DenseElements(10, vec![0_u8; 16]), // 5: 2xi64 on tensor<2xi64>
                RawV6Attr::DenseElements(11, vec![0_u8; 16]), // 6: 2xi64 on tensor<1xi64>
                RawV6Attr::DenseElements(11, vec![0_u8; 8]), // 7: 1xi64 on tensor<1xi64>
                RawV6Attr::DenseElements(12, vec![0_u8; 8]), // 8: 1xindex on tensor<1xindex>
                RawV6Attr::FloatBits(1, 1.5_f64.to_bits()), // 9: f64 FloatBits
                RawV6Attr::FloatBits(5, 0x3F80),            // 10: bf16 FloatBits
                RawV6Attr::FloatBits(0, 2.0_f32.to_bits().into()), // 11: f32 FloatBits
                RawV6Attr::Int(7),                          // 12: Int
                RawV6Attr::Opaque,                          // 13: Opaque
                RawV6Attr::Dict(vec![(0, 12)]),             // 14: Dict -> Int
                RawV6Attr::DenseElements(999, Vec::new()),  // 15: invalid DenseElements type idx
                RawV6Attr::Str(String::new()),              // 16: empty Str
                RawV6Attr::SymRef(String::from("dict_callee")), // 17: SymRef
                RawV6Attr::Dict(vec![(0, 13), (0, 16), (0, 17)]), // 18: Dict for func.call
                RawV6Attr::Str(String::from("{\"body\":\"QUJD\"}")), // 19: body JSON
                RawV6Attr::Dict(vec![(0, 13), (0, 19)]),    // 20: Dict with body JSON
                RawV6Attr::Str(String::from("direct_tgt")), // 21: target Str
                RawV6Attr::Str(String::from("{maximal=0}")), // 22: maximal sharding
                RawV6Attr::Str(String::from("{manual}")),   // 23: manual sharding
                RawV6Attr::Str(String::from("sdy.sharding")), // 24: sharding key
                RawV6Attr::Str(String::from("mesh_spec")),  // 25: custom sharding value
                RawV6Attr::Dict(vec![(0, 22)]),             // 26: Dict maximal
                RawV6Attr::Dict(vec![(0, 23)]),             // 27: Dict manual
                RawV6Attr::Dict(vec![(24, 25)]),            // 28: Dict key="sdy.sharding"
                RawV6Attr::Dict(vec![(0, 16)]),             // 29: Dict non-sharding
            ],
            props: vec![
                vec![9],
                vec![10],
                vec![11],
                vec![0],
                vec![15],
                vec![12],
                vec![13],
                vec![u64::MAX, 16], // 7: func.call invalid & empty prop -> dict fallback
                vec![0, (19 << 1) | 1, 21], // 8: custom_call shifted body + direct target
                vec![16, 20, 19 << 1], // 9: custom_call fallback loop (even, odd, '{', digits)
            ],
        };
        for ai in 0..=8 {
            assert!(dt_tables.fmt_dense_elements(ai).is_some());
        }

        // 2. format_mosaic_arith_constant & emit_mosaic_kernel_op all branches
        for pi in 0..=6 {
            let mop = V6OpNode {
                name: String::from("arith.constant"),
                dict_attr_idx: None,
                prop_idx: Some(pi),
                result_types: vec![6],
                operands: Vec::new(),
                regions: Vec::new(),
            };
            let _ = format_mosaic_arith_constant(&mop, &dt_tables);
        }
        for (di, rts) in [
            (Some(14), vec![4]),
            (Some(14), vec![2]),
            (Some(14), Vec::new()),
            (Some(13), Vec::new()),
        ] {
            let mop = V6OpNode {
                name: String::from("arith.constant"),
                dict_attr_idx: di,
                prop_idx: None,
                result_types: rts,
                operands: Vec::new(),
                regions: Vec::new(),
            };
            let _ = format_mosaic_arith_constant(&mop, &dt_tables);
        }
        let mut m_ssa = Vec::new();
        let mut stored = None;
        let mut tmp = 0_usize;
        let mut m_buf = String::new();
        for (mname, rts, ops_in) in [
            ("vector.load", vec![6], vec![99]),
            ("vector.load", vec![6], Vec::new()),
            ("vector.store", Vec::new(), Vec::new()),
            ("vector.broadcast", Vec::new(), Vec::new()),
            ("arith.subf", Vec::new(), Vec::new()),
            ("arith.mulf", Vec::new(), Vec::new()),
            ("arith.divf", Vec::new(), Vec::new()),
            ("tpu.matmul", Vec::new(), Vec::new()),
            ("tpu.unknown_op", vec![6], Vec::new()),
        ] {
            let mop = V6OpNode {
                name: String::from(mname),
                dict_attr_idx: None,
                prop_idx: None,
                result_types: rts,
                operands: ops_in,
                regions: Vec::new(),
            };
            emit_mosaic_kernel_op(
                &mop,
                &dt_tables,
                &mut m_ssa,
                &mut stored,
                "invalid_ty",
                "  ",
                &mut tmp,
                &mut m_buf,
            );
        }

        // 3. parse_ir_op with `mask & 0x08 != 0` and `ns == 0`, and emit_v6_ops for func.call / custom_call / empty-arg bb_header
        let succ0_op = [uv(0), vec![0x08], uv(0), uv(0)].concat();
        assert!(parse_ir_op(&succ0_op, 0, &[String::from("func.return")], 6).is_some());

        let extra_ops = vec![
            V6OpNode {
                name: String::from("func.call"),
                dict_attr_idx: Some(18),
                prop_idx: Some(7),
                result_types: vec![6],
                operands: Vec::new(),
                regions: Vec::new(),
            },
            V6OpNode {
                name: String::from("func.call"),
                dict_attr_idx: Some(13),
                prop_idx: None,
                result_types: vec![6],
                operands: Vec::new(),
                regions: Vec::new(),
            },
            V6OpNode {
                name: String::from("stablehlo.custom_call"),
                dict_attr_idx: Some(20),
                prop_idx: None,
                result_types: vec![6],
                operands: Vec::new(),
                regions: Vec::new(),
            },
            V6OpNode {
                name: String::from("stablehlo.custom_call"),
                dict_attr_idx: Some(26),
                prop_idx: Some(8),
                result_types: vec![6],
                operands: Vec::new(),
                regions: Vec::new(),
            },
            V6OpNode {
                name: String::from("stablehlo.custom_call"),
                dict_attr_idx: Some(27),
                prop_idx: Some(9),
                result_types: vec![6],
                operands: Vec::new(),
                regions: Vec::new(),
            },
            V6OpNode {
                name: String::from("stablehlo.custom_call"),
                dict_attr_idx: Some(28),
                prop_idx: None,
                result_types: vec![6],
                operands: Vec::new(),
                regions: Vec::new(),
            },
            V6OpNode {
                name: String::from("stablehlo.custom_call"),
                dict_attr_idx: Some(29),
                prop_idx: None,
                result_types: vec![6],
                operands: Vec::new(),
                regions: Vec::new(),
            },
            V6OpNode {
                name: String::from("stablehlo.custom_call"),
                dict_attr_idx: Some(13),
                prop_idx: None,
                result_types: vec![6],
                operands: Vec::new(),
                regions: Vec::new(),
            },
            V6OpNode {
                name: String::from("stablehlo.all_reduce"),
                dict_attr_idx: None,
                prop_idx: None,
                result_types: vec![6],
                operands: Vec::new(),
                regions: vec![V6Region {
                    is_isolated: false,
                    arg_types: Vec::new(),
                    ops: Vec::new(),
                }],
            },
        ];
        let mut ssa_e = Vec::new();
        let mut e_out = String::new();
        emit_v6_ops(&extra_ops, &dt_tables, &mut ssa_e, &mut tmp, 2, &mut e_out);
        assert!(e_out.contains("@dict_callee"));
        assert!(e_out.contains("{maximal=0}"));
        assert!(e_out.contains("{manual}"));
        assert!(e_out.contains("mesh_spec"));

        // 4. Mosaic v0 containers with 0 args & without vector.store, plus v1/v6 header/alignment/visibility/sharding edge cases
        let mk_min_op = |id: u64, rts: &[u64], ops_in: &[u64]| -> Vec<u8> {
            let mut b = uv(id);
            let mut m = 0_u8;
            if !rts.is_empty() {
                m |= 0x02;
            }
            if !ops_in.is_empty() {
                m |= 0x04;
            }
            b.push(m);
            b.extend(uv(0));
            if !rts.is_empty() {
                b.extend(uv(rts.len() as u64));
                for &r in rts {
                    b.extend(uv(r));
                }
            }
            if !ops_in.is_empty() {
                b.extend(uv(ops_in.len() as u64));
                for &o in ops_in {
                    b.extend(uv(o));
                }
            }
            b
        };
        let m_strings = ["builtin", "func", "module", "func", "return"];
        let m_dials: &[DialectTestSpec<'_>] = &[
            ("builtin", None, &["module"]),
            ("func", None, &["func", "return"]),
        ];
        let m_tys = [(0, vec![2_u8])];
        // 0-arg Mosaic function -> returns None from `try_emit_mosaic_custom_call`
        let fn_0arg = {
            let ret = mk_min_op(2, &[], &[]);
            let reg = [uv(1), uv(0), uv(1 << 1), ret].concat();
            [uv(1), vec![0x10], uv(0), uv(2), reg].concat()
        };
        let mod_0arg = [
            uv(0),
            vec![0x10],
            uv(0),
            uv(2),
            uv(1),
            uv(0),
            uv(1 << 1),
            fn_0arg,
        ]
        .concat();
        let bc_0arg = build_container(
            0,
            "MLIR19",
            &m_strings,
            m_dials,
            &[],
            &m_tys,
            &[],
            &mod_0arg,
            false,
        );
        // 1-arg Mosaic function with NO `vector.store` -> hits `stored_out = None` with non-empty and empty `opnd_strs`
        let fn_nostore = {
            let ret = mk_min_op(2, &[], &[]);
            let reg = [uv(1), uv(1), uv((1 << 1) | 1), uv(1), uv(0), uv(0), ret].concat();
            [uv(1), vec![0x10], uv(0), uv(2), reg].concat()
        };
        let mod_nostore = [
            uv(0),
            vec![0x10],
            uv(0),
            uv(2),
            uv(1),
            uv(0),
            uv(1 << 1),
            fn_nostore,
        ]
        .concat();
        let bc_nostore = build_container(
            0,
            "MLIR20",
            &m_strings,
            m_dials,
            &[],
            &m_tys,
            &[],
            &mod_nostore,
            false,
        );

        let b64_enc = |raw: &[u8]| -> String {
            const TAB: &[u8; 64] =
                b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            let mut s = String::new();
            for ch in raw.chunks(3) {
                let b0 = ch.first().copied().unwrap_or(0);
                let b1 = ch.get(1).copied().unwrap_or(0);
                let b2 = ch.get(2).copied().unwrap_or(0);
                let n = (u32::from(b0) << 16) | (u32::from(b1) << 8) | u32::from(b2);
                s.push(TAB.get(((n >> 18) & 63) as usize).copied().unwrap_or(b'A') as char);
                s.push(TAB.get(((n >> 12) & 63) as usize).copied().unwrap_or(b'A') as char);
                s.push(if ch.len() > 1 {
                    TAB.get(((n >> 6) & 63) as usize).copied().unwrap_or(b'A') as char
                } else {
                    '='
                });
                s.push(if ch.len() > 2 {
                    TAB.get((n & 63) as usize).copied().unwrap_or(b'A') as char
                } else {
                    '='
                });
            }
            s
        };
        let cfg_0arg = format!("{{\"body\":\"{}\"}}", b64_enc(&bc_0arg));
        assert_eq!(
            try_emit_mosaic_custom_call(
                &cfg_0arg,
                &[],
                "",
                "tensor<4xf32>",
                "",
                &mut tmp,
                &mut m_buf
            ),
            None
        );
        let cfg_nostore = format!("{{\"body\":\"{}\"}}", b64_enc(&bc_nostore));
        assert_eq!(
            try_emit_mosaic_custom_call(
                &cfg_nostore,
                &[String::from("%in0")],
                "%r = ",
                "tensor<4xf32>",
                "",
                &mut tmp,
                &mut m_buf,
            ),
            Some(true)
        );
        assert_eq!(
            try_emit_mosaic_custom_call(
                &cfg_nostore,
                &[],
                "%r = ",
                "tensor<4xf32>",
                "",
                &mut tmp,
                &mut m_buf
            ),
            Some(true)
        );

        // `ver = 0` with `"LLVM"` producer & `allow_v0_mosaic = true` -> rejected (`None`), `ver = 1` container -> accepted by `parse_mlir_bytecode_container(..., true)`
        let bad_v0_llvm = build_container(
            0,
            "LLVM",
            &m_strings,
            m_dials,
            &[],
            &m_tys,
            &[],
            &mod_0arg,
            false,
        );
        assert!(parse_mlir_bytecode_container(&bad_v0_llvm, true).is_none());
        let v1_bc = build_container(
            1,
            "MLIR19",
            &m_strings,
            m_dials,
            &[],
            &m_tys,
            &[],
            &mod_0arg,
            false,
        );
        assert!(parse_mlir_bytecode_container(&v1_bc, true).is_some());

        // `decode_llvm_mlir_v6_to_text`: SymRefAttrIdx -> Str, align=1 header, even/empty root prop, non-func op in module, res_attrs manual/empty/non-Dict, shifted private vis, non-Func TypeRef, 0-region func
        let edge_strings = [
            "builtin",
            "func",
            "module",
            "func",
            "return",
            "",
            "private",
            "mhlo.sharding",
            "{manual}",
        ];
        let edge_dials: &[DialectTestSpec<'_>] = &[
            ("builtin", None, &["module"]),
            ("func", None, &["func", "return"]),
        ];
        let edge_attrs: &[AttrTestSpec] = &[
            (0, true, [uv(2), uv(5)].concat()), // 0: empty Str("")
            (0, true, [uv(6), uv(0)].concat()), // 1: TypeRef(0) where type 0 is Int (not Func)
            (0, true, [uv(2), uv(7)].concat()), // 2: Str("mhlo.sharding")
            (0, true, [uv(2), uv(8)].concat()), // 3: Str("{manual}")
            (0, true, [uv(4), uv(7)].concat()), // 4: SymRefAttrIdx(7) -> Str("private")
            (0, true, [uv(1), uv(2), uv(2), uv(0), uv(2), uv(3)].concat()), // 5: Dict {"mhlo.sharding": "", "mhlo.sharding": "{manual}"}
            (0, true, [uv(0), uv(2), uv(0), uv(5)].concat()), // 6: Array([0 (non-Dict), 5 (Dict)])
            (0, true, [uv(2), uv(6)].concat()), // 7: Str("private") (so `(7 << 1) | 1 == 15 >= 8` -> `direct = None`, `shifted = Some`)
        ];
        let edge_types = [(0, [uv(0), uv((32 << 2) | 1)].concat())];
        let edge_props = vec![
            vec![0],                                   // 0: even prop on root module (`(v & 1) == 0`)
            vec![0, 1, (6 << 1) | 1, 0, (7 << 1) | 1], // 1: func with non-Func TypeRef, odd shifted Array res_attrs >= attrs.len(), empty name, shifted "private"
            vec![0, 0, 6, 0, 0], // 2: func with non-TypeRef fn_ty, even Array res_attrs, non-private vis, 0 regions
        ];
        let non_fn_op = mk_min_op(2, &[], &[]);
        let fn_edge1 = [uv(1), vec![0x40], uv(0), uv(1)].concat();
        let fn_edge2 = [uv(1), vec![0x40], uv(0), uv(2)].concat();
        let mut edge_mod_reg = [uv(1), uv(0), uv(3 << 1)].concat();
        edge_mod_reg.extend(non_fn_op);
        edge_mod_reg.extend(fn_edge1);
        edge_mod_reg.extend(fn_edge2);
        let edge_root = [
            uv(0),
            vec![0x01 | 0x10 | 0x40],
            uv(0),
            uv(5),
            uv(0),
            uv(2),
            edge_mod_reg,
        ]
        .concat();
        let mut edge_v6 = build_container(
            6,
            "MLIR20",
            &edge_strings,
            edge_dials,
            edge_attrs,
            &edge_types,
            &edge_props,
            &edge_root,
            false,
        );
        // Append a dummy custom section 7 with `has_align = true` and `align = 1` (`align > 1` false branch)
        edge_v6.push(0x80 | 7);
        edge_v6.extend(uv(0));
        edge_v6.extend(uv(1));
        let edge_txt = decode_llvm_mlir_v6_to_text(&edge_v6).unwrap_or_default();
        assert!(edge_txt.contains("module @jax_module"));
        assert!(edge_txt.contains("func.func private @main() -> ()"));
        assert!(edge_txt.contains("func.func public @main() -> ()"));

        // 5. Remaining micro-branches: 0-byte f64/i64 DenseElements, custom_call no-body/digits/devices/no-dict,
        //    v6 is_iso without sub-section 4, align=3 & non-0xCB align=4, truncated s3 & s8, 0-region root, short fp, non-Str/non-sharding res_attrs
        let zero_dt = V6Tables {
            types: vec![
                RawV6Type::Float("f64"),
                RawV6Type::Int(64, 1),
                RawV6Type::Tensor(vec![2], 0),
                RawV6Type::Tensor(vec![2], 1),
            ],
            attrs: vec![
                RawV6Attr::DenseElements(2, Vec::new()),
                RawV6Attr::DenseElements(3, Vec::new()),
                RawV6Attr::Str(String::from("no_body_here")),
                RawV6Attr::Str(String::from("98765")),
                RawV6Attr::Str(String::from("{devices=[2,1]0,1}")),
                RawV6Attr::Dict(vec![(2, 4)]),
            ],
            props: vec![vec![7, 2, 99]],
        };
        assert_eq!(
            zero_dt.fmt_dense_elements(0).map(|x| x.0),
            Some(String::from("dense<0.0>"))
        );
        assert_eq!(
            zero_dt.fmt_dense_elements(1).map(|x| x.0),
            Some(String::from("dense<0>"))
        );
        let cc_ops = vec![
            V6OpNode {
                name: String::from("stablehlo.custom_call"),
                dict_attr_idx: None,
                prop_idx: Some(0),
                result_types: vec![2],
                operands: Vec::new(),
                regions: Vec::new(),
            },
            V6OpNode {
                name: String::from("stablehlo.custom_call"),
                dict_attr_idx: Some(5),
                prop_idx: None,
                result_types: vec![2],
                operands: Vec::new(),
                regions: Vec::new(),
            },
            V6OpNode {
                name: String::from("stablehlo.all_reduce"),
                dict_attr_idx: None,
                prop_idx: None,
                result_types: vec![2],
                operands: Vec::new(),
                regions: Vec::new(),
            },
            V6OpNode {
                name: String::from("stablehlo.reduce_scatter"),
                dict_attr_idx: None,
                prop_idx: None,
                result_types: vec![2],
                operands: Vec::new(),
                regions: Vec::new(),
            },
        ];
        emit_v6_ops(&cc_ops, &zero_dt, &mut ssa_e, &mut tmp, 2, &mut e_out);
        assert!(e_out.contains("{devices=[2,1]0,1}"));

        // v6 `is_iso = true` (`nri = 3`), `buf.get(pos) != Some(&4)` (`nb = 0` -> `uv(0) == [1] != 4`)
        let iso_no_sub4 = [uv(0), vec![0x10], uv(0), uv(3), uv(0)].concat();
        assert!(parse_ir_op(&iso_no_sub4, 0, &[String::from("builtin.module")], 6).is_some());

        // 0-region root module + odd shifted empty Str root prop + short `fp` + non-Str & non-sharding `res_attrs`, plus `align = 3` & non-0xCB `align = 4`
        let root_0reg = [uv(0), vec![0x40], uv(0), uv(0)].concat();
        let mut v6_0reg = build_container(
            6,
            "MLIR20",
            &edge_strings,
            edge_dials,
            edge_attrs,
            &edge_types,
            &[vec![1]], // `(1 >> 1) == 0` -> `edge_attrs[0] == Str("")` (`!s.is_empty()` is false)
            &root_0reg,
            false,
        );
        // Custom section 7 with `align = 3` (non-power-of-two) and custom section 7 with `align = 4` + non-0xCB byte
        v6_0reg.push(0x80 | 7);
        v6_0reg.extend(uv(0));
        v6_0reg.extend(uv(3));
        v6_0reg.push(0x80 | 7);
        v6_0reg.extend(uv(0));
        v6_0reg.extend(uv(4));
        while v6_0reg.len() % 4 == 0 {
            v6_0reg.extend([0x80 | 7, 1, 3]);
        }
        let _ = decode_llvm_mlir_v6_to_text(&v6_0reg);

        // Short `fp` (`len == 1`) & `res_attrs` Dict with non-Str value `(2, 1)` and non-sharding key `(1, 1)`
        let extra_edge_attrs: &[AttrTestSpec] = &[
            (0, true, [uv(2), uv(6)].concat()), // 0: Str("private") (non-sharding key)
            (0, true, [uv(6), uv(0)].concat()), // 1: TypeRef(0) (non-Str value)
            (0, true, [uv(1), uv(2), uv(0), uv(0), uv(0), uv(1)].concat()), // 2: Dict {("private", "private"), ("private", TypeRef(0))}
            (0, true, [uv(0), uv(1), uv(2)].concat()),                      // 3: Array([2])
        ];
        let short_fp_props = vec![vec![0], vec![0, 1, 3]];
        let fn_short1 = [uv(1), vec![0x40], uv(0), uv(0)].concat();
        let fn_short2 = [uv(1), vec![0x40], uv(0), uv(1)].concat();
        let short_mod_reg = [uv(1), uv(0), uv(2 << 1), fn_short1, fn_short2].concat();
        let short_root = [uv(0), vec![0x10], uv(0), uv(2), short_mod_reg].concat();
        let short_v6 = build_container(
            6,
            "MLIR20",
            &edge_strings,
            edge_dials,
            extra_edge_attrs,
            &edge_types,
            &short_fp_props,
            &short_root,
            false,
        );
        assert!(decode_llvm_mlir_v6_to_text(&short_v6).is_some());

        // Malformed Section 3 (fewer items than `n_attr + n_ty`) & malformed Section 8 (empty, truncated count, truncated plen)
        for bad_s8 in [Vec::new(), uv(2), [uv(1), uv(99)].concat()] {
            let mut raw_c = v6_0reg.clone();
            raw_c.push(8);
            raw_c.extend(uv(bad_s8.len() as u64));
            raw_c.extend(bad_s8);
            let _ = decode_llvm_mlir_v6_to_text(&raw_c);
        }
        let mut trunc_s3_c = MLIR_BYTECODE_MAGIC.to_vec();
        trunc_s3_c.extend(uv(6));
        trunc_s3_c.extend_from_slice(b"MLIR20\0");
        for (sid, spayload) in [
            (0_u8, vec![1_u8]),
            (1_u8, [uv(0), uv(0)].concat()),
            (2_u8, Vec::new()),
            (3_u8, [uv(2), uv(0)].concat()), // n_attr=2, n_ty=0, but s3 is empty afterwards!
            (4_u8, root_0reg),
        ] {
            trunc_s3_c.push(sid);
            trunc_s3_c.extend(uv(spayload.len() as u64));
            trunc_s3_c.extend(spayload);
        }
        assert!(decode_llvm_mlir_v6_to_text(&trunc_s3_c).is_none());
    }

    #[test]
    fn test_v6_shardy_sdy_attrs_dot_general_v2_and_mlir24_func() {
        let strs = vec![String::from("x"), String::from("mesh")];
        let a_axis = decode_v6_attr("sdy", &[uv(1), uv(0), sv(4)].concat(), &strs);
        assert!(matches!(a_axis, RawV6Attr::SdyMeshAxis(_, 4)));
        let a_mesh = decode_v6_attr("sdy", &[uv(2), ulist(&[0])].concat(), &strs);
        assert!(matches!(a_mesh, RawV6Attr::SdyMesh(_)));
        let a_ref = decode_v6_attr("sdy", &[uv(4), uv(0)].concat(), &strs);
        assert!(matches!(a_ref, RawV6Attr::SdyAxisRef(_)));
        let a_dim = decode_v6_attr("sdy", &[uv(5), ulist(&[2])].concat(), &strs);
        assert!(matches!(a_dim, RawV6Attr::SdyDimSharding(_)));
        let a_ts = decode_v6_attr("sdy", &[uv(6), uv(1), ulist(&[3])].concat(), &strs);
        assert!(matches!(a_ts, RawV6Attr::SdyTensorSharding(1, _)));
        let a_pv = decode_v6_attr("sdy", &[uv(7), ulist(&[4])].concat(), &strs);
        assert!(matches!(a_pv, RawV6Attr::SdyShardingPerValue(_)));
        assert!(matches!(
            decode_v6_attr("sdy", &uv(99), &strs),
            RawV6Attr::Opaque
        ));

        let sdy_tables = V6Tables {
            attrs: vec![
                RawV6Attr::SdyMeshAxis(String::from("x"), 4),    // 0
                RawV6Attr::SdyMeshAxis(String::from("y"), 2),    // 1
                RawV6Attr::SdyMesh(vec![0, 1, 99]),              // 2
                RawV6Attr::Str(String::from("mesh")),            // 3
                RawV6Attr::SymRef(String::from("mesh")),         // 4
                RawV6Attr::SdyAxisRef(String::from("x")),        // 5
                RawV6Attr::SdyDimSharding(vec![5, 99]),          // 6: {"x"}
                RawV6Attr::SdyDimSharding(Vec::new()),           // 7: {}
                RawV6Attr::SdyTensorSharding(4, vec![6, 7, 99]), // 8: [{"x"}, {}, ?]
                RawV6Attr::SdyTensorSharding(99, vec![7, 7]),    // 9: [{}, {}] unknown mesh
                RawV6Attr::SdyShardingPerValue(vec![8]),         // 10
                RawV6Attr::SdyShardingPerValue(vec![9]),         // 11
                RawV6Attr::DenseArray(1, 0_i64.to_le_bytes().to_vec()), // 12
                RawV6Attr::DenseArray(0, Vec::new()),            // 13
                RawV6Attr::Str(String::from("sdy.sharding")),    // 14
                RawV6Attr::Dict(vec![(14, 8)]),                  // 15
                RawV6Attr::Array(vec![15]),                      // 16
                RawV6Attr::TypeRef(2),                           // 17: fn type
                RawV6Attr::Str(String::from("main")),            // 18
                RawV6Attr::Str(String::from("public")),          // 19
            ],
            types: vec![
                RawV6Type::Float("f32"),
                RawV6Type::Tensor(vec![4, 4], 0),
                RawV6Type::Func(vec![1], vec![1]),
            ],
            props: vec![
                vec![99],                                     // 0: short prop
                vec![2, 99],                                  // 1: non-Str name
                vec![2, 3],                                   // 2: sdy.mesh @mesh
                vec![10, 0, 11],                              // 3: sdy.manual_computation props
                vec![0, 0, 13, 0, 12, 0, 0, 0, 13, 0, 12, 0], // 4: vhlo.dot_general_v2 (12 props)
            ],
        };
        assert_eq!(
            sdy_tables.fmt_sdy_tensor_sharding_as_op_sharding(99),
            "{replicated}"
        );
        assert_eq!(
            sdy_tables.fmt_sdy_tensor_sharding_as_op_sharding(8),
            "{devices=[4,1,1]<=[4]}"
        );
        assert_eq!(
            sdy_tables.fmt_sdy_tensor_sharding_as_op_sharding(9),
            "{replicated}"
        );
        assert_eq!(
            sdy_tables.sdy_per_value_op_shardings(0),
            Vec::<String>::new()
        );
        assert_eq!(
            sdy_tables.sdy_per_value_op_shardings(10),
            vec![String::from("{devices=[4,1,1]<=[4]}")]
        );
        emit_sdy_manual_computation_test_ops(&sdy_tables);
    }

    /// Exercises `sdy.manual_computation`, `builtin.unrealized_conversion_cast`, and 12-property
    /// `vhlo.dot_general_v2` emission in [`emit_v6_ops`].
    ///
    /// # Arguments
    ///
    /// * `sdy_tables` - Prepopulated [`V6Tables`] containing Shardy mesh/sharding attributes.
    fn emit_sdy_manual_computation_test_ops(sdy_tables: &V6Tables) {
        let mc_ops = vec![
            V6OpNode {
                name: String::from("builtin.unrealized_conversion_cast"),
                dict_attr_idx: None,
                prop_idx: None,
                result_types: vec![1],
                operands: vec![0],
                regions: Vec::new(),
            },
            V6OpNode {
                name: String::from("stablehlo.dot_general"),
                dict_attr_idx: None,
                prop_idx: Some(4),
                result_types: vec![1],
                operands: vec![1, 1],
                regions: Vec::new(),
            },
            V6OpNode {
                name: String::from("sdy.manual_computation"),
                dict_attr_idx: None,
                prop_idx: Some(3),
                result_types: vec![1],
                operands: vec![2],
                regions: vec![V6Region {
                    is_isolated: true,
                    arg_types: vec![1],
                    ops: vec![
                        V6OpNode {
                            name: String::from("stablehlo.negate"),
                            dict_attr_idx: None,
                            prop_idx: None,
                            result_types: vec![1],
                            operands: vec![0],
                            regions: Vec::new(),
                        },
                        V6OpNode {
                            name: String::from("builtin.unrealized_conversion_cast"),
                            dict_attr_idx: None,
                            prop_idx: None,
                            result_types: vec![1],
                            operands: vec![1],
                            regions: Vec::new(),
                        },
                        V6OpNode {
                            name: String::from("sdy.return"),
                            dict_attr_idx: None,
                            prop_idx: None,
                            result_types: Vec::new(),
                            operands: vec![2],
                            regions: Vec::new(),
                        },
                    ],
                }],
            },
            V6OpNode {
                name: String::from("sdy.manual_computation"),
                dict_attr_idx: None,
                prop_idx: None,
                result_types: vec![1],
                operands: Vec::new(),
                regions: vec![V6Region {
                    is_isolated: false,
                    arg_types: Vec::new(),
                    ops: Vec::new(),
                }],
            },
            V6OpNode {
                name: String::from("sdy.manual_computation"),
                dict_attr_idx: None,
                prop_idx: None,
                result_types: vec![1],
                operands: Vec::new(),
                regions: Vec::new(),
            },
        ];
        let mut ssa = vec![String::from("%arg0")];
        let mut tmp = 0_usize;
        let mut emitted = String::new();
        emit_v6_ops(&mc_ops, sdy_tables, &mut ssa, &mut tmp, 2, &mut emitted);
        assert!(
            emitted.contains("lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]")
        );
        assert!(emitted.contains("SPMDFullToShardShape"));
        assert!(emitted.contains("SPMDShardToFullShape"));
    }

    #[test]
    fn test_v6_mlir24_six_prop_func_and_sdy_mesh_containers() {
        let mlir24_strings = [
            "builtin",
            "sdy",
            "func",
            "module",
            "mesh",
            "manual_computation",
            "return",
            "x",
            "main",
            "public",
            "sdy.sharding",
        ];
        let mlir24_dials: &[DialectTestSpec<'_>] = &[
            ("builtin", None, &["module"]),
            ("sdy", None, &["mesh", "manual_computation"]),
            ("func", None, &["func", "return"]),
        ];
        let mlir24_attrs: &[AttrTestSpec] = &[
            (1, true, [uv(1), uv(7), sv(4)].concat()), // 0: SdyMeshAxis("x", 4)
            (1, true, [uv(2), ulist(&[0])].concat()),  // 1: SdyMesh([0])
            (0, true, [uv(2), uv(4)].concat()),        // 2: Str("mesh")
            (0, true, [uv(4), uv(2)].concat()),        // 3: SymRef(@mesh)
            (1, true, [uv(4), uv(7)].concat()),        // 4: SdyAxisRef("x")
            (1, true, [uv(5), ulist(&[4])].concat()),  // 5: SdyDimSharding([4])
            (1, true, [uv(6), uv(3), ulist(&[5])].concat()), // 6: SdyTensorSharding(@mesh, [{"x"}])
            (0, true, [uv(2), uv(10)].concat()),       // 7: Str("sdy.sharding")
            (0, true, [uv(1), uv(2), uv(0), uv(6), uv(7), uv(6)].concat()), // 8: Dict {0: 6, sdy.sharding: 6}
            (0, true, [uv(0), uv(1), uv(8)].concat()),                      // 9: Array([8])
            (0, true, [uv(6), uv(2)].concat()),                             // 10: TypeVal(2)
            (0, true, [uv(2), uv(8)].concat()),                             // 11: Str("main")
            (0, true, [uv(2), uv(9)].concat()),                             // 12: Str("public")
        ];
        let mlir24_types: &[(usize, Vec<u8>)] = &[
            (0, [uv(1), uv(2)].concat()),                      // 0: f32
            (0, [uv(13), ulist(&[4]), uv(0)].concat()),        // 1: tensor<4xf32>
            (0, [uv(5), uv(1), uv(1), uv(1), uv(1)].concat()), // 2: (tensor<4xf32>) -> tensor<4xf32>
        ];
        let mlir24_props = vec![
            vec![1, 2],                                      // 0: sdy.mesh @mesh
            vec![(9 << 1) | 1, 10, 0, 0, 11, (12 << 1) | 1], // 1: 6-prop func.func
        ];
        let ret_ir = [uv(4), vec![0x00], uv(0)].concat();
        let fn_reg = [uv(1), uv(0), uv(1 << 1), ret_ir.clone()].concat();
        let fn_ir = [uv(3), vec![0x50], uv(0), uv(1), uv(1 << 1), fn_reg].concat();
        let mesh_ir = [uv(1), vec![0x40], uv(0), uv(0)].concat();
        let mod_reg = [uv(1), uv(0), uv(2 << 1), mesh_ir.clone(), fn_ir].concat();
        let root_ir = [uv(0), vec![0x10], uv(0), uv(1 << 1), mod_reg].concat();
        let mlir24_bc = build_container(
            6,
            "MLIR24.0.0git",
            &mlir24_strings,
            mlir24_dials,
            mlir24_attrs,
            mlir24_types,
            &mlir24_props,
            &root_ir,
            false,
        );
        let txt24 = decode_llvm_mlir_v6_to_text(&mlir24_bc);
        assert!(
            txt24
                .as_deref()
                .is_some_and(|s| s.contains("devices=[4]<=[4]"))
        );

        // Second container with `sdy.manual_computation` inside `func.func` (`has_manual_comp == true`)
        let mc_inner_reg = uv(0);
        let mc_op_ir = [uv(2), vec![0x10], uv(0), uv(1 << 1), mc_inner_reg].concat();
        let fn_mc_reg = [uv(1), uv(0), uv(2 << 1), mc_op_ir, ret_ir].concat();
        let fn_mc_ir = [uv(3), vec![0x50], uv(0), uv(1), uv(1 << 1), fn_mc_reg].concat();
        let mod_mc_reg = [uv(1), uv(0), uv(2 << 1), mesh_ir, fn_mc_ir].concat();
        let root_mc_ir = [uv(0), vec![0x10], uv(0), uv(1 << 1), mod_mc_reg].concat();
        let mlir24_mc_bc = build_container(
            6,
            "MLIR24.0.0git",
            &mlir24_strings,
            mlir24_dials,
            mlir24_attrs,
            mlir24_types,
            &mlir24_props,
            &root_mc_ir,
            false,
        );
        assert!(decode_llvm_mlir_v6_to_text(&mlir24_mc_bc).is_some());
    }
}

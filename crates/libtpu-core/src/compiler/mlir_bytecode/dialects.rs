//! `OpenXLA` Shardy (`sdy`) and JAX Mosaic (`tpu`) custom dialect bytecode readers and writers.
//!
//! Implements `SdyAttrCode` (`0..=5` matching `openxla/shardy/dialect/sdy/ir/bytecode.cpp`) and
//! `TpuMosaicBytecodeCode` (`#tpu.tiled`, `#tpu.memory_space`, `#tpu.dimension_semantics`,
//! `!tpu.semaphore`, `!tpu.dma_semaphore`).

use super::stream::{DialectBytecodeReader, DialectBytecodeWriter, MlirSectionId};
use crate::compiler::ast::{
    MlirAttribute, MlirType, MosaicDialectAttr, SdyDimensionSharding, SdyMeshAttr,
    SdyTensorShardingAttr, TpuMemorySpace, TpuSemaphoreKind,
};
use crate::error::{LibTpuError, LibTpuResult};

/// `OpenXLA` Shardy (`sdy`) custom attribute tags (`0..=5`) from `shardy/dialect/sdy/ir/bytecode.cpp`.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SdyAttrCode {
    /// `#sdy.mesh<[axis0 = size0, ...], device_ids = [...]>` (`0`).
    MeshAttr = 0,
    /// `#sdy.dim_sharding<{axis0, ?}, priority>` (`1`).
    DimensionShardingAttr = 1,
    /// `#sdy.sharding<@mesh_or_inline, [dim_shardings...], replicated_axes, unreduced_axes>` (`2`).
    TensorShardingAttr = 2,
    /// `#sdy.sharding_per_value<[#sdy.sharding<...>, ...]>` (`3`).
    TensorShardingPerValueAttr = 3,
    /// `#sdy<manual_axes{"x", "y"}>` (`4`).
    ManualAxesAttr = 4,
    /// `#sdy<mesh_axis "x" = 4>` (`5`).
    MeshAxisAttr = 5,
}

impl SdyAttrCode {
    /// Converts a raw varint tag (`0..=5`) into an [`SdyAttrCode`].
    ///
    /// # Arguments
    ///
    /// * `tag` - Raw `PrefixVarInt` tag from Section `2` (`kAttrType`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MlirBytecodeDecodeError`] if `tag > 5`.
    pub const fn from_tag(tag: u64) -> LibTpuResult<Self> {
        match tag {
            0 => Ok(Self::MeshAttr),
            1 => Ok(Self::DimensionShardingAttr),
            2 => Ok(Self::TensorShardingAttr),
            3 => Ok(Self::TensorShardingPerValueAttr),
            4 => Ok(Self::ManualAxesAttr),
            5 => Ok(Self::MeshAxisAttr),
            _ => Err(LibTpuError::MlirBytecodeDecodeError {
                section_id: MlirSectionId::AttrType.as_u8(),
                offset: 0,
                reason: "Unknown Shardy (sdy) attribute tag (> 5)",
            }),
        }
    }
}

/// Decoded Shardy (`sdy`) custom attribute record from Section `2` (`kAttrType`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SdyCustomAttr {
    /// `#sdy.mesh<[axis_str_idx = size, ...], device_ids>`.
    Mesh {
        /// `(axis_name_string_idx, axis_size)` pairs.
        axes: Vec<(u32, i64)>,
        /// Optional explicit logical device ID ordering.
        device_ids: Vec<i64>,
    },
    /// `#sdy.dim_sharding<{axes...}, is_closed, priority?>`.
    DimensionSharding {
        /// Axis string indices in Section `0` (`kString`).
        axis_string_indices: Vec<u32>,
        /// `true` if closed (`{}` without trailing `?`), `false` if open (`{..., ?}`).
        is_closed: bool,
        /// Optional propagation priority (`p0`, `p1`, ...).
        priority: Option<i64>,
    },
    /// `#sdy.sharding<@mesh, [dim_shardings...], replicated_axes, unreduced_axes>`.
    TensorSharding {
        /// Mesh symbol string index in Section `0` (`kString`).
        mesh_name_string_idx: u32,
        /// Per-dimension sharding specs `(axis_string_indices, is_closed, priority)`.
        dim_shardings: Vec<(Vec<u32>, bool, Option<i64>)>,
        /// Replicated axis string indices.
        replicated_axis_indices: Vec<u32>,
        /// Unreduced axis string indices.
        unreduced_axis_indices: Vec<u32>,
    },
    /// `#sdy.sharding_per_value<[sharding_attr_idx0, ...]>`.
    TensorShardingPerValue(Vec<u32>),
    /// `#sdy<manual_axes{axis_str_idx0, ...}>`.
    ManualAxes(Vec<u32>),
    /// `#sdy<mesh_axis name_idx = size>`.
    MeshAxis {
        /// Axis name string index.
        name_string_idx: u32,
        /// Axis size.
        size: i64,
    },
}

impl SdyCustomAttr {
    /// Serializes this [`SdyCustomAttr`] into `writer`.
    ///
    /// # Arguments
    ///
    /// * `writer` - Destination [`DialectBytecodeWriter`].
    pub fn encode(&self, writer: &mut DialectBytecodeWriter) {
        match self {
            Self::Mesh { axes, device_ids } => {
                writer.write_varint(SdyAttrCode::MeshAttr as u64);
                writer.write_list(axes, |w, &(name_idx, sz)| {
                    w.write_varint(u64::from(name_idx));
                    w.write_signed_varint(sz);
                });
                writer.write_list(device_ids, |w, &id| w.write_signed_varint(id));
            }
            Self::DimensionSharding {
                axis_string_indices,
                is_closed,
                priority,
            } => {
                writer.write_varint(SdyAttrCode::DimensionShardingAttr as u64);
                writer.write_list(axis_string_indices, |w, &idx| {
                    w.write_varint(u64::from(idx));
                });
                writer.write_bool(*is_closed);
                writer.write_bool(priority.is_some());
                if let Some(p) = priority {
                    writer.write_signed_varint(*p);
                }
            }
            Self::TensorSharding {
                mesh_name_string_idx,
                dim_shardings,
                replicated_axis_indices,
                unreduced_axis_indices,
            } => {
                writer.write_varint(SdyAttrCode::TensorShardingAttr as u64);
                writer.write_varint(u64::from(*mesh_name_string_idx));
                writer.write_list(dim_shardings, |w, (axes, closed, prio)| {
                    w.write_list(axes, |wi, &idx| wi.write_varint(u64::from(idx)));
                    w.write_bool(*closed);
                    w.write_bool(prio.is_some());
                    if let Some(p) = prio {
                        w.write_signed_varint(*p);
                    }
                });
                writer.write_list(replicated_axis_indices, |w, &idx| {
                    w.write_varint(u64::from(idx));
                });
                writer.write_list(unreduced_axis_indices, |w, &idx| {
                    w.write_varint(u64::from(idx));
                });
            }
            Self::TensorShardingPerValue(indices) => {
                writer.write_varint(SdyAttrCode::TensorShardingPerValueAttr as u64);
                writer.write_list(indices, |w, &idx| w.write_varint(u64::from(idx)));
            }
            Self::ManualAxes(axes) => {
                writer.write_varint(SdyAttrCode::ManualAxesAttr as u64);
                writer.write_list(axes, |w, &idx| w.write_varint(u64::from(idx)));
            }
            Self::MeshAxis {
                name_string_idx,
                size,
            } => {
                writer.write_varint(SdyAttrCode::MeshAxisAttr as u64);
                writer.write_varint(u64::from(*name_string_idx));
                writer.write_signed_varint(*size);
            }
        }
    }

    /// Decodes an [`SdyCustomAttr`] from `reader`.
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
        let code = SdyAttrCode::from_tag(reader.read_varint()?)?;
        match code {
            SdyAttrCode::MeshAttr => {
                let axes = reader.read_list(|r| {
                    let name_idx = r.read_varint()? as u32;
                    let sz = r.read_signed_varint()?;
                    Ok((name_idx, sz))
                })?;
                let device_ids = reader.read_list(DialectBytecodeReader::read_signed_varint)?;
                Ok(Self::Mesh { axes, device_ids })
            }
            SdyAttrCode::DimensionShardingAttr => {
                let axis_string_indices = reader.read_list(|r| Ok(r.read_varint()? as u32))?;
                let is_closed = reader.read_bool()?;
                let priority = if reader.read_bool()? {
                    Some(reader.read_signed_varint()?)
                } else {
                    None
                };
                Ok(Self::DimensionSharding {
                    axis_string_indices,
                    is_closed,
                    priority,
                })
            }
            SdyAttrCode::TensorShardingAttr => {
                let mesh_name_string_idx = reader.read_varint()? as u32;
                let dim_shardings = reader.read_list(|r| {
                    let axes = r.read_list(|ri| Ok(ri.read_varint()? as u32))?;
                    let closed = r.read_bool()?;
                    let prio = if r.read_bool()? {
                        Some(r.read_signed_varint()?)
                    } else {
                        None
                    };
                    Ok((axes, closed, prio))
                })?;
                let replicated_axis_indices = reader.read_list(|r| Ok(r.read_varint()? as u32))?;
                let unreduced_axis_indices = reader.read_list(|r| Ok(r.read_varint()? as u32))?;
                Ok(Self::TensorSharding {
                    mesh_name_string_idx,
                    dim_shardings,
                    replicated_axis_indices,
                    unreduced_axis_indices,
                })
            }
            SdyAttrCode::TensorShardingPerValueAttr => {
                let indices = reader.read_list(|r| Ok(r.read_varint()? as u32))?;
                Ok(Self::TensorShardingPerValue(indices))
            }
            SdyAttrCode::ManualAxesAttr => {
                let axes = reader.read_list(|r| Ok(r.read_varint()? as u32))?;
                Ok(Self::ManualAxes(axes))
            }
            SdyAttrCode::MeshAxisAttr => Ok(Self::MeshAxis {
                name_string_idx: reader.read_varint()? as u32,
                size: reader.read_signed_varint()?,
            }),
        }
    }

    /// Converts this [`SdyCustomAttr`] into a canonical [`MlirAttribute`].
    ///
    /// # Arguments
    ///
    /// * `strings` - Section `0` (`kString`) table.
    /// * `resolved_attrs` - Previously-resolved [`MlirAttribute`] table.
    #[must_use]
    pub fn to_mlir_attribute(
        &self,
        strings: &[String],
        resolved_attrs: &[MlirAttribute],
    ) -> MlirAttribute {
        let lookup_str = |idx: u32| -> String {
            strings
                .get(idx as usize)
                .cloned()
                .unwrap_or_else(|| format!("axis_{idx}"))
        };
        match self {
            Self::Mesh { axes, device_ids } => MlirAttribute::SdyMesh(SdyMeshAttr {
                name: String::from("mesh"),
                axes: axes
                    .iter()
                    .map(|&(idx, sz)| (lookup_str(idx), usize::try_from(sz.max(1)).unwrap_or(1)))
                    .collect(),
                device_ids: device_ids
                    .iter()
                    .filter_map(|&d| usize::try_from(d).ok())
                    .collect(),
            }),
            Self::DimensionSharding {
                axis_string_indices,
                is_closed,
                priority,
            } => MlirAttribute::SdyTensorSharding(SdyTensorShardingAttr {
                mesh_name: String::from("mesh"),
                dim_shardings: vec![SdyDimensionSharding {
                    axes: axis_string_indices.iter().map(|&i| lookup_str(i)).collect(),
                    is_open: !*is_closed,
                    priority: *priority,
                }],
                replicated_axes: Vec::new(),
                unreduced_axes: Vec::new(),
            }),
            Self::TensorSharding {
                mesh_name_string_idx,
                dim_shardings,
                replicated_axis_indices,
                unreduced_axis_indices,
            } => MlirAttribute::SdyTensorSharding(SdyTensorShardingAttr {
                mesh_name: lookup_str(*mesh_name_string_idx),
                dim_shardings: dim_shardings
                    .iter()
                    .map(|(axes, closed, prio)| SdyDimensionSharding {
                        axes: axes.iter().map(|&i| lookup_str(i)).collect(),
                        is_open: !*closed,
                        priority: *prio,
                    })
                    .collect(),
                replicated_axes: replicated_axis_indices
                    .iter()
                    .map(|&i| lookup_str(i))
                    .collect(),
                unreduced_axes: unreduced_axis_indices
                    .iter()
                    .map(|&i| lookup_str(i))
                    .collect(),
            }),
            Self::TensorShardingPerValue(indices) => {
                let mut shardings = Vec::new();
                for &idx in indices {
                    if let Some(MlirAttribute::SdyTensorSharding(sh)) =
                        resolved_attrs.get(idx as usize)
                    {
                        shardings.push(sh.clone());
                    }
                }
                MlirAttribute::SdyTensorShardingPerValue(shardings)
            }
            Self::ManualAxes(axes) => MlirAttribute::Array(
                axes.iter()
                    .map(|&idx| MlirAttribute::String(lookup_str(idx)))
                    .collect(),
            ),
            Self::MeshAxis {
                name_string_idx,
                size,
            } => MlirAttribute::SdyMesh(SdyMeshAttr {
                name: String::from("mesh"),
                axes: vec![(
                    lookup_str(*name_string_idx),
                    usize::try_from((*size).max(1)).unwrap_or(1),
                )],
                device_ids: Vec::new(),
            }),
        }
    }
}

/// JAX Mosaic (`tpu`) custom attribute & type bytecode tags (`0..=5`).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TpuMosaicBytecodeCode {
    /// `#tpu.memory_space<hbm|vmem|smem|cmem|semaphore_mem>` (`0`).
    MemorySpaceAttr = 0,
    /// `#tpu.tiled<(8, 128), [1, 1]>` (`1`).
    TiledLayoutAttr = 1,
    /// `#tpu.dimension_semantics<parallel|arbitrary>` (`2`).
    DimensionSemanticsAttr = 2,
    /// `#tpu.core_type<tc|sc_scalar_subcore|sc_vector_subcore>` (`3`).
    CoreTypeAttr = 3,
    /// `!tpu.semaphore` (`4`).
    SemaphoreType = 4,
    /// `!tpu.dma_semaphore` (`5`).
    DmaSemaphoreType = 5,
}

impl TpuMosaicBytecodeCode {
    /// Converts a raw varint tag (`0..=5`) into a [`TpuMosaicBytecodeCode`].
    ///
    /// # Arguments
    ///
    /// * `tag` - Raw `PrefixVarInt` tag from Section `2` (`kAttrType`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MlirBytecodeDecodeError`] if `tag > 5`.
    pub const fn from_tag(tag: u64) -> LibTpuResult<Self> {
        match tag {
            0 => Ok(Self::MemorySpaceAttr),
            1 => Ok(Self::TiledLayoutAttr),
            2 => Ok(Self::DimensionSemanticsAttr),
            3 => Ok(Self::CoreTypeAttr),
            4 => Ok(Self::SemaphoreType),
            5 => Ok(Self::DmaSemaphoreType),
            _ => Err(LibTpuError::MlirBytecodeDecodeError {
                section_id: MlirSectionId::AttrType.as_u8(),
                offset: 0,
                reason: "Unknown Mosaic (tpu) bytecode tag (> 5)",
            }),
        }
    }
}

/// Decoded JAX Mosaic (`tpu`) custom attribute or type entry from Section `2` (`kAttrType`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TpuMosaicCustomEntry {
    /// `#tpu.memory_space<...>`.
    MemorySpace(TpuMemorySpace),
    /// `#tpu.tiled<tiles, tile_strides>`.
    TiledLayout {
        /// Nested tile extents (e.g., `vec![vec![8, 128]]`).
        tiles: Vec<Vec<i64>>,
        /// Tile strides.
        tile_strides: Vec<i64>,
    },
    /// `#tpu.dimension_semantics<[parallel|arbitrary, ...]>`.
    DimensionSemantics(Vec<String>),
    /// `#tpu.core_type<...>`.
    CoreType(String),
    /// `!tpu.semaphore` or `!tpu.dma_semaphore`.
    Semaphore(TpuSemaphoreKind),
}

impl TpuMosaicCustomEntry {
    /// Serializes this [`TpuMosaicCustomEntry`] into `writer`.
    ///
    /// # Arguments
    ///
    /// * `writer` - Destination [`DialectBytecodeWriter`].
    pub fn encode(&self, writer: &mut DialectBytecodeWriter) {
        match self {
            Self::MemorySpace(space) => {
                writer.write_varint(TpuMosaicBytecodeCode::MemorySpaceAttr as u64);
                let code = match space {
                    TpuMemorySpace::Hbm => 0,
                    TpuMemorySpace::Vmem => 1,
                    TpuMemorySpace::Smem => 2,
                    TpuMemorySpace::Cmem => 3,
                    TpuMemorySpace::SemaphoreMem => 4,
                };
                writer.write_varint(code);
            }
            Self::TiledLayout {
                tiles,
                tile_strides,
            } => {
                writer.write_varint(TpuMosaicBytecodeCode::TiledLayoutAttr as u64);
                writer.write_list(tiles, |w, tile| {
                    w.write_list(tile, |wi, &d| wi.write_signed_varint(d));
                });
                writer.write_list(tile_strides, |w, &s| w.write_signed_varint(s));
            }
            Self::DimensionSemantics(sems) => {
                writer.write_varint(TpuMosaicBytecodeCode::DimensionSemanticsAttr as u64);
                writer.write_list(sems, |w, s| w.write_nul_string(s));
            }
            Self::CoreType(core) => {
                writer.write_varint(TpuMosaicBytecodeCode::CoreTypeAttr as u64);
                writer.write_nul_string(core);
            }
            Self::Semaphore(TpuSemaphoreKind::Regular) => {
                writer.write_varint(TpuMosaicBytecodeCode::SemaphoreType as u64);
            }
            Self::Semaphore(TpuSemaphoreKind::Dma) => {
                writer.write_varint(TpuMosaicBytecodeCode::DmaSemaphoreType as u64);
            }
        }
    }

    /// Decodes a [`TpuMosaicCustomEntry`] from `reader`.
    ///
    /// # Arguments
    ///
    /// * `reader` - Source [`DialectBytecodeReader`].
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MlirBytecodeDecodeError`] if the tag is invalid or payload is truncated.
    pub fn decode(reader: &mut DialectBytecodeReader<'_>) -> LibTpuResult<Self> {
        let code = TpuMosaicBytecodeCode::from_tag(reader.read_varint()?)?;
        match code {
            TpuMosaicBytecodeCode::MemorySpaceAttr => {
                let space = match reader.read_varint()? {
                    1 => TpuMemorySpace::Vmem,
                    2 => TpuMemorySpace::Smem,
                    3 => TpuMemorySpace::Cmem,
                    4 => TpuMemorySpace::SemaphoreMem,
                    _ => TpuMemorySpace::Hbm,
                };
                Ok(Self::MemorySpace(space))
            }
            TpuMosaicBytecodeCode::TiledLayoutAttr => {
                let tiles =
                    reader.read_list(|r| r.read_list(DialectBytecodeReader::read_signed_varint))?;
                let tile_strides = reader.read_list(DialectBytecodeReader::read_signed_varint)?;
                Ok(Self::TiledLayout {
                    tiles,
                    tile_strides,
                })
            }
            TpuMosaicBytecodeCode::DimensionSemanticsAttr => {
                let sems = reader.read_list(|r| Ok(String::from(r.read_nul_string()?)))?;
                Ok(Self::DimensionSemantics(sems))
            }
            TpuMosaicBytecodeCode::CoreTypeAttr => {
                Ok(Self::CoreType(String::from(reader.read_nul_string()?)))
            }
            TpuMosaicBytecodeCode::SemaphoreType => Ok(Self::Semaphore(TpuSemaphoreKind::Regular)),
            TpuMosaicBytecodeCode::DmaSemaphoreType => Ok(Self::Semaphore(TpuSemaphoreKind::Dma)),
        }
    }

    /// Converts this [`TpuMosaicCustomEntry`] into an [`MlirAttribute`] if it represents an attribute.
    #[must_use]
    pub fn to_mlir_attribute(&self) -> Option<MlirAttribute> {
        match self {
            Self::MemorySpace(space) => Some(MlirAttribute::MosaicAttr(
                MosaicDialectAttr::MemorySpace(*space),
            )),
            Self::TiledLayout {
                tiles,
                tile_strides,
            } => Some(MlirAttribute::MosaicAttr(MosaicDialectAttr::TiledLayout {
                tiles: tiles.clone(),
                tile_strides: tile_strides.clone(),
            })),
            Self::DimensionSemantics(sems) => Some(MlirAttribute::MosaicAttr(
                MosaicDialectAttr::DimensionSemantics(sems.join(", ")),
            )),
            Self::CoreType(ct) => Some(MlirAttribute::MosaicAttr(MosaicDialectAttr::CoreType(
                ct.clone(),
            ))),
            Self::Semaphore(_) => None,
        }
    }

    /// Converts this [`TpuMosaicCustomEntry`] into an [`MlirType`] if it represents a semaphore type.
    #[must_use]
    pub const fn to_mlir_type(&self) -> Option<MlirType> {
        match self {
            Self::Semaphore(kind) => Some(MlirType::Semaphore(*kind)),
            _ => None,
        }
    }
}

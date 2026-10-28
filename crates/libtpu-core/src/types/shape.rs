//! Validated tensor shapes, TPU tiled memory layouts, and typestate-tracked device buffers.
//!
//! Defines [`MemoryLayout`], [`Shape`] (`TensorShape`) with 2D `(8, 128)` and 3D sub-byte
//! hardware tile permutation (`linearize_to_tpu_tiles` / `delinearize_from_tpu_tiles`), and
//! [`TypedDeviceBuffer`] with compile-time [`Unallocated`] / [`Allocated`] typestate tracking.

use crate::error::{LibTpuError, LibTpuResult};
use crate::types::address::{ByteSize, DeviceAddress};
use crate::types::element::ElementType;
use crate::types::ids::{BufferId, DeviceId};
use smallvec::SmallVec;
use std::marker::PhantomData;

/// Stack-inlined dimension vector capable of storing up to 6 tensor dimensions without heap allocation.
pub type DimVec = SmallVec<[i64; 6]>;

/// Stack-inlined dynamic dimension flag vector capable of storing up to 6 flags without heap allocation.
pub type DynamicDimVec = SmallVec<[bool; 6]>;

/// Memory layout specification for a TPU buffer (`PJRT_Buffer_MemoryLayout`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryLayout {
    /// Tiled layout with minor-to-major ordering and TPU tile dimensions (e.g., `(8, 128)`).
    Tiled {
        /// Dimension permutation from most-minor (fastest varying) to most-major.
        minor_to_major: Vec<i64>,
        /// Flattened tile dimensions.
        tile_dims: Vec<i64>,
        /// Number of dimensions in each tile level.
        tile_dim_sizes: Vec<usize>,
    },
    /// Explicit byte strides per dimension.
    Strides {
        /// Stride in bytes along each dimension.
        byte_strides: Vec<i64>,
    },
}

impl MemoryLayout {
    /// Creates the default TPU tiled layout (`(8, 128)` for rank >= 2) for a given rank.
    ///
    /// # Arguments
    ///
    /// * `rank` - Number of dimensions in the tensor.
    #[must_use]
    pub fn default_tpu_tiled(rank: usize) -> Self {
        Self::default_tpu_tiled_for_type(rank, ElementType::F32)
    }

    /// Creates the canonical TPU physical tiled layout for a given `rank` and [`ElementType`],
    /// including second-level sub-packing tiles (`(2, 1)` for 16-bit, `(4, 1)` for 8-bit, `(8, 1)` for 4-bit).
    ///
    /// # Arguments
    ///
    /// * `rank` - Number of dimensions in the tensor.
    /// * `element_type` - Tensor element data type.
    #[must_use]
    pub fn default_tpu_tiled_for_type(rank: usize, element_type: ElementType) -> Self {
        let minor_to_major: Vec<i64> = (0..rank)
            .rev()
            .map(|idx| i64::try_from(idx).unwrap_or(0))
            .collect();
        if rank >= 2 {
            let subtile = match element_type.bit_size() {
                16 => Some([2_i64, 1]),
                8 => Some([4_i64, 1]),
                4 => Some([8_i64, 1]),
                _ => None,
            };
            if let Some([s0, s1]) = subtile {
                Self::Tiled {
                    minor_to_major,
                    tile_dims: vec![8, 128, s0, s1],
                    tile_dim_sizes: vec![2, 2],
                }
            } else {
                Self::Tiled {
                    minor_to_major,
                    tile_dims: vec![8, 128],
                    tile_dim_sizes: vec![2],
                }
            }
        } else if rank == 1 {
            Self::Tiled {
                minor_to_major,
                tile_dims: vec![128],
                tile_dim_sizes: vec![1],
            }
        } else {
            Self::Tiled {
                minor_to_major,
                tile_dims: Vec::new(),
                tile_dim_sizes: Vec::new(),
            }
        }
    }
}

/// Validated tensor shape, element type, dynamic dimension flags, and layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    /// Element data type.
    pub element_type: ElementType,
    /// Logical dimension sizes.
    pub dims: Vec<i64>,
    /// Dynamic dimension flags (`true` if dimension is dynamically bounded).
    pub dynamic_dims: Vec<bool>,
    /// Memory layout on device.
    pub layout: MemoryLayout,
}

impl Shape {
    /// Constructs and validates a new [`Shape`].
    ///
    /// # Arguments
    ///
    /// * `element_type` - Tensor element type.
    /// * `dims` - Logical dimension extents (must all be `>= 0`).
    /// * `dynamic_dims` - Optional dynamic dimension indicators.
    /// * `layout` - Optional memory layout (defaults to [`MemoryLayout::default_tpu_tiled`]).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidShape`] if any dimension is negative or if `dynamic_dims`
    /// has a non-empty length that differs from `dims.len()`.
    pub fn new(
        element_type: ElementType,
        dims: Vec<i64>,
        dynamic_dims: Option<Vec<bool>>,
        layout: Option<MemoryLayout>,
    ) -> LibTpuResult<Self> {
        if element_type == ElementType::Invalid {
            return Err(LibTpuError::InvalidShape {
                reason: String::from("ElementType::Invalid is not permitted in a valid Shape"),
            });
        }
        for &d in &dims {
            if d < 0 {
                return Err(LibTpuError::InvalidShape {
                    reason: format!("Negative dimension size {d} is not allowed"),
                });
            }
        }
        let resolved_dynamic = match dynamic_dims {
            Some(flags) => {
                if flags.len() != dims.len() {
                    return Err(LibTpuError::InvalidShape {
                        reason: format!(
                            "dynamic_dims length ({}) does not match dims length ({})",
                            flags.len(),
                            dims.len()
                        ),
                    });
                }
                flags
            }
            None => vec![false; dims.len()],
        };
        let resolved_layout = layout.unwrap_or_else(|| MemoryLayout::default_tpu_tiled(dims.len()));
        let bits = element_type.bit_size();
        if bits > 0 {
            let mut num_elements: usize = 1;
            for &d in &dims {
                let extent = usize::try_from(d).unwrap_or(usize::MAX);
                num_elements =
                    num_elements
                        .checked_mul(extent)
                        .ok_or_else(|| LibTpuError::InvalidShape {
                            reason: String::from("Tensor element count overflow"),
                        })?;
            }
            let _total_bits =
                num_elements
                    .checked_mul(bits)
                    .ok_or_else(|| LibTpuError::InvalidShape {
                        reason: String::from("Tensor bit size overflow"),
                    })?;
        }
        Ok(Self {
            element_type,
            dims,
            dynamic_dims: resolved_dynamic,
            layout: resolved_layout,
        })
    }

    /// Computes the unpadded host byte size of a tensor with this validated [`Shape`].
    #[must_use]
    pub fn unpadded_byte_size(&self) -> ByteSize {
        let bits = self.element_type.bit_size();
        if bits == 0 {
            return ByteSize(0);
        }
        let mut num_elements: usize = 1;
        for &d in &self.dims {
            let extent = usize::try_from(d).unwrap_or(0);
            num_elements = num_elements.saturating_mul(extent);
        }
        let total_bits = num_elements.saturating_mul(bits);
        ByteSize(total_bits.div_ceil(8))
    }

    /// Computes the on-device HBM byte size aligned to 256-byte TPU DMA granule boundaries.
    #[must_use]
    pub fn on_device_byte_size(&self) -> ByteSize {
        let unpadded = self.unpadded_byte_size().0;
        if unpadded == 0 {
            return ByteSize(256);
        }
        let aligned = unpadded.div_ceil(256) * 256;
        ByteSize(aligned)
    }

    /// Returns the indices of all dynamic dimensions in this shape.
    #[must_use]
    pub fn dynamic_dimension_indices(&self) -> Vec<usize> {
        self.dynamic_dims
            .iter()
            .enumerate()
            .filter_map(|(idx, &is_dyn)| if is_dyn { Some(idx) } else { None })
            .collect()
    }

    /// Returns the number of dimensions (rank) of this shape.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Returns a stack-inlined [`DimVec`] (`SmallVec<[i64; 6]>`) copy of the logical dimensions.
    #[must_use]
    pub fn inline_dims(&self) -> DimVec {
        DimVec::from_slice(&self.dims)
    }

    /// Returns a stack-inlined [`DynamicDimVec`] (`SmallVec<[bool; 6]>`) copy of the dynamic dimension flags.
    #[must_use]
    pub fn inline_dynamic_dims(&self) -> DynamicDimVec {
        DynamicDimVec::from_slice(&self.dynamic_dims)
    }

    /// Computes the total number of scalar elements in this shape.
    #[must_use]
    pub fn element_count(&self) -> usize {
        let mut count: usize = 1;
        for &d in &self.dims {
            let extent = usize::try_from(d).unwrap_or(0);
            count = count.saturating_mul(extent);
        }
        count
    }

    /// Converts a flat row-major linear element index into N-dimensional coordinates.
    ///
    /// # Arguments
    ///
    /// * `linear_idx` - Flat row-major element index.
    #[must_use]
    pub fn linear_to_coords(&self, linear_idx: usize) -> Vec<usize> {
        let mut coords = Vec::with_capacity(self.dims.len());
        let mut rem = linear_idx;
        for &d in self.dims.iter().rev() {
            let extent = usize::try_from(d).unwrap_or(0).max(1);
            coords.push(rem % extent);
            rem /= extent;
        }
        coords.reverse();
        coords
    }

    /// Converts N-dimensional coordinates into a flat row-major linear element index.
    ///
    /// # Arguments
    ///
    /// * `coords` - Coordinate slice of length equal to `self.rank()`.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidShape`] if `coords.len() != self.rank()` or any coordinate is out of bounds.
    pub fn coords_to_linear(&self, coords: &[usize]) -> LibTpuResult<usize> {
        if coords.len() != self.dims.len() {
            return Err(LibTpuError::InvalidShape {
                reason: format!(
                    "Coordinate rank {} does not match shape rank {}",
                    coords.len(),
                    self.dims.len()
                ),
            });
        }
        let mut idx: usize = 0;
        for (&c, &d) in coords.iter().zip(self.dims.iter()) {
            let extent = usize::try_from(d).unwrap_or(0);
            if c >= extent {
                return Err(LibTpuError::InvalidShape {
                    reason: format!("Coordinate {c} out of bounds for dimension extent {extent}"),
                });
            }
            idx = idx.saturating_mul(extent).saturating_add(c);
        }
        Ok(idx)
    }

    /// Returns a new [`Shape`] with dimensions permuted by `permutation`.
    ///
    /// # Arguments
    ///
    /// * `permutation` - Bijection of `0..self.rank()`.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidShape`] if `permutation` is not a valid bijection of `0..self.rank()`.
    pub fn transpose(&self, permutation: &[usize]) -> LibTpuResult<Self> {
        if permutation.len() != self.dims.len() {
            return Err(LibTpuError::InvalidShape {
                reason: format!(
                    "Permutation length {} does not match rank {}",
                    permutation.len(),
                    self.dims.len()
                ),
            });
        }
        let mut seen = vec![false; self.dims.len()];
        let mut new_dims = Vec::with_capacity(self.dims.len());
        let mut new_dyn = Vec::with_capacity(self.dims.len());
        for &axis in permutation {
            let Some(seen_flag) = seen.get_mut(axis) else {
                return Err(LibTpuError::InvalidShape {
                    reason: format!("Permutation axis {axis} out of range"),
                });
            };
            if *seen_flag {
                return Err(LibTpuError::InvalidShape {
                    reason: format!("Duplicate axis {axis} in permutation"),
                });
            }
            *seen_flag = true;
            new_dims.push(self.dims.get(axis).copied().unwrap_or(0));
            new_dyn.push(self.dynamic_dims.get(axis).copied().unwrap_or(false));
        }
        Self::new(self.element_type, new_dims, Some(new_dyn), None)
    }

    /// Computes the hardware tile permutation parameters `(pack_factor, sublanes, lanes)` for this shape:
    /// - 32/64/128-bit elements: 2D `(1, 8, 128)` tile (`8x128`)
    /// - 16-bit elements (`F16`, `Bf16`, `S16`, `U16`): 3D `(2, 4, 128)` sub-tile packing
    /// - 8-bit elements (`S8`, `U8`, `Pred`, `F8*`): 3D `(4, 4, 128)` sub-tile packing
    /// - Sub-byte elements (`S4`, `U4`, `F4E2M1Fn`, `S2`, `U2`): 3D `(8, 4, 128)` sub-tile packing
    #[must_use]
    pub const fn tpu_tile_geometry(&self) -> (usize, usize, usize) {
        let bits = self.element_type.bit_size();
        if bits == 16 {
            (2, 4, 128)
        } else if bits == 8 {
            (4, 4, 128)
        } else if bits < 8 {
            (8, 4, 128)
        } else {
            (1, 8, 128)
        }
    }

    /// Computes the bijective element index mapping from TPU tiled order (`(tr, tc, sr, sc, pack)`)
    /// to row-major linear element indices `0..self.element_count()`.
    #[must_use]
    pub fn tile_permutation_indices(&self) -> Vec<usize> {
        let total = self.element_count();
        if self.dims.len() < 2 {
            return (0..total).collect();
        }
        let rank = self.dims.len();
        let rows = usize::try_from(self.dims.get(rank - 2).copied().unwrap_or(1))
            .unwrap_or(1)
            .max(1);
        let cols = usize::try_from(self.dims.get(rank - 1).copied().unwrap_or(1))
            .unwrap_or(1)
            .max(1);
        let plane = (rows * cols).max(1);
        let batch = (total / plane).max(1);
        let (pack, sub_r, tile_c) = self.tpu_tile_geometry();
        let tile_r = pack * sub_r;

        let mut perm = Vec::with_capacity(total);
        for b in 0..batch {
            for tr in 0..rows.div_ceil(tile_r) {
                for tc in 0..cols.div_ceil(tile_c) {
                    for sr in 0..sub_r {
                        for sc in 0..tile_c {
                            for p in 0..pack {
                                let r = tr * tile_r + sr * pack + p;
                                let c = tc * tile_c + sc;
                                if r < rows && c < cols {
                                    perm.push((b * rows + r) * cols + c);
                                }
                            }
                        }
                    }
                }
            }
        }
        perm
    }

    /// Linearizes host row-major bytes into 256-byte-aligned TPU tiled HBM bytes (`TpuDriver_LinearizeShape`),
    /// applying 2D `(8, 128)` and 3D `(2, 4, 128)` / `(4, 4, 128)` / `(8, 4, 128)` hardware tile permutation.
    ///
    /// # Arguments
    ///
    /// * `host_bytes` - Source row-major host byte slice.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if `host_bytes.len() > self.on_device_byte_size().0`.
    pub fn linearize_to_tpu_tiles(&self, host_bytes: &[u8]) -> LibTpuResult<Vec<u8>> {
        let on_dev = self.on_device_byte_size().0;
        if host_bytes.len() > on_dev {
            return Err(LibTpuError::InvalidArgument {
                parameter: "host_bytes",
                reason: format!(
                    "Host payload ({} bytes) exceeds on-device size ({on_dev} bytes)",
                    host_bytes.len()
                ),
            });
        }
        let unpadded = self.unpadded_byte_size().0;
        let mut src_padded = Vec::with_capacity(unpadded.max(host_bytes.len()));
        src_padded.extend_from_slice(host_bytes);
        if src_padded.len() < unpadded {
            src_padded.resize(unpadded, 0);
        }
        let bits = self.element_type.bit_size();
        let mut out = vec![0_u8; on_dev];
        if self.dims.len() >= 2 && bits >= 8 {
            let elem_bytes = bits / 8;
            let perm = self.tile_permutation_indices();
            for (tiled_idx, &src_idx) in perm.iter().enumerate() {
                let src_off = src_idx * elem_bytes;
                let dst_off = tiled_idx * elem_bytes;
                for (dst_b, &src_b) in out
                    .iter_mut()
                    .skip(dst_off)
                    .take(elem_bytes)
                    .zip(src_padded.iter().skip(src_off).take(elem_bytes))
                {
                    *dst_b = src_b;
                }
            }
        } else {
            for (dst_b, &src_b) in out.iter_mut().zip(src_padded.iter()) {
                *dst_b = src_b;
            }
        }
        Ok(out)
    }

    /// Delinearizes TPU tiled HBM bytes back into unpadded host row-major bytes (`TpuDriver_DelinearizeShape`),
    /// inverting the 2D `(8, 128)` and 3D `(2, 4, 128)` / `(4, 4, 128)` / `(8, 4, 128)` hardware tile permutation.
    ///
    /// # Arguments
    ///
    /// * `tiled_bytes` - Source TPU tiled HBM byte slice.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if `tiled_bytes.len() < self.unpadded_byte_size().0`.
    pub fn delinearize_from_tpu_tiles(&self, tiled_bytes: &[u8]) -> LibTpuResult<Vec<u8>> {
        let unpadded = self.unpadded_byte_size().0;
        if tiled_bytes.len() < unpadded {
            return Err(LibTpuError::InvalidArgument {
                parameter: "tiled_bytes",
                reason: format!(
                    "Tiled buffer ({} bytes) smaller than unpadded size ({unpadded} bytes)",
                    tiled_bytes.len()
                ),
            });
        }
        let bits = self.element_type.bit_size();
        let mut out = vec![0_u8; unpadded];
        if self.dims.len() >= 2 && bits >= 8 {
            let elem_bytes = bits / 8;
            let perm = self.tile_permutation_indices();
            for (tiled_idx, &dst_idx) in perm.iter().enumerate() {
                let src_off = tiled_idx * elem_bytes;
                let dst_off = dst_idx * elem_bytes;
                for (dst_b, &src_b) in out
                    .iter_mut()
                    .skip(dst_off)
                    .take(elem_bytes)
                    .zip(tiled_bytes.iter().skip(src_off).take(elem_bytes))
                {
                    *dst_b = src_b;
                }
            }
        } else {
            for (dst_b, &src_b) in out.iter_mut().zip(tiled_bytes.iter()) {
                *dst_b = src_b;
            }
        }
        Ok(out)
    }

    /// Alias for [`Shape::linearize_to_tpu_tiles`].
    ///
    /// # Arguments
    ///
    /// * `host_bytes` - Source row-major host byte slice.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if `host_bytes.len() > self.on_device_byte_size().0`.
    pub fn linearize_bytes(&self, host_bytes: &[u8]) -> LibTpuResult<Vec<u8>> {
        self.linearize_to_tpu_tiles(host_bytes)
    }

    /// Alias for [`Shape::delinearize_from_tpu_tiles`].
    ///
    /// # Arguments
    ///
    /// * `tiled_bytes` - Source TPU tiled HBM byte slice.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if `tiled_bytes.len() < self.unpadded_byte_size().0`.
    pub fn delinearize_bytes(&self, tiled_bytes: &[u8]) -> LibTpuResult<Vec<u8>> {
        self.delinearize_from_tpu_tiles(tiled_bytes)
    }

    /// Converts linear host row-major bytes into hardware TPU tiled layout (`linear_to_tpu_tiled`).
    ///
    /// # Arguments
    ///
    /// * `linear_bytes` - Source row-major linear byte slice.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if `linear_bytes.len() > self.on_device_byte_size().0`.
    pub fn linear_to_tpu_tiled(&self, linear_bytes: &[u8]) -> LibTpuResult<Vec<u8>> {
        self.linearize_to_tpu_tiles(linear_bytes)
    }

    /// Converts hardware TPU tiled bytes back into linear host row-major layout (`tpu_tiled_to_linear`).
    ///
    /// # Arguments
    ///
    /// * `tiled_bytes` - Source TPU tiled byte slice.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if `tiled_bytes.len() < self.unpadded_byte_size().0`.
    pub fn tpu_tiled_to_linear(&self, tiled_bytes: &[u8]) -> LibTpuResult<Vec<u8>> {
        self.delinearize_from_tpu_tiles(tiled_bytes)
    }
}

/// Type alias for [`Shape`] (`TensorShape`).
pub type TensorShape = Shape;

/// Typestate marker for an unallocated device buffer.
#[derive(Debug, Clone, Copy)]
pub struct Unallocated;

/// Typestate marker for an allocated device buffer backed by HBM.
#[derive(Debug, Clone, Copy)]
pub struct Allocated;

/// Typestate-tracked device buffer handle transitioning from [`Unallocated`] to [`Allocated`].
#[derive(Debug, Clone)]
pub struct TypedDeviceBuffer<State> {
    /// Unique identifier for the buffer.
    pub id: BufferId,
    /// Target device ID.
    pub device_id: DeviceId,
    /// Validated tensor shape.
    pub shape: Shape,
    /// Optional allocated HBM address.
    pub address: Option<DeviceAddress>,
    /// Phantom typestate marker.
    _state: PhantomData<State>,
}

impl TypedDeviceBuffer<Unallocated> {
    /// Creates a new [`TypedDeviceBuffer<Unallocated>`] before HBM allocation.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique buffer identifier.
    /// * `device_id` - Target TPU device identifier.
    /// * `shape` - Validated tensor shape.
    #[must_use]
    pub const fn new_unallocated(id: BufferId, device_id: DeviceId, shape: Shape) -> Self {
        Self {
            id,
            device_id,
            shape,
            address: None,
            _state: PhantomData,
        }
    }

    /// Transitions this buffer into the [`Allocated`] typestate by binding an HBM [`DeviceAddress`].
    ///
    /// # Arguments
    ///
    /// * `address` - Allocated HBM device address.
    #[must_use]
    pub fn bind_allocation(self, address: DeviceAddress) -> TypedDeviceBuffer<Allocated> {
        TypedDeviceBuffer {
            id: self.id,
            device_id: self.device_id,
            shape: self.shape,
            address: Some(address),
            _state: PhantomData,
        }
    }
}

impl TypedDeviceBuffer<Allocated> {
    /// Returns the guaranteed [`DeviceAddress`] for this allocated buffer.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn device_address(&self) -> DeviceAddress {
        self.address.unwrap_or(DeviceAddress(0))
    }
}

#[cfg(test)]
#[allow(
    for_loops_over_fallibles,
    clippy::cognitive_complexity,
    clippy::too_many_lines
)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_shape_and_typestate_buffer() {
        assert!(Shape::new(ElementType::Invalid, vec![1], None, None).is_err());
        assert!(Shape::new(ElementType::F32, vec![-1], None, None).is_err());
        assert!(Shape::new(ElementType::F32, vec![4, 4], Some(vec![true]), None).is_err());

        let scalar = Shape::new(ElementType::Token, vec![], None, None);
        assert!(scalar.is_ok());
        for s in scalar {
            assert_eq!(s.unpadded_byte_size(), ByteSize(0));
            assert_eq!(s.on_device_byte_size(), ByteSize(256));
        }

        let r0 = MemoryLayout::default_tpu_tiled(0);
        let r1 = MemoryLayout::default_tpu_tiled(1);
        let r2 = MemoryLayout::default_tpu_tiled(2);
        let r2_bf16 = MemoryLayout::default_tpu_tiled_for_type(2, ElementType::Bf16);
        let r2_s8 = MemoryLayout::default_tpu_tiled_for_type(2, ElementType::S8);
        let r2_s4 = MemoryLayout::default_tpu_tiled_for_type(2, ElementType::S4);
        assert_ne!(r0, r1);
        assert_ne!(r1, r2);
        assert_ne!(r2, r2_bf16);
        assert_ne!(r2_bf16, r2_s8);
        assert_ne!(r2_s8, r2_s4);

        let rank1 = Shape::new(
            ElementType::F32,
            vec![10],
            Some(vec![true]),
            Some(MemoryLayout::Strides {
                byte_strides: vec![4],
            }),
        );
        assert!(rank1.is_ok());
        for s in rank1 {
            assert_eq!(s.unpadded_byte_size(), ByteSize(40));
            assert_eq!(s.on_device_byte_size(), ByteSize(256));
            assert_eq!(s.dynamic_dimension_indices(), vec![0]);
            assert_ne!(format!("{s:?}"), "");
        }

        let rank1_default = Shape::new(ElementType::S4, vec![3], None, None);
        assert!(rank1_default.is_ok());
        for s in rank1_default {
            assert_eq!(s.unpadded_byte_size(), ByteSize(2));
        }

        let rank2 = Shape::new(
            ElementType::Bf16,
            vec![8, 128],
            Some(vec![false, true]),
            None,
        );
        assert!(rank2.is_ok());
        for s in rank2 {
            assert_eq!(s.unpadded_byte_size(), ByteSize(2048));
            assert_eq!(s.on_device_byte_size(), ByteSize(2048));
            assert_eq!(s.dynamic_dimension_indices(), vec![1]);

            let unalloc = TypedDeviceBuffer::new_unallocated(BufferId(1), DeviceId(0), s);
            let unalloc_cloned = unalloc.clone();
            assert_ne!(format!("{unalloc_cloned:?}"), "");
            let alloc = unalloc.bind_allocation(DeviceAddress(0x4000));
            let alloc_cloned = alloc.clone();
            assert_eq!(alloc.device_address(), DeviceAddress(0x4000));
            drop(alloc);
            assert_eq!(alloc_cloned.device_address(), DeviceAddress(0x4000));
            assert_ne!(format!("{alloc_cloned:?}"), "");
            assert_ne!(format!("{Unallocated:?}"), "");
            assert_ne!(format!("{Allocated:?}"), "");
        }

        assert!(Shape::new(ElementType::F32, vec![i64::MAX, i64::MAX], None, None).is_err());
        assert!(Shape::new(ElementType::C128, vec![i64::MAX], None, None).is_err());

        for s in Shape::new(ElementType::F32, vec![2, 3], None, None) {
            assert_eq!(s.rank(), 2);
            assert_eq!(s.inline_dims().as_slice(), &[2, 3]);
            assert_eq!(s.inline_dynamic_dims().as_slice(), &[false, false]);
            assert_eq!(s.element_count(), 6);
            assert_eq!(s.linear_to_coords(4), vec![1, 1]);
            assert_eq!(s.coords_to_linear(&[1, 1]).ok(), Some(4));
            assert!(s.coords_to_linear(&[1]).is_err());
            assert!(s.coords_to_linear(&[2, 0]).is_err());

            let tr = s.transpose(&[1, 0]);
            assert!(tr.is_ok());
            for t in tr {
                assert_eq!(t.dims, vec![3, 2]);
            }
            assert!(s.transpose(&[0]).is_err());
            assert!(s.transpose(&[2, 0]).is_err());
            assert!(s.transpose(&[0, 0]).is_err());

            let host_bytes: Vec<u8> = (0_u8..24).collect();
            let tiled = s.linearize_bytes(&host_bytes);
            assert!(tiled.is_ok());
            for tb in tiled {
                assert_eq!(tb.len(), 256);
                let roundtrip = s.delinearize_bytes(&tb);
                assert_eq!(roundtrip.ok(), Some(host_bytes.clone()));
            }
            let short_tiled = s.linearize_to_tpu_tiles(&[1, 2, 3, 4]);
            assert!(short_tiled.is_ok());
            assert!(s.linearize_to_tpu_tiles(&vec![0_u8; 512]).is_err());
            assert!(s.delinearize_from_tpu_tiles(&[0_u8; 4]).is_err());
        }

        for (et, dims, expected_geom) in [
            (ElementType::F32, vec![4_i64], (1, 8, 128)),
            (ElementType::Bf16, vec![2, 3], (2, 4, 128)),
            (ElementType::U8, vec![2, 3], (4, 4, 128)),
            (ElementType::S4, vec![2, 4], (8, 4, 128)),
        ] {
            for sh in TensorShape::new(et, dims, None, None) {
                assert_eq!(sh.tpu_tile_geometry(), expected_geom);
                let perm = sh.tile_permutation_indices();
                assert_eq!(perm.len(), sh.element_count());
                let raw: Vec<u8> = (0..sh.unpadded_byte_size().0)
                    .map(|i| u8::try_from(i % 251).unwrap_or(0))
                    .collect();
                for tb in sh.linear_to_tpu_tiled(&raw) {
                    assert_eq!(sh.tpu_tiled_to_linear(&tb).ok(), Some(raw.clone()));
                }
                for tb in sh.linearize_bytes(&raw) {
                    assert_eq!(sh.delinearize_bytes(&tb).ok(), Some(raw.clone()));
                }
            }
        }
    }
}

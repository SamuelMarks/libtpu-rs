//! Multi-rank [`CollectiveCommunicator`] and [`CrossHostTransferRegistry`] (`Phase 3.2`).
//!
//! Provides exact typed numeric element decoding/reduction/encoding across all PJRT numeric
//! element types (`F32`, `F64`, `Bf16`, `F16`, `S8`, `S16`, `S32`, `S64`, `U8`, `U16`, `U32`,
//! `U64`, `Pred`, and `FP8` formats) for `AllReduce`, `ReduceScatter`, `AllGather`, `AllToAll`,
//! and `CollectivePermute`, alongside a cross-host HBM transfer rendezvous registry pairing
//! `send_buffers` and `receive_buffers` by transfer key/tag.

use crate::error::{LibTpuError, LibTpuResult};
use crate::hal::TpuHal;
use crate::pjrt::{PjrtCoreBuffer, PjrtCoreEvent};
use crate::runtime_state::{SharedTpuRuntimeState, global_shared_runtime_state};
use crate::types::{CollectiveReductionOp, CommunicatorId, DeviceAddress, DeviceId, ElementType};
use parking_lot::Mutex;
use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, OnceLock};

/// Callback invoked with `(buffer_index, serialized_descriptor)` during cross-host transfers.
pub type CrossHostDescriptorCallback<'a> = &'a mut dyn FnMut(usize, &[u8]);

/// Returns the exact scalar byte width for supported collective numeric [`ElementType`]s.
///
/// # Arguments
///
/// * `elem_type` - Target PJRT [`ElementType`].
///
/// # Errors
///
/// Returns [`LibTpuError::InvalidArgument`] if `elem_type` is non-numeric or composite.
pub fn numeric_element_byte_width(elem_type: ElementType) -> LibTpuResult<usize> {
    match elem_type {
        ElementType::Pred
        | ElementType::S8
        | ElementType::U8
        | ElementType::F8E5M2
        | ElementType::F8E4M3Fn
        | ElementType::F8E4M3B11Fnuz
        | ElementType::F8E5M2Fnuz
        | ElementType::F8E4M3Fnuz
        | ElementType::F8E8M0Fnu => Ok(1),
        ElementType::S16 | ElementType::U16 | ElementType::F16 | ElementType::Bf16 => Ok(2),
        ElementType::S32 | ElementType::U32 | ElementType::F32 => Ok(4),
        ElementType::S64 | ElementType::U64 | ElementType::F64 => Ok(8),
        _ => Err(LibTpuError::InvalidArgument {
            parameter: "op_type",
            reason: format!("Unsupported collective reduction element type: {elem_type}"),
        }),
    }
}

/// Combines two `i64` scalars according to `op`.
///
/// # Arguments
///
/// * `lhs` - Accumulator signed integer scalar.
/// * `rhs` - Incoming signed integer scalar.
/// * `op` - [`CollectiveReductionOp`] combiner.
const fn combine_i64(lhs: i64, rhs: i64, op: CollectiveReductionOp) -> i64 {
    match op {
        CollectiveReductionOp::Sum => lhs.wrapping_add(rhs),
        CollectiveReductionOp::Product => lhs.wrapping_mul(rhs),
        CollectiveReductionOp::Min => {
            if lhs <= rhs {
                lhs
            } else {
                rhs
            }
        }
        CollectiveReductionOp::Max => {
            if lhs >= rhs {
                lhs
            } else {
                rhs
            }
        }
    }
}

/// Combines two `u64` scalars according to `op`.
///
/// # Arguments
///
/// * `lhs` - Accumulator unsigned integer scalar.
/// * `rhs` - Incoming unsigned integer scalar.
/// * `op` - [`CollectiveReductionOp`] combiner.
const fn combine_u64(lhs: u64, rhs: u64, op: CollectiveReductionOp) -> u64 {
    match op {
        CollectiveReductionOp::Sum => lhs.wrapping_add(rhs),
        CollectiveReductionOp::Product => lhs.wrapping_mul(rhs),
        CollectiveReductionOp::Min => {
            if lhs <= rhs {
                lhs
            } else {
                rhs
            }
        }
        CollectiveReductionOp::Max => {
            if lhs >= rhs {
                lhs
            } else {
                rhs
            }
        }
    }
}

/// Reduces `buffers` element-by-element using exact native integer or IEEE-754 floating-point
/// arithmetic for `elem_type` and `op`.
///
/// # Arguments
///
/// * `buffers` - Non-empty slice of equal-length input tensor byte slices.
/// * `elem_type` - Numeric [`ElementType`] of the input tensors.
/// * `op` - [`CollectiveReductionOp`] combiner (`Sum`, `Product`, `Min`, or `Max`).
///
/// # Errors
///
/// Returns [`LibTpuError::InvalidArgument`] if `buffers` is empty, buffer sizes mismatch,
/// `elem_type` is non-numeric, or byte length is not a multiple of element width.
#[allow(clippy::too_many_lines)]
pub fn reduce_numeric_buffers(
    buffers: &[&[u8]],
    elem_type: ElementType,
    op: CollectiveReductionOp,
) -> LibTpuResult<Vec<u8>> {
    let elem_width = numeric_element_byte_width(elem_type)?;
    let Some(&first) = buffers.first() else {
        return Err(LibTpuError::InvalidArgument {
            parameter: "send_buffer",
            reason: String::from("Collective reduction requires at least one participant buffer"),
        });
    };
    let byte_len = first.len();
    if byte_len % elem_width != 0 {
        return Err(LibTpuError::InvalidArgument {
            parameter: "send_buffer",
            reason: format!(
                "Buffer byte length {byte_len} is not divisible by {elem_type} width ({elem_width}B)"
            ),
        });
    }
    for (idx, buf) in buffers.iter().enumerate().skip(1) {
        if buf.len() != byte_len {
            return Err(LibTpuError::InvalidArgument {
                parameter: "send_buffer",
                reason: format!(
                    "Rank {idx} buffer size ({}B) mismatches rank 0 size ({byte_len}B)",
                    buf.len()
                ),
            });
        }
    }
    if buffers.len() == 1 || byte_len == 0 {
        return Ok(first.to_vec());
    }

    match elem_type {
        ElementType::Pred => {
            let mut acc: Vec<bool> = first.iter().map(|&b| b != 0).collect();
            for buf in buffers.iter().skip(1) {
                for (dst, &b) in acc.iter_mut().zip(buf.iter()) {
                    let rhs = b != 0;
                    *dst = match op {
                        CollectiveReductionOp::Sum | CollectiveReductionOp::Max => *dst || rhs,
                        CollectiveReductionOp::Product | CollectiveReductionOp::Min => *dst && rhs,
                    };
                }
            }
            Ok(acc.into_iter().map(u8::from).collect())
        }
        ElementType::S8 => {
            #[allow(clippy::cast_possible_wrap)]
            let mut acc: Vec<i64> = first.iter().map(|&b| i64::from(b as i8)).collect();
            for buf in buffers.iter().skip(1) {
                for (dst, &b) in acc.iter_mut().zip(buf.iter()) {
                    #[allow(clippy::cast_possible_wrap)]
                    let rhs = i64::from(b as i8);
                    *dst = combine_i64(*dst, rhs, op);
                }
            }
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            Ok(acc.into_iter().map(|v| (v as i8) as u8).collect())
        }
        ElementType::S16 => {
            let mut acc: Vec<i64> = first
                .chunks_exact(2)
                .map(|c| {
                    i64::from(i16::from_le_bytes(
                        <[u8; 2]>::try_from(c).unwrap_or_default(),
                    ))
                })
                .collect();
            for buf in buffers.iter().skip(1) {
                for (dst, c) in acc.iter_mut().zip(buf.chunks_exact(2)) {
                    let rhs = i64::from(i16::from_le_bytes(
                        <[u8; 2]>::try_from(c).unwrap_or_default(),
                    ));
                    *dst = combine_i64(*dst, rhs, op);
                }
            }
            let mut out = Vec::with_capacity(byte_len);
            for v in acc {
                #[allow(clippy::cast_possible_truncation)]
                out.extend_from_slice(&(v as i16).to_le_bytes());
            }
            Ok(out)
        }
        ElementType::S32 => {
            let mut acc: Vec<i64> = first
                .chunks_exact(4)
                .map(|c| {
                    i64::from(i32::from_le_bytes(
                        <[u8; 4]>::try_from(c).unwrap_or_default(),
                    ))
                })
                .collect();
            for buf in buffers.iter().skip(1) {
                for (dst, c) in acc.iter_mut().zip(buf.chunks_exact(4)) {
                    let rhs = i64::from(i32::from_le_bytes(
                        <[u8; 4]>::try_from(c).unwrap_or_default(),
                    ));
                    *dst = combine_i64(*dst, rhs, op);
                }
            }
            let mut out = Vec::with_capacity(byte_len);
            for v in acc {
                #[allow(clippy::cast_possible_truncation)]
                out.extend_from_slice(&(v as i32).to_le_bytes());
            }
            Ok(out)
        }
        ElementType::S64 => {
            let mut acc: Vec<i64> = first
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(<[u8; 8]>::try_from(c).unwrap_or_default()))
                .collect();
            for buf in buffers.iter().skip(1) {
                for (dst, c) in acc.iter_mut().zip(buf.chunks_exact(8)) {
                    let rhs = i64::from_le_bytes(<[u8; 8]>::try_from(c).unwrap_or_default());
                    *dst = combine_i64(*dst, rhs, op);
                }
            }
            let mut out = Vec::with_capacity(byte_len);
            for v in acc {
                out.extend_from_slice(&v.to_le_bytes());
            }
            Ok(out)
        }
        ElementType::U8 => {
            let mut acc: Vec<u64> = first.iter().map(|&b| u64::from(b)).collect();
            for buf in buffers.iter().skip(1) {
                for (dst, &b) in acc.iter_mut().zip(buf.iter()) {
                    *dst = combine_u64(*dst, u64::from(b), op);
                }
            }
            #[allow(clippy::cast_possible_truncation)]
            Ok(acc.into_iter().map(|v| v as u8).collect())
        }
        ElementType::U16 => {
            let mut acc: Vec<u64> = first
                .chunks_exact(2)
                .map(|c| {
                    u64::from(u16::from_le_bytes(
                        <[u8; 2]>::try_from(c).unwrap_or_default(),
                    ))
                })
                .collect();
            for buf in buffers.iter().skip(1) {
                for (dst, c) in acc.iter_mut().zip(buf.chunks_exact(2)) {
                    let rhs = u64::from(u16::from_le_bytes(
                        <[u8; 2]>::try_from(c).unwrap_or_default(),
                    ));
                    *dst = combine_u64(*dst, rhs, op);
                }
            }
            let mut out = Vec::with_capacity(byte_len);
            for v in acc {
                #[allow(clippy::cast_possible_truncation)]
                out.extend_from_slice(&(v as u16).to_le_bytes());
            }
            Ok(out)
        }
        ElementType::U32 => {
            let mut acc: Vec<u64> = first
                .chunks_exact(4)
                .map(|c| {
                    u64::from(u32::from_le_bytes(
                        <[u8; 4]>::try_from(c).unwrap_or_default(),
                    ))
                })
                .collect();
            for buf in buffers.iter().skip(1) {
                for (dst, c) in acc.iter_mut().zip(buf.chunks_exact(4)) {
                    let rhs = u64::from(u32::from_le_bytes(
                        <[u8; 4]>::try_from(c).unwrap_or_default(),
                    ));
                    *dst = combine_u64(*dst, rhs, op);
                }
            }
            let mut out = Vec::with_capacity(byte_len);
            for v in acc {
                #[allow(clippy::cast_possible_truncation)]
                out.extend_from_slice(&(v as u32).to_le_bytes());
            }
            Ok(out)
        }
        ElementType::U64 => {
            let mut acc: Vec<u64> = first
                .chunks_exact(8)
                .map(|c| u64::from_le_bytes(<[u8; 8]>::try_from(c).unwrap_or_default()))
                .collect();
            for buf in buffers.iter().skip(1) {
                for (dst, c) in acc.iter_mut().zip(buf.chunks_exact(8)) {
                    let rhs = u64::from_le_bytes(<[u8; 8]>::try_from(c).unwrap_or_default());
                    *dst = combine_u64(*dst, rhs, op);
                }
            }
            let mut out = Vec::with_capacity(byte_len);
            for v in acc {
                out.extend_from_slice(&v.to_le_bytes());
            }
            Ok(out)
        }
        _ => {
            let mut acc = elem_type.unpack_to_f64_vec(first);
            for buf in buffers.iter().skip(1) {
                let vals = elem_type.unpack_to_f64_vec(buf);
                for (dst, &rhs) in acc.iter_mut().zip(vals.iter()) {
                    *dst = match op {
                        CollectiveReductionOp::Sum => *dst + rhs,
                        CollectiveReductionOp::Product => *dst * rhs,
                        CollectiveReductionOp::Min => dst.min(rhs),
                        CollectiveReductionOp::Max => dst.max(rhs),
                    };
                }
            }
            Ok(elem_type.pack_from_f64_slice(&acc))
        }
    }
}

/// Multi-rank collective communicator backing `PJRT_Collectives_Communicator` (`Phase 3.2`).
#[derive(Debug, Clone)]
pub struct CollectiveCommunicator {
    /// Unique communicator identifier.
    pub comm_id: CommunicatorId,
    /// Participating logical TPU [`DeviceId`]s in rank order (`0..world_size`).
    pub devices: Vec<DeviceId>,
    /// Local participant rank index within `0..world_size`.
    pub rank: usize,
    /// Total number of participating ranks (`devices.len()`).
    pub world_size: usize,
    /// Shared per-rank staged payload map across peer communicators in the same replica group.
    group_staging: Arc<Mutex<BTreeMap<usize, Vec<u8>>>>,
    /// Attached [`SharedTpuRuntimeState`] for ICI collective telemetry and `XSpace` spans.
    runtime_state: Arc<SharedTpuRuntimeState>,
}

impl CollectiveCommunicator {
    /// Creates a new [`CollectiveCommunicator`] attached to [`global_shared_runtime_state`].
    ///
    /// # Arguments
    ///
    /// * `comm_id` - Unique [`CommunicatorId`].
    /// * `devices` - Non-empty ordered list of participating [`DeviceId`]s.
    /// * `rank` - Local rank within `0..devices.len()`.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if `devices` is empty or `rank >= devices.len()`.
    pub fn new(comm_id: CommunicatorId, devices: Vec<DeviceId>, rank: usize) -> LibTpuResult<Self> {
        Self::with_runtime_state(
            comm_id,
            devices,
            rank,
            Arc::clone(global_shared_runtime_state()),
        )
    }

    /// Creates a new [`CollectiveCommunicator`] bound to an explicit [`SharedTpuRuntimeState`].
    ///
    /// # Arguments
    ///
    /// * `comm_id` - Unique [`CommunicatorId`].
    /// * `devices` - Non-empty ordered list of participating [`DeviceId`]s.
    /// * `rank` - Local rank within `0..devices.len()`.
    /// * `runtime_state` - Shared telemetry and `XSpace` trace state.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if `devices` is empty or `rank >= devices.len()`.
    pub fn with_runtime_state(
        comm_id: CommunicatorId,
        devices: Vec<DeviceId>,
        rank: usize,
        runtime_state: Arc<SharedTpuRuntimeState>,
    ) -> LibTpuResult<Self> {
        let world_size = devices.len();
        if world_size == 0 {
            return Err(LibTpuError::InvalidArgument {
                parameter: "devices",
                reason: String::from("CollectiveCommunicator requires at least 1 device"),
            });
        }
        if rank >= world_size {
            return Err(LibTpuError::InvalidArgument {
                parameter: "rank",
                reason: format!(
                    "Communicator rank {rank} out of bounds for world_size {world_size}"
                ),
            });
        }
        Ok(Self {
            comm_id,
            devices,
            rank,
            world_size,
            group_staging: Arc::new(Mutex::new(BTreeMap::new())),
            runtime_state,
        })
    }

    /// Creates a full `world_size` group of peer [`CollectiveCommunicator`]s sharing a common
    /// multi-participant rendezvous staging buffer.
    ///
    /// # Arguments
    ///
    /// * `comm_id` - Unique [`CommunicatorId`].
    /// * `devices` - Non-empty ordered slice of participating [`DeviceId`]s.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if `devices` is empty.
    pub fn create_group(comm_id: CommunicatorId, devices: &[DeviceId]) -> LibTpuResult<Vec<Self>> {
        let world_size = devices.len();
        if world_size == 0 {
            return Err(LibTpuError::InvalidArgument {
                parameter: "devices",
                reason: String::from("CollectiveCommunicator group requires at least 1 device"),
            });
        }
        let shared_staging = Arc::new(Mutex::new(BTreeMap::new()));
        let rt = Arc::clone(global_shared_runtime_state());
        let mut group = Vec::with_capacity(world_size);
        for rank in 0..world_size {
            group.push(Self {
                comm_id,
                devices: devices.to_vec(),
                rank,
                world_size,
                group_staging: Arc::clone(&shared_staging),
                runtime_state: Arc::clone(&rt),
            });
        }
        Ok(group)
    }

    /// Stages `payload` for `rank` in the shared group rendezvous buffer.
    ///
    /// # Arguments
    ///
    /// * `rank` - Participant rank (`0..world_size`).
    /// * `payload` - Raw input tensor bytes contributed by `rank`.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if `rank >= self.world_size`.
    pub fn stage_rank_buffer(&self, rank: usize, payload: Vec<u8>) -> LibTpuResult<()> {
        if rank >= self.world_size {
            return Err(LibTpuError::InvalidArgument {
                parameter: "rank",
                reason: format!(
                    "Rank {rank} out of bounds for world_size {}",
                    self.world_size
                ),
            });
        }
        self.group_staging.lock().insert(rank, payload);
        Ok(())
    }

    /// Clears any previously staged rank buffers from the shared group rendezvous map.
    pub fn clear_staged_buffers(&self) {
        self.group_staging.lock().clear();
    }

    /// Formats a descriptive string representation (`PJRT_Communicator_ToString`).
    #[must_use]
    pub fn to_debug_string(&self) -> String {
        let dev_list = self
            .devices
            .iter()
            .map(|d| d.0.to_string())
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "TpuCollectiveCommunicator(id={}, rank={}, world_size={}, devices=[{dev_list}])",
            self.comm_id.0, self.rank, self.world_size
        )
    }

    /// Collects ordered rank payloads `(0..world_size)` combining `local_send` at `self.rank`
    /// with staged peer payloads (defaulting unpopulated peers to `default_peer`).
    ///
    /// # Arguments
    ///
    /// * `local_send` - Local rank's send buffer bytes.
    /// * `default_peer` - Fallback byte slice used for any unstaged peer rank.
    fn materialize_world_buffers(&self, local_send: &[u8], default_peer: &[u8]) -> Vec<Vec<u8>> {
        let staged_snapshot = self.group_staging.lock().clone();
        let mut world = Vec::with_capacity(self.world_size);
        for r in 0..self.world_size {
            if r == self.rank {
                world.push(local_send.to_vec());
            } else if let Some(staged) = staged_snapshot.get(&r) {
                world.push(staged.clone());
            } else {
                world.push(default_peer.to_vec());
            }
        }
        world
    }

    /// Executes `AllReduce` (`Sum`, `Product`, `Min`, or `Max`) across single-rank, 2-buffer,
    /// or N-device staged replica groups.
    ///
    /// # Arguments
    ///
    /// * `send_bytes` - Local rank input tensor bytes.
    /// * `peer_bytes` - Optional explicit peer/accumulator buffer when no group staging is present.
    /// * `elem_type` - Tensor [`ElementType`].
    /// * `op` - [`CollectiveReductionOp`] combiner.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] on invalid element type or mismatched buffer sizes.
    pub fn all_reduce(
        &self,
        send_bytes: &[u8],
        peer_bytes: Option<&[u8]>,
        elem_type: ElementType,
        op: CollectiveReductionOp,
    ) -> LibTpuResult<Vec<u8>> {
        let has_staged_peers = self.group_staging.lock().keys().any(|&r| r != self.rank);
        let result = if self.world_size > 1 && has_staged_peers {
            let owned = self.materialize_world_buffers(send_bytes, send_bytes);
            let refs: Vec<&[u8]> = owned.iter().map(Vec::as_slice).collect();
            reduce_numeric_buffers(&refs, elem_type, op)?
        } else if let Some(peer) = peer_bytes.filter(|_| self.world_size > 1) {
            reduce_numeric_buffers(&[send_bytes, peer], elem_type, op)?
        } else {
            reduce_numeric_buffers(&[send_bytes], elem_type, op)?
        };
        self.runtime_state
            .record_collective("all_reduce", u64::try_from(result.len()).unwrap_or(0));
        Ok(result)
    }

    /// Executes an N-participant `AllReduce` directly across `replica_buffers`.
    ///
    /// # Arguments
    ///
    /// * `replica_buffers` - Slice of participant send buffers (`1..=N`).
    /// * `elem_type` - Tensor [`ElementType`].
    /// * `op` - [`CollectiveReductionOp`] combiner.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if `replica_buffers` is empty or sizes mismatch.
    pub fn all_reduce_group(
        replica_buffers: &[&[u8]],
        elem_type: ElementType,
        op: CollectiveReductionOp,
    ) -> LibTpuResult<Vec<u8>> {
        reduce_numeric_buffers(replica_buffers, elem_type, op)
    }

    /// Executes `ReduceScatter` on this communicator, reducing across `world_size` equal chunks
    /// and returning this communicator's `rank`-th chunk of size `send_bytes.len() / world_size`.
    ///
    /// # Arguments
    ///
    /// * `send_bytes` - Local rank input buffer (must be divisible by `world_size * elem_width`).
    /// * `recv_len` - Expected output buffer size in bytes (`send_bytes.len() / world_size`).
    /// * `elem_type` - Tensor [`ElementType`].
    /// * `op` - [`CollectiveReductionOp`] combiner.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if `send_bytes` is not divisible by `world_size`
    /// or `recv_len != send_bytes.len() / world_size`.
    pub fn reduce_scatter(
        &self,
        send_bytes: &[u8],
        recv_len: usize,
        elem_type: ElementType,
        op: CollectiveReductionOp,
    ) -> LibTpuResult<Vec<u8>> {
        let elem_width = numeric_element_byte_width(elem_type)?;
        let required_granule = self.world_size.saturating_mul(elem_width);
        if send_bytes.is_empty() || send_bytes.len() % required_granule != 0 {
            return Err(LibTpuError::InvalidArgument {
                parameter: "send_buffer",
                reason: format!(
                    "ReduceScatter send_buffer ({}B) is not divisible by world_size ({}) * elem_width ({elem_width}B)",
                    send_bytes.len(),
                    self.world_size
                ),
            });
        }
        let expected_recv = send_bytes.len() / self.world_size;
        if recv_len != expected_recv {
            return Err(LibTpuError::InvalidArgument {
                parameter: "recv_buffer",
                reason: format!(
                    "ReduceScatter recv_buffer size ({recv_len}B) mismatches expected shard size ({expected_recv}B)"
                ),
            });
        }
        let has_staged_peers = self.group_staging.lock().keys().any(|&r| r != self.rank);
        let reduced = if self.world_size > 1 && has_staged_peers {
            let owned = self.materialize_world_buffers(send_bytes, send_bytes);
            let refs: Vec<&[u8]> = owned.iter().map(Vec::as_slice).collect();
            reduce_numeric_buffers(&refs, elem_type, op)?
        } else {
            reduce_numeric_buffers(&[send_bytes], elem_type, op)?
        };
        let offset = self.rank * expected_recv;
        let shard = reduced
            .get(offset..offset + expected_recv)
            .unwrap_or(&[])
            .to_vec();
        self.runtime_state
            .record_collective("reduce_scatter", u64::try_from(shard.len()).unwrap_or(0));
        Ok(shard)
    }

    /// Executes an N-participant `ReduceScatter` across `replica_buffers`, returning one shard per rank.
    ///
    /// # Arguments
    ///
    /// * `replica_buffers` - Slice of `world_size` equal-sized input buffers.
    /// * `elem_type` - Tensor [`ElementType`].
    /// * `op` - [`CollectiveReductionOp`] combiner.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if `replica_buffers` is empty or not divisible by `world_size`.
    pub fn reduce_scatter_group(
        replica_buffers: &[&[u8]],
        elem_type: ElementType,
        op: CollectiveReductionOp,
    ) -> LibTpuResult<Vec<Vec<u8>>> {
        let world_size = replica_buffers.len();
        let elem_width = numeric_element_byte_width(elem_type)?;
        let Some(&first) = replica_buffers.first() else {
            return Err(LibTpuError::InvalidArgument {
                parameter: "replica_buffers",
                reason: String::from("ReduceScatter group requires at least 1 rank buffer"),
            });
        };
        if first.is_empty() || first.len() % (world_size * elem_width) != 0 {
            return Err(LibTpuError::InvalidArgument {
                parameter: "send_buffer",
                reason: format!(
                    "ReduceScatter buffer ({}B) is not divisible by world_size ({world_size}) * {elem_width}B",
                    first.len()
                ),
            });
        }
        let full = reduce_numeric_buffers(replica_buffers, elem_type, op)?;
        let shard_bytes = full.len() / world_size;
        Ok(full.chunks_exact(shard_bytes).map(<[u8]>::to_vec).collect())
    }

    /// Executes `AllGather`, concatenating shard bytes from all `world_size` ranks in group order
    /// into an output buffer of size `send_bytes.len() * world_size`.
    ///
    /// # Arguments
    ///
    /// * `send_bytes` - Local rank input shard bytes.
    /// * `recv_len` - Expected output buffer size (`send_bytes.len() * world_size`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if `recv_len != send_bytes.len() * world_size`
    /// or any staged peer shard has a mismatched byte length.
    pub fn all_gather(&self, send_bytes: &[u8], recv_len: usize) -> LibTpuResult<Vec<u8>> {
        let expected_recv = send_bytes.len().saturating_mul(self.world_size);
        if send_bytes.is_empty() || recv_len != expected_recv {
            return Err(LibTpuError::InvalidArgument {
                parameter: "recv_buffer",
                reason: format!(
                    "AllGather recv_buffer ({recv_len}B) mismatches send_buffer ({}B) * world_size ({})",
                    send_bytes.len(),
                    self.world_size
                ),
            });
        }
        let world = self.materialize_world_buffers(send_bytes, send_bytes);
        let refs: Vec<&[u8]> = world.iter().map(Vec::as_slice).collect();
        let out = Self::all_gather_group(&refs)?;
        self.runtime_state
            .record_collective("all_gather", u64::try_from(out.len()).unwrap_or(0));
        Ok(out)
    }

    /// Concatenates equal-sized `rank_shards` in rank order (`0..world_size`).
    ///
    /// # Arguments
    ///
    /// * `rank_shards` - Non-empty slice of equal-length per-rank byte shards.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if `rank_shards` is empty or shard sizes mismatch.
    pub fn all_gather_group(rank_shards: &[&[u8]]) -> LibTpuResult<Vec<u8>> {
        let Some(&first) = rank_shards.first() else {
            return Err(LibTpuError::InvalidArgument {
                parameter: "rank_shards",
                reason: String::from("AllGather group requires at least 1 rank shard"),
            });
        };
        let shard_len = first.len();
        let mut out = Vec::with_capacity(shard_len * rank_shards.len());
        for (idx, &shard) in rank_shards.iter().enumerate() {
            if shard.len() != shard_len {
                return Err(LibTpuError::InvalidArgument {
                    parameter: "rank_shards",
                    reason: format!(
                        "AllGather rank {idx} shard ({}B) mismatches rank 0 shard ({shard_len}B)",
                        shard.len()
                    ),
                });
            }
            out.extend_from_slice(shard);
        }
        Ok(out)
    }

    /// Executes `AllToAll` on this communicator, splitting each rank's buffer into `world_size`
    /// equal blocks and transposing block `[src][self.rank]` into `[src]` of the local output.
    ///
    /// # Arguments
    ///
    /// * `send_bytes` - Local rank input buffer (must be divisible by `world_size`).
    /// * `recv_len` - Expected output buffer size (`send_bytes.len()`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if `send_bytes` is not divisible by `world_size`
    /// or `recv_len != send_bytes.len()`.
    pub fn all_to_all(&self, send_bytes: &[u8], recv_len: usize) -> LibTpuResult<Vec<u8>> {
        if send_bytes.is_empty()
            || send_bytes.len() % self.world_size != 0
            || recv_len != send_bytes.len()
        {
            return Err(LibTpuError::InvalidArgument {
                parameter: "send_buffer",
                reason: format!(
                    "AllToAll requires non-empty send_buffer ({}B) divisible by world_size ({}) and matching recv_buffer ({recv_len}B)",
                    send_bytes.len(),
                    self.world_size
                ),
            });
        }
        let world = self.materialize_world_buffers(send_bytes, send_bytes);
        let refs: Vec<&[u8]> = world.iter().map(Vec::as_slice).collect();
        let mut all_outputs = Self::all_to_all_group(&refs)?;
        let out = all_outputs.swap_remove(self.rank);
        self.runtime_state
            .record_collective("all_to_all", u64::try_from(out.len()).unwrap_or(0));
        Ok(out)
    }

    /// Executes an N-participant `AllToAll` block transpose across `rank_buffers`.
    ///
    /// # Arguments
    ///
    /// * `rank_buffers` - Slice of `world_size` equal-length rank buffers.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if `rank_buffers` is empty, sizes mismatch,
    /// or buffer byte length is not divisible by `rank_buffers.len()`.
    pub fn all_to_all_group(rank_buffers: &[&[u8]]) -> LibTpuResult<Vec<Vec<u8>>> {
        let world_size = rank_buffers.len();
        let Some(&first) = rank_buffers.first() else {
            return Err(LibTpuError::InvalidArgument {
                parameter: "rank_buffers",
                reason: String::from("AllToAll group requires at least 1 rank buffer"),
            });
        };
        let buf_len = first.len();
        if buf_len == 0 || buf_len % world_size != 0 {
            return Err(LibTpuError::InvalidArgument {
                parameter: "rank_buffers",
                reason: format!(
                    "AllToAll buffer ({buf_len}B) is not divisible by world_size ({world_size})"
                ),
            });
        }
        for (idx, &buf) in rank_buffers.iter().enumerate().skip(1) {
            if buf.len() != buf_len {
                return Err(LibTpuError::InvalidArgument {
                    parameter: "rank_buffers",
                    reason: format!(
                        "AllToAll rank {idx} buffer ({}B) mismatches rank 0 ({buf_len}B)",
                        buf.len()
                    ),
                });
            }
        }
        let block_bytes = buf_len / world_size;
        let mut outputs = vec![Vec::with_capacity(buf_len); world_size];
        for src_buf in rank_buffers {
            for (dst_out, block) in outputs.iter_mut().zip(src_buf.chunks_exact(block_bytes)) {
                dst_out.extend_from_slice(block);
            }
        }
        Ok(outputs)
    }

    /// Executes `CollectivePermute` on this communicator according to explicit `(src_rank, dst_rank)`
    /// pairs, zero-filling unmapped destinations.
    ///
    /// # Arguments
    ///
    /// * `send_bytes` - Local rank input buffer bytes.
    /// * `recv_len` - Expected output buffer byte length (`send_bytes.len()`).
    /// * `src_dst_pairs` - Explicit `(source_rank, target_rank)` routing pairs.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if `recv_len != send_bytes.len()` or any rank in
    /// `src_dst_pairs` exceeds `self.world_size`.
    pub fn collective_permute(
        &self,
        send_bytes: &[u8],
        recv_len: usize,
        src_dst_pairs: &[(usize, usize)],
    ) -> LibTpuResult<Vec<u8>> {
        if recv_len != send_bytes.len() {
            return Err(LibTpuError::InvalidArgument {
                parameter: "recv_buffer",
                reason: format!(
                    "CollectivePermute recv_buffer ({recv_len}B) mismatches send_buffer ({}B)",
                    send_bytes.len()
                ),
            });
        }
        let zeros = vec![0_u8; send_bytes.len()];
        let world = self.materialize_world_buffers(send_bytes, &zeros);
        let refs: Vec<&[u8]> = world.iter().map(Vec::as_slice).collect();
        let mut routed = Self::collective_permute_group(&refs, src_dst_pairs)?;
        let out = routed.swap_remove(self.rank);
        self.runtime_state
            .record_collective("collective_permute", u64::try_from(out.len()).unwrap_or(0));
        Ok(out)
    }

    /// Executes an N-participant `CollectivePermute` across `rank_buffers` using `src_dst_pairs`,
    /// zero-filling any unmapped destination ranks.
    ///
    /// # Arguments
    ///
    /// * `rank_buffers` - Slice of `world_size` equal-sized rank buffers.
    /// * `src_dst_pairs` - Slice of `(src_rank, dst_rank)` directed routing pairs.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if `rank_buffers` is empty, sizes mismatch,
    /// or any `(src_rank, dst_rank)` index is `>= world_size`.
    pub fn collective_permute_group(
        rank_buffers: &[&[u8]],
        src_dst_pairs: &[(usize, usize)],
    ) -> LibTpuResult<Vec<Vec<u8>>> {
        let world_size = rank_buffers.len();
        let Some(&first) = rank_buffers.first() else {
            return Err(LibTpuError::InvalidArgument {
                parameter: "rank_buffers",
                reason: String::from("CollectivePermute group requires at least 1 rank buffer"),
            });
        };
        let buf_len = first.len();
        for (idx, &buf) in rank_buffers.iter().enumerate().skip(1) {
            if buf.len() != buf_len {
                return Err(LibTpuError::InvalidArgument {
                    parameter: "rank_buffers",
                    reason: format!(
                        "CollectivePermute rank {idx} buffer ({}B) mismatches rank 0 ({buf_len}B)",
                        buf.len()
                    ),
                });
            }
        }
        let mut outputs = vec![vec![0_u8; buf_len]; world_size];
        for &(src, dst) in src_dst_pairs {
            let (Some(src_slice), Some(dst_vec)) = (rank_buffers.get(src), outputs.get_mut(dst))
            else {
                return Err(LibTpuError::InvalidArgument {
                    parameter: "source_target_pairs",
                    reason: format!(
                        "CollectivePermute pair ({src} -> {dst}) out of range for world_size {world_size}"
                    ),
                });
            };
            dst_vec.copy_from_slice(src_slice);
        }
        Ok(outputs)
    }
}

/// Waiting destination HBM buffer registered when `receive_buffers` arrives before `send_buffers`.
struct PendingCrossHostReceive {
    /// Destination logical [`DeviceId`].
    device_id: DeviceId,
    /// Destination HBM [`DeviceAddress`].
    address: DeviceAddress,
    /// Destination unpadded byte capacity.
    byte_len: usize,
    /// Backing [`TpuHal`] used to write received HBM payload bytes.
    hal: Arc<dyn TpuHal>,
    /// Completion event signaled once payload bytes land in destination HBM.
    done_event: Arc<PjrtCoreEvent>,
}

/// Synchronized mailbox pairing `PJRT_CrossHostTransfers` `SendBuffers` and `ReceiveBuffers` by key/tag.
pub struct CrossHostTransferRegistry {
    /// Staged outbound payloads waiting for a matching `receive_buffers` call, keyed by `transfer_key`.
    staged_sends: Mutex<BTreeMap<i64, VecDeque<Vec<Vec<u8>>>>>,
    /// Waiting inbound destination buffers registered before `send_buffers`, keyed by `transfer_key`.
    pending_receives: Mutex<BTreeMap<i64, VecDeque<Vec<PendingCrossHostReceive>>>>,
    /// Attached [`SharedTpuRuntimeState`] recording cross-host DMA bytes.
    runtime_state: Arc<SharedTpuRuntimeState>,
}

impl std::fmt::Debug for CrossHostTransferRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CrossHostTransferRegistry")
            .field("staged_keys", &self.staged_sends.lock().len())
            .field("pending_receive_keys", &self.pending_receives.lock().len())
            .finish_non_exhaustive()
    }
}

impl Default for CrossHostTransferRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl CrossHostTransferRegistry {
    /// Creates an empty [`CrossHostTransferRegistry`] attached to [`global_shared_runtime_state`].
    #[must_use]
    pub fn new() -> Self {
        Self::with_runtime_state(Arc::clone(global_shared_runtime_state()))
    }

    /// Creates an empty [`CrossHostTransferRegistry`] bound to `runtime_state`.
    ///
    /// # Arguments
    ///
    /// * `runtime_state` - Shared telemetry and `XSpace` trace state.
    #[must_use]
    pub const fn with_runtime_state(runtime_state: Arc<SharedTpuRuntimeState>) -> Self {
        Self {
            staged_sends: Mutex::new(BTreeMap::new()),
            pending_receives: Mutex::new(BTreeMap::new()),
            runtime_state,
        }
    }

    /// Synthesizes a deterministic wire transfer descriptor for `(transfer_key, buffer_index, byte_len)`.
    ///
    /// # Arguments
    ///
    /// * `transfer_key` - Rendezvous tag pairing sender and receiver.
    /// * `buffer_index` - Ordinal of the buffer within the batch.
    /// * `byte_len` - Unpadded payload byte length.
    #[must_use]
    pub fn format_serialized_descriptor(
        transfer_key: i64,
        buffer_index: usize,
        byte_len: usize,
    ) -> Vec<u8> {
        format!("LTPU_XHOST_V1:key={transfer_key}:idx={buffer_index}:bytes={byte_len}").into_bytes()
    }

    /// Reads HBM payload bytes from `src_buffers`, invokes `descriptor_cb` when supplied,
    /// fulfills any waiting `receive_buffers` call for `transfer_key` (or stages payloads in the
    /// mailbox), and returns ready send completion [`PjrtCoreEvent`]s.
    ///
    /// # Arguments
    ///
    /// * `transfer_key` - Rendezvous key/tag pairing sender and receiver.
    /// * `src_buffers` - Source [`PjrtCoreBuffer`] handles to transmit.
    /// * `descriptor_cb` - Optional callback invoked with `(buffer_index, serialized_descriptor)`.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if `src_buffers` is empty, any buffer is deleted, or HBM DMA fails.
    pub fn send_buffers(
        &self,
        transfer_key: i64,
        src_buffers: &[&PjrtCoreBuffer],
        mut descriptor_cb: Option<CrossHostDescriptorCallback<'_>>,
    ) -> LibTpuResult<Vec<Arc<PjrtCoreEvent>>> {
        if src_buffers.is_empty() {
            return Err(LibTpuError::InvalidArgument {
                parameter: "buffers",
                reason: String::from("CrossHostTransfers SendBuffers requires >= 1 buffer"),
            });
        }
        let mut payloads = Vec::with_capacity(src_buffers.len());
        let mut events = Vec::with_capacity(src_buffers.len());
        let mut total_bytes = 0_u64;
        for (idx, buf) in src_buffers.iter().enumerate() {
            let bytes = buf.to_host_bytes()?;
            total_bytes = total_bytes.saturating_add(u64::try_from(bytes.len()).unwrap_or(0));
            if let Some(cb) = descriptor_cb.as_deref_mut() {
                let desc = Self::format_serialized_descriptor(transfer_key, idx, bytes.len());
                cb(idx, &desc);
            }
            payloads.push(bytes);
            events.push(Arc::new(PjrtCoreEvent::ready_ok()));
        }

        let waiting = {
            let mut recv_guard = self.pending_receives.lock();
            let popped = recv_guard
                .get_mut(&transfer_key)
                .and_then(VecDeque::pop_front);
            if recv_guard
                .get(&transfer_key)
                .is_some_and(VecDeque::is_empty)
            {
                recv_guard.remove(&transfer_key);
            }
            popped
        };

        if let Some(recv_batch) = waiting {
            if recv_batch.len() != payloads.len() {
                for slot in &recv_batch {
                    slot.done_event.fail_with_reason(format!(
                        "CrossHostTransfers buffer count mismatch: sent {} vs expected {}",
                        payloads.len(),
                        recv_batch.len()
                    ));
                }
                return Err(LibTpuError::InvalidArgument {
                    parameter: "buffers",
                    reason: format!(
                        "CrossHostTransfers buffer count mismatch for key {transfer_key}"
                    ),
                });
            }
            for (payload, slot) in payloads.iter().zip(recv_batch.iter()) {
                let copy_len = slot.byte_len.min(payload.len());
                let slice = payload.get(..copy_len).unwrap_or(payload);
                slot.hal
                    .write_device_memory(slot.device_id, slot.address, slice)?;
                slot.done_event.set(None);
            }
        } else {
            self.staged_sends
                .lock()
                .entry(transfer_key)
                .or_default()
                .push_back(payloads);
        }

        self.runtime_state.record_dma_d2d(total_bytes);
        Ok(events)
    }

    /// Pairs `dst_buffers` with a matching `send_buffers` batch for `transfer_key`, copying real
    /// HBM payload bytes immediately when already staged (or registering pending HBM destinations
    /// completed when `send_buffers` arrives) and returning completion [`PjrtCoreEvent`]s.
    ///
    /// # Arguments
    ///
    /// * `transfer_key` - Rendezvous key/tag pairing sender and receiver.
    /// * `dst_buffers` - Destination [`PjrtCoreBuffer`] handles receiving HBM payloads.
    /// * `descriptor_cb` - Optional callback invoked with `(buffer_index, serialized_descriptor)`.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if `dst_buffers` is empty, any destination buffer is deleted,
    /// staged buffer counts mismatch, or HBM write fails.
    pub fn receive_buffers(
        &self,
        transfer_key: i64,
        dst_buffers: &[&PjrtCoreBuffer],
        mut descriptor_cb: Option<CrossHostDescriptorCallback<'_>>,
    ) -> LibTpuResult<Vec<Arc<PjrtCoreEvent>>> {
        if dst_buffers.is_empty() {
            return Err(LibTpuError::InvalidArgument {
                parameter: "buffers",
                reason: String::from("CrossHostTransfers ReceiveBuffers requires >= 1 buffer"),
            });
        }
        for (idx, dst) in dst_buffers.iter().enumerate() {
            if dst.is_deleted() {
                return Err(LibTpuError::BufferDeleted {
                    buffer_id: dst.id.0,
                });
            }
            if let Some(cb) = descriptor_cb.as_deref_mut() {
                let desc = Self::format_serialized_descriptor(
                    transfer_key,
                    idx,
                    dst.shape.unpadded_byte_size().0,
                );
                cb(idx, &desc);
            }
        }

        let staged = {
            let mut send_guard = self.staged_sends.lock();
            let popped = send_guard
                .get_mut(&transfer_key)
                .and_then(VecDeque::pop_front);
            if send_guard
                .get(&transfer_key)
                .is_some_and(VecDeque::is_empty)
            {
                send_guard.remove(&transfer_key);
            }
            popped
        };

        if let Some(payloads) = staged {
            if payloads.len() != dst_buffers.len() {
                return Err(LibTpuError::InvalidArgument {
                    parameter: "buffers",
                    reason: format!(
                        "CrossHostTransfers staged count ({}) mismatches destination count ({})",
                        payloads.len(),
                        dst_buffers.len()
                    ),
                });
            }
            let mut events = Vec::with_capacity(dst_buffers.len());
            for (payload, dst) in payloads.iter().zip(dst_buffers.iter()) {
                let copy_len = dst.shape.unpadded_byte_size().0.min(payload.len());
                let slice = payload.get(..copy_len).unwrap_or(payload);
                dst.hal
                    .write_device_memory(dst.device_id, dst.address, slice)?;
                events.push(Arc::new(PjrtCoreEvent::ready_ok()));
            }
            Ok(events)
        } else {
            let mut slots = Vec::with_capacity(dst_buffers.len());
            let mut events = Vec::with_capacity(dst_buffers.len());
            for dst in dst_buffers {
                let ev = Arc::new(PjrtCoreEvent::pending());
                slots.push(PendingCrossHostReceive {
                    device_id: dst.device_id,
                    address: dst.address,
                    byte_len: dst.shape.unpadded_byte_size().0,
                    hal: Arc::clone(&dst.hal),
                    done_event: Arc::clone(&ev),
                });
                events.push(ev);
            }
            self.pending_receives
                .lock()
                .entry(transfer_key)
                .or_default()
                .push_back(slots);
            Ok(events)
        }
    }
}

/// Process-wide singleton [`CrossHostTransferRegistry`] backing `PJRT_CrossHostTransfers_Extension`.
static GLOBAL_CROSS_HOST_TRANSFER_REGISTRY: OnceLock<CrossHostTransferRegistry> = OnceLock::new();

/// Returns a reference to the process-wide [`CrossHostTransferRegistry`].
#[must_use]
pub fn global_cross_host_transfer_registry() -> &'static CrossHostTransferRegistry {
    GLOBAL_CROSS_HOST_TRANSFER_REGISTRY.get_or_init(CrossHostTransferRegistry::new)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::config::TpuInitConfig;
    use crate::pjrt::PjrtCoreClient;
    use crate::types::Shape;

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_numeric_element_reductions_all_types_and_ops() -> LibTpuResult<()> {
        assert!(numeric_element_byte_width(ElementType::Token).is_err());
        assert!(reduce_numeric_buffers(&[], ElementType::F32, CollectiveReductionOp::Sum).is_err());
        assert!(
            reduce_numeric_buffers(&[&[1, 2, 3]], ElementType::F32, CollectiveReductionOp::Sum)
                .is_err()
        );
        assert!(
            reduce_numeric_buffers(
                &[&[1, 2, 3, 4], &[1, 2, 3, 4, 5, 6, 7, 8]],
                ElementType::F32,
                CollectiveReductionOp::Sum
            )
            .is_err()
        );
        assert_eq!(
            reduce_numeric_buffers(&[&[7, 8]], ElementType::U8, CollectiveReductionOp::Sum)?,
            vec![7, 8]
        );
        assert_eq!(
            reduce_numeric_buffers(&[&[], &[]], ElementType::U8, CollectiveReductionOp::Sum)?,
            Vec::<u8>::new()
        );

        assert_eq!(
            reduce_numeric_buffers(
                &[&[0, 1, 1], &[0, 0, 1]],
                ElementType::Pred,
                CollectiveReductionOp::Sum
            )?,
            vec![0, 1, 1]
        );
        assert_eq!(
            reduce_numeric_buffers(
                &[&[0, 1, 1], &[0, 0, 1]],
                ElementType::Pred,
                CollectiveReductionOp::Product
            )?,
            vec![0, 0, 1]
        );

        for op in [
            CollectiveReductionOp::Sum,
            CollectiveReductionOp::Product,
            CollectiveReductionOp::Min,
            CollectiveReductionOp::Max,
        ] {
            let _ = reduce_numeric_buffers(&[&[2_u8, 10_u8], &[3_u8, 5_u8]], ElementType::S8, op)?;
            let _ = reduce_numeric_buffers(
                &[&(-10_i16).to_le_bytes(), &25_i16.to_le_bytes()],
                ElementType::S16,
                op,
            )?;
            let _ = reduce_numeric_buffers(
                &[&(-100_i32).to_le_bytes(), &40_i32.to_le_bytes()],
                ElementType::S32,
                op,
            )?;
            let _ = reduce_numeric_buffers(&[&[4_u8, 10_u8], &[9_u8, 2_u8]], ElementType::U8, op)?;
            let _ = reduce_numeric_buffers(
                &[&100_u16.to_le_bytes(), &50_u16.to_le_bytes()],
                ElementType::U16,
                op,
            )?;
            let _ = reduce_numeric_buffers(
                &[&1000_u32.to_le_bytes(), &500_u32.to_le_bytes()],
                ElementType::U32,
                op,
            )?;
            let _ = reduce_numeric_buffers(
                &[&1000_u64.to_le_bytes(), &2000_u64.to_le_bytes()],
                ElementType::U64,
                op,
            )?;
        }
        let big_a: i64 = (1_i64 << 60) + 3;
        let big_b: i64 = 4;
        let sum_s64 = reduce_numeric_buffers(
            &[&big_a.to_le_bytes(), &big_b.to_le_bytes()],
            ElementType::S64,
            CollectiveReductionOp::Sum,
        )?;
        assert_eq!(sum_s64, (big_a + big_b).to_le_bytes());
        let min_s64 = reduce_numeric_buffers(
            &[&big_a.to_le_bytes(), &big_b.to_le_bytes()],
            ElementType::S64,
            CollectiveReductionOp::Min,
        )?;
        assert_eq!(min_s64, big_b.to_le_bytes());
        let max_s64 = reduce_numeric_buffers(
            &[&big_b.to_le_bytes(), &big_a.to_le_bytes()],
            ElementType::S64,
            CollectiveReductionOp::Max,
        )?;
        assert_eq!(max_s64, big_a.to_le_bytes());
        let prod_s64 = reduce_numeric_buffers(
            &[&3_i64.to_le_bytes(), &(-5_i64).to_le_bytes()],
            ElementType::S64,
            CollectiveReductionOp::Product,
        )?;
        assert_eq!(prod_s64, (-15_i64).to_le_bytes());

        let f32_a: Vec<u8> = [1.5_f32, -2.0_f32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let f32_b: Vec<u8> = [2.5_f32, 4.0_f32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        for op in [
            CollectiveReductionOp::Sum,
            CollectiveReductionOp::Product,
            CollectiveReductionOp::Min,
            CollectiveReductionOp::Max,
        ] {
            let _ = reduce_numeric_buffers(&[&f32_a, &f32_b], ElementType::F32, op)?;
        }
        for dt in [
            ElementType::F16,
            ElementType::Bf16,
            ElementType::F64,
            ElementType::F8E5M2,
            ElementType::F8E4M3Fn,
            ElementType::F8E4M3B11Fnuz,
            ElementType::F8E5M2Fnuz,
            ElementType::F8E4M3Fnuz,
            ElementType::F8E8M0Fnu,
        ] {
            let b0 = dt.pack_from_f64_slice(&[2.0, 4.0]);
            let b1 = dt.pack_from_f64_slice(&[2.0, 1.0]);
            let summed = reduce_numeric_buffers(&[&b0, &b1], dt, CollectiveReductionOp::Sum)?;
            assert_eq!(summed.len(), b0.len());
        }
        Ok(())
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_collective_communicator_multi_rank_and_error_paths() -> LibTpuResult<()> {
        assert!(CollectiveCommunicator::new(CommunicatorId(1), Vec::new(), 0).is_err());
        assert!(CollectiveCommunicator::new(CommunicatorId(1), vec![DeviceId(0)], 1).is_err());
        assert!(CollectiveCommunicator::create_group(CommunicatorId(1), &[]).is_err());

        let c1 = CollectiveCommunicator::new(CommunicatorId(10), vec![DeviceId(0)], 0)?;
        assert!(c1.to_debug_string().contains("world_size=1"));
        assert!(c1.stage_rank_buffer(1, vec![1]).is_err());
        let id_bytes = 5.0_f32.to_le_bytes();
        assert_eq!(
            c1.all_reduce(
                &id_bytes,
                None,
                ElementType::F32,
                CollectiveReductionOp::Sum
            )?,
            id_bytes
        );
        assert_eq!(
            c1.reduce_scatter(&id_bytes, 4, ElementType::F32, CollectiveReductionOp::Sum)?,
            id_bytes
        );

        let group =
            CollectiveCommunicator::create_group(CommunicatorId(20), &[DeviceId(0), DeviceId(1)])?;
        let [g0, g1] = group.as_slice() else {
            return Err(LibTpuError::InvalidArgument {
                parameter: "group",
                reason: String::from("expected 2-rank group"),
            });
        };
        let two_peer = g0.all_reduce(
            &2.0_f32.to_le_bytes(),
            Some(&3.0_f32.to_le_bytes()),
            ElementType::F32,
            CollectiveReductionOp::Sum,
        )?;
        assert_eq!(two_peer, 5.0_f32.to_le_bytes());

        let r0: Vec<u8> = [10.0_f32, 20.0_f32]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let r1: Vec<u8> = [3.0_f32, 7.0_f32]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        g0.stage_rank_buffer(0, r0.clone())?;
        g1.stage_rank_buffer(1, r1.clone())?;

        let ar0 = g0.all_reduce(&r0, None, ElementType::F32, CollectiveReductionOp::Sum)?;
        let expected_ar: Vec<u8> = [13.0_f32, 27.0_f32]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        assert_eq!(ar0, expected_ar);
        assert_eq!(
            CollectiveCommunicator::all_reduce_group(
                &[&r0, &r1],
                ElementType::F32,
                CollectiveReductionOp::Sum
            )?,
            expected_ar
        );

        assert!(
            g0.reduce_scatter(
                &[1, 2, 3, 4],
                2,
                ElementType::F32,
                CollectiveReductionOp::Sum
            )
            .is_err()
        );
        assert!(
            g0.reduce_scatter(&r0, 8, ElementType::F32, CollectiveReductionOp::Sum)
                .is_err()
        );
        assert_eq!(
            g0.reduce_scatter(&r0, 4, ElementType::F32, CollectiveReductionOp::Sum)?,
            13.0_f32.to_le_bytes()
        );
        assert_eq!(
            g1.reduce_scatter(&r1, 4, ElementType::F32, CollectiveReductionOp::Sum)?,
            27.0_f32.to_le_bytes()
        );
        assert!(
            CollectiveCommunicator::reduce_scatter_group(
                &[],
                ElementType::F32,
                CollectiveReductionOp::Sum
            )
            .is_err()
        );
        assert!(
            CollectiveCommunicator::reduce_scatter_group(
                &[&[1, 2, 3, 4], &[1, 2, 3, 4]],
                ElementType::F32,
                CollectiveReductionOp::Sum
            )
            .is_err()
        );
        let rs_grp = CollectiveCommunicator::reduce_scatter_group(
            &[&r0, &r1],
            ElementType::F32,
            CollectiveReductionOp::Sum,
        )?;
        assert_eq!(
            rs_grp,
            vec![
                13.0_f32.to_le_bytes().to_vec(),
                27.0_f32.to_le_bytes().to_vec()
            ]
        );

        assert!(g0.all_gather(&r0, 4).is_err());
        let ag = g0.all_gather(&r0, 16)?;
        assert_eq!(ag.get(..8), Some(r0.as_slice()));
        assert_eq!(ag.get(8..), Some(r1.as_slice()));
        assert!(CollectiveCommunicator::all_gather_group(&[]).is_err());
        assert!(CollectiveCommunicator::all_gather_group(&[&[1, 2], &[1]]).is_err());

        assert!(g0.all_to_all(&[1, 2, 3], 3).is_err());
        assert!(g0.all_to_all(&r0, 4).is_err());
        let a2a_0 = g0.all_to_all(&r0, 8)?;
        let a2a_1 = g1.all_to_all(&r1, 8)?;
        assert_eq!(a2a_0.get(..4), Some(&10.0_f32.to_le_bytes()[..]));
        assert_eq!(a2a_0.get(4..), Some(&3.0_f32.to_le_bytes()[..]));
        assert_eq!(a2a_1.get(..4), Some(&20.0_f32.to_le_bytes()[..]));
        assert_eq!(a2a_1.get(4..), Some(&7.0_f32.to_le_bytes()[..]));
        assert!(CollectiveCommunicator::all_to_all_group(&[]).is_err());
        assert!(CollectiveCommunicator::all_to_all_group(&[&[1], &[1]]).is_err());
        assert!(CollectiveCommunicator::all_to_all_group(&[&[1, 2], &[1, 2, 3, 4]]).is_err());

        assert!(g0.collective_permute(&r0, 4, &[(0, 1)]).is_err());
        assert!(g0.collective_permute(&r0, 8, &[(0, 9)]).is_err());
        assert!(g0.collective_permute(&r0, 8, &[(9, 0)]).is_err());
        let cp_0 = g0.collective_permute(&r0, 8, &[(0, 1)])?;
        let cp_1 = g1.collective_permute(&r1, 8, &[(0, 1)])?;
        assert_eq!(cp_0, vec![0_u8; 8]);
        assert_eq!(cp_1, r0);
        assert!(CollectiveCommunicator::collective_permute_group(&[], &[]).is_err());
        assert!(
            CollectiveCommunicator::collective_permute_group(&[&[1, 2], &[1]], &[(0, 1)]).is_err()
        );

        g0.clear_staged_buffers();
        assert!(
            g0.reduce_scatter(&[], 0, ElementType::F32, CollectiveReductionOp::Sum)
                .is_err()
        );
        assert_eq!(
            g0.reduce_scatter(&r0, 4, ElementType::F32, CollectiveReductionOp::Sum)?,
            10.0_f32.to_le_bytes()
        );
        assert!(
            CollectiveCommunicator::reduce_scatter_group(
                &[&[], &[]],
                ElementType::F32,
                CollectiveReductionOp::Sum
            )
            .is_err()
        );
        assert!(g0.all_gather(&[], 0).is_err());
        assert!(g0.all_to_all(&[], 0).is_err());
        assert!(CollectiveCommunicator::all_to_all_group(&[&[], &[]]).is_err());
        let unstaged_ag = g0.all_gather(&[1, 2], 4)?;
        assert_eq!(unstaged_ag, vec![1, 2, 1, 2]);
        Ok(())
    }

    #[test]
    fn test_cross_host_transfer_registry_send_first_and_recv_first() -> LibTpuResult<()> {
        let client = PjrtCoreClient::from_config(&TpuInitConfig::default());
        let shape = Shape::new(ElementType::U8, vec![4], None, None)?;
        let src_a = client.buffer_from_host(DeviceId(0), shape.clone(), &[11, 22, 33, 44])?;
        let dst_a = client.buffer_from_host(DeviceId(1), shape.clone(), &[0, 0, 0, 0])?;
        let dst_b = client.buffer_from_host(DeviceId(1), shape.clone(), &[0, 0, 0, 0])?;
        let extra = client.buffer_from_host(DeviceId(0), shape, &[9, 9, 9, 9])?;

        let reg = CrossHostTransferRegistry::default();
        assert_ne!(format!("{reg:?}"), "");
        assert!(reg.send_buffers(1, &[], None).is_err());
        assert!(reg.receive_buffers(1, &[], None).is_err());

        let mut send_descs = Vec::new();
        let mut recv_descs = Vec::new();
        let mut s_cb = |idx: usize, d: &[u8]| send_descs.push((idx, d.to_vec()));
        let mut r_cb = |idx: usize, d: &[u8]| recv_descs.push((idx, d.to_vec()));
        let s_evs = reg.send_buffers(101, &[&src_a], Some(&mut s_cb))?;
        assert!(s_evs.first().is_some_and(|e| e.is_ready()));
        assert_eq!(send_descs.len(), 1);
        let r_evs = reg.receive_buffers(101, &[&dst_a], Some(&mut r_cb))?;
        assert!(r_evs.first().is_some_and(|e| e.is_ready()));
        assert_eq!(recv_descs.len(), 1);
        assert_eq!(dst_a.to_host_bytes()?, vec![11, 22, 33, 44]);

        let _ = reg.send_buffers(102, &[&src_a, &extra], None)?;
        assert!(reg.receive_buffers(102, &[&dst_a], None).is_err());

        let pending_evs = reg.receive_buffers(202, &[&dst_b], None)?;
        let p0 = pending_evs
            .first()
            .ok_or_else(|| LibTpuError::InvalidArgument {
                parameter: "pending_evs",
                reason: String::from("missing pending event"),
            })?;
        assert!(!p0.is_ready());
        let _ = reg.send_buffers(202, &[&src_a], None)?;
        assert!(p0.is_ready());
        p0.await_ready()?;
        assert_eq!(dst_b.to_host_bytes()?, vec![11, 22, 33, 44]);

        let mismatch_evs = reg.receive_buffers(203, &[&dst_b], None)?;
        let m0 = mismatch_evs
            .first()
            .ok_or_else(|| LibTpuError::InvalidArgument {
                parameter: "mismatch_evs",
                reason: String::from("missing mismatch event"),
            })?;
        assert!(reg.send_buffers(203, &[&src_a, &extra], None).is_err());
        assert!(m0.is_ready());
        assert!(m0.await_ready().is_err());

        let _ = reg.send_buffers(303, &[&src_a], None)?;
        let _ = reg.send_buffers(303, &[&extra], None)?;
        let _ = reg.receive_buffers(303, &[&dst_a], None)?;
        let _ = reg.receive_buffers(303, &[&dst_b], None)?;
        assert_eq!(dst_b.to_host_bytes()?, vec![9, 9, 9, 9]);

        let q1 = reg.receive_buffers(404, &[&dst_a], None)?;
        let q2 = reg.receive_buffers(404, &[&dst_b], None)?;
        let _ = reg.send_buffers(404, &[&src_a], None)?;
        assert!(q1.first().is_some_and(|e| e.is_ready()));
        assert!(q2.first().is_some_and(|e| !e.is_ready()));
        let _ = reg.send_buffers(404, &[&extra], None)?;
        assert!(q2.first().is_some_and(|e| e.is_ready()));

        extra.delete()?;
        assert!(reg.receive_buffers(505, &[&extra], None).is_err());
        assert_ne!(format!("{:?}", global_cross_host_transfer_registry()), "");
        Ok(())
    }
}

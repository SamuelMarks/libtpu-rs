//! SPMEM scratchpad tiling (`SpmemScratchpadTiler`), deterministic COO -> CSR input sorting
//! (`CooCsrSorter`), and deduplicated embedding gather/gradient-coalescing (`EmbeddingGatherDedupEngine`).

use crate::error::{LibTpuError, LibTpuResult};
use crate::proto::SparseCoreTileExecutionScheduleProto;

/// Single sparse coordinate lookup entry `(sample_id, row_id, weight)` (`CooSparseEntry`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CooSparseEntry {
    /// Output batch sample index (`0..batch_size`).
    pub sample_id: u32,
    /// Vocabulary row identifier (`0..vocab_size`).
    pub row_id: u32,
    /// Per-lookup combiner weight (`1.0` for unweighted lookups).
    pub weight: f32,
}

/// Compressed Sparse Row (`CSR`) batch produced by [`CooCsrSorter`] (`CsrSparseBatch`).
#[derive(Debug, Clone, PartialEq)]
pub struct CsrSparseBatch {
    /// Number of batch samples (`row_pointers.len() == batch_size + 1`).
    pub batch_size: usize,
    /// CSR row pointer offsets (`0..=nnz`).
    pub row_pointers: Vec<u32>,
    /// Sorted vocabulary row IDs (`len == nnz`).
    pub col_row_ids: Vec<u32>,
    /// Aligned lookup weights (`len == nnz`).
    pub weights: Vec<f32>,
    /// Sorted unique vocabulary row IDs required by this batch (`len <= nnz`).
    pub unique_row_ids: Vec<u32>,
    /// Dense index into `unique_row_ids` for each entry in `col_row_ids` (`len == nnz`).
    pub gather_to_unique_idx: Vec<usize>,
}

/// Pooling combiner reduction mode for embedding bags (`EmbeddingCombiner`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingCombiner {
    /// Unweighted or weighted sum (`sum_i w_i * E[r_i]`).
    Sum,
    /// Arithmetic mean (`(sum_i w_i * E[r_i]) / sum_i w_i`).
    Mean,
    /// Square-root normalized sum (`(sum_i w_i * E[r_i]) / sqrt(sum_i w_i^2)`).
    Sqrtn,
}

/// Hardware-equivalent COO -> CSR radix sorter and vocabulary row deduplicator (`CooCsrSorter`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CooCsrSorter;

impl CooCsrSorter {
    /// Sorts `entries` by `(sample_id, row_id)`, builds CSR row pointers for `batch_size`, and
    /// deduplicates `row_id`s into a compact gather table.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - Number of samples in the batch (`>= 1`).
    /// * `vocab_size` - Vocabulary row count upper bound.
    /// * `entries` - Unsorted [`CooSparseEntry`] slice.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::SparseCoreCodecError`] if `batch_size == 0` or any `sample_id` / `row_id` is out of bounds.
    pub fn sort_and_dedup(
        batch_size: usize,
        vocab_size: u32,
        entries: &[CooSparseEntry],
    ) -> LibTpuResult<CsrSparseBatch> {
        if batch_size == 0 {
            return Err(LibTpuError::SparseCoreCodecError {
                core_id: 0,
                pc_offset: 0,
                reason: String::from("CSR batch_size must be >= 1"),
            });
        }
        for e in entries {
            if (e.sample_id as usize >= batch_size) || (e.row_id >= vocab_size) {
                return Err(LibTpuError::SparseCoreCodecError {
                    core_id: 0,
                    pc_offset: 0,
                    reason: format!(
                        "COO entry (sample={}, row={}) out of bounds (batch_size={batch_size}, vocab_size={vocab_size})",
                        e.sample_id, e.row_id
                    ),
                });
            }
        }

        let mut sorted = entries.to_vec();
        sorted.sort_by_key(|e| (e.sample_id, e.row_id));

        let mut row_pointers = vec![0_u32; batch_size + 1];
        let mut col_row_ids = Vec::with_capacity(sorted.len());
        let mut weights = Vec::with_capacity(sorted.len());
        for e in &sorted {
            row_pointers
                .get_mut((e.sample_id as usize) + 1)
                .into_iter()
                .for_each(|cnt| *cnt = cnt.saturating_add(1));
            col_row_ids.push(e.row_id);
            weights.push(e.weight);
        }
        for i in 0..batch_size {
            let prev = row_pointers.get(i).copied().unwrap_or(0);
            row_pointers
                .get_mut(i + 1)
                .into_iter()
                .for_each(|cur| *cur = cur.saturating_add(prev));
        }

        let mut unique_row_ids = col_row_ids.clone();
        unique_row_ids.sort_unstable();
        unique_row_ids.dedup();

        let gather_to_unique_idx = col_row_ids
            .iter()
            .map(|r| unique_row_ids.binary_search(r).unwrap_or(0))
            .collect();

        Ok(CsrSparseBatch {
            batch_size,
            row_pointers,
            col_row_ids,
            weights,
            unique_row_ids,
            gather_to_unique_idx,
        })
    }
}

/// SPMEM scratchpad tiler (`SpmemScratchpadTiler`) partitioning vocabulary tables across SPMEM tiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpmemScratchpadTiler {
    /// Total per-SparseCore SPMEM scratchpad capacity in bytes (e.g., `262_144` or `524_288`).
    pub spmem_capacity_bytes: u32,
}

impl SpmemScratchpadTiler {
    /// Creates an [`SpmemScratchpadTiler`] with `spmem_capacity_bytes`.
    ///
    /// # Arguments
    ///
    /// * `spmem_capacity_bytes` - Per-core SPMEM scratchpad byte budget.
    #[must_use]
    pub const fn new(spmem_capacity_bytes: u32) -> Self {
        Self {
            spmem_capacity_bytes,
        }
    }

    /// Computes bank-aligned [`SparseCoreTileExecutionScheduleProto`] tiles for an embedding table.
    ///
    /// # Arguments
    ///
    /// * `vocab_rows` - Total vocabulary row count (`>= 1`).
    /// * `embedding_dim` - Embedding vector width (`>= 1`).
    /// * `num_optimizer_slots` - Number of auxiliary optimizer state slots per row (`0..=3`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::SparseCoreCodecError`] if a single vocabulary row + slots exceeds `spmem_capacity_bytes`.
    pub fn plan_tiles(
        &self,
        vocab_rows: u32,
        embedding_dim: u32,
        num_optimizer_slots: u32,
    ) -> LibTpuResult<Vec<SparseCoreTileExecutionScheduleProto>> {
        let row_act_bytes = embedding_dim.max(1).saturating_mul(4);
        let row_total_bytes = row_act_bytes.saturating_mul(1 + num_optimizer_slots);
        if (vocab_rows == 0) || (row_total_bytes > self.spmem_capacity_bytes) {
            return Err(LibTpuError::SparseCoreCodecError {
                core_id: 0,
                pc_offset: 0,
                reason: format!(
                    "Single row ({row_total_bytes} B) exceeds SPMEM capacity ({} B) or vocab_rows == 0",
                    self.spmem_capacity_bytes
                ),
            });
        }

        let rows_per_tile = (self.spmem_capacity_bytes / row_total_bytes).max(1);
        let mut schedules = Vec::new();
        let mut start_row = 0_u32;
        let mut tile_id = 0_u32;
        while start_row < vocab_rows {
            let count = (vocab_rows - start_row).min(rows_per_tile);
            let tile_bytes = count.saturating_mul(row_total_bytes);
            let base_off = (tile_id.saturating_mul(tile_bytes)) % self.spmem_capacity_bytes.max(1);
            schedules.push(SparseCoreTileExecutionScheduleProto {
                tile_id,
                tec_start_pc: tile_id * 4,
                tec_bundle_count: count.max(1),
                spmem_base_offset: base_off,
                spmem_size_bytes: tile_bytes,
            });
            start_row = start_row.saturating_add(count);
            tile_id = tile_id.saturating_add(1);
        }
        Ok(schedules)
    }
}

/// Deduplicated embedding table gather & backward gradient coalescing engine (`EmbeddingGatherDedupEngine`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EmbeddingGatherDedupEngine;

impl EmbeddingGatherDedupEngine {
    /// Performs deduplicated embedding table gather + combiner reduction (`Sum`, `Mean`, `Sqrtn`)
    /// producing `[batch_size * embedding_dim]` activations.
    ///
    /// # Arguments
    ///
    /// * `table` - Flattened `[vocab_size * embedding_dim]` embedding weights.
    /// * `embedding_dim` - Embedding vector width.
    /// * `csr` - Sorted & deduplicated [`CsrSparseBatch`].
    /// * `combiner` - [`EmbeddingCombiner`] reduction mode.
    #[must_use]
    pub fn forward_gather_dedup(
        table: &[f32],
        embedding_dim: usize,
        csr: &CsrSparseBatch,
        combiner: EmbeddingCombiner,
    ) -> Vec<f32> {
        // Step 1: Gather each unique row once into SPMEM staging buffer
        let mut spmem_unique = vec![0.0_f32; csr.unique_row_ids.len() * embedding_dim];
        for (u_idx, &row_id) in csr.unique_row_ids.iter().enumerate() {
            let src_base = (row_id as usize) * embedding_dim;
            let dst_base = u_idx * embedding_dim;
            let src_slice = table.get(src_base..src_base + embedding_dim).unwrap_or(&[]);
            spmem_unique
                .get_mut(dst_base..dst_base + embedding_dim)
                .into_iter()
                .for_each(|dst| {
                    dst.iter_mut()
                        .zip(src_slice.iter())
                        .for_each(|(d, &s)| *d = s);
                });
        }

        // Step 2: Combine deduplicated rows per sample using CSR row pointers
        let mut out = vec![0.0_f32; csr.batch_size * embedding_dim];
        for sample in 0..csr.batch_size {
            let beg = csr.row_pointers.get(sample).copied().unwrap_or(0) as usize;
            let end = csr.row_pointers.get(sample + 1).copied().unwrap_or(0) as usize;
            let mut w_sum = 0.0_f32;
            let mut w_sq_sum = 0.0_f32;
            let out_base = sample * embedding_dim;
            for nz_idx in beg..end {
                let u_idx = csr.gather_to_unique_idx.get(nz_idx).copied().unwrap_or(0);
                let w = csr.weights.get(nz_idx).copied().unwrap_or(1.0);
                w_sum += w;
                w_sq_sum = f32::mul_add(w, w, w_sq_sum);
                let row_base = u_idx * embedding_dim;
                for d in 0..embedding_dim {
                    let val = spmem_unique.get(row_base + d).copied().unwrap_or(0.0);
                    out.get_mut(out_base + d)
                        .into_iter()
                        .for_each(|acc| *acc = f32::mul_add(w, val, *acc));
                }
            }
            let norm = match combiner {
                EmbeddingCombiner::Sum => 1.0_f32,
                EmbeddingCombiner::Mean => {
                    if w_sum.abs() > 1e-12 {
                        1.0 / w_sum
                    } else {
                        1.0
                    }
                }
                EmbeddingCombiner::Sqrtn => {
                    if w_sq_sum > 1e-12 {
                        1.0 / w_sq_sum.sqrt()
                    } else {
                        1.0
                    }
                }
            };
            if (norm - 1.0).abs() > 1e-12 {
                out.get_mut(out_base..out_base + embedding_dim)
                    .into_iter()
                    .flatten()
                    .for_each(|v| *v *= norm);
            }
        }
        out
    }

    /// Coalesces upstream activation gradients `[batch_size * embedding_dim]` into dense per-unique-row
    /// gradients `[csr.unique_row_ids.len() * embedding_dim]` for single-write optimizer updates.
    ///
    /// # Arguments
    ///
    /// * `upstream_grad` - Upstream `[batch_size * embedding_dim]` gradient tensor.
    /// * `embedding_dim` - Embedding vector width.
    /// * `csr` - Sorted & deduplicated [`CsrSparseBatch`].
    #[must_use]
    pub fn backward_coalesce_gradients(
        upstream_grad: &[f32],
        embedding_dim: usize,
        csr: &CsrSparseBatch,
    ) -> Vec<f32> {
        let mut unique_grads = vec![0.0_f32; csr.unique_row_ids.len() * embedding_dim];
        for sample in 0..csr.batch_size {
            let beg = csr.row_pointers.get(sample).copied().unwrap_or(0) as usize;
            let end = csr.row_pointers.get(sample + 1).copied().unwrap_or(0) as usize;
            let up_base = sample * embedding_dim;
            for nz_idx in beg..end {
                let u_idx = csr.gather_to_unique_idx.get(nz_idx).copied().unwrap_or(0);
                let w = csr.weights.get(nz_idx).copied().unwrap_or(1.0);
                let dst_base = u_idx * embedding_dim;
                for d in 0..embedding_dim {
                    let g = upstream_grad.get(up_base + d).copied().unwrap_or(0.0);
                    unique_grads
                        .get_mut(dst_base + d)
                        .into_iter()
                        .for_each(|dst| *dst = f32::mul_add(w, g, *dst));
                }
            }
        }
        unique_grads
    }
}

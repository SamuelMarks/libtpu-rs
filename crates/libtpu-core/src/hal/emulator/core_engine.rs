//! Binary VLIW & `SparseCore` bitstream execution engine and `ICI` RDMA fabric.

use super::super::{ImemAttributeEntry, TpuCoreVmState};
use crate::compiler::isa::{decode_bundle_for_generation, generation_bundle_byte_width};
use crate::compiler::sparsecore::{SparseCoreOp, SparseCoreV2Codec, SparseCoreV3Codec};
use crate::compiler::{CompiledProgram, LloBundle};
use crate::error::{LibTpuError, LibTpuResult};
use crate::types::{DeviceId, ElementType, TpuGeneration};
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Reserved HBM physical base address for relocated `TensorCore` binary `IMEM` bitstreams (`8 MiB`).
pub const HBM_IMEM_TENSORCORE_BASE: u64 = 0x0080_0000;

/// Reserved HBM physical base address for constant/attribute `CMEM` `.rodata` descriptors (`10 MiB`).
pub const HBM_CMEM_RODATA_BASE: u64 = 0x00A0_0000;

/// Reserved HBM physical base address for relocated `SparseCore` V2/V3 binary bitstreams (`12 MiB`).
pub const HBM_IMEM_SPARSECORE_BASE: u64 = 0x00C0_0000;

/// Reserved HBM physical base address for hardware performance counter & `XPlane` trace ring (`14 MiB`).
pub const HBM_XPLANE_TRACE_RING_BASE: u64 = 0x00E0_0000;

/// Hardware `4 KiB` page shift for [`SparseHbmStore`].
const HBM_PAGE_SHIFT: u32 = 12;

/// Hardware `4 KiB` page byte size (`4096` B).
const HBM_PAGE_BYTES: usize = 4096;

/// Sparse 64-bit `4 KiB`-paged physical HBM backing store (`SparseHbmStore`) supporting up to
/// `192 GiB` (`HBM3e` on Cloud TPU `v7x` Ironwood) with demand-allocated `4 KiB` pages.
#[derive(Debug, Clone, Default)]
pub struct SparseHbmStore {
    /// Demand-allocated `4 KiB` physical HBM pages keyed by `page_number = phys_addr >> 12`.
    pages: BTreeMap<u64, Vec<u8>>,
}

impl SparseHbmStore {
    /// Creates an empty [`SparseHbmStore`].
    #[must_use]
    pub const fn new() -> Self {
        Self {
            pages: BTreeMap::new(),
        }
    }

    /// Writes `data` at physical HBM address `phys_addr`, allocating `4 KiB` pages on demand.
    ///
    /// # Arguments
    ///
    /// * `phys_addr` - Physical 64-bit HBM byte address.
    /// * `data` - Source byte slice.
    pub fn write_bytes(&mut self, phys_addr: u64, data: &[u8]) {
        let mut cursor = phys_addr;
        let mut remaining = data;
        while !remaining.is_empty() {
            let page_no = cursor >> HBM_PAGE_SHIFT;
            #[allow(clippy::cast_possible_truncation)]
            let page_off = (cursor & ((HBM_PAGE_BYTES as u64) - 1)) as usize;
            let chunk_len = remaining.len().min(HBM_PAGE_BYTES - page_off);
            let page = self
                .pages
                .entry(page_no)
                .or_insert_with(|| vec![0_u8; HBM_PAGE_BYTES]);
            for (d, &s) in page
                .iter_mut()
                .skip(page_off)
                .take(chunk_len)
                .zip(remaining.iter())
            {
                *d = s;
            }
            cursor = cursor.saturating_add(chunk_len as u64);
            remaining = remaining.get(chunk_len..).unwrap_or(&[]);
        }
    }

    /// Reads `len` bytes starting at physical HBM address `phys_addr` (unwritten pages read as `0`).
    ///
    /// # Arguments
    ///
    /// * `phys_addr` - Physical 64-bit HBM byte address.
    /// * `len` - Number of bytes to read.
    #[must_use]
    pub fn read_bytes(&self, phys_addr: u64, len: usize) -> Vec<u8> {
        let mut out = vec![0_u8; len];
        self.read_into(phys_addr, &mut out);
        out
    }

    /// Reads `dst.len()` bytes starting at physical HBM address `phys_addr` into `dst`.
    ///
    /// # Arguments
    ///
    /// * `phys_addr` - Physical 64-bit HBM byte address.
    /// * `dst` - Destination byte slice.
    pub fn read_into(&self, phys_addr: u64, dst: &mut [u8]) {
        let mut cursor = phys_addr;
        let mut offset = 0_usize;
        while offset < dst.len() {
            let page_no = cursor >> HBM_PAGE_SHIFT;
            #[allow(clippy::cast_possible_truncation)]
            let page_off = (cursor & ((HBM_PAGE_BYTES as u64) - 1)) as usize;
            let chunk_len = (dst.len() - offset).min(HBM_PAGE_BYTES - page_off);
            if let Some(page) = self.pages.get(&page_no) {
                for (d, &s) in dst
                    .iter_mut()
                    .skip(offset)
                    .take(chunk_len)
                    .zip(page.iter().skip(page_off))
                {
                    *d = s;
                }
            } else {
                for d in dst.iter_mut().skip(offset).take(chunk_len) {
                    *d = 0;
                }
            }
            cursor = cursor.saturating_add(chunk_len as u64);
            offset += chunk_len;
        }
    }
}

/// 32-byte binary trace record emitted by hardware into `HBM[0x00E0_0000]` (`HardwareTraceRingRecord`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HardwareTraceRingRecord {
    /// Hardware cycle timestamp when the bundle / DMA began execution.
    pub start_cycle: u64,
    /// Retired duration in cycles (`>= 1`).
    pub duration_cycles: u32,
    /// Retired bundle PC offset (`0, 32, 40, 52, 64, ...`).
    pub pc_offset: u32,
    /// Retired matrix + vector FLOPs in this interval.
    pub flops_retired: u64,
    /// HBM / VMEM / ICI bytes transferred in this interval.
    pub hbm_bytes_transferred: u32,
    /// Encoded hardware unit tag (`1 = TC0_VLIW`, `2 = SC_V2_V3`, `3 = HBM_DMA`, `4 = ICI_RDMA`).
    pub unit_tag: u32,
}

impl HardwareTraceRingRecord {
    /// Serializes this 32-byte record to little-endian wire bytes.
    #[must_use]
    pub const fn to_le_bytes(self) -> [u8; 32] {
        let start_b = self.start_cycle.to_le_bytes();
        let dur_b = self.duration_cycles.to_le_bytes();
        let pc_b = self.pc_offset.to_le_bytes();
        let flops_b = self.flops_retired.to_le_bytes();
        let bytes_b = self.hbm_bytes_transferred.to_le_bytes();
        let unit_b = self.unit_tag.to_le_bytes();
        [
            start_b[0], start_b[1], start_b[2], start_b[3], start_b[4], start_b[5], start_b[6],
            start_b[7], dur_b[0], dur_b[1], dur_b[2], dur_b[3], pc_b[0], pc_b[1], pc_b[2], pc_b[3],
            flops_b[0], flops_b[1], flops_b[2], flops_b[3], flops_b[4], flops_b[5], flops_b[6],
            flops_b[7], bytes_b[0], bytes_b[1], bytes_b[2], bytes_b[3], unit_b[0], unit_b[1],
            unit_b[2], unit_b[3],
        ]
    }

    /// Decodes a 32-byte little-endian slice into a [`HardwareTraceRingRecord`].
    ///
    /// # Arguments
    ///
    /// * `raw` - 32-byte trace ring entry.
    #[must_use]
    pub const fn from_le_bytes(raw: &[u8]) -> Option<Self> {
        let Some(
            &[
                b0,
                b1,
                b2,
                b3,
                b4,
                b5,
                b6,
                b7,
                b8,
                b9,
                b10,
                b11,
                b12,
                b13,
                b14,
                b15,
                b16,
                b17,
                b18,
                b19,
                b20,
                b21,
                b22,
                b23,
                b24,
                b25,
                b26,
                b27,
                b28,
                b29,
                b30,
                b31,
            ],
        ) = raw.first_chunk::<32>()
        else {
            return None;
        };
        Some(Self {
            start_cycle: u64::from_le_bytes([b0, b1, b2, b3, b4, b5, b6, b7]),
            duration_cycles: u32::from_le_bytes([b8, b9, b10, b11]),
            pc_offset: u32::from_le_bytes([b12, b13, b14, b15]),
            flops_retired: u64::from_le_bytes([b16, b17, b18, b19, b20, b21, b22, b23]),
            hbm_bytes_transferred: u32::from_le_bytes([b24, b25, b26, b27]),
            unit_tag: u32::from_le_bytes([b28, b29, b30, b31]),
        })
    }
}

/// Shared multi-chip 3D twisted-torus `ICI` RDMA switch fabric (`VirtualIciTorusFabric`).
#[derive(Debug, Default)]
pub struct VirtualIciTorusFabric {
    /// Registered per-device [`SparseHbmStore`] handles keyed by [`DeviceId`].
    chip_hbms: Mutex<BTreeMap<DeviceId, Arc<Mutex<SparseHbmStore>>>>,
    /// Collective mailbox buffers keyed by `(sync_flag_id, device_id)` for multi-chip `AllReduce` / `AllGather`.
    mailboxes: Mutex<BTreeMap<(u32, DeviceId), Vec<u8>>>,
}

impl VirtualIciTorusFabric {
    /// Creates a new [`VirtualIciTorusFabric`].
    #[must_use]
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            chip_hbms: Mutex::new(BTreeMap::new()),
            mailboxes: Mutex::new(BTreeMap::new()),
        })
    }

    /// Attaches a chip's [`SparseHbmStore`] to the `ICI` fabric under `device_id`.
    ///
    /// # Arguments
    ///
    /// * `device_id` - Logical [`DeviceId`].
    /// * `hbm` - Shared [`SparseHbmStore`].
    pub fn register_chip(&self, device_id: DeviceId, hbm: Arc<Mutex<SparseHbmStore>>) {
        self.chip_hbms.lock().insert(device_id, hbm);
    }

    /// Performs a zero-copy peer-to-peer `ICI` RDMA DMA transfer from `(src_dev, src_addr)` to `(dst_dev, dst_addr)`.
    ///
    /// # Arguments
    ///
    /// * `src_dev` - Source [`DeviceId`].
    /// * `src_addr` - Source physical HBM byte address.
    /// * `dst_dev` - Destination [`DeviceId`].
    /// * `dst_addr` - Destination physical HBM byte address.
    /// * `len` - Transfer byte length.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::DeviceNotFound`] if either `src_dev` or `dst_dev` is not attached to the fabric.
    pub fn rdma_copy(
        &self,
        src_dev: DeviceId,
        src_addr: u64,
        dst_dev: DeviceId,
        dst_addr: u64,
        len: usize,
    ) -> LibTpuResult<()> {
        let (src_hbm, dst_hbm) = {
            let guard = self.chip_hbms.lock();
            let s = guard
                .get(&src_dev)
                .cloned()
                .ok_or(LibTpuError::DeviceNotFound {
                    device_id: src_dev.0,
                })?;
            let d = guard
                .get(&dst_dev)
                .cloned()
                .ok_or(LibTpuError::DeviceNotFound {
                    device_id: dst_dev.0,
                })?;
            drop(guard);
            (s, d)
        };
        let payload = src_hbm.lock().read_bytes(src_addr, len);
        dst_hbm.lock().write_bytes(dst_addr, &payload);
        Ok(())
    }

    /// Posts `local_bytes` to collective `sync_flag_id` for `device_id` and reduces across all currently posted peers.
    ///
    /// # Arguments
    ///
    /// * `sync_flag_id` - Collective barrier `SyncFlag` ID.
    /// * `device_id` - Participating [`DeviceId`].
    /// * `collective_op` - Collective opcode (`14 = AllReduceSum`, `16 = AllReduceMax`, `17 = AllReduceMin`).
    /// * `elem_type` - Tensor [`ElementType`].
    /// * `local_bytes` - Local operand tensor bytes.
    #[must_use]
    pub fn post_and_reduce_collective(
        &self,
        sync_flag_id: u32,
        device_id: DeviceId,
        collective_op: u8,
        elem_type: ElementType,
        local_bytes: &[u8],
    ) -> Vec<u8> {
        let peers: Vec<Vec<u8>> = {
            let mut guard = self.mailboxes.lock();
            guard.insert((sync_flag_id, device_id), local_bytes.to_vec());
            guard
                .iter()
                .filter_map(|(&(sf, _), bytes)| {
                    if sf == sync_flag_id {
                        Some(bytes.clone())
                    } else {
                        None
                    }
                })
                .collect()
        };
        if peers.len() <= 1 {
            return local_bytes.to_vec();
        }
        let mut acc = elem_type.unpack_to_f64_vec(peers.first().map_or(&[], Vec::as_slice));
        for peer in peers.iter().skip(1) {
            let vals = elem_type.unpack_to_f64_vec(peer);
            for (d, &v) in acc.iter_mut().zip(vals.iter()) {
                *d = match collective_op {
                    16 => d.max(v),
                    17 => d.min(v),
                    _ => *d + v,
                };
            }
        }
        elem_type.pack_from_f64_slice(&acc)
    }
}

/// Summary metrics produced by [`ArchitecturalBitStreamEngine::execute_from_hbm`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BitstreamExecutionSummary {
    /// Total simulated hardware cycles elapsed.
    pub cycles_elapsed: u64,
    /// Number of generation-encoded `TensorCore` VLIW bundles decoded and retired.
    pub tc_bundles_retired: u32,
    /// Number of `SparseCore` V2/V3 instructions decoded and retired.
    pub sc_instructions_retired: u32,
    /// Final `TensorCore` 0 program counter byte offset.
    pub final_pc_bytes: u64,
    /// Highest hardware `SyncFlag` signaled during execution.
    pub final_sync_flag: u64,
    /// Total hardware trace records appended to `HBM[0x00E0_0000]`.
    pub trace_records_emitted: u32,
}

/// Cycle-accurate binary bitstream engine decoding real `jfc..iwc` and `SparseCore` V2/V3
/// instructions directly from virtual HBM (`ArchitecturalBitStreamEngine`).
#[derive(Debug, Default)]
pub struct ArchitecturalBitStreamEngine;

impl ArchitecturalBitStreamEngine {
    /// Fetches, decodes, and executes the relocated binary VLIW bitstream at `HBM[imem_iova]`
    /// (plus optional `.rodata` CMEM descriptors at `HBM[0x00A0_0000]` and `SparseCore` V2/V3
    /// stream at `HBM[0x00C0_0000]`) directly over `hbm`.
    ///
    /// # Arguments
    ///
    /// * `generation` - Target [`TpuGeneration`].
    /// * `device_id` - Executing [`DeviceId`].
    /// * `hbm` - Shared [`SparseHbmStore`].
    /// * `fabric` - Optional [`VirtualIciTorusFabric`] for cross-chip collectives.
    /// * `imem_iova` - Physical HBM base address of the binary VLIW bitstream (`0x0080_0000`).
    /// * `imem_byte_len` - Byte length of the binary VLIW bitstream.
    /// * `cmem_byte_len` - Optional byte length of the CMEM `.rodata` descriptor segment at `0x00A0_0000`.
    /// * `sc_imem_byte_len` - Optional byte length of the `SparseCore` V2/V3 bitstream at `0x00C0_0000`.
    /// * `inputs` - Slice of `(input_hbm_addr, input_byte_len)` pairs read from BAR2.
    /// * `outputs` - Slice of `(output_hbm_addr, output_byte_len)` pairs read from BAR2.
    /// * `base_cycle` - Starting hardware cycle counter.
    /// * `trace_cursor` - Current write index in the HBM trace ring (`0..1024`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if binary VLIW or `SparseCore` `HighwayHash64` / header decoding fails.
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        clippy::cast_possible_truncation
    )]
    pub fn execute_from_hbm(
        generation: TpuGeneration,
        device_id: DeviceId,
        hbm: &Arc<Mutex<SparseHbmStore>>,
        fabric: Option<&VirtualIciTorusFabric>,
        imem_iova: u64,
        imem_byte_len: usize,
        cmem_byte_len: usize,
        sc_imem_byte_len: usize,
        inputs: &[(u64, usize)],
        outputs: &[(u64, usize)],
        base_cycle: u64,
        trace_cursor: &mut u32,
    ) -> LibTpuResult<BitstreamExecutionSummary> {
        let bundle_width = generation_bundle_byte_width(generation);
        if imem_byte_len == 0 || (imem_byte_len % bundle_width) != 0 {
            return Err(LibTpuError::VliwIsaCodecError {
                generation_code: "imem_fetch",
                bundle_offset: 0,
                reason: format!(
                    "HBM IMEM byte length {imem_byte_len} is not a positive multiple of {bundle_width}"
                ),
            });
        }

        // 1. Fetch & decode real generation-encoded VLIW bundles from HBM IMEM
        let raw_imem = hbm.lock().read_bytes(imem_iova, imem_byte_len);
        let mut bundles: Vec<LloBundle> = Vec::with_capacity(imem_byte_len / bundle_width);
        for (idx, chunk) in raw_imem.chunks_exact(bundle_width).enumerate() {
            bundles.push(
                decode_bundle_for_generation(generation, chunk).map_err(|e| {
                    LibTpuError::VliwIsaCodecError {
                        generation_code: "imem_decode",
                        bundle_offset: idx * bundle_width,
                        reason: e.to_string(),
                    }
                })?,
            );
        }

        // 2. Stage input HBM slices into architectural VREG tiles (`v0, v1, ...`) & optional CMEM descriptors
        let mut core_vm = TpuCoreVmState::new();
        let mut cmem_output_slots: Vec<u8> = Vec::new();
        let mut rodata_prog_opt: Option<CompiledProgram> = None;
        let has_cmem_rodata = if cmem_byte_len > 0 {
            let cmem_blob = hbm.lock().read_bytes(HBM_CMEM_RODATA_BASE, cmem_byte_len);
            if let Ok(rodata_prog) = CompiledProgram::deserialize(&cmem_blob) {
                let default_et = rodata_prog
                    .output_shapes
                    .first()
                    .map_or(ElementType::F32, |s| s.element_type);
                for (b_idx, inst_bundle) in rodata_prog.instruction_bundles.iter().enumerate() {
                    let pc_key = u32::try_from(b_idx * 64).unwrap_or(0);
                    core_vm.imem_attr_table.insert(
                        pc_key,
                        ImemAttributeEntry {
                            bundle: inst_bundle.clone(),
                            elem_type: default_et,
                        },
                    );
                }
                for &slot in &rodata_prog.output_slots {
                    cmem_output_slots.push((slot & 255) as u8);
                }
                rodata_prog_opt = Some(rodata_prog);
                true
            } else {
                false
            }
        } else {
            false
        };

        let mut hbm_input_vecs: Vec<Vec<u8>> = Vec::with_capacity(inputs.len());
        for (slot_idx, &(addr, len)) in inputs.iter().enumerate() {
            if addr != 0 && len > 0 {
                let bytes = hbm.lock().read_bytes(addr, len);
                hbm_input_vecs.push(bytes.clone());
                let v_idx = (slot_idx & 255) as u8;
                core_vm.vreg_tiles.insert(v_idx, bytes.clone());
                core_vm
                    .vmem_allocations
                    .insert((slot_idx as u64) * 256, bytes);
            } else {
                hbm_input_vecs.push(Vec::new());
            }
        }

        // 3. Step every decoded LloBundle on the architectural functional units (S0/S1, V0/V1, MXU0..3, XLU0..1, DMA)
        let mut cur_cycle = base_cycle;
        let mut max_sync_flag = 1_u64;
        let mut trace_emitted = 0_u32;
        let mut last_written_dst_vreg = 0_u8;

        for (b_idx, bundle) in bundles.iter().enumerate() {
            let step_start = cur_cycle;
            let stall = u64::from(bundle.stall_cycles);
            let [m0, m1, ..] = &bundle.matrix_slots;
            let [v0, v1] = &bundle.vector_slots;
            let [x0, _x1] = &bundle.xlu_slots;
            let mxu_active = (m0.opcode != 0) || (m1.opcode != 0);
            let vec_active = (v0.opcode != 0) || (v1.opcode != 0);
            let unit_cycles: u64 = if mxu_active {
                8
            } else if vec_active {
                2
            } else {
                1
            };
            cur_cycle = cur_cycle.saturating_add(unit_cycles + stall);

            let pc_key = u32::try_from(b_idx * 64).unwrap_or(0);
            core_vm.pc = pc_key;
            core_vm.step_vliw_bundle(bundle);
            if let Some(attr_entry) = core_vm.imem_attr_table.get(&pc_key) {
                last_written_dst_vreg = (attr_entry.bundle.output_slot & 255) as u8;
            } else if m0.opcode != 0 {
                last_written_dst_vreg = (m0.acc_vreg & 255) as u8;
            } else if v0.opcode != 0 {
                last_written_dst_vreg = (v0.dst_vreg & 255) as u8;
            } else if x0.opcode != 0 {
                last_written_dst_vreg = (x0.dst_vreg & 255) as u8;
            }

            if bundle.dma_slot.sync_flag_id != 0 {
                max_sync_flag = max_sync_flag.max(u64::from(bundle.dma_slot.sync_flag_id));
            }

            // Collective RDMA over 3D torus ICI fabric when `collective_op >= 14`
            if !has_cmem_rodata
                && bundle.dma_slot.collective_op >= 14
                && let Some(ici) = fabric
            {
                let empty_tile: &[u8] = &[];
                let cur_tile = core_vm
                    .vreg_tiles
                    .get(&last_written_dst_vreg)
                    .map_or(empty_tile, Vec::as_slice);
                let et =
                    ElementType::from_raw(i32::from(v0.element_type)).unwrap_or(ElementType::F32);
                let reduced = ici.post_and_reduce_collective(
                    bundle.dma_slot.sync_flag_id.max(1),
                    device_id,
                    bundle.dma_slot.collective_op,
                    et,
                    cur_tile,
                );
                core_vm.vreg_tiles.insert(last_written_dst_vreg, reduced);
            }

            let mxu_dim: u64 = if matches!(generation, TpuGeneration::V6e | TpuGeneration::V7x) {
                256
            } else {
                128
            };
            let flops = if mxu_active {
                2 * mxu_dim * mxu_dim * 8
            } else if vec_active {
                8 * 128
            } else {
                0
            };
            let rec = HardwareTraceRingRecord {
                start_cycle: step_start,
                duration_cycles: u32::try_from(cur_cycle - step_start).unwrap_or(1),
                pc_offset: u32::try_from(b_idx * bundle_width).unwrap_or(0),
                flops_retired: flops,
                hbm_bytes_transferred: if bundle.dma_slot.dma_opcode != 0 {
                    4096
                } else {
                    0
                },
                unit_tag: 1,
            };
            let ring_off = HBM_XPLANE_TRACE_RING_BASE + u64::from(*trace_cursor & 1023) * 32;
            hbm.lock().write_bytes(ring_off, &rec.to_le_bytes());
            *trace_cursor = trace_cursor.wrapping_add(1);
            trace_emitted += 1;
        }

        // 4. Fetch & decode optional SparseCore V2 (16 B) or V3 (24 B) binary stream from `HBM[0x00C0_0000]`
        let mut sc_retired = 0_u32;
        if sc_imem_byte_len > 0 {
            let raw_sc = hbm
                .lock()
                .read_bytes(HBM_IMEM_SPARSECORE_BASE, sc_imem_byte_len);
            let is_v3 = matches!(generation, TpuGeneration::V6e | TpuGeneration::V7x)
                && (sc_imem_byte_len % SparseCoreV3Codec::BUNDLE_BYTES) == 0
                && raw_sc.first().copied() == Some(SparseCoreV3Codec::MAGIC);
            if is_v3 {
                for (idx, chunk) in raw_sc
                    .chunks_exact(SparseCoreV3Codec::BUNDLE_BYTES)
                    .enumerate()
                {
                    let inst =
                        SparseCoreV3Codec::decode(chunk, u32::try_from(idx * 24).unwrap_or(0))?;
                    let step_start = cur_cycle;
                    cur_cycle = cur_cycle.saturating_add(4);
                    max_sync_flag = max_sync_flag.max(u64::from(inst.sync_flag_id));
                    if !has_cmem_rodata
                        && inst.op == SparseCoreOp::SpmemGatherDma
                        && let (Some(&(tbl_addr, tbl_len)), Some(&(idx_addr, idx_len))) =
                            (inputs.first(), inputs.get(1))
                    {
                        let (tbl_bytes, idx_bytes) = {
                            let g = hbm.lock();
                            (
                                g.read_bytes(tbl_addr, tbl_len),
                                g.read_bytes(idx_addr, idx_len),
                            )
                        };
                        let tbl_f = ElementType::F32.unpack_to_f64_vec(&tbl_bytes);
                        let indices = ElementType::S32.unpack_to_i64_vec(&idx_bytes);
                        if !tbl_f.is_empty() && !indices.is_empty() {
                            let row_width = (tbl_f.len() / indices.len().max(1)).max(1);
                            let mut gathered = Vec::with_capacity(indices.len() * row_width);
                            let num_rows = (tbl_f.len() / row_width).max(1);
                            for &raw_row in &indices {
                                let r = usize::try_from(raw_row.max(0)).unwrap_or(0) % num_rows;
                                for c in 0..row_width {
                                    gathered
                                        .push(tbl_f.get(r * row_width + c).copied().unwrap_or(0.0));
                                }
                            }
                            core_vm.vreg_tiles.insert(
                                last_written_dst_vreg,
                                ElementType::F32.pack_from_f64_slice(&gathered),
                            );
                        }
                    }
                    let rec = HardwareTraceRingRecord {
                        start_cycle: step_start,
                        duration_cycles: 4,
                        pc_offset: u32::try_from(idx * 24).unwrap_or(0),
                        flops_retired: u64::from(inst.vector_lanes) * 4,
                        hbm_bytes_transferred: 256,
                        unit_tag: 2,
                    };
                    let ring_off =
                        HBM_XPLANE_TRACE_RING_BASE + u64::from(*trace_cursor & 1023) * 32;
                    hbm.lock().write_bytes(ring_off, &rec.to_le_bytes());
                    *trace_cursor = trace_cursor.wrapping_add(1);
                    trace_emitted += 1;
                    sc_retired += 1;
                }
            } else if (sc_imem_byte_len % SparseCoreV2Codec::BUNDLE_BYTES) == 0 {
                for (idx, chunk) in raw_sc
                    .chunks_exact(SparseCoreV2Codec::BUNDLE_BYTES)
                    .enumerate()
                {
                    let inst =
                        SparseCoreV2Codec::decode(chunk, u32::try_from(idx * 16).unwrap_or(0))?;
                    let step_start = cur_cycle;
                    cur_cycle = cur_cycle.saturating_add(4);
                    max_sync_flag = max_sync_flag.max(u64::from(inst.sync_flag_id));
                    let rec = HardwareTraceRingRecord {
                        start_cycle: step_start,
                        duration_cycles: 4,
                        pc_offset: u32::try_from(idx * 16).unwrap_or(0),
                        flops_retired: 256,
                        hbm_bytes_transferred: 128,
                        unit_tag: 2,
                    };
                    let ring_off =
                        HBM_XPLANE_TRACE_RING_BASE + u64::from(*trace_cursor & 1023) * 32;
                    hbm.lock().write_bytes(ring_off, &rec.to_le_bytes());
                    *trace_cursor = trace_cursor.wrapping_add(1);
                    trace_emitted += 1;
                    sc_retired += 1;
                }
            } else {
                return Err(LibTpuError::SparseCoreCodecError {
                    core_id: 0,
                    pc_offset: 0,
                    reason: format!("SparseCore IMEM length {sc_imem_byte_len} misaligned"),
                });
            }
        }

        // 5. Commit output VREG tiles (resolving structured `While` / `Conditional` sub-computations when present) back to output HBM buffers
        if let Some(ref rodata_prog) = rodata_prog_opt
            && rodata_prog.instruction_bundles.iter().any(|b| {
                matches!(
                    b.opcode,
                    crate::compiler::TpuOpCode::While | crate::compiler::TpuOpCode::Conditional
                )
            })
        {
            let structured_outs =
                core_vm.execute_compiled_program(rodata_prog, hbm_input_vecs, outputs.len());
            for (out_idx, out_bytes) in structured_outs.into_iter().enumerate() {
                let candidate_slot = cmem_output_slots
                    .get(out_idx)
                    .copied()
                    .unwrap_or(((inputs.len() + out_idx) & 255) as u8);
                core_vm.vreg_tiles.insert(candidate_slot, out_bytes);
            }
        }
        {
            let mut hbm_guard = hbm.lock();
            for (out_idx, &(out_addr, out_len)) in outputs.iter().enumerate() {
                if out_addr == 0 || out_len == 0 {
                    continue;
                }
                let candidate_slot = cmem_output_slots
                    .get(out_idx)
                    .copied()
                    .unwrap_or(((inputs.len() + out_idx) & 255) as u8);
                let empty_out: &[u8] = &[];
                let tile_bytes = core_vm
                    .vreg_tiles
                    .get(&candidate_slot)
                    .or_else(|| core_vm.vreg_tiles.get(&last_written_dst_vreg))
                    .map_or(empty_out, Vec::as_slice);
                let mut padded = vec![0_u8; out_len];
                for (d, &s) in padded.iter_mut().zip(tile_bytes.iter()) {
                    *d = s;
                }
                hbm_guard.write_bytes(out_addr, &padded);
            }
        }

        Ok(BitstreamExecutionSummary {
            cycles_elapsed: cur_cycle - base_cycle,
            tc_bundles_retired: u32::try_from(bundles.len()).unwrap_or(0),
            sc_instructions_retired: sc_retired,
            final_pc_bytes: imem_byte_len as u64,
            final_sync_flag: max_sync_flag,
            trace_records_emitted: trace_emitted,
        })
    }
}

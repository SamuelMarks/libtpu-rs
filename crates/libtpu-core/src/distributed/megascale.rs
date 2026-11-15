//! `MegaScale` Multi-Slice Data-Center Network (`DCN`) Runtime (`Phase 8.3`).
//!
//! Implements:
//! - [`MegascaleCongestionController`]: RTT-gradient and ECN-marking congestion window (`cwnd_bytes`)
//!   and pacing rate (`pacing_rate_gbps`) state machine over [`crate::proto::MegascaleCongestionControlConfigProto`].
//! - [`MegaScaleMultiSocketTransport`]: Asynchronous multi-socket TCP/DCN striped collective transport
//!   exchanging wire-framed [`crate::proto::MegascaleSendRequestProto`] (`MegascaleTransferBufferRequestProto` +
//!   FNV/Highway64 checksum + optional Brotli payload compression) and [`crate::proto::MegascaleTransferBufferResponseProto`].
//! - [`HierarchicalDcnIciRuntime`]: Two-level `DCN` + `ICI` collective schedules (`AllReduce`,
//!   `ReduceScatter`, `AllGather`, `AllToAll`, `RaggedAllToAll`) combining intra-slice
//!   [`IciCollectiveEngine`] with inter-slice multi-socket DCN striping.
//! - [`MegaScaleFullActionGraphRunner`]: Complete dataflow executor for all 16 [`crate::proto::mega_scale_action::Action`]
//!   variants in `MegaScaleActionGraph`.
//! - [`RapidEyeTelemetryBuilder`]: Structured [`crate::proto::MegaScaleRuntimeErrorProto`], [`crate::proto::RapidEyeInfoProto`], and
//!   [`crate::proto::RapidEyeErrorDigestProto`] root-cause telemetry synthesis.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::time::Instant;

use parking_lot::Mutex;

use crate::distributed::ici::{IciCollectiveAlgorithm, IciCollectiveEngine};
use crate::error::{LibTpuError, LibTpuResult};
use crate::proto::{
    ExecutorTelemetryProto, MegaScaleActionGraphProto, MegaScaleRuntimeErrorProto,
    MegascaleChannelSliceDescriptorProto, MegascaleCollectiveScheduleProto,
    MegascaleCongestionControlConfigProto, MegascaleCongestionWindowStateProto,
    MegascaleRequestIdProto, MegascaleSendRequestProto, MegascaleTransferBufferRequestProto,
    MegascaleTransferBufferResponseProto, PendingGraphStateProto, ProtoSchemaMessage,
    RapidEyeErrorDigestProto, RapidEyeInfoProto, RuntimeStateSummaryProto, TpuCoreHloProto,
    mega_scale_action, mega_scale_runtime_error, rapid_eye_error_digest_proto,
};
use crate::types::{ChipCoordinates, CollectiveReductionOp, ElementType, TpuGeneration};

/// Deterministic 64-bit integrity digest for [`MegascaleTransferBufferRequestProto::checksum`].
///
/// # Arguments
///
/// * `bytes` - Raw payload bytes to hash.
#[must_use]
pub fn compute_megascale_payload_checksum(bytes: &[u8]) -> u64 {
    let mut hash = 0xCBF2_9CE4_8422_2325_u64;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0100_0000_01B3);
    }
    hash
}

/// RTT-gradient and ECN-feedback congestion window and packet-pacing controller
/// (`MegascaleCongestionController`) operating on [`MegascaleCongestionControlConfigProto`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MegascaleCongestionController {
    /// Underlying protobuf configuration and per-channel telemetry table.
    pub config: MegascaleCongestionControlConfigProto,
    /// Target baseline fabric RTT in nanoseconds (`target_rtt_ns`).
    pub target_rtt_ns: u64,
}

impl Default for MegascaleCongestionController {
    fn default() -> Self {
        Self::new(
            MegascaleCongestionControlConfigProto {
                algorithm_name: String::from("BBRv2"),
                initial_cwnd_bytes: 262_144,
                min_cwnd_bytes: 16_384,
                max_cwnd_bytes: 16_777_216,
                active_windows: Vec::new(),
            },
            4,
            50_000,
        )
    }
}

impl MegascaleCongestionController {
    /// Creates a new [`MegascaleCongestionController`] with `num_channels` initialized windows.
    ///
    /// # Arguments
    ///
    /// * `config` - [`MegascaleCongestionControlConfigProto`] bounds and algorithm name.
    /// * `num_channels` - Number of parallel DCN socket channels to initialize.
    /// * `target_rtt_ns` - Target baseline DCN fabric RTT in nanoseconds.
    #[must_use]
    pub fn new(
        mut config: MegascaleCongestionControlConfigProto,
        num_channels: u32,
        target_rtt_ns: u64,
    ) -> Self {
        let min_w = config.min_cwnd_bytes.max(4096);
        let max_w = config.max_cwnd_bytes.max(min_w);
        let init_w = config.initial_cwnd_bytes.clamp(min_w, max_w);
        config.min_cwnd_bytes = min_w;
        config.max_cwnd_bytes = max_w;
        config.initial_cwnd_bytes = init_w;
        let base_rtt = target_rtt_ns.max(1_000);
        if config.active_windows.is_empty() {
            let count = num_channels.max(1);
            let raw_gbps = init_w.saturating_mul(8).div_ceil(base_rtt).max(1);
            let gbps = u32::try_from(raw_gbps).unwrap_or(u32::MAX);
            for ch in 0..count {
                config
                    .active_windows
                    .push(MegascaleCongestionWindowStateProto {
                        channel_id: ch,
                        cwnd_bytes: init_w,
                        rtt_nanoseconds: base_rtt,
                        ecn_marked_packets: 0,
                        pacing_rate_gbps: gbps,
                    });
            }
        }
        Self {
            config,
            target_rtt_ns: base_rtt,
        }
    }

    /// Records an observed acknowledgment sample `(acked_bytes, sample_rtt_ns, ecn_ce_marked)` on
    /// `channel_id`, updating `cwnd_bytes`, smoothed `rtt_nanoseconds`, `ecn_marked_packets`, and
    /// `pacing_rate_gbps` according to `BBRv2` / `TIMELY` / `Swift` windowing rules.
    ///
    /// # Arguments
    ///
    /// * `channel_id` - DCN channel index.
    /// * `acked_bytes` - Newly acknowledged payload byte count.
    /// * `sample_rtt_ns` - Measured round-trip latency in nanoseconds.
    /// * `ecn_ce_marked` - Whether the fabric switch set the ECN Congestion Experienced bit.
    pub fn on_ack_sample(
        &mut self,
        channel_id: u32,
        acked_bytes: u64,
        sample_rtt_ns: u64,
        ecn_ce_marked: bool,
    ) -> MegascaleCongestionWindowStateProto {
        let min_w = self.config.min_cwnd_bytes;
        let max_w = self.config.max_cwnd_bytes;
        let target_rtt = self.target_rtt_ns;
        let is_timely = self.config.algorithm_name == "TIMELY";

        if !self
            .config
            .active_windows
            .iter()
            .any(|w| w.channel_id == channel_id)
        {
            self.config
                .active_windows
                .push(MegascaleCongestionWindowStateProto {
                    channel_id,
                    cwnd_bytes: self.config.initial_cwnd_bytes,
                    rtt_nanoseconds: target_rtt,
                    ecn_marked_packets: 0,
                    pacing_rate_gbps: 10,
                });
        }

        let mut updated = MegascaleCongestionWindowStateProto::default();
        for win in &mut self.config.active_windows {
            if win.channel_id == channel_id {
                let clamped_rtt = sample_rtt_ns.max(1_000);
                win.rtt_nanoseconds = ((win.rtt_nanoseconds * 7) + clamped_rtt) / 8;
                if ecn_ce_marked {
                    win.ecn_marked_packets = win.ecn_marked_packets.saturating_add(1);
                    win.cwnd_bytes = ((win.cwnd_bytes * 3) / 4).max(min_w);
                } else if clamped_rtt > target_rtt.saturating_mul(2) {
                    let dec = if is_timely {
                        win.cwnd_bytes / 8
                    } else {
                        win.cwnd_bytes / 4
                    };
                    win.cwnd_bytes = win.cwnd_bytes.saturating_sub(dec).max(min_w);
                } else {
                    let gain = (acked_bytes / 2).max(4096);
                    win.cwnd_bytes = win.cwnd_bytes.saturating_add(gain).min(max_w);
                }
                let raw_gbps = win
                    .cwnd_bytes
                    .saturating_mul(8)
                    .div_ceil(win.rtt_nanoseconds.max(1))
                    .max(1);
                win.pacing_rate_gbps = u32::try_from(raw_gbps).unwrap_or(u32::MAX);
                updated = *win;
                break;
            }
        }
        updated
    }

    /// Computes the inter-packet pacing delay in nanoseconds required to transmit `packet_bytes`
    /// on `channel_id` without exceeding its active `pacing_rate_gbps`.
    ///
    /// # Arguments
    ///
    /// * `channel_id` - DCN channel index.
    /// * `packet_bytes` - Outgoing frame byte size.
    #[must_use]
    pub fn compute_pacing_delay_ns(&self, channel_id: u32, packet_bytes: usize) -> u64 {
        let rate_gbps = self
            .config
            .active_windows
            .iter()
            .find(|w| w.channel_id == channel_id)
            .map_or(100_u32, |w| w.pacing_rate_gbps.max(1));
        let bits = (packet_bytes as u64).saturating_mul(8);
        bits.div_ceil(u64::from(rate_gbps))
    }
}

/// Asynchronous multi-socket TCP/DCN collective transport (`MegaScaleMultiSocketTransport`).
///
/// Stripes `MegascaleSendRequestProto` / `MegascaleTransferBufferResponseProto` frames across
/// `num_sockets` parallel OS TCP channels and feeds RTT samples into [`MegascaleCongestionController`].
#[derive(Debug, Clone)]
pub struct MegaScaleMultiSocketTransport {
    /// Number of parallel TCP socket stripes per collective exchange (`>= 1`).
    pub num_sockets: usize,
    /// Whether to Brotli-compress DCN transfer payloads.
    pub enable_compression: bool,
    /// Synchronized congestion controller tracking per-socket `cwnd` and RTT.
    pub congestion: Arc<Mutex<MegascaleCongestionController>>,
    /// Keyed mailbox of transferred DCN buffers (`transfer_key -> bytes`).
    pub mailbox: Arc<Mutex<BTreeMap<String, Vec<u8>>>>,
    /// Optional deterministic wire-bitflip fault flag for resilience verification.
    pub inject_checksum_fault: bool,
}

impl MegaScaleMultiSocketTransport {
    /// Creates a new [`MegaScaleMultiSocketTransport`] with `num_sockets` parallel TCP stripes.
    ///
    /// # Arguments
    ///
    /// * `num_sockets` - Parallel TCP socket count (`1..=16`).
    /// * `enable_compression` - Enables Brotli payload compression on the wire.
    /// * `congestion` - Initial [`MegascaleCongestionController`].
    #[must_use]
    pub fn new(
        num_sockets: usize,
        enable_compression: bool,
        congestion: MegascaleCongestionController,
    ) -> Self {
        Self {
            num_sockets: num_sockets.clamp(1, 16),
            enable_compression,
            congestion: Arc::new(Mutex::new(congestion)),
            mailbox: Arc::new(Mutex::new(BTreeMap::new())),
            inject_checksum_fault: false,
        }
    }

    /// Configures deterministic wire checksum bitflip injection for fault-path testing.
    ///
    /// # Arguments
    ///
    /// * `inject` - When `true`, corrupts outgoing frame checksums by `1` bit.
    #[must_use]
    pub const fn with_checksum_fault_injection(mut self, inject: bool) -> Self {
        self.inject_checksum_fault = inject;
        self
    }

    /// Writes a 4-byte length-prefixed Protobuf frame onto `stream`.
    ///
    /// # Arguments
    ///
    /// * `stream` - Connected TCP stream.
    /// * `bytes` - Serialized Protobuf message bytes.
    fn write_len_prefixed(stream: &mut TcpStream, bytes: &[u8]) -> LibTpuResult<()> {
        let len = u32::try_from(bytes.len()).unwrap_or(0);
        stream.write_all(&len.to_be_bytes())?;
        stream.write_all(bytes)?;
        stream.flush()?;
        Ok(())
    }

    /// Reads a 4-byte length-prefixed Protobuf frame from `stream`.
    ///
    /// # Arguments
    ///
    /// * `stream` - Connected TCP stream.
    fn read_len_prefixed(stream: &mut TcpStream) -> LibTpuResult<Vec<u8>> {
        let mut hdr = [0_u8; 4];
        stream.read_exact(&mut hdr)?;
        let len = u32::from_be_bytes(hdr) as usize;
        let mut buf = vec![0_u8; len];
        stream.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Compresses `raw` with Brotli when `compress` is `true`.
    ///
    /// # Arguments
    ///
    /// * `raw` - Uncompressed payload bytes.
    /// * `compress` - Whether to apply Brotli compression.
    fn encode_wire_payload(raw: &[u8], compress: bool) -> Vec<u8> {
        if !compress {
            return raw.to_vec();
        }
        let mut out = Vec::new();
        {
            let mut w = brotli::CompressorWriter::new(&mut out, 4096, 4, 20);
            let _ = w.write_all(raw);
        }
        out
    }

    /// Decompresses `wire` with Brotli when `compressed` is `true`.
    ///
    /// # Arguments
    ///
    /// * `wire` - Wire payload bytes.
    /// * `compressed` - Whether `wire` is Brotli-compressed.
    fn decode_wire_payload(wire: &[u8], compressed: bool) -> LibTpuResult<Vec<u8>> {
        if !compressed {
            return Ok(wire.to_vec());
        }
        let mut out = Vec::new();
        let mut dec = brotli::Decompressor::new(wire, 4096);
        dec.read_to_end(&mut out)
            .map_err(|e| LibTpuError::DeserializationFailed {
                reason: format!("MegaScale DCN Brotli decompression failed: {e}"),
            })?;
        Ok(out)
    }

    /// Accepts a single striped TCP connection on `listener`, verifies its checksum, and replies with `TransferBufferResponse`.
    ///
    /// # Arguments
    ///
    /// * `listener` - Bound TCP stripe listener.
    pub(crate) fn accept_and_verify_stripe(listener: &TcpListener) -> LibTpuResult<Vec<u8>> {
        let (mut sock, _) = listener.accept()?;
        let raw_req = Self::read_len_prefixed(&mut sock)?;
        let send_req = MegascaleSendRequestProto::decode_proto(&raw_req)?;
        let Some(hdr) = send_req.request_header else {
            return Err(LibTpuError::DeserializationFailed {
                reason: String::from("SendRequest missing TransferBufferRequest"),
            });
        };
        let decoded = Self::decode_wire_payload(&send_req.payload, hdr.is_compressed)?;
        if hdr.has_checksum {
            let actual = compute_megascale_payload_checksum(&decoded);
            if actual != hdr.checksum {
                return Err(LibTpuError::MegascaleRuntimeFault {
                    slice_id: hdr.slice_id,
                    task_id: hdr.tpu_host_id,
                    error_type: 2,
                    hang_error_type: 0,
                    message: format!(
                        "DCN checksum mismatch: expected 0x{:016x}, got 0x{:016x}",
                        hdr.checksum, actual
                    ),
                });
            }
        }
        let resp = MegascaleTransferBufferResponseProto {
            received_time_us: hdr.send_time_us.saturating_add(25),
        };
        Self::write_len_prefixed(&mut sock, &resp.encode_proto())?;
        Ok(decoded)
    }

    /// Transfers `payload` under `transfer_key` across `self.num_sockets` concurrent TCP listeners.
    ///
    /// Verifies `MegascaleSendRequestProto` checksums, stores the reassembled buffer in `self.mailbox`,
    /// and records per-channel RTT samples.
    ///
    /// # Arguments
    ///
    /// * `transfer_key` - Unique collective/action transfer key.
    /// * `src_slice` - Source slice index.
    /// * `src_host` - Source TPU host index.
    /// * `payload` - Raw byte buffer to stripe across `self.num_sockets` TCP sockets.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if socket I/O, Protobuf decoding, decompression, or checksum verification fails.
    pub fn striped_send_receive(
        &self,
        transfer_key: &str,
        src_slice: i32,
        src_host: i32,
        payload: &[u8],
    ) -> LibTpuResult<Vec<u8>> {
        let stripes = self.num_sockets.max(1);
        let chunk_size = payload.len().div_ceil(stripes).max(1);
        let mut listeners = Vec::with_capacity(stripes);
        let mut addrs: Vec<SocketAddr> = Vec::with_capacity(stripes);
        for _ in 0..stripes {
            let l = TcpListener::bind("127.0.0.1:0")?;
            addrs.push(l.local_addr()?);
            listeners.push(l);
        }

        let compress = self.enable_compression;
        let corrupt_crc = self.inject_checksum_fault;
        let recv_chunks: Mutex<BTreeMap<usize, Vec<u8>>> = Mutex::new(BTreeMap::new());
        let first_err: Mutex<Option<LibTpuError>> = Mutex::new(None);

        std::thread::scope(|s| {
            let recv_ref = &recv_chunks;
            let err_ref = &first_err;

            for (idx, listener) in listeners.into_iter().enumerate() {
                s.spawn(move || match Self::accept_and_verify_stripe(&listener) {
                    Ok(bytes) => {
                        recv_ref.lock().insert(idx, bytes);
                    }
                    Err(e) => {
                        *err_ref.lock() = Some(e);
                    }
                });
            }

            for (idx, &addr) in addrs.iter().enumerate() {
                let start = (idx * chunk_size).min(payload.len());
                let end = (start + chunk_size).min(payload.len());
                let slice_bytes = payload.get(start..end).unwrap_or(&[]);
                let key_stripe = format!("{transfer_key}/stripe_{idx}");
                let cong_arc = Arc::clone(&self.congestion);

                s.spawn(move || {
                    let t0 = Instant::now();
                    let res = (|| -> LibTpuResult<()> {
                        let wire_body = Self::encode_wire_payload(slice_bytes, compress);
                        let req = MegascaleSendRequestProto {
                            request_header: Some(MegascaleTransferBufferRequestProto {
                                key: key_stripe,
                                payload_size: wire_body.len() as u64,
                                numa_node: Some(0),
                                send_time_us: 1_000,
                                activation_time_us: 1_000,
                                request_id: Some(MegascaleRequestIdProto {
                                    process_id: u64::try_from(src_slice.max(0)).unwrap_or(0),
                                    seq: idx as u64,
                                }),
                                slice_id: src_slice,
                                tpu_host_id: src_host,
                                is_compressed: compress,
                                has_checksum: true,
                                checksum: compute_megascale_payload_checksum(slice_bytes)
                                    ^ u64::from(corrupt_crc),
                                output_offset: Some(start as u64),
                                output_size: Some(slice_bytes.len() as u64),
                                incarnation_id: Some(1),
                            }),
                            payload: wire_body,
                        };
                        let encoded = req.encode_proto();
                        let ch_id = u32::try_from(idx).unwrap_or(0);
                        let _pacing_ns = cong_arc
                            .lock()
                            .compute_pacing_delay_ns(ch_id, encoded.len());
                        let mut stream = TcpStream::connect(addr)?;
                        Self::write_len_prefixed(&mut stream, &encoded)?;
                        let raw_ack = Self::read_len_prefixed(&mut stream)?;
                        let _ack = MegascaleTransferBufferResponseProto::decode_proto(&raw_ack)?;
                        let rtt_ns = u64::try_from(t0.elapsed().as_nanos()).unwrap_or(10_000);
                        let _ = cong_arc.lock().on_ack_sample(
                            ch_id,
                            slice_bytes.len() as u64,
                            rtt_ns,
                            false,
                        );
                        Ok(())
                    })();
                    if let Err(e) = res {
                        *err_ref.lock() = Some(e);
                    }
                });
            }
        });

        if let Some(err) = first_err.into_inner() {
            return Err(err);
        }

        let ordered = recv_chunks.into_inner();
        let mut reassembled = Vec::with_capacity(payload.len());
        for bytes in ordered.into_values() {
            reassembled.extend_from_slice(&bytes);
        }
        self.mailbox
            .lock()
            .insert(String::from(transfer_key), reassembled.clone());
        Ok(reassembled)
    }
}

/// Two-level hierarchical `DCN` + `ICI` collective execution engine (`HierarchicalDcnIciRuntime`).
#[derive(Debug, Clone, Copy, Default)]
pub struct HierarchicalDcnIciRuntime;

impl HierarchicalDcnIciRuntime {
    /// Executes a 3-stage hierarchical `AllReduce` across `slices_of_chip_buffers`.
    ///
    /// Performs:
    /// 1. Intra-slice 2D/3D `ICI` `ReduceScatter` within each slice (`[num_chips_per_slice]` rails),
    /// 2. Inter-slice multi-socket `DCN` `AllReduce` per rail across all slices over `transport`,
    /// 3. Intra-slice 2D/3D `ICI` `AllGather` reconstructing full buffers on every chip.
    ///
    /// # Arguments
    ///
    /// * `transport` - Multi-socket DCN transport.
    /// * `slice_mesh` - Intra-slice 2D/3D chip bounds.
    /// * `slices_of_chip_buffers` - `[num_slices][chips_per_slice]` input byte buffers.
    /// * `elem_type` - Tensor [`ElementType`].
    /// * `reduction_op` - [`CollectiveReductionOp`] combiner.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if slice shapes are empty/inconsistent or DCN transport fails.
    pub fn execute_hierarchical_all_reduce(
        transport: &MegaScaleMultiSocketTransport,
        slice_mesh: ChipCoordinates,
        slices_of_chip_buffers: &[Vec<Vec<u8>>],
        elem_type: ElementType,
        reduction_op: CollectiveReductionOp,
    ) -> LibTpuResult<(Vec<Vec<Vec<u8>>>, MegascaleCollectiveScheduleProto)> {
        let Some(first_slice) = slices_of_chip_buffers.first() else {
            return Err(LibTpuError::InvalidArgument {
                parameter: "slices_of_chip_buffers",
                reason: String::from("Hierarchical AllReduce requires at least one slice"),
            });
        };
        let chips_per_slice = first_slice.len();
        if chips_per_slice == 0
            || slices_of_chip_buffers
                .iter()
                .any(|s| s.len() != chips_per_slice)
        {
            return Err(LibTpuError::InvalidArgument {
                parameter: "slices_of_chip_buffers",
                reason: String::from("All slices must have the same non-zero chip count"),
            });
        }

        // Stage 1: Intra-slice ICI ReduceScatter
        let mut scattered_per_slice = Vec::with_capacity(slices_of_chip_buffers.len());
        for slice_bufs in slices_of_chip_buffers {
            let (shards, _) = IciCollectiveEngine::execute_reduce_scatter_scheduled(
                IciCollectiveAlgorithm::MeshTorus2d3d,
                slice_mesh,
                slice_bufs,
                elem_type,
                reduction_op,
            );
            scattered_per_slice.push(shards);
        }

        // Stage 2: Inter-slice multi-socket DCN AllReduce per rail
        let num_slices = slices_of_chip_buffers.len();
        let mut dcn_reduced_rails: Vec<Vec<u8>> = Vec::with_capacity(chips_per_slice);
        let mut descriptors = Vec::new();
        for rail in 0..chips_per_slice {
            let mut rail_acc: Option<Vec<u8>> = None;
            for (s_idx, slice_shards) in scattered_per_slice.iter().enumerate() {
                let shard = slice_shards.get(rail).map_or(&[][..], Vec::as_slice);
                let key = format!("hier_ar/rail_{rail}/slice_{s_idx}");
                let wire_rx = transport.striped_send_receive(
                    &key,
                    i32::try_from(s_idx).unwrap_or(0),
                    i32::try_from(rail).unwrap_or(0),
                    shard,
                )?;
                rail_acc = Some(match rail_acc {
                    None => wire_rx,
                    Some(prev) => IciCollectiveEngine::reduce_pair_typed(
                        &prev,
                        &wire_rx,
                        elem_type,
                        reduction_op,
                    ),
                });
                descriptors.push(MegascaleChannelSliceDescriptorProto {
                    channel_id: u32::try_from(rail).unwrap_or(0),
                    src_slice_id: i32::try_from(s_idx).unwrap_or(0),
                    dst_slice_id: i32::try_from((s_idx + 1) % num_slices).unwrap_or(0),
                    payload_bytes: shard.len() as u64,
                    ici_ring_stage: 1,
                });
            }
            dcn_reduced_rails.push(rail_acc.unwrap_or_default());
        }

        // Stage 3: Intra-slice ICI AllGather
        let (gathered_slice, _) = IciCollectiveEngine::execute_all_gather_scheduled(
            IciCollectiveAlgorithm::MeshTorus2d3d,
            slice_mesh,
            &dcn_reduced_rails,
            elem_type,
        );
        let out_slices = vec![gathered_slice; num_slices];
        let red_code = match reduction_op {
            CollectiveReductionOp::Sum => 0,
            CollectiveReductionOp::Product => 1,
            CollectiveReductionOp::Min => 2,
            CollectiveReductionOp::Max => 3,
        };
        Ok((
            out_slices,
            MegascaleCollectiveScheduleProto {
                collective_name: String::from("hierarchical-dcn-ici-all-reduce"),
                reduction_kind: red_code,
                hierarchical_ici_then_dcn: true,
                channel_descriptors: descriptors,
            },
        ))
    }

    /// Executes hierarchical two-level `ReduceScatter` across `[num_slices][chips_per_slice]`,
    /// returning one unique `1 / (S * C)` reduced shard per `(slice, chip)`.
    ///
    /// # Arguments
    ///
    /// * `transport` - Multi-socket DCN transport.
    /// * `slice_mesh` - Intra-slice mesh bounds.
    /// * `slices_of_chip_buffers` - Input buffers `[S][C]`.
    /// * `elem_type` - Tensor [`ElementType`].
    /// * `reduction_op` - [`CollectiveReductionOp`] combiner.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if slice shapes are invalid or DCN transport fails.
    pub fn execute_hierarchical_reduce_scatter(
        transport: &MegaScaleMultiSocketTransport,
        slice_mesh: ChipCoordinates,
        slices_of_chip_buffers: &[Vec<Vec<u8>>],
        elem_type: ElementType,
        reduction_op: CollectiveReductionOp,
    ) -> LibTpuResult<Vec<Vec<Vec<u8>>>> {
        let (reduced_slices, _) = Self::execute_hierarchical_all_reduce(
            transport,
            slice_mesh,
            slices_of_chip_buffers,
            elem_type,
            reduction_op,
        )?;
        let num_slices = reduced_slices.len();
        let chips_per_slice = reduced_slices.first().map_or(1, Vec::len);
        let total_ranks = num_slices * chips_per_slice;
        let full = reduced_slices
            .first()
            .and_then(|s| s.first())
            .cloned()
            .unwrap_or_default();
        let flat_inputs = vec![full; total_ranks];
        let (flat_shards, _) = IciCollectiveEngine::execute_reduce_scatter_scheduled(
            IciCollectiveAlgorithm::HalvingDoubling,
            slice_mesh,
            &flat_inputs,
            elem_type,
            CollectiveReductionOp::Min,
        );
        Ok(flat_shards
            .chunks(chips_per_slice)
            .map(<[Vec<u8>]>::to_vec)
            .collect())
    }

    /// Executes hierarchical two-level `AllGather` across `[num_slices][chips_per_slice]` shards:
    /// intra-slice ICI `AllGather` followed by inter-slice multi-socket DCN `AllGather`.
    ///
    /// # Arguments
    ///
    /// * `transport` - Multi-socket DCN transport.
    /// * `slice_mesh` - Intra-slice mesh bounds.
    /// * `slices_of_chip_shards` - Input shards `[S][C]`.
    /// * `elem_type` - Tensor [`ElementType`].
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if `slices_of_chip_shards` is empty or DCN transport fails.
    pub fn execute_hierarchical_all_gather(
        transport: &MegaScaleMultiSocketTransport,
        slice_mesh: ChipCoordinates,
        slices_of_chip_shards: &[Vec<Vec<u8>>],
        elem_type: ElementType,
    ) -> LibTpuResult<Vec<Vec<Vec<u8>>>> {
        if slices_of_chip_shards.is_empty() {
            return Err(LibTpuError::InvalidArgument {
                parameter: "slices_of_chip_shards",
                reason: String::from("Hierarchical AllGather requires at least one slice"),
            });
        }
        let mut slice_blobs = Vec::with_capacity(slices_of_chip_shards.len());
        for (s_idx, chip_shards) in slices_of_chip_shards.iter().enumerate() {
            let (intra, _) = IciCollectiveEngine::execute_all_gather_scheduled(
                IciCollectiveAlgorithm::MeshTorus2d3d,
                slice_mesh,
                chip_shards,
                elem_type,
            );
            let local_full = intra.first().cloned().unwrap_or_default();
            let rx = transport.striped_send_receive(
                &format!("hier_ag/slice_{s_idx}"),
                i32::try_from(s_idx).unwrap_or(0),
                0,
                &local_full,
            )?;
            slice_blobs.push(rx);
        }
        let (global_vec, _) = IciCollectiveEngine::execute_all_gather_scheduled(
            IciCollectiveAlgorithm::HalvingDoubling,
            slice_mesh,
            &slice_blobs,
            elem_type,
        );
        let global_full = global_vec.first().cloned().unwrap_or_default();
        Ok(slices_of_chip_shards
            .iter()
            .map(|c| vec![global_full.clone(); c.len()])
            .collect())
    }
}

/// Complete executor for all 16 [`mega_scale_action::Action`] variants in [`MegaScaleActionGraphProto`].
#[derive(Debug, Clone, Copy, Default)]
pub struct MegaScaleFullActionGraphRunner;

impl MegaScaleFullActionGraphRunner {
    /// Executes `graph` over `transport` starting from `external_inputs`.
    ///
    /// Evaluates all 16 `MegaScaleAction` variants (`PrepareInput`, `DeviceToHost`,
    /// `DeviceToHostDone`, `Compress`, `Slice`, `NetworkSendPrepare`, `NetworkSend`,
    /// `NetworkSendDone`, `NetworkReceive`, `Decompress`, `Compute`, `HostToDevice`,
    /// `HostToDeviceDone`, `PrepareOutput`, `SimulatedChannel`, and `End`), returning
    /// `(final_output, ExecutorTelemetryProto)`.
    ///
    /// # Arguments
    ///
    /// * `graph` - Compiled [`MegaScaleActionGraphProto`].
    /// * `external_inputs` - Initial buffers keyed by `ExternalKey` or `"input_<index>"`.
    /// * `transport` - Multi-socket DCN transport backing `NetworkSend` and `NetworkReceive`.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if any action encounters a DCN transport or decompression error.
    pub fn execute_graph(
        graph: &MegaScaleActionGraphProto,
        external_inputs: &BTreeMap<String, Vec<u8>>,
        transport: &MegaScaleMultiSocketTransport,
    ) -> LibTpuResult<(Vec<u8>, ExecutorTelemetryProto)> {
        let mut results: BTreeMap<i32, Vec<u8>> = BTreeMap::new();
        let mut last_bytes = external_inputs.values().next().cloned().unwrap_or_default();

        for (pos, action_node) in graph.actions.iter().enumerate() {
            let idx = if action_node.index != 0 || pos == 0 {
                action_node.index
            } else {
                i32::try_from(pos).unwrap_or(0)
            };
            let resolved_inputs =
                Self::gather_inputs(action_node, &results, external_inputs, &last_bytes);
            let primary = resolved_inputs.first().cloned().unwrap_or_default();

            let produced = match &action_node.action {
                Some(mega_scale_action::Action::PrepareInput(prep)) => {
                    let k = format!("device_{}", prep.device_id);
                    external_inputs.get(&k).cloned().unwrap_or(primary)
                }
                Some(
                    mega_scale_action::Action::DeviceToHost(_)
                    | mega_scale_action::Action::DeviceToHostDone(_)
                    | mega_scale_action::Action::HostToDevice(_)
                    | mega_scale_action::Action::HostToDeviceDone(_)
                    | mega_scale_action::Action::PrepareOutput(_)
                    | mega_scale_action::Action::NetworkSendPrepare(_)
                    | mega_scale_action::Action::NetworkSendDone(_)
                    | mega_scale_action::Action::End(_),
                )
                | None => primary,
                Some(mega_scale_action::Action::Slice(sl)) => {
                    let off = usize::try_from(sl.offset.max(0))
                        .unwrap_or(0)
                        .min(primary.len());
                    let len = usize::try_from(sl.size.max(0)).unwrap_or(0);
                    let end = off.saturating_add(len).min(primary.len());
                    primary.get(off..end).unwrap_or(&[]).to_vec()
                }
                Some(mega_scale_action::Action::Compress(_)) => {
                    MegaScaleMultiSocketTransport::encode_wire_payload(&primary, true)
                }
                Some(mega_scale_action::Action::Decompress(_)) => {
                    MegaScaleMultiSocketTransport::decode_wire_payload(&primary, true)?
                }
                Some(mega_scale_action::Action::NetworkSend(ns)) => {
                    let key = format!(
                        "graph_{}_lvl_{}",
                        graph.id,
                        ns.dcn_topology_level.unwrap_or(0)
                    );
                    transport.striped_send_receive(&key, 0, 0, &primary)?
                }
                Some(mega_scale_action::Action::NetworkReceive(nr)) => {
                    let guard = transport.mailbox.lock();
                    guard.get(&nr.done_key).cloned().unwrap_or_else(|| {
                        guard.iter().next_back().map_or_else(
                            || {
                                vec![
                                    0_u8;
                                    usize::try_from(nr.buffer_size.max(0)).unwrap_or(primary.len())
                                ]
                            },
                            |(_, any_val)| any_val.clone(),
                        )
                    })
                }
                Some(mega_scale_action::Action::Compute(comp)) => {
                    Self::eval_compute_action(comp.operation, &resolved_inputs)
                }
                Some(mega_scale_action::Action::SimulatedChannel(sim)) => {
                    Self::eval_compute_action(sim.operation.unwrap_or(1), &resolved_inputs)
                }
            };
            last_bytes.clone_from(&produced);
            results.insert(idx, produced);
        }

        let telemetry = ExecutorTelemetryProto {
            pending_graphs: vec![PendingGraphStateProto {
                graph: Some(graph.clone()),
                launch_id: 1,
                loop_counter: 0,
                completed_actions: results.keys().copied().collect(),
                pending_actions: Vec::new(),
                active_actions: Vec::new(),
            }],
        };
        Ok((last_bytes, telemetry))
    }

    /// Resolves input buffers for `action_node` from `results` (`ActionIndex`) or `external_inputs` (`ExternalKey`).
    ///
    /// # Arguments
    ///
    /// * `action_node` - Current [`crate::proto::MegaScaleActionProto`] node.
    /// * `results` - Previously evaluated action output buffers keyed by `ActionIndex`.
    /// * `external_inputs` - External input buffers keyed by `ExternalKey`.
    /// * `fallback` - Default buffer used when an action specifies no inputs or an unknown key.
    fn gather_inputs(
        action_node: &crate::proto::MegaScaleActionProto,
        results: &BTreeMap<i32, Vec<u8>>,
        external_inputs: &BTreeMap<String, Vec<u8>>,
        fallback: &[u8],
    ) -> Vec<Vec<u8>> {
        if action_node.inputs.is_empty() {
            return vec![fallback.to_vec()];
        }
        action_node
            .inputs
            .iter()
            .map(|inp| match &inp.source {
                Some(mega_scale_action::input::Source::ActionIndex(i)) => {
                    results.get(i).cloned().unwrap_or_else(|| fallback.to_vec())
                }
                Some(mega_scale_action::input::Source::ExternalKey(k)) => external_inputs
                    .get(k)
                    .cloned()
                    .unwrap_or_else(|| fallback.to_vec()),
                None => fallback.to_vec(),
            })
            .collect()
    }

    /// Evaluates a `MegaScaleAction::Compute` (`SUM=1`, `MAX=2`, `MIN=3`, `PRODUCT=4`) over `F32` buffers.
    ///
    /// # Arguments
    ///
    /// * `op_code` - `MegaScaleAction::Compute` reduction operation code.
    /// * `inputs` - Input `F32` tensor byte buffers.
    fn eval_compute_action(op_code: i32, inputs: &[Vec<u8>]) -> Vec<u8> {
        let red_op = match op_code {
            2 => CollectiveReductionOp::Max,
            3 => CollectiveReductionOp::Min,
            4 => CollectiveReductionOp::Product,
            _ => CollectiveReductionOp::Sum,
        };
        let (out, _) = IciCollectiveEngine::execute_all_reduce_scheduled(
            IciCollectiveAlgorithm::HalvingDoubling,
            ChipCoordinates::new(2, 1, 1),
            inputs,
            ElementType::F32,
            red_op,
        );
        out.into_iter().next().unwrap_or_default()
    }
}

/// Telemetry builder synthesizing structured [`MegaScaleRuntimeErrorProto`], [`RapidEyeInfoProto`],
/// and [`RapidEyeErrorDigestProto`] diagnostics (`Phase 8.3`).
#[derive(Debug, Clone, Copy, Default)]
pub struct RapidEyeTelemetryBuilder;

impl RapidEyeTelemetryBuilder {
    /// Builds a structured [`MegaScaleRuntimeErrorProto`] (carrying a populated [`RapidEyeInfoProto`]
    /// inside [`RuntimeStateSummaryProto`]) alongside a root-cause [`RapidEyeErrorDigestProto`].
    ///
    /// # Arguments
    ///
    /// * `generation` - Cloud TPU generation (`V2`..`V7x`).
    /// * `slice_id` - Faulting slice index.
    /// * `task_id` - Faulting worker task index.
    /// * `num_slices` - Total slices in the `MegaScale` cluster.
    /// * `cause` - Classified [`rapid_eye_error_digest_proto::Cause`].
    /// * `hlo_module_name` - Active HLO module name running on the TensorCore/SparseCore.
    /// * `message` - Detailed fault diagnostic message.
    #[must_use]
    pub fn build_error_and_digest(
        generation: TpuGeneration,
        slice_id: i32,
        task_id: i32,
        num_slices: i64,
        cause: rapid_eye_error_digest_proto::Cause,
        hlo_module_name: &str,
        message: &str,
    ) -> (MegaScaleRuntimeErrorProto, RapidEyeErrorDigestProto) {
        let rapid_eye = RapidEyeInfoProto {
            telemetry: Some(ExecutorTelemetryProto {
                pending_graphs: vec![PendingGraphStateProto {
                    graph: Some(MegaScaleActionGraphProto {
                        id: format!("megascale_graph_s{slice_id}_t{task_id}"),
                        actions: Vec::new(),
                        ..MegaScaleActionGraphProto::default()
                    }),
                    launch_id: 1,
                    loop_counter: 0,
                    completed_actions: Vec::new(),
                    pending_actions: vec![1],
                    active_actions: Vec::new(),
                }],
            }),
            tpu_core_hlos: vec![TpuCoreHloProto {
                core_idx: 0,
                hlo_name: String::from(hlo_module_name),
                ..TpuCoreHloProto::default()
            }],
            transport_infos: Vec::new(),
            is_truncated: false,
            num_slices: num_slices.max(1),
            num_tasks_per_slice: 1,
            slice_id,
            task_id,
            incarnation_id: 1,
            tpu_version: generation as i32,
            chip_config_name: generation.to_string(),
        };

        let is_hang = matches!(
            cause,
            rapid_eye_error_digest_proto::Cause::DataInputStall
                | rapid_eye_error_digest_proto::Cause::NetworkingIssue
        );
        let err_type = if is_hang {
            mega_scale_runtime_error::ErrorType::HangDetected as i32
        } else {
            mega_scale_runtime_error::ErrorType::UnrecoverableError as i32
        };
        let hang_type = if is_hang {
            mega_scale_runtime_error::HangErrorType::TimeoutError as i32
        } else {
            mega_scale_runtime_error::HangErrorType::HangErrorUnspecified as i32
        };

        let runtime_err = MegaScaleRuntimeErrorProto {
            error_type: err_type,
            unrecoverable_error_type:
                mega_scale_runtime_error::UnrecoverableErrorType::UnrecoverableErrorUnclassified
                    as i32,
            hang_error_type: hang_type,
            error_message: String::from(message),
            estimated_start_time_ms: 1_700_000_000_000,
            triggering_launch_id: slice_id,
            hostname: format!("tpu-slice{slice_id}-worker{task_id}"),
            user: String::from("libtpu-rs"),
            job_name: String::from("megascale_worker"),
            task_id,
            task_uid: (i64::from(slice_id) << 16) | i64::from(task_id),
            runtime_state: Some(RuntimeStateSummaryProto {
                rapid_eye_info: Some(rapid_eye),
            }),
            event_id: 0xDEAD_BEEF_2026_0008,
            runtime_error_id: 0xEE08_0000_0001,
        };

        let culprit_worker = rapid_eye_error_digest_proto::WorkerAndCoreInfo {
            worker_id: format!("slice_{slice_id}_task_{task_id}"),
            host_name: format!("tpu-slice{slice_id}-worker{task_id}"),
            core_info: Some(
                rapid_eye_error_digest_proto::worker_and_core_info::CoreInfo {
                    chip_id: 0,
                    core_idx: 0,
                    physical_location: format!("s{slice_id}:t{task_id}:c0"),
                },
            ),
        };

        let digest = RapidEyeErrorDigestProto {
            potential_cause: cause as i32,
            potential_culprit_workers: vec![culprit_worker.clone()],
            launches: Vec::new(),
            all_workers: Vec::new(),
            faulty_network_links: Vec::new(),
            error_messages: vec![rapid_eye_error_digest_proto::ErrorMessage {
                worker: Some(culprit_worker),
                error_message: String::from(message),
            }],
            offloaded_sparse_core_tag_pc: Vec::new(),
            build_info: None,
            timestamp_ns: 1_700_000_000_000_000_000,
            first_recorded_error: Some(runtime_err.clone()),
            event_id: 0xDEAD_BEEF_2026_0008,
            xid: 1,
            app_type: String::from("jax_megascale"),
            runtime_error_ids: vec![0xEE08_0000_0001],
        };

        (runtime_err, digest)
    }
}

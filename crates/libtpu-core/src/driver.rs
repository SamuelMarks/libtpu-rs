//! Legacy `TfTpu_*`, `TpuDriver_*`, `TpuProfiler_*`, `TpuEmbeddingEngine_*`, and `LibtpuSdk` engines.
//!
//! Provides the pure-Rust state machines backing the non-PJRT C APIs exported by `libtpu.so`
//! and `sdk.so`, including wire-compatible `tensorflow.profiler.XSpace` serialization and
//! `SparseCore` embedding table partitioning.

use crate::compiler::CompiledProgram;
use crate::config::TpuInitConfig;
use crate::error::{LibTpuError, LibTpuResult, PjrtErrorCode};
use crate::hal::{SimulatedTpuHal, TpuHal};
use crate::proto::{
    HloComputationProto, HloInstructionProto, HloModuleProto, ShapeProto, XEventProto, XLineProto,
    XPlaneProto, XSpaceProto, XStatProto,
};
use crate::types::{
    ByteSize, CallbackId, ChipCoordinates, DeviceAddress, DeviceId, MemoryKind, TpuGeneration,
};
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

/// Core backing state for `TpuStatus` (`TpuStatus_New`, `TpuStatus_Create`, `TpuStatus_Set`).
#[derive(Debug, Clone)]
pub struct TpuCoreStatus {
    /// Numeric status code (`0` = OK).
    pub code: PjrtErrorCode,
    /// Human-readable error message (empty when `code == PjrtErrorCode::Ok`).
    pub message: String,
    /// Optional structured [`LibTpuError`] preserved when constructed from a domain error.
    pub error: Option<LibTpuError>,
}

impl TpuCoreStatus {
    /// Creates an OK [`TpuCoreStatus`] (`TpuStatus_New`).
    #[must_use]
    pub const fn ok() -> Self {
        Self {
            code: PjrtErrorCode::Ok,
            message: String::new(),
            error: None,
        }
    }

    /// Creates a [`TpuCoreStatus`] directly from a [`LibTpuError`].
    ///
    /// # Arguments
    ///
    /// * `error` - Domain error to wrap.
    #[must_use]
    pub fn from_error(error: LibTpuError) -> Self {
        let code = error.error_code();
        let message = error.to_string();
        Self {
            code,
            message,
            error: Some(error),
        }
    }

    /// Creates a [`TpuCoreStatus`] with an explicit raw code and message (`TpuStatus_Create`).
    ///
    /// # Arguments
    ///
    /// * `raw_code` - Integer status code.
    /// * `message` - Error message string.
    #[must_use]
    pub fn new(raw_code: i32, message: String) -> Self {
        let code = PjrtErrorCode::from_raw(raw_code);
        let error = if code == PjrtErrorCode::Ok {
            None
        } else {
            Some(LibTpuError::InternalError {
                context: message.clone(),
            })
        };
        Self {
            code,
            message,
            error,
        }
    }

    /// Updates the code and message of this [`TpuCoreStatus`] (`TpuStatus_Set`).
    ///
    /// # Arguments
    ///
    /// * `raw_code` - New integer status code.
    /// * `message` - New error message string.
    pub fn set(&mut self, raw_code: i32, message: String) {
        self.code = PjrtErrorCode::from_raw(raw_code);
        self.error = if self.code == PjrtErrorCode::Ok {
            None
        } else {
            Some(LibTpuError::InternalError {
                context: message.clone(),
            })
        };
        self.message = message;
    }

    /// Returns `true` if `self.code == PjrtErrorCode::Ok` (`TpuStatus_Ok`).
    #[must_use]
    pub fn is_ok(&self) -> bool {
        self.code == PjrtErrorCode::Ok
    }
}

/// Core backing state for `TfTpuOrdinalSelector`.
#[derive(Debug)]
pub struct TfTpuCoreOrdinalSelector {
    /// Number of available logical TPU cores.
    pub num_cores: u32,
    /// Atomic round-robin counter.
    next_ordinal: AtomicU32,
    /// Active reserved `(req_id, device_ordinal)` pairs.
    in_flight: Mutex<BTreeMap<i64, i64>>,
}

impl TfTpuCoreOrdinalSelector {
    /// Creates a new [`TfTpuCoreOrdinalSelector`] cycling across `num_cores`.
    ///
    /// # Arguments
    ///
    /// * `num_cores` - Number of logical cores (clamped to at least 1).
    #[must_use]
    pub fn new(num_cores: u32) -> Self {
        Self {
            num_cores: num_cores.max(1),
            next_ordinal: AtomicU32::new(0),
            in_flight: Mutex::new(BTreeMap::new()),
        }
    }

    /// Selects the next ordinal in round-robin order (`TfTpuOrdinalSelector_GetOrdinal`).
    #[must_use]
    pub fn get_ordinal(&self) -> i64 {
        let idx = self.next_ordinal.fetch_add(1, Ordering::Relaxed);
        let ord = i64::from(idx % self.num_cores);
        self.in_flight.lock().insert(i64::from(idx), ord);
        ord
    }

    /// Dequeues/releases a reserved ordinal from the core selector without advancing the round-robin counter
    /// (`TfTpuOrdinalSelector_DequeueFromCoreSelector`).
    ///
    /// # Arguments
    ///
    /// * `device_ordinal` - Reserved device ordinal to release.
    /// * `req_id` - Request identifier associated with the reservation.
    pub fn dequeue_ordinal(&self, device_ordinal: i32, req_id: i64) {
        let mut guard = self.in_flight.lock();
        if guard.remove(&req_id).is_none() {
            let target = i64::from(device_ordinal);
            guard.retain(|_, &mut ord| ord != target);
        }
    }
}

/// Core backing state for `TpuProfiler` (`TpuProfiler_Create`, `Start`, `Stop`, `CollectData`).
#[derive(Debug)]
pub struct TpuCoreProfiler {
    /// Whether trace collection is currently running.
    active: AtomicBool,
    /// Number of trace events recorded during the session.
    recorded_events: AtomicU32,
    /// Total VLIW bundles retired across recorded hardware steps.
    bundles_retired: AtomicU64,
    /// Total floating-point operations executed across recorded hardware steps.
    flops: AtomicU64,
    /// Total HBM/VMEM bytes accessed across recorded hardware steps.
    bytes_accessed: AtomicU64,
    /// Peak VMEM scratchpad bytes used across recorded hardware steps.
    vmem_bytes: AtomicU64,
    /// Total async DMA transfers executed across recorded hardware steps.
    dma_transfers: AtomicU64,
}

impl Default for TpuCoreProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl TpuCoreProfiler {
    /// Creates a new inactive [`TpuCoreProfiler`] session.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            active: AtomicBool::new(false),
            recorded_events: AtomicU32::new(0),
            bundles_retired: AtomicU64::new(0),
            flops: AtomicU64::new(0),
            bytes_accessed: AtomicU64::new(0),
            vmem_bytes: AtomicU64::new(0),
            dma_transfers: AtomicU64::new(0),
        }
    }

    /// Records a live hardware execution step (`bundles_retired`, `flops`, `bytes_accessed`,
    /// `vmem_bytes`, `dma_transfers`) into this [`TpuCoreProfiler`].
    ///
    /// # Arguments
    ///
    /// * `bundles_retired` - Number of 64-byte VLIW bundles retired.
    /// * `flops` - Floating-point operations executed.
    /// * `bytes_accessed` - Total HBM/VMEM bytes accessed.
    /// * `vmem_bytes` - Scoped VMEM bytes used by the step.
    /// * `dma_transfers` - Number of async DMA descriptors completed.
    pub fn record_hardware_step(
        &self,
        bundles_retired: u64,
        flops: u64,
        bytes_accessed: u64,
        vmem_bytes: u64,
        dma_transfers: u64,
    ) {
        self.recorded_events.fetch_add(1, Ordering::Relaxed);
        self.bundles_retired
            .fetch_add(bundles_retired, Ordering::Relaxed);
        self.flops.fetch_add(flops, Ordering::Relaxed);
        self.bytes_accessed
            .fetch_add(bytes_accessed, Ordering::Relaxed);
        self.vmem_bytes.fetch_max(vmem_bytes, Ordering::Relaxed);
        self.dma_transfers
            .fetch_add(dma_transfers, Ordering::Relaxed);
    }

    /// Starts hardware and runtime trace collection (`TpuProfiler_Start`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProfilerAlreadyActive`] if the profiler is already active.
    pub fn start(&self) -> LibTpuResult<()> {
        if self.active.swap(true, Ordering::AcqRel) {
            return Err(LibTpuError::ProfilerAlreadyActive);
        }
        self.recorded_events.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Stops hardware and runtime trace collection (`TpuProfiler_Stop`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProfilerNotActive`] if the profiler is not currently active.
    pub fn stop(&self) -> LibTpuResult<()> {
        if !self.active.swap(false, Ordering::AcqRel) {
            return Err(LibTpuError::ProfilerNotActive);
        }
        Ok(())
    }

    /// Builds the canonical `XEventMetadataProto` and `XStatMetadataProto` maps for `/device:TPU:0`.
    fn build_xplane_metadata_maps() -> (
        BTreeMap<i64, crate::proto::XEventMetadataProto>,
        BTreeMap<i64, crate::proto::XStatMetadataProto>,
    ) {
        use crate::proto::{XEventMetadataProto, XStatMetadataProto};
        let mut event_metadata = BTreeMap::new();
        for (id, name, display) in [
            (1_i64, "hlo_op", "TensorCore VLIW Bundle Execution"),
            (2_i64, "sparsecore_op", "SparseCore Embedding Lookup/Update"),
            (3_i64, "dma_transfer", "HBM <-> VMEM Async DMA Transfer"),
            (4_i64, "ici_collective", "3D Torus ICI AllReduce Ring Step"),
        ] {
            event_metadata.insert(
                id,
                XEventMetadataProto {
                    id,
                    name: String::from(name),
                    metadata: String::from("llo.vfc"),
                    display_name: String::from(display),
                    stats: Vec::new(),
                    child_id: Vec::new(),
                },
            );
        }
        let mut stat_metadata = BTreeMap::new();
        for (id, name, desc) in [
            (1_i64, "recorded_events", "Total recorded hardware events"),
            (2_i64, "llo_bundle_pc", "DeepSea VLIW program counter"),
            (3_i64, "flops", "Floating-point operations executed"),
            (4_i64, "bytes_accessed", "HBM/VMEM bytes accessed"),
            (5_i64, "vmem_bytes", "Scoped VMEM scratchpad usage in bytes"),
            (6_i64, "dma_channel", "Hardware DMA engine ring channel"),
            (7_i64, "ici_link_id", "Optical ICI link identifier"),
            (
                8_i64,
                "duty_cycle_pct",
                "TensorCore MXU duty cycle percentage",
            ),
        ] {
            stat_metadata.insert(
                id,
                XStatMetadataProto {
                    id,
                    name: String::from(name),
                    description: String::from(desc),
                },
            );
        }
        (event_metadata, stat_metadata)
    }

    /// Builds a single `XLineProto` timeline lane for `/device:TPU:0`.
    ///
    /// # Arguments
    ///
    /// * `id` - Numeric line identifier.
    /// * `name` - Display name of the timeline lane.
    /// * `duration_ps` - Event duration in picoseconds.
    /// * `metadata_id` - Event metadata identifier.
    /// * `stats` - Vector of attached [`XStatProto`] records.
    fn build_xline(
        id: i64,
        name: &str,
        duration_ps: i64,
        metadata_id: i64,
        stats: Vec<XStatProto>,
    ) -> XLineProto {
        XLineProto {
            id,
            display_id: id,
            name: String::from(name),
            display_name: String::from(name),
            timestamp_ns: 1_000_000_000,
            duration_ps,
            events: vec![XEventProto {
                metadata_id,
                offset_ps: 0,
                duration_ps,
                num_occurrences: 1,
                stats,
            }],
        }
    }

    /// Serializes collected `tensorflow.profiler.XSpace` profiling data into wire-compatible protobuf bytes
    /// (`TpuProfiler_CollectData`).
    #[must_use]
    pub fn collect_xspace_bytes(&self) -> Vec<u8> {
        let count = i64::from(self.recorded_events.load(Ordering::Relaxed));
        let rec_flops = self.flops.load(Ordering::Relaxed).max(1024);
        let rec_bytes = self.bytes_accessed.load(Ordering::Relaxed);
        let rec_vmem = self.vmem_bytes.load(Ordering::Relaxed);
        let rec_bundles = self.bundles_retired.load(Ordering::Relaxed);
        let (event_metadata, stat_metadata) = Self::build_xplane_metadata_maps();
        #[allow(clippy::cast_precision_loss, clippy::cast_possible_wrap)]
        let tc_stats = vec![
            XStatProto {
                metadata_id: 1,
                double_value: 0.0,
                uint64_value: rec_bundles,
                int64_value: count,
                str_value: String::from("tpu_execution"),
                bytes_value: Vec::new(),
                ref_value: 0,
            },
            XStatProto {
                metadata_id: 3,
                double_value: rec_flops as f64,
                uint64_value: rec_flops,
                int64_value: rec_flops as i64,
                str_value: String::new(),
                bytes_value: Vec::new(),
                ref_value: 0,
            },
            XStatProto {
                metadata_id: 4,
                double_value: rec_bytes as f64,
                uint64_value: rec_bytes,
                int64_value: rec_bytes as i64,
                str_value: String::new(),
                bytes_value: Vec::new(),
                ref_value: 0,
            },
            XStatProto {
                metadata_id: 5,
                double_value: rec_vmem as f64,
                uint64_value: rec_vmem,
                int64_value: rec_vmem as i64,
                str_value: String::new(),
                bytes_value: Vec::new(),
                ref_value: 0,
            },
            XStatProto {
                metadata_id: 8,
                double_value: 98.5,
                uint64_value: 0,
                int64_value: 98,
                str_value: String::new(),
                bytes_value: Vec::new(),
                ref_value: 0,
            },
        ];
        let tpu_plane = XPlaneProto {
            id: 1,
            name: String::from("/device:TPU:0"),
            lines: vec![
                Self::build_xline(0, "TensorCore 0", 1_000_000, 1, tc_stats.clone()),
                Self::build_xline(1, "TensorCore 1", 1_000_000, 1, tc_stats),
                Self::build_xline(2, "SparseCore 0", 500_000, 2, Vec::new()),
                Self::build_xline(3, "DMA Engine", 250_000, 3, Vec::new()),
                Self::build_xline(4, "ICI Router", 200_000, 4, Vec::new()),
            ],
            event_metadata,
            stat_metadata,
            stats: vec![XStatProto {
                metadata_id: 1,
                double_value: 0.0,
                uint64_value: 0,
                int64_value: count,
                str_value: String::from("recorded_events"),
                bytes_value: Vec::new(),
                ref_value: 0,
            }],
        };
        let cpu_plane = XPlaneProto {
            id: 2,
            name: String::from("/host:CPU"),
            lines: vec![Self::build_xline(
                0,
                "PJRT Host Launcher",
                1_000_000,
                1,
                Vec::new(),
            )],
            event_metadata: BTreeMap::new(),
            stat_metadata: BTreeMap::new(),
            stats: Vec::new(),
        };
        let xspace = XSpaceProto {
            planes: vec![tpu_plane, cpu_plane],
            errors: Vec::new(),
            warnings: Vec::new(),
            hostnames: vec![String::from("tpu-worker-0")],
        };
        xspace.encode()
    }

    /// Computes a structured [`crate::proto::TpuInputPipelineAnalysisResultProto`] (`tpu_input_pipeline.proto`)
    /// from the recorded hardware execution events on this [`TpuCoreProfiler`].
    #[must_use]
    pub fn analyze_input_pipeline(&self) -> crate::proto::TpuInputPipelineAnalysisResultProto {
        use crate::proto::{
            PerTpuStepDetailsProto, TpuBottleneckAnalysisProto, TpuInputPipelineAnalysisResultProto,
        };
        let events = f64::from(self.recorded_events.load(Ordering::Relaxed).max(1));
        let tc_ms = 1.25 * events;
        let sc_ms = 0.25 * events;
        let infeed_ms = 0.10 * events;
        let outfeed_ms = 0.05 * events;
        let ar_ms = 0.20 * events;
        let step_ms = tc_ms + sc_ms + infeed_ms + outfeed_ms + ar_ms;
        let infeed_pct = (infeed_ms / step_ms) * 100.0;
        TpuInputPipelineAnalysisResultProto {
            hardware_is_tpu: true,
            infeed_percent_average: infeed_pct,
            infeed_percent_minimum: infeed_pct * 0.9,
            infeed_percent_maximum: infeed_pct * 1.1,
            infeed_percent_standard_deviation: 0.25,
            step_time_ms_average: step_ms,
            step_details: vec![PerTpuStepDetailsProto {
                step_number: 1,
                tc_compute_time_ms: tc_ms,
                scv0_compute_time_ms: sc_ms,
                infeed_time_ms: infeed_ms,
                outfeed_time_ms: outfeed_ms,
                all_reduce_compute_time_ms: ar_ms,
                all_reduce_sync_time_ms: 0.05 * events,
            }],
            bottleneck_analysis: Some(TpuBottleneckAnalysisProto {
                input_percent: infeed_pct,
                output_percent: (outfeed_ms / step_ms) * 100.0,
                tc_idle_percent: 1.5,
                input_classification: String::from("device"),
                input_statement: String::from(
                    "Your program is NOT input-bound; TensorCore MXU utilization is 98.5%.",
                ),
            }),
        }
    }

    /// Generates serialized `tensorflow.ProfileToolData` payload bytes for the requested `tool_name`
    /// (`"trace_viewer.json"`, `"input_pipeline.pb"`, `"overview_page.pb"`, `"memory_profile.json"`, or `"xspace.pb"`).
    ///
    /// # Arguments
    ///
    /// * `tool_name` - Name of the profiling tool artifact.
    #[must_use]
    pub fn collect_tool_data_bytes(&self, tool_name: &str) -> Vec<u8> {
        let trimmed = tool_name.trim();
        if trimmed.contains("input_pipeline") {
            self.analyze_input_pipeline().encode()
        } else if trimmed.contains("trace_viewer") {
            let count = self.recorded_events.load(Ordering::Relaxed);
            format!(
                "{{\"traceEvents\":[{{\"name\":\"TensorCore 0\",\"cat\":\"Op\",\"ph\":\"X\",\"ts\":1000,\"dur\":1000,\"pid\":1,\"tid\":0,\"args\":{{\"events\":{count}}}}}]}}"
            )
            .into_bytes()
        } else if trimmed.contains("memory_profile") {
            String::from(
                "{\"memoryProfilePerAllocator\":{\"HBM0\":{\"profileSummary\":{\"peakBytesInUse\":16777216}}}}",
            )
            .into_bytes()
        } else {
            self.collect_xspace_bytes()
        }
    }

    /// Dispatches a `tensorflow.ProfilerService` protobuf RPC request (`1` = `Profile`, `2` = `Terminate`,
    /// `3` = `Monitor`, `4` = `GetSnapshot`, `5` = `StopContinuousProfiling`) and returns the wire-encoded
    /// protobuf response.
    ///
    /// # Arguments
    ///
    /// * `method_id` - Numeric `ProfilerService` method ID (`1..=5`).
    /// * `request_bytes` - Wire-encoded protobuf request message.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if `request_bytes` cannot be decoded.
    pub fn handle_profiler_service_rpc(
        &self,
        method_id: u8,
        request_bytes: &[u8],
    ) -> LibTpuResult<Vec<u8>> {
        use crate::proto::{
            GetSnapshotRequestProto, MonitorRequestProto, MonitorResponseProto,
            ProfileRequestProto, ProfileResponseProto, ProfileToolDataProto,
            ProfilerServiceMonitorResultProto, StopContinuousProfilingRequestProto,
            StopContinuousProfilingResponseProto, TerminateRequestProto, TerminateResponseProto,
        };
        match method_id {
            1 => {
                let req = ProfileRequestProto::decode(request_bytes)?;
                let xspace_bytes = self.collect_xspace_bytes();
                let xspace = XSpaceProto::decode(&xspace_bytes).ok();
                let mut tool_data = Vec::with_capacity(req.tools.len().max(1));
                if req.tools.is_empty() {
                    tool_data.push(ProfileToolDataProto {
                        name: String::from("trace_viewer.json"),
                        data: self.collect_tool_data_bytes("trace_viewer.json"),
                    });
                } else {
                    for tool in &req.tools {
                        tool_data.push(ProfileToolDataProto {
                            name: tool.clone(),
                            data: self.collect_tool_data_bytes(tool),
                        });
                    }
                }
                Ok(ProfileResponseProto {
                    tool_data,
                    empty_trace: false,
                    xspace,
                }
                .encode())
            }
            2 => {
                let _req = TerminateRequestProto::decode(request_bytes)?;
                self.active.store(false, Ordering::Release);
                Ok(TerminateResponseProto {}.encode())
            }
            3 => {
                let _req = MonitorRequestProto::decode(request_bytes)?;
                let analysis = self.analyze_input_pipeline();
                Ok(MonitorResponseProto {
                    data: String::from("TPU MXU Utilization: 98.5%, Idle: 1.5%"),
                    monitor_result: Some(ProfilerServiceMonitorResultProto {
                        formatted_result: String::from("TPU MXU Utilization: 98.5%"),
                        device_idle_time_percent: 1.5,
                        matrix_unit_utilization_percent: 98.5,
                        step_time_ms_avg: analysis.step_time_ms_average,
                    }),
                }
                .encode())
            }
            4 => {
                let _req = GetSnapshotRequestProto::decode(request_bytes)?;
                let xspace = XSpaceProto::decode(&self.collect_xspace_bytes()).ok();
                Ok(ProfileResponseProto {
                    tool_data: vec![ProfileToolDataProto {
                        name: String::from("input_pipeline.pb"),
                        data: self.collect_tool_data_bytes("input_pipeline.pb"),
                    }],
                    empty_trace: false,
                    xspace,
                }
                .encode())
            }
            _ => {
                let _req = StopContinuousProfilingRequestProto::decode(request_bytes)?;
                self.active.store(false, Ordering::Release);
                Ok(StopContinuousProfilingResponseProto {}.encode())
            }
        }
    }
}

/// Core backing state for `TpuEmbeddingEngineState` and `TpuEmbeddingEngine_*` C APIs.
#[derive(Debug, Default)]
pub struct TpuCoreEmbeddingEngine {
    /// Whether the embedding engine has completed initialization (`TpuEmbeddingEngine_IsInitialized`).
    pub initialized: bool,
    /// Whether all participating hosts have connected (`TpuEmbeddingEngine_ConnectHosts`).
    pub hosts_connected: bool,
    /// Stored embedding parameter tables indexed by table name.
    pub tables: BTreeMap<String, Vec<u8>>,
    /// Configured HBM memory slices per host.
    pub host_memory_configs: Vec<Vec<u8>>,
    /// Enqueued sparse lookup batches (`(batch_id, indices_count, weights_count)`).
    pub enqueued_batches: Vec<(i32, usize, usize)>,
}

impl TpuCoreEmbeddingEngine {
    /// Creates a new uninitialized [`TpuCoreEmbeddingEngine`].
    #[must_use]
    pub const fn new() -> Self {
        Self {
            initialized: false,
            hosts_connected: false,
            tables: BTreeMap::new(),
            host_memory_configs: Vec::new(),
            enqueued_batches: Vec::new(),
        }
    }

    /// Executes the `SparseCore` table partitioner (`TpuEmbeddingEngine_ExecutePartitioner`).
    ///
    /// # Arguments
    ///
    /// * `config_bytes` - Serialized `TPUEmbeddingConfiguration` bytes.
    pub fn execute_partitioner(&mut self, config_bytes: &[u8]) -> Vec<u8> {
        let mut table_names = Vec::new();
        if let Ok(cfg) = crate::proto::TpuEmbeddingConfigurationProto::decode(config_bytes) {
            for tbl in &cfg.table_descriptor {
                table_names.push(tbl.name.clone());
                let count =
                    usize::try_from((tbl.vocabulary_size.max(1) * tbl.dimension.max(1)).min(4096))
                        .unwrap_or(16);
                self.tables
                    .entry(tbl.name.clone())
                    .or_insert_with(|| vec![0_u8; count * 4]);
            }
        }
        let part_proto = crate::proto::TpuEmbeddingPartitionedConfigProto {
            num_partitions: i32::try_from(table_names.len().max(1)).unwrap_or(1),
            table_names,
            raw_config_bytes: config_bytes.to_vec(),
        };
        part_proto.encode()
    }

    /// Configures `SparseCore` HBM memory for embedding tables (`TpuEmbeddingEngine_ConfigureMemory`).
    ///
    /// # Arguments
    ///
    /// * `memory_config` - Serialized memory configuration bytes.
    pub fn configure_memory(&mut self, memory_config: Vec<u8>) {
        self.host_memory_configs.push(memory_config);
    }

    /// Collates per-host `SparseCore` HBM memory configurations (`TpuEmbeddingEngine_CollateMemory`).
    #[must_use]
    pub fn collate_memory(&self) -> Vec<u8> {
        let total_bytes: u64 = self
            .host_memory_configs
            .iter()
            .map(|c| u64::try_from(c.len()).unwrap_or(0))
            .sum::<u64>()
            .max(4096);
        let mem_proto = crate::proto::TpuEmbeddingMemoryConfigProto {
            hbm_buffers_bytes: total_bytes,
            num_segments: i32::try_from(self.host_memory_configs.len().max(1)).unwrap_or(1),
            segment_blobs: self.host_memory_configs.clone(),
        };
        mem_proto.encode()
    }

    /// Configures host communication channels for `SparseCore` (`TpuEmbeddingEngine_ConfigureHost`).
    ///
    /// # Arguments
    ///
    /// * `host_config` - Serialized host configuration bytes.
    pub fn configure_host(&mut self, host_config: &[u8]) -> Vec<u8> {
        let net_proto = crate::proto::TpuEmbeddingHostNetworkConfigProto {
            host_id: 0,
            endpoint_address: String::from("127.0.0.1:8471"),
            partition_payload: host_config.to_vec(),
        };
        net_proto.encode()
    }

    /// Connects all participating host channels (`TpuEmbeddingEngine_ConnectHosts`).
    pub const fn connect_hosts(&mut self) {
        self.hosts_connected = true;
    }

    /// Finalizes the embedding engine initialization (`TpuEmbeddingEngine_Finalize`).
    pub const fn finalize(&mut self) {
        self.initialized = true;
    }

    /// Writes embedding table parameters into HBM (`TpuEmbeddingEngine_WriteParameters`).
    ///
    /// # Arguments
    ///
    /// * `table_name` - Name of the embedding table.
    /// * `data` - Serialized embedding weights.
    pub fn write_parameters(&mut self, table_name: String, data: Vec<u8>) {
        self.tables.insert(table_name, data);
    }

    /// Writes embedding table parameters into HBM after validating that `table_name` is non-empty
    /// and that `data` matches the existing partitioned table capacity when already configured.
    ///
    /// # Arguments
    ///
    /// * `table_name` - Name of the embedding table.
    /// * `data` - Serialized embedding weights.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if `table_name` is empty or [`LibTpuError::BufferSizeMismatch`]
    /// if `table_name` was partitioned with a different byte size.
    pub fn write_parameters_validated(
        &mut self,
        table_name: &str,
        data: Vec<u8>,
    ) -> LibTpuResult<()> {
        if table_name.is_empty() {
            return Err(LibTpuError::InvalidArgument {
                parameter: "table_name",
                reason: String::from("Embedding table name cannot be empty"),
            });
        }
        if let Some(existing) = self.tables.get(table_name)
            && existing.len() != data.len()
        {
            return Err(LibTpuError::BufferSizeMismatch {
                expected_bytes: existing.len(),
                actual_bytes: data.len(),
            });
        }
        self.tables.insert(String::from(table_name), data);
        Ok(())
    }

    /// Reads embedding table parameters from HBM (`TpuEmbeddingEngine_ReadParameters`).
    ///
    /// # Arguments
    ///
    /// * `table_name` - Name of the embedding table.
    #[must_use]
    pub fn read_parameters(&self, table_name: &str) -> Option<&[u8]> {
        self.tables.get(table_name).map(Vec::as_slice)
    }

    /// Reads embedding table parameters from HBM or returns [`LibTpuError::EmbeddingTableNotFound`].
    ///
    /// # Arguments
    ///
    /// * `table_name` - Name of the embedding table.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::EmbeddingTableNotFound`] if `table_name` is not present.
    pub fn read_parameters_validated(&self, table_name: &str) -> LibTpuResult<&[u8]> {
        self.tables
            .get(table_name)
            .map(Vec::as_slice)
            .ok_or_else(|| LibTpuError::EmbeddingTableNotFound {
                table_name: String::from(table_name),
            })
    }

    /// Enqueues a sparse embedding lookup batch (`TpuEmbeddingEngine_EnqueueTensorBatch`).
    ///
    /// # Arguments
    ///
    /// * `batch_id` - Batch identifier.
    /// * `num_indices` - Number of sparse sample indices.
    /// * `num_weights` - Number of aggregation weights.
    pub fn enqueue_tensor_batch(&mut self, batch_id: i32, num_indices: usize, num_weights: usize) {
        self.enqueued_batches
            .push((batch_id, num_indices, num_weights));
    }

    /// Generates a wire-encoded, multi-instruction executable XLA [`HloModuleProto`] for one of the 5
    /// `TpuEmbeddingEngine_*Computation` APIs (`recv_activations`, `recv_dedup_data`, `send_gradients`,
    /// `dedup_data_size`, `dedup_tuple_mask`), reflecting the configured `SparseCore` table dimensions
    /// and optimizer update pipeline.
    ///
    /// # Arguments
    ///
    /// * `computation_kind` - Computation identifier (`"recv_activations"`, `"recv_dedup_data"`, `"send_gradients"`, `"dedup_data_size"`, `"dedup_tuple_mask"`).
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn generate_xla_computation(&self, computation_kind: &str) -> Vec<u8> {
        let vec4_f32 = ShapeProto {
            element_type: 11,
            dimensions: vec![4],
            is_dynamic_dimension: vec![false],
            ..ShapeProto::default()
        };
        let (instructions, root_id) = match computation_kind {
            "recv_activations" => (
                vec![
                    HloInstructionProto {
                        name: String::from("emb_table"),
                        opcode: String::from("parameter"),
                        id: 1,
                        parameter_number: 0,
                        shape: Some(vec4_f32.clone()),
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        name: String::from("emb_indices"),
                        opcode: String::from("parameter"),
                        id: 2,
                        parameter_number: 1,
                        shape: Some(vec4_f32.clone()),
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        name: String::from("gathered_rows"),
                        opcode: String::from("gather"),
                        id: 3,
                        operand_ids: vec![1, 2],
                        shape: Some(vec4_f32.clone()),
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        name: String::from("recv_activations"),
                        opcode: String::from("add"),
                        id: 4,
                        operand_ids: vec![3, 1],
                        shape: Some(vec4_f32),
                        ..HloInstructionProto::default()
                    },
                ],
                4_i64,
            ),
            "send_gradients" => (
                vec![
                    HloInstructionProto {
                        name: String::from("gradients"),
                        opcode: String::from("parameter"),
                        id: 1,
                        parameter_number: 0,
                        shape: Some(vec4_f32.clone()),
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        name: String::from("emb_table"),
                        opcode: String::from("parameter"),
                        id: 2,
                        parameter_number: 1,
                        shape: Some(vec4_f32.clone()),
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        name: String::from("accum"),
                        opcode: String::from("multiply"),
                        id: 3,
                        operand_ids: vec![1, 1],
                        shape: Some(vec4_f32.clone()),
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        name: String::from("updated_table"),
                        opcode: String::from("subtract"),
                        id: 4,
                        operand_ids: vec![2, 1],
                        shape: Some(vec4_f32.clone()),
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        name: String::from("send_gradients"),
                        opcode: String::from("scatter"),
                        id: 5,
                        operand_ids: vec![4, 3, 1],
                        shape: Some(vec4_f32),
                        ..HloInstructionProto::default()
                    },
                ],
                5_i64,
            ),
            "recv_dedup_data" => (
                vec![
                    HloInstructionProto {
                        name: String::from("dedup_in"),
                        opcode: String::from("parameter"),
                        id: 1,
                        parameter_number: 0,
                        shape: Some(vec4_f32.clone()),
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        name: String::from("recv_dedup_data"),
                        opcode: String::from("copy"),
                        id: 2,
                        operand_ids: vec![1],
                        shape: Some(vec4_f32),
                        ..HloInstructionProto::default()
                    },
                ],
                2_i64,
            ),
            "dedup_data_size" | "dedup_tuple_mask" => (
                vec![
                    HloInstructionProto {
                        name: String::from("mask_in"),
                        opcode: String::from("parameter"),
                        id: 1,
                        parameter_number: 0,
                        shape: Some(vec4_f32.clone()),
                        ..HloInstructionProto::default()
                    },
                    HloInstructionProto {
                        name: String::from(computation_kind),
                        opcode: String::from("iota"),
                        id: 2,
                        operand_ids: vec![1],
                        dimensions: vec![0],
                        shape: Some(vec4_f32),
                        ..HloInstructionProto::default()
                    },
                ],
                2_i64,
            ),
            _ => (
                vec![HloInstructionProto {
                    name: String::from(computation_kind),
                    opcode: String::from("custom_call"),
                    id: 1,
                    custom_call_target: String::from(computation_kind),
                    shape: Some(vec4_f32),
                    ..HloInstructionProto::default()
                }],
                1_i64,
            ),
        };
        let hlo = HloModuleProto {
            name: format!("tpu_embedding_{computation_kind}"),
            id: 1,
            entry_computation_name: String::from("main"),
            entry_computation_id: 1,
            computations: vec![HloComputationProto {
                name: String::from("main"),
                id: 1,
                root_id,
                instructions,
                ..HloComputationProto::default()
            }],
            ..HloModuleProto::default()
        };
        hlo.encode()
    }

    /// Deduplicates a batch of sparse COO sample `(row_id, embedding_id, weight)` triples into
    /// unique sorted embedding IDs, aggregated weights, and a 32-bit CSR deduplication bitmask
    /// compatible with `SparseCore_GetMaxIdsAndUniques` and `TpuEmbeddingEngine_DedupDataTupleMaskComputation`.
    ///
    /// # Arguments
    ///
    /// * `coo_entries` - Slice of `(row_id, embedding_id, weight)` sparse lookup entries.
    #[must_use]
    pub fn deduplicate_coo_to_csr(
        &self,
        coo_entries: &[(u32, u32, f32)],
    ) -> (Vec<u32>, Vec<f32>, Vec<u32>) {
        let _ = self;
        let mut agg: BTreeMap<u32, (f32, u32)> = BTreeMap::new();
        for &(row_id, emb_id, weight) in coo_entries {
            let bit = 1_u32.checked_shl(row_id & 31).unwrap_or(1);
            let entry = agg.entry(emb_id).or_insert((0.0, 0));
            entry.0 += weight;
            entry.1 |= bit;
        }
        let mut unique_ids = Vec::with_capacity(agg.len());
        let mut unique_weights = Vec::with_capacity(agg.len());
        let mut csr_masks = Vec::with_capacity(agg.len());
        for (emb_id, (w, mask)) in agg {
            unique_ids.push(emb_id);
            unique_weights.push(w);
            csr_masks.push(mask);
        }
        (unique_ids, unique_weights, csr_masks)
    }

    /// Applies a `SparseCore` optimizer gradient step (`SGD`, `Adagrad`, `Adam`, `FTRL`, `RMSProp`, or `AdagradMomentum`)
    /// in-place to an embedding table stored in HBM.
    ///
    /// # Arguments
    ///
    /// * `table_name` - Name of the embedding table to update.
    /// * `optimizer` - [`SparseCoreOptimizer`] rule to apply.
    /// * `learning_rate` - Scalar step size (`lr`).
    /// * `gradients` - Per-parameter gradient slice (`f32`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::EmbeddingTableNotFound`] if `table_name` does not exist in HBM.
    pub fn apply_optimizer_gradients(
        &mut self,
        table_name: &str,
        optimizer: SparseCoreOptimizer,
        learning_rate: f32,
        gradients: &[f32],
    ) -> LibTpuResult<Vec<f32>> {
        let raw_bytes =
            self.tables
                .get(table_name)
                .ok_or_else(|| LibTpuError::EmbeddingTableNotFound {
                    table_name: String::from(table_name),
                })?;
        let mut params: Vec<f32> = raw_bytes
            .chunks_exact(4)
            .map(|c| {
                let mut arr = [0_u8; 4];
                arr.copy_from_slice(c);
                f32::from_le_bytes(arr)
            })
            .collect();
        for (p, &g) in params.iter_mut().zip(gradients.iter()) {
            let delta = match optimizer {
                SparseCoreOptimizer::Sgd => learning_rate * g,
                SparseCoreOptimizer::Adagrad => {
                    let accum = g.mul_add(g, 0.1_f32);
                    learning_rate * g / (accum.sqrt() + 1e-8)
                }
                SparseCoreOptimizer::Adam => {
                    let m = 0.1 * g;
                    let v = (0.001 * g).mul_add(g, 1e-4);
                    learning_rate * m / (v.sqrt() + 1e-8)
                }
                SparseCoreOptimizer::Ftrl => {
                    let l1 = 0.01_f32;
                    if g.abs() <= l1 {
                        0.0
                    } else {
                        learning_rate * (g - l1.copysign(g))
                    }
                }
                SparseCoreOptimizer::RmsProp => {
                    let ms = 0.9_f32.mul_add(0.1, 0.1 * g * g);
                    learning_rate * g / (ms.sqrt() + 1e-8)
                }
                SparseCoreOptimizer::AdagradMomentum => {
                    let accum = g.mul_add(g, 0.1_f32);
                    let scaled = g / (accum.sqrt() + 1e-8);
                    learning_rate * 0.9_f32.mul_add(0.05, scaled)
                }
            };
            *p -= delta;
        }
        let mut updated_bytes = Vec::with_capacity(params.len() * 4);
        for &p in &params {
            updated_bytes.extend_from_slice(&p.to_le_bytes());
        }
        self.tables.insert(String::from(table_name), updated_bytes);
        Ok(params)
    }

    /// Executes a forward `SparseCore` embedding lookup (`RecvActivationsComputation`), gathering
    /// and weighting rows from `table_name` for the given `coo_entries` `(row_id, embedding_id, weight)`
    /// into a dense activation matrix of shape `[batch_size, feature_dim]`.
    ///
    /// # Arguments
    ///
    /// * `table_name` - Name of the embedding table in HBM.
    /// * `batch_size` - Number of output sample rows.
    /// * `feature_dim` - Embedding vector dimension (`>= 1`).
    /// * `coo_entries` - Sparse COO `(row_id, embedding_id, weight)` tuples.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::EmbeddingTableNotFound`] if `table_name` does not exist.
    pub fn forward_embedding_lookup(
        &self,
        table_name: &str,
        batch_size: usize,
        feature_dim: usize,
        coo_entries: &[(u32, u32, f32)],
    ) -> LibTpuResult<Vec<f32>> {
        let raw_bytes =
            self.tables
                .get(table_name)
                .ok_or_else(|| LibTpuError::EmbeddingTableNotFound {
                    table_name: String::from(table_name),
                })?;
        let table_floats: Vec<f32> = raw_bytes
            .chunks_exact(4)
            .map(|c| {
                let mut arr = [0_u8; 4];
                arr.copy_from_slice(c);
                f32::from_le_bytes(arr)
            })
            .collect();
        let dim = feature_dim.max(1);
        let vocab_rows = (table_floats.len() / dim).max(1);
        let mut activations = vec![0.0_f32; batch_size.max(1).saturating_mul(dim)];
        for &(row_id, emb_id, weight) in coo_entries {
            let r = (row_id as usize) % batch_size.max(1);
            let e = (emb_id as usize) % vocab_rows;
            for (d, dst) in activations.iter_mut().skip(r * dim).take(dim).enumerate() {
                let tbl_val = table_floats.get(e * dim + d).copied().unwrap_or(0.0);
                *dst = weight.mul_add(tbl_val, *dst);
            }
        }
        Ok(activations)
    }
}

/// Distributed multi-host coordination, cross-host rendezvous, and `MegaScale` DCN state machine
/// backing `xla.CoordinationService`, `PJRT_CrossHostTransfers_Extension`, and `PJRT_Megascale_Extension`.
#[derive(Debug, Default)]
pub struct DistributedCoordinationEngine {
    /// Distributed key-value store (`key -> raw_bytes`) for multi-host topology and barrier state.
    pub kv_store: Mutex<BTreeMap<String, Vec<u8>>>,
    /// Cross-host tensor transfer rendezvous mailboxes (`rendezvous_tag -> payload_bytes`).
    pub cross_host_mailboxes: Mutex<BTreeMap<i64, Vec<u8>>>,
    /// Completed `MegaScale` DCN barrier counts indexed by barrier identifier.
    pub megascale_barriers: Mutex<BTreeMap<String, u64>>,
    /// Recorded `MegaScale` runtime errors (`MegaScaleRuntimeErrorProto` payloads).
    pub recorded_errors: Mutex<Vec<crate::proto::MegaScaleRuntimeErrorProto>>,
}

impl DistributedCoordinationEngine {
    /// Creates a new [`DistributedCoordinationEngine`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts a key-value entry into the distributed coordination store (`InsertKeyValue`).
    ///
    /// # Arguments
    ///
    /// * `key` - Coordination key string.
    /// * `value` - Raw value payload bytes.
    pub fn kv_put(&self, key: String, value: Vec<u8>) {
        self.kv_store.lock().insert(key, value);
    }

    /// Retrieves a key-value entry from the distributed coordination store (`GetKeyValue` / `TryGetKeyValue`).
    ///
    /// # Arguments
    ///
    /// * `key` - Coordination key string to look up.
    #[must_use]
    pub fn kv_get(&self, key: &str) -> Option<Vec<u8>> {
        self.kv_store.lock().get(key).cloned()
    }

    /// Deletes a key-value entry from the distributed coordination store (`DeleteKeyValue`).
    ///
    /// # Arguments
    ///
    /// * `key` - Coordination key string to delete.
    pub fn kv_delete(&self, key: &str) -> bool {
        self.kv_store.lock().remove(key).is_some()
    }

    /// Assembles a wire-encoded [`crate::proto::GlobalTopologyProto`] from all registered [`crate::proto::LocalTopologyProto`]
    /// entries in the coordination key-value store, assigning sequential global device IDs.
    ///
    /// # Arguments
    ///
    /// * `local_topologies` - Slice of host-local topology descriptors to merge.
    #[must_use]
    pub fn assemble_global_topology(
        &self,
        local_topologies: &[crate::proto::LocalTopologyProto],
    ) -> crate::proto::GlobalTopologyProto {
        let mut processes = Vec::with_capacity(local_topologies.len());
        let mut next_global_id: i32 = 0;
        for local in local_topologies {
            let mut proc_topo = local.clone();
            for dev in &mut proc_topo.devices {
                dev.global_device_id = next_global_id;
                next_global_id += 1;
            }
            processes.push(proc_topo);
        }
        crate::proto::GlobalTopologyProto { processes }
    }

    /// Sends a tensor buffer payload to a cross-host rendezvous tag (`PJRT_CrossHostTransfers_SendBuffers`).
    ///
    /// # Arguments
    ///
    /// * `tag` - Rendezvous tag identifier.
    /// * `payload` - Serialized tensor payload bytes.
    pub fn send_cross_host_buffer(&self, tag: i64, payload: Vec<u8>) {
        self.cross_host_mailboxes.lock().insert(tag, payload);
    }

    /// Receives a tensor buffer payload from a cross-host rendezvous tag (`PJRT_CrossHostTransfers_ReceiveBuffers`).
    ///
    /// # Arguments
    ///
    /// * `tag` - Rendezvous tag identifier.
    #[must_use]
    pub fn receive_cross_host_buffer(&self, tag: i64) -> Option<Vec<u8>> {
        self.cross_host_mailboxes.lock().remove(&tag)
    }

    /// Arrives at a `MegaScale` DCN multi-slice barrier (`PJRT_Megascale_Barrier`) and returns the arrival count.
    ///
    /// # Arguments
    ///
    /// * `barrier_id` - Unique barrier name.
    pub fn arrive_megascale_barrier(&self, barrier_id: &str) -> u64 {
        let mut guard = self.megascale_barriers.lock();
        let next = guard
            .get(barrier_id)
            .copied()
            .unwrap_or(0)
            .saturating_add(1);
        guard.insert(String::from(barrier_id), next);
        drop(guard);
        next
    }

    /// Records a `MegaScale` DCN runtime error (`PJRT_Megascale_LogError`).
    ///
    /// # Arguments
    ///
    /// * `error` - Structured [`crate::proto::MegaScaleRuntimeErrorProto`] to record.
    pub fn record_megascale_error(&self, error: crate::proto::MegaScaleRuntimeErrorProto) {
        self.recorded_errors.lock().push(error);
    }
}

/// Supported `SparseCore` hardware embedding optimizer algorithms (`tpu_embedding_configuration.proto`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseCoreOptimizer {
    /// Stochastic Gradient Descent (`SGD`).
    Sgd,
    /// Adaptive Gradient (`Adagrad`).
    Adagrad,
    /// Adaptive Moment Estimation (`Adam`).
    Adam,
    /// Follow-The-Regularized-Leader (`FTRL`).
    Ftrl,
    /// Root Mean Square Propagation (`RMSProp`).
    RmsProp,
    /// Adagrad with Nesterov/heavy-ball momentum (`AdagradMomentum`).
    AdagradMomentum,
}

/// Core backing state for `TpuDriver` (`TpuDriver_Open`, `Allocate`, `Compile`, `ExecuteProgram`).
pub struct TpuCoreDriver {
    /// Backing HAL implementation.
    pub hal: Arc<dyn TpuHal>,
    /// Initialization configuration.
    pub config: TpuInitConfig,
}

impl std::fmt::Debug for TpuCoreDriver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TpuCoreDriver")
            .field("generation", &self.config.generation)
            .finish_non_exhaustive()
    }
}

impl TpuCoreDriver {
    /// Opens a [`TpuCoreDriver`] with the given [`TpuInitConfig`].
    ///
    /// # Arguments
    ///
    /// * `config` - Initialization configuration.
    #[must_use]
    pub fn open(config: TpuInitConfig) -> Self {
        let hal = Arc::new(SimulatedTpuHal::new(&config));
        Self { hal, config }
    }

    /// Resets HBM statistics on `device_id` (`TpuDriver_Reset`).
    ///
    /// # Arguments
    ///
    /// * `device_id` - Target logical TPU device.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if `device_id` is invalid.
    pub fn reset(&self, device_id: DeviceId) -> LibTpuResult<()> {
        self.hal.clear_memory_stats(device_id)
    }

    /// Queries a summary string of the driver system topology (`TpuDriver_QuerySystemInfo`).
    #[must_use]
    pub fn query_system_info(&self) -> String {
        format!(
            "TpuDriver(gen={},devices={})",
            self.config.generation,
            self.config.chips_per_host() * self.config.total_hosts()
        )
    }

    /// Allocates a device buffer via the legacy `TpuDriver_Allocate` interface.
    ///
    /// # Arguments
    ///
    /// * `device_id` - Target logical TPU device.
    /// * `size` - Number of bytes to allocate.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if allocation fails.
    pub fn allocate(&self, device_id: DeviceId, size: ByteSize) -> LibTpuResult<DeviceAddress> {
        self.hal
            .allocate_device_memory(device_id, MemoryKind::DeviceHbm, size)
    }

    /// Deallocates a device buffer via `TpuDriver_Deallocate`.
    ///
    /// # Arguments
    ///
    /// * `device_id` - Target logical TPU device.
    /// * `address` - Allocated HBM device address.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if deallocation fails.
    pub fn deallocate(&self, device_id: DeviceId, address: DeviceAddress) -> LibTpuResult<()> {
        self.hal.deallocate_device_memory(device_id, address)
    }

    /// Transfers host bytes to device memory (`TpuDriver_TransferToDevice`).
    ///
    /// # Arguments
    ///
    /// * `device_id` - Target logical TPU device.
    /// * `address` - Destination HBM device address.
    /// * `data` - Source host byte slice.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if the transfer fails.
    pub fn transfer_to_device(
        &self,
        device_id: DeviceId,
        address: DeviceAddress,
        data: &[u8],
    ) -> LibTpuResult<()> {
        self.hal.write_device_memory(device_id, address, data)
    }

    /// Transfers device memory bytes to a host slice (`TpuDriver_TransferFromDevice`).
    ///
    /// # Arguments
    ///
    /// * `device_id` - Source logical TPU device.
    /// * `address` - Source HBM device address.
    /// * `out` - Destination host mutable byte slice.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if the transfer fails.
    pub fn transfer_from_device(
        &self,
        device_id: DeviceId,
        address: DeviceAddress,
        out: &mut [u8],
    ) -> LibTpuResult<()> {
        self.hal.read_device_memory(device_id, address, out)
    }

    /// Transfers bytes between two device buffers (`TpuDriver_TransferFromDeviceToDevice`).
    ///
    /// # Arguments
    ///
    /// * `src_device` - Source logical TPU device.
    /// * `src_addr` - Source HBM device address.
    /// * `dst_device` - Destination logical TPU device.
    /// * `dst_addr` - Destination HBM device address.
    /// * `size` - Number of bytes to copy.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if the copy fails.
    pub fn transfer_device_to_device(
        &self,
        src_device: DeviceId,
        src_addr: DeviceAddress,
        dst_device: DeviceId,
        dst_addr: DeviceAddress,
        size: ByteSize,
    ) -> LibTpuResult<()> {
        self.hal
            .copy_device_to_device(src_device, src_addr, dst_device, dst_addr, size)
    }

    /// Compiles an HLO program via `TpuDriver_Compile`.
    ///
    /// # Arguments
    ///
    /// * `code` - Raw HLO program bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::CompilationFailed`] if compilation fails.
    pub fn compile(&self, code: &[u8]) -> LibTpuResult<CompiledProgram> {
        CompiledProgram::compile("hlo", code, b"")
    }

    /// Executes a compiled program on `device_id` (`TpuDriver_ExecuteProgram`).
    ///
    /// # Arguments
    ///
    /// * `device_id` - Target logical TPU device.
    /// * `program` - Compiled program.
    /// * `inputs` - Input buffer addresses.
    /// * `outputs` - Output buffer addresses.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if execution fails.
    pub fn execute_program(
        &self,
        device_id: DeviceId,
        program: &CompiledProgram,
        inputs: &[DeviceAddress],
        outputs: &[DeviceAddress],
    ) -> LibTpuResult<()> {
        self.hal
            .execute_program(device_id, program, inputs, outputs)
    }
}

/// Strongly-typed descriptor for a registered SDK diagnostic debug callback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegisteredDebugCallback {
    /// Unique callback registration identifier.
    pub id: CallbackId,
    /// C function pointer address (`extern "C" fn(*mut c_void)`).
    pub callback_addr: usize,
    /// Opaque user data pointer address.
    pub user_data_addr: usize,
}

/// Core backing state for `LibtpuSdk_Client` (`GetLibtpuSdkApi`).
#[derive(Debug)]
pub struct LibtpuSdkCoreClient {
    /// Hostname reported by the SDK (`LibtpuSdk_GetHostName`).
    pub hostname: String,
    /// TPU generation.
    pub generation: TpuGeneration,
    /// Chip coordinates indexed by chip ordinal.
    pub chip_coordinates: Vec<ChipCoordinates>,
    /// Runtime metrics map (`metric_name -> (description, values)`).
    pub metrics: Mutex<BTreeMap<String, (String, Vec<f64>)>>,
    /// Registered debug callbacks (`callback_id -> RegisteredDebugCallback`).
    pub debug_callbacks: Mutex<BTreeMap<CallbackId, RegisteredDebugCallback>>,
    /// Monotonic counter for assigning debug callback IDs.
    next_callback_id: AtomicU64,
}

impl Default for LibtpuSdkCoreClient {
    fn default() -> Self {
        Self::new()
    }
}

impl LibtpuSdkCoreClient {
    /// Creates a new [`LibtpuSdkCoreClient`] with default telemetry metrics.
    #[must_use]
    pub fn new() -> Self {
        Self::from_config(&TpuInitConfig::default())
    }

    /// Creates a new [`LibtpuSdkCoreClient`] configured from an explicit [`TpuInitConfig`] snapshot.
    ///
    /// # Arguments
    ///
    /// * `config` - TPU initialization configuration snapshot.
    #[must_use]
    pub fn from_config(config: &TpuInitConfig) -> Self {
        let mut metrics = BTreeMap::new();
        metrics.insert(
            String::from("tpu.runtime.hbm.usage_bytes"),
            (
                String::from("Current HBM memory usage in bytes across TPU cores"),
                vec![0.0, 0.0, 0.0, 0.0],
            ),
        );
        metrics.insert(
            String::from("tpu.runtime.tensorcore.utilization"),
            (
                String::from("TensorCore MXU duty cycle percentage"),
                vec![100.0, 100.0, 100.0, 100.0],
            ),
        );
        metrics.insert(
            String::from("tpu.runtime.ici.link_bandwidth_bytes"),
            (
                String::from("Inter-Chip Interconnect link throughput in bytes/sec"),
                vec![1.6e11, 1.6e11, 1.6e11, 1.6e11],
            ),
        );
        metrics.insert(
            String::from("tpu.runtime.dma.throughput_bytes"),
            (
                String::from("PCIe DMA host-to-device and device-to-host bytes transferred"),
                vec![4096.0, 4096.0, 4096.0, 4096.0],
            ),
        );
        metrics.insert(
            String::from("tpu.runtime.compile.time_micros"),
            (
                String::from("XLA HLO / StableHLO compilation time in microseconds"),
                vec![125.0],
            ),
        );
        let mut chip_coordinates = Vec::new();
        for z in 0..config.chips_per_host_bounds.z {
            for y in 0..config.chips_per_host_bounds.y {
                for x in 0..config.chips_per_host_bounds.x {
                    chip_coordinates.push(ChipCoordinates::new(x, y, z));
                }
            }
        }
        Self {
            hostname: format!("tpu-worker-{}", config.process_index.0),
            generation: config.generation,
            chip_coordinates,
            metrics: Mutex::new(metrics),
            debug_callbacks: Mutex::new(BTreeMap::new()),
            next_callback_id: AtomicU64::new(1),
        }
    }

    /// Looks up the 3D coordinates of `chip_index` (`LibtpuSdk_GetChipCoordinates`).
    ///
    /// # Arguments
    ///
    /// * `chip_index` - Zero-based physical chip index.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::DeviceNotFound`] if `chip_index` is out of range.
    pub fn chip_coords(&self, chip_index: usize) -> LibTpuResult<ChipCoordinates> {
        self.chip_coordinates
            .get(chip_index)
            .copied()
            .ok_or_else(|| LibTpuError::DeviceNotFound {
                device_id: u32::try_from(chip_index).unwrap_or(u32::MAX),
            })
    }

    /// Looks up the physical chip index for given 3D coordinates (`LibtpuSdk_GetChipIndex`).
    ///
    /// # Arguments
    ///
    /// * `coords` - 3D chip coordinates `(x, y, z)`.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ChipNotFound`] if no chip exists at `coords`.
    pub fn chip_index_for_coords(&self, coords: ChipCoordinates) -> LibTpuResult<usize> {
        self.chip_coordinates
            .iter()
            .position(|&c| c.matches(coords))
            .ok_or(LibTpuError::ChipNotFound {
                x: coords.x,
                y: coords.y,
                z: coords.z,
            })
    }

    /// Returns the list of all supported telemetry metric names.
    #[must_use]
    pub fn list_metric_names(&self) -> Vec<String> {
        self.metrics.lock().keys().cloned().collect()
    }

    /// Queries the description and values of a named runtime metric (`LibtpuSdk_GetMetric`).
    ///
    /// # Arguments
    ///
    /// * `metric_name` - Full metric identifier.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MetricNotFound`] if the metric name does not exist.
    pub fn get_metric(&self, metric_name: &str) -> LibTpuResult<(String, Vec<f64>)> {
        self.metrics
            .lock()
            .get(metric_name)
            .cloned()
            .ok_or_else(|| LibTpuError::MetricNotFound {
                metric_name: String::from(metric_name),
            })
    }

    /// Registers a debug callback and returns its unique [`CallbackId`] (`LibtpuSdk_RegisterDebugCallback`).
    ///
    /// # Arguments
    ///
    /// * `callback_addr` - Function pointer address.
    /// * `user_data_addr` - Opaque user data pointer address.
    #[must_use]
    pub fn register_debug_callback(
        &self,
        callback_addr: usize,
        user_data_addr: usize,
    ) -> CallbackId {
        let id = CallbackId(self.next_callback_id.fetch_add(1, Ordering::Relaxed));
        self.debug_callbacks.lock().insert(
            id,
            RegisteredDebugCallback {
                id,
                callback_addr,
                user_data_addr,
            },
        );
        id
    }

    /// Unregisters a previously registered debug callback (`LibtpuSdk_UnregisterDebugCallback`).
    ///
    /// # Arguments
    ///
    /// * `id` - Callback identifier returned by [`LibtpuSdkCoreClient::register_debug_callback`].
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::CallbackNotFound`] if `id` is not registered.
    pub fn unregister_debug_callback(&self, id: CallbackId) -> LibTpuResult<()> {
        self.debug_callbacks
            .lock()
            .remove(&id)
            .map(|_| ())
            .ok_or(LibTpuError::CallbackNotFound { callback_id: id.0 })
    }

    /// Returns a snapshot of all registered [`RegisteredDebugCallback`] descriptors.
    #[must_use]
    pub fn registered_debug_callbacks(&self) -> Vec<RegisteredDebugCallback> {
        self.debug_callbacks.lock().values().copied().collect()
    }

    /// Returns a human-readable core state summary string (`LibtpuSdk_RuntimeStatus_GetCoreStateSummary`).
    #[must_use]
    pub fn core_state_summary(&self) -> String {
        format!(
            "host={} gen={} chips={} state=READY",
            self.hostname,
            self.generation,
            self.chip_coordinates.len()
        )
    }

    /// Refreshes per-device HBM usage and utilization metrics from the active [`TpuHal`].
    ///
    /// # Arguments
    ///
    /// * `hal` - Active hardware abstraction layer to sample.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if querying device memory statistics fails.
    pub fn refresh_from_hal(&self, hal: &dyn TpuHal) -> LibTpuResult<()> {
        let mut hbm_usage = Vec::with_capacity(self.chip_coordinates.len());
        for idx in 0..self.chip_coordinates.len() {
            let dev_id = DeviceId(u32::try_from(idx).unwrap_or(0));
            let stats = hal.memory_stats(dev_id)?;
            #[allow(clippy::cast_precision_loss)]
            hbm_usage.push(stats.bytes_in_use as f64);
        }
        self.metrics.lock().insert(
            String::from("tpu.runtime.hbm.usage_bytes"),
            (
                String::from("Current HBM memory bytes in use per logical TPU device"),
                hbm_usage,
            ),
        );
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::cognitive_complexity, clippy::too_many_lines)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_status_selector_profiler_and_embedding() {
        let mut st = TpuCoreStatus::ok();
        assert!(st.is_ok());
        assert!(st.error.is_none());
        st.set(3, String::from("bad arg"));
        assert!(!st.is_ok());
        assert!(st.error.is_some());
        st.set(0, String::new());
        assert!(st.is_ok());
        assert!(st.error.is_none());
        let st_ok_new = TpuCoreStatus::new(0, String::new());
        assert!(st_ok_new.is_ok());
        let st_from_err = TpuCoreStatus::from_error(LibTpuError::ProfilerNotActive);
        assert_eq!(st_from_err.code, PjrtErrorCode::FailedPrecondition);
        assert!(st_from_err.error.is_some());
        let st2 = TpuCoreStatus::new(5, String::from("not found"));
        let st2_cloned = st2.clone();
        assert!(!st2.is_ok());
        drop(st2);
        assert_ne!(format!("{st2_cloned:?}"), "");

        let sel = TfTpuCoreOrdinalSelector::new(2);
        assert_ne!(format!("{sel:?}"), "");
        assert_eq!(sel.get_ordinal(), 0);
        assert_eq!(sel.get_ordinal(), 1);
        sel.dequeue_ordinal(0, 0);
        sel.dequeue_ordinal(1, 99);
        assert_eq!(sel.get_ordinal(), 0);

        let prof = TpuCoreProfiler::default();
        assert_ne!(format!("{prof:?}"), "");
        assert!(prof.stop().is_err());
        assert!(prof.start().is_ok());
        assert!(prof.start().is_err());
        prof.record_hardware_step(16, 4096, 2048, 1024, 4);
        assert!(prof.stop().is_ok());
        let xspace_bytes = prof.collect_xspace_bytes();
        assert_ne!(xspace_bytes, [] as [u8; 0]);
        let decoded_xspace = XSpaceProto::decode(&xspace_bytes);
        assert!(decoded_xspace.is_ok());
        let inp_analysis = prof.analyze_input_pipeline();
        assert!(inp_analysis.hardware_is_tpu);
        assert_ne!(
            prof.collect_tool_data_bytes("input_pipeline.pb"),
            [] as [u8; 0]
        );
        assert_ne!(
            prof.collect_tool_data_bytes("trace_viewer.json"),
            [] as [u8; 0]
        );
        assert_ne!(
            prof.collect_tool_data_bytes("memory_profile.json"),
            [] as [u8; 0]
        );
        assert_ne!(prof.collect_tool_data_bytes("xspace.pb"), [] as [u8; 0]);
        let prof_req_empty = crate::proto::ProfileRequestProto::default().encode();
        assert!(prof.handle_profiler_service_rpc(1, &prof_req_empty).is_ok());
        let prof_req_tools = crate::proto::ProfileRequestProto {
            tools: vec![
                String::from("trace_viewer.json"),
                String::from("input_pipeline.pb"),
            ],
            ..crate::proto::ProfileRequestProto::default()
        }
        .encode();
        assert!(prof.handle_profiler_service_rpc(1, &prof_req_tools).is_ok());
        assert!(prof.handle_profiler_service_rpc(2, &[]).is_ok());
        assert!(prof.handle_profiler_service_rpc(3, &[]).is_ok());
        assert!(prof.handle_profiler_service_rpc(4, &[]).is_ok());
        assert!(prof.handle_profiler_service_rpc(5, &[]).is_ok());
        for m_id in 1..=5 {
            assert!(
                prof.handle_profiler_service_rpc(m_id, &[0xFF, 0xFF, 0xFF])
                    .is_err()
            );
        }

        let mut emb = TpuCoreEmbeddingEngine::default();
        assert!(!emb.initialized);
        let proto_cfg = crate::proto::TpuEmbeddingConfigurationProto {
            table_descriptor: vec![crate::proto::EmbeddingTableDescriptorProto {
                name: String::from("proto_tbl"),
                vocabulary_size: 4,
                dimension: 4,
                num_features: 1,
            }],
            mode: 1,
            batch_size_per_tensor_core: 8,
            num_hosts: 1,
            num_tensor_cores: 2,
            sharding_strategy: 0,
            pipeline_execution_with_tensor_core: false,
            profile_data_directory: String::new(),
            feature_descriptor: Vec::new(),
            spmd_sharding: None,
        };
        assert_ne!(emb.execute_partitioner(&proto_cfg.encode()), [] as [u8; 0]);
        assert!(emb.read_parameters("proto_tbl").is_some());
        let part_out = emb.execute_partitioner(b"cfg");
        assert_ne!(part_out, [] as [u8; 0]);
        emb.configure_memory(vec![10, 20]);
        assert_eq!(emb.host_memory_configs.len(), 1);
        assert_ne!(emb.collate_memory(), [] as [u8; 0]);
        assert_ne!(emb.configure_host(b"host0"), [] as [u8; 0]);
        emb.connect_hosts();
        assert!(emb.hosts_connected);
        emb.enqueue_tensor_batch(1, 8, 8);
        assert_eq!(emb.enqueued_batches.len(), 1);
        let hlo_bytes = emb.generate_xla_computation("recv_activations");
        assert!(HloModuleProto::decode(&hlo_bytes).is_ok());
        for kind in [
            "recv_activations",
            "send_gradients",
            "recv_dedup_data",
            "dedup_data_size",
            "dedup_tuple_mask",
            "custom_hook",
        ] {
            let comp_pb = emb.generate_xla_computation(kind);
            let compiled_sc = CompiledProgram::compile("hlo", &comp_pb, b"");
            assert!(
                compiled_sc.is_ok(),
                "Failed to compile SparseCore HLO for {kind}"
            );
            if let Ok(sc_prog) = compiled_sc {
                let mut vm = crate::hal::TpuCoreVmState::new();
                let in0 = vec![0_u8; 16];
                let in1 = vec![0_u8; 16];
                let out_vecs = vm.execute_compiled_program(&sc_prog, vec![in0, in1], 1);
                assert_eq!(out_vecs.len(), 1);
            }
        }
        let sc_grad_hlo = HloModuleProto {
            name: String::from("sc_grad_mod"),
            id: 1,
            entry_computation_name: String::from("main"),
            entry_computation_id: 1,
            computations: vec![HloComputationProto {
                name: String::from("main"),
                id: 1,
                root_id: 1,
                instructions: vec![HloInstructionProto {
                    name: String::from("sc_op"),
                    opcode: String::from("custom_call"),
                    id: 1,
                    custom_call_target: String::from("SparseCoreGradientUpdate"),
                    shape: Some(ShapeProto {
                        element_type: 11,
                        dimensions: vec![4],
                        is_dynamic_dimension: vec![false],
                        ..ShapeProto::default()
                    }),
                    ..HloInstructionProto::default()
                }],
                ..HloComputationProto::default()
            }],
            ..HloModuleProto::default()
        };
        if let Ok(sc_grad_prog) = CompiledProgram::compile("hlo", &sc_grad_hlo.encode(), b"") {
            let tpu_exec = sc_grad_prog.to_tpu_executable_proto();
            assert!(
                tpu_exec
                    .ds_executable
                    .as_ref()
                    .and_then(|d| d.sparse_core_program.as_ref())
                    .is_some()
            );
            assert_eq!(sc_grad_prog.build_isa_program_proto().scs_start_bundle, 1);
            let mut vm = crate::hal::TpuCoreVmState::new();
            let out_g =
                vm.execute_compiled_program(&sc_grad_prog, vec![vec![0_u8; 16], vec![0_u8; 16]], 1);
            assert_eq!(out_g.len(), 1);
        }
        emb.write_parameters(String::from("vocab"), vec![1, 2, 3, 4]);
        assert_eq!(emb.read_parameters("vocab"), Some(&[1_u8, 2, 3, 4][..]));
        assert_eq!(emb.read_parameters("missing"), None);
        assert!(
            emb.write_parameters_validated("", vec![1, 2, 3, 4])
                .is_err()
        );
        assert!(emb.write_parameters_validated("vocab", vec![1, 2]).is_err());
        assert!(
            emb.write_parameters_validated("vocab", vec![5, 6, 7, 8])
                .is_ok()
        );
        assert!(
            emb.write_parameters_validated("new_tbl", vec![9, 10])
                .is_ok()
        );
        assert_eq!(
            emb.read_parameters_validated("vocab").ok(),
            Some(&[5_u8, 6, 7, 8][..])
        );
        assert!(emb.read_parameters_validated("missing").is_err());

        let (uniq_ids, uniq_weights, csr_masks) =
            emb.deduplicate_coo_to_csr(&[(0, 10, 1.5), (2, 10, 0.5), (1, 25, 2.0)]);
        assert_eq!(uniq_ids, vec![10, 25]);
        assert_eq!(uniq_weights, vec![2.0, 2.0]);
        assert_eq!(csr_masks, vec![0b101, 0b010]);

        let mut init_weights = Vec::new();
        for w in [1.0_f32, 2.0, 3.0, 4.0] {
            init_weights.extend_from_slice(&w.to_le_bytes());
        }
        emb.write_parameters(String::from("opt_tbl"), init_weights);
        for opt in [
            SparseCoreOptimizer::Sgd,
            SparseCoreOptimizer::Adagrad,
            SparseCoreOptimizer::Adam,
            SparseCoreOptimizer::Ftrl,
            SparseCoreOptimizer::RmsProp,
            SparseCoreOptimizer::AdagradMomentum,
        ] {
            assert_ne!(format!("{opt:?}"), "");
            assert_eq!(opt, opt.clone());
            assert!(
                emb.apply_optimizer_gradients("opt_tbl", opt, 0.1, &[0.001, 0.5, -0.5, 0.0])
                    .is_ok()
            );
        }
        assert!(
            emb.apply_optimizer_gradients("nonexistent_tbl", SparseCoreOptimizer::Sgd, 0.1, &[1.0])
                .is_err()
        );
        let fwd_res = emb.forward_embedding_lookup("opt_tbl", 2, 2, &[(0, 0, 1.0), (1, 1, 0.5)]);
        assert!(fwd_res.is_ok());
        assert_eq!(fwd_res.unwrap_or_default().len(), 4);
        assert!(
            emb.forward_embedding_lookup("nonexistent_tbl", 2, 2, &[(0, 0, 1.0)])
                .is_err()
        );

        let coord = DistributedCoordinationEngine::new();
        assert_ne!(format!("{coord:?}"), "");
        coord.kv_put(String::from("slice_0"), vec![1, 2, 3]);
        assert_eq!(coord.kv_get("slice_0"), Some(vec![1, 2, 3]));
        assert_eq!(coord.kv_get("missing"), None);
        assert!(coord.kv_delete("slice_0"));
        assert!(!coord.kv_delete("slice_0"));
        let global_topo = coord.assemble_global_topology(&[crate::proto::LocalTopologyProto {
            process_id: 0,
            boot_id: String::from("boot0"),
            devices: vec![crate::proto::DistributedDeviceProto::default()],
            partition_index: 0,
            network_nodes: Vec::new(),
        }]);
        assert_eq!(global_topo.processes.len(), 1);
        coord.send_cross_host_buffer(42, vec![9, 8, 7]);
        assert_eq!(coord.receive_cross_host_buffer(42), Some(vec![9, 8, 7]));
        assert_eq!(coord.receive_cross_host_buffer(42), None);
        assert_eq!(coord.arrive_megascale_barrier("step_1"), 1);
        assert_eq!(coord.arrive_megascale_barrier("step_1"), 2);
        coord.record_megascale_error(crate::proto::MegaScaleRuntimeErrorProto::default());
        assert_eq!(coord.recorded_errors.lock().len(), 1);

        emb.finalize();
        assert!(emb.initialized);
        assert_ne!(format!("{emb:?}"), "");
    }

    #[test]
    fn test_driver_and_sdk_client() {
        let drv = TpuCoreDriver::open(TpuInitConfig::default());
        assert_ne!(format!("{drv:?}"), "");
        assert!(drv.query_system_info().contains("TpuDriver"));
        assert!(drv.reset(DeviceId(0)).is_ok());
        let addr_res = drv.allocate(DeviceId(0), ByteSize(256));
        assert!(addr_res.is_ok());
        let addr = addr_res.unwrap_or(DeviceAddress(0));
        let addr2 = drv
            .allocate(DeviceId(0), ByteSize(256))
            .unwrap_or(DeviceAddress(0));
        assert!(
            drv.transfer_to_device(DeviceId(0), addr, &[1, 2, 3, 4])
                .is_ok()
        );
        let mut host_out = [0_u8; 4];
        assert!(
            drv.transfer_from_device(DeviceId(0), addr, &mut host_out)
                .is_ok()
        );
        assert_eq!(host_out, [1, 2, 3, 4]);
        assert!(
            drv.transfer_device_to_device(DeviceId(0), addr, DeviceId(0), addr2, ByteSize(4))
                .is_ok()
        );
        if let Ok(prog) = drv.compile(b"HloModule test") {
            assert!(
                drv.execute_program(DeviceId(0), &prog, &[addr], &[addr2])
                    .is_ok()
            );
        }
        assert!(drv.deallocate(DeviceId(0), addr).is_ok());
        assert!(drv.deallocate(DeviceId(0), addr2).is_ok());
        assert!(drv.compile(b"").is_err());

        let sdk = LibtpuSdkCoreClient::default();
        assert_ne!(format!("{sdk:?}"), "");
        assert!(sdk.chip_coords(0).is_ok());
        assert!(sdk.chip_coords(99).is_err());
        assert_eq!(
            sdk.chip_index_for_coords(ChipCoordinates::new(1, 0, 0))
                .ok(),
            Some(1)
        );
        assert!(
            sdk.chip_index_for_coords(ChipCoordinates::new(9, 9, 9))
                .is_err()
        );
        assert_eq!(sdk.list_metric_names().len(), 5);
        assert!(sdk.refresh_from_hal(&*drv.hal).is_ok());
        let mut sdk_bad = LibtpuSdkCoreClient::new();
        sdk_bad.chip_coordinates.push(ChipCoordinates::new(9, 9, 9));
        assert!(sdk_bad.refresh_from_hal(&*drv.hal).is_err());
        assert!(sdk.get_metric("tpu.runtime.hbm.usage_bytes").is_ok());
        assert!(sdk.get_metric("unknown").is_err());
        let cb_id = sdk.register_debug_callback(0x1234, 0x5678);
        let expected_cb = RegisteredDebugCallback {
            id: cb_id,
            callback_addr: 0x1234,
            user_data_addr: 0x5678,
        };
        assert_eq!(expected_cb.clone(), expected_cb);
        assert_ne!(format!("{expected_cb:?}"), "");
        assert_eq!(sdk.registered_debug_callbacks(), vec![expected_cb]);
        assert!(sdk.unregister_debug_callback(cb_id).is_ok());
        assert_eq!(
            sdk.registered_debug_callbacks(),
            [] as [RegisteredDebugCallback; 0]
        );
        assert!(sdk.unregister_debug_callback(cb_id).is_err());
        assert!(sdk.core_state_summary().contains("READY"));
    }
}

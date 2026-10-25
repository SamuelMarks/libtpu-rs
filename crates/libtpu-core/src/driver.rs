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
}

impl TpuCoreStatus {
    /// Creates an OK [`TpuCoreStatus`] (`TpuStatus_New`).
    #[must_use]
    pub const fn ok() -> Self {
        Self {
            code: PjrtErrorCode::Ok,
            message: String::new(),
        }
    }

    /// Creates a [`TpuCoreStatus`] with an explicit raw code and message (`TpuStatus_Create`).
    ///
    /// # Arguments
    ///
    /// * `raw_code` - Integer status code.
    /// * `message` - Error message string.
    #[must_use]
    pub const fn new(raw_code: i32, message: String) -> Self {
        Self {
            code: PjrtErrorCode::from_raw(raw_code),
            message,
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
        }
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

    /// Serializes collected `tensorflow.profiler.XSpace` profiling data into wire-compatible protobuf bytes
    /// (`TpuProfiler_CollectData`).
    #[must_use]
    pub fn collect_xspace_bytes(&self) -> Vec<u8> {
        let count = i64::from(self.recorded_events.load(Ordering::Relaxed));
        let event = XEventProto {
            metadata_id: 1,
            offset_ps: 0,
            duration_ps: 1_000_000,
            stats: vec![XStatProto {
                metadata_id: 1,
                int64_value: count,
                str_value: String::from("tpu_execution"),
            }],
        };
        let line = XLineProto {
            id: 0,
            name: String::from("TensorCore 0"),
            timestamp_ns: 1_000_000_000,
            events: vec![event],
        };
        let plane = XPlaneProto {
            id: 1,
            name: String::from("/device:TPU:0"),
            lines: vec![line],
            stats: vec![XStatProto {
                metadata_id: 2,
                int64_value: count,
                str_value: String::from("recorded_events"),
            }],
        };
        let xspace = XSpaceProto {
            planes: vec![plane],
            hostnames: vec![String::from("tpu-worker-0")],
        };
        xspace.encode()
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

    /// Reads embedding table parameters from HBM (`TpuEmbeddingEngine_ReadParameters`).
    ///
    /// # Arguments
    ///
    /// * `table_name` - Name of the embedding table.
    #[must_use]
    pub fn read_parameters(&self, table_name: &str) -> Option<&[u8]> {
        self.tables.get(table_name).map(Vec::as_slice)
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

    /// Generates a wire-encoded XLA [`HloModuleProto`] for one of the 5 `TpuEmbeddingEngine_*Computation` APIs.
    ///
    /// # Arguments
    ///
    /// * `computation_kind` - Computation identifier (`"recv_activations"`, `"recv_dedup_data"`, `"send_gradients"`, `"dedup_data_size"`, `"dedup_tuple_mask"`).
    #[must_use]
    pub fn generate_xla_computation(&self, computation_kind: &str) -> Vec<u8> {
        let hlo = HloModuleProto {
            name: format!("tpu_embedding_{computation_kind}"),
            id: 1,
            entry_computation_name: String::from("main"),
            entry_computation_id: 1,
            computations: vec![HloComputationProto {
                name: String::from("main"),
                id: 1,
                root_id: 1,
                instructions: vec![HloInstructionProto {
                    name: String::from(computation_kind),
                    opcode: String::from("custom_call"),
                    id: 1,
                    operand_ids: Vec::new(),
                    parameter_number: 0,
                    custom_call_target: String::from(computation_kind),
                    shape: Some(ShapeProto {
                        element_type: 11,
                        dimensions: vec![4],
                        tuple_shapes: Vec::new(),
                        layout: None,
                        is_dynamic_dimension: vec![false],
                    }),
                }],
            }],
        };
        hlo.encode()
    }
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
    /// Registered debug callbacks (`callback_id -> (fn_ptr_addr, user_data_addr)`).
    pub debug_callbacks: Mutex<BTreeMap<CallbackId, (usize, usize)>>,
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
        Self {
            hostname: String::from("tpu-worker-0"),
            generation: TpuGeneration::V5e,
            chip_coordinates: vec![
                ChipCoordinates::new(0, 0, 0),
                ChipCoordinates::new(1, 0, 0),
                ChipCoordinates::new(0, 1, 0),
                ChipCoordinates::new(1, 1, 0),
            ],
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
    /// Returns [`LibTpuError::DeviceNotFound`] if no chip exists at `coords`.
    pub fn chip_index_for_coords(&self, coords: ChipCoordinates) -> LibTpuResult<usize> {
        self.chip_coordinates
            .iter()
            .position(|&c| c.matches(coords))
            .ok_or(LibTpuError::DeviceNotFound {
                device_id: coords.x,
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
        self.debug_callbacks
            .lock()
            .insert(id, (callback_addr, user_data_addr));
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

    /// Returns a snapshot of all registered debug `(callback_addr, user_data_addr)` pairs.
    #[must_use]
    pub fn registered_debug_callbacks(&self) -> Vec<(usize, usize)> {
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
}

#[cfg(test)]
#[allow(for_loops_over_fallibles, clippy::cognitive_complexity)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_status_selector_profiler_and_embedding() {
        let mut st = TpuCoreStatus::ok();
        assert!(st.is_ok());
        st.set(3, String::from("bad arg"));
        assert!(!st.is_ok());
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
        assert!(prof.stop().is_ok());
        let xspace_bytes = prof.collect_xspace_bytes();
        assert_ne!(xspace_bytes, [] as [u8; 0]);
        let decoded_xspace = XSpaceProto::decode(&xspace_bytes);
        assert!(decoded_xspace.is_ok());

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
        emb.write_parameters(String::from("vocab"), vec![1, 2, 3, 4]);
        assert_eq!(emb.read_parameters("vocab"), Some(&[1_u8, 2, 3, 4][..]));
        assert_eq!(emb.read_parameters("missing"), None);
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
        for prog in drv.compile(b"HloModule test") {
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
        assert!(sdk.get_metric("tpu.runtime.hbm.usage_bytes").is_ok());
        assert!(sdk.get_metric("unknown").is_err());
        let cb_id = sdk.register_debug_callback(0x1234, 0x5678);
        assert_eq!(sdk.registered_debug_callbacks(), vec![(0x1234, 0x5678)]);
        assert!(sdk.unregister_debug_callback(cb_id).is_ok());
        assert_eq!(sdk.registered_debug_callbacks(), [] as [(usize, usize); 0]);
        assert!(sdk.unregister_debug_callback(cb_id).is_err());
        assert!(sdk.core_state_summary().contains("READY"));
    }
}

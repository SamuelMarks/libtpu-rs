//! Native `PyO3` (`abi3`) Python FFI bridge for `libtpu-rs` (`libtpu._C` and `libtpu/sdk.so`).
//!
//! Provides [`PyTpuRuntime`], [`PySdkBridge`], `DLPack` v1.0 C-ABI capsules ([`dlpack`]), and native
//! PEP 489 `PyO3` (`abi3`) module entrypoints ([`PyInit__C`], [`PyInit_sdk`], [`PyInit_libtpu_py`])
//! mapping [`libtpu_core::error::LibTpuError`] cleanly to typed Python `TpuError` exceptions.

#![cfg_attr(all(test, coverage_nightly), feature(coverage_attribute))]

pub mod dlpack;
pub mod pyo3_modules;

pub use pyo3_modules::{
    PyInit__C, PyInit_libtpu_py, PyInit_sdk, lib_tpu_error_to_py_err, tpu_error_type,
};

use libtpu_core::config::TpuInitConfig;
use libtpu_core::driver::{LibtpuSdkCoreClient, TpuCoreDriver, TpuCoreProfiler};
use libtpu_core::error::LibTpuResult;
use libtpu_core::pjrt::PjrtCoreClient;
use libtpu_core::runtime_state::{SharedTpuRuntimeState, global_shared_runtime_state};
use libtpu_core::types::{DeviceId, ElementType, MemorySpaceId, Shape, TpuGeneration};
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};

/// Process-wide cache of [`PyTpuRuntime`] instances keyed by `init_args`.
static SHARED_PY_RUNTIMES: OnceLock<Mutex<BTreeMap<String, Arc<PyTpuRuntime>>>> = OnceLock::new();

/// Process-wide cache of [`PySdkBridge`] instances keyed by [`TpuGeneration`] discriminant.
static SHARED_PY_SDK_BRIDGES: OnceLock<Mutex<BTreeMap<i32, Arc<PySdkBridge>>>> = OnceLock::new();

/// High-level Rust session backing the `libtpu` Python package (`PyTpuRuntime`).
#[derive(Debug)]
pub struct PyTpuRuntime {
    /// Active initialization configuration.
    pub config: TpuInitConfig,
    /// Underlying PJRT core client.
    pub client: PjrtCoreClient,
}

impl PyTpuRuntime {
    /// Creates a new [`PyTpuRuntime`] from a `--flag=value` initialization argument string.
    ///
    /// # Arguments
    ///
    /// * `init_args` - Whitespace-separated CLI flags (e.g., `"--tpu_generation=v5e"`).
    ///
    /// # Errors
    ///
    /// Returns [`libtpu_core::error::LibTpuError`] if `init_args` contains invalid flags.
    pub fn from_init_args(init_args: &str) -> LibTpuResult<Self> {
        let mut config = TpuInitConfig::default();
        config.apply_init_args(init_args)?;
        let client = PjrtCoreClient::from_config(&config);
        Ok(Self { config, client })
    }

    /// Returns or initializes the shared process-wide [`Arc<PyTpuRuntime>`] for `init_args`.
    ///
    /// # Arguments
    ///
    /// * `init_args` - Whitespace-separated CLI flags (e.g., `"--tpu_generation=v5e"`).
    ///
    /// # Errors
    ///
    /// Returns [`libtpu_core::error::LibTpuError`] if `init_args` contains invalid flags.
    pub fn shared(init_args: &str) -> LibTpuResult<Arc<Self>> {
        let key = init_args.trim().to_owned();
        let cache = SHARED_PY_RUNTIMES.get_or_init(|| Mutex::new(BTreeMap::new()));
        let cached = cache.lock().get(&key).cloned();
        if let Some(existing) = cached {
            return Ok(existing);
        }
        let created = Arc::new(Self::from_init_args(&key)?);
        cache.lock().insert(key, Arc::clone(&created));
        Ok(created)
    }

    /// Returns a reference to the process-wide [`SharedTpuRuntimeState`].
    #[must_use]
    pub fn runtime_state(&self) -> &'static Arc<SharedTpuRuntimeState> {
        global_shared_runtime_state()
    }

    /// Returns the PJRT API version tuple `(major, minor)` (`(0, 117)`).
    #[must_use]
    pub const fn pjrt_api_version() -> (i32, i32) {
        (0, 117)
    }

    /// Returns the active [`TpuGeneration`] of this runtime.
    #[must_use]
    pub const fn generation(&self) -> TpuGeneration {
        self.config.generation
    }

    /// Returns the number of addressable TPU devices in this runtime.
    #[must_use]
    pub fn device_count(&self) -> usize {
        self.client.addressable_device_indices.len()
    }

    /// Serializes the active TPU topology into wire-compatible `TpuTopologyArgsProto` bytes.
    #[must_use]
    pub fn serialize_topology(&self) -> Vec<u8> {
        self.client.topology.serialize()
    }

    /// Compiles an MLIR/HLO program and returns its serialized `TpuSerializedExecutableProto` bytes.
    ///
    /// # Arguments
    ///
    /// * `format` - Program format (`"mlir"`, `"hlo"`, or `"hlo_with_config"`).
    /// * `code` - Program source or bytecode bytes.
    ///
    /// # Errors
    ///
    /// Returns [`libtpu_core::error::LibTpuError`] if compilation fails.
    pub fn compile_and_serialize(&self, format: &str, code: &[u8]) -> LibTpuResult<Vec<u8>> {
        let loaded = self.client.compile_and_load(format, code, b"")?;
        Ok(loaded.executable.program.serialize())
    }

    /// Deserializes a compiled TPU executable payload and verifies its 32-byte fingerprint.
    ///
    /// # Arguments
    ///
    /// * `serialized` - Serialized executable bytes produced by [`Self::compile_and_serialize`].
    ///
    /// # Errors
    ///
    /// Returns [`libtpu_core::error::LibTpuError`] if deserialization fails.
    pub fn deserialize_executable_fingerprint(&self, serialized: &[u8]) -> LibTpuResult<String> {
        let loaded = self.client.deserialize_and_load(serialized)?;
        Ok(loaded.executable.program.fingerprint)
    }

    /// Inspects a shared library binary image (`ELF` or `Mach-O`) via `goblin`
    /// and returns its parsed [`libtpu_core::elf::BinarySymbolReport`].
    ///
    /// # Arguments
    ///
    /// * `binary_bytes` - Raw shared library binary bytes.
    ///
    /// # Errors
    ///
    /// Returns [`libtpu_core::error::LibTpuError`] if binary parsing fails.
    pub fn inspect_binary_symbol_names(
        binary_bytes: &[u8],
    ) -> LibTpuResult<libtpu_core::elf::BinarySymbolReport> {
        libtpu_core::elf::inspect_binary_symbols(binary_bytes)
    }

    /// Compiles and executes an MLIR or HLO program on `device_id` with an arbitrary multi-dimensional [`Shape`].
    ///
    /// # Arguments
    ///
    /// * `device_id` - Target logical TPU device ID.
    /// * `format` - Program format (`"mlir"`, `"hlo"`, or `"hlo_with_config"`).
    /// * `code` - Program source or bytecode bytes.
    /// * `shape` - Input tensor [`Shape`].
    /// * `inputs` - Slice of raw byte slices for each input operand.
    ///
    /// # Errors
    ///
    /// Returns [`libtpu_core::error::LibTpuError`] if compilation, allocation, or execution fails.
    pub fn execute_shaped_tensor(
        &self,
        device_id: DeviceId,
        format: &str,
        code: &[u8],
        shape: &Shape,
        inputs: &[&[u8]],
    ) -> LibTpuResult<Vec<Vec<u8>>> {
        let loaded = self.client.compile_and_load(format, code, b"")?;
        let mut device_buffers = Vec::with_capacity(inputs.len());
        for &in_bytes in inputs {
            device_buffers.push(self.client.buffer_from_host(
                device_id,
                shape.clone(),
                in_bytes,
            )?);
        }
        let buf_refs: Vec<_> = device_buffers.iter().collect();
        let outs = loaded.execute_on_device(device_id, MemorySpaceId(0), &buf_refs)?;
        outs.into_iter().map(|b| b.to_host_bytes()).collect()
    }

    /// Compiles and executes an MLIR program on `device_id` doubling a 4-element `f32` input.
    ///
    /// # Arguments
    ///
    /// * `device_id` - Target logical TPU device ID.
    /// * `mlir_code` - MLIR source bytes.
    /// * `input` - Input 4-element `f32` array.
    ///
    /// # Errors
    ///
    /// Returns [`libtpu_core::error::LibTpuError`] if `device_id` is invalid, poisoned, or `mlir_code` fails to compile.
    pub fn execute_f32_4(
        &self,
        device_id: DeviceId,
        mlir_code: &[u8],
        input: &[f32; 4],
    ) -> LibTpuResult<[f32; 4]> {
        let shape = Shape {
            element_type: ElementType::F32,
            dims: vec![4],
            dynamic_dims: vec![false],
            layout: libtpu_core::types::MemoryLayout::default_tpu_tiled(1),
        };
        let b_in: Vec<u8> = input.iter().flat_map(|&v| v.to_le_bytes()).collect();
        let host_outs =
            self.execute_shaped_tensor(device_id, "mlir", mlir_code, &shape, &[&b_in, &b_in])?;
        let mut out_arr = [0.0_f32; 4];
        for raw in host_outs {
            for (dst, chunk) in out_arr.iter_mut().zip(raw.chunks_exact(4)) {
                let mut bytes = [0_u8; 4];
                bytes.copy_from_slice(chunk);
                *dst = f32::from_le_bytes(bytes);
            }
        }
        Ok(out_arr)
    }

    /// Encodes an MLIR source module into a binary `ML\xefR` bytecode container, compiles the
    /// bytecode container on `self.client`, and executes it on `device_id` with `inputs`.
    ///
    /// # Arguments
    ///
    /// * `device_id` - Target logical TPU device ID.
    /// * `mlir_source` - Textual MLIR module string to encode into `ML\xefR` bytecode.
    /// * `shape` - Input/output tensor [`Shape`].
    /// * `inputs` - Slice of raw byte slices for each input operand.
    ///
    /// # Errors
    ///
    /// Returns [`libtpu_core::error::LibTpuError`] if parsing, bytecode compilation, or execution fails.
    pub fn compile_mlir_bytecode_and_execute(
        &self,
        device_id: DeviceId,
        mlir_source: &str,
        shape: &Shape,
        inputs: &[&[u8]],
    ) -> LibTpuResult<Vec<Vec<u8>>> {
        libtpu_core::compiler::parse_mlir_ast_module(mlir_source).and_then(|ast_mod| {
            let bc_bytes =
                libtpu_core::compiler::MlirBytecodeHeader::encode_from_ast_module(&ast_mod);
            self.execute_shaped_tensor(device_id, "mlir", &bc_bytes, shape, inputs)
        })
    }

    /// Compiles a wire-encoded `xla.HloModuleProto` payload (`hlo_proto_bytes`) and executes it
    /// on `device_id` with `inputs`.
    ///
    /// # Arguments
    ///
    /// * `device_id` - Target logical TPU device ID.
    /// * `hlo_proto_bytes` - Wire-serialized `xla.HloModuleProto` bytes.
    /// * `shape` - Input/output tensor [`Shape`].
    /// * `inputs` - Slice of raw byte slices for each input operand.
    ///
    /// # Errors
    ///
    /// Returns [`libtpu_core::error::LibTpuError`] if HLO protobuf compilation or execution fails.
    pub fn compile_and_execute_hlo_proto(
        &self,
        device_id: DeviceId,
        hlo_proto_bytes: &[u8],
        shape: &Shape,
        inputs: &[&[u8]],
    ) -> LibTpuResult<Vec<Vec<u8>>> {
        self.execute_shaped_tensor(device_id, "hlo", hlo_proto_bytes, shape, inputs)
    }

    /// Compiles `mlir_code`, runs [`libtpu_core::compiler::CompiledProgram::pass_hlo_rematerialization`]
    /// and [`libtpu_core::compiler::CompiledProgram::build_input_output_buffer_alias_config_proto`],
    /// and returns `(rematerialized_count, total_aliased_bytes, barnacore_program_count)`.
    ///
    /// # Arguments
    ///
    /// * `mlir_code` - MLIR module source bytes.
    /// * `memory_limit_bytes` - Target peak memory budget in bytes (`>= 256`).
    ///
    /// # Errors
    ///
    /// Returns [`libtpu_core::error::LibTpuError`] if compilation fails.
    pub fn query_rematerialization_and_barnacore(
        &self,
        mlir_code: &[u8],
        memory_limit_bytes: u64,
    ) -> LibTpuResult<(u32, u64, usize)> {
        let loaded = self.client.compile_and_load("mlir", mlir_code, b"")?;
        let mut prog = loaded.executable.program;
        let remat = prog.pass_hlo_rematerialization(memory_limit_bytes);
        let alias_cfg = prog.build_input_output_buffer_alias_config_proto();
        let barnacore = prog.build_barnacore_executable_proto();
        Ok((
            remat.rematerialized_count,
            alias_cfg.total_aliased_bytes,
            barnacore.programs.len(),
        ))
    }
}

/// High-level Rust bridge backing `libtpu.sdk` (`tpumonitoring`, `tpuz`, `slice`, `debug`).
#[derive(Debug)]
pub struct PySdkBridge {
    /// Underlying `LibtpuSdk_Client` core state.
    pub sdk_client: LibtpuSdkCoreClient,
    /// Underlying `TpuDriver` diagnostic state.
    pub driver: TpuCoreDriver,
    /// Underlying `XSpace` profiler session.
    pub profiler: TpuCoreProfiler,
}

impl PySdkBridge {
    /// Creates a new [`PySdkBridge`] for the given [`TpuGeneration`].
    ///
    /// # Arguments
    ///
    /// * `generation` - Target TPU hardware generation.
    #[must_use]
    pub fn new(generation: TpuGeneration) -> Self {
        let cfg = TpuInitConfig {
            generation,
            ..TpuInitConfig::default()
        };
        Self {
            sdk_client: LibtpuSdkCoreClient::from_config(&cfg),
            driver: TpuCoreDriver::open(cfg),
            profiler: TpuCoreProfiler::new(),
        }
    }

    /// Returns or initializes the shared process-wide [`Arc<PySdkBridge>`] for `generation`
    /// backed by [`global_shared_runtime_state`].
    ///
    /// # Arguments
    ///
    /// * `generation` - Target TPU hardware generation.
    #[must_use]
    pub fn shared(generation: TpuGeneration) -> Arc<Self> {
        let key = generation as i32;
        let cache = SHARED_PY_SDK_BRIDGES.get_or_init(|| Mutex::new(BTreeMap::new()));
        let cached = cache.lock().get(&key).cloned();
        if let Some(existing) = cached {
            return existing;
        }
        let created = Arc::new(Self::new(generation));
        cache.lock().insert(key, Arc::clone(&created));
        created
    }

    /// Queries the driver system information string (`TpuDriver_QuerySystemInfo`).
    #[must_use]
    pub fn query_system_info(&self) -> String {
        self.driver.query_system_info()
    }

    /// Queries the core state summary string from `LibtpuSdkCoreClient`.
    #[must_use]
    pub fn core_state_summary(&self) -> String {
        self.sdk_client.core_state_summary()
    }

    /// Queries the core state summary string, optionally appending recent HLO program history.
    ///
    /// # Arguments
    ///
    /// * `include_hlo_info` - Whether to append active/completed HLO program metadata.
    #[must_use]
    pub fn core_state_summary_with_hlo(&self, include_hlo_info: bool) -> String {
        let base = self.sdk_client.core_state_summary();
        if !include_hlo_info {
            return base;
        }
        let progs = self.sdk_client.runtime_state().recent_core_programs();
        let hlo_count = progs.len();
        format!("{base} hlo_programs={hlo_count}")
    }

    /// Queries a live runtime metric by name (`tpu.runtime.hbm.usage_bytes`, etc.).
    ///
    /// # Arguments
    ///
    /// * `metric_name` - Canonical metric name.
    ///
    /// # Errors
    ///
    /// Returns [`libtpu_core::error::LibTpuError::MetricNotFound`] if `metric_name` is unknown.
    pub fn get_metric_values(&self, metric_name: &str) -> LibTpuResult<Vec<f64>> {
        Ok(self.sdk_client.get_metric(metric_name)?.1)
    }

    /// Captures a serialized `tensorflow.profiler.XSpace` trace payload.
    ///
    /// # Errors
    ///
    /// Returns [`libtpu_core::error::LibTpuError`] if the profiler state transition fails.
    pub fn capture_xspace(&self) -> LibTpuResult<Vec<u8>> {
        self.profiler.start()?;
        let _ = self.profiler.stop();
        Ok(self.profiler.collect_xspace_bytes())
    }

    /// Resolves a hierarchical `DeepSea` CSR path in the hardware CSR catalog for this bridge's
    /// generation and returns its physical BAR0 MMIO byte offset (`None` if unknown).
    ///
    /// # Arguments
    ///
    /// * `path` - Hierarchical CSR register path string.
    #[must_use]
    pub fn lookup_csr_by_path(&self, path: &str) -> Option<u64> {
        let map = libtpu_core::hal::ChipRegisterMap::for_generation(self.driver.config.generation);
        map.resolve_hierarchical_csr_path(path)
            .map(|d| d.byte_offset)
    }

    /// Resolves a physical BAR0 MMIO byte offset in the hardware CSR catalog for this bridge's
    /// generation and returns its canonical hierarchical register name (`None` if unknown).
    ///
    /// # Arguments
    ///
    /// * `byte_offset` - Physical BAR0 MMIO byte offset.
    #[must_use]
    pub fn lookup_csr_by_offset(&self, byte_offset: u64) -> Option<&'static str> {
        let map = libtpu_core::hal::ChipRegisterMap::for_generation(self.driver.config.generation);
        map.lookup_by_offset(byte_offset).map(|d| d.full_path)
    }

    /// Collects serialized `tensorflow.ProfileToolData` bytes for `tool_name`
    /// (`"trace_viewer.json"`, `"input_pipeline.pb"`, `"memory_profile.json"`, `"xspace.pb"`).
    ///
    /// # Arguments
    ///
    /// * `tool_name` - Profiling tool artifact name.
    #[must_use]
    pub fn collect_profiler_tool_data(&self, tool_name: &str) -> Vec<u8> {
        self.profiler.collect_tool_data_bytes(tool_name)
    }
}

/// Executes an in-memory multi-replica `AllReduce` sum across `f32` slices via [`libtpu_core::distributed::CollectiveRuntimeEngine`].
///
/// # Arguments
///
/// * `replicas` - Slice of 4-element `f32` arrays (one per replica).
#[must_use]
pub fn execute_multi_replica_all_reduce_f32_4(replicas: &[[f32; 4]]) -> Vec<[f32; 4]> {
    let raw_bufs: Vec<Vec<u8>> = replicas
        .iter()
        .map(|arr| arr.iter().flat_map(|&v| v.to_le_bytes()).collect())
        .collect();
    let reduced = libtpu_core::distributed::CollectiveRuntimeEngine::execute_all_reduce(
        &raw_bufs,
        ElementType::F32,
        libtpu_core::types::CollectiveReductionOp::Sum,
    );
    reduced
        .into_iter()
        .map(|buf| {
            let mut out = [0.0_f32; 4];
            for (dst, chunk) in out.iter_mut().zip(buf.chunks_exact(4)) {
                let mut b = [0_u8; 4];
                b.copy_from_slice(chunk);
                *dst = f32::from_le_bytes(b);
            }
            out
        })
        .collect()
}

/// Returns the number of verified embedded `protodesc_cold` `.proto` descriptors (`293`).
#[must_use]
pub const fn protodesc_293_catalog_count() -> usize {
    libtpu_core::elf::PROTODESC_COLD_293_CATALOG.len()
}

/// Verifies that all 293 embedded `protodesc_cold` `.proto` descriptors are present in
/// [`libtpu_core::proto::global_descriptor_pool`] and returns `293`.
///
/// # Errors
///
/// Returns [`libtpu_core::error::LibTpuError`] if any descriptor is missing.
pub fn verify_protodesc_cold_293() -> LibTpuResult<usize> {
    libtpu_core::elf::verify_protodesc_cold_293_catalog_closure(
        libtpu_core::elf::PROTODESC_COLD_293_CATALOG,
    )
}

/// Assembles a multi-line `DeepSea` LLO VLIW assembly listing into 64-byte `IMEM` binary words.
///
/// # Arguments
///
/// * `listing` - Multi-line `DeepSea` LLO VLIW assembly text.
///
/// # Errors
///
/// Returns [`libtpu_core::error::LibTpuError`] if any bundle line is malformed.
pub fn assemble_llo_vliw(listing: &str) -> LibTpuResult<Vec<u8>> {
    let bundles = libtpu_core::compiler::parse_llo_assembly(listing)?;
    let mut out = Vec::with_capacity(bundles.len() * 64);
    for b in &bundles {
        out.extend_from_slice(&b.to_bytes());
    }
    Ok(out)
}

/// Dispatches a wire-encoded [`libtpu_core::proto::HostCommandEnvelopeProto`] on an emulated
/// [`libtpu_core::hal::LinuxDeviceHal`] continuation queue.
///
/// # Arguments
///
/// * `envelope_bytes` - Wire-encoded `HostCommandEnvelopeProto` payload.
///
/// # Errors
///
/// Returns [`libtpu_core::error::LibTpuError`] if `envelope_bytes` is malformed.
pub fn service_host_command_envelope(envelope_bytes: &[u8]) -> LibTpuResult<u64> {
    let cfg = TpuInitConfig::default();
    libtpu_core::hal::LinuxDeviceHal::open(
        libtpu_core::hal::KernelDriverMode::Gasket,
        libtpu_core::hal::EmulatedKernelDriverBackend::default(),
        &cfg,
    )
    .and_then(|hal| hal.service_continuation_queue(envelope_bytes))
}

/// Disassembles 64-byte `IMEM` binary words into a canonical `DeepSea` LLO VLIW assembly listing.
///
/// # Arguments
///
/// * `imem_bytes` - Raw 64-byte-aligned `IMEM` bundle words.
#[must_use]
pub fn disassemble_llo_vliw(imem_bytes: &[u8]) -> String {
    let bundles = libtpu_core::compiler::CompiledProgram::decode_architectural_imem(imem_bytes);
    libtpu_core::compiler::format_llo_assembly(&bundles)
}

/// Executes an in-memory multi-replica `CollectiveBroadcast` across `f32` slices from `source_replica_id`.
///
/// # Arguments
///
/// * `replicas` - Slice of 4-element `f32` arrays (one per replica).
/// * `source_replica_id` - Index of the root replica (`0..N`).
#[must_use]
pub fn execute_multi_replica_collective_broadcast_f32_4(
    replicas: &[[f32; 4]],
    source_replica_id: usize,
) -> Vec<[f32; 4]> {
    let raw_bufs: Vec<Vec<u8>> = replicas
        .iter()
        .map(|arr| arr.iter().flat_map(|&v| v.to_le_bytes()).collect())
        .collect();
    let broadcasted =
        libtpu_core::distributed::CollectiveRuntimeEngine::execute_collective_broadcast(
            &raw_bufs,
            source_replica_id,
        );
    broadcasted
        .into_iter()
        .map(|buf| {
            let mut out = [0.0_f32; 4];
            for (dst, chunk) in out.iter_mut().zip(buf.chunks_exact(4)) {
                let mut b = [0_u8; 4];
                b.copy_from_slice(chunk);
                *dst = f32::from_le_bytes(b);
            }
            out
        })
        .collect()
}

/// Executes an in-memory 2-replica `RaggedAllToAll` exchanging 2-element halves between replica 0 and replica 1.
///
/// # Arguments
///
/// * `replicas` - Slice of 4-element `f32` arrays (`2` replicas).
#[must_use]
pub fn execute_multi_replica_ragged_all_to_all_f32_4(replicas: &[[f32; 4]]) -> Vec<[f32; 4]> {
    let raw_bufs: Vec<Vec<u8>> = replicas
        .iter()
        .map(|arr| arr.iter().flat_map(|&v| v.to_le_bytes()).collect())
        .collect();
    let offsets = vec![vec![0, 2], vec![0, 2]];
    let sizes = vec![vec![2, 2], vec![2, 2]];
    let exchanged = libtpu_core::distributed::CollectiveRuntimeEngine::execute_ragged_all_to_all(
        &raw_bufs,
        ElementType::F32,
        &offsets,
        &sizes,
        &offsets,
    );
    exchanged
        .into_iter()
        .map(|buf| {
            let mut out = [0.0_f32; 4];
            for (dst, chunk) in out.iter_mut().zip(buf.chunks_exact(4)) {
                let mut b = [0_u8; 4];
                b.copy_from_slice(chunk);
                *dst = f32::from_le_bytes(b);
            }
            out
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::too_many_lines)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::dlpack::{
        DlDataType, DlDevice, DlManagedTensor, DlTensor, compute_compact_strides,
        consume_dlpack_capsule, dlpack_managed_tensor_deleter, export_dlpack_capsule,
        unconsumed_dlpack_capsule_destructor,
    };
    use crate::pyo3_modules::{
        c_module_exec, create_exception_or_fallback, sdk_module_exec, set_module_attr,
    };
    use core::ffi::c_void;
    use pyo3::PyTypeInfo;
    use pyo3::exceptions::PyRuntimeError;
    use pyo3::ffi;
    use pyo3::prelude::*;
    use pyo3::types::PyModule;

    /// Builds a minimal 64-bit ELF shared object exporting `"GetPjrtApi"` in `.dynsym`.
    #[allow(clippy::indexing_slicing)]
    fn build_test_elf64_dynsym() -> Vec<u8> {
        let mut buf = vec![0_u8; 64];
        buf[0..4].copy_from_slice(b"\x7fELF");
        buf[4] = 2;
        buf[5] = 1;
        buf[6] = 1;
        buf[16..18].copy_from_slice(&3_u16.to_le_bytes());
        buf[18..20].copy_from_slice(&62_u16.to_le_bytes());
        buf[20..24].copy_from_slice(&1_u32.to_le_bytes());
        buf[52..54].copy_from_slice(&64_u16.to_le_bytes());
        let shstrtab = b"\0.shstrtab\0.dynstr\0.dynsym\0";
        let shstr_off = buf.len();
        buf.extend_from_slice(shstrtab);
        let dynstr = b"\0GetPjrtApi\0";
        let dynstr_off = buf.len();
        buf.extend_from_slice(dynstr);
        while buf.len() % 8 != 0 {
            buf.push(0);
        }
        let dynsym_off = buf.len();
        buf.extend_from_slice(&[0_u8; 24]);
        buf.extend_from_slice(&1_u32.to_le_bytes());
        buf.push(0x12);
        buf.push(0);
        buf.extend_from_slice(&1_u16.to_le_bytes());
        buf.extend_from_slice(&0x1000_u64.to_le_bytes());
        buf.extend_from_slice(&32_u64.to_le_bytes());
        let dynsym_size = (buf.len() - dynsym_off) as u64;
        let shoff = buf.len();
        buf[40..48].copy_from_slice(&(shoff as u64).to_le_bytes());
        buf[58..60].copy_from_slice(&64_u16.to_le_bytes());
        buf[60..62].copy_from_slice(&4_u16.to_le_bytes());
        buf[62..64].copy_from_slice(&1_u16.to_le_bytes());
        let push_shdr = |out: &mut Vec<u8>,
                         name: u32,
                         sh_type: u32,
                         offset: u64,
                         size: u64,
                         link: u32,
                         entsize: u64| {
            let mut sh = [0_u8; 64];
            sh[0..4].copy_from_slice(&name.to_le_bytes());
            sh[4..8].copy_from_slice(&sh_type.to_le_bytes());
            sh[24..32].copy_from_slice(&offset.to_le_bytes());
            sh[32..40].copy_from_slice(&size.to_le_bytes());
            sh[40..44].copy_from_slice(&link.to_le_bytes());
            sh[56..64].copy_from_slice(&entsize.to_le_bytes());
            out.extend_from_slice(&sh);
        };
        push_shdr(&mut buf, 0, 0, 0, 0, 0, 0);
        push_shdr(
            &mut buf,
            1,
            3,
            shstr_off as u64,
            shstrtab.len() as u64,
            0,
            0,
        );
        push_shdr(
            &mut buf,
            11,
            3,
            dynstr_off as u64,
            dynstr.len() as u64,
            0,
            0,
        );
        push_shdr(&mut buf, 19, 11, dynsym_off as u64, dynsym_size, 2, 24);
        buf
    }

    #[test]
    fn test_py_runtime_and_sdk_bridge() {
        let expected_proto_cnt = libtpu_core::elf::parse_golden_proto_descriptors(include_str!(
            "../../../docs/reverse_engineering/golden_proto_descriptors.json"
        ))
        .map_or(0, |m| m.proto_files_count);
        assert_eq!(PyTpuRuntime::pjrt_api_version(), (0, 117));
        assert_eq!(protodesc_293_catalog_count(), expected_proto_cnt);
        assert_eq!(verify_protodesc_cold_293(), Ok(expected_proto_cnt));

        let llo_src = "[pc=0000 gen=v5 stall=0] { S0(op=1, $s0 <- $s0, $s0, imm=0) | V0(AddF32, $v0 <- $v0, $v1, $v0) }";
        let imem_res = assemble_llo_vliw(llo_src);
        assert!(imem_res.is_ok());
        if let Ok(imem) = imem_res {
            assert_eq!(imem.len(), 64);
            assert!(disassemble_llo_vliw(&imem).contains("pc=0000"));
        }
        assert!(assemble_llo_vliw("bad_line").is_err());

        let env_pb = libtpu_core::proto::HostCommandEnvelopeProto {
            opcode: 1,
            channel_id: 1,
            continuation_id: 99,
            handler_id: 3,
            payload: b"hello".to_vec(),
            ..libtpu_core::proto::HostCommandEnvelopeProto::default()
        };
        assert!(service_host_command_envelope(&env_pb.encode()).is_ok());
        assert!(service_host_command_envelope(&[0xFF, 0xFF, 0xFF]).is_err());

        let rep = execute_multi_replica_all_reduce_f32_4(&[[1.0, 2.0, 3.0, 4.0]; 2]);
        assert_eq!(rep.len(), 2);
        assert_eq!(rep.first(), Some(&[2.0, 4.0, 6.0, 8.0]));
        let bcast = execute_multi_replica_collective_broadcast_f32_4(
            &[[9.0, 8.0, 7.0, 6.0], [1.0, 1.0, 1.0, 1.0]],
            0,
        );
        assert_eq!(bcast.get(1), Some(&[9.0, 8.0, 7.0, 6.0]));
        let ragged = execute_multi_replica_ragged_all_to_all_f32_4(&[
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ]);
        assert_eq!(ragged.first(), Some(&[1.0, 2.0, 5.0, 6.0]));
        assert_eq!(ragged.get(1), Some(&[3.0, 4.0, 7.0, 8.0]));

        assert!(PyTpuRuntime::inspect_binary_symbol_names(b"not_a_binary").is_err());
        let mut macho_hdr = vec![0_u8; 32];
        if let Some(s) = macho_hdr.get_mut(0..4) {
            s.copy_from_slice(&0xFEED_FACF_u32.to_le_bytes());
        }
        if let Some(s) = macho_hdr.get_mut(4..8) {
            s.copy_from_slice(&0x0100_000C_u32.to_le_bytes());
        }
        if let Some(s) = macho_hdr.get_mut(12..16) {
            s.copy_from_slice(&6_u32.to_le_bytes());
        }
        assert!(PyTpuRuntime::inspect_binary_symbol_names(&macho_hdr).is_ok());

        assert!(PyTpuRuntime::from_init_args("--tpu_generation=invalid").is_err());
        let rt_res = PyTpuRuntime::from_init_args("--tpu_generation=v6e");
        assert!(rt_res.is_ok());
        if let Ok(rt) = rt_res {
            assert_ne!(format!("{rt:?}"), "");
            assert_eq!(rt.generation(), TpuGeneration::V6e);
            assert_eq!(rt.device_count(), 4);
            assert_ne!(rt.serialize_topology(), [] as [u8; 0]);

            let mlir = b"module @jit_add { func.func public @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> { %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32> return %0 : tensor<4xf32> } }";
            let ser = rt.compile_and_serialize("mlir", mlir).unwrap_or_default();
            assert_ne!(ser, [] as [u8; 0]);
            let fp = rt
                .deserialize_executable_fingerprint(&ser)
                .unwrap_or_default();
            assert_eq!(fp.len(), 32);
            assert!(rt.compile_and_serialize("unknown", mlir).is_err());
            assert!(rt.deserialize_executable_fingerprint(b"bad").is_err());

            assert_eq!(
                rt.execute_f32_4(DeviceId(0), mlir, &[1.5, 2.5, 3.5, 4.5]),
                Ok([3.0, 5.0, 7.0, 9.0])
            );
            assert!(rt.execute_f32_4(DeviceId(99), mlir, &[1.0; 4]).is_err());
            assert!(
                rt.execute_f32_4(DeviceId(0), b"SYNTAX_ERROR", &[1.0; 4])
                    .is_err()
            );

            let shape_1d = Shape {
                element_type: ElementType::F32,
                dims: vec![4],
                dynamic_dims: vec![false],
                layout: libtpu_core::types::MemoryLayout::default_tpu_tiled(1),
            };
            let raw_in: Vec<u8> = [2.0_f32, 4.0, 6.0, 8.0]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let mlir_str = core::str::from_utf8(mlir).unwrap_or("");
            let bc_outs = rt.compile_mlir_bytecode_and_execute(
                DeviceId(0),
                mlir_str,
                &shape_1d,
                &[&raw_in, &raw_in],
            );
            assert!(bc_outs.is_ok());
            assert!(
                rt.compile_mlir_bytecode_and_execute(DeviceId(0), "bad {{{", &shape_1d, &[])
                    .is_err()
            );

            let ast_m = libtpu_core::compiler::parse_mlir_ast_module(mlir_str);
            assert!(ast_m.is_ok());
            if let Ok(am) = ast_m
                && let Ok(hlo_mod_pb) = libtpu_core::compiler::stablehlo_to_hlo_module_proto(&am)
            {
                use libtpu_core::proto::ProtoSchemaMessage as _;
                let hlo_pb = hlo_mod_pb.encode_proto();
                assert!(
                    rt.compile_and_execute_hlo_proto(
                        DeviceId(0),
                        &hlo_pb,
                        &shape_1d,
                        &[&raw_in, &raw_in]
                    )
                    .is_ok()
                );
            }
            assert!(
                rt.compile_and_execute_hlo_proto(DeviceId(0), &[0xFF, 0xFF], &shape_1d, &[])
                    .is_err()
            );
            let q = rt.query_rematerialization_and_barnacore(mlir, 512);
            assert!(q.is_ok());
            assert!(
                rt.query_rematerialization_and_barnacore(b"SYNTAX_ERROR", 512)
                    .is_err()
            );
            let bad_shape = Shape {
                element_type: ElementType::F32,
                dims: vec![8],
                dynamic_dims: vec![false],
                layout: libtpu_core::types::MemoryLayout::default_tpu_tiled(1),
            };
            assert!(
                rt.execute_shaped_tensor(DeviceId(0), "mlir", mlir, &bad_shape, &[&raw_in])
                    .is_err()
            );
        }

        let bridge = PySdkBridge::new(TpuGeneration::V5e);
        assert_ne!(format!("{bridge:?}"), "");
        assert_ne!(bridge.query_system_info(), "");
        assert_ne!(bridge.core_state_summary(), "");
        assert!(
            bridge
                .get_metric_values("tpu.runtime.hbm.usage_bytes")
                .is_ok()
        );
        assert!(bridge.get_metric_values("unknown").is_err());
        assert_ne!(bridge.capture_xspace().unwrap_or_default(), [] as [u8; 0]);
        assert_eq!(bridge.lookup_csr_by_path("nonexistent"), None);
        let v5e_map = libtpu_core::hal::ChipRegisterMap::for_generation(TpuGeneration::V5e);
        let v5e_csrs = v5e_map.csr_descriptors();
        if let Some(first_csr) = v5e_csrs.first() {
            assert_eq!(
                bridge.lookup_csr_by_path(first_csr.full_path),
                Some(first_csr.byte_offset)
            );
            assert!(bridge.lookup_csr_by_offset(first_csr.byte_offset).is_some());
        }
        assert_eq!(bridge.lookup_csr_by_offset(0xDEAD_BEEF), None);
        assert_ne!(
            bridge.collect_profiler_tool_data("xspace.pb"),
            [] as [u8; 0]
        );
        let _ = bridge.profiler.start();
        assert!(bridge.capture_xspace().is_err());
    }

    #[test]
    fn test_pyo3_native_modules_and_dlpack() {
        Python::initialize();
        Python::attach(|py| {
            // 1. Verify PyInit__C, PyInit_sdk, and PyInit_libtpu_py PEP 489 multi-phase module defs.
            // SAFETY: Holding the GIL in an initialized CPython interpreter.
            let raw_c_def = unsafe { PyInit__C() };
            assert!(!raw_c_def.is_null());
            // SAFETY: Holding the GIL.
            let raw_sdk_def = unsafe { PyInit_sdk() };
            assert!(!raw_sdk_def.is_null());
            // SAFETY: Holding the GIL.
            let raw_alias_def = unsafe { PyInit_libtpu_py() };
            assert!(!raw_alias_def.is_null());

            // Null and non-module error paths on c_module_exec and sdk_module_exec.
            // SAFETY: Passing null explicitly tests the null guard.
            assert_eq!(unsafe { c_module_exec(core::ptr::null_mut()) }, -1);
            // SAFETY: Passing null explicitly tests the null guard.
            assert_eq!(unsafe { sdk_module_exec(core::ptr::null_mut()) }, -1);

            let Ok(int_obj) = 7_i32.into_pyobject(py).map(Bound::into_any);
            // SAFETY: `int_obj.as_ptr()` is a valid borrowed `PyObject*`; `setattr` fails on `int`.
            assert_eq!(unsafe { c_module_exec(int_obj.as_ptr()) }, -1);
            // SAFETY: Clearing the expected AttributeError restored by `c_module_exec`.
            unsafe {
                ffi::PyErr_Clear();
            }
            // SAFETY: `int_obj.as_ptr()` is a valid borrowed `PyObject*`; `setattr` fails on `int`.
            assert_eq!(unsafe { sdk_module_exec(int_obj.as_ptr()) }, -1);
            // SAFETY: Clearing the expected AttributeError restored by `sdk_module_exec`.
            unsafe {
                ffi::PyErr_Clear();
            }

            // Exercise fallback branch when CPython rejects a dotless exception name.
            let fb_ty = create_exception_or_fallback(py, c"InvalidNoDotName");
            assert!(fb_ty.bind(py).is(PyRuntimeError::type_object(py)));

            // Exercise `set_module_attr` Err path directly.
            let c_mod = PyModule::new(py, "_C").map(Bound::into_any);
            assert!(c_mod.is_ok());
            if let Ok(c_mod) = c_mod {
                assert!(
                    set_module_attr(&c_mod, "bad", Err(PyRuntimeError::new_err("synthetic")))
                        .is_err()
                );
                // SAFETY: `c_mod` is a valid `PyModule`.
                assert_eq!(unsafe { c_module_exec(c_mod.as_ptr()) }, 0);

                let expected_proto_cnt = libtpu_core::elf::parse_golden_proto_descriptors(
                    include_str!("../../../docs/reverse_engineering/golden_proto_descriptors.json"),
                )
                .map_or(0, |m| m.proto_files_count);
                let ver: String = c_mod
                    .getattr("__version__")
                    .and_then(|v| v.extract())
                    .unwrap_or_default();
                assert_eq!(ver, libtpu_core::LIBTPU_PYPI_VERSION);

                let pjrt_ver: (i32, i32) = c_mod
                    .getattr("pjrt_api_version")
                    .and_then(|f| f.call0())
                    .and_then(|v| v.extract())
                    .unwrap_or((0, 0));
                assert_eq!(pjrt_ver, (0, 117));
                if let Ok(bind_fn) = c_mod.getattr("bind_c_api_runtime_sync") {
                    assert!(bind_fn.call1((0_usize,)).is_ok());
                    assert!(bind_fn.call0().is_err());
                }

                let proto_cnt: usize = c_mod
                    .getattr("protodesc_cold_count")
                    .and_then(|f| f.call0())
                    .and_then(|v| v.extract())
                    .unwrap_or(0);
                assert_eq!(proto_cnt, expected_proto_cnt);

                let parse_fn = c_mod.getattr("parse_init_args");
                assert!(parse_fn.is_ok());
                if let Ok(pf) = parse_fn {
                    let parsed: (String, i32, u32, u32) = pf
                        .call1(("--tpu_generation=v6e --megascale_num_slices=2",))
                        .and_then(|v| v.extract())
                        .unwrap_or_default();
                    assert_eq!(parsed.0, "TPU v6 lite");
                    assert_eq!(parsed.2, 2);
                    assert!(pf.call1((123,)).is_err());
                    let err_res = pf.call1(("--tpu_generation=invalid",));
                    assert!(err_res.is_err());
                    if let Err(py_err) = err_res {
                        assert!(py_err.is_instance(py, &tpu_error_type(py)));
                    }
                }

                // pack_dtype_f64 & unpack_dtype_f64 (BF16 = 13, F8E4M3FN = 16, S4 = 21, Invalid = 0).
                if let (Ok(pack_f), Ok(unpack_f)) = (
                    c_mod.getattr("pack_dtype_f64"),
                    c_mod.getattr("unpack_dtype_f64"),
                ) {
                    let raw_bf16: Vec<u8> = pack_f
                        .call1((13_i32, vec![1.5_f64, -2.0_f64]))
                        .and_then(|v| v.extract())
                        .unwrap_or_default();
                    assert_eq!(raw_bf16.len(), 4);
                    let unpacked_bf16: Vec<f64> = unpack_f
                        .call1((13_i32, raw_bf16))
                        .and_then(|v| v.extract())
                        .unwrap_or_default();
                    assert_eq!(unpacked_bf16, vec![1.5, -2.0]);
                    assert!(pack_f.call1((0_i32, vec![1.0_f64])).is_err());
                    assert!(pack_f.call1((999_i32, vec![1.0_f64])).is_err());
                    assert!(pack_f.call1(("bad",)).is_err());
                    assert!(unpack_f.call1((0_i32, vec![0_u8])).is_err());
                    assert!(unpack_f.call1((999_i32, vec![0_u8])).is_err());
                    assert!(unpack_f.call1(("bad",)).is_err());
                }

                // compile_and_execute happy + error paths.
                if let Ok(exec_f) = c_mod.getattr("compile_and_execute") {
                    let mlir = b"module @m { func.func public @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> { %0 = stablehlo.add %arg0, %arg1 : tensor<2xf32> return %0 : tensor<2xf32> } }".to_vec();
                    let a_bytes: Vec<u8> = [3.0_f32, 4.0]
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect();
                    let outs: Vec<Vec<u8>> = exec_f
                        .call1((
                            "mlir",
                            mlir.clone(),
                            vec![2_i64],
                            11_i32,
                            vec![a_bytes.clone(), a_bytes],
                            "--tpu_generation=v5e",
                        ))
                        .and_then(|v| v.extract())
                        .unwrap_or_default();
                    assert_eq!(outs.len(), 1);
                    assert!(exec_f.call1(("bad_arity",)).is_err());
                    assert!(
                        exec_f
                            .call1((
                                "mlir",
                                mlir.clone(),
                                vec![2_i64],
                                0_i32,
                                Vec::<Vec<u8>>::new(),
                                ""
                            ))
                            .is_err()
                    );
                    assert!(
                        exec_f
                            .call1((
                                "mlir",
                                mlir,
                                vec![2_i64],
                                11_i32,
                                Vec::<Vec<u8>>::new(),
                                "--tpu_generation=invalid",
                            ))
                            .is_err()
                    );
                    assert!(
                        exec_f
                            .call1((
                                "mlir",
                                b"SYNTAX_ERROR".to_vec(),
                                vec![2_i64],
                                11_i32,
                                Vec::<Vec<u8>>::new(),
                                "",
                            ))
                            .is_err()
                    );
                }

                // PEP 3118 buffer_view happy + error paths.
                if let Ok(bv_f) = c_mod.getattr("buffer_view") {
                    let mv = bv_f.call1((b"abcd".as_slice(),));
                    assert!(mv.is_ok());
                    assert!(bv_f.call0().is_err());
                    assert!(bv_f.call1((42_i32,)).is_err());
                }

                // DLPack export + consume + unconsumed destructor + 0D scalar + strided + null deleter.
                if let (Ok(exp_f), Ok(con_f)) = (
                    c_mod.getattr("export_dlpack"),
                    c_mod.getattr("consume_dlpack"),
                ) {
                    let payload: Vec<u8> = [10.0_f32, 20.0, 30.0, 40.0]
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect();
                    let cap = exp_f.call1((
                        payload.clone(),
                        vec![2_i64, 2_i64],
                        2_u8,
                        32_u8,
                        1_i32,
                        0_i32,
                    ));
                    assert!(cap.is_ok());
                    if let Ok(cap_obj) = cap {
                        let res: (Vec<u8>, Vec<i64>, u8, u8, u16, i32, i32) = con_f
                            .call1((&cap_obj,))
                            .and_then(|v| v.extract())
                            .unwrap_or_default();
                        assert_eq!(res.0, payload);
                        assert_eq!(res.1, vec![2, 2]);
                        assert_eq!(res.2, 2);
                        assert_eq!(res.3, 32);
                        // Second consumption of `"used_dltensor"` must fail cleanly with `TpuError`!
                        assert!(con_f.call1((&cap_obj,)).is_err());
                        // Dropping `cap_obj` now triggers `unconsumed_dlpack_capsule_destructor` on `"used_dltensor"`.
                    }
                    // 0-D scalar capsule consumption + separate unconsumed capsule drop!
                    let scalar_cap = exp_f.call1((
                        Vec::<u8>::new(),
                        Vec::<i64>::new(),
                        2_u8,
                        32_u8,
                        1_i32,
                        0_i32,
                    ));
                    assert!(scalar_cap.is_ok());
                    if let Ok(sc_obj) = scalar_cap {
                        assert!(con_f.call1((&sc_obj,)).is_ok());
                    }
                    let drop_cap =
                        exp_f.call1((vec![1_u8, 2, 3, 4], vec![1_i64], 2_u8, 32_u8, 1_i32, 0_i32));
                    assert!(drop_cap.is_ok());
                    drop(drop_cap);

                    assert!(exp_f.call0().is_err());
                    assert!(con_f.call0().is_err());
                }

                // inspect_binary_symbols happy + error paths (using cross-platform synthetic ELF64 with .dynsym).
                if let Ok(sym_f) = c_mod.getattr("inspect_binary_symbols") {
                    let elf_bytes = build_test_elf64_dynsym();
                    let syms: Vec<String> = sym_f
                        .call1((elf_bytes,))
                        .and_then(|v| v.extract())
                        .unwrap_or_default();
                    assert!(syms.iter().any(|s| s == "GetPjrtApi"));
                    assert!(sym_f.call1((b"invalid".to_vec(),)).is_err());
                    assert!(sym_f.call0().is_err());
                }
                if let Ok(dec_f) = c_mod.getattr("decode_mlir_v6_text") {
                    let empty_s: String = dec_f
                        .call1((b"not_mlir".to_vec(),))
                        .and_then(|v| v.extract())
                        .unwrap_or_default();
                    assert_eq!(empty_s, "");
                    assert!(dec_f.call0().is_err());
                }
            }

            // 2. Verify SDK module (`PyInit_sdk` / `sdk_module_exec`).
            let sdk_mod = PyModule::new(py, "sdk").map(Bound::into_any);
            assert!(sdk_mod.is_ok());
            if let Ok(sdk_mod) = sdk_mod {
                // SAFETY: `sdk_mod` is a valid `PyModule`.
                assert_eq!(unsafe { sdk_module_exec(sdk_mod.as_ptr()) }, 0);

                let metrics: Vec<String> = sdk_mod
                    .getattr("list_supported_metrics")
                    .and_then(|f| f.call0())
                    .and_then(|v| v.extract())
                    .unwrap_or_default();
                assert_ne!(metrics, [] as [String; 0]);

                if let Ok(gm_f) = sdk_mod.getattr("get_metric") {
                    assert!(gm_f.call1(("tpu.runtime.hbm.usage_bytes", "v5e")).is_ok());
                    assert!(gm_f.call1(("unknown_metric", "v5e")).is_err());
                    assert!(
                        gm_f.call1(("tpu.runtime.hbm.usage_bytes", "bad_gen"))
                            .is_err()
                    );
                    assert!(gm_f.call0().is_err());
                }
                if let Ok(cc_f) = sdk_mod.getattr("get_chip_coordinates") {
                    let coords: Vec<(u32, u32, u32, i32)> = cc_f
                        .call1(("v5e",))
                        .and_then(|v| v.extract())
                        .unwrap_or_default();
                    assert_ne!(coords, [] as [(u32, u32, u32, i32); 0]);
                    assert!(cc_f.call1(("bad_gen",)).is_err());
                    assert!(cc_f.call0().is_err());
                }
                if let Ok(cs_f) = sdk_mod.getattr("get_core_state_summary") {
                    let summary: String = cs_f
                        .call1(("v5e",))
                        .and_then(|v| v.extract())
                        .unwrap_or_default();
                    assert_ne!(summary, "");
                    let summary_hlo: String = cs_f
                        .call1(("v5e", true))
                        .and_then(|v| v.extract())
                        .unwrap_or_default();
                    assert!(summary_hlo.contains("hlo_programs="));
                    assert!(cs_f.call1(("bad_gen",)).is_err());
                    assert!(cs_f.call0().is_err());
                }
                if let Ok(bind_fn) = sdk_mod.getattr("bind_c_api_runtime_sync") {
                    assert!(bind_fn.call1((0_usize,)).is_ok());
                }
                if let Ok(shared_rt) = PyTpuRuntime::shared("--tpu_generation=v5e") {
                    assert!(Arc::strong_count(shared_rt.runtime_state()) >= 1);
                }
                if let Ok(xs_f) = sdk_mod.getattr("capture_xspace") {
                    let xb: Vec<u8> = xs_f
                        .call1(("v5e",))
                        .and_then(|v| v.extract())
                        .unwrap_or_default();
                    assert_ne!(xb, [] as [u8; 0]);
                    assert!(xs_f.call1(("bad_gen",)).is_err());
                    assert!(xs_f.call0().is_err());
                }
                if let Ok(csr_f) = sdk_mod.getattr("lookup_csr_offset") {
                    let v5e_map =
                        libtpu_core::hal::ChipRegisterMap::for_generation(TpuGeneration::V5e);
                    let v5e_csrs = v5e_map.csr_descriptors();
                    if let Some(first) = v5e_csrs.first() {
                        assert!(csr_f.call1((first.full_path, "v5e")).is_ok());
                    }
                    assert!(csr_f.call1(("missing_csr", "v5e")).is_err());
                    assert!(csr_f.call1(("missing_csr", "bad_gen")).is_err());
                    assert!(csr_f.call0().is_err());
                }
            }

            // 3. Exercise remaining DLPack edge cases: null deleter, null strides, strided transpose, null data, null manager_ctx.
            assert_eq!(compute_compact_strides(&[2, 3, 4]), vec![12, 4, 1]);
            // SAFETY: Null pointer is explicitly handled as a no-op.
            unsafe {
                dlpack_managed_tensor_deleter(core::ptr::null_mut());
            }
            let empty_managed = Box::into_raw(Box::new(DlManagedTensor {
                dl_tensor: DlTensor {
                    data: core::ptr::null_mut(),
                    device: DlDevice {
                        device_type: 1,
                        device_id: 0,
                    },
                    ndim: 0,
                    dtype: DlDataType {
                        code: 2,
                        bits: 32,
                        lanes: 1,
                    },
                    shape: core::ptr::null_mut(),
                    strides: core::ptr::null_mut(),
                    byte_offset: 0,
                },
                manager_ctx: core::ptr::null_mut(),
                deleter: None,
            }));
            // SAFETY: `empty_managed` was allocated via `Box::into_raw` with null `manager_ctx`.
            unsafe {
                dlpack_managed_tensor_deleter(empty_managed);
            }

            // Strided 2x2 transpose with `deleter: None` and non-zero `byte_offset`, plus null-strides 1D.
            let mut raw_mat: [u32; 5] = [999, 1, 2, 3, 4];
            let mut dims_2d: [i64; 2] = [2, 2];
            let mut transposed_strides: [i64; 2] = [1, 2];
            let mut strided_managed = DlManagedTensor {
                dl_tensor: DlTensor {
                    data: raw_mat.as_mut_ptr().cast::<c_void>(),
                    device: DlDevice {
                        device_type: 1,
                        device_id: 0,
                    },
                    ndim: 2,
                    dtype: DlDataType {
                        code: 1,
                        bits: 32,
                        lanes: 1,
                    },
                    shape: dims_2d.as_mut_ptr(),
                    strides: transposed_strides.as_mut_ptr(),
                    byte_offset: 4,
                },
                manager_ctx: core::ptr::null_mut(),
                deleter: None,
            };
            // SAFETY: `strided_managed` lives on stack across `consume_dlpack_capsule` with `deleter: None`.
            let cap_strided = unsafe {
                let ptr = ffi::PyCapsule_New(
                    (&raw mut strided_managed).cast::<c_void>(),
                    c"dltensor".as_ptr(),
                    None,
                );
                Bound::from_owned_ptr(py, ptr)
            };
            let consumed_strided = consume_dlpack_capsule(&cap_strided);
            assert!(consumed_strided.is_ok());
            if let Ok(cs) = consumed_strided {
                let expected: Vec<u8> = [1_u32, 3, 2, 4]
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect();
                assert_eq!(cs.raw_bytes, expected);
            }

            // Null-strides + null-data 0-size tensor, non-zero null-data tensor, and null-shape 1D tensor.
            let mut zero_dim: [i64; 1] = [0];
            let mut null_strides_managed = DlManagedTensor {
                dl_tensor: DlTensor {
                    data: core::ptr::null_mut(),
                    device: DlDevice {
                        device_type: 1,
                        device_id: 0,
                    },
                    ndim: 1,
                    dtype: DlDataType {
                        code: 2,
                        bits: 32,
                        lanes: 1,
                    },
                    shape: zero_dim.as_mut_ptr(),
                    strides: core::ptr::null_mut(),
                    byte_offset: 0,
                },
                manager_ctx: core::ptr::null_mut(),
                deleter: None,
            };
            // SAFETY: Stack `DlManagedTensor` with `deleter: None`.
            let cap_zero = unsafe {
                let ptr = ffi::PyCapsule_New(
                    (&raw mut null_strides_managed).cast::<c_void>(),
                    c"dltensor".as_ptr(),
                    None,
                );
                Bound::from_owned_ptr(py, ptr)
            };
            let cz = consume_dlpack_capsule(&cap_zero);
            assert!(cz.is_ok());
            if let Ok(c) = cz {
                assert_eq!(c.raw_bytes, [] as [u8; 0]);
            }

            let mut two_dim: [i64; 1] = [2];
            let mut null_data_nonzero_managed = DlManagedTensor {
                dl_tensor: DlTensor {
                    data: core::ptr::null_mut(),
                    device: DlDevice {
                        device_type: 1,
                        device_id: 0,
                    },
                    ndim: 1,
                    dtype: DlDataType {
                        code: 2,
                        bits: 32,
                        lanes: 1,
                    },
                    shape: two_dim.as_mut_ptr(),
                    strides: core::ptr::null_mut(),
                    byte_offset: 0,
                },
                manager_ctx: core::ptr::null_mut(),
                deleter: None,
            };
            // SAFETY: Stack `DlManagedTensor` with `deleter: None`.
            let cap_nd = unsafe {
                let ptr = ffi::PyCapsule_New(
                    (&raw mut null_data_nonzero_managed).cast::<c_void>(),
                    c"dltensor".as_ptr(),
                    None,
                );
                Bound::from_owned_ptr(py, ptr)
            };
            assert_eq!(
                consume_dlpack_capsule(&cap_nd).map_or(0, |c| c.raw_bytes.len()),
                8
            );

            let mut null_shape_managed = DlManagedTensor {
                dl_tensor: DlTensor {
                    data: core::ptr::null_mut(),
                    device: DlDevice {
                        device_type: 1,
                        device_id: 0,
                    },
                    ndim: 1,
                    dtype: DlDataType {
                        code: 2,
                        bits: 32,
                        lanes: 1,
                    },
                    shape: core::ptr::null_mut(),
                    strides: core::ptr::null_mut(),
                    byte_offset: 0,
                },
                manager_ctx: core::ptr::null_mut(),
                deleter: None,
            };
            // SAFETY: Stack `DlManagedTensor` with `deleter: None`.
            let cap_missing_shape = unsafe {
                let ptr = ffi::PyCapsule_New(
                    (&raw mut null_shape_managed).cast::<c_void>(),
                    c"dltensor".as_ptr(),
                    None,
                );
                Bound::from_owned_ptr(py, ptr)
            };
            assert!(consume_dlpack_capsule(&cap_missing_shape).is_ok());

            // Directly test `export_dlpack_capsule` and `unconsumed_dlpack_capsule_destructor`.
            let cap_direct = export_dlpack_capsule(
                py,
                &[1, 2, 3, 4],
                &[1],
                DlDataType {
                    code: 1,
                    bits: 32,
                    lanes: 1,
                },
                DlDevice {
                    device_type: 1,
                    device_id: 0,
                },
            );
            assert!(cap_direct.is_ok());
            if let Ok(cd) = cap_direct {
                // Invoke destructor on valid unconsumed capsule, then rename so drop doesn't double-free.
                // SAFETY: `cd` is an unconsumed `"dltensor"` capsule.
                unsafe {
                    unconsumed_dlpack_capsule_destructor(cd.as_ptr());
                    let _ = ffi::PyCapsule_SetName(cd.as_ptr(), c"used_dltensor".as_ptr());
                }
            }
        });
    }
}

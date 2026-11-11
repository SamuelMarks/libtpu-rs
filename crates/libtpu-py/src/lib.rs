//! Python FFI bridge and drop-in helper entrypoints for `libtpu-rs` (`libtpu` and `libtpu.sdk`).
//!
//! Provides [`PyTpuRuntime`], [`PySdkBridge`], and C-ABI / `PyInit_sdk` entrypoints used by the
//! `libtpu` Python package to resolve library paths, manage `LIBTPU_INIT_ARGS`, query
//! `GetPjrtApi()` and `GetLibtpuSdkApi()`, and execute TPU programs.

#![cfg_attr(all(test, coverage_nightly), feature(coverage_attribute))]

use libtpu_core::config::TpuInitConfig;
use libtpu_core::driver::{LibtpuSdkCoreClient, TpuCoreDriver, TpuCoreProfiler};
use libtpu_core::error::LibTpuResult;
use libtpu_core::pjrt::PjrtCoreClient;
use libtpu_core::types::{DeviceId, ElementType, MemorySpaceId, Shape, TpuGeneration};

/// `CPython` Stable ABI (`abi3`) object header (`PyObject`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CPythonObjectHead {
    /// Reference count (`ob_refcnt`).
    pub ob_refcnt: isize,
    /// Object type pointer (`ob_type`, `0` before interpreter initialization).
    pub ob_type: usize,
}

/// `CPython` Stable ABI (`abi3`) module definition base header (`PyModuleDef_Base`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CPythonModuleDefBase {
    /// Embedded `PyObject` header (`PyObject_HEAD_INIT(NULL)`).
    pub ob_base: CPythonObjectHead,
    /// Module init function pointer (`m_init`, `0`).
    pub m_init: usize,
    /// Module state index (`m_index`, `0`).
    pub m_index: isize,
    /// Module copy dictionary pointer (`m_copy`, `0`).
    pub m_copy: usize,
}

/// `CPython` `METH_VARARGS` calling convention flag (`0x0001`).
pub const METH_VARARGS: i32 = 0x0001;

/// `CPython` `METH_NOARGS` calling convention flag (`0x0004`).
pub const METH_NOARGS: i32 = 0x0004;

/// `CPython` multi-phase module initialization slot ID `Py_mod_exec` (`2`).
pub const PY_MOD_EXEC_SLOT: i32 = 2;

/// `CPython` Stable ABI (`abi3`) C function pointer type (`PyCFunction`).
pub type PyCFunction =
    unsafe extern "C" fn(*mut std::ffi::c_void, *mut std::ffi::c_void) -> *mut std::ffi::c_void;

/// `CPython` Stable ABI (`abi3`) method descriptor (`PyMethodDef`, `32` bytes on 64-bit targets).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CPythonMethodDef {
    /// Null-terminated method name (`ml_name`).
    pub ml_name: *const std::os::raw::c_char,
    /// C callback function pointer (`ml_meth`, `0` for sentinel).
    pub ml_meth: Option<PyCFunction>,
    /// Calling convention flags (`ml_flags`, e.g., [`METH_NOARGS`] or [`METH_VARARGS`]).
    pub ml_flags: i32,
    /// Reserved 4-byte structure padding on 64-bit targets.
    pub pad: i32,
    /// Null-terminated method docstring (`ml_doc`).
    pub ml_doc: *const std::os::raw::c_char,
}

// SAFETY: `CPythonMethodDef` holds only static `'static` C-string pointers and C function pointers.
unsafe impl Sync for CPythonMethodDef {}

/// `CPython` Stable ABI (`abi3`) `Py_mod_exec` callback signature (`unsafe extern "C" fn(*mut c_void) -> i32`).
pub type PyModExecFunction = unsafe extern "C" fn(*mut std::ffi::c_void) -> i32;

/// `CPython` Stable ABI (`abi3`) multi-phase module slot descriptor (`PyModuleDef_Slot`, `16` bytes on 64-bit targets).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CPythonModuleDefSlot {
    /// Slot identifier (`slot`, e.g., [`PY_MOD_EXEC_SLOT`] or `0` for sentinel).
    pub slot: i32,
    /// Explicit 4-byte alignment padding.
    pub pad: i32,
    /// Slot callback pointer (`value`).
    pub value: Option<PyModExecFunction>,
}

// SAFETY: `CPythonModuleDefSlot` holds only static C function pointers and integers.
unsafe impl Sync for CPythonModuleDefSlot {}

/// Native `CPython` `METH_NOARGS` callback returning the active `PY_SDK_ABI_HEADER` descriptor address
/// (`_get_sdk_abi_header`).
///
/// # Arguments
///
/// * `self_obj` - Module object pointer (`PyObject*`).
/// * `_args` - Unused argument pointer (`NULL` for `METH_NOARGS`).
///
/// # Safety
///
/// Safe to invoke from both `CPython` and standalone C/Rust callers; returns `self_obj` if non-null,
/// or `&raw const PY_SDK_ABI_HEADER` cast to `*mut c_void`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn py_sdk_get_abi_header(
    self_obj: *mut std::ffi::c_void,
    _args: *mut std::ffi::c_void,
) -> *mut std::ffi::c_void {
    if !self_obj.is_null() {
        return self_obj;
    }
    (&raw const PY_SDK_ABI_HEADER)
        .cast_mut()
        .cast::<std::ffi::c_void>()
}

/// Native `CPython` `METH_NOARGS` callback returning a non-null C-string pointer for the SDK version
/// (`_get_sdk_version`).
///
/// # Arguments
///
/// * `_self_obj` - Module object pointer (`PyObject*`).
/// * `_args` - Unused argument pointer (`NULL` for `METH_NOARGS`).
///
/// # Safety
///
/// Returns a static `'static` C-string pointer cast to `*mut c_void`.
#[unsafe(no_mangle)]
pub const unsafe extern "C" fn py_sdk_get_version(
    _self_obj: *mut std::ffi::c_void,
    _args: *mut std::ffi::c_void,
) -> *mut std::ffi::c_void {
    c"0.0.52".as_ptr().cast_mut().cast::<std::ffi::c_void>()
}

/// Native `CPython` `METH_NOARGS` callback returning the default chip count (`4`) encoded as a pointer-sized integer
/// (`_get_chip_count`).
///
/// # Arguments
///
/// * `_self_obj` - Module object pointer (`PyObject*`).
/// * `_args` - Unused argument pointer (`NULL` for `METH_NOARGS`).
///
/// # Safety
///
/// Safe to invoke from C/Rust tests.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn py_sdk_get_chip_count(
    _self_obj: *mut std::ffi::c_void,
    _args: *mut std::ffi::c_void,
) -> *mut std::ffi::c_void {
    let count = LibtpuSdkCoreClient::new().chip_coordinates.len();
    std::ptr::without_provenance_mut(count)
}

/// Native `CPython` `Py_mod_exec` multi-phase initialization slot callback (`py_sdk_mod_exec`).
///
/// # Arguments
///
/// * `module_obj` - Module object pointer (`PyObject*`).
///
/// # Safety
///
/// Returns `0` (`success`) when `module_obj` is non-null, or `-1` when `module_obj` is null.
#[unsafe(no_mangle)]
pub const unsafe extern "C" fn py_sdk_mod_exec(module_obj: *mut std::ffi::c_void) -> i32 {
    if module_obj.is_null() { -1 } else { 0 }
}

/// Static `PyMethodDef` table exported by [`PySdkAbiHeader`].
pub static PY_SDK_METHOD_TABLE: [CPythonMethodDef; 4] = [
    CPythonMethodDef {
        ml_name: c"_get_sdk_abi_header".as_ptr(),
        ml_meth: Some(py_sdk_get_abi_header),
        ml_flags: METH_NOARGS,
        pad: 0,
        ml_doc: c"Returns the static PySdkAbiHeader descriptor pointer.".as_ptr(),
    },
    CPythonMethodDef {
        ml_name: c"_get_sdk_version".as_ptr(),
        ml_meth: Some(py_sdk_get_version),
        ml_flags: METH_NOARGS,
        pad: 0,
        ml_doc: c"Returns the libtpu.sdk version C-string pointer.".as_ptr(),
    },
    CPythonMethodDef {
        ml_name: c"_get_chip_count".as_ptr(),
        ml_meth: Some(py_sdk_get_chip_count),
        ml_flags: METH_NOARGS,
        pad: 0,
        ml_doc: c"Returns the discovered TPU chip count.".as_ptr(),
    },
    CPythonMethodDef {
        ml_name: std::ptr::null(),
        ml_meth: None,
        ml_flags: 0,
        pad: 0,
        ml_doc: std::ptr::null(),
    },
];

/// Static `PyModuleDef_Slot` table for `libtpu.sdk`.
pub static PY_SDK_SLOT_TABLE: [CPythonModuleDefSlot; 2] = [
    CPythonModuleDefSlot {
        slot: PY_MOD_EXEC_SLOT,
        pad: 0,
        value: Some(py_sdk_mod_exec),
    },
    CPythonModuleDefSlot {
        slot: 0,
        pad: 0,
        value: None,
    },
];

/// `CPython` Stable ABI (`abi3`) module definition (`PyModuleDef`) returned by [`PyInit_sdk`].
#[repr(C)]
#[derive(Debug)]
pub struct PySdkAbiHeader {
    /// Base `PyModuleDef_Base` header (`PyModuleDef_HEAD_INIT`).
    pub m_base: CPythonModuleDefBase,
    /// Null-terminated C module name (`"sdk\0"`).
    pub m_name: *const std::os::raw::c_char,
    /// Null-terminated C module docstring.
    pub m_doc: *const std::os::raw::c_char,
    /// Per-module state size in bytes (`-1` for stateless global module).
    pub m_size: isize,
    /// Method table pointer (`m_methods` -> [`PY_SDK_METHOD_TABLE`]).
    pub m_methods: *const CPythonMethodDef,
    /// Multi-phase slot table pointer (`m_slots`, `NULL` for single-phase `PyModule_Create2` compatibility).
    pub m_slots: *const CPythonModuleDefSlot,
    /// GC traverse callback (`m_traverse`, `0`).
    pub m_traverse: usize,
    /// GC clear callback (`m_clear`, `0`).
    pub m_clear: usize,
    /// Module free callback (`m_free`, `0`).
    pub m_free: usize,
    /// `LibtpuSdk_Api` struct size in bytes (`144`).
    pub struct_size: usize,
    /// PJRT ABI minor version (`116`).
    pub abi_version: usize,
}

// SAFETY: `PySdkAbiHeader` contains only static `'static` C-string/table pointers and integer fields.
unsafe impl Sync for PySdkAbiHeader {}

/// Static [`PySdkAbiHeader`] (`PyModuleDef`) singleton exposed to C/Python extension loaders via [`PyInit_sdk`].
static PY_SDK_ABI_HEADER: PySdkAbiHeader = PySdkAbiHeader {
    m_base: CPythonModuleDefBase {
        ob_base: CPythonObjectHead {
            ob_refcnt: 1,
            ob_type: 0,
        },
        m_init: 0,
        m_index: 0,
        m_copy: 0,
    },
    m_name: c"sdk".as_ptr(),
    m_doc: c"libtpu.sdk CPython Stable ABI (abi3) module".as_ptr(),
    m_size: -1,
    m_methods: PY_SDK_METHOD_TABLE.as_ptr(),
    m_slots: PY_SDK_SLOT_TABLE.as_ptr(),
    m_traverse: 0,
    m_clear: 0,
    m_free: 0,
    struct_size: 144,
    abi_version: 116,
};

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
        let client = PjrtCoreClient::from_simulated(&config);
        Ok(Self { config, client })
    }

    /// Returns the PJRT API version tuple `(major, minor)` (`(0, 116)`).
    #[must_use]
    pub const fn pjrt_api_version() -> (i32, i32) {
        (0, 116)
    }

    /// Returns the raw [`PySdkAbiHeader`] vtable descriptor pointer for `libtpu.sdk` consumers.
    #[must_use]
    pub const fn sdk_api_ptr() -> *const PySdkAbiHeader {
        &raw const PY_SDK_ABI_HEADER
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
        let mut host_outs = Vec::with_capacity(outs.len());
        for out_buf in outs {
            host_outs.push(out_buf.to_host_bytes()?);
        }
        Ok(host_outs)
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
        let ast_mod = libtpu_core::compiler::parse_mlir_ast_module(mlir_source)?;
        let bc_bytes = libtpu_core::compiler::MlirBytecodeHeader::encode_from_ast_module(&ast_mod);
        self.execute_shaped_tensor(device_id, "mlir", &bc_bytes, shape, inputs)
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
            sdk_client: LibtpuSdkCoreClient::new(),
            driver: TpuCoreDriver::open(cfg),
            profiler: TpuCoreProfiler::new(),
        }
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

    /// Resolves a hierarchical `DeepSea` CSR path in the 13,261+ CSR catalog for this bridge's
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

    /// Resolves a physical BAR0 MMIO byte offset in the 13,261+ CSR catalog for this bridge's
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

/// `CPython` Stable ABI (`abi3`) version constant (`1013`).
pub const PYTHON_ABI3_VERSION: i32 = 1013;

/// Function pointer type for `CPython`'s `Py_IsInitialized` (`() -> c_int`).
pub type PyIsInitializedFn = unsafe extern "C" fn() -> i32;

/// Function pointer type for `CPython`'s `PyModule_Create2` (`(*const PySdkAbiHeader, c_int) -> *const PySdkAbiHeader`).
pub type PyModuleCreate2Fn =
    unsafe extern "C" fn(*const PySdkAbiHeader, i32) -> *const PySdkAbiHeader;

/// Initializes the `libtpu.sdk` `CPython` `abi3` module via `PyModule_Create2`.
///
/// When an active `CPython` interpreter is detected (`is_init_fn() != 0`), invokes
/// `create_fn`, falling back to the static [`PySdkAbiHeader`] descriptor when invoked outside `CPython`.
///
/// # Arguments
///
/// * `is_init_fn` - Optional `Py_IsInitialized` function pointer.
/// * `create_fn` - Optional `PyModule_Create2` function pointer.
#[must_use]
pub fn init_py_sdk_module(
    is_init_fn: Option<PyIsInitializedFn>,
    create_fn: Option<PyModuleCreate2Fn>,
) -> *const PySdkAbiHeader {
    if let (Some(is_init), Some(create)) = (is_init_fn, create_fn) {
        // SAFETY: `is_init` is a valid C function pointer with no arguments.
        let active = unsafe { is_init() };
        if active != 0 {
            // SAFETY: `PY_SDK_ABI_HEADER` is a valid static `PyModuleDef`-compatible structure.
            let created = unsafe { create(&raw const PY_SDK_ABI_HEADER, PYTHON_ABI3_VERSION) };
            if !created.is_null() {
                return created;
            }
        }
    }
    &raw const PY_SDK_ABI_HEADER
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
    let hal = libtpu_core::hal::LinuxDeviceHal::open(
        libtpu_core::hal::KernelDriverMode::Gasket,
        libtpu_core::hal::EmulatedKernelDriverBackend::default(),
        &cfg,
    )?;
    hal.service_continuation_queue(envelope_bytes)
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

/// Native C-ABI entrypoint matching `PyInit_sdk` in `libtpu/sdk.so` (`0x004198c0`).
///
/// Returns the static [`PySdkAbiHeader`] descriptor pointer so C/Python
/// extension loaders and `dlsym(..., "PyInit_sdk")` callers can inspect `libtpu.sdk`.
#[unsafe(no_mangle)]
pub const extern "C" fn PyInit_sdk() -> *const PySdkAbiHeader {
    &raw const PY_SDK_ABI_HEADER
}

/// Alias C-ABI module initializer (`PyInit_libtpu_py`) returning [`PySdkAbiHeader`].
#[unsafe(no_mangle)]
pub const extern "C" fn PyInit_libtpu_py() -> *const PySdkAbiHeader {
    &raw const PY_SDK_ABI_HEADER
}

#[cfg(test)]
#[allow(clippy::too_many_lines)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    /// Mock `Py_IsInitialized` returning `0` (not initialized).
    unsafe extern "C" fn mock_py_not_initialized() -> i32 {
        0
    }

    /// Mock `Py_IsInitialized` returning `1` (initialized).
    unsafe extern "C" fn mock_py_initialized() -> i32 {
        1
    }

    /// Mock `PyModule_Create2` returning `def` (`*const PySdkAbiHeader`).
    ///
    /// # Arguments
    ///
    /// * `def` - Module definition pointer.
    /// * `_ver` - Module ABI version integer.
    unsafe extern "C" fn mock_py_module_create_ok(
        def: *const PySdkAbiHeader,
        _ver: i32,
    ) -> *const PySdkAbiHeader {
        def
    }

    /// Mock `PyModule_Create2` returning `NULL`.
    ///
    /// # Arguments
    ///
    /// * `_def` - Module definition pointer.
    /// * `_ver` - Module ABI version integer.
    unsafe extern "C" fn mock_py_module_create_null(
        _def: *const PySdkAbiHeader,
        _ver: i32,
    ) -> *const PySdkAbiHeader {
        std::ptr::null()
    }

    #[test]
    fn test_py_tpu_runtime() {
        assert_eq!(PyTpuRuntime::pjrt_api_version(), (0, 116));
        assert_eq!(protodesc_293_catalog_count(), 293);
        assert_eq!(verify_protodesc_cold_293(), Ok(293));
        let llo_sample = "[pc=0000 gen=v5 stall=0] { S0(op=1, $s0 <- $s0, $s0, imm=0) | V0(AddF32, $v0 <- $v0, $v1, $v0) }";
        let imem_out = assemble_llo_vliw(llo_sample);
        assert!(imem_out.is_ok());
        if let Ok(imem) = imem_out {
            assert_eq!(imem.len(), 64);
            assert!(disassemble_llo_vliw(&imem).contains("pc=0000"));
        }
        assert!(assemble_llo_vliw("bad_line").is_err());
        let hc_env = libtpu_core::proto::HostCommandEnvelopeProto {
            opcode: 1,
            continuation_id: 10,
            device_id: 0,
            core_index: 0,
            handler_id: 2,
            payload: b"pkt".to_vec(),
        };
        assert_eq!(service_host_command_envelope(&hc_env.encode()), Ok(12));
        assert!(service_host_command_envelope(&[0xFF, 0xFF, 0xFF]).is_err());
        let ar_res = execute_multi_replica_all_reduce_f32_4(&[
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 20.0, 30.0, 40.0],
        ]);
        assert_eq!(
            ar_res,
            vec![[11.0, 22.0, 33.0, 44.0], [11.0, 22.0, 33.0, 44.0]]
        );
        let bc_res = execute_multi_replica_collective_broadcast_f32_4(
            &[[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]],
            1,
        );
        assert_eq!(
            bc_res,
            vec![[10.0, 20.0, 30.0, 40.0], [10.0, 20.0, 30.0, 40.0]]
        );
        let rag_res = execute_multi_replica_ragged_all_to_all_f32_4(&[
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 20.0, 30.0, 40.0],
        ]);
        assert_eq!(rag_res.len(), 2);
        assert!(!PyTpuRuntime::sdk_api_ptr().is_null());
        assert!(!PyInit_sdk().is_null());
        assert_eq!(PyInit_libtpu_py(), PyInit_sdk());
        assert!(!PY_SDK_ABI_HEADER.m_methods.is_null());
        assert!(!PY_SDK_ABI_HEADER.m_slots.is_null());
        assert_eq!(PY_SDK_METHOD_TABLE[0].ml_flags, METH_NOARGS);
        assert_ne!(
            format!("{:?}{:?}", PY_SDK_METHOD_TABLE[0], PY_SDK_SLOT_TABLE[0]),
            ""
        );
        assert_eq!(PY_SDK_SLOT_TABLE[0].slot, PY_MOD_EXEC_SLOT);
        // SAFETY: Testing `py_sdk_get_abi_header`, `py_sdk_get_version`, `py_sdk_get_chip_count`, and `py_sdk_mod_exec` with null and non-null pointers.
        unsafe {
            let hdr_ptr = py_sdk_get_abi_header(std::ptr::null_mut(), std::ptr::null_mut());
            assert!(!hdr_ptr.is_null());
            let ver_ptr = py_sdk_get_version(std::ptr::null_mut(), std::ptr::null_mut());
            assert!(!ver_ptr.is_null());
            let chip_cnt_ptr = py_sdk_get_chip_count(std::ptr::null_mut(), std::ptr::null_mut());
            assert_eq!(chip_cnt_ptr.addr(), 4);
            let mut dummy_mod: i32 = 1;
            let self_ptr = std::ptr::from_mut(&mut dummy_mod).cast::<std::ffi::c_void>();
            assert_eq!(
                py_sdk_get_abi_header(self_ptr, std::ptr::null_mut()),
                self_ptr
            );
            assert_eq!(py_sdk_mod_exec(std::ptr::null_mut()), -1);
            assert_eq!(py_sdk_mod_exec(self_ptr), 0);
        }
        assert_eq!(init_py_sdk_module(None, None), &raw const PY_SDK_ABI_HEADER);
        assert_eq!(
            init_py_sdk_module(Some(mock_py_not_initialized), None),
            &raw const PY_SDK_ABI_HEADER
        );
        assert_eq!(
            init_py_sdk_module(
                Some(mock_py_not_initialized),
                Some(mock_py_module_create_ok),
            ),
            &raw const PY_SDK_ABI_HEADER
        );
        assert_eq!(
            init_py_sdk_module(Some(mock_py_initialized), Some(mock_py_module_create_ok)),
            &raw const PY_SDK_ABI_HEADER
        );
        assert_eq!(
            init_py_sdk_module(Some(mock_py_initialized), Some(mock_py_module_create_null)),
            &raw const PY_SDK_ABI_HEADER
        );
        assert_eq!(PY_SDK_ABI_HEADER.struct_size, 144);
        assert_eq!(PY_SDK_ABI_HEADER.abi_version, 116);
        assert_eq!(PY_SDK_ABI_HEADER.m_size, -1);
        assert_eq!(PY_SDK_ABI_HEADER.m_base.ob_base.ob_refcnt, 1);
        assert_eq!(PY_SDK_ABI_HEADER.m_base.clone(), PY_SDK_ABI_HEADER.m_base);
        assert_eq!(
            PY_SDK_ABI_HEADER.m_base.ob_base.clone(),
            PY_SDK_ABI_HEADER.m_base.ob_base
        );
        assert!(!PY_SDK_ABI_HEADER.m_name.is_null());
        assert!(!PY_SDK_ABI_HEADER.m_doc.is_null());
        assert_ne!(format!("{PY_SDK_ABI_HEADER:?}"), "");
        assert!(matches!(
            PyTpuRuntime::inspect_binary_symbol_names(b"not_a_binary"),
            Err(libtpu_core::error::LibTpuError::GoblinParseFailed(_))
        ));
        let mut macho_hdr = Vec::with_capacity(32);
        macho_hdr.extend_from_slice(&0xfeed_facf_u32.to_le_bytes());
        macho_hdr.extend_from_slice(&0x0100_0007_u32.to_le_bytes());
        macho_hdr.extend_from_slice(&0_u32.to_le_bytes());
        macho_hdr.extend_from_slice(&6_u32.to_le_bytes());
        macho_hdr.resize(32, 0);
        assert!(PyTpuRuntime::inspect_binary_symbol_names(&macho_hdr).is_ok());
        assert!(matches!(
            PyTpuRuntime::from_init_args("--tpu_generation=invalid"),
            Err(libtpu_core::error::LibTpuError::InvalidArgument {
                parameter: "generation",
                ..
            })
        ));

        let rt_res = PyTpuRuntime::from_init_args("--tpu_generation=v6e");
        assert!(rt_res.is_ok());
        if let Ok(rt) = rt_res {
            assert_ne!(format!("{rt:?}"), "");
            assert_eq!(rt.generation(), TpuGeneration::V6e);
            assert_eq!(rt.device_count(), 4);
            assert_ne!(rt.serialize_topology(), [] as [u8; 0]);
            let ser_res = rt.compile_and_serialize("mlir", b"module @py_add { stablehlo.add }");
            assert!(ser_res.is_ok());
            if let Ok(ser_bytes) = ser_res {
                let fp = rt.deserialize_executable_fingerprint(&ser_bytes);
                assert_eq!(fp.ok().map(|s| s.len()), Some(32));
            }
            assert!(matches!(
                rt.compile_and_serialize("bad", b""),
                Err(libtpu_core::error::LibTpuError::CompilationFailed { .. })
            ));
            assert!(matches!(
                rt.deserialize_executable_fingerprint(b"bad"),
                Err(libtpu_core::error::LibTpuError::DeserializationFailed { .. })
            ));
            let sum = rt.execute_f32_4(
                DeviceId(0),
                b"module @py_add { stablehlo.add }",
                &[1.0, 2.0, 3.0, 4.0],
            );
            assert_eq!(sum.ok(), Some([2.0, 4.0, 6.0, 8.0]));
            let bc_sh = Shape {
                element_type: ElementType::F32,
                dims: vec![4],
                dynamic_dims: vec![false],
                layout: libtpu_core::types::MemoryLayout::default_tpu_tiled(1),
            };
            let bc_in = ElementType::F32.pack_from_f64_slice(&[1.0, 2.0, 3.0, 4.0]);
            let bc_res = rt.compile_mlir_bytecode_and_execute(
                DeviceId(0),
                "module @py_bc { func.func public @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> { %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32> return %0 : tensor<4xf32> } }",
                &bc_sh,
                &[&bc_in, &bc_in],
            );
            assert!(bc_res.is_ok());
            let emb_gen = libtpu_core::driver::TpuCoreEmbeddingEngine::new();
            let hlo_pb = emb_gen.generate_xla_computation("recv_activations");
            let hlo_res =
                rt.compile_and_execute_hlo_proto(DeviceId(0), &hlo_pb, &bc_sh, &[&bc_in, &bc_in]);
            assert!(hlo_res.is_ok());
            assert!(
                rt.compile_mlir_bytecode_and_execute(DeviceId(0), "", &bc_sh, &[&bc_in])
                    .is_err()
            );
            assert!(matches!(
                rt.execute_f32_4(DeviceId(0), b"", &[1.0; 4]),
                Err(libtpu_core::error::LibTpuError::CompilationFailed { .. })
            ));
            assert!(matches!(
                rt.execute_f32_4(DeviceId(99), b"module @py_add { stablehlo.add }", &[1.0; 4]),
                Err(libtpu_core::error::LibTpuError::DeviceNotFound { device_id: 99 })
            ));
            let _ = rt
                .client
                .hal
                .poison_device(DeviceId(0), String::from("poisoned"));
            assert!(matches!(
                rt.execute_f32_4(DeviceId(0), b"module @py_add { stablehlo.add }", &[1.0; 4]),
                Err(libtpu_core::error::LibTpuError::ExecutionFailed { device_id: 0, .. })
            ));
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
        assert!(matches!(
            bridge.get_metric_values("unknown_metric"),
            Err(libtpu_core::error::LibTpuError::MetricNotFound { .. })
        ));
        let xspace = bridge.capture_xspace();
        assert!(xspace.is_ok());
        let csr_off = bridge.lookup_csr_by_path("tc_0.tcs_core_state.RUN_STATUS");
        assert!(csr_off.is_some());
        if let Some(off) = csr_off {
            assert!(bridge.lookup_csr_by_offset(off).is_some());
        }
        assert_ne!(
            bridge.collect_profiler_tool_data("trace_viewer.json"),
            [] as [u8; 0]
        );
        let _ = bridge.profiler.start();
        assert!(matches!(
            bridge.capture_xspace(),
            Err(libtpu_core::error::LibTpuError::ProfilerAlreadyActive)
        ));
    }
}

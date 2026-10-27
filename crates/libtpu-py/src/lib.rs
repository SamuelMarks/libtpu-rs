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
    /// Method table pointer (`m_methods`, `0`).
    pub m_methods: usize,
    /// Multi-phase slot table pointer (`m_slots`, `0`).
    pub m_slots: usize,
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

// SAFETY: `PySdkAbiHeader` contains only static `'static` C-string pointers and integer fields.
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
    m_methods: 0,
    m_slots: 0,
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
        let loaded = self.client.compile_and_load("mlir", mlir_code, b"")?;
        let buf0 = self.client.buffer_from_host(device_id, shape, &b_in)?;
        let outs = loaded.execute_on_device(device_id, MemorySpaceId(0), &[&buf0, &buf0])?;
        let mut out_arr = [0.0_f32; 4];
        for out_buf in outs {
            let mut raw = [0_u8; 16];
            let _ = self
                .client
                .hal
                .read_device_memory(device_id, out_buf.address, &mut raw);
            for (dst, chunk) in out_arr.iter_mut().zip(raw.chunks_exact(4)) {
                let mut bytes = [0_u8; 4];
                bytes.copy_from_slice(chunk);
                *dst = f32::from_le_bytes(bytes);
            }
        }
        Ok(out_arr)
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
        self.profiler.stop()?;
        Ok(self.profiler.collect_xspace_bytes())
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

/// Native C-ABI entrypoint matching `PyInit_sdk` in `libtpu/sdk.so` (`0x004198c0`).
///
/// Returns the static [`PySdkAbiHeader`] descriptor pointer so C/Python
/// extension loaders and `dlsym(..., "PyInit_sdk")` callers can inspect `libtpu.sdk`.
#[unsafe(no_mangle)]
pub const extern "C" fn PyInit_sdk() -> *const PySdkAbiHeader {
    &raw const PY_SDK_ABI_HEADER
}

#[cfg(test)]
#[allow(for_loops_over_fallibles)]
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
        assert!(!PyTpuRuntime::sdk_api_ptr().is_null());
        assert!(!PyInit_sdk().is_null());
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
        assert!(PyTpuRuntime::inspect_binary_symbol_names(b"not_a_binary").is_err());
        let mut macho_hdr = Vec::with_capacity(32);
        macho_hdr.extend_from_slice(&0xfeed_facf_u32.to_le_bytes());
        macho_hdr.extend_from_slice(&0x0100_0007_u32.to_le_bytes());
        macho_hdr.extend_from_slice(&0_u32.to_le_bytes());
        macho_hdr.extend_from_slice(&6_u32.to_le_bytes());
        macho_hdr.resize(32, 0);
        assert!(PyTpuRuntime::inspect_binary_symbol_names(&macho_hdr).is_ok());
        assert!(PyTpuRuntime::from_init_args("--tpu_generation=invalid").is_err());

        let rt_res = PyTpuRuntime::from_init_args("--tpu_generation=v6e");
        assert!(rt_res.is_ok());
        for rt in rt_res {
            assert_ne!(format!("{rt:?}"), "");
            assert_eq!(rt.generation(), TpuGeneration::V6e);
            assert_eq!(rt.device_count(), 4);
            assert_ne!(rt.serialize_topology(), [] as [u8; 0]);
            let ser_res = rt.compile_and_serialize("mlir", b"module @py_add { stablehlo.add }");
            assert!(ser_res.is_ok());
            for ser_bytes in ser_res {
                let fp = rt.deserialize_executable_fingerprint(&ser_bytes);
                assert_eq!(fp.ok().map(|s| s.len()), Some(32));
            }
            assert!(rt.compile_and_serialize("bad", b"").is_err());
            assert!(rt.deserialize_executable_fingerprint(b"bad").is_err());
            let sum = rt.execute_f32_4(
                DeviceId(0),
                b"module @py_add { stablehlo.add }",
                &[1.0, 2.0, 3.0, 4.0],
            );
            assert_eq!(sum.ok(), Some([2.0, 4.0, 6.0, 8.0]));
            assert!(rt.execute_f32_4(DeviceId(0), b"", &[1.0; 4]).is_err());
            assert!(
                rt.execute_f32_4(DeviceId(99), b"module @py_add { stablehlo.add }", &[1.0; 4],)
                    .is_err()
            );
            let _ = rt
                .client
                .hal
                .poison_device(DeviceId(0), String::from("poisoned"));
            assert!(
                rt.execute_f32_4(DeviceId(0), b"module @py_add { stablehlo.add }", &[1.0; 4],)
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
        assert!(bridge.get_metric_values("unknown_metric").is_err());
        let xspace = bridge.capture_xspace();
        assert!(xspace.is_ok());
        let _ = bridge.profiler.start();
        assert!(bridge.capture_xspace().is_err());
    }
}

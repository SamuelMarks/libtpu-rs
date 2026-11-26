//! Native `PyO3` (`abi3`) `CPython` extension modules (`libtpu._C` and `libtpu/sdk.so`).
//!
//! Exports PEP 489 multi-phase module initializers [`PyInit__C`], [`PyInit_sdk`], and
//! [`PyInit_libtpu_py`] backed by genuine `pyo3` (`abi3`) runtime bindings and mapping every
//! [`libtpu_core::error::LibTpuError`] cleanly to a typed Python `TpuError` exception.

use crate::dlpack::{DlDataType, DlDevice, consume_dlpack_capsule, export_dlpack_capsule};
use crate::{PySdkBridge, PyTpuRuntime};
use core::cell::UnsafeCell;
use core::ffi::{CStr, c_int, c_void};
use libtpu_core::config::TpuInitConfig;
use libtpu_core::error::LibTpuError;
use libtpu_core::types::{DeviceId, ElementType, MemoryLayout, Shape, TpuGeneration};
use pyo3::PyTypeInfo;
use pyo3::exceptions::PyRuntimeError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBytes, PyCFunction, PyDict, PyTuple, PyType};

/// Cached `libtpu.TpuError` Python exception class (`RuntimeError` subclass).
static TPU_ERROR_CLASS: PyOnceLock<Py<PyType>> = PyOnceLock::new();

/// Number of PEP 489 / PEP 703 `PyModuleDef_Slot` entries (`3` on `Py_GIL_DISABLED`, `2` otherwise).
#[cfg(Py_GIL_DISABLED)]
const MODULE_SLOT_COUNT: usize = 3;

/// Number of PEP 489 `PyModuleDef_Slot` entries on standard GIL `CPython` builds.
#[cfg(not(Py_GIL_DISABLED))]
const MODULE_SLOT_COUNT: usize = 2;

/// Thread-safe static wrapper around a PEP 489 `PyModuleDef_Slot` table.
struct SyncModuleSlots(UnsafeCell<[ffi::PyModuleDef_Slot; MODULE_SLOT_COUNT]>);

// SAFETY: `PyModuleDef_Slot` entries contain immutable function pointers read by `CPython` under the GIL.
unsafe impl Sync for SyncModuleSlots {}

/// Thread-safe static wrapper around a `CPython` `PyModuleDef` descriptor.
struct SyncModuleDef(UnsafeCell<ffi::PyModuleDef>);

// SAFETY: `CPython` mutates `PyModuleDef::m_base` atomically or under the import lock / GIL.
unsafe impl Sync for SyncModuleDef {}

/// Allocates a Python exception type `qualname` subclassing `RuntimeError`, falling back to
/// `RuntimeError` if `qualname` is rejected by `CPython`.
///
/// # Arguments
///
/// * `py` - Active `CPython` GIL token.
/// * `qualname` - Dotted `"module.ClassName"` exception identifier.
pub(crate) fn create_exception_or_fallback(py: Python<'_>, qualname: &'static CStr) -> Py<PyType> {
    let base = PyRuntimeError::type_object(py);
    PyErr::new_type(
        py,
        qualname,
        Some(c"Typed exception raised when a libtpu C/Rust runtime operation fails."),
        Some(&base),
        None,
    )
    .unwrap_or_else(|_| base.unbind())
}

/// Returns the shared `libtpu.TpuError` Python exception type, initializing it once per process.
///
/// # Arguments
///
/// * `py` - Active `CPython` GIL token.
#[must_use]
pub fn tpu_error_type(py: Python<'_>) -> Bound<'_, PyType> {
    TPU_ERROR_CLASS
        .get_or_init(py, || create_exception_or_fallback(py, c"libtpu.TpuError"))
        .bind(py)
        .clone()
}

/// Converts a [`LibTpuError`] into a typed Python `libtpu.TpuError(code, message)` exception
/// carrying `.code` (`PjrtErrorCode`) and `.sdk_code` (`LibtpuSdkErrorCode`) attributes.
///
/// # Arguments
///
/// * `py` - Active `CPython` GIL token.
/// * `err` - Workspace [`LibTpuError`] value to convert.
#[must_use]
pub fn lib_tpu_error_to_py_err(py: Python<'_>, err: &LibTpuError) -> PyErr {
    let pjrt_code = err.error_code() as i32;
    let sdk_code = err.sdk_code() as i32;
    let message = err.to_string();
    let exc_cls = tpu_error_type(py);
    let py_err = PyErr::from_type(exc_cls, (pjrt_code, message.clone()));
    let val = py_err.value(py);
    let _ = val.setattr("code", pjrt_code);
    let _ = val.setattr("sdk_code", sdk_code);
    let _ = val.setattr("message", message);
    py_err
}

/// Attaches a Python attribute `name` onto `module` from a fallible `Bound<PyAny>` constructor.
///
/// # Arguments
///
/// * `module` - Target Python module or object.
/// * `name` - Attribute identifier.
/// * `value` - Fallible attribute value result.
///
/// # Errors
///
/// Returns [`PyErr`] if `value` is `Err` or `module.setattr` fails.
pub(crate) fn set_module_attr(
    module: &Bound<'_, PyAny>,
    name: &str,
    value: PyResult<Bound<'_, PyAny>>,
) -> PyResult<()> {
    let bound_val = value?;
    module.setattr(name, bound_val)
}

/// Registers a typed closure function `name` on `module`.
///
/// # Arguments
///
/// * `py` - Active `CPython` GIL token.
/// * `module` - Target Python module.
/// * `attr_name` - Attribute name to bind on `module`.
/// * `c_name` - NUL-terminated function name for `__name__`.
/// * `c_doc` - NUL-terminated docstring for `__doc__`.
/// * `closure` - Rust closure accepting `(args, kwargs)`.
///
/// # Errors
///
/// Returns [`PyErr`] if `PyCFunction` allocation or `setattr` fails.
fn register_closure<F, R>(
    py: Python<'_>,
    module: &Bound<'_, PyAny>,
    attr_name: &str,
    c_name: &'static CStr,
    c_doc: &'static CStr,
    closure: F,
) -> PyResult<()>
where
    F: Fn(&Bound<'_, PyTuple>, Option<&Bound<'_, PyDict>>) -> PyResult<R> + Send + Sync + 'static,
    for<'p> R: IntoPyObject<'p>,
{
    let cfn = PyCFunction::new_closure(py, Some(c_name), Some(c_doc), closure).map(Bound::into_any);
    set_module_attr(module, attr_name, cfn)
}

/// Parses a generation name (`"v2"`..`"v7x"`) into [`TpuGeneration`], mapping invalid names to `TpuError`.
///
/// # Arguments
///
/// * `py` - Active `CPython` GIL token.
/// * `gen_name` - Generation identifier string.
///
/// # Errors
///
/// Returns `TpuError` (`PyErr`) if `gen_name` is unrecognized.
fn parse_generation_py(py: Python<'_>, gen_name: &str) -> PyResult<TpuGeneration> {
    TpuGeneration::from_name(gen_name).map_err(|e| lib_tpu_error_to_py_err(py, &e))
}

/// Resolves an integer XLA `PrimitiveType` discriminant (`1..=28`) into [`ElementType`].
///
/// # Arguments
///
/// * `py` - Active `CPython` GIL token.
/// * `dtype_code` - Raw XLA `PrimitiveType` integer.
///
/// # Errors
///
/// Returns `TpuError` (`PyErr`) if `dtype_code` is out of range or `Invalid`.
fn parse_element_type_py(py: Python<'_>, dtype_code: i32) -> PyResult<ElementType> {
    let elem = ElementType::from_raw(dtype_code).map_err(|e| lib_tpu_error_to_py_err(py, &e))?;
    if elem == ElementType::Invalid {
        return Err(lib_tpu_error_to_py_err(
            py,
            &LibTpuError::InvalidArgument {
                parameter: "dtype_code",
                reason: format!("unsupported XLA PrimitiveType discriminant {dtype_code}"),
            },
        ));
    }
    Ok(elem)
}

/// Populates core metadata, dtype codecs, and compiler helpers onto `libtpu._C`.
///
/// # Arguments
///
/// * `py` - Active `CPython` GIL token.
/// * `module` - Target `libtpu._C` module object.
///
/// # Errors
///
/// Returns [`PyErr`] if module attribute registration fails.
fn register_c_core_functions(py: Python<'_>, module: &Bound<'_, PyAny>) -> PyResult<()> {
    let mut res = module.setattr("__version__", libtpu_core::LIBTPU_PYPI_VERSION);
    res = res.and_then(|()| module.setattr("TpuError", tpu_error_type(py)));
    res = res.and_then(|()| {
        register_closure(
            py,
            module,
            "pjrt_api_version",
            c"pjrt_api_version",
            c"Returns the `(major, minor)` PJRT C API version tuple.",
            |_args, _kwargs| Ok(PyTpuRuntime::pjrt_api_version()),
        )
    });
    res = res.and_then(|()| {
        register_closure(
            py,
            module,
            "protodesc_cold_count",
            c"protodesc_cold_count",
            c"Returns the number of reconstructed `.proto` schemas in `protodesc_cold`.",
            |_args, _kwargs| Ok(libtpu_core::proto::protodesc_cold_293_file_count()),
        )
    });
    res = res.and_then(|()| {
        register_closure(
            py,
            module,
            "parse_init_args",
            c"parse_init_args",
            c"Validates `LIBTPU_INIT_ARGS` flags and returns `(generation, process_index, num_slices, slice_id)`.",
            |args, _kwargs| {
                let py_tok = args.py();
                let (raw_flags,): (String,) = args.extract()?;
                let mut cfg = TpuInitConfig::default();
                cfg.apply_init_args(&raw_flags)
                    .map_err(|e| lib_tpu_error_to_py_err(py_tok, &e))?;
                Ok((
                    cfg.generation.to_string(),
                    cfg.process_index.0,
                    cfg.megascale_num_slices,
                    cfg.megascale_slice_id,
                ))
            },
        )
    });
    res = res.and_then(|()| {
        register_closure(
            py,
            module,
            "pack_dtype_f64",
            c"pack_dtype_f64",
            c"Packs a list of `float` values into exact TPU `ElementType` wire bytes.",
            |args, _kwargs| {
                let py_tok = args.py();
                let (dtype_code, values): (i32, Vec<f64>) = args.extract()?;
                let elem = parse_element_type_py(py_tok, dtype_code)?;
                let packed = elem.pack_from_f64_slice(&values);
                Ok(PyBytes::new(py_tok, &packed).unbind())
            },
        )
    });
    res = res.and_then(|()| {
        register_closure(
            py,
            module,
            "unpack_dtype_f64",
            c"unpack_dtype_f64",
            c"Unpacks TPU `ElementType` wire bytes into a list of `float` values.",
            |args, _kwargs| {
                let py_tok = args.py();
                let (dtype_code, raw_bytes): (i32, Vec<u8>) = args.extract()?;
                let elem = parse_element_type_py(py_tok, dtype_code)?;
                Ok(elem.unpack_to_f64_vec(&raw_bytes))
            },
        )
    });
    res
}

/// Tuple argument signature extracted by `compile_and_execute`.
type CompileExecArgs = (String, Vec<u8>, Vec<i64>, i32, Vec<Vec<u8>>, String);

/// Populates runtime execution, PEP 3118 `memoryview`, `DLPack`, and ELF/Mach-O helpers onto `libtpu._C`.
///
/// # Arguments
///
/// * `py` - Active `CPython` GIL token.
/// * `module` - Target `libtpu._C` module object.
///
/// # Errors
///
/// Returns [`PyErr`] if module attribute registration fails.
#[allow(clippy::too_many_lines)]
fn register_c_io_functions(py: Python<'_>, module: &Bound<'_, PyAny>) -> PyResult<()> {
    let mut res = register_closure(
        py,
        module,
        "compile_and_execute",
        c"compile_and_execute",
        c"Compiles and executes an MLIR/HLO program on `PyTpuRuntime` returning output byte buffers.",
        |args, _kwargs| {
            let py_tok = args.py();
            let (fmt, code, dims, dtype_code, inputs, init_args): CompileExecArgs =
                args.extract()?;
            let elem = parse_element_type_py(py_tok, dtype_code)?;
            let rt = PyTpuRuntime::shared(&init_args)
                .map_err(|e| lib_tpu_error_to_py_err(py_tok, &e))?;
            let rank = dims.len();
            let shape = Shape {
                element_type: elem,
                dynamic_dims: vec![false; rank],
                layout: MemoryLayout::default_tpu_tiled(rank),
                dims,
            };
            let in_refs: Vec<&[u8]> = inputs.iter().map(Vec::as_slice).collect();
            let outs = rt
                .execute_shaped_tensor(DeviceId(0), &fmt, &code, &shape, &in_refs)
                .map_err(|e| lib_tpu_error_to_py_err(py_tok, &e))?;
            let py_outs: Vec<Py<PyBytes>> = outs
                .iter()
                .map(|b| PyBytes::new(py_tok, b).unbind())
                .collect();
            Ok(py_outs)
        },
    );
    res = res.and_then(|()| {
        register_closure(
            py,
            module,
            "buffer_view",
            c"buffer_view",
            c"Constructs a zero-copy PEP 3118 `memoryview` over an exporter object.",
            |args, _kwargs| {
                let py_tok = args.py();
                let (exporter,): (Bound<'_, PyAny>,) = args.extract()?;
                // SAFETY: `exporter.as_ptr()` is a valid borrowed `PyObject*` under the GIL.
                let mv_ptr = unsafe { ffi::PyMemoryView_FromObject(exporter.as_ptr()) };
                // SAFETY: `PyMemoryView_FromObject` returns a new reference or null with Python error set.
                let bound_mv = unsafe { Bound::from_owned_ptr_or_err(py_tok, mv_ptr)? };
                Ok(bound_mv.unbind())
            },
        )
    });
    res = res.and_then(|()| {
        register_closure(
            py,
            module,
            "export_dlpack",
            c"export_dlpack",
            c"Exports contiguous tensor bytes as a `DLPack` v1.0 `dltensor` PyCapsule.",
            |args, _kwargs| {
                let py_tok = args.py();
                let (payload, shape, code, bits, dev_type, dev_id): (
                    Vec<u8>,
                    Vec<i64>,
                    u8,
                    u8,
                    i32,
                    i32,
                ) = args.extract()?;
                export_dlpack_capsule(
                    py_tok,
                    &payload,
                    &shape,
                    DlDataType {
                        code,
                        bits,
                        lanes: 1,
                    },
                    DlDevice {
                        device_type: dev_type,
                        device_id: dev_id,
                    },
                )
                .map(Bound::unbind)
            },
        )
    });
    res = res.and_then(|()| {
        register_closure(
            py,
            module,
            "consume_dlpack",
            c"consume_dlpack",
            c"Consumes a `DLPack` v1.0 `dltensor` PyCapsule into `(bytes, shape, code, bits, lanes, dev_type, dev_id)`.",
            |args, _kwargs| {
                let py_tok = args.py();
                let (cap_obj,): (Bound<'_, PyAny>,) = args.extract()?;
                let consumed = consume_dlpack_capsule(&cap_obj)
                    .map_err(|e| lib_tpu_error_to_py_err(py_tok, &e))?;
                Ok((
                    PyBytes::new(py_tok, &consumed.raw_bytes).unbind(),
                    consumed.shape,
                    consumed.dtype_code,
                    consumed.bits,
                    consumed.lanes,
                    consumed.device_type,
                    consumed.device_id,
                ))
            },
        )
    });
    res = res.and_then(|()| {
        register_closure(
            py,
            module,
            "inspect_binary_symbols",
            c"inspect_binary_symbols",
            c"Parses exported dynamic symbols from an ELF or Mach-O shared library image.",
            |args, _kwargs| {
                let py_tok = args.py();
                let (raw_bin,): (Vec<u8>,) = args.extract()?;
                let report = PyTpuRuntime::inspect_binary_symbol_names(&raw_bin)
                    .map_err(|e| lib_tpu_error_to_py_err(py_tok, &e))?;
                let names: Vec<String> = report
                    .exported_symbols
                    .into_iter()
                    .map(|s| s.name)
                    .collect();
                Ok(names)
            },
        )
    });
    res = res.and_then(|()| {
        register_closure(
            py,
            module,
            "decode_mlir_v6_text",
            c"decode_mlir_v6_text",
            c"Decodes an LLVM/StableHLO `ML\\xefR` bytecode container into synthesized MLIR text.",
            |args, _kwargs| {
                let (raw_bc,): (Vec<u8>,) = args.extract()?;
                Ok(
                    libtpu_core::compiler::mlir_bytecode::decode_llvm_mlir_v6_to_text(&raw_bc)
                        .unwrap_or_default(),
                )
            },
        )
    });
    res = res.and_then(|()| register_runtime_sync_binder(py, module));
    res
}

/// Registers `bind_c_api_runtime_sync(vtable_addr: int)` onto `module` so `libtpu._C` and `sdk.so`
/// synchronize telemetry with `libtpu.so`.
///
/// # Arguments
///
/// * `py` - Active `CPython` GIL token.
/// * `module` - Target Python module object.
///
/// # Errors
///
/// Returns [`PyErr`] if attribute binding fails.
fn register_runtime_sync_binder(py: Python<'_>, module: &Bound<'_, PyAny>) -> PyResult<()> {
    register_closure(
        py,
        module,
        "bind_c_api_runtime_sync",
        c"bind_c_api_runtime_sync",
        c"Binds `libtpu.so`'s `ExternalRuntimeSyncVtable` pointer into `SharedTpuRuntimeState`.",
        |args, _kwargs| {
            let (vtable_addr,): (usize,) = args.extract()?;
            let ptr = vtable_addr as *const libtpu_core::runtime_state::ExternalRuntimeSyncVtable;
            // SAFETY: Caller passes either `0` (unbind) or a `'static` vtable exported by `libtpu.so`.
            unsafe {
                libtpu_core::runtime_state::global_shared_runtime_state()
                    .bind_peer_sync_vtable(ptr);
            }
            Ok(())
        },
    )
}

/// Populates the `libtpu._C` native module (`PyInit__C`).
///
/// # Arguments
///
/// * `py` - Active `CPython` GIL token.
/// * `module` - Borrowed Python module object.
///
/// # Errors
///
/// Returns [`PyErr`] if attribute binding fails.
pub(crate) fn populate_c_module(py: Python<'_>, module: &Bound<'_, PyAny>) -> PyResult<()> {
    register_c_core_functions(py, module).and_then(|()| register_c_io_functions(py, module))
}

/// Populates the `libtpu/sdk.so` native module (`PyInit_sdk`).
///
/// # Arguments
///
/// * `py` - Active `CPython` GIL token.
/// * `module` - Borrowed Python module object.
///
/// # Errors
///
/// Returns [`PyErr`] if attribute binding fails.
#[allow(clippy::too_many_lines)]
pub(crate) fn populate_sdk_module(py: Python<'_>, module: &Bound<'_, PyAny>) -> PyResult<()> {
    let mut res = module.setattr("__version__", libtpu_core::LIBTPU_PYPI_VERSION);
    res = res.and_then(|()| module.setattr("TpuError", tpu_error_type(py)));
    res = res.and_then(|()| {
        register_closure(
            py,
            module,
            "list_supported_metrics",
            c"list_supported_metrics",
            c"Returns the canonical list of supported `libtpu.sdk` runtime metric names.",
            |_args, _kwargs| {
                let bridge = PySdkBridge::shared(TpuGeneration::V5e);
                Ok(bridge.sdk_client.list_metric_names())
            },
        )
    });
    res = res.and_then(|()| {
        register_closure(
            py,
            module,
            "get_metric",
            c"get_metric",
            c"Queries `(description, values)` for `metric_name` on `generation`.",
            |args, _kwargs| {
                let py_tok = args.py();
                let (metric_name, gen_name): (String, String) = args.extract()?;
                let generation = parse_generation_py(py_tok, &gen_name)?;
                let bridge = PySdkBridge::shared(generation);
                let (desc, vals) = bridge
                    .sdk_client
                    .get_metric(&metric_name)
                    .map_err(|e| lib_tpu_error_to_py_err(py_tok, &e))?;
                Ok((desc, vals))
            },
        )
    });
    res = res.and_then(|()| {
        register_closure(
            py,
            module,
            "get_chip_coordinates",
            c"get_chip_coordinates",
            c"Returns `(x, y, z, chip_id)` tuples for all chips in `generation`.",
            |args, _kwargs| {
                let py_tok = args.py();
                let (gen_name,): (String,) = args.extract()?;
                let generation = parse_generation_py(py_tok, &gen_name)?;
                let bridge = PySdkBridge::shared(generation);
                let coords: Vec<(u32, u32, u32, i32)> = bridge
                    .sdk_client
                    .chip_coordinates
                    .iter()
                    .enumerate()
                    .map(|(idx, c)| (c.x, c.y, c.z, i32::try_from(idx).unwrap_or(0)))
                    .collect();
                Ok(coords)
            },
        )
    });
    res = res.and_then(|()| {
        register_closure(
            py,
            module,
            "get_core_state_summary",
            c"get_core_state_summary",
            c"Returns the human-readable TensorCore state summary for `generation`.",
            |args, _kwargs| {
                let py_tok = args.py();
                let (gen_name, include_hlo) = args
                    .extract::<(String, bool)>()
                    .or_else(|_| args.extract::<(String,)>().map(|(g,)| (g, false)))?;
                let generation = parse_generation_py(py_tok, &gen_name)?;
                let bridge = PySdkBridge::shared(generation);
                Ok(bridge.core_state_summary_with_hlo(include_hlo))
            },
        )
    });
    res = res.and_then(|()| {
        register_closure(
            py,
            module,
            "capture_xspace",
            c"capture_xspace",
            c"Captures a serialized `tensorflow.profiler.XSpace` protobuf byte payload.",
            |args, _kwargs| {
                let py_tok = args.py();
                let (gen_name,): (String,) = args.extract()?;
                let generation = parse_generation_py(py_tok, &gen_name)?;
                let bridge = PySdkBridge::shared(generation);
                let _ = bridge.profiler.start();
                let _ = bridge.profiler.stop();
                let xspace = bridge.profiler.collect_xspace_bytes();
                Ok(PyBytes::new(py_tok, &xspace).unbind())
            },
        )
    });
    res = res.and_then(|()| {
        register_closure(
            py,
            module,
            "lookup_csr_offset",
            c"lookup_csr_offset",
            c"Resolves a hierarchical hardware CSR path on `generation` to its BAR0 byte offset.",
            |args, _kwargs| {
                let py_tok = args.py();
                let (csr_path, gen_name): (String, String) = args.extract()?;
                let generation = parse_generation_py(py_tok, &gen_name)?;
                let bridge = PySdkBridge::shared(generation);
                bridge.lookup_csr_by_path(&csr_path).ok_or_else(|| {
                    lib_tpu_error_to_py_err(
                        py_tok,
                        &LibTpuError::InvalidArgument {
                            parameter: "csr_path",
                            reason: format!("unknown CSR path '{csr_path}'"),
                        },
                    )
                })
            },
        )
    });
    res = res.and_then(|()| register_runtime_sync_binder(py, module));
    res
}

/// PEP 489 `Py_mod_exec` callback initializing `libtpu._C`.
///
/// # Arguments
///
/// * `module` - Borrowed `PyObject*` module instance supplied by `CPython`.
///
/// # Safety
///
/// Must be called with the `CPython` GIL held; `module` must be null or a valid borrowed `PyObject*`.
pub(crate) unsafe extern "C" fn c_module_exec(module: *mut ffi::PyObject) -> c_int {
    if module.is_null() {
        return -1;
    }
    Python::attach(|py| {
        // SAFETY: `module` was checked non-null and is a valid borrowed `PyObject*`.
        let bound = unsafe { Bound::from_borrowed_ptr(py, module) };
        match populate_c_module(py, &bound) {
            Ok(()) => 0,
            Err(err) => {
                err.restore(py);
                -1
            }
        }
    })
}

/// PEP 489 `Py_mod_exec` callback initializing `libtpu/sdk.so` (`PyInit_sdk`).
///
/// # Arguments
///
/// * `module` - Borrowed `PyObject*` module instance supplied by `CPython`.
///
/// # Safety
///
/// Must be called with the `CPython` GIL held; `module` must be null or a valid borrowed `PyObject*`.
pub(crate) unsafe extern "C" fn sdk_module_exec(module: *mut ffi::PyObject) -> c_int {
    if module.is_null() {
        return -1;
    }
    Python::attach(|py| {
        // SAFETY: `module` was checked non-null and is a valid borrowed `PyObject*`.
        let bound = unsafe { Bound::from_borrowed_ptr(py, module) };
        match populate_sdk_module(py, &bound) {
            Ok(()) => 0,
            Err(err) => {
                err.restore(py);
                -1
            }
        }
    })
}

/// PEP 489 / PEP 703 slot array for `libtpu._C`.
static C_MODULE_SLOTS: SyncModuleSlots = SyncModuleSlots(UnsafeCell::new([
    ffi::PyModuleDef_Slot {
        slot: ffi::Py_mod_exec,
        value: c_module_exec as *mut c_void,
    },
    #[cfg(Py_GIL_DISABLED)]
    ffi::PyModuleDef_Slot {
        slot: ffi::Py_mod_gil,
        value: ffi::Py_MOD_GIL_NOT_USED,
    },
    ffi::PyModuleDef_Slot {
        slot: 0,
        value: core::ptr::null_mut(),
    },
]));

/// PEP 489 `PyModuleDef` descriptor for `libtpu._C`.
static C_MODULE_DEF: SyncModuleDef = SyncModuleDef(UnsafeCell::new(ffi::PyModuleDef {
    m_base: ffi::PyModuleDef_HEAD_INIT,
    m_name: c"_C".as_ptr(),
    m_doc: c"Native PyO3 (abi3) Rust extension backing `libtpu._C`.".as_ptr(),
    m_size: 0,
    m_methods: core::ptr::null_mut(),
    m_slots: C_MODULE_SLOTS.0.get().cast(),
    m_traverse: None,
    m_clear: None,
    m_free: None,
}));

/// PEP 489 / PEP 703 slot array for `libtpu/sdk.so`.
static SDK_MODULE_SLOTS: SyncModuleSlots = SyncModuleSlots(UnsafeCell::new([
    ffi::PyModuleDef_Slot {
        slot: ffi::Py_mod_exec,
        value: sdk_module_exec as *mut c_void,
    },
    #[cfg(Py_GIL_DISABLED)]
    ffi::PyModuleDef_Slot {
        slot: ffi::Py_mod_gil,
        value: ffi::Py_MOD_GIL_NOT_USED,
    },
    ffi::PyModuleDef_Slot {
        slot: 0,
        value: core::ptr::null_mut(),
    },
]));

/// PEP 489 `PyModuleDef` descriptor for `libtpu/sdk.so` (`PyInit_sdk`).
static SDK_MODULE_DEF: SyncModuleDef = SyncModuleDef(UnsafeCell::new(ffi::PyModuleDef {
    m_base: ffi::PyModuleDef_HEAD_INIT,
    m_name: c"sdk".as_ptr(),
    m_doc: c"Native PyO3 (abi3) Rust extension backing `libtpu/sdk.so` (`PyInit_sdk`).".as_ptr(),
    m_size: 0,
    m_methods: core::ptr::null_mut(),
    m_slots: SDK_MODULE_SLOTS.0.get().cast(),
    m_traverse: None,
    m_clear: None,
    m_free: None,
}));

/// PEP 489 `PyModuleDef` descriptor for `libtpu_py` (`PyInit_libtpu_py`).
static LIBTPU_PY_MODULE_DEF: SyncModuleDef = SyncModuleDef(UnsafeCell::new(ffi::PyModuleDef {
    m_base: ffi::PyModuleDef_HEAD_INIT,
    m_name: c"libtpu_py".as_ptr(),
    m_doc: c"Native PyO3 (abi3) Rust extension entrypoint `libtpu_py`.".as_ptr(),
    m_size: 0,
    m_methods: core::ptr::null_mut(),
    m_slots: C_MODULE_SLOTS.0.get().cast(),
    m_traverse: None,
    m_clear: None,
    m_free: None,
}));

/// Native `PyO3` (`abi3`) PEP 489 module initializer for `libtpu._C` (`PyInit__C`).
///
/// # Safety
///
/// Must be called by an initialized `CPython` interpreter holding the GIL.
#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyInit__C() -> *mut ffi::PyObject {
    // SAFETY: `C_MODULE_DEF` is a valid static `PyModuleDef` initialized under the `CPython` GIL.
    unsafe { ffi::PyModuleDef_Init(C_MODULE_DEF.0.get()) }
}

/// Native `PyO3` (`abi3`) PEP 489 module initializer for `libtpu/sdk.so` (`PyInit_sdk`).
///
/// # Safety
///
/// Must be called by an initialized `CPython` interpreter holding the GIL.
#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyInit_sdk() -> *mut ffi::PyObject {
    // SAFETY: `SDK_MODULE_DEF` is a valid static `PyModuleDef` initialized under the `CPython` GIL.
    unsafe { ffi::PyModuleDef_Init(SDK_MODULE_DEF.0.get()) }
}

/// Native `PyO3` (`abi3`) PEP 489 module initializer for `libtpu_py` (`PyInit_libtpu_py`).
///
/// # Safety
///
/// Must be called by an initialized `CPython` interpreter holding the GIL.
#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyInit_libtpu_py() -> *mut ffi::PyObject {
    // SAFETY: `LIBTPU_PY_MODULE_DEF` is a valid static `PyModuleDef` initialized under the `CPython` GIL.
    unsafe { ffi::PyModuleDef_Init(LIBTPU_PY_MODULE_DEF.0.get()) }
}

//! Zero-copy and strided `DLPack` v1.0 C-ABI (`dltensor` / `used_dltensor`) capsule bridge for `PyO3`.

use core::ffi::{c_int, c_void};
use libtpu_core::error::{LibTpuError, LibTpuResult};
use pyo3::ffi;
use pyo3::prelude::*;

/// `DLPack` v1.0 `DLDevice` C-ABI descriptor (`8` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DlDevice {
    /// `DLPack` device type discriminant (`1` = `kDLCPU`, `14` = `kDLOneAPI`).
    pub device_type: i32,
    /// Logical device ordinal (`0..N`).
    pub device_id: i32,
}

/// `DLPack` v1.0 `DLDataType` C-ABI descriptor (`4` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DlDataType {
    /// `DLPack` type code (`0` int, `1` uint, `2` float, `4` bfloat, `5` complex, `6` bool).
    pub code: u8,
    /// Scalar bit width (`1..=128`).
    pub bits: u8,
    /// SIMD vector lanes (`1` for standard tensors).
    pub lanes: u16,
}

/// `DLPack` v1.0 `DLTensor` C-ABI view (`48` bytes on 64-bit targets).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DlTensor {
    /// Base data pointer.
    pub data: *mut c_void,
    /// Device descriptor.
    pub device: DlDevice,
    /// Number of dimensions (`>= 0`).
    pub ndim: i32,
    /// Element data type descriptor.
    pub dtype: DlDataType,
    /// Pointer to `ndim` shape extents (`NULL` allowed when `ndim == 0`).
    pub shape: *mut i64,
    /// Optional pointer to `ndim` element strides (`NULL` indicates compact row-major).
    pub strides: *mut i64,
    /// Byte offset added to `data`.
    pub byte_offset: u64,
}

/// Deleter callback signature for [`DlManagedTensor`].
pub type DlManagedTensorDeleter = unsafe extern "C" fn(*mut DlManagedTensor);

/// `DLPack` v1.0 `DLManagedTensor` C-ABI ownership wrapper (`64` bytes on 64-bit targets).
#[repr(C)]
#[derive(Debug)]
pub struct DlManagedTensor {
    /// Embedded [`DlTensor`] descriptor.
    pub dl_tensor: DlTensor,
    /// Producer-private heap context pointer.
    pub manager_ctx: *mut c_void,
    /// Producer deleter callback invoked exactly once by the consumer (or capsule GC).
    pub deleter: Option<DlManagedTensorDeleter>,
}

/// Heap-pinned backing storage owned by an exported [`DlManagedTensor`].
#[derive(Debug)]
struct OwnedDlPackContext {
    /// Pinned tensor payload bytes (at least 8 bytes allocated so `as_mut_ptr()` is non-null).
    data: Vec<u8>,
    /// Pinned dimension extents.
    shape: Vec<i64>,
    /// Pinned compact row-major element strides.
    strides: Vec<i64>,
}

/// Decoded metadata and contiguous host bytes extracted from a consumed `"dltensor"` `PyCapsule`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConsumedDlPackTensor {
    /// Materialized contiguous little-endian tensor bytes.
    pub raw_bytes: Vec<u8>,
    /// Logical dimension extents.
    pub shape: Vec<i64>,
    /// `DLPack` `DLDataTypeCode` discriminant.
    pub dtype_code: u8,
    /// Scalar bit width.
    pub bits: u8,
    /// Vector lane count.
    pub lanes: u16,
    /// Source `DLDevice::device_type`.
    pub device_type: i32,
    /// Source `DLDevice::device_id`.
    pub device_id: i32,
}

/// Computes compact row-major element strides for `shape`.
///
/// # Arguments
///
/// * `shape` - Slice of dimension extents.
#[must_use]
pub fn compute_compact_strides(shape: &[i64]) -> Vec<i64> {
    let mut strides = vec![1_i64; shape.len()];
    let mut running = 1_i64;
    for (stride_slot, &dim) in strides.iter_mut().zip(shape.iter()).rev() {
        *stride_slot = running;
        running = running.saturating_mul(dim.max(1));
    }
    strides
}

/// C-ABI deleter freeing an `OwnedDlPackContext` and its enclosing [`DlManagedTensor`].
///
/// # Arguments
///
/// * `managed_ptr` - Heap pointer allocated in [`export_dlpack_capsule`].
///
/// # Safety
///
/// `managed_ptr` must either be null or a valid heap pointer produced by [`export_dlpack_capsule`]
/// that has not yet been freed.
pub unsafe extern "C" fn dlpack_managed_tensor_deleter(managed_ptr: *mut DlManagedTensor) {
    if !managed_ptr.is_null() {
        // SAFETY: `managed_ptr` was allocated via `Box::into_raw` in `export_dlpack_capsule`.
        let managed = unsafe { Box::from_raw(managed_ptr) };
        if !managed.manager_ctx.is_null() {
            // SAFETY: `manager_ctx` was allocated via `Box::into_raw(Box::new(OwnedDlPackContext))`.
            let _ctx = unsafe { Box::from_raw(managed.manager_ctx.cast::<OwnedDlPackContext>()) };
        }
    }
}

/// `PyCapsule_Destructor` freeing unconsumed `"dltensor"` capsules while ignoring `"used_dltensor"` capsules.
///
/// # Arguments
///
/// * `capsule` - Borrowed `PyObject*` referencing the `PyCapsule` being finalized by `CPython`.
///
/// # Safety
///
/// Must only be invoked by `CPython`'s `PyCapsule` finalizer with the GIL held.
pub unsafe extern "C" fn unconsumed_dlpack_capsule_destructor(capsule: *mut ffi::PyObject) {
    // SAFETY: `capsule` is a live `PyCapsule` passed by `CPython` during finalization.
    let is_unconsumed = unsafe { ffi::PyCapsule_IsValid(capsule, c"dltensor".as_ptr()) != 0 };
    if is_unconsumed {
        // SAFETY: `PyCapsule_IsValid` succeeded for `"dltensor"`.
        let ptr = unsafe {
            ffi::PyCapsule_GetPointer(capsule, c"dltensor".as_ptr()).cast::<DlManagedTensor>()
        };
        // SAFETY: `ptr` is an unconsumed heap `DlManagedTensor` created by `export_dlpack_capsule`.
        unsafe {
            dlpack_managed_tensor_deleter(ptr);
        }
    } else {
        // Clear any pending error left if a third-party consumer mutated capsule state unexpectedly.
        // SAFETY: Attached to `CPython` interpreter inside capsule destructor.
        unsafe {
            ffi::PyErr_Clear();
        }
    }
}

/// Exports `payload_bytes` and `shape` as a `DLPack` v1.0 `"dltensor"` `PyCapsule`.
///
/// # Arguments
///
/// * `py` - Active `CPython` GIL token.
/// * `payload_bytes` - Contiguous little-endian tensor bytes.
/// * `shape` - Logical dimension extents.
/// * `dtype` - [`DlDataType`] descriptor.
/// * `device` - [`DlDevice`] descriptor.
///
/// # Errors
///
/// Returns [`PyErr`] if `CPython` fails to allocate the `PyCapsule`.
pub fn export_dlpack_capsule<'py>(
    py: Python<'py>,
    payload_bytes: &[u8],
    shape: &[i64],
    dtype: DlDataType,
    device: DlDevice,
) -> PyResult<Bound<'py, PyAny>> {
    let mut pinned = Vec::with_capacity(payload_bytes.len().max(8));
    pinned.extend_from_slice(payload_bytes);
    if pinned.is_empty() {
        pinned.resize(8, 0);
    }
    let shape_vec = shape.to_vec();
    let strides_vec = compute_compact_strides(&shape_vec);
    let mut ctx = Box::new(OwnedDlPackContext {
        data: pinned,
        shape: shape_vec,
        strides: strides_vec,
    });
    let data_ptr = ctx.data.as_mut_ptr().cast::<c_void>();
    let ndim = i32::try_from(ctx.shape.len()).unwrap_or(0);
    let (shape_ptr, strides_ptr) = if ctx.shape.is_empty() {
        (core::ptr::null_mut(), core::ptr::null_mut())
    } else {
        (ctx.shape.as_mut_ptr(), ctx.strides.as_mut_ptr())
    };
    let ctx_ptr = Box::into_raw(ctx).cast::<c_void>();
    let managed = Box::new(DlManagedTensor {
        dl_tensor: DlTensor {
            data: data_ptr,
            device,
            ndim,
            dtype,
            shape: shape_ptr,
            strides: strides_ptr,
            byte_offset: 0,
        },
        manager_ctx: ctx_ptr,
        deleter: Some(dlpack_managed_tensor_deleter),
    });
    let managed_ptr = Box::into_raw(managed);
    // SAFETY: `managed_ptr` is non-null and `"dltensor"` is a valid NUL-terminated C string.
    let raw_cap = unsafe {
        ffi::PyCapsule_New(
            managed_ptr.cast::<c_void>(),
            c"dltensor".as_ptr(),
            Some(unconsumed_dlpack_capsule_destructor),
        )
    };
    // SAFETY: `PyCapsule_New` returns a newly owned `PyObject*` or null with a Python exception set.
    unsafe { Bound::from_owned_ptr_or_err(py, raw_cap) }
}

/// Copies a potentially strided tensor buffer described by [`DlTensor`] into compact row-major bytes.
///
/// # Arguments
///
/// * `base_ptr` - Starting byte pointer (`dl_tensor.data + dl_tensor.byte_offset`).
/// * `shape` - Non-negative dimension extents.
/// * `strides_opt` - Optional per-dimension element strides.
/// * `elem_bytes` - Byte width of a single element (`>= 1`).
///
/// # Safety
///
/// When total element count is non-zero and `base_ptr` is non-null, all element offsets addressed
/// by `shape` and `strides_opt` must be valid for reads of `elem_bytes` bytes.
unsafe fn copy_dl_tensor_bytes(
    base_ptr: *const u8,
    shape: &[i64],
    strides_opt: Option<&[i64]>,
    elem_bytes: usize,
) -> Vec<u8> {
    let num_elements = shape.iter().fold(1_usize, |acc, &d| {
        acc.saturating_mul(usize::try_from(d.max(0)).unwrap_or(0))
    });
    let total_bytes = num_elements.saturating_mul(elem_bytes);
    if total_bytes == 0 || base_ptr.is_null() {
        return vec![0_u8; total_bytes];
    }
    let compact_strides = compute_compact_strides(shape);
    let is_contiguous = strides_opt.is_none_or(|st| st == compact_strides.as_slice());
    if is_contiguous {
        // SAFETY: Producer guarantees `total_bytes` contiguous readable bytes at `base_ptr`.
        return unsafe { core::slice::from_raw_parts(base_ptr, total_bytes).to_vec() };
    }
    let strides = strides_opt.unwrap_or(&compact_strides);
    let mut out = Vec::with_capacity(total_bytes);
    for linear in 0..num_elements {
        let mut rem = linear;
        let mut elem_offset = 0_isize;
        for (&dim_i64, &st_i64) in shape.iter().zip(strides.iter()).rev() {
            let dim_u = usize::try_from(dim_i64.max(1)).unwrap_or(1);
            let coord = isize::try_from(rem % dim_u).unwrap_or(0);
            rem /= dim_u;
            let st = isize::try_from(st_i64).unwrap_or(0);
            elem_offset = elem_offset.saturating_add(coord.saturating_mul(st));
        }
        let byte_off = elem_offset.saturating_mul(isize::try_from(elem_bytes).unwrap_or(1));
        // SAFETY: `byte_off` addresses a valid element within the producer's strided tensor allocation.
        let src = unsafe { base_ptr.offset(byte_off) };
        // SAFETY: Each element occupies `elem_bytes` readable bytes.
        let slice = unsafe { core::slice::from_raw_parts(src, elem_bytes) };
        out.extend_from_slice(slice);
    }
    out
}

/// Validates and consumes a `DLPack` v1.0 `"dltensor"` `PyCapsule`, renaming it to `"used_dltensor"`
/// and invoking its `DLManagedTensor::deleter` after copying its payload.
///
/// # Arguments
///
/// * `capsule` - Borrowed Python object expected to be an unconsumed `"dltensor"` `PyCapsule`.
///
/// # Errors
///
/// Returns [`LibTpuError::NullPointer`] if `capsule` is not a valid unconsumed `"dltensor"` `PyCapsule`.
pub fn consume_dlpack_capsule(capsule: &Bound<'_, PyAny>) -> LibTpuResult<ConsumedDlPackTensor> {
    let cap_ptr = capsule.as_ptr();
    // SAFETY: `cap_ptr` is a live borrowed `PyObject*` under the active GIL.
    let valid: c_int = unsafe { ffi::PyCapsule_IsValid(cap_ptr, c"dltensor".as_ptr()) };
    if valid == 0 {
        return Err(LibTpuError::NullPointer {
            parameter: "dltensor_capsule",
        });
    }
    // SAFETY: `PyCapsule_IsValid` confirmed `cap_ptr` is a `"dltensor"` capsule with a non-null pointer.
    let managed_ptr = unsafe {
        ffi::PyCapsule_GetPointer(cap_ptr, c"dltensor".as_ptr()).cast::<DlManagedTensor>()
    };
    // SAFETY: `managed_ptr` points to a live `DlManagedTensor` owned by the unconsumed capsule.
    let dlt = unsafe { (*managed_ptr).dl_tensor };
    let ndim = usize::try_from(dlt.ndim.max(0)).unwrap_or(0);
    let shape: Vec<i64> = if ndim == 0 || dlt.shape.is_null() {
        Vec::new()
    } else {
        // SAFETY: `dlt.shape` points to `ndim` `i64` dimension extents.
        unsafe { core::slice::from_raw_parts(dlt.shape, ndim).to_vec() }
    };
    let strides_vec: Option<Vec<i64>> = if ndim == 0 || dlt.strides.is_null() {
        None
    } else {
        // SAFETY: `dlt.strides` points to `ndim` `i64` element strides when non-null.
        Some(unsafe { core::slice::from_raw_parts(dlt.strides, ndim).to_vec() })
    };
    let total_bits = usize::from(dlt.dtype.bits.max(8)) * usize::from(dlt.dtype.lanes.max(1));
    let elem_bytes = total_bits.div_ceil(8).max(1);
    let base_ptr = if dlt.data.is_null() {
        core::ptr::null()
    } else {
        let off = usize::try_from(dlt.byte_offset).unwrap_or(0);
        // SAFETY: Advancing `dlt.data` by `dlt.byte_offset` bytes per `DLPack` specification.
        unsafe { dlt.data.cast::<u8>().add(off).cast_const() }
    };
    // SAFETY: `base_ptr`, `shape`, and `strides_vec` come from the validated `DlTensor`.
    let raw_bytes =
        unsafe { copy_dl_tensor_bytes(base_ptr, &shape, strides_vec.as_deref(), elem_bytes) };

    // Rename capsule to `"used_dltensor"` BEFORE invoking deleter per `DLPack` v1.0 protocol.
    // SAFETY: `cap_ptr` is a valid `PyCapsule` and `"used_dltensor"` has `'static` lifetime.
    unsafe {
        let _ = ffi::PyCapsule_SetName(cap_ptr, c"used_dltensor".as_ptr());
    }
    // SAFETY: `managed_ptr` is still valid prior to calling its producer deleter.
    if let Some(deleter_fn) = unsafe { (*managed_ptr).deleter } {
        // SAFETY: Consumer must invoke `deleter(managed_ptr)` once after reading the tensor.
        unsafe {
            deleter_fn(managed_ptr);
        }
    }

    Ok(ConsumedDlPackTensor {
        raw_bytes,
        shape,
        dtype_code: dlt.dtype.code,
        bits: dlt.dtype.bits,
        lanes: dlt.dtype.lanes,
        device_type: dlt.device.device_type,
        device_id: dlt.device.device_id,
    })
}

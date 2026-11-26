//! PJRT Collectives and Cross-Host Transfers extension entrypoints (`0.0.53.1`).
//!
//! Implements `PJRT_Collectives_Extension` and `PJRT_CrossHostTransfers_Extension`.

use super::args::*;
use super::{dispatch_typed_extension_args, dispatch_typed_fallible_extension_args};
use crate::pjrt_helpers::{
    alloc_shaped_pjrt_buffer, borrow_live_handle, box_ready_event, box_registered_handle,
    take_live_handle, to_c_string,
};
use crate::pjrt_types::{PJRT_Buffer, PJRT_Collectives_Communicator, PJRT_Error, PJRT_Event};
use libtpu_core::distributed::{CollectiveCommunicator, global_cross_host_transfer_registry};
use libtpu_core::error::LibTpuError;
use libtpu_core::pjrt::{PjrtCoreBuffer, PjrtCoreEvent};
use libtpu_core::types::{CollectiveReductionOp, CommunicatorId, DeviceId};
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::ptr;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// Monotonic sequence counter assigning unique IDs to [`PJRT_Collectives_Communicator`] groups.
static COMMUNICATOR_ID_SEQ: AtomicU64 = AtomicU64::new(0);

/// Monotonic counter of executed collective operations across communicators.
pub(super) static COLLECTIVE_OPS_EXECUTED: AtomicU64 = AtomicU64::new(0);

/// Active replica-group slot dispenser pairing sequential `PJRT_Collectives_CreateCommunicator`
/// calls over identical `device_ids` into a shared multi-rank [`CollectiveCommunicator`] group.
static PENDING_COMM_GROUPS: OnceLock<Mutex<BTreeMap<Vec<DeviceId>, Vec<CollectiveCommunicator>>>> =
    OnceLock::new();

/// Returns a reference to the process-wide [`PENDING_COMM_GROUPS`] map.
fn pending_comm_groups() -> &'static Mutex<BTreeMap<Vec<DeviceId>, Vec<CollectiveCommunicator>>> {
    PENDING_COMM_GROUPS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// Strongly-typed handler for [`PJRT_Collectives_Destroy`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Collectives_Destroy_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid mutable [`PJRT_Collectives_Destroy_Args`] reference.
unsafe fn handle_collectives_destroy(args: &mut PJRT_Collectives_Destroy_Args) {
    args.collectives = ptr::null_mut();
}

/// Destroys a `PJRT_Collectives` instance (`PJRT_Collectives_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Collectives_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Collectives_Destroy_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Collectives_Destroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Collectives_Destroy_Args>(
            raw_args,
            handle_collectives_destroy,
        )
    }
}

/// Allocates the next rank's [`CollectiveCommunicator`] for `devices`, creating a fresh shared
/// group via [`CollectiveCommunicator::create_group`] whenever no unassigned ranks remain.
///
/// # Arguments
///
/// * `devices` - Ordered participating logical [`DeviceId`]s (`>= 1`).
///
/// # Errors
///
/// Returns [`LibTpuError`] if `devices` is empty.
pub(super) fn allocate_group_communicator(
    devices: &[DeviceId],
) -> Result<CollectiveCommunicator, LibTpuError> {
    if devices.len() > 1 {
        let mut seen = std::collections::BTreeSet::new();
        for &dev in devices {
            if !seen.insert(dev.0) {
                return Err(LibTpuError::InvalidArgument {
                    parameter: "device_ids",
                    reason: format!("duplicate device ID {} in communicator group", dev.0),
                });
            }
        }
        let mut guard = pending_comm_groups().lock();
        if let Some(mut remaining) = guard.remove(devices) {
            let next = remaining.remove(0);
            if !remaining.is_empty() {
                guard.insert(devices.to_vec(), remaining);
            }
            return Ok(next);
        }
    }
    let comm_id = CommunicatorId(COMMUNICATOR_ID_SEQ.fetch_add(1, Ordering::Relaxed) + 1);
    let mut group = CollectiveCommunicator::create_group(comm_id, devices)?;
    let first = group.remove(0);
    if !group.is_empty() {
        pending_comm_groups().lock().insert(devices.to_vec(), group);
    }
    Ok(first)
}

/// Strongly-typed fallible handler for [`PJRT_Collectives_CreateCommunicator`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Collectives_CreateCommunicator_Args`] structure.
///
/// # Errors
///
/// Returns [`LibTpuError::InvalidArgument`] if `num_devices > 0 && device_ids.is_null()` or any
/// device ID is negative.
///
/// # Safety
///
/// `args.device_ids` must be null or point to `args.num_devices` valid `i32` device IDs.
unsafe fn handle_collectives_create_communicator(
    args: &mut PJRT_Collectives_CreateCommunicator_Args,
) -> Result<(), LibTpuError> {
    if args.num_devices > 0 && args.device_ids.is_null() {
        return Err(LibTpuError::InvalidArgument {
            parameter: "device_ids",
            reason: String::from("PJRT_Collectives_CreateCommunicator device_ids is null"),
        });
    }
    let devices: Vec<DeviceId> = if args.num_devices == 0 {
        vec![DeviceId(0)]
    } else {
        // SAFETY: `device_ids` is non-null with `num_devices` elements.
        let slice = unsafe { std::slice::from_raw_parts(args.device_ids, args.num_devices) };
        let mut parsed = Vec::with_capacity(slice.len());
        for &raw_id in slice {
            let id_u32 = u32::try_from(raw_id).map_err(|_| LibTpuError::InvalidArgument {
                parameter: "device_ids",
                reason: format!("Communicator device ID {raw_id} must be non-negative"),
            })?;
            parsed.push(DeviceId(id_u32));
        }
        parsed
    };
    let inner = allocate_group_communicator(&devices)?;
    let communicator_id = inner.comm_id;
    let device_ids = devices
        .iter()
        .map(|d| i32::try_from(d.0).unwrap_or(0))
        .collect();
    let descriptor = to_c_string(&inner.to_debug_string());
    args.communicator = box_registered_handle(PJRT_Collectives_Communicator {
        inner,
        communicator_id,
        device_ids,
        descriptor,
    });
    Ok(())
}

/// Creates a `PJRT_Collectives_Communicator` (`PJRT_Collectives_CreateCommunicator`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Collectives_CreateCommunicator_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Collectives_CreateCommunicator_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Collectives_CreateCommunicator(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PJRT_Collectives_CreateCommunicator_Args>(
            raw_args,
            handle_collectives_create_communicator,
        )
    }
}

/// Strongly-typed fallible handler for [`PJRT_Communicator_Destroy`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Communicator_Destroy_Args`] structure.
///
/// # Errors
///
/// Returns [`LibTpuError::InvalidArgument`] if `communicator` is null or unrecognized.
///
/// # Safety
///
/// `args` must be a valid mutable [`PJRT_Communicator_Destroy_Args`] reference.
unsafe fn handle_communicator_destroy(
    args: &mut PJRT_Communicator_Destroy_Args,
) -> Result<(), LibTpuError> {
    let _comm =
        take_live_handle::<PJRT_Collectives_Communicator>(args.communicator).ok_or_else(|| {
            LibTpuError::InvalidArgument {
                parameter: "communicator",
                reason: String::from("null or unregistered PJRT_Collectives_Communicator handle"),
            }
        })?;
    args.communicator = ptr::null_mut();
    Ok(())
}

/// Destroys a `PJRT_Collectives_Communicator` (`PJRT_Communicator_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Communicator_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Communicator_Destroy_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Communicator_Destroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PJRT_Communicator_Destroy_Args>(
            raw_args,
            handle_communicator_destroy,
        )
    }
}

/// Validates and borrows `(communicator, send_buffer, recv_buffer)` for a collective operation,
/// staging the local rank's `send_bytes` on multi-rank communicators.
///
/// # Arguments
///
/// * `comm_ptr` - Live `PJRT_Collectives_Communicator*` handle.
/// * `send_ptr` - Source `PJRT_Buffer*` handle.
/// * `recv_ptr` - Destination `PJRT_Buffer*` handle.
///
/// # Errors
///
/// Returns [`LibTpuError`] if any handle is null/invalid or source HBM read fails.
///
/// # Safety
///
/// `send_ptr` and `recv_ptr` must be null or valid `PJRT_Buffer*` handles.
unsafe fn prepare_collective_io<'a>(
    comm_ptr: *mut PJRT_Collectives_Communicator,
    send_ptr: *mut PJRT_Buffer,
    recv_ptr: *mut PJRT_Buffer,
) -> Result<
    (
        &'a PJRT_Collectives_Communicator,
        &'a PJRT_Buffer,
        &'a PJRT_Buffer,
        Vec<u8>,
    ),
    LibTpuError,
> {
    let comm = borrow_live_handle::<PJRT_Collectives_Communicator>(comm_ptr.cast_const())
        .ok_or_else(|| LibTpuError::InvalidArgument {
            parameter: "communicator",
            reason: String::from("null or unregistered PJRT_Collectives_Communicator handle"),
        })?;
    // SAFETY: `send_ptr` is null or a valid `PJRT_Buffer*`.
    let send_buf = unsafe { send_ptr.as_ref() }.ok_or_else(|| LibTpuError::InvalidArgument {
        parameter: "send_buffer",
        reason: String::from("PJRT_Communicator send_buffer is null"),
    })?;
    // SAFETY: `recv_ptr` is null or a valid `PJRT_Buffer*`.
    let recv_buf = unsafe { recv_ptr.as_ref() }.ok_or_else(|| LibTpuError::InvalidArgument {
        parameter: "recv_buffer",
        reason: String::from("PJRT_Communicator recv_buffer is null"),
    })?;
    let send_bytes = send_buf.inner.to_host_bytes()?;
    if comm.inner.world_size > 1 {
        let _ = comm
            .inner
            .stage_rank_buffer(comm.inner.rank, send_bytes.clone());
    }
    Ok((comm, send_buf, recv_buf, send_bytes))
}

/// Writes `out_bytes` into `recv_buf` HBM and populates `event_out` with a completed [`PJRT_Event`].
///
/// # Arguments
///
/// * `recv_buf` - Destination [`PJRT_Buffer`].
/// * `out_bytes` - Computed collective result bytes.
/// * `event_out` - Output slot receiving the allocated ready `PJRT_Event*`.
///
/// # Errors
///
/// Returns [`LibTpuError`] if destination HBM write fails.
fn commit_collective_output(
    recv_buf: &PJRT_Buffer,
    out_bytes: &[u8],
    event_out: &mut *mut PJRT_Event,
) -> Result<(), LibTpuError> {
    recv_buf.inner.hal.write_device_memory(
        recv_buf.inner.device_id,
        recv_buf.inner.address,
        out_bytes,
    )?;
    COLLECTIVE_OPS_EXECUTED.fetch_add(1, Ordering::Relaxed);
    *event_out = box_ready_event();
    Ok(())
}

/// Strongly-typed fallible handler for [`PJRT_Communicator_AllReduce`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Communicator_AllReduce_Args`] structure.
///
/// # Errors
///
/// Returns [`LibTpuError`] on null handles, invalid `reduction_op`, or mismatched buffer sizes.
///
/// # Safety
///
/// `args` must contain null or valid `PJRT_Buffer*` handles.
unsafe fn handle_communicator_all_reduce(
    args: &mut PJRT_Communicator_AllReduce_Args,
) -> Result<(), LibTpuError> {
    let op = CollectiveReductionOp::from_raw(args.reduction_op)?;
    // SAFETY: Caller upholds buffer pointer validity.
    let (comm, send_buf, recv_buf, send_bytes) =
        unsafe { prepare_collective_io(args.communicator, args.send_buffer, args.recv_buffer) }?;
    let peer_bytes = if args.send_buffer != args.recv_buffer {
        Some(recv_buf.inner.to_host_bytes()?)
    } else {
        None
    };
    let reduced = comm.inner.all_reduce(
        &send_bytes,
        peer_bytes.as_deref(),
        send_buf.inner.shape.element_type,
        op,
    )?;
    commit_collective_output(recv_buf, &reduced, &mut args.event)
}

/// Strongly-typed fallible handler for [`PJRT_Communicator_ReduceScatter`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Communicator_AllReduce_Args`] structure.
///
/// # Errors
///
/// Returns [`LibTpuError`] on null handles, invalid `reduction_op`, or shard size mismatch.
///
/// # Safety
///
/// `args` must contain null or valid `PJRT_Buffer*` handles.
unsafe fn handle_communicator_reduce_scatter(
    args: &mut PJRT_Communicator_AllReduce_Args,
) -> Result<(), LibTpuError> {
    let op = CollectiveReductionOp::from_raw(args.reduction_op)?;
    // SAFETY: Caller upholds buffer pointer validity.
    let (comm, send_buf, recv_buf, send_bytes) =
        unsafe { prepare_collective_io(args.communicator, args.send_buffer, args.recv_buffer) }?;
    let recv_len = recv_buf.inner.shape.unpadded_byte_size().0;
    let shard =
        comm.inner
            .reduce_scatter(&send_bytes, recv_len, send_buf.inner.shape.element_type, op)?;
    commit_collective_output(recv_buf, &shard, &mut args.event)
}

/// Executes an `AllReduce` collective (`PJRT_Communicator_AllReduce`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Communicator_AllReduce_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Communicator_AllReduce_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Communicator_AllReduce(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PJRT_Communicator_AllReduce_Args>(
            raw_args,
            handle_communicator_all_reduce,
        )
    }
}

/// Executes a `ReduceScatter` collective (`PJRT_Communicator_ReduceScatter`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Communicator_AllReduce_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Communicator_AllReduce_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Communicator_ReduceScatter(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PJRT_Communicator_AllReduce_Args>(
            raw_args,
            handle_communicator_reduce_scatter,
        )
    }
}

/// Strongly-typed fallible handler for [`PJRT_Communicator_AllGather`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Communicator_AllGather_Args`] structure.
///
/// # Errors
///
/// Returns [`LibTpuError`] on null handles or mismatched `recv_buffer` size (`send_len * world_size`).
///
/// # Safety
///
/// `args` must contain null or valid `PJRT_Buffer*` handles.
unsafe fn handle_communicator_all_gather(
    args: &mut PJRT_Communicator_AllGather_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: Caller upholds buffer pointer validity.
    let (comm, _send_buf, recv_buf, send_bytes) =
        unsafe { prepare_collective_io(args.communicator, args.send_buffer, args.recv_buffer) }?;
    let recv_len = recv_buf.inner.shape.unpadded_byte_size().0;
    let gathered = comm.inner.all_gather(&send_bytes, recv_len)?;
    commit_collective_output(recv_buf, &gathered, &mut args.event)
}

/// Executes an `AllGather` collective (`PJRT_Communicator_AllGather`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Communicator_AllGather_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Communicator_AllGather_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Communicator_AllGather(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PJRT_Communicator_AllGather_Args>(
            raw_args,
            handle_communicator_all_gather,
        )
    }
}

/// Strongly-typed fallible handler for [`PJRT_Communicator_CollectivePermute`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Communicator_CollectivePermute_Args`] structure.
///
/// # Errors
///
/// Returns [`LibTpuError`] on null handles, malformed `(src_rank, dst_rank)` pairs, or out-of-range ranks.
///
/// # Safety
///
/// `args` must contain null or valid `PJRT_Buffer*` and `target_ranks` pointers.
unsafe fn handle_communicator_collective_permute(
    args: &mut PJRT_Communicator_CollectivePermute_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: Caller upholds buffer pointer validity.
    let (comm, _send_buf, recv_buf, send_bytes) =
        unsafe { prepare_collective_io(args.communicator, args.send_buffer, args.recv_buffer) }?;
    let pairs: Vec<(usize, usize)> = if args.target_ranks.is_null() {
        (0..comm.inner.world_size).map(|r| (r, r)).collect()
    } else if args.count == 0 {
        Vec::new()
    } else {
        if args.count % 2 != 0 {
            return Err(LibTpuError::InvalidArgument {
                parameter: "target_ranks",
                reason: format!(
                    "CollectivePermute target_ranks count ({}) must be an even number of (src, dst) entries",
                    args.count
                ),
            });
        }
        // SAFETY: `target_ranks` is non-null with `args.count` `i32` entries.
        let raw_slice = unsafe { std::slice::from_raw_parts(args.target_ranks, args.count) };
        let mut parsed = Vec::with_capacity(args.count / 2);
        for (raw_src, raw_dst) in raw_slice
            .iter()
            .copied()
            .step_by(2)
            .zip(raw_slice.iter().copied().skip(1).step_by(2))
        {
            let src = usize::try_from(raw_src).map_err(|_| LibTpuError::InvalidArgument {
                parameter: "target_ranks",
                reason: format!("Negative source rank {raw_src} in CollectivePermute"),
            })?;
            let dst = usize::try_from(raw_dst).map_err(|_| LibTpuError::InvalidArgument {
                parameter: "target_ranks",
                reason: format!("Negative destination rank {raw_dst} in CollectivePermute"),
            })?;
            parsed.push((src, dst));
        }
        parsed
    };
    let recv_len = recv_buf.inner.shape.unpadded_byte_size().0;
    let permuted = comm
        .inner
        .collective_permute(&send_bytes, recv_len, &pairs)?;
    commit_collective_output(recv_buf, &permuted, &mut args.event)
}

/// Strongly-typed fallible handler for [`PJRT_Communicator_AllToAll`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Communicator_CollectivePermute_Args`] structure.
///
/// # Errors
///
/// Returns [`LibTpuError`] on null handles or buffer sizes not divisible by `world_size`.
///
/// # Safety
///
/// `args` must contain null or valid `PJRT_Buffer*` handles.
unsafe fn handle_communicator_all_to_all(
    args: &mut PJRT_Communicator_CollectivePermute_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: Caller upholds buffer pointer validity.
    let (comm, _send_buf, recv_buf, send_bytes) =
        unsafe { prepare_collective_io(args.communicator, args.send_buffer, args.recv_buffer) }?;
    let recv_len = recv_buf.inner.shape.unpadded_byte_size().0;
    let transposed = comm.inner.all_to_all(&send_bytes, recv_len)?;
    commit_collective_output(recv_buf, &transposed, &mut args.event)
}

/// Executes a `CollectivePermute` (`PJRT_Communicator_CollectivePermute`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Communicator_CollectivePermute_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Communicator_CollectivePermute_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Communicator_CollectivePermute(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PJRT_Communicator_CollectivePermute_Args>(
            raw_args,
            handle_communicator_collective_permute,
        )
    }
}

/// Executes an `AllToAll` collective (`PJRT_Communicator_AllToAll`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Communicator_CollectivePermute_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Communicator_CollectivePermute_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Communicator_AllToAll(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PJRT_Communicator_CollectivePermute_Args>(
            raw_args,
            handle_communicator_all_to_all,
        )
    }
}

/// Strongly-typed fallible handler for [`PJRT_Communicator_ToString`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Communicator_ToString_Args`] structure.
///
/// # Errors
///
/// Returns [`LibTpuError::InvalidArgument`] if `communicator` is null or unrecognized.
///
/// # Safety
///
/// `args` must be a valid mutable [`PJRT_Communicator_ToString_Args`] reference.
unsafe fn handle_communicator_to_string(
    args: &mut PJRT_Communicator_ToString_Args,
) -> Result<(), LibTpuError> {
    let comm = borrow_live_handle::<PJRT_Collectives_Communicator>(args.communicator).ok_or_else(
        || LibTpuError::InvalidArgument {
            parameter: "communicator",
            reason: String::from("null or unregistered PJRT_Collectives_Communicator handle"),
        },
    )?;
    let bytes = comm.descriptor.as_bytes();
    args.str_out = bytes.as_ptr().cast::<std::os::raw::c_char>();
    args.str_size = bytes.len();
    Ok(())
}

/// Returns string description of a communicator (`PJRT_Communicator_ToString`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_Communicator_ToString_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_Communicator_ToString_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_Communicator_ToString(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PJRT_Communicator_ToString_Args>(
            raw_args,
            handle_communicator_to_string,
        )
    }
}

/// Extracts non-null [`PjrtCoreBuffer`] references from `args.buffers[0..args.num_buffers]`.
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_CrossHostTransfers_Args`] structure.
///
/// # Errors
///
/// Returns [`LibTpuError::InvalidArgument`] if `buffers` is null, `num_buffers == 0`, or any
/// entry in `buffers` is null.
///
/// # Safety
///
/// When `args.buffers` is non-null, it must point to `args.num_buffers` valid `*mut PJRT_Buffer` entries.
unsafe fn collect_cross_host_buffers<'a>(
    args: &PJRT_CrossHostTransfers_Args,
) -> Result<Vec<&'a PjrtCoreBuffer>, LibTpuError> {
    if args.buffers.is_null() || args.num_buffers == 0 {
        return Err(LibTpuError::InvalidArgument {
            parameter: "buffers",
            reason: String::from(
                "PJRT_CrossHostTransfers requires non-null buffers with num_buffers >= 1",
            ),
        });
    }
    // SAFETY: `args.buffers` is non-null with `args.num_buffers` entries.
    let ptrs = unsafe { std::slice::from_raw_parts(args.buffers, args.num_buffers) };
    let mut core_bufs = Vec::with_capacity(ptrs.len());
    for (idx, &buf_ptr) in ptrs.iter().enumerate() {
        // SAFETY: Each non-null `buf_ptr` points to a valid `PJRT_Buffer`.
        let buf_ref = unsafe { buf_ptr.as_ref() }.ok_or_else(|| LibTpuError::InvalidArgument {
            parameter: "buffers",
            reason: format!("PJRT_CrossHostTransfers buffers[{idx}] is null"),
        })?;
        core_bufs.push(&buf_ref.inner);
    }
    Ok(core_bufs)
}

/// Wraps the first [`PjrtCoreEvent`] of a cross-host batch into a heap [`PJRT_Event`].
///
/// Wires asynchronous completion notifications when the underlying transfer is still pending.
///
/// # Arguments
///
/// * `core_events` - Completion events returned by [`libtpu_core::distributed::CrossHostTransferRegistry`].
fn wrap_cross_host_batch_event(core_events: &[std::sync::Arc<PjrtCoreEvent>]) -> *mut PJRT_Event {
    let Some(first) = core_events.first().filter(|ev| !ev.is_ready()) else {
        return box_ready_event();
    };
    let pending_ptr = box_registered_handle(PJRT_Event {
        inner: PjrtCoreEvent::pending(),
    });
    let pending_addr = pending_ptr as usize;
    first.on_ready(Box::new(move |err_opt| {
        if let Some(c_ev) = borrow_live_handle::<PJRT_Event>(pending_addr as *const PJRT_Event) {
            c_ev.inner.set(err_opt);
        }
    }));
    pending_ptr
}

/// Strongly-typed fallible handler for [`PJRT_CrossHostTransfers_SendBuffers`] and
/// [`PJRT_CrossHostTransfers_CopyToRemoteDevice`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_CrossHostTransfers_Args`] structure.
///
/// # Errors
///
/// Returns [`LibTpuError`] if `client` is null or any buffer transfer fails.
///
/// # Safety
///
/// `args.client` and `args.buffers` must be null or valid pointers.
unsafe fn handle_cross_host_send(
    args: &mut PJRT_CrossHostTransfers_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `args.client` is null or a valid `PJRT_Client*`.
    let client = unsafe { args.client.as_ref() }.ok_or_else(|| LibTpuError::InvalidArgument {
        parameter: "client",
        reason: String::from("PJRT_CrossHostTransfers client is null"),
    })?;
    // SAFETY: Caller upholds `args.buffers` slice validity.
    let core_bufs = unsafe { collect_cross_host_buffers(args) }?;
    let key = i64::try_from(args.transfer_key).unwrap_or(i64::MAX);
    let events = global_cross_host_transfer_registry().send_buffers(key, &core_bufs, None)?;
    {
        let mut arena = client.ext_arena.lock();
        arena.cross_host_transfers = arena.cross_host_transfers.saturating_add(1);
    }
    args.out_event = wrap_cross_host_batch_event(&events);
    Ok(())
}

/// Strongly-typed fallible handler for [`PJRT_CrossHostTransfers_ReceiveBuffers`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_CrossHostTransfers_Args`] structure.
///
/// # Errors
///
/// Returns [`LibTpuError`] if `client` is null or destination buffer registration/copy fails.
///
/// # Safety
///
/// `args.client` and `args.buffers` must be null or valid pointers.
unsafe fn handle_cross_host_receive(
    args: &mut PJRT_CrossHostTransfers_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `args.client` is null or a valid `PJRT_Client*`.
    let client = unsafe { args.client.as_ref() }.ok_or_else(|| LibTpuError::InvalidArgument {
        parameter: "client",
        reason: String::from("PJRT_CrossHostTransfers client is null"),
    })?;
    // SAFETY: Caller upholds `args.buffers` slice validity.
    let core_bufs = unsafe { collect_cross_host_buffers(args) }?;
    let key = i64::try_from(args.transfer_key).unwrap_or(i64::MAX);
    let events = global_cross_host_transfer_registry().receive_buffers(key, &core_bufs, None)?;
    {
        let mut arena = client.ext_arena.lock();
        arena.cross_host_transfers = arena.cross_host_transfers.saturating_add(1);
    }
    args.out_event = wrap_cross_host_batch_event(&events);
    Ok(())
}

/// Strongly-typed fallible handler for [`PJRT_CrossHostTransfers_MakeReceiveBuffers`].
///
/// Allocates uninitialized destination `PJRT_Buffer*` slots when `buffers` contains null entries
/// and registers them in [`libtpu_core::distributed::CrossHostTransferRegistry`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_CrossHostTransfers_Args`] structure.
///
/// # Errors
///
/// Returns [`LibTpuError`] if `client` or `buffers` is null or `num_buffers == 0`.
///
/// # Safety
///
/// `args.client` must be null or a valid `PJRT_Client*`, and `args.buffers` must be null or point
/// to `args.num_buffers` writable `*mut PJRT_Buffer` slots.
unsafe fn handle_cross_host_make_receive(
    args: &mut PJRT_CrossHostTransfers_Args,
) -> Result<(), LibTpuError> {
    if args.client.is_null() {
        return Err(LibTpuError::InvalidArgument {
            parameter: "client",
            reason: String::from("PJRT_CrossHostTransfers_MakeReceiveBuffers client is null"),
        });
    }
    if args.buffers.is_null() || args.num_buffers == 0 {
        return Err(LibTpuError::InvalidArgument {
            parameter: "buffers",
            reason: String::from(
                "PJRT_CrossHostTransfers_MakeReceiveBuffers requires non-null buffers array",
            ),
        });
    }
    // SAFETY: `args.client` is non-null.
    let client_ref = unsafe { &*args.client };
    let default_dev = client_ref
        .device_ptrs
        .first()
        .copied()
        .unwrap_or(ptr::null_mut());
    let default_mem = client_ref
        .memory_ptrs
        .first()
        .copied()
        .unwrap_or(ptr::null_mut());
    let out_slots = args.buffers.cast_mut();
    // SAFETY: `out_slots` points to `args.num_buffers` slots.
    let slot_slice = unsafe { std::slice::from_raw_parts_mut(out_slots, args.num_buffers) };
    for slot in slot_slice.iter_mut() {
        if slot.is_null() {
            let dim: i64 = 4;
            *slot = alloc_shaped_pjrt_buffer(
                args.client,
                default_dev,
                default_mem,
                libtpu_core::types::ElementType::F32.as_raw(),
                ptr::addr_of!(dim),
                1,
            );
        }
    }
    // SAFETY: All slots in `args.buffers[0..args.num_buffers]` are now valid `PJRT_Buffer*`.
    unsafe { handle_cross_host_receive(args) }
}

/// Creates cross-host receive buffers (`PJRT_CrossHostTransfers_MakeReceiveBuffers`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_CrossHostTransfers_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_CrossHostTransfers_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_CrossHostTransfers_MakeReceiveBuffers(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PJRT_CrossHostTransfers_Args>(
            raw_args,
            handle_cross_host_make_receive,
        )
    }
}

/// Copies a buffer to a remote host device (`PJRT_CrossHostTransfers_CopyToRemoteDevice`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_CrossHostTransfers_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_CrossHostTransfers_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_CrossHostTransfers_CopyToRemoteDevice(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PJRT_CrossHostTransfers_Args>(
            raw_args,
            handle_cross_host_send,
        )
    }
}

/// Receives cross-host buffers (`PJRT_CrossHostTransfers_ReceiveBuffers`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_CrossHostTransfers_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_CrossHostTransfers_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_CrossHostTransfers_ReceiveBuffers(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PJRT_CrossHostTransfers_Args>(
            raw_args,
            handle_cross_host_receive,
        )
    }
}

/// Sends cross-host buffers (`PJRT_CrossHostTransfers_SendBuffers`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`PJRT_CrossHostTransfers_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`PJRT_CrossHostTransfers_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_CrossHostTransfers_SendBuffers(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Delegated to `dispatch_typed_fallible_extension_args`.
    unsafe {
        dispatch_typed_fallible_extension_args::<PJRT_CrossHostTransfers_Args>(
            raw_args,
            handle_cross_host_send,
        )
    }
}

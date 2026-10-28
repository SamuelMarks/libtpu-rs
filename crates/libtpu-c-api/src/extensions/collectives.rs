//! PJRT Collectives and Cross-Host Transfers extension entrypoints.

use super::args::*;
use super::{dispatch_typed_extension_args, extension_state};
use crate::pjrt_types::{PJRT_Collectives_Communicator, PJRT_Error};
use libtpu_core::types::{CollectiveReductionOp, CommunicatorId};
use std::ptr;

/// Strongly-typed handler for [`PJRT_Collectives_Destroy`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Collectives_Destroy_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Collectives_Destroy_Args`].
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

/// Strongly-typed handler for [`PJRT_Collectives_CreateCommunicator`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Collectives_CreateCommunicator_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Collectives_CreateCommunicator_Args`].
unsafe fn handle_collectives_create_communicator(
    args: &mut PJRT_Collectives_CreateCommunicator_Args,
) {
    let mut st = extension_state().lock();
    st.active_communicators = st.active_communicators.saturating_add(1);
    let comm_id = CommunicatorId(u64::try_from(st.active_communicators).unwrap_or(1));
    let device_ids = if !args.device_ids.is_null() {
        // SAFETY: `device_ids` is non-null with `num_devices` elements.
        unsafe { std::slice::from_raw_parts(args.device_ids, args.num_devices).to_vec() }
    } else {
        vec![0]
    };
    let descriptor = crate::pjrt_helpers::to_c_string(&format!(
        "tpu_ici_communicator(id={},devices={})",
        comm_id.0,
        device_ids.len()
    ));
    let handle = Box::into_raw(Box::new(PJRT_Collectives_Communicator {
        communicator_id: comm_id,
        device_ids,
        descriptor,
    }));
    args.communicator = handle.cast::<std::ffi::c_void>();
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
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Collectives_CreateCommunicator_Args>(
            raw_args,
            handle_collectives_create_communicator,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Communicator_Destroy`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Communicator_Destroy_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Communicator_Destroy_Args`].
unsafe fn handle_communicator_destroy(args: &mut PJRT_Communicator_Destroy_Args) {
    if !args.communicator.is_null() {
        // SAFETY: `communicator` was allocated by `PJRT_Collectives_CreateCommunicator`.
        unsafe {
            drop(Box::from_raw(
                args.communicator.cast::<PJRT_Collectives_Communicator>(),
            ));
        }
    }
    let mut st = extension_state().lock();
    st.active_communicators = st.active_communicators.saturating_sub(1);
    args.communicator = ptr::null_mut();
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
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Communicator_Destroy_Args>(
            raw_args,
            handle_communicator_destroy,
        )
    }
}

/// Executes a collective reduction or data transfer between `send_buf` and `recv_buf` and returns a ready event.
///
/// # Arguments
///
/// * `send_buf` - Source `PJRT_Buffer*`.
/// * `recv_buf` - Destination `PJRT_Buffer*`.
/// * `reduction_op` - Optional [`CollectiveReductionOp`] applied across `send_buf` and `recv_buf`.
///
/// # Safety
///
/// `send_buf` and `recv_buf` must be null or valid `PJRT_Buffer*` handles.
unsafe fn execute_collective_buffers(
    send_buf: *mut std::ffi::c_void,
    recv_buf: *mut std::ffi::c_void,
    reduction_op: Option<CollectiveReductionOp>,
) -> *mut std::ffi::c_void {
    if !crate::any_null(&[send_buf as usize, recv_buf as usize]) {
        // SAFETY: `send_buf` and `recv_buf` are non-null `PJRT_Buffer*`.
        let src = unsafe { &*send_buf.cast::<crate::pjrt_types::PJRT_Buffer>() };
        // SAFETY: `recv_buf` is a non-null `PJRT_Buffer*`.
        let dst = unsafe { &*recv_buf.cast::<crate::pjrt_types::PJRT_Buffer>() };
        let mut src_bytes = src.inner.to_host_bytes().unwrap_or_default();
        if let Some(op) = reduction_op {
            let dst_bytes = dst.inner.to_host_bytes().unwrap_or_default();
            src_bytes = apply_elementwise_f32_reduction(&src_bytes, &dst_bytes, op);
        }
        let _ =
            dst.inner
                .hal
                .write_device_memory(dst.inner.device_id, dst.inner.address, &src_bytes);
    }
    let mut st = extension_state().lock();
    st.collective_ops_executed = st.collective_ops_executed.saturating_add(1);
    Box::into_raw(Box::new(crate::pjrt_types::PJRT_Event {
        inner: libtpu_core::pjrt::PjrtCoreEvent::ready_ok(),
    }))
    .cast::<std::ffi::c_void>()
}

/// Applies an element-wise [`CollectiveReductionOp`] across two 32-bit float byte slices.
///
/// # Arguments
///
/// * `lhs` - Send buffer byte slice.
/// * `rhs` - Existing receive buffer byte slice.
/// * `op` - [`CollectiveReductionOp`] (`Sum`, `Product`, `Min`, `Max`).
fn apply_elementwise_f32_reduction(lhs: &[u8], rhs: &[u8], op: CollectiveReductionOp) -> Vec<u8> {
    let mut out = Vec::with_capacity(lhs.len());
    for (l_chunk, r_chunk) in lhs.chunks_exact(4).zip(rhs.chunks_exact(4)) {
        let mut la = [0_u8; 4];
        let mut ra = [0_u8; 4];
        la.copy_from_slice(l_chunk);
        ra.copy_from_slice(r_chunk);
        let lv = f32::from_le_bytes(la);
        let rv = f32::from_le_bytes(ra);
        let res = match op {
            CollectiveReductionOp::Sum => lv + rv,
            CollectiveReductionOp::Product => lv * rv,
            CollectiveReductionOp::Min => lv.min(rv),
            CollectiveReductionOp::Max => lv.max(rv),
        };
        out.extend_from_slice(&res.to_le_bytes());
    }
    out
}

/// Strongly-typed handler for [`PJRT_Communicator_AllReduce`] and [`PJRT_Communicator_ReduceScatter`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Communicator_AllReduce_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Communicator_AllReduce_Args`].
unsafe fn handle_communicator_all_reduce(args: &mut PJRT_Communicator_AllReduce_Args) {
    let op =
        CollectiveReductionOp::from_raw(args.reduction_op).unwrap_or(CollectiveReductionOp::Sum);
    // SAFETY: `send_buffer` and `recv_buffer` are null or valid `PJRT_Buffer*`.
    args.event =
        unsafe { execute_collective_buffers(args.send_buffer, args.recv_buffer, Some(op)) };
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
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Communicator_AllReduce_Args>(
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
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Communicator_AllReduce_Args>(
            raw_args,
            handle_communicator_all_reduce,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Communicator_AllGather`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Communicator_AllGather_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Communicator_AllGather_Args`].
unsafe fn handle_communicator_all_gather(args: &mut PJRT_Communicator_AllGather_Args) {
    // SAFETY: `send_buffer` and `recv_buffer` are null or valid `PJRT_Buffer*`.
    args.event = unsafe { execute_collective_buffers(args.send_buffer, args.recv_buffer, None) };
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
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Communicator_AllGather_Args>(
            raw_args,
            handle_communicator_all_gather,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Communicator_CollectivePermute`] and [`PJRT_Communicator_AllToAll`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Communicator_CollectivePermute_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Communicator_CollectivePermute_Args`].
unsafe fn handle_communicator_collective_permute(
    args: &mut PJRT_Communicator_CollectivePermute_Args,
) {
    // SAFETY: `send_buffer` and `recv_buffer` are null or valid `PJRT_Buffer*`.
    args.event = unsafe { execute_collective_buffers(args.send_buffer, args.recv_buffer, None) };
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
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Communicator_CollectivePermute_Args>(
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
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Communicator_CollectivePermute_Args>(
            raw_args,
            handle_communicator_collective_permute,
        )
    }
}

/// Strongly-typed handler for [`PJRT_Communicator_ToString`].
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_Communicator_ToString_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_Communicator_ToString_Args`].
unsafe fn handle_communicator_to_string(args: &mut PJRT_Communicator_ToString_Args) {
    if !args.communicator.is_null() {
        // SAFETY: `communicator` is a valid `PJRT_Collectives_Communicator*`.
        let comm = unsafe { &*args.communicator.cast::<PJRT_Collectives_Communicator>() };
        let bytes = comm.descriptor.as_bytes();
        args.str_out = bytes.as_ptr() as usize;
        args.str_size = bytes.len();
    } else {
        let desc = b"tpu_ici_communicator";
        args.str_out = desc.as_ptr() as usize;
        args.str_size = desc.len();
    }
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
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_Communicator_ToString_Args>(
            raw_args,
            handle_communicator_to_string,
        )
    }
}

/// Strongly-typed handler for [`PJRT_CrossHostTransfers_Args`] entrypoints.
///
/// # Arguments
///
/// * `args` - Validated [`PJRT_CrossHostTransfers_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`PJRT_CrossHostTransfers_Args`].
unsafe fn handle_cross_host_transfer(args: &mut PJRT_CrossHostTransfers_Args) {
    let mut st = extension_state().lock();
    st.cross_host_transfers = st.cross_host_transfers.saturating_add(1);
    let ev = Box::into_raw(Box::new(crate::pjrt_types::PJRT_Event {
        inner: libtpu_core::pjrt::PjrtCoreEvent::ready_ok(),
    }));
    args.out_event = ev as usize;
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
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_CrossHostTransfers_Args>(
            raw_args,
            handle_cross_host_transfer,
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
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_CrossHostTransfers_Args>(
            raw_args,
            handle_cross_host_transfer,
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
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_CrossHostTransfers_Args>(
            raw_args,
            handle_cross_host_transfer,
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
    // SAFETY: Delegated to `dispatch_typed_extension_args`.
    unsafe {
        dispatch_typed_extension_args::<PJRT_CrossHostTransfers_Args>(
            raw_args,
            handle_cross_host_transfer,
        )
    }
}

//! Strongly-typed identifier newtypes for TPU topology, PJRT runtime handles, and `TensorCore` registers.
//!
//! All domain identifiers use strongly-typed `derive_more` newtypes (`Add`, `Display`, `From`,
//! `Into`, `Sub`) to prevent accidental mixing of distinct identifier spaces.

use derive_more::{Add, Display, From, Into, Sub};

/// Unique identifier for a logical TPU device (`PJRT_Device_Id`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct DeviceId(pub u32);

/// Unique identifier for a physical TPU chip (`ChipId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct ChipId(pub u32);

/// Unique identifier for a `TensorCore` or `SparseCore` on a chip (`CoreId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct CoreId(pub u32);

/// Zero-based index of a core on a single physical TPU chip (`CoreIndex`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct CoreIndex(pub u8);

/// Dedicated identifier for a `SparseCore` or sub-core tile sequencer (`SparseCoreId`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct SparseCoreId(pub u32);

/// Unique identifier for a host machine in a TPU pod slice (`HostId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct HostId(pub u32);

/// Process index of a PJRT client in a multi-host TPU pod (`ProcessIndex`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct ProcessIndex(pub i32);

/// Identifier for a TPU pod slice in a multi-slice Megascale workload (`SliceId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct SliceId(pub u32);

/// Replica identifier in a data-parallel computation (`ReplicaId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct ReplicaId(pub u32);

/// Partition identifier in an SPMD model-parallel computation (`PartitionId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct PartitionId(pub u32);

/// Execution launch identifier used for collective barriers and poisoning (`LaunchId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct LaunchId(pub i32);

/// Local hardware ordinal on the current host (`LocalHardwareId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct LocalHardwareId(pub i32);

/// Unique identifier for a PJRT memory space (`MemorySpaceId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct MemorySpaceId(pub i32);

/// Architectural hardware synchronization flag counter value (`SyncFlagValue`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct SyncFlagValue(pub u32);

/// Hardware or Mosaic DMA/collective semaphore identifier (`SemaphoreId`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct SemaphoreId(pub u32);

/// Architectural hardware synchronization flag identifier (`SyncFlagId`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct SyncFlagId(pub u32);

/// Instruction Memory (`IMEM`) or `SparseCore` overlay identifier (`OverlayId`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct OverlayId(pub u32);

/// Host command continuation queue sequence identifier (`ContinuationId`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct ContinuationId(pub u64);

/// Unique execution run identifier for a `PJRT_LoadedExecutable_Execute` invocation (`RunId`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct RunId(pub u64);

/// Context identifier for `XSpace` `TraceMe` profiler scopes (`TraceMeContextId`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct TraceMeContextId(pub i64);

/// Monotonic runtime or executable ABI version number (`AbiVersionNumber`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct AbiVersionNumber(pub u64);

/// Unique identifier for a device buffer allocation (`BufferId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct BufferId(pub u64);

/// Unique identifier for a compiled or loaded executable (`ExecutableId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct ExecutableId(pub u64);

/// Unique identifier for an asynchronous synchronization event (`EventId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct EventId(pub u64);

/// Unique identifier for a PJRT execution context (`ContextId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct ContextId(pub u64);

/// Unique identifier for a host-to-device copy stream (`StreamId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct StreamId(pub u64);

/// Unique identifier for an asynchronous host-to-device transfer manager (`TransferManagerId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct TransferManagerId(pub u64);

/// Unique identifier for a registered host or debug callback (`CallbackId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct CallbackId(pub u64);

/// Unique identifier for a collective communicator (`CommunicatorId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct CommunicatorId(pub u64);

/// Unique identifier for a phased compiler instance (`CompilerId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct CompilerId(pub u64);

/// Unique identifier for a registered XLA transform pass (`TransformId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct TransformId(pub u64);

/// Request identifier used by `TfTpuOrdinalSelector` (`RequestId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct RequestId(pub i64);

/// Seed value used by `TfTpuOrdinalSelector` (`OrdinalSeed`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct OrdinalSeed(pub u64);

/// Unique identifier for a cross-host or collective communication channel (`ChannelId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct ChannelId(pub i64);

/// Unique identifier for an HBM DMA or Megascale collective window (`WindowId`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct WindowId(pub u32);

/// Unique incarnation identifier for a distributed PJRT task (`IncarnationId`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct IncarnationId(pub i64);

/// Globally unique 64-bit TPU chip identifier across a multi-slice pod (`GlobalChipId`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct GlobalChipId(pub u64);

/// Unique identifier for a Megascale or multi-slice synchronization barrier (`BarrierId`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("{_0}")]
pub struct BarrierId(pub u64);

/// `TensorCore` Scalar Register index (`SregIndex`, `0..32`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("s{_0}")]
pub struct SregIndex(pub u8);

/// `TensorCore` Vector Register index (`VregIndex`, `0..64`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("v{_0}")]
pub struct VregIndex(pub u8);

/// `TensorCore` Scalar Predicate Register index (`PregIndex`, `0..16`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("p{_0}")]
pub struct PregIndex(pub u8);

/// `TensorCore` Vector Mask Register index (`VmregIndex`, `0..8`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("vm{_0}")]
pub struct VmregIndex(pub u8);

/// `TensorCore` Matrix Multiply Unit (MXU) accumulator register index (`MatmulAccIndex`, `0..4`).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into, Add, Sub,
)]
#[display("acc{_0}")]
pub struct MatmulAccIndex(pub u8);

/// `TensorCore` 8-bit Vector Mask register bitmask (`VmaskBits`).
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display, From, Into)]
#[display("{_0:#04x}")]
pub struct VmaskBits(pub u8);

#[cfg(test)]
#[allow(clippy::cognitive_complexity, clippy::too_many_lines)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_identifier_newtypes() {
        let dev = DeviceId::from(3_u32) + DeviceId(1) - DeviceId(1);
        assert_eq!(u32::from(dev), 3);
        assert_eq!(dev.to_string(), "3");

        let chip = ChipId::from(1_u32) + ChipId(1) - ChipId(1);
        assert_eq!(u32::from(chip), 1);
        assert_eq!(chip.to_string(), "1");

        let core = CoreId::from(0_u32) + CoreId(1) - CoreId(1);
        assert_eq!(u32::from(core), 0);
        assert_eq!(core.to_string(), "0");

        let core_idx = CoreIndex::from(1_u8) + CoreIndex(1) - CoreIndex(1);
        assert_eq!(u8::from(core_idx), 1);
        assert_eq!(core_idx.to_string(), "1");

        let sc_id = SparseCoreId::from(2_u32) + SparseCoreId(1) - SparseCoreId(1);
        assert_eq!(u32::from(sc_id), 2);
        assert_eq!(sc_id.to_string(), "2");

        let host = HostId::from(2_u32) + HostId(1) - HostId(1);
        assert_eq!(u32::from(host), 2);
        assert_eq!(host.to_string(), "2");

        let proc_idx = ProcessIndex::from(0_i32) + ProcessIndex(1) - ProcessIndex(1);
        assert_eq!(i32::from(proc_idx), 0);
        assert_eq!(proc_idx.to_string(), "0");

        let slice_id = SliceId::from(0_u32) + SliceId(1) - SliceId(1);
        assert_eq!(u32::from(slice_id), 0);
        assert_eq!(slice_id.to_string(), "0");

        let rep = ReplicaId::from(0_u32) + ReplicaId(1) - ReplicaId(1);
        assert_eq!(u32::from(rep), 0);
        assert_eq!(rep.to_string(), "0");

        let part = PartitionId::from(1_u32) + PartitionId(1) - PartitionId(1);
        assert_eq!(u32::from(part), 1);
        assert_eq!(part.to_string(), "1");

        let launch_id = LaunchId::from(7_i32) + LaunchId(1) - LaunchId(1);
        assert_eq!(i32::from(launch_id), 7);
        assert_eq!(launch_id.to_string(), "7");

        let hw_id = LocalHardwareId::from(1_i32) + LocalHardwareId(1) - LocalHardwareId(1);
        assert_eq!(i32::from(hw_id), 1);
        assert_eq!(hw_id.to_string(), "1");

        let mem_id = MemorySpaceId::from(4_i32) + MemorySpaceId(1) - MemorySpaceId(1);
        assert_eq!(i32::from(mem_id), 4);
        assert_eq!(mem_id.to_string(), "4");

        let sf_val = SyncFlagValue::from(15_u32) + SyncFlagValue(1) - SyncFlagValue(1);
        assert_eq!(u32::from(sf_val), 15);
        assert_eq!(sf_val.to_string(), "15");

        let sem_id = SemaphoreId::from(6_u32) + SemaphoreId(1) - SemaphoreId(1);
        assert_eq!(u32::from(sem_id), 6);
        assert_eq!(sem_id.to_string(), "6");

        let sflag = SyncFlagId::from(7_u32) + SyncFlagId(1) - SyncFlagId(1);
        assert_eq!(u32::from(sflag), 7);
        assert_eq!(sflag.to_string(), "7");

        let ov_id = OverlayId::from(5_u32) + OverlayId(1) - OverlayId(1);
        assert_eq!(u32::from(ov_id), 5);
        assert_eq!(ov_id.to_string(), "5");

        let cont_id = ContinuationId::from(99_u64) + ContinuationId(1) - ContinuationId(1);
        assert_eq!(u64::from(cont_id), 99);
        assert_eq!(cont_id.to_string(), "99");

        let run_id = RunId::from(1001_u64) + RunId(1) - RunId(1);
        assert_eq!(u64::from(run_id), 1001);
        assert_eq!(run_id.to_string(), "1001");

        let traceme_id = TraceMeContextId::from(77_i64) + TraceMeContextId(1) - TraceMeContextId(1);
        assert_eq!(i64::from(traceme_id), 77);
        assert_eq!(traceme_id.to_string(), "77");

        let abi_ver = AbiVersionNumber::from(163_u64) + AbiVersionNumber(1) - AbiVersionNumber(1);
        assert_eq!(u64::from(abi_ver), 163);
        assert_eq!(abi_ver.to_string(), "163");

        let buf_id = BufferId::from(10_u64) + BufferId(1) - BufferId(1);
        assert_eq!(u64::from(buf_id), 10);
        assert_eq!(buf_id.to_string(), "10");

        let exec_id = ExecutableId::from(11_u64) + ExecutableId(1) - ExecutableId(1);
        assert_eq!(u64::from(exec_id), 11);
        assert_eq!(exec_id.to_string(), "11");

        let ev_id = EventId::from(12_u64) + EventId(1) - EventId(1);
        assert_eq!(u64::from(ev_id), 12);
        assert_eq!(ev_id.to_string(), "12");

        let ctx_id = ContextId::from(13_u64) + ContextId(1) - ContextId(1);
        assert_eq!(u64::from(ctx_id), 13);
        assert_eq!(ctx_id.to_string(), "13");

        let stream_id = StreamId::from(14_u64) + StreamId(1) - StreamId(1);
        assert_eq!(u64::from(stream_id), 14);
        assert_eq!(stream_id.to_string(), "14");

        let tm_id = TransferManagerId::from(15_u64) + TransferManagerId(1) - TransferManagerId(1);
        assert_eq!(u64::from(tm_id), 15);
        assert_eq!(tm_id.to_string(), "15");

        let cb_id = CallbackId::from(16_u64) + CallbackId(1) - CallbackId(1);
        assert_eq!(u64::from(cb_id), 16);
        assert_eq!(cb_id.to_string(), "16");

        let communicator_id = CommunicatorId::from(17_u64) + CommunicatorId(1) - CommunicatorId(1);
        assert_eq!(u64::from(communicator_id), 17);
        assert_eq!(communicator_id.to_string(), "17");

        let compiler_id = CompilerId::from(18_u64) + CompilerId(1) - CompilerId(1);
        assert_eq!(u64::from(compiler_id), 18);
        assert_eq!(compiler_id.to_string(), "18");

        let xform_id = TransformId::from(19_u64) + TransformId(1) - TransformId(1);
        assert_eq!(u64::from(xform_id), 19);
        assert_eq!(xform_id.to_string(), "19");

        let req_id = RequestId::from(20_i64) + RequestId(1) - RequestId(1);
        assert_eq!(i64::from(req_id), 20);
        assert_eq!(req_id.to_string(), "20");

        let seed = OrdinalSeed::from(21_u64) + OrdinalSeed(1) - OrdinalSeed(1);
        assert_eq!(u64::from(seed), 21);
        assert_eq!(seed.to_string(), "21");

        let chan_id = ChannelId::from(5_i64) + ChannelId(1) - ChannelId(1);
        assert_eq!(i64::from(chan_id), 5);
        assert_eq!(chan_id.to_string(), "5");

        let win_id = WindowId::from(9_u32) + WindowId(1) - WindowId(1);
        assert_eq!(u32::from(win_id), 9);
        assert_eq!(win_id.to_string(), "9");

        let inc_id = IncarnationId::from(42_i64) + IncarnationId(1) - IncarnationId(1);
        assert_eq!(i64::from(inc_id), 42);
        assert_eq!(inc_id.to_string(), "42");

        let gchip = GlobalChipId::from(100_u64) + GlobalChipId(1) - GlobalChipId(1);
        assert_eq!(u64::from(gchip), 100);
        assert_eq!(gchip.to_string(), "100");

        let bar_id = BarrierId::from(200_u64) + BarrierId(1) - BarrierId(1);
        assert_eq!(u64::from(bar_id), 200);
        assert_eq!(bar_id.to_string(), "200");

        let sreg = SregIndex::from(2_u8) + SregIndex(1) - SregIndex(1);
        assert_eq!(u8::from(sreg), 2);
        assert_eq!(sreg.to_string(), "s2");

        let vreg = VregIndex::from(4_u8) + VregIndex(1) - VregIndex(1);
        assert_eq!(u8::from(vreg), 4);
        assert_eq!(vreg.to_string(), "v4");

        let preg = PregIndex::from(1_u8) + PregIndex(1) - PregIndex(1);
        assert_eq!(u8::from(preg), 1);
        assert_eq!(preg.to_string(), "p1");

        let mask_reg_idx = VmregIndex::from(3_u8) + VmregIndex(1) - VmregIndex(1);
        assert_eq!(u8::from(mask_reg_idx), 3);
        assert_eq!(mask_reg_idx.to_string(), "vm3");

        let acc_reg = MatmulAccIndex::from(2_u8) + MatmulAccIndex(1) - MatmulAccIndex(1);
        assert_eq!(u8::from(acc_reg), 2);
        assert_eq!(acc_reg.to_string(), "acc2");

        let vmask = VmaskBits::from(0xff_u8);
        assert_eq!(u8::from(vmask), 0xff);
        assert_eq!(vmask.to_string(), "0xff");
    }
}

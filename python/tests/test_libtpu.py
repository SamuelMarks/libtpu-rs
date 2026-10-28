"""Unit test suite for the `libtpu` and `libtpu.sdk` Python packages achieving 100% coverage."""

from __future__ import annotations

import ast
import ctypes
import json
import os
from pathlib import Path
from unittest import mock

import pytest

import libtpu
import libtpu.sdk
import libtpu.sdk.slice as sdk_slice
from libtpu.sdk import debug, tpumonitoring, tpuz


def test_library_path_and_init_args(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies `get_library_path`, `configure_library_path`, and `set_init_args` / `get_init_args`.

    Args:
        monkeypatch: Pytest fixture for environment and attribute patching.
    """
    path = libtpu.get_library_path()
    assert Path(path).exists()

    with mock.patch.object(Path, "exists", return_value=False):
        fallback = libtpu.get_library_path()
        assert fallback.endswith("libtpu.dylib")

    monkeypatch.delenv("TPU_LIBRARY_PATH", raising=False)
    configured = libtpu.configure_library_path()
    assert configured == path
    assert os.environ["TPU_LIBRARY_PATH"] == path

    monkeypatch.setenv("TPU_LIBRARY_PATH", "/custom/libtpu.so")
    assert libtpu.configure_library_path() == "/custom/libtpu.so"

    libtpu.set_init_args(["--tpu_generation=v6e", "--xla_tpu_allow_async_allocations=true"])
    assert libtpu.get_init_args() == [
        "--tpu_generation=v6e",
        "--xla_tpu_allow_async_allocations=true",
    ]
    libtpu.set_init_args("--tpu_generation=v4")
    assert libtpu.get_init_args() == ["--tpu_generation=v4"]
    assert libtpu.initialize() == "/custom/libtpu.so"
    assert libtpu.initialize(["--tpu_generation=v5p"]) == "/custom/libtpu.so"
    assert libtpu.get_init_args() == ["--tpu_generation=v5p"]
    assert libtpu.Client is libtpu.TpuClient
    assert libtpu.DeviceBuffer is libtpu.TpuBuffer
    assert libtpu.LoadedExecutable is libtpu.TpuExecutable
    assert libtpu.TpuGeneration.V6E.value == 7
    assert libtpu.DataType.F32.value == 11


def test_client_buffer_executable_and_profiler() -> None:
    """Verifies `TpuClient`, `TpuBuffer`, `TpuExecutable`, `TpuProfiler`, and `TpuError`."""
    client = libtpu.TpuClient()
    assert client.device_count() == 4

    b0 = client.buffer_from_floats(0, [1.5, 2.5, 3.5, 4.5])
    b1 = client.buffer_from_floats(0, [10.0, 20.0, 30.0, 40.0])

    with pytest.raises(libtpu.TpuError) as exc_info:
        client.buffer_from_floats(99, [1.0, 2.0, 3.0, 4.0])
    assert exc_info.value.code == 5

    with pytest.raises(libtpu.TpuError):
        client.compile("SYNTAX_ERROR", fmt="mlir")

    executable = client.compile("module @add { stablehlo.add }", fmt="mlir")
    assert len(executable.fingerprint()) == 32
    assert executable.flops() > 0.0
    assert b"stablehlo.add" in executable.optimized_program_bytes()
    ser_bytes = executable.serialize()
    deser_exec = client.deserialize_executable(ser_bytes)
    out_buf = deser_exec.execute(0, [b0, b1])
    assert out_buf.shape() == (4,)
    assert out_buf.dtype() == libtpu.DataType.F32
    assert out_buf.tile_count() == 1
    assert out_buf.to_float_list() == [11.5, 22.5, 33.5, 44.5]
    deser_exec.close()

    copied_b0 = client.copy_buffer_to_device(b0, 1)
    assert copied_b0.to_float_list() == [1.5, 2.5, 3.5, 4.5]
    copied_b0.close()

    multi_outs = executable.execute_multi(0, [b0, b1], num_outputs=1)
    assert len(multi_outs) == 1
    assert multi_outs[0].to_float_list() == [11.5, 22.5, 33.5, 44.5]
    multi_outs[0].close()

    # Rectangular 2x3 * 3x2 -> 2x2 dot_general execution
    m0 = client.buffer_from_floats(0, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dims=[2, 3])
    m1 = client.buffer_from_floats(0, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dims=[3, 2])
    dot_exec = client.compile(
        "module @rect_dot { func.func @main("
        "%arg0: tensor<2x3xf32>, %arg1: tensor<3x2xf32>) -> tensor<2x2xf32> {\n"
        " %0 = stablehlo.dot_general %arg0, %arg1 : "
        "(tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>\n"
        " return %0 : tensor<2x2xf32>\n} }",
        fmt="mlir",
    )
    dot_out = dot_exec.execute(0, [m0, m1])
    assert dot_out.shape() == (2, 2)
    assert dot_out.to_float_list() == [22.0, 28.0, 49.0, 64.0]
    dot_out.close()
    dot_exec.close()
    m0.close()
    m1.close()

    i0 = client.buffer_from_ints(0, [10, 20, 30, 40])
    i1 = client.buffer_from_ints(0, [1, 2, 3, 4])
    exec_s32 = client.compile("module @add_s32 { s32_add }", fmt="mlir")
    out_s32 = exec_s32.execute(0, [i0, i1])
    assert out_s32.to_int_list() == [11, 22, 33, 44]
    out_s32.close()
    i0.close()
    i1.close()
    exec_s32.close()

    for dt in libtpu.DataType:
        assert dt.bit_size() >= 0
        assert dt.byte_size_for_elements(4) >= 0

    bf16_buf = client.buffer_from_bytes(
        0, libtpu.DataType.BF16, b"\x00\x3f\x00\x40\x40\x40\x80\x40", [2, 2]
    )
    assert bf16_buf.dtype() == libtpu.DataType.BF16
    assert bf16_buf.shape() == (2, 2)
    assert bf16_buf.byte_size() == 8
    assert bf16_buf.to_bytes() == b"\x00\x3f\x00\x40\x40\x40\x80\x40"
    bf16_buf.close()

    s4_buf = client.buffer_from_bytes(0, libtpu.DataType.S4, b"\x12\x34", [4])
    assert s4_buf.dtype() == libtpu.DataType.S4
    assert s4_buf.byte_size() == 2
    assert s4_buf.to_bytes() == b"\x12\x34"
    s4_buf.close()

    out_buf.close()
    out_buf.close()
    b0.close()
    b1.close()
    executable.close()
    executable.close()
    client.close()
    client.close()

    prof = libtpu.TpuProfiler()
    assert prof.start() is True
    assert prof.stop() is True
    data = prof.collect_data()
    assert b"/device:TPU:0" in data
    prof.close()
    prof.close()


def test_sdk_submodules() -> None:
    """Verifies `tpumonitoring`, `slice`, `tpuz`, and `debug` submodules."""
    metrics = tpumonitoring.list_supported_metrics()
    assert "tpu.runtime.hbm.usage_bytes" in metrics
    assert tpumonitoring.list_metric_names() == metrics
    assert tpumonitoring.Metric is tpumonitoring.LibtpuSdkMetric
    assert len(tpumonitoring.get_chip_info()) == 4

    metric = libtpu.sdk.get_metric("tpu.runtime.hbm.usage_bytes")
    assert "HBM" in metric.help()
    assert len(metric.data()) == 4

    with pytest.raises(libtpu.TpuError):
        tpumonitoring.get_metric("nonexistent.metric")

    coords = sdk_slice.get_chip_coordinates()
    assert len(coords) == 4
    assert coords[0].hostname == "tpu-worker-0"
    assert coords[0].chip_index == 0
    assert coords[0].coordinates == [0, 0, 0]
    assert sdk_slice.get_host_name(0) == "tpu-worker-0"
    assert sdk_slice.get_chip_index(0) == 0
    assert len(sdk_slice.get_slice_topology()) == 4

    summary = tpuz.get_core_state_summary(include_hlo_info=True)
    assert "READY" in summary
    assert tpuz.start_server(8466) == 8466
    assert b"/device:TPU:0" in tpuz.collect_xspace()

    sdk_client = tpumonitoring.SdkClient()
    sdk_client.close()
    sdk_client.close()

    triggered: list[int] = []
    cb_id1 = debug.register_debug_callback(lambda: triggered.append(1))
    cb_id2 = debug.register_callback(lambda: triggered.append(2))
    assert cb_id1 > 0 and cb_id2 > cb_id1
    debug.trigger_fault_injection()
    assert triggered == [1, 2]
    debug.unregister_debug_callback(cb_id1)
    debug.unregister_callback(cb_id2)
    with pytest.raises(libtpu.TpuError):
        debug.unregister_debug_callback(cb_id1)


def test_dlopen_abi_symbols_and_docstring_audit() -> None:
    """Verifies exported C ABI symbols via `dlopen` and 100% Python docstring coverage."""
    cdll = ctypes.CDLL(libtpu.get_library_path())
    exported_symbols = (
        "GetPjrtApi",
        "GetLibtpuSdkApi",
        "TfTpu_Initialize",
        "TF_InitKernel",
        "TFNPD_InitPlugin",
        "SetGlobalTPUArrayOp_DoWork",
        "TfTpu_GetTpuPartitionedCallParams",
        "TpuConfigurationApi_FreeCharArray",
        "TpuTransferManager_GetInfeedLayout",
        "SparseCore_GetMaxIdsAndUniques",
        "TpuStatus_New",
        "TpuStatus_Create",
        "TpuStatus_Set",
        "TpuStatus_Free",
        "TpuStatus_Message",
        "TpuStatus_Code",
        "TpuStatus_Ok",
        "TfTpuOrdinalSelector_Create",
        "TfTpuOrdinalSelector_Destroy",
        "TfTpuOrdinalSelector_GetOrdinal",
        "TfTpuOrdinalSelector_DequeueFromCoreSelector",
        "TpuUtil_GetTopologyPtr",
        "TpuUtil_GetXlaPadSizeFromTpuTopology",
        "TpuTopology_LogicalDevicesPerHost",
        "TpuTopology_LogicalDevicesPerChip",
        "TpuTopology_HostCount",
        "TpuTopology_ChipsPerHost",
        "TpuTopology_ChipBounds_X",
        "TpuTopology_ChipBounds_Y",
        "TpuTopology_ChipBounds_Z",
        "TpuTopology_HasChip",
        "TpuTopology_CoreForId",
        "TpuTopology_Core",
        "TpuTopology_NumCores",
        "TpuTopology_Cores",
        "TpuTopology_IdForHost",
        "TpuTopology_Version",
        "TpuTopology_AvailableCoreCount",
        "TpuTopology_AvailableCoresPerChip",
        "TpuTopology_MaybeAvailableSparseCoresPerLogicalDevice",
        "TpuCoreLocation_ChipCoordinates",
        "TpuCoreLocation_HostCoordinates",
        "TpuCoreLocation_Index",
        "TpuCoreLocation_Id",
        "TpuHostLocation_Id",
        "TpuHostLocation_NumCores",
        "TpuHostLocation_Cores",
        "TpuMeshState_Create",
        "TpuMeshState_Free",
        "TpuMeshState_MeshCommonState",
        "TpuProfiler_Create",
        "TpuProfiler_Destroy",
        "TpuProfiler_Start",
        "TpuProfiler_Stop",
        "TpuProfiler_CollectData",
        "TpuEmbeddingEngineState_Create",
        "TpuEmbeddingEngineState_Free",
        "TpuEmbeddingEngineState_GetState",
        "TpuEmbeddingEngine_ExecutePartitioner",
        "TpuEmbeddingEngine_ConfigureMemory",
        "TpuEmbeddingEngine_CollateMemory",
        "TpuEmbeddingEngine_ConfigureHost",
        "TpuEmbeddingEngine_ConnectHosts",
        "TpuEmbeddingEngine_Finalize",
        "TpuEmbeddingEngine_IsInitialized",
        "TpuEmbeddingEngine_WriteParameters",
        "TpuEmbeddingEngine_ReadParameters",
        "TpuEmbeddingTensorBatchFixedState_Create",
        "TpuEmbeddingTensorBatchFixedState_Destroy",
        "TpuEmbeddingEngine_EnqueueTensorBatch",
        "TpuEmbeddingEngine_RecvActivationsComputation",
        "TpuEmbeddingEngine_RecvTPUEmbeddingDeduplicationDataComputation",
        "TpuEmbeddingEngine_SendTPUEmbeddingGradientsComputation",
        "TpuEmbeddingEngine_DedupDataSizeComputation",
        "TpuEmbeddingEngine_DedupDataTupleMaskComputation",
        "TpuDriver_Initialize",
        "TpuDriver_Open",
        "TpuDriver_Close",
        "InitializeCreateMeshGlobalState",
        "ConfigureDistributedTpuOp_DoWork",
        "WaitForDistributedTpuOp_DoWork",
        "InitializeHostForDistributedTpuOp_DoWork",
        "SetGlobalTPUArrayOp",
        "DisconnectDistributedTpuChipsOp_DoWork",
        "ShutdownDistributedTpuOp_DoWork",
        "TpuNodeContext_Initialize",
        "TpuNodeContext_Create",
        "TpuNodeContext_CloseTpuHost",
        "TpuNodeContext_StopChipHeartbeats",
        "TpuNodeContext_CompilationsDisabled",
        "TpuCompile_CompileAndBuild",
        "TpuCompile_XrtCompileAndBuild",
        "TpuExecutable_LoadProgramAndEnqueueToStream",
        "TpuResourceMgr_Create",
        "TpuResourceMgr_Destroy",
        "TpuStatus_Destroy",
        "TpuDriver_FreeStatus",
        "TpuTopology_AnAvailableCore",
        "TpuTopology_AvailableCores",
        "TpuTopology_AvailableCoresCount",
        "TpuTopology_ChipCoordinates",
        "TpuTopology_ChipCount",
        "TpuTopology_CoreCount",
        "TpuTopology_CoreId",
        "TpuTopology_CoreIndexOnChip",
        "TpuTopology_CoreLocations",
        "TpuTopology_CurrentHost",
        "TpuTopology_HostBounds_X",
        "TpuTopology_HostBounds_Y",
        "TpuTopology_HostBounds_Z",
        "TpuTopology_HostLocations",
        "TpuDriver_Reset",
        "TpuDriver_QuerySystemInfo",
        "TpuDriver_FreeSystemInfo",
        "TpuDriver_Allocate",
        "TpuDriver_AllocateTuple",
        "TpuDriver_AllocateShape",
        "TpuDriver_Deallocate",
        "TpuDriver_TransferToDevice",
        "TpuDriver_TransferFromDevice",
        "TpuDriver_TransferFromDeviceToDevice",
        "TpuDriver_CompileProgram",
        "TpuDriver_CompileProgramFromText",
        "TpuDriver_LoadProgram",
        "TpuDriver_UnloadProgram",
        "TpuDriver_ExecuteProgram",
        "TpuDriver_CreateEvent",
        "TpuDriver_FreeEvent",
        "TpuDriver_AddCallback",
        "TpuDriver_WaitForEvent",
        "TpuDriver_ComputeLinearizedBytesFromShape",
        "TpuDriver_LinearizeShape",
        "TpuDriver_DelinearizeShape",
        "TpuDriver_GetCompiledProgramShape",
        "TpuDriver_FreeCompiledProgramShape",
        "TpuDriver_Version",
    )
    for sym in exported_symbols:
        assert hasattr(cdll, sym), f"Missing exported C ABI symbol: {sym}"

    manifest_path = (
        Path(libtpu.__file__).resolve().parent.parent.parent
        / "docs"
        / "reverse_engineering"
        / "golden_binary_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["libtpu_so"]["sha256"]) == 64
    assert len(manifest["sdk_so"]["sha256"]) == 64
    assert manifest["libtpu_so"]["exported_functions_count"] == 78
    assert len(manifest["libtpu_so"]["exported_functions"]) == 78
    for fn_meta in manifest["libtpu_so"]["exported_functions"]:
        assert hasattr(cdll, fn_meta["name"])

    cdll.GetPjrtApi.argtypes = []
    cdll.GetPjrtApi.restype = ctypes.c_void_p
    pjrt_ptr = cdll.GetPjrtApi()
    assert pjrt_ptr is not None and pjrt_ptr != 0
    pjrt_words = (ctypes.c_uint64 * 143).from_address(pjrt_ptr)
    assert pjrt_words[0] == 1144
    slot_addrs = [int(pjrt_words[slot_idx]) for slot_idx in range(5, 143)]
    for addr in slot_addrs:
        assert addr != 0
    assert len(set(slot_addrs)) == 138

    ext_ptr = int(pjrt_words[1])
    ext_types: list[int] = []
    ext_fn_addrs: list[int] = []
    while ext_ptr != 0:
        ext_hdr = (ctypes.c_uint64 * 3).from_address(ext_ptr)
        struct_sz = int(ext_hdr[0])
        ext_type = int(ext_hdr[1] & 0xFFFFFFFF)
        ext_types.append(ext_type)
        if ext_type != 1:
            num_fns = (struct_sz - 24) // 8
            fn_words = (ctypes.c_uint64 * (3 + num_fns)).from_address(ext_ptr)
            for i in range(3, 3 + num_fns):
                ext_fn_addrs.append(int(fn_words[i]))
        ext_ptr = int(ext_hdr[2])
    assert ext_types == [29, 28, 26, 25, 20, 19, 18, 17, 16, 14, 9, 12, 11, 7, 6, 8, 1]
    assert len(ext_fn_addrs) == 130
    assert len(set(ext_fn_addrs)) == 130

    # Verify direct PJRT_Api C vtable execution
    pjrt_fn_type = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)
    fn_client_create = pjrt_fn_type(int(pjrt_words[15]))
    fn_client_destroy = pjrt_fn_type(int(pjrt_words[16]))
    fn_client_compile = pjrt_fn_type(int(pjrt_words[25]))
    fn_buf_from_host = pjrt_fn_type(int(pjrt_words[27]))
    fn_exec_destroy = pjrt_fn_type(int(pjrt_words[55]))
    fn_exec_execute = pjrt_fn_type(int(pjrt_words[60]))
    fn_buf_destroy = pjrt_fn_type(int(pjrt_words[63]))
    fn_buf_to_host = pjrt_fn_type(int(pjrt_words[75]))

    create_words = (ctypes.c_uint64 * 11)(88, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    assert fn_client_create(ctypes.byref(create_words)) is None
    raw_pjrt_client = int(create_words[10])
    assert raw_pjrt_client != 0

    in_floats = (ctypes.c_float * 4)(2.0, 4.0, 6.0, 8.0)
    dims_arr = (ctypes.c_int64 * 1)(4)
    h2d_words = (ctypes.c_uint64 * 15)(
        120,
        0,
        raw_pjrt_client,
        ctypes.addressof(in_floats),
        11,
        ctypes.addressof(dims_arr),
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    assert fn_buf_from_host(ctypes.byref(h2d_words)) is None
    raw_buf = int(h2d_words[14])
    assert raw_buf != 0

    mlir_src = b"module @vtable_add { stablehlo.add }"
    mlir_fmt = b"mlir"
    prog_words = (ctypes.c_uint64 * 6)(
        48,
        0,
        ctypes.cast(ctypes.c_char_p(mlir_src), ctypes.c_void_p).value or 0,
        len(mlir_src),
        ctypes.cast(ctypes.c_char_p(mlir_fmt), ctypes.c_void_p).value or 0,
        len(mlir_fmt),
    )
    comp_words = (ctypes.c_uint64 * 7)(
        56,
        0,
        raw_pjrt_client,
        ctypes.addressof(prog_words),
        0,
        0,
        0,
    )
    assert fn_client_compile(ctypes.byref(comp_words)) is None
    raw_exec = int(comp_words[6])
    assert raw_exec != 0

    in_list = (ctypes.c_uint64 * 2)(raw_buf, raw_buf)
    in_lists = (ctypes.c_uint64 * 1)(ctypes.addressof(in_list))
    out_list = (ctypes.c_uint64 * 1)(0)
    out_lists = (ctypes.c_uint64 * 1)(ctypes.addressof(out_list))
    exec_words = (ctypes.c_uint64 * 10)(
        80,
        0,
        raw_exec,
        0,
        ctypes.addressof(in_lists),
        1,
        2,
        ctypes.addressof(out_lists),
        0,
        0,
    )
    assert fn_exec_execute(ctypes.byref(exec_words)) is None
    raw_out_buf = int(out_list[0])
    assert raw_out_buf != 0

    out_floats = (ctypes.c_float * 4)(0.0, 0.0, 0.0, 0.0)
    d2h_words = (ctypes.c_uint64 * 7)(
        56,
        0,
        raw_out_buf,
        0,
        ctypes.addressof(out_floats),
        16,
        0,
    )
    assert fn_buf_to_host(ctypes.byref(d2h_words)) is None
    assert list(out_floats) == [4.0, 8.0, 12.0, 16.0]

    for b_handle in (raw_buf, raw_out_buf):
        b_del = (ctypes.c_uint64 * 3)(24, 0, b_handle)
        assert fn_buf_destroy(ctypes.byref(b_del)) is None
    e_del = (ctypes.c_uint64 * 3)(24, 0, raw_exec)
    assert fn_exec_destroy(ctypes.byref(e_del)) is None
    c_del = (ctypes.c_uint64 * 3)(24, 0, raw_pjrt_client)
    assert fn_client_destroy(ctypes.byref(c_del)) is None

    # Verify all 138 PJRT_Api slots against golden_struct_layouts.json min_struct_size
    golden_path = (
        Path(libtpu.__file__).resolve().parent.parent.parent
        / "docs"
        / "reverse_engineering"
        / "golden_struct_layouts.json"
    )
    golden_layouts = json.loads(golden_path.read_text(encoding="utf-8"))
    pjrt_u64_fn_type = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_void_p)
    pjrt_void_fn_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    fn_err_destroy = pjrt_void_fn_type(int(pjrt_words[5]))
    fn_err_message = pjrt_void_fn_type(int(pjrt_words[6]))
    fn_ev_destroy = pjrt_u64_fn_type(int(pjrt_words[10]))
    for slot_entry in golden_layouts["pjrt_slots"]:
        word_idx = int(slot_entry["word_index"])
        min_sz = slot_entry["min_struct_size"]
        if word_idx >= 7 and min_sz is not None:
            slot_fn = pjrt_u64_fn_type(int(pjrt_words[word_idx]))
            small_buf = (ctypes.c_uint64 * 32)()
            small_buf[0] = int(min_sz) - 1
            err_ptr = int(slot_fn(ctypes.byref(small_buf)))
            assert err_ptr != 0, (
                f"Slot word {word_idx} accepted struct_size={int(min_sz) - 1} < {min_sz}"
            )
            msg_buf = (ctypes.c_uint64 * 5)(40, 0, err_ptr, 0, 0)
            fn_err_message(ctypes.byref(msg_buf))
            err_text = ctypes.string_at(int(msg_buf[3]), int(msg_buf[4])).decode("utf-8")
            assert "struct_size" in err_text
            err_del = (ctypes.c_uint64 * 3)(24, 0, err_ptr)
            fn_err_destroy(ctypes.byref(err_del))

            valid_buf = (ctypes.c_uint64 * 32)()
            valid_buf[0] = int(min_sz)
            ok_err = int(slot_fn(ctypes.byref(valid_buf)))
            if ok_err != 0:
                msg_buf2 = (ctypes.c_uint64 * 5)(40, 0, ok_err, 0, 0)
                fn_err_message(ctypes.byref(msg_buf2))
                err_text2 = ctypes.string_at(int(msg_buf2[3]), int(msg_buf2[4])).decode("utf-8")
                assert "struct_size" not in err_text2, (
                    f"Slot word {word_idx} rejected valid struct_size={min_sz}: {err_text2}"
                )
                err_del2 = (ctypes.c_uint64 * 3)(24, 0, ok_err)
                fn_err_destroy(ctypes.byref(err_del2))
            if word_idx == 15 and int(valid_buf[10]) != 0:
                cl_del = (ctypes.c_uint64 * 3)(24, 0, int(valid_buf[10]))
                assert fn_client_destroy(ctypes.byref(cl_del)) is None
            if word_idx == 131 and int(valid_buf[2]) != 0:
                ev_del = (ctypes.c_uint64 * 3)(24, 0, int(valid_buf[2]))
                assert int(fn_ev_destroy(ctypes.byref(ev_del))) == 0

    for mod_path in (
        Path(libtpu.__file__),
        Path(libtpu.sdk.__file__),
        Path(tpumonitoring.__file__),
        Path(sdk_slice.__file__),
        Path(tpuz.__file__),
        Path(debug.__file__),
    ):
        source = mod_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert ast.get_docstring(tree) is not None, f"Missing module docstring in {mod_path}"
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                doc = ast.get_docstring(node)
                assert doc is not None, f"Missing docstring on {node.name} in {mod_path}"
                if isinstance(node, ast.FunctionDef):
                    params = [a.arg for a in node.args.args if a.arg != "self"]
                    for param in params:
                        assert f"{param}:" in doc, (
                            f"Missing arg doc for {param} in {node.name} ({mod_path})"
                        )

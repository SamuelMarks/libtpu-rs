"""Unit and branch-complete tests for Phase 11.1/11.2 (`PyO3`, `ml_dtypes`, PEP 3118, `DLPack`)."""

from __future__ import annotations

import ctypes
import importlib.util
import json
import subprocess
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import ml_dtypes
import numpy as np
import pytest

import libtpu
import libtpu._interop as interop
import libtpu.sdk
import scripts.maturin_backend as mb


def test_native_pyo3_c_and_sdk_modules() -> None:
    """Verifies live `PyO3` (`abi3`) `libtpu._C` and `libtpu/sdk.so` modules and error mapping."""
    interop._NATIVE_EXT_CACHE.clear()
    c_mod = libtpu.load_native_c_module()
    assert c_mod is libtpu.load_native_c_module()
    assert c_mod.__version__ == libtpu.__version__
    assert c_mod.pjrt_api_version() == (0, 116)
    assert c_mod.protodesc_cold_count() == 738

    sdk_mod = libtpu.sdk.load_native_sdk_module()
    assert sdk_mod is libtpu.load_native_sdk_module()
    metrics = sdk_mod.list_supported_metrics()
    assert "tpu.runtime.hbm.usage_bytes" in metrics
    desc, vals = sdk_mod.get_metric("tpu.runtime.hbm.usage_bytes", "v5e")
    assert len(desc) > 0
    assert len(vals) > 0

    with pytest.raises(libtpu.TpuError) as exc_info:
        interop.call_native_c("parse_init_args", "--tpu_generation=invalid")
    assert exc_info.value.code != 0

    with (
        patch.object(Path, "is_file", return_value=False),
        pytest.raises(FileNotFoundError),
    ):
        interop._resolve_pyo3_extension_path(prefer_sdk=False)

    with (
        patch.dict(interop._NATIVE_EXT_CACHE, {}, clear=True),
        patch.object(importlib.util, "spec_from_loader", return_value=None),
        pytest.raises(ImportError),
    ):
        interop._load_extension_symbol("libtpu._broken", "_C", prefer_sdk=False)


def test_tpu_device_and_topology_wrappers() -> None:
    """Verifies `TpuDevice`, `TpuTopology`, and `TpuClient` device/topology queries."""
    interop._INTEROP_CDLL_CACHE.clear()
    with libtpu.TpuTopology.create("v5e-2x2") as topo:
        assert topo.platform_name() == "tpu"
        assert len(topo.platform_version()) > 0
        assert topo.device_count() == 4
        ser = topo.serialize()
        assert len(ser) > 0

    with libtpu.TpuTopology.deserialize(ser) as topo2:
        assert topo2.device_count() == 4
    topo2.close()

    cdll = interop._load_interop_cdll()
    bad_args = interop._PjrtTopologyDeserializeArgs(8, 0, 0, 0, 0)
    raw_err = cdll.PJRT_TopologyDescription_Deserialize(ctypes.byref(bad_args))
    with pytest.raises(libtpu.TpuError):
        interop._check_interop_pjrt_error(cdll, raw_err)

    with libtpu.TpuClient() as client:
        all_devs = client.devices()
        addr_devs = client.addressable_devices()
        assert len(all_devs) == 4
        assert len(addr_devs) == 4

        dev0 = client.lookup_device(0)
        assert dev0.id() == 0
        assert dev0.process_index() == 0
        assert "TPU" in dev0.device_kind()
        assert len(dev0.debug_string()) > 0
        assert "TpuDevice" in dev0.to_string()
        assert "TpuDevice" in repr(dev0)
        assert dev0.is_addressable() is True
        assert dev0.local_hardware_id() == 0
        assert dev0.coords() == (0, 0, 0)
        assert dev0.core_on_chip() == 0

        # Empty attribute table fallback branch on `coords()` and `core_on_chip()`.
        empty_dev = interop.TpuDevice(interop._load_interop_cdll(), ctypes.c_void_p(0))
        assert empty_dev.coords() == (0, 0, 0)
        assert empty_dev.core_on_chip() == 0

        with pytest.raises(libtpu.TpuError):
            client.lookup_device(99)

        with client.topology() as client_topo:
            assert client_topo.device_count() == 4
            assert client_topo.platform_name() == "tpu"


def test_pep3118_numpy_ml_dtypes_and_dlpack_roundtrip() -> None:
    """Verifies PEP 3118 `__buffer__`, `ml_dtypes` NumPy interop, and `DLPack` round-trips."""
    with libtpu.TpuClient() as client:
        # 1. bfloat16 round-trip via `buffer_from_numpy`, `to_numpy()`, and `np.asarray()`.
        bf16_in = np.array([[1.5, -2.5], [3.25, -4.0]], dtype=ml_dtypes.bfloat16)
        with client.buffer_from_numpy(0, bf16_in) as bf16_buf:
            assert bf16_buf.dtype() == libtpu.DataType.BF16
            assert bf16_buf.shape() == (2, 2)
            assert bytes(bf16_buf.__buffer__(0)) == bf16_in.tobytes()
            assert bytes(bf16_buf.to_memoryview()) == bf16_in.tobytes()
            bf16_out = bf16_buf.to_numpy()
            assert bf16_out.dtype == np.dtype(ml_dtypes.bfloat16)
            np.testing.assert_array_equal(bf16_out, bf16_in)
            as_f64 = bf16_buf.__array__(dtype=np.float64)
            assert as_f64.dtype == np.float64
            np.testing.assert_allclose(as_f64, bf16_in.astype(np.float64))
            with pytest.raises(ValueError, match="copy"):
                bf16_buf.__array__(copy=False)

        # 2. float8 (`float8_e4m3fn`, `float8_e5m2`) and `float4_e2m1fn` round-trips.
        for f_dtype, expected_tpu_dt in (
            (ml_dtypes.float8_e4m3fn, libtpu.DataType.F8E4M3FN),
            (ml_dtypes.float8_e5m2, libtpu.DataType.F8E5M2),
            (ml_dtypes.float4_e2m1fn, libtpu.DataType.F4E2M1FN),
        ):
            fp_arr = np.array([1.0, -1.5, 2.0, 0.5], dtype=f_dtype)
            with client.buffer_from_numpy(0, fp_arr) as fp_buf:
                assert fp_buf.dtype() == expected_tpu_dt
                np.testing.assert_array_equal(fp_buf.to_numpy(), fp_arr)
                np.testing.assert_array_equal(fp_buf.__array__(), fp_arr)

        # 3. Signed & unsigned sub-byte `int4`/`uint4`/`int2`/`uint2` round-trips.
        s4_arr = np.array([-8, -3, 0, 7], dtype=ml_dtypes.int4)
        with client.buffer_from_numpy(0, s4_arr) as s4_buf:
            assert s4_buf.dtype() == libtpu.DataType.S4
            assert s4_buf.byte_size() == 2
            ai = s4_buf.__array_interface__
            assert ai["shape"] == (4,)
            assert isinstance(ai["data"], bytes)
            assert len(ai["data"]) == 4
            s4_out = s4_buf.to_numpy()
            assert s4_out.dtype == np.dtype(ml_dtypes.int4)
            np.testing.assert_array_equal(s4_out.astype(np.int32), np.array([-8, -3, 0, 7]))

        u4_arr = np.array([0, 5, 10, 15], dtype=ml_dtypes.uint4)
        with client.buffer_from_numpy(0, u4_arr) as u4_buf:
            assert u4_buf.dtype() == libtpu.DataType.U4
            u4_data = u4_buf.__array_interface__["data"]
            assert isinstance(u4_data, bytes)
            assert len(u4_data) == 4
            np.testing.assert_array_equal(u4_buf.to_numpy().astype(np.int32), [0, 5, 10, 15])

        s2_arr = np.array([-2, -1, 0, 1], dtype=ml_dtypes.int2)
        with client.buffer_from_numpy(0, s2_arr) as s2_buf:
            assert s2_buf.dtype() == libtpu.DataType.S2
            np.testing.assert_array_equal(s2_buf.to_numpy().astype(np.int32), [-2, -1, 0, 1])

        u2_arr = np.array([0, 1, 2, 3], dtype=ml_dtypes.uint2)
        with client.buffer_from_numpy(0, u2_arr) as u2_buf:
            assert u2_buf.dtype() == libtpu.DataType.U2
            np.testing.assert_array_equal(u2_buf.to_numpy().astype(np.int32), [0, 1, 2, 3])

        # Unsupported NumPy dtype & non-representable TOKEN buffer errors.
        with pytest.raises(libtpu.TpuError):
            client.buffer_from_numpy(0, np.array(["abc", "def"]))
        with (
            client.buffer_from_bytes(0, libtpu.DataType.TOKEN, b"", [1]) as tok_buf,
            pytest.raises(libtpu.TpuError),
        ):
            tok_buf.to_numpy()

        # 4. DLPack (`__dlpack__`, `__dlpack_device__`, `from_dlpack`, and `np.from_dlpack`).
        np_src = np.array([[1.25, 2.5], [3.75, 5.0]], dtype=np.float32)
        with libtpu.from_dlpack(np_src, client=client, device_id=0) as dl_buf:
            assert dl_buf.__dlpack_device__() == (14, 0)
            np.testing.assert_array_equal(dl_buf.to_numpy(), np_src)

            # Copy to device 1 preserves device_id in `__dlpack_device__()`.
            with client.copy_buffer_to_device(dl_buf, 1) as dl_buf_dev1:
                assert dl_buf_dev1.__dlpack_device__() == (14, 1)

            # Cross-device copy=False must raise BufferError; same-device copy=False succeeds.
            with pytest.raises(BufferError):
                dl_buf.__dlpack__(dl_device=(1, 0), copy=False)
            same_cap = dl_buf.__dlpack__(dl_device=(14, 0), copy=False)
            with libtpu.from_dlpack(same_cap, client=client) as round_buf:
                np.testing.assert_array_equal(round_buf.to_numpy(), np_src)
            # Re-consuming an already-used `"used_dltensor"` capsule must raise `TpuError`!
            with pytest.raises(libtpu.TpuError):
                libtpu.from_dlpack(same_cap, client=client)

            # Export to CPU `(1, 0)` and consume via `np.from_dlpack` via a lightweight CPU proxy.
            class _CpuDlPackView:
                """CPU DLPack adapter delegating to `TpuBuffer.__dlpack__` on CPU `(1, 0)`."""

                def __dlpack_device__(self) -> tuple[int, int]:
                    """Returns the `(kDLCPU, 0)` DLPack device tuple."""
                    return (1, 0)

                def __dlpack__(self, stream: int | None = None) -> object:
                    """Exports a host-copied CPU DLPack capsule.

                    Args:
                        stream: Optional stream identifier.

                    Returns:
                        `"dltensor"` `PyCapsule` on CPU `(1, 0)`.
                    """
                    return dl_buf.__dlpack__(stream=stream, dl_device=(1, 0), copy=True)

            np_zero_copy = np.from_dlpack(_CpuDlPackView())
            np.testing.assert_array_equal(np_zero_copy, np_src)

        # Default `client=None` path on `from_dlpack` + unsupported DLPack `(code, bits)` error.
        with libtpu.from_dlpack(np.array([10, 20], dtype=np.int32)) as auto_buf:
            assert auto_buf.to_int_list() == [10, 20]

        bad_cap = interop.call_native_c("export_dlpack", b"\x00\x00\x00", [1], 7, 24, 1, 0)
        with pytest.raises(libtpu.TpuError):
            libtpu.from_dlpack(bad_cap, client=client)


def test_maturin_backend_hooks_and_wheel_injection(tmp_path: Path) -> None:
    """Verifies 100% of `scripts/maturin_backend.py` PEP 517 hooks and wheel `RECORD` logic.

    Args:
        tmp_path: Temporary workspace directory fixture.
    """
    assert mb._urlsafe_sha256_digest(b"abc").startswith("sha256=")

    with patch.dict("os.environ", {"CARGO_TARGET_DIR": str(tmp_path / "env_target")}):
        assert mb._resolve_cargo_target_dir(tmp_path) == tmp_path / "env_target"

    fake_meta = json.dumps({"target_directory": str(tmp_path / "meta_target")})
    with (
        patch.dict("os.environ", {"CARGO_TARGET_DIR": ""}),
        patch.object(
            subprocess,
            "run",
            return_value=subprocess.CompletedProcess(["cargo"], 0, stdout=fake_meta, stderr=""),
        ),
    ):
        assert mb._resolve_cargo_target_dir(tmp_path) == tmp_path / "meta_target"

    empty_meta = json.dumps({"target_directory": ""})
    with (
        patch.dict("os.environ", {"CARGO_TARGET_DIR": ""}),
        patch.object(
            subprocess,
            "run",
            return_value=subprocess.CompletedProcess(["cargo"], 0, stdout=empty_meta, stderr=""),
        ),
    ):
        assert mb._resolve_cargo_target_dir(tmp_path) == tmp_path / "target"

    with (
        patch.dict("os.environ", {"CARGO_TARGET_DIR": ""}),
        patch.object(
            subprocess,
            "run",
            return_value=subprocess.CompletedProcess(["cargo"], 1, stdout="", stderr="err"),
        ),
    ):
        assert mb._resolve_cargo_target_dir(tmp_path) == tmp_path / "target"

    # `_find_or_build_pjrt_cdylib`: existing file vs cargo build vs failure.
    pre_dir = tmp_path / "python" / "libtpu"
    pre_dir.mkdir(parents=True)
    (pre_dir / "libtpu.dylib").write_bytes(b"PJRT_DYLIB")
    assert mb._find_or_build_pjrt_cdylib(tmp_path) == ("libtpu.dylib", b"PJRT_DYLIB")
    (pre_dir / "libtpu.dylib").unlink()

    def _populate_release(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        """Simulates `cargo build --release` populating `target/release/libtpu.so`.

        Args:
            *_args: Ignored `subprocess.run` positional arguments.
            **_kwargs: Ignored `subprocess.run` keyword arguments.

        Returns:
            Mock zero-exit `CompletedProcess`.
        """
        rel = tmp_path / "target" / "release"
        rel.mkdir(parents=True, exist_ok=True)
        (rel / "libtpu.so").write_bytes(b"BUILT_SO")
        return subprocess.CompletedProcess(["cargo"], 0, stdout="", stderr="")

    with (
        patch.object(mb, "_resolve_cargo_target_dir", return_value=tmp_path / "target"),
        patch.object(subprocess, "run", side_effect=_populate_release),
    ):
        assert mb._find_or_build_pjrt_cdylib(tmp_path) == ("libtpu.so", b"BUILT_SO")
    (tmp_path / "target" / "release" / "libtpu.so").unlink()

    with (
        patch.object(mb, "_resolve_cargo_target_dir", return_value=tmp_path / "target"),
        patch.object(
            subprocess,
            "run",
            return_value=subprocess.CompletedProcess(["cargo"], 0, stdout="", stderr=""),
        ),
        pytest.raises(RuntimeError, match="Failed to locate"),
    ):
        mb._find_or_build_pjrt_cdylib(tmp_path)

    # `inject_dropin_binaries_into_wheel`: missing RECORD/_C, injection, and idempotent re-run.
    no_rec_whl = tmp_path / "no_record.whl"
    with zipfile.ZipFile(no_rec_whl, "w") as zf:
        zf.writestr("libtpu/__init__.py", b"")
    with pytest.raises(RuntimeError, match="RECORD"):
        mb.inject_dropin_binaries_into_wheel(no_rec_whl, tmp_path)

    no_c_whl = tmp_path / "no_c.whl"
    with zipfile.ZipFile(no_c_whl, "w") as zf:
        zf.writestr("libtpu-0.0.52.dist-info/RECORD", b"")
    with pytest.raises(RuntimeError, match="_C"):
        mb.inject_dropin_binaries_into_wheel(no_c_whl, tmp_path)

    (pre_dir / "libtpu.dylib").write_bytes(b"PJRT_BIN")
    valid_whl = tmp_path / "libtpu-0.0.52-cp39-abi3-any.whl"
    with zipfile.ZipFile(valid_whl, "w") as zf:
        zf.writestr("libtpu/_C.abi3.so", b"PYO3_EXT")
        zf.writestr("libtpu-0.0.52.dist-info/RECORD", b"")

    mb.inject_dropin_binaries_into_wheel(valid_whl, tmp_path)
    # Second pass exercises already-present `sdk.so` and `libtpu.so` branches!
    mb.inject_dropin_binaries_into_wheel(valid_whl, tmp_path)
    with zipfile.ZipFile(valid_whl, "r") as zf:
        names = set(zf.namelist())
        assert {
            "libtpu/_C.abi3.so",
            "libtpu/sdk.so",
            "libtpu/libtpu.so",
            "libtpu/libtpu.dylib",
        } <= names

    (pre_dir / "libtpu.dylib").unlink()
    (pre_dir / "libtpu.so").write_bytes(b"PJRT_LINUX_SO")
    linux_whl = tmp_path / "libtpu-0.0.52-cp39-abi3-linux.whl"
    with zipfile.ZipFile(linux_whl, "w") as zf:
        zf.writestr("libtpu/_C.abi3.so", b"PYO3_EXT")
        zf.writestr("libtpu-0.0.52.dist-info/RECORD", b"")
    mb.inject_dropin_binaries_into_wheel(linux_whl, tmp_path)

    # PEP 517 wrapper pass-throughs.
    mock_maturin = MagicMock()
    mock_maturin.get_requires_for_build_wheel.return_value = ["a"]
    mock_maturin.get_requires_for_build_sdist.return_value = ["b"]
    mock_maturin.get_requires_for_build_editable.return_value = ["c"]
    mock_maturin.prepare_metadata_for_build_wheel.return_value = "meta.dist-info"
    mock_maturin.prepare_metadata_for_build_editable.return_value = "edit.dist-info"
    mock_maturin.build_wheel.return_value = valid_whl.name
    mock_maturin.build_editable.return_value = "editable.whl"
    mock_maturin.build_sdist.return_value = "src.tar.gz"
    with patch.object(mb, "maturin", mock_maturin):
        assert mb.get_requires_for_build_wheel() == ["a"]
        assert mb.get_requires_for_build_sdist() == ["b"]
        assert mb.get_requires_for_build_editable() == ["c"]
        assert mb.prepare_metadata_for_build_wheel(str(tmp_path)) == "meta.dist-info"
        assert mb.prepare_metadata_for_build_editable(str(tmp_path)) == "edit.dist-info"
        assert mb.build_wheel(str(tmp_path)) == valid_whl.name
        assert mb.build_editable(str(tmp_path)) == "editable.whl"
        assert mb.build_sdist(str(tmp_path)) == "src.tar.gz"

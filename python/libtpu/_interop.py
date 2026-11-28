"""Native `PyO3` (`abi3`) loader, `TpuDevice`, `TpuTopology`, `ml_dtypes`, and `DLPack` bridge."""

from __future__ import annotations

import ctypes
import importlib.machinery
import importlib.util
import sys
import types
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Protocol, Union, cast, runtime_checkable

import libtpu

if TYPE_CHECKING:
    from typing import TypeAlias

    import numpy as np
    import numpy.typing as npt
    from typing_extensions import CapsuleType

    PyCapsule: TypeAlias = CapsuleType
else:

    class PyCapsule(Protocol):
        """Opaque protocol representing a `CPython` `PyCapsule` handle."""

        _pycapsule_opaque: ClassVar[str]


@runtime_checkable
class SupportsDLPack(Protocol):
    """Protocol for DLPack v1.0 tensor producers implementing `__dlpack__`."""

    __dlpack__: Callable[..., PyCapsule]
    __dlpack_device__: Callable[[], tuple[int, int]]


class SupportsArrayInterface(Protocol):
    """Protocol for host array producers exposing `__array_interface__`."""

    __array_interface__: libtpu.ArrayInterfaceDict


NativeArg = Union[
    int,
    float,
    bool,
    str,
    bytes,
    bytearray,
    memoryview,
    Sequence[int],
    Sequence[Sequence[float]],
    PyCapsule,
    None,
]
NativeReturn = Union[
    int,
    str,
    bytes,
    memoryview,
    tuple[int, int],
    list[str],
    list[list[float]],
    tuple[bytes, Sequence[int], int, int, int, int, int],
    PyCapsule,
    None,
]


class _PjrtDeviceGetDescriptionArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_Device_GetDescription_Args` (`32` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("device", ctypes.c_void_p),
        ("device_description", ctypes.c_void_p),
    ]


class _PjrtDeviceIsAddressableArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_Device_IsAddressable_Args` (`32` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("device", ctypes.c_void_p),
        ("is_addressable", ctypes.c_bool),
        ("pad", ctypes.c_uint8 * 7),
    ]


class _PjrtDeviceLocalHardwareIdArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_Device_LocalHardwareId_Args` (`32` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("device", ctypes.c_void_p),
        ("local_hardware_id", ctypes.c_int32),
        ("pad", ctypes.c_int32),
    ]


class _PjrtDeviceDescriptionIdArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_DeviceDescription_Id_Args` (`32` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("device_description", ctypes.c_void_p),
        ("id", ctypes.c_int32),
        ("pad", ctypes.c_int32),
    ]


class _PjrtDeviceDescriptionStrArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_DeviceDescription_Kind_Args` (`40` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("device_description", ctypes.c_void_p),
        ("text_ptr", ctypes.c_void_p),
        ("text_size", ctypes.c_size_t),
    ]


class _PjrtDeviceDescriptionAttributesArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_DeviceDescription_Attributes_Args` (`40` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("device_description", ctypes.c_void_p),
        ("num_attributes", ctypes.c_size_t),
        ("attributes", ctypes.c_void_p),
    ]


class _PjrtNamedValueAttr(ctypes.Structure):
    """C ABI structure matching `PJRT_NamedValue` (`64` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("name", ctypes.c_void_p),
        ("name_size", ctypes.c_size_t),
        ("value_type", ctypes.c_int32),
        ("pad0", ctypes.c_int32),
        ("value_payload", ctypes.c_size_t),
        ("value_size", ctypes.c_size_t),
    ]


class _PjrtTopologyCreateArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_TopologyDescription_Create_Args` (`56` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("topology_name", ctypes.c_void_p),
        ("topology_name_size", ctypes.c_size_t),
        ("create_options", ctypes.c_void_p),
        ("num_options", ctypes.c_size_t),
        ("topology", ctypes.c_void_p),
    ]


class _PjrtTopologyDestroyArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_TopologyDescription_Destroy_Args` (`24` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("topology", ctypes.c_void_p),
    ]


class _PjrtTopologyStrArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_TopologyDescription_Platform{Name,Version}_Args` (`40` B)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("topology", ctypes.c_void_p),
        ("text_ptr", ctypes.c_void_p),
        ("text_size", ctypes.c_size_t),
    ]


class _PjrtTopologyGetDescriptionsArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_TopologyDescription_GetDeviceDescriptions_Args` (`40` B)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("topology", ctypes.c_void_p),
        ("descriptions", ctypes.c_void_p),
        ("num_descriptions", ctypes.c_size_t),
    ]


class _PjrtTopologySerializeArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_TopologyDescription_Serialize_Args` (`56` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("topology", ctypes.c_void_p),
        ("serialized_bytes", ctypes.c_void_p),
        ("serialized_bytes_size", ctypes.c_size_t),
        ("serialized_topology", ctypes.c_void_p),
        ("serialized_topology_deleter", ctypes.c_void_p),
    ]


class _PjrtTopologyDeserializeArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_TopologyDescription_Deserialize_Args` (`40` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("serialized_topology", ctypes.c_void_p),
        ("serialized_topology_size", ctypes.c_size_t),
        ("topology", ctypes.c_void_p),
    ]


class _PjrtClientTopologyArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_Client_TopologyDescription_Args` (`32` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("client", ctypes.c_void_p),
        ("topology", ctypes.c_void_p),
    ]


class _PjrtClientDevicesLocalArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_Client_Devices_Args` (`40` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("client", ctypes.c_void_p),
        ("devices", ctypes.c_void_p),
        ("num_devices", ctypes.c_size_t),
    ]


class _PjrtErrorCodeLocalArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_Error_GetCode_Args` (`32` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("error", ctypes.c_void_p),
        ("code", ctypes.c_int32),
        ("pad", ctypes.c_int32),
    ]


class _PjrtErrorMessageLocalArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_Error_Message_Args` (`40` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("error", ctypes.c_void_p),
        ("message", ctypes.c_void_p),
        ("message_size", ctypes.c_size_t),
    ]


class _PjrtErrorDestroyLocalArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_Error_Destroy_Args` (`24` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("error", ctypes.c_void_p),
    ]


_TOPO_DELETER_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
_NATIVE_EXT_CACHE: dict[str, types.ModuleType] = {}
_INTEROP_CDLL_CACHE: dict[str, ctypes.CDLL] = {}


def _load_interop_cdll() -> ctypes.CDLL:
    """Loads and caches the `libtpu` shared library via `libtpu.get_library_path()`.

    Returns:
        Configured `ctypes.CDLL` handle.
    """
    import libtpu

    path = libtpu.get_library_path()
    cached = _INTEROP_CDLL_CACHE.get(path)
    if cached is not None:
        return cached
    lib = ctypes.CDLL(path)
    for err_fn in ("PJRT_Error_GetCode", "PJRT_Error_Message", "PJRT_Error_Destroy"):
        fn_obj = getattr(lib, err_fn)
        fn_obj.argtypes = [ctypes.c_void_p]
        fn_obj.restype = ctypes.c_void_p if err_fn != "PJRT_Error_Destroy" else None
    _configure_topology_device_cdll(lib)
    _INTEROP_CDLL_CACHE[path] = lib
    return lib


def _check_interop_pjrt_error(lib: ctypes.CDLL, err_ptr: int | None) -> None:
    """Raises `libtpu.TpuError` and destroys `err_ptr` when `err_ptr` is non-null.

    Args:
        lib: Loaded `ctypes.CDLL` handle.
        err_ptr: Raw `PJRT_Error*` address.

    Raises:
        TpuError: If `err_ptr` is non-zero.
    """
    if err_ptr:
        import libtpu

        code_args = _PjrtErrorCodeLocalArgs(
            ctypes.sizeof(_PjrtErrorCodeLocalArgs), 0, err_ptr, 0, 0
        )
        lib.PJRT_Error_GetCode(ctypes.byref(code_args))
        msg_args = _PjrtErrorMessageLocalArgs(
            ctypes.sizeof(_PjrtErrorMessageLocalArgs), 0, err_ptr, 0, 0
        )
        lib.PJRT_Error_Message(ctypes.byref(msg_args))
        text = ctypes.string_at(int(msg_args.message or 0), int(msg_args.message_size)).decode(
            "utf-8", errors="replace"
        )
        del_args = _PjrtErrorDestroyLocalArgs(ctypes.sizeof(_PjrtErrorDestroyLocalArgs), 0, err_ptr)
        lib.PJRT_Error_Destroy(ctypes.byref(del_args))
        raise libtpu.TpuError(int(code_args.code), text)


def _configure_topology_device_cdll(lib: ctypes.CDLL) -> None:
    """Configures `argtypes` and `restype` for device and topology C-API functions on `lib`.

    Args:
        lib: Loaded `libtpu` `ctypes.CDLL` handle.
    """
    for fn_name in (
        "PJRT_Device_GetDescription",
        "PJRT_Device_IsAddressable",
        "PJRT_Device_LocalHardwareId",
        "PJRT_DeviceDescription_Id",
        "PJRT_DeviceDescription_ProcessIndex",
        "PJRT_DeviceDescription_Attributes",
        "PJRT_DeviceDescription_Kind",
        "PJRT_DeviceDescription_DebugString",
        "PJRT_DeviceDescription_ToString",
        "PJRT_TopologyDescription_Create",
        "PJRT_TopologyDescription_Destroy",
        "PJRT_TopologyDescription_PlatformName",
        "PJRT_TopologyDescription_PlatformVersion",
        "PJRT_TopologyDescription_GetDeviceDescriptions",
        "PJRT_TopologyDescription_Serialize",
        "PJRT_TopologyDescription_Deserialize",
        "PJRT_Client_TopologyDescription",
        "PJRT_Client_Devices",
        "PJRT_Client_AddressableDevices",
    ):
        fn_obj = getattr(lib, fn_name)
        fn_obj.argtypes = [ctypes.c_void_p]
        fn_obj.restype = ctypes.c_void_p


def _resolve_pyo3_extension_path(prefer_sdk: bool) -> Path:
    """Locates the compiled `PyO3` (`abi3`) native extension binary on disk.

    Args:
        prefer_sdk: If `True`, checks bundled `sdk.so`/`sdk.dylib` before `_C*.so`.

    Returns:
        Resolved `Path` to the native `PyO3` shared library.

    Raises:
        FileNotFoundError: If no compiled `PyO3` extension binary exists on disk.
    """
    pkg_dir = Path(__file__).resolve().parent
    repo_root = pkg_dir.parent.parent
    c_globs = sorted(pkg_dir.glob("_C*.so")) + sorted(pkg_dir.glob("_C*.pyd"))
    sdk_candidates = [pkg_dir / "sdk.so", pkg_dir / "sdk.dylib", pkg_dir / "sdk.pyd"]
    ordered_pkg = (sdk_candidates + c_globs) if prefer_sdk else (c_globs + sdk_candidates)
    for candidate in ordered_pkg:
        if candidate.is_file():
            return candidate
    cargo_existing: list[Path] = []
    for target_root in libtpu._cargo_target_roots(repo_root):
        for profile in ("debug", "release"):
            for name in ("liblibtpu_py.dylib", "liblibtpu_py.so", "libtpu_py.dll"):
                cand = target_root / profile / name
                if cand.is_file():
                    cargo_existing.append(cand)
    if cargo_existing:
        return max(cargo_existing, key=lambda p: p.stat().st_mtime_ns)
    raise FileNotFoundError("Compiled libtpu-py PyO3 extension library not found")


def _load_extension_symbol(
    module_qualname: str, symbol_mod_name: str, prefer_sdk: bool
) -> types.ModuleType:
    """Loads a PEP 489 `PyO3` extension module (`PyInit__C` or `PyInit_sdk`) and caches it.

    Args:
        module_qualname: Python `sys.modules` registration key.
        symbol_mod_name: Bare C initializer suffix (`"_C"` or `"sdk"`).
        prefer_sdk: Whether to prioritize `libtpu/sdk.so` over `libtpu/_C*.so`.

    Returns:
        Initialized `types.ModuleType` instance.

    Raises:
        ImportError: If `ExtensionFileLoader` fails to create a module spec.
    """
    cached = _NATIVE_EXT_CACHE.get(module_qualname)
    if cached is not None:
        return cached
    ext_path = _resolve_pyo3_extension_path(prefer_sdk=prefer_sdk)
    loader = importlib.machinery.ExtensionFileLoader(symbol_mod_name, str(ext_path))
    spec = importlib.util.spec_from_loader(symbol_mod_name, loader)
    if spec is None:
        raise ImportError(f"Unable to build module spec for {symbol_mod_name} at {ext_path}")
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    cdll = _load_interop_cdll()
    cdll.libtpu_rs_runtime_sync_vtable.argtypes = []
    cdll.libtpu_rs_runtime_sync_vtable.restype = ctypes.c_void_p
    sync_vt_addr = int(cdll.libtpu_rs_runtime_sync_vtable() or 0)
    mod.bind_c_api_runtime_sync(sync_vt_addr)
    sys.modules[module_qualname] = mod
    _NATIVE_EXT_CACHE[module_qualname] = mod
    return mod


def load_native_c_module() -> types.ModuleType:
    """Loads and returns the native `PyO3` (`abi3`) `libtpu._C` extension module (`PyInit__C`).

    Returns:
        Loaded `libtpu._C` native extension module.
    """
    return _load_extension_symbol("libtpu._C", "_C", prefer_sdk=False)


def load_native_sdk_module() -> types.ModuleType:
    """Loads and returns the native `PyO3` (`abi3`) `libtpu/sdk.so` extension module (`PyInit_sdk`).

    Returns:
        Loaded `sdk` native extension module.
    """
    return _load_extension_symbol("libtpu._sdk_native", "sdk", prefer_sdk=True)


def call_native_c(fn_name: str, *args: NativeArg) -> NativeReturn:
    """Invokes `fn_name` on `libtpu._C`, translating native exceptions to `libtpu.TpuError`.

    Args:
        fn_name: Exported function name on `libtpu._C`.
        *args: Positional arguments forwarded to `fn_name`.

    Returns:
        Return value from the native `PyO3` function.

    Raises:
        TpuError: If the native `PyO3` function raises `libtpu.TpuError`.
    """
    mod = load_native_c_module()
    native_err_cls = mod.TpuError
    try:
        return cast(NativeReturn, getattr(mod, fn_name)(*args))
    except native_err_cls as exc:
        code = int(getattr(exc, "code", 2))
        msg = str(getattr(exc, "message", str(exc)))
        raise libtpu.TpuError(code, msg) from exc


class TpuDevice:
    """Strongly-typed Python wrapper over `PJRT_Device*` and `PJRT_DeviceDescription*`."""

    def __init__(self, lib: ctypes.CDLL, handle: ctypes.c_void_p) -> None:
        """Initializes a `TpuDevice` wrapper.

        Args:
            lib: Loaded `ctypes.CDLL` handle.
            handle: Non-null `PJRT_Device*` pointer owned by a `TpuClient`.
        """
        _configure_topology_device_cdll(lib)
        self._lib: ctypes.CDLL = lib
        self._handle: ctypes.c_void_p = handle
        desc_args = _PjrtDeviceGetDescriptionArgs(
            ctypes.sizeof(_PjrtDeviceGetDescriptionArgs), 0, handle.value or 0, 0
        )
        lib.PJRT_Device_GetDescription(ctypes.byref(desc_args))
        self._desc_handle: int = int(desc_args.device_description or 0)

    def id(self) -> int:
        """Returns the global logical device ID via `PJRT_DeviceDescription_Id`.

        Returns:
            Global logical device ordinal.
        """
        args = _PjrtDeviceDescriptionIdArgs(
            ctypes.sizeof(_PjrtDeviceDescriptionIdArgs), 0, self._desc_handle, 0, 0
        )
        self._lib.PJRT_DeviceDescription_Id(ctypes.byref(args))
        return int(args.id)

    def process_index(self) -> int:
        """Returns the host process index of this device (`PJRT_DeviceDescription_ProcessIndex`).

        Returns:
            Zero-based host process index.
        """
        args = _PjrtDeviceDescriptionIdArgs(
            ctypes.sizeof(_PjrtDeviceDescriptionIdArgs), 0, self._desc_handle, 0, 0
        )
        self._lib.PJRT_DeviceDescription_ProcessIndex(ctypes.byref(args))
        return int(args.id)

    def device_kind(self) -> str:
        """Returns the hardware SKU kind string via `PJRT_DeviceDescription_Kind`.

        Returns:
            Hardware SKU kind string (e.g., `"TPU v5 lite"`).
        """
        args = _PjrtDeviceDescriptionStrArgs(
            ctypes.sizeof(_PjrtDeviceDescriptionStrArgs), 0, self._desc_handle, 0, 0
        )
        self._lib.PJRT_DeviceDescription_Kind(ctypes.byref(args))
        return ctypes.string_at(int(args.text_ptr or 0), int(args.text_size)).decode("utf-8")

    def debug_string(self) -> str:
        """Returns the diagnostic debug string via `PJRT_DeviceDescription_DebugString`.

        Returns:
            Human-readable debug string for this device.
        """
        args = _PjrtDeviceDescriptionStrArgs(
            ctypes.sizeof(_PjrtDeviceDescriptionStrArgs), 0, self._desc_handle, 0, 0
        )
        self._lib.PJRT_DeviceDescription_DebugString(ctypes.byref(args))
        return ctypes.string_at(int(args.text_ptr or 0), int(args.text_size)).decode("utf-8")

    def to_string(self) -> str:
        """Returns the canonical display string via `PJRT_DeviceDescription_ToString`.

        Returns:
            Display string representation (e.g., `"TpuDevice(id=0, ...)"`).
        """
        args = _PjrtDeviceDescriptionStrArgs(
            ctypes.sizeof(_PjrtDeviceDescriptionStrArgs), 0, self._desc_handle, 0, 0
        )
        self._lib.PJRT_DeviceDescription_ToString(ctypes.byref(args))
        return ctypes.string_at(int(args.text_ptr or 0), int(args.text_size)).decode("utf-8")

    def is_addressable(self) -> bool:
        """Returns whether this device is locally addressable (`PJRT_Device_IsAddressable`).

        Returns:
            `True` if locally addressable.
        """
        args = _PjrtDeviceIsAddressableArgs(
            ctypes.sizeof(_PjrtDeviceIsAddressableArgs),
            0,
            self._handle.value or 0,
            False,
            (ctypes.c_uint8 * 7)(),
        )
        self._lib.PJRT_Device_IsAddressable(ctypes.byref(args))
        return bool(args.is_addressable)

    def local_hardware_id(self) -> int:
        """Returns the local PCIe/chip hardware ordinal via `PJRT_Device_LocalHardwareId`.

        Returns:
            Local hardware ID integer.
        """
        args = _PjrtDeviceLocalHardwareIdArgs(
            ctypes.sizeof(_PjrtDeviceLocalHardwareIdArgs), 0, self._handle.value or 0, 0, 0
        )
        self._lib.PJRT_Device_LocalHardwareId(ctypes.byref(args))
        return int(args.local_hardware_id)

    def coords(self) -> tuple[int, int, int]:
        """Returns the 3D torus `(x, y, z)` coordinates from `PJRT_DeviceDescription_Attributes`.

        Returns:
            Tuple `(x, y, z)` of integer chip coordinates.
        """
        args = _PjrtDeviceDescriptionAttributesArgs(
            ctypes.sizeof(_PjrtDeviceDescriptionAttributesArgs), 0, self._desc_handle, 0, 0
        )
        self._lib.PJRT_DeviceDescription_Attributes(ctypes.byref(args))
        base = int(args.attributes or 0)
        stride = ctypes.sizeof(_PjrtNamedValueAttr)
        for idx in range(int(args.num_attributes)):
            nv = _PjrtNamedValueAttr.from_address(base + idx * stride)
            name = ctypes.string_at(int(nv.name or 0), int(nv.name_size)).decode("utf-8")
            if name == "coords" and int(nv.value_size) >= 3:
                arr = (ctypes.c_int64 * 3).from_address(int(nv.value_payload))
                return (int(arr[0]), int(arr[1]), int(arr[2]))
        return (0, 0, 0)

    def core_on_chip(self) -> int:
        """Returns the `TensorCore` ordinal on the ASIC from `PJRT_DeviceDescription_Attributes`.

        Returns:
            Zero-based `TensorCore` ordinal on the chip.
        """
        args = _PjrtDeviceDescriptionAttributesArgs(
            ctypes.sizeof(_PjrtDeviceDescriptionAttributesArgs), 0, self._desc_handle, 0, 0
        )
        self._lib.PJRT_DeviceDescription_Attributes(ctypes.byref(args))
        base = int(args.attributes or 0)
        stride = ctypes.sizeof(_PjrtNamedValueAttr)
        for idx in range(int(args.num_attributes)):
            nv = _PjrtNamedValueAttr.from_address(base + idx * stride)
            name = ctypes.string_at(int(nv.name or 0), int(nv.name_size)).decode("utf-8")
            if name == "core_on_chip":
                raw = int(nv.value_payload).to_bytes(8, sys.byteorder, signed=False)
                return int.from_bytes(raw, sys.byteorder, signed=True)
        return 0

    def __repr__(self) -> str:
        """Returns the developer string representation of this `TpuDevice`.

        Returns:
            Canonical `to_string()` description.
        """
        return self.to_string()


class TpuTopology:
    """Strongly-typed RAII wrapper over `PJRT_TopologyDescription*`."""

    def __init__(self, lib: ctypes.CDLL, handle: ctypes.c_void_p, *, owned: bool) -> None:
        """Initializes a `TpuTopology` wrapper.

        Args:
            lib: Loaded `ctypes.CDLL` handle.
            handle: Non-null `PJRT_TopologyDescription*` pointer.
            owned: Whether `close()` should invoke `PJRT_TopologyDescription_Destroy`.
        """
        _configure_topology_device_cdll(lib)
        self._lib: ctypes.CDLL = lib
        self._handle: ctypes.c_void_p = handle
        self._owned: bool = owned

    @classmethod
    def create(cls, topology_name: str = "v5e-2x2") -> TpuTopology:
        """Creates an standalone `TpuTopology` via `PJRT_TopologyDescription_Create`.

        Args:
            topology_name: Canonical TPU topology string (e.g., `"v5e-2x2"`, `"v6e-2x2"`).

        Returns:
            Newly allocated owning `TpuTopology` instance.
        """
        lib = _load_interop_cdll()
        raw_name = topology_name.encode("utf-8")
        c_name = ctypes.create_string_buffer(raw_name, len(raw_name))
        args = _PjrtTopologyCreateArgs(
            ctypes.sizeof(_PjrtTopologyCreateArgs),
            0,
            ctypes.addressof(c_name),
            len(raw_name),
            0,
            0,
            0,
        )
        err = lib.PJRT_TopologyDescription_Create(ctypes.byref(args))
        _check_interop_pjrt_error(lib, err)
        return cls(lib, ctypes.c_void_p(int(args.topology or 0)), owned=True)

    @classmethod
    def deserialize(cls, serialized: bytes) -> TpuTopology:
        """Deserializes a `TpuTopology` from wire bytes via `PJRT_TopologyDescription_Deserialize`.

        Args:
            serialized: Wire bytes produced by `TpuTopology.serialize()`.

        Returns:
            Newly allocated owning `TpuTopology` instance.
        """
        lib = _load_interop_cdll()
        c_buf = ctypes.create_string_buffer(serialized, len(serialized))
        args = _PjrtTopologyDeserializeArgs(
            ctypes.sizeof(_PjrtTopologyDeserializeArgs),
            0,
            ctypes.addressof(c_buf),
            len(serialized),
            0,
        )
        err = lib.PJRT_TopologyDescription_Deserialize(ctypes.byref(args))
        _check_interop_pjrt_error(lib, err)
        return cls(lib, ctypes.c_void_p(int(args.topology or 0)), owned=True)

    def platform_name(self) -> str:
        """Returns the PJRT platform name (`"tpu"`) via `PJRT_TopologyDescription_PlatformName`.

        Returns:
            Platform name string.
        """
        args = _PjrtTopologyStrArgs(
            ctypes.sizeof(_PjrtTopologyStrArgs), 0, self._handle.value or 0, 0, 0
        )
        self._lib.PJRT_TopologyDescription_PlatformName(ctypes.byref(args))
        return ctypes.string_at(int(args.text_ptr or 0), int(args.text_size)).decode("utf-8")

    def platform_version(self) -> str:
        """Returns the PJRT platform version string via `PJRT_TopologyDescription_PlatformVersion`.

        Returns:
            Platform version string.
        """
        args = _PjrtTopologyStrArgs(
            ctypes.sizeof(_PjrtTopologyStrArgs), 0, self._handle.value or 0, 0, 0
        )
        self._lib.PJRT_TopologyDescription_PlatformVersion(ctypes.byref(args))
        return ctypes.string_at(int(args.text_ptr or 0), int(args.text_size)).decode("utf-8")

    def device_count(self) -> int:
        """Returns the number of device descriptions in this topology.

        Returns:
            Total logical TPU device count in this topology.
        """
        args = _PjrtTopologyGetDescriptionsArgs(
            ctypes.sizeof(_PjrtTopologyGetDescriptionsArgs), 0, self._handle.value or 0, 0, 0
        )
        self._lib.PJRT_TopologyDescription_GetDeviceDescriptions(ctypes.byref(args))
        return int(args.num_descriptions)

    def serialize(self) -> bytes:
        """Serializes this topology via `PJRT_TopologyDescription_Serialize`.

        Returns:
            Wire-encoded `PjRtTopologyDescriptionProto` bytes.
        """
        args = _PjrtTopologySerializeArgs(
            ctypes.sizeof(_PjrtTopologySerializeArgs), 0, self._handle.value or 0, 0, 0, 0, 0
        )
        self._lib.PJRT_TopologyDescription_Serialize(ctypes.byref(args))
        payload = ctypes.string_at(int(args.serialized_bytes or 0), int(args.serialized_bytes_size))
        deleter = _TOPO_DELETER_TYPE(int(args.serialized_topology_deleter or 0))
        deleter(args.serialized_topology)
        return payload

    def close(self) -> None:
        """Destroys the underlying `PJRT_TopologyDescription*` when owned."""
        if self._owned and self._handle.value:
            args = _PjrtTopologyDestroyArgs(
                ctypes.sizeof(_PjrtTopologyDestroyArgs), 0, self._handle.value
            )
            self._lib.PJRT_TopologyDescription_Destroy(ctypes.byref(args))
        self._handle = ctypes.c_void_p(0)

    def __enter__(self) -> TpuTopology:
        """Enters a context manager block returning `self`.

        Returns:
            `self`.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Closes this `TpuTopology` upon exiting a context manager block.

        Args:
            exc_type: Exception class if an exception was raised, else `None`.
            exc_val: Exception instance if an exception was raised, else `None`.
            exc_tb: Traceback if an exception was raised, else `None`.
        """
        _ = (exc_type, exc_val, exc_tb)
        self.close()


def _resolve_numpy_dtype(dt: libtpu.DataType) -> np.dtype[np.generic]:
    """Maps a `libtpu.DataType` to its exact `numpy.dtype` (including `ml_dtypes` extensions).

    Args:
        dt: `libtpu.DataType` discriminant.

    Returns:
        Corresponding `numpy.dtype` object.

    Raises:
        TpuError: If `dt` is `INVALID`, `TOKEN`, or `TUPLE`.
    """
    import ml_dtypes
    import numpy as np

    table: dict[int, np.dtype[np.generic]] = {
        1: np.dtype(np.bool_),
        2: np.dtype(np.int8),
        3: np.dtype(np.int16),
        4: np.dtype(np.int32),
        5: np.dtype(np.int64),
        6: np.dtype(np.uint8),
        7: np.dtype(np.uint16),
        8: np.dtype(np.uint32),
        9: np.dtype(np.uint64),
        10: np.dtype(np.float16),
        11: np.dtype(np.float32),
        12: np.dtype(np.float64),
        13: np.dtype(ml_dtypes.bfloat16),
        14: np.dtype(np.complex64),
        15: np.dtype(np.complex128),
        16: np.dtype(ml_dtypes.float8_e5m2),
        17: np.dtype(ml_dtypes.float8_e4m3fn),
        18: np.dtype(ml_dtypes.float8_e4m3b11fnuz),
        19: np.dtype(ml_dtypes.float8_e5m2fnuz),
        20: np.dtype(ml_dtypes.float8_e4m3fnuz),
        21: np.dtype(ml_dtypes.int4),
        22: np.dtype(ml_dtypes.uint4),
        24: np.dtype(ml_dtypes.int2),
        25: np.dtype(ml_dtypes.uint2),
        26: np.dtype(ml_dtypes.float4_e2m1fn),
        27: np.dtype(ml_dtypes.float8_e8m0fnu),
    }
    np_dt = table.get(int(dt.value))
    if np_dt is None:
        raise libtpu.TpuError(3, f"DataType {dt.name} cannot be represented as a NumPy ndarray")
    return np_dt


def _resolve_libtpu_dtype_from_numpy(np_dtype: npt.DTypeLike) -> libtpu.DataType:
    """Maps a `numpy.dtype` (including `ml_dtypes` `bfloat16`/`float8`/`int4`) to `DataType`.

    Args:
        np_dtype: `numpy.dtype` instance.

    Returns:
        Matching `libtpu.DataType` enum variant.

    Raises:
        TpuError: If `np_dtype` is unsupported.
    """
    import ml_dtypes
    import numpy as np

    reverse_map: list[tuple[np.dtype[np.generic], libtpu.DataType]] = [
        (np.dtype(np.bool_), libtpu.DataType.PRED),
        (np.dtype(np.int8), libtpu.DataType.S8),
        (np.dtype(np.int16), libtpu.DataType.S16),
        (np.dtype(np.int32), libtpu.DataType.S32),
        (np.dtype(np.int64), libtpu.DataType.S64),
        (np.dtype(np.uint8), libtpu.DataType.U8),
        (np.dtype(np.uint16), libtpu.DataType.U16),
        (np.dtype(np.uint32), libtpu.DataType.U32),
        (np.dtype(np.uint64), libtpu.DataType.U64),
        (np.dtype(np.float16), libtpu.DataType.F16),
        (np.dtype(np.float32), libtpu.DataType.F32),
        (np.dtype(np.float64), libtpu.DataType.F64),
        (np.dtype(ml_dtypes.bfloat16), libtpu.DataType.BF16),
        (np.dtype(np.complex64), libtpu.DataType.C64),
        (np.dtype(np.complex128), libtpu.DataType.C128),
        (np.dtype(ml_dtypes.float8_e5m2), libtpu.DataType.F8E5M2),
        (np.dtype(ml_dtypes.float8_e4m3fn), libtpu.DataType.F8E4M3FN),
        (np.dtype(ml_dtypes.float8_e4m3b11fnuz), libtpu.DataType.F8E4M3B11FNUZ),
        (np.dtype(ml_dtypes.float8_e5m2fnuz), libtpu.DataType.F8E5M2FNUZ),
        (np.dtype(ml_dtypes.float8_e4m3fnuz), libtpu.DataType.F8E4M3FNUZ),
        (np.dtype(ml_dtypes.int4), libtpu.DataType.S4),
        (np.dtype(ml_dtypes.uint4), libtpu.DataType.U4),
        (np.dtype(ml_dtypes.int2), libtpu.DataType.S2),
        (np.dtype(ml_dtypes.uint2), libtpu.DataType.U2),
        (np.dtype(ml_dtypes.float4_e2m1fn), libtpu.DataType.F4E2M1FN),
        (np.dtype(ml_dtypes.float8_e8m0fnu), libtpu.DataType.F8E8M0FNU),
    ]
    target = np.dtype(np_dtype)
    for candidate_dt, tpu_dt in reverse_map:
        if target == candidate_dt:
            return tpu_dt
    raise libtpu.TpuError(3, f"Unsupported NumPy dtype: {target}")


def pack_subbyte_elements(element_bytes: bytes, bits: int) -> bytes:
    """Packs 1-byte-per-element `ml_dtypes` sub-byte values (`2`/`4`-bit) into TPU wire bytes.

    Args:
        element_bytes: Per-element low-bit bytes from `ndarray.tobytes()`.
        bits: Element bit width (`2` or `4`).

    Returns:
        Packed sub-byte wire payload.
    """
    elems_per_byte = 8 // bits
    mask = (1 << bits) - 1
    out = bytearray((len(element_bytes) + elems_per_byte - 1) // elems_per_byte)
    for idx, raw_b in enumerate(element_bytes):
        slot = idx // elems_per_byte
        shift = (idx % elems_per_byte) * bits
        out[slot] |= (raw_b & mask) << shift
    return bytes(out)


def unpack_subbyte_elements(
    packed_bytes: bytes, num_elements: int, bits: int, signed: bool
) -> bytes:
    """Unpacks TPU sub-byte wire bytes (`2`/`4`-bit) into 1-byte-per-element `ml_dtypes` bytes.

    Args:
        packed_bytes: Packed sub-byte HBM wire payload.
        num_elements: Logical scalar element count.
        bits: Element bit width (`2` or `4`).
        signed: Whether to sign-extend two's-complement values to 8 bits for `ml_dtypes`.

    Returns:
        Unpacked 1-byte-per-element buffer suitable for `np.frombuffer(..., dtype=ml_dtypes.int4)`.
    """
    elems_per_byte = 8 // bits
    mask = (1 << bits) - 1
    sign_bit = 1 << (bits - 1)
    out = bytearray(num_elements)
    for idx in range(num_elements):
        raw_byte = packed_bytes[idx // elems_per_byte]
        shift = (idx % elems_per_byte) * bits
        val = (raw_byte >> shift) & mask
        if signed and (val & sign_bit):
            val = (val - (1 << bits)) & 0xFF
        out[idx] = val
    return bytes(out)


def buffer_to_numpy(
    buf: libtpu.TpuBuffer,
    requested_dtype: npt.DTypeLike | None = None,
) -> npt.NDArray[
    np.bool_
    | np.int8
    | np.int16
    | np.int32
    | np.int64
    | np.uint8
    | np.uint16
    | np.uint32
    | np.uint64
    | np.float16
    | np.float32
    | np.float64
    | np.complex64
    | np.complex128
]:
    """Materializes a `TpuBuffer` as a `numpy.ndarray` (including `ml_dtypes` types).

    Args:
        buf: Source `TpuBuffer`.
        requested_dtype: Optional cast target `numpy.dtype`.

    Returns:
        Populated `numpy.ndarray` matching `buf.shape()`.
    """
    import numpy as np

    dt = buf.dtype()
    np_dt = _resolve_numpy_dtype(dt)
    shape = buf.shape()
    num_elements = 1
    for dim in shape:
        num_elements *= dim
    raw = buf.to_bytes()
    bits = dt.bit_size()
    if bits in (2, 4):
        signed = dt in (libtpu.DataType.S4, libtpu.DataType.S2)
        raw = unpack_subbyte_elements(raw, num_elements, bits, signed=signed)
    arr = np.frombuffer(raw, dtype=np_dt, count=num_elements).reshape(shape).copy()
    if requested_dtype is not None:
        arr = arr.astype(requested_dtype)
    return cast(
        "npt.NDArray[np.bool_ | np.int8 | np.int16 | np.int32 | np.int64 | np.uint8"
        " | np.uint16 | np.uint32 | np.uint64 | np.float16 | np.float32 | np.float64"
        " | np.complex64 | np.complex128]",
        arr,
    )


def client_buffer_from_numpy(
    client: libtpu.TpuClient,
    device_id: int,
    array: npt.ArrayLike | SupportsArrayInterface,
) -> libtpu.TpuBuffer:
    """Transfers a `numpy.ndarray` (including `ml_dtypes`) onto `device_id`.

    Args:
        client: Active `TpuClient`.
        device_id: Destination logical TPU device ID.
        array: Input `numpy.ndarray` or host array producer.

    Returns:
        Allocated `TpuBuffer` on `device_id`.
    """
    import numpy as np

    arr = np.ascontiguousarray(cast("npt.ArrayLike", array))
    dt = _resolve_libtpu_dtype_from_numpy(arr.dtype)
    dims: list[int] = [int(d) for d in arr.shape]
    raw = arr.tobytes()
    bits = dt.bit_size()
    if bits in (2, 4):
        raw = pack_subbyte_elements(raw, bits)
    return client.buffer_from_bytes(device_id, dt, raw, dims)


def _map_dlpack_code_bits_to_dtype(code: int, bits: int) -> libtpu.DataType:
    """Maps a `DLPack` `(code, bits)` pair to `libtpu.DataType`.

    Args:
        code: `DLDataTypeCode` (`0` int, `1` uint, `2` float, `4` bfloat, `5` complex, `6` bool).
        bits: Scalar bit width.

    Returns:
        Matching `libtpu.DataType` enum value.

    Raises:
        TpuError: If `(code, bits)` does not correspond to a supported TPU `DataType`.
    """
    mapping: dict[tuple[int, int], libtpu.DataType] = {
        (6, 8): libtpu.DataType.PRED,
        (0, 4): libtpu.DataType.S4,
        (0, 8): libtpu.DataType.S8,
        (0, 16): libtpu.DataType.S16,
        (0, 32): libtpu.DataType.S32,
        (0, 64): libtpu.DataType.S64,
        (1, 4): libtpu.DataType.U4,
        (1, 8): libtpu.DataType.U8,
        (1, 16): libtpu.DataType.U16,
        (1, 32): libtpu.DataType.U32,
        (1, 64): libtpu.DataType.U64,
        (2, 8): libtpu.DataType.F8E4M3FN,
        (2, 16): libtpu.DataType.F16,
        (2, 32): libtpu.DataType.F32,
        (2, 64): libtpu.DataType.F64,
        (4, 16): libtpu.DataType.BF16,
        (5, 64): libtpu.DataType.C64,
        (5, 128): libtpu.DataType.C128,
    }
    resolved = mapping.get((code, bits))
    if resolved is None:
        raise libtpu.TpuError(3, f"Unsupported DLPack dtype (code={code}, bits={bits})")
    return resolved


def from_dlpack(
    ext_tensor: SupportsDLPack | npt.NDArray[np.generic] | PyCapsule,
    *,
    client: libtpu.TpuClient | None = None,
    device_id: int = 0,
) -> libtpu.TpuBuffer:
    """Imports an external tensor or `"dltensor"` `PyCapsule` into a `TpuBuffer` via `DLPack` v1.0.

    Args:
        ext_tensor: Producer implementing `__dlpack__()` or a raw `"dltensor"` `PyCapsule`.
        client: Optional existing `TpuClient` (a new client is created when `None`).
        device_id: Destination logical TPU device ID (`0` by default).

    Returns:
        `TpuBuffer` containing the imported tensor on `device_id`.

    Raises:
        TpuError: If `ext_tensor` is invalid, already consumed, or has an unsupported dtype.
    """
    dlpack_fn = getattr(ext_tensor, "__dlpack__", None)
    capsule = cast(PyCapsule, dlpack_fn() if callable(dlpack_fn) else ext_tensor)
    consumed = cast(
        tuple[bytes, Sequence[int], int, int, int, int, int],
        call_native_c("consume_dlpack", capsule),
    )
    raw_bytes, shape_list, code, bits, _lanes, _dev_type, _dev_id = consumed
    dt = _map_dlpack_code_bits_to_dtype(int(code), int(bits))
    active_client = client if client is not None else libtpu.TpuClient()
    return active_client.buffer_from_bytes(device_id, dt, bytes(raw_bytes), list(shape_list))


def query_client_devices(
    lib: ctypes.CDLL, client_handle: int, addressable_only: bool
) -> list[TpuDevice]:
    """Queries all or addressable `TpuDevice` instances from `client_handle`.

    Args:
        lib: Loaded `ctypes.CDLL` handle.
        client_handle: Raw `PJRT_Client*` address.
        addressable_only: Whether to query `PJRT_Client_AddressableDevices` only.

    Returns:
        List of `TpuDevice` wrappers.
    """
    _configure_topology_device_cdll(lib)
    dev_args = _PjrtClientDevicesLocalArgs(
        ctypes.sizeof(_PjrtClientDevicesLocalArgs), 0, client_handle, 0, 0
    )
    if addressable_only:
        lib.PJRT_Client_AddressableDevices(ctypes.byref(dev_args))
    else:
        lib.PJRT_Client_Devices(ctypes.byref(dev_args))
    count = int(dev_args.num_devices)
    ptrs = (ctypes.c_void_p * count).from_address(int(dev_args.devices or 0))
    return [TpuDevice(lib, ctypes.c_void_p(int(ptrs[i] or 0))) for i in range(count)]


def query_client_topology(lib: ctypes.CDLL, client_handle: int) -> TpuTopology:
    """Returns a non-owning `TpuTopology` view for `client_handle`.

    Args:
        lib: Loaded `ctypes.CDLL` handle.
        client_handle: Raw `PJRT_Client*` address.

    Returns:
        Non-owning `TpuTopology` wrapper.
    """
    _configure_topology_device_cdll(lib)
    args = _PjrtClientTopologyArgs(ctypes.sizeof(_PjrtClientTopologyArgs), 0, client_handle, 0)
    err = lib.PJRT_Client_TopologyDescription(ctypes.byref(args))
    _check_interop_pjrt_error(lib, err)
    return TpuTopology(lib, ctypes.c_void_p(int(args.topology or 0)), owned=False)

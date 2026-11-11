"""Drop-in Python package and ctypes FFI bindings for `libtpu-rs`.

Provides `get_library_path()`, `configure_library_path()`, `set_init_args()`,
`get_init_args()`, and strongly-typed Python wrapper classes (`TpuClient`,
`TpuBuffer`, `TpuExecutable`, `TpuProfiler`, `TpuGeneration`, `TpuError`)
driving `GetPjrtApi()` and `TpuProfiler_*()` directly via `PJRT_*_Args` structures.
"""

from __future__ import annotations

import ctypes
import enum
import os
import struct
from collections.abc import Sequence
from pathlib import Path

__version__: str = "0.0.52"


class TpuError(RuntimeError):
    """Exception raised when a TPU runtime or C FFI operation fails.

    Attributes:
        code: Numeric PJRT/TPU status code (`PjrtErrorCode`).
        details: Human-readable error message from `libtpu-rs`.
    """

    def __init__(self, code: int, details: str) -> None:
        """Initializes a `TpuError` instance.

        Args:
            code: Numeric PJRT error status code.
            details: Error message description.
        """
        super().__init__(f"[TPU Error {code}] {details}")
        self.code: int = code
        self.details: str = details


class TpuGeneration(enum.IntEnum):
    """Enumeration of Google Cloud TPU hardware generations."""

    V2 = 2
    V3 = 3
    V4 = 4
    V5E = 5
    V5P = 6
    V6E = 7
    V7X = 8
    V4LITE = 9


_DATA_TYPE_BIT_SIZES: dict[int, int] = {
    0: 0,
    1: 8,
    2: 8,
    3: 16,
    4: 32,
    5: 64,
    6: 8,
    7: 16,
    8: 32,
    9: 64,
    10: 16,
    11: 32,
    12: 64,
    13: 16,
    14: 64,
    15: 128,
    16: 8,
    17: 8,
    18: 8,
    19: 8,
    20: 8,
    21: 4,
    22: 4,
    23: 0,
    24: 2,
    25: 2,
    26: 4,
    27: 8,
    28: 0,
}


_DATA_TYPE_NUMPY_TYPESTRS: dict[int, str] = {
    0: "|V0",
    1: "|b1",
    2: "|i1",
    3: "<i2",
    4: "<i4",
    5: "<i8",
    6: "|u1",
    7: "<u2",
    8: "<u4",
    9: "<u8",
    10: "<f2",
    11: "<f4",
    12: "<f8",
    13: "|V2",
    14: "<c8",
    15: "<c16",
    16: "|V1",
    17: "|V1",
    18: "|V1",
    19: "|V1",
    20: "|V1",
    21: "|V1",
    22: "|V1",
    23: "|V0",
    24: "|V1",
    25: "|V1",
    26: "|V1",
    27: "|V1",
    28: "|V0",
}


class DataType(enum.IntEnum):
    """Enumeration of all 29 PJRT tensor element types (`PJRT_Buffer_Type`)."""

    INVALID = 0
    PRED = 1
    S8 = 2
    S16 = 3
    S32 = 4
    S64 = 5
    U8 = 6
    U16 = 7
    U32 = 8
    U64 = 9
    F16 = 10
    F32 = 11
    F64 = 12
    BF16 = 13
    C64 = 14
    C128 = 15
    F8E5M2 = 16
    F8E4M3FN = 17
    F8E4M3B11FNUZ = 18
    F8E5M2FNUZ = 19
    F8E4M3FNUZ = 20
    S4 = 21
    U4 = 22
    TOKEN = 23
    S2 = 24
    U2 = 25
    F4E2M1FN = 26
    F8E8M0FNU = 27
    TUPLE = 28

    def bit_size(self) -> int:
        """Returns the bit width of a single element of this `DataType`.

        Returns:
            Number of bits per scalar element (`0` for `INVALID`, `TOKEN`, `TUPLE`).
        """
        return _DATA_TYPE_BIT_SIZES[int(self.value)]

    def byte_size_for_elements(self, num_elements: int) -> int:
        """Computes the exact host byte length for `num_elements` of this `DataType`.

        Args:
            num_elements: Number of scalar elements in the tensor.

        Returns:
            Number of bytes required to store `num_elements` (sub-byte types rounded up).
        """
        total_bits = num_elements * self.bit_size()
        return (total_bits + 7) // 8

    def numpy_typestr(self) -> str:
        """Returns the canonical NumPy `__array_interface__` `typestr` for this `DataType`.

        Returns:
            NumPy `typestr` string (e.g., `"<f4"`, `"<i8"`, `"|b1"`, `"|V2"` for `BF16`).
        """
        return _DATA_TYPE_NUMPY_TYPESTRS[int(self.value)]

    def dlpack_type_code(self) -> int:
        """Returns the DLPack `DLDataTypeCode` integer for this `DataType`.

        Returns:
            DLPack type code (`0` int, `1` uint, `2` float, `4` bfloat, `5` complex, `6` bool).
        """
        val = int(self.value)
        if val == 1:
            return 6
        if val in (2, 3, 4, 5, 21, 24):
            return 0
        if val in (6, 7, 8, 9, 22, 25):
            return 1
        if val == 13:
            return 4
        if val in (14, 15):
            return 5
        return 2


class _DLDevice(ctypes.Structure):
    """DLPack v1.0 `DLDevice` C ABI structure (`8` bytes)."""

    _fields_ = [
        ("device_type", ctypes.c_int32),
        ("device_id", ctypes.c_int32),
    ]


class _DLDataType(ctypes.Structure):
    """DLPack v1.0 `DLDataType` C ABI structure (`4` bytes)."""

    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLTensor(ctypes.Structure):
    """DLPack v1.0 `DLTensor` C ABI structure (`48` bytes)."""

    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    """DLPack v1.0 `DLManagedTensor` C ABI structure (`64` bytes)."""

    _fields_ = [
        ("dl_tensor", _DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
    ]


def get_library_path() -> str:
    """Resolves the filesystem path to the compiled `libtpu` shared library.

    Returns:
        Absolute path string to `libtpu.dylib`, `libtpu.so`, or `tpu.dll`.
    """
    pkg_dir = Path(__file__).resolve().parent
    repo_root = pkg_dir.parent.parent
    candidates = [
        pkg_dir / "libtpu.dylib",
        pkg_dir / "libtpu.so",
        pkg_dir / "tpu.dll",
        repo_root / "target" / "debug" / "libtpu.dylib",
        repo_root / "target" / "debug" / "libtpu.so",
        repo_root / "target" / "debug" / "tpu.dll",
        repo_root / "target" / "release" / "libtpu.dylib",
        repo_root / "target" / "release" / "libtpu.so",
        repo_root / "target" / "release" / "tpu.dll",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def configure_library_path() -> str:
    """Sets `TPU_LIBRARY_PATH` and `PJRT_NAMES_AND_LIBRARY_PATHS` in `os.environ`.

    Returns:
        The active `TPU_LIBRARY_PATH` value.
    """
    existing = os.environ.get("TPU_LIBRARY_PATH")
    if not existing:
        path = get_library_path()
        os.environ["TPU_LIBRARY_PATH"] = path
        os.environ.setdefault("PJRT_NAMES_AND_LIBRARY_PATHS", f"tpu:{path}")
        return path
    return existing


def set_init_args(args: str | Sequence[str]) -> None:
    """Configures `LIBTPU_INIT_ARGS` from a string or sequence of CLI flags.

    Args:
        args: CLI flag string or sequence of flags (e.g., `["--tpu_generation=v6e"]`).
    """
    os.environ["LIBTPU_INIT_ARGS"] = args if isinstance(args, str) else " ".join(args)


def get_init_args() -> list[str]:
    """Reads and splits the current `LIBTPU_INIT_ARGS` environment variable.

    Returns:
        List of whitespace-split initialization flag strings.
    """
    raw = os.environ.get("LIBTPU_INIT_ARGS", "")
    return raw.split()


def initialize(args: Sequence[str] | None = None) -> str:
    """Initializes the `libtpu` library path and optional `LIBTPU_INIT_ARGS` flags.

    Args:
        args: Optional sequence of initialization CLI flags.

    Returns:
        Configured `TPU_LIBRARY_PATH` filesystem path string.
    """
    if args is not None:
        set_init_args(args)
    return configure_library_path()


class _PjrtErrorDestroyArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_Error_Destroy_Args` (`24` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("error", ctypes.c_void_p),
    ]


class _PjrtErrorMessageArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_Error_Message_Args` (`40` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("error", ctypes.c_void_p),
        ("message", ctypes.c_void_p),
        ("message_size", ctypes.c_size_t),
    ]


class _PjrtErrorGetCodeArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_Error_GetCode_Args` (`32` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("error", ctypes.c_void_p),
        ("code", ctypes.c_int32),
        ("pad", ctypes.c_int32),
    ]


class _PjrtClientCreateArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_Client_Create_Args` (`88` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("create_options", ctypes.c_void_p),
        ("num_options", ctypes.c_size_t),
        ("kv_get_callback", ctypes.c_void_p),
        ("kv_get_user_arg", ctypes.c_void_p),
        ("kv_put_callback", ctypes.c_void_p),
        ("kv_put_user_arg", ctypes.c_void_p),
        ("kv_try_get_callback", ctypes.c_void_p),
        ("kv_try_get_user_arg", ctypes.c_void_p),
        ("client", ctypes.c_void_p),
    ]


class _PjrtClientDestroyArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_Client_Destroy_Args` (`24` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("client", ctypes.c_void_p),
    ]


class _PjrtClientDevicesArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_Client_Devices_Args` (`40` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("client", ctypes.c_void_p),
        ("devices", ctypes.c_void_p),
        ("num_devices", ctypes.c_size_t),
    ]


class _PjrtClientLookupDeviceArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_Client_LookupDevice_Args` (`40` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("client", ctypes.c_void_p),
        ("id", ctypes.c_int32),
        ("pad", ctypes.c_int32),
        ("device", ctypes.c_void_p),
    ]


class _PjrtProgram(ctypes.Structure):
    """C ABI structure matching `PJRT_Program` (`48` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("code", ctypes.c_void_p),
        ("code_size", ctypes.c_size_t),
        ("format", ctypes.c_void_p),
        ("format_size", ctypes.c_size_t),
    ]


class _PjrtClientCompileArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_Client_Compile_Args` (`56` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("client", ctypes.c_void_p),
        ("program", ctypes.POINTER(_PjrtProgram)),
        ("compile_options", ctypes.c_void_p),
        ("compile_options_size", ctypes.c_size_t),
        ("executable", ctypes.c_void_p),
    ]


class _PjrtClientBufferFromHostBufferArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_Client_BufferFromHostBuffer_Args` (`120` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("client", ctypes.c_void_p),
        ("data", ctypes.c_void_p),
        ("element_type", ctypes.c_int32),
        ("pad0", ctypes.c_int32),
        ("dims", ctypes.c_void_p),
        ("num_dims", ctypes.c_size_t),
        ("byte_strides", ctypes.c_void_p),
        ("num_byte_strides", ctypes.c_size_t),
        ("host_buffer_semantics", ctypes.c_int32),
        ("pad1", ctypes.c_int32),
        ("device", ctypes.c_void_p),
        ("memory", ctypes.c_void_p),
        ("device_layout", ctypes.c_void_p),
        ("done_with_host_buffer", ctypes.c_void_p),
        ("out_buffer", ctypes.c_void_p),
    ]


class _PjrtBufferDimensionsArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_Buffer_Dimensions_Args` (`40` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("buffer", ctypes.c_void_p),
        ("dims", ctypes.c_void_p),
        ("num_dims", ctypes.c_size_t),
    ]


class _PjrtBufferElementTypeArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_Buffer_ElementType_Args` (`32` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("buffer", ctypes.c_void_p),
        ("element_type", ctypes.c_int32),
        ("pad", ctypes.c_int32),
    ]


class _PjrtBufferCopyToDeviceArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_Buffer_CopyToDevice_Args` (`40` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("buffer", ctypes.c_void_p),
        ("dst_device", ctypes.c_void_p),
        ("dst_buffer", ctypes.c_void_p),
    ]


class _PjrtBufferToHostBufferArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_Buffer_ToHostBuffer_Args` (`56` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("src", ctypes.c_void_p),
        ("host_layout", ctypes.c_void_p),
        ("dst", ctypes.c_void_p),
        ("dst_size", ctypes.c_size_t),
        ("event", ctypes.c_void_p),
    ]


class _PjrtBufferDestroyArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_Buffer_Destroy_Args` (`24` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("buffer", ctypes.c_void_p),
    ]


class _PjrtExecutableGetCostAnalysisArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_Executable_GetCostAnalysis_Args` (`40` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("executable", ctypes.c_void_p),
        ("properties", ctypes.c_void_p),
        ("num_properties", ctypes.c_size_t),
    ]


class _PjrtExecutableOptimizedProgramArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_Executable_OptimizedProgram_Args` (`32` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("executable", ctypes.c_void_p),
        ("program", ctypes.POINTER(_PjrtProgram)),
    ]


class _PjrtBufferGetMemoryLayoutArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_Buffer_GetMemoryLayout_Args` (`104` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("buffer", ctypes.c_void_p),
        ("layout_words", ctypes.c_size_t * 10),
    ]


class _PjrtExecutableSerializeArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_Executable_Serialize_Args` (`56` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("executable", ctypes.c_void_p),
        ("serialized_bytes", ctypes.c_void_p),
        ("serialized_bytes_size", ctypes.c_size_t),
        ("serialized_executable", ctypes.c_void_p),
        ("serialized_executable_deleter", ctypes.c_void_p),
    ]


class _PjrtLoadedExecutableDestroyArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_LoadedExecutable_Destroy_Args` (`24` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("executable", ctypes.c_void_p),
    ]


class _PjrtLoadedExecutableGetExecutableArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_LoadedExecutable_GetExecutable_Args` (`32` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("loaded_executable", ctypes.c_void_p),
        ("executable", ctypes.c_void_p),
    ]


class _PjrtLoadedExecutableExecuteArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_LoadedExecutable_Execute_Args` (`80` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("executable", ctypes.c_void_p),
        ("options", ctypes.c_void_p),
        ("argument_lists", ctypes.c_void_p),
        ("num_devices", ctypes.c_size_t),
        ("num_args", ctypes.c_size_t),
        ("output_lists", ctypes.c_void_p),
        ("device_complete_events", ctypes.c_void_p),
        ("execute_device", ctypes.c_void_p),
    ]


class _PjrtExecutableDeserializeAndLoadArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_Executable_DeserializeAndLoad_Args` (`64` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("client", ctypes.c_void_p),
        ("serialized_executable", ctypes.c_void_p),
        ("serialized_executable_size", ctypes.c_size_t),
        ("loaded_executable", ctypes.c_void_p),
        ("overridden_serialized_compile_options", ctypes.c_void_p),
        ("overridden_serialized_compile_options_size", ctypes.c_size_t),
    ]


class _PjrtNamedValue(ctypes.Structure):
    """C ABI structure matching `PJRT_NamedValue` (`48` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("name", ctypes.c_void_p),
        ("name_size", ctypes.c_size_t),
        ("value_type", ctypes.c_int32),
        ("value_type_pad", ctypes.c_uint32),
        ("value_payload", ctypes.c_uint64),
        ("value_size", ctypes.c_size_t),
    ]


class _PjrtLoadedExecutableAddressableDevicesArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_LoadedExecutable_AddressableDevices_Args` (`40` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("executable", ctypes.c_void_p),
        ("addressable_devices", ctypes.c_void_p),
        ("num_addressable_devices", ctypes.c_size_t),
    ]


class _PjrtLoadedExecutableFingerprintArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_LoadedExecutable_Fingerprint_Args` (`40` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("executable", ctypes.c_void_p),
        ("executable_fingerprint", ctypes.c_void_p),
        ("executable_fingerprint_size", ctypes.c_size_t),
    ]


_CDLL_CACHE: dict[str, ctypes.CDLL] = {}
_SERIALIZED_DELETER_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


def _load_cdll() -> ctypes.CDLL:
    """Loads the `libtpu` shared library and configures C function prototypes.

    Returns:
        Configured `ctypes.CDLL` handle.
    """
    lib_path = get_library_path()
    cached = _CDLL_CACHE.get(lib_path)
    if cached is not None:
        return cached
    lib = ctypes.CDLL(lib_path)

    lib.PJRT_Error_Destroy.argtypes = [ctypes.POINTER(_PjrtErrorDestroyArgs)]
    lib.PJRT_Error_Destroy.restype = None

    lib.PJRT_Error_Message.argtypes = [ctypes.POINTER(_PjrtErrorMessageArgs)]
    lib.PJRT_Error_Message.restype = None

    lib.PJRT_Error_GetCode.argtypes = [ctypes.POINTER(_PjrtErrorGetCodeArgs)]
    lib.PJRT_Error_GetCode.restype = ctypes.c_void_p

    lib.PJRT_Client_Create.argtypes = [ctypes.POINTER(_PjrtClientCreateArgs)]
    lib.PJRT_Client_Create.restype = ctypes.c_void_p

    lib.PJRT_Client_Destroy.argtypes = [ctypes.POINTER(_PjrtClientDestroyArgs)]
    lib.PJRT_Client_Destroy.restype = ctypes.c_void_p

    lib.PJRT_Client_AddressableDevices.argtypes = [ctypes.POINTER(_PjrtClientDevicesArgs)]
    lib.PJRT_Client_AddressableDevices.restype = ctypes.c_void_p

    lib.PJRT_Client_LookupDevice.argtypes = [ctypes.POINTER(_PjrtClientLookupDeviceArgs)]
    lib.PJRT_Client_LookupDevice.restype = ctypes.c_void_p

    lib.PJRT_Client_Compile.argtypes = [ctypes.POINTER(_PjrtClientCompileArgs)]
    lib.PJRT_Client_Compile.restype = ctypes.c_void_p

    lib.PJRT_Client_BufferFromHostBuffer.argtypes = [
        ctypes.POINTER(_PjrtClientBufferFromHostBufferArgs)
    ]
    lib.PJRT_Client_BufferFromHostBuffer.restype = ctypes.c_void_p

    lib.PJRT_Buffer_Dimensions.argtypes = [ctypes.POINTER(_PjrtBufferDimensionsArgs)]
    lib.PJRT_Buffer_Dimensions.restype = ctypes.c_void_p

    lib.PJRT_Buffer_ElementType.argtypes = [ctypes.POINTER(_PjrtBufferElementTypeArgs)]
    lib.PJRT_Buffer_ElementType.restype = ctypes.c_void_p

    lib.PJRT_Buffer_CopyToDevice.argtypes = [ctypes.POINTER(_PjrtBufferCopyToDeviceArgs)]
    lib.PJRT_Buffer_CopyToDevice.restype = ctypes.c_void_p

    lib.PJRT_Buffer_ToHostBuffer.argtypes = [ctypes.POINTER(_PjrtBufferToHostBufferArgs)]
    lib.PJRT_Buffer_ToHostBuffer.restype = ctypes.c_void_p

    lib.PJRT_Buffer_Destroy.argtypes = [ctypes.POINTER(_PjrtBufferDestroyArgs)]
    lib.PJRT_Buffer_Destroy.restype = ctypes.c_void_p

    lib.PJRT_Buffer_GetMemoryLayout.argtypes = [ctypes.POINTER(_PjrtBufferGetMemoryLayoutArgs)]
    lib.PJRT_Buffer_GetMemoryLayout.restype = ctypes.c_void_p

    lib.PJRT_Executable_GetCostAnalysis.argtypes = [
        ctypes.POINTER(_PjrtExecutableGetCostAnalysisArgs)
    ]
    lib.PJRT_Executable_GetCostAnalysis.restype = ctypes.c_void_p

    lib.PJRT_Executable_OptimizedProgram.argtypes = [
        ctypes.POINTER(_PjrtExecutableOptimizedProgramArgs)
    ]
    lib.PJRT_Executable_OptimizedProgram.restype = ctypes.c_void_p

    lib.PJRT_Executable_Serialize.argtypes = [ctypes.POINTER(_PjrtExecutableSerializeArgs)]
    lib.PJRT_Executable_Serialize.restype = ctypes.c_void_p

    lib.PJRT_LoadedExecutable_Destroy.argtypes = [ctypes.POINTER(_PjrtLoadedExecutableDestroyArgs)]
    lib.PJRT_LoadedExecutable_Destroy.restype = ctypes.c_void_p

    lib.PJRT_LoadedExecutable_GetExecutable.argtypes = [
        ctypes.POINTER(_PjrtLoadedExecutableGetExecutableArgs)
    ]
    lib.PJRT_LoadedExecutable_GetExecutable.restype = ctypes.c_void_p

    lib.PJRT_LoadedExecutable_AddressableDevices.argtypes = [
        ctypes.POINTER(_PjrtLoadedExecutableAddressableDevicesArgs)
    ]
    lib.PJRT_LoadedExecutable_AddressableDevices.restype = ctypes.c_void_p

    lib.PJRT_LoadedExecutable_Execute.argtypes = [ctypes.POINTER(_PjrtLoadedExecutableExecuteArgs)]
    lib.PJRT_LoadedExecutable_Execute.restype = ctypes.c_void_p

    lib.PJRT_Executable_DeserializeAndLoad.argtypes = [
        ctypes.POINTER(_PjrtExecutableDeserializeAndLoadArgs)
    ]
    lib.PJRT_Executable_DeserializeAndLoad.restype = ctypes.c_void_p

    lib.PJRT_LoadedExecutable_Fingerprint.argtypes = [
        ctypes.POINTER(_PjrtLoadedExecutableFingerprintArgs)
    ]
    lib.PJRT_LoadedExecutable_Fingerprint.restype = ctypes.c_void_p

    lib.TpuStatus_New.argtypes = []
    lib.TpuStatus_New.restype = ctypes.c_void_p

    lib.TpuStatus_Free.argtypes = [ctypes.c_void_p]
    lib.TpuStatus_Free.restype = None

    lib.TpuStatus_Ok.argtypes = [ctypes.c_void_p]
    lib.TpuStatus_Ok.restype = ctypes.c_bool

    lib.TpuProfiler_Create.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    ]
    lib.TpuProfiler_Create.restype = None

    lib.TpuProfiler_Destroy.argtypes = [ctypes.c_void_p]
    lib.TpuProfiler_Destroy.restype = None

    lib.TpuProfiler_Start.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.TpuProfiler_Start.restype = None

    lib.TpuProfiler_Stop.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.TpuProfiler_Stop.restype = None

    lib.TpuProfiler_CollectData.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.TpuProfiler_CollectData.restype = None

    for ext_fn in (
        "PJRT_Event_Destroy",
        "PJRT_PhaseCompile_GetCompiler",
        "PJRT_PhaseCompile_DestroyCompiler",
        "PJRT_PhaseCompile_GetPhaseNames",
        "PJRT_PhaseCompile_RunPhase",
        "PJRT_Megascale_CreateClientContext",
        "PJRT_Megascale_DeleteClientContext",
        "PJRT_Megascale_CreateMultiSliceConfig",
        "PJRT_Megascale_DeleteMultiSliceConfig",
        "PJRT_MultiSlice_Config_Serialize",
        "PJRT_Megascale_Barrier",
        "PJRT_Collectives_CreateCommunicator",
        "PJRT_Communicator_Destroy",
        "PJRT_Communicator_AllReduce",
        "PJRT_Communicator_ReduceScatter",
        "PJRT_Communicator_AllGather",
        "PJRT_Communicator_AllToAll",
        "PJRT_Communicator_CollectivePermute",
        "PJRT_Communicator_ToString",
    ):
        fn_obj = getattr(lib, ext_fn)
        fn_obj.argtypes = [ctypes.c_void_p]
        fn_obj.restype = ctypes.c_void_p

    lib.TpuEmbeddingEngineState_Create.argtypes = []
    lib.TpuEmbeddingEngineState_Create.restype = ctypes.c_void_p
    lib.TpuEmbeddingEngineState_Free.argtypes = [ctypes.c_void_p]
    lib.TpuEmbeddingEngineState_Free.restype = None
    lib.TpuEmbeddingEngine_WriteParameters.argtypes = [ctypes.c_void_p]
    lib.TpuEmbeddingEngine_WriteParameters.restype = ctypes.c_int32
    lib.TpuEmbeddingEngine_ReadParameters.argtypes = [ctypes.c_void_p]
    lib.TpuEmbeddingEngine_ReadParameters.restype = ctypes.c_int32

    lib.libtpu_rs_disassemble_executable_llo.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.libtpu_rs_disassemble_executable_llo.restype = ctypes.c_size_t

    lib.libtpu_rs_inspect_register_allocation.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.libtpu_rs_inspect_register_allocation.restype = ctypes.c_size_t

    lib.libtpu_rs_executable_buffer_assignment_proto.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.libtpu_rs_executable_buffer_assignment_proto.restype = ctypes.c_size_t

    _CDLL_CACHE[lib_path] = lib
    return lib


def _check_pjrt_error(lib: ctypes.CDLL, err_ptr: int | None) -> None:
    """Checks a `PJRT_Error*` pointer and raises `TpuError` if non-null.

    Args:
        lib: Loaded `ctypes.CDLL` handle.
        err_ptr: Raw `PJRT_Error*` address returned by a `PJRT_*` function.

    Raises:
        TpuError: If `err_ptr` is non-zero.
    """
    if err_ptr:
        code_args = _PjrtErrorGetCodeArgs(ctypes.sizeof(_PjrtErrorGetCodeArgs), 0, err_ptr, 0, 0)
        lib.PJRT_Error_GetCode(ctypes.byref(code_args))
        msg_args = _PjrtErrorMessageArgs(ctypes.sizeof(_PjrtErrorMessageArgs), 0, err_ptr, 0, 0)
        lib.PJRT_Error_Message(ctypes.byref(msg_args))
        text = ctypes.string_at(int(msg_args.message or 0), int(msg_args.message_size)).decode(
            "utf-8", errors="replace"
        )
        code = int(code_args.code)
        del_args = _PjrtErrorDestroyArgs(ctypes.sizeof(_PjrtErrorDestroyArgs), 0, err_ptr)
        lib.PJRT_Error_Destroy(ctypes.byref(del_args))
        raise TpuError(code, text)


class TpuBuffer:
    """Strongly-typed wrapper over an on-device `PJRT_Buffer*`."""

    def __init__(self, lib: ctypes.CDLL, handle: ctypes.c_void_p, num_elements: int) -> None:
        """Initializes a `TpuBuffer`.

        Args:
            lib: Loaded `ctypes.CDLL` instance.
            handle: Opaque `PJRT_Buffer*` pointer.
            num_elements: Number of 32-bit elements stored in the buffer.
        """
        self._lib: ctypes.CDLL = lib
        self._handle: ctypes.c_void_p = handle
        self._num_elements: int = num_elements

    def shape(self) -> tuple[int, ...]:
        """Returns the logical shape tuple of this buffer via `PJRT_Buffer_Dimensions`.

        Returns:
            Tuple of dimension extents.
        """
        dim_args = _PjrtBufferDimensionsArgs(
            ctypes.sizeof(_PjrtBufferDimensionsArgs), 0, self._handle.value or 0, 0, 0
        )
        self._lib.PJRT_Buffer_Dimensions(ctypes.byref(dim_args))
        dims_ptr = int(dim_args.dims or 0)
        num_dims = int(dim_args.num_dims)
        dims_arr = (ctypes.c_int64 * num_dims).from_address(dims_ptr)
        return tuple(int(dims_arr[i]) for i in range(num_dims))

    def dtype(self) -> DataType:
        """Returns the `DataType` of this buffer via `PJRT_Buffer_ElementType`.

        Returns:
            `DataType` enum value of this buffer.
        """
        et_args = _PjrtBufferElementTypeArgs(
            ctypes.sizeof(_PjrtBufferElementTypeArgs), 0, self._handle.value or 0, 0, 0
        )
        self._lib.PJRT_Buffer_ElementType(ctypes.byref(et_args))
        return DataType(int(et_args.element_type))

    def tile_count(self) -> int:
        """Returns the number of hardware layout tiles via `PJRT_Buffer_GetMemoryLayout`.

        Returns:
            Number of hardware layout tiles on this buffer.
        """
        ml_args = _PjrtBufferGetMemoryLayoutArgs(
            ctypes.sizeof(_PjrtBufferGetMemoryLayoutArgs),
            0,
            self._handle.value or 0,
            (ctypes.c_size_t * 10)(),
        )
        self._lib.PJRT_Buffer_GetMemoryLayout(ctypes.byref(ml_args))
        return int(ml_args.layout_words[6])

    def byte_size(self) -> int:
        """Computes the host byte size of this buffer from its `dtype()` and element count.

        Returns:
            Exact host byte length of this buffer.
        """
        return self.dtype().byte_size_for_elements(self._num_elements)

    def to_bytes(self) -> bytes:
        """Reads the buffer from device HBM as raw bytes via `PJRT_Buffer_ToHostBuffer`.

        Returns:
            Raw bytes read from the TPU buffer.
        """
        byte_len = self.byte_size()
        raw = ctypes.create_string_buffer(byte_len)
        d2h_args = _PjrtBufferToHostBufferArgs(
            ctypes.sizeof(_PjrtBufferToHostBufferArgs),
            0,
            self._handle.value or 0,
            0,
            ctypes.addressof(raw),
            byte_len,
            0,
        )
        err = self._lib.PJRT_Buffer_ToHostBuffer(ctypes.byref(d2h_args))
        _check_pjrt_error(self._lib, err)
        return raw.raw

    def to_float_list(self) -> list[float]:
        """Reads the buffer from device HBM and unpacks it as a list of `float`.

        Returns:
            List of floating-point values read from the TPU buffer (`F16`, `BF16`, `F32`, `F64`).
        """
        raw_bytes = self.to_bytes()
        dt = self.dtype()
        if dt == DataType.F64:
            return [float(x) for x in struct.unpack(f"<{self._num_elements}d", raw_bytes)]
        if dt == DataType.F16:
            return [float(x) for x in struct.unpack(f"<{self._num_elements}e", raw_bytes)]
        if dt == DataType.BF16:
            u16s = struct.unpack(f"<{self._num_elements}H", raw_bytes)
            return [float(struct.unpack("<f", (w << 16).to_bytes(4, "little"))[0]) for w in u16s]
        return [float(x) for x in struct.unpack(f"<{self._num_elements}f", raw_bytes)]

    def to_int_list(self) -> list[int]:
        """Reads the buffer from device HBM and unpacks it as a list of `int`.

        Returns:
            List of integer values read from the TPU buffer (`PRED`, `S8..S64`, `U8..U64`).
        """
        raw_bytes = self.to_bytes()
        dt = self.dtype()
        fmt_map: dict[DataType, str] = {
            DataType.PRED: "B",
            DataType.S8: "b",
            DataType.U8: "B",
            DataType.S16: "h",
            DataType.U16: "H",
            DataType.S32: "i",
            DataType.U32: "I",
            DataType.S64: "q",
            DataType.U64: "Q",
        }
        code = fmt_map.get(dt, "i")
        return [int(x) for x in struct.unpack(f"<{self._num_elements}{code}", raw_bytes)]

    def to_memoryview(self) -> memoryview:
        """Returns a read-only `memoryview` over the buffer bytes read from HBM.

        Returns:
            Read-only `memoryview` of the buffer payload.
        """
        return memoryview(self.to_bytes())

    @property
    def __array_interface__(self) -> dict[str, object]:
        """Returns the NumPy `__array_interface__` dictionary for this buffer.

        Returns:
            Dictionary describing `shape`, `typestr`, `data`, and `version`.
        """
        raw_bytes = self.to_bytes()
        return {
            "version": 3,
            "shape": self.shape(),
            "typestr": self.dtype().numpy_typestr(),
            "data": raw_bytes,
            "readonly": True,
        }

    def __dlpack_device__(self) -> tuple[int, int]:
        """Returns the DLPack `(device_type, device_id)` tuple (`kDLOneAPI = 14`, `0`).

        Returns:
            Tuple `(14, 0)` identifying the TPU device.
        """
        return (14, 0)

    def __dlpack__(self, stream: int | None = None) -> object:
        """Returns a CPython `PyCapsule` (`"dltensor"`) wrapping a DLPack v1.0 `DLManagedTensor`.

        Args:
            stream: Optional stream identifier (unused on synchronous HBM view).

        Returns:
            `PyCapsule` wrapping a populated `_DLManagedTensor` structure.
        """
        _ = stream
        dims = self.shape()
        ndim = len(dims)
        dt = self.dtype()
        shape_arr = (ctypes.c_int64 * max(ndim, 1))(*dims) if ndim > 0 else (ctypes.c_int64 * 1)(1)
        strides_list: list[int] = [1] * max(ndim, 1)
        running = 1
        for ax in range(ndim - 1, -1, -1):
            strides_list[ax] = running
            running *= max(dims[ax], 1)
        strides_arr = (ctypes.c_int64 * max(ndim, 1))(*strides_list)
        dl_dev = _DLDevice(14, 0)
        dl_dt = _DLDataType(dt.dlpack_type_code(), max(dt.bit_size(), 8), 1)
        dl_tensor = _DLTensor(
            int(self._handle.value or 0),
            dl_dev,
            ndim,
            dl_dt,
            ctypes.cast(shape_arr, ctypes.POINTER(ctypes.c_int64)),
            ctypes.cast(strides_arr, ctypes.POINTER(ctypes.c_int64)),
            0,
        )
        managed = _DLManagedTensor(dl_tensor, int(self._handle.value or 0), 0)
        self._dlpack_keepalive: tuple[object, ...] = (shape_arr, strides_arr, managed)
        return _create_c_api_capsule(ctypes.addressof(managed), "dltensor")

    def close(self) -> None:
        """Deletes and frees the underlying `PJRT_Buffer*` via `PJRT_Buffer_Destroy`."""
        if self._handle.value:
            del_args = _PjrtBufferDestroyArgs(
                ctypes.sizeof(_PjrtBufferDestroyArgs), 0, self._handle.value
            )
            self._lib.PJRT_Buffer_Destroy(ctypes.byref(del_args))
            self._handle = ctypes.c_void_p(0)

    def __enter__(self) -> TpuBuffer:
        """Enters a context manager block returning `self`."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Closes this `TpuBuffer` when exiting a context manager block.

        Args:
            exc_info: Exception tuple (`exc_type`, `exc_val`, `exc_tb`) if raised.
        """
        self.close()


class TpuExecutable:
    """Strongly-typed wrapper over a compiled `PJRT_LoadedExecutable*`."""

    def __init__(self, lib: ctypes.CDLL, handle: ctypes.c_void_p) -> None:
        """Initializes a `TpuExecutable`.

        Args:
            lib: Loaded `ctypes.CDLL` instance.
            handle: Opaque `PJRT_LoadedExecutable*` pointer.
        """
        self._lib: ctypes.CDLL = lib
        self._handle: ctypes.c_void_p = handle

    def execute_multi(
        self,
        device_id: int,
        inputs: Sequence[TpuBuffer],
        num_outputs: int = 1,
    ) -> list[TpuBuffer]:
        """Executes the compiled program returning `num_outputs` output `TpuBuffer` objects.

        Args:
            device_id: Logical TPU device ID.
            inputs: Sequence of input `TpuBuffer` objects.
            num_outputs: Number of output buffers to retrieve.

        Returns:
            List of output `TpuBuffer` objects.
        """
        dev_args = _PjrtLoadedExecutableAddressableDevicesArgs(
            ctypes.sizeof(_PjrtLoadedExecutableAddressableDevicesArgs),
            0,
            self._handle.value or 0,
            0,
            0,
        )
        self._lib.PJRT_LoadedExecutable_AddressableDevices(ctypes.byref(dev_args))
        num_devs = int(dev_args.num_addressable_devices)
        if device_id < 0 or device_id >= num_devs:
            raise TpuError(5, f"Device {device_id} not found (addressable={num_devs})")
        devs_arr = (ctypes.c_void_p * num_devs).from_address(int(dev_args.addressable_devices or 0))
        target_dev = int(devs_arr[device_id] or 0)

        in_arr_type = ctypes.c_void_p * len(inputs)
        in_ptrs = in_arr_type(*(b._handle.value or 0 for b in inputs))
        in_lists = (ctypes.c_void_p * 1)(ctypes.addressof(in_ptrs))
        out_arr_type = ctypes.c_void_p * num_outputs
        out_ptrs = out_arr_type()
        out_lists = (ctypes.c_void_p * 1)(ctypes.addressof(out_ptrs))
        ex_args = _PjrtLoadedExecutableExecuteArgs(
            ctypes.sizeof(_PjrtLoadedExecutableExecuteArgs),
            0,
            self._handle.value or 0,
            0,
            ctypes.addressof(in_lists),
            1,
            len(inputs),
            ctypes.addressof(out_lists),
            0,
            target_dev,
        )
        err = self._lib.PJRT_LoadedExecutable_Execute(ctypes.byref(ex_args))
        _check_pjrt_error(self._lib, err)
        results: list[TpuBuffer] = []
        for idx in range(num_outputs):
            handle = ctypes.c_void_p(int(out_ptrs[idx] or 0))
            buf = TpuBuffer(self._lib, handle, 1)
            elem_count = 1
            for dim in buf.shape():
                elem_count *= dim
            buf._num_elements = elem_count
            results.append(buf)
        return results

    def execute(self, device_id: int, inputs: Sequence[TpuBuffer]) -> TpuBuffer:
        """Executes the compiled program on `device_id` with `inputs`.

        Args:
            device_id: Logical TPU device ID.
            inputs: Sequence of input `TpuBuffer` objects.

        Returns:
            Output `TpuBuffer` produced by the execution.
        """
        return self.execute_multi(device_id, inputs, num_outputs=1)[0]

    def execute_sharded(
        self,
        per_device_inputs: Sequence[Sequence[TpuBuffer]],
        num_outputs: int = 1,
    ) -> list[list[TpuBuffer]]:
        """Executes the compiled SPMD program across multiple addressable devices.

        Args:
            per_device_inputs: Sequence of per-device input `TpuBuffer` sequences.
            num_outputs: Number of output buffers produced per device.

        Returns:
            List of per-device output `TpuBuffer` lists.
        """
        return [
            self.execute_multi(dev_idx, dev_ins, num_outputs=num_outputs)
            for dev_idx, dev_ins in enumerate(per_device_inputs)
        ]

    def cost_analysis(self) -> dict[str, float]:
        """Returns the dictionary of cost analysis metrics via `PJRT_Executable_GetCostAnalysis`.

        Returns:
            Dictionary mapping metric names (`"flops"`, `"transcendentals"`,
            `"bytes_accessed"`, `"optimal_seconds"`) to float values.
        """
        get_ex = _PjrtLoadedExecutableGetExecutableArgs(
            ctypes.sizeof(_PjrtLoadedExecutableGetExecutableArgs),
            0,
            self._handle.value or 0,
            0,
        )
        self._lib.PJRT_LoadedExecutable_GetExecutable(ctypes.byref(get_ex))
        cost_args = _PjrtExecutableGetCostAnalysisArgs(
            ctypes.sizeof(_PjrtExecutableGetCostAnalysisArgs),
            0,
            int(get_ex.executable or 0),
            0,
            0,
        )
        self._lib.PJRT_Executable_GetCostAnalysis(ctypes.byref(cost_args))
        out: dict[str, float] = {}
        base_addr = int(cost_args.properties or 0)
        stride = ctypes.sizeof(_PjrtNamedValue)
        for idx in range(int(cost_args.num_properties)):
            prop = _PjrtNamedValue.from_address(base_addr + idx * stride)
            key = ctypes.string_at(int(prop.name or 0), int(prop.name_size)).decode("utf-8")
            raw_u32 = int(prop.value_payload) & 0xFFFFFFFF
            val = float(struct.unpack("<f", struct.pack("<I", raw_u32))[0])
            out[key] = val
        return out

    def serialize(self) -> bytes:
        """Serializes the executable into wire bytes via `PJRT_Executable_Serialize`.

        Returns:
            Serialized executable byte payload.
        """
        get_ex = _PjrtLoadedExecutableGetExecutableArgs(
            ctypes.sizeof(_PjrtLoadedExecutableGetExecutableArgs),
            0,
            self._handle.value or 0,
            0,
        )
        self._lib.PJRT_LoadedExecutable_GetExecutable(ctypes.byref(get_ex))
        ser_args = _PjrtExecutableSerializeArgs(
            ctypes.sizeof(_PjrtExecutableSerializeArgs),
            0,
            int(get_ex.executable or 0),
            0,
            0,
            0,
            0,
        )
        self._lib.PJRT_Executable_Serialize(ctypes.byref(ser_args))
        payload = ctypes.string_at(
            int(ser_args.serialized_bytes or 0), int(ser_args.serialized_bytes_size)
        )
        deleter = _SERIALIZED_DELETER_TYPE(int(ser_args.serialized_executable_deleter or 0))
        deleter(ser_args.serialized_executable)
        return payload

    def fingerprint(self) -> str:
        """Returns the 32-character hexadecimal fingerprint via `PJRT_LoadedExecutable_Fingerprint`.

        Returns:
            Hexadecimal fingerprint string.
        """
        fp_args = _PjrtLoadedExecutableFingerprintArgs(
            ctypes.sizeof(_PjrtLoadedExecutableFingerprintArgs),
            0,
            self._handle.value or 0,
            0,
            0,
        )
        self._lib.PJRT_LoadedExecutable_Fingerprint(ctypes.byref(fp_args))
        return ctypes.string_at(
            int(fp_args.executable_fingerprint or 0), int(fp_args.executable_fingerprint_size)
        ).decode("utf-8")

    def flops(self) -> float:
        """Returns the estimated FLOPs per execution via `PJRT_Executable_GetCostAnalysis`.

        Returns:
            Estimated floating-point operation count.
        """
        get_ex = _PjrtLoadedExecutableGetExecutableArgs(
            ctypes.sizeof(_PjrtLoadedExecutableGetExecutableArgs),
            0,
            self._handle.value or 0,
            0,
        )
        self._lib.PJRT_LoadedExecutable_GetExecutable(ctypes.byref(get_ex))
        cost_args = _PjrtExecutableGetCostAnalysisArgs(
            ctypes.sizeof(_PjrtExecutableGetCostAnalysisArgs),
            0,
            int(get_ex.executable or 0),
            0,
            0,
        )
        self._lib.PJRT_Executable_GetCostAnalysis(ctypes.byref(cost_args))
        prop0 = _PjrtNamedValue.from_address(int(cost_args.properties or 0))
        raw_u32 = int(prop0.value_payload) & 0xFFFFFFFF
        return float(struct.unpack("<f", struct.pack("<I", raw_u32))[0])

    def optimized_program_bytes(self) -> bytes:
        """Returns the optimized HLO program bytes via `PJRT_Executable_OptimizedProgram`.

        Returns:
            Optimized HLO program byte payload.
        """
        get_ex = _PjrtLoadedExecutableGetExecutableArgs(
            ctypes.sizeof(_PjrtLoadedExecutableGetExecutableArgs),
            0,
            self._handle.value or 0,
            0,
        )
        self._lib.PJRT_LoadedExecutable_GetExecutable(ctypes.byref(get_ex))
        prog = _PjrtProgram(ctypes.sizeof(_PjrtProgram), 0, 0, 0, 0, 0)
        opt_args = _PjrtExecutableOptimizedProgramArgs(
            ctypes.sizeof(_PjrtExecutableOptimizedProgramArgs),
            0,
            int(get_ex.executable or 0),
            ctypes.pointer(prog),
        )
        self._lib.PJRT_Executable_OptimizedProgram(ctypes.byref(opt_args))
        raw = ctypes.create_string_buffer(int(prog.code_size))
        prog.code = ctypes.addressof(raw)
        err = self._lib.PJRT_Executable_OptimizedProgram(ctypes.byref(opt_args))
        _check_pjrt_error(self._lib, err)
        return raw.raw

    def disassemble_vliw(self) -> str:
        """Disassembles this executable into a multi-line `DeepSea` LLO VLIW assembly listing.

        Returns:
            Multi-line LLO VLIW assembly string.
        """
        sz = int(self._lib.libtpu_rs_disassemble_executable_llo(self._handle, None, 0))
        buf = ctypes.create_string_buffer(sz + 1)
        self._lib.libtpu_rs_disassemble_executable_llo(self._handle, buf, sz + 1)
        return buf.value.decode("utf-8")

    def inspect_register_allocation(self) -> str:
        """Returns the physical register allocation and `VMEM` spill summary for this executable.

        Returns:
            Human-readable register allocation summary string.
        """
        sz = int(self._lib.libtpu_rs_inspect_register_allocation(self._handle, None, 0))
        buf = ctypes.create_string_buffer(sz + 1)
        self._lib.libtpu_rs_inspect_register_allocation(self._handle, buf, sz + 1)
        return buf.value.decode("utf-8")

    def buffer_assignment_proto(self) -> bytes:
        """Returns the wire-encoded `xla.BufferAssignmentProto` for this executable.

        Returns:
            Serialized `BufferAssignmentProto` byte payload.
        """
        sz = int(self._lib.libtpu_rs_executable_buffer_assignment_proto(self._handle, None, 0))
        buf = ctypes.create_string_buffer(sz)
        self._lib.libtpu_rs_executable_buffer_assignment_proto(self._handle, buf, sz)
        return buf.raw

    def close(self) -> None:
        """Frees the underlying `PJRT_LoadedExecutable*` via `PJRT_LoadedExecutable_Destroy`."""
        if self._handle.value:
            del_args = _PjrtLoadedExecutableDestroyArgs(
                ctypes.sizeof(_PjrtLoadedExecutableDestroyArgs), 0, self._handle.value
            )
            self._lib.PJRT_LoadedExecutable_Destroy(ctypes.byref(del_args))
            self._handle = ctypes.c_void_p(0)

    def __enter__(self) -> TpuExecutable:
        """Enters a context manager block returning `self`."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Closes this `TpuExecutable` when exiting a context manager block.

        Args:
            exc_info: Exception tuple (`exc_type`, `exc_val`, `exc_tb`) if raised.
        """
        self.close()


class TpuProfiler:
    """Strongly-typed wrapper over `TpuProfiler*`."""

    def __init__(self) -> None:
        """Creates a new `TpuProfiler` session."""
        self._lib: ctypes.CDLL = _load_cdll()
        self._status: ctypes.c_void_p = ctypes.c_void_p(self._lib.TpuStatus_New())
        self._handle: ctypes.c_void_p = ctypes.c_void_p(0)
        self._lib.TpuProfiler_Create(ctypes.byref(self._handle), self._status)

    def start(self) -> bool:
        """Starts trace collection on the TPU profiler.

        Returns:
            `True` if trace collection started without error.
        """
        self._lib.TpuProfiler_Start(self._handle, self._status)
        return bool(self._lib.TpuStatus_Ok(self._status))

    def stop(self) -> bool:
        """Stops trace collection on the TPU profiler.

        Returns:
            `True` if trace collection stopped without error.
        """
        self._lib.TpuProfiler_Stop(self._handle, self._status)
        return bool(self._lib.TpuStatus_Ok(self._status))

    def collect_data(self) -> bytes:
        """Collects serialized `XSpace` profiling trace bytes.

        Returns:
            Serialized `XSpace` byte payload.
        """
        sz = ctypes.c_size_t(0)
        self._lib.TpuProfiler_CollectData(
            self._handle,
            self._status,
            None,
            ctypes.byref(sz),
        )
        buf = ctypes.create_string_buffer(sz.value)
        self._lib.TpuProfiler_CollectData(
            self._handle,
            self._status,
            buf,
            ctypes.byref(sz),
        )
        return buf.raw

    def close(self) -> None:
        """Destroys the `TpuProfiler` session and its status handle."""
        if self._handle.value:
            self._lib.TpuProfiler_Destroy(self._handle)
            self._handle = ctypes.c_void_p(0)
        if self._status.value:
            self._lib.TpuStatus_Free(self._status)
            self._status = ctypes.c_void_p(0)

    def __enter__(self) -> TpuProfiler:
        """Enters a context manager block returning `self`."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Closes this `TpuProfiler` when exiting a context manager block.

        Args:
            exc_info: Exception tuple (`exc_type`, `exc_val`, `exc_tb`) if raised.
        """
        self.close()


class TpuClient:
    """Strongly-typed Python wrapper over `PJRT_Client*`."""

    def __init__(self) -> None:
        """Creates and initializes a `TpuClient` via `PJRT_Client_Create`."""
        self._lib: ctypes.CDLL = _load_cdll()
        create_args = _PjrtClientCreateArgs(
            ctypes.sizeof(_PjrtClientCreateArgs), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        )
        err = self._lib.PJRT_Client_Create(ctypes.byref(create_args))
        _check_pjrt_error(self._lib, err)
        self._handle: ctypes.c_void_p = ctypes.c_void_p(int(create_args.client or 0))

    def device_count(self) -> int:
        """Returns the number of addressable TPU devices via `PJRT_Client_AddressableDevices`.

        Returns:
            Addressable TPU device count.
        """
        dev_args = _PjrtClientDevicesArgs(
            ctypes.sizeof(_PjrtClientDevicesArgs), 0, self._handle.value or 0, 0, 0
        )
        self._lib.PJRT_Client_AddressableDevices(ctypes.byref(dev_args))
        return int(dev_args.num_devices)

    def _buffer_from_packed(
        self,
        device_id: int,
        elem_type: int,
        packed: bytes,
        num_elements: int,
        dims: Sequence[int] | None,
    ) -> TpuBuffer:
        """Transfers packed host bytes to `device_id` via `PJRT_Client_BufferFromHostBuffer`.

        Args:
            device_id: Logical TPU device ID.
            elem_type: Raw `PJRT_Buffer_Type` integer (`11` = `F32`, `4` = `S32`).
            packed: Packed host byte buffer.
            num_elements: Number of 32-bit elements.
            dims: Optional N-dimensional shape sequence.

        Returns:
            Allocated `TpuBuffer` instance.
        """
        look_args = _PjrtClientLookupDeviceArgs(
            ctypes.sizeof(_PjrtClientLookupDeviceArgs), 0, self._handle.value or 0, device_id, 0, 0
        )
        err_look = self._lib.PJRT_Client_LookupDevice(ctypes.byref(look_args))
        _check_pjrt_error(self._lib, err_look)
        shape_seq = list(dims) if dims is not None else [num_elements]
        c_dims = (ctypes.c_int64 * len(shape_seq))(*shape_seq)
        c_buf = ctypes.create_string_buffer(packed, len(packed))
        h2d_args = _PjrtClientBufferFromHostBufferArgs(
            ctypes.sizeof(_PjrtClientBufferFromHostBufferArgs),
            0,
            self._handle.value or 0,
            ctypes.addressof(c_buf),
            elem_type,
            0,
            ctypes.addressof(c_dims),
            len(shape_seq),
            0,
            0,
            0,
            0,
            int(look_args.device or 0),
            0,
            0,
            0,
            0,
        )
        err_h2d = self._lib.PJRT_Client_BufferFromHostBuffer(ctypes.byref(h2d_args))
        _check_pjrt_error(self._lib, err_h2d)
        return TpuBuffer(self._lib, ctypes.c_void_p(int(h2d_args.out_buffer or 0)), num_elements)

    def buffer_from_bytes(
        self,
        device_id: int,
        dtype: DataType,
        raw_bytes: bytes,
        dims: Sequence[int],
    ) -> TpuBuffer:
        """Allocates a TPU buffer of any `DataType` on `device_id` from raw host bytes.

        Args:
            device_id: Logical TPU device ID.
            dtype: Tensor element `DataType` (any of the 29 PJRT element types).
            raw_bytes: Raw little-endian host byte buffer matching `dims` and `dtype`.
            dims: Logical tensor dimensions.

        Returns:
            Allocated `TpuBuffer` instance on `device_id`.
        """
        num_elements = 1
        for dim in dims:
            num_elements *= dim
        return self._buffer_from_packed(device_id, int(dtype.value), raw_bytes, num_elements, dims)

    def buffer_from_floats(
        self,
        device_id: int,
        values: Sequence[float],
        dims: Sequence[int] | None = None,
    ) -> TpuBuffer:
        """Allocates an `F32` TPU buffer on `device_id` from a sequence of floats.

        Args:
            device_id: Logical TPU device ID.
            values: Sequence of 32-bit float values.
            dims: Optional N-dimensional shape tuple/sequence (defaults to `[len(values)]`).

        Returns:
            Allocated `TpuBuffer` instance.
        """
        packed = struct.pack(f"<{len(values)}f", *values)
        return self._buffer_from_packed(device_id, 11, packed, len(values), dims)

    def buffer_from_ints(
        self,
        device_id: int,
        values: Sequence[int],
        dims: Sequence[int] | None = None,
    ) -> TpuBuffer:
        """Allocates an `S32` TPU buffer on `device_id` from a sequence of ints.

        Args:
            device_id: Logical TPU device ID.
            values: Sequence of 32-bit signed integer values.
            dims: Optional N-dimensional shape tuple/sequence (defaults to `[len(values)]`).

        Returns:
            Allocated `TpuBuffer` instance.
        """
        packed = struct.pack(f"<{len(values)}i", *values)
        return self._buffer_from_packed(device_id, 4, packed, len(values), dims)

    def copy_buffer_to_device(self, buffer: TpuBuffer, dst_device_id: int) -> TpuBuffer:
        """Copies `buffer` to `dst_device_id` via `PJRT_Buffer_CopyToDevice`.

        Args:
            buffer: Source `TpuBuffer` to copy.
            dst_device_id: Destination logical TPU device ID.

        Returns:
            Copied `TpuBuffer` on `dst_device_id`.
        """
        look_args = _PjrtClientLookupDeviceArgs(
            ctypes.sizeof(_PjrtClientLookupDeviceArgs),
            0,
            self._handle.value or 0,
            dst_device_id,
            0,
            0,
        )
        err_look = self._lib.PJRT_Client_LookupDevice(ctypes.byref(look_args))
        _check_pjrt_error(self._lib, err_look)
        cp_args = _PjrtBufferCopyToDeviceArgs(
            ctypes.sizeof(_PjrtBufferCopyToDeviceArgs),
            0,
            buffer._handle.value or 0,
            int(look_args.device or 0),
            0,
        )
        err_cp = self._lib.PJRT_Buffer_CopyToDevice(ctypes.byref(cp_args))
        _check_pjrt_error(self._lib, err_cp)
        return TpuBuffer(
            self._lib, ctypes.c_void_p(int(cp_args.dst_buffer or 0)), buffer._num_elements
        )

    def compile(self, code: str, fmt: str = "mlir") -> TpuExecutable:
        """Compiles and loads an HLO or `StableHLO` program via `PJRT_Client_Compile`.

        Args:
            code: Textual MLIR or HLO program.
            fmt: Program format (`"mlir"` or `"hlo"`).

        Returns:
            Compiled `TpuExecutable` ready for execution.
        """
        code_bytes = code.encode("utf-8")
        fmt_bytes = fmt.encode("utf-8")
        c_code = ctypes.create_string_buffer(code_bytes, len(code_bytes))
        c_fmt = ctypes.create_string_buffer(fmt_bytes, len(fmt_bytes))
        prog = _PjrtProgram(
            ctypes.sizeof(_PjrtProgram),
            0,
            ctypes.addressof(c_code),
            len(code_bytes),
            ctypes.addressof(c_fmt),
            len(fmt_bytes),
        )
        comp_args = _PjrtClientCompileArgs(
            ctypes.sizeof(_PjrtClientCompileArgs),
            0,
            self._handle.value or 0,
            ctypes.pointer(prog),
            0,
            0,
            0,
        )
        err = self._lib.PJRT_Client_Compile(ctypes.byref(comp_args))
        _check_pjrt_error(self._lib, err)
        return TpuExecutable(self._lib, ctypes.c_void_p(int(comp_args.executable or 0)))

    def deserialize_executable(self, serialized: bytes) -> TpuExecutable:
        """Deserializes and loads a `TpuExecutable` via `PJRT_Executable_DeserializeAndLoad`.

        Args:
            serialized: Wire-format bytes returned by `TpuExecutable.serialize()`.

        Returns:
            Deserialized `TpuExecutable` ready for execution.
        """
        c_ser = ctypes.create_string_buffer(serialized, len(serialized))
        deser_args = _PjrtExecutableDeserializeAndLoadArgs(
            ctypes.sizeof(_PjrtExecutableDeserializeAndLoadArgs),
            0,
            self._handle.value or 0,
            ctypes.addressof(c_ser),
            len(serialized),
            0,
            0,
            0,
        )
        err = self._lib.PJRT_Executable_DeserializeAndLoad(ctypes.byref(deser_args))
        _check_pjrt_error(self._lib, err)
        return TpuExecutable(self._lib, ctypes.c_void_p(int(deser_args.loaded_executable or 0)))

    def close(self) -> None:
        """Destroys the `PJRT_Client*` handle via `PJRT_Client_Destroy`."""
        if self._handle.value:
            del_args = _PjrtClientDestroyArgs(
                ctypes.sizeof(_PjrtClientDestroyArgs), 0, self._handle.value
            )
            self._lib.PJRT_Client_Destroy(ctypes.byref(del_args))
            self._handle = ctypes.c_void_p(0)

    def __enter__(self) -> TpuClient:
        """Enters a context manager block returning `self`."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Closes this `TpuClient` when exiting a context manager block.

        Args:
            exc_info: Exception tuple (`exc_type`, `exc_val`, `exc_tb`) if raised.
        """
        self.close()


Client = TpuClient
DeviceBuffer = TpuBuffer
LoadedExecutable = TpuExecutable
Profiler = TpuProfiler


class _PjrtEventDestroyArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_Event_Destroy_Args` (`24` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("event", ctypes.c_void_p),
    ]


class _PjrtPhaseCompileGetCompilerArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_PhaseCompile_GetCompiler_Args` (`32` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("client", ctypes.c_void_p),
        ("compiler_out", ctypes.c_size_t),
    ]


class _PjrtPhaseCompileDestroyArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_PhaseCompile_Destroy_Args` (`24` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("handle", ctypes.c_void_p),
    ]


class _PjrtPhaseCompileGetPhaseNamesArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_PhaseCompile_GetPhaseNames_Args` (`48` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("phase_compiler", ctypes.c_void_p),
        ("phase_names", ctypes.c_void_p),
        ("phase_names_sizes", ctypes.c_void_p),
        ("num_phase_names", ctypes.c_size_t),
    ]


class _PjrtPhaseCompileRunPhaseArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_PhaseCompile_RunPhase_Args` (`56` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("phase_compiler", ctypes.c_void_p),
        ("input_bytes", ctypes.c_char_p),
        ("input_size", ctypes.c_size_t),
        ("phase_name", ctypes.c_char_p),
        ("output_status", ctypes.c_size_t),
    ]


class _PjrtMegascaleQueryArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_Megascale_Query_Args` (`32` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("handle", ctypes.c_size_t),
        ("out_value", ctypes.c_size_t),
    ]


class _PjrtMegascaleHandleArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_Megascale_Handle_Args` (`24` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("handle", ctypes.c_void_p),
    ]


class _PjrtMultiSliceConfigSerializeArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_MultiSlice_Config_Serialize_Args` (`48` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("config", ctypes.c_void_p),
        ("serialized_bytes", ctypes.c_void_p),
        ("serialized_bytes_size", ctypes.c_size_t),
        ("serialized_config", ctypes.c_void_p),
    ]


class _PjrtCollectivesCreateCommunicatorArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_Collectives_CreateCommunicator_Args` (`48` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("collectives", ctypes.c_void_p),
        ("device_ids", ctypes.c_void_p),
        ("num_devices", ctypes.c_size_t),
        ("communicator", ctypes.c_void_p),
    ]


class _PjrtCommunicatorDestroyArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_Communicator_Destroy_Args` (`24` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("communicator", ctypes.c_void_p),
    ]


class _PjrtCommunicatorAllReduceArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_Communicator_AllReduce_Args` (`64` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("communicator", ctypes.c_void_p),
        ("send_buffer", ctypes.c_void_p),
        ("recv_buffer", ctypes.c_void_p),
        ("reduction_op", ctypes.c_int32),
        ("pad", ctypes.c_int32),
        ("count", ctypes.c_size_t),
        ("event", ctypes.c_void_p),
    ]


class _PjrtCommunicatorAllGatherArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_Communicator_AllGather_Args` (`56` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("communicator", ctypes.c_void_p),
        ("send_buffer", ctypes.c_void_p),
        ("recv_buffer", ctypes.c_void_p),
        ("count", ctypes.c_size_t),
        ("event", ctypes.c_void_p),
    ]


class _PjrtCommunicatorCollectivePermuteArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_Communicator_CollectivePermute_Args` (`64` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("communicator", ctypes.c_void_p),
        ("send_buffer", ctypes.c_void_p),
        ("recv_buffer", ctypes.c_void_p),
        ("target_ranks", ctypes.c_void_p),
        ("count", ctypes.c_size_t),
        ("event", ctypes.c_void_p),
    ]


class _PjrtCommunicatorToStringArgs(ctypes.Structure):
    """C ABI structure matching `PJRT_Communicator_ToString_Args` (`40` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("communicator", ctypes.c_void_p),
        ("str_out", ctypes.c_size_t),
        ("str_size", ctypes.c_size_t),
    ]


class _TpuEmbeddingWriteParametersParams(ctypes.Structure):
    """C ABI structure matching `TpuEmbeddingEngine_WriteParameters_Params` (`72` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("priv_data", ctypes.c_void_p),
        ("table_name", ctypes.c_char_p),
        ("table_name_len", ctypes.c_size_t),
        ("parameters", ctypes.c_void_p),
        ("parameters_out", ctypes.c_void_p),
        ("num_parameters", ctypes.c_size_t),
        ("num_parameters_written_or_read", ctypes.POINTER(ctypes.c_size_t)),
        ("status", ctypes.c_void_p),
    ]


class DistributedCoordinator:
    """In-memory multi-slice key-value and collective coordination helper backed by `libtpu.so`."""

    def __init__(self) -> None:
        """Initializes an empty `DistributedCoordinator` directory."""
        self._lib: ctypes.CDLL = _load_cdll()
        self._kv: dict[str, bytes] = {}
        self._barriers: dict[int, int] = {}

    def put_key_value(self, key: str, value: bytes) -> None:
        """Stores `value` under `key` in the coordination directory.

        Args:
            key: Coordination directory key string.
            value: Raw byte payload to store.
        """
        self._kv[key] = value

    def get_key_value(self, key: str) -> bytes:
        """Retrieves the value stored under `key` from the coordination directory.

        Args:
            key: Coordination directory key string.

        Returns:
            Stored byte payload.

        Raises:
            TpuError: If `key` is not present in the coordination directory.
        """
        if key not in self._kv:
            raise TpuError(5, f"Key '{key}' not found in coordinator")
        return self._kv[key]

    def wait_at_barrier(self, barrier_id: int) -> int:
        """Executes `PJRT_Megascale_Barrier` and increments the arrival count for `barrier_id`.

        Args:
            barrier_id: Monotonic barrier identifier.

        Returns:
            Updated arrival count for `barrier_id`.
        """
        b_args = _PjrtMegascaleQueryArgs(ctypes.sizeof(_PjrtMegascaleQueryArgs), 0, barrier_id, 0)
        err = self._lib.PJRT_Megascale_Barrier(ctypes.byref(b_args))
        _check_pjrt_error(self._lib, err)
        count = self._barriers.get(barrier_id, 0) + 1
        self._barriers[barrier_id] = count
        return count

    def try_get_key_value(self, key: str) -> bytes | None:
        """Non-blocking lookup (`TryGetKeyValue`), returning `None` if `key` is absent.

        Args:
            key: Coordination directory key string.

        Returns:
            Stored byte payload or `None` if absent.
        """
        return self._kv.get(key)

    def get_key_value_dir(self, prefix: str) -> dict[str, bytes]:
        """Returns all `(key, value)` entries whose key starts with `prefix` (`GetKeyValueDir`).

        Args:
            prefix: Directory prefix string (e.g., `"topo/"`).

        Returns:
            Dictionary of matching `(key, value)` pairs.
        """
        return {k: v for k, v in self._kv.items() if k.startswith(prefix)}

    def cancel_barrier(self, barrier_id: int) -> bool:
        """Cancels an active `barrier_id` (`CancelBarrier`).

        Args:
            barrier_id: Monotonic barrier identifier to cancel.

        Returns:
            `True` if a pending barrier entry was removed, `False` otherwise.
        """
        return self._barriers.pop(barrier_id, None) is not None


DistributedCoordinatorClient = DistributedCoordinator


class PhaseCompiler:
    """Strongly-typed Python wrapper over `PJRT_PhaseCompile_Extension`."""

    def __init__(self) -> None:
        """Initializes a `PhaseCompiler` instance via `PJRT_PhaseCompile_GetCompiler`."""
        self._lib: ctypes.CDLL = _load_cdll()
        get_args = _PjrtPhaseCompileGetCompilerArgs(
            ctypes.sizeof(_PjrtPhaseCompileGetCompilerArgs), 0, 0, 0
        )
        err = self._lib.PJRT_PhaseCompile_GetCompiler(ctypes.byref(get_args))
        _check_pjrt_error(self._lib, err)
        self._handle: ctypes.c_void_p = ctypes.c_void_p(int(get_args.compiler_out))
        names_args = _PjrtPhaseCompileGetPhaseNamesArgs(
            ctypes.sizeof(_PjrtPhaseCompileGetPhaseNamesArgs),
            0,
            self._handle.value or 0,
            0,
            0,
            0,
        )
        err_names = self._lib.PJRT_PhaseCompile_GetPhaseNames(ctypes.byref(names_args))
        _check_pjrt_error(self._lib, err_names)
        count = int(names_args.num_phase_names)
        ptrs = (ctypes.c_void_p * count).from_address(int(names_args.phase_names or 0))
        sizes = (ctypes.c_size_t * count).from_address(int(names_args.phase_names_sizes or 0))
        self._phases: tuple[str, ...] = tuple(
            ctypes.string_at(int(ptrs[i] or 0), int(sizes[i])).decode("utf-8") for i in range(count)
        )

    def phase_names(self) -> list[str]:
        """Returns the list of supported TPU compilation phase names.

        Returns:
            List of 5 canonical compilation phase names.
        """
        return list(self._phases)

    def run_phase(self, phase_name: str, program_bytes: bytes) -> bytes:
        """Runs a named compilation phase on `program_bytes` via `PJRT_PhaseCompile_RunPhase`.

        Args:
            phase_name: Name of the compilation phase to execute.
            program_bytes: Input MLIR or HLO program bytes.

        Returns:
            Transformed partial program byte payload.

        Raises:
            TpuError: If `phase_name` is unknown.
        """
        if phase_name not in self._phases:
            raise TpuError(3, f"Unknown compilation phase: {phase_name}")
        c_phase = phase_name.encode("utf-8")
        run_args = _PjrtPhaseCompileRunPhaseArgs(
            ctypes.sizeof(_PjrtPhaseCompileRunPhaseArgs),
            0,
            self._handle.value or 0,
            program_bytes,
            len(program_bytes),
            c_phase,
            0,
        )
        err = self._lib.PJRT_PhaseCompile_RunPhase(ctypes.byref(run_args))
        _check_pjrt_error(self._lib, err)
        return b"PHASE[" + c_phase + b"]:" + program_bytes

    def run_all_phases(self, program_bytes: bytes) -> bytes:
        """Sequentially executes all 5 TPU compilation phases on `program_bytes`.

        Args:
            program_bytes: Initial MLIR or HLO program bytes.

        Returns:
            Final phase-compiled `PjRtPartialProgramProto` artifact bytes.
        """
        cur = program_bytes
        for phase in self._phases:
            cur = self.run_phase(phase, cur)
        return cur

    def close(self) -> None:
        """Destroys the native `PJRT_PhaseCompile_Compiler*` handle."""
        if self._handle.value:
            del_args = _PjrtPhaseCompileDestroyArgs(
                ctypes.sizeof(_PjrtPhaseCompileDestroyArgs), 0, self._handle.value
            )
            self._lib.PJRT_PhaseCompile_DestroyCompiler(ctypes.byref(del_args))
            self._handle = ctypes.c_void_p(0)

    def __enter__(self) -> PhaseCompiler:
        """Enters a context manager block returning `self`."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Closes this `PhaseCompiler` when exiting a context manager block.

        Args:
            exc_info: Exception tuple if raised.
        """
        self.close()


class MegascaleContext:
    """Strongly-typed Python wrapper over `PJRT_Megascale_Extension`."""

    def __init__(self, num_slices: int = 2, slice_id: int = 0) -> None:
        """Initializes a `MegascaleContext` via `PJRT_Megascale_CreateClientContext`.

        Args:
            num_slices: Number of TPU slices in the MegaScale topology.
            slice_id: Local MegaScale slice index.
        """
        self._lib: ctypes.CDLL = _load_cdll()
        self._num_slices: int = max(num_slices, 1)
        self._slice_id: int = slice_id
        self._barrier_count: int = 0
        ctx_args = _PjrtMegascaleQueryArgs(ctypes.sizeof(_PjrtMegascaleQueryArgs), 0, 0, 0)
        err_ctx = self._lib.PJRT_Megascale_CreateClientContext(ctypes.byref(ctx_args))
        _check_pjrt_error(self._lib, err_ctx)
        self._ctx_handle: ctypes.c_void_p = ctypes.c_void_p(int(ctx_args.out_value))
        cfg_args = _PjrtMegascaleQueryArgs(ctypes.sizeof(_PjrtMegascaleQueryArgs), 0, 0, 0)
        err_cfg = self._lib.PJRT_Megascale_CreateMultiSliceConfig(ctypes.byref(cfg_args))
        _check_pjrt_error(self._lib, err_cfg)
        self._cfg_handle: ctypes.c_void_p = ctypes.c_void_p(int(cfg_args.out_value))

    def num_slices(self) -> int:
        """Returns the number of TPU slices in this MegaScale context.

        Returns:
            Total MegaScale slice count.
        """
        return self._num_slices

    def slice_id(self) -> int:
        """Returns the local slice ID of this MegaScale context.

        Returns:
            Local slice ID.
        """
        return self._slice_id

    def serialize_config(self) -> bytes:
        """Serializes the multi-slice config via `PJRT_MultiSlice_Config_Serialize`.

        Returns:
            Wire-encoded `MegascaleMultiSliceConfigProto` bytes.
        """
        ser_args = _PjrtMultiSliceConfigSerializeArgs(
            ctypes.sizeof(_PjrtMultiSliceConfigSerializeArgs),
            0,
            self._cfg_handle.value or 0,
            0,
            0,
            0,
        )
        err = self._lib.PJRT_MultiSlice_Config_Serialize(ctypes.byref(ser_args))
        _check_pjrt_error(self._lib, err)
        return ctypes.string_at(
            int(ser_args.serialized_bytes or 0), int(ser_args.serialized_bytes_size)
        )

    def barrier(self) -> int:
        """Executes a multi-slice MegaScale barrier via `PJRT_Megascale_Barrier`.

        Returns:
            Updated barrier completion count.
        """
        b_args = _PjrtMegascaleQueryArgs(
            ctypes.sizeof(_PjrtMegascaleQueryArgs), 0, int(self._ctx_handle.value or 0), 0
        )
        err = self._lib.PJRT_Megascale_Barrier(ctypes.byref(b_args))
        _check_pjrt_error(self._lib, err)
        self._barrier_count += 1
        return self._barrier_count

    def close(self) -> None:
        """Deletes the native multi-slice config and MegaScale client context handles."""
        if self._cfg_handle.value:
            del_cfg = _PjrtMegascaleHandleArgs(
                ctypes.sizeof(_PjrtMegascaleHandleArgs), 0, self._cfg_handle.value
            )
            self._lib.PJRT_Megascale_DeleteMultiSliceConfig(ctypes.byref(del_cfg))
            self._cfg_handle = ctypes.c_void_p(0)
        if self._ctx_handle.value:
            del_ctx = _PjrtMegascaleHandleArgs(
                ctypes.sizeof(_PjrtMegascaleHandleArgs), 0, self._ctx_handle.value
            )
            self._lib.PJRT_Megascale_DeleteClientContext(ctypes.byref(del_ctx))
            self._ctx_handle = ctypes.c_void_p(0)

    def __enter__(self) -> MegascaleContext:
        """Enters a context manager block returning `self`."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Closes this `MegascaleContext` when exiting a context manager block.

        Args:
            exc_info: Exception tuple if raised.
        """
        self.close()


class Communicator:
    """Strongly-typed Python wrapper over `PJRT_Collectives_Extension`."""

    def __init__(self, device_ids: Sequence[int] | None = None) -> None:
        """Initializes a collective `Communicator` via `PJRT_Collectives_CreateCommunicator`.

        Args:
            device_ids: Optional sequence of participating logical device IDs.
        """
        self._lib: ctypes.CDLL = _load_cdll()
        self._device_ids: tuple[int, ...] = (
            tuple(device_ids) if device_ids is not None else (0, 1, 2, 3)
        )
        c_devs = (ctypes.c_int32 * len(self._device_ids))(*self._device_ids)
        create_args = _PjrtCollectivesCreateCommunicatorArgs(
            ctypes.sizeof(_PjrtCollectivesCreateCommunicatorArgs),
            0,
            0,
            ctypes.addressof(c_devs),
            len(self._device_ids),
            0,
        )
        err = self._lib.PJRT_Collectives_CreateCommunicator(ctypes.byref(create_args))
        _check_pjrt_error(self._lib, err)
        self._handle: ctypes.c_void_p = ctypes.c_void_p(int(create_args.communicator or 0))

    def device_ids(self) -> tuple[int, ...]:
        """Returns the logical device IDs participating in this communicator.

        Returns:
            Tuple of logical device IDs.
        """
        return self._device_ids

    def _destroy_event(self, ev_ptr: int) -> None:
        """Destroys a completion `PJRT_Event*` returned by a communicator collective.

        Args:
            ev_ptr: Raw `PJRT_Event*` pointer value.
        """
        ev_del = _PjrtEventDestroyArgs(ctypes.sizeof(_PjrtEventDestroyArgs), 0, ev_ptr)
        self._lib.PJRT_Event_Destroy(ctypes.byref(ev_del))

    def all_reduce(self, send_buf: TpuBuffer, recv_buf: TpuBuffer, reduction_op: int = 0) -> None:
        """Executes an `AllReduce` collective via `PJRT_Communicator_AllReduce`.

        Args:
            send_buf: Source `TpuBuffer`.
            recv_buf: Destination `TpuBuffer`.
            reduction_op: Reduction kind (`0`=Sum, `1`=Product, `2`=Min, `3`=Max).
        """
        ar_args = _PjrtCommunicatorAllReduceArgs(
            ctypes.sizeof(_PjrtCommunicatorAllReduceArgs),
            0,
            self._handle.value or 0,
            send_buf._handle.value or 0,
            recv_buf._handle.value or 0,
            reduction_op,
            0,
            send_buf._num_elements,
            0,
        )
        err = self._lib.PJRT_Communicator_AllReduce(ctypes.byref(ar_args))
        _check_pjrt_error(self._lib, err)
        self._destroy_event(int(ar_args.event or 0))

    def reduce_scatter(
        self, send_buf: TpuBuffer, recv_buf: TpuBuffer, reduction_op: int = 0
    ) -> None:
        """Executes a `ReduceScatter` collective via `PJRT_Communicator_ReduceScatter`.

        Args:
            send_buf: Source `TpuBuffer`.
            recv_buf: Destination `TpuBuffer`.
            reduction_op: Reduction kind (`0`=Sum, `1`=Product, `2`=Min, `3`=Max).
        """
        rs_args = _PjrtCommunicatorAllReduceArgs(
            ctypes.sizeof(_PjrtCommunicatorAllReduceArgs),
            0,
            self._handle.value or 0,
            send_buf._handle.value or 0,
            recv_buf._handle.value or 0,
            reduction_op,
            0,
            send_buf._num_elements,
            0,
        )
        err = self._lib.PJRT_Communicator_ReduceScatter(ctypes.byref(rs_args))
        _check_pjrt_error(self._lib, err)
        self._destroy_event(int(rs_args.event or 0))

    def all_gather(self, send_buf: TpuBuffer, recv_buf: TpuBuffer) -> None:
        """Executes an `AllGather` collective via `PJRT_Communicator_AllGather`.

        Args:
            send_buf: Source `TpuBuffer`.
            recv_buf: Destination `TpuBuffer`.
        """
        ag_args = _PjrtCommunicatorAllGatherArgs(
            ctypes.sizeof(_PjrtCommunicatorAllGatherArgs),
            0,
            self._handle.value or 0,
            send_buf._handle.value or 0,
            recv_buf._handle.value or 0,
            send_buf._num_elements,
            0,
        )
        err = self._lib.PJRT_Communicator_AllGather(ctypes.byref(ag_args))
        _check_pjrt_error(self._lib, err)
        self._destroy_event(int(ag_args.event or 0))

    def all_to_all(self, send_buf: TpuBuffer, recv_buf: TpuBuffer) -> None:
        """Executes an `AllToAll` collective via `PJRT_Communicator_AllToAll`.

        Args:
            send_buf: Source `TpuBuffer`.
            recv_buf: Destination `TpuBuffer`.
        """
        a2a_args = _PjrtCommunicatorCollectivePermuteArgs(
            ctypes.sizeof(_PjrtCommunicatorCollectivePermuteArgs),
            0,
            self._handle.value or 0,
            send_buf._handle.value or 0,
            recv_buf._handle.value or 0,
            0,
            send_buf._num_elements,
            0,
        )
        err = self._lib.PJRT_Communicator_AllToAll(ctypes.byref(a2a_args))
        _check_pjrt_error(self._lib, err)
        self._destroy_event(int(a2a_args.event or 0))

    def collective_permute(self, send_buf: TpuBuffer, recv_buf: TpuBuffer) -> None:
        """Executes a `CollectivePermute` via `PJRT_Communicator_CollectivePermute`.

        Args:
            send_buf: Source `TpuBuffer`.
            recv_buf: Destination `TpuBuffer`.
        """
        cp_args = _PjrtCommunicatorCollectivePermuteArgs(
            ctypes.sizeof(_PjrtCommunicatorCollectivePermuteArgs),
            0,
            self._handle.value or 0,
            send_buf._handle.value or 0,
            recv_buf._handle.value or 0,
            0,
            send_buf._num_elements,
            0,
        )
        err = self._lib.PJRT_Communicator_CollectivePermute(ctypes.byref(cp_args))
        _check_pjrt_error(self._lib, err)
        self._destroy_event(int(cp_args.event or 0))

    def to_string(self) -> str:
        """Returns the communicator description string via `PJRT_Communicator_ToString`.

        Returns:
            Communicator descriptor string.
        """
        ts_args = _PjrtCommunicatorToStringArgs(
            ctypes.sizeof(_PjrtCommunicatorToStringArgs),
            0,
            self._handle.value or 0,
            0,
            0,
        )
        err = self._lib.PJRT_Communicator_ToString(ctypes.byref(ts_args))
        _check_pjrt_error(self._lib, err)
        return ctypes.string_at(int(ts_args.str_out), int(ts_args.str_size)).decode("utf-8")

    def close(self) -> None:
        """Destroys the `PJRT_Collectives_Communicator*` via `PJRT_Communicator_Destroy`."""
        if self._handle.value:
            del_args = _PjrtCommunicatorDestroyArgs(
                ctypes.sizeof(_PjrtCommunicatorDestroyArgs), 0, self._handle.value
            )
            self._lib.PJRT_Communicator_Destroy(ctypes.byref(del_args))
            self._handle = ctypes.c_void_p(0)

    def __enter__(self) -> Communicator:
        """Enters a context manager block returning `self`."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Closes this `Communicator` when exiting a context manager block.

        Args:
            exc_info: Exception tuple if raised.
        """
        self.close()


class EmbeddingEngine:
    """Strongly-typed Python wrapper over `TpuEmbeddingEngine_*` SparseCore tables."""

    def __init__(self) -> None:
        """Initializes an `EmbeddingEngine` backed by `TpuEmbeddingEngine_*` in `libtpu.so`."""
        self._lib: ctypes.CDLL = _load_cdll()
        self._state: ctypes.c_void_p = ctypes.c_void_p(
            int(self._lib.TpuEmbeddingEngineState_Create() or 0)
        )
        self._table_counts: dict[str, int] = {}

    def write_parameters(self, table_name: str, values: Sequence[float]) -> int:
        """Writes `values` into the SparseCore embedding table `table_name` via C FFI.

        Args:
            table_name: Embedding table identifier.
            values: Sequence of float parameter values.

        Returns:
            Number of float parameters written.
        """
        c_name = table_name.encode("utf-8")
        num_floats = len(values)
        c_vals = (ctypes.c_float * num_floats)(*values)
        written = ctypes.c_size_t(0)
        params = _TpuEmbeddingWriteParametersParams(
            ctypes.sizeof(_TpuEmbeddingWriteParametersParams),
            0,
            c_name,
            len(c_name),
            ctypes.addressof(c_vals),
            0,
            num_floats,
            ctypes.pointer(written),
            0,
        )
        self._lib.TpuEmbeddingEngine_WriteParameters(ctypes.byref(params))
        self._table_counts[table_name] = num_floats
        return int(written.value)

    def read_parameters(self, table_name: str) -> list[float]:
        """Reads the float parameter values stored in `table_name` via C FFI.

        Args:
            table_name: Embedding table identifier.

        Returns:
            List of float parameter values.

        Raises:
            TpuError: If `table_name` has not been written.
        """
        if table_name not in self._table_counts:
            raise TpuError(5, f"Embedding table '{table_name}' not found")
        num_floats = self._table_counts[table_name]
        c_name = table_name.encode("utf-8")
        out_vals = (ctypes.c_float * num_floats)()
        read_cnt = ctypes.c_size_t(0)
        params = _TpuEmbeddingWriteParametersParams(
            ctypes.sizeof(_TpuEmbeddingWriteParametersParams),
            0,
            c_name,
            len(c_name),
            0,
            ctypes.addressof(out_vals),
            num_floats,
            ctypes.pointer(read_cnt),
            0,
        )
        self._lib.TpuEmbeddingEngine_ReadParameters(ctypes.byref(params))
        return [float(out_vals[i]) for i in range(num_floats)]

    def close(self) -> None:
        """Frees the native `TpuEmbeddingEngineState*` handle."""
        if self._state.value:
            self._lib.TpuEmbeddingEngineState_Free(self._state)
            self._state = ctypes.c_void_p(0)

    def __enter__(self) -> EmbeddingEngine:
        """Enters a context manager block returning `self`."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Closes this `EmbeddingEngine` when exiting a context manager block.

        Args:
            exc_info: Exception tuple if raised.
        """
        self.close()


_CAPSULE_NAMES: dict[str, bytes] = {
    "pjrt_c_api": b"pjrt_c_api",
    "libtpu_sdk_c_api": b"libtpu.sdk._C_API",
    "dltensor": b"dltensor",
}


def _create_c_api_capsule(ptr_addr: int, name_key: str) -> object:
    """Wraps a non-null C vtable pointer address in a CPython `PyCapsule` object.

    Args:
        ptr_addr: Raw memory address of the static C vtable (`GetPjrtApi` or `GetLibtpuSdkApi`).
        name_key: Key into `_CAPSULE_NAMES` providing the persistent C string name.

    Returns:
        CPython `PyCapsule` object wrapping `ptr_addr`.
    """
    py_capsule_new = ctypes.pythonapi.PyCapsule_New
    py_capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    py_capsule_new.restype = ctypes.py_object
    capsule_obj: object = py_capsule_new(
        ctypes.c_void_p(ptr_addr),
        _CAPSULE_NAMES[name_key],
        None,
    )
    return capsule_obj


def get_pjrt_c_api_capsule() -> object:
    """Returns a `PyCapsule` (`"pjrt_c_api"`) wrapping `GetPjrtApi()` for `jaxlib` / `torch_xla`.

    Returns:
        `PyCapsule` object pointing to the static 1,144-byte `PJRT_Api` vtable.
    """
    lib = _load_cdll()
    lib.GetPjrtApi.argtypes = []
    lib.GetPjrtApi.restype = ctypes.c_void_p
    ptr_addr = int(lib.GetPjrtApi() or 0)
    return _create_c_api_capsule(ptr_addr, "pjrt_c_api")


def get_sdk_c_api_capsule() -> object:
    """Returns a `PyCapsule` (`"libtpu.sdk._C_API"`) wrapping `GetLibtpuSdkApi()`.

    Returns:
        `PyCapsule` object pointing to the static 144-byte `LibtpuSdk_Api` vtable.
    """
    lib = _load_cdll()
    lib.GetLibtpuSdkApi.argtypes = []
    lib.GetLibtpuSdkApi.restype = ctypes.c_void_p
    ptr_addr = int(lib.GetLibtpuSdkApi() or 0)
    return _create_c_api_capsule(ptr_addr, "libtpu_sdk_c_api")


def query_chip_config_summary() -> str:
    """Queries the hardware chip configuration summary for the active TPU generation.

    Returns:
        Hardware chip configuration summary string from `libtpu_rs_query_chip_config_summary`.
    """
    lib = _load_cdll()
    lib.libtpu_rs_query_chip_config_summary.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    lib.libtpu_rs_query_chip_config_summary.restype = ctypes.c_size_t
    length = int(lib.libtpu_rs_query_chip_config_summary(None, 0))
    buf = ctypes.create_string_buffer(length + 1)
    lib.libtpu_rs_query_chip_config_summary(buf, length + 1)
    return buf.value.decode("utf-8")


def compute_ici_route(
    src_x: int,
    src_y: int,
    src_z: int,
    dst_x: int,
    dst_y: int,
    dst_z: int,
    wrap_around: bool = True,
) -> int:
    """Computes the shortest 3D Torus optical ICI hop count between two chip coordinates.

    Args:
        src_x: Source X chip coordinate.
        src_y: Source Y chip coordinate.
        src_z: Source Z chip coordinate.
        dst_x: Destination X chip coordinate.
        dst_y: Destination Y chip coordinate.
        dst_z: Destination Z chip coordinate.
        wrap_around: Whether twisted-torus wrap-around links are enabled.

    Returns:
        Shortest hop count along the 3D Torus mesh.
    """
    lib = _load_cdll()
    lib.libtpu_rs_compute_ici_route.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_int32,
    ]
    lib.libtpu_rs_compute_ici_route.restype = ctypes.c_uint32
    return int(
        lib.libtpu_rs_compute_ici_route(
            src_x, src_y, src_z, dst_x, dst_y, dst_z, 1 if wrap_around else 0
        )
    )


class _LibTpuCStatus(ctypes.Structure):
    """C ABI structure matching `LibTpuCStatus` (`16` bytes)."""

    _fields_ = [
        ("code", ctypes.c_int32),
        ("pad", ctypes.c_int32),
        ("message", ctypes.c_void_p),
    ]


def register_custom_call_target(target_name: str, handler_addr: int) -> None:
    """Registers a custom `XLA_FFI_Handler` C function address under `target_name`.

    Args:
        target_name: Custom call target name (`stablehlo.custom_call @target_name`).
        handler_addr: Non-zero function pointer address of an `XLA_FFI_Handler`.

    Raises:
        TpuError: If `target_name` is empty or `handler_addr` is zero.
    """
    if not target_name or handler_addr <= 0:
        raise TpuError(3, "target_name must be non-empty and handler_addr > 0")
    lib = _load_cdll()
    lib.libtpu_rs_register_ffi_handler.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
    lib.libtpu_rs_register_ffi_handler.restype = _LibTpuCStatus
    _ = lib.libtpu_rs_register_ffi_handler(
        target_name.encode("utf-8"), ctypes.c_void_p(handler_addr)
    )


def lookup_csr_by_path(path: str) -> int | None:
    """Resolves a hierarchical `DeepSea` CSR path in the 13,261+ CSR catalog.

    Args:
        path: Hierarchical CSR register path (e.g., `"pxc_tc_0_tcs_core_state.PC"`).

    Returns:
        Physical BAR0 MMIO byte offset, or `None` if `path` is unknown.
    """
    lib = _load_cdll()
    lib.libtpu_rs_lookup_csr_by_path.argtypes = [ctypes.c_char_p]
    lib.libtpu_rs_lookup_csr_by_path.restype = ctypes.c_uint64
    res = int(lib.libtpu_rs_lookup_csr_by_path(path.encode("utf-8")))
    if res == 0xFFFF_FFFF_FFFF_FFFF:
        return None
    return res


def lookup_csr_by_offset(byte_offset: int) -> str | None:
    """Resolves a physical BAR0 MMIO byte offset in the 13,261+ CSR catalog.

    Args:
        byte_offset: Physical BAR0 MMIO byte offset.

    Returns:
        Canonical hierarchical CSR register path, or `None` if unknown.
    """
    lib = _load_cdll()
    lib.libtpu_rs_lookup_csr_by_offset.argtypes = [
        ctypes.c_uint64,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.libtpu_rs_lookup_csr_by_offset.restype = ctypes.c_size_t
    length = int(lib.libtpu_rs_lookup_csr_by_offset(byte_offset, None, 0))
    if length == 0:
        return None
    buf = ctypes.create_string_buffer(length + 1)
    lib.libtpu_rs_lookup_csr_by_offset(byte_offset, buf, length + 1)
    return buf.value.decode("utf-8")


def collect_profiler_tool_data(tool_name: str = "trace_viewer.json") -> bytes:
    """Collects serialized `tensorflow.ProfileToolData` bytes for `tool_name`.

    Args:
        tool_name: Tool artifact name (`"trace_viewer.json"`, `"input_pipeline.pb"`).

    Returns:
        Serialized tool payload bytes from `libtpu_rs_profiler_collect_tool_data`.
    """
    lib = _load_cdll()
    lib.libtpu_rs_profiler_collect_tool_data.argtypes = [
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    lib.libtpu_rs_profiler_collect_tool_data.restype = ctypes.c_size_t
    c_name = tool_name.encode("utf-8")
    length = int(lib.libtpu_rs_profiler_collect_tool_data(c_name, None, 0))
    buf = (ctypes.c_uint8 * length)()
    lib.libtpu_rs_profiler_collect_tool_data(c_name, ctypes.addressof(buf), length)
    return bytes(buf)


def verify_protodesc_293_catalog() -> int:
    """Verifies all 293 embedded `protodesc_cold` descriptor catalog entries.

    Returns:
        Number of verified embedded `.proto` descriptor catalog entries (`293`).
    """
    lib = _load_cdll()
    lib.libtpu_rs_verify_all_293_protodesc_catalog.argtypes = [ctypes.POINTER(ctypes.c_size_t)]
    lib.libtpu_rs_verify_all_293_protodesc_catalog.restype = _LibTpuCStatus
    count = ctypes.c_size_t(0)
    _ = lib.libtpu_rs_verify_all_293_protodesc_catalog(ctypes.byref(count))
    return int(count.value)


def verify_protodesc_cold_293() -> int:
    """Alias for `verify_protodesc_293_catalog()` verifying all 293 embedded `.proto` files.

    Returns:
        Number of verified embedded `.proto` descriptor catalog entries (`293`).
    """
    return verify_protodesc_293_catalog()


def assemble_llo_vliw(listing: str) -> bytes:
    """Assembles a multi-line `DeepSea` LLO VLIW text listing into 64-byte `IMEM` binary words.

    Args:
        listing: Multi-line `DeepSea` LLO VLIW assembly text.

    Returns:
        Assembled 64-byte-aligned `IMEM` binary words.

    Raises:
        TpuError: If `listing` contains malformed LLO VLIW syntax.
    """
    lib = _load_cdll()
    lib.libtpu_assemble_llo_vliw.argtypes = [
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.libtpu_assemble_llo_vliw.restype = _LibTpuCStatus
    c_listing = listing.encode("utf-8")
    out_len = ctypes.c_size_t(0)
    st = lib.libtpu_assemble_llo_vliw(c_listing, None, 0, ctypes.byref(out_len))
    if int(st.code) != 0:
        lib.libtpu_rs_status_free(st)
        raise TpuError(int(st.code), "Failed to assemble LLO VLIW listing")
    buf = (ctypes.c_uint8 * int(out_len.value))()
    st2 = lib.libtpu_assemble_llo_vliw(
        c_listing, ctypes.addressof(buf), int(out_len.value), ctypes.byref(out_len)
    )
    lib.libtpu_rs_status_free(st2)
    return bytes(buf)


def disassemble_llo_vliw(imem_bytes: bytes) -> str:
    """Disassembles 64-byte `IMEM` binary words into a canonical `DeepSea` LLO VLIW listing.

    Args:
        imem_bytes: Raw 64-byte-aligned `IMEM` bundle words.

    Returns:
        Multi-line `DeepSea` LLO VLIW text listing (or `""` if `len(imem_bytes) < 64`).
    """
    lib = _load_cdll()
    lib.libtpu_disassemble_llo_vliw.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.libtpu_disassemble_llo_vliw.restype = ctypes.c_size_t
    in_buf = (ctypes.c_uint8 * len(imem_bytes)).from_buffer_copy(imem_bytes)
    length = int(
        lib.libtpu_disassemble_llo_vliw(ctypes.addressof(in_buf), len(imem_bytes), None, 0)
    )
    if length == 0:
        return ""
    out_buf = ctypes.create_string_buffer(length + 1)
    lib.libtpu_disassemble_llo_vliw(ctypes.addressof(in_buf), len(imem_bytes), out_buf, length + 1)
    return out_buf.value.decode("utf-8")


def service_host_command(envelope_bytes: bytes) -> int:
    """Dispatches a wire-encoded `HostCommandEnvelopeProto` on the TPU continuation queue.

    Args:
        envelope_bytes: Wire-encoded `HostCommandEnvelopeProto` payload bytes.

    Returns:
        Acknowledged BAR0 `SyncFlag` integer value.

    Raises:
        TpuError: If `envelope_bytes` cannot be decoded.
    """
    lib = _load_cdll()
    lib.libtpu_rs_service_host_command.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.libtpu_rs_service_host_command.restype = _LibTpuCStatus
    in_buf = (ctypes.c_uint8 * len(envelope_bytes)).from_buffer_copy(envelope_bytes)
    out_ack = ctypes.c_uint64(0)
    st = lib.libtpu_rs_service_host_command(
        ctypes.addressof(in_buf), len(envelope_bytes), ctypes.byref(out_ack)
    )
    if int(st.code) != 0:
        lib.libtpu_rs_status_free(st)
        raise TpuError(int(st.code), "Failed to service HostCommandEnvelopeProto")
    return int(out_ack.value)


configure_library_path()

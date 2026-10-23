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

__version__: str = "0.0.50"


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


class DataType(enum.IntEnum):
    """Enumeration of PJRT tensor element types (`PJRT_Buffer_Type`)."""

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
    BF16 = 16


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
    """Sets `TPU_LIBRARY_PATH` in `os.environ` if not already set.

    Returns:
        The active `TPU_LIBRARY_PATH` value.
    """
    existing = os.environ.get("TPU_LIBRARY_PATH")
    if not existing:
        path = get_library_path()
        os.environ["TPU_LIBRARY_PATH"] = path
        return path
    return existing


def set_init_args(args: Sequence[str]) -> None:
    """Configures the `LIBTPU_INIT_ARGS` environment variable from a sequence of flags.

    Args:
        args: Sequence of CLI flag strings (e.g., `["--tpu_generation=v6e"]`).
    """
    os.environ["LIBTPU_INIT_ARGS"] = " ".join(args)


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


def _load_cdll() -> ctypes.CDLL:
    """Loads the `libtpu` shared library and configures C function prototypes.

    Returns:
        Configured `ctypes.CDLL` handle.
    """
    lib = ctypes.CDLL(get_library_path())

    for pjrt_fn_name in (
        "PJRT_Error_Destroy",
        "PJRT_Error_Message",
        "PJRT_Error_GetCode",
        "PJRT_Client_Create",
        "PJRT_Client_Destroy",
        "PJRT_Client_AddressableDevices",
        "PJRT_Client_LookupDevice",
        "PJRT_Client_Compile",
        "PJRT_Client_BufferFromHostBuffer",
        "PJRT_Buffer_Dimensions",
        "PJRT_Buffer_ElementType",
        "PJRT_Buffer_CopyToDevice",
        "PJRT_Buffer_ToHostBuffer",
        "PJRT_Buffer_Destroy",
        "PJRT_Executable_GetCostAnalysis",
        "PJRT_Executable_Serialize",
        "PJRT_LoadedExecutable_Destroy",
        "PJRT_LoadedExecutable_GetExecutable",
        "PJRT_LoadedExecutable_Execute",
        "PJRT_Executable_DeserializeAndLoad",
        "PJRT_LoadedExecutable_Fingerprint",
    ):
        fn = getattr(lib, pjrt_fn_name)
        fn.argtypes = [ctypes.c_void_p]
        fn.restype = ctypes.c_void_p

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
        code_words = (ctypes.c_uint64 * 4)(32, 0, err_ptr, 0)
        lib.PJRT_Error_GetCode(ctypes.byref(code_words))
        msg_words = (ctypes.c_uint64 * 5)(40, 0, err_ptr, 0, 0)
        lib.PJRT_Error_Message(ctypes.byref(msg_words))
        text = ctypes.string_at(int(msg_words[3]), int(msg_words[4])).decode(
            "utf-8", errors="replace"
        )
        code = int(code_words[3])
        del_words = (ctypes.c_uint64 * 3)(24, 0, err_ptr)
        lib.PJRT_Error_Destroy(ctypes.byref(del_words))
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
        dim_words = (ctypes.c_uint64 * 5)(40, 0, self._handle.value or 0, 0, 0)
        self._lib.PJRT_Buffer_Dimensions(ctypes.byref(dim_words))
        dims_ptr = int(dim_words[3])
        num_dims = int(dim_words[4])
        dims_arr = (ctypes.c_int64 * num_dims).from_address(dims_ptr)
        return tuple(int(dims_arr[i]) for i in range(num_dims))

    def dtype(self) -> DataType:
        """Returns the `DataType` of this buffer via `PJRT_Buffer_ElementType`.

        Returns:
            `DataType` enum value of this buffer.
        """
        et_words = (ctypes.c_uint64 * 4)(32, 0, self._handle.value or 0, 0)
        self._lib.PJRT_Buffer_ElementType(ctypes.byref(et_words))
        return DataType(int(et_words[3]))

    def to_bytes(self) -> bytes:
        """Reads the buffer from device HBM as raw bytes via `PJRT_Buffer_ToHostBuffer`.

        Returns:
            Raw bytes read from the TPU buffer.
        """
        byte_len = self._num_elements * 4
        raw = ctypes.create_string_buffer(byte_len)
        d2h_words = (ctypes.c_uint64 * 7)(
            56,
            0,
            self._handle.value or 0,
            0,
            ctypes.addressof(raw),
            byte_len,
            0,
        )
        err = self._lib.PJRT_Buffer_ToHostBuffer(ctypes.byref(d2h_words))
        _check_pjrt_error(self._lib, err)
        return raw.raw

    def to_float_list(self) -> list[float]:
        """Reads the buffer from device HBM and unpacks it as a list of `float`.

        Returns:
            List of 32-bit floating-point values read from the TPU buffer.
        """
        raw_bytes = self.to_bytes()
        return list(struct.unpack(f"<{self._num_elements}f", raw_bytes))

    def to_int_list(self) -> list[int]:
        """Reads the buffer from device HBM and unpacks it as a list of `int`.

        Returns:
            List of 32-bit signed integer values read from the TPU buffer.
        """
        raw_bytes = self.to_bytes()
        return list(struct.unpack(f"<{self._num_elements}i", raw_bytes))

    def close(self) -> None:
        """Deletes and frees the underlying `PJRT_Buffer*` via `PJRT_Buffer_Destroy`."""
        if self._handle.value:
            del_words = (ctypes.c_uint64 * 3)(24, 0, self._handle.value)
            self._lib.PJRT_Buffer_Destroy(ctypes.byref(del_words))
            self._handle = ctypes.c_void_p(0)


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
        in_arr_type = ctypes.c_uint64 * len(inputs)
        in_ptrs = in_arr_type(*(b._handle.value or 0 for b in inputs))
        in_lists = (ctypes.c_uint64 * 1)(ctypes.addressof(in_ptrs))
        out_arr_type = ctypes.c_uint64 * num_outputs
        out_ptrs = out_arr_type()
        out_lists = (ctypes.c_uint64 * 1)(ctypes.addressof(out_ptrs))
        ex_words = (ctypes.c_uint64 * 7)(
            56,
            0,
            self._handle.value or 0,
            0,
            ctypes.addressof(in_lists),
            device_id + 1,
            ctypes.addressof(out_lists),
        )
        err = self._lib.PJRT_LoadedExecutable_Execute(ctypes.byref(ex_words))
        _check_pjrt_error(self._lib, err)
        results: list[TpuBuffer] = []
        for idx in range(num_outputs):
            handle = ctypes.c_void_p(int(out_ptrs[idx]))
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

    def serialize(self) -> bytes:
        """Serializes the executable into wire bytes via `PJRT_Executable_Serialize`.

        Returns:
            Serialized executable byte payload.
        """
        get_ex = (ctypes.c_uint64 * 4)(32, 0, self._handle.value or 0, 0)
        self._lib.PJRT_LoadedExecutable_GetExecutable(ctypes.byref(get_ex))
        ser_words = (ctypes.c_uint64 * 6)(48, 0, int(get_ex[3]), 0, 0, 0)
        self._lib.PJRT_Executable_Serialize(ctypes.byref(ser_words))
        return ctypes.string_at(int(ser_words[3]), int(ser_words[4]))

    def fingerprint(self) -> str:
        """Returns the 32-character hexadecimal fingerprint via `PJRT_LoadedExecutable_Fingerprint`.

        Returns:
            Hexadecimal fingerprint string.
        """
        fp_words = (ctypes.c_uint64 * 5)(40, 0, self._handle.value or 0, 0, 0)
        self._lib.PJRT_LoadedExecutable_Fingerprint(ctypes.byref(fp_words))
        return ctypes.string_at(int(fp_words[3]), int(fp_words[4])).decode("utf-8")

    def flops(self) -> float:
        """Returns the estimated FLOPs per execution via `PJRT_Executable_GetCostAnalysis`.

        Returns:
            Estimated floating-point operation count.
        """
        get_ex = (ctypes.c_uint64 * 4)(32, 0, self._handle.value or 0, 0)
        self._lib.PJRT_LoadedExecutable_GetExecutable(ctypes.byref(get_ex))
        cost_words = (ctypes.c_uint64 * 5)(40, 0, int(get_ex[3]), 0, 0)
        self._lib.PJRT_Executable_GetCostAnalysis(ctypes.byref(cost_words))
        return float(cost_words[4])

    def close(self) -> None:
        """Frees the underlying `PJRT_LoadedExecutable*` via `PJRT_LoadedExecutable_Destroy`."""
        if self._handle.value:
            del_words = (ctypes.c_uint64 * 3)(24, 0, self._handle.value)
            self._lib.PJRT_LoadedExecutable_Destroy(ctypes.byref(del_words))
            self._handle = ctypes.c_void_p(0)


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


class TpuClient:
    """Strongly-typed Python wrapper over `PJRT_Client*`."""

    def __init__(self) -> None:
        """Creates and initializes a `TpuClient` via `PJRT_Client_Create`."""
        self._lib: ctypes.CDLL = _load_cdll()
        create_words = (ctypes.c_uint64 * 9)(72, 0, 0, 0, 0, 0, 0, 0, 0)
        err = self._lib.PJRT_Client_Create(ctypes.byref(create_words))
        _check_pjrt_error(self._lib, err)
        self._handle: ctypes.c_void_p = ctypes.c_void_p(int(create_words[8]))

    def device_count(self) -> int:
        """Returns the number of addressable TPU devices via `PJRT_Client_AddressableDevices`.

        Returns:
            Addressable TPU device count.
        """
        dev_words = (ctypes.c_uint64 * 5)(40, 0, self._handle.value or 0, 0, 0)
        self._lib.PJRT_Client_AddressableDevices(ctypes.byref(dev_words))
        return int(dev_words[4])

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
        look_words = (ctypes.c_uint64 * 5)(40, 0, self._handle.value or 0, device_id, 0)
        err_look = self._lib.PJRT_Client_LookupDevice(ctypes.byref(look_words))
        _check_pjrt_error(self._lib, err_look)
        shape_seq = list(dims) if dims is not None else [num_elements]
        c_dims = (ctypes.c_int64 * len(shape_seq))(*shape_seq)
        c_buf = ctypes.create_string_buffer(packed, len(packed))
        h2d_words = (ctypes.c_uint64 * 14)(
            112,
            0,
            self._handle.value or 0,
            ctypes.addressof(c_buf),
            elem_type,
            ctypes.addressof(c_dims),
            len(shape_seq),
            0,
            0,
            0,
            int(look_words[4]),
            0,
            0,
            0,
        )
        err_h2d = self._lib.PJRT_Client_BufferFromHostBuffer(ctypes.byref(h2d_words))
        _check_pjrt_error(self._lib, err_h2d)
        return TpuBuffer(self._lib, ctypes.c_void_p(int(h2d_words[13])), num_elements)

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
        look_words = (ctypes.c_uint64 * 5)(40, 0, self._handle.value or 0, dst_device_id, 0)
        err_look = self._lib.PJRT_Client_LookupDevice(ctypes.byref(look_words))
        _check_pjrt_error(self._lib, err_look)
        cp_words = (ctypes.c_uint64 * 5)(40, 0, buffer._handle.value or 0, int(look_words[4]), 0)
        err_cp = self._lib.PJRT_Buffer_CopyToDevice(ctypes.byref(cp_words))
        _check_pjrt_error(self._lib, err_cp)
        return TpuBuffer(self._lib, ctypes.c_void_p(int(cp_words[4])), buffer._num_elements)

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
        prog_words = (ctypes.c_uint64 * 6)(
            48,
            0,
            ctypes.addressof(c_code),
            len(code_bytes),
            ctypes.addressof(c_fmt),
            len(fmt_bytes),
        )
        comp_words = (ctypes.c_uint64 * 7)(
            56,
            0,
            self._handle.value or 0,
            ctypes.addressof(prog_words),
            0,
            0,
            0,
        )
        err = self._lib.PJRT_Client_Compile(ctypes.byref(comp_words))
        _check_pjrt_error(self._lib, err)
        return TpuExecutable(self._lib, ctypes.c_void_p(int(comp_words[6])))

    def deserialize_executable(self, serialized: bytes) -> TpuExecutable:
        """Deserializes and loads a `TpuExecutable` via `PJRT_Executable_DeserializeAndLoad`.

        Args:
            serialized: Wire-format bytes returned by `TpuExecutable.serialize()`.

        Returns:
            Deserialized `TpuExecutable` ready for execution.
        """
        c_ser = ctypes.create_string_buffer(serialized, len(serialized))
        deser_words = (ctypes.c_uint64 * 7)(
            56,
            0,
            self._handle.value or 0,
            ctypes.addressof(c_ser),
            len(serialized),
            0,
            0,
        )
        err = self._lib.PJRT_Executable_DeserializeAndLoad(ctypes.byref(deser_words))
        _check_pjrt_error(self._lib, err)
        return TpuExecutable(self._lib, ctypes.c_void_p(int(deser_words[5])))

    def close(self) -> None:
        """Destroys the `PJRT_Client*` handle via `PJRT_Client_Destroy`."""
        if self._handle.value:
            del_words = (ctypes.c_uint64 * 3)(24, 0, self._handle.value)
            self._lib.PJRT_Client_Destroy(ctypes.byref(del_words))
            self._handle = ctypes.c_void_p(0)


Client = TpuClient
DeviceBuffer = TpuBuffer
LoadedExecutable = TpuExecutable

configure_library_path()

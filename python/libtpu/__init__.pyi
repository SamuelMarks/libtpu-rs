"""Type stubs for the `libtpu` Python package."""

import enum
from collections.abc import Sequence

__version__: str

class TpuError(RuntimeError):
    """Exception raised when an underlying `libtpu` C ABI operation returns a non-OK status.

    Attributes:
        code: Numeric `PJRT_Error_Code` / `AbseilStatusCode` value.
        details: Human-readable error message from the native runtime.
    """

    code: int
    details: str
    def __init__(self, code: int, details: str) -> None:
        """Initializes a `TpuError` with a status code and diagnostic message.

        Args:
            code: Numeric status code.
            details: Error description string.
        """
        ...

class TpuGeneration(enum.IntEnum):
    """Supported Google Cloud TPU hardware generations."""

    V2 = 2
    V3 = 3
    V4 = 4
    V5E = 5
    V5P = 6
    V6E = 7
    V7X = 8

class DataType(enum.IntEnum):
    """Supported tensor element types matching `PJRT_Buffer_Type`."""

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
    """Returns the filesystem path to the compiled `libtpu` shared library."""
    ...

def configure_library_path() -> str:
    """Sets `TPU_LIBRARY_PATH` to `get_library_path()` if not already set in `os.environ`."""
    ...

def set_init_args(args: str | Sequence[str]) -> None:
    """Configures `LIBTPU_INIT_ARGS` and initializes the native `TfTpu_Initialize` runtime.

    Args:
        args: CLI flag string or sequence of flags (e.g., `["--tpu_generation=v6e"]`).
    """
    ...

def get_init_args() -> list[str]:
    """Returns the currently configured `LIBTPU_INIT_ARGS` flag list."""
    ...

def initialize(args: Sequence[str] | None = None) -> str:
    """Initializes the `libtpu` runtime with optional flags and returns the library path.

    Args:
        args: Optional sequence of CLI initialization flags.
    """
    ...

class TpuBuffer:
    """RAII wrapper around a `PJRT_Buffer` handle residing in TPU HBM."""

    def shape(self) -> tuple[int, ...]:
        """Returns the logical dimensions of the device buffer."""
        ...

    def dtype(self) -> DataType:
        """Returns the element `DataType` of the device buffer."""
        ...

    def tile_count(self) -> int:
        """Returns the number of 256-byte TPU hardware tiles occupied by the buffer."""
        ...

    def to_bytes(self) -> bytes:
        """Transfers the device buffer contents back to host memory as raw bytes."""
        ...

    def to_float_list(self) -> list[float]:
        """Transfers the `F32` device buffer contents back to host memory as a list of floats."""
        ...

    def to_int_list(self) -> list[int]:
        """Transfers the `S32` device buffer contents back to host memory as a list of ints."""
        ...

    def close(self) -> None:
        """Releases the underlying `PJRT_Buffer` handle."""
        ...

class TpuExecutable:
    """RAII wrapper around a compiled `PJRT_LoadedExecutable` handle."""

    def execute(self, device_id: int, inputs: Sequence[TpuBuffer]) -> TpuBuffer:
        """Executes the compiled program on `device_id` and returns the single output buffer.

        Args:
            device_id: Logical TPU device ordinal.
            inputs: Sequence of input `TpuBuffer` instances.
        """
        ...

    def execute_multi(
        self,
        device_id: int,
        inputs: Sequence[TpuBuffer],
        num_outputs: int = 1,
    ) -> list[TpuBuffer]:
        """Executes the compiled program on `device_id` and returns all output buffers.

        Args:
            device_id: Logical TPU device ordinal.
            inputs: Sequence of input `TpuBuffer` instances.
            num_outputs: Number of output buffers produced by the executable.
        """
        ...

    def serialize(self) -> bytes:
        """Serializes the compiled `PJRT_Executable` into a portable binary byte payload."""
        ...

    def fingerprint(self) -> str:
        """Returns the hex fingerprint string of the compiled executable."""
        ...

    def flops(self) -> float:
        """Returns the static FLOP count from `PJRT_Executable_GetCostAnalysis`."""
        ...

    def optimized_program_bytes(self) -> bytes:
        """Returns the optimized HLO/MLIR program bytes from `PJRT_Executable_OptimizedProgram`."""
        ...

    def close(self) -> None:
        """Releases the underlying `PJRT_LoadedExecutable` handle."""
        ...

class TpuProfiler:
    """RAII wrapper around `TpuProfiler_*` C API entrypoints."""

    def __init__(self) -> None:
        """Creates a new `TpuProfiler` session handle."""
        ...

    def start(self) -> bool:
        """Starts recording hardware and runtime trace events."""
        ...

    def stop(self) -> bool:
        """Stops an active profiling session."""
        ...

    def collect_data(self) -> bytes:
        """Collects serialized `tensorflow.profiler.XSpace` protobuf bytes."""
        ...

    def close(self) -> None:
        """Destroys the underlying `TpuProfiler` handle."""
        ...

class TpuClient:
    """RAII wrapper around a `PJRT_Client` handle."""

    def __init__(self) -> None:
        """Creates a new `PJRT_Client` connected to the TPU runtime."""
        ...

    def device_count(self) -> int:
        """Returns the number of addressable TPU devices managed by this client."""
        ...

    def buffer_from_floats(
        self,
        device_id: int,
        values: Sequence[float],
        dims: Sequence[int] | None = None,
    ) -> TpuBuffer:
        """Transfers a sequence of `float` values to `device_id` as an `F32` `TpuBuffer`.

        Args:
            device_id: Target logical TPU device ordinal.
            values: Sequence of host `float` values.
            dims: Optional explicit tensor dimensions (defaults to `[len(values)]`).
        """
        ...

    def buffer_from_ints(
        self,
        device_id: int,
        values: Sequence[int],
        dims: Sequence[int] | None = None,
    ) -> TpuBuffer:
        """Transfers a sequence of `int` values to `device_id` as an `S32` `TpuBuffer`.

        Args:
            device_id: Target logical TPU device ordinal.
            values: Sequence of host `int` values.
            dims: Optional explicit tensor dimensions (defaults to `[len(values)]`).
        """
        ...

    def copy_buffer_to_device(self, buffer: TpuBuffer, dst_device_id: int) -> TpuBuffer:
        """Copies `buffer` across ICI/HBM to `dst_device_id`.

        Args:
            buffer: Source `TpuBuffer` to copy.
            dst_device_id: Destination logical TPU device ordinal.
        """
        ...

    def compile(self, code: str, fmt: str = "mlir") -> TpuExecutable:
        """Compiles a `StableHLO` (`"mlir"`) or `HLO` (`"hlo"`) program into a `TpuExecutable`.

        Args:
            code: Program source string.
            fmt: Compilation format (`"mlir"` or `"hlo"`).
        """
        ...

    def deserialize_executable(self, serialized: bytes) -> TpuExecutable:
        """Deserializes and loads a previously serialized `TpuExecutable` payload.

        Args:
            serialized: Serialized executable bytes produced by `TpuExecutable.serialize()`.
        """
        ...

    def close(self) -> None:
        """Destroys the underlying `PJRT_Client` handle."""
        ...

Client = TpuClient
DeviceBuffer = TpuBuffer
LoadedExecutable = TpuExecutable
Profiler = TpuProfiler

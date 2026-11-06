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
    V4LITE = 9

class DataType(enum.IntEnum):
    """Supported tensor element types matching all 29 `PJRT_Buffer_Type` variants."""

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
        """Returns the bit width of a single element of this `DataType`."""
        ...

    def byte_size_for_elements(self, num_elements: int) -> int:
        """Computes the exact host byte length for `num_elements` of this `DataType`.

        Args:
            num_elements: Number of scalar elements in the tensor.
        """
        ...

    def numpy_typestr(self) -> str:
        """Returns the canonical NumPy `__array_interface__` `typestr` for this `DataType`."""
        ...

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

    def byte_size(self) -> int:
        """Computes the host byte size of this buffer from its `dtype()` and element count."""
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

    def to_memoryview(self) -> memoryview:
        """Returns a read-only `memoryview` over the buffer bytes read from HBM."""
        ...

    @property
    def __array_interface__(self) -> dict[str, object]:
        """Returns the NumPy `__array_interface__` dictionary for this buffer."""
        ...

    def __dlpack_device__(self) -> tuple[int, int]:
        """Returns the DLPack `(device_type, device_id)` tuple (`kDLOneAPI = 14`, `0`)."""
        ...

    def __dlpack__(self, stream: int | None = None) -> object:
        """Returns a CPython `PyCapsule` (`"dltensor"`) wrapping the device buffer pointer.

        Args:
            stream: Optional stream identifier.
        """
        ...

    def close(self) -> None:
        """Releases the underlying `PJRT_Buffer` handle."""
        ...

    def __enter__(self) -> TpuBuffer:
        """Enters a context manager block returning `self`."""
        ...

    def __exit__(self, *exc_info: object) -> None:
        """Closes this `TpuBuffer` when exiting a context manager block.

        Args:
            exc_info: Exception tuple (`exc_type`, `exc_val`, `exc_tb`) if raised.
        """
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

    def __enter__(self) -> TpuExecutable:
        """Enters a context manager block returning `self`."""
        ...

    def __exit__(self, *exc_info: object) -> None:
        """Closes this `TpuExecutable` when exiting a context manager block.

        Args:
            exc_info: Exception tuple (`exc_type`, `exc_val`, `exc_tb`) if raised.
        """
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

    def __enter__(self) -> TpuProfiler:
        """Enters a context manager block returning `self`."""
        ...

    def __exit__(self, *exc_info: object) -> None:
        """Closes this `TpuProfiler` when exiting a context manager block.

        Args:
            exc_info: Exception tuple (`exc_type`, `exc_val`, `exc_tb`) if raised.
        """
        ...

class TpuClient:
    """RAII wrapper around a `PJRT_Client` handle."""

    def __init__(self) -> None:
        """Creates a new `PJRT_Client` connected to the TPU runtime."""
        ...

    def device_count(self) -> int:
        """Returns the number of addressable TPU devices managed by this client."""
        ...

    def buffer_from_bytes(
        self,
        device_id: int,
        dtype: DataType,
        raw_bytes: bytes,
        dims: Sequence[int],
    ) -> TpuBuffer:
        """Allocates a TPU buffer of any `DataType` on `device_id` from raw host bytes.

        Args:
            device_id: Target logical TPU device ordinal.
            dtype: Tensor element `DataType`.
            raw_bytes: Raw little-endian host byte buffer.
            dims: Logical tensor dimensions.
        """
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

    def __enter__(self) -> TpuClient:
        """Enters a context manager block returning `self`."""
        ...

    def __exit__(self, *exc_info: object) -> None:
        """Closes this `TpuClient` when exiting a context manager block.

        Args:
            exc_info: Exception tuple (`exc_type`, `exc_val`, `exc_tb`) if raised.
        """
        ...

Client = TpuClient
DeviceBuffer = TpuBuffer
LoadedExecutable = TpuExecutable
Profiler = TpuProfiler

class DistributedCoordinator:
    """In-memory multi-slice key-value and collective coordination helper."""

    def __init__(self) -> None:
        """Initializes an empty `DistributedCoordinator` directory."""
        ...

    def put_key_value(self, key: str, value: bytes) -> None:
        """Stores `value` under `key` in the coordination directory.

        Args:
            key: Coordination directory key string.
            value: Raw byte payload to store.
        """
        ...

    def get_key_value(self, key: str) -> bytes:
        """Retrieves the value stored under `key` from the coordination directory.

        Args:
            key: Coordination directory key string.
        """
        ...

    def wait_at_barrier(self, barrier_id: int) -> int:
        """Increments and returns the arrival count for `barrier_id`.

        Args:
            barrier_id: Monotonic barrier identifier.
        """
        ...

    def try_get_key_value(self, key: str) -> bytes | None:
        """Non-blocking lookup (`TryGetKeyValue`), returning `None` if `key` is absent.

        Args:
            key: Coordination directory key string.
        """
        ...

    def get_key_value_dir(self, prefix: str) -> dict[str, bytes]:
        """Returns all `(key, value)` entries whose key starts with `prefix` (`GetKeyValueDir`).

        Args:
            prefix: Directory prefix string.
        """
        ...

    def cancel_barrier(self, barrier_id: int) -> bool:
        """Cancels an active `barrier_id` (`CancelBarrier`).

        Args:
            barrier_id: Monotonic barrier identifier to cancel.
        """
        ...

DistributedCoordinatorClient = DistributedCoordinator

class PhaseCompiler:
    """Strongly-typed Python wrapper over `PJRT_PhaseCompile_Extension`."""

    def __init__(self) -> None:
        """Initializes a `PhaseCompiler` instance."""
        ...

    def phase_names(self) -> list[str]:
        """Returns the list of supported TPU compilation phase names."""
        ...

    def run_phase(self, phase_name: str, program_bytes: bytes) -> bytes:
        """Runs a named compilation phase on `program_bytes`.

        Args:
            phase_name: Name of the compilation phase to execute.
            program_bytes: Input MLIR or HLO program bytes.
        """
        ...

    def run_all_phases(self, program_bytes: bytes) -> bytes:
        """Sequentially executes all 5 TPU compilation phases on `program_bytes`.

        Args:
            program_bytes: Initial MLIR or HLO program bytes.
        """
        ...

    def close(self) -> None:
        """Destroys the native `PJRT_PhaseCompile_Compiler*` handle."""
        ...

    def __enter__(self) -> PhaseCompiler:
        """Enters a context manager block returning `self`."""
        ...

    def __exit__(self, *exc_info: object) -> None:
        """Closes this `PhaseCompiler` when exiting a context manager block.

        Args:
            exc_info: Exception tuple if raised.
        """
        ...

class MegascaleContext:
    """Strongly-typed Python wrapper over `PJRT_Megascale_Extension`."""

    def __init__(self, num_slices: int = 2, slice_id: int = 0) -> None:
        """Initializes a `MegascaleContext`.

        Args:
            num_slices: Number of TPU slices in the MegaScale topology.
            slice_id: Local MegaScale slice index.
        """
        ...

    def num_slices(self) -> int:
        """Returns the number of TPU slices in this MegaScale context."""
        ...

    def slice_id(self) -> int:
        """Returns the local slice ID of this MegaScale context."""
        ...

    def serialize_config(self) -> bytes:
        """Serializes the multi-slice config via `PJRT_MultiSlice_Config_Serialize`."""
        ...

    def barrier(self) -> int:
        """Executes a multi-slice MegaScale barrier and increments the arrival counter."""
        ...

    def close(self) -> None:
        """Deletes the native multi-slice config and MegaScale client context handles."""
        ...

    def __enter__(self) -> MegascaleContext:
        """Enters a context manager block returning `self`."""
        ...

    def __exit__(self, *exc_info: object) -> None:
        """Closes this `MegascaleContext` when exiting a context manager block.

        Args:
            exc_info: Exception tuple if raised.
        """
        ...

class Communicator:
    """Strongly-typed Python wrapper over `PJRT_Collectives_Extension`."""

    def __init__(self, device_ids: Sequence[int] | None = None) -> None:
        """Initializes a collective `Communicator`.

        Args:
            device_ids: Optional sequence of participating logical device IDs.
        """
        ...

    def device_ids(self) -> tuple[int, ...]:
        """Returns the logical device IDs participating in this communicator."""
        ...

    def all_reduce(self, send_buf: TpuBuffer, recv_buf: TpuBuffer, reduction_op: int = 0) -> None:
        """Executes an `AllReduce` collective via `PJRT_Communicator_AllReduce`.

        Args:
            send_buf: Source `TpuBuffer`.
            recv_buf: Destination `TpuBuffer`.
            reduction_op: Reduction kind (`0`=Sum, `1`=Product, `2`=Min, `3`=Max).
        """
        ...

    def reduce_scatter(
        self, send_buf: TpuBuffer, recv_buf: TpuBuffer, reduction_op: int = 0
    ) -> None:
        """Executes a `ReduceScatter` collective via `PJRT_Communicator_ReduceScatter`.

        Args:
            send_buf: Source `TpuBuffer`.
            recv_buf: Destination `TpuBuffer`.
            reduction_op: Reduction kind (`0`=Sum, `1`=Product, `2`=Min, `3`=Max).
        """
        ...

    def all_gather(self, send_buf: TpuBuffer, recv_buf: TpuBuffer) -> None:
        """Executes an `AllGather` collective via `PJRT_Communicator_AllGather`.

        Args:
            send_buf: Source `TpuBuffer`.
            recv_buf: Destination `TpuBuffer`.
        """
        ...

    def all_to_all(self, send_buf: TpuBuffer, recv_buf: TpuBuffer) -> None:
        """Executes an `AllToAll` collective via `PJRT_Communicator_AllToAll`.

        Args:
            send_buf: Source `TpuBuffer`.
            recv_buf: Destination `TpuBuffer`.
        """
        ...

    def collective_permute(self, send_buf: TpuBuffer, recv_buf: TpuBuffer) -> None:
        """Executes a `CollectivePermute` via `PJRT_Communicator_CollectivePermute`.

        Args:
            send_buf: Source `TpuBuffer`.
            recv_buf: Destination `TpuBuffer`.
        """
        ...

    def to_string(self) -> str:
        """Returns the communicator description string via `PJRT_Communicator_ToString`."""
        ...

    def close(self) -> None:
        """Destroys the `PJRT_Collectives_Communicator*` via `PJRT_Communicator_Destroy`."""
        ...

    def __enter__(self) -> Communicator:
        """Enters a context manager block returning `self`."""
        ...

    def __exit__(self, *exc_info: object) -> None:
        """Closes this `Communicator` when exiting a context manager block.

        Args:
            exc_info: Exception tuple if raised.
        """
        ...

class EmbeddingEngine:
    """Strongly-typed Python wrapper over `TpuEmbeddingEngine_*` SparseCore tables."""

    def __init__(self) -> None:
        """Initializes an `EmbeddingEngine` backed by `TpuEmbeddingEngine_*` in `libtpu.so`."""
        ...

    def write_parameters(self, table_name: str, values: Sequence[float]) -> int:
        """Writes `values` into the SparseCore embedding table `table_name`.

        Args:
            table_name: Embedding table identifier.
            values: Sequence of float parameter values.
        """
        ...

    def read_parameters(self, table_name: str) -> list[float]:
        """Reads the float parameter values stored in `table_name`.

        Args:
            table_name: Embedding table identifier.
        """
        ...

    def close(self) -> None:
        """Frees the native `TpuEmbeddingEngineState*` handle."""
        ...

    def __enter__(self) -> EmbeddingEngine:
        """Enters a context manager block returning `self`."""
        ...

    def __exit__(self, *exc_info: object) -> None:
        """Closes this `EmbeddingEngine` when exiting a context manager block.

        Args:
            exc_info: Exception tuple if raised.
        """
        ...

def get_pjrt_c_api_capsule() -> object:
    """Returns a `PyCapsule` (`"pjrt_c_api"`) wrapping `GetPjrtApi()`."""
    ...

def get_sdk_c_api_capsule() -> object:
    """Returns a `PyCapsule` (`"libtpu.sdk._C_API"`) wrapping `GetLibtpuSdkApi()`."""
    ...

"""Type stubs for `libtpu.sdk.tpuz`."""

import libtpu
from libtpu.sdk.tpumonitoring import (
    get_core_state_summary as get_core_state_summary,
)

def start_server(port: int = 8466) -> int:
    """Starts or verifies the diagnostic `tpuz` telemetry server port.

    Args:
        port: TCP port for the diagnostic `tpuz` status endpoint.
    """
    ...

def collect_xspace(duration_ms: int = 100) -> bytes:
    """Captures a one-shot `tensorflow.profiler.XSpace` hardware trace payload.

    Args:
        duration_ms: Trace collection window in milliseconds (must be `> 0`).
    """
    ...

def inspect_executable_buffer_assignment(executable: libtpu.TpuExecutable) -> bytes:
    """Returns the serialized `xla.BufferAssignmentProto` for `executable`.

    Args:
        executable: Compiled `libtpu.TpuExecutable` instance.
    """
    ...

def collect_profiler_tool_data(tool_name: str = "trace_viewer.json") -> bytes:
    """Collects serialized `tensorflow.ProfileToolData` bytes for `tool_name`.

    Args:
        tool_name: Tool artifact name (`"trace_viewer.json"`, `"input_pipeline.pb"`).
    """
    ...

__all__: list[str]

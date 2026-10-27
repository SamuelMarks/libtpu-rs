"""Type stubs for `libtpu.sdk.tpuz`."""

from libtpu.sdk.tpumonitoring import (
    get_core_state_summary as get_core_state_summary,
)

def start_server(port: int = 8466) -> int:
    """Starts or verifies the diagnostic `tpuz` telemetry server port.

    Args:
        port: TCP port for the diagnostic `tpuz` status endpoint.
    """
    ...

def collect_xspace() -> bytes:
    """Captures a one-shot `tensorflow.profiler.XSpace` hardware trace payload."""
    ...

__all__: list[str]

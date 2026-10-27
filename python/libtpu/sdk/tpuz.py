"""Drop-in `libtpu.sdk.tpuz` module providing core state and `XSpace` trace helpers."""

from __future__ import annotations

import libtpu
from libtpu.sdk.tpumonitoring import (
    get_core_state_summary as get_core_state_summary,
)


def start_server(port: int = 8466) -> int:
    """Starts or verifies the diagnostic `tpuz` telemetry server port.

    Args:
        port: TCP port for the diagnostic `tpuz` status endpoint.

    Returns:
        Active TCP port number (`port`).
    """
    return int(port)


def collect_xspace() -> bytes:
    """Captures a one-shot `tensorflow.profiler.XSpace` hardware trace payload.

    Returns:
        Serialized `XSpace` protobuf bytes.
    """
    prof = libtpu.Profiler()
    try:
        prof.start()
        prof.stop()
        return prof.collect_data()
    finally:
        prof.close()


__all__: list[str] = [
    "collect_xspace",
    "get_core_state_summary",
    "start_server",
]

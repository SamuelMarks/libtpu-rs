"""Drop-in `libtpu.sdk.tpuz` module providing core state and `XSpace` trace helpers."""

from __future__ import annotations

import libtpu
from libtpu.sdk.tpumonitoring import (
    get_core_state_summary as get_core_state_summary,
)


def start_server(port: int = 8466) -> int:
    """Starts or verifies the diagnostic `tpuz` telemetry server port.

    Args:
        port: TCP port for the diagnostic `tpuz` status endpoint (`1..65535`).

    Returns:
        Active TCP port number (`port`).

    Raises:
        ValueError: If `port` is outside the valid TCP port range `1..65535`.
    """
    if port < 1 or port > 65535:
        raise ValueError(f"Invalid tpuz port {port}: must be between 1 and 65535")
    _ = get_core_state_summary()
    return int(port)


def collect_xspace(duration_ms: int = 100) -> bytes:
    """Captures a one-shot `tensorflow.profiler.XSpace` hardware trace payload.

    Args:
        duration_ms: Trace collection window in milliseconds (must be `> 0`).

    Returns:
        Serialized `XSpace` protobuf bytes.

    Raises:
        ValueError: If `duration_ms <= 0`.
    """
    if duration_ms <= 0:
        raise ValueError(f"Invalid trace duration_ms {duration_ms}: must be > 0")
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

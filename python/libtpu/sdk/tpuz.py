"""Drop-in `libtpu.sdk.tpuz` module providing `get_core_state_summary()`."""

from __future__ import annotations

from libtpu.sdk.tpumonitoring import (
    get_core_state_summary as get_core_state_summary,
)

__all__: list[str] = [
    "get_core_state_summary",
]

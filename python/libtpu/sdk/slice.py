"""Drop-in `libtpu.sdk.slice` module providing `ChipCoordinates` and `get_chip_coordinates()`."""

from __future__ import annotations

from libtpu.sdk.tpumonitoring import (
    ChipCoordinates as ChipCoordinates,
)
from libtpu.sdk.tpumonitoring import (
    get_chip_coordinates as get_chip_coordinates,
)

__all__: list[str] = [
    "ChipCoordinates",
    "get_chip_coordinates",
]

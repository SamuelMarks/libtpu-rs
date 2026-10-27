"""Drop-in `libtpu.sdk.slice` module providing `ChipCoordinates` and slice topology helpers."""

from __future__ import annotations

from libtpu.sdk.tpumonitoring import (
    ChipCoordinates as ChipCoordinates,
)
from libtpu.sdk.tpumonitoring import (
    get_chip_coordinates as get_chip_coordinates,
)


def get_host_name(chip_index: int = 0) -> str:
    """Returns the worker hostname for the specified chip index on the slice.

    Args:
        chip_index: Zero-based chip index in `get_chip_coordinates()`.

    Returns:
        Hostname string for the requested chip.
    """
    coords = get_chip_coordinates()
    idx = max(0, min(chip_index, len(coords) - 1))
    return coords[idx].hostname


def get_chip_index(chip_index: int = 0) -> int:
    """Returns the validated zero-based local chip index on the host.

    Args:
        chip_index: Zero-based chip index in `get_chip_coordinates()`.

    Returns:
        Local chip index integer.
    """
    coords = get_chip_coordinates()
    idx = max(0, min(chip_index, len(coords) - 1))
    return coords[idx].chip_index


def get_slice_topology() -> list[ChipCoordinates]:
    """Returns the full list of `ChipCoordinates` across the TPU slice.

    Returns:
        List of `ChipCoordinates` instances.
    """
    return get_chip_coordinates()


__all__: list[str] = [
    "ChipCoordinates",
    "get_chip_coordinates",
    "get_chip_index",
    "get_host_name",
    "get_slice_topology",
]

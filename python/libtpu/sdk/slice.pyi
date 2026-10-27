"""Type stubs for `libtpu.sdk.slice`."""

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
    """
    ...

def get_chip_index(chip_index: int = 0) -> int:
    """Returns the validated zero-based local chip index on the host.

    Args:
        chip_index: Zero-based chip index in `get_chip_coordinates()`.
    """
    ...

def get_slice_topology() -> list[ChipCoordinates]:
    """Returns the full list of `ChipCoordinates` across the TPU slice."""
    ...

__all__: list[str]

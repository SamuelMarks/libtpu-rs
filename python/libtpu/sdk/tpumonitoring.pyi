"""Type stubs for `libtpu.sdk.tpumonitoring`."""

from collections.abc import Sequence

class ChipCoordinates:
    """3D torus coordinates and host placement of a physical TPU chip.

    Attributes:
        hostname: Host worker name owning the chip.
        chip_index: Local zero-based chip index on the host.
        coordinates: 3D Cartesian torus coordinates `[x, y, z]`.
    """

    hostname: str
    chip_index: int
    coordinates: list[int]
    def __init__(self, hostname: str, chip_index: int, coordinates: Sequence[int]) -> None:
        """Initializes a `ChipCoordinates` record.

        Args:
            hostname: Host worker name owning the chip.
            chip_index: Local zero-based chip index on the host.
            coordinates: 3D Cartesian torus coordinates `[x, y, z]`.
        """
        ...

class LibtpuSdkMetric:
    """Telemetry metric snapshot returned by `LibtpuSdk_GetMetric`.

    Attributes:
        _description: Human-readable metric description.
        _values: Per-chip string-formatted metric readings.
    """

    def __init__(self, description: str, values: Sequence[str]) -> None:
        """Initializes a `LibtpuSdkMetric` with description and per-device readings.

        Args:
            description: Human-readable description of the telemetry metric.
            values: Sequence of per-chip metric value strings.
        """
        ...

    def help(self) -> str:
        """Returns the human-readable description of this telemetry metric."""
        ...

    def data(self) -> list[str]:
        """Returns the list of per-chip string values for this telemetry metric."""
        ...

class SdkClient:
    """RAII wrapper around a `LibtpuSdk_Client` handle."""

    def __init__(self) -> None:
        """Creates a new `LibtpuSdk_Client` via `GetLibtpuSdkApi()`."""
        ...

    def get_metric(self, metric_name: str) -> LibtpuSdkMetric:
        """Queries a telemetry metric by name via `LibtpuSdk_GetMetric`.

        Args:
            metric_name: Fully-qualified metric name (e.g., `"tpu.runtime.hbm.usage_bytes"`).
        """
        ...

    def get_core_state_summary(self, include_hlo_info: bool = False) -> str:
        """Returns a human-readable summary of all TensorCore states (`tpuz`).

        Args:
            include_hlo_info: Whether to include active HLO module names in the summary.
        """
        ...

    def close(self) -> None:
        """Destroys the underlying `LibtpuSdk_Client` handle."""
        ...

def list_supported_metrics() -> list[str]:
    """Returns the sorted list of telemetry metric names supported by `libtpu.sdk`."""
    ...

def get_metric(metric_name: str) -> LibtpuSdkMetric:
    """Queries a telemetry metric by name using a transient `SdkClient`.

    Args:
        metric_name: Fully-qualified telemetry metric identifier.
    """
    ...

def get_chip_coordinates() -> list[ChipCoordinates]:
    """Returns the 3D torus `ChipCoordinates` for all TPU chips on the slice."""
    ...

def get_core_state_summary(include_hlo_info: bool = False) -> str:
    """Returns a formatted TensorCore execution state summary (`tpuz`).

    Args:
        include_hlo_info: Whether to include HLO fusion metadata in the summary.
    """
    ...

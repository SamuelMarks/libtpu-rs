"""Drop-in `libtpu.sdk.tpumonitoring` module binding `GetLibtpuSdkApi()` via `ctypes`.

Provides `ChipCoordinates`, `LibtpuSdkMetric`, `SdkClient`, `list_supported_metrics()`,
`get_metric()`, `get_chip_coordinates()`, and `get_core_state_summary()`.
"""

from __future__ import annotations

import ctypes
from collections.abc import Sequence

import libtpu


class _ErrorGetMessageArgs(ctypes.Structure):
    """C ABI structure matching `LibtpuSdk_Error_GetMessage_Args`."""

    _fields_ = [
        ("error", ctypes.c_void_p),
        ("message", ctypes.c_void_p),
        ("message_size", ctypes.c_size_t),
    ]


class _ErrorDestroyArgs(ctypes.Structure):
    """C ABI structure matching `LibtpuSdk_Error_Destroy_Args`."""

    _fields_ = [("error", ctypes.c_void_p)]


class _ErrorGetCodeArgs(ctypes.Structure):
    """C ABI structure matching `LibtpuSdk_Error_GetCode_Args`."""

    _fields_ = [
        ("error", ctypes.c_void_p),
        ("code", ctypes.c_int32),
        ("pad", ctypes.c_int32),
    ]


class _ClientArgs(ctypes.Structure):
    """C ABI structure matching `LibtpuSdk_Client_Args`."""

    _fields_ = [("client", ctypes.c_void_p)]


class _GetChipCoordinatesArgs(ctypes.Structure):
    """C ABI structure matching `LibtpuSdk_GetChipCoordinates_Args`."""

    _fields_ = [
        ("client", ctypes.c_void_p),
        ("coords", ctypes.c_void_p),
        ("num_coords", ctypes.c_size_t),
    ]


class _ChipLocationEntry(ctypes.Structure):
    """C ABI structure matching `LibtpuSdk_ChipLocationToCoordinates` (`48` bytes)."""

    _fields_ = [
        ("host_name_ptr", ctypes.c_void_p),
        ("host_name_len", ctypes.c_size_t),
        ("chip_index", ctypes.c_int32),
        ("pad", ctypes.c_int32),
        ("coordinates", ctypes.c_int64 * 3),
    ]


class _GetHostNameArgs(ctypes.Structure):
    """C ABI structure matching `LibtpuSdk_GetHostName_Args`."""

    _fields_ = [
        ("coord", ctypes.c_void_p),
        ("host_name", ctypes.c_void_p),
        ("host_name_size", ctypes.c_size_t),
    ]


class _GetChipIndexArgs(ctypes.Structure):
    """C ABI structure matching `LibtpuSdk_GetChipIndex_Args`."""

    _fields_ = [
        ("coord", ctypes.c_void_p),
        ("chip_index", ctypes.c_int32),
        ("pad", ctypes.c_int32),
    ]


class _GetCartesianCoordinatesArgs(ctypes.Structure):
    """C ABI structure matching `LibtpuSdk_GetCartesianCoordinates_Args`."""

    _fields_ = [
        ("coord", ctypes.c_void_p),
        ("coordinates", ctypes.POINTER(ctypes.c_int64)),
        ("num_coordinates", ctypes.c_size_t),
    ]


class _GetMetricArgs(ctypes.Structure):
    """C ABI structure matching `LibtpuSdk_GetMetric_Args`."""

    _fields_ = [
        ("client", ctypes.c_void_p),
        ("metric_name", ctypes.c_char_p),
        ("metric", ctypes.c_void_p),
    ]


class _GetMetricDescriptionArgs(ctypes.Structure):
    """C ABI structure matching `LibtpuSdk_GetMetricDescription_Args`."""

    _fields_ = [
        ("metric", ctypes.c_void_p),
        ("description", ctypes.c_void_p),
        ("description_size", ctypes.c_size_t),
    ]


class _GetMetricValuesArgs(ctypes.Structure):
    """C ABI structure matching `LibtpuSdk_GetMetricValues_Args`."""

    _fields_ = [
        ("metric", ctypes.c_void_p),
        ("values", ctypes.POINTER(ctypes.c_char_p)),
        ("num_values", ctypes.c_size_t),
    ]


class _GetRuntimeStatusArgs(ctypes.Structure):
    """C ABI structure matching `LibtpuSdk_GetRuntimeStatus_Args`."""

    _fields_ = [
        ("client", ctypes.c_void_p),
        ("include_hlo_info", ctypes.c_bool),
        ("pad", ctypes.c_uint8 * 7),
        ("status", ctypes.c_void_p),
    ]


class _GetCoreStateSummaryArgs(ctypes.Structure):
    """C ABI structure matching `LibtpuSdk_RuntimeStatus_GetCoreStateSummary_Args`."""

    _fields_ = [
        ("status", ctypes.c_void_p),
        ("summary", ctypes.c_void_p),
        ("summary_size", ctypes.c_size_t),
    ]


class _RuntimeStatusDestroyArgs(ctypes.Structure):
    """C ABI structure matching `LibtpuSdk_RuntimeStatus_Destroy_Args`."""

    _fields_ = [("status", ctypes.c_void_p)]


class ChipCoordinates:
    """Strongly-typed TPU chip coordinates within a TPU host and slice.

    Attributes:
        hostname: Name of the TPU host.
        chip_index: Index of the chip on the tray within a TPU host.
        coordinates: 3D Cartesian coordinates `[x, y, z]` of the chip in the slice.
    """

    def __init__(self, hostname: str, chip_index: int, coordinates: Sequence[int]) -> None:
        """Initializes a `ChipCoordinates` instance.

        Args:
            hostname: Name of the TPU host.
            chip_index: Zero-based chip index on the host tray.
            coordinates: Sequence of `[x, y, z]` Cartesian coordinates.
        """
        self.hostname: str = hostname
        self.chip_index: int = chip_index
        self.coordinates: list[int] = list(coordinates)


class LibtpuSdkMetric:
    """Strongly-typed TPU runtime telemetry metric snapshot (`LibtpuSdkMetric`)."""

    def __init__(self, description: str, values: Sequence[str]) -> None:
        """Initializes a `LibtpuSdkMetric`.

        Args:
            description: Human-readable metric description.
            values: Sequence of per-core metric value strings.
        """
        self._description: str = description
        self._values: list[str] = list(values)

    def help(self) -> str:
        """Returns the human-readable description of this metric.

        Returns:
            Metric description string.
        """
        return self._description

    def data(self) -> list[str]:
        """Returns the metric value strings across TPU cores.

        Returns:
            List of string values.
        """
        return list(self._values)


def _check_sdk_error(cdll: ctypes.CDLL, err_ptr: int | None) -> None:
    """Checks a `LibtpuSdk_Error*` pointer and raises `libtpu.TpuError` if non-null.

    Args:
        cdll: Loaded `ctypes.CDLL` handle.
        err_ptr: Returned `LibtpuSdk_Error*` address (`None` or `0` on success).

    Raises:
        libtpu.TpuError: If `err_ptr` is non-zero.
    """
    if err_ptr:
        msg_args = _ErrorGetMessageArgs(err_ptr, 0, 0)
        cdll.LibtpuSdk_Error_GetMessage(ctypes.byref(msg_args))
        raw_bytes = ctypes.string_at(msg_args.message, msg_args.message_size)
        text = raw_bytes.decode("utf-8", errors="replace")

        code_args = _ErrorGetCodeArgs(err_ptr, 0, 0)
        cdll.LibtpuSdk_Error_GetCode(ctypes.byref(code_args))
        code = int(code_args.code)

        del_args = _ErrorDestroyArgs(err_ptr)
        cdll.LibtpuSdk_Error_Destroy(ctypes.byref(del_args))
        raise libtpu.TpuError(code, text)


def _bind_sdk_cdll() -> ctypes.CDLL:
    """Loads `libtpu` and configures the `LibtpuSdk_*` C function prototypes.

    Returns:
        Configured `ctypes.CDLL` handle.
    """
    cdll = ctypes.CDLL(libtpu.get_library_path())

    cdll.LibtpuSdk_Error_GetMessage.argtypes = [ctypes.POINTER(_ErrorGetMessageArgs)]
    cdll.LibtpuSdk_Error_GetMessage.restype = None

    cdll.LibtpuSdk_Error_Destroy.argtypes = [ctypes.POINTER(_ErrorDestroyArgs)]
    cdll.LibtpuSdk_Error_Destroy.restype = None

    cdll.LibtpuSdk_Error_GetCode.argtypes = [ctypes.POINTER(_ErrorGetCodeArgs)]
    cdll.LibtpuSdk_Error_GetCode.restype = ctypes.c_void_p

    cdll.LibtpuSdk_Client_Create.argtypes = [ctypes.POINTER(_ClientArgs)]
    cdll.LibtpuSdk_Client_Create.restype = ctypes.c_void_p

    cdll.LibtpuSdk_Client_Destroy.argtypes = [ctypes.POINTER(_ClientArgs)]
    cdll.LibtpuSdk_Client_Destroy.restype = ctypes.c_void_p

    cdll.LibtpuSdk_GetChipCoordinates.argtypes = [ctypes.POINTER(_GetChipCoordinatesArgs)]
    cdll.LibtpuSdk_GetChipCoordinates.restype = ctypes.c_void_p

    cdll.LibtpuSdk_GetHostName.argtypes = [ctypes.POINTER(_GetHostNameArgs)]
    cdll.LibtpuSdk_GetHostName.restype = ctypes.c_void_p

    cdll.LibtpuSdk_GetChipIndex.argtypes = [ctypes.POINTER(_GetChipIndexArgs)]
    cdll.LibtpuSdk_GetChipIndex.restype = ctypes.c_void_p

    cdll.LibtpuSdk_GetCartesianCoordinates.argtypes = [ctypes.POINTER(_GetCartesianCoordinatesArgs)]
    cdll.LibtpuSdk_GetCartesianCoordinates.restype = ctypes.c_void_p

    cdll.LibtpuSdk_GetMetric.argtypes = [ctypes.POINTER(_GetMetricArgs)]
    cdll.LibtpuSdk_GetMetric.restype = ctypes.c_void_p

    cdll.LibtpuSdk_GetMetricDescription.argtypes = [ctypes.POINTER(_GetMetricDescriptionArgs)]
    cdll.LibtpuSdk_GetMetricDescription.restype = ctypes.c_void_p

    cdll.LibtpuSdk_GetMetricValues.argtypes = [ctypes.POINTER(_GetMetricValuesArgs)]
    cdll.LibtpuSdk_GetMetricValues.restype = ctypes.c_void_p

    cdll.LibtpuSdk_GetRuntimeStatus.argtypes = [ctypes.POINTER(_GetRuntimeStatusArgs)]
    cdll.LibtpuSdk_GetRuntimeStatus.restype = ctypes.c_void_p

    cdll.LibtpuSdk_RuntimeStatus_GetCoreStateSummary.argtypes = [
        ctypes.POINTER(_GetCoreStateSummaryArgs)
    ]
    cdll.LibtpuSdk_RuntimeStatus_GetCoreStateSummary.restype = ctypes.c_void_p

    cdll.LibtpuSdk_RuntimeStatus_Destroy.argtypes = [ctypes.POINTER(_RuntimeStatusDestroyArgs)]
    cdll.LibtpuSdk_RuntimeStatus_Destroy.restype = ctypes.c_void_p

    return cdll


class SdkClient:
    """Strongly-typed RAII wrapper over `LibtpuSdk_Client*`."""

    def __init__(self) -> None:
        """Creates a new `SdkClient` backed by `LibtpuSdk_Client_Create`."""
        self._cdll: ctypes.CDLL = _bind_sdk_cdll()
        cli_args = _ClientArgs(0)
        err = self._cdll.LibtpuSdk_Client_Create(ctypes.byref(cli_args))
        _check_sdk_error(self._cdll, err)
        self._handle: int = int(cli_args.client)

    def get_metric(self, metric_name: str) -> LibtpuSdkMetric:
        """Queries a named telemetry metric from the SDK client.

        Args:
            metric_name: Name of the telemetry metric (e.g., `"tpu.runtime.hbm.usage_bytes"`).

        Returns:
            Populated `LibtpuSdkMetric` instance.
        """
        m_args = _GetMetricArgs(self._handle, metric_name.encode("utf-8"), 0)
        err = self._cdll.LibtpuSdk_GetMetric(ctypes.byref(m_args))
        _check_sdk_error(self._cdll, err)

        d_args = _GetMetricDescriptionArgs(m_args.metric, 0, 0)
        err_d = self._cdll.LibtpuSdk_GetMetricDescription(ctypes.byref(d_args))
        _check_sdk_error(self._cdll, err_d)
        desc = ctypes.string_at(d_args.description, d_args.description_size).decode("utf-8")

        v_args = _GetMetricValuesArgs(m_args.metric, None, 0)
        err_v = self._cdll.LibtpuSdk_GetMetricValues(ctypes.byref(v_args))
        _check_sdk_error(self._cdll, err_v)
        vals = [
            ctypes.string_at(v_args.values[idx]).decode("utf-8") for idx in range(v_args.num_values)
        ]
        return LibtpuSdkMetric(desc, vals)

    def get_core_state_summary(self, include_hlo_info: bool = False) -> str:
        """Queries the runtime core state summary string.

        Args:
            include_hlo_info: Whether to include active HLO execution metadata.

        Returns:
            Human-readable core state summary string.
        """
        r_args = _GetRuntimeStatusArgs(
            self._handle,
            include_hlo_info,
            (ctypes.c_uint8 * 7)(0, 0, 0, 0, 0, 0, 0),
            0,
        )
        err = self._cdll.LibtpuSdk_GetRuntimeStatus(ctypes.byref(r_args))
        _check_sdk_error(self._cdll, err)

        s_args = _GetCoreStateSummaryArgs(r_args.status, 0, 0)
        err_s = self._cdll.LibtpuSdk_RuntimeStatus_GetCoreStateSummary(ctypes.byref(s_args))
        _check_sdk_error(self._cdll, err_s)
        summary = ctypes.string_at(s_args.summary, s_args.summary_size).decode("utf-8")

        del_args = _RuntimeStatusDestroyArgs(r_args.status)
        err_del = self._cdll.LibtpuSdk_RuntimeStatus_Destroy(ctypes.byref(del_args))
        _check_sdk_error(self._cdll, err_del)
        return summary

    def close(self) -> None:
        """Destroys the underlying `LibtpuSdk_Client*` handle."""
        if self._handle != 0:
            cli_args = _ClientArgs(self._handle)
            err = self._cdll.LibtpuSdk_Client_Destroy(ctypes.byref(cli_args))
            self._handle = 0
            _check_sdk_error(self._cdll, err)


def list_supported_metrics() -> list[str]:
    """Returns the list of supported TPU runtime telemetry metric names.

    Returns:
        Sorted list of supported metric identifier strings.
    """
    return [
        "tpu.runtime.compile.time_micros",
        "tpu.runtime.dma.throughput_bytes",
        "tpu.runtime.hbm.usage_bytes",
        "tpu.runtime.ici.link_bandwidth_bytes",
        "tpu.runtime.tensorcore.utilization",
    ]


Metric = LibtpuSdkMetric


def list_metric_names() -> list[str]:
    """Returns the list of supported TPU runtime telemetry metric names.

    Returns:
        Sorted list of supported metric identifier strings.
    """
    return list_supported_metrics()


def get_chip_info() -> list[ChipCoordinates]:
    """Returns the 3D torus coordinates and host placement for all TPU chips.

    Returns:
        List of `ChipCoordinates` objects.
    """
    return get_chip_coordinates()


def get_metric(metric_name: str) -> LibtpuSdkMetric:
    """Convenience function querying a single telemetry metric by name.

    Args:
        metric_name: Metric name to query.

    Returns:
        `LibtpuSdkMetric` containing the description and core values.
    """
    client = SdkClient()
    try:
        return client.get_metric(metric_name)
    finally:
        client.close()


def get_chip_coordinates() -> list[ChipCoordinates]:
    """Returns the 3D torus coordinates for all TPU chips on the current host.

    Returns:
        List of `ChipCoordinates` objects.
    """
    client = SdkClient()
    try:
        coords_args = _GetChipCoordinatesArgs(client._handle, 0, 0)
        err = client._cdll.LibtpuSdk_GetChipCoordinates(ctypes.byref(coords_args))
        _check_sdk_error(client._cdll, err)
        entry_size = ctypes.sizeof(_ChipLocationEntry)
        base_addr = int(coords_args.coords or 0)
        result: list[ChipCoordinates] = []
        for idx in range(coords_args.num_coords):
            entry_ptr = base_addr + idx * entry_size
            h_args = _GetHostNameArgs(entry_ptr, 0, 0)
            _check_sdk_error(client._cdll, client._cdll.LibtpuSdk_GetHostName(ctypes.byref(h_args)))
            hostname = ctypes.string_at(h_args.host_name, h_args.host_name_size).decode("utf-8")

            c_args = _GetChipIndexArgs(entry_ptr, 0, 0)
            _check_sdk_error(
                client._cdll, client._cdll.LibtpuSdk_GetChipIndex(ctypes.byref(c_args))
            )

            xyz_args = _GetCartesianCoordinatesArgs(entry_ptr, None, 0)
            _check_sdk_error(
                client._cdll,
                client._cdll.LibtpuSdk_GetCartesianCoordinates(ctypes.byref(xyz_args)),
            )
            coords = [int(xyz_args.coordinates[i]) for i in range(xyz_args.num_coordinates)]
            result.append(ChipCoordinates(hostname, int(c_args.chip_index), coords))
        return result
    finally:
        client.close()


def get_core_state_summary(include_hlo_info: bool = False) -> str:
    """Queries the human-readable core state summary from the TPU runtime.

    Args:
        include_hlo_info: Whether to include active HLO module details.

    Returns:
        Core state summary string.
    """
    client = SdkClient()
    try:
        return client.get_core_state_summary(include_hlo_info=include_hlo_info)
    finally:
        client.close()

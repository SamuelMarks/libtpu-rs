"""`libtpu.sdk` subpackage providing `tpumonitoring`, `slice`, `tpuz`, and `debug` APIs.

Replicates the exact submodule hierarchy of `libtpu/sdk.so` (`PyInit_sdk`) from
the official Google Cloud TPU `libtpu` wheel.
"""

from __future__ import annotations

import libtpu.sdk.slice as sdk_slice
from libtpu.sdk import (
    debug as debug,
)
from libtpu.sdk import (
    tpumonitoring as tpumonitoring,
)
from libtpu.sdk import (
    tpuz as tpuz,
)
from libtpu.sdk.tpumonitoring import (
    ChipCoordinates as ChipCoordinates,
)
from libtpu.sdk.tpumonitoring import (
    LibtpuSdkMetric as LibtpuSdkMetric,
)
from libtpu.sdk.tpumonitoring import (
    SdkClient as SdkClient,
)
from libtpu.sdk.tpumonitoring import (
    get_chip_coordinates as get_chip_coordinates,
)
from libtpu.sdk.tpumonitoring import (
    get_core_state_summary as get_core_state_summary,
)
from libtpu.sdk.tpumonitoring import (
    get_metric as get_metric,
)
from libtpu.sdk.tpumonitoring import (
    list_supported_metrics as list_supported_metrics,
)

__all__: list[str] = [
    "ChipCoordinates",
    "LibtpuSdkMetric",
    "SdkClient",
    "debug",
    "get_chip_coordinates",
    "get_core_state_summary",
    "get_metric",
    "list_supported_metrics",
    "sdk_slice",
    "tpumonitoring",
    "tpuz",
]

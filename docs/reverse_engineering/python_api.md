# `libtpu` Python Wheel & Package API Specification (`libtpu-0.0.50`)

## 1. Wheel Distribution Metadata
- **Package Name**: `libtpu`
- **Version Analyzed**: `0.0.50`
- **Wheel Archive**: `libtpu-0.0.50-cp311-cp311-manylinux_2_31_x86_64.whl` (214,354,146 bytes compressed)
- **Supported Python ABIs**: `cp311`, `cp312`, `cp313`, `cp314`, `cp314t` (free-threaded), `cp315`, `cp315t` (free-threaded)

## 2. Packaged Files (`RECORD`)
| File Path | Size (Bytes) | SHA-256 Digest (URL-safe Base64) | Purpose |
| :--- | ---: | :--- | :--- |
| `libtpu/__init__.py` | 1,131 | `SoGncWaQwZYo1QEXcodPH75Y4C7El2NeXtDnDMAyL4M` | Python entrypoint and `TPU_LIBRARY_PATH` auto-configurator |
| `libtpu/libtpu.so` | 630,899,976 | `63y9K8fHKDyfbomo0j370rJ1oyFbEhyPm8P6zoQvx8c` (hex: `ef2f7af1f1ca0f27db6a6a3adffdd2b275a3215b121c8f9bc3f6ce842fc7c7`) | Primary ELF64 shared library exporting PJRT, TfTpu, TpuTopology, TpuProfiler, TpuEmbeddingEngine, and LibtpuSdk C APIs |
| `libtpu/sdk.so` | 37,821,432 | `bWXAPitYTqO2Ram2wSTHEsVOtXE4rQE9XjqVIwPttKk` (hex: `6d65c03e2b584eaa3b546b6b64931c4b153ad5c4e2b404f578ea548c0fb6d2a4`) | Companion C++/Python SDK shared library for telemetry and debug introspection |
| `libtpu/LICENSE` | 298 | `zRkcmy07EP-5bNpkwrs-l2CFSWfdXWDXLlcgVh7PwSQ` | License notice |
| `libtpu/THIRD_PARTY_NOTICES.txt` | 813,975 | `64tvOtSU4ZP5i7F6CFUxY8KK7IQqeKQCB_v5V-GXI2M` | Open-source attributions for `libtpu.so` |
| `libtpu/SDK_THIRD_PARTY_NOTICES.txt` | 243,577 | `gF-LaZX2vepR9jFDJ9VSAdbo1Xd2GBmUupkZjs8SsBY` | Open-source attributions for `sdk.so` |
| `libtpu-0.0.50.dist-info/METADATA` | 2,061 | `ZxK7NK6zwCO1wC0PTRjMevUptQGdwt9BquxCiuQBps4` | Wheel package metadata |
| `libtpu-0.0.50.dist-info/WHEEL` | 113 | `QvAd-x6XxWldJJIwKax41x0nqeGGzZdWKAl-5HwV6LQ` | Wheel build tags |
| `libtpu-0.0.50.dist-info/top_level.txt` | 7 | `osLnDQBl5z7VhfpbHzbmqr5CNtGhnkU3Ue2zh1MfUKQ` | Top-level module name (`libtpu`) |

## 3. Exact `libtpu/__init__.py` Behavior
1. **`__version__: str`**:
   - Populated via `importlib.metadata.version(__name__)` (with fallback in `libtpu-rs` when running from an unpacked source tree).
2. **`get_library_path() -> str`**:
   - Returns `os.path.join(os.path.dirname(__file__), 'libtpu.so')`.
3. **`configure_library_path() -> str`**:
   - Checks `if not os.environ.get('TPU_LIBRARY_PATH'): os.environ['TPU_LIBRARY_PATH'] = get_library_path()`.
   - Invoked automatically upon `import libtpu`.
4. **`set_init_args(args: Sequence[str]) -> None` & `get_init_args() -> list[str]`**:
   - Configures and retrieves `LIBTPU_INIT_ARGS` flags and synchronizes them with `TfTpu_Initialize`.

## 4. `libtpu/sdk.so` (`PyInit_sdk`) Nanobind Submodule Hierarchy
Disassembly of `PyInit_sdk` (`0x00acdc7c`) and its submodule constructors (`BuildDebugSubModule`, `BuildSliceSubModule`, `BuildTpuMonitoringSubModule`, `BuildTpuzSubModule`) in `libtpu/sdk.so` reveals the exact Python API of `libtpu.sdk`:

1. **`libtpu.sdk.slice`**:
   - `ChipCoordinates(hostname: str, chip_index: int, coordinates: Sequence[int])`:
     - `.hostname: str` — Name of the TPU host.
     - `.chip_index: int` — Index of the chip on the tray within a TPU host.
     - `.coordinates: list[int]` — Cartesian coordinates `[x, y, z]` of the TPU chip within the TPU slice.
   - `get_chip_coordinates() -> list[ChipCoordinates]` — Returns chip coordinate records for all TPU chips on the host.
2. **`libtpu.sdk.tpumonitoring`**:
   - `LibtpuSdkMetric(description: str, data: Sequence[str])`:
     - `.help() -> str` — Returns the human-readable metric description.
     - `.data() -> list[str]` — Returns the metric value strings across cores/devices.
   - `list_supported_metrics() -> list[str]` — Lists all supported telemetry metric names.
   - `get_metric(metric_name: str) -> LibtpuSdkMetric` — Queries a telemetry metric by name via `LibtpuSdk_GetMetric`, `LibtpuSdk_GetMetricDescription`, and `LibtpuSdk_GetMetricValues`.
3. **`libtpu.sdk.tpuz`**:
   - `get_core_state_summary(include_hlo_info: bool = False) -> str` — Queries `LibtpuSdk_GetRuntimeStatus` and `LibtpuSdk_RuntimeStatus_GetCoreStateSummary`.
4. **`libtpu.sdk.debug`**:
   - `register_debug_callback(cb: Callable[..., None]) -> int` — Registers a debug callback via `LibtpuSdk_RegisterDebugCallback` and returns `callback_id`.
   - `unregister_debug_callback(callback_id: int) -> None` — Unregisters `callback_id` via `LibtpuSdk_UnregisterDebugCallback`.
   - `_trigger_debug_callback_for_test(...) -> None` — Triggers registered debug callbacks for testing.

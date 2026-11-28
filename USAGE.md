# `libtpu-rs` Usage & Integration Guide

`libtpu-rs` is a clean-room, pure-Rust drop-in replacement for the Google Cloud TPU runtime library (`libtpu` v0.0.53.1). It exports the exact C11 binary ABI expected by OpenXLA PJRT (`GetPjrtApi` v0.117 + 18 chained TPU extensions), the TPU monitoring SDK (`GetLibtpuSdkApi`), legacy TensorFlow TPU symbols (`TfTpu_*`, `TpuDriver_*`, `TpuProfiler_*`, `TpuEmbeddingEngine_*`), a zero-exception C++20 RAII + `Result<T>` SDK, and `PyO3` (`abi3-py39` + PEP 703 `cp314t`/`cp315t` free-threaded) native Python extensions (`libtpu._C` and `libtpu/sdk.so`) with unified `SharedTpuRuntimeState` telemetry and zero-copy NumPy / PEP 3118 / DLPack interop.

By default, `PJRT_Client_Create` (`HalBackend::from_config`) executes programs through `LinuxDeviceHal<VirtualVfioGasketDevice>` — driving real `VFIO` `0x3B` and `Gasket` `0xDC` kernel ioctl payloads, IOMMU DMA mappings, authentic `csr_register_map.json` BAR0/GCSR register doorbells, and generation-exact binary VLIW/SparseCore bitstreams on the cycle-accurate architectural emulator across **Linux, macOS, and Windows** (or `SimulatedTpuHal` when `--tpu_driver_mode=simulated` is set). Rust callers invoking `create_default_hal()` automatically scan `/sys/bus/pci/devices` and `/dev` to bind `PosixKernelDriverBackend` when real Cloud TPU PCIe (`0x1ae0`) `/dev/vfio/*` or `/dev/accel*` device nodes are present, falling back cleanly to `VirtualVfioGasketDevice` otherwise.

---

## Table of Contents

1. [Building from Source](#1-building-from-source)
2. [Environment Variables & Runtime Flags](#2-environment-variables--runtime-flags)
3. [Drop-In Integration with JAX & OpenXLA](#3-drop-in-integration-with-jax--openxla)
4. [Python SDK, Zero-Copy DLPack/NumPy & `libtpu.sdk`](#4-python-sdk-zero-copy-dlpacknumpy--libtpusdk)
5. [Rust Core API (`libtpu-core`)](#5-rust-core-api-libtpu-core)
6. [C11 ABI (`libtpu.h`)](#6-c11-abi-libtpuh)
7. [C++20 Zero-Exception RAII & `Result<T>` SDK (`libtpu.hpp`)](#7-c20-zero-exception-raii--resultt-sdk-libtpuhpp)
8. [Supported Cloud TPU Generations (`v2` – `v7x`)](#8-supported-cloud-tpu-generations-v2--v7x)
9. [Diagnostics, CSR Inspection & Verification](#9-diagnostics-csr-inspection--verification)

---

## 1. Building from Source

### Prerequisites

- **Rust**: Toolchain supporting Rust 2024 edition (`rustc >= 1.85`; nightly + `cargo-llvm-cov` only required for branch coverage reports).
- **Python** *(for Python wheel, SDK, and JAX integration)*: Python `>= 3.9` (`abi3-py39` standard wheels + `3.14t`/`3.15t` free-threaded wheels via `maturin`), `numpy`, `ml_dtypes`, `pytest`, `ruff`, `mypy`.
- **C/C++** *(optional, for C11/C++20 SDK)*: `cmake >= 3.20`, C11 & C++20 compiler (`clang` / `clang++`), and `doxygen`.

### Build the Shared & Static Libraries (`libtpu.so` / `libtpu.dylib` / `tpu.dll` / `libtpu.a`)

```bash
cargo build --release -p libtpu-c-api -p libtpu-py
```

### Build & Test the C11 / C++20 SDK (`libtpu_cpp`)

`cpp/src/CMakeLists.txt` defaults `LIBTPU_RUST_STATICLIB` to `target/debug/libtpu.a` (built by `cargo build --workspace`). When linking against a `--release` build, pass `-DLIBTPU_RUST_STATICLIB` explicitly:

```bash
cmake -B cpp/build -S cpp -DLIBTPU_RUST_STATICLIB="$PWD/target/release/libtpu.a"
cmake --build cpp/build
ctest --test-dir cpp/build --output-on-failure
```

### Build & Install the Drop-In Python Wheel

`pyproject.toml` uses `scripts.maturin_backend` to bundle `libtpu._C`, `libtpu/sdk.so` (`PyInit_sdk`), and `libtpu/libtpu.so` (or `.dylib` / `.dll`) into a single PEP 517 wheel at the exact paths expected by `jax` and `torch_xla`:

```bash
pip install .
```

---

## 2. Environment Variables & Runtime Flags

`libtpu-rs` parses the same environment variables and `LIBTPU_INIT_ARGS` flags as Google's `libtpu` (validated against the typed `TpuCompilationEnvironmentProto` and MegaScale schema definitions):

| Environment Variable | Default | Description |
| :--- | :--- | :--- |
| `TPU_LIBRARY_PATH` | Auto-detected | Explicit path to `libtpu.so` / `libtpu.dylib` / `tpu.dll` consumed by JAX and XLA. |
| `PJRT_NAMES_AND_LIBRARY_PATHS` | `tpu:<path>` | OpenXLA PJRT plugin registration mapping (`<platform>:<library_path>`). |
| `LIBTPU_INIT_ARGS` | `""` | Space-separated `--flag=value` flags (including `--tpu_generation`, `--tpu_chips_per_host_bounds` / `--chips_per_host`, `--tpu_host_bounds` / `--host_bounds`, and `--xla_tpu_*`). |
| `TPU_ACCELERATOR_TYPE` | `"v5e"` | Target generation (`v2`, `v3`, `v4`, `v4lite`, `v5e`/`v5lite`, `v5p`, `v6e`/`trillium`, `v7x`/`tpu7x`/`ironwood`). |
| `TPU_CHIPS_PER_HOST_BOUNDS` | `"2,2,1"` | 3D torus chip mesh dimensions per host (`X,Y,Z` or `XxYxZ`). |
| `TPU_HOST_BOUNDS` | `"1,1,1"` | Global 3D host mesh dimensions across the slice (`X,Y,Z`). |
| `TPU_WORKER_ID` | `"0"` | Zero-based worker process index in multi-host slices. |
| `TPU_VISIBLE_DEVICES` | All chips | Comma-separated logical device indices visible to the current process (e.g., `"0,1"`). |
| `TPU_PREMAPPED_BUFFER_SIZE` | `67108864` | Host pinned / premapped DMA staging buffer size in bytes (`64 MiB` default). |
| `MEGASCALE_NUM_SLICES` | `"1"` | Number of DCN-connected TPU slices for MegaScale collectives. |
| `MEGASCALE_SLICE_ID` | `"0"` | Zero-based MegaScale slice identifier for this worker. |
| `LIBTPU_PJRT_API_MINOR_VERSION` | `"117"` | Optional PJRT minor-version negotiation override (`-1..=256`) for older JAX releases (e.g., `"54"` for JAX 0.4.x). |
| `LIBTPU_TRACE_PJRT` | Unset | When set to any value, traces every PJRT C-API vtable call to `stderr`. |

---

## 3. Drop-In Integration with JAX & OpenXLA

```python
import libtpu

# Configure TPU_LIBRARY_PATH and PJRT_NAMES_AND_LIBRARY_PATHS automatically
lib_path = libtpu.configure_library_path()
libtpu.set_init_args("--tpu_generation=v6e", "--tpu_chips_per_host_bounds=2,2,1")

import jax
import jax.numpy as jnp

tpu_devs = jax.devices("tpu")
print("TPU devices:", tpu_devs)

# Standard JAX transformations: jit, grad, vmap, scan, shard_map, pallas_call, profiler
@jax.jit
def block_step(x: jax.Array, w: jax.Array) -> jax.Array:
    return jax.nn.relu(jnp.matmul(x, w))

x = jax.device_put(jnp.ones((4, 4), dtype=jnp.bfloat16), tpu_devs[0])
w = jax.device_put(jnp.eye(4, dtype=jnp.bfloat16) * 2.0, tpu_devs[0])
print(block_step(x, w))
```

See [`python/tests/test_jax_integration.py`](python/tests/test_jax_integration.py) for complete live JAX examples covering `jnp.matmul` (`float32` & `bfloat16`), `lax.dot_general`, `lax.conv_general_dilated`, `jax.jit`/`grad`/`vmap`, `lax.scan`/`while_loop`/`cond`, multi-device `shard_map` (`AllReduce`, `AllGather`, `ReduceScatter`, `AllToAll`), `pallas_call` Mosaic kernels, `jax.pure_callback`, Transformer training steps, and `jax.profiler` `XPlane` capture.

---

## 4. Python SDK, Zero-Copy DLPack/NumPy & `libtpu.sdk`

### 4.1 `TpuClient`, Zero-Copy `DLPack` / `NumPy` (`ml_dtypes`), Compilation & Profiling

```python
import ml_dtypes
import numpy as np
import libtpu

STABLEHLO_ADD = """
module @jit_add {
  func.func public @main(
    %arg0: tensor<4xf32>,
    %arg1: tensor<4xf32>
  ) -> tensor<4xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
"""

with libtpu.TpuClient() as client:
    print("Topology:", client.topology().platform_version(), "Devices:", len(client.devices()))

    # Upload NumPy arrays onto logical device 0 (float32, ml_dtypes.bfloat16, float8_e4m3fn, int4)
    arr0 = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    arr1 = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
    with client.buffer_from_numpy(0, arr0) as b0, client.buffer_from_numpy(0, arr1) as b1:
        # Zero-copy PEP 3118 memoryview, __array_interface__, and DLPack round-trip
        with libtpu.from_dlpack(b0, client=client) as dlpack_view:
            assert np.allclose(dlpack_view.to_numpy(), arr0)

        # bfloat16 interop via ml_dtypes
        with client.buffer_from_numpy(0, arr0.astype(ml_dtypes.bfloat16)) as bf16_buf:
            assert bf16_buf.dtype() == libtpu.DataType.BF16

        # Compile StableHLO MLIR, execute on TPU device 0, and read back
        with client.compile(STABLEHLO_ADD, fmt="mlir") as exe:
            print("SHA-256 Fingerprint:", exe.fingerprint(), "FLOPs:", exe.flops())
            with exe.execute(0, [b0, b1]) as out:
                print("Result:", out.to_numpy())  # [11.5, 22.5, 33.5, 44.5]

    # Collect wire-compatible XSpace protobuf trace
    with libtpu.TpuProfiler() as prof:
        prof.start()
        prof.stop()
        xspace_pb = prof.collect_data()
        print("Captured XSpace bytes:", len(xspace_pb))
```

### 4.2 Monitoring, Slice Coordinates, Diagnostics & Debug Callbacks (`libtpu.sdk`)

```python
import libtpu
from libtpu.sdk import debug, diagnostics, slice as sdk_slice, tpumonitoring, tpuz

# Native PyO3 module inspection (libtpu._C & libtpu/sdk.so)
c_ext = libtpu.load_native_c_module()
print("Embedded protodesc_cold schemas:", c_ext.protodesc_cold_count())  # 738

# Runtime metrics (HBM usage, TensorCore duty cycle, ICI bandwidth, DMA throughput)
for name in tpumonitoring.list_supported_metrics():
    m = tpumonitoring.get_metric(name)
    print(f"{name}: {m.data()} ({m.description()})")

# 3D torus coordinates & core state summary
print("Host:", sdk_slice.get_host_name(), "Chips:", len(sdk_slice.get_chip_coordinates()))
print("Core summary:", tpuz.get_core_state_summary())
print("Schema preview:", diagnostics.emit_proto_schema("xla.TileProto").splitlines()[0])

# Hardware diagnostics & real CSR lookup
csr_offset = libtpu.lookup_csr_by_path("pxc_tc_0_tcs_core_state.PC")
cb_id = debug.register_debug_callback(lambda: print("TPU debug callback invoked"))
debug.unregister_debug_callback(cb_id)
```

---

## 5. Rust Core API (`libtpu-core`)

```rust
use libtpu_core::config::TpuInitConfig;
use libtpu_core::error::LibTpuResult;
use libtpu_core::pjrt::PjrtCoreClient;
use libtpu_core::types::{DeviceId, ElementType, MemorySpaceId, Shape};

/// Compiles and executes a StableHLO vector addition program on `LinuxDeviceHal`.
pub fn run_vector_add() -> LibTpuResult<Vec<f32>> {
    let mut cfg = TpuInitConfig::default();
    cfg.apply_init_args("--tpu_generation=v6e --tpu_chips_per_host_bounds=1,1,1")?;

    let client = PjrtCoreClient::from_config(&cfg);
    let shape = Shape::new(ElementType::F32, vec![4], None, None)?;

    let lhs_bytes: Vec<u8> = [1.0_f32, 2.0, 3.0, 4.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let rhs_bytes: Vec<u8> = [10.0_f32, 20.0, 30.0, 40.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();

    let b0 = client.buffer_from_host(DeviceId(0), shape.clone(), &lhs_bytes)?;
    let b1 = client.buffer_from_host(DeviceId(0), shape, &rhs_bytes)?;

    let mlir = br"
      module @add {
        func.func public @main(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
          %0 = stablehlo.add %a, %b : tensor<4xf32>
          return %0 : tensor<4xf32>
        }
      }
    ";

    let loaded = client.compile_and_load("mlir", mlir, b"")?;
    let outputs = loaded.execute_on_device(DeviceId(0), MemorySpaceId(0), &[&b0, &b1])?;
    let raw = outputs
        .first()
        .map(|buf| buf.to_host_bytes())
        .transpose()?
        .unwrap_or_default();

    Ok(raw
        .chunks_exact(4)
        .map(|c| {
            let mut arr = [0_u8; 4];
            arr.copy_from_slice(c);
            f32::from_le_bytes(arr)
        })
        .collect())
}
```

---

## 6. C11 ABI (`libtpu.h`)

`cpp/src/libtpu.h` compiles cleanly under `-std=c11 -Wall -Wextra -Wpedantic -Wconversion -Werror` (verified in `cpp/tests/test_libtpu_c11.c`):

```c
#include "libtpu.h"
#include <stddef.h>

int verify_pjrt_c11_lifecycle(void) {
  const PJRT_Api *api = GetPjrtApi();
  PJRT_Plugin_Initialize_Args init_args = {
      sizeof(PJRT_Plugin_Initialize_Args),
      NULL,
  };
  if (api->pjrt_plugin_initialize(&init_args) != NULL) {
    return 1;
  }

  PJRT_Client_Create_Args create_args = {
      sizeof(PJRT_Client_Create_Args),
      NULL, NULL, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
  };
  if (api->pjrt_client_create(&create_args) != NULL) {
    return 2;
  }

  PJRT_Client_Destroy_Args destroy_args = {
      sizeof(PJRT_Client_Destroy_Args),
      NULL,
      create_args.client,
  };
  return api->pjrt_client_destroy(&destroy_args) == NULL ? 0 : 3;
}
```

- **OpenXLA PJRT v0.117 (`GetPjrtApi()`)**:
  - `PJRT_Api` (`1,144` bytes, 138 function pointers).
  - All **18 chained TPU extensions**:
    1. `PJRT_FFI_Extension` (`ext_type = 5`, `48` B)
    2. `PJRT_XlaTransform_Extension` (`ext_type = 24`, `56` B)
    3. `PJRT_HostMemoryAllocator_Extension` (`ext_type = 23`, `32` B)
    4. `PJRT_MultiSlice_Extension` (`ext_type = 22`, `64` B)
    5. `PJRT_Collectives_Extension` (`ext_type = 21`, `96` B)
    6. `PJRT_AbiVersion_Extension` (`ext_type = 20`, `120` B)
    7. `PJRT_Shardings_Extension` (`ext_type = 19`, `40` B)
    8. `PJRT_Megascale_Extension` (`ext_type = 18`, `288` B)
    9. `PJRT_TpuExecutable_Extension` (`ext_type = 17`, `96` B)
    10. `PJRT_TpuTopology_Extension` (`ext_type = 16`, `280` B)
    11. `PJRT_Callback_Extension` (`ext_type = 14`, `40` B)
    12. `PJRT_PhaseCompile_Extension` (`ext_type = 9`, `192` B)
    13. `PJRT_CrossHostTransfers_Extension` (`ext_type = 12`, `56` B)
    14. `PJRT_ExecutableMetadata_Extension` (`ext_type = 13`, `40` B)
    15. `PJRT_MemoryDescriptions_Extension` (`ext_type = 6`, `40` B)
    16. `PJRT_Layouts_Extension` (`ext_type = 4`, `80` B)
    17. `PJRT_RawBuffer_Extension` (`ext_type = 8`, `80` B)
    18. `PJRT_Profiler_Extension` (`ext_type = 1`, `40` B) + `PLUGIN_Profiler_Api` (`104` B)
- **TPU SDK, Driver & Legacy TensorFlow C Symbols**: `GetLibtpuSdkApi()` (`144` B, `17` slots), `TpuDriver_Open` (`TpuDriverFn`, `224` B, `28` slots), `TfTpu_*`, `TpuTopology_*`, `TpuProfiler_*`, `TpuEmbeddingEngine_*` (all `75` Google `libtpu.so` `.dynsym` exports + `3` legacy `TpuDriver_*`), and `45` ergonomic `libtpu_rs_*` C helpers (`522` exported C symbols total in `libtpu-c-api`).

---

## 7. C++20 Zero-Exception RAII & `Result<T>` SDK (`libtpu.hpp`)

`cpp/src/libtpu.hpp` and `cpp/src/libtpu.cpp` build with `-std=c++20 -fno-exceptions -fno-rtti -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Werror` and provide move-only RAII wrappers (`Plugin`, `Client`, `TopologyDescription`, `Device`, `DeviceDescription`, `Memory`, `Buffer`, `RawBuffer`, `AsyncTransferManager`, `CopyToDeviceStream`, `DlPackTensor`, `Executable`, `LoadedExecutable`, `PhaseCompiler`, `XlaTransform`, `ExecuteContext`, `Event`, `CollectiveCommunicator`, `MegaScaleContext`, `MultiSliceConfig`, `FfiCustomCallRegistry`, `Profiler`, `EmbeddingEngine`, `SdkClient`) plus monadic `libtpu::Result<T>`:

```cpp
#include "libtpu.hpp"
#include <cstdint>
#include <cstdio>
#include <string_view>

int main() {
  libtpu::Client client;
  if (!libtpu::Client::create(&client).is_ok()) {
    return 1;
  }

  const int64_t dims[1] = {4};
  const float a[4] = {1.0F, 2.0F, 3.0F, 4.0F};
  const float b[4] = {5.0F, 6.0F, 7.0F, 8.0F};

  libtpu::Buffer b0;
  libtpu::Buffer b1;
  (void)client.buffer_from_host(0, libtpu::DataType::F32, dims, 1,
                                reinterpret_cast<const uint8_t *>(a),
                                sizeof(a), &b0);
  (void)client.buffer_from_host(0, libtpu::DataType::F32, dims, 1,
                                reinterpret_cast<const uint8_t *>(b),
                                sizeof(b), &b1);

  constexpr std::string_view kMlir =
      "module @m { func.func public @main(%x: tensor<4xf32>, %y: tensor<4xf32>)"
      " -> tensor<4xf32> { %0 = stablehlo.multiply %x, %y : tensor<4xf32> "
      "return %0 : tensor<4xf32> } }";

  libtpu::LoadedExecutable exec;
  if (!client.compile("mlir", kMlir, &exec).is_ok()) {
    return 2;
  }

  libtpu::Buffer out;
  if (!exec.execute_binary(0, b0, b1, &out).is_ok()) {
    return 3;
  }

  float host_out[4] = {};
  (void)out.to_host(reinterpret_cast<uint8_t *>(host_out), sizeof(host_out));

  // Zero-exception monadic Result<T> chaining
  const libtpu::Result<float> scaled =
      libtpu::Result<float>(host_out[3]).map([](const float &v) noexcept {
        return v * 2.0F;
      });
  std::printf("scaled[3] = %.1f\n", static_cast<double>(scaled.value_or(0.0F)));
  return 0;
}
```

---

## 8. Supported Cloud TPU Generations (`v2` – `v7x`)

| Generation | Codename / ISA | PCIe Device ID (`0x1ae0:*`) | Kernel Driver | Hardware ABI | HBM / Logical Device | VMEM / Core | MXU Array | Physical `TensorCores` / `SparseCores` (per Chip) | Logical PJRT Devices / Chip |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TPU v2** | Jellyfish (`jfc`) | `0x0027` | Gasket (`0xdc`) | `0x10000` | `8 GiB` (`16 GiB`/chip) | `16 MiB` | `128x128` | `2` / `0` *(BarnaCore)* | `2` |
| **TPU v3** | Dragonfish (`dfc`) | `0x0027` | Gasket (`0xdc`) | `0x10000` | `16 GiB` (`32 GiB`/chip) | `16 MiB` | `128x128` | `2` / `0` *(BarnaCore)* | `2` |
| **TPU v4** | Pufferfish (`pfc`) | `0x0056` | Gasket (`0xdc`) / VFIO (`0x3b`) | `0x10001` | `32 GiB` | `32 MiB` | `128x128` | `2` *(Megacore)* / `4` | `1` |
| **TPU v4lite** | Pufferfish Lite (`pfc`) | `0x005e` | VFIO (`0x3b`) | `0x10001` | `16 GiB` | `16 MiB` | `128x128` | `1` / `0` | `1` |
| **TPU v5e** | Viperfish Lite (`vfc`) | `0x0063` | VFIO (`0x3b`) | `0x10001` | `16 GiB` | `32 MiB` | `128x128` | `1` / `2` | `1` |
| **TPU v5p** | Viperfish (`vfc`) | `0x0062` | VFIO (`0x3b`) | `0x10003` | `95 GiB` | `64 MiB` | `128x128` | `2` *(Megacore)* / `4` | `1` |
| **TPU v6e** | Ghostfish / Trillium (`gfc`) | `0x006f` | VFIO (`0x3b`) | `0x10003` | `32 GiB` | `64 MiB` | `256x256` | `1` / `2` | `1` |
| **TPU7x** | Ironwood (`iwc`) | `0x0076` | VFIO (`0x3b`) | `0x10003` | `192 GiB` | `64 MiB` | `256x256` | `2` *(Megacore)* / `4` | `1` |

---

## 9. Diagnostics, CSR Inspection & Verification

Run all formatting, compilation, Clippy/Rustdoc/Ruff/Mypy/Doxygen lints, binary ABI verification, 1:1 `.proto` <-> `.rs` schema sync, Rust/Python/C11/C++20 + live JAX tests, and 100.0% doc/line/function/branch coverage checks with:

```bash
python3 scripts/pre_commit.py all --check
```

Individual stages can also be run directly (`fmt`, `build`, `clippy`, `abi`, `proto-sync`, `test`, `shields`).

# `libtpu-rs` Usage & Integration Guide

`libtpu-rs` is a clean-room, pure-Rust drop-in replacement for the Google Cloud TPU runtime library (`libtpu` v0.0.50). It exports the exact C11 binary ABI expected by OpenXLA PJRT (`GetPjrtApi` v0.116 with 138 vtable slots and 17 chained TPU extensions), `LibtpuSdk_Api` (`GetLibtpuSdkApi`), `TpuDriver_*` (`TpuDriver_Initialize`), `TpuEmbeddingEngine_*` (`SparseCore`), `TpuProfiler_*` (`XSpace`), `TpuTopology_*`, and `TfTpu_*` entrypoints.

Because `libtpu-rs` has zero closed-source binary dependencies and includes both a Linux Gasket/VFIO hardware backend (`/dev/accel*`, `/dev/vfio/*`) and a deterministic in-memory `SimulatedTpuHal` (supporting TPU `v2`, `v3`, `v4`, `v5e`, `v5p`, `v6e`, and `v7x`), you can use it either on physical Cloud TPU VMs or locally on macOS/Linux/Windows workstations for offline TPU compilation, topology simulation, profiling, and testing.

---

## Table of Contents

1. [Building from Source](#1-building-from-source)
2. [Hacking JAX, TensorFlow, and PyTorch/XLA (`torch-tpu`) to Use `libtpu-rs`](#2-hacking-jax-tensorflow-and-pytorchxla-torch-tpu-to-use-libtpu-rs)
   - [2.1 JAX (`jax` / `jaxlib`)](#21-hacking-a-jax-install-to-use-libtpu-rs)
   - [2.2 PyTorch/XLA (`torch_xla` / `torch-tpu`)](#22-hacking-a-pytorchxla-torch-tpu-install-to-use-libtpu-rs)
   - [2.3 TensorFlow (`tensorflow` / `tensorflow-tpu`)](#23-hacking-a-tensorflow-install-to-use-libtpu-rs)
   - [2.4 Universal Dynamic Linker Override (`LD_PRELOAD` / `DYLD_INSERT_LIBRARIES`)](#24-universal-dynamic-linker-override-ld_preload--dyld_insert_libraries)
   - [2.5 Drop-in Wheel Replacement in Virtual Environments & Containers](#25-drop-in-wheel-replacement-in-virtual-environments--containers)
   - [2.6 Extreme Binary Surgery: In-Place `.so` / `.dylib` Replacement (`libtpu.so` & `sdk.so`)](#26-extreme-binary-surgery-in-place-so--dylib-replacement-libtpuso--sdkso)
   - [2.7 ELF `DT_SONAME` / `DT_RPATH` & Mach-O `LC_ID_DYLIB` Header Patching](#27-elf-dt_soname--dt_rpath--mach-o-lc_id_dylib-header-patching)
   - [2.8 Repacking an Official `libtpu-*.whl` Archive with `libtpu-rs` Binaries](#28-repacking-an-official-libtpu-whl-archive-with-libtpu-rs-binaries)
   - [2.9 Bypassing Hardware PCI/Sysfs Detection Gates (`0x1ae0` & `/dev/accel*` Spoofing)](#29-bypassing-hardware-pcisysfs-detection-gates-0x1ae0--devaccel-spoofing)
   - [2.10 Intercepting Hardcoded `dlopen("/lib/libtpu.so", ...)` via a `LD_PRELOAD` C Shim](#210-intercepting-hardcoded-dlopenliblibtpuso--via-a-ld_preload-c-shim)
3. [Environment Variables & `LIBTPU_INIT_ARGS` Reference](#3-environment-variables--libtpu_init_args-reference)
4. [Python SDK Usage (`libtpu` & `libtpu.sdk`)](#4-python-sdk-usage-libtpu--libtpusdk)
   - [4.1 Core PJRT Client, Buffers, Compilation & Execution](#41-core-pjrt-client-buffers-compilation--execution)
   - [4.2 Hardware & Runtime Profiling (`TpuProfiler` / `XSpace`)](#42-hardware--runtime-profiling-tpuprofiler--xspace)
   - [4.3 Telemetry, Slice Coordinates & Diagnostics (`libtpu.sdk`)](#43-telemetry-slice-coordinates--diagnostics-libtpusdk)
5. [Rust Crate Usage (`libtpu-core`)](#5-rust-crate-usage-libtpu-core)
   - [5.1 High-Level `PjrtCoreClient` & `SimulatedTpuHal`](#51-high-level-pjrtcoreclient--simulatedtpuhal)
   - [5.2 Linux Kernel Gasket & VFIO Hardware Backend](#52-linux-kernel-gasket--vfio-hardware-backend)
   - [5.3 `SparseCore` Embedding Engine, Legacy `TpuDriver` & `TpuCoreProfiler`](#53-sparsecore-embedding-engine-legacy-tpudriver--tpucoreprofiler)
6. [C11 API Usage (`libtpu.h`)](#6-c11-api-usage-libtpuh)
   - [6.1 Ergonomic `libtpu_rs_*` C11 Wrappers](#61-ergonomic-libtpu_rs_-c11-wrappers)
   - [6.2 Raw OpenXLA `GetPjrtApi()` Vtable & TPU Extensions](#62-raw-openxla-getpjrtapi-vtable--tpu-extensions)
7. [C++20 RAII SDK Usage (`libtpu.hpp`)](#7-c20-raii-sdk-usage-libtpuhpp)
8. [Supported HLO / `StableHLO` Operations & Compiler Pipeline](#8-supported-hlo--stablehlo-operations--compiler-pipeline)
9. [Troubleshooting & Verification](#9-troubleshooting--verification)

---

## 1. Building from Source

### 1.1 Build the Shared Library (`libtpu.so` / `libtpu.dylib` / `tpu.dll`)

Compile the workspace in release mode:

```bash
cargo build --workspace --release
```

This produces the drop-in C ABI shared library in `target/release/`:
- **Linux**: `target/release/libtpu.so`
- **macOS**: `target/release/libtpu.dylib`
- **Windows**: `target/release/tpu.dll`

### 1.2 Install the Python Package (`libtpu`)

Install the Python package in editable mode (or build a wheel) and copy/symlink the compiled shared library into `python/libtpu/` if deploying standalone:

```bash
# Editable install (automatically locates target/release/libtpu.{so,dylib} or target/debug/libtpu.{so,dylib})
pip install -e .

# Optional: bundle the compiled shared library directly inside the Python package directory
# On Linux:
cp target/release/libtpu.so python/libtpu/libtpu.so
# On macOS:
cp target/release/libtpu.dylib python/libtpu/libtpu.dylib
```

### 1.3 Build the C11 / C++20 Library & Tests

```bash
cmake -B cpp/build -S cpp
cmake --build cpp/build
ctest --test-dir cpp/build --output-on-failure
```

---

## 2. Hacking JAX, TensorFlow, and PyTorch/XLA (`torch-tpu`) to Use `libtpu-rs`

Official ML frameworks (`jax`, `torch_xla`, `tensorflow`) load `libtpu.so` dynamically at runtime via `dlopen()` and resolve `GetPjrtApi` (or legacy `TfTpu_Initialize` / `TpuDriver_Initialize`). Because `libtpu-rs` exports the exact symbol table and struct layouts of `libtpu` v0.0.50, you can hijack any existing framework installation to run against `libtpu-rs` without recompiling the framework.

Below are the specific mechanisms for each framework.

---

### 2.1 Hacking a JAX Install to Use `libtpu-rs`

JAX discovers the TPU PJRT plugin in three ways:
1. Checking `--xla_tpu_library_path` or `TPU_LIBRARY_PATH` environment variable.
2. Importing `libtpu` in Python and calling `libtpu.get_library_path()`.
3. Scanning `jax_plugins` entrypoints registered in the Python environment.

#### Method A: Environment Variable Override (Zero File Modifications)

Point `TPU_LIBRARY_PATH` and `PJRT_DEVICE` to your compiled `libtpu-rs` binary before starting Python:

```bash
# Linux:
export TPU_LIBRARY_PATH="/abs/path/to/libtpu-rs/target/release/libtpu.so"
# macOS:
export TPU_LIBRARY_PATH="/abs/path/to/libtpu-rs/target/release/libtpu.dylib"

# Optional: configure simulated topology (e.g., 8x TPU v6e chips across 2 hosts)
export TPU_ACCELERATOR_TYPE="v6e"
export TPU_CHIPS_PER_HOST_BOUNDS="2,2,1"
export TPU_HOST_BOUNDS="1,1,1"
export LIBTPU_INIT_ARGS="--tpu_generation=v6e --xla_tpu_allow_async_allocations=true"

# Force JAX to select the TPU backend
export JAX_PLATFORMS="tpu"
```

In Python, you can also configure JAX before importing `jax.numpy` by importing `libtpu-rs`'s `libtpu` package first:

```python
import os
import libtpu

# Configure topology and flags before JAX initializes the PJRT plugin
libtpu.initialize([
    "--tpu_generation=v6e",
    "--tpu_chips_per_host_bounds=2,2,1",
    "--tpu_host_bounds=1,1,1",
])
os.environ["JAX_PLATFORMS"] = "tpu"

import jax
import jax.numpy as jnp

print("Library path:", libtpu.get_library_path())
print("JAX devices:", jax.devices())
```

#### Method B: Registering `libtpu-rs` Explicitly via `jax._src.xla_bridge`

If you are running on a platform (such as macOS or a CPU-only Linux container) where `jaxlib` does not automatically register the `tpu` PJRT C API plugin unless a physical PCI device `0x1ae0` is detected, you can force-register the `libtpu-rs` PJRT plugin directly with `jaxlib.xla_extension`:

```python
import os
import libtpu
import jax._src.xla_bridge as xb
from jax._src.lib import xla_client

# 1. Configure simulated TPU hardware generation & mesh
libtpu.initialize(["--tpu_generation=v5e", "--tpu_chips_per_host_bounds=2,2,1"])
lib_path = libtpu.get_library_path()

# 2. Force-load the PJRT C API plugin into XLA bridge
xb.register_plugin(
    "tpu",
    priority=500,
    library_path=lib_path,
    options=None,
)
os.environ["JAX_PLATFORMS"] = "tpu"

import jax
print(jax.devices("tpu"))
```

#### Method C: Overwriting the Official `libtpu` Wheel in `site-packages`

If `jax[tpu]` was installed via `pip install "jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"`, it pulls in the official `libtpu-nightly` or `libtpu` package containing `site-packages/libtpu/libtpu.so` and `site-packages/libtpu/sdk.so`.

You can replace it in-place with `libtpu-rs`:

```bash
# 1. Find where the official libtpu package is installed
SITE_LIBTPU=$(python3 -c "import libtpu, pathlib; print(pathlib.Path(libtpu.__file__).parent)")
echo "Official libtpu directory: ${SITE_LIBTPU}"

# 2. Back up the official binary (optional)
mv "${SITE_LIBTPU}/libtpu.so" "${SITE_LIBTPU}/libtpu.so.official.bak"

# 3. Symlink or copy libtpu-rs into place
ln -sf "$(pwd)/target/release/libtpu.so" "${SITE_LIBTPU}/libtpu.so"

# Or replace the entire Python package with libtpu-rs:
pip install --force-reinstall --no-deps -e /abs/path/to/libtpu-rs
cp target/release/libtpu.so python/libtpu/libtpu.so
```

---

### 2.2 Hacking a PyTorch/XLA (`torch_xla` / `torch-tpu`) Install to Use `libtpu-rs`

PyTorch/XLA (`torch_xla`) uses OpenXLA PJRT to drive Cloud TPUs when `PJRT_DEVICE=TPU` is set. `torch_xla` locates `libtpu.so` via `TPU_LIBRARY_PATH` or via the `libtpu` Python module (`libtpu.get_library_path()`).

#### Method A: Environment Variables (`PJRT_DEVICE=TPU` + `TPU_LIBRARY_PATH`)

```bash
export PJRT_DEVICE="TPU"
export TPU_LIBRARY_PATH="/abs/path/to/libtpu-rs/target/release/libtpu.so"
export LIBTPU_INIT_ARGS="--tpu_generation=v5p --tpu_chips_per_host_bounds=2,2,1"
```

Then run your PyTorch/XLA workload as normal:

```python
import os
import libtpu

# Ensure TPU_LIBRARY_PATH points to libtpu-rs before importing torch_xla
libtpu.initialize(["--tpu_generation=v5p", "--tpu_chips_per_host_bounds=2,2,1"])
os.environ["PJRT_DEVICE"] = "TPU"

import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

print("Active TPU library:", os.environ["TPU_LIBRARY_PATH"])
device = xm.xla_device()
print("PyTorch/XLA device:", device)
```

#### Method B: Patching `torch_xla` Bundled `libtpu.so`

Some `torch_xla` / `torch-tpu` distributions bundle their own copy of `libtpu.so` inside `site-packages/torch_xla/lib/libtpu.so` or `site-packages/libtpu/libtpu.so`. To replace all bundled copies in a virtual environment or Conda environment:

```bash
# Find every libtpu.so in the active Python environment and symlink to libtpu-rs
PYTHON_PREFIX=$(python3 -c "import sys; print(sys.prefix)")
find "${PYTHON_PREFIX}" -name "libtpu.so" -exec sh -c '
  for target do
    echo "Replacing ${target} -> /abs/path/to/libtpu-rs/target/release/libtpu.so"
    mv "${target}" "${target}.bak"
    ln -sf "/abs/path/to/libtpu-rs/target/release/libtpu.so" "${target}"
  done
' sh {} +
```

#### Method C: Multi-Slice / Megascale Simulation in `torch_xla`

`libtpu-rs` implements all 29 entrypoints of `PJRT_Megascale_Extension` (`ext_type = 18`) and `PJRT_MultiSlice_Extension` (`ext_type = 26`). You can test multi-slice PyTorch/XLA topologies locally by exporting:

```bash
export PJRT_DEVICE="TPU"
export TPU_LIBRARY_PATH="/abs/path/to/libtpu-rs/target/release/libtpu.so"
export MEGASCALE_NUM_SLICES="4"
export MEGASCALE_SLICE_ID="0"
export TPU_HOST_BOUNDS="2,2,1"
export TPU_CHIPS_PER_HOST_BOUNDS="2,2,1"
export TPU_WORKER_ID="0"
export TPU_WORKER_HOSTNAMES="127.0.0.1,127.0.0.1,127.0.0.1,127.0.0.1"
```

---

### 2.3 Hacking a TensorFlow Install to Use `libtpu-rs`

TensorFlow (`tensorflow` / `tensorflow-tpu`) interacts with `libtpu.so` through three ABI layers, all of which are implemented by `libtpu-rs`:
1. **PJRT Plugin (`GetPjrtApi`)**: Used by modern TensorFlow XLA/PJRT device registration (`NEXT_PLUGGABLE_DEVICE`).
2. **Legacy `TfTpu_*` & `TpuTopology_*` C APIs**: `TfTpu_Initialize`, `SetGlobalTPUArrayOp_DoWork`, `ConfigureDistributedTpuOp_DoWork`, `WaitForDistributedTpuOp_DoWork`, `InitializeHostForDistributedTpuOp_DoWork`, `DisconnectDistributedTpuChipsOp_DoWork`, `ShutdownDistributedTpuOp_DoWork`, `TfTpuOrdinalSelector_*`, and `TpuConfigurationApi_*`.
3. **`SparseCore` Embedding Engine (`TpuEmbeddingEngine_*`)**: Used by `tf.tpu.experimental.embedding.TPUEmbedding` (`ExecutePartitioner`, `ConfigureMemory`, `CollateMemory`, `ConfigureHost`, `ConnectHosts`, `Finalize`, `WriteParameters`, `ReadParameters`, `EnqueueTensorBatch`, and the 5 XLA HLO computation generators).
4. **Legacy `TpuDriver_*` (`TpuDriver_Initialize`)**: Used by `tf.profiler` and `TpuDriver`.

#### Method A: Setting `TPU_LIBRARY_PATH` Before Importing `tensorflow`

TensorFlow's C++ `tpu_initializer_helper.cc` reads `TPU_LIBRARY_PATH` (defaulting to `/lib/libtpu.so` if unset) when loading `libtpu.so`:

```bash
export TPU_LIBRARY_PATH="/abs/path/to/libtpu-rs/target/release/libtpu.so"
export LIBTPU_INIT_ARGS="--tpu_generation=v5e --tpu_chips_per_host_bounds=2,2,1"
```

In Python:

```python
import os
import libtpu

# Must be called before `import tensorflow as tf`
libtpu.initialize(["--tpu_generation=v5e", "--tpu_chips_per_host_bounds=2,2,1"])

import tensorflow as tf

# Initialize the TPU cluster resolver (works against libtpu-rs local runtime)
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)
print("Number of TPU replicas:", strategy.num_replicas_in_sync)
```

#### Method B: System-Wide `/lib/libtpu.so` Symlink on Linux Containers / VMs

Many TensorFlow builds hardcode `/lib/libtpu.so` or `/usr/lib/libtpu.so` as the fallback when `TPU_LIBRARY_PATH` is not propagated into subprocess workers:

```bash
sudo ln -sf /abs/path/to/libtpu-rs/target/release/libtpu.so /lib/libtpu.so
sudo ln -sf /abs/path/to/libtpu-rs/target/release/libtpu.so /usr/lib/libtpu.so
sudo ldconfig
```

---

### 2.4 Universal Dynamic Linker Override (`LD_PRELOAD` / `DYLD_INSERT_LIBRARIES`)

If a compiled binary, custom C++ trainer, or subprocess ignores `TPU_LIBRARY_PATH` and loads `libtpu.so` from a hardcoded path or `RPATH`, you can force the OS dynamic linker to interpose `libtpu-rs` before any other library:

#### On Linux (`LD_PRELOAD` + `LD_LIBRARY_PATH`):

```bash
export LD_LIBRARY_PATH="/abs/path/to/libtpu-rs/target/release:${LD_LIBRARY_PATH}"
export LD_PRELOAD="/abs/path/to/libtpu-rs/target/release/libtpu.so"
python3 train.py
```

#### On macOS (`DYLD_INSERT_LIBRARIES` + `DYLD_LIBRARY_PATH`):

```bash
export DYLD_LIBRARY_PATH="/abs/path/to/libtpu-rs/target/release:${DYLD_LIBRARY_PATH}"
export DYLD_INSERT_LIBRARIES="/abs/path/to/libtpu-rs/target/release/libtpu.dylib"
python3 train.py
```

---

### 2.5 Drop-in Wheel Replacement in Virtual Environments & Containers

Because `libtpu-rs`'s `pyproject.toml` defines `name = "libtpu"` and `version = "0.0.50"`, it satisfies `pip` dependency checks for packages that depend on `libtpu==0.0.50` (such as `jax[tpu]` or `torch_xla`).

To replace the official `libtpu` wheel in any `venv`, `uv`, `poetry`, or Docker image:

```bash
# 1. Build the Rust shared library
cargo build --workspace --release

# 2. Copy the shared library into the Python package folder so the wheel is self-contained
cp target/release/libtpu.so python/libtpu/ 2>/dev/null || cp target/release/libtpu.dylib python/libtpu/

# 3. Force-reinstall into the target Python environment without touching JAX/PyTorch/TF
pip install --force-reinstall --no-deps .
```

Verify that your Python environment is now backed by `libtpu-rs`:

```bash
python3 -c "import libtpu, ctypes; lib = ctypes.CDLL(libtpu.get_library_path()); print('Loaded libtpu-rs from:', libtpu.get_library_path()); assert hasattr(lib, 'libtpu_rs_client_create')"
```

---

### 2.6 Extreme Binary Surgery: In-Place `.so` / `.dylib` Replacement (`libtpu.so` & `sdk.so`)

Sometimes environment variables (`TPU_LIBRARY_PATH`) are stripped or ignored by cluster launchers (`mpirun`, `slurm`, `ray`, `torchrun`, GKE TPU webhook injectors, or `tensorflow` C++ worker daemons). In those environments, **physically overwriting the `.so` files on disk** is the most bulletproof way to guarantee every process and subprocess uses `libtpu-rs`.

The official Google `libtpu` wheel contains **two** shared objects inside `site-packages/libtpu/`:
1. `site-packages/libtpu/libtpu.so` (~594 MB closed-source C++ PJRT & TPU driver binary)
2. `site-packages/libtpu/sdk.so` (CPython C-extension module exporting `PyInit_sdk`, which dynamically links against `libtpu.so` via `RPATH=$ORIGIN` and resolves `GetLibtpuSdkApi`)

Because `libtpu-rs`'s `libtpu.so` exports **both** the PJRT/driver symbols (`GetPjrtApi`, `TfTpu_*`, `TpuDriver_*`) **and** `GetLibtpuSdkApi` (with the exact 144-byte `LibtpuSdk_Api` vtable expected by the official `sdk.so` binary):

#### Option 1: Replace `libtpu.so` Only (Keep Official `sdk.so` Running on Top of `libtpu-rs`!)

Because `libtpu-rs` is 100% ABI-compatible with `GetLibtpuSdkApi`, you can overwrite `site-packages/libtpu/libtpu.so` and leave the official C++ `site-packages/libtpu/sdk.so` untouched—the official `sdk.so` will dynamically link against `libtpu-rs`'s `libtpu.so` and call `GetLibtpuSdkApi()` seamlessly:

```bash
SITE_LIBTPU=$(python3 -c "import libtpu, pathlib; print(pathlib.Path(libtpu.__file__).parent)")

# Overwrite the 594 MB official libtpu.so with the ~2 MB Rust libtpu.so
# Important: Use `cp -f` or `install` (which unlinks first) rather than writing in-place
# so any running process with the old .so mmap'd does not SIGBUS!
install -m 0755 target/release/libtpu.so "${SITE_LIBTPU}/libtpu.so"
```

> **Critical Linux `mmap` / `ETXTBSY` Warning**: Never use `cat target/release/libtpu.so > "${SITE_LIBTPU}/libtpu.so"` on a live system. Overwriting an already-`mmap`'d shared library in-place mutates active pages and triggers `SIGBUS` in running Python processes. Always `rm -f` + `cp` (or `install -m 0755` / `mv -f`), which atomically unlinks the old inode and creates a new inode.

#### Option 2: Replace Both `libtpu.so` AND `sdk.so` with Pure `libtpu-rs`

If you want zero closed-source C++ code left in `site-packages/libtpu/` (or if you are running on a Python version like 3.12/3.13/3.14 where the official `cp311`-compiled `sdk.so` refuses to import):

```bash
SITE_LIBTPU=$(python3 -c "import libtpu, pathlib; print(pathlib.Path(libtpu.__file__).parent)")

# 1. Remove the official C-extension sdk.so and __init__.py
rm -f "${SITE_LIBTPU}/sdk.so" "${SITE_LIBTPU}/libtpu.so"

# 2. Copy libtpu-rs's Python package + pure-Python ctypes `sdk/` subpackage + Rust `libtpu.so`
cp -rf python/libtpu/* "${SITE_LIBTPU}/"
install -m 0755 target/release/libtpu.so "${SITE_LIBTPU}/libtpu.so"
```

---

### 2.7 ELF `DT_SONAME` / `DT_RPATH` & Mach-O `LC_ID_DYLIB` Header Patching

If a precompiled C++ binary (such as a custom TensorFlow serving binary, `sdk.so`, or `libtorch_xla.so`) links directly against `libtpu.so` via ELF `DT_NEEDED` rather than `dlopen()`, the dynamic loader checks `DT_SONAME` (on Linux) or `LC_ID_DYLIB` (on macOS).

#### Linux (`patchelf`): Matching `DT_SONAME` and Injecting `RPATH`

The official `libtpu.so` has `SONAME` set to `libtpu.so`. You can verify or stamp `SONAME` on `libtpu-rs` and patch consumer binaries (`libtorch_xla.so`, `_xlAC.so`, `xla_extension.so`) so they always resolve `libtpu-rs` first:

```bash
# 1. Ensure DT_SONAME on libtpu-rs is set to "libtpu.so"
patchelf --set-soname libtpu.so target/release/libtpu.so

# 2. Force jaxlib's xla_extension.so or torch_xla's _XLAC.so to search libtpu-rs's directory first
XLA_EXT=$(python3 -c "import jaxlib.xla_extension as x; print(x.__file__)")
patchelf --set-rpath "/abs/path/to/libtpu-rs/target/release:\$ORIGIN" "${XLA_EXT}"
```

You can also pass `-C link-arg=-Wl,-soname,libtpu.so` directly via `RUSTFLAGS` when compiling `libtpu-rs` on Linux:

```bash
RUSTFLAGS="-C link-arg=-Wl,-soname,libtpu.so" cargo build --workspace --release
```

#### macOS (`install_name_tool`): Rewriting `@rpath/libtpu.dylib`

On macOS, if a binary links against `@rpath/libtpu.dylib` or a hardcoded path:

```bash
# Set the install_name ID of libtpu.dylib
install_name_tool -id "@rpath/libtpu.dylib" target/release/libtpu.dylib

# Point any consumer Mach-O binary directly to libtpu-rs
install_name_tool -change "@rpath/libtpu.dylib" \
  "/abs/path/to/libtpu-rs/target/release/libtpu.dylib" \
  /path/to/consumer_binary.so
```

---

### 2.8 Repacking an Official `libtpu-*.whl` Archive with `libtpu-rs` Binaries

In locked-down production environments where `pip install --require-hashes` or an internal PyPI mirror is used, `pip` validates the wheel filename, metadata, and `RECORD` SHA-256 digests. You can unpack an official `libtpu-0.0.50-*.whl`, swap the binary with `libtpu-rs`, recompute `.dist-info/RECORD` hashes using `wheel pack`, and deploy the modified wheel:

```bash
# 1. Unpack the official wheel
wheel unpack libtpu-0.0.50-cp311-cp311-manylinux_2_31_x86_64.whl -d /tmp/wheel_edit

# 2. Replace the official libtpu.so (and optionally sdk.so) with libtpu-rs
cp -f target/release/libtpu.so /tmp/wheel_edit/libtpu-0.0.50/libtpu/libtpu.so
rm -f /tmp/wheel_edit/libtpu-0.0.50/libtpu/sdk.so
cp -rf python/libtpu/* /tmp/wheel_edit/libtpu-0.0.50/libtpu/

# 3. Repack the wheel (automatically regenerates SHA-256 hashes in RECORD)
wheel pack /tmp/wheel_edit/libtpu-0.0.50 -d /tmp/repacked_wheels/
```

---

### 2.9 Bypassing Hardware PCI/Sysfs Detection Gates (`0x1ae0` & `/dev/accel*` Spoofing)

Replacing `libtpu.so` is enough for `libtpu` itself, **but** higher-level Python wrappers in `jax` (`jax/_src/clusters/cloud_tpu_cluster.py`), `torch_xla`, and `tensorflow` sometimes check whether the host machine is a "real" Cloud TPU VM **before** they even call `dlopen("libtpu.so")`!

Specifically, framework startup code may check:
1. Whether `/dev/accel0` or `/dev/vfio/0` exists on the filesystem.
2. Whether `/sys/bus/pci/devices/*/vendor` contains Google's PCI vendor ID (`0x1ae0`).
3. Whether GCE metadata (`http://metadata.google.internal/computeMetadata/v1/instance/attributes/tpu-env`) is reachable.

Here is how to bypass every one of those framework pre-checks on a non-TPU workstation or container:

#### Gate 1: Bypassing JAX's TPU Probing Check

In modern JAX, if `JAX_PLATFORMS=tpu` is set explicitly, JAX skips the automatic PCI device check and immediately loads `libtpu.so` via `TPU_LIBRARY_PATH`. If a library checks `jax.extend.backend.backends()`, ensure these three variables are exported together:

```bash
export TPU_LIBRARY_PATH="/abs/path/to/libtpu-rs/target/release/libtpu.so"
export JAX_PLATFORMS="tpu"
export TPU_SKIP_MDS_QUERY="true"
```

#### Gate 2: Creating Dummy `/dev/accel0` & `/dev/vfio/0` Nodes (Linux Container / Root)

If a framework or launcher checks `os.path.exists("/dev/accel0")` or `os.path.exists("/dev/vfio/vfio")` before initializing PJRT:

```bash
# Inside a Docker container or Linux host, create harmless character/regular file stubs:
sudo mkdir -p /dev/vfio
sudo touch /dev/accel0 /dev/accel1 /dev/accel2 /dev/accel3
sudo touch /dev/vfio/vfio /dev/vfio/0 /dev/vfio/1 /dev/vfio/2 /dev/vfio/3
sudo chmod 666 /dev/accel* /dev/vfio/*
```

> **Note**: When `libtpu-rs`'s `PJRT_Client_Create` is called, it uses `SimulatedTpuHal` automatically for full in-memory HLO execution, while `LinuxKernelTpuHal` handles Gasket (`0xdc`) and VFIO (`0x3b`) ioctls when targeting character devices.

#### Gate 3: Spoofing Google PCI Vendor `0x1ae0` in Userspace (`sitecustomize.py`)

If a Python library (like `cloud-tpu-client` or a framework launcher) scans `/sys/bus/pci/devices/` or queries `http://metadata.google.internal` in Python before loading `libtpu.so`, drop a `sitecustomize.py` into your `PYTHONPATH` so it runs automatically at Python startup before any user code:

```python
# sitecustomize.py — Automatically loaded by Python at interpreter startup
import os
import pathlib
import libtpu

# 1. Force TPU_LIBRARY_PATH to libtpu-rs
os.environ["TPU_LIBRARY_PATH"] = libtpu.get_library_path()
os.environ.setdefault("JAX_PLATFORMS", "tpu")
os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "1")

# 2. Monkeypatch os.path.exists for hardcoded /dev/accel0 or /lib/libtpu.so checks
_orig_exists = os.path.exists

def _spoofed_exists(path: object) -> bool:
    s = str(path)
    if s in ("/dev/accel0", "/dev/vfio/0", "/dev/vfio/vfio", "/lib/libtpu.so"):
        return True
    return _orig_exists(path)

os.path.exists = _spoofed_exists
```

Run any unmodified training script with `PYTHONPATH=.:$PYTHONPATH python3 train.py` and the hook will transparently activate `libtpu-rs`.

---

### 2.10 Intercepting Hardcoded `dlopen("/lib/libtpu.so", ...)` via a `LD_PRELOAD` C Shim

If you do **not** have root access on a shared Linux cluster to write `/lib/libtpu.so`, and you are running a precompiled C++ binary that calls `dlopen("/lib/libtpu.so", RTLD_NOW)` with a hardcoded path (ignoring `TPU_LIBRARY_PATH`), `LD_PRELOAD` alone on `libtpu.so` might not stop `dlopen("/lib/libtpu.so")` from failing with `ENOENT` if `/lib/libtpu.so` does not exist on disk.

You can redirect any `dlopen()` call targeting `libtpu.so` to your `libtpu-rs` binary in userspace without root permissions using a 15-line `LD_PRELOAD` interceptor:

```c
// dlopen_redirect.c
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>

void *dlopen(const char *filename, int flags) {
  static void *(*real_dlopen)(const char *, int) = NULL;
  if (!real_dlopen) {
    real_dlopen = (void *(*)(const char *, int))dlsym(RTLD_NEXT, "dlopen");
  }
  if (filename && strstr(filename, "libtpu")) {
    const char *override = getenv("TPU_LIBRARY_PATH");
    if (override && override[0] != '\0') {
      return real_dlopen(override, flags);
    }
  }
  return real_dlopen(filename, flags);
}
```

Compile and preload it:

```bash
cc -shared -fPIC -O2 dlopen_redirect.c -ldl -o libdlopen_redirect.so
export TPU_LIBRARY_PATH="/abs/path/to/libtpu-rs/target/release/libtpu.so"
export LD_PRELOAD="$(pwd)/libdlopen_redirect.so"
./precompiled_tpu_binary
```

Every `dlopen()` call whose path contains `"libtpu"` (whether `/lib/libtpu.so`, `/usr/lib/libtpu.so`, or a `site-packages` path) will now unconditionally load `libtpu-rs`.

---

## 3. Environment Variables & `LIBTPU_INIT_ARGS` Reference

`libtpu-rs` parses both standard Cloud TPU environment variables and `LIBTPU_INIT_ARGS` CLI flags via `libtpu_core::config::TpuInitConfig`.

### 3.1 Environment Variables

| Environment Variable | Default | Description |
| :--- | :--- | :--- |
| `TPU_LIBRARY_PATH` | Auto-resolved | Explicit filesystem path to `libtpu.so` / `libtpu.dylib` / `tpu.dll`. Automatically set by `import libtpu`. |
| `LIBTPU_INIT_ARGS` | `""` | Whitespace-separated `--tpu_*` and `--xla_tpu_*` runtime flags (see table below). |
| `TPU_ACCELERATOR_TYPE` | `"v5e"` | Hardware generation (`v2`, `v3`, `v4`, `v5e` / `v5litepod`, `v5p`, `v6e` / `trillium`, `v7x` / `ironwood`). Accepts slice suffixes like `v5litepod-16` or `v6e-8`. |
| `TPU_CHIPS_PER_HOST_BOUNDS` | `"2,2,1"` | 3D torus dimensions `(x,y,z)` of TPU chips on each host (accepts `"2,2,1"` or `"2x2x1"`). |
| `TPU_HOST_BOUNDS` / `TPU_PROCESS_BOUNDS` | `"1,1,1"` | 3D torus dimensions `(x,y,z)` of hosts in the TPU pod slice. |
| `TPU_VISIBLE_DEVICES` | All devices | Comma-separated allowlist of logical device IDs visible to the process (e.g., `"0,1,2,3"`). |
| `TPU_WORKER_ID` | `"0"` | Zero-based process index (`PJRT_Client_ProcessIndex`) of the current worker in a multi-host slice. |
| `TPU_PREMAPPED_BUFFER_SIZE` | `"67108864"` (64 MiB) | Size in bytes of the pre-mapped DMA host bounce buffer reported by `PJRT_HostMemoryAllocator_Extension`. |
| `MEGASCALE_NUM_SLICES` | `"1"` | Total number of TPU slices in a Megascale multi-slice workload. |
| `MEGASCALE_SLICE_ID` | `"0"` | Zero-based slice identifier for the current slice in a Megascale workload. |
| `TPU_WORKER_HOSTNAMES` | Passthrough | Comma-separated worker hostnames or IP addresses in a multi-host slice. |
| `TPU_MESH_CONTROLLER_ADDRESS` | Passthrough | Address of the TPU mesh controller coordinator. |
| `TPU_MESH_CONTROLLER_PORT` | Passthrough | Port of the TPU mesh controller coordinator. |
| `TPU_STDERR_LOG_LEVEL` / `TPU_MIN_LOG_LEVEL` | Passthrough | Logging verbosity level. |
| `LIBTPU_CHECKPOINTING_ENABLED` | Passthrough | Enables Megascale/runtime checkpointing support. |
| `TPU_RUNTIME_METRICS_PORTS` | Passthrough | Ports exposed for runtime telemetry metrics. |

### 3.2 `LIBTPU_INIT_ARGS` Flags

Flags passed in `LIBTPU_INIT_ARGS` (or via `libtpu.set_init_args([...])` / `TfTpu_Initialize`) override environment defaults:

| Flag | Example | Description |
| :--- | :--- | :--- |
| `--tpu_generation=<gen>` / `--xla_tpu_generation=<gen>` | `--tpu_generation=v6e` | Sets the simulated or target TPU hardware generation (`v2`, `v3`, `v4`, `v5e`, `v5p`, `v6e`, `v7x`). |
| `--tpu_chips_per_host_bounds=<x,y,z>` | `--tpu_chips_per_host_bounds=4x2x1` | Sets the 3D chip bounds per host. |
| `--tpu_host_bounds=<x,y,z>` | `--tpu_host_bounds=2,2,1` | Sets the 3D host bounds in the slice. |
| `--xla_tpu_allow_async_allocations=<bool>` / `--tpu_allow_async_allocations=<bool>` | `--xla_tpu_allow_async_allocations=true` | Enables or disables asynchronous HBM allocations (`true`/`1` or `false`/`0`). |
| `--xla_tpu_*` (Any custom XLA TPU flag) | `--xla_tpu_enable_ flash_attention=true` | Stored in `TpuInitConfig::extra_flags` and exposed to PJRT plugin attributes and `TpuCompilationEnvironmentProto`. |

### 3.3 Supported TPU Hardware Generations

| Generation Enum | String Aliases | Cores / Chip | Default HBM / Chip | Default VMEM / Core | MXU Dim |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `TpuGeneration::V2` (`2`) | `v2`, `tpu v2` | 2 | 8 GiB | 16 MiB | 128x128 |
| `TpuGeneration::V3` (`3`) | `v3`, `tpu v3` | 2 | 16 GiB | 16 MiB | 128x128 |
| `TpuGeneration::V4` (`4`) | `v4`, `tpu v4` | 1 (Megacore) | 32 GiB | 16 MiB | 128x128 |
| `TpuGeneration::V5e` (`5`) | `v5e`, `v5lite`, `v5litepod`, `tpu v5 lite` | 1 | 16 GiB | 16 MiB | 128x128 |
| `TpuGeneration::V5p` (`6`) | `v5p`, `tpu v5p` | 2 | 95 GiB | 32 MiB | 128x128 |
| `TpuGeneration::V6e` (`7`) | `v6e`, `trillium`, `tpu v6e` | 1 | 32 GiB | 32 MiB | 256x256 |
| `TpuGeneration::V7x` (`8`) | `v7x`, `ironwood`, `tpu v7x` | 2 | 192 GiB | 64 MiB | 256x256 |

---

## 4. Python SDK Usage (`libtpu` & `libtpu.sdk`)

The `libtpu` Python package provides both the official Google Cloud `libtpu` API (`get_library_path`, `configure_library_path`, `set_init_args`, `get_init_args`, `libtpu.sdk.tpumonitoring`, `libtpu.sdk.slice`, `libtpu.sdk.tpuz`, `libtpu.sdk.debug`) and high-level Python classes (`TpuClient`, `TpuBuffer`, `TpuExecutable`, `TpuProfiler`) for interacting directly with the PJRT C API without needing JAX or PyTorch installed.

### 4.1 Core PJRT Client, Buffers, Compilation & Execution

```python
import libtpu

# 1. Configure initialization flags
libtpu.initialize([
    "--tpu_generation=v6e",
    "--tpu_chips_per_host_bounds=2,2,1",
])

# 2. Create a PJRT TPU Client
client = libtpu.TpuClient()
print(f"Addressable TPU devices: {client.device_count()}")

# 3. Allocate F32 and S32 buffers on Device 0
buf_a = client.buffer_from_floats(device_id=0, values=[1.0, 2.0, 3.0, 4.0], dims=[2, 2])
buf_b = client.buffer_from_floats(device_id=0, values=[10.0, 20.0, 30.0, 40.0], dims=[2, 2])

print("Buffer A shape:", buf_a.shape())
print("Buffer A dtype:", buf_a.dtype())  # DataType.F32

# 4. Cross-device buffer copy (Device 0 -> Device 1)
buf_a_dev1 = client.copy_buffer_to_device(buf_a, dst_device_id=1)
assert buf_a_dev1.to_float_list() == [1.0, 2.0, 3.0, 4.0]
buf_a_dev1.close()

# 5. Compile a StableHLO / MLIR program
mlir_program = """
module @elementwise_add {
  func.func public @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}
"""
executable = client.compile(mlir_program, fmt="mlir")
print("Executable fingerprint:", executable.fingerprint())
print("Estimated FLOPs:", executable.flops())

# 6. Execute on Device 0 and read results back to host
out_buf = executable.execute(device_id=0, inputs=[buf_a, buf_b])
print("Execution result:", out_buf.to_float_list())  # [11.0, 22.0, 33.0, 44.0]

# 7. Ahead-of-Time (AOT) Serialization and Deserialization
serialized_bytes = executable.serialize()
reloaded_exec = client.deserialize_executable(serialized_bytes)
out_buf2 = reloaded_exec.execute(device_id=0, inputs=[buf_a, buf_b])
assert out_buf2.to_float_list() == [11.0, 22.0, 33.0, 44.0]

# 8. Clean up resources
out_buf.close()
out_buf2.close()
reloaded_exec.close()
executable.close()
buf_a.close()
buf_b.close()
client.close()
```

### 4.2 Hardware & Runtime Profiling (`TpuProfiler` / `XSpace`)

Capture wire-compatible `tensorflow.profiler.XSpace` protobuf traces directly from Python:

```python
import libtpu

profiler = libtpu.TpuProfiler()
assert profiler.start() is True

# Run workloads while profiler is active...
client = libtpu.TpuClient()
buf = client.buffer_from_floats(0, [1.0, 2.0, 3.0, 4.0])
buf.close()
client.close()

assert profiler.stop() is True
xspace_protobuf_bytes = profiler.collect_data()
print(f"Collected {len(xspace_protobuf_bytes)} bytes of XSpace profiling data")
profiler.close()
```

### 4.3 Telemetry, Slice Coordinates & Diagnostics (`libtpu.sdk`)

`libtpu.sdk` replicates the exact module hierarchy of the official `libtpu/sdk.so` extension (`tpumonitoring`, `slice`, `tpuz`, and `debug`):

```python
from libtpu.sdk import tpumonitoring, slice as tpu_slice, tpuz, debug

# 1. List and query runtime telemetry metrics
metrics = tpumonitoring.list_supported_metrics()
print("Supported metrics:", metrics)
# ['tpu.runtime.compile.time_micros', 'tpu.runtime.dma.throughput_bytes',
#  'tpu.runtime.hbm.usage_bytes', 'tpu.runtime.ici.link_bandwidth_bytes',
#  'tpu.runtime.tensorcore.utilization']

hbm_metric = tpumonitoring.get_metric("tpu.runtime.hbm.usage_bytes")
print("Description:", hbm_metric.help())
print("Per-core values:", hbm_metric.data())

# 2. Query 3D torus chip coordinates across the host/slice
for coord in tpu_slice.get_chip_coordinates():
    print(f"Host={coord.hostname} Chip={coord.chip_index} XYZ={coord.coordinates}")

# 3. Query TPU core state summary (tpuz)
print("Core state summary:", tpuz.get_core_state_summary(include_hlo_info=True))

# 4. Register and unregister runtime debug callbacks
events = []
cb_id = debug.register_debug_callback(lambda: events.append("triggered"))
debug.unregister_debug_callback(cb_id)
```

---

## 5. Rust Crate Usage (`libtpu-core`)

You can embed `libtpu-core` directly in Rust applications for native TPU compilation, simulation, or direct Linux kernel device control without going through C FFI.

Add `libtpu-core` to your `Cargo.toml`:

```toml
[dependencies]
libtpu-core = { path = "crates/libtpu-core" }
```

### 5.1 High-Level `PjrtCoreClient` & `SimulatedTpuHal`

```rust
use libtpu_core::config::TpuInitConfig;
use libtpu_core::error::LibTpuResult;
use libtpu_core::pjrt::PjrtCoreClient;
use libtpu_core::types::{ChipCoordinates, DeviceId, ElementType, MemorySpaceId, Shape, TpuGeneration};

fn main() -> LibTpuResult<()> {
    // 1. Configure a 4-chip TPU v6e host
    let config = TpuInitConfig {
        generation: TpuGeneration::V6e,
        chips_per_host_bounds: ChipCoordinates::new(2, 2, 1),
        host_bounds: ChipCoordinates::new(1, 1, 1),
        ..TpuInitConfig::default()
    };

    // 2. Initialize a PJRT client backed by SimulatedTpuHal
    let client = PjrtCoreClient::with_simulated_hal(&config)?;
    println!("Discovered {} TPU devices", client.devices.len());

    // 3. Allocate and transfer host F32 tensors to Device 0 HBM
    let shape = Shape::new(ElementType::F32, vec![4], None, None)?;
    let mut bytes_a = Vec::with_capacity(16);
    let mut bytes_b = Vec::with_capacity(16);
    for val in [2.0_f32, 4.0, 6.0, 8.0] {
        bytes_a.extend_from_slice(&val.to_le_bytes());
    }
    for val in [3.0_f32, 5.0, 7.0, 9.0] {
        bytes_b.extend_from_slice(&val.to_le_bytes());
    }

    let buf_a = client.buffer_from_host(DeviceId(0), shape.clone(), &bytes_a)?;
    let buf_b = client.buffer_from_host(DeviceId(0), shape, &bytes_b)?;

    // 4. Compile and load a StableHLO multiplication program
    let mlir = br#"
    module @mul {
      func.func public @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
        %0 = stablehlo.multiply %arg0, %arg1 : tensor<4xf32>
        return %0 : tensor<4xf32>
      }
    }
    "#;
    let executable = client.compile_and_load("mlir", mlir, b"")?;

    // 5. Execute on Device 0 and copy result back to host
    let outputs = executable.execute_on_device(
        DeviceId(0),
        MemorySpaceId(0),
        &[&buf_a, &buf_b],
    )?;
    let out_bytes = outputs[0].to_host_bytes()?;
    let out_floats: Vec<f32> = out_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    assert_eq!(out_floats, vec![6.0, 20.0, 42.0, 72.0]);

    Ok(())
}
```

### 5.2 Linux Kernel Gasket & VFIO Hardware Backend

On physical Google Cloud TPU VMs exposing `/dev/accel*` (Gasket driver, ioctl magic `0xdc`) or `/dev/vfio/*` (VFIO PCIe driver, ioctl magic `0x3b`, PCI Vendor `0x1ae0`), you can use `LinuxKernelTpuHal`:

```rust
use libtpu_core::config::TpuInitConfig;
use libtpu_core::error::LibTpuResult;
use libtpu_core::hal::{LinuxKernelTpuHal, SystemIoctlDispatcher, TpuHal};
use libtpu_core::pjrt::PjrtCoreClient;
use libtpu_core::types::ProcessIndex;
use std::sync::Arc;

fn create_hardware_client() -> LibTpuResult<PjrtCoreClient> {
    let config = TpuInitConfig::default();
    let hal = Arc::new(LinuxKernelTpuHal::new(
        &config,
        Arc::new(SystemIoctlDispatcher),
    ));
    PjrtCoreClient::new(hal, ProcessIndex(0))
}
```

### 5.3 `SparseCore` Embedding Engine, Legacy `TpuDriver` & `TpuCoreProfiler`

`libtpu_core::driver` provides pure-Rust engines for `SparseCore` embedding partitioning, `XSpace` profiling, and `TpuDriver`:

```rust
use libtpu_core::config::TpuInitConfig;
use libtpu_core::driver::{TpuCoreDriver, TpuCoreEmbeddingEngine, TpuCoreProfiler};
use libtpu_core::proto::{EmbeddingTableDescriptorProto, TpuEmbeddingConfigurationProto, XSpaceProto};
use libtpu_core::types::{ByteSize, DeviceId};

fn driver_and_embedding_example() {
    // 1. SparseCore Embedding Engine partitioning & parameter management
    let mut engine = TpuCoreEmbeddingEngine::new();
    let cfg_proto = TpuEmbeddingConfigurationProto {
        table_descriptor: vec![EmbeddingTableDescriptorProto {
            name: String::from("user_embeddings"),
            vocabulary_size: 1024,
            dimension: 64,
            num_features: 1,
        }],
        mode: 1,
        batch_size_per_tensor_core: 128,
        num_hosts: 1,
        num_tensor_cores: 4,
    };
    let partitioned_bytes = engine.execute_partitioner(&cfg_proto.encode());
    assert!(!partitioned_bytes.is_empty());
    engine.configure_memory(vec![0_u8; 64]);
    let _collated = engine.collate_memory();
    engine.connect_hosts();
    engine.finalize();

    // 2. XSpace Profiler
    let profiler = TpuCoreProfiler::new();
    let _ = profiler.start();
    let _ = profiler.stop();
    let xspace_bytes = profiler.collect_xspace_bytes();
    assert!(XSpaceProto::decode(&xspace_bytes).is_ok());

    // 3. TpuDriver
    let driver = TpuCoreDriver::open(TpuInitConfig::default());
    if let Ok(addr) = driver.allocate(DeviceId(0), ByteSize(1024)) {
        let _ = driver.transfer_to_device(DeviceId(0), addr, &[1, 2, 3, 4]);
        let _ = driver.deallocate(DeviceId(0), addr);
    }
}
```

---

## 6. C11 API Usage (`libtpu.h`)

The C11 header `cpp/src/libtpu.h` declares all exported symbols in `libtpu-rs`.

### 6.1 Ergonomic `libtpu_rs_*` C11 Wrappers

For C11 applications that do not want to manually construct 100-byte `PJRT_*_Args` structs, `libtpu.h` provides concise `libtpu_rs_*` functions:

```c
#include "libtpu.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

int main(void) {
  PJRT_Client *client = NULL;
  LibTpuCStatus st = libtpu_rs_client_create(&client);
  assert(st.code == 0);

  int32_t num_devices = libtpu_rs_client_device_count(client);
  printf("Addressable TPU devices: %d
", num_devices);

  // Allocate two 4-element F32 buffers (element_type = 11) on Device 0
  int64_t dims[1] = {4};
  float in0[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float in1[4] = {10.0f, 20.0f, 30.0f, 40.0f};

  PJRT_Buffer *buf0 = NULL;
  PJRT_Buffer *buf1 = NULL;
  st = libtpu_rs_buffer_from_host(client, 0, 11, dims, 1,
                                  (const uint8_t *)in0, sizeof(in0), &buf0);
  assert(st.code == 0);
  st = libtpu_rs_buffer_from_host(client, 0, 11, dims, 1,
                                  (const uint8_t *)in1, sizeof(in1), &buf1);
  assert(st.code == 0);

  // Compile an HLO add program
  const char *hlo = "HloModule add
ENTRY main {
"
                    "  a = f32[4] parameter(0)
"
                    "  b = f32[4] parameter(1)
"
                    "  ROOT res = f32[4] add(a, b)
}";
  PJRT_LoadedExecutable *exec = NULL;
  st = libtpu_rs_compile(client, "hlo", (const uint8_t *)hlo, strlen(hlo), &exec);
  assert(st.code == 0);

  // Execute on Device 0
  const PJRT_Buffer *inputs[2] = {buf0, buf1};
  PJRT_Buffer *out_buf = NULL;
  st = libtpu_rs_execute(exec, 0, inputs, 2, &out_buf);
  assert(st.code == 0);

  // Read output back to host
  float out_host[4] = {0};
  st = libtpu_rs_buffer_to_host(out_buf, (uint8_t *)out_host, sizeof(out_host));
  assert(st.code == 0);
  printf("Result: [%.1f, %.1f, %.1f, %.1f]
",
         out_host[0], out_host[1], out_host[2], out_host[3]);

  // Clean up
  libtpu_rs_buffer_destroy(out_buf);
  libtpu_rs_executable_destroy(exec);
  libtpu_rs_buffer_destroy(buf1);
  libtpu_rs_buffer_destroy(buf0);
  libtpu_rs_client_destroy(client);
  return 0;
}
```

Compile and link against `libtpu`:

```bash
cc -std=c11 -Icpp/src main.c -Ltarget/release -ltpu -Wl,-rpath,@executable_path/target/release -o main
./main
```

### 6.2 Raw OpenXLA `GetPjrtApi()` Vtable & TPU Extensions

Frameworks like OpenXLA, JAX, and PyTorch/XLA call `GetPjrtApi()` (or `GetTpuPjrtApi()`) and walk the `extension_start` linked list of 17 TPU extensions:

```c
#include "libtpu.h"
#include <assert.h>
#include <stdio.h>

void inspect_pjrt_vtable(void) {
  const PJRT_Api *api = GetPjrtApi();
  assert(api != NULL);
  assert(api->struct_size == 1144);
  assert(api->pjrt_api_version.major_version == 0);
  assert(api->pjrt_api_version.minor_version == 116);

  // Walk all 17 chained TPU PJRT extensions
  size_t count = 0;
  for (const PJRT_Extension_Base *ext = api->extension_start;
       ext != NULL;
       ext = ext->next) {
    printf("Extension #%zu: type=%d, size=%zu bytes
",
           ++count, ext->ext_type, ext->struct_size);
  }
  assert(count == 17);
}
```

The 17 chained extensions exported by `GetPjrtApi()->extension_start` are:
1. `PJRT_XlaTransform_Extension` (`ext_type = 29`, `56` bytes)
2. `PJRT_HostMemoryAllocator_Extension` (`ext_type = 28`, `32` bytes)
3. `PJRT_MultiSlice_Extension` (`ext_type = 26`, `64` bytes)
4. `PJRT_Collectives_Extension` (`ext_type = 25`, `96` bytes)
5. `PJRT_AbiVersion_Extension` (`ext_type = 20`, `120` bytes)
6. `PJRT_Shardings_Extension` (`ext_type = 19`, `40` bytes)
7. `PJRT_Megascale_Extension` (`ext_type = 18`, `248` bytes)
8. `PJRT_TpuExecutable_Extension` (`ext_type = 17`, `96` bytes)
9. `PJRT_TpuTopology_Extension` (`ext_type = 16`, `272` bytes)
10. `PJRT_Callback_Extension` (`ext_type = 14`, `40` bytes)
11. `PJRT_PhaseCompile_Extension` (`ext_type = 9`, `64` bytes)
12. `PJRT_CrossHostTransfers_Extension` (`ext_type = 12`, `56` bytes)
13. `PJRT_ExecutableMetadata_Extension` (`ext_type = 11`, `40` bytes)
14. `PJRT_MemoryDescriptions_Extension` (`ext_type = 7`, `40` bytes)
15. `PJRT_Layouts_Extension` (`ext_type = 6`, `80` bytes)
16. `PJRT_RawBuffer_Extension` (`ext_type = 8`, `80` bytes)
17. `PJRT_Profiler_Extension` (`ext_type = 1`, `40` bytes, backed by `PLUGIN_Profiler_Api`)

---

## 7. C++20 RAII SDK Usage (`libtpu.hpp`)

`cpp/src/libtpu.hpp` and `cpp/src/libtpu.cpp` provide a modern, zero-exception (`-fno-exceptions`) C++20 RAII wrapper around `libtpu.h` with move-only resource management:

```cpp
#include "libtpu.hpp"
#include <cassert>
#include <cstdio>
#include <cstring>

int main() {
  // 1. Create a PJRT Client
  libtpu::Client client;
  libtpu::Status st = libtpu::Client::create(&client);
  assert(st.is_ok());
  std::printf("TPU devices: %d
", client.device_count());

  // 2. Allocate device buffers on Device 0
  const int64_t dims[1] = {4};
  const float a[4] = {5.0f, 10.0f, 15.0f, 20.0f};
  const float b[4] = {1.0f, 2.0f, 3.0f, 4.0f};

  libtpu::Buffer buf_a;
  libtpu::Buffer buf_b;
  assert(client.buffer_from_host(0, libtpu::DataType::F32, dims, 1,
                                 reinterpret_cast<const uint8_t *>(a),
                                 sizeof(a), &buf_a).is_ok());
  assert(client.buffer_from_host(0, libtpu::DataType::F32, dims, 1,
                                 reinterpret_cast<const uint8_t *>(b),
                                 sizeof(b), &buf_b).is_ok());

  // 3. Compile a StableHLO subtraction program
  const char *mlir = "module @sub { stablehlo.subtract }";
  libtpu::LoadedExecutable exec;
  assert(client.compile("mlir", reinterpret_cast<const uint8_t *>(mlir),
                        std::strlen(mlir), &exec).is_ok());

  // 4. Execute and read back result
  libtpu::Buffer out_buf;
  assert(exec.execute_binary(0, buf_a, buf_b, &out_buf).is_ok());

  float result[4] = {};
  assert(out_buf.to_host(reinterpret_cast<uint8_t *>(result), sizeof(result)).is_ok());
  std::printf("Result: [%.1f, %.1f, %.1f, %.1f]
",
              result[0], result[1], result[2], result[3]);

  // 5. Query Topology, Profiler, and LibtpuSdk
  libtpu::Topology topo;
  std::printf("Topology mesh: %dx%dx%d (gen v%d)
",
              topo.chip_bounds_x(), topo.chip_bounds_y(), topo.chip_bounds_z(),
              topo.version());

  libtpu::Profiler profiler;
  assert(profiler.start());
  assert(profiler.stop());
  std::printf("Captured XSpace bytes: %zu
", profiler.collect_size());

  libtpu::SdkClient sdk;
  std::printf("SDK chip count: %zu, HBM metric values: %zu
",
              sdk.chip_count(),
              sdk.metric_value_count("tpu.runtime.hbm.usage_bytes"));

  return 0;
}
```

---

## 8. Supported HLO / `StableHLO` Operations & Compiler Pipeline

`libtpu-core` includes a built-in 6-pass HLO / `StableHLO` compiler (`libtpu_core::compiler::CompiledProgram`) that parses textual MLIR (`"mlir"`), textual HLO (`"hlo"`), and binary protobuf `HloModuleProto` (`"hlo"` / `"hlo_with_config"`).

### 8.1 Compilation Phases (`PJRT_PhaseCompile_Extension`)

1. `hlo_legalization`
2. `layout_assignment` (assigns 128x8 tiled TPU MXU layouts)
3. `sharding_propagation` (`xla.OpSharding` extraction)
4. `fusion_and_vmem_allocation` (computes HBM and VMEM scratchpad footprint)
5. `llo_scheduling` (schedules VPU/MXU/DMA instructions)
6. `binary_emission` (emits deterministic `LTPUBIN1` executable payload)

### 8.2 Executed Operations on `SimulatedTpuHal`

When executing compiled programs on `SimulatedTpuHal`, the following HLO / `StableHLO` operations are evaluated deterministically on the HBM buffers:

| Operation (`HloOp`) | Recognized Mnemonics (`StableHLO` / HLO) | Behavior on `F32` / `S32` Tensors |
| :--- | :--- | :--- |
| `HloOp::Identity` | `return`, `copy`, `bitcast`, `reshape` | Copies input 0 to output(s). |
| `HloOp::Add` | `stablehlo.add`, `mhlo.add`, `add` | Elementwise `a[i] + b[i]`. |
| `HloOp::Subtract` | `stablehlo.subtract`, `mhlo.subtract`, `subtract` | Elementwise `a[i] - b[i]`. |
| `HloOp::Multiply` | `stablehlo.multiply`, `mhlo.multiply`, `multiply` | Elementwise `a[i] * b[i]`. |
| `HloOp::Divide` | `stablehlo.divide`, `mhlo.divide`, `divide` | Safe elementwise `a[i] / b[i]`. |
| `HloOp::Maximum` | `stablehlo.maximum`, `mhlo.maximum`, `maximum` | Elementwise `max(a[i], b[i])`. |
| `HloOp::Minimum` | `stablehlo.minimum`, `mhlo.minimum`, `minimum` | Elementwise `min(a[i], b[i])`. |
| `HloOp::Negate` | `stablehlo.negate`, `mhlo.negate`, `negate` | Elementwise `-a[i]`. |
| `HloOp::Abs` | `stablehlo.abs`, `mhlo.abs`, `abs` | Elementwise `\|a[i]\|`. |
| `HloOp::Exp` | `stablehlo.exponential`, `mhlo.exponential`, `exponential` | Elementwise `exp(a[i])`. |
| `HloOp::DotGeneral` | `stablehlo.dot_general`, `stablehlo.dot`, `dot` | Matrix multiplication / dot product. |
| `HloOp::ReduceSum` | `stablehlo.reduce`, `reduce` | Sum reduction across input elements. |
| `HloOp::AllReduceSum` | `stablehlo.all_reduce`, `all-reduce` | Cross-replica sum reduction (`a[i] + b[i]`). |
| `HloOp::AllGather` | `stablehlo.all_gather`, `all-gather` | Concatenates shard slices across inputs. |
| `HloOp::CustomCall` | `stablehlo.custom_call`, `custom-call` | Custom kernel invocation / passthrough. |

Additionally, `PJRT_Triton_Compile` compiles Triton MLIR kernels (`tt.func`) into `LTPUTRITON1` payloads with computed VMEM scratchpad sizes.

---

## 9. Troubleshooting & Verification

### 9.1 Verifying Which `libtpu` Library Is Loaded in a Process

If you are unsure whether JAX, PyTorch/XLA, or TensorFlow loaded the official `libtpu.so` or `libtpu-rs`:

1. **Check `/proc/<pid>/maps` (Linux) or `vmmap <pid>` (macOS)**:
   ```bash
   # Linux:
   grep libtpu /proc/$$/maps
   ```
2. **Probe for `libtpu-rs`-specific C11 symbols in Python**:
   Both official `libtpu.so` and `libtpu-rs` export `GetPjrtApi`, `GetLibtpuSdkApi`, and `TfTpu_Initialize`, whereas only `libtpu-rs` exports `libtpu_rs_client_create`:
   ```python
   import ctypes, libtpu
   cdll = ctypes.CDLL(libtpu.get_library_path())
   is_rust_impl = hasattr(cdll, "libtpu_rs_client_create")
   print("Using libtpu-rs:", is_rust_impl)
   ```

### 9.2 Running the Full Verification Suite

To run all formatting, compilation, linting, type-checking, CTest, Pytest, and 100% doc/test coverage checks locally:

```bash
python3 scripts/pre_commit.py all
```

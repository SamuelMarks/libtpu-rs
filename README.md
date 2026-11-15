# `libtpu-rs`

[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/SamuelMarks/libtpu-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/SamuelMarks/libtpu-rs/actions/workflows/ci.yml)
[![Release](https://github.com/SamuelMarks/libtpu-rs/actions/workflows/release.yml/badge.svg)](https://github.com/SamuelMarks/libtpu-rs/actions/workflows/release.yml)
[![% doc coverage](https://img.shields.io/badge/doc%20coverage-100%25-brightgreen.svg)](#verification--quality-gates)
[![% test coverage](https://img.shields.io/badge/test%20coverage-100%25-brightgreen.svg)](#verification--quality-gates)

**Clean-room, pure-Rust drop-in replacement for Google's closed-source Cloud TPU runtime (`libtpu` v0.0.52).**

`libtpu-rs` implements the complete **OpenXLA PJRT C API (`PJRT_Api` v0.116 — 1,144 bytes, 138 function slots)**, all **17 chained TPU PJRT extensions** plus **`PJRT_FFI_Extension`** (`xla::ffi` custom calls), the **17-slot `LibtpuSdk_Api` (`144` B)**, the **28-slot `TpuDriverFn` (`224` B)**, legacy TensorFlow TPU symbols (`TfTpu_*`, `TpuStatus_*`, `TpuTopology_*`, `TpuProfiler_*`, `TpuEmbeddingEngine_*`), a zero-exception **C++20 RAII & monadic `Result<T>` SDK**, and a drop-in **Python wheel (`libtpu` + `libtpu.sdk`)** powered by **`PyO3` (`abi3-py39`)**, zero-copy **DLPack / PEP 3118 / `ml_dtypes`**, and verified end-to-end against live **`jax`**.

---

## Highlights

- **100% Binary C ABI Parity with Official `libtpu.so` (`v0.0.52`)**:
  - Exports all **79 Google `libtpu` dynamic C symbols** (`GetPjrtApi`, `GetLibtpuSdkApi`, `PyInit_sdk`, `TfTpu_*`, `TpuDriver_*`, `TpuProfiler_*`, `TpuTopology_*`, `TpuEmbeddingEngine_*`, plus ergonomic `libtpu_rs_*` C helpers) with lifetime-bound arenas and 100% `ffi_catch_unwind` panic containment across every `extern "C"` boundary.
  - Exact `#[repr(C)]` struct sizes, field offsets, and vtable layouts verified at compile time in both Rust (`offset_of!`) and C11 (`_Static_assert` in `cpp/tests/test_libtpu_c11.c`), including `PJRT_Api` (`1,144` B), all **17 chained `PJRT_Extension_Base` descriptors** (`PJRT_XlaTransform` -> ... -> `PJRT_TpuTopology` `280` B -> ... -> `PJRT_Profiler` `40` B + `PLUGIN_Profiler_Api` `80` B), `PJRT_FFI_Extension` (`48` B), `LibtpuSdk_Api` (`144` B), and `TpuDriverFn` (`224` B).
- **1:1 `.proto` <-> Rust `struct` Synchronization (738 `protodesc_cold` Schemas + `protos/libtpu_ext/`)**:
  - All **738** embedded `google.protobuf.FileDescriptorProto` schemas reconstructed from `libtpu.so` `protodesc_cold` (encompassing the **293** primary TPU, XLA, HLO, PJRT, Shardy, DeepSea, SparseCore/BarnaCore, MegaScale, XPlane, and CoordinationService schemas plus all transitive dependencies) together with `protos/libtpu_ext/*.proto` compile directly via `protox` + `prost-build` into **7,266** strongly-typed Rust structs and **1,139** enums with zero hand-written `*Proto` structs (`scripts/pre_commit.py proto-sync`).
- **Spec-Compliant MLIR / VHLO / StableHLO / Shardy / Mosaic Compiler & Generation-Exact VLIW / SparseCore Backend**:
  - Upstream-compliant **MLIR Bytecode v1–v6** reader/writer (`b"ML\xefR"`) and `winnow` textual parsers supporting `builtin`, `vhlo` (`v0.9.0`–`v1.11.0+`), `stablehlo`, `chlo`, `sdy` (Shardy), and JAX `tpu` (Mosaic / Pallas) dialects, plus lossless `StableHLO` <-> `xla::HloModuleProto` conversion.
  - **XLA-TPU Middle-End Optimizer**: `ShardyPropagationPass`, `GspmdPartitionerPass`, `DotDecomposerPass`, `ConvolutionLegalizerPass`, `(8, 128)` `TpuLayoutAssignmentPass`, `MxuTilingPass` (`128x128` `v2`–`v5p` and `256x256` `v6e`/`v7x`), `HloRematerializationPass`, `MemorySpaceAssignmentPass` (HBM/VMEM/SMEM/CMEM + async ping-pong DMA), `GlobalDecreasingSizeBestFitHeapSimulator`, and `LearnedCostModelEvaluator`.
  - **Low-Level Optimizer (LLO) & Multi-Generation Binary ISA Codecs**: `LloIr`, `SwingModuloPipeliner`, `LloRegisterAllocator`, generation-exact VLIW binary encoders/disassemblers (`JellyfishIsaCodec` `jfc` v2, `DragonfishIsaCodec` `dfc` v3, `PufferfishIsaCodec` `pfc` v4/v4lite, `ViperfishIsaCodec` `vfc` v5e/v5p, `GhostfishIsaCodec` `gfc` v6e, `IronwoodIsaCodec` `iwc` v7x), and **SparseCore / BarnaCore V2 & V3** codecs with all **19** gradient optimizers.
- **Real Linux Kernel Driver (`VFIO` `0x3B` & `Gasket` `0xDC`), Real CSR Register Maps & Cycle-Accurate Emulator**:
  - Sysfs PCIe `0x1ae0` enumerator (`v2` through `v7x`), authentic `csr_register_map.json` BAR0/GCSR register offsets (extracted from Brotli `*_pf_csrs.br` bitfields and **194,929** static `.lrodata` CSR entries across `V2`–`V7x`), IOMMU/hugepage/`dma-buf` mapping, and non-simulated `LinuxDeviceHal` hardware ring/doorbell/MSI-X `eventfd` execution driven in CI over `VirtualVfioGasketDevice` (executing real binary VLIW & SparseCore bitstreams).
- **Distributed Coordination, 2D/3D Twisted-Torus ICI & MegaScale DCN Collectives**:
  - Wire-compatible HTTP/2 + Protobuf gRPC `tensorflow.CoordinationService` (all 19 RPCs), deadlock-free 2D torus and 3D twisted-torus ICI routing (`AllReduce`, `ReduceScatter`, `AllGather`, `AllToAll`, `CollectivePermute`, `RaggedAllToAll` across all **29 `ElementType`s** including `BF16`, `FP8`, `FP4` `E2M1FN`, `E8M0FNU`), and multi-socket TCP/DCN MegaScale runtime.
- **First-Class Rust, C11, C++20, and Python (`PyO3` + Live `jax`) SDKs**:
  - **Rust (`libtpu-core`)**: Safe, strongly-typed APIs using `#[repr(transparent)]` hardware/memory/ISA newtypes and a single unified `LibTpuError` enum (`derive_more::{Display, Error, From}`) — **zero `.unwrap()`, zero `.expect()`, zero `anyhow`/`thiserror`**.
  - **C11 & C++20 (`cpp/src/`)**: Strict `-std=c11` header (`libtpu.h`) and `-fno-exceptions -fno-rtti` C++20 RAII wrappers (`libtpu.hpp` / `libtpu.cpp`) with monadic `libtpu::Result<T>` (`value_or`, `map`, `and_then`).
  - **Python (`python/libtpu/` & `crates/libtpu-py`)**: Drop-in `libtpu` & `libtpu.sdk` (`tpumonitoring`, `slice`, `tpuz`, `debug`, `diagnostics`) with `PyO3` (`abi3-py39`) native modules (`libtpu._C`, `libtpu/sdk.so`), PEP 3118 buffer protocol, NumPy `__array_interface__` (`ml_dtypes`), DLPack (`__dlpack__`, `from_dlpack`), and live `jax` numerical parity (`python/tests/test_jax_integration.py`).

---

## Architecture Overview

```text
+-----------------------------------------------------------------------------------+
|                   User Applications & ML Frameworks                               |
|   JAX / OpenXLA / PyTorch-XLA  |  Python libtpu.sdk  |  C11 & C++20 RAII SDK      |
+--------------------------------+---------------------+----------------------------+
                                 |                     |
                                 v                     v
+-----------------------------------------------------------------------------------+
|  crates/libtpu-c-api (cdylib / staticlib: libtpu.so | libtpu.dylib | tpu.dll)     |
|  * GetPjrtApi() -> PJRT_Api v0.116 (1,144 B, 138 slots) + 17 TPU Extensions       |
|  * PJRT_FFI_Extension (48 B) | GetLibtpuSdkApi() (144 B) | TpuDriverFn (224 B)    |
|  * Legacy TfTpu_*, TpuStatus_*, TpuTopology_*, TpuProfiler_*, TpuEmbeddingEngine_*|
|  * crates/libtpu-py: PyO3 (abi3-py39) libtpu._C & PyInit_sdk + DLPack / PEP 3118  |
+-----------------------------------------------------------------------------------+
                                 |
                                 v
+-----------------------------------------------------------------------------------+
|  crates/libtpu-core (100% Safe Core, Unified LibTpuError via derive_more)         |
|  * proto/       : 738 protodesc_cold + 6 ext .proto -> 7,266 prost structs (1:1)  |
|  * compiler/    : MLIR v1-v6 / VHLO / StableHLO / Shardy / Mosaic -> Middle-End   |
|                   (GSPMD, (8,128) Tiling, MSA) -> LLO -> jfc/dfc/pfc/vfc/gfc/iwc  |
|                   VLIW + SparseCore V2/V3 Binary Codecs & Riegeli Relocations     |
|  * distributed/ : HTTP/2 gRPC CoordinationService + 2D/3D ICI + MegaScale TCP/DCN |
|  * hal/         : Sysfs 0x1ae0 + VFIO (0x3B) / Gasket (0xDC) + Real CSR Maps      |
|                   LinuxDeviceHal over Physical TPU or VirtualVfioGasketDevice     |
|  * elf/         : ELF64/Mach-O, CFI/RTTI, Brotli CSR & protodesc_cold Forensics   |
+-----------------------------------------------------------------------------------+
```

See **[`ARCHITECTURE.md`](ARCHITECTURE.md)** for the deep-dive architectural reference and **[`USAGE.md`](USAGE.md)** for Rust, C11, C++20, Python, and JAX usage examples.

---

## Repository Layout

| Path | Description |
| :--- | :--- |
| `crates/libtpu-core/` | Pure-Rust core: 7,266 generated `prost` structs (`738` `protodesc_cold` + `6` `libtpu_ext` `.proto` files), MLIR/StableHLO/Shardy/Mosaic frontend, XLA-TPU middle-end & MSA, LLO + multi-gen VLIW/SparseCore codecs, `LinuxDeviceHal` + `VirtualVfioGasketDevice`, gRPC `CoordinationService` + ICI + MegaScale, and ELF forensics. |
| `crates/libtpu-c-api/` | ABI-exact `#[repr(C)]` C FFI (`GetPjrtApi`, 17 TPU extensions, `PJRT_FFI_Extension`, `GetLibtpuSdkApi`, `TpuDriver_*`, `TfTpu_*`, `libtpu_rs_*`) with lifetime-bound arenas and 100% `ffi_catch_unwind`. |
| `crates/libtpu-py/` | `PyO3` (`abi3-py39`) native CPython extensions (`libtpu._C` and `libtpu/sdk.so` `PyInit_sdk`) + zero-copy `DLManagedTensor` capsule bridge. |
| `cpp/` | Two-level CMake project (`cpp/CMakeLists.txt` + `cpp/src/CMakeLists.txt`), C11 header (`libtpu.h`), C11 `_Static_assert` verification (`test_libtpu_c11.c`), and zero-exception C++20 RAII + `Result<T>` SDK (`libtpu.hpp`, `libtpu.cpp`, `test_libtpu.cpp`). |
| `python/libtpu/` | Drop-in `libtpu` and `libtpu.sdk` (`tpumonitoring`, `slice`, `tpuz`, `debug`, `diagnostics`) with PEP 561 `.pyi` stubs, PEP 3118 buffer protocol, NumPy/`ml_dtypes` interop, and DLPack. |
| `python/tests/` | 100% branch-covered unit tests plus live end-to-end `jax` numerical parity suite (`test_jax_integration.py`). |
| `protos/libtpu_ext/` | Canonical `proto3` extension schemas for pure-Rust HLO passes, Shardy/Mosaic/LLO metadata, MegaScale, and XPlane profiler RPCs (`prost-build` compiled 1:1). |
| `docs/reverse_engineering/` | Ground-truth `protodesc_cold` `.proto` tree (738 files), real `csr_register_map.json` (`V2`–`V7x`), `golden_binary_manifest.json`, `golden_struct_layouts.json`, and ABI specifications. |
| `scripts/` | `pre_commit.py` quality gate, `maturin_backend.py` wheel builder, `fetch_artifacts.py`, `sync_oss_anchors.py`, `re_extract.py`, `ghidra_lift_isa_and_abi.py`, and `differential_oracle.py`. |

---

## Quick Start

### 1. Rust Workspace

```bash
cargo build --workspace --all-targets
cargo test --workspace --all-targets
```

### 2. Python Drop-In Package & Live JAX Integration

```bash
pip install -e .
pytest python/tests/test_jax_integration.py -v
```

```python
import libtpu

# Point JAX / XLA at the compiled pure-Rust libtpu shared library
libtpu.configure_library_path()
libtpu.set_init_args("--tpu_generation=v5p", "--tpu_chips_per_host_bounds=2,2,1")

import jax
import jax.numpy as jnp

x = jax.device_put(jnp.array([[1.0, 2.0], [3.0, 4.0]]), jax.devices("tpu")[0])
print(jnp.matmul(x, x))
```

### 3. C11 & C++20 Zero-Exception SDK

```bash
cmake -B cpp/build -S cpp -DENABLE_COVERAGE=ON
cmake --build cpp/build
ctest --test-dir cpp/build --output-on-failure
```

---

## Verification & Quality Gates

Every commit is gated by `scripts/pre_commit.py` and `.github/workflows/ci.yml` enforcing:

- **100.0% Documentation Coverage**: Across every file, module, struct/class/enum, field, function, and parameter in Rust (`cargo doc` `-D missing_docs` + `rustdoc --coverage` + AST `# Arguments` auditor), Python (`ruff` Google pydocstyle `D` + AST parameter auditor), and C11/C++20 (`doxygen` `WARN_AS_ERROR = YES`, `WARN_NO_PARAMDOC = YES` + XML auditor).
- **100.0% Function, Line, and Branch/Region Test Coverage**: Verified via `cargo +nightly llvm-cov --workspace --branch` (Rust), `pytest --cov=libtpu --cov=scripts --cov-branch --cov-fail-under=100` (Python), and `llvm-profdata` + `llvm-cov` + ASan/UBSan (C11/C++20), with zero `coverage(off)` on production code and zero `# pragma: no cover`.
- **Strict Typing, Zero Panics & Single `derive_more` Error Enum**:
  - Workspace `Cargo.toml` forbids `unsafe_op_in_unsafe_fn` violations and denies `unwrap_used`, `expect_used`, `panic`, `todo`, `unimplemented`, `unreachable`, and `indexing_slicing` across `pedantic` + `nursery` Clippy lints.
  - Zero `anyhow` or `thiserror`: all fallible Rust operations return `Result<T, LibTpuError>` powered by `derive_more`.
  - Python passes `mypy --strict` across all modules, scripts, and `.pyi` stubs; C11/C++20 compile cleanly with `-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -fno-exceptions -fno-rtti -Werror`.
- **1:1 `.proto` <-> `.rs` Schema Sync (`proto-sync`) & Binary ABI Gate (`abi`)**: Verifies bidirectional parity across all 744 `.proto` files (`738` `protodesc_cold` + `6` `protos/libtpu_ext/`) and dynamically inspects all 79 Google `libtpu` exported symbols (plus `libtpu_rs_*` C helpers), 138 `PJRT_Api` slots, and 17 PJRT extension descriptors.

Run the complete local verification suite with:

```bash
python3 scripts/pre_commit.py all --check
```

---

## License

Licensed under either of:

- Apache License, Version 2.0 ([`LICENSE-APACHE`](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([`LICENSE-MIT`](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

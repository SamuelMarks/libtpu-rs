# `libtpu-rs`

[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/SamuelMarks/libtpu-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/SamuelMarks/libtpu-rs/actions/workflows/ci.yml)
[![Release Binaries & Python Wheels](https://github.com/SamuelMarks/libtpu-rs/actions/workflows/release.yml/badge.svg)](https://github.com/SamuelMarks/libtpu-rs/actions/workflows/release.yml)
[![Rust Edition](https://img.shields.io/badge/rust-2024-orange.svg)](https://www.rust-lang.org/)
[![% doc coverage](https://img.shields.io/badge/doc%20coverage-100%25-brightgreen.svg)](#verification--quality-gates)
[![% test coverage](https://img.shields.io/badge/test%20coverage-100%25-brightgreen.svg)](#verification--quality-gates)

Clean-room Rust drop-in replacement for the Google Cloud TPU runtime library (`libtpu` v0.0.50), providing 100% binary ABI compatibility across OpenXLA PJRT (`PJRT_Api` vtable + 17 chained PJRT extensions), `LibtpuSdk_Api` (`libtpu.sdk`), `TpuDriver_*`, `TpuEmbeddingEngine_*`, `TpuProfiler_*`, `TpuTopology_*`, and `TfTpu_*` entrypoints with first-class C11, C++20, and Python bindings.

- **[USAGE.md](USAGE.md)**: Full usage documentation, API examples across Python/Rust/C11/C++20, and instructions on hooking JAX, PyTorch/XLA (`torch-tpu`), and TensorFlow installations to use `libtpu-rs`.
- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Exhaustive 7-layer system architecture, memory safety & concurrency model, VLIW ISA, and binary ABI specification.

---

## Architecture & Workspace Layout

- **`crates/libtpu-core`**: Pure-Rust TPU runtime core, unified `LibTpuError` (`derive_more`), strongly-typed domain primitives (`types.rs`), wire-compatible Protobuf codecs & `prost_reflect` descriptor pool (`proto.rs`, `build.rs`), `LIBTPU_INIT_ARGS` & environment parser (`config.rs`), 6-pass HLO/`StableHLO` compiler & 5-phase `PJRT_PhaseCompile` engine (`compiler.rs`), ELF64/Mach-O binary forensics & `iced-x86` disassembler (`elf.rs`), Linux Gasket/VFIO (`0xdc`/`0x3b` ioctl) + simulated Hardware Abstraction Layer (`hal.rs`), PJRT engine (`pjrt.rs`), and `TpuDriver`/`LibtpuSdk`/`TpuEmbeddingEngine` drivers (`driver.rs`).
- **`crates/libtpu-c-api`**: Drop-in `cdylib`/`staticlib`/`rlib` (`libtpu.so` / `libtpu.dylib` / `tpu.dll`) exporting all 79 official `.dynsym` C ABI functions (`78` from `libtpu.so` + `PyInit_sdk` from `sdk.so`), 18 ergonomic `libtpu_rs_*` C/C++ entrypoints (`rust_ergonomic_api.rs`), the 138-slot `PJRT_Api` v0.116 vtable (`1,144` bytes across `pjrt_*_api.rs` & `pjrt_types.rs`), all 17 chained `PJRT_Extension_Base` structures (`130` extension functions + `8` `PLUGIN_Profiler_Api` functions = `138` extension/profiler entrypoints in `extensions.rs`), the 28-slot `TpuDriverFn` vtable (`224` bytes in `tpu_driver_types.rs` & `legacy_tf_api.rs`), and the 17-slot `LibtpuSdk_Api` v0.1 vtable (`144` bytes in `sdk_api.rs`).
- **`crates/libtpu-py`**: Rust FFI bridge crate (`PyTpuRuntime`, `PySdkBridge`) for Python bindings (`crates/libtpu-py/src/lib.rs`).
- **`cpp/`**: Strict C11 ABI header (`cpp/src/libtpu.h`), modern zero-exception C++20 RAII SDK (`cpp/src/libtpu.hpp`, `cpp/src/libtpu.cpp`), Doxygen configuration (`cpp/Doxyfile`), and CMake test suite (`cpp/tests/test_libtpu.cpp`).
- **`python/`**: Drop-in `libtpu` and `libtpu.sdk` (`slice`, `tpumonitoring`, `tpuz`, `debug`) Python package with PEP 561 `.pyi` type stubs and `pytest` suite (`python/tests/test_libtpu.py`, `python/tests/test_pre_commit.py`).
- **`docs/reverse_engineering/`**: Forensic binary reverse-engineering specifications for `libtpu` v0.0.50 (`exported_symbols.md`, `pjrt_abi_layout.md`, `kernel_hardware_protocol.md`, `python_api.md`), golden verification manifests (`golden_binary_manifest.json`, `golden_struct_layouts.json`, `golden_proto_descriptors.json`, `csr_register_map.json`), and 64 extracted clean-room Protobuf schemas (`docs/reverse_engineering/protos/*.proto`).
- **`scripts/`**: Cross-platform `pre-commit` verification and coverage-shield pipeline (`scripts/pre_commit.py`) and deterministic ELF64/wheel binary forensics & golden manifest verifier (`scripts/re_extract.py`).

---

## Verification & Quality Gates

All quality gates are enforced automatically via `pre-commit` (`scripts/pre_commit.py`):

```bash
# Run the full pre-commit verification pipeline (fmt, build, clippy/lint, abi, test, shields)
python3 scripts/pre_commit.py all

# Run in strict read-only CI check mode
python3 scripts/pre_commit.py all --check

# Verify golden binary manifests, struct layouts, proto descriptors, and CSR maps
python3 scripts/re_extract.py --verify-golden

# Install git pre-commit hooks
pre-commit install
pre-commit run --all-files
```

- **Code Formatting (`fmt` — Auto-Fix + Recheck)**:
  - Rust: `cargo fmt --all` followed by `cargo fmt --all -- --check`
  - Python: `ruff format python/ scripts/` followed by `ruff format --check python/ scripts/`
  - C/C++: `clang-format -i -style=LLVM` followed by `clang-format --dry-run --Werror -style=LLVM`
- **Build & Compilation (`build`)**:
  - Rust: `cargo build --workspace --all-targets`
  - C/C++: `cmake -B cpp/build -S cpp -DENABLE_COVERAGE=ON && cmake --build cpp/build`
- **Strict Linting & Type Checking (`clippy` / `lint`)**:
  - Rust: `cargo clippy --workspace --all-targets --all-features -- -D warnings`
  - Rustdoc: `RUSTDOCFLAGS="-D warnings -D missing_docs -D rustdoc::broken_intra_doc_links" cargo doc --workspace --no-deps --document-private-items`
  - Python: `ruff check python/ scripts/` and `mypy --strict python/ scripts/`
  - C/C++: `doxygen cpp/Doxyfile` (`WARN_AS_ERROR = YES`)
- **Binary ABI & Source Invariant Verification (`abi`)**:
  - Verifies all 79 official `libtpu.so` / `sdk.so` dynamic symbols, `#[repr(C)]` struct sizes, non-null `PJRT_Api` (1,144 B, 138 slots), 17-node `PJRT_Extension_Base` linked list, `LibtpuSdk_Api` (144 B, 17 slots), `TpuDriverFn` (224 B, 28 slots), and enforces zero `.unwrap()` / `.expect()` / `anyhow` / `thiserror`.
- **100% Documentation & Test Coverage (`test` & `shields`)**:
  - Rust: `cargo +nightly llvm-cov --workspace --branch --fail-under-functions 100 --fail-under-lines 100 --fail-under-regions 100`
  - C/C++: `ctest --test-dir cpp/build --output-on-failure` and `llvm-cov export` (100% function, line, and branch coverage)
  - Python: `pytest --cov=libtpu --cov=scripts --cov-branch --cov-report=term-missing --cov-fail-under=100`

---

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/Apache-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

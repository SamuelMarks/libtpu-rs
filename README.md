# `libtpu-rs`

[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI & Pre-commit Quality Gates](https://github.com/SamuelMarks/libtpu-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/SamuelMarks/libtpu-rs/actions/workflows/ci.yml)
[![Release Binaries & Python Wheels](https://github.com/SamuelMarks/libtpu-rs/actions/workflows/release.yml/badge.svg)](https://github.com/SamuelMarks/libtpu-rs/actions/workflows/release.yml)
[![Rust Edition](https://img.shields.io/badge/rust-2024-orange.svg)](https://www.rust-lang.org/)
[![% doc coverage](https://img.shields.io/badge/doc%20coverage-100%25-brightgreen.svg)](#verification--quality-gates)
[![% test coverage](https://img.shields.io/badge/test%20coverage-100%25-brightgreen.svg)](#verification--quality-gates)

Clean-room Rust drop-in replacement for the Google Cloud TPU runtime library (`libtpu` v0.0.50), providing 100% binary ABI compatibility across OpenXLA PJRT (`PJRT_Api` vtable + 17 chained PJRT extensions), `LibtpuSdk_Api` (`libtpu.sdk`), `TpuDriver_*`, `TpuEmbeddingEngine_*`, `TpuProfiler_*`, `TpuTopology_*`, and `TfTpu_*` entrypoints with first-class C11, C++20, and Python bindings.

For full usage documentation, API examples across Python/Rust/C11/C++20, and instructions on hooking JAX, PyTorch/XLA (`torch-tpu`), and TensorFlow installations to use `libtpu-rs`, see **[USAGE.md](USAGE.md)**.

---

## Architecture & Workspace Layout

- **`crates/libtpu-core`**: Pure-Rust TPU runtime core, unified `LibTpuError` (`derive_more`), strongly-typed domain primitives (`types.rs`), wire-compatible Protobuf codecs (`proto.rs`), `LIBTPU_INIT_ARGS` & environment parser (`config.rs`), 6-pass HLO/`StableHLO` compiler (`compiler.rs`), Linux Gasket/VFIO (`0xdc`/`0x3b` ioctl) + simulated Hardware Abstraction Layer (`hal.rs`), PJRT engine (`pjrt.rs`), and `TpuDriver`/`LibtpuSdk`/`TpuEmbeddingEngine` drivers (`driver.rs`).
- **`crates/libtpu-c-api`**: Drop-in `cdylib` (`libtpu.so` / `libtpu.dylib`) exporting all 75 `.dynsym` C ABI functions, the 138-slot `PJRT_Api` vtable (`1,144` bytes), all 17 chained `PJRT_Extension_Base` structures (`113` extension functions), the 28-slot `TpuDriverFn` vtable (`224` bytes), and the 17-slot `LibtpuSdk_Api` vtable (`144` bytes).
- **`crates/libtpu-py`**: Rust FFI bridge crate for Python bindings.
- **`cpp/`**: Strict C11 ABI header (`cpp/src/libtpu.h`), modern C++20 RAII SDK (`cpp/src/libtpu.hpp`, `cpp/src/libtpu.cpp`), Doxygen configuration (`cpp/Doxyfile`), and CMake test suite (`cpp/tests/test_libtpu.cpp`).
- **`python/`**: Drop-in `libtpu` and `libtpu.sdk` (`slice`, `tpumonitoring`, `tpuz`, `debug`) Python package with PEP 561 `.pyi` type stubs and `pytest` suite (`python/tests/test_libtpu.py`, `python/tests/test_pre_commit.py`).
- **`docs/reverse_engineering/`**: Forensic binary reverse-engineering specifications for `libtpu` v0.0.50 (`exported_symbols.md`, `pjrt_abi_layout.md`, `kernel_hardware_protocol.md`, `python_api.md`).
- **`scripts/`**: Cross-platform `pre-commit` verification and coverage-shield pipeline (`scripts/pre_commit.py`).

---

## Verification & Quality Gates

All quality gates are enforced automatically via `pre-commit` (`scripts/pre_commit.py`):

```bash
# Run the full pre-commit verification pipeline (format, build, lint, test, coverage & shields)
python3 scripts/pre_commit.py all

# Install git pre-commit hooks
pre-commit install
pre-commit run --all-files
```

- **Code Formatting (Auto-Fix + Recheck)**:
  - Rust: `cargo fmt --all` followed by `cargo fmt --all -- --check`
  - Python: `ruff format python/ scripts/` followed by `ruff format --check python/ scripts/`
  - C/C++: `clang-format -i -style=LLVM` followed by `clang-format --dry-run --Werror -style=LLVM`
- **Build & Compilation**:
  - Rust: `cargo build --workspace --all-targets`
  - C/C++: `cmake -B cpp/build -S cpp -DENABLE_COVERAGE=ON && cmake --build cpp/build`
- **Strict Linting & Type Checking**:
  - Rust: `cargo clippy --workspace --all-targets --all-features -- -D warnings`
  - Rustdoc: `RUSTDOCFLAGS="-D warnings -D missing_docs -D rustdoc::broken_intra_doc_links" cargo doc --workspace --no-deps --document-private-items`
  - Python: `ruff check python/ scripts/` and `mypy --strict python/ scripts/`
  - C/C++: `doxygen cpp/Doxyfile` (`WARN_AS_ERROR = YES`)
- **100% Documentation & Test Coverage**:
  - Rust: `cargo +nightly llvm-cov --workspace --branch --fail-under-functions 100 --fail-under-lines 100 --fail-under-regions 100`
  - C/C++: `ctest --test-dir cpp/build --output-on-failure` and `llvm-cov report` (100% function, line, region, and branch coverage)
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

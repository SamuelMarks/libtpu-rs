# `libtpu-rs` Architecture Guide

This document provides an exhaustive architectural reference for **`libtpu-rs`**, a pure-Rust, zero-external-C++-dependency, ABI-compatible re-implementation of Google Cloud's `libtpu.so` (`v0.0.50`, `PJRT_Api` v0.116) and `libtpu/sdk/sdk.so` (`LibtpuSdk_Api` v0.1), paired with C11 headers, a zero-exception C++20 RAII SDK, and a drop-in Python wheel package (`libtpu` + `libtpu.sdk`).

---

## Table of Contents

1. [System Overview & Design Philosophy](#1-system-overview--design-philosophy)
2. [Workspace & Directory Layout](#2-workspace--directory-layout)
3. [Multi-Layered System Architecture](#3-multi-layered-system-architecture)
4. [Layer 1: Core Domain & Unified Error Model (`libtpu-core`)](#4-layer-1-core-domain--unified-error-model-libtpu-core)
   - [4.1 Strongly-Typed Domain Primitives (`types.rs`)](#41-strongly-typed-domain-primitives-typesrs)
   - [4.2 Unified Error Hierarchy (`error.rs`)](#42-unified-error-hierarchy-errorrs)
   - [4.3 Runtime Configuration & Environment Parser (`config.rs`)](#43-runtime-configuration--environment-parser-configrs)
   - [4.4 Zero-Dependency Protobuf Wire Codec (`proto.rs`)](#44-zero-dependency-protobuf-wire-codec-protors)
5. [Layer 2: HLO & StableHLO Compiler Pipeline (`compiler.rs`)](#5-layer-2-hlo--stablehlo-compiler-pipeline-compilerrs)
   - [5.1 Input Formats & Lexical/Syntactic Frontend](#51-input-formats--lexicalsyntactic-frontend)
   - [5.2 VLIW Instruction Bundle ISA (`TpuOpCode` & `TpuInstructionBundle`)](#52-vliw-instruction-bundle-isa-tpuopcode--tpuinstructionbundle)
   - [5.3 Six-Pass Lowering & Optimization Pipeline](#53-six-pass-lowering--optimization-pipeline)
   - [5.4 Phased Compilation (`PJRT_PhaseCompile_Extension`)](#54-phased-compilation-pjrt_phasecompile_extension)
   - [5.5 Cost Analysis, Memory Stats, Fingerprinting & Serialization](#55-cost-analysis-memory-stats-fingerprinting--serialization)
6. [Layer 3: Hardware Abstraction Layer (`hal.rs`)](#6-layer-3-hardware-abstraction-layer-halrs)
   - [6.1 The `TpuHal` Trait Contract](#61-the-tpuhal-trait-contract)
   - [6.2 Deterministic In-Memory Simulator (`SimulatedTpuHal`)](#62-deterministic-in-memory-simulator-simulatedtpuhal)
   - [6.3 Linux Kernel Driver Backend (`LinuxDeviceHal`, Gasket & VFIO)](#63-linux-kernel-driver-backend-linuxdevicehal-gasket--vfio)
   - [6.4 PCIe BAR0 MMIO Control & Status Register (CSR) Map](#64-pcie-bar0-mmio-control--status-register-csr-map)
   - [6.5 Injectable OS System Call Interface (`SyscallProvider`)](#65-injectable-os-system-call-interface-syscallprovider)
7. [Layer 4: PJRT & Legacy Runtime Engines (`pjrt.rs` & `driver.rs`)](#7-layer-4-pjrt--legacy-runtime-engines-pjrtrs--driverrs)
   - [7.1 PJRT Core Domain Engine (`pjrt.rs`)](#71-pjrt-core-domain-engine-pjrtrs)
   - [7.2 Legacy TensorFlow, Driver, Profiler, Embedding & SDK Engines (`driver.rs`)](#72-legacy-tensorflow-driver-profiler-embedding--sdk-engines-driverrs)
8. [Layer 5: C ABI & Binary Layout (`libtpu-c-api`)](#8-layer-5-c-abi--binary-layout-libtpu-c-api)
   - [8.1 `PJRT_Api` v0.116 Vtable (`1,144` Bytes, 138 Slots)](#81-pjrt_api-v0116-vtable-1144-bytes-138-slots)
   - [8.2 The 17-Node TPU PJRT Extension Linked List (`extensions.rs`)](#82-the-17-node-tpu-pjrt-extension-linked-list-extensionsrs)
   - [8.3 `LibtpuSdk_Api` v0.1 Vtable (`144` Bytes, 17 Slots)](#83-libtpusdk_api-v01-vtable-144-bytes-17-slots)
   - [8.4 Legacy `TfTpu_*`, `TpuDriver_*`, `TpuProfiler_*` & `TpuEmbeddingEngine_*` C Symbols](#84-legacy-tftpu_-tpudriver_-tpuprofiler_--tpuembeddingengine_-c-symbols)
   - [8.5 Ergonomic `libtpu_rs_*` C Entrypoints](#85-ergonomic-libtpu_rs_-c-entrypoints)
9. [Layer 6: C11 Header & C++20 RAII SDK (`cpp/`)](#9-layer-6-c11-header--c20-raii-sdk-cpp)
   - [9.1 C11 Header (`cpp/src/libtpu.h`)](#91-c11-header-cppsrclibtpu-h)
   - [9.2 Zero-Exception C++20 RAII Wrappers (`cpp/src/libtpu.hpp` & `libtpu.cpp`)](#92-zero-exception-c20-raii-wrappers-cppsrclibtpu-hpp--libtpucpp)
10. [Layer 7: Python Wheel Package & SDK (`libtpu-py` & `python/libtpu/`)](#10-layer-7-python-wheel-package--sdk-libtpu-py--pythonlibtpu)
    - [10.1 Rust Python Bridge (`crates/libtpu-py`)](#101-rust-python-bridge-crateslibtpu-py)
    - [10.2 Top-Level `libtpu` Python Module (`python/libtpu/__init__.py`)](#102-top-level-libtpu-python-module-pythonlibtpu__init__py)
    - [10.3 `libtpu.sdk` Subpackage (`tpumonitoring`, `slice`, `tpuz`, `debug`)](#103-libtpusdk-subpackage-tpumonitoring-slice-tpuz-debug)
11. [End-to-End Data & Control Flows](#11-end-to-end-data--control-flows)
    - [11.1 Client Initialization & Topology Discovery](#111-client-initialization--topology-discovery)
    - [11.2 Program Compilation (`StableHLO` / `HloModuleProto` -> `LoadedExecutable`)](#112-program-compilation-stablehlo--hlomoduleproto---loadedexecutable)
    - [11.3 Host-to-Device DMA, VLIW Execution & Device-to-Host Readback](#113-host-to-device-dma-vliw-execution--device-to-host-readback)
12. [Concurrency, Memory Safety & Lifetime Model](#12-concurrency-memory-safety--lifetime-model)
13. [Quality Standards & Verification Pipeline (`scripts/pre_commit.py`)](#13-quality-standards--verification-pipeline-scriptspre_commitpy)

---

## 1. System Overview & Design Philosophy

Google Cloud's proprietary `libtpu.so` is a monolithic ~631 MB shared library linking XLA, LLVM, MLIR, Abseil, gRPC, Protobuf, and low-level kernel driver clients (`asic_sw::driver`). Frameworks such as **JAX**, **PyTorch/XLA**, and **TensorFlow** interact with `libtpu.so` strictly across well-defined C ABI boundaries (`GetPjrtApi`, `GetLibtpuSdkApi`, `TfTpu_*`, `TpuDriver_*`, `TpuProfiler_*`, and `TpuEmbeddingEngine_*`).

`libtpu-rs` re-implements the complete surface area of `libtpu.so` and `libtpu/sdk/sdk.so` in safe, idiomatic Rust with the following architectural invariants:

1. **100% Binary ABI Compatibility**: Every exported C symbol, vtable offset, struct size, alignment padding byte, and extension type discriminant matches the reverse-engineered `libtpu-0.0.50` binary (`PJRT_Api` v0.116 = `1,144` bytes; `LibtpuSdk_Api` v0.1 = `144` bytes; `TpuDriverFn` = `224` bytes), verified via compile-time `const _: () = assert!(...)` assertions.
2. **Zero External Runtime Dependencies**: Protobuf wire serialization, HLO/StableHLO lexing and parsing, VLIW bundle lowering, and POSIX `ioctl` wrappers are implemented directly in `libtpu-core`. The only Rust crate dependency across the entire workspace is `derive_more` (for ergonomic trait derivations on domain types and the unified error enum).
3. **Deterministic Hardware Simulation Everywhere**: A cycle- and shape-accurate `SimulatedTpuHal` and `EmulatedKernelDriverBackend` allow multi-host TPU pods (`v2`, `v3`, `v4`, `v5e`, `v5p`, `v6e`, `v7x`), ICI collectives, HBM capacity limits, BAR0 MMIO CSRs, and Gasket/VFIO `ioctl` page tables to be tested deterministically on macOS, Linux, and Windows without physical TPU PCIe cards.
4. **Strict Safety & Quality Enforcement**:
   - Zero `.unwrap()` or `.expect()` calls across all library and test code (`clippy::unwrap_used = "deny"`, `clippy::expect_used = "deny"`).
   - Single workspace-wide error enum (`LibTpuError`) powered by `derive_more` (`From`, `Display`, `Error`) — no `anyhow` or stringly-typed error erasure.
   - `missing_docs = "forbid"`, `missing_safety_doc = "forbid"`, and `undocumented_unsafe_blocks = "forbid"`.
   - 100% documentation coverage and 100% line/function/branch test coverage across Rust, Python, and C++20.

---

## 2. Workspace & Directory Layout

```text
libtpu-rs/
├── Cargo.toml                         # Rust workspace manifest & strict rustc/clippy lint table
├── pyproject.toml                     # Python package, ruff, mypy (--strict), and pytest config
├── README.md                          # User-facing overview, quickstart, and coverage shields
├── USAGE.md                           # Comprehensive usage & framework integration guide
├── ARCHITECTURE.md                    # This architectural reference document
├── crates/
│   ├── libtpu-core/                   # Pure-Rust domain engine, compiler, HAL, PJRT & driver core
│   │   ├── Cargo.toml
│   │   ├── build.rs                   # Compiles .proto schemas into FileDescriptorSet via protox/prost
│   │   └── src/
│   │       ├── lib.rs                 # Crate root & workspace source invariants test
│   │       ├── elf.rs                 # ELF64/Mach-O forensics module root & re-exports
│   │       ├── elf/
│   │       │   ├── carver.rs          # protodesc_cold Protobuf carver, .proto renderer & filewrapper_toc
│   │       │   ├── disassembler.rs    # iced-x86 prologue analyzer, CFG builder & ioctl callsite scanner
│   │       │   ├── manifest.rs        # SHA-256/BLAKE3 digests & golden binary/layout manifest parsers
│   │       │   ├── relocations.rs     # .rela.dyn resolver & PJRT/SDK/Driver/Itanium C++ vtable reconstructors
│   │       │   └── symbols.rs         # goblin/object symbol inspector, gimli .eh_frame & C++ demangler
│   │       ├── error.rs               # Unified LibTpuError enum & PJRT/Abseil/SDK status codes
│   │       ├── types.rs               # Domain primitives module root & re-exports
│   │       ├── types/
│   │       │   ├── address.rs         # Memory addresses, sizes, alignments & PCIe/kernel primitives
│   │       │   ├── element.rs         # ElementType (PJRT_Buffer_Type) & f16/bf16/f8/sub-byte codecs
│   │       │   ├── enums.rs           # Runtime, PJRT, DMA, collective, routing & memory space enums
│   │       │   ├── hardware.rs        # TpuGeneration, TpuCoreTypeEnum & 3D torus ChipCoordinates
│   │       │   ├── ids.rs             # Newtype IDs (derive_more) for topology, handles & registers
│   │       │   └── shape.rs           # MemoryLayout, Shape (TPU tile permutation) & TypedDeviceBuffer
│   │       ├── config.rs              # TpuInitConfig & LIBTPU_INIT_ARGS / env variable parser
│   │       ├── proto.rs               # Wire-compatible Protobuf codecs & prost_reflect DescriptorPool
│   │       ├── compiler.rs            # HLO/StableHLO lexer, parser, 6-pass compiler & VLIW ISA
│   │       ├── hal.rs                 # TpuHal trait, SimulatedTpuHal, LinuxDeviceHal (Gasket/VFIO)
│   │       ├── pjrt.rs                # Thread-safe PJRT core objects (Client, Buffer, Executable...)
│   │       └── driver.rs              # Legacy TfTpu, TpuDriver, TpuProfiler, Embedding & SDK engines
│   ├── libtpu-c-api/                  # #[repr(C)] ABI layer exporting libtpu.so & sdk.so symbols
│   │   ├── Cargo.toml
│   │   ├── build.rs                   # Configures DT_SONAME (libtpu.so) & LC_ID_DYLIB (@rpath/libtpu.dylib)
│   │   ├── cbindgen.toml              # C header generation configuration
│   │   └── src/
│   │       ├── lib.rs                 # GetPjrtApi vtable assembly & ABI verification test suite
│   │       ├── pjrt_types.rs          # PJRT_Api (1,144 B) & 138 #[repr(C)] PJRT_*_Args module root
│   │       ├── pjrt_types/
│   │       │   ├── api.rs             # PJRT_Api v0.116 vtable, PJRT_Api_Version & PJRT_Extension_Base
│   │       │   ├── buffer_args.rs     # #[repr(C)] PJRT_Buffer_*, Stream & AsyncTransferManager args
│   │       │   ├── client_args.rs     # #[repr(C)] PJRT_Client_* & PJRT_Compile_Args structs
│   │       │   ├── device_args.rs     # #[repr(C)] PJRT_DeviceDescription_*, Device, Memory & Topology args
│   │       │   ├── error_args.rs      # #[repr(C)] PJRT_Error_*, Plugin, Event & ExecuteContext args
│   │       │   ├── executable_args.rs # #[repr(C)] PJRT_Executable_* & PJRT_LoadedExecutable_* args
│   │       │   └── handles.rs         # Opaque PJRT C handles, common descriptors & PjrtArgsStruct trait
│   │       ├── pjrt_error_api.rs      # PJRT error, plugin initialization, and event C entrypoints
│   │       ├── pjrt_client_api.rs     # PJRT client, topology description, and context C entrypoints
│   │       ├── pjrt_device_api.rs     # PJRT device, device description, and memory C entrypoints
│   │       ├── pjrt_executable_api.rs # PJRT executable and loaded executable C entrypoints
│   │       ├── pjrt_buffer_api.rs     # PJRT buffer, stream, and async transfer manager C entrypoints
│   │       ├── extensions.rs          # 17 PJRT extension chain coordinator & runtime state
│   │       ├── extensions/
│   │       │   ├── args.rs            # #[repr(C)] extension argument structs, enums & size assertions
│   │       │   ├── collectives.rs     # PJRT Collectives & CrossHostTransfers extensions
│   │       │   ├── compiler.rs        # PJRT XlaTransform, Callback, PhaseCompile & Triton extensions
│   │       │   ├── executable.rs      # PJRT AbiVersion, Shardings, TpuExecutable & Metadata extensions
│   │       │   ├── megascale.rs       # PJRT MultiSlice & Megascale extensions
│   │       │   ├── memory.rs          # PJRT HostMemoryAllocator, MemoryDescriptions, Layouts & RawBuffer
│   │       │   ├── profiler.rs        # PJRT Profiler extension & PLUGIN_Profiler_Api vtable
│   │       │   └── topology.rs        # PJRT TpuTopology extension (31 topology query entrypoints)
│   │       ├── tpu_driver_types.rs    # LibtpuSdk_Api (144 B), TpuDriverFn (224 B) & legacy structs
│   │       ├── sdk_api.rs             # GetLibtpuSdkApi & 17 LibtpuSdk_* C entrypoints
│   │       ├── legacy_tf_api.rs       # TfTpu_*, TpuDriver_*, TpuProfiler_*, TpuEmbeddingEngine_* & PyInit_sdk
│   │       └── rust_ergonomic_api.rs  # 18 ergonomic libtpu_rs_* C/C++ helper entrypoints
│   └── libtpu-py/                     # Rust helper crate for Python runtime sessions
│       ├── Cargo.toml
│       └── src/
│           └── lib.rs                 # PyTpuRuntime & PySdkBridge high-level execution helpers
├── cpp/                               # C11 & C++20 native bindings and CMake build system
│   ├── CMakeLists.txt                 # Root CMake configuration (C11, C++20, llvm-cov support)
│   ├── Doxyfile                       # Strict Doxygen configuration (WARN_AS_ERROR = YES)
│   ├── src/
│   │   ├── CMakeLists.txt             # Library target & Rust cdylib linkage
│   │   ├── libtpu.h                   # Complete C11 ABI header
│   │   ├── libtpu.hpp                 # Zero-exception C++20 RAII wrapper classes
│   │   └── libtpu.cpp                 # C++20 method implementations
│   └── tests/
│       └── test_libtpu.cpp            # C++20 unit & integration test suite
├── python/                            # Drop-in Python wheel package (`libtpu` & `libtpu.sdk`)
│   ├── libtpu/
│   │   ├── __init__.py                # Core Python API (TpuClient, TpuBuffer, TpuExecutable...)
│   │   ├── __init__.pyi               # PEP 561 type stubs
│   │   ├── py.typed                   # PEP 561 marker
│   │   └── sdk/
│   │       ├── __init__.py            # Re-exports tpumonitoring, slice, tpuz, debug (+ .pyi stubs)
│   │       ├── tpumonitoring.py       # Runtime telemetry & metric queries via GetLibtpuSdkApi
│   │       ├── slice.py               # 3D torus chip coordinate & topology queries
│   │       ├── tpuz.py                # Core state summary & HLO diagnostic dumps
│   │       └── debug.py               # C FFI debug callback registration & invocation
│   └── tests/
│       ├── test_libtpu.py             # End-to-end Python PJRT, profiler & SDK tests
│       └── test_pre_commit.py         # Unit tests for the pre-commit & forensics pipelines
├── docs/
│   └── reverse_engineering/           # Binary analysis specifications of official libtpu-0.0.50
│       ├── exported_symbols.md        # All 78 exported libtpu.so ELF symbols & C signatures
│       ├── pjrt_abi_layout.md         # Exact byte offsets of PJRT_Api, extensions & LibtpuSdk_Api
│       ├── kernel_hardware_protocol.md# Gasket/VFIO ioctls, BAR0 MMIO CSRs & Protobuf wire schemas
│       ├── python_api.md              # Python wheel & libtpu.sdk API specification
│       ├── golden_binary_manifest.json# Golden SHA-256, symbol, and vtable layout manifest
│       ├── golden_struct_layouts.json # Disassembled prologue struct sizes & field offsets
│       ├── golden_proto_descriptors.json # Topologically sorted embedded Protobuf descriptors
│       ├── csr_register_map.json      # Per-generation DeepSea BAR0 CSR & SyncFlag offsets
│       └── protos/                    # 64 clean-room .proto schemas carved from libtpu.so
└── scripts/
    ├── pre_commit.py                  # Cross-language formatting, linting, ABI, testing & shield updater
    └── re_extract.py                  # Clean-room ELF64/wheel forensics & golden manifest verifier
```

---

## 3. Multi-Layered System Architecture

`libtpu-rs` is structured as a strict, acyclic stack of seven architectural layers. Lower layers never depend on higher layers, and all `unsafe extern "C"` pointer manipulation is isolated exclusively to `crates/libtpu-c-api` (plus the 3-function POSIX `libc` FFI module in `hal.rs`).

```text
┌───────────────────────────────────────────────────────────────────────────────────┐
│                     External Consumers & Language Bindings                        │
│  ┌─────────────────────┐  ┌─────────────────────────┐  ┌───────────────────────┐  │
│  │  JAX / PyTorch/XLA  │  │  Python `libtpu` &      │  │  C++20 RAII SDK       │  │
│  │  & TensorFlow C API │  │  `libtpu.sdk` (`ctypes`)│  │  (`cpp/src/libtpu.*`) │  │
│  └──────────┬──────────┘  └────────────┬────────────┘  └───────────┬───────────┘  │
└─────────────┼──────────────────────────┼───────────────────────────┼──────────────┘
              │ dlopen / C ABI           │ ctypes CDLL               │ C11 Header
              ▼                          ▼                           ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│                   Layer 5: C ABI Boundary (`crates/libtpu-c-api`)                 │
│  • `PJRT_Api` v0.116 (1,144 B, 138 slots) + 17-node `PJRT_Extension_Base` list    │
│  • `LibtpuSdk_Api` v0.1 (144 B, 17 slots) + `TpuDriverFn` (224 B, 28 slots)       │
│  • 325 `#[no_mangle] pub unsafe extern "C"` symbols & `libtpu_rs_*` C helpers     │
└────────────────────────────────────────┬──────────────────────────────────────────┘
                                         │ Safe Rust Calls
                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│            Layer 4: Runtime State Machines (`pjrt.rs` & `driver.rs`)              │
│  • `PjrtCoreClient`, `PjrtCoreDevice`, `PjrtCoreMemory`, `PjrtCoreBuffer`,        │
│    `PjrtCoreExecutable`, `PjrtCoreLoadedExecutable`, `PjrtCoreEvent`,             │
│    `PjrtCoreTopology`, `PjrtCoreCopyToDeviceStream`, `PjrtCoreAsyncTransferManager`│
│  • `TpuCoreDriver`, `TpuCoreProfiler`, `TpuCoreEmbeddingEngine`, `LibtpuSdkCoreClient`│
└───────────────────┬───────────────────────────────────────────┬───────────────────┘
                    │                                           │
                    ▼                                           ▼
┌───────────────────────────────────────┐   ┌───────────────────────────────────────┐
│  Layer 2: Compiler (`compiler.rs`)    │   │   Layer 3: Hardware HAL (`hal.rs`)    │
│  • `HloLexer` & `HloParser`           │   │  • `TpuHal` trait                     │
│  • 45 `TpuOpCode` VLIW opcodes        │   │  • `SimulatedTpuHal` (3D ICI torus,   │
│  • 6-pass optimizer & layout engine   │   │    HBM allocator, VLIW interpreter)   │
│  • Cost analyzer & 128-bit FNV-1a     │   │  • `LinuxDeviceHal` (Gasket `0xdc` &  │
│  • `LTPUEXEC` & Protobuf serializer   │   │    VFIO `0x3b` ioctls, BAR0 MMIO CSRs)│
└───────────────────┬───────────────────┘   └───────────────────┬───────────────────┘
                    │                                           │
                    └───────────────────────┬───────────────────┘
                                            ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│        Layer 1: Core Primitives, Config, Protobuf & Error (`libtpu-core`)         │
│  • `types.rs`: Newtypes (`DeviceId`, `ChipId`, `ByteSize`...), `Shape`, `ElementType`│
│  • `error.rs`: Single `LibTpuError` enum (`derive_more`) + PJRT/Abseil/SDK codes  │
│  • `config.rs`: `TpuInitConfig` & `LIBTPU_INIT_ARGS` / environment parser         │
│  • `proto.rs`: Zero-dependency Protobuf wire codec (`ProtoWriter`/`ProtoReader`)  │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Layer 1: Core Domain & Unified Error Model (`libtpu-core`)

### 4.1 Strongly-Typed Domain Primitives (`types.rs`)

To prevent primitive obsession and accidental mixing of indices, addresses, offsets, and byte counts, `crates/libtpu-core/src/types.rs` wraps every scalar domain concept in a `#[repr(transparent)]` newtype powered by `derive_more` (`Display`, `From`, `Into`, `Add`, `Sub`, `AddAssign`, `SubAssign`, `LowerHex`, `UpperHex`, `Binary`, `Octal`):

| Newtype | Inner Type | Purpose |
| :--- | :--- | :--- |
| `DeviceId` | `u32` | Global logical TPU device ordinal across a slice |
| `ChipId` | `u32` | Physical TPU ASIC package identifier |
| `CoreId` | `u32` | `TensorCore` index within a physical TPU chip (`0` or `1`) |
| `HostId` | `u32` | Worker host VM identifier within a multi-host pod slice |
| `ProcessIndex` | `i32` | PJRT process index (`PJRT_Client_ProcessIndex`) |
| `LocalHardwareId` | `i32` | Host-local PCIe device ordinal (`-1` for remote devices) |
| `MemorySpaceId` | `i32` | Unique PJRT memory space identifier (`HBM`, `VMEM`, `PinnedHost`) |
| `BufferId` | `u64` | Globally unique `PJRT_Buffer` allocation handle ID |
| `ExecutableId` | `u64` | Globally unique `PJRT_Executable` handle ID |
| `EventId` | `u64` | Globally unique `PJRT_Event` completion handle ID |
| `CallbackId` | `u64` | Registered `LibtpuSdk` debug callback handle ID |
| `DeviceAddress` | `u64` | Virtual/physical HBM or VMEM address in TPU memory space (`0x1000_0000+`) |
| `ByteSize` | `usize` | Memory size or transfer length in bytes |
| `MmioOffset` | `u64` | Byte offset into a PCIe BAR MMIO register aperture |
| `IoctlCommand` | `u64` | Raw Linux `ioctl` command word (`GASKET_IOCTL_*` / `VFIO_*`) |
| `FileDescriptor` | `i32` | POSIX device file descriptor (`/dev/accel*` or `/dev/vfio/*`) |
| `EventFdHandle` | `i32` | Linux `eventfd` interrupt notification descriptor |
| `SliceId` | `u32` | Megascale multi-slice DCN slice identifier |
| `CommunicatorId` | `u64` | PJRT collective clique communicator identifier |
| `AbiVersionNumber` | `u64` | Runtime and executable ABI version stamp |
| `TransformId` | `u64` | XLA HLO transform pipeline identifier |

In addition to newtypes, `types.rs` defines the core hardware and tensor representations:

- **`TpuGeneration`**: Models all seven Google Cloud TPU ASIC generations (`V2`, `V3`, `V4`, `V5e`, `V5p`, `V6e`, `V7x`), encapsulating hardware-specific parameters:
  - **PCIe Device IDs**: `0x0027` (`v2`/`v3`), `0x0056` (`v4`), `0x0063` (`v5e`), `0x0062` (`v5p`), `0x006f` (`v6e`), `0x0073` (`v7x`). (All paired with Google Vendor ID `0x1ae0`).
  - **Logical Devices (`TensorCore`s) per Chip**: `2` for dual-core generations (`v2`, `v3`, `v4`, `v5p`, `v7x`) and `1` for single-core high-efficiency generations (`v5e`, `v6e`).
  - **HBM Capacity per Device**: `8 GiB` (`v2`), `16 GiB` (`v3`, `v4`, `v5e`), `32 GiB` (`v6e`), `48 GiB` (`v5p`), `96 GiB` (`v7x`).
  - **On-Chip VMEM Scratchpad Capacity**: `16 MiB` (`v2`, `v3`), `32 MiB` (`v4`, `v5e`), `64 MiB` (`v5p`, `v6e`), `128 MiB` (`v7x`).
  - **SparseCore Count**: `0` (`v2`, `v3`), `1` (`v5e`), `2` (`v4`, `v6e`), `4` (`v5p`, `v7x`).
  - **MXU Systolic Array Width**: `128x128` (`v2`–`v5p`) and `256x256` (`v6e` Trillium, `v7x` Ironwood).
- **`ChipCoordinates`**: 3D torus coordinates `(x, y, z)` with Manhattan distance calculation on 3D meshes.
- **`ElementType`**: All 15 PJRT primitive tensor element types (`Pred`, `S8`, `S16`, `S32`, `S64`, `U8`, `U16`, `U32`, `U64`, `F16`, `F32`, `F64`, `C64`, `C128`, `Bf16`), including lossless IEEE 754 `f16` and `bf16` software bit-packing/unpacking (`pack_from_f64_slice` / `unpack_to_f64_vec`) so the VLIW simulator executes half-precision and brain-float operations natively.
- **`MemoryKind`**: Classifies memory spaces into `DeviceHbm` (`"device"`, ID `100`), `PinnedHost` (`"pinned_host"`, ID `101`), `UnpinnedHost` (`"unpinned_host"`, ID `102`), and `Vmem` (`"vmem"`, ID `103`).
- **`MemoryLayout` & `Shape`**: Models N-dimensional tensor shapes, dynamic dimension masks, minor-to-major dimension orderings, and TPU `(8, 128)` matrix sublane tiling (`default_tpu_tiled`). Provides `unpadded_byte_size()`, `on_device_byte_size()` (aligned to 256-byte TPU HBM granules), N-dimensional coordinate-to-linear and linear-to-coordinate mappings, `reshape()`, and `transpose()`.

### 4.2 Unified Error Hierarchy (`error.rs`)

In accordance with workspace quality standards, `libtpu-rs` forbids `anyhow` and ad-hoc error types. Every fallible function in the workspace returns `LibTpuResult<T> = Result<T, LibTpuError>`.

`LibTpuError` (`crates/libtpu-core/src/error.rs`) is a single comprehensive enum using `derive_more::{Display, Error, From}` covering all failure domains:

- **ABI & Pointer Validation**: `NullPointer`, `InvalidStructSize`, `VersionMismatch`, `InvalidUtf8` (`#[from] Utf8ErrorWrapper`), `InteriorNul` (`#[from] NulErrorWrapper`).
- **Arguments, Shapes & Elements**: `InvalidArgument`, `UnsupportedElementType`, `ShapeMismatch`.
- **Topology & Memory Management**: `DeviceNotFound`, `MemorySpaceNotFound`, `OutOfMemory`, `BufferDeleted`, `TransferFailed`, `StreamOverflow`, `DmaMappingNotFound`.
- **Compilation & Execution**: `CompilationFailed`, `PhaseCompilationFailed`, `DeserializationFailed`, `ExecutableDeleted`, `ExecutionFailed`.
- **Hardware & Kernel Driver**: `IoctlFailed`, `MmioOutOfBounds`, `HardwareIoError`, `LockPoisoned`.
- **Profiler, SDK & Protobuf**: `ProfilerAlreadyActive`, `ProfilerNotActive`, `MetricNotFound`, `CallbackNotFound`, `ProtobufDecodeError`.

`LibTpuError` maps deterministically onto all three C status code domains used by Google's runtime:
1. **`PjrtErrorCode`** (`PJRT_Error_Code`): `Ok (0)`, `Cancelled (1)`, `Unknown (2)`, `InvalidArgument (3)`, `DeadlineExceeded (4)`, `NotFound (5)`, `AlreadyExists (6)`, `PermissionDenied (7)`, `ResourceExhausted (8)`, `FailedPrecondition (9)`, `Aborted (10)`, `OutOfRange (11)`, `Unimplemented (12)`, `Internal (13)`, `Unavailable (14)`, `DataLoss (15)`, `Unauthenticated (16)`.
2. **`AbseilStatusCode`**: `absl::StatusCode` integer codes (`0..=16`) for `TpuStatus_*` and `TfTpu_*` APIs.
3. **`LibtpuSdkErrorCode`**: Status codes for `LibtpuSdk_Error_GetCode`.

### 4.3 Runtime Configuration & Environment Parser (`config.rs`)

`TpuInitConfig` (`crates/libtpu-core/src/config.rs`) replicates the flag and environment variable parsing behavior of `libtpu.so`:

- **Process-Wide Synchronization**: A global thread-safe `OnceLock<Mutex<TpuInitConfig>>` (`global_init_config()` / `set_global_init_config()` / `apply_global_cli_args()`) synchronizes configuration across `TfTpu_Initialize`, `SetGlobalTPUArrayOp_DoWork`, `PJRT_Client_Create`, and Python's `libtpu.set_init_args()`.
- **Environment Variables Parsed (`TpuInitConfig::from_env_map`)**:
  - `TPU_ACCELERATOR_TYPE` -> `generation` (`v2`..`v7x`)
  - `TPU_CHIPS_PER_HOST_BOUNDS` -> `chips_per_host_bounds` (e.g., `"2,2,1"` or `"2x2x1"`)
  - `TPU_HOST_BOUNDS` / `TPU_PROCESS_BOUNDS` -> `host_bounds`
  - `TPU_WORKER_ID` -> `process_index`
  - `TPU_VISIBLE_DEVICES` -> `visible_devices` allowlist
  - `TPU_PREMAPPED_BUFFER_SIZE` -> `premapped_buffer_size` (defaults to `64 MiB`)
  - `MEGASCALE_NUM_SLICES` & `MEGASCALE_SLICE_ID` -> Multi-slice DCN configuration
  - `LIBTPU_INIT_ARGS` -> Whitespace-separated `--xla_tpu_*` and `--tpu_*` flags (`--tpu_generation`, `--xla_tpu_allow_async_allocations`, `--tpu_chips_per_host_bounds`, `--tpu_host_bounds`, plus arbitrary passthrough flags stored in `extra_flags`).

### 4.4 Zero-Dependency Protobuf Wire Codec (`proto.rs`)

Because `libtpu.so` exchanges serialized Protocol Buffer messages across its C ABI (`PJRT_Client_Compile`, `PJRT_TopologyDescription_Serialize`, `TpuProfiler_CollectData`, `TpuEmbeddingEngine_*`, etc.), `crates/libtpu-core/src/proto.rs` implements a complete, zero-dependency Protocol Buffers binary wire-format encoder (`ProtoWriter`) and decoder (`ProtoReader`) supporting wire types `0` (Varint / ZigZag `sint64`), `1` (64-bit fixed), `2` (Length-delimited bytes, strings, sub-messages, and packed repeated fields), and `5` (32-bit fixed).

Sixteen wire-compatible Google/XLA/TensorFlow protobuf schemas are implemented with full round-trip `encode()` and `decode()`:

1. `TileProto` (`xla.TileProto`)
2. `LayoutProto` (`xla.LayoutProto`)
3. `ShapeProto` (`xla.ShapeProto`)
4. `OpShardingProto` (`xla.OpSharding`)
5. `PrecisionConfigProto` (`xla.PrecisionConfig`)
6. `HloInstructionProto` (`xla.HloInstructionProto`)
7. `HloComputationProto` (`xla.HloComputationProto`)
8. `HloModuleProto` (`xla.HloModuleProto`)
9. `HloModuleConfigProto` (`xla.HloModuleConfigProto`)
10. `HloModuleProtoWithConfig` (`xla.HloModuleProtoWithConfig`)
11. `ExecutableBuildOptionsProto` (`xla.ExecutableBuildOptionsProto`)
12. `CompileOptionsProto` (`xla.CompileOptionsProto`)
13. `TpuTopologyArgsProto` (`tpu.TpuTopologyArgsProto`)
14. `TpuSerializedExecutableProto` (`tpu.TpuSerializedExecutableProto`)
15. `TpuEmbeddingConfigurationProto` + `EmbeddingTableDescriptorProto` + `TpuEmbeddingPartitionedConfigProto` + `TpuEmbeddingMemoryConfigProto` + `TpuEmbeddingHostNetworkConfigProto` (`tensorflow.tpu.TPUEmbeddingConfiguration`)
16. `XSpaceProto` + `XPlaneProto` + `XLineProto` + `XEventProto` + `XStatProto` (`tensorflow.profiler.XSpace`)

---

## 5. Layer 2: HLO & StableHLO Compiler Pipeline (`compiler.rs`)

`crates/libtpu-core/src/compiler.rs` implements the compiler pipeline backing `PJRT_Client_Compile`, `PJRT_Compile`, `TpuDriver_CompileProgram`, `TpuDriver_CompileProgramFromText`, and `PJRT_PhaseCompile_Extension`.

### 5.1 Input Formats & Lexical/Syntactic Frontend

`CompiledProgram::compile(format, code, compile_options)` accepts three PJRT program formats (`"mlir"`, `"hlo"`, and `"hlo_with_config"`):
- **MLIR Bytecode Detection**: Strips the 4-byte MLIR bytecode magic header (`b"ML\xefR"`) if present.
- **Binary Protobuf Lowering**: Automatically detects and decodes binary `HloModuleProto` or `HloModuleProtoWithConfig` payloads via `lower_from_hlo_proto`.
- **Textual Lexer & Recursive-Descent Parser**: `HloLexer` tokenizes textual `StableHLO` MLIR (`module @name { func.func @main(...) -> ... }`) and textual XLA HLO (`HloModule name ... ENTRY %main ...`) into `HloToken` streams (`Ident`, `SsaValue`, `Symbol`, `Number`, `Punct`, `Arrow`). `HloParser` and `CompiledProgram::parse_textual_program` extract SSA parameter bindings, tensor shapes (`tensor<2x4xf32>` or `f32[2,4]{1,0}`), instruction attributes (`dot_general` contracting/batching dimensions, `permutation`, `broadcast_dimensions`, `slice` starts/limits/strides, `dense<...>` constant literals), and build a topologically ordered SSA DAG of `TpuInstructionBundle`s.

### 5.2 VLIW Instruction Bundle ISA (`TpuOpCode` & `TpuInstructionBundle`)

Each lowered operation is represented as a `TpuInstructionBundle` targeting the TPU `TensorCore` Scalar, Vector (VPU), Matrix Multiply (MXU), or Collective (ICI) execution units. The ISA supports **45 opcodes** (`TpuOpCode` `0..=44`):

| Category | Opcodes (`TpuOpCode`) | Execution Unit |
| :--- | :--- | :--- |
| **Data Movement & View** | `Identity (0)`, `Reshape (36)`, `Transpose (37)`, `BroadcastInDim (38)`, `Slice (39)`, `Concatenate (40)`, `Convert (41)`, `Constant (42)`, `Iota (32)`, `Reverse (33)` | Scalar / VMEM / DMA |
| **Elementwise Arithmetic** | `AddF32 (1)`, `AddS32 (2)`, `MulF32 (3)`, `SubF32 (5)`, `DivF32 (6)`, `MaxF32 (7)`, `MinF32 (8)`, `NegF32 (10)`, `Abs (16)`, `Floor (22)`, `Ceil (23)`, `Rem (25)` | Vector Processing Unit (VPU) |
| **Transcendentals** | `ExpF32 (11)`, `TanhF32 (12)`, `Log (17)`, `Sqrt (18)`, `Rsqrt (19)`, `Sine (20)`, `Cosine (21)` | Extended Unary Pipeline (EUP / VPU) |
| **Bitwise, Logical & Compare** | `XorBytes (4)`, `Not (24)`, `And (26)`, `Or (27)`, `CompareEq (28)`, `CompareLt (29)`, `CompareGt (30)`, `Select (31)` | Vector ALU |
| **Matrix Multiplication** | `DotGeneralF32 (9)` (2D & 3D batched matrix contraction) | Matrix Multiply Unit (MXU Systolic Array) |
| **Reductions** | `ReduceSumF32 (13)`, `ReduceMax (34)`, `ReduceMin (35)` | VPU Cross-Lane Reduction |
| **Multi-Chip Collectives** | `AllReduceSumF32 (14)`, `AllGather (43)`, `ReduceScatter (44)` | Inter-Chip Interconnect (ICI Router) |
| **Custom Calls** | `CustomCall (15)` (`TpuCustomCall` / `XLA_FFI`) | Custom Call Dispatcher |

### 5.3 Six-Pass Lowering & Optimization Pipeline

Every compiled program runs through a deterministic 6-pass optimization pipeline (`CompiledProgram::run_all_passes()`):

1. **Pass 1 (`pass_hlo_verification_and_shape_inference`)**: Verifies operand arities, validates shape ranks, and aligns code size to 256-byte instruction cache granules.
2. **Pass 2 (`pass_spmd_sharding_propagation`)**: Propagates `num_replicas` and `num_partitions` from `CompileOptionsProto.executable_build_options`.
3. **Pass 3 (`pass_tpu_layout_assignment_and_tiling`)**: Assigns canonical TPU `(8, 128)` matrix sublane tiled layouts (`MemoryLayout::default_tpu_tiled`) to all output tensors.
4. **Pass 4 (`pass_hlo_operator_fusion`)**: Fuses producer-consumer instruction chains and assigns sequential program counter (`pc_offset`) slots.
5. **Pass 5 (`pass_vmem_scratchpad_allocation`)**: Computes temporary on-chip VMEM scratchpad memory requirements (`temp_size_in_bytes`).
6. **Pass 6 (`pass_llo_vliw_bundle_scheduling`)**: Assigns hardware completion `SyncFlag` IDs (`sync_flag_id`) to each VLIW instruction bundle.

### 5.4 Phased Compilation (`PJRT_PhaseCompile_Extension`)

To support JAX and XLA ahead-of-time phased compilation debugging, `CompiledProgram::run_compilation_phase` implements the 5 canonical named phases exposed by `PJRT_PhaseCompile_GetPhaseNames` and `PJRT_PhaseCompile_RunPhase`:
1. `"hlo_verification"`
2. `"tpu_layout_assignment"`
3. `"tpu_memory_planning"`
4. `"vliw_bundle_emission"`
5. `"binary_serialization"`

### 5.5 Cost Analysis, Memory Stats, Fingerprinting & Serialization

- **Static Cost Analysis (`CostAnalysis`)**: Computes estimated `flops` (accounting for `2 * M * N * K` matrix contractions vs. elementwise operations), `transcendentals`, `bytes_accessed`, and `optimal_seconds` for `PJRT_Executable_GetCostAnalysis`.
- **Memory Statistics (`CompiledMemoryStats`)**: Tracks `generated_code_size_in_bytes`, `argument_size_in_bytes`, `output_size_in_bytes`, `alias_size_in_bytes`, and `temp_size_in_bytes` for `PJRT_Executable_GetCompiledMemoryStats`.
- **Deterministic Fingerprinting**: Computes a 128-bit FNV-1a hexadecimal digest (`32` characters) over the normalized format, source bytes, and compile options (`PJRT_Executable_Fingerprint` / `PJRT_LoadedExecutable_Fingerprint`).
- **Dual Format Serialization**: Supports both `libtpu-rs` native binary serialization (with magic header `b"LTPUEXEC"` and version `1`) and wire-compatible `TpuSerializedExecutableProto` protobuf serialization.

---

## 6. Layer 3: Hardware Abstraction Layer (`hal.rs`)

`crates/libtpu-core/src/hal.rs` decouples the PJRT and legacy driver engines from the underlying execution substrate through the `TpuHal` trait.

### 6.1 The `TpuHal` Trait Contract

```rust
pub trait TpuHal: Send + Sync + std::fmt::Debug {
    fn discover_topology(&self) -> LibTpuResult<TopologySpec>;
    fn allocate_device_memory(&self, device_id: DeviceId, memory_kind: MemoryKind, size: ByteSize) -> LibTpuResult<DeviceAddress>;
    fn deallocate_device_memory(&self, device_id: DeviceId, address: DeviceAddress) -> LibTpuResult<()>;
    fn write_device_memory(&self, device_id: DeviceId, address: DeviceAddress, data: &[u8]) -> LibTpuResult<()>;
    fn read_device_memory(&self, device_id: DeviceId, address: DeviceAddress, out: &mut [u8]) -> LibTpuResult<()>;
    fn copy_device_to_device(&self, src_device: DeviceId, src_addr: DeviceAddress, dst_device: DeviceId, dst_addr: DeviceAddress, size: ByteSize) -> LibTpuResult<()>;
    fn execute_program(&self, device_id: DeviceId, program: &CompiledProgram, inputs: &[DeviceAddress], outputs: &[DeviceAddress]) -> LibTpuResult<()>;
    fn memory_stats(&self, device_id: DeviceId) -> LibTpuResult<DeviceMemoryStats>;
    fn clear_memory_stats(&self, device_id: DeviceId) -> LibTpuResult<()>;
    fn poison_device(&self, device_id: DeviceId, reason: String) -> LibTpuResult<()>;
}
```

### 6.2 Deterministic In-Memory Simulator (`SimulatedTpuHal`)

`SimulatedTpuHal` provides a complete, thread-safe, in-memory TPU pod slice simulator:
- **3D Torus Topology Synthesis**: Synthesizes multi-host 3D torus meshes (`host_bounds` × `chips_per_host_bounds` × `logical_devices_per_chip`), assigning accurate `DeviceId`, `ChipId`, `CoreId`, `HostId`, `ProcessIndex`, `LocalHardwareId`, and `is_addressable` flags, while honoring `TPU_VISIBLE_DEVICES`.
- **Page-Aligned HBM Allocator**: Tracks virtual HBM allocations starting at `0x1000_0000` (4 KiB page-aligned, 256-byte minimum granule), enforcing per-generation HBM capacity limits and maintaining live `DeviceMemoryStats` (`bytes_in_use`, `peak_bytes_in_use`, `num_allocs`, `largest_alloc_size`, `largest_free_block_bytes`).
- **Shape-Aware VLIW Bundle Interpreter**: Executes arbitrary multi-instruction SSA DAGs in `CompiledProgram`, evaluating all 45 `TpuOpCode` operations across all 15 `ElementType`s (`F32`, `BF16`, `F16`, `S32`, `Pred`, etc.), including 2D/3D batched `DotGeneralF32`, N-dimensional `Transpose`, `BroadcastInDim`, strided `Slice`, `Concatenate`, `Select`, and cross-replica ICI collectives (`execute_collective` for `AllReduceSumF32`, `AllGather`, and `ReduceScatter`).
- **Device Poisoning**: Supports `PJRT_Device_PoisonExecution` to inject sticky fault states into specific devices for fault-tolerance testing.

### 6.3 Linux Kernel Driver Backend (`LinuxDeviceHal`, Gasket & VFIO)

`LinuxDeviceHal` implements the user-space hardware protocol used by `libtpu.so` (`asic_sw::driver`) to communicate with physical Cloud TPU PCIe devices on Linux hosts via two kernel driver modes (`KernelDriverMode`):

1. **Gasket Driver (`KernelDriverMode::Gasket`, `/dev/accel*`, Magic `'0xdc'`)**:
   - Used on TPU v2, v3, and v4 VMs.
   - Exact `#[repr(C)]` `ioctl` commands and payload sizes verified at compile time:
     - `GASKET_IOCTL_NUMBER_PAGE_TABLES` (`0xc020_dc05`, 32 B)
     - `GASKET_IOCTL_PARTITION_PAGE_TABLE` (`0x4020_dc07`, 32 B)
     - `GASKET_IOCTL_MAP_BUFFER` (`0x4020_dc08`, `GasketPageTableIoctl`, 32 B)
     - `GASKET_IOCTL_UNMAP_BUFFER` (`0x4020_dc09`, `GasketPageTableIoctl`, 32 B)
     - `GASKET_IOCTL_SET_EVENTFD` (`0x4020_dc0b`, `GasketInterruptEventFd`, 32 B)
     - `GASKET_IOCTL_CLEAR_EVENTFD` (`0x4008_dc0c`, 8 B)
     - `GASKET_IOCTL_MAP_DMABUF` (`0x4018_dc0d`, `GasketPageTableDmaBuf`, 24 B)
2. **VFIO Driver (`KernelDriverMode::Vfio`, `/dev/vfio/vfio`, Magic `'0x3b'`)**:
   - Used on TPU v5e, v5p, v6e (Trillium), and v7x (Ironwood) VMs.
   - Exact `#[repr(C)]` `ioctl` commands and payload sizes verified at compile time:
     - `VFIO_DEVICE_SET_IRQS_IOCTL` (`0x0000_3b6e`, `VfioIrqSetIoctl`, 24 B)
     - `VFIO_IOMMU_MAP_DMA_IOCTL` (`0x0000_3b71`, `VfioDmaMapIoctl`, 32 B)
     - `VFIO_IOMMU_UNMAP_DMA_IOCTL` (`0x0000_3b72`, `VfioDmaUnmapIoctl`, 24 B)

### 6.4 PCIe BAR0 MMIO Control & Status Register (CSR) Map

Both `SimulatedTpuHal` and `LinuxDeviceHal` maintain a cycle-accurate PCIe BAR0 MMIO Control & Status Register bank (`EmulatedKernelDriverBackend`):

| Constant | BAR0 Offset | Description |
| :--- | :--- | :--- |
| `BAR0_CSR_TENSORCORE_RUN_CONTROL` | `0x0000_1000` | `TensorCore` 0 Run Control CSR (`0` = halted, `1` = running) |
| `BAR0_CSR_TENSORCORE_SYNC_FLAG` / `PC` | `0x0000_1008` | `TensorCore` 0 Program Counter / `SyncFlag` Completion Counter |
| `BAR0_CSR_TENSORCORE_STATUS` | `0x0000_1010` | `TensorCore` 0 Execution Status CSR (`2` = completed) |
| `BAR0_CSR_SYNC_FLAG_BASE` | `0x0000_2000` | Hardware `SyncFlag` table base (`4096` × 32-bit sync counters) |
| `BAR0_CSR_SPARSECORE_RUN_CONTROL` | `0x0000_4000` | `SparseCore` 0 Run Control CSR |
| `BAR0_CSR_HBM_DMA_RING_HEAD` | `0x0000_8000` | HBM DMA descriptor ring consumer head pointer |
| `BAR0_CSR_HBM_DMA_RING_TAIL` | `0x0000_8008` | HBM DMA descriptor ring producer tail doorbell |
| `BAR0_CSR_ICI_ROUTER_LINK_STATE` | `0x0000_9000` | Inter-Chip Interconnect (ICI) optical/electrical router link state |
| `BAR0_CSR_THERMAL_MILLICELSIUS` | `0x0000_A000` | ASIC junction temperature in millicelsius (default `45,000` m°C) |
| `BAR0_CSR_POWER_MILLIWATTS` | `0x0000_A008` | Board power draw in milliwatts (default `175,000` mW) |
| `BAR0_CSR_DUTY_CYCLE_PERCENT` | `0x0000_A010` | `TensorCore` MXU duty cycle percentage (`0..=100%`) |

### 6.5 Injectable OS System Call Interface (`SyscallProvider`)

To allow 100% test coverage of `LinuxDeviceHal` (including `EBADF` and kernel `ioctl` failure paths) without requiring root privileges or physical `/dev/accel0` device nodes, `LinuxDeviceHal` accepts an injectable `Arc<dyn SyscallProvider>`:
- **`EmulatedKernelDriverBackend`**: In-memory implementation tracking open file descriptors, IOMMU page table IOVA mappings, and registered `eventfd` interrupt vectors.
- **`PosixKernelDriverBackend`**: Real OS backend invoking `libc::open`, `libc::ioctl`, and `libc::close` directly.

---

## 7. Layer 4: PJRT & Legacy Runtime Engines (`pjrt.rs` & `driver.rs`)

### 7.1 PJRT Core Domain Engine (`pjrt.rs`)

`crates/libtpu-core/src/pjrt.rs` implements the safe, thread-safe Rust domain objects that back every opaque `PJRT_*` handle:

- **`PjrtCoreClient`**: Owns the `Arc<dyn TpuHal>`, discovered `PjrtCoreTopology`, all logical `PjrtCoreDevice`s, addressable device indices, `PjrtCoreMemory` spaces (creating `DeviceHbm`, `Vmem`, and `PinnedHost` memory spaces per device), and active host DMA mappings (`dma_map` / `dma_unmap`).
- **`PjrtCoreDevice` & `PjrtCoreDeviceDescription`**: Expose device IDs, process indices, hardware coordinates (`coords_attr`), debug/display strings, and memory space associations.
- **`PjrtCoreBuffer`**: Represents an allocated tensor in TPU HBM. Tracks atomic deletion (`deleted: AtomicBool`), external reference counts (`external_ref_count: AtomicI64`), and provides `to_host_bytes()`, `copy_to_device()`, and `bitcast()`.
- **`PjrtCoreExecutable` & `PjrtCoreLoadedExecutable`**: Wrap `CompiledProgram` and manage multi-buffer on-device execution (`execute_on_device`), cost analysis, and memory stats.
- **`PjrtCoreEvent`**: Lock-free `AtomicBool` readiness flag paired with an optional `Mutex<Option<LibTpuError>>` supporting `is_ready()`, `set()`, `get_error()`, and `await_ready()`.
- **`PjrtCoreCopyToDeviceStream` & `PjrtCoreAsyncTransferManager`**: Manage chunked and asynchronous host-to-device tensor streaming with bounds checking (`StreamOverflow`).

### 7.2 Legacy TensorFlow, Driver, Profiler, Embedding & SDK Engines (`driver.rs`)

`crates/libtpu-core/src/driver.rs` implements the state machines backing the non-PJRT C APIs exported by `libtpu.so` and `sdk.so`:

- **`TpuCoreStatus`**: Backing state for `TpuStatus_*` (`code: PjrtErrorCode`, `message: String`).
- **`TfTpuCoreOrdinalSelector`**: Lock-free round-robin device ordinal selector (`TfTpuOrdinalSelector_*`).
- **`TpuCoreProfiler`**: Manages profiling lifecycle (`start()` / `stop()`, rejecting double-start with `ProfilerAlreadyActive` and stop-when-idle with `ProfilerNotActive`) and serializes wire-compatible `tensorflow.profiler.XSpace` protobuf traces (`collect_xspace_bytes()`).
- **`TpuCoreEmbeddingEngine`**: Implements the `SparseCore` embedding lifecycle (`execute_partitioner`, `configure_memory`, `collate_memory`, `configure_host`, `connect_hosts`, `finalize`, `write_parameters`, `read_parameters`, `enqueue_tensor_batch`) and generates wire-encoded `HloModuleProto` computations for the 5 `TpuEmbeddingEngine_*Computation` entrypoints (`RecvTPUEmbeddingActivations`, `RecvTPUEmbeddingDeduplicationData`, `SendTPUEmbeddingGradients`, `TPUEmbeddingDeduplicationDataSize`, `TPUEmbeddingDeduplicationTupleMask`).
- **`TpuCoreDriver`**: Backing engine for the 28-slot `TpuDriver_*` C API (`open`, `reset`, `query_system_info`, `allocate`, `deallocate`, `transfer_to_device`, `transfer_from_device`, `transfer_device_to_device`, `compile`, `execute_program`).
- **`LibtpuSdkCoreClient`**: Backing engine for `GetLibtpuSdkApi`, maintaining 3D chip coordinates, runtime telemetry metrics (`tpu.runtime.hbm.usage_bytes`, `tpu.runtime.tensorcore.utilization`, `tpu.runtime.ici.link_bandwidth_bytes`, `tpu.runtime.dma.throughput_bytes`, `tpu.runtime.compile.time_micros`), core state summaries, and registered debug callbacks.

---

## 8. Layer 5: C ABI & Binary Layout (`libtpu-c-api`)

`crates/libtpu-c-api` compiles to both a dynamic shared library (`cdylib`: `libtpu.dylib` / `libtpu.so` / `tpu.dll`), a static archive (`staticlib`), and a Rust library (`rlib`). It exports all **325 ELF symbols** from `libtpu-0.0.50` plus `GetLibtpuSdkApi` and ergonomic `libtpu_rs_*` entrypoints.

### 8.1 `PJRT_Api` v0.116 Vtable (`1,144` Bytes, 138 Slots)

Returned by `GetPjrtApi()`, the static `PJRT_API_VTBL` structure in `crates/libtpu-c-api/src/pjrt_types.rs` has an exact size of **`1,144` bytes (`0x478`)**:

- Offset `0x000` (`8` B): `struct_size = 1144` (`0x478`)
- Offset `0x008` (`8` B): `extension_start` -> Points to `&EXT_01_XLA_TRANSFORM.base`
- Offset `0x010` (`24` B): `pjrt_api_version` (`struct_size = 24`, `extension_start = NULL`, `major_version = 0`, `minor_version = 116`)
- Offsets `0x028..=0x470` (`138 * 8 = 1,104` B): 138 named `extern "C"` function pointers (`PJRT_Error_Destroy` through `PJRT_TopologyDescription_GetMemorySpaceKindIds`).

Every single `PJRT_*_Args` structure (e.g., `PJRT_Client_Create_Args` = 72 B, `PJRT_Client_BufferFromHostBuffer_Args` = 112 B, `PJRT_LoadedExecutable_Execute_Args` = 56 B) is defined with `#[repr(C)]` and verified via compile-time `size_of` assertions.

### 8.2 The 17-Node TPU PJRT Extension Linked List (`extensions.rs`)

`PJRT_Api.extension_start` points to a static singly-linked list of **17 `#[repr(C)]` PJRT extension descriptors** in `crates/libtpu-c-api/src/extensions.rs`, matching the exact chain order, `ext_type` discriminants, and byte sizes of `libtpu.so`:

| Order | Extension Struct | `ext_type` | Size (Bytes) | Key Capabilities |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `PJRT_XlaTransform_Extension` | `29` (`0x1d`) | `56` | HLO transform pipeline registration (`Register`, `Apply`, `Clear`, `Count`) |
| 2 | `PJRT_HostMemoryAllocator_Extension` | `28` (`0x1c`) | `32` | Custom host pinned memory allocator (`Allocate`, `Free`) |
| 3 | `PJRT_MultiSlice_Extension` | `26` (`0x1a`) | `64` | Multi-slice config queries (`NumSlices`, `NumDevicesPerSlice`, `SliceId`) |
| 4 | `PJRT_Collectives_Extension` | `25` (`0x19`) | `96` | Cross-device communicator (`CreateCommunicator`, `AllReduce`, `ReduceScatter`, `AllGather`, `CollectivePermute`, `AllToAll`) |
| 5 | `PJRT_AbiVersion_Extension` | `20` (`0x14`) | `120` | Runtime & executable ABI compatibility checks and proto serialization |
| 6 | `PJRT_Shardings_Extension` | `19` (`0x13`) | `40` | Parameter & output `OpShardingProto` extraction |
| 7 | `PJRT_Megascale_Extension` | `18` (`0x12`) | `248` | 29 entrypoints for DCN multi-slice Megascale collectives, barrier, and error aggregation |
| 8 | `PJRT_TpuExecutable_Extension` | `17` (`0x11`) | `96` | Target arguments, HLO module with config, core program ABI, and predetermined error checks |
| 9 | `PJRT_TpuTopology_Extension` | `16` (`0x10`) | `272` | 32 entrypoints for 3D torus coordinates, chip/host bounds, subslices, and routing strategies |
| 10 | `PJRT_Callback_Extension` | `14` (`0x0e`) | `40` | Runtime event callback registration & invocation (`RegisterCallback`, `InvokeCallback`) |
| 11 | `PJRT_PhaseCompile_Extension` | `9` (`0x09`) | `64` | Phased compilation (`GetCompiler`, `RunPhase`, `GetPhaseNames`, `CBuffersDestroy`) |
| 12 | `PJRT_CrossHostTransfers_Extension` | `12` (`0x0c`) | `56` | Cross-host DCN buffer send/receive (`MakeCrossHostReceiveBuffers`, `CopyToRemoteDevice`) |
| 13 | `PJRT_ExecutableMetadata_Extension` | `11` (`0x0b`) | `40` | Post-compilation metadata retrieval & destruction |
| 14 | `PJRT_MemoryDescriptions_Extension` | `7` (`0x07`) | `40` | Device memory space kind & capacity description queries |
| 15 | `PJRT_Layouts_Extension` | `6` (`0x06`) | `80` | Default TPU `(8, 128)` tiled layout queries & serialization (`MemoryLayout_Destroy`, `Serialize`) |
| 16 | `PJRT_RawBuffer_Extension` | `8` (`0x08`) | `80` | Zero-copy raw buffer view creation, device pointers, and host-to-device raw copies |
| 17 | `PJRT_Profiler_Extension` | `1` (`0x01`) | `40` | Embeds `PLUGIN_Profiler_Api` (`72` B, 8 slots) producing wire-compatible `XSpace` traces |

### 8.3 `LibtpuSdk_Api` v0.1 Vtable (`144` Bytes, 17 Slots)

Returned by `GetLibtpuSdkApi()`, the static `LIBTPU_SDK_API_VTBL` structure in `crates/libtpu-c-api/src/tpu_driver_types.rs` has an exact size of **`144` bytes (`0x90`)**:
- Offset `0x00` (`8` B): `version_header = 0x0000_0001_0000_0000` (`major = 0`, `minor = 1`)
- Offsets `0x08..=0x88` (`17 * 8 = 136` B): 17 function pointers (`LibtpuSdk_Error_GetMessage` through `LibtpuSdk_UnregisterDebugCallback`).

### 8.4 Legacy `TfTpu_*`, `TpuDriver_*`, `TpuProfiler_*` & `TpuEmbeddingEngine_*` C Symbols

`crates/libtpu-c-api/src/lib.rs` also exports the complete legacy TensorFlow and C driver surface:
- **Initialization & Lifecycle**: `TfTpu_Initialize`, `SetGlobalTPUArrayOp_DoWork`, `InitializeCreateMeshGlobalState`, `ConfigureDistributedTpuOp_DoWork`, `WaitForDistributedTpuOp_DoWork`, `ShutdownDistributedTpuOp_DoWork`, `DisconnectDistributedTpuChipsOp_DoWork`, `TFNPD_InitPlugin`, `TpuNodeContext_*`, `TpuResourceManager_*`.
- **Topology & Mesh State**: `TpuTopology_AvailableCoreCount`, `TpuTopology_ChipBounds_*`, `TpuTopology_HasChip`, `TpuTopology_Core`, `TpuCoreLocation_*`, `TpuHostLocation_*`, `TpuMeshState_*`.
- **Direct C Driver (`TpuDriver_*`)**: `TpuDriver_Initialize` plus all 28 vtable operations in `TpuDriverFn` (`224` bytes).
- **Profiler (`TpuProfiler_*`)**: `TpuProfiler_Create`, `TpuProfiler_Destroy`, `TpuProfiler_Start`, `TpuProfiler_Stop`, `TpuProfiler_CollectData`.
- **SparseCore Embedding Engine (`TpuEmbeddingEngine_*`)**: All 15 lifecycle, parameter, batch enqueue, and XLA computation generation symbols.

### 8.5 Ergonomic `libtpu_rs_*` C Entrypoints

For C11 and C++20 applications that prefer direct function calls over manually constructing `PJRT_*_Args` structs, `libtpu-c-api` exports 18 clean `libtpu_rs_*` functions (`libtpu_rs_status_free`, `libtpu_rs_client_create`, `libtpu_rs_client_destroy`, `libtpu_rs_client_device_count`, `libtpu_rs_buffer_from_host`, `libtpu_rs_buffer_to_host`, `libtpu_rs_buffer_dimensions`, `libtpu_rs_buffer_element_type`, `libtpu_rs_buffer_copy_to_device`, `libtpu_rs_buffer_destroy`, `libtpu_rs_compile`, `libtpu_rs_executable_serialize`, `libtpu_rs_executable_deserialize`, `libtpu_rs_executable_fingerprint`, `libtpu_rs_executable_flops`, `libtpu_rs_executable_destroy`, `libtpu_rs_execute`, `libtpu_rs_execute_multi`).

---

## 9. Layer 6: C11 Header & C++20 RAII SDK (`cpp/`)

Following CMake best practices, all C/C++ headers and sources reside in `cpp/src/` with a dedicated `cpp/src/CMakeLists.txt` and root `cpp/CMakeLists.txt`.

### 9.1 C11 Header (`cpp/src/libtpu.h`)

Provides a self-contained, C11- and C++20-compatible header declaring all opaque PJRT/TPU handles (`PJRT_Client`, `PJRT_Buffer`, `PJRT_LoadedExecutable`, `TpuProfiler`, `TpuStatus`, `TpuTopology`, `LibtpuSdk_Client`, etc.), `PJRT_Api` and `LibtpuSdk_Api` vtables, and all `libtpu_rs_*` C functions.

### 9.2 Zero-Exception C++20 RAII Wrappers (`cpp/src/libtpu.hpp` & `libtpu.cpp`)

The `libtpu` C++20 namespace provides move-only, copy-deleted, `noexcept`, `[[nodiscard]]` RAII classes that deterministically manage underlying C handles without C++ exceptions:

- **`libtpu::Status`**: Owns a `LibTpuCStatus` and frees its message buffer in `~Status() noexcept` via `libtpu_rs_status_free`.
- **`libtpu::Client`**: Owns a `PJRT_Client*`, providing `Client::create()`, `device_count()`, `buffer_from_host()`, `compile()`, and `deserialize_executable()`.
- **`libtpu::Buffer`**: Owns a `PJRT_Buffer*`, providing `to_host()`, `rank()`, `dimension()`, `element_type()`, and `copy_to_device()`.
- **`libtpu::LoadedExecutable`**: Owns a `PJRT_LoadedExecutable*`, providing `execute_binary()`, `execute()` (multi-input/multi-output), `serialize()`, `fingerprint()`, and `flops()`.
- **`libtpu::Profiler`**: Owns a `TpuProfiler*` and `TpuStatus*`, providing `start()`, `stop()`, and `collect_size()`.
- **`libtpu::Topology`**: Wraps the global `TpuTopology*` singleton, exposing `logical_devices_per_host()`, `chip_bounds_x/y/z()`, and `version()`.
- **`libtpu::SdkClient`**: Owns a `LibtpuSdk_Client*` created via `GetLibtpuSdkApi()`, exposing `core_state_summary_size()`, `metric_value_count()`, and `chip_count()`.

---

## 10. Layer 7: Python Wheel Package & SDK (`libtpu-py` & `python/libtpu/`)

### 10.1 Rust Python Bridge (`crates/libtpu-py`)

Provides `PyTpuRuntime`, a Rust session wrapper over `TpuInitConfig` and `PjrtCoreClient` for high-level Rust-to-Python embedding scenarios.

### 10.2 Top-Level `libtpu` Python Module (`python/libtpu/__init__.py`)

Replicates the official Google `libtpu` wheel API while adding direct `ctypes` bindings over `GetPjrtApi()` and `TpuProfiler_*()`:
- **Drop-In Environment & Library Discovery**:
  - `get_library_path() -> str`: Resolves the compiled `libtpu.dylib` / `libtpu.so` / `tpu.dll`.
  - `configure_library_path() -> str`: Populates `os.environ["TPU_LIBRARY_PATH"]` for JAX and XLA plugin discovery.
  - `set_init_args(args)`, `get_init_args() -> list[str]`, `initialize(args) -> str`: Manage `LIBTPU_INIT_ARGS`.
- **High-Level Python Classes**:
  - `TpuClient`: Drives `PJRT_Client_Create`, `PJRT_Client_AddressableDevices`, `PJRT_Client_BufferFromHostBuffer`, `PJRT_Buffer_CopyToDevice`, `PJRT_Client_Compile`, and `PJRT_Executable_DeserializeAndLoad`.
  - `TpuBuffer`: Wraps `PJRT_Buffer*` with `shape()`, `dtype()`, `to_bytes()`, `to_float_list()`, `to_int_list()`, and `close()`.
  - `TpuExecutable`: Wraps `PJRT_LoadedExecutable*` with `execute()`, `execute_multi()`, `serialize()`, `fingerprint()`, `flops()`, and `close()`.
  - `TpuProfiler`: Wraps `TpuProfiler*` with `start()`, `stop()`, `collect_data()`, and `close()`.

### 10.3 `libtpu.sdk` Subpackage (`tpumonitoring`, `slice`, `tpuz`, `debug`)

Matches the official `libtpu.sdk` Python namespace, driving `GetLibtpuSdkApi()` via `ctypes`:
- **`libtpu.sdk.tpumonitoring`**: `list_supported_metrics()`, `get_metric_description(name)`, `get_metric(name) -> Metric` (with `.description()` and `.data()`).
- **`libtpu.sdk.slice`**: `ChipCoordinates` dataclass, `get_chip_coordinates()`, `get_host_name()`, `get_chip_index()`, `get_cartesian_coordinates()`.
- **`libtpu.sdk.tpuz`**: `RuntimeStatus` class (`get_core_state_summary()`), `get_runtime_status(include_hlo_info=...)`, `get_core_state_summary()`.
- **`libtpu.sdk.debug`**: `register_debug_callback(fn) -> int`, `unregister_debug_callback(id)`, `_trigger_debug_callback_for_test(status_code, msg)`.

---

## 11. End-to-End Data & Control Flows

### 11.1 Client Initialization & Topology Discovery

```text
Caller (JAX / C++ / Python)
  │
  ├── 1. (Optional) Sets LIBTPU_INIT_ARGS / TPU_ACCELERATOR_TYPE / TfTpu_Initialize()
  │      └── Updates global `GLOBAL_INIT_CONFIG` (`TpuInitConfig`)
  │
  ├── 2. Calls `GetPjrtApi()` -> Returns static `&PJRT_API_VTBL` (1,144 B)
  │
  └── 3. Calls `PJRT_Client_Create(&args)`
         ├── Snapshots `global_init_config()`
         ├── Instantiates `SimulatedTpuHal::new(&cfg)` (or `LinuxDeviceHal`)
         │   └── Synthesizes 3D torus `TopologySpec` (`host_bounds` × `chips_per_host_bounds`)
         ├── Constructs `PjrtCoreClient` with `PjrtCoreDevice`s & `PjrtCoreMemory` spaces
         └── Allocates `#[repr(C)]` `PJRT_Client` handle and returns `nullptr` (`PJRT_Error*` = OK)
```

### 11.2 Program Compilation (`StableHLO` / `HloModuleProto` -> `LoadedExecutable`)

```text
Caller invokes `PJRT_Client_Compile(&args)`
  │
  ├── 1. Validates non-null `client` and `program` pointers
  ├── 2. Decodes `CompileOptionsProto` (extracting `num_replicas` & `num_partitions`)
  ├── 3. Parses program payload:
  │      ├── Binary `HloModuleProto` / `HloModuleProtoWithConfig` -> `lower_from_hlo_proto()`
  │      └── Textual `StableHLO` MLIR / XLA HLO -> `HloLexer` -> `parse_textual_program()`
  ├── 4. Emits SSA DAG of `TpuInstructionBundle`s with `InstructionAttrs`
  ├── 5. Runs 6-pass optimizer (`run_all_passes()`):
  │      ├── Verification & shape inference
  │      ├── SPMD sharding propagation
  │      ├── TPU `(8, 128)` matrix sublane tiling layout assignment
  │      ├── Producer-consumer operator fusion
  │      ├── VMEM scratchpad allocation
  │      └── VLIW bundle scheduling & `SyncFlag` assignment
  ├── 6. Computes `CostAnalysis`, `CompiledMemoryStats`, and 128-bit FNV-1a `fingerprint`
  └── 7. Wraps in `PjrtCoreLoadedExecutable` and returns `PJRT_LoadedExecutable*`
```

### 11.3 Host-to-Device DMA, VLIW Execution & Device-to-Host Readback

```text
1. Host-to-Device DMA (`PJRT_Client_BufferFromHostBuffer`):
   Host Bytes ──► `Shape::new()` validation
              ──► `hal.allocate_device_memory()` (4 KiB page-aligned HBM @ `0x1000_0000+`)
              ──► `hal.write_device_memory()` (copies bytes & increments `BAR0_CSR_HBM_DMA_RING_TAIL/HEAD`)
              ──► Returns `PJRT_Buffer*`

2. On-Device Execution (`PJRT_LoadedExecutable_Execute`):
   Input `PJRT_Buffer*`s ──► Validates neither executable nor buffers are deleted
                         ──► Allocates output HBM buffers for each shape in `program.output_shapes`
                         ──► `hal.execute_program()`:
                             ├── Checks device is not poisoned
                             ├── Loads input HBM buffers into virtual VLIW register slots
                             ├── Evaluates each `TpuInstructionBundle` in topological order
                             ├── Writes result registers to output HBM allocations
                             └── Updates `BAR0_CSR_TENSORCORE_PC`, `RUN_CONTROL`, `STATUS`, and `SYNC_FLAG_BASE`
                         ──► Returns output `PJRT_Buffer*` array and ready `PJRT_Event*`

3. Device-to-Host Readback (`PJRT_Buffer_ToHostBuffer`):
   `PJRT_Buffer*` ──► `hal.read_device_memory()` (copies HBM bytes to caller buffer & updates DMA ring CSRs)
                  ──► Returns ready `PJRT_Event*`
```

---

## 12. Concurrency, Memory Safety & Lifetime Model

1. **Thread Safety (`Send + Sync`)**:
   - All core runtime structs (`PjrtCoreClient`, `PjrtCoreBuffer`, `PjrtCoreEvent`, `SimulatedTpuHal`, `LinuxDeviceHal`, `EmulatedKernelDriverBackend`, `TpuCoreProfiler`, `LibtpuSdkCoreClient`) are `Send + Sync`.
   - Fast-path state flags (`deleted`, `ready`, `active`, `next_ordinal`, `NEXT_OBJECT_ID`, `EXTENSION_CALL_COUNTER`) use lock-free `std::sync::atomic` primitives (`AtomicBool`, `AtomicU32`, `AtomicI64`, `AtomicU64`) with explicit `Acquire`/`Release`/`AcqRel` memory orderings.
   - Mutable collections (device HBM maps, BAR0 register maps, IOMMU mappings, telemetry metrics, and debug callback registries) are protected by `std::sync::Mutex` with poison recovery (`unwrap_or_else(PoisonError::into_inner)`) or explicit `LibTpuError::LockPoisoned` propagation.
2. **FFI Ownership & Lifetime Rules**:
   - Every heap-allocated C handle (`PJRT_Client`, `PJRT_Buffer`, `PJRT_Executable`, `PJRT_LoadedExecutable`, `PJRT_Event`, `PJRT_Error`, `TpuStatus`, `TpuDriver`, `TpuProfiler`, `LibtpuSdk_Client`, `LibtpuSdk_Error`) is created via `Box::into_raw(Box::new(...))` and reclaimed deterministically in its corresponding `*_Destroy` or `*_Free` entrypoint via `drop(Box::from_raw(...))`.
   - `PJRT_Client` owns its child `PJRT_Device`, `PJRT_DeviceDescription`, `PJRT_Memory`, and `PJRT_TopologyDescription` pointers for the lifetime of the client, matching XLA's PJRT ownership contract where callers read device/memory pointers without freeing them individually.
   - String views returned across the C ABI (such as `PJRT_Error_Message`, `PJRT_Client_PlatformName`, `LibtpuSdk_GetMetricDescription`) point into `CString` buffers cached inside the owning handle struct, ensuring the pointer remains valid for the handle's entire lifetime without leaking memory.

---

## 13. Quality Standards & Verification Pipeline (`scripts/pre_commit.py`)

The repository enforces strict quality gates via `scripts/pre_commit.py`:

1. **Formatting (`fmt`)**:
   - Rust: `cargo fmt --all -- --check`
   - Python: `ruff format --check python/ scripts/`
   - C/C++: `clang-format --dry-run --Werror --style=LLVM`
2. **Compilation (`build`)**:
   - Rust: `cargo build --workspace --all-targets`
   - C/C++: `cmake -B cpp/build -S cpp -DENABLE_COVERAGE=ON && cmake --build cpp/build`
3. **Linting & Static Analysis (`clippy` / `lint`)**:
   - Rust Clippy: `cargo clippy --workspace --all-targets --all-features -- -D warnings` (enforcing `pedantic`, `nursery`, `unwrap_used = "deny"`, `expect_used = "deny"`, `missing_docs = "forbid"`, `missing_safety_doc = "forbid"`, `undocumented_unsafe_blocks = "forbid"`).
   - Rustdoc: `RUSTDOCFLAGS="-D warnings -D missing_docs -D rustdoc::broken_intra_doc_links" cargo doc --workspace --no-deps --document-private-items`.
   - Python: `ruff check python/ scripts/` and `mypy --strict python/ scripts/`.
   - C/C++: `doxygen cpp/Doxyfile` (`WARN_AS_ERROR = YES`).
4. **Binary ABI & Source Invariant Verification (`abi`)**:
   - Verifies all 79 official `libtpu.so` / `sdk.so` dynamic symbols, `#[repr(C)]` struct sizes, non-null `PJRT_Api` (1,144 B, 138 slots), 17-node `PJRT_Extension_Base` linked list, `LibtpuSdk_Api` (144 B, 17 slots), `TpuDriverFn` (224 B, 28 slots), and zero `.unwrap()` / `.expect()` / `anyhow` / `thiserror`.
5. **Testing (`test`)**:
   - Rust: `cargo test --workspace --all-targets`
   - Python: `pytest --cov=libtpu --cov=scripts --cov-branch --cov-fail-under=100`
   - C/C++: `ctest --test-dir cpp/build --output-on-failure`
6. **Coverage Verification & Shield Updates (`shields`)**:
   - Computes aggregate documentation coverage across Rust (`rustdoc --show-coverage`), Python (`ast` docstring inspection), and C/C++ (Doxygen XML) -> **100%**.
   - Computes aggregate test coverage across Rust (`cargo +nightly llvm-cov --workspace --branch`), Python (`pytest-cov --cov-branch`), and C/C++ (`llvm-cov export`) -> **100%**.

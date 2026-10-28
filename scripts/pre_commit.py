#!/usr/bin/env python3
"""Cross-platform Pre-commit Hook Pipeline for libtpu-rs.

Performs:
1. Code formatting auto-fix and recheck across Rust (`cargo fmt`), Python (`ruff format`),
   and C/C++ (`clang-format`).
2. Compilation and build check across Rust (`cargo build --workspace --all-targets`)
   and C/C++ (`cmake`).
3. Strict linting, static analysis, and doc generation across Rust (`cargo clippy`, `cargo doc`),
   Python (`ruff check`, `mypy --strict`), and C/C++ (`doxygen`).
4. Unit and integration test suite execution across Rust (`cargo test`), Python (`pytest`),
   and C/C++ (`ctest`).
5. Official Google `libtpu` (`libtpu.so` v0.0.50) binary ABI & Python wheel API compatibility
   verification (`PJRT_Api` v0.116 1,144-byte vtable, `LibtpuSdk_Api` v0.1 144-byte vtable,
   17-node PJRT extension linked list, 79 exported dynamic C symbols, `#[repr(C)]` struct sizes).
6. Doc & test coverage calculation and `% doc coverage` / `% test coverage` shield updates
   in `README.md`.
"""

from __future__ import annotations

import argparse
import ast
import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

GOOGLE_LIBTPU_EXPORTED_SYMBOLS: tuple[str, ...] = (
    "GetPjrtApi",
    "GetLibtpuSdkApi",
    "PyInit_sdk",
    "TfTpu_Initialize",
    "TF_InitKernel",
    "TFNPD_InitPlugin",
    "SetGlobalTPUArrayOp_DoWork",
    "TfTpu_GetTpuPartitionedCallParams",
    "TpuConfigurationApi_FreeCharArray",
    "TpuTransferManager_GetInfeedLayout",
    "SparseCore_GetMaxIdsAndUniques",
    "TfTpuOrdinalSelector_Create",
    "TfTpuOrdinalSelector_Destroy",
    "TfTpuOrdinalSelector_GetOrdinal",
    "TfTpuOrdinalSelector_DequeueFromCoreSelector",
    "TpuStatus_New",
    "TpuStatus_Create",
    "TpuStatus_Set",
    "TpuStatus_Free",
    "TpuStatus_Message",
    "TpuStatus_Code",
    "TpuStatus_Ok",
    "TpuUtil_GetTopologyPtr",
    "TpuUtil_GetXlaPadSizeFromTpuTopology",
    "TpuTopology_LogicalDevicesPerHost",
    "TpuTopology_LogicalDevicesPerChip",
    "TpuTopology_HostCount",
    "TpuTopology_ChipsPerHost",
    "TpuTopology_ChipBounds_X",
    "TpuTopology_ChipBounds_Y",
    "TpuTopology_ChipBounds_Z",
    "TpuTopology_HasChip",
    "TpuTopology_CoreForId",
    "TpuTopology_Core",
    "TpuTopology_NumCores",
    "TpuTopology_Cores",
    "TpuTopology_IdForHost",
    "TpuTopology_Version",
    "TpuTopology_AvailableCoreCount",
    "TpuTopology_AvailableCoresPerChip",
    "TpuTopology_MaybeAvailableSparseCoresPerLogicalDevice",
    "TpuCoreLocation_ChipCoordinates",
    "TpuCoreLocation_HostCoordinates",
    "TpuCoreLocation_Index",
    "TpuCoreLocation_Id",
    "TpuHostLocation_Id",
    "TpuHostLocation_NumCores",
    "TpuHostLocation_Cores",
    "TpuMeshState_Create",
    "TpuMeshState_Free",
    "TpuMeshState_MeshCommonState",
    "TpuProfiler_Create",
    "TpuProfiler_Destroy",
    "TpuProfiler_Start",
    "TpuProfiler_Stop",
    "TpuProfiler_CollectData",
    "TpuEmbeddingEngineState_Create",
    "TpuEmbeddingEngineState_Free",
    "TpuEmbeddingEngineState_GetState",
    "TpuEmbeddingEngine_ExecutePartitioner",
    "TpuEmbeddingEngine_ConfigureMemory",
    "TpuEmbeddingEngine_CollateMemory",
    "TpuEmbeddingEngine_ConfigureHost",
    "TpuEmbeddingEngine_ConnectHosts",
    "TpuEmbeddingEngine_Finalize",
    "TpuEmbeddingEngine_IsInitialized",
    "TpuEmbeddingEngine_WriteParameters",
    "TpuEmbeddingEngine_ReadParameters",
    "TpuEmbeddingTensorBatchFixedState_Create",
    "TpuEmbeddingTensorBatchFixedState_Destroy",
    "TpuEmbeddingEngine_EnqueueTensorBatch",
    "TpuEmbeddingEngine_RecvActivationsComputation",
    "TpuEmbeddingEngine_RecvTPUEmbeddingDeduplicationDataComputation",
    "TpuEmbeddingEngine_SendTPUEmbeddingGradientsComputation",
    "TpuEmbeddingEngine_DedupDataSizeComputation",
    "TpuEmbeddingEngine_DedupDataTupleMaskComputation",
    "TpuDriver_Initialize",
    "TpuDriver_Open",
    "TpuDriver_Close",
)

EXPECTED_PJRT_EXTENSION_CHAIN: tuple[tuple[int, int, int], ...] = (
    (29, 56, 4),
    (28, 32, 1),
    (26, 64, 5),
    (25, 96, 9),
    (20, 120, 12),
    (19, 40, 2),
    (18, 248, 28),
    (17, 96, 9),
    (16, 272, 31),
    (14, 40, 2),
    (9, 64, 5),
    (12, 56, 4),
    (11, 40, 2),
    (7, 40, 2),
    (6, 80, 7),
    (8, 80, 7),
    (1, 40, 2),
)

EXPECTED_PYTHON_WHEEL_SYMBOLS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "python/libtpu/__init__.py",
        ("get_library_path", "configure_library_path", "set_init_args", "get_init_args"),
    ),
    (
        "python/libtpu/sdk/tpumonitoring.py",
        (
            "ChipCoordinates",
            "LibtpuSdkMetric",
            "SdkClient",
            "list_supported_metrics",
            "get_metric",
            "get_chip_coordinates",
            "get_core_state_summary",
        ),
    ),
    ("python/libtpu/sdk/tpuz.py", ("get_core_state_summary",)),
    ("python/libtpu/sdk/slice.py", ("ChipCoordinates", "get_chip_coordinates")),
    (
        "python/libtpu/sdk/debug.py",
        (
            "register_debug_callback",
            "unregister_debug_callback",
            "_trigger_debug_callback_for_test",
        ),
    ),
)

EXPECTED_REPR_C_STRUCT_SIZES: tuple[tuple[str, int], ...] = (
    ("PJRT_Api", 1144),
    ("PJRT_Api_Version", 24),
    ("LibtpuSdk_Api", 144),
    ("TpuDriverFn", 224),
    ("TFNPD_PluginParams", 40),
    ("TpuPartitionedCall_Params", 24),
    ("XLA_Shape", 96),
    ("TpuSerializedProto", 16),
    ("TpuEmbeddingEngine_ExecutePartitioner_Params", 56),
    ("TpuEmbeddingEngine_ConfigureMemory_Params", 64),
    ("TpuEmbeddingEngine_CollateMemory_Params", 56),
    ("TpuEmbeddingEngine_ConfigureHost_Params", 96),
    ("TpuEmbeddingEngine_ConnectHosts_Params", 40),
    ("TpuEmbeddingEngine_Finalize_Params", 64),
    ("TpuEmbeddingEngine_IsInitialized_Params", 48),
    ("TpuEmbeddingEngine_WriteParameters_Params", 72),
    ("TpuEmbeddingEngine_EnqueueTensorBatch_Params", 56),
    ("TpuEmbeddingEngine_Computation_Params", 96),
    ("PJRT_XlaTransform_Extension", 56),
    ("PJRT_HostMemoryAllocator_Extension", 32),
    ("PJRT_MultiSlice_Extension", 64),
    ("PJRT_Collectives_Extension", 96),
    ("PJRT_AbiVersion_Extension", 120),
    ("PJRT_Shardings_Extension", 40),
    ("PJRT_Megascale_Extension", 248),
    ("PJRT_TpuExecutable_Extension", 96),
    ("PJRT_TpuTopology_Extension", 272),
    ("PJRT_Callback_Extension", 40),
    ("PJRT_PhaseCompile_Extension", 64),
    ("PJRT_CrossHostTransfers_Extension", 56),
    ("PJRT_ExecutableMetadata_Extension", 40),
    ("PJRT_MemoryDescriptions_Extension", 40),
    ("PJRT_Layouts_Extension", 80),
    ("PJRT_RawBuffer_Extension", 80),
    ("PJRT_Profiler_Extension", 40),
)


def get_repo_root() -> Path:
    """Returns the workspace root directory.

    Returns:
        Path to the repository root directory.
    """
    return Path(__file__).resolve().parent.parent


def get_augmented_env() -> dict[str, str]:
    """Returns `os.environ` augmented with standard cargo and toolchain paths.

    Returns:
        Dictionary of environment variables with augmented `PATH`.
    """
    env = dict(os.environ)
    for git_var in ("GIT_INDEX_FILE", "GIT_DIR", "GIT_WORK_TREE"):
        env.pop(git_var, None)

    cargo_bin = Path.home() / ".cargo" / "bin"
    current_path = env.get("PATH", "")
    paths = [p for p in current_path.split(os.pathsep) if p]

    candidate_dirs = [
        cargo_bin,
        Path("/opt/homebrew/bin"),
        Path("/usr/local/bin"),
        Path("/Library/Frameworks/Python.framework/Versions/3.9/bin"),
    ]
    for candidate in candidate_dirs:
        if candidate.exists() and str(candidate) not in paths:
            paths.insert(0, str(candidate))

    env["PATH"] = os.pathsep.join(paths)
    return env


def run_command(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str] | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Runs a shell command cross-platform and raises on non-zero exit status.

    Args:
        cmd: Command and arguments list to execute.
        cwd: Working directory for the subprocess.
        env: Optional environment dictionary override.
        capture_output: Whether to capture stdout and stderr.

    Returns:
        CompletedProcess instance with text output.
    """
    effective_env = env if env is not None else get_augmented_env()
    executable = cmd[0]
    resolved = shutil.which(executable, path=effective_env.get("PATH"))
    resolved_cmd = [resolved, *cmd[1:]] if resolved else list(cmd)

    return subprocess.run(
        resolved_cmd,
        cwd=cwd,
        env=effective_env,
        check=True,
        capture_output=capture_output,
        text=True,
    )


def get_cpp_files(repo_root: Path) -> list[str]:
    """Collects all C and C++ source/header files in `cpp/`.

    Args:
        repo_root: Path to the workspace root directory.

    Returns:
        Sorted list of relative file paths for C/C++ files.
    """
    cpp_files: list[str] = []
    for pattern in ("cpp/src/*.h", "cpp/src/*.hpp", "cpp/src/*.cpp", "cpp/tests/*.cpp"):
        for path in sorted(repo_root.glob(pattern)):
            cpp_files.append(str(path.relative_to(repo_root)))
    return cpp_files


def get_python_targets(repo_root: Path) -> list[str]:
    """Returns Python directories/files to format and lint.

    Args:
        repo_root: Path to the workspace root directory.

    Returns:
        List of relative directory/file paths for Python tooling.
    """
    targets = ["python/"]
    scripts_dir = repo_root / "scripts"
    if scripts_dir.exists():
        targets.append("scripts/")
    return targets


def hook_cargo_fmt(repo_root: Path, check_only: bool = False) -> int:
    """Runs `cargo fmt` auto-formatting (unless `check_only` is set) and rechecks Rust files.

    Args:
        repo_root: Path to the workspace root directory.
        check_only: If True, skip auto-formatting and only run the verification check.

    Returns:
        Zero on success.
    """
    if not check_only:
        print("==> [fmt] Auto-fixing Rust formatting with cargo fmt...")
        run_command(["cargo", "fmt", "--all"], cwd=repo_root)

    print("==> [fmt] Rechecking Rust formatting (cargo fmt --all -- --check)...")
    run_command(["cargo", "fmt", "--all", "--", "--check"], cwd=repo_root)
    return 0


def hook_ruff_format(repo_root: Path, check_only: bool = False) -> int:
    """Runs `ruff format` auto-formatting (unless `check_only` is set) and rechecks Python files.

    Args:
        repo_root: Path to the workspace root directory.
        check_only: If True, skip auto-formatting and only run the verification check.

    Returns:
        Zero on success.
    """
    py_targets = get_python_targets(repo_root)
    if not check_only:
        print("==> [fmt] Auto-fixing Python formatting with ruff format...")
        run_command(["ruff", "format", *py_targets], cwd=repo_root)

    print("==> [fmt] Rechecking Python formatting (ruff format --check)...")
    run_command(["ruff", "format", "--check", *py_targets], cwd=repo_root)
    return 0


def hook_clang_format(repo_root: Path, check_only: bool = False) -> int:
    """Runs `clang-format --style=LLVM` auto-formatting and rechecks C/C++ files.

    Args:
        repo_root: Path to the workspace root directory.
        check_only: If True, skip auto-formatting and only run the verification check.

    Returns:
        Zero on success.
    """
    cpp_files = get_cpp_files(repo_root)
    if not cpp_files:
        return 0

    if not check_only:
        print("==> [fmt] Auto-fixing C/C++ formatting with clang-format --style=LLVM...")
        run_command(
            ["clang-format", "-i", "--style=LLVM", *cpp_files],
            cwd=repo_root,
        )

    print("==> [fmt] Rechecking C/C++ formatting (clang-format --dry-run --Werror --style=LLVM)...")
    run_command(
        ["clang-format", "--dry-run", "--Werror", "--style=LLVM", *cpp_files],
        cwd=repo_root,
    )
    return 0


def hook_fmt(repo_root: Path, check_only: bool = False) -> int:
    """First runs auto-formatting fixes (unless `check_only` is set), then rechecks formatting.

    Args:
        repo_root: Path to the workspace root directory.
        check_only: If True, skip auto-formatting and only run the verification checks.

    Returns:
        Zero on success.
    """
    hook_cargo_fmt(repo_root, check_only=check_only)
    hook_ruff_format(repo_root, check_only=check_only)
    hook_clang_format(repo_root, check_only=check_only)

    print("==> [fmt] Code formatting verified successfully.")
    return 0


def hook_build(repo_root: Path, check_only: bool = False) -> int:
    """Builds all Rust workspace targets and C/C++ CMake targets.

    Args:
        repo_root: Path to the workspace root directory.
        check_only: Unused flag kept for uniform stage handler signature.

    Returns:
        Zero on success.
    """
    _ = check_only
    print("==> [build] Compiling Rust workspace packages and targets (cargo build)...")
    run_command(["cargo", "build", "--workspace", "--all-targets"], cwd=repo_root)

    print("==> [build] Configuring and building C/C++ targets (cmake)...")
    run_command(
        ["cmake", "-B", "cpp/build", "-S", "cpp", "-DENABLE_COVERAGE=ON"],
        cwd=repo_root,
    )
    run_command(["cmake", "--build", "cpp/build"], cwd=repo_root)

    print("==> [build] Workspace compiled successfully.")
    return 0


def hook_clippy(repo_root: Path, check_only: bool = False) -> int:
    """Runs strict linters across Rust, Python, and C/C++.

    Args:
        repo_root: Path to the workspace root directory.
        check_only: Unused flag kept for uniform stage handler signature.

    Returns:
        Zero on success.
    """
    _ = check_only
    py_targets = get_python_targets(repo_root)

    print("==> [lint] Running cargo clippy with strictest workspace lints...")
    run_command(
        [
            "cargo",
            "clippy",
            "--workspace",
            "--all-targets",
            "--all-features",
            "--",
            "-D",
            "warnings",
        ],
        cwd=repo_root,
    )

    print("==> [lint] Verifying rustdoc lints...")
    env = get_augmented_env()
    env["RUSTDOCFLAGS"] = "-D warnings -D missing_docs -D rustdoc::broken_intra_doc_links"
    run_command(
        ["cargo", "doc", "--workspace", "--no-deps", "--document-private-items"],
        cwd=repo_root,
        env=env,
    )

    print("==> [lint] Running ruff check on Python files...")
    run_command(["ruff", "check", *py_targets], cwd=repo_root)

    print("==> [lint] Running mypy --strict on Python files...")
    run_command(["mypy", "--strict", *py_targets], cwd=repo_root)

    print("==> [lint] Running Doxygen strict documentation check on C/C++ files...")
    (repo_root / "cpp" / "build" / "doxygen").mkdir(parents=True, exist_ok=True)
    run_command(["doxygen", "cpp/Doxyfile"], cwd=repo_root)

    print("==> [lint] All lint and documentation checks passed with zero warnings.")
    return 0


def hook_test(repo_root: Path, check_only: bool = False) -> int:
    """Runs all Rust, Python, and C/C++ test suites.

    Args:
        repo_root: Path to the workspace root directory.
        check_only: Unused flag kept for uniform stage handler signature.

    Returns:
        Zero on success.
    """
    _ = check_only
    print("==> [test] Running Rust test suite (cargo test)...")
    run_command(["cargo", "test", "--workspace", "--all-targets"], cwd=repo_root)

    print("==> [test] Running Python test suite (pytest)...")
    run_command(
        [
            "pytest",
            "--cov=libtpu",
            "--cov=scripts",
            "--cov-branch",
            "--cov-report=term-missing",
            "--cov-fail-under=100",
        ],
        cwd=repo_root,
    )

    print("==> [test] Running C/C++ test suite (ctest)...")
    build_dir = repo_root / "cpp" / "build"
    if not (build_dir / "CMakeCache.txt").exists():
        run_command(
            ["cmake", "-B", "cpp/build", "-S", "cpp", "-DENABLE_COVERAGE=ON"],
            cwd=repo_root,
        )
        run_command(["cmake", "--build", "cpp/build"], cwd=repo_root)

    env = get_augmented_env()
    env["LLVM_PROFILE_FILE"] = str(repo_root / "cpp" / "build" / "test_libtpu.profraw")
    run_command(
        ["ctest", "--test-dir", "cpp/build", "--output-on-failure"],
        cwd=repo_root,
        env=env,
    )

    print("==> [test] All Rust, Python, and C/C++ tests passed.")
    return 0


def get_badge_color(pct: float) -> str:
    """Maps a coverage percentage threshold to a standard shields.io badge color.

    Args:
        pct: Coverage percentage (0.0 to 100.0).

    Returns:
        Shields.io color string.
    """
    if pct >= 95.0:
        return "brightgreen"
    if pct >= 90.0:
        return "green"
    if pct >= 80.0:
        return "yellowgreen"
    if pct >= 70.0:
        return "yellow"
    if pct >= 60.0:
        return "orange"
    return "red"


def format_pct(pct: float) -> str:
    """Formats a percentage value cleanly for badge display.

    Args:
        pct: Coverage percentage (0.0 to 100.0).

    Returns:
        Formatted percentage string ending with `%`.
    """
    if round(pct, 1) >= 100.0 or pct.is_integer():
        return f"{int(round(pct))}%"
    return f"{pct:.1f}%"


def _has_xml_text_or_children(elem: ET.Element | None) -> bool:
    """Checks whether an XML element has non-empty text or child elements.

    Args:
        elem: Optional XML element to inspect.

    Returns:
        True if the element has text or children, False otherwise.
    """
    if elem is None:
        return False
    if elem.text and elem.text.strip():
        return True
    return len(list(elem)) > 0


def calculate_rust_doc_coverage(repo_root: Path, force_run: bool = True) -> tuple[int, int]:
    """Calculates documented vs total items across all Rust workspace crates via rustdoc JSON.

    Args:
        repo_root: Path to the workspace root directory.
        force_run: Whether to regenerate rustdoc JSON files.

    Returns:
        Tuple of `(with_docs, total_items)`.
    """
    doc_dir = repo_root / "target" / "doc"
    if force_run and doc_dir.exists():
        for old_json in doc_dir.glob("*.json"):
            old_json.unlink(missing_ok=True)

    doc_json_files = sorted(doc_dir.glob("*.json")) if doc_dir.exists() else []
    if force_run or not doc_json_files:
        print("==> [shields] Generating rustdoc JSON documentation coverage...")
        env = get_augmented_env()
        env["RUSTDOCFLAGS"] = "-Z unstable-options --show-coverage --output-format json"
        run_command(
            ["cargo", "+nightly", "doc", "--workspace", "--no-deps"],
            cwd=repo_root,
            env=env,
        )
        doc_json_files = sorted(doc_dir.glob("*.json")) if doc_dir.exists() else []

    total_items = 0
    with_docs = 0
    for json_file in doc_json_files:
        data = json.loads(json_file.read_text(encoding="utf-8"))
        for _file_path, stats in data.items():
            if isinstance(stats, dict) and "total" in stats and "with_docs" in stats:
                total_items += int(stats["total"])
                with_docs += int(stats["with_docs"])

    rs_arg_docs, rs_arg_total = calculate_rust_arg_doc_coverage(repo_root)
    return with_docs + rs_arg_docs, total_items + rs_arg_total


def calculate_rust_arg_doc_coverage(repo_root: Path) -> tuple[int, int]:
    """Calculates documented vs total parameters across all Rust `fn` items.

    Args:
        repo_root: Path to the workspace root directory.

    Returns:
        Tuple of `(documented_args, total_args)`.
    """
    crates_dir = repo_root / "crates"
    if not crates_dir.exists():
        return 0, 0

    fn_re = re.compile(
        r"((?:\s*///[^\n]*\n)+(?:\s*#\[[^\n]*\n)*)\s*(?:pub(?:\([^)]*\))?\s+)?(?:const\s+)?(?:unsafe\s+)?(?:extern\s+\"C\"\s+)?fn\s+([A-Za-z0-9_]+)\s*(?:<[^>]*>)?\s*\(([^)]*)\)",
        re.MULTILINE,
    )
    param_re = re.compile(r"^\s*(?:mut\s+)?([A-Za-z0-9_]+)\s*:")
    total_args = 0
    doc_args = 0

    for rs_file in sorted(crates_dir.rglob("*.rs")):
        text = rs_file.read_text(encoding="utf-8")
        for match in fn_re.finditer(text):
            doc_block, _fn_name, raw_params = match.groups()
            for raw_param in raw_params.split(","):
                stripped = raw_param.strip()
                if not stripped or stripped in ("self", "&self", "&mut self"):
                    continue
                pm = param_re.match(stripped)
                if pm:
                    param_name = pm.group(1)
                    total_args += 1
                    if "# Arguments" in doc_block and f"* `{param_name}` -" in doc_block:
                        doc_args += 1

    return doc_args, total_args


def calculate_python_doc_coverage(repo_root: Path) -> tuple[int, int]:
    """Calculates documented vs total modules, classes, functions, and args in `.py`/`.pyi`.

    Args:
        repo_root: Path to the workspace root directory.

    Returns:
        Tuple of `(with_docs, total_items)`.
    """
    total_items = 0
    with_docs = 0
    py_files = sorted((repo_root / "python" / "libtpu").rglob("*.py"))
    py_files.extend(sorted((repo_root / "python" / "libtpu").rglob("*.pyi")))
    scripts_dir = repo_root / "scripts"
    if scripts_dir.exists():
        py_files.extend(sorted(scripts_dir.rglob("*.py")))
        py_files.extend(sorted(scripts_dir.rglob("*.pyi")))

    for py_file in py_files:
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        total_items += 1
        if ast.get_docstring(tree):
            with_docs += 1
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                total_items += 1
                if ast.get_docstring(node):
                    with_docs += 1
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                total_items += 1
                doc = ast.get_docstring(node) or ""
                if doc:
                    with_docs += 1
                for arg in node.args.args:
                    if arg.arg in ("self", "cls"):
                        continue
                    total_items += 1
                    if "Args:" in doc and f"{arg.arg}:" in doc:
                        with_docs += 1

    return with_docs, total_items


def calculate_cpp_doc_coverage(repo_root: Path, force_run: bool = True) -> tuple[int, int]:
    """Calculates documented vs total compounds, members, and params via Doxygen XML.

    Args:
        repo_root: Path to the workspace root directory.
        force_run: Whether to rerun Doxygen before inspecting XML output.

    Returns:
        Tuple of `(with_docs, total_items)`.
    """
    xml_dir = repo_root / "cpp" / "build" / "doxygen" / "xml"
    if force_run or not xml_dir.exists():
        (repo_root / "cpp" / "build" / "doxygen").mkdir(parents=True, exist_ok=True)
        run_command(["doxygen", "cpp/Doxyfile"], cwd=repo_root)

    total_items = 0
    with_docs = 0
    if xml_dir.exists():
        for xml_file in sorted(xml_dir.glob("*.xml")):
            if xml_file.name in ("index.xml", "combine.xslt"):
                continue
            root = ET.parse(xml_file).getroot()
            for compounddef in root.findall(".//compounddef"):
                kind = compounddef.get("kind")
                if kind in ("class", "struct", "file", "group", "namespace"):
                    total_items += 1
                    brief = compounddef.find("briefdescription")
                    deta = compounddef.find("detaileddescription")
                    if _has_xml_text_or_children(brief) or _has_xml_text_or_children(deta):
                        with_docs += 1
                for memberdef in compounddef.findall(".//memberdef"):
                    if memberdef.get("kind") == "friend":
                        continue
                    total_items += 1
                    brief = memberdef.find("briefdescription")
                    deta = memberdef.find("detaileddescription")
                    if _has_xml_text_or_children(brief) or _has_xml_text_or_children(deta):
                        with_docs += 1
                    if memberdef.get("kind") == "function":
                        doc_param_names = {
                            (pn.text or "").strip()
                            for pn in memberdef.findall(
                                ".//parameterlist[@kind='param']/parameteritem/parameternamelist/parametername"
                            )
                        }
                        for param in memberdef.findall("param"):
                            decl = param.find("declname")
                            if decl is not None and decl.text and decl.text.strip():
                                total_items += 1
                                if decl.text.strip() in doc_param_names:
                                    with_docs += 1

    return with_docs, total_items


def calculate_doc_coverage(repo_root: Path, force_run: bool = True) -> float:
    """Calculates aggregate documentation coverage across Rust, Python, and C/C++.

    Args:
        repo_root: Path to the workspace root directory.
        force_run: Whether to force regeneration of documentation artifacts.

    Returns:
        Aggregate documentation coverage percentage (0.0 to 100.0).
    """
    rs_docs, rs_total = calculate_rust_doc_coverage(repo_root, force_run=force_run)
    py_docs, py_total = calculate_python_doc_coverage(repo_root)
    cpp_docs, cpp_total = calculate_cpp_doc_coverage(repo_root, force_run=force_run)

    with_docs = rs_docs + py_docs + cpp_docs
    total_items = rs_total + py_total + cpp_total
    if total_items == 0:
        return 100.0

    pct = (with_docs / total_items) * 100.0
    print(
        f"==> [shields] Doc coverage: {with_docs}/{total_items} items "
        f"(Rust: {rs_docs}/{rs_total}, Python: {py_docs}/{py_total}, "
        f"C/C++: {cpp_docs}/{cpp_total}) -> {pct:.2f}%"
    )
    return pct


def calculate_test_coverage(repo_root: Path, force_run: bool = True) -> float:
    """Calculates aggregate test coverage across Rust (`cargo llvm-cov`) and Python (`pytest-cov`).

    Args:
        repo_root: Path to the workspace root directory.
        force_run: Whether to force execution of coverage runs.

    Returns:
        Minimum test coverage percentage across all measured suites (0.0 to 100.0).
    """
    cov_path = repo_root / "target" / "llvm-cov" / "coverage.json"
    if force_run and cov_path.exists():
        cov_path.unlink(missing_ok=True)

    if force_run or not cov_path.exists():
        print("==> [shields] Generating Rust test coverage report via cargo llvm-cov...")
        cov_path.parent.mkdir(parents=True, exist_ok=True)
        run_command(
            [
                "cargo",
                "+nightly",
                "llvm-cov",
                "--workspace",
                "--branch",
                "--summary-only",
                "--json",
                "--output-path",
                str(cov_path),
            ],
            cwd=repo_root,
        )

    data = json.loads(cov_path.read_text(encoding="utf-8"))
    totals = data["data"][0]["totals"]
    lines_info = totals["lines"]
    total_lines = int(lines_info["count"])
    covered_lines = int(lines_info["covered"])
    line_pct = float(lines_info["percent"])

    funcs_info = totals.get("functions", {})
    func_pct = float(funcs_info.get("percent", line_pct))

    branches_info = totals.get("branches", {})
    branch_pct = (
        float(branches_info["percent"]) if int(branches_info.get("count", 0)) > 0 else line_pct
    )

    py_cov_path = repo_root / "target" / "llvm-cov" / "py_coverage.json"
    if force_run or not py_cov_path.exists():
        print("==> [shields] Generating Python test coverage report via pytest...")
        py_cov_path.parent.mkdir(parents=True, exist_ok=True)
        run_command(
            [
                "pytest",
                "--cov=libtpu",
                "--cov=scripts",
                "--cov-branch",
                f"--cov-report=json:{py_cov_path}",
                "-q",
            ],
            cwd=repo_root,
        )

    py_pct = 100.0
    if py_cov_path.exists():
        py_data = json.loads(py_cov_path.read_text(encoding="utf-8"))
        py_pct = float(py_data.get("totals", {}).get("percent_covered", 100.0))

    cpp_pct = calculate_cpp_test_coverage(repo_root, force_run=force_run)
    pct = min(line_pct, func_pct, branch_pct, py_pct, cpp_pct)
    print(
        f"==> [shields] Test coverage: Rust {covered_lines}/{total_lines} lines "
        f"(lines: {line_pct:.2f}%, functions: {func_pct:.2f}%, branches: {branch_pct:.2f}%), "
        f"Python: {py_pct:.2f}%, C/C++: {cpp_pct:.2f}% -> {pct:.2f}%"
    )
    return pct


def calculate_cpp_test_coverage(repo_root: Path, force_run: bool = True) -> float:
    """Calculates C/C++ line, function, and branch coverage via `llvm-profdata` and `llvm-cov`.

    Args:
        repo_root: Path to the workspace root directory.
        force_run: Whether to merge `.profraw` and export `cpp_coverage.json`.

    Returns:
        Minimum of C/C++ line, function, and branch coverage percentages (0.0 to 100.0).
    """
    cpp_cov_path = repo_root / "target" / "llvm-cov" / "cpp_coverage.json"
    profraw = repo_root / "cpp" / "build" / "test_libtpu.profraw"
    profdata = repo_root / "cpp" / "build" / "test_libtpu.profdata"
    test_bin = repo_root / "cpp" / "build" / "test_libtpu"

    if force_run and test_bin.exists():
        env = get_augmented_env()
        if not profraw.exists():
            env["LLVM_PROFILE_FILE"] = str(profraw)
            run_command(
                ["ctest", "--test-dir", "cpp/build", "--output-on-failure"],
                cwd=repo_root,
                env=env,
            )
        use_xcrun = bool(shutil.which("xcrun", path=env.get("PATH")))
        prof_prefix = ["xcrun", "llvm-profdata"] if use_xcrun else ["llvm-profdata"]
        cov_prefix = ["xcrun", "llvm-cov"] if use_xcrun else ["llvm-cov"]
        run_command(
            [*prof_prefix, "merge", "-sparse", str(profraw), "-o", str(profdata)],
            cwd=repo_root,
            env=env,
        )
        res = run_command(
            [
                *cov_prefix,
                "export",
                "--format=text",
                str(test_bin),
                f"-instr-profile={profdata}",
                str(repo_root / "cpp" / "src" / "libtpu.cpp"),
                str(repo_root / "cpp" / "src" / "libtpu.hpp"),
            ],
            cwd=repo_root,
            env=env,
            capture_output=True,
        )
        cpp_cov_path.parent.mkdir(parents=True, exist_ok=True)
        cpp_cov_path.write_text(res.stdout, encoding="utf-8")

    if not cpp_cov_path.exists():
        return 100.0

    cpp_data = json.loads(cpp_cov_path.read_text(encoding="utf-8"))
    cpp_totals = cpp_data["data"][0]["totals"]
    cpp_line_pct = float(cpp_totals["lines"]["percent"])
    cpp_func_pct = float(cpp_totals.get("functions", {}).get("percent", cpp_line_pct))
    cpp_branches = cpp_totals.get("branches", {})
    cpp_branch_pct = (
        float(cpp_branches["percent"]) if int(cpp_branches.get("count", 0)) > 0 else cpp_line_pct
    )
    return min(cpp_line_pct, cpp_func_pct, cpp_branch_pct)


def update_readme_shields(
    repo_root: Path,
    doc_pct: float,
    test_pct: float,
    check_only: bool = False,
) -> bool:
    """Adds or updates `% doc coverage` and `% test coverage` shields in `README.md`.

    Args:
        repo_root: Path to the workspace root directory.
        doc_pct: Documentation coverage percentage.
        test_pct: Test coverage percentage.
        check_only: If True, do not write changes and report whether `README.md` is out of date.

    Returns:
        True if `README.md` was modified (or would be modified in `check_only` mode),
        False otherwise.
    """
    nl = chr(10)
    readme_path = repo_root / "README.md"
    if not readme_path.exists():
        if check_only:
            print(f"Error: README.md not found at {readme_path}", file=sys.stderr)
            return True
        readme_path.write_text(f"# `libtpu-rs`{nl}", encoding="utf-8")

    content = readme_path.read_text(encoding="utf-8")

    doc_pct_str = format_pct(doc_pct)
    test_pct_str = format_pct(test_pct)
    doc_color = get_badge_color(doc_pct)
    test_color = get_badge_color(test_pct)

    doc_encoded = doc_pct_str.replace("%", "%25")
    test_encoded = test_pct_str.replace("%", "%25")

    doc_badge_url = f"https://img.shields.io/badge/doc%20coverage-{doc_encoded}-{doc_color}.svg"
    test_badge_url = f"https://img.shields.io/badge/test%20coverage-{test_encoded}-{test_color}.svg"

    doc_shield = f"[![% doc coverage]({doc_badge_url})](#verification--quality-gates)"
    test_shield = f"[![% test coverage]({test_badge_url})](#verification--quality-gates)"

    doc_pattern = re.compile(
        rf"\[?!\[[^\]]*(?:%\s*)?doc\s+coverage[^\]]*\]\([^{nl}]+\)",
        re.IGNORECASE,
    )
    test_pattern = re.compile(
        rf"\[?!\[[^\]]*(?:%\s*)?test\s+coverage[^\]]*\]\([^{nl}]+\)",
        re.IGNORECASE,
    )

    has_doc = bool(doc_pattern.search(content))
    has_test = bool(test_pattern.search(content))

    new_content = content
    if has_doc:
        new_content = doc_pattern.sub(doc_shield, new_content, count=1)
    if has_test:
        new_content = test_pattern.sub(test_shield, new_content, count=1)

    both_shields = f"{doc_shield}{nl}{test_shield}"

    if not has_doc and not has_test:
        lines = new_content.splitlines(keepends=True)
        updated_lines: list[str] = []
        inserted = False
        for line in lines:
            updated_lines.append(line)
            if not inserted and line.startswith("# "):
                updated_lines.append(f"{nl}{both_shields}{nl}")
                inserted = True
        if not inserted:
            updated_lines.insert(0, f"{both_shields}{nl}{nl}")
        new_content = "".join(updated_lines)
    elif has_doc and not has_test:
        new_content = doc_pattern.sub(both_shields, new_content, count=1)
    elif has_test and not has_doc:
        new_content = test_pattern.sub(both_shields, new_content, count=1)

    if new_content != content:
        if check_only:
            print(
                f"Error: README.md shields are out of date. Expected:{nl}"
                f"    {doc_shield}{nl}    {test_shield}",
                file=sys.stderr,
            )
            return True
        readme_path.write_text(new_content, encoding="utf-8")
        print(
            f"==> [shields] Updated README.md with shields:{nl}"
            f"    {doc_shield}{nl}    {test_shield}"
        )
        return True

    print("==> [shields] README.md shields are already up-to-date.")
    return False


def resolve_compiled_cdylib(repo_root: Path) -> Path | None:
    """Resolves the compiled `libtpu` shared library path (`libtpu.dylib`, `libtpu.so`, `tpu.dll`).

    Args:
        repo_root: Path to the workspace root directory.

    Returns:
        Path to the compiled shared library if found, or None otherwise.
    """
    candidates = [
        repo_root / "target" / "debug" / "libtpu.dylib",
        repo_root / "target" / "debug" / "libtpu.so",
        repo_root / "target" / "debug" / "tpu.dll",
        repo_root / "target" / "release" / "libtpu.dylib",
        repo_root / "target" / "release" / "libtpu.so",
        repo_root / "target" / "release" / "tpu.dll",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def verify_rust_and_c_abi_sources(repo_root: Path) -> list[str]:
    """Verifies Rust `#[repr(C)]` struct size assertions and `cpp/src/libtpu.h` symbol declarations.

    Args:
        repo_root: Path to the workspace root directory.

    Returns:
        List of ABI violation diagnostic strings (empty when 100% compatible).
    """
    errors: list[str] = []
    c_api_src = repo_root / "crates" / "libtpu-c-api" / "src"
    header_path = repo_root / "cpp" / "src" / "libtpu.h"

    if not c_api_src.exists() or not header_path.exists():
        errors.append("Missing crates/libtpu-c-api/src or cpp/src/libtpu.h")
        return errors

    rs_combined = "\n".join(p.read_text(encoding="utf-8") for p in sorted(c_api_src.rglob("*.rs")))
    header_text = header_path.read_text(encoding="utf-8")

    for sym in GOOGLE_LIBTPU_EXPORTED_SYMBOLS:
        if sym not in rs_combined:
            errors.append(f"Symbol {sym} missing from crates/libtpu-c-api/src")
        if sym not in header_text:
            errors.append(f"Symbol {sym} missing from cpp/src/libtpu.h")

    for struct_name, expected_size in EXPECTED_REPR_C_STRUCT_SIZES:
        size_token = f"size_of::<{struct_name}>()"
        if size_token not in rs_combined:
            errors.append(
                f"Compile-time size_of assertion missing for #[repr(C)] "
                f"struct {struct_name} ({expected_size} B)"
            )

    for rel_py, required_names in EXPECTED_PYTHON_WHEEL_SYMBOLS:
        py_path = repo_root / rel_py
        if not py_path.exists():
            errors.append(f"Missing Python wheel API module {rel_py}")
            continue
        py_text = py_path.read_text(encoding="utf-8")
        for req_name in required_names:
            if req_name not in py_text:
                errors.append(f"Python wheel API symbol {req_name} missing from {rel_py}")

    cargo_toml = repo_root / "Cargo.toml"
    if cargo_toml.exists():
        cargo_text = cargo_toml.read_text(encoding="utf-8")
        if "anyhow" in cargo_text or "thiserror" in cargo_text:
            errors.append("Forbidden dependency anyhow/thiserror found in Cargo.toml")

    error_rs = repo_root / "crates" / "libtpu-core" / "src" / "error.rs"
    if error_rs.exists():
        error_text = error_rs.read_text(encoding="utf-8")
        if "pub enum LibTpuError" not in error_text or "Display, Error, From" not in error_text:
            errors.append("LibTpuError in error.rs must derive Display, Error, From")

    for rs_file in sorted((repo_root / "crates").rglob("*.rs")):
        rs_text = rs_file.read_text(encoding="utf-8")
        if ".unwrap(" in rs_text or ".expect(" in rs_text:
            errors.append(f"Forbidden .unwrap() or .expect() found in {rs_file.name}")

    return errors


def verify_binary_vtable_abi(cdylib_path: Path) -> list[str]:
    """Loads `cdylib_path` via `ctypes.CDLL` and verifies `libtpu.so` binary ABI invariants.

    Args:
        cdylib_path: Path to the compiled `libtpu` dynamic library.

    Returns:
        List of binary ABI violation diagnostic strings (empty when 100% compatible).
    """
    errors: list[str] = []
    lib = ctypes.CDLL(str(cdylib_path))

    for sym in GOOGLE_LIBTPU_EXPORTED_SYMBOLS:
        if not hasattr(lib, sym):
            errors.append(f"Exported symbol '{sym}' missing from dynamic symbol table")

    if errors:
        return errors

    lib.GetPjrtApi.restype = ctypes.c_void_p
    pjrt_api_addr = int(lib.GetPjrtApi() or 0)
    lib.GetLibtpuSdkApi.restype = ctypes.c_void_p
    sdk_api_addr = int(lib.GetLibtpuSdkApi() or 0)
    if pjrt_api_addr == 0 or sdk_api_addr == 0:
        errors.append("GetPjrtApi() or GetLibtpuSdkApi() returned NULL")
        return errors

    qwords = (ctypes.c_uint64 * 143).from_address(pjrt_api_addr)
    struct_size = int(qwords[0])
    ver_size = int(qwords[2])
    ver_major = ctypes.c_int32.from_address(pjrt_api_addr + 0x20).value
    ver_minor = ctypes.c_int32.from_address(pjrt_api_addr + 0x24).value
    null_pjrt_slots = [idx for idx in range(5, 143) if int(qwords[idx]) == 0]
    if (struct_size, ver_size, ver_major, ver_minor) != (1144, 24, 0, 116) or null_pjrt_slots:
        errors.append(
            f"PJRT_Api header/slots mismatch: size={struct_size}, "
            f"ver=({ver_size},{ver_major},{ver_minor}), null_slots={null_pjrt_slots}"
        )

    ext_addr = int(qwords[1])
    actual_chain: list[tuple[int, int, int]] = []
    while ext_addr != 0 and len(actual_chain) < 32:
        ext_size = int(ctypes.c_uint64.from_address(ext_addr).value)
        ext_type = int(ctypes.c_int32.from_address(ext_addr + 8).value)
        next_addr = int(ctypes.c_uint64.from_address(ext_addr + 16).value)
        fn_slots = (ext_size - 24) // 8
        actual_chain.append((ext_type, ext_size, fn_slots))
        ext_addr = next_addr

    if tuple(actual_chain) != EXPECTED_PJRT_EXTENSION_CHAIN:
        errors.append(
            f"PJRT_Extension_Base chain mismatch: "
            f"expected {EXPECTED_PJRT_EXTENSION_CHAIN}, got {tuple(actual_chain)}"
        )

    sdk_qwords = (ctypes.c_uint64 * 18).from_address(sdk_api_addr)
    sdk_major = int(ctypes.c_int32.from_address(sdk_api_addr).value)
    sdk_minor = int(ctypes.c_int32.from_address(sdk_api_addr + 4).value)
    null_sdk_slots = [idx for idx in range(1, 18) if int(sdk_qwords[idx]) == 0]
    if (sdk_major, sdk_minor) != (0, 1) or null_sdk_slots:
        errors.append(
            f"LibtpuSdk_Api mismatch: ver=({sdk_major},{sdk_minor}), null_slots={null_sdk_slots}"
        )

    lib.TpuDriver_Open.argtypes = [ctypes.c_char_p]
    lib.TpuDriver_Open.restype = ctypes.c_void_p
    lib.TpuDriver_Close.argtypes = [ctypes.c_void_p]
    lib.TpuDriver_Close.restype = None
    drv_ptr = int(lib.TpuDriver_Open(b"local://") or 0)
    if drv_ptr == 0:
        errors.append("TpuDriver_Open returned NULL")
        return errors

    fn_table_addr = int(ctypes.c_uint64.from_address(drv_ptr).value)
    drv_slots = (ctypes.c_uint64 * 28).from_address(fn_table_addr)
    null_drv_slots = [idx for idx in range(28) if int(drv_slots[idx]) == 0]
    lib.TpuDriver_Close(drv_ptr)
    if null_drv_slots:
        errors.append(f"TpuDriverFn null slots: {null_drv_slots}")

    return errors


def hook_abi(repo_root: Path, check_only: bool = False) -> int:
    """Verifies 100% binary C ABI and Python SDK parity with official Google `libtpu` (`v0.0.50`).

    Args:
        repo_root: Path to the workspace root directory.
        check_only: Unused flag kept for uniform stage handler signature.

    Returns:
        Zero on success, or non-zero if any ABI compatibility check fails.
    """
    _ = check_only
    print("==> [abi] Verifying Google libtpu (v0.0.50) C ABI & vtable compatibility...")
    src_errors = verify_rust_and_c_abi_sources(repo_root)
    if src_errors:
        for err_msg in src_errors:
            print(f"ABI Error: {err_msg}", file=sys.stderr)
        return 1

    cdylib_path = resolve_compiled_cdylib(repo_root)
    if cdylib_path is None:
        run_command(["cargo", "build", "-p", "libtpu-c-api"], cwd=repo_root)
        cdylib_path = resolve_compiled_cdylib(repo_root)

    if cdylib_path is None:
        print("ABI Error: Compiled libtpu cdylib not found", file=sys.stderr)
        return 1

    bin_errors = verify_binary_vtable_abi(cdylib_path)
    if bin_errors:
        for err_msg in bin_errors:
            print(f"ABI Error: {err_msg}", file=sys.stderr)
        return 1

    print(
        f"==> [abi] Verified {len(GOOGLE_LIBTPU_EXPORTED_SYMBOLS)} exported symbols, "
        "PJRT_Api v0.116 (1144B, 138 slots), 17 PJRT extensions, LibtpuSdk_Api v0.1 (144B), "
        "and TpuDriverFn (224B, 28 slots)."
    )
    return 0


def hook_shields(
    repo_root: Path,
    check_only: bool = False,
    force_run: bool = True,
) -> int:
    """Collects doc and test coverage and updates/verifies shields in `README.md`.

    Args:
        repo_root: Path to the workspace root directory.
        check_only: If True, fail if `README.md` would be modified.
        force_run: Whether to force fresh coverage collection.

    Returns:
        Zero on success, or non-zero if coverage is below 100% or `check_only` fails.
    """
    doc_pct = calculate_doc_coverage(repo_root, force_run=force_run)
    test_pct = calculate_test_coverage(repo_root, force_run=force_run)
    changed = update_readme_shields(repo_root, doc_pct, test_pct, check_only=check_only)
    if doc_pct < 100.0 or test_pct < 100.0:
        print(
            f"Error: Coverage below 100% threshold (doc={doc_pct:.2f}%, test={test_pct:.2f}%)",
            file=sys.stderr,
        )
        return 1
    if check_only and changed:
        return 1
    return 0


def hook_all(repo_root: Path, check_only: bool = False) -> int:
    """Executes the complete pre-commit validation pipeline in sequence.

    Args:
        repo_root: Path to the workspace root directory.
        check_only: If True, run in strict read-only verification mode.

    Returns:
        Zero on success, or non-zero if any stage fails.
    """
    print("========================================================")
    print(" Running libtpu-rs Pre-commit Hook Pipeline (Python)")
    print("========================================================")
    for stage_fn in (hook_fmt, hook_build, hook_clippy, hook_abi, hook_test, hook_shields):
        rc = stage_fn(repo_root, check_only=check_only)
        if rc != 0:
            return rc
    print("========================================================")
    print(" All pre-commit checks passed successfully!")
    print("========================================================")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the `libtpu-rs` pre-commit pipeline.

    Args:
        argv: Optional list of CLI arguments (defaults to `sys.argv[1:]`).

    Returns:
        Process exit code (0 on success, non-zero on failure).
    """
    parser = argparse.ArgumentParser(
        description="libtpu-rs Cross-Platform Pre-commit Hook Pipeline"
    )
    parser.add_argument(
        "stage",
        nargs="?",
        choices=[
            "fmt",
            "cargo-fmt",
            "ruff-format",
            "clang-format",
            "build",
            "clippy",
            "lint",
            "abi",
            "test",
            "shields",
            "all",
        ],
        default="all",
        help="Pipeline stage to execute (default: all)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run in strict CI verification mode without modifying tracked files",
    )
    args = parser.parse_args(argv)
    repo_root = get_repo_root()
    check_only = bool(args.check)
    stage = str(args.stage)

    try:
        if stage == "fmt":
            return hook_fmt(repo_root, check_only=check_only)
        if stage == "cargo-fmt":
            return hook_cargo_fmt(repo_root, check_only=check_only)
        if stage == "ruff-format":
            return hook_ruff_format(repo_root, check_only=check_only)
        if stage == "clang-format":
            return hook_clang_format(repo_root, check_only=check_only)
        if stage == "build":
            return hook_build(repo_root, check_only=check_only)
        if stage in ("clippy", "lint"):
            return hook_clippy(repo_root, check_only=check_only)
        if stage == "abi":
            return hook_abi(repo_root, check_only=check_only)
        if stage == "test":
            return hook_test(repo_root, check_only=check_only)
        if stage == "shields":
            return hook_shields(repo_root, check_only=check_only)
        return hook_all(repo_root, check_only=check_only)
    except subprocess.CalledProcessError as err:
        cmd_str = " ".join(err.cmd) if isinstance(err.cmd, list) else str(err.cmd)
        print(
            f"Command failed with exit code {err.returncode}: {cmd_str}",
            file=sys.stderr,
        )
        return int(err.returncode)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

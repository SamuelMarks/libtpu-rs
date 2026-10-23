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
5. Doc & test coverage calculation and `% doc coverage` / `% test coverage` shield updates
   in `README.md`.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


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


def hook_fmt(repo_root: Path, check_only: bool = False) -> int:
    """First runs auto-formatting fixes (unless `check_only` is set), then rechecks formatting.

    Args:
        repo_root: Path to the workspace root directory.
        check_only: If True, skip auto-formatting and only run the verification checks.

    Returns:
        Zero on success.
    """
    py_targets = get_python_targets(repo_root)
    cpp_files = get_cpp_files(repo_root)

    if not check_only:
        print("==> [fmt] Auto-fixing Rust formatting with cargo fmt...")
        run_command(["cargo", "fmt", "--all"], cwd=repo_root)

        print("==> [fmt] Auto-fixing Python formatting with ruff format...")
        run_command(["ruff", "format", *py_targets], cwd=repo_root)

        if cpp_files:
            print("==> [fmt] Auto-fixing C/C++ formatting with clang-format...")
            run_command(
                ["clang-format", "-i", "-style=LLVM", *cpp_files],
                cwd=repo_root,
            )

    print("==> [fmt] Rechecking Rust formatting (cargo fmt --all -- --check)...")
    run_command(["cargo", "fmt", "--all", "--", "--check"], cwd=repo_root)

    print("==> [fmt] Rechecking Python formatting (ruff format --check)...")
    run_command(["ruff", "format", "--check", *py_targets], cwd=repo_root)

    if cpp_files:
        print("==> [fmt] Rechecking C/C++ formatting (clang-format --dry-run --Werror)...")
        run_command(
            ["clang-format", "--dry-run", "--Werror", "-style=LLVM", *cpp_files],
            cwd=repo_root,
        )

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
    if not build_dir.exists():
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

    return with_docs, total_items


def calculate_python_doc_coverage(repo_root: Path) -> tuple[int, int]:
    """Calculates documented vs total modules, classes, and functions across Python files.

    Args:
        repo_root: Path to the workspace root directory.

    Returns:
        Tuple of `(with_docs, total_items)`.
    """
    total_items = 0
    with_docs = 0
    py_files = sorted((repo_root / "python" / "libtpu").rglob("*.py"))
    scripts_dir = repo_root / "scripts"
    if scripts_dir.exists():
        py_files.extend(sorted(scripts_dir.rglob("*.py")))

    for py_file in py_files:
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        total_items += 1
        if ast.get_docstring(tree):
            with_docs += 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                total_items += 1
                if ast.get_docstring(node):
                    with_docs += 1

    return with_docs, total_items


def calculate_cpp_doc_coverage(repo_root: Path, force_run: bool = True) -> tuple[int, int]:
    """Calculates documented vs total compounds and members across C/C++ via Doxygen XML.

    Args:
        repo_root: Path to the workspace root directory.
        force_run: Whether to rerun Doxygen before inspecting XML output.

    Returns:
        Tuple of `(with_docs, total_items)`.
    """
    xml_dir = repo_root / "cpp" / "cpp" / "build" / "doxygen_xml"
    if force_run or not xml_dir.exists():
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

    pct = min(line_pct, func_pct, branch_pct, py_pct)
    print(
        f"==> [shields] Test coverage: Rust {covered_lines}/{total_lines} lines "
        f"(lines: {line_pct:.2f}%, functions: {func_pct:.2f}%, branches: {branch_pct:.2f}%), "
        f"Python: {py_pct:.2f}% -> {pct:.2f}%"
    )
    return pct


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
    for stage_fn in (hook_fmt, hook_build, hook_clippy, hook_test, hook_shields):
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
        choices=["fmt", "build", "clippy", "lint", "test", "shields", "all"],
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
        if stage == "build":
            return hook_build(repo_root, check_only=check_only)
        if stage in ("clippy", "lint"):
            return hook_clippy(repo_root, check_only=check_only)
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

"""Unit tests for the `scripts/pre_commit.py` pipeline and shield updater."""

from __future__ import annotations

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import scripts.pre_commit as pc

NL = chr(10)


def test_pre_commit_helpers_and_badges(tmp_path: Path) -> None:
    """Tests helper functions, badge formatting, and XML inspection in `scripts/pre_commit.py`.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
    """
    root = pc.get_repo_root()
    assert (root / "Cargo.toml").exists()

    env = pc.get_augmented_env()
    assert "PATH" in env
    assert "GIT_INDEX_FILE" not in env

    with patch.dict("os.environ", {"PATH": ""}):
        env2 = pc.get_augmented_env()
        assert "PATH" in env2

    cpp_files = pc.get_cpp_files(root)
    assert "cpp/src/libtpu.h" in cpp_files
    assert "cpp/src/libtpu.hpp" in cpp_files
    assert "cpp/src/libtpu.cpp" in cpp_files

    assert pc.get_python_targets(root) == ["python/", "scripts/"]
    assert pc.get_python_targets(tmp_path) == ["python/"]

    assert pc.get_badge_color(100.0) == "brightgreen"
    assert pc.get_badge_color(92.0) == "green"
    assert pc.get_badge_color(85.0) == "yellowgreen"
    assert pc.get_badge_color(75.0) == "yellow"
    assert pc.get_badge_color(65.0) == "orange"
    assert pc.get_badge_color(40.0) == "red"

    assert pc.format_pct(100.0) == "100%"
    assert pc.format_pct(90.0) == "90%"
    assert pc.format_pct(94.56) == "94.6%"

    assert not pc._has_xml_text_or_children(None)
    empty_el = ET.fromstring("<briefdescription>   </briefdescription>")
    assert not pc._has_xml_text_or_children(empty_el)
    text_el = ET.fromstring("<briefdescription>Doc text</briefdescription>")
    assert pc._has_xml_text_or_children(text_el)
    child_el = ET.fromstring("<briefdescription><para>Doc</para></briefdescription>")
    assert pc._has_xml_text_or_children(child_el)

    res = pc.run_command(["python3", "-c", "print('ok')"], cwd=tmp_path, capture_output=True)
    assert res.stdout.strip() == "ok"

    res_unres = pc.run_command(
        [str(Path(sys.executable)), "-c", "print('direct')"],
        cwd=tmp_path,
        env={"PATH": ""},
        capture_output=True,
    )
    assert res_unres.stdout.strip() == "direct"


def test_update_readme_shields_variants(tmp_path: Path) -> None:
    """Tests `update_readme_shields` across missing, partial, and up-to-date `README.md` files.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
    """
    assert pc.update_readme_shields(tmp_path, 100.0, 100.0, check_only=True)
    assert not (tmp_path / "README.md").exists()

    assert pc.update_readme_shields(tmp_path, 100.0, 100.0, check_only=False)
    content = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "% doc coverage" in content
    assert "% test coverage" in content

    assert not pc.update_readme_shields(tmp_path, 100.0, 100.0, check_only=False)
    assert pc.update_readme_shields(tmp_path, 85.5, 92.0, check_only=True)

    no_h1 = tmp_path / "no_h1"
    no_h1.mkdir()
    (no_h1 / "README.md").write_text(f"Plain text without heading{NL}", encoding="utf-8")
    assert pc.update_readme_shields(no_h1, 100.0, 100.0)
    assert "% doc coverage" in (no_h1 / "README.md").read_text(encoding="utf-8")

    only_doc = tmp_path / "only_doc"
    only_doc.mkdir()
    (only_doc / "README.md").write_text(
        f"# Title{NL}{NL}[![% doc coverage](https://img.shields.io/badge/doc-90%25-green.svg)](#){NL}",
        encoding="utf-8",
    )
    assert pc.update_readme_shields(only_doc, 100.0, 100.0)
    text_only_doc = (only_doc / "README.md").read_text(encoding="utf-8")
    assert "% doc coverage" in text_only_doc
    assert "% test coverage" in text_only_doc

    only_test = tmp_path / "only_test"
    only_test.mkdir()
    (only_test / "README.md").write_text(
        f"# Title{NL}{NL}[![% test coverage](https://img.shields.io/badge/test-90%25-green.svg)](#){NL}",
        encoding="utf-8",
    )
    assert pc.update_readme_shields(only_test, 100.0, 100.0)
    text_only_test = (only_test / "README.md").read_text(encoding="utf-8")
    assert "% doc coverage" in text_only_test
    assert "% test coverage" in text_only_test

    with_ci_shields = tmp_path / "with_ci_shields"
    with_ci_shields.mkdir()
    ci_badge = "[![CI & Pre-commit Quality Gates](https://github.com/SamuelMarks/libtpu-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/SamuelMarks/libtpu-rs/actions/workflows/ci.yml)"
    rel_badge = "[![Release Binaries & Python Wheels](https://github.com/SamuelMarks/libtpu-rs/actions/workflows/release.yml/badge.svg)](https://github.com/SamuelMarks/libtpu-rs/actions/workflows/release.yml)"
    lic_badge = "[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](https://opensource.org/licenses/Apache-2.0)"
    (with_ci_shields / "README.md").write_text(
        f"# `libtpu-rs`{NL}{NL}{lic_badge}{NL}{ci_badge}{NL}{rel_badge}{NL}"
        f"[![% doc coverage](https://img.shields.io/badge/doc%20coverage-95%25-green.svg)](#verification--quality-gates){NL}"
        f"[![% test coverage](https://img.shields.io/badge/test%20coverage-95%25-green.svg)](#verification--quality-gates){NL}",
        encoding="utf-8",
    )
    assert pc.update_readme_shields(with_ci_shields, 100.0, 100.0)
    updated_with_ci = (with_ci_shields / "README.md").read_text(encoding="utf-8")
    assert f"{lic_badge}{NL}{ci_badge}{NL}{rel_badge}" in updated_with_ci
    assert "doc%20coverage-100%25-brightgreen.svg" in updated_with_ci
    assert "test%20coverage-100%25-brightgreen.svg" in updated_with_ci


def test_coverage_calculators(tmp_path: Path) -> None:
    """Tests Rust, Python, and C++ documentation and test coverage calculators.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
    """
    doc_dir = tmp_path / "target" / "doc"
    doc_dir.mkdir(parents=True)
    (doc_dir / "sample.json").write_text(
        json.dumps(
            {
                "src/lib.rs": {"total": 10, "with_docs": 10},
                "ignored": "not-a-dict",
            }
        ),
        encoding="utf-8",
    )
    rs_docs, rs_total = pc.calculate_rust_doc_coverage(tmp_path, force_run=False)
    assert (rs_docs, rs_total) == (10, 10)

    def fake_rustdoc(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
        """Writes synthetic rustdoc JSON output for testing.

        Args:
            cmd: Command list.
            cwd: Working directory.
            env: Environment dictionary.
        """
        _ = (cmd, env)
        (cwd / "target" / "doc" / "gen.json").write_text(
            json.dumps({"src/lib.rs": {"total": 5, "with_docs": 5}}),
            encoding="utf-8",
        )

    with patch.object(pc, "run_command", side_effect=fake_rustdoc):
        rs_docs2, rs_total2 = pc.calculate_rust_doc_coverage(tmp_path, force_run=True)
        assert (rs_docs2, rs_total2) == (5, 5)

    rs_crate = tmp_path / "crates" / "demo" / "src"
    rs_crate.mkdir(parents=True)
    (rs_crate / "lib.rs").write_text(
        f"/// Docs.{NL}/// # Arguments{NL}/// * `x` - X value.{NL}"
        f"pub fn documented(&self, x: i32, y: i32, (a, b): (u8, u8)) {{}}{NL}"
        f"/// Docs.{NL}pub fn no_args() {{}}{NL}",
        encoding="utf-8",
    )
    assert pc.calculate_rust_arg_doc_coverage(tmp_path) == (1, 2)

    py_pkg = tmp_path / "python" / "libtpu"
    py_pkg.mkdir(parents=True)
    (py_pkg / "mod.py").write_text(
        f'"""Module doc."""{NL}{NL}'
        f'class Cls:{NL}    """Class doc."""{NL}'
        f"    def m(self, a: int) -> None:{NL}"
        f'        """Method.{NL}{NL}        Args:{NL}            a: A.{NL}        """{NL}{NL}'
        f'def fn(b: int) -> None:{NL}    """Fn doc."""{NL}',
        encoding="utf-8",
    )
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "helper.py").write_text(
        f"class UndocCls:{NL}    pass{NL}{NL}def undoc() -> None:{NL}    pass{NL}",
        encoding="utf-8",
    )
    py_docs, py_total = pc.calculate_python_doc_coverage(tmp_path)
    assert (py_docs, py_total) == (5, 9)

    xml_dir = tmp_path / "cpp" / "build" / "doxygen" / "xml"
    xml_dir.mkdir(parents=True)
    (xml_dir / "index.xml").write_text("<doxygenindex/>", encoding="utf-8")
    (xml_dir / "struct_foo.xml").write_text(
        """<doxygen>
        <compounddef kind="struct">
            <briefdescription><para>Struct doc</para></briefdescription>
            <detaileddescription></detaileddescription>
            <sectiondef>
                <memberdef kind="function">
                    <briefdescription><para>Fn doc</para></briefdescription>
                    <detaileddescription>
                        <para><parameterlist kind="param"><parameteritem>
                            <parameternamelist><parametername>x</parametername></parameternamelist>
                        </parameteritem></parameterlist></para>
                    </detaileddescription>
                    <param><declname>x</declname></param>
                    <param><declname>y</declname></param>
                    <param><type>void</type></param>
                </memberdef>
                <memberdef kind="friend">
                    <briefdescription></briefdescription>
                    <detaileddescription></detaileddescription>
                </memberdef>
                <memberdef kind="variable">
                    <briefdescription></briefdescription>
                    <detaileddescription></detaileddescription>
                </memberdef>
            </sectiondef>
        </compounddef>
        <compounddef kind="struct">
            <briefdescription></briefdescription>
            <detaileddescription></detaileddescription>
        </compounddef>
        <compounddef kind="dir">
            <briefdescription></briefdescription>
        </compounddef>
        </doxygen>""",
        encoding="utf-8",
    )
    cpp_docs, cpp_total = pc.calculate_cpp_doc_coverage(tmp_path, force_run=False)
    assert (cpp_docs, cpp_total) == (3, 6)

    with patch.object(pc, "run_command"):
        cpp_docs2, cpp_total2 = pc.calculate_cpp_doc_coverage(tmp_path, force_run=True)
        assert (cpp_docs2, cpp_total2) == (3, 6)

    empty_root = tmp_path / "empty"
    (empty_root / "python" / "libtpu").mkdir(parents=True)
    with patch.object(pc, "run_command"):
        assert pc.calculate_doc_coverage(empty_root, force_run=False) == 100.0

    with (
        patch.object(pc, "calculate_rust_doc_coverage", return_value=(10, 10)),
        patch.object(pc, "calculate_python_doc_coverage", return_value=(5, 5)),
        patch.object(pc, "calculate_cpp_doc_coverage", return_value=(5, 5)),
    ):
        assert pc.calculate_doc_coverage(tmp_path, force_run=False) == 100.0

    cov_dir = tmp_path / "target" / "llvm-cov"
    cov_dir.mkdir(parents=True)
    (cov_dir / "coverage.json").write_text(
        json.dumps(
            {
                "data": [
                    {
                        "totals": {
                            "lines": {"count": 100, "covered": 100, "percent": 100.0},
                            "functions": {"count": 20, "covered": 20, "percent": 100.0},
                            "branches": {"count": 10, "covered": 10, "percent": 100.0},
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (cov_dir / "py_coverage.json").write_text(
        json.dumps({"totals": {"percent_covered": 100.0}}),
        encoding="utf-8",
    )
    assert pc.calculate_test_coverage(tmp_path, force_run=False) == 100.0

    def fake_cov_run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
        """Writes synthetic llvm-cov JSON output for testing.

        Args:
            cmd: Command list.
            cwd: Working directory.
            env: Environment dictionary.
        """
        _ = env
        if "llvm-cov" in cmd:
            (cwd / "target" / "llvm-cov" / "coverage.json").write_text(
                json.dumps(
                    {
                        "data": [
                            {
                                "totals": {
                                    "lines": {
                                        "count": 100,
                                        "covered": 100,
                                        "percent": 100.0,
                                    },
                                    "functions": {
                                        "count": 20,
                                        "covered": 20,
                                        "percent": 100.0,
                                    },
                                    "branches": {
                                        "count": 0,
                                        "covered": 0,
                                        "percent": 0.0,
                                    },
                                }
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

    (cov_dir / "py_coverage.json").unlink()
    with patch.object(pc, "run_command", side_effect=fake_cov_run):
        assert pc.calculate_test_coverage(tmp_path, force_run=True) == 100.0

    cpp_build = tmp_path / "cpp" / "build"
    cpp_build.mkdir(parents=True, exist_ok=True)
    test_bin = cpp_build / "test_libtpu"
    test_bin.write_text("bin", encoding="utf-8")

    def fake_cpp_cov(
        cmd: list[str],
        cwd: Path,
        env: dict[str, str] | None = None,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        """Simulates ctest, llvm-profdata, and llvm-cov export.

        Args:
            cmd: Command list.
            cwd: Working directory.
            env: Environment dictionary.
            capture_output: Whether stdout is captured.

        Returns:
            CompletedProcess with synthetic llvm-cov JSON.
        """
        _ = (cwd, env, capture_output)
        out = json.dumps(
            {
                "data": [
                    {
                        "totals": {
                            "lines": {"count": 50, "covered": 50, "percent": 100.0},
                            "functions": {"count": 10, "covered": 10, "percent": 100.0},
                            "branches": {"count": 8, "covered": 8, "percent": 100.0},
                        }
                    }
                ]
            }
        )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=out, stderr="")

    with patch.object(pc, "run_command", side_effect=fake_cpp_cov):
        assert pc.calculate_cpp_test_coverage(tmp_path, force_run=True) == 100.0
    (cpp_build / "test_libtpu.profraw").write_text("prof", encoding="utf-8")
    with (
        patch.object(pc, "run_command", side_effect=fake_cpp_cov),
        patch("shutil.which", return_value=None),
    ):
        assert pc.calculate_cpp_test_coverage(tmp_path, force_run=True) == 100.0

    (cov_dir / "cpp_coverage.json").write_text(
        json.dumps(
            {
                "data": [
                    {
                        "totals": {
                            "lines": {"count": 50, "covered": 50, "percent": 100.0},
                            "functions": {"count": 10, "covered": 10, "percent": 100.0},
                            "branches": {"count": 0, "covered": 0, "percent": 0.0},
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert pc.calculate_cpp_test_coverage(tmp_path, force_run=False) == 100.0

    (cov_dir / "cpp_coverage.json").write_text(
        json.dumps(
            {
                "data": [
                    {
                        "totals": {
                            "lines": {"count": 50, "covered": 45, "percent": 90.0},
                            "functions": {"count": 10, "covered": 8, "percent": 80.0},
                            "branches": {"count": 8, "covered": 6, "percent": 75.0},
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert pc.calculate_cpp_test_coverage(tmp_path, force_run=False) == 75.0


def test_hooks_and_cli_dispatch(tmp_path: Path) -> None:
    """Tests stage hooks (`fmt`, `build`, `clippy`, `test`, `shields`, `all`) and `main`.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
    """
    (tmp_path / "cpp" / "src").mkdir(parents=True)
    (tmp_path / "cpp" / "src" / "sample.cpp").write_text(f"int x = 0;{NL}", encoding="utf-8")

    with patch.object(pc, "run_command") as mock_run:
        assert pc.hook_fmt(tmp_path, check_only=False) == 0
        assert mock_run.call_count == 6

    with patch.object(pc, "run_command") as mock_run_no_cpp:
        empty_ws = tmp_path / "no_cpp"
        empty_ws.mkdir()
        assert pc.hook_fmt(empty_ws, check_only=True) == 0
        assert mock_run_no_cpp.call_count == 2
        assert pc.hook_fmt(empty_ws, check_only=False) == 0
        assert mock_run_no_cpp.call_count == 6

    with patch.object(pc, "run_command") as mock_build:
        assert pc.hook_build(tmp_path) == 0
        assert mock_build.call_count == 3

    with patch.object(pc, "run_command") as mock_lint:
        assert pc.hook_clippy(tmp_path) == 0
        assert mock_lint.call_count == 5

    with patch.object(pc, "run_command") as mock_test:
        assert pc.hook_test(tmp_path) == 0
        assert mock_test.call_count == 5

    (tmp_path / "cpp" / "build").mkdir(parents=True)
    with patch.object(pc, "run_command") as mock_test_existing:
        assert pc.hook_test(tmp_path) == 0
        assert mock_test_existing.call_count == 3

    (tmp_path / "README.md").write_text(
        f"# Title{NL}{NL}"
        f"[![% doc coverage](https://img.shields.io/badge/doc%20coverage-100%25-brightgreen.svg)](#verification--quality-gates){NL}"
        f"[![% test coverage](https://img.shields.io/badge/test%20coverage-100%25-brightgreen.svg)](#verification--quality-gates){NL}",
        encoding="utf-8",
    )
    with (
        patch.object(pc, "calculate_doc_coverage", return_value=100.0),
        patch.object(pc, "calculate_test_coverage", return_value=100.0),
    ):
        assert pc.hook_shields(tmp_path, check_only=True, force_run=False) == 0

    with (
        patch.object(pc, "calculate_doc_coverage", return_value=95.0),
        patch.object(pc, "calculate_test_coverage", return_value=100.0),
    ):
        assert pc.hook_shields(tmp_path, check_only=False, force_run=False) == 1

    (tmp_path / "README.md").write_text(f"# Outdated{NL}", encoding="utf-8")
    with (
        patch.object(pc, "calculate_doc_coverage", return_value=100.0),
        patch.object(pc, "calculate_test_coverage", return_value=100.0),
    ):
        assert pc.hook_shields(tmp_path, check_only=True, force_run=False) == 1

    with (
        patch.object(pc, "hook_fmt", return_value=0),
        patch.object(pc, "hook_build", return_value=0),
        patch.object(pc, "hook_clippy", return_value=0),
        patch.object(pc, "hook_abi", return_value=0),
        patch.object(pc, "hook_test", return_value=0),
        patch.object(pc, "hook_shields", return_value=0),
    ):
        assert pc.hook_all(tmp_path) == 0

    with (
        patch.object(pc, "hook_fmt", return_value=0),
        patch.object(pc, "hook_build", return_value=2),
    ):
        assert pc.hook_all(tmp_path) == 2

    with (
        patch.object(pc, "hook_fmt", return_value=0),
        patch.object(pc, "hook_build", return_value=0),
        patch.object(pc, "hook_clippy", return_value=0),
        patch.object(pc, "hook_abi", return_value=0),
        patch.object(pc, "hook_test", return_value=0),
        patch.object(pc, "hook_shields", return_value=0),
        patch.object(pc, "hook_all", return_value=0),
    ):
        assert pc.main(["fmt", "--check"]) == 0
        assert pc.main(["build"]) == 0
        assert pc.main(["clippy"]) == 0
        assert pc.main(["lint"]) == 0
        assert pc.main(["abi"]) == 0
        assert pc.main(["test"]) == 0
        assert pc.main(["shields"]) == 0
        assert pc.main(["all"]) == 0

    def raise_called_process_error(repo_root: Path, check_only: bool = False) -> int:
        """Raises `subprocess.CalledProcessError` for testing error handling in `main`.

        Args:
            repo_root: Repository root path.
            check_only: Check-only flag.

        Returns:
            Never returns; always raises `CalledProcessError`.
        """
        _ = (repo_root, check_only)
        raise subprocess.CalledProcessError(returncode=3, cmd=["cargo", "fmt"])

    with patch.object(pc, "hook_fmt", side_effect=raise_called_process_error):
        assert pc.main(["fmt"]) == 3


def test_abi_compatibility_hook(tmp_path: Path) -> None:
    """Tests `hook_abi` and Google `libtpu` binary ABI compatibility verifiers.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    import ctypes

    real_root = pc.get_repo_root()
    assert pc.verify_rust_and_c_abi_sources(real_root) == []
    real_cdylib = pc.resolve_compiled_cdylib(real_root)
    assert real_cdylib is not None
    assert pc.verify_binary_vtable_abi(real_cdylib) == []
    assert pc.hook_abi(real_root, check_only=True) == 0

    # Test resolve_compiled_cdylib when no library exists vs when one exists
    assert pc.resolve_compiled_cdylib(tmp_path) is None
    fake_lib = tmp_path / "target" / "debug" / "libtpu.dylib"
    fake_lib.parent.mkdir(parents=True, exist_ok=True)
    fake_lib.write_text("stub", encoding="utf-8")
    assert pc.resolve_compiled_cdylib(tmp_path) == fake_lib

    # Test verify_rust_and_c_abi_sources failure branches
    assert pc.verify_rust_and_c_abi_sources(tmp_path) == [
        "Missing crates/libtpu-c-api/src or cpp/src/libtpu.h"
    ]
    c_src = tmp_path / "crates" / "libtpu-c-api" / "src"
    c_src.mkdir(parents=True, exist_ok=True)
    (c_src / "lib.rs").write_text("// empty .unwrap()", encoding="utf-8")
    h_dir = tmp_path / "cpp" / "src"
    h_dir.mkdir(parents=True, exist_ok=True)
    (h_dir / "libtpu.h").write_text("/* empty */", encoding="utf-8")
    py_init = tmp_path / "python" / "libtpu"
    py_init.mkdir(parents=True, exist_ok=True)
    (py_init / "__init__.py").write_text("# empty", encoding="utf-8")
    assert pc.verify_rust_and_c_abi_sources(tmp_path) != []
    core_src = tmp_path / "crates" / "libtpu-core" / "src"
    core_src.mkdir(parents=True, exist_ok=True)
    (core_src / "error.rs").write_text("// missing enum", encoding="utf-8")
    (tmp_path / "Cargo.toml").write_text("anyhow = '1.0'", encoding="utf-8")
    src_errs = pc.verify_rust_and_c_abi_sources(tmp_path)
    assert any("missing from crates/libtpu-c-api/src" in e for e in src_errs)
    assert any("missing from cpp/src/libtpu.h" in e for e in src_errs)
    assert any("size_of assertion missing" in e for e in src_errs)
    assert any("Missing Python wheel API module" in e for e in src_errs)
    assert any("Python wheel API symbol" in e for e in src_errs)
    assert any("Forbidden dependency anyhow/thiserror" in e for e in src_errs)
    assert any("LibTpuError in error.rs must derive" in e for e in src_errs)
    assert any("Forbidden .unwrap() or .expect()" in e for e in src_errs)

    # Test hook_abi source error branch
    assert pc.hook_abi(tmp_path) == 1

    # Test hook_abi when cdylib is built on demand vs missing after build vs binary errors
    with (
        patch.object(pc, "verify_rust_and_c_abi_sources", return_value=[]),
        patch.object(pc, "resolve_compiled_cdylib", side_effect=[None, None]),
        patch.object(pc, "run_command"),
    ):
        assert pc.hook_abi(tmp_path) == 1

    with (
        patch.object(pc, "verify_rust_and_c_abi_sources", return_value=[]),
        patch.object(pc, "resolve_compiled_cdylib", return_value=real_cdylib),
        patch.object(pc, "verify_binary_vtable_abi", return_value=["bad vtable"]),
    ):
        assert pc.hook_abi(tmp_path) == 1

    # Test verify_binary_vtable_abi error branches using synthetic ctypes buffers
    class _MissingSymLib:
        """Stub CDLL missing exported symbols."""

    with patch.object(ctypes, "CDLL", return_value=_MissingSymLib()):
        errs = pc.verify_binary_vtable_abi(fake_lib)
        assert any("missing from dynamic symbol table" in e for e in errs)

    class _NullApiLib:
        """Stub CDLL returning NULL from GetPjrtApi / GetLibtpuSdkApi."""

        def __getattr__(self, _name: str) -> object:
            """Returns a callable returning 0 for any symbol lookup.

            Args:
                _name: Attribute name.

            Returns:
                Callable returning 0.
            """
            return lambda *_a: 0

    with patch.object(ctypes, "CDLL", return_value=_NullApiLib()):
        errs = pc.verify_binary_vtable_abi(fake_lib)
        assert errs == ["GetPjrtApi() or GetLibtpuSdkApi() returned NULL"]

    bad_pjrt_buf = (ctypes.c_uint64 * 143)()
    bad_pjrt_buf[0] = 999
    bad_sdk_buf = (ctypes.c_uint64 * 18)()
    bad_drv_fn_buf = (ctypes.c_uint64 * 28)()
    bad_drv_buf = (ctypes.c_uint64 * 2)(ctypes.addressof(bad_drv_fn_buf), 0)

    class _CorruptVtableLib:
        """Stub CDLL with corrupt vtable headers and NULL slots."""

        def __init__(self, drv_ret: int) -> None:
            """Initializes the corrupt vtable stub.

            Args:
                drv_ret: Value returned by `TpuDriver_Open`.
            """
            self._drv_ret = drv_ret
            for sym in pc.GOOGLE_LIBTPU_EXPORTED_SYMBOLS:
                setattr(self, sym, lambda *_a: 0)
            self.GetPjrtApi = lambda: ctypes.addressof(bad_pjrt_buf)
            self.GetLibtpuSdkApi = lambda: ctypes.addressof(bad_sdk_buf)
            self.TpuDriver_Open = lambda _w: self._drv_ret
            self.TpuDriver_Close = lambda _d: None

    with patch.object(ctypes, "CDLL", return_value=_CorruptVtableLib(0)):
        errs = pc.verify_binary_vtable_abi(fake_lib)
        assert any("PJRT_Api header/slots mismatch" in e for e in errs)
        assert any("PJRT_Extension_Base chain mismatch" in e for e in errs)
        assert any("LibtpuSdk_Api mismatch" in e for e in errs)
        assert any("TpuDriver_Open returned NULL" in e for e in errs)

    with patch.object(
        ctypes, "CDLL", return_value=_CorruptVtableLib(ctypes.addressof(bad_drv_buf))
    ):
        errs = pc.verify_binary_vtable_abi(fake_lib)
        assert any("TpuDriverFn null slots" in e for e in errs)


def test_github_actions_workflows_present_and_complete() -> None:
    """Verifies `.github/workflows/ci.yml` and `release.yml` exist and cover all hooks."""
    root = pc.get_repo_root()
    ci_path = root / ".github" / "workflows" / "ci.yml"
    release_path = root / ".github" / "workflows" / "release.yml"
    assert ci_path.exists()
    assert release_path.exists()

    ci_text = ci_path.read_text(encoding="utf-8")
    for stage in ("fmt", "build", "clippy", "test", "all"):
        assert f"scripts/pre_commit.py {stage}" in ci_text
    assert "pre-commit run --all-files" in ci_text

    release_text = release_path.read_text(encoding="utf-8")
    assert '"v*"' in release_text
    assert ".whl" in release_text
    for target in (
        "x86_64-unknown-linux-gnu",
        "aarch64-unknown-linux-gnu",
        "aarch64-apple-darwin",
        "x86_64-pc-windows-msvc",
    ):
        assert target in release_text
    assert "x86_64-apple-darwin" not in release_text

    readme_text = (root / "README.md").read_text(encoding="utf-8")
    assert "actions/workflows/ci.yml/badge.svg" in readme_text
    assert "actions/workflows/release.yml/badge.svg" in readme_text


def _build_synthetic_elf64() -> bytes:
    """Constructs a minimal valid 64-bit little-endian ELF image with `.dynsym` and `.rela.dyn`.

    Returns:
        Raw ELF64 byte payload containing an exported `GetPjrtApi` function symbol and relocations.
    """
    import struct

    import scripts.re_extract as re_ext

    shstrtab = b"\x00.shstrtab\x00.dynstr\x00.dynsym\x00.rela.dyn\x00"
    dynstr = b"\x00GetPjrtApi\x00"
    sym_null = struct.pack("<IBBHQQ", 0, 0, 0, 0, 0, 0)
    sym_func = struct.pack(
        "<IBBHQQ",
        1,
        (re_ext.STB_GLOBAL << 4) | re_ext.STT_FUNC,
        0,
        1,
        0x0B4F9B90,
        5,
    )
    sym_weak_empty = struct.pack(
        "<IBBHQQ",
        0,
        (re_ext.STB_WEAK << 4) | re_ext.STT_FUNC,
        0,
        1,
        0x1000,
        4,
    )
    sym_not_func = struct.pack("<IBBHQQ", 1, 0, 0, 0, 0, 0)
    dynsym = sym_null + sym_func + sym_weak_empty + sym_not_func

    rela_rel = struct.pack("<QQQ", 0x3000, re_ext.R_X86_64_RELATIVE, 0x0B4F9B90)
    rela_other = struct.pack("<QQQ", 0x3008, 1, 0)
    rela_dyn = rela_rel + rela_other

    ehdr_size = 64
    off_shstrtab = ehdr_size
    off_dynstr = off_shstrtab + len(shstrtab)
    off_dynsym = off_dynstr + len(dynstr)
    off_rela = off_dynsym + len(dynsym)
    off_shdr = off_rela + len(rela_dyn)

    sh_null = struct.pack("<IIQQQQIIQQ", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    sh_shstr = struct.pack(
        "<IIQQQQIIQQ",
        1,
        re_ext.SHT_STRTAB,
        0,
        0,
        off_shstrtab,
        len(shstrtab),
        0,
        0,
        1,
        0,
    )
    sh_dynstr = struct.pack(
        "<IIQQQQIIQQ",
        11,
        re_ext.SHT_STRTAB,
        0,
        0,
        off_dynstr,
        len(dynstr),
        0,
        0,
        1,
        0,
    )
    sh_dynsym = struct.pack(
        "<IIQQQQIIQQ",
        19,
        re_ext.SHT_DYNSYM,
        0,
        0,
        off_dynsym,
        len(dynsym),
        2,
        0,
        8,
        24,
    )
    sh_rela = struct.pack(
        "<IIQQQQIIQQ",
        27,
        re_ext.SHT_RELA,
        0,
        0,
        off_rela,
        len(rela_dyn),
        0,
        0,
        8,
        24,
    )

    e_ident = (
        re_ext.ELF_MAGIC + bytes([re_ext.ELFCLASS64, re_ext.ELFDATA2LSB, 1, 0]) + (b"\x00" * 8)
    )
    ehdr_tail = struct.pack(
        "<HHIQQQIHHHHHH",
        3,
        62,
        1,
        0,
        0,
        off_shdr,
        0,
        ehdr_size,
        0,
        0,
        64,
        5,
        1,
    )
    return (
        e_ident
        + ehdr_tail
        + shstrtab
        + dynstr
        + dynsym
        + rela_dyn
        + sh_null
        + sh_shstr
        + sh_dynstr
        + sh_dynsym
        + sh_rela
    )


def test_re_extract_forensics_and_validation(tmp_path: Path) -> None:
    """Tests `scripts/re_extract.py` ELF64 parsing, wheel inspection, and manifest validation.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    import io
    import struct
    import urllib.request
    import zipfile

    import pytest

    import scripts.re_extract as re_ext

    assert re_ext.compute_sha256_bytes(b"abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )

    class _FakeResp(io.BytesIO):
        """Fake HTTP response context manager."""

        def __enter__(self) -> _FakeResp:
            """Enters context manager.

            Returns:
                Self.
            """
            return self

        def __exit__(self, *_args: object) -> None:
            """Exits context manager.

            Args:
                *_args: Exception arguments.
            """

    fake_pypi_json = json.dumps(
        {
            "urls": [
                {"filename": "libtpu-0.0.50.tar.gz", "url": "https://x/sdist", "digests": {}},
                {
                    "filename": "libtpu-0.0.50-cp311-manylinux_2_31_x86_64.whl",
                    "url": "https://files.pythonhosted.org/libtpu.whl",
                    "digests": {"sha256": "deadbeef"},
                },
            ]
        }
    ).encode("utf-8")
    with patch.object(
        urllib.request, "urlopen", side_effect=lambda *_a, **_kw: _FakeResp(fake_pypi_json)
    ):
        url, sha = re_ext.fetch_pypi_wheel_url("0.0.50", "x86_64")
        assert url == "https://files.pythonhosted.org/libtpu.whl"
        assert sha == "deadbeef"
        with pytest.raises(ValueError, match="No libtpu==0.0.50 wheel found"):
            re_ext.fetch_pypi_wheel_url("0.0.50", "s390x")

    with pytest.raises(ValueError, match="Invalid ELF header"):
        re_ext.parse_elf64_sections_and_dynsym(b"not-an-elf")

    bad_class = re_ext.ELF_MAGIC + bytes([1, 1]) + (b"\x00" * 58)
    with pytest.raises(ValueError, match="Unsupported ELF class"):
        re_ext.parse_elf64_sections_and_dynsym(bad_class)

    no_shdr = (
        re_ext.ELF_MAGIC
        + bytes([re_ext.ELFCLASS64, re_ext.ELFDATA2LSB, 1, 0])
        + (b"\x00" * 8)
        + struct.pack("<HHIQQQIHHHHHH", 3, 62, 1, 0, 0, 0, 0, 64, 0, 0, 64, 0, 0)
    )
    assert re_ext.parse_elf64_sections_and_dynsym(no_shdr) == ([], [])

    oob_shdr = (
        re_ext.ELF_MAGIC
        + bytes([re_ext.ELFCLASS64, re_ext.ELFDATA2LSB, 1, 0])
        + (b"\x00" * 8)
        + struct.pack("<HHIQQQIHHHHHH", 3, 62, 1, 0, 0, 4096, 0, 64, 0, 0, 64, 2, 99)
    )
    with pytest.raises(ValueError, match="Malformed ELF64 binary"):
        re_ext.parse_elf64_sections_and_dynsym(oob_shdr)

    elf_bytes = _build_synthetic_elf64()
    sections, symbols = re_ext.parse_elf64_sections_and_dynsym(elf_bytes)
    assert any(s.name == ".dynsym" for s in sections)
    assert len(symbols) == 1
    assert symbols[0].name == "GetPjrtApi"
    assert symbols[0].vaddr == 0x0B4F9B90

    relocs = re_ext.resolve_elf64_rela_relocations(elf_bytes)
    assert len(relocs) == 1
    assert relocs[0].offset == 0x3000
    assert relocs[0].addend == 0x0B4F9B90

    word_map = {
        0x1000: 1144,
        0x1008: 0x2000,
        0x1010: 24,
        0x1018: 116 << 32,
        0x1028: 0x0C5A0F90,
        0x2000: 32,
        0x2008: 28,
        0x2010: 0,
        0x2018: 0x0C5B0000,
    }
    vtable = re_ext.reconstruct_pjrt_and_extension_vtables(word_map, 0x1000, num_slots=1)
    assert vtable["struct_size"] == 1144
    assert vtable["minor_version"] == 116
    assert len(vtable["extensions"]) == 1  # type: ignore[arg-type]

    ioc = re_ext.decode_linux_ioctl_command(0x4020DC08)
    assert ioc == {"dir": 1, "size": 32, "type": 0xDC, "nr": 8}
    cpp_vt = re_ext.reconstruct_itanium_vtable(
        {0x5000: 0, 0x5008: 0x6000, 0x5010: 0x7000}, 0x5000, 1
    )
    assert cpp_vt["typeinfo_vaddr"] == 0x6000
    assert cpp_vt["methods"] == [0x7000]

    code1 = bytes(
        [
            0x48,
            0x83,
            0x3F,
            0x28,
            0x48,
            0x8B,
            0x07,
            0x48,
            0x8B,
            0x47,
            0x10,
            0x48,
            0x89,
            0x47,
            0x18,
            0x8B,
            0x47,
            0x20,
            0x0F,
            0xB6,
            0x47,
            0x24,
            0x90,
            0xC3,
        ]
    )
    rep1 = re_ext.disassemble_x86_64_prologue(code1)
    assert rep1.expected_struct_size == 40
    assert len(rep1.field_offsets) == 5

    code2 = bytes(
        [
            0xBE,
            0x58,
            0x00,
            0x00,
            0x00,
            0xBE,
            0x60,
            0x00,
            0x00,
            0x00,
            0x48,
            0x81,
            0x3F,
            0x90,
            0x00,
            0x00,
            0x00,
            0x48,
            0x8B,
            0x87,
            0x00,
            0x4C,
            0x01,
            0x00,
            0x8B,
            0x00,
            0x0F,
            0xB6,
            0x00,
            0x00,
            0x0F,
            0x1F,
            0x00,
            0x00,
        ]
    )
    rep2 = re_ext.disassemble_x86_64_prologue(code2)
    assert rep2.expected_struct_size == 144

    proto_sec = (
        b"\x00\x0a\x02ab\x0a\x0exla_data.proto\x12\x03xla"
        b"\x0a\x09hlo.proto"
        b"\x0a\x0dbad_pkg.proto\x12\xff"
        b"\x0a\x08notproto\x00"
    )
    carved = re_ext.carve_protodesc_cold(proto_sec)
    assert len(carved) == 3
    assert carved[0] == {"name": "xla_data.proto", "package": "xla"}
    assert carved[1] == {"name": "hlo.proto", "package": ""}

    blob = b"\x00tpu_tiles.cfg\x00TILE_8x128"
    toc = (
        struct.pack("<QQQ", 0, 0, 0)
        + struct.pack("<QQQ", 999, 0, 4)
        + struct.pack("<QQQ", 1, 15, 10)
        + struct.pack("<QQQ", 15, 15, 10)
    )
    toc_entries = re_ext.extract_filewrapper_toc(toc, blob)
    assert len(toc_entries) == 2
    assert toc_entries[0] == ("tpu_tiles.cfg", b"TILE_8x128")

    whl_path = tmp_path / "libtpu-0.0.50.whl"
    with zipfile.ZipFile(whl_path, "w") as zf:
        zf.writestr("libtpu/__init__.py", "# init")
        zf.writestr("libtpu/libtpu.so", elf_bytes)

    summary = re_ext.inspect_wheel_archive(whl_path)
    assert summary["wheel"] == "libtpu-0.0.50.whl"
    assert "libtpu/libtpu.so" in summary["binaries"]  # type: ignore[operator]

    real_root = pc.get_repo_root()
    assert re_ext.validate_golden_manifests(real_root) == []
    assert re_ext.main(["--repo-root", str(real_root), "--wheel", str(whl_path)]) == 0
    assert re_ext.main(["--repo-root", str(real_root), "--verify-golden"]) == 0

    assert re_ext.validate_golden_manifests(tmp_path) == [
        "Missing golden_binary_manifest.json or golden_struct_layouts.json"
    ]
    assert re_ext.main(["--repo-root", str(tmp_path)]) == 1

    bad_re = tmp_path / "docs" / "reverse_engineering"
    bad_re.mkdir(parents=True)
    (bad_re / "golden_binary_manifest.json").write_text("{}", encoding="utf-8")
    (bad_re / "golden_struct_layouts.json").write_text("{}", encoding="utf-8")
    bad_errs = re_ext.validate_golden_manifests(tmp_path)
    assert len(bad_errs) >= 10

    mismatch_manifest = {
        "libtpu_so": {"pjrt_vtable_slots": [{"min_struct_size": 24}] * 138},
    }
    mismatch_layouts = {
        "pjrt_slots": [{"min_struct_size": None}] * 138,
    }
    (bad_re / "golden_binary_manifest.json").write_text(
        json.dumps(mismatch_manifest), encoding="utf-8"
    )
    (bad_re / "golden_struct_layouts.json").write_text(
        json.dumps(mismatch_layouts), encoding="utf-8"
    )
    (bad_re / "golden_proto_descriptors.json").write_text(
        json.dumps({"topologically_closed": False, "descriptors": []}), encoding="utf-8"
    )
    (bad_re / "csr_register_map.json").write_text(json.dumps({"generations": {}}), encoding="utf-8")
    mismatch_errs = re_ext.validate_golden_manifests(tmp_path)
    assert any("min_struct_size mismatch" in e for e in mismatch_errs)
    assert any("incomplete descriptor closure" in e for e in mismatch_errs)
    assert any("missing TPU generation CSR maps" in e for e in mismatch_errs)

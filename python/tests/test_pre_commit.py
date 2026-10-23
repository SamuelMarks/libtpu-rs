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

    py_pkg = tmp_path / "python" / "libtpu"
    py_pkg.mkdir(parents=True)
    (py_pkg / "mod.py").write_text(
        f'"""Module doc."""{NL}{NL}'
        f'class Cls:{NL}    """Class doc."""{NL}{NL}'
        f'def fn() -> None:{NL}    """Fn doc."""{NL}',
        encoding="utf-8",
    )
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "helper.py").write_text(
        f"def undoc() -> None:{NL}    pass{NL}", encoding="utf-8"
    )
    py_docs, py_total = pc.calculate_python_doc_coverage(tmp_path)
    assert (py_docs, py_total) == (3, 5)

    xml_dir = tmp_path / "cpp" / "cpp" / "build" / "doxygen_xml"
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
                    <detaileddescription></detaileddescription>
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
    assert (cpp_docs, cpp_total) == (2, 4)

    with patch.object(pc, "run_command"):
        cpp_docs2, cpp_total2 = pc.calculate_cpp_doc_coverage(tmp_path, force_run=True)
        assert (cpp_docs2, cpp_total2) == (2, 4)

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
        patch.object(pc, "hook_test", return_value=0),
        patch.object(pc, "hook_shields", return_value=0),
        patch.object(pc, "hook_all", return_value=0),
    ):
        assert pc.main(["fmt", "--check"]) == 0
        assert pc.main(["build"]) == 0
        assert pc.main(["clippy"]) == 0
        assert pc.main(["lint"]) == 0
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

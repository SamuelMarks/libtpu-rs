"""PEP 517 `maturin` build backend bundling `libtpu._C`, `libtpu/libtpu.so`, and `libtpu/sdk.so`.

Wraps upstream `maturin` PEP 517 hooks so `pip install .` builds the native `PyO3` (`abi3`)
extension (`libtpu._C`) and guarantees that `libtpu/libtpu.so` (`GetPjrtApi`) and `libtpu/sdk.so`
(`PyInit_sdk`) are present at the exact paths expected by `jax`, `jaxlib`, and `torch_xla`, with
recomputed PEP 427 `.dist-info/RECORD` SHA-256 digests.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
import subprocess
import zipfile
from collections.abc import Mapping
from pathlib import Path

import maturin


def _urlsafe_sha256_digest(payload: bytes) -> str:
    """Computes a PEP 427 `RECORD` `sha256=<urlsafe-b64>` digest without `=` padding.

    Args:
        payload: Raw file bytes stored in the wheel archive.

    Returns:
        Formatted `sha256=<digest>` string.
    """
    raw_digest = hashlib.sha256(payload).digest()
    encoded = base64.urlsafe_b64encode(raw_digest).rstrip(b"=").decode("ascii")
    return f"sha256={encoded}"


def _resolve_cargo_target_dir(repo_root: Path) -> Path:
    """Resolves the active Cargo target directory honoring `CARGO_TARGET_DIR` and Cargo metadata.

    Args:
        repo_root: Repository root directory containing `Cargo.toml`.

    Returns:
        Resolved Cargo target directory `Path`.
    """
    env_dir = os.environ.get("CARGO_TARGET_DIR", "").strip()
    if env_dir:
        return Path(env_dir)
    proc = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0 and '"target_directory":' in proc.stdout:
        import json

        meta = json.loads(proc.stdout)
        target_dir_str = meta.get("target_directory")
        if isinstance(target_dir_str, str) and target_dir_str:
            return Path(target_dir_str)
    return repo_root / "target"


def _find_or_build_pjrt_cdylib(repo_root: Path) -> tuple[str, bytes]:
    """Finds an existing `libtpu-c-api` shared library or compiles one via `cargo build`.

    Args:
        repo_root: Repository root directory containing `Cargo.toml`.

    Returns:
        Tuple `(native_filename, binary_bytes)` for the compiled PJRT shared library.

    Raises:
        RuntimeError: If `cargo build -p libtpu-c-api` fails or produces no shared library.
    """
    candidate_names = ("libtpu.so", "libtpu.dylib", "tpu.dll")
    target_dir = _resolve_cargo_target_dir(repo_root)
    search_dirs = [
        repo_root / "python" / "libtpu",
        repo_root / "target" / "release",
        repo_root / "target" / "debug",
        target_dir / "release",
        target_dir / "debug",
    ]
    for directory in search_dirs:
        for name in candidate_names:
            candidate = directory / name
            if candidate.is_file():
                return name, candidate.read_bytes()

    subprocess.run(
        ["cargo", "build", "-p", "libtpu-c-api", "--release"],
        cwd=repo_root,
        check=True,
    )
    for directory in (target_dir / "release", repo_root / "target" / "release"):
        for name in candidate_names:
            candidate = directory / name
            if candidate.is_file():
                return name, candidate.read_bytes()
    raise RuntimeError("Failed to locate compiled libtpu-c-api shared library after cargo build")


def inject_dropin_binaries_into_wheel(wheel_path: Path, repo_root: Path) -> None:
    """Bundles `libtpu/libtpu.so` and `libtpu/sdk.so` into `wheel_path` with updated `RECORD`.

    Args:
        wheel_path: Path to the `.whl` archive produced by `maturin`.
        repo_root: Repository root directory used to locate `libtpu-c-api` artifacts.

    Raises:
        RuntimeError: If the wheel contains neither a `_C` extension nor a `RECORD` manifest.
    """
    with zipfile.ZipFile(wheel_path, "r") as zin:
        entries: dict[str, bytes] = {name: zin.read(name) for name in zin.namelist()}

    record_name = next((n for n in entries if n.endswith(".dist-info/RECORD")), None)
    if record_name is None:
        raise RuntimeError(f"Wheel {wheel_path} is missing .dist-info/RECORD")

    c_ext_name = next(
        (
            n
            for n in entries
            if n.startswith("libtpu/_C") and n.endswith((".so", ".pyd", ".dylib", ".dll"))
        ),
        None,
    )
    if c_ext_name is None:
        raise RuntimeError(f"Wheel {wheel_path} is missing compiled libtpu._C native extension")

    if "libtpu/sdk.so" not in entries:
        entries["libtpu/sdk.so"] = entries[c_ext_name]

    if "libtpu/libtpu.so" not in entries:
        native_name, pjrt_bytes = _find_or_build_pjrt_cdylib(repo_root)
        entries["libtpu/libtpu.so"] = pjrt_bytes
        native_arc_path = f"libtpu/{native_name}"
        if native_arc_path not in entries:
            entries[native_arc_path] = pjrt_bytes

    record_buf = io.StringIO()
    writer = csv.writer(record_buf, lineterminator=chr(10))
    for arc_name in sorted(entries):
        if arc_name == record_name:
            continue
        blob = entries[arc_name]
        writer.writerow([arc_name, _urlsafe_sha256_digest(blob), str(len(blob))])
    writer.writerow([record_name, "", ""])
    entries[record_name] = record_buf.getvalue().encode("utf-8")

    tmp_wheel = wheel_path.with_suffix(".whl.tmp")
    with zipfile.ZipFile(tmp_wheel, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for arc_name in sorted(entries):
            zout.writestr(arc_name, entries[arc_name])
    tmp_wheel.replace(wheel_path)


def get_requires_for_build_wheel(config_settings: Mapping[str, object] | None = None) -> list[str]:
    """Returns extra build requirements for wheel builds via `maturin`.

    Args:
        config_settings: Optional PEP 517 configuration dictionary.

    Returns:
        List of extra PEP 508 requirement strings.
    """
    return list(maturin.get_requires_for_build_wheel(config_settings))


def get_requires_for_build_sdist(config_settings: Mapping[str, object] | None = None) -> list[str]:
    """Returns extra build requirements for sdist builds via `maturin`.

    Args:
        config_settings: Optional PEP 517 configuration dictionary.

    Returns:
        List of extra PEP 508 requirement strings.
    """
    return list(maturin.get_requires_for_build_sdist(config_settings))


def get_requires_for_build_editable(
    config_settings: Mapping[str, object] | None = None,
) -> list[str]:
    """Returns extra build requirements for editable installs via `maturin`.

    Args:
        config_settings: Optional PEP 517 configuration dictionary.

    Returns:
        List of extra PEP 508 requirement strings.
    """
    return list(maturin.get_requires_for_build_editable(config_settings))


def prepare_metadata_for_build_wheel(
    metadata_directory: str,
    config_settings: Mapping[str, object] | None = None,
) -> str:
    """Prepares `.dist-info` metadata directory for wheel builds via `maturin`.

    Args:
        metadata_directory: Target directory where `.dist-info` is created.
        config_settings: Optional PEP 517 configuration dictionary.

    Returns:
        Basename of the created `.dist-info` directory.
    """
    return str(maturin.prepare_metadata_for_build_wheel(metadata_directory, config_settings))


def prepare_metadata_for_build_editable(
    metadata_directory: str,
    config_settings: Mapping[str, object] | None = None,
) -> str:
    """Prepares `.dist-info` metadata directory for editable installs via `maturin`.

    Args:
        metadata_directory: Target directory where `.dist-info` is created.
        config_settings: Optional PEP 517 configuration dictionary.

    Returns:
        Basename of the created `.dist-info` directory.
    """
    return str(maturin.prepare_metadata_for_build_editable(metadata_directory, config_settings))


def build_wheel(
    wheel_directory: str,
    config_settings: Mapping[str, object] | None = None,
    metadata_directory: str | None = None,
) -> str:
    """Builds a `libtpu` wheel bundling `libtpu._C`, `libtpu/libtpu.so`, and `libtpu/sdk.so`.

    Args:
        wheel_directory: Destination directory for the built `.whl` file.
        config_settings: Optional PEP 517 configuration dictionary.
        metadata_directory: Optional pre-generated `.dist-info` directory path.

    Returns:
        Basename of the built `.whl` archive inside `wheel_directory`.
    """
    wheel_filename = str(maturin.build_wheel(wheel_directory, config_settings, metadata_directory))
    repo_root = Path(__file__).resolve().parent.parent
    inject_dropin_binaries_into_wheel(Path(wheel_directory) / wheel_filename, repo_root)
    return wheel_filename


def build_editable(
    wheel_directory: str,
    config_settings: Mapping[str, object] | None = None,
    metadata_directory: str | None = None,
) -> str:
    """Builds an editable `libtpu` wheel via `maturin`.

    Args:
        wheel_directory: Destination directory for the editable `.whl` file.
        config_settings: Optional PEP 517 configuration dictionary.
        metadata_directory: Optional pre-generated `.dist-info` directory path.

    Returns:
        Basename of the built editable `.whl` archive inside `wheel_directory`.
    """
    return str(maturin.build_editable(wheel_directory, config_settings, metadata_directory))


def build_sdist(
    sdist_directory: str,
    config_settings: Mapping[str, object] | None = None,
) -> str:
    """Builds a source distribution (`.tar.gz`) archive via `maturin`.

    Args:
        sdist_directory: Destination directory for the built `.tar.gz` archive.
        config_settings: Optional PEP 517 configuration dictionary.

    Returns:
        Basename of the built `.tar.gz` archive inside `sdist_directory`.
    """
    return str(maturin.build_sdist(sdist_directory, config_settings))

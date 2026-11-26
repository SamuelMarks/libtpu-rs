#!/usr/bin/env python3
"""Deterministic Wheel & Binary Artifact Fetcher and Differ for `libtpu-rs`.

Resolves, downloads, verifies SHA-256 checksums, extracts `libtpu/libtpu.so` and
`libtpu/sdk.so` into `.cache/wheels/`, and diffs `.dynsym` C ABI exports, `PJRT_Api`
versions, and `protodesc_cold` schema inventories across official `libtpu` releases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import struct
import sys
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.re_extract import (
    ElfSectionHeader as ElfSectionHeader,
)
from scripts.re_extract import (
    carve_protodesc_cold,
    parse_elf64_sections_and_dynsym,
)

PYPI_LIBTPU_JSON_URL: str = "https://pypi.org/pypi/libtpu/json"
GCS_LIBTPU_INDEX_URL: str = "https://storage.googleapis.com/jax-releases/libtpu_releases.html"

TARGET_PLATFORMS: tuple[str, ...] = ("manylinux_2_31_x86_64",)

SUPPORTED_PLATFORM_TAGS: tuple[str, ...] = (
    "manylinux_2_31_x86_64",
    "manylinux_2_35_aarch64",
)

PINNED_LIBTPU_0_0_53_1_SHA256: dict[str, str] = {
    "libtpu-0.0.53.1-cp311-cp311-manylinux_2_31_x86_64.whl": (
        "e300a3ccd01933894103303c9886c6c0dc88fcd3166928cecfd1e36b54ae95d3"
    ),
    "libtpu-0.0.53.1-cp312-cp312-manylinux_2_31_x86_64.whl": (
        "54eff1c73872c1c5697d70e158b52e0c4a5286184c3582e2075d439b4ea028cf"
    ),
    "libtpu-0.0.53.1-cp313-cp313-manylinux_2_31_x86_64.whl": (
        "91544c2141c9e5310ab354d48446bb91fe927630b07003bee2b3675bb2b35446"
    ),
    "libtpu-0.0.53.1-cp314-cp314-manylinux_2_31_x86_64.whl": (
        "30d2c3ba20047080e6bae8dbacb62115a627780d7a8f9a66163fd085f032b5ba"
    ),
    "libtpu-0.0.53.1-cp314-cp314t-manylinux_2_31_x86_64.whl": (
        "838295bfc8560136d0ebe4eecb55786eb2d5031a38e3d921b7e58b1977fab0b8"
    ),
    "libtpu-0.0.53.1-cp315-cp315-manylinux_2_31_x86_64.whl": (
        "a84b8e09ceda97bd89fd06a86b9a05c43e5d351c4b9b1b11e927326dcad9ba0d"
    ),
    "libtpu-0.0.53.1-cp315-cp315t-manylinux_2_31_x86_64.whl": (
        "fa3ceb1df9c3395e8175b3917ebb4347202d4a03c69a24656026f7ff85abea7f"
    ),
}

PINNED_WHEEL_SHA256: dict[str, str] = dict(PINNED_LIBTPU_0_0_53_1_SHA256)

PINNED_BINARY_SHA256: dict[tuple[str, str], str] = {
    (
        "0.0.53.1",
        "libtpu/libtpu.so",
    ): "9b098d865d7ed58592646df3d7ebe988dadfb3f444011ddae99b230bbfcb3dc8",
    (
        "0.0.53.1",
        "libtpu/sdk.so",
    ): "630c772352b470e1f32b7459ac3d60790f8d4d5000747390ebdeb743c1f33186",
    (
        "0.0.53.1",
        "libtpu/sdk.so:freethreaded",
    ): "7fb29a9b394037ac61d984948afa55147c49248cf75cd6e518c445e4a80284f6",
}


class ArtifactFetchError(RuntimeError):
    """Raised when wheel resolution, checksum verification, or ELF inspection fails."""


@dataclass(frozen=True)
class WheelArtifactSpec:
    """Resolved metadata for a remote or cached `libtpu` wheel archive."""

    version: str
    filename: str
    url: str
    sha256: str
    platform_tag: str
    python_tag: str
    size_bytes: int


@dataclass(frozen=True)
class ExtractedBinaryDigest:
    """Verified digest and filesystem path for an extracted ELF shared library."""

    member_name: str
    extracted_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class WheelInspectionReport:
    """Binary ABI and embedded schema summary extracted from a `libtpu` wheel."""

    wheel_filename: str
    wheel_sha256: str
    version: str
    binaries: tuple[ExtractedBinaryDigest, ...]
    exported_symbols: tuple[str, ...]
    pjrt_struct_size: int
    pjrt_version_major: int
    pjrt_version_minor: int
    protodesc_proto_names: tuple[str, ...]


@dataclass(frozen=True)
class WheelBinaryDiff:
    """Structured multi-release diff between two `libtpu` wheel archives."""

    old_wheel: str
    new_wheel: str
    old_pjrt_version: tuple[int, int, int]
    new_pjrt_version: tuple[int, int, int]
    added_symbols: tuple[str, ...]
    removed_symbols: tuple[str, ...]
    added_protos: tuple[str, ...]
    removed_protos: tuple[str, ...]


UrlReader = Callable[[str], bytes]


def default_url_reader(url: str) -> bytes:
    """Fetches raw bytes from an HTTPS URL with a deterministic User-Agent header.

    Args:
        url: Fully-qualified `https://` URL to read.

    Returns:
        Raw response body bytes.

    Raises:
        ArtifactFetchError: If the URL scheme is not `https://`.
    """
    if not url.startswith("https://"):
        raise ArtifactFetchError(f"Refusing non-HTTPS artifact URL: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "libtpu-rs-artifact-fetcher/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data: bytes = resp.read()
        return data


def compute_sha256_bytes(payload: bytes) -> str:
    """Computes the lowercase hexadecimal SHA-256 digest of `payload`.

    Args:
        payload: Raw byte buffer to hash.

    Returns:
        64-character lowercase hexadecimal SHA-256 string.
    """
    return hashlib.sha256(payload).hexdigest()


def compute_sha256_file(path: Path) -> str:
    """Computes the streaming lowercase hexadecimal SHA-256 digest of a file on disk.

    Args:
        path: Filesystem path to hash.

    Returns:
        64-character lowercase hexadecimal SHA-256 string.
    """
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_wheel_tags(filename: str) -> tuple[str, str, str]:
    """Extracts `(version, python_tag, platform_tag)` from a PEP 427 wheel filename.

    Args:
        filename: PEP 427 wheel filename ending in `.whl`.

    Returns:
        Tuple of `(version, python_tag, platform_tag)`.

    Raises:
        ArtifactFetchError: If `filename` is not a valid 5-part or 6-part wheel name.
    """
    if not filename.endswith(".whl"):
        raise ArtifactFetchError(f"Invalid wheel filename (missing .whl suffix): {filename}")
    stem = filename[:-4]
    parts = stem.split("-")
    if len(parts) < 5:
        raise ArtifactFetchError(f"Malformed PEP 427 wheel filename: {filename}")
    version = parts[1]
    abi_tag = parts[-2]
    python_tag = abi_tag if abi_tag.endswith("t") and abi_tag.startswith("cp") else parts[-3]
    platform_tag = parts[-1]
    return version, python_tag, platform_tag


def resolve_pypi_wheels(
    version: str = "0.0.53.1",
    platform_tag: str = "manylinux_2_31_x86_64",
    python_tag: str | None = "cp311",
    pypi_json_url: str = PYPI_LIBTPU_JSON_URL,
    reader: UrlReader = default_url_reader,
) -> list[WheelArtifactSpec]:
    """Resolves matching `libtpu` wheel artifacts from the PyPI JSON API.

    Args:
        version: Target `libtpu` version string (or `"latest"` for latest stable).
        platform_tag: Target platform tag (`manylinux_2_31_x86_64` or `manylinux_2_35_aarch64`).
        python_tag: Optional CPython ABI tag filter (e.g., `"cp311"`), or `None` for all tags.
        pypi_json_url: PyPI JSON metadata endpoint URL.
        reader: Injectable URL byte fetcher.

    Returns:
        Sorted list of resolved `WheelArtifactSpec` entries.

    Raises:
        ArtifactFetchError: If the version is missing or no wheels match the requested filters.
    """
    payload = json.loads(reader(pypi_json_url).decode("utf-8"))
    target_version = str(payload["info"]["version"]) if version == "latest" else version
    releases = payload.get("releases", {})
    if target_version not in releases:
        raise ArtifactFetchError(f"Version '{target_version}' not found in PyPI releases")

    resolved: list[WheelArtifactSpec] = []
    for entry in releases[target_version]:
        filename = str(entry.get("filename", ""))
        if not filename.endswith(".whl"):
            continue
        ver, py_tag, plat_tag = parse_wheel_tags(filename)
        if plat_tag != platform_tag:
            continue
        if python_tag is not None and py_tag != python_tag:
            continue
        sha256 = str(entry.get("digests", {}).get("sha256", "")).lower()
        pinned = PINNED_WHEEL_SHA256.get(filename)
        if pinned is not None and sha256 != pinned:
            raise ArtifactFetchError(
                f"PyPI SHA-256 mismatch for {filename}: expected {pinned}, got {sha256}"
            )
        resolved.append(
            WheelArtifactSpec(
                version=ver,
                filename=filename,
                url=str(entry["url"]),
                sha256=sha256,
                platform_tag=plat_tag,
                python_tag=py_tag,
                size_bytes=int(entry.get("size", 0)),
            )
        )

    resolved.sort(key=lambda item: item.filename)
    if not resolved:
        raise ArtifactFetchError(
            f"No wheels matched version={target_version}, platform={platform_tag}, "
            f"python_tag={python_tag}"
        )
    return resolved


def resolve_gcs_nightly_wheels(
    platform_filter: str = "manylinux",
    index_url: str = GCS_LIBTPU_INDEX_URL,
    reader: UrlReader = default_url_reader,
) -> list[str]:
    """Parses wheel download links from the Google Cloud Storage `libtpu` release index.

    Args:
        platform_filter: Substring filter applied to wheel hrefs.
        index_url: GCS HTML index URL.
        reader: Injectable URL byte fetcher.

    Returns:
        Sorted list of matching wheel URLs or relative hrefs.
    """
    html = reader(index_url).decode("utf-8", errors="replace")
    hrefs = re.findall(r'href="([^"]+\.whl)"', html)
    return sorted(h for h in hrefs if platform_filter in h)


def download_and_extract_wheel(
    spec: WheelArtifactSpec,
    cache_dir: Path,
    reader: UrlReader = default_url_reader,
) -> tuple[Path, tuple[ExtractedBinaryDigest, ...]]:
    """Downloads a wheel to `cache_dir` (reusing verified cache) and extracts `.so` binaries.

    Args:
        spec: Resolved wheel specification including expected SHA-256 digest.
        cache_dir: Root cache directory (e.g., `.cache/wheels`).
        reader: Injectable URL byte fetcher.

    Returns:
        Tuple of `(wheel_path, extracted_binaries)`.

    Raises:
        ArtifactFetchError: If wheel or extracted `.so` SHA-256 verification fails.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    wheel_path = cache_dir / spec.filename

    if wheel_path.exists():
        cached_digest = compute_sha256_file(wheel_path)
        if cached_digest != spec.sha256:
            wheel_path.unlink()

    if not wheel_path.exists():
        raw_wheel = reader(spec.url)
        actual_digest = compute_sha256_bytes(raw_wheel)
        if actual_digest != spec.sha256:
            raise ArtifactFetchError(
                f"Downloaded wheel SHA-256 mismatch for {spec.filename}: "
                f"expected {spec.sha256}, got {actual_digest}"
            )
        wheel_path.write_bytes(raw_wheel)

    version_root = cache_dir / spec.version
    extract_root = version_root / spec.python_tag / spec.platform_tag
    extract_root.mkdir(parents=True, exist_ok=True)
    is_freethreaded = spec.python_tag.endswith("t")

    extracted: list[ExtractedBinaryDigest] = []
    with zipfile.ZipFile(wheel_path, "r") as zf:
        for py_member in sorted(name for name in zf.namelist() if name.endswith(".py")):
            py_bytes = zf.read(py_member)
            for py_root in (extract_root / "python", version_root / "python"):
                py_dest = py_root / py_member
                py_dest.parent.mkdir(parents=True, exist_ok=True)
                py_dest.write_bytes(py_bytes)

        so_members = sorted(name for name in zf.namelist() if name.endswith(".so"))
        for member in so_members:
            dest_path = extract_root / Path(member).name
            if member.endswith("/sdk.so"):
                normalized_key = (
                    "libtpu/sdk.so:freethreaded" if is_freethreaded else "libtpu/sdk.so"
                )
            else:
                normalized_key = "libtpu/libtpu.so"
            expected_so_sha = PINNED_BINARY_SHA256.get((spec.version, normalized_key))

            if not dest_path.exists() or (
                expected_so_sha is not None and compute_sha256_file(dest_path) != expected_so_sha
            ):
                with zf.open(member, "r") as src, dest_path.open("wb") as dst:
                    while True:
                        chunk = src.read(4 * 1024 * 1024)
                        if not chunk:
                            break
                        dst.write(chunk)

            so_sha = compute_sha256_file(dest_path)
            if expected_so_sha is not None and so_sha != expected_so_sha:
                raise ArtifactFetchError(
                    f"Extracted binary SHA-256 mismatch for {member}: "
                    f"expected {expected_so_sha}, got {so_sha}"
                )

            top_so_name = (
                "sdk_freethreaded.so"
                if (member.endswith("/sdk.so") and is_freethreaded)
                else Path(member).name
            )
            top_so_path = version_root / top_so_name
            if not top_so_path.exists():
                shutil.copyfile(dest_path, top_so_path)

            extracted.append(
                ExtractedBinaryDigest(
                    member_name=member,
                    extracted_path=str(dest_path),
                    size_bytes=dest_path.stat().st_size,
                    sha256=so_sha,
                )
            )

    return wheel_path, tuple(extracted)


def _scan_protodesc_filenames(section_bytes: bytes) -> tuple[str, ...]:
    """Scans a `protodesc_cold` byte slice for length-prefixed `.proto` descriptor filenames.

    Args:
        section_bytes: Raw `protodesc_cold` ELF section bytes.

    Returns:
        Sorted tuple of unique `.proto` file paths discovered in `section_bytes`.
    """
    return tuple(sorted({entry["name"] for entry in carve_protodesc_cold(section_bytes)}))


def _extract_pjrt_header_from_sections(
    raw_so: bytes,
    sections: list[ElfSectionHeader],
) -> tuple[int, int, int]:
    """Attempts to read `(struct_size, major, minor)` of the `PJRT_Api` table in an ELF.

    Args:
        raw_so: Raw ELF64 binary byte buffer.
        sections: Parsed `ElfSectionHeader` descriptors for `raw_so`.

    Returns:
        Tuple of `(pjrt_struct_size, major_version, minor_version)`, or `(0, 0, 0)` if absent.
    """
    sec_map = {sec.name: sec for sec in sections}
    for sec_name in (".data.rel.ro", ".rodata", ".data"):
        sec = sec_map.get(sec_name)
        if sec is None:
            continue
        raw = raw_so[sec.sh_offset : sec.sh_offset + sec.sh_size]
        limit = len(raw) - 40
        offset = 0
        while offset <= limit:
            idx = raw.find(struct.pack("<Q", 24), offset)
            if idx == -1 or idx < 16 or idx + 24 > len(raw):
                break
            struct_size = struct.unpack_from("<Q", raw, idx - 16)[0]
            major, minor = struct.unpack_from("<ii", raw, idx + 16)
            if (
                512 <= struct_size <= 4096
                and struct_size % 8 == 0
                and 0 <= major <= 5
                and 1 <= minor <= 500
            ):
                return int(struct_size), int(major), int(minor)
            offset = idx + 8

    text_sec = sec_map.get(".text")
    if text_sec is not None:
        text_raw = raw_so[text_sec.sh_offset : text_sec.sh_offset + text_sec.sh_size]
        ver_pat = b"\x48\xc7\x40\x10\x18\x00\x00\x00"
        pos = text_raw.find(ver_pat)
        if pos >= 8 and pos + 32 <= len(text_raw):
            win = text_raw[pos - 16 : pos + 32]
            sz_idx = win.find(b"\x48\xc7\x00")
            ver_idx = win.find(b"\x49\xb9")
            if (
                sz_idx != -1
                and sz_idx + 7 <= len(win)
                and ver_idx != -1
                and ver_idx + 10 <= len(win)
            ):
                s_size = struct.unpack_from("<I", win, sz_idx + 3)[0]
                maj, min_v = struct.unpack_from("<ii", win, ver_idx + 2)
                if (
                    512 <= s_size <= 4096
                    and s_size % 8 == 0
                    and 0 <= maj <= 5
                    and 1 <= min_v <= 500
                ):
                    return int(s_size), int(maj), int(min_v)

    return 0, 0, 0


def inspect_wheel_archive(wheel_path: Path) -> WheelInspectionReport:
    """Inspects an on-disk `.whl` archive and extracts ELF symbol, PJRT, and proto inventories.

    Args:
        wheel_path: Path to the `.whl` zip archive on disk.

    Returns:
        Populated `WheelInspectionReport` instance.
    """
    version, _, _ = parse_wheel_tags(wheel_path.name)
    wheel_sha = compute_sha256_file(wheel_path)
    binaries: list[ExtractedBinaryDigest] = []
    exported_symbols: set[str] = set()
    proto_names: tuple[str, ...] = ()
    pjrt_header = (0, 0, 0)

    with zipfile.ZipFile(wheel_path, "r") as zf:
        for member in sorted(name for name in zf.namelist() if name.endswith(".so")):
            raw_so = zf.read(member)
            binaries.append(
                ExtractedBinaryDigest(
                    member_name=member,
                    extracted_path=member,
                    size_bytes=len(raw_so),
                    sha256=compute_sha256_bytes(raw_so),
                )
            )
            sections, dynsyms = parse_elf64_sections_and_dynsym(raw_so)
            for sym in dynsyms:
                exported_symbols.add(sym.name)
            for sec in sections:
                if sec.name == "protodesc_cold":
                    proto_names = _scan_protodesc_filenames(
                        raw_so[sec.sh_offset : sec.sh_offset + sec.sh_size]
                    )
            if pjrt_header == (0, 0, 0):
                pjrt_header = _extract_pjrt_header_from_sections(raw_so, sections)

    return WheelInspectionReport(
        wheel_filename=wheel_path.name,
        wheel_sha256=wheel_sha,
        version=version,
        binaries=tuple(binaries),
        exported_symbols=tuple(sorted(exported_symbols)),
        pjrt_struct_size=pjrt_header[0],
        pjrt_version_major=pjrt_header[1],
        pjrt_version_minor=pjrt_header[2],
        protodesc_proto_names=proto_names,
    )


def diff_wheel_reports(
    old_report: WheelInspectionReport,
    new_report: WheelInspectionReport,
) -> WheelBinaryDiff:
    """Computes ABI symbol, PJRT version, and `protodesc_cold` deltas between two wheels.

    Args:
        old_report: Baseline `WheelInspectionReport`.
        new_report: Target `WheelInspectionReport`.

    Returns:
        Structured `WheelBinaryDiff` describing all binary and schema changes.
    """
    old_syms = set(old_report.exported_symbols)
    new_syms = set(new_report.exported_symbols)
    old_protos = set(old_report.protodesc_proto_names)
    new_protos = set(new_report.protodesc_proto_names)
    return WheelBinaryDiff(
        old_wheel=old_report.wheel_filename,
        new_wheel=new_report.wheel_filename,
        old_pjrt_version=(
            old_report.pjrt_struct_size,
            old_report.pjrt_version_major,
            old_report.pjrt_version_minor,
        ),
        new_pjrt_version=(
            new_report.pjrt_struct_size,
            new_report.pjrt_version_major,
            new_report.pjrt_version_minor,
        ),
        added_symbols=tuple(sorted(new_syms - old_syms)),
        removed_symbols=tuple(sorted(old_syms - new_syms)),
        added_protos=tuple(sorted(new_protos - old_protos)),
        removed_protos=tuple(sorted(old_protos - new_protos)),
    )


def resolve_pinned_wheels(
    version: str = "0.0.53.1",
    platform_tag: str = "manylinux_2_31_x86_64",
    python_tag: str | None = None,
    cache_dir: Path | None = None,
) -> list[WheelArtifactSpec]:
    """Builds `WheelArtifactSpec` records from `PINNED_WHEEL_SHA256` and verifies cached files.

    Args:
        version: Target `libtpu` version string.
        platform_tag: PEP 427 platform tag filter.
        python_tag: Optional CPython tag filter (or `None` for all tags).
        cache_dir: Optional wheel cache directory whose existing `.whl` files are verified.

    Returns:
        Sorted list of `WheelArtifactSpec` records matching the requested filters.

    Raises:
        ArtifactFetchError: If no pinned wheels match or a cached `.whl` fails SHA-256 verification.
    """
    specs: list[WheelArtifactSpec] = []
    for filename, sha256 in sorted(PINNED_WHEEL_SHA256.items()):
        ver, py_tag, plat_tag = parse_wheel_tags(filename)
        if ver != version or plat_tag != platform_tag:
            continue
        if python_tag is not None and py_tag != python_tag:
            continue
        size_bytes = 0
        if cache_dir is not None:
            cached_path = cache_dir / filename
            if cached_path.exists():
                actual = compute_sha256_file(cached_path)
                if actual != sha256:
                    raise ArtifactFetchError(
                        f"Cached wheel SHA-256 mismatch for {filename}: "
                        f"expected {sha256}, got {actual}"
                    )
                size_bytes = cached_path.stat().st_size
        specs.append(
            WheelArtifactSpec(
                version=ver,
                filename=filename,
                url=f"https://files.pythonhosted.org/packages/py3/l/libtpu/{filename}",
                sha256=sha256,
                platform_tag=plat_tag,
                python_tag=py_tag,
                size_bytes=size_bytes,
            )
        )
    if not specs:
        raise ArtifactFetchError(
            f"No pinned wheels matched version={version}, platform={platform_tag}, "
            f"python_tag={python_tag}"
        )
    return specs


def main(argv: list[str] | None = None, reader: UrlReader = default_url_reader) -> int:
    """CLI entry point for fetching, inspecting, and diffing `libtpu` wheels.

    Args:
        argv: Optional CLI argument list (defaults to `sys.argv[1:]`).
        reader: Injectable URL byte fetcher for deterministic offline testing.

    Returns:
        Exit status code (`0` on success, `1` on error).
    """
    parser = argparse.ArgumentParser(description="Fetch, verify, and diff official libtpu wheels.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/wheels"),
        help="Target directory for cached wheels and extracted .so binaries",
    )
    parser.add_argument("--version", default="0.0.53.1", help="Target PyPI libtpu version")
    parser.add_argument(
        "--platform",
        default=TARGET_PLATFORMS[0],
        choices=list(SUPPORTED_PLATFORM_TAGS),
        help="PEP 427 wheel platform tag",
    )
    parser.add_argument("--python-tag", default="cp311", help="CPython tag (e.g., cp311 or all)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify pinned wheel metadata and cached SHA-256 digests without downloading",
    )
    parser.add_argument("--fetch", action="store_true", help="Download and extract matching wheel")
    parser.add_argument("--inspect", type=Path, help="Inspect an on-disk .whl archive")
    parser.add_argument(
        "--diff",
        nargs=2,
        type=Path,
        metavar=("OLD_WHL", "NEW_WHL"),
        help="Diff two on-disk .whl archives",
    )
    args = parser.parse_args(argv)

    try:
        if args.diff is not None:
            old_rep = inspect_wheel_archive(args.diff[0])
            new_rep = inspect_wheel_archive(args.diff[1])
            print(json.dumps(asdict(diff_wheel_reports(old_rep, new_rep)), indent=2))
            return 0
        if args.inspect is not None:
            print(json.dumps(asdict(inspect_wheel_archive(args.inspect)), indent=2))
            return 0
        py_filter = None if str(args.python_tag) == "all" else str(args.python_tag)
        if args.dry_run:
            pinned_specs = resolve_pinned_wheels(
                version=str(args.version),
                platform_tag=str(args.platform),
                python_tag=py_filter,
                cache_dir=Path(args.cache_dir),
            )
            print(json.dumps([asdict(s) for s in pinned_specs], indent=2))
            return 0
        specs = resolve_pypi_wheels(
            version=str(args.version),
            platform_tag=str(args.platform),
            python_tag=py_filter,
            reader=reader,
        )
        if args.fetch:
            results = []
            for spec in specs:
                wheel_path, binaries = download_and_extract_wheel(
                    spec,
                    cache_dir=Path(args.cache_dir),
                    reader=reader,
                )
                results.append(
                    {
                        "wheel": str(wheel_path),
                        "binaries": [asdict(b) for b in binaries],
                    }
                )
            print(json.dumps(results[0] if len(results) == 1 else results, indent=2))
            return 0
        print(json.dumps([asdict(s) for s in specs], indent=2))
        return 0
    except ArtifactFetchError as exc:
        print(f"ArtifactFetchError: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

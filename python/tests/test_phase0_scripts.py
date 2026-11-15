"""Unit and branch tests for `scripts/fetch_artifacts.py` and `scripts/sync_oss_anchors.py`."""

from __future__ import annotations

import io
import json
import runpy
import struct
import sys
import urllib.request
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

import scripts.fetch_artifacts as fa
import scripts.sync_oss_anchors as oa


class _FakeHttpResponse(io.BytesIO):
    """Deterministic `urlopen` context-manager stub returning in-memory bytes."""

    def __enter__(self) -> _FakeHttpResponse:
        """Enters the HTTP response context manager.

        Returns:
            Self reference.
        """
        return self

    def __exit__(self, *_exc: object) -> None:
        """Exits the HTTP response context manager.

        Args:
            *_exc: Optional exception tuple.
        """


def _build_test_elf64(
    symbol_names: tuple[str, ...],
    include_protodesc: bool,
    pjrt_header: tuple[int, int, int] | None,
) -> bytes:
    """Builds a valid 64-bit little-endian ELF shared object for `pyelftools` inspection.

    Args:
        symbol_names: Exported global function names to emit into `.dynsym`.
        include_protodesc: Whether to attach a `protodesc_cold` section.
        pjrt_header: Optional `(struct_size, major, minor)` tuple to embed in `.rodata`.

    Returns:
        Serialized ELF64 binary bytes.
    """
    sh_names = [b"", b".shstrtab", b".dynstr", b".dynsym"]
    if include_protodesc:
        sh_names.append(b"protodesc_cold")
    if pjrt_header is not None:
        if pjrt_header[0] == 0:
            sh_names.append(b".data.rel.ro")
        sh_names.append(b".rodata")

    sh_offsets: dict[bytes, int] = {}
    shstr_parts: list[bytes] = []
    cursor = 0
    for name in sh_names:
        sh_offsets[name] = cursor
        chunk = name + b"\x00"
        shstr_parts.append(chunk)
        cursor += len(chunk)
    shstrtab = b"".join(shstr_parts)

    dynstr_parts = [b"\x00"]
    sym_entries = [
        # STN_UNDEF null symbol
        struct.pack("<IBBHQQ", 0, 0, 0, 0, 0, 0),
        # Local symbol (branch coverage for non-global symbol)
        struct.pack("<IBBHQQ", 0, 0x02, 0, 1, 0x1000, 8),
        # Global undefined symbol (SHN_UNDEF == 0)
        struct.pack("<IBBHQQ", 0, 0x12, 0, 0, 0, 0),
        # Global defined symbol with empty name (st_name == 0)
        struct.pack("<IBBHQQ", 0, 0x12, 0, 1, 0x1008, 8),
    ]
    dyn_cursor = 1
    for sym_name in symbol_names:
        raw_sym = sym_name.encode("ascii") + b"\x00"
        sym_entries.append(struct.pack("<IBBHQQ", dyn_cursor, 0x12, 0, 1, 0x2000, 16))
        dynstr_parts.append(raw_sym)
        dyn_cursor += len(raw_sym)
    dynstr = b"".join(dynstr_parts)
    dynsym = b"".join(sym_entries)

    protodesc_data = b"\x0a\x09hlo.proto\x0a\x07hlo.proto\x0a\x7fshort.proto"

    rodata = b""
    if pjrt_header is not None:
        sz, maj, min_v = pjrt_header
        if sz == 0:
            # Exercises idx < 16 break branch
            rodata = struct.pack("<Q", 24) + (b"\x00" * 40)
        else:
            # Include an invalid candidate first (size=16) so the search loop iterates once
            bad_entry = struct.pack("<QQQQii", 16, 0, 24, 0, 0, 1) + (b"\x00" * 16)
            good_entry = struct.pack("<QQQQii", sz, 0, 24, 0, maj, min_v) + (b"\x00" * 16)
            rodata = bad_entry + good_entry

    ehdr_size = 64
    body_chunks: list[tuple[bytes, int, bytes, int, int]] = [
        (b".shstrtab", 3, shstrtab, 0, 0),
        (b".dynstr", 3, dynstr, 0, 0),
        (b".dynsym", 11, dynsym, 2, 24),
    ]
    if include_protodesc:
        body_chunks.append((b"protodesc_cold", 1, protodesc_data, 0, 0))
    if pjrt_header is not None:
        if pjrt_header[0] == 0:
            body_chunks.append((b".data.rel.ro", 1, b"\x00" * 16, 0, 0))
        body_chunks.append((b".rodata", 1, rodata, 0, 0))

    payload = bytearray()
    sec_headers = [struct.pack("<IIQQQQIIQQ", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)]
    file_off = ehdr_size
    for sec_name, sh_type, data, sh_link, entsize in body_chunks:
        while (file_off + len(payload)) % 8 != 0:
            payload.append(0)
        cur_off = ehdr_size + len(payload)
        payload.extend(data)
        sec_headers.append(
            struct.pack(
                "<IIQQQQIIQQ",
                sh_offsets[sec_name],
                sh_type,
                2,
                0,
                cur_off,
                len(data),
                sh_link,
                1,
                8,
                entsize,
            )
        )

    while (ehdr_size + len(payload)) % 8 != 0:
        payload.append(0)
    shoff = ehdr_size + len(payload)
    shdr_blob = b"".join(sec_headers)

    e_ident = b"\x7fELF" + bytes([2, 1, 1, 0]) + (b"\x00" * 8)
    ehdr_rest = struct.pack(
        "<HHIQQQIHHHHHH",
        3,
        62,
        1,
        0,
        0,
        shoff,
        0,
        ehdr_size,
        0,
        0,
        64,
        len(sec_headers),
        1,
    )
    return e_ident + ehdr_rest + bytes(payload) + shdr_blob


def test_fetch_artifacts_url_hashing_and_tag_parsing(tmp_path: Path) -> None:
    """Tests URL reader guards, SHA-256 helpers, and PEP 427 wheel filename parsing.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
    """
    with pytest.raises(fa.ArtifactFetchError, match="Refusing non-HTTPS"):
        fa.default_url_reader("http://insecure.example.com/libtpu.whl")

    with patch.object(
        urllib.request,
        "urlopen",
        side_effect=lambda *_a, **_kw: _FakeHttpResponse(b"wheel-bytes"),
    ):
        assert fa.default_url_reader("https://example.com/libtpu.whl") == b"wheel-bytes"

    empty_file = tmp_path / "empty.bin"
    empty_file.write_bytes(b"")
    assert fa.compute_sha256_file(empty_file) == fa.compute_sha256_bytes(b"")

    sample_file = tmp_path / "sample.bin"
    sample_file.write_bytes(b"libtpu-rs")
    assert fa.compute_sha256_file(sample_file) == fa.compute_sha256_bytes(b"libtpu-rs")

    assert fa.parse_wheel_tags("libtpu-0.0.52-cp311-cp311-manylinux_2_31_x86_64.whl") == (
        "0.0.52",
        "cp311",
        "manylinux_2_31_x86_64",
    )
    with pytest.raises(fa.ArtifactFetchError, match="missing .whl suffix"):
        fa.parse_wheel_tags("libtpu-0.0.52.tar.gz")
    with pytest.raises(fa.ArtifactFetchError, match="Malformed PEP 427"):
        fa.parse_wheel_tags("libtpu-0.0.52.whl")


def test_resolve_download_inspect_and_diff_wheels(tmp_path: Path) -> None:
    """Tests PyPI/GCS wheel resolution, cached extraction, ELF inspection, and diffing.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
    """
    cp311_whl = "libtpu-0.0.52-cp311-cp311-manylinux_2_31_x86_64.whl"
    cp312_whl = "libtpu-0.0.52-cp312-cp312-manylinux_2_31_x86_64.whl"
    pypi_doc = {
        "info": {"version": "0.0.52"},
        "releases": {
            "0.0.52": [
                {"filename": "libtpu-0.0.52.tar.gz", "url": "https://x/s.tar.gz", "digests": {}},
                {
                    "filename": "libtpu-0.0.52-cp311-cp311-manylinux_2_35_aarch64.whl",
                    "url": "https://x/arm.whl",
                    "size": 10,
                    "digests": {"sha256": "aa" * 32},
                },
                {
                    "filename": cp312_whl,
                    "url": "https://x/cp312.whl",
                    "size": 20,
                    "digests": {"sha256": fa.PINNED_WHEEL_SHA256[cp312_whl]},
                },
                {
                    "filename": cp311_whl,
                    "url": "https://x/cp311.whl",
                    "size": 30,
                    "digests": {"sha256": fa.PINNED_WHEEL_SHA256[cp311_whl]},
                },
            ]
        },
    }
    raw_json = json.dumps(pypi_doc).encode("utf-8")

    specs_latest = fa.resolve_pypi_wheels(
        version="latest",
        platform_tag="manylinux_2_31_x86_64",
        python_tag=None,
        reader=lambda _u: raw_json,
    )
    assert [s.filename for s in specs_latest] == [cp311_whl, cp312_whl]

    arm_specs = fa.resolve_pypi_wheels(
        version="0.0.52",
        platform_tag="manylinux_2_35_aarch64",
        python_tag="cp311",
        reader=lambda _u: raw_json,
    )
    assert len(arm_specs) == 1

    with pytest.raises(fa.ArtifactFetchError, match="not found in PyPI"):
        fa.resolve_pypi_wheels(version="9.9.9", reader=lambda _u: raw_json)

    with pytest.raises(fa.ArtifactFetchError, match="No wheels matched"):
        fa.resolve_pypi_wheels(
            version="0.0.52",
            platform_tag="manylinux_2_31_x86_64",
            python_tag="cp39",
            reader=lambda _u: raw_json,
        )

    tampered_doc = {
        "info": {"version": "0.0.52"},
        "releases": {
            "0.0.52": [
                {
                    "filename": cp311_whl,
                    "url": "https://x/cp311.whl",
                    "size": 30,
                    "digests": {"sha256": "00" * 32},
                }
            ]
        },
    }
    with pytest.raises(fa.ArtifactFetchError, match="PyPI SHA-256 mismatch"):
        fa.resolve_pypi_wheels(reader=lambda _u: json.dumps(tampered_doc).encode("utf-8"))

    gcs_html = (
        b'<a href="libtpu-0.0.53.dev1-cp311-cp311-manylinux_2_31_x86_64.whl">x</a>'
        b'<a href="other-win_amd64.whl">y</a>'
    )
    assert fa.resolve_gcs_nightly_wheels(reader=lambda _u: gcs_html) == [
        "libtpu-0.0.53.dev1-cp311-cp311-manylinux_2_31_x86_64.whl"
    ]

    # Build two synthetic wheels to test download, cache-hit, corruption, inspect, and diff
    elf_old = _build_test_elf64(("GetPjrtApi", "OldSym"), True, (1144, 0, 115))
    elf_sdk_no_pjrt = _build_test_elf64(("PyInit_sdk",), False, None)
    elf_new = _build_test_elf64(("GetPjrtApi", "NewSym"), False, (0, 0, 0))

    old_whl_buf = io.BytesIO()
    with zipfile.ZipFile(old_whl_buf, "w") as zf:
        zf.writestr("libtpu/__init__.py", "# init")
        zf.writestr("libtpu/libtpu.so", elf_old)
        zf.writestr("libtpu/sdk.so", elf_sdk_no_pjrt)
    old_whl_bytes = old_whl_buf.getvalue()

    new_whl_buf = io.BytesIO()
    with zipfile.ZipFile(new_whl_buf, "w") as zf:
        zf.writestr("libtpu/libtpu.so", elf_new)
    new_whl_bytes = new_whl_buf.getvalue()

    custom_spec = fa.WheelArtifactSpec(
        version="0.0.99",
        filename="libtpu-0.0.99-cp311-cp311-manylinux_2_31_x86_64.whl",
        url="https://example.com/libtpu-0.0.99.whl",
        sha256=fa.compute_sha256_bytes(old_whl_bytes),
        platform_tag="manylinux_2_31_x86_64",
        python_tag="cp311",
        size_bytes=len(old_whl_bytes),
    )

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    # Pre-seed corrupt cached wheel to test automatic eviction + re-download
    (cache_dir / custom_spec.filename).write_bytes(b"corrupt")

    with pytest.raises(fa.ArtifactFetchError, match="Downloaded wheel SHA-256 mismatch"):
        fa.download_and_extract_wheel(custom_spec, cache_dir, reader=lambda _u: b"wrong-bytes")

    whl_disk, bins = fa.download_and_extract_wheel(
        custom_spec, cache_dir, reader=lambda _u: old_whl_bytes
    )
    assert whl_disk.exists()
    assert len(bins) == 2

    # Second call exercises verified cache hit and already-extracted .so reuse
    whl_disk2, bins2 = fa.download_and_extract_wheel(
        custom_spec,
        cache_dir,
        reader=lambda _u: (_ for _ in ()).throw(AssertionError("unexpected network call")),
    )
    assert whl_disk2 == whl_disk
    assert bins2 == bins

    # Test pinned .so digest mismatch protection
    pinned_key = ("0.0.99", "libtpu/libtpu.so")
    with (
        patch.dict(fa.PINNED_BINARY_SHA256, {pinned_key: "ff" * 32}),
        pytest.raises(fa.ArtifactFetchError, match="Extracted binary SHA-256 mismatch"),
    ):
        fa.download_and_extract_wheel(custom_spec, cache_dir, reader=lambda _u: old_whl_bytes)

    # Test pinned .so digest match reuse branch
    with patch.dict(
        fa.PINNED_BINARY_SHA256,
        {pinned_key: fa.compute_sha256_bytes(elf_old)},
    ):
        _, bins3 = fa.download_and_extract_wheel(
            custom_spec, cache_dir, reader=lambda _u: old_whl_bytes
        )
        assert len(bins3) == 2

    new_whl_path = tmp_path / "libtpu-0.0.100-cp311-cp311-manylinux_2_31_x86_64.whl"
    new_whl_path.write_bytes(new_whl_bytes)

    rep_old = fa.inspect_wheel_archive(whl_disk)
    rep_new = fa.inspect_wheel_archive(new_whl_path)
    assert rep_old.pjrt_version_minor == 115
    assert "hlo.proto" in rep_old.protodesc_proto_names
    assert rep_new.pjrt_struct_size == 0

    diff = fa.diff_wheel_reports(rep_old, rep_new)
    assert diff.added_symbols == ("NewSym",)
    assert diff.removed_symbols == ("OldSym", "PyInit_sdk")
    assert "hlo.proto" in diff.removed_protos

    # Exercise CLI modes
    custom_pypi = json.dumps(
        {
            "info": {"version": "0.0.99"},
            "releases": {
                "0.0.99": [
                    {
                        "filename": custom_spec.filename,
                        "url": custom_spec.url,
                        "size": custom_spec.size_bytes,
                        "digests": {"sha256": custom_spec.sha256},
                    }
                ]
            },
        }
    ).encode("utf-8")

    def mux_reader(url: str) -> bytes:
        """Routes mock PyPI metadata vs wheel payload URLs.

        Args:
            url: Target HTTPS URL.

        Returns:
            Mock payload bytes.
        """
        return custom_pypi if url.endswith("/json") else old_whl_bytes

    assert fa.main(["--version", "0.0.99"], reader=mux_reader) == 0
    assert (
        fa.main(
            ["--version", "0.0.99", "--fetch", "--cache-dir", str(tmp_path / "cli_cache")],
            reader=mux_reader,
        )
        == 0
    )
    assert fa.main(["--inspect", str(whl_disk)], reader=mux_reader) == 0
    assert fa.main(["--diff", str(whl_disk), str(new_whl_path)], reader=mux_reader) == 0
    assert fa.main(["--version", "missing"], reader=mux_reader) == 1

    with (
        patch.object(sys, "argv", ["fetch_artifacts.py", "--inspect", str(whl_disk)]),
        pytest.raises(SystemExit) as exc_info,
    ):
        runpy.run_path(str(Path(fa.__file__)), run_name="__main__")
    assert exc_info.value.code == 0


def test_sync_oss_anchors_full_coverage(tmp_path: Path) -> None:
    """Tests `scripts/sync_oss_anchors.py` URL construction, token extraction, caching, and CLI.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
    """
    with pytest.raises(oa.OssAnchorError, match="Refusing non-HTTPS"):
        oa.default_text_reader("http://insecure.example.com/file.h")

    with patch.object(
        urllib.request,
        "urlopen",
        side_effect=lambda *_a, **_kw: _FakeHttpResponse(b"#define PJRT_API_MAJOR 0"),
    ):
        assert "PJRT_API_MAJOR" in oa.default_text_reader("https://example.com/pjrt_c_api.h")

    pin = oa.get_pin_by_key("linux_uapi")
    assert (
        oa.build_raw_github_url(pin, "include/uapi/linux/vfio.h")
        == "https://raw.githubusercontent.com/torvalds/linux/v6.12/include/uapi/linux/vfio.h"
    )

    with pytest.raises(oa.OssAnchorError, match="Unknown OSS anchor key"):
        oa.get_pin_by_key("nonexistent_key")

    sample_src = """#define VFIO_API_VERSION 0
def VHLO_AddOp : VHLO_Op {}
message HloInstructionProto {
}
"""
    calls: list[str] = []

    def fake_reader(url: str) -> str:
        """Records fetched URLs and returns deterministic specification text.

        Args:
            url: Requested raw GitHub URL.

        Returns:
            Synthetic C/TableGen/Proto specification text.
        """
        calls.append(url)
        return sample_src

    anchor_cache = tmp_path / "oss_cache"
    summaries = oa.sync_all_anchors(
        cache_dir=anchor_cache,
        repo_keys=["linux_uapi"],
        reader=fake_reader,
    )
    assert len(summaries) == 2
    assert len(calls) == 2
    assert summaries[0].c_macros == ("VFIO_API_VERSION",)
    assert summaries[0].tablegen_defs == ("VHLO_AddOp",)
    assert summaries[0].proto_messages == ("HloInstructionProto",)

    # Second sync must hit local disk cache without invoking fake_reader
    summaries_cached = oa.sync_all_anchors(
        cache_dir=anchor_cache,
        repo_keys=["linux_uapi"],
        reader=fake_reader,
    )
    assert summaries_cached == summaries
    assert len(calls) == 2

    # Sync all pins
    all_summaries = oa.sync_all_anchors(cache_dir=tmp_path / "all_oss", reader=fake_reader)
    assert len(all_summaries) >= 30

    # CLI paths: list all, list filtered, sync filtered, error handling, and __main__
    assert oa.main([], reader=fake_reader) == 0
    assert oa.main(["--repo-key", "linux_gasket"], reader=fake_reader) == 0
    assert (
        oa.main(
            ["--sync", "--repo-key", "linux_gasket", "--cache-dir", str(anchor_cache)],
            reader=fake_reader,
        )
        == 0
    )
    assert oa.main(["--repo-key", "bad_key"], reader=fake_reader) == 1

    with (
        patch.object(sys, "argv", ["sync_oss_anchors.py", "--repo-key", "linux_uapi"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        runpy.run_path(str(Path(oa.__file__)), run_name="__main__")
    assert exc_info.value.code == 0

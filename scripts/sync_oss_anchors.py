#!/usr/bin/env python3
"""Open-Source Reference Pinning and Specification Extractor for `libtpu-rs`.

Pins exact upstream commits and extracts canonical bytecode, C ABI, dialect, protobuf,
and kernel UAPI specifications from `llvm/llvm-project` (MLIR), `openxla/stablehlo`,
`openxla/shardy`, `openxla/xla`, `tensorflow/tensorflow`, `openxla/xprof`, `jax-ml/jax`
(Mosaic/Pallas), and `torvalds/linux` (Gasket/Apex & VFIO/DMA-BUF).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


class OssAnchorError(RuntimeError):
    """Raised when an open-source anchor specification cannot be fetched or parsed."""


@dataclass(frozen=True)
class OssRepoPin:
    """Pinned upstream GitHub repository commit and raw content template."""

    key: str
    repo: str
    commit: str
    files: tuple[str, ...]


@dataclass(frozen=True)
class ExtractedSpecSummary:
    """Structured specification tokens extracted from an upstream anchor file."""

    repo_key: str
    commit: str
    relative_path: str
    sha256: str
    c_macros: tuple[str, ...]
    tablegen_defs: tuple[str, ...]
    proto_messages: tuple[str, ...]


OSS_ANCHOR_PINS: tuple[OssRepoPin, ...] = (
    OssRepoPin(
        key="llvm_mlir",
        repo="llvm/llvm-project",
        commit="llvmorg-20.1.0",
        files=(
            "mlir/include/mlir/Bytecode/Encoding.h",
            "mlir/lib/Bytecode/Reader/BytecodeReader.cpp",
            "mlir/lib/Bytecode/Writer/BytecodeWriter.cpp",
            "mlir/include/mlir/IR/BuiltinDialectBytecode.td",
        ),
    ),
    OssRepoPin(
        key="openxla_stablehlo",
        repo="openxla/stablehlo",
        commit="v1.9.3",
        files=(
            "stablehlo/dialect/VhloOps.td",
            "stablehlo/dialect/VhloBytecode.cpp",
            "stablehlo/dialect/VhloTypes.cpp",
            "stablehlo/dialect/StablehloOps.td",
            "stablehlo/dialect/ChloOps.td",
        ),
    ),
    OssRepoPin(
        key="openxla_shardy",
        repo="openxla/shardy",
        commit="main",
        files=(
            "shardy/dialect/sdy/ir/ops.td",
            "shardy/dialect/sdy/ir/attrs.td",
            "shardy/dialect/sdy/ir/bytecode.cpp",
        ),
    ),
    OssRepoPin(
        key="openxla_xla",
        repo="openxla/xla",
        commit="main",
        files=(
            "xla/pjrt/c/pjrt_c_api.h",
            "xla/pjrt/c/pjrt_c_api_phase_compile_extension.h",
            "xla/pjrt/c/pjrt_c_api_layouts_extension.h",
            "xla/pjrt/c/pjrt_c_api_memory_descriptions_extension.h",
            "xla/pjrt/c/pjrt_c_api_ffi_extension.h",
            "xla/pjrt/c/pjrt_c_api_profiler_extension.h",
            "xla/pjrt/c/pjrt_c_api_triton_extension.h",
            "xla/pjrt/c/pjrt_c_api_raw_buffer_extension.h",
            "xla/ffi/api/c_api.h",
            "xla/service/hlo.proto",
            "xla/xla_data.proto",
            "xla/xla.proto",
            "xla/service/memory_space_assignment/memory_space_assignment.proto",
        ),
    ),
    OssRepoPin(
        key="tensorflow_and_xprof",
        repo="tensorflow/tensorflow",
        commit="v2.18.0",
        files=(
            "tensorflow/core/protobuf/tpu/tpu_embedding_configuration.proto",
            "tensorflow/core/protobuf/tpu/optimization_parameters.proto",
            "tensorflow/core/protobuf/tpu/compile_metadata.proto",
            "tensorflow/core/protobuf/tpu/topology.proto",
            "tensorflow/compiler/tf2xla/tf2xla.proto",
            "third_party/xla/third_party/tsl/tsl/profiler/protobuf/xplane.proto",
            "third_party/xla/third_party/tsl/tsl/distributed_runtime/coordination/coordination_service.proto",
        ),
    ),
    OssRepoPin(
        key="jax_mosaic",
        repo="jax-ml/jax",
        commit="jax-v0.5.0",
        files=(
            "jaxlib/mosaic/dialect/tpu/tpu.td",
            "jax/_src/lib/__init__.py",
            "jax/experimental/pallas/ops/tpu/random.py",
        ),
    ),
    OssRepoPin(
        key="linux_gasket",
        repo="torvalds/linux",
        commit="v6.6",
        files=(
            "drivers/staging/gasket/gasket.h",
            "drivers/staging/gasket/apex.h",
        ),
    ),
    OssRepoPin(
        key="linux_uapi",
        repo="torvalds/linux",
        commit="v6.12",
        files=(
            "include/uapi/linux/vfio.h",
            "include/uapi/linux/dma-buf.h",
        ),
    ),
)

UrlTextReader = Callable[[str], str]


def default_text_reader(url: str) -> str:
    """Fetches UTF-8 text from an HTTPS raw GitHub URL.

    Args:
        url: Fully-qualified `https://` raw source URL.

    Returns:
        Decoded UTF-8 text body.

    Raises:
        OssAnchorError: If `url` does not start with `https://`.
    """
    if not url.startswith("https://"):
        raise OssAnchorError(f"Refusing non-HTTPS anchor URL: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "libtpu-rs-oss-anchor-sync/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw: bytes = resp.read()
        return raw.decode("utf-8", errors="replace")


def build_raw_github_url(pin: OssRepoPin, relative_path: str) -> str:
    """Constructs the `raw.githubusercontent.com` URL for a pinned repository file.

    Args:
        pin: Pinned upstream repository descriptor.
        relative_path: Repository-relative path to the source specification file.

    Returns:
        Deterministic `https://raw.githubusercontent.com/...` URL.
    """
    return f"https://raw.githubusercontent.com/{pin.repo}/{pin.commit}/{relative_path}"


def extract_spec_tokens(
    repo_key: str,
    commit: str,
    relative_path: str,
    source_text: str,
) -> ExtractedSpecSummary:
    """Extracts C macros, MLIR TableGen `def`s, and Protobuf `message` names from `source_text`.

    Args:
        repo_key: Logical anchor repository key.
        commit: Pinned git ref or commit hash.
        relative_path: Repository-relative file path.
        source_text: Raw UTF-8 text of the specification file.

    Returns:
        Populated `ExtractedSpecSummary` containing deterministic sorted token lists.
    """
    digest = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    macros = sorted(
        set(re.findall(r"^\s*#\s*define\s+([A-Z][A-Z0-9_]{2,})\b", source_text, re.MULTILINE))
    )
    td_defs = sorted(
        set(re.findall(r"^\s*def\s+([A-Za-z0-9_]+)\s*[:{]", source_text, re.MULTILINE))
    )
    proto_msgs = sorted(
        set(re.findall(r"^\s*message\s+([A-Za-z0-9_]+)\s*\{", source_text, re.MULTILINE))
    )
    return ExtractedSpecSummary(
        repo_key=repo_key,
        commit=commit,
        relative_path=relative_path,
        sha256=digest,
        c_macros=tuple(macros),
        tablegen_defs=tuple(td_defs),
        proto_messages=tuple(proto_msgs),
    )


def get_pin_by_key(repo_key: str) -> OssRepoPin:
    """Looks up a configured `OssRepoPin` by its logical key.

    Args:
        repo_key: Key matching one of `OSS_ANCHOR_PINS` (e.g., `"openxla_xla"`).

    Returns:
        Matching `OssRepoPin` descriptor.

    Raises:
        OssAnchorError: If `repo_key` is unknown.
    """
    for pin in OSS_ANCHOR_PINS:
        if pin.key == repo_key:
            return pin
    raise OssAnchorError(f"Unknown OSS anchor key: '{repo_key}'")


def sync_anchor_pin(
    pin: OssRepoPin,
    cache_dir: Path,
    reader: UrlTextReader = default_text_reader,
) -> list[ExtractedSpecSummary]:
    """Fetches (or loads cached) files for `pin` and extracts specification summaries.

    Args:
        pin: Pinned upstream repository descriptor.
        cache_dir: Local cache root directory (e.g., `.cache/oss_anchors`).
        reader: Injectable HTTPS text fetcher.

    Returns:
        List of `ExtractedSpecSummary` records in `pin.files` order.
    """
    summaries: list[ExtractedSpecSummary] = []
    repo_cache_root = cache_dir / pin.key / pin.commit
    for rel_path in pin.files:
        local_file = repo_cache_root / rel_path
        if local_file.exists():
            text = local_file.read_text(encoding="utf-8")
        else:
            url = build_raw_github_url(pin, rel_path)
            text = reader(url)
            local_file.parent.mkdir(parents=True, exist_ok=True)
            local_file.write_text(text, encoding="utf-8")
        summaries.append(extract_spec_tokens(pin.key, pin.commit, rel_path, text))
    return summaries


def sync_all_anchors(
    cache_dir: Path,
    repo_keys: list[str] | None = None,
    reader: UrlTextReader = default_text_reader,
) -> list[ExtractedSpecSummary]:
    """Synchronizes all (or selected) open-source anchor pins into `cache_dir`.

    Args:
        cache_dir: Local cache root directory (e.g., `.cache/oss_anchors`).
        repo_keys: Optional subset of `OssRepoPin.key` identifiers to sync.
        reader: Injectable HTTPS text fetcher.

    Returns:
        Combined list of `ExtractedSpecSummary` records across synced repositories.
    """
    pins = [get_pin_by_key(k) for k in repo_keys] if repo_keys else list(OSS_ANCHOR_PINS)
    results: list[ExtractedSpecSummary] = []
    for pin in pins:
        results.extend(sync_anchor_pin(pin, cache_dir=cache_dir, reader=reader))
    return results


def main(argv: list[str] | None = None, reader: UrlTextReader = default_text_reader) -> int:
    """CLI entry point for listing or syncing pinned open-source anchor specifications.

    Args:
        argv: Optional CLI argument list (defaults to `sys.argv[1:]`).
        reader: Injectable HTTPS text fetcher.

    Returns:
        Exit status code (`0` on success, `1` on error).
    """
    parser = argparse.ArgumentParser(
        description="Pin and extract open-source MLIR/StableHLO/OpenXLA/TF/JAX/Linux specs."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/oss_anchors"),
        help="Target cache directory for fetched upstream specifications",
    )
    parser.add_argument(
        "--repo-key",
        action="append",
        dest="repo_keys",
        help="Optional anchor key filter (may be repeated)",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Download missing anchor files into --cache-dir and emit token summaries",
    )
    args = parser.parse_args(argv)

    try:
        if not args.sync:
            pins = (
                [get_pin_by_key(k) for k in args.repo_keys]
                if args.repo_keys
                else list(OSS_ANCHOR_PINS)
            )
            print(json.dumps([asdict(p) for p in pins], indent=2))
            return 0
        summaries = sync_all_anchors(
            cache_dir=Path(args.cache_dir),
            repo_keys=args.repo_keys,
            reader=reader,
        )
        print(json.dumps([asdict(s) for s in summaries], indent=2))
        return 0
    except OssAnchorError as exc:
        print(f"OssAnchorError: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

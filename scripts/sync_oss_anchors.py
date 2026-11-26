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


OSS_REPO_PINS: tuple[OssRepoPin, ...] = (
    OssRepoPin(
        key="llvm_mlir",
        repo="llvm/llvm-project",
        commit="c744694c6ae1d12bbf7d0447b4bc91ee94a1904c",
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
        commit="9fe4a1e9cedbe466af2d620c2fcf87459df32eb6",
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
        commit="ab30307107fad95e652ad1e3bfa06532bbda1905",
        files=(
            "shardy/dialect/sdy/ir/ops.td",
            "shardy/dialect/sdy/ir/attrs.td",
            "shardy/dialect/sdy/ir/bytecode.cpp",
        ),
    ),
    OssRepoPin(
        key="openxla_xla",
        repo="openxla/xla",
        commit="814b4d72d1d4a9677aab9e2f7cfcc91253103b8f",
        files=(
            "xla/pjrt/c/pjrt_c_api.h",
            "xla/pjrt/c/pjrt_c_api_phase_compile_extension.h",
            "xla/pjrt/c/pjrt_c_api_layouts_extension.h",
            "xla/pjrt/c/pjrt_c_api_memory_descriptions_extension.h",
            "xla/pjrt/c/pjrt_c_api_ffi_extension.h",
            "xla/pjrt/c/pjrt_c_api_profiler_extension.h",
            "xla/pjrt/c/pjrt_c_api_triton_extension.h",
            "xla/pjrt/c/pjrt_c_api_raw_buffer_extension.h",
            "xla/backends/profiler/plugin/profiler_c_api.h",
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
        commit="ebe9cca4ad4d14afb4b5110e6bd0d609849e3cf2",
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

OSS_ANCHOR_PINS: tuple[OssRepoPin, ...] = OSS_REPO_PINS

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


def _iter_pjrt_struct_bodies(cleaned_header: str) -> list[tuple[str, str]]:
    """Extracts brace-balanced `(struct_name, body)` pairs for `struct PJRT_*` definitions.

    Args:
        cleaned_header: Comment- and preprocessor-stripped C header text.

    Returns:
        Ordered list of `(struct_name, inner_body_text)` pairs.
    """
    results: list[tuple[str, str]] = []
    for head_m in re.finditer(r"\bstruct\s+(PJRT_[A-Za-z0-9_]+)\s*\{", cleaned_header):
        sname = head_m.group(1)
        start = head_m.end()
        depth = 1
        pos = start
        n = len(cleaned_header)
        while pos < n and depth > 0:
            ch = cleaned_header[pos]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            pos += 1
        if depth == 0:
            results.append((sname, cleaned_header[start : pos - 1]))
    return results


def _split_top_level_c_decls(body: str) -> list[str]:
    """Splits a C struct/union body into `;`-terminated top-level member declarations.

    Args:
        body: Inner text of a C `struct` or `union` block.

    Returns:
        Non-empty trimmed member declaration strings (without trailing `;`).
    """
    decls: list[str] = []
    cur: list[str] = []
    depth = 0
    for ch in body:
        if ch == "{":
            depth += 1
            cur.append(ch)
        elif ch == "}":
            depth = max(0, depth - 1)
            cur.append(ch)
        elif ch == ";" and depth == 0:
            stmt = " ".join("".join(cur).split())
            if stmt:
                decls.append(stmt)
            cur = []
        else:
            cur.append(ch)
    return decls


def _measure_c_decl_slot_bytes(decl: str, known_sizes: dict[str, int]) -> tuple[str | None, int]:
    """Extracts the optional field name and 8-byte-aligned x86_64 size of a C member declaration.

    Args:
        decl: Single `;`-stripped C member declaration (may be an anonymous `union { ... }`).
        known_sizes: Previously computed byte sizes of value-embedded `PJRT_*` structs.

    Returns:
        Tuple `(field_name_or_none, aligned_size_bytes)`.
    """
    union_m = re.match(r"^union\s*\{(.*)\}$", decl, re.DOTALL)
    if union_m is not None:
        sub_sizes = [
            _measure_c_decl_slot_bytes(sub, known_sizes)[1]
            for sub in _split_top_level_c_decls(union_m.group(1))
        ]
        return None, max(sub_sizes, default=0)

    fn_ptr_m = re.search(r"\(\s*\*\s*([A-Za-z0-9_]+)\s*\)", decl)
    if fn_ptr_m is not None:
        return fn_ptr_m.group(1), 8

    ident_m = re.match(r"^(.*?)\b([A-Za-z0-9_]+)\s*(?:\[([0-9]+)\])?$", decl)
    if ident_m is None:
        return None, 0
    type_prefix, field_name, arr_count_str = ident_m.groups()
    count = int(arr_count_str) if arr_count_str is not None else 1
    clean_type = type_prefix.replace("const", "").replace("struct", "").strip()
    elem_size = 8 if "*" in clean_type else known_sizes.get(clean_type, 8)
    return field_name, elem_size * count


def parse_pjrt_c_api_header(header_text: str) -> dict[str, object]:
    """Parses `PJRT_API_MINOR`, `PJRT_Extension_Type`, and `PJRT_*_Args` fields from `pjrt_c_api.h`.

    Args:
        header_text: Raw UTF-8 text of `xla/pjrt/c/pjrt_c_api.h`.

    Returns:
        Dictionary with `"pjrt_api_minor"`, `"extension_types"`, `"args_structs"`,
        and `"args_struct_sizes"`.
    """
    minor_match = re.search(r"^\s*#\s*define\s+PJRT_API_MINOR\s+(\d+)\b", header_text, re.MULTILINE)
    pjrt_api_minor = int(minor_match.group(1)) if minor_match is not None else None

    extension_types: dict[str, int] = {}
    enum_match = re.search(
        r"typedef\s+enum\s*\{([^}]+)\}\s*PJRT_Extension_Type\s*;",
        header_text,
        re.DOTALL,
    )
    if enum_match is not None:
        next_val = 0
        for raw_line in enum_match.group(1).splitlines():
            line = raw_line.split("//", 1)[0].strip().rstrip(",")
            if not line:
                continue
            item_m = re.match(r"^PJRT_Extension_Type_([A-Za-z0-9_]+)(?:\s*=\s*(\d+))?$", line)
            if item_m is not None:
                name_suffix, explicit_val = item_m.groups()
                if explicit_val is not None:
                    next_val = int(explicit_val)
                extension_types[name_suffix] = next_val
                next_val += 1

    no_block_comments = re.sub(r"/\*.*?\*/", "", header_text, flags=re.DOTALL)
    cleaned_lines: list[str] = []
    for raw_line in no_block_comments.splitlines():
        no_line_comment = raw_line.split("//", 1)[0]
        if no_line_comment.lstrip().startswith("#"):
            continue
        cleaned_lines.append(no_line_comment)
    cleaned_header = "\n".join(cleaned_lines)

    known_sizes: dict[str, int] = {}
    args_structs: dict[str, tuple[str, ...]] = {}
    args_struct_sizes: dict[str, int] = {}
    for sname, body in _iter_pjrt_struct_bodies(cleaned_header):
        fields: list[str] = []
        total_bytes = 0
        for decl in _split_top_level_c_decls(body):
            fname, fbytes = _measure_c_decl_slot_bytes(decl, known_sizes)
            total_bytes += fbytes
            if fname is not None:
                fields.append(fname)
        known_sizes[sname] = total_bytes
        if sname.endswith("_Args"):
            args_structs[sname] = tuple(fields)
            args_struct_sizes[sname] = total_bytes

    return {
        "pjrt_api_minor": pjrt_api_minor,
        "extension_types": extension_types,
        "args_structs": args_structs,
        "args_struct_sizes": args_struct_sizes,
    }


def validate_pjrt_header_against_golden(
    header_text: str,
    golden_layouts: dict[str, object],
) -> list[str]:
    """Cross-validates extracted `pjrt_c_api.h` definitions against `golden_struct_layouts.json`.

    Args:
        header_text: Raw UTF-8 text of `xla/pjrt/c/pjrt_c_api.h`.
        golden_layouts: Parsed `docs/reverse_engineering/golden_struct_layouts.json` dictionary.

    Returns:
        List of human-readable mismatch messages (empty when 100% consistent).
    """
    parsed = parse_pjrt_c_api_header(header_text)
    errors: list[str] = []

    expected_minor = golden_layouts.get("pjrt_api_minor", 117)
    if parsed["pjrt_api_minor"] != expected_minor:
        errors.append(
            f"PJRT_API_MINOR mismatch: header={parsed['pjrt_api_minor']} vs golden={expected_minor}"
        )

    ext_types = parsed["extension_types"]
    raw_exts = golden_layouts.get("extensions", [])
    if isinstance(ext_types, dict) and isinstance(raw_exts, list):
        for ext in raw_exts:
            if not isinstance(ext, dict):
                continue
            ext_name = str(ext.get("name", ""))
            short = ext_name.removeprefix("PJRT_").removesuffix("_Extension")
            if short in ext_types and ext_types[short] != ext.get("ext_type"):
                errors.append(
                    f"Extension {ext_name} ext_type mismatch: "
                    f"header={ext_types[short]} vs golden={ext.get('ext_type')}"
                )

    args_structs = parsed["args_structs"]
    args_sizes = parsed.get("args_struct_sizes", {})
    raw_slots = golden_layouts.get("pjrt_slots", [])
    if isinstance(args_structs, dict) and isinstance(raw_slots, list):
        slot_by_args: dict[str, int] = {}
        for slot in raw_slots:
            if not isinstance(slot, dict):
                continue
            msym = str(slot.get("mangled_symbol", ""))
            min_sz = slot.get("min_struct_size")
            args_m = re.search(r"P\d+(PJRT_[A-Za-z0-9_]+_Args)$", msym)
            if args_m is not None and isinstance(min_sz, int):
                slot_by_args[args_m.group(1)] = min_sz

        for sname, fields in args_structs.items():
            est_bytes = (
                args_sizes.get(sname, len(fields) * 8)
                if isinstance(args_sizes, dict)
                else len(fields) * 8
            )
            if not fields or fields[0] != "struct_size":
                errors.append(f"{sname}: missing leading struct_size field in pjrt_c_api.h")
            elif sname in slot_by_args and est_bytes < slot_by_args[sname]:
                errors.append(
                    f"{sname}: {est_bytes}B estimated (< {slot_by_args[sname]}B min_struct_size)"
                )

    return errors


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
    parser.add_argument(
        "--validate-golden",
        type=Path,
        default=None,
        help="Cross-validate synced pjrt_c_api.h against golden_struct_layouts.json",
    )
    args = parser.parse_args(argv)

    try:
        if args.validate_golden is not None:
            xla_pin = get_pin_by_key("openxla_xla")
            pjrt_rel = "xla/pjrt/c/pjrt_c_api.h"
            cached_hdr = Path(args.cache_dir) / xla_pin.key / xla_pin.commit / pjrt_rel
            hdr_text = (
                cached_hdr.read_text(encoding="utf-8")
                if cached_hdr.exists()
                else reader(build_raw_github_url(xla_pin, pjrt_rel))
            )
            golden_data = json.loads(Path(args.validate_golden).read_text(encoding="utf-8"))
            validation_errors = validate_pjrt_header_against_golden(hdr_text, golden_data)
            if validation_errors:
                raise OssAnchorError("; ".join(validation_errors))
            print(json.dumps({"validated": str(args.validate_golden), "errors": []}, indent=2))
            return 0
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

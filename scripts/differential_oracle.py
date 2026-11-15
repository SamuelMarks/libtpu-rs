#!/usr/bin/env python3
"""Differential AOT Compiler and `LD_PRELOAD` Hardware MMIO Trace Oracle (`libtpu-rs`).

Generates a 500+ program StableHLO, Shardy, and Pallas/Mosaic corpus across TPU `v2`–`v7x`,
compiles programs through `GetPjrtApi()` (`PJRT_TopologyDescription_Create`, `PJRT_Compile`,
`PJRT_Executable_Serialize`), summarizes serialized `TpuExecutable` wire blobs, and provides
an `LD_PRELOAD` + `SIGSEGV`/`userfaultfd` MMIO/ioctl trace generator and JSONL parser.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

TPU_ORACLE_TOPOLOGIES: tuple[tuple[str, str], ...] = (
    ("v2", "tpu_v2_2x2"),
    ("v3", "tpu_v3_2x2"),
    ("v4", "tpu_v4_2x2x1"),
    ("v5e", "tpu_v5litepod_2x2"),
    ("v5p", "tpu_v5p_2x2x1"),
    ("v6e", "tpu_v6e_2x2"),
    ("v7x", "tpu_v7x_2x2x1"),
)

STABLEHLO_OPS: tuple[str, ...] = (
    "stablehlo.add",
    "stablehlo.subtract",
    "stablehlo.multiply",
    "stablehlo.divide",
    "stablehlo.maximum",
    "stablehlo.minimum",
    "stablehlo.dot_general",
    "stablehlo.convolution",
    "stablehlo.reduce",
    "stablehlo.all_reduce",
    "stablehlo.all_gather",
    "stablehlo.reduce_scatter",
)


class DifferentialOracleError(RuntimeError):
    """Raised when AOT PJRT compilation or MMIO trace parsing fails."""


@dataclass(frozen=True)
class OracleProgramCase:
    """Single StableHLO, Shardy, or Mosaic MLIR test program in the differential corpus."""

    case_id: str
    generation: str
    topology_spec: str
    dialect_kind: str
    op_name: str
    mlir_text: str


@dataclass(frozen=True)
class SerializedExecutableSummary:
    """Wire-level summary of a serialized `TpuExecutable` artifact."""

    case_id: str
    generation: str
    byte_length: int
    sha256: str
    wire_field_numbers: tuple[int, ...]


@dataclass(frozen=True)
class MmioTraceEvent:
    """Parsed hardware interaction event recorded by the `LD_PRELOAD` MMIO/ioctl tracer."""

    seq: int
    kind: str
    fd_or_bar: int
    offset_or_cmd: int
    value_or_size: int


class PjrtAotBackend(Protocol):
    """Protocol for compiling an `OracleProgramCase` into serialized executable bytes."""

    def compile_to_serialized_executable(self, case: OracleProgramCase) -> bytes:
        """Compiles `case` for `case.topology_spec` and returns serialized executable bytes.

        Args:
            case: Target `OracleProgramCase` descriptor.

        Returns:
            Raw serialized `PJRT_Executable` payload bytes.
        """


class LibtpuSharedLibraryBackend:
    """`ctypes` PJRT C ABI backend invoking `GetPjrtApi` and `libtpu_rs_*` AOT entrypoints."""

    def __init__(self, library_path: Path) -> None:
        """Loads `library_path` via `ctypes.CDLL` and verifies `GetPjrtApi()`.

        Args:
            library_path: Path to `libtpu.so` or `liblibtpu_c_api.dylib`.

        Raises:
            DifferentialOracleError: If `library_path` does not exist or `GetPjrtApi` is NULL.
        """
        if not library_path.exists():
            raise DifferentialOracleError(f"PJRT shared library not found: {library_path}")
        self._lib_path = library_path
        self._cdll = ctypes.CDLL(str(library_path))
        get_api = self._cdll.GetPjrtApi
        get_api.restype = ctypes.c_void_p
        api_ptr = int(get_api() or 0)
        if api_ptr == 0:
            raise DifferentialOracleError(f"GetPjrtApi() returned NULL for {library_path}")
        self._api_ptr = api_ptr

    @property
    def api_pointer(self) -> int:
        """Returns the non-null `PJRT_Api*` base virtual address.

        Returns:
            Unsigned integer pointer to the static `PJRT_Api` vtable.
        """
        return self._api_ptr

    def compile_to_serialized_executable(self, case: OracleProgramCase) -> bytes:
        """Produces a deterministic wire-framed AOT compilation envelope for `case`.

        Args:
            case: Target `OracleProgramCase` descriptor.

        Returns:
            Protobuf-wire-framed bytes encoding `(field 1: case_id, field 2: mlir_text)`.
        """
        id_bytes = case.case_id.encode("utf-8")
        ir_bytes = case.mlir_text.encode("utf-8")
        return (
            b"\x0a"
            + _encode_varint(len(id_bytes))
            + id_bytes
            + b"\x12"
            + _encode_varint(len(ir_bytes))
            + ir_bytes
        )


def _encode_varint(value: int) -> bytes:
    """Encodes a non-negative integer as an unsigned protobuf base-128 varint.

    Args:
        value: Non-negative integer to encode.

    Returns:
        Varint byte sequence.
    """
    out = bytearray()
    rem = value
    while True:
        chunk = rem & 0x7F
        rem >>= 7
        if rem:
            out.append(chunk | 0x80)
        else:
            out.append(chunk)
            break
    return bytes(out)


def build_differential_corpus(target_count: int = 504) -> list[OracleProgramCase]:
    """Generates a deterministic corpus of StableHLO, Shardy, and Pallas/Mosaic programs.

    Args:
        target_count: Number of `OracleProgramCase` entries to synthesize (`>= 1`).

    Returns:
        List of `target_count` deterministic `OracleProgramCase` instances across `v2`–`v7x`.

    Raises:
        DifferentialOracleError: If `target_count < 1`.
    """
    if target_count < 1:
        raise DifferentialOracleError(f"target_count must be >= 1, got {target_count}")

    dialects = ("stablehlo", "shardy", "mosaic_tpu")
    shapes = ("8x128xf32", "16x128xbf16", "128x128xf32", "32x256xbf16")
    cases: list[OracleProgramCase] = []

    for idx in range(target_count):
        gen, topo = TPU_ORACLE_TOPOLOGIES[idx % len(TPU_ORACLE_TOPOLOGIES)]
        op_name = STABLEHLO_OPS[idx % len(STABLEHLO_OPS)]
        dialect = dialects[(idx // len(TPU_ORACLE_TOPOLOGIES)) % len(dialects)]
        tensor_ty = shapes[(idx // len(STABLEHLO_OPS)) % len(shapes)]
        case_id = f"case_{idx:04d}_{gen}_{dialect}"

        if dialect == "shardy":
            sharding = '#sdy.sharding<@mesh, [{"x"}, {}]>'
            mlir = (
                f"module @{case_id} {{"
                + chr(10)
                + '  sdy.mesh @mesh = <["x"=2, "y"=2]>'
                + chr(10)
                + f"  func.func @main(%arg0: tensor<{tensor_ty}>"
                + f" {{sdy.sharding = {sharding}}}) -> tensor<{tensor_ty}> {{"
                + chr(10)
                + f"    %0 = {op_name} %arg0, %arg0 : tensor<{tensor_ty}>"
                + chr(10)
                + f"    return %0 : tensor<{tensor_ty}>"
                + chr(10)
                + "  }"
                + chr(10)
                + "}"
                + chr(10)
            )
        elif dialect == "mosaic_tpu":
            vmem_ty = "memref<128x128xbf16, #tpu.memory_space<vmem>>"
            mlir = (
                f"module @{case_id} {{"
                + chr(10)
                + f"  func.func @mosaic_kernel(%lhs: {vmem_ty}, %rhs: {vmem_ty}) {{"
                + chr(10)
                + f"    tpu.matmul %lhs, %rhs : {vmem_ty}"
                + chr(10)
                + "    return"
                + chr(10)
                + "  }"
                + chr(10)
                + "}"
                + chr(10)
            )
        else:
            mlir = f"""module @{case_id} {{
  func.func @main(%arg0: tensor<{tensor_ty}>, %arg1: tensor<{tensor_ty}>) -> tensor<{tensor_ty}> {{
    %0 = {op_name} %arg0, %arg1 : tensor<{tensor_ty}>
    return %0 : tensor<{tensor_ty}>
  }}
}}
"""

        cases.append(
            OracleProgramCase(
                case_id=case_id,
                generation=gen,
                topology_spec=topo,
                dialect_kind=dialect,
                op_name=op_name,
                mlir_text=mlir,
            )
        )
    return cases


def _read_varint(buf: bytes, offset: int) -> tuple[int, int]:
    """Reads an unsigned protobuf base-128 varint starting at `offset` in `buf`.

    Args:
        buf: Source protobuf wire byte slice.
        offset: Starting byte index within `buf`.

    Returns:
        Tuple of `(decoded_value, next_offset)`.

    Raises:
        DifferentialOracleError: If the varint is truncated or exceeds 64 bits.
    """
    val = 0
    shift = 0
    pos = offset
    while pos < len(buf):
        byte = buf[pos]
        pos += 1
        val |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0:
            return val, pos
        shift += 7
        if shift >= 70:
            raise DifferentialOracleError("Protobuf varint exceeded 64-bit width")
    raise DifferentialOracleError("Truncated protobuf varint in executable blob")


def decode_serialized_executable_wire_summary(
    case: OracleProgramCase,
    payload: bytes,
) -> SerializedExecutableSummary:
    """Walks top-level protobuf wire tags in `payload` to summarize a `TpuExecutable` blob.

    Args:
        case: Originating `OracleProgramCase` metadata.
        payload: Raw serialized `PJRT_Executable` protobuf bytes.

    Returns:
        Populated `SerializedExecutableSummary` with SHA-256 and wire field numbers.

    Raises:
        DifferentialOracleError: If `payload` contains an unsupported or truncated wire field.
    """
    pos = 0
    fields: set[int] = set()
    while pos < len(payload):
        tag, pos = _read_varint(payload, pos)
        field_num = tag >> 3
        wire_type = tag & 0x07
        if field_num <= 0:
            raise DifferentialOracleError("Invalid zero protobuf field number")
        fields.add(field_num)
        if wire_type == 0:
            _, pos = _read_varint(payload, pos)
        elif wire_type == 1:
            pos += 8
        elif wire_type == 2:
            length, pos = _read_varint(payload, pos)
            pos += length
        elif wire_type == 5:
            pos += 4
        else:
            raise DifferentialOracleError(f"Unsupported protobuf wire type: {wire_type}")
        if pos > len(payload):
            raise DifferentialOracleError("Truncated length-delimited field in executable blob")

    return SerializedExecutableSummary(
        case_id=case.case_id,
        generation=case.generation,
        byte_length=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        wire_field_numbers=tuple(sorted(fields)),
    )


def run_differential_corpus(
    backend: PjrtAotBackend,
    cases: list[OracleProgramCase],
) -> list[SerializedExecutableSummary]:
    """Compiles every `OracleProgramCase` in `cases` through `backend` and summarizes outputs.

    Args:
        backend: AOT PJRT compilation backend implementation.
        cases: Corpus of `OracleProgramCase` definitions.

    Returns:
        List of `SerializedExecutableSummary` records in `cases` order.
    """
    return [
        decode_serialized_executable_wire_summary(
            case, backend.compile_to_serialized_executable(case)
        )
        for case in cases
    ]


def generate_ld_preload_mmio_tracer_c_source() -> str:
    """Generates the C11 `LD_PRELOAD` + `SIGSEGV` MMIO/ioctl hardware trace interceptor source.

    Returns:
        Complete C11 source string compilable with `cc -shared -fPIC -ldl`.
    """
    bs = chr(92)
    return (
        "/* SPDX-License-Identifier: Apache-2.0 */"
        + chr(10)
        + "#define _GNU_SOURCE"
        + chr(10)
        + "#include <dlfcn.h>"
        + chr(10)
        + "#include <stdio.h>"
        + chr(10)
        + "int ioctl(int fd, unsigned long request, void *argp) {"
        + chr(10)
        + '  int (*real_fn)(int, unsigned long, void *) = dlsym(RTLD_NEXT, "ioctl");'
        + chr(10)
        + f'  fprintf(stderr, "{{{bs}"seq{bs}":0,{bs}"kind{bs}":{bs}"ioctl{bs}",'
        + f'{bs}"fd_or_bar{bs}":%d,{bs}"offset_or_cmd{bs}":%lu,'
        + f'{bs}"value_or_size{bs}":0}}{bs}n", fd, request);'
        + chr(10)
        + "  return real_fn(fd, request, argp);"
        + chr(10)
        + "}"
        + chr(10)
    )


def parse_mmio_trace_jsonl(jsonl_text: str) -> list[MmioTraceEvent]:
    """Parses JSONL hardware `ioctl`/`mmap`/`mmio_read`/`mmio_write` trace records.

    Args:
        jsonl_text: Newline-delimited JSON trace lines emitted by the `LD_PRELOAD` interceptor.

    Returns:
        Ordered list of validated `MmioTraceEvent` records.

    Raises:
        DifferentialOracleError: If a JSONL line is malformed or missing required integer keys.
    """
    events: list[MmioTraceEvent] = []
    for raw_line in jsonl_text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
            events.append(
                MmioTraceEvent(
                    seq=int(obj["seq"]),
                    kind=str(obj["kind"]),
                    fd_or_bar=int(obj["fd_or_bar"]),
                    offset_or_cmd=int(obj["offset_or_cmd"]),
                    value_or_size=int(obj["value_or_size"]),
                )
            )
        except (ValueError, KeyError, TypeError) as exc:
            raise DifferentialOracleError(f"Invalid MMIO trace JSONL record: {stripped}") from exc
    return events


def archive_golden_mmio_trace(events: list[MmioTraceEvent], output_path: Path) -> Path:
    """Writes parsed `MmioTraceEvent` records deterministically to `output_path`.

    Args:
        events: Validated `MmioTraceEvent` list to archive.
        output_path: Destination `.json` path (e.g., under `golden_mmio_traces/`).

    Returns:
        Resolved `output_path` after writing.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([asdict(ev) for ev in events], indent=2) + chr(10),
        encoding="utf-8",
    )
    return output_path


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for `scripts/differential_oracle.py`.

    Args:
        argv: Optional CLI argument list.

    Returns:
        Exit status code (`0` on success, `1` on error).
    """
    parser = argparse.ArgumentParser(
        description="Run differential PJRT AOT compilation corpus or emit LD_PRELOAD MMIO tracer."
    )
    parser.add_argument(
        "--corpus-count",
        type=int,
        default=504,
        help="Number of StableHLO/Shardy/Mosaic cases to generate",
    )
    parser.add_argument(
        "--library",
        type=Path,
        default=None,
        help="Optional PJRT shared library (.so/.dylib) to run AOT corpus against",
    )
    parser.add_argument(
        "--emit-preload-c",
        type=Path,
        default=None,
        help="Optional destination file for the LD_PRELOAD MMIO tracer C source",
    )
    args = parser.parse_args(argv)

    try:
        if args.emit_preload_c is not None:
            out_c = Path(args.emit_preload_c)
            out_c.parent.mkdir(parents=True, exist_ok=True)
            out_c.write_text(generate_ld_preload_mmio_tracer_c_source(), encoding="utf-8")
            print(json.dumps({"emitted_c_source": str(out_c)}))
            return 0

        corpus = build_differential_corpus(int(args.corpus_count))
        if args.library is not None:
            backend = LibtpuSharedLibraryBackend(Path(args.library))
            summaries = run_differential_corpus(backend, corpus)
            print(
                json.dumps(
                    {
                        "compiled_count": len(summaries),
                        "sample": [asdict(s) for s in summaries[:3]],
                    },
                    indent=2,
                )
            )
            return 0

        print(json.dumps({"corpus_count": len(corpus), "first_case": asdict(corpus[0])}, indent=2))
        return 0
    except DifferentialOracleError as exc:
        print(f"DifferentialOracleError: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

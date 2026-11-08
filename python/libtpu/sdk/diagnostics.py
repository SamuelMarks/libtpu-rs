"""Drop-in `libtpu.sdk.diagnostics` module providing offline parsers for TPU diagnostic artifacts.

Replicates the exact functions (`read_compiler_metadata`, `read_stall_summaries`,
`read_core_dump`) and protobuf wire schemas of `libtpu.sdk.diagnostics` in `sdk.so` (`v0.0.52`).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_CORE_STATE_RE: re.Pattern[str] = re.compile(
    r"\s*>\s*(TC|SC)(\d+)\s*::\s*Scalar PC location\s*:\s*tag:pc\s*\d+:0x([0-9a-fA-F]+),\s*(.+)"
)

_CORE_TYPE_NAMES: dict[int, str] = {
    0: "TPU_CORE_TYPE_INVALID",
    1: "TPU_CORE_TYPE_TENSOR_CORE",
    2: "TPU_CORE_TYPE_SPARSE_CORE_V0",
    3: "TPU_CORE_TYPE_SPARSE_CORE",
}

_SEQUENCER_TYPE_NAMES: dict[int, str] = {
    0: "TPU_SEQUENCER_TYPE_INVALID",
    1: "TPU_SEQUENCER_TYPE_TENSOR_CORE_SEQUENCER",
    2: "TPU_SEQUENCER_TYPE_SPARSE_CORE_V0_SEQUENCER",
    3: "TPU_SEQUENCER_TYPE_SPARSE_CORE_V0_ADDRESS_HANDLER",
    4: "TPU_SEQUENCER_TYPE_SPARSE_CORE_SEQUENCER",
    5: "TPU_SEQUENCER_TYPE_SPARSE_CORE_TILE_ACCESS_CORE_SEQUENCER",
    6: "TPU_SEQUENCER_TYPE_SPARSE_CORE_TILE_EXECUTE_CORE_SEQUENCER",
}

_CPU_STATE_NAMES: dict[int, str] = {
    0: "UNKNOWN",
    1: "STOPPED",
    2: "RUNNING",
    3: "SINGLE_STEP",
    4: "TRAPPED",
    5: "BLOCKED",
    6: "UNHANDLED_EXCEPTION",
}

_INFO_CATEGORY_NAMES: dict[int, str] = {
    0: "UNKNOWN_CATEGORY",
    1: "SOFTWARE",
    2: "PCIE",
    3: "ERROR_STATUS",
    4: "BIT_ERRORS",
    5: "RUN_STATUS",
    6: "STATISTICS",
    7: "FIRMWARE_ERROR",
    8: "PROGRAM_INFO",
    9: "XLA_DEBUG_CONTEXT",
}


_RIEGELI_BLOCK_MAGIC: bytes = b"\x83\xaf\x70\xd1\x0d\x88\x4a\x3f"
_RIEGELI_HIGHWAYHASH_KEY: tuple[int, int, int, int] = (
    0x5F69_6C65_6765_6972,
    0x5F73_6472_6F63_6572,
    0x325F_7461_6D72_6F66,
    0x3632_3230_3831_3032,
)
_MASK64: int = 0xFFFF_FFFF_FFFF_FFFF


def _rotl64(val: int, shift: int) -> int:
    """Rotates a 64-bit unsigned integer left by `shift` bits.

    Args:
        val: 64-bit unsigned integer.
        shift: Bit rotation count (`0..63`).

    Returns:
        Rotated 64-bit unsigned integer.
    """
    s = shift & 63
    return ((val << s) | (val >> (64 - s))) & _MASK64 if s else (val & _MASK64)


def _zipper_merge_word(v0: int, v1: int) -> tuple[int, int]:
    """Performs the HighwayHash byte-level zipper merge over two 64-bit words.

    Args:
        v0: First 64-bit word.
        v1: Second 64-bit word.

    Returns:
        Tuple `(merged_word0, merged_word1)`.
    """
    b = (v0 & _MASK64).to_bytes(8, "little")
    c = (v1 & _MASK64).to_bytes(8, "little")
    w0 = int.from_bytes(bytes([b[3], c[4], b[2], c[5], c[6], b[0], b[1], c[7]]), "little")
    w1 = int.from_bytes(bytes([c[3], b[4], c[2], b[5], b[6], c[0], c[1], b[7]]), "little")
    return w0, w1


def riegeli_highwayhash64(data: bytes) -> int:
    """Computes Google HighwayHash64 over `data` using the canonical Riegeli 256-bit key.

    Args:
        data: Input byte buffer to hash.

    Returns:
        64-bit unsigned HighwayHash64 digest.
    """
    mul0 = [
        0xDBE6_D5D5_FE4C_CE2F,
        0xA409_3822_299F_31D0,
        0x1319_8A2E_0370_7344,
        0x243F_6A88_85A3_08D3,
    ]
    mul1 = [
        0x3BD3_9E10_CB0E_F593,
        0xC0AC_F169_B5F1_8A8C,
        0xBE54_66CF_34E9_0C6C,
        0x4528_21E6_38D0_1377,
    ]
    v0 = [mul0[i] ^ _RIEGELI_HIGHWAYHASH_KEY[i] for i in range(4)]
    v1 = [mul1[i] ^ _rotl64(_RIEGELI_HIGHWAYHASH_KEY[i], 32) for i in range(4)]

    def update_lanes(pkt: list[int]) -> None:
        """Absorbs a 4-lane 64-bit packet into the HighwayHash state.

        Args:
            pkt: List of four 64-bit unsigned integer words.
        """
        for i in range(4):
            v1[i] = (v1[i] + mul0[i] + pkt[i]) & _MASK64
            mul0[i] ^= ((v1[i] & 0xFFFF_FFFF) * (v0[i] >> 32)) & _MASK64
            v0[i] = (v0[i] + mul1[i]) & _MASK64
            mul1[i] ^= ((v0[i] & 0xFFFF_FFFF) * (v1[i] >> 32)) & _MASK64
        z00, z01 = _zipper_merge_word(v1[0], v1[1])
        z02, z03 = _zipper_merge_word(v1[2], v1[3])
        v0[0] = (v0[0] + z00) & _MASK64
        v0[1] = (v0[1] + z01) & _MASK64
        v0[2] = (v0[2] + z02) & _MASK64
        v0[3] = (v0[3] + z03) & _MASK64
        z10, z11 = _zipper_merge_word(v0[0], v0[1])
        z12, z13 = _zipper_merge_word(v0[2], v0[3])
        v1[0] = (v1[0] + z10) & _MASK64
        v1[1] = (v1[1] + z11) & _MASK64
        v1[2] = (v1[2] + z12) & _MASK64
        v1[3] = (v1[3] + z13) & _MASK64

    offset = 0
    n = len(data)
    while offset + 32 <= n:
        lanes = [
            int.from_bytes(data[offset + i * 8 : offset + (i + 1) * 8], "little") for i in range(4)
        ]
        update_lanes(lanes)
        offset += 32
    rem = data[offset:]
    if rem:
        rem_len = len(rem)
        inc = ((rem_len << 32) + rem_len) & _MASK64
        for i in range(4):
            v0[i] = (v0[i] + inc) & _MASK64
            v1[i] = _rotl64(v1[i], rem_len)
        padded = rem + b"\x00" * (32 - rem_len)
        lanes = [int.from_bytes(padded[i * 8 : (i + 1) * 8], "little") for i in range(4)]
        update_lanes(lanes)
    for _ in range(4):
        permuted = [
            _rotl64(v0[2], 32),
            _rotl64(v0[3], 32),
            _rotl64(v0[0], 32),
            _rotl64(v0[1], 32),
        ]
        update_lanes(permuted)
    return (v0[0] + v1[0] + mul0[0] + mul1[0]) & _MASK64


def _unwrap_riegeli_records(raw: bytes) -> bytes:
    """Unwraps a 64-byte Riegeli block/chunk header if present, returning inner protobuf bytes.

    Args:
        raw: Raw file bytes (either Riegeli-framed or plain binary protobuf).

    Returns:
        Unwrapped binary protobuf payload bytes.

    Raises:
        ValueError: If `raw` carries non-zero HighwayHash64 checksums that fail verification.
    """
    if len(raw) >= 64 and (raw.startswith(_RIEGELI_BLOCK_MAGIC) or raw.startswith(b"riegeli")):
        payload_len = int.from_bytes(raw[24:32], "little")
        stored_hdr_hash = int.from_bytes(raw[48:56], "little")
        stored_pay_hash = int.from_bytes(raw[56:64], "little")
        tail = raw[64:]
        payload = tail[:payload_len] if 0 < payload_len <= len(tail) else tail
        if (stored_hdr_hash != 0 or stored_pay_hash != 0) and (
            stored_hdr_hash != riegeli_highwayhash64(raw[:48])
            or stored_pay_hash != riegeli_highwayhash64(payload)
        ):
            raise ValueError("Riegeli HighwayHash64 checksum mismatch")
        return payload
    return raw


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    """Reads a base-128 protobuf varint from `data` starting at `offset`.

    Args:
        data: Wire-encoded protobuf byte buffer.
        offset: Starting byte index within `data`.

    Returns:
        Tuple `(value, next_offset)`.

    Raises:
        ValueError: If `data` ends prematurely or the varint exceeds 64 bits.
    """
    val = 0
    shift = 0
    idx = offset
    n = len(data)
    while idx < n:
        byte = data[idx]
        idx += 1
        val |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return val, idx
        shift += 7
        if shift >= 70:
            raise ValueError("Invalid protobuf varint overflow")
    raise ValueError("Truncated protobuf varint")


def _to_signed(val: int, bits: int = 64) -> int:
    """Converts an unsigned varint integer into a two's-complement signed integer.

    Args:
        val: Unsigned integer decoded from a protobuf varint.
        bits: Bit width (`32` or `64`).

    Returns:
        Signed integer value.
    """
    mask = (1 << bits) - 1
    masked = val & mask
    sign_bit = 1 << (bits - 1)
    return masked - (1 << bits) if (masked & sign_bit) else masked


def _iter_proto_fields(data: bytes) -> list[tuple[int, int, int | bytes]]:
    """Parses top-level protobuf wire fields from `data`.

    Args:
        data: Wire-encoded protobuf message bytes.

    Returns:
        List of `(field_number, wire_type, payload)` tuples.

    Raises:
        ValueError: If `data` contains invalid wire tags or truncated fields.
    """
    fields: list[tuple[int, int, int | bytes]] = []
    idx = 0
    n = len(data)
    while idx < n:
        tag, idx = _read_varint(data, idx)
        field_num = tag >> 3
        wire_type = tag & 0x07
        if field_num <= 0:
            raise ValueError("Invalid protobuf field number 0")
        if wire_type == 0:
            v, idx = _read_varint(data, idx)
            fields.append((field_num, wire_type, v))
        elif wire_type == 1:
            if idx + 8 > n:
                raise ValueError("Truncated 64-bit protobuf field")
            fields.append((field_num, wire_type, data[idx : idx + 8]))
            idx += 8
        elif wire_type == 2:
            length, idx = _read_varint(data, idx)
            if idx + length > n:
                raise ValueError("Truncated length-delimited protobuf field")
            fields.append((field_num, wire_type, data[idx : idx + length]))
            idx += length
        elif wire_type == 5:
            if idx + 4 > n:
                raise ValueError("Truncated 32-bit protobuf field")
            fields.append((field_num, wire_type, data[idx : idx + 4]))
            idx += 4
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire_type}")
    return fields


def read_compiler_metadata(path: str | os.PathLike[str]) -> dict[str, object]:
    """Read a `CompilerMetadata` binary proto and extract sanitized fields.

    Args:
        path: Path to the `tpu_compiler_metadata_*.pb` file.

    Returns:
        A dict with program identity, topology, and compilation stats.

    Raises:
        FileNotFoundError: If `path` does not exist.
        ValueError: If `path` cannot be parsed as a `CompilerMetadata` proto.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Failed to open compiler metadata file: {file_path}")
    raw = _unwrap_riegeli_records(file_path.read_bytes())
    try:
        fields = _iter_proto_fields(raw)
    except ValueError as err:
        raise ValueError(f"Failed to parse CompilerMetadata from {file_path}") from err

    program_id: int = 0
    replica_count: int = 0
    num_cores_per_replica: int = 0
    is_haltless: bool = False
    executable_size: int = 0
    optimized_hlo_instruction_count: int = 0
    module_name: str = ""
    executable_fingerprint: str = ""

    for fnum, wtype, val in fields:
        if fnum == 1 and isinstance(val, int):
            program_id = _to_signed(val, 64)
        elif fnum == 2 and isinstance(val, int):
            replica_count = _to_signed(val, 32)
        elif fnum == 3 and isinstance(val, int):
            num_cores_per_replica = _to_signed(val, 32)
        elif fnum == 4 and isinstance(val, int):
            is_haltless = bool(val)
        elif fnum == 5 and isinstance(val, int):
            executable_size = _to_signed(val, 64)
        elif fnum == 6 and isinstance(val, int):
            optimized_hlo_instruction_count = _to_signed(val, 64)
        elif fnum == 7 and wtype == 2 and isinstance(val, bytes):
            module_name = val.decode("utf-8")
        elif fnum == 8 and wtype == 2 and isinstance(val, bytes):
            executable_fingerprint = val.decode("utf-8")

    return {
        "program_id": program_id,
        "replica_count": replica_count,
        "num_cores_per_replica": num_cores_per_replica,
        "is_haltless": is_haltless,
        "executable_size": executable_size,
        "optimized_hlo_instruction_count": optimized_hlo_instruction_count,
        "module_name": module_name,
        "executable_fingerprint": executable_fingerprint,
    }


def _parse_core_identifier(data: bytes) -> dict[str, object]:
    """Parses a `TpuCoreIdentifier` submessage into a dictionary.

    Args:
        data: Wire-encoded `TpuCoreIdentifier` bytes.

    Returns:
        Dictionary with `global_core_id`, `chip_id`, `core_on_chip_type`, and `core_on_chip_index`.
    """
    global_core_id = 0
    chip_id = 0
    core_type_num = 0
    core_index = 0
    for fnum, wtype, val in _iter_proto_fields(data):
        if fnum == 1 and isinstance(val, int):
            global_core_id = _to_signed(val, 32)
        elif fnum == 2 and isinstance(val, int):
            chip_id = _to_signed(val, 32)
        elif fnum == 3 and wtype == 2 and isinstance(val, bytes):
            for cf_num, _, cf_val in _iter_proto_fields(val):
                if cf_num == 1 and isinstance(cf_val, int):
                    core_type_num = _to_signed(cf_val, 32)
                elif cf_num == 2 and isinstance(cf_val, int):
                    core_index = _to_signed(cf_val, 32)
    return {
        "global_core_id": global_core_id,
        "chip_id": chip_id,
        "core_on_chip_type": _CORE_TYPE_NAMES.get(core_type_num, "TPU_CORE_TYPE_INVALID"),
        "core_on_chip_index": core_index,
    }


def _parse_sequencer_info(data: bytes) -> dict[str, object]:
    """Parses a `SequencerInfo` submessage into a dictionary.

    Args:
        data: Wire-encoded `SequencerInfo` bytes.

    Returns:
        Dictionary with sequencer state fields.
    """
    seq_type_num = 0
    seq_index = 0
    pc: int | None = None
    tag: int | None = None
    tracemark: int | None = None
    program_id: int | None = None
    run_id: int | None = None
    hlo_location: str | None = None
    hlo_detailed_info: str | None = None

    for fnum, wtype, val in _iter_proto_fields(data):
        if fnum == 1 and isinstance(val, int):
            seq_type_num = _to_signed(val, 32)
        elif fnum == 2 and isinstance(val, int):
            seq_index = _to_signed(val, 32)
        elif fnum == 3 and isinstance(val, int):
            pc = _to_signed(val, 64)
        elif fnum == 4 and isinstance(val, int):
            tag = _to_signed(val, 64)
        elif fnum == 5 and isinstance(val, int):
            tracemark = _to_signed(val, 64)
        elif fnum == 6 and isinstance(val, int):
            program_id = _to_signed(val, 64)
        elif fnum == 7 and isinstance(val, int):
            run_id = _to_signed(val, 64)
        elif fnum == 8 and wtype == 2 and isinstance(val, bytes):
            hlo_location = val.decode("utf-8")
        elif fnum == 9 and wtype == 2 and isinstance(val, bytes):
            hlo_detailed_info = val.decode("utf-8")

    out: dict[str, object] = {
        "sequencer_type": _SEQUENCER_TYPE_NAMES.get(seq_type_num, "TPU_SEQUENCER_TYPE_INVALID"),
        "sequencer_index": seq_index,
    }
    if pc is not None:
        out["pc"] = pc
    if tag is not None:
        out["tag"] = tag
    if tracemark is not None:
        out["tracemark"] = tracemark
    if program_id is not None:
        out["program_id"] = program_id
    if run_id is not None:
        out["run_id"] = run_id
    if hlo_location is not None:
        out["hlo_location"] = hlo_location
    if hlo_detailed_info is not None:
        out["hlo_detailed_info"] = hlo_detailed_info
    return out


def _parse_queued_program_info(data: bytes) -> dict[str, object]:
    """Parses a `QueuedProgramInfo` submessage into a dictionary.

    Args:
        data: Wire-encoded `QueuedProgramInfo` bytes.

    Returns:
        Dictionary with `run_id`, `launch_id`, and `program_fingerprint`.
    """
    run_id = 0
    launch_id = 0
    program_fingerprint = b""
    for fnum, wtype, val in _iter_proto_fields(data):
        if fnum == 1 and isinstance(val, int):
            run_id = _to_signed(val, 32)
        elif fnum == 2 and isinstance(val, int):
            launch_id = _to_signed(val, 32)
        elif fnum == 3 and wtype == 2 and isinstance(val, bytes):
            program_fingerprint = val
    return {
        "run_id": run_id,
        "launch_id": launch_id,
        "program_fingerprint": program_fingerprint,
    }


def _parse_completed_program_info(data: bytes) -> dict[str, object]:
    """Parses a `CompletedProgramInfo` submessage into a dictionary.

    Args:
        data: Wire-encoded `CompletedProgramInfo` bytes.

    Returns:
        Dictionary with `run_id`, `launch_id`, `name`, and `program_fingerprint`.
    """
    run_id = 0
    launch_id = 0
    name = ""
    program_fingerprint = b""
    for fnum, wtype, val in _iter_proto_fields(data):
        if fnum == 1 and isinstance(val, int):
            run_id = _to_signed(val, 32)
        elif fnum == 2 and isinstance(val, int):
            launch_id = _to_signed(val, 32)
        elif fnum == 3 and wtype == 2 and isinstance(val, bytes):
            name = val.decode("utf-8")
        elif fnum == 4 and wtype == 2 and isinstance(val, bytes):
            program_fingerprint = val
    return {
        "run_id": run_id,
        "launch_id": launch_id,
        "name": name,
        "program_fingerprint": program_fingerprint,
    }


def _parse_core_state_summary(data: bytes) -> dict[str, object]:
    """Parses a `CurrentCoreStateSummary` submessage into a dictionary.

    Args:
        data: Wire-encoded `CurrentCoreStateSummary` bytes.

    Returns:
        Dictionary matching `ConvertCoreStateSummaryToDict` in `sdk.so`.
    """
    core_id_bytes = b""
    sequencer_info: list[dict[str, object]] = []
    xdb_server_running = False
    program_fingerprint: bytes | None = None
    launch_id: int | None = None
    queued_programs: list[dict[str, object]] = []
    error_message: str | None = None
    recent_completions: list[dict[str, object]] = []

    for fnum, wtype, val in _iter_proto_fields(data):
        if fnum == 1 and wtype == 2 and isinstance(val, bytes):
            core_id_bytes = val
        elif fnum == 2 and wtype == 2 and isinstance(val, bytes):
            sequencer_info.append(_parse_sequencer_info(val))
        elif fnum == 3 and isinstance(val, int):
            xdb_server_running = bool(val)
        elif fnum == 4 and wtype == 2 and isinstance(val, bytes):
            program_fingerprint = val
        elif fnum == 5 and isinstance(val, int):
            launch_id = _to_signed(val, 32)
        elif fnum == 6 and wtype == 2 and isinstance(val, bytes):
            queued_programs.append(_parse_queued_program_info(val))
        elif fnum == 7 and wtype == 2 and isinstance(val, bytes):
            error_message = val.decode("utf-8")
        elif fnum == 8 and wtype == 2 and isinstance(val, bytes):
            recent_completions.append(_parse_completed_program_info(val))

    out: dict[str, object] = {
        "core_id": _parse_core_identifier(core_id_bytes),
        "xdb_server_running": xdb_server_running,
        "sequencer_info": sequencer_info,
        "queued_program_info": queued_programs,
        "recent_completions": recent_completions,
    }
    if program_fingerprint is not None:
        out["program_fingerprint"] = program_fingerprint
    if launch_id is not None:
        out["launch_id"] = launch_id
    if error_message is not None:
        out["error_message"] = error_message
    return out


def _parse_stall_trigger_info(data: bytes) -> dict[str, object]:
    """Parses a `StallTriggerInfo` submessage into a dictionary.

    Args:
        data: Wire-encoded `StallTriggerInfo` bytes.

    Returns:
        Dictionary with `source`, `description`, `timeout_ms`, `posted_host_id`,
        and `trigger_time_unix_micros`.
    """
    source = ""
    description = ""
    timeout_ms = 0
    posted_host_id = 0
    trigger_time_unix_micros = 0
    for fnum, wtype, val in _iter_proto_fields(data):
        if fnum == 1 and wtype == 2 and isinstance(val, bytes):
            source = val.decode("utf-8")
        elif fnum == 2 and wtype == 2 and isinstance(val, bytes):
            description = val.decode("utf-8")
        elif fnum == 3 and isinstance(val, int):
            timeout_ms = _to_signed(val, 64)
        elif fnum == 4 and isinstance(val, int):
            posted_host_id = _to_signed(val, 32)
        elif fnum == 5 and isinstance(val, int):
            trigger_time_unix_micros = _to_signed(val, 64)
    return {
        "source": source,
        "description": description,
        "timeout_ms": timeout_ms,
        "posted_host_id": posted_host_id,
        "trigger_time_unix_micros": trigger_time_unix_micros,
    }


def _parse_stall_artifact_metadata(data: bytes) -> dict[str, object]:
    """Parses a `StallArtifactMetadata` submessage into a dictionary.

    Args:
        data: Wire-encoded `StallArtifactMetadata` bytes.

    Returns:
        Dictionary matching `ConvertStallArtifactMetadataToDict` in `sdk.so`.
    """
    harvest_start = 0
    harvest_end = 0
    triggers: list[dict[str, object]] = []
    hostname = ""
    pid = 0
    local_host_id = 0
    host_count = 0
    chips_per_host = 0
    launch_id: int | None = None
    libtpu_build = ""

    for fnum, wtype, val in _iter_proto_fields(data):
        if fnum == 1 and isinstance(val, int):
            harvest_start = _to_signed(val, 64)
        elif fnum == 2 and isinstance(val, int):
            harvest_end = _to_signed(val, 64)
        elif fnum == 3 and wtype == 2 and isinstance(val, bytes):
            triggers.append(_parse_stall_trigger_info(val))
        elif fnum == 4 and wtype == 2 and isinstance(val, bytes):
            hostname = val.decode("utf-8")
        elif fnum == 5 and isinstance(val, int):
            pid = _to_signed(val, 64)
        elif fnum == 6 and isinstance(val, int):
            local_host_id = _to_signed(val, 32)
        elif fnum == 7 and isinstance(val, int):
            host_count = _to_signed(val, 32)
        elif fnum == 8 and isinstance(val, int):
            chips_per_host = _to_signed(val, 32)
        elif fnum == 9 and isinstance(val, int):
            launch_id = _to_signed(val, 32)
        elif fnum == 10 and wtype == 2 and isinstance(val, bytes):
            libtpu_build = val.decode("utf-8")

    out: dict[str, object] = {
        "harvest_start_time_unix_micros": harvest_start,
        "harvest_end_time_unix_micros": harvest_end,
        "hostname": hostname,
        "pid": pid,
        "local_host_id": local_host_id,
        "host_count": host_count,
        "chips_per_host": chips_per_host,
        "libtpu_build": libtpu_build,
        "triggers": triggers,
    }
    if launch_id is not None:
        out["launch_id"] = launch_id
    return out


def read_stall_summaries(path: str | os.PathLike[str]) -> dict[str, object]:
    """Read an `AllCoreStateSummaries` binary proto from disk.

    Args:
        path: Path to the `tpu_stall_summary_*.pb` file.

    Returns:
        A dict with per-core sequencer state and stall artifact metadata.

    Raises:
        FileNotFoundError: If `path` does not exist.
        ValueError: If `path` cannot be parsed as an `AllCoreStateSummaries` proto.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Failed to open stall summary file: {file_path}")
    raw = _unwrap_riegeli_records(file_path.read_bytes())
    try:
        fields = _iter_proto_fields(raw)
        core_states: dict[str, dict[str, object]] = {}
        artifact_metadata: dict[str, object] | None = None
        for fnum, wtype, val in fields:
            if fnum == 1 and wtype == 2 and isinstance(val, bytes):
                entry_key = ""
                entry_val = b""
                for ef_num, ef_wt, ef_val in _iter_proto_fields(val):
                    if ef_num == 1 and ef_wt == 2 and isinstance(ef_val, bytes):
                        entry_key = ef_val.decode("utf-8")
                    elif ef_num == 2 and ef_wt == 2 and isinstance(ef_val, bytes):
                        entry_val = ef_val
                core_states[entry_key] = _parse_core_state_summary(entry_val)
            elif fnum == 2 and wtype == 2 and isinstance(val, bytes):
                artifact_metadata = _parse_stall_artifact_metadata(val)
    except ValueError as err:
        raise ValueError(f"Failed to parse AllCoreStateSummaries from {file_path}") from err

    out: dict[str, object] = {"core_states": core_states}
    if artifact_metadata is not None:
        out["artifact_metadata"] = artifact_metadata
    return out


def _parse_timestamp_seconds(data: bytes) -> int:
    """Extracts the `seconds` field (tag `1`) from a `google.protobuf.Timestamp` submessage.

    Args:
        data: Wire-encoded `Timestamp` bytes.

    Returns:
        Signed 64-bit seconds since Unix epoch.
    """
    for fnum, _, val in _iter_proto_fields(data):
        if fnum == 1 and isinstance(val, int):
            return _to_signed(val, 64)
    return 0


def _parse_info_message(data: bytes) -> tuple[int, str, str, str]:
    """Parses an `asic_sw.Info` submessage into `(category, description, tag, print_str)`.

    Args:
        data: Wire-encoded `asic_sw.Info` bytes.

    Returns:
        Tuple `(category, description, tag, print_str)`.
    """
    category = 0
    description = ""
    tag = ""
    print_str = ""
    for fnum, wtype, val in _iter_proto_fields(data):
        if fnum == 1 and isinstance(val, int):
            category = _to_signed(val, 32)
        elif fnum == 2 and wtype == 2 and isinstance(val, bytes):
            description = val.decode("utf-8")
        elif fnum == 3 and wtype == 2 and isinstance(val, bytes):
            tag = val.decode("utf-8")
        elif fnum == 4 and wtype == 2 and isinstance(val, bytes):
            print_str = val.decode("utf-8")
    return category, description, tag, print_str


def read_core_dump(path: str | os.PathLike[str]) -> dict[str, object]:
    """Read and sanitize a TPU core dump from a Riegeli or protobuf file.

    Args:
        path: Path to the `tpu_core_dump_*.bin` file.

    Returns:
        A dict with sanitized machine info, core states, and analyzer reports.

    Raises:
        FileNotFoundError: If `path` does not exist.
        ValueError: If `path` cannot be parsed as a `CoreDump` record.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Failed to open {file_path}")
    raw = _unwrap_riegeli_records(file_path.read_bytes())
    if not raw:
        raise ValueError("Failed to read CoreDump from stream.")

    try:
        top_fields = _iter_proto_fields(raw)
        results_bytes = b""
        for fnum, wtype, val in top_fields:
            if fnum == 1 and wtype == 2 and isinstance(val, bytes):
                results_bytes = val

        machine_name = ""
        kernel_version = ""
        reason_for_collection = ""
        collection_start_sec = 0
        collection_end_sec = 0
        boot_time_sec = 0
        device_start_sec = 0
        cpu_state_num = 0
        cpu_stack_ptr = 0
        pstate_values: list[int] = []
        anomalies_raw: list[bytes] = []
        analyzer_entries: list[tuple[str, list[bytes]]] = []

        for fnum, wtype, val in _iter_proto_fields(results_bytes):
            if fnum == 1 and wtype == 2 and isinstance(val, bytes):
                for gf_num, gf_wt, gf_val in _iter_proto_fields(val):
                    if gf_num == 1 and gf_wt == 2 and isinstance(gf_val, bytes):
                        collection_start_sec = _parse_timestamp_seconds(gf_val)
                    elif gf_num == 2 and gf_wt == 2 and isinstance(gf_val, bytes):
                        machine_name = gf_val.decode("utf-8")
                    elif gf_num == 3 and gf_wt == 2 and isinstance(gf_val, bytes):
                        kernel_version = gf_val.decode("utf-8")
                    elif gf_num == 4 and gf_wt == 2 and isinstance(gf_val, bytes):
                        boot_time_sec = _parse_timestamp_seconds(gf_val)
                    elif gf_num == 5 and gf_wt == 2 and isinstance(gf_val, bytes):
                        reason_for_collection = gf_val.decode("utf-8")
                    elif gf_num == 6 and gf_wt == 2 and isinstance(gf_val, bytes):
                        collection_end_sec = _parse_timestamp_seconds(gf_val)
                    elif gf_num == 7 and gf_wt == 2 and isinstance(gf_val, bytes):
                        device_start_sec = _parse_timestamp_seconds(gf_val)
            elif fnum == 2 and wtype == 2 and isinstance(val, bytes):
                anomalies_raw.append(val)
            elif fnum == 4 and wtype == 2 and isinstance(val, bytes):
                for cf_num, _, cf_val in _iter_proto_fields(val):
                    if cf_num == 1 and isinstance(cf_val, int):
                        cpu_state_num = _to_signed(cf_val, 32)
                    elif cf_num == 2 and isinstance(cf_val, int):
                        cpu_stack_ptr = cf_val
            elif fnum == 5 and wtype == 2 and isinstance(val, bytes):
                akey = ""
                ainfos: list[bytes] = []
                for mf_num, mf_wt, mf_val in _iter_proto_fields(val):
                    if mf_num == 1 and mf_wt == 2 and isinstance(mf_val, bytes):
                        akey = mf_val.decode("utf-8")
                    elif mf_num == 2 and mf_wt == 2 and isinstance(mf_val, bytes):
                        for af_num, af_wt, af_val in _iter_proto_fields(mf_val):
                            if af_num == 1 and af_wt == 2 and isinstance(af_val, bytes):
                                ainfos.append(af_val)
                analyzer_entries.append((akey, ainfos))
            elif fnum == 6 and wtype == 2 and isinstance(val, bytes):
                for pf_num, pf_wt, pf_val in _iter_proto_fields(val):
                    if pf_num == 1 and pf_wt == 2 and isinstance(pf_val, bytes):
                        pval = 0
                        for ps_num, _, ps_val in _iter_proto_fields(pf_val):
                            if ps_num == 2 and isinstance(ps_val, int):
                                pval = _to_signed(ps_val, 32)
                        pstate_values.append(pval)
    except ValueError as err:
        raise ValueError("Failed to read CoreDump from stream.") from err

    core_states: list[dict[str, object]] = []
    analyzer_reports: list[dict[str, object]] = []

    if cpu_state_num != 0 or cpu_stack_ptr != 0:
        state_name = _CPU_STATE_NAMES.get(cpu_state_num, "UNKNOWN_STATE")
        analyzer_reports.append(
            {
                "analyzer_name": "GeneralInfo",
                "category": "RUN_STATUS",
                "tag": "CpuState",
                "message": f"State: {state_name} | PC: 0x0 | SP: 0x{cpu_stack_ptr:x}",
            }
        )

    if anomalies_raw:
        for info_bytes in anomalies_raw:
            cat, desc, tag_s, print_s = _parse_info_message(info_bytes)
            if 1 <= cat <= 9:
                cat_name = _INFO_CATEGORY_NAMES.get(cat, "UNKNOWN_CATEGORY")
                analyzer_reports.append(
                    {
                        "analyzer_name": "Anomaly",
                        "category": cat_name,
                        "tag": tag_s or "<none>",
                        "message": f"[{cat_name} | {tag_s or '<none>'}] {print_s or desc}",
                    }
                )
    else:
        analyzer_reports.append(
            {
                "analyzer_name": "GeneralInfo",
                "category": "GENERAL",
                "tag": "Anomalies",
                "message": "No anomalies recorded in core dump.",
            }
        )

    for akey, ainfos in sorted(analyzer_entries, key=lambda item: item[0]):
        for info_bytes in ainfos:
            cat, desc, tag_s, print_s = _parse_info_message(info_bytes)
            m = _CORE_STATE_RE.match(print_s)
            if m is not None:
                core_states.append(
                    {
                        "core_type": m.group(1),
                        "core_id": int(m.group(2)),
                        "pc": int(m.group(3), 16),
                        "hlo_name": m.group(4),
                    }
                )
            elif 1 <= cat <= 9:
                cat_name = _INFO_CATEGORY_NAMES.get(cat, "UNKNOWN_CATEGORY")
                analyzer_reports.append(
                    {
                        "analyzer_name": akey,
                        "category": cat_name,
                        "tag": tag_s or "<none>",
                        "message": f"[{cat_name} | {tag_s or '<none>'}] {print_s or desc}",
                    }
                )

    return {
        "machine_info": {
            "machine_name": machine_name,
            "kernel_version": kernel_version,
            "reason_for_collection": reason_for_collection,
        },
        "timestamps": {
            "collection_start_sec": collection_start_sec,
            "collection_end_sec": collection_end_sec,
            "boot_time_sec": boot_time_sec,
            "device_start_sec": device_start_sec,
        },
        "hardware": {
            "pstate": pstate_values,
        },
        "core_states": core_states,
        "analyzer_reports": analyzer_reports,
    }


def emit_proto_schema(full_message_name: str) -> str:
    """Emits a canonical `proto3` `.proto` schema string from the compiled Rust protobuf descriptor.

    Args:
        full_message_name: Fully-qualified protobuf message name (e.g., `"xla.TileProto"`).

    Returns:
        Canonical `proto3` `.proto` schema string.

    Raises:
        ValueError: If `full_message_name` is empty or not registered in `libtpu-rs`.
    """
    import ctypes

    from libtpu import _load_cdll

    if not full_message_name:
        raise ValueError("full_message_name must not be empty")
    cdll = _load_cdll()
    fn = cdll.libtpu_rs_emit_proto_schema
    fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
    fn.restype = ctypes.c_size_t
    c_name = full_message_name.encode("utf-8")
    length = int(fn(c_name, None, 0))
    if length == 0:
        raise ValueError(f"Unknown protobuf message schema: {full_message_name}")
    buf = ctypes.create_string_buffer(length + 1)
    fn(c_name, buf, length + 1)
    return buf.value.decode("utf-8")


__all__: list[str] = [
    "emit_proto_schema",
    "read_compiler_metadata",
    "read_core_dump",
    "read_stall_summaries",
    "riegeli_highwayhash64",
]

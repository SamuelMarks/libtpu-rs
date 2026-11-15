"""Type stubs for `libtpu.sdk.diagnostics`."""

import os
from typing import TypedDict

class CompilerMetadataReport(TypedDict):
    """Sanitized compiler metadata dictionary returned by `read_compiler_metadata`."""

    program_id: int
    replica_count: int
    num_cores_per_replica: int
    is_haltless: bool
    executable_size: int
    optimized_hlo_instruction_count: int
    module_name: str
    executable_fingerprint: str

class _StallSummariesOptional(TypedDict, total=False):
    """Optional `artifact_metadata` field for `StallSummariesReport`."""

    artifact_metadata: dict[str, object]

class StallSummariesReport(_StallSummariesOptional):
    """Stall summary dictionary returned by `read_stall_summaries`."""

    core_states: dict[str, dict[str, object]]

class CoreDumpMachineInfo(TypedDict):
    """Machine metadata block in `CoreDumpReport`."""

    machine_name: str
    kernel_version: str
    reason_for_collection: str

class CoreDumpTimestamps(TypedDict):
    """Timestamp block in `CoreDumpReport`."""

    collection_start_sec: int
    collection_end_sec: int
    boot_time_sec: int
    device_start_sec: int

class CoreDumpHardware(TypedDict):
    """Hardware power-state block in `CoreDumpReport`."""

    pstate: list[int]

class CoreDumpCoreStateEntry(TypedDict):
    """Parsed scalar PC core state entry in `CoreDumpReport`."""

    core_type: str
    core_id: int
    pc: int
    hlo_name: str

class CoreDumpAnalyzerReport(TypedDict):
    """Analyzer report entry in `CoreDumpReport`."""

    analyzer_name: str
    category: str
    tag: str
    message: str

class CoreDumpReport(TypedDict):
    """Sanitized core dump dictionary returned by `read_core_dump`."""

    machine_info: CoreDumpMachineInfo
    timestamps: CoreDumpTimestamps
    hardware: CoreDumpHardware
    core_states: list[CoreDumpCoreStateEntry]
    analyzer_reports: list[CoreDumpAnalyzerReport]

def riegeli_highwayhash64(data: bytes) -> int:
    """Computes Google HighwayHash64 over `data` using the canonical Riegeli 256-bit key.

    Args:
        data: Input byte buffer to hash.
    """
    ...

def read_compiler_metadata(path: str | os.PathLike[str]) -> CompilerMetadataReport:
    """Read a `CompilerMetadata` binary proto and extract sanitized fields.

    Args:
        path: Path to the `tpu_compiler_metadata_*.pb` file.
    """
    ...

def read_stall_summaries(path: str | os.PathLike[str]) -> StallSummariesReport:
    """Read an `AllCoreStateSummaries` binary proto from disk.

    Args:
        path: Path to the `tpu_stall_summary_*.pb` file.
    """
    ...

def read_core_dump(path: str | os.PathLike[str]) -> CoreDumpReport:
    """Read and sanitize a TPU core dump from a Riegeli or protobuf file.

    Args:
        path: Path to the `tpu_core_dump_*.bin` file.
    """
    ...

def emit_proto_schema(full_message_name: str) -> str:
    """Emits a canonical `proto3` `.proto` schema string from the compiled Rust protobuf descriptor.

    Args:
        full_message_name: Fully-qualified protobuf message name (e.g., `"xla.TileProto"`).
    """
    ...

def protodesc_cold_file_count() -> int:
    """Queries the number of embedded `protodesc_cold` `.proto` descriptors (`293`)."""
    ...

def invoke_coordination_grpc(
    method_path: str,
    payload: bytes,
    compressed: bool = False,
) -> bytes:
    """Invokes a `tensorflow.CoordinationService` RPC method using 5-byte gRPC wire framing.

    Args:
        method_path: HTTP/2 gRPC method path (e.g., `"/tensorflow.CoordinationService/Barrier"`).
        payload: Wire-encoded protobuf request message bytes.
        compressed: Whether the 5-byte gRPC frame compressed flag is set (`1` vs `0`).
    """
    ...

def compute_torus_route_hops(
    src_x: int,
    src_y: int,
    src_z: int,
    dst_x: int,
    dst_y: int,
    dst_z: int,
    wrap_around: bool = True,
) -> int:
    """Computes the shortest 3D Torus optical ICI hop count between two chip coordinates.

    Args:
        src_x: Source X chip coordinate.
        src_y: Source Y chip coordinate.
        src_z: Source Z chip coordinate.
        dst_x: Destination X chip coordinate.
        dst_y: Destination Y chip coordinate.
        dst_z: Destination Z chip coordinate.
        wrap_around: Whether twisted-torus wrap-around links are enabled.
    """
    ...

__all__: list[str]

"""Type stubs for `libtpu.sdk.diagnostics`."""

import os

def riegeli_highwayhash64(data: bytes) -> int:
    """Computes Google HighwayHash64 over `data` using the canonical Riegeli 256-bit key.

    Args:
        data: Input byte buffer to hash.
    """
    ...

def read_compiler_metadata(path: str | os.PathLike[str]) -> dict[str, object]:
    """Read a `CompilerMetadata` binary proto and extract sanitized fields.

    Args:
        path: Path to the `tpu_compiler_metadata_*.pb` file.
    """
    ...

def read_stall_summaries(path: str | os.PathLike[str]) -> dict[str, object]:
    """Read an `AllCoreStateSummaries` binary proto from disk.

    Args:
        path: Path to the `tpu_stall_summary_*.pb` file.
    """
    ...

def read_core_dump(path: str | os.PathLike[str]) -> dict[str, object]:
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

__all__: list[str]

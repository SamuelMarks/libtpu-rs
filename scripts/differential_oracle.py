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
import struct
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


class PjrtErrorDestroyArgs(ctypes.Structure):
    """`#[repr(C)]` wrapper for `PJRT_Error_Destroy_Args` (`24` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("error", ctypes.c_void_p),
    ]


class PjrtErrorMessageArgs(ctypes.Structure):
    """`#[repr(C)]` wrapper for `PJRT_Error_Message_Args` (`40` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("error", ctypes.c_void_p),
        ("message", ctypes.c_void_p),
        ("message_size", ctypes.c_size_t),
    ]


class PjrtPluginInitializeArgs(ctypes.Structure):
    """`#[repr(C)]` wrapper for `PJRT_Plugin_Initialize_Args` (`16` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
    ]


class PjrtEventDestroyArgs(ctypes.Structure):
    """`#[repr(C)]` wrapper for `PJRT_Event_Destroy_Args` (`24` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("event", ctypes.c_void_p),
    ]


class PjrtEventAwaitArgs(ctypes.Structure):
    """`#[repr(C)]` wrapper for `PJRT_Event_Await_Args` (`24` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("event", ctypes.c_void_p),
    ]


class PjrtTopologyDescriptionCreateArgs(ctypes.Structure):
    """`#[repr(C)]` wrapper for `PJRT_TopologyDescription_Create_Args` (`56` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("topology_name", ctypes.c_char_p),
        ("topology_name_size", ctypes.c_size_t),
        ("create_options", ctypes.c_void_p),
        ("num_options", ctypes.c_size_t),
        ("topology", ctypes.c_void_p),
    ]


class PjrtTopologyDescriptionDestroyArgs(ctypes.Structure):
    """`#[repr(C)]` wrapper for `PJRT_TopologyDescription_Destroy_Args` (`24` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("topology", ctypes.c_void_p),
    ]


class PjrtProgram(ctypes.Structure):
    """`#[repr(C)]` wrapper for `PJRT_Program` (`40` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("code", ctypes.c_char_p),
        ("code_size", ctypes.c_size_t),
        ("format", ctypes.c_char_p),
        ("format_size", ctypes.c_size_t),
    ]


class PjrtCompileArgs(ctypes.Structure):
    """`#[repr(C)]` wrapper for `PJRT_Compile_Args` (`64` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("topology", ctypes.c_void_p),
        ("program", ctypes.POINTER(PjrtProgram)),
        ("compile_options", ctypes.c_char_p),
        ("compile_options_size", ctypes.c_size_t),
        ("client", ctypes.c_void_p),
        ("executable", ctypes.c_void_p),
    ]


class PjrtClientCreateArgs(ctypes.Structure):
    """`#[repr(C)]` wrapper for `PJRT_Client_Create_Args` (`88` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("create_options", ctypes.c_void_p),
        ("num_options", ctypes.c_size_t),
        ("kv_get_callback", ctypes.c_void_p),
        ("kv_get_user_arg", ctypes.c_void_p),
        ("kv_put_callback", ctypes.c_void_p),
        ("kv_put_user_arg", ctypes.c_void_p),
        ("client", ctypes.c_void_p),
        ("kv_try_get_callback", ctypes.c_void_p),
        ("kv_try_get_user_arg", ctypes.c_void_p),
    ]


class PjrtClientDestroyArgs(ctypes.Structure):
    """`#[repr(C)]` wrapper for `PJRT_Client_Destroy_Args` (`24` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("client", ctypes.c_void_p),
    ]


class PjrtClientAddressableDevicesArgs(ctypes.Structure):
    """`#[repr(C)]` wrapper for `PJRT_Client_AddressableDevices_Args` (`40` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("client", ctypes.c_void_p),
        ("addressable_devices", ctypes.POINTER(ctypes.c_void_p)),
        ("num_addressable_devices", ctypes.c_size_t),
    ]


class PjrtClientCompileArgs(ctypes.Structure):
    """`#[repr(C)]` wrapper for `PJRT_Client_Compile_Args` (`56` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("client", ctypes.c_void_p),
        ("program", ctypes.POINTER(PjrtProgram)),
        ("compile_options", ctypes.c_char_p),
        ("compile_options_size", ctypes.c_size_t),
        ("executable", ctypes.c_void_p),
    ]


class PjrtClientBufferFromHostBufferArgs(ctypes.Structure):
    """`#[repr(C)]` wrapper for `PJRT_Client_BufferFromHostBuffer_Args` (`120` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("client", ctypes.c_void_p),
        ("data", ctypes.c_void_p),
        ("element_type", ctypes.c_int32),
        ("pad0", ctypes.c_int32),
        ("dims", ctypes.POINTER(ctypes.c_int64)),
        ("num_dims", ctypes.c_size_t),
        ("byte_strides", ctypes.POINTER(ctypes.c_int64)),
        ("num_byte_strides", ctypes.c_size_t),
        ("host_buffer_semantics", ctypes.c_int32),
        ("pad1", ctypes.c_int32),
        ("device", ctypes.c_void_p),
        ("memory", ctypes.c_void_p),
        ("device_layout", ctypes.c_void_p),
        ("done_with_host_buffer", ctypes.c_void_p),
        ("out_buffer", ctypes.c_void_p),
    ]


class PjrtExecutableSerializeArgs(ctypes.Structure):
    """`#[repr(C)]` wrapper for `PJRT_Executable_Serialize_Args` (`56` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("executable", ctypes.c_void_p),
        ("serialized_bytes", ctypes.c_void_p),
        ("serialized_bytes_size", ctypes.c_size_t),
        ("serialized_executable", ctypes.c_void_p),
        ("serialized_executable_deleter", ctypes.c_void_p),
    ]


class PjrtExecutableDestroyArgs(ctypes.Structure):
    """`#[repr(C)]` wrapper for `PJRT_Executable_Destroy_Args` (`24` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("executable", ctypes.c_void_p),
    ]


class PjrtLoadedExecutableDestroyArgs(ctypes.Structure):
    """`#[repr(C)]` wrapper for `PJRT_LoadedExecutable_Destroy_Args` (`24` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("executable", ctypes.c_void_p),
    ]


class PjrtLoadedExecutableGetExecutableArgs(ctypes.Structure):
    """`#[repr(C)]` wrapper for `PJRT_LoadedExecutable_GetExecutable_Args` (`32` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("loaded_executable", ctypes.c_void_p),
        ("executable", ctypes.c_void_p),
    ]


class PjrtExecuteOptions(ctypes.Structure):
    """`#[repr(C)]` wrapper for `PJRT_ExecuteOptions` (`120` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("send_callbacks", ctypes.c_void_p),
        ("recv_callbacks", ctypes.c_void_p),
        ("num_send_ops", ctypes.c_size_t),
        ("num_recv_ops", ctypes.c_size_t),
        ("launch_id", ctypes.c_int32),
        ("pad0", ctypes.c_int32),
        ("non_donatable_input_indices", ctypes.c_void_p),
        ("num_non_donatable_input_indices", ctypes.c_size_t),
        ("context", ctypes.c_void_p),
        ("call_location", ctypes.c_char_p),
        ("num_tasks", ctypes.c_size_t),
        ("task_ids", ctypes.c_void_p),
        ("incarnation_ids", ctypes.c_void_p),
    ]


class PjrtLoadedExecutableExecuteArgs(ctypes.Structure):
    """`#[repr(C)]` wrapper for `PJRT_LoadedExecutable_Execute_Args` (`80` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("executable", ctypes.c_void_p),
        ("options", ctypes.POINTER(PjrtExecuteOptions)),
        ("argument_lists", ctypes.c_void_p),
        ("num_devices", ctypes.c_size_t),
        ("num_args", ctypes.c_size_t),
        ("output_lists", ctypes.c_void_p),
        ("device_complete_events", ctypes.c_void_p),
        ("execute_device", ctypes.c_void_p),
    ]


class PjrtBufferToHostBufferArgs(ctypes.Structure):
    """`#[repr(C)]` wrapper for `PJRT_Buffer_ToHostBuffer_Args` (`56` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("src", ctypes.c_void_p),
        ("host_layout", ctypes.c_void_p),
        ("dst", ctypes.c_void_p),
        ("dst_size", ctypes.c_size_t),
        ("event", ctypes.c_void_p),
    ]


class PjrtBufferDestroyArgs(ctypes.Structure):
    """`#[repr(C)]` wrapper for `PJRT_Buffer_Destroy_Args` (`24` bytes)."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("buffer", ctypes.c_void_p),
    ]


PJRT_Error_Destroy_Args = PjrtErrorDestroyArgs
PJRT_Error_Message_Args = PjrtErrorMessageArgs
PJRT_Plugin_Initialize_Args = PjrtPluginInitializeArgs
PJRT_Event_Destroy_Args = PjrtEventDestroyArgs
PJRT_Event_Await_Args = PjrtEventAwaitArgs
PJRT_TopologyDescription_Create_Args = PjrtTopologyDescriptionCreateArgs
PJRT_TopologyDescription_Destroy_Args = PjrtTopologyDescriptionDestroyArgs
PJRT_Program = PjrtProgram
PJRT_Compile_Args = PjrtCompileArgs
PJRT_Client_Create_Args = PjrtClientCreateArgs
PJRT_Client_Destroy_Args = PjrtClientDestroyArgs
PJRT_Client_AddressableDevices_Args = PjrtClientAddressableDevicesArgs
PJRT_Client_Compile_Args = PjrtClientCompileArgs
PJRT_Client_BufferFromHostBuffer_Args = PjrtClientBufferFromHostBufferArgs
PJRT_Executable_Serialize_Args = PjrtExecutableSerializeArgs
PJRT_Executable_Destroy_Args = PjrtExecutableDestroyArgs
PJRT_LoadedExecutable_Destroy_Args = PjrtLoadedExecutableDestroyArgs
PJRT_LoadedExecutable_GetExecutable_Args = PjrtLoadedExecutableGetExecutableArgs
PJRT_ExecuteOptions = PjrtExecuteOptions
PJRT_LoadedExecutable_Execute_Args = PjrtLoadedExecutableExecuteArgs
PJRT_Buffer_ToHostBuffer_Args = PjrtBufferToHostBufferArgs
PJRT_Buffer_Destroy_Args = PjrtBufferDestroyArgs


_PJRT_FN_PROTO = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)
_PJRT_VOID_FN_PROTO = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


class LibtpuSharedLibraryBackend:
    """`ctypes` PJRT C ABI backend invoking `GetPjrtApi` vtable entrypoints directly."""

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
        self._plugin_initialized = False

    @property
    def api_pointer(self) -> int:
        """Returns the non-null `PJRT_Api*` base virtual address.

        Returns:
            Unsigned integer pointer to the static `PJRT_Api` vtable.
        """
        return self._api_ptr

    def _slot_ptr(self, byte_offset: int) -> int:
        """Reads a 64-bit function pointer at `byte_offset` inside `PJRT_Api`.

        Args:
            byte_offset: Byte offset of the slot within `PJRT_Api`.

        Returns:
            Resolved non-zero function pointer address.

        Raises:
            DifferentialOracleError: If the requested vtable slot is NULL or out of range.
        """
        struct_sz = ctypes.c_size_t.from_address(self._api_ptr).value
        if byte_offset + 8 > struct_sz:
            raise DifferentialOracleError(
                f"PJRT_Api struct_size={struct_sz} too small for slot offset 0x{byte_offset:x}"
            )
        fn_addr = int(ctypes.c_void_p.from_address(self._api_ptr + byte_offset).value or 0)
        if fn_addr == 0:
            raise DifferentialOracleError(f"PJRT_Api slot at 0x{byte_offset:x} is NULL")
        return fn_addr

    def _call_pjrt(self, byte_offset: int, args_struct: ctypes.Structure) -> None:
        """Invokes a `PJRT_Error* (*)(Args*)` vtable slot and raises on error.

        Args:
            byte_offset: Byte offset of the target `PJRT_Api` slot.
            args_struct: Populated `ctypes.Structure` argument block.

        Raises:
            DifferentialOracleError: If the PJRT entrypoint returns a non-null `PJRT_Error*`.
        """
        fn = _PJRT_FN_PROTO(self._slot_ptr(byte_offset))
        err_ptr = int(fn(ctypes.byref(args_struct)) or 0)
        if err_ptr != 0:
            msg_text = f"PJRT call at 0x{byte_offset:x} failed"
            msg_fn = _PJRT_VOID_FN_PROTO(self._slot_ptr(0x030))
            msg_args = PJRT_Error_Message_Args(
                struct_size=ctypes.sizeof(PJRT_Error_Message_Args),
                extension_start=None,
                error=err_ptr,
                message=None,
                message_size=0,
            )
            msg_fn(ctypes.byref(msg_args))
            if msg_args.message:
                msg_text = ctypes.string_at(msg_args.message, msg_args.message_size).decode(
                    "utf-8", errors="replace"
                )
            del_fn = _PJRT_VOID_FN_PROTO(self._slot_ptr(0x028))
            del_args = PJRT_Error_Destroy_Args(
                struct_size=ctypes.sizeof(PJRT_Error_Destroy_Args),
                extension_start=None,
                error=err_ptr,
            )
            del_fn(ctypes.byref(del_args))
            raise DifferentialOracleError(msg_text)

    def _ensure_initialized(self) -> None:
        """Calls `PJRT_Plugin_Initialize` (`0x040`) once on this backend instance."""
        if not self._plugin_initialized:
            init_args = PJRT_Plugin_Initialize_Args(
                struct_size=ctypes.sizeof(PJRT_Plugin_Initialize_Args),
                extension_start=None,
            )
            self._call_pjrt(0x040, init_args)
            self._plugin_initialized = True

    def _destroy_event_if_set(self, event_ptr: int) -> None:
        """Awaits and destroys a non-null `PJRT_Event*` handle.

        Args:
            event_ptr: Raw `PJRT_Event*` address (no-op if `0`).
        """
        if event_ptr != 0:
            try:
                await_args = PJRT_Event_Await_Args(
                    struct_size=ctypes.sizeof(PJRT_Event_Await_Args),
                    extension_start=None,
                    event=event_ptr,
                )
                self._call_pjrt(0x068, await_args)
            finally:
                ev_del = PJRT_Event_Destroy_Args(
                    struct_size=ctypes.sizeof(PJRT_Event_Destroy_Args),
                    extension_start=None,
                    event=event_ptr,
                )
                self._call_pjrt(0x050, ev_del)

    def compile_to_serialized_executable(self, case: OracleProgramCase) -> bytes:
        """Compiles `case` AOT through `PJRT_TopologyDescription_Create` + `PJRT_Compile`.

        Args:
            case: Target `OracleProgramCase` descriptor.

        Returns:
            Serialized `TpuExecutable` bytes (`LTPUEXEC` / protobuf wire format) produced by
            `PJRT_Executable_Serialize`.

        Raises:
            DifferentialOracleError: If topology creation, compilation, or serialization fails.
        """
        self._ensure_initialized()
        topo_bytes = f"{case.generation}:{case.topology_spec}".encode()
        topo_create = PJRT_TopologyDescription_Create_Args(
            struct_size=ctypes.sizeof(PJRT_TopologyDescription_Create_Args),
            extension_start=None,
            topology_name=topo_bytes,
            topology_name_size=len(topo_bytes),
            create_options=None,
            num_options=0,
            topology=None,
        )
        self._call_pjrt(0x2B8, topo_create)
        topo_handle = int(topo_create.topology or 0)
        exec_handle = 0
        try:
            code_bytes = case.mlir_text.encode("utf-8")
            fmt_bytes = b"mlir"
            prog = PJRT_Program(
                struct_size=ctypes.sizeof(PJRT_Program),
                extension_start=None,
                code=code_bytes,
                code_size=len(code_bytes),
                format=fmt_bytes,
                format_size=len(fmt_bytes),
            )
            comp_args = PJRT_Compile_Args(
                struct_size=ctypes.sizeof(PJRT_Compile_Args),
                extension_start=None,
                topology=topo_handle,
                program=ctypes.pointer(prog),
                compile_options=b"",
                compile_options_size=0,
                client=None,
                executable=None,
            )
            self._call_pjrt(0x2F0, comp_args)
            exec_handle = int(comp_args.executable or 0)

            ser_args = PJRT_Executable_Serialize_Args(
                struct_size=ctypes.sizeof(PJRT_Executable_Serialize_Args),
                extension_start=None,
                executable=exec_handle,
                serialized_bytes=None,
                serialized_bytes_size=0,
                serialized_executable=None,
                serialized_executable_deleter=None,
            )
            self._call_pjrt(0x1B0, ser_args)
            try:
                raw_bytes = ctypes.string_at(
                    int(ser_args.serialized_bytes or 0),
                    int(ser_args.serialized_bytes_size),
                )
            finally:
                deleter_addr = int(ser_args.serialized_executable_deleter or 0)
                ser_handle = int(ser_args.serialized_executable or 0)
                _PJRT_VOID_FN_PROTO(deleter_addr)(ctypes.c_void_p(ser_handle))
            return raw_bytes
        finally:
            if exec_handle != 0:
                ex_del = PJRT_Executable_Destroy_Args(
                    struct_size=ctypes.sizeof(PJRT_Executable_Destroy_Args),
                    extension_start=None,
                    executable=exec_handle,
                )
                self._call_pjrt(0x168, ex_del)
            if topo_handle != 0:
                tp_del = PJRT_TopologyDescription_Destroy_Args(
                    struct_size=ctypes.sizeof(PJRT_TopologyDescription_Destroy_Args),
                    extension_start=None,
                    topology=topo_handle,
                )
                self._call_pjrt(0x2C0, tp_del)

    def compile_and_execute_smoke(self) -> list[float]:
        """Executes an end-to-end H2D -> `stablehlo.add` -> D2H smoke test via `GetPjrtApi()`.

        Returns:
            4-element `float` list (`[11.0, 22.0, 33.0, 44.0]`) read back from device HBM.

        Raises:
            DifferentialOracleError: If client creation, H2D, compilation, execution, or D2H fails.
        """
        self._ensure_initialized()
        create_args = PJRT_Client_Create_Args(
            struct_size=ctypes.sizeof(PJRT_Client_Create_Args),
            extension_start=None,
            create_options=None,
            num_options=0,
            kv_get_callback=None,
            kv_get_user_arg=None,
            kv_put_callback=None,
            kv_put_user_arg=None,
            client=None,
            kv_try_get_callback=None,
            kv_try_get_user_arg=None,
        )
        self._call_pjrt(0x078, create_args)
        client_handle = int(create_args.client or 0)
        loaded_exec = 0
        buf_a = 0
        buf_b = 0
        out_buf = 0
        try:
            dev_args = PJRT_Client_AddressableDevices_Args(
                struct_size=ctypes.sizeof(PJRT_Client_AddressableDevices_Args),
                extension_start=None,
                client=client_handle,
                addressable_devices=None,
                num_addressable_devices=0,
            )
            self._call_pjrt(0x0A8, dev_args)
            if int(dev_args.num_addressable_devices) < 1 or not bool(dev_args.addressable_devices):
                raise DifferentialOracleError("PJRT client reported 0 addressable devices")
            dev0 = int(dev_args.addressable_devices[0] or 0)

            mlir = (
                b"module @smoke_add {\n"
                b"  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>)"
                b" -> tensor<4xf32> {\n"
                b"    %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>\n"
                b"    return %0 : tensor<4xf32>\n"
                b"  }\n"
                b"}\n"
            )
            prog = PJRT_Program(
                struct_size=ctypes.sizeof(PJRT_Program),
                extension_start=None,
                code=mlir,
                code_size=len(mlir),
                format=b"mlir",
                format_size=4,
            )
            ccomp = PJRT_Client_Compile_Args(
                struct_size=ctypes.sizeof(PJRT_Client_Compile_Args),
                extension_start=None,
                client=client_handle,
                program=ctypes.pointer(prog),
                compile_options=b"",
                compile_options_size=0,
                executable=None,
            )
            self._call_pjrt(0x0C8, ccomp)
            loaded_exec = int(ccomp.executable or 0)

            get_ex = PJRT_LoadedExecutable_GetExecutable_Args(
                struct_size=ctypes.sizeof(PJRT_LoadedExecutable_GetExecutable_Args),
                extension_start=None,
                loaded_executable=loaded_exec,
                executable=None,
            )
            self._call_pjrt(0x1C0, get_ex)
            sub_exec = int(get_ex.executable or 0)
            ex_del = PJRT_Executable_Destroy_Args(
                struct_size=ctypes.sizeof(PJRT_Executable_Destroy_Args),
                extension_start=None,
                executable=sub_exec,
            )
            self._call_pjrt(0x168, ex_del)

            dims = (ctypes.c_int64 * 1)(4)
            lhs_arr = (ctypes.c_float * 4)(1.0, 2.0, 3.0, 4.0)
            rhs_arr = (ctypes.c_float * 4)(10.0, 20.0, 30.0, 40.0)

            h2d_a = PJRT_Client_BufferFromHostBuffer_Args(
                struct_size=ctypes.sizeof(PJRT_Client_BufferFromHostBuffer_Args),
                extension_start=None,
                client=client_handle,
                data=ctypes.cast(lhs_arr, ctypes.c_void_p),
                element_type=11,
                pad0=0,
                dims=dims,
                num_dims=1,
                byte_strides=None,
                num_byte_strides=0,
                host_buffer_semantics=0,
                pad1=0,
                device=dev0,
                memory=None,
                device_layout=None,
                done_with_host_buffer=None,
                out_buffer=None,
            )
            self._call_pjrt(0x0D8, h2d_a)
            self._destroy_event_if_set(int(h2d_a.done_with_host_buffer or 0))
            buf_a = int(h2d_a.out_buffer or 0)

            h2d_b = PJRT_Client_BufferFromHostBuffer_Args(
                struct_size=ctypes.sizeof(PJRT_Client_BufferFromHostBuffer_Args),
                extension_start=None,
                client=client_handle,
                data=ctypes.cast(rhs_arr, ctypes.c_void_p),
                element_type=11,
                pad0=0,
                dims=dims,
                num_dims=1,
                byte_strides=None,
                num_byte_strides=0,
                host_buffer_semantics=0,
                pad1=0,
                device=dev0,
                memory=None,
                device_layout=None,
                done_with_host_buffer=None,
                out_buffer=None,
            )
            self._call_pjrt(0x0D8, h2d_b)
            self._destroy_event_if_set(int(h2d_b.done_with_host_buffer or 0))
            buf_b = int(h2d_b.out_buffer or 0)

            arg_row = (ctypes.c_void_p * 2)(buf_a, buf_b)
            arg_lists = (ctypes.c_void_p * 1)(ctypes.addressof(arg_row))
            out_row = (ctypes.c_void_p * 1)(0)
            out_lists = (ctypes.c_void_p * 1)(ctypes.addressof(out_row))
            complete_events = (ctypes.c_void_p * 1)(0)
            exec_opts = PJRT_ExecuteOptions(
                struct_size=ctypes.sizeof(PJRT_ExecuteOptions),
                extension_start=None,
                send_callbacks=None,
                recv_callbacks=None,
                num_send_ops=0,
                num_recv_ops=0,
                launch_id=1,
                pad0=0,
                non_donatable_input_indices=None,
                num_non_donatable_input_indices=0,
                context=None,
                call_location=None,
                num_tasks=0,
                task_ids=None,
                incarnation_ids=None,
            )
            exec_args = PJRT_LoadedExecutable_Execute_Args(
                struct_size=ctypes.sizeof(PJRT_LoadedExecutable_Execute_Args),
                extension_start=None,
                executable=loaded_exec,
                options=ctypes.pointer(exec_opts),
                argument_lists=ctypes.cast(arg_lists, ctypes.c_void_p),
                num_devices=1,
                num_args=2,
                output_lists=ctypes.cast(out_lists, ctypes.c_void_p),
                device_complete_events=ctypes.cast(complete_events, ctypes.c_void_p),
                execute_device=None,
            )
            self._call_pjrt(0x1E0, exec_args)
            self._destroy_event_if_set(int(complete_events[0] or 0))
            out_buf = int(out_row[0] or 0)

            dst_arr = (ctypes.c_float * 4)(0.0, 0.0, 0.0, 0.0)
            d2h = PJRT_Buffer_ToHostBuffer_Args(
                struct_size=ctypes.sizeof(PJRT_Buffer_ToHostBuffer_Args),
                extension_start=None,
                src=out_buf,
                host_layout=None,
                dst=ctypes.cast(dst_arr, ctypes.c_void_p),
                dst_size=16,
                event=None,
            )
            self._call_pjrt(0x258, d2h)
            self._destroy_event_if_set(int(d2h.event or 0))
            return [float(dst_arr[i]) for i in range(4)]
        finally:
            for b_ptr in (out_buf, buf_b, buf_a):
                if b_ptr != 0:
                    b_del = PJRT_Buffer_Destroy_Args(
                        struct_size=ctypes.sizeof(PJRT_Buffer_Destroy_Args),
                        extension_start=None,
                        buffer=b_ptr,
                    )
                    self._call_pjrt(0x1F8, b_del)
            if loaded_exec != 0:
                le_del = PJRT_LoadedExecutable_Destroy_Args(
                    struct_size=ctypes.sizeof(PJRT_LoadedExecutable_Destroy_Args),
                    extension_start=None,
                    executable=loaded_exec,
                )
                self._call_pjrt(0x1B8, le_del)
            if client_handle != 0:
                cl_del = PJRT_Client_Destroy_Args(
                    struct_size=ctypes.sizeof(PJRT_Client_Destroy_Args),
                    extension_start=None,
                    client=client_handle,
                )
                self._call_pjrt(0x080, cl_del)


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
    if payload.startswith(b"LTPUEXEC"):
        if len(payload) < 25:
            raise DifferentialOracleError("Truncated LTPUEXEC binary header")
        version_le = struct.unpack_from("<I", payload, 8)[0]
        if version_le != 1:
            raise DifferentialOracleError(f"Unsupported LTPUEXEC format version: {version_le}")
        fmt_len = struct.unpack_from("<I", payload, 13)[0]
        pos = 17 + fmt_len
        if pos + 4 > len(payload):
            raise DifferentialOracleError("Truncated LTPUEXEC format section")
        src_len = struct.unpack_from("<I", payload, pos)[0]
        pos += 4 + src_len
        if pos + 4 > len(payload):
            raise DifferentialOracleError("Truncated LTPUEXEC source section")
        opts_len = struct.unpack_from("<I", payload, pos)[0]
        pos += 4 + opts_len
        if pos != len(payload):
            raise DifferentialOracleError("Mismatched LTPUEXEC payload length")
        return SerializedExecutableSummary(
            case_id=case.case_id,
            generation=case.generation,
            byte_length=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
            wire_field_numbers=(1, 2, 3, 4),
        )

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

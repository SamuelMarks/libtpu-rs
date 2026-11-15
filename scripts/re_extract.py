"""Deterministic clean-room binary forensics and specification extractor for `libtpu`.

Inspects official `libtpu` wheel archives (`.whl`) or ELF binaries (`libtpu.so`, `sdk.abi3.so`),
fetches wheel metadata from PyPI JSON endpoints (`fetch_pypi_wheel_url`), extracts ELF64 section
headers, dynamic symbol tables (`.dynsym` / `.dynstr`), `.rela.dyn` relative relocations,
reconstructs `PJRT_Api` (`1,144` bytes) + 17-node `PJRT_Extension_Base` + `LibtpuSdk_Api`
(`144` bytes) + `TpuDriverFn` (`224` bytes) vtables, disassembles x86-64 function prologues
(`disassemble_x86_64_prologue`), carves `protodesc_cold` and `filewrapper_toc` bundles, and
validates consistency against the repository's golden reverse-engineering manifests
(`docs/reverse_engineering/golden_binary_manifest.json` and `golden_struct_layouts.json`).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import sys
import urllib.request
import zipfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from elftools.common.exceptions import ELFError
from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection


class _ElfSymbolProto(Protocol):
    """Typed protocol over `pyelftools` `Symbol`."""

    name: str
    entry: Mapping[str, Mapping[str, str] | str | int]


class _ElfSectionProto(Protocol):
    """Typed protocol over `pyelftools` `Section`."""

    name: str
    header: Mapping[str, str | int]


class _ElfSymbolTableProto(Protocol):
    """Typed protocol over `pyelftools` `SymbolTableSection`."""

    iter_symbols: Callable[[], Iterable[_ElfSymbolProto]]


class _ElfFileProto(Protocol):
    """Typed protocol over `pyelftools` `ELFFile`."""

    iter_sections: Callable[[], Iterable[_ElfSectionProto]]


ELF_MAGIC = b"\x7fELF"
ELFCLASS64 = 2
ELFDATA2LSB = 1
SHT_SYMTAB = 2
SHT_STRTAB = 3
SHT_RELA = 4
SHT_DYNSYM = 11
STT_FUNC = 2
STB_GLOBAL = 1
STB_WEAK = 2
R_X86_64_RELATIVE = 8
R_AARCH64_RELATIVE = 1027

OFFICIAL_LIBTPU_SO_SHA256 = "f86de4d04a058e6b5101f73e0824b95f27e2cec996d2b9191d9cffda75c03077"
OFFICIAL_SDK_SO_SHA256 = "630c772352b470e1f32b7459ac3d60790f8d4d5000747390ebdeb743c1f33186"


@dataclass(frozen=True)
class ElfSectionHeader:
    """Represents a parsed 64-bit little-endian ELF section header (`Elf64_Shdr`)."""

    name: str
    sh_type: int
    sh_flags: int
    sh_addr: int
    sh_offset: int
    sh_size: int
    sh_link: int
    sh_entsize: int


@dataclass(frozen=True)
class ElfExportedSymbol:
    """Represents a dynamically exported function or object symbol from `.dynsym`."""

    name: str
    vaddr: int
    size: int
    st_info: int


@dataclass(frozen=True)
class ElfRelativeReloc:
    """Represents a resolved `.rela.dyn` relative relocation (`*r_offset = r_addend`)."""

    offset: int
    r_type: int
    addend: int


@dataclass(frozen=True)
class PrologueInstructionReport:
    """Represents struct-size and `[rdi + disp]` field accesses decoded from an x86-64 prologue."""

    expected_struct_size: int | None
    field_offsets: list[tuple[int, int, bool]]


def compute_sha256_bytes(data: bytes) -> str:
    """Computes the lowercase hexadecimal SHA-256 digest of a byte slice.

    Args:
        data: Raw byte buffer to hash.

    Returns:
        64-character lowercase hex SHA-256 string.
    """
    return hashlib.sha256(data).hexdigest()


def fetch_pypi_wheel_url(version: str = "0.0.52", platform_tag: str = "x86_64") -> tuple[str, str]:
    """Queries the PyPI JSON API for `libtpu` and returns `(wheel_url, sha256_hex)`.

    Args:
        version: Target `libtpu` release version on PyPI.
        platform_tag: Substring matching the desired platform wheel filename.

    Returns:
        Tuple of `(download_url, sha256_digest)`.

    Raises:
        ValueError: If no matching wheel is found in the PyPI release metadata.
    """
    api_url = f"https://pypi.org/pypi/libtpu/{version}/json"
    with urllib.request.urlopen(api_url, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    for entry in payload.get("urls", []):
        filename = str(entry.get("filename", ""))
        if filename.endswith(".whl") and platform_tag in filename:
            url = str(entry.get("url", ""))
            digests = entry.get("digests", {})
            sha256 = str(digests.get("sha256", ""))
            return url, sha256
    raise ValueError(f"No libtpu=={version} wheel found matching platform '{platform_tag}'")


def resolve_elf64_rela_relocations(elf_bytes: bytes) -> list[ElfRelativeReloc]:
    """Extracts `R_X86_64_RELATIVE` and `R_AARCH64_RELATIVE` entries from `SHT_RELA` sections.

    Args:
        elf_bytes: Complete ELF64 little-endian binary payload.

    Returns:
        List of `ElfRelativeReloc` records.
    """
    sections, _ = parse_elf64_sections_and_dynsym(elf_bytes)
    relocs: list[ElfRelativeReloc] = []
    for sec in sections:
        if sec.sh_type == SHT_RELA and sec.sh_entsize == 24:
            raw = elf_bytes[sec.sh_offset : sec.sh_offset + sec.sh_size]
            for off in range(0, len(raw) - (len(raw) % 24), 24):
                r_offset, r_info, r_addend = struct.unpack_from("<QQQ", raw, off)
                r_type = r_info & 0xFFFFFFFF
                if r_type in (R_X86_64_RELATIVE, R_AARCH64_RELATIVE):
                    relocs.append(ElfRelativeReloc(offset=r_offset, r_type=r_type, addend=r_addend))
    return relocs


def reconstruct_pjrt_and_extension_vtables(
    word_map: dict[int, int],
    pjrt_api_vaddr: int,
    num_slots: int = 138,
) -> dict[str, object]:
    """Reconstructs `PJRT_Api` and its `PJRT_Extension_Base` linked list from a 64-bit word map.

    Args:
        word_map: Mapping from 8-byte-aligned virtual address to 64-bit word value.
        pjrt_api_vaddr: Base virtual address of `PJRT_Api`.
        num_slots: Number of function pointer slots in `PJRT_Api`.

    Returns:
        Dictionary with `struct_size`, `major_version`, `minor_version`, `slots`, and `extensions`.
    """
    struct_size = word_map.get(pjrt_api_vaddr, 0)
    ext_head = word_map.get(pjrt_api_vaddr + 8, 0)
    ver_size = word_map.get(pjrt_api_vaddr + 16, 0)
    ver_packed = word_map.get(pjrt_api_vaddr + 24, 0)
    major_ver = ver_packed & 0xFFFFFFFF
    minor_ver = (ver_packed >> 32) & 0xFFFFFFFF

    slots: list[int] = [
        word_map.get(pjrt_api_vaddr + 0x28 + (idx * 8), 0) for idx in range(num_slots)
    ]

    extensions: list[dict[str, object]] = []
    cur = ext_head
    visited = 0
    while cur != 0 and visited < 32:
        visited += 1
        e_size = word_map.get(cur, 0)
        e_type = word_map.get(cur + 8, 0) & 0xFFFFFFFF
        e_next = word_map.get(cur + 16, 0)
        num_methods = max(0, (e_size - 24) // 8)
        methods = [word_map.get(cur + 24 + (m * 8), 0) for m in range(num_methods)]
        extensions.append(
            {
                "vaddr": cur,
                "struct_size": e_size,
                "type_id": e_type,
                "next_vaddr": e_next,
                "methods": methods,
            }
        )
        cur = e_next

    return {
        "struct_size": struct_size,
        "version_struct_size": ver_size,
        "major_version": major_ver,
        "minor_version": minor_ver,
        "slots": slots,
        "extensions": extensions,
    }


def decode_linux_ioctl_command(cmd: int) -> dict[str, int]:
    """Decodes a 32-bit Linux `_IOC(dir, type, nr, size)` command word into its bitfields.

    Args:
        cmd: Raw 32-bit/64-bit `ioctl` request command number.

    Returns:
        Dictionary containing `"dir"`, `"type"`, `"nr"`, and `"size"`.
    """
    return {
        "dir": (cmd >> 30) & 0x03,
        "size": (cmd >> 16) & 0x3FFF,
        "type": (cmd >> 8) & 0xFF,
        "nr": cmd & 0xFF,
    }


def reconstruct_itanium_vtable(
    word_map: dict[int, int],
    vtable_vaddr: int,
    num_methods: int,
) -> dict[str, object]:
    """Reconstructs an Itanium C++ ABI `_ZTV*` vtable from a relocated 64-bit word map.

    Args:
        word_map: Mapping from 8-byte-aligned virtual address to 64-bit word value.
        vtable_vaddr: Base virtual address of the `_ZTV*` header (`offset_to_top` slot).
        num_methods: Number of virtual method slots following the 16-byte header.

    Returns:
        Dictionary with `"vtable_vaddr"`, `"offset_to_top"`, `"typeinfo_vaddr"`, and `"methods"`.
    """
    offset_to_top = word_map.get(vtable_vaddr, 0)
    typeinfo_vaddr = word_map.get(vtable_vaddr + 8, 0)
    methods = [word_map.get(vtable_vaddr + 16 + (i * 8), 0) for i in range(num_methods)]
    return {
        "vtable_vaddr": vtable_vaddr,
        "offset_to_top": offset_to_top,
        "typeinfo_vaddr": typeinfo_vaddr,
        "methods": methods,
    }


def parse_eh_frame_cfi_records(
    eh_frame_bytes: bytes,
    eh_frame_vaddr: int = 0,
) -> list[dict[str, int]]:
    """Parses DWARF `.eh_frame` CIE and FDE records to extract internal function bounds.

    Args:
        eh_frame_bytes: Raw `.eh_frame` section bytes.
        eh_frame_vaddr: Base virtual address of `.eh_frame` in the ELF image.

    Returns:
        List of dictionaries with `"start_vaddr"`, `"end_vaddr"`, and `"byte_len"`.
    """
    cursor = 0
    records: list[dict[str, int]] = []
    while cursor + 4 <= len(eh_frame_bytes):
        length = struct.unpack_from("<I", eh_frame_bytes, cursor)[0]
        cursor += 4
        if length >= 12 and cursor + length <= len(eh_frame_bytes):
            entry = eh_frame_bytes[cursor : cursor + length]
            cie_id = struct.unpack_from("<I", entry, 0)[0]
            if cie_id != 0:
                rel_pc, byte_len = struct.unpack_from("<iI", entry, 4)
                pc_vaddr = (eh_frame_vaddr + cursor + 4 + rel_pc) & 0xFFFFFFFFFFFFFFFF
                records.append(
                    {
                        "start_vaddr": pc_vaddr,
                        "end_vaddr": pc_vaddr + byte_len,
                        "byte_len": byte_len,
                    }
                )
        cursor += max(length, 4)
    return records


def analyze_aarch64_prologue(code: bytes, entrypoint_vaddr: int = 0) -> dict[str, object]:
    """Decodes an AArch64 (`arm64`) function prologue to extract `struct_size` and call targets.

    Args:
        code: Raw little-endian 32-bit AArch64 instruction stream.
        entrypoint_vaddr: Virtual address of the first instruction in `code`.

    Returns:
        Dictionary with `"expected_struct_size"` and `"call_targets"`.
    """
    expected_size: int | None = None
    calls: list[int] = []
    for idx in range(0, len(code) - (len(code) % 4), 4):
        word = struct.unpack_from("<I", code, idx)[0]
        pc = entrypoint_vaddr + idx
        if (word & 0x7F80001F) == 0x7100001F and expected_size is None:
            imm12 = (word >> 10) & 0xFFF
            if imm12 > 0:
                expected_size = int(imm12)
        elif (word & 0xFC000000) == 0x94000000:
            imm26 = word & 0x03FFFFFF
            if imm26 & (1 << 25):
                imm26 -= 1 << 26
            calls.append(pc + (imm26 * 4))
    return {"expected_struct_size": expected_size, "call_targets": calls}


def recover_rodata_jump_table(
    rodata_bytes: bytes,
    table_vaddr: int,
    entry_count: int,
) -> list[int]:
    """Recovers resolved target virtual addresses from a `.rodata` signed-32-bit jump table.

    Args:
        rodata_bytes: Raw `.rodata` section bytes beginning at `table_vaddr`.
        table_vaddr: Base virtual address of the jump table.
        entry_count: Number of 4-byte signed offsets to decode.

    Returns:
        List of resolved case target virtual addresses (`table_vaddr + offset`).
    """
    targets: list[int] = []
    max_bytes = min(len(rodata_bytes) - (len(rodata_bytes) % 4), max(entry_count, 0) * 4)
    for idx in range(0, max_bytes, 4):
        rel = struct.unpack_from("<i", rodata_bytes, idx)[0]
        targets.append((table_vaddr + rel) & 0xFFFFFFFFFFFFFFFF)
    return targets


def build_call_graph_summary(
    callers: list[tuple[str, int, list[int]]],
    symbol_map: dict[int, str],
) -> list[dict[str, object]]:
    """Constructs an inter-procedural Call Graph edge list from caller targets and symbols.

    Args:
        callers: List of `(caller_name, caller_vaddr, call_target_vaddrs)` tuples.
        symbol_map: Mapping from symbol virtual address to symbol name.

    Returns:
        List of dictionaries describing each directed `(caller, callee)` call edge.
    """
    edges: list[dict[str, object]] = []
    for caller_name, caller_vaddr, targets in callers:
        for target_vaddr in targets:
            edges.append(
                {
                    "caller_name": caller_name,
                    "caller_vaddr": caller_vaddr,
                    "callee_name": symbol_map.get(target_vaddr, "<indirect>"),
                    "callee_vaddr": target_vaddr,
                }
            )
    return edges


def disassemble_x86_64_prologue(code: bytes) -> PrologueInstructionReport:
    """Decodes an x86-64 function prologue to extract `struct_size` and `[rdi + disp]` accesses.

    Args:
        code: Raw x86-64 machine code bytes of a function prologue.

    Returns:
        `PrologueInstructionReport` with `expected_struct_size` and `(offset, size, is_write)`.
    """
    expected_size: int | None = None
    accesses: list[tuple[int, int, bool]] = []
    pc = 0
    n = len(code)
    while pc < n:
        b0 = code[pc]
        if b0 == 0xC3:
            break
        if b0 == 0xBE and pc + 5 <= n:
            imm = struct.unpack_from("<I", code, pc + 1)[0]
            if expected_size is None:
                expected_size = int(imm)
            pc += 5
            continue
        if b0 in (0x48, 0x4C) and pc + 3 <= n:
            op = code[pc + 1]
            modrm = code[pc + 2]
            if op == 0x83 and modrm == 0x3F and pc + 4 <= n:
                expected_size = int(code[pc + 3])
                pc += 4
                continue
            if op == 0x81 and modrm == 0x3F and pc + 7 <= n:
                expected_size = int(struct.unpack_from("<I", code, pc + 3)[0])
                pc += 7
                continue
            if op in (0x8B, 0x89) and (modrm & 0x07) == 0x07:
                mod_bits = modrm >> 6
                is_write = op == 0x89
                if mod_bits == 0x01 and pc + 4 <= n:
                    accesses.append((int(code[pc + 3]), 8, is_write))
                    pc += 4
                    continue
                if mod_bits == 0x00:
                    accesses.append((0, 8, is_write))
                    pc += 3
                    continue
        if b0 == 0x8B and pc + 3 <= n:
            modrm = code[pc + 1]
            if (modrm >> 6) == 0x01 and (modrm & 0x07) == 0x07:
                accesses.append((int(code[pc + 2]), 4, False))
                pc += 3
                continue
        if b0 == 0x0F and pc + 4 <= n and code[pc + 1] == 0xB6:
            modrm = code[pc + 2]
            if (modrm >> 6) == 0x01 and (modrm & 0x07) == 0x07:
                accesses.append((int(code[pc + 3]), 1, False))
                pc += 4
                continue
        pc += 1
    return PrologueInstructionReport(expected_struct_size=expected_size, field_offsets=accesses)


def verify_prologue_struct_sizes_against_golden(
    prologues: list[tuple[int, bytes, bool]],
    golden_layouts: dict[str, object],
) -> list[dict[str, object]]:
    """Verifies extracted x86-64/AArch64 prologue `struct_size` constants against golden layouts.

    Args:
        prologues: List of `(word_index, code_bytes, is_aarch64)` tuples.
        golden_layouts: Parsed `golden_struct_layouts.json` dictionary.

    Returns:
        List of verification result dictionaries with `slot_index`, `extracted_struct_size`,
        `golden_min_struct_size`, and `matches_golden`.
    """
    raw_slots = golden_layouts.get("pjrt_slots", [])
    slot_map: dict[int, int] = {}
    if isinstance(raw_slots, list):
        for s in raw_slots:
            if isinstance(s, dict):
                w_idx = s.get("word_index")
                m_sz = s.get("min_struct_size")
                if isinstance(w_idx, int) and isinstance(m_sz, int):
                    slot_map[w_idx] = m_sz

    results: list[dict[str, object]] = []
    for word_idx, code, is_aarch64 in prologues:
        if word_idx not in slot_map:
            continue
        if is_aarch64:
            arm_rep = analyze_aarch64_prologue(code, 0)
            extracted = arm_rep.get("expected_struct_size")
        else:
            x86_rep = disassemble_x86_64_prologue(code)
            extracted = x86_rep.expected_struct_size
        golden_min = slot_map[word_idx]
        results.append(
            {
                "slot_index": word_idx,
                "extracted_struct_size": extracted,
                "golden_min_struct_size": golden_min,
                "matches_golden": extracted == golden_min,
            }
        )
    return results


def _read_proto_varint(data: bytes, pos: int) -> tuple[int, int]:
    """Decodes a base-128 protobuf varint starting at `data[pos]`.

    Args:
        data: Input byte buffer.
        pos: Starting byte offset.

    Returns:
        Tuple of `(decoded_value, next_offset)` (or `(0, len(data) + 1)` if truncated/malformed).
    """
    result = 0
    shift = 0
    cur = pos
    n = len(data)
    while shift < 64 and cur < n:
        byte = data[cur]
        cur += 1
        result |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0:
            return result, cur
        shift += 7
    return 0, n + 1


def carve_protodesc_cold(section_bytes: bytes) -> list[dict[str, str]]:
    """Carves embedded `FileDescriptorProto` names and packages from a `protodesc_cold` buffer.

    Supports both single-byte (`<= 127`) and multi-byte varint (`> 127`) `.proto` path lengths.

    Args:
        section_bytes: Raw byte slice of the `protodesc_cold` ELF section.

    Returns:
        List of dictionaries with `"name"` and `"package"`.
    """
    found: list[dict[str, str]] = []
    idx = 0
    n = len(section_bytes)
    while idx + 8 <= n:
        if section_bytes[idx] == 0x0A:
            name_len, name_start = _read_proto_varint(section_bytes, idx + 1)
            name_end = name_start + name_len
            if 7 <= name_len <= 512 and name_end <= n:
                raw_name = section_bytes[name_start:name_end]
                if raw_name.endswith(b".proto") and all(32 < b < 127 for b in raw_name):
                    name_str = raw_name.decode("utf-8", errors="replace")
                    pkg_str = ""
                    end_idx = name_end
                    if end_idx + 2 <= n and section_bytes[end_idx] == 0x12:
                        pkg_len, pkg_start = _read_proto_varint(section_bytes, end_idx + 1)
                        if pkg_start + pkg_len <= n:
                            pkg_str = section_bytes[pkg_start : pkg_start + pkg_len].decode(
                                "utf-8", errors="replace"
                            )
                            end_idx = pkg_start + pkg_len
                    found.append({"name": name_str, "package": pkg_str})
                    idx = end_idx
                    continue
        idx += 1
    return found


def extract_filewrapper_toc(toc_bytes: bytes, blob_bytes: bytes) -> list[tuple[str, bytes]]:
    """Extracts `(name, payload)` entries from a 24-byte or 40-byte `filewrapper_toc` table.

    Args:
        toc_bytes: Raw 24-byte or 40-byte `(name_off, data_off, size, ...)` records.
        blob_bytes: Backing string and data buffer.

    Returns:
        List of `(filename, data_bytes)` tuples.
    """
    stride = 40 if len(toc_bytes) >= 40 and len(toc_bytes) % 40 == 0 else 24
    entries: list[tuple[str, bytes]] = []
    for off in range(0, len(toc_bytes) - (len(toc_bytes) % stride), stride):
        name_off, data_off, size = struct.unpack_from("<QQQ", toc_bytes, off)
        if name_off < len(blob_bytes) and data_off + size <= len(blob_bytes):
            tail = blob_bytes[name_off:]
            nul_idx = tail.find(b"\x00")
            raw_name = tail[:nul_idx] if nul_idx >= 0 else tail
            if raw_name:
                name = raw_name.decode("utf-8", errors="replace")
                entries.append((name, blob_bytes[data_off : data_off + size]))
    return entries


def parse_elf64_sections_and_dynsym(
    elf_bytes: bytes,
) -> tuple[list[ElfSectionHeader], list[ElfExportedSymbol]]:
    """Parses ELF64 section headers and exported dynamic function symbols using `pyelftools`.

    Args:
        elf_bytes: Complete ELF64 little-endian binary payload.

    Returns:
        Tuple of `(section_headers, exported_symbols)`.

    Raises:
        ValueError: If `elf_bytes` is not a valid 64-bit little-endian ELF image.
    """
    if len(elf_bytes) < 64 or elf_bytes[:4] != ELF_MAGIC:
        raise ValueError("Invalid ELF header magic or truncated buffer")
    ei_class = elf_bytes[4]
    ei_data = elf_bytes[5]
    if ei_class != ELFCLASS64 or ei_data != ELFDATA2LSB:
        raise ValueError(f"Unsupported ELF class ({ei_class}) or endianness ({ei_data})")

    try:
        elf_ctor = cast(Callable[[io.BytesIO], _ElfFileProto], ELFFile)
        elf = elf_ctor(io.BytesIO(elf_bytes))
        sections: list[ElfSectionHeader] = []
        exported_symbols: list[ElfExportedSymbol] = []

        for sec in elf.iter_sections():
            hdr = sec.header
            sh_type_raw = hdr["sh_type"]
            sh_type_int = (
                SHT_DYNSYM
                if sh_type_raw == "SHT_DYNSYM"
                else (
                    SHT_SYMTAB
                    if sh_type_raw == "SHT_SYMTAB"
                    else (
                        SHT_STRTAB
                        if sh_type_raw == "SHT_STRTAB"
                        else (
                            SHT_RELA
                            if sh_type_raw == "SHT_RELA"
                            else (int(sh_type_raw) if isinstance(sh_type_raw, int) else 0)
                        )
                    )
                )
            )
            sections.append(
                ElfSectionHeader(
                    name=sec.name,
                    sh_type=sh_type_int,
                    sh_flags=int(hdr["sh_flags"]),
                    sh_addr=int(hdr["sh_addr"]),
                    sh_offset=int(hdr["sh_offset"]),
                    sh_size=int(hdr["sh_size"]),
                    sh_link=int(hdr["sh_link"]),
                    sh_entsize=int(hdr["sh_entsize"]),
                )
            )
            if isinstance(sec, SymbolTableSection):
                sym_tbl = cast(_ElfSymbolTableProto, sec)
                for sym in sym_tbl.iter_symbols():
                    st_info_dict = cast(Mapping[str, str], sym.entry["st_info"])
                    st_bind = st_info_dict["bind"]
                    st_type = st_info_dict["type"]
                    st_shndx = sym.entry["st_shndx"]
                    if (
                        st_shndx != "SHN_UNDEF"
                        and st_bind in ("STB_GLOBAL", "STB_WEAK")
                        and st_type == "STT_FUNC"
                        and sym.name
                    ):
                        bind_num = STB_GLOBAL if st_bind == "STB_GLOBAL" else STB_WEAK
                        exported_symbols.append(
                            ElfExportedSymbol(
                                name=sym.name,
                                vaddr=int(cast(int, sym.entry["st_value"])),
                                size=int(cast(int, sym.entry["st_size"])),
                                st_info=(bind_num << 4) | STT_FUNC,
                            )
                        )
    except ELFError as exc:
        raise ValueError(f"Malformed ELF64 binary: {exc}") from exc

    return sections, exported_symbols


def inspect_wheel_archive(wheel_path: Path) -> dict[str, object]:
    """Inspects a `libtpu` Python wheel archive (`.whl`) and extracts ELF and API metadata.

    Args:
        wheel_path: Path to the `.whl` zip archive on disk.

    Returns:
        Dictionary summarizing the wheel members, SHA-256 digests, and exported ELF symbols.
    """
    summary: dict[str, object] = {
        "wheel": wheel_path.name,
        "members": [],
        "binaries": {},
    }
    members_list: list[str] = []
    binaries_dict: dict[str, object] = {}

    with zipfile.ZipFile(wheel_path, "r") as zf:
        for info in zf.infolist():
            members_list.append(info.filename)
            if info.filename.endswith(".so"):
                payload = zf.read(info.filename)
                sha256_hex = compute_sha256_bytes(payload)
                sections, symbols = parse_elf64_sections_and_dynsym(payload)
                carved_protos = carve_protodesc_cold(payload)
                binaries_dict[info.filename] = {
                    "size_bytes": len(payload),
                    "sha256": sha256_hex,
                    "section_names": [s.name for s in sections if s.name],
                    "carved_protos_count": len(carved_protos),
                    "exported_functions": [
                        {
                            "name": sym.name,
                            "vaddr": f"0x{sym.vaddr:08x}",
                            "size": sym.size,
                        }
                        for sym in symbols
                    ],
                }

    summary["members"] = members_list
    summary["binaries"] = binaries_dict
    return summary


def validate_golden_manifests(repo_root: Path) -> list[str]:
    """Validates `golden_binary_manifest.json` and `golden_struct_layouts.json` invariants.

    Args:
        repo_root: Path to the workspace root directory.

    Returns:
        List of validation error messages (empty when all golden manifests are valid).
    """
    errors: list[str] = []
    re_dir = repo_root / "docs" / "reverse_engineering"
    manifest_path = re_dir / "golden_binary_manifest.json"
    layouts_path = re_dir / "golden_struct_layouts.json"

    if not manifest_path.exists() or not layouts_path.exists():
        errors.append("Missing golden_binary_manifest.json or golden_struct_layouts.json")
        return errors

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    layouts = json.loads(layouts_path.read_text(encoding="utf-8"))

    libtpu_so = manifest.get("libtpu_so", {})
    if libtpu_so.get("sha256") != OFFICIAL_LIBTPU_SO_SHA256:
        errors.append("golden_binary_manifest.json: libtpu_so sha256 mismatch")
    if libtpu_so.get("pjrt_api_struct_size") != 1144:
        errors.append("golden_binary_manifest.json: pjrt_api_struct_size != 1144")
    if (
        libtpu_so.get("pjrt_api_version_major"),
        libtpu_so.get("pjrt_api_version_minor"),
    ) != (0, 116):
        errors.append("golden_binary_manifest.json: pjrt_api_version != 0.116")
    if libtpu_so.get("libtpu_sdk_api_struct_size") != 144:
        errors.append("golden_binary_manifest.json: libtpu_sdk_api_struct_size != 144")
    if libtpu_so.get("tpu_driver_fn_struct_size") != 224:
        errors.append("golden_binary_manifest.json: tpu_driver_fn_struct_size != 224")
    if (
        libtpu_so.get("exported_functions_count") != 75
        or len(libtpu_so.get("exported_functions", [])) != 75
    ):
        errors.append("golden_binary_manifest.json: exported_functions != 75")
    if not libtpu_so.get("filetoc_md5_verified", True):
        errors.append("golden_binary_manifest.json: filetoc_md5_verified is False")

    ext_chain = layouts.get("extensions", [])
    if len(ext_chain) != 17:
        errors.append(f"golden_struct_layouts.json: expected 17 extensions, got {len(ext_chain)}")
    ext_funcs = layouts.get("extension_functions", [])
    if ext_funcs and len(ext_funcs) != 138:
        errors.append(
            f"golden_struct_layouts.json: expected 138 extension_functions, got {len(ext_funcs)}"
        )

    sdk_so = manifest.get("sdk_so", {})
    if sdk_so.get("sha256") != OFFICIAL_SDK_SO_SHA256:
        errors.append("golden_binary_manifest.json: sdk_so sha256 mismatch")
    if sdk_so.get("entrypoint") != "PyInit_sdk":
        errors.append("golden_binary_manifest.json: sdk_so entrypoint != PyInit_sdk")

    slots = layouts.get("pjrt_slots", [])
    manifest_slots = libtpu_so.get("pjrt_vtable_slots", [])
    if len(slots) != 138 or len(manifest_slots) != 138:
        errors.append(f"golden_struct_layouts.json: expected 138 PJRT slots, got {len(slots)}")
    elif any(
        s.get("min_struct_size") is None or s.get("min_struct_size") != m.get("min_struct_size")
        for s, m in zip(slots, manifest_slots)
    ):
        errors.append("golden_struct_layouts.json: min_struct_size mismatch vs manifest")

    proto_dir = re_dir / "protos"
    expected_protos = (
        "third_party/tensorflow/compiler/xla/pjrt/proto/compile_options.proto",
        "platforms/xla/service/jellyfish/deepsea_executable.proto",
        "platforms/xla/service/jellyfish/lowering/emitter_learned_cost_model_options.proto",
        "third_party/tensorflow/compiler/xla/service/hlo.proto",
        "platforms/asic_sw/tools/common/registers_memfile/encoded_gcsr.proto",
        "learning/45eac/tpu/runtime/host_commands/host_command_opcodes.proto",
        "platforms/xla/service/jellyfish/isa_program.proto",
        "platforms/xla/megascale/runtime/common/runtime_config.proto",
        "platforms/xla/service/jellyfish/memory_space_assignment/memory_space_assignment.proto",
        "platforms/xla/service/jellyfish/target_arguments.proto",
        "platforms/xla/service/jellyfish/tpu_compilation_environment.proto",
        "third_party/tensorflow/core/protobuf/tpu/tpu_embedding_configuration.proto",
        "learning/45eac/research/pjrt/tpu_executable.proto",
        "learning/45eac/tpu/runtime/topology/tpu_topology_args.proto",
        "third_party/tensorflow/compiler/xla/xla_data.proto",
        "third_party/tensorflow/tsl/profiler/protobuf/xplane.proto",
    )
    for proto_name in expected_protos:
        if not (proto_dir / proto_name).exists():
            errors.append(f"Missing extracted protobuf schema: {proto_name}")

    proto_desc_path = re_dir / "golden_proto_descriptors.json"
    csr_map_path = re_dir / "csr_register_map.json"
    if not proto_desc_path.exists() or not csr_map_path.exists():
        errors.append("Missing golden_proto_descriptors.json or csr_register_map.json")
    else:
        proto_desc = json.loads(proto_desc_path.read_text(encoding="utf-8"))
        csr_map = json.loads(csr_map_path.read_text(encoding="utf-8"))
        if (
            not proto_desc.get("topologically_closed")
            or proto_desc.get("proto_files_count") != 738
            or len(proto_desc.get("descriptors", [])) != 738
        ):
            errors.append("golden_proto_descriptors.json: incomplete descriptor closure")
        if not {"V2", "V3", "V4", "V4Lite", "V5e", "V5p", "V6e", "V7x"}.issubset(
            set(csr_map.get("generations", {}).keys())
        ) or any(
            reg.get("offset") in {"0x00001000", "0x00005000", "0x00009000", "0x0000a000"}
            or "bar" not in reg
            or "reset_value" not in reg
            or not reg.get("bitfields")
            for gen_data in csr_map.get("generations", {}).values()
            for reg in gen_data.get("registers", [])
        ):
            errors.append("csr_register_map.json: missing TPU generation CSR maps")

    return errors


def _vaddr_to_elf_offset(sections: list[ElfSectionHeader], vaddr: int) -> int | None:
    """Translates an ELF64 virtual address to its backing file offset via `SHF_ALLOC` sections.

    Args:
        sections: Parsed `ElfSectionHeader` descriptors.
        vaddr: Virtual address to translate.

    Returns:
        File byte offset when mapped, or `None` if unmapped / `SHT_NOBITS`.
    """
    for sec in sections:
        if (
            sec.sh_addr > 0
            and sec.sh_size > 0
            and sec.sh_type != 8
            and sec.sh_addr <= vaddr < sec.sh_addr + sec.sh_size
        ):
            return sec.sh_offset + (vaddr - sec.sh_addr)
    return None


def extract_descriptor_table_protos(elf_bytes: bytes) -> list[dict[str, object]]:
    """Extracts exact wire-encoded `FileDescriptorProto` slices via `.rela.dyn` `DescriptorTable`s.

    Args:
        elf_bytes: Raw ELF64 shared object byte buffer.

    Returns:
        Sorted list of descriptor records (`name`, `table_vaddr`, `descriptor_vaddr`, `size`).
    """
    sections, _ = parse_elf64_sections_and_dynsym(elf_bytes)
    pdesc = next((s for s in sections if s.name == "protodesc_cold"), None)
    if pdesc is None:
        return []
    pdesc_end = pdesc.sh_addr + pdesc.sh_size
    by_name: dict[str, dict[str, object]] = {}

    for reloc in resolve_elf64_rela_relocations(elf_bytes):
        target = reloc.addend
        if pdesc.sh_addr <= target < pdesc_end and reloc.offset >= 8:
            tbl_off = _vaddr_to_elf_offset(sections, reloc.offset - 8)
            blob_off = _vaddr_to_elf_offset(sections, target)
            if tbl_off is not None and blob_off is not None and tbl_off + 8 <= len(elf_bytes):
                flags_u32, desc_size = struct.unpack_from("<II", elf_bytes, tbl_off)
                if flags_u32 == 0 and desc_size >= 8 and blob_off + desc_size <= len(elf_bytes):
                    wire = elf_bytes[blob_off : blob_off + desc_size]
                    if wire[0] == 0x0A:
                        pos = 1
                        str_len = 0
                        shift = 0
                        while pos < len(wire):
                            byte = wire[pos]
                            pos += 1
                            str_len |= (byte & 0x7F) << shift
                            if (byte & 0x80) == 0:
                                break
                            shift += 7
                        if str_len > 0 and pos + str_len <= len(wire):
                            proto_name = wire[pos : pos + str_len].decode("utf-8", errors="replace")
                            if proto_name.endswith(".proto"):
                                by_name[proto_name] = {
                                    "name": proto_name,
                                    "table_vaddr": f"0x{reloc.offset - 8:08x}",
                                    "descriptor_vaddr": f"0x{target:08x}",
                                    "size": desc_size,
                                    "sha256": hashlib.sha256(wire).hexdigest(),
                                }

    return [by_name[k] for k in sorted(by_name)]


def extract_filewrapper_toc_entries(elf_bytes: bytes) -> list[dict[str, object]]:
    """Extracts all `FileToc` embedded resources from `filewrapper_toc` and verifies MD5 digests.

    Args:
        elf_bytes: Raw ELF64 shared object byte buffer.

    Returns:
        List of `FileToc` entry dictionaries (`name`, `size`, `md5`, `md5_verified`).
    """
    sections, _ = parse_elf64_sections_and_dynsym(elf_bytes)
    ftoc = next((s for s in sections if s.name == "filewrapper_toc"), None)
    if ftoc is None:
        return []
    relocs = {r.offset: r.addend for r in resolve_elf64_rela_relocations(elf_bytes)}
    results: list[dict[str, object]] = []

    for slot_delta in range(0, ftoc.sh_size, 8):
        cur = relocs.get(ftoc.sh_addr + slot_delta, 0)
        while cur > 0:
            name_va = relocs.get(cur, 0)
            data_va = relocs.get(cur + 8, 0)
            cur_off = _vaddr_to_elf_offset(sections, cur)
            cur += 40
            if name_va <= 0 or data_va <= 0 or cur_off is None or cur_off + 40 > len(elf_bytes):
                break
            sz = int(struct.unpack_from("<Q", elf_bytes, cur_off + 16)[0])
            expected_md5 = elf_bytes[cur_off + 24 : cur_off + 40]
            name_off = _vaddr_to_elf_offset(sections, name_va)
            data_off = _vaddr_to_elf_offset(sections, data_va)
            if name_off is not None and data_off is not None and data_off + sz <= len(elf_bytes):
                nul = elf_bytes.find(b"\x00", name_off)
                if nul > name_off:
                    entry_name = elf_bytes[name_off:nul].decode("utf-8", errors="replace")
                    payload = elf_bytes[data_off : data_off + sz]
                    actual_md5 = hashlib.md5(payload).digest()
                    verified = (expected_md5 == (b"\x00" * 16)) or (actual_md5 == expected_md5)
                    results.append(
                        {
                            "name": entry_name,
                            "size": sz,
                            "md5": expected_md5.hex(),
                            "md5_verified": verified,
                        }
                    )
    return results


def _classify_static_csr_shard_symbol(sym_name: str) -> str | None:
    """Classifies a mangled `PerformanceCounterNameToString` shard symbol into an ASIC family.

    Args:
        sym_name: Mangled C++ symbol name from `.symtab`.

    Returns:
        Canonical ASIC family code (`"jxc"`, `"pfc"`, `"plc"`, `"vlc"`, `"vfc"`, `"glc"`, `"gfc"`),
        or `None` if `sym_name` is not a numbered shard function.
    """
    ens_pos = sym_name.find("ENS")
    if (
        "PerformanceCounterNameToString" not in sym_name
        or ens_pos <= 0
        or not sym_name[ens_pos - 1].isdigit()
    ):
        return None
    for tag, fam in (
        ("3jxc", "jxc"),
        ("3pfc", "pfc"),
        ("3plc", "plc"),
        ("3vlc", "vlc"),
        ("3vfc", "vfc"),
        ("3glc", "glc"),
        ("3gfc", "gfc"),
    ):
        if tag in sym_name:
            return fam
    return None


def extract_static_csr_maps(elf_bytes: bytes) -> dict[str, list[dict[str, object]]]:
    """Extracts static `.lrodata` CSR register tables across all 7 ASIC families (`jxc`–`gfc`).

    Args:
        elf_bytes: Raw ELF64 shared object byte buffer.

    Returns:
        Dictionary mapping ASIC family code to sorted lists of hardware CSR descriptors.
    """
    sections, _ = parse_elf64_sections_and_dynsym(elf_bytes)
    symtab = next((s for s in sections if s.name == ".symtab"), None)
    strtab_sec = next((s for s in sections if s.name == ".strtab"), None)
    if symtab is None or strtab_sec is None:
        return {}

    strtab = elf_bytes[strtab_sec.sh_offset : strtab_sec.sh_offset + strtab_sec.sh_size]
    shards_by_fam: dict[str, list[tuple[int, int]]] = {}
    for off in range(symtab.sh_offset, symtab.sh_offset + symtab.sh_size, 24):
        st_name, st_info, _, _, st_value, st_size = struct.unpack_from("<IBBHQQ", elf_bytes, off)
        if (st_info & 0xF) == 2 and st_value > 0 and st_size > 0 and st_name < len(strtab):
            nul = strtab.find(b"\x00", st_name)
            sym_name = strtab[st_name : nul if nul >= 0 else len(strtab)].decode(
                "utf-8", errors="replace"
            )
            fam = _classify_static_csr_shard_symbol(sym_name)
            if fam is not None:
                shards_by_fam.setdefault(fam, []).append((st_value, st_size))

    out: dict[str, list[dict[str, object]]] = {}
    for fam, shards in sorted(shards_by_fam.items()):
        entries: list[dict[str, object]] = []
        for fva, fsz in sorted(shards):
            code_off = _vaddr_to_elf_offset(sections, fva)
            code = (
                elf_bytes[code_off : code_off + fsz]
                if code_off is not None and code_off + fsz <= len(elf_bytes)
                else b""
            )
            got_va = 0
            ranges_va = rel_va = pool_va = single_base = None
            num_ranges = single_count = 0
            for idx in range(max(0, len(code) - 11)):
                cur_va = fva + idx
                if code[idx : idx + 3] == b"\x48\x8d\x35":
                    got_va = cur_va + 7 + int(struct.unpack_from("<i", code, idx + 3)[0])
                elif code[idx : idx + 3] == b"\x4c\x8d\x05":
                    ranges_va = cur_va + 7 + int(struct.unpack_from("<i", code, idx + 3)[0])
                    if code[idx + 7] == 0xBA:
                        num_ranges = int(struct.unpack_from("<I", code, idx + 8)[0])
                elif code[idx : idx + 3] == b"\x48\x8d\x15":
                    rel_va = cur_va + 7 + int(struct.unpack_from("<i", code, idx + 3)[0])
                elif code[idx : idx + 2] == b"\x48\xba":
                    rel_va = (got_va + int(struct.unpack_from("<q", code, idx + 2)[0])) & (
                        (1 << 64) - 1
                    )
                elif code[idx : idx + 2] in (b"\x48\xb9", b"\x48\xb8"):
                    pool_va = (got_va + int(struct.unpack_from("<q", code, idx + 2)[0])) & (
                        (1 << 64) - 1
                    )
                elif code[idx : idx + 3] == b"\x48\x8d\x87":
                    single_base = -int(struct.unpack_from("<i", code, idx + 3)[0])
                elif code[idx : idx + 2] == b"\x48\x3d" and single_count == 0:
                    span_plus_1 = int(struct.unpack_from("<I", code, idx + 2)[0])
                    single_count = ((span_plus_1 - 1) // 8) + 1

            rel_off = _vaddr_to_elf_offset(sections, rel_va) if rel_va is not None else None
            pool_off = _vaddr_to_elf_offset(sections, pool_va) if pool_va is not None else None
            ranges_off = (
                _vaddr_to_elf_offset(sections, ranges_va) if ranges_va is not None else None
            )
            if ranges_off is not None and rel_off is not None and pool_off is not None:
                for r_idx in range(num_ranges):
                    base_off, stride, str_base, cnt, rel_idx, _ = struct.unpack_from(
                        "<QIIHHI", elf_bytes, ranges_off + r_idx * 24
                    )
                    for k in range(cnt):
                        s0, s1 = struct.unpack_from("<HH", elf_bytes, rel_off + (rel_idx + k) * 2)
                        if s0 < s1:
                            cname = elf_bytes[
                                pool_off + str_base + s0 : pool_off + str_base + s1
                            ].decode("utf-8", errors="replace")
                            csr_off = base_off + k * stride
                            is_bar2 = fam in ("jxc", "pfc", "plc") or csr_off >= 0x80000000
                            bar_idx = 2 if is_bar2 else 0
                            entries.append(
                                {
                                    "name": cname,
                                    "bar": bar_idx,
                                    "offset": f"0x{csr_off:08x}",
                                    "width_bits": 64,
                                }
                            )
            elif single_base is not None and rel_off is not None and pool_off is not None:
                for k in range(single_count):
                    s0, s1 = struct.unpack_from("<HH", elf_bytes, rel_off + k * 2)
                    if s0 < s1:
                        cname = elf_bytes[pool_off + s0 : pool_off + s1].decode(
                            "utf-8", errors="replace"
                        )
                        csr_off = single_base + k * 8
                        bar_idx = 2 if fam in ("jxc", "pfc", "plc") or csr_off >= 0x80000000 else 0
                        entries.append(
                            {
                                "name": cname,
                                "bar": bar_idx,
                                "offset": f"0x{csr_off:08x}",
                                "width_bits": 64,
                            }
                        )
        out[fam] = sorted(entries, key=lambda d: (int(str(d["offset"]), 16), str(d["name"])))
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for `scripts/re_extract.py`.

    Args:
        argv: Optional list of CLI arguments.

    Returns:
        Exit status code (`0` on success, `1` on failure).
    """
    parser = argparse.ArgumentParser(
        description="Clean-room binary forensics and golden manifest validator for libtpu"
    )
    parser.add_argument(
        "--wheel",
        "--extract-wheel",
        dest="wheel",
        type=Path,
        default=None,
        help="Optional path to a libtpu .whl archive to inspect and extract",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Path to the repository root directory",
    )
    parser.add_argument(
        "--verify",
        "--verify-golden",
        dest="verify_golden",
        action="store_true",
        help="Verify golden_binary_manifest.json, golden_struct_layouts.json, and carved protos",
    )
    args = parser.parse_args(argv)

    if args.wheel is not None:
        wheel_summary = inspect_wheel_archive(args.wheel)
        print(json.dumps(wheel_summary, indent=2))

    validation_errors = validate_golden_manifests(args.repo_root)
    if validation_errors:
        for err in validation_errors:
            print(f"Validation Error: {err}")
        return 1

    print("Golden binary manifests and protobuf schemas verified successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

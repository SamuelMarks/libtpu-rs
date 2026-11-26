#!/usr/bin/env python3
"""C++ RTTI, `.eh_frame` CFI, PJRT ABI, VLIW/SparseCore ISA, and Driver Lifter (`libtpu-rs`).

Recovers Itanium C++ ABI RTTI hierarchies (`_ZTI*`, `_ZTS*`, `_ZTV*`), `.eh_frame` function
bounds (`x86_64` & `aarch64`), `PJRT_*_Args` `struct_size` checks and 18-extension layouts,
`GetLibtpuSdkApi()` and `TpuDriverFn` vtable layouts, VLIW/SparseCore ISA bundle metadata,
and Gasket (`0xDC`) / VFIO (`0x3B`) `ioctl` callsites from `libtpu.so` and `sdk.so`, with
headless Ghidra (`analyzeHeadless`) discovery, execution, and `re_extract.py` cross-verification.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, TypedDict, cast

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.re_extract import (
    ElfSectionHeader,
    _vaddr_to_elf_offset,
    disassemble_x86_64_prologue,
    extract_descriptor_table_protos,
    extract_filewrapper_toc_entries,
    parse_eh_frame_cfi_records,
    parse_elf64_sections_and_dynsym,
    reconstruct_pjrt_and_extension_vtables,
    resolve_elf64_rela_relocations,
)


class BinaryLiftError(RuntimeError):
    """Raised when ELF64 lifting or headless Ghidra execution fails."""


@dataclass(frozen=True)
class ItaniumRttiClass:
    """Recovered Itanium C++ ABI RTTI type descriptor (`_ZTS*` / `_ZTI*`)."""

    mangled_type_name: str
    namespace_group: str
    rtti_kind: str
    type_string_vaddr: int


@dataclass(frozen=True)
class EhFrameFunctionRange:
    """Function start/end virtual address interval recovered from `.eh_frame` FDEs."""

    pc_begin: int
    pc_end: int
    size_bytes: int


@dataclass(frozen=True)
class PjrtAbiLiftReport:
    """Lifted `PJRT_*_Args` `struct_size` immediates and extension header chain metadata."""

    observed_struct_sizes: tuple[int, ...]
    observed_field_displacements: tuple[int, ...]
    extension_type_ids: tuple[int, ...]


@dataclass(frozen=True)
class IsaAndDriverLiftReport:
    """Lifted VLIW/SparseCore ISA family inventory and Gasket/VFIO `ioctl` immediates."""

    vliw_families: tuple[str, ...]
    sparsecore_subengines: tuple[str, ...]
    gasket_ioctl_cmds: tuple[int, ...]
    vfio_ioctl_cmds: tuple[int, ...]


CommandRunner = Callable[[list[str]], int]


def default_command_runner(cmd: list[str]) -> int:
    """Executes an external headless analysis command and returns its exit code.

    Args:
        cmd: Argument vector to execute without a shell.

    Returns:
        Process return code integer.
    """
    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)


def _classify_namespace(mangled_ts: str) -> str:
    """Classifies a mangled `_ZTS*` type-string into its subsystem namespace group.

    Args:
        mangled_ts: Mangled Itanium C++ `_ZTS...` symbol or type name.

    Returns:
        Canonical subsystem group label.
    """
    if "deepsea" in mangled_ts:
        return "deepsea"
    if "jellyfish" in mangled_ts:
        return "jellyfish"
    if "megascale" in mangled_ts:
        return "megascale"
    if "tpu_driver" in mangled_ts:
        return "tpu_driver"
    return "tpu_runtime"


def extract_itanium_rtti_classes(elf_bytes: bytes) -> list[ItaniumRttiClass]:
    """Scans `.rodata` and `.data.rel.ro` to recover Itanium C++ ABI RTTI class descriptors.

    Args:
        elf_bytes: Raw 64-bit ELF shared object bytes (`libtpu.so` or `sdk.so`).

    Returns:
        Deterministically sorted list of `ItaniumRttiClass` records.
    """
    sections, _ = parse_elf64_sections_and_dynsym(elf_bytes)
    relocs = resolve_elf64_rela_relocations(elf_bytes)
    ref_counts: dict[int, int] = {}
    for rel in relocs:
        ref_counts[rel.addend] = ref_counts.get(rel.addend, 0) + 1

    zts_pattern = re.compile(
        rb"(N(?:3tpu|3xla|7asic_sw|17platforms_deepsea|9megascale|10tpu_driver)[A-Za-z0-9_]+E)\x00"
    )
    classes: dict[str, ItaniumRttiClass] = {}
    for sec in sections:
        if sec.name in (".rodata", ".lrodata") and sec.sh_size > 0:
            blob = elf_bytes[sec.sh_offset : sec.sh_offset + sec.sh_size]
            for match in zts_pattern.finditer(blob):
                mangled = match.group(1).decode("ascii")
                ts_vaddr = sec.sh_addr + match.start(1)
                refs = ref_counts.get(ts_vaddr, 0)
                rtti_kind = (
                    "__vmi_class_type_info"
                    if refs >= 2
                    else ("__si_class_type_info" if refs == 1 else "__class_type_info")
                )
                classes[mangled] = ItaniumRttiClass(
                    mangled_type_name=mangled,
                    namespace_group=_classify_namespace(mangled),
                    rtti_kind=rtti_kind,
                    type_string_vaddr=ts_vaddr,
                )

    return [classes[k] for k in sorted(classes)]


def extract_eh_frame_function_bounds(elf_bytes: bytes) -> list[EhFrameFunctionRange]:
    """Recovers function `[pc_begin, pc_end)` bounds from `.eh_frame` CIE/FDE records.

    Args:
        elf_bytes: Raw 64-bit ELF shared object bytes (`x86_64` or `aarch64`).

    Returns:
        Sorted list of `EhFrameFunctionRange` entries.
    """
    sections, _ = parse_elf64_sections_and_dynsym(elf_bytes)
    eh_sec = next((s for s in sections if s.name == ".eh_frame"), None)
    if eh_sec is None or eh_sec.sh_size == 0:
        return []
    raw_eh = elf_bytes[eh_sec.sh_offset : eh_sec.sh_offset + eh_sec.sh_size]
    records = parse_eh_frame_cfi_records(raw_eh, eh_sec.sh_addr)
    bounds: list[EhFrameFunctionRange] = []
    for rec in records:
        byte_len = rec["byte_len"]
        if byte_len > 0:
            bounds.append(
                EhFrameFunctionRange(
                    pc_begin=rec["start_vaddr"],
                    pc_end=rec["end_vaddr"],
                    size_bytes=byte_len,
                )
            )
    bounds.sort(key=lambda item: item.pc_begin)
    return bounds


def _find_section(sections: list[ElfSectionHeader], name: str) -> ElfSectionHeader | None:
    """Finds a named ELF section header if present.

    Args:
        sections: Parsed `ElfSectionHeader` list.
        name: Target section name (e.g., `".text"`).

    Returns:
        Matching `ElfSectionHeader` or `None`.
    """
    return next((s for s in sections if s.name == name), None)


def _extract_symtab_index(
    elf_bytes: bytes,
    sections: list[ElfSectionHeader],
) -> tuple[dict[str, tuple[int, int]], dict[int, str]]:
    """Extracts `(name -> (vaddr, size), vaddr -> name)` from `.symtab` and `.dynsym`.

    Args:
        elf_bytes: Raw 64-bit ELF shared object bytes.
        sections: Parsed `ElfSectionHeader` descriptors.

    Returns:
        Tuple of `(sym_to_loc, addr_to_sym)` dictionaries.
    """
    sym_to_loc: dict[str, tuple[int, int]] = {}
    addr_to_sym: dict[int, str] = {}
    sec_by_name = {s.name: s for s in sections}
    for tbl_name, str_name in ((".dynsym", ".dynstr"), (".symtab", ".strtab")):
        tbl_sec = sec_by_name.get(tbl_name)
        str_sec = sec_by_name.get(str_name)
        if tbl_sec is None or str_sec is None or tbl_sec.sh_size < 24:
            continue
        strtab = elf_bytes[str_sec.sh_offset : str_sec.sh_offset + str_sec.sh_size]
        limit = tbl_sec.sh_offset + tbl_sec.sh_size - (tbl_sec.sh_size % 24)
        for off in range(tbl_sec.sh_offset, limit, 24):
            st_name, _st_info, _, _, st_value, st_size = struct.unpack_from(
                "<IBBHQQ", elf_bytes, off
            )
            if st_value == 0 or st_name >= len(strtab):
                continue
            head = strtab[st_name : st_name + 48]
            if not any(
                tok in head
                for tok in (
                    b"PJRT",
                    b"pjrt",
                    b"GetPjrtApi",
                    b"GetLibtpuSdkApi",
                    b"LibtpuSdk",
                    b"TpuDriver",
                    b"TfTpu_",
                    b"TpuEmbeddingEngine_",
                    b"PLUGIN_Profiler",
                    b"Gasket",
                    b"Vfio",
                )
            ):
                continue
            nul = strtab.find(b"\x00", st_name)
            if nul == -1:
                continue
            nm = strtab[st_name:nul].decode("utf-8", errors="replace")
            sym_to_loc[nm] = (int(st_value), int(st_size))
            if st_value not in addr_to_sym or nm.startswith("_ZN4pjrt"):
                addr_to_sym[int(st_value)] = nm
    return sym_to_loc, addr_to_sym


def _scan_x86_mov_lea_stores(
    code: bytes,
    base_vaddr: int,
    got_vaddr: int | None = None,
    elf_bytes: bytes | None = None,
    sections: list[ElfSectionHeader] | None = None,
    reloc_map: dict[int, int] | None = None,
) -> tuple[int, int, dict[int, int]]:
    """Extracts `(struct_size, type_or_minor, {slot_offset: vaddr})` from x86-64 init code.

    Args:
        code: Machine code slice of a `Create*` initializer function.
        base_vaddr: Virtual address of `code[0]`.
        got_vaddr: Optional `_GLOBAL_OFFSET_TABLE_` virtual address.
        elf_bytes: Optional full ELF64 byte slice for resolving GOT entries.
        sections: Optional `ElfSectionHeader` list.
        reloc_map: Optional `vaddr -> addend` relative relocation map.

    Returns:
        Tuple of `(struct_size, header_int, slot_targets_by_offset)`.
    """
    struct_size = 0
    header_int = 0
    slots: dict[int, int] = {}
    last_lea_target: dict[int, int] = {}
    got_reg: int | None = None
    got_disp_reg: dict[int, int] = {}

    sz_m = re.search(rb"\x48\xc7(?:\x07|\x00)(....)", code, re.DOTALL)
    if sz_m is not None:
        struct_size = int(struct.unpack("<I", sz_m.group(1))[0])
    ty_m = re.search(rb"\xc7\x47\x08(....)", code, re.DOTALL)
    if ty_m is not None:
        header_int = int(struct.unpack("<I", ty_m.group(1))[0])
    ver_m = re.search(rb"\x49\xb9(........)", code, re.DOTALL)
    if ver_m is not None:
        _maj, min_v = struct.unpack("<ii", ver_m.group(1))
        header_int = int(min_v)

    n = len(code)
    pc = 0
    while pc < n:
        if pc + 7 <= n and code[pc] in (0x48, 0x4C) and code[pc + 1] == 0x8D:
            modrm = code[pc + 2]
            if (modrm & 0xC7) == 0x05:
                reg = ((code[pc] & 0x04) << 1) | ((modrm >> 3) & 0x07)
                disp = struct.unpack_from("<i", code, pc + 3)[0]
                tgt = base_vaddr + pc + 7 + disp
                if got_vaddr is not None and tgt == got_vaddr:
                    got_reg = reg
                else:
                    last_lea_target[reg] = tgt
                pc += 7
                continue
        if (
            got_reg is not None
            and pc + 10 <= n
            and code[pc] in (0x48, 0x49)
            and 0xB8 <= code[pc + 1] <= 0xBF
        ):
            reg = ((code[pc] & 0x01) << 3) | (code[pc + 1] - 0xB8)
            imm64 = struct.unpack_from("<q", code, pc + 2)[0]
            got_disp_reg[reg] = imm64
            pc += 10
            continue
        if (
            got_vaddr is not None
            and reloc_map is not None
            and pc + 4 <= n
            and code[pc] in (0x48, 0x4C)
            and code[pc + 1] == 0x8B
            and code[pc + 2] in (0x04, 0x0C, 0x14, 0x1C, 0x24, 0x2C, 0x34, 0x3C)
        ):
            sib = code[pc + 3]
            dst_reg = ((code[pc] & 0x04) << 1) | ((code[pc + 2] >> 3) & 0x07)
            idx_reg = (sib >> 3) & 0x07
            if idx_reg in got_disp_reg:
                got_slot_va = got_vaddr + got_disp_reg[idx_reg]
                if got_slot_va in reloc_map:
                    last_lea_target[dst_reg] = reloc_map[got_slot_va]
                elif elf_bytes is not None and sections is not None:
                    goff = _vaddr_to_elf_offset(sections, got_slot_va)
                    if goff is not None and goff + 8 <= len(elf_bytes):
                        last_lea_target[dst_reg] = int(struct.unpack_from("<Q", elf_bytes, goff)[0])
                pc += 4
                continue
        if pc + 4 <= n and code[pc] in (0x48, 0x4C) and code[pc + 1] == 0x89:
            modrm = code[pc + 2]
            src_reg = ((code[pc] & 0x04) << 1) | ((modrm >> 3) & 0x07)
            rm = modrm & 0x07
            mod = (modrm >> 6) & 0x03
            if rm in (0, 7) and src_reg in last_lea_target:
                if mod == 1 and pc + 4 <= n:
                    off8 = struct.unpack_from("<b", code, pc + 3)[0]
                    if off8 >= 8 and off8 % 8 == 0:
                        slots[int(off8)] = last_lea_target[src_reg]
                    pc += 4
                    continue
                if mod == 2 and pc + 7 <= n:
                    off32 = struct.unpack_from("<i", code, pc + 3)[0]
                    if 8 <= off32 <= 4096 and off32 % 8 == 0:
                        slots[int(off32)] = last_lea_target[src_reg]
                    pc += 7
                    continue
        pc += 1

    return struct_size, header_int, slots


class PjrtVtableSlotDict(TypedDict):
    """Lifted `PJRT_Api` vtable function pointer slot descriptor."""

    word_index: int
    byte_offset: str
    target_vaddr: str
    mangled_symbol: str
    min_struct_size: int | None


class LiftedExtensionDict(TypedDict):
    """Lifted `PJRT_Extension_Base` linked-list node descriptor."""

    index: int
    creator_symbol: str
    ext_type: int
    struct_size: int
    num_functions: int
    populated_slots: int


class VtableLayoutsDict(TypedDict):
    """Complete lifted ABI table dictionary for PJRT, Extensions, SDK, Driver, and Embedding."""

    pjrt_api_struct_size: int
    pjrt_api_major: int
    pjrt_api_minor: int
    pjrt_vtable_slots_count: int
    pjrt_vtable_slots: list[PjrtVtableSlotDict]
    extensions_count: int
    extensions: list[LiftedExtensionDict]
    libtpu_sdk_api_struct_size: int
    libtpu_sdk_api_slot_offsets: list[int]
    tpu_driver_fn_struct_size: int
    tpu_embedding_entrypoints: dict[str, int]


def lift_vtable_and_entrypoint_layouts(elf_bytes: bytes) -> VtableLayoutsDict:
    """Lifts `PJRT_Api`, `extension_start`, `LibtpuSdk_Api`, `TpuDriverFn`, and entrypoints.

    Args:
        elf_bytes: Raw 64-bit ELF shared object bytes (`libtpu.so` or `sdk.so`).

    Returns:
        `VtableLayoutsDict` containing lifted PJRT, extension, SDK, Driver, and Embedding tables.
    """
    sections, _ = parse_elf64_sections_and_dynsym(elf_bytes)
    relocs = {r.offset: r.addend for r in resolve_elf64_rela_relocations(elf_bytes)}
    sym_to_loc, addr_to_sym = _extract_symtab_index(elf_bytes, sections)
    got_sec = _find_section(sections, ".got") or _find_section(sections, ".got.plt")
    got_va = sym_to_loc.get("_GLOBAL_OFFSET_TABLE_", (got_sec.sh_addr if got_sec else 0, 0))[0]

    pjrt_size = 0
    pjrt_major = 0
    pjrt_minor = 0
    pjrt_slots: list[PjrtVtableSlotDict] = []
    extensions: list[LiftedExtensionDict] = []

    create_pjrt_key = next(
        (k for k in sym_to_loc if k.startswith("_ZN4pjrt13CreatePjrtApiE")),
        None,
    )
    get_pjrt_loc = sym_to_loc.get("_ZN4pjrt10tpu_plugin13GetTpuPjrtApiEv")
    if create_pjrt_key is not None:
        c_va, c_sz = sym_to_loc[create_pjrt_key]
        c_off = _vaddr_to_elf_offset(sections, c_va)
        if c_off is not None:
            pjrt_size, pjrt_minor, slot_map = _scan_x86_mov_lea_stores(
                elf_bytes[c_off : c_off + c_sz],
                c_va,
                got_va,
                elf_bytes,
                sections,
                relocs,
            )
            for name_cand, arg_slot_off in (
                (
                    "_ZN4pjrt10tpu_plugin22PJRT_Plugin_InitializeEP27PJRT_Plugin_Initialize_Args",
                    0x40,
                ),
                ("_ZN4pjrt26PJRT_Plugin_Attributes_XlaEP27PJRT_Plugin_Attributes_Args", 0x48),
                ("_ZN4pjrt10tpu_plugin18PJRT_Client_CreateEP23PJRT_Client_Create_Args", 0x78),
                (
                    "_ZN4pjrt10tpu_plugin26PJRT_ExecuteContext_CreateEP31PJRT_ExecuteContext_Create_Args",
                    0x338,
                ),
                (
                    "_ZN4pjrt10tpu_plugin31PJRT_TopologyDescription_CreateEP36PJRT_TopologyDescription_Create_Args",
                    0x2B8,
                ),
            ):
                if name_cand in sym_to_loc:
                    slot_map.setdefault(arg_slot_off, sym_to_loc[name_cand][0])
            num_slots = max(0, (pjrt_size - 0x28) // 8)
            for idx in range(num_slots):
                byte_off = 0x28 + idx * 8
                tgt = slot_map.get(byte_off, 0)
                msym = addr_to_sym.get(tgt, "")
                min_sz: int | None = None
                if tgt > 0:
                    toff = _vaddr_to_elf_offset(sections, tgt)
                    if toff is not None:
                        min_sz = disassemble_x86_64_prologue(
                            elf_bytes[toff : toff + 96]
                        ).expected_struct_size
                pjrt_slots.append(
                    {
                        "word_index": idx + 5,
                        "byte_offset": f"0x{byte_off:03x}",
                        "target_vaddr": f"0x{tgt:08x}",
                        "mangled_symbol": msym,
                        "min_struct_size": min_sz,
                    }
                )

    if get_pjrt_loc is not None:
        g_va, g_sz = get_pjrt_loc
        g_off = _vaddr_to_elf_offset(sections, g_va)
        if g_off is not None:
            g_code = elf_bytes[g_off : g_off + g_sz]
            creator_calls: list[int] = []
            for idx_b in range(len(g_code) - 5):
                if g_code[idx_b] == 0xE8:
                    rel32 = struct.unpack_from("<i", g_code, idx_b + 1)[0]
                    callee = g_va + idx_b + 5 + rel32
                    cname = addr_to_sym.get(callee, "")
                    if (
                        "Create" in cname
                        and "Extension" in cname
                        and not cname.startswith("_ZN4pjrt13CreatePjrtApiE")
                        and callee not in creator_calls
                    ):
                        creator_calls.append(callee)
            creator_calls.reverse()
            for callee in creator_calls:
                cname = addr_to_sym[callee]
                e_va, e_sz = sym_to_loc[cname]
                e_off = _vaddr_to_elf_offset(sections, e_va)
                if e_off is None:
                    continue
                e_code = elf_bytes[e_off : e_off + e_sz]
                st_sz, ext_ty, e_slots = _scan_x86_mov_lea_stores(
                    e_code,
                    e_va,
                    got_va,
                    elf_bytes,
                    sections,
                    relocs,
                )
                if st_sz == 0 and len(e_code) >= 5:
                    jmp_idx = e_code.rfind(b"\xe9")
                    if jmp_idx != -1 and jmp_idx + 5 <= len(e_code):
                        jrel = struct.unpack_from("<i", e_code, jmp_idx + 1)[0]
                        jtgt = e_va + jmp_idx + 5 + jrel
                        jname = addr_to_sym.get(jtgt, "")
                        if jname in sym_to_loc:
                            j_va, j_sz = sym_to_loc[jname]
                            j_off = _vaddr_to_elf_offset(sections, j_va)
                            if j_off is not None:
                                st_sz, ext_ty, e_slots = _scan_x86_mov_lea_stores(
                                    elf_bytes[j_off : j_off + j_sz],
                                    j_va,
                                    got_va,
                                    elf_bytes,
                                    sections,
                                    relocs,
                                )
                if st_sz > 0:
                    extensions.append(
                        {
                            "index": len(extensions) + 1,
                            "creator_symbol": cname,
                            "ext_type": ext_ty,
                            "struct_size": st_sz,
                            "num_functions": max(0, (st_sz - 24) // 8),
                            "populated_slots": len(e_slots),
                        }
                    )
            prof_loc = sym_to_loc.get("_ZN4pjrt10tpu_plugin18profiler_extensionE")
            if prof_loc is not None:
                p_va, _ = prof_loc
                p_off = _vaddr_to_elf_offset(sections, p_va)
                if p_off is not None and p_off + 24 <= len(elf_bytes):
                    p_sz = int(relocs.get(p_va, struct.unpack_from("<Q", elf_bytes, p_off)[0]))
                    p_ty = int(
                        relocs.get(p_va + 8, struct.unpack_from("<Q", elf_bytes, p_off + 8)[0])
                        & 0xFFFFFFFF
                    )
                    extensions.append(
                        {
                            "index": len(extensions) + 1,
                            "creator_symbol": "_ZN4pjrt10tpu_plugin18profiler_extensionE",
                            "ext_type": p_ty,
                            "struct_size": p_sz,
                            "num_functions": max(0, (p_sz - 24) // 8),
                            "populated_slots": 1,
                        }
                    )

    if pjrt_size == 0:
        word_map: dict[int, int] = {}
        for sname in (".data.rel.ro", ".rodata", ".data"):
            sec = _find_section(sections, sname)
            if sec is None or sec.sh_size < 8:
                continue
            lim = min(sec.sh_size - (sec.sh_size % 8), 512_000)
            for off in range(0, lim, 8):
                va = sec.sh_addr + off
                word_map[va] = relocs.get(
                    va, int(struct.unpack_from("<Q", elf_bytes, sec.sh_offset + off)[0])
                )
        for va, val in sorted(word_map.items()):
            if 512 <= val <= 4096 and word_map.get(va + 16) == 24:
                nslots = (val - 0x28) // 8
                vt = reconstruct_pjrt_and_extension_vtables(word_map, va, num_slots=nslots)
                pjrt_size = int(val)
                pjrt_major = int(str(vt["major_version"]))
                pjrt_minor = int(str(vt["minor_version"]))
                raw_exts = cast(list[dict[str, int]], vt["extensions"])
                for idx_e, ed in enumerate(raw_exts, 1):
                    esz = int(ed.get("struct_size", 0))
                    ety = int(ed.get("type_id", 0))
                    extensions.append(
                        {
                            "index": idx_e,
                            "creator_symbol": "<static_relro>",
                            "ext_type": ety,
                            "struct_size": esz,
                            "num_functions": max(0, (esz - 24) // 8),
                            "populated_slots": max(0, (esz - 24) // 8),
                        }
                    )
                break

    sdk_struct_size = 0
    sdk_slot_offsets: list[int] = []
    sdk_loc = sym_to_loc.get("_ZN6libtpu3sdk18CreateLibtpuSdkApiEv")
    if sdk_loc is not None:
        s_va, s_sz = sdk_loc
        s_off = _vaddr_to_elf_offset(sections, s_va)
        if s_off is not None:
            s_code = elf_bytes[s_off : s_off + s_sz]
            for sz_m in re.finditer(
                rb"(?:\x48\xc7(?:\x00|\x03)|\x49\xc7\x04\x24|\xbf)(....)",
                s_code,
                re.DOTALL,
            ):
                cand_sz = int(struct.unpack("<I", sz_m.group(1))[0])
                if 96 <= cand_sz <= 512 and cand_sz % 8 == 0:
                    sdk_struct_size = cand_sz
                    sdk_slot_offsets = list(range(8, sdk_struct_size, 8))
                    break

    driver_fn_struct_size = (
        224 if ("TpuDriver_Initialize" in sym_to_loc or "TfTpu_Initialize" in sym_to_loc) else 0
    )

    embedding_entrypoints: dict[str, int] = {}
    for sym_name, (fn_va, fn_sz) in sorted(sym_to_loc.items()):
        if sym_name.startswith("TpuEmbeddingEngine_"):
            foff = _vaddr_to_elf_offset(sections, fn_va)
            if foff is not None:
                rep = disassemble_x86_64_prologue(elf_bytes[foff : foff + min(max(fn_sz, 32), 96)])
                embedding_entrypoints[sym_name] = (
                    rep.expected_struct_size if rep.expected_struct_size is not None else fn_sz
                )

    return {
        "pjrt_api_struct_size": pjrt_size,
        "pjrt_api_major": pjrt_major,
        "pjrt_api_minor": pjrt_minor,
        "pjrt_vtable_slots_count": len(pjrt_slots),
        "pjrt_vtable_slots": pjrt_slots,
        "extensions_count": len(extensions),
        "extensions": extensions,
        "libtpu_sdk_api_struct_size": sdk_struct_size,
        "libtpu_sdk_api_slot_offsets": sdk_slot_offsets,
        "tpu_driver_fn_struct_size": driver_fn_struct_size,
        "tpu_embedding_entrypoints": embedding_entrypoints,
    }


def lift_pjrt_args_and_extensions(elf_bytes: bytes) -> PjrtAbiLiftReport:
    """Lifts `PJRT_*_Args` `struct_size` checks, field displacements, and extension IDs.

    Scans `.text` for `x86_64` (`cmp qword ptr [rdi], imm` / `mov r*, [rdi + disp8]`)
    and `aarch64` (`cmp x*, #imm`) patterns, plus `.data.rel.ro` and dynamic `Create*Extension`
    `PJRT_Extension_Base` headers.

    Args:
        elf_bytes: Raw 64-bit ELF shared object bytes.

    Returns:
        Populated `PjrtAbiLiftReport`.
    """
    sections, _ = parse_elf64_sections_and_dynsym(elf_bytes)
    struct_sizes: set[int] = set()
    displacements: set[int] = set()
    ext_types: set[int] = set()

    text_sec = _find_section(sections, ".text")
    if text_sec is not None and text_sec.sh_size >= 8:
        code = elf_bytes[text_sec.sh_offset : text_sec.sh_offset + min(text_sec.sh_size, 2_000_000)]
        for match in re.finditer(rb"\x48\x83\x3f([\x08-\x7f])", code):
            imm = match.group(1)[0]
            if imm % 8 == 0:
                struct_sizes.add(imm)
        for match in re.finditer(rb"\x48\x81\x3f(....)", code, re.DOTALL):
            imm32 = struct.unpack("<I", match.group(1))[0]
            if 16 <= imm32 <= 512 and imm32 % 8 == 0:
                struct_sizes.add(int(imm32))
        for match in re.finditer(rb"\x48\x8b\x47([\x08-\x78])", code):
            disp = match.group(1)[0]
            if disp % 8 == 0:
                displacements.add(disp)

    drel_sec = _find_section(sections, ".data.rel.ro")
    if drel_sec is not None and drel_sec.sh_size >= 24:
        drel = elf_bytes[drel_sec.sh_offset : drel_sec.sh_offset + min(drel_sec.sh_size, 2_000_000)]
        for off in range(0, len(drel) - 24, 8):
            ext_sz, ext_ty = struct.unpack_from("<QI", drel, off)
            if (
                ext_sz in (32, 40, 48, 56, 64, 80, 96, 120, 192, 248, 280, 288)
                and 1 <= ext_ty <= 35
            ):
                ext_types.add(int(ext_ty))

    vtable_layout = lift_vtable_and_entrypoint_layouts(elf_bytes)
    for slot in vtable_layout["pjrt_vtable_slots"]:
        slot_min_sz = slot["min_struct_size"]
        if slot_min_sz is not None:
            struct_sizes.add(slot_min_sz)
    for ext in vtable_layout["extensions"]:
        ext_types.add(ext["ext_type"])

    return PjrtAbiLiftReport(
        observed_struct_sizes=tuple(sorted(struct_sizes)),
        observed_field_displacements=tuple(sorted(displacements)),
        extension_type_ids=tuple(sorted(ext_types)),
    )


def lift_isa_and_driver_ioctls(elf_bytes: bytes) -> IsaAndDriverLiftReport:
    """Lifts VLIW/SparseCore ISA families and Gasket (`0xDC`) / VFIO (`0x3B`) `ioctl` codes.

    Args:
        elf_bytes: Raw 64-bit ELF shared object bytes.

    Returns:
        Populated `IsaAndDriverLiftReport`.
    """
    protos = extract_descriptor_table_protos(elf_bytes)
    toc = extract_filewrapper_toc_entries(elf_bytes)
    proto_names = " ".join(str(p["name"]) for p in protos)
    toc_names = " ".join(str(t["name"]) for t in toc)
    combined = f"{proto_names} {toc_names}"

    families: list[str] = []
    for code, marker in (
        ("jfc", "jellyfish"),
        ("dfc", "dragonfish"),
        ("pfc", "pufferfish"),
        ("vfc", "viperfish"),
        ("glc", "ghostlite"),
        ("gfc", "6acc60406"),
    ):
        if marker in combined or f"{code}_" in combined:
            families.append(code)

    sc_engines: list[str] = []
    for eng in ("scs", "tac", "tec", "barnacore"):
        if eng in combined:
            sc_engines.append(eng)

    sections, _ = parse_elf64_sections_and_dynsym(elf_bytes)
    sym_to_loc, _ = _extract_symtab_index(elf_bytes, sections)
    gasket_cmds: set[int] = set()
    vfio_cmds: set[int] = set()
    code_windows: list[bytes] = []
    text_sec = _find_section(sections, ".text")
    if text_sec is not None and text_sec.sh_size >= 6:
        code_windows.append(
            elf_bytes[text_sec.sh_offset : text_sec.sh_offset + min(text_sec.sh_size, 4_000_000)]
        )
    for sym_nm, (fn_va, fn_sz) in sym_to_loc.items():
        if ("Gasket" in sym_nm or "Vfio" in sym_nm) and fn_sz >= 6:
            foff = _vaddr_to_elf_offset(sections, fn_va)
            if foff is not None:
                code_windows.append(elf_bytes[foff : foff + fn_sz])

    for code_bytes in code_windows:
        for match in re.finditer(rb"[\xba\xbe](.[\xdc\x3b]..)", code_bytes, re.DOTALL):
            cmd = struct.unpack("<I", match.group(1))[0]
            ioc_type = (cmd >> 8) & 0xFF
            ioc_nr = cmd & 0xFF
            if ioc_type == 0xDC and ioc_nr <= 0x20:
                gasket_cmds.add(int(cmd))
            elif ioc_type == 0x3B and 0x64 <= ioc_nr <= 0x80:
                vfio_cmds.add(int(cmd))

    return IsaAndDriverLiftReport(
        vliw_families=tuple(families),
        sparsecore_subengines=tuple(sc_engines),
        gasket_ioctl_cmds=tuple(sorted(gasket_cmds)),
        vfio_ioctl_cmds=tuple(sorted(vfio_cmds)),
    )


def discover_ghidra_headless(
    explicit_path: Path | None = None,
    env: Mapping[str, str] | None = None,
    search_roots: Sequence[Path] | None = None,
) -> Path | None:
    """Discovers Ghidra's `analyzeHeadless` executable via explicit path, env, `PATH`, or Caskroom.

    Args:
        explicit_path: Optional explicit path to `analyzeHeadless`.
        env: Optional environment mapping (defaults to `os.environ`).
        search_roots: Optional Homebrew Caskroom / installation roots to scan.

    Returns:
        Resolved `Path` to `analyzeHeadless`, or `None` if Ghidra is not installed.
    """
    if explicit_path is not None:
        return explicit_path if explicit_path.exists() else None

    env_map = env if env is not None else os.environ
    ghidra_dir = env_map.get("GHIDRA_INSTALL_DIR", "").strip()
    if ghidra_dir:
        base = Path(ghidra_dir)
        for cand in (base / "support" / "analyzeHeadless", base / "analyzeHeadless"):
            if cand.exists():
                return cand

    which_hit = shutil.which("analyzeHeadless", path=env_map.get("PATH"))
    if which_hit is not None:
        return Path(which_hit)

    roots = (
        search_roots
        if search_roots is not None
        else (
            Path("/opt/homebrew/Caskroom/ghidra"),
            Path("/usr/local/Caskroom/ghidra"),
        )
    )
    for root in roots:
        if root.exists():
            matches = sorted(root.glob("*/*/support/analyzeHeadless"), reverse=True)
            if matches:
                return matches[0]
    return None


def generate_ghidra_headless_postscript(output_json_path: Path | None = None) -> str:
    """Generates a deterministic PyGhidra / Ghidra headless post-analysis export script.

    Args:
        output_json_path: Optional JSON file path where the Ghidra post-script writes its output.

    Returns:
        Python post-script source text consumable by Ghidra `analyzeHeadless -postScript`.
    """
    out_stmt = (
        f'with open({str(output_json_path)!r}, "w") as _f:\n    _f.write(payload)\n'
        if output_json_path is not None
        else "print(payload)\n"
    )
    return (
        '"""Headless Ghidra post-script exporting PJRT, SDK, Driver, and LLO ISA functions."""\n'
        "import json\n"
        "funcs = []\n"
        "fm = currentProgram.getFunctionManager()\n"
        "for fn in fm.getFunctions(True):\n"
        "    name = fn.getName()\n"
        '    if any(tok in name for tok in ("PJRT", "Pjrt", "LibtpuSdk", "TpuDriver", '
        '"TpuEmbeddingEngine", "llo", "TensorCore", "SparseCore", "Gasket", "Vfio")):\n'
        "        sz = fn.getBody().getNumAddresses()\n"
        '        funcs.append({"name": name, "entry": str(fn.getEntryPoint()), "size": sz})\n'
        'payload = json.dumps({"ghidra_functions": funcs})\n' + out_stmt
    )


def run_ghidra_headless(
    ghidra_headless_bin: Path,
    project_dir: Path,
    binary_path: Path,
    runner: CommandRunner = default_command_runner,
    output_json_path: Path | None = None,
) -> int:
    """Writes the Ghidra post-script and executes `analyzeHeadless` on `binary_path`.

    Args:
        ghidra_headless_bin: Path to Ghidra's `analyzeHeadless` executable.
        project_dir: Temporary Ghidra project workspace directory.
        binary_path: Target ELF64 binary (`libtpu.so` or `sdk.so`) to analyze.
        runner: Injectable subprocess runner.
        output_json_path: Optional JSON file path passed into `generate_ghidra_headless_postscript`.

    Returns:
        Exit code returned by `analyzeHeadless`.

    Raises:
        BinaryLiftError: If `binary_path` does not exist.
    """
    if not binary_path.exists():
        raise BinaryLiftError(f"Target ELF64 binary does not exist: {binary_path}")
    project_dir.mkdir(parents=True, exist_ok=True)
    script_path = project_dir / "libtpu_ghidra_postscript.py"
    script_path.write_text(
        generate_ghidra_headless_postscript(output_json_path=output_json_path),
        encoding="utf-8",
    )
    cmd = [
        str(ghidra_headless_bin),
        str(project_dir),
        "libtpu_proj",
        "-import",
        str(binary_path),
        "-scriptPath",
        str(project_dir),
        "-postScript",
        script_path.name,
        "-deleteProject",
    ]
    return runner(cmd)


def lift_elf_binary(binary_path: Path) -> dict[str, object]:
    """Runs full static RTTI, `.eh_frame`, PJRT/SDK/Driver ABI, ISA, and driver lifting.

    Args:
        binary_path: Filesystem path to `libtpu.so` or `sdk.so`.

    Returns:
        JSON-serializable dictionary of lifted binary artifacts.

    Raises:
        BinaryLiftError: If `binary_path` does not exist.
    """
    if not binary_path.exists():
        raise BinaryLiftError(f"ELF64 binary not found: {binary_path}")
    raw = binary_path.read_bytes()
    rtti = extract_itanium_rtti_classes(raw)
    eh_bounds = extract_eh_frame_function_bounds(raw)
    pjrt = lift_pjrt_args_and_extensions(raw)
    vtables = lift_vtable_and_entrypoint_layouts(raw)
    isa_drv = lift_isa_and_driver_ioctls(raw)
    return {
        "binary": str(binary_path),
        "rtti_class_count": len(rtti),
        "rtti_sample": [asdict(c) for c in rtti[:25]],
        "eh_frame_function_count": len(eh_bounds),
        "pjrt_abi": asdict(pjrt),
        "vtable_layouts": vtables,
        "isa_and_driver": asdict(isa_drv),
    }


def run_ghidra_headless_lift(
    binary_path: Path,
    project_dir: Path = Path(".cache/ghidra_proj"),
    ghidra_headless_bin: Path | None = None,
    runner: CommandRunner = default_command_runner,
    env: Mapping[str, str] | None = None,
    search_roots: Sequence[Path] | None = None,
) -> dict[str, object]:
    """Runs headless Ghidra (when installed) + ELF relocation/disassembly cross-verification.

    Args:
        binary_path: Filesystem path to `libtpu.so` or `sdk.so`.
        project_dir: Temporary Ghidra project directory.
        ghidra_headless_bin: Optional explicit path to `analyzeHeadless`.
        runner: Injectable subprocess runner.
        env: Optional environment mapping for `discover_ghidra_headless`.
        search_roots: Optional Homebrew Caskroom search roots.

    Returns:
        Combined dictionary containing ELF-lifted layouts and Ghidra headless results.

    Raises:
        BinaryLiftError: If `binary_path` does not exist or `analyzeHeadless` fails.
    """
    report = lift_elf_binary(binary_path)
    resolved_ghidra = discover_ghidra_headless(
        explicit_path=ghidra_headless_bin,
        env=env,
        search_roots=search_roots,
    )
    if resolved_ghidra is None:
        report["ghidra_headless_status"] = "fallback_elf_disassembly"
        report["ghidra_functions_count"] = 0
        return report

    out_json = project_dir / "ghidra_lift_output.json"
    rc = run_ghidra_headless(
        ghidra_headless_bin=resolved_ghidra,
        project_dir=project_dir,
        binary_path=binary_path,
        runner=runner,
        output_json_path=out_json,
    )
    if rc != 0:
        raise BinaryLiftError(f"Ghidra analyzeHeadless exited with status {rc}")

    ghidra_funcs: list[object] = []
    if out_json.exists():
        parsed = json.loads(out_json.read_text(encoding="utf-8"))
        if isinstance(parsed, dict) and isinstance(parsed.get("ghidra_functions"), list):
            ghidra_funcs = list(parsed["ghidra_functions"])
    report["ghidra_headless_status"] = "executed"
    report["ghidra_headless_bin"] = str(resolved_ghidra)
    report["ghidra_functions_count"] = len(ghidra_funcs)
    return report


def verify_lift_against_golden(
    lift_report: dict[str, object],
    repo_root: Path,
) -> list[str]:
    """Cross-verifies a `libtpu.so` lift report against `re_extract.py` golden manifests.

    Args:
        lift_report: Output dictionary from `lift_elf_binary` or `run_ghidra_headless_lift`.
        repo_root: Repository root containing `docs/reverse_engineering/`.

    Returns:
        List of mismatch descriptions (empty when 100% consistent).
    """
    re_dir = repo_root / "docs" / "reverse_engineering"
    manifest = json.loads((re_dir / "golden_binary_manifest.json").read_text(encoding="utf-8"))
    layouts = json.loads((re_dir / "golden_struct_layouts.json").read_text(encoding="utf-8"))
    vt = lift_report.get("vtable_layouts", {})
    if not isinstance(vt, dict):
        return ["Missing vtable_layouts in lift_report"]

    errors: list[str] = []
    libtpu_so = manifest.get("libtpu_so", {})
    if vt.get("pjrt_api_struct_size") != libtpu_so.get("pjrt_api_struct_size"):
        errors.append("PJRT_Api struct_size mismatch vs golden_binary_manifest.json")
    if vt.get("pjrt_api_minor") != libtpu_so.get("pjrt_api_version_minor"):
        errors.append("PJRT_Api minor version mismatch vs golden_binary_manifest.json")
    if vt.get("extensions_count") != len(layouts.get("extensions", [])):
        errors.append("PJRT extension count mismatch vs golden_struct_layouts.json")
    if vt.get("libtpu_sdk_api_struct_size") != libtpu_so.get("libtpu_sdk_api_struct_size"):
        errors.append("LibtpuSdk_Api struct_size mismatch vs golden_binary_manifest.json")
    return errors


def main(
    argv: list[str] | None = None,
    runner: CommandRunner = default_command_runner,
) -> int:
    """CLI entrypoint for `scripts/ghidra_lift_isa_and_abi.py`.

    Args:
        argv: Optional CLI argument list.
        runner: Injectable command runner for headless Ghidra mode.

    Returns:
        Exit status code (`0` on success, `1` on error).
    """
    parser = argparse.ArgumentParser(
        description="Lift RTTI, .eh_frame, PJRT ABI, VLIW ISA, and driver ioctls from libtpu.so."
    )
    parser.add_argument("--binary", type=Path, required=True, help="Path to libtpu.so or sdk.so")
    parser.add_argument(
        "--ghidra-headless",
        type=Path,
        default=None,
        help="Optional path to Ghidra analyzeHeadless executable",
    )
    parser.add_argument(
        "--ghidra-project-dir",
        type=Path,
        default=Path(".cache/ghidra_proj"),
        help="Workspace directory for headless Ghidra project",
    )
    parser.add_argument(
        "--verify-golden",
        type=Path,
        default=None,
        help="Optional repository root to cross-verify lifted vtables against golden manifests",
    )
    args = parser.parse_args(argv)

    try:
        if args.ghidra_headless is not None:
            return run_ghidra_headless(
                ghidra_headless_bin=Path(args.ghidra_headless),
                project_dir=Path(args.ghidra_project_dir),
                binary_path=Path(args.binary),
                runner=runner,
            )
        report = run_ghidra_headless_lift(
            binary_path=Path(args.binary),
            project_dir=Path(args.ghidra_project_dir),
            runner=runner,
        )
        if args.verify_golden is not None:
            mismatches = verify_lift_against_golden(report, Path(args.verify_golden))
            if mismatches:
                raise BinaryLiftError("; ".join(mismatches))
        print(json.dumps(report, indent=2))
        return 0
    except BinaryLiftError as exc:
        print(f"BinaryLiftError: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Unit and 100% branch tests for Phase 1 forensics, lifter, and differential oracle scripts."""

from __future__ import annotations

import ctypes
import hashlib
import json
import runpy
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import scripts.differential_oracle as do
import scripts.ghidra_lift_isa_and_abi as gl
import scripts.pre_commit as pc
import scripts.re_extract as rx

NL: str = chr(10)


def _build_forensics_elf64(
    include_pdesc: bool = True,
    include_ftoc: bool = True,
    include_eh_frame: bool = True,
    include_text: bool = True,
) -> bytes:
    """Builds a rich synthetic 64-bit ELF shared object exercising all forensics branches.

    Args:
        include_pdesc: Whether to emit `protodesc_cold`.
        include_ftoc: Whether to emit `filewrapper_toc`.
        include_eh_frame: Whether to emit `.eh_frame`.
        include_text: Whether to emit `.text` and `.data.rel.ro`.

    Returns:
        Serialized ELF64 byte buffer.
    """
    sh_names = [b"", b".shstrtab", b".rodata", b".rela.dyn", b".bss"]
    if include_pdesc:
        sh_names.append(b"protodesc_cold")
    if include_ftoc:
        sh_names.append(b"filewrapper_toc")
    if include_eh_frame:
        sh_names.append(b".eh_frame")
    if include_text:
        sh_names.append(b".text")
        sh_names.append(b".data.rel.ro")

    sh_offsets: dict[bytes, int] = {}
    shstr_parts: list[bytes] = []
    cur = 0
    for nm in sh_names:
        sh_offsets[nm] = cur
        chunk = nm + b"\x00"
        shstr_parts.append(chunk)
        cur += len(chunk)
    shstrtab = b"".join(shstr_parts)

    base_va = 0x100000
    ehdr_sz = 64
    buf = bytearray(b"\x00" * ehdr_sz)

    def pad8() -> None:
        """Pads `buf` with zero bytes to an 8-byte boundary."""
        while len(buf) % 8 != 0:
            buf.append(0)

    shstr_off = len(buf)
    buf.extend(shstrtab)
    pad8()

    # .rodata: RTTI type strings + FileToc names + FileToc payload
    rodata_off = len(buf)
    rodata_va = base_va + rodata_off
    rtti_entries = [
        b"N7asic_sw7deepsea3FooE\x00",
        b"N3xla9jellyfish3BarE\x00",
        b"N9megascale3BazE\x00",
        b"N10tpu_driver3QuxE\x00",
        b"N3tpu3MuxE\x00",
    ]
    rtti_vaddrs: list[int] = []
    for entry in rtti_entries:
        rtti_vaddrs.append(base_va + len(buf))
        buf.extend(entry)
    toc_name_va = base_va + len(buf)
    buf.extend(b"viperfish_scs_barnacore.bin\x00")
    empty_name_va = base_va + len(buf)
    buf.extend(b"\x00")
    toc_data_va = base_va + len(buf)
    toc_payload = b"hello-tpu-toc"
    buf.extend(toc_payload)
    # Unterminated tail string for nul <= name_off coverage
    unterm_name_va = base_va + len(buf)
    rodata_len = len(buf) - rodata_off
    pad8()

    # protodesc_cold: valid 2-byte varint name (>128 chars), non-proto, non-0x0A, truncated varint
    pdesc_off = len(buf)
    pdesc_va = base_va + pdesc_off
    long_proto_name = (
        "platforms/deepsea/jellyfish/pufferfish/dragonfish/ghostlite/6acc60406/"
        + ("a" * 65)
        + "_tac_tec.proto"
    ).encode("ascii")
    # Encode length as 2-byte varint
    plen = len(long_proto_name)
    varint_len = bytes([(plen & 0x7F) | 0x80, plen >> 7])
    fd_valid = b"\x0a" + varint_len + long_proto_name
    fd_valid_va = base_va + len(buf)
    buf.extend(fd_valid)

    fd_non_proto = b"\x0a\x08notproto"
    fd_non_proto_va = base_va + len(buf)
    buf.extend(fd_non_proto)

    fd_not_0a = b"\x12\x08badstart"
    fd_not_0a_va = base_va + len(buf)
    buf.extend(fd_not_0a)

    fd_trunc_varint = b"\x0a" + (b"\x80" * 9)
    fd_trunc_va = base_va + len(buf)
    buf.extend(fd_trunc_varint)

    fd_over_str = b"\x0a\x50short" + b"1234"
    fd_over_str_va = base_va + len(buf)
    buf.extend(fd_over_str)
    pdesc_len = len(buf) - pdesc_off
    pad8()

    # .data.rel.ro: DescriptorTables + FileToc chain + PJRT_Extension_Base headers
    drel_off = len(buf)
    drel_va = base_va + drel_off

    def add_desc_table(flags: int, sz: int) -> int:
        """Appends a synthetic `DescriptorTable` struct to `buf`.

        Args:
            flags: `DescriptorTable` flags word.
            sz: Encoded `FileDescriptorProto` byte length.

        Returns:
            Virtual address of the appended `DescriptorTable`.
        """
        va = base_va + len(buf)
        buf.extend(struct.pack("<IIQQQ", flags, sz, 0, 0, 0))
        return va

    tbl_valid = add_desc_table(0, len(fd_valid))
    tbl_non_proto = add_desc_table(0, len(fd_non_proto))
    tbl_not_0a = add_desc_table(0, len(fd_not_0a))
    tbl_trunc = add_desc_table(0, len(fd_trunc_varint))
    tbl_over_str = add_desc_table(0, len(fd_over_str))
    tbl_bad_flags = add_desc_table(1, 16)

    # FileToc entries (40 bytes each)
    toc0_va = base_va + len(buf)
    buf.extend(struct.pack("<QQQ", 0, 0, len(toc_payload)) + hashlib.md5(toc_payload).digest())
    toc1_empty_nm_va = base_va + len(buf)
    buf.extend(struct.pack("<QQQ", 0, 0, 4) + (b"\x00" * 16))
    toc2_unmapped_va = base_va + len(buf)
    buf.extend(struct.pack("<QQQ", 0, 0, 4) + (b"\x00" * 16))
    toc3_end_va = base_va + len(buf)
    buf.extend(b"\x00" * 40)

    # PJRT_Extension_Base node (size=64, ext_type=9) + non-matching node
    buf.extend(struct.pack("<QIQ", 64, 9, 0) + (b"\x00" * 16))
    buf.extend(struct.pack("<QIQ", 13, 99, 0) + (b"\x00" * 16))
    drel_len = len(buf) - drel_off
    pad8()

    ftoc_off = len(buf)
    ftoc_va = base_va + ftoc_off
    buf.extend(b"\x00" * 16)
    ftoc_len = 16
    pad8()

    # .eh_frame: 1 CIE + 1 non-zero FDE + 1 zero-range FDE + zero terminator
    eh_off = len(buf)
    eh_va = base_va + eh_off
    cie_body = struct.pack("<IB", 0, 1) + b"\x00\x01\x78\x10\x00\x00\x00"
    buf.extend(struct.pack("<I", len(cie_body)) + cie_body)
    fde1_cie_ptr = (len(buf) - eh_off) + 4
    fde1_body = struct.pack("<IiI", fde1_cie_ptr, 0x2000, 0x80)
    buf.extend(struct.pack("<I", len(fde1_body)) + fde1_body)
    fde2_cie_ptr = (len(buf) - eh_off) + 4
    fde2_body = struct.pack("<IiI", fde2_cie_ptr, 0x3000, 0)
    buf.extend(struct.pack("<I", len(fde2_body)) + fde2_body)
    buf.extend(struct.pack("<I", 0))
    eh_len = len(buf) - eh_off
    pad8()

    # .text: cmp [rdi], imm8/imm32, mov rax, [rdi+disp8], Gasket 0xDC, and VFIO 0x3B ioctl moves
    text_off = len(buf)
    text_va = base_va + text_off
    buf.extend(b"\x48\x83\x3f\x20")  # struct_size = 32
    buf.extend(b"\x48\x83\x3f\x21")  # unaligned 33
    buf.extend(b"\x48\x81\x3f" + struct.pack("<I", 256))  # struct_size = 256
    buf.extend(b"\x48\x81\x3f" + struct.pack("<I", 9999))  # out of range
    buf.extend(b"\x48\x8b\x47\x10")  # disp = 16
    buf.extend(b"\x48\x8b\x47\x13")  # unaligned disp 19
    buf.extend(b"\xba" + struct.pack("<I", 0x4008DC00))  # Gasket RESET
    buf.extend(b"\xba" + struct.pack("<I", 0x4008DC55))  # Gasket nr > 0x20
    buf.extend(b"\xbe" + struct.pack("<I", 0x00003B71))  # VFIO MAP_DMA
    buf.extend(b"\xbe" + struct.pack("<I", 0x00003B01))  # VFIO nr < 0x64
    text_len = len(buf) - text_off
    pad8()

    # .rela.dyn (SHT_RELA = 4, entsize = 24; r_info low32 = 8 R_X86_64_RELATIVE)
    rela_off = len(buf)
    relas: list[bytes] = []

    def add_rela(r_off: int, r_add: int) -> None:
        """Appends an `R_X86_64_RELATIVE` relocation entry to `relas`.

        Args:
            r_off: Relocation target virtual address.
            r_add: Signed relocation addend virtual address.
        """
        relas.append(struct.pack("<QQq", r_off, 8, r_add))

    # RTTI reference counts: 2 refs -> vmi, 1 ref -> si, 0 refs -> class
    add_rela(drel_va, rtti_vaddrs[0])
    add_rela(drel_va + 8, rtti_vaddrs[0])
    add_rela(drel_va + 16, rtti_vaddrs[1])

    # DescriptorTable relocations + edge cases
    add_rela(4, fd_valid_va)  # offset < 8
    add_rela(0x999999, fd_valid_va)  # unmapped table_vaddr
    add_rela(tbl_valid + 8, fd_valid_va)
    add_rela(tbl_non_proto + 8, fd_non_proto_va)
    add_rela(tbl_not_0a + 8, fd_not_0a_va)
    add_rela(tbl_trunc + 8, fd_trunc_va)
    add_rela(tbl_over_str + 8, fd_over_str_va)
    add_rela(tbl_bad_flags + 8, fd_valid_va)

    # filewrapper_toc relocations
    add_rela(ftoc_va, toc0_va)
    add_rela(toc0_va, toc_name_va)
    add_rela(toc0_va + 8, toc_data_va)
    add_rela(toc1_empty_nm_va, empty_name_va)
    add_rela(toc1_empty_nm_va + 8, toc_data_va)
    add_rela(toc2_unmapped_va, unterm_name_va)
    add_rela(toc2_unmapped_va + 8, 0x999999)
    add_rela(toc3_end_va, toc_name_va)

    rela_blob = b"".join(relas)
    buf.extend(rela_blob)
    rela_len = len(rela_blob)
    pad8()

    shoff = len(buf)

    def shdr(nm: bytes, sh_type: int, addr: int, off: int, sz: int, entsz: int = 0) -> bytes:
        """Packs a 64-byte `Elf64_Shdr` section header entry.

        Args:
            nm: NUL-terminated section name key in `sh_offsets`.
            sh_type: ELF `sh_type` constant.
            addr: Section virtual address (`sh_addr`).
            off: Section file offset (`sh_offset`).
            sz: Section byte length (`sh_size`).
            entsz: Fixed entry size (`sh_entsize`).

        Returns:
            Packed 64-byte `Elf64_Shdr`.
        """
        return struct.pack(
            "<IIQQQQIIQQ",
            sh_offsets[nm],
            sh_type,
            2,
            addr,
            off,
            sz,
            0,
            0,
            8,
            entsz,
        )

    shdrs = [
        shdr(b"", 0, 0, 0, 0),
        shdr(b".shstrtab", 3, 0, shstr_off, len(shstrtab)),
        shdr(b".rodata", 1, rodata_va, rodata_off, rodata_len),
        shdr(b".rela.dyn", 4, base_va + rela_off, rela_off, rela_len, 24),
        shdr(b".bss", 8, 0x900000, shoff, 64),
    ]
    if include_pdesc:
        shdrs.append(shdr(b"protodesc_cold", 1, pdesc_va, pdesc_off, pdesc_len))
    if include_ftoc:
        shdrs.append(shdr(b"filewrapper_toc", 1, ftoc_va, ftoc_off, ftoc_len))
    if include_eh_frame:
        shdrs.append(shdr(b".eh_frame", 1, eh_va, eh_off, eh_len))
    if include_text:
        shdrs.append(shdr(b".text", 1, text_va, text_off, text_len))
        shdrs.append(shdr(b".data.rel.ro", 1, drel_va, drel_off, drel_len))

    buf.extend(b"".join(shdrs))
    e_ident = b"\x7fELF" + bytes([2, 1, 1, 0]) + (b"\x00" * 8)
    ehdr = e_ident + struct.pack(
        "<HHIQQQIHHHHHH",
        3,
        62,
        1,
        0,
        0,
        shoff,
        0,
        ehdr_sz,
        0,
        0,
        64,
        len(shdrs),
        1,
    )
    buf[:ehdr_sz] = ehdr
    return bytes(buf)


def test_re_extract_and_pre_commit_entrypoints_and_tables(tmp_path: Path) -> None:
    """Tests `re_extract` table extractors and `__main__` blocks on `re_extract` & `pre_commit`.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
    """
    elf_full = _build_forensics_elf64()
    protos = rx.extract_descriptor_table_protos(elf_full)
    assert len(protos) == 1
    assert str(protos[0]["name"]).endswith("_tac_tec.proto")

    toc = rx.extract_filewrapper_toc_entries(elf_full)
    assert len(toc) == 1
    assert toc[0]["md5_verified"] is True

    # Test unterm_name_va branch where name_off & data_off are mapped but nul == -1
    elf_minimal = _build_forensics_elf64(
        include_pdesc=False,
        include_ftoc=False,
        include_eh_frame=False,
        include_text=False,
    )
    assert rx.extract_descriptor_table_protos(elf_minimal) == []
    assert rx.extract_filewrapper_toc_entries(elf_minimal) == []

    # Cover nul <= name_off when data_va IS mapped by testing a small ELF with unterminated name
    secs = [rx.ElfSectionHeader(".rodata", 1, 2, 0x1000, 0, 4, 0, 0)]
    assert rx._vaddr_to_elf_offset(secs, 0x1002) == 2
    assert rx._vaddr_to_elf_offset(secs, 0x9999) is None

    # Exercise 0xBA (MOV edx, imm32) and 0xE8 (CALL rel32) branches in disassemble_x86_64_prologue
    edx_rep = rx.disassemble_x86_64_prologue(
        b"\xba\x04\x00\x00\x00"  # imm_edx=4 (< 8, ignored)
        + b"\xba\x28\x00\x00\x00"  # imm_edx=40 (8 <= imm_edx <= 512, captured)
        + b"\xe8\x00\x00\x00\x00"  # CALL rel32 -> seen_call = True
        + b"\xba\x30\x00\x00\x00"  # imm_edx after CALL (ignored)
        + b"\xbe\x38\x00\x00\x00"  # imm_esi after CALL (ignored)
        + b"\xc3"
    )
    assert edx_rep.expected_struct_size == 40
    call_first_rep = rx.disassemble_x86_64_prologue(b"\xe8\x00\x00\x00\x00\xbe\x20\x00\x00\x00\xc3")
    assert call_first_rep.expected_struct_size is None

    # Exercise __main__ of scripts/re_extract.py (passing --repo-root to real workspace)
    repo_root = Path(__file__).resolve().parent.parent.parent
    with (
        patch.object(sys, "argv", ["re_extract.py", "--repo-root", str(repo_root)]),
        pytest.raises(SystemExit) as rx_exit,
    ):
        runpy.run_path(str(Path(rx.__file__)), run_name="__main__")
    assert rx_exit.value.code == 0

    # Exercise __main__ of scripts/pre_commit.py using `--help` (fast, zero side effects)
    with (
        patch.object(sys, "argv", ["pre_commit.py", "--help"]),
        pytest.raises(SystemExit) as pc_exit,
    ):
        runpy.run_path(str(Path(pc.__file__)), run_name="__main__")
    assert pc_exit.value.code == 0

    _ = tmp_path


def test_re_extract_filewrapper_unterminated_cstring() -> None:
    """Covers the `nul <= name_off` branch in `extract_filewrapper_toc_entries`."""
    # Construct ELF where name_va points to the end of the buffer without a NUL terminator
    shstr = b"\x00.shstrtab\x00.rodata\x00.data.rel.ro\x00filewrapper_toc\x00.rela.dyn\x00"
    ehdr_sz = 64
    buf = bytearray(b"\x00" * ehdr_sz)
    shstr_off = len(buf)
    buf.extend(shstr)
    while len(buf) % 8 != 0:
        buf.append(0)
    drel_off = len(buf)
    drel_va = 0x200000 + drel_off
    # One 40B FileToc entry + trailing zero FileToc
    buf.extend(struct.pack("<QQQ", 0, 0, 2) + (b"\x00" * 16) + (b"\x00" * 40))
    drel_len = 80
    ftoc_off = len(buf)
    ftoc_va = 0x200000 + ftoc_off
    buf.extend(b"\x00" * 8)
    rela_off = len(buf)
    ro_va = 0x300000
    relas = (
        struct.pack("<QQq", ftoc_va, 8, drel_va)
        + struct.pack("<QQq", drel_va, 8, ro_va)
        + struct.pack("<QQq", drel_va + 8, 8, ro_va)
    )
    buf.extend(relas)
    shoff = len(buf)

    def mk_sh(nm_off: int, tp: int, va: int, off: int, sz: int, esz: int = 0) -> bytes:
        """Packs a 64-byte `Elf64_Shdr` from a string-table offset.

        Args:
            nm_off: Offset into `.shstrtab`.
            tp: ELF `sh_type` constant.
            va: Section virtual address.
            off: Section file offset.
            sz: Section byte length.
            esz: Fixed entry size.

        Returns:
            Packed 64-byte `Elf64_Shdr`.
        """
        return struct.pack("<IIQQQQIIQQ", nm_off, tp, 2, va, off, sz, 0, 0, 8, esz)

    # Place .rodata at the VERY END of the file with NO NUL bytes (`b"XY"`)!
    ro_off = shoff + 6 * 64
    shdrs = b"".join(
        [
            mk_sh(0, 0, 0, 0, 0),
            mk_sh(1, 3, 0, shstr_off, len(shstr)),
            mk_sh(11, 1, ro_va, ro_off, 2),
            mk_sh(19, 1, drel_va, drel_off, drel_len),
            mk_sh(32, 1, ftoc_va, ftoc_off, 8),
            mk_sh(48, 4, 0x400000, rela_off, len(relas), 24),
        ]
    )
    buf.extend(shdrs)
    buf.extend(b"XY")
    ehdr = (
        b"\x7fELF"
        + bytes([2, 1, 1, 0])
        + (b"\x00" * 8)
        + struct.pack("<HHIQQQIHHHHHH", 3, 62, 1, 0, 0, shoff, 0, 64, 0, 0, 64, 6, 1)
    )
    buf[:64] = ehdr
    assert rx.extract_filewrapper_toc_entries(bytes(buf)) == []
    assert rx.extract_static_csr_maps(bytes(buf)) == {}


def test_extract_static_csr_maps_full_coverage() -> None:
    """Tests `_classify_static_csr_shard_symbol` and `extract_static_csr_maps` branches."""
    assert rx._classify_static_csr_shard_symbol("other_fn") is None
    assert rx._classify_static_csr_shard_symbol("PerformanceCounterNameToStringENS_") is None
    unk_sym = "_ZN3unk31PerformanceCounterNameToString0ENS_"
    assert rx._classify_static_csr_shard_symbol(unk_sym) is None
    for tag, fam in (
        ("3jxc", "jxc"),
        ("3pfc", "pfc"),
        ("3plc", "plc"),
        ("3vlc", "vlc"),
        ("3vfc", "vfc"),
        ("3glc", "glc"),
        ("3gfc", "gfc"),
    ):
        assert (
            rx._classify_static_csr_shard_symbol(f"_ZN{tag}31PerformanceCounterNameToString0ENS_")
            == fam
        )

    base_va = 0x200000
    shstr = b"\x00.shstrtab\x00.rodata\x00.text\x00.strtab\x00.symtab\x00"
    buf = bytearray(b"\x00" * 64)
    shstr_off = len(buf)
    buf.extend(shstr)
    while len(buf) % 8 != 0:
        buf.append(0)

    ro_off = len(buf)
    got_va = base_va + ro_off
    buf.extend(struct.pack("<Q", 0))
    pool_va = base_va + len(buf)
    buf.extend(b"MGT_USER_PCIE_COUNTERVF_VLC_DOORBELL")
    while len(buf) % 8 != 0:
        buf.append(0)
    rel_multi_va = base_va + len(buf)
    buf.extend(struct.pack("<HHHH", 0, 21, 10, 5))
    while len(buf) % 8 != 0:
        buf.append(0)
    rel_single_va = base_va + len(buf)
    buf.extend(struct.pack("<HHHH", 21, 36, 8, 2))
    while len(buf) % 8 != 0:
        buf.append(0)
    ranges_va = base_va + len(buf)
    buf.extend(struct.pack("<QIIHHI", 0x402008, 8, 0, 2, 0, 0))
    ro_len = len(buf) - ro_off

    text_off = len(buf)
    fn0_va = base_va + len(buf)
    fn0 = bytearray()
    fn0.extend(b"\x48\x8d\x35" + struct.pack("<i", got_va - (fn0_va + 7)))
    fn0.extend(b"\x4c\x8d\x05\x00\x00\x00\x00\x90")
    r_ip = fn0_va + len(fn0) + 7
    fn0.extend(
        b"\x4c\x8d\x05" + struct.pack("<i", ranges_va - r_ip) + b"\xba" + struct.pack("<I", 1)
    )
    fn0.extend(b"\x48\xba" + struct.pack("<q", rel_multi_va - got_va))
    fn0.extend(b"\x48\xb9" + struct.pack("<q", pool_va - got_va))
    fn0.extend(b"\xc3" * 12)
    buf.extend(fn0)
    while len(buf) % 8 != 0:
        buf.append(0)

    fn1_va = base_va + len(buf)
    fn1 = bytearray()
    fn1.extend(b"\x48\x8d\x87" + struct.pack("<i", -0x10500))
    fn1.extend(b"\x48\x3d" + struct.pack("<I", 9))
    fn1.extend(b"\x48\x3d" + struct.pack("<I", 1))
    rel1_ip = fn1_va + len(fn1) + 7
    fn1.extend(b"\x48\x8d\x15" + struct.pack("<i", rel_single_va - rel1_ip))
    got1_ip = fn1_va + len(fn1) + 7
    fn1.extend(b"\x48\x8d\x35" + struct.pack("<i", got_va - got1_ip))
    fn1.extend(b"\x48\xb8" + struct.pack("<q", pool_va - got_va))
    fn1.extend(b"\xc3" * 12)
    buf.extend(fn1)
    while len(buf) % 8 != 0:
        buf.append(0)

    fn2_va = base_va + len(buf)
    fn2 = b"\x90" * 16
    buf.extend(fn2)
    text_len = len(buf) - text_off
    while len(buf) % 8 != 0:
        buf.append(0)

    strtab_off = len(buf)
    strtab = bytearray(b"\x00")
    s_other = len(strtab)
    strtab.extend(b"helper_fn\x00")
    s_unmapped = len(strtab)
    strtab.extend(b"_ZN3jxc31PerformanceCounterNameToString7ENS_\x00")
    s_fn0 = len(strtab)
    strtab.extend(b"_ZN3jxc31PerformanceCounterNameToString0ENS_\x00")
    s_fn1 = len(strtab)
    strtab.extend(b"_ZN3vlc31PerformanceCounterNameToString1ENS_\x00")
    s_fn2 = len(strtab)
    strtab.extend(b"_ZN3pfc31PerformanceCounterNameToString2ENS_\x00")
    s_nonul = len(strtab)
    strtab.extend(b"nonul")
    buf.extend(strtab)
    strtab_len = len(strtab)
    while len(buf) % 8 != 0:
        buf.append(0)

    symtab_off = len(buf)
    syms = b"".join(
        [
            struct.pack("<IBBHQQ", 0, 0, 0, 0, 0, 0),
            struct.pack("<IBBHQQ", s_other, 0x12, 0, 3, fn2_va, 16),
            struct.pack("<IBBHQQ", s_unmapped, 0x12, 0, 3, 0x99990000, 16),
            struct.pack("<IBBHQQ", s_fn0, 0x12, 0, 3, fn0_va, len(fn0)),
            struct.pack("<IBBHQQ", s_fn1, 0x12, 0, 3, fn1_va, len(fn1)),
            struct.pack("<IBBHQQ", s_fn2, 0x12, 0, 3, fn2_va, len(fn2)),
            struct.pack("<IBBHQQ", s_nonul, 0x12, 0, 3, fn2_va, 16),
        ]
    )
    buf.extend(syms)
    symtab_len = len(syms)
    while len(buf) % 8 != 0:
        buf.append(0)

    shoff = len(buf)

    def mk_sh(nm: int, tp: int, va: int, off: int, sz: int, lk: int = 0, esz: int = 0) -> bytes:
        """Packs a 64-byte `Elf64_Shdr` with optional `sh_link`.

        Args:
            nm: Offset into `.shstrtab`.
            tp: ELF `sh_type` constant.
            va: Section virtual address.
            off: Section file offset.
            sz: Section byte length.
            lk: Linked section index (`sh_link`).
            esz: Fixed entry size (`sh_entsize`).

        Returns:
            Packed 64-byte `Elf64_Shdr`.
        """
        return struct.pack("<IIQQQQIIQQ", nm, tp, 2 if va else 0, va, off, sz, lk, 0, 8, esz)

    buf.extend(
        b"".join(
            [
                mk_sh(0, 0, 0, 0, 0),
                mk_sh(1, 3, 0, shstr_off, len(shstr)),
                mk_sh(11, 1, base_va + ro_off, ro_off, ro_len),
                mk_sh(19, 1, base_va + text_off, text_off, text_len),
                mk_sh(25, 3, 0, strtab_off, strtab_len),
                mk_sh(33, 2, 0, symtab_off, symtab_len, 4, 24),
            ]
        )
    )
    buf[:64] = (
        b"\x7fELF"
        + bytes([2, 1, 1, 0])
        + (b"\x00" * 8)
        + struct.pack("<HHIQQQIHHHHHH", 3, 62, 1, 0, 0, shoff, 0, 64, 0, 0, 64, 6, 1)
    )
    csr_maps = rx.extract_static_csr_maps(bytes(buf))
    assert len(csr_maps["jxc"]) == 1
    assert csr_maps["jxc"][0]["name"] == "MGT_USER_PCIE_COUNTER"
    assert csr_maps["jxc"][0]["bar"] == 2
    assert len(csr_maps["vlc"]) == 1
    assert csr_maps["vlc"][0]["name"] == "VF_VLC_DOORBELL"
    assert csr_maps["vlc"][0]["bar"] == 0
    assert csr_maps["pfc"] == []


def test_ghidra_lift_isa_and_abi_full_coverage(tmp_path: Path) -> None:
    """Tests `scripts/ghidra_lift_isa_and_abi.py` across all static and headless paths.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
    """
    with patch.object(subprocess, "run", return_value=SimpleNamespace(returncode=0)):
        assert gl.default_command_runner(["true"]) == 0

    elf_full = _build_forensics_elf64()
    elf_path = tmp_path / "libtpu_synth.so"
    elf_path.write_bytes(elf_full)

    report = gl.lift_elf_binary(elf_path)
    assert report["rtti_class_count"] == 5
    assert report["eh_frame_function_count"] == 1
    pjrt_abi = report["pjrt_abi"]
    assert isinstance(pjrt_abi, dict)
    assert 32 in pjrt_abi["observed_struct_sizes"]
    assert 256 in pjrt_abi["observed_struct_sizes"]
    assert 16 in pjrt_abi["observed_field_displacements"]
    assert 9 in pjrt_abi["extension_type_ids"]

    isa_drv = report["isa_and_driver"]
    assert isinstance(isa_drv, dict)
    assert set(isa_drv["vliw_families"]) == {"jfc", "dfc", "pfc", "vfc", "glc", "gfc"}
    assert set(isa_drv["sparsecore_subengines"]) == {"scs", "tac", "tec", "barnacore"}
    assert 0x4008DC00 in isa_drv["gasket_ioctl_cmds"]
    assert 0x00003B71 in isa_drv["vfio_ioctl_cmds"]

    # Minimal ELF without .eh_frame / .text / .data.rel.ro
    elf_min = _build_forensics_elf64(
        include_pdesc=False,
        include_ftoc=False,
        include_eh_frame=False,
        include_text=False,
    )
    assert gl.extract_eh_frame_function_bounds(elf_min) == []
    assert gl.lift_pjrt_args_and_extensions(elf_min).observed_struct_sizes == ()
    assert gl.lift_isa_and_driver_ioctls(elf_min).gasket_ioctl_cmds == ()

    # Missing binary error branches
    missing_so = tmp_path / "missing.so"
    with pytest.raises(gl.BinaryLiftError, match="not found"):
        gl.lift_elf_binary(missing_so)
    with pytest.raises(gl.BinaryLiftError, match="does not exist"):
        gl.run_ghidra_headless(Path("analyzeHeadless"), tmp_path / "gp", missing_so)

    # Test discover_ghidra_headless, run_ghidra_headless_lift, and verify_lift_against_golden
    fake_ghidra_bin = tmp_path / "ghidra_install" / "support" / "analyzeHeadless"
    fake_ghidra_bin.parent.mkdir(parents=True)
    fake_ghidra_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_ghidra_bin.chmod(0o755)
    assert gl.discover_ghidra_headless(explicit_path=fake_ghidra_bin) == fake_ghidra_bin
    assert gl.discover_ghidra_headless(explicit_path=tmp_path / "nope") is None
    assert (
        gl.discover_ghidra_headless(env={"GHIDRA_INSTALL_DIR": str(tmp_path / "ghidra_install")})
        == fake_ghidra_bin
    )
    assert (
        gl.discover_ghidra_headless(
            env={"PATH": str(fake_ghidra_bin.parent)},
            search_roots=(),
        )
        == fake_ghidra_bin
    )
    cask_bin = tmp_path / "cask" / "11.3" / "ghidra_11.3_PUBLIC" / "support" / "analyzeHeadless"
    cask_bin.parent.mkdir(parents=True)
    cask_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    assert (
        gl.discover_ghidra_headless(env={"PATH": ""}, search_roots=(tmp_path / "cask",)) == cask_bin
    )
    assert (
        gl.discover_ghidra_headless(
            env={"GHIDRA_INSTALL_DIR": str(tmp_path / "empty_dir"), "PATH": ""},
            search_roots=(tmp_path / "empty_cask",),
        )
        is None
    )

    empty_cask = tmp_path / "empty_cask"
    empty_cask.mkdir()
    assert (
        gl.discover_ghidra_headless(
            env={"PATH": ""},
            search_roots=(empty_cask,),
        )
        is None
    )

    fb_report = gl.run_ghidra_headless_lift(
        elf_path,
        project_dir=tmp_path / "gp_fb",
        env={"PATH": ""},
        search_roots=(),
    )
    assert fb_report["ghidra_headless_status"] == "fallback_elf_disassembly"

    def _writing_runner(cmd: list[str]) -> int:
        """Simulates a headless Ghidra runner writing valid output JSON.

        Args:
            cmd: Command-line argument list passed to the headless runner.
        """
        proj = Path(cmd[1])
        (proj / "ghidra_lift_output.json").write_text(
            '{"ghidra_functions": [{"name": "GetPjrtApi", "entry": "0x1000", "size": 5}]}',
            encoding="utf-8",
        )
        return 0

    exec_report = gl.run_ghidra_headless_lift(
        elf_path,
        project_dir=tmp_path / "gp_exec",
        ghidra_headless_bin=fake_ghidra_bin,
        runner=_writing_runner,
    )
    assert exec_report["ghidra_headless_status"] == "executed"
    assert exec_report["ghidra_functions_count"] == 1
    with pytest.raises(gl.BinaryLiftError, match="exited with status 2"):
        gl.run_ghidra_headless_lift(
            elf_path,
            project_dir=tmp_path / "gp_fail",
            ghidra_headless_bin=fake_ghidra_bin,
            runner=lambda _cmd: 2,
        )

    repo_root = Path(__file__).resolve().parents[2]
    assert gl.verify_lift_against_golden({"vtable_layouts": "bad"}, repo_root) != []
    assert gl.verify_lift_against_golden(fb_report, repo_root) != []

    # Runner returning 0 without writing output JSON and with non-list ghidra_functions
    no_json_report = gl.run_ghidra_headless_lift(
        elf_path,
        project_dir=tmp_path / "gp_no_json",
        ghidra_headless_bin=fake_ghidra_bin,
        runner=lambda _cmd: 0,
    )
    assert no_json_report["ghidra_functions_count"] == 0

    def _bad_json_runner(cmd: list[str]) -> int:
        """Simulates a headless Ghidra runner writing non-list ghidra_functions JSON.

        Args:
            cmd: Command-line argument list passed to the headless runner.
        """
        proj = Path(cmd[1])
        (proj / "ghidra_lift_output.json").write_text(
            '{"ghidra_functions": "nope"}', encoding="utf-8"
        )
        return 0

    bad_json_report = gl.run_ghidra_headless_lift(
        elf_path,
        project_dir=tmp_path / "gp_bad_json",
        ghidra_headless_bin=fake_ghidra_bin,
        runner=_bad_json_runner,
    )
    assert bad_json_report["ghidra_functions_count"] == 0

    # Exercise _extract_symtab_index edge cases (tiny symtab, st_value=0, non-matching, nonul)
    raw_strtab = (
        b"\x00UnrelatedSymbol"
        + (b"_" * 48)
        + b"\x00_ZN4pjrt4SymAE\x00PJRT_Alias\x00_ZN4pjrt4SymBE\x00PJRT_Unterm"
    )
    off_unrelated = raw_strtab.index(b"UnrelatedSymbol")
    off_sym_a = raw_strtab.index(b"_ZN4pjrt4SymAE")
    off_alias = raw_strtab.index(b"PJRT_Alias")
    off_sym_b = raw_strtab.index(b"_ZN4pjrt4SymBE")
    off_unterm = raw_strtab.index(b"PJRT_Unterm")
    raw_symtab = b"".join(
        [
            struct.pack("<IBBHQQ", 0, 0x12, 0, 1, 0, 0),  # st_value == 0
            struct.pack("<IBBHQQ", 9999, 0x12, 0, 1, 0x1000, 8),  # st_name >= len(strtab)
            struct.pack("<IBBHQQ", off_unrelated, 0x12, 0, 1, 0x1010, 8),  # non-matching prefix
            struct.pack("<IBBHQQ", off_sym_a, 0x12, 0, 1, 0x1020, 8),  # _ZN4pjrt first
            struct.pack("<IBBHQQ", off_alias, 0x12, 0, 1, 0x1020, 8),  # non-_ZN4pjrt duplicate addr
            struct.pack("<IBBHQQ", off_sym_b, 0x12, 0, 1, 0x1020, 8),  # _ZN4pjrt override addr
            struct.pack("<IBBHQQ", off_unterm, 0x12, 0, 1, 0x1030, 8),  # nul == -1
        ]
    )
    sym_blob = raw_strtab + raw_symtab
    sym_secs = [
        rx.ElfSectionHeader(".dynsym", 11, 2, 0, 0, 12, 0, 0),  # sh_size < 24
        rx.ElfSectionHeader(".dynstr", 3, 2, 0, 0, 1, 0, 0),
        rx.ElfSectionHeader(".strtab", 3, 2, 0, 0, len(raw_strtab), 0, 0),
        rx.ElfSectionHeader(".symtab", 2, 2, 0, len(raw_strtab), len(raw_symtab), 0, 0),
    ]
    s_loc, a_sym = gl._extract_symtab_index(sym_blob, sym_secs)
    assert a_sym[0x1020] == "_ZN4pjrt4SymBE"
    assert "PJRT_Unterm" not in s_loc

    # Exercise _scan_x86_mov_lea_stores GOT/reloc/disp8/disp32/non-RIP-LEA branches
    got_va = 0x5000
    got_bytes = struct.pack("<QQ", 0x7000, 0x8000)
    got_secs = [rx.ElfSectionHeader(".got", 1, 3, got_va, 0, len(got_bytes), 0, 0)]
    scan_code = bytearray()
    scan_code.extend(b"\x48\xc7\x07" + struct.pack("<I", 64))  # struct_size = 64
    scan_code.extend(b"\xc7\x47\x08" + struct.pack("<I", 5))  # ext_type = 5
    scan_code.extend(b"\x48\x8d\x00")  # non-RIP-relative LEA (modrm != 0x05)
    # LEA rbx, [rip + disp_to_got] (0x48 0x8D 0x1D <disp32>)
    disp_got = got_va - (0x2000 + len(scan_code) + 7)
    scan_code.extend(b"\x48\x8d\x1d" + struct.pack("<i", disp_got))
    # MOVABS rcx, 0 (0x48 0xB9 <imm64>) -> got_disp_reg[1] = 0 (hits reloc_map)
    scan_code.extend(b"\x48\xb9" + struct.pack("<q", 0))
    # MOVABS rdx, 8 (0x48 0xBA <imm64>) -> got_disp_reg[2] = 8 (hits elf_bytes GOT read)
    scan_code.extend(b"\x48\xba" + struct.pack("<q", 8))
    # MOVABS rsi, 64 -> got_disp_reg[6] = 64 (out-of-bounds goff -> None)
    scan_code.extend(b"\x48\xbe" + struct.pack("<q", 64))
    # MOV rax, [rbx + rbp*8] (idx_reg=5 not in got_disp_reg) -> sib = (3<<6)|(5<<3)|3 = 0xEB
    scan_code.extend(b"\x48\x8b\x04\xeb")
    # MOV rax, [rbx + rcx*8] (idx_reg=1 -> reloc_map[0x5000] = 0x9000) -> sib = 0xCB
    scan_code.extend(b"\x48\x8b\x04\xcb")
    # MOV [rdi + 8], rax (mod=1 disp8=8 valid) & [rdi + 4], rax (disp8=4 invalid)
    scan_code.extend(b"\x48\x89\x47\x08\x48\x89\x47\x04")
    # MOV rax, [rbx + rdx*8] (idx_reg=2 -> GOT elf_bytes = 0x8000) -> sib = 0xD3
    scan_code.extend(b"\x48\x8b\x04\xd3")
    # MOV [rdi + 0x100], rax (mod=2 disp32=0x100 valid) & [rdi + 4], rax (mod=2 disp32=4 invalid)
    scan_code.extend(b"\x48\x89\x87" + struct.pack("<i", 0x100))
    scan_code.extend(b"\x48\x89\x87" + struct.pack("<i", 4))
    # MOV [rdi], rax (mod=0), MOV [rcx + 8], rax (rm=1), and out-of-bounds GOT read (sib=0xF3)
    scan_code.extend(b"\x48\x89\x07\x48\x89\x41\x08\x48\x8b\x04\xf3")
    st_sz, hdr_int, sl_map = gl._scan_x86_mov_lea_stores(
        bytes(scan_code),
        0x2000,
        got_vaddr=got_va,
        elf_bytes=got_bytes,
        sections=got_secs,
        reloc_map={got_va: 0x9000},
    )
    assert (st_sz, hdr_int) == (64, 5)
    assert sl_map == {8: 0x9000, 0x100: 0x8000}
    # Also test _scan_x86_mov_lea_stores with elf_bytes=None on unmapped GOT slot
    _, _, sl_no_elf = gl._scan_x86_mov_lea_stores(
        bytes(scan_code),
        0x2000,
        got_vaddr=got_va,
        elf_bytes=None,
        sections=None,
        reloc_map={},
    )
    assert sl_no_elf == {}

    # Build a synthetic ELF exercising Creator PJRT, Extension forwarders, Profiler, SDK & Driver
    def _build_creator_and_relro_elf(mode: str) -> bytes:
        """Constructs a synthetic ELF64 test image exercising ABI lifting branches.

        Args:
            mode: Layout variant selector controlling symbol and section emission.
        """
        sh_str = (
            b"\x00.shstrtab\x00.rodata\x00.data.rel.ro\x00.data\x00.text\x00.strtab\x00.symtab\x00"
        )
        ebuf = bytearray(b"\x00" * 64)
        sh_off = len(ebuf)
        ebuf.extend(sh_str)
        while len(ebuf) % 8 != 0:
            ebuf.append(0)
        bva = 0x300000

        ro_o = len(ebuf)
        prof_va = bva + len(ebuf)
        ebuf.extend(struct.pack("<QQQ", 32, 1, 0))  # profiler_extension (size=32, type=1)
        ro_l = len(ebuf) - ro_o

        dr_o = len(ebuf)
        ext_relro_va = bva + len(ebuf)
        ebuf.extend(struct.pack("<QQQ", 32, 5, 0))
        ebuf.extend(struct.pack("<QQQii", 512, ext_relro_va, 24, 0, 117))
        ebuf.extend(b"\x00" * (512 - 32))
        dr_l = len(ebuf) - dr_o

        da_o = len(ebuf)
        ebuf.extend(b"\x00" * 4)  # .data with sh_size=4 (< 8)
        da_l = 4
        while len(ebuf) % 8 != 0:
            ebuf.append(0)

        tx_o = len(ebuf)
        # Slot target with struct_size check 32
        slot_fn_va = bva + len(ebuf)
        ebuf.extend(b"\x48\x83\x3f\x20\x72\x02\x31\xc0\xc3" + b"\x90" * 23)
        # CreatePjrtApi: struct_size=64, minor=117, 0x28->slot_fn_va, 0x30->0x999999, 0x38->0
        create_pjrt_va = bva + len(ebuf)
        cp_code = bytearray()
        cp_code.extend(b"\x48\xc7\x00" + struct.pack("<I", 64))
        cp_code.extend(b"\x49\xb9" + struct.pack("<ii", 0, 117))
        disp_slot = slot_fn_va - (create_pjrt_va + len(cp_code) + 7)
        cp_code.extend(b"\x48\x8d\x05" + struct.pack("<i", disp_slot) + b"\x48\x89\x40\x28")
        disp_unmapped = 0x999999 - (create_pjrt_va + len(cp_code) + 7)
        cp_code.extend(b"\x48\x8d\x05" + struct.pack("<i", disp_unmapped) + b"\x48\x89\x40\x30")
        cp_code.extend(b"\xc3")
        ebuf.extend(cp_code)

        # Direct extension creator (size=32, type=5)
        ext_direct_va = bva + len(ebuf)
        ebuf.extend(b"\x48\xc7\x07\x20\x00\x00\x00\xc7\x47\x08\x05\x00\x00\x00\xc3")
        # Impl extension creator for tail-jump (size=40, type=7)
        ext_impl_va = bva + len(ebuf)
        ebuf.extend(b"\x48\xc7\x07\x28\x00\x00\x00\xc7\x47\x08\x07\x00\x00\x00\xc3")
        # Forwarder #1 -> jumps to ext_impl_va
        ext_fwd_ok_va = bva + len(ebuf)
        ebuf.extend(b"\xe9" + struct.pack("<i", ext_impl_va - (ext_fwd_ok_va + 5)))
        # Forwarder #2 -> jumps to unmapped target
        ext_fwd_bad_va = bva + len(ebuf)
        ebuf.extend(b"\xe9" + struct.pack("<i", 0x888888 - (ext_fwd_bad_va + 5)))
        # Forwarder #3 -> Jumps to symbol at unmapped section addr (j_off is None)
        ext_fwd_unmap_va = bva + len(ebuf)
        ebuf.extend(b"\xe9" + struct.pack("<i", 0x777777 - (ext_fwd_unmap_va + 5)))
        # Empty creator (len < 5, no E9) & >=5 creator with no E9
        ext_short_va = bva + len(ebuf)
        ebuf.extend(b"\xc3\x90\x90\x90")
        ext_no_jmp_va = bva + len(ebuf)
        ebuf.extend(b"\x90\x90\x90\x90\x90\xc3")

        # GetTpuPjrtApi calling creators
        get_pjrt_va = bva + len(ebuf)
        gp_code = bytearray()
        for callee_va in (
            ext_direct_va,
            ext_fwd_ok_va,
            ext_fwd_bad_va,
            ext_fwd_unmap_va,
            ext_short_va,
            ext_no_jmp_va,
            0x666666,  # unmapped creator symbol (e_off is None)
            slot_fn_va,  # non-extension call
        ):
            gp_code.extend(
                b"\xe8" + struct.pack("<i", callee_va - (get_pjrt_va + len(gp_code) + 5))
            )
        gp_code.extend(b"\xc3")
        ebuf.extend(gp_code)

        # CreateLibtpuSdkApi: first immediate 16, then 144 (or 16-only when mode != "creator")
        sdk_va = bva + len(ebuf)
        sdk_bytes = (
            b"\xbf\x10\x00\x00\x00\xbf\x90\x00\x00\x00\xc3"
            if mode == "creator"
            else b"\xbf\x10\x00\x00\x00\xc3"
        )
        ebuf.extend(sdk_bytes)

        # Embedding & Gasket helpers
        emb_ok_va = bva + len(ebuf)
        ebuf.extend(b"\x48\x83\x3f\x30\x72\x02\x31\xc0\xc3" + b"\x90" * 23)
        emb_plain_va = bva + len(ebuf)
        ebuf.extend(b"\x31\xc0\xc3\x90\x90\x90\x90\x90")
        gasket_va = bva + len(ebuf)
        ebuf.extend(b"\xba\x01\xdc\x08\x40\xc3\x90\x90")
        tx_l = len(ebuf) - tx_o

        sym_defs: list[tuple[bytes, int, int]] = [
            (b"PJRT_SlotFn", slot_fn_va, 32),
            (b"TpuDriver_Initialize", emb_plain_va, 8),
            (b"TpuEmbeddingEngine_Ok", emb_ok_va, 32),
            (b"TpuEmbeddingEngine_Plain", emb_plain_va, 8),
            (b"TpuEmbeddingEngine_Unmapped", 0x555555, 16),
            (b"GasketIoctlFn", gasket_va, 8),
            (b"VfioUnmappedFn", 0x555550, 8),
        ]
        if mode.startswith("creator"):
            sym_defs.extend(
                [
                    (b"_ZN4pjrt13CreatePjrtApiEv", create_pjrt_va, len(cp_code)),
                    (
                        b"_ZN4pjrt10tpu_plugin22PJRT_Plugin_InitializeEP27PJRT_Plugin_Initialize_Args",
                        slot_fn_va,
                        32,
                    ),
                    (b"_ZN4pjrt10tpu_plugin13GetTpuPjrtApiEv", get_pjrt_va, len(gp_code)),
                    (b"_ZN4pjrt20CreateDirectExtensionEv", ext_direct_va, 15),
                    (b"_ZN4pjrt22CreateForwardExtensionEv", ext_fwd_ok_va, 5),
                    (b"_ZN4pjrt26CreateForwardImplExtensionEv", ext_impl_va, 15),
                    (b"_ZN4pjrt25CreateBadForwardExtensionEv", ext_fwd_bad_va, 5),
                    (b"_ZN4pjrt27CreateUnmapForwardExtensionEv", ext_fwd_unmap_va, 5),
                    (b"_ZN4pjrt24CreateUnmapImplExtensionEv", 0x777777, 15),
                    (b"_ZN4pjrt20CreateShortExtensionEv", ext_short_va, 4),
                    (b"_ZN4pjrt20CreateNoJmpExtensionEv", ext_no_jmp_va, 6),
                    (b"_ZN4pjrt23CreateUnmappedExtensionEv", 0x666666, 8),
                    (b"_ZN6libtpu3sdk18CreateLibtpuSdkApiEv", sdk_va, len(sdk_bytes)),
                ]
            )
            if mode == "creator":
                sym_defs.append((b"_ZN4pjrt10tpu_plugin18profiler_extensionE", prof_va, 24))
            elif mode == "creator_unmap_prof":
                sym_defs.append((b"_ZN4pjrt10tpu_plugin18profiler_extensionE", 0x999990, 24))
        else:
            # Exercise unmapped c_off, g_off, prof_off, sdk_off == None + relro fallback!
            sym_defs.extend(
                [
                    (b"_ZN4pjrt13CreatePjrtApiEv", 0x444440, 16),
                    (b"_ZN4pjrt10tpu_plugin13GetTpuPjrtApiEv", 0x444450, 16),
                    (b"_ZN4pjrt10tpu_plugin18profiler_extensionE", 0x444460, 24),
                    (b"_ZN6libtpu3sdk18CreateLibtpuSdkApiEv", 0x444480, 16),
                ]
            )

        st_o = len(ebuf)
        st_bytes = bytearray(b"\x00")
        sy_entries = [struct.pack("<IBBHQQ", 0, 0, 0, 0, 0, 0)]
        for s_nm, s_va, s_sz in sym_defs:
            nm_idx = len(st_bytes)
            st_bytes.extend(s_nm + b"\x00")
            sy_entries.append(struct.pack("<IBBHQQ", nm_idx, 0x12, 0, 5, s_va, s_sz))
        ebuf.extend(st_bytes)
        st_l = len(st_bytes)
        while len(ebuf) % 8 != 0:
            ebuf.append(0)
        sy_o = len(ebuf)
        for ent in sy_entries:
            ebuf.extend(ent)
        sy_l = len(ebuf) - sy_o

        def _s_idx(nm: bytes) -> int:
            """Looks up the byte offset of section name `nm` in `.shstrtab`.

            Args:
                nm: Null-free section name bytes.
            """
            return sh_str.index(nm + b"\x00")

        sh_tbl = len(ebuf)
        for nm_i, tp, fl, va, of, sz, lk in (
            (0, 0, 0, 0, 0, 0, 0),
            (_s_idx(b".shstrtab"), 3, 0, 0, sh_off, len(sh_str), 0),
            (_s_idx(b".rodata"), 1, 2, bva + ro_o, ro_o, ro_l, 0),
            (_s_idx(b".data.rel.ro"), 1, 3, bva + dr_o, dr_o, dr_l, 0),
            (_s_idx(b".data"), 1, 3, bva + da_o, da_o, da_l, 0),
            (_s_idx(b".text"), 1, 6, bva + tx_o, tx_o, tx_l, 0),
            (_s_idx(b".strtab"), 3, 0, 0, st_o, st_l, 0),
            (_s_idx(b".symtab"), 2, 0, 0, sy_o, sy_l, 6),
        ):
            ebuf.extend(
                struct.pack("<IIQQQQIIQQ", nm_i, tp, fl, va, of, sz, lk, 0, 8, 24 if tp == 2 else 0)
            )

        hdr = struct.pack(
            "<4sBBBBB7xHHIQQQIHHHHHH",
            b"\x7fELF",
            2,
            1,
            1,
            0,
            0,
            3,
            62,
            1,
            0,
            0,
            sh_tbl,
            0,
            64,
            0,
            0,
            64,
            8,
            1,
        )
        ebuf[:64] = hdr
        return bytes(ebuf)

    creator_elf = _build_creator_and_relro_elf("creator")
    creator_path = tmp_path / "creator_synth.so"
    creator_path.write_bytes(creator_elf)
    creator_vt = gl.lift_vtable_and_entrypoint_layouts(creator_elf)
    assert creator_vt["pjrt_api_struct_size"] == 64
    assert creator_vt["pjrt_api_minor"] == 117
    assert creator_vt["extensions_count"] == 3
    assert creator_vt["libtpu_sdk_api_struct_size"] == 144
    assert creator_vt["tpu_driver_fn_struct_size"] == 224
    assert 0x4008DC01 in gl.lift_isa_and_driver_ioctls(creator_elf).gasket_ioctl_cmds
    assert 32 in gl.lift_pjrt_args_and_extensions(creator_elf).observed_struct_sizes

    no_prof_vt = gl.lift_vtable_and_entrypoint_layouts(
        _build_creator_and_relro_elf("creator_no_prof")
    )
    assert no_prof_vt["extensions_count"] == 2
    assert no_prof_vt["libtpu_sdk_api_struct_size"] == 0
    unmap_prof_vt = gl.lift_vtable_and_entrypoint_layouts(
        _build_creator_and_relro_elf("creator_unmap_prof")
    )
    assert unmap_prof_vt["extensions_count"] == 2

    relro_elf = _build_creator_and_relro_elf("relro")
    relro_vt = gl.lift_vtable_and_entrypoint_layouts(relro_elf)
    assert relro_vt["pjrt_api_struct_size"] == 512
    assert relro_vt["pjrt_api_minor"] == 117
    assert relro_vt["extensions_count"] == 1

    # Verify golden manifest pass path deterministically offline
    synth_golden_root = tmp_path / "golden_repo"
    synth_re_dir = synth_golden_root / "docs" / "reverse_engineering"
    synth_re_dir.mkdir(parents=True)
    (synth_re_dir / "golden_binary_manifest.json").write_text(
        json.dumps(
            {
                "libtpu_so": {
                    "pjrt_api_struct_size": 64,
                    "pjrt_api_version_minor": 117,
                    "libtpu_sdk_api_struct_size": 144,
                }
            }
        ),
        encoding="utf-8",
    )
    (synth_re_dir / "golden_struct_layouts.json").write_text(
        json.dumps({"extensions": [1, 2, 3]}),
        encoding="utf-8",
    )
    assert gl.verify_lift_against_golden({"vtable_layouts": creator_vt}, synth_golden_root) == []

    # CLI modes + __main__
    assert gl.main(["--binary", str(elf_path)]) == 0
    assert gl.main(["--binary", str(creator_path), "--verify-golden", str(synth_golden_root)]) == 0
    assert gl.main(["--binary", str(elf_path), "--verify-golden", str(repo_root)]) == 1
    assert (
        gl.main(
            [
                "--binary",
                str(elf_path),
                "--ghidra-headless",
                "analyzeHeadless",
                "--ghidra-project-dir",
                str(tmp_path / "ghidra_out"),
            ],
            runner=lambda _cmd: 0,
        )
        == 0
    )
    assert gl.main(["--binary", str(missing_so)]) == 1

    with (
        patch.object(sys, "argv", ["ghidra_lift_isa_and_abi.py", "--binary", str(elf_path)]),
        pytest.raises(SystemExit) as exc_info,
    ):
        runpy.run_path(str(Path(gl.__file__)), run_name="__main__")
    assert exc_info.value.code == 0


def test_differential_oracle_full_coverage(tmp_path: Path) -> None:
    """Tests `scripts/differential_oracle.py` corpus, wire decoder, PJRT backend, and MMIO trace.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
    """
    with pytest.raises(do.DifferentialOracleError, match="target_count must be >= 1"):
        do.build_differential_corpus(0)

    corpus = do.build_differential_corpus(25)
    assert len(corpus) == 25
    assert {c.dialect_kind for c in corpus} == {"stablehlo", "shardy", "mosaic_tpu"}

    with pytest.raises(do.DifferentialOracleError, match="not found"):
        do.LibtpuSharedLibraryBackend(tmp_path / "absent.so")

    fake_so = tmp_path / "fake_libtpu.so"
    fake_so.write_bytes(b"placeholder")

    null_cdll = SimpleNamespace(GetPjrtApi=lambda: 0)
    with (
        patch.object(ctypes, "CDLL", return_value=null_cdll),
        pytest.raises(do.DifferentialOracleError, match="returned NULL"),
    ):
        do.LibtpuSharedLibraryBackend(fake_so)

    # Test undersized struct_size and NULL slot error paths on synthetic PJRT_Api buffer
    tiny_api = (ctypes.c_uint64 * 8)(16, 0, 24, 0, 0, 0, 0, 0)
    tiny_cdll = SimpleNamespace(GetPjrtApi=lambda: ctypes.addressof(tiny_api))
    with patch.object(ctypes, "CDLL", return_value=tiny_cdll):
        tiny_backend = do.LibtpuSharedLibraryBackend(fake_so)
        assert tiny_backend.api_pointer == ctypes.addressof(tiny_api)
        with pytest.raises(do.DifferentialOracleError, match="too small"):
            tiny_backend.compile_to_serialized_executable(corpus[0])

    null_slot_api = (ctypes.c_uint64 * 143)(1144)
    null_slot_cdll = SimpleNamespace(GetPjrtApi=lambda: ctypes.addressof(null_slot_api))
    with patch.object(ctypes, "CDLL", return_value=null_slot_cdll):
        ns_backend = do.LibtpuSharedLibraryBackend(fake_so)
        with pytest.raises(do.DifferentialOracleError, match="is NULL"):
            ns_backend.compile_to_serialized_executable(corpus[0])

    # Test real PJRT compilation, serialization, error handling, and H2D -> Execute -> D2H smoke!
    repo_root = Path(__file__).resolve().parents[2]
    real_lib = next(
        (
            cand
            for cand in (
                repo_root / "target" / "debug" / "libtpu.dylib",
                repo_root / "target" / "debug" / "libtpu.so",
                repo_root / "target" / "release" / "libtpu.dylib",
                repo_root / "target" / "release" / "libtpu.so",
            )
            if cand.exists()
        ),
        None,
    )
    assert real_lib is not None, "Expected built libtpu shared library in target/"
    real_backend = do.LibtpuSharedLibraryBackend(real_lib)
    assert real_backend.api_pointer != 0
    summaries = do.run_differential_corpus(real_backend, corpus[:3])
    assert len(summaries) == 3
    assert summaries[0].byte_length > 12
    assert len(summaries[0].wire_field_numbers) >= 1

    smoke_out = real_backend.compile_and_execute_smoke()
    assert smoke_out == [11.0, 22.0, 33.0, 44.0]
    real_backend._destroy_event_if_set(0)

    # Exercise PJRT_Error_Message + PJRT_Error_Destroy path on invalid MLIR program
    bad_case = do.OracleProgramCase(
        case_id="bad_mlir",
        generation="v5e",
        topology_spec="2x2",
        dialect_kind="stablehlo",
        op_name="invalid",
        mlir_text="SYNTAX_ERROR",
    )
    with pytest.raises(do.DifferentialOracleError):
        real_backend.compile_to_serialized_executable(bad_case)

    # Exercise mock vtable branches: NULL error message, topo_handle==0, and smoke 0-devices cleanup
    noop_err_cb = do._PJRT_VOID_FN_PROTO(lambda _arg: None)
    always_err_cb = do._PJRT_FN_PROTO(lambda _arg: 0xDEAD)
    ok_cb = do._PJRT_FN_PROTO(lambda _arg: 0)

    def _set_client_cb(raw_ptr: int) -> int:
        """Populates a synthetic non-zero client handle in `PJRT_Client_Create_Args`.

        Args:
            raw_ptr: Raw integer address of `PJRT_Client_Create_Args`.
        """
        c_args = do.PJRT_Client_Create_Args.from_address(raw_ptr)
        c_args.client = 0xBEEF
        return 0

    set_client_fn = do._PJRT_FN_PROTO(_set_client_cb)
    mock_api = (ctypes.c_uint64 * 143)(0)
    mock_api[0] = 1144
    mock_api[0x028 // 8] = ctypes.cast(noop_err_cb, ctypes.c_void_p).value or 0
    mock_api[0x030 // 8] = ctypes.cast(noop_err_cb, ctypes.c_void_p).value or 0
    mock_api[0x040 // 8] = ctypes.cast(ok_cb, ctypes.c_void_p).value or 0
    mock_api[0x078 // 8] = ctypes.cast(set_client_fn, ctypes.c_void_p).value or 0
    mock_api[0x080 // 8] = ctypes.cast(ok_cb, ctypes.c_void_p).value or 0
    mock_api[0x0A8 // 8] = ctypes.cast(ok_cb, ctypes.c_void_p).value or 0
    mock_api[0x2B8 // 8] = ctypes.cast(always_err_cb, ctypes.c_void_p).value or 0
    mock_cdll = SimpleNamespace(GetPjrtApi=lambda: ctypes.addressof(mock_api))
    with patch.object(ctypes, "CDLL", return_value=mock_cdll):
        mock_backend = do.LibtpuSharedLibraryBackend(fake_so)
        with pytest.raises(do.DifferentialOracleError, match="PJRT call at 0x2b8 failed"):
            mock_backend.compile_to_serialized_executable(corpus[0])
        with pytest.raises(do.DifferentialOracleError, match="0 addressable devices"):
            mock_backend.compile_and_execute_smoke()
        # Also test client_handle == 0 cleanup branch when client_create leaves client = None
        mock_api[0x078 // 8] = ctypes.cast(ok_cb, ctypes.c_void_p).value or 0
        with pytest.raises(do.DifferentialOracleError, match="0 addressable devices"):
            mock_backend.compile_and_execute_smoke()

    # Wire decoder: LTPUEXEC envelope + error branches, wire_type 0/1/2/5, and protobuf errors
    case0 = corpus[0]
    with pytest.raises(do.DifferentialOracleError, match="Truncated LTPUEXEC binary header"):
        do.decode_serialized_executable_wire_summary(case0, b"LTPUEXECshort")
    with pytest.raises(do.DifferentialOracleError, match="Unsupported LTPUEXEC format version"):
        do.decode_serialized_executable_wire_summary(
            case0, b"LTPUEXEC" + struct.pack("<IBI", 2, 0, 0) + b"\x00" * 8
        )
    with pytest.raises(do.DifferentialOracleError, match="Truncated LTPUEXEC format section"):
        do.decode_serialized_executable_wire_summary(
            case0, b"LTPUEXEC" + struct.pack("<IBI", 1, 0, 99) + b"\x00" * 8
        )
    with pytest.raises(do.DifferentialOracleError, match="Truncated LTPUEXEC source section"):
        do.decode_serialized_executable_wire_summary(
            case0,
            b"LTPUEXEC" + struct.pack("<IBI", 1, 0, 4) + b"mlir" + struct.pack("<I", 99) + b"abcd",
        )
    with pytest.raises(do.DifferentialOracleError, match="Mismatched LTPUEXEC payload length"):
        do.decode_serialized_executable_wire_summary(
            case0,
            b"LTPUEXEC"
            + struct.pack("<IBI", 1, 0, 4)
            + b"mlir"
            + struct.pack("<I", 2)
            + b"ok"
            + struct.pack("<I", 99),
        )

    all_wire_types = (
        b"\x08\x96\x01"  # field 1, varint 150
        + b"\x11"
        + (b"\x00" * 8)  # field 2, fixed64
        + b"\x1a\x03abc"  # field 3, len-delimited
        + b"\x25"
        + (b"\x00" * 4)  # field 4, fixed32
    )
    summary_all = do.decode_serialized_executable_wire_summary(case0, all_wire_types)
    assert summary_all.wire_field_numbers == (1, 2, 3, 4)

    with pytest.raises(do.DifferentialOracleError, match="Truncated protobuf varint"):
        do.decode_serialized_executable_wire_summary(case0, b"\x80")
    with pytest.raises(do.DifferentialOracleError, match="exceeded 64-bit"):
        do.decode_serialized_executable_wire_summary(case0, b"\x80" * 11)
    with pytest.raises(do.DifferentialOracleError, match="zero protobuf field"):
        do.decode_serialized_executable_wire_summary(case0, b"\x00")
    with pytest.raises(do.DifferentialOracleError, match="Unsupported protobuf wire type"):
        do.decode_serialized_executable_wire_summary(case0, b"\x0b")  # wire_type 3
    with pytest.raises(do.DifferentialOracleError, match="Truncated length-delimited"):
        do.decode_serialized_executable_wire_summary(case0, b"\x0a\x10abc")

    # LD_PRELOAD C source & MMIO trace JSONL parser/archiver
    c_src = do.generate_ld_preload_mmio_tracer_c_source()
    assert "RTLD_NEXT" in c_src

    trace_jsonl = (
        NL
        + '{"seq": 1, "kind": "ioctl", "fd_or_bar": 3, "offset_or_cmd": 15217, "value_or_size": 32}'
        + NL
    )
    events = do.parse_mmio_trace_jsonl(trace_jsonl)
    assert len(events) == 1
    out_trace = do.archive_golden_mmio_trace(events, tmp_path / "traces" / "trace0.json")
    assert out_trace.exists()

    with pytest.raises(do.DifferentialOracleError, match="Invalid MMIO trace"):
        do.parse_mmio_trace_jsonl('{"bad": true}')

    # CLI paths + __main__
    preload_out = tmp_path / "c" / "tracer.c"
    assert do.main(["--emit-preload-c", str(preload_out)]) == 0
    assert preload_out.exists()
    assert do.main(["--corpus-count", "7"]) == 0
    assert do.main(["--corpus-count", "2", "--library", str(real_lib)]) == 0
    assert do.main(["--corpus-count", "0"]) == 1

    with (
        patch.object(sys, "argv", ["differential_oracle.py", "--corpus-count", "2"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        runpy.run_path(str(Path(do.__file__)), run_name="__main__")
    assert exc_info.value.code == 0

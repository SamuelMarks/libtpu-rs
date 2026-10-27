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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from elftools.common.exceptions import ELFError
from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection

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

OFFICIAL_LIBTPU_SO_SHA256 = "ef2f7af1f1ca0f27db6a6a3adffdd2b275a3215b121c8f9bc3f6ce842fc7c700"
OFFICIAL_SDK_SO_SHA256 = "6d65c03e2b584eaa3b546b6b64931c4b153ad5c4e2b404f578ea548c0fb6d2a4"


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


def fetch_pypi_wheel_url(version: str = "0.0.50", platform_tag: str = "x86_64") -> tuple[str, str]:
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


def carve_protodesc_cold(section_bytes: bytes) -> list[dict[str, str]]:
    """Carves embedded `FileDescriptorProto` names and packages from a `protodesc_cold` buffer.

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
            name_len = section_bytes[idx + 1]
            if 7 <= name_len <= 120 and idx + 2 + name_len <= n:
                raw_name = section_bytes[idx + 2 : idx + 2 + name_len]
                if raw_name.endswith(b".proto") and all(32 < b < 127 for b in raw_name):
                    name_str = raw_name.decode("utf-8", errors="replace")
                    pkg_str = ""
                    end_idx = idx + 2 + name_len
                    if end_idx + 2 <= n and section_bytes[end_idx] == 0x12:
                        pkg_len = section_bytes[end_idx + 1]
                        if end_idx + 2 + pkg_len <= n:
                            pkg_str = section_bytes[end_idx + 2 : end_idx + 2 + pkg_len].decode(
                                "utf-8", errors="replace"
                            )
                            end_idx += 2 + pkg_len
                    found.append({"name": name_str, "package": pkg_str})
                    idx = end_idx
                    continue
        idx += 1
    return found


def extract_filewrapper_toc(toc_bytes: bytes, blob_bytes: bytes) -> list[tuple[str, bytes]]:
    """Extracts `(name, payload)` entries from a 24-byte `filewrapper_toc` table of contents.

    Args:
        toc_bytes: Raw 24-byte `(name_off, data_off, size)` records.
        blob_bytes: Backing string and data buffer.

    Returns:
        List of `(filename, data_bytes)` tuples.
    """
    entries: list[tuple[str, bytes]] = []
    for off in range(0, len(toc_bytes) - (len(toc_bytes) % 24), 24):
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
        elf_cls: Any = ELFFile
        elf = elf_cls(io.BytesIO(elf_bytes))
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
                sym_sec: Any = sec
                for sym in sym_sec.iter_symbols():
                    st_info_dict = sym.entry["st_info"]
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
                                vaddr=int(sym.entry["st_value"]),
                                size=int(sym.entry["st_size"]),
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
                binaries_dict[info.filename] = {
                    "size_bytes": len(payload),
                    "sha256": sha256_hex,
                    "section_names": [s.name for s in sections if s.name],
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
        libtpu_so.get("exported_functions_count") != 78
        or len(libtpu_so.get("exported_functions", [])) != 78
    ):
        errors.append("golden_binary_manifest.json: exported_functions != 78")

    ext_chain = layouts.get("extensions", [])
    if len(ext_chain) != 17:
        errors.append(f"golden_struct_layouts.json: expected 17 extensions, got {len(ext_chain)}")

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
        "compile_options.proto",
        "deepsea_executable.proto",
        "emitter_learned_cost_model_options.proto",
        "hlo.proto",
        "host_command_opcodes.proto",
        "isa_program.proto",
        "megascale.proto",
        "memory_scheduler.proto",
        "memory_space_assignment.proto",
        "runtime_config.proto",
        "target_arguments.proto",
        "tpu_compilation_environment.proto",
        "tpu_embedding_configuration.proto",
        "tpu_executable.proto",
        "tpu_runtime.proto",
        "tpu_topology.proto",
        "tpu_version.proto",
        "xla_data.proto",
        "xplane.proto",
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
            or len(proto_desc.get("descriptors", [])) < 59
        ):
            errors.append("golden_proto_descriptors.json: incomplete descriptor closure")
        if set(csr_map.get("generations", {}).keys()) != {"V3", "V4", "V5e", "V5p", "V6e"}:
            errors.append("csr_register_map.json: missing TPU generation CSR maps")

    return errors


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
        type=Path,
        default=None,
        help="Optional path to a libtpu .whl archive to inspect",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Path to the repository root directory",
    )
    parser.add_argument(
        "--verify-golden",
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


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

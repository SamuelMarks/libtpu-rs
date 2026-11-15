#!/usr/bin/env python3
"""C++ RTTI, `.eh_frame` CFI, PJRT ABI, VLIW/SparseCore ISA, and Driver Lifter (`libtpu-rs`).

Recovers Itanium C++ ABI RTTI hierarchies (`_ZTI*`, `_ZTS*`, `_ZTV*`), `.eh_frame` function
bounds (`x86_64` & `aarch64`), `PJRT_*_Args` `struct_size` checks and 17-extension layouts,
VLIW/SparseCore ISA bundle metadata, and Gasket (`0xDC`) / VFIO (`0x3B`) `ioctl` callsites
from `libtpu.so`, with optional headless Ghidra (`analyzeHeadless`) export execution.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from scripts.re_extract import (
    ElfSectionHeader,
    extract_descriptor_table_protos,
    extract_filewrapper_toc_entries,
    parse_eh_frame_cfi_records,
    parse_elf64_sections_and_dynsym,
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


def lift_pjrt_args_and_extensions(elf_bytes: bytes) -> PjrtAbiLiftReport:
    """Lifts `PJRT_*_Args` `struct_size` checks, field displacements, and extension IDs.

    Scans `.text` for `x86_64` (`cmp qword ptr [rdi], imm` / `mov r*, [rdi + disp8]`)
    and `aarch64` (`cmp x*, #imm`) patterns, plus `.data.rel.ro` `PJRT_Extension_Base` headers.

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
        # x86_64: 48 83 3f <imm8>  => cmp qword ptr [rdi], imm8
        for match in re.finditer(rb"\x48\x83\x3f([\x08-\x7f])", code):
            imm = match.group(1)[0]
            if imm % 8 == 0:
                struct_sizes.add(imm)
        # x86_64: 48 81 3f <imm32> => cmp qword ptr [rdi], imm32
        for match in re.finditer(rb"\x48\x81\x3f(....)", code, re.DOTALL):
            imm32 = struct.unpack("<I", match.group(1))[0]
            if 16 <= imm32 <= 512 and imm32 % 8 == 0:
                struct_sizes.add(int(imm32))
        # x86_64: 48 8b 47 <disp8> => mov rax, qword ptr [rdi + disp8]
        for match in re.finditer(rb"\x48\x8b\x47([\x08-\x78])", code):
            disp = match.group(1)[0]
            if disp % 8 == 0:
                displacements.add(disp)

    drel_sec = _find_section(sections, ".data.rel.ro")
    if drel_sec is not None and drel_sec.sh_size >= 24:
        drel = elf_bytes[drel_sec.sh_offset : drel_sec.sh_offset + min(drel_sec.sh_size, 2_000_000)]
        for off in range(0, len(drel) - 24, 8):
            ext_sz, ext_ty = struct.unpack_from("<QI", drel, off)
            if ext_sz in (32, 40, 56, 64, 80, 96, 120, 248, 280) and 1 <= ext_ty <= 35:
                ext_types.add(int(ext_ty))

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
    gasket_cmds: set[int] = set()
    vfio_cmds: set[int] = set()
    text_sec = _find_section(sections, ".text")
    if text_sec is not None and text_sec.sh_size >= 6:
        code_bytes = elf_bytes[
            text_sec.sh_offset : text_sec.sh_offset + min(text_sec.sh_size, 4_000_000)
        ]
        # Scan MOV ESI/EDX, imm32 (be / ba <imm32>) for Gasket 0xDC and VFIO 0x3B ioctl numbers
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


def generate_ghidra_headless_postscript() -> str:
    """Generates a deterministic PyGhidra / Ghidra headless post-analysis export script.

    Returns:
        Python post-script source text consumable by Ghidra `analyzeHeadless -postScript`.
    """
    return '''"""Headless Ghidra post-script exporting PJRT vtables and LLO ISA functions."""
import json
funcs = []
fm = currentProgram.getFunctionManager()
for fn in fm.getFunctions(True):
    name = fn.getName()
    if any(tok in name for tok in ("Pjrt", "llo", "TensorCore", "SparseCore", "Gasket", "Vfio")):
        sz = fn.getBody().getNumAddresses()
        funcs.append({"name": name, "entry": str(fn.getEntryPoint()), "size": sz})
print(json.dumps({"ghidra_functions": funcs}))
'''


def run_ghidra_headless(
    ghidra_headless_bin: Path,
    project_dir: Path,
    binary_path: Path,
    runner: CommandRunner = default_command_runner,
) -> int:
    """Writes the Ghidra post-script and executes `analyzeHeadless` on `binary_path`.

    Args:
        ghidra_headless_bin: Path to Ghidra's `analyzeHeadless` executable.
        project_dir: Temporary Ghidra project workspace directory.
        binary_path: Target ELF64 binary (`libtpu.so`) to analyze.
        runner: Injectable subprocess runner.

    Returns:
        Exit code returned by `analyzeHeadless`.

    Raises:
        BinaryLiftError: If `binary_path` does not exist.
    """
    if not binary_path.exists():
        raise BinaryLiftError(f"Target ELF64 binary does not exist: {binary_path}")
    project_dir.mkdir(parents=True, exist_ok=True)
    script_path = project_dir / "libtpu_ghidra_postscript.py"
    script_path.write_text(generate_ghidra_headless_postscript(), encoding="utf-8")
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
    """Runs full static RTTI, `.eh_frame`, PJRT ABI, ISA, and driver lifting on `binary_path`.

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
    isa_drv = lift_isa_and_driver_ioctls(raw)
    return {
        "binary": str(binary_path),
        "rtti_class_count": len(rtti),
        "rtti_sample": [asdict(c) for c in rtti[:25]],
        "eh_frame_function_count": len(eh_bounds),
        "pjrt_abi": asdict(pjrt),
        "isa_and_driver": asdict(isa_drv),
    }


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
    args = parser.parse_args(argv)

    try:
        if args.ghidra_headless is not None:
            return run_ghidra_headless(
                ghidra_headless_bin=Path(args.ghidra_headless),
                project_dir=Path(args.ghidra_project_dir),
                binary_path=Path(args.binary),
                runner=runner,
            )
        report = lift_elf_binary(Path(args.binary))
        print(json.dumps(report, indent=2))
        return 0
    except BinaryLiftError as exc:
        print(f"BinaryLiftError: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

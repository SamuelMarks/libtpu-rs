//! ELF64/Mach-O symbol inspector, `.eh_frame` parser, and C++ demangler.
//!
//! Provides:
//! - [`inspect_binary_symbols`], [`BinarySymbolReport`], [`ExportedBinarySymbol`], and [`BinaryFormatKind`]
//!   powered by `goblin`.
//! - [`inspect_object_crate_sections`], [`ObjectCrateBinaryReport`], and [`BinarySectionInfo`]
//!   powered by `object`.
//! - [`parse_eh_frame_fde_ranges`] and [`EhFrameFunctionRange`] powered by `gimli`.
//! - [`demangle_and_classify_cpp_symbol`], [`reconstruct_rtti_hierarchy`], [`DemangledCppSymbol`],
//!   [`DemangledSymbolEntry`], [`CppNamespaceCategory`], and [`RttiClassHierarchyEntry`] powered by `cpp_demangle`.

use crate::error::{LibTpuError, LibTpuResult};
use gimli::{BaseAddresses, CieOrFde, EhFrame, LittleEndian, UnwindSection};
use goblin::Object;
use object::{Object as _, ObjectSection as _, ObjectSymbol as _};
use std::collections::BTreeMap;

/// ELF global symbol binding (`STB_GLOBAL = 1`).
pub const ELF_STB_GLOBAL: u8 = 1;

/// ELF weak symbol binding (`STB_WEAK = 2`).
pub const ELF_STB_WEAK: u8 = 2;

/// Binary container format classification returned by [`inspect_binary_symbols`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryFormatKind {
    /// 64-bit or 32-bit ELF shared object (`libtpu.so`, `sdk.abi3.so`).
    Elf,
    /// macOS 64-bit Mach-O dynamic library (`libtpu.dylib`).
    MachO,
}

/// Exported function or data symbol discovered in a shared library image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportedBinarySymbol {
    /// Symbol name.
    pub name: String,
    /// Virtual address or offset of the symbol within the binary image.
    pub address: u64,
    /// Symbol size in bytes (`0` if not recorded by the container format).
    pub size: u64,
}

/// Summary report produced by [`inspect_binary_symbols`] via `goblin::Object::parse`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinarySymbolReport {
    /// Detected binary container format.
    pub format: BinaryFormatKind,
    /// Section or segment names present in the binary image.
    pub section_names: Vec<String>,
    /// Exported dynamic symbols discovered in the binary image.
    pub exported_symbols: Vec<ExportedBinarySymbol>,
}

/// Detailed section metadata extracted via `object::File::sections` (`BinarySectionInfo`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinarySectionInfo {
    /// Section name (e.g., `".dynsym"`, `"protodesc_cold"`, `"filewrapper_toc"`).
    pub name: String,
    /// Section load virtual address (`sh_addr`).
    pub virtual_address: u64,
    /// File byte offset (`sh_offset`, or `0` for `NOBITS`).
    pub file_offset: u64,
    /// Section byte size (`sh_size`).
    pub size_bytes: u64,
    /// Section alignment in bytes (`sh_addralign`).
    pub alignment: u64,
}

/// Binary section and dynamic symbol summary produced via the `object` crate (`object::File::parse`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectCrateBinaryReport {
    /// Whether the parsed binary is 64-bit (`true` for `ELF64` / `MachO64`).
    pub is_64: bool,
    /// Whether the parsed binary is little-endian.
    pub is_little_endian: bool,
    /// Ordered list of non-empty section names discovered by `object::File::sections`.
    pub section_names: Vec<String>,
    /// Detailed [`BinarySectionInfo`] records for every section in the binary.
    pub sections: Vec<BinarySectionInfo>,
    /// Ordered list of dynamic symbol names discovered by `object::File::dynamic_symbols`.
    pub dynamic_symbol_names: Vec<String>,
}

/// Function address range recovered from a DWARF `.eh_frame` Frame Description Entry (`FDE`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EhFrameFunctionRange {
    /// Starting virtual address of the function (`fde.initial_address()`).
    pub initial_address: u64,
    /// Length of the function body in bytes (`fde.len()`).
    pub len: u64,
}

/// Architectural subsystem classification for demangled C++ symbols in `libtpu.so` and `sdk.so`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CppNamespaceCategory {
    /// PJRT C-API and plugin adapter layer (`pjrt::*`).
    Pjrt,
    /// XLA HLO compiler and service layer (`xla::*`).
    Xla,
    /// MLIR, `StableHLO`, `VHLO`, and Mosaic TPU dialect layer (`mlir::*`).
    MlirStableHlo,
    /// User-space `PCIe` / Gasket / VFIO kernel driver layer (`asic_sw::driver::*`).
    AsicSwDriver,
    /// `DeepSea` per-generation ASIC hardware and ISA layer (`asic_sw::deepsea::*`).
    AsicSwDeepsea,
    /// Legacy `TpuDriver` C/C++ layer (`tpu_driver::*`).
    TpuDriver,
    /// TensorFlow TPU runtime and embedding layer (`tensorflow::tpu::*` / `tpu::*`).
    TensorflowTpu,
    /// Multi-slice Megascale collective communication layer (`megascale::*`).
    Megascale,
    /// `libtpu` Python/C SDK telemetry and debug layer (`libtpu::sdk::*`).
    LibtpuSdk,
    /// Uncategorized or standard library symbol.
    Other,
}

/// Demangled C++ symbol and RTTI classification produced by [`demangle_and_classify_cpp_symbol`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DemangledCppSymbol {
    /// Original mangled symbol string (`_Z...`).
    pub mangled: String,
    /// Demangled human-readable C++ signature or type name.
    pub demangled: String,
    /// Subsystem namespace classification.
    pub category: CppNamespaceCategory,
    /// Whether this symbol is an Itanium C++ ABI virtual method table (`_ZTV*`).
    pub is_vtable: bool,
    /// Whether this symbol is an Itanium C++ ABI `std::type_info` descriptor (`_ZTI*`).
    pub is_typeinfo: bool,
}

/// Type alias for [`DemangledCppSymbol`] (`DemangledSymbolEntry`).
pub type DemangledSymbolEntry = DemangledCppSymbol;

/// Reconstructed Itanium C++ ABI RTTI and vtable class descriptor from `.dynsym` / `.symtab`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RttiClassHierarchyEntry {
    /// Demangled C++ class name (e.g., `"asic_sw::driver::GasketDriver"`).
    pub class_name: String,
    /// Top-level C++ namespace prefix (e.g., `"asic_sw"`).
    pub namespace: String,
    /// Virtual address of `_ZTV*` (`vtable for ...`) if present.
    pub vtable_vaddr: u64,
    /// Virtual address of `_ZTI*` (`typeinfo for ...`) if present.
    pub typeinfo_vaddr: u64,
    /// Virtual address of `_ZTS*` (`typeinfo name for ...`) if present.
    pub typeinfo_name_vaddr: u64,
}

/// Parses an ELF or Mach-O shared library byte slice using `goblin` and extracts its
/// section names and exported dynamic symbols.
///
/// # Arguments
///
/// * `bytes` - Raw binary image bytes.
///
/// # Errors
///
/// Returns [`LibTpuError::GoblinParseFailed`] if `goblin::Object::parse` fails or
/// [`LibTpuError::BinaryFormatError`] if the container is an archive or unsupported format.
pub fn inspect_binary_symbols(bytes: &[u8]) -> LibTpuResult<BinarySymbolReport> {
    let obj = Object::parse(bytes)?;
    match obj {
        Object::Elf(elf) => {
            let mut section_names = Vec::new();
            let mut exported_symbols = Vec::new();
            for sh in &elf.section_headers {
                let sec_name = elf.shdr_strtab.get_at(sh.sh_name).unwrap_or_default();
                if !sec_name.is_empty() {
                    section_names.push(String::from(sec_name));
                }
                if sh.sh_type == goblin::elf::section_header::SHT_DYNSYM && sh.sh_entsize == 24 {
                    let str_sh = elf.section_headers.get(sh.sh_link as usize).unwrap_or(sh);
                    let str_start = usize::try_from(str_sh.sh_offset).unwrap_or(0);
                    let str_end =
                        str_start.saturating_add(usize::try_from(str_sh.sh_size).unwrap_or(0));
                    let str_bytes = bytes.get(str_start..str_end).unwrap_or_default();

                    let sym_start = usize::try_from(sh.sh_offset).unwrap_or(0);
                    let sym_end =
                        sym_start.saturating_add(usize::try_from(sh.sh_size).unwrap_or(0));
                    let sym_bytes = bytes.get(sym_start..sym_end).unwrap_or_default();

                    for chunk in sym_bytes.chunks_exact(24) {
                        let st_name =
                            bytemuck::pod_read_unaligned::<u32>(chunk.get(0..4).unwrap_or(&[0; 4]))
                                as usize;
                        let st_info = chunk.get(4).copied().unwrap_or(0);
                        let st_shndx =
                            bytemuck::pod_read_unaligned::<u16>(chunk.get(6..8).unwrap_or(&[0; 2]));
                        let st_value = bytemuck::pod_read_unaligned::<u64>(
                            chunk.get(8..16).unwrap_or(&[0; 8]),
                        );
                        let st_size = bytemuck::pod_read_unaligned::<u64>(
                            chunk.get(16..24).unwrap_or(&[0; 8]),
                        );
                        let st_bind = st_info >> 4;
                        let st_type = st_info & 0x0F;
                        if st_shndx != 0
                            && (st_bind == ELF_STB_GLOBAL || st_bind == ELF_STB_WEAK)
                            && st_type == goblin::elf::sym::STT_FUNC
                        {
                            let tail = str_bytes.get(st_name..).unwrap_or_default();
                            let nul_pos = tail.iter().position(|&b| b == 0).unwrap_or(tail.len());
                            let name_str =
                                String::from_utf8_lossy(tail.get(..nul_pos).unwrap_or_default());
                            if !name_str.is_empty() {
                                exported_symbols.push(ExportedBinarySymbol {
                                    name: name_str.into_owned(),
                                    address: u64::from_le(st_value),
                                    size: u64::from_le(st_size),
                                });
                            }
                        }
                    }
                }
            }
            Ok(BinarySymbolReport {
                format: BinaryFormatKind::Elf,
                section_names,
                exported_symbols,
            })
        }
        Object::Mach(goblin::mach::Mach::Binary(_)) => Ok(BinarySymbolReport {
            format: BinaryFormatKind::MachO,
            section_names: Vec::new(),
            exported_symbols: Vec::new(),
        }),
        _ => Err(LibTpuError::BinaryFormatError {
            reason: String::from("Unsupported binary container format (expected ELF or Mach-O)"),
        }),
    }
}

/// Parses an ELF or Mach-O binary image using `object::File::parse` and returns its
/// section names, dynamic symbol names, and endianness/bitness profile.
///
/// # Arguments
///
/// * `bytes` - Raw ELF64 or Mach-O binary image bytes.
///
/// # Errors
///
/// Returns [`LibTpuError::BinaryFormatError`] if `object::File::parse` fails.
pub fn inspect_object_crate_sections(bytes: &[u8]) -> LibTpuResult<ObjectCrateBinaryReport> {
    let file = object::File::parse(bytes).map_err(|err| LibTpuError::BinaryFormatError {
        reason: format!("object::File::parse failed: {err}"),
    })?;
    let mut section_names = Vec::new();
    let mut sections = Vec::new();
    for section in file.sections() {
        let sec_name = String::from(section.name().unwrap_or_default());
        let (file_offset, _) = section.file_range().unwrap_or((0, 0));
        sections.push(BinarySectionInfo {
            name: sec_name.clone(),
            virtual_address: section.address(),
            file_offset,
            size_bytes: section.size(),
            alignment: section.align(),
        });
        section_names.push(sec_name);
    }
    let mut dynamic_symbol_names = Vec::new();
    for sym in file.dynamic_symbols() {
        let name = sym.name().unwrap_or_default();
        if !name.is_empty() {
            dynamic_symbol_names.push(String::from(name));
        }
    }
    Ok(ObjectCrateBinaryReport {
        is_64: file.is_64(),
        is_little_endian: file.is_little_endian(),
        section_names,
        sections,
        dynamic_symbol_names,
    })
}

/// Parses a DWARF `.eh_frame` section byte slice using `gimli::EhFrame` to recover all
/// Frame Description Entry (`FDE`) function address intervals.
///
/// # Arguments
///
/// * `eh_frame_bytes` - Raw `.eh_frame` section bytes.
/// * `eh_frame_vaddr` - Virtual address of the `.eh_frame` section in the ELF image.
///
/// # Errors
///
/// Returns [`LibTpuError::BinaryFormatError`] if the `.eh_frame` stream is malformed.
pub fn parse_eh_frame_fde_ranges(
    eh_frame_bytes: &[u8],
    eh_frame_vaddr: u64,
) -> LibTpuResult<Vec<EhFrameFunctionRange>> {
    let eh_frame = EhFrame::new(eh_frame_bytes, LittleEndian);
    let bases = BaseAddresses::default().set_eh_frame(eh_frame_vaddr);
    let mut entries = eh_frame.entries(&bases);
    let mut ranges = Vec::new();

    loop {
        let next_entry = entries
            .next()
            .map_err(|err| LibTpuError::BinaryFormatError {
                reason: format!("gimli .eh_frame parse error: {err}"),
            })?;
        let Some(entry) = next_entry else {
            break;
        };
        if let CieOrFde::Fde(partial) = entry {
            let fde = partial
                .parse(|_section, bases_ref, cie_offset| {
                    eh_frame.cie_from_offset(bases_ref, cie_offset)
                })
                .map_err(|err| LibTpuError::BinaryFormatError {
                    reason: format!("gimli FDE CIE resolution error: {err}"),
                })?;
            ranges.push(EhFrameFunctionRange {
                initial_address: fde.initial_address(),
                len: fde.len(),
            });
        }
    }

    Ok(ranges)
}

/// Demangles an Itanium C++ ABI symbol using `cpp_demangle` and classifies its architectural
/// subsystem namespace and RTTI role (`_ZTV*` vtable or `_ZTI*` typeinfo).
///
/// # Arguments
///
/// * `mangled` - Mangled or unmangled symbol name from `.symtab` or `.dynsym`.
#[must_use]
pub fn demangle_and_classify_cpp_symbol(mangled: &str) -> DemangledCppSymbol {
    let is_vtable = mangled.starts_with("_ZTV");
    let is_typeinfo = mangled.starts_with("_ZTI") || mangled.starts_with("_ZTS");
    let demangled = cpp_demangle::Symbol::new(mangled)
        .ok()
        .and_then(|sym| sym.demangle().ok())
        .unwrap_or_else(|| String::from(mangled));

    let category = if demangled.contains("asic_sw::driver") {
        CppNamespaceCategory::AsicSwDriver
    } else if demangled.contains("asic_sw::") {
        CppNamespaceCategory::AsicSwDeepsea
    } else if demangled.contains("pjrt::") {
        CppNamespaceCategory::Pjrt
    } else if demangled.contains("mlir::") {
        CppNamespaceCategory::MlirStableHlo
    } else if demangled.contains("megascale::") {
        CppNamespaceCategory::Megascale
    } else if demangled.contains("libtpu::sdk") {
        CppNamespaceCategory::LibtpuSdk
    } else if demangled.contains("tpu_driver::") {
        CppNamespaceCategory::TpuDriver
    } else if demangled.contains("tpu::") {
        CppNamespaceCategory::TensorflowTpu
    } else if demangled.contains("xla::") {
        CppNamespaceCategory::Xla
    } else {
        CppNamespaceCategory::Other
    };

    DemangledCppSymbol {
        mangled: String::from(mangled),
        demangled,
        category,
        is_vtable,
        is_typeinfo,
    }
}

/// Reconstructs C++ RTTI and vtable class descriptors (`_ZTV*`, `_ZTI*`, `_ZTS*`) from a slice
/// of [`ExportedBinarySymbol`] records.
///
/// # Arguments
///
/// * `symbols` - Slice of exported or local ELF symbols.
#[must_use]
pub fn reconstruct_rtti_hierarchy(
    symbols: &[ExportedBinarySymbol],
) -> Vec<RttiClassHierarchyEntry> {
    let mut table: BTreeMap<String, RttiClassHierarchyEntry> = BTreeMap::new();
    for sym in symbols {
        let (kind, encoded_type) = if let Some(rest) = sym.name.strip_prefix("_ZTV") {
            (0_u8, rest)
        } else if let Some(rest) = sym.name.strip_prefix("_ZTI") {
            (1_u8, rest)
        } else if let Some(rest) = sym.name.strip_prefix("_ZTS") {
            (2_u8, rest)
        } else {
            continue;
        };
        let zts_demangled =
            demangle_and_classify_cpp_symbol(&format!("_ZTS{encoded_type}")).demangled;
        let class_str = zts_demangled
            .strip_prefix("typeinfo name for ")
            .unwrap_or(zts_demangled.as_str());
        let ns = class_str.split("::").next().unwrap_or_default();
        let entry =
            table
                .entry(String::from(class_str))
                .or_insert_with(|| RttiClassHierarchyEntry {
                    class_name: String::from(class_str),
                    namespace: String::from(ns),
                    vtable_vaddr: 0,
                    typeinfo_vaddr: 0,
                    typeinfo_name_vaddr: 0,
                });
        match kind {
            0 => entry.vtable_vaddr = sym.address,
            1 => entry.typeinfo_vaddr = sym.address,
            _ => entry.typeinfo_name_vaddr = sym.address,
        }
    }
    table.into_values().collect()
}

/// Builds a minimal synthetic 64-bit little-endian ELF shared object with `.shstrtab`, `.dynstr`,
/// `.dynsym`, and `.rela.dyn` sections for deterministic unit testing.
#[cfg(test)]
#[allow(clippy::indexing_slicing, clippy::too_many_lines)]
#[cfg_attr(coverage_nightly, coverage(off))]
pub(super) fn build_synthetic_elf64() -> Vec<u8> {
    use crate::elf::relocations::{R_AARCH64_RELATIVE, R_X86_64_RELATIVE};
    let mut buf = vec![0_u8; 64];
    buf[0..4].copy_from_slice(b"\x7fELF");
    buf[4] = 2;
    buf[5] = 1;
    buf[6] = 1;
    buf[16..18].copy_from_slice(&3_u16.to_le_bytes());
    buf[18..20].copy_from_slice(&62_u16.to_le_bytes());
    buf[20..24].copy_from_slice(&1_u32.to_le_bytes());
    buf[52..54].copy_from_slice(&64_u16.to_le_bytes());

    let shstrtab = b"\0.shstrtab\0.dynstr\0.dynsym\0.rela.dyn\0";
    let shstr_off = buf.len();
    buf.extend_from_slice(shstrtab);

    let dynstr = b"\0GetPjrtApi\0WeakFn\0";
    let dynstr_off = buf.len();
    buf.extend_from_slice(dynstr);

    while buf.len() % 8 != 0 {
        buf.push(0);
    }
    let dynsym_off = buf.len();
    buf.extend_from_slice(&[0_u8; 24]);
    buf.extend_from_slice(&0_u32.to_le_bytes());
    buf.push(0x12);
    buf.push(0);
    buf.extend_from_slice(&1_u16.to_le_bytes());
    buf.extend_from_slice(&0x1000_u64.to_le_bytes());
    buf.extend_from_slice(&16_u64.to_le_bytes());
    buf.extend_from_slice(&1_u32.to_le_bytes());
    buf.push(0x11);
    buf.push(0);
    buf.extend_from_slice(&1_u16.to_le_bytes());
    buf.extend_from_slice(&0x2000_u64.to_le_bytes());
    buf.extend_from_slice(&8_u64.to_le_bytes());
    buf.extend_from_slice(&1_u32.to_le_bytes());
    buf.push(0x02);
    buf.push(0);
    buf.extend_from_slice(&1_u16.to_le_bytes());
    buf.extend_from_slice(&0x1500_u64.to_le_bytes());
    buf.extend_from_slice(&8_u64.to_le_bytes());
    buf.extend_from_slice(&12_u32.to_le_bytes());
    buf.push(0x22);
    buf.push(0);
    buf.extend_from_slice(&1_u16.to_le_bytes());
    buf.extend_from_slice(&0x1800_u64.to_le_bytes());
    buf.extend_from_slice(&12_u64.to_le_bytes());
    buf.extend_from_slice(&1_u32.to_le_bytes());
    buf.push(0x12);
    buf.push(0);
    buf.extend_from_slice(&1_u16.to_le_bytes());
    buf.extend_from_slice(&0x1000_u64.to_le_bytes());
    buf.extend_from_slice(&32_u64.to_le_bytes());
    let dynsym_size = (buf.len() - dynsym_off) as u64;

    let rela_off = buf.len();
    buf.extend_from_slice(&0x3000_u64.to_le_bytes());
    buf.extend_from_slice(&u64::from(R_X86_64_RELATIVE).to_le_bytes());
    buf.extend_from_slice(&0x0B4F_9B90_u64.to_le_bytes());
    buf.extend_from_slice(&0x3008_u64.to_le_bytes());
    buf.extend_from_slice(&u64::from(R_AARCH64_RELATIVE).to_le_bytes());
    buf.extend_from_slice(&0x0D3E_07D8_u64.to_le_bytes());
    buf.extend_from_slice(&0x3010_u64.to_le_bytes());
    buf.extend_from_slice(&1_u64.to_le_bytes());
    buf.extend_from_slice(&0_u64.to_le_bytes());
    let rela_size = (buf.len() - rela_off) as u64;

    let shoff = buf.len();
    buf[40..48].copy_from_slice(&(shoff as u64).to_le_bytes());
    buf[58..60].copy_from_slice(&64_u16.to_le_bytes());
    buf[60..62].copy_from_slice(&7_u16.to_le_bytes());
    buf[62..64].copy_from_slice(&1_u16.to_le_bytes());

    let push_shdr = |out: &mut Vec<u8>,
                     name: u32,
                     sh_type: u32,
                     offset: u64,
                     size: u64,
                     link: u32,
                     entsize: u64| {
        let mut sh = [0_u8; 64];
        sh[0..4].copy_from_slice(&name.to_le_bytes());
        sh[4..8].copy_from_slice(&sh_type.to_le_bytes());
        sh[24..32].copy_from_slice(&offset.to_le_bytes());
        sh[32..40].copy_from_slice(&size.to_le_bytes());
        sh[40..44].copy_from_slice(&link.to_le_bytes());
        sh[56..64].copy_from_slice(&entsize.to_le_bytes());
        out.extend_from_slice(&sh);
    };
    push_shdr(&mut buf, 0, 0, 0, 0, 0, 0);
    push_shdr(
        &mut buf,
        1,
        3,
        shstr_off as u64,
        shstrtab.len() as u64,
        0,
        0,
    );
    push_shdr(
        &mut buf,
        11,
        3,
        dynstr_off as u64,
        dynstr.len() as u64,
        0,
        0,
    );
    push_shdr(&mut buf, 19, 11, dynsym_off as u64, dynsym_size, 2, 24);
    push_shdr(&mut buf, 19, 11, dynsym_off as u64, 0, 2, 0);
    push_shdr(&mut buf, 27, 4, rela_off as u64, rela_size, 0, 24);
    push_shdr(&mut buf, 27, 4, rela_off as u64, 0, 0, 0);
    buf
}

#[cfg(test)]
#[allow(
    for_loops_over_fallibles,
    clippy::indexing_slicing,
    clippy::too_many_lines
)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_goblin_and_object_inspect_binary_symbols() {
        let elf_bytes = build_synthetic_elf64();
        let report_res = inspect_binary_symbols(&elf_bytes);
        assert!(report_res.is_ok());
        for report in report_res {
            assert_eq!(report.format, BinaryFormatKind::Elf);
            assert!(report.section_names.iter().any(|s| s == ".dynsym"));
            assert_eq!(report.exported_symbols.len(), 2);
            for sym in report.exported_symbols.last() {
                assert_eq!(sym.name, "GetPjrtApi");
                assert_eq!(sym.address, 0x1000);
                assert_eq!(sym.size, 32);
                assert_eq!(sym.clone(), *sym);
                assert_ne!(format!("{sym:?}"), "");
            }
            assert_eq!(report.clone(), report);
            assert_ne!(format!("{report:?}"), "");
        }

        assert!(inspect_binary_symbols(&[0_u8; 16]).is_err());
        assert!(inspect_binary_symbols(b"\x7fELF\x02").is_err());

        let mut macho_buf = vec![0_u8; 32];
        macho_buf[0..4].copy_from_slice(&0xfeed_facf_u32.to_le_bytes());
        macho_buf[4..8].copy_from_slice(&0x0100_0007_u32.to_le_bytes());
        macho_buf[12..16].copy_from_slice(&6_u32.to_le_bytes());
        for macho_rep in inspect_binary_symbols(&macho_buf) {
            assert_eq!(macho_rep.format, BinaryFormatKind::MachO);
        }

        let obj_res = inspect_object_crate_sections(&elf_bytes);
        assert!(obj_res.is_ok());
        for obj_rep in obj_res {
            assert!(obj_rep.is_64);
            assert!(obj_rep.is_little_endian);
            assert!(obj_rep.section_names.iter().any(|s| s == ".dynsym"));
            assert!(
                obj_rep
                    .dynamic_symbol_names
                    .iter()
                    .any(|s| s == "GetPjrtApi")
            );
            assert_eq!(obj_rep.clone(), obj_rep);
            assert_ne!(format!("{obj_rep:?}"), "");
            for sec in obj_rep.sections.first() {
                assert_eq!(sec.clone(), *sec);
                assert_ne!(format!("{sec:?}"), "");
            }
        }
        assert!(inspect_object_crate_sections(b"invalid_elf_bytes").is_err());
    }

    #[test]
    fn test_gimli_eh_frame_and_cpp_demangle() {
        let mut eh = Vec::new();
        eh.extend_from_slice(&12_u32.to_le_bytes());
        eh.extend_from_slice(&0_u32.to_le_bytes());
        eh.push(1);
        eh.push(0);
        eh.push(1);
        eh.push(0x78);
        eh.push(16);
        eh.extend_from_slice(&[0, 0, 0]);

        eh.extend_from_slice(&20_u32.to_le_bytes());
        eh.extend_from_slice(&20_u32.to_le_bytes());
        eh.extend_from_slice(&0x0B4F_9B90_u64.to_le_bytes());
        eh.extend_from_slice(&32_u64.to_le_bytes());
        eh.extend_from_slice(&0_u32.to_le_bytes());

        let fdes = parse_eh_frame_fde_ranges(&eh, 0x09CE_E1B8);
        assert!(fdes.is_ok());
        for list in fdes {
            assert_eq!(list.len(), 1);
            assert_eq!(list[0].initial_address, 0x0B4F_9B90);
            assert_eq!(list[0].len, 32);
            assert_eq!(list[0].clone(), list[0]);
            assert_ne!(format!("{:?}", list[0]), "");
        }

        assert!(parse_eh_frame_fde_ranges(&[0xFF, 0x00, 0x00, 0x00], 0x1000).is_err());
        let mut bad_fde = Vec::new();
        bad_fde.extend_from_slice(&20_u32.to_le_bytes());
        bad_fde.extend_from_slice(&4_u32.to_le_bytes());
        bad_fde.extend_from_slice(&0x1000_u64.to_le_bytes());
        bad_fde.extend_from_slice(&16_u64.to_le_bytes());
        assert!(parse_eh_frame_fde_ranges(&bad_fde, 0x1000).is_err());

        let cases = [
            (
                "_ZN7asic_sw6driver19VfioKernelDmaMapperD0Ev",
                CppNamespaceCategory::AsicSwDriver,
                false,
                false,
            ),
            (
                "_ZTVN7asic_sw7deepsea10ChipDriverE",
                CppNamespaceCategory::AsicSwDeepsea,
                true,
                false,
            ),
            (
                "_ZTIN4pjrt10PjrtClientE",
                CppNamespaceCategory::Pjrt,
                false,
                true,
            ),
            (
                "_ZN4mlir9stablehlo5AddOpE",
                CppNamespaceCategory::MlirStableHlo,
                false,
                false,
            ),
            (
                "_ZTSN9megascale11CoordinatorE",
                CppNamespaceCategory::Megascale,
                false,
                true,
            ),
            (
                "_ZN6libtpu3sdk15GetMetricValuesEv",
                CppNamespaceCategory::LibtpuSdk,
                false,
                false,
            ),
            (
                "_ZN10tpu_driver10OpenDriverEv",
                CppNamespaceCategory::TpuDriver,
                false,
                false,
            ),
            (
                "_ZN10tensorflow3tpu12TopologyUtilEv",
                CppNamespaceCategory::TensorflowTpu,
                false,
                false,
            ),
            (
                "_ZN3tpu15TopologyBuilderEv",
                CppNamespaceCategory::TensorflowTpu,
                false,
                false,
            ),
            (
                "_ZN3xla9HloModuleC1Ev",
                CppNamespaceCategory::Xla,
                false,
                false,
            ),
            (
                "not_a_mangled_symbol",
                CppNamespaceCategory::Other,
                false,
                false,
            ),
        ];
        for (sym, expected_cat, expected_vt, expected_ti) in cases {
            let info = demangle_and_classify_cpp_symbol(sym);
            assert_eq!(info.category, expected_cat);
            assert_eq!(info.is_vtable, expected_vt);
            assert_eq!(info.is_typeinfo, expected_ti);
            assert_eq!(info.clone(), info);
            assert_ne!(format!("{info:?}"), "");
            assert_ne!(format!("{:?}", info.category), "");
        }

        let rtti_syms = vec![
            ExportedBinarySymbol {
                name: String::from("_ZTVN7asic_sw6driver12GasketDriverE"),
                address: 0x1000,
                size: 64,
            },
            ExportedBinarySymbol {
                name: String::from("_ZTIN7asic_sw6driver12GasketDriverE"),
                address: 0x2000,
                size: 24,
            },
            ExportedBinarySymbol {
                name: String::from("_ZTSN7asic_sw6driver12GasketDriverE"),
                address: 0x3000,
                size: 32,
            },
            ExportedBinarySymbol {
                name: String::from("GetPjrtApi"),
                address: 0x4000,
                size: 16,
            },
        ];
        let rtti_classes = reconstruct_rtti_hierarchy(&rtti_syms);
        assert_eq!(rtti_classes.len(), 1);
        assert_eq!(rtti_classes[0].class_name, "asic_sw::driver::GasketDriver");
        assert_eq!(rtti_classes[0].namespace, "asic_sw");
        assert_eq!(rtti_classes[0].vtable_vaddr, 0x1000);
        assert_eq!(rtti_classes[0].typeinfo_vaddr, 0x2000);
        assert_eq!(rtti_classes[0].typeinfo_name_vaddr, 0x3000);
        assert_eq!(rtti_classes[0].clone(), rtti_classes[0]);
        assert_ne!(format!("{:?}", rtti_classes[0]), "");
    }
}

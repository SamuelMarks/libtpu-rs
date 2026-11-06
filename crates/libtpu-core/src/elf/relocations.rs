//! ELF64 `.rela.dyn` relocation resolver and static C/C++ vtable reconstructors.
//!
//! Provides:
//! - [`resolve_relative_relocations`] and [`ElfRelativeRelocation`] for `R_X86_64_RELATIVE`
//!   and `R_AARCH64_RELATIVE` relocations.
//! - [`reconstruct_pjrt_vtable`], [`ReconstructedPjrtVtable`], and [`ExtractedExtensionNode`]
//!   for `PJRT_Api` and the 17-node `PJRT_Extension_Base` linked list.
//! - [`reconstruct_libtpu_sdk_vtable`], [`reconstruct_tpu_driver_vtable`], and [`ReconstructedFlatApiVtable`]
//!   for `LibtpuSdk_Api` (`144` bytes) and `TpuDriverFn` (`224` bytes).
//! - [`reconstruct_itanium_cpp_vtable`] and [`ReconstructedCppVtable`] for Itanium C++ ABI
//!   `_ZTV*` virtual method tables and `_ZTI*` / `_ZTS*` RTTI descriptors.

use crate::elf::symbols::demangle_and_classify_cpp_symbol;
use crate::error::{LibTpuError, LibTpuResult};
use goblin::Object;
use std::collections::BTreeMap;

/// x86-64 relative relocation type (`R_X86_64_RELATIVE = 8`).
pub const R_X86_64_RELATIVE: u32 = 8;

/// `AArch64` relative relocation type (`R_AARCH64_RELATIVE = 1027`).
pub const R_AARCH64_RELATIVE: u32 = 1027;

/// Resolved `.rela.dyn` relative relocation entry (`*r_offset = r_addend`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ElfRelativeRelocation {
    /// Target virtual address (`r_offset`) where the relocated pointer is stored.
    pub offset: u64,
    /// Relocation type (`R_X86_64_RELATIVE` or `R_AARCH64_RELATIVE`).
    pub r_type: u32,
    /// Resolved target virtual address (`r_addend`).
    pub addend: u64,
}

/// Extracted `PJRT_Extension_Base` node from a static vtable chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractedExtensionNode {
    /// Virtual address of the `PJRT_Extension_Base` header.
    pub vaddr: u64,
    /// Byte size of the extension structure (`struct_size` at `+0x00`).
    pub struct_size: u64,
    /// Numeric `PJRT_Extension_Type` discriminant (`type` at `+0x08`).
    pub ext_type: u32,
    /// Pointer to the next extension node in the linked list (`next` at `+0x10`).
    pub next_vaddr: u64,
    /// Resolved function pointer virtual addresses starting at `+0x18`.
    pub method_vaddrs: Vec<u64>,
}

/// Reconstructed `PJRT_Api` vtable and extension linked list from relocated binary words.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReconstructedPjrtVtable {
    /// Total byte size of `PJRT_Api` (`1,144` bytes in `libtpu-0.0.52`).
    pub struct_size: u64,
    /// Head virtual address of the `PJRT_Extension_Base` linked list (`+0x08`).
    pub extension_start_vaddr: u64,
    /// Embedded `PJRT_Api_Version.struct_size` (`24` bytes at `+0x10`).
    pub version_struct_size: u64,
    /// Embedded `PJRT_Api_Version.major_version` (`0` at `+0x18`).
    pub major_version: i32,
    /// Embedded `PJRT_Api_Version.minor_version` (`116` at `+0x1c`).
    pub minor_version: i32,
    /// Function pointer virtual addresses for all `PJRT_Api` slots (`+0x28..`).
    pub slot_vaddrs: Vec<u64>,
    /// Walked `PJRT_Extension_Base` linked list nodes.
    pub extensions: Vec<ExtractedExtensionNode>,
}

/// Reconstructed Itanium C++ ABI virtual method table (`_ZTV*`) and RTTI type descriptor (`_ZTI*`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReconstructedCppVtable {
    /// Virtual address of the `_ZTV*` symbol header (`offset_to_top` slot).
    pub vtable_vaddr: u64,
    /// Signed offset-to-top word at `vtable_vaddr + 0x00` (`0` for primary inheritance).
    pub offset_to_top: i64,
    /// Virtual address of the `_ZTI*` `std::type_info` structure at `vtable_vaddr + 0x08`.
    pub typeinfo_vaddr: u64,
    /// Demangled C++ class name resolved from the `_ZTS*` typeinfo string at `typeinfo_vaddr + 0x08`.
    pub class_name: String,
    /// Ordered virtual method function pointer addresses starting at `vtable_vaddr + 0x10`.
    pub method_vaddrs: Vec<u64>,
}

/// Reconstructed `LibtpuSdk_Api` (`144` bytes, 17 slots) or `TpuDriverFn` (`224` bytes, 28 slots) C-API vtable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReconstructedFlatApiVtable {
    /// Virtual address of the C-API table in `.data.rel.ro` or `.bss`.
    pub base_vaddr: u64,
    /// Total structure byte size (`144` for `LibtpuSdk_Api`, `224` for `TpuDriverFn`).
    pub struct_size: u64,
    /// Extracted function pointer virtual addresses in slot order.
    pub slot_vaddrs: Vec<u64>,
}

/// Classification of an Itanium C++ ABI RTTI descriptor (`__cxxabiv1::*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItaniumRttiKind {
    /// Root class with no base classes (`__cxxabiv1::__class_type_info`).
    ClassTypeInfo,
    /// Single public non-virtual inheritance (`__cxxabiv1::__si_class_type_info`).
    SingleInheritance,
    /// Virtual or multiple inheritance (`__cxxabiv1::__vmi_class_type_info`).
    VirtualMultipleInheritance,
}

/// Complete Itanium C++ ABI RTTI and vtable inheritance node (`ItaniumRttiClassDescriptor`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ItaniumRttiClassDescriptor {
    /// Virtual address of the `_ZTI*` typeinfo structure in `.data.rel.ro`.
    pub typeinfo_vaddr: u64,
    /// Virtual address of the associated `_ZTV*` vtable, if discovered.
    pub vtable_vaddr: u64,
    /// Inheritance kind (`ClassTypeInfo`, `SingleInheritance`, or `VirtualMultipleInheritance`).
    pub kind: ItaniumRttiKind,
    /// Demangled C++ class name (`"asic_sw::driver::VfioDriver"`, etc.).
    pub class_name: String,
    /// Top-level C++ namespace (`"asic_sw"`, `"xla"`, `"pjrt"`, `"megascale"`).
    pub namespace: String,
    /// Ordered list of base class `_ZTI*` virtual addresses (`1` for SI, `N` for VMI).
    pub base_typeinfo_vaddrs: Vec<u64>,
    /// Ordered list of demangled base class names resolved from `base_typeinfo_vaddrs`.
    pub base_class_names: Vec<String>,
    /// Virtual method function pointer addresses extracted from the associated `_ZTV*`.
    pub virtual_method_vaddrs: Vec<u64>,
}

/// Reconstructor for Itanium C++ ABI RTTI inheritance graphs (`CppRttiHierarchyExtractor`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CppRttiHierarchyExtractor;

impl CppRttiHierarchyExtractor {
    /// Resolves a demangled C++ class name from a `_ZTS*` string virtual address in `rodata_blobs`.
    ///
    /// # Arguments
    ///
    /// * `rodata_blobs` - Mapping from `.rodata` virtual address to NUL-terminated `_ZTS*` string bytes.
    /// * `zts_vaddr` - Virtual address of the `_ZTS*` string in `.rodata`.
    fn resolve_zts_name(rodata_blobs: &BTreeMap<u64, Vec<u8>>, zts_vaddr: u64) -> String {
        let raw_zts = rodata_blobs
            .get(&zts_vaddr)
            .map(|bytes| {
                let nul_pos = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
                String::from_utf8_lossy(bytes.get(..nul_pos).unwrap_or_default()).into_owned()
            })
            .unwrap_or_default();
        if raw_zts.is_empty() {
            return String::new();
        }
        let mangled = if raw_zts.starts_with("_ZTS") {
            raw_zts
        } else {
            format!("_ZTS{raw_zts}")
        };
        let demangled = demangle_and_classify_cpp_symbol(&mangled).demangled;
        String::from(
            demangled
                .strip_prefix("typeinfo name for ")
                .unwrap_or(demangled.as_str()),
        )
    }

    /// Scans a relocated 64-bit word map (`word_map`) and `.rodata` string map (`rodata_blobs`)
    /// to reconstruct the full C++ RTTI class hierarchy (`__class_type_info`,
    /// `__si_class_type_info`, and `__vmi_class_type_info`) and associate `_ZTV*` vtables.
    ///
    /// # Arguments
    ///
    /// * `word_map` - Mapping from 8-byte-aligned `.data.rel.ro` virtual address to relocated 64-bit word.
    /// * `rodata_blobs` - Mapping from `.rodata` virtual address to NUL-terminated `_ZTS*` string bytes.
    /// * `class_ti_vtable` - Virtual address of `_ZTVN10__cxxabiv117__class_type_infoE + 16`.
    /// * `si_ti_vtable` - Virtual address of `_ZTVN10__cxxabiv120__si_class_type_infoE + 16`.
    /// * `vmi_ti_vtable` - Virtual address of `_ZTVN10__cxxabiv121__vmi_class_type_infoE + 16`.
    #[must_use]
    pub fn extract_hierarchy(
        word_map: &BTreeMap<u64, u64>,
        rodata_blobs: &BTreeMap<u64, Vec<u8>>,
        class_ti_vtable: u64,
        si_ti_vtable: u64,
        vmi_ti_vtable: u64,
    ) -> Vec<ItaniumRttiClassDescriptor> {
        let mut ti_names: BTreeMap<u64, String> = BTreeMap::new();
        let mut raw_nodes: Vec<(u64, ItaniumRttiKind, String, Vec<u64>)> = Vec::new();

        for (&vaddr, &word0) in word_map {
            let kind_opt = if word0 == class_ti_vtable {
                Some(ItaniumRttiKind::ClassTypeInfo)
            } else if word0 == si_ti_vtable {
                Some(ItaniumRttiKind::SingleInheritance)
            } else if word0 == vmi_ti_vtable {
                Some(ItaniumRttiKind::VirtualMultipleInheritance)
            } else {
                None
            };
            let Some(kind) = kind_opt else {
                continue;
            };
            let zts_ptr = word_map.get(&vaddr.wrapping_add(8)).copied().unwrap_or(0);
            let class_name = Self::resolve_zts_name(rodata_blobs, zts_ptr);
            if class_name.is_empty() {
                continue;
            }
            let mut bases = Vec::new();
            match kind {
                ItaniumRttiKind::ClassTypeInfo => {}
                ItaniumRttiKind::SingleInheritance => {
                    let base_ti = word_map.get(&vaddr.wrapping_add(16)).copied().unwrap_or(0);
                    bases.push(base_ti);
                }
                ItaniumRttiKind::VirtualMultipleInheritance => {
                    let flags_and_count =
                        word_map.get(&vaddr.wrapping_add(16)).copied().unwrap_or(0);
                    let base_count = ((flags_and_count >> 32) & 0xFFFF) as usize;
                    for idx in 0..base_count.min(16) {
                        let base_off = 24_u64.wrapping_add((idx as u64).wrapping_mul(16));
                        let base_ti = word_map
                            .get(&vaddr.wrapping_add(base_off))
                            .copied()
                            .unwrap_or(0);
                        bases.push(base_ti);
                    }
                }
            }
            ti_names.insert(vaddr, class_name.clone());
            raw_nodes.push((vaddr, kind, class_name, bases));
        }

        let mut ti_to_vtable: BTreeMap<u64, (u64, Vec<u64>)> = BTreeMap::new();
        for (&vaddr, &word0) in word_map {
            let ti_ptr = word_map.get(&vaddr.wrapping_add(8)).copied().unwrap_or(0);
            if word0 == 0 && ti_names.contains_key(&ti_ptr) {
                let mut methods = Vec::new();
                for m_idx in 0..8_u64 {
                    let m_vaddr = vaddr.wrapping_add(16).wrapping_add(m_idx.wrapping_mul(8));
                    let fn_ptr = word_map.get(&m_vaddr).copied().unwrap_or(0);
                    if fn_ptr != 0 {
                        methods.push(fn_ptr);
                    }
                }
                ti_to_vtable.insert(ti_ptr, (vaddr, methods));
            }
        }

        raw_nodes
            .into_iter()
            .map(|(ti_vaddr, kind, class_name, base_typeinfo_vaddrs)| {
                let namespace = String::from(class_name.split("::").next().unwrap_or_default());
                let base_class_names = base_typeinfo_vaddrs
                    .iter()
                    .filter_map(|b| ti_names.get(b).cloned())
                    .collect();
                let (vtable_vaddr, virtual_method_vaddrs) = ti_to_vtable
                    .get(&ti_vaddr)
                    .cloned()
                    .unwrap_or((0, Vec::new()));
                ItaniumRttiClassDescriptor {
                    typeinfo_vaddr: ti_vaddr,
                    vtable_vaddr,
                    kind,
                    class_name,
                    namespace,
                    base_typeinfo_vaddrs,
                    base_class_names,
                    virtual_method_vaddrs,
                }
            })
            .collect()
    }
}

/// Parses `.rela.dyn` (`SHT_RELA`, 24-byte `Elf64_Rela` entries) from an ELF64 image and
/// extracts all `R_X86_64_RELATIVE` (`8`) and `R_AARCH64_RELATIVE` (`1027`) relocations.
///
/// # Arguments
///
/// * `bytes` - Raw ELF64 binary bytes.
///
/// # Errors
///
/// Returns [`LibTpuError`] if `bytes` is not a valid ELF image.
pub fn resolve_relative_relocations(bytes: &[u8]) -> LibTpuResult<Vec<ElfRelativeRelocation>> {
    let obj = Object::parse(bytes)?;
    let Object::Elf(elf) = obj else {
        return Err(LibTpuError::BinaryFormatError {
            reason: String::from("Expected ELF64 binary for relocation resolution"),
        });
    };
    let mut relocations = Vec::new();
    for sh in &elf.section_headers {
        if sh.sh_type == goblin::elf::section_header::SHT_RELA && sh.sh_entsize == 24 {
            let start = usize::try_from(sh.sh_offset).unwrap_or(0);
            let end = start.saturating_add(usize::try_from(sh.sh_size).unwrap_or(0));
            let rela_bytes = bytes.get(start..end).unwrap_or_default();
            for chunk in rela_bytes.chunks_exact(24) {
                let r_offset = u64::from_le(bytemuck::pod_read_unaligned::<u64>(
                    chunk.get(0..8).unwrap_or(&[0; 8]),
                ));
                let r_info = u64::from_le(bytemuck::pod_read_unaligned::<u64>(
                    chunk.get(8..16).unwrap_or(&[0; 8]),
                ));
                let r_addend = u64::from_le(bytemuck::pod_read_unaligned::<u64>(
                    chunk.get(16..24).unwrap_or(&[0; 8]),
                ));
                #[allow(clippy::cast_possible_truncation)]
                let r_type = (r_info & 0xFFFF_FFFF) as u32;
                let is_rel = matches!(r_type, R_X86_64_RELATIVE | R_AARCH64_RELATIVE);
                if is_rel {
                    relocations.push(ElfRelativeRelocation {
                        offset: r_offset,
                        r_type,
                        addend: r_addend,
                    });
                }
            }
        }
    }
    Ok(relocations)
}

/// Reconstructs a `PJRT_Api` vtable and its `PJRT_Extension_Base` linked list from a virtual
/// address -> 64-bit word memory map (populated from `.data.rel.ro` + `.rela.dyn` relocations).
///
/// # Arguments
///
/// * `word_map` - Mapping from 8-byte-aligned virtual address to 64-bit word value.
/// * `pjrt_api_vaddr` - Base virtual address of the `PJRT_Api` structure.
/// * `num_slots` - Number of function pointer slots in `PJRT_Api` (e.g., `138`).
#[must_use]
pub fn reconstruct_pjrt_vtable(
    word_map: &BTreeMap<u64, u64>,
    pjrt_api_vaddr: u64,
    num_slots: usize,
) -> ReconstructedPjrtVtable {
    let struct_size = word_map.get(&pjrt_api_vaddr).copied().unwrap_or(0);
    let extension_start_vaddr = word_map
        .get(&pjrt_api_vaddr.wrapping_add(8))
        .copied()
        .unwrap_or(0);
    let version_struct_size = word_map
        .get(&pjrt_api_vaddr.wrapping_add(16))
        .copied()
        .unwrap_or(0);
    let version_packed = word_map
        .get(&pjrt_api_vaddr.wrapping_add(24))
        .copied()
        .unwrap_or(0);
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let major_version = (version_packed & 0xFFFF_FFFF) as i32;
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let minor_version = (version_packed >> 32) as i32;

    let mut slot_vaddrs = Vec::with_capacity(num_slots);
    for idx in 0..num_slots {
        let off = 0x28_u64.wrapping_add((idx as u64).wrapping_mul(8));
        let target = word_map
            .get(&pjrt_api_vaddr.wrapping_add(off))
            .copied()
            .unwrap_or(0);
        slot_vaddrs.push(target);
    }

    let mut extensions = Vec::new();
    let mut cur_ext = extension_start_vaddr;
    for _ in 0..32 {
        if cur_ext == 0 {
            break;
        }
        let ext_size = word_map.get(&cur_ext).copied().unwrap_or(0);
        #[allow(clippy::cast_possible_truncation)]
        let ext_type = word_map.get(&cur_ext.wrapping_add(8)).copied().unwrap_or(0) as u32;
        let next_vaddr = word_map
            .get(&cur_ext.wrapping_add(16))
            .copied()
            .unwrap_or(0);
        let method_words = usize::try_from(ext_size.saturating_sub(24) / 8).unwrap_or(0);
        let mut method_vaddrs = Vec::with_capacity(method_words);
        for m_idx in 0..method_words {
            let m_off = 24_u64.wrapping_add((m_idx as u64).wrapping_mul(8));
            let m_val = word_map
                .get(&cur_ext.wrapping_add(m_off))
                .copied()
                .unwrap_or(0);
            method_vaddrs.push(m_val);
        }
        extensions.push(ExtractedExtensionNode {
            vaddr: cur_ext,
            struct_size: ext_size,
            ext_type,
            next_vaddr,
            method_vaddrs,
        });
        cur_ext = next_vaddr;
    }

    ReconstructedPjrtVtable {
        struct_size,
        extension_start_vaddr,
        version_struct_size,
        major_version,
        minor_version,
        slot_vaddrs,
        extensions,
    }
}

/// Reconstructs an Itanium C++ ABI virtual method table (`_ZTV*`) and resolves its `_ZTI*` / `_ZTS*`
/// RTTI class name from relocated 64-bit words and `.rodata` string blobs.
///
/// # Arguments
///
/// * `word_map` - Mapping from 8-byte-aligned virtual address to relocated 64-bit word value.
/// * `rodata_blobs` - Mapping from `.rodata` virtual address to NUL-terminated `_ZTS*` string bytes.
/// * `vtable_vaddr` - Base virtual address of the `_ZTV*` symbol (`offset_to_top` slot).
/// * `num_methods` - Number of virtual method slots following the 16-byte Itanium ABI header.
#[must_use]
pub fn reconstruct_itanium_cpp_vtable(
    word_map: &BTreeMap<u64, u64>,
    rodata_blobs: &BTreeMap<u64, Vec<u8>>,
    vtable_vaddr: u64,
    num_methods: usize,
) -> ReconstructedCppVtable {
    #[allow(clippy::cast_possible_wrap)]
    let offset_to_top = word_map.get(&vtable_vaddr).copied().unwrap_or(0) as i64;
    let typeinfo_vaddr = word_map
        .get(&vtable_vaddr.wrapping_add(8))
        .copied()
        .unwrap_or(0);
    let zts_vaddr = word_map
        .get(&typeinfo_vaddr.wrapping_add(8))
        .copied()
        .unwrap_or(0);
    let raw_zts = rodata_blobs
        .get(&zts_vaddr)
        .map(|bytes| {
            let nul_pos = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
            String::from_utf8_lossy(bytes.get(..nul_pos).unwrap_or_default()).into_owned()
        })
        .unwrap_or_default();
    let class_name = if raw_zts.is_empty() {
        String::new()
    } else if raw_zts.starts_with("_ZTS") {
        demangle_and_classify_cpp_symbol(&raw_zts).demangled
    } else {
        demangle_and_classify_cpp_symbol(&format!("_ZTS{raw_zts}")).demangled
    };

    let mut method_vaddrs = Vec::with_capacity(num_methods);
    for idx in 0..num_methods {
        let off = 16_u64.wrapping_add((idx as u64).wrapping_mul(8));
        let fn_vaddr = word_map
            .get(&vtable_vaddr.wrapping_add(off))
            .copied()
            .unwrap_or(0);
        method_vaddrs.push(fn_vaddr);
    }

    ReconstructedCppVtable {
        vtable_vaddr,
        offset_to_top,
        typeinfo_vaddr,
        class_name,
        method_vaddrs,
    }
}

/// Reconstructs the 144-byte `LibtpuSdk_Api` table (`struct_size` at `+0x00` followed by 17 function pointers)
/// from a relocated virtual-address word map.
///
/// # Arguments
///
/// * `word_map` - Mapping from 8-byte-aligned virtual address to 64-bit word value.
/// * `sdk_api_vaddr` - Base virtual address of `LibtpuSdk_Api`.
#[must_use]
pub fn reconstruct_libtpu_sdk_vtable(
    word_map: &BTreeMap<u64, u64>,
    sdk_api_vaddr: u64,
) -> ReconstructedFlatApiVtable {
    let struct_size = word_map.get(&sdk_api_vaddr).copied().unwrap_or(0);
    let mut slot_vaddrs = Vec::with_capacity(17);
    for idx in 0..17_u64 {
        let off = 8_u64.wrapping_add(idx.wrapping_mul(8));
        slot_vaddrs.push(
            word_map
                .get(&sdk_api_vaddr.wrapping_add(off))
                .copied()
                .unwrap_or(0),
        );
    }
    ReconstructedFlatApiVtable {
        base_vaddr: sdk_api_vaddr,
        struct_size,
        slot_vaddrs,
    }
}

/// Reconstructs the 224-byte `TpuDriverFn` table (28 consecutive function pointers at `+0x00..+0xE0`)
/// from a relocated virtual-address word map.
///
/// # Arguments
///
/// * `word_map` - Mapping from 8-byte-aligned virtual address to 64-bit word value.
/// * `driver_fn_vaddr` - Base virtual address of `TpuDriverFn`.
#[must_use]
pub fn reconstruct_tpu_driver_vtable(
    word_map: &BTreeMap<u64, u64>,
    driver_fn_vaddr: u64,
) -> ReconstructedFlatApiVtable {
    let mut slot_vaddrs = Vec::with_capacity(28);
    for idx in 0..28_u64 {
        let off = idx.wrapping_mul(8);
        slot_vaddrs.push(
            word_map
                .get(&driver_fn_vaddr.wrapping_add(off))
                .copied()
                .unwrap_or(0),
        );
    }
    ReconstructedFlatApiVtable {
        base_vaddr: driver_fn_vaddr,
        struct_size: 224,
        slot_vaddrs,
    }
}

#[cfg(test)]
#[allow(clippy::indexing_slicing, clippy::too_many_lines)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::elf::symbols::build_synthetic_elf64;

    #[test]
    fn test_resolve_relative_relocations_and_vtable_reconstructors() {
        let elf_bytes = build_synthetic_elf64();
        let relocs = resolve_relative_relocations(&elf_bytes);
        assert!(relocs.is_ok());
        if let Ok(r_list) = relocs {
            assert_eq!(r_list.len(), 2);
            assert_eq!(r_list[0].offset, 0x3000);
            assert_eq!(r_list[0].r_type, R_X86_64_RELATIVE);
            assert_eq!(r_list[0].addend, 0x0B4F_9B90);
            assert_eq!(r_list[1].r_type, R_AARCH64_RELATIVE);
            assert_eq!(r_list[0].clone(), r_list[0]);
            assert_ne!(format!("{:?}", r_list[0]), "");
        }

        assert!(resolve_relative_relocations(b"\x7fELF\x02").is_err());
        let mut macho_buf = vec![0_u8; 32];
        macho_buf[0..4].copy_from_slice(&0xfeed_facf_u32.to_le_bytes());
        macho_buf[4..8].copy_from_slice(&0x0100_0007_u32.to_le_bytes());
        macho_buf[12..16].copy_from_slice(&6_u32.to_le_bytes());
        assert!(resolve_relative_relocations(&macho_buf).is_err());

        let mut word_map = BTreeMap::new();
        let base = 0x1000_0000_u64;
        let ext0 = 0x2000_0000_u64;
        word_map.insert(base, 1144);
        word_map.insert(base + 8, ext0);
        word_map.insert(base + 16, 24);
        word_map.insert(base + 24, 116_u64 << 32);
        word_map.insert(base + 0x28, 0x0C5A_0F90);
        word_map.insert(base + 0x30, 0x0C5A_1030);

        word_map.insert(ext0, 32);
        word_map.insert(ext0 + 8, 28);
        word_map.insert(ext0 + 16, 0);
        word_map.insert(ext0 + 24, 0x0C5B_0000);

        let vtable = reconstruct_pjrt_vtable(&word_map, base, 2);
        assert_eq!(vtable.struct_size, 1144);
        assert_eq!(vtable.major_version, 0);
        assert_eq!(vtable.minor_version, 116);
        assert_eq!(vtable.slot_vaddrs, vec![0x0C5A_0F90, 0x0C5A_1030]);
        assert_eq!(vtable.extensions.len(), 1);
        assert_eq!(vtable.extensions[0].ext_type, 28);
        assert_eq!(vtable.extensions[0].method_vaddrs, vec![0x0C5B_0000]);
        assert_eq!(vtable.clone(), vtable);
        assert_ne!(format!("{vtable:?}"), "");
        assert_eq!(vtable.extensions[0].clone(), vtable.extensions[0]);
        assert_ne!(format!("{:?}", vtable.extensions[0]), "");

        word_map.insert(ext0 + 16, ext0);
        let cyclic = reconstruct_pjrt_vtable(&word_map, base, 1);
        assert_eq!(cyclic.extensions.len(), 32);

        let mut rodata_blobs = BTreeMap::new();
        let vt_vaddr = 0x5000_0000_u64;
        let ti_vaddr = 0x5000_1000_u64;
        let zts_vaddr = 0x5000_2000_u64;
        word_map.insert(vt_vaddr, 0);
        word_map.insert(vt_vaddr + 8, ti_vaddr);
        word_map.insert(vt_vaddr + 16, 0x0A00_0100);
        word_map.insert(vt_vaddr + 24, 0x0A00_0200);
        word_map.insert(ti_vaddr + 8, zts_vaddr);
        rodata_blobs.insert(zts_vaddr, b"_ZTSN7asic_sw6driver12GasketDriverE\0".to_vec());

        let cpp_vt = reconstruct_itanium_cpp_vtable(&word_map, &rodata_blobs, vt_vaddr, 2);
        assert_eq!(cpp_vt.vtable_vaddr, vt_vaddr);
        assert_eq!(cpp_vt.offset_to_top, 0);
        assert_eq!(cpp_vt.typeinfo_vaddr, ti_vaddr);
        assert!(cpp_vt.class_name.contains("GasketDriver"));
        assert_eq!(cpp_vt.method_vaddrs, vec![0x0A00_0100, 0x0A00_0200]);
        assert_eq!(cpp_vt.clone(), cpp_vt);
        assert_ne!(format!("{cpp_vt:?}"), "");

        rodata_blobs.insert(zts_vaddr, b"N4pjrt10PjrtClientE".to_vec());
        let cpp_vt2 = reconstruct_itanium_cpp_vtable(&word_map, &rodata_blobs, vt_vaddr, 1);
        assert!(cpp_vt2.class_name.contains("PjrtClient"));

        rodata_blobs.insert(zts_vaddr, b"\0".to_vec());
        let cpp_vt3 = reconstruct_itanium_cpp_vtable(&word_map, &rodata_blobs, vt_vaddr, 0);
        assert_eq!(cpp_vt3.class_name, "");
        let cpp_vt4 = reconstruct_itanium_cpp_vtable(&BTreeMap::new(), &BTreeMap::new(), 0x9999, 1);
        assert_eq!(cpp_vt4.class_name, "");
        assert_eq!(cpp_vt4.method_vaddrs, vec![0]);

        let sdk_base = 0x6000_0000_u64;
        word_map.insert(sdk_base, 144);
        word_map.insert(sdk_base + 8, 0x0C2C_A900);
        let sdk_vt = reconstruct_libtpu_sdk_vtable(&word_map, sdk_base);
        assert_eq!(sdk_vt.base_vaddr, sdk_base);
        assert_eq!(sdk_vt.struct_size, 144);
        assert_eq!(sdk_vt.slot_vaddrs.len(), 17);
        assert_eq!(sdk_vt.slot_vaddrs[0], 0x0C2C_A900);
        assert_eq!(sdk_vt.clone(), sdk_vt);
        assert_ne!(format!("{sdk_vt:?}"), "");

        let drv_base = 0x7000_0000_u64;
        word_map.insert(drv_base, 0x0C7A_5E00);
        let drv_vt = reconstruct_tpu_driver_vtable(&word_map, drv_base);
        assert_eq!(drv_vt.base_vaddr, drv_base);
        assert_eq!(drv_vt.struct_size, 224);
        assert_eq!(drv_vt.slot_vaddrs.len(), 28);
        assert_eq!(drv_vt.slot_vaddrs[0], 0x0C7A_5E00);

        // Test CppRttiHierarchyExtractor across ClassTypeInfo, SingleInheritance, and VirtualMultipleInheritance
        let extractor = CppRttiHierarchyExtractor;
        assert_eq!(extractor.clone(), CppRttiHierarchyExtractor);
        assert_ne!(format!("{extractor:?}"), "");

        let mut rtti_words = BTreeMap::new();
        let mut rtti_rodata = BTreeMap::new();
        let class_ti_vt = 0x0100_0000_u64;
        let si_ti_vt = 0x0100_0100_u64;
        let vmi_ti_vt = 0x0100_0200_u64;

        // 1. Root class: asic_sw::driver::DeviceDriver (at 0x0200_0000)
        let base_ti = 0x0200_0000_u64;
        let base_zts = 0x0300_0000_u64;
        rtti_words.insert(base_ti, class_ti_vt);
        rtti_words.insert(base_ti + 8, base_zts);
        rtti_rodata.insert(base_zts, b"_ZTSN7asic_sw6driver12DeviceDriverE\0".to_vec());

        // 2. Single-inheritance class: asic_sw::driver::VfioDriver : public DeviceDriver (at 0x0200_0100)
        let si_ti = 0x0200_0100_u64;
        let si_zts = 0x0300_0100_u64;
        rtti_words.insert(si_ti, si_ti_vt);
        rtti_words.insert(si_ti + 8, si_zts);
        rtti_words.insert(si_ti + 16, base_ti);
        rtti_rodata.insert(si_zts, b"N7asic_sw6driver10VfioDriverE\0".to_vec());

        // Associated _ZTV for VfioDriver at 0x0400_0000
        rtti_words.insert(0x0400_0000, 0);
        rtti_words.insert(0x0400_0008, si_ti);
        rtti_words.insert(0x0400_0010, 0x0A00_1111);
        rtti_words.insert(0x0400_0018, 0x0A00_2222);
        rtti_words.insert(0x0400_0020, 0); // terminator
        rtti_words.insert(0x0400_0100, 0); // word0 == 0 with non-typeinfo +8

        // 3. VMI class: xla::tpu::TpuTransferManager (at 0x0200_0200) inheriting from base_ti + si_ti
        let vmi_ti = 0x0200_0200_u64;
        let vmi_zts = 0x0300_0200_u64;
        rtti_words.insert(vmi_ti, vmi_ti_vt);
        rtti_words.insert(vmi_ti + 8, vmi_zts);
        rtti_words.insert(vmi_ti + 16, 2_u64 << 32); // base_count = 2
        rtti_words.insert(vmi_ti + 24, base_ti);
        rtti_words.insert(vmi_ti + 32, 0x02);
        rtti_words.insert(vmi_ti + 40, si_ti);
        rtti_words.insert(vmi_ti + 48, 0x02);
        rtti_rodata.insert(vmi_zts, b"_ZTSN3xla3tpu18TpuTransferManagerE\0".to_vec());

        // 4. Empty ZTS entry to exercise skip branch
        rtti_words.insert(0x0200_0300, class_ti_vt);
        rtti_words.insert(0x0200_0308, 0x0300_0300);

        let hierarchy = CppRttiHierarchyExtractor::extract_hierarchy(
            &rtti_words,
            &rtti_rodata,
            class_ti_vt,
            si_ti_vt,
            vmi_ti_vt,
        );
        assert_eq!(hierarchy.len(), 3);
        assert_eq!(hierarchy[0].kind, ItaniumRttiKind::ClassTypeInfo);
        assert_eq!(hierarchy[0].class_name, "asic_sw::driver::DeviceDriver");
        assert_eq!(hierarchy[1].kind, ItaniumRttiKind::SingleInheritance);
        assert_eq!(hierarchy[1].class_name, "asic_sw::driver::VfioDriver");
        assert_eq!(hierarchy[1].vtable_vaddr, 0x0400_0000);
        assert_eq!(
            hierarchy[1].virtual_method_vaddrs,
            vec![0x0A00_1111, 0x0A00_2222]
        );
        assert_eq!(
            hierarchy[1].base_class_names,
            vec!["asic_sw::driver::DeviceDriver"]
        );
        assert_eq!(
            hierarchy[2].kind,
            ItaniumRttiKind::VirtualMultipleInheritance
        );
        assert_eq!(hierarchy[2].base_class_names.len(), 2);
        assert_eq!(hierarchy[0].clone(), hierarchy[0]);
        assert_ne!(format!("{:?}", hierarchy[0]), "");
        assert_ne!(format!("{:?}", hierarchy[0].kind), "");
    }
}

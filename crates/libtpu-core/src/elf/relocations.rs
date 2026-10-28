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
    /// Total byte size of `PJRT_Api` (`1,144` bytes in `libtpu-0.0.50`).
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
#[allow(
    for_loops_over_fallibles,
    clippy::indexing_slicing,
    clippy::too_many_lines
)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::elf::symbols::build_synthetic_elf64;

    #[test]
    fn test_resolve_relative_relocations_and_vtable_reconstructors() {
        let elf_bytes = build_synthetic_elf64();
        let relocs = resolve_relative_relocations(&elf_bytes);
        assert!(relocs.is_ok());
        for r_list in relocs {
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
    }
}

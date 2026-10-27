//! Clean-room ELF64 and Mach-O binary forensics, relocation resolver, and `iced-x86` disassembler.
//!
//! Also provides `gimli` `.eh_frame` unwind boundary parsing, `cpp_demangle` RTTI/vtable
//! classification, and embedded `protodesc_cold` / `filewrapper_toc` resource carving:
//! - [`inspect_binary_symbols`], [`BinarySymbolReport`], [`ExportedBinarySymbol`], and [`BinaryFormatKind`]
//!   to extract section headers, `.dynsym` / `.symtab` symbols, and binding flags (`STB_GLOBAL` / `STB_WEAK`).
//! - [`resolve_relative_relocations`] and [`ElfRelativeRelocation`] to resolve `.rela.dyn`
//!   (`R_X86_64_RELATIVE` / `R_AARCH64_RELATIVE`) entries and reconstruct static vtables (`PJRT_Api`,
//!   `LibtpuSdk_Api`, `TpuDriverFn`, and the 17-node `PJRT_Extension_Base` linked list).
//! - [`disassemble_x86_64_prologue`], [`build_x86_64_cfg`], [`ControlFlowGraphReport`],
//!   [`PrologueLayoutReport`], and [`RdiFieldAccess`] powered by `iced_x86` to decode x86-64
//!   function prologues, basic blocks, branch/call targets, RIP-relative memory references,
//!   and `[rdi + disp]` field accesses.
//! - [`parse_eh_frame_fde_ranges`] and [`EhFrameFunctionRange`] powered by `gimli` to recover
//!   exact function `[start_vaddr, end_vaddr)` boundaries from `.eh_frame`.
//! - [`demangle_and_classify_cpp_symbol`], [`DemangledCppSymbol`], and [`CppNamespaceCategory`]
//!   powered by `cpp_demangle` to demangle Itanium C++ ABI symbols and vtables.
//! - [`carve_protodesc_cold_entries`], [`decode_carved_file_descriptor`], [`render_proto_schema`],
//!   [`CarvedProtoDesc`], [`extract_filewrapper_toc`], [`extract_indirect_filewrapper_toc`],
//!   [`FileTocEntry`], and [`compute_artifact_digests`] to carve and decode embedded Protobuf
//!   `FileDescriptorProto` schemas and `filewrapper_toc` bundles from `libtpu.so` and `sdk.so`.

use crate::error::{LibTpuError, LibTpuResult};
use gimli::{BaseAddresses, CieOrFde, EhFrame, LittleEndian, UnwindSection};
use goblin::Object;
use iced_x86::{Decoder, DecoderOptions, FlowControl, Mnemonic, OpKind, Register};
use object::{Object as _, ObjectSection as _, ObjectSymbol as _};
use prost::Message;
use prost_types::FileDescriptorProto;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io::Read as _;

/// x86-64 relative relocation type (`R_X86_64_RELATIVE = 8`).
pub const R_X86_64_RELATIVE: u32 = 8;

/// `AArch64` relative relocation type (`R_AARCH64_RELATIVE = 1027`).
pub const R_AARCH64_RELATIVE: u32 = 1027;

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

/// Decoded register dereference off `rdi` (`[rdi + disp]`) inside an x86-64 function prologue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RdiFieldAccess {
    /// Byte offset from `rdi` (`0x00`, `0x08`, `0x10`, ...).
    pub byte_offset: u32,
    /// Operand width in bytes (`1`, `4`, or `8`).
    pub access_size: u8,
    /// Whether the instruction writes to (`true`) or reads from (`false`) `[rdi + byte_offset]`.
    pub is_write: bool,
}

/// Struct layout metadata inferred by disassembling an x86-64 C-API function prologue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrologueLayoutReport {
    /// Expected minimum `struct_size` extracted from `cmp qword ptr [rdi], IMM` or `mov esi, IMM`.
    pub expected_struct_size: Option<u32>,
    /// Ordered list of `[rdi + disp]` field accesses observed in the prologue.
    pub field_accesses: Vec<RdiFieldAccess>,
}

/// Control Flow Graph (CFG) summary constructed by `iced_x86` over an x86-64 function body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlFlowGraphReport {
    /// Total number of valid instructions decoded.
    pub instruction_count: usize,
    /// Number of basic block leader addresses identified.
    pub basic_block_count: usize,
    /// Direct conditional and unconditional branch target addresses.
    pub branch_targets: Vec<u64>,
    /// Direct `call rel32` target virtual addresses.
    pub call_targets: Vec<u64>,
    /// RIP-relative `[rip + disp32]` memory target virtual addresses.
    pub rip_relative_targets: Vec<u64>,
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

/// Carved `FileDescriptorProto` entry recovered from the `protodesc_cold` ELF section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CarvedProtoDesc {
    /// Protobuf schema file name (e.g., `"xla_data.proto"`).
    pub name: String,
    /// Protobuf package name (e.g., `"xla"`).
    pub package: String,
    /// Complete wire-encoded `google.protobuf.FileDescriptorProto` bytes.
    pub raw_bytes: Vec<u8>,
}

/// Embedded static resource entry recovered from the `filewrapper_toc` ELF section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileTocEntry {
    /// Embedded file name.
    pub name: String,
    /// Embedded payload bytes.
    pub data: Vec<u8>,
}

/// Cryptographic SHA-256 and BLAKE3 digest pair for binary artifact verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactDigestReport {
    /// Byte length of the hashed payload.
    pub size_bytes: usize,
    /// Lowercase hexadecimal SHA-256 digest (`64` hex characters).
    pub sha256_hex: String,
    /// Lowercase hexadecimal BLAKE3 digest (`64` hex characters).
    pub blake3_hex: String,
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

/// Disassembles an x86-64 function prologue (`code`) using `iced_x86::Decoder` to extract the
/// expected `PJRT_*_Args` `struct_size` and all `[rdi + disp]` field dereferences.
///
/// Decodes full variable-length x86-64 instructions cleanly (never losing synchronization on
/// multi-byte instructions) and tracks:
/// - `cmp qword ptr [rdi], imm` (`Mnemonic::Cmp`)
/// - `mov esi, imm` (`Mnemonic::Mov` into `Register::ESI` for `PJRT_CheckMatchingStructSizes`)
/// - Memory reads and writes with base `Register::RDI` (`[rdi + disp]`).
///
/// # Arguments
///
/// * `code` - Raw x86-64 machine code bytes of a function prologue.
#[must_use]
pub fn disassemble_x86_64_prologue(code: &[u8]) -> PrologueLayoutReport {
    let mut expected_struct_size: Option<u32> = None;
    let mut field_accesses = Vec::new();
    let mut rdi_aliases = vec![Register::RDI];
    let mut decoder = Decoder::with_ip(64, code, 0x1000_0000, DecoderOptions::NONE);

    while decoder.can_decode() {
        let instr = decoder.decode();
        if instr.mnemonic() == Mnemonic::Ret {
            break;
        }
        if instr.mnemonic() == Mnemonic::Mov
            && instr.op0_kind() == OpKind::Register
            && instr.op1_kind() == OpKind::Register
            && rdi_aliases.contains(&instr.op1_register())
        {
            let dst_reg = instr.op0_register();
            if !rdi_aliases.contains(&dst_reg) {
                rdi_aliases.push(dst_reg);
            }
            continue;
        }
        if instr.mnemonic() == Mnemonic::Cmp
            && instr.op0_kind() == OpKind::Memory
            && rdi_aliases.contains(&instr.memory_base())
            && instr.memory_displacement64() == 0
        {
            #[allow(clippy::cast_possible_truncation)]
            let imm = instr.immediate(1) as u32;
            expected_struct_size = Some(imm);
        } else if instr.mnemonic() == Mnemonic::Mov
            && instr.op0_kind() == OpKind::Register
            && instr.op0_register() == Register::ESI
            && instr.op1_kind() == OpKind::Immediate32
        {
            let imm = instr.immediate32();
            expected_struct_size = Some(expected_struct_size.unwrap_or(imm));
        } else if instr.op0_kind() == OpKind::Memory
            && rdi_aliases.contains(&instr.memory_base())
            && instr.mnemonic() == Mnemonic::Mov
        {
            #[allow(clippy::cast_possible_truncation)]
            let disp = instr.memory_displacement64() as u32;
            #[allow(clippy::cast_possible_truncation)]
            let access_size = instr.memory_size().size() as u8;
            field_accesses.push(RdiFieldAccess {
                byte_offset: disp,
                access_size,
                is_write: true,
            });
        } else if instr.op1_kind() == OpKind::Memory
            && rdi_aliases.contains(&instr.memory_base())
            && matches!(
                instr.mnemonic(),
                Mnemonic::Mov | Mnemonic::Movzx | Mnemonic::Movsx | Mnemonic::Movsxd
            )
        {
            #[allow(clippy::cast_possible_truncation)]
            let disp = instr.memory_displacement64() as u32;
            #[allow(clippy::cast_possible_truncation)]
            let access_size = instr.memory_size().size() as u8;
            field_accesses.push(RdiFieldAccess {
                byte_offset: disp,
                access_size,
                is_write: false,
            });
        }
    }

    PrologueLayoutReport {
        expected_struct_size,
        field_accesses,
    }
}

/// Decodes an x86-64 machine code slice using `iced_x86` and constructs a Control Flow Graph
/// (CFG) report with basic block counts, branch targets, call targets, and RIP-relative memory
/// target addresses.
///
/// # Arguments
///
/// * `code` - Raw x86-64 machine code bytes.
/// * `base_ip` - Starting virtual instruction pointer (`RIP`) for `code`.
#[must_use]
pub fn build_x86_64_cfg(code: &[u8], base_ip: u64) -> ControlFlowGraphReport {
    let mut decoder = Decoder::with_ip(64, code, base_ip, DecoderOptions::NONE);
    let mut instruction_count = 0_usize;
    let mut leaders = std::collections::BTreeSet::new();
    if !code.is_empty() {
        leaders.insert(base_ip);
    }
    let mut branch_targets = Vec::new();
    let mut call_targets = Vec::new();
    let mut rip_relative_targets = Vec::new();

    while decoder.can_decode() {
        let instr = decoder.decode();
        if instr.is_invalid() {
            continue;
        }
        instruction_count += 1;
        if instr.is_ip_rel_memory_operand() {
            rip_relative_targets.push(instr.ip_rel_memory_address());
        }
        match instr.flow_control() {
            FlowControl::ConditionalBranch | FlowControl::UnconditionalBranch => {
                let target = instr.near_branch_target();
                branch_targets.push(target);
                leaders.insert(target);
                leaders.insert(instr.next_ip());
            }
            FlowControl::Call => {
                call_targets.push(instr.near_branch_target());
            }
            FlowControl::Return if decoder.can_decode() => {
                leaders.insert(instr.next_ip());
            }
            _ => {}
        }
    }

    ControlFlowGraphReport {
        instruction_count,
        basic_block_count: leaders.len(),
        branch_targets,
        call_targets,
        rip_relative_targets,
    }
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

/// Reads a protobuf base-128 varint starting at `bytes[pos]`, returning `(value, next_pos)`.
///
/// # Arguments
///
/// * `bytes` - Input byte slice.
/// * `pos` - Starting byte offset.
fn read_proto_varint(bytes: &[u8], pos: usize) -> Option<(u64, usize)> {
    let mut result = 0_u64;
    let mut shift = 0_u32;
    let mut cur = pos;
    while shift < 64 {
        let byte = bytes.get(cur).copied()?;
        cur += 1;
        result |= u64::from(byte & 0x7F) << shift;
        if (byte & 0x80) == 0 {
            return Some((result, cur));
        }
        shift += 7;
    }
    None
}

/// Advances across all valid consecutive `google.protobuf.FileDescriptorProto` wire fields
/// starting at `start_idx`, returning the exclusive end byte offset.
///
/// # Arguments
///
/// * `bytes` - `protodesc_cold` byte slice.
/// * `start_idx` - Offset of the leading `0x0A` (`name`) tag.
fn scan_file_descriptor_wire_end(bytes: &[u8], start_idx: usize) -> usize {
    let mut cur = start_idx;
    while cur < bytes.len() {
        // Stop if we encounter a new top-level `0x0A <len> "...proto"` FileDescriptorProto header
        if cur > start_idx && bytes.get(cur).copied().unwrap_or(0) == 0x0A {
            let next_len = usize::from(bytes.get(cur + 1).copied().unwrap_or(0));
            let next_name = bytes.get(cur + 2..cur + 2 + next_len).unwrap_or_default();
            if next_name.ends_with(b".proto") {
                break;
            }
        }
        let Some((tag, after_tag)) = read_proto_varint(bytes, cur) else {
            break;
        };
        let field_num = tag >> 3;
        let wire_type = tag & 0x07;
        // FileDescriptorProto valid field numbers are 1..=14
        if !(1..=14).contains(&field_num) {
            break;
        }
        match wire_type {
            0 => {
                let Some((_, after_val)) = read_proto_varint(bytes, after_tag) else {
                    break;
                };
                cur = after_val;
            }
            2 => {
                let Some((len_u64, after_len)) = read_proto_varint(bytes, after_tag) else {
                    break;
                };
                let len = usize::try_from(len_u64).unwrap_or(usize::MAX);
                let next_pos = after_len.saturating_add(len);
                if next_pos > bytes.len() {
                    break;
                }
                cur = next_pos;
            }
            _ => break,
        }
    }
    cur
}

/// Scans a `protodesc_cold` section byte slice for wire-encoded `FileDescriptorProto` entries.
///
/// Matches entries starting with tag `0x0a` (field 1: `string name` ending in `".proto"`),
/// preserving all subsequent `FileDescriptorProto` fields (`message_type`, `enum_type`,
/// `dependency`, `syntax`).
///
/// # Arguments
///
/// * `section_bytes` - Raw byte slice of the `protodesc_cold` section.
#[must_use]
pub fn carve_protodesc_cold_entries(section_bytes: &[u8]) -> Vec<CarvedProtoDesc> {
    let mut results = Vec::new();
    let mut idx = 0_usize;
    while idx + 8 <= section_bytes.len() {
        if section_bytes.get(idx).copied().unwrap_or(0) == 0x0A {
            let name_len = usize::from(section_bytes.get(idx + 1).copied().unwrap_or(0));
            let name_slice = section_bytes
                .get(idx + 2..idx + 2 + name_len)
                .unwrap_or_default();
            if name_slice.ends_with(b".proto") {
                let end_idx = scan_file_descriptor_wire_end(section_bytes, idx);
                let raw_bytes = section_bytes.get(idx..end_idx).unwrap_or_default().to_vec();
                let (name, pkg) = FileDescriptorProto::decode(raw_bytes.as_slice()).map_or_else(
                    |_| {
                        (
                            String::from_utf8_lossy(name_slice).into_owned(),
                            String::new(),
                        )
                    },
                    |fd| (fd.name.unwrap_or_default(), fd.package.unwrap_or_default()),
                );
                results.push(CarvedProtoDesc {
                    name,
                    package: pkg,
                    raw_bytes,
                });
                idx = end_idx.max(idx + 1);
                continue;
            }
        }
        idx += 1;
    }
    results
}

/// Decodes a [`CarvedProtoDesc`] into a strongly-typed `prost_types::FileDescriptorProto`.
///
/// # Arguments
///
/// * `carved` - Carved descriptor entry from [`carve_protodesc_cold_entries`].
///
/// # Errors
///
/// Returns [`LibTpuError`] if `carved.raw_bytes` is not a valid
/// wire-encoded `FileDescriptorProto`.
pub fn decode_carved_file_descriptor(
    carved: &CarvedProtoDesc,
) -> LibTpuResult<FileDescriptorProto> {
    FileDescriptorProto::decode(carved.raw_bytes.as_slice()).map_err(LibTpuError::from)
}

/// Builds a live `prost_reflect::DescriptorPool` from a slice of carved [`CarvedProtoDesc`] entries.
///
/// Decodes each `FileDescriptorProto` and registers them as a `prost_types::FileDescriptorSet`
/// in a `prost_reflect::DescriptorPool`.
///
/// # Arguments
///
/// * `entries` - Carved protobuf file descriptor entries from `protodesc_cold`.
///
/// # Errors
///
/// Returns [`LibTpuError`] if any descriptor fails wire decoding or topological resolution in `prost_reflect`.
pub fn build_descriptor_pool_from_carved_protos(
    entries: &[CarvedProtoDesc],
) -> LibTpuResult<prost_reflect::DescriptorPool> {
    let mut files = Vec::with_capacity(entries.len());
    for entry in entries {
        files.push(decode_carved_file_descriptor(entry)?);
    }
    let fds = prost_types::FileDescriptorSet { file: files };
    prost_reflect::DescriptorPool::from_file_descriptor_set(fds).map_err(LibTpuError::from)
}

/// Maps a `prost_types::field_descriptor_proto::Type` integer discriminant (`1..=18`)
/// to its canonical `.proto` scalar type keyword.
///
/// # Arguments
///
/// * `type_id` - Optional protobuf `FieldDescriptorProto.Type` integer tag (`1..=18`).
#[must_use]
pub const fn proto_scalar_type_name(type_id: Option<i32>) -> Option<&'static str> {
    match type_id {
        Some(1) => Some("double"),
        Some(2) => Some("float"),
        Some(3) => Some("int64"),
        Some(4) => Some("uint64"),
        Some(5) => Some("int32"),
        Some(6) => Some("fixed64"),
        Some(7) => Some("fixed32"),
        Some(8) => Some("bool"),
        Some(9) => Some("string"),
        Some(10) => Some("group"),
        Some(12) => Some("bytes"),
        Some(13) => Some("uint32"),
        Some(15) => Some("sfixed32"),
        Some(16) => Some("sfixed64"),
        Some(17) => Some("sint32"),
        Some(18) => Some("sint64"),
        _ => None,
    }
}

/// Resolves the canonical `.proto` type string for a `prost_types::FieldDescriptorProto`.
///
/// # Arguments
///
/// * `field` - Field descriptor to inspect.
#[must_use]
fn resolve_proto_field_type(field: &prost_types::FieldDescriptorProto) -> &str {
    field
        .type_name
        .as_deref()
        .filter(|t| !t.is_empty())
        .map_or_else(
            || proto_scalar_type_name(field.r#type).unwrap_or("bytes"),
            |t_name| t_name.trim_start_matches('.'),
        )
}

/// Renders an `EnumDescriptorProto` block at the given indentation depth.
///
/// # Arguments
///
/// * `out` - Destination buffer.
/// * `en` - Enum descriptor to render.
/// * `indent` - Leading whitespace prefix.
fn render_proto_enum(out: &mut String, en: &prost_types::EnumDescriptorProto, indent: &str) {
    let en_name = en.name.as_deref().unwrap_or("UnnamedEnum");
    let _ = writeln!(out, "\n{indent}enum {en_name} {{");
    for val in &en.value {
        let v_name = val.name.as_deref().unwrap_or("UNKNOWN");
        let v_num = val.number.unwrap_or(0);
        let _ = writeln!(out, "{indent}  {v_name} = {v_num};");
    }
    let _ = writeln!(out, "{indent}}}");
}

/// Renders a `DescriptorProto` message (including nested enums, nested messages, `map<K, V>` fields,
/// `oneof` blocks, and field labels) into `out`.
///
/// # Arguments
///
/// * `out` - Destination buffer.
/// * `msg` - Message descriptor to render.
/// * `indent` - Leading whitespace prefix.
/// * `is_proto3` - Whether the enclosing file uses `proto3` syntax.
fn render_proto_message(
    out: &mut String,
    msg: &prost_types::DescriptorProto,
    indent: &str,
    is_proto3: bool,
) {
    let msg_name = msg.name.as_deref().unwrap_or("UnnamedMessage");
    let _ = writeln!(out, "\n{indent}message {msg_name} {{");
    let inner_indent = format!("{indent}  ");

    let mut map_entries = BTreeMap::new();
    for nested in &msg.nested_type {
        let is_map = nested
            .options
            .as_ref()
            .and_then(|opts| opts.map_entry)
            .unwrap_or(false);
        if is_map {
            let entry_name = nested.name.as_deref().unwrap_or_default();
            let key_type = nested
                .field
                .iter()
                .find(|f| f.number == Some(1))
                .map_or("string", resolve_proto_field_type);
            let val_type = nested
                .field
                .iter()
                .find(|f| f.number == Some(2))
                .map_or("bytes", resolve_proto_field_type);
            map_entries.insert(entry_name, (key_type, val_type));
        }
    }

    for en in &msg.enum_type {
        render_proto_enum(out, en, &inner_indent);
    }

    for nested in &msg.nested_type {
        let is_map = nested
            .options
            .as_ref()
            .and_then(|opts| opts.map_entry)
            .unwrap_or(false);
        if !is_map {
            render_proto_message(out, nested, &inner_indent, is_proto3);
        }
    }

    for (oneof_idx, oneof) in msg.oneof_decl.iter().enumerate() {
        let oneof_name = oneof.name.as_deref().unwrap_or("unnamed_oneof");
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let target_idx = oneof_idx as i32;
        let members: Vec<&prost_types::FieldDescriptorProto> = msg
            .field
            .iter()
            .filter(|f| f.oneof_index == Some(target_idx) && !f.proto3_optional.unwrap_or(false))
            .collect();
        if !members.is_empty() {
            let _ = writeln!(out, "{inner_indent}oneof {oneof_name} {{");
            for field in members {
                let f_name = field.name.as_deref().unwrap_or("field");
                let f_num = field.number.unwrap_or(1);
                let f_type = resolve_proto_field_type(field);
                let _ = writeln!(out, "{inner_indent}  {f_type} {f_name} = {f_num};");
            }
            let _ = writeln!(out, "{inner_indent}}}");
        }
    }

    for field in &msg.field {
        if field.oneof_index.is_some() && !field.proto3_optional.unwrap_or(false) {
            continue;
        }
        let f_name = field.name.as_deref().unwrap_or("field");
        let f_num = field.number.unwrap_or(1);
        let raw_type = resolve_proto_field_type(field);
        let short_type = raw_type.rsplit('.').next().unwrap_or(raw_type);
        if let Some(&(k_ty, v_ty)) = map_entries.get(short_type) {
            let _ = writeln!(out, "{inner_indent}map<{k_ty}, {v_ty}> {f_name} = {f_num};");
            continue;
        }
        let label_prefix = match field.label {
            Some(3) => "repeated ",
            Some(2) => "required ",
            Some(1) if !is_proto3 || field.proto3_optional.unwrap_or(false) => "optional ",
            _ => "",
        };
        let _ = writeln!(
            out,
            "{inner_indent}{label_prefix}{raw_type} {f_name} = {f_num};"
        );
    }

    let _ = writeln!(out, "{indent}}}");
}

/// Renders a `prost_types::FileDescriptorProto` into canonical `.proto` source text.
///
/// Reconstructs `syntax`, `package`, `import`, top-level and nested `enum` blocks,
/// top-level and nested `message` blocks, `oneof` declarations, `map<K, V>` fields,
/// field labels (`optional`, `required`, `repeated`), and all 18 protobuf scalar types.
///
/// # Arguments
///
/// * `fd` - Decoded `FileDescriptorProto` schema.
#[must_use]
pub fn render_proto_schema(fd: &FileDescriptorProto) -> String {
    let mut out = String::new();
    let syntax = fd.syntax.as_deref().unwrap_or("proto2");
    let is_proto3 = syntax == "proto3";
    let _ = writeln!(out, r#"syntax = "{syntax}";"#);
    if let Some(pkg) = fd.package.as_deref().filter(|p| !p.is_empty()) {
        let _ = writeln!(out, "package {pkg};");
    }
    for dep in &fd.dependency {
        let _ = writeln!(out, r#"import "{dep}";"#);
    }
    for en in &fd.enum_type {
        render_proto_enum(&mut out, en, "");
    }
    for msg in &fd.message_type {
        render_proto_message(&mut out, msg, "", is_proto3);
    }
    out
}

/// Parses a `filewrapper_toc` table of contents (`24`-byte or `40`-byte `FileToc` entries)
/// backed by a companion data blob.
///
/// # Arguments
///
/// * `toc_bytes` - Raw byte slice containing 24-byte `(name_off: u64, data_off: u64, size: u64)` or 40-byte `(name_off, data_off, size, md5_lo, md5_hi)` records.
/// * `blob_bytes` - Backing string and payload data buffer referenced by `toc_bytes`.
#[must_use]
pub fn extract_filewrapper_toc(toc_bytes: &[u8], blob_bytes: &[u8]) -> Vec<FileTocEntry> {
    let stride = if toc_bytes.len() >= 40 && toc_bytes.len() % 40 == 0 {
        40
    } else {
        24
    };
    let mut entries = Vec::new();
    for chunk in toc_bytes.chunks_exact(stride) {
        let name_off = usize::try_from(u64::from_le(bytemuck::pod_read_unaligned::<u64>(
            chunk.get(0..8).unwrap_or(&[0; 8]),
        )))
        .unwrap_or(usize::MAX);
        let data_off = usize::try_from(u64::from_le(bytemuck::pod_read_unaligned::<u64>(
            chunk.get(8..16).unwrap_or(&[0; 8]),
        )))
        .unwrap_or(usize::MAX);
        let data_size = usize::try_from(u64::from_le(bytemuck::pod_read_unaligned::<u64>(
            chunk.get(16..24).unwrap_or(&[0; 8]),
        )))
        .unwrap_or(0);

        let Some(name_tail) = blob_bytes.get(name_off..) else {
            continue;
        };
        let Some(data_slice) = blob_bytes.get(data_off..data_off.saturating_add(data_size)) else {
            continue;
        };
        let nul_pos = name_tail
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(name_tail.len());
        let name =
            String::from_utf8_lossy(name_tail.get(..nul_pos).unwrap_or_default()).into_owned();
        if !name.is_empty() {
            entries.push(FileTocEntry {
                name,
                data: data_slice.to_vec(),
            });
        }
    }
    entries
}

/// Resolves an indirect `filewrapper_toc` pointer array into [`FileTocEntry`] items.
///
/// In `libtpu.so`, each 8-byte slot in `filewrapper_toc` is relocated via `.rela.dyn` to a
/// null-terminated array of 40-byte `FileToc` structs (`name_ptr`, `data_ptr`, `size`, `md5[16]`)
/// in `.data.rel.ro`.
///
/// # Arguments
///
/// * `toc_slot_vaddrs` - Virtual addresses of the 8-byte pointer slots inside `filewrapper_toc`.
/// * `reloc_map` - Mapping from relocated virtual address (`r_offset`) to target `r_addend`.
/// * `size_word_map` - Mapping from virtual address (`toc_struct_vaddr + 16`) to payload byte length.
/// * `rodata_blobs` - Mapping from virtual address to raw byte slice in `.rodata`.
#[must_use]
pub fn extract_indirect_filewrapper_toc(
    toc_slot_vaddrs: &[u64],
    reloc_map: &BTreeMap<u64, u64>,
    size_word_map: &BTreeMap<u64, u64>,
    rodata_blobs: &BTreeMap<u64, Vec<u8>>,
) -> Vec<FileTocEntry> {
    let mut entries = Vec::new();
    for &slot_vaddr in toc_slot_vaddrs {
        let Some(&base_toc_vaddr) = reloc_map.get(&slot_vaddr) else {
            continue;
        };
        let mut elem_idx = 0_u64;
        while let Some(&name_vaddr) = reloc_map.get(&base_toc_vaddr.wrapping_add(elem_idx * 40)) {
            let entry_vaddr = base_toc_vaddr.wrapping_add(elem_idx * 40);
            elem_idx += 1;
            let Some(&data_vaddr) = reloc_map.get(&entry_vaddr.wrapping_add(8)) else {
                break;
            };
            let size = usize::try_from(
                size_word_map
                    .get(&entry_vaddr.wrapping_add(16))
                    .copied()
                    .unwrap_or(0),
            )
            .unwrap_or(0);
            let Some(name_bytes) = rodata_blobs.get(&name_vaddr) else {
                continue;
            };
            let Some(data_bytes) = rodata_blobs.get(&data_vaddr) else {
                continue;
            };
            let nul_pos = name_bytes
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(name_bytes.len());
            let name =
                String::from_utf8_lossy(name_bytes.get(..nul_pos).unwrap_or_default()).into_owned();
            let payload = data_bytes
                .get(..size.min(data_bytes.len()))
                .unwrap_or_default();
            if !name.is_empty() {
                entries.push(FileTocEntry {
                    name,
                    data: payload.to_vec(),
                });
            }
        }
    }
    entries
}

/// Decompresses an embedded Brotli-compressed (`*_pf_csrs.br`) hardware CSR table from `libtpu.so`.
///
/// # Arguments
///
/// * `compressed` - Raw Brotli-compressed payload extracted from `filewrapper_toc`.
///
/// # Errors
///
/// Returns [`LibTpuError::BinaryFormatError`] if `compressed` is not a valid Brotli stream.
pub fn decompress_brotli_csr_table(compressed: &[u8]) -> LibTpuResult<Vec<u8>> {
    let mut decompressor = brotli::Decompressor::new(compressed, 4096);
    let mut out = Vec::new();
    decompressor
        .read_to_end(&mut out)
        .map_err(|err| LibTpuError::BinaryFormatError {
            reason: format!("brotli CSR decompression failed: {err}"),
        })?;
    Ok(out)
}

/// Exported `.dynsym` function record inside [`GoldenLibtpuSo`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoldenExportedSymbol {
    /// Symbol name.
    pub name: String,
    /// Hex virtual address string (`"0x..."`).
    pub vaddr: String,
    /// Function byte size (`st_size`).
    pub size: u64,
}

/// `PJRT_Api` vtable slot record inside [`GoldenLibtpuSo`] and [`GoldenStructLayouts`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoldenPjrtSlot {
    /// 64-bit word index in `PJRT_Api` (`5..=142`).
    pub word_index: usize,
    /// Hex byte offset in `PJRT_Api` (`"0x28".."0x470"`).
    pub byte_offset: String,
    /// Minimum `struct_size` validated by the entrypoint.
    pub min_struct_size: usize,
    /// Hex virtual address of the handler function in `libtpu.so`.
    pub vaddr: String,
    /// Mangled C++ symbol name in `libtpu.so`.
    pub mangled_symbol: String,
}

/// `PJRT_Extension_Base` node record inside [`GoldenStructLayouts`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoldenExtensionNode {
    /// Extension node index (`1..=17`).
    pub index: usize,
    /// Extension struct name (e.g., `"PJRT_XlaTransform_Extension"`).
    pub name: String,
    /// `PJRT_Extension_Type` discriminant.
    pub ext_type: u32,
    /// Extension `struct_size` in bytes.
    pub struct_size: usize,
    /// Number of function pointers in the extension struct.
    pub num_functions: usize,
}

/// `libtpu/libtpu.so` section of [`GoldenBinaryManifest`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoldenLibtpuSo {
    /// Total file size in bytes (`630,899,976`).
    pub size_bytes: u64,
    /// Lowercase 64-char hex SHA-256 digest.
    pub sha256: String,
    /// `PJRT_Api` struct size (`1144`).
    pub pjrt_api_struct_size: usize,
    /// `PJRT_Api_Version.major_version` (`0`).
    pub pjrt_api_version_major: i32,
    /// `PJRT_Api_Version.minor_version` (`116`).
    pub pjrt_api_version_minor: i32,
    /// `LibtpuSdk_Api` struct size (`144`).
    pub libtpu_sdk_api_struct_size: usize,
    /// `TpuDriverFn` struct size (`224`).
    pub tpu_driver_fn_struct_size: usize,
    /// Number of exported `.dynsym` functions (`78`).
    pub exported_functions_count: usize,
    /// Exported `.dynsym` function list.
    pub exported_functions: Vec<GoldenExportedSymbol>,
    /// 138 `PJRT_Api` vtable slots.
    pub pjrt_vtable_slots: Vec<GoldenPjrtSlot>,
}

/// `libtpu/sdk.so` section of [`GoldenBinaryManifest`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoldenSdkSo {
    /// Total file size in bytes (`37,821,432`).
    pub size_bytes: u64,
    /// Lowercase 64-char hex SHA-256 digest.
    pub sha256: String,
    /// Python C-extension entrypoint (`"PyInit_sdk"`).
    pub entrypoint: String,
    /// Exported Python submodules (`["slice", "tpumonitoring", "tpuz", "debug"]`).
    pub submodules: Vec<String>,
}

/// Strongly-typed deserialized representation of `docs/reverse_engineering/golden_binary_manifest.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoldenBinaryManifest {
    /// `PyPI` package name (`"libtpu"`).
    pub package: String,
    /// Package version (`"0.0.50"`).
    pub version: String,
    /// Wheel filename.
    pub wheel: String,
    /// `libtpu/libtpu.so` manifest.
    pub libtpu_so: GoldenLibtpuSo,
    /// `libtpu/sdk.so` manifest.
    pub sdk_so: GoldenSdkSo,
}

/// Strongly-typed deserialized representation of `docs/reverse_engineering/golden_struct_layouts.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoldenStructLayouts {
    /// `PJRT_Api` struct size (`1144`).
    pub pjrt_api_size: usize,
    /// `PJRT_Api_Version` struct size (`24`).
    pub pjrt_api_version_size: usize,
    /// `PJRT_Extension_Base` struct size (`24`).
    pub pjrt_extension_base_size: usize,
    /// `PLUGIN_Profiler_Api` struct size (`80`).
    pub plugin_profiler_api_size: usize,
    /// `LibtpuSdk_Api` struct size (`144`).
    pub libtpu_sdk_api_size: usize,
    /// `TpuDriverFn` struct size (`224`).
    pub tpu_driver_fn_size: usize,
    /// 17 `PJRT_Extension_Base` nodes.
    pub extensions: Vec<GoldenExtensionNode>,
    /// 138 `PJRT_Api` vtable slots.
    pub pjrt_slots: Vec<GoldenPjrtSlot>,
}

/// Parses and validates a `golden_binary_manifest.json` string using `serde_json`.
///
/// # Arguments
///
/// * `json_str` - UTF-8 JSON content of `golden_binary_manifest.json`.
///
/// # Errors
///
/// Returns [`LibTpuError::BinaryFormatError`] if `json_str` is invalid JSON or violates schema invariants.
pub fn parse_golden_binary_manifest(json_str: &str) -> LibTpuResult<GoldenBinaryManifest> {
    let manifest: GoldenBinaryManifest =
        serde_json::from_str(json_str).map_err(|err| LibTpuError::BinaryFormatError {
            reason: format!("golden_binary_manifest.json parse error: {err}"),
        })?;
    if manifest.libtpu_so.sha256.len() != 64
        || manifest.sdk_so.sha256.len() != 64
        || manifest.libtpu_so.exported_functions.len()
            != manifest.libtpu_so.exported_functions_count
    {
        return Err(LibTpuError::BinaryFormatError {
            reason: String::from("golden_binary_manifest.json invariant violation"),
        });
    }
    Ok(manifest)
}

/// Parses and validates a `golden_struct_layouts.json` string using `serde_json`.
///
/// # Arguments
///
/// * `json_str` - UTF-8 JSON content of `golden_struct_layouts.json`.
///
/// # Errors
///
/// Returns [`LibTpuError::BinaryFormatError`] if `json_str` is invalid JSON or violates schema invariants.
pub fn parse_golden_struct_layouts(json_str: &str) -> LibTpuResult<GoldenStructLayouts> {
    let layouts: GoldenStructLayouts =
        serde_json::from_str(json_str).map_err(|err| LibTpuError::BinaryFormatError {
            reason: format!("golden_struct_layouts.json parse error: {err}"),
        })?;
    if layouts.pjrt_api_size != 1144
        || layouts.extensions.len() != 17
        || layouts.pjrt_slots.len() != 138
    {
        return Err(LibTpuError::BinaryFormatError {
            reason: String::from("golden_struct_layouts.json invariant violation"),
        });
    }
    Ok(layouts)
}

/// Computes both SHA-256 and BLAKE3 digests over a binary artifact byte slice.
///
/// # Arguments
///
/// * `bytes` - Raw binary payload to hash.
#[must_use]
pub fn compute_artifact_digests(bytes: &[u8]) -> ArtifactDigestReport {
    let sha256_digest = Sha256::digest(bytes);
    let mut sha256_hex = String::with_capacity(64);
    for b in sha256_digest {
        let _ = write!(sha256_hex, "{b:02x}");
    }
    let blake3_hex = blake3::hash(bytes).to_hex().to_string();
    ArtifactDigestReport {
        size_bytes: bytes.len(),
        sha256_hex,
        blake3_hex,
    }
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

/// Type alias for [`DemangledCppSymbol`] (`DemangledSymbolEntry`).
pub type DemangledSymbolEntry = DemangledCppSymbol;

/// Complete disassembled function forensics specification (`DisassembledFunctionSpec`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisassembledFunctionSpec {
    /// Function entrypoint virtual address.
    pub entrypoint_vaddr: u64,
    /// Extracted prologue `struct_size` validation and `[rdi + disp]` field accesses.
    pub prologue: PrologueLayoutReport,
    /// Reconstructed basic-block control-flow graph (`ControlFlowGraphReport`).
    pub cfg: ControlFlowGraphReport,
    /// Decoded Linux `ioctl` callsites inside the function body.
    pub ioctl_callsites: Vec<DecodedIoctlCallsite>,
}

/// Disassembles an x86-64 function body at `entrypoint_vaddr` into a [`DisassembledFunctionSpec`].
///
/// # Arguments
///
/// * `code` - Raw x86-64 machine code bytes of the function.
/// * `entrypoint_vaddr` - Starting virtual instruction pointer (`RIP`) of the function.
#[must_use]
pub fn disassemble_function_spec(code: &[u8], entrypoint_vaddr: u64) -> DisassembledFunctionSpec {
    DisassembledFunctionSpec {
        entrypoint_vaddr,
        prologue: disassemble_x86_64_prologue(code),
        cfg: build_x86_64_cfg(code, entrypoint_vaddr),
        ioctl_callsites: scan_x86_64_ioctl_callsites(code, entrypoint_vaddr),
    }
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

/// Decoded Linux `_IOC(dir, type, nr, size)` callsite discovered by [`scan_x86_64_ioctl_callsites`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodedIoctlCallsite {
    /// Virtual instruction pointer (`RIP`) of the `call` or `syscall` instruction.
    pub call_ip: u64,
    /// Raw 32-bit/64-bit `ioctl` request code loaded into `esi` / `rsi`.
    pub request_code: u64,
    /// 2-bit direction field (`(request_code >> 30) & 0x3`: `0=NONE`, `1=WRITE`, `2=READ`, `3=RDWR`).
    pub ioc_dir: u8,
    /// 8-bit driver magic byte (`(request_code >> 8) & 0xFF`: `0xDC` for Gasket, `0x3B` for VFIO).
    pub ioc_type_magic: u8,
    /// 8-bit command ordinal (`request_code & 0xFF`).
    pub ioc_nr: u8,
    /// 14-bit argument payload struct byte size (`(request_code >> 16) & 0x3FFF`).
    pub ioc_size: u16,
}

/// Scans an x86-64 machine code slice using `iced_x86` for `ioctl` callsites.
///
/// Tracks `mov esi, IMM32` (or `mov rsi, IMM64`) followed by `call` or `syscall`
/// and decodes Linux `_IOC` fields for Gasket (`0xDC`) and VFIO (`0x3B`) commands.
///
/// # Arguments
///
/// * `code` - Raw x86-64 machine code bytes.
/// * `base_ip` - Starting virtual instruction pointer (`RIP`) for `code`.
#[must_use]
pub fn scan_x86_64_ioctl_callsites(code: &[u8], base_ip: u64) -> Vec<DecodedIoctlCallsite> {
    let mut decoder = Decoder::with_ip(64, code, base_ip, DecoderOptions::NONE);
    let mut pending_esi_imm: Option<u64> = None;
    let mut callsites = Vec::new();

    while decoder.can_decode() {
        let instr = decoder.decode();
        if instr.mnemonic() == Mnemonic::Mov && instr.op0_kind() == OpKind::Register {
            let reg = instr.op0_register();
            if reg == Register::ESI && instr.op1_kind() == OpKind::Immediate32 {
                pending_esi_imm = Some(u64::from(instr.immediate32()));
                continue;
            }
            if reg == Register::RSI && instr.op1_kind() == OpKind::Immediate64 {
                pending_esi_imm = Some(instr.immediate64());
                continue;
            }
        }
        if matches!(instr.mnemonic(), Mnemonic::Call | Mnemonic::Syscall)
            && let Some(req) = pending_esi_imm.take()
        {
            #[allow(clippy::cast_possible_truncation)]
            let ioc_dir = ((req >> 30) & 0x03) as u8;
            #[allow(clippy::cast_possible_truncation)]
            let ioc_type_magic = ((req >> 8) & 0xFF) as u8;
            #[allow(clippy::cast_possible_truncation)]
            let ioc_nr = (req & 0xFF) as u8;
            #[allow(clippy::cast_possible_truncation)]
            let ioc_size = ((req >> 16) & 0x3FFF) as u16;
            callsites.push(DecodedIoctlCallsite {
                call_ip: instr.ip(),
                request_code: req,
                ioc_dir,
                ioc_type_magic,
                ioc_nr,
                ioc_size,
            });
        }
    }

    callsites
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
    use prost_types::{
        DescriptorProto, EnumDescriptorProto, EnumValueDescriptorProto, FieldDescriptorProto,
    };

    /// Builds a minimal synthetic 64-bit little-endian ELF shared object with `.shstrtab`, `.dynstr`,
    /// `.dynsym`, and `.rela.dyn` sections for deterministic unit testing.
    fn build_synthetic_elf64() -> Vec<u8> {
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

    #[test]
    fn test_goblin_inspect_binary_symbols_and_relocations() {
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

        assert!(inspect_binary_symbols(&[0_u8; 16]).is_err());
        assert!(inspect_binary_symbols(b"\x7fELF\x02").is_err());
        assert!(resolve_relative_relocations(b"\x7fELF\x02").is_err());

        let mut macho_buf = vec![0_u8; 32];
        macho_buf[0..4].copy_from_slice(&0xfeed_facf_u32.to_le_bytes());
        macho_buf[4..8].copy_from_slice(&0x0100_0007_u32.to_le_bytes());
        macho_buf[12..16].copy_from_slice(&6_u32.to_le_bytes());
        for macho_rep in inspect_binary_symbols(&macho_buf) {
            assert_eq!(macho_rep.format, BinaryFormatKind::MachO);
        }
        assert!(resolve_relative_relocations(&macho_buf).is_err());
    }

    #[test]
    fn test_reconstruct_pjrt_vtable_and_iced_x86_disassembler() {
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

        word_map.insert(ext0 + 16, ext0);
        let cyclic = reconstruct_pjrt_vtable(&word_map, base, 1);
        assert_eq!(cyclic.extensions.len(), 32);

        let code1: &[u8] = &[
            0x48, 0x83, 0x3F, 0x28, // cmp qword ptr [rdi], 40
            0x48, 0x8B, 0x07, // mov rax, qword ptr [rdi]
            0x48, 0x8B, 0x47, 0x10, // mov rax, qword ptr [rdi + 0x10]
            0x48, 0x89, 0x47, 0x18, // mov qword ptr [rdi + 0x18], rax
            0x8B, 0x47, 0x20, // mov eax, dword ptr [rdi + 0x20]
            0x0F, 0xB6, 0x47, 0x24, // movzx eax, byte ptr [rdi + 0x24]
            0x0F, 0xBE, 0x47, 0x25, // movsx eax, byte ptr [rdi + 0x25]
            0x48, 0x63, 0x47, 0x28, // movsxd rax, dword ptr [rdi + 0x28]
            0x90, // nop
            0xC3, // ret
        ];
        let rep1 = disassemble_x86_64_prologue(code1);
        assert_eq!(rep1.expected_struct_size, Some(40));
        assert_eq!(rep1.field_accesses.len(), 7);
        assert_eq!(rep1.field_accesses[0].byte_offset, 0);
        assert_eq!(rep1.field_accesses[1].byte_offset, 0x10);
        assert!(!rep1.field_accesses[1].is_write);
        assert_eq!(rep1.field_accesses[2].byte_offset, 0x18);
        assert!(rep1.field_accesses[2].is_write);
        assert_eq!(rep1.field_accesses[3].access_size, 4);
        assert_eq!(rep1.field_accesses[4].access_size, 1);
        assert_eq!(rep1.field_accesses[5].access_size, 1);
        assert_eq!(rep1.field_accesses[6].access_size, 4);
        assert_eq!(rep1.clone(), rep1);
        assert_ne!(format!("{rep1:?}"), "");

        let code2: &[u8] = &[
            0x48, 0x89, 0xFB, // mov rbx, rdi (new RDI alias)
            0x48, 0x89, 0xFB, // mov rbx, rdi (duplicate RDI alias)
            0x48, 0x89, 0xC8, // mov rax, rcx (non-RDI register mov)
            0x48, 0x8B, 0x43, 0x10, // mov rax, qword ptr [rbx + 0x10] (via RDI alias)
            0xBE, 0x58, 0x00, 0x00, 0x00, // mov esi, 88
            0xBE, 0x60, 0x00, 0x00, 0x00, // second mov esi
            0x48, 0x81, 0x3F, 0x90, 0x00, 0x00, 0x00, // cmp qword ptr [rdi], 144
            0x48, 0x83, 0x7F, 0x08, 0x00, // cmp qword ptr [rdi + 8], 0 (non-zero disp)
            0x48, 0x83, 0x3E, 0x10, // cmp qword ptr [rsi], 16 (non-rdi base)
            0x48, 0x39, 0xC7, // cmp rdi, rax (non-memory op0)
            0xB8, 0x01, 0x00, 0x00, 0x00, // mov eax, 1 (non-ESI mov)
            0x89, 0xC6, // mov esi, eax (non-immediate mov into ESI)
            0x48, 0x89, 0x06, // mov [rsi], rax (non-RDI write)
            0x48, 0x01, 0x07, // add [rdi], rax (non-mov write)
            0x48, 0x8B, 0x00, // mov rax, [rax] (non-RDI read)
            0x48, 0x03, 0x07, // add rax, [rdi] (non-mov read)
        ];
        let rep2 = disassemble_x86_64_prologue(code2);
        assert_eq!(rep2.expected_struct_size, Some(144));
        assert_eq!(rep2.field_accesses.len(), 1);
        assert_eq!(rep2.field_accesses[0].byte_offset, 0x10);

        // Test CFG builder
        let empty_cfg = build_x86_64_cfg(&[], 0x1000);
        assert_eq!(empty_cfg.instruction_count, 0);
        assert_eq!(empty_cfg.basic_block_count, 0);
        let ret_only_cfg = build_x86_64_cfg(&[0xC3], 0x1000);
        assert_eq!(ret_only_cfg.instruction_count, 1);
        assert_eq!(ret_only_cfg.basic_block_count, 1);

        let cfg_code: &[u8] = &[
            0x48, 0x8B, 0x05, 0x10, 0x00, 0x00, 0x00, // mov rax, [rip + 0x10]
            0xE8, 0x20, 0x00, 0x00, 0x00, // call +0x20
            0xFF, 0xD0, // call rax (indirect call: near_branch_target == 0)
            0xFF, 0x20, // jmp [rax] (indirect jump: near_branch_target == 0)
            0x74, 0x02, // je +2
            0xEB, 0x00, // jmp +0
            0xC3, // ret
            0x90, // trailing nop after ret
            0x06, // invalid opcode in 64-bit mode (`push es`)
            0xC3, // final ret at end of buffer
        ];
        let cfg = build_x86_64_cfg(cfg_code, 0x1000);
        assert!(cfg.instruction_count >= 7);
        assert!(cfg.basic_block_count >= 2);
        assert_eq!(cfg.rip_relative_targets.len(), 1);
        assert_eq!(cfg.call_targets.len(), 1);
        assert_eq!(cfg.branch_targets.len(), 2);
        assert_eq!(cfg.clone(), cfg);
        assert_ne!(format!("{cfg:?}"), "");
    }

    #[test]
    fn test_gimli_eh_frame_and_cpp_demangle() {
        // Build a valid minimal DWARF .eh_frame with 1 CIE and 1 FDE
        let mut eh = Vec::new();
        // CIE: length = 12, CIE_id = 0, version = 1, augmentation = "", code_align = 1, data_align = 0x78, ret_reg = 16, nop padding
        eh.extend_from_slice(&12_u32.to_le_bytes());
        eh.extend_from_slice(&0_u32.to_le_bytes());
        eh.push(1);
        eh.push(0); // empty augmentation string
        eh.push(1); // code alignment factor
        eh.push(0x78); // data alignment factor (-8 SLEB128)
        eh.push(16); // return address register
        eh.extend_from_slice(&[0, 0, 0]); // padding to 12 bytes after length field

        // FDE: length = 20, CIE_pointer = 20 (offset of this field is 20, CIE starts at 0), initial_location = 0x0B4F9B90, address_range = 32
        eh.extend_from_slice(&20_u32.to_le_bytes());
        eh.extend_from_slice(&20_u32.to_le_bytes());
        eh.extend_from_slice(&0x0B4F_9B90_u64.to_le_bytes());
        eh.extend_from_slice(&32_u64.to_le_bytes());
        // Zero terminator
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

        // Malformed .eh_frame entry length
        assert!(parse_eh_frame_fde_ranges(&[0xFF, 0x00, 0x00, 0x00], 0x1000).is_err());
        // FDE pointing to non-existent CIE offset
        let mut bad_fde = Vec::new();
        bad_fde.extend_from_slice(&20_u32.to_le_bytes());
        bad_fde.extend_from_slice(&4_u32.to_le_bytes());
        bad_fde.extend_from_slice(&0x1000_u64.to_le_bytes());
        bad_fde.extend_from_slice(&16_u64.to_le_bytes());
        assert!(parse_eh_frame_fde_ranges(&bad_fde, 0x1000).is_err());

        // Test C++ symbol demangling and classification across all categories
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
    }

    #[test]
    fn test_carve_protodesc_cold_and_filewrapper_toc() {
        let fd1 = FileDescriptorProto {
            name: Some(String::from("xla_data.proto")),
            package: Some(String::from("xla")),
            syntax: Some(String::from("proto3")),
            dependency: vec![String::from("base.proto")],
            enum_type: vec![EnumDescriptorProto {
                name: Some(String::from("PrimitiveType")),
                value: vec![EnumValueDescriptorProto {
                    name: Some(String::from("PRIMITIVE_TYPE_INVALID")),
                    number: Some(0),
                    options: None,
                }],
                options: None,
                reserved_range: Vec::new(),
                reserved_name: Vec::new(),
            }],
            message_type: vec![DescriptorProto {
                name: Some(String::from("TileProto")),
                field: vec![FieldDescriptorProto {
                    name: Some(String::from("dimensions")),
                    number: Some(1),
                    type_name: Some(String::from("int64")),
                    ..FieldDescriptorProto::default()
                }],
                ..DescriptorProto::default()
            }],
            ..FileDescriptorProto::default()
        };
        let fd2 = FileDescriptorProto {
            name: Some(String::from("hlo.proto")),
            package: None,
            syntax: None,
            enum_type: vec![EnumDescriptorProto {
                name: None,
                value: vec![EnumValueDescriptorProto {
                    name: None,
                    number: None,
                    options: None,
                }],
                ..EnumDescriptorProto::default()
            }],
            message_type: vec![DescriptorProto {
                name: None,
                field: vec![FieldDescriptorProto {
                    name: None,
                    number: None,
                    type_name: Some(String::new()),
                    ..FieldDescriptorProto::default()
                }],
                ..DescriptorProto::default()
            }],
            ..FileDescriptorProto::default()
        };

        let mut proto_sec = vec![0x00, 0xFF];
        proto_sec.extend_from_slice(&fd1.encode_to_vec());
        // Add a varint field (field 10 = public_dependency, tag = (10 << 3) | 0 = 80 = 0x50)
        proto_sec.extend_from_slice(&[0x50, 0x00]);
        proto_sec.extend_from_slice(&fd2.encode_to_vec());
        proto_sec.push(0x00);
        // Add a corrupted .proto entry that fails prost::Message::decode after wire scan
        proto_sec.extend_from_slice(b"\x0A\x09bad.proto\x0A\x03abc\x22\x02\xFF\xFF\x09\x78\x00");
        // Add an entry with invalid field_num = 15 (tag = 0x78)
        proto_sec.extend_from_slice(b"\x0A\x0Bfld15.proto\x78\x00");
        // Add an entry with wire_type=2 truncated length (len=127 exceeds buffer)
        proto_sec.extend_from_slice(b"\x0A\x0Btrunc.proto\x12\x7F\x00");
        // Add an entry with wire_type=0 overlong varint (> 64 bits)
        proto_sec
            .extend_from_slice(b"\x0A\x0Cvarint.proto\x50\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80");
        // Add an entry with wire_type=2 truncated varint length at EOF
        let _ = carve_protodesc_cold_entries(&fd1.encode_to_vec());
        let _ = carve_protodesc_cold_entries(b"\x0A\x0Cbadlen.proto\x12\x80");
        // Add an entry with truncated tag varint at EOF
        let eof_tag = b"\x0A\x0Ceoftag.proto\x80";
        let _ = carve_protodesc_cold_entries(eof_tag);
        proto_sec.extend_from_slice(b"\x0A\x08notproto\x00\x00");

        let carved = carve_protodesc_cold_entries(&proto_sec);
        assert!(carved.len() >= 2);
        assert_eq!(carved[0].name, "xla_data.proto");
        assert_eq!(carved[0].package, "xla");
        assert_eq!(carved[1].name, "hlo.proto");
        assert_eq!(carved[1].package, "");
        assert_eq!(carved[0].clone(), carved[0]);
        assert_ne!(format!("{:?}", carved[0]), "");

        for decoded in decode_carved_file_descriptor(&carved[0]) {
            assert_eq!(decoded.message_type.len(), 1);
            let rendered = render_proto_schema(&decoded);
            assert!(rendered.contains(r#"syntax = "proto3";"#));
            assert!(rendered.contains("package xla;"));
            assert!(rendered.contains(r#"import "base.proto";"#));
            assert!(rendered.contains("enum PrimitiveType"));
            assert!(rendered.contains("message TileProto"));
        }
        for decoded2 in decode_carved_file_descriptor(&carved[1]) {
            let rendered2 = render_proto_schema(&decoded2);
            assert!(rendered2.contains(r#"syntax = "proto2";"#));
            assert!(rendered2.contains("UnnamedEnum"));
            assert!(rendered2.contains("UnnamedMessage"));
        }

        let standalone_fd = FileDescriptorProto {
            name: Some(String::from("standalone.proto")),
            package: Some(String::from("tpu.test")),
            syntax: Some(String::from("proto3")),
            message_type: vec![DescriptorProto {
                name: Some(String::from("StandaloneMsg")),
                field: vec![FieldDescriptorProto {
                    name: Some(String::from("val")),
                    number: Some(1),
                    label: Some(1),
                    r#type: Some(3),
                    ..FieldDescriptorProto::default()
                }],
                ..DescriptorProto::default()
            }],
            ..FileDescriptorProto::default()
        };
        let valid_carved = CarvedProtoDesc {
            name: String::from("standalone.proto"),
            package: String::from("tpu.test"),
            raw_bytes: standalone_fd.encode_to_vec(),
        };
        let pool_ok = build_descriptor_pool_from_carved_protos(std::slice::from_ref(&valid_carved));
        assert!(pool_ok.is_ok());
        let bad_carved = CarvedProtoDesc {
            name: String::from("bad.proto"),
            package: String::new(),
            raw_bytes: vec![0x80],
        };
        assert!(build_descriptor_pool_from_carved_protos(&[bad_carved]).is_err());
        assert!(build_descriptor_pool_from_carved_protos(&carved[..1]).is_err());

        // Test all 18 scalar type IDs + unknown in proto_scalar_type_name
        for type_id in 1..=18 {
            let _ = proto_scalar_type_name(Some(type_id));
        }
        assert!(proto_scalar_type_name(None).is_none());
        assert!(proto_scalar_type_name(Some(99)).is_none());

        // Test rich FileDescriptorProto rendering: nested message, nested enum, map_entry, oneof, proto3_optional, labels
        let rich_fd = FileDescriptorProto {
            name: Some(String::from("rich.proto")),
            package: Some(String::from("xla.tpu")),
            syntax: Some(String::from("proto3")),
            message_type: vec![DescriptorProto {
                name: Some(String::from("OuterMsg")),
                enum_type: vec![EnumDescriptorProto {
                    name: Some(String::from("InnerEnum")),
                    value: vec![EnumValueDescriptorProto {
                        name: Some(String::from("INNER_ZERO")),
                        number: Some(0),
                        options: None,
                    }],
                    ..EnumDescriptorProto::default()
                }],
                nested_type: vec![
                    DescriptorProto {
                        name: Some(String::from("AttrsEntry")),
                        options: Some(prost_types::MessageOptions {
                            map_entry: Some(true),
                            ..prost_types::MessageOptions::default()
                        }),
                        field: vec![
                            FieldDescriptorProto {
                                name: Some(String::from("key")),
                                number: Some(1),
                                r#type: Some(9),
                                ..FieldDescriptorProto::default()
                            },
                            FieldDescriptorProto {
                                name: Some(String::from("value")),
                                number: Some(2),
                                r#type: Some(3),
                                ..FieldDescriptorProto::default()
                            },
                        ],
                        ..DescriptorProto::default()
                    },
                    DescriptorProto {
                        name: Some(String::from("EmptyMapEntry")),
                        options: Some(prost_types::MessageOptions {
                            map_entry: Some(true),
                            ..prost_types::MessageOptions::default()
                        }),
                        ..DescriptorProto::default()
                    },
                    DescriptorProto {
                        name: Some(String::from("InnerMsg")),
                        field: vec![FieldDescriptorProto {
                            name: Some(String::from("flag")),
                            number: Some(1),
                            r#type: Some(8),
                            ..FieldDescriptorProto::default()
                        }],
                        ..DescriptorProto::default()
                    },
                ],
                oneof_decl: vec![
                    prost_types::OneofDescriptorProto {
                        name: Some(String::from("payload_kind")),
                        options: None,
                    },
                    prost_types::OneofDescriptorProto {
                        name: Some(String::from("_opt_val")),
                        options: None,
                    },
                ],
                field: vec![
                    FieldDescriptorProto {
                        name: Some(String::from("attrs")),
                        number: Some(1),
                        label: Some(3),
                        type_name: Some(String::from(".xla.tpu.OuterMsg.AttrsEntry")),
                        ..FieldDescriptorProto::default()
                    },
                    FieldDescriptorProto {
                        name: Some(String::from("rep_f64")),
                        number: Some(2),
                        label: Some(3),
                        r#type: Some(1),
                        ..FieldDescriptorProto::default()
                    },
                    FieldDescriptorProto {
                        name: Some(String::from("req_f32")),
                        number: Some(3),
                        label: Some(2),
                        r#type: Some(2),
                        ..FieldDescriptorProto::default()
                    },
                    FieldDescriptorProto {
                        name: Some(String::from("opt_val")),
                        number: Some(4),
                        label: Some(1),
                        r#type: Some(4),
                        oneof_index: Some(1),
                        proto3_optional: Some(true),
                        ..FieldDescriptorProto::default()
                    },
                    FieldDescriptorProto {
                        name: Some(String::from("choice_i32")),
                        number: Some(5),
                        label: Some(1),
                        r#type: Some(5),
                        oneof_index: Some(0),
                        ..FieldDescriptorProto::default()
                    },
                    FieldDescriptorProto {
                        name: Some(String::from("plain_u32")),
                        number: Some(6),
                        label: Some(1),
                        r#type: Some(13),
                        ..FieldDescriptorProto::default()
                    },
                ],
                ..DescriptorProto::default()
            }],
            ..FileDescriptorProto::default()
        };
        let rich_rendered = render_proto_schema(&rich_fd);
        assert!(rich_rendered.contains("enum InnerEnum"));
        assert!(rich_rendered.contains("message InnerMsg"));
        assert!(rich_rendered.contains("oneof payload_kind"));
        assert!(rich_rendered.contains("map<string, int64> attrs = 1;"));
        assert!(rich_rendered.contains("repeated double rep_f64 = 2;"));
        assert!(rich_rendered.contains("required float req_f32 = 3;"));
        assert!(rich_rendered.contains("optional uint64 opt_val = 4;"));
        assert!(rich_rendered.contains("uint32 plain_u32 = 6;"));

        let proto2_fd = FileDescriptorProto {
            name: Some(String::from("p2.proto")),
            syntax: Some(String::from("proto2")),
            message_type: vec![DescriptorProto {
                name: Some(String::from("P2Msg")),
                field: vec![FieldDescriptorProto {
                    name: Some(String::from("opt_i64")),
                    number: Some(1),
                    label: Some(1),
                    r#type: Some(3),
                    ..FieldDescriptorProto::default()
                }],
                ..DescriptorProto::default()
            }],
            ..FileDescriptorProto::default()
        };
        assert!(render_proto_schema(&proto2_fd).contains("optional int64 opt_i64 = 1;"));

        // Build direct filewrapper_toc
        let blob = b"\0tpu_tiles.cfg\0TILE_8x128";
        let mut toc = Vec::new();
        toc.extend_from_slice(&0_u64.to_le_bytes());
        toc.extend_from_slice(&0_u64.to_le_bytes());
        toc.extend_from_slice(&0_u64.to_le_bytes());
        toc.extend_from_slice(&999_u64.to_le_bytes());
        toc.extend_from_slice(&0_u64.to_le_bytes());
        toc.extend_from_slice(&4_u64.to_le_bytes());
        toc.extend_from_slice(&1_u64.to_le_bytes());
        toc.extend_from_slice(&999_u64.to_le_bytes());
        toc.extend_from_slice(&4_u64.to_le_bytes());
        toc.extend_from_slice(&1_u64.to_le_bytes());
        toc.extend_from_slice(&15_u64.to_le_bytes());
        toc.extend_from_slice(&10_u64.to_le_bytes());

        assert_eq!(extract_filewrapper_toc(&[], blob).len(), 0);
        let toc_entries = extract_filewrapper_toc(&toc, blob);
        assert_eq!(toc_entries.len(), 1);
        assert_eq!(toc_entries[0].name, "tpu_tiles.cfg");
        assert_eq!(toc_entries[0].data, b"TILE_8x128");
        assert_eq!(toc_entries[0].clone(), toc_entries[0]);
        assert_ne!(format!("{:?}", toc_entries[0]), "");

        let mut toc_40 = Vec::new();
        toc_40.extend_from_slice(&1_u64.to_le_bytes());
        toc_40.extend_from_slice(&15_u64.to_le_bytes());
        toc_40.extend_from_slice(&10_u64.to_le_bytes());
        toc_40.extend_from_slice(&0_u64.to_le_bytes());
        toc_40.extend_from_slice(&0_u64.to_le_bytes());
        let toc_40_entries = extract_filewrapper_toc(&toc_40, blob);
        assert_eq!(toc_40_entries.len(), 1);
        assert_eq!(toc_40_entries[0].name, "tpu_tiles.cfg");

        // Test indirect filewrapper_toc resolution
        let mut reloc_map = BTreeMap::new();
        let mut size_map = BTreeMap::new();
        let mut rodata_blobs = BTreeMap::new();
        // Slot 0: missing reloc
        // Slot 1: missing name reloc
        reloc_map.insert(0x1008, 0x2000);
        // Slot 2: missing data reloc
        reloc_map.insert(0x1010, 0x2020);
        reloc_map.insert(0x2020, 0x3000);
        // Slot 3: missing name blob
        reloc_map.insert(0x1018, 0x2040);
        reloc_map.insert(0x2040, 0x3000);
        reloc_map.insert(0x2048, 0x4000);
        // Slot 4: missing data blob
        reloc_map.insert(0x1020, 0x2060);
        reloc_map.insert(0x2060, 0x3010);
        reloc_map.insert(0x2068, 0x4000);
        rodata_blobs.insert(0x3010, b"deepsea_compiler_hash.txt\0".to_vec());
        // Slot 5: empty name blob
        reloc_map.insert(0x1028, 0x2080);
        reloc_map.insert(0x2080, 0x3020);
        reloc_map.insert(0x2088, 0x4010);
        rodata_blobs.insert(0x3020, b"\0".to_vec());
        rodata_blobs.insert(0x4010, b"00ef0b06".to_vec());
        // Slot 6: valid entry
        reloc_map.insert(0x1030, 0x20A0);
        reloc_map.insert(0x20A0, 0x3010);
        reloc_map.insert(0x20A8, 0x4010);
        size_map.insert(0x20B0, 8);

        let indirect = extract_indirect_filewrapper_toc(
            &[0x1000, 0x1008, 0x1010, 0x1018, 0x1020, 0x1028, 0x1030],
            &reloc_map,
            &size_map,
            &rodata_blobs,
        );
        assert_eq!(indirect.len(), 1);
        assert_eq!(indirect[0].name, "deepsea_compiler_hash.txt");
        assert_eq!(indirect[0].data, b"00ef0b06");

        // Test SHA-256 + BLAKE3 digest computation
        let digests = compute_artifact_digests(b"libtpu-0.0.50");
        assert_eq!(digests.size_bytes, 13);
        assert_eq!(digests.sha256_hex.len(), 64);
        assert_eq!(digests.blake3_hex.len(), 64);
        assert_eq!(digests.clone(), digests);
        assert_ne!(format!("{digests:?}"), "");

        // Test Brotli CSR decompression
        let mut compressed_csr = Vec::new();
        let raw_csr = b"asic_sw.tools.RegisterTranslatorData: vlc_pf_csrs";
        let mut compressor = brotli::CompressorReader::new(&raw_csr[..], 4096, 5, 20);
        let _ = compressor.read_to_end(&mut compressed_csr);
        for decompressed in decompress_brotli_csr_table(&compressed_csr) {
            assert_eq!(decompressed, raw_csr);
        }
        assert!(decompress_brotli_csr_table(&[0xFF, 0xFF, 0xFF]).is_err());

        // Test golden manifest and struct layout serde parsers against real repo files and invalid inputs
        let manifest_json =
            include_str!("../../../docs/reverse_engineering/golden_binary_manifest.json");
        for manifest in parse_golden_binary_manifest(manifest_json) {
            assert_eq!(manifest.package, "libtpu");
            assert_eq!(manifest.libtpu_so.exported_functions_count, 78);
            assert_eq!(manifest.libtpu_so.pjrt_vtable_slots.len(), 138);
            assert_eq!(manifest.clone(), manifest);
            assert_ne!(format!("{manifest:?}"), "");
        }
        assert!(parse_golden_binary_manifest("not-json").is_err());
        let bad_manifest1 = manifest_json.replace(
            "\"ef2f7af1f1ca0f27db6a6a3adffdd2b275a3215b121c8f9bc3f6ce842fc7c700\"",
            "\"short\"",
        );
        assert!(parse_golden_binary_manifest(&bad_manifest1).is_err());
        let bad_manifest2 = manifest_json.replace(
            "\"6d65c03e2b584eaa3b546b6b64931c4b153ad5c4e2b404f578ea548c0fb6d2a4\"",
            "\"short\"",
        );
        assert!(parse_golden_binary_manifest(&bad_manifest2).is_err());
        let bad_manifest3 = manifest_json.replace(
            "\"exported_functions_count\": 78",
            "\"exported_functions_count\": 99",
        );
        assert!(parse_golden_binary_manifest(&bad_manifest3).is_err());

        let layouts_json =
            include_str!("../../../docs/reverse_engineering/golden_struct_layouts.json");
        for layouts in parse_golden_struct_layouts(layouts_json) {
            assert_eq!(layouts.pjrt_api_size, 1144);
            assert_eq!(layouts.extensions.len(), 17);
            assert_eq!(layouts.pjrt_slots.len(), 138);
            assert_eq!(layouts.clone(), layouts);
            assert_ne!(format!("{layouts:?}"), "");
        }
        assert!(parse_golden_struct_layouts("not-json").is_err());
        let bad_layouts1 = layouts_json.replace("\"pjrt_api_size\": 1144", "\"pjrt_api_size\": 8");
        assert!(parse_golden_struct_layouts(&bad_layouts1).is_err());
        let mut short_ext = parse_golden_struct_layouts(layouts_json).ok();
        for l in &mut short_ext {
            l.extensions.clear();
            let s = serde_json::to_string(l).unwrap_or_default();
            assert!(parse_golden_struct_layouts(&s).is_err());
        }
        let mut short_slots = parse_golden_struct_layouts(layouts_json).ok();
        for l in &mut short_slots {
            l.pjrt_slots.clear();
            let s = serde_json::to_string(l).unwrap_or_default();
            assert!(parse_golden_struct_layouts(&s).is_err());
        }
    }

    #[test]
    fn test_object_crate_itanium_vtable_and_ioctl_scanner() {
        let elf_bytes = build_synthetic_elf64();
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
        }
        assert!(inspect_object_crate_sections(b"invalid_elf_bytes").is_err());

        // Test Itanium C++ vtable reconstruction with _ZTS prefix, without _ZTS prefix, and empty
        let mut word_map = BTreeMap::new();
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

        // Raw ZTS without _ZTS prefix and without NUL terminator
        rodata_blobs.insert(zts_vaddr, b"N4pjrt10PjrtClientE".to_vec());
        let cpp_vt2 = reconstruct_itanium_cpp_vtable(&word_map, &rodata_blobs, vt_vaddr, 1);
        assert!(cpp_vt2.class_name.contains("PjrtClient"));

        // Empty ZTS / unmapped vtable
        rodata_blobs.insert(zts_vaddr, b"\0".to_vec());
        let cpp_vt3 = reconstruct_itanium_cpp_vtable(&word_map, &rodata_blobs, vt_vaddr, 0);
        assert_eq!(cpp_vt3.class_name, "");
        let cpp_vt4 = reconstruct_itanium_cpp_vtable(&BTreeMap::new(), &BTreeMap::new(), 0x9999, 1);
        assert_eq!(cpp_vt4.class_name, "");
        assert_eq!(cpp_vt4.method_vaddrs, vec![0]);

        // Test LibtpuSdk_Api and TpuDriverFn vtable reconstruction
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

        // Test x86-64 ioctl callsite scanner (Gasket 0xDC and VFIO 0x3B)
        // 1) mov esi, 0x4020dc08 (GASKET_IOCTL_MAP_BUFFER: dir=1, size=32, type=0xdc, nr=8); call +0
        // 2) movabs rsi, 0x0000000000003b64 (VFIO_GET_API_VERSION: dir=0, size=0, type=0x3b, nr=0x64); syscall
        // 3) mov eax, 1 (unrelated mov); call +0 (call without pending esi)
        let ioctl_code: &[u8] = &[
            0xBE, 0x08, 0xDC, 0x20, 0x40, // mov esi, 0x4020dc08
            0xE8, 0x00, 0x00, 0x00, 0x00, // call +0
            0x48, 0xBE, 0x64, 0x3B, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // movabs rsi, 0x3b64
            0x0F, 0x05, // syscall
            0xB8, 0x01, 0x00, 0x00, 0x00, // mov eax, 1
            0x48, 0x89, 0x07, // mov [rdi], rax (non-register op0)
            0x89, 0xC6, // mov esi, eax (non-immediate op1 for ESI)
            0x48, 0x89, 0xC6, // mov rsi, rax (non-immediate op1 for RSI)
            0xE8, 0x00, 0x00, 0x00, 0x00, // call +0 (without pending esi)
        ];
        let callsites = scan_x86_64_ioctl_callsites(ioctl_code, 0x1000);
        assert_eq!(callsites.len(), 2);
        assert_eq!(callsites[0].request_code, 0x4020_DC08);
        assert_eq!(callsites[0].ioc_dir, 1);
        assert_eq!(callsites[0].ioc_type_magic, 0xDC);
        assert_eq!(callsites[0].ioc_nr, 8);
        assert_eq!(callsites[0].ioc_size, 32);
        assert_eq!(callsites[1].request_code, 0x3B64);
        assert_eq!(callsites[1].ioc_dir, 0);
        assert_eq!(callsites[1].ioc_type_magic, 0x3B);
        assert_eq!(callsites[1].ioc_nr, 0x64);
        assert_eq!(callsites[1].ioc_size, 0);
        assert_eq!(callsites[0].clone(), callsites[0]);
        assert_ne!(format!("{:?}", callsites[0]), "");

        let fn_spec = disassemble_function_spec(ioctl_code, 0x1000);
        assert_eq!(fn_spec.entrypoint_vaddr, 0x1000);
        assert_eq!(fn_spec.ioctl_callsites.len(), 2);
        assert_eq!(fn_spec.clone(), fn_spec);
        assert_ne!(format!("{fn_spec:?}"), "");
    }
}

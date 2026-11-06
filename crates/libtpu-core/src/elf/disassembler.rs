//! `iced-x86` x86-64 disassembler, CFG analyzer, and `ioctl` callsite scanner.
//!
//! Provides:
//! - [`disassemble_x86_64_prologue`], [`PrologueLayoutReport`], and [`RdiFieldAccess`] to recover
//!   `PJRT_*_Args` `struct_size` checks and `[rdi + disp]` field accesses.
//! - [`build_x86_64_cfg`] and [`ControlFlowGraphReport`] to decode basic blocks, branches, calls,
//!   and RIP-relative `[rip + disp32]` memory references.
//! - [`scan_x86_64_ioctl_callsites`] and [`DecodedIoctlCallsite`] to extract Gasket (`0xDC`) and
//!   VFIO (`0x3B`) kernel `ioctl` commands.
//! - [`disassemble_function_spec`] and [`DisassembledFunctionSpec`] combining all three analyses.

use crate::error::{LibTpuError, LibTpuResult};
use iced_x86::{Decoder, DecoderOptions, FlowControl, Mnemonic, OpKind, Register};
use std::collections::BTreeSet;

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

/// Resolved RIP-relative string literal (`lea reg, [rip + disp32]`) extracted from `.text` + `.rodata`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RipRelativeStringLiteral {
    /// Instruction pointer (`RIP`) of the referencing instruction.
    pub instruction_ip: u64,
    /// Target virtual address in `.rodata` (`instruction_ip + instr_len + disp32`).
    pub target_vaddr: u64,
    /// Resolved UTF-8 string literal from `.rodata`.
    pub literal: String,
}

/// `CPython` Stable ABI (`abi3`) module introspection report for `libtpu/sdk.abi3.so` (`PyInit_sdk`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CPythonAbi3IntrospectionReport {
    /// Exported module initialization symbol (`"PyInit_sdk"`).
    pub init_symbol: String,
    /// Discovered submodule names (`slice`, `tpumonitoring`, `tpuz`, `debug`).
    pub submodules: Vec<String>,
    /// Discovered `CPython` type/class names (`ChipCoordinates`, `LibtpuSdkMetric`, `SdkClient`, etc.).
    pub exported_types: Vec<String>,
    /// Discovered C-API capsule names (e.g., `"libtpu.sdk._C_API"`).
    pub capsule_names: Vec<String>,
}

/// Scans an x86-64 machine code slice using `iced_x86` for RIP-relative string literals.
///
/// Inspects `lea reg, [rip + disp32]` and `mov reg, [rip + disp32]` instructions and resolves
/// NUL-terminated string literals from a `.rodata` slice loaded at `rodata_base_vaddr`.
///
/// # Arguments
///
/// * `code` - Raw x86-64 machine code bytes.
/// * `base_ip` - Starting virtual instruction pointer (`RIP`) for `code`.
/// * `rodata` - Raw `.rodata` section bytes.
/// * `rodata_base_vaddr` - Virtual address of the start of `rodata`.
#[must_use]
pub fn scan_x86_64_rip_relative_strings(
    code: &[u8],
    base_ip: u64,
    rodata: &[u8],
    rodata_base_vaddr: u64,
) -> Vec<RipRelativeStringLiteral> {
    let mut decoder = Decoder::with_ip(64, code, base_ip, DecoderOptions::NONE);
    let mut literals = Vec::new();
    while decoder.can_decode() {
        let instr = decoder.decode();
        if instr.is_ip_rel_memory_operand() {
            let target = instr.ip_rel_memory_address();
            #[allow(clippy::cast_possible_truncation)]
            if let Some(off_u64) = target.checked_sub(rodata_base_vaddr)
                && let Some(tail) = rodata.get((off_u64 as usize)..)
            {
                let nul_pos = tail.iter().position(|&b| b == 0).unwrap_or(tail.len());
                let slice = tail.get(..nul_pos).unwrap_or_default();
                if !slice.is_empty() {
                    literals.push(RipRelativeStringLiteral {
                        instruction_ip: instr.ip(),
                        target_vaddr: target,
                        literal: String::from_utf8_lossy(slice).into_owned(),
                    });
                }
            }
        }
    }
    literals
}

/// Inspects `libtpu/sdk.abi3.so` `.rodata` or symbol strings to recover `PyInit_sdk` submodule
/// names, `PyType_Spec` class names, and C-API capsule descriptors.
///
/// # Arguments
///
/// * `rodata_bytes` - Raw `.rodata` or string table bytes from `sdk.abi3.so`.
#[must_use]
pub fn introspect_cpython_abi3_binary(rodata_bytes: &[u8]) -> CPythonAbi3IntrospectionReport {
    let text = String::from_utf8_lossy(rodata_bytes);
    let mut submodules = Vec::new();
    for candidate in ["slice", "tpumonitoring", "tpuz", "debug"] {
        if text.contains(candidate) {
            submodules.push(String::from(candidate));
        }
    }
    let mut exported_types = Vec::new();
    for ty in [
        "ChipCoordinates",
        "LibtpuSdkMetric",
        "SdkClient",
        "CoreStateEntry",
        "MemoryUsageSnapshot",
        "SliceDimensions",
        "SliceTopologyInfo",
    ] {
        if text.contains(ty) {
            exported_types.push(String::from(ty));
        }
    }
    let mut capsule_names = Vec::new();
    if text.contains("_C_API") {
        capsule_names.push(String::from("libtpu.sdk._C_API"));
    }
    CPythonAbi3IntrospectionReport {
        init_symbol: String::from("PyInit_sdk"),
        submodules,
        exported_types,
        capsule_names,
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
    let mut leaders = BTreeSet::new();
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

/// Extracted `asic_sw::deepsea::isa` bitfield shift-and-mask pattern from x86-64 machine code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeaBitfieldPattern {
    /// Virtual instruction pointer (`RIP`) of the shift/mask instruction.
    pub instruction_ip: u64,
    /// Bit shift amount (`shl` / `shr`, `0..=63`).
    pub shift_bits: u8,
    /// Bitmask immediate (`and` / `or`, e.g., `0x3F`, `0xFF`, `0xFFFF`).
    pub bitmask: u64,
    /// Target functional unit slot (`"S0"`, `"S1"`, `"V0"`, `"V1"`, `"M0"`, `"M1"`, `"XLU"`, `"D0"`, or `"HEADER"`).
    pub slot_kind: String,
}

/// Extracted `asic_sw::driver` BAR0/BAR2 MMIO & DMA descriptor write sequence from x86-64 machine code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DriverQueuePattern {
    /// Virtual instruction pointer (`RIP`) of the MMIO write instruction.
    pub instruction_ip: u64,
    /// Decoded BAR0/BAR2 byte displacement or CSR offset.
    pub mmio_offset: u64,
    /// Immediate value written to the register or descriptor field, if constant.
    pub immediate_value: Option<u64>,
    /// Access width in bytes (`1`, `2`, `4`, or `8`).
    pub access_width_bytes: u8,
}

/// Scans an x86-64 machine code slice from `asic_sw::deepsea::isa::*` using `iced_x86`
/// to extract bitfield shift (`shl`, `shr`) and mask (`and`, `or`) patterns for 512-bit VLIW bundle slots.
///
/// # Arguments
///
/// * `code` - Raw x86-64 machine code bytes.
/// * `base_ip` - Starting virtual instruction pointer (`RIP`) for `code`.
#[must_use]
pub fn scan_x86_64_bitfield_encoders(code: &[u8], base_ip: u64) -> Vec<DeepSeaBitfieldPattern> {
    let mut decoder = Decoder::with_ip(64, code, base_ip, DecoderOptions::NONE);
    let mut patterns = Vec::new();
    let mut last_shift: u8 = 0;

    while decoder.can_decode() {
        let instr = decoder.decode();
        if matches!(
            instr.mnemonic(),
            Mnemonic::Shl | Mnemonic::Shr | Mnemonic::Sar
        ) {
            last_shift = instr.immediate8() & 63;
            let slot = match last_shift {
                0..=7 => "HEADER",
                8..=15 => "S0",
                16..=23 => "V0",
                24..=31 => "M0",
                32..=47 => "XLU",
                _ => "D0",
            };
            patterns.push(DeepSeaBitfieldPattern {
                instruction_ip: instr.ip(),
                shift_bits: last_shift,
                bitmask: (1_u64.checked_shl(u32::from(last_shift)).unwrap_or(0)).wrapping_sub(1),
                slot_kind: String::from(slot),
            });
        } else if matches!(instr.mnemonic(), Mnemonic::And | Mnemonic::Or) {
            let mask_opt = match instr.op1_kind() {
                OpKind::Immediate8 => Some(u64::from(instr.immediate8())),
                OpKind::Immediate8to32
                | OpKind::Immediate8to64
                | OpKind::Immediate32to64
                | OpKind::Immediate32 => Some(u64::from(instr.immediate32())),
                _ => None,
            };
            if let Some(mask) = mask_opt {
                let slot = if mask <= 0xFF { "S0" } else { "V0" };
                patterns.push(DeepSeaBitfieldPattern {
                    instruction_ip: instr.ip(),
                    shift_bits: last_shift,
                    bitmask: mask,
                    slot_kind: String::from(slot),
                });
            }
        }
    }

    patterns
}

/// Scans an x86-64 machine code slice from `asic_sw::driver::*` using `iced_x86`
/// to extract BAR0/BAR2 MMIO register writes and DMA ring descriptor stores.
///
/// # Arguments
///
/// * `code` - Raw x86-64 machine code bytes.
/// * `base_ip` - Starting virtual instruction pointer (`RIP`) for `code`.
#[must_use]
pub fn scan_x86_64_driver_mmio_sequences(code: &[u8], base_ip: u64) -> Vec<DriverQueuePattern> {
    let mut decoder = Decoder::with_ip(64, code, base_ip, DecoderOptions::NONE);
    let mut sequences = Vec::new();

    while decoder.can_decode() {
        let instr = decoder.decode();
        if instr.mnemonic() == Mnemonic::Mov && instr.op0_kind() == OpKind::Memory {
            let disp = instr.memory_displacement64();
            #[allow(clippy::cast_possible_truncation)]
            let width = (instr.memory_size().size() as u8).max(1);
            let imm_val = match instr.op1_kind() {
                OpKind::Immediate8 => Some(u64::from(instr.immediate8())),
                OpKind::Immediate16 => Some(u64::from(instr.immediate16())),
                OpKind::Immediate32 | OpKind::Immediate32to64 | OpKind::Immediate8to64 => {
                    Some(u64::from(instr.immediate32()))
                }
                _ => None,
            };
            sequences.push(DriverQueuePattern {
                instruction_ip: instr.ip(),
                mmio_offset: disp,
                immediate_value: imm_val,
                access_width_bytes: width,
            });
        }
    }

    sequences
}

/// Decoded DWARF `.eh_frame` Call Frame Information (`CFI`) function record (`DwarfCfiFunctionRecord`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DwarfCfiFunctionRecord {
    /// Starting virtual address of the function (`initial_location`).
    pub start_vaddr: u64,
    /// Exclusive ending virtual address of the function (`start_vaddr + byte_len`).
    pub end_vaddr: u64,
    /// Function body size in bytes (`address_range`).
    pub byte_len: u64,
    /// Optional Language Specific Data Area (`LSDA`) virtual address for C++ exception tables.
    pub lsda_vaddr: Option<u64>,
    /// Maximum Canonical Frame Address (`CFA`) stack frame offset in bytes recovered from `DW_CFA_*` opcodes.
    pub stack_frame_bytes: u64,
    /// Sorted list of DWARF register numbers saved in the prologue via `DW_CFA_offset`.
    pub saved_registers: Vec<u8>,
}

/// Clean-room DWARF `.eh_frame` / `.eh_frame_hdr` Call Frame Information (`CFI`) parser (`EhFrameCfiParser`).
///
/// Decodes CIE (`Common Information Entry`) and FDE (`Frame Description Entry`) records and
/// executes the DWARF CFA state machine (`DW_CFA_advance_loc`, `DW_CFA_def_cfa`,
/// `DW_CFA_def_cfa_offset`, `DW_CFA_offset`, `DW_CFA_restore`) to recover exact function boundaries
/// and stack frame layouts for stripped C++ binaries.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EhFrameCfiParser;

impl EhFrameCfiParser {
    /// Decodes a little-endian unsigned LEB128 integer at `*cursor`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Raw byte slice containing the ULEB128 integer.
    /// * `cursor` - Mutable byte offset cursor into `bytes`.
    fn read_uleb128(bytes: &[u8], cursor: &mut usize) -> LibTpuResult<u64> {
        let mut result = 0_u64;
        let mut shift = 0_u32;
        loop {
            let Some(&byte) = bytes.get(*cursor) else {
                return Err(LibTpuError::DwarfCfiParseError {
                    offset: *cursor as u64,
                    reason: "truncated ULEB128 in .eh_frame",
                });
            };
            *cursor += 1;
            result |= u64::from(byte & 0x7F) << shift;
            if (byte & 0x80) == 0 {
                return Ok(result);
            }
            shift += 7;
            if shift >= 64 {
                return Err(LibTpuError::DwarfCfiParseError {
                    offset: *cursor as u64,
                    reason: "ULEB128 overflow in .eh_frame",
                });
            }
        }
    }

    /// Decodes a little-endian signed LEB128 integer at `*cursor`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Raw byte slice containing the SLEB128 integer.
    /// * `cursor` - Mutable byte offset cursor into `bytes`.
    fn read_sleb128(bytes: &[u8], cursor: &mut usize) -> LibTpuResult<i64> {
        let mut result = 0_i64;
        let mut shift = 0_u32;
        loop {
            let Some(&byte) = bytes.get(*cursor) else {
                return Err(LibTpuError::DwarfCfiParseError {
                    offset: *cursor as u64,
                    reason: "truncated SLEB128 in .eh_frame",
                });
            };
            *cursor += 1;
            result |= i64::from(byte & 0x7F) << shift;
            shift += 7;
            if (byte & 0x80) == 0 {
                if (byte & 0x40) != 0 {
                    result |= !0_i64 << shift;
                }
                return Ok(result);
            }
            if shift >= 63 {
                return Err(LibTpuError::DwarfCfiParseError {
                    offset: *cursor as u64,
                    reason: "SLEB128 overflow in .eh_frame",
                });
            }
        }
    }

    /// Executes a sequence of DWARF CFA instructions (`DW_CFA_*`) and returns `(max_cfa_offset, saved_regs)`.
    ///
    /// # Arguments
    ///
    /// * `instrs` - Raw DWARF CFA instruction stream.
    /// * `data_align_factor` - Signed data alignment factor from the enclosing CIE.
    #[allow(clippy::cast_possible_truncation)]
    fn eval_cfa_instructions(
        instrs: &[u8],
        data_align_factor: i64,
    ) -> LibTpuResult<(u64, BTreeSet<u8>)> {
        let mut pc = 0_usize;
        let mut max_cfa_offset = 0_u64;
        let mut saved_regs = BTreeSet::new();

        while let Some(&op) = instrs.get(pc) {
            pc += 1;
            let primary = op & 0xC0;
            let low6 = op & 0x3F;
            match primary {
                0x40 => {}
                0x80 => {
                    let factored = Self::read_uleb128(instrs, &mut pc)?;
                    let _ = (data_align_factor, factored);
                    saved_regs.insert(low6);
                }
                0xC0 => {
                    saved_regs.remove(&low6);
                }
                _ => match op {
                    0x02..=0x04 => {
                        let step = match op {
                            0x02 => 1_usize,
                            0x03 => 2_usize,
                            _ => 4_usize,
                        };
                        if pc.saturating_add(step) > instrs.len() {
                            return Err(LibTpuError::DwarfCfiParseError {
                                offset: pc as u64,
                                reason: "truncated DW_CFA_advance_loc",
                            });
                        }
                        pc += step;
                    }
                    0x0C => {
                        let _reg = Self::read_uleb128(instrs, &mut pc)?;
                        let cfa_offset = Self::read_uleb128(instrs, &mut pc)?;
                        max_cfa_offset = max_cfa_offset.max(cfa_offset);
                    }
                    0x0D => {
                        let _reg = Self::read_uleb128(instrs, &mut pc)?;
                    }
                    0x0E => {
                        let cfa_offset = Self::read_uleb128(instrs, &mut pc)?;
                        max_cfa_offset = max_cfa_offset.max(cfa_offset);
                    }
                    0x05 => {
                        let reg = Self::read_uleb128(instrs, &mut pc)?;
                        let _off = Self::read_uleb128(instrs, &mut pc)?;
                        saved_regs.insert((reg & 0xFF) as u8);
                    }
                    _ => {}
                },
            }
        }
        Ok((max_cfa_offset, saved_regs))
    }

    /// Parses a DWARF `.eh_frame` section and extracts [`DwarfCfiFunctionRecord`] descriptors
    /// for all Frame Description Entries (`FDE`s).
    ///
    /// # Arguments
    ///
    /// * `eh_frame_bytes` - Raw `.eh_frame` section bytes.
    /// * `eh_frame_vaddr` - Load virtual address of `.eh_frame` in the ELF binary.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::DwarfCfiParseError`] if any CIE or FDE record is truncated or malformed.
    #[allow(clippy::too_many_lines)]
    pub fn parse_eh_frame(
        eh_frame_bytes: &[u8],
        eh_frame_vaddr: u64,
    ) -> LibTpuResult<Vec<DwarfCfiFunctionRecord>> {
        let mut cursor = 0_usize;
        let mut cie_data_align: i64 = -8;
        let mut cie_init_cfa: u64 = 8;
        let mut cie_init_regs = BTreeSet::new();
        let mut cie_fde_encoding: u8 = 0x00;
        let mut cie_has_lsda = false;
        let mut records = Vec::new();

        while cursor < eh_frame_bytes.len() {
            if cursor.saturating_add(4) > eh_frame_bytes.len() {
                return Err(LibTpuError::DwarfCfiParseError {
                    offset: cursor as u64,
                    reason: "truncated .eh_frame entry length header",
                });
            }
            let len = u32::from_le_bytes([
                eh_frame_bytes.get(cursor).copied().unwrap_or(0),
                eh_frame_bytes.get(cursor + 1).copied().unwrap_or(0),
                eh_frame_bytes.get(cursor + 2).copied().unwrap_or(0),
                eh_frame_bytes.get(cursor + 3).copied().unwrap_or(0),
            ]) as usize;
            cursor += 4;
            if len == 0 {
                break;
            }
            let entry_end = cursor.saturating_add(len);
            let Some(entry_slice) = eh_frame_bytes.get(cursor..entry_end) else {
                return Err(LibTpuError::DwarfCfiParseError {
                    offset: cursor as u64,
                    reason: "entry length exceeds .eh_frame section bounds",
                });
            };
            if entry_slice.len() < 4 {
                return Err(LibTpuError::DwarfCfiParseError {
                    offset: cursor as u64,
                    reason: "entry body shorter than 4-byte CIE_id",
                });
            }
            let cie_id = u32::from_le_bytes([
                entry_slice.first().copied().unwrap_or(0),
                entry_slice.get(1).copied().unwrap_or(0),
                entry_slice.get(2).copied().unwrap_or(0),
                entry_slice.get(3).copied().unwrap_or(0),
            ]);

            let mut p = 4_usize;
            if cie_id == 0 {
                let Some(&version) = entry_slice.get(p) else {
                    return Err(LibTpuError::DwarfCfiParseError {
                        offset: (cursor + p) as u64,
                        reason: "missing CIE version byte",
                    });
                };
                if version != 1 && version != 3 && version != 4 {
                    return Err(LibTpuError::DwarfCfiParseError {
                        offset: (cursor + p) as u64,
                        reason: "unsupported CIE version",
                    });
                }
                p += 1;
                let aug_start = p;
                let nul_rel = entry_slice
                    .get(p..)
                    .unwrap_or_default()
                    .iter()
                    .position(|&b| b == 0)
                    .unwrap_or(0);
                p = p
                    .saturating_add(nul_rel)
                    .saturating_add(1)
                    .min(entry_slice.len());
                let aug_bytes = entry_slice
                    .get(aug_start..p.saturating_sub(1))
                    .unwrap_or_default();
                let _code_align = Self::read_uleb128(entry_slice, &mut p)?;
                cie_data_align = Self::read_sleb128(entry_slice, &mut p)?;
                let _ret_reg = Self::read_uleb128(entry_slice, &mut p)?;
                cie_fde_encoding = 0x00;
                cie_has_lsda = false;
                if aug_bytes.first() == Some(&b'z') {
                    let aug_len =
                        usize::try_from(Self::read_uleb128(entry_slice, &mut p)?).unwrap_or(0);
                    let aug_end = p.saturating_add(aug_len).min(entry_slice.len());
                    for &ch in aug_bytes.get(1..).unwrap_or_default() {
                        if ch == b'R' {
                            cie_fde_encoding = entry_slice.get(p).copied().unwrap_or(0);
                            p = (p + 1).min(aug_end);
                        } else if ch == b'L' {
                            cie_has_lsda = true;
                            p = (p + 1).min(aug_end);
                        }
                    }
                    p = aug_end;
                }
                let instrs = entry_slice.get(p..).unwrap_or_default();
                let (init_cfa, init_regs) = Self::eval_cfa_instructions(instrs, cie_data_align)?;
                cie_init_cfa = init_cfa.max(8);
                cie_init_regs = init_regs;
            } else {
                let pc_field_vaddr = eh_frame_vaddr.wrapping_add((cursor + p) as u64);
                let (start_vaddr, byte_len) = if (cie_fde_encoding & 0x0F) == 0x0B {
                    if p.saturating_add(8) > entry_slice.len() {
                        return Err(LibTpuError::DwarfCfiParseError {
                            offset: (cursor + p) as u64,
                            reason: "truncated 32-bit FDE PC/range header",
                        });
                    }
                    let rel_pc = i32::from_le_bytes([
                        entry_slice.get(p).copied().unwrap_or(0),
                        entry_slice.get(p + 1).copied().unwrap_or(0),
                        entry_slice.get(p + 2).copied().unwrap_or(0),
                        entry_slice.get(p + 3).copied().unwrap_or(0),
                    ]);
                    let range32 = u32::from_le_bytes([
                        entry_slice.get(p + 4).copied().unwrap_or(0),
                        entry_slice.get(p + 5).copied().unwrap_or(0),
                        entry_slice.get(p + 6).copied().unwrap_or(0),
                        entry_slice.get(p + 7).copied().unwrap_or(0),
                    ]);
                    p += 8;
                    #[allow(clippy::cast_sign_loss)]
                    let resolved = if (cie_fde_encoding & 0x70) == 0x10 {
                        pc_field_vaddr.wrapping_add(i64::from(rel_pc) as u64)
                    } else {
                        i64::from(rel_pc) as u64
                    };
                    (resolved, u64::from(range32))
                } else {
                    if p.saturating_add(16) > entry_slice.len() {
                        return Err(LibTpuError::DwarfCfiParseError {
                            offset: (cursor + p) as u64,
                            reason: "truncated 64-bit FDE PC/range header",
                        });
                    }
                    let start64 = u64::from_le_bytes([
                        entry_slice.get(p).copied().unwrap_or(0),
                        entry_slice.get(p + 1).copied().unwrap_or(0),
                        entry_slice.get(p + 2).copied().unwrap_or(0),
                        entry_slice.get(p + 3).copied().unwrap_or(0),
                        entry_slice.get(p + 4).copied().unwrap_or(0),
                        entry_slice.get(p + 5).copied().unwrap_or(0),
                        entry_slice.get(p + 6).copied().unwrap_or(0),
                        entry_slice.get(p + 7).copied().unwrap_or(0),
                    ]);
                    let len64 = u64::from_le_bytes([
                        entry_slice.get(p + 8).copied().unwrap_or(0),
                        entry_slice.get(p + 9).copied().unwrap_or(0),
                        entry_slice.get(p + 10).copied().unwrap_or(0),
                        entry_slice.get(p + 11).copied().unwrap_or(0),
                        entry_slice.get(p + 12).copied().unwrap_or(0),
                        entry_slice.get(p + 13).copied().unwrap_or(0),
                        entry_slice.get(p + 14).copied().unwrap_or(0),
                        entry_slice.get(p + 15).copied().unwrap_or(0),
                    ]);
                    p += 16;
                    (start64, len64)
                };

                let mut lsda_vaddr = None;
                if cie_has_lsda {
                    let aug_len =
                        usize::try_from(Self::read_uleb128(entry_slice, &mut p)?).unwrap_or(0);
                    let lsda_raw = u32::from_le_bytes([
                        entry_slice.get(p).copied().unwrap_or(0),
                        entry_slice.get(p + 1).copied().unwrap_or(0),
                        entry_slice.get(p + 2).copied().unwrap_or(0),
                        entry_slice.get(p + 3).copied().unwrap_or(0),
                    ]);
                    if lsda_raw != 0 {
                        lsda_vaddr = Some(u64::from(lsda_raw));
                    }
                    p = p.saturating_add(aug_len).min(entry_slice.len());
                }

                let fde_instrs = entry_slice.get(p..).unwrap_or_default();
                let (fde_cfa, mut fde_regs) =
                    Self::eval_cfa_instructions(fde_instrs, cie_data_align)?;
                for &r in &cie_init_regs {
                    fde_regs.insert(r);
                }
                records.push(DwarfCfiFunctionRecord {
                    start_vaddr,
                    end_vaddr: start_vaddr.wrapping_add(byte_len),
                    byte_len,
                    lsda_vaddr,
                    stack_frame_bytes: fde_cfa.max(cie_init_cfa),
                    saved_registers: fde_regs.into_iter().collect(),
                });
            }

            cursor = entry_end;
        }

        Ok(records)
    }
}

/// Disassembled `AArch64` (`arm64`) function forensics specification (`AArch64FunctionSpec`) for
/// `manylinux_2_31_aarch64` `libtpu.so` binaries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AArch64FunctionSpec {
    /// Function entrypoint virtual address.
    pub entrypoint_vaddr: u64,
    /// Total 32-bit instructions decoded.
    pub instruction_count: usize,
    /// Expected `struct_size` extracted from `ldr xN, [x0]; cmp xN, #imm`.
    pub expected_struct_size: Option<u32>,
    /// Direct branch (`B`, `B.cond`, `CBZ`, `CBNZ`, `TBZ`, `TBNZ`) target virtual addresses.
    pub branch_targets: Vec<u64>,
    /// Direct function call (`BL`) target virtual addresses.
    pub call_targets: Vec<u64>,
    /// Page-relative `ADRP` + `ADD` resolved `.rodata` string or data virtual addresses.
    pub adrp_targets: Vec<u64>,
    /// Decoded `ioctl` / `SVC` request numbers loaded into `w1` / `x1`.
    pub ioctl_requests: Vec<u64>,
}

/// Clean-room `AArch64` (`arm64`) fixed-width 32-bit instruction analyzer (`AArch64ForensicsAnalyzer`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AArch64ForensicsAnalyzer;

impl AArch64ForensicsAnalyzer {
    /// Sign-extends a `bits`-wide unsigned integer to 64 bits.
    ///
    /// # Arguments
    ///
    /// * `val` - Raw unsigned bitfield value.
    /// * `bits` - Bit width of the signed immediate (`14`, `19`, `21`, or `26`).
    #[allow(clippy::cast_possible_wrap)]
    const fn sign_extend(val: u64, bits: u32) -> i64 {
        let shift = 64_u32.saturating_sub(bits);
        ((val << shift) as i64) >> shift
    }

    /// Analyzes an `AArch64` (`arm64`) machine code slice starting at `entrypoint_vaddr`.
    ///
    /// Decodes:
    /// - `CMP Xn, #imm12` (`0xF100_001F` mask) for `PJRT_*_Args` `struct_size` checks
    /// - `B` (`0x1400_0000`) and `BL` (`0x9400_0000`) 26-bit PC-relative branches/calls
    /// - `B.cond` (`0x5400_0000`) and `CBZ`/`CBNZ` (`0x3400_0000`) 19-bit PC-relative branches
    /// - `TBZ`/`TBNZ` (`0x3600_0000`) 14-bit PC-relative bit-test branches
    /// - `ADRP Xd, #page` (`0x9000_0000`) paired with `ADD Xd, Xd, #lo12` (`0x9100_0000`)
    /// - `MOVZ W1, #imm16` / `MOVK W1, #imm16, LSL #16` followed by `BL` or `SVC #0` (`0xD400_0001`) for `ioctl` calls.
    ///
    /// # Arguments
    ///
    /// * `code` - Raw little-endian 32-bit `AArch64` instruction bytes.
    /// * `entrypoint_vaddr` - Starting virtual address (`PC`) of `code`.
    #[must_use]
    #[allow(clippy::indexing_slicing)]
    pub fn analyze_function(code: &[u8], entrypoint_vaddr: u64) -> AArch64FunctionSpec {
        let mut instruction_count = 0_usize;
        let mut expected_struct_size: Option<u32> = None;
        let mut branch_targets = Vec::new();
        let mut call_targets = Vec::new();
        let mut adrp_targets = Vec::new();
        let mut ioctl_requests = Vec::new();
        let mut pending_adrp_page: [u64; 32] = [0; 32];
        let mut pending_w1_ioctl: Option<u64> = None;

        for (idx, chunk) in code.chunks_exact(4).enumerate() {
            instruction_count += 1;
            let pc = entrypoint_vaddr.wrapping_add((idx as u64).wrapping_mul(4));
            let word = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);

            if (word & 0x7F80_001F) == 0x7100_001F {
                let imm12 = (word >> 10) & 0x0FFF;
                if expected_struct_size.is_none() && imm12 > 0 {
                    expected_struct_size = Some(imm12);
                }
            } else if (word & 0x7C00_0000) == 0x1400_0000 {
                let imm26 = u64::from(word & 0x03FF_FFFF);
                #[allow(clippy::cast_sign_loss)]
                let offset = (Self::sign_extend(imm26, 26) * 4) as u64;
                let target = pc.wrapping_add(offset);
                if (word & 0x8000_0000) != 0 {
                    call_targets.push(target);
                    if let Some(req) = pending_w1_ioctl.take() {
                        ioctl_requests.push(req);
                    }
                } else {
                    branch_targets.push(target);
                }
            } else if (word & 0xFE00_0000) == 0x5400_0000 || (word & 0x7E00_0000) == 0x3400_0000 {
                let imm19 = u64::from((word >> 5) & 0x0007_FFFF);
                #[allow(clippy::cast_sign_loss)]
                let offset = (Self::sign_extend(imm19, 19) * 4) as u64;
                branch_targets.push(pc.wrapping_add(offset));
            } else if (word & 0x7E00_0000) == 0x3600_0000 {
                let imm14 = u64::from((word >> 5) & 0x3FFF);
                #[allow(clippy::cast_sign_loss)]
                let offset = (Self::sign_extend(imm14, 14) * 4) as u64;
                branch_targets.push(pc.wrapping_add(offset));
            } else if (word & 0x9F00_0000) == 0x9000_0000 {
                let rd = (word & 0x1F) as usize;
                let immlo = u64::from((word >> 29) & 0x3);
                let immhi = u64::from((word >> 5) & 0x0007_FFFF);
                let imm21 = (immhi << 2) | immlo;
                #[allow(clippy::cast_sign_loss)]
                let page_offset = (Self::sign_extend(imm21, 21) << 12) as u64;
                pending_adrp_page[rd] = (pc & !0x0FFF_u64).wrapping_add(page_offset);
            } else if (word & 0xFFC0_0000) == 0x9100_0000 {
                let rd = (word & 0x1F) as usize;
                let rn = ((word >> 5) & 0x1F) as usize;
                let imm12 = u64::from((word >> 10) & 0x0FFF);
                let base_page = pending_adrp_page[rn];
                if base_page != 0 {
                    adrp_targets.push(base_page.wrapping_add(imm12));
                    pending_adrp_page[rd] = 0;
                }
            } else if (word & 0x7F80_001F) == 0x5280_0001 {
                let hw = ((word >> 21) & 0x3) * 16;
                let imm16 = u64::from((word >> 5) & 0xFFFF);
                pending_w1_ioctl = Some(imm16 << hw);
            } else if (word & 0x7F80_001F) == 0x7280_0001 {
                let hw = ((word >> 21) & 0x3) * 16;
                let imm16 = u64::from((word >> 5) & 0xFFFF);
                let prev = pending_w1_ioctl.unwrap_or(0) & !(0xFFFF_u64 << hw);
                pending_w1_ioctl = Some(prev | (imm16 << hw));
            } else if word == 0xD400_0001
                && let Some(req) = pending_w1_ioctl.take()
            {
                ioctl_requests.push(req);
            }
        }

        AArch64FunctionSpec {
            entrypoint_vaddr,
            instruction_count,
            expected_struct_size,
            branch_targets,
            call_targets,
            adrp_targets,
            ioctl_requests,
        }
    }
}

/// Recovered `.rodata` relative jump-table (`switch` dispatch table) descriptor (`JumpTableDescriptor`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JumpTableDescriptor {
    /// Virtual address of the indirect jump instruction (`jmp rax` or `br xN`).
    pub indirect_jump_ip: u64,
    /// Base virtual address of the `.rodata` signed-32-bit offset table.
    pub table_base_vaddr: u64,
    /// Number of case entries decoded from the jump table.
    pub entry_count: usize,
    /// Resolved target virtual addresses (`table_base_vaddr + i32_offset`) for each case index `0..entry_count`.
    pub case_target_vaddrs: Vec<u64>,
}

/// Directed edge in an inter-procedural Call Graph (`CallGraphEdge`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallGraphEdge {
    /// Caller function symbol name.
    pub caller_name: String,
    /// Caller entrypoint virtual address.
    pub caller_vaddr: u64,
    /// Callee function or virtual method symbol name (`"<indirect>"` if unresolved).
    pub callee_name: String,
    /// Callee target virtual address.
    pub callee_vaddr: u64,
    /// Whether this edge originates from an Itanium `_ZTV*` virtual method table slot.
    pub is_virtual_dispatch: bool,
}

/// Inter-procedural Call Graph (`CallGraphReport`) linking `.symtab`/`.dynsym` functions to direct `CALL`/`BL` and `_ZTV*` targets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallGraphReport {
    /// Total number of caller functions analyzed.
    pub function_count: usize,
    /// Directed call edges recovered across direct and virtual dispatch targets.
    pub edges: Vec<CallGraphEdge>,
}

/// Recovers `.rodata` relative 32-bit jump tables (`lea rReg, [rip + disp]; movsxd rIdx, dword ptr [rBase + rIdx*4]; add rIdx, rBase; jmp rIdx`)
/// from an `x86-64` instruction stream and a `.rodata` slice.
///
/// # Arguments
///
/// * `code` - Raw `x86-64` machine code bytes.
/// * `base_ip` - Starting virtual instruction pointer (`RIP`) for `code`.
/// * `rodata_bytes` - Raw `.rodata` section bytes containing signed 32-bit relative offsets.
/// * `rodata_vaddr` - Base virtual address of `rodata_bytes`.
/// * `max_entries` - Maximum number of 4-byte jump-table entries to decode per table.
#[must_use]
pub fn recover_x86_64_jump_table(
    code: &[u8],
    base_ip: u64,
    rodata_bytes: &[u8],
    rodata_vaddr: u64,
    max_entries: usize,
) -> Vec<JumpTableDescriptor> {
    let mut decoder = Decoder::with_ip(64, code, base_ip, DecoderOptions::NONE);
    let mut pending_table_base: Option<u64> = None;
    let mut saw_scaled_movsxd = false;
    let mut tables = Vec::new();
    let entry_cap = max_entries.max(1);

    while decoder.can_decode() {
        let instr = decoder.decode();
        if instr.mnemonic() == Mnemonic::Lea && instr.is_ip_rel_memory_operand() {
            pending_table_base = Some(instr.ip_rel_memory_address());
            saw_scaled_movsxd = false;
        } else if instr.mnemonic() == Mnemonic::Movsxd {
            saw_scaled_movsxd = instr.memory_index_scale() == 4;
        } else if instr.flow_control() == FlowControl::IndirectBranch && saw_scaled_movsxd {
            if let Some(tbl_vaddr) = pending_table_base.take() {
                let rel_off = usize::try_from(tbl_vaddr.saturating_sub(rodata_vaddr)).unwrap_or(0);
                let slice = rodata_bytes.get(rel_off..).unwrap_or_default();
                let mut targets = Vec::new();
                for chunk in slice.chunks_exact(4).take(entry_cap) {
                    let disp = i32::from_le_bytes([
                        chunk.first().copied().unwrap_or(0),
                        chunk.get(1).copied().unwrap_or(0),
                        chunk.get(2).copied().unwrap_or(0),
                        chunk.get(3).copied().unwrap_or(0),
                    ]);
                    #[allow(clippy::cast_sign_loss)]
                    let target = tbl_vaddr.wrapping_add(i64::from(disp) as u64);
                    targets.push(target);
                }
                tables.push(JumpTableDescriptor {
                    indirect_jump_ip: instr.ip(),
                    table_base_vaddr: tbl_vaddr,
                    entry_count: targets.len(),
                    case_target_vaddrs: targets,
                });
            }
            saw_scaled_movsxd = false;
        }
    }

    tables
}

/// Recovers `.rodata` relative 32-bit jump tables from an `AArch64` (`arm64`) instruction stream.
///
/// Matches `adrp/add xBase, #tbl; ldrsw xOff, [xBase, xIdx, lsl #2]; add xTarget, xBase, xOff; br xTarget`.
///
/// # Arguments
///
/// * `code` - Raw little-endian 32-bit `AArch64` instruction bytes.
/// * `entrypoint_vaddr` - Starting virtual address (`PC`) of `code`.
/// * `rodata_bytes` - Raw `.rodata` section bytes containing signed 32-bit relative offsets.
/// * `rodata_vaddr` - Base virtual address of `rodata_bytes`.
/// * `max_entries` - Maximum number of 4-byte jump-table entries to decode per table.
#[must_use]
#[allow(clippy::indexing_slicing)]
pub fn recover_aarch64_jump_table(
    code: &[u8],
    entrypoint_vaddr: u64,
    rodata_bytes: &[u8],
    rodata_vaddr: u64,
    max_entries: usize,
) -> Vec<JumpTableDescriptor> {
    let spec = AArch64ForensicsAnalyzer::analyze_function(code, entrypoint_vaddr);
    let mut saw_ldrsw = false;
    let mut tables = Vec::new();
    let entry_cap = max_entries.max(1);
    let mut adrp_iter = spec.adrp_targets.into_iter();

    for (idx, chunk) in code.chunks_exact(4).enumerate() {
        let pc = entrypoint_vaddr.wrapping_add((idx as u64).wrapping_mul(4));
        let word = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        // LDRSW Xt, [Xn, Xm, LSL #2] (`0xB8A0_7800` mask `0xFFE0_FC00`)
        if (word & 0xFFE0_FC00) == 0xB8A0_7800 {
            saw_ldrsw = true;
        } else if (word & 0xFFFF_FC1F) == 0xD61F_0000 && saw_ldrsw {
            let tbl_vaddr = adrp_iter.next().unwrap_or(rodata_vaddr);
            let rel_off = usize::try_from(tbl_vaddr.saturating_sub(rodata_vaddr)).unwrap_or(0);
            let slice = rodata_bytes.get(rel_off..).unwrap_or_default();
            let mut targets = Vec::new();
            for c in slice.chunks_exact(4).take(entry_cap) {
                let disp = i32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                #[allow(clippy::cast_sign_loss)]
                let target = tbl_vaddr.wrapping_add(i64::from(disp) as u64);
                targets.push(target);
            }
            tables.push(JumpTableDescriptor {
                indirect_jump_ip: pc,
                table_base_vaddr: tbl_vaddr,
                entry_count: targets.len(),
                case_target_vaddrs: targets,
            });
            saw_ldrsw = false;
        }
    }

    tables
}

/// Constructs an inter-procedural [`CallGraphReport`] linking caller functions (`(name, vaddr, code_bytes)`)
/// to their direct `CALL` targets in `symbol_table` and Itanium `_ZTV*` virtual method targets.
///
/// # Arguments
///
/// * `functions` - Slice of `(caller_name, caller_vaddr, machine_code_bytes)` tuples.
/// * `symbol_table` - Slice of [`super::ExportedBinarySymbol`] records for resolving target addresses to symbol names.
/// * `vtable_targets` - Slice of `(vtable_owner_class, method_vaddr)` pairs recovered from `.rela.dyn` / `_ZTV*`.
#[must_use]
pub fn build_interprocedural_call_graph(
    functions: &[(&str, u64, &[u8])],
    symbol_table: &[super::ExportedBinarySymbol],
    vtable_targets: &[(&str, u64)],
) -> CallGraphReport {
    let mut addr_to_name = std::collections::BTreeMap::new();
    for sym in symbol_table {
        addr_to_name.insert(sym.address, sym.name.clone());
    }
    let mut edges = Vec::new();

    for &(caller_name, caller_vaddr, code) in functions {
        let cfg = build_x86_64_cfg(code, caller_vaddr);
        for target_vaddr in cfg.call_targets {
            let callee_name = addr_to_name
                .get(&target_vaddr)
                .cloned()
                .unwrap_or_else(|| String::from("<indirect>"));
            edges.push(CallGraphEdge {
                caller_name: String::from(caller_name),
                caller_vaddr,
                callee_name,
                callee_vaddr: target_vaddr,
                is_virtual_dispatch: false,
            });
        }
    }

    for &(vtable_class, method_vaddr) in vtable_targets {
        let callee_name = addr_to_name
            .get(&method_vaddr)
            .cloned()
            .unwrap_or_else(|| String::from("<virtual_method>"));
        edges.push(CallGraphEdge {
            caller_name: String::from(vtable_class),
            caller_vaddr: 0,
            callee_name,
            callee_vaddr: method_vaddr,
            is_virtual_dispatch: true,
        });
    }

    CallGraphReport {
        function_count: functions.len(),
        edges,
    }
}

/// Result of comparing an x86-64 or `AArch64` prologue's extracted `struct_size` against [`super::GoldenStructLayouts`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrologueStructSizeVerification {
    /// `PJRT_Api` 64-bit word index (`5..=142`) or extension slot ordinal.
    pub slot_index: usize,
    /// Mangled C++ or C symbol name for the entrypoint.
    pub symbol_name: String,
    /// Extracted minimum `struct_size` from the prologue (`CMP QWORD PTR [RDI], imm` or `CMP Xn, #imm12`).
    pub extracted_struct_size: Option<usize>,
    /// Expected minimum `struct_size` recorded in `golden_struct_layouts.json`.
    pub golden_min_struct_size: usize,
    /// Whether the extracted prologue constant matches the golden specification.
    pub matches_golden: bool,
}

/// Scans a batch of x86-64 or `AArch64` PJRT entrypoint prologues (`(slot_index, code_bytes, is_aarch64)`)
/// and verifies each extracted `struct_size` constant against [`super::GoldenStructLayouts`].
///
/// # Arguments
///
/// * `prologues` - Slice of `(word_index, machine_code_bytes, is_aarch64)` tuples.
/// * `golden_layouts` - Parsed [`super::GoldenStructLayouts`] reference.
#[must_use]
pub fn verify_prologue_struct_sizes_against_golden(
    prologues: &[(usize, &[u8], bool)],
    golden_layouts: &super::GoldenStructLayouts,
) -> Vec<PrologueStructSizeVerification> {
    let mut results = Vec::with_capacity(prologues.len());
    for &(word_idx, code, is_aarch64) in prologues {
        let extracted = if is_aarch64 {
            AArch64ForensicsAnalyzer::analyze_function(code, 0)
                .expected_struct_size
                .map(|v| v as usize)
        } else {
            disassemble_x86_64_prologue(code)
                .expected_struct_size
                .map(|v| v as usize)
        };
        if let Some(slot) = golden_layouts
            .pjrt_slots
            .iter()
            .find(|s| s.word_index == word_idx)
        {
            let matches_golden = extracted == Some(slot.min_struct_size);
            results.push(PrologueStructSizeVerification {
                slot_index: word_idx,
                symbol_name: slot.mangled_symbol.clone(),
                extracted_struct_size: extracted,
                golden_min_struct_size: slot.min_struct_size,
                matches_golden,
            });
        }
    }
    results
}

#[cfg(test)]
#[allow(
    clippy::indexing_slicing,
    clippy::too_many_lines,
    clippy::cast_possible_truncation
)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_iced_x86_prologue_cfg_and_ioctl_scanner() {
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
        assert_eq!(rep1.field_accesses[0].clone(), rep1.field_accesses[0]);
        assert_ne!(format!("{:?}", rep1.field_accesses[0]), "");

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

        let empty_cfg = build_x86_64_cfg(&[], 0x1000);
        assert_eq!(empty_cfg.instruction_count, 0);
        assert_eq!(empty_cfg.basic_block_count, 0);
        let ret_only_cfg = build_x86_64_cfg(&[0xC3], 0x1000);
        assert_eq!(ret_only_cfg.instruction_count, 1);
        assert_eq!(ret_only_cfg.basic_block_count, 1);

        let cfg_code: &[u8] = &[
            0x48, 0x8B, 0x05, 0x10, 0x00, 0x00, 0x00, // mov rax, [rip + 0x10]
            0xE8, 0x20, 0x00, 0x00, 0x00, // call +0x20
            0xFF, 0xD0, // call rax
            0xFF, 0x20, // jmp [rax]
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

        let ioctl_code: &[u8] = &[
            0xBE, 0x08, 0xDC, 0x20, 0x40, // mov esi, 0x4020dc08
            0xE8, 0x00, 0x00, 0x00, 0x00, // call +0
            0x48, 0xBE, 0x64, 0x3B, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // movabs rsi, 0x3b64
            0x0F, 0x05, // syscall
            0xB8, 0x01, 0x00, 0x00, 0x00, // mov eax, 1
            0x48, 0x89, 0x07, // mov [rdi], rax
            0x89, 0xC6, // mov esi, eax
            0x48, 0x89, 0xC6, // mov rsi, rax
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

        // Test RIP-relative string scanner (`lea rdi, [rip + 0x19]` at 0x1000 -> next_ip 0x1007 + 0x19 = 0x1020)
        let rip_code: &[u8] = &[
            0x48, 0x8D, 0x3D, 0x19, 0x00, 0x00, 0x00, // lea rdi, [rip + 0x19] -> 0x1020
            0x48, 0x8D, 0x3D, 0x16, 0x00, 0x00,
            0x00, // lea rdi, [rip + 0x16] -> 0x1024 (empty nul)
            0x48, 0x8D, 0x3D, 0xF0, 0xFF, 0xFF,
            0xFF, // lea rdi, [rip - 0x10] -> before rodata
            0x48, 0x8D, 0x3D, 0xFF, 0x00, 0x00, 0x00, // lea rdi, [rip + 0xFF] -> past rodata
            0x90, // nop
        ];
        let rodata = b"ERR\0\0";
        let strs = scan_x86_64_rip_relative_strings(rip_code, 0x1000, rodata, 0x1020);
        assert_eq!(strs.len(), 1);
        assert_eq!(strs[0].instruction_ip, 0x1000);
        assert_eq!(strs[0].target_vaddr, 0x1020);
        assert_eq!(strs[0].literal, "ERR");
        assert_eq!(strs[0].clone(), strs[0]);
        assert_ne!(format!("{:?}", strs[0]), "");

        let abi3_rep = introspect_cpython_abi3_binary(
            b"PyInit_sdk\0slice\0tpumonitoring\0tpuz\0debug\0ChipCoordinates\0_C_API\0",
        );
        assert_eq!(abi3_rep.init_symbol, "PyInit_sdk");
        assert_eq!(abi3_rep.submodules.len(), 4);
        assert_eq!(abi3_rep.exported_types, vec!["ChipCoordinates"]);
        assert_eq!(abi3_rep.capsule_names, vec!["libtpu.sdk._C_API"]);
        assert_eq!(abi3_rep.clone(), abi3_rep);
        assert_ne!(format!("{abi3_rep:?}"), "");

        let empty_abi3 = introspect_cpython_abi3_binary(b"none");
        assert_eq!(empty_abi3.submodules, Vec::<String>::new());
        assert_eq!(empty_abi3.exported_types, Vec::<String>::new());
        assert_eq!(empty_abi3.capsule_names, Vec::<String>::new());

        let bitfield_code: &[u8] = &[
            0x48, 0xC1, 0xE0, 0x04, // shl rax, 4 (HEADER)
            0x48, 0xC1, 0xE0, 0x0C, // shl rax, 12 (S0)
            0x48, 0xC1, 0xE8, 0x14, // shr rax, 20 (V0)
            0x48, 0xC1, 0xF8, 0x1C, // sar rax, 28 (M0)
            0x48, 0xC1, 0xE0, 0x24, // shl rax, 36 (XLU)
            0x48, 0xC1, 0xE0, 0x34, // shl rax, 52 (D0)
            0x24, 0x0F, // and al, 0x0F (Immediate8 -> S0)
            0x83, 0xE0, 0x0F, // and eax, 0x0F (Immediate8to32 -> S0)
            0x25, 0xFF, 0xFF, 0x00, 0x00, // and eax, 0xFFFF (Immediate32 -> V0)
            0x25, 0xFF, 0xFF, 0xFF, 0x00, // and eax, 0xFFFFFF (Immediate32 -> M0)
            0x48, 0x81, 0xE0, 0xFF, 0xFF, 0xFF,
            0x00, // and rax, 0xFFFFFF (Immediate32to64 -> M0)
            0x48, 0x83, 0xC8, 0x03, // or rax, 3 (Immediate8to64 -> D0)
            0x48, 0x21, 0xD8, // and rax, rbx (non-immediate)
            0x90, // nop (neither shift nor mask)
        ];
        let bf_patterns = scan_x86_64_bitfield_encoders(bitfield_code, 0x2000);
        assert!(bf_patterns.len() >= 10);
        assert_eq!(bf_patterns[0].slot_kind, "HEADER");
        assert_eq!(bf_patterns[1].slot_kind, "S0");
        assert_eq!(bf_patterns[2].slot_kind, "V0");
        assert_eq!(bf_patterns[3].slot_kind, "M0");
        assert_eq!(bf_patterns[4].slot_kind, "XLU");
        assert_eq!(bf_patterns[5].slot_kind, "D0");
        assert_eq!(bf_patterns[0].clone(), bf_patterns[0]);
        assert_ne!(format!("{:?}", bf_patterns[0]), "");

        let mmio_code: &[u8] = &[
            0xC6, 0x47, 0x10, 0x01, // mov byte ptr [rdi + 0x10], 1 (Immediate8)
            0x66, 0xC7, 0x47, 0x14, 0x02, 0x00, // mov word ptr [rdi + 0x14], 2 (Immediate16)
            0xC7, 0x47, 0x18, 0x03, 0x00, 0x00,
            0x00, // mov dword ptr [rdi + 0x18], 3 (Immediate32)
            0x48, 0xC7, 0x47, 0x20, 0x04, 0x00, 0x00,
            0x00, // mov qword ptr [rdi + 0x20], 4 (Immediate32to64)
            0x48, 0x89, 0x47, 0x28, // mov qword ptr [rdi + 0x28], rax (register)
            0x48, 0x8B, 0x47, 0x28, // mov rax, qword ptr [rdi + 0x28] (op0 != Memory)
            0x90, // nop
        ];
        let mmio_seqs = scan_x86_64_driver_mmio_sequences(mmio_code, 0x3000);
        assert_eq!(mmio_seqs.len(), 5);
        assert_eq!(mmio_seqs[0].mmio_offset, 0x10);
        assert_eq!(mmio_seqs[0].immediate_value, Some(1));
        assert_eq!(mmio_seqs[1].immediate_value, Some(2));
        assert_eq!(mmio_seqs[2].immediate_value, Some(3));
        assert_eq!(mmio_seqs[3].immediate_value, Some(4));
        assert_eq!(mmio_seqs[4].immediate_value, None);
        assert_eq!(mmio_seqs[0].clone(), mmio_seqs[0]);
        assert_ne!(format!("{:?}", mmio_seqs[0]), "");
    }

    #[test]
    fn test_eh_frame_cfi_parser_and_aarch64_forensics() {
        let parser = EhFrameCfiParser;
        assert_eq!(parser.clone(), EhFrameCfiParser);
        assert_ne!(format!("{parser:?}"), "");

        // 1. Build a valid .eh_frame stream with CIE ("zRL", DW_EH_PE_pcrel | DW_EH_PE_sdata4 = 0x1B) and FDE
        let mut eh = Vec::new();
        // CIE body: cie_id=0 (4B), version=1 (1B), aug="zRL\0" (4B), code_align=1 (1B), data_align=-8 (0x78, 1B),
        // ret_reg=16 (1B), aug_len=2 (1B),lsda_enc=0x03 (1B), fde_enc=0x1B (1B),
        // DW_CFA_def_cfa(7, 8) = [0x0C, 0x07, 0x08], DW_CFA_offset(16, 1) = [0x90, 0x01], DW_CFA_nop = [0x00]
        let cie_body: &[u8] = &[
            0, 0, 0, 0, // CIE_id = 0
            1, // version = 1
            b'z', b'R', b'L', 0,    // augmentation "zRL"
            1,    // code_align = 1
            0x78, // data_align = -8 (SLEB128)
            16,   // return_address_register = 16
            2, 0x1B, 0x03, // aug_len = 2, R=0x1B (pcrel|sdata4), L=0x03
            0x0C, 0x07, 0x08, // DW_CFA_def_cfa(rsp, 8)
            0x90, 0x01, // DW_CFA_offset(rip=16, 1)
            0x00, // DW_CFA_nop
        ];
        eh.extend_from_slice(&(cie_body.len() as u32).to_le_bytes());
        eh.extend_from_slice(cie_body);

        // FDE 1: pcrel|sdata4 with LSDA and rich DW_CFA instructions
        let mut fde_body = Vec::new();
        fde_body.extend_from_slice(&24_u32.to_le_bytes()); // cie_pointer != 0
        fde_body.extend_from_slice(&0x1000_i32.to_le_bytes()); // rel_pc
        fde_body.extend_from_slice(&64_u32.to_le_bytes()); // range = 64 bytes
        fde_body.push(4); // aug_len = 4
        fde_body.extend_from_slice(&0x0900_0000_u32.to_le_bytes()); // non-zero LSDA
        fde_body.extend_from_slice(&[
            0x41, // DW_CFA_advance_loc(1)
            0x0E, 0x20, // DW_CFA_def_cfa_offset(32)
            0x86, 0x02, // DW_CFA_offset(rbp=6, 2)
            0x02, 0x04, // DW_CFA_advance_loc1(4)
            0x03, 0x08, 0x00, // DW_CFA_advance_loc2(8)
            0x04, 0x10, 0x00, 0x00, 0x00, // DW_CFA_advance_loc4(16)
            0x0D, 0x06, // DW_CFA_def_cfa_register(6)
            0x05, 0x03, 0x04, // DW_CFA_offset_extended(rbx=3, 4)
            0x05, 0x80, 0x02, 0x01, // DW_CFA_offset_extended(reg=256 > u8::MAX, 1)
            0xC3, // DW_CFA_restore(rbx=3)
            0xFF, // unknown opcode -> terminates CFA loop cleanly
        ]);
        eh.extend_from_slice(&(fde_body.len() as u32).to_le_bytes());
        eh.extend_from_slice(&fde_body);
        eh.extend_from_slice(&0_u32.to_le_bytes()); // zero-length terminator

        let parsed = EhFrameCfiParser::parse_eh_frame(&eh, 0x0800_0000);
        assert!(parsed.is_ok());
        if let Ok(records) = parsed {
            assert_eq!(records.len(), 1);
            assert_eq!(records[0].byte_len, 64);
            assert_eq!(records[0].lsda_vaddr, Some(0x0900_0000));
            assert_eq!(records[0].stack_frame_bytes, 32);
            assert!(records[0].saved_registers.contains(&6));
            assert!(!records[0].saved_registers.contains(&3));
            assert_eq!(records[0].clone(), records[0]);
            assert_ne!(format!("{:?}", records[0]), "");
        }

        // 2. Test 64-bit absolute FDE encoding and positive multi-byte SLEB128 / ULEB128
        let mut eh64 = Vec::new();
        let cie64: &[u8] = &[
            0, 0, 0, 0, // CIE_id = 0
            3, // version = 3
            0, // empty augmentation ""
            0x81, 0x01, // multi-byte ULEB128 code_align = 129
            0x88, 0x00, // multi-byte positive SLEB128 data_align = +8
            16,   // ret_reg
        ];
        eh64.extend_from_slice(&(cie64.len() as u32).to_le_bytes());
        eh64.extend_from_slice(cie64);
        let mut fde64 = Vec::new();
        fde64.extend_from_slice(&16_u32.to_le_bytes());
        fde64.extend_from_slice(&0x0B4F_9B90_u64.to_le_bytes());
        fde64.extend_from_slice(&48_u64.to_le_bytes());
        eh64.extend_from_slice(&(fde64.len() as u32).to_le_bytes());
        eh64.extend_from_slice(&fde64);
        let parsed64 = EhFrameCfiParser::parse_eh_frame(&eh64, 0x1000);
        assert!(parsed64.is_ok());
        if let Ok(recs) = parsed64 {
            assert_eq!(recs.len(), 1);
            assert_eq!(recs[0].start_vaddr, 0x0B4F_9B90);
            assert_eq!(recs[0].end_vaddr, 0x0B4F_9B90 + 48);
        }

        // 3. Test non-pcrel sdata4 FDE encoding (0x0B) and zero LSDA
        let mut eh_sdata4 = Vec::new();
        let cie_sdata4: &[u8] = &[
            0, 0, 0, 0, 4, b'z', b'R', b'L', b'P', 0, 1, 0x78, 16, 2, 0x0B, 0x03,
        ];
        eh_sdata4.extend_from_slice(&(cie_sdata4.len() as u32).to_le_bytes());
        eh_sdata4.extend_from_slice(cie_sdata4);
        let mut fde_sdata4 = Vec::new();
        fde_sdata4.extend_from_slice(&20_u32.to_le_bytes());
        fde_sdata4.extend_from_slice(&0x2000_i32.to_le_bytes());
        fde_sdata4.extend_from_slice(&16_u32.to_le_bytes());
        fde_sdata4.push(4);
        fde_sdata4.extend_from_slice(&0_u32.to_le_bytes()); // zero LSDA
        eh_sdata4.extend_from_slice(&(fde_sdata4.len() as u32).to_le_bytes());
        eh_sdata4.extend_from_slice(&fde_sdata4);
        let parsed_sdata4 = EhFrameCfiParser::parse_eh_frame(&eh_sdata4, 0x1000);
        assert!(parsed_sdata4.is_ok());
        if let Ok(recs) = parsed_sdata4 {
            assert_eq!(recs[0].start_vaddr, 0x2000);
            assert_eq!(recs[0].lsda_vaddr, None);
        }

        // 4. Test error branches in EhFrameCfiParser
        assert!(EhFrameCfiParser::parse_eh_frame(&[1, 2], 0).is_err());
        assert!(EhFrameCfiParser::parse_eh_frame(&[10, 0, 0, 0, 1, 2], 0).is_err());
        assert!(EhFrameCfiParser::parse_eh_frame(&[2, 0, 0, 0, 0, 0], 0).is_err());
        assert!(EhFrameCfiParser::parse_eh_frame(&[4, 0, 0, 0, 0, 0, 0, 0], 0).is_err());
        assert!(EhFrameCfiParser::parse_eh_frame(&[5, 0, 0, 0, 0, 0, 0, 0, 99], 0).is_err());
        // Truncated ULEB128
        assert!(EhFrameCfiParser::parse_eh_frame(&[6, 0, 0, 0, 0, 0, 0, 0, 1, 0], 0).is_err());
        // Overflow ULEB128 (10 bytes with 0x80 set)
        let mut uleb_ovf = vec![16, 0, 0, 0, 0, 0, 0, 0, 1, 0];
        uleb_ovf.extend_from_slice(&[0x80; 10]);
        assert!(EhFrameCfiParser::parse_eh_frame(&uleb_ovf, 0).is_err());
        // Truncated SLEB128
        assert!(EhFrameCfiParser::parse_eh_frame(&[7, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], 0).is_err());
        // Overflow SLEB128
        let mut sleb_ovf = vec![17, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1];
        sleb_ovf.extend_from_slice(&[0x80; 10]);
        assert!(EhFrameCfiParser::parse_eh_frame(&sleb_ovf, 0).is_err());
        // Truncated 64-bit FDE
        assert!(
            EhFrameCfiParser::parse_eh_frame(&[8, 0, 0, 0, 4, 0, 0, 0, 1, 2, 3, 4], 0).is_err()
        );
        // Truncated 32-bit FDE
        let mut trunc_fde32 = Vec::new();
        trunc_fde32.extend_from_slice(&(cie_sdata4.len() as u32).to_le_bytes());
        trunc_fde32.extend_from_slice(cie_sdata4);
        trunc_fde32.extend_from_slice(&6_u32.to_le_bytes());
        trunc_fde32.extend_from_slice(&[4, 0, 0, 0, 1, 2]);
        assert!(EhFrameCfiParser::parse_eh_frame(&trunc_fde32, 0).is_err());
        // Truncated DW_CFA_advance_loc1 / loc2 / loc4 / DW_CFA_offset
        assert!(EhFrameCfiParser::eval_cfa_instructions(&[0x02], -8).is_err());
        assert!(EhFrameCfiParser::eval_cfa_instructions(&[0x03, 0x01], -8).is_err());
        assert!(EhFrameCfiParser::eval_cfa_instructions(&[0x04, 0x01, 0x02], -8).is_err());
        assert!(EhFrameCfiParser::eval_cfa_instructions(&[0x80], -8).is_err());
        assert!(EhFrameCfiParser::eval_cfa_instructions(&[0x0C], -8).is_err());
        assert!(EhFrameCfiParser::eval_cfa_instructions(&[0x0C, 0x07], -8).is_err());
        assert!(EhFrameCfiParser::eval_cfa_instructions(&[0x0D], -8).is_err());
        assert!(EhFrameCfiParser::eval_cfa_instructions(&[0x0E], -8).is_err());
        assert!(EhFrameCfiParser::eval_cfa_instructions(&[0x05], -8).is_err());
        assert!(EhFrameCfiParser::eval_cfa_instructions(&[0x05, 0x03], -8).is_err());

        // 5. Test AArch64ForensicsAnalyzer
        let aarch64_analyzer = AArch64ForensicsAnalyzer;
        assert_eq!(aarch64_analyzer.clone(), AArch64ForensicsAnalyzer);
        assert_ne!(format!("{aarch64_analyzer:?}"), "");

        let mut arm_code = Vec::new();
        // CMP X0, #0 (0xF100_001F -> imm12=0, ignored)
        arm_code.extend_from_slice(&0xF100_001F_u32.to_le_bytes());
        // CMP X0, #88 (0xF101_601F)
        arm_code.extend_from_slice(&0xF101_601F_u32.to_le_bytes());
        // CMP X0, #96 (0xF101_801F -> ignored because expected_struct_size already set)
        arm_code.extend_from_slice(&0xF101_801F_u32.to_le_bytes());
        // B.EQ +8 (0x5400_0040)
        arm_code.extend_from_slice(&0x5400_0040_u32.to_le_bytes());
        // CBZ X0, +12 (0xB400_0060)
        arm_code.extend_from_slice(&0xB400_0060_u32.to_le_bytes());
        // TBZ W0, #0, +8 (0x3600_0040)
        arm_code.extend_from_slice(&0x3600_0040_u32.to_le_bytes());
        // ADRP X2, +4096 (immlo=1, immhi=0, rd=2 -> 0xB000_0002)
        arm_code.extend_from_slice(&0xB000_0002_u32.to_le_bytes());
        // ADD X2, X2, #0x40 (rd=2, rn=2, imm12=0x40 -> 0x9101_0042)
        arm_code.extend_from_slice(&0x9101_0042_u32.to_le_bytes());
        // ADD X3, X4, #0x10 (rn=4 has no ADRP page -> 0x9100_4083)
        arm_code.extend_from_slice(&0x9100_4083_u32.to_le_bytes());
        // BL +0x80 without pending ioctl (0x9400_0020)
        arm_code.extend_from_slice(&0x9400_0020_u32.to_le_bytes());
        // SVC #0 without pending ioctl (0xD400_0001)
        arm_code.extend_from_slice(&0xD400_0001_u32.to_le_bytes());
        // MOVZ W1, #0xDC08 (0x529B_8101)
        arm_code.extend_from_slice(&0x529B_8101_u32.to_le_bytes());
        // MOVK W1, #0x4020, LSL #16 (0x72A8_0401)
        arm_code.extend_from_slice(&0x72A8_0401_u32.to_le_bytes());
        // BL +0x100 (0x9400_0040)
        arm_code.extend_from_slice(&0x9400_0040_u32.to_le_bytes());
        // MOVZ W1, #0x3B64 (0x5287_6C81)
        arm_code.extend_from_slice(&0x5287_6C81_u32.to_le_bytes());
        // SVC #0 (0xD400_0001)
        arm_code.extend_from_slice(&0xD400_0001_u32.to_le_bytes());
        // B +0x20 (0x1400_0008)
        arm_code.extend_from_slice(&0x1400_0008_u32.to_le_bytes());
        // RET (0xD65F_03C0)
        arm_code.extend_from_slice(&0xD65F_03C0_u32.to_le_bytes());

        let arm_spec = AArch64ForensicsAnalyzer::analyze_function(&arm_code, 0x0040_0000);
        assert_eq!(arm_spec.entrypoint_vaddr, 0x0040_0000);
        assert_eq!(arm_spec.instruction_count, 18);
        assert_eq!(arm_spec.expected_struct_size, Some(88));
        assert_eq!(arm_spec.branch_targets.len(), 4);
        assert_eq!(arm_spec.call_targets.len(), 2);
        assert_eq!(arm_spec.adrp_targets, vec![0x0040_1040]);
        assert_eq!(arm_spec.ioctl_requests, vec![0x4020_DC08, 0x3B64]);
        assert_eq!(arm_spec.clone(), arm_spec);
        assert_ne!(format!("{arm_spec:?}"), "");

        // 6. Test x86-64 and AArch64 jump-table recovery + inter-procedural Call Graph
        let mut rodata = Vec::new();
        rodata.extend_from_slice(&0x10_i32.to_le_bytes());
        rodata.extend_from_slice(&0x20_i32.to_le_bytes());
        rodata.extend_from_slice(&(-0x08_i32).to_le_bytes());
        let x86_jt_code: &[u8] = &[
            0x48, 0x8D, 0x15, 0xF9, 0x0F, 0x00, 0x00, // lea rdx, [rip + 0x0FF9] -> 0x2000
            0x48, 0x8D, 0x04, 0x0A, // lea rax, [rdx + rcx] (non-RIP-relative lea)
            0x48, 0x63, 0x02, // movsxd rax, dword ptr [rdx] (scale=1)
            0xFF, 0xE0, // jmp rax with saw_scaled_movsxd == false
            0x48, 0x63, 0x04, 0x8A, // movsxd rax, dword ptr [rdx + rcx*4] (scale=4)
            0x48, 0x01, 0xD0, // add rax, rdx
            0xFF, 0xE0, // jmp rax with saw_scaled_movsxd == true
            0x48, 0x63, 0x04, 0x8A, // movsxd without preceding lea
            0xFF, 0xE0, // jmp rax without pending table
        ];
        let x86_jts = recover_x86_64_jump_table(x86_jt_code, 0x1000, &rodata, 0x2000, 3);
        assert_eq!(x86_jts.len(), 1);
        assert_eq!(x86_jts[0].table_base_vaddr, 0x2000);
        assert_eq!(x86_jts[0].entry_count, 3);
        assert_eq!(x86_jts[0].case_target_vaddrs, vec![0x2010, 0x2020, 0x1FF8]);
        assert_eq!(x86_jts[0].clone(), x86_jts[0]);
        assert_ne!(format!("{:?}", x86_jts[0]), "");

        let mut arm_jt_code = Vec::new();
        // ADRP X2, +4096 (0xB000_0002 -> 0x0040_1000)
        arm_jt_code.extend_from_slice(&0xB000_0002_u32.to_le_bytes());
        // ADD X2, X2, #0 (0x9100_0042 -> 0x0040_1000)
        arm_jt_code.extend_from_slice(&0x9100_0042_u32.to_le_bytes());
        // BR X3 before LDRSW (0xD61F_0060)
        arm_jt_code.extend_from_slice(&0xD61F_0060_u32.to_le_bytes());
        // LDRSW X3, [X2, X0, LSL #2] (0xB8A0_7843)
        arm_jt_code.extend_from_slice(&0xB8A0_7843_u32.to_le_bytes());
        // ADD X2, X2, X3 (0x8B03_0042)
        arm_jt_code.extend_from_slice(&0x8B03_0042_u32.to_le_bytes());
        // BR X2 (0xD61F_0040)
        arm_jt_code.extend_from_slice(&0xD61F_0040_u32.to_le_bytes());
        let arm_jts =
            recover_aarch64_jump_table(&arm_jt_code, 0x0040_0000, &rodata, 0x0040_1000, 3);
        assert_eq!(arm_jts.len(), 1);
        assert_eq!(arm_jts[0].table_base_vaddr, 0x0040_1000);
        assert_eq!(
            arm_jts[0].case_target_vaddrs,
            vec![0x0040_1010, 0x0040_1020, 0x0040_0FF8]
        );

        let caller_code: &[u8] = &[
            0xE8, 0xFB, 0x0F, 0x00, 0x00, // call 0x2000
            0xE8, 0xF6, 0x1F, 0x00, 0x00, // call 0x3000 (unresolved)
            0xC3,
        ];
        let sym_tbl = vec![super::super::ExportedBinarySymbol {
            name: String::from("pjrt::CreatePjrtApi"),
            address: 0x2000,
            size: 32,
        }];
        let cg = build_interprocedural_call_graph(
            &[("GetPjrtApi", 0x1000, caller_code)],
            &sym_tbl,
            &[
                ("asic_sw::driver::GasketDriver", 0x2000),
                ("asic_sw::driver::VfioDriver", 0x4000),
            ],
        );
        assert_eq!(cg.function_count, 1);
        assert_eq!(cg.edges.len(), 4);
        assert_eq!(cg.edges[0].callee_name, "pjrt::CreatePjrtApi");
        assert!(!cg.edges[0].is_virtual_dispatch);
        assert_eq!(cg.edges[1].callee_name, "<indirect>");
        assert_eq!(cg.edges[2].callee_name, "pjrt::CreatePjrtApi");
        assert!(cg.edges[2].is_virtual_dispatch);
        assert_eq!(cg.edges[3].callee_name, "<virtual_method>");
        assert_eq!(cg.clone(), cg);
        assert_ne!(format!("{cg:?}"), "");
        assert_eq!(cg.edges[0].clone(), cg.edges[0]);
        assert_ne!(format!("{:?}", cg.edges[0]), "");

        let layouts_json =
            include_str!("../../../../docs/reverse_engineering/golden_struct_layouts.json");
        if let Ok(golden) = super::super::parse_golden_struct_layouts(layouts_json)
            && let Some(first_slot) = golden.pjrt_slots.first()
        {
            let sz_u8 = (first_slot.min_struct_size & 0x7F) as u8;
            let x86_pro: [u8; 5] = [0x48, 0x83, 0x3F, sz_u8, 0xC3];
            let arm_word: u32 = 0xF100_001F | ((first_slot.min_struct_size as u32 & 0xFFF) << 10);
            let arm_pro = arm_word.to_le_bytes();
            let verifications = verify_prologue_struct_sizes_against_golden(
                &[
                    (first_slot.word_index, &x86_pro, false),
                    (first_slot.word_index, &arm_pro, true),
                    (9999, &x86_pro, false),
                ],
                &golden,
            );
            assert_eq!(verifications.len(), 2);
            assert!(verifications[0].matches_golden);
            assert!(verifications[1].matches_golden);
            assert_eq!(verifications[0].clone(), verifications[0]);
            assert_ne!(format!("{:?}", verifications[0]), "");
        }
    }
}

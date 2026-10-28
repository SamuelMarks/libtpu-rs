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

#[cfg(test)]
#[allow(clippy::indexing_slicing, clippy::too_many_lines)]
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
    }
}

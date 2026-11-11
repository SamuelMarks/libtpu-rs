//! Clean-room ELF64 and Mach-O binary forensics, relocation resolver, and `iced-x86` disassembler.
//!
//! Organized into focused semantic submodules:
//! - [`symbols`]: [`inspect_binary_symbols`], [`inspect_object_crate_sections`], [`parse_eh_frame_fde_ranges`],
//!   [`demangle_and_classify_cpp_symbol`], [`reconstruct_rtti_hierarchy`], [`BinarySymbolReport`],
//!   [`ObjectCrateBinaryReport`], [`EhFrameFunctionRange`], [`DemangledCppSymbol`], and [`RttiClassHierarchyEntry`].
//! - [`relocations`]: [`resolve_relative_relocations`], [`reconstruct_pjrt_vtable`],
//!   [`reconstruct_libtpu_sdk_vtable`], [`reconstruct_tpu_driver_vtable`], [`reconstruct_itanium_cpp_vtable`],
//!   [`ElfRelativeRelocation`], [`ReconstructedPjrtVtable`], [`ReconstructedFlatApiVtable`], and [`ReconstructedCppVtable`].
//! - [`disassembler`]: [`disassemble_x86_64_prologue`], [`build_x86_64_cfg`], [`scan_x86_64_ioctl_callsites`],
//!   [`disassemble_function_spec`], [`PrologueLayoutReport`], [`ControlFlowGraphReport`], [`DecodedIoctlCallsite`],
//!   and [`DisassembledFunctionSpec`].
//! - [`carver`]: [`carve_protodesc_cold_entries`], [`decode_carved_file_descriptor`],
//!   [`build_descriptor_pool_from_carved_protos`], [`render_proto_schema`], [`carve_protodesc_cold_descriptors`],
//!   [`verify_descriptor_topological_closure`], [`extract_filewrapper_toc`], [`extract_indirect_filewrapper_toc`],
//!   [`decompress_brotli_csr_table`], [`CarvedProtoDesc`], [`CarvedFileDescriptorSummary`], and [`FileTocEntry`].
//! - [`manifest`]: [`compute_artifact_digests`], [`parse_golden_binary_manifest`], [`parse_golden_struct_layouts`],
//!   [`ArtifactDigestReport`], [`GoldenBinaryManifest`], and [`GoldenStructLayouts`].

/// Embedded `protodesc_cold` Protobuf carver and `filewrapper_toc` extractor.
pub mod carver;
/// `iced-x86` x86-64 disassembler, CFG analyzer, and `ioctl` callsite scanner.
pub mod disassembler;
/// Artifact digests (`SHA-256`/`BLAKE3`) and golden manifest parsers.
pub mod manifest;
/// ELF64 `.rela.dyn` relocation resolver and C/C++ vtable reconstructors.
pub mod relocations;
/// ELF64/Mach-O symbol inspector, `.eh_frame` parser, and C++ demangler.
pub mod symbols;

pub use self::carver::{
    BrotliRegisterTreeEntry, CanonicalProtoPathMapping, CarvedFileDescriptorSummary,
    CarvedFirmwareOverlay, CarvedProtoDesc, EmbeddedChipResourceBundle, FileTocEntry,
    FileTocFirmwareExtractor, FirmwareOverlayTarget, PROTODESC_COLD_293_CATALOG,
    ProtodescColdCatalogEntry, build_descriptor_pool_from_carved_protos,
    carve_protodesc_cold_descriptors, carve_protodesc_cold_entries, compute_rfc1321_md5,
    decode_carved_file_descriptor, decode_embedded_chip_resources, decompress_brotli_csr_table,
    extract_carved_firmware_overlays, extract_filewrapper_toc, extract_indirect_filewrapper_toc,
    format_md5_hex, parse_and_verify_brotli_csr_tree, proto_scalar_type_name, render_proto_schema,
    resolve_embedded_proto_canonical_target, verify_descriptor_topological_closure,
    verify_filewrapper_toc_md5, verify_protodesc_cold_293_catalog_closure,
    verify_protodesc_cold_293_closure,
};
pub use self::disassembler::{
    AArch64ForensicsAnalyzer, AArch64FunctionSpec, CPythonAbi3IntrospectionReport, CallGraphEdge,
    CallGraphReport, ControlFlowGraphReport, DecodedIoctlCallsite, DeepSeaBitfieldPattern,
    DisassembledFunctionSpec, DriverBarMmioPollLoop, DriverQueuePattern, DwarfCfiFunctionRecord,
    EhFrameCfiParser, FdePrologueParityReport, IsaBitfieldSpec, IsaBitfieldSpecTable, IsaSlotKind,
    JumpTableDescriptor, PrologueLayoutReport, PrologueStructSizeVerification, RdiFieldAccess,
    RipRelativeStringLiteral, build_interprocedural_call_graph, build_x86_64_cfg,
    disassemble_function_spec, disassemble_x86_64_prologue, extract_driver_bar_mmio_poll_loops,
    extract_vliw_bitfield_pack_masks, introspect_cpython_abi3_binary, recover_aarch64_jump_table,
    recover_x86_64_jump_table, scan_x86_64_bitfield_encoders, scan_x86_64_driver_mmio_sequences,
    scan_x86_64_ioctl_callsites, scan_x86_64_rip_relative_strings,
    verify_extension_struct_layouts_against_golden, verify_fde_prologue_stack_frame_parity,
    verify_prologue_struct_sizes_against_golden,
};
pub use self::manifest::{
    ArtifactDigestReport, GoldenBinaryManifest, GoldenExportedSymbol, GoldenExtensionNode,
    GoldenLibtpuSo, GoldenPjrtSlot, GoldenProtoDescriptorEntry, GoldenProtoDescriptorsManifest,
    GoldenSdkSo, GoldenStructLayouts, compute_artifact_digests, parse_golden_binary_manifest,
    parse_golden_proto_descriptors, parse_golden_struct_layouts,
};
pub use self::relocations::{
    CppRttiHierarchyExtractor, ElfRelativeRelocation, ExtractedExtensionNode,
    ItaniumRttiClassDescriptor, ItaniumRttiKind, R_AARCH64_RELATIVE, R_X86_64_RELATIVE,
    ReconstructedCppVtable, ReconstructedFlatApiVtable, ReconstructedPjrtVtable,
    reconstruct_itanium_cpp_vtable, reconstruct_libtpu_sdk_vtable, reconstruct_pjrt_vtable,
    reconstruct_tpu_driver_vtable, resolve_relative_relocations,
};
pub use self::symbols::{
    BinaryFormatKind, BinarySectionInfo, BinarySymbolReport, CppNamespaceCategory,
    DemangledCppSymbol, DemangledSymbolEntry, ELF_STB_GLOBAL, ELF_STB_WEAK, EhFrameFunctionRange,
    EhFrameHdrEntry, Elf64Ehdr, Elf64Phdr, Elf64Rela, Elf64Shdr, Elf64Sym, ElfProgramHeaderInfo,
    ExportedBinarySymbol, ObjectCrateBinaryReport, RttiClassHierarchyEntry,
    demangle_and_classify_cpp_symbol, evaluate_eh_frame_cfa_state, inspect_binary_symbols,
    inspect_elf_program_headers, inspect_object_crate_sections, parse_eh_frame_fde_ranges,
    parse_eh_frame_hdr_table, parse_elf64_ehdr_zero_copy, reconstruct_rtti_hierarchy,
};

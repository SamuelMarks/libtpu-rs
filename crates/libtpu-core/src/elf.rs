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
    CarvedFileDescriptorSummary, CarvedProtoDesc, FileTocEntry,
    build_descriptor_pool_from_carved_protos, carve_protodesc_cold_descriptors,
    carve_protodesc_cold_entries, decode_carved_file_descriptor, decompress_brotli_csr_table,
    extract_filewrapper_toc, extract_indirect_filewrapper_toc, proto_scalar_type_name,
    render_proto_schema, verify_descriptor_topological_closure,
};
pub use self::disassembler::{
    ControlFlowGraphReport, DecodedIoctlCallsite, DisassembledFunctionSpec, PrologueLayoutReport,
    RdiFieldAccess, build_x86_64_cfg, disassemble_function_spec, disassemble_x86_64_prologue,
    scan_x86_64_ioctl_callsites,
};
pub use self::manifest::{
    ArtifactDigestReport, GoldenBinaryManifest, GoldenExportedSymbol, GoldenExtensionNode,
    GoldenLibtpuSo, GoldenPjrtSlot, GoldenSdkSo, GoldenStructLayouts, compute_artifact_digests,
    parse_golden_binary_manifest, parse_golden_struct_layouts,
};
pub use self::relocations::{
    ElfRelativeRelocation, ExtractedExtensionNode, R_AARCH64_RELATIVE, R_X86_64_RELATIVE,
    ReconstructedCppVtable, ReconstructedFlatApiVtable, ReconstructedPjrtVtable,
    reconstruct_itanium_cpp_vtable, reconstruct_libtpu_sdk_vtable, reconstruct_pjrt_vtable,
    reconstruct_tpu_driver_vtable, resolve_relative_relocations,
};
pub use self::symbols::{
    BinaryFormatKind, BinarySectionInfo, BinarySymbolReport, CppNamespaceCategory,
    DemangledCppSymbol, DemangledSymbolEntry, ELF_STB_GLOBAL, ELF_STB_WEAK, EhFrameFunctionRange,
    ExportedBinarySymbol, ObjectCrateBinaryReport, RttiClassHierarchyEntry,
    demangle_and_classify_cpp_symbol, inspect_binary_symbols, inspect_object_crate_sections,
    parse_eh_frame_fde_ranges, reconstruct_rtti_hierarchy,
};

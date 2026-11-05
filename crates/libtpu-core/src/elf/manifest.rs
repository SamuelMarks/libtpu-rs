//! Artifact digests (`SHA-256`/`BLAKE3`) and golden manifest parsers.
//!
//! Provides:
//! - [`compute_artifact_digests`] and [`ArtifactDigestReport`].
//! - [`parse_golden_binary_manifest`], [`GoldenBinaryManifest`], [`GoldenLibtpuSo`],
//!   [`GoldenSdkSo`], [`GoldenExportedSymbol`], and [`GoldenPjrtSlot`].
//! - [`parse_golden_struct_layouts`], [`GoldenStructLayouts`], and [`GoldenExtensionNode`].

use crate::error::{LibTpuError, LibTpuResult};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt::Write as _;

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

/// Single `.proto` descriptor metadata record in `docs/reverse_engineering/golden_proto_descriptors.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoldenProtoDescriptorEntry {
    /// Canonical `.proto` filename.
    pub name: String,
    /// Protobuf package namespace.
    pub package: String,
    /// Protobuf syntax (`"proto2"` or `"proto3"`).
    pub syntax: String,
    /// Imported `.proto` dependencies.
    pub imports: Vec<String>,
}

/// Strongly-typed deserialized representation of `docs/reverse_engineering/golden_proto_descriptors.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoldenProtoDescriptorsManifest {
    /// Source ELF section name (`"protodesc_cold"`).
    pub section_name: String,
    /// Virtual address of `protodesc_cold` in `libtpu.so`.
    pub section_vaddr: String,
    /// Byte length of `protodesc_cold` in `libtpu.so`.
    pub section_size_bytes: usize,
    /// Whether the carved descriptor set is topologically closed under imports.
    pub topologically_closed: bool,
    /// Count of carved `.proto` files.
    pub proto_files_count: usize,
    /// Ordered list of carved `.proto` descriptor records.
    pub descriptors: Vec<GoldenProtoDescriptorEntry>,
}

/// Parses and validates a `golden_proto_descriptors.json` string using `serde_json`.
///
/// # Arguments
///
/// * `json_str` - UTF-8 JSON content of `golden_proto_descriptors.json`.
///
/// # Errors
///
/// Returns [`LibTpuError::BinaryFormatError`] if `json_str` is invalid JSON or violates topological closure invariants.
pub fn parse_golden_proto_descriptors(
    json_str: &str,
) -> LibTpuResult<GoldenProtoDescriptorsManifest> {
    let manifest: GoldenProtoDescriptorsManifest =
        serde_json::from_str(json_str).map_err(|err| LibTpuError::BinaryFormatError {
            reason: format!("golden_proto_descriptors.json parse error: {err}"),
        })?;
    if manifest.section_name != "protodesc_cold"
        || !manifest.topologically_closed
        || manifest.descriptors.is_empty()
    {
        return Err(LibTpuError::BinaryFormatError {
            reason: String::from("golden_proto_descriptors.json invariant violation"),
        });
    }
    Ok(manifest)
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

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_digests_and_golden_manifests() {
        let digests = compute_artifact_digests(b"libtpu-0.0.50");
        assert_eq!(digests.size_bytes, 13);
        assert_eq!(digests.sha256_hex.len(), 64);
        assert_eq!(digests.blake3_hex.len(), 64);
        assert_eq!(digests.clone(), digests);
        assert_ne!(format!("{digests:?}"), "");

        let manifest_json =
            include_str!("../../../../docs/reverse_engineering/golden_binary_manifest.json");
        if let Ok(manifest) = parse_golden_binary_manifest(manifest_json) {
            assert_eq!(manifest.package, "libtpu");
            assert_eq!(manifest.libtpu_so.exported_functions_count, 78);
            assert_eq!(manifest.libtpu_so.pjrt_vtable_slots.len(), 138);
            assert_eq!(manifest.clone(), manifest);
            assert_ne!(format!("{manifest:?}"), "");
            if let Some(fn_sym) = manifest.libtpu_so.exported_functions.first() {
                assert_eq!(fn_sym.clone(), *fn_sym);
                assert_ne!(format!("{fn_sym:?}"), "");
            }
            if let Some(slot) = manifest.libtpu_so.pjrt_vtable_slots.first() {
                assert_eq!(slot.clone(), *slot);
                assert_ne!(format!("{slot:?}"), "");
            }
            assert_eq!(manifest.sdk_so.clone(), manifest.sdk_so);
            assert_ne!(format!("{:?}", manifest.sdk_so), "");
        }
        assert!(parse_golden_binary_manifest("not-json").is_err());
        let bad_manifest1 = manifest_json.replace(
            r#""ef2f7af1f1ca0f27db6a6a3adffdd2b275a3215b121c8f9bc3f6ce842fc7c700""#,
            r#""short""#,
        );
        assert!(parse_golden_binary_manifest(&bad_manifest1).is_err());
        let bad_manifest2 = manifest_json.replace(
            r#""6d65c03e2b584eaa3b546b6b64931c4b153ad5c4e2b404f578ea548c0fb6d2a4""#,
            r#""short""#,
        );
        assert!(parse_golden_binary_manifest(&bad_manifest2).is_err());
        let bad_manifest3 = manifest_json.replace(
            r#""exported_functions_count": 78"#,
            r#""exported_functions_count": 99"#,
        );
        assert!(parse_golden_binary_manifest(&bad_manifest3).is_err());

        let layouts_json =
            include_str!("../../../../docs/reverse_engineering/golden_struct_layouts.json");
        if let Ok(layouts) = parse_golden_struct_layouts(layouts_json) {
            assert_eq!(layouts.pjrt_api_size, 1144);
            assert_eq!(layouts.extensions.len(), 17);
            assert_eq!(layouts.pjrt_slots.len(), 138);
            assert_eq!(layouts.clone(), layouts);
            assert_ne!(format!("{layouts:?}"), "");
            if let Some(ext) = layouts.extensions.first() {
                assert_eq!(ext.clone(), *ext);
                assert_ne!(format!("{ext:?}"), "");
            }
        }
        assert!(parse_golden_struct_layouts("not-json").is_err());
        let bad_layouts1 =
            layouts_json.replace(r#""pjrt_api_size": 1144"#, r#""pjrt_api_size": 8"#);
        assert!(parse_golden_struct_layouts(&bad_layouts1).is_err());
        if let Ok(mut l) = parse_golden_struct_layouts(layouts_json) {
            l.extensions.clear();
            let s = serde_json::to_string(&l).unwrap_or_default();
            assert!(parse_golden_struct_layouts(&s).is_err());
        }
        if let Ok(mut l) = parse_golden_struct_layouts(layouts_json) {
            l.pjrt_slots.clear();
            let s = serde_json::to_string(&l).unwrap_or_default();
            assert!(parse_golden_struct_layouts(&s).is_err());
        }

        let proto_json =
            include_str!("../../../../docs/reverse_engineering/golden_proto_descriptors.json");
        if let Ok(proto_manifest) = parse_golden_proto_descriptors(proto_json) {
            assert_eq!(proto_manifest.section_name, "protodesc_cold");
            assert!(proto_manifest.topologically_closed);
            assert!(proto_manifest.descriptors.len() >= 59);
            assert_eq!(proto_manifest.clone(), proto_manifest);
            assert_ne!(format!("{proto_manifest:?}"), "");
            if let Some(first_desc) = proto_manifest.descriptors.first() {
                assert_eq!(first_desc.clone(), *first_desc);
                assert_ne!(format!("{first_desc:?}"), "");
            }
        }
        assert!(parse_golden_proto_descriptors("not-json").is_err());
        let bad_proto1 = proto_json.replace(r#""protodesc_cold""#, r#""wrong_sec""#);
        assert!(parse_golden_proto_descriptors(&bad_proto1).is_err());
        let bad_proto2 = proto_json.replace(
            r#""topologically_closed": true"#,
            r#""topologically_closed": false"#,
        );
        assert!(parse_golden_proto_descriptors(&bad_proto2).is_err());
        if let Ok(mut pm) = parse_golden_proto_descriptors(proto_json) {
            pm.descriptors.clear();
            let s = serde_json::to_string(&pm).unwrap_or_default();
            assert!(parse_golden_proto_descriptors(&s).is_err());
        }
    }
}

//! Core pure-Rust implementation of the Google Cloud TPU runtime (`libtpu-rs`).
//!
//! Provides strongly-typed domain primitives, a unified error model, a Hardware Abstraction Layer
//! (HAL) for both simulated and Linux VFIO/Gasket TPU devices, an HLO/`StableHLO` compiler and
//! serializer, a complete PJRT state engine, and legacy `TfTpu` / `TpuDriver` / `LibtpuSdk` engines.

#![cfg_attr(all(test, coverage_nightly), feature(coverage_attribute))]

pub mod compiler;
pub mod config;
pub mod driver;
pub mod elf;
pub mod error;
pub mod hal;
pub mod pjrt;
pub mod proto;
pub mod types;

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use std::fs;
    use std::path::Path;

    #[test]
    fn test_workspace_rust_doc_coverage_audit() {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let crates_dir = manifest_dir.parent().unwrap_or(manifest_dir);
        let rs_files = [
            crates_dir.join("libtpu-core/src/lib.rs"),
            crates_dir.join("libtpu-core/src/elf.rs"),
            crates_dir.join("libtpu-core/src/error.rs"),
            crates_dir.join("libtpu-core/src/types.rs"),
            crates_dir.join("libtpu-core/src/config.rs"),
            crates_dir.join("libtpu-core/src/proto.rs"),
            crates_dir.join("libtpu-core/src/compiler.rs"),
            crates_dir.join("libtpu-core/src/hal.rs"),
            crates_dir.join("libtpu-core/src/pjrt.rs"),
            crates_dir.join("libtpu-core/src/driver.rs"),
            crates_dir.join("libtpu-c-api/src/lib.rs"),
            crates_dir.join("libtpu-c-api/src/pjrt_types.rs"),
            crates_dir.join("libtpu-c-api/src/pjrt_error_api.rs"),
            crates_dir.join("libtpu-c-api/src/pjrt_client_api.rs"),
            crates_dir.join("libtpu-c-api/src/pjrt_device_api.rs"),
            crates_dir.join("libtpu-c-api/src/pjrt_executable_api.rs"),
            crates_dir.join("libtpu-c-api/src/pjrt_buffer_api.rs"),
            crates_dir.join("libtpu-c-api/src/extensions.rs"),
            crates_dir.join("libtpu-c-api/src/tpu_driver_types.rs"),
            crates_dir.join("libtpu-c-api/src/sdk_api.rs"),
            crates_dir.join("libtpu-c-api/src/legacy_tf_api.rs"),
            crates_dir.join("libtpu-c-api/src/rust_ergonomic_api.rs"),
            crates_dir.join("libtpu-py/src/lib.rs"),
        ];
        for path in rs_files {
            let content = fs::read_to_string(&path).unwrap_or_default();
            assert!(content.starts_with("//!"));
        }
    }
}

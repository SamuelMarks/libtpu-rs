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
    use std::path::{Path, PathBuf};

    fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    collect_rs_files(&path, out);
                } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
                    out.push(path);
                }
            }
        }
    }

    #[test]
    fn test_workspace_rust_doc_coverage_audit() {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let crates_dir = manifest_dir.parent().unwrap_or(manifest_dir);
        let mut rs_files = Vec::new();
        collect_rs_files(crates_dir, &mut rs_files);
        assert!(rs_files.len() >= 80);
        for path in rs_files {
            let content = fs::read_to_string(&path).unwrap_or_default();
            assert!(
                content.starts_with("//!"),
                "Missing //! file doc header in {}",
                path.display()
            );
        }
    }
}

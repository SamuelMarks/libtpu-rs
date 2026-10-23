//! Python FFI bridge and drop-in helper entrypoints for `libtpu-rs`.
//!
//! Provides [`PyTpuRuntime`] and C-ABI helper functions used by the `libtpu` Python package
//! to resolve library paths, manage `LIBTPU_INIT_ARGS`, and execute TPU programs.

#![cfg_attr(all(test, coverage_nightly), feature(coverage_attribute))]

use libtpu_core::config::TpuInitConfig;
use libtpu_core::error::LibTpuResult;
use libtpu_core::pjrt::PjrtCoreClient;
use libtpu_core::types::{DeviceId, ElementType, MemorySpaceId, Shape, TpuGeneration};

/// High-level Rust session backing the `libtpu` Python package (`PyTpuRuntime`).
#[derive(Debug)]
pub struct PyTpuRuntime {
    /// Active initialization configuration.
    pub config: TpuInitConfig,
    /// Underlying PJRT core client.
    pub client: PjrtCoreClient,
}

impl PyTpuRuntime {
    /// Creates a new [`PyTpuRuntime`] from a `--flag=value` initialization argument string.
    ///
    /// # Arguments
    ///
    /// * `init_args` - Whitespace-separated CLI flags (e.g., `"--tpu_generation=v5e"`).
    ///
    /// # Errors
    ///
    /// Returns [`libtpu_core::error::LibTpuError`] if `init_args` contains invalid flags.
    pub fn from_init_args(init_args: &str) -> LibTpuResult<Self> {
        let mut config = TpuInitConfig::default();
        config.apply_init_args(init_args)?;
        let client = PjrtCoreClient::from_simulated(&config);
        Ok(Self { config, client })
    }

    /// Returns the PJRT API version tuple `(major, minor)` (`(0, 116)`).
    pub fn pjrt_api_version() -> (i32, i32) {
        (0, 116)
    }

    /// Returns the active [`TpuGeneration`] of this runtime.
    pub fn generation(&self) -> TpuGeneration {
        self.config.generation
    }

    /// Returns the number of addressable TPU devices in this runtime.
    pub fn device_count(&self) -> usize {
        self.client.addressable_device_indices.len()
    }

    /// Compiles and executes an MLIR program on `device_id` doubling a 4-element `f32` input.
    ///
    /// # Arguments
    ///
    /// * `device_id` - Target logical TPU device ID.
    /// * `mlir_code` - MLIR source bytes.
    /// * `input` - Input 4-element `f32` array.
    ///
    /// # Errors
    ///
    /// Returns [`libtpu_core::error::LibTpuError`] if `device_id` is invalid, poisoned, or `mlir_code` fails to compile.
    pub fn execute_f32_4(
        &self,
        device_id: DeviceId,
        mlir_code: &[u8],
        input: &[f32; 4],
    ) -> LibTpuResult<[f32; 4]> {
        let shape = Shape {
            element_type: ElementType::F32,
            dims: vec![4],
            dynamic_dims: vec![false],
            layout: libtpu_core::types::MemoryLayout::default_tpu_tiled(1),
        };
        let mut b_in = Vec::with_capacity(16);
        for &v in input {
            b_in.extend_from_slice(&v.to_le_bytes());
        }
        let loaded = self.client.compile_and_load("mlir", mlir_code, b"")?;
        let buf0 = self.client.buffer_from_host(device_id, shape, &b_in)?;
        let outs = loaded.execute_on_device(device_id, MemorySpaceId(0), &[&buf0, &buf0])?;
        let mut out_arr = [0.0_f32; 4];
        for out_buf in outs {
            let mut raw = [0_u8; 16];
            let _ = self
                .client
                .hal
                .read_device_memory(device_id, out_buf.address, &mut raw);
            for (dst, chunk) in out_arr.iter_mut().zip(raw.chunks_exact(4)) {
                let mut bytes = [0_u8; 4];
                bytes.copy_from_slice(chunk);
                *dst = f32::from_le_bytes(bytes);
            }
        }
        Ok(out_arr)
    }
}

#[cfg(test)]
#[allow(for_loops_over_fallibles)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_py_tpu_runtime() {
        assert_eq!(PyTpuRuntime::pjrt_api_version(), (0, 116));
        assert!(PyTpuRuntime::from_init_args("--tpu_generation=invalid").is_err());

        let rt_res = PyTpuRuntime::from_init_args("--tpu_generation=v6e");
        assert!(rt_res.is_ok());
        for rt in rt_res {
            assert!(!format!("{rt:?}").is_empty());
            assert_eq!(rt.generation(), TpuGeneration::V6e);
            assert_eq!(rt.device_count(), 4);
            let sum = rt.execute_f32_4(
                DeviceId(0),
                b"module @py_add { stablehlo.add }",
                &[1.0, 2.0, 3.0, 4.0],
            );
            assert_eq!(sum.ok(), Some([2.0, 4.0, 6.0, 8.0]));
            assert!(rt.execute_f32_4(DeviceId(0), b"", &[1.0; 4]).is_err());
            assert!(rt
                .execute_f32_4(DeviceId(99), b"module @py_add { stablehlo.add }", &[1.0; 4],)
                .is_err());
            let _ = rt
                .client
                .hal
                .poison_device(DeviceId(0), String::from("poisoned"));
            assert!(rt
                .execute_f32_4(DeviceId(0), b"module @py_add { stablehlo.add }", &[1.0; 4],)
                .is_err());
        }
    }
}

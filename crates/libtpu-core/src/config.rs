//! Runtime configuration and `LIBTPU_INIT_ARGS` / environment variable parser.
//!
//! Replicates the flag and environment parsing behavior of `libtpu.so` (`LIBTPU_INIT_ARGS`,
//! `TPU_ACCELERATOR_TYPE`, `TPU_CHIPS_PER_HOST_BOUNDS`, `TPU_HOST_BOUNDS`, `TPU_VISIBLE_DEVICES`,
//! `TPU_PREMAPPED_BUFFER_SIZE`, `MEGASCALE_NUM_SLICES`, and `MEGASCALE_SLICE_ID`).

use crate::error::{LibTpuError, LibTpuResult};
use crate::types::{ByteSize, ChipCoordinates, DeviceId, ProcessIndex, TpuGeneration};
use lexopt::Arg;
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::sync::OnceLock;
use winnow::Parser;
use winnow::ascii::{digit1, multispace0};
use winnow::combinator::{delimited, separated};
use winnow::error::ContextError;
use winnow::token::one_of;

/// Parses a 3D mesh coordinate triple (`"2,2,1"`, `"2x2x1"`, or `"2X2X1"`) using `winnow`.
///
/// # Arguments
///
/// * `input` - Mutable input string slice reference for `winnow`.
///
/// # Errors
///
/// Returns a `winnow` [`ContextError`] if the input does not match three positive integers.
fn winnow_parse_bounds_vec(input: &mut &str) -> Result<Vec<u32>, ContextError> {
    let uint_token = delimited(multispace0, digit1.try_map(str::parse::<u32>), multispace0);
    separated(1..=4, uint_token, one_of([',', 'x', 'X'])).parse_next(input)
}

/// Process-wide global [`TpuInitConfig`] synchronized across `TfTpu_Initialize`,
/// `SetGlobalTPUArrayOp_DoWork`, `PJRT_Client_Create`, and `libtpu.set_init_args`.
static GLOBAL_INIT_CONFIG: OnceLock<Mutex<TpuInitConfig>> = OnceLock::new();

/// Returns a reference to the process-wide [`Mutex<TpuInitConfig>`].
fn global_config_mutex() -> &'static Mutex<TpuInitConfig> {
    GLOBAL_INIT_CONFIG.get_or_init(|| Mutex::new(TpuInitConfig::default()))
}

/// Returns a cloned snapshot of the current process-wide [`TpuInitConfig`].
#[must_use]
pub fn global_init_config() -> TpuInitConfig {
    global_config_mutex().lock().clone()
}

/// Replaces the process-wide [`TpuInitConfig`].
///
/// # Arguments
///
/// * `config` - New configuration state to install globally.
pub fn set_global_init_config(config: TpuInitConfig) {
    *global_config_mutex().lock() = config;
}

/// Applies CLI flag tokens to the process-wide [`TpuInitConfig`] (`TfTpu_Initialize`).
///
/// # Arguments
///
/// * `args` - Slice of flag strings.
///
/// # Errors
///
/// Returns [`LibTpuError::InvalidArgument`] if any flag token is malformed.
pub fn apply_global_cli_args(args: &[String]) -> LibTpuResult<()> {
    let joined = args.join(" ");
    let mut current = global_init_config();
    current.apply_init_args(&joined)?;
    set_global_init_config(current);
    Ok(())
}

/// Strongly-typed runtime configuration for initializing a TPU driver or PJRT client.
#[derive(Debug, Clone)]
pub struct TpuInitConfig {
    /// Target TPU hardware architecture generation.
    pub generation: TpuGeneration,
    /// 3D bounds `(x, y, z)` of chips per host.
    pub chips_per_host_bounds: ChipCoordinates,
    /// 3D bounds `(x, y, z)` of hosts in the TPU pod slice.
    pub host_bounds: ChipCoordinates,
    /// Optional explicit allowlist of visible device IDs.
    pub visible_devices: Option<Vec<DeviceId>>,
    /// Premapped DMA host bounce buffer size in bytes.
    pub premapped_buffer_size: ByteSize,
    /// Process index of the current worker in a multi-host slice.
    pub process_index: ProcessIndex,
    /// Total number of slices in a Megascale multi-slice workload.
    pub megascale_num_slices: u32,
    /// Current slice identifier in a Megascale workload.
    pub megascale_slice_id: u32,
    /// Whether asynchronous HBM allocations are enabled.
    pub allow_async_allocations: bool,
    /// Additional parsed `--xla_tpu_*` and `--tpu_*` key-value flags from `LIBTPU_INIT_ARGS`.
    pub extra_flags: BTreeMap<String, String>,
}

impl Default for TpuInitConfig {
    fn default() -> Self {
        Self {
            generation: TpuGeneration::V5e,
            chips_per_host_bounds: ChipCoordinates::new(2, 2, 1),
            host_bounds: ChipCoordinates::new(1, 1, 1),
            visible_devices: None,
            premapped_buffer_size: ByteSize(64 * 1024 * 1024),
            process_index: ProcessIndex(0),
            megascale_num_slices: 1,
            megascale_slice_id: 0,
            allow_async_allocations: true,
            extra_flags: BTreeMap::new(),
        }
    }
}

impl TpuInitConfig {
    /// Parses a 3D coordinate bound string such as `"2,2,1"` or `"2x2x1"` using `winnow`.
    ///
    /// # Arguments
    ///
    /// * `raw` - String representation of 3 positive integers separated by `,` or `x`.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if the string does not contain 3 positive integers.
    pub fn parse_bounds(raw: &str) -> LibTpuResult<ChipCoordinates> {
        let coords = winnow_parse_bounds_vec.parse(raw.trim()).map_err(|_| {
            LibTpuError::InvalidArgument {
                parameter: "bounds",
                reason: format!("Expected 3 coordinates 'x,y,z', got '{raw}'"),
            }
        })?;
        let [x, y, z] = coords.as_slice() else {
            return Err(LibTpuError::InvalidArgument {
                parameter: "bounds",
                reason: format!("Expected 3 coordinates 'x,y,z', got '{raw}'"),
            });
        };
        if *x == 0 || *y == 0 || *z == 0 {
            return Err(LibTpuError::InvalidArgument {
                parameter: "bounds",
                reason: String::from("Mesh bounds must be strictly positive (>= 1)"),
            });
        }
        Ok(ChipCoordinates::new(*x, *y, *z))
    }

    /// Parses a whitespace-separated `LIBTPU_INIT_ARGS` string via `lexopt` and applies flags to this config.
    ///
    /// # Arguments
    ///
    /// * `init_args` - Raw CLI flag string (e.g., `"--xla_tpu_allow_async_allocations=false --tpu_generation=v6e"`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if any flag is malformed or has an invalid value.
    pub fn apply_init_args(&mut self, init_args: &str) -> LibTpuResult<()> {
        let normalized_tokens: Vec<String> = init_args
            .split_whitespace()
            .map(|tok| {
                if tok.starts_with('-') && !tok.starts_with("--") && tok.len() > 1 {
                    format!("-{tok}")
                } else {
                    String::from(tok)
                }
            })
            .collect();
        let mut parser = lexopt::Parser::from_args(&normalized_tokens);
        while let Some(arg) = parser.next()? {
            let Arg::Long(raw_key) = arg else {
                return Err(LibTpuError::InvalidArgument {
                    parameter: "LIBTPU_INIT_ARGS",
                    reason: format!("Unexpected positional or short argument '{arg:?}'"),
                });
            };
            let key_owned = String::from(raw_key.trim_start_matches('-'));
            if key_owned.is_empty() {
                return Err(LibTpuError::InvalidArgument {
                    parameter: "LIBTPU_INIT_ARGS",
                    reason: String::from("Empty flag key"),
                });
            }
            let value_os = parser.optional_value();
            let value = value_os
                .as_deref()
                .and_then(|v| v.to_str())
                .unwrap_or("true");
            match key_owned.as_str() {
                "tpu_generation" | "xla_tpu_generation" => {
                    self.generation = TpuGeneration::from_name(value)?;
                }
                "xla_tpu_allow_async_allocations" | "tpu_allow_async_allocations" => {
                    self.allow_async_allocations = match value {
                        "true" | "1" => true,
                        "false" | "0" => false,
                        _ => {
                            return Err(LibTpuError::InvalidArgument {
                                parameter: "xla_tpu_allow_async_allocations",
                                reason: format!("Expected boolean, got '{value}'"),
                            });
                        }
                    };
                }
                "tpu_chips_per_host_bounds" => {
                    self.chips_per_host_bounds = Self::parse_bounds(value)?;
                }
                "tpu_host_bounds" => {
                    self.host_bounds = Self::parse_bounds(value)?;
                }
                _ => {
                    self.extra_flags.insert(key_owned, String::from(value));
                }
            }
        }
        Ok(())
    }

    /// Builds a [`TpuInitConfig`] from an explicit map of environment variables.
    ///
    /// # Arguments
    ///
    /// * `env_map` - Key-value pairs representing process environment variables.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if any environment variable contains invalid data.
    pub fn from_env_map(env_map: &BTreeMap<String, String>) -> LibTpuResult<Self> {
        let mut cfg = Self::default();
        if let Some(accel) = env_map.get("TPU_ACCELERATOR_TYPE") {
            cfg.generation = TpuGeneration::from_name(accel)?;
        }
        if let Some(chip_bounds) = env_map.get("TPU_CHIPS_PER_HOST_BOUNDS") {
            cfg.chips_per_host_bounds = Self::parse_bounds(chip_bounds)?;
        }
        for host_key in ["TPU_HOST_BOUNDS", "TPU_PROCESS_BOUNDS"] {
            if let Some(host_bounds) = env_map.get(host_key) {
                cfg.host_bounds = Self::parse_bounds(host_bounds)?;
            }
        }
        if let Some(worker_id) = env_map.get("TPU_WORKER_ID") {
            let wid =
                worker_id
                    .trim()
                    .parse::<i32>()
                    .map_err(|_| LibTpuError::InvalidArgument {
                        parameter: "TPU_WORKER_ID",
                        reason: format!("Invalid worker ID '{worker_id}'"),
                    })?;
            cfg.process_index = ProcessIndex(wid);
        }
        for passthrough_key in [
            "TPU_LIBRARY_PATH",
            "TPU_PROCESS_ADDRESSES",
            "TPU_PROCESS_PORT",
            "TPU_WORKER_HOSTNAMES",
            "TPU_SKIP_MDS_QUERY",
            "TPU_TOPOLOGY",
            "TPU_TOPOLOGY_WRAP",
            "TPU_MESH_CONTROLLER_ADDRESS",
            "TPU_MESH_CONTROLLER_PORT",
            "TPU_STDERR_LOG_LEVEL",
            "TPU_MIN_LOG_LEVEL",
            "LIBTPU_CHECKPOINTING_ENABLED",
            "TPU_RUNTIME_METRICS_PORTS",
        ] {
            if let Some(val) = env_map.get(passthrough_key) {
                cfg.extra_flags
                    .insert(String::from(passthrough_key), val.clone());
            }
        }
        if let Some(vis) = env_map.get("TPU_VISIBLE_DEVICES") {
            let mut ids = Vec::new();
            for item in vis.split(',') {
                let trimmed = item.trim();
                if !trimmed.is_empty() {
                    let parsed =
                        trimmed
                            .parse::<u32>()
                            .map_err(|_| LibTpuError::InvalidArgument {
                                parameter: "TPU_VISIBLE_DEVICES",
                                reason: format!("Invalid device ID '{item}'"),
                            })?;
                    ids.push(DeviceId(parsed));
                }
            }
            cfg.visible_devices = Some(ids);
        }
        if let Some(premap) = env_map.get("TPU_PREMAPPED_BUFFER_SIZE") {
            let bytes =
                premap
                    .trim()
                    .parse::<usize>()
                    .map_err(|_| LibTpuError::InvalidArgument {
                        parameter: "TPU_PREMAPPED_BUFFER_SIZE",
                        reason: format!("Invalid byte size '{premap}'"),
                    })?;
            cfg.premapped_buffer_size = ByteSize(bytes);
        }
        if let Some(slices) = env_map.get("MEGASCALE_NUM_SLICES") {
            let count = slices
                .trim()
                .parse::<u32>()
                .map_err(|_| LibTpuError::InvalidArgument {
                    parameter: "MEGASCALE_NUM_SLICES",
                    reason: format!("Invalid slice count '{slices}'"),
                })?;
            cfg.megascale_num_slices = count;
        }
        if let Some(slice_id) = env_map.get("MEGASCALE_SLICE_ID") {
            let id = slice_id
                .trim()
                .parse::<u32>()
                .map_err(|_| LibTpuError::InvalidArgument {
                    parameter: "MEGASCALE_SLICE_ID",
                    reason: format!("Invalid slice ID '{slice_id}'"),
                })?;
            cfg.megascale_slice_id = id;
        }
        if let Some(init_args) = env_map.get("LIBTPU_INIT_ARGS") {
            cfg.apply_init_args(init_args)?;
        }
        Ok(cfg)
    }

    /// Returns the total number of hosts in the configured TPU pod slice.
    #[must_use]
    pub const fn total_hosts(&self) -> u32 {
        self.host_bounds.x * self.host_bounds.y * self.host_bounds.z
    }

    /// Returns the number of physical TPU chips per host.
    #[must_use]
    pub const fn chips_per_host(&self) -> u32 {
        self.chips_per_host_bounds.x * self.chips_per_host_bounds.y * self.chips_per_host_bounds.z
    }
}

#[cfg(test)]
#[allow(for_loops_over_fallibles, clippy::cognitive_complexity)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_parse_bounds_and_init_args() {
        let b = TpuInitConfig::parse_bounds("2,2,1");
        assert!(b.is_ok());
        assert!(TpuInitConfig::parse_bounds("2,2").is_err());
        assert!(TpuInitConfig::parse_bounds("a,2,1").is_err());
        assert!(TpuInitConfig::parse_bounds("2,b,1").is_err());
        assert!(TpuInitConfig::parse_bounds("2,2,c").is_err());
        assert!(TpuInitConfig::parse_bounds("0,2,1").is_err());
        assert!(TpuInitConfig::parse_bounds("2,0,1").is_err());
        assert!(TpuInitConfig::parse_bounds("2,2,0").is_err());

        let mut cfg = TpuInitConfig::default();
        assert_eq!(cfg.total_hosts(), 1);
        assert_eq!(cfg.chips_per_host(), 4);
        assert_ne!(format!("{cfg:?}"), "");
        let cloned = cfg.clone();
        assert_eq!(cloned.chips_per_host(), 4);

        assert!(cfg
            .apply_init_args(
                "--tpu_generation=v6e --xla_tpu_allow_async_allocations=false --tpu_chips_per_host_bounds=4x2x1 --tpu_host_bounds=2,1,1 --tpu_driver_mode=vfio --xla_tpu_use_sparse_core=true --xla_tpu_custom_flag"
            )
            .is_ok());
        assert_eq!(cfg.generation, TpuGeneration::V6e);
        assert!(!cfg.allow_async_allocations);
        assert_eq!(cfg.chips_per_host(), 8);
        assert_eq!(cfg.total_hosts(), 2);
        assert_eq!(
            cfg.extra_flags
                .get("xla_tpu_custom_flag")
                .map(String::as_str),
            Some("true")
        );

        assert!(
            cfg.apply_init_args("-xla_tpu_allow_async_allocations=1")
                .is_ok()
        );
        assert!(cfg.allow_async_allocations);

        assert!(cfg.apply_init_args("---").is_err());
        assert!(cfg.apply_init_args("-").is_err());
        assert!(cfg.apply_init_args("-- positional").is_err());
        assert!(cfg.apply_init_args("positional_token").is_err());
        assert!(cfg.apply_init_args("--=val").is_err());
        assert!(
            cfg.apply_init_args("--xla_tpu_allow_async_allocations=maybe")
                .is_err()
        );
        assert!(cfg.apply_init_args("--tpu_generation=invalid").is_err());
        assert!(
            cfg.apply_init_args("--tpu_chips_per_host_bounds=bad")
                .is_err()
        );
        assert!(cfg.apply_init_args("--tpu_host_bounds=bad").is_err());
    }

    #[test]
    fn test_from_env_map() {
        assert!(TpuInitConfig::from_env_map(&BTreeMap::new()).is_ok());
        let mut env = BTreeMap::new();
        env.insert(String::from("TPU_ACCELERATOR_TYPE"), String::from("v4"));
        env.insert(
            String::from("TPU_CHIPS_PER_HOST_BOUNDS"),
            String::from("2,2,1"),
        );
        env.insert(String::from("TPU_HOST_BOUNDS"), String::from("1,2,1"));
        env.insert(
            String::from("TPU_VISIBLE_DEVICES"),
            String::from("0, 1, , 2,"),
        );
        env.insert(
            String::from("TPU_PREMAPPED_BUFFER_SIZE"),
            String::from("1048576"),
        );
        env.insert(String::from("MEGASCALE_NUM_SLICES"), String::from("4"));
        env.insert(String::from("MEGASCALE_SLICE_ID"), String::from("2"));
        env.insert(
            String::from("LIBTPU_INIT_ARGS"),
            String::from("--xla_tpu_flag=val"),
        );
        env.insert(String::from("TPU_WORKER_ID"), String::from("1"));
        env.insert(
            String::from("TPU_WORKER_HOSTNAMES"),
            String::from("worker-0,worker-1"),
        );

        let cfg = TpuInitConfig::from_env_map(&env);
        assert!(cfg.is_ok());
        for c in cfg {
            assert_eq!(c.generation, TpuGeneration::V4);
            assert_eq!(c.total_hosts(), 2);
            assert_eq!(c.process_index, ProcessIndex(1));
            assert_eq!(c.premapped_buffer_size, ByteSize(1_048_576));
            assert_eq!(c.megascale_num_slices, 4);
            assert_eq!(c.megascale_slice_id, 2);
            assert_eq!(c.visible_devices.as_ref().map(Vec::len), Some(3));
        }

        assert!(apply_global_cli_args(&[String::from("--tpu_generation=v6e")]).is_ok());
        assert!(apply_global_cli_args(&[String::from("---")]).is_err());
        assert_eq!(global_init_config().generation, TpuGeneration::V6e);
        set_global_init_config(TpuInitConfig::default());
        assert_eq!(global_init_config().generation, TpuGeneration::V5e);

        let bad_cases = [
            ("TPU_ACCELERATOR_TYPE", "bad"),
            ("TPU_CHIPS_PER_HOST_BOUNDS", "bad"),
            ("TPU_HOST_BOUNDS", "bad"),
            ("TPU_WORKER_ID", "bad"),
            ("TPU_VISIBLE_DEVICES", "0,abc"),
            ("TPU_PREMAPPED_BUFFER_SIZE", "abc"),
            ("MEGASCALE_NUM_SLICES", "abc"),
            ("MEGASCALE_SLICE_ID", "abc"),
            ("LIBTPU_INIT_ARGS", "---"),
        ];
        for (k, v) in bad_cases {
            let mut bad_env = BTreeMap::new();
            bad_env.insert(String::from(k), String::from(v));
            assert!(TpuInitConfig::from_env_map(&bad_env).is_err());
        }
    }

    proptest::proptest! {
        #[test]
        fn proptest_bounds_and_init_args(
            x in 1_u32..32,
            y in 1_u32..32,
            z in 1_u32..32,
            flag_val in "[a-zA-Z0-9_]{1,12}"
        ) {
            let comma_str = format!("{x},{y},{z}");
            let x_str = format!("{x}x{y}X{z}");
            proptest::prop_assert_eq!(
                TpuInitConfig::parse_bounds(&comma_str).ok(),
                Some(ChipCoordinates::new(x, y, z))
            );
            proptest::prop_assert_eq!(
                TpuInitConfig::parse_bounds(&x_str).ok(),
                Some(ChipCoordinates::new(x, y, z))
            );
            let mut cfg = TpuInitConfig::default();
            let args = format!("--tpu_chips_per_host_bounds={comma_str} --xla_tpu_custom={flag_val}");
            proptest::prop_assert!(cfg.apply_init_args(&args).is_ok());
            proptest::prop_assert_eq!(cfg.chips_per_host_bounds, ChipCoordinates::new(x, y, z));
        }
    }
}

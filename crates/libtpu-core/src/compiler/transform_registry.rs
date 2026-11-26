//! Custom XLA compiler pass registry (`XlaTransformRegistry`) backing `PJRT_XlaTransform_Extension`.
//!
//! Stores named compiler transforms registered via C-API (`PJRT_XlaTransform_Register`) or Rust
//! hooks and applies them deterministically to [`super::ast::AstModule`] and
//! [`crate::proto::HloModuleProto`] IR representations during [`super::CompiledProgram::compile`].

use super::ast::{AstModule, MlirAttribute};
use crate::error::{LibTpuError, LibTpuResult};
use crate::proto::HloModuleProto;
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::sync::OnceLock;

/// Optional Rust IR transformation hook invoked on registered passes during compilation.
pub type XlaTransformHook =
    fn(&str, Option<&mut AstModule>, &mut HloModuleProto) -> LibTpuResult<()>;

/// Metadata and optional execution hook for a single registered XLA transform pass.
#[derive(Debug, Clone, Copy)]
pub struct XlaTransformEntry {
    /// Opaque C `factory` pointer or caller token supplied at registration time.
    pub factory_token: usize,
    /// Optional in-process IR transformation callback.
    pub hook: Option<XlaTransformHook>,
}

/// Thread-safe registry of custom XLA HLO / `StableHLO` transforms (`PJRT_XlaTransform_Extension`).
#[derive(Debug, Default)]
pub struct XlaTransformRegistry {
    /// Registered passes keyed by unique pass name in deterministic lexicographical order.
    transforms: Mutex<BTreeMap<String, XlaTransformEntry>>,
}

impl XlaTransformRegistry {
    /// Creates an empty [`XlaTransformRegistry`].
    #[must_use]
    pub const fn new() -> Self {
        Self {
            transforms: Mutex::new(BTreeMap::new()),
        }
    }

    /// Registers or replaces a named XLA transform pass in the registry.
    ///
    /// # Arguments
    ///
    /// * `name` - Non-empty unique pass identifier (e.g., `"custom_fusion_pass"`).
    /// * `factory_token` - Opaque C factory function address or caller token.
    /// * `hook` - Optional Rust [`XlaTransformHook`] executed during compilation.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if `name` is empty or whitespace-only.
    pub fn register_transform(
        &self,
        name: &str,
        factory_token: usize,
        hook: Option<XlaTransformHook>,
    ) -> LibTpuResult<()> {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Err(LibTpuError::InvalidArgument {
                parameter: "name",
                reason: String::from("XLA transform pass name cannot be empty"),
            });
        }
        self.transforms.lock().insert(
            String::from(trimmed),
            XlaTransformEntry {
                factory_token,
                hook,
            },
        );
        Ok(())
    }

    /// Unregisters a previously registered XLA transform pass by `name`.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the pass to remove.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if `name` is empty or not registered.
    pub fn unregister_transform(&self, name: &str) -> LibTpuResult<()> {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Err(LibTpuError::InvalidArgument {
                parameter: "name",
                reason: String::from("XLA transform pass name cannot be empty"),
            });
        }
        self.transforms
            .lock()
            .remove(trimmed)
            .map(|_| ())
            .ok_or_else(|| LibTpuError::InvalidArgument {
                parameter: "name",
                reason: format!("XLA transform pass '{trimmed}' is not registered"),
            })
    }

    /// Returns the sorted list of all currently registered XLA transform pass names.
    #[must_use]
    pub fn list_transforms(&self) -> Vec<String> {
        self.transforms.lock().keys().cloned().collect()
    }

    /// Returns the number of currently registered XLA transform passes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.transforms.lock().len()
    }

    /// Returns `true` when zero XLA transform passes are currently registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.transforms.lock().is_empty()
    }

    /// Removes all registered transforms from this registry.
    pub fn clear(&self) {
        self.transforms.lock().clear();
    }

    /// Applies all registered transforms in deterministic order to `ast_module` and `hlo_module`,
    /// recording applied pass names in module frontend attributes and returning the pass count.
    ///
    /// # Arguments
    ///
    /// * `ast_module` - Optional mutable [`AstModule`] when compiling MLIR / bytecode sources.
    /// * `hlo_module` - Mutable [`HloModuleProto`] representation of the compiled program.
    ///
    /// # Errors
    ///
    /// Propagates any [`LibTpuError`] returned by a registered [`XlaTransformHook`].
    pub fn apply_registered_transforms(
        &self,
        mut ast_module: Option<&mut AstModule>,
        hlo_module: &mut HloModuleProto,
    ) -> LibTpuResult<usize> {
        let snapshot: Vec<(String, XlaTransformEntry)> = self
            .transforms
            .lock()
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        if snapshot.is_empty() {
            return Ok(0);
        }

        let joined_names = snapshot
            .iter()
            .map(|(n, _)| n.as_str())
            .collect::<Vec<_>>()
            .join(",");
        if let Some(ast) = ast_module.as_deref_mut() {
            ast.attributes.insert(
                String::from("xla_transforms_applied"),
                MlirAttribute::String(joined_names.clone()),
            );
        }
        let mut fe_attrs = hlo_module.frontend_attributes.take().unwrap_or_default();
        fe_attrs
            .map
            .insert(String::from("xla_transforms_applied"), joined_names);
        hlo_module.frontend_attributes = Some(fe_attrs);

        for (pass_name, entry) in &snapshot {
            if let Some(hook_fn) = entry.hook {
                hook_fn(pass_name, ast_module.as_deref_mut(), hlo_module)?;
            }
        }
        Ok(snapshot.len())
    }
}

/// Process-wide singleton [`XlaTransformRegistry`] backing `PJRT_XlaTransform_Extension` and `CompiledProgram`.
static GLOBAL_XLA_TRANSFORM_REGISTRY: OnceLock<XlaTransformRegistry> = OnceLock::new();

/// Returns a reference to the process-wide [`XlaTransformRegistry`].
#[must_use]
pub fn global_xla_transform_registry() -> &'static XlaTransformRegistry {
    GLOBAL_XLA_TRANSFORM_REGISTRY.get_or_init(XlaTransformRegistry::new)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    /// Test transform hook mutating `ast` and `hlo` or failing when `pass_name == "pass_err"`.
    fn sample_hook(
        pass_name: &str,
        ast: Option<&mut AstModule>,
        hlo: &mut HloModuleProto,
    ) -> LibTpuResult<()> {
        if pass_name == "pass_err" {
            return Err(LibTpuError::CompilationFailed {
                reason: String::from("synthetic transform hook failure"),
            });
        }
        if let Some(ast_mod) = ast {
            ast_mod
                .attributes
                .insert(format!("hook_{pass_name}"), MlirAttribute::Bool(true));
        }
        hlo.name = format!("{}_{pass_name}", hlo.name);
        Ok(())
    }

    #[test]
    fn test_xla_transform_registry_lifecycle_and_application() -> LibTpuResult<()> {
        let reg = XlaTransformRegistry::default();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);

        let mut hlo = HloModuleProto {
            name: String::from("base"),
            ..HloModuleProto::default()
        };
        assert_eq!(reg.apply_registered_transforms(None, &mut hlo)?, 0);

        assert!(reg.register_transform("   ", 0, None).is_err());
        assert!(reg.unregister_transform("").is_err());
        assert!(reg.unregister_transform("missing").is_err());

        reg.register_transform("pass_b", 0x100, None)?;
        reg.register_transform("pass_a", 0x200, Some(sample_hook))?;
        assert!(!reg.is_empty());
        assert_eq!(reg.len(), 2);
        assert_eq!(
            reg.list_transforms(),
            vec![String::from("pass_a"), String::from("pass_b")]
        );

        let mut ast = AstModule::default();
        assert_eq!(
            reg.apply_registered_transforms(Some(&mut ast), &mut hlo)?,
            2
        );
        assert_eq!(hlo.name, "base_pass_a");
        assert_eq!(
            ast.attributes.get("xla_transforms_applied"),
            Some(&MlirAttribute::String(String::from("pass_a,pass_b")))
        );
        assert_eq!(
            ast.attributes.get("hook_pass_a"),
            Some(&MlirAttribute::Bool(true))
        );
        let fe_val = hlo
            .frontend_attributes
            .as_ref()
            .and_then(|fa| fa.map.get("xla_transforms_applied").cloned());
        assert_eq!(fe_val, Some(String::from("pass_a,pass_b")));

        reg.unregister_transform("pass_b")?;
        assert_eq!(reg.list_transforms(), vec![String::from("pass_a")]);

        reg.register_transform("pass_err", 0x300, Some(sample_hook))?;
        assert!(reg.apply_registered_transforms(None, &mut hlo).is_err());

        reg.clear();
        assert!(reg.is_empty());
        assert!(global_xla_transform_registry().len() < 1024);
        Ok(())
    }
}

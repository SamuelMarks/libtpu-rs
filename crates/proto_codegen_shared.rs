//! Shared `protox` + `prost-build` topological shard planner and code generator.
//!
//! Partitions all 744 `.proto` schemas (`738` carved `protodesc_cold` files + `6` `libtpu_ext`
//! schemas) across `NUM_PROTO_SHARDS` workspace crates using Tarjan's strongly-connected
//! components algorithm so each crate compiles ~1,200 messages (~2 GB peak `rustc` RSS) while
//! preserving 100% 1:1 `generated_prost_schemas` type identity in `libtpu-core`.

use heck::{ToSnakeCase, ToUpperCamelCase};
use prost_types::{DescriptorProto, EnumDescriptorProto, FileDescriptorSet};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// Total number of topological Protobuf workspace shard crates (`libtpu-proto-s0`..`s5`).
pub const NUM_PROTO_SHARDS: usize = 6;

/// Soft message-count threshold per non-leaf shard before advancing to the next shard.
const NON_LEAF_SHARD_TARGET_MSGS: usize = 1250;

/// Types receiving an extra `#[derive(Eq)]` attribute in `prost-build`.
pub const EXTRA_EQ_TYPES: &[&str] = &[
    ".libtpu.ext.deepsea.LloBundleScheduleProto",
    ".libtpu.ext.deepsea.LloSoftwarePipelineScheduleProto",
    ".libtpu.ext.megascale.MegascaleCongestionControlConfigProto",
];

/// Sanitizes a Rust identifier following `prost-build`'s exact keyword and leading-digit rules.
///
/// # Arguments
///
/// * `raw` - Candidate Rust identifier produced by `heck` case conversion.
#[must_use]
pub fn prost_sanitize_identifier(raw: &str) -> String {
    match raw {
        "as" | "break" | "const" | "continue" | "else" | "enum" | "false" | "fn" | "for" | "if"
        | "impl" | "in" | "let" | "loop" | "match" | "mod" | "move" | "mut" | "pub" | "ref"
        | "return" | "static" | "struct" | "trait" | "true" | "type" | "unsafe" | "use"
        | "where" | "while" | "dyn" | "abstract" | "become" | "box" | "do" | "final" | "macro"
        | "override" | "priv" | "typeof" | "unsized" | "virtual" | "yield" | "async" | "await"
        | "try" | "gen" => format!("r#{raw}"),
        "_" | "super" | "self" | "Self" | "extern" | "crate" => format!("{raw}_"),
        s if s.chars().next().is_some_and(char::is_numeric) => format!("_{raw}"),
        _ => String::from(raw),
    }
}

/// Converts a Protobuf identifier to `prost-build`'s `snake_case` Rust module/field identifier.
///
/// # Arguments
///
/// * `ident` - Raw Protobuf package segment or message name.
#[must_use]
pub fn prost_to_snake(ident: &str) -> String {
    prost_sanitize_identifier(&ident.to_snake_case())
}

/// Converts a Protobuf identifier to `prost-build`'s `UpperCamelCase` Rust type identifier.
///
/// # Arguments
///
/// * `ident` - Raw Protobuf message or enum name.
#[must_use]
pub fn prost_to_upper_camel(ident: &str) -> String {
    prost_sanitize_identifier(&ident.to_upper_camel_case())
}

/// Converts a dotted Protobuf package name into its `::`-joined Rust module path.
///
/// # Arguments
///
/// * `pkg` - Protobuf package string (may be empty).
#[must_use]
pub fn package_to_rust_mod_path(pkg: &str) -> String {
    if pkg.is_empty() {
        String::new()
    } else {
        pkg.split('.')
            .map(prost_to_snake)
            .collect::<Vec<_>>()
            .join("::")
    }
}

/// Converts a dotted Protobuf package name into the `.rs` filename emitted by `prost-build`.
///
/// # Arguments
///
/// * `pkg` - Protobuf package string (may be empty).
#[must_use]
fn package_to_prost_filename(pkg: &str) -> String {
    if pkg.is_empty() {
        String::from("_.rs")
    } else {
        let joined = pkg
            .split('.')
            .map(prost_to_snake)
            .collect::<Vec<_>>()
            .join(".");
        format!("{joined}.rs")
    }
}

/// Recursively collects all `.proto` file paths under `dir` and registers `cargo:rerun-if-changed`.
///
/// # Arguments
///
/// * `dir` - Directory to traverse recursively.
/// * `out_files` - Output vector receiving discovered `.proto` paths.
///
/// # Errors
///
/// Returns [`std::io::Error`] if reading `dir` or any subdirectory fails.
fn collect_proto_files_recursively(
    dir: &Path,
    out_files: &mut Vec<PathBuf>,
) -> std::io::Result<()> {
    println!("cargo:rerun-if-changed={}", dir.display());
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|res| res.ok().map(|e| e.path()))
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            collect_proto_files_recursively(&path, out_files)?;
        } else if path.extension().and_then(|s| s.to_str()) == Some("proto") {
            println!("cargo:rerun-if-changed={}", path.display());
            out_files.push(path);
        }
    }
    Ok(())
}

/// Removes stale `.rs` files in `out_dir` prior to `prost-build` code generation.
///
/// # Arguments
///
/// * `out_dir` - Cargo `OUT_DIR` directory containing generated Rust artifacts.
///
/// # Errors
///
/// Returns [`std::io::Error`] if enumerating or removing stale `.rs` files in `out_dir` fails.
pub fn clean_stale_generated_rs(out_dir: &Path) -> std::io::Result<()> {
    if !out_dir.exists() {
        return Ok(());
    }
    for entry in std::fs::read_dir(out_dir)? {
        let path = entry?.path();
        if path.extension().and_then(|s| s.to_str()) == Some("rs") {
            std::fs::remove_file(path)?;
        }
    }
    Ok(())
}

/// Returns `true` if `pkg` is a `libtpu.ext.*` schema compiled locally in `libtpu-core`.
///
/// # Arguments
///
/// * `pkg` - Dotted Protobuf package name.
#[must_use]
pub fn is_core_local_package(pkg: &str) -> bool {
    pkg == "libtpu" || pkg.starts_with("libtpu.")
}

/// Injects rustdoc comments onto every `prost-build` item in `out_dir`.
///
/// # Arguments
///
/// * `out_dir` - Cargo `OUT_DIR` directory containing `prost-build` generated `.rs` files.
/// * `include_filename` - Name of the `prost-build` module include file to document.
///
/// # Errors
///
/// Returns [`std::io::Error`] if reading or writing generated `.rs` files in `out_dir` fails.
pub fn ensure_doc_comments_in_generated_rs(
    out_dir: &Path,
    include_filename: &str,
) -> std::io::Result<()> {
    for entry in std::fs::read_dir(out_dir)? {
        let path = entry?.path();
        if path.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        let fname = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if fname == include_filename {
            let raw_inc = std::fs::read_to_string(&path)?;
            let mut doc_inc = String::with_capacity(raw_inc.len() + 4096);
            for line in raw_inc.lines() {
                if line.trim_start().starts_with("pub mod ") {
                    let indent: String = line.chars().take_while(|c| c.is_whitespace()).collect();
                    let _ = writeln!(doc_inc, "{indent}/// Generated protobuf package module.");
                }
                let _ = writeln!(doc_inc, "{line}");
            }
            std::fs::write(&path, doc_inc)?;
            continue;
        }
        if fname.starts_with("generated_") {
            continue;
        }

        let raw = std::fs::read_to_string(&path)?;
        let mut out = String::with_capacity(raw.len() + 4096);
        let mut pending_docs: Vec<String> = Vec::new();
        let mut pending_attrs: Vec<String> = Vec::new();
        for line in raw.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("///") || trimmed.starts_with("//!") {
                pending_docs.push(String::from(line));
                continue;
            }
            if trimmed.starts_with("#[") {
                pending_attrs.push(String::from(line));
                continue;
            }
            let is_item = trimmed.starts_with("pub struct ")
                || trimmed.starts_with("pub enum ")
                || trimmed.starts_with("pub mod ")
                || trimmed.starts_with("pub type ")
                || (trimmed.starts_with("pub ") && trimmed.contains(": "))
                || (!pending_attrs.is_empty()
                    && !trimmed.is_empty()
                    && !trimmed.starts_with('}')
                    && !trimmed.starts_with("//"))
                || (trimmed.contains(" = ") && trimmed.ends_with(','));

            if is_item {
                let indent: String = line.chars().take_while(|c| c.is_whitespace()).collect();
                let _ = writeln!(out, "{indent}/// Generated protobuf schema item.");
                pending_docs.clear();
            }
            for doc_line in std::mem::take(&mut pending_docs) {
                let safe_doc = doc_line.replace('[', "(").replace(']', ")");
                let _ = writeln!(out, "{safe_doc}");
            }
            for attr_line in std::mem::take(&mut pending_attrs) {
                let _ = writeln!(out, "{attr_line}");
            }
            let _ = writeln!(out, "{line}");
        }
        for doc_line in pending_docs {
            let safe_doc = doc_line.replace('[', "(").replace(']', ")");
            let _ = writeln!(out, "{safe_doc}");
        }
        for attr_line in pending_attrs {
            let _ = writeln!(out, "{attr_line}");
        }
        std::fs::write(&path, out)?;
    }
    Ok(())
}

/// Counts non-map messages recursively within a Protobuf `DescriptorProto`.
///
/// # Arguments
///
/// * `msg` - Protobuf `DescriptorProto` subtree root.
#[must_use]
fn count_non_map_messages(msg: &DescriptorProto) -> usize {
    if msg
        .options
        .as_ref()
        .and_then(|o| o.map_entry)
        .unwrap_or(false)
    {
        return 0;
    }
    let nested_sum: usize = msg.nested_type.iter().map(count_non_map_messages).sum();
    1 + nested_sum
}

/// Mutable state for Tarjan's strongly-connected components traversal over Protobuf packages.
struct TarjanState<'a> {
    /// Directed package dependency adjacency map.
    deps: &'a BTreeMap<String, BTreeSet<String>>,
    /// Next discovery index counter.
    counter: usize,
    /// Discovery index per package.
    index: BTreeMap<String, usize>,
    /// Low-link index per package.
    lowlink: BTreeMap<String, usize>,
    /// Active DFS stack membership set.
    on_stack: BTreeSet<String>,
    /// Active DFS stack.
    stack: Vec<String>,
    /// Output SCCs in topological order (dependencies precede dependents).
    sccs: Vec<Vec<String>>,
}

impl TarjanState<'_> {
    /// Visits `v` and its transitive dependencies in Tarjan's SCC DFS.
    ///
    /// # Arguments
    ///
    /// * `v` - Package name to visit.
    fn strong_connect(&mut self, v: &str) {
        let v_idx = self.counter;
        self.counter += 1;
        self.index.insert(String::from(v), v_idx);
        self.lowlink.insert(String::from(v), v_idx);
        self.stack.push(String::from(v));
        self.on_stack.insert(String::from(v));

        if let Some(neighbors) = self.deps.get(v) {
            for w in neighbors {
                if !self.index.contains_key(w.as_str()) {
                    self.strong_connect(w);
                    let w_low = self.lowlink.get(w.as_str()).copied().unwrap_or(v_idx);
                    if let Some(v_low) = self.lowlink.get_mut(v) {
                        *v_low = (*v_low).min(w_low);
                    }
                } else if self.on_stack.contains(w.as_str()) {
                    let w_idx = self.index.get(w.as_str()).copied().unwrap_or(v_idx);
                    if let Some(v_low) = self.lowlink.get_mut(v) {
                        *v_low = (*v_low).min(w_idx);
                    }
                }
            }
        }

        let is_root = self.lowlink.get(v).copied() == self.index.get(v).copied();
        if is_root {
            let mut comp = Vec::new();
            while let Some(w) = self.stack.pop() {
                self.on_stack.remove(&w);
                let done = w == v;
                comp.push(w);
                if done {
                    break;
                }
            }
            comp.sort();
            self.sccs.push(comp);
        }
    }
}

/// Package dependency graph and message counts extracted from a `FileDescriptorSet`.
type PackageGraph = (
    BTreeMap<String, usize>,
    BTreeMap<String, BTreeSet<String>>,
    BTreeMap<String, BTreeSet<String>>,
);

/// Builds `(pkg_msgs, pkg_deps, rev_deps)` over all non-core-local packages in `fds`.
///
/// # Arguments
///
/// * `fds` - Combined `FileDescriptorSet` covering all 744 `.proto` files.
#[must_use]
fn build_package_dependency_graph(fds: &FileDescriptorSet) -> PackageGraph {
    let mut file_to_pkg: BTreeMap<String, String> = BTreeMap::new();
    let mut pkg_msgs: BTreeMap<String, usize> = BTreeMap::new();
    for file in &fds.file {
        let pkg = file.package.as_deref().unwrap_or("");
        if pkg == "google.protobuf" || is_core_local_package(pkg) {
            continue;
        }
        if let Some(name) = file.name.as_deref() {
            file_to_pkg.insert(String::from(name), String::from(pkg));
        }
        let msg_cnt: usize = file.message_type.iter().map(count_non_map_messages).sum();
        let enum_cnt = file.enum_type.len();
        *pkg_msgs.entry(String::from(pkg)).or_insert(0) += msg_cnt + enum_cnt;
    }

    let mut pkg_deps: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut rev_deps: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for pkg in pkg_msgs.keys() {
        pkg_deps.entry(pkg.clone()).or_default();
        rev_deps.entry(pkg.clone()).or_default();
    }
    for file in &fds.file {
        let pkg = file.package.as_deref().unwrap_or("");
        if pkg == "google.protobuf" || is_core_local_package(pkg) {
            continue;
        }
        for dep_file in &file.dependency {
            if let Some(dep_pkg) = file_to_pkg.get(dep_file.as_str())
                && dep_pkg != pkg
                && dep_pkg != "google.protobuf"
            {
                pkg_deps
                    .entry(String::from(pkg))
                    .or_default()
                    .insert(dep_pkg.clone());
                rev_deps
                    .entry(dep_pkg.clone())
                    .or_default()
                    .insert(String::from(pkg));
            }
        }
    }
    (pkg_msgs, pkg_deps, rev_deps)
}

/// Computes a deterministic topological mapping from Protobuf `package` name to shard index
/// `0..NUM_PROTO_SHARDS` such that every imported package resides in a shard `<=` its importer.
///
/// # Arguments
///
/// * `fds` - Combined `FileDescriptorSet` covering all 744 `.proto` files.
#[must_use]
pub fn compute_package_shards(fds: &FileDescriptorSet) -> BTreeMap<String, usize> {
    let (pkg_msgs, pkg_deps, rev_deps) = build_package_dependency_graph(fds);

    let mut tarjan = TarjanState {
        deps: &pkg_deps,
        counter: 0,
        index: BTreeMap::new(),
        lowlink: BTreeMap::new(),
        on_stack: BTreeSet::new(),
        stack: Vec::new(),
        sccs: Vec::new(),
    };
    for pkg in pkg_msgs.keys() {
        if !tarjan.index.contains_key(pkg.as_str()) {
            tarjan.strong_connect(pkg);
        }
    }

    let mut non_leaves: Vec<Vec<String>> = Vec::new();
    let mut leaves: Vec<Vec<String>> = Vec::new();
    for comp in tarjan.sccs {
        let comp_set: BTreeSet<&str> = comp.iter().map(String::as_str).collect();
        let has_external_dependent = comp.iter().any(|p| {
            rev_deps
                .get(p.as_str())
                .is_some_and(|rds| rds.iter().any(|r| !comp_set.contains(r.as_str())))
        });
        if has_external_dependent {
            non_leaves.push(comp);
        } else {
            leaves.push(comp);
        }
    }

    let mut pkg_to_shard: BTreeMap<String, usize> = BTreeMap::new();
    let mut loads = [0_usize; NUM_PROTO_SHARDS];

    for comp in &non_leaves {
        let comp_set: BTreeSet<&str> = comp.iter().map(String::as_str).collect();
        let min_shard = comp
            .iter()
            .filter_map(|p| pkg_deps.get(p.as_str()))
            .flatten()
            .filter(|d| !comp_set.contains(d.as_str()))
            .filter_map(|d| pkg_to_shard.get(d.as_str()).copied())
            .max()
            .unwrap_or(0);
        let weight: usize = comp
            .iter()
            .map(|p| pkg_msgs.get(p.as_str()).copied().unwrap_or(0))
            .sum();
        let mut target = min_shard;
        while target + 1 < NUM_PROTO_SHARDS
            && loads.get(target).copied().unwrap_or(0) > 0
            && loads.get(target).copied().unwrap_or(0) + weight > NON_LEAF_SHARD_TARGET_MSGS
        {
            target += 1;
        }
        for p in comp {
            pkg_to_shard.insert(p.clone(), target);
        }
        if let Some(slot) = loads.get_mut(target) {
            *slot += weight;
        }
    }

    leaves.sort_by(|a, b| {
        let wa: usize = a
            .iter()
            .map(|p| pkg_msgs.get(p.as_str()).copied().unwrap_or(0))
            .sum();
        let wb: usize = b
            .iter()
            .map(|p| pkg_msgs.get(p.as_str()).copied().unwrap_or(0))
            .sum();
        wb.cmp(&wa).then_with(|| a.cmp(b))
    });

    for comp in &leaves {
        let comp_set: BTreeSet<&str> = comp.iter().map(String::as_str).collect();
        let min_shard = comp
            .iter()
            .filter_map(|p| pkg_deps.get(p.as_str()))
            .flatten()
            .filter(|d| !comp_set.contains(d.as_str()))
            .filter_map(|d| pkg_to_shard.get(d.as_str()).copied())
            .max()
            .unwrap_or(0);
        let weight: usize = comp
            .iter()
            .map(|p| pkg_msgs.get(p.as_str()).copied().unwrap_or(0))
            .sum();
        let best = (min_shard..NUM_PROTO_SHARDS)
            .min_by_key(|&k| loads.get(k).copied().unwrap_or(usize::MAX))
            .unwrap_or(min_shard);
        for p in comp {
            pkg_to_shard.insert(p.clone(), best);
        }
        if let Some(slot) = loads.get_mut(best) {
            *slot += weight;
        }
    }

    // Ensure empty package (if present) always has an entry.
    if pkg_msgs.contains_key("") {
        pkg_to_shard.entry(String::new()).or_insert(0);
    }

    pkg_to_shard
}

/// Recursively registers exact `extern_path` mappings for an external enum descriptor.
///
/// # Arguments
///
/// * `proto_prefix` - Dotted Protobuf scope (leading dot excluded; may be empty).
/// * `rust_prefix` - Fully-qualified Rust module path (e.g., `"::libtpu_proto_s0::xla"`).
/// * `enm` - Protobuf `EnumDescriptorProto` to register.
/// * `extern_map` - Output map from `".pkg.Enum"` to `"::shard::pkg::Enum"`.
fn collect_extern_enum(
    proto_prefix: &str,
    rust_prefix: &str,
    enm: &EnumDescriptorProto,
    extern_map: &mut BTreeMap<String, String>,
) {
    let Some(name) = enm.name.as_deref().filter(|s| !s.is_empty()) else {
        return;
    };
    let fq_proto = if proto_prefix.is_empty() {
        format!(".{name}")
    } else {
        format!(".{proto_prefix}.{name}")
    };
    let ident = prost_to_upper_camel(name);
    let fq_rust = format!("{rust_prefix}::{ident}");
    extern_map.insert(fq_proto, fq_rust);
}

/// Recursively registers exact `extern_path` mappings for a message and all nested messages/enums.
///
/// # Arguments
///
/// * `proto_prefix` - Dotted Protobuf scope (leading dot excluded; may be empty).
/// * `rust_prefix` - Fully-qualified Rust module path (e.g., `"::libtpu_proto_s0::xla"`).
/// * `msg` - Protobuf `DescriptorProto` to traverse.
/// * `extern_map` - Output map from `".pkg.Msg"` to `"::shard::pkg::Msg"`.
fn collect_extern_message(
    proto_prefix: &str,
    rust_prefix: &str,
    msg: &DescriptorProto,
    extern_map: &mut BTreeMap<String, String>,
) {
    if msg
        .options
        .as_ref()
        .and_then(|o| o.map_entry)
        .unwrap_or(false)
    {
        return;
    }
    let Some(name) = msg.name.as_deref().filter(|s| !s.is_empty()) else {
        return;
    };
    let next_proto = if proto_prefix.is_empty() {
        String::from(name)
    } else {
        format!("{proto_prefix}.{name}")
    };
    let fq_proto = format!(".{next_proto}");
    let ident = prost_to_upper_camel(name);
    let fq_rust = format!("{rust_prefix}::{ident}");
    extern_map.insert(fq_proto, fq_rust);

    let child_mod = prost_to_snake(name);
    let next_rust = format!("{rust_prefix}::{child_mod}");
    for enm in &msg.enum_type {
        collect_extern_enum(&next_proto, &next_rust, enm, extern_map);
    }
    for nested in &msg.nested_type {
        collect_extern_message(&next_proto, &next_rust, nested, extern_map);
    }
}

/// Recursively collects `(full_proto_name, rust_type_path)` entries for a message and its nested
/// non-map messages relative to `root_prefix`.
///
/// # Arguments
///
/// * `proto_prefix` - Enclosing Protobuf scope (e.g., `"xla"` or `"xla.OuterMsg"`).
/// * `rust_mod_prefix` - Enclosing Rust module path (e.g., `"xla"` or `"xla::outer_msg"`).
/// * `root_prefix` - Leading Rust prefix (`"crate"` inside a shard crate or `"generated_prost_schemas"` in `libtpu-core`).
/// * `msg` - Protobuf `DescriptorProto` to traverse.
/// * `entries` - Output map from canonical `full_proto_name` to `rust_type_path`.
pub fn collect_message_bijection_entries(
    proto_prefix: &str,
    rust_mod_prefix: &str,
    root_prefix: &str,
    msg: &DescriptorProto,
    entries: &mut BTreeMap<String, String>,
) {
    if msg
        .options
        .as_ref()
        .and_then(|o| o.map_entry)
        .unwrap_or(false)
    {
        return;
    }
    let Some(msg_name) = msg.name.as_deref().filter(|s| !s.is_empty()) else {
        return;
    };
    let full_proto_name = if proto_prefix.is_empty() {
        String::from(msg_name)
    } else {
        format!("{proto_prefix}.{msg_name}")
    };
    let struct_ident = prost_to_upper_camel(msg_name);
    let rust_type_path = if rust_mod_prefix.is_empty() {
        format!("{root_prefix}::{struct_ident}")
    } else {
        format!("{root_prefix}::{rust_mod_prefix}::{struct_ident}")
    };
    entries.insert(full_proto_name.clone(), rust_type_path);

    let child_mod = prost_to_snake(msg_name);
    let next_rust_mod = if rust_mod_prefix.is_empty() {
        child_mod
    } else {
        format!("{rust_mod_prefix}::{child_mod}")
    };
    for nested in &msg.nested_type {
        collect_message_bijection_entries(
            &full_proto_name,
            &next_rust_mod,
            root_prefix,
            nested,
            entries,
        );
    }
}

/// Compiles both the 738-file `golden_fds` and the 744-file `combined_fds` via `protox`.
///
/// # Arguments
///
/// * `manifest_dir` - Crate `CARGO_MANIFEST_DIR` path (`crates/<crate-name>`).
///
/// # Errors
///
/// Returns [`std::io::Error`] if discovering `.proto` files or running `protox` fails.
pub fn load_golden_and_combined_fds(
    manifest_dir: &Path,
) -> std::io::Result<(FileDescriptorSet, FileDescriptorSet)> {
    let proto_dir = manifest_dir
        .join("../../docs/reverse_engineering/protos")
        .canonicalize()
        .unwrap_or_else(|_| manifest_dir.join("../../docs/reverse_engineering/protos"));
    let ext_proto_dir = manifest_dir
        .join("../../protos")
        .canonicalize()
        .unwrap_or_else(|_| manifest_dir.join("../../protos"));

    let mut golden_proto_paths = Vec::new();
    collect_proto_files_recursively(&proto_dir, &mut golden_proto_paths)?;
    golden_proto_paths.sort();

    let mut ext_proto_paths = Vec::new();
    if ext_proto_dir.exists() {
        collect_proto_files_recursively(&ext_proto_dir, &mut ext_proto_paths)?;
        ext_proto_paths.sort();
    }

    let golden_fds =
        protox::compile(&golden_proto_paths, [&proto_dir]).map_err(std::io::Error::other)?;
    let mut all_proto_paths = golden_proto_paths;
    all_proto_paths.extend(ext_proto_paths);
    let combined_fds = protox::compile(&all_proto_paths, [&proto_dir, &ext_proto_dir])
        .map_err(std::io::Error::other)?;
    Ok((golden_fds, combined_fds))
}

/// Compiles a single topological Protobuf shard crate (`libtpu-proto-s{shard_index}`).
///
/// # Arguments
///
/// * `shard_index` - Zero-based shard index in `0..NUM_PROTO_SHARDS`.
///
/// # Errors
///
/// Returns [`std::io::Error`] if `protox`, `prost-build`, or `OUT_DIR` file emission fails.
pub fn build_proto_shard(shard_index: usize) -> std::io::Result<()> {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default());
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap_or_default());
    clean_stale_generated_rs(&out_dir)?;

    let (_golden_fds, combined_fds) = load_golden_and_combined_fds(&manifest_dir)?;
    let pkg_to_shard = compute_package_shards(&combined_fds);

    let mut extern_map: BTreeMap<String, String> = BTreeMap::new();
    let mut shard_files = Vec::new();
    for file in &combined_fds.file {
        let pkg = file.package.as_deref().unwrap_or("");
        if is_core_local_package(pkg) {
            continue;
        }
        if pkg == "google.protobuf" {
            shard_files.push(file.clone());
            continue;
        }
        let assigned = pkg_to_shard.get(pkg).copied().unwrap_or(0);
        if assigned <= shard_index {
            shard_files.push(file.clone());
        }
        if assigned < shard_index {
            let rust_mod = package_to_rust_mod_path(pkg);
            let crate_prefix = if rust_mod.is_empty() {
                format!("::libtpu_proto_s{assigned}")
            } else {
                format!("::libtpu_proto_s{assigned}::{rust_mod}")
            };
            for enm in &file.enum_type {
                collect_extern_enum(pkg, &crate_prefix, enm, &mut extern_map);
            }
            for msg in &file.message_type {
                collect_extern_message(pkg, &crate_prefix, msg, &mut extern_map);
            }
        }
    }

    let mut config = prost_build::Config::new();
    config.btree_map(["."]);
    for eq_ty in EXTRA_EQ_TYPES {
        config.type_attribute(eq_ty, "#[derive(Eq)]");
    }
    for (fq_proto, fq_rust) in &extern_map {
        config.extern_path(fq_proto, fq_rust);
    }
    config.include_file("generated_shard_include.rs");
    config.compile_fds(FileDescriptorSet { file: shard_files })?;
    if !out_dir.join("generated_shard_include.rs").exists() {
        std::fs::write(out_dir.join("generated_shard_include.rs"), "")?;
    }
    ensure_doc_comments_in_generated_rs(&out_dir, "generated_shard_include.rs")?;

    let trait_path = if shard_index == 0 {
        "crate::ProtoSchemaId"
    } else {
        "::libtpu_proto_s0::ProtoSchemaId"
    };
    let mut entries: BTreeMap<String, String> = BTreeMap::new();
    for file in &combined_fds.file {
        let pkg = file.package.as_deref().unwrap_or("");
        if pkg == "google.protobuf" || is_core_local_package(pkg) {
            continue;
        }
        if pkg_to_shard.get(pkg).copied().unwrap_or(0) != shard_index {
            continue;
        }
        let rust_mod = package_to_rust_mod_path(pkg);
        for msg in &file.message_type {
            collect_message_bijection_entries(pkg, &rust_mod, "crate", msg, &mut entries);
        }
    }
    let mut trait_code = String::with_capacity(entries.len() * 140);
    for (full_name, rust_ty) in &entries {
        let _ = writeln!(trait_code, "impl {trait_path} for {rust_ty} {{");
        let _ = writeln!(
            trait_code,
            "    const PROTO_FULL_NAME: &'static str = {full_name:?};"
        );
        let _ = writeln!(trait_code, "}}");
    }
    std::fs::write(out_dir.join("generated_shard_traits.rs"), trait_code)?;
    Ok(())
}

/// Tree node representing a Rust module segment in the unified `generated_prost_schemas` facade.
#[derive(Default)]
struct ReexportModuleNode {
    /// Optional `(shard_index, rust_mod_path, top_level_idents)` when re-exported from a shard crate.
    leaf_reexport: Option<(usize, String, Vec<String>)>,
    /// Optional local `OUT_DIR` `.rs` filename when compiled directly inside `libtpu-core`.
    local_include_file: Option<String>,
    /// Nested child package segments.
    children: BTreeMap<String, Self>,
}

impl ReexportModuleNode {
    /// Recursively inserts a package's re-exported items into the module hierarchy.
    ///
    /// # Arguments
    ///
    /// * `segments` - Remaining `snake_case` Rust module segments.
    /// * `shard_idx` - Owning shard index `0..NUM_PROTO_SHARDS`.
    /// * `rust_mod` - Full `::`-joined Rust module path inside the shard crate.
    /// * `items` - Top-level `pub struct` / `pub enum` / `pub mod` identifiers in the package.
    fn insert(
        &mut self,
        segments: &[String],
        shard_idx: usize,
        rust_mod: String,
        items: Vec<String>,
    ) {
        if let Some((first, rest)) = segments.split_first() {
            self.children
                .entry(first.clone())
                .or_default()
                .insert(rest, shard_idx, rust_mod, items);
        } else {
            self.leaf_reexport = Some((shard_idx, rust_mod, items));
        }
    }

    /// Recursively inserts a locally compiled `OUT_DIR` `.rs` file into the module hierarchy.
    ///
    /// # Arguments
    ///
    /// * `segments` - Remaining `snake_case` Rust module segments.
    /// * `filename` - Generated `.rs` filename in `libtpu-core` `OUT_DIR`.
    fn insert_local(&mut self, segments: &[String], filename: String) {
        if let Some((first, rest)) = segments.split_first() {
            self.children
                .entry(first.clone())
                .or_default()
                .insert_local(rest, filename);
        } else {
            self.local_include_file = Some(filename);
        }
    }

    /// Recursively emits `pub mod` and `pub use` / `include!` statements into `out`.
    ///
    /// # Arguments
    ///
    /// * `out` - Destination Rust source buffer.
    /// * `depth` - Current indentation depth.
    fn emit(&self, out: &mut String, depth: usize) {
        let indent = "    ".repeat(depth);
        if let Some(ref fname) = self.local_include_file {
            let _ = writeln!(
                out,
                "{indent}include!(concat!(env!(\"OUT_DIR\"), \"/{fname}\"));"
            );
        }
        if let Some((shard_idx, ref rust_mod, ref items)) = self.leaf_reexport
            && !items.is_empty()
        {
            let joined = items.join(", ");
            if rust_mod.is_empty() {
                let _ = writeln!(
                    out,
                    "{indent}pub use ::libtpu_proto_s{shard_idx}::{{{joined}}};"
                );
            } else {
                let _ = writeln!(
                    out,
                    "{indent}pub use ::libtpu_proto_s{shard_idx}::{rust_mod}::{{{joined}}};"
                );
            }
        }
        for (seg, child) in &self.children {
            let _ = writeln!(out, "{indent}/// Generated protobuf package module.");
            let _ = writeln!(out, "{indent}pub mod {seg} {{");
            child.emit(out, depth + 1);
            let _ = writeln!(out, "{indent}}}");
        }
    }
}

/// Extracts column-0 `pub struct`, `pub enum`, and `pub mod` identifiers from a `prost-build`
/// package `.rs` file.
///
/// # Arguments
///
/// * `rs_path` - Path to the generated `<pkg>.rs` file in `OUT_DIR`.
///
/// # Errors
///
/// Returns [`std::io::Error`] if reading `rs_path` fails.
fn extract_top_level_prost_idents(rs_path: &Path) -> std::io::Result<Vec<String>> {
    if !rs_path.exists() {
        return Ok(Vec::new());
    }
    let content = std::fs::read_to_string(rs_path)?;
    let mut idents: BTreeSet<String> = BTreeSet::new();
    for line in content.lines() {
        for prefix in ["pub struct ", "pub enum ", "pub mod "] {
            if let Some(rest) = line.strip_prefix(prefix) {
                let ident: String = rest
                    .chars()
                    .take_while(|&c| c.is_ascii_alphanumeric() || c == '_' || c == '#')
                    .collect();
                if !ident.is_empty() {
                    idents.insert(ident);
                }
            }
        }
    }
    Ok(idents.into_iter().collect())
}

/// Emits `generated_proto_include.rs` in `libtpu-core` re-exporting all 6 pre-compiled shard
/// crates into the canonical `generated_prost_schemas` module tree.
///
/// # Arguments
///
/// * `fds` - Combined `FileDescriptorSet` covering all 744 `.proto` files.
/// * `pkg_to_shard` - Deterministic package-to-shard map.
/// * `out_dir` - `libtpu-core` `OUT_DIR` containing generated `.rs` package files.
///
/// # Errors
///
/// Returns [`std::io::Error`] if reading package `.rs` files or writing `generated_proto_include.rs` fails.
pub fn write_core_reexport_include(
    fds: &FileDescriptorSet,
    pkg_to_shard: &BTreeMap<String, usize>,
    out_dir: &Path,
) -> std::io::Result<()> {
    let mut root = ReexportModuleNode::default();
    let mut seen_pkgs: BTreeSet<String> = BTreeSet::new();
    for file in &fds.file {
        let pkg = file.package.as_deref().unwrap_or("");
        if pkg == "google.protobuf" || !seen_pkgs.insert(String::from(pkg)) {
            continue;
        }
        let segments: Vec<String> = if pkg.is_empty() {
            Vec::new()
        } else {
            pkg.split('.').map(prost_to_snake).collect()
        };
        if is_core_local_package(pkg) {
            root.insert_local(&segments, package_to_prost_filename(pkg));
            continue;
        }
        let shard_idx = pkg_to_shard.get(pkg).copied().unwrap_or(0);
        let rs_file = out_dir.join(package_to_prost_filename(pkg));
        let items = extract_top_level_prost_idents(&rs_file)?;
        if items.is_empty() {
            continue;
        }
        let rust_mod = package_to_rust_mod_path(pkg);
        root.insert(&segments, shard_idx, rust_mod, items);
    }

    let mut out = String::with_capacity(16_384);
    root.emit(&mut out, 0);
    std::fs::write(out_dir.join("generated_proto_include.rs"), out)
}

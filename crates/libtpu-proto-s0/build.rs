//! Build script compiling Protobuf topological shard 0 (`libtpu-proto-s0`).

/// Shared topological Protobuf shard planner and `prost-build` driver.
#[path = "../proto_codegen_shared.rs"]
pub mod proto_codegen;

/// Entrypoint for `libtpu-proto-s0` build script execution.
///
/// # Errors
///
/// Returns [`std::io::Error`] if `protox` or `prost-build` fails to compile shard 0.
fn main() -> std::io::Result<()> {
    proto_codegen::build_proto_shard(0)
}

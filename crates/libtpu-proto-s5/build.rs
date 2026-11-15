//! Build script compiling Protobuf topological shard 5 (`libtpu-proto-s5`).

/// Shared topological Protobuf shard planner and `prost-build` driver.
#[path = "../proto_codegen_shared.rs"]
pub mod proto_codegen;

/// Entrypoint for `libtpu-proto-s5` build script execution.
///
/// # Errors
///
/// Returns [`std::io::Error`] if `protox` or `prost-build` fails to compile shard 5.
fn main() -> std::io::Result<()> {
    proto_codegen::build_proto_shard(5)
}

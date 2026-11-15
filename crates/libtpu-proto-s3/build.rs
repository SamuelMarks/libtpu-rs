//! Build script compiling Protobuf topological shard 3 (`libtpu-proto-s3`).

/// Shared topological Protobuf shard planner and `prost-build` driver.
#[path = "../proto_codegen_shared.rs"]
pub mod proto_codegen;

/// Entrypoint for `libtpu-proto-s3` build script execution.
///
/// # Errors
///
/// Returns [`std::io::Error`] if `protox` or `prost-build` fails to compile shard 3.
fn main() -> std::io::Result<()> {
    proto_codegen::build_proto_shard(3)
}

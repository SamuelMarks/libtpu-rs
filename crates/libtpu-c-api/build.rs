//! Build script for `libtpu-c-api` configuring drop-in dynamic linker metadata
//! (`DT_SONAME = libtpu.so` on Linux/ELF targets, `LC_ID_DYLIB = @rpath/libtpu.dylib` on macOS).

/// Entrypoint for `cargo` build script execution.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cbindgen.toml");
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "linux" || target_os == "android" {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-soname,libtpu.so");
    } else if target_os == "macos" {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-install_name,@rpath/libtpu.dylib");
    }
}

//! Build script configuring macOS dynamic symbol lookup for `cdylib` `PyO3` extension targets.

/// Emits `-Wl,-undefined,dynamic_lookup` for macOS `cdylib` targets so `abi3` extensions
/// resolve `CPython` symbols dynamically at import time.
fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_OS");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-undefined,dynamic_lookup");
    }
}

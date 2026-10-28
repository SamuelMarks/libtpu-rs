//! `#[repr(C)]` PJRT C API structures matching `libtpu.so` (`PJRT_Api` v0.116, 1,144 bytes).
//!
//! Defines the exact binary layout of `PJRT_Api` (with all 138 function pointer slots named
//! explicitly), `PJRT_Api_Version`, `PJRT_Extension_Base`, opaque PJRT handles, and all 138
//! `#[repr(C)]` `PJRT_*_Args` structures with compile-time byte size assertions.

/// `#[repr(C)]` `PJRT_Api` vtable (`v0.116`, 1,144 bytes), `PJRT_Api_Version`, and `PJRT_Extension_Base`.
pub mod api;
/// `#[repr(C)]` argument structures for buffer, stream, and async transfer entrypoints.
pub mod buffer_args;
/// `#[repr(C)]` argument structures for `PJRT_Client_*` and `PJRT_Compile` entrypoints.
pub mod client_args;
/// `#[repr(C)]` argument structures for device, memory, and topology entrypoints.
pub mod device_args;
/// `#[repr(C)]` argument structures for error, plugin, event, and context entrypoints.
pub mod error_args;
/// `#[repr(C)]` argument structures for `PJRT_Executable_*` and `PJRT_LoadedExecutable_*` entrypoints.
pub mod executable_args;
/// Opaque PJRT C API handles, common `#[repr(C)]` descriptors, and the [`PjrtArgsStruct`] trait.
pub mod handles;

pub use api::*;
pub use buffer_args::*;
pub use client_args::*;
pub use device_args::*;
pub use error_args::*;
pub use executable_args::*;
pub use handles::*;

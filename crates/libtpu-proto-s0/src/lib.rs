//! Topological Protobuf schema shard 0 (`libtpu-proto-s0`) and [`ProtoSchemaId`] trait.

#![allow(
    clippy::nursery,
    clippy::cargo,
    clippy::doc_markdown,
    clippy::struct_excessive_bools,
    clippy::too_many_lines,
    clippy::large_enum_variant,
    clippy::enum_variant_names,
    clippy::module_inception,
    clippy::similar_names,
    clippy::derive_partial_eq_without_eq,
    clippy::must_use_candidate,
    clippy::unreadable_literal,
    unreachable_pub,
    unused_qualifications,
    variant_size_differences
)]

/// Associates a `prost`-generated Rust message struct with its canonical `.proto` full name.
pub trait ProtoSchemaId: ::prost::Message + Default + Clone + PartialEq {
    /// Canonical fully-qualified `.proto` message name (e.g., `"xla.HloModuleProto"`).
    const PROTO_FULL_NAME: &'static str;
}

include!(concat!(env!("OUT_DIR"), "/generated_shard_include.rs"));
include!(concat!(env!("OUT_DIR"), "/generated_shard_traits.rs"));

//! Topological Protobuf schema shard 1 (`libtpu-proto-s1`).

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

use libtpu_proto_s0 as _;

include!(concat!(env!("OUT_DIR"), "/generated_shard_include.rs"));
include!(concat!(env!("OUT_DIR"), "/generated_shard_traits.rs"));

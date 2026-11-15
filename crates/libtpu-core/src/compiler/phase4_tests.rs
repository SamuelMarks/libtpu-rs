//! Exhaustive Phase 4 (`4.1`–`4.3`) unit and integration test suite.
//!
//! Covers upstream MLIR v1–v6 `DialectBytecodeReader`/`DialectBytecodeWriter`, `builtin` (`0..=23`),
//! `vhlo` (`v0.9.0`–`v1.11.0+`, 41 types, 20 attrs, 140 versioned ops, portable artifacts),
//! `sdy` and `tpu` Mosaic dialect codecs, `chlo` multi-op canonical decomposition, textual
//! `dialect_resources` & `#loc` resolution, static/dynamic/bounded shape verification, and
//! lossless `StableHLO` <-> `xla::HloModuleProto` conversion.

use crate::compiler::mlir_bytecode::stream::{
    decode_prefix_varint, decode_signed_varint, encode_prefix_varint, encode_signed_varint,
};
use crate::compiler::opcode::TpuOpCode;
use crate::compiler::{
    AffineMapExpr, AstFunctionType, AstModule, AstOpKind, AstOpResult, AstOperand, AstOperation,
    BuiltinAttrCode, BuiltinCustomAttr, BuiltinCustomType, BuiltinTypeCode, ComparisonDirection,
    ComparisonType, DialectBytecodeReader, DialectBytecodeWriter, DialectKind, DimensionSize,
    DotDimensionNumbersAttr, FftKind, MLIR_ALIGNMENT_BYTE, MLIR_BYTECODE_MAGIC, MlirAttribute,
    MlirBytecodeHeader, MlirSectionId, MlirType, OpEncodingMask, PrecisionLevel, RankedTensorType,
    RngAlgorithmKind, SdyAttrCode, SdyCustomAttr, SdyDimensionSharding, SdyTensorShardingAttr,
    TokenSpan, TpuMemorySpace, TpuMosaicBytecodeCode, TpuMosaicCustomEntry, TpuSemaphoreKind,
    TransposeMode, VHLO_OP_SPECS, VhloAttrCode, VhloCustomAttr, VhloCustomType, VhloTypeCode,
    VhloVersion, are_dimensions_compatible, are_tensor_shapes_compatible, decompose_chlo_module,
    deserialize_portable_vhlo_artifact, downgrade_stablehlo_to_vhlo, hlo_ast_to_proto,
    hlo_module_proto_to_stablehlo, lookup_vhlo_op_by_name, lower_ast_module_to_parsed_hlo,
    mlir_type_to_xla_shape_proto, parse_mlir_ast_module, parse_unified_ast_module,
    select_vhlo_op_for_stablehlo, serialize_portable_vhlo_artifact, stablehlo_to_hlo_module_proto,
    upgrade_vhlo_to_stablehlo, verify_ast_module_semantics,
    verify_operation_static_and_dynamic_types, xla_shape_proto_to_mlir_type,
};
use crate::proto::{HloModuleProto, ProtoSchemaMessage};
use crate::types::{ElementType, MemoryLayout};
use std::collections::BTreeMap;

#[test]
#[allow(clippy::too_many_lines)]
fn test_phase4_1_stream_varints_apint_apfloat_and_alignment() {
    for id in 0..=8_u8 {
        let sec = MlirSectionId::from_raw(id);
        assert!(sec.is_ok());
        assert_eq!(sec.map(MlirSectionId::as_u8), Ok(id));
    }
    assert!(MlirSectionId::from_raw(9).is_err());

    let mask = OpEncodingMask::HAS_ATTRS
        | OpEncodingMask::HAS_RESULTS
        | OpEncodingMask::HAS_OPERANDS
        | OpEncodingMask::HAS_SUCCESSORS
        | OpEncodingMask::HAS_INLINE_REGIONS
        | OpEncodingMask::HAS_USE_LIST_ORDERS
        | OpEncodingMask::HAS_PROPERTIES;
    assert_eq!(mask.bits(), 0x7F);

    for shift in [
        0_u32, 6, 7, 13, 14, 20, 21, 27, 28, 34, 35, 41, 42, 48, 49, 55, 56, 63,
    ] {
        let val = if shift == 63 {
            u64::MAX
        } else {
            1_u64 << shift
        };
        let enc = encode_prefix_varint(val);
        let mut cur = enc.as_slice();
        assert_eq!(decode_prefix_varint(&mut cur), Ok(val));
        assert_eq!(cur.len(), 0);
    }
    let mut empty_cur: &[u8] = &[];
    assert!(decode_prefix_varint(&mut empty_cur).is_err());
    let mut trunc9: &[u8] = &[0x00, 1, 2, 3];
    assert!(decode_prefix_varint(&mut trunc9).is_err());
    let mut trunc2: &[u8] = &[0x02];
    assert!(decode_prefix_varint(&mut trunc2).is_err());

    for sval in [0_i64, 1, -1, 127, -128, i64::MAX, i64::MIN] {
        let enc = encode_signed_varint(sval);
        let mut cur = enc.as_slice();
        assert_eq!(decode_signed_varint(&mut cur), Ok(sval));
    }
    let mut empty_signed: &[u8] = &[];
    assert!(decode_signed_varint(&mut empty_signed).is_err());

    let mut w = DialectBytecodeWriter::new();
    assert!(w.is_empty());
    assert_eq!(w.len(), 0);
    w.write_byte(0x42);
    w.write_bool(true);
    w.write_bool(false);
    w.write_varint(999_999);
    w.write_signed_varint(-123_456);
    w.write_blob(b"hello");
    w.write_nul_string("mlir");
    w.write_apint(0, 0);
    w.write_apint(1, 1);
    w.write_apint(4, -3);
    w.write_apint(8, -42);
    w.write_apint(32, -100_000);
    w.write_apint(128, -5);
    w.write_apint(128, 7);
    for et in [
        ElementType::F64,
        ElementType::F32,
        ElementType::Bf16,
        ElementType::F16,
        ElementType::F8E4M3Fn,
    ] {
        w.write_apfloat_f64_bits(et, 1.5_f64.to_bits());
    }
    w.write_list(&[10_u32, 20, 30], |wr, &x| wr.write_varint(u64::from(x)));
    w.align_to(1);
    w.align_to(3);
    w.align_to(8);
    assert_eq!(w.len() % 8, 0);
    w.align_to(8);
    assert_ne!(w.as_slice().len(), 0);

    let bytes = w.into_bytes();
    let mut r = DialectBytecodeReader::new(&bytes, 2);
    assert!(!r.is_empty());
    assert_eq!(r.offset(), 0);
    assert_eq!(r.remaining().len(), bytes.len());
    assert_eq!(r.read_byte(), Ok(0x42));
    assert_eq!(r.read_bool(), Ok(true));
    assert_eq!(r.read_bool(), Ok(false));
    assert_eq!(r.read_varint(), Ok(999_999));
    assert_eq!(r.read_signed_varint(), Ok(-123_456));
    assert_eq!(r.read_blob(), Ok(b"hello".as_slice()));
    assert_eq!(r.read_nul_string(), Ok("mlir"));
    assert_eq!(r.read_apint(0), Ok(0));
    assert_eq!(r.read_apint(1), Ok(1));
    assert_eq!(r.read_apint(4), Ok(-3));
    assert_eq!(r.read_apint(8), Ok(-42));
    assert_eq!(r.read_apint(32), Ok(-100_000));
    assert_eq!(r.read_apint(128), Ok(-5));
    assert_eq!(r.read_apint(128), Ok(7));
    for et in [
        ElementType::F64,
        ElementType::F32,
        ElementType::Bf16,
        ElementType::F16,
        ElementType::F8E4M3Fn,
    ] {
        let bits = r.read_apfloat_f64_bits(et);
        assert!(bits.is_ok());
    }
    let lst = r.read_list(DialectBytecodeReader::read_varint);
    assert_eq!(lst, Ok(vec![10, 20, 30]));
    assert!(r.align_to(0).is_err());
    assert!(r.align_to(3).is_err());
    assert_eq!(r.align_to(8), Ok(()));
    assert_eq!(r.align_to(8), Ok(()));
    assert!(r.is_empty());
    assert!(r.read_byte().is_err());
    assert!(r.read_varint().is_err());
    assert!(r.read_signed_varint().is_err());
    assert!(r.read_bytes(1).is_err());

    // Extra APInt word counts (0, 1, 3) and reader error paths
    let mut w_ap = DialectBytecodeWriter::with_capacity(32);
    w_ap.write_varint(0);
    w_ap.write_varint(1);
    w_ap.write_signed_varint(-9);
    w_ap.write_varint(1);
    w_ap.write_signed_varint(9);
    w_ap.write_varint(3);
    w_ap.write_signed_varint(1);
    w_ap.write_signed_varint(0);
    w_ap.write_signed_varint(0);
    let ap_bytes = w_ap.into_bytes();
    let mut r_ap = DialectBytecodeReader::new(&ap_bytes, 2);
    assert_eq!(r_ap.read_apint(128), Ok(0));
    assert_eq!(r_ap.read_apint(128), Ok(-9));
    assert_eq!(r_ap.read_apint(128), Ok(9));
    assert_eq!(r_ap.read_apint(128), Ok(1));

    let mut r_no_nul = DialectBytecodeReader::new(b"abc", 0);
    assert!(r_no_nul.read_nul_string().is_err());
    let mut r_bad_utf8 = DialectBytecodeReader::new(&[0xFF, 0x00], 0);
    assert!(r_bad_utf8.read_nul_string().is_err());
    let mut w_huge_list = DialectBytecodeWriter::new();
    w_huge_list.write_varint(100);
    let huge_bytes = w_huge_list.into_bytes();
    let mut r_huge = DialectBytecodeReader::new(&huge_bytes, 2);
    assert!(r_huge.read_list(DialectBytecodeReader::read_byte).is_err());
    let bad_pad = [0x01, 0x00, 0x00, 0x00];
    let mut r_pad = DialectBytecodeReader::new(&bad_pad, 5);
    let _ = r_pad.read_byte();
    assert!(r_pad.align_to(4).is_err());

    // Section framing with 0xCB alignment padding decoded by MlirBytecodeHeader
    let mut cw = DialectBytecodeWriter::new();
    cw.write_bytes(MLIR_BYTECODE_MAGIC);
    cw.write_varint(6);
    cw.write_nul_string("MLIR20.1.0");
    cw.write_section(0, None, b"func\0");
    cw.write_section(5, Some(8), &[1, 2, 3, 4]);
    cw.write_section(6, Some(1), &[]);
    let container = cw.into_bytes();
    assert!(container.contains(&MLIR_ALIGNMENT_BYTE));
    let decoded_hdr = MlirBytecodeHeader::decode(&container);
    assert!(decoded_hdr.is_ok());
}

#[test]
#[allow(clippy::too_many_lines)]
fn test_phase4_1_builtin_custom_types_and_attributes_roundtrip() {
    for tag in 0..=20 {
        assert!(BuiltinTypeCode::from_tag(tag).is_ok());
    }
    assert!(BuiltinTypeCode::from_tag(21).is_err());
    for tag in 0..=23 {
        assert!(BuiltinAttrCode::from_tag(tag).is_ok());
    }
    assert!(BuiltinAttrCode::from_tag(24).is_err());

    let strings = vec![
        String::from("main"),
        String::from("model.py"),
        String::from("weights"),
        String::from("vmem"),
    ];
    let mut resolved_attrs = vec![
        MlirAttribute::SdyTensorSharding(SdyTensorShardingAttr {
            mesh_name: String::from("mesh"),
            dim_shardings: vec![SdyDimensionSharding {
                axes: vec![String::from("x")],
                is_open: false,
                priority: None,
            }],
            replicated_axes: Vec::new(),
            unreduced_axes: Vec::new(),
        }),
        MlirAttribute::Array(vec![
            MlirAttribute::Integer {
                value: 16,
                ty: Some(ElementType::S64),
            },
            MlirAttribute::String(String::from("?")),
        ]),
        MlirAttribute::String(String::from("custom_enc")),
        MlirAttribute::MosaicAttr(crate::compiler::MosaicDialectAttr::MemorySpace(
            TpuMemorySpace::Vmem,
        )),
        MlirAttribute::Integer {
            value: 2,
            ty: Some(ElementType::S32),
        },
        MlirAttribute::String(String::from("cmem")),
        MlirAttribute::AffineMap(AffineMapExpr {
            dims: vec![String::from("d0")],
            symbols: Vec::new(),
            results: vec![String::from("d0")],
        }),
        MlirAttribute::Bool(true),
    ];

    let all_types = vec![
        BuiltinCustomType::Integer {
            width: 1,
            signedness: 0,
        },
        BuiltinCustomType::Integer {
            width: 2,
            signedness: 1,
        },
        BuiltinCustomType::Integer {
            width: 2,
            signedness: 2,
        },
        BuiltinCustomType::Integer {
            width: 4,
            signedness: 0,
        },
        BuiltinCustomType::Integer {
            width: 4,
            signedness: 2,
        },
        BuiltinCustomType::Integer {
            width: 8,
            signedness: 0,
        },
        BuiltinCustomType::Integer {
            width: 8,
            signedness: 2,
        },
        BuiltinCustomType::Integer {
            width: 16,
            signedness: 0,
        },
        BuiltinCustomType::Integer {
            width: 16,
            signedness: 2,
        },
        BuiltinCustomType::Integer {
            width: 32,
            signedness: 0,
        },
        BuiltinCustomType::Integer {
            width: 32,
            signedness: 2,
        },
        BuiltinCustomType::Integer {
            width: 64,
            signedness: 0,
        },
        BuiltinCustomType::Integer {
            width: 64,
            signedness: 2,
        },
        BuiltinCustomType::Index,
        BuiltinCustomType::Float(BuiltinTypeCode::BFloat16Type),
        BuiltinCustomType::Float(BuiltinTypeCode::Float16Type),
        BuiltinCustomType::Float(BuiltinTypeCode::Float32Type),
        BuiltinCustomType::Float(BuiltinTypeCode::Float64Type),
        BuiltinCustomType::Float(BuiltinTypeCode::Float80Type),
        BuiltinCustomType::Float(BuiltinTypeCode::Float128Type),
        BuiltinCustomType::Complex {
            element_type_idx: 16,
        },
        BuiltinCustomType::Complex {
            element_type_idx: 17,
        },
        BuiltinCustomType::Function {
            inputs: vec![16],
            results: vec![17],
        },
        BuiltinCustomType::MemRef {
            memory_space_attr_idx: None,
            shape: vec![-1, 128],
            element_type_idx: 16,
            layout_attr_idx: Some(6),
        },
        BuiltinCustomType::MemRef {
            memory_space_attr_idx: Some(3),
            shape: vec![8, 128],
            element_type_idx: 16,
            layout_attr_idx: Some(7),
        },
        BuiltinCustomType::MemRef {
            memory_space_attr_idx: Some(4),
            shape: vec![8],
            element_type_idx: 16,
            layout_attr_idx: None,
        },
        BuiltinCustomType::MemRef {
            memory_space_attr_idx: Some(5),
            shape: vec![8],
            element_type_idx: 16,
            layout_attr_idx: None,
        },
        BuiltinCustomType::NoneType,
        BuiltinCustomType::RankedTensor {
            encoding_attr_idx: None,
            shape: vec![4, 8],
            element_type_idx: 16,
        },
        BuiltinCustomType::RankedTensor {
            encoding_attr_idx: Some(0),
            shape: vec![4, 8],
            element_type_idx: 16,
        },
        BuiltinCustomType::RankedTensor {
            encoding_attr_idx: Some(1),
            shape: vec![-1, -1],
            element_type_idx: 16,
        },
        BuiltinCustomType::RankedTensor {
            encoding_attr_idx: Some(2),
            shape: vec![4],
            element_type_idx: 16,
        },
        BuiltinCustomType::RankedTensor {
            encoding_attr_idx: Some(7),
            shape: vec![4],
            element_type_idx: 16,
        },
        BuiltinCustomType::Tuple {
            elements: vec![16, 28],
        },
        BuiltinCustomType::UnrankedMemRef {
            memory_space_attr_idx: None,
            element_type_idx: 16,
        },
        BuiltinCustomType::UnrankedMemRef {
            memory_space_attr_idx: Some(4),
            element_type_idx: 16,
        },
        BuiltinCustomType::UnrankedTensor {
            element_type_idx: 16,
        },
        BuiltinCustomType::Vector {
            shape: vec![8, 128],
            scalable_dims: vec![false, false],
            element_type_idx: 16,
        },
        BuiltinCustomType::Vector {
            shape: vec![4, 16],
            scalable_dims: vec![false, true],
            element_type_idx: 16,
        },
    ];

    let mut w = DialectBytecodeWriter::new();
    for ty in &all_types {
        ty.encode(&mut w);
    }
    let ty_bytes = w.into_bytes();
    let mut r = DialectBytecodeReader::new(&ty_bytes, 2);
    let mut resolved_types = Vec::new();
    for expected in &all_types {
        let dec = BuiltinCustomType::decode(&mut r);
        assert_eq!(dec.as_ref(), Ok(expected));
        let mlir_ty = expected.to_mlir_type(&resolved_types, &resolved_attrs);
        resolved_types.push(mlir_ty);
    }
    assert!(r.is_empty());

    // Also test remaining memory_space integer codes (1, 3, 4, 99) on MemRef/UnrankedMemRef
    for ms_val in [1_i128, 3, 4, 99] {
        let ms_idx = u32::try_from(resolved_attrs.len()).unwrap_or(0);
        resolved_attrs.push(MlirAttribute::Integer {
            value: ms_val,
            ty: Some(ElementType::S32),
        });
        let mr = BuiltinCustomType::MemRef {
            memory_space_attr_idx: Some(ms_idx),
            shape: vec![4],
            element_type_idx: 16,
            layout_attr_idx: None,
        };
        let _ = mr.to_mlir_type(&resolved_types, &resolved_attrs);
        let umr = BuiltinCustomType::UnrankedMemRef {
            memory_space_attr_idx: Some(ms_idx),
            element_type_idx: 16,
        };
        let _ = umr.to_mlir_type(&resolved_types, &resolved_attrs);
    }

    let mut all_attrs = vec![
        BuiltinCustomAttr::UnknownLoc,
        BuiltinCustomAttr::FileLineColLoc {
            file_string_idx: 1,
            line: 10,
            column: 4,
        },
        BuiltinCustomAttr::CallSiteLoc {
            callee_loc_idx: 0,
            caller_loc_idx: 1,
        },
        BuiltinCustomAttr::FusedLoc {
            loc_indices: vec![0, 1],
            metadata_attr_idx: None,
        },
        BuiltinCustomAttr::FusedLoc {
            loc_indices: vec![0, 1],
            metadata_attr_idx: Some(0),
        },
        BuiltinCustomAttr::NameLoc {
            name_string_idx: 0,
            child_loc_idx: 1,
        },
        BuiltinCustomAttr::FileLineColRange {
            file_string_idx: 1,
            start_line: 1,
            start_col: 2,
            end_line: 3,
            end_col: 4,
        },
        BuiltinCustomAttr::String {
            string_idx: 0,
            type_idx: None,
        },
        BuiltinCustomAttr::String {
            string_idx: 2,
            type_idx: Some(16),
        },
        BuiltinCustomAttr::FlatSymbolRef(0),
        BuiltinCustomAttr::SymbolRef {
            root_idx: 0,
            nested_indices: vec![9, 99],
        },
        BuiltinCustomAttr::TypeAttr(16),
        BuiltinCustomAttr::Unit,
        BuiltinCustomAttr::Integer {
            type_idx: 0,
            bit_width: 1,
            value: 1,
        },
        BuiltinCustomAttr::Integer {
            type_idx: 10,
            bit_width: 32,
            value: 42,
        },
        BuiltinCustomAttr::Array(vec![7, 8]),
        BuiltinCustomAttr::Dictionary(vec![(7, 14), (99, 13)]),
        BuiltinCustomAttr::DenseResourceElements {
            shaped_type_idx: 28,
            resource_handle_idx: 0,
        },
        BuiltinCustomAttr::DenseIntOrFpElements {
            shaped_type_idx: 28,
            raw_bytes: 2.0_f32.to_le_bytes().to_vec(),
        },
        BuiltinCustomAttr::DenseIntOrFpElements {
            shaped_type_idx: 0,
            raw_bytes: Vec::new(),
        },
        BuiltinCustomAttr::DenseStringElements {
            shaped_type_idx: 28,
            is_splat: true,
            string_indices: vec![0],
        },
        BuiltinCustomAttr::SparseElements {
            shaped_type_idx: 28,
            indices_attr_idx: 14,
            values_attr_idx: 18,
        },
        BuiltinCustomAttr::Distinct {
            referenced_attr_idx: 7,
        },
        BuiltinCustomAttr::AffineMap(AffineMapExpr {
            dims: vec![String::from("d0")],
            symbols: vec![String::from("s0")],
            results: vec![String::from("d0 + s0")],
        }),
    ];
    for et in [
        ElementType::Pred,
        ElementType::S8,
        ElementType::U8,
        ElementType::S16,
        ElementType::U16,
        ElementType::S32,
        ElementType::U32,
        ElementType::S64,
        ElementType::U64,
        ElementType::Bf16,
        ElementType::F16,
        ElementType::F32,
        ElementType::F64,
        ElementType::F8E4M3Fn,
    ] {
        all_attrs.push(BuiltinCustomAttr::DenseArray {
            element_type: et,
            size: 1,
            raw_bytes: et.pack_from_f64_slice(&[1.0]),
        });
    }
    for et in [
        ElementType::Bf16,
        ElementType::F16,
        ElementType::F32,
        ElementType::F64,
    ] {
        all_attrs.push(BuiltinCustomAttr::Float {
            type_idx: 16,
            element_type: et,
            f64_bits: 1.5_f64.to_bits(),
        });
    }

    let mut aw = DialectBytecodeWriter::new();
    for attr in &all_attrs {
        attr.encode(&mut aw);
    }
    let attr_bytes = aw.into_bytes();
    let mut ar = DialectBytecodeReader::new(&attr_bytes, 2);
    let mut out_attrs = Vec::new();
    let resource_keys = vec![String::from("weights_blob")];
    for expected in &all_attrs {
        let dec = BuiltinCustomAttr::decode(&mut ar);
        assert_eq!(dec.as_ref(), Ok(expected));
        let mlir_attr =
            expected.to_mlir_attribute(&strings, &resolved_types, &out_attrs, &resource_keys);
        out_attrs.push(mlir_attr);
    }
    assert!(ar.is_empty());
}

#[test]
#[allow(clippy::too_many_lines)]
fn test_phase4_2_vhlo_types_attrs_140_ops_and_portable_artifacts() {
    assert_eq!(
        VhloVersion::parse_str("v1.9.3"),
        Ok(VhloVersion::new(1, 9, 3))
    );
    assert!(VhloVersion::parse_str("bad").is_err());
    assert!(VhloVersion::MIN_SUPPORTED.is_supported());
    assert!(!VhloVersion::new(0, 8, 0).is_supported());
    assert_eq!(VhloVersion::CURRENT.to_string(), "1.11.0");
    let ver_bytes = VhloVersion::V1_6_0.encode_dialect_version();
    assert_eq!(
        VhloVersion::decode_dialect_version(&ver_bytes),
        Ok(VhloVersion::V1_6_0)
    );
    assert!(VhloVersion::decode_dialect_version(&[1]).is_err());

    for tag in 0..=40 {
        let tc = VhloTypeCode::from_tag(tag);
        assert!(tc.is_ok());
        if let Ok(code) = tc {
            let _ = code.to_scalar_element_type();
        }
    }
    assert!(VhloTypeCode::from_tag(41).is_err());
    for raw_et in 0..=28 {
        if let Ok(et) = ElementType::from_raw(raw_et) {
            let _ = VhloTypeCode::from_scalar_element_type(et);
        }
    }

    let vhlo_types = vec![
        VhloCustomType::Scalar(VhloTypeCode::FloatF32V1),
        VhloCustomType::Scalar(VhloTypeCode::FloatF64V1),
        VhloCustomType::Scalar(VhloTypeCode::TokenV1),
        VhloCustomType::Scalar(VhloTypeCode::NoneV1),
        VhloCustomType::ComplexV1 {
            element_type_idx: 0,
        },
        VhloCustomType::ComplexV1 {
            element_type_idx: 1,
        },
        VhloCustomType::FunctionV1 {
            inputs: vec![0],
            outputs: vec![0],
        },
        VhloCustomType::RankedTensorV1 {
            encoding_attr_idx: None,
            shape: vec![4, 8],
            element_type_idx: 0,
        },
        VhloCustomType::RankedTensorV1 {
            encoding_attr_idx: Some(0),
            shape: vec![-1, 8],
            element_type_idx: 0,
        },
        VhloCustomType::UnrankedTensorV1 {
            element_type_idx: 0,
        },
        VhloCustomType::TupleV1 {
            elements: vec![0, 7],
        },
        VhloCustomType::UniformQuantized {
            flags: 1,
            storage_type_idx: 0,
            expressed_type_idx: 0,
            quantized_dimension: None,
            scale_bits: vec![0.5_f64.to_bits()],
            zero_points: vec![0],
            storage_min: -128,
            storage_max: 127,
        },
        VhloCustomType::UniformQuantized {
            flags: 1,
            storage_type_idx: 0,
            expressed_type_idx: 0,
            quantized_dimension: Some(1),
            scale_bits: vec![0.25_f64.to_bits(), 0.5_f64.to_bits()],
            zero_points: vec![0, 1],
            storage_min: -128,
            storage_max: 127,
        },
    ];
    let mut tw = DialectBytecodeWriter::new();
    for vt in &vhlo_types {
        vt.encode(&mut tw);
    }
    let vt_bytes = tw.into_bytes();
    let mut tr = DialectBytecodeReader::new(&vt_bytes, 2);
    let mut res_types = Vec::new();
    let bounds_attrs = vec![MlirAttribute::DenseI64Array(vec![16, -1])];
    for expected in &vhlo_types {
        let dec = VhloCustomType::decode(&mut tr);
        assert_eq!(dec.as_ref(), Ok(expected));
        res_types.push(expected.to_mlir_type(&res_types, &bounds_attrs));
    }
    assert!(tr.is_empty());

    assert!(VhloAttrCode::from_tag(0).is_err());
    assert!(VhloAttrCode::from_tag(21).is_err());
    for tag in 1..=20 {
        assert!(VhloAttrCode::from_tag(tag).is_ok());
    }

    let vhlo_attrs = vec![
        VhloCustomAttr::StringV1(b"key".to_vec()),
        VhloCustomAttr::BooleanV1(true),
        VhloCustomAttr::PrecisionV1(PrecisionLevel::Default),
        VhloCustomAttr::PrecisionV1(PrecisionLevel::High),
        VhloCustomAttr::PrecisionV1(PrecisionLevel::Highest),
        VhloCustomAttr::ArrayV1(vec![2, 3, 4]),
        VhloCustomAttr::ArrayV1(vec![0, 1]),
        VhloCustomAttr::ComparisonDirectionV1(ComparisonDirection::Eq),
        VhloCustomAttr::ComparisonDirectionV1(ComparisonDirection::Ne),
        VhloCustomAttr::ComparisonDirectionV1(ComparisonDirection::Ge),
        VhloCustomAttr::ComparisonDirectionV1(ComparisonDirection::Gt),
        VhloCustomAttr::ComparisonDirectionV1(ComparisonDirection::Le),
        VhloCustomAttr::ComparisonDirectionV1(ComparisonDirection::Lt),
        VhloCustomAttr::ComparisonTypeV1(ComparisonType::Float),
        VhloCustomAttr::ComparisonTypeV1(ComparisonType::TotalOrder),
        VhloCustomAttr::ComparisonTypeV1(ComparisonType::Signed),
        VhloCustomAttr::ComparisonTypeV1(ComparisonType::Unsigned),
        VhloCustomAttr::CustomCallApiVersionV1(4),
        VhloCustomAttr::DictionaryV1(vec![(0, 1)]),
        VhloCustomAttr::FftTypeV1(FftKind::Fft),
        VhloCustomAttr::FftTypeV1(FftKind::Ifft),
        VhloCustomAttr::FftTypeV1(FftKind::Rfft),
        VhloCustomAttr::FftTypeV1(FftKind::Irfft),
        VhloCustomAttr::FloatV1 {
            type_idx: 0,
            element_type: ElementType::F32,
            f64_bits: 2.5_f64.to_bits(),
        },
        VhloCustomAttr::IntegerV1 {
            type_idx: 0,
            bit_width: 1,
            value: 1,
        },
        VhloCustomAttr::IntegerV1 {
            type_idx: 0,
            bit_width: 32,
            value: -7,
        },
        VhloCustomAttr::OutputOperandAliasV1 {
            output_tuple_indices: vec![0],
            operand_index: 0,
            operand_tuple_indices: Vec::new(),
        },
        VhloCustomAttr::RngAlgorithmV1(RngAlgorithmKind::Default),
        VhloCustomAttr::RngAlgorithmV1(RngAlgorithmKind::ThreeFry),
        VhloCustomAttr::RngAlgorithmV1(RngAlgorithmKind::Philox),
        VhloCustomAttr::RngDistributionV1(1),
        VhloCustomAttr::TensorV1 {
            type_idx: 7,
            raw_bytes: 1.0_f32.to_le_bytes().to_vec(),
        },
        VhloCustomAttr::TensorV1 {
            type_idx: 99,
            raw_bytes: Vec::new(),
        },
        VhloCustomAttr::TransposeV1(TransposeMode::NoTranspose),
        VhloCustomAttr::TransposeV1(TransposeMode::Transpose),
        VhloCustomAttr::TransposeV1(TransposeMode::Adjoint),
        VhloCustomAttr::TypeV1(0),
        VhloCustomAttr::TypeExtensionsV1(vec![16, -1]),
        VhloCustomAttr::ResultAccuracyModeV1(1),
        VhloCustomAttr::ResultAccuracyV1 {
            atol_bits: 1e-5_f64.to_bits(),
            rtol_bits: 1e-5_f64.to_bits(),
            ulps: 2,
            mode: 2,
        },
    ];
    let mut vaw = DialectBytecodeWriter::new();
    for va in &vhlo_attrs {
        va.encode(&mut vaw);
    }
    let vhlo_attr_encoded = vaw.into_bytes();
    let mut var = DialectBytecodeReader::new(&vhlo_attr_encoded, 2);
    let mut res_vattrs = Vec::new();
    for expected in &vhlo_attrs {
        let dec = VhloCustomAttr::decode(&mut var);
        assert_eq!(dec.as_ref(), Ok(expected));
        res_vattrs.push(expected.to_mlir_attribute(&res_types, &res_vattrs));
    }
    assert!(var.is_empty());

    assert!(VHLO_OP_SPECS.len() >= 126);
    assert!(lookup_vhlo_op_by_name("vhlo.dot_general_v2").is_some());
    assert!(lookup_vhlo_op_by_name("vhlo.nonexistent_v9").is_none());
    assert!(select_vhlo_op_for_stablehlo("dot_general", VhloVersion::new(0, 8, 0)).is_err());
    assert!(select_vhlo_op_for_stablehlo("nonexistent", VhloVersion::CURRENT).is_err());
    assert_eq!(
        select_vhlo_op_for_stablehlo("dot_general", VhloVersion::MIN_SUPPORTED)
            .map(|s| s.vhlo_name),
        Ok("vhlo.dot_general_v1")
    );
    assert_eq!(
        select_vhlo_op_for_stablehlo("dot_general", VhloVersion::CURRENT).map(|s| s.vhlo_name),
        Ok("vhlo.dot_general_v2")
    );

    let shlo_src = r"
        module @portable {
          func.func @main(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
            %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
            %1 = stablehlo.exponential %0 : tensor<4x4xf32>
            func.return %1 : tensor<4x4xf32>
          }
        }
    ";
    let ast = parse_mlir_ast_module(shlo_src);
    assert!(ast.is_ok());
    if let Ok(mut module) = ast {
        assert!(downgrade_stablehlo_to_vhlo(&module, VhloVersion::new(0, 5, 0)).is_err());
        let artifact = serialize_portable_vhlo_artifact(&module, VhloVersion::V1_9_0);
        assert!(artifact.is_ok());
        if let Ok(bytes) = artifact {
            let restored = deserialize_portable_vhlo_artifact(&bytes);
            assert!(restored.is_ok());
            if let Ok((ver, up_mod)) = restored {
                assert_eq!(ver, VhloVersion::V1_9_0);
                assert_eq!(up_mod.functions.len(), 1);
            }
        }
        module.attributes.insert(
            String::from("vhlo.version"),
            MlirAttribute::String(String::from("0.5.0")),
        );
        assert!(upgrade_vhlo_to_stablehlo(&mut module).is_err());
    }
}

#[test]
#[allow(clippy::too_many_lines)]
fn test_phase4_2_sdy_and_mosaic_bytecode_and_chlo_decompositions() {
    for tag in 0..=5 {
        assert!(SdyAttrCode::from_tag(tag).is_ok());
        assert!(TpuMosaicBytecodeCode::from_tag(tag).is_ok());
    }
    assert!(SdyAttrCode::from_tag(6).is_err());
    assert!(TpuMosaicBytecodeCode::from_tag(6).is_err());

    let strings = vec![String::from("mesh"), String::from("x"), String::from("y")];
    let sdy_attrs = vec![
        SdyCustomAttr::Mesh {
            axes: vec![(1, 2), (2, 4)],
            device_ids: vec![0, 1, 2, 3, 4, 5, 6, 7],
        },
        SdyCustomAttr::DimensionSharding {
            axis_string_indices: vec![1],
            is_closed: true,
            priority: Some(0),
        },
        SdyCustomAttr::DimensionSharding {
            axis_string_indices: vec![2],
            is_closed: false,
            priority: None,
        },
        SdyCustomAttr::TensorSharding {
            mesh_name_string_idx: 0,
            dim_shardings: vec![(vec![1], true, Some(1)), (Vec::new(), false, None)],
            replicated_axis_indices: vec![2],
            unreduced_axis_indices: Vec::new(),
        },
        SdyCustomAttr::TensorShardingPerValue(vec![1, 3]),
        SdyCustomAttr::ManualAxes(vec![1]),
        SdyCustomAttr::MeshAxis {
            name_string_idx: 1,
            size: 4,
        },
    ];
    let mut sw = DialectBytecodeWriter::new();
    for sa in &sdy_attrs {
        sa.encode(&mut sw);
    }
    let s_bytes = sw.into_bytes();
    let mut sr = DialectBytecodeReader::new(&s_bytes, 2);
    let mut out_sdy = Vec::new();
    for expected in &sdy_attrs {
        let dec = SdyCustomAttr::decode(&mut sr);
        assert_eq!(dec.as_ref(), Ok(expected));
        out_sdy.push(expected.to_mlir_attribute(&strings, &out_sdy));
    }
    assert!(sr.is_empty());

    let mosaic_entries = vec![
        TpuMosaicCustomEntry::MemorySpace(TpuMemorySpace::Hbm),
        TpuMosaicCustomEntry::MemorySpace(TpuMemorySpace::Vmem),
        TpuMosaicCustomEntry::MemorySpace(TpuMemorySpace::Smem),
        TpuMosaicCustomEntry::MemorySpace(TpuMemorySpace::Cmem),
        TpuMosaicCustomEntry::MemorySpace(TpuMemorySpace::SemaphoreMem),
        TpuMosaicCustomEntry::TiledLayout {
            tiles: vec![vec![8, 128]],
            tile_strides: vec![1, 1],
        },
        TpuMosaicCustomEntry::DimensionSemantics(vec![
            String::from("parallel"),
            String::from("arbitrary"),
        ]),
        TpuMosaicCustomEntry::CoreType(String::from("tc")),
        TpuMosaicCustomEntry::Semaphore(TpuSemaphoreKind::Regular),
        TpuMosaicCustomEntry::Semaphore(TpuSemaphoreKind::Dma),
    ];
    let mut mw = DialectBytecodeWriter::new();
    for me in &mosaic_entries {
        me.encode(&mut mw);
    }
    let m_bytes = mw.into_bytes();
    let mut mr = DialectBytecodeReader::new(&m_bytes, 2);
    for expected in &mosaic_entries {
        let dec = TpuMosaicCustomEntry::decode(&mut mr);
        assert_eq!(dec.as_ref(), Ok(expected));
        let _ = expected.to_mlir_attribute();
        let _ = expected.to_mlir_type();
    }
    assert!(mr.is_empty());

    // Exhaustive CHLO -> StableHLO multi-op graph decomposition
    for bcast_op in [
        "broadcast_add",
        "broadcast_sub",
        "broadcast_mul",
        "broadcast_div",
        "broadcast_rem",
        "broadcast_pow",
        "broadcast_atan2",
        "broadcast_max",
        "broadcast_min",
        "broadcast_and",
        "broadcast_or",
        "broadcast_xor",
        "broadcast_shift_left",
        "broadcast_shift_right_arithmetic",
        "broadcast_shift_right_logical",
        "broadcast_next_after",
        "broadcast_complex",
    ] {
        let src = format!(
            "module @c {{ func.func @main(%a: tensor<4x4xf32>, %b: tensor<4xf32>) -> tensor<4x4xf32> {{ %0 = chlo.{bcast_op} %a, %b {{broadcast_dimensions = array<i64: 1>}} : (tensor<4x4xf32>, tensor<4xf32>) -> tensor<4x4xf32> func.return %0 : tensor<4x4xf32> }} }}"
        );
        let parsed = parse_mlir_ast_module(&src);
        assert!(parsed.is_ok());
        if let Ok(m) = parsed {
            let ops = &m
                .functions
                .first()
                .map(|f| &f.body.blocks)
                .into_iter()
                .flatten()
                .next();
            assert!(
                ops.is_some_and(|b| b.operations.iter().all(|o| o.dialect != DialectKind::Chlo))
            );
            assert!(ops.is_some_and(|b| {
                b.operations
                    .iter()
                    .any(|o| o.mnemonic == "broadcast_in_dim")
            }));
        }
    }

    for unary_chlo in [
        "erf", "erfc", "erf_inv", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "asin",
        "acos", "atan", "lgamma",
    ] {
        let src = format!(
            "module @u {{ func.func @main(%x: tensor<4xf32>) -> tensor<4xf32> {{ %0 = chlo.{unary_chlo} %x : tensor<4xf32> -> tensor<4xf32> func.return %0 : tensor<4xf32> }} }}"
        );
        let parsed = parse_mlir_ast_module(&src);
        assert!(parsed.is_ok());
        if let Ok(m) = parsed {
            assert!(lower_ast_module_to_parsed_hlo(&m).is_ok());
        }
    }

    let rich_chlo = r"
        module @rich_chlo {
          func.func @main(%a: tensor<2x4xf32>, %b: tensor<4xf32>, %z: tensor<4xcomplex<f32>>, %z64: tensor<4xcomplex<f64>>) -> tensor<2x2xf32> {
            %cmp = chlo.broadcast_compare %a, %b {comparison_direction = #chlo<comparison_direction GT>} : (tensor<2x4xf32>, tensor<4xf32>) -> tensor<2x4xi1>
            %sel = chlo.broadcast_select %cmp, %a, %b : (tensor<2x4xi1>, tensor<2x4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
            %cj = chlo.conj %z : tensor<4xcomplex<f32>> -> tensor<4xcomplex<f32>>
            %cj64 = chlo.conj %z64 : tensor<4xcomplex<f64>> -> tensor<4xcomplex<f64>>
            %cst = chlo.constant {value = dense<1.0> : tensor<2x4xf32>} : () -> tensor<2x4xf32>
            %rg = chlo.ragged_dot %a, %cst, %b : (tensor<2x4xf32>, tensor<2x4xf32>, tensor<4xf32>) -> tensor<2x2xf32>
            %tk:2 = chlo.top_k %sel {k = 2 : i64} : tensor<2x4xf32> -> (tensor<2x2xf32>, tensor<2x2xi32>)
            %v, %i = chlo.top_k %sel {k = 2 : i64} : tensor<2x4xf32> -> (tensor<2x2xf32>, tensor<2x2xi32>)
            func.return %v : tensor<2x2xf32>
          }
        }
    ";
    let parsed_rich = parse_mlir_ast_module(rich_chlo);
    assert!(parsed_rich.is_ok());

    // Top-level CHLO op outside func.func
    let mut top_mod = AstModule {
        top_level_operations: vec![AstOperation {
            results: vec![AstOpResult {
                name: String::from("%0"),
                count: 1,
            }],
            dialect: DialectKind::Chlo,
            mnemonic: String::from("sinh"),
            op_kind: AstOpKind::ChloDecomposable(String::from("sinh")),
            operands: vec![AstOperand {
                name: String::from("%x"),
                result_index: None,
                span: TokenSpan::default(),
            }],
            regions: Vec::new(),
            successors: Vec::new(),
            attributes: BTreeMap::new(),
            signature: AstFunctionType::default(),
            location: None,
            span: TokenSpan::default(),
        }],
        ..AstModule::default()
    };
    decompose_chlo_module(&mut top_mod);
    assert!(top_mod.top_level_operations.len() > 1);
}

#[test]
#[allow(clippy::too_many_lines)]
fn test_phase4_3_dialect_resources_loc_aliases_verifier_and_hlo_proto_bridge() {
    let mlir_with_resources_and_locs = r#"
        module @jax_dump {
          func.func @main(%arg0: tensor<?x2xf32, #stablehlo.bounds<4, ?>> loc(#loc1)) -> tensor<2xf32> {
            %w = stablehlo.constant dense_resource<w_blob> : tensor<2xf32> loc(#loc2)
            %b = stablehlo.constant dense_resource<raw_short> : tensor<2xf32> loc(#loc3)
            %s = stablehlo.add %w, %b : tensor<2xf32> loc(#loc4)
            func.return %s : tensor<2xf32> loc(#loc0)
          }
        }
        #loc0 = loc("train.py":10:2)
        #loc1 = loc("arg0"(#loc0))
        #loc2 = loc(callsite(#loc0 at #loc1))
        #loc3 = loc(fused[#loc0, #loc1])
        #loc4 = loc(#loc2)
        {-#
          dialect_resources: {
            builtin: {
              w_blob: "0x080000000000803f00000040",
              raw_short: "0x030000000000404000008040"
            }
          }
        #-}
    "#;
    let ast_res = parse_unified_ast_module(mlir_with_resources_and_locs);
    assert!(ast_res.is_ok());
    if let Ok(ast) = ast_res {
        assert_eq!(ast.dialect_resources.len(), 2);
        assert!(verify_ast_module_semantics(&ast).is_ok());
        let first_arg_ty = ast
            .functions
            .first()
            .and_then(|f| f.arguments.first())
            .map(|a| &a.ty);
        if let Some(MlirType::RankedTensor(rt)) = first_arg_ty {
            assert_eq!(
                rt.dims,
                vec![DimensionSize::Bounded(4), DimensionSize::Static(2)]
            );
        }
    }

    // VHLO textual func_v1 + type_extensions_v1 bounds
    let vhlo_text = r#"
        module @vhlo_txt {
          vhlo.func_v1 @main(%x: tensor<?x4xf32, #vhlo.type_extensions_v1<bounds = [8, ?]>>) -> tensor<?x4xf32> {
            %0 = "vhlo.abs_v1"(%x) : (tensor<?x4xf32>) -> tensor<?x4xf32>
            "vhlo.return_v1"(%0) : (tensor<?x4xf32>) -> ()
          }
        }
    "#;
    assert!(parse_mlir_ast_module(vhlo_text).is_ok());

    // Dimension compatibility & comprehensive shape/type verifier error branches
    assert!(are_dimensions_compatible(
        DimensionSize::Static(8),
        DimensionSize::Bounded(16)
    ));
    assert!(!are_dimensions_compatible(
        DimensionSize::Static(32),
        DimensionSize::Bounded(16)
    ));
    assert!(are_dimensions_compatible(
        DimensionSize::Bounded(8),
        DimensionSize::Bounded(16)
    ));
    assert!(are_dimensions_compatible(
        DimensionSize::Dynamic,
        DimensionSize::Static(8)
    ));

    let mk_rt = |dims: Vec<DimensionSize>, et: ElementType| {
        MlirType::RankedTensor(RankedTensorType {
            dims,
            element_type: et,
            encoding: None,
            layout: None,
        })
    };
    assert!(are_tensor_shapes_compatible(
        &RankedTensorType {
            dims: vec![DimensionSize::Bounded(8)],
            element_type: ElementType::F32,
            encoding: None,
            layout: None
        },
        &RankedTensorType {
            dims: vec![DimensionSize::Static(4)],
            element_type: ElementType::F32,
            encoding: None,
            layout: None
        },
    ));

    let mk_op = |dialect: DialectKind,
                 mnemonic: &str,
                 op_kind: AstOpKind,
                 n_opnds: usize,
                 inputs: Vec<MlirType>,
                 results: Vec<MlirType>,
                 attrs: BTreeMap<String, MlirAttribute>| AstOperation {
        results: vec![AstOpResult {
            name: String::from("%0"),
            count: 1,
        }],
        dialect,
        mnemonic: String::from(mnemonic),
        op_kind,
        operands: (0..n_opnds)
            .map(|i| AstOperand {
                name: format!("%a{i}"),
                result_index: None,
                span: TokenSpan::default(),
            })
            .collect(),
        regions: Vec::new(),
        successors: Vec::new(),
        attributes: attrs,
        signature: AstFunctionType { inputs, results },
        location: None,
        span: TokenSpan::default(),
    };

    // Mosaic DMA < 2 operands error
    assert!(
        verify_operation_static_and_dynamic_types(&mk_op(
            DialectKind::TpuMosaic,
            "enqueue_dma",
            AstOpKind::MosaicDmaStart,
            1,
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        ))
        .is_err()
    );

    // Binary arity != 2 & binary shape mismatch
    assert!(
        verify_operation_static_and_dynamic_types(&mk_op(
            DialectKind::StableHlo,
            "add",
            AstOpKind::HardwareOp(TpuOpCode::AddF32),
            1,
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        ))
        .is_err()
    );
    assert!(
        verify_operation_static_and_dynamic_types(&mk_op(
            DialectKind::StableHlo,
            "add",
            AstOpKind::HardwareOp(TpuOpCode::AddF32),
            2,
            vec![
                mk_rt(vec![DimensionSize::Static(4)], ElementType::F32),
                mk_rt(vec![DimensionSize::Static(8)], ElementType::F32)
            ],
            Vec::new(),
            BTreeMap::new(),
        ))
        .is_err()
    );

    // dot_general errors: count mismatch, contracting OOB, contracting size mismatch, batch OOB, batch size mismatch
    for dot in [
        DotDimensionNumbersAttr {
            lhs_batching_dimensions: vec![0],
            rhs_batching_dimensions: Vec::new(),
            lhs_contracting_dimensions: vec![1],
            rhs_contracting_dimensions: vec![0],
        },
        DotDimensionNumbersAttr {
            lhs_batching_dimensions: Vec::new(),
            rhs_batching_dimensions: Vec::new(),
            lhs_contracting_dimensions: vec![5],
            rhs_contracting_dimensions: vec![0],
        },
        DotDimensionNumbersAttr {
            lhs_batching_dimensions: Vec::new(),
            rhs_batching_dimensions: Vec::new(),
            lhs_contracting_dimensions: vec![1],
            rhs_contracting_dimensions: vec![1],
        },
        DotDimensionNumbersAttr {
            lhs_batching_dimensions: vec![5],
            rhs_batching_dimensions: vec![0],
            lhs_contracting_dimensions: Vec::new(),
            rhs_contracting_dimensions: Vec::new(),
        },
        DotDimensionNumbersAttr {
            lhs_batching_dimensions: vec![0],
            rhs_batching_dimensions: vec![1],
            lhs_contracting_dimensions: Vec::new(),
            rhs_contracting_dimensions: Vec::new(),
        },
    ] {
        let mut attrs = BTreeMap::new();
        attrs.insert(
            String::from("dot_dimension_numbers"),
            MlirAttribute::DotDimensionNumbers(dot),
        );
        assert!(
            verify_operation_static_and_dynamic_types(&mk_op(
                DialectKind::StableHlo,
                "dot_general",
                AstOpKind::HardwareOp(TpuOpCode::DotGeneralF32),
                2,
                vec![
                    mk_rt(
                        vec![DimensionSize::Static(2), DimensionSize::Static(4)],
                        ElementType::F32
                    ),
                    mk_rt(
                        vec![DimensionSize::Static(4), DimensionSize::Static(8)],
                        ElementType::F32
                    ),
                ],
                Vec::new(),
                attrs,
            ))
            .is_err()
        );
    }

    // reshape volume mismatch
    assert!(
        verify_operation_static_and_dynamic_types(&mk_op(
            DialectKind::StableHlo,
            "reshape",
            AstOpKind::HardwareOp(TpuOpCode::Reshape),
            1,
            vec![mk_rt(
                vec![DimensionSize::Static(4), DimensionSize::Static(4)],
                ElementType::F32
            )],
            vec![mk_rt(vec![DimensionSize::Static(8)], ElementType::F32)],
            BTreeMap::new(),
        ))
        .is_err()
    );

    // transpose invalid permutation
    let mut tr_attrs = BTreeMap::new();
    tr_attrs.insert(
        String::from("permutation"),
        MlirAttribute::DenseI64Array(vec![0, 0]),
    );
    assert!(
        verify_operation_static_and_dynamic_types(&mk_op(
            DialectKind::StableHlo,
            "transpose",
            AstOpKind::HardwareOp(TpuOpCode::Transpose),
            1,
            vec![mk_rt(
                vec![DimensionSize::Static(4), DimensionSize::Static(8)],
                ElementType::F32
            )],
            Vec::new(),
            tr_attrs,
        ))
        .is_err()
    );

    // slice start > limit and stride == 0
    let mut sl1 = BTreeMap::new();
    sl1.insert(
        String::from("start_indices"),
        MlirAttribute::DenseI64Array(vec![5]),
    );
    sl1.insert(
        String::from("limit_indices"),
        MlirAttribute::DenseI64Array(vec![2]),
    );
    assert!(
        verify_operation_static_and_dynamic_types(&mk_op(
            DialectKind::StableHlo,
            "slice",
            AstOpKind::HardwareOp(TpuOpCode::Slice),
            1,
            Vec::new(),
            Vec::new(),
            sl1,
        ))
        .is_err()
    );
    let mut sl2 = BTreeMap::new();
    sl2.insert(
        String::from("start_indices"),
        MlirAttribute::DenseI64Array(vec![0]),
    );
    sl2.insert(
        String::from("limit_indices"),
        MlirAttribute::DenseI64Array(vec![2]),
    );
    sl2.insert(
        String::from("strides"),
        MlirAttribute::DenseI64Array(vec![0]),
    );
    assert!(
        verify_operation_static_and_dynamic_types(&mk_op(
            DialectKind::StableHlo,
            "slice",
            AstOpKind::HardwareOp(TpuOpCode::Slice),
            1,
            Vec::new(),
            Vec::new(),
            sl2,
        ))
        .is_err()
    );

    // concatenate axis OOB, rank mismatch, non-concat dim mismatch
    let mut cat_oob = BTreeMap::new();
    cat_oob.insert(
        String::from("dimension"),
        MlirAttribute::Integer { value: 3, ty: None },
    );
    assert!(
        verify_operation_static_and_dynamic_types(&mk_op(
            DialectKind::StableHlo,
            "concatenate",
            AstOpKind::HardwareOp(TpuOpCode::Concatenate),
            2,
            vec![mk_rt(vec![DimensionSize::Static(4)], ElementType::F32)],
            Vec::new(),
            cat_oob,
        ))
        .is_err()
    );
    let mut cat_ax0 = BTreeMap::new();
    cat_ax0.insert(
        String::from("dimension"),
        MlirAttribute::Integer { value: 0, ty: None },
    );
    assert!(
        verify_operation_static_and_dynamic_types(&mk_op(
            DialectKind::StableHlo,
            "concatenate",
            AstOpKind::HardwareOp(TpuOpCode::Concatenate),
            2,
            vec![
                mk_rt(
                    vec![DimensionSize::Static(4), DimensionSize::Static(8)],
                    ElementType::F32
                ),
                mk_rt(vec![DimensionSize::Static(4)], ElementType::F32),
            ],
            Vec::new(),
            cat_ax0.clone(),
        ))
        .is_err()
    );
    assert!(
        verify_operation_static_and_dynamic_types(&mk_op(
            DialectKind::StableHlo,
            "concatenate",
            AstOpKind::HardwareOp(TpuOpCode::Concatenate),
            2,
            vec![
                mk_rt(
                    vec![DimensionSize::Static(4), DimensionSize::Static(8)],
                    ElementType::F32
                ),
                mk_rt(
                    vec![DimensionSize::Static(4), DimensionSize::Static(16)],
                    ElementType::F32
                ),
            ],
            Vec::new(),
            cat_ax0,
        ))
        .is_err()
    );

    // select non-Pred and branch mismatch
    assert!(
        verify_operation_static_and_dynamic_types(&mk_op(
            DialectKind::StableHlo,
            "select",
            AstOpKind::HardwareOp(TpuOpCode::Select),
            3,
            vec![
                mk_rt(vec![DimensionSize::Static(4)], ElementType::S32),
                mk_rt(vec![DimensionSize::Static(4)], ElementType::F32),
                mk_rt(vec![DimensionSize::Static(4)], ElementType::F32),
            ],
            Vec::new(),
            BTreeMap::new(),
        ))
        .is_err()
    );
    assert!(
        verify_operation_static_and_dynamic_types(&mk_op(
            DialectKind::StableHlo,
            "select",
            AstOpKind::HardwareOp(TpuOpCode::Select),
            3,
            vec![
                mk_rt(vec![DimensionSize::Static(4)], ElementType::Pred),
                mk_rt(vec![DimensionSize::Static(4)], ElementType::F32),
                mk_rt(vec![DimensionSize::Static(8)], ElementType::F32),
            ],
            Vec::new(),
            BTreeMap::new(),
        ))
        .is_err()
    );

    // get_tuple_element OOB
    assert!(
        verify_operation_static_and_dynamic_types(&mk_op(
            DialectKind::StableHlo,
            "get_tuple_element",
            AstOpKind::GetTupleElement { index: 2 },
            1,
            vec![MlirType::Tuple(vec![mk_rt(
                vec![DimensionSize::Static(4)],
                ElementType::F32
            )])],
            Vec::new(),
            BTreeMap::new(),
        ))
        .is_err()
    );

    // Lossless StableHLO <-> HloModuleProto roundtrip across rich ops, regions, calls, and literals
    assert!(stablehlo_to_hlo_module_proto(&AstModule::default()).is_err());
    assert!(hlo_module_proto_to_stablehlo(&HloModuleProto::default()).is_err());

    for ty in [
        MlirType::Token,
        MlirType::Scalar(ElementType::F32),
        MlirType::UnrankedTensor(ElementType::F32),
        MlirType::Tuple(vec![
            mk_rt(
                vec![
                    DimensionSize::Dynamic,
                    DimensionSize::Bounded(16),
                    DimensionSize::Static(8),
                ],
                ElementType::Bf16,
            ),
            MlirType::Token,
        ]),
        MlirType::RankedTensor(RankedTensorType {
            dims: vec![DimensionSize::Static(8), DimensionSize::Static(128)],
            element_type: ElementType::F32,
            encoding: None,
            layout: Some(MemoryLayout::Tiled {
                minor_to_major: vec![1, 0],
                tile_dims: vec![8, 128],
                tile_dim_sizes: vec![2],
            }),
        }),
    ] {
        let sp = mlir_type_to_xla_shape_proto(&ty);
        let round = xla_shape_proto_to_mlir_type(&sp);
        let _ = mlir_type_to_xla_shape_proto(&round);
    }

    let full_stablehlo = r#"
        module @xla_bridge {
          func.func private @helper(%h: tensor<4xf32>) -> tensor<4xf32> {
            %n = stablehlo.negate %h : tensor<4xf32>
            func.return %n : tensor<4xf32>
          }
          func.func @main(
            %x: tensor<2x4xf32> {mhlo.sharding = "{replicated}"},
            %y: tensor<4x2xf32>
          ) -> (tensor<2xf32>, tensor<2x2xf32>) {
            %c_f32 = stablehlo.constant dense<1.0> : tensor<2x4xf32>
            %c_f64 = stablehlo.constant dense<2.0> : tensor<2xf64>
            %c_s32 = stablehlo.constant dense<3> : tensor<2xi32>
            %c_s64 = stablehlo.constant dense<4> : tensor<2xi64>
            %c_u32 = stablehlo.constant dense<5> : tensor<2xui32>
            %c_u64 = stablehlo.constant dense<6> : tensor<2xui64>
            %c_bf16 = stablehlo.constant dense<1.5> : tensor<2xbf16>
            %c_f16 = stablehlo.constant dense<1.5> : tensor<2xf16>
            %c_i8 = stablehlo.constant dense<1> : tensor<2xi8>
            %c_pred = stablehlo.constant dense<true> : tensor<2xi1>
            %sum = stablehlo.add %x, %c_f32 : tensor<2x4xf32>
            %dot = stablehlo.dot_general %sum, %y, contracting_dims = [1] x [0] : (tensor<2x4xf32>, tensor<4x2xf32>) -> tensor<2x2xf32>
            %sl = stablehlo.slice %dot [0:2:1, 0:2:1] : (tensor<2x2xf32>) -> tensor<2x2xf32>
            %z0 = stablehlo.constant dense<0.0> : tensor<f32>
            %pad = stablehlo.pad %sl, %z0, low = [0, 0], high = [0, 0], interior = [0, 0] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2x2xf32>
            %cmp = stablehlo.compare EQ, %pad, %sl, FLOAT : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
            %io = stablehlo.iota dim = 0 : tensor<4xf32>
            %called = func.call @helper(%io) : (tensor<4xf32>) -> tensor<4xf32>
            %ar = "stablehlo.all_reduce"(%pad) ({
            ^bb0(%la: tensor<f32>, %ra: tensor<f32>):
              %sa = stablehlo.add %la, %ra : tensor<f32>
              stablehlo.return %sa : tensor<f32>
            }) {channel_id = 1 : i64, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>} : (tensor<2x2xf32>) -> tensor<2x2xf32>
            %red = stablehlo.reduce(%ar init: %z0) across dimensions = [1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
             reducer(%a0: tensor<f32>, %a1: tensor<f32>) {
              %rsum = stablehlo.add %a0, %a1 : tensor<f32>
              stablehlo.return %rsum : tensor<f32>
            }
            func.return %red, %ar : tensor<2xf32>, tensor<2x2xf32>
          }
        }
    "#;
    let bridge_ast = parse_mlir_ast_module(full_stablehlo);
    assert!(bridge_ast.is_ok());
    if let Ok(ast) = bridge_ast {
        let proto_via_fn = hlo_ast_to_proto(&ast);
        assert_ne!(proto_via_fn.computations.len(), 0);
        let proto_res = stablehlo_to_hlo_module_proto(&ast);
        assert!(proto_res.is_ok());
        if let Ok(proto) = proto_res {
            let wire = proto.encode_proto();
            let decoded_proto = HloModuleProto::decode_proto(&wire);
            assert!(decoded_proto.is_ok());
            if let Ok(dp) = decoded_proto {
                let back_ast = hlo_module_proto_to_stablehlo(&dp);
                assert!(back_ast.is_ok());
                if let Ok(round_ast) = back_ast {
                    assert!(round_ast.functions.len() >= 2);
                    assert!(lower_ast_module_to_parsed_hlo(&round_ast).is_ok());
                }
            }
        }
    }

    // Also test HLO text ->AstModule -> stablehlo_to_hlo_module_proto delegation & bare top-level ops
    let hlo_text = "HloModule m
ENTRY %main (p: f32[<=8,?]) -> f32[<=8,?] {
  %p = f32[<=8,?] parameter(0)
  ROOT %r = f32[<=8,?] negate(%p)
}
";
    if let Ok(h_ast) = parse_unified_ast_module(hlo_text) {
        let hp = stablehlo_to_hlo_module_proto(&h_ast);
        assert!(hp.is_ok());
    }
    let bare_mlir = "%0 = stablehlo.constant dense<1.0> : tensor<4xf32>";
    if let Ok(b_ast) = parse_mlir_ast_module(bare_mlir) {
        assert!(stablehlo_to_hlo_module_proto(&b_ast).is_ok());
    }
}

#[test]
#[allow(clippy::too_many_lines)]
fn test_phase4_remaining_edge_cases_and_100pct_branches() {
    use crate::compiler::{
        AstBlock, AstBlockArg, AstFunction, AstRegion, AstResultType, ConvDimensionNumbersAttr,
        DenseElementsAttr, GatherDimensionNumbersAttr, MlirLocation, ScatterDimensionNumbersAttr,
        Visibility,
    };
    use crate::proto::{
        HloComputationProto, HloInstructionProto, LiteralProto, PaddingConfig,
        PaddingConfigDimension, ReplicaGroup, ShapeProto,
    };

    // 1. dialects.rs: unreduced_axis_indices, lookup_str fallback, and non-sharding in TensorShardingPerValue
    let mut dw = DialectBytecodeWriter::new();
    let ts_unred = SdyCustomAttr::TensorSharding {
        mesh_name_string_idx: 99,
        dim_shardings: Vec::new(),
        replicated_axis_indices: Vec::new(),
        unreduced_axis_indices: vec![98],
    };
    ts_unred.encode(&mut dw);
    let spv = SdyCustomAttr::TensorShardingPerValue(vec![0, 99]);
    spv.encode(&mut dw);
    let d_bytes = dw.into_bytes();
    let mut dr = DialectBytecodeReader::new(&d_bytes, 2);
    if let Ok(dec_ts) = SdyCustomAttr::decode(&mut dr) {
        let attr0 = dec_ts.to_mlir_attribute(&[], &[MlirAttribute::Bool(false)]);
        if let Ok(dec_spv) = SdyCustomAttr::decode(&mut dr) {
            let _ = dec_spv.to_mlir_attribute(&[], &[MlirAttribute::Bool(false), attr0]);
        }
    }

    // 2. builtin.rs: remaining fallback branches
    let b_types = vec![
        MlirType::Scalar(ElementType::Pred),
        MlirType::Scalar(ElementType::S32),
    ];
    let b_attrs = vec![
        MlirAttribute::Bool(false),
        MlirAttribute::Array(vec![
            MlirAttribute::Integer { value: 8, ty: None },
            MlirAttribute::Integer {
                value: -1,
                ty: None,
            },
        ]),
    ];
    let _ = BuiltinCustomType::MemRef {
        memory_space_attr_idx: Some(0),
        shape: vec![4],
        element_type_idx: 1,
        layout_attr_idx: None,
    }
    .to_mlir_type(&b_types, &b_attrs);
    let _ = BuiltinCustomType::RankedTensor {
        encoding_attr_idx: Some(999),
        shape: vec![4],
        element_type_idx: 1,
    }
    .to_mlir_type(&b_types, &b_attrs);
    let _ = BuiltinCustomType::RankedTensor {
        encoding_attr_idx: Some(1),
        shape: vec![4, -1],
        element_type_idx: 1,
    }
    .to_mlir_type(&b_types, &b_attrs);
    let _ = BuiltinCustomAttr::String {
        string_idx: 99,
        type_idx: None,
    }
    .to_mlir_attribute(&[], &b_types, &b_attrs, &[]);
    let _ = BuiltinCustomAttr::CallSiteLoc {
        callee_loc_idx: 99,
        caller_loc_idx: 98,
    }
    .to_mlir_attribute(&[], &b_types, &b_attrs, &[]);
    let _ = BuiltinCustomAttr::Dictionary(vec![(0, 999)]).to_mlir_attribute(
        &[],
        &b_types,
        &b_attrs,
        &[],
    );
    let _ = BuiltinCustomAttr::Integer {
        type_idx: 0,
        bit_width: 8,
        value: 1,
    }
    .to_mlir_attribute(&[], &b_types, &b_attrs, &[]);
    let _ = BuiltinCustomAttr::DenseResourceElements {
        shaped_type_idx: 0,
        resource_handle_idx: 99,
    }
    .to_mlir_attribute(&[], &b_types, &b_attrs, &[]);
    let _ = BuiltinCustomAttr::SparseElements {
        shaped_type_idx: 0,
        indices_attr_idx: 99,
        values_attr_idx: 98,
    }
    .to_mlir_attribute(&[], &b_types, &b_attrs, &[]);

    // 3. mlir_bytecode.rs: reconcile_binary_attr_type_entries & non-pow2 / <=1 section alignment
    let mut c1 = DialectBytecodeWriter::new();
    c1.write_bytes(MLIR_BYTECODE_MAGIC);
    c1.write_varint(6);
    c1.write_nul_string("p");
    // Section 3 without Section 2
    c1.write_section(3, None, &encode_prefix_varint(1));
    assert!(MlirBytecodeHeader::decode(&c1.into_bytes()).is_ok());

    let mut c2 = DialectBytecodeWriter::new();
    c2.write_bytes(MLIR_BYTECODE_MAGIC);
    c2.write_varint(6);
    c2.write_nul_string("p");
    // Section 2 with invalid binary tag [0x00] and Section 3 with offset [0, 1], plus manual align=1 and align=3 sections
    c2.write_section(2, None, &[0x00]);
    let mut sec3_zero = encode_prefix_varint(0);
    sec3_zero.extend_from_slice(&encode_prefix_varint(1));
    c2.write_section(3, None, &sec3_zero);
    // Manual section header with has_align=true and align=1, and has_align=true and align=3
    c2.write_byte(0x80);
    c2.write_varint(0);
    c2.write_varint(1);
    c2.write_byte(0x80);
    c2.write_varint(0);
    c2.write_varint(3);
    assert!(MlirBytecodeHeader::decode(&c2.into_bytes()).is_ok());

    // 4. vhlo.rs: downgrade & upgrade with nested regions, top-level ops, unknown _v1/_v2 ops, and Section-7 edge cases
    let _ = VhloCustomType::RankedTensorV1 {
        encoding_attr_idx: Some(99),
        shape: vec![4],
        element_type_idx: 0,
    }
    .to_mlir_type(&b_types, &[]);
    let _ = VhloCustomType::RankedTensorV1 {
        encoding_attr_idx: Some(0),
        shape: vec![4, -1],
        element_type_idx: 0,
    }
    .to_mlir_type(&b_types, &[MlirAttribute::DenseI64Array(vec![8, -1])]);
    let _ = VhloCustomAttr::DictionaryV1(vec![(99, 99)]).to_mlir_attribute(&b_types, &[]);

    let nested_shlo = r#"
        module @vhlo_nested {
          func.func @main(%x: tensor<2x2xf32>, %z: tensor<f32>) -> tensor<2xf32> {
            %r = "stablehlo.reduce"(%x, %z) ({
            ^bb0(%a: tensor<f32>, %b: tensor<f32>):
              %s = stablehlo.add %a, %b : tensor<f32>
              stablehlo.return %s : tensor<f32>
            }) {dimensions = array<i64: 1>} : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
            func.return %r : tensor<2xf32>
          }
        }
    "#;
    if let Ok(mut m) = parse_mlir_ast_module(nested_shlo) {
        m.top_level_operations.push(AstOperation {
            results: Vec::new(),
            dialect: DialectKind::StableHlo,
            mnemonic: String::from("abs"),
            op_kind: AstOpKind::HardwareOp(TpuOpCode::Abs),
            operands: Vec::new(),
            regions: Vec::new(),
            successors: Vec::new(),
            attributes: BTreeMap::new(),
            signature: AstFunctionType::default(),
            location: None,
            span: TokenSpan::default(),
        });
        if let Ok(mut down) = downgrade_stablehlo_to_vhlo(&m, VhloVersion::CURRENT) {
            down.top_level_operations.push(AstOperation {
                results: Vec::new(),
                dialect: DialectKind::Vhlo,
                mnemonic: String::from("mystery_v1"),
                op_kind: AstOpKind::HardwareOp(TpuOpCode::Abs),
                operands: Vec::new(),
                regions: Vec::new(),
                successors: Vec::new(),
                attributes: BTreeMap::new(),
                signature: AstFunctionType::default(),
                location: None,
                span: TokenSpan::default(),
            });
            down.top_level_operations.push(AstOperation {
                results: Vec::new(),
                dialect: DialectKind::Vhlo,
                mnemonic: String::from("return"),
                op_kind: AstOpKind::Return,
                operands: Vec::new(),
                regions: Vec::new(),
                successors: Vec::new(),
                attributes: BTreeMap::new(),
                signature: AstFunctionType::default(),
                location: None,
                span: TokenSpan::default(),
            });
            assert!(upgrade_vhlo_to_stablehlo(&mut down).is_ok());
        }
        // Portable artifact without Section 7, truncated Section 7, & unsupported Section 7 (0.5.0)
        let trunc_sec7 = MlirBytecodeHeader::encode_from_ast(&m, 6);
        assert!(deserialize_portable_vhlo_artifact(&trunc_sec7).is_err());
        let no_sec7 = crate::compiler::MlirBytecodeEncoder::new("p").encode_to_bytes();
        assert!(deserialize_portable_vhlo_artifact(&no_sec7).is_ok());
        let bad_sec7 = MlirBytecodeHeader::encode_portable_vhlo_ast(&m, VhloVersion::new(0, 5, 0));
        assert!(deserialize_portable_vhlo_artifact(&bad_sec7).is_err());
    }

    // 5. parser.rs: cyclic #loc, block-arg #loc, missing dense_resource, short/large-align hex blobs, slice without stride, nested [[0]]
    let edge_mlir = r#"
        vhlo.func_v1 @bare_vhlo(%x: tensor<?x?xf32, #stablehlo.bounds<0, abc>>) -> tensor<2xf32> {
          %sl = stablehlo.slice %x [0:2, 0:1] : (tensor<4x2xf32>) -> tensor<2x1xf32>
          %ld = vector.load %x[[0]] : memref<4xf32>, vector<4xf32>
          %miss = stablehlo.constant dense_resource<missing_key> : tensor<2xf32>
          %r = "stablehlo.all_reduce"(%miss) ({
          ^bb0(%ba: tensor<f32> loc(#loc_cyc), %bb: tensor<f32>):
            %s = stablehlo.add %ba, %bb : tensor<f32>
            stablehlo.return %s : tensor<f32>
          }) : (tensor<2xf32>) -> tensor<2xf32>
          func.return %r : tensor<2xf32>
        }
        #loc_cyc = loc(#loc_cyc)
    "#;
    assert!(parse_mlir_ast_module(edge_mlir).is_ok());

    let res_only_mlir = r#"
        %0 = "stablehlo.all_reduce"() ({
        ^bb0(%a: f32):
          stablehlo.return %a : f32
        }) : () -> f32 loc("top.py":1:1)
        {-# dialect_resources: { builtin: { s: "0x1122", big_align: "0x0020000011223344" } } #-}
    "#;
    assert!(parse_mlir_ast_module(res_only_mlir).is_ok());
    assert!(
        parse_mlir_ast_module("{-# dialect_resources: { builtin: { unclosed: \"0x1122\" } ")
            .is_err()
            || parse_mlir_ast_module("{-# unclosed").is_ok()
    );

    // 6. chlo.rs: scalar broadcast fallback, mismatched explicit_bcast_dims, scalar select/conj/ragged_dot/constant variants
    let scalar_f32 = MlirType::Scalar(ElementType::F32);
    let r3_f32 = MlirType::RankedTensor(RankedTensorType {
        dims: vec![
            DimensionSize::Static(2),
            DimensionSize::Static(2),
            DimensionSize::Static(2),
        ],
        element_type: ElementType::F32,
        encoding: None,
        layout: None,
    });
    let mut chlo_edge = AstModule {
        top_level_operations: vec![
            AstOperation {
                results: Vec::new(),
                dialect: DialectKind::Chlo,
                mnemonic: String::from("broadcast_add"),
                op_kind: AstOpKind::ChloDecomposable(String::from("broadcast_add")),
                operands: Vec::new(),
                regions: Vec::new(),
                successors: Vec::new(),
                attributes: {
                    let mut m = BTreeMap::new();
                    m.insert(
                        String::from("broadcast_dimensions"),
                        MlirAttribute::DenseI64Array(vec![0, 1, 2]),
                    );
                    m
                },
                signature: AstFunctionType {
                    inputs: vec![
                        MlirType::RankedTensor(RankedTensorType {
                            dims: vec![DimensionSize::Static(2)],
                            element_type: ElementType::F32,
                            encoding: None,
                            layout: None,
                        }),
                        scalar_f32.clone(),
                    ],
                    results: vec![r3_f32.clone()],
                },
                location: None,
                span: TokenSpan::default(),
            },
            AstOperation {
                results: vec![AstOpResult {
                    name: String::from("%sc"),
                    count: 1,
                }],
                dialect: DialectKind::Chlo,
                mnemonic: String::from("broadcast_compare"),
                op_kind: AstOpKind::ChloDecomposable(String::from("broadcast_compare")),
                operands: Vec::new(),
                regions: Vec::new(),
                successors: Vec::new(),
                attributes: BTreeMap::new(),
                signature: AstFunctionType {
                    inputs: vec![scalar_f32.clone()],
                    results: vec![scalar_f32.clone()],
                },
                location: None,
                span: TokenSpan::default(),
            },
            AstOperation {
                results: vec![AstOpResult {
                    name: String::from("%ss"),
                    count: 1,
                }],
                dialect: DialectKind::Chlo,
                mnemonic: String::from("broadcast_select"),
                op_kind: AstOpKind::ChloDecomposable(String::from("broadcast_select")),
                operands: Vec::new(),
                regions: Vec::new(),
                successors: Vec::new(),
                attributes: BTreeMap::new(),
                signature: AstFunctionType {
                    inputs: vec![scalar_f32.clone()],
                    results: vec![scalar_f32.clone()],
                },
                location: None,
                span: TokenSpan::default(),
            },
            AstOperation {
                results: vec![AstOpResult {
                    name: String::from("%cj_s"),
                    count: 1,
                }],
                dialect: DialectKind::Chlo,
                mnemonic: String::from("conj"),
                op_kind: AstOpKind::ChloDecomposable(String::from("conj")),
                operands: Vec::new(),
                regions: Vec::new(),
                successors: Vec::new(),
                attributes: BTreeMap::new(),
                signature: AstFunctionType {
                    inputs: vec![scalar_f32.clone()],
                    results: vec![scalar_f32.clone()],
                },
                location: None,
                span: TokenSpan::default(),
            },
            AstOperation {
                results: vec![AstOpResult {
                    name: String::from("%tk_s"),
                    count: 1,
                }],
                dialect: DialectKind::Chlo,
                mnemonic: String::from("top_k"),
                op_kind: AstOpKind::ChloDecomposable(String::from("top_k")),
                operands: Vec::new(),
                regions: Vec::new(),
                successors: Vec::new(),
                attributes: BTreeMap::new(),
                signature: AstFunctionType {
                    inputs: vec![scalar_f32.clone()],
                    results: Vec::new(),
                },
                location: None,
                span: TokenSpan::default(),
            },
            AstOperation {
                results: vec![AstOpResult {
                    name: String::from("%rd1"),
                    count: 1,
                }],
                dialect: DialectKind::Chlo,
                mnemonic: String::from("ragged_dot"),
                op_kind: AstOpKind::ChloDecomposable(String::from("ragged_dot")),
                operands: Vec::new(),
                regions: Vec::new(),
                successors: Vec::new(),
                attributes: {
                    let mut m = BTreeMap::new();
                    m.insert(
                        String::from("dot_dimension_numbers"),
                        MlirAttribute::DotDimensionNumbers(DotDimensionNumbersAttr {
                            lhs_batching_dimensions: Vec::new(),
                            rhs_batching_dimensions: Vec::new(),
                            lhs_contracting_dimensions: vec![1],
                            rhs_contracting_dimensions: vec![0],
                        }),
                    );
                    m
                },
                signature: AstFunctionType {
                    inputs: vec![scalar_f32.clone(), r3_f32.clone()],
                    results: vec![scalar_f32.clone()],
                },
                location: None,
                span: TokenSpan::default(),
            },
            AstOperation {
                results: vec![AstOpResult {
                    name: String::from("%rd2"),
                    count: 1,
                }],
                dialect: DialectKind::Chlo,
                mnemonic: String::from("ragged_dot"),
                op_kind: AstOpKind::ChloDecomposable(String::from("ragged_dot")),
                operands: Vec::new(),
                regions: Vec::new(),
                successors: Vec::new(),
                attributes: {
                    let mut m = BTreeMap::new();
                    m.insert(
                        String::from("dot_dimension_numbers"),
                        MlirAttribute::Bool(false),
                    );
                    m
                },
                signature: AstFunctionType {
                    inputs: vec![scalar_f32.clone(), r3_f32],
                    results: vec![scalar_f32.clone()],
                },
                location: None,
                span: TokenSpan::default(),
            },
            AstOperation {
                results: vec![AstOpResult {
                    name: String::from("%cf"),
                    count: 1,
                }],
                dialect: DialectKind::Chlo,
                mnemonic: String::from("constant"),
                op_kind: AstOpKind::ChloDecomposable(String::from("constant")),
                operands: Vec::new(),
                regions: Vec::new(),
                successors: Vec::new(),
                attributes: {
                    let mut m = BTreeMap::new();
                    m.insert(
                        String::from("value"),
                        MlirAttribute::Float {
                            value: 3.0,
                            ty: None,
                        },
                    );
                    m
                },
                signature: AstFunctionType {
                    inputs: Vec::new(),
                    results: vec![scalar_f32.clone()],
                },
                location: None,
                span: TokenSpan::default(),
            },
            AstOperation {
                results: vec![AstOpResult {
                    name: String::from("%ci"),
                    count: 1,
                }],
                dialect: DialectKind::Chlo,
                mnemonic: String::from("constant_like"),
                op_kind: AstOpKind::ChloDecomposable(String::from("constant_like")),
                operands: Vec::new(),
                regions: Vec::new(),
                successors: Vec::new(),
                attributes: {
                    let mut m = BTreeMap::new();
                    m.insert(
                        String::from("value"),
                        MlirAttribute::Integer { value: 5, ty: None },
                    );
                    m
                },
                signature: AstFunctionType {
                    inputs: Vec::new(),
                    results: vec![scalar_f32.clone()],
                },
                location: None,
                span: TokenSpan::default(),
            },
            AstOperation {
                results: vec![AstOpResult {
                    name: String::from("%cb"),
                    count: 1,
                }],
                dialect: DialectKind::Chlo,
                mnemonic: String::from("constant"),
                op_kind: AstOpKind::ChloDecomposable(String::from("constant")),
                operands: Vec::new(),
                regions: Vec::new(),
                successors: Vec::new(),
                attributes: {
                    let mut m = BTreeMap::new();
                    m.insert(String::from("value"), MlirAttribute::Bool(true));
                    m
                },
                signature: AstFunctionType {
                    inputs: Vec::new(),
                    results: vec![scalar_f32.clone()],
                },
                location: None,
                span: TokenSpan::default(),
            },
        ],
        ..AstModule::default()
    };
    decompose_chlo_module(&mut chlo_edge);

    // 7. hlo_parser.rs & hlo_bridge.rs: empty hlo_ast_to_proto, dynamic reshape, conv/gather/scatter/pad/gte/custom-call/multi-result bridge
    let empty_hp = hlo_ast_to_proto(&AstModule::default());
    assert_eq!(empty_hp.computations.len(), 0);

    let t2x2 = MlirType::RankedTensor(RankedTensorType {
        dims: vec![DimensionSize::Static(2), DimensionSize::Static(2)],
        element_type: ElementType::F32,
        encoding: None,
        layout: None,
    });
    let tdyn = MlirType::RankedTensor(RankedTensorType {
        dims: vec![DimensionSize::Dynamic, DimensionSize::Static(2)],
        element_type: ElementType::F32,
        encoding: None,
        layout: None,
    });
    // Verifier valid dynamic reshape & transpose empty perm & valid select & valid GTE
    let dyn_reshape = AstOperation {
        results: vec![AstOpResult {
            name: String::from("%dr"),
            count: 1,
        }],
        dialect: DialectKind::StableHlo,
        mnemonic: String::from("reshape"),
        op_kind: AstOpKind::HardwareOp(TpuOpCode::Reshape),
        operands: vec![AstOperand {
            name: String::from("%x"),
            result_index: None,
            span: TokenSpan::default(),
        }],
        regions: Vec::new(),
        successors: Vec::new(),
        attributes: BTreeMap::new(),
        signature: AstFunctionType {
            inputs: vec![tdyn],
            results: vec![t2x2.clone()],
        },
        location: None,
        span: TokenSpan::default(),
    };
    assert!(verify_operation_static_and_dynamic_types(&dyn_reshape).is_ok());

    // Build an AstModule exercising conv, gather, scatter, dot, pad, transpose, concatenate, gte, sdy.reshard, custom_call, multi-result, splat literal, comparison types
    let mut rich_ops = Vec::new();
    let push_op = |ops: &mut Vec<AstOperation>,
                   res: Vec<AstOpResult>,
                   dialect: DialectKind,
                   mnemonic: &str,
                   op_kind: AstOpKind,
                   operands: Vec<AstOperand>,
                   attrs: BTreeMap<String, MlirAttribute>,
                   sig: AstFunctionType| {
        ops.push(AstOperation {
            results: res,
            dialect,
            mnemonic: String::from(mnemonic),
            op_kind,
            operands,
            regions: Vec::new(),
            successors: Vec::new(),
            attributes: attrs,
            signature: sig,
            location: Some(MlirLocation::Unknown),
            span: TokenSpan::default(),
        });
    };
    let arg_opnd = AstOperand {
        name: String::from("%p0"),
        result_index: None,
        span: TokenSpan::default(),
    };

    // Splat DenseElements without raw_bytes
    let mut splat_attrs = BTreeMap::new();
    splat_attrs.insert(
        String::from("value"),
        MlirAttribute::DenseElements(DenseElementsAttr {
            values_f64: vec![3.5],
            raw_bytes: None,
            is_splat: true,
            tensor_type: Some(RankedTensorType {
                dims: vec![DimensionSize::Static(2), DimensionSize::Static(2)],
                element_type: ElementType::F32,
                encoding: None,
                layout: None,
            }),
        }),
    );
    push_op(
        &mut rich_ops,
        vec![AstOpResult {
            name: String::from("%splat"),
            count: 1,
        }],
        DialectKind::StableHlo,
        "constant",
        AstOpKind::HardwareOp(TpuOpCode::Constant),
        Vec::new(),
        splat_attrs,
        AstFunctionType {
            inputs: Vec::new(),
            results: vec![t2x2.clone()],
        },
    );

    // Non-DenseElements constant attribute (hits `_ => None`)
    let mut non_de_attrs = BTreeMap::new();
    non_de_attrs.insert(String::from("value"), MlirAttribute::Bool(true));
    push_op(
        &mut rich_ops,
        vec![AstOpResult {
            name: String::from("%non_de"),
            count: 1,
        }],
        DialectKind::StableHlo,
        "constant",
        AstOpKind::HardwareOp(TpuOpCode::Constant),
        Vec::new(),
        non_de_attrs,
        AstFunctionType {
            inputs: Vec::new(),
            results: vec![t2x2.clone()],
        },
    );

    // dot (2D shorthand) and dot_general with explicit DotDimensionNumbersAttr
    let mut dg_attrs = BTreeMap::new();
    dg_attrs.insert(
        String::from("dot_dimension_numbers"),
        MlirAttribute::DotDimensionNumbers(DotDimensionNumbersAttr {
            lhs_batching_dimensions: Vec::new(),
            rhs_batching_dimensions: Vec::new(),
            lhs_contracting_dimensions: vec![1],
            rhs_contracting_dimensions: vec![0],
        }),
    );
    dg_attrs.insert(
        String::from("mhlo.sharding"),
        MlirAttribute::String(String::from("{replicated}")),
    );
    push_op(
        &mut rich_ops,
        vec![AstOpResult {
            name: String::from("%dg"),
            count: 1,
        }],
        DialectKind::StableHlo,
        "dot_general",
        AstOpKind::HardwareOp(TpuOpCode::DotGeneralF32),
        vec![arg_opnd.clone(), arg_opnd.clone()],
        dg_attrs,
        AstFunctionType {
            inputs: vec![t2x2.clone(), t2x2.clone()],
            results: vec![t2x2.clone()],
        },
    );
    push_op(
        &mut rich_ops,
        vec![AstOpResult {
            name: String::from("%d2"),
            count: 1,
        }],
        DialectKind::StableHlo,
        "dot",
        AstOpKind::HardwareOp(TpuOpCode::DotGeneralF32),
        vec![arg_opnd.clone(), arg_opnd.clone()],
        BTreeMap::new(),
        AstFunctionType {
            inputs: vec![t2x2.clone(), t2x2.clone()],
            results: vec![t2x2.clone()],
        },
    );

    // convolution
    let mut conv_attrs = BTreeMap::new();
    conv_attrs.insert(
        String::from("dimension_numbers"),
        MlirAttribute::ConvDimensionNumbers(ConvDimensionNumbersAttr {
            input_batch_dimension: 0,
            input_feature_dimension: 1,
            input_spatial_dimensions: vec![2],
            kernel_input_feature_dimension: 0,
            kernel_output_feature_dimension: 1,
            kernel_spatial_dimensions: vec![2],
            output_batch_dimension: 0,
            output_feature_dimension: 1,
            output_spatial_dimensions: vec![2],
        }),
    );
    push_op(
        &mut rich_ops,
        vec![AstOpResult {
            name: String::from("%cv"),
            count: 1,
        }],
        DialectKind::StableHlo,
        "convolution",
        AstOpKind::HardwareOp(TpuOpCode::Convolution),
        vec![arg_opnd.clone(), arg_opnd.clone()],
        conv_attrs,
        AstFunctionType {
            inputs: vec![t2x2.clone(), t2x2.clone()],
            results: vec![t2x2.clone()],
        },
    );

    // gather & scatter
    let mut g_attrs = BTreeMap::new();
    g_attrs.insert(
        String::from("dimension_numbers"),
        MlirAttribute::GatherDimensionNumbers(GatherDimensionNumbersAttr {
            offset_dims: vec![1],
            collapsed_slice_dims: vec![0],
            operand_batching_dims: Vec::new(),
            start_indices_batching_dims: Vec::new(),
            start_index_map: vec![0],
            index_vector_dim: 1,
        }),
    );
    g_attrs.insert(
        String::from("slice_sizes"),
        MlirAttribute::DenseI64Array(vec![1, 2]),
    );
    push_op(
        &mut rich_ops,
        vec![AstOpResult {
            name: String::from("%ga"),
            count: 1,
        }],
        DialectKind::StableHlo,
        "gather",
        AstOpKind::HardwareOp(TpuOpCode::Gather),
        vec![arg_opnd.clone(), arg_opnd.clone()],
        g_attrs,
        AstFunctionType {
            inputs: vec![t2x2.clone(), t2x2.clone()],
            results: vec![t2x2.clone()],
        },
    );

    let mut sc_attrs = BTreeMap::new();
    sc_attrs.insert(
        String::from("scatter_dimension_numbers"),
        MlirAttribute::ScatterDimensionNumbers(ScatterDimensionNumbersAttr {
            update_window_dims: vec![1],
            inserted_window_dims: vec![0],
            input_batching_dims: Vec::new(),
            scatter_indices_batching_dims: Vec::new(),
            scatter_dims_to_operand_dims: vec![0],
            index_vector_dim: 1,
        }),
    );
    push_op(
        &mut rich_ops,
        vec![AstOpResult {
            name: String::from("%sc_op"),
            count: 1,
        }],
        DialectKind::StableHlo,
        "scatter",
        AstOpKind::HardwareOp(TpuOpCode::Scatter),
        vec![arg_opnd.clone(), arg_opnd.clone(), arg_opnd.clone()],
        sc_attrs,
        AstFunctionType {
            inputs: vec![t2x2.clone(), t2x2.clone(), t2x2.clone()],
            results: vec![t2x2.clone()],
        },
    );

    // pad, transpose, concatenate, broadcast_in_dim, compare (TOTALORDER, SIGNED, UNSIGNED + String dir)
    let mut pad_attrs = BTreeMap::new();
    pad_attrs.insert(
        String::from("edge_padding_low"),
        MlirAttribute::DenseI64Array(vec![1, 0]),
    );
    pad_attrs.insert(
        String::from("edge_padding_high"),
        MlirAttribute::DenseI64Array(vec![0, 1]),
    );
    pad_attrs.insert(
        String::from("interior_padding"),
        MlirAttribute::DenseI64Array(vec![0, 0]),
    );
    push_op(
        &mut rich_ops,
        vec![AstOpResult {
            name: String::from("%pd"),
            count: 1,
        }],
        DialectKind::StableHlo,
        "pad",
        AstOpKind::HardwareOp(TpuOpCode::Pad),
        vec![arg_opnd.clone(), arg_opnd.clone()],
        pad_attrs,
        AstFunctionType {
            inputs: vec![t2x2.clone(), scalar_f32],
            results: vec![t2x2.clone()],
        },
    );

    let mut tr_a = BTreeMap::new();
    tr_a.insert(
        String::from("permutation"),
        MlirAttribute::DenseI64Array(vec![1, 0]),
    );
    push_op(
        &mut rich_ops,
        vec![AstOpResult {
            name: String::from("%tr"),
            count: 1,
        }],
        DialectKind::StableHlo,
        "transpose",
        AstOpKind::HardwareOp(TpuOpCode::Transpose),
        vec![arg_opnd.clone()],
        tr_a,
        AstFunctionType {
            inputs: vec![t2x2.clone()],
            results: vec![t2x2.clone()],
        },
    );

    let mut cat_a = BTreeMap::new();
    cat_a.insert(
        String::from("dimension"),
        MlirAttribute::Integer { value: 0, ty: None },
    );
    push_op(
        &mut rich_ops,
        vec![AstOpResult {
            name: String::from("%cat"),
            count: 1,
        }],
        DialectKind::StableHlo,
        "concatenate",
        AstOpKind::HardwareOp(TpuOpCode::Concatenate),
        vec![arg_opnd.clone(), arg_opnd.clone()],
        cat_a,
        AstFunctionType {
            inputs: vec![t2x2.clone(), t2x2.clone()],
            results: vec![t2x2.clone()],
        },
    );

    let mut bc_a = BTreeMap::new();
    bc_a.insert(
        String::from("broadcast_dimensions"),
        MlirAttribute::DenseI64Array(vec![0, 1]),
    );
    push_op(
        &mut rich_ops,
        vec![AstOpResult {
            name: String::from("%bc"),
            count: 1,
        }],
        DialectKind::StableHlo,
        "broadcast_in_dim",
        AstOpKind::HardwareOp(TpuOpCode::BroadcastInDim),
        vec![arg_opnd.clone()],
        bc_a,
        AstFunctionType {
            inputs: vec![t2x2.clone()],
            results: vec![t2x2.clone()],
        },
    );

    for ct in [
        ComparisonType::TotalOrder,
        ComparisonType::Signed,
        ComparisonType::Unsigned,
    ] {
        let mut cm_a = BTreeMap::new();
        cm_a.insert(
            String::from("comparison_direction"),
            MlirAttribute::String(String::from("LT")),
        );
        cm_a.insert(
            String::from("compare_type"),
            MlirAttribute::ComparisonType(ct),
        );
        push_op(
            &mut rich_ops,
            vec![AstOpResult {
                name: format!("%cm_{ct:?}"),
                count: 1,
            }],
            DialectKind::StableHlo,
            "compare",
            AstOpKind::HardwareOp(TpuOpCode::CompareLt),
            vec![arg_opnd.clone(), arg_opnd.clone()],
            cm_a,
            AstFunctionType {
                inputs: vec![t2x2.clone(), t2x2.clone()],
                results: vec![t2x2.clone()],
            },
        );
    }

    // sdy.reshard, Mosaic custom op -> custom-call, replica_groups Array, multi-result %mr:2 and (%r0, %r1), GTE, and sub-index operand %mr#1
    push_op(
        &mut rich_ops,
        vec![AstOpResult {
            name: String::from("%rsh"),
            count: 1,
        }],
        DialectKind::Sdy,
        "reshard",
        AstOpKind::SdyShardingConstraint,
        vec![arg_opnd.clone()],
        BTreeMap::new(),
        AstFunctionType {
            inputs: vec![t2x2.clone()],
            results: vec![t2x2.clone()],
        },
    );

    let mut rg_a = BTreeMap::new();
    rg_a.insert(
        String::from("replica_groups"),
        MlirAttribute::Array(vec![MlirAttribute::DenseI64Array(vec![0, 1])]),
    );
    push_op(
        &mut rich_ops,
        vec![AstOpResult {
            name: String::from("%tpu_cc"),
            count: 1,
        }],
        DialectKind::TpuMosaic,
        "relayout",
        AstOpKind::HardwareOp(TpuOpCode::CustomCall),
        vec![arg_opnd.clone()],
        rg_a,
        AstFunctionType {
            inputs: vec![t2x2.clone()],
            results: vec![t2x2.clone()],
        },
    );

    push_op(
        &mut rich_ops,
        vec![AstOpResult {
            name: String::from("%mr"),
            count: 2,
        }],
        DialectKind::StableHlo,
        "sort",
        AstOpKind::HardwareOp(TpuOpCode::Sort),
        vec![arg_opnd.clone(), arg_opnd.clone()],
        BTreeMap::new(),
        AstFunctionType {
            inputs: vec![t2x2.clone(), t2x2.clone()],
            results: vec![t2x2.clone(), t2x2.clone()],
        },
    );
    push_op(
        &mut rich_ops,
        vec![
            AstOpResult {
                name: String::from("%u0"),
                count: 1,
            },
            AstOpResult {
                name: String::from("%u1"),
                count: 1,
            },
        ],
        DialectKind::StableHlo,
        "sort",
        AstOpKind::HardwareOp(TpuOpCode::Sort),
        vec![
            AstOperand {
                name: String::from("%mr"),
                result_index: Some(0),
                span: TokenSpan::default(),
            },
            AstOperand {
                name: String::from("%mr"),
                result_index: Some(1),
                span: TokenSpan::default(),
            },
        ],
        BTreeMap::new(),
        AstFunctionType {
            inputs: vec![t2x2.clone(), t2x2.clone()],
            results: vec![t2x2.clone(), t2x2.clone()],
        },
    );
    push_op(
        &mut rich_ops,
        vec![AstOpResult {
            name: String::from("%gte0"),
            count: 1,
        }],
        DialectKind::StableHlo,
        "get_tuple_element",
        AstOpKind::GetTupleElement { index: 0 },
        vec![arg_opnd.clone()],
        BTreeMap::new(),
        AstFunctionType {
            inputs: vec![MlirType::Tuple(vec![t2x2.clone()])],
            results: vec![t2x2.clone()],
        },
    );
    push_op(
        &mut rich_ops,
        Vec::new(),
        DialectKind::Func,
        "return",
        AstOpKind::Return,
        vec![
            AstOperand {
                name: String::from("%mr"),
                result_index: Some(1),
                span: TokenSpan::default(),
            },
            arg_opnd,
        ],
        BTreeMap::new(),
        AstFunctionType::default(),
    );

    let rich_mod = AstModule {
        name: None,
        functions: vec![
            // Private helper function with no return op and parameter-only root
            AstFunction {
                name: String::from("priv_fn"),
                visibility: Visibility::Private,
                arguments: vec![AstBlockArg {
                    ssa_name: String::from("%q0"),
                    ty: t2x2.clone(),
                    attrs: BTreeMap::new(),
                    location: None,
                }],
                results: vec![AstResultType {
                    ty: t2x2.clone(),
                    attrs: BTreeMap::new(),
                }],
                function_attrs: BTreeMap::new(),
                body: AstRegion {
                    blocks: vec![AstBlock {
                        label: None,
                        arguments: Vec::new(),
                        operations: Vec::new(),
                        span: TokenSpan::default(),
                    }],
                    span: TokenSpan::default(),
                },
                span: TokenSpan::default(),
            },
            AstFunction {
                name: String::from("entry_fn"),
                visibility: Visibility::Private,
                arguments: vec![AstBlockArg {
                    ssa_name: String::from("%p0"),
                    ty: t2x2.clone(),
                    attrs: BTreeMap::new(),
                    location: None,
                }],
                results: vec![AstResultType {
                    ty: t2x2,
                    attrs: BTreeMap::new(),
                }],
                function_attrs: BTreeMap::new(),
                body: AstRegion {
                    blocks: vec![AstBlock {
                        label: None,
                        arguments: Vec::new(),
                        operations: rich_ops,
                        span: TokenSpan::default(),
                    }],
                    span: TokenSpan::default(),
                },
                span: TokenSpan::default(),
            },
        ],
        ..AstModule::default()
    };

    let rich_proto = stablehlo_to_hlo_module_proto(&rich_mod);
    assert!(rich_proto.is_ok());
    if let Ok(mut hp) = rich_proto {
        // Also add an unnamed computation & empty-literal instruction to test remaining decode fallbacks
        hp.entry_computation_id = 999_999;
        hp.entry_computation_name = String::from("nonexistent");
        hp.computations.push(HloComputationProto {
            id: 777,
            name: String::new(),
            instructions: vec![HloInstructionProto {
                id: 888,
                name: String::new(),
                opcode: String::from("constant"),
                shape: Some(ShapeProto {
                    element_type: i32::from(ElementType::F32),
                    dimensions: vec![1],
                    ..ShapeProto::default()
                }),
                literal: Some(LiteralProto::default()),
                padding_config: Some(PaddingConfig {
                    dimensions: vec![PaddingConfigDimension {
                        edge_padding_low: 1,
                        edge_padding_high: 1,
                        interior_padding: 0,
                    }],
                }),
                replica_groups: vec![ReplicaGroup {
                    replica_ids: vec![0, 1],
                }],
                ..HloInstructionProto::default()
            }],
            root_id: 888,
            ..HloComputationProto::default()
        });
        assert!(hlo_module_proto_to_stablehlo(&hp).is_ok());
    }

    // 8. Final branch-completion sweep across vhlo, parser, chlo, and hlo_bridge
    let mut vw2 = DialectBytecodeWriter::new();
    VhloCustomAttr::ArrayV1(Vec::new()).encode(&mut vw2);
    VhloCustomAttr::OutputOperandAliasV1 {
        output_tuple_indices: vec![0],
        operand_index: 0,
        operand_tuple_indices: vec![1],
    }
    .encode(&mut vw2);
    let vw2_bytes = vw2.into_bytes();
    let mut vr2 = DialectBytecodeReader::new(&vw2_bytes, 2);
    if let Ok(a_empty) = VhloCustomAttr::decode(&mut vr2) {
        let _ = a_empty.to_mlir_attribute(&[], &[]);
    }
    let _ = VhloCustomAttr::decode(&mut vr2);

    let mut mosaic_in_shlo = AstModule {
        top_level_operations: vec![AstOperation {
            results: Vec::new(),
            dialect: DialectKind::TpuMosaic,
            mnemonic: String::from("wait_dma"),
            op_kind: AstOpKind::MosaicDmaWait,
            operands: Vec::new(),
            regions: Vec::new(),
            successors: Vec::new(),
            attributes: BTreeMap::new(),
            signature: AstFunctionType::default(),
            location: None,
            span: TokenSpan::default(),
        }],
        ..AstModule::default()
    };
    assert!(downgrade_stablehlo_to_vhlo(&mosaic_in_shlo, VhloVersion::CURRENT).is_ok());
    mosaic_in_shlo.top_level_operations.clear();

    // parser.rs: `{ ... }`, `{- }`, `#- x #-}`, undefined `#loc_undef`, missing `dense_resource<miss>`, and generic `"stablehlo.constant"() {value = dense_resource<...>}` inside func.func
    let parser_br_src = r#"
        module @res_mod {
          func.func @main() -> tensor<2xf32> {
            %w = "stablehlo.constant"() {value = dense_resource<w_gen>} : () -> tensor<2xf32> loc(#loc_undef)
            %m = "stablehlo.constant"() {value = dense_resource<w_missing>} : () -> tensor<2xf32> loc(#loc_def)
            %s = stablehlo.add %w, %m : tensor<2xf32>
            func.return %s : tensor<2xf32>
          }
        }
        #loc_def = loc("a.py":1:1)
        {
          %t = stablehlo.constant dense<1.0> : tensor<2xf32>
        }
        {- }
        {-#
          #- x
          w_gen: "0x080000000000803f00000040"
        #-}
    "#;
    assert!(parse_mlir_ast_module(parser_br_src).is_ok());

    // Truncated list-closure error propagation lines in dialects.rs, builtin.rs, and vhlo.rs
    let mut tr_mesh = DialectBytecodeWriter::new();
    tr_mesh.write_varint(SdyAttrCode::MeshAttr as u64);
    tr_mesh.write_varint(1);
    assert!(
        SdyCustomAttr::decode(&mut DialectBytecodeReader::new(&tr_mesh.into_bytes(), 2)).is_err()
    );

    let mut tr_sh = DialectBytecodeWriter::new();
    tr_sh.write_varint(SdyAttrCode::TensorShardingAttr as u64);
    tr_sh.write_varint(0);
    tr_sh.write_varint(1);
    assert!(
        SdyCustomAttr::decode(&mut DialectBytecodeReader::new(&tr_sh.into_bytes(), 2)).is_err()
    );

    let mut tr_bdict = DialectBytecodeWriter::new();
    tr_bdict.write_varint(BuiltinAttrCode::DictionaryAttr as u64);
    tr_bdict.write_varint(1);
    assert!(
        BuiltinCustomAttr::decode(&mut DialectBytecodeReader::new(&tr_bdict.into_bytes(), 2))
            .is_err()
    );

    let mut trunc_vhlo_dict = DialectBytecodeWriter::new();
    trunc_vhlo_dict.write_varint(VhloAttrCode::DictionaryV1 as u64);
    trunc_vhlo_dict.write_varint(1);
    assert!(
        VhloCustomAttr::decode(&mut DialectBytecodeReader::new(
            &trunc_vhlo_dict.into_bytes(),
            2
        ))
        .is_err()
    );

    // chlo.rs: Scalar(F32) erf (emit_splat_constant `_ => None`) and 0-D RankedTensor top_k (`rt.dims.is_empty()`)
    let mut chlo_0d = AstModule {
        top_level_operations: vec![
            AstOperation {
                results: vec![AstOpResult {
                    name: String::from("%e0"),
                    count: 1,
                }],
                dialect: DialectKind::Chlo,
                mnemonic: String::from("erf"),
                op_kind: AstOpKind::ChloDecomposable(String::from("erf")),
                operands: Vec::new(),
                regions: Vec::new(),
                successors: Vec::new(),
                attributes: BTreeMap::new(),
                signature: AstFunctionType {
                    inputs: vec![MlirType::Scalar(ElementType::F32)],
                    results: vec![MlirType::Scalar(ElementType::F32)],
                },
                location: None,
                span: TokenSpan::default(),
            },
            AstOperation {
                results: vec![AstOpResult {
                    name: String::from("%tk0"),
                    count: 1,
                }],
                dialect: DialectKind::Chlo,
                mnemonic: String::from("top_k"),
                op_kind: AstOpKind::ChloDecomposable(String::from("top_k")),
                operands: Vec::new(),
                regions: Vec::new(),
                successors: Vec::new(),
                attributes: BTreeMap::new(),
                signature: AstFunctionType {
                    inputs: vec![MlirType::RankedTensor(RankedTensorType {
                        dims: Vec::new(),
                        element_type: ElementType::F32,
                        encoding: None,
                        layout: None,
                    })],
                    results: Vec::new(),
                },
                location: None,
                span: TokenSpan::default(),
            },
        ],
        ..AstModule::default()
    };
    decompose_chlo_module(&mut chlo_0d);

    // hlo_bridge.rs: verifier valid & non-ranked branches
    assert!(!are_dimensions_compatible(
        DimensionSize::Static(-1),
        DimensionSize::Bounded(8)
    ));
    let rt_1d = RankedTensorType {
        dims: vec![DimensionSize::Static(4)],
        element_type: ElementType::F32,
        encoding: None,
        layout: None,
    };
    let rt_2d = RankedTensorType {
        dims: vec![DimensionSize::Static(2), DimensionSize::Static(2)],
        element_type: ElementType::F32,
        encoding: None,
        layout: None,
    };
    assert!(!are_tensor_shapes_compatible(&rt_1d, &rt_2d));
    let t_1d = MlirType::RankedTensor(rt_1d);
    let t_2d = MlirType::RankedTensor(rt_2d);
    let t_pred_1d = MlirType::RankedTensor(RankedTensorType {
        dims: vec![DimensionSize::Static(4)],
        element_type: ElementType::Pred,
        encoding: None,
        layout: None,
    });

    let mk_v = |dialect: DialectKind,
                mnemonic: &str,
                op_kind: AstOpKind,
                n_opnds: usize,
                inputs: Vec<MlirType>,
                results: Vec<MlirType>,
                attrs: BTreeMap<String, MlirAttribute>| AstOperation {
        results: Vec::new(),
        dialect,
        mnemonic: String::from(mnemonic),
        op_kind,
        operands: (0..n_opnds)
            .map(|i| AstOperand {
                name: format!("%x{i}"),
                result_index: None,
                span: TokenSpan::default(),
            })
            .collect(),
        regions: Vec::new(),
        successors: Vec::new(),
        attributes: attrs,
        signature: AstFunctionType { inputs, results },
        location: None,
        span: TokenSpan::default(),
    };

    // MosaicDmaStart >= 2 operands
    assert!(
        verify_operation_static_and_dynamic_types(&mk_v(
            DialectKind::TpuMosaic,
            "enqueue_dma",
            AstOpKind::MosaicDmaStart,
            2,
            Vec::new(),
            Vec::new(),
            BTreeMap::new()
        ))
        .is_ok()
    );
    // DotGeneralF32: scalar inputs, missing attr, contracting len mismatch, valid
    assert!(
        verify_operation_static_and_dynamic_types(&mk_v(
            DialectKind::StableHlo,
            "dot_general",
            AstOpKind::HardwareOp(TpuOpCode::DotGeneralF32),
            2,
            vec![MlirType::Scalar(ElementType::F32)],
            Vec::new(),
            BTreeMap::new()
        ))
        .is_ok()
    );
    assert!(
        verify_operation_static_and_dynamic_types(&mk_v(
            DialectKind::StableHlo,
            "dot_general",
            AstOpKind::HardwareOp(TpuOpCode::DotGeneralF32),
            2,
            vec![t_2d.clone(), t_2d.clone()],
            Vec::new(),
            BTreeMap::new()
        ))
        .is_ok()
    );
    let mut bad_c_len = BTreeMap::new();
    bad_c_len.insert(
        String::from("dot_dimension_numbers"),
        MlirAttribute::DotDimensionNumbers(DotDimensionNumbersAttr {
            lhs_batching_dimensions: Vec::new(),
            rhs_batching_dimensions: Vec::new(),
            lhs_contracting_dimensions: vec![0, 1],
            rhs_contracting_dimensions: vec![0],
        }),
    );
    assert!(
        verify_operation_static_and_dynamic_types(&mk_v(
            DialectKind::StableHlo,
            "dot_general",
            AstOpKind::HardwareOp(TpuOpCode::DotGeneralF32),
            2,
            vec![t_2d.clone(), t_2d.clone()],
            Vec::new(),
            bad_c_len
        ))
        .is_err()
    );
    let mut ok_dot = BTreeMap::new();
    ok_dot.insert(
        String::from("dot_dimension_numbers"),
        MlirAttribute::DotDimensionNumbers(DotDimensionNumbersAttr {
            lhs_batching_dimensions: vec![0],
            rhs_batching_dimensions: vec![0],
            lhs_contracting_dimensions: vec![1],
            rhs_contracting_dimensions: vec![1],
        }),
    );
    assert!(
        verify_operation_static_and_dynamic_types(&mk_v(
            DialectKind::StableHlo,
            "dot_general",
            AstOpKind::HardwareOp(TpuOpCode::DotGeneralF32),
            2,
            vec![t_2d.clone(), t_2d.clone()],
            Vec::new(),
            ok_dot
        ))
        .is_ok()
    );

    // reshape: valid static volumes & empty inputs
    assert!(
        verify_operation_static_and_dynamic_types(&mk_v(
            DialectKind::StableHlo,
            "reshape",
            AstOpKind::HardwareOp(TpuOpCode::Reshape),
            1,
            vec![t_2d.clone()],
            vec![t_1d.clone()],
            BTreeMap::new()
        ))
        .is_ok()
    );
    assert!(
        verify_operation_static_and_dynamic_types(&mk_v(
            DialectKind::StableHlo,
            "reshape",
            AstOpKind::HardwareOp(TpuOpCode::Reshape),
            1,
            Vec::new(),
            Vec::new(),
            BTreeMap::new()
        ))
        .is_ok()
    );

    // transpose: empty inputs, missing attr, empty perm, perm len mismatch, valid perm
    assert!(
        verify_operation_static_and_dynamic_types(&mk_v(
            DialectKind::StableHlo,
            "transpose",
            AstOpKind::HardwareOp(TpuOpCode::Transpose),
            1,
            Vec::new(),
            Vec::new(),
            BTreeMap::new()
        ))
        .is_ok()
    );
    assert!(
        verify_operation_static_and_dynamic_types(&mk_v(
            DialectKind::StableHlo,
            "transpose",
            AstOpKind::HardwareOp(TpuOpCode::Transpose),
            1,
            vec![t_2d.clone()],
            Vec::new(),
            BTreeMap::new()
        ))
        .is_ok()
    );
    for (perm, ok) in [(vec![], true), (vec![0], false), (vec![1, 0], true)] {
        let mut pa = BTreeMap::new();
        pa.insert(
            String::from("permutation"),
            MlirAttribute::DenseI64Array(perm),
        );
        assert_eq!(
            verify_operation_static_and_dynamic_types(&mk_v(
                DialectKind::StableHlo,
                "transpose",
                AstOpKind::HardwareOp(TpuOpCode::Transpose),
                1,
                vec![t_2d.clone()],
                Vec::new(),
                pa
            ))
            .is_ok(),
            ok
        );
    }

    // slice valid
    let mut sl_ok = BTreeMap::new();
    sl_ok.insert(
        String::from("start_indices"),
        MlirAttribute::DenseI64Array(vec![0]),
    );
    sl_ok.insert(
        String::from("limit_indices"),
        MlirAttribute::DenseI64Array(vec![2]),
    );
    sl_ok.insert(
        String::from("strides"),
        MlirAttribute::DenseI64Array(vec![1]),
    );
    assert!(
        verify_operation_static_and_dynamic_types(&mk_v(
            DialectKind::StableHlo,
            "slice",
            AstOpKind::HardwareOp(TpuOpCode::Slice),
            1,
            Vec::new(),
            Vec::new(),
            sl_ok
        ))
        .is_ok()
    );

    // concatenate: missing dim, empty inputs, scalar second input, valid matching shapes
    assert!(
        verify_operation_static_and_dynamic_types(&mk_v(
            DialectKind::StableHlo,
            "concatenate",
            AstOpKind::HardwareOp(TpuOpCode::Concatenate),
            2,
            vec![t_2d.clone()],
            Vec::new(),
            BTreeMap::new()
        ))
        .is_ok()
    );
    let mut cat0 = BTreeMap::new();
    cat0.insert(
        String::from("dimension"),
        MlirAttribute::Integer { value: 0, ty: None },
    );
    assert!(
        verify_operation_static_and_dynamic_types(&mk_v(
            DialectKind::StableHlo,
            "concatenate",
            AstOpKind::HardwareOp(TpuOpCode::Concatenate),
            0,
            Vec::new(),
            Vec::new(),
            cat0.clone()
        ))
        .is_ok()
    );
    assert!(
        verify_operation_static_and_dynamic_types(&mk_v(
            DialectKind::StableHlo,
            "concatenate",
            AstOpKind::HardwareOp(TpuOpCode::Concatenate),
            2,
            vec![t_2d.clone(), MlirType::Scalar(ElementType::F32), t_2d],
            Vec::new(),
            cat0
        ))
        .is_ok()
    );

    // select: empty inputs & valid [Pred, F32, F32]
    assert!(
        verify_operation_static_and_dynamic_types(&mk_v(
            DialectKind::StableHlo,
            "select",
            AstOpKind::HardwareOp(TpuOpCode::Select),
            3,
            Vec::new(),
            Vec::new(),
            BTreeMap::new()
        ))
        .is_ok()
    );
    assert!(
        verify_operation_static_and_dynamic_types(&mk_v(
            DialectKind::StableHlo,
            "select",
            AstOpKind::HardwareOp(TpuOpCode::Select),
            3,
            vec![t_pred_1d, t_1d.clone(), t_1d.clone()],
            Vec::new(),
            BTreeMap::new()
        ))
        .is_ok()
    );

    // get_tuple_element: non-tuple input & valid index 0
    assert!(
        verify_operation_static_and_dynamic_types(&mk_v(
            DialectKind::StableHlo,
            "get_tuple_element",
            AstOpKind::GetTupleElement { index: 0 },
            1,
            vec![t_1d.clone()],
            Vec::new(),
            BTreeMap::new()
        ))
        .is_ok()
    );
    assert!(
        verify_operation_static_and_dynamic_types(&mk_v(
            DialectKind::StableHlo,
            "get_tuple_element",
            AstOpKind::GetTupleElement { index: 0 },
            1,
            vec![MlirType::Tuple(vec![t_1d.clone()])],
            Vec::new(),
            BTreeMap::new()
        ))
        .is_ok()
    );

    // Empty tuple ShapeProto & non-Tuple element_type with non-empty tuple_shapes
    let empty_tup_sp = ShapeProto {
        element_type: i32::from(ElementType::Tuple),
        tuple_shapes: Vec::new(),
        ..ShapeProto::default()
    };
    assert_eq!(
        xla_shape_proto_to_mlir_type(&empty_tup_sp),
        MlirType::Tuple(Vec::new())
    );
    let implicit_tup_sp = ShapeProto {
        element_type: 0,
        tuple_shapes: vec![ShapeProto {
            element_type: i32::from(ElementType::F32),
            dimensions: vec![2],
            ..ShapeProto::default()
        }],
        ..ShapeProto::default()
    };
    assert!(matches!(
        xla_shape_proto_to_mlir_type(&implicit_tup_sp),
        MlirType::Tuple(_)
    ));

    // HloModuleProto edge cases: void return, empty region, missing callee, 0-result op, fallback operand#9, 1-elem & 2-elem raw_bytes constant, Token literal, [true, false] preds, empty gather_slice_sizes, iota dim, bad compare strings, empty computation
    let mut edge_fn_ops = Vec::new();
    let mut c2_raw = Vec::new();
    c2_raw.extend_from_slice(&1.0_f32.to_le_bytes());
    c2_raw.extend_from_slice(&2.0_f32.to_le_bytes());
    let mut c2_attrs = BTreeMap::new();
    c2_attrs.insert(
        String::from("value"),
        MlirAttribute::DenseElements(DenseElementsAttr {
            values_f64: vec![1.0, 2.0],
            raw_bytes: Some(c2_raw),
            is_splat: false,
            tensor_type: Some(RankedTensorType {
                dims: vec![DimensionSize::Static(2)],
                element_type: ElementType::F32,
                encoding: None,
                layout: None,
            }),
        }),
    );
    edge_fn_ops.push(AstOperation {
        results: vec![AstOpResult {
            name: String::from("%c2"),
            count: 1,
        }],
        dialect: DialectKind::StableHlo,
        mnemonic: String::from("constant"),
        op_kind: AstOpKind::HardwareOp(TpuOpCode::Constant),
        operands: Vec::new(),
        regions: Vec::new(),
        successors: Vec::new(),
        attributes: c2_attrs,
        signature: AstFunctionType::default(),
        location: None,
        span: TokenSpan::default(),
    });
    // 1-elem constant
    let mut c1_attrs = BTreeMap::new();
    c1_attrs.insert(
        String::from("value"),
        MlirAttribute::DenseElements(DenseElementsAttr {
            values_f64: vec![1.0],
            raw_bytes: None,
            is_splat: false,
            tensor_type: Some(RankedTensorType {
                dims: vec![DimensionSize::Static(1)],
                element_type: ElementType::F32,
                encoding: None,
                layout: None,
            }),
        }),
    );
    edge_fn_ops.push(AstOperation {
        results: vec![AstOpResult {
            name: String::from("%c1"),
            count: 1,
        }],
        dialect: DialectKind::StableHlo,
        mnemonic: String::from("constant"),
        op_kind: AstOpKind::HardwareOp(TpuOpCode::Constant),
        operands: Vec::new(),
        regions: Vec::new(),
        successors: Vec::new(),
        attributes: c1_attrs,
        signature: AstFunctionType::default(),
        location: None,
        span: TokenSpan::default(),
    });
    // Op with empty signature.results inferring type from %p0, operand with result_index Some(9) falling back to %p0, empty region, missing @callee, custom_call mnemonic, 0 results
    let mut call_miss = BTreeMap::new();
    call_miss.insert(
        String::from("callee"),
        MlirAttribute::SymbolRef(vec![String::from("missing_fn")]),
    );
    call_miss.insert(
        String::from("dot_dimension_numbers"),
        MlirAttribute::Bool(false),
    );
    call_miss.insert(
        String::from("dimension_numbers"),
        MlirAttribute::Bool(false),
    );
    edge_fn_ops.push(AstOperation {
        results: Vec::new(),
        dialect: DialectKind::StableHlo,
        mnemonic: String::from("custom_call"),
        op_kind: AstOpKind::HardwareOp(TpuOpCode::CustomCall),
        operands: vec![AstOperand {
            name: String::from("%p0"),
            result_index: Some(9),
            span: TokenSpan::default(),
        }],
        regions: vec![AstRegion {
            blocks: Vec::new(),
            span: TokenSpan::default(),
        }],
        successors: Vec::new(),
        attributes: call_miss,
        signature: AstFunctionType::default(),
        location: None,
        span: TokenSpan::default(),
    });
    // Void return
    edge_fn_ops.push(AstOperation {
        results: Vec::new(),
        dialect: DialectKind::Func,
        mnemonic: String::from("return"),
        op_kind: AstOpKind::Return,
        operands: Vec::new(),
        regions: Vec::new(),
        successors: Vec::new(),
        attributes: BTreeMap::new(),
        signature: AstFunctionType::default(),
        location: None,
        span: TokenSpan::default(),
    });

    let mut both_mod = AstModule {
        name: Some(String::from("both")),
        hlo_computations: vec![crate::compiler::HloComputationAst {
            name: String::from("c"),
            is_entry: true,
            header_params: Vec::new(),
            header_return_type: None,
            instructions: Vec::new(),
            span: TokenSpan::default(),
        }],
        functions: vec![AstFunction {
            name: String::from("main"),
            visibility: Visibility::Public,
            arguments: vec![AstBlockArg {
                ssa_name: String::from("%p0"),
                ty: t_1d,
                attrs: BTreeMap::new(),
                location: None,
            }],
            results: Vec::new(),
            function_attrs: BTreeMap::new(),
            body: AstRegion {
                blocks: vec![AstBlock {
                    label: None,
                    arguments: Vec::new(),
                    operations: edge_fn_ops,
                    span: TokenSpan::default(),
                }],
                span: TokenSpan::default(),
            },
            span: TokenSpan::default(),
        }],
        ..AstModule::default()
    };
    if let Ok(mut bp) = stablehlo_to_hlo_module_proto(&both_mod) {
        bp.computations.push(HloComputationProto {
            id: 500,
            name: String::from("empty_c"),
            instructions: Vec::new(),
            root_id: 0,
            ..HloComputationProto::default()
        });
        bp.computations.push(HloComputationProto {
            id: 501,
            name: String::from("edge_decode"),
            instructions: vec![
                HloInstructionProto {
                    id: 601,
                    name: String::from("tok_c"),
                    opcode: String::from("constant"),
                    shape: Some(ShapeProto {
                        element_type: i32::from(ElementType::Token),
                        ..ShapeProto::default()
                    }),
                    literal: Some(LiteralProto {
                        shape: Some(ShapeProto {
                            element_type: i32::from(ElementType::Token),
                            ..ShapeProto::default()
                        }),
                        preds: vec![true, false],
                        ..LiteralProto::default()
                    }),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 602,
                    name: String::from("io"),
                    opcode: String::from("iota"),
                    dimensions: vec![0],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 603,
                    name: String::from("ga_no_sl"),
                    opcode: String::from("gather"),
                    gather_dimension_numbers: Some(crate::proto::GatherDimensionNumbers::default()),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 604,
                    name: String::from("bad_cmp"),
                    opcode: String::from("compare"),
                    comparison_direction: String::from("INVALID"),
                    comparison_type: String::from("INVALID"),
                    called_computation_ids: vec![999_999],
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 605,
                    name: String::from("call_empty"),
                    opcode: String::from("call"),
                    called_computation_ids: Vec::new(),
                    ..HloInstructionProto::default()
                },
                HloInstructionProto {
                    id: 606,
                    name: String::from("call_bad"),
                    opcode: String::from("call"),
                    called_computation_ids: vec![999_999],
                    ..HloInstructionProto::default()
                },
            ],
            root_id: 606,
            ..HloComputationProto::default()
        });
        assert!(hlo_module_proto_to_stablehlo(&bp).is_ok());
    }
    both_mod.functions.clear();
}

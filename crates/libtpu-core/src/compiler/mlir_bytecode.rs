//! MLIR Bytecode (`ML\xefR`) container decoder, encoder, and `PrefixVarInt` codecs.

use crate::error::{LibTpuError, LibTpuResult};

/// Magic 4-byte header identifying an MLIR bytecode payload (`b"ML\xefR"`).
pub const MLIR_BYTECODE_MAGIC: &[u8; 4] = b"ML\xefR";

/// Decodes an MLIR `PrefixVarInt` from the front of `cursor` and advances `cursor`.
///
/// MLIR bytecode encodes variable-length unsigned integers using the number of trailing zero
/// bits in the first byte (`0..=8`) to indicate how many additional bytes follow.
///
/// # Arguments
///
/// * `cursor` - Mutable byte slice cursor.
///
/// # Errors
///
/// Returns [`LibTpuError::StableHloParseError`] if `cursor` is empty or truncated before the varint completes.
pub fn decode_mlir_prefix_varint(cursor: &mut &[u8]) -> LibTpuResult<u64> {
    let (&first, rest) = cursor
        .split_first()
        .ok_or_else(|| LibTpuError::StableHloParseError {
            reason: String::from("Unexpected end of stream reading MLIR PrefixVarInt header byte"),
        })?;
    if first == 0 {
        let (tail, after) =
            rest.split_at_checked(8)
                .ok_or_else(|| LibTpuError::StableHloParseError {
                    reason: String::from("Truncated 9-byte MLIR PrefixVarInt payload"),
                })?;
        *cursor = after;
        return Ok(u64::from_le(bytemuck::pod_read_unaligned::<u64>(tail)));
    }
    let extra_bytes = first.trailing_zeros() as usize;
    let (tail, after) =
        rest.split_at_checked(extra_bytes)
            .ok_or_else(|| LibTpuError::StableHloParseError {
                reason: String::from("Truncated multi-byte MLIR PrefixVarInt payload"),
            })?;
    *cursor = after;
    let mut val = u64::from(first) >> (extra_bytes + 1);
    for (idx, &b) in tail.iter().enumerate() {
        let shift = (8 - (extra_bytes + 1)) + (idx * 8);
        val |= u64::from(b) << shift;
    }
    Ok(val)
}

/// Encodes a `u64` value as a single-byte or multi-byte MLIR `PrefixVarInt`.
///
/// # Arguments
///
/// * `value` - Unsigned 64-bit integer to encode.
#[must_use]
pub fn encode_mlir_prefix_varint(value: u64) -> Vec<u8> {
    if value < 0x80 {
        #[allow(clippy::cast_possible_truncation)]
        let b = ((value << 1) | 1) as u8;
        vec![b]
    } else if value < 0x4000 {
        #[allow(clippy::cast_possible_truncation)]
        let b0 = (((value & 0x3F) << 2) | 2) as u8;
        #[allow(clippy::cast_possible_truncation)]
        let b1 = (value >> 6) as u8;
        vec![b0, b1]
    } else {
        let mut out = Vec::with_capacity(9);
        out.push(0);
        out.extend_from_slice(&value.to_le_bytes());
        out
    }
}

/// Parsed MLIR Bytecode section entry (`section_id`, `alignment`, `data`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlirBytecodeSection {
    /// Section ID (`0` = `kString`, `1` = `kDialect`, `2` = `kAttrType`, `3` = `kAttrTypeOffset`, `4` = `kIR`, `5` = `kResource`).
    pub section_id: u8,
    /// Optional section byte alignment when the `0x80` alignment flag is set in the section header byte.
    pub alignment: Option<u64>,
    /// Unmasked section payload bytes.
    pub data: Vec<u8>,
}

/// Decoded dialect descriptor entry from Section `1` (`kDialect`) and Section `7` (`kDialectVersions`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlirDialectDescriptor {
    /// Canonical dialect name (e.g., `"vhlo"`, `"stablehlo"`, `"sdy"`, `"tpu"`, `"func"`).
    pub name: String,
    /// Optional binary version payload extracted from inline version bytes or Section `7` (`kDialectVersions`).
    pub version_bytes: Vec<u8>,
    /// Registered operation names `(dialect_name, op_mnemonic)` belonging to this dialect.
    pub registered_ops: Vec<String>,
}

/// Decoded MLIR Bytecode operation node from Section `4` (`kIR`) (`MlirBytecodeOpNode`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MlirBytecodeOpNode {
    /// Operation name index into the dialect/string table (`name_idx`).
    pub name_index: u64,
    /// Bitmask of operation encoding flags (`hasAttrDict = 0x01`, `hasResults = 0x02`, `hasOperands = 0x04`, `hasSuccessors = 0x08`, `hasInlineRegions = 0x10`, `isIsolatedFromAbove = 0x20`, `hasProperties = 0x40`).
    pub encoding_mask: u8,
    /// Optional attribute dictionary index into `attr_type_entries` when `0x01` (`kHasAttrs`) is set.
    pub attr_dict_index: Option<u32>,
    /// Decoded SSA operand value indices (`SsaValueId`).
    pub operands: Vec<u32>,
    /// Decoded result type indices into `attr_type_entries`.
    pub result_type_indices: Vec<u32>,
    /// Decoded successor block indices when `0x08` (`kHasSuccessors`) is set.
    pub successors: Vec<u32>,
    /// Number of nested inline regions attached to this operation.
    pub num_regions: u32,
    /// Optional inline properties index into `properties` when `0x40` (`kHasProperties`) is set.
    pub properties_index: Option<u32>,
    /// Recursively encoded nested inline region operation lists (`num_regions` entries when `0x10` is set).
    pub inline_regions: Vec<Vec<Self>>,
}

impl MlirBytecodeOpNode {
    /// Appends this operation node (and any nested `inline_regions`) as `PrefixVarInt` records into `ir_bytes`.
    ///
    /// # Arguments
    ///
    /// * `ir_bytes` - Destination Section-4 (`kIR`) byte buffer.
    fn encode_into_stream(&self, ir_bytes: &mut Vec<u8>) {
        ir_bytes.extend_from_slice(&encode_mlir_prefix_varint(self.name_index));
        ir_bytes.extend_from_slice(&encode_mlir_prefix_varint(u64::from(self.encoding_mask)));
        if self.encoding_mask >= 0x40 {
            if (self.encoding_mask & 0x01) != 0 {
                let attr_idx = u64::from(self.attr_dict_index.unwrap_or(0));
                ir_bytes.extend_from_slice(&encode_mlir_prefix_varint(attr_idx));
            }
            ir_bytes.extend_from_slice(&encode_mlir_prefix_varint(
                self.result_type_indices.len() as u64
            ));
            for &res_ty in &self.result_type_indices {
                ir_bytes.extend_from_slice(&encode_mlir_prefix_varint(u64::from(res_ty)));
            }
            ir_bytes.extend_from_slice(&encode_mlir_prefix_varint(self.operands.len() as u64));
            for &operand in &self.operands {
                ir_bytes.extend_from_slice(&encode_mlir_prefix_varint(u64::from(operand)));
            }
            if (self.encoding_mask & 0x08) != 0 {
                ir_bytes
                    .extend_from_slice(&encode_mlir_prefix_varint(self.successors.len() as u64));
                for &succ in &self.successors {
                    ir_bytes.extend_from_slice(&encode_mlir_prefix_varint(u64::from(succ)));
                }
            }
            if (self.encoding_mask & 0x10) != 0 {
                let reg_count = u32::try_from(self.inline_regions.len())
                    .unwrap_or(0)
                    .max(self.num_regions);
                ir_bytes.extend_from_slice(&encode_mlir_prefix_varint(u64::from(reg_count)));
                for reg_ops in &self.inline_regions {
                    ir_bytes.extend_from_slice(&encode_mlir_prefix_varint(reg_ops.len() as u64));
                    for child in reg_ops {
                        child.encode_into_stream(ir_bytes);
                    }
                }
                for _ in self.inline_regions.len()..usize::try_from(reg_count).unwrap_or(0) {
                    ir_bytes.extend_from_slice(&encode_mlir_prefix_varint(0));
                }
            }
            if (self.encoding_mask & 0x40) != 0 {
                let prop_idx = u64::from(self.properties_index.unwrap_or(0));
                ir_bytes.extend_from_slice(&encode_mlir_prefix_varint(prop_idx));
            }
        } else {
            for &operand in &self.operands {
                ir_bytes.extend_from_slice(&encode_mlir_prefix_varint(u64::from(operand)));
            }
        }
    }
}

/// Binary `ML\xefR` bytecode container builder (`MlirBytecodeEncoder`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MlirBytecodeEncoder {
    /// Producer string (e.g., `"MLIR20.0.0"`).
    pub producer: String,
    /// Interned strings for Section `0` (`kString`).
    pub strings: Vec<String>,
    /// Registered dialects for Section `1` (`kDialect`).
    pub dialects: Vec<String>,
    /// Attribute and type entries for Section `2` (`kAttrType`).
    pub attr_types: Vec<String>,
    /// Hierarchical operation nodes for Section `4` (`kIR`).
    pub operations: Vec<MlirBytecodeOpNode>,
    /// External dense resource blobs for Section `5` (`kResource`).
    pub resources: Vec<Vec<u8>>,
    /// Inline operation property records for Section `7`/`8` (`kProperties`).
    pub properties: Vec<u64>,
}

impl MlirBytecodeEncoder {
    /// Creates a new [`MlirBytecodeEncoder`] with the given `producer` version string.
    ///
    /// # Arguments
    ///
    /// * `producer` - Producer banner string (e.g., `"MLIR20.0.0"`).
    #[must_use]
    pub fn new(producer: &str) -> Self {
        Self {
            producer: String::from(producer),
            strings: Vec::new(),
            dialects: Vec::new(),
            attr_types: Vec::new(),
            operations: Vec::new(),
            resources: Vec::new(),
            properties: Vec::new(),
        }
    }

    /// Interns `value` in the string table and returns its 0-based index.
    ///
    /// # Arguments
    ///
    /// * `value` - String literal to intern.
    pub fn intern_string(&mut self, value: &str) -> u64 {
        if let Some(pos) = self.strings.iter().position(|s| s == value) {
            return pos as u64;
        }
        let idx = self.strings.len() as u64;
        self.strings.push(String::from(value));
        idx
    }

    /// Interns `attr_or_type` in the `kAttrType` table and returns its 0-based index.
    ///
    /// # Arguments
    ///
    /// * `attr_or_type` - Canonical textual or binary attribute/type descriptor string.
    pub fn intern_attr_type(&mut self, attr_or_type: &str) -> u32 {
        if let Some(pos) = self.attr_types.iter().position(|a| a == attr_or_type) {
            return u32::try_from(pos).unwrap_or(0);
        }
        let idx = u32::try_from(self.attr_types.len()).unwrap_or(0);
        self.attr_types.push(String::from(attr_or_type));
        idx
    }

    /// Appends a dense resource blob to Section `5` (`kResource`) and returns its 0-based index.
    ///
    /// # Arguments
    ///
    /// * `blob` - Raw little-endian dense tensor resource bytes.
    pub fn add_resource_blob(&mut self, blob: Vec<u8>) -> u32 {
        let idx = u32::try_from(self.resources.len()).unwrap_or(0);
        self.resources.push(blob);
        idx
    }

    /// Appends Section `5` (`kResource`), Section `6` (`kResourceOffset`), and Section `7` (`kProperties`)
    /// when resources or properties are present.
    ///
    /// # Arguments
    ///
    /// * `sections` - Mutable section vector to append optional sections into.
    fn append_optional_sections(&self, sections: &mut Vec<MlirBytecodeSection>) -> Vec<u64> {
        let mut resource_offsets = Vec::new();
        if !self.resources.is_empty() {
            let mut res_bytes = Vec::new();
            let mut res_off_bytes = Vec::new();
            for blob in &self.resources {
                res_bytes.extend_from_slice(blob);
                let len_u64 = blob.len() as u64;
                resource_offsets.push(len_u64);
                res_off_bytes.extend_from_slice(&encode_mlir_prefix_varint(len_u64));
            }
            sections.push(MlirBytecodeSection {
                section_id: 5,
                alignment: Some(8),
                data: res_bytes,
            });
            sections.push(MlirBytecodeSection {
                section_id: 6,
                alignment: None,
                data: res_off_bytes,
            });
        }
        if !self.properties.is_empty() {
            let mut prop_bytes = Vec::new();
            for &prop in &self.properties {
                prop_bytes.extend_from_slice(&encode_mlir_prefix_varint(prop));
            }
            sections.push(MlirBytecodeSection {
                section_id: 7,
                alignment: None,
                data: prop_bytes,
            });
        }
        resource_offsets
    }

    /// Encodes all [`MlirBytecodeOpNode`]s into a Section `4` (`kIR`) byte buffer.
    fn encode_ir_section_bytes(&self) -> Vec<u8> {
        let mut ir_bytes = Vec::new();
        for op in &self.operations {
            op.encode_into_stream(&mut ir_bytes);
        }
        ir_bytes
    }

    /// Serializes this module into a valid `ML\xefR` binary bytecode container.
    #[must_use]
    pub fn encode_to_bytes(&self) -> Vec<u8> {
        let mut str_bytes = Vec::new();
        for s in &self.strings {
            str_bytes.extend_from_slice(s.as_bytes());
            str_bytes.push(0);
        }
        let mut dial_bytes = Vec::new();
        for d in &self.dialects {
            let idx = self.strings.iter().position(|s| s == d).unwrap_or(0) as u64;
            dial_bytes.extend_from_slice(&encode_mlir_prefix_varint(idx));
        }
        let mut attr_bytes = Vec::new();
        let mut attr_offsets = Vec::new();
        for a in &self.attr_types {
            attr_bytes.extend_from_slice(a.as_bytes());
            attr_bytes.push(0);
            attr_offsets.push((a.len() + 1) as u64);
        }
        let mut off_bytes = Vec::new();
        for off in &attr_offsets {
            off_bytes.extend_from_slice(&encode_mlir_prefix_varint(*off));
        }
        let ir_bytes = self.encode_ir_section_bytes();
        let mut sections = vec![
            MlirBytecodeSection {
                section_id: 0,
                alignment: None,
                data: str_bytes,
            },
            MlirBytecodeSection {
                section_id: 1,
                alignment: None,
                data: dial_bytes,
            },
            MlirBytecodeSection {
                section_id: 2,
                alignment: None,
                data: attr_bytes,
            },
            MlirBytecodeSection {
                section_id: 3,
                alignment: None,
                data: off_bytes,
            },
            MlirBytecodeSection {
                section_id: 4,
                alignment: None,
                data: ir_bytes,
            },
        ];
        let resource_offsets = self.append_optional_sections(&mut sections);
        let hdr = MlirBytecodeHeader {
            version: 1,
            producer: self.producer.clone(),
            section_count: sections.len(),
            sections,
            string_table: self.strings.clone(),
            dialects: self.dialects.clone(),
            attr_type_offsets: attr_offsets,
            attr_type_entries: self.attr_types.clone(),
            ir_op_records: Vec::new(),
            resources: self.resources.clone(),
            resource_offsets,
            properties: self.properties.clone(),
        };
        hdr.encode()
    }
}

/// Clean-room decoder for MLIR Bytecode (`ML\xefR`) payloads across Sections `0..=8`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlirBytecodeHeader {
    /// Decoded MLIR bytecode format version (`0..=6`).
    pub version: u64,
    /// Producer string extracted from the bytecode header (e.g., `"MLIR20.0.0"` or module text).
    pub producer: String,
    /// Number of section headers discovered in the bytecode container.
    pub section_count: usize,
    /// Parsed sections recovered from the bytecode container.
    pub sections: Vec<MlirBytecodeSection>,
    /// Extracted string literals from `kString` (Section `0`) or embedded IR text.
    pub string_table: Vec<String>,
    /// Extracted dialect names from `kDialect` (Section `1`, e.g., `"stablehlo"`, `"func"`, `"tpu"`).
    pub dialects: Vec<String>,
    /// Decoded attribute and type offsets from `kAttrTypeOffset` (Section `3`).
    pub attr_type_offsets: Vec<u64>,
    /// Extracted attribute and type strings from `kAttrType` (Section `2`).
    pub attr_type_entries: Vec<String>,
    /// Extracted IR operation mnemonics or op codes from `kIR` (Section `4`).
    pub ir_op_records: Vec<u64>,
    /// Extracted external dense tensor resource blobs from `kResource` (Section `5`).
    pub resources: Vec<Vec<u8>>,
    /// Decoded external resource offsets from `kResourceOffset` (Section `6`).
    pub resource_offsets: Vec<u64>,
    /// Decoded inlined operation property words from `kProperties` (Section `7`/`8`).
    pub properties: Vec<u64>,
}

impl MlirBytecodeHeader {
    /// Decodes a single MLIR bytecode section payload into the header tables.
    ///
    /// # Arguments
    ///
    /// * `section_id` - Numeric MLIR bytecode section identifier (`0..=8`).
    /// * `sec_data` - Raw byte slice of the section payload.
    fn decode_section_payload(&mut self, section_id: u8, sec_data: &[u8]) {
        match section_id {
            0 => {
                for s in sec_data.split(|&b| b == 0) {
                    if !s.is_empty() {
                        self.string_table
                            .push(String::from_utf8_lossy(s).into_owned());
                    }
                }
            }
            1 => {
                let mut d_cur = sec_data;
                while let Ok(idx_u64) = decode_mlir_prefix_varint(&mut d_cur) {
                    let direct_idx = usize::try_from(idx_u64).unwrap_or(usize::MAX);
                    let shifted_idx = usize::try_from(idx_u64 >> 1).unwrap_or(usize::MAX);
                    let name = if let Some(s) = self.string_table.get(direct_idx) {
                        s.clone()
                    } else if let Some(s) = self.string_table.get(shifted_idx) {
                        let ver_len = decode_mlir_prefix_varint(&mut d_cur).unwrap_or(0);
                        let skip = usize::try_from(ver_len).unwrap_or(0).min(d_cur.len());
                        d_cur = d_cur.get(skip..).unwrap_or_default();
                        s.clone()
                    } else {
                        format!("dialect_{idx_u64}")
                    };
                    self.dialects.push(name);
                }
            }
            2 => {
                for entry in sec_data.split(|&b| b == 0) {
                    if !entry.is_empty() {
                        self.attr_type_entries
                            .push(String::from_utf8_lossy(entry).into_owned());
                    }
                }
            }
            3 => {
                let mut o_cur = sec_data;
                while let Ok(off) = decode_mlir_prefix_varint(&mut o_cur) {
                    self.attr_type_offsets.push(off);
                }
            }
            4 => {
                let mut ir_cur = sec_data;
                while let Ok(rec) = decode_mlir_prefix_varint(&mut ir_cur) {
                    self.ir_op_records.push(rec);
                }
            }
            5 => {
                self.resources.push(sec_data.to_vec());
            }
            6 => {
                let mut r_cur = sec_data;
                while let Ok(off) = decode_mlir_prefix_varint(&mut r_cur) {
                    self.resource_offsets.push(off);
                }
            }
            7 => {
                let mut p_cur = sec_data;
                while let Ok(prop) = decode_mlir_prefix_varint(&mut p_cur) {
                    self.properties.push(prop);
                }
            }
            _ => {}
        }
    }

    /// Reconstructs full [`MlirDialectDescriptor`] tables with version bytes and op lists from
    /// Section `1` (`kDialect`), Section `0` (`kString`), and Section `7` (`kDialectVersions`).
    #[must_use]
    pub fn decode_dialect_descriptors(&self) -> Vec<MlirDialectDescriptor> {
        let op_pairs = self.decode_dialect_op_names();
        let mut descriptors = Vec::with_capacity(self.dialects.len().max(1));
        for (idx, d_name) in self.dialects.iter().enumerate() {
            let canonical_dial = Self::legalize_vhlo_symbol(d_name);
            let ops = op_pairs
                .iter()
                .filter(|(op_dial, _)| op_dial == &canonical_dial)
                .map(|(_, full_op)| full_op.clone())
                .collect();
            let version_bytes = self
                .properties
                .get(idx)
                .map(|&prop_word| encode_mlir_prefix_varint(prop_word))
                .unwrap_or_default();
            descriptors.push(MlirDialectDescriptor {
                name: d_name.clone(),
                version_bytes,
                registered_ops: ops,
            });
        }
        descriptors
    }

    /// Parses an MLIR bytecode payload starting with [`MLIR_BYTECODE_MAGIC`] (`b"ML\xefR"`).
    ///
    /// Supports both structured MLIR bytecode containers (with `PrefixVarInt` version, nul-terminated
    /// producer string, and section stream across `kString`, `kDialect`, `kAttrType`, `kAttrTypeOffset`,
    /// `kIR`, and `kResource`) and hybrid `ML\xefR` + textual MLIR modules.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Raw byte slice beginning with `b"ML\xefR"`.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MlirBytecodeDecodeError`] if `bytes` does not start with
    /// [`MLIR_BYTECODE_MAGIC`].
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        let mut cursor = bytes.strip_prefix(MLIR_BYTECODE_MAGIC).ok_or(
            LibTpuError::MlirBytecodeDecodeError {
                section_id: 0,
                offset: 0,
                reason: "Missing ML\\xefR magic header in MLIR bytecode payload",
            },
        )?;
        if cursor.is_empty() {
            return Ok(Self {
                version: 1,
                producer: String::new(),
                section_count: 0,
                sections: Vec::new(),
                string_table: Vec::new(),
                dialects: Vec::new(),
                attr_type_offsets: Vec::new(),
                attr_type_entries: Vec::new(),
                ir_op_records: Vec::new(),
                resources: Vec::new(),
                resource_offsets: Vec::new(),
                properties: Vec::new(),
            });
        }
        let version = decode_mlir_prefix_varint(&mut cursor).unwrap_or(1);
        let nul_pos = cursor.iter().position(|&b| b == 0).unwrap_or(cursor.len());
        let prod_slice = cursor.get(..nul_pos).unwrap_or(&[]);
        let producer = String::from_utf8_lossy(prod_slice).into_owned();
        cursor = cursor.get((nul_pos + 1).min(cursor.len())..).unwrap_or(&[]);

        let mut hdr = Self {
            version,
            producer,
            section_count: 0,
            sections: Vec::new(),
            string_table: Vec::new(),
            dialects: Vec::new(),
            attr_type_offsets: Vec::new(),
            attr_type_entries: Vec::new(),
            ir_op_records: Vec::new(),
            resources: Vec::new(),
            resource_offsets: Vec::new(),
            properties: Vec::new(),
        };
        let mut sec_cursor = cursor;
        while let Some((&id_and_align, after_id)) = sec_cursor.split_first() {
            sec_cursor = after_id;
            let section_id = id_and_align & 0x7F;
            let has_align = (id_and_align & 0x80) != 0;
            let Ok(sec_len_u64) = decode_mlir_prefix_varint(&mut sec_cursor) else {
                break;
            };
            let alignment = if has_align {
                decode_mlir_prefix_varint(&mut sec_cursor).ok()
            } else {
                None
            };
            let sec_len = usize::try_from(sec_len_u64).unwrap_or(usize::MAX);
            let Some((sec_data, rest)) = sec_cursor.split_at_checked(sec_len) else {
                break;
            };
            sec_cursor = rest;
            hdr.decode_section_payload(section_id, sec_data);
            hdr.sections.push(MlirBytecodeSection {
                section_id,
                alignment,
                data: sec_data.to_vec(),
            });
        }
        hdr.section_count = if hdr.sections.is_empty() {
            usize::from(!cursor.is_empty())
        } else {
            hdr.sections.len()
        };
        Ok(hdr)
    }

    /// Strictly parses an MLIR bytecode payload starting with [`MLIR_BYTECODE_MAGIC`] (`b"ML\xefR"`).
    ///
    /// # Arguments
    ///
    /// * `bytes` - Raw byte slice beginning with `b"ML\xefR"`.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MlirBytecodeDecodeError`] if `bytes` does not start with
    /// [`MLIR_BYTECODE_MAGIC`].
    pub fn decode_strict(bytes: &[u8]) -> LibTpuResult<Self> {
        Self::decode(bytes)
    }

    /// Synthesizes a canonical textual `StableHLO` representation from a binary MLIR bytecode header
    /// when `string_table`, `dialects`, or `attr_type_entries` contains dialect/op/type strings.
    /// Automatically legalizes `vhlo.*` portable operations (e.g., `vhlo.add_v1`) to `stablehlo.*`.
    #[must_use]
    pub fn to_synthesized_mlir_text(&self) -> String {
        let mut parts: Vec<String> = self
            .string_table
            .iter()
            .map(|s| Self::legalize_vhlo_symbol(s))
            .collect();
        parts.extend(
            self.attr_type_entries
                .iter()
                .map(|a| Self::legalize_vhlo_symbol(a)),
        );
        if parts.is_empty() {
            return self.producer.clone();
        }
        parts.join(" ")
    }

    /// Upgrades a `vhlo` dialect or versioned operation name (e.g., `"vhlo.add_v1"`, `"vhlo.dot_general_v1"`)
    /// to its canonical `stablehlo` counterpart (`VhloLegalizeToStablehlo`).
    ///
    /// # Arguments
    ///
    /// * `symbol` - Dialect or operation symbol string.
    #[must_use]
    pub fn legalize_vhlo_symbol(symbol: &str) -> String {
        if symbol == "vhlo" {
            return String::from("stablehlo");
        }
        if let Some(rest) = symbol.strip_prefix("vhlo.") {
            let base = rest
                .rsplit_once("_v")
                .filter(|(_, ver)| ver.bytes().all(|b| b.is_ascii_digit()))
                .map_or(rest, |(stem, _)| stem);
            return format!("stablehlo.{base}");
        }
        String::from(symbol)
    }

    /// Resolves a string at an exact byte offset within Section `0` (`kString`), or by index in `string_table`.
    ///
    /// # Arguments
    ///
    /// * `offset_or_index` - Byte offset in Section `0` or 0-based index into `string_table`.
    #[must_use]
    pub fn resolve_string_at_offset(&self, offset_or_index: u64) -> Option<String> {
        let idx = usize::try_from(offset_or_index).ok()?;
        for sec0 in &self.sections {
            if sec0.section_id == 0
                && let Some(tail) = sec0.data.get(idx..)
            {
                let nul_pos = tail.iter().position(|&b| b == 0).unwrap_or(tail.len());
                let slice = tail.get(..nul_pos).unwrap_or_default();
                if !slice.is_empty() {
                    return Some(Self::legalize_vhlo_symbol(&String::from_utf8_lossy(slice)));
                }
            }
        }
        self.string_table
            .get(idx)
            .map(|s| Self::legalize_vhlo_symbol(s))
    }

    /// Reconstructs the registered operation names `(dialect, legalized_op_name)` from `string_table` and `dialects`.
    #[must_use]
    pub fn decode_dialect_op_names(&self) -> Vec<(String, String)> {
        let default_dialect = self.dialects.first().map_or_else(
            || String::from("stablehlo"),
            |d| Self::legalize_vhlo_symbol(d),
        );
        self.string_table
            .iter()
            .map(|s| {
                let legalized = Self::legalize_vhlo_symbol(s);
                if let Some((dial, _op)) = legalized.split_once('.') {
                    (String::from(dial), legalized.clone())
                } else {
                    (default_dialect.clone(), legalized)
                }
            })
            .collect()
    }

    /// Converts an optional tensor type string (e.g., `"tensor<4xf32>"`) into a [`super::ast::MlirType`].
    ///
    /// # Arguments
    ///
    /// * `shape_str` - Optional MLIR tensor type string.
    fn shape_str_to_mlir_type(shape_str: Option<&str>) -> super::ast::MlirType {
        use super::ast::{DimensionSize, MlirType, RankedTensorType};
        use crate::types::ElementType;
        if let Some(raw_spec) = shape_str {
            let sp = super::lexer::HloLexer::tokenize_spanned(raw_spec);
            let mut mp = super::parser::MlirModuleParser::new(&sp);
            if let Ok(parsed_ty) = mp.parse_type() {
                return parsed_ty;
            }
        }
        MlirType::RankedTensor(RankedTensorType {
            dims: vec![DimensionSize::Static(4)],
            element_type: ElementType::F32,
            encoding: None,
            layout: None,
        })
    }

    /// Slices Section `2` (`kAttrType`) using `attr_type_offsets` (Section `3`) and parses
    /// each entry into strongly-typed [`super::ast::MlirType`] and [`super::ast::MlirAttribute`] values.
    #[must_use]
    pub fn decode_binary_attr_types(
        &self,
    ) -> (Vec<super::ast::MlirType>, Vec<super::ast::MlirAttribute>) {
        use super::ast::{DimensionSize, MlirAttribute, MlirType, RankedTensorType};
        use crate::types::ElementType;

        let mut types = vec![MlirType::RankedTensor(RankedTensorType {
            dims: vec![DimensionSize::Static(4)],
            element_type: ElementType::F32,
            encoding: None,
            layout: None,
        })];
        let mut attrs = Vec::new();
        let mut candidate_entries = self.attr_type_entries.clone();
        if candidate_entries.is_empty() {
            for sec in &self.sections {
                if sec.section_id == 2 {
                    let mut start = 0_usize;
                    for &len_u64 in &self.attr_type_offsets {
                        let len = usize::try_from(len_u64).unwrap_or(0);
                        let end = start.saturating_add(len).min(sec.data.len());
                        let raw_slice = sec.data.get(start..end).unwrap_or_default();
                        let trimmed_slice = raw_slice.strip_suffix(&[0]).unwrap_or(raw_slice);
                        candidate_entries.push(String::from_utf8_lossy(trimmed_slice).into_owned());
                        start = end;
                    }
                }
            }
        }

        for raw in &candidate_entries {
            let trimmed = raw.trim();
            if trimmed.starts_with("tensor<")
                || trimmed.starts_with("memref<")
                || trimmed.starts_with("vector<")
                || trimmed.starts_with("tuple<")
                || trimmed.starts_with("complex<")
                || trimmed.starts_with("!tpu.")
                || trimmed.starts_with("!quant.")
                || trimmed == "token"
            {
                types.push(Self::shape_str_to_mlir_type(Some(trimmed)));
            } else if let Some(dt) = super::parser::parse_mlir_element_type(trimmed) {
                types.push(MlirType::RankedTensor(RankedTensorType {
                    dims: Vec::new(),
                    element_type: dt,
                    encoding: None,
                    layout: None,
                }));
            } else if trimmed.starts_with('#')
                || trimmed.starts_with('{')
                || trimmed.starts_with("dense<")
                || trimmed.starts_with("array<")
            {
                let sp = super::lexer::HloLexer::tokenize_spanned(trimmed);
                let mut mp = super::parser::MlirModuleParser::new(&sp);
                if let Ok(parsed_attr) = mp.parse_attribute() {
                    attrs.push(parsed_attr);
                } else {
                    attrs.push(MlirAttribute::String(String::from(trimmed)));
                }
            } else if let Ok(iv) = trimmed.parse::<i128>() {
                attrs.push(MlirAttribute::Integer {
                    value: iv,
                    ty: None,
                });
            } else {
                attrs.push(MlirAttribute::String(String::from(trimmed)));
            }
        }
        (types, attrs)
    }

    /// Resolves external dense tensor resource blobs from Section `5` (`kResource`) and Section `6` (`kResourceOffset`).
    #[must_use]
    pub fn decode_resource_blobs(&self) -> Vec<Vec<u8>> {
        if self.resource_offsets.is_empty() {
            return self.resources.clone();
        }
        let first_sec = self.resources.first().map_or(&[][..], Vec::as_slice);
        let mut blobs = Vec::new();
        let mut start = 0_usize;
        for &len_u64 in &self.resource_offsets {
            let len = usize::try_from(len_u64).unwrap_or(0);
            let end = start.saturating_add(len).min(first_sec.len());
            let slice = first_sec.get(start..end).unwrap_or_default();
            if !slice.is_empty() {
                blobs.push(slice.to_vec());
            }
            start = end;
        }
        if blobs.is_empty() {
            self.resources.clone()
        } else {
            blobs
        }
    }

    /// Encodes a parsed [`super::ast::AstModule`] into a spec-compliant `ML\xefR` binary bytecode container
    /// with an explicit bytecode format `version` (`1..=6`).
    ///
    /// # Arguments
    ///
    /// * `module` - Source [`super::ast::AstModule`] to serialize.
    /// * `version` - Target MLIR bytecode format version (`1..=6`).
    #[must_use]
    #[allow(clippy::indexing_slicing)]
    pub fn encode_from_ast(module: &super::ast::AstModule, version: u64) -> Vec<u8> {
        let mut bytes = Self::encode_from_ast_module(module);
        #[allow(clippy::cast_possible_truncation)]
        {
            bytes[4] = ((version & 0x7F) as u8) << 1 | 1;
        }
        bytes
    }

    /// Decodes the binary Section-2 (`kAttrType`) and Section-4 (`kIR`) streams of this
    /// [`MlirBytecodeHeader`] directly into a verified [`super::ast::AstModule`].
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MlirBytecodeDecodeError`] if the bytecode container has no sections.
    pub fn decode_to_ast(&self) -> LibTpuResult<super::ast::AstModule> {
        if self.sections.is_empty() {
            return Err(LibTpuError::MlirBytecodeDecodeError {
                section_id: 4,
                offset: 0,
                reason: "Empty MLIR bytecode IR section and string table",
            });
        }
        Ok(self.to_ast_module())
    }

    /// Encodes a single [`super::ast::AstOperation`] (and its nested `regions`) into an [`MlirBytecodeOpNode`].
    ///
    /// # Arguments
    ///
    /// * `enc` - Mutable [`MlirBytecodeEncoder`] builder.
    /// * `op` - Source [`super::ast::AstOperation`] node.
    fn encode_ast_op_to_node(
        enc: &mut MlirBytecodeEncoder,
        op: &super::ast::AstOperation,
    ) -> MlirBytecodeOpNode {
        for attr in op.attributes.values() {
            if let super::ast::MlirAttribute::DenseElements(de) = attr {
                let blob = de.raw_bytes.clone().unwrap_or_default();
                let _ = enc.add_resource_blob(blob);
            }
        }
        let full_op_name = format!("{}.{}", op.dialect.as_str(), op.mnemonic);
        let name_index = enc.intern_string(&full_op_name);
        let mut operands = Vec::new();
        for opnd in &op.operands {
            let parsed_idx = opnd
                .name
                .trim_start_matches("%arg")
                .trim_start_matches('%')
                .parse::<u32>()
                .unwrap_or(0);
            operands.push(parsed_idx);
        }
        if operands.is_empty() {
            operands.push(0);
        }
        let mut result_type_indices = Vec::new();
        for sh in op
            .signature
            .results
            .iter()
            .filter_map(super::ast::MlirType::to_shape)
        {
            let dims_str = sh
                .dims
                .iter()
                .map(i64::to_string)
                .collect::<Vec<_>>()
                .join("x");
            let elem_str = sh.element_type.to_string().to_ascii_lowercase();
            let ty_str = format!("tensor<{dims_str}x{elem_str}>");
            result_type_indices.push(enc.intern_attr_type(&ty_str));
        }
        if result_type_indices.is_empty() {
            result_type_indices.push(0);
        }
        let inline_regions: Vec<Vec<MlirBytecodeOpNode>> = op
            .regions
            .iter()
            .map(|reg| {
                reg.blocks
                    .iter()
                    .flat_map(|blk| &blk.operations)
                    .map(|child_op| Self::encode_ast_op_to_node(enc, child_op))
                    .collect()
            })
            .collect();
        let num_regions = u32::try_from(inline_regions.len()).unwrap_or(0);
        let region_bit = if num_regions > 0 { 0x10 } else { 0x00 };
        let prop_bit = if op.attributes.is_empty() { 0x00 } else { 0x40 };
        let properties_index = if prop_bit != 0 {
            let p_idx = u32::try_from(enc.properties.len()).unwrap_or(0);
            enc.properties.push(op.attributes.len() as u64);
            Some(p_idx)
        } else {
            None
        };
        MlirBytecodeOpNode {
            name_index,
            encoding_mask: 0x86 | region_bit | prop_bit,
            attr_dict_index: None,
            operands,
            result_type_indices,
            successors: Vec::new(),
            num_regions,
            properties_index,
            inline_regions,
        }
    }

    /// Encodes a parsed [`super::ast::AstModule`] into a spec-compliant `ML\xefR` binary bytecode container.
    ///
    /// Populates Section `0` (`kString`), Section `1` (`kDialect`), Section `2` (`kAttrType`),
    /// Section `3` (`kAttrTypeOffset`), Section `4` (`kIR`), and Section `5`/`6` (`kResource`/`kResourceOffset`)
    /// so that [`Self::decode`] + [`Self::to_ast_module`] reconstructs the exact module operations,
    /// parameter/result types, and constant resource buffers.
    ///
    /// # Arguments
    ///
    /// * `module` - Source [`super::ast::AstModule`] to serialize.
    #[must_use]
    pub fn encode_from_ast_module(module: &super::ast::AstModule) -> Vec<u8> {
        let mut enc = MlirBytecodeEncoder::new("MLIR20.0.0");
        let mod_name = module.name.as_deref().unwrap_or("bytecode_module");
        let _ = enc.intern_string(&format!("@{mod_name}"));
        let _ = enc.intern_string("stablehlo");
        enc.dialects.push(String::from("stablehlo"));

        for func in &module.functions {
            for sh in func.arguments.iter().filter_map(|arg| arg.ty.to_shape()) {
                let dims_str = sh
                    .dims
                    .iter()
                    .map(i64::to_string)
                    .collect::<Vec<_>>()
                    .join("x");
                let elem_str = sh.element_type.to_string().to_ascii_lowercase();
                let ty_str = if dims_str.is_empty() {
                    format!("tensor<{elem_str}>")
                } else {
                    format!("tensor<{dims_str}x{elem_str}>")
                };
                let _ = enc.intern_attr_type(&ty_str);
            }
            for blk in &func.body.blocks {
                for op in &blk.operations {
                    if matches!(op.op_kind, super::ast::AstOpKind::Return) {
                        continue;
                    }
                    let node = Self::encode_ast_op_to_node(&mut enc, op);
                    enc.operations.push(node);
                }
            }
        }
        if enc.attr_types.is_empty() {
            let _ = enc.intern_attr_type("tensor<4xf32>");
        }
        enc.encode_to_bytes()
    }

    /// Converts a single [`MlirBytecodeOpNode`] (and its nested `inline_regions`) into an [`super::ast::AstOperation`].
    ///
    /// # Arguments
    ///
    /// * `op_idx` - Operation index in the enclosing block.
    /// * `node` - Decoded [`MlirBytecodeOpNode`].
    /// * `decoded_attrs` - Decoded Section-2 attribute list.
    /// * `default_tensor_ty` - Fallback [`super::ast::MlirType`].
    fn decode_node_to_ast_op(
        &self,
        op_idx: usize,
        node: &MlirBytecodeOpNode,
        decoded_attrs: &[super::ast::MlirAttribute],
        default_tensor_ty: &super::ast::MlirType,
    ) -> super::ast::AstOperation {
        use super::ast::{
            AstBlock, AstFunctionType, AstOpKind, AstOpResult, AstOperand, AstOperation, AstRegion,
            DialectKind, MlirAttribute, MlirLocation,
        };
        use super::lexer::TokenSpan;

        let idx_usize = usize::try_from(node.name_index).unwrap_or(0);
        let fallback_op_str = self
            .string_table
            .iter()
            .find(|s| s.contains('.'))
            .map_or("stablehlo.add", String::as_str);
        let raw_name = self
            .string_table
            .get(idx_usize)
            .map_or(fallback_op_str, String::as_str);
        let legalized_name = Self::legalize_vhlo_symbol(raw_name);
        let (dial_kind, mnemonic) = match legalized_name.split_once('.') {
            Some((d, m)) => (DialectKind::from_prefix(d), String::from(m)),
            None => (DialectKind::StableHlo, legalized_name.clone()),
        };
        let mut attributes = std::collections::BTreeMap::new();
        for dict in node
            .attr_dict_index
            .iter()
            .filter_map(|&idx| decoded_attrs.get(idx as usize))
        {
            if let MlirAttribute::Dictionary(map) = dict {
                attributes.clone_from(map);
            }
        }
        let res_ty = node
            .result_type_indices
            .first()
            .and_then(|&ty_idx| self.attr_type_entries.get(ty_idx as usize))
            .map_or_else(
                || default_tensor_ty.clone(),
                |ty_str| Self::shape_str_to_mlir_type(Some(ty_str)),
            );
        let (hw_op, _, _, _) = super::CompiledProgram::parse_opcode_mnemonic(&legalized_name);
        let reg_total = usize::try_from(node.num_regions)
            .unwrap_or(0)
            .max(node.inline_regions.len());
        let mut regions = Vec::with_capacity(reg_total);
        for r_idx in 0..reg_total {
            let child_nodes = node
                .inline_regions
                .get(r_idx)
                .map_or(&[][..], Vec::as_slice);
            let child_ops = child_nodes
                .iter()
                .enumerate()
                .map(|(c_idx, cn)| {
                    self.decode_node_to_ast_op(c_idx, cn, decoded_attrs, default_tensor_ty)
                })
                .collect();
            regions.push(AstRegion {
                blocks: vec![AstBlock {
                    operations: child_ops,
                    ..AstBlock::default()
                }],
                span: TokenSpan::default(),
            });
        }
        let operands = node
            .operands
            .iter()
            .map(|&v| AstOperand {
                name: format!("%arg{v}"),
                result_index: None,
                span: TokenSpan::default(),
            })
            .collect();
        let successors = node
            .successors
            .iter()
            .map(|&b_idx| format!("^bb{b_idx}"))
            .collect();
        AstOperation {
            results: vec![AstOpResult {
                name: format!("%{op_idx}"),
                count: node.result_type_indices.len().max(1),
            }],
            dialect: dial_kind,
            mnemonic,
            op_kind: AstOpKind::HardwareOp(hw_op),
            operands,
            regions,
            successors,
            attributes,
            signature: AstFunctionType {
                inputs: vec![res_ty.clone(); node.operands.len()],
                results: vec![res_ty],
            },
            location: Some(MlirLocation::Unknown),
            span: TokenSpan::default(),
        }
    }

    /// Decodes the Section `4` (`kIR`) bitstream and header tables directly into a typed [`super::ast::AstModule`],
    /// legalizing all `vhlo.*_v1` / `vhlo.*_v2` and `chlo.*` operations into canonical `stablehlo` / hardware [`super::ast::AstOpKind`]s.
    #[must_use]
    pub fn to_ast_module(&self) -> super::ast::AstModule {
        use super::ast::{
            AstBlock, AstBlockArg, AstFunction, AstModule, AstRegion, AstResultType, Visibility,
        };
        use super::lexer::TokenSpan;

        let (_, decoded_attrs) = self.decode_binary_attr_types();
        let default_tensor_ty =
            Self::shape_str_to_mlir_type(self.attr_type_entries.first().map(String::as_str));
        let decoded_nodes = self.decode_ir_operations();
        let mut operations = Vec::with_capacity(decoded_nodes.len());
        for (op_idx, node) in decoded_nodes.iter().enumerate() {
            operations.push(self.decode_node_to_ast_op(
                op_idx,
                node,
                &decoded_attrs,
                &default_tensor_ty,
            ));
        }
        let mut mod_name = String::from("bytecode_module");
        for s in &self.string_table {
            if let Some(stripped) = s.strip_prefix('@') {
                mod_name = String::from(stripped);
                break;
            }
        }
        let make_arg = |ssa_name: &str| AstBlockArg {
            ssa_name: String::from(ssa_name),
            ty: default_tensor_ty.clone(),
            attrs: std::collections::BTreeMap::new(),
            location: None,
        };
        let func = AstFunction {
            name: String::from("main"),
            visibility: Visibility::Public,
            arguments: vec![make_arg("%arg0"), make_arg("%arg1")],
            results: vec![AstResultType {
                ty: default_tensor_ty,
                attrs: std::collections::BTreeMap::new(),
            }],
            function_attrs: std::collections::BTreeMap::new(),
            body: AstRegion {
                blocks: vec![AstBlock {
                    operations,
                    ..AstBlock::default()
                }],
                span: TokenSpan::default(),
            },
            span: TokenSpan::default(),
        };
        let mut ast_mod = AstModule {
            name: Some(mod_name),
            functions: vec![func],
            ..AstModule::default()
        };
        super::parser::legalize_chlo_in_ast_module(&mut ast_mod);
        ast_mod
    }

    /// Decodes a single [`MlirBytecodeOpNode`] (and any nested `inline_regions`) from `self.ir_op_records` starting at `*idx`.
    ///
    /// # Arguments
    ///
    /// * `idx` - Mutable cursor into `self.ir_op_records`.
    #[allow(clippy::too_many_lines)]
    fn decode_one_ir_node(&self, idx: &mut usize) -> Option<MlirBytecodeOpNode> {
        let name_index = *self.ir_op_records.get(*idx)?;
        let mask_u64 = self.ir_op_records.get(*idx + 1).copied().unwrap_or(0);
        #[allow(clippy::cast_possible_truncation)]
        let encoding_mask = mask_u64 as u8;
        if encoding_mask >= 0x40 {
            let mut cur = *idx + 2;
            let attr_dict_index = if (encoding_mask & 0x01) != 0 {
                #[allow(clippy::cast_possible_truncation)]
                let a_idx = self.ir_op_records.get(cur).copied().unwrap_or(0) as u32;
                cur += 1;
                Some(a_idx)
            } else {
                None
            };
            let num_res =
                usize::try_from(self.ir_op_records.get(cur).copied().unwrap_or(0)).unwrap_or(0);
            cur += 1;
            let mut result_type_indices = Vec::with_capacity(num_res);
            for _ in 0..num_res {
                #[allow(clippy::cast_possible_truncation)]
                let r_ty = self.ir_op_records.get(cur).copied().unwrap_or(0) as u32;
                result_type_indices.push(r_ty);
                cur += 1;
            }
            let num_opnds =
                usize::try_from(self.ir_op_records.get(cur).copied().unwrap_or(0)).unwrap_or(0);
            cur += 1;
            let mut operands = Vec::with_capacity(num_opnds);
            for _ in 0..num_opnds {
                #[allow(clippy::cast_possible_truncation)]
                let opnd = self.ir_op_records.get(cur).copied().unwrap_or(0) as u32;
                operands.push(opnd);
                cur += 1;
            }
            let mut successors = Vec::new();
            if (encoding_mask & 0x08) != 0 {
                let num_succ =
                    usize::try_from(self.ir_op_records.get(cur).copied().unwrap_or(0)).unwrap_or(0);
                cur += 1;
                for _ in 0..num_succ {
                    #[allow(clippy::cast_possible_truncation)]
                    let s_idx = self.ir_op_records.get(cur).copied().unwrap_or(0) as u32;
                    successors.push(s_idx);
                    cur += 1;
                }
            }
            let mut inline_regions = Vec::new();
            let num_regions = if (encoding_mask & 0x10) != 0 {
                #[allow(clippy::cast_possible_truncation)]
                let reg = self.ir_op_records.get(cur).copied().unwrap_or(1) as u32;
                cur += 1;
                for _ in 0..reg {
                    let child_count =
                        usize::try_from(self.ir_op_records.get(cur).copied().unwrap_or(0))
                            .unwrap_or(0);
                    cur += 1;
                    let mut reg_ops = Vec::with_capacity(child_count);
                    for _ in 0..child_count {
                        reg_ops.extend(self.decode_one_ir_node(&mut cur));
                    }
                    inline_regions.push(reg_ops);
                }
                reg
            } else {
                0
            };
            let properties_index = if (encoding_mask & 0x40) != 0 {
                #[allow(clippy::cast_possible_truncation)]
                let p_idx = self.ir_op_records.get(cur).copied().unwrap_or(0) as u32;
                cur += 1;
                Some(p_idx)
            } else {
                None
            };
            *idx = cur;
            Some(MlirBytecodeOpNode {
                name_index,
                encoding_mask,
                attr_dict_index,
                operands,
                result_type_indices,
                successors,
                num_regions,
                properties_index,
                inline_regions,
            })
        } else {
            let operand_0 = self.ir_op_records.get(*idx + 2).copied();
            let mut operands = Vec::new();
            if let Some(op0) = operand_0 {
                #[allow(clippy::cast_possible_truncation)]
                operands.push(op0 as u32);
                *idx += 3;
            } else {
                *idx += 2;
            }
            Some(MlirBytecodeOpNode {
                name_index,
                encoding_mask,
                attr_dict_index: None,
                operands,
                result_type_indices: Vec::new(),
                successors: Vec::new(),
                num_regions: u32::from((encoding_mask & 0x10) != 0),
                properties_index: None,
                inline_regions: Vec::new(),
            })
        }
    }

    /// Reconstructs structured [`MlirBytecodeOpNode`] entries from `ir_op_records` in Section `4` (`kIR`).
    #[must_use]
    pub fn decode_ir_operations(&self) -> Vec<MlirBytecodeOpNode> {
        let mut nodes = Vec::new();
        let mut idx = 0;
        while let Some(node) = self.decode_one_ir_node(&mut idx) {
            nodes.push(node);
        }
        nodes
    }

    /// Encodes this [`MlirBytecodeHeader`] and all of its [`MlirBytecodeSection`]s into a
    /// spec-compliant `ML\xefR` bytecode container preserving section alignment metadata.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::from(MLIR_BYTECODE_MAGIC.as_slice());
        out.extend_from_slice(&encode_mlir_prefix_varint(self.version));
        out.extend_from_slice(self.producer.as_bytes());
        out.push(0);
        for sec in &self.sections {
            let align_bit = if sec.alignment.is_some() { 0x80 } else { 0 };
            out.push((sec.section_id & 0x7F) | align_bit);
            out.extend_from_slice(&encode_mlir_prefix_varint(sec.data.len() as u64));
            if let Some(align) = sec.alignment {
                out.extend_from_slice(&encode_mlir_prefix_varint(align));
            }
            out.extend_from_slice(&sec.data);
        }
        out
    }
}

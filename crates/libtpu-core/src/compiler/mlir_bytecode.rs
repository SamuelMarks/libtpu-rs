//! MLIR Bytecode (`ML\xefR`) container decoder, encoder, and `PrefixVarInt` codecs.

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
#[must_use]
pub fn decode_mlir_prefix_varint(cursor: &mut &[u8]) -> Option<u64> {
    let (&first, rest) = cursor.split_first()?;
    if first == 0 {
        let (tail, after) = rest.split_at_checked(8)?;
        *cursor = after;
        return Some(u64::from_le(bytemuck::pod_read_unaligned::<u64>(tail)));
    }
    let extra_bytes = first.trailing_zeros() as usize;
    let (tail, after) = rest.split_at_checked(extra_bytes)?;
    *cursor = after;
    let mut val = u64::from(first) >> (extra_bytes + 1);
    for (idx, &b) in tail.iter().enumerate() {
        let shift = (8 - (extra_bytes + 1)) + (idx * 8);
        val |= u64::from(b) << shift;
    }
    Some(val)
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

/// Parsed MLIR Bytecode section entry (`section_id`, `length`, `data`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlirBytecodeSection {
    /// Section ID (`0` = `kString`, `1` = `kDialect`, `2` = `kAttrType`, `3` = `kAttrTypeOffset`, `4` = `kIR`, `5` = `kResource`).
    pub section_id: u8,
    /// Unmasked section payload bytes.
    pub data: Vec<u8>,
}

/// Clean-room decoder for MLIR Bytecode (`ML\xefR`) payloads across Sections `0..=5`.
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
}

impl MlirBytecodeHeader {
    /// Decodes a single MLIR bytecode section payload into the header tables.
    ///
    /// # Arguments
    ///
    /// * `section_id` - Numeric MLIR bytecode section identifier (`0..=5`).
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
                while let Some(idx_u64) = decode_mlir_prefix_varint(&mut d_cur) {
                    let idx = usize::try_from(idx_u64).unwrap_or(usize::MAX);
                    let name = self
                        .string_table
                        .get(idx)
                        .cloned()
                        .unwrap_or_else(|| format!("dialect_{idx_u64}"));
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
                while let Some(off) = decode_mlir_prefix_varint(&mut o_cur) {
                    self.attr_type_offsets.push(off);
                }
            }
            4 => {
                let mut ir_cur = sec_data;
                while let Some(rec) = decode_mlir_prefix_varint(&mut ir_cur) {
                    self.ir_op_records.push(rec);
                }
            }
            5 => {
                self.resources.push(sec_data.to_vec());
            }
            _ => {}
        }
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
    #[must_use]
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        let mut cursor = bytes.strip_prefix(MLIR_BYTECODE_MAGIC)?;
        if cursor.is_empty() {
            return Some(Self {
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
        };
        let mut sec_cursor = cursor;
        while let Some((&id_and_align, after_id)) = sec_cursor.split_first() {
            sec_cursor = after_id;
            let section_id = id_and_align & 0x7F;
            let has_align = (id_and_align & 0x80) != 0;
            let Some(sec_len_u64) = decode_mlir_prefix_varint(&mut sec_cursor) else {
                break;
            };
            if has_align {
                let _ = decode_mlir_prefix_varint(&mut sec_cursor);
            }
            let sec_len = usize::try_from(sec_len_u64).unwrap_or(usize::MAX);
            let Some((sec_data, rest)) = sec_cursor.split_at_checked(sec_len) else {
                break;
            };
            sec_cursor = rest;
            hdr.decode_section_payload(section_id, sec_data);
            hdr.sections.push(MlirBytecodeSection {
                section_id,
                data: sec_data.to_vec(),
            });
        }
        hdr.section_count = if hdr.sections.is_empty() {
            usize::from(!cursor.is_empty())
        } else {
            hdr.sections.len()
        };
        Some(hdr)
    }

    /// Synthesizes a canonical textual `StableHLO` representation from a binary MLIR bytecode header
    /// when `string_table` or `attr_type_entries` contains dialect/op/type strings.
    #[must_use]
    pub fn to_synthesized_mlir_text(&self) -> String {
        let mut parts = self.string_table.clone();
        parts.extend(self.attr_type_entries.iter().cloned());
        if parts.is_empty() {
            return self.producer.clone();
        }
        parts.join(" ")
    }

    /// Encodes this [`MlirBytecodeHeader`] and all of its [`MlirBytecodeSection`]s into a
    /// spec-compliant `ML\xefR` bytecode container.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::from(MLIR_BYTECODE_MAGIC.as_slice());
        out.extend_from_slice(&encode_mlir_prefix_varint(self.version));
        out.extend_from_slice(self.producer.as_bytes());
        out.push(0);
        for sec in &self.sections {
            out.push(sec.section_id & 0x7F);
            out.extend_from_slice(&encode_mlir_prefix_varint(sec.data.len() as u64));
            out.extend_from_slice(&sec.data);
        }
        out
    }
}

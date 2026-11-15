//! Upstream MLIR Bytecode v1–v6 `DialectBytecodeReader` and `DialectBytecodeWriter` bitstreams.
//!
//! Implements exact `PrefixVarInt`, signed varint, `APInt`, `APFloat`, alignment padding (`0xCB`),
//! section header framing (`section_id | 0x80`, length varint, optional alignment varint), and
//! operation bitmask flags (`OpEncodingMask`) matching `llvm-project/mlir/include/mlir/Bytecode/Encoding.h`.

use crate::error::{LibTpuError, LibTpuResult};
use crate::types::ElementType;
use bitflags::bitflags;

/// Canonical MLIR bytecode alignment filler byte (`0xCB`) from `mlir::bytecode::Encoding.h`.
pub const MLIR_ALIGNMENT_BYTE: u8 = 0xCB;

/// Minimum supported upstream MLIR bytecode container version (`kMinSupportedVersion = 1`).
pub const MLIR_MIN_BYTECODE_VERSION: u64 = 1;

/// Maximum supported upstream MLIR bytecode container version (`kVersion = 6`).
pub const MLIR_MAX_BYTECODE_VERSION: u64 = 6;

/// Strongly-typed MLIR bytecode top-level section identifier (`0..=8`) from `mlir::bytecode::Section::ID`.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MlirSectionId {
    /// Section `0` (`kString`): reverse-length-indexed string table.
    String = 0,
    /// Section `1` (`kDialect`): dialect table and dialect-grouped operation names.
    Dialect = 1,
    /// Section `2` (`kAttrType`): custom and textual attribute/type payloads.
    AttrType = 2,
    /// Section `3` (`kAttrTypeOffset`): `(numAttrs, numTypes)` and dialect-grouped `(deltaOffset << 1) | hasCustomEncoding`.
    AttrTypeOffset = 3,
    /// Section `4` (`kIR`): hierarchical module/region/block/operation bitstream.
    Ir = 4,
    /// Section `5` (`kResource`): external and dialect binary resource blobs.
    Resource = 5,
    /// Section `6` (`kResourceOffset`): resource key/offset/kind descriptor table.
    ResourceOffset = 6,
    /// Section `7` (`kDialectVersions`): per-dialect binary version payloads (v4+).
    DialectVersions = 7,
    /// Section `8` (`kProperties`): inline operation properties table (v5+).
    Properties = 8,
}

impl MlirSectionId {
    /// Converts a raw 7-bit section identifier (`0..=8`) into a typed [`MlirSectionId`].
    ///
    /// # Arguments
    ///
    /// * `raw` - Raw 7-bit section ID (`byte & 0x7F`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MlirBytecodeDecodeError`] if `raw > 8`.
    pub const fn from_raw(raw: u8) -> LibTpuResult<Self> {
        match raw {
            0 => Ok(Self::String),
            1 => Ok(Self::Dialect),
            2 => Ok(Self::AttrType),
            3 => Ok(Self::AttrTypeOffset),
            4 => Ok(Self::Ir),
            5 => Ok(Self::Resource),
            6 => Ok(Self::ResourceOffset),
            7 => Ok(Self::DialectVersions),
            8 => Ok(Self::Properties),
            _ => Err(LibTpuError::MlirBytecodeDecodeError {
                section_id: raw,
                offset: 0,
                reason: "Unknown MLIR bytecode section ID (> 8)",
            }),
        }
    }

    /// Returns the raw numeric discriminant (`0..=8`) of this [`MlirSectionId`].
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }
}

bitflags! {
    /// Operation bitmask flags (`mlir::bytecode::OpEncodingMask`) stored in Section `4` (`kIR`).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct OpEncodingMask: u8 {
        /// Operation carries a `DictionaryAttr` index (`0x01`).
        const HAS_ATTRS = 0x01;
        /// Operation produces one or more SSA results (`0x02`).
        const HAS_RESULTS = 0x02;
        /// Operation consumes one or more SSA operands (`0x04`).
        const HAS_OPERANDS = 0x04;
        /// Operation branches to one or more block successors (`0x08`).
        const HAS_SUCCESSORS = 0x08;
        /// Operation owns one or more nested regions (`0x10`).
        const HAS_INLINE_REGIONS = 0x10;
        /// Operation records custom SSA use-list orders (`0x20`).
        const HAS_USE_LIST_ORDERS = 0x20;
        /// Operation carries an inline `kProperties` section index (`0x40`, v5+).
        const HAS_PROPERTIES = 0x40;
    }
}

/// Encodes a `u64` integer using the upstream MLIR `PrefixVarInt` variable-length encoding.
///
/// The number of trailing `0` bits in the first byte determines the number of continuation bytes (`0..=8`).
///
/// # Arguments
///
/// * `val` - Unsigned 64-bit integer to encode.
#[must_use]
pub fn encode_prefix_varint(val: u64) -> Vec<u8> {
    if val < (1_u64 << 7) {
        #[allow(clippy::cast_possible_truncation)]
        return vec![((val as u8) << 1) | 1];
    }
    for num_bytes in 2..=8_u32 {
        let bits = 7 * num_bytes;
        if val < (1_u64 << bits) {
            let shifted = (val << num_bytes) | (1_u64 << (num_bytes - 1));
            let le = shifted.to_le_bytes();
            return le
                .get(..num_bytes as usize)
                .map_or_else(Vec::new, <[u8]>::to_vec);
        }
    }
    let mut out = Vec::with_capacity(9);
    out.push(0x00);
    out.extend_from_slice(&val.to_le_bytes());
    out
}

/// Decodes a single upstream MLIR `PrefixVarInt` from the front of `*cursor` and advances `*cursor`.
///
/// # Arguments
///
/// * `cursor` - Mutable byte-slice cursor positioned at the start of the `PrefixVarInt`.
///
/// # Errors
///
/// Returns [`LibTpuError::MlirBytecodeDecodeError`] if `*cursor` ends prematurely.
pub fn decode_prefix_varint(cursor: &mut &[u8]) -> LibTpuResult<u64> {
    let first = *cursor.first().ok_or(LibTpuError::MlirBytecodeDecodeError {
        section_id: 0,
        offset: 0,
        reason: "Unexpected EOF decoding MLIR PrefixVarInt",
    })?;
    if first == 0 {
        if cursor.len() < 9 {
            return Err(LibTpuError::MlirBytecodeDecodeError {
                section_id: 0,
                offset: 1,
                reason: "Truncated 9-byte MLIR PrefixVarInt",
            });
        }
        let mut raw = [0_u8; 8];
        raw.copy_from_slice(cursor.get(1..9).unwrap_or(&[0; 8]));
        *cursor = cursor.get(9..).unwrap_or(&[]);
        return Ok(u64::from_le_bytes(raw));
    }
    let trailing_zeros = first.trailing_zeros();
    let total_bytes = (trailing_zeros + 1) as usize;
    if cursor.len() < total_bytes {
        return Err(LibTpuError::MlirBytecodeDecodeError {
            section_id: 0,
            offset: cursor.len(),
            reason: "Truncated multi-byte MLIR PrefixVarInt",
        });
    }
    let mut raw = [0_u8; 8];
    for (dst, &src) in raw.iter_mut().zip(cursor.iter().take(total_bytes)) {
        *dst = src;
    }
    *cursor = cursor.get(total_bytes..).unwrap_or(&[]);
    Ok(u64::from_le_bytes(raw) >> (trailing_zeros + 1))
}

/// Encodes a signed `i64` using MLIR's zig-zag `(val << 1) ^ (val >> 63)` `PrefixVarInt` representation.
///
/// # Arguments
///
/// * `val` - Signed 64-bit integer to zig-zag encode.
#[must_use]
#[allow(clippy::cast_sign_loss)]
pub fn encode_signed_varint(val: i64) -> Vec<u8> {
    let zigzag = ((val << 1) ^ (val >> 63)) as u64;
    encode_prefix_varint(zigzag)
}

/// Decodes a zig-zag encoded signed `i64` from `*cursor`.
///
/// # Arguments
///
/// * `cursor` - Mutable byte-slice cursor positioned at the signed `PrefixVarInt`.
///
/// # Errors
///
/// Returns [`LibTpuError::MlirBytecodeDecodeError`] if `*cursor` ends prematurely.
#[allow(clippy::cast_possible_wrap)]
pub fn decode_signed_varint(cursor: &mut &[u8]) -> LibTpuResult<i64> {
    let raw = decode_prefix_varint(cursor)?;
    let shifted = (raw >> 1) as i64;
    let mask = -((raw & 1) as i64);
    Ok(shifted ^ mask)
}

/// Spec-compliant MLIR `DialectBytecodeReader` over a section or custom dialect bitstream.
#[derive(Debug, Clone)]
pub struct DialectBytecodeReader<'a> {
    /// Original immutable slice for computing byte offsets on errors and alignment.
    full: &'a [u8],
    /// Current unread suffix slice.
    cursor: &'a [u8],
    /// Section identifier for structured error reporting.
    section_id: u8,
}

impl<'a> DialectBytecodeReader<'a> {
    /// Creates a new [`DialectBytecodeReader`] over `bytes` tagged with `section_id`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Raw section or attribute/type payload slice.
    /// * `section_id` - Owning [`MlirSectionId`] numeric value for error diagnostics.
    #[must_use]
    pub const fn new(bytes: &'a [u8], section_id: u8) -> Self {
        Self {
            full: bytes,
            cursor: bytes,
            section_id,
        }
    }

    /// Returns the current byte offset from the start of the underlying buffer.
    #[must_use]
    pub const fn offset(&self) -> usize {
        self.full.len().saturating_sub(self.cursor.len())
    }

    /// Returns `true` when all bytes in the reader have been consumed.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.cursor.is_empty()
    }

    /// Returns the unread remaining slice.
    #[must_use]
    pub const fn remaining(&self) -> &'a [u8] {
        self.cursor
    }

    /// Reads a single raw byte from the stream.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MlirBytecodeDecodeError`] on unexpected EOF.
    pub fn read_byte(&mut self) -> LibTpuResult<u8> {
        let off = self.offset();
        let b = *self
            .cursor
            .first()
            .ok_or(LibTpuError::MlirBytecodeDecodeError {
                section_id: self.section_id,
                offset: off,
                reason: "Unexpected EOF reading byte",
            })?;
        self.cursor = self.cursor.get(1..).unwrap_or(&[]);
        Ok(b)
    }

    /// Reads a boolean byte (`0` -> `false`, non-zero -> `true`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MlirBytecodeDecodeError`] on unexpected EOF.
    pub fn read_bool(&mut self) -> LibTpuResult<bool> {
        Ok(self.read_byte()? != 0)
    }

    /// Reads an unsigned `PrefixVarInt` from the stream.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MlirBytecodeDecodeError`] if the varint is truncated.
    pub fn read_varint(&mut self) -> LibTpuResult<u64> {
        let off = self.offset();
        decode_prefix_varint(&mut self.cursor).map_err(|_| LibTpuError::MlirBytecodeDecodeError {
            section_id: self.section_id,
            offset: off,
            reason: "Truncated PrefixVarInt in DialectBytecodeReader",
        })
    }

    /// Reads a zig-zag signed `PrefixVarInt` from the stream.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MlirBytecodeDecodeError`] if the varint is truncated.
    pub fn read_signed_varint(&mut self) -> LibTpuResult<i64> {
        let off = self.offset();
        decode_signed_varint(&mut self.cursor).map_err(|_| LibTpuError::MlirBytecodeDecodeError {
            section_id: self.section_id,
            offset: off,
            reason: "Truncated signed PrefixVarInt in DialectBytecodeReader",
        })
    }

    /// Reads exactly `len` raw bytes from the stream.
    ///
    /// # Arguments
    ///
    /// * `len` - Exact byte count to consume.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MlirBytecodeDecodeError`] if fewer than `len` bytes remain.
    pub fn read_bytes(&mut self, len: usize) -> LibTpuResult<&'a [u8]> {
        let off = self.offset();
        let out = self
            .cursor
            .get(..len)
            .ok_or(LibTpuError::MlirBytecodeDecodeError {
                section_id: self.section_id,
                offset: off,
                reason: "Unexpected EOF reading fixed byte slice",
            })?;
        self.cursor = self.cursor.get(len..).unwrap_or(&[]);
        Ok(out)
    }

    /// Reads a length-prefixed byte blob (`size: varint` followed by `size` raw bytes).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MlirBytecodeDecodeError`] if the length or payload is truncated.
    pub fn read_blob(&mut self) -> LibTpuResult<&'a [u8]> {
        let len_u64 = self.read_varint()?;
        let len = usize::try_from(len_u64).unwrap_or(usize::MAX);
        self.read_bytes(len)
    }

    /// Reads a NUL-terminated UTF-8 string (`str\0`) from the stream.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MlirBytecodeDecodeError`] if no NUL terminator is found or UTF-8 is invalid.
    pub fn read_nul_string(&mut self) -> LibTpuResult<&'a str> {
        let off = self.offset();
        let nul_pos = self.cursor.iter().position(|&b| b == 0).ok_or(
            LibTpuError::MlirBytecodeDecodeError {
                section_id: self.section_id,
                offset: off,
                reason: "Missing NUL terminator in MLIR bytecode string",
            },
        )?;
        let raw = self.cursor.get(..nul_pos).unwrap_or(&[]);
        self.cursor = self.cursor.get(nul_pos + 1..).unwrap_or(&[]);
        std::str::from_utf8(raw).map_err(|_| LibTpuError::MlirBytecodeDecodeError {
            section_id: self.section_id,
            offset: off,
            reason: "Invalid UTF-8 in MLIR bytecode string",
        })
    }

    /// Reads an `APInt` value of `bit_width` bits matching MLIR's `readAPIntWithKnownWidth`.
    ///
    /// Widths `<= 8` read a single byte; widths `9..=64` read a zig-zag signed varint;
    /// widths `> 64` read `numWords` varint followed by little-endian 64-bit limbs.
    ///
    /// # Arguments
    ///
    /// * `bit_width` - Integer width in bits.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MlirBytecodeDecodeError`] if the encoded `APInt` is truncated.
    pub fn read_apint(&mut self, bit_width: u32) -> LibTpuResult<i128> {
        if bit_width == 0 {
            return Ok(0);
        }
        if bit_width <= 8 {
            let b = self.read_byte()?;
            if bit_width == 1 {
                return Ok(i128::from(b & 1));
            }
            let shift = 8_u32.saturating_sub(bit_width);
            #[allow(clippy::cast_possible_wrap)]
            let signed = ((b << shift) as i8) >> shift;
            return Ok(i128::from(signed));
        }
        if bit_width <= 64 {
            let v = self.read_signed_varint()?;
            return Ok(i128::from(v));
        }
        let num_words = self.read_varint()?;
        let lo = if num_words >= 1 {
            self.read_signed_varint()?
        } else {
            0
        };
        let hi = if num_words >= 2 {
            self.read_signed_varint()?
        } else if lo < 0 {
            -1
        } else {
            0
        };
        for _ in 2..num_words {
            let _ = self.read_signed_varint()?;
        }
        #[allow(clippy::cast_sign_loss)]
        let lo_bits = u128::from(lo as u64);
        #[allow(clippy::cast_sign_loss)]
        let hi_bits = u128::from(hi as u64) << 64;
        #[allow(clippy::cast_possible_wrap)]
        Ok((hi_bits | lo_bits) as i128)
    }

    /// Reads an `APFloat` bit pattern matching MLIR's `readAPFloatWithKnownSemantics` and converts it to IEEE-754 `f64` bits.
    ///
    /// # Arguments
    ///
    /// * `element_type` - Target floating-point [`ElementType`].
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MlirBytecodeDecodeError`] if the float payload is truncated.
    pub fn read_apfloat_f64_bits(&mut self, element_type: ElementType) -> LibTpuResult<u64> {
        let bit_width = u32::try_from(element_type.bit_size()).unwrap_or(32).max(8);
        #[allow(clippy::cast_sign_loss)]
        let raw_bits = self.read_apint(bit_width)? as u128;
        let f64_val = match element_type {
            ElementType::F64 => {
                f64::from_bits(u64::try_from(raw_bits & u128::from(u64::MAX)).unwrap_or(0))
            }
            ElementType::F32 => {
                #[allow(clippy::cast_possible_truncation)]
                let f = f32::from_bits(raw_bits as u32);
                f64::from(f)
            }
            ElementType::Bf16 => {
                #[allow(clippy::cast_possible_truncation)]
                let bf = half::bf16::from_bits(raw_bits as u16);
                f64::from(bf.to_f32())
            }
            ElementType::F16 => {
                #[allow(clippy::cast_possible_truncation)]
                let h = half::f16::from_bits(raw_bits as u16);
                f64::from(h.to_f32())
            }
            _ => {
                #[allow(clippy::cast_possible_truncation)]
                let b = raw_bits as u8;
                element_type
                    .unpack_to_f64_vec(&[b])
                    .first()
                    .copied()
                    .unwrap_or(0.0)
            }
        };
        Ok(f64_val.to_bits())
    }

    /// Validates and converts a `varint` list element count against remaining section bytes.
    ///
    /// # Arguments
    ///
    /// * `count_u64` - Raw decoded element count.
    /// * `off` - Stream offset for error reporting.
    fn check_list_count(&self, count_u64: u64, off: usize) -> LibTpuResult<usize> {
        let count = usize::try_from(count_u64).unwrap_or(usize::MAX);
        if count > self.cursor.len().saturating_add(1) {
            return Err(LibTpuError::MlirBytecodeDecodeError {
                section_id: self.section_id,
                offset: off,
                reason: "List element count exceeds remaining section bytes",
            });
        }
        Ok(count)
    }

    /// Reads a `varint`-prefixed homogeneous list using `elem_fn`.
    ///
    /// # Arguments
    ///
    /// * `elem_fn` - Closure decoding a single list element from `self`.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MlirBytecodeDecodeError`] if the count or any element fails to decode.
    pub fn read_list<T, F>(&mut self, mut elem_fn: F) -> LibTpuResult<Vec<T>>
    where
        F: FnMut(&mut Self) -> LibTpuResult<T>,
    {
        let off = self.offset();
        let count_u64 = self.read_varint()?;
        let count = self.check_list_count(count_u64, off)?;
        let mut items = Vec::with_capacity(count);
        for _ in 0..count {
            items.push(elem_fn(self)?);
        }
        Ok(items)
    }

    /// Advances the reader to the next multiple of `alignment` (power of two), verifying `0xCB` filler bytes.
    ///
    /// # Arguments
    ///
    /// * `alignment` - Power-of-two byte alignment (`1`, `2`, `4`, `8`, ...).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MlirBytecodeDecodeError`] if `alignment` is zero/non-power-of-two,
    /// padding is truncated, or a padding byte is not `0xCB`.
    pub fn align_to(&mut self, alignment: usize) -> LibTpuResult<()> {
        let off = self.offset();
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(LibTpuError::MlirBytecodeDecodeError {
                section_id: self.section_id,
                offset: off,
                reason: "Section alignment must be a non-zero power of two",
            });
        }
        let rem = off & (alignment - 1);
        if rem == 0 {
            return Ok(());
        }
        let pad_len = alignment - rem;
        let pad_bytes = self.read_bytes(pad_len)?;
        if pad_bytes.iter().any(|&b| b != MLIR_ALIGNMENT_BYTE) {
            return Err(LibTpuError::MlirBytecodeDecodeError {
                section_id: self.section_id,
                offset: off,
                reason: "Invalid non-0xCB alignment padding byte in MLIR bytecode",
            });
        }
        Ok(())
    }
}

/// Spec-compliant MLIR `DialectBytecodeWriter` for serializing sections and custom dialect entries.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DialectBytecodeWriter {
    /// Output byte buffer.
    buf: Vec<u8>,
}

impl DialectBytecodeWriter {
    /// Creates an empty [`DialectBytecodeWriter`].
    #[must_use]
    pub const fn new() -> Self {
        Self { buf: Vec::new() }
    }

    /// Creates an empty [`DialectBytecodeWriter`] with pre-allocated `capacity`.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Initial byte capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buf: Vec::with_capacity(capacity),
        }
    }

    /// Returns the number of bytes written so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Returns `true` if zero bytes have been written.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Borrows the underlying byte slice.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.buf
    }

    /// Consumes the writer and returns the encoded byte vector.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }

    /// Writes a single raw byte.
    ///
    /// # Arguments
    ///
    /// * `byte` - Byte value to append.
    pub fn write_byte(&mut self, byte: u8) {
        self.buf.push(byte);
    }

    /// Writes a boolean byte (`0` or `1`).
    ///
    /// # Arguments
    ///
    /// * `val` - Boolean flag to append.
    pub fn write_bool(&mut self, val: bool) {
        self.buf.push(u8::from(val));
    }

    /// Appends raw `bytes` directly to the stream.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Raw byte slice to append.
    pub fn write_bytes(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    /// Appends an unsigned `PrefixVarInt`.
    ///
    /// # Arguments
    ///
    /// * `val` - Unsigned 64-bit integer to encode.
    pub fn write_varint(&mut self, val: u64) {
        self.buf.extend_from_slice(&encode_prefix_varint(val));
    }

    /// Appends a zig-zag signed `PrefixVarInt`.
    ///
    /// # Arguments
    ///
    /// * `val` - Signed 64-bit integer to encode.
    pub fn write_signed_varint(&mut self, val: i64) {
        self.buf.extend_from_slice(&encode_signed_varint(val));
    }

    /// Appends a length-prefixed byte blob (`len: varint` + `bytes`).
    ///
    /// # Arguments
    ///
    /// * `blob` - Raw byte payload to length-prefix and append.
    pub fn write_blob(&mut self, blob: &[u8]) {
        self.write_varint(blob.len() as u64);
        self.write_bytes(blob);
    }

    /// Appends a NUL-terminated UTF-8 string (`s\0`).
    ///
    /// # Arguments
    ///
    /// * `s` - UTF-8 string slice without interior NULs.
    pub fn write_nul_string(&mut self, s: &str) {
        self.buf.extend_from_slice(s.as_bytes());
        self.buf.push(0);
    }

    /// Writes an `APInt` of `bit_width` bits matching MLIR's `writeAPIntWithKnownWidth`.
    ///
    /// # Arguments
    ///
    /// * `bit_width` - Integer bit width.
    /// * `value` - Signed 128-bit value to write.
    pub fn write_apint(&mut self, bit_width: u32, value: i128) {
        if bit_width == 0 {
            return;
        }
        if bit_width <= 8 {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let mask = if bit_width == 8 {
                0xFF_u8
            } else {
                (1_u8 << bit_width) - 1
            };
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            self.write_byte((value as u8) & mask);
            return;
        }
        if bit_width <= 64 {
            #[allow(clippy::cast_possible_truncation)]
            self.write_signed_varint(value as i64);
            return;
        }
        #[allow(clippy::cast_possible_truncation)]
        let lo = value as i64;
        #[allow(clippy::cast_possible_truncation)]
        let hi = (value >> 64) as i64;
        self.write_varint(2);
        self.write_signed_varint(lo);
        self.write_signed_varint(hi);
    }

    /// Writes an `APFloat` (provided as IEEE-754 `f64` bits) converted to `element_type` semantics.
    ///
    /// # Arguments
    ///
    /// * `element_type` - Target floating-point [`ElementType`].
    /// * `f64_bits` - Source `f64::to_bits()` representation.
    pub fn write_apfloat_f64_bits(&mut self, element_type: ElementType, f64_bits: u64) {
        let f64_val = f64::from_bits(f64_bits);
        match element_type {
            ElementType::F64 => {
                #[allow(clippy::cast_possible_wrap)]
                self.write_apint(64, i128::from(f64_bits as i64));
            }
            ElementType::F32 => {
                #[allow(clippy::cast_possible_truncation)]
                let bits32 = (f64_val as f32).to_bits();
                self.write_apint(32, i128::from(i32::from_ne_bytes(bits32.to_ne_bytes())));
            }
            ElementType::Bf16 => {
                #[allow(clippy::cast_possible_truncation)]
                let bits16 = half::bf16::from_f32(f64_val as f32).to_bits();
                self.write_apint(16, i128::from(i16::from_ne_bytes(bits16.to_ne_bytes())));
            }
            ElementType::F16 => {
                #[allow(clippy::cast_possible_truncation)]
                let bits16 = half::f16::from_f32(f64_val as f32).to_bits();
                self.write_apint(16, i128::from(i16::from_ne_bytes(bits16.to_ne_bytes())));
            }
            _ => {
                let packed = element_type.pack_from_f64_slice(&[f64_val]);
                let b = packed.first().copied().unwrap_or(0);
                self.write_apint(8, i128::from(b));
            }
        }
    }

    /// Writes a `varint`-prefixed homogeneous list using `elem_fn`.
    ///
    /// # Arguments
    ///
    /// * `items` - Slice of items to serialize.
    /// * `elem_fn` - Closure writing each item into `self`.
    pub fn write_list<T, F>(&mut self, items: &[T], mut elem_fn: F)
    where
        F: FnMut(&mut Self, &T),
    {
        self.write_varint(items.len() as u64);
        for item in items {
            elem_fn(self, item);
        }
    }

    /// Pads the stream to `alignment` (if `> 1` and power of two) with `0xCB` (`MLIR_ALIGNMENT_BYTE`).
    ///
    /// # Arguments
    ///
    /// * `alignment` - Desired byte alignment.
    pub fn align_to(&mut self, alignment: usize) {
        if alignment <= 1 || !alignment.is_power_of_two() {
            return;
        }
        let rem = self.buf.len() & (alignment - 1);
        if rem != 0 {
            let pad_len = alignment - rem;
            self.buf
                .resize(self.buf.len() + pad_len, MLIR_ALIGNMENT_BYTE);
        }
    }

    /// Writes a complete MLIR bytecode section with optional alignment header and `0xCB` pre-data padding.
    ///
    /// Matches `BytecodeWriter.cpp` `writeSection`:
    /// 1. `(section_id & 0x7F) | (has_alignment << 7)`
    /// 2. `length` (`PrefixVarInt`)
    /// 3. If `alignment` is `Some(align)` (`align > 1`, power of two): `align` (`PrefixVarInt`) followed by `0xCB` bytes until `self.len()` is a multiple of `align`.
    /// 4. Section `data` bytes.
    ///
    /// # Arguments
    ///
    /// * `section_id` - Section ID (`0..=8`).
    /// * `alignment` - Optional power-of-two byte alignment.
    /// * `data` - Encoded section body bytes.
    pub fn write_section(&mut self, section_id: u8, alignment: Option<u64>, data: &[u8]) {
        let valid_align = alignment.filter(|&a| a > 1 && a.is_power_of_two());
        let id_and_flag = (section_id & 0x7F) | if valid_align.is_some() { 0x80 } else { 0 };
        self.write_byte(id_and_flag);
        self.write_varint(data.len() as u64);
        if let Some(align) = valid_align {
            self.write_varint(align);
            self.align_to(usize::try_from(align).unwrap_or(1));
        }
        self.write_bytes(data);
    }
}

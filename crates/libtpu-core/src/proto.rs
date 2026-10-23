//! Pure-Rust, zero-panic Protobuf wire-format encoder and decoder for `libtpu-rs`.
//!
//! Implements wire-compatible binary serialization and deserialization matching the
//! embedded protobuf schemas in `libtpu.so`:
//! - `xla.ShapeProto`, `xla.LayoutProto`, `xla.TileProto`, `xla.ProgramShapeProto`
//! - `xla.HloInstructionProto`, `xla.HloComputationProto`, `xla.HloModuleProto`, `xla.OpSharding`
//! - `xla.ExecutableBuildOptionsProto`, `xla.CompileOptionsProto`
//! - `tpu.TpuTopologyArgsProto`, `tpu.TpuCompilationEnvironmentProto`, `tpu.TpuExecutableProto`, `tpu.TpuSerializedExecutableProto`
//! - `tensorflow.profiler.XSpace`, `XPlane`, `XLine`, `XEvent`, `XEventMetadata`, `XStat`, `XStatMetadata`
//! - `tensorflow.tpu.TPUEmbeddingConfiguration`, `TableDescriptor`, `FeatureDescriptor`

use crate::error::{LibTpuError, LibTpuResult};

/// Protobuf wire type discriminant (3-bit value in tag varint).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireType {
    /// Variable-length integer (`0`).
    Varint = 0,
    /// 64-bit fixed-width integer or double (`1`).
    Fixed64 = 1,
    /// Length-delimited bytes, string, nested message, or packed repeated field (`2`).
    LengthDelimited = 2,
    /// 32-bit fixed-width integer or float (`5`).
    Fixed32 = 5,
}

impl WireType {
    /// Parses a 3-bit wire type value from a decoded tag varint.
    ///
    /// # Arguments
    ///
    /// * `raw` - Raw 3-bit wire type integer (`tag & 0x7`).
    /// * `message_type` - Name of the enclosing message for diagnostics.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if `raw` is not `0`, `1`, `2`, or `5`.
    pub fn from_raw(raw: u64, message_type: &'static str) -> LibTpuResult<Self> {
        match raw {
            0 => Ok(Self::Varint),
            1 => Ok(Self::Fixed64),
            2 => Ok(Self::LengthDelimited),
            5 => Ok(Self::Fixed32),
            _ => Err(LibTpuError::ProtobufDecodeError {
                message_type,
                reason: format!("Unsupported protobuf wire type {raw}"),
            }),
        }
    }
}

/// Low-level protobuf binary wire writer.
#[derive(Debug, Default, Clone)]
pub struct ProtoWriter {
    /// Underlying encoded byte buffer.
    buf: Vec<u8>,
}

impl ProtoWriter {
    /// Creates an empty [`ProtoWriter`].
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    /// Consumes the writer and returns the encoded byte vector.
    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }

    /// Appends a base-128 varint to the buffer.
    ///
    /// # Arguments
    ///
    /// * `value` - Unsigned 64-bit integer to encode as a varint.
    pub fn write_varint(&mut self, mut value: u64) {
        while value >= 0x80 {
            let low_seven = u8::try_from(value & 0x7F).unwrap_or(0);
            self.buf.push(low_seven | 0x80);
            value >>= 7;
        }
        self.buf.push(u8::try_from(value).unwrap_or(0));
    }

    /// Writes a field tag (`(field_number << 3) | wire_type`).
    ///
    /// # Arguments
    ///
    /// * `field_number` - 1-based protobuf field number.
    /// * `wire_type` - Wire type of the field.
    pub fn write_tag(&mut self, field_number: u32, wire_type: WireType) {
        let tag = (u64::from(field_number) << 3) | (wire_type as u64);
        self.write_varint(tag);
    }

    /// Writes a `uint64` / `int64` / `int32` / `bool` varint field if non-zero.
    ///
    /// # Arguments
    ///
    /// * `field_number` - Protobuf field number.
    /// * `value` - Unsigned 64-bit varint value.
    pub fn write_uint64(&mut self, field_number: u32, value: u64) {
        if value != 0 {
            self.write_tag(field_number, WireType::Varint);
            self.write_varint(value);
        }
    }

    /// Writes a signed `int64` varint field if non-zero.
    ///
    /// # Arguments
    ///
    /// * `field_number` - Protobuf field number.
    /// * `value` - Signed 64-bit integer value.
    pub fn write_int64(&mut self, field_number: u32, value: i64) {
        self.write_uint64(field_number, u64::from_ne_bytes(value.to_ne_bytes()));
    }

    /// Writes a signed `int32` varint field if non-zero.
    ///
    /// # Arguments
    ///
    /// * `field_number` - Protobuf field number.
    /// * `value` - Signed 32-bit integer value.
    pub fn write_int32(&mut self, field_number: u32, value: i32) {
        self.write_int64(field_number, i64::from(value));
    }

    /// Writes a `bool` varint field if `true`.
    ///
    /// # Arguments
    ///
    /// * `field_number` - Protobuf field number.
    /// * `value` - Boolean value.
    pub fn write_bool(&mut self, field_number: u32, value: bool) {
        if value {
            self.write_uint64(field_number, 1);
        }
    }

    /// Writes a 64-bit IEEE `double` field (`WireType::Fixed64`) if non-zero.
    ///
    /// # Arguments
    ///
    /// * `field_number` - Protobuf field number.
    /// * `value` - 64-bit floating-point value.
    pub fn write_double(&mut self, field_number: u32, value: f64) {
        if value.to_bits() != 0 {
            self.write_tag(field_number, WireType::Fixed64);
            self.buf.extend_from_slice(&value.to_le_bytes());
        }
    }

    /// Writes a 32-bit IEEE `float` field (`WireType::Fixed32`) if non-zero.
    ///
    /// # Arguments
    ///
    /// * `field_number` - Protobuf field number.
    /// * `value` - 32-bit floating-point value.
    pub fn write_float(&mut self, field_number: u32, value: f32) {
        if value.to_bits() != 0 {
            self.write_tag(field_number, WireType::Fixed32);
            self.buf.extend_from_slice(&value.to_le_bytes());
        }
    }

    /// Writes a length-delimited byte slice (`bytes`, `string`, or nested message).
    ///
    /// # Arguments
    ///
    /// * `field_number` - Protobuf field number.
    /// * `bytes` - Payload bytes.
    pub fn write_bytes(&mut self, field_number: u32, bytes: &[u8]) {
        if !bytes.is_empty() {
            self.write_tag(field_number, WireType::LengthDelimited);
            self.write_varint(bytes.len() as u64);
            self.buf.extend_from_slice(bytes);
        }
    }

    /// Writes a length-delimitedbyte slice unconditionally (even if empty, used for nested messages).
    ///
    /// # Arguments
    ///
    /// * `field_number` - Protobuf field number.
    /// * `bytes` - Serialized nested message bytes.
    pub fn write_message(&mut self, field_number: u32, bytes: &[u8]) {
        self.write_tag(field_number, WireType::LengthDelimited);
        self.write_varint(bytes.len() as u64);
        self.buf.extend_from_slice(bytes);
    }

    /// Writes a UTF-8 `string` field if non-empty.
    ///
    /// # Arguments
    ///
    /// * `field_number` - Protobuf field number.
    /// * `text` - String value.
    pub fn write_string(&mut self, field_number: u32, text: &str) {
        self.write_bytes(field_number, text.as_bytes());
    }

    /// Writes a packed repeated `int64` slice.
    ///
    /// # Arguments
    ///
    /// * `field_number` - Protobuf field number.
    /// * `values` - Slice of 64-bit signed integers.
    pub fn write_packed_int64(&mut self, field_number: u32, values: &[i64]) {
        if values.is_empty() {
            return;
        }
        let mut sub = Self::new();
        for &v in values {
            sub.write_varint(u64::from_ne_bytes(v.to_ne_bytes()));
        }
        self.write_bytes(field_number, &sub.into_bytes());
    }
}

/// Low-level protobuf binary wire reader.
#[derive(Debug, Clone)]
pub struct ProtoReader<'a> {
    /// Remaining unread byte slice.
    data: &'a [u8],
    /// Current read cursor offset.
    pos: usize,
    /// Name of the message being decoded for error reporting.
    message_type: &'static str,
}

impl<'a> ProtoReader<'a> {
    /// Creates a new [`ProtoReader`] over `data`.
    ///
    /// # Arguments
    ///
    /// * `data` - Input binary protobuf slice.
    /// * `message_type` - Message type name for error diagnostics.
    pub fn new(data: &'a [u8], message_type: &'static str) -> Self {
        Self {
            data,
            pos: 0,
            message_type,
        }
    }

    /// Returns `true` if all bytes in the slice have been consumed.
    pub fn is_eof(&self) -> bool {
        self.pos >= self.data.len()
    }

    /// Reads a base-128 varint from the stream.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if the stream ends prematurely or the varint exceeds 64 bits.
    pub fn read_varint(&mut self) -> LibTpuResult<u64> {
        let mut result: u64 = 0;
        let mut shift: u32 = 0;
        while shift < 64 {
            let byte =
                *self
                    .data
                    .get(self.pos)
                    .ok_or_else(|| LibTpuError::ProtobufDecodeError {
                        message_type: self.message_type,
                        reason: String::from("Unexpected EOF while reading varint"),
                    })?;
            self.pos += 1;
            result |= u64::from(byte & 0x7F) << shift;
            if (byte & 0x80) == 0 {
                return Ok(result);
            }
            shift += 7;
        }
        Err(LibTpuError::ProtobufDecodeError {
            message_type: self.message_type,
            reason: String::from("Varint exceeded 64 bits"),
        })
    }

    /// Reads a signed 64-bit integer varint from the stream.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if reading the varint fails.
    pub fn read_int64(&mut self) -> LibTpuResult<i64> {
        let raw = self.read_varint()?;
        Ok(i64::from_ne_bytes(raw.to_ne_bytes()))
    }

    /// Reads a signed 32-bit integer varint from the stream.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if reading the varint fails or overflows `i32`.
    pub fn read_int32(&mut self) -> LibTpuResult<i32> {
        let val64 = self.read_int64()?;
        i32::try_from(val64).map_err(|_| LibTpuError::ProtobufDecodeError {
            message_type: self.message_type,
            reason: String::from("Int32 varint overflow"),
        })
    }

    /// Reads the next `(field_number, wire_type)` tag from the stream.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if the tag varint or field number (`0`) is invalid.
    pub fn read_tag(&mut self) -> LibTpuResult<(u32, WireType)> {
        let raw = self.read_varint()?;
        let field_number =
            u32::try_from(raw >> 3).map_err(|_| LibTpuError::ProtobufDecodeError {
                message_type: self.message_type,
                reason: String::from("Field number overflow"),
            })?;
        if field_number == 0 {
            return Err(LibTpuError::ProtobufDecodeError {
                message_type: self.message_type,
                reason: String::from("Invalid field number 0"),
            });
        }
        let wire_type = WireType::from_raw(raw & 0x7, self.message_type)?;
        Ok((field_number, wire_type))
    }

    /// Reads a 64-bit little-endian fixed word (`Fixed64` or `double`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if fewer than 8 bytes remain.
    pub fn read_fixed64(&mut self) -> LibTpuResult<u64> {
        let end = self
            .pos
            .checked_add(8)
            .ok_or_else(|| LibTpuError::ProtobufDecodeError {
                message_type: self.message_type,
                reason: String::from("Fixed64 offset overflow"),
            })?;
        let slice =
            self.data
                .get(self.pos..end)
                .ok_or_else(|| LibTpuError::ProtobufDecodeError {
                    message_type: self.message_type,
                    reason: String::from("Unexpected EOF while reading Fixed64"),
                })?;
        self.pos = end;
        let mut arr = [0_u8; 8];
        arr.copy_from_slice(slice);
        Ok(u64::from_le_bytes(arr))
    }

    /// Reads a 32-bit little-endian fixed word (`Fixed32` or `float`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if fewer than 4 bytes remain.
    pub fn read_fixed32(&mut self) -> LibTpuResult<u32> {
        let end = self
            .pos
            .checked_add(4)
            .ok_or_else(|| LibTpuError::ProtobufDecodeError {
                message_type: self.message_type,
                reason: String::from("Fixed32 offset overflow"),
            })?;
        let slice =
            self.data
                .get(self.pos..end)
                .ok_or_else(|| LibTpuError::ProtobufDecodeError {
                    message_type: self.message_type,
                    reason: String::from("Unexpected EOF while reading Fixed32"),
                })?;
        self.pos = end;
        let mut arr = [0_u8; 4];
        arr.copy_from_slice(slice);
        Ok(u32::from_le_bytes(arr))
    }

    /// Reads a length-delimited byte slice.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if the length varint overflows or exceeds remaining bytes.
    pub fn read_bytes(&mut self) -> LibTpuResult<&'a [u8]> {
        let len = usize::try_from(self.read_varint()?).unwrap_or(usize::MAX);
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| LibTpuError::ProtobufDecodeError {
                message_type: self.message_type,
                reason: String::from("Length-delimited offset overflow"),
            })?;
        let slice =
            self.data
                .get(self.pos..end)
                .ok_or_else(|| LibTpuError::ProtobufDecodeError {
                    message_type: self.message_type,
                    reason: String::from("Unexpected EOF in length-delimited field"),
                })?;
        self.pos = end;
        Ok(slice)
    }

    /// Reads a length-delimited UTF-8 string.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if the bytes are truncated or not valid UTF-8.
    pub fn read_string(&mut self) -> LibTpuResult<String> {
        let bytes = self.read_bytes()?;
        std::str::from_utf8(bytes)
            .map(String::from)
            .map_err(|e| LibTpuError::ProtobufDecodeError {
                message_type: self.message_type,
                reason: format!("Invalid UTF-8 in protobuf string: {e}"),
            })
    }

    /// Reads a packed or unpacked `int64` field depending on `wire_type`.
    ///
    /// # Arguments
    ///
    /// * `wire_type` - Wire type from the field tag (`Varint` or `LengthDelimited`).
    /// * `out` - Destination vector to append decoded `i64` values into.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if the wire type is neither `Varint` nor `LengthDelimited`.
    pub fn read_repeated_int64(
        &mut self,
        wire_type: WireType,
        out: &mut Vec<i64>,
    ) -> LibTpuResult<()> {
        match wire_type {
            WireType::Varint => {
                let v = self.read_int64()?;
                out.push(v);
                Ok(())
            }
            WireType::LengthDelimited => {
                let bytes = self.read_bytes()?;
                let mut sub = ProtoReader::new(bytes, self.message_type);
                while !sub.is_eof() {
                    let v = sub.read_int64()?;
                    out.push(v);
                }
                Ok(())
            }
            _ => Err(LibTpuError::ProtobufDecodeError {
                message_type: self.message_type,
                reason: format!("Invalid wire type {wire_type:?} for repeated int64"),
            }),
        }
    }

    /// Skips an unknown field value according to its [`WireType`].
    ///
    /// # Arguments
    ///
    /// * `wire_type` - Wire type of the unknown field to skip.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if the field payload is truncated.
    pub fn skip_field(&mut self, wire_type: WireType) -> LibTpuResult<()> {
        match wire_type {
            WireType::Varint => {
                let _ = self.read_varint()?;
            }
            WireType::Fixed64 => {
                let _ = self.read_fixed64()?;
            }
            WireType::LengthDelimited => {
                let _ = self.read_bytes()?;
            }
            WireType::Fixed32 => {
                let _ = self.read_fixed32()?;
            }
        }
        Ok(())
    }
}

/// Wire-compatible `xla.TileProto` (`xla_data.proto`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TileProto {
    /// Tile dimensions (e.g., `[8, 128]`), field `1`.
    pub dimensions: Vec<i64>,
}

impl TileProto {
    /// Encodes this [`TileProto`] to protobuf wire bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ProtoWriter::new();
        w.write_packed_int64(1, &self.dimensions);
        w.into_bytes()
    }

    /// Decodes a [`TileProto`] from protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `TileProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        let mut r = ProtoReader::new(bytes, "xla.TileProto");
        let mut out = Self::default();
        while !r.is_eof() {
            let (field, wt) = r.read_tag()?;
            if field == 1 {
                r.read_repeated_int64(wt, &mut out.dimensions)?;
            } else {
                r.skip_field(wt)?;
            }
        }
        Ok(out)
    }
}

/// Wire-compatible `xla.LayoutProto` (`xla_data.proto`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LayoutProto {
    /// Minor-to-major dimension ordering (field `1`).
    pub minor_to_major: Vec<i64>,
    /// Tiles applied to the layout (field `2`).
    pub tiles: Vec<TileProto>,
    /// Element size in bits (field `3`).
    pub element_size_in_bits: i64,
    /// Memory space ID (`0` = default HBM, `1` = VMEM, `3` = pinned host, field `4`).
    pub memory_space: i64,
}

impl LayoutProto {
    /// Encodes this [`LayoutProto`] to protobuf wire bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ProtoWriter::new();
        w.write_packed_int64(1, &self.minor_to_major);
        for tile in &self.tiles {
            w.write_message(2, &tile.encode());
        }
        w.write_int64(3, self.element_size_in_bits);
        w.write_int64(4, self.memory_space);
        w.into_bytes()
    }

    /// Decodes a [`LayoutProto`] from protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `LayoutProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        let mut r = ProtoReader::new(bytes, "xla.LayoutProto");
        let mut out = Self::default();
        while !r.is_eof() {
            let (field, wt) = r.read_tag()?;
            match (field, wt) {
                (1, _) => r.read_repeated_int64(wt, &mut out.minor_to_major)?,
                (2, WireType::LengthDelimited) => {
                    let sub = r.read_bytes()?;
                    out.tiles.push(TileProto::decode(sub)?);
                }
                (3, WireType::Varint) => out.element_size_in_bits = r.read_int64()?,
                (4, WireType::Varint) => out.memory_space = r.read_int64()?,
                _ => r.skip_field(wt)?,
            }
        }
        Ok(out)
    }
}

/// Wire-compatible `xla.ShapeProto` (`xla_data.proto`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ShapeProto {
    /// Primitive element type (`PrimitiveType`, field `2`).
    pub element_type: i32,
    /// Dimension extents (field `3`).
    pub dimensions: Vec<i64>,
    /// Tuple subshapes if `element_type == TUPLE` (field `4`).
    pub tuple_shapes: Vec<Self>,
    /// Optional physical layout (field `5`).
    pub layout: Option<LayoutProto>,
    /// Dynamic dimension flags (field `6`).
    pub is_dynamic_dimension: Vec<bool>,
}

impl ShapeProto {
    /// Encodes this [`ShapeProto`] to protobuf wire bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ProtoWriter::new();
        w.write_int32(2, self.element_type);
        w.write_packed_int64(3, &self.dimensions);
        for sub in &self.tuple_shapes {
            w.write_message(4, &sub.encode());
        }
        if let Some(layout) = &self.layout {
            w.write_message(5, &layout.encode());
        }
        for &dyn_flag in &self.is_dynamic_dimension {
            w.write_tag(6, WireType::Varint);
            w.write_varint(u64::from(dyn_flag));
        }
        w.into_bytes()
    }

    /// Decodes a [`ShapeProto`] from protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `ShapeProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        let mut r = ProtoReader::new(bytes, "xla.ShapeProto");
        let mut out = Self::default();
        while !r.is_eof() {
            let (field, wt) = r.read_tag()?;
            match (field, wt) {
                (2, WireType::Varint) => out.element_type = r.read_int32()?,
                (3, _) => r.read_repeated_int64(wt, &mut out.dimensions)?,
                (4, WireType::LengthDelimited) => {
                    let sub = r.read_bytes()?;
                    out.tuple_shapes.push(Self::decode(sub)?);
                }
                (5, WireType::LengthDelimited) => {
                    let sub = r.read_bytes()?;
                    out.layout = Some(LayoutProto::decode(sub)?);
                }
                (6, WireType::Varint) => out.is_dynamic_dimension.push(r.read_varint()? != 0),
                _ => r.skip_field(wt)?,
            }
        }
        Ok(out)
    }
}

/// Wire-compatible `xla.OpSharding` (`xla_data.proto`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OpShardingProto {
    /// Sharding type (`0` = Replicated, `1` = Maximal, `2` = Tuple, `3` = Other/Tiled, field `1`).
    pub sharding_type: i32,
    /// Tile assignment dimensions (field `3`).
    pub tile_assignment_dimensions: Vec<i64>,
    /// Tile assignment devices (field `4`).
    pub tile_assignment_devices: Vec<i64>,
}

impl OpShardingProto {
    /// Encodes this [`OpShardingProto`] to protobuf wire bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ProtoWriter::new();
        w.write_int32(1, self.sharding_type);
        w.write_packed_int64(3, &self.tile_assignment_dimensions);
        w.write_packed_int64(4, &self.tile_assignment_devices);
        w.into_bytes()
    }

    /// Decodes an [`OpShardingProto`] from protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `OpSharding` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        let mut r = ProtoReader::new(bytes, "xla.OpSharding");
        let mut out = Self::default();
        while !r.is_eof() {
            let (field, wt) = r.read_tag()?;
            match (field, wt) {
                (1, WireType::Varint) => out.sharding_type = r.read_int32()?,
                (3, _) => r.read_repeated_int64(wt, &mut out.tile_assignment_dimensions)?,
                (4, _) => r.read_repeated_int64(wt, &mut out.tile_assignment_devices)?,
                _ => r.skip_field(wt)?,
            }
        }
        Ok(out)
    }
}

/// Wire-compatible `xla.HloInstructionProto` (`hlo.proto`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HloInstructionProto {
    /// Instruction name (field `1`).
    pub name: String,
    /// Instruction opcode string (e.g., `"add"`, `"multiply"`, `"dot"`, `"parameter"`, field `2`).
    pub opcode: String,
    /// Output shape of this instruction (field `3`).
    pub shape: Option<ShapeProto>,
    /// Unique instruction ID within the module (field `7`).
    pub id: i64,
    /// Operand instruction IDs (field `8`).
    pub operand_ids: Vec<i64>,
    /// Parameter number for `opcode == "parameter"` (field `9`).
    pub parameter_number: i64,
    /// Custom call target name (field `14`).
    pub custom_call_target: String,
}

impl HloInstructionProto {
    /// Encodes this [`HloInstructionProto`] to protobuf wire bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ProtoWriter::new();
        w.write_string(1, &self.name);
        w.write_string(2, &self.opcode);
        if let Some(s) = &self.shape {
            w.write_message(3, &s.encode());
        }
        w.write_int64(7, self.id);
        w.write_packed_int64(8, &self.operand_ids);
        w.write_int64(9, self.parameter_number);
        w.write_string(14, &self.custom_call_target);
        w.into_bytes()
    }

    /// Decodes an [`HloInstructionProto`] from protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `HloInstructionProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        let mut r = ProtoReader::new(bytes, "xla.HloInstructionProto");
        let mut out = Self::default();
        while !r.is_eof() {
            let (field, wt) = r.read_tag()?;
            match (field, wt) {
                (1, WireType::LengthDelimited) => out.name = r.read_string()?,
                (2, WireType::LengthDelimited) => out.opcode = r.read_string()?,
                (3, WireType::LengthDelimited) => {
                    let sub = r.read_bytes()?;
                    out.shape = Some(ShapeProto::decode(sub)?);
                }
                (7, WireType::Varint) => out.id = r.read_int64()?,
                (8, _) => r.read_repeated_int64(wt, &mut out.operand_ids)?,
                (9, WireType::Varint) => out.parameter_number = r.read_int64()?,
                (14, WireType::LengthDelimited) => out.custom_call_target = r.read_string()?,
                _ => r.skip_field(wt)?,
            }
        }
        Ok(out)
    }
}

/// Wire-compatible `xla.HloComputationProto` (`hlo.proto`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HloComputationProto {
    /// Computation name (field `1`).
    pub name: String,
    /// Instructions in topological order (field `2`).
    pub instructions: Vec<HloInstructionProto>,
    /// Computation ID (field `5`).
    pub id: i64,
    /// Root instruction ID (field `6`).
    pub root_id: i64,
}

impl HloComputationProto {
    /// Encodes this [`HloComputationProto`] to protobuf wire bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ProtoWriter::new();
        w.write_string(1, &self.name);
        for inst in &self.instructions {
            w.write_message(2, &inst.encode());
        }
        w.write_int64(5, self.id);
        w.write_int64(6, self.root_id);
        w.into_bytes()
    }

    /// Decodes an [`HloComputationProto`] from protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `HloComputationProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        let mut r = ProtoReader::new(bytes, "xla.HloComputationProto");
        let mut out = Self::default();
        while !r.is_eof() {
            let (field, wt) = r.read_tag()?;
            match (field, wt) {
                (1, WireType::LengthDelimited) => out.name = r.read_string()?,
                (2, WireType::LengthDelimited) => {
                    let sub = r.read_bytes()?;
                    out.instructions.push(HloInstructionProto::decode(sub)?);
                }
                (5, WireType::Varint) => out.id = r.read_int64()?,
                (6, WireType::Varint) => out.root_id = r.read_int64()?,
                _ => r.skip_field(wt)?,
            }
        }
        Ok(out)
    }
}

/// Wire-compatible `xla.HloModuleProto` (`hlo.proto`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HloModuleProto {
    /// Module name (field `1`).
    pub name: String,
    /// Entry computation name (field `2`).
    pub entry_computation_name: String,
    /// Entry computation ID (field `3`).
    pub entry_computation_id: i64,
    /// Computations belonging to this module (field `4`).
    pub computations: Vec<HloComputationProto>,
    /// Module ID (field `6`).
    pub id: i64,
}

impl HloModuleProto {
    /// Encodes this [`HloModuleProto`] to protobuf wire bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ProtoWriter::new();
        w.write_string(1, &self.name);
        w.write_string(2, &self.entry_computation_name);
        w.write_int64(3, self.entry_computation_id);
        for comp in &self.computations {
            w.write_message(4, &comp.encode());
        }
        w.write_int64(6, self.id);
        w.into_bytes()
    }

    /// Decodes an [`HloModuleProto`] from protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `HloModuleProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        let mut r = ProtoReader::new(bytes, "xla.HloModuleProto");
        let mut out = Self::default();
        while !r.is_eof() {
            let (field, wt) = r.read_tag()?;
            match (field, wt) {
                (1, WireType::LengthDelimited) => out.name = r.read_string()?,
                (2, WireType::LengthDelimited) => out.entry_computation_name = r.read_string()?,
                (3, WireType::Varint) => out.entry_computation_id = r.read_int64()?,
                (4, WireType::LengthDelimited) => {
                    let sub = r.read_bytes()?;
                    out.computations.push(HloComputationProto::decode(sub)?);
                }
                (6, WireType::Varint) => out.id = r.read_int64()?,
                _ => r.skip_field(wt)?,
            }
        }
        Ok(out)
    }
}

/// Wire-compatible `xla.ExecutableBuildOptionsProto` (`compile_options.proto`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExecutableBuildOptionsProto {
    /// Device ordinal (`-1` if unspecified, field `1`).
    pub device_ordinal: i64,
    /// Result layout (field `2`).
    pub result_layout: Option<ShapeProto>,
    /// Number of replicas (field `4`).
    pub num_replicas: i64,
    /// Number of SPMD partitions (field `5`).
    pub num_partitions: i64,
    /// Whether SPMD partitioning is enabled (field `6`).
    pub use_spmd_partitioning: bool,
}

impl ExecutableBuildOptionsProto {
    /// Encodes this [`ExecutableBuildOptionsProto`] to protobuf wire bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ProtoWriter::new();
        w.write_int64(1, self.device_ordinal);
        if let Some(res) = &self.result_layout {
            w.write_message(2, &res.encode());
        }
        w.write_int64(4, self.num_replicas);
        w.write_int64(5, self.num_partitions);
        w.write_bool(6, self.use_spmd_partitioning);
        w.into_bytes()
    }

    /// Decodes an [`ExecutableBuildOptionsProto`] from protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `ExecutableBuildOptionsProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        let mut r = ProtoReader::new(bytes, "xla.ExecutableBuildOptionsProto");
        let mut out = Self::default();
        while !r.is_eof() {
            let (field, wt) = r.read_tag()?;
            match (field, wt) {
                (1, WireType::Varint) => out.device_ordinal = r.read_int64()?,
                (2, WireType::LengthDelimited) => {
                    let sub = r.read_bytes()?;
                    out.result_layout = Some(ShapeProto::decode(sub)?);
                }
                (4, WireType::Varint) => out.num_replicas = r.read_int64()?,
                (5, WireType::Varint) => out.num_partitions = r.read_int64()?,
                (6, WireType::Varint) => out.use_spmd_partitioning = r.read_varint()? != 0,
                _ => r.skip_field(wt)?,
            }
        }
        Ok(out)
    }
}

/// Wire-compatible `xla.CompileOptionsProto` (`compile_options.proto`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompileOptionsProto {
    /// Argument layouts (field `1`).
    pub argument_layouts: Vec<ShapeProto>,
    /// Whether parameters are tupled (field `2`).
    pub parameter_is_tupled_arguments: bool,
    /// Build options (field `3`).
    pub executable_build_options: Option<ExecutableBuildOptionsProto>,
    /// Compile portable executable flag (field `4`).
    pub compile_portable_executable: bool,
    /// Profile version (field `5`).
    pub profile_version: i64,
}

impl CompileOptionsProto {
    /// Encodes this [`CompileOptionsProto`] to protobuf wire bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ProtoWriter::new();
        for layout in &self.argument_layouts {
            w.write_message(1, &layout.encode());
        }
        w.write_bool(2, self.parameter_is_tupled_arguments);
        if let Some(opts) = &self.executable_build_options {
            w.write_message(3, &opts.encode());
        }
        w.write_bool(4, self.compile_portable_executable);
        w.write_int64(5, self.profile_version);
        w.into_bytes()
    }

    /// Decodes a [`CompileOptionsProto`] from protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `CompileOptionsProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        let mut r = ProtoReader::new(bytes, "xla.CompileOptionsProto");
        let mut out = Self::default();
        while !r.is_eof() {
            let (field, wt) = r.read_tag()?;
            match (field, wt) {
                (1, WireType::LengthDelimited) => {
                    let sub = r.read_bytes()?;
                    out.argument_layouts.push(ShapeProto::decode(sub)?);
                }
                (2, WireType::Varint) => out.parameter_is_tupled_arguments = r.read_varint()? != 0,
                (3, WireType::LengthDelimited) => {
                    let sub = r.read_bytes()?;
                    out.executable_build_options = Some(ExecutableBuildOptionsProto::decode(sub)?);
                }
                (4, WireType::Varint) => out.compile_portable_executable = r.read_varint()? != 0,
                (5, WireType::Varint) => out.profile_version = r.read_int64()?,
                _ => r.skip_field(wt)?,
            }
        }
        Ok(out)
    }
}

/// Wire-compatible `xla.DeviceAssignmentProto.ComputationDevice` (`xla_data.proto`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ComputationDeviceProto {
    /// Device IDs assigned to each replica of this computation (field `1`).
    pub replica_device_ids: Vec<i64>,
}

impl ComputationDeviceProto {
    /// Encodes this [`ComputationDeviceProto`] to protobuf wire bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ProtoWriter::new();
        w.write_packed_int64(1, &self.replica_device_ids);
        w.into_bytes()
    }

    /// Decodes a [`ComputationDeviceProto`] from protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `ComputationDeviceProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        let mut r = ProtoReader::new(bytes, "xla.ComputationDeviceProto");
        let mut out = Self::default();
        while !r.is_eof() {
            let (field, wt) = r.read_tag()?;
            if field == 1 {
                r.read_repeated_int64(wt, &mut out.replica_device_ids)?;
            } else {
                r.skip_field(wt)?;
            }
        }
        Ok(out)
    }
}

/// Wire-compatible `xla.DeviceAssignmentProto` (`xla_data.proto`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DeviceAssignmentProto {
    /// Number of data-parallel replicas (field `1`).
    pub replica_count: i32,
    /// Number of model-parallel computations/partitions (field `2`).
    pub computation_count: i32,
    /// Per-computation replica device assignments (field `3`).
    pub computation_devices: Vec<ComputationDeviceProto>,
}

impl DeviceAssignmentProto {
    /// Encodes this [`DeviceAssignmentProto`] to protobuf wire bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ProtoWriter::new();
        w.write_int32(1, self.replica_count);
        w.write_int32(2, self.computation_count);
        for cd in &self.computation_devices {
            w.write_message(3, &cd.encode());
        }
        w.into_bytes()
    }

    /// Decodes a [`DeviceAssignmentProto`] from protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `DeviceAssignmentProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        let mut r = ProtoReader::new(bytes, "xla.DeviceAssignmentProto");
        let mut out = Self::default();
        while !r.is_eof() {
            let (field, wt) = r.read_tag()?;
            match (field, wt) {
                (1, WireType::Varint) => out.replica_count = r.read_int32()?,
                (2, WireType::Varint) => out.computation_count = r.read_int32()?,
                (3, WireType::LengthDelimited) => {
                    let sub = r.read_bytes()?;
                    out.computation_devices
                        .push(ComputationDeviceProto::decode(sub)?);
                }
                _ => r.skip_field(wt)?,
            }
        }
        Ok(out)
    }
}

/// Wire-compatible `tpu.TpuCompilationEnvironmentProto` (`tpu_compilation_environment.proto`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TpuCompilationEnvironmentProto {
    /// XLA TPU Megacore fusion toggle (field `1`).
    pub xla_tpu_enable_megacore_fusion: bool,
    /// XLA TPU async collective fusion toggle (field `2`).
    pub xla_tpu_enable_async_collective_fusion: bool,
    /// XLA TPU VMEM limit in KiB (field `3`).
    pub xla_tpu_scoped_vmem_limit_kib: i64,
    /// Raw custom flag string overrides (field `4`).
    pub custom_flags: String,
}

impl TpuCompilationEnvironmentProto {
    /// Encodes this [`TpuCompilationEnvironmentProto`] to protobuf wire bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ProtoWriter::new();
        w.write_bool(1, self.xla_tpu_enable_megacore_fusion);
        w.write_bool(2, self.xla_tpu_enable_async_collective_fusion);
        w.write_int64(3, self.xla_tpu_scoped_vmem_limit_kib);
        w.write_string(4, &self.custom_flags);
        w.into_bytes()
    }

    /// Decodes a [`TpuCompilationEnvironmentProto`] from protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `TpuCompilationEnvironmentProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        let mut r = ProtoReader::new(bytes, "tpu.TpuCompilationEnvironmentProto");
        let mut out = Self::default();
        while !r.is_eof() {
            let (field, wt) = r.read_tag()?;
            match (field, wt) {
                (1, WireType::Varint) => {
                    out.xla_tpu_enable_megacore_fusion = r.read_varint()? != 0;
                }
                (2, WireType::Varint) => {
                    out.xla_tpu_enable_async_collective_fusion = r.read_varint()? != 0;
                }
                (3, WireType::Varint) => out.xla_tpu_scoped_vmem_limit_kib = r.read_int64()?,
                (4, WireType::LengthDelimited) => out.custom_flags = r.read_string()?,
                _ => r.skip_field(wt)?,
            }
        }
        Ok(out)
    }
}

/// Wire-compatible `tensorflow.profiler.XStatMetadata` (`xplane.proto`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct XStatMetadataProto {
    /// Metadata ID (field `1`).
    pub id: i64,
    /// Stat name (field `2`).
    pub name: String,
    /// Stat description (field `3`).
    pub description: String,
}

impl XStatMetadataProto {
    /// Encodes this [`XStatMetadataProto`] to protobuf wire bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ProtoWriter::new();
        w.write_int64(1, self.id);
        w.write_string(2, &self.name);
        w.write_string(3, &self.description);
        w.into_bytes()
    }

    /// Decodes an [`XStatMetadataProto`] from protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `XStatMetadata` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        let mut r = ProtoReader::new(bytes, "tensorflow.profiler.XStatMetadata");
        let mut out = Self::default();
        while !r.is_eof() {
            let (field, wt) = r.read_tag()?;
            match (field, wt) {
                (1, WireType::Varint) => out.id = r.read_int64()?,
                (2, WireType::LengthDelimited) => out.name = r.read_string()?,
                (3, WireType::LengthDelimited) => out.description = r.read_string()?,
                _ => r.skip_field(wt)?,
            }
        }
        Ok(out)
    }
}

/// Wire-compatible `tensorflow.profiler.XEventMetadata` (`xplane.proto`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct XEventMetadataProto {
    /// Metadata ID (field `1`).
    pub id: i64,
    /// Event name (field `2`).
    pub name: String,
    /// Display name (field `3`).
    pub display_name: String,
}

impl XEventMetadataProto {
    /// Encodes this [`XEventMetadataProto`] to protobuf wire bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ProtoWriter::new();
        w.write_int64(1, self.id);
        w.write_string(2, &self.name);
        w.write_string(3, &self.display_name);
        w.into_bytes()
    }

    /// Decodes an [`XEventMetadataProto`] from protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `XEventMetadata` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        let mut r = ProtoReader::new(bytes, "tensorflow.profiler.XEventMetadata");
        let mut out = Self::default();
        while !r.is_eof() {
            let (field, wt) = r.read_tag()?;
            match (field, wt) {
                (1, WireType::Varint) => out.id = r.read_int64()?,
                (2, WireType::LengthDelimited) => out.name = r.read_string()?,
                (3, WireType::LengthDelimited) => out.display_name = r.read_string()?,
                _ => r.skip_field(wt)?,
            }
        }
        Ok(out)
    }
}

/// Wire-compatible `tpu.TpuTopologyArgsProto` (`tpu_topology.proto`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TpuTopologyArgsProto {
    /// TPU platform generation enum (`TpuVersionEnum`, field `1`).
    pub version: i32,
    /// Number of hosts in the pod slice (field `2`).
    pub num_hosts: i32,
    /// Number of chips per host (field `3`).
    pub chips_per_host: i32,
    /// Mesh dimensions `[x, y, z]` (field `4`).
    pub mesh_bounds: Vec<i64>,
    /// Platform version string (field `5`).
    pub platform_version: String,
    /// Deterministic topology fingerprint (field `6`).
    pub fingerprint: String,
}

impl TpuTopologyArgsProto {
    /// Encodes this [`TpuTopologyArgsProto`] to protobuf wire bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ProtoWriter::new();
        w.write_int32(1, self.version);
        w.write_int32(2, self.num_hosts);
        w.write_int32(3, self.chips_per_host);
        w.write_packed_int64(4, &self.mesh_bounds);
        w.write_string(5, &self.platform_version);
        w.write_string(6, &self.fingerprint);
        w.into_bytes()
    }

    /// Decodes a [`TpuTopologyArgsProto`] from protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `TpuTopologyArgsProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        let mut r = ProtoReader::new(bytes, "tpu.TpuTopologyArgsProto");
        let mut out = Self::default();
        while !r.is_eof() {
            let (field, wt) = r.read_tag()?;
            match (field, wt) {
                (1, WireType::Varint) => out.version = r.read_int32()?,
                (2, WireType::Varint) => out.num_hosts = r.read_int32()?,
                (3, WireType::Varint) => out.chips_per_host = r.read_int32()?,
                (4, _) => r.read_repeated_int64(wt, &mut out.mesh_bounds)?,
                (5, WireType::LengthDelimited) => out.platform_version = r.read_string()?,
                (6, WireType::LengthDelimited) => out.fingerprint = r.read_string()?,
                _ => r.skip_field(wt)?,
            }
        }
        Ok(out)
    }
}

/// Wire-compatible `tpu.TpuSerializedExecutableProto` (`tpu_executable.proto`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TpuSerializedExecutableProto {
    /// Executable name (field `1`).
    pub name: String,
    /// SHA-256 fingerprint hex string (field `2`).
    pub fingerprint: String,
    /// Number of replicas (field `3`).
    pub num_replicas: i32,
    /// Number of partitions (field `4`).
    pub num_partitions: i32,
    /// Encoded VLIW instruction bundle opcodes (field `5`).
    pub instruction_opcodes: Vec<i64>,
    /// Output shape protos (field `6`).
    pub output_shapes: Vec<ShapeProto>,
    /// Serialized `CompileOptionsProto` (field `7`).
    pub compile_options_bytes: Vec<u8>,
    /// Serialized HLO / `StableHLO` module text or proto (field `8`).
    pub hlo_module_bytes: Vec<u8>,
}

/// Type alias for [`TpuSerializedExecutableProto`] (`TpuExecutableSerializedProto`).
pub type TpuExecutableSerializedProto = TpuSerializedExecutableProto;

impl TpuSerializedExecutableProto {
    /// Encodes this [`TpuSerializedExecutableProto`] to protobuf wire bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ProtoWriter::new();
        w.write_string(1, &self.name);
        w.write_string(2, &self.fingerprint);
        w.write_int32(3, self.num_replicas);
        w.write_int32(4, self.num_partitions);
        w.write_packed_int64(5, &self.instruction_opcodes);
        for shape in &self.output_shapes {
            w.write_message(6, &shape.encode());
        }
        w.write_bytes(7, &self.compile_options_bytes);
        w.write_bytes(8, &self.hlo_module_bytes);
        w.into_bytes()
    }

    /// Decodes a [`TpuSerializedExecutableProto`] from protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `TpuSerializedExecutableProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        let mut r = ProtoReader::new(bytes, "tpu.TpuSerializedExecutableProto");
        let mut out = Self::default();
        while !r.is_eof() {
            let (field, wt) = r.read_tag()?;
            match (field, wt) {
                (1, WireType::LengthDelimited) => out.name = r.read_string()?,
                (2, WireType::LengthDelimited) => out.fingerprint = r.read_string()?,
                (3, WireType::Varint) => out.num_replicas = r.read_int32()?,
                (4, WireType::Varint) => out.num_partitions = r.read_int32()?,
                (5, _) => r.read_repeated_int64(wt, &mut out.instruction_opcodes)?,
                (6, WireType::LengthDelimited) => {
                    let sub = r.read_bytes()?;
                    out.output_shapes.push(ShapeProto::decode(sub)?);
                }
                (7, WireType::LengthDelimited) => {
                    out.compile_options_bytes = r.read_bytes()?.to_vec();
                }
                (8, WireType::LengthDelimited) => {
                    out.hlo_module_bytes = r.read_bytes()?.to_vec();
                }
                _ => r.skip_field(wt)?,
            }
        }
        Ok(out)
    }
}

/// Wire-compatible `tensorflow.profiler.XStat` (`xplane.proto`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct XStatProto {
    /// Metadata ID (field `1`).
    pub metadata_id: i64,
    /// Integer stat value (field `3`).
    pub int64_value: i64,
    /// String stat value (field `6`).
    pub str_value: String,
}

impl XStatProto {
    /// Encodes this [`XStatProto`] to protobuf wire bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ProtoWriter::new();
        w.write_int64(1, self.metadata_id);
        w.write_int64(3, self.int64_value);
        w.write_string(6, &self.str_value);
        w.into_bytes()
    }

    /// Decodes an [`XStatProto`] from protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `XStat` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        let mut r = ProtoReader::new(bytes, "tensorflow.profiler.XStat");
        let mut out = Self::default();
        while !r.is_eof() {
            let (field, wt) = r.read_tag()?;
            match (field, wt) {
                (1, WireType::Varint) => out.metadata_id = r.read_int64()?,
                (3, WireType::Varint) => out.int64_value = r.read_int64()?,
                (6, WireType::LengthDelimited) => out.str_value = r.read_string()?,
                _ => r.skip_field(wt)?,
            }
        }
        Ok(out)
    }
}

/// Wire-compatible `tensorflow.profiler.XEvent` (`xplane.proto`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct XEventProto {
    /// Event metadata ID (field `1`).
    pub metadata_id: i64,
    /// Offset in picoseconds from the line start timestamp (field `2`).
    pub offset_ps: i64,
    /// Duration in picoseconds (field `3`).
    pub duration_ps: i64,
    /// Attached event statistics (field `5`).
    pub stats: Vec<XStatProto>,
}

impl XEventProto {
    /// Encodes this [`XEventProto`] to protobuf wire bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ProtoWriter::new();
        w.write_int64(1, self.metadata_id);
        w.write_int64(2, self.offset_ps);
        w.write_int64(3, self.duration_ps);
        for st in &self.stats {
            w.write_message(5, &st.encode());
        }
        w.into_bytes()
    }

    /// Decodes an [`XEventProto`] from protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `XEvent` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        let mut r = ProtoReader::new(bytes, "tensorflow.profiler.XEvent");
        let mut out = Self::default();
        while !r.is_eof() {
            let (field, wt) = r.read_tag()?;
            match (field, wt) {
                (1, WireType::Varint) => out.metadata_id = r.read_int64()?,
                (2, WireType::Varint) => out.offset_ps = r.read_int64()?,
                (3, WireType::Varint) => out.duration_ps = r.read_int64()?,
                (5, WireType::LengthDelimited) => {
                    let sub = r.read_bytes()?;
                    out.stats.push(XStatProto::decode(sub)?);
                }
                _ => r.skip_field(wt)?,
            }
        }
        Ok(out)
    }
}

/// Wire-compatible `tensorflow.profiler.XLine` (`xplane.proto`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct XLineProto {
    /// Line ID (field `1`).
    pub id: i64,
    /// Line name (e.g., `"TensorCore 0"`, field `2`).
    pub name: String,
    /// Start timestamp in nanoseconds (field `3`).
    pub timestamp_ns: i64,
    /// Events recorded on this timeline (field `4`).
    pub events: Vec<XEventProto>,
}

impl XLineProto {
    /// Encodes this [`XLineProto`] to protobuf wire bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ProtoWriter::new();
        w.write_int64(1, self.id);
        w.write_string(2, &self.name);
        w.write_int64(3, self.timestamp_ns);
        for ev in &self.events {
            w.write_message(4, &ev.encode());
        }
        w.into_bytes()
    }

    /// Decodes an [`XLineProto`] from protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `XLine` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        let mut r = ProtoReader::new(bytes, "tensorflow.profiler.XLine");
        let mut out = Self::default();
        while !r.is_eof() {
            let (field, wt) = r.read_tag()?;
            match (field, wt) {
                (1, WireType::Varint) => out.id = r.read_int64()?,
                (2, WireType::LengthDelimited) => out.name = r.read_string()?,
                (3, WireType::Varint) => out.timestamp_ns = r.read_int64()?,
                (4, WireType::LengthDelimited) => {
                    let sub = r.read_bytes()?;
                    out.events.push(XEventProto::decode(sub)?);
                }
                _ => r.skip_field(wt)?,
            }
        }
        Ok(out)
    }
}

/// Wire-compatible `tensorflow.profiler.XPlane` (`xplane.proto`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct XPlaneProto {
    /// Plane ID (field `1`).
    pub id: i64,
    /// Plane name (e.g., `"/device:TPU:0"`, field `2`).
    pub name: String,
    /// Timelines within this plane (field `3`).
    pub lines: Vec<XLineProto>,
    /// Plane-level statistics (field `6`).
    pub stats: Vec<XStatProto>,
}

impl XPlaneProto {
    /// Encodes this [`XPlaneProto`] to protobuf wire bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ProtoWriter::new();
        w.write_int64(1, self.id);
        w.write_string(2, &self.name);
        for line in &self.lines {
            w.write_message(3, &line.encode());
        }
        for st in &self.stats {
            w.write_message(6, &st.encode());
        }
        w.into_bytes()
    }

    /// Decodes an [`XPlaneProto`] from protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `XPlane` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        let mut r = ProtoReader::new(bytes, "tensorflow.profiler.XPlane");
        let mut out = Self::default();
        while !r.is_eof() {
            let (field, wt) = r.read_tag()?;
            match (field, wt) {
                (1, WireType::Varint) => out.id = r.read_int64()?,
                (2, WireType::LengthDelimited) => out.name = r.read_string()?,
                (3, WireType::LengthDelimited) => {
                    let sub = r.read_bytes()?;
                    out.lines.push(XLineProto::decode(sub)?);
                }
                (6, WireType::LengthDelimited) => {
                    let sub = r.read_bytes()?;
                    out.stats.push(XStatProto::decode(sub)?);
                }
                _ => r.skip_field(wt)?,
            }
        }
        Ok(out)
    }
}

/// Wire-compatible `tensorflow.profiler.XSpace` (`xplane.proto`) returned by `TpuProfiler_CollectData`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct XSpaceProto {
    /// Planes in the trace (field `1`).
    pub planes: Vec<XPlaneProto>,
    /// Hostnames participating in the trace (field `3`).
    pub hostnames: Vec<String>,
}

impl XSpaceProto {
    /// Encodes this [`XSpaceProto`] to protobuf wire bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ProtoWriter::new();
        for plane in &self.planes {
            w.write_message(1, &plane.encode());
        }
        for host in &self.hostnames {
            w.write_string(3, host);
        }
        w.into_bytes()
    }

    /// Decodes an [`XSpaceProto`] from protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `XSpace` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        let mut r = ProtoReader::new(bytes, "tensorflow.profiler.XSpace");
        let mut out = Self::default();
        while !r.is_eof() {
            let (field, wt) = r.read_tag()?;
            match (field, wt) {
                (1, WireType::LengthDelimited) => {
                    let sub = r.read_bytes()?;
                    out.planes.push(XPlaneProto::decode(sub)?);
                }
                (3, WireType::LengthDelimited) => out.hostnames.push(r.read_string()?),
                _ => r.skip_field(wt)?,
            }
        }
        Ok(out)
    }
}

/// Wire-compatible `tensorflow.tpu.TPUEmbeddingConfiguration.TableDescriptor`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EmbeddingTableDescriptorProto {
    /// Table name (field `1`).
    pub name: String,
    /// Vocabulary size (number of rows, field `2`).
    pub vocabulary_size: i64,
    /// Embedding dimension width (field `3`).
    pub dimension: i64,
    /// Number of features referencing this table (field `4`).
    pub num_features: i64,
}

impl EmbeddingTableDescriptorProto {
    /// Encodes this [`EmbeddingTableDescriptorProto`] to protobuf wire bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ProtoWriter::new();
        w.write_string(1, &self.name);
        w.write_int64(2, self.vocabulary_size);
        w.write_int64(3, self.dimension);
        w.write_int64(4, self.num_features);
        w.into_bytes()
    }

    /// Decodes an [`EmbeddingTableDescriptorProto`] from protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `TableDescriptor` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        let mut r = ProtoReader::new(bytes, "tensorflow.tpu.TableDescriptor");
        let mut out = Self::default();
        while !r.is_eof() {
            let (field, wt) = r.read_tag()?;
            match (field, wt) {
                (1, WireType::LengthDelimited) => out.name = r.read_string()?,
                (2, WireType::Varint) => out.vocabulary_size = r.read_int64()?,
                (3, WireType::Varint) => out.dimension = r.read_int64()?,
                (4, WireType::Varint) => out.num_features = r.read_int64()?,
                _ => r.skip_field(wt)?,
            }
        }
        Ok(out)
    }
}

/// Wire-compatible `tensorflow.tpu.TPUEmbeddingConfiguration`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TpuEmbeddingConfigurationProto {
    /// Table descriptors (field `1`).
    pub table_descriptor: Vec<EmbeddingTableDescriptorProto>,
    /// Mode (`1` = Inference, `2` = Training, field `2`).
    pub mode: i32,
    /// Batch size per `TensorCore` (field `3`).
    pub batch_size_per_tensor_core: i32,
    /// Number of hosts (field `4`).
    pub num_hosts: i32,
    /// Number of `TensorCores` (field `5`).
    pub num_tensor_cores: i32,
}

impl TpuEmbeddingConfigurationProto {
    /// Encodes this [`TpuEmbeddingConfigurationProto`] to protobuf wire bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ProtoWriter::new();
        for tbl in &self.table_descriptor {
            w.write_message(1, &tbl.encode());
        }
        w.write_int32(2, self.mode);
        w.write_int32(3, self.batch_size_per_tensor_core);
        w.write_int32(4, self.num_hosts);
        w.write_int32(5, self.num_tensor_cores);
        w.into_bytes()
    }

    /// Decodes a [`TpuEmbeddingConfigurationProto`] from protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `TPUEmbeddingConfiguration` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        let mut r = ProtoReader::new(bytes, "tensorflow.tpu.TPUEmbeddingConfiguration");
        let mut out = Self::default();
        while !r.is_eof() {
            let (field, wt) = r.read_tag()?;
            match (field, wt) {
                (1, WireType::LengthDelimited) => {
                    let sub = r.read_bytes()?;
                    out.table_descriptor
                        .push(EmbeddingTableDescriptorProto::decode(sub)?);
                }
                (2, WireType::Varint) => out.mode = r.read_int32()?,
                (3, WireType::Varint) => out.batch_size_per_tensor_core = r.read_int32()?,
                (4, WireType::Varint) => out.num_hosts = r.read_int32()?,
                (5, WireType::Varint) => out.num_tensor_cores = r.read_int32()?,
                _ => r.skip_field(wt)?,
            }
        }
        Ok(out)
    }
}

/// Wire-compatible `xla.HloModuleProtoWithConfig` pairing an [`HloModuleProto`] with compilation flags.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HloModuleProtoWithConfig {
    /// Embedded `xla.HloModuleProto` (field `1`).
    pub hlo_module: Option<HloModuleProto>,
    /// Embedded `xla.ExecutableBuildOptionsProto` config (field `2`).
    pub config: Option<ExecutableBuildOptionsProto>,
}

impl HloModuleProtoWithConfig {
    /// Encodes this [`HloModuleProtoWithConfig`] to protobuf wire bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ProtoWriter::new();
        if let Some(m) = &self.hlo_module {
            w.write_message(1, &m.encode());
        }
        if let Some(c) = &self.config {
            w.write_message(2, &c.encode());
        }
        w.into_bytes()
    }

    /// Decodes an [`HloModuleProtoWithConfig`] from protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `HloModuleProtoWithConfig` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        let mut r = ProtoReader::new(bytes, "xla.HloModuleProtoWithConfig");
        let mut out = Self::default();
        while !r.is_eof() {
            let (field, wt) = r.read_tag()?;
            match (field, wt) {
                (1, WireType::LengthDelimited) => {
                    let sub = r.read_bytes()?;
                    out.hlo_module = Some(HloModuleProto::decode(sub)?);
                }
                (2, WireType::LengthDelimited) => {
                    let sub = r.read_bytes()?;
                    out.config = Some(ExecutableBuildOptionsProto::decode(sub)?);
                }
                _ => r.skip_field(wt)?,
            }
        }
        Ok(out)
    }
}

/// Wire-compatible `pjrt.RuntimeAbiVersionProto` for `PJRT_AbiVersion_Extension`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RuntimeAbiVersionProto {
    /// Platform ID (field `1`).
    pub platform_id: u64,
    /// Runtime ABI version string (field `2`).
    pub version_string: String,
    /// PJRT API minor version (field `3`).
    pub pjrt_api_minor: i32,
}

impl RuntimeAbiVersionProto {
    /// Encodes this [`RuntimeAbiVersionProto`] to protobuf wire bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ProtoWriter::new();
        w.write_uint64(1, self.platform_id);
        w.write_string(2, &self.version_string);
        w.write_int32(3, self.pjrt_api_minor);
        w.into_bytes()
    }

    /// Decodes a [`RuntimeAbiVersionProto`] from protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `RuntimeAbiVersionProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        let mut r = ProtoReader::new(bytes, "pjrt.RuntimeAbiVersionProto");
        let mut out = Self::default();
        while !r.is_eof() {
            let (field, wt) = r.read_tag()?;
            match (field, wt) {
                (1, WireType::Varint) => out.platform_id = r.read_varint()?,
                (2, WireType::LengthDelimited) => out.version_string = r.read_string()?,
                (3, WireType::Varint) => out.pjrt_api_minor = r.read_int32()?,
                _ => r.skip_field(wt)?,
            }
        }
        Ok(out)
    }
}

/// Wire-compatible `pjrt.ExecutableAbiVersionProto` for `PJRT_AbiVersion_Extension`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExecutableAbiVersionProto {
    /// Platform ID (field `1`).
    pub platform_id: u64,
    /// Compiler ABI version string (field `2`).
    pub compiler_version: String,
    /// Target TPU generation code (field `3`).
    pub target_generation: i32,
}

impl ExecutableAbiVersionProto {
    /// Encodes this [`ExecutableAbiVersionProto`] to protobuf wire bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ProtoWriter::new();
        w.write_uint64(1, self.platform_id);
        w.write_string(2, &self.compiler_version);
        w.write_int32(3, self.target_generation);
        w.into_bytes()
    }

    /// Decodes an [`ExecutableAbiVersionProto`] from protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `ExecutableAbiVersionProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        let mut r = ProtoReader::new(bytes, "pjrt.ExecutableAbiVersionProto");
        let mut out = Self::default();
        while !r.is_eof() {
            let (field, wt) = r.read_tag()?;
            match (field, wt) {
                (1, WireType::Varint) => out.platform_id = r.read_varint()?,
                (2, WireType::LengthDelimited) => out.compiler_version = r.read_string()?,
                (3, WireType::Varint) => out.target_generation = r.read_int32()?,
                _ => r.skip_field(wt)?,
            }
        }
        Ok(out)
    }
}

/// Wire-compatible `tpu.TpuEmbeddingPartitionedConfigProto` produced by `TpuEmbeddingEngine_ExecutePartitioner`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TpuEmbeddingPartitionedConfigProto {
    /// Number of SparseCore partitions across the slice (field `1`).
    pub num_partitions: i32,
    /// Partitioned table names (field `2`).
    pub table_names: Vec<String>,
    /// Original serialized `TpuEmbeddingConfigurationProto` bytes (field `3`).
    pub raw_config_bytes: Vec<u8>,
}

impl TpuEmbeddingPartitionedConfigProto {
    /// Encodes this [`TpuEmbeddingPartitionedConfigProto`] to protobuf wire bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ProtoWriter::new();
        w.write_int32(1, self.num_partitions);
        for name in &self.table_names {
            w.write_string(2, name);
        }
        w.write_bytes(3, &self.raw_config_bytes);
        w.into_bytes()
    }

    /// Decodes a [`TpuEmbeddingPartitionedConfigProto`] from protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `TpuEmbeddingPartitionedConfigProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        let mut r = ProtoReader::new(bytes, "tpu.TpuEmbeddingPartitionedConfigProto");
        let mut out = Self::default();
        while !r.is_eof() {
            let (field, wt) = r.read_tag()?;
            match (field, wt) {
                (1, WireType::Varint) => out.num_partitions = r.read_int32()?,
                (2, WireType::LengthDelimited) => out.table_names.push(r.read_string()?),
                (3, WireType::LengthDelimited) => out.raw_config_bytes = r.read_bytes()?.to_vec(),
                _ => r.skip_field(wt)?,
            }
        }
        Ok(out)
    }
}

/// Wire-compatible `tpu.TpuEmbeddingMemoryConfigProto` used by `TpuEmbeddingEngine_ConfigureMemory`
/// and `TpuEmbeddingEngine_CollateMemory`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TpuEmbeddingMemoryConfigProto {
    /// Total HBM bytes reserved for SparseCore lookup tables (field `1`).
    pub hbm_buffers_bytes: u64,
    /// Number of collated host memory segments (field `2`).
    pub num_segments: i32,
    /// Collated per-host memory configuration blobs (field `3`).
    pub segment_blobs: Vec<Vec<u8>>,
}

impl TpuEmbeddingMemoryConfigProto {
    /// Encodes this [`TpuEmbeddingMemoryConfigProto`] to protobuf wire bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ProtoWriter::new();
        w.write_uint64(1, self.hbm_buffers_bytes);
        w.write_int32(2, self.num_segments);
        for blob in &self.segment_blobs {
            w.write_bytes(3, blob);
        }
        w.into_bytes()
    }

    /// Decodes a [`TpuEmbeddingMemoryConfigProto`] from protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `TpuEmbeddingMemoryConfigProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        let mut r = ProtoReader::new(bytes, "tpu.TpuEmbeddingMemoryConfigProto");
        let mut out = Self::default();
        while !r.is_eof() {
            let (field, wt) = r.read_tag()?;
            match (field, wt) {
                (1, WireType::Varint) => out.hbm_buffers_bytes = r.read_varint()?,
                (2, WireType::Varint) => out.num_segments = r.read_int32()?,
                (3, WireType::LengthDelimited) => {
                    out.segment_blobs.push(r.read_bytes()?.to_vec());
                }
                _ => r.skip_field(wt)?,
            }
        }
        Ok(out)
    }
}

/// Wire-compatible `tpu.TpuEmbeddingHostNetworkConfigProto` used by `TpuEmbeddingEngine_ConfigureHost`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TpuEmbeddingHostNetworkConfigProto {
    /// Host ordinal index (field `1`).
    pub host_id: i32,
    /// Network endpoint address (field `2`).
    pub endpoint_address: String,
    /// Raw partition configuration payload (field `3`).
    pub partition_payload: Vec<u8>,
}

impl TpuEmbeddingHostNetworkConfigProto {
    /// Encodes this [`TpuEmbeddingHostNetworkConfigProto`] to protobuf wire bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ProtoWriter::new();
        w.write_int32(1, self.host_id);
        w.write_string(2, &self.endpoint_address);
        w.write_bytes(3, &self.partition_payload);
        w.into_bytes()
    }

    /// Decodes a [`TpuEmbeddingHostNetworkConfigProto`] from protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `TpuEmbeddingHostNetworkConfigProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        let mut r = ProtoReader::new(bytes, "tpu.TpuEmbeddingHostNetworkConfigProto");
        let mut out = Self::default();
        while !r.is_eof() {
            let (field, wt) = r.read_tag()?;
            match (field, wt) {
                (1, WireType::Varint) => out.host_id = r.read_int32()?,
                (2, WireType::LengthDelimited) => out.endpoint_address = r.read_string()?,
                (3, WireType::LengthDelimited) => out.partition_payload = r.read_bytes()?.to_vec(),
                _ => r.skip_field(wt)?,
            }
        }
        Ok(out)
    }
}

/// Wire-compatible `megascale.MultiSliceConfigProto` for `PJRT_MultiSlice_Extension` and `PJRT_Megascale_Extension`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MegascaleMultiSliceConfigProto {
    /// Total number of slices in the multi-slice training job (field `1`).
    pub num_slices: i32,
    /// Local slice identifier (`0..num_slices`, field `2`).
    pub slice_id: i32,
    /// Number of logical TPU devices per slice (field `3`).
    pub num_devices_per_slice: i32,
    /// Coordinator network endpoint address (field `4`).
    pub coordinator_address: String,
}

impl MegascaleMultiSliceConfigProto {
    /// Encodes this [`MegascaleMultiSliceConfigProto`] to protobuf wire bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ProtoWriter::new();
        w.write_int32(1, self.num_slices);
        w.write_int32(2, self.slice_id);
        w.write_int32(3, self.num_devices_per_slice);
        w.write_string(4, &self.coordinator_address);
        w.into_bytes()
    }

    /// Decodes a [`MegascaleMultiSliceConfigProto`] from protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `MultiSliceConfigProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        let mut r = ProtoReader::new(bytes, "megascale.MultiSliceConfigProto");
        let mut out = Self::default();
        while !r.is_eof() {
            let (field, wt) = r.read_tag()?;
            match (field, wt) {
                (1, WireType::Varint) => out.num_slices = r.read_int32()?,
                (2, WireType::Varint) => out.slice_id = r.read_int32()?,
                (3, WireType::Varint) => out.num_devices_per_slice = r.read_int32()?,
                (4, WireType::LengthDelimited) => out.coordinator_address = r.read_string()?,
                _ => r.skip_field(wt)?,
            }
        }
        Ok(out)
    }
}

/// Wire-compatible `megascale.DCNTopologyProto` for `PJRT_Megascale_GetDcnTopology`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MegascaleDcnTopologyProto {
    /// Number of DCN slices (field `1`).
    pub num_slices: i32,
    /// Number of hosts per slice (field `2`).
    pub num_hosts_per_slice: i32,
    /// Transport type (`0` = Default/TCP, `1` = RoCEv2, `2` = ICI, field `3`).
    pub transport_type: i32,
}

impl MegascaleDcnTopologyProto {
    /// Encodes this [`MegascaleDcnTopologyProto`] to protobuf wire bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ProtoWriter::new();
        w.write_int32(1, self.num_slices);
        w.write_int32(2, self.num_hosts_per_slice);
        w.write_int32(3, self.transport_type);
        w.into_bytes()
    }

    /// Decodes a [`MegascaleDcnTopologyProto`] from protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `DCNTopologyProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        let mut r = ProtoReader::new(bytes, "megascale.DCNTopologyProto");
        let mut out = Self::default();
        while !r.is_eof() {
            let (field, wt) = r.read_tag()?;
            match (field, wt) {
                (1, WireType::Varint) => out.num_slices = r.read_int32()?,
                (2, WireType::Varint) => out.num_hosts_per_slice = r.read_int32()?,
                (3, WireType::Varint) => out.transport_type = r.read_int32()?,
                _ => r.skip_field(wt)?,
            }
        }
        Ok(out)
    }
}

/// Wire-compatible `megascale.RuntimeMetricsProto` for `PJRT_Megascale_CollectMetrics`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MegascaleRuntimeMetricsProto {
    /// Number of completed cross-slice barriers (field `1`).
    pub completed_barriers: u64,
    /// Total collective operations executed (field `2`).
    pub collective_ops_count: u64,
    /// Active Megascale contexts (field `3`).
    pub active_contexts: i32,
}

impl MegascaleRuntimeMetricsProto {
    /// Encodes this [`MegascaleRuntimeMetricsProto`] to protobuf wire bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ProtoWriter::new();
        w.write_uint64(1, self.completed_barriers);
        w.write_uint64(2, self.collective_ops_count);
        w.write_int32(3, self.active_contexts);
        w.into_bytes()
    }

    /// Decodes a [`MegascaleRuntimeMetricsProto`] from protobuf wire bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized `RuntimeMetricsProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if decoding fails.
    pub fn decode(bytes: &[u8]) -> LibTpuResult<Self> {
        let mut r = ProtoReader::new(bytes, "megascale.RuntimeMetricsProto");
        let mut out = Self::default();
        while !r.is_eof() {
            let (field, wt) = r.read_tag()?;
            match (field, wt) {
                (1, WireType::Varint) => out.completed_barriers = r.read_varint()?,
                (2, WireType::Varint) => out.collective_ops_count = r.read_varint()?,
                (3, WireType::Varint) => out.active_contexts = r.read_int32()?,
                _ => r.skip_field(wt)?,
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_wire_primitives_and_error_paths() {
        assert_eq!(WireType::from_raw(0, "Test").ok(), Some(WireType::Varint));
        assert_eq!(WireType::from_raw(1, "Test").ok(), Some(WireType::Fixed64));
        assert_eq!(
            WireType::from_raw(2, "Test").ok(),
            Some(WireType::LengthDelimited)
        );
        assert_eq!(WireType::from_raw(5, "Test").ok(), Some(WireType::Fixed32));
        assert!(WireType::from_raw(3, "Test").is_err());

        let mut w = ProtoWriter::new();
        w.write_uint64(1, 0);
        w.write_uint64(1, 300);
        w.write_bool(2, false);
        w.write_bool(2, true);
        w.write_double(3, 0.0);
        w.write_double(3, 2.5);
        w.write_float(4, 0.0);
        w.write_float(4, 1.5);
        w.write_bytes(5, &[]);
        w.write_packed_int64(6, &[]);
        let bytes = w.into_bytes();

        let mut r = ProtoReader::new(&bytes, "Test");
        while !r.is_eof() {
            let (_f, wt) = r.read_tag().unwrap_or((1, WireType::Varint));
            assert!(r.skip_field(wt).is_ok());
        }

        // Unpacked repeated int64 & invalid wire type for repeated int64
        let mut w2 = ProtoWriter::new();
        w2.write_int64(1, 42);
        w2.write_double(1, 1.0);
        let b2 = w2.into_bytes();
        let mut r2 = ProtoReader::new(&b2, "Test");
        let mut vals = Vec::new();
        let (_, wt1) = r2.read_tag().unwrap_or((1, WireType::Varint));
        assert!(r2.read_repeated_int64(wt1, &mut vals).is_ok());
        assert_eq!(vals, vec![42]);
        let (_, wt2) = r2.read_tag().unwrap_or((1, WireType::Fixed64));
        assert!(r2.read_repeated_int64(wt2, &mut vals).is_err());

        // Error conditions: empty varint, 10-byte varint overflow, field 0, truncated fixed64/32/bytes, invalid utf8
        let mut r_empty = ProtoReader::new(&[], "Err");
        assert!(r_empty.read_varint().is_err());

        let overlong = [0x80_u8; 11];
        let mut r_over = ProtoReader::new(&overlong, "Err");
        assert!(r_over.read_varint().is_err());

        let field_zero = [0x00_u8];
        let mut r_zero = ProtoReader::new(&field_zero, "Err");
        assert!(r_zero.read_tag().is_err());

        let field_overflow = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x7F];
        let mut r_fover = ProtoReader::new(&field_overflow, "Err");
        assert!(r_fover.read_tag().is_err());
        let mut r_i32_over = ProtoReader::new(&field_overflow, "Err");
        assert!(r_i32_over.read_int32().is_err());

        let mut r_short = ProtoReader::new(&[1, 2], "Err");
        assert!(r_short.read_fixed64().is_err());
        assert!(r_short.read_fixed32().is_err());

        let mut r_pos_max = ProtoReader {
            data: &[1, 2],
            pos: usize::MAX,
            message_type: "Err",
        };
        assert!(r_pos_max.read_fixed64().is_err());
        assert!(r_pos_max.read_fixed32().is_err());

        let mut w_bad_len = ProtoWriter::new();
        w_bad_len.write_varint(100);
        let bad_len_bytes = w_bad_len.into_bytes();
        let mut r_bad_len = ProtoReader::new(&bad_len_bytes, "Err");
        assert!(r_bad_len.read_bytes().is_err());

        let mut w_huge_len = ProtoWriter::new();
        w_huge_len.write_varint(u64::MAX);
        let huge_bytes = w_huge_len.into_bytes();
        let mut r_huge = ProtoReader::new(&huge_bytes, "Err");
        assert!(r_huge.read_bytes().is_err());

        let mut w_bad_utf8 = ProtoWriter::new();
        w_bad_utf8.write_varint(2);
        let mut bad_utf8_bytes = w_bad_utf8.into_bytes();
        bad_utf8_bytes.extend_from_slice(&[0xFF, 0xFE]);
        let mut r_utf8 = ProtoReader::new(&bad_utf8_bytes, "Err");
        assert!(r_utf8.read_string().is_err());

        // Exercise ? error branches in read_tag, read_bytes, read_string, read_repeated_int64, and skip_field
        let mut r_eof = ProtoReader::new(&[], "Err");
        assert!(r_eof.read_tag().is_err());
        assert!(r_eof.read_bytes().is_err());
        assert!(r_eof.read_string().is_err());
        let mut dummy_vec = Vec::new();
        assert!(r_eof
            .read_repeated_int64(WireType::Varint, &mut dummy_vec)
            .is_err());
        assert!(r_eof
            .read_repeated_int64(WireType::LengthDelimited, &mut dummy_vec)
            .is_err());
        assert!(r_eof.skip_field(WireType::Varint).is_err());
        assert!(r_eof.skip_field(WireType::Fixed64).is_err());
        assert!(r_eof.skip_field(WireType::LengthDelimited).is_err());
        assert!(r_eof.skip_field(WireType::Fixed32).is_err());

        let mut w_trunc_packed = ProtoWriter::new();
        w_trunc_packed.write_bytes(1, &[0x80]);
        let trunc_packed_bytes = w_trunc_packed.into_bytes();
        let mut r_tp = ProtoReader::new(&trunc_packed_bytes, "Err");
        let _ = r_tp.read_tag();
        assert!(r_tp
            .read_repeated_int64(WireType::LengthDelimited, &mut dummy_vec)
            .is_err());
    }

    #[test]
    fn test_all_protobuf_messages_roundtrip_and_unknown_tags() {
        let tile = TileProto {
            dimensions: vec![8, 128],
        };
        let mut tile_bytes = tile.encode();
        let mut extra = ProtoWriter::new();
        extra.write_uint64(99, 1);
        tile_bytes.extend_from_slice(&extra.clone().into_bytes());
        assert_eq!(TileProto::decode(&tile_bytes).ok(), Some(tile.clone()));

        let layout = LayoutProto {
            minor_to_major: vec![1, 0],
            tiles: vec![tile],
            element_size_in_bits: 32,
            memory_space: 1,
        };
        let mut layout_bytes = layout.encode();
        layout_bytes.extend_from_slice(&extra.clone().into_bytes());
        assert_eq!(
            LayoutProto::decode(&layout_bytes).ok(),
            Some(layout.clone())
        );

        let sub_shape = ShapeProto {
            element_type: 11,
            dimensions: vec![4],
            tuple_shapes: vec![],
            layout: None,
            is_dynamic_dimension: vec![false],
        };
        let shape = ShapeProto {
            element_type: 11,
            dimensions: vec![8, 128],
            tuple_shapes: vec![sub_shape],
            layout: Some(layout),
            is_dynamic_dimension: vec![false, true],
        };
        let mut shape_bytes = shape.encode();
        shape_bytes.extend_from_slice(&extra.clone().into_bytes());
        assert_eq!(ShapeProto::decode(&shape_bytes).ok(), Some(shape.clone()));

        let sharding = OpShardingProto {
            sharding_type: 3,
            tile_assignment_dimensions: vec![2, 2],
            tile_assignment_devices: vec![0, 1, 2, 3],
        };
        let mut sharding_bytes = sharding.encode();
        sharding_bytes.extend_from_slice(&extra.clone().into_bytes());
        assert_eq!(
            OpShardingProto::decode(&sharding_bytes).ok(),
            Some(sharding)
        );

        let inst = HloInstructionProto {
            name: String::from("add.1"),
            opcode: String::from("add"),
            shape: Some(shape.clone()),
            id: 3,
            operand_ids: vec![1, 2],
            parameter_number: 1,
            custom_call_target: String::from("TpuCustomCall"),
        };
        let mut inst_bytes = inst.encode();
        inst_bytes.extend_from_slice(&extra.clone().into_bytes());
        assert_eq!(
            HloInstructionProto::decode(&inst_bytes).ok(),
            Some(inst.clone())
        );

        let comp = HloComputationProto {
            name: String::from("main"),
            instructions: vec![inst],
            id: 1,
            root_id: 3,
        };
        let mut comp_bytes = comp.encode();
        comp_bytes.extend_from_slice(&extra.clone().into_bytes());
        assert_eq!(
            HloComputationProto::decode(&comp_bytes).ok(),
            Some(comp.clone())
        );

        let module = HloModuleProto {
            name: String::from("test_module"),
            entry_computation_name: String::from("main"),
            entry_computation_id: 1,
            computations: vec![comp],
            id: 42,
        };
        let mut mod_bytes = module.encode();
        mod_bytes.extend_from_slice(&extra.clone().into_bytes());
        assert_eq!(HloModuleProto::decode(&mod_bytes).ok(), Some(module));

        let build_opts = ExecutableBuildOptionsProto {
            device_ordinal: 1,
            result_layout: Some(shape.clone()),
            num_replicas: 1,
            num_partitions: 2,
            use_spmd_partitioning: true,
        };
        let mut bopts_bytes = build_opts.encode();
        bopts_bytes.extend_from_slice(&extra.clone().into_bytes());
        assert_eq!(
            ExecutableBuildOptionsProto::decode(&bopts_bytes).ok(),
            Some(build_opts.clone())
        );

        let empty_copts = CompileOptionsProto::default();
        assert_eq!(
            CompileOptionsProto::decode(&empty_copts.encode()).ok(),
            Some(empty_copts)
        );

        let compile_opts = CompileOptionsProto {
            argument_layouts: vec![shape.clone()],
            parameter_is_tupled_arguments: true,
            executable_build_options: Some(build_opts),
            compile_portable_executable: true,
            profile_version: 7,
        };
        let mut copts_bytes = compile_opts.encode();
        copts_bytes.extend_from_slice(&extra.clone().into_bytes());
        assert_eq!(
            CompileOptionsProto::decode(&copts_bytes).ok(),
            Some(compile_opts)
        );

        let topo_proto = TpuTopologyArgsProto {
            version: 5,
            num_hosts: 1,
            chips_per_host: 4,
            mesh_bounds: vec![2, 2, 1],
            platform_version: String::from("0.0.50"),
            fingerprint: String::from("abc123"),
        };
        let mut topo_bytes = topo_proto.encode();
        topo_bytes.extend_from_slice(&extra.clone().into_bytes());
        assert_eq!(
            TpuTopologyArgsProto::decode(&topo_bytes).ok(),
            Some(topo_proto)
        );

        let exec_proto = TpuSerializedExecutableProto {
            name: String::from("exec_0"),
            fingerprint: String::from("deadbeef"),
            num_replicas: 1,
            num_partitions: 1,
            instruction_opcodes: vec![1, 2],
            output_shapes: vec![shape],
            compile_options_bytes: vec![1, 2, 3],
            hlo_module_bytes: vec![4, 5, 6],
        };
        let mut exec_bytes = exec_proto.encode();
        exec_bytes.extend_from_slice(&extra.clone().into_bytes());
        assert_eq!(
            TpuSerializedExecutableProto::decode(&exec_bytes).ok(),
            Some(exec_proto)
        );

        let stat = XStatProto {
            metadata_id: 1,
            int64_value: 100,
            str_value: String::from("ok"),
        };
        let mut stat_bytes = stat.encode();
        stat_bytes.extend_from_slice(&extra.clone().into_bytes());
        assert_eq!(XStatProto::decode(&stat_bytes).ok(), Some(stat.clone()));

        let ev = XEventProto {
            metadata_id: 2,
            offset_ps: 1000,
            duration_ps: 5000,
            stats: vec![stat.clone()],
        };
        let mut ev_bytes = ev.encode();
        ev_bytes.extend_from_slice(&extra.clone().into_bytes());
        assert_eq!(XEventProto::decode(&ev_bytes).ok(), Some(ev.clone()));

        let line = XLineProto {
            id: 1,
            name: String::from("TensorCore 0"),
            timestamp_ns: 123_456,
            events: vec![ev],
        };
        let mut line_bytes = line.encode();
        line_bytes.extend_from_slice(&extra.clone().into_bytes());
        assert_eq!(XLineProto::decode(&line_bytes).ok(), Some(line.clone()));

        let plane = XPlaneProto {
            id: 1,
            name: String::from("/device:TPU:0"),
            lines: vec![line],
            stats: vec![stat],
        };
        let mut plane_bytes = plane.encode();
        plane_bytes.extend_from_slice(&extra.clone().into_bytes());
        assert_eq!(XPlaneProto::decode(&plane_bytes).ok(), Some(plane.clone()));

        let xspace = XSpaceProto {
            planes: vec![plane],
            hostnames: vec![String::from("localhost")],
        };
        let mut xspace_bytes = xspace.encode();
        xspace_bytes.extend_from_slice(&extra.clone().into_bytes());
        assert_eq!(XSpaceProto::decode(&xspace_bytes).ok(), Some(xspace));

        let tbl = EmbeddingTableDescriptorProto {
            name: String::from("user_table"),
            vocabulary_size: 1024,
            dimension: 64,
            num_features: 2,
        };
        let mut tbl_bytes = tbl.encode();
        tbl_bytes.extend_from_slice(&extra.clone().into_bytes());
        assert_eq!(
            EmbeddingTableDescriptorProto::decode(&tbl_bytes).ok(),
            Some(tbl.clone())
        );

        let emb_cfg = TpuEmbeddingConfigurationProto {
            table_descriptor: vec![tbl],
            mode: 2,
            batch_size_per_tensor_core: 32,
            num_hosts: 1,
            num_tensor_cores: 4,
        };
        let mut emb_bytes = emb_cfg.encode();
        emb_bytes.extend_from_slice(&extra.clone().into_bytes());
        assert_eq!(
            TpuEmbeddingConfigurationProto::decode(&emb_bytes).ok(),
            Some(emb_cfg)
        );

        let comp_dev = ComputationDeviceProto {
            replica_device_ids: vec![0, 1],
        };
        let mut cd_bytes = comp_dev.encode();
        cd_bytes.extend_from_slice(&extra.clone().into_bytes());
        assert_eq!(
            ComputationDeviceProto::decode(&cd_bytes).ok(),
            Some(comp_dev.clone())
        );

        let dev_assign = DeviceAssignmentProto {
            replica_count: 2,
            computation_count: 1,
            computation_devices: vec![comp_dev],
        };
        let mut da_bytes = dev_assign.encode();
        da_bytes.extend_from_slice(&extra.clone().into_bytes());
        assert_eq!(
            DeviceAssignmentProto::decode(&da_bytes).ok(),
            Some(dev_assign)
        );

        let comp_env = TpuCompilationEnvironmentProto {
            xla_tpu_enable_megacore_fusion: true,
            xla_tpu_enable_async_collective_fusion: true,
            xla_tpu_scoped_vmem_limit_kib: 16384,
            custom_flags: String::from("--xla_tpu_flag=true"),
        };
        let mut ce_bytes = comp_env.encode();
        ce_bytes.extend_from_slice(&extra.clone().into_bytes());
        assert_eq!(
            TpuCompilationEnvironmentProto::decode(&ce_bytes).ok(),
            Some(comp_env)
        );

        let stat_meta = XStatMetadataProto {
            id: 1,
            name: String::from("flops"),
            description: String::from("FLOP count"),
        };
        let mut sm_bytes = stat_meta.encode();
        sm_bytes.extend_from_slice(&extra.clone().into_bytes());
        assert_eq!(XStatMetadataProto::decode(&sm_bytes).ok(), Some(stat_meta));

        let ev_meta = XEventMetadataProto {
            id: 2,
            name: String::from("execute"),
            display_name: String::from("TPU Execute"),
        };
        let mut em_bytes = ev_meta.encode();
        em_bytes.extend_from_slice(&extra.clone().into_bytes());
        assert_eq!(XEventMetadataProto::decode(&em_bytes).ok(), Some(ev_meta));

        let empty_hmc = HloModuleProtoWithConfig::default();
        assert_eq!(
            HloModuleProtoWithConfig::decode(&empty_hmc.encode()).ok(),
            Some(empty_hmc)
        );
        let hmc = HloModuleProtoWithConfig {
            hlo_module: Some(HloModuleProto::decode(&mod_bytes).unwrap_or_default()),
            config: Some(ExecutableBuildOptionsProto::decode(&bopts_bytes).unwrap_or_default()),
        };
        let mut hmc_bytes = hmc.encode();
        hmc_bytes.extend_from_slice(&extra.clone().into_bytes());
        assert_eq!(HloModuleProtoWithConfig::decode(&hmc_bytes).ok(), Some(hmc));

        let r_abi = RuntimeAbiVersionProto {
            platform_id: 0x7470_7500,
            version_string: String::from("0.0.50"),
            pjrt_api_minor: 116,
        };
        let mut r_abi_bytes = r_abi.encode();
        r_abi_bytes.extend_from_slice(&extra.clone().into_bytes());
        assert_eq!(
            RuntimeAbiVersionProto::decode(&r_abi_bytes).ok(),
            Some(r_abi)
        );

        let e_abi = ExecutableAbiVersionProto {
            platform_id: 0x7470_7500,
            compiler_version: String::from("0.0.50"),
            target_generation: 5,
        };
        let mut e_abi_bytes = e_abi.encode();
        e_abi_bytes.extend_from_slice(&extra.clone().into_bytes());
        assert_eq!(
            ExecutableAbiVersionProto::decode(&e_abi_bytes).ok(),
            Some(e_abi)
        );

        let part_cfg = TpuEmbeddingPartitionedConfigProto {
            num_partitions: 2,
            table_names: vec![String::from("user_table")],
            raw_config_bytes: vec![1, 2, 3],
        };
        let mut part_cfg_bytes = part_cfg.encode();
        part_cfg_bytes.extend_from_slice(&extra.clone().into_bytes());
        assert_eq!(
            TpuEmbeddingPartitionedConfigProto::decode(&part_cfg_bytes).ok(),
            Some(part_cfg)
        );

        let mem_cfg = TpuEmbeddingMemoryConfigProto {
            hbm_buffers_bytes: 4096,
            num_segments: 1,
            segment_blobs: vec![vec![10, 20]],
        };
        let mut mem_cfg_bytes = mem_cfg.encode();
        mem_cfg_bytes.extend_from_slice(&extra.clone().into_bytes());
        assert_eq!(
            TpuEmbeddingMemoryConfigProto::decode(&mem_cfg_bytes).ok(),
            Some(mem_cfg)
        );

        let host_net_cfg = TpuEmbeddingHostNetworkConfigProto {
            host_id: 1,
            endpoint_address: String::from("127.0.0.1:8471"),
            partition_payload: vec![5, 6],
        };
        let mut host_net_bytes = host_net_cfg.encode();
        host_net_bytes.extend_from_slice(&extra.clone().into_bytes());
        assert_eq!(
            TpuEmbeddingHostNetworkConfigProto::decode(&host_net_bytes).ok(),
            Some(host_net_cfg)
        );

        let ms_cfg = MegascaleMultiSliceConfigProto {
            num_slices: 2,
            slice_id: 1,
            num_devices_per_slice: 4,
            coordinator_address: String::from("127.0.0.1:8080"),
        };
        let mut ms_cfg_bytes = ms_cfg.encode();
        ms_cfg_bytes.extend_from_slice(&extra.clone().into_bytes());
        assert_eq!(
            MegascaleMultiSliceConfigProto::decode(&ms_cfg_bytes).ok(),
            Some(ms_cfg)
        );

        let dcn_topo = MegascaleDcnTopologyProto {
            num_slices: 2,
            num_hosts_per_slice: 1,
            transport_type: 1,
        };
        let mut dcn_bytes = dcn_topo.encode();
        dcn_bytes.extend_from_slice(&extra.clone().into_bytes());
        assert_eq!(
            MegascaleDcnTopologyProto::decode(&dcn_bytes).ok(),
            Some(dcn_topo)
        );

        let ms_metrics = MegascaleRuntimeMetricsProto {
            completed_barriers: 4,
            collective_ops_count: 8,
            active_contexts: 1,
        };
        let mut ms_metrics_bytes = ms_metrics.encode();
        ms_metrics_bytes.extend_from_slice(&extra.into_bytes());
        assert_eq!(
            MegascaleRuntimeMetricsProto::decode(&ms_metrics_bytes).ok(),
            Some(ms_metrics)
        );

        for i in 0..part_cfg_bytes.len() {
            let _ =
                TpuEmbeddingPartitionedConfigProto::decode(part_cfg_bytes.get(..i).unwrap_or(&[]));
        }
        for i in 0..mem_cfg_bytes.len() {
            let _ = TpuEmbeddingMemoryConfigProto::decode(mem_cfg_bytes.get(..i).unwrap_or(&[]));
        }
        for i in 0..host_net_bytes.len() {
            let _ =
                TpuEmbeddingHostNetworkConfigProto::decode(host_net_bytes.get(..i).unwrap_or(&[]));
        }
        for i in 0..ms_cfg_bytes.len() {
            let _ = MegascaleMultiSliceConfigProto::decode(ms_cfg_bytes.get(..i).unwrap_or(&[]));
        }
        for i in 0..dcn_bytes.len() {
            let _ = MegascaleDcnTopologyProto::decode(dcn_bytes.get(..i).unwrap_or(&[]));
        }
        for i in 0..ms_metrics_bytes.len() {
            let _ = MegascaleRuntimeMetricsProto::decode(ms_metrics_bytes.get(..i).unwrap_or(&[]));
        }

        for i in 0..hmc_bytes.len() {
            let _ = HloModuleProtoWithConfig::decode(hmc_bytes.get(..i).unwrap_or(&[]));
        }
        for i in 0..r_abi_bytes.len() {
            let _ = RuntimeAbiVersionProto::decode(r_abi_bytes.get(..i).unwrap_or(&[]));
        }
        for i in 0..e_abi_bytes.len() {
            let _ = ExecutableAbiVersionProto::decode(e_abi_bytes.get(..i).unwrap_or(&[]));
        }

        for i in 0..cd_bytes.len() {
            let _ = ComputationDeviceProto::decode(cd_bytes.get(..i).unwrap_or(&[]));
        }
        for i in 0..da_bytes.len() {
            let _ = DeviceAssignmentProto::decode(da_bytes.get(..i).unwrap_or(&[]));
        }
        for i in 0..ce_bytes.len() {
            let _ = TpuCompilationEnvironmentProto::decode(ce_bytes.get(..i).unwrap_or(&[]));
        }
        for i in 0..sm_bytes.len() {
            let _ = XStatMetadataProto::decode(sm_bytes.get(..i).unwrap_or(&[]));
        }
        for i in 0..em_bytes.len() {
            let _ = XEventMetadataProto::decode(em_bytes.get(..i).unwrap_or(&[]));
        }

        // Prefix truncation sweep across every message decoder to cover all `?` error paths
        for i in 0..tile_bytes.len() {
            let _ = TileProto::decode(tile_bytes.get(..i).unwrap_or(&[]));
        }
        for i in 0..layout_bytes.len() {
            let _ = LayoutProto::decode(layout_bytes.get(..i).unwrap_or(&[]));
        }
        for i in 0..shape_bytes.len() {
            let _ = ShapeProto::decode(shape_bytes.get(..i).unwrap_or(&[]));
        }
        for i in 0..sharding_bytes.len() {
            let _ = OpShardingProto::decode(sharding_bytes.get(..i).unwrap_or(&[]));
        }
        for i in 0..inst_bytes.len() {
            let _ = HloInstructionProto::decode(inst_bytes.get(..i).unwrap_or(&[]));
        }
        for i in 0..comp_bytes.len() {
            let _ = HloComputationProto::decode(comp_bytes.get(..i).unwrap_or(&[]));
        }
        for i in 0..mod_bytes.len() {
            let _ = HloModuleProto::decode(mod_bytes.get(..i).unwrap_or(&[]));
        }
        for i in 0..bopts_bytes.len() {
            let _ = ExecutableBuildOptionsProto::decode(bopts_bytes.get(..i).unwrap_or(&[]));
        }
        for i in 0..copts_bytes.len() {
            let _ = CompileOptionsProto::decode(copts_bytes.get(..i).unwrap_or(&[]));
        }
        for i in 0..topo_bytes.len() {
            let _ = TpuTopologyArgsProto::decode(topo_bytes.get(..i).unwrap_or(&[]));
        }
        for i in 0..exec_bytes.len() {
            let _ = TpuSerializedExecutableProto::decode(exec_bytes.get(..i).unwrap_or(&[]));
        }
        for i in 0..stat_bytes.len() {
            let _ = XStatProto::decode(stat_bytes.get(..i).unwrap_or(&[]));
        }
        for i in 0..ev_bytes.len() {
            let _ = XEventProto::decode(ev_bytes.get(..i).unwrap_or(&[]));
        }
        for i in 0..line_bytes.len() {
            let _ = XLineProto::decode(line_bytes.get(..i).unwrap_or(&[]));
        }
        for i in 0..plane_bytes.len() {
            let _ = XPlaneProto::decode(plane_bytes.get(..i).unwrap_or(&[]));
        }
        for i in 0..xspace_bytes.len() {
            let _ = XSpaceProto::decode(xspace_bytes.get(..i).unwrap_or(&[]));
        }
        for i in 0..tbl_bytes.len() {
            let _ = EmbeddingTableDescriptorProto::decode(tbl_bytes.get(..i).unwrap_or(&[]));
        }
        for i in 0..emb_bytes.len() {
            let _ = TpuEmbeddingConfigurationProto::decode(emb_bytes.get(..i).unwrap_or(&[]));
        }

        // Corrupt nested sub-message payloads (length=1, byte=0x80) to trigger nested `decode(sub)?` errors
        for field_no in 1..=8 {
            let mut w_corrupt = ProtoWriter::new();
            w_corrupt.write_bytes(field_no, &[0x80]);
            let corrupt = w_corrupt.into_bytes();
            let _ = LayoutProto::decode(&corrupt);
            let _ = ShapeProto::decode(&corrupt);
            let _ = HloInstructionProto::decode(&corrupt);
            let _ = HloComputationProto::decode(&corrupt);
            let _ = HloModuleProto::decode(&corrupt);
            let _ = HloModuleProtoWithConfig::decode(&corrupt);
            let _ = ExecutableBuildOptionsProto::decode(&corrupt);
            let _ = CompileOptionsProto::decode(&corrupt);
            let _ = TpuSerializedExecutableProto::decode(&corrupt);
            let _ = XEventProto::decode(&corrupt);
            let _ = XLineProto::decode(&corrupt);
            let _ = XPlaneProto::decode(&corrupt);
            let _ = XSpaceProto::decode(&corrupt);
            let _ = TpuEmbeddingConfigurationProto::decode(&corrupt);
            let _ = DeviceAssignmentProto::decode(&corrupt);

            let bad_varint = [u8::try_from(field_no << 3).unwrap_or(8), 0x80];
            let _ = TpuEmbeddingPartitionedConfigProto::decode(&bad_varint);
            let _ = TpuEmbeddingMemoryConfigProto::decode(&bad_varint);
            let _ = TpuEmbeddingHostNetworkConfigProto::decode(&bad_varint);
            let _ = MegascaleMultiSliceConfigProto::decode(&bad_varint);
            let _ = MegascaleDcnTopologyProto::decode(&bad_varint);
            let _ = MegascaleRuntimeMetricsProto::decode(&bad_varint);
        }
    }
}

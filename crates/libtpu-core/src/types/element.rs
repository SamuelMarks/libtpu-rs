//! Tensor element data types (`PJRT_Buffer_Type`) and low-precision scalar codecs.
//!
//! Defines [`ElementType`] (`Pred`, `S2`..`S64`, `U2`..`U64`, `F16`, `Bf16`, `F32`, `F64`,
//! `F8E5M2`, `F8E4M3FN`, `F4E2M1FN`, `C64`, `C128`, `Token`, `Tuple`) along with bit/byte
//! sizing and hardware-accurate conversion, packing, and unpacking routines.

use crate::error::{LibTpuError, LibTpuResult};
use derive_more::Display;
use num_enum::{IntoPrimitive, TryFromPrimitive};

/// IEEE 754 16-bit half-precision floating-point scalar type (`f16`) re-exported from `half`.
pub type F16Scalar = half::f16;

/// 16-bit Brain Floating-Point scalar type (`bf16`) re-exported from `half`.
pub type Bf16Scalar = half::bf16;

/// Tensor element data type matching `PJRT_Buffer_Type`.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Display, TryFromPrimitive, IntoPrimitive)]
pub enum ElementType {
    /// Invalid or unspecified element type (`0`).
    #[display("INVALID")]
    Invalid = 0,
    /// Boolean predicate (`1`).
    #[display("PRED")]
    Pred = 1,
    /// 8-bit signed integer (`2`).
    #[display("S8")]
    S8 = 2,
    /// 16-bit signed integer (`3`).
    #[display("S16")]
    S16 = 3,
    /// 32-bit signed integer (`4`).
    #[display("S32")]
    S32 = 4,
    /// 64-bit signed integer (`5`).
    #[display("S64")]
    S64 = 5,
    /// 8-bit unsigned integer (`6`).
    #[display("U8")]
    U8 = 6,
    /// 16-bit unsigned integer (`7`).
    #[display("U16")]
    U16 = 7,
    /// 32-bit unsigned integer (`8`).
    #[display("U32")]
    U32 = 8,
    /// 64-bit unsigned integer (`9`).
    #[display("U64")]
    U64 = 9,
    /// IEEE 16-bit floating-point (`10`).
    #[display("F16")]
    F16 = 10,
    /// IEEE 32-bit floating-point (`11`).
    #[display("F32")]
    F32 = 11,
    /// IEEE 64-bit floating-point (`12`).
    #[display("F64")]
    F64 = 12,
    /// Brain floating-point 16-bit (`13`).
    #[display("BF16")]
    Bf16 = 13,
    /// 64-bit complex floating-point (`14`).
    #[display("C64")]
    C64 = 14,
    /// 128-bit complex floating-point (`15`).
    #[display("C128")]
    C128 = 15,
    /// 8-bit floating-point E5M2 (`16`).
    #[display("F8E5M2")]
    F8E5M2 = 16,
    /// 8-bit floating-point E4M3FN (`17`).
    #[display("F8E4M3FN")]
    F8E4M3Fn = 17,
    /// 8-bit floating-point E4M3B11FNUZ (`18`).
    #[display("F8E4M3B11FNUZ")]
    F8E4M3B11Fnuz = 18,
    /// 8-bit floating-point E5M2FNUZ (`19`).
    #[display("F8E5M2FNUZ")]
    F8E5M2Fnuz = 19,
    /// 8-bit floating-point E4M3FNUZ (`20`).
    #[display("F8E4M3FNUZ")]
    F8E4M3Fnuz = 20,
    /// 4-bit signed integer (`21`).
    #[display("S4")]
    S4 = 21,
    /// 4-bit unsigned integer (`22`).
    #[display("U4")]
    U4 = 22,
    /// XLA ordering token (`23`).
    #[display("TOKEN")]
    Token = 23,
    /// 2-bit signed integer (`24`).
    #[display("S2")]
    S2 = 24,
    /// 2-bit unsigned integer (`25`).
    #[display("U2")]
    U2 = 25,
    /// 4-bit floating-point E2M1FN (`26`).
    #[display("F4E2M1FN")]
    F4E2M1Fn = 26,
    /// 8-bit block scaling exponent E8M0FNU (`27`).
    #[display("F8E8M0FNU")]
    F8E8M0Fnu = 27,
    /// XLA tuple composite type (`28`).
    #[display("TUPLE")]
    Tuple = 28,
}

impl ElementType {
    /// Converts a raw `PJRT_Buffer_Type` integer into an [`ElementType`].
    ///
    /// # Arguments
    ///
    /// * `raw` - Raw integer code from C FFI.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::UnsupportedElementType`] if `raw` is outside `0..=28`.
    pub const fn from_raw(raw: i32) -> LibTpuResult<Self> {
        match raw {
            0 => Ok(Self::Invalid),
            1 => Ok(Self::Pred),
            2 => Ok(Self::S8),
            3 => Ok(Self::S16),
            4 => Ok(Self::S32),
            5 => Ok(Self::S64),
            6 => Ok(Self::U8),
            7 => Ok(Self::U16),
            8 => Ok(Self::U32),
            9 => Ok(Self::U64),
            10 => Ok(Self::F16),
            11 => Ok(Self::F32),
            12 => Ok(Self::F64),
            13 => Ok(Self::Bf16),
            14 => Ok(Self::C64),
            15 => Ok(Self::C128),
            16 => Ok(Self::F8E5M2),
            17 => Ok(Self::F8E4M3Fn),
            18 => Ok(Self::F8E4M3B11Fnuz),
            19 => Ok(Self::F8E5M2Fnuz),
            20 => Ok(Self::F8E4M3Fnuz),
            21 => Ok(Self::S4),
            22 => Ok(Self::U4),
            23 => Ok(Self::Token),
            24 => Ok(Self::S2),
            25 => Ok(Self::U2),
            26 => Ok(Self::F4E2M1Fn),
            27 => Ok(Self::F8E8M0Fnu),
            28 => Ok(Self::Tuple),
            _ => Err(LibTpuError::UnsupportedElementType { raw_type: raw }),
        }
    }

    /// Returns the raw integer code matching `PJRT_Buffer_Type`.
    #[must_use]
    pub const fn as_raw(self) -> i32 {
        self as i32
    }

    /// Returns the size in bits of a single element of this type.
    #[must_use]
    pub const fn bit_size(self) -> usize {
        match self {
            Self::Invalid | Self::Token | Self::Tuple => 0,
            Self::S2 | Self::U2 => 2,
            Self::S4 | Self::U4 | Self::F4E2M1Fn => 4,
            Self::Pred
            | Self::S8
            | Self::U8
            | Self::F8E5M2
            | Self::F8E4M3Fn
            | Self::F8E4M3B11Fnuz
            | Self::F8E5M2Fnuz
            | Self::F8E4M3Fnuz
            | Self::F8E8M0Fnu => 8,
            Self::S16 | Self::U16 | Self::F16 | Self::Bf16 => 16,
            Self::S32 | Self::U32 | Self::F32 => 32,
            Self::S64 | Self::U64 | Self::F64 | Self::C64 => 64,
            Self::C128 => 128,
        }
    }

    /// Returns the size in bytes of a single element (rounded up to at least 1 byte for non-empty types).
    #[must_use]
    pub const fn byte_size(self) -> usize {
        self.bit_size().div_ceil(8)
    }

    /// Converts an IEEE-754 `f32` to 16-bit `bf16` bits using `half::bf16` round-to-nearest-even.
    ///
    /// # Arguments
    ///
    /// * `val` - 32-bit float value.
    #[must_use]
    pub fn f32_to_bf16_bits(val: f32) -> u16 {
        half::bf16::from_f32(val).to_bits()
    }

    /// Converts 16-bit `bf16` bits to an IEEE-754 `f32` using `half::bf16`.
    ///
    /// # Arguments
    ///
    /// * `bits` - 16-bit `bf16` bit pattern.
    #[must_use]
    pub fn bf16_bits_to_f32(bits: u16) -> f32 {
        half::bf16::from_bits(bits).to_f32()
    }

    /// Converts an IEEE-754 `f32` to 16-bit `f16` bits using `half::f16` round-to-nearest-even.
    ///
    /// # Arguments
    ///
    /// * `val` - 32-bit float value.
    #[must_use]
    pub fn f32_to_f16_bits(val: f32) -> u16 {
        half::f16::from_f32(val).to_bits()
    }

    /// Converts 16-bit `f16` bits to an IEEE-754 `f32` using `half::f16`.
    ///
    /// # Arguments
    ///
    /// * `bits` - 16-bit `f16` bit pattern.
    #[must_use]
    pub fn f16_bits_to_f32(bits: u16) -> f32 {
        half::f16::from_bits(bits).to_f32()
    }

    /// Converts an IEEE-754 `f32` to 8-bit `F8E4M3FN` bits using `float8::F8E4M3`.
    ///
    /// # Arguments
    ///
    /// * `val` - 32-bit float value.
    #[must_use]
    pub const fn f32_to_f8e4m3_bits(val: f32) -> u8 {
        float8::F8E4M3::from_f32(val).to_bits()
    }

    /// Converts 8-bit `F8E4M3FN` bits to an IEEE-754 `f32` using `float8::F8E4M3`.
    ///
    /// # Arguments
    ///
    /// * `bits` - 8-bit `F8E4M3FN` bit pattern.
    #[must_use]
    pub const fn f8e4m3_bits_to_f32(bits: u8) -> f32 {
        float8::F8E4M3::from_bits(bits).to_f32()
    }

    /// Converts an IEEE-754 `f32` to 8-bit `F8E5M2` bits using `float8::F8E5M2`.
    ///
    /// # Arguments
    ///
    /// * `val` - 32-bit float value.
    #[must_use]
    pub const fn f32_to_f8e5m2_bits(val: f32) -> u8 {
        float8::F8E5M2::from_f32(val).to_bits()
    }

    /// Converts 8-bit `F8E5M2` bits to an IEEE-754 `f32` using `float8::F8E5M2`.
    ///
    /// # Arguments
    ///
    /// * `bits` - 8-bit `F8E5M2` bit pattern.
    #[must_use]
    pub const fn f8e5m2_bits_to_f32(bits: u8) -> f32 {
        float8::F8E5M2::from_bits(bits).to_f32()
    }

    /// Packs a slice of `num_complex::Complex64` values into little-endian `C128` (`[f64; 2]`) bytes.
    ///
    /// # Arguments
    ///
    /// * `values` - Slice of complex double-precision numbers.
    #[must_use]
    pub fn pack_complex64_slice(values: &[num_complex::Complex64]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len().saturating_mul(16));
        for c in values {
            out.extend_from_slice(&c.re.to_le_bytes());
            out.extend_from_slice(&c.im.to_le_bytes());
        }
        out
    }

    /// Unpacks little-endian `C128` (`[f64; 2]`) bytes into a vector of `num_complex::Complex64`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Raw little-endian `C128` byte payload.
    #[must_use]
    pub fn unpack_to_complex64_vec(bytes: &[u8]) -> Vec<num_complex::Complex64> {
        let scalars = Self::C128.unpack_to_f64_vec(bytes);
        scalars
            .chunks_exact(2)
            .map(|pair| {
                num_complex::Complex64::new(
                    pair.first().copied().unwrap_or(0.0),
                    pair.get(1).copied().unwrap_or(0.0),
                )
            })
            .collect()
    }

    /// Unpacks a raw little-endian byte buffer of this [`ElementType`] into `f64` scalar values
    /// (for `C64` and `C128`, unpacks interleaved `[re0, im0, re1, im1, ...]` scalars).
    ///
    /// # Arguments
    ///
    /// * `bytes` - Raw little-endian tensor payload.
    #[must_use]
    #[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
    pub fn unpack_to_f64_vec(self, bytes: &[u8]) -> Vec<f64> {
        let aligned_len = |elem_size: usize| (bytes.len() / elem_size) * elem_size;
        match self {
            Self::Pred | Self::U8 => bytes
                .iter()
                .map(|&b| {
                    if self == Self::Pred {
                        if b != 0 { 1.0 } else { 0.0 }
                    } else {
                        f64::from(b)
                    }
                })
                .collect(),
            Self::S8 | Self::S4 | Self::S2 => bytes
                .iter()
                .map(|&b| {
                    let raw = i8::from_le_bytes([b]);
                    let clamped = match self {
                        Self::S4 => raw.clamp(-8, 7),
                        Self::S2 => raw.clamp(-2, 1),
                        _ => raw,
                    };
                    f64::from(clamped)
                })
                .collect(),
            Self::U4 | Self::U2 => bytes
                .iter()
                .map(|&b| {
                    let masked = if self == Self::U4 { b & 0x0F } else { b & 0x03 };
                    f64::from(masked)
                })
                .collect(),
            Self::F8E4M3Fn | Self::F8E4M3B11Fnuz | Self::F8E4M3Fnuz => bytes
                .iter()
                .map(|&b| f64::from(Self::f8e4m3_bits_to_f32(b)))
                .collect(),
            Self::F8E5M2 | Self::F8E5M2Fnuz => bytes
                .iter()
                .map(|&b| f64::from(Self::f8e5m2_bits_to_f32(b)))
                .collect(),
            Self::S16 => bytemuck::pod_collect_to_vec::<u8, i16>(
                bytes.get(..aligned_len(2)).unwrap_or_default(),
            )
            .into_iter()
            .map(|v| f64::from(i16::from_le(v)))
            .collect(),
            Self::U16 => bytemuck::pod_collect_to_vec::<u8, u16>(
                bytes.get(..aligned_len(2)).unwrap_or_default(),
            )
            .into_iter()
            .map(|v| f64::from(u16::from_le(v)))
            .collect(),
            Self::F16 => bytemuck::pod_collect_to_vec::<u8, u16>(
                bytes.get(..aligned_len(2)).unwrap_or_default(),
            )
            .into_iter()
            .map(|v| f64::from(F16Scalar::from_bits(u16::from_le(v)).to_f32()))
            .collect(),
            Self::Bf16 => bytemuck::pod_collect_to_vec::<u8, u16>(
                bytes.get(..aligned_len(2)).unwrap_or_default(),
            )
            .into_iter()
            .map(|v| f64::from(Bf16Scalar::from_bits(u16::from_le(v)).to_f32()))
            .collect(),
            Self::S32 => bytemuck::pod_collect_to_vec::<u8, i32>(
                bytes.get(..aligned_len(4)).unwrap_or_default(),
            )
            .into_iter()
            .map(|v| f64::from(i32::from_le(v)))
            .collect(),
            Self::U32 => bytemuck::pod_collect_to_vec::<u8, u32>(
                bytes.get(..aligned_len(4)).unwrap_or_default(),
            )
            .into_iter()
            .map(|v| f64::from(u32::from_le(v)))
            .collect(),
            Self::F32 | Self::C64 => bytemuck::pod_collect_to_vec::<u8, u32>(
                bytes.get(..aligned_len(4)).unwrap_or_default(),
            )
            .into_iter()
            .map(|bits| f64::from(f32::from_bits(u32::from_le(bits))))
            .collect(),
            Self::S64 => bytemuck::pod_collect_to_vec::<u8, i64>(
                bytes.get(..aligned_len(8)).unwrap_or_default(),
            )
            .into_iter()
            .map(|v| i64::from_le(v) as f64)
            .collect(),
            Self::U64 => bytemuck::pod_collect_to_vec::<u8, u64>(
                bytes.get(..aligned_len(8)).unwrap_or_default(),
            )
            .into_iter()
            .map(|v| u64::from_le(v) as f64)
            .collect(),
            Self::F64 | Self::C128 => bytemuck::pod_collect_to_vec::<u8, u64>(
                bytes.get(..aligned_len(8)).unwrap_or_default(),
            )
            .into_iter()
            .map(|bits| f64::from_bits(u64::from_le(bits)))
            .collect(),
            _ => bytes.iter().map(|&b| f64::from(b)).collect(),
        }
    }

    /// Packs a slice of `f64` scalar values into a little-endian byte vector of this [`ElementType`].
    ///
    /// # Arguments
    ///
    /// * `values` - Scalar values to pack.
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    pub fn pack_from_f64_slice(self, values: &[f64]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len().saturating_mul(self.byte_size().max(1)));
        match self {
            Self::Pred => {
                for &v in values {
                    out.push(u8::from(v != 0.0));
                }
            }
            Self::S8 | Self::S4 | Self::S2 => {
                for &v in values {
                    let raw = v as i8;
                    let clamped = match self {
                        Self::S4 => raw.clamp(-8, 7),
                        Self::S2 => raw.clamp(-2, 1),
                        _ => raw,
                    };
                    out.extend_from_slice(&clamped.to_le_bytes());
                }
            }
            Self::U4 | Self::U2 => {
                for &v in values {
                    let raw = v as u8;
                    let masked = if self == Self::U4 {
                        raw & 0x0F
                    } else {
                        raw & 0x03
                    };
                    out.push(masked);
                }
            }
            Self::F8E4M3Fn | Self::F8E4M3B11Fnuz | Self::F8E4M3Fnuz => {
                for &v in values {
                    out.push(Self::f32_to_f8e4m3_bits(v as f32));
                }
            }
            Self::F8E5M2 | Self::F8E5M2Fnuz => {
                for &v in values {
                    out.push(Self::f32_to_f8e5m2_bits(v as f32));
                }
            }
            Self::S16 => {
                for &v in values {
                    out.extend_from_slice(&(v as i16).to_le_bytes());
                }
            }
            Self::U16 => {
                for &v in values {
                    out.extend_from_slice(&(v as u16).to_le_bytes());
                }
            }
            Self::F16 => {
                for &v in values {
                    out.extend_from_slice(&Self::f32_to_f16_bits(v as f32).to_le_bytes());
                }
            }
            Self::Bf16 => {
                for &v in values {
                    out.extend_from_slice(&Self::f32_to_bf16_bits(v as f32).to_le_bytes());
                }
            }
            Self::S32 => {
                for &v in values {
                    out.extend_from_slice(&(v as i32).to_le_bytes());
                }
            }
            Self::U32 => {
                for &v in values {
                    out.extend_from_slice(&(v as u32).to_le_bytes());
                }
            }
            Self::F32 | Self::C64 => {
                for &v in values {
                    out.extend_from_slice(&(v as f32).to_le_bytes());
                }
            }
            Self::S64 => {
                for &v in values {
                    out.extend_from_slice(&(v as i64).to_le_bytes());
                }
            }
            Self::U64 => {
                for &v in values {
                    out.extend_from_slice(&(v as u64).to_le_bytes());
                }
            }
            Self::F64 | Self::C128 => {
                for &v in values {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
            _ => {
                for &v in values {
                    out.push(v as u8);
                }
            }
        }
        out
    }
}

#[cfg(test)]
#[allow(
    for_loops_over_fallibles,
    clippy::cognitive_complexity,
    clippy::too_many_lines,
    clippy::float_cmp
)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_element_type_codecs() {
        for raw in 0..=28 {
            let et = ElementType::from_raw(raw);
            assert!(et.is_ok());
            for t in et {
                assert_eq!(t.as_raw(), raw);
                let _ = t.bit_size();
                let _ = t.byte_size();
                assert_ne!(t.to_string(), "");
                assert_ne!(format!("{t:?}"), "");
            }
        }
        assert!(ElementType::from_raw(99).is_err());
        assert_eq!(ElementType::try_from(11_i32).ok(), Some(ElementType::F32));
        assert_eq!(i32::from(ElementType::Bf16), 13);

        assert_eq!(ElementType::f16_bits_to_f32(0), 0.0);
        assert!(ElementType::f16_bits_to_f32(0x7c00).is_infinite());
        assert_eq!(ElementType::f32_to_f16_bits(0.0), 0);
        assert_eq!(ElementType::f32_to_f16_bits(100_000.0), 0x7c00);
        let f16_one = ElementType::f32_to_f16_bits(1.5);
        assert!((ElementType::f16_bits_to_f32(f16_one) - 1.5).abs() < 1e-3);
        let bf16_two = ElementType::f32_to_bf16_bits(2.5);
        assert!((ElementType::bf16_bits_to_f32(bf16_two) - 2.5).abs() < 1e-2);
        let f8e4m3_val = ElementType::f32_to_f8e4m3_bits(1.5);
        assert!((ElementType::f8e4m3_bits_to_f32(f8e4m3_val) - 1.5).abs() < 1e-1);
        let f8e5m2_val = ElementType::f32_to_f8e5m2_bits(2.0);
        assert!((ElementType::f8e5m2_bits_to_f32(f8e5m2_val) - 2.0).abs() < 1e-1);

        let c_vals = [
            num_complex::Complex64::new(1.5, -2.5),
            num_complex::Complex64::new(3.25, 4.75),
        ];
        let c_packed = ElementType::pack_complex64_slice(&c_vals);
        let c_unpacked = ElementType::unpack_to_complex64_vec(&c_packed);
        assert_eq!(c_unpacked, c_vals);

        for et in [
            ElementType::Pred,
            ElementType::S8,
            ElementType::U8,
            ElementType::S16,
            ElementType::U16,
            ElementType::F16,
            ElementType::Bf16,
            ElementType::S32,
            ElementType::U32,
            ElementType::F32,
            ElementType::C64,
            ElementType::S64,
            ElementType::U64,
            ElementType::F64,
            ElementType::C128,
            ElementType::F8E4M3Fn,
            ElementType::F8E4M3B11Fnuz,
            ElementType::F8E4M3Fnuz,
            ElementType::F8E5M2,
            ElementType::F8E5M2Fnuz,
            ElementType::S4,
            ElementType::U4,
            ElementType::S2,
            ElementType::U2,
            ElementType::F4E2M1Fn,
        ] {
            let packed = et.pack_from_f64_slice(&[0.0, 2.0, 3.0, 4.0]);
            let unpacked = et.unpack_to_f64_vec(&packed);
            assert_eq!(unpacked.len(), 4);
        }

        let f16_nan = F16Scalar::from_f32(f32::NAN);
        assert!(ElementType::f16_bits_to_f32(f16_nan.to_bits()).is_nan());
        let bf16_inf = Bf16Scalar::from_f32(f32::INFINITY);
        assert!(ElementType::bf16_bits_to_f32(bf16_inf.to_bits()).is_infinite());
    }
}

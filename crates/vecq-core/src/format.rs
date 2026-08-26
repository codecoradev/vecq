//! Single-file persistence format.
//!
//! Layout (little-endian):
//! ```text
//! [magic "VECQ" u32][version u16][reserved u16]
//! [dim u32][seed u64][count u32]
//! [scales: count entries]
//! [codes: count * (padded/2) bytes]
//! ```
//!
//! Version 1: scales stored as f32 (4 bytes each).
//! Version 1.1 (stored as 257): scales stored as f16 (2 bytes each).
//!
//! Readers accept both; writers emit 1.1. The seed is stored in the header so
//! the random sign diagonal can be regenerated identically on any platform:
//! identical file -> identical query results, bit for bit.

use crate::store::VecqIndex;

const MAGIC: u32 = u32::from_le_bytes(*b"VECQ");
const V1: u16 = 1;
const V1_1: u16 = 257;

#[derive(Debug)]
pub enum Error {
    NotAStableFile,
    UnsupportedVersion(u16),
    Truncated,
    DimMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::NotAStableFile => write!(f, "not a vecq file"),
            Error::UnsupportedVersion(v) => write!(f, "unsupported version {v}"),
            Error::Truncated => write!(f, "file truncated"),
            Error::DimMismatch { expected, got } => {
                write!(f, "dim mismatch: expected {expected}, got {got}")
            }
        }
    }
}

impl std::error::Error for Error {}

/// IEEE 754 half-precision encode (round to nearest even), no dependencies.
fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7F_FFFF;
    if exp == 0xFF {
        // Inf / NaN
        return sign | 0x7C00 | if mant != 0 { 0x0200 } else { 0 };
    }
    let unbiased = exp - 127;
    if unbiased > 15 {
        return sign | 0x7C00; // overflow -> Inf
    }
    if unbiased >= -14 {
        // Normal half
        let half_exp = (unbiased + 15) as u32;
        let half_mant = mant >> 13;
        let mut out = sign | ((half_exp << 10) as u16) | half_mant as u16;
        // Round to nearest even on the dropped 13 bits.
        let round = mant & 0x1FFF;
        if round > 0x1000 || (round == 0x1000 && (half_mant & 1 == 1)) {
            out = out.wrapping_add(1);
        }
        out
    } else {
        // Subnormal half
        let m = mant | 0x80_0000; // implicit leading 1
        let shift = (-unbiased - 14 + 13) as u32;
        if shift >= 32 {
            return sign;
        }
        let half_mant = m >> shift;
        let mut out = sign | half_mant as u16;
        let round_bits = m & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        if round_bits > halfway || (round_bits == halfway && (half_mant & 1 == 1)) {
            out = out.wrapping_add(1);
        }
        out
    }
}

/// IEEE 754 half-precision decode.
fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x03FF) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // Subnormal: normalize
            let mut e = -1i32;
            let mut m = mant;
            while m & 0x0400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03FF;
            sign | (((113 + e) as u32) << 23) | (m << 13)
        }
    } else if exp == 0x1F {
        sign | 0x7F80_0000 | (mant << 13)
    } else {
        sign | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

impl VecqIndex {
    /// Serialize the index to bytes (format version 1.1, f16 scales).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(24 + self.codes.len() + self.scales.len() * 2);
        out.extend_from_slice(&MAGIC.to_le_bytes());
        out.extend_from_slice(&V1_1.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&(self.dim as u32).to_le_bytes());
        out.extend_from_slice(&self.seed.to_le_bytes());
        out.extend_from_slice(&(self.n as u32).to_le_bytes());
        for s in &self.scales {
            out.extend_from_slice(&f32_to_f16_bits(*s).to_le_bytes());
        }
        out.extend_from_slice(&self.codes);
        out
    }

    /// Parse an index from bytes produced by [`to_bytes`] (or a v1 file).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let rd_u32 = |b: &[u8]| u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        let rd_u16 = |b: &[u8]| u16::from_le_bytes([b[0], b[1]]);
        if bytes.len() < 24 {
            return Err(Error::Truncated);
        }
        if rd_u32(&bytes[0..4]) != MAGIC {
            return Err(Error::NotAStableFile);
        }
        let version = rd_u16(&bytes[4..6]);
        if version != V1 && version != V1_1 {
            return Err(Error::UnsupportedVersion(version));
        }
        let dim = rd_u32(&bytes[8..12]) as usize;
        let seed = u64::from_le_bytes(bytes[12..20].try_into().unwrap());
        let count = rd_u32(&bytes[20..24]) as usize;

        let padded = crate::rhdh::padded_dim(dim);
        let codes_bytes = padded / 2;
        let scale_bytes = if version == V1 { 4 } else { 2 };
        let expected = 24 + count * (scale_bytes + codes_bytes);
        if bytes.len() < expected {
            return Err(Error::Truncated);
        }

        let mut scales = Vec::with_capacity(count);
        let mut off = 24;
        for _ in 0..count {
            let s = if version == V1 {
                f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap())
            } else {
                f16_bits_to_f32(rd_u16(&bytes[off..off + 2]))
            };
            scales.push(s);
            off += scale_bytes;
        }
        let mut index = VecqIndex::new(dim, seed);
        index.codes = bytes[off..off + count * codes_bytes].to_vec();
        index.scales = scales;
        index.n = count;
        Ok(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(dim: usize, salt: u64) -> Vec<f32> {
        let mut x = salt | 1;
        let mut v = Vec::with_capacity(dim);
        for _ in 0..dim {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            v.push((x as f32 / u32::MAX as f32 - 0.5) * 2.0);
        }
        let norm: f32 = v.iter().map(|a| a * a).sum::<f32>().sqrt();
        v.into_iter().map(|a| a / norm).collect()
    }

    #[test]
    fn round_trip_is_identical() {
        let mut idx = VecqIndex::new(384, 99);
        for i in 0..50 {
            idx.add(&unit(384, i * 7 + 1));
        }
        let bytes = idx.to_bytes();
        let back = VecqIndex::from_bytes(&bytes).expect("parse");
        assert_eq!(back.len(), 50);
        assert_eq!(back.dim(), 384);
        assert_eq!(back.seed(), 99);
        // Same file -> identical search results (the cross-platform guarantee).
        let q = unit(384, 4242);
        let r1 = back.search(&q, 10);
        let bytes2 = back.to_bytes();
        assert_eq!(bytes, bytes2, "re-serialize must be byte-identical");
        let back2 = VecqIndex::from_bytes(&bytes2).unwrap();
        assert_eq!(r1, back2.search(&q, 10));
        // f16 scales perturb scores by <1e-3: top-10 overlap must stay >= 9/10
        // and the top-1 must match.
        let r0 = idx.search(&q, 10);
        assert_eq!(r0[0].0, r1[0].0);
        let overlap = r0
            .iter()
            .filter(|(i, _)| r1.iter().any(|(j, _)| i == j))
            .count();
        assert!(overlap >= 9, "top-10 overlap {overlap}");
    }

    #[test]
    fn v11_file_smaller_and_v1_readable() {
        let mut idx = VecqIndex::new(384, 7);
        for i in 0..30 {
            idx.add(&unit(384, i + 3));
        }
        let v11 = idx.to_bytes();
        // v1 file (f32 scales) — synthesize manually.
        let mut v1 = Vec::new();
        v1.extend_from_slice(&MAGIC.to_le_bytes());
        v1.extend_from_slice(&V1.to_le_bytes());
        v1.extend_from_slice(&0u16.to_le_bytes());
        v1.extend_from_slice(&(384u32).to_le_bytes());
        v1.extend_from_slice(&7u64.to_le_bytes());
        v1.extend_from_slice(&(30u32).to_le_bytes());
        for s in &idx.scales {
            v1.extend_from_slice(&s.to_le_bytes());
        }
        // codes live at a known offset in v11: 24 + 30*2
        let codes = &v11[24 + 30 * 2..];
        v1.extend_from_slice(codes);
        let from_v1 = VecqIndex::from_bytes(&v1).expect("v1 readable");
        assert_eq!(from_v1.len(), 30);
        // Search results identical (f16 rounding of scales is below tie threshold here)
        let q = unit(384, 55);
        assert_eq!(idx.search(&q, 5), from_v1.search(&q, 5));
        // v1.1 must be 2 bytes/vector smaller
        assert_eq!(v1.len() - v11.len(), 30 * 2);
    }

    #[test]
    fn deterministic_across_instances() {
        let mut a = VecqIndex::new(64, 5);
        let mut b = VecqIndex::new(64, 5);
        for i in 0..10 {
            let v = unit(64, i + 100);
            a.add(&v);
            b.add(&v);
        }
        assert_eq!(a.to_bytes(), b.to_bytes());
    }

    #[test]
    fn rejects_garbage() {
        assert!(matches!(
            VecqIndex::from_bytes(b"nonsense-not-a-file-at-all"),
            Err(Error::NotAStableFile)
        ));
        assert!(matches!(VecqIndex::from_bytes(&[]), Err(Error::Truncated)));
    }

    #[test]
    fn f16_round_trip_accuracy() {
        // Scales are ~1/sqrt(d)-ish small positives; verify decode(encode(x)) ~ x.
        for &x in &[0.001f32, 0.03, 0.5, 1.0, 3.7, 100.0] {
            let back = f16_bits_to_f32(f32_to_f16_bits(x));
            assert!((back - x).abs() / x < 0.001, "{x} -> {back}");
        }
    }
}

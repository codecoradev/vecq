//! Single-file persistence format v1.
//!
//! Layout (little-endian):
//! ```text
//! [magic "VECQ" u32][version u16 = 1][reserved u16]
//! [dim u32][seed u64][count u32]
//! [scales: count * f32]
//! [codes: count * (padded/2) bytes]
//! ```
//!
//! The seed is stored in the header so the random sign diagonal can be
//! regenerated identically on any platform: identical file -> identical
//! query results, bit for bit.

use crate::store::VecqIndex;

const MAGIC: u32 = u32::from_le_bytes(*b"VECQ");
const VERSION: u16 = 1;

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

impl VecqIndex {
    /// Serialize the index to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(24 + self.codes.len() + self.scales.len() * 4);
        out.extend_from_slice(&MAGIC.to_le_bytes());
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&(self.dim as u32).to_le_bytes());
        out.extend_from_slice(&self.seed.to_le_bytes());
        out.extend_from_slice(&(self.n as u32).to_le_bytes());
        for s in &self.scales {
            out.extend_from_slice(&s.to_le_bytes());
        }
        out.extend_from_slice(&self.codes);
        out
    }

    /// Parse an index from bytes produced by [`to_bytes`].
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
        if version != VERSION {
            return Err(Error::UnsupportedVersion(version));
        }
        let dim = rd_u32(&bytes[8..12]) as usize;
        let seed = u64::from_le_bytes(bytes[12..20].try_into().unwrap());
        let count = rd_u32(&bytes[20..24]) as usize;

        let padded = crate::rhdh::padded_dim(dim);
        let codes_bytes = padded / 2;
        let expected = 24 + count * (4 + codes_bytes);
        if bytes.len() < expected {
            return Err(Error::Truncated);
        }

        let mut scales = Vec::with_capacity(count);
        let mut off = 24;
        for _ in 0..count {
            scales.push(f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()));
            off += 4;
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
        // Identical search results
        let q = unit(384, 4242);
        assert_eq!(idx.search(&q, 10), back.search(&q, 10));
    }

    #[test]
    fn deterministic_across_instances() {
        // Two indexes built independently with the same seed must produce
        // byte-identical files — the cross-platform determinism guarantee.
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
}

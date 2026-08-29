//! Zero-copy read-only index view (`VecqView`) — issue #25.
//!
//! Parse the v1.2/v1.3/v1.4/v1.5 on-disk layout without copying codes or
//! scales, so an index can be served straight from a memory map. Any owner
//! works: `memmap2::Mmap`, `&[u8]`, `Box<[u8]>`, `Vec<u8>` — the view only
//! needs `&[u8]`.
//!
//! Scoring goes through the same shared kernel dispatch as [`crate::store::
//! VecqIndex`] and the layout is identical, so a view and a loaded index
//! over the same bytes return **bit-identical** results (tested).
//!
//! Views are always dense: `to_bytes` drops tombstones before writing, and
//! the keyed API layer is in-memory by design (#10/#16) — not available on
//! a borrowed, read-only slice.

use crate::format::{f16_bits_to_f32, Error, MAGIC, V1_2, V1_3, V1_4, V1_5};
use crate::rhdh::Rhdh;
use crate::store::{score_batch4, score_raw_dispatch};

fn rd_u16(b: &[u8]) -> u16 {
    u16::from_le_bytes([b[0], b[1]])
}

fn rd_u32(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

/// Prepared query for a [`VecqView`] (same fields and semantics as the
/// index-side [`PreparedQuery`], freed from its lifetime by cloning the
/// small rotated-query buffer).
pub struct ViewQuery {
    rotated: Vec<f32>,
    lut: [f32; 16],
}

/// Read-only, zero-copy view over an index file (or any bytes in the same
/// layout). Generic over the byte owner's lifetime.
pub struct VecqView<'a> {
    dim: usize,
    working_dim: usize,
    padded: usize,
    n: usize,
    bits: u8,
    residual: bool,
    transform: Rhdh,
    codes: &'a [u8],
    scales_raw: &'a [u8], // 2 bytes per vector, LE u16 f16 bits
    codes2: Option<&'a [u8]>,
    scales2_raw: Option<&'a [u8]>,
}

impl<'a> VecqView<'a> {
    /// Parse `bytes` as a vecq index file (v1.2+) without copying payloads.
    /// v1 files (f32 scales) are not view-eligible: their scale blocks are
    /// not the 2-byte layout shared by every current writer.
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, Error> {
        if bytes.len() < 24 || rd_u32(&bytes[0..4]) != MAGIC {
            return Err(Error::NotAStableFile);
        }
        let version = rd_u16(&bytes[4..6]);
        if version != V1_2 && version != V1_3 && version != V1_4 && version != V1_5 {
            return Err(Error::UnsupportedVersion(version));
        }
        let dim = rd_u32(&bytes[8..12]) as usize;
        let seed = u64::from_le_bytes(bytes[12..20].try_into().unwrap());
        let count = rd_u32(&bytes[20..24]) as usize;
        let working_dim = match rd_u16(&bytes[6..8]) as usize {
            0 => dim,
            w if w <= dim => w,
            w => {
                return Err(Error::InvalidWorkingDim {
                    dim,
                    working_dim: w,
                })
            }
        };
        let mut off = 24usize;
        let bits = if version == V1_5 {
            if bytes.len() < 25 {
                return Err(Error::Truncated);
            }
            let w = bytes[24];
            if !matches!(w, 4..=6) {
                return Err(Error::InvalidWidth { width: w });
            }
            off += 1;
            w
        } else {
            4
        };
        let padded = crate::rhdh::padded_dim(working_dim);
        let codes_bytes = (padded * bits as usize).div_ceil(8);
        let expected = off + count * (2 + codes_bytes);
        if bytes.len() < expected {
            return Err(Error::Truncated);
        }
        let scales_raw = &bytes[off..off + count * 2];
        let codes = &bytes[off + count * 2..off + count * (2 + codes_bytes)];
        off += count * (2 + codes_bytes);
        let mut residual = false;
        let mut codes2 = None;
        let mut scales2_raw = None;
        if version == V1_4 {
            if bytes.len() < off + count * (2 + codes_bytes) {
                return Err(Error::Truncated);
            }
            scales2_raw = Some(&bytes[off..off + count * 2]);
            codes2 = Some(&bytes[off + count * 2..off + count * (2 + codes_bytes)]);
            off += count * (2 + codes_bytes);
            residual = true;
        }
        // v1.3+ trail a keyed-slot table: validate its extent too, so a view
        // rejects any file the full loader would reject (no silent acceptance
        // of truncation in trailing sections the view itself never reads).
        if version == V1_3 || version == V1_4 || version == V1_5 {
            if bytes.len() < off + 4 {
                return Err(Error::Truncated);
            }
            let entries = rd_u32(&bytes[off..off + 4]) as usize;
            if bytes.len() < off + 4 + entries * 12 {
                return Err(Error::Truncated);
            }
        }
        Ok(Self {
            dim,
            working_dim,
            padded,
            n: count,
            bits,
            residual,
            transform: Rhdh::new(working_dim, seed),
            codes,
            scales_raw,
            codes2,
            scales2_raw,
        })
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn working_dim(&self) -> usize {
        self.working_dim
    }

    /// Code width of the viewed file (4, 5, or 6 bits).
    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// Whether the file carries a residual second pass (v1.4).
    pub fn is_residual(&self) -> bool {
        self.residual
    }

    /// Prepare a query: truncate to `working_dim`, normalize, rotate, then
    /// normalize again — mirroring the index-side `prepare_query` exactly so
    /// both paths see identical rotated queries.
    pub fn prepare_query(&self, q: &[f32]) -> ViewQuery {
        assert_eq!(q.len(), self.dim);
        let norm: f32 = q[..self.working_dim]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        assert!(norm > 0.0, "zero vector");
        let unit: Vec<f32> = q[..self.working_dim].iter().map(|x| x / norm).collect();
        let mut rotated = Vec::with_capacity(self.padded);
        self.transform.apply(&unit, &mut rotated);
        let rnorm: f32 = rotated.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in rotated.iter_mut() {
            *x /= rnorm;
        }
        let mut lut = [0f32; 16];
        for (c, slot) in lut.iter_mut().enumerate() {
            *slot = crate::lloyd::dequantize_4bit(c as u8);
        }
        ViewQuery { rotated, lut }
    }

    /// Asymmetric score of vector `idx` — same kernel dispatch and
    /// association order as [`crate::store::VecqIndex::score`].
    pub fn score(&self, pq: &ViewQuery, idx: usize) -> f32 {
        let base = idx * self.bytes_per_vector();
        let codes = &self.codes[base..base + self.bytes_per_vector()];
        let q = &pq.rotated[..self.padded];
        let raw0 = score_raw_dispatch(codes, q, &pq.lut, self.bits);
        if !self.residual {
            let s = u16::from_le_bytes(self.scales_raw[idx * 2..idx * 2 + 2].try_into().unwrap());
            return raw0 * f16_bits_to_f32(s);
        }
        let codes1 = &self.codes2.unwrap()[base..base + self.bytes_per_vector()];
        let raw1 = score_raw_dispatch(codes1, q, &pq.lut, self.bits);
        let s = u16::from_le_bytes(self.scales_raw[idx * 2..idx * 2 + 2].try_into().unwrap());
        let s2 = u16::from_le_bytes(
            self.scales2_raw.unwrap()[idx * 2..idx * 2 + 2]
                .try_into()
                .unwrap(),
        );
        raw0 * f16_bits_to_f32(s) + raw1 * f16_bits_to_f32(s2)
    }

    /// Brute-force top-k over the borrowed codes — same bounded heap,
    /// key encoding, batched kernel dispatch, and output ordering as
    /// [`crate::store::VecqIndex::search`].
    pub fn search(&self, q: &[f32], k: usize) -> Vec<(usize, f32)> {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;
        let pq = self.prepare_query(q);
        let k = k.min(self.n).max(1);
        let bpv = self.bytes_per_vector();
        let key = |s: f32| -> u32 {
            let b = s.to_bits();
            if b & 0x8000_0000 != 0 {
                !b
            } else {
                b ^ 0x8000_0000
            }
        };
        let mut heap: BinaryHeap<Reverse<(u32, usize)>> = BinaryHeap::with_capacity(k + 1);
        let consider = |s: f32, idx: usize, heap: &mut BinaryHeap<Reverse<(u32, usize)>>| {
            let ks = key(s);
            if heap.len() < k {
                heap.push(Reverse((ks, idx)));
            } else if ks > heap.peek().map(|r| r.0 .0).unwrap_or(0) {
                heap.push(Reverse((ks, idx)));
                heap.pop();
            }
        };
        let combine = |r0: f32, r1: Option<f32>, si: usize| -> f32 {
            match r1 {
                Some(r1) => {
                    let s =
                        u16::from_le_bytes(self.scales_raw[si * 2..si * 2 + 2].try_into().unwrap());
                    let s2 = u16::from_le_bytes(
                        self.scales2_raw.unwrap()[si * 2..si * 2 + 2]
                            .try_into()
                            .unwrap(),
                    );
                    r0 * f16_bits_to_f32(s) + r1 * f16_bits_to_f32(s2)
                }
                None => {
                    let s =
                        u16::from_le_bytes(self.scales_raw[si * 2..si * 2 + 2].try_into().unwrap());
                    r0 * f16_bits_to_f32(s)
                }
            }
        };
        let q_rot = &pq.rotated[..self.padded];
        let mut idx = 0;
        // Batch-4 scoring over contiguous slices — the exact loop shape of
        // the index search, so the view keeps the same kernel setup costs.
        while idx + 4 <= self.n {
            let codes4 = &self.codes[idx * bpv..(idx + 4) * bpv];
            let raw = score_batch4(codes4, q_rot, &pq.lut, self.bits);
            let raw1 = if self.residual {
                Some(score_batch4(
                    &self.codes2.unwrap()[idx * bpv..(idx + 4) * bpv],
                    q_rot,
                    &pq.lut,
                    self.bits,
                ))
            } else {
                None
            };
            for (v, &r) in raw.iter().enumerate() {
                consider(combine(r, raw1.map(|a| a[v]), idx + v), idx + v, &mut heap);
            }
            idx += 4;
        }
        while idx < self.n {
            consider(self.score(&pq, idx), idx, &mut heap);
            idx += 1;
        }
        let key_undo = |k: u32| -> u32 {
            if k & 0x8000_0000 != 0 {
                k ^ 0x8000_0000
            } else {
                !k
            }
        };
        let mut out: Vec<(usize, f32)> = heap
            .into_iter()
            .map(|r| (r.0 .1, f32::from_bits(key_undo(r.0 .0))))
            .collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("no NaN scores"));
        out
    }

    /// Bytes per vector's code block at this view's width.
    fn bytes_per_vector(&self) -> usize {
        (self.padded * self.bits as usize).div_ceil(8)
    }
}

#[cfg(test)]
mod tests {
    use super::VecqView;
    use crate::store::VecqIndex;

    fn rand_unit(dim: usize, salt: u64) -> Vec<f32> {
        let mut x = salt | 1;
        let mut v = Vec::with_capacity(dim);
        for _ in 0..dim {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            v.push((x as f32 / u32::MAX as f32 - 0.5) * 2.0);
        }
        let n: f32 = v.iter().map(|a| a * a).sum::<f32>().sqrt();
        v.iter_mut().for_each(|a| *a /= n);
        v
    }

    #[test]
    fn view_matches_loaded_index_bitwise() {
        // Same bytes -> view and fully-loaded index must return identical
        // slot order AND identical score bits, at every width.
        let dim = 128;
        for bits in [4u8, 5, 6] {
            let mut idx = VecqIndex::new(dim, 42);
            idx.set_bits(bits);
            for i in 0..20 {
                idx.add(&rand_unit(dim, i + 11));
            }
            let bytes = idx.to_bytes();
            let loaded = VecqIndex::from_bytes(&bytes).unwrap();
            let view = VecqView::from_bytes(&bytes).unwrap();
            assert_eq!(view.len(), 20);
            assert_eq!(view.bits(), bits);
            for qi in 0..5 {
                let q = rand_unit(dim, 900 + qi);
                let a = loaded.search(&q, 7);
                let b = view.search(&q, 7);
                assert_eq!(a.len(), b.len(), "bits {bits} q{qi}");
                for ((sa, fa), (sb, fb)) in a.iter().zip(b.iter()) {
                    assert_eq!(sa, sb, "bits {bits} q{qi}");
                    assert_eq!(fa.to_bits(), fb.to_bits(), "bits {bits} q{qi}");
                }
            }
        }
    }

    #[test]
    fn view_supports_residual_and_working_dim() {
        // v1.4 residual files and v1.2 working_dim files both parse to
        // bit-identical views.
        let dim = 128;
        let mut resid = VecqIndex::with_residual(dim, 7);
        for i in 0..12 {
            resid.add(&rand_unit(dim, i + 300));
        }
        let bytes = resid.to_bytes();
        let loaded = VecqIndex::from_bytes(&bytes).unwrap();
        let view = VecqView::from_bytes(&bytes).unwrap();
        assert!(view.is_residual());
        let q = rand_unit(dim, 500);
        for (sa, fa) in loaded.search(&q, 5) {
            let (_, fb) = view
                .search(&q, 5)
                .into_iter()
                .find(|(sb, _)| *sb == sa)
                .unwrap();
            assert_eq!(fa.to_bits(), fb.to_bits());
        }

        let mut wd = VecqIndex::with_working_dim(256, 64, 21);
        for i in 0..10 {
            wd.add(&rand_unit(256, i + 700));
        }
        let bytes = wd.to_bytes();
        let loaded = VecqIndex::from_bytes(&bytes).unwrap();
        let view = VecqView::from_bytes(&bytes).unwrap();
        assert_eq!(view.working_dim(), 64);
        let q = rand_unit(256, 999);
        for (sa, fa) in loaded.search(&q, 5) {
            let (_, fb) = view
                .search(&q, 5)
                .into_iter()
                .find(|(sb, _)| *sb == sa)
                .unwrap();
            assert_eq!(fa.to_bits(), fb.to_bits());
        }
    }

    #[test]
    fn view_rejects_v1_and_truncated_bytes() {
        let mut idx = VecqIndex::new(64, 3);
        idx.add(&rand_unit(64, 1));
        let bytes = idx.to_bytes();
        // v1.3 default: patch the version down to v1 (f32 scales era) —
        // views only accept v1.2+.
        let mut v1 = bytes.clone();
        v1[4] = 1;
        v1[5] = 0;
        assert!(matches!(
            VecqView::from_bytes(&v1),
            Err(crate::format::Error::UnsupportedVersion(1))
        ));
        // Too short / wrong magic must error, never panic.
        assert!(VecqView::from_bytes(&bytes[..10]).is_err());
        let mut bad = bytes.clone();
        bad[0] = b'X';
        assert!(VecqView::from_bytes(&bad).is_err());
        // Truncated payload.
        assert!(VecqView::from_bytes(&bytes[..bytes.len() - 1]).is_err());
    }
}

//! Quantized store: nibble-packed 4-bit codes + per-vector norm correction,
//! with asymmetric inner-product scoring (query f32, database 4-bit).

use crate::lloyd;
use crate::rhdh::{padded_dim, Rhdh};

/// A quantized vector database in memory.
///
/// Each vector is stored as `padded_dim / 2` bytes of 4-bit Lloyd-Max codes
/// (computed after RHDH rotation) plus one f32 correction factor. The score
/// against an f32 query is an unbiased estimate of the cosine similarity
/// after undoing the per-vector quantization scale.
pub struct VecqIndex {
    pub(crate) dim: usize,
    padded: usize,
    pub(crate) seed: u64,
    transform: Rhdh,
    pub(crate) codes: Vec<u8>, // n * padded/2 nibbles, low nibble = dim i*2
    pub(crate) scales: Vec<f32>, // per-vector dequantization scale
    pub(crate) n: usize,
}

impl VecqIndex {
    /// Create an empty index for `dim`-dimensional unit vectors.
    /// `seed` must be persisted with the index for cross-platform determinism.
    pub fn new(dim: usize, seed: u64) -> Self {
        let padded = padded_dim(dim);
        Self {
            dim,
            padded,
            seed,
            transform: Rhdh::new(dim, seed),
            codes: Vec::new(),
            scales: Vec::new(),
            n: 0,
        }
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

    pub fn seed(&self) -> u64 {
        self.seed
    }

    #[cfg(test)]
    pub(crate) fn padded(&self) -> usize {
        self.padded
    }

    /// Quantize and add one vector (any norm; normalized internally).
    pub fn add(&mut self, v: &[f32]) {
        assert_eq!(v.len(), self.dim, "vector dim mismatch");
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm > 0.0, "zero vector");
        let unit: Vec<f32> = v.iter().map(|x| x / norm).collect();

        let mut rotated = Vec::with_capacity(self.padded);
        self.transform.apply(&unit, &mut rotated);

        // Quantize to 4-bit codes, nibble-packed.
        let bytes_per_vec = self.padded / 2;
        let base = self.codes.len();
        self.codes.resize(base + bytes_per_vec, 0);
        let mut sum_sq = 0f32;
        for (i, &x) in rotated.iter().enumerate() {
            let code = lloyd::quantize_4bit(x);
            let b = base + i / 2;
            if i % 2 == 0 {
                self.codes[b] |= code;
            } else {
                self.codes[b] |= code << 4;
            }
            sum_sq += lloyd::dequantize_4bit(code).powi(2);
        }

        // Scale so that the stored vector is unit-norm: dequantized vector q
        // has norm sqrt(sum_sq); asymmetric scoring multiplies by 1/sqrt(sum_sq).
        self.scales.push(1.0 / sum_sq.sqrt());
        self.n += 1;
    }

    /// Prepare an f32 query in rotated space (call once per query).
    pub fn prepare_query(&self, q: &[f32]) -> PreparedQuery {
        assert_eq!(q.len(), self.dim);
        let norm: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        let unit: Vec<f32> = q.iter().map(|x| x / norm).collect();
        let mut rotated = Vec::with_capacity(self.padded);
        self.transform.apply(&unit, &mut rotated);
        // Normalize so ||rotated|| == 1 despite the unnormalized FWHT
        // (which scales norms by sqrt(padded)).
        let rnorm: f32 = rotated.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in rotated.iter_mut() {
            *x /= rnorm;
        }
        // Precompute dequantized lookup for fast scoring: for each of the 16
        // codes, the contribution when multiplied with the query coordinate.
        let mut lut = [0f32; 16];
        for (c, slot) in lut.iter_mut().enumerate() {
            *slot = lloyd::dequantize_4bit(c as u8);
        }
        PreparedQuery { rotated, lut, norm }
    }

    /// Asymmetric score of vector `idx` against a prepared query.
    /// Returns estimated cosine similarity in [-1, 1].
    ///
    /// Dispatches to the explicit NEON path on aarch64 and the fixed
    /// 8-bucket scalar path elsewhere. Both use the identical association
    /// order (per code byte: mul, mul, add, then add into bucket j; final
    /// pairwise tree), so they produce the same f32 bits — guarded by
    /// `neon_matches_scalar_bitwise` in tests.
    #[inline]
    pub fn score(&self, pq: &PreparedQuery, idx: usize) -> f32 {
        let base = idx * (self.padded / 2);
        let codes = &self.codes[base..base + self.padded / 2];
        let q = &pq.rotated[..self.padded];
        #[cfg(target_arch = "aarch64")]
        {
            // NEON is baseline on aarch64.
            let raw = unsafe { neon::score_neon(codes, q, &pq.lut) };
            raw * self.scales[idx]
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            score_scalar(codes, q, &pq.lut) * self.scales[idx]
        }
    }

    /// Brute-force top-k search. Returns (index, score) sorted by score desc.
    ///
    /// Uses a bounded min-heap of size k (no O(n log n) sort, no O(n)
    /// allocation per query): push while the heap is not full, then only
    /// push-and-pop when the candidate beats the current k-th score.
    pub fn search(&self, q: &[f32], k: usize) -> Vec<(usize, f32)> {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let pq = self.prepare_query(q);
        let k = k.min(self.n).max(1);
        #[cfg(target_arch = "aarch64")]
        let bpv = self.padded / 2;
        // f32 -> u32 monotonic key (NaN-safe, preserves total order):
        // flip all bits for negatives, flip sign bit for positives.
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
        #[cfg(target_arch = "aarch64")]
        let q_rot = &pq.rotated[..self.padded];
        let mut idx = 0;
        #[cfg(target_arch = "aarch64")]
        {
            // Batch 4 vectors per pass: shared q loads + LUT setup.
            while idx + 4 <= self.n {
                let codes4 = &self.codes[idx * bpv..(idx + 4) * bpv];
                let raw = unsafe { neon::score_neon4(codes4, q_rot, &pq.lut) };
                for (v, &r) in raw.iter().enumerate() {
                    consider(r * self.scales[idx + v], idx + v, &mut heap);
                }
                idx += 4;
            }
        }
        while idx < self.n {
            consider(self.score(&pq, idx), idx, &mut heap);
            idx += 1;
        }
        // Inverse of `key`: undo the sign flip to recover the exact f32 bits.
        let key_undo = |k: u32| -> u32 {
            if k & 0x8000_0000 != 0 {
                k ^ 0x8000_0000 // was a positive float
            } else {
                !k // was a negative float
            }
        };
        let mut out: Vec<(usize, f32)> = heap
            .into_iter()
            .map(|r| (r.0 .1, f32::from_bits(key_undo(r.0 .0))))
            .collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("no NaN scores"));
        out
    }
}

/// Reference 8-bucket scalar scoring. Bucket j accumulates byte j, j+8, ...
/// of every 8-byte block; final reduction is a fixed pairwise tree.
#[cfg_attr(target_arch = "aarch64", cfg(test))]
pub(crate) fn score_scalar(codes: &[u8], q: &[f32], lut: &[f32; 16]) -> f32 {
    let nb = codes.len();
    let mut acc = [0f32; 8];
    let mut i = 0;
    while i + 8 <= nb {
        for j in 0..8 {
            let b = codes[i + j];
            let c = (i + j) * 2;
            acc[j] += q[c] * lut[(b & 0x0F) as usize] + q[c + 1] * lut[(b >> 4) as usize];
        }
        i += 8;
    }
    let mut tail = 0f32;
    while i < nb {
        let b = codes[i];
        let c = i * 2;
        tail += q[c] * lut[(b & 0x0F) as usize] + q[c + 1] * lut[(b >> 4) as usize];
        i += 1;
    }
    let s01 = acc[0] + acc[1];
    let s23 = acc[2] + acc[3];
    let s45 = acc[4] + acc[5];
    let s67 = acc[6] + acc[7];
    (s01 + s23) + (s45 + s67) + tail
}

#[cfg(target_arch = "aarch64")]
mod neon {
    //! Explicit NEON scoring path, bit-identical to [`score_scalar`].
    //!
    //! Why manual intrinsics: the scalar loop gathers `lut[nibble]` with a
    //! data-dependent index, which LLVM's vectorizer refuses to
    //! auto-vectorize (verified in disassembly: zero fmla in `search`).
    //! The LUT gather maps naturally to `vqtbl4q_u8`.
    //!
    //! Bit-identity with the scalar path is structural: per 8-byte block,
    //! byte j's term `q_even*lut[lo] + q_odd*lut[hi]` (vmul, vmul, vadd —
    //! Rust never contracts into FMA) is added into accumulator lane j,
    //! blocks in increasing order, and the final reduction uses the same
    //! pairwise tree. Guarded by `neon_matches_scalar_bitwise`.
    use std::arch::aarch64::*;

    /// Gather 16 f32 from the 16-entry LUT given per-lane nibble indices.
    ///
    /// The 64-byte LUT (16 little-endian f32) is a `uint8x16x4_t` table.
    /// Four `vqtbl4q_u8` gathers produce byte-plane k (k=0..3) of all 16
    /// floats; a 4x16 byte transpose then rebuilds the 4 f32x4 registers.
    #[inline]
    unsafe fn gather16(tbl: uint8x16x4_t, nibbles: uint8x16_t) -> [float32x4_t; 4] {
        let idx = vmulq_u8(nibbles, vdupq_n_u8(4)); // byte offset of each lane's float
        let one = vdupq_n_u8(1);
        let two = vdupq_n_u8(2);
        let b0 = vqtbl4q_u8(tbl, idx);
        let b1 = vqtbl4q_u8(tbl, vaddq_u8(idx, one));
        let b2 = vqtbl4q_u8(tbl, vaddq_u8(idx, two));
        let b3 = vqtbl4q_u8(tbl, vaddq_u8(idx, vdupq_n_u8(3)));
        // Transpose: float j = (b0[j], b1[j], b2[j], b3[j]).
        let z01 = vzip1q_u8(b0, b1); // u16 lanes (b0j, b1j)
        let z23 = vzip1q_u8(b2, b3); // u16 lanes (b2j, b3j)
        let z01b = vzip2q_u8(b0, b1);
        let z23b = vzip2q_u8(b2, b3);
        let lo16 = vreinterpretq_u16_u8(z01);
        let hi16 = vreinterpretq_u16_u8(z23);
        let lo16b = vreinterpretq_u16_u8(z01b);
        let hi16b = vreinterpretq_u16_u8(z23b);
        [
            vreinterpretq_f32_u8(vreinterpretq_u8_u16(vzip1q_u16(lo16, hi16))),
            vreinterpretq_f32_u8(vreinterpretq_u8_u16(vzip2q_u16(lo16, hi16))),
            vreinterpretq_f32_u8(vreinterpretq_u8_u16(vzip1q_u16(lo16b, hi16b))),
            vreinterpretq_f32_u8(vreinterpretq_u8_u16(vzip2q_u16(lo16b, hi16b))),
        ]
    }

    /// NEON scoring over one vector's codes. See module docs.
    #[inline]
    pub unsafe fn score_neon(codes: &[u8], q: &[f32], lut: &[f32; 16]) -> f32 {
        // Build the 64-byte LUT table for vqtbl4q_u8.
        let mut bytes = [0u8; 64];
        for (c, &v) in lut.iter().enumerate() {
            bytes[c * 4..c * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        let tbl = uint8x16x4_t(
            vld1q_u8(bytes[0..16].as_ptr()),
            vld1q_u8(bytes[16..32].as_ptr()),
            vld1q_u8(bytes[32..48].as_ptr()),
            vld1q_u8(bytes[48..64].as_ptr()),
        );
        // acc_lo lanes 0-3 = scalar buckets 0-3; acc_hi lanes 0-3 = 4-7.
        let mut acc_lo = vdupq_n_f32(0.0);
        let mut acc_hi = vdupq_n_f32(0.0);
        let nb = codes.len();
        let mut i = 0;
        while i + 8 <= nb {
            let b8 = vld1_u8(codes.as_ptr().add(i)); // 8 code bytes (safe load)
                                                     // Nibble layout for the gather: lanes 0-7 = low nibbles (even
                                                     // dims), lanes 8-15 = high nibbles (odd dims).
            let lo = vand_u8(b8, vdup_n_u8(0x0F));
            let hi = vshr_n_u8(b8, 4);
            let nibbles = vcombine_u8(lo, hi); // [lo_0..lo_7, hi_0..hi_7]
            let g = gather16(tbl, nibbles);
            // g[0] = lut[lo_0..3], g[1] = lut[lo_4..7],
            // g[2] = lut[hi_0..3], g[3] = lut[hi_4..7].
            // Load q[2i .. 2i+16) and deinterleave even/odd dims.
            let q0 = vld1q_f32(q.as_ptr().add(i * 2));
            let q1 = vld1q_f32(q.as_ptr().add(i * 2 + 4));
            let q2 = vld1q_f32(q.as_ptr().add(i * 2 + 8));
            let q3 = vld1q_f32(q.as_ptr().add(i * 2 + 12));
            let q_even_lo = vuzp1q_f32(q0, q1); // dims 2i, 2i+2, 2i+4, 2i+6
            let q_even_hi = vuzp1q_f32(q2, q3);
            let q_odd_lo = vuzp2q_f32(q0, q1);
            let q_odd_hi = vuzp2q_f32(q2, q3);
            // term = q_even*lut[lo] + q_odd*lut[hi]  (mul, mul, add — no FMA)
            let t_lo = vaddq_f32(vmulq_f32(q_even_lo, g[0]), vmulq_f32(q_odd_lo, g[2]));
            let t_hi = vaddq_f32(vmulq_f32(q_even_hi, g[1]), vmulq_f32(q_odd_hi, g[3]));
            acc_lo = vaddq_f32(acc_lo, t_lo);
            acc_hi = vaddq_f32(acc_hi, t_hi);
            i += 8;
        }
        // Extract buckets and reduce with the scalar pairwise tree.
        let mut acc = [0f32; 8];
        acc[0] = vgetq_lane_f32(acc_lo, 0);
        acc[1] = vgetq_lane_f32(acc_lo, 1);
        acc[2] = vgetq_lane_f32(acc_lo, 2);
        acc[3] = vgetq_lane_f32(acc_lo, 3);
        acc[4] = vgetq_lane_f32(acc_hi, 0);
        acc[5] = vgetq_lane_f32(acc_hi, 1);
        acc[6] = vgetq_lane_f32(acc_hi, 2);
        acc[7] = vgetq_lane_f32(acc_hi, 3);
        // Scalar tail for the last (< 8) code bytes. padded is a multiple of
        // 8 elements (padded/2 bytes multiple of 4), so nb % 8 is 0 or 4.
        let mut tail = 0f32;
        while i < nb {
            let b = codes[i];
            let c = i * 2;
            tail += q[c] * lut[(b & 0x0F) as usize] + q[c + 1] * lut[(b >> 4) as usize];
            i += 1;
        }
        let s01 = acc[0] + acc[1];
        let s23 = acc[2] + acc[3];
        let s45 = acc[4] + acc[5];
        let s67 = acc[6] + acc[7];
        (s01 + s23) + (s45 + s67) + tail
    }

    /// Score 4 consecutive vectors at once, amortizing the q loads and LUT
    /// table setup across all 4. Each vector accumulates in the exact same
    /// per-lane order as [`score_neon`], so results are bit-identical.
    ///
    /// Returns raw (pre-scale) scores; the caller multiplies by `scales`.
    #[inline]
    pub unsafe fn score_neon4(codes4: &[u8], q: &[f32], lut: &[f32; 16]) -> [f32; 4] {
        let mut bytes = [0u8; 64];
        for (c, &v) in lut.iter().enumerate() {
            bytes[c * 4..c * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        let tbl = uint8x16x4_t(
            vld1q_u8(bytes[0..16].as_ptr()),
            vld1q_u8(bytes[16..32].as_ptr()),
            vld1q_u8(bytes[32..48].as_ptr()),
            vld1q_u8(bytes[48..64].as_ptr()),
        );
        let nb = codes4.len() / 4; // bytes per vector
        let mut acc_lo = [vdupq_n_f32(0.0); 4];
        let mut acc_hi = [vdupq_n_f32(0.0); 4];
        let mut i = 0;
        while i + 8 <= nb {
            // Shared q loads for this block: q[2i .. 2i+16), deinterleaved.
            let q0 = vld1q_f32(q.as_ptr().add(i * 2));
            let q1 = vld1q_f32(q.as_ptr().add(i * 2 + 4));
            let q2 = vld1q_f32(q.as_ptr().add(i * 2 + 8));
            let q3 = vld1q_f32(q.as_ptr().add(i * 2 + 12));
            let q_even_lo = vuzp1q_f32(q0, q1);
            let q_even_hi = vuzp1q_f32(q2, q3);
            let q_odd_lo = vuzp2q_f32(q0, q1);
            let q_odd_hi = vuzp2q_f32(q2, q3);
            for v in 0..4 {
                let b8 = vld1_u8(codes4.as_ptr().add(v * nb + i));
                let lo = vand_u8(b8, vdup_n_u8(0x0F));
                let hi = vshr_n_u8(b8, 4);
                let nibbles = vcombine_u8(lo, hi);
                let g = gather16(tbl, nibbles);
                let t_lo = vaddq_f32(vmulq_f32(q_even_lo, g[0]), vmulq_f32(q_odd_lo, g[2]));
                let t_hi = vaddq_f32(vmulq_f32(q_even_hi, g[1]), vmulq_f32(q_odd_hi, g[3]));
                acc_lo[v] = vaddq_f32(acc_lo[v], t_lo);
                acc_hi[v] = vaddq_f32(acc_hi[v], t_hi);
            }
            i += 8;
        }
        let mut out = [0f32; 4];
        for v in 0..4 {
            // Same lane extraction + pairwise reduction as score_neon.
            let mut a = [0f32; 8];
            a[0] = vgetq_lane_f32(acc_lo[v], 0);
            a[1] = vgetq_lane_f32(acc_lo[v], 1);
            a[2] = vgetq_lane_f32(acc_lo[v], 2);
            a[3] = vgetq_lane_f32(acc_lo[v], 3);
            a[4] = vgetq_lane_f32(acc_hi[v], 0);
            a[5] = vgetq_lane_f32(acc_hi[v], 1);
            a[6] = vgetq_lane_f32(acc_hi[v], 2);
            a[7] = vgetq_lane_f32(acc_hi[v], 3);
            // Scalar tail for the last (< 8) code bytes.
            let mut tail = 0f32;
            let mut j = i;
            while j < nb {
                let b = codes4[v * nb + j];
                let c = j * 2;
                tail += q[c] * lut[(b & 0x0F) as usize] + q[c + 1] * lut[(b >> 4) as usize];
                j += 1;
            }
            let s01 = a[0] + a[1];
            let s23 = a[2] + a[3];
            let s45 = a[4] + a[5];
            let s67 = a[6] + a[7];
            out[v] = (s01 + s23) + (s45 + s67) + tail;
        }
        out
    }
}

/// A query preprocessed in the quantized domain.
pub struct PreparedQuery {
    rotated: Vec<f32>,
    lut: [f32; 16],
    #[allow(dead_code)]
    norm: f32,
}

/// Exact cosine similarity between two f32 vectors (ground truth helper).
pub fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0f32;
    let mut na = 0f32;
    let mut nb = 0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    dot / (na.sqrt() * nb.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_unit(dim: usize, seed: u64) -> Vec<f32> {
        // xorshift normals, then normalize
        let mut x = seed | 1;
        let mut v = Vec::with_capacity(dim);
        for _ in 0..dim {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let u1 = ((x >> 11) as f64 / (1u64 << 53) as f64).max(1e-12);
            x ^= x << 16;
            let u2 = (x >> 11) as f64 / (1u64 << 53) as f64;
            v.push(((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32);
        }
        let norm: f32 = v.iter().map(|a| a * a).sum::<f32>().sqrt();
        v.into_iter().map(|a| a / norm).collect()
    }

    #[test]
    fn score_correlates_with_exact_cosine() {
        let dim = 128;
        let mut idx = VecqIndex::new(dim, 7);
        let base: Vec<Vec<f32>> = (0..200).map(|i| rand_unit(dim, i + 1)).collect();
        for v in &base {
            idx.add(v);
        }
        let q = rand_unit(dim, 999);
        let pq = idx.prepare_query(&q);
        let exact: Vec<f32> = base.iter().map(|v| cosine_f32(&q, v)).collect();
        let mut max_err = 0f32;
        for (i, &e) in exact.iter().enumerate().take(200) {
            let est = idx.score(&pq, i);
            max_err = max_err.max((est - e).abs());
        }
        assert!(max_err < 0.2, "max score error {max_err}");
    }

    #[test]
    fn score_reproducible_and_close_to_naive() {
        let dim = 128;
        let mut idx = VecqIndex::new(dim, 13);
        for i in 0..50 {
            idx.add(&rand_unit(dim, i + 21));
        }
        let q = rand_unit(dim, 321);
        let pq = idx.prepare_query(&q);
        for vi in 0..50 {
            let base = vi * (idx.padded() / 2);
            let mut naive = 0f32;
            for i in 0..idx.padded() {
                let b = idx.codes[base + i / 2];
                let code = if i % 2 == 0 { b & 0x0F } else { b >> 4 };
                naive += pq.rotated[i] * pq.lut[code as usize];
            }
            let s = idx.score(&pq, vi);
            assert_eq!(s.to_bits(), idx.score(&pq, vi).to_bits());
            assert!((s - naive * idx.scales[vi]).abs() < 1e-5, "vector {vi}");
        }
    }

    #[test]
    fn neon_matches_scalar_bitwise() {
        let dim = 128;
        let mut idx = VecqIndex::new(dim, 42);
        for i in 0..30 {
            idx.add(&rand_unit(dim, i + 500));
        }
        let q = rand_unit(dim, 777);
        let pq = idx.prepare_query(&q);
        for vi in 0..30 {
            let base = vi * (idx.padded() / 2);
            let codes = &idx.codes[base..base + idx.padded() / 2];
            let qslice = &pq.rotated[..idx.padded()];
            #[cfg(target_arch = "aarch64")]
            {
                let neon = unsafe { neon::score_neon(codes, qslice, &pq.lut) };
                let scalar = score_scalar(codes, qslice, &pq.lut);
                assert_eq!(
                    neon.to_bits(),
                    scalar.to_bits(),
                    "vector {vi}: NEON and scalar diverged"
                );
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                let _ = (base, codes, qslice);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon4_matches_neon_bitwise() {
        let dim = 128;
        let mut idx = VecqIndex::new(dim, 91);
        for i in 0..12 {
            idx.add(&rand_unit(dim, i + 90));
        }
        let q = rand_unit(dim, 1234);
        let pq = idx.prepare_query(&q);
        let bpv = idx.padded() / 2;
        for chunk_start in (0..12).step_by(4) {
            let codes4 = &idx.codes[chunk_start * bpv..(chunk_start + 4) * bpv];
            let qslice = &pq.rotated[..idx.padded()];
            {
                let batched = unsafe { neon::score_neon4(codes4, qslice, &pq.lut) };
                for v in 0..4 {
                    let single = unsafe {
                        neon::score_neon(&codes4[v * bpv..(v + 1) * bpv], qslice, &pq.lut)
                    };
                    assert_eq!(
                        batched[v].to_bits(),
                        single.to_bits(),
                        "chunk {chunk_start} vec {v}: neon4 diverged from neon"
                    );
                }
            }
        }
    }

    #[test]
    fn search_returns_sorted_results() {
        let dim = 64;
        let mut idx = VecqIndex::new(dim, 3);
        for i in 0..50 {
            idx.add(&rand_unit(dim, i * 31 + 5));
        }
        let q = rand_unit(dim, 77);
        let res = idx.search(&q, 5);
        assert_eq!(res.len(), 5);
        for w in res.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
    }

    #[test]
    fn quantized_size_is_one_eighth() {
        let dim = 384;
        let mut idx = VecqIndex::new(dim, 1);
        idx.add(&rand_unit(dim, 11));
        assert_eq!(idx.codes.len(), 512 / 2);
    }
}

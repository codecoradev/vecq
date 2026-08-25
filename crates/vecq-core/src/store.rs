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
        let mut sum_sq = 0f32; // ||dequantized - nothing||: accumulate ||q||^2
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
    #[inline]
    pub fn score(&self, pq: &PreparedQuery, idx: usize) -> f32 {
        let base = idx * (self.padded / 2);
        let mut dot = 0f32;
        for i in 0..self.padded {
            let b = self.codes[base + i / 2];
            let code = if i % 2 == 0 { b & 0x0F } else { b >> 4 };
            dot += pq.rotated[i] * pq.lut[code as usize];
        }
        // Divide by ||q_rotated|| * ||dequant||; both are unit after
        // prepare_query normalization and the stored scale respectively.
        dot * self.scales[idx]
    }

    /// Brute-force top-k search. Returns (index, score) sorted by score desc.
    pub fn search(&self, q: &[f32], k: usize) -> Vec<(usize, f32)> {
        let pq = self.prepare_query(q);
        let k = k.min(self.n);
        let mut scored: Vec<(usize, f32)> =
            (0..self.n).map(|idx| (idx, self.score(&pq, idx))).collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("no NaN scores"));
        scored.truncate(k);
        scored
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
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
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
        // Exact scores
        let exact: Vec<f32> = base.iter().map(|v| cosine_f32(&q, v)).collect();
        let mut max_err = 0f32;
        for (i, &e) in exact.iter().enumerate().take(200) {
            let est = idx.score(&pq, i);
            max_err = max_err.max((est - e).abs());
        }
        // Per-dimension quantization error ~0.107 var; aggregated over 128
        // rotated dims the cosine estimate error should stay well below 0.2.
        assert!(max_err < 0.2, "max score error {max_err}");
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
        // f32 = 384*4 = 1536 bytes; quantized = padded(512)/2 + 4 = 260
        assert_eq!(idx.codes.len(), 512 / 2);
        assert_eq!(idx.codes.len() + 4, 260);
    }
}

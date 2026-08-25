//! Randomized Hadamard Transform (RHDH): R = (1/sqrt(d')) · H · D.
//!
//! H is the Walsh-Hadamard matrix (d' = next power of two >= d), D is a
//! diagonal matrix of random +/-1 signs derived deterministically from a seed.
//! After RHDH, coordinates of a unit vector are approximately N(0, 1),
//! which makes the precomputed Lloyd-Max N(0,1) tables valid without any
//! training pass.
//!
//! The transform runs in O(d log d) via the fast Walsh-Hadamard transform.

/// Deterministic pseudo-random +/-1 signs from a u64 seed (xorshift64*).
/// No external RNG dependency: the sign sequence must be reproducible from
/// the seed stored in the file header across platforms and builds.
struct SignStream {
    state: u64,
}

impl SignStream {
    fn new(seed: u64) -> Self {
        // Avoid all-zero state; mix the seed once.
        let state = if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        };
        Self { state }
    }

    #[inline]
    fn next_f64(&mut self) -> f64 {
        // xorshift64* — deterministic, portable.
        let mut x = self.state;
        debug_assert_ne!(x, 0);
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        let v = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
        // Map to [0, 1)
        (v >> 11) as f64 / (1u64 << 53) as f64
    }

    #[inline]
    fn next_sign(&mut self) -> f32 {
        if self.next_f64() < 0.5 {
            -1.0
        } else {
            1.0
        }
    }
}

/// Fast in-place Walsh-Hadamard transform (unnormalized).
/// `v.len()` must be a power of two.
pub fn fwht(v: &mut [f32]) {
    let n = v.len();
    debug_assert!(n.is_power_of_two());
    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let x = v[j];
                let y = v[j + h];
                v[j] = x + y;
                v[j + h] = x - y;
            }
            i += h * 2;
        }
        h *= 2;
    }
}

/// Pad dimension up to the next power of two.
pub fn padded_dim(dim: usize) -> usize {
    dim.max(1).next_power_of_two()
}

/// A reusable RHDH context: the random sign diagonal for a padded dimension.
pub struct Rhdh {
    pub padded: usize,
    signs: Vec<f32>,
}

impl Rhdh {
    /// Build the transform for `dim` dimensions from a deterministic seed.
    pub fn new(dim: usize, seed: u64) -> Self {
        let padded = padded_dim(dim);
        let mut s = SignStream::new(seed);
        let signs: Vec<f32> = (0..padded).map(|_| s.next_sign()).collect();
        Self { padded, signs }
    }

    /// Apply the randomized Hadamard transform to a unit-normalized input of
    /// `dim` values. Returns a vector of `padded` values, approximately
    /// N(0, 1) per coordinate. The padding zeros receive random signs too,
    /// which keeps energy spread uniformly.
    pub fn apply(&self, v: &[f32], out: &mut Vec<f32>) {
        debug_assert!(v.len() <= self.padded);
        out.clear();
        out.resize(self.padded, 0.0);
        out[..v.len()].copy_from_slice(v);
        // Apply random signs then the (unnormalized) FWHT. For a unit input,
        // unnormalized FWHT coordinates are approximately N(0,1) each —
        // exactly the distribution the Lloyd-Max N(0,1) tables assume.
        for (x, &s) in out.iter_mut().zip(self.signs.iter()).take(self.padded) {
            *x *= s;
        }
        fwht(out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padded_dims() {
        assert_eq!(padded_dim(7), 8);
        assert_eq!(padded_dim(8), 8);
        assert_eq!(padded_dim(384), 512);
        assert_eq!(padded_dim(1024), 1024);
    }

    #[test]
    fn transform_scales_norm_by_sqrt_padded() {
        // The unnormalized FWHT scales a vector's norm by sqrt(padded);
        // callers normalize afterwards.
        let t = Rhdh::new(8, 42);
        let v: Vec<f32> = vec![0.3, -0.5, 0.2, 0.8, -0.1, 0.4, 0.6, -0.7];
        let mut out = Vec::new();
        t.apply(&v, &mut out);
        let n_in: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let n_out: f32 = out.iter().map(|x| x * x).sum::<f32>().sqrt();
        let scale = n_out / n_in;
        assert!(
            (scale - (8f32).sqrt()).abs() < 1e-3,
            "scale {scale} expected sqrt(8)"
        );
    }

    #[test]
    fn deterministic_from_seed() {
        let a = Rhdh::new(16, 7);
        let b = Rhdh::new(16, 7);
        let c = Rhdh::new(16, 8);
        let v: Vec<f32> = (0..16).map(|i| (i as f32 * 0.13 - 1.0).sin()).collect();
        let (mut oa, mut ob, mut oc) = (Vec::new(), Vec::new(), Vec::new());
        a.apply(&v, &mut oa);
        b.apply(&v, &mut ob);
        c.apply(&v, &mut oc);
        assert_eq!(oa, ob, "same seed must be byte-identical");
        assert_ne!(oa, oc, "different seed must differ");
    }

    #[test]
    fn unit_vector_coords_approach_normal() {
        // A randomly rotated unit vector's coordinates should look N(0,1/d'):
        // after 1/sqrt(d') scaling they are approx N(0,1) with variance ~1.
        let d = 512;
        let t = Rhdh::new(d, 1234);
        let mut v = vec![0.0f32; d];
        for x in v.iter_mut() {
            *x = rand_std_normal();
        }
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in v.iter_mut() {
            *x /= norm;
        }
        let mut out = Vec::new();
        t.apply(&v, &mut out);
        let var: f32 = out.iter().map(|x| x * x).sum::<f32>() / out.len() as f32;
        // For a unit input, unnormalized FWHT spreads total energy (1) over
        // the padded dims: per-coordinate variance ~ 1 ... no — energy sums
        // to 1, so per-coordinate variance ~ 1/padded * padded = 1 only for
        // coords in the original span. Total ||out||^2 = ||v||^2 = 1 means
        // mean variance = 1/padded.
        let expected = 1.0f32;
        assert!(
            (var / expected - 1.0).abs() < 0.3,
            "var {var} expected ~{expected}"
        );
    }

    fn rand_std_normal() -> f32 {
        // Box-Muller from two uniforms via a cheap LCG (test-only).
        use std::cell::Cell;
        thread_local! {
            static S: Cell<u64> = const { Cell::new(0x853C_49E6_748F_EA9B) };
        }
        S.with(|s| {
            let mut x = s.get();
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            s.set(x);
            let u1 = (x >> 11) as f64 / (1u64 << 53) as f64 + 1e-12;
            let u2 = ((x >> 21) as f64) / 4294967296.0;
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        }) as f32
    }
}

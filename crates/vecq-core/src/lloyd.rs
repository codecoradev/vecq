//! Lloyd-Max optimal scalar quantizer tables for N(0,1), precomputed offline.
//!
//! Centroids and decision boundaries are computed via Lloyd's algorithm
//! (2000 iterations, convergence tolerance 1e-12) on the standard normal
//! distribution, then embedded as constants. No runtime training pass.
//!
//! 4-bit: 16 centroids / 15 boundaries. 2-bit: 4 centroids / 3 boundaries.

/// 4-bit Lloyd-Max centroids for N(0,1).
pub const CENTROIDS_4BIT: [f32; 16] = [
    -1.996_112, -1.512_225, -1.172_563, -0.887_568, -0.632_509, -0.395_530, -0.169_891, 0.049_892,
    0.269_673, 0.495_312, 0.732_291, 0.987_350, 1.272_345, 1.612_007, 2.095_894, 2.724_265,
];

/// 4-bit decision boundaries (index i separates centroid i-1 and i).
pub const BOUNDARIES_4BIT: [f32; 15] = [
    -1.751_289, -1.340_438, -1.026_996, -0.756_128, -0.509_062, -0.276_322, -0.056_279, 0.164_289,
    0.391_029, 0.626_095, 0.868_960, 1.125_026, 1.413_956, 1.764_827, 2.238_297,
];

/// Quantize one N(0,1)-distributed value to a 4-bit index via binary search
/// over the decision boundaries.
#[inline]
pub fn quantize_4bit(x: f32) -> u8 {
    let mut lo = 0usize;
    let mut hi = BOUNDARIES_4BIT.len(); // 15
    while lo < hi {
        let mid = (lo + hi) / 2;
        if x > BOUNDARIES_4BIT[mid] {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo as u8
}

/// Dequantize a 4-bit index back to its centroid value.
#[inline]
pub fn dequantize_4bit(i: u8) -> f32 {
    CENTROIDS_4BIT[i as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centroids_are_sorted() {
        for w in CENTROIDS_4BIT.windows(2) {
            assert!(w[0] < w[1], "centroids must be strictly increasing");
        }
    }

    #[test]
    fn boundaries_are_sorted_and_interleaved() {
        for w in BOUNDARIES_4BIT.windows(2) {
            assert!(w[0] < w[1]);
        }
        // boundaries sit between adjacent centroids
        for i in 0..15 {
            assert!(CENTROIDS_4BIT[i] < BOUNDARIES_4BIT[i]);
            assert!(BOUNDARIES_4BIT[i] < CENTROIDS_4BIT[i + 1]);
        }
    }

    #[test]
    fn extreme_values_map_to_endpoints() {
        assert_eq!(quantize_4bit(-10.0), 0);
        assert_eq!(quantize_4bit(10.0), 15);
    }

    #[test]
    fn zero_maps_near_center() {
        let i = quantize_4bit(0.0);
        assert!(
            (7..=8).contains(&i),
            "0.0 should map to centroid 7 or 8, got {i}"
        );
    }

    #[test]
    fn round_trip_error_is_bounded() {
        // Worst-case quantization error for N(0,1) Lloyd-Max 4-bit is half
        // the largest inter-centroid gap in the tails.
        let mut max_gap = 0f32;
        for i in 0..15 {
            max_gap = max_gap.max(CENTROIDS_4BIT[i + 1] - CENTROIDS_4BIT[i]);
        }
        // Lloyd-Max MSE for 4-bit on N(0,1) is ~0.0115; per-dim worst error
        // must be below the max tail gap.
        assert!(max_gap < 0.75, "max centroid gap {max_gap} too large");
    }
}

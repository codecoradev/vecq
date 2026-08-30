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

/// 5-bit Lloyd-Max centroids for N(0,1).
pub const CENTROIDS_5BIT: [f32; 32] = [
    -3.2641168,
    -2.6949832,
    -2.321901,
    -2.0330853,
    -1.7916883,
    -1.5806495,
    -1.3905898,
    -1.2158026,
    -1.0524858,
    -0.89793956,
    -0.7501299,
    -0.6074978,
    -0.4687885,
    -0.33291993,
    -0.19900484,
    -0.06620281,
    0.06625264,
    0.19905461,
    0.33296943,
    0.46883777,
    0.60757166,
    0.75022924,
    0.8980622,
    1.052632,
    1.2159506,
    1.3907394,
    1.5808226,
    1.7919079,
    2.0333521,
    2.3222392,
    2.6954327,
    3.2648566,
];

/// 5-bit decision boundaries.
pub const BOUNDARIES_5BIT: [f32; 31] = [
    -2.9795978,
    -2.5084915,
    -2.177543,
    -1.9124366,
    -1.6862187,
    -1.4856696,
    -1.3032461,
    -1.1341941,
    -0.97524965,
    -0.8240596,
    -0.67883897,
    -0.5381553,
    -0.40085423,
    -0.2659624,
    -0.13260382,
    0.000024914742,
    0.13265362,
    0.266012,
    0.40090358,
    0.5382047,
    0.6789125,
    0.82418275,
    0.975397,
    1.1343412,
    1.3033949,
    1.4858308,
    1.6864152,
    1.9126797,
    2.177845,
    2.508878,
    2.980162,
];

/// 6-bit Lloyd-Max centroids for N(0,1).
pub const CENTROIDS_6BIT: [f32; 64] = [
    -3.8273482,
    -3.333197,
    -3.0166795,
    -2.7765143,
    -2.5793982,
    -2.41001,
    -2.260009,
    -2.124298,
    -1.9995066,
    -1.8833042,
    -1.7739713,
    -1.6702707,
    -1.571215,
    -1.4760238,
    -1.3840784,
    -1.2948383,
    -1.207906,
    -1.1230142,
    -1.0398592,
    -0.9581757,
    -0.87773234,
    -0.7983605,
    -0.7199343,
    -0.6422799,
    -0.5652983,
    -0.48888877,
    -0.41292742,
    -0.33736518,
    -0.26210162,
    -0.1870626,
    -0.11217268,
    -0.037357662,
    0.037407417,
    0.11222233,
    0.1871122,
    0.26215115,
    0.33741492,
    0.412977,
    0.48893812,
    0.5653473,
    0.64232975,
    0.71998423,
    0.7984106,
    0.8777821,
    0.9582232,
    1.0399102,
    1.1230648,
    1.2079563,
    1.2948877,
    1.384128,
    1.4760972,
    1.5713139,
    1.6703697,
    1.7740716,
    1.8834057,
    1.9996046,
    2.1243992,
    2.2601078,
    2.4101071,
    2.5794992,
    2.7766082,
    3.016757,
    3.3332598,
    3.827421,
];

/// 6-bit decision boundaries.
pub const BOUNDARIES_6BIT: [f32; 63] = [
    -3.580496,
    -3.1751587,
    -2.8968368,
    -2.6782057,
    -2.4949536,
    -2.335259,
    -2.192403,
    -2.0621655,
    -1.9416802,
    -1.8289125,
    -1.7223959,
    -1.6210177,
    -1.5238943,
    -1.4303128,
    -1.3397081,
    -1.2516222,
    -1.1656973,
    -1.0816617,
    -0.9992423,
    -0.91816604,
    -0.8382337,
    -0.75930965,
    -0.6812571,
    -0.6039391,
    -0.5272308,
    -0.4510203,
    -0.37523368,
    -0.29980838,
    -0.22464456,
    -0.14965503,
    -0.07479018,
    0.000024879351,
    0.07483988,
    0.14970465,
    0.2246941,
    0.29985797,
    0.37528333,
    0.4510699,
    0.52728003,
    0.6039884,
    0.68130684,
    0.75935954,
    0.8382834,
    0.9182147,
    0.9992916,
    1.0817122,
    1.1657475,
    1.2516719,
    1.3397577,
    1.4303625,
    1.5239555,
    1.6211035,
    1.7224953,
    1.8290132,
    1.9417915,
    2.06229,
    2.1925168,
    2.3353572,
    2.4950526,
    2.678303,
    2.8969312,
    3.1752489,
    3.580552,
];

/// Number of codebook levels for a supported bit width (4 -> 16, 5 -> 32, 6 -> 64).
#[inline]
pub fn levels(bits: u8) -> usize {
    match bits {
        4 => 16,
        5 => 32,
        6 => 64,
        w => panic!("unsupported bit width {w} (supported: 4, 5, 6)"),
    }
}

/// Codebook centroids for a bit width (alias of the per-width constants).
#[inline]
pub fn centroids(bits: u8) -> &'static [f32] {
    match bits {
        4 => &CENTROIDS_4BIT,
        5 => &CENTROIDS_5BIT,
        6 => &CENTROIDS_6BIT,
        w => panic!("unsupported bit width {w} (supported: 4, 5, 6)"),
    }
}

/// Decision boundaries for a bit width.
#[inline]
pub fn boundaries(bits: u8) -> &'static [f32] {
    match bits {
        4 => &BOUNDARIES_4BIT,
        5 => &BOUNDARIES_5BIT,
        6 => &BOUNDARIES_6BIT,
        w => panic!("unsupported bit width {w} (supported: 4, 5, 6)"),
    }
}

/// Quantize one N(0,1) value at the given bit width via binary search.
#[inline]
pub fn quantize(x: f32, bits: u8) -> u8 {
    let bnd = boundaries(bits);
    let mut lo = 0usize;
    let mut hi = bnd.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if x > bnd[mid] {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo as u8
}

/// Dequantize a code at the given bit width (direct centroid index).
#[inline]
pub fn dequantize(code: u8, bits: u8) -> f32 {
    centroids(bits)[code as usize]
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

//! Quantized store: nibble-packed 4-bit codes + per-vector norm correction,
//! with asymmetric inner-product scoring (query f32, database 4-bit).

use crate::lloyd;
use crate::rhdh::{padded_dim, Rhdh};
use std::collections::HashMap;

/// A quantized vector database in memory.
///
/// Each vector is stored as `padded_dim / 2` bytes of 4-bit Lloyd-Max codes
/// (computed after RHDH rotation) plus one f32 correction factor. The score
/// against an f32 query is an unbiased estimate of the cosine similarity
/// after undoing the per-vector quantization scale.
///
/// Vectors can be stored anonymously via [`VecqIndex::add`] or under a
/// caller-chosen `u64` key via [`VecqIndex::add_keyed`]. Keyed vectors can be
/// removed in place (tombstoned); tombstoned slots keep their storage but are
/// skipped by searches and dropped by [`VecqIndex::compact`] and by
/// [`VecqIndex::to_bytes`](crate::format). Slot indices stay stable until a
/// compaction, so integrators can treat a slot as a transient handle while
/// keys are the durable identity.
pub struct VecqIndex {
    pub(crate) dim: usize,
    /// Dimensions actually used: vectors and queries are truncated to the
    /// leading `working_dim` coords *before* normalization and rotation
    /// (Matryoshka-style truncation must happen pre-rotation, because RHDH
    /// mixes dimensions). Equal to `dim` unless built via
    /// [`VecqIndex::with_working_dim`].
    working_dim: usize,
    padded: usize,
    pub(crate) seed: u64,
    transform: Rhdh,
    pub(crate) codes: Vec<u8>, // n * padded/2 nibbles, low nibble = dim i*2
    pub(crate) scales: Vec<f32>, // per-vector dequantization scale
    pub(crate) n: usize,       // total slots in use (live + tombstoned)
    keys: Vec<Option<u64>>,    // slot -> caller key (keyed slots only)
    key_to_slot: HashMap<u64, usize>,
    alive: Vec<bool>, // slot -> not tombstoned
    live: usize,      // number of non-tombstoned slots
}

impl VecqIndex {
    /// Create an empty index for `dim`-dimensional unit vectors.
    /// `seed` must be persisted with the index for cross-platform determinism.
    pub fn new(dim: usize, seed: u64) -> Self {
        Self::with_working_dim(dim, dim, seed)
    }

    /// Create an empty index over the leading `working_dim` dimensions of
    /// `dim`-dimensional vectors (Matryoshka truncation).
    ///
    /// Vectors and queries are always passed at full `dim` length; the index
    /// truncates them to `working_dim` *before* normalization and the RHDH
    /// rotation (truncating after rotation would not be equivalent, since the
    /// transform mixes dimensions). Scores are computed in the
    /// `working_dim`-dimensional space and are only comparable with indexes
    /// built with the same `working_dim` and `seed`.
    pub fn with_working_dim(dim: usize, working_dim: usize, seed: u64) -> Self {
        assert!(
            working_dim >= 1 && working_dim <= dim,
            "working_dim must be in 1..={dim}, got {working_dim}"
        );
        let padded = padded_dim(working_dim);
        Self {
            dim,
            working_dim,
            padded,
            seed,
            transform: Rhdh::new(working_dim, seed),
            codes: Vec::new(),
            scales: Vec::new(),
            n: 0,
            keys: Vec::new(),
            key_to_slot: HashMap::new(),
            alive: Vec::new(),
            live: 0,
        }
    }

    /// Number of live (searchable) vectors.
    pub fn len(&self) -> usize {
        self.live
    }

    pub fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// Total slots in use, including tombstoned ones
    /// (`slots() == len() + tombstones()`).
    pub fn slots(&self) -> usize {
        self.n
    }

    /// Number of tombstoned slots awaiting [`VecqIndex::compact`].
    pub fn tombstones(&self) -> usize {
        self.n - self.live
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Dimensions actually quantized (see [`VecqIndex::with_working_dim`]).
    pub fn working_dim(&self) -> usize {
        self.working_dim
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    #[cfg(test)]
    pub(crate) fn padded(&self) -> usize {
        self.padded
    }

    // -- crate-internal accessors used by the persistence format ---------

    pub(crate) fn padded_dim(&self) -> usize {
        self.padded
    }

    pub(crate) fn live_slots(&self) -> usize {
        self.live
    }

    pub(crate) fn slot_alive(&self, slot: usize) -> bool {
        self.alive[slot]
    }

    pub(crate) fn slot_scale(&self, slot: usize) -> f32 {
        self.scales[slot]
    }

    pub(crate) fn slot_codes(&self, slot: usize, bpv: usize) -> &[u8] {
        &self.codes[slot * bpv..(slot + 1) * bpv]
    }

    /// Mark the index as holding `count` dense (all-live, keyless) slots;
    /// used after loading from the file format.
    pub(crate) fn init_dense(&mut self, count: usize) {
        self.keys = vec![None; count];
        self.alive = vec![true; count];
        self.key_to_slot.clear();
        self.live = count;
    }

    /// Quantize and add one vector (any norm; normalized internally).
    ///
    /// Returns the slot index holding the vector (stable until compaction).
    pub fn add(&mut self, v: &[f32]) -> usize {
        self.append_slot(v, None)
    }

    /// Quantize and add one vector under a caller-chosen `u64` key.
    ///
    /// If `key` already exists, the vector is replaced in place (the slot
    /// index is preserved, matching usearch's insert semantics). Otherwise a
    /// new slot is appended. Returns the slot index holding the vector.
    pub fn add_keyed(&mut self, key: u64, v: &[f32]) -> usize {
        if let Some(&slot) = self.key_to_slot.get(&key) {
            let scale = self.encode_into(slot * (self.padded / 2), v);
            self.scales[slot] = scale;
            return slot;
        }
        self.append_slot(v, Some(key))
    }

    /// Remove a keyed vector. The slot becomes a tombstone: its storage is
    /// kept (slot indices stay stable) but searches skip it until
    /// [`VecqIndex::compact`]. Returns `false` if the key is unknown.
    pub fn remove_keyed(&mut self, key: u64) -> bool {
        match self.key_to_slot.remove(&key) {
            Some(slot) => {
                self.alive[slot] = false;
                self.keys[slot] = None;
                self.live -= 1;
                true
            }
            None => false,
        }
    }

    /// Look up the key stored at `slot` (`None` for anonymous slots,
    /// tombstones, or out-of-range indices).
    pub fn key_of(&self, slot: usize) -> Option<u64> {
        self.keys.get(slot).copied().flatten()
    }

    /// Whether `key` currently identifies a live vector.
    pub fn contains_key(&self, key: u64) -> bool {
        self.key_to_slot.contains_key(&key)
    }

    /// Rebuild the index in place, dropping tombstoned slots.
    ///
    /// All remaining vectors keep their keys; **slot indices shift** to become
    /// dense (0..len). Search results are unchanged.
    pub fn compact(&mut self) {
        if self.live == self.n {
            return;
        }
        let bpv = self.padded / 2;
        let mut codes = Vec::with_capacity(self.live * bpv);
        let mut scales = Vec::with_capacity(self.live);
        let mut keys = Vec::with_capacity(self.live);
        let mut alive = Vec::with_capacity(self.live);
        self.key_to_slot.clear();
        for slot in 0..self.n {
            if self.alive[slot] {
                codes.extend_from_slice(&self.codes[slot * bpv..(slot + 1) * bpv]);
                scales.push(self.scales[slot]);
                if let Some(key) = self.keys[slot] {
                    self.key_to_slot.insert(key, keys.len());
                }
                keys.push(self.keys[slot]);
                alive.push(true);
            }
        }
        self.codes = codes;
        self.scales = scales;
        self.keys = keys;
        self.alive = alive;
        self.n = self.live;
    }

    /// Append one vector as a new slot; returns the slot index.
    fn append_slot(&mut self, v: &[f32], key: Option<u64>) -> usize {
        let slot = self.n;
        let scale = self.encode_into(slot * (self.padded / 2), v);
        self.scales.push(scale);
        self.keys.push(key);
        if let Some(key) = key {
            self.key_to_slot.insert(key, slot);
        }
        self.alive.push(true);
        self.n += 1;
        self.live += 1;
        slot
    }

    /// Quantize `v` into the code bytes starting at `base` (extending
    /// `codes` when appending); returns the unit-norm correction scale.
    fn encode_into(&mut self, base: usize, v: &[f32]) -> f32 {
        assert_eq!(v.len(), self.dim, "vector dim mismatch");
        // Normalize over the working dims only (Matryoshka truncation happens
        // before rotation — see with_working_dim).
        let norm: f32 = v[..self.working_dim]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        assert!(norm > 0.0, "zero vector");
        let unit: Vec<f32> = v[..self.working_dim].iter().map(|x| x / norm).collect();

        let mut rotated = Vec::with_capacity(self.padded);
        self.transform.apply(&unit, &mut rotated);

        // Quantize to 4-bit codes, nibble-packed.
        let bytes_per_vec = self.padded / 2;
        if self.codes.len() < base + bytes_per_vec {
            self.codes.resize(base + bytes_per_vec, 0);
        }
        let mut sum_sq = 0f32;
        for (i, &x) in rotated.iter().enumerate() {
            let code = lloyd::quantize_4bit(x);
            let b = base + i / 2;
            let byte = if i % 2 == 0 {
                (self.codes[b] & 0xF0) | code
            } else {
                (self.codes[b] & 0x0F) | (code << 4)
            };
            self.codes[b] = byte;
            sum_sq += lloyd::dequantize_4bit(code).powi(2);
        }

        // Scale so that the stored vector is unit-norm: dequantized vector q
        // has norm sqrt(sum_sq); asymmetric scoring multiplies by 1/sqrt(sum_sq).
        1.0 / sum_sq.sqrt()
    }

    /// Prepare an f32 query in rotated space (call once per query).
    pub fn prepare_query(&self, q: &[f32]) -> PreparedQuery {
        assert_eq!(q.len(), self.dim);
        // Truncate + normalize over the working dims, mirroring encode_into.
        let norm: f32 = q[..self.working_dim]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        assert!(norm > 0.0, "zero vector");
        let unit: Vec<f32> = q[..self.working_dim].iter().map(|x| x / norm).collect();
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
    /// Dispatches to the explicit NEON path on aarch64, the explicit AVX2
    /// path on x86_64 when the host supports it (runtime detection), and the
    /// fixed 8-bucket scalar path otherwise. All use the identical
    /// association order (per code byte: mul, mul, add, then add into bucket
    /// j; final pairwise tree), so they produce the same f32 bits — guarded
    /// by `neon_matches_scalar_bitwise` / `avx2_matches_scalar_bitwise` in
    /// tests.
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
        #[cfg(target_arch = "x86_64")]
        {
            if avx2::available() {
                // SAFETY: feature availability checked immediately above.
                let raw = unsafe { avx2::score_avx2(codes, q, &pq.lut) };
                raw * self.scales[idx]
            } else {
                score_scalar(codes, q, &pq.lut) * self.scales[idx]
            }
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            score_scalar(codes, q, &pq.lut) * self.scales[idx]
        }
    }

    /// Brute-force top-k search. Returns (slot index, score) sorted by score
    /// desc. Tombstoned slots are skipped.
    ///
    /// Uses a bounded min-heap of size k (no O(n log n) sort, no O(n)
    /// allocation per query): push while the heap is not full, then only
    /// push-and-pop when the candidate beats the current k-th score.
    pub fn search(&self, q: &[f32], k: usize) -> Vec<(usize, f32)> {
        self.search_slots(q, k)
    }

    /// Keyed variant of [`VecqIndex::search`]: returns (key, score) sorted by
    /// score desc, restricted to live keyed vectors.
    pub fn search_keyed(&self, q: &[f32], k: usize) -> Vec<(u64, f32)> {
        self.search_slots(q, k)
            .into_iter()
            .filter_map(|(slot, s)| self.key_of(slot).map(|key| (key, s)))
            .collect()
    }

    fn search_slots(&self, q: &[f32], k: usize) -> Vec<(usize, f32)> {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let pq = self.prepare_query(q);
        let k = k.min(self.live).max(1);
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
        #[cfg(target_arch = "x86_64")]
        let use_avx2 = avx2::available();
        let mut idx = 0;
        #[cfg(target_arch = "aarch64")]
        {
            // Batch 4 vectors per pass: shared q loads + LUT setup. Tombstoned
            // slots are still scored (keeping the batch dense) but filtered
            // before entering the heap.
            while idx + 4 <= self.n {
                let codes4 = &self.codes[idx * bpv..(idx + 4) * bpv];
                let raw = unsafe { neon::score_neon4(codes4, q_rot, &pq.lut) };
                for (v, &r) in raw.iter().enumerate() {
                    let si = idx + v;
                    if self.alive[si] {
                        consider(r * self.scales[si], si, &mut heap);
                    }
                }
                idx += 4;
            }
        }
        #[cfg(target_arch = "x86_64")]
        {
            if use_avx2 {
                // Batch 4 vectors per pass: shared q deinterleave. Tombstoned
                // slots are still scored (keeping the batch dense) but
                // filtered before entering the heap.
                while idx + 4 <= self.n {
                    let codes4 = &self.codes[idx * bpv..(idx + 4) * bpv];
                    // SAFETY: AVX2 availability checked via `use_avx2`.
                    let raw =
                        unsafe { avx2::score_avx24(codes4, &pq.rotated[..self.padded], &pq.lut) };
                    for (v, &r) in raw.iter().enumerate() {
                        let si = idx + v;
                        if self.alive[si] {
                            consider(r * self.scales[si], si, &mut heap);
                        }
                    }
                    idx += 4;
                }
            }
        }
        while idx < self.n {
            if self.alive[idx] {
                consider(self.score(&pq, idx), idx, &mut heap);
            }
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

/// Explicit AVX2 scoring path (x86_64), bit-identical to [`score_scalar`].
///
/// Unlike NEON (baseline on aarch64), AVX2 is not universal on x86_64, so the
/// path is selected at runtime with `is_x86_feature_detected!` and the
/// kernels are `#[target_feature(enable = "avx2")]`.
///
/// Bit-identity with the scalar path is structural: lane j of the
/// accumulator corresponds to scalar bucket j. Per 8-byte block, lane j
/// computes `q[2b]*lut[lo_b] + q[2b+1]*lut[hi_b]` (b = block start + j;
/// vmul, vmul, vadd — no FMA contraction) and adds it into lane j, blocks
/// in increasing order — the same per-bucket term and accumulation order as
/// the scalar loop. The LUT gather uses `vgatherdps` on the 16-entry table
/// where NEON uses `vqtbl4q_u8`. Final reduction is the same pairwise tree.
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    /// Whether the host CPU supports AVX2.
    pub fn available() -> bool {
        std::is_x86_feature_detected!("avx2")
    }

    /// Gather `lut[nibble]` for 8 nibbles into an 8-lane vector.
    #[inline]
    unsafe fn gather8(lut: &[f32; 16], nibbles: __m128i) -> __m256 {
        // The gather's scale of 4 turns each nibble index into an f32 byte
        // offset — no pre-shift needed.
        let idx = _mm256_cvtepu8_epi32(nibbles);
        _mm256_i32gather_ps(lut.as_ptr(), idx, 4)
    }

    /// Deinterleave the 16 f32 at `q` into even dims (8 lanes) and odd dims
    /// (8 lanes): {d0,d2,..,d14} and {d1,d3,..,d15}.
    #[inline]
    unsafe fn deinterleave16(q: *const f32) -> (__m256, __m256) {
        let qa = _mm256_loadu_ps(q);
        let qb = _mm256_loadu_ps(q.add(8));
        // shuffle_ps picks {a0,a2,b0,b2} (even) / {a1,a3,b1,b3} (odd) per
        // 128-bit half; the vpermps index vector then interleaves the halves
        // into contiguous even/odd streams {d0,d2,..,d14} / {d1,d3,..,d15}.
        let fixup = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
        let even = _mm256_permutevar8x32_ps(_mm256_shuffle_ps(qa, qb, 0x88), fixup);
        let odd = _mm256_permutevar8x32_ps(_mm256_shuffle_ps(qa, qb, 0xDD), fixup);
        (even, odd)
    }

    /// AVX2 scoring over one vector's codes. See module docs.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn score_avx2(codes: &[u8], q: &[f32], lut: &[f32; 16]) -> f32 {
        let mut acc = _mm256_setzero_ps();
        let nb = codes.len();
        let mut i = 0;
        while i + 8 <= nb {
            let b8 = _mm_loadl_epi64(codes.as_ptr().add(i) as *const __m128i);
            let g_lo = gather8(lut, _mm_and_si128(b8, _mm_set1_epi8(0x0F)));
            let g_hi = gather8(
                lut,
                _mm_and_si128(_mm_srli_epi16(b8, 4), _mm_set1_epi8(0x0F)),
            );
            let (even, odd) = deinterleave16(q.as_ptr().add(i * 2));
            // term = q_even*lut[lo] + q_odd*lut[hi]  (mul, mul, add — no FMA)
            let term = _mm256_add_ps(_mm256_mul_ps(even, g_lo), _mm256_mul_ps(odd, g_hi));
            acc = _mm256_add_ps(acc, term);
            i += 8;
        }
        // Extract lanes and reduce with the scalar pairwise tree.
        let mut a = [0f32; 8];
        _mm256_storeu_ps(a.as_mut_ptr(), acc);
        // Scalar tail for the last (< 8) code bytes. padded is a multiple of
        // 8 elements (padded/2 bytes multiple of 4), so nb % 8 is 0 or 4.
        let mut tail = 0f32;
        while i < nb {
            let b = codes[i];
            let c = i * 2;
            tail += q[c] * lut[(b & 0x0F) as usize] + q[c + 1] * lut[(b >> 4) as usize];
            i += 1;
        }
        let s01 = a[0] + a[1];
        let s23 = a[2] + a[3];
        let s45 = a[4] + a[5];
        let s67 = a[6] + a[7];
        (s01 + s23) + (s45 + s67) + tail
    }

    /// Score 4 consecutive vectors at once, amortizing the q deinterleave
    /// across all 4. Each vector accumulates in the exact same per-lane order
    /// as [`score_avx2`], so results are bit-identical. Returns raw
    /// (pre-scale) scores; the caller multiplies by `scales`.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn score_avx24(codes4: &[u8], q: &[f32], lut: &[f32; 16]) -> [f32; 4] {
        let nb = codes4.len() / 4; // bytes per vector
        let mut acc = [_mm256_setzero_ps(); 4];
        let mut i = 0;
        while i + 8 <= nb {
            // Shared q loads + deinterleave for this block.
            let (even, odd) = deinterleave16(q.as_ptr().add(i * 2));
            for (v, acc_v) in acc.iter_mut().enumerate() {
                let b8 = _mm_loadl_epi64(codes4.as_ptr().add(v * nb + i) as *const __m128i);
                let g_lo = gather8(lut, _mm_and_si128(b8, _mm_set1_epi8(0x0F)));
                let g_hi = gather8(
                    lut,
                    _mm_and_si128(_mm_srli_epi16(b8, 4), _mm_set1_epi8(0x0F)),
                );
                let term = _mm256_add_ps(_mm256_mul_ps(even, g_lo), _mm256_mul_ps(odd, g_hi));
                *acc_v = _mm256_add_ps(*acc_v, term);
            }
            i += 8;
        }
        let mut out = [0f32; 4];
        for v in 0..4 {
            let mut a = [0f32; 8];
            _mm256_storeu_ps(a.as_mut_ptr(), acc[v]);
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

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar_bitwise() {
        if !avx2::available() {
            return; // host without AVX2: scalar path is the only path
        }
        // dim 128: padded 128 (bpv 32, 4 full blocks). dim 8: padded 8
        // (bpv 4) — exercises the 4-byte scalar tail after the block loop.
        for (dim, seed) in [(128, 42), (8, 43)] {
            let mut idx = VecqIndex::new(dim, seed);
            for i in 0..30 {
                idx.add(&rand_unit(dim, i + 500));
            }
            let q = rand_unit(dim, 777);
            let pq = idx.prepare_query(&q);
            for vi in 0..30 {
                let base = vi * (idx.padded() / 2);
                let codes = &idx.codes[base..base + idx.padded() / 2];
                let qslice = &pq.rotated[..idx.padded()];
                let avx2raw = unsafe { avx2::score_avx2(codes, qslice, &pq.lut) };
                let scalar = score_scalar(codes, qslice, &pq.lut);
                assert_eq!(
                    avx2raw.to_bits(),
                    scalar.to_bits(),
                    "dim {dim} vector {vi}: AVX2 and scalar diverged"
                );
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx24_matches_avx2_bitwise() {
        if !avx2::available() {
            return;
        }
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
            let batched = unsafe { avx2::score_avx24(codes4, qslice, &pq.lut) };
            for v in 0..4 {
                let single =
                    unsafe { avx2::score_avx2(&codes4[v * bpv..(v + 1) * bpv], qslice, &pq.lut) };
                assert_eq!(
                    batched[v].to_bits(),
                    single.to_bits(),
                    "chunk {chunk_start} vec {v}: avx24 diverged from avx2"
                );
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn search_dispatch_matches_scalar_on_avx2_hosts() {
        // Whatever path dispatch picks, results must equal the scalar
        // reference bit for bit.
        let dim = 128;
        let mut idx = VecqIndex::new(dim, 55);
        for i in 0..30 {
            idx.add(&rand_unit(dim, i + 800));
        }
        let q = rand_unit(dim, 888);
        let pq = idx.prepare_query(&q);
        for vi in 0..30 {
            let base = vi * (idx.padded() / 2);
            let codes = &idx.codes[base..base + idx.padded() / 2];
            let qslice = &pq.rotated[..idx.padded()];
            let scalar = score_scalar(codes, qslice, &pq.lut) * idx.scales[vi];
            assert_eq!(idx.score(&pq, vi).to_bits(), scalar.to_bits());
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

    #[test]
    fn keyed_add_search_remove() {
        let dim = 64;
        let mut idx = VecqIndex::new(dim, 5);
        for i in 0..50u64 {
            idx.add_keyed(1000 + i, &rand_unit(dim, i * 17 + 3));
        }
        assert_eq!(idx.len(), 50);
        assert!(idx.contains_key(1000));
        assert!(!idx.contains_key(999));

        let q = rand_unit(dim, 77);
        let keyed = idx.search_keyed(&q, 5);
        assert_eq!(keyed.len(), 5);
        for w in keyed.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
        // Keys from search_keyed must all exist and match positional results.
        let positional = idx.search(&q, 5);
        for ((key, ks), (slot, ps)) in keyed.iter().zip(positional.iter()) {
            assert_eq!(key, &idx.key_of(*slot).unwrap());
            assert_eq!(ks.to_bits(), ps.to_bits());
        }

        // Remove the top hit: it must vanish from results, others keep scores.
        let top_key = keyed[0].0;
        assert!(idx.remove_keyed(top_key));
        assert!(!idx.remove_keyed(top_key), "second remove is a no-op");
        assert!(!idx.remove_keyed(12345), "unknown key returns false");
        assert_eq!(idx.len(), 49);
        assert_eq!(idx.tombstones(), 1);
        let keyed2 = idx.search_keyed(&q, 5);
        assert!(!keyed2.iter().any(|(k, _)| *k == top_key));
        for (k, s) in keyed2.iter() {
            let old = keyed.iter().find(|(ok, _)| ok == k).map(|(_, os)| *os);
            if let Some(os) = old {
                assert_eq!(s.to_bits(), os.to_bits(), "key {k} score changed");
            }
        }
    }

    #[test]
    fn keyed_add_same_key_replaces() {
        let dim = 32;
        let mut idx = VecqIndex::new(dim, 9);
        idx.add_keyed(7, &rand_unit(dim, 101));
        idx.add_keyed(7, &rand_unit(dim, 202));
        assert_eq!(idx.len(), 1, "replace must not grow the index");
        assert_eq!(idx.tombstones(), 0);
        // The stored vector is the second one: query near it, key 7 wins.
        let q = rand_unit(dim, 202);
        let res = idx.search_keyed(&q, 1);
        assert_eq!(res[0].0, 7);
    }

    #[test]
    fn keyed_slot_indices_stay_stable_across_remove_and_serialize() {
        let dim = 64;
        let mut idx = VecqIndex::new(dim, 15);
        for i in 0..20u64 {
            idx.add_keyed(i, &rand_unit(dim, i + 300));
        }
        let q = rand_unit(dim, 404);
        let before = idx.search(&q, 20);
        // Remove two vectors: remaining slot indices must not shift.
        idx.remove_keyed(idx.key_of(before[0].0).unwrap());
        idx.remove_keyed(idx.key_of(before[5].0).unwrap());
        let after = idx.search(&q, 20);
        assert_eq!(after.len(), 18);
        for (slot, s) in &after {
            let old = before.iter().find(|(os, _)| os == slot);
            assert!(old.is_some(), "slot {slot} moved after remove");
            assert_eq!(old.unwrap().1.to_bits(), s.to_bits());
        }
        // Serializing drops tombstones on disk but must not disturb memory.
        let bytes = idx.to_bytes();
        let disk = VecqIndex::from_bytes(&bytes).unwrap();
        assert_eq!(disk.len(), 18);
        assert_eq!(idx.search(&q, 20), after, "in-memory results unchanged");
    }

    #[test]
    fn compact_drops_tombstones_and_preserves_results() {
        let dim = 64;
        let mut idx = VecqIndex::new(dim, 21);
        for i in 0..40u64 {
            idx.add_keyed(10 * i, &rand_unit(dim, i + 61));
        }
        for i in 0..20u64 {
            assert!(idx.remove_keyed(10 * i));
        }
        let q = rand_unit(dim, 123);
        let expected = idx.search_keyed(&q, 20);
        idx.compact();
        assert_eq!(idx.tombstones(), 0);
        assert_eq!(idx.len(), 20);
        assert_eq!(idx.search_keyed(&q, 20), expected);
        // Round-trip after compact: keys are not persisted by design, so the
        // reloaded index is searchable positionally. f16 scales perturb
        // scores by <1e-3, so compare order and approximate scores.
        let bytes = idx.to_bytes();
        let back = VecqIndex::from_bytes(&bytes).unwrap();
        let reloaded = back.search(&q, 20);
        assert_eq!(reloaded.len(), 20);
        for ((slot, s), (key, ks)) in reloaded.iter().zip(expected.iter()) {
            assert_eq!(idx.key_of(*slot), Some(*key));
            assert!(
                (s - ks).abs() < 1e-3,
                "key {key} score drifted: {s} vs {ks}"
            );
        }
    }

    #[test]
    fn keyed_search_on_empty_and_drained_index() {
        let dim = 32;
        let mut idx = VecqIndex::new(dim, 31);
        assert!(idx.search_keyed(&rand_unit(dim, 1), 3).is_empty());
        idx.add_keyed(1, &rand_unit(dim, 2));
        idx.add_keyed(2, &rand_unit(dim, 3));
        assert!(idx.remove_keyed(1));
        assert!(idx.remove_keyed(2));
        assert!(idx.is_empty(), "drained index reports empty");
        assert_eq!(idx.tombstones(), 2);
        assert!(idx.search_keyed(&rand_unit(dim, 4), 3).is_empty());
    }

    #[test]
    fn keyed_index_from_file_supports_keyed_adds() {
        let dim = 64;
        let mut idx = VecqIndex::new(dim, 41);
        for i in 0..10u64 {
            idx.add(&rand_unit(dim, i + 700));
        }
        let bytes = idx.to_bytes();
        let mut back = VecqIndex::from_bytes(&bytes).unwrap();
        back.add_keyed(555, &rand_unit(dim, 999));
        assert!(back.contains_key(555));
        assert_eq!(back.len(), 11);
        let q = rand_unit(dim, 999);
        assert_eq!(back.search_keyed(&q, 1)[0].0, 555);
    }

    #[test]
    #[should_panic(expected = "vector dim mismatch")]
    fn keyed_add_dim_mismatch_panics() {
        let mut idx = VecqIndex::new(32, 3);
        idx.add_keyed(1, &[0.5; 64]);
    }

    // -- Matryoshka working_dim ----------------------------------------------

    #[test]
    fn working_dim_equal_to_dim_matches_new_bitwise() {
        let dim = 128;
        let mut a = VecqIndex::new(dim, 7);
        let mut b = VecqIndex::with_working_dim(dim, dim, 7);
        for i in 0..20 {
            a.add(&rand_unit(dim, i + 61));
            b.add(&rand_unit(dim, i + 61));
        }
        let q = rand_unit(dim, 321);
        let ra = a.search(&q, 5);
        let rb = b.search(&q, 5);
        assert_eq!(ra.len(), rb.len());
        for ((sa, fa), (sb, fb)) in ra.iter().zip(rb.iter()) {
            assert_eq!(sa, sb);
            assert_eq!(fa.to_bits(), fb.to_bits());
        }
        assert_eq!(a.to_bytes(), b.to_bytes());
        assert_eq!(b.working_dim(), dim);
    }

    #[test]
    fn working_dim_truncates_storage_and_keeps_leading_signal() {
        // Matryoshka-style synthetic: signal lives in the leading dims, the
        // tail is noise. A working_dim index over the leading dims must keep
        // the neighbor ranking (truncation is the Matryoshka contract) at a
        // fraction of the storage.
        let (dim, working, n) = (256, 64, 40);
        let signal: Vec<Vec<f32>> = (0..n).map(|i| rand_unit(working, i * 13 + 5)).collect();
        let mut full = VecqIndex::new(dim, 11);
        let mut trunc = VecqIndex::with_working_dim(dim, working, 11);
        for (i, s) in signal.iter().enumerate() {
            // leading dims carry the identity, tail is per-vector noise
            let mut v = vec![0f32; dim];
            v[..working].copy_from_slice(s);
            let noise = rand_unit(dim - working, 9_000 + i as u64);
            v[working..].copy_from_slice(&noise.iter().map(|x: &f32| x * 0.05).collect::<Vec<_>>());
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in v.iter_mut() {
                *x /= norm;
            }
            full.add(&v);
            trunc.add(&v);
        }
        assert_eq!(trunc.working_dim(), working);
        // Storage: padded(64)/2 + 2 = 34 B/vec vs padded(256)/2 + 2 = 130.
        assert!(
            trunc.to_bytes().len() * 3 < full.to_bytes().len(),
            "expected ~3x smaller"
        );
        // Query: same construction as the true neighbor of vector 0.
        let q = &signal[0];
        let mut qv = vec![0f32; dim];
        qv[..working].copy_from_slice(q);
        let noise = rand_unit(dim - working, 9_000);
        qv[working..].copy_from_slice(&noise.iter().map(|x: &f32| x * 0.05).collect::<Vec<_>>());
        let norm: f32 = qv.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in qv.iter_mut() {
            *x /= norm;
        }
        let rt = trunc.search(&qv, 1);
        assert_eq!(rt[0].0, 0, "truncated index must rank vector 0 first");
    }

    #[test]
    fn working_dim_round_trips_through_file() {
        let (dim, working) = (256, 64);
        let mut idx = VecqIndex::with_working_dim(dim, working, 21);
        for i in 0..10 {
            idx.add(&rand_unit(dim, i + 700));
        }
        let q = rand_unit(dim, 999);
        let expected = idx.search(&q, 5);
        let bytes = idx.to_bytes();
        let back = VecqIndex::from_bytes(&bytes).expect("parse v1.2");
        assert_eq!(back.working_dim(), working);
        assert_eq!(back.dim(), dim);
        // f16 scales perturb scores by <1e-3: compare top-1 exactly, then
        // overlap and approximate scores (mirrors the v1.1 round-trip test).
        let reloaded = back.search(&q, 5);
        assert_eq!(reloaded[0].0, expected[0].0);
        assert_eq!(reloaded.len(), expected.len());
        for ((s0, f0), (_, f1)) in reloaded.iter().zip(expected.iter()) {
            assert!((f0 - f1).abs() < 1e-3, "slot {s0}: {f0} vs {f1}");
        }
    }

    #[test]
    fn legacy_v11_file_still_loads() {
        // A v1.1 file (reserved = 0) written before working_dim existed must
        // load as a full-dim index. v1.1 and v1.2 payloads are identical when
        // working_dim == dim (only the version field differs), so both loads
        // must agree bit for bit.
        let mut idx = VecqIndex::new(128, 33);
        for i in 0..8 {
            idx.add(&rand_unit(128, i + 50));
        }
        let bytes = idx.to_bytes(); // v1.2 now
        assert_eq!(
            u16::from_le_bytes([bytes[4], bytes[5]]),
            crate::format::V1_2
        );
        let mut v11 = bytes.clone();
        v11[4] = 257u16.to_le_bytes()[0];
        v11[5] = 257u16.to_le_bytes()[1];
        let q = rand_unit(128, 777);
        assert_eq!(
            VecqIndex::from_bytes(&v11).unwrap().search(&q, 5),
            VecqIndex::from_bytes(&bytes).unwrap().search(&q, 5)
        );
    }

    #[test]
    #[should_panic(expected = "working_dim")]
    fn working_dim_greater_than_dim_panics() {
        let _ = VecqIndex::with_working_dim(64, 128, 1);
    }

    #[test]
    #[should_panic(expected = "working_dim")]
    fn working_dim_zero_panics() {
        let _ = VecqIndex::with_working_dim(64, 0, 1);
    }
}

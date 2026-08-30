//! Head-to-head comparison of 4-bit-per-dim quantizers on identical data
//! (issue #28): vecq vs TurboQuant-MSE vs RaBitQ, against exact f32 ground
//! truth.
//!
//! Fairness notes:
//! - All engines score IP on unit vectors (== cosine ordering).
//! - TurboQuant is scanned ADC-style via a replicated codebook LUT; the codebook
//!   is deterministic (`generate_codebook(dim, 4, 100)` — the exact call
//!   TurboQuantMSE::new makes), verified at runtime against `dequantize`.
//! - RaBitQ trains with an internal rayon pool; timing is reported as-is.
//! - Bytes/vector are the per-vector code sizes; shared global structures
//!   (rotation matrices, codebooks) are excluded for every engine.

use std::time::Instant;

use rabitq_rs::brute_force::{BruteForceRabitqIndex, BruteForceSearchParams};
use rabitq_rs::{Metric, RotatorType};
use turboquant::codebook::generate_codebook;
use turboquant::turboquant_mse::TurboQuantMSE;
use turboquant::utils::normalize;
use vecq_core::VecqIndex;

/// Deterministic RNG (same family as the other bench bins).
struct Rng(u64);

impl Rng {
    fn normal(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        let u1 = ((self.0 >> 11) as f64 / (1u64 << 53) as f64).max(1e-12);
        self.0 ^= self.0 << 16;
        let u2 = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    fn unit(&mut self, dim: usize) -> Vec<f32> {
        let mut v: Vec<f32> = (0..dim).map(|_| self.normal() as f32).collect();
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter_mut().for_each(|x| *x /= n);
        v
    }
}

/// Clustered dataset mimicking real embedding structure.
fn clustered(n: usize, dim: usize, clusters: usize, seed: u64, spread: f32) -> Vec<Vec<f32>> {
    let mut rng = Rng(seed | 1);
    let centroids: Vec<Vec<f32>> = (0..clusters).map(|_| rng.unit(dim)).collect();
    (0..n)
        .map(|i| {
            let c = &centroids[i % clusters];
            let mut v: Vec<f32> = c
                .iter()
                .map(|&x| x + spread * rng.normal() as f32)
                .collect();
            let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.iter_mut().for_each(|x| *x /= n);
            v
        })
        .collect()
}

fn exact_top_k(base: &[Vec<f32>], q: &[f32], k: usize) -> Vec<usize> {
    let mut scores: Vec<(usize, f32)> = base
        .iter()
        .enumerate()
        .map(|(i, v)| (i, vecq_core::cosine_f32(q, v)))
        .collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scores.into_iter().take(k).map(|(i, _)| i).collect()
}

fn recall_at_k(truth: &[usize], got: &[usize]) -> f32 {
    truth.iter().filter(|t| got.contains(t)).count() as f32 / truth.len() as f32
}

struct EngineResult {
    name: String,
    bytes_per_vec: f64,
    build_ms: f64,
    ms_per_query: f64,
    recall10: f32,
}

fn bench_vecq(base: &[Vec<f32>], queries: &[Vec<f32>], k: usize) -> EngineResult {
    let dim = base[0].len();
    let t0 = Instant::now();
    let mut idx = VecqIndex::new(dim, 42);
    for v in base {
        idx.add(v);
    }
    let build = t0.elapsed().as_secs_f64() * 1000.0;
    let t0 = Instant::now();
    for q in queries {
        // Timing loop: search only, no ground-truth work.
        let _ = idx.search(q, k);
    }
    let query_ms = t0.elapsed().as_secs_f64() * 1000.0 / queries.len() as f64;
    let mut recalls = Vec::new();
    for q in queries {
        let truth = exact_top_k(base, q, k);
        let got: Vec<usize> = idx.search(q, k).into_iter().map(|(i, _)| i).collect();
        recalls.push(recall_at_k(&truth, &got));
    }
    let padded = dim.max(1).next_power_of_two();
    EngineResult {
        name: "vecq 4-bit".into(),
        bytes_per_vec: (padded / 2 + 2) as f64,
        build_ms: build,
        ms_per_query: query_ms,
        recall10: recalls.iter().sum::<f32>() / recalls.len() as f32,
    }
}

fn bench_turboquant(base: &[Vec<f32>], queries: &[Vec<f32>], k: usize) -> Option<EngineResult> {
    let dim = base[0].len();
    let t0 = Instant::now();
    // Build the quantizer around OUR codebook instance so the scan LUT is
    // guaranteed identical to the encoder's (generate_codebook is a pure
    // function of (dim, bits, iterations)).
    let cb = generate_codebook(dim, 4, 100).ok()?;
    let tq = TurboQuantMSE::with_codebook(dim, cb.clone(), 42).ok()?;
    let encoded: Vec<Vec<u8>> = base
        .iter()
        .map(|v| {
            let x: Vec<f64> = v.iter().map(|&x| x as f64).collect();
            let x = normalize(&x).expect("unit vector");
            tq.quantize(&x).expect("quantize").indices
        })
        .collect();
    let build = t0.elapsed().as_secs_f64() * 1000.0;
    // SDC product table: score = sum_j prod[q_idx[j]][db_idx[j]].
    let mut prod = [[0f32; 16]; 16];
    for (a, row) in prod.iter_mut().enumerate() {
        for (b, cell) in row.iter_mut().enumerate() {
            *cell = (cb.dequantize_scalar(a as u8) * cb.dequantize_scalar(b as u8)) as f32;
        }
    }
    let t0 = Instant::now();
    let mut recalls = Vec::new();
    for q in queries {
        // Symmetric distance computation: quantize the query with the same
        // codebook, then LUT-scan the database codes.
        let qx: Vec<f64> = q.iter().map(|&x| x as f64).collect();
        let qn = normalize(&qx).expect("unit vector");
        let qp = tq.quantize(&qn).expect("quantize");
        let q_idx = &qp.indices;
        let mut scored: Vec<(usize, f32)> = encoded
            .iter()
            .enumerate()
            .map(|(i, codes)| {
                let mut s = 0f32;
                for (j, &c) in codes.iter().enumerate() {
                    s += prod[q_idx[j] as usize][c as usize];
                }
                (i, s)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let got: Vec<usize> = scored.into_iter().take(k).map(|(i, _)| i).collect();
        let truth = exact_top_k(base, q, k);
        recalls.push(recall_at_k(&truth, &got));
    }
    let query_ms = t0.elapsed().as_secs_f64() * 1000.0 / queries.len() as f64;
    Some(EngineResult {
        name: "TurboQuant-MSE 4-bit (SDC)".into(),
        bytes_per_vec: dim as f64 / 2.0,
        build_ms: build,
        ms_per_query: query_ms,
        recall10: recalls.iter().sum::<f32>() / recalls.len() as f32,
    })
}

fn bench_rabitq(base: &[Vec<f32>], queries: &[Vec<f32>], k: usize) -> Option<EngineResult> {
    let t0 = Instant::now();
    let index = BruteForceRabitqIndex::train(
        base,
        4,
        Metric::InnerProduct,
        RotatorType::FhtKacRotator,
        42,
        true,
    )
    .ok()?;
    let build = t0.elapsed().as_secs_f64() * 1000.0;
    let t0 = Instant::now();
    let mut recalls = Vec::new();
    for q in queries {
        let got = index
            .search(q, BruteForceSearchParams::new(k))
            .expect("search");
        let got: Vec<usize> = got.into_iter().map(|r| r.id).collect();
        let truth = exact_top_k(base, q, k);
        recalls.push(recall_at_k(&truth, &got));
    }
    let query_ms = t0.elapsed().as_secs_f64() * 1000.0 / queries.len() as f64;
    let dim = base[0].len();
    // 4 bits/dim packed + per-vector norms (v_norm + ipnorm inverse, f32 each).
    Some(EngineResult {
        name: "RaBitQ 4-bit brute (FHT-Kac)".into(),
        bytes_per_vec: dim as f64 / 2.0 + 8.0,
        build_ms: build,
        ms_per_query: query_ms,
        recall10: recalls.iter().sum::<f32>() / recalls.len() as f32,
    })
}

fn bench_f32(base: &[Vec<f32>], queries: &[Vec<f32>], k: usize) -> EngineResult {
    let t0 = Instant::now();
    let mut recalls = Vec::new();
    for q in queries {
        let truth = exact_top_k(base, q, k);
        recalls.push(recall_at_k(&truth, &truth));
    }
    EngineResult {
        name: "f32 brute (reference)".into(),
        bytes_per_vec: (base[0].len() * 4) as f64,
        build_ms: 0.0,
        ms_per_query: t0.elapsed().as_secs_f64() * 1000.0 / queries.len() as f64,
        recall10: recalls.iter().sum::<f32>() / recalls.len() as f32,
    }
}

fn run_dataset(n: usize, dim: usize) {
    println!("\n## n={n}, dim={dim}, 200 queries, k=10, clustered (spread 0.5)\n");
    println!("| engine | bytes/vec | build ms | ms/query | recall@10 |");
    println!("|---|---|---|---|---|");
    let base = clustered(n, dim, 200, 42, 0.5);
    let queries = clustered(200, dim, 200, 777, 0.5);
    let f32r = bench_f32(&base, &queries, 10);
    println!(
        "| {} | {} | — | {:.3} | {:.3} (ref) |",
        f32r.name, f32r.bytes_per_vec, f32r.ms_per_query, f32r.recall10
    );
    let vq = bench_vecq(&base, &queries, 10);
    println!(
        "| {} | {} | {:.0} | {:.3} | {:.3} |",
        vq.name, vq.bytes_per_vec, vq.build_ms, vq.ms_per_query, vq.recall10
    );
    if let Some(tq) = bench_turboquant(&base, &queries, 10) {
        println!(
            "| {} | {} | {:.0} | {:.3} | {:.3} |",
            tq.name, tq.bytes_per_vec, tq.build_ms, tq.ms_per_query, tq.recall10
        );
    }
    if let Some(rq) = bench_rabitq(&base, &queries, 10) {
        println!(
            "| {} | {} | {:.0} | {:.3} | {:.3} |",
            rq.name, rq.bytes_per_vec, rq.build_ms, rq.ms_per_query, rq.recall10
        );
    }
}

fn main() {
    println!("# vecq vs quantizer head-to-head (single-threaded, release)");
    run_dataset(10_000, 768);
    run_dataset(10_000, 384);
    run_dataset(1_000, 384);
}

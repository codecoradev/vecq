use std::time::Instant;

use vecq_core::VecqIndex;

/// xorshift64* PRNG — deterministic datasets, no external deps.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0 | 1;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    fn unit_vec(&mut self, dim: usize) -> Vec<f32> {
        let v: Vec<f32> = (0..dim).map(|_| self.normal() as f32).collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.into_iter().map(|x| x / norm).collect()
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
    }
    dot // vectors are unit-norm already
}

fn brute_topk(base: &[Vec<f32>], q: &[f32], k: usize) -> Vec<usize> {
    let mut scored: Vec<(f32, usize)> = base
        .iter()
        .enumerate()
        .map(|(i, v)| (cosine(q, v), i))
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    scored.into_iter().take(k).map(|(_, i)| i).collect()
}

fn recall(n: usize, dim: usize, k: usize, seed: u64, perturb: f64) -> (f32, f32) {
    let mut rng = Rng(seed);
    let base: Vec<Vec<f32>> = (0..n).map(|_| rng.unit_vec(dim)).collect();
    // Queries: perturbed copies of base vectors (realistic near-duplicates).
    let queries: Vec<Vec<f32>> = (0..100)
        .map(|i| {
            let src = &base[(i * 7919) % n];
            let noise: Vec<f32> = src
                .iter()
                .map(|&x| x + perturb as f32 * rng.normal() as f32)
                .collect();
            let norm: f32 = noise.iter().map(|x| x * x).sum::<f32>().sqrt();
            noise.iter().map(|x| x / norm).collect()
        })
        .collect();

    let mut index = VecqIndex::new(dim, seed);
    let t0 = Instant::now();
    for v in &base {
        index.add(v);
    }
    let build_ms = t0.elapsed().as_millis();

    let mut hits = 0usize;
    let t1 = Instant::now();
    for q in &queries {
        let est = index.search(q, k);
        let truth = brute_topk(&base, q, k);
        for t in &truth {
            if est.iter().any(|(i, _)| i == t) {
                hits += 1;
            }
        }
    }
    let search_ms = t1.elapsed().as_millis();
    let recall = hits as f32 / (queries.len() * k) as f32;
    println!(
        "n={n} dim={dim} k={k} perturb={perturb}: recall@{k}={recall:.3} build={build_ms}ms search={search_ms}ms ({} queries, {:.2}ms/q)",
        queries.len(),
        search_ms as f64 / queries.len() as f64
    );
    (recall, build_ms as f32)
}

/// Clustered dataset: `c` cluster centers, each vector = center + noise.
/// Mimics real embedding structure (topics), where top-10 spans a meaningful
/// similarity range instead of pure orthogonal noise.
fn recall_clustered(
    n: usize,
    dim: usize,
    k: usize,
    seed: u64,
    spread: f64,
    clusters: usize,
) -> f32 {
    let mut rng = Rng(seed);
    let centers: Vec<Vec<f32>> = (0..clusters).map(|_| rng.unit_vec(dim)).collect();
    let mk = |rng: &mut Rng, ci: usize| -> Vec<f32> {
        let c = &centers[ci];
        let v: Vec<f32> = c
            .iter()
            .map(|&x| x + spread as f32 * rng.normal() as f32)
            .collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.into_iter().map(|x| x / norm).collect()
    };
    let base: Vec<Vec<f32>> = (0..n).map(|i| mk(&mut rng, i % clusters)).collect();
    let queries: Vec<Vec<f32>> = (0..100)
        .map(|i| mk(&mut rng, (i * 37) % clusters))
        .collect();

    let mut index = VecqIndex::new(dim, seed);
    for v in &base {
        index.add(v);
    }

    let mut hits = 0usize;
    for q in &queries {
        let est = index.search(q, k);
        let truth = brute_topk(&base, q, k);
        for t in &truth {
            if est.iter().any(|(i, _)| i == t) {
                hits += 1;
            }
        }
    }
    let recall = hits as f32 / (queries.len() * k) as f32;
    println!("n={n} dim={dim} k={k} clusters={clusters} spread={spread}: recall@{k}={recall:.3}");
    recall
}

fn main() {
    println!("== vecq recall benchmark vs f32 brute-force ground truth ==\n");
    println!("-- clustered (realistic embedding structure) --");
    for &spread in &[0.3, 0.5, 0.7] {
        recall_clustered(1_000, 384, 10, 42, spread, 50);
    }
    recall_clustered(10_000, 384, 10, 42, 0.5, 200);
    recall_clustered(10_000, 768, 10, 42, 0.5, 200);
    println!("\n-- adversarial: orthogonal random vectors (worst case) --");
    recall(1_000, 384, 10, 42, 0.3);

    // Score error sanity: max |est - exact| on 200 vectors
    let mut rng = Rng(7);
    let base: Vec<Vec<f32>> = (0..200).map(|_| rng.unit_vec(128)).collect();
    let mut index = VecqIndex::new(128, 7);
    for v in &base {
        index.add(v);
    }
    let q = rng.unit_vec(128);
    let pq = index.prepare_query(&q);
    let mut max_err = 0f32;
    for (i, v) in base.iter().enumerate() {
        let est = index.score(&pq, i);
        let exact = cosine(&q, v);
        max_err = max_err.max((est - exact).abs());
    }
    println!("\nmax |score error| (dim=128, n=200): {max_err:.4}");
    println!(
        "\nmemory: 4-bit codes = {} bytes/vec vs f32 = {} bytes/vec ({}x reduction)",
        512 / 2 + 4,
        384 * 4,
        (384 * 4) as f32 / (512f32 / 2.0 + 4.0)
    );
}

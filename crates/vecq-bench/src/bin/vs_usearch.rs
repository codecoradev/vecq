//! vecq vs usearch: throughput + memory + recall on the real-embedding dataset.

use std::fs;
use std::time::Instant;

use usearch::ffi::{IndexOptions, MetricKind, ScalarKind};
use usearch::Index;
use vecq_core::VecqIndex;

fn load_f32(path: &str, n: usize, dim: usize) -> Vec<Vec<f32>> {
    let bytes = fs::read(path).expect("read");
    (0..n)
        .map(|i| {
            (0..dim)
                .map(|j| {
                    f32::from_le_bytes(
                        bytes[(i * dim + j) * 4..(i * dim + j) * 4 + 4]
                            .try_into()
                            .unwrap(),
                    )
                })
                .collect()
        })
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn main() {
    let dir = "/tmp/vecq-bench";
    let (nb, nq, dim) = (2000usize, 100usize, 768usize);
    let base = load_f32(&format!("{dir}/base.f32"), nb, dim);
    let queries = load_f32(&format!("{dir}/queries.f32"), nq, dim);

    // ---- ground truth ----
    let truth: Vec<Vec<usize>> = queries
        .iter()
        .map(|q| {
            let mut s: Vec<(f32, usize)> = base
                .iter()
                .enumerate()
                .map(|(i, v)| (cosine(q, v), i))
                .collect();
            s.sort_by(|a, b| b.0.total_cmp(&a.0));
            s.truncate(10);
            s.into_iter().map(|(_, i)| i).collect()
        })
        .collect();

    // ---- vecq ----
    let t = Instant::now();
    let mut vq = VecqIndex::new(dim, 42);
    for v in &base {
        vq.add(v);
    }
    let vq_build = t.elapsed();
    let vq_bytes = vq.to_bytes().len();
    let t = Instant::now();
    let mut vq_hits = 0;
    for (qi, q) in queries.iter().enumerate() {
        for r in vq.search(q, 10) {
            if truth[qi].contains(&r.0) {
                vq_hits += 1;
            }
        }
    }
    let vq_search = t.elapsed();
    let vq_recall = vq_hits as f32 / (nq * 10) as f32;

    // ---- usearch f32 ----
    let opts = IndexOptions {
        dimensions: dim,
        metric: MetricKind::Cos,
        quantization: ScalarKind::F32,
        ..Default::default()
    };
    let us = Index::new(&opts).expect("create index");
    let t = Instant::now();
    us.reserve(nb).expect("reserve");
    for (i, v) in base.iter().enumerate() {
        us.add(i as u64, v.as_slice()).expect("add");
    }
    let us_build = t.elapsed();
    let us_bytes = nb * dim * 4;
    let t = Instant::now();
    let mut us_hits = 0;
    for (qi, q) in queries.iter().enumerate() {
        let res = us.search(q.as_slice(), 10).expect("search");
        for k in &res.keys {
            if truth[qi].contains(&(*k as usize)) {
                us_hits += 1;
            }
        }
    }
    let us_search = t.elapsed();
    let us_recall = us_hits as f32 / (nq * 10) as f32;

    println!("| engine | build | search (100 q) | ms/q | recall@10 | bytes/vector |");
    println!("|---|---|---|---|---|---|");
    println!(
        "| vecq 4-bit | {:.0} ms | {:.0} ms | {:.2} | {:.3} | {} |",
        vq_build.as_millis(),
        vq_search.as_millis(),
        vq_search.as_millis() as f64 / nq as f64,
        vq_recall,
        vq_bytes / nb
    );
    println!(
        "| usearch f32 | {:.0} ms | {:.0} ms | {:.2} | {:.3} | {} |",
        us_build.as_millis(),
        us_search.as_millis(),
        us_search.as_millis() as f64 / nq as f64,
        us_recall,
        us_bytes / nb
    );
}

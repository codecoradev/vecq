//! End-to-end verification for issue #39: build real-EmbeddingGemma
//! indexes at 4/5/6-bit widths plus residual, measure bytes/vector,
//! recall@1/@10 and scan latency. Mirrors the diagnostic numbers that
//! motivated the width option.

use std::fs;
use std::time::Instant;

use vecq_core::VecqIndex;

fn load_f32(path: &str, n: usize, dim: usize) -> Vec<Vec<f32>> {
    let bytes = fs::read(path).expect("read file");
    assert_eq!(bytes.len(), n * dim * 4, "file size mismatch");
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
    a.iter().zip(b).map(|x| x.0 * x.1).sum()
}

fn main() {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/vecq-bench".to_string());
    let meta = fs::read_to_string(format!("{dir}/meta.json")).expect("meta.json");
    let get = |k: &str| -> usize {
        let i = meta.find(&format!("\"{k}\"")).expect(k) + k.len() + 4;
        let rest = &meta[i..];
        let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap();
        rest[..end].parse().unwrap()
    };
    let (nb, nq, dim) = (get("n_base"), get("n_query"), get("dim"));
    println!("dataset: n={nb} queries={nq} dim={dim}");
    let base = load_f32(&format!("{dir}/base.f32"), nb, dim);
    let queries = load_f32(&format!("{dir}/queries.f32"), nq, dim);

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

    for (label, bits, residual) in [
        ("plain 4-bit ", 4u8, false),
        ("plain 5-bit ", 5, false),
        ("plain 6-bit ", 6, false),
        ("4-bit+resid ", 4, true),
    ] {
        let mut idx = if residual {
            VecqIndex::with_residual(dim, 42)
        } else {
            let mut i = VecqIndex::new(dim, 42);
            i.set_bits(bits);
            i
        };
        let t0 = Instant::now();
        for v in &base {
            idx.add(v);
        }
        let build = t0.elapsed();

        // Serialized size per vector (the on-disk number users care about).
        let bytes = idx.to_bytes();
        let per_vec = bytes.len() as f32 / idx.len() as f32;

        let t1 = Instant::now();
        let mut hits1 = 0usize;
        let mut hits10 = 0usize;
        for (qi, q) in queries.iter().enumerate() {
            let got = idx.search(q, 10);
            let g: Vec<usize> = got.iter().map(|(i, _)| *i).collect();
            if g[0] == truth[qi][0] {
                hits1 += 1;
            }
            hits10 += truth[qi].iter().filter(|t| g.contains(t)).count();
        }
        let dt = t1.elapsed();
        let ms_q = dt.as_secs_f64() * 1e3 / nq as f64;

        println!(
            "{label}: build {:>6.1?} | {:.0} B/vec | r@1 {:.3} | r@10 {:.3} | {:.2} ms/q",
            build,
            per_vec,
            hits1 as f32 / nq as f32,
            hits10 as f32 / (nq * 10) as f32,
            ms_q
        );
    }
}

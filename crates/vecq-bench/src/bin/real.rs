use std::fs;
use std::time::Instant;

use vecq_core::VecqIndex;

/// Load [n][dim] f32 raw file.
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
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).map(String::as_str).unwrap_or("/tmp/vecq-bench");
    let meta = fs::read_to_string(format!("{dir}/meta.json")).expect("meta.json");
    // Poor man's JSON parse (no deps).
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

    // Ground truth: exact f32 brute force.
    let t0 = Instant::now();
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
    println!(
        "f32 brute force: {:.2?} total ({:.2} ms/q)",
        t0.elapsed(),
        t0.elapsed().as_millis() as f64 / nq as f64
    );

    // vecq quantized.
    let t1 = Instant::now();
    let mut index = VecqIndex::new(dim, 42);
    for v in &base {
        index.add(v);
    }
    let build = t1.elapsed();
    let t2 = Instant::now();
    let mut hits10 = 0;
    let mut hits1 = 0;
    for (qi, q) in queries.iter().enumerate() {
        let res = index.search(q, 10);
        if res[0].0 == truth[qi][0] {
            hits1 += 1;
        }
        for t in &truth[qi] {
            if res.iter().any(|(i, _)| i == t) {
                hits10 += 1;
            }
        }
    }
    let search = t2.elapsed();
    println!(
        "vecq: build={build:.2?} search={search:.2?} ({:.2} ms/q)",
        search.as_millis() as f64 / nq as f64
    );
    println!("recall@1 = {:.3}", hits1 as f32 / nq as f32);
    println!("recall@10 = {:.3}", hits10 as f32 / (nq * 10) as f32);

    // Memory comparison.
    let quant_bytes = index.to_bytes().len();
    let f32_bytes = nb * dim * 4;
    println!(
        "storage: vecq={} bytes ({:.1} MB), f32={} bytes ({:.1} MB) -> {:.2}x reduction",
        quant_bytes,
        quant_bytes as f64 / 1e6,
        f32_bytes,
        f32_bytes as f64 / 1e6,
        f32_bytes as f64 / quant_bytes as f64
    );
}

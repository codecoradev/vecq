//! Diagnostic: does in-memory scoring (f32 scales) differ from the persisted
//! artifact (f16 scales, reloaded from to_bytes)?
use std::fs;

use vecq_core::VecqIndex;

fn load_f32(path: &str, n: usize, dim: usize) -> Vec<Vec<f32>> {
    let bytes = fs::read(path).expect("read file");
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

fn meta_get(meta: &str, k: &str) -> usize {
    let i = meta.find(&format!("\"{k}\"")).expect(k) + k.len() + 4;
    let rest = &meta[i..];
    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap();
    rest[..end].parse().unwrap()
}

fn main() {
    let dir = "/tmp/vecq-bench";
    let meta = fs::read_to_string(format!("{dir}/meta.json")).unwrap();
    let (nb, nq, dim) = (
        meta_get(&meta, "n_base"),
        meta_get(&meta, "n_query"),
        meta_get(&meta, "dim"),
    );
    let base = load_f32(&format!("{dir}/base.f32"), nb, dim);
    let queries = load_f32(&format!("{dir}/queries.f32"), nq, dim);

    for &bits in &[5u8, 4] {
        let mut idx = VecqIndex::new(dim, 42);
        if bits != 5 {
            idx.set_bits(bits);
        }
        for v in &base {
            idx.add(v);
        }
        let bytes = idx.to_bytes();
        let reloaded = VecqIndex::from_bytes(&bytes).expect("reload");

        let mut score_diffs = 0usize;
        let mut max_delta = 0.0f32;
        let mut list_diffs = 0usize;
        for q in &queries {
            let a = idx.search(q, 10);
            let b = reloaded.search(q, 10);
            if a != b {
                list_diffs += 1;
            }
            // compare scores for the ids returned by the in-memory search
            for (id, sa) in &a {
                let sb = b.iter().find(|(i, _)| i == id).map(|(_, s)| *s);
                if let Some(sb) = sb {
                    let d = (sa - sb).abs();
                    if d > 0.0 {
                        score_diffs += 1;
                        max_delta = max_delta.max(d);
                    }
                }
            }
        }
        println!(
            "bits={bits}: top10 list diffs in-memory vs reloaded = {list_diffs}/{nq}, nonzero score deltas = {score_diffs}, max |delta| = {max_delta:.2e}"
        );
    }
}

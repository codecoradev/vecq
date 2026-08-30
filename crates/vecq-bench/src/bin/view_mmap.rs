//! mmap cold-start harness for #25: time-to-first-query for a full load
//! (`VecqIndex::from_bytes`) vs a zero-copy view (`VecqView` over
//! `memmap2::Mmap`) at growing collection sizes.
//!
//! Run after generating the dataset (see BENCHMARK.md), e.g.:
//!   cargo run --release -p vecq-bench --bin view_mmap -- 12000
//! The optional argument scales the real dataset by repetition so the
//! load-vs-view gap is visible at 10k+ vectors.

use std::fs;
use std::io::Write;
use std::time::Instant;

use memmap2::Mmap;
use vecq_core::{VecqIndex, VecqView};

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

fn main() {
    let target_n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);
    let dir = "/tmp/vecq-bench";
    let meta = fs::read_to_string(format!("{dir}/meta.json")).expect("meta.json");
    let get = |k: &str| -> usize {
        let i = meta.find(&format!("\"{k}\"")).expect(k) + k.len() + 4;
        let rest = &meta[i..];
        let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap();
        rest[..end].parse().unwrap()
    };
    let (nb, nq, dim) = (get("n_base"), get("n_query"), get("dim"));
    let reps = target_n.div_ceil(nb);
    let n = nb * reps;
    println!("dataset: n={n} ({reps}x{nb}) queries={nq} dim={dim}");

    let base = load_f32(&format!("{dir}/base.f32"), nb, dim);
    let queries = load_f32(&format!("{dir}/queries.f32"), nq, dim);

    // Build a 5-bit index of n vectors and persist it.
    let t0 = Instant::now();
    let mut idx = VecqIndex::new(dim, 42);
    for r in 0..reps {
        for v in &base {
            idx.add(v);
        }
        let _ = r;
    }
    let build = t0.elapsed();
    let bytes = idx.to_bytes();
    let file_path = format!("{dir}/index_{n}.vecq");
    let mut f = fs::File::create(&file_path).expect("create file");
    f.write_all(&bytes).expect("write file");
    drop(f);
    println!(
        "built {:?}, file {:.1} MB ({} B/vec)",
        build,
        bytes.len() as f64 / 1e6,
        bytes.len() / n
    );

    // Full load: read + parse + copy payloads into heap structures.
    let t1 = Instant::now();
    let raw = std::fs::read(&file_path).expect("read");
    let full = VecqIndex::from_bytes(&raw).expect("parse");
    let full_load = t1.elapsed();

    // mmap view: map (lazy) + header parse only.
    let t2 = Instant::now();
    let file = fs::File::open(&file_path).expect("open");
    let map = unsafe { Mmap::map(&file).expect("mmap") };
    let view = VecqView::from_bytes(&map).expect("parse view");
    let view_ready = t2.elapsed();

    // Correctness cross-check on a few queries, then time-to-first-query.
    let q0 = &queries[0];
    let a = full.search(q0, 10);
    let b = view.search(q0, 10);
    for ((sa, fa), (sb, fb)) in a.iter().zip(b.iter()) {
        assert_eq!(sa, sb);
        assert_eq!(fa.to_bits(), fb.to_bits());
    }

    let t3 = Instant::now();
    let first = view.search(&queries[0], 10);
    let view_first = t3.elapsed();
    let t4 = Instant::now();
    let firstf = full.search(&queries[0], 10);
    let full_first = t4.elapsed();
    assert_eq!(first.len(), firstf.len());

    println!("full load (read+parse+copy): {:?}", full_load);
    println!("mmap view ready (map+parse):  {:?}", view_ready);
    println!("first query through view:     {:?}", view_first);
    println!("first query through full:     {:?}", full_first);
    println!("warm comparison (100 queries):");
    let t5 = Instant::now();
    for q in &queries {
        let _ = view.search(q, 10);
    }
    let vw = t5.elapsed();
    let t6 = Instant::now();
    for q in &queries {
        let _ = full.search(q, 10);
    }
    let fw = t6.elapsed();
    println!(
        "  view {:?} ({:.2} ms/q) | full {:?} ({:.2} ms/q)",
        vw,
        vw.as_secs_f64() * 1e3 / nq as f64,
        fw,
        fw.as_secs_f64() * 1e3 / nq as f64
    );
}

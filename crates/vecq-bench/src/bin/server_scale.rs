//! Server-scale benchmark profile for #52: 100K real EmbeddingGemma vectors
//! (768-dim, seeded), measuring plain / working_dim-256 modes through a
//! zero-copy `VecqView` over mmap, plus the 4-bit cascade (#22) rescore on
//! the full index, against exact f32 cosine ground truth — the same
//! methodology as the `real` harness, so the server and edge tables share
//! one ground-truth pipeline.
//!
//! Generate the dataset first (see docs/BENCHMARK.md "Server scale"):
//!   python3 scripts/gen_dataset.py --n-base 100000 --n-query 200 \
//!       --out /tmp/vecq-bench-100k
//!   cargo run --release -p vecq-bench --bin server_scale
//!
//! Options: `--dir <path>` overrides the dataset dir, `--skip-gt` skips the
//! exact ground-truth pass (timing-only rerun; recall columns are omitted).
//!
//! Everything here is single-threaded by design — the parallel scan path
//! (#51) will extend this harness with a threads dimension. Recall values
//! are deterministic for a given dataset; timings are host-specific.

use std::fs;
use std::io::Write;
use std::time::{Duration, Instant};

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

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// One result row of the final table.
struct Row {
    mode: String,
    build: Option<Duration>,
    file_bytes: usize,
    ms_q: f64,
    recall1: Option<f32>,
    recall10: Option<f32>,
    note: &'static str,
}

fn meta_get(meta: &str, k: &str) -> usize {
    let i = meta.find(&format!("\"{k}\"")).expect(k) + k.len() + 4;
    let rest = &meta[i..];
    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap();
    rest[..end].parse().unwrap()
}

/// Build an index over `base` with the given width / working dim.
fn build_index(
    dim: usize,
    working_dim: usize,
    bits: u8,
    base: &[Vec<f32>],
) -> (VecqIndex, Duration) {
    let t0 = Instant::now();
    let mut idx = if working_dim == dim {
        VecqIndex::new(dim, 42)
    } else {
        VecqIndex::with_working_dim(dim, working_dim, 42)
    };
    if bits != 5 {
        idx.set_bits(bits);
    }
    for v in base {
        idx.add(v);
    }
    (idx, t0.elapsed())
}

fn write_index(dir: &str, name: &str, idx: &VecqIndex) -> (String, usize) {
    let bytes = idx.to_bytes();
    let path = format!("{dir}/{name}.vecq");
    let mut f = fs::File::create(&path).expect("create file");
    f.write_all(&bytes).expect("write file");
    (path, bytes.len())
}

/// Time `queries` against an mmap'd view, scoring recall against `gt`
/// (exact top-10 index lists, `None` with --skip-gt).
fn bench_view(
    path: &str,
    queries: &[Vec<f32>],
    gt: Option<&[Vec<usize>]>,
) -> (f64, Option<f32>, Option<f32>) {
    let file = fs::File::open(path).expect("open index file");
    let map = unsafe { Mmap::map(&file).expect("mmap") };
    let view = VecqView::from_bytes(&map).expect("parse view");
    let mut hits1 = 0usize;
    let mut hits10 = 0usize;
    let t = Instant::now();
    for (qi, q) in queries.iter().enumerate() {
        let res = view.search(q, 10);
        if let Some(gt) = gt {
            if res[0].0 == gt[qi][0] {
                hits1 += 1;
            }
            for t in &gt[qi] {
                if res.iter().any(|(i, _)| i == t) {
                    hits10 += 1;
                }
            }
        }
    }
    let elapsed = t.elapsed();
    let nq = queries.len() as f64;
    let ms_q = elapsed.as_secs_f64() * 1e3 / nq;
    let recall1 = gt.map(|_| hits1 as f32 / queries.len() as f32);
    let recall10 = gt.map(|_| hits10 as f32 / (queries.len() * 10) as f32);
    (ms_q, recall1, recall10)
}

fn main() {
    let mut dir = "/tmp/vecq-bench-100k".to_string();
    let mut skip_gt = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--dir" => dir = args.next().expect("--dir needs a value"),
            "--skip-gt" => skip_gt = true,
            other => panic!("unknown arg: {other}"),
        }
    }

    let meta = fs::read_to_string(format!("{dir}/meta.json")).expect("meta.json");
    let (nb, nq, dim) = (
        meta_get(&meta, "n_base"),
        meta_get(&meta, "n_query"),
        meta_get(&meta, "dim"),
    );
    println!("dataset: n={nb} queries={nq} dim={dim} (dir {dir})");

    let base = load_f32(&format!("{dir}/base.f32"), nb, dim);
    let queries = load_f32(&format!("{dir}/queries.f32"), nq, dim);

    // Ground truth: exact f32 brute force (same code shape as `real`), which
    // doubles as the f32 scan-cost reference at this N.
    let gt: Option<Vec<Vec<usize>>> = if skip_gt {
        println!("ground truth: skipped (--skip-gt); recall columns omitted");
        None
    } else {
        let t0 = Instant::now();
        let gt: Vec<Vec<usize>> = queries
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
        let e = t0.elapsed();
        println!(
            "ground truth: exact f32 cosine brute force {:.2?} total ({:.2} ms/q, single-thread f32 scan reference)",
            e,
            e.as_secs_f64() * 1e3 / nq as f64
        );
        Some(gt)
    };
    let gts = gt.as_deref();

    let mut rows: Vec<Row> = Vec::new();

    // -- plain 5-bit (default) and working_dim-256, through mmap views ------
    for (bits, wd, name, note) in [
        (5u8, dim, "server_5bit_full", "default width"),
        (5, 256, "server_5bit_wd256", "Matryoshka working_dim"),
    ] {
        let (idx, build) = build_index(dim, wd, bits, &base);
        let (path, file_bytes) = write_index(&dir, name, &idx);
        drop(idx);
        let (ms_q, r1, r10) = bench_view(&path, &queries, gts);
        rows.push(Row {
            mode: if wd == dim {
                "plain 5-bit view (default)".to_string()
            } else {
                format!("plain 5-bit view, wd={wd}")
            },
            build: Some(build),
            file_bytes,
            ms_q,
            recall1: r1,
            recall10: r10,
            note,
        });
    }

    // -- plain 4-bit view + cascade (#22) on the same persisted artifact ----
    // Methodology rule (see scale_probe): in-memory scoring (f32 scales)
    // differs from the persisted f16 artifact at ~5e-4, which reorders
    // tie-heavy top-10 lists. All rows are therefore measured against the
    // SAME reloaded-from-file state so the columns are comparable. Cascade
    // signatures are derived in memory from the reloaded codes (enable_cascade
    // on a freshly loaded index) — same discipline as a serving process that
    // mmaps the file, then enables the cascade.
    let (idx4, build4) = build_index(dim, dim, 4, &base);
    let (path4, file_bytes4) = write_index(&dir, "server_4bit_full", &idx4);
    drop(idx4);
    let (ms_q, r1, r10) = bench_view(&path4, &queries, gts);
    rows.push(Row {
        mode: "plain 4-bit view".to_string(),
        build: Some(build4),
        file_bytes: file_bytes4,
        ms_q,
        recall1: r1,
        recall10: r10,
        note: "cascade prefilter base",
    });

    // Reload the persisted (f16) artifact, then enable the cascade on it.
    let raw4 = fs::read(&path4).expect("read 4-bit file");
    let mut idx4r = VecqIndex::from_bytes(&raw4).expect("parse 4-bit file");
    idx4r.enable_cascade();
    for &r in &[50usize, 100, 200, 400] {
        let mut hits1 = 0usize;
        let mut hits10 = 0usize;
        let t = Instant::now();
        for (qi, q) in queries.iter().enumerate() {
            let res = idx4r.search_cascade(q, 10, r);
            if let Some(gt) = gts {
                if res[0].0 == gt[qi][0] {
                    hits1 += 1;
                }
                for t in &gt[qi] {
                    if res.iter().any(|(i, _)| i == t) {
                        hits10 += 1;
                    }
                }
            }
        }
        let elapsed = t.elapsed();
        rows.push(Row {
            mode: format!("cascade 4-bit r={r}"),
            build: None,
            file_bytes: file_bytes4,
            ms_q: elapsed.as_secs_f64() * 1e3 / nq as f64,
            recall1: gts.map(|_| hits1 as f32 / nq as f32),
            recall10: gts.map(|_| hits10 as f32 / (nq * 10) as f32),
            note: "signatures in RAM, not in file",
        });
    }

    // -- table ---------------------------------------------------------------
    println!(
        "\n{:<28} {:>10} {:>10} {:>8} {:>9} {:>7} {:>8}  note",
        "mode", "build", "file MB", "B/vec", "ms/q", "r@1", "r@10"
    );
    for row in &rows {
        let build = row
            .build
            .map(|b| format!("{:.2?}", b))
            .unwrap_or_else(|| "—".into());
        let r1 = row
            .recall1
            .map(|v| format!("{v:.3}"))
            .unwrap_or_else(|| "—".into());
        let r10 = row
            .recall10
            .map(|v| format!("{v:.3}"))
            .unwrap_or_else(|| "—".into());
        println!(
            "{:<28} {:>10} {:>10.1} {:>8} {:>9.2} {:>7} {:>8}  {}",
            row.mode,
            build,
            row.file_bytes as f64 / 1e6,
            row.file_bytes / nb,
            row.ms_q,
            r1,
            r10,
            row.note
        );
    }
    println!(
        "\nsingle-threaded, aarch64 release; recall deterministic, timings host-specific.\nExtends the edge profile (n=2000, docs/BENCHMARK.md) to server scale for #52."
    );
}

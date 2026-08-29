# vecq

> Training-free vector quantization at configurable width (4/5/6-bit) and search — the "SQLite profile" for vector storage on edge devices.

`vecq` compresses dense embeddings **~5x** into a single deterministic file, using a zero-dependency pure-Rust crate. No training pass, no server, no C++.

```
f32 index:     3,072 bytes/vector (768-dim)
vecq 5-bit:      642 bytes/vector  (4.8x smaller, recall@10 0.979)
vecq 4-bit:      514 bytes/vector  (6.0x smaller, recall@10 0.958)
vecq residual: 1,028 bytes/vector  (3.0x smaller, recall@10 0.984)
```

It is the semantic-search engine for on-device and offline-first workloads — the layer below [uteke](https://github.com/codecoradev/uteke), the SQLite-based memory engine, where it is available as an optional search backend.

## Why vecq

| | vecq | HNSW libraries (usearch etc.) | server engines (Qdrant) |
|---|---|---|---|
| bytes/vector (768-dim) | **642** | 3,072 | 3,072+ |
| dependencies | **none** (pure Rust) | C++ FFI | full server |
| build index (2k vectors) | **75 ms** | 893 ms | — |
| deterministic across platforms | **yes, bit-identical** | no | n/a |
| recall@10 (real embeddings) | 0.979 | 0.995 | 0.995 |

**Use vecq when** your vectors live on a device: mobile apps, embedded, offline-first local search, shipping a pre-built index inside a binary. Size, cold-start, and determinism matter more there than last-mile recall.

**Don't use vecq when** you need exact search, payload filtering, or server-scale throughput — use a real vector database. vecq is deliberately small and simple, not a Qdrant replacement.

## How it works

1. **RHDH rotation** — random diagonal sign + Walsh-Hadamard transform spreads any vector's energy so coordinates become approximately N(0,1) (the flatness property). The sign seed lives in the file header, so results are bit-identical on any architecture.
2. **Lloyd-Max quantization** — optimal scalar quantizer centroids for N(0,1), precomputed offline and embedded as constants for 4, 5, and 6 bits (default **5-bit** = compression/recall sweet spot). No training on your data.
3. **Asymmetric scoring** — queries stay f32; only the database is quantized. The score is an unbiased cosine-similarity estimate with per-vector scale correction, computed with batched SIMD kernels (explicit NEON on aarch64, runtime-detected AVX2 on x86_64) and a portable scalar path elsewhere — all produce identical bits, guarded by bitwise parity tests.

Based on techniques validated in the RaBitQ / MonaVec line of research (random rotation + fixed optimal quantizers, training-free).

## Modes

| mode | bytes/vec | recall@10 | ms/query | use when |
|---|---|---|---|---|
| **5-bit (default)** | 642 | 0.979 | 3.21 | the compression/recall sweet spot |
| 4-bit | 514 | 0.958 | 0.89 | maximum squeeze; required for cascade search |
| 6-bit | 770 | 0.980 | 3.24 | residual-class recall at 25% less storage |
| 4-bit + residual | 1,028 | 0.984 | 1.76 | maximum recall; fastest high-recall path |

Real EmbeddingGemma, 768-dim, aarch64 release, n=2,000 — full methodology and the width matrix in [`docs/BENCHMARK.md`](docs/BENCHMARK.md).

## Usage

```rust
use vecq_core::{VecqIndex, VecqView};

let mut index = VecqIndex::new(768, 42 /* seed */);
// Optional: pick the Lloyd-Max width before the first `add` (default 5-bit).
index.set_bits(5);
// Optional recall mode: `VecqIndex::with_residual(768, 42)` adds a second
// Lloyd-Max pass over the first pass's residual — ~2x storage, the best
// recall, and the fastest high-recall path (see Modes above).
for v in &vectors { index.add(v); }

let hits: Vec<(usize, f32)> = index.search(&query, 10);

// Keyed layer: stable u64 identity + removal for incremental workloads.
index.add_keyed(1001, &doc_vec);
index.add_keyed_multi(1001, &doc_vec_2);   // several vectors under one key
index.relabel(1001, 2002);                 // rename a key in place
index.remove_keyed(2002);                  // tombstone; `compact()` reclaims the slot
let keyed_hits: Vec<(u64, f32)> = index.search_keyed(&query, 10);

// Cascade search (opt-in approximate, 4-bit width): rank by cheap 2-bit
// codes, rescore the closest r with the full 4-bit path. Deterministic;
// r >= n is exactly `search`. Prefilter quality is data-dependent.
index.enable_cascade();
let approx: Vec<(usize, f32)> = index.search_cascade(&query, 10, 200);

// Single-file persistence, deterministic across platforms.
// Tombstones are dropped on disk; keys persist (format v1.3 keyed table).
let bytes = index.to_bytes();
let back = VecqIndex::from_bytes(&bytes).unwrap();
assert_eq!(index.search(&query, 10), back.search(&query, 10));

// Zero-copy serving: parse file bytes without copying payloads — point it
// at an mmap'd file for large read-only indexes (map + parse in
// microseconds; results bit-identical to the loaded index).
let file = std::fs::File::open("index.vecq").unwrap();
let map = unsafe { memmap2::Mmap::map(&file).unwrap() };
let view = VecqView::from_bytes(&map).unwrap();
let same_hits = view.search(&query, 10);
```

## Guarantees

- **Deterministic**: same file + same query → identical result bits on any platform. The seed lives in the header, the scoring path has a fixed association order and no FMA contraction, and unit tests enforce SIMD/scalar bit-identity at every width.
- **Zero dependencies** in `vecq-core`'s quantization path.
- **Recall@10 ≥ 0.95** on real embedding data at every width: 0.958 (4-bit) up to 0.984 (residual) — see `docs/BENCHMARK.md`.
- **Forward-compatible format**: readers accept v1 (f32 scales), v1.1 (f16 scales), v1.2 (Matryoshka working_dim), v1.3 (keyed-slot table), v1.4 (residual codes) and v1.5 (explicit width byte) files.

## Matryoshka models

Embedding models trained Matryoshka-style (EmbeddingGemma, OpenAI `text-embedding-3-*`, …) degrade gracefully when truncated to their leading dimensions. Build an index over the leading `working_dim` dims to cut storage and scan cost roughly proportionally:

```rust
// 768-dim model, quantize only the leading 256 dims:
let mut index = VecqIndex::with_working_dim(768, 256, 42);
index.add(&embedding);          // always pass the full 768-dim vector
let hits = index.search(&query, 10);

// Truncation happens BEFORE normalization + rotation (RHDH mixes dims, so
// post-rotation truncation would not be Matryoshka-equivalent). Scores are
// comparable only with indexes sharing the same working_dim and seed.
```

## Performance

Default width (5-bit), aarch64, single-threaded, 2,000 real EmbeddingGemma vectors (768-dim): search **3.21 ms/query**, build **75 ms**, recall@10 **0.979**. The 4-bit width trades to 0.89 ms/query @ 0.958; residual trades up to 0.984 @ 1.76 ms/query. Full methodology, the width matrix, the usearch comparison, and the per-architecture scoring-path matrix (NEON / AVX2 / scalar, all bit-identical) in [`docs/BENCHMARK.md`](docs/BENCHMARK.md).

## Persistence & serving

- **SQLite BLOB** for mutable, transactional, embedded storage: schema, save/load pattern, atomicity, measured latencies at 1k/10k/50k vectors, and pitfalls — [`docs/SQLITE.md`](docs/SQLITE.md).
- **Zero-copy views** for large read-only serving: `VecqView` over an mmap'd file is ready ~76x faster than a full load at 12k vectors (map+parse in microseconds), with identical results — see "When to skip the BLOB" in [`docs/SQLITE.md`](docs/SQLITE.md).

## Used by

- [uteke](https://github.com/codecoradev/uteke) — SQLite-based memory engine; `vecq` is an optional search backend (`--features vecq`) for mobile/embedded deployments of the same engine.

## Status

`v0.x` — file formats v1.2–v1.5 documented and frozen, readers accept v1–v1.5; the library API is stable on the `VecqIndex` path, with `VecqView` for read-only zero-copy serving. crates.io publication is prepared (`cargo package` passes) and will follow once the v0.x API settles.

## License

Apache-2.0

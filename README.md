# vecq

> Training-free 4-bit vector quantization and search — the "SQLite profile" for vector storage on edge devices.

`vecq` compresses dense embeddings **~6x** into a single deterministic file, using a zero-dependency pure-Rust crate. No training pass, no server, no C++.

```
f32 index:  3,072 bytes/vector (768-dim)
vecq:           514 bytes/vector  (5.98x smaller)
```

It is the semantic-search engine for on-device and offline-first workloads — the layer below [uteke](https://github.com/codecoradev/uteke), the SQLite-based memory engine, where it is available as an optional search backend.

## Why vecq

| | vecq | HNSW libraries (usearch etc.) | server engines (Qdrant) |
|---|---|---|---|
| bytes/vector (768-dim) | **514** | 3,072 | 3,072+ |
| dependencies | **none** (pure Rust) | C++ FFI | full server |
| build index (2k vectors) | **64 ms** | 893 ms | — |
| deterministic across platforms | **yes, bit-identical** | no | n/a |
| recall@10 (real embeddings) | 0.958 | 0.995 | 0.995 |

**Use vecq when** your vectors live on a device: mobile apps, embedded, offline-first local search, shipping a pre-built index inside a binary. Size, cold-start, and determinism matter more there than last-mile recall.

**Don't use vecq when** you need exact search, payload filtering, or server-scale throughput — use a real vector database. vecq is deliberately small and simple, not a Qdrant replacement.

## How it works

1. **RHDH rotation** — random diagonal sign + Walsh-Hadamard transform spreads any vector's energy so coordinates become approximately N(0,1) (the flatness property). The sign seed lives in the file header, so results are bit-identical on any architecture.
2. **Lloyd-Max 4-bit quantization** — optimal scalar quantizer centroids for N(0,1), precomputed offline and embedded as constants. No training on your data.
3. **Asymmetric scoring** — queries stay f32; only the database is quantized. The score is an unbiased cosine-similarity estimate with per-vector scale correction, computed with an explicit NEON path on aarch64 and a portable scalar path elsewhere — both produce identical bits.

Based on techniques validated in the RaBitQ / MonaVec line of research (random rotation + fixed optimal quantizers, training-free).

## Usage

```rust
use vecq_core::VecqIndex;

let mut index = VecqIndex::new(768, 42 /* seed */);
for v in &vectors { index.add(v); }

let hits: Vec<(usize, f32)> = index.search(&query, 10);

// Keyed layer: stable u64 identity + removal for incremental workloads.
index.add_keyed(1001, &doc_vec);
index.remove_keyed(1001); // tombstone; `compact()` reclaims the slot
let keyed_hits: Vec<(u64, f32)> = index.search_keyed(&query, 10);

// Single-file persistence, deterministic across platforms.
// Tombstones are dropped on disk; keys live in your own metadata table.
let bytes = index.to_bytes();
let back = VecqIndex::from_bytes(&bytes).unwrap();
assert_eq!(index.search(&query, 10), back.search(&query, 10));
```

## Guarantees

- **Deterministic**: same file + same query → identical result bits on any platform. The seed lives in the header, the scoring path has a fixed association order and no FMA contraction, and a unit test enforces SIMD/scalar bit-identity.
- **Zero dependencies** in `vecq-core`'s quantization path.
- **Recall@10 ≥ 0.95** on real embedding data at 6x compression (see `docs/BENCHMARK.md`).
- **Forward-compatible format**: readers accept v1 (f32 scales) and v1.1 (f16 scales) files.

## Performance

Measured on aarch64, single-threaded, 2,000 real EmbeddingGemma vectors (768-dim): search **0.89 ms/query**, build **64 ms**, recall@10 **0.958**. Full methodology, comparison against usearch, and the per-architecture scoring-path matrix (NEON / AVX2 / scalar, all bit-identical) in [`docs/BENCHMARK.md`](docs/BENCHMARK.md).

## SQLite integration

Storing the index inside your SQLite database as a BLOB (schema, save/load pattern, atomicity, measured latencies at 1k/10k/50k vectors, and pitfalls): [`docs/SQLITE.md`](docs/SQLITE.md).

## Used by

- [uteke](https://github.com/codecoradev/uteke) — SQLite-based memory engine; `vecq` is an optional search backend (`--features vecq`) for mobile/embedded deployments of the same engine.

## Status

`v0.x` — file format v1.1 is frozen; the library API is stable on the `VecqIndex` path. Published to crates.io as `vecq-core`.

## License

Apache-2.0

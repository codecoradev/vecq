# vecq

> Training-free 4-bit vector quantization and search — the "SQLite profile" for vector storage.

`vecq` compresses dense embeddings **~6x** with a zero-dependency Rust crate: no training pass, no server, single deterministic file.

```
f32 index: 3,072 bytes/vector (768-dim)
vecq:         ~400 bytes/vector  (~7.7x smaller)
```

## How it works

1. **RHDH rotation** — random diagonal sign + Walsh-Hadamard transform spreads any vector's energy so coordinates become approximately N(0,1) (flatness property). The sign seed lives in the file header, so results are bit-identical on any architecture.
2. **Lloyd-Max 4-bit quantization** — optimal scalar quantizer centroids for N(0,1), precomputed offline and embedded as constants. No training on your data.
3. **Asymmetric scoring** — queries stay f32; only the database is quantized. The score is an unbiased cosine-similarity estimate with per-vector scale correction.

Based on techniques validated in the RaBitQ / MonaVec line of research (random-rotation + fixed optimal quantizers, training-free).

## Usage

```rust
use vecq_core::VecqIndex;

let mut index = VecqIndex::new(768, 42 /* seed */);
for v in &vectors { index.add(v); }

let hits: Vec<(usize, f32)> = index.search(&query, 10);

// Single-file persistence, deterministic across platforms.
let bytes = index.to_bytes();
let back = VecqIndex::from_bytes(&bytes).unwrap();
assert_eq!(index.search(&query, 10), back.search(&query, 10));
```

## Guarantees

- **Deterministic**: same file + same query → identical results, any platform (seeded PRNG in header, no hardware float assumptions in the scoring path).
- **Zero dependencies** in `vecq-core`'s quantization path.
- **Recall@10 ≥ 0.9** on real embedding data at 6x compression (see `docs/BENCHMARK.md`).

## Status

Spike / pre-release (`v0.0.x`). API will change. Benchmark numbers in `docs/BENCHMARK.md`.

## License

MIT

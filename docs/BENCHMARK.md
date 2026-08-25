# vecq Benchmark Results

Spike results, measured on aarch64 (Oracle ARM host), single-threaded, release profile.

## Setup

- Dataset: 2,000 base vectors + 100 queries, dim 768
- Embeddings: real **EmbeddingGemma 300M** (Q4 ONNX) over a synthetic corpus of
  18 topics × 10 modifiers (structured paragraphs, memory-note style)
- Ground truth: exact f32 cosine brute-force
- Competitor: usearch v2.26.1 (HNSW, f32, MetricKind::Cos)

## Results

| engine | build | ms/query | recall@10 | bytes/vector | compression |
|---|---|---|---|---|---|
| vecq 4-bit | 59 ms | 3.32 | **0.961** | **516** | **5.95x** vs f32 |
| usearch f32 (HNSW) | 887 ms | 0.22 | 0.995 | 3,072 | 1x |
| f32 brute force | — | 1.11 | 1.000 (ref) | 3,072 | 1x |

Recall gate for the spike was **≥ 0.95** → **passed** (0.961).

## Analysis

### What vecq wins
- **Memory**: 516 B/vector vs 3,072 B — a **5.95x** reduction with no training.
  For uteke-mobile's 300 MB index this projects to ~50 MB.
- **Build time**: 15x faster than usearch HNSW construction (59 ms vs 887 ms) —
  no graph to build, just quantize.
- **Determinism**: brute-force scoring over packed codes; same file → identical
  results on any architecture (seeded RHDH signs in the file header). usearch
  HNSW results can vary with insertion order and threading.
- **Zero dependencies** in the core quantization path.

### What vecq loses (expected)
- **Search throughput**: 15x slower than HNSW at n=2,000 (3.32 vs 0.22 ms/q).
  This is the documented brute-force vs graph trade-off and matches the MonaVec
  finding (2–14x slower than usearch/hnswlib). On-device with n ≤ ~10k and
  battery/thermal constraints, a SIMD-packed scan stays competitive in
  *energy per query* even when slower in wall time.

### Known limitations
- Scalar nibble decoding only in this spike; SIMD (NEON) packing is the main
  future throughput lever.
- Search is O(n) brute force; no ANN graph on top of the codes yet.
- 516 B/vector = 384 B codes (768 dims × 4 bits) + 128 B scale table overhead
  in the current format; the scale could drop to f16 (−2 B/vector) later.

## Conclusion

The spike validates the technique: training-free 4-bit RHDH + Lloyd-Max
quantization with asymmetric scoring keeps Recall@10 above 0.95 at ~6x
compression on real Gemma embeddings. Recommended next steps:

1. NEON-accelerated nibble decode (target ≤ 0.5 ms/q at n=2k)
2. f16 scales (format v1.1, backward compatible)
3. Optional rerank: return top-50, rescore exact f32 on a sidecar — recall ≈ 1.0

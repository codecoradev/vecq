# vecq Benchmark Results

Spike results, measured on aarch64 (Oracle ARM host), single-threaded, release profile.

## Setup

- Dataset: 2,000 base vectors + 100 queries, dim 768
- Embeddings: real **EmbeddingGemma 300M** (Q4 ONNX) over a synthetic corpus of
  18 topics × 10 modifiers (structured paragraphs, memory-note style)
- Ground truth: exact f32 cosine brute-force
- Competitor: usearch v2.26.1 (HNSW, f32, MetricKind::Cos)
- vecq format v1.1 (f16 scales, 2 B/vector)

## Results

| engine | build | ms/query | recall@10 | bytes/vector | compression |
|---|---|---|---|---|---|
| vecq 4-bit | 64 ms | 0.89 | **0.961** | **514** | **5.98x** vs f32 |
| usearch f32 (HNSW) | 893 ms | 0.23 | 0.995 | 3,072 | 1x |
| f32 brute force | — | 1.12 | 1.000 (ref) | 3,072 | 1x |

Recall@1 = 0.910. Recall gate for the spike was **≥ 0.95** → **passed**.

## Changelog vs first spike measurement

- **Search 1.75x faster** (3.32 → 0.89 ms/q after NEON + batching): the scoring loop now uses a
  fixed 8-lane accumulation pattern that LLVM auto-vectorizes while keeping a
  platform-independent association order (cross-platform determinism intact).
- **2 bytes/vector smaller** (516 → 514): format v1.1 stores scales as f16
  (round-trip verified, readers still accept v1 files with f32 scales).
  Measured compression improved from 5.95x to 5.98x.

## Analysis

### What vecq wins
- **Memory**: 514 B/vector vs 3,072 B — a **5.98x** reduction with no training.
  A 300 MB f32 index projects to ~50 MB (e.g. a 768-dim index of ~100k memories).
- **Build time**: 14x faster than usearch HNSW construction (64 ms vs 893 ms) —
  no graph to build, just quantize.
- **Determinism**: fixed-order accumulation over packed codes; same file →
  identical results on any architecture (seeded RHDH signs in the file
  header). usearch HNSW results can vary with insertion order and threading.
- **Zero dependencies** in the core quantization path.

### What vecq loses (expected)
- **Search throughput**: 8x slower than HNSW at n=2,000 (0.89 vs 0.23 ms/q).
  This is the documented brute-force vs graph trade-off and matches the
  MonaVec finding (2–14x slower than usearch/hnswlib). On-device with n ≤ ~10k
  and battery/thermal constraints, sub-2 ms/query is already interactive.

### Known limitations
- Explicit NEON intrinsics (tbl-based nibble LUT gather) is the next
  throughput lever; the current gain comes from auto-vectorization only.
- Search is O(n) brute force; no ANN graph on top of the codes yet.
- f16 scales perturb scores by <1e-3; ranking ties near the cutoff can shift
  by one position (covered by tests: top-10 overlap ≥ 9/10, top-1 unchanged).

## Scoring paths per architecture

All paths produce **bit-identical scores** (same association order, no FMA
contraction); a unit test enforces AVX2 == scalar and NEON == scalar on
overlapping inputs. `search()` additionally batches 4 vectors per pass on
both SIMD paths.

> **Format note:** writers emit format v1.2 since the Matryoshka `working_dim`
> feature (issue #24) — identical payload to v1.1 for full-dim indexes, with
> the reserved header field carrying `working_dim` for truncated indexes.
> Readers accept v1, v1.1 and v1.2.

| architecture | path | selection |
|---|---|---|
| aarch64 (Apple Silicon, ARM servers) | explicit NEON (`vqtbl4q_u8` LUT gather), 4-vector batching | compile time (NEON is baseline) |
| x86_64 with AVX2 | explicit AVX2 (`vgatherdps` LUT gather), 4-vector batching | runtime (`is_x86_feature_detected!`) |
| x86_64 without AVX2 | portable scalar (reference association order) | runtime fallback |
| other targets | portable scalar | compile time |

The measured table above is aarch64 (NEON path). x86_64 AVX2 numbers are
pending measurement on native hardware; the expected gain over the scalar
path is roughly 1.5–2.5x on LUT-heavy scoring workloads (gather throughput
bound). Rosetta-emulated runs are explicitly **not** used as x86_64
benchmarks — Rosetta neither advertises AVX2 via CPUID nor reflects native
throughput.

## Head-to-head vs other 4-bit quantizers (issue #28)

Harness: `cargo run -p vecq-bench --release --bin vs_quantizers` — identical
synthetic clustered dataset for every engine (n=10k/1k, 200 queries), exact
f32 cosine ground truth, aarch64 (Apple Silicon) single-threaded, release.
Recall numbers are deterministic across runs; timings vary ~±20%.

| dataset | engine | bytes/vec | build ms | ms/query | recall@10 |
|---|---|---|---|---|---|
| n=10k, dim=768 | f32 brute (ref) | 3072 | — | 8.9 | 1.000 |
| | **vecq 4-bit** | 514 | **94** | **2.1–2.7** | **0.840** |
| | TurboQuant-MSE 4-bit (SDC) | 384 | 1080 | 15.0–15.7 | 0.798 |
| | RaBitQ 4-bit brute (FHT-Kac) | 392 | 185–250 | 37.9–39.3 | 0.827 |
| n=10k, dim=384 | **vecq 4-bit** | 258 | **41** | **~1.0** | **0.846** |
| | TurboQuant-MSE 4-bit (SDC) | 192 | ~270 | 7.3–9.0 | 0.813 |
| | RaBitQ 4-bit brute (FHT-Kac) | 200 | 85–100 | 18.6–29.4 | 0.843 |
| n=1k, dim=384 | **vecq 4-bit** | 258 | 7 | **0.19–0.34** | **0.875** |
| | TurboQuant-MSE 4-bit (SDC) | 192 | ~275 | ~1.1 | 0.844 |
| | RaBitQ 4-bit brute (FHT-Kac) | 200 | 18 | ~2.3 | 0.873 |

Methodology notes (honest labeling):
- Recall is **not** comparable to the 0.958 EmbeddingGemma table above — this
  dataset is the harder synthetic clustered set used by `vecq-bench`.
- vecq scans 4-bit codes with the explicit NEON kernel + 4-vector batching.
- TurboQuant-MSE: symmetric distance computation (both query and database in
  the shared Lloyd-Max codebook domain) — the crate exposes no ADC path or
  rotation accessor; codebook bytes are excluded from bytes/vec.
- RaBitQ: `rabitq-rs` 0.9 brute-force index as implemented (train uses its
  internal rayon pool); bytes/vec = 4-bit codes + two per-vector f32 norms.
- vecq bytes/vec include the per-vector f16 scale and power-of-two padding
  (768 → 1024, a ~25% padding tax the competitors don't pay).
- x86_64 numbers pending native measurement (see scoring-path table).

**Go/no-go for follow-ups:** vecq already leads both competitors on scan
speed (5–14x) and recall at dim 768. Residual quantization (#23) should
therefore be evaluated as a **recall lift at the same 514 B budget** (e.g.
4-bit + residual at equal total bytes vs the competitors' plain 4-bit), and
the binary-signature cascade (#22) remains the scan-speed lever for larger
n. Verdict: proceed with both, benchmarked against this baseline.

## Conclusion

The spike validates the technique: training-free 4-bit RHDH + Lloyd-Max
quantization with asymmetric scoring keeps Recall@10 above 0.95 at ~6x
compression on real Gemma embeddings. Recommended next steps:

1. ~~Explicit NEON nibble decode (target ≤ 0.8 ms/q at n=2k)~~ — done
   (explicit NEON + AVX2 nibble-gather paths with bit-identity tests)
2. Optional rerank: return top-50, rescore exact f32 on a sidecar — recall ≈ 1.0
3. Top-k selection without full sort (bounded binary heap) — done

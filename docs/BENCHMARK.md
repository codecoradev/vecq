# vecq Benchmark Results

Spike results, measured on aarch64 (Oracle ARM host), single-threaded, release profile.

## Setup

- Dataset: 2,000 base vectors + 100 queries, dim 768 (a second, server-scale
  profile with 100,000 base vectors + 200 queries is documented in
  [Server scale](#server-scale-issue-52) below)
- Embeddings: real **EmbeddingGemma 300M** (Q4 ONNX) over a synthetic corpus of
  18 topics × 10 modifiers (structured paragraphs, memory-note style)
- Ground truth: exact f32 cosine brute-force
- Competitor: usearch v2.26.1 (HNSW, f32, MetricKind::Cos)
- vecq file format v1.5 (width byte); the index under test uses the 5-bit default

## Results

| engine | build | ms/query | recall@10 | bytes/vector | compression |
|---|---|---|---|---|---|
| vecq 5-bit (default) | 75 ms | 3.21 | **0.979** | 642 | **4.78x** vs f32 |
| vecq 4-bit | 64 ms | 0.89 | 0.958 | 514 | 5.98x vs f32 |
| usearch f32 (HNSW) | 893 ms | 0.23 | 0.995 | 3,072 | 1x |
| f32 brute force | — | 1.12 | 1.000 (ref) | 3,072 | 1x |

Recall@1 = 0.970 (default) / 0.910 (4-bit). Recall gate for the spike was **≥ 0.95** → **passed** (both widths).

## Width matrix (#39/#40, Aug 2026)

All modes on the same dataset, aarch64 release, post wide-kernel:

| mode | recall@1 | recall@10 | bytes/vec | compression | ms/q |
|---|---|---|---|---|---|
| plain 4-bit | 0.910 | 0.958 | 514 | 5.98x | 0.89 |
| plain 5-bit (default) | 0.970 | 0.979 | 642 | 4.78x | 3.21 |
| plain 6-bit | 0.960 | 0.980 | 770 | 3.99x | 3.24 |
| plain 4-bit + residual | **0.990** | **0.984** | 1,028 | 2.99x | 1.76 |

- Default is the compression/recall sweet spot; 6-bit matches residual-class
  recall at 25% less storage; residual is the recall mode and — while 5/6-bit
  scoring is extraction-bound — also the fastest high-recall option.
- 5/6-bit latency ceiling: `vqtbl` cannot address 128/256-byte centroid
  tables, so vector gather needs a range-split (≈2–3x the 4-bit gather cost).
  Split-layout codes (separate nibble/high-bit streams) project only
  ~2.2–2.7 ms/q — tracked with the full analysis in issue #40.

## Server scale (issue #52)

Same pipeline as the edge profile, scaled to 100,000 base vectors + 200
queries (dim 768), measured single-threaded on aarch64 (Oracle ARM host,
4 cores), release profile:

| mode | build | file | B/vec | ms/q | recall@1 | recall@10 |
|---|---|---|---|---|---|---|
| f32 brute force (GT reference) | — | 307.2 MB | 3,072 | 61.0 | 1.000 (ref) | 1.000 (ref) |
| plain 5-bit view (default) | 4.3 s | 64.2 MB | 642 | 168.8 | 0.350 | 0.850 |
| plain 5-bit view, wd=256 | 1.0 s | 16.2 MB | 162 | 42.2 | 0.400 | 0.772 |
| plain 4-bit view | 3.5 s | 51.4 MB | 514 | 51.2 | 0.355 | 0.818 |
| cascade 4-bit r=50 | — | 51.4 MB | 514 | 77.4 | 0.375 | 0.817 |
| cascade 4-bit r=100 | — | 51.4 MB | 514 | 79.5 | 0.380 | 0.817 |
| cascade 4-bit r=200 | — | 51.4 MB | 514 | 78.9 | 0.345 | 0.818 |
| cascade 4-bit r=400 | — | 51.4 MB | 514 | 80.5 | 0.345 | 0.818 |

Reproduce:

```sh
python3 scripts/gen_dataset.py --n-base 100000 --n-query 200 --out /tmp/vecq-bench-100k
cargo run --release -p vecq-bench --bin server_scale
```

Methodology (honest labeling):

- The dataset is seeded and reproducible; the corpus builder is byte-identical
  to the edge profile's, so the server base set is a strict superset of the
  edge one. The published run embedded the corpus on x86 (Modal CPU workers,
  onnxruntime 1.25.0) using the same model files and the same generator code —
  embeddings from the two runtimes agree at cosine ≥ 0.9996 on a 64-vector
  probe — while every vecq-side number (quantization, search, ground truth)
  is measured on the aarch64 host against exactly this persisted dataset.
- All rows are scored against the **persisted f16 artifact** (mmap'd view or
  file-reloaded index), not in-memory f32-scale state: f16 scales perturb
  scores by ≤ ~5e-4, which reorders tie-heavy top-10 lists, so mixing the two
  representations makes columns incomparable (`scale_probe` harness documents
  the delta). Cascade signatures are derived in memory from the reloaded
  codes, as a serving process would.
- Recall values are deterministic for a given dataset; timings are
  host-specific. The edge-profile recall tables are unaffected.

Findings:

- **Storage and build scale linearly and hold**: 4.78x compression at 100K
  (64.2 MB vs 307.2 MB), 4.3 s single-threaded build. The compression story
  does not degrade with N — a 100K × 768 index fits in ~64 MB of RAM or page
  cache.
- **Recall is N-dependent**: with 50x more data the true top-10 neighbors sit
  much closer together, and ~1e-3 quantization noise flips rankings — 5-bit
  recall@10 falls 0.974 → 0.850 and recall@1 0.940 → 0.350 (4-bit: 0.957 →
  0.818). Where recall at server N matters, the available levers are 6-bit /
  residual (edge-profile recall advantage carries structurally, at their
  storage cost) — measure on your corpus before committing.
- **Single-thread scan at server N**: 5/6-bit scoring is extraction-bound and
  at 100K loses to exact f32 brute force (169 vs 61 ms/q); only 4-bit
  (51 ms/q) and wd=256 (42 ms/q) stay ahead of the f32 scan. vecq's server
  pitch at this N is the footprint, not raw single-thread latency.
- **Cascade (#22) is not the server-scale throughput lever**: prefilter + r
  rescore costs as much as the whole plain 4-bit scan (77–80 vs 51 ms/q) at
  equal recall (r ≥ 200 saturates to plain). The remaining lever is a
  parallel scan (#51): same kernels over chunks + fixed-order merge.
- **wd=256 (Matryoshka truncation)**: 19x compression vs f32 (162 B/vec) and
  the fastest scan, but recall on this 18-topic corpus is materially lower
  (r@10 0.772) — truncation quality is corpus-dependent, evaluate per
  workload.

## Changelog vs first spike measurement

- **Search 1.75x faster** (3.32 → 0.89 ms/q after NEON + batching): the scoring loop now uses a
  fixed 8-lane accumulation pattern that LLVM auto-vectorizes while keeping a
  platform-independent association order (cross-platform determinism intact).
- **2 bytes/vector smaller** (516 → 514): format v1.1 stores scales as f16
  (round-trip verified, readers still accept v1 files with f32 scales).
  Measured compression improved from 5.95x to 5.98x.

## Analysis

### What vecq wins
- **Memory**: 642 B/vector at default width vs 3,072 B — a **4.78x** reduction
  with no training (5.98x at 4-bit). A 300 MB f32 index projects to ~63 MB
  (e.g. a 768-dim index of ~100k memories).
- **Build time**: 14x faster than usearch HNSW construction (64 ms vs 893 ms) —
  no graph to build, just quantize.
- **Determinism**: fixed-order accumulation over packed codes; same file →
  identical results on any architecture (seeded RHDH signs in the file
  header). usearch HNSW results can vary with insertion order and threading.
- **Zero dependencies** in the core quantization path.

### What vecq loses (expected)
- **Search throughput**: 14x slower than HNSW at default width (3.21 vs
  0.23 ms/q); the 4-bit width closes it to 4x (0.89). This is the documented
  brute-force vs graph trade-off and matches the MonaVec finding (2–14x
  slower than usearch/hnswlib). On-device with n ≤ ~10k and battery/thermal
  constraints, sub-2 ms/query (4-bit, residual) is already interactive.

### Known limitations
- 5/6-bit scoring is extraction-bound on aarch64 (3.2 ms/q vs 0.89 at 4-bit);
  the remaining lever (split-layout codes) has a bounded ceiling — see the
  width matrix above and issue #40.
- Search is O(n) brute force; no ANN graph on top of the codes yet.
- f16 scales perturb scores by <1e-3; ranking ties near the cutoff can shift
  by one position (covered by tests: top-10 overlap ≥ 9/10, top-1 unchanged).

## Scoring paths per architecture

All paths produce **bit-identical scores** (same association order, no FMA
contraction); a unit test enforces AVX2 == scalar and NEON == scalar on
overlapping inputs. `search()` additionally batches 4 vectors per pass on
both SIMD paths.

> **Format note:** writers emit format v1.3 — v1.2 added the Matryoshka
> `working_dim` header field (issue #24), v1.3 appends a keyed-slot table so
> keyed APIs survive save/reload (issue #32). Readers accept v1, v1.1, v1.2
> and v1.3.

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

## Residual quantization (issue #23)

Opt-in mode (`VecqIndex::with_residual`): a second Lloyd-Max pass codes the
residual left by the first pass; scoring adds the second term. Format v1.4
appends the second code block per vector — readers accept v1–v3.

**Estimator fix (required for the mode to help at all):** the first
implementation divided the two-term score by the approximate norm
`sqrt(sum_sq + rms²·padded)`. The actually-scored reconstruction is
`x̂ = d0 + rms·d1`, whose exact norm includes the cross term `2·rms·⟨d0,d1⟩`
(nonzero: the second-pass codes inherit the correlated quantization-error
pattern, not an independent Gaussian) and the true quantized energy `⟨d1,d1⟩`.
The approximation injected per-vector score distortion that *raised* score
variance (+40% sd, bias unchanged) and flipped top-10 rankings: recall on the
adversarial clustered set **dropped** to 0.58 vs plain 0.66 despite 4.5x
better reconstruction MSE. Fix: compute the exact ‖x̂‖² at encode time from
the stored code pairs and fold it into the stored scale coefficients — no
format change, scoring kernels untouched. Same fix also restored the
self-consistency invariant score(v, v) ≤ 1.0 (the old denominator produced
cosine > 1.0 on some vectors).

Measured after the fix, real EmbeddingGemma dataset (n=2000 + 100 queries,
dim 768, same corpus as the table above), aarch64 release:

| mode | recall@1 | recall@10 | bytes/vec | compression | scan cost |
|---|---|---|---|---|---|
| plain 4-bit | 0.910 | 0.958 | 514 | 5.98x | 1.00x |
| plain + residual | **0.990** | **0.984** | 1.028 | ~3.0x | 1.43x |
| usearch f32 HNSW | — | 0.995 | 3,072 | 1x | — |

Honest labeling:
- Residual roughly doubles the storage (second 4-bit block + second f16
  scale) — it is a recall mode, not a free lunch. Plain stays the default.
- The 1.43x scan cost is the second accumulate term; the cascade (#22)
  remains the throughput lever and composes orthogonally.
- The real-dataset comparison lives in
  `crates/vecq-core/tests/real_residual_validation.rs` (ignored by default;
  requires the dataset from `cto/scripts/gen_embeddings.py`). Re-run it in
  release before merging any estimator/format change.
- Plain-path numbers are unchanged by the fix by construction (single-term
  path untouched); re-verified: `real`/`vs_usearch`/`vs_quantizers` recall
  bit-identical to the baseline tables above.

## Conclusion

The spike validates the technique: training-free RHDH + Lloyd-Max
quantization (4/5/6-bit, default 5-bit) with asymmetric scoring keeps
Recall@10 well above 0.95 at 4.8–6x compression on real Gemma embeddings.
Recommended next steps:

1. ~~Explicit NEON nibble decode (target ≤ 0.8 ms/q at n=2k)~~ — done
   (explicit NEON + AVX2 nibble-gather paths with bit-identity tests)
2. Optional rerank: return top-50, rescore exact f32 on a sidecar — recall ≈ 1.0
3. Top-k selection without full sort (bounded binary heap) — done

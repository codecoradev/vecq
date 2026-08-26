# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] — 2026-08-26

First public release. Training-free 4-bit vector quantization and search.

### Added
- RHDH rotation (random diagonal sign + Walsh-Hadamard) with seeded header storage
- Lloyd-Max 4-bit scalar quantization, centroids embedded as constants
- Asymmetric scoring: f32 query against quantized database, per-vector scale correction
- Explicit NEON scoring path (aarch64), bit-identical to the scalar path (enforced by test)
- 4-vector batched NEON scoring with shared query loads
- Bounded min-heap top-k search (NaN-safe monotonic score keys, no per-query O(n) allocation)
- Single-file persistence, format v1.1 (f16 scales); readers accept v1 (f32 scales)
- Deterministic across platforms: fixed association order, no FMA contraction, seeded PRNG in header
- Benchmark suite vs usearch (`crates/vecq-bench`): 0.89 ms/query, recall@10 0.958, 5.98x compression on real 768-dim embeddings

## [0.0.1] — 2026-08-25

Initial spike: format v1, scalar scoring, first measurements.

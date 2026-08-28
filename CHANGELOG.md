# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed
- Keyed API now survives save/reload: file format v1.3 stores a keyed-slot table, and `from_bytes` restores the full key→slot map (#32)

## [0.2.0] — 2026-08-28

### Added
- Keyed API: `add_keyed` (insert-or-replace under a stable `u64` key), `remove_keyed` (tombstones), `search_keyed`, `compact`, plus `key_of`/`contains_key`/`slots`/`tombstones` introspection (#10, #16)
- Multi-vectors-per-key (`add_keyed_multi`, `remove_keyed_at`) and `relabel` — keyed parity with usearch (#26, #33)
- x86_64 AVX2 scoring path with runtime detection, bit-identical to the scalar/NEON paths (enforced by tests) and 4-vector batching (#11, #17)
- Matryoshka-aware `working_dim` truncation: `VecqIndex::with_working_dim(dim, working_dim, seed)` quantizes only the leading dims of Matryoshka-trained embeddings (#24, #30)
- SQLite BLOB storage guide (`docs/SQLITE.md`): schema shapes, save/load pattern, atomicity, measured latencies, pitfalls (#12, #18)
- Head-to-head benchmark vs TurboQuant-MSE and RaBitQ at 4 bits (`vs_quantizers` harness + results in `docs/BENCHMARK.md`) (#28, #34)
- Per-architecture scoring-path table in `docs/BENCHMARK.md`

### Changed
- File format v1.2: the reserved header field now carries `working_dim` (0 = full dim); payload layout identical to v1.1, readers still accept v1 and v1.1
- `len()`/`is_empty()` report live (non-tombstoned) vector counts; `slots()` reports total

### Fixed
- CLA bot exemption now matches actual bot logins (`dependabot[bot]`, not `app/dependabot`) so dependabot PRs pass CI (#14)

## [0.1.1] — 2026-08-26

### Added
- Release pipeline: push a `vX.Y.Z` tag on `main` to publish `vecq-core` to crates.io and create the GitHub Release (adapted from the uteke release workflow).

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

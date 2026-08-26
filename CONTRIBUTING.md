# Contributing to vecq

vecq is a solo-maintained project with a strong product direction: **the smallest reliable training-free vector index for edge devices**. Contributions are welcome, but alignment matters more than volume.

This document helps you decide *whether* and *how* to contribute in a way that's likely to get merged, so neither of us wastes time.

## Ground rules

- **Determinism is non-negotiable.** Any change to the scoring path must preserve bit-identical results across platforms (NEON vs scalar) and format compatibility. Tests enforce this; keep it that way.
- **Zero dependencies in the core quantization path.** If your change needs a crate, it needs a very good reason.
- **Benchmarks or it didn't happen.** Performance claims must come with a reproducible run (`crates/vecq-bench`), not intuition.

## Workflow

1. Fork, branch from `develop` (feature branches only; `main` and `develop` are protected).
2. `cargo fmt --all && cargo clippy --all-targets -- -D warnings && cargo test --workspace` must pass locally.
3. Open a PR against `develop`. CI runs fmt/clippy/test/build on every PR.
4. Sign the CLA when the bot asks.

## What gets merged fast

- SIMD paths for other architectures (SSE/AVX/NEON variants) **with bit-identity tests**
- Portability fixes, format-reader edge cases
- Benchmark improvements, documentation

## What won't get merged

- Approximate/graph indexes (HNSW etc.) — out of scope by design
- New dependencies in the core path
- Anything that breaks file-format backward compatibility

## Security

See [SECURITY.md](SECURITY.md).

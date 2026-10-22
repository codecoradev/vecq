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

## License

Contributions are licensed to the project under the Apache-2.0 license (inbound = outbound). No additional agreement is required.

## Contributions are unpaid

Contributing to this project is **voluntary and unpaid**. There is no
compensation, payment, bounty, or financial reward of any kind for
contributions — now or in the future. You contribute on your own time,
at your own discretion, because you want to improve the project.

If any paid-contribution program is ever introduced, it will be announced
explicitly and this document will be updated. Until then, assume every
contribution is volunteer work under the license terms above.

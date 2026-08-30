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

## CLA

All contributions (code, docs, tests, configuration) require a signed
Contributor License Agreement before a pull request can be merged:

- 📋 **Individual?** → [Sign the Individual CLA](https://codecoradev.github.io/cla/?type=individual)
- 🏢 **Contributing on behalf of a company?** → [Sign the Corporate CLA](https://codecoradev.github.io/cla/?type=corporate)

The CLA is a license agreement, not a copyright assignment — you keep
ownership of your work. Signing takes a couple of minutes and is stored
in the [codecoradev/.github](https://github.com/codecoradev/.github)
repository; a bot checks it automatically on every pull request.

## Contributions are unpaid

Contributing to this project is **voluntary and unpaid**. There is no
compensation, payment, bounty, or financial reward of any kind for
contributions — now or in the future. You contribute on your own time,
at your own discretion, because you want to improve the project.

If any paid-contribution program is ever introduced, it will be announced
explicitly and this document will be updated. Until then, assume every
contribution is volunteer work under the license terms above.

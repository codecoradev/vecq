#!/usr/bin/env python3
"""Generate the real EmbeddingGemma dataset used by the vecq benchmarks.

Two documented profiles (see docs/BENCHMARK.md):

  edge (defaults)  --n-base 2000   --n-query 100 --out /tmp/vecq-bench
  server           --n-base 100000 --n-query 200 --out /tmp/vecq-bench-100k

The corpus builder is byte-identical to the original spike generator, so the
server-scale base vectors are a strict superset of the edge ones (same seed 1
draw sequence) and both profiles share one methodology: real EmbeddingGemma
300M (Q4 ONNX) embeddings over a synthetic corpus of 18 topics x 10 modifiers,
mean-pooled and L2-normalized.

Requires: onnxruntime, tokenizers, numpy (bench-only; vecq-core stays zero-dep).
"""
import argparse
import json
import os

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

TOPICS = [
    "rust memory safety", "flutter state management", "vector databases",
    "mobile battery optimization", "sqlite on android", "machine learning inference",
    "indonesian street food", "coffee brewing methods", "mountain hiking gear",
    "startup funding basics", "customer retention tactics", "email deliverability",
    "quantum computing basics", "solar panel efficiency", "electric vehicle charging",
    "jazz music theory", "film editing techniques", "urban gardening",
]
MODIFIERS = [
    "for beginners", "in production", "common pitfalls", "advanced guide",
    "case study", "best practices", "2026 update", "lessons learned",
    "performance notes", "checklist",
]


def sample_texts(n, seed):
    rng = np.random.default_rng(seed)
    texts = []
    for i in range(n):
        t = TOPICS[rng.integers(len(TOPICS))]
        m = MODIFIERS[rng.integers(len(MODIFIERS))]
        body = (
            f"Notes on {t} {m}. "
            f"The key insight about {t} involves careful attention to detail and repeated practice. "
            f"When working with {t}, engineers often observe measurable improvements after iteration. "
            f"Chapter {i % 97}: applications of {t} {m} in real projects show consistent results "
            f"across datasets and environments, with variance depending on configuration choices."
        )
        texts.append(body)
    return texts


def embed(texts, sess, tok, batch=64):
    embs = []
    n_batches = (len(texts) + batch - 1) // batch
    for s in range(0, len(texts), batch):
        chunk = texts[s:s + batch]
        enc = tok.encode_batch([f"title: none\ntext: {t}" for t in chunk])
        # dynamic max len for this batch
        maxlen = min(2048, max(len(e.ids) for e in enc))
        ids = np.zeros((len(chunk), maxlen), dtype=np.int64)
        mask = np.zeros((len(chunk), maxlen), dtype=np.int64)
        for i, e in enumerate(enc):
            ids[i, :len(e.ids)] = e.ids[:maxlen]
            mask[i, :len(e.ids)] = e.attention_mask[:maxlen]
        out = sess.run(None, {"input_ids": ids, "attention_mask": mask})[0]
        m = mask[:, :, None].astype(np.float32)
        emb = (out * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        embs.append(emb.astype(np.float32))
        done = min(s + batch, len(texts))
        if (s // batch) % 16 == 0 or done == len(texts):
            print(f"  embedded {done}/{len(texts)} (batch {s // batch + 1}/{n_batches})",
                  flush=True)
    return np.concatenate(embs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-base", type=int, default=2000)
    ap.add_argument("--n-query", type=int, default=100)
    ap.add_argument("--base-seed", type=int, default=1)
    ap.add_argument("--query-seed", type=int, default=2)
    ap.add_argument("--model", default="/opt/data/models/embeddinggemma-q4/onnx/model_q4.onnx")
    ap.add_argument("--tokenizer", default="/opt/data/models/embeddinggemma-q4/tokenizer.json")
    ap.add_argument("--out", default="/tmp/vecq-bench")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tok = Tokenizer.from_file(args.tokenizer)
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    print("input:", [i.name for i in sess.get_inputs()], flush=True)
    base = embed(sample_texts(args.n_base, args.base_seed), sess, tok)
    queries = embed(sample_texts(args.n_query, args.query_seed), sess, tok)
    print("base:", base.shape, "queries:", queries.shape)
    base.tofile(f"{args.out}/base.f32")
    queries.tofile(f"{args.out}/queries.f32")
    with open(f"{args.out}/meta.json", "w") as f:
        json.dump({
            "n_base": args.n_base,
            "n_query": args.n_query,
            "dim": int(base.shape[1]),
            "base_seed": args.base_seed,
            "query_seed": args.query_seed,
        }, f)
    print("written to", args.out)


if __name__ == "__main__":
    main()

"""Generate reference vectors for the JS<->NumPy parity test.

Loads the exported fp16 weights (the same numbers the browser sees), runs the
reference forward pass in float64, and writes expected outputs for:

  1. "short"  — logits after a short prompt
  2. "window" — logits after a long prompt that forces the JS engine's
                KV-cache rebuild (context slide), replicated here
  3. "greedy" — 60 characters of deterministic argmax generation

Run after training:  python3 tests/make_parity_vectors.py
Then verify with:    node tests/test_parity.mjs
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model import forward, load_weights_json  # noqa: E402


def js_window(tokens, block_size):
    """Replicate engine.js push(): when the cache is full, keep the most
    recent 3/4 of the window and rebuild."""
    ctx = []
    for t in tokens:
        if len(ctx) >= block_size:
            ctx = ctx[-int(block_size * 0.75):]
        ctx.append(t)
    return ctx


def last_logits(params, cfg, ctx):
    logits, _, _ = forward(params, cfg, np.array([ctx]), collect=False)
    return logits[0, -1]


def main():
    weights_path = os.path.join(ROOT, "weights", "weights.json")
    params, cfg, itos = load_weights_json(weights_path, dtype=np.float64)
    stoi = {c: i for i, c in enumerate(itos)}
    enc = lambda s: [stoi[c] for c in s if c in stoi]

    cases = {}

    # 1. short prompt, no window slide
    short = "ROMEO:"
    cases["short"] = {
        "prompt": short,
        "logits": [float(x) for x in last_logits(params, cfg, enc(short))],
    }

    # 2. long prompt that crosses block_size -> exercises the rebuild path
    with open(os.path.join(ROOT, "data", "input.txt"), encoding="utf-8") as f:
        long_prompt = f.read(cfg.block_size + 72)
    ctx = js_window(enc(long_prompt), cfg.block_size)
    assert len(enc(long_prompt)) > cfg.block_size, "window case must overflow context"
    cases["window"] = {
        "prompt": long_prompt,
        "logits": [float(x) for x in last_logits(params, cfg, ctx)],
    }

    # 3. deterministic greedy generation (argmax, stays inside one window)
    ids = enc("ROMEO:")
    n = 60
    assert len(ids) + n < cfg.block_size
    out = []
    for _ in range(n):
        nxt = int(np.argmax(last_logits(params, cfg, ids)))
        ids.append(nxt)
        out.append(nxt)
    cases["greedy"] = {"prompt": "ROMEO:", "n": n, "text": "".join(itos[i] for i in out)}

    path = os.path.join(ROOT, "tests", "parity_vectors.json")
    with open(path, "w") as f:
        json.dump(cases, f, indent=1)
    print(f"wrote {path}")
    print(f"greedy continuation: {cases['greedy']['text']!r}")


if __name__ == "__main__":
    main()

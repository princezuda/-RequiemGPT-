"""Train GlassBox GPT on a plain-text file, character-level, pure NumPy.

    python3 train.py --data data/input.txt --out out

Defaults train a ~0.63M-parameter model (3 layers, 4 heads, 128-dim,
128-char context) — small enough to train on a laptop CPU in well under an
hour and to ship as a single HTML file, large enough to write recognizable
Shakespeare-flavored dialogue.

Implements AdamW + linear warmup + cosine decay + global-norm gradient
clipping, all in NumPy, all in this file.
"""

import argparse
import json
import os
import time

import numpy as np

from model import (Config, backward, count_params, export_weights_json,
                   forward, generate, init_params)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_data(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.uint16)
    n = int(len(data) * 0.9)
    return data[:n], data[n:], chars


def get_batch(data, batch_size, block_size, rng):
    ix = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i:i + block_size] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1:i + 1 + block_size] for i in ix]).astype(np.int64)
    return x, y


# ---------------------------------------------------------------------------
# AdamW
# ---------------------------------------------------------------------------


class AdamW:
    def __init__(self, params, lr=3e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01):
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads, lr):
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mhat = self.m[k] / bc1
            vhat = self.v[k] / bc2
            if params[k].ndim >= 2:  # decay weights, not biases/layernorm
                params[k] *= 1.0 - lr * self.wd
            params[k] -= lr * mhat / (np.sqrt(vhat) + self.eps)


def clip_global_norm(grads, max_norm=1.0):
    total = np.sqrt(sum(float((g * g).sum()) for g in grads.values()))
    if total > max_norm:
        s = max_norm / (total + 1e-6)
        for k in grads:
            grads[k] *= s
    return total


def lr_at(step, base_lr, min_lr, warmup, total):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    if step >= total:
        return min_lr
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * progress))


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def save_ckpt(path, params, opt, step, cfg, itos, val_loss):
    blob = {f"p::{k}": v for k, v in params.items()}
    blob.update({f"m::{k}": v for k, v in opt.m.items()})
    blob.update({f"v::{k}": v for k, v in opt.v.items()})
    meta = dict(step=step, opt_t=opt.t, config=cfg.to_dict(), itos="".join(itos),
                val_loss=val_loss)
    blob["__meta"] = np.array(json.dumps(meta))
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp.removesuffix(".npz"), **blob)
    os.replace(tmp, path)


def load_ckpt(path):
    z = np.load(path)
    meta = json.loads(str(z["__meta"]))
    params = {k[3:]: z[k].copy() for k in z.files if k.startswith("p::")}
    m = {k[3:]: z[k].copy() for k in z.files if k.startswith("m::")}
    v = {k[3:]: z[k].copy() for k in z.files if k.startswith("v::")}
    return params, m, v, meta


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def estimate_loss(params, cfg, data, batch_size, iters, rng):
    losses = []
    for _ in range(iters):
        x, y = get_batch(data, batch_size, cfg.block_size, rng)
        _, loss, _ = forward(params, cfg, x, y, collect=False)
        losses.append(loss)
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/input.txt")
    ap.add_argument("--out", default="out")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--n-layer", type=int, default=3)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--n-embd", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--min-lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-iters", type=int, default=20)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--resume", default=None, help="checkpoint .npz to resume from")
    ap.add_argument("--export", default=None, help="write weights JSON here when done")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    train_data, val_data, itos = load_data(args.data)

    cfg = Config(vocab_size=len(itos), block_size=args.block_size,
                 n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)

    start_step = 0
    if args.resume:
        params, m, v, meta = load_ckpt(args.resume)
        cfg = Config.from_dict(meta["config"])
        itos = list(meta["itos"])
        opt = AdamW(params, weight_decay=args.weight_decay)
        opt.m, opt.v, opt.t = m, v, meta["opt_t"]
        start_step = meta["step"]
        print(f"resumed from {args.resume} at step {start_step}")
    else:
        params = init_params(cfg, rng)
        opt = AdamW(params, weight_decay=args.weight_decay)

    n_params = count_params(params)
    print(f"config: {cfg}")
    print(f"parameters: {n_params:,}  |  train tokens: {len(train_data):,}  "
          f"|  vocab: {cfg.vocab_size}")

    log_path = os.path.join(args.out, "log.txt")
    best_val = float("inf")
    t_last = time.time()
    for step in range(start_step, args.steps):
        lr = lr_at(step, args.lr, args.min_lr, args.warmup, args.steps)
        x, y = get_batch(train_data, args.batch_size, cfg.block_size, rng)
        _, loss, cache = forward(params, cfg, x, y)
        grads = backward(params, cfg, cache)
        gnorm = clip_global_norm(grads, 1.0)
        opt.step(params, grads, lr)

        if step % args.log_every == 0 or step == args.steps - 1:
            dt = (time.time() - t_last) / max(1, args.log_every)
            t_last = time.time()
            tok_s = args.batch_size * cfg.block_size / max(dt, 1e-9)
            line = (f"step {step:5d} | loss {loss:.4f} | lr {lr:.2e} | "
                    f"gnorm {gnorm:.2f} | {dt * 1000:.0f} ms/step | {tok_s:,.0f} tok/s")
            print(line, flush=True)
            with open(log_path, "a") as f:
                f.write(line + "\n")

        if (step > 0 and step % args.eval_every == 0) or step == args.steps - 1:
            val = estimate_loss(params, cfg, val_data, args.batch_size,
                                args.eval_iters, rng)
            line = f"step {step:5d} | VAL loss {val:.4f}"
            print(line, flush=True)
            with open(log_path, "a") as f:
                f.write(line + "\n")
            save_ckpt(os.path.join(args.out, "ckpt.npz"), params, opt,
                      step + 1, cfg, itos, val)
            if val < best_val:
                best_val = val
                save_ckpt(os.path.join(args.out, "ckpt_best.npz"), params, opt,
                          step + 1, cfg, itos, val)

    # a taste of the model, straight from the trainer
    stoi = {c: i for i, c in enumerate(itos)}
    prompt = [stoi.get(c, 0) for c in "ROMEO:"]
    out = generate(params, cfg, prompt, 300, temperature=0.8, top_k=40,
                   rng=np.random.default_rng(0))
    print("\n--- sample ---")
    print("".join(itos[t] for t in out))

    if args.export:
        meta = dict(val_loss=best_val, params=n_params,
                    trained_on="tiny-shakespeare (public domain)",
                    steps=args.steps)
        export_weights_json(params, cfg, itos, args.export, meta)
        print(f"\nexported fp16 weights -> {args.export}")


if __name__ == "__main__":
    main()

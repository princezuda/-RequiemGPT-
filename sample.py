"""Generate text in the terminal from a trained checkpoint or exported weights.

    python3 sample.py --ckpt out/ckpt_best.npz --prompt "JULIET:" -n 400
    python3 sample.py --weights weights/weights.json --prompt "ROMEO:"
"""

import argparse

import numpy as np

from model import Config, generate, load_weights_json
from train import load_ckpt


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", help=".npz checkpoint from train.py")
    src.add_argument("--weights", help="exported weights.json (fp16, what the browser runs)")
    ap.add_argument("--prompt", default="ROMEO:")
    ap.add_argument("-n", "--num-chars", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.ckpt:
        params, _, _, meta = load_ckpt(args.ckpt)
        cfg = Config.from_dict(meta["config"])
        itos = list(meta["itos"])
        print(f"[ckpt @ step {meta['step']}, val loss {meta['val_loss']:.4f}]")
    else:
        params, cfg, itos = load_weights_json(args.weights)
        print("[fp16 export — identical numbers to the browser engine]")

    stoi = {c: i for i, c in enumerate(itos)}
    prompt_ids = [stoi[c] for c in args.prompt if c in stoi] or [stoi["\n"]]
    rng = np.random.default_rng(args.seed)
    out = generate(params, cfg, prompt_ids, args.num_chars,
                   temperature=args.temperature, top_k=args.top_k, rng=rng)
    print("".join(itos[t] for t in out))


if __name__ == "__main__":
    main()

"""GlassBox GPT — a complete GPT in pure NumPy.

Forward pass, hand-derived backward pass, sampling. No PyTorch, no autograd,
no framework: every gradient in this file is written out by hand and verified
against finite differences in tests/test_gradients.py.

Architecture (GPT-2 style, pre-norm, learned positions, no dropout):

    token embedding + position embedding
    N x [ LayerNorm -> causal multi-head self-attention -> residual
          LayerNorm -> MLP (4x expansion, tanh-GELU)    -> residual ]
    LayerNorm -> linear head -> softmax over vocabulary

Parameters live in a flat dict of NumPy arrays so the whole model is
inspectable, serializable, and trivially mirrored by the JavaScript engine
in web/engine.js (which must produce bit-comparable results — see
tests/test_parity.mjs).
"""

import base64
import json
from dataclasses import asdict, dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Config:
    vocab_size: int = 65
    block_size: int = 128   # max context length
    n_layer: int = 3
    n_head: int = 4
    n_embd: int = 128

    @property
    def head_dim(self) -> int:
        assert self.n_embd % self.n_head == 0
        return self.n_embd // self.n_head

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return Config(**{k: int(v) for k, v in d.items()})


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def init_params(cfg: Config, rng: np.random.Generator, dtype=np.float32) -> dict:
    """GPT-2 style init: N(0, 0.02), residual projections scaled down by
    1/sqrt(2*n_layer) so the residual stream variance stays controlled."""
    std = 0.02
    res_std = std / np.sqrt(2 * cfg.n_layer)
    C, V, T = cfg.n_embd, cfg.vocab_size, cfg.block_size

    def normal(shape, s):
        return rng.normal(0.0, s, size=shape).astype(dtype)

    p = {
        "wte": normal((V, C), std),
        "wpe": normal((T, C), std),
        "lnf.g": np.ones(C, dtype=dtype),
        "lnf.b": np.zeros(C, dtype=dtype),
        "lm.w": normal((C, V), std),
        "lm.b": np.zeros(V, dtype=dtype),
    }
    for i in range(cfg.n_layer):
        p[f"l{i}.ln1.g"] = np.ones(C, dtype=dtype)
        p[f"l{i}.ln1.b"] = np.zeros(C, dtype=dtype)
        p[f"l{i}.attn.wqkv"] = normal((C, 3 * C), std)
        p[f"l{i}.attn.bqkv"] = np.zeros(3 * C, dtype=dtype)
        p[f"l{i}.attn.wo"] = normal((C, C), res_std)
        p[f"l{i}.attn.bo"] = np.zeros(C, dtype=dtype)
        p[f"l{i}.ln2.g"] = np.ones(C, dtype=dtype)
        p[f"l{i}.ln2.b"] = np.zeros(C, dtype=dtype)
        p[f"l{i}.mlp.w1"] = normal((C, 4 * C), std)
        p[f"l{i}.mlp.b1"] = np.zeros(4 * C, dtype=dtype)
        p[f"l{i}.mlp.w2"] = normal((4 * C, C), res_std)
        p[f"l{i}.mlp.b2"] = np.zeros(C, dtype=dtype)
    return p


def count_params(params: dict) -> int:
    return sum(int(a.size) for a in params.values())


# ---------------------------------------------------------------------------
# Primitive ops (forward + backward pairs)
# ---------------------------------------------------------------------------

GELU_K = 0.7978845608028654  # sqrt(2/pi)

# Note: x*x*x instead of x**3 — NumPy's pow ufunc is an order of magnitude
# slower than two multiplies, and GELU dominated the training profile.


def gelu(x):
    """tanh-approximated GELU (same formula the JS engine uses)."""
    return 0.5 * x * (1.0 + np.tanh(GELU_K * (x + 0.044715 * (x * x * x))))


def gelu_forward(x):
    """Returns (y, tanh_u); the tanh is cached so backward never recomputes it."""
    t = np.tanh(GELU_K * (x + 0.044715 * (x * x * x)))
    return 0.5 * x * (1.0 + t), t


def gelu_backward(x, t, dy):
    """d/dx [0.5 x (1 + tanh(u))] = 0.5(1+t) + 0.5 x (1-t^2) du/dx."""
    du_dx = GELU_K * (1.0 + 3 * 0.044715 * (x * x))
    return dy * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du_dx)


def layernorm(x, g, b, eps=1e-5):
    """Normalize over the last axis. Returns y plus what backward needs."""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    xhat = (x - mu) * rstd
    return xhat * g + b, (xhat, rstd)


def layernorm_backward(dy, g, cache):
    """Standard LayerNorm backward.

    With xhat = (x - mu) * rstd and dxhat = dy * g:
        dx = rstd * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat))
    where means are taken over the normalized (last) axis.
    """
    xhat, rstd = cache
    dxhat = dy * g
    dg = (dy * xhat).reshape(-1, xhat.shape[-1]).sum(axis=0)
    db = dy.reshape(-1, dy.shape[-1]).sum(axis=0)
    dx = rstd * (
        dxhat
        - dxhat.mean(axis=-1, keepdims=True)
        - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True)
    )
    return dx, dg, db


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def split_heads(x, n_head):
    """(B, T, C) -> (B, H, T, hs)"""
    B, T, C = x.shape
    return x.reshape(B, T, n_head, C // n_head).transpose(0, 2, 1, 3)


def merge_heads(x):
    """(B, H, T, hs) -> (B, T, C)"""
    B, H, T, hs = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, T, H * hs)


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


def forward(params, cfg: Config, idx, targets=None, collect=True):
    """Run the model.

    idx:     (B, T) int array of token ids, T <= cfg.block_size
    targets: optional (B, T) int array for next-token cross-entropy

    Returns (logits, loss, cache). `cache` (when collect=True) holds every
    intermediate needed by backward(), including per-layer attention maps
    under cache["layers"][i]["p"] — that's what the visualizer reads.
    """
    B, T = idx.shape
    assert T <= cfg.block_size, f"sequence length {T} > block_size {cfg.block_size}"
    H, hs = cfg.n_head, cfg.head_dim
    scale = 1.0 / np.sqrt(hs)

    x = params["wte"][idx] + params["wpe"][:T]  # (B, T, C)
    causal = np.tril(np.ones((T, T), dtype=bool))

    layer_caches = []
    for i in range(cfg.n_layer):
        # --- attention block ---
        a, ln1c = layernorm(x, params[f"l{i}.ln1.g"], params[f"l{i}.ln1.b"])
        qkv = a @ params[f"l{i}.attn.wqkv"] + params[f"l{i}.attn.bqkv"]
        q, k, v = np.split(qkv, 3, axis=-1)
        q, k, v = split_heads(q, H), split_heads(k, H), split_heads(v, H)

        att = (q @ k.transpose(0, 1, 3, 2)) * scale          # (B, H, T, T)
        att = np.where(causal, att, -np.inf)                 # no peeking ahead
        p = softmax(att)
        yh = p @ v                                           # (B, H, T, hs)
        y = merge_heads(yh)                                  # (B, T, C)
        o = y @ params[f"l{i}.attn.wo"] + params[f"l{i}.attn.bo"]
        x = x + o

        # --- MLP block ---
        a2, ln2c = layernorm(x, params[f"l{i}.ln2.g"], params[f"l{i}.ln2.b"])
        h1 = a2 @ params[f"l{i}.mlp.w1"] + params[f"l{i}.mlp.b1"]
        h2, tanh_u = gelu_forward(h1)
        m = h2 @ params[f"l{i}.mlp.w2"] + params[f"l{i}.mlp.b2"]
        x = x + m

        if collect:
            layer_caches.append(
                dict(a=a, ln1c=ln1c, q=q, k=k, v=v, p=p, y=y,
                     a2=a2, ln2c=ln2c, h1=h1, h2=h2, tanh_u=tanh_u)
            )

    xf, lnfc = layernorm(x, params["lnf.g"], params["lnf.b"])
    logits = xf @ params["lm.w"] + params["lm.b"]            # (B, T, V)

    loss = None
    probs = None
    if targets is not None:
        probs = softmax(logits)
        nll = -np.log(
            np.maximum(probs[np.arange(B)[:, None], np.arange(T)[None, :], targets], 1e-12)
        )
        loss = float(nll.mean())

    cache = None
    if collect:
        cache = dict(idx=idx, layers=layer_caches, xf=xf, lnfc=lnfc,
                     probs=probs, targets=targets, scale=scale)
    return logits, loss, cache


# ---------------------------------------------------------------------------
# Backward (the whole point of this file)
# ---------------------------------------------------------------------------


def backward(params, cfg: Config, cache) -> dict:
    """Hand-derived gradients of the mean cross-entropy loss w.r.t. every
    parameter. Mirrors forward() exactly, walked in reverse."""
    idx, targets, probs = cache["idx"], cache["targets"], cache["probs"]
    assert targets is not None and probs is not None, "backward needs a loss"
    B, T = idx.shape
    H = cfg.n_head
    scale = cache["scale"]
    grads = {}

    # Softmax + cross-entropy fused gradient: (p - onehot) / N
    dlogits = probs.copy()
    dlogits[np.arange(B)[:, None], np.arange(T)[None, :], targets] -= 1.0
    dlogits /= B * T

    # lm head
    xf = cache["xf"]
    grads["lm.w"] = xf.reshape(-1, xf.shape[-1]).T @ dlogits.reshape(-1, dlogits.shape[-1])
    grads["lm.b"] = dlogits.sum(axis=(0, 1))
    dxf = dlogits @ params["lm.w"].T

    # final LayerNorm
    dx, grads["lnf.g"], grads["lnf.b"] = layernorm_backward(dxf, params["lnf.g"], cache["lnfc"])

    # transformer blocks, in reverse
    for i in reversed(range(cfg.n_layer)):
        c = cache["layers"][i]

        # --- MLP block backward (x = x + m) ---
        dm = dx                                   # gradient flowing into m
        grads[f"l{i}.mlp.w2"] = c["h2"].reshape(-1, c["h2"].shape[-1]).T @ dm.reshape(-1, dm.shape[-1])
        grads[f"l{i}.mlp.b2"] = dm.sum(axis=(0, 1))
        dh2 = dm @ params[f"l{i}.mlp.w2"].T
        dh1 = gelu_backward(c["h1"], c["tanh_u"], dh2)
        grads[f"l{i}.mlp.w1"] = c["a2"].reshape(-1, c["a2"].shape[-1]).T @ dh1.reshape(-1, dh1.shape[-1])
        grads[f"l{i}.mlp.b1"] = dh1.sum(axis=(0, 1))
        da2 = dh1 @ params[f"l{i}.mlp.w1"].T
        dx2, grads[f"l{i}.ln2.g"], grads[f"l{i}.ln2.b"] = layernorm_backward(
            da2, params[f"l{i}.ln2.g"], c["ln2c"]
        )
        dx = dx + dx2                             # residual: gradient adds

        # --- attention block backward (x = x + o) ---
        do = dx
        y = c["y"]
        grads[f"l{i}.attn.wo"] = y.reshape(-1, y.shape[-1]).T @ do.reshape(-1, do.shape[-1])
        grads[f"l{i}.attn.bo"] = do.sum(axis=(0, 1))
        dy = do @ params[f"l{i}.attn.wo"].T       # (B, T, C)
        dyh = split_heads(dy, H)                  # (B, H, T, hs)

        p, q, k, v = c["p"], c["q"], c["k"], c["v"]
        dp = dyh @ v.transpose(0, 1, 3, 2)        # (B, H, T, T)
        dv = p.transpose(0, 1, 3, 2) @ dyh        # (B, H, T, hs)
        # softmax backward; masked positions have p == 0 so they contribute 0
        ds = p * (dp - (dp * p).sum(axis=-1, keepdims=True))
        dq = (ds @ k) * scale
        dk = (ds.transpose(0, 1, 3, 2) @ q) * scale

        dqkv = np.concatenate([merge_heads(dq), merge_heads(dk), merge_heads(dv)], axis=-1)
        a = c["a"]
        grads[f"l{i}.attn.wqkv"] = a.reshape(-1, a.shape[-1]).T @ dqkv.reshape(-1, dqkv.shape[-1])
        grads[f"l{i}.attn.bqkv"] = dqkv.sum(axis=(0, 1))
        da = dqkv @ params[f"l{i}.attn.wqkv"].T
        dx1, grads[f"l{i}.ln1.g"], grads[f"l{i}.ln1.b"] = layernorm_backward(
            da, params[f"l{i}.ln1.g"], c["ln1c"]
        )
        dx = dx + dx1                             # residual: gradient adds

    # embeddings
    grads["wte"] = np.zeros_like(params["wte"])
    np.add.at(grads["wte"], idx, dx)
    grads["wpe"] = np.zeros_like(params["wpe"])
    grads["wpe"][:T] = dx.sum(axis=0)
    return grads


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def generate(params, cfg: Config, idx, n_new, temperature=0.8, top_k=40, rng=None):
    """Autoregressively extend idx (1, T) by n_new tokens."""
    rng = rng or np.random.default_rng()
    idx = np.array(idx, dtype=np.int64).reshape(1, -1)
    for _ in range(n_new):
        window = idx[:, -cfg.block_size:]
        logits, _, _ = forward(params, cfg, window, collect=False)
        logits = logits[0, -1] / max(temperature, 1e-6)
        if top_k and top_k < cfg.vocab_size:
            kth = np.partition(logits, -top_k)[-top_k]
            logits = np.where(logits < kth, -np.inf, logits)
        p = softmax(logits)
        nxt = rng.choice(cfg.vocab_size, p=p)
        idx = np.concatenate([idx, [[nxt]]], axis=1)
    return idx[0]


# ---------------------------------------------------------------------------
# Export — fp16 weights as base64 JSON, consumed by web/engine.js
# ---------------------------------------------------------------------------


def export_weights_json(params, cfg: Config, itos, path, meta=None):
    tensors = {}
    for name, arr in params.items():
        tensors[name] = {
            "shape": list(arr.shape),
            "data": base64.b64encode(arr.astype("<f2").tobytes()).decode("ascii"),
        }
    payload = {
        "format": "glassbox-fp16-v1",
        "config": cfg.to_dict(),
        "itos": "".join(itos),
        "meta": meta or {},
        "tensors": tensors,
    }
    with open(path, "w") as f:
        json.dump(payload, f)
    return path


def load_weights_json(path, dtype=np.float32):
    """Load an exported weights file back into (params, cfg, itos).
    Values round-trip through fp16, exactly like the JS engine sees them."""
    with open(path) as f:
        payload = json.load(f)
    cfg = Config.from_dict(payload["config"])
    params = {}
    for name, t in payload["tensors"].items():
        raw = np.frombuffer(base64.b64decode(t["data"]), dtype="<f2")
        params[name] = raw.reshape(t["shape"]).astype(dtype)
    return params, cfg, list(payload["itos"])

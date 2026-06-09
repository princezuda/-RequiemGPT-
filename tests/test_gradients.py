"""Verify the hand-derived backward pass against central finite differences.

Every single parameter element of a small model is perturbed (+eps / -eps),
the loss recomputed, and the numerical derivative compared to the analytic
gradient from model.backward(). In float64 these should agree to ~1e-9;
we assert a comfortable 1e-5.

Run directly (no pytest needed):  python3 tests/test_gradients.py
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import Config, backward, forward, generate, init_params  # noqa: E402

TINY = Config(vocab_size=11, block_size=6, n_layer=2, n_head=2, n_embd=8)


def _tiny_setup(seed=7):
    rng = np.random.default_rng(seed)
    params = init_params(TINY, rng, dtype=np.float64)
    # non-degenerate ln gains/biases so their gradients are exercised properly
    for k in params:
        if k.endswith(".g"):
            params[k] = params[k] + rng.normal(0, 0.1, params[k].shape)
        if k.endswith(".b") and params[k].ndim == 1:
            params[k] = params[k] + rng.normal(0, 0.1, params[k].shape)
    idx = rng.integers(0, TINY.vocab_size, size=(2, TINY.block_size))
    targets = rng.integers(0, TINY.vocab_size, size=(2, TINY.block_size))
    return params, idx, targets


def test_gradients_match_finite_differences():
    params, idx, targets = _tiny_setup()
    _, _, cache = forward(params, TINY, idx, targets)
    grads = backward(params, TINY, cache)

    # Central differences have a float64 cancellation noise floor around
    # 1e-10 with eps=1e-5, so compare with atol + rtol, not pure relative err.
    eps, atol, rtol = 1e-5, 1e-9, 1e-5
    worst = (0.0, None)
    checked = 0
    for name in sorted(params):
        p = params[name]
        g = grads[name]
        assert g.shape == p.shape, f"{name}: grad shape {g.shape} != param {p.shape}"
        flat_p, flat_g = p.reshape(-1), g.reshape(-1)
        for j in range(flat_p.size):
            orig = flat_p[j]
            flat_p[j] = orig + eps
            _, lp, _ = forward(params, TINY, idx, targets, collect=False)
            flat_p[j] = orig - eps
            _, lm, _ = forward(params, TINY, idx, targets, collect=False)
            flat_p[j] = orig
            num = (lp - lm) / (2 * eps)
            ana = flat_g[j]
            err = abs(num - ana)
            tol = atol + rtol * (abs(num) + abs(ana))
            checked += 1
            if err / tol > worst[0]:
                worst = (err / tol, f"{name}[{j}] num={num:.3e} ana={ana:.3e}")
            assert err < tol, (
                f"gradient mismatch {name}[{j}]: numerical={num:.6e} "
                f"analytic={ana:.6e} |diff|={err:.2e} tol={tol:.2e}"
            )
    print(f"  checked {checked} parameter elements; worst err/tol {worst[0]:.3f} ({worst[1]})")


def test_loss_decreases_overfitting_one_batch():
    """If the gradients are right, plain SGD must be able to memorize one batch."""
    params, idx, targets = _tiny_setup(seed=3)
    _, first, cache = forward(params, TINY, idx, targets)
    lr = 0.5
    last = first
    for _ in range(300):
        grads = backward(params, TINY, cache)
        for k in params:
            params[k] -= lr * grads[k]
        _, last, cache = forward(params, TINY, idx, targets)
    print(f"  one-batch overfit: loss {first:.3f} -> {last:.3f}")
    assert last < 0.10, f"failed to memorize a single batch: {first:.3f} -> {last:.3f}"
    assert last < first / 10


def test_causality():
    """Changing a future token must not change past logits (causal mask works)."""
    params, idx, _ = _tiny_setup(seed=11)
    logits_a, _, _ = forward(params, TINY, idx, collect=False)
    idx2 = idx.copy()
    idx2[:, -1] = (idx2[:, -1] + 1) % TINY.vocab_size  # perturb only last token
    logits_b, _, _ = forward(params, TINY, idx2, collect=False)
    np.testing.assert_allclose(logits_a[:, :-1], logits_b[:, :-1], atol=1e-12)
    print("  causality holds: past logits unaffected by future tokens")


def test_generate_shapes_and_range():
    params, idx, _ = _tiny_setup(seed=5)
    out = generate(params, TINY, idx[:1, :3], n_new=8, temperature=1.0, top_k=5,
                   rng=np.random.default_rng(0))
    assert out.shape == (11,)
    assert out.min() >= 0 and out.max() < TINY.vocab_size
    print("  generate(): shapes and token ranges ok")


if __name__ == "__main__":
    t0 = time.time()
    for fn in [test_gradients_match_finite_differences,
               test_loss_decreases_overfitting_one_batch,
               test_causality,
               test_generate_shapes_and_range]:
        print(f"* {fn.__name__}")
        fn()
    print(f"ALL TESTS PASSED in {time.time() - t0:.1f}s")

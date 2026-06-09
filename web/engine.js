/* GlassBox GPT — browser/Node inference engine in vanilla JavaScript.
 *
 * Mirrors model.py exactly: same architecture, same tanh-GELU constants,
 * same LayerNorm epsilon, same weight layouts. Numerical parity with the
 * NumPy reference is asserted in tests/test_parity.mjs.
 *
 * Engineering notes:
 *  - Incremental KV cache: generating one character costs ~0.7M multiply-
 *    adds instead of ~84M for a full re-forward. When the context window
 *    fills up, the engine keeps the most recent 3/4 of it and rebuilds the
 *    cache once (positions are absolute, so a slide invalidates K/V).
 *  - All accumulation happens in float64 (JS numbers), which lets the
 *    parity test compare against a float64 NumPy forward at ~1e-9 tolerance.
 *  - Per-token attention rows are captured for every layer and head; that
 *    is what the visualizer renders.
 *
 * No dependencies. Runs from file:// in a browser or via `node` for tests.
 */
"use strict";

const GlassBox = (() => {

  // ---- weight decoding -----------------------------------------------

  function b64ToBytes(b64) {
    if (typeof Buffer !== "undefined") {
      return new Uint8Array(Buffer.from(b64, "base64"));
    }
    const s = atob(b64);
    const out = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
    return out;
  }

  // IEEE 754 half -> double. (Little-endian byte pairs.)
  function decodeF16(bytes) {
    const n = bytes.length / 2;
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const u = bytes[2 * i] | (bytes[2 * i + 1] << 8);
      const sign = (u >> 15) & 1 ? -1 : 1;
      const exp = (u >> 10) & 0x1f;
      const mant = u & 0x3ff;
      let v;
      if (exp === 0) v = mant * Math.pow(2, -24);            // subnormal
      else if (exp === 31) v = mant ? NaN : Infinity;         // inf/nan
      else v = (mant + 1024) * Math.pow(2, exp - 25);         // normal
      out[i] = sign * v;
    }
    return out;
  }

  function decodeWeights(payload) {
    const tensors = {};
    for (const [name, t] of Object.entries(payload.tensors)) {
      const arr = decodeF16(b64ToBytes(t.data));
      arr.shape = t.shape;
      tensors[name] = arr;
    }
    return tensors;
  }

  // ---- math helpers ----------------------------------------------------

  const GELU_K = 0.7978845608028654; // sqrt(2/pi), same constant as model.py

  function gelu(x) {
    return 0.5 * x * (1 + Math.tanh(GELU_K * (x + 0.044715 * x * x * x)));
  }

  // y = LayerNorm(x) * g + b over a length-C vector
  function layernorm(x, g, b, out) {
    const C = x.length;
    let mu = 0;
    for (let i = 0; i < C; i++) mu += x[i];
    mu /= C;
    let varr = 0;
    for (let i = 0; i < C; i++) { const d = x[i] - mu; varr += d * d; }
    varr /= C;
    const rstd = 1 / Math.sqrt(varr + 1e-5);
    for (let i = 0; i < C; i++) out[i] = (x[i] - mu) * rstd * g[i] + b[i];
  }

  // out[j] = sum_i x[i] * W[i*cols + j] + b[j]   (vector @ matrix, row-major)
  function vecmat(x, W, b, out, cols) {
    const n = x.length;
    if (b) { for (let j = 0; j < cols; j++) out[j] = b[j]; }
    else { for (let j = 0; j < cols; j++) out[j] = 0; }
    for (let i = 0; i < n; i++) {
      const xi = x[i];
      if (xi === 0) continue;
      const row = i * cols;
      for (let j = 0; j < cols; j++) out[j] += xi * W[row + j];
    }
  }

  function softmaxInPlace(x, n) {
    let m = -Infinity;
    for (let i = 0; i < n; i++) if (x[i] > m) m = x[i];
    let s = 0;
    for (let i = 0; i < n; i++) { x[i] = Math.exp(x[i] - m); s += x[i]; }
    for (let i = 0; i < n; i++) x[i] /= s;
  }

  // ---- the engine ------------------------------------------------------

  class Engine {
    constructor(payload) {
      this.cfg = payload.config;
      this.itos = Array.from(payload.itos);
      this.stoi = {};
      this.itos.forEach((ch, i) => { this.stoi[ch] = i; });
      this.w = decodeWeights(payload);
      this.meta = payload.meta || {};

      const { n_embd: C, n_layer: L, block_size: T } = this.cfg;
      // KV cache: per layer, [T, C] row-major (head h occupies cols h*hs..)
      this.K = []; this.V = [];
      for (let l = 0; l < L; l++) {
        this.K.push(new Float64Array(T * C));
        this.V.push(new Float64Array(T * C));
      }
      // scratch buffers (reused across tokens — no per-token allocation)
      this.x = new Float64Array(C);
      this.a = new Float64Array(C);
      this.qkv = new Float64Array(3 * C);
      this.attY = new Float64Array(C);
      this.o = new Float64Array(C);
      this.h1 = new Float64Array(4 * C);
      this.h2 = new Float64Array(4 * C);
      this.scores = new Float64Array(T);
      this.logits = new Float64Array(this.cfg.vocab_size);
      this.ctx = [];       // token ids currently in the KV cache, in order
      this.atts = null;    // attention rows from the most recent push()
    }

    nParams() {
      let n = 0;
      for (const t of Object.values(this.w)) n += t.length;
      return n;
    }

    encode(str) {
      const ids = [];
      for (const ch of str) {
        if (ch in this.stoi) ids.push(this.stoi[ch]);
        else if ("\t" === ch && " " in this.stoi) ids.push(this.stoi[" "]);
        // characters outside the training vocabulary are dropped
      }
      return ids;
    }

    decode(ids) { return ids.map((i) => this.itos[i]).join(""); }

    reset() { this.ctx = []; this.atts = null; }

    /** Feed a sequence of prompt tokens; returns logits after the last one. */
    feed(ids) {
      let out = null;
      for (let i = 0; i < ids.length; i++) {
        out = this.push(ids[i], { needLogits: i === ids.length - 1 });
      }
      return out;
    }

    /**
     * Process one token through the model at the next cache position.
     * Returns { logits, atts } where atts[l][h] is a Float64Array of the
     * attention distribution this token's query produced over the context.
     */
    push(id, opts = {}) {
      const needLogits = opts.needLogits !== false;
      const quiet = opts.quiet === true; // rebuild mode: skip att capture
      const { n_embd: C, n_layer: L, n_head: H, block_size: T } = this.cfg;
      const hs = C / H;
      const scale = 1 / Math.sqrt(hs);

      // Context full -> keep the freshest 3/4 and rebuild the cache once.
      if (this.ctx.length >= T) {
        const keep = this.ctx.slice(-Math.floor(T * 0.75));
        this.ctx = [];
        for (const kid of keep) this.push(kid, { needLogits: false, quiet: true });
      }

      const t = this.ctx.length; // absolute position of this token
      const { x, a, qkv, attY, o, h1, h2, scores } = this;
      const w = this.w;

      for (let c = 0; c < C; c++) x[c] = w["wte"][id * C + c] + w["wpe"][t * C + c];

      const atts = quiet ? null : [];
      for (let l = 0; l < L; l++) {
        // -- attention block --
        layernorm(x, w[`l${l}.ln1.g`], w[`l${l}.ln1.b`], a);
        vecmat(a, w[`l${l}.attn.wqkv`], w[`l${l}.attn.bqkv`], qkv, 3 * C);
        const K = this.K[l], V = this.V[l];
        for (let c = 0; c < C; c++) {       // cache k and v for position t
          K[t * C + c] = qkv[C + c];
          V[t * C + c] = qkv[2 * C + c];
        }
        const headRows = quiet ? null : [];
        for (let h = 0; h < H; h++) {
          const qOff = h * hs;
          for (let j = 0; j <= t; j++) {    // causal: only positions <= t
            let s = 0;
            const kOff = j * C + qOff;
            for (let d = 0; d < hs; d++) s += qkv[qOff + d] * K[kOff + d];
            scores[j] = s * scale;
          }
          softmaxInPlace(scores, t + 1);
          if (headRows) headRows.push(scores.slice(0, t + 1));
          for (let d = 0; d < hs; d++) {
            let s = 0;
            for (let j = 0; j <= t; j++) s += scores[j] * V[j * C + qOff + d];
            attY[qOff + d] = s;
          }
        }
        if (atts) atts.push(headRows);
        vecmat(attY, w[`l${l}.attn.wo`], w[`l${l}.attn.bo`], o, C);
        for (let c = 0; c < C; c++) x[c] += o[c];

        // -- MLP block --
        layernorm(x, w[`l${l}.ln2.g`], w[`l${l}.ln2.b`], a);
        vecmat(a, w[`l${l}.mlp.w1`], w[`l${l}.mlp.b1`], h1, 4 * C);
        for (let c = 0; c < 4 * C; c++) h2[c] = gelu(h1[c]);
        vecmat(h2, w[`l${l}.mlp.w2`], w[`l${l}.mlp.b2`], o, C);
        for (let c = 0; c < C; c++) x[c] += o[c];
      }

      this.ctx.push(id);
      if (!quiet) this.atts = atts;

      if (!needLogits) return null;
      layernorm(x, w["lnf.g"], w["lnf.b"], a);
      vecmat(a, w["lm.w"], w["lm.b"], this.logits, this.cfg.vocab_size);
      return { logits: this.logits, atts: this.atts };
    }
  }

  // ---- sampling --------------------------------------------------------

  /**
   * Temperature + top-k sampling. Returns the chosen id plus the full
   * (renormalized) distribution actually sampled from — the visualizer
   * shows exactly what the dice saw.
   */
  function sample(logits, { temperature = 0.8, topK = 40, rng = Math.random } = {}) {
    const V = logits.length;
    const scaled = new Float64Array(V);
    const t = Math.max(temperature, 1e-6);
    for (let i = 0; i < V; i++) scaled[i] = logits[i] / t;
    if (topK > 0 && topK < V) {
      const sorted = Array.from(scaled).sort((a, b) => b - a);
      const kth = sorted[topK - 1];
      for (let i = 0; i < V; i++) if (scaled[i] < kth) scaled[i] = -Infinity;
    }
    softmaxInPlace(scaled, V);
    let u = rng();
    let id = V - 1;
    let acc = 0;
    for (let i = 0; i < V; i++) {
      acc += scaled[i];
      if (u <= acc) { id = i; break; }
    }
    return { id, probs: scaled };
  }

  return { Engine, sample, decodeWeights, decodeF16, b64ToBytes };
})();

if (typeof module !== "undefined" && module.exports) module.exports = GlassBox;
if (typeof globalThis !== "undefined") globalThis.GlassBox = GlassBox;

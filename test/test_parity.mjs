/* Cross-language parity test: the JavaScript engine must reproduce the
 * NumPy reference to floating-point accuracy.
 *
 *   node tests/test_parity.mjs
 *
 * Both sides start from identical fp16 weights and accumulate in float64,
 * so disagreement beyond summation-order noise (~1e-12) means a real bug.
 * Tolerance is set at 1e-6 — a thousand times stricter than anything a
 * sampler could notice, a million times looser than what we expect.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { createRequire } from "node:module";

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const require = createRequire(import.meta.url);
const GlassBox = require(join(ROOT, "web", "engine.js"));

const weights = JSON.parse(readFileSync(join(ROOT, "weights", "weights.json"), "utf8"));
const vectors = JSON.parse(readFileSync(join(ROOT, "tests", "parity_vectors.json"), "utf8"));

const engine = new GlassBox.Engine(weights);
let failures = 0;

function check(name, got, want, tol) {
  let worst = 0;
  for (let i = 0; i < want.length; i++) {
    worst = Math.max(worst, Math.abs(got[i] - want[i]));
  }
  const ok = worst < tol;
  console.log(`${ok ? "PASS" : "FAIL"} ${name}: max |js - numpy| = ${worst.toExponential(2)} (tol ${tol})`);
  if (!ok) failures++;
}

// 1 + 2: logits parity, including the KV-cache rebuild path
for (const name of ["short", "window"]) {
  const c = vectors[name];
  engine.reset();
  const out = engine.feed(engine.encode(c.prompt));
  check(`logits/${name}`, out.logits, c.logits, 1e-6);
}

// 3: end-to-end greedy generation must match character-for-character
{
  const c = vectors.greedy;
  engine.reset();
  let res = engine.feed(engine.encode(c.prompt));
  let text = "";
  for (let i = 0; i < c.n; i++) {
    let best = 0;
    for (let j = 1; j < res.logits.length; j++) if (res.logits[j] > res.logits[best]) best = j;
    text += engine.itos[best];
    res = engine.push(best);
  }
  const ok = text === c.text;
  console.log(`${ok ? "PASS" : "FAIL"} greedy ${c.n}-char generation ${ok ? "identical" : `mismatch:\n  js:    ${JSON.stringify(text)}\n  numpy: ${JSON.stringify(c.text)}`}`);
  if (!ok) failures++;
}

// 4: attention rows are valid probability distributions
{
  engine.reset();
  const out = engine.feed(engine.encode("First Citizen:\nWe are"));
  for (let l = 0; l < engine.cfg.n_layer; l++) {
    for (let h = 0; h < engine.cfg.n_head; h++) {
      const row = out.atts[l][h];
      const sum = row.reduce((a, b) => a + b, 0);
      if (Math.abs(sum - 1) > 1e-9 || row.some((p) => p < 0)) {
        console.log(`FAIL attention L${l}h${h} not a distribution (sum=${sum})`);
        failures++;
      }
    }
  }
  if (!failures) console.log("PASS attention rows are normalized distributions (all layers/heads)");
}

if (failures) { console.error(`\n${failures} parity check(s) FAILED`); process.exit(1); }
console.log("\nALL PARITY CHECKS PASSED — the browser runs the same model NumPy trained.");

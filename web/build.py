"""Assemble the single-file artifact: dist/index.html.

Inlines web/engine.js and the exported fp16 weights into web/template.html.
The result is fully self-contained — it runs from file://, offline, forever.

    python3 web/build.py [--weights weights/weights.json] [--out dist/index.html]
"""

import argparse
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(ROOT, "weights", "weights.json"))
    ap.add_argument("--template", default=os.path.join(ROOT, "web", "template.html"))
    ap.add_argument("--engine", default=os.path.join(ROOT, "web", "engine.js"))
    ap.add_argument("--out", default=os.path.join(ROOT, "dist", "index.html"))
    args = ap.parse_args()

    with open(args.weights) as f:
        weights = json.load(f)
    with open(args.template) as f:
        html = f.read()
    with open(args.engine) as f:
        engine = f.read()

    # Compact JSON; escape '<' so no '</script>' (or anything HTML-parser
    # hostile) can appear inside the inline script, regardless of vocab.
    weights_js = json.dumps(weights, separators=(",", ":")).replace("<", "\\u003c")
    assert "</script" not in weights_js.lower()
    assert "</script" not in engine.lower(), "engine.js must not contain </script"

    html = html.replace("const GLASSBOX_WEIGHTS = /*__WEIGHTS_JSON__*/null;",
                        "const GLASSBOX_WEIGHTS = " + weights_js + ";")
    html = html.replace("/*__ENGINE_JS__*/", engine)
    assert "__WEIGHTS_JSON__" not in html and "__ENGINE_JS__" not in html

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html)

    size = os.path.getsize(args.out)
    n_params = sum(math.prod(t["shape"]) for t in weights["tensors"].values())
    print(f"built {args.out}  ({size / 1e6:.2f} MB, {n_params:,} params, "
          f"{len(weights['tensors'])} tensors)")


if __name__ == "__main__":
    main()

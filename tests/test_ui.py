"""Headless-browser test of the built artifact (dist/index.html).

Loads the single file from file://, lets the model generate for a few
seconds, and asserts that every panel of the visualizer is alive: text is
being written, probability bars are rendered, attention strips populate,
the context meter advances, and the console stays free of errors. Also
exercises pause/resume, presets, and layer tabs.

Requires: pip install playwright && playwright install chromium
(Optional dev dependency — the artifact itself needs nothing.)

    python3 tests/test_ui.py [--screenshot docs/screenshot.png]
"""

import argparse
import os
import sys
import time

from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screenshot", default=None)
    ap.add_argument("--seconds", type=float, default=6.0)
    args = ap.parse_args()

    dist = os.path.join(ROOT, "dist", "index.html")
    assert os.path.exists(dist), "build first: python3 web/build.py"

    errors = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1380, "height": 900})
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto("file://" + dist)

        # header stats render real numbers
        params_txt = page.text_content("#stat-params").strip()
        assert params_txt and params_txt != "…", "param count not rendered"
        print(f"  model loaded: {params_txt} params")

        # generation auto-starts; give it a few seconds at the default speed
        time.sleep(args.seconds)
        out1 = page.text_content("#output")
        assert len(out1) > 40, f"expected generated text, got {len(out1)} chars"
        print(f"  generated {len(out1)} chars of output")

        bars = page.locator("#probs .pb").count()
        assert bars == 10, f"expected 10 probability bars, got {bars}"
        chosen = page.locator("#probs .pb.chosen").count()
        assert chosen == 1, f"expected exactly 1 chosen bar, got {chosen}"
        print(f"  probability panel: {bars} bars, 1 chosen")

        heads = page.locator("#heads .headrow").count()
        assert heads >= 4, f"expected attention head rows, got {heads}"
        glow = page.locator("#output span.g").count()
        assert glow > 5, "attention glow spans missing from output"
        print(f"  attention: {heads} head rows, {glow} glow spans on text")

        ctx = page.text_content("#ctxlabel")
        assert int(ctx.split("/")[0].strip()) > 5, f"context meter stuck: {ctx}"
        print(f"  context meter: {ctx.strip()}")

        # pause stops the stream
        page.click("#go")
        a = len(page.text_content("#output"))
        time.sleep(1.2)
        b = len(page.text_content("#output"))
        assert a == b, "pause did not stop generation"
        print("  pause works")

        # layer tabs re-render the strips
        page.locator("#tabs button", has_text="L2").click()
        heads_l2 = page.locator("#heads .headrow").count()
        assert heads_l2 == 4, f"expected 4 rows for single layer, got {heads_l2}"
        print("  layer tabs work")

        # presets restart generation with a new prompt
        page.locator("#presets button", has_text="JULIET:").click()
        time.sleep(1.5)
        out2 = page.text_content("#output")
        assert out2.startswith("JULIET:"), "preset did not restart from new prompt"
        print("  presets work")

        if args.screenshot:
            os.makedirs(os.path.dirname(os.path.join(ROOT, args.screenshot)), exist_ok=True)
            page.locator("#tabs button", has_text="mean").click()
            time.sleep(min(args.seconds + 4, 14))  # let it write enough for the photo
            page.screenshot(path=os.path.join(ROOT, args.screenshot))
            print(f"  screenshot -> {args.screenshot}")

        browser.close()

    if errors:
        print("CONSOLE ERRORS:\n" + "\n".join(errors))
        sys.exit(1)
    print("UI TEST PASSED — every panel alive, zero console errors")


if __name__ == "__main__":
    main()

"""How much space is free? Builds the paper and prints the fit checks (added in the rewrite round, W0).

  - pages of the two-column check and of the reserve check (the two-column check with the Step 2 slot)
  - if the two-column check is over six pages: the overflow, as text lines on pages 7 and later
  - if it fits: the tallest slot (steps of 0.02 textheight, one column wide) that still fits in six pages
  - the abstract's word count (build_paper.plain_abstract)
  - PASS only if both checks have <= 6 pages and the abstract has <= 200 words
Run after paper_numbers.py. Writes results/slack.json. --no-build reuses the PDFs already built.
"""
import json
import os
import shutil
import sys

import common as C
import build_paper as B
from pdf_tools import pdf_info

MAXP, MAXW, STEP, SLOT = 6, 200, 0.02, B.SLOT_DEFAULT


def overflow_lines(pdf):
    from pypdf import PdfReader
    r = PdfReader(pdf)
    return sum(len([ln for ln in (p.extract_text() or "").splitlines() if ln.strip()]) for p in r.pages[MAXP:])


def main():
    if "--no-build" not in sys.argv:
        B.fill()
        for m, pdf in (("main_single.tex", "AIR2027_submission.pdf"), ("main_twocol.tex", "AIR2027_twocolumn_check.pdf")):
            B.latex(m)
            shutil.copy(os.path.join(B.BUILD, m[:-4] + ".pdf"), os.path.join(B.PAPER, pdf))
    B.plain_abstract()
    words = len(open(os.path.join(B.PAPER, "ABSTRACT.txt")).read().split("ABSTRACT\n", 1)[1].split("\n\nKEYWORDS")[0].split())
    two = pdf_info(os.path.join(B.PAPER, "AIR2027_twocolumn_check.pdf"))["pages"]
    rpages, _ = B.build_reserve(SLOT)
    out = {"twocolumn_pages": two, "reserve_pages": rpages, "slot_height": SLOT, "abstract_words": words}
    print(f"two-column check: {two} pages")
    print(f"reserve check (slot {SLOT:.2f} textheight): {rpages} pages")
    if two > MAXP:
        out["overflow_lines"] = overflow_lines(os.path.join(B.PAPER, "AIR2027_twocolumn_check.pdf"))
        out["free_slot"] = None
        print(f"OVER by {out['overflow_lines']} text lines on pages {MAXP + 1}+")
    else:
        lo, hi = 0, 50          # slot = k*STEP; find the largest k that fits (k=0: nothing fits)
        cache = {}
        def fits(k):
            if k not in cache:
                cache[k] = B.build_reserve(round(k * STEP, 2), quick=True)[0] <= MAXP
            return cache[k]
        if fits(1):
            lo = 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                lo, hi = (mid, hi) if fits(mid) else (lo, mid)
        else:
            lo = 0
        out["free_slot"] = round(lo * STEP, 2)
        print(f"free space: tallest slot that fits in {MAXP} pages = {out['free_slot']:.2f} textheight"
              + (" (none)" if lo == 0 else ""))
    print(f"abstract: {words} words (limit {MAXW})")
    ok = two <= MAXP and rpages <= MAXP and words <= MAXW
    out["pass"] = ok
    json.dump(out, open(os.path.join(C.RESULTS, "slack.json"), "w"), indent=1)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

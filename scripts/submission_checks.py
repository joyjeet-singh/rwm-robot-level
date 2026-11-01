"""
Submission checks for both PDFs: anonymity (text and metadata), page counts, fonts embedded, no unresolved
references. Writes results/submission_checks.json; exits non-zero on a failure.
AIR 2027 call: single-column initial submission, no page limit; camera-ready at most 6 two-column pages;
strict double-blind ("author names and identities must not appear anywhere in the submitted paper").
"""
import json
import os
import re
import subprocess
import sys

import common as C

PAPER = os.path.join(C.AIR, "paper")
# identifying strings: the author's name, handles, place, accounts, and the project's public locations
IDENT = [r"joyjeet", r"\bsingh\b", r"chandigarh", r"punjab", r"gmail", r"github\.com", r"huggingface",
         r"orcid", r"0009-0005", r"testingrobo", r"manukau", r"joyjeetsingh", r"rwm-reproduction"]


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def check(pdf):
    p = os.path.join(PAPER, pdf)
    text = run(["pdftotext", "-layout", p, "-"])
    info = run(["pdfinfo", p])
    fonts = run(["pdffonts", p]).splitlines()[2:]
    hits = sorted({m for pat in IDENT for m in re.findall(pat, text + info, flags=re.I)})
    meta = {k.strip(): v.strip() for k, v in (ln.split(":", 1) for ln in info.splitlines() if ":" in ln)}
    not_embedded = [ln for ln in fonts if ln.split()[-5:-4] == ["no"]]
    return {"pages": int(meta.get("Pages", 0)), "identifying_strings": hits,
            "metadata": {k: meta.get(k, "") for k in ("Title", "Author", "Creator", "Producer")},
            "fonts_not_embedded": not_embedded, "question_marks_refs": len(re.findall(r"\[\?\]|\?\?", text))}


if __name__ == "__main__":
    out = {"single": check("AIR2027_submission.pdf"), "twocol": check("AIR2027_twocolumn_check.pdf")}
    fail = []
    for k, v in out.items():
        if v["identifying_strings"]:
            fail.append(f"{k}: identifying strings {v['identifying_strings']}")
        if v["metadata"]["Author"]:
            fail.append(f"{k}: author metadata")
        if v["fonts_not_embedded"]:
            fail.append(f"{k}: fonts not embedded")
        if v["question_marks_refs"]:
            fail.append(f"{k}: unresolved references")
    if out["twocol"]["pages"] > 6:
        fail.append(f"two-column build has {out['twocol']['pages']} pages (> 6)")
    out["failures"] = fail
    json.dump(out, open(os.path.join(C.RESULTS, "submission_checks.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    sys.exit(1 if fail else 0)

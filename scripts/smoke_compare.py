"""Compare a smoke-test copy (.cache/smoke) with the stored results.

Usage: ./py scripts/smoke_compare.py [smoke_dir]
JSON: numbers equal within 1e-9 relative, strings exactly; for paper_numbers.json every `value`.
PDFs: page counts, and pdftotext text with whitespace normalised.
Exit status 1 if any file differs by more than 1e-9 relative.
"""
import json, os, sys

import pdf_tools

HERE = os.path.dirname(os.path.abspath(__file__))
AIR = os.path.dirname(HERE)
REL = 1e-9
worst = {}


def walk(a, b, path, diffs, key):
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        if a != b:
            diffs.append((path, a, b))
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        d = abs(a - b)
        r = d / max(abs(a), abs(b), 1e-300) if d else 0.0
        worst[key] = max(worst.get(key, 0.0), r)
        if r > REL and d > 1e-300:
            diffs.append((path, a, b))
    elif isinstance(a, str) and isinstance(b, str):
        if a != b:
            diffs.append((path, a[:60], b[:60]))
    elif isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                diffs.append((f"{path}/{k}", "missing" if k not in a else "present", "missing" if k not in b else "present"))
            else:
                walk(a[k], b[k], f"{path}/{k}", diffs, key)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append((path, f"len {len(a)}", f"len {len(b)}"))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}[{i}]", diffs, key)
    else:
        diffs.append((path, type(a).__name__, type(b).__name__))


def values_only(d):
    """paper_numbers.json: compare each entry's `value` only."""
    if isinstance(d, dict) and "value" in d and not isinstance(d["value"], dict):
        return d["value"]
    if isinstance(d, dict):
        return {k: values_only(v) for k, v in d.items()}
    return d


def main():
    smoke = sys.argv[1] if len(sys.argv) > 1 else os.path.join(AIR, ".cache", "smoke")
    bad = 0
    res = os.path.join(smoke, "results")
    for f in sorted(os.listdir(res)):
        if not f.endswith(".json"):
            continue
        a_path, b_path = os.path.join(AIR, "results", f), os.path.join(res, f)
        if not os.path.exists(a_path):
            print(f"{f}: new in smoke copy")
            continue
        a, b = json.load(open(a_path)), json.load(open(b_path))
        if f == "paper_numbers.json":
            a, b = values_only(a), values_only(b)
        diffs = []
        walk(a, b, "", diffs, f)
        status = "IDENTICAL" if not diffs else f"{len(diffs)} DIFFERENCES"
        print(f"{f}: {status} (max rel diff {worst.get(f, 0.0):.2e})")
        for d in diffs[:5]:
            print("   ", d)
        bad += bool(diffs)
    for rel in ("paper/AIR2027_submission.pdf", "paper/AIR2027_twocolumn_check.pdf"):
        a, b = os.path.join(AIR, rel), os.path.join(smoke, rel)
        if not os.path.exists(b):
            print(f"{rel}: MISSING in smoke copy")
            bad += 1
            continue
        ia, ib = pdf_tools.pdf_info(a), pdf_tools.pdf_info(b)
        pa, pb = ia['pages'], ib['pages']
        same_text = pdf_tools.norm(ia['text']) == pdf_tools.norm(ib['text'])
        # pypdf (no poppler) places word gaps differently for different TeX builds: also compare with all spaces removed
        same_chars = pdf_tools.nospace(ia['text']) == pdf_tools.nospace(ib['text'])
        kind = 'identical' if same_text else ('identical apart from word spacing (extraction)' if same_chars else 'DIFFERENT')
        print(f"{rel}: pages {pa} vs {pb}; text {kind}")
        bad += (pa != pb) or not same_chars
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

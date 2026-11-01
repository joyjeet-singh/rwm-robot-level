"""
Every entry of paper/refs.bib must come from a bibliography an earlier study checked against its source record:
same key in that study's bib_verified.json, same title (case and punctuation ignored) in its refs.bib.
Every entry must be cited, and every citation must have an entry. -> results/bib_check.json
"""
import json
import os
import re
import sys

import common as C

PAPER = os.path.join(C.AIR, "paper")
STUDIES = {s: (os.path.join(C.PDM, s, "results", "bib_verified.json"), os.path.join(C.PDM, s, "paper", "refs.bib"))
           for s in ("PerconAI", "ICPAIAS", "NGCAPS")}
ALIASES = {"janner2019": ("janner2019mbpo",)}           # NGCAPS's key for the same paper


def entries(path):
    txt = open(path).read()
    out = {}
    for chunk in re.split(r"\n(?=@)", "\n" + txt):
        m = re.match(r"@\w+\{([^,]+),", chunk.strip())
        if not m:
            continue
        t = re.search(r"\btitle\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}", chunk, flags=re.S)
        out[m.group(1).strip()] = t.group(1) if t else ""
    return out


def norm(t):
    return re.sub(r"[^a-z0-9]", "", t.lower().replace("\\\"", "").replace("{", "").replace("}", ""))


def verified_keys(path):
    d = json.load(open(path))
    if "entries" in d:
        e = d["entries"]
        return set(e) if isinstance(e, dict) else {x.get("key") for x in e}
    return {k for k in d if k != "note"}


ours = entries(os.path.join(PAPER, "refs.bib"))
src = {s: (verified_keys(v), entries(b)) for s, (v, b) in STUDIES.items()}
report, bad = {}, []
for key, title in ours.items():
    found = []
    for s, (vk, bib) in src.items():
        for k in (key,) + ALIASES.get(key, ()):
            if k in vk and k in bib and norm(bib[k]) == norm(title):
                found.append(f"{s}:{k}")
    report[key] = found
    if not found:
        bad.append(key)
body = open(os.path.join(PAPER, "body.template.tex")).read()
cited = set()
for m in re.finditer(r"\\cite[pt]?\{([^}]*)\}", body):
    cited |= {c.strip() for c in m.group(1).split(",")}
out = {"entries": report, "not_traced": bad, "cited_without_entry": sorted(cited - set(ours)),
       "entry_never_cited": sorted(set(ours) - cited)}
json.dump(out, open(os.path.join(C.RESULTS, "bib_check.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "entries"}, indent=1))
sys.exit(1 if (bad or out["cited_without_entry"]) else 0)

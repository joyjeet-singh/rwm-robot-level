"""Build submission/AIR2027_supplement.zip (Step 1, session S7).

Rebuilds the round-1 supplement (prereg/, scripts/, results/, paper templates + refs.bib, README_SUPPLEMENT.md) and adds
the Step 1 material: prereg/PREREG_S1.md, the s1_* scripts and test_s1.py, scripts/lite/ (stubs included), the new result
files (results/s1_*, raw_S1.npz, results/s1/), and a Step 1 section in README_SUPPLEMENT.md.

Anonymity: files are copied to .cache/supp_stage/ first. The few lines that name a person, a handle or a local path are
neutralised in that copy only (the working files are untouched). Then the staged copy is scanned for the identifying
patterns of scripts/submission_checks.py plus "/Users/", "Downloads" and "joyjeetsingh"; any hit left stops the build.
The three registration files are copied byte for byte (their hashes are evidence); "Downloads" is allowed in them
because it is a generic folder name, as in round 1.

Never included: this script, history.bundle, .git, runs/, .cache/, .pylib/, paper/build/, __pycache__, *.pyc, *.lock, .DS_Store.

    ./py scripts/build_supplement.py            full zip (about 70 MB); builds, scans it, unzips it into .cache/supp_check
    ./py scripts/build_supplement.py --slim     same without any raw_*.npz (small), submission/AIR2027_supplement_slim.zip

results/s1/raw/ (391 per-model files) is left out of both: every array in it is in results/raw_S1.npz.
"""
import json, re, shutil, sys, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGE = ROOT / ".cache" / "supp_stage"
CHECK = ROOT / ".cache" / "supp_check"
OUT = ROOT / "submission" / ("AIR2027_supplement_slim.zip" if "--slim" in sys.argv else "AIR2027_supplement.zip")
LIMIT_MB = 80  # round 1 was 33 MB; the full Step 1 zip is about 70 MB (raw_S1.npz alone is 38 MB)
SLIM = "--slim" in sys.argv

SKIP_NAMES = {".DS_Store", "__pycache__"}
SKIP_SUFFIX = {".pyc", ".lock"}
NOT_PACKED = {"scripts/build_supplement.py"}  # holds the scan patterns (names, paths); packaging tool, not analysis code
BYTE_EXACT = {"prereg/PREREG.md", "prereg/PREREG_FT.md", "prereg/PREREG_S1.md"}

# identifying patterns: scripts/submission_checks.py IDENT (read below) plus these three
EXTRA_PATTERNS = [r"/Users/", r"Downloads", r"joyjeetsingh"]


def ident_patterns():
    src = (ROOT / "scripts" / "submission_checks.py").read_text()
    m = re.search(r"^IDENT = \[(.*?)\]", src, flags=re.S | re.M)
    return [eval(s) for s in re.findall(r'r"[^"]*"', m.group(1))]


def want(rel: Path) -> bool:
    r = rel.as_posix()
    if r.startswith("results/s1/raw/"):
        return False  # per-model pieces; every array in them is also in results/raw_S1.npz (checked in S7)
    if SLIM and rel.suffix == ".npz":
        return False  # slim variant: no raw arrays at all
    return not (set(rel.parts) & SKIP_NAMES or rel.suffix in SKIP_SUFFIX)


def collect():
    files = []
    for p in sorted((ROOT / "prereg").glob("*.md")):
        files.append(p)
    for p in sorted((ROOT / "scripts").rglob("*")):
        if p.is_file() and want(p.relative_to(ROOT)):
            files.append(p)
    for p in sorted((ROOT / "results").rglob("*")):
        if p.is_file() and want(p.relative_to(ROOT)):
            files.append(p)
    for n in ("refs.bib", "appendix.template.tex", "body.template.tex"):
        files.append(ROOT / "paper" / n)
    return [p.relative_to(ROOT) for p in files if p.relative_to(ROOT).as_posix() not in NOT_PACKED]


# (file, regex, replacement, what): applied to the staged copy only
NEUTRALISE = [
    ("scripts/submission_checks.py", r'IDENT = \[.*?\]\n',
     'IDENT = [r"<author name and handle patterns removed from this copy>", r"github\\.com", r"huggingface", r"orcid"]\n',
     "author-name patterns in scripts/submission_checks.py"),
    ("scripts/stage_out.py", r'DEVICE = "[^"]*"', 'DEVICE = "<path of the working copy>"',
     "a local path in scripts/stage_out.py"),
    ("scripts/common.py", r'PDM_ROOT=~/Downloads/PDM\.', 'PDM_ROOT=<the PDM folder>.',
     "a path in a comment of scripts/common.py"),
]


def reg_facts():
    pn = json.load(open(ROOT / "results" / "paper_numbers.json"))
    return pn["s1_reg_commit"]["value"], pn["s1_reg_time"]["value"]


def readme(commit, when, changed):
    return f"""# Supplement for reviewers

Anonymous copy of the material behind the paper. Nothing here is needed to read the paper.

| Folder | Holds |
|---|---|
| `prereg/` | `PREREG.md` (the main study's questions and decision rules), `PREREG_FT.md` (study FT) and `PREREG_S1.md` (studies CL and AR), each written before the data it judges existed |
| `scripts/` | all measurement, statistics, table and check code (`common.py` holds shared helpers); `scripts/lite/` runs the lite policy training on a CPU, with a stub module in place of the logging service |
| `results/` | every result file the paper's numbers are read from, including `raw_*.npz` (kept) and `results/s1/` (progress and baseline-training records; its per-model array files are left out because `raw_S1.npz` holds all of them) |
| `paper/` | the paper's source templates (every number is a `{{{{key}}}}` placeholder) and `refs.bib` |

{"**This is the slim copy: no `raw_*.npz` file is in it** (the full copy has them; the other result files are the same)." if SLIM else ""}

## Run order (from `scripts/`)
`checks.py`, `compute.py F18 X5 REL`, `stats.py`, `check_numbers.py`, `exploratory.py`, then the revision-round
analyses (`revision_analyses.py`, `ft_eval.py`, `ft_stats.py`, `revision_numbers.py`), then the Step 1 analyses
(`s1_train_baseline.py`, `s1_eval.py`, `s1_stats.py`, `s1_budget.py`, `s1_extra.py`, `s1_numbers.py`; `test_s1.py` checks the
code on random-weight models), `make_figures.py`, `check_bib.py`, `paper_numbers.py`, `build_paper.py`, `check_claims.py`,
`submission_checks.py`.
The scripts need the authors' model checkpoints and earlier-study files, which are not included; the
result files in `results/` are what the paper reads. Stored results were computed with Python 3.13, torch 2.15 and numpy 2.5;
the Step 1 job ran on a CPU-only laptop with Python 3.11 and torch 2.2.

## Registration times (UTC)
- `PREREG.md` committed 2026-11-01 08:17:57; measurement code 08:18:58; analysis code 08:21:35; first registered data (`raw_F18.npz`) 08:31:41 (`results/timeline.json`).
- `PREREG_FT.md` and `ft_stats.py` committed 2026-11-01 16:35:59, before any fine-tuned model existed (git commit time).

## Step 1 (studies CL and AR; study TB)
- Study CL (a policy in the loop) and study AR (other simulator architectures, 12 newly trained models) are pre-registered
  in `prereg/PREREG_S1.md`, committed {when} (git commit `{commit}`), before any simulator was driven by the policy's
  joint commands and before any of the newly trained simulators was scored. The statistics are `scripts/s1_stats.py`,
  committed with it and run unchanged; hashes are in `results/prereg_hash_s1.json`.
- Result files: `results/s1_models.json` (per-model measurements), `results/s1_stats.json` and `results/s1_report.txt` (the
  registered verdicts), `results/raw_S1.npz` (per-trajectory arrays) and `results/s1/` (job records), `results/s1_probe.json` (timing probes).
- Study TB (training length) is descriptive, not registered: `scripts/s1_budget.py` reads stored files only, writes
  `results/s1_budget.json` and `results/s1_budget.txt`. `scripts/s1_extra.py` (`results/s1_extra.json`) holds exploratory
  additions written after the registered analysis had run.
- `scripts/lite/` is the authors' lite policy training made to run on a CPU (`run_lite.py`, `adapter.py`, `test_lite.py`,
  `stubs/`); it is engineering and its output is not a result of the paper.

## Edits made for anonymity
In this copy only: {changed}. The three registration files are byte-identical to the registered files
(their hashes are evidence); `prereg/PREREG.md` mentions a generic download folder path, with no name.
"""


def stage():
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    rels = collect()
    for rel in rels:
        dst = STAGE / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst)
    done = []
    for rel, rx, repl, what in NEUTRALISE:
        f = STAGE / rel
        t = f.read_text()
        t2, n = re.subn(rx, lambda m: repl, t, count=1, flags=re.S)
        if n == 1 and t2 != t:
            f.write_text(t2)
            done.append(what)
    commit, when = reg_facts()
    (STAGE / "README_SUPPLEMENT.md").write_text(readme(commit, when, "; ".join(done) or "none needed"))
    return rels, done


def scan(base: Path, label: str):
    pats = [re.compile(p, re.I) for p in ident_patterns() + EXTRA_PATTERNS]
    hits = []
    for f in sorted(base.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(base).as_posix()
        data = f.read_bytes()
        text = data.decode("utf-8", "ignore")  # binary files: scan their readable bytes too
        for p in pats:
            if p.pattern == "Downloads" and rel in BYTE_EXACT:
                continue
            if rel == "scripts/submission_checks.py" and p.pattern in ("huggingface", "orcid", r"github\.com"):
                continue  # the checker's own generic patterns, kept in the copy as in round 1
            if p.search(text) or p.search(rel):
                hits.append((rel, p.pattern))
    print(f"scan ({label}): {len(hits)} hit(s)")
    for h in hits[:20]:
        print("  HIT", h)
    return hits


def main():
    rels, done = stage()
    for rel in BYTE_EXACT:
        assert (STAGE / rel).read_bytes() == (ROOT / rel).read_bytes(), rel + " not byte-identical"
    hits = scan(STAGE, "staged copy")
    if hits:
        sys.exit("BLOCKED: identifying text left in the staged copy; extend NEUTRALISE and rerun")
    OUT.parent.mkdir(exist_ok=True)
    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted(STAGE.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(STAGE).as_posix())
    mb = OUT.stat().st_size / 1e6
    print(f"wrote {OUT.relative_to(ROOT)}: {mb:.1f} MB, {len(list(STAGE.rglob('*')))} entries; neutralised: {done}")
    if mb > LIMIT_MB:
        sys.exit(f"BLOCKED: zip is {mb:.1f} MB, over the {LIMIT_MB} MB kept in round 1")
    if CHECK.exists():
        shutil.rmtree(CHECK)
    CHECK.mkdir(parents=True)
    with zipfile.ZipFile(OUT) as z:
        bad = z.testzip()
        assert bad is None, bad
        z.extractall(CHECK)
    names = {p.relative_to(CHECK).as_posix() for p in CHECK.rglob("*") if p.is_file()}
    for banned in ("history.bundle",):
        assert not any(banned in n for n in names)
    assert not any(n.startswith(".git/") or "/.git/" in n or "__pycache__" in n for n in names)
    hits = scan(CHECK, "unzipped zip")
    if hits:
        sys.exit("BLOCKED: identifying text in the zip")
    print("zip scan clean")


if __name__ == "__main__":
    main()

"""
Job S3 (STEP1_PLAN.md S3; prereg/PREREG_S1.md): the measurements of studies CL and AR. No statistic is computed here.

  ./py scripts/s1_eval.py              the registered job (run by step1/jobs/S3.zsh)
  ./py scripts/s1_eval.py --dry-run    a code check on random-weight models, a few of each kind, into .cache/s1_dry/

Joint commands are the 12 joint position targets given at every step (the actions); the velocity command is the
walking speed and turning rate the policy is asked to follow.

Four phases, in order:
  1 checks    test_s1 tests (i), (vi) and (ix) -> checks S1-C2, S1-C3 and S1-C5
  2 CL        s1_rollout.score_gru on F18's 97 simulators and X5's 255, each on its own arena, and on the released
              model (reference only, all 20 trajectories). Every item (policy, closed-loop derivative, noise) is
              computed; the recorded-command values are recomputed for all F18 models, every fifth X5 model in table
              order, and the released model, for check S1-C1.
  3 training  the 12 new runs (seeds 3 and 4 of tf_s7 and ar_s7 for the MLP, RSSM and Transformer) with
              s1_train_baseline.train, one at a time at 2 threads (results/s1_probe.json S2: two at a time would
              finish 16% sooner, below the registered 30%). Skipped if S1-C5 failed.
  4 AR        s1_arch.score_arch on every baseline run (the reproduction's 26 and the new 12), final weights, F18's
              arena -> check S1-C4 (every model of every pool scored; runs with a non-finite training loss or
              non-finite weights excluded and listed). Then results/raw_S1.npz is assembled.

Outputs: results/s1_models.json (software, checks, per-model means and counts, training summaries),
results/s1/raw/<tag>/<key>.npz (per-model arrays, written as each model finishes), results/raw_S1.npz (all of
them, keys "<tag>__<key>__<field>"), results/s1/progress.json (the job's progress; only this script writes it).

Resumable: a model already in s1_models.json is skipped; a run whose final weights and record exist is not
retrained. An interrupted training run is moved to runs/s1_interrupted/ and restarts from iteration 0, so it is
identical to an uninterrupted run. Progress is written only from the work itself (after each model, test and every
50 training iterations), never from a timer.
"""
import argparse
import datetime
import glob
import json
import os
import platform
import shutil
import sys
import time
import traceback

import numpy as np
import torch

import common as C
import compute
import s1_arch as A
import s1_policy as SP
import s1_rollout as S
import stats as ST

torch.set_num_threads(2)

NEW_RUNS = [(arch, regime, "s7", seed) for arch in ("mlp", "transformer", "rssm") for regime in ("tf", "ar")
            for seed in (3, 4)]
REPRO_RUNS = os.path.join(C.RWM, "runs")
C1_EVERY_X5 = 5
GROUP_ONE_STEP = {"B (teacher forced)", "B 10k iterations", "B"}


# ------------------------------------------------------------------ small helpers
def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def dump_atomic(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def model_mean(v):
    """A model's value: the mean over its trajectories; one non-finite trajectory makes it +inf (ranked worst)."""
    v = np.asarray(v, dtype=np.float64)
    m = float(np.mean(v)) if np.all(np.isfinite(v)) else float("inf")
    return m


def software():
    import scipy
    import tensordict
    return {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
            "scipy": scipy.__version__, "tensordict": tensordict.__version__, "platform": platform.platform(),
            "torch_threads": torch.get_num_threads()}


class Job:
    def __init__(self, dry=False):
        self.dry = dry
        self.root = os.path.join(C.AIR, ".cache", "s1_dry") if dry else C.RESULTS
        self.models_json = os.path.join(self.root, "s1_models.json")
        self.raw_dir = os.path.join(self.root, "s1", "raw")
        self.progress_json = os.path.join(self.root, "s1", "progress.json")
        self.raw_npz = os.path.join(self.root, "raw_S1.npz")
        self.runs_dir = os.path.join(C.AIR, ".cache", "s1_dry", "runs") if dry else os.path.join(C.AIR, "runs",
                                                                                                  "s1_baselines")
        self.interrupted = os.path.join(C.AIR, ".cache", "s1_dry", "interrupted") if dry else \
            os.path.join(C.AIR, "runs", "s1_interrupted")
        if os.path.exists(self.models_json):
            self.db = json.load(open(self.models_json))
        else:
            self.db = {"purpose": "Step 1 job S3: measurements of studies CL and AR (prereg/PREREG_S1.md)",
                       "dry_run": dry, "created": now(), "software": software(), "checks": {},
                       "models": {"F18": {}, "X5": {}, "REL": {}, "AR": {}}, "training": {}, "phases": {}}
        self.db["software_last_session"] = software()
        self.t0 = time.time()

    def save(self):
        self.db["updated"] = now()
        dump_atomic(self.db, self.models_json)

    def progress(self, phase, item, done, left, **extra):
        dump_atomic({"phase": phase, "item": item, "done": int(done), "left": int(left), "time": now(),
                     "seconds_since_start": round(time.time() - self.t0, 1), **extra}, self.progress_json)

    def save_raw(self, tag, key, arrays):
        d = os.path.join(self.raw_dir, tag)
        os.makedirs(d, exist_ok=True)
        tmp = os.path.join(d, key + ".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, os.path.join(d, key + ".npz"))


# ------------------------------------------------------------------ phase 1: checks
def phase_checks(job, skip_tests=False):
    import test_s1 as T
    names = {"i": "S1-C2", "vi": "S1-C3", "ix": "S1-C5"}
    todo = [t for t in ("i", "vi", "ix") if names[t] not in job.db["checks"]]
    for n, t in enumerate(todo):
        job.progress("1 checks", f"test ({t})", n, len(todo) - n)
        if skip_tests:
            job.db["checks"][names[t]] = {"pass": True, "detail": "dry run: test not run", "dry_run": True}
            job.save()
            continue
        T.RESULTS.clear()
        t1 = time.time()
        try:
            T.TESTS[t]()
            res = {k: {"pass": ok, "detail": det} for k, (ok, det) in T.RESULTS.items()}
            ok = bool(res) and all(v["pass"] for v in res.values())
        except Exception:
            res, ok = {"exception": {"pass": False, "detail": traceback.format_exc()[-2000:]}}, False
        torch.set_num_threads(2)
        job.db["checks"][names[t]] = {"pass": ok, "test": f"test_s1 ({t})", "results": res,
                                      "seconds": time.time() - t1, "time": now()}
        job.save()
        print(f"  {names[t]} (test {t}): {'PASS' if ok else 'FAIL'} in {time.time() - t1:.0f}s", flush=True)
    job.progress("1 checks", "done", len(todo), 0)


# ------------------------------------------------------------------ phase 2: study CL
def randomise(m, seed):
    """Dry run only: every floating parameter replaced by random values of the same shape (no trained weights)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            if torch.is_floating_point(p):
                p.copy_(torch.randn(p.shape, generator=g, dtype=p.dtype) * 0.05)
    return m


def fold_of(arena_id):
    return "all" if arena_id == "all" else (int(arena_id[0]), int(arena_id[1]))


def cl_entries(dry):
    meta = json.load(open(os.path.join(C.RESULTS, "models.json")))
    out = []
    for tag in ("F18", "X5", "REL"):
        for i, m in enumerate(meta[tag]):
            rec = tag in ("F18", "REL") or i % C1_EVERY_X5 == 0
            out.append((tag, i, m, rec))
    if dry:     # a few of each kind: trained, compressed (structured and not), five heads, every-fifth X5
        pick = {("F18", 0), ("F18", 57), ("F18", 97 - 1), ("X5", 0), ("X5", 1)}
        f18 = meta["F18"]
        pick |= {("F18", next(i for i, m in enumerate(f18) if m["ensemble"] == 5))}
        out = [e for e in out if (e[0], e[1]) in pick]
    return out


def cl_group(m):
    if m["variant"] is not None:
        return "compressed"
    return "one-step" if m["group"] in GROUP_ONE_STEP else "rollout"


def phase_cl(job):
    entries = cl_entries(job.dry)
    todo = [e for e in entries if e[2]["key"] not in job.db["models"][e[0]]]
    actor = SP.load_actor()
    for n, (tag, i, m, rec) in enumerate(todo):
        key = m["key"]
        job.progress("2 CL", f"{tag} {key}", len(entries) - len(todo) + n, len(todo) - n)
        t1 = time.time()
        ar, sc = compute.setup_arena(fold_of(m["arena"]))
        model, _ = compute.build(os.path.join(C.PDM, m["path"]), m["ensemble"], m["variant"])
        if job.dry:
            randomise(model, i)
        items = S.ITEMS if rec else tuple(k for k in S.ITEMS if k != "rec")
        out = S.score_gru(model, ar, sc, items=items, p=1, actor=actor)
        sec = out.pop("seconds")
        job.save_raw(tag, key, out)
        e = {"index": i, "pool": m["pool"], "group": m["group"], "cl_group": cl_group(m), "variant": m["variant"],
             "arena": m["arena"], "ensemble": m["ensemble"], "n_traj": int(len(ar["starts"])),
             "sha256_parent": m["sha256"], "recomputed_rec": bool(rec)}
        for k in ("lam1_pol", "rJ_pol", "lam1_cl", "lam1_noise", "rJ_noise") + (("lam1_rec", "rJ_rec") if rec else ()):
            e[k] = model_mean(out[k])
        e["ret"] = float(np.mean(out["ret"]))
        e["ret_unc"] = float(np.mean(out["ret_unc"]))
        e["ret_terms"] = [float(x) for x in np.mean(out["ret_terms"], 0)]
        e["n_term_steps"] = int(np.sum(out["n_term"]))
        for k in ("div_pol", "div_cl", "div_noise"):
            e["n_" + k] = int(np.sum(np.asarray(out[k]) > 0))
        e["seconds"] = {k: float(v) for k, v in sec.items()}
        e["seconds_total"] = time.time() - t1
        job.db["models"][tag][key] = e
        job.save()
        print(f"  CL {tag} {key}: {time.time() - t1:.1f}s; diverged pol/cl/noise "
              f"{e['n_div_pol']}/{e['n_div_cl']}/{e['n_div_noise']}", flush=True)
    check_c1(job)
    job.progress("2 CL", "done", len(entries), 0)


def check_c1(job):
    """S1-C1: recomputed recorded-command lambda_1 and r_J against the stored registered values (model_table.json)."""
    table = json.load(open(os.path.join(C.RESULTS, "model_table.json")))
    res, ok_all = {}, True
    for tag in ("F18", "X5"):
        stored = {r["key"]: r for r in table[tag]}
        done = {k: v for k, v in job.db["models"][tag].items() if v["recomputed_rec"]}
        n_expected = len(table[tag]) if tag == "F18" else len(range(0, len(table[tag]), C1_EVERY_X5))
        r = {"n_expected": n_expected, "n_checked": len(done)}
        for chk, sk, nk, kind in (("lam1", "lam1", "lam1_rec", "abs"), ("rJ", "rJ", "rJ_rec", "rel")):
            a = np.array([done[k][nk] for k in done], dtype=float)
            b = np.array([stored[k][sk] for k in done], dtype=float)
            both_bad = ~np.isfinite(a) & ~np.isfinite(b)
            with np.errstate(invalid="ignore"):
                diff = np.abs(a - b) if kind == "abs" else np.abs(a - b) / np.maximum(np.abs(b), 1e-12)
            within = both_bad | (np.isfinite(diff) & (diff <= 1e-3))
            rho = ST.spearman(a, b) if len(a) > 2 else float("nan")
            frac = float(within.mean()) if len(a) else 0.0
            r[chk] = {"tolerance": "1e-3 per second (absolute)" if kind == "abs" else "1e-3 relative",
                      "fraction_within": frac, "spearman_recomputed_vs_stored": rho,
                      "max_difference": float(np.nanmax(diff)) if np.isfinite(diff).any() else None,
                      "outside": [k for k, w in zip(done, within) if not w],
                      "pass": bool(frac >= 0.95 and np.isfinite(rho) and rho >= 0.999)}
            ok_all &= r[chk]["pass"]
        ok_all &= r["n_checked"] == r["n_expected"]
        res[tag] = r
    job.db["checks"]["S1-C1"] = {"pass": bool(ok_all), "per_pool": res, "time": now(),
                                 "rule": ("for >= 95% of the checked models |d lam1| <= 1e-3 /s and |d rJ| <= 1e-3 "
                                          "relative (both non-finite = agreement), and Spearman(recomputed, stored) "
                                          ">= 0.999, for each pool and check; all F18 models and every fifth X5 "
                                          "model in table order checked")}
    job.save()
    print(f"  S1-C1: {'PASS' if ok_all else 'FAIL'}", flush=True)


# ------------------------------------------------------------------ phase 3: training
def run_dir_new(job, arch, regime, spec, seed):
    import s1_train_baseline as TB
    return os.path.join(job.runs_dir, TB.run_name(arch, regime, spec, seed))


def run_complete(rd, iters):
    name = os.path.basename(rd)[len("s1_baseline_"):]
    return os.path.exists(os.path.join(rd, f"weights_{iters}.pt")) and \
        os.path.exists(os.path.join(rd, f"baseline_run_{name}.json"))


def phase_training(job):
    import s1_train_baseline as TB
    iters = 2 if job.dry else 2500
    if not job.db["checks"].get("S1-C5", {}).get("pass", False):
        job.db["phases"]["training"] = "skipped: S1-C5 failed or missing; AR uses the existing runs only"
        job.save()
        print("  training skipped (S1-C5)", flush=True)
        return
    runs = NEW_RUNS if not job.dry else NEW_RUNS[:2] + NEW_RUNS[4:5] + NEW_RUNS[8:9]
    if not job.dry:     # the copy's default output folder is the one this job reads
        assert os.path.abspath(TB.RUNS_DIR) == os.path.abspath(job.runs_dir), TB.RUNS_DIR
    for n, (arch, regime, spec, seed) in enumerate(runs):
        rd = run_dir_new(job, arch, regime, spec, seed)
        name = os.path.basename(rd)
        if run_complete(rd, iters):
            if name not in job.db["training"]:
                job.db["training"][name] = training_record(rd, iters, recovered=True)
                job.save()
            continue
        if os.path.exists(rd):          # interrupted: set aside, restart from iteration 0
            os.makedirs(job.interrupted, exist_ok=True)
            shutil.move(rd, os.path.join(job.interrupted, f"{name}_{datetime.datetime.now():%Y%m%d-%H%M%S}"))
            job.db.setdefault("interrupted_runs", []).append({"run": name, "time": now()})
            job.save()
        job.progress("3 training", name, n, len(runs) - n, iteration=0, iterations=iters)
        t1 = time.time()

        def prog(run, it, its, loss, secs, n=n):
            job.progress("3 training", run, n, len(runs) - n, iteration=it, iterations=its,
                         last_total_loss=float(loss), seconds_run=round(secs, 1))

        TB.train(arch, regime, spec, seed, iters=iters, out_dir=job.runs_dir if job.dry else None, threads=2,
                 progress=prog)
        torch.set_num_threads(2)
        job.db["training"][name] = training_record(rd, iters)
        job.save()
        print(f"  trained {name}: {time.time() - t1:.0f}s; final loss finite "
              f"{job.db['training'][name]['final_loss_finite']}", flush=True)
    job.progress("3 training", "done", len(runs), 0)


def training_record(rd, iters, recovered=False):
    name = os.path.basename(rd)[len("s1_baseline_"):]
    rec = json.load(open(os.path.join(rd, f"baseline_run_{name}.json")))
    sd = torch.load(os.path.join(rd, f"weights_{iters}.pt"), map_location="cpu", weights_only=False)
    wf = all(bool(torch.isfinite(v).all()) for v in sd["model_state_dict"].values() if torch.is_floating_point(v))
    tot = rec["curves"]["total"]
    return {"iterations": rec["hyperparameters"]["iterations"], "final_total_loss": float(tot[-1]),
            "final_loss_finite": bool(np.isfinite(tot[-1])), "loss_ever_nonfinite": bool(not np.all(np.isfinite(tot))),
            "weights_finite": wf, "wall_clock_s": rec["wall_clock_s"], "threads": rec["threads"],
            "holdout_episodes": rec["hyperparameters"]["holdout_episodes"], "recovered_record": recovered,
            "weights": os.path.relpath(os.path.join(rd, f"weights_{iters}.pt"), C.AIR)}


# ------------------------------------------------------------------ phase 4: study AR
def ar_runs(job):
    old = sorted(d for d in glob.glob(os.path.join(REPRO_RUNS, "baseline_*_seed*")) if os.path.isdir(d))
    if job.dry:
        old = [d for d in old if any(s in os.path.basename(d) for s in ("mlp_tf_s7_seed0", "rssm_tf_x1v1_seed0",
                                                                        "transformer_ar_matched_seed0"))]
    if not job.dry:     # S0's list: the reproduction's 26 baseline runs (test (vi) checks the same list)
        assert len(old) == 26, f"expected S0's 26 baseline runs, found {len(old)}"
    new = [run_dir_new(job, *r) for r in NEW_RUNS]
    return old, new


def phase_ar(job):
    old, new = ar_runs(job)
    ar, sc = compute.setup_arena((1, 8))
    excluded = {}
    for rd in new:
        name = os.path.basename(rd)
        t = job.db["training"].get(name)
        if t is None:
            excluded[name] = "not trained (missing)"
        elif not (t["final_loss_finite"] and t["weights_finite"]) or t["loss_ever_nonfinite"]:
            excluded[name] = "training loss became non-finite, or non-finite weights"
    todo = [rd for rd in old + new if os.path.basename(rd) not in job.db["models"]["AR"]
            and os.path.basename(rd) not in excluded]
    n_all = len(old) + len(new)
    for n, rd in enumerate(todo):
        name = os.path.basename(rd)
        job.progress("4 AR", name, n_all - len(todo) + n, len(todo) - n)
        t1 = time.time()
        if job.dry:
            m0, info = A.load(rd)
            randomise(m0, n)
            out = A.score_arch(rd, ar, sc, model=m0)
            out["info"].update(info)
        else:
            out = A.score_arch(rd, ar, sc)
        info, sec = out.pop("info"), out.pop("seconds")
        job.save_raw("AR", name, out)
        e = {"arch": info["arch"], "regime": info["regime"], "spec": info["spec"], "seed": int(info["seed"]),
             "iter": int(info["iter"]), "path": info["path"], "sha256": info["sha256"],
             "new_run": name.startswith("s1_baseline_")}
        for k in ("E1", "E8", "E24", "E100", "E368", "lam1_rec", "rJ_rec"):
            e[k] = model_mean(out[k])
        e["n_div_rec"] = int(np.sum(np.asarray(out["div_rec"]) > 0))
        e["n_lam_div"] = int(np.sum(np.asarray(out["lam_div"]).astype(bool)))
        e["seconds"] = {k: float(v) for k, v in sec.items()}
        job.db["models"]["AR"][name] = e
        job.save()
        print(f"  AR {name}: {time.time() - t1:.1f}s; E100 {e['E100']:.4f}; diverged {e['n_div_rec']}", flush=True)
    check_c4(job, old, new, excluded)
    job.progress("4 AR", "done", n_all, 0)


def check_c4(job, old, new, excluded):
    meta = json.load(open(os.path.join(C.RESULTS, "models.json")))
    missing = {}
    for tag in ("F18", "X5", "REL"):
        keys = [m["key"] for m in meta[tag]]
        if job.dry:
            keys = [k for k in keys if k in job.db["models"][tag]]
        missing[tag] = [k for k in keys if k not in job.db["models"][tag]]
    missing["AR"] = [os.path.basename(rd) for rd in old + new
                     if os.path.basename(rd) not in job.db["models"]["AR"] and os.path.basename(rd) not in excluded]
    n_old = len(old)
    ok = all(not v for v in missing.values()) and (job.dry or (n_old == 26 and not excluded))
    job.db["checks"]["S1-C4"] = {"pass": bool(ok), "missing": missing, "excluded_new_runs": excluded,
                                 "n_existing_runs": n_old, "n_new_runs_expected": len(new),
                                 "n_new_runs_scored": sum(v["new_run"] for v in job.db["models"]["AR"].values()),
                                 "n_scored": {t: len(v) for t, v in job.db["models"].items()}, "time": now()}
    job.save()
    print(f"  S1-C4: {'PASS' if ok else 'FAIL'}; excluded {excluded}", flush=True)


def assemble_raw(job):
    arrays = {}
    for tag in ("F18", "X5", "REL", "AR"):
        for key in job.db["models"][tag]:
            with np.load(os.path.join(job.raw_dir, tag, key + ".npz")) as z:
                for f in z.files:
                    arrays[f"{tag}__{key}__{f}"] = z[f]
    tmp = job.raw_npz[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, job.raw_npz)
    job.db["raw_npz"] = {"path": os.path.relpath(job.raw_npz, C.AIR), "n_arrays": len(arrays), "time": now()}
    job.save()


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="random-weight models, a few of each kind, .cache only")
    ap.add_argument("--with-tests", action="store_true", help="dry run: run tests (i), (vi), (ix) too")
    args = ap.parse_args()
    job = Job(dry=args.dry_run)
    phases = (("1 checks", lambda: phase_checks(job, skip_tests=args.dry_run and not args.with_tests)),
              ("2 CL", lambda: phase_cl(job)), ("3 training", lambda: phase_training(job)),
              ("4 AR", lambda: phase_ar(job)))
    for name, fn in phases:
        t1 = time.time()
        print(f"== phase {name}: start {now()}", flush=True)
        fn()
        job.db["phases"][name] = {"done": now(), "seconds": time.time() - t1}
        job.save()
        print(f"== phase {name}: done in {time.time() - t1:.0f}s", flush=True)
    assemble_raw(job)
    job.progress("done", "raw_S1.npz assembled", 1, 0)
    c = job.db["checks"]
    print("checks: " + ", ".join(f"{k} {'PASS' if v['pass'] else 'FAIL'}" for k, v in sorted(c.items())), flush=True)


if __name__ == "__main__":
    main()

"""Option D (D0): tests of the record and fine-tuning code (OPTIOND_PLAN.md D0 step 4).  ./py scripts/d/test_d0.py

Each test runs in its own process (the lite, MuJoCo and paper modules set their own import paths). Everything written
goes under .cache/d_test/ and .cache/d_eval/; nothing under results/ or runs/.
  1. checks.py: REL on the paper's recorded arena gives the E8 and lambda1 of results/model_table.json, and the r_J that
     Step 2 recomputed on this Mac (results/s2/checks.json); the common-units model of an own-scale simulator equals
     converting its predictions;
  2. sims.load("REL") equals the released checkpoint; sims.load("REL__u10") reproduces Step 2's variant record;
  3. walk_test.py on REL reproduces results/c/calibration.json's tracking and u_bar;
  4. evaluate.py on RWM's released policy, 2 schedules x 1 start, gives scripts/c/evaluate.py's episode scores;
  5. walking windows come only from sources 0, 1, 2; no window crosses an episode end; a fall only at a window's last
     row; no test-set episode is in the rung;
  6. finetune.py's loss at step 0 on a fixed batch equals the rebuild's loss for the released model (same seed);
  7. a run advanced to 3 steps and then to 5 equals one advanced straight to 5, weight for weight;
  8. a save holds the authors' scales, loads with common.load, and runs one step inside the lite adapter;
  9. a teacher-forced step and a gentle step run, and change the weights.
"""
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
AIR = os.path.dirname(SCRIPTS)
TMP = os.path.join(AIR, ".cache", "d_test")
OUT = os.path.join(TMP, "results.json")


def ok(name, cond, detail):
    print(json.dumps({"test": name, "passed": bool(cond), "detail": detail}), flush=True)
    return bool(cond)


# ------------------------------------------------------------------------------------------------ the tests
def t1():
    sys.path.insert(0, HERE)
    import numpy as np
    import torch
    import checks as K
    import sims
    torch.set_num_threads(2)
    out = os.path.join(TMP, "t1_REL_recorded.json")
    if os.path.exists(out):
        os.remove(out)
    r = K.run("REL", "recorded", out)["sets"]["recorded"]
    t = json.load(open(os.path.join(AIR, "results", "model_table.json")))["REL"][0]
    mac = json.load(open(os.path.join(AIR, "results", "s2", "checks.json")))["validation_REL_vs_stored"]
    rel = lambda a, b: abs(a - b) / abs(b)   # noqa: E731
    good = ok("1a E8 = model_table (relative 1e-6)", rel(r["E8"], t["E8"]) <= 1e-6, [r["E8"], t["E8"]])
    good &= ok("1b lambda1 = model_table (relative 1e-6)", rel(r["lambda1"], t["lam1"]) <= 1e-6,
               [r["lambda1"], t["lam1"]])
    good &= ok("1c r_J = Step 2's recomputation on this Mac (1e-9), and within 0.5% of model_table",
               rel(r["rJ"], mac["rJ"]["recomputed"]) <= 1e-9 and rel(r["rJ"], t["rJ"]) <= 5e-3,
               [r["rJ"], mac["rJ"]["recomputed"], t["rJ"]])
    # common units: folded model == converting the predictions (own-scale simulator, 4 windows)
    m, mean, std, rec = sims.load("sim_10k_N8_s0")
    am, asd = sims.authors_scales()
    st_a, ac, st_o, _ = K.test_set("expert", mean, std, 4)
    import lyap
    with torch.no_grad():
        p_own = lyap.rollout_map(m, st_o, ac).double().numpy() * std + mean
        p_fold = lyap.rollout_map(K.common_units(m, mean, std), st_a, ac).double().numpy() * asd + am
    d = float(np.max(np.abs(p_own - p_fold)[:, 32:40] / asd))
    good &= ok("1d common-units model = converted predictions (first 8 forecast steps, 4 windows; < 1e-3 in the "
               "authors' units)", d < 1e-3, d)
    same = K.common_units(*sims.load("REL")[:3]) is not None
    m0, a0, s0, _ = sims.load("REL")
    good &= ok("1e REL is used as it is (no folding)", K.common_units(m0, a0, s0) is m0 and same, "identity")
    return good


def t2():
    sys.path.insert(0, HERE)
    import torch
    import sims
    import common as C
    m, mean, std, rec = sims.load("REL")
    sd = torch.load(C.W_RELEASED, map_location="cpu", weights_only=False)["system_dynamics_state_dict"]
    msd = m.state_dict()
    eq = sorted(sd) == sorted(msd) and all(torch.equal(sd[k], msd[k]) for k in sd)
    am, asd = sims.authors_scales()
    good = ok("2a sims.load('REL') = the released checkpoint, authors' scales", eq and (mean == am).all()
              and (std == asd).all(), rec["sha256"][:12])
    _, _, _, r10 = sims.load("REL__u10")
    want = json.load(open(os.path.join(AIR, "results", "s2", "checks.json")))["models"]["REL__u10"]["variant_info"]
    good &= ok("2b sims.load('REL__u10') reproduces Step 2's variant record", r10["variant_info"] == want,
               r10["variant_info"])
    names = sims.EXISTING
    good &= ok("2c 17 simulators, all load", len(names) == 17 and all(os.path.exists(sims.path_of(n)) for n in names),
               len(names))
    return good


def t3():
    sys.path.insert(0, HERE)
    import walk_test as WT
    out = os.path.join(TMP, "t3_walk_REL.json")
    if os.path.exists(out):
        os.remove(out)
    r = WT.run("REL", out, job="test")
    c = json.load(open(os.path.join(AIR, "results", "c", "calibration.json")))["simulators"]["REL"]
    return ok("3 walk test on REL = calibration.json (tracking and u_bar, 1e-9)",
              abs(r["tracking_per_step"] - c["tracking_per_step"]) <= 1e-9 and abs(r["u_bar"] - c["u_bar"]) <= 1e-9,
              [r["tracking_per_step"], c["tracking_per_step"], r["u_bar"], c["u_bar"]])


def t4():
    sys.path.insert(0, HERE)
    import importlib.util
    spec = importlib.util.spec_from_file_location("d_evaluate", os.path.join(HERE, "evaluate.py"))
    DE = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(DE)
    CE, C = DE.CE, DE.C
    CE.F.PROGRESS = os.path.join(TMP, "c_progress.json")     # Option C's evaluator must not touch results/c/
    for f in ("_test_d0_ref", "_test_d0_new"):
        for ext in (".json", ".jsonl"):
            p = os.path.join(AIR, ".cache", "d_eval", f + ext)
            if os.path.exists(p):
                os.remove(p)
    p = os.path.join(AIR, ".cache", "c_eval__test_d0_ref.json")
    for q in (p, os.path.join(AIR, ".cache", "c_eval__test_d0_ref.jsonl")):
        if os.path.exists(q):
            os.remove(q)
    ref = CE.evaluate("_test_d0_ref", None, (0, 1), 1, False, out=p, actor_arg=C.W_RELEASED)
    new = DE.evaluate("_test_d0_new", C.W_RELEASED, (0, 1), 1, False, job="test")
    a = [(e["schedule"], e["start"], e["score"], e["fell"]) for e in ref["episodes"]]
    b = [(e["schedule"], e["start"], e["score"], e["fell"]) for e in new["episodes"]]
    good = ok("4a evaluate.py = scripts/c/evaluate.py on RWM's released policy, 2 schedules x 1 start", a == b, b)
    try:
        DE.check_allowed("W_s0_d80__p1")
        blocked = False
    except SystemExit:
        blocked = True
    good &= ok("4b the guard refuses a ladder policy before registration", blocked or DE.registered(), blocked)
    return good


def t5():
    sys.path.insert(0, HERE)
    import numpy as np
    import finetune as FT
    good = True
    a = np.load(os.path.join(FT.DATA_DIR, "rung_1M.npy"), mmap_mode="r")
    src = np.asarray(a[:, FT.SOURCE_COL]).astype(np.int64)
    ep = np.load(os.path.join(FT.DATA_DIR, "rung_1M.ep.npy"))
    fall = np.load(os.path.join(FT.DATA_DIR, "rung_1M.fall.npy")).astype(np.int64)
    good &= ok("5a episode column = .ep.npy; fall column = .fall.npy",
               np.array_equal(np.asarray(a[:, FT.EP_COL]).astype(np.int64), ep)
               and np.array_equal((np.asarray(a[:, 65]) > 0.5).astype(np.int64), fall), "")
    for data, N in (("walking", 8), ("walking", 1), ("mixed", 8)):
        ds = FT.windows(data, N)
        L = ds.window
        s = ds.starts
        bad_src = np.cumsum(np.r_[0, ~np.isin(src, FT.WALKING_SOURCES)])
        cf = np.cumsum(np.r_[0, fall])
        one_ep = np.all([np.array_equal(ep[s], ep[s + k]) for k in range(L)])
        no_mid_fall = np.all(cf[s + L - 1] - cf[s] == 0)
        if data == "walking":
            good &= ok(f"5b {data} N={N}: every row of every window is from source 0, 1 or 2",
                       np.all(bad_src[s + L] - bad_src[s] == 0), len(s))
        else:
            good &= ok(f"5b {data} N={N}: windows from every source", len(np.unique(src[s])) > 3,
                       int(len(np.unique(src[s]))))
        good &= ok(f"5c {data} N={N}: no window crosses an episode end; a fall only at the last row",
                   one_ep and no_mid_fall, int((fall[s + L - 1] > 0).sum()))
        st, ac, ext, con, term = ds.batch(np.arange(min(64, len(s))))
        am, asd = FT.authors_scales()
        raw = np.asarray(a[s[0]:s[0] + L, :45], dtype=np.float64)
        good &= ok(f"5d {data} N={N}: windows in the authors' scales",
                   np.allclose(st[0].double().numpy(), (raw - am) / asd, atol=1e-4), "")
    man = json.load(open(os.path.join(AIR, "optc", "data_manifest.json")))
    test_eps = set()
    for f in ("test_expert.npy", "test_mixed.npy"):
        t = np.load(os.path.join(FT.DATA_DIR, f), mmap_mode="r")
        test_eps |= set(np.unique(np.asarray(t[:, 66]).astype(np.int64)).tolist())
    good &= ok("5e no test-set episode is in the rung", not (test_eps & set(np.unique(ep).tolist())),
               {"test_episodes": len(test_eps), "manifest_held_out": list(man.get("held_out", {}))[:3]})
    return good


def _spec(**k):
    base = {"data": "walking", "lr": 1e-4, "N": 8, "seed": 0, "rung": "100k"}
    base.update(k)
    return base


def t6():
    sys.path.insert(0, HERE)
    import torch
    import finetune as FT
    import common as C
    torch.set_num_threads(2)
    sp = {"run": "_test_t6", **_spec(rung="1M")}
    ds = FT.windows("walking", 8)
    batch = ds.batch(list(range(0, 256 * 997, 997)))
    model, opt, gen = FT.setup(sp)
    torch.manual_seed(123)
    ours = float(FT.loss(model, batch, False, C.data()[0]["loss_weights"]))
    ref = C.load(C.W_RELEASED, 5)
    ref.train()
    opt2 = FT.T.make_optimizer(ref, C.data()[0])
    torch.manual_seed(123)
    _, theirs = FT.T.train_step(ref, opt2, batch, C.data()[0]["loss_weights"])
    return ok("6 step-0 loss on a fixed batch = the rebuild's loss for the released model", abs(ours - theirs) <= 1e-6
              * max(1.0, abs(theirs)), [ours, theirs])


def _weights(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]


def t7():
    sys.path.insert(0, HERE)
    import torch
    import finetune as FT
    runs = os.path.join(TMP, "runs")
    for r in ("_test_a", "_test_b"):
        shutil.rmtree(os.path.join(runs, r), ignore_errors=True)
        shutil.rmtree(os.path.join(runs, r + "_d5"), ignore_errors=True)
    sp = _spec()
    FT.advance("_test_a", 3, runs, test_spec=sp, saves=(5,), job="test")
    FT.advance("_test_a", 5, runs, test_spec=sp, saves=(5,), job="test")
    FT.advance("_test_b", 5, runs, test_spec=sp, saves=(5,), job="test")
    a, b = _weights(os.path.join(runs, "_test_a_d5", "final.pt")), _weights(os.path.join(runs, "_test_b_d5", "final.pt"))
    same = sorted(a) == sorted(b) and all(torch.equal(a[k], b[k]) for k in a)
    sa = FT.status("_test_a", runs)
    return ok("7 run to 3 then 5 = run straight to 5, weight for weight", same and sa["step"] == 5, sa["saved"])


def t8():
    sys.path.insert(0, HERE)
    import numpy as np
    import torch
    import finetune as FT
    import common as C
    import sims
    runs = os.path.join(TMP, "runs")
    path = os.path.join(runs, "_test_a_d5", "final.pt")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    am, asd = FT.authors_scales()
    good = ok("8a the save holds the authors' scales and the run's settings",
              np.allclose(ck["state_mean"], am, atol=0) and np.allclose(ck["state_std"], asd, atol=0)
              and ck["run"] == "_test_a" and ck["steps"] == 5, {k: ck[k] for k in ("run", "steps", "data", "lr", "N")})
    m = C.load(path, 5)
    sys.path.insert(0, os.path.join(SCRIPTS, "lite"))
    from adapter import LiteDynamicsAdapter
    ds = FT.windows("walking", 8, rung="100k")
    st, ac, _, _, _ = ds.batch(np.arange(16))
    ad = LiteDynamicsAdapter(m)
    ad.eval()
    with torch.no_grad():
        ad.reset()
        out = ad.forward(st[:, :32], ac[:, :32])
    good &= ok("8b loads with common.load; one step in the lite adapter gives finite (16, 45) states",
               out[0].shape == (16, 45) and bool(torch.isfinite(out[0]).all()), list(out[0].shape))
    sims.RUNS_D = runs                          # the ladder route of sims.load, on the test save
    m2, mean2, std2, rec = sims.load("_test_a_d5")
    good &= ok("8c sims.load reads it as a ladder simulator in the authors' scales",
               rec["kind"] == "ladder" and rec["scales"] == "authors" and rec["steps"] == 5, rec["kind"])
    return good


def t8env():
    """One imagination step of the lite env on the test save (walk_test's route)."""
    sys.path.insert(0, HERE)
    import walk_test as WT
    import sims
    sims.RUNS_D = os.path.join(TMP, "runs")
    out = os.path.join(TMP, "t8_walk.json")
    if os.path.exists(out):
        os.remove(out)
    r = WT.run("_test_a_d5", out, n_envs=16, n_steps=2, job="test")
    return ok("8d the save runs inside the lite env (16 robots, 2 imagined steps)", r["imagine"]["steps_run"] == 2,
              {k: r[k] for k in ("falls", "tracking_per_step", "u_bar")})


def t9():
    sys.path.insert(0, HERE)
    import torch
    import finetune as FT
    import common as C
    runs = os.path.join(TMP, "runs")
    rel = torch.load(C.W_RELEASED, map_location="cpu", weights_only=False)["system_dynamics_state_dict"]
    good = True
    for name, sp in (("_test_tf", _spec(N=1)), ("_test_gentle", _spec(lr=1e-5)), ("_test_mixed", _spec(data="mixed"))):
        shutil.rmtree(os.path.join(runs, name), ignore_errors=True)
        shutil.rmtree(os.path.join(runs, name + "_d1"), ignore_errors=True)
        s = FT.advance(name, 1, runs, test_spec=sp, saves=(1,), job="test")
        w = _weights(os.path.join(runs, name + "_d1", "final.pt"))
        changed = sum(not torch.equal(w[k], rel[k]) for k in rel)
        good &= ok(f"9 {name}: one step runs, its loss is finite, and it changes the weights",
                   changed > 0 and s["nonfinite_at"] is None and s["step"] == 1,
                   {"tensors_changed": changed, "of": len(rel), "loss": s["loss_last"]["total"]})
    return good


TESTS = {"t1": ("./py", t1), "t2": ("./py", t2), "t3": ("./py", t3), "t4": ("./py-mj", t4), "t5": ("./py", t5),
         "t6": ("./py", t6), "t7": ("./py", t7), "t8": ("./py", t8), "t8env": ("./py", t8env), "t9": ("./py", t9)}


def main():
    if len(sys.argv) > 1:
        sys.exit(0 if TESTS[sys.argv[1]][1]() else 1)
    os.makedirs(TMP, exist_ok=True)
    env = dict(os.environ, D_PROGRESS=os.path.join(TMP, "progress.json"))
    res, t0 = {}, time.time()
    for k, (py, _) in TESTS.items():
        t = time.time()
        p = subprocess.run([py, os.path.join("scripts", "d", "test_d0.py"), k], cwd=AIR, env=env,
                           capture_output=True, text=True)
        lines = [json.loads(x) for x in p.stdout.splitlines() if x.startswith('{"test"')]
        res[k] = {"returncode": p.returncode, "checks": lines, "seconds": round(time.time() - t, 1)}
        if p.returncode != 0 or not lines:
            res[k]["stderr_tail"] = p.stderr[-1500:]
        print(f"{k}: {'PASS' if p.returncode == 0 and lines and all(x['passed'] for x in lines) else 'FAIL'} "
              f"({res[k]['seconds']} s)", flush=True)
        for x in lines:
            print(f"   {'ok ' if x['passed'] else 'BAD'} {x['test']}: {str(x['detail'])[:150]}")
        if p.returncode != 0 and not lines:
            print(p.stderr[-1500:])
    allok = all(r["returncode"] == 0 and r["checks"] and all(x["passed"] for x in r["checks"]) for r in res.values())
    json.dump({"all_passed": allok, "seconds": round(time.time() - t0, 1), "tests": res}, open(OUT, "w"), indent=1)
    print(f"ALL {'PASSED' if allok else 'FAILED'} in {time.time() - t0:.0f} s")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()

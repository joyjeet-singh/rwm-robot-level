"""Option C (C1) tests.

    ./py-mj scripts/c/test_c1.py --part gen      # generator against Step 2's evaluator; a small data set end to end
    ./py    scripts/c/test_c1.py --part train    # trainer's loader against the rebuild's, first 20 losses

Outputs go to .cache/ only; these are checks of the code, never results.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AIR = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(AIR, "scripts"))
FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail != "" else ""), flush=True)
    if not ok:
        FAILS.append(name)


def part_gen():
    import mujoco

    sys.path.insert(0, os.path.join(AIR, "scripts", "s2"))
    import fidelity as F
    import gen_data as G
    import s1_policy as P
    import torch
    torch.set_num_threads(2)

    # ---- 1. the generator against Step 2's evaluator, authors' policy, same seed and commands, 20 s
    world = G.World()
    actor = F.torch_actor(P.load_actor())
    worst = 0.0
    for e, j in ((0, 0), (3, 2)):
        rec = []
        env = world.env
        orig = env.step

        def hook(a):
            out = orig(a)
            rec.append((out[1]["state"].copy(), out[1]["contacts"].copy(), np.array(a, dtype=np.float64), out[1]["fell"]))
            return out
        env.step = hook
        F.run_policy_episode(env, actor, e, j)
        env.step = orig
        off, yaw = F.eval_start(e, j)
        spec = {"joint_offset": off, "yaw": yaw, "cmd": F.schedule(e, 1001), "physics": None, "pushes": {}}
        rows, fell, broken = G.simulate(world, actor, spec, noise=0.0, smooth=False, rng=None, steps=1000)
        check(f"evaluator episode ({e},{j}) same length", len(rows) - 1 == len(rec), f"{len(rows) - 1} {len(rec)}")
        n = min(len(rows) - 1, len(rec))
        d = max(np.abs(rows[1:n + 1, :45] - np.array([r[0] for r in rec[:n]])).max(),
                np.abs(rows[1:n + 1, 45:57] - np.array([r[2] for r in rec[:n]])).max(),
                np.abs(rows[1:n + 1, 57:65] - np.array([r[1] for r in rec[:n]])).max())
        worst = max(worst, float(d))
        check(f"evaluator episode ({e},{j}) fall flag", fell == rec[n - 1][3] and not broken, f"{n} steps, fell {fell}")
    check("generator equals Step 2's evaluator to 1e-6 (states, joint commands, contacts)", worst < 1e-6, f"{worst:.2e}")

    # ---- 2. physics changes take effect and are undone exactly
    env, m = world.env, world.env.m
    nom = {k: np.array(v, copy=True) for k, v in (("friction", m.geom_friction), ("priority", m.geom_priority),
                                                  ("mass", m.body_mass), ("inertia", m.body_inertia),
                                                  ("subtree", m.body_subtreemass))}
    world.apply({"friction": 0.4, "payload": 5.0, "kp_scale": 1.1, "kd_scale": 0.9})
    check("payload is added to the base", abs(m.body_mass[env.base] - nom["mass"][env.base] - 5.0) < 1e-12)
    check("motor gains are scaled", abs(env.motor.kp / world.nominal["kp"] - 1.1) < 1e-12)
    env.reset(cmd=(0.0, 0.0, 0.0))
    for _ in range(25):
        env.step(np.zeros(12))
    frs = []
    for i in range(env.d.ncon):
        c = env.d.contact[i]
        if env.g_floor in (c.geom1, c.geom2) and any(g in (c.geom1, c.geom2) for g in world.feet):
            frs.append(float(c.friction[0]))
    check("feet use the drawn friction against the floor", len(frs) > 0 and all(abs(f - 0.4) < 1e-12 for f in frs),
          f"{len(frs)} contacts")
    world.apply(None)
    same = (np.array_equal(m.geom_friction, nom["friction"]) and np.array_equal(m.geom_priority, nom["priority"]) and
            np.array_equal(m.body_mass, nom["mass"]) and np.array_equal(m.body_inertia, nom["inertia"]) and
            np.allclose(m.body_subtreemass, nom["subtree"], atol=1e-12))
    check("physics changes are undone exactly", same and env.motor.kp == world.nominal["kp"])

    # ---- 3. a small data set, end to end
    data = os.path.join(AIR, ".cache", "test_c1_data")
    shutil.rmtree(data, ignore_errors=True)
    man = os.path.join(data, "manifest.json")
    os.makedirs(data)
    base = ["--data-dir", data, "--manifest", man, "--train-total", "40000", "--chunk", "1500", "--n-mixed", "8",
            "--n-expert", "2", "--n-d6", "2", "--no-progress"]
    G.main(["--worker", "0", "--nworkers", "2", *base])
    G.main(["--worker", "1", "--nworkers", "2", *base])
    stamps = {f: os.path.getmtime(os.path.join(data, f)) for f in sorted(os.listdir(data))}
    G.main(["--worker", "0", "--nworkers", "2", *base])
    check("a re-run skips finished shards", stamps == {f: os.path.getmtime(os.path.join(data, f))
                                                        for f in sorted(os.listdir(data))})
    G.main(["--finish", *base, "--rungs", "2k=2000,8k=8000,30k=30000"])
    rec = json.load(open(man))
    r = rec["rungs"]
    check("rungs are nested", set(r["2k"]["episodes"]) <= set(r["8k"]["episodes"]) <= set(r["30k"]["episodes"]))
    share = {g: sum(s["share"] for s in rec["sources"] if s["group"] == g) for g in G.GROUPS}
    ok = all(abs(c["transitions"] - share[g] * 30000) <= 1000 for g, c in r["30k"]["by_group"].items())
    check("each group is within one episode of its share of the 30k rung", ok,
          {g: c["transitions"] for g, c in r["30k"]["by_group"].items()})
    tr = [s for s in rec["sources"] if s["group"] == "trained"]
    counts = [rec["per_source"][str(s["id"])]["transitions"] for s in tr]
    check("all 22 trained policies generated", len(tr) == 22 and min(counts) > 0, f"{min(counts)}..{max(counts)}")

    # determinism: the same shard twice
    data2 = os.path.join(AIR, ".cache", "test_c1_data2")
    shutil.rmtree(data2, ignore_errors=True)
    os.makedirs(data2)
    source = next(s for s in G.sources() if s["id"] == 25)
    get = G.actors()
    G.make_train_shard(world, get, source, 0, 1500, os.path.join(data2, "a.npy"))
    G.make_train_shard(world, get, source, 0, 1500, os.path.join(data2, "b.npy"))
    check("a shard is reproducible", np.array_equal(np.load(os.path.join(data2, "a.npy")), np.load(os.path.join(data2, "b.npy"))))

    # ---- 4. episode structure and the window rule on the 30k rung
    a = np.load(os.path.join(data, "rung_30k.npy"), mmap_mode="r")
    ep, fall = np.load(os.path.join(data, "rung_30k.ep.npy")), np.load(os.path.join(data, "rung_30k.fall.npy"))
    cuts = np.flatnonzero(ep[1:] != ep[:-1]) + 1
    first, last = np.r_[0, cuts], np.r_[cuts, len(ep)] - 1
    check("episode ids are contiguous and unique", len(first) == len(set(ep[first].tolist())), len(first))
    check("every episode starts with a zero joint command", np.abs(np.asarray(a[first, 45:57])).max() == 0.0)
    check("a fall appears only at an episode's last row", set(np.flatnonzero(fall).tolist()) <= set(last.tolist()),
          f"{int(fall.sum())} falls")
    check("the rung holds falls to learn from", int(fall.sum()) > 0)
    check("episode lengths at most 1,000 rows", int((last - first + 1).max()) <= 1000, int((last - first + 1).max()))
    import train_sim as TS
    for L in (33, 40):
        ds = TS.RungWindows(a, ep, fall, np.zeros(45), np.ones(45), L)
        sw = ds.starts[:, None] + np.arange(L)
        same_ep = bool((ep[sw] == ep[sw[:, :1]]).all())
        early = bool(fall[sw[:, :-1]].any())
        expected = sum(max(0, int(l) - L + 1) for l in (last - first + 1))
        check(f"windows of {L}: none crosses an episode end", same_ep, len(ds))
        check(f"windows of {L}: a fall only at the last row", not early and bool(fall[sw[:, -1]].any()))
        check(f"windows of {L}: every window inside an episode is used", len(ds) == expected, f"{len(ds)} of {expected}")
    if FAILS:
        raise SystemExit(f"FAILED: {FAILS}")
    print("ALL PASS (gen)")
    shutil.rmtree(data2, ignore_errors=True)
    return mujoco.__version__


def part_train():
    import torch

    import common as C
    import train_sim as TS
    R, T = C.R, TS.T
    torch.set_num_threads(2)
    paths = R.repo_paths()
    cfg = R.load_reference_config(paths["lite"])
    data, ep = R.load_data(paths["csv"], verbose=False)
    train_eps = [0, 2, 3, 4, 5, 6, 7, 9]
    ref = T.WindowDataset(data, ep, train_eps, cfg, window=T.WINDOW)
    mine = TS.RungWindows(data, ep, data[:, 65] > 0.5, cfg["state_data_mean"], cfg["state_data_std"], T.WINDOW,
                          episodes=train_eps)
    check("same windows as the rebuild's loader", np.array_equal(ref.starts, mine.starts), len(mine))
    g1, g2 = torch.Generator().manual_seed(0), torch.Generator().manual_seed(0)
    same = all(all(torch.equal(x, y) for x, y in zip(ref.sample(256, g1), mine.sample(256, g2))) for _ in range(3))
    check("same batches as the rebuild's loader", same)

    stored = json.load(open(os.path.join(C.RWM, "results", "step5_armA_seed0_ens5.json")))
    curves = stored["curves"]
    check("the stored rebuild run used batch 256 and five networks",
          stored["hyperparameters"]["batch"] == 256 == TS.BATCH and stored["hyperparameters"]["ensemble"] == 5)
    model, opt, gen = TS.setup(cfg, 0)
    hist = TS.new_hist()
    for _ in range(20):
        TS.one_step(model, opt, mine, gen, 256, False, cfg["loss_weights"], hist)
    worst = max(max(abs(a - b) for a, b in zip(hist[k], curves[k][:20]))
                for k in ("state", "bound", "contact", "termination", "total"))
    check("first 20 losses equal the rebuild's to 1e-4", worst < 1e-4, f"{worst:.2e}")
    if FAILS:
        raise SystemExit(f"FAILED: {FAILS}")
    print("ALL PASS (train)")


def part_job():
    """On the small data set of --part gen: resume after an interruption, and the checks job, end to end."""
    import torch

    import sim_checks as SC
    import train_sim as TS
    torch.set_num_threads(2)
    data = os.path.join(AIR, ".cache", "test_c1_data")
    runs = os.path.join(AIR, ".cache", "test_c1_runs")
    shutil.rmtree(runs, ignore_errors=True)
    base = ["--rung", "30k", "--seed", "0", "--threads", "2", "--data-dir", data, "--runs", runs, "--no-progress",
            "--ckpt-every", "2"]
    TS.main(base + ["--N", "1", "--steps", "3", "--name", "whole"])
    TS.main(base + ["--N", "1", "--steps", "3", "--name", "cut"])
    for f in ("final.pt", "train.json"):
        os.remove(os.path.join(runs, "cut", f))
    check("a rolling checkpoint is left to resume from", os.path.exists(os.path.join(runs, "cut", "rolling_2.pt")))
    TS.main(base + ["--N", "1", "--steps", "3", "--name", "cut"])
    a = torch.load(os.path.join(runs, "whole", "final.pt"), weights_only=False)["model_state_dict"]
    b = torch.load(os.path.join(runs, "cut", "final.pt"), weights_only=False)["model_state_dict"]
    check("an interrupted run resumes to the same weights", all(torch.equal(a[k], b[k]) for k in a))
    h = json.load(open(os.path.join(runs, "cut", "train.json")))
    check("the resumed run has one loss per step", len(h["curves"]["total"]) == 3, len(h["curves"]["total"]))
    stamp = os.path.getmtime(os.path.join(runs, "cut", "final.pt"))
    TS.main(base + ["--N", "1", "--steps", "3", "--name", "cut"])
    check("a finished run is skipped", os.path.getmtime(os.path.join(runs, "cut", "final.pt")) == stamp)

    TS.main(["--rung", "30k", "--seed", "0", "--threads", "2", "--data-dir", data, "--runs", runs, "--no-progress",
             "--N", "8", "--steps", "2", "--name", "n8"])
    out = os.path.join(runs, "checks_out.json")
    SC.main(["--sims", "n8", "--out", out, "--data-dir", data, "--runs", runs, "--threads", "2", "--no-progress"])
    r = json.load(open(out))["sims"]["n8"]["sets"]
    keys = ["windows", "E1", "floor_E1", "E8", "floor_E8", "E24", "floor_E24", "E100", "floor_E100", "lambda1", "rJ"]
    finite = all(np.isfinite(r[s][k]) for s in r for k in keys)
    check("checks are computed and finite on both test sets", finite and set(r) == {"mixed", "expert"},
          {s: (r[s]["windows"], r[s]["episodes_too_short"]) for s in r})
    if FAILS:
        raise SystemExit(f"FAILED: {FAILS}")
    print("ALL PASS (job)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["gen", "train", "job"], required=True)
    a = ap.parse_args()
    {"gen": part_gen, "train": part_train, "job": part_job}[a.part]()

"""Option C (C2) tests. Each part prints PASS lines and exits non-zero on failure.

    ./py-mj scripts/c/test_c2.py --part eval       # evaluator: Step 2 equality, score, standing reference, guard
    ./py    scripts/c/test_c2.py --part starts     # start-state rule on the real rungs and on a constructed case
    ./py    scripts/c/test_c2.py --part predict    # the wrapper's one-step predictions against train_sim's own
    ./py    scripts/c/test_c2.py --part short      # one iteration, 64 robots, 10k pilot simulator, nonzero uncertainty
    ./py    scripts/c/test_c2.py --part weight     # the uncertainty-weight flag reaches the reward's penalty
    ./py    scripts/c/test_c2.py --part rule       # pilot_rule on fixtures: every branch
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AIR = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)


def ok(msg):
    print(f"PASS: {msg}", flush=True)


# ------------------------------------------------------------------ evaluator (./py-mj)
def part_eval():
    import evaluate as EV
    F = EV.F
    import s1_policy as P
    env = F.make_env("cal", reward=True)
    actor = F.torch_actor(P.load_actor(None))

    # the episode loop equals Step 2's on a recorded schedule (recorded schedule 0, start 0, and a held-out-free start)
    for e, j in ((0, 0), (3, 2)):
        a = F.run_policy_episode(env, actor, e, j)
        off, yaw = F.eval_start(e, j)
        b = EV.run_episode(env, actor, F.schedule(e, 1001), off, yaw, e, j)
        for k in ("fell", "steps", "ret", "speed_err", "turn_err", "two_feet", "duty"):
            assert a[k] == b[k], (e, j, k, a[k], b[k])
        assert not b["diverged"]
    ok("episode loop equals fidelity.run_policy_episode (fell, steps, return, errors, gait)")

    # the step score: perfect tracking is 1, a known error gives the formula's value
    s = np.zeros(45)
    c = np.array([0.5, -0.5, 1.0])
    s[0:2], s[5] = c[:2], c[2]
    assert EV.step_score(s, c) == 1.0
    s[0] += 0.5
    assert abs(EV.step_score(s, c) - np.exp(-1.0)) < 1e-12
    ok("step score: 1 for perfect tracking, exp(-1) for a speed error of 0.5")

    # schedules are in range, and change only at 10 s
    for e in range(EV.N_SCHEDULES):
        c = EV.walk_schedule(e)
        assert c.shape == (1001, 3) and np.abs(c).max() <= 1.0
        assert np.all(c[:500] == c[0]) and np.all(c[500:] == c[500]) and np.any(c[0] != c[500])
    assert np.array_equal(EV.walk_schedule(3), EV.walk_schedule(3))
    ok("24 schedules in range, one change at 10 s, repeatable")

    # a fall stops the sum: every remaining step scores 0 (a policy with full joint commands falls)
    falling = lambda obs: np.ones(12) * 3.0  # noqa: E731
    off, yaw = EV.walk_start(0, 0)
    r = EV.run_episode(env, falling, EV.walk_schedule(0), off, yaw, 0, 0)
    assert r["fell"], "the test policy did not fall"
    assert 0 <= r["score"] <= r["steps"] / 1000 + 1e-12, r
    ok(f"a fall ends scoring: fell at step {r['steps']}, score {r['score']:.4f} is at most the share of steps survived")

    # the standing-still reference is exact and near the value the plan gives
    ref = EV.standing_reference(range(EV.N_SCHEDULES))
    assert 0.03 < ref < 0.2, ref
    by_hand = np.mean([np.exp(-np.sum(EV.walk_schedule(e)[:1000] ** 2, axis=1) / 0.25).mean() for e in range(EV.N_SCHEDULES)])
    assert abs(ref - by_hand) < 1e-12
    ok(f"standing-still reference {ref:.4f} (plan: about 0.09 for uniform commands)")

    # a policy that holds the standing pose scores near the reference
    if os.path.exists(os.path.join(AIR, ".cache", "c_eval__test_stand__iNone.json")):
        os.remove(os.path.join(AIR, ".cache", "c_eval__test_stand__iNone.json"))
    part = os.path.join(AIR, ".cache", "c_eval__test_stand__iNone.jsonl")
    if os.path.exists(part):
        os.remove(part)
    sub = range(0, EV.N_SCHEDULES, 6)
    rec = EV.evaluate("_test_stand", None, sub, 1, recorded=False)
    gap = rec["score"] - rec["standing_reference"]
    assert rec["summary"]["fall_rate"] == 0.0, rec["summary"]
    assert abs(gap) < 0.05, (rec["score"], rec["standing_reference"])
    ok(f"standing pose scores {rec['score']:.4f}, reference {rec['standing_reference']:.4f} on the same {len(sub)} schedules")

    # the guard
    try:
        if EV.registered():
            ok("guard not tested: prereg/PREREG_C.md is already committed")
        else:
            EV.check_allowed("sim_1M_N8_s0__p1")
            raise AssertionError("guard let a main-pool policy through")
    except SystemExit as e:
        assert "PREREG_C" in str(e), e
        for name in (EV.PILOT + ("authors", "_test_x")):
            EV.check_allowed(name)
        ok("guard: refuses a main-pool policy, allows the authors, pilot and `_test` names")
    assert "sim_6M_N8_s0__w0.25__p0" in EV.PILOT and "sim_6M_N8_s0__w2__p0" in EV.PILOT, EV.PILOT
    ok("retry policy names are allowed by the guard")


# ------------------------------------------------------------------ start states (./py)
def part_starts():
    import train_policy as W
    # constructed case: a 200-row fall episode, a 200-row no-fall episode, tilt in a known place
    n = 200
    a = np.zeros((2 * n, 68), dtype=np.float32)
    a[:, 8] = -1.0
    a[40:45, 7] = 0.9                         # episode 0, rows 40-44: tilted about 45 degrees
    a[40:45, 8] = -0.9
    ep = np.r_[np.zeros(n), np.ones(n)].astype(np.int64)
    fall = np.zeros(2 * n, dtype=bool)
    fall[n - 1] = True
    got = W.valid_starts(a, fall, [(0, n), (n, 2 * n)])
    ep0, ep1 = got[got < n], got[got >= n] - n
    want0 = [s for s in range(0, n - 32 - 50 + 1) if not (s <= 44 and s + 31 >= 40)]
    assert ep0.tolist() == want0, (ep0[:5], len(ep0), len(want0))
    assert ep1.tolist() == list(range(0, n - 32 + 1))
    # the last allowed window ends 50 steps before the fall; the next one would end 49 before it
    assert ep0.max() + 31 + 50 == n - 1
    ok("start rule: tilted rows excluded, windows end at least 1 s before a fall, no-fall episodes usable to their end")

    # the real rungs
    for rung in ("10k", "100k", "6M"):
        ep_arr = np.load(os.path.join(W.DATA_DIR, f"rung_{rung}.ep.npy"))
        spans = W.choose_episodes(rung, ep_arr)
        rows = sum(e - s for s, e in spans)
        assert rows <= W.MAX_ROWS, (rung, rows)
        if len(ep_arr) <= W.MAX_ROWS:
            assert rows == len(ep_arr)
        again = W.choose_episodes(rung, ep_arr)
        assert again == spans
        print(f"  rung {rung}: {len(spans)} episodes, {rows} rows of {len(ep_arr)}")
    ck_mean = np.zeros(45)
    ck_std = np.ones(45)
    st, ac, starts, info = W.build_starts("10k", ck_mean, ck_std)
    assert st.shape[1:] == (32, 45) and ac.shape[1:] == (32, 12) and len(starts) == len(st) <= W.MAX_ROWS
    a10 = np.load(os.path.join(W.DATA_DIR, "rung_10k.npy"), mmap_mode="r")
    i = len(starts) // 2
    assert np.array_equal(st[i].numpy(), a10[starts[i]:starts[i] + 32, :45].astype(np.float32)), "windows misread"
    assert np.array_equal(ac[i].numpy(), a10[starts[i]:starts[i] + 32, 45:57].astype(np.float32)), "windows misread"
    print(f"  {json.dumps(info)}")
    ok("start windows read from the rung's rows (10k), row caps hold on 10k, 100k and 6M, episode choice repeatable")


# ------------------------------------------------------------------ predictions (./py)
def part_predict():
    import torch
    import train_policy as W
    import train_sim as TS
    import sim_checks as SC
    torch.set_num_threads(2)
    sim, rung = "sim_6M_N8_s0", "6M"

    # train_sim's side: its loader and sim_checks' loading
    path = W.sim_path(sim)
    m0, mean, std, ck = SC.load_sim(path)
    assert ck["rung"] == rung
    ds, stats = TS.load_rung(rung, W.HISTORY + 1)
    assert stats["state_mean"] == list(ck["state_mean"]) and stats["state_std"] == list(ck["state_std"]), \
        "the rung's normalisation differs from the simulator's own"

    # the wrapper's side: Step 2's route, and the lite dataset with the wrapper's windows and normalisation installed
    row = W.sim_row_of(sim)
    adapter = W.TP.build_sim(row)
    adapter.eval()
    st, ac, starts, info = W.build_starts(rung, ck["state_mean"], ck["state_std"])
    print(f"  {json.dumps(info)}")

    class Lite:                                # the lite dataset's own arithmetic (train.py), copied line for line
        action_data_mean = torch.zeros(12)
        action_data_std = torch.ones(12)

        def normalize(self, state_data=None, action_data=None):
            state_data = (state_data - self.state_data_mean) / self.state_data_std if state_data is not None else None
            action_data = (action_data - self.action_data_mean) / self.action_data_std if action_data is not None else None
            return state_data, action_data

        def denormalize(self, state_data, action_data):
            state_data = state_data * self.state_data_std + self.state_data_mean if state_data is not None else None
            action_data = action_data * self.action_data_std + self.action_data_mean if action_data is not None else None
            return state_data, action_data

    lite = Lite()
    W.install_starts(lite, st, ac, ck["state_mean"], ck["state_std"])
    for name in ("extension_data", "contact_data", "termination_data"):
        assert getattr(lite, name).shape[:2] == st.shape[:2]

    # 100 windows that the training loader can also use (the row after the window lies in the same episode)
    a = np.load(os.path.join(W.DATA_DIR, f"rung_{rung}.npy"), mmap_mode="r")
    ep = np.load(os.path.join(W.DATA_DIR, f"rung_{rung}.ep.npy"), mmap_mode="r")
    pick = []
    gen = np.random.default_rng(20277)
    for i in gen.permutation(len(starts)):
        s = int(starts[i])
        if s + 32 < len(ep) and ep[s + 32] == ep[s] and np.searchsorted(ds.starts, s) < len(ds.starts) \
                and ds.starts[np.searchsorted(ds.starts, s)] == s:
            pick.append((int(i), np.searchsorted(ds.starts, s)))
        if len(pick) == 100:
            break
    assert len(pick) == 100, len(pick)
    idx_w, idx_d = (np.array(x) for x in zip(*pick))

    t_st, t_ac, _, _, _ = ds.batch(idx_d)     # train_sim's normalised window
    w_st = lite.state_data[idx_w]             # the wrapper's start states
    w_ac = lite.action_data[idx_w]
    print(f"  max |windows: wrapper - train_sim| states {float((w_st - t_st[:, :32]).abs().max()):.2e}, "
          f"actions {float((w_ac - t_ac[:, :32]).abs().max()):.2e}")
    assert (w_st - t_st[:, :32]).abs().max() == 0, "start windows differ from train_sim's"
    assert (w_ac - t_ac[:, :32]).abs().max() == 0, "start actions differ from train_sim's"

    with torch.no_grad():
        m0.reset()
        out0 = m0(t_st[:, :32], t_ac[:, 1:33])                       # as compute_state_loss feeds it (i = 0)
        pred0 = out0[0]
        adapter.reset()
        new_action = lite.normalize(None, t_ac[:, 32])[1]            # env.imagination_step's own operation
        hist = torch.cat([w_ac[:, 1:], new_action.unsqueeze(1)], 1)
        out1 = adapter.forward(w_st, hist, None)
        pred1 = out1[0]
        dn0 = pred0.double() * torch.as_tensor(std) + torch.as_tensor(mean)
        dn1 = lite.denormalize(pred1, None)[0]
    d_norm = float((pred1 - pred0).abs().max())
    d_den = float((dn1.double() - dn0).abs().max())
    print(f"  max |prediction: wrapper - train_sim| normalised {d_norm:.2e}, denormalised {d_den:.2e}")
    assert d_norm <= 1e-6, d_norm
    ok(f"wrapper equals train_sim on 100 windows of the 6M rung: {d_norm:.2e} (limit 1e-6)")
    assert d_den < 1e-4, d_den
    ok(f"the same in physical units, {d_den:.2e} (float32 arithmetic of the lite dataset)")

    # a deliberately wrong normalisation is caught by the same comparison
    with torch.no_grad():
        adapter.reset()
        off = adapter.forward(((w_st * lite.state_data_std + lite.state_data_mean) - lite.state_data_mean) / (lite.state_data_std * 1.1),
                              hist, None)[0]
    assert float((off - pred0).abs().max()) > 1e-3
    ok("the comparison detects a 10% error in normalisation")


# ------------------------------------------------------------------ one iteration (./py)
def part_short():
    name = "_test_c2_short"
    rc = subprocess.run([os.path.join(AIR, "py"), os.path.join(HERE, "train_policy.py"), "--sim", "sim_10k_N8_s0", "--seed", "0",
                         "--name", name, "--iterations", "1", "--num_envs", "64", "--saves", "1", "--threads", "2",
                         "--job", "test"], cwd=AIR, capture_output=True, text=True)
    open(os.path.join(AIR, ".cache", "logs", "c2_short.log"), "w").write(rc.stdout + rc.stderr)
    assert rc.returncode == 0, rc.stderr[-1500:]
    d = os.path.join(AIR, "runs", "c", name)
    assert os.path.exists(os.path.join(d, "actor_i1.pt")) and os.path.exists(os.path.join(d, "actor.pt"))
    run = os.path.join(AIR, "runs", "c", "lite", "runs", "lite", "s2_" + name)
    logged = [json.loads(line) for line in open(os.path.join(run, "metrics.jsonl"))]
    unc = [r["Imagination/epistemic_uncertainty"] for r in logged if "Imagination/epistemic_uncertainty" in r]
    assert unc and unc[-1] > 0, unc
    ok(f"one iteration on the 10k simulator: epistemic uncertainty {unc[-1]:.4f}, actors saved, {json.load(open(os.path.join(d, 'starts.json')))['windows']} start windows")
    import torch
    sa = torch.load(os.path.join(d, "actor_i1.pt"), map_location="cpu", weights_only=False)["model_state_dict"]
    sb = torch.load(os.path.join(d, "actor.pt"), map_location="cpu", weights_only=False)["model_state_dict"]
    assert sorted(sa) == sorted(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)
    ok("the exact-count save (iteration 1) equals the last checkpoint's export")


def part_weight():
    import torch
    import train_policy as W
    import train as T
    import policy_training  # noqa: F401
    name = "_test_c2_w"
    W.TP.sim_row = lambda sim: W.sim_row_of(sim)
    W.patch_lite(2.5, "sim_10k_N8_s0", {999}, os.path.join(W.RUNS, name))
    cfg = T.resolve_task_config("anymal_d_flat")
    assert cfg.environment_config.uncertainty_penalty_weight == -2.5
    ok("uncertainty weight 2.5 reaches the lite config as -2.5")
    import envs.base as B
    src = open(B.__file__).read()
    assert 'self.uncertainty_penalty_weight * self.epistemic_uncertainty * self._step_dt' in src
    ok("the penalty enters the reward as weight x epistemic uncertainty x dt (lite envs/base.py)")
    print(f"  (torch {torch.__version__})")


# ------------------------------------------------------------------ the rule (./py)
def fixture(root, scores_6, falls_6, score_10k, authors=0.6):
    """scores_6: weight -> {iters: score}; falls_6: weight -> fall rate at every save."""
    for d in ("results/c/eval", "optc"):
        os.makedirs(os.path.join(root, d), exist_ok=True)

    def rec(score, fall, ci=0.02):
        return {"score": score, "score_ci95": [score - ci, score + ci], "standing_reference": 0.09,
                "summary": {"fall_rate": fall, "ret": 1.0, "speed_err": 0.1, "turn_err": 0.1, "diverged": 0}}

    def put(name, score, fall):
        json.dump(rec(score, fall), open(os.path.join(root, "results/c/eval", name + ".json"), "w"))

    json.dump(rec(authors, 0.0), open(os.path.join(root, "results/c_authors.json"), "w"))
    json.dump({"passed": True}, open(os.path.join(root, "results/c/gate_a.json"), "w"))
    for w, by in scores_6.items():
        base = "sim_6M_N8_s0__p0" if w == 1.0 else f"sim_6M_N8_s0__w{w:g}__p0"
        for n, s in by.items():
            put(f"{base}__i{n}", s, falls_6[w])
    for n in (500, 1000, 1500, 2000):
        put(f"sim_10k_N8_s0__p0__i{n}", score_10k, 0.1)


def part_rule():
    import pilot_rule as R
    all_saves = lambda s: {n: s for n in (500, 1000, 1500, 2000)}  # noqa: E731

    def run(scores, falls, s10k=0.2, retry_expect=None):
        root = tempfile.mkdtemp(prefix="c2_rule_", dir=os.path.join(AIR, ".cache"))
        try:
            fixture(root, scores, falls, s10k)
            weights = R.decide(R.Paths(root))
            if retry_expect is not None:
                assert weights == retry_expect, (weights, retry_expect)
            if weights:
                for w in weights:
                    if w not in scores:
                        raise AssertionError(f"fixture has no weight {w}")
            res = R.report(R.Paths(root))
            assert R.decide(R.Paths(root)) == weights and R.report(R.Paths(root))["status"] == res["status"]    # idempotent
            return res, root
        finally:
            pass

    res, root = run({1.0: {500: 0.30, 1000: 0.45, 1500: 0.5, 2000: 0.5}}, {1.0: 0.05}, retry_expect=[])
    assert res["status"] == "PASSED" and res["kept_weight"] == 1.0 and res["iterations"] == 1000, res
    assert len(open(os.path.join(root, "results/c_pilot.txt")).read().splitlines()) <= 40
    ok("Gate B passes at weight 1.0; iteration rule picks the smallest save at 90% of the 2,000-iteration score")

    res, _ = run({1.0: all_saves(0.5), 2.0: all_saves(0.5), 4.0: all_saves(0.6)}, {1.0: 0.25, 2.0: 0.1, 4.0: 0.1},
                 retry_expect=[2.0, 4.0])
    assert res["kept_weight"] == 4.0, res["kept_weight"]
    ok("falls of 20% or more: weights 2 and 4, the higher scorer of those passing is kept")

    res, _ = run({1.0: all_saves(0.2), 0.5: all_saves(0.25), 0.25: all_saves(0.35)}, {1.0: 0.1, 0.5: 0.1, 0.25: 0.1},
                 retry_expect=[0.5, 0.25])
    assert res["kept_weight"] == 0.25, res["kept_weight"]
    ok("low score without falls: weights 0.5 and 0.25")

    res, _ = run({1.0: all_saves(0.5), 2.0: all_saves(0.9), 4.0: all_saves(0.5)}, {1.0: 0.2, 2.0: 0.5, 4.0: 0.1},
                 retry_expect=[2.0, 4.0])
    assert res["kept_weight"] == 4.0, res["kept_weight"]
    ok("exactly 20% falls counts as failing; a high scorer that falls too often is not kept")

    res, root = run({1.0: all_saves(0.5), 2.0: all_saves(0.5), 4.0: all_saves(0.5)}, {1.0: 0.3, 2.0: 0.3, 4.0: 0.3},
                    retry_expect=[2.0, 4.0])
    assert res["status"] == "BLOCKED" and res["kept_weight"] is None
    assert "Pilot job: BLOCKED" in open(os.path.join(root, "optc/DECISIONS.md")).read()
    assert open(os.path.join(root, "optc/DECISIONS.md")).read().count("Pilot job: BLOCKED") == 1, "appended twice"
    assert R.main(["report", "--root", root]) == 3
    ok("no weight passes: BLOCKED, one entry in DECISIONS.md, exit 3")

    res, _ = run({1.0: all_saves(0.5)}, {1.0: 0.05}, s10k=0.49)
    assert res["status"] == "PASSED" and not res["gate_c"]["passed"] and "2k" in res["gate_c"]["action"], res["gate_c"]
    ok("Gate C: overlapping intervals call for the 2k rung, and do not stop the pilot")

    res, _ = run({1.0: all_saves(0.5)}, {1.0: 0.05}, s10k=0.2)
    assert res["gate_c"]["passed"]
    ok("Gate C passes when the 10k policy is lower and the intervals are apart")
    for d in os.listdir(os.path.join(AIR, ".cache")):
        if d.startswith("c2_rule_"):
            shutil.rmtree(os.path.join(AIR, ".cache", d))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=("eval", "starts", "predict", "short", "weight", "rule"))
    a = ap.parse_args()
    {"eval": part_eval, "starts": part_starts, "predict": part_predict, "short": part_short, "weight": part_weight,
     "rule": part_rule}[a.part]()
    print(f"[test_c2] {a.part}: all passed")

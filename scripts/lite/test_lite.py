"""Tests for scripts/lite: adapter equality and a 2-iteration policy training run.  ./py scripts/lite/test_lite.py

 1. adapter == our rebuild, on 200 recorded windows (relative 1e-6), for a 1-head and a 5-head model;
 2. adapter with model_ids and reset_partial == the upstream SystemDynamicsEnsemble with the same weights
    (released checkpoint), over a 32-step history, a 1-step call, a partial reset and another 1-step call;
 3. reset() clears the hidden state; reset_partial() zeroes exactly the listed robots;
 4. 2 training iterations run, with finite rewards and losses, from the released model and from a rebuilt
    parent (with imagination starts restricted to training episodes), and write only into runs/lite/;
 5. the restriction keeps only windows lying in training episodes.
Writes only runs/lite/test_*. Uses 2 threads.
"""
import os
import shutil
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_lite as RL  # noqa: E402  (sets sys.path: stubs, lite scripts, rsl_rl_rwm; patches torch.load)
import common  # noqa: E402
from adapter import LiteDynamicsAdapter  # noqa: E402
from rsl_rl.modules import SystemDynamicsEnsemble  # noqa: E402

torch.set_num_threads(2)
R = common.R
OK = []


def check(name, cond, info=""):
    OK.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + (f"  {info}" if info else ""), flush=True)


def rel(a, b):
    a, b = a.double(), b.double()
    return float(((a - b).abs() / (b.abs() + 1e-12)).max()) if a.numel() else 0.0


def windows(n=200, hist=32, seed=0):
    """n recorded windows of `hist` rows (normalised state, action), each inside one episode."""
    cfg, dat, ep = common.data()
    rng = np.random.default_rng(seed)
    ok = np.array([i for i in range(len(dat) - hist) if ep[i] >= 0 and ep[i] == ep[i + hist - 1]])
    s = rng.choice(ok, n, replace=False)
    idx = s[:, None] + np.arange(hist)[None]
    raw = dat[idx]
    st = torch.as_tensor(R.normalise_state(raw[:, :, R.STATE_COLS], cfg["state_data_mean"], cfg["state_data_std"]),
                         dtype=torch.float32)
    ac = torch.as_tensor(raw[:, :, R.ACTION_COLS], dtype=torch.float32)
    return st, ac


def main():
    t0 = time.time()
    cfg, _, _ = common.data()
    st, ac = windows()
    arch = cfg["architecture_config"]

    # 1. adapter == rebuild --------------------------------------------------------------------
    for label, path, ens in (("1 head  (A_s0)", RL.pick_model("rebuilt:A_s0")[1], 1),
                             ("5 heads (released)", common.W_RELEASED, 5)):
        net = common.load(path, ens)
        ad = LiteDynamicsAdapter(net)
        with torch.no_grad():
            net.reset(); ref = net.forward(st, ac)
            ad.reset(); out = ad.forward(st, ac)
        names = ("mean", "aleatoric", "epistemic", None, "contact", "termination")
        worst = max(rel(out[i], ref[j]) for i, j in ((0, 0), (1, 1), (2, 2), (4, 3), (5, 4)))
        check(f"adapter == rebuild on 200 windows, {label}", worst <= 1e-6, f"worst relative diff {worst:.2e}")
        check(f"adapter outputs: extensions None, shapes ok, {label}",
              out[3] is None and out[0].shape == (200, 45) and out[1].shape == (200,) and out[2].shape == (200,)
              and out[4].shape == (200, 8) and out[5].shape == (200, 1))

    # 2. adapter + model_ids + reset_partial == upstream SystemDynamicsEnsemble ----------------
    up = SystemDynamicsEnsemble(45, 12, 0, 8, 1, "cpu", ensemble_size=5, history_horizon=32, architecture_config=arch)
    ck = torch.load(common.W_RELEASED)
    up.load_state_dict(ck["system_dynamics_state_dict"]); up.eval()
    ad = LiteDynamicsAdapter(common.load(common.W_RELEASED, 5)); ad.eval()
    ids = torch.randint(0, 5, (1, 200, 1), generator=torch.Generator().manual_seed(1))
    st2, ac2 = windows(200, 34, seed=2)
    drop = torch.tensor([3, 17, 120])
    worst = 0.0
    with torch.no_grad():
        up.reset(); ad.reset()
        for seg in ((0, 32), (32, 33), None, (33, 34)):
            if seg is None:
                up.reset_partial(drop); ad.reset_partial(drop)
                continue
            a, b = seg
            u = up.forward(st2[:, a:b], ac2[:, a:b], ids)
            v = ad.forward(st2[:, a:b], ac2[:, a:b], ids)
            for x, y in zip(u, v):
                if x is not None:
                    worst = max(worst, rel(y, x))
    check("adapter == upstream SystemDynamicsEnsemble (model_ids, 3 calls, partial reset)", worst <= 1e-6,
          f"worst relative diff {worst:.2e}")

    # 3. reset semantics -----------------------------------------------------------------------
    with torch.no_grad():
        ad.reset(); ad.forward(st2[:, :32], ac2[:, :32], ids)
        h0 = [b.memory.hidden_states.clone() for b in (ad.net.state_base, ad.net.auxiliary_base)]
        ad.reset_partial(drop)
        h1 = [b.memory.hidden_states for b in (ad.net.state_base, ad.net.auxiliary_base)]
        keep = torch.ones(200, dtype=torch.bool); keep[drop] = False
        zeroed = all(float(h[:, drop].abs().max()) == 0.0 for h in h1)
        same = all(torch.equal(a[:, keep], b[:, keep]) for a, b in zip(h0, h1))
        had_state = all(float(h[:, drop].abs().max()) > 0.0 for h in h0)
        ad.reset_partial(drop)      # idempotent
        ad.reset()
        none = ad.net.state_base.memory.hidden_states is None and ad.net.auxiliary_base.memory.hidden_states is None
        ad.reset_partial(drop)      # no state: must not fail
    check("reset_partial zeroes exactly the listed robots; reset clears; partial reset with no state is a no-op",
          zeroed and same and had_state and none)

    # 4. two training iterations ---------------------------------------------------------------
    for name, model, extra in (("test_released", "released", []), ("test_rebuilt", "rebuilt:A_s0", ["--holdout", "1,8"])):
        shutil.rmtree(os.path.join(RL.AIR, "runs", "lite", name), ignore_errors=True)
        RL.LOGGED.clear()
        s = RL.main(["--name", name, "--model", model, "--num_envs", "64", "--max_iterations", "2"] + extra)
        ok = (s["step_reward_all_finite"] and s["losses_all_finite"] and len(s["seconds_per_iter_each"]) == 2
              and "policy_1.pt" in s["checkpoints"])
        check(f"2 iterations, 64 robots: {model}", ok, f"{s['seconds_per_iter_mean']:.2f} s/iter")
        shutil.rmtree(os.path.join(RL.AIR, "runs", "lite", name), ignore_errors=True)
        if extra:
            kept, total = s["windows_kept_total"]
            check("holdout 1,8 removes windows and keeps some", 0 < kept < total, f"{kept} of {total}")

    # 5. the restriction only keeps training-episode windows -----------------------------------
    _, dat, ep = common.data()
    h, f = 32, 8
    rows = RL.window_rows(dat[:, -1:], h, f)

    n = len(rows)

    class DS:
        def __len__(self):
            return n
    ds = DS()
    for nm in ("state_data", "action_data", "extension_data", "contact_data", "termination_data"):
        setattr(ds, nm, torch.arange(n).float().view(n, 1, 1))
    csv = os.path.join(common.LITE, "assets", "data", "state_action_data_0.csv")
    kept, total = RL.restrict_windows(ds, csv, common.train_episodes((1, 8)), h, f)
    kept_rows = np.array(rows)[ds.state_data.view(-1).long().numpy()]
    inside = all(np.isin(ep[r:r + h + f], common.train_episodes((1, 8))).all() for r in kept_rows)
    outside_excluded = not any(np.isin(ep[r:r + h + f], [1, 8]).any() for r in kept_rows)
    check("restriction keeps only windows inside training episodes", inside and outside_excluded and kept == len(kept_rows),
          f"{kept} of {total}")

    print(f"{sum(OK)}/{len(OK)} checks pass in {time.time() - t0:.0f} s")
    sys.exit(0 if all(OK) else 1)


if __name__ == "__main__":
    main()

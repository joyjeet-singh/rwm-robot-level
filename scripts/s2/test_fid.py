"""
Step 2 (M2) tests of the fidelity and calibration code (STEP2_PLAN.md M2). Training episodes only; no held-out
number and no walking statistic is computed here.

(a) Replay of recorded joint commands on one training window (trajectory 0, start 32, 100 steps) runs and gives
    finite values; the placed start state reproduces the recorded state (all but torque) and its lowest foot
    touches the ground; the first 8 steps of that window equal the 8-step window from the same start.
(b) Window positions equal evalh's: window_plan's starts are evalh.starts for every horizon; the windows
    window_errors visits are exactly those; with the hold-the-last-row predictor it reproduces evalh.window_floor
    on the training arena for h = 1, 8, 24, 100.
(c) Held-out episodes are refused: calibrate.check_episodes, calibrate.evaluate and the calibrate.py command line
    refuse episodes 1 and 8; fidelity.replay refuses them while the calibration is not frozen.
(d) The calibrated-model writer gives a model equal to set_constants on M1's model (written under .cache/), apart
    from the compiler's armature-derived quantities. M1's model's constants are the grid point (0.8, 0.02, 0, 0),
    and that point's text compiles to exactly M1's model. One grid point evaluates on one training trajectory.
(e) Code smoke: the policy-episode loop runs 50 steps with the authors' policy and gives finite values (nothing
    measured); the gait statistics run on training episode 0's recorded rows.

  ./py-mj scripts/s2/test_fid.py [--write]     # --write: results/s2/m2_tests.json (never overwritten)
"""
import json
import os
import subprocess
import sys
import time

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import calibrate as CB  # noqa: E402
import fidelity as F  # noqa: E402
import mj_env as E  # noqa: E402

C, evalh = F.C, F.evalh
OUT = os.path.join(F.RES_DIR, "m2_tests.json")


def test_a():
    env = F.make_env("uncal")
    rep = F.Replayer(env)
    ar = F.raw_arena(F.TRAIN)
    raw = ar["raw"]
    k = 32
    rep.set_state(raw[0, k - 1, :45], raw[0, k - 1:k, 45:57])
    s0 = env.state()
    place_err = float(np.abs(s0[:33] - raw[0, k - 1, :33]).max())
    low = min(env.d.geom_xpos[g][2] - env.m.geom_size[g][0] for g in env.g_foot)
    P100 = rep.run(raw[0, k:k + 100, 45:57])
    unstable = rep.unstable()
    rep.set_state(raw[0, k - 1, :45], raw[0, k - 1:k, 45:57])
    P8 = rep.run(raw[0, k:k + 8, 45:57])
    prefix = float(np.abs(P8 - P100[:8]).max())
    cfg = C.data()[0]
    e = np.abs((P100 - cfg["state_data_mean"]) / cfg["state_data_std"] - ar["st"][0, k:k + 100].numpy()).mean(1)
    ok = (np.all(np.isfinite(P100)) and not unstable and place_err < 1e-6 and abs(low) < 1e-9 and prefix == 0.0
          and np.all(np.isfinite(e)))
    return ok, {"place_err": place_err, "lowest_foot_m": float(low), "prefix_maxdiff": prefix,
                "unstable": unstable, "err_step1": float(e[0]), "err_step8_mean": float(e[:8].mean()),
                "err_step100_mean": float(e.mean())}


def test_b():
    plan, starts = F.window_plan()
    same = all(starts[h] == evalh.starts(h, evalh.STRIDE[h]) for h in F.HORIZONS)
    ar = F.raw_arena(F.TRAIN)
    st = ar["st"]
    seen = {h: set() for h in F.HORIZONS}

    def hold(b, k, L):
        for h in F.HORIZONS:
            if k in starts[h] and h <= L:
                seen[h].add((b, k))
        return np.repeat(ar["raw"][b, k - 1:k, :45], L, 0), False

    cfg = C.data()[0]
    # hold-last in raw units, compared in normalised units: identical to evalh's floor up to float32 rounding
    res, _ = F.window_errors(hold, st.numpy(), F.HORIZONS, ar["raw"])
    B = st.shape[0]
    visited = all(seen[h] == {(b, k) for b in range(B) for k in evalh.starts(h, evalh.STRIDE[h])}
                  for h in F.HORIZONS)
    diffs = {h: float(np.abs(res[h]["E"] - evalh.window_floor(st, h)).max()) for h in F.HORIZONS}
    ok = same and visited and max(diffs.values()) < 1e-5
    return ok, {"starts_equal": same, "windows_visited_equal": visited, "floor_maxdiff": diffs,
                "n_windows_per_traj": {h: len(starts[h]) for h in F.HORIZONS}, "B": B}


def refused(fn):
    try:
        fn()
    except SystemExit as ex:
        return "held-out" in str(ex) or "not frozen" in str(ex)
    return False


def test_c():
    r1 = refused(lambda: CB.check_episodes((1,)))
    r2 = refused(lambda: CB.evaluate(CB.POINTS[0], episodes=(0, 8)))
    cli = subprocess.run([sys.executable, os.path.join(HERE, "calibrate.py"), "--episodes", "0,1",
                          "--worker", "0", "--workers", "1"], capture_output=True, text=True)
    r3 = cli.returncode != 0 and "held-out" in (cli.stderr + cli.stdout)
    frozen = os.path.exists(F.CAL_JSON)
    r4 = refused(lambda: F.replay(F.make_env("uncal"), F.HELD_OUT, horizons=(1,))) if not frozen else None
    ok = r1 and r2 and r3 and (r4 is None or r4)
    return ok, {"check_episodes": r1, "evaluate": r2, "cli": r3, "replay_before_freeze": r4,
                "calibration_frozen_at_test": frozen}


def test_d():
    p = {"friction": 1.2, "timeconst": 0.005, "armature": 0.05, "damping": 1.0}
    path = os.path.join(C.AIR, ".cache", f"m2_test_cal_{os.getpid()}.xml")
    CB.write_cal_xml(p, path)                       # asserts equality with set_constants internally
    m = mujoco.MjModel.from_xml_path(path)
    os.remove(path)
    base = mujoco.MjModel.from_xml_path(E.MODEL_XML)
    env = E.MjEnv(reward=False)
    legs = env.vadr
    unc = {"friction": float(np.unique(base.geom_friction[:, 0])[0]),
           "timeconst": float(np.unique(base.geom_solref[:, 0])[0]),
           "armature": float(np.unique(base.dof_armature[legs])[0]),
           "damping": float(np.unique(base.dof_damping[legs])[0])}
    uniq = all(len(np.unique(a)) == 1 for a in (base.geom_friction[:, 0], base.geom_solref[:, 0],
                                                  base.dof_armature[legs], base.dof_damping[legs]))
    # the uncalibrated grid point's text compiles to exactly M1's model (every numeric array)
    u = mujoco.MjModel.from_xml_string(CB.cal_xml_text(unc))
    same = all(np.array_equal(getattr(base, f), getattr(u, f)) for f in dir(base)
               if isinstance(getattr(base, f, None), np.ndarray) and not f.startswith("_"))
    # one grid point evaluated on one training trajectory (the job's unit of work)
    t0 = time.time()
    r = CB.evaluate(CB.POINTS[1], traj=[0])
    ok = (uniq and unc == {"friction": 0.8, "timeconst": 0.02, "armature": 0.0, "damping": 0.0} and same
          and float(m.geom_friction[0, 0]) == 1.2 and float(m.dof_damping[legs][0]) == 1.0
          and np.isfinite(r["E8"]) and np.isfinite(r["E1"]))
    return ok, {"uncalibrated_constants": unc, "uncal_text_equals_M1_model": same, "written_model_ok": True,
                "one_traj_seconds": round(time.time() - t0, 2)}


def test_e():
    import s1_policy as P
    env = F.make_env("uncal", reward=True)
    actor = F.torch_actor(P.load_actor())
    cmd = F.schedule(0, 51)
    off, yaw = F.eval_start(0, 0)
    env.reset(joint_offset=off, yaw=yaw, cmd=cmd[0])
    fin = True
    for k in range(50):
        env.cmd = cmd[k]
        _, info = env.step(actor(E.observation(env.s, env.cmd, env.last_action)))
        fin &= bool(np.all(np.isfinite(info["state"])) and np.isfinite(info["reward"]))
    _, dat, ep = C.data()
    rows = np.where(ep == 0)[0][1:]
    g = F.gait_stats(dat[rows][:, 0:2], dat[rows][:, 5], F.schedule(0, len(rows) + 1)[1:],
                     (dat[rows][:, F.FOOT_COLS] > 0.5).astype(float))
    same_starts = np.allclose(F.eval_start(3, 5)[0], F.eval_start(3, 5)[0]) and \
        not np.allclose(F.eval_start(3, 5)[0], F.eval_start(3, 6)[0])
    ok = fin and env.t == 50 and all(np.isfinite(v) for v in (g["two_feet"], g["speed_err"])) and same_starts
    return ok, {"steps": env.t, "finite": fin, "seeded_starts": bool(same_starts)}


def main():
    import torch
    torch.set_num_threads(2)
    res, ok_all = {}, True
    for name, fn in [("a", test_a), ("b", test_b), ("c", test_c), ("d", test_d), ("e", test_e)]:
        t0 = time.time()
        ok, info = fn()
        info["seconds"] = round(time.time() - t0, 2)
        res[name] = {"pass": bool(ok), **info}
        ok_all &= bool(ok)
        print(f"({name}) {'PASS' if ok else 'FAIL'}", json.dumps(info, default=float)[:400])
    res["all_pass"] = ok_all
    res["sha256"] = {f: C.sha256(os.path.join(HERE, f)) for f in ("fidelity.py", "calibrate.py", "test_fid.py")}
    if "--write" in sys.argv:
        F.write_new(OUT, res)
        print("wrote", OUT)
    print("ALL PASS" if ok_all else "SOME FAILED")


if __name__ == "__main__":
    main()

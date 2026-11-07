"""
Step 2 (M2): calibration of the hand-built MuJoCo ANYmal D model (STEP2_PLAN.md, "Calibration"; fixed design).

- Grid over four constants (108 points): foot friction 0.6, 0.8, 1.0, 1.2; contact time constant 0.005, 0.01,
  0.02 s; joint armature 0, 0.02, 0.05 kg m^2; joint damping 0, 0.5, 1.0 N m s/rad (mj_env.set_constants applies
  them; friction to every shape and the floor, armature and damping to the 12 leg joints).
- Objective: the mean 8-step error E8 (scripts/s2/fidelity.py replay, evalh's windows) over the training
  episodes' arena (episodes 0, 2-7, 9: 16 trajectories); ties broken by the one-step error E1. Exact ties left after
  that (MuJoCo's refsafe runs the time constant 0.005 s as 2 x 0.005 = 0.01 s, so those points are identical) go
  to the larger time constant, then to the earlier grid point (step2/DECISIONS.md M1, M2).
- Held-out episodes 1 and 8 are refused.
- Resumable: one file per grid point in results/s2/calib/; finished points are skipped. Workers split the grid.
- Each grid point is evaluated from its own compiled model text (cal_xml_text), the same text --freeze writes.
- --freeze picks the point, writes step2/model/anymal_d_cal.xml (M1's model text with the chosen constants)
  and step2/model/calibration.json (frozen: true, with sha256 sums), and makes both read-only. Nothing is
  overwritten.

  ./py-mj scripts/s2/calibrate.py --worker 0 --workers 4
  ./py-mj scripts/s2/calibrate.py --freeze
"""
import argparse
import glob
import itertools
import json
import os
import stat
import sys
import time

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fidelity as F  # noqa: E402
import mj_env as E  # noqa: E402

C = F.C
GRID = {"friction": (0.6, 0.8, 1.0, 1.2), "timeconst": (0.005, 0.01, 0.02),
        "armature": (0.0, 0.02, 0.05), "damping": (0.0, 0.5, 1.0)}
POINTS = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
CAL_DIR = os.path.join(F.RES_DIR, "calib")


def check_episodes(episodes):
    bad = sorted(set(episodes) & set(F.HELD_OUT))
    if bad:
        raise SystemExit(f"calibration refuses held-out episodes {bad}: training episodes only")
    return tuple(episodes)


def point_name(i, p):
    return f"p{i:03d}_f{p['friction']}_t{p['timeconst']}_a{p['armature']}_d{p['damping']}"


def evaluate(p, episodes=F.TRAIN, traj=None):
    """E8 and E1 (evalh windows, replay of recorded joint commands) at constants p on training episodes."""
    episodes = check_episodes(episodes)
    path = point_xml(p)
    env = E.MjEnv(xml=path, reward=False)
    os.remove(path)
    r = F.replay(env, episodes, horizons=(1, 8), traj=traj)
    h = r["horizons"]
    return {"constants": p, "episodes": list(episodes), "xml_sha256": sha_text(cal_xml_text(p)),
            "effective_timeconst": max(p["timeconst"], 2 * env.m.opt.timestep),
            "E8": h["8"]["mean"], "E1": h["1"]["mean"], "E8_per_traj": h["8"]["per_traj"],
            "E1_per_traj": h["1"]["per_traj"], "torque_Nm_E8": h["8"]["torque_Nm"],
            "groups_E8": h["8"]["groups"], "unstable_windows": int(sum(r["unstable_windows"])),
            "seconds": r["seconds"]}


def sha_text(t):
    return __import__("hashlib").sha256(t.encode()).hexdigest()


def n_done():
    return len(glob.glob(os.path.join(CAL_DIR, "p*.json")))


def worker(i, n, episodes=F.TRAIN):
    episodes = check_episodes(episodes)
    os.makedirs(CAL_DIR, exist_ok=True)
    for idx, p in enumerate(POINTS):
        if idx % n != i:
            continue
        path = os.path.join(CAL_DIR, point_name(idx, p) + ".json")
        if os.path.exists(path):
            continue
        res = evaluate(p, episodes)
        res["index"] = idx
        F.write_new(path, res)
        F.progress("calib", "calibrate grid", n_done(), len(POINTS))
        print(f"[{idx:3d}] {p} E8 {res['E8']:.5f} E1 {res['E1']:.5f} ({res['seconds']:.0f} s)", flush=True)


def choose(rows):
    """Lowest E8; ties by E1; exact ties left: larger time constant, then earlier grid point."""
    return min(rows, key=lambda r: (r["E8"], r["E1"], -r["constants"]["timeconst"], r["index"]))


DERIVED = ("actuator_acc0", "body_invweight0", "dof_M0", "dof_invweight0")   # recomputed by the compiler


def cal_xml_text(p):
    """
    M1's model with the constants p, as MJCF text. The XML text is edited, not rebuilt (MjSpec.to_xml rounds
    numbers to 6 digits, which would undo M1's inertia precision): the default geom's friction and time constant
    (every shape and the floor inherit them) and each leg joint's armature and damping. Every numeric array of the
    compiled model equals mj_env.set_constants applied to M1's model, except the compile-time quantities in
    DERIVED, which the compiler recomputes from the armature (set_constants at run time leaves them stale; they
    enter the contact solver, so grid points are evaluated from this text, the same text that is frozen).
    """
    import xml.etree.ElementTree as ET
    tree = ET.parse(E.MODEL_XML, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
    root = tree.getroot()
    dg = root.find("default/geom")
    fr = dg.get("friction").split()
    dg.set("friction", " ".join([repr(float(p["friction"]))] + fr[1:]))
    sr = dg.get("solref").split()
    dg.set("solref", " ".join([repr(float(p["timeconst"]))] + sr[1:]))
    assert all(g.get("friction") is None and g.get("solref") is None for g in root.iter("geom") if g is not dg)
    n = 0
    for j in root.iter("joint"):
        if j.get("name") in E.JOINTS:
            j.set("armature", repr(float(p["armature"])))
            j.set("damping", repr(float(p["damping"])))
            n += 1
    assert n == 12
    xml = ET.tostring(root, encoding="unicode")
    ref = mujoco.MjModel.from_xml_path(E.MODEL_XML)
    E.set_constants(ref, **p)
    new = mujoco.MjModel.from_xml_string(xml)
    for f in dir(ref):
        a = getattr(ref, f, None)
        if isinstance(a, np.ndarray) and a.dtype.kind in "fiu" and not f.startswith("_") and f not in DERIVED:
            assert a.shape == getattr(new, f).shape and np.array_equal(a, getattr(new, f)), f
    assert new.opt.timestep == ref.opt.timestep and new.nq == ref.nq and new.ngeom == ref.ngeom
    return xml


def point_xml(p):
    """The grid point's model file under .cache/ (rewritten each time; the frozen copy is step2/model/)."""
    d = os.path.join(C.AIR, ".cache", "s2_calib_xml")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"f{p['friction']}_t{p['timeconst']}_a{p['armature']}_d{p['damping']}_{os.getpid()}.xml")
    open(path, "w").write(cal_xml_text(p))
    return path


def write_cal_xml(p, path):
    xml = cal_xml_text(p)
    if os.path.exists(path):
        raise SystemExit(f"{path} exists; not overwritten")
    open(path, "w").write(xml)
    return path


def freeze():
    if os.path.exists(F.CAL_JSON):
        print("freeze: done already")
        return F.frozen_calibration()
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(CAL_DIR, "p*.json")))]
    if len(rows) != len(POINTS) or sorted(r["index"] for r in rows) != list(range(len(POINTS))):
        raise SystemExit(f"grid incomplete: {len(rows)} of {len(POINTS)} points")
    for r in rows:
        check_episodes(r["episodes"])
        assert r["constants"] == POINTS[r["index"]]
    best = choose(rows)
    ties = [r["index"] for r in rows if r["E8"] == best["E8"] and r["E1"] == best["E1"]]
    if not os.path.exists(F.CAL_XML):
        write_cal_xml(best["constants"], F.CAL_XML)
    if C.sha256(F.CAL_XML) != best["xml_sha256"]:
        raise SystemExit("the calibrated model file differs from the model the grid point was evaluated with")
    rank = sorted(rows, key=lambda r: (r["E8"], r["E1"], -r["constants"]["timeconst"], r["index"]))
    cal = {
        "what": "Frozen calibration of the hand-built MuJoCo ANYmal D model (scripts/s2/calibrate.py)",
        "frozen": True, "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "episodes": list(F.TRAIN), "grid": GRID,
        "rule": "min mean E8 over the training arena; ties by E1; exact ties: larger time constant, earlier point",
        "chosen": best["constants"], "chosen_index": best["index"],
        "effective_timeconst": best["effective_timeconst"], "exact_ties": ties,
        "objective": {"E8": best["E8"], "E1": best["E1"], "torque_Nm_E8": best["torque_Nm_E8"]},
        "uncalibrated_point": {k: v for k, v in zip(GRID, (0.8, 0.02, 0.0, 0.0))},
        "top10": [{"index": r["index"], **r["constants"], "E8": r["E8"], "E1": r["E1"]} for r in rank[:10]],
        "worst": {"index": rank[-1]["index"], **rank[-1]["constants"], "E8": rank[-1]["E8"]},
        "unstable_windows_total": int(sum(r["unstable_windows"] for r in rows)),
        "base_xml_sha256": C.sha256(E.MODEL_XML), "cal_xml_sha256": C.sha256(F.CAL_XML),
        "pd_fit_sha256": C.sha256(os.path.join(F.RES_DIR, "pd_fit.json")),
    }
    F.write_new(F.CAL_JSON, cal)
    for pth in (F.CAL_XML, F.CAL_JSON):
        os.chmod(pth, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    F.progress("calib", "calibration frozen")
    print("frozen:", best["constants"], f"E8 {best['E8']:.5f} E1 {best['E1']:.5f}", "ties", ties)
    return cal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--episodes", default=",".join(map(str, F.TRAIN)))
    ap.add_argument("--freeze", action="store_true")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(1)
    if a.freeze:
        freeze()
    else:
        worker(a.worker, a.workers, tuple(int(x) for x in a.episodes.split(",")))


if __name__ == "__main__":
    main()

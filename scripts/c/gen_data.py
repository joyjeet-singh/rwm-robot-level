"""Option C (C1): MuJoCo data generator. Hand-built calibrated MuJoCo ANYmal D with a fitted PD motor; not Isaac Lab,
not hardware.

    ./py-mj scripts/c/gen_data.py --worker 0 --nworkers 2     # one of two workers; skips shards that exist
    ./py-mj scripts/c/gen_data.py --finish                    # after every worker: manifest, rungs, rung files

Design (OPTIONC_PLAN.md, "Data from MuJoCo"); choices it does not settle are in optc/DECISIONS.md, C1:
  * episode: 20 s at 50 Hz, ended early by a fall (Step 2's rule). Row 0 is the reset state with a zero action, as in
    the recording; row k >= 1 holds the state after step k and the joint command sent during step k, so a row r pairs
    with the command of row r + 1 exactly as in the recorded data. 999 steps give the recording's 1,000 rows;
  * commands: each axis uniform in -1 to 1, a new one every 10 s;
  * physics: half the episodes calibrated, half with foot friction 0.4-0.8, base payload -5 to +5 kg, motor gains +-10%;
  * pushes: up to 0.5 m/s every 10-15 s in every training and mixed-test episode;
  * sources by share of transitions: see SOURCES;
  * columns: 45 state, 12 action, 8 contacts, fall, episode id, source id (float32).

Every episode draws from np.random.default_rng([ROOT, set, source, batch, episode]), so a shard is the same whichever
worker makes it and whenever. Shards appear by atomic rename and a finished shard is never redone.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AIR = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(AIR, "scripts", "s2"))
sys.path.insert(0, os.path.join(AIR, "scripts"))
import mujoco  # noqa: E402

import fidelity as F  # noqa: E402
import mj_env as E  # noqa: E402

ROOT = 20273
RUNG_SEED = 20274
STEPS = 999                     # steps per full episode; with the reset row, the recording's 1,000 rows
CMD_STEPS = 500                 # a new velocity command every 10 s
PUSH_STEPS = (500, 750)         # 10-15 s between pushes
PUSH_MAX = 0.5                  # m/s
FRICTION = (0.4, 0.8)
PAYLOAD = (-5.0, 5.0)           # kg
GAIN = (0.9, 1.1)
SMOOTH_TAU = 0.5                # s, time constant of the smooth random joint commands
SMOOTH_STD = 0.5                # action units (0.25 rad at the action scale of 0.5)
CHUNK = 25_000                  # transitions per shard
TRAIN_TOTAL = 6_000_000
RUNGS = {"10k": 10_000, "100k": 100_000, "1M": 1_000_000, "6M": 6_000_000}
N_COLS = 68
EP_MULT, BATCH_MULT = 500_000, 4_000       # episode id = source * EP_MULT + batch * BATCH_MULT + episode (< 2^24)
SET_CODES = {"mixed": 27, "expert": 28, "d6": 29}
SET_TAGS = {"train": 1, "mixed": 2, "expert": 3, "d6": 4}
DATA_DIR = os.path.join(AIR, "optc", "data")
MANIFEST = os.path.join(AIR, "optc", "data_manifest.json")
PROGRESS = os.path.join(AIR, "results", "c", "progress.json")
RUNS_S2 = os.path.join(AIR, "runs", "s2")


def trained_policies():
    names = sorted(d for d in os.listdir(RUNS_S2)
                   if not d.startswith("_") and os.path.exists(os.path.join(RUNS_S2, d, "actor.pt")))
    if len(names) != 22:
        raise SystemExit(f"expected Step 2's 22 trained policies, found {len(names)}")
    return names


def sources():
    """Source table: id, name, group, share of transitions, how it acts."""
    out = [{"id": 0, "name": "authors", "group": "expert", "share": 0.30, "actor": "authors", "noise": 0.0},
           {"id": 1, "name": "authors_noise0.1", "group": "noise0.1", "share": 0.125, "actor": "authors", "noise": 0.1},
           {"id": 2, "name": "authors_noise0.3", "group": "noise0.3", "share": 0.125, "actor": "authors", "noise": 0.3}]
    names = trained_policies()
    for i, n in enumerate(names):
        out.append({"id": 3 + i, "name": n, "group": "trained", "share": 0.25 / len(names), "actor": n, "noise": 0.0})
    out.append({"id": 25, "name": "smooth_random", "group": "smooth", "share": 0.10, "actor": None, "noise": 0.0})
    out.append({"id": 26, "name": "authors_noise0.6", "group": "noise0.6", "share": 0.10, "actor": "authors",
                "noise": 0.6})
    assert abs(sum(s["share"] for s in out) - 1.0) < 1e-9
    return out


GROUPS = ("expert", "noise0.1", "noise0.3", "trained", "smooth", "noise0.6")


class World:
    """The calibrated environment, and the per-episode physics changes (undone exactly)."""

    def __init__(self):
        self.env = F.make_env("cal", reward=False)
        env, m = self.env, self.env.m
        self.feet = list(env.g_foot) + list(env.g_shank)
        self.nominal = {"friction": m.geom_friction[self.feet].copy(), "priority": m.geom_priority[self.feet].copy(),
                        "mass": float(m.body_mass[env.base]), "inertia": m.body_inertia[env.base].copy(),
                        "kp": env.motor.kp, "kd": env.motor.kd}
        self.varied = False

    def apply(self, phys):
        env, m, n = self.env, self.env.m, self.nominal
        if phys is None:
            if self.varied:
                m.geom_friction[self.feet] = n["friction"]
                m.geom_priority[self.feet] = n["priority"]
                m.body_mass[env.base] = n["mass"]
                m.body_inertia[env.base] = n["inertia"]
                env.motor.kp, env.motor.kd = n["kp"], n["kd"]
                mujoco.mj_setConst(m, env.d)
                self.varied = False
            return
        # MuJoCo takes the larger friction of two touching shapes unless one has the higher priority; the floor is
        # at the calibrated 0.6, so the feet get priority 1 and their friction is the one used.
        m.geom_friction[self.feet, 0] = phys["friction"]
        m.geom_priority[self.feet] = 1
        mass = n["mass"] + phys["payload"]
        m.body_mass[env.base] = mass
        m.body_inertia[env.base] = n["inertia"] * mass / n["mass"]
        env.motor.kp, env.motor.kd = n["kp"] * phys["kp_scale"], n["kd"] * phys["kd_scale"]
        mujoco.mj_setConst(m, env.d)
        self.varied = True


def draw_spec(rng, varied, pushes):
    """Everything an episode needs except the step-by-step action noise, drawn in a fixed order."""
    off = rng.uniform(-F.JOINT_JITTER, F.JOINT_JITTER, 12)
    yaw = float(rng.uniform(-np.pi, np.pi))
    c0, c1 = rng.uniform(-1.0, 1.0, 3), rng.uniform(-1.0, 1.0, 3)
    n = max(STEPS + 2, 1001)
    cmd = np.zeros((n, 3))
    cmd[:CMD_STEPS], cmd[CMD_STEPS:] = c0, c1
    coin = rng.random()
    draws = dict(friction=float(rng.uniform(*FRICTION)), payload=float(rng.uniform(*PAYLOAD)),
                 kp_scale=float(rng.uniform(*GAIN)), kd_scale=float(rng.uniform(*GAIN)))
    plist, t = {}, 0
    while True:
        t += int(rng.integers(PUSH_STEPS[0], PUSH_STEPS[1] + 1))
        if t >= STEPS:
            break
        mag, ang = rng.uniform(0.0, PUSH_MAX), rng.uniform(-np.pi, np.pi)
        plist[t] = (float(mag * np.cos(ang)), float(mag * np.sin(ang)))
    return {"joint_offset": off, "yaw": yaw, "cmd": cmd, "physics": draws if (varied and coin < 0.5) else None,
            "pushes": plist if pushes else {}}


BAD = (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL, mujoco.mjtWarning.mjWARN_BADQACC)


def simulate(world, act, spec, noise=0.0, smooth=False, rng=None, steps=STEPS):
    """One episode. Returns (rows (T + 1, 66) float64, fell, broken).

    broken: MuJoCo warned of a divergence or a state was not finite. The bad step is not stored and the episode ends
    there, without a fall label; it is counted in the manifest.
    """
    env = world.env
    world.apply(spec["physics"])
    env.reset(joint_offset=spec["joint_offset"], yaw=spec["yaw"], cmd=spec["cmd"][0])
    rows = np.zeros((steps + 1, 66))
    rows[0, :45] = env.s
    rows[0, 57:65] = env.contacts()
    if smooth:
        alpha = float(np.exp(-E.STEP_DT / SMOOTH_TAU))
        a_prev = SMOOTH_STD * rng.standard_normal(12)
    n, fell, broken = 1, False, False
    for k in range(steps):
        env.cmd = np.asarray(spec["cmd"][k])
        if smooth:
            a_prev = alpha * a_prev + math.sqrt(1.0 - alpha * alpha) * SMOOTH_STD * rng.standard_normal(12)
            a = a_prev.copy()
        else:
            a = act(E.observation(env.s, env.cmd, env.last_action))
            if noise > 0.0:
                a = a + noise * rng.standard_normal(12)
        if k in spec["pushes"]:
            env.d.qvel[0] += spec["pushes"][k][0]
            env.d.qvel[1] += spec["pushes"][k][1]
        _, info = env.step(a)
        if not np.all(np.isfinite(info["state"])) or any(env.d.warning[w].number > 0 for w in BAD):
            broken = True
            break
        rows[n, :45], rows[n, 45:57], rows[n, 57:65] = info["state"], a, info["contacts"]
        rows[n, 65] = float(info["fell"])
        n += 1
        if info["fell"]:
            fell = True
            break
    return rows[:n], fell, broken


def actors():
    """A cache: actor name -> function obs (48,) -> joint command (12,)."""
    import s1_policy as P
    import torch
    torch.set_num_threads(1)
    cache = {}

    def get(name):
        if name not in cache:
            path = None if name == "authors" else os.path.join(RUNS_S2, name, "actor.pt")
            cache[name] = F.torch_actor(P.load_actor(path))
        return cache[name]
    return get


def set_sources(kind, n, table):
    """The source of each episode of a held-out set."""
    by_group = {g: [s for s in table if s["group"] == g] for g in GROUPS}
    if kind != "mixed":
        return [by_group["expert"][0]] * n
    counts = {"expert": 0.30, "noise0.1": 0.125, "noise0.3": 0.125, "trained": 0.25, "smooth": 0.10, "noise0.6": 0.10}
    rng = np.random.default_rng([ROOT, 99])
    sizes = {g: int(round(n * f)) for g, f in counts.items()}
    sizes["expert"] += n - sum(sizes.values())
    out = []
    for g in GROUPS:
        for _ in range(sizes[g]):
            out.append(by_group[g][int(rng.integers(len(by_group[g])))])
    assert len(out) == n
    return out


def shard_path(data_dir, kind, source=None, batch=None):
    if kind == "train":
        return os.path.join(data_dir, f"train_s{source:02d}_b{batch:03d}.npy")
    return os.path.join(data_dir, f"test_{kind}.npy" if kind != "d6" else "d6_train.npy")


def make_plan(args, table):
    """All shards, in a fixed shuffled order, so that the two workers share the kinds of work evenly."""
    plan = []
    for s in table:
        target = s["share"] * args.train_total
        n = max(1, math.ceil(target / args.chunk))
        per = math.ceil(target / n)
        plan += [{"kind": "train", "source": s["id"], "batch": b, "rows": per} for b in range(n)]
    plan += [{"kind": k, "episodes": m} for k, m in (("mixed", args.n_mixed), ("expert", args.n_expert),
                                                     ("d6", args.n_d6)) if m > 0]
    order = np.random.default_rng([ROOT, 98]).permutation(len(plan))
    return [plan[i] for i in order]


def pack(rows, ep_id, source):
    out = np.zeros((len(rows), N_COLS), dtype=np.float32)
    out[:, :66] = rows
    out[:, 66], out[:, 67] = ep_id, source
    return out


def save_shard(path, rows):
    tmp = path + f".{os.getpid()}.tmp"
    with open(tmp, "wb") as fh:
        np.save(fh, rows)
    os.replace(tmp, path)


def make_train_shard(world, get, s, batch, want, path):
    act = get(s["actor"]) if s["actor"] else None
    smooth = s["actor"] is None
    out, have, i, nbroken = [], 0, 0, 0
    while have < want:
        if i >= BATCH_MULT:
            raise SystemExit(f"source {s['name']} batch {batch}: {BATCH_MULT} episodes before {want} transitions")
        rng = np.random.default_rng([ROOT, SET_TAGS["train"], s["id"], batch, i])
        spec = draw_spec(rng, varied=True, pushes=True)
        rows, fell, broken = simulate(world, act, spec, noise=s["noise"], smooth=smooth, rng=rng)
        nbroken += int(broken)
        ep_id = s["id"] * EP_MULT + batch * BATCH_MULT + i
        out.append(pack(rows, ep_id, s["id"]))
        have += len(rows)
        i += 1
    world.apply(None)
    save_shard(path, np.concatenate(out))
    return have, nbroken


def make_set_shard(world, get, kind, n, table, path):
    srcs = set_sources(kind, n, table)
    code, out = SET_CODES[kind], []
    mixed = kind == "mixed"
    for i, s in enumerate(srcs):
        rng = np.random.default_rng([ROOT, SET_TAGS[kind], s["id"], 0, i])
        spec = draw_spec(rng, varied=mixed, pushes=mixed)
        act = get(s["actor"]) if s["actor"] else None
        rows, _, _ = simulate(world, act, spec, noise=s["noise"], smooth=s["actor"] is None, rng=rng)
        out.append(pack(rows, code * EP_MULT + i, s["id"]))
    world.apply(None)
    save_shard(path, np.concatenate(out))


def progress(step, done, total, path=PROGRESS):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rec = {"job": "data", "step": step, "done": int(done), "total": int(total),
           "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh)
    os.replace(tmp, path)


def worker(args):
    table = sources()
    plan = make_plan(args, table)
    os.makedirs(args.data_dir, exist_ok=True)
    mine = [p for i, p in enumerate(plan) if i % args.nworkers == args.worker]
    world, get = World(), actors()
    by_id = {s["id"]: s for s in table}
    t0, rows_done, batches = time.time(), 0, 0
    for p in mine:
        path = shard_path(args.data_dir, p["kind"], p.get("source"), p.get("batch"))
        if os.path.exists(path):
            continue
        if args.time_limit and time.time() - t0 > args.time_limit:
            break
        if p["kind"] == "train":
            n, _ = make_train_shard(world, get, by_id[p["source"]], p["batch"], p["rows"], path)
            rows_done += n
        else:
            make_set_shard(world, get, p["kind"], p["episodes"], table, path)
        batches += 1
        done = sum(os.path.exists(shard_path(args.data_dir, q["kind"], q.get("source"), q.get("batch"))) for q in plan)
        if not args.no_progress:
            progress("generate shards", done, len(plan))
    return {"worker": args.worker, "rows": rows_done, "shards": batches, "seconds": time.time() - t0}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def episode_table(path):
    """(ep_id, source, start, length, fell, thigh_rows) of each episode in a shard, and its row count."""
    a = np.load(path, mmap_mode="r")
    ep = np.asarray(a[:, 66]).astype(np.int64)
    src = np.asarray(a[:, 67]).astype(np.int64)
    fall = np.asarray(a[:, 65]) > 0.5
    thigh = np.asarray(a[:, 57:61]).sum(1) > 0
    cuts = np.flatnonzero(ep[1:] != ep[:-1]) + 1
    starts, ends = np.r_[0, cuts], np.r_[cuts, len(ep)]
    return [(int(ep[s]), int(src[s]), int(s), int(e - s), bool(fall[s:e].any()), int(thigh[s:e].sum()))
            for s, e in zip(starts, ends)], len(ep)


def group_sequences(eps, table):
    """Per group, one fixed random order of whole episodes; a rung is a prefix of it, so rungs are nested.

    A group with several sources (the 22 trained policies) takes its next episode from the source with the fewest
    transitions so far, so the equal shares hold from small rungs on.
    """
    rng = np.random.default_rng(RUNG_SEED)
    by_source = {}
    for e in eps:
        by_source.setdefault(e["source"], []).append(e)
    out = {}
    for g in GROUPS:
        members = [s["id"] for s in table if s["group"] == g]
        order = {m: [by_source[m][i] for i in rng.permutation(len(by_source[m]))] for m in members if m in by_source}
        tie = {m: int(r) for r, m in enumerate(rng.permutation(members))}
        ptr, count, seq = {m: 0 for m in order}, {m: 0 for m in order}, []
        while True:
            live = [m for m in order if ptr[m] < len(order[m])]
            if not live:
                break
            m = min(live, key=lambda x: (count[x], tie[x]))
            e = order[m][ptr[m]]
            seq.append(e)
            ptr[m] += 1
            count[m] += e["length"]
        out[g] = seq
    return out


def rung_lists(seqs, table, sizes):
    share = {g: sum(s["share"] for s in table if s["group"] == g) for g in GROUPS}
    rungs = {}
    for name, size in sizes.items():
        chosen, counts = [], {}
        for g in GROUPS:
            target, cum, n = share[g] * size, 0, 0
            while cum < target and n < len(seqs[g]):
                cum += seqs[g][n]["length"]
                n += 1
            if cum < target:
                raise SystemExit(f"rung {name}: group {g} has only {cum} transitions, needs {target:.0f}")
            if n > 1 and target - (cum - seqs[g][n - 1]["length"]) < cum - target:   # the nearer whole-episode cut
                n -= 1
                cum -= seqs[g][n]["length"]
            chosen += seqs[g][:n]
            counts[g] = {"episodes": n, "transitions": cum}
        rungs[name] = {"episodes": chosen, "by_group": counts}
    return rungs


def write_rung_file(data_dir, name, episodes):
    """One float32 array of the rung's rows (whole episodes, 68 columns), with its episode ids, falls and statistics."""
    n = sum(e["length"] for e in episodes)
    path = os.path.join(data_dir, f"rung_{name}.npy")
    tmp = path + f".{os.getpid()}.tmp"
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(n, N_COLS))
    pos, opened = 0, {}
    by_file = sorted(episodes, key=lambda e: (e["file"], e["start"]))
    for e in by_file:
        if e["file"] not in opened:
            opened.clear()                                 # one shard open at a time keeps the memory small
            opened[e["file"]] = np.load(os.path.join(data_dir, e["file"]), mmap_mode="r")
        out[pos:pos + e["length"]] = opened[e["file"]][e["start"]:e["start"] + e["length"]]
        pos += e["length"]
    out.flush()
    del out
    os.replace(tmp, path)
    a = np.load(path, mmap_mode="r")
    total, sq, chunk = np.zeros(45), np.zeros(45), 500_000
    for i in range(0, n, chunk):
        x = np.asarray(a[i:i + chunk, :45], dtype=np.float64)
        total += x.sum(0)
        sq += (x * x).sum(0)
    mean = total / n
    var = sq / n - mean * mean
    assert np.all(var > 0), f"rung {name}: a state column has no spread"
    stats = {"rows": int(n), "episodes": len(episodes), "state_mean": mean.tolist(),
             "state_std": np.sqrt(var).tolist()}
    with open(os.path.join(data_dir, f"rung_{name}.stats.json"), "w") as fh:
        json.dump(stats, fh)
    np.save(os.path.join(data_dir, f"rung_{name}.ep.npy"), np.asarray(a[:, 66]).astype(np.int64))
    np.save(os.path.join(data_dir, f"rung_{name}.fall.npy"), np.asarray(a[:, 65]) > 0.5)
    return stats


def finish(args):
    table = sources()
    plan = make_plan(args, table)
    missing = [p for p in plan if not os.path.exists(shard_path(args.data_dir, p["kind"], p.get("source"), p.get("batch")))]
    if missing:
        raise SystemExit(f"{len(missing)} shards are missing, first {missing[0]}")
    eps, shards, per_source = [], {}, {s["id"]: {"episodes": 0, "transitions": 0, "falls": 0, "thigh_rows": 0}
                                       for s in table}
    sets = {}
    for i, p in enumerate(plan):
        path = shard_path(args.data_dir, p["kind"], p.get("source"), p.get("batch"))
        name = os.path.basename(path)
        tab, n = episode_table(path)
        shards[name] = {"rows": n, "episodes": len(tab), "sha256": sha256(path)}
        rec = [{"id": t[0], "source": t[1], "file": name, "start": t[2], "length": t[3], "fell": t[4]} for t in tab]
        if p["kind"] == "train":
            eps += rec
            for t in tab:
                c = per_source[t[1]]
                c["episodes"] += 1
                c["transitions"] += t[3]
                c["falls"] += int(t[4])
                c["thigh_rows"] += t[5]
        else:
            sets[p["kind"]] = rec
        if not args.no_progress and i % 20 == 0:
            progress("manifest: shard scan", i, len(plan))
    seqs = group_sequences(eps, table)
    sizes = {k: v for k, v in args.rung_sizes.items()}
    rungs = rung_lists(seqs, table, sizes)
    stats = {}
    jobs = list(rungs.items()) + ([("d6", {"episodes": sets["d6"]})] if "d6" in sets else [])
    for j, (name, r) in enumerate(jobs):
        stats[name] = write_rung_file(args.data_dir, name, r["episodes"])
        if not args.no_progress:
            progress("rung files", j + 1, len(jobs))
    code = sha256(os.path.abspath(__file__))
    man = {
        "what": "Option C data: hand-built calibrated MuJoCo ANYmal D with a fitted PD motor (not Isaac Lab, not hardware)",
        "generator": os.path.relpath(os.path.abspath(__file__), AIR), "generator_sha256": code,
        "seeds": {"root": ROOT, "rungs": RUNG_SEED, "episode": "default_rng([root, set tag, source, batch, episode])",
                  "set_tags": SET_TAGS},
        "settings": {"steps": STEPS, "cmd_steps": CMD_STEPS, "push_steps": PUSH_STEPS, "push_max": PUSH_MAX,
                     "friction": FRICTION, "payload": PAYLOAD, "gain": GAIN, "smooth_tau": SMOOTH_TAU,
                     "smooth_std": SMOOTH_STD, "train_total": args.train_total},
        "columns": {"state": [0, 45], "action": [45, 57], "contacts": [57, 65], "fall": 65, "episode_id": 66,
                    "source_id": 67},
        "sources": table, "per_source": {str(k): v for k, v in per_source.items()},
        "actors_sha256": {n: sha256(os.path.join(RUNS_S2, n, "actor.pt")) for n in trained_policies()},
        "shards": shards,
        "rungs": {k: {"episodes": [e["id"] for e in v["episodes"]], "by_group": v["by_group"],
                      "rows": stats[k]["rows"]} for k, v in rungs.items()},
        "rung_stats": stats,
        "held_out": {k: {"episodes": [e["id"] for e in v], "rows": sum(e["length"] for e in v),
                         "falls": sum(int(e["fell"]) for e in v)} for k, v in sets.items()},
    }
    tmp = args.manifest + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(man, fh)
    os.replace(tmp, args.manifest)
    if not args.no_progress:
        progress("manifest written", 1, 1)
    return man


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=None)
    ap.add_argument("--nworkers", type=int, default=2)
    ap.add_argument("--finish", action="store_true")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--train-total", type=int, default=TRAIN_TOTAL, help="below 6M for tests only")
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--n-mixed", type=int, default=40)
    ap.add_argument("--n-expert", type=int, default=20)
    ap.add_argument("--n-d6", type=int, default=8)
    ap.add_argument("--time-limit", type=float, default=None, help="stop starting shards after this many seconds")
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--rungs", default=None, help="name=size,... (tests only)")
    a = ap.parse_args(argv)
    a.data_dir = os.path.abspath(a.data_dir)
    a.rung_sizes = RUNGS if a.rungs is None else {k: int(v) for k, v in (x.split("=") for x in a.rungs.split(","))}
    if a.finish:
        man = finish(a)
        print(json.dumps({"rungs": {k: v["rows"] for k, v in man["rungs"].items()},
                          "held_out": {k: v["rows"] for k, v in man["held_out"].items()}}))
    elif a.worker is not None:
        print(json.dumps(worker(a)))
    else:
        ap.error("--worker or --finish")


if __name__ == "__main__":
    main()

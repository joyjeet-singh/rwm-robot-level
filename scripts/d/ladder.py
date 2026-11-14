"""Option D (D1): the ladder job's driver (OPTIOND_PLAN.md, Stage 2).  ./py

    ./py scripts/d/ladder.py run                    # the ladder, two workers (resumable; skips finished work)
    ./py scripts/d/ladder.py fill                   # checks and walk test for any tested simulator lacking them
    ./py scripts/d/ladder.py final                  # results/d_ladder.json and the final table in results/d_ladder.txt
    ./py scripts/d/ladder.py run --toy spec.json --state-dir DIR     # stand-ins for fine-tuning, training, measuring
    ./py scripts/d/ladder.py run --rehearsal --state-dir DIR         # real code at toy size (`_test_dreh`)

The design (fixed; never changed here):
  * verdict rule (`verdict_rule`): seeds 1 and 2; a third (seed 3) only when exactly one of the two failed; the
    simulator failed when at least two of its policies failed. A policy failed when its walking score is below 0.15,
    or it was stopped by the stop rule, or its training failed by its simulator (`policy_failed`);
  * search rule (`search_rule`): 80 steps; then 20 if it failed, else 320; then 5 / 40 / 160 / 1,280 by the table;
    three tests per run. A run is fine-tuned only as far as its next test needs;
  * the queue: two workers; a free worker claims the next run in the design's order whose start condition can be
    decided (`W_s1` only if `W_s0` found a boundary, `M_s1` only if `M_s0` did); a run that does not start is recorded
    with its reason; no run starts after 56 hours of job time (the driver's running time, summed over resumes); a
    started run finishes its tests;
  * a run's loop: fine-tune to the next step count (one fine-tuning at a time: a lock); checks (primary) and the walk
    test (secondary) before the first policy; policies as the verdict rule asks, each measured with evaluate.py as
    soon as it is trained; the verdict; repeat. A loss that is not finite (or a save whose weights are not finite)
    ends the run; its tested simulators stay. Checks never decide what is tested;
  * the crash guard: a training that crashes inside the lite code is run again; if it crashes the same way twice, the
    same code runs on REL for 3 iterations (`_test_guard_<policy>`); if that crashes too the fault is the code's and
    the job stops with an error (nothing is recorded as failed); otherwise the policy failed by its simulator;
  * sanity gate: if neither `W_s0` nor `M_s0` can produce a simulator at 80 steps with finite weights, the job writes
    optd/DECISIONS.md and optd/LOG.md and exits 3. Any other failure exits 1.
State: one file per run in results/d/ladder/<run>.json (and job.json, the job clock), rewritten after every step, so
the job resumes where it stopped (an interrupted policy restarts from iteration 0). One line per measured policy is
appended to results/d_ladder.txt.
"""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
AIR = os.path.dirname(SCRIPTS)
sys.path.insert(0, HERE)

ORDER = ("W_s0", "M_s0", "Wtf_s0", "W_s1", "M_s1", "Wg_s0", "Mg_s0")
START_IF = {"W_s1": "W_s0", "M_s1": "M_s0"}
GATE_RUNS = ("W_s0", "M_s0")
FIRST = 80
SAVES = (5, 20, 40, 80, 160, 320, 1280)
LIMIT_HOURS = 56.0
FAIL_SCORE = 0.15
STATE_DIR = os.path.join(AIR, "results", "d", "ladder")
TEXT = os.path.join(AIR, "results", "d_ladder.txt")
FINAL_JSON = os.path.join(AIR, "results", "d_ladder.json")
LOGS = os.path.join(AIR, ".cache", "logs")
FAILED, NOT_FAILED = "failed", "did not fail"
TERMINAL = ("done", "ended", "skipped")


class GateBlocked(Exception):
    pass


class JobError(Exception):
    pass


# ------------------------------------------------------------------ the design's rules (pure)
def policy_failed(p):
    """A policy failed: walking score below 0.15, or stopped by the stop rule, or failed by its simulator."""
    if p["status"] in ("stopped", "failed_by_simulator"):
        return True
    if p["status"] != "finished":
        raise ValueError(f"unknown policy status {p['status']!r}")
    return float(p["score"]) < FAIL_SCORE


def verdict_rule(failed):
    """From the measured policies so far (failed or not, in seed order 1, 2, 3): ("train", next seed) or
    ("verdict", "failed" | "did not fail")."""
    failed = [bool(f) for f in failed]
    n, k = len(failed), sum(failed)
    if n == 0:
        return ("train", 1)
    if n == 1:
        return ("train", 2)
    if n == 2:
        if k == 1:
            return ("train", 3)
        return ("verdict", FAILED if k == 2 else NOT_FAILED)
    if n == 3:
        if failed[0] == failed[1]:
            raise ValueError("a third policy is trained only after a split")
        return ("verdict", FAILED if k >= 2 else NOT_FAILED)
    raise ValueError(f"{n} policies: at most three")


def search_rule(tests):
    """From a run's tests so far [(steps, failed), ...] in order: the next step count, or "done"."""
    t = [(int(s), bool(f)) for s, f in tests]
    if not t:
        return FIRST
    if t[0][0] != FIRST:
        raise ValueError("the first test is at 80 steps")
    if len(t) == 1:
        return 20 if t[0][1] else 320
    second = 20 if t[0][1] else 320
    if t[1][0] != second:
        raise ValueError(f"the second test must be at {second} steps")
    if len(t) == 2:
        return {(True, True): 5, (True, False): 40, (False, True): 160, (False, False): 1280}[(t[0][1], t[1][1])]
    if len(t) == 3:
        return "done"
    raise ValueError("three tests per run")


def found_boundary(tests):
    v = [t["verdict"] for t in tests if t.get("verdict")]
    return FAILED in v and NOT_FAILED in v


# ------------------------------------------------------------------ backends
class Real:
    """The real code: scripts/d/finetune.py, checks.py, walk_test.py, train_policy.py (./py) and evaluate.py (./py-mj).
    `rehearsal` = a dict of toy sizes (run `_test_dreh`): everything small, written under .cache/ or `_test*` names."""

    def __init__(self, rehearsal=None, job="ladder"):
        self.reh = rehearsal
        self.job = job
        self.procs = set()
        self.plock = threading.Lock()
        os.makedirs(LOGS, exist_ok=True)

    def hour_seconds(self):
        return 3600.0

    def poll(self):
        return 2.0 if self.reh else 30.0

    def _cmd(self, args, log):
        with open(os.path.join(LOGS, log), "a") as fh:
            fh.write(f"\n=== {time.strftime('%F %T')} {' '.join(args)}\n")
            fh.flush()
            p = subprocess.Popen(args, cwd=AIR, stdout=fh, stderr=subprocess.STDOUT)
            with self.plock:
                self.procs.add(p)
            try:
                rc = p.wait()
            finally:
                with self.plock:
                    self.procs.discard(p)
        return rc

    def kill_all(self):
        with self.plock:
            for p in list(self.procs):
                try:
                    p.terminate()
                except OSError:
                    pass

    def finetune(self, run, to):
        import finetune as FT
        if self.reh:
            code = (f"import sys; sys.path.insert(0, 'scripts/d'); import finetune as FT; "
                    f"FT.advance({run!r}, {int(to)}, test_spec={self.reh['ft_spec']!r}, saves={tuple(self.reh['saves'])!r}, "
                    f"job={self.job!r})")
            rc = self._cmd(["./py", "-c", code], "ladder_finetune.log")
        else:
            rc = self._cmd(["./py", "scripts/d/finetune.py", "--run", run, "--to", str(to), "--job", self.job,
                            "--threads", "2"], "ladder_finetune.log")
        if rc != 0:
            raise JobError(f"fine-tuning {run} to {to} exited {rc} (.cache/logs/ladder_finetune.log)")
        return FT.status(run)

    def sim_ready(self, sim):
        import torch
        path = os.path.join(AIR, "runs", "d", sim, "final.pt")
        if not os.path.exists(path):
            return False, "not saved"
        ck = torch.load(path, map_location="cpu", weights_only=False)
        bad = [k for k, v in ck["model_state_dict"].items() if torch.is_tensor(v) and v.is_floating_point()
               and not bool(torch.isfinite(v).all())]
        return (not bad), (f"weights not finite: {bad[:3]}" if bad else "ok")

    def checks(self, sim):
        args = ["./py", "scripts/d/checks.py", "--sim", sim, "--job", self.job, "--threads", "2"]
        if self.reh:
            args += ["--out", os.path.join(self.reh["dir"], "checks", sim + ".json"), "--max-windows",
                     str(self.reh["windows"])]
        return self._cmd(args, "ladder_checks.log") == 0

    def walk(self, sim):
        args = ["./py", "scripts/d/walk_test.py", "--sim", sim, "--job", self.job, "--threads", "2"]
        if self.reh:
            args += ["--out", os.path.join(self.reh["dir"], "walk", sim + ".json"), "--robots",
                     str(self.reh["robots"]), "--steps", str(self.reh["walk_steps"])]
        return self._cmd(args, "ladder_walk.log") == 0

    def _size(self):
        return (["--iterations", str(self.reh["iterations"]), "--num_envs", str(self.reh["num_envs"])]
                if self.reh else [])

    def train(self, sim, seed, attempt):
        rc = self._cmd(["./py", "scripts/d/train_policy.py", "--sim", sim, "--seed", str(seed), "--attempt",
                        str(attempt), "--job", self.job] + self._size(), "ladder_train.log")
        return {0: "ok", 4: "crash"}.get(rc, f"error (exit {rc})")

    def train_record(self, name):
        import train_policy as TPD
        return TPD.done_record(name)

    def _crashes(self, name):
        import train_policy as TPD
        d = TPD.paths(name)["dir"]
        out = []
        for a in (1, 2):
            f = os.path.join(d, f"crash_{a}.json")
            if os.path.exists(f):
                with open(f) as fh:
                    out.append(json.load(fh))
        return out

    def crash_count(self, name):
        return len(self._crashes(name))

    def same_crash(self, name):
        c = self._crashes(name)
        return len(c) == 2 and c[0]["type"] == c[1]["type"]

    def guard(self, name, seed):
        import shutil
        import train_policy as TPD
        g = "_test_guard_" + name
        size = ["--num_envs", str(self.reh["num_envs"])] if self.reh else ["--num_envs", "4096"]
        rc = self._cmd(["./py", "scripts/d/train_policy.py", "--sim", "REL", "--seed", str(seed), "--name", g,
                        "--iterations", "3", "--job", self.job] + size, "ladder_guard.log")
        p = TPD.paths(g)
        for q in (p["dir"], p["lite"]):
            shutil.rmtree(q, ignore_errors=True)
        if os.path.exists(p["res"]):
            os.remove(p["res"])
        return {"ok": rc == 0, "exit": rc, "name": g, "iterations": 3, "sim": "REL"}

    def mark_failed(self, sim, seed, guard):
        rc = self._cmd(["./py", "scripts/d/train_policy.py", "--sim", sim, "--seed", str(seed), "--mark-failed",
                        "--guard", json.dumps(guard), "--job", self.job] + self._size(), "ladder_train.log")
        if rc != 0:
            raise JobError(f"--mark-failed {sim} seed {seed} exited {rc}")
        import train_policy as TPD
        return TPD.done_record(TPD.policy_name(sim, seed))

    def measure(self, name):
        import train_policy as TPD
        actor = TPD.paths(name)["actor"]
        args = ["./py-mj", "scripts/d/evaluate.py", "--policy", name, "--actor", actor, "--job", self.job]
        if self.reh:
            out = os.path.join(self.reh["dir"], "eval", name + ".json")
            args += ["--schedules", "0", "--starts", "1", "--no-recorded", "--out", out]
        else:
            out = os.path.join(AIR, "results", "d", "eval", name + ".json")
        rc = self._cmd(args, "ladder_eval.log")
        if rc != 0 or not os.path.exists(out):
            raise JobError(f"measuring {name} exited {rc} (.cache/logs/ladder_eval.log)")
        with open(out) as fh:
            r = json.load(fh)
        return {"score": r["score"], "fall_rate": r["fall_rate"], "physics_return": r.get("physics_return"),
                "eval": os.path.relpath(out, AIR)}


class Toy:
    """Stand-ins for fine-tuning, training and measuring (tests). The spec (JSON):
      fail_from {run: steps}  a simulator of the run fails (all its policies fail) from that step count on; absent or
                              null: never fails
      scores {policy: score}  explicit walking scores (override the above for that policy)
      stopped [policy]        stopped by the stop rule;  nonfinite_policy [policy]: a loss that is not finite
      nonfinite {run: step}   the run's loss is not finite at that step
      crashes {policy: n}     the first n training attempts crash inside the lite code
      different_crash [policy]  its two crashes differ;  guard_crash: true  the crash guard crashes on REL too
      checks_fail [sim]       the checks cannot be computed (non-blocking)
      hours {policy, measure, finetune_step, checks}   job hours each stand-in takes;  hour_seconds: real seconds per
                              job hour."""

    def __init__(self, spec, state_dir):
        self.s = spec
        self.path = os.path.join(state_dir, "toy_world.json")
        self.lock = threading.Lock()
        h = {"policy": 2.05, "measure": 0.07, "finetune_step": 5.0 / 3600, "checks": 0.05}
        h.update(spec.get("hours", {}))
        self.h = h
        self.procs = set()

    def hour_seconds(self):
        return float(self.s.get("hour_seconds", 0.002))

    def poll(self):
        return self.hour_seconds()

    def kill_all(self):
        pass

    def _sleep(self, hours):
        time.sleep(hours * self.hour_seconds())

    def _world(self, fn=None):
        with self.lock:
            w = json.load(open(self.path)) if os.path.exists(self.path) else {"step": {}, "saved": [], "records": {},
                                                                                  "crash": {}, "calls": []}
            out = fn(w) if fn else None
            if fn:
                with open(self.path + ".tmp", "w") as fh:
                    json.dump(w, fh)
                os.replace(self.path + ".tmp", self.path)
            return w if fn is None else out

    def _sim_fails(self, sim):
        run, steps = sim.rsplit("_d", 1)
        f = self.s.get("fail_from", {}).get(run)
        return f is not None and int(steps) >= int(f)

    def finetune(self, run, to):
        bad = self.s.get("nonfinite", {}).get(run)

        def go(w):
            st = w["step"].get(run, 0)
            target = to if bad is None or to < bad else bad - 1
            for s in SAVES:
                if st < s <= target:
                    w["saved"].append(f"{run}_d{s}")
            w["step"][run] = max(st, target)
            w["calls"].append(["finetune", run, to])
            return {"run": run, "step": w["step"][run], "nonfinite_at": bad if (bad is not None and to >= bad)
                    else None, "_moved": max(0, target - st)}
        r = self._world(go)
        self._sleep(r.pop("_moved") * self.h["finetune_step"])
        return r

    def sim_ready(self, sim):
        return (sim in self._world()["saved"]), "toy"

    def checks(self, sim):
        self._sleep(self.h["checks"])
        return sim not in self.s.get("checks_fail", [])

    def walk(self, sim):
        return True

    def train(self, sim, seed, attempt):
        name = f"{sim}__p{seed}"
        self._sleep(self.h["policy"])
        if self.s.get("crashes", {}).get(name, 0) >= attempt:
            def c(w):
                w["crash"][name] = w["crash"].get(name, 0) + 1
            self._world(c)
            return "crash"
        status = ("stopped" if name in self.s.get("stopped", []) else
                  "failed_by_simulator" if name in self.s.get("nonfinite_policy", []) else "finished")

        def rec(w):
            w["records"][name] = {"policy": name, "status": status, "measure": status != "failed_by_simulator"}
            w["calls"].append(["train", name])
        self._world(rec)
        return "ok"

    def train_record(self, name):
        return self._world()["records"].get(name)

    def crash_count(self, name):
        return self._world()["crash"].get(name, 0)

    def same_crash(self, name):
        return name not in self.s.get("different_crash", [])

    def guard(self, name, seed):
        def c(w):
            w["calls"].append(["guard", name])
        self._world(c)
        return {"ok": not self.s.get("guard_crash", False), "name": "_test_guard_" + name}

    def mark_failed(self, sim, seed, guard):
        name = f"{sim}__p{seed}"

        def rec(w):
            w["records"][name] = {"policy": name, "status": "failed_by_simulator", "measure": False, "guard": guard}
        self._world(rec)
        return self.train_record(name)

    def measure(self, name):
        self._sleep(self.h["measure"])
        sim = name.rsplit("__p", 1)[0]
        sc = self.s.get("scores", {}).get(name, 0.06 if self._sim_fails(sim) else 0.6)
        return {"score": float(sc), "fall_rate": 0.9 if sc < FAIL_SCORE else 0.3, "physics_return": 10 * sc,
                "eval": "toy"}


# ------------------------------------------------------------------ the driver
class Driver:
    def __init__(self, backend, state_dir=STATE_DIR, text=TEXT, runs=ORDER, limit_hours=LIMIT_HOURS, workers=2,
                 stop_after=None, steps_plan=None, gate_files=True):
        self.b = backend
        self.dir = state_dir
        self.text = text
        self.runs = tuple(runs)
        self.limit = float(limit_hours)
        self.workers = workers
        self.stop_after = stop_after
        self.steps_plan = steps_plan              # rehearsal: fixed step counts instead of the search
        self.gate_files = gate_files
        self.lock = threading.RLock()
        self.ft_lock = threading.Lock()
        self.active = set()
        self.abort = threading.Event()
        self.error = None
        self.recorded = 0
        os.makedirs(self.dir, exist_ok=True)
        self.state = {r: self._load(r) for r in self.runs}
        job = self._read(os.path.join(self.dir, "job.json")) or {"elapsed_hours": 0.0, "invocations": 0}
        job["invocations"] = int(job.get("invocations", 0)) + 1
        self.base_hours = float(job["elapsed_hours"])
        self.t0 = time.time()
        self.job = job
        self._save_job()

    # -- files
    @staticmethod
    def _read(path):
        if not os.path.exists(path):
            return None
        with open(path) as fh:
            return json.load(fh)

    def _load(self, run):
        return self._read(os.path.join(self.dir, f"{run}.json")) or {"run": run, "status": "waiting", "tests": []}

    def _atomic(self, obj, path):
        tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
        with open(tmp, "w") as fh:
            json.dump(obj, fh, indent=1, default=float)
        os.replace(tmp, path)

    def elapsed(self):
        return self.base_hours + (time.time() - self.t0) / self.b.hour_seconds()

    def _save_job(self):
        self.job["elapsed_hours"] = self.elapsed()
        self.job["limit_hours"] = self.limit
        self.job["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._atomic(self.job, os.path.join(self.dir, "job.json"))

    def save(self, run):
        with self.lock:
            self._atomic(self.state[run], os.path.join(self.dir, f"{run}.json"))
            self._save_job()

    def line(self, run, t, p):
        with self.lock:
            new = not os.path.exists(self.text) or os.path.getsize(self.text) == 0
            with open(self.text, "a") as fh:
                if new:
                    fh.write("Option D ladder: one line per measured policy, as measured (walking score in the "
                             "calibrated hand-built MuJoCo ANYmal D; fails below 0.15)\n")
                sc = "-" if p["score"] is None else f"{p['score']:.3f}"
                fr = "-" if p["fall_rate"] is None else f"{p['fall_rate']:.2f}"
                fh.write(f"{time.strftime('%F %T')}  {run:7s} {t['sim']:14s} {p['policy']:18s} {p['status']:19s} "
                         f"score {sc}  falls {fr}  {'FAILED' if p['failed'] else 'walks'}\n")

    def progress(self, step):
        try:
            import dprogress
            dprogress.write("ladder", step)
        except Exception:
            pass

    # -- the queue
    def claim(self):
        """The next run of the design's order whose start condition can be decided (None: none now). A started run
        (status running, not held by a worker) is resumed first; it finishes whatever the clock says."""
        with self.lock:
            h = self.elapsed()
            for r in self.runs:
                st = self.state[r]
                if r in self.active or st["status"] in TERMINAL:
                    continue
                if st["status"] == "running":
                    self.active.add(r)
                    return r
            for r in self.runs:
                st = self.state[r]
                if r in self.active or st["status"] != "waiting":
                    continue
                if h >= self.limit:
                    self.skip(r, f"no run starts after {self.limit:g} hours of job time (at {h:.1f} h)")
                    continue
                parent = START_IF.get(r)
                if parent is not None and parent in self.runs:
                    pst = self.state[parent]
                    if pst["status"] == "skipped":
                        self.skip(r, f"{parent} did not run")
                        continue
                    if pst["status"] not in ("done", "ended"):
                        continue                               # cannot be decided yet
                    if not found_boundary(pst["tests"]):
                        self.skip(r, f"{parent} found no boundary")
                        continue
                st.update(status="running", started_hours=h, started=time.strftime("%Y-%m-%dT%H:%M:%S"))
                self.active.add(r)
                self.save(r)
                return r
            return None

    def skip(self, r, why):
        self.state[r].update(status="skipped", reason=why)
        self.save(r)

    def unsettled(self):
        with self.lock:
            return [r for r in self.runs if self.state[r]["status"] not in TERMINAL]

    def worker(self, wid):
        try:
            while not self.abort.is_set():
                r = self.claim()
                if r is None:
                    if not self.unsettled():
                        return
                    time.sleep(self.b.poll())
                    continue
                try:
                    self.do_run(r)
                finally:
                    with self.lock:
                        self.active.discard(r)
        except BaseException as e:                     # noqa: BLE001  (stop both workers, report)
            self.error = self.error or (type(e).__name__, e, traceback.format_exc())
            self.abort.set()
            self.b.kill_all()

    def heartbeat(self):
        while not self.abort.wait(min(60.0, 30 * self.b.hour_seconds())):
            with self.lock:
                self._save_job()

    def go(self):
        hb = threading.Thread(target=self.heartbeat, daemon=True)
        hb.start()
        ws = [threading.Thread(target=self.worker, args=(i,)) for i in range(self.workers)]
        for w in ws:
            w.start()
        for w in ws:
            w.join()
        self.abort.set()
        with self.lock:
            self._save_job()
        if self.error:
            kind = self.error[0]
            if kind == "GateBlocked":
                self.gate_note(str(self.error[1]))
                return 3
            print(f"[ladder] ERROR {kind}: {self.error[1]}\n{self.error[2] if len(self.error) > 2 else ''}")
            return 1
        return 0

    # -- a run
    def next_steps(self, st):
        if self.steps_plan is not None:
            k = len(st["tests"])
            return self.steps_plan[k] if k < len(self.steps_plan) else "done"
        return search_rule([(t["steps"], t["verdict"] == FAILED) for t in st["tests"]])

    def finish(self, r, status, **kw):
        with self.lock:
            self.state[r].update(status=status, finished_hours=self.elapsed(), **kw)
            self.save(r)

    def gate_check(self):
        """Sanity gate: neither W_s0 nor M_s0 can produce a simulator at 80 steps with finite weights."""
        with self.lock:
            if not all(g in self.runs for g in GATE_RUNS):
                return
            sts = [self.state[g] for g in GATE_RUNS]
            if all(s["status"] == "ended" and not any(t["steps"] == FIRST for t in s["tests"]) for s in sts):
                raise GateBlocked("neither W_s0 nor M_s0 produced a simulator at 80 steps with finite weights: "
                                  + "; ".join(f"{s['run']}: {s.get('ended', {}).get('why')}" for s in sts))

    def do_run(self, r):
        while True:
            if self.abort.is_set():
                return
            st = self.state[r]
            tests = st["tests"]
            if tests and tests[-1].get("verdict") is None:
                t = tests[-1]
            else:
                nxt = self.next_steps(st)
                if nxt == "done":
                    self.finish(r, "done", boundary=found_boundary(tests))
                    return
                self.progress(f"fine-tune {r} to {nxt}")
                with self.ft_lock:
                    fs = self.b.finetune(r, nxt)
                sim = f"{r}_d{nxt}"
                ready, why = self.b.sim_ready(sim)
                if not ready:
                    why = (f"loss not finite at step {fs.get('nonfinite_at')}" if fs.get("nonfinite_at") is not None
                           else why)
                    self.finish(r, "ended", ended={"why": why, "wanted_steps": nxt, "at_step": fs.get("step")},
                                boundary=found_boundary(tests))
                    self.gate_check()
                    return
                t = {"steps": nxt, "sim": sim, "checks": None, "walk": None, "policies": [], "verdict": None}
                with self.lock:
                    tests.append(t)
                    self.save(r)
            self.do_test(r, t)

    def do_test(self, r, t):
        sim = t["sim"]
        if t["checks"] is None:
            self.progress(f"checks {sim}")
            t["checks"] = "ok" if self.b.checks(sim) else "failed (non-blocking; retried by the job's last step)"
            self.save(r)
        if t["walk"] is None:
            self.progress(f"walk test {sim}")
            t["walk"] = "ok" if self.b.walk(sim) else "failed (secondary; retried by the job's last step)"
            self.save(r)
        while True:
            if self.abort.is_set():
                return
            act, val = verdict_rule([p["failed"] for p in t["policies"]])
            if act == "verdict":
                with self.lock:
                    t["verdict"] = val
                    t["third_policy"] = len(t["policies"]) == 3
                    self.save(r)
                return
            p = self.policy(sim, val)
            with self.lock:
                t["policies"].append(p)
                self.save(r)
                self.line(r, t, p)
                self.recorded += 1
                if self.stop_after is not None and self.recorded >= self.stop_after:
                    os._exit(9)                        # toy and rehearsal: stands for the job being killed

    def policy(self, sim, seed):
        name = f"{sim}__p{seed}"
        rec = self.b.train_record(name)
        crashes = self.b.crash_count(name)
        guard = None
        while rec is None:
            if crashes >= 2:
                if not self.b.same_crash(name):
                    raise JobError(f"{name}: two different crashes inside the lite code; not a result")
                self.progress(f"crash guard for {name}")
                guard = self.b.guard(name, seed)
                if not guard["ok"]:
                    raise JobError(f"{name}: the crash guard crashed on REL too; the fault is the code's "
                                   f"(nothing recorded as failed)")
                rec = self.b.mark_failed(sim, seed, guard)
                break
            self.progress(f"train {name}")
            out = self.b.train(sim, seed, crashes + 1)
            if out == "ok":
                rec = self.b.train_record(name)
                if rec is None:
                    raise JobError(f"{name}: training ended without a record")
            elif out == "crash":
                crashes += 1
            else:
                raise JobError(f"{name}: training {out} (not a crash inside the lite code)")
        p = {"policy": name, "seed": seed, "status": rec["status"], "crashes": crashes,
             "stopped_at": rec.get("stopped_at")}
        if rec.get("measure"):
            self.progress(f"measure {name}")
            p.update(self.b.measure(name))
        else:
            p.update(score=0.0, fall_rate=1.0, physics_return=None, eval=None,
                     note="failed by its simulator: walking score 0, fall rate 1 (Step 2's rule), not measured")
        p["failed"] = policy_failed(p)
        return p

    def gate_note(self, why):
        if not self.gate_files:
            print(f"[ladder] GATE: {why}")
            return
        stamp = time.strftime("%F %H:%M")
        with open(os.path.join(AIR, "optd", "DECISIONS.md"), "a") as fh:
            fh.write(f"\n## Ladder job: BLOCKED at the sanity gate ({stamp})\n- {why}\n- Options: stop Option D here "
                     f"and report it; or a new plan for the fine-tuning (not this plan's choice). Recommendation: "
                     f"report it as it is.\n")
        with open(os.path.join(AIR, "optd", "LOG.md"), "a") as fh:
            fh.write(f"\n## Ladder job status: BLOCKED ({stamp})\nSanity gate: {why}\n")


# ------------------------------------------------------------------ after the driver: fill and final
def tested(state_dir=STATE_DIR):
    out = []
    for r in ORDER:
        p = os.path.join(state_dir, f"{r}.json")
        if os.path.exists(p):
            with open(p) as fh:
                st = json.load(fh)
            out += [(r, t) for t in st["tests"]]
    return out


def fill(state_dir=STATE_DIR):
    """Checks (primary) and the walk test (secondary) of any tested simulator that lacks them (Real backend)."""
    b = Real()
    missing = []
    for r, t in tested(state_dir):
        s = t["sim"]
        for kind, fn, d in (("checks", b.checks, "checks"), ("walk", b.walk, "walk")):
            if not os.path.exists(os.path.join(AIR, "results", "d", d, s + ".json")):
                if not fn(s):
                    missing.append(f"{kind} {s}")
    print(json.dumps({"missing_after_fill": missing}))
    return 0


def final(state_dir=STATE_DIR, out_json=FINAL_JSON, text=TEXT):
    runs = {}
    for r in ORDER:
        p = os.path.join(state_dir, f"{r}.json")
        runs[r] = json.load(open(p)) if os.path.exists(p) else {"run": r, "status": "not reached"}
    job = json.load(open(os.path.join(state_dir, "job.json"))) if os.path.exists(os.path.join(state_dir, "job.json")) \
        else {}
    sims = []
    for r, t in tested(state_dir):
        sc = [p["score"] for p in t["policies"]]
        sims.append({"sim": t["sim"], "run": r, "steps": t["steps"], "verdict": t["verdict"],
                     "policies": t["policies"], "mean_score": sum(sc) / len(sc) if sc else None,
                     "third_policy": t.get("third_policy"),
                     "checks": os.path.exists(os.path.join(AIR, "results", "d", "checks", t["sim"] + ".json")),
                     "walk": os.path.exists(os.path.join(AIR, "results", "d", "walk", t["sim"] + ".json"))})
    rec = {"what": "Option D ladder (Stage 2): every run, test and policy, from the driver's state files",
           "fail_score": FAIL_SCORE, "limit_hours": LIMIT_HOURS, "job": job, "runs": runs, "tested": sims,
           "n_tested": len(sims), "n_third_policy": sum(bool(s["third_policy"]) for s in sims),
           "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if os.path.exists(out_json):
        raise SystemExit(f"{out_json} exists; not overwriting")
    with open(out_json, "w") as fh:
        json.dump(rec, fh, indent=1)
    lines = ["", f"=== Final ladder table ({rec['created']}); job time {job.get('elapsed_hours', 0):.1f} h ===",
             "run      status   tests (steps: verdict, mean score, policies)"]
    for r in ORDER:
        st = runs[r]
        why = st.get("reason") or (st.get("ended") or {}).get("why") or ""
        lines.append(f"{r:8s} {st['status']:8s} {why}")
        for t in st.get("tests", []):
            sc = [p["score"] for p in t["policies"]]
            ms = f"{sum(sc) / len(sc):.3f}" if sc else "-"
            ps = ", ".join(f"p{p['seed']} {p['score']:.3f}{'*' if p['failed'] else ''}" for p in t["policies"])
            lines.append(f"         {t['steps']:5d}: {t['verdict'] or 'unfinished':12s} mean {ms}  ({ps})")
    lines.append(f"tested simulators {rec['n_tested']}; tests that needed a third policy {rec['n_third_policy']}; "
                 "* = failed policy")
    with open(text, "a") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[-40:]))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "fill", "final"))
    ap.add_argument("--state-dir", default=STATE_DIR)
    ap.add_argument("--text", default=None)
    ap.add_argument("--out", default=FINAL_JSON)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--toy", default=None, help="toy spec (JSON file); stand-ins for the heavy work")
    ap.add_argument("--rehearsal", action="store_true", help="real code at toy size, run `_test_dreh`")
    ap.add_argument("--runs", default=None, help="comma list (toy only)")
    ap.add_argument("--limit-hours", type=float, default=LIMIT_HOURS)
    ap.add_argument("--stop-after", type=int, default=None, help="toy and rehearsal: stop after n policies")
    a = ap.parse_args(argv)
    special = a.toy or a.rehearsal
    if a.cmd == "run" and not special and (a.runs or a.stop_after or a.limit_hours != LIMIT_HOURS or a.state_dir != STATE_DIR
                        or a.workers != 2):
        raise SystemExit("--runs, --stop-after, --limit-hours, --state-dir and --workers are for toy and rehearsal")
    if a.cmd == "run" and special and a.state_dir == STATE_DIR:
        raise SystemExit("toy and rehearsal need their own --state-dir")
    text = a.text or (os.path.join(a.state_dir, "d_ladder.txt") if a.state_dir != STATE_DIR else TEXT)
    if a.cmd == "fill":
        return fill(a.state_dir)
    if a.cmd == "final":
        return final(a.state_dir, a.out, text)
    if a.toy:
        with open(a.toy) as fh:
            spec = json.load(fh)
        b = Toy(spec, a.state_dir)
        runs = a.runs.split(",") if a.runs else ORDER
        d = Driver(b, a.state_dir, text, runs, a.limit_hours, a.workers, a.stop_after, gate_files=False)
    elif a.rehearsal:
        reh = {"dir": a.state_dir, "ft_spec": {"data": "walking", "lr": 1e-4, "N": 8, "seed": 0, "rung": "1M"},
               "saves": (2,), "windows": 4, "robots": 64, "walk_steps": 16, "iterations": 2, "num_envs": 64}
        b = Real(rehearsal=reh, job="rehearsal")
        d = Driver(b, a.state_dir, text, ("_test_dreh",), a.limit_hours, 1, a.stop_after, steps_plan=(2,),
                   gate_files=False)
    else:
        d = Driver(Real(), STATE_DIR, TEXT, ORDER, LIMIT_HOURS, 2)
    rc = d.go()
    print(json.dumps({"exit": rc, "elapsed_hours": round(d.elapsed(), 3),
                      "runs": {r: d.state[r]["status"] for r in d.runs}}))
    return rc


if __name__ == "__main__":
    sys.exit(main())

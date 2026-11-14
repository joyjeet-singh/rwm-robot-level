"""Option D (D0): one fine-tuning run of the released model on MuJoCo data.  ./py

    ./py scripts/d/finetune.py --run W_s0 --to 80        # advance run W_s0 to 80 steps; saves runs/d/W_s0_d{5,20,40,80}
    ./py scripts/d/finetune.py --run W_s0 --status       # print the run's state

OPTIOND_PLAN.md, Stage 2, "Fine-tuning", from Option C's trainer (scripts/c/train_sim.py: RungWindows, one_step and the
rebuild's seeding order):
  * start: every weight of the released model (pretrain_rnn_ens.pt), all five networks; scales: the authors' fixed
    scales (anymal_d_flat_cfg.py), never the data's own; actions stay raw;
  * loss and optimiser: the rebuild's (rwm_model.compute_loss with the reference loss weights, which equals the
    authors' loss); Adam with weight decay 1e-5 and the run's learning rate, a fresh optimiser state (the review's
    choice; optd/DECISIONS.md D0-2); batch 256 windows; no gradient clipping; everything trained;
  * windows: 32 + 8 rows (rollout, the model predicts 8 steps from its own outputs) or 32 + 1 (teacher-forced), inside
    one episode, a fall only at the last row (the lite loader's rule);
  * data: Option C's rung `rung_1M`; mixed = every row; walking = the episodes of sources 0, 1 and 2 (RWM's released
    policy with action noise 0, 0.1 and 0.3);
  * seed: torch and numpy seeds before the model is built (the rebuild's order), and the batch order's generator;
  * a step is one update on one batch; saves at 5, 20, 40, 80, 160, 320 and 1,280 steps, each a simulator
    `runs/d/<run>_d<steps>/final.pt` (state dict, the authors' scales, run settings);
  * a loss that is not finite ends the run: nothing after it is saved, and `runs/d/<run>/run.json` records the step;
  * resumable: the resume state (model, optimiser, batch generator, torch and numpy random states, loss history) is
    written at every save, every 100 steps and when an invocation stops, to `runs/d/<run>/state.pt`. A run advanced to
    3 steps and then to 5 equals one advanced straight to 5, weight for weight.
One fine-tuning at a time, on 2 threads (the job's business).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
AIR = os.path.dirname(SCRIPTS)
sys.path.insert(0, os.path.join(SCRIPTS, "c"))
import train_sim as TS  # noqa: E402  (Option C's trainer)
sys.path.insert(0, HERE)
import dprogress  # noqa: E402

C, M, T = TS.C, TS.M, TS.T
RUNS_D = os.path.join(AIR, "runs", "d")
DATA_DIR = os.path.join(AIR, "optc", "data")
RUNG = "1M"
WALKING_SOURCES = (0, 1, 2)
SOURCE_COL, EP_COL = 67, 66
SAVES = (5, 20, 40, 80, 160, 320, 1280)
BATCH = 256
HISTORY = 32
STATE_EVERY = 100
# run: (data, learning rate, N, seed); OPTIOND_PLAN.md, Stage 2, the runs' table, in its order
RUNS = {"W_s0": ("walking", 1e-4, 8, 0), "M_s0": ("mixed", 1e-4, 8, 0), "Wtf_s0": ("walking", 1e-4, 1, 0),
        "W_s1": ("walking", 1e-4, 8, 1), "M_s1": ("mixed", 1e-4, 8, 1), "Wg_s0": ("walking", 1e-5, 8, 0),
        "Mg_s0": ("mixed", 1e-5, 8, 0)}


def spec(run, test_spec=None):
    if run in RUNS:
        d, lr, n, s = RUNS[run]
        return {"run": run, "data": d, "lr": lr, "N": n, "seed": s, "rung": RUNG}
    if run.startswith("_test") and test_spec:
        return {"run": run, **test_spec}
    raise SystemExit(f"unknown run {run!r}; the plan's runs are {list(RUNS)} (`_test*` names for tests)")


def authors_scales():
    cfg = C.data()[0]
    return np.asarray(cfg["state_data_mean"], np.float64), np.asarray(cfg["state_data_std"], np.float64)


def windows(data, N, rung=RUNG, data_dir=DATA_DIR):
    """Option C's loader on the rung, in the authors' scales; walking keeps the episodes of sources 0, 1, 2."""
    a = np.load(os.path.join(data_dir, f"rung_{rung}.npy"), mmap_mode="r")
    ep = np.load(os.path.join(data_dir, f"rung_{rung}.ep.npy"))
    fall = np.load(os.path.join(data_dir, f"rung_{rung}.fall.npy"))
    am, asd = authors_scales()
    episodes = None
    if data == "walking":
        src = np.asarray(a[:, SOURCE_COL]).astype(np.int64)
        episodes = set(np.unique(ep[np.isin(src, WALKING_SOURCES)]).tolist())
    elif data != "mixed":
        raise ValueError(data)
    return TS.RungWindows(a, ep, fall, am, asd, HISTORY + N, episodes=episodes)


def setup(sp, cfg=None):
    """Seeds, then the model with every weight of the released model, Adam, and the batch generator."""
    cfg = cfg or C.data()[0]
    torch.manual_seed(sp["seed"])
    np.random.seed(sp["seed"])
    model = M.build_from_config(cfg, ensemble_size=5)
    sd = torch.load(C.W_RELEASED, map_location="cpu", weights_only=False)["system_dynamics_state_dict"]
    model.load_state_dict(sd, strict=True)
    opt = torch.optim.Adam(model.parameters(), lr=sp["lr"], weight_decay=cfg["weight_decay"])
    gen = torch.Generator().manual_seed(sp["seed"])
    return model, opt, gen


def loss(model, batch, teacher_forcing, weights):
    """The rebuild's weighted loss on one batch (no update)."""
    state, action, ext, contact, term = batch
    model.train()
    model.reset()
    terms = model.compute_loss(state, action, ext, contact, term, teacher_forcing=teacher_forcing, loss_type="mse")
    return M.weighted_total(terms, weights)


def _atomic_save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def _write_json(obj, path):
    with open(path + ".tmp", "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(path + ".tmp", path)


def save_state(path, model, opt, gen, hist, step):
    _atomic_save({"model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(),
                  "gen_state": gen.get_state(), "torch_state": torch.get_rng_state(),
                  "numpy_state": np.random.get_state(), "hist": hist, "step": int(step)}, path)


def save_sim(sp, model, step, runs):
    am, asd = authors_scales()
    name = f"{sp['run']}_d{step}"
    _atomic_save({"model_state_dict": model.state_dict(), "state_mean": am.tolist(), "state_std": asd.tolist(),
                  "ensemble": 5, "iter": int(step), "steps": int(step), "run": sp["run"], "data": sp["data"],
                  "lr": sp["lr"], "N": sp["N"], "seed": sp["seed"], "rung": sp["rung"], "batch": BATCH,
                  "start": "pretrain_rnn_ens.pt (every weight)", "scales": "authors"},
                 os.path.join(runs, name, "final.pt"))
    return name


def status(run, runs=RUNS_D):
    p = os.path.join(runs, run, "run.json")
    return json.load(open(p)) if os.path.exists(p) else {"run": run, "step": 0, "saved": [], "nonfinite_at": None}


def advance(run, to, runs=RUNS_D, threads=2, test_spec=None, saves=SAVES, job="ladder", data_dir=DATA_DIR):
    """Advance `run` to `to` steps (resuming), saving each step count of `saves` it passes. Returns run.json."""
    sp = spec(run, test_spec)
    torch.set_num_threads(threads)
    run_dir = os.path.join(runs, run)
    os.makedirs(run_dir, exist_ok=True)
    st = status(run, runs)
    if st.get("nonfinite_at") is not None:
        print(f"[finetune] {run}: ended by a loss that is not finite at step {st['nonfinite_at']}")
        return st
    if st["step"] >= to:
        print(f"[finetune] {run}: at step {st['step']}, nothing to do")
        return st
    cfg = C.data()[0]
    ds = windows(sp["data"], sp["N"], sp["rung"], data_dir)
    model, opt, gen = setup(sp, cfg)
    hist, step = TS.new_hist(), 0
    state_path = os.path.join(run_dir, "state.pt")
    if os.path.exists(state_path):
        ck = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        opt.load_state_dict(ck["optimizer_state_dict"])
        gen.set_state(ck["gen_state"])
        torch.set_rng_state(ck["torch_state"])
        np.random.set_state(ck["numpy_state"])
        hist, step = ck["hist"], int(ck["step"])
        print(f"[finetune] {run}: resuming at step {step}")
    if step != st["step"]:
        raise SystemExit(f"{run}: resume state at step {step}, run.json says {st['step']}")
    st.update({"run": run, **{k: sp[k] for k in ("data", "lr", "N", "seed", "rung")}, "batch": BATCH,
               "windows": len(ds), "weight_decay": cfg["weight_decay"], "saves": list(saves),
               "torch": torch.__version__, "threads": threads})
    teacher_forcing = sp["N"] == 1
    weights = cfg["loss_weights"]
    t0, last = time.perf_counter(), time.time()
    start = step
    while step < to:
        TS.one_step(model, opt, ds, gen, BATCH, teacher_forcing, weights, hist)
        if not np.isfinite(hist["total"][-1]):
            st.update(nonfinite_at=step + 1, step=step, ended="loss not finite")
            print(f"[finetune] {run}: loss not finite at step {step + 1}; the run ends")
            break
        step += 1
        if step in saves:
            st.setdefault("saved", []).append(save_sim(sp, model, step, runs))
            save_state(state_path, model, opt, gen, hist, step)
        elif step % STATE_EVERY == 0 or step == to:
            save_state(state_path, model, opt, gen, hist, step)
        if step in saves or step % STATE_EVERY == 0 or step == to or time.time() - last > 300:
            st["step"] = step
            st["loss_last"] = {k: float(np.mean(v[-20:])) for k, v in hist.items()}
            _write_json(st, os.path.join(run_dir, "run.json"))
            dprogress.write(job, f"fine-tune {run}", step, to)
            last = time.time()
    st["seconds_per_step_last"] = (time.perf_counter() - t0) / max(1, step - start)
    st["curves"] = {k: hist[k] for k in ("total", "state", "grad_norm") if k in hist}
    _write_json(st, os.path.join(run_dir, "run.json"))
    return st


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--to", type=int, default=None)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--runs", default=RUNS_D)
    ap.add_argument("--job", default="ladder")
    a = ap.parse_args(argv)
    if a.status or a.to is None:
        s = status(a.run, a.runs)
        print(json.dumps({k: s.get(k) for k in ("run", "step", "saved", "nonfinite_at")}))
        return s
    s = advance(a.run, a.to, a.runs, a.threads, job=a.job)
    print(json.dumps({k: s.get(k) for k in ("run", "step", "saved", "nonfinite_at", "seconds_per_step_last")}))
    return s


if __name__ == "__main__":
    main()

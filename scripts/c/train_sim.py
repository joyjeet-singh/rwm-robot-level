"""Option C (C1): train one five-network simulator of the MuJoCo ANYmal D on a data rung.

    ./py scripts/c/train_sim.py --rung 6M --N 8 --seed 0 --steps S --threads 2    # -> runs/c/sim_6M_N8_s0/final.pt

Copied, in substance, from the rebuild's training driver and loader; only the data loading differs:
  source ../rwm_repro/scripts/step5_train.py  sha256 dc4e9a16474e1923457127efb8dac91f84c8fba6c31685b25059c820d212db21
  source ../rwm_repro/src/rwm_train.py        sha256 7aeaa2c0d1b19376f1a13c87f0cc1126a97d0b886aee7bda2f3dbfcbb02174a8

Same as the rebuild: rwm_model.build_from_config with `ensemble_size` 5, its Adam settings (learning rate and weight
decay from the reference config), batch 256 (the value the rebuild's A5 runs recorded, results/step5_armA_seed0_ens5.json,
not the lite config's 1,024; see optc/DECISIONS.md), the loss and its weights, seeding order, no gradient clipping.

Changed, as OPTIONC_PLAN.md says:
  * the loader (`RungWindows`): windows of 32 + N rows from a memory-mapped rung, never crossing an episode end, a fall
    only at a window's last row (the lite loader's rule). The rebuild's two checks for the recorded file (column 65 all
    zero, shape (10000, 66)) are not used here;
  * normalisation: the rung's own state mean and spread (actions stay raw), stored in the checkpoint;
  * N = 8 (rollout-trained) or N = 1 (teacher-forced).

Resumable: a rolling checkpoint every 250 steps holds the model, optimiser, random states and loss history. A finished
run (final.pt and train.json) is skipped.
"""
import argparse
import json
import os
import resource
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
AIR = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(AIR, "scripts"))
import common as C  # noqa: E402

R, M = C.R, C.M
import rwm_train as T  # noqa: E402

DATA_DIR = os.path.join(AIR, "optc", "data")
RUNS = os.path.join(AIR, "runs", "c")
PROGRESS = os.path.join(AIR, "results", "c", "progress.json")
HISTORY = 32
BATCH = 256                     # the rebuild's (see the module text)
TERMS = M.TOTAL_WEIGHT_KEYS
CKPT_EVERY = 250
PROGRESS_EVERY = 50


class RungWindows:
    """Windows of `window` rows from an array of rows, with the interface of rwm_train.WindowDataset.

    data: rows with at least 66 columns (45 state, 12 action, 8 contacts, fall); may be memory-mapped.
    ep: episode id per row, whole episodes contiguous; negative ids are excluded.
    A window starts at s if rows s .. s + window - 1 share an episode and rows s .. s + window - 2 hold no fall.
    """

    def __init__(self, data, ep, fall, mean, std, window, episodes=None):
        self.data, self.window = data, int(window)
        ep = np.asarray(ep)
        fall = np.asarray(fall).astype(np.int64)
        self.mean, self.std = np.asarray(mean, dtype=np.float64), np.asarray(std, dtype=np.float64)
        n, L = len(ep), self.window
        first, last = ep[:n - L + 1], ep[L - 1:]
        cf = np.cumsum(np.r_[0, fall])
        ok = (first >= 0) & (first == last) & (cf[L - 1:n] - cf[:n - L + 1] == 0)
        starts = np.flatnonzero(ok)
        if episodes is not None:
            starts = starts[np.isin(ep[starts], list(episodes))]
        assert len(starts) > 0, "no valid windows"
        self.starts = starts
        self.offs = np.arange(L)

    def __len__(self):
        return len(self.starts)

    def batch(self, idx):
        s = self.starts[np.asarray(idx)]
        rows = np.asarray(self.data[s[:, None] + self.offs], dtype=np.float64)
        return (torch.as_tensor((rows[:, :, :45] - self.mean) / self.std, dtype=torch.float32),
                torch.as_tensor(rows[:, :, 45:57], dtype=torch.float32),
                torch.zeros(len(s), self.window, 0),
                torch.as_tensor(rows[:, :, 57:65], dtype=torch.float32),
                torch.as_tensor(rows[:, :, [65]], dtype=torch.float32))

    def sample(self, batch_size, generator):
        idx = torch.randint(0, len(self), (batch_size,), generator=generator)
        return self.batch(idx)


def load_rung(name, window, data_dir=DATA_DIR):
    a = np.load(os.path.join(data_dir, f"rung_{name}.npy"), mmap_mode="r")
    ep = np.load(os.path.join(data_dir, f"rung_{name}.ep.npy"))
    fall = np.load(os.path.join(data_dir, f"rung_{name}.fall.npy"))
    st = json.load(open(os.path.join(data_dir, f"rung_{name}.stats.json")))
    return RungWindows(a, ep, fall, st["state_mean"], st["state_std"], window), st


def grad_norm(model):
    tot = 0.0
    for p in model.parameters():
        if p.grad is not None:
            tot += float(p.grad.detach().pow(2).sum())
    return tot ** 0.5


def new_hist():
    return {k: [] for k in (*TERMS, "total", "grad_norm")}


def one_step(model, opt, ds, gen, batch, teacher_forcing, weights, hist):
    """One step, in the rebuild's order (step5_train.py:255-265)."""
    state, action, ext, contact, term = ds.sample(batch, gen)
    model.train()
    model.reset()
    opt.zero_grad(set_to_none=True)
    terms = model.compute_loss(state, action, ext, contact, term, teacher_forcing=teacher_forcing, loss_type="mse")
    total = M.weighted_total(terms, weights)
    total.backward()
    gn = grad_norm(model)
    opt.step()
    for k, v in zip(TERMS, terms):
        hist[k].append(float(v))
    hist["total"].append(float(total))
    hist["grad_norm"].append(gn)


def setup(cfg, seed, ensemble=5):
    """Seeding order of the rebuild (step5_train.py:241-245)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = M.build_from_config(cfg, ensemble_size=ensemble)
    opt = T.make_optimizer(model, cfg)
    gen = torch.Generator().manual_seed(seed)
    return model, opt, gen


def progress(name, done, total, path=PROGRESS):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rec = {"job": "train_sim", "step": f"train {name}", "done": int(done), "total": int(total),
           "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh)
    os.replace(tmp, path)


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 30)   # macOS reports bytes


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", required=True, help="10k, 100k, 1M, 6M or d6")
    ap.add_argument("--N", type=int, choices=[1, 8], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=0, help="S; required unless --probe")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--name", default=None)
    ap.add_argument("--probe", type=int, default=0, help="time this many steps, write nothing")
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--ckpt-every", type=int, default=CKPT_EVERY, help="below 250 for tests only")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    name = a.name or f"sim_{a.rung}_N{a.N}_s{a.seed}"
    run_dir = os.path.join(a.runs, name)
    final, summary = os.path.join(run_dir, "final.pt"), os.path.join(run_dir, "train.json")
    if not a.probe and os.path.exists(final) and os.path.exists(summary):
        print(f"[train_sim] {name}: finished, skipped")
        return
    if not a.probe and a.steps <= 0:
        ap.error("--steps is required")

    cfg = R.load_reference_config(R.repo_paths()["lite"])
    window = HISTORY + a.N
    ds, stats = load_rung(a.rung, window, a.data_dir)
    teacher_forcing = a.N == 1
    weights = cfg["loss_weights"]
    model, opt, gen = setup(cfg, a.seed)

    if a.probe:
        hist, times = new_hist(), []
        for _ in range(a.probe):
            t = time.perf_counter()
            one_step(model, opt, ds, gen, a.batch, teacher_forcing, weights, hist)
            times.append(time.perf_counter() - t)
        print(json.dumps({"rung": a.rung, "N": a.N, "windows": len(ds), "steps": a.probe, "threads": a.threads,
                          "seconds_per_step": float(np.mean(times[1:])), "seconds_first_step": times[0],
                          "peak_rss_gb": peak_rss_gb()}))
        return

    os.makedirs(run_dir, exist_ok=True)
    hist, start, ckpts = new_hist(), 0, sorted(f for f in os.listdir(run_dir) if f.startswith("rolling_"))
    if ckpts:
        latest = max(ckpts, key=lambda f: int(f[len("rolling_"):-3]))
        ck = torch.load(os.path.join(run_dir, latest), map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        opt.load_state_dict(ck["optimizer_state_dict"])
        gen.set_state(ck["gen_state"])
        torch.set_rng_state(ck["torch_state"])
        hist, start = ck["hist"], int(ck["iter"])
        print(f"[train_sim] {name}: resuming at step {start}")

    t0 = time.perf_counter()
    for it in range(start, a.steps):
        one_step(model, opt, ds, gen, a.batch, teacher_forcing, weights, hist)
        step = it + 1
        if not a.no_progress and (step % PROGRESS_EVERY == 0 or step == a.steps):
            progress(name, step, a.steps)
        if step % a.ckpt_every == 0 and step < a.steps:
            p = os.path.join(run_dir, f"rolling_{step}.pt")
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(),
                        "gen_state": gen.get_state(), "torch_state": torch.get_rng_state(), "hist": hist, "iter": step},
                       p + ".tmp")
            os.replace(p + ".tmp", p)
            old = sorted((f for f in os.listdir(run_dir) if f.startswith("rolling_")),
                         key=lambda f: int(f[len("rolling_"):-3]))
            for f in old[:-2]:
                os.remove(os.path.join(run_dir, f))
    elapsed = time.perf_counter() - t0

    tmp = final + ".tmp"
    torch.save({"model_state_dict": model.state_dict(), "iter": a.steps, "seed": a.seed, "rung": a.rung, "N": a.N,
                "state_mean": stats["state_mean"], "state_std": stats["state_std"], "ensemble": 5}, tmp)
    os.replace(tmp, final)
    rec = {"name": name, "rung": a.rung, "N": a.N, "seed": a.seed, "steps": a.steps, "batch": a.batch,
           "teacher_forcing": teacher_forcing, "ensemble": 5, "windows": len(ds), "rows": stats["rows"],
           "learning_rate": cfg["learning_rate"], "weight_decay": cfg["weight_decay"], "loss_weights": weights,
           "seconds_this_run": elapsed, "seconds_per_step": elapsed / max(1, a.steps - start),
           "torch": torch.__version__, "threads": a.threads, "peak_rss_gb": peak_rss_gb(), "curves": hist}
    with open(summary + ".tmp", "w") as fh:
        json.dump(rec, fh)
    os.replace(summary + ".tmp", summary)
    print(json.dumps({k: rec[k] for k in ("name", "steps", "seconds_per_step", "peak_rss_gb")}))


if __name__ == "__main__":
    main()

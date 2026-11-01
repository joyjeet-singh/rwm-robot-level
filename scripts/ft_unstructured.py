"""Fine-tune one magnitude-pruned A model with the pruning mask held (study FT, prereg/PREREG_FT.md).

Copied from ../PerconAI/scripts/e3_finetune.py (sha256 b4e92994a5e277304ab4be8e22fdac2904d6ff69ac9a24235adad785616802b2).
Changes, and only these: PerconAI's env and harness helpers are replaced by their equivalents in common.py
(data, load, run_path, fold_dir, train_episodes, sha256); the start is compress.unstructured(fp32 model, level)
instead of perconai.prune.structured; every pruned entry is set to exactly zero after every opt.step(); the final
count of zeros is checked and recorded (C-FT1), and a settings difference is recorded without training (C-FT3), so
that a failed check excludes the arm instead of stopping the study (PREREG_FT.md section 7); outputs go to this folder; the probe times one compute.per_model scoring.

The loop mirrors ../rwm_repro/scripts/step5_train.py:241-275 without its saving: Arm A's rollout loss
(teacher_forcing=False, loss "mse"), the fold's training episodes only (40-row windows), batch 256, Adam at the
learning rate and weight decay of the reference config, no gradient clipping, fresh optimizer state. Before training
it asserts that these settings equal the source run's record (`hyperparameters`: batch, learning_rate, weight_decay,
loss_weights, gradient_clipping, train_episodes, n_train_windows). Seeds: torch.manual_seed, np.random.seed and the
batch generator all take the run's seed. Threads: 2, recorded.

  ./py scripts/ft_unstructured.py <id> <fold a-b> <seed> <level> <iterations>
      id = ft_u30_<rid> or ft_u50_<rid>, rid = common.fold_dir("A", fold, seed)
      -> runs/ft/<id>/weights_<iterations>.pt (gitignored) and results/raw/ft/<id>.json
  ./py scripts/ft_unstructured.py --probe <iterations>
      fold 1-8, seed 0, level 0.30, nothing saved but results/ft_probe.json: seconds per iteration, one timed
      per_model scoring of the parent, projected hours per arm and for scoring, and the time rule's decision
"""
import json
import math
import os
import platform
import sys
import time

import numpy as np
import torch

import common as C
import compress as K

import rwm_model as M  # noqa: E402  (on sys.path through common)
import rwm_train as T  # noqa: E402

SCHEMA = "air2027.ft.v1"
ITERS = 500
BATCH = 256
THREADS = 2
TERMS = M.TOTAL_WEIGHT_KEYS
LOG_EVERY = 25
SEEDS = (0, 1, 2)
LEVELS = {"u30": 0.30, "u50": 0.50}
CAP_HOURS = 8.0
CODE = ("AIR 2027/scripts/ft_unstructured.py", "AIR 2027/scripts/compress.py", "AIR 2027/scripts/common.py",
        "rwm_repro/src/rwm_train.py", "rwm_repro/src/rwm_model.py")


def out_path(*parts):
    p = os.path.abspath(os.path.join(C.AIR, *parts))
    if os.path.commonpath([p, C.AIR]) != C.AIR:
        raise PermissionError(f"refusing to write outside AIR 2027: {p}")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def run_id(fold, seed):
    return C.fold_dir("A", fold, seed)


def ft_id(level, fold, seed):
    return f"ft_u{int(round(level * 100))}_{run_id(fold, seed)}"


def source_record(rid, fold):
    name = f"step5_{rid}.json" if tuple(fold) == (1, 8) else f"x3_run_{rid[len('x3_'):]}.json"
    with open(os.path.join(C.RWM, "results", name)) as f:
        return os.path.join("rwm_repro", "results", name), json.load(f)["hyperparameters"]


def masks(model):
    """[(name, weight, bool mask of its pruned entries)] for every pruned weight matrix of a fresh pruned copy."""
    return [(n, w, (w.detach() == 0)) for n, w in K.weight_matrices(model)]


def hold(mk):
    with torch.no_grad():
        for _, w, m in mk:
            w.masked_fill_(m, 0.0)


def zeros(model):
    return int(sum(int((w.detach() == 0).sum()) for _, w in K.weight_matrices(model)))


def setup(fold, seed, level, strict=True):
    torch.set_num_threads(THREADS)
    rid = run_id(fold, seed)
    cfg, data, ep = C.data()
    train_eps = C.train_episodes(fold)
    ds = T.WindowDataset(data, ep, train_eps, cfg)
    model, pinfo = K.unstructured(C.load(C.run_path(rid), 1), level)
    mk = masks(model)
    assert sum(int(m.sum()) for _, _, m in mk) == pinfo["pruned_entries"] == zeros(model), "mask != pruned count"
    rec_path, hp = source_record(rid, fold)
    want = {"batch": BATCH, "learning_rate": cfg["learning_rate"], "weight_decay": cfg["weight_decay"],
            "loss_weights": cfg["loss_weights"], "gradient_clipping": None,
            "train_episodes": list(train_eps), "n_train_windows": len(ds)}
    diff = {k: (v, hp.get(k)) for k, v in want.items() if hp.get(k) != v}
    if hp.get("holdout_episodes") != sorted(fold):
        diff["holdout_episodes"] = (sorted(fold), hp.get("holdout_episodes"))
    if strict:
        assert not diff, f"settings differ from {rec_path} (ours, record): {diff}"
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = T.make_optimizer(model, cfg)
    gen = torch.Generator().manual_seed(seed)
    if not strict:
        return rid, cfg, ds, model, pinfo, mk, opt, gen, rec_path, want, diff
    return rid, cfg, ds, model, pinfo, mk, opt, gen, rec_path, want


def grad_norm(model):
    """step5_train.py:52-57."""
    return sum(float(p.grad.detach().pow(2).sum()) for p in model.parameters() if p.grad is not None) ** 0.5


def train(model, mk, opt, ds, gen, cfg, iters, verbose=True):
    hist = {k: [] for k in TERMS}
    hist["total"], hist["grad_norm"], hist["s"] = [], [], []
    t0 = time.perf_counter()
    for it in range(iters):
        ti = time.perf_counter()
        state, action, ext, contact, term = ds.sample(BATCH, gen)
        model.train()
        model.reset()
        opt.zero_grad(set_to_none=True)
        terms = model.compute_loss(state, action, ext, contact, term, teacher_forcing=False, loss_type="mse")
        total = M.weighted_total(terms, cfg["loss_weights"])
        total.backward()
        gn = grad_norm(model)
        opt.step()
        hold(mk)                                                    # the mask is held after every step
        for k, v in zip(TERMS, terms):
            hist[k].append(float(v))
        hist["total"].append(float(total))
        hist["grad_norm"].append(gn)
        hist["s"].append(time.perf_counter() - ti)
        if verbose and (it % LOG_EVERY == 0 or it == iters - 1):
            print(f"  {it:>5d} state {float(terms[0]):.5f} total {float(total):.5f} |grad| {gn:.3f}"
                  f" {time.perf_counter() - t0:.0f} s", flush=True)
    model.eval()
    model.reset()
    return hist, time.perf_counter() - t0


def mask_check(model, mk, pinfo):
    """C-FT1: every pruned entry exactly zero, and the count of zeros equals the count pruned."""
    held = all(bool((w.detach()[m] == 0).all()) for _, w, m in mk)
    nz = zeros(model)
    return {"mask_entries_all_zero": held, "mask_zeros": nz, "pruned_entries": pinfo["pruned_entries"],
            "passed": bool(held and nz == pinfo["pruned_entries"])}


def versions():
    return {"torch_version": torch.__version__, "numpy_version": np.__version__,
            "python_version": platform.python_version(), "machine": platform.platform(),
            "code_sha256": {k: C.sha256(os.path.join(C.PDM, k)) for k in CODE}}


def run(rid_out, fold, seed, level, iters):
    """Exit 0 whenever the run's record is written, whatever its checks found (PREREG_FT.md section 7: a failed check
    excludes the arm; it does not stop the study). A crash exits non-zero and writes nothing."""
    assert rid_out == ft_id(level, fold, seed), (rid_out, ft_id(level, fold, seed))
    rid, cfg, ds, model, pinfo, mk, opt, gen, rec_path, hp, diff = setup(fold, seed, level, strict=False)
    if diff:                                                        # C-FT3 failed: no training, no weights
        out = {"schema": SCHEMA, "id": rid_out, "source_run": rid, "source_record": rec_path,
               "settings_check": "differ", "settings_diff": {k: list(v) for k, v in diff.items()},
               "fold": list(fold), "seed": seed, "level": level, "trained": False, **versions()}
        with open(out_path("results", "raw", "ft", f"{rid_out}.json"), "w") as f:
            json.dump(out, f, indent=1)
        print(f"{rid_out}: C-FT3 FAILED, settings differ from {rec_path}: {diff}; not trained", flush=True)
        return 0
    print(f"{rid_out}: {rid} -> unstructured {level:.2f}, {iters} iterations, {len(ds)} windows", flush=True)
    hist, wall = train(model, mk, opt, ds, gen, cfg, iters)
    bad = [k for k, v in hist.items() if not all(map(math.isfinite, v))]
    chk = mask_check(model, mk, pinfo)
    if not chk["passed"]:
        print(f"{rid_out}: C-FT1 FAILED: {chk}", flush=True)
    wpath = out_path("runs", "ft", rid_out, f"weights_{iters}.pt")
    torch.save({"model_state_dict": model.state_dict(), "level": level, "iter": iters, "source_run": rid,
                "seed": seed, "fold": list(fold)}, wpath)
    src = C.run_path(rid)
    prune = {k: pinfo[k] for k in ("kind", "level", "pool_entries", "pruned_entries", "threshold_abs",
                                   "largest_kept_abs_min", "per_matrix")}
    out = {"schema": SCHEMA, "id": rid_out, "source_run": rid, "source_weights_path": os.path.relpath(src, C.PDM),
           "source_weights_sha256": C.sha256(src), "source_record": rec_path, "settings_check": "equal",
           "settings_diff": {}, "trained": True,
           "fold": list(fold), "seed": seed, "level": level, "prune": prune,
           "pruned_entries": pinfo["pruned_entries"], "mask_zeros": chk["mask_zeros"], "mask_check": chk,
           "hyperparameters": {**hp, "iterations": iters, "rnn_hidden_size": 256, "rnn_num_layers": 2,
                               "teacher_forcing": False, "loss_type": "mse", "ensemble": 1,
                               "holdout_episodes": sorted(fold), "optimizer": "Adam, fresh state",
                               "mask": "pruned entries set to 0 after every opt.step()",
                               "torch_threads": torch.get_num_threads()},
           "wall_clock_s": wall, "s_per_iter": wall / iters, "non_finite_traces": bad,
           "final_terms": {k: hist[k][-1] for k in list(TERMS) + ["total"]},
           "curves": {k: v for k, v in hist.items() if k != "s"},
           "weights_path": os.path.relpath(wpath, C.AIR), "weights_sha256": C.sha256(wpath), **versions()}
    with open(out_path("results", "raw", "ft", f"{rid_out}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"{rid_out}: done, {wall:.0f} s ({wall / iters:.2f} s/iter); final total {hist['total'][-1]:.5f};"
          f" non-finite traces {bad}; C-FT1 {'passed' if chk['passed'] else 'FAILED'}")
    return 0


def probe(iters):
    import compute as X
    t0 = time.perf_counter()
    rid, cfg, ds, model, pinfo, mk, opt, gen, _, _ = setup((1, 8), 0, 0.30)
    t_setup = time.perf_counter() - t0
    hist, wall = train(model, mk, opt, ds, gen, cfg, iters, verbose=False)
    s = hist["s"][1:]                                               # the first iteration warms up
    spi = float(np.mean(s))
    t0 = time.perf_counter()
    ar, sc = X.setup_arena((1, 8))
    t_arena = time.perf_counter() - t0
    t0 = time.perf_counter()
    X.per_model(C.load(C.run_path(rid), 1), ar, sc)
    t_score = time.perf_counter() - t0
    n_par = len(C.FOLDS) * len(SEEDS)
    run_h = n_par * (ITERS * spi + t_setup) / 3600                  # one fine-tuned arm, 15 runs
    n_models = {"without_u50": 4 * n_par, "with_u50": 6 * n_par}    # P, U30, U30F, CTRL (+ U50, U50F)
    score_h = {k: (n * t_score + len(C.FOLDS) * t_arena) / 3600 for k, n in n_models.items()}
    total_with = 2 * run_h + score_h["with_u50"]
    run_u50 = total_with <= CAP_HOURS
    out = {"probe": "fold 1-8, seed 0, level 0.30", "iterations": iters, "wall_s": wall,
           "s_per_iter_mean_after_first": spi, "s_per_iter_max": float(np.max(hist["s"])), "setup_s": t_setup,
           "finite": all(math.isfinite(x) for x in hist["total"]), "mask_check": mask_check(model, mk, pinfo),
           "score_one_model_s": t_score, "arena_setup_s": t_arena, "runs_per_arm": n_par,
           "iterations_per_run": ITERS, "projected_hours": {"U30F": run_h, "U50F": run_h,
                                                            "scoring_without_u50": score_h["without_u50"],
                                                            "scoring_with_u50": score_h["with_u50"],
                                                            "total_with_u50": total_with,
                                                            "total_without_u50": run_h + score_h["without_u50"]},
           "cap_hours": CAP_HOURS, "run_u50f": run_u50,
           "arms": ["P", "U30", "U30F", "CTRL"] + (["U50", "U50F"] if run_u50 else []),
           "torch_threads": torch.get_num_threads(), "torch_version": torch.__version__,
           "numpy_version": np.__version__, "python_version": platform.python_version(),
           "machine": platform.platform(),
           "note": "U50F is projected at U30F's speed: pruning leaves the matrices dense, so the cost is the same"}
    with open(out_path("results", "ft_probe.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"{spi:.3f} s/iter (max {out['s_per_iter_max']:.3f}), setup {t_setup:.1f} s, one scoring {t_score:.1f} s")
    print(f"projected: U30F {run_h:.2f} h, U50F {run_h:.2f} h, scoring {score_h['with_u50']:.2f} h;"
          f" total {total_with:.2f} h (cap {CAP_HOURS}) -> U50F {'runs' if run_u50 else 'does not run'}")
    return 0


def main():
    if sys.argv[1] == "--probe":
        sys.exit(probe(int(sys.argv[2])))
    rid_out, fold, seed, level, iters = sys.argv[1:6]
    fold = tuple(int(x) for x in fold.split("-"))
    sys.exit(run(rid_out, fold, int(seed), float(level), int(iters)))


if __name__ == "__main__":
    main()

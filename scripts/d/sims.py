"""Option D (D0): one place that knows every simulator.  ./py

    import sims
    m, mean, std, rec = sims.load("REL__u10")       # the model in eval mode, its scales (45,), a short record
    sims.EXISTING                                    # the 17 simulators of the record (OPTIOND_PLAN.md, Stage 1)

Simulators:
  * `REL`: the released checkpoint (pretrain_rnn_ens.pt), five networks, the authors' scales;
  * `REL__<v>`: its eight compressed copies (Step 2), the variant applied by scripts/compress.py and checked against the
    variant record Step 2 stored in results/s2/checks.json (as scripts/s2/train_policy.py's build_sim checks it);
  * `A_s0`, `B_s0`, `A_s0__u50`, `A5_s0`: rows of pool F18 in results/models.json (rebuilt from the ten recorded
    episodes with 1 and 8 held out), sha256 checked, variant applied;
  * Option C's four (`runs/c/<name>/final.pt`): trained from scratch on MuJoCo data, in their rung's own scales;
  * ladder simulators (`runs/d/<name>/final.pt`, written by scripts/d/finetune.py): the authors' scales.

`rec["scales"]` is "authors" or "own"; `rec["holdout"]` is the recorded episodes Step 2 held out of the simulator's
start states ("1,8" for the F18 rows), "" otherwise. Nothing here writes a file.
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
AIR = os.path.dirname(SCRIPTS)
for p in (SCRIPTS,):
    if p not in sys.path:
        sys.path.append(p)
import common as C  # noqa: E402
import compress as K  # noqa: E402

RUNS_C = os.path.join(AIR, "runs", "c")
RUNS_D = os.path.join(AIR, "runs", "d")
VARIANTS = K.VARIANTS
REL_COPIES = tuple(f"REL__{v}" for v in VARIANTS)
REBUILT = ("A_s0", "B_s0", "A_s0__u50")
A5 = ("A5_s0",)
OPTC = ("sim_6M_N8_s0", "sim_6M_N8_s0__2S", "sim_10k_N8_s0", "sim_10k_N8_s0__2S")
EXISTING = ("REL",) + REL_COPIES + REBUILT + A5 + OPTC          # 17
HOW = {"REL": "released model (RWM's own; five networks)",
       **{s: f"compressed copy of REL ({s[5:]}; Step 2)" for s in REL_COPIES},
       "A_s0": "rebuilt, one network, rollout-trained on 8 of the 10 recorded episodes (Step 2 pilot)",
       "B_s0": "rebuilt, one network, teacher-forced on 8 of the 10 recorded episodes (Step 2 pilot)",
       "A_s0__u50": "A_s0 with 50% unstructured pruning (Step 2 pilot)",
       "A5_s0": "rebuilt, five networks, rollout-trained on 8 of the 10 recorded episodes (Option C)",
       "sim_6M_N8_s0": "Option C, from scratch on 6M MuJoCo transitions, 2,572 steps, own scales",
       "sim_6M_N8_s0__2S": "Option C, from scratch on 6M MuJoCo transitions, 5,144 steps, own scales",
       "sim_10k_N8_s0": "Option C, from scratch on 10k MuJoCo transitions, 2,572 steps, own scales",
       "sim_10k_N8_s0__2S": "Option C, from scratch on 10k MuJoCo transitions, 5,144 steps, own scales"}
HOLDOUT_OF_ARENA = {"18": "1,8", "all": ""}


def authors_scales():
    cfg, _, _ = C.data()
    return (np.asarray(cfg["state_data_mean"], dtype=np.float64), np.asarray(cfg["state_data_std"], dtype=np.float64))


def _json(path):
    with open(path) as fh:
        return json.load(fh)


def _models_row(key):
    t = _json(os.path.join(AIR, "results", "models.json"))
    if key == "REL":
        return t["REL"][0]
    rows = [r for r in t["F18"] if r["key"] == key]
    if len(rows) != 1:
        raise KeyError(f"{key} is not one row of pool F18 in results/models.json")
    return rows[0]


def _variant(m, variant, want):
    """compress.apply, its info recorded as compute.build records it, and checked against the stored record."""
    m, info = K.apply(m, variant)
    if info.get("kind") == "int8w":
        info = {"kind": "int8w", "max_scale": max(info["scales"].values())}
    elif info.get("kind") == "unstructured":
        info = {k: info[k] for k in ("kind", "level", "pool_entries", "pruned_entries", "threshold_abs")}
    elif info.get("kind") == "structured":
        info = {"kind": "structured", "width": info["width"]}
    for k, v in (want or {}).items():
        got = info.get(k)
        same = (abs(float(got) - float(v)) <= 1e-9 * max(1.0, abs(float(v)))
                if isinstance(v, (int, float)) and not isinstance(v, bool) else got == v)
        if not same:
            raise SystemExit(f"variant {variant}: {k} = {got}, the stored record has {v}")
    m.eval()
    return m, info


def path_of(name):
    if name == "REL" or name in REL_COPIES:
        return C.W_RELEASED
    if name in REBUILT + A5:
        return os.path.join(C.PDM, _models_row(name)["path"])
    if name in OPTC:
        return os.path.join(RUNS_C, name, "final.pt")
    return os.path.join(RUNS_D, name, "final.pt")


def kind_of(name):
    if name == "REL":
        return "released"
    if name in REL_COPIES:
        return "compressed"
    if name in REBUILT:
        return "rebuilt"
    if name in A5:
        return "rebuilt5"
    if name in OPTC:
        return "optc"
    return "ladder"


def load(name):
    """(model in eval mode, state mean (45,), state spread (45,), record)."""
    kind = kind_of(name)
    path = path_of(name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"simulator {name}: {path} is missing")
    am, asd = authors_scales()
    rec = {"name": name, "kind": kind, "how": HOW.get(name, "ladder: REL fine-tuned on MuJoCo data (Option D)"),
           "path": os.path.relpath(path, C.PDM), "sha256": C.sha256(path), "variant": None, "variant_info": None,
           "scales": "authors", "holdout": ""}
    if kind in ("released", "compressed"):
        row = _models_row("REL")
        if rec["sha256"] != row["sha256"]:
            raise SystemExit("the released checkpoint differs from results/models.json (sha256)")
        m = C.load(path, 5)
        if kind == "compressed":
            v = name.split("__", 1)[1]
            want = _json(os.path.join(AIR, "results", "s2", "checks.json"))["models"][name]["variant_info"]
            m, info = _variant(m, v, want)
            rec.update(variant=v, variant_info=info)
        rec["ensemble"] = 5
        return m, am, asd, rec
    if kind in ("rebuilt", "rebuilt5"):
        row = _models_row(name)
        if rec["sha256"] != row["sha256"]:
            raise SystemExit(f"weights of {name} differ from results/models.json (sha256)")
        m = C.load(path, int(row["ensemble"]))
        if row["variant"]:
            m, info = _variant(m, row["variant"], row["variant_info"])
            rec.update(variant=row["variant"], variant_info=info)
        rec.update(ensemble=int(row["ensemble"]), holdout=HOLDOUT_OF_ARENA[str(row["arena"])])
        return m, am, asd, rec
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = C.load(path, int(ck["ensemble"]))
    mean, std = np.asarray(ck["state_mean"], dtype=np.float64), np.asarray(ck["state_std"], dtype=np.float64)
    rec["ensemble"] = int(ck["ensemble"])
    if kind == "optc":
        rec.update(scales="own", rung=ck["rung"], N=ck["N"], steps=int(ck["iter"]), seed=ck["seed"])
    else:
        if not (np.allclose(mean, am, rtol=0, atol=1e-6) and np.allclose(std, asd, rtol=0, atol=1e-6)):
            raise SystemExit(f"ladder simulator {name} does not hold the authors' scales")
        rec.update({k: ck[k] for k in ("run", "steps", "data", "lr", "N", "seed") if k in ck})
    return m, mean, std, rec


def ladder_names():
    """Ladder simulators saved so far (runs/d/<run>_d<steps>/final.pt)."""
    if not os.path.isdir(RUNS_D):
        return []
    return sorted(n for n in os.listdir(RUNS_D) if "_d" in n and os.path.exists(os.path.join(RUNS_D, n, "final.pt"))
                  and not n.startswith("_"))


def all_names():
    return list(EXISTING) + ladder_names()

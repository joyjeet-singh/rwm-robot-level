"""
The paper's two figures, from results/raw_X5.npz, results/model_table.json and results/stats.json.
Palette: the first three slots of the dataviz reference palette (validated all-pairs, light mode), with marker
shape as secondary encoding and direct labels, so identity never rests on colour alone.
  figures/fig_horizon.pdf   per-step error against forecast step on the 20 held-out trajectories
  figures/fig_checks.pdf    one-step error and lambda_1 against 100-step error, pool P (F18)
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import common as C  # noqa: E402

BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8985", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "legend.fontsize": 6.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6, "pdf.fonttype": 42,
                     "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
FOLDS = [C.fid(f) for f in C.FOLDS]


def group_curve(raw, table, pred):
    """Per-step error, models averaged within each fold-trajectory, then over the 20 trajectories."""
    cur = []
    for f in FOLDS:
        ks = [r["key"] for r in table if r["arena"] == f and pred(r)]
        cur.append(np.mean([raw[f"err_{k}"] for k in ks], 0))                     # (4, 368)
    return np.concatenate(cur).mean(0)


def fig_horizon():
    raw = np.load(os.path.join(C.RESULTS, "raw_X5.npz"))
    table = json.load(open(os.path.join(C.RESULTS, "model_table.json")))["X5"]
    k = np.arange(1, 369)
    A = group_curve(raw, table, lambda r: r["group"] == "A")
    B = group_curve(raw, table, lambda r: r["group"] == "B")
    U = group_curve(raw, table, lambda r: r["variant"] == "u30")
    F = np.concatenate([raw[f"floor_{f}"] for f in FOLDS]).mean(0)
    fig, ax = plt.subplots(figsize=(3.45, 2.15))
    for h, txt in ((24, "24 steps\n(RWM)"), (100, "100 steps\n(RWM-U)")):
        ax.axvline(h, color="#d6d5d0", lw=0.8, zorder=0)
        ax.text(h * 1.06, 0.012, txt, color=MUTED, fontsize=5.8, va="bottom")
    ax.plot(k, F, color=GRAY, lw=1.2, ls=(0, (4, 2)), label="hold last state (floor)")
    ax.plot(k, B, color=ORANGE, lw=1.6, label="teacher-forced")
    ax.plot(k, U, color=AQUA, lw=1.6, label="rollout-trained, 30% pruned")
    ax.plot(k, A, color=BLUE, lw=1.6, label="rollout-trained")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 368)
    ax.set_ylim(0.01, 4)
    ax.set_xlabel("forecast step k (0.02 s each)")
    ax.set_ylabel("error at step k (state std units)")
    ax.grid(True, which="major", color="#ecebe7", lw=0.5)
    sec = ax.secondary_xaxis("top", functions=(lambda x: x * C.DT, lambda t: t / C.DT))
    sec.set_xlabel("simulated time (s)", fontsize=6.5, color=MUTED)
    sec.tick_params(labelsize=6, colors=MUTED)
    ax.legend(loc="upper left", frameon=False, handlelength=2.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(C.FIGURES, "fig_horizon.pdf"))
    fig.savefig(os.path.join(C.FIGURES, "fig_horizon.png"), dpi=200)
    plt.close(fig)
    return {"A_k1": float(A[0]), "B_k1": float(B[0]), "U_k1": float(U[0]), "F_k1": float(F[0]),
            "A_k368": float(A[-1]), "B_k368": float(B[-1]), "U_k368": float(U[-1]), "F_k368": float(F[-1]),
            "B_cross_floor_step": int(np.argmax(B > F) + 1) if (B > F).any() else None,
            "U_cross_floor_step": int(np.argmax(U > F) + 1) if (U > F).any() else None,
            "B_below_A_steps": int((B < A).sum())}


def fig_checks():
    table = json.load(open(os.path.join(C.RESULTS, "model_table.json")))["F18"]
    st = json.load(open(os.path.join(C.RESULTS, "stats.json")))
    fl = st["floor_F18"]["E100"]
    fam = {"R": ("rollout-trained", BLUE, "o"), "T": ("trained one step ahead", ORANGE, "^"),
           "C": ("compressed (rollout-trained parent)", AQUA, "s")}
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.85), sharey=True)
    for ax, (xk, xl, rk) in zip(axs, (("E1", "one-step error $E_1$", "Q1"),
                                      ("lam1", r"Lyapunov exponent $\lambda_1$ (1/s)", "Q1"))):
        for p, (lab, colr, mk) in fam.items():
            rr = [r for r in table if r["pool"] == p]
            x = np.array([r[xk] for r in rr])
            y = np.array([r["E100"] for r in rr])
            ax.scatter(x, y, s=11, c=colr, marker=mk, edgecolors="white", linewidths=0.4, label=lab, zorder=3)
        ax.axhline(fl, color=GRAY, lw=0.9, ls=(0, (4, 2)), zorder=1)
        ax.set_yscale("log")
        if xk == "E1":
            ax.set_xscale("log")
            rho = st[rk]["rho_base"]
        else:
            ax.axvline(0, color="#d6d5d0", lw=0.8, zorder=0)
            rho = st[rk]["rho_new"]
        ax.set_xlabel(xl)
        ax.text(0.04, 0.95, f"Spearman {rho:+.2f}", transform=ax.transAxes, fontsize=6.3, color=INK, va="top")
        ax.grid(True, which="major", color="#ecebe7", lw=0.5)
    axs[0].set_ylabel("100-step error $E_{100}$")
    axs[0].text(0.97, fl * 1.08, "floor", transform=axs[0].get_yaxis_transform(), ha="right", fontsize=5.8,
                color=MUTED)
    h, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=3, frameon=False, fontsize=5.6, handletextpad=0.2,
               columnspacing=0.8, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(pad=0.3, rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(C.FIGURES, "fig_checks.pdf"))
    fig.savefig(os.path.join(C.FIGURES, "fig_checks.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(C.FIGURES, exist_ok=True)
    facts = fig_horizon()
    fig_checks()
    json.dump(facts, open(os.path.join(C.RESULTS, "figure_facts.json"), "w"), indent=1)
    print(facts)

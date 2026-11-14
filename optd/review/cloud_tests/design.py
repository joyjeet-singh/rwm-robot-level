"""Design calculation: how many policies per tested simulator?
A simulated ladder study (the plan's search, 6 runs x 3 tests + REL) under different replication rules.
Per-policy scores are bimodal: a policy 'fails' with probability q (score near 0.06) or walks (score near m).
Reports cost, how well the measured unit scores track the true ones, label errors, and the power of
registered-style tests (bootstrap interval of a rank correlation, and of a difference of two)."""
import sys, json
import numpy as np
from scipy.stats import rankdata

T_FAIL = 0.15
HOURS_PER_POLICY = 2.05

def q_of(dose, dstar, w):
    return 1.0 / (1.0 + np.exp(-(np.log2(dose) - np.log2(dstar)) / w))

def draw(rng, q, m, sd_walk):
    if rng.random() < q:
        return float(np.clip(rng.normal(0.06, 0.03), 0.0, 0.14))
    return float(np.clip(rng.normal(m, sd_walk), 0.15, 0.95))

class Sim:
    def __init__(self, rng, q, m, sd_walk):
        self.rng, self.q, self.m, self.sd = rng, q, m, sd_walk
        self.scores = []
    def need(self, k):
        while len(self.scores) < k:
            self.scores.append(draw(self.rng, self.q, self.m, self.sd))
    @property
    def mu(self):
        return self.q * 0.06 + (1 - self.q) * self.m
    def fails(self, k):
        self.need(k)
        return [s < T_FAIL for s in self.scores[:k]]

def search_failed(sim, decide):
    """Is the simulator 'failed' for the search?  first: its first policy; pair: both of two policies."""
    if decide == "first":
        return sim.fails(1)[0]
    f = sim.fails(2)
    if decide == "pair":
        return f[0] and f[1]
    if f[0] == f[1]:                          # "maj": two policies, and a third only if they disagree
        return f[0]
    return sum(sim.fails(3)) >= 2

def replicate(sim, rule):
    f1 = sim.fails(1)[0]
    if rule == "1": k = 1
    elif rule == "2nf": k = 1 if f1 else 2
    elif rule == "2": k = 2
    elif rule == "2+1":
        f = sim.fails(2); k = 2 if f[0] == f[1] else 3
    elif rule == "3": k = 3
    elif rule == "3+2":
        f = sim.fails(3); k = 3 if sum(f) in (0, 3) else 5
    elif rule == "5": k = 5
    elif rule == "2..5":                      # add policies until one side leads by two, or five are trained
        k = 2
        while True:
            f = sim.fails(k); nf = sum(f)
            if abs(2 * nf - k) >= 2 or k >= 5: break
            k += 1
    else: raise ValueError(rule)
    sim.need(k)
    sc = sim.scores[:k]
    nf = sum(s < T_FAIL for s in sc)
    return float(np.mean(sc)), (nf * 2 > k), k

def run_search(rng, dstar, w, m, sd_walk, decide):
    sims = {}
    def test(d):
        s = Sim(rng, q_of(d, dstar, w), m, sd_walk); sims[d] = s
        return search_failed(s, decide)
    f80 = test(80)
    if f80:
        f20 = test(20); test(5 if f20 else 40)
    else:
        f320 = test(320); test(160 if f320 else 1280)
    return sims

def boot_tests(rng, A, B, s, nb=400):
    n = len(s)
    idx = rng.integers(0, n, (nb, n))
    def rho(x, y):
        rx = rankdata(x[idx], axis=1); ry = rankdata(y[idx], axis=1)
        rx = rx - rx.mean(1, keepdims=True); ry = ry - ry.mean(1, keepdims=True)
        den = np.sqrt((rx ** 2).sum(1) * (ry ** 2).sum(1))
        ok = den > 0
        out = np.full(nb, np.nan); out[ok] = (rx * ry).sum(1)[ok] / den[ok]
        return out
    ra, rb = rho(A, s), rho(B, s)
    ok = np.isfinite(ra) & np.isfinite(rb)
    if ok.sum() < nb * 0.8: return False, False
    t0 = np.percentile(ra[ok], 2.5) > 0
    t1 = np.percentile((ra - rb)[ok], 1.25) > 0
    return bool(t0), bool(t1)

def sp(x, y):
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0: return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])

FAMILY_ORDER = [0, 1, 2, 0, 1, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0]     # W M Wtf W M Wg Mg W M Wtf Wg Mg ...

def study(rng, w, rule, decide, n_runs=6, sd_walk=0.09, rho_a=0.75, rho_b=0.2, fam=0.0):
    units = [Sim(rng, 0.02, 0.65, sd_walk)]            # REL: two existing policies
    units[0].need(2)
    fam_of = [-1]
    fam_d = {f: rng.uniform(np.log2(2.5), np.log2(2560)) for f in range(5)}
    fam_m = {f: 0.62 + rng.normal(0, 0.05) for f in range(5)}
    for r in range(n_runs):
        f = FAMILY_ORDER[r]
        if fam > 0:        # a run's boundary lies near its family's; so does its walking level
            dstar = 2.0 ** float(np.clip(fam_d[f] + rng.normal(0, 0.7), np.log2(2.5), np.log2(2560)))
            m = float(np.clip(fam_m[f] + rng.normal(0, 0.02), 0.35, 0.8))
        else:
            dstar = 2.0 ** rng.uniform(np.log2(2.5), np.log2(2560))
            m = float(np.clip(0.62 + rng.normal(0, 0.05), 0.35, 0.8))
        sims = run_search(rng, dstar, w, m, sd_walk, decide)
        units += list(sims.values()); fam_of += [f] * len(sims)
    mu = np.array([u.mu for u in units])
    true_failed = np.array([u.q > 0.5 for u in units])
    meas, lab, k = [], [], []
    for i, u in enumerate(units):
        if i == 0: a, b, c = float(np.mean(u.scores[:2])), False, 0
        else: a, b, c = replicate(u, rule)
        meas.append(a); lab.append(b); k.append(c)
    meas = np.array(meas); lab = np.array(lab)
    # two checks with fixed relation to the TRUE unit scores
    z = (rankdata(mu) - (len(mu) + 1) / 2) / len(mu) * 3.46          # standardised rank of the truth
    def noise():
        e = rng.normal(size=len(mu))
        if fam > 0:
            fe = {f: rng.normal() for f in range(-1, 5)}
            e = np.sqrt(fam) * np.array([fe[f] for f in fam_of]) + np.sqrt(1 - fam) * e
        return e
    A = rho_a * z + np.sqrt(1 - rho_a ** 2) * noise()
    B = rho_b * z + np.sqrt(1 - rho_b ** 2) * noise()
    t0, t1 = boot_tests(rng, A, B, meas)
    o0, o1 = boot_tests(rng, A, B, mu)
    mixed = np.array([0.2 < u.q < 0.8 for u in units[1:]])
    ff = [kk for u, kk in zip(units[1:], k[1:]) if u.scores[0] < T_FAIL]
    return dict(k_failed=float(np.mean(ff)) if ff else np.nan, policies=sum(k), track=sp(meas, mu), label_err=float((lab[1:] != true_failed[1:]).mean()),
                t0=t0, t1=t1, o0=o0, o1=o1, n_failed=int(true_failed.sum()), mixed=float(mixed.mean()),
                disagree=float(np.mean([ (u.scores[0] < T_FAIL) != (u.scores[1] < T_FAIL) for u in units[1:] if len(u.scores) >= 2])) if any(len(u.scores) >= 2 for u in units[1:]) else np.nan)

if __name__ == "__main__":
    n_st = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    fam = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    out = {}
    conds = [("2nf", "first", 6), ("2", "first", 6), ("2+1", "maj", 6), ("3", "maj", 6),
             ("1", "first", 9), ("2nf", "first", 9), ("2", "first", 9), ("2+1", "maj", 9), ("3", "maj", 9),
             ("1", "first", 12), ("2nf", "first", 12), ("2", "first", 12), ("1", "first", 18)]
    print(f"families share {fam:.0%} of the checks' noise" if fam else "runs independent")
    for (ra, rb), ename in (((0.7, -0.4), "large effect"), ((0.75, 0.2), "moderate effect")):
        for w, wname in ((0.4, "sharp boundary"), (1.5, "gradual boundary")):
            print(f"\n== {ename}; {wname}")
            print(f"{'rule':5s} {'runs':>4s} {'sims':>4s} {'policies':>8s} {'hours':>6s} {'track':>6s} {'label err':>9s} {'P(A>0)':>7s} {'P(A>B)':>7s}")
            for rule, decide, nr in conds:
                rng = np.random.default_rng(2027)
                R = [study(rng, w, rule, decide, n_runs=nr, rho_a=ra, rho_b=rb, fam=fam) for _ in range(n_st)]
                g = lambda key: float(np.nanmean([r[key] for r in R]))
                row = dict(policies=g("policies"), hours=g("policies") * HOURS_PER_POLICY / 2, track=g("track"), label_err=g("label_err"),
                           t0=g("t0"), t1=g("t1"), o0=g("o0"), o1=g("o1"), disagree=g("disagree"))
                out[f"{ename}|{wname}|{rule}|{nr}"] = row
                print(f"{rule:5s} {nr:4d} {3*nr:4d} {row['policies']:8.1f} {row['hours']:6.1f} {row['track']:6.3f} {row['label_err']:9.3f} {row['t0']:7.2f} {row['t1']:7.2f}   (no seed noise: {row['o0']:.2f}, {row['o1']:.2f})")
    json.dump(out, open(f"design_out_fam{int(fam*100)}.json", "w"), indent=1)

# copied unchanged from ../NGCAPS/scripts/ranks.py, sha256 516678e6cef51f4fad1c0c378a0d14f292aa72613cd49aeda1effc3461d20be7
# copied from ../ICMAI/scripts/ranks.py, sha256 7cf40fb66166f779c098912e6befe59a263c69f33098b4f03359d12324d2e57f
# vendored from ICPAIAS/scripts/ranks.py, sha256 0624d356d18b7d79352ad33098d9af5161b3670711228819568a1036f420ea0b
"""
Average ranks (ties share the mean of their positions), numpy only, so the analysis needs no scipy.
Identical to scipy.stats.rankdata(x, method="average") for 1-D input.
"""
import numpy as np


def rankdata(x):
    x = np.asarray(x).ravel()
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    # boundaries of runs of equal values in the sorted array
    new = np.concatenate(([True], xs[1:] != xs[:-1]))
    start = np.flatnonzero(new)
    end = np.concatenate((start[1:], [len(xs)]))
    avg = (start + end + 1) / 2.0            # mean of 1-based positions start+1 .. end
    ranks_sorted = np.repeat(avg, end - start)
    out = np.empty(len(x), dtype=float)
    out[order] = ranks_sorted
    return out


def spearman(a, b):
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


if __name__ == "__main__":
    try:
        from scipy.stats import rankdata as sp_rank
    except ImportError:
        raise SystemExit("scipy not installed: self-test skipped (the analysis itself does not need scipy)")
    rng = np.random.default_rng(0)
    for _ in range(200):
        v = rng.integers(0, 20, rng.integers(1, 300)).astype(float)
        assert np.array_equal(rankdata(v), sp_rank(v)), "mismatch with scipy"
    w = rng.normal(size=10000)
    assert np.array_equal(rankdata(w), sp_rank(w))
    print("ranks.rankdata matches scipy.stats.rankdata on 201 test vectors")

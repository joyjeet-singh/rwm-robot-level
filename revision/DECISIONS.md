# Decisions

## R0
- poppler: `brew install poppler` was running ~40 min compiling Python 3.15 from source on Intel/macOS 13; stopped it and used the pypdf route the plan allows when brew is unavailable. Conservative: no further system changes. Author may install poppler later; the scripts then use it automatically.
- Git identity: left as the existing global one (the plan sets neutral values only when none exists).

## R1
- λ1 from `log1` cannot reproduce `model_table` lam1 bit for bit: the cloud run summed in torch (`lyap.py:98`), and numpy, torch 2.2 and fsum all differ in the last bit for some models (max 2.2e-16). The script asserts |Δ| < 1e-12 and asserts that every rank-based registered number is identical; the registered numbers themselves are asserted exactly from `model_table.json` through stats.py's own functions.
- Environment: `./py` puts `.pylib` (numpy 2.2.6, pulled in by R0's scipy install) ahead of the venv's numpy 1.26.4. torch 2.2.2 then cannot convert numpy arrays ("Numpy is not available"). R0_env.md records numpy 1.26.4. R1 needed no torch: `common.arena` is called with `C.tensors` stubbed for the episode list. Not changed (outside R1's block). **Any session that trains or scores models (R3) will fail under `./py` until this is resolved.** Options: remove `.pylib/numpy*` (scipy 1.13.1 runs on numpy 1.26), or reinstall scipy into `.pylib` with `--no-deps`. Recommendation: remove `.pylib/numpy` and `numpy-2.2.6.dist-info`, then rerun R0's smoke check.
- Registered draws for Holm are taken by wrapping `common.ci` while stats.py's `boot_diff`/`boot_diff_within` run, so that no loop is copied.
- A2a (reviewer SHOULD FIX): each F18 subset resamples its own clusters (all 57, trained 57, compressed 5, RC 48), so a subset never has zero clusters. With 5 clusters, the compressed intervals are unreliable.

# C0 facts (2026-11-09)

**D0, recorded data** (`state_action_data_0.csv`, 10,000 rows): 0 rows with a fall (column 65); 1 row with any thigh contact (columns 57–60).

**Lite code** (`robotic_world_model_lite/scripts/`)
- Robots 8,192 (`configs/base_cfg.py:19`); Step 2 used 4,096. Iterations 500 (`anymal_d_flat_cfg.py:109`); 24 steps per robot per iteration (`base_cfg.py:147`).
- Imagined episode: at most 256 steps (`base_cfg.py:20`).
- Commands: each axis uniform in −1 to 1; 2% of robots get zero (`envs/anymal_d_flat.py:23-28`). New command every 100–120 steps, per robot (`anymal_d_flat_cfg.py:31`; `envs/base.py:187`).
- Uncertainty weight `uncertainty_penalty_weight` = −1.0 (`anymal_d_flat_cfg.py:30`), added per step × dt (`base.py:166`). Uncertainty is the members' standard deviation, summed over state dimensions.
- A fall ends an episode only through the predictor: `sigmoid(terminations).round()`, threshold 0.5 (`anymal_d_flat.py:94`; `base.py:173-175`). There are no state limits.
- Start states: 32-row windows of the CSV, chosen uniformly (`train.py:168-173`).
- Imagined falls are not logged; only `Train/mean_episode_length` hints at them.

**Rebuild** (`rwm_repro/src/`)
- Batch 1,024 and learning rate 1e-4, from the lite config. 2,500 iterations recorded as a known difference (`rwm_train.py:94-95`).
- Ensemble size from `build_from_config` (`rwm_model.py:424-428`); N from `forecast_horizon` (8).
- Loader asserts `data[:, 65] == 0` everywhere (`rwm_data.py:223`) and shape (10000, 66) (`:243`). C1's copy must drop both.

**Step 2 route** (`train_policy.py --route rebuilt`): `sim_row` accepts `A5_s0` (ensemble 5); `build_sim` checks the sha256; start states exclude episodes 1 and 8. It writes under `results/s2`, `runs/s2` and `runs/lite`; `diag.py` redirects these to `results/c` and `runs/c`.

**Papers** (air-bib-checker). The fetch tool gave summaries, so these quotes are **unconfirmed**; check them against the papers before citing.
- RWM-U (2504.16680): weight 1.0; 100-step imagination; 2,500 iterations; 4,096 environments; five networks (Tables S10–S11). Data 1M sim/real mixtures, best 800K sim + 200K real (Table 1); 6M listed in Table S10. Benchmark dataset sizes not found.
- RWM (2501.10100): 6M transitions, "simulation data induced by policies trained for similar tasks under varied dynamics".

**Differences from RWM-U to keep in mind:** the lite code's imagined episode is 256 steps, against RWM-U's reported 100.

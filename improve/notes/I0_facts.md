# I0 facts (sources read through a page summariser: re-check quotes against the raw HTML before printing them)
## RWM, arXiv 2501.10100 (latest v5, 14 Dec 2025; v4 read too; no difference found in these items)
URLs: https://arxiv.org/abs/2501.10100, /html/2501.10100v4, /html/2501.10100v5. Pages read to ~char 100000 of ~100400.
- Table S11 (MBPO-PPO): "imagination environments" 4096; "imagination steps per iteration" 100.
- Sec. 4.1: "The model is trained with history horizon M=32 and forecast horizon N=8."
- Table 1, RWM pretraining "state transitions": 6M.
- Sec. 4.3: "RWM-AR significantly outperforms its teacher-forcing counterpart (RWM-TF)." App. A.4.1 (varying N): N=1 is teacher forcing, fast, "this setting leads to poor autoregressive performance"; longer N improves long-term accuracy at higher training cost.
## RWM-U, arXiv 2504.16680 (latest v3, 8 Jan 2026)
URLs: https://arxiv.org/abs/2504.16680, /html/2504.16680v3.
- App. A.3.2 Table S11: "imagination steps per iteration" 100; Intro: "100-step episodic rollouts". T not given in main text (Alg. 1 step 6).
- Penalised: Eq. 5, r~ = r - lambda*u (u = ensemble variance, "epistemic uncertainty"), Sec. 4.2.
## iKCE, arXiv 2607.05966 (v1, 7 Jul 2026, RSS world-model workshop; only version)
URLs: https://arxiv.org/abs/2607.05966, /html/2607.05966.
- Sec. III-A: "iKCE measures how far each predicted next state departs from the kinematic extrapolation of its predecessor" (closed-form constant-velocity or -acceleration predictor); "necessary but not sufficient to certify dynamic imagination". Sec. V: "structural, not predictive, by design".
- Evaluated on one model, one task: DreamerV3 on DMC walker-walk (App. A Table II).
- Diagnoses one model (imagined vs real-physics rollouts; friction sweep). Not used to rank models; no ranking result found. So the paper's kinematic residual (same idea, used to rank simulators) is a different use, not a replication.
## Released code (../robotic_world_model_lite, ../rsl_rl_rwm)
- base_cfg.py:147 `num_steps_per_env: int = 24`; rsl_rl_rwm/config/example_config.yaml:4 `num_steps_per_env: 24  # number of steps per environment per iteration`. This is the PPO collection length per env per update, not an imagination horizon. No config sets imagination steps or envs.
- anymal_d_flat_cfg.py:68-69 history 32, forecast 8.
## Where the paper's values came from
- "24 steps" (body l.35; Fig. 1 line, make_figures.py:46): hard-coded; likely taken from num_steps_per_env. Correct: RWM imagines 100 steps per iteration with 4096 envs (RWM Table S11), same as RWM-U (100). Fig. 1's 24-step line should become 100 (and RWM and RWM-U lines coincide). The 24-step evaluation column in tables is a chosen evaluation horizon and stays, but must not be called RWM's.
- "0.13%": key budget_pct <- rwm_repro/results/paper_numbers.json c2_pct 0.133 <- results/task_c2_data_budget.json ours_as_fraction_of_reference 0.00133. Reference = RWM's pretraining data; Table 1 says 6M transitions (check this is what the reference counts). Keep; cite RWM Table 1.
- "750 times": scale_ratio <- 100/c2_pct rounded to 10 (revision_numbers.py:144); source ratio_reference_over_ours 750.84. Consistent; no correction needed, only a source citation.

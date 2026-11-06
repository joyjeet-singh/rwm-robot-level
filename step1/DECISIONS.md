# Step 1 decisions

## S1 (2026-11-06)
- **S1-1 One exponent.** All λ1 use `p=1`. lyap.lyapunov always draws 4 start columns and keeps the first, so λ1 is the same for every p. Test (iii) shows a match to the stored p=4 values within 4e-12 relative. This cuts the cost about fourfold.
- **S1-2 No observation noise.** The configured environment has `observation_noise=True` (base_cfg.py:24). The measurement must be deterministic for JVPs, and task1c used none, so the policy's observation is built without noise. Action noise is also off (actor mean).
- **S1-3 Noisy commands.** The seed is the trajectory's index within its arena (0–3), so every model on an arena sees the same sequence. Noise is added to rows START+1…399 (forecast steps 2–368); step 1 keeps the recorded command, as in the policy condition. The std is computed over all 10,000 CSV rows (including the zero rows at resets), with ddof 0.
- **S1-4 λ1_cl start directions.** These are lyap.lyapunov's draw in the (h, s) coordinates, padded with zero in the 12 previous-command coordinates. Those coordinates enter the tangent norm, which gives a finite-time offset that vanishes with length (test ii: +3e-3 /s at gain 0.3 and +3e-4 /s at 0.1, against a true value of 5 /s).
- **S1-5 Imagined return.** The 12 weighted terms of `_compute_imagination_reward_terms`, times step_dt, summed over closed-loop steps 2–368.
  - The velocity command stays fixed per recorded segment: no resampling, no interval pushes, no termination resets.
  - Air- and contact-time buffers start at zero at step 2.
  - The uncertainty penalty (weight −1; nonzero only for five-head models) is reported apart as `ret_unc`, not added. Head-flagged terminations are counted (`n_term`), not acted on.
  - The authors' `dof_acc_l2` uses observation slice 12:24, which holds the joint positions, not the velocities (anymal_d_flat.py:105). This is reproduced as written: the authors' own method is called.
- **S1-6 Divergence.** λ1_pol, λ1_noise and r̄J use the float32 rollout's divergence step. λ1_cl uses its own float64 trajectory. A non-finite λ is also set to +inf.
- **S1-7 Probe scope.** The probe times score_gru only. Weight loading and compression (`compute.build`) are not included.

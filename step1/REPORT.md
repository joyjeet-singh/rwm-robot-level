# AIR 2027 Step 1: report for the author

**What was done.** Three CPU-only studies went into the appendix (A8–A10); two body sentences changed; the
two-column check is still six pages. Studies CL and AR were registered (`prereg/PREREG_S1.md`, 2026-11-06 12:02 UTC)
before any trained simulator was driven by a policy. The job ran about 11 h, mostly training 12 new baselines. All
registered checks passed. Scope: one robot, one trotting gait, simulated data;
no policy was tested in a physics simulator.

**Registered verdicts.**
- **CL1** (kinematic residual, released policy's joint commands): **WORSE WITH POLICY COMMANDS**, −0.120
  [−0.220, −0.052]; its ρ with the 100-step error fell from 0.95 to 0.83.
- **CL2** (growth rate): **NOT SETTLED**, −0.032 [−0.113, +0.041].
- **AR1** (growth rate against one-step error, MLP/RSSM/Transformer): **LABEL-FREE BETTER**, +0.796 [+0.368, +1.025].
  It rests on the MLP and Transformer, not the RSSM.
- **AR2** (residual against 8-step error): **NOT SETTLED**.

**Secondary readings** (95%, uncorrected). On the five-split pool the residual again ranked worse. Noisy recorded
commands changed neither check. The closed-loop
growth rate did not rank simulators (ρ ≈ 0). Less accurate simulators gave the policy a lower imagined return
(ρ = −0.86). No rollout diverged. The one-step/100-step reversal of teacher forcing held in 5 of 5 seed pairs for the
MLP and Transformer, 0 of 5 for the RSSM.

**Training length** (descriptive). Loss was still falling in 6 of 6 runs at 2,500 iterations and 5 of 6 at 10,000
(slopes about ten times smaller). After longer training the 100-step error was lower for both methods. At both lengths,
in all three seeds, teacher forcing stayed better at one step and worse at 100 steps.

**Lite policy training** (not in the paper). The authors' lite code trains a policy inside our simulators on this Mac
with no install (two shims, a wandb stub). Projected hours for 500 iterations, timed while the job used two cores:
0.4–0.5 h at 1,024 robots, 1.6–1.9 h at 4,096, about 3.7 h at 8,192 (extrapolated).

**Step 2 recommendation:** train policies on the CPU, about 2 h each at 4,096 robots (ten policies ≈ 20 h of jobs).
A free notebook GPU pays off only for many policies. Use 5-head rebuilds and
`--holdout` (`step1/notes/S4_lite.md`). The physics-simulator test itself was not timed; if it needs Isaac Lab, it
needs an NVIDIA GPU. That is the decision Step 2 waits on.

**Please check.**
1. Appendix A8–A10 and the two changed body sentences.
2. The supplement is 69.6 MB (slim copy 1.4 MB, no raw arrays); check CMT's size limit (`SUBMIT.md`).
3. The paper gives the registration commit's short hash and time: keep this repository private until decisions.
4. `figures/fig_horizon.*` differ from the last commit, as before Step 1; never staged. Commit or discard them.
5. Declined review points and reasons: `step1/DECISIONS.md`, "S8 declined".

**Optional action.** Ask the RWM authors for the policy and settings that produced the released data. A policy-driven
rollout now has no recorded future; with those, the closed-loop checks could be tested against ground truth.

# C1 root cause: what D0–D4 showed

Descriptive. The world is a hand-built calibrated MuJoCo ANYmal D, not Isaac Lab or hardware. Source: `results/c_diag.txt`, `results/c_diag.json`.

**What the diagnosis showed**
- **D0.** The ten recorded episodes hold no fall (0 of 10,000 rows) and one row with thigh contact.
- **D1.**
  - The single-network simulators (`A_s0`, `B_s0`, `A_s0__u50`) almost never imagine a fall (at most 0.2% of 512 episodes with sampled actions, 1.0% with mean actions), yet those policies fell in all 80 MuJoCo episodes.
  - Given the true history, they predicted real falls within 0.2 s in 0–12% of cases.
  - The released five-network model (`REL`) also missed all 8 full-window falls of its policy, which fell in 10% of episodes.
- **D2.** Failed policies' imagined state–action pairs lie 6.6–7.9 times further from the training data than the authors' policy's MuJoCo pairs. For `REL` the ratio is 1.9.
- **D3, the penalty alone** (five networks, same 8,000 transitions, no falls).
  - Seed 0 did not fall (0 of 80) but did not walk: speed error 0.80 m/s, return −57.8.
  - Seed 1 fell in all 80 episodes.
  - **The penalty alone did not fix it.**
  - In imagination, `A5_s0` predicted a fall in every episode by about step 17. The uncertainty term (−0.049 per step) was far larger than any tracking reward.
- **D4.** Only 12.8% of training commands lie within 0.2 of a recorded command on every axis.

**Causes supported**
1. The data show no failures, so the fall predictor has nothing to learn from (D0, D1).
2. Imagination lets policies leave the data (D2), and few commands are covered (D4).
3. Step 2's single-network design had no penalty. This was a design mistake, but fixing it alone was not enough.

**Causes weakened**
- "The penalty was all that was missing" (D3).

**Unexplained**
- Why `A5_s0`'s fall predictor fires in nearly every imagined episode, although it was trained only on all-zero fall labels and the single-network models never fire. D3's imagined episodes were cut to about 17 steps by it.

**Open until D5 and D6**
- D5: does the penalty matter once data are plentiful?
- D6: does expert-only MuJoCo data at 8,000 transitions reproduce the failure, separating data source (D3 against D6) from falls and variety (D6 against the 10k rung)?

The plan continues unchanged.

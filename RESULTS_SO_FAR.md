# AIR 2027 — results so far (14 November 2026)

**What this is.** One place that lists every simulator, model and policy the project has, and every result measured with them, with the file each number comes from. Nothing here is new work. It is the record to write the paper from, whatever Option D adds.

**Status labels.** Each result carries one of three:
- **In the paper:** the paper already prints it, through its number keys.
- **On the Mac:** measured on your Mac and stored in result files, but not yet used by the paper.
- **Cloud only:** measured in the review, on a copy of the project. The paper may not use such a number until it is re-measured on the Mac. Option D's first job does that.

## 1. Where the paper stands

- **Title:** "One Step Ahead Is Not Enough: Testing a Learned Quadruped Simulator at the Horizon It Is Used".
- **Build of 8 November:** six two-column pages with no spare room; abstract 198 words; every check passes, including 181 claim assertions (`results/slack.json`, `results/claims_check.json`).
- **What it claims** (`paper/ABSTRACT.txt`):
  1. One-step checks misjudge training and compression choices (descriptive).
  2. Fine-tuning restores a pruned simulator's skill (registered).
  3. Checks that need no recorded future only partly replace checks several steps ahead (registered).
  4. In a hand-built physics model, whether checks predict policies' return stayed unsettled (registered; nine versions of one simulator).
- **The submission window** is 15 November to 31 December 2026.

## 2. What we have

| Group | How many | Made in | Status |
|---|---|---|---|
| Released model (`REL`), RWM's own | 1 | by RWM's authors | In the paper |
| GRU simulators, main pool | 97: 57 trained, 40 compressed copies | original study | In the paper |
| GRU simulators, five-split pool | 255: 55 trained, 200 compressed copies | original study | In the paper |
| Fine-tuned pruned copies | 90 models from 30 runs | study FT | In the paper |
| MLP, RSSM and Transformer simulators | 38, of which 12 newly trained | Step 1 | In the paper |
| MuJoCo model of ANYmal D, hand-built and calibrated | 1 | Step 2 | In the paper |
| Policies trained in `REL` and its 8 compressed copies | 18 registered, 1 pilot | Step 2 | In the paper |
| Policies trained in rebuilt single-network simulators | 3 pilot | Step 2 | In the paper |
| MuJoCo data | 6 million transitions in four nested sizes, and two test sets | Option C | On the Mac |
| Simulators trained from scratch on MuJoCo data | 4 | Option C | On the Mac |
| Policies trained in those simulators, and in `A5_s0` | 4 runs, and 2 | Option C | On the Mac |
| `REL` adapted to MuJoCo data | 2 | the review | Cloud only |
| Policies trained in `REL` and in an adapted copy | 1, and 2 | the review | Cloud only |

Eleven trained GRU simulators are in both pools, so the paper counts 101 trained simulators.

## 3. Registered results (all in the paper)

"Check A vs check B" is the difference between their rank correlations with the target. Positive favours A.

| Study | Question | Estimate | Interval | Verdict |
|---|---|---|---|---|
| Original | Q1: growth rate vs one-step error, ranking 100-step error (97 simulators) | +0.48 | 97.5% [+0.20, +0.79] | growth rate better |
| Original | Q2: growth rate vs 8-step error | −0.04 | 97.5% [−0.20, +0.14] | not settled |
| Original | Q3: the same on the five-split pool (255) | −0.12 and −0.28 | 95% [−0.22, −0.03] and [−0.37, −0.22] | replication failed |
| Original | Q4: kinematic residual vs one-step and vs 8-step error | +0.63 and +0.12 | 95% [+0.41, +0.88] and [+0.02, +0.23] | residual better (secondary) |
| Original | Q5: Q1 and Q2 without the one-step-trained models | +0.27 and +0.01 | 95% [+0.04, +0.51] and [−0.16, +0.20] | better; not settled |
| Original | Q6: does 30% pruning raise the growth rate? | +0.03 | 95% [−0.02, +0.07] | not settled |
| FT | FT1: skill regained at 100 steps by fine-tuning a 30%-pruned copy | 0.64 | 97.5% [0.54, 0.76] | fine-tuning restores skill |
| FT | FT2: the fine-tuned copy's skill against a control | 1.00 | 97.5% [0.995, 1.004] | nearly as skilled |
| Step 1 | CL1: kinematic residual with the policy's own joint commands | −0.12 | 97.5% [−0.22, −0.05] | worse with policy commands |
| Step 1 | CL2: growth rate with the policy's own joint commands | −0.03 | 97.5% [−0.11, +0.04] | not settled |
| Step 1 | AR1: growth rate vs one-step error on MLP, RSSM, Transformer | +0.80 | 97.5% [+0.37, +1.03] | growth rate better |
| Step 1 | AR2: kinematic residual vs 8-step error, same simulators | +0.11 | 97.5% [−0.08, +0.36] | not settled |
| Step 2 | R0: does 100-step error predict physics return? (9 simulators) | +0.23 | 95% [−0.87, +0.81] | not settled |
| Step 2 | R1: growth rate vs one-step error, predicting physics return | −0.68 | 97.5% [−1.50, 0.00] | not settled |
| Step 2 | R2: kinematic residual vs 8-step error, predicting physics return | −0.05 | 97.5% [−0.47, 0.00] | not settled |

Sources: `results/stats_report.txt`, `results/ft_report.txt`, `results/s1_report.txt`, `results/s2_stats.txt`.

One Step 1 reading bears on the physics study: less accurate simulators gave the policy a lower imagined return (rank correlation −0.86 with 100-step error, 95% [−0.91, −0.77]; secondary).

## 4. The physics study so far (Step 2; in the paper)

**The MuJoCo model passed its gate.** RWM's released policy fell in 1.25% of 80 episodes. Two feet were down 78.9% of the time, against 84.4% recorded. Speed-tracking error was 0.069 m/s, against 0.072 recorded (`results/s2_fidelity.txt`).

**The pilot** (one policy each, 500 iterations; `results/s2_pilot.txt`):

| Policy trained in | Falls | Physics return |
|---|---|---|
| `REL` (five networks) | 10% | 18.6 |
| `A_s0` (rebuilt, one network) | 100% | −10.4 |
| `B_s0` (rebuilt, one network) | 100% | −4.0 |
| `A_s0__u50` (pruned copy of `A_s0`) | 100% | −6.6 |
| RWM's released policy, for reference | 1.3% | 26.3 |

**The registered pool:** `REL` and eight compressed copies, two policies each (`results/s2_stats.txt`). Checks are on the paper's recorded data.

| Simulator | Physics return | Falls | E1 | E8 | E100 | Growth rate | Kinematic residual |
|---|---|---|---|---|---|---|---|
| `REL` | 16.8 | 37% | 0.083 | 0.093 | 0.40 | 0.70 | 3.3 |
| `REL__fp16` | 15.2 | 39% | 0.083 | 0.093 | 0.40 | 0.61 | 3.3 |
| `REL__u10` | 9.4 | 38% | 0.134 | 0.155 | 0.52 | 0.78 | 3.5 |
| `REL__u20` | 3.9 | 19% | 0.219 | 0.298 | 1.32 | 0.80 | 7.0 |
| `REL__int8w` | 2.7 | 52% | 0.286 | 0.408 | 1.21 | 0.96 | 7.3 |
| `REL__u30` | −27.8 | 54% | 0.298 | 0.487 | 1.05 | 0.23 | 7.0 |
| `REL__s192` | −60.8 | 56% | 0.254 | 0.524 | 1.12 | 0.19 | 7.1 |
| `REL__u50` | 9.9 | 0% | 0.370 | 0.958 | 7.56 | 1.80 | 363 |
| `REL__s128` | 9.7 | 0% | 0.326 | 0.883 | 5.17 | 1.28 | 187 |

- The two most damaged copies gave policies that stand still. They never fall, so the physics return ranks them third and fourth.
- Policy seeds can disagree: `REL__u30`'s first policy fell in every episode and its second in 9%.
- The growth rate picked badly here: its top three simulators were `REL__s192`, `REL__u30` and `REL__fp16`.

## 5. Option C (on the Mac; not in the paper)

**Why Step 2's rebuilt simulators failed** (`results/c_diag.txt`, `optc/notes/C1_rootcause.md`):
- The ten recorded episodes hold no fall, so a simulator trained on them cannot learn what a fall looks like.
- The single-network simulators almost never imagine a fall, yet their policies fell in all 80 MuJoCo episodes.
- The failed policies' imagined states lie 6.6 to 7.9 times further from the training data than the released policy's real states. For `REL`'s policy the ratio is 1.9.
- The uncertainty penalty alone did not fix it. Two policies trained in `A5_s0` (five networks, same data) did not walk: one stood (0% falls, speed error 0.80 m/s), one fell in every episode. In imagination `A5_s0` predicted a fall in every episode by about step 17.

**The data.** 6 million MuJoCo transitions: 55% from RWM's released policy (clean and with light noise), 25% from Step 2's 22 trained policies, 10% from heavy noise and 10% from random motion; half the episodes with varied physics. Nested sizes of 10 thousand, 100 thousand, 1 million and 6 million, and two held-out test sets (`optc/data_manifest.json`).

**The four simulators,** trained from scratch, checked on held-out MuJoCo data. Errors here are in each simulator's own units, so compare them only with the "hold the last state" floor (`results/c/pilot_sims.json`, `pilot_sims_2S.json`).

| Simulator | Data | Training steps | E8 (floor) | E100 (floor) | Kinematic residual (true data) | Growth rate |
|---|---|---|---|---|---|---|
| `sim_6M_N8_s0` | 6M | 2,572 | 0.097 (0.39) | 0.200 (0.51) | 9.1 (0.59) | 0.18 |
| `sim_6M_N8_s0__2S` | 6M | 5,144 | 0.079 (0.39) | 0.148 (0.51) | 2.1 (0.59) | 0.18 |
| `sim_10k_N8_s0` | 10k | 2,572 | 0.153 (0.45) | 0.287 (0.59) | 13.4 (0.39) | 0.13 |
| `sim_10k_N8_s0__2S` | 10k | 5,144 | 0.148 (0.45) | 0.258 (0.59) | 13.6 (0.39) | 0.24 |

- **Gate A passed:** on held-out data every error is far below the floor (`results/c/gate_a.json`).
- **Gate A2 failed, twice.** The gate runs RWM's released policy inside the simulator (`results/c/calibration.json`, `calibration_2S.json`, `results/c_pilot.txt`).

| Simulator | Released policy falls in imagination | Tracking, as a share of MuJoCo's | Disagreement between its networks |
|---|---|---|---|
| `REL` | 0% | 87% | 0.46 |
| `sim_6M_N8_s0` | 15.5% | 31% | 1.54 |
| `sim_6M_N8_s0__2S` | 26.6% | 49% | 1.01 |
| `sim_10k_N8_s0` | 42.6% | 35% | 2.05 |
| `sim_10k_N8_s0__2S` | 62.9% | 54% | 1.71 |
| `A5_s0` | 2.9% | 43% | 1.87 |

The gate's limits were at most 10% falls and at least 80% of the tracking.

- **The first pilot failed** (`results/c_pilot1.txt`). The walking score is 0.942 for RWM's released policy and 0.069 for standing still.

| Policy trained in | Walking score at 500 / 1,000 / 1,500 / 2,000 iterations | Falls | What happened in imagination |
|---|---|---|---|
| `sim_6M_N8_s0` | 0.017 / 0.002 / 0.001 / 0.000 | 85% to 100% | imagined episodes shrank from 228 steps to 1 |
| `sim_10k_N8_s0` | 0.011 / 0.004 / 0.054 / 0.008 | 26% to 100% | imagined episodes lasted 1 to 3 steps from the start |

So the policies learned to end the imagined episode at once: the penalty for the networks' disagreement outweighed any reward for walking.

## 6. The review (cloud only; details in `optd/REVIEW.md`)

- **Why Gate A2 failed.** In the from-scratch simulators the imagined robot answers a new velocity command too slowly, leans and drifts. Two design mistakes of the plan caused it: the scaling of the states, and far too little training for this Mac.
- **The released model matches MuJoCo** for RWM's released policy: 0% falls against 1%, and tracking 0.0254 per step against 0.0251.
- **Adapting the released model** to MuJoCo walking data for 160 steps halved its error on held-out MuJoCo data (one-step error 0.120 to 0.061).
- **A policy trained in the adapted model falls, with every seed tried.** Same code and settings for all four policies; only the simulator and the seed differ:

| Policy trained in | Walking score in MuJoCo at 300 / 400 / 500 iterations | Falls at 500 |
|---|---|---|
| released model, seed 0 | 0.25 / 0.57 / 0.64 | 44% |
| adapted model, seed 0 | 0.04 / 0.07 / 0.10 | 86% |
| adapted model, seed 1 | 0.04 / 0.05 / 0.06 | 97% |
| adapted model, seed 2 | 0.10 / 0.09 / 0.12 | 88% |

- **Inside their own simulators the two policies look equally good** (tracking 0.0260 and 0.0257 per step).
- **Why.** Fed the failing policy's real joint commands, both simulators predict about 8° of lean after half a second. In MuJoCo the robot leans 20°.
- **A hint about the checks,** from this one pair. Short-horizon errors rate the adapted model better than the released one (E8 0.081 against 0.166). The paper's long-horizon checks rate it slightly worse (368-step error 0.86 against 0.71; kinematic residual 2.45 against 2.19; growth rate 0.76 against 0.74).
- **Walking scores of all 22 of Step 2's policies,** re-measured with the walking score. For each of the nine simulators, its two policies agree on failed or not (a score below 0.15). Three simulators walk: `REL` 0.60 and 0.57 (its seed-0 pilot 0.77), `REL__fp16` 0.60 and 0.62, `REL__u10` 0.25 and 0.53. Six fail, every policy below 0.07. The three rebuilt pilots score 0.01 or less. `optd/REVIEW.md` section 13 has the table.

Every number in this section was measured on a cloud copy. Each policy result is one seed, except the adapted model's (three seeds) and Step 2's (two each).

## 7. Every simulator that has had a policy trained in it

| Simulator | What it is | How its policies did in MuJoCo | Status |
|---|---|---|---|
| `REL` | released model | walk: scores 0.57–0.77; falls 8–43% depending on the protocol | In the paper (returns); cloud only (scores) |
| `REL__fp16`, `REL__u10` | lightly compressed copies | walk, a little worse than `REL` | In the paper |
| `REL__u20`, `REL__int8w` | compressed copies | poor walking | In the paper |
| `REL__u30`, `REL__s192` | compressed copies | mostly fall; seeds disagree on how often, but by walking score both seeds fail | In the paper (falls); cloud only (scores) |
| `REL__u50`, `REL__s128` | heavily compressed copies | stand still | In the paper |
| `A_s0`, `B_s0`, `A_s0__u50` | rebuilt, one network, 8 thousand recorded transitions | fall in every episode | In the paper |
| `A5_s0` | rebuilt, five networks, same data | one stands, one falls | On the Mac |
| `sim_6M_N8_s0`, `sim_10k_N8_s0` | from scratch on MuJoCo data | fall; the policy ends imagined episodes at once | On the Mac |
| `sim_6M_N8_s0__2S`, `sim_10k_N8_s0__2S` | the same, trained twice as long | no policy trained: stopped at Gate A2 | On the Mac |
| adapted model (160 steps, walking data) | `REL` with its training continued | falls with each of three seeds: scores 0.10, 0.06, 0.12 | Cloud only |
| adapted model (320 steps, mixed data) | the same, on mixed data | no policy trained | Cloud only |

**The pattern so far.** Only the released model and its lightest copies have given walking policies. Every simulator we trained or adapted ourselves has given policies that fall or stand, including the ones whose prediction errors are lower than the released model's.

## 8. What the paper can use now, and what needs the Mac first

- **Ready now** (result files exist on the Mac): Option C's data, its four simulators, Gate A, the two failures of Gate A2, the first pilot, and the root cause of Step 2's rebuilt simulators. They need number keys and a short appendix section; Option D's paper session writes both.
- **Needs the Mac first:** everything in section 6. Option D's first job re-measures the walking scores of Step 2's 18 policies. Its second job re-measures the adaptation result, with two policy seeds per simulator and a third when those two disagree.
- **Not to be claimed** until then: that adapting the released model makes its policies worse, and that any check foresaw it.

## 9. File map

| What | File |
|---|---|
| Original study | `results/stats.json`, `results/stats_report.txt`, `results/model_table.json` |
| Study FT | `results/ft_stats.json`, `results/ft_report.txt` |
| Step 1 | `results/s1_stats.json`, `results/s1_report.txt`, `step1/REPORT.md` |
| Step 2 | `results/s2_stats.json` and `.txt`, `results/s2_main.json`, `results/s2_pilot.txt`, `results/s2_fidelity.txt`, `step2/REPORT.md` |
| Step 2's policies | `runs/s2/<policy>/actor.pt`; checkpoints every 50 iterations in `runs/lite/s2_<policy>/` |
| Option C | `results/c_diag.txt`, `results/c_pilot1.txt`, `results/c_pilot.txt`, `results/c_authors.json`, `results/c/` |
| Option C's simulators and data | `runs/c/`, `optc/data/` |
| The review | `optd/REVIEW.md`, `optd/review/cloud_tests/` |
| Every number the paper prints | `results/paper_numbers.json` |

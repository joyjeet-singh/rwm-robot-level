# Review behind Option D (11–14 November 2026)

**What this is.** Option C's pilot job stopped at Gate A2 twice. Before rewriting the plan, the files were copied to a cloud workspace and tested there. This file lists what each test measured.

**Status of these numbers.** None was measured on the Mac, and none is part of the study. D0 and the preparation job re-measure everything the plan relies on. The registration lists this file as known beforehand.

**The copy is faithful.**
- The authors' policy scores 0.9422 with 0% falls on the walking protocol; the Mac's `results/c_authors.json` says 0.9421.
- The released model's walk test gives tracking 0.0254 per step and disagreement 0.461; the Mac's `results/c/calibration.json` says 0.0254 and 0.464.
- The cloud copy runs torch 2.15 and MuJoCo 4.0.0, not the Mac's versions.

Scripts are in `optd/review/cloud_tests/`. "Common units" means errors divided by the authors' scales.

## 1. The walk test, simulator by simulator
The authors' policy, noise-free, in each simulator's imagination (512 robots, 256 steps, the lite code's commands, pushes and observation noise). MuJoCo is the truth, measured under the same conditions with 96 robots.

| Simulator | Predicted falls | Tracking per step | Median tilt |
|---|---|---|---|
| MuJoCo (truth) | 1% | 0.0251 | 1.6° |
| released model (`REL`) | 0% | 0.0254 | 1.4° |
| `REL`, started from MuJoCo states | 0% | 0.0254 | 1.5° |
| `A5_s0` | 2% | 0.0124 | about 2° |
| `sim_6M_N8_s0` | 12% | 0.0092 | 11.7° |
| `sim_6M_N8_s0__2S` | 26% | 0.0139 | 7.6° |
| `sim_10k_N8_s0__2S` | 56% | 0.0156 | about 4°, then tipping |

Perfect tracking earns 0.03 per step.

## 2. What does not cause the failure
`sim_6M_N8_s0__2S`, same test with parts switched off:

| Condition | Predicted falls | Tracking per step |
|---|---|---|
| as in policy training | 26% | 0.0139 |
| no pushes | 24% | 0.0145 |
| no observation noise | 24% | 0.0138 |
| neither | 23% | 0.0144 |
| mean of the five networks instead of one | 22% | 0.0147 |

The median robot tracks as badly as the average one (0.0140 against 0.0139), so a few diverging robots do not explain it.

## 3. What goes wrong: a slow response, then drift
Median speed error (m/s) after a new velocity command at step 0, without pushes or noise:

| Step | MuJoCo | `REL` | `sim_6M_N8_s0__2S` | `A5_s0` |
|---|---|---|---|---|
| 0 | 0.93 | 1.05 | 0.97 | 1.07 |
| 8 | 0.46 | 0.54 | 0.84 | 0.90 |
| 16 | 0.23 | 0.26 | 0.76 | 0.68 |
| 32 | 0.07 | 0.07 | 0.59 | 0.43 |
| 96 | 0.04 | 0.05 | 0.28 | 0.26 |

- **Tilt.** In `sim_6M_N8_s0__2S` the median tilt rises from 1.4° to about 7° within 24 steps and stays there. After the second command change it reaches 15°, and 41° for the worst tenth.
- **The fall predictor is unreliable.**
  - In `sim_6M_N8_s0__2S`, 54% of predicted falls happen while the robot is upright (tilt under 20°).
  - In `sim_6M_N8_s0`, 74% of the robots not predicted to fall lean past 45° at some point.
  - `A5_s0` predicts few falls, and every one of them with the robot upright.
- **`A5_s0`** keeps the robot level but does not follow turn commands: its median turn error stays near 0.45 rad/s.

## 4. Scaling: the authors' fixed scales against our data's own
Option C's trainer divided each state by the spread of its own rung (`optc/data/rung_6M.stats.json`). The authors divide by fixed scales. The loss is a sum of squared scaled errors, so a state's weight goes as one over its scale squared.

| State | Authors' scale | 6M data's spread | Weight in our loss, relative to the authors' |
|---|---|---|---|
| vertical speed | 0.10 | 0.153 | 1/2.3 |
| roll rate | 0.30 | 0.795 | 1/7 |
| gravity direction, x (pitch) | 0.02 | 0.084 | 1/18 |
| gravity direction, y (roll) | 0.02 | 0.218 | 1/118 |
| gravity direction, z | 0.04 | 0.212 | 1/28 |
| knee angle (LF) | 0.15 | 0.678 | 1/20 |

For expert-only data (`rung_d6`) the spread of the z component is 0.0004, so its own-data scale would blow that state up a hundredfold.

## 5. Training amount
- **The authors' loop** (`model_training.py`) makes a full pass over the data in every iteration; the default is 500 iterations.
- **Ours:** 2,572 steps × 256 windows is 0.12 of a pass over the 5.58 million windows of the 6M rung; 5,144 steps is 0.24 of a pass.
- **Still improving when it stopped:** the state loss was 6.9 at step 2,500 and 5.0 at step 5,000.
- **Cost on the Mac:** 5.3 s per step, so 7.6 hours for 5,144 steps.

## 6. Open-loop errors on held-out MuJoCo data, common units
180 windows from the expert test set; rollout from the mean of the five networks.

| Simulator | E1 | E8 | E24 | E100 |
|---|---|---|---|---|
| hold the last state | 0.203 | 0.608 | 0.829 | 0.782 |
| `REL` (never trained on MuJoCo) | 0.120 | 0.149 | 0.180 | 0.341 |
| `sim_6M_N8_s0__2S` | 0.087 | 0.123 | 0.152 | 0.261 |
| `sim_6M_N8_s0` | 0.103 | 0.152 | 0.187 | 0.383 |
| `sim_10k_N8_s0__2S` | 0.141 | 0.200 | 0.242 | 0.364 |
| `A5_s0` | 0.158 | 0.305 | 0.367 | 0.428 |

- `sim_6M_N8_s0__2S` beats the released model on every one of these errors, and fails the walk test that the released model passes.
- The released model's largest one-step error is in the joint torques (0.25, against 0.10 for `sim_6M_N8_s0__2S`). MuJoCo uses the fitted PD motor, not Isaac's motor network.
- On tilt over 24 steps the released model is better (0.26 against 0.34), without ever having seen MuJoCo data.

## 7. The paper's long-horizon checks, common units, expert test set
40 windows of 400 rows. Residual scales come from the test set's own true rows.

| Simulator | r̄J | r̄A | λ1 | E368 |
|---|---|---|---|---|
| true data | 0.66 | — | — | — |
| `REL` | 2.19 | 94 | +0.74 | 0.71 |
| `sim_6M_N8_s0__2S` | 2.32 | 288 | +0.22 | 1.62 |
| `sim_6M_N8_s0` | 10.3 | 873 | +0.16 | 5.88 |
| `sim_10k_N8_s0__2S` | 23.9 | 1,665 | +0.30 | 8.33 |
| `A5_s0` | 1.85 | 8.8 | −0.53 | 0.56 |

λ1 was computed on the first 12 windows only.

## 8. Fine-tuning the released model on MuJoCo data
From the released model's weights, with the authors' scales; learning rate 1e-4; batch 256; rollout windows (N = 8); everything trained. Open-loop errors are on 180 held-out windows per test set, in common units. The walk test uses 384 robots.

**Walking data only:** sources 0, 1 and 2 of Option C's 100k rung (56 thousand transitions, 54 thousand windows).

| Steps | Expert set: E1 / E8 / E24 / E100 | Mixed set: E1 / E100 | Walk test: falls | Tracking per step | Disagreement ū |
|---|---|---|---|---|---|
| 0 (released) | 0.120 / 0.149 / 0.180 / 0.341 | 0.263 / 0.567 | 0% | 0.0255 | 0.46 |
| 80 | 0.067 / 0.079 / 0.096 / 0.285 | 0.212 / 0.544 | 0.3% | 0.0249 | 1.37 |
| 160 | 0.061 / 0.072 / 0.086 / 0.270 | 0.201 / 0.508 | 0.3% | 0.0250 | 0.66 |

- The one-step torque error falls from 0.25 to 0.09.
- The training loss on the states falls from about 10.5 to 4.0.
- The 160-step model is called `FT160` below.

**Mixed data:** all of the 100k rung (95 thousand windows; 45% from weaker policies, heavy noise and random motion).

| Steps | Loss | Expert set: E1 / E8 / E24 / E100 | Mixed set: E1 / E8 / E24 / E100 | Walk test: falls | Tracking per step | Disagreement ū |
|---|---|---|---|---|---|---|
| 0 (released) | about 50 | 0.120 / 0.149 / 0.180 / 0.341 | 0.263 / 0.328 / 0.395 / 0.567 | 0% | 0.0255 | 0.46 |
| 80 | 27.4 | 0.100 / 0.129 / 0.163 / 0.523 | 0.221 / 0.266 / 0.327 / 0.715 | 0% | 0.0241 | 2.12 |
| 160 | 24.5 | 0.092 / 0.119 / 0.151 / 0.490 | 0.212 / 0.251 / 0.308 / 0.676 | 0% | 0.0245 | 2.28 |
| 240 | 22.1 | 0.086 / 0.109 / 0.137 / 0.480 | 0.207 / 0.241 / 0.293 / 0.672 | 0% | 0.0244 | 2.25 |
| 320 | 19.7 | 0.082 / 0.103 / 0.128 / 0.457 | 0.202 / 0.233 / 0.283 / 0.633 | 0% | 0.0244 | 2.28 |

- Short-horizon errors fall; the 100-step error rises and has not come back after 320 steps.
- The five networks disagree five times as much as in the released model, and that has not settled either.
- The 320-step model is called `FTmix320` below.

## 9. Policies
**Walking scores of existing policies** (Option C's 96-episode protocol):

| Policy | Score | Falls |
|---|---|---|
| authors' policy | 0.942 | 0% |
| `REL__p0` (Step 2, 500 iterations) | 0.769 | 8% |
| `REL__p1` | 0.605 | 38% |
| `REL__p2` | 0.574 | 31% |
| `REL__u50__p1` (stands still) | 0.069 | 0% |

Standing perfectly still scores 0.0686 on this protocol, so the score separates standing from walking.

**A policy trained inside `FT160`.** The authors' lite code with its own settings: 4,096 robots, uncertainty weight 1.0, seed 0. Start states were MuJoCo walking windows (source 0 of the 100k rung).

| Iterations | In imagination: speed term / turn term | In MuJoCo: walking score | Falls |
|---|---|---|---|
| 200 | 0.63 / 0.43 | 0.054 | 68% |
| 300 | 0.78 / 0.46 | — | — |
| 400 | 0.79 / 0.44 | 0.036 | 90% |
| 500 | 0.81 / 0.42 | 0.045 | 92% |

- Inside the simulator it learns as fast as Step 2's `REL__p0` did inside the released model (0.44, 0.64, 0.77 and 0.82 for the speed term at 200, 300, 400 and 499 iterations).
- A turn term of 0.44 is what a policy earns by never turning. Both policies are immature at 500 iterations.
- The run was interrupted at iteration 400 and continued from its saved state.

**The same 300-iteration policy, judged three ways** under the imagination's conditions (commands, pushes, noise):

| Judge | Falls | Tracking per step | Lean |
|---|---|---|---|
| `FT160`, the simulator it was trained in | 0% | 0.0217 | 2% of robots ever lean past 30° |
| the released model | 3% | 0.0205 | 12% ever lean past 30° |
| MuJoCo | 83% | 0.0091 | median lean 16° |

MuJoCo's fall count here is a little harsher than the simulators': the robot first walks 100 warm-up steps from standing under the policy being judged, and a fall during them counts. The simulators start from recorded walking states. The same holds for the second three-way table below. Option D's line-up test removes the difference by warming up with the authors' policy.

**Controls: the same training in the released model and in `FT160`.** One script, one set of settings and one seed (0) for both; start states are the authors' recorded data, as in Step 2. Only the simulator differs.

| Iterations | Released model, in imagination: speed / turn term | In MuJoCo: score | Falls | `FT160`, in imagination: speed / turn term | In MuJoCo: score | Falls |
|---|---|---|---|---|---|---|
| 200 | 0.28 / 0.43 | 0.075 | 52% | 0.53 / 0.43 | 0.039 | 91% |
| 300 | 0.56 / 0.44 | 0.251 | 49% | 0.84 / 0.44 | 0.037 | 97% |
| 400 | 0.78 / 0.45 | 0.574 | 35% | 0.84 / 0.46 | 0.074 | 89% |
| 500 | 0.83 / 0.46 | 0.640 | 44% | 0.83 / 0.45 | 0.100 | 86% |

- The released model's policy starts to walk between 300 and 400 iterations and reaches 0.64, in line with Step 2's three policies (0.57–0.77).
- `FT160`'s policy learns faster inside its simulator and does not walk in MuJoCo at any point.
- So the failure belongs to the fine-tuned simulator. It is not the start states (it fails with both kinds), the script (the same script gives a walking policy in the released model) or too little training.

**The two 500-iteration policies, each judged three ways** under the imagination's conditions:

| Policy trained in | Judge | Falls | Tracking per step | Lean |
|---|---|---|---|---|
| released model | the released model | 0% | 0.0260 | 0.5% of robots ever lean past 30° |
| | MuJoCo | 42% | 0.0192 | median lean 5.0° |
| `FT160` | `FT160` | 0% | 0.0257 | 0.3% ever lean past 30° |
| | the released model | 1% | 0.0248 | 5% ever lean past 30° |
| | MuJoCo | 91% | 0.0101 | median lean 21.8° |

- Every simulator is optimistic about a policy trained inside it. The released model predicts no falls for its own policy, and MuJoCo shows 42% (warm-up included).
- The gap is far larger for `FT160`.

## 10. Why that policy falls: both simulators are too optimistic about it
The 400-iteration policy was recorded in MuJoCo (64 robots, up to 300 steps; 94% fell, and half of the episodes were over within 97 steps). Each simulator was then fed the recorded joint commands from the true 32-step history, and its prediction 24 steps ahead was compared with what happened (536 windows; common units).

| Simulator, data | E1 | E8 | E24 | Lean after 24 steps: true / predicted (median) |
|---|---|---|---|---|
| released model, expert test windows | 0.120 | 0.149 | 0.180 | 1.2° / 1.6° |
| released model, this policy's windows | 0.483 | 0.833 | 1.246 | 20.5° / 8.1° |
| `FT160`, expert test windows | 0.061 | 0.072 | 0.086 | 1.2° / 1.2° |
| `FT160`, this policy's windows | 0.479 | 0.851 | 1.271 | 20.5° / 7.7° |

- In the last 48 steps before a fall the true lean reaches 47°. The released model predicts 12°, and `FT160` 13°.
- Fine-tuning on walking data halved the error on walking data and changed nothing for this policy.

**Does mixed data teach the simulator that this policy fails?** Partly. The same policy inside the mixed-data fine-tune (section 8):

| Steps | Predicted falls | Predicted tracking per step | Robots ever leaning past 30° | Open-loop E24 on its windows | Predicted lean (true 20.5°) |
|---|---|---|---|---|---|
| 0 (released) | 4% | 0.0218 | 10% | 1.246 | 8.1° |
| 80 | 8% | 0.0157 | 44% | 1.052 | 10.3° |
| 160 | 8% | 0.0157 | 50% | 1.010 | 11.0° |
| 240 | 10% | 0.0148 | 61% | 0.986 | 11.7° |
| 320 | 7% | 0.0139 | 57% | 0.980 | 11.8° |

In MuJoCo this policy falls in 90–94% of episodes. Its 300-iteration version falls in 83% and tracks 0.0091 per step.

## 11. The paper's long-horizon checks on three simulators
The paper's 400-row windows and rollout, common units. Residual scales come from the expert test set's own true rows. λ1 uses the first 12 windows. E8 and E100 here are from the 400-row windows, so they differ a little from section 6.

| Test set | Simulator | E8 | E100 | E368 | r̄J | r̄A | λ1 |
|---|---|---|---|---|---|---|---|
| expert (40 windows) | released | 0.166 | 0.387 | 0.711 | 2.19 | 94 | +0.741 |
| | `FT160` | 0.081 | 0.318 | 0.862 | 2.45 | 128 | +0.762 |
| | `FTmix320` | 0.117 | 0.564 | 1.385 | 3.76 | 255 | +0.900 |
| mixed (67 windows) | released | 0.318 | 0.551 | 0.944 | 3.09 | 95 | +0.633 |
| | `FT160` | 0.230 | 0.492 | 0.937 | 3.12 | 116 | +0.671 |
| | `FTmix320` | 0.238 | 0.671 | 1.531 | 4.63 | 361 | +1.462 |

- **Short horizons favour `FT160`:** its E8 and E100 are lower than the released model's on both test sets.
- **Long horizons do not:** on the expert set its E368, r̄J, r̄A and λ1 are all a little worse than the released model's. On the mixed set r̄J, r̄A and λ1 are a little worse, and E368 ties.
- **Its policy is far worse** (section 9).
- This is one pair of simulators and one policy seed. It is a hint, and the reason for the ladder; it is not a result.

## 12. What was not tested
- Fine-tuning on the 1M rung, beyond 320 steps, with a gentler learning rate, or teacher-forced.
- Fine-tuning for fewer than 80 steps.
- A policy trained inside a mixed-data fine-tune.
- More than one policy seed per simulator in sections 1–11: every policy result there is one seed. Section 13 adds two more seeds for `FT160`, and Step 2's pairs.
- More than 500 policy iterations.
- Anything on the Mac.

## 13. How many policies per simulator (14 November)

**The question.** The first early-failure draft of Option D trained one policy in a simulator whose first policy failed, and two in one whose first policy walked. How many policies should a tested simulator get, for the best chance that the ladder gives a clear answer? Everything in this section was measured on the cloud copy.

**Step 2's 22 policies, re-measured with the walking score** (the Mac's actor files; Option C's protocol, 96 episodes of 1,000 steps; `score_s2.py`):

| Simulator | Seed 1 | Seed 2 | Same verdict? |
|---|---|---|---|
| `REL` | 0.604 | 0.574 | yes: walks (the seed-0 pilot: 0.769) |
| `REL__fp16` | 0.600 | 0.621 | yes: walks |
| `REL__u10` | 0.249 | 0.531 | yes: walks, but far apart |
| `REL__int8w` | 0.058 | 0.060 | yes: fails |
| `REL__u20` | 0.063 | 0.061 | yes: fails |
| `REL__u30` | 0.004 | 0.062 | yes: fails |
| `REL__s192` | 0.051 | 0.007 | yes: fails |
| `REL__u50` | 0.069 | 0.069 | yes: stands still |
| `REL__s128` | 0.069 | 0.069 | yes: stands still |

The other pilots (seed 0): `A_s0` 0.011, `B_s0` 0.006, `A_s0__u50` 0.005.

- **All nine pairs agree** on failed or not. Step 2's simulators lie far from the boundary: their policies walk well or not at all.
- **The walking score's verdict is steadier than Step 2's fall rate.** By fall rate, `REL__u30`'s two policies split (100% and 9%). By walking score both fail (0.004 and 0.062): the second stands rather than walks.
- **Walking policies of one simulator differ by about 0.12** (pooled standard deviation). Most of it comes from `REL__u10`, the walking copy nearest the boundary.
- A score of 0.069 is the standing reference (0.0686): those policies stand still.

**The failed adapted simulator with three seeds.** `FT160`, trained with section 9's control script and settings (Step 2's start states); only the seed differs (`ppo.py --seed`).

| Seed | 200 | 300 | 400 | 500 iterations | Falls at 500 |
|---|---|---|---|---|---|
| 0 (section 9) | 0.039 | 0.037 | 0.074 | 0.100 | 86% |
| 1 | 0.043 | 0.038 | 0.053 | 0.063 | 97% |
| 2 | 0.057 | 0.098 | 0.088 | 0.116 | 88% |

All three fail, at every checkpoint. `FT160`'s failure is not one unlucky policy.

**Where a second or third policy matters.** Far from the boundary it changes nothing: ten simulators out of ten gave the same verdict with every seed. Near the boundary, where the search puts most of its tests on purpose, one policy is close to a coin toss, and each verdict steers the next test.

**A simulated ladder** (`design.py`, recalibrated in `design_cal.py`):
- It plays the plan's search 300 times per setting: six runs of three tests, plus `REL`.
- Each policy either fails (score near 0.06), with a chance that rises with the amount of adaptation, or walks (scores spread by 0.12, as measured above).
- Two made-up checks are tied to the simulators' true scores, one strongly and one weakly.
- Each study ends with the registered kind of test: a bootstrap interval for one rank correlation ("main test"), and one for the difference of two ("comparison").
- Settings: a sharp boundary or a gradual one; a large or a moderate difference between the two checks; runs independent, or sharing half of the checks' noise within a family.
- It counts how often each test gives a clear answer (its interval excludes zero), and how often the search's verdict on a simulator is wrong.

Large difference between the checks, families sharing half their noise (the cautious setting); sharp boundary / gradual boundary. "Hours" is policies × 2.05 hours / 2 workers, with no early stops.

| Rule | Runs | Hours | Wrong verdicts | Main test clear | Comparison clear |
|---|---|---|---|---|---|
| one policy for a failed simulator (first draft) | 6 | 28 | 0.12 / 0.25 | 0.69 / 0.44 | 0.61 / 0.37 |
| two for every simulator | 6 | 37 | 0.12 / 0.25 | 0.76 / 0.59 | 0.68 / 0.43 |
| **two, and a third after a split (chosen)** | 6 | 40 / 43 | 0.07 / 0.20 | 0.73 / 0.62 | 0.68 / 0.48 |
| three for every simulator | 6 | 55 | 0.08 / 0.21 | 0.81 / 0.65 | 0.76 / 0.53 |
| five for every simulator | 6 | 92 | 0.06 / 0.17 | 0.88 / 0.82 | 0.78 / 0.71 |
| two, and a third after a split | 7 | 47 / 51 | 0.08 / 0.21 | 0.82 / 0.65 | 0.75 / 0.54 |
| two, and a third after a split | 9 | 60 / 65 | 0.08 / 0.21 | 0.91 / 0.80 | 0.86 / 0.71 |

The other seven settings (`design_cal_fam0.log`, `design_cal_fam50.log`) and the first runs with a spread of 0.09 (`design_run1.log`, `design_run_fam.log`) show the same pattern. Each value is from 300 studies, so it is uncertain by about 0.03.

- **The first draft's rule is the weakest in every setting.** Across all sixteen settings, the chosen rule gives a clear answer more often (by 0.01 to 0.23) and makes a fifth to two-fifths fewer wrong verdicts, for 12 to 15 hours more.
- **Three policies for every simulator change no verdict.** A third policy cannot overturn two that agree. It costs 12 to 15 hours more, for a steadier mean score.
- **Beyond three, more simulators help more per hour than more policies.** Nine runs with the chosen rule (60 to 65 hours) do as well as five policies for every simulator (92 hours), or better.
- **Caveats.** The model is a sketch. Real repeat runs may be more alike than its families allow, which would shrink the gain from more simulators. It assumes both workers are always busy. The plan gives each worker whole runs, so a seventh run costs about 15 hours of waiting, not 7.

**What the literature says.**
- Colas et al. (2018) explain how the number of random seeds relates to the chances of statistical errors, and give guidelines for choosing it for a t-test or a bootstrap interval.
- Agarwal et al. (2021): with only a few runs per task, a result cannot ignore its uncertainty; report intervals, not single numbers.
- Mathieu et al. (AdaStop) adapt the number of runs to the data: runs are added only while a comparison is unclear, and stop as early as possible, with a low chance of error. The chosen rule is the smallest version of this idea: a third policy only when the first two disagree.

**The rule chosen.** Two policies for every tested simulator; a third when exactly one of the two failed; the verdict is the majority. With six runs, the ladder would take about two days instead of about a day and a half.

**The author's choice: all seven runs** (14 November). The list's seventh run is allowed, and no run starts after 56 hours of job time. In the simulated ladder the seventh run gained about as much as a third policy for every simulator (seven runs with the chosen rule against six, in the eight settings of `design_cal.py`: the main test was clear 0.01 to 0.16 more often, the comparison up to 0.07 more often; on average about 0.08 and 0.05, against about 0.07 and 0.05 for a third policy for every simulator). The ladder now takes about two and a half days.

**Still uncertain.** How often ladder simulators will split is unknown: Step 2's simulators and `FT160` lie far from the boundary. The ladder table will show it (the plan reports how often a test needed a third policy).

# C2R: Amendment 1's reading checked (step 3) and calibration (step 4)

Sources: `results/c/c2r_facts.json` (`scripts/c/c2r_facts.py`), `results/c/calibration.json`, `results/c_pilot1.txt`.

**Reward scaling.** Each per-step term is weight × term × 0.02 s (lite `envs/base.py`). The logged terms are an episode sum divided by the limit's duration (256 × 0.02 = 5.12 s): a rate per second over the limit, so they shrink when episodes end early. For the released model, logged terms × 5.12 equal the logged return (−3.06 against −3.06 at iteration 100). In our runs they agree only roughly, because the two average different sets of episodes. Perfect tracking earns 0.03 per step, so the fall cost of 4.0 is about 133 steps of it, as the amendment says.

**Fall cost.** None. The 12 terms plus the uncertainty penalty include no termination term. A predicted fall only ends and resets the episode.

**First pilot's start states, by data source** (6M; 10k in brackets): expert 37% (34%), noise 0.1 9% (11%), noise 0.3 14% (11%), noise 0.6 9% (10%), smooth 14% (11%), trained 17% (22%). None came from the authors' recorded data. Under the amendment's rule: 6M, 96,017 windows from 103 episodes; 10k, 2,907 windows from only 3 episodes.

**One partial disagreement** (`optc/DECISIONS.md`, C2R 1). In the 6M run, the summed penalties outweighed tracking in every logged iteration, as the amendment says. But vertical speed plus tilt exceeded the uncertainty charge in only 71 of 1,994 iterations, 52 of them before iteration 100. From about iteration 10 onwards, the uncertainty charge was usually the largest single penalty.

**Calibration** (closed loop; the authors' policy, noise-free):

| simulator | ū | λ | falls | tracking/step |
|---|---|---|---|---|
| released model | 0.46 | 1 | 0% | 0.025 |
| 6M | 1.54 | 0.30 | 16% | 0.009 |
| 10k | 2.05 | 0.23 | 43% | 0.010 |
| `A5_s0` | 1.87 | 0.25 | 3% | 0.012 |

In MuJoCo the same policy earns 0.029 per step from tracking.

**Gate A2 failed.** The 6M simulator lets the authors' policy fall in 16% of episodes (limit 10%), and gives it 31% of its MuJoCo tracking (need 80%). Its largest charge is vertical speed: 0.089 per step, against 0.00007 in MuJoCo. This fits the kinematic residual of 9. The pilot job therefore retrains both pilot simulators with 2 × S steps first.

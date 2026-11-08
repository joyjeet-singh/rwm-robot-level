# AIR 2027 Step 2: report

**Question.** Do the paper's checks predict how well policies trained inside learned simulators walk in a hand-built MuJoCo ANYmal D model?

**Gate 1 (passed).** RWM's released policy in the calibrated model fell in 1.25% of 80 episodes (gate: under 10%). Two feet were down 78.9% of the time, against 84.4% recorded. Speed-tracking error was 0.069 m/s, against 0.072 recorded.

**Pilot (Gate 2: two of three conditions passed).** A policy trained in the released five-network model fell in 10% of episodes. Policies from three rebuilt single-network simulators fell in every episode; why was not studied. That failed condition P2, so the pre-set fallback pool replaced the 25-simulator pool: the released model and eight compressed copies, 18 policies.

**Registered verdicts: all not settled.**

| Test | Estimate | Interval | Verdict |
|---|---|---|---|
| R0: 100-step error vs physics return | +0.23 | 95% [−0.87, +0.81] | not settled |
| R1: growth rate vs one-step error | −0.68 | 97.5% [−1.50, 0.00] | not settled |
| R2: kinematic residual vs 8-step error | −0.05 | 97.5% [−0.47, 0.00] | not settled |

Every secondary and exploratory analysis was also not settled. R1's and R2's point estimates favoured the error checks, but both intervals include zero. The study neither supports nor rules out any check. The policies from the two best-scoring simulators still fell in 31–45% of episodes. Policies from the two most damaged copies stood still, never fell, and ranked third and fourth by physics return, so that measure rewards staying upright as much as walking.

**What the physics model can and cannot show.** It can show how a policy behaves in one hand-built model, with a fitted PD motor because the ANYdrive network file could not be obtained. It is not the simulator that produced the data, and it is not hardware. The test is narrow: nine versions of one simulator, differing by compression only, with in-sample checks. No rebuilt simulator produced a walking policy in the pilot.

**Review.** Referee score: **borderline**. The statistics reviewer recomputed the main results; they match the registration, and no result changed. Its wording faults are fixed, including a reversed sign in a table caption. Declined points, with reasons, are in `step2/DECISIONS.md` ("M7 declined"): chiefly E100 as the target, the title, the shared parent, and thin novelty in the first half.

**Pipeline.** All checks pass: identical builds, six pages with no spare room, abstract 198 words, 181 claim assertions, numbers, references, double-blind scans, rebuilt supplement (0 hits).

**What the author must check.**
- Whether the referee's points on novelty and E100 justify more work before submitting.
- The AI-use statement, and the paragraphs flagged in `improve/REPORT.md`, in your own words.
- The corrected facts against RWM, RWM-U and iKCE.
- The supplement for your name before any upload. It holds no git history; the paper now says it lists the registrations' order.
- Submission window: 15 Nov to 31 Dec 2026. Nothing was submitted or uploaded.

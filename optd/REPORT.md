# Option D: report

**From-scratch simulators failed; what replaced them.** Option C's simulators, trained from scratch on data from our
MuJoCo ANYmal D, could not keep the released walking policy upright. Probable causes: wrong state scaling, and far too
little training (about a tenth of one pass over the data). Option D instead fine-tuned the released model on MuJoCo
data, in runs searched for the amount of fine-tuning at which policies stop walking.

**Errors and policies.** Six runs tested 18 fine-tuned simulators (40 policies). All 18 had lower 1-step and 8-step
error than the released model; only 2 had lower 100-step error. Yet 13 of the 18 failed: at least two policies scored
below 0.15 (released policy 0.942, standing still 0.069, the released model's own policies 0.59).
Across the 19 simulators, 1-step and 8-step error ranked walking partly in reverse (ρ about −0.42; descriptive).

**Skipped or stopped.** Run `M_s1` (mixed data, seed 1) was skipped: its start rule needed `M_s0` to find a walking
boundary, and every mixed-data save failed at every tested step. No policy hit the collapse stop rule. Three
mixed-data policies failed by their simulator (a non-finite loss or a repeated crash) and count as failed. Four
simulators needed a third policy. The job ran about 53 hours.

**Verdicts.** The premise R0 and the primary tests R1–R2 were NOT SETTLED (19 simulators). Secondary: the walk test's
imagined tracking and the networks' disagreement ranked walking well (ρ +0.79, +0.84) and beat 8-step error; on the
pooled 27 simulators R0 was SUPPORTED and the kinematic residual beat 8-step error. These wins are helped by the error
checks' reversed ranking, and the walk test may only measure how far a save moved from the released model. No check
separated failed simulators from the others.

**What the MuJoCo world can show.** A hand-built ANYmal D with a fitted PD motor, not Isaac Lab or hardware, it
shows whether a policy trained in a learned simulator walks in a second, independent physics model, not whether it
transfers to the robot. The saves are versions of one simulator near the walking boundary, so intervals are
optimistic, and the amount of fine-tuning may drive every correlation.

**Referee's score: borderline.** Main points: the recommendation's scope (now limited to prediction error),
wording, and labels. Fixes: `optd/DECISIONS.md` D4-1, D4-2; the rest are under "D4 declined" (no
space, or new analysis needed). The statistics reviewer found `results/d_stats.json` consistent with
`prereg/PREREG_D.md`; its four SHOULD FIX points were fixed.

**The author must check:**
1. Where `results/d_stats.*` came from: they predated D2 and their writer is unrecorded (DECISIONS D2-1); a
   rerun was byte-identical.
2. The abstract, the "Policies in a physics model" paragraph, Limitations and Conclusion, cut hard to fit six
   two-column pages (0.06 textheight free).
3. That the registrations' timestamps are self-reported git commits (the paper says so).
4. The declined referee points, especially Q1's dependence on pool make-up and novelty against [29].
5. The supplement zips in `submission/`, rebuilt in D4 (scan 0 hits), not committed.

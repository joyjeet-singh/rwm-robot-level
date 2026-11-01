# Internal review of the AIR 2027 submission (1 November 2026)

This review is of `paper/AIR2027_submission.pdf` as built on 1 November 2026, with 10 pages single-column and 6 two-column. Its points map to the traceability table at the end of `REVISION_PLAN.md`.

**Verdict.** This is careful, unusually honest work, and a few of its findings are genuinely useful. But:
- the main new result is mixed;
- the novelty is modest;
- the scope is narrow (no controller, no hardware, one gait);
- several sentences claim more than the data show.

For AIR 2027 it would score borderline to weak accept. With the text fixes below, it should be a fairly safe accept there. It wouldn't be competitive at ICRA, CoRL or RA-L unless it adds a policy trained in the simulator, or a second robot.

## What reviewers will like
- **The rebuild is checked down to the gradients.** Its outputs, losses and gradients are identical to the release's, which very few papers bother to show.
- **The questions were fixed in advance.** The new study's questions and decision rules were committed before its measurements were run. Failures are reported as plainly as successes, including a failed replication.
- **Concrete findings anyone using RWM would want:**
  - The ensemble's uncertainty shrinks toward zero, because the loss as written is lowest at zero spread.
  - The released training windows join the end of one episode onto the start of the next.
  - The release's evaluation feeds each command one step late.
  - Pruning 30% of weights looks almost harmless one step ahead (90% of the skill kept), but by 7 s it is worse than predicting that nothing changes.
- **It counts independent trajectories, not overlapping windows,** which avoids a common way of overstating certainty.
- **Both figures are clear,** and the closing checklist is practical.

## Main problems, most serious first

**1. RWM's own paper already reports the first two headline results [P1].**
- RWM measures multi-step error and says its rollout-trained model "significantly outperforms its teacher-forcing counterpart".
- RWM's Appendix A.4.1 finds that a longer training horizon improves long-horizon accuracy, and that one-step training does poorly.
- Our paper never says this, and its introduction implies that simulators like this one are checked one step ahead.
- **Fix:** present these rows as independent confirmation. Aim the one-step criticism at practice that really uses one-step checks; for example, RAMBO (following MOPO and COMBO) keeps the best 5 of 7 models by error on held-out single steps. Add *Objective Mismatch in Model-based RL* (Lambert et al., L4DC 2020).

**2. The new result is mixed, and the abstract leads with the part that didn't hold [P2].**
- The growth rate (how fast a tiny error grows as the simulator runs on its own predictions):
  - beat one-step error on the 97-simulator set;
  - tied with the 8-step check;
  - failed its registered replication on the 255-simulator five-split set.
- The paper says the rules were committed "before the data that tested them existed". But the growth rates of the 57 trained models in the main set had already been computed and seen.
- The registration file is open about what was known: the growth rate already ranked these models well (0.85–0.90), and teacher forcing was already known to win one step ahead and lose over long horizons. The paper says none of this.
- **Fix:** add one sentence saying what was known in advance, and lead with what held:
  - one-step error ranks training choices backwards;
  - the kinematic residual ranked best;
  - an 8-step check is nearly as good and trivial to run.

**3. Overclaims [P3].**
- **"Matched or beat every other check on both pools":** on the five-split set this analysis was added after the results were seen.
- **"Pruning adds bias rather than instability":** true for 10–30% pruning. But 50% pruning and removing units raise the growth rate by +0.26 to +0.40 per second on average, and 7 of the 50 unit-removed copies in the five-split set become unstable.
- **"Negative for those trained on their own rollouts":** RWM's released model is rollout-trained, on about 770 times more data, and its growth rate is +0.70/s. The paper never reconciles this.
- **"Without using any recorded future":** the label-free checks still replay the recorded future commands.
- **"A defect that only a short-horizon check reveals" (the stale command):** this turns "not resolved" into "not there". On the four held-out trajectories, the earlier study found it raises the 100-step error by +22.5% [+6.3, +41.7].

**4. The check comparison isn't quite like-for-like [P4].**
- **The checks see different horizons.** The label-free checks look at a 368-step run; the labelled ones see 1 or 8 steps. This is exploratory, from the stored results, with rank agreement against the 2-second error:
  - over the same 100 steps as the target, the residual still scores +0.94;
  - over just 8 steps it is level with the 8-step error on the 97-set (+0.83 vs +0.84), and below it on the five splits (+0.88 vs +0.95);
  - the growth rate is useless over 8 steps (+0.21).
- **Checks and target share data.** They use the same test trajectories, and every 100-step window starts with an 8-step window that the 8-step check also scores.
- **The real use case isn't tested.** A recording that supplies the history and commands also contains the future states. Label-free checks matter when a policy supplies the commands inside imagination, and that case is never tested.
- **The replication isn't fully independent:** one of its five splits is the original test split.

**5. Narrow scope, flagged late [P5].**
- One robot, one trotting gait on flat ground, ten 20-second simulated episodes. Recorded commands are replayed rather than chosen by a controller. There is no policy and no hardware.
- At 2 s, both learned simulators place the robot less accurately than assuming it keeps its last velocity (0.51 m and 0.84 m, against 0.45 m).

**6. The uncertainty ranges are probably too narrow [P6].** They treat the eight compressed copies of each parent as independent, and they hold the four test trajectories fixed. Several secondary tests are also not adjusted for how many tests were run.

**7. Presentation [P7].**
- **Factual error:** Table 1's caption says "Six readings", but the table has seven rows.
- **One comparison, two numbers:** Table 1 says teacher forcing is 5.0× worse at 368 steps; the text says 5.85×, using a different error measure.
- **Table 1 mixes model sets,** and the stale-command row points the opposite way from the rest.
- **Too dense:**
  - Results from five earlier studies are packed into six pages.
  - Section 4.3 condenses a whole earlier study into one paragraph.
  - The small-caps verdicts will read as jargon.
- **Missing basics:** optimizer, learning rate and batch size; how "teacher-forced" differs from "N = 1"; why width 124.
- **Figure 2 shows the wrong check:** it plots the growth rate, not the residual that actually ranked best.
- **Hard to verify:** many results rest on "earlier, unpublished analyses", and the registration times are our own git commits.
- **Reference [1]:** it cites RWM as an arXiv preprint, but RWM was a poster at the NeurIPS 2025 workshop on Embodied World Models.
- **"302 versions":** two-thirds are automatic compressed copies of 25 parents.
- **AI-use statement:** it says "the author" under an "Anonymous Author(s)" byline. Rewrite the introduction and discussion in your own voice.

**Related work to add [RW]:**
- Malik et al. (ICML 2019) and Kuleshov et al. (ICML 2018), on recalibrating uncertainty.
- Mikhaeil et al. (NeurIPS 2022) and Hess et al. (ICML 2023), on teacher forcing, chaos and Lyapunov exponents in recurrent-network training.
- Manek & Kolter (NeurIPS 2019), on stable learned dynamics.

**Found while planning [N1]:** an earlier registered study of these same models (PerconAI) already fine-tuned the unit-removed copies, over three seeds. Width 192 kept most of an equally fine-tuned full-width model's skill, width 128 was not settled, and narrow models trained from scratch did as well or better at 100 steps. The paper says only that width 192 "needs fine-tuning".

## Fixes
1. **Quick corrections:** the caption, the 5.0/5.85 mismatch, "the author", and reference [1].
2. **Credit RWM:** acknowledge RWM's own results, and add the missing references.
3. **Rebalance the claims:**
   - Re-centre the abstract and introduction on what held.
   - Label the five-split residual result as exploratory, everywhere it appears.
   - Limit "bias, not instability" to moderate pruning.
   - Reconcile the released model's positive growth rate.
   - Write "no recorded future states", and drop "only" in the stale-command sentence.
   - Add one sentence on what was known before registration.
4. **Two quick analyses:** add the horizon-matched comparison and the wider ranges, labelled exploratory.
5. **Appendix for reviewers [F5]:** use the first submission's lack of a page limit for a short appendix, and cut it for the camera-ready.
6. **New experiment [F6]:** briefly fine-tune the pruned copies and re-measure.

## Sources
- RWM: https://arxiv.org/html/2501.10100v4 ; venue: https://neurips.cc/virtual/2025/loc/san-diego/123991
- Lambert et al., *Objective Mismatch* (L4DC 2020): https://mlanthology.org/l4dc/2020/lambert2020l4dc-objective
- Malik et al. (ICML 2019): https://proceedings.mlr.press/v97/malik19a.html
- Mikhaeil et al. (NeurIPS 2022): https://arxiv.org/abs/2110.07238v3
- Hess et al. (ICML 2023): https://proceedings.mlr.press/v202/hess23a.html
- Rigter et al., RAMBO-RL (Appendix B.1): https://arxiv.org/pdf/2204.12581

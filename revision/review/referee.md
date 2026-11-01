# Referee report: "One Step Ahead Is Not Enough: Testing a Learned Quadruped Simulator at the Horizon It Is Used"

(air-referee, R10, 2026-11-02; given only `revision/review/paper.txt`. Saved verbatim.)

## 1. Summary and recommendation

The authors rebuild RWM's open GRU dynamics model for ANYmal D and check that it matches the release exactly. They then train or compress about 300 variants on 10,000 simulated steps (10 trotting episodes) and score them on held-out episodes with recorded commands replayed. One-step error ranks the training objective and the training horizon in the opposite order from 2 s error, and it hides most of the loss from pruning without retraining. The released ensemble's spread is too small. In a pre-registered comparison of checks, a label-free kinematic residual and a finite-time Lyapunov "growth rate" rank 100-step error better than one-step error. The growth rate's advantage did not replicate on a pool made mostly of compressed copies. The scope is stated honestly, the registration is careful and the limitations section is unusually candid. However, the abstract and the checklist are worded more generally than the evidence allows, several tables and passages contradict each other, and the statistics will be hard going for this audience.

**Recommendation: major revision.** The core observations are sound but mostly confirm known results. The headline claims need narrowing, and the paper needs heavy restructuring to fit six pages and to be readable by roboticists.

## 2. Major points (ranked by impact on the decision)

1. **Headline claim about the residual is worded more strongly than its registration status supports (abstract, p1; Sec. 4.2, p6; Table 7, p14).** The abstract says: "In a registered comparison over 97 simulators, a kinematic-consistency residual ranked the 2-second error best." The two primary questions were about λ1, not the residual. The residual's lead over E8 is a secondary test (Q4, +0.12 [+0.02, +0.23]) that fails Holm's correction (p = 0.053). Its ranking of the 55 trained five-split models (+0.93) was also known before registration (p4; A2). *Fix:* lead the abstract with the primary results. Call the residual's lead "secondary, uncorrected; not significant after Holm's correction", and say that its strength was partly known in advance.

2. **The pruning lesson only holds without retraining (Table 1, p4; contribution bullet 1, p2; checklist item 2, p6).** Deep Compression [11], cited as the standard, retrains after pruning. After 500 fine-tuning iterations, the one-step and 100-step values of K are indistinguishable (Table 10, "+0.1 [−1.2, 0.9]", p16). So the one-step check misleads only for pruning without retraining. *Fix:* say "pruning without retraining" in the abstract, in bullet 1 and in the checklist. Also define what counts as "passed" in "passed a pruned model" (p6): no acceptance threshold is given.

3. **The "stability account" is causal in tone but fails on the model that matters (Sec. 4.1, p4–5; Sec. 4.2, p6).** The text says "The growth rate explains why", but the evidence is correlational: 15 models, and five N values that are never listed. The released RWM model, trained on rollouts on 750 times more data, has λ1 = +0.70/s and the lowest one-step error. Checklist item 3 ("A positive growth rate flags a simulator that amplifies errors", p7) would therefore flag the best model, and 15 of the 17 worse-than-floor simulators have λ1 ≤ 0. *Fix:* replace "explains" with "is associated with". Report the released model's E100. State in the checklist that λ1 > 0 does not imply a poor simulator, and that λ1 ≤ 0 does not imply a good one.

4. **The comparison of checks does not give each check the same information (Sec. 4.2; Table 4, p12).** λ1 and r̄J are computed over 368-step rollouts on the same trajectories as the 100-step target, whereas E1 and E8 see only 1 or 8 steps. When the windows are matched at 8 steps, λ1 collapses (+0.21 on F18, −0.02 on X5) and r̄J ties with or trails E8. The advantage therefore comes from rolling the model out long, which needs no labels but does need the recorded commands. Also, r̄J is normalised "in units of the recording's own residual" (p3), so it is not fully label-free. *Fix:* frame the claim as "long label-free rollouts versus short labelled ones" and say which recording supplies the normaliser. Move the window-matched analysis into the main text.

5. **The unit of independence and the resampling scheme are inconsistent (p3 vs A4, p12).** Sec. 3 says correlation intervals "resample models". A4 says "The registered intervals resample the held-out trajectories. The wider intervals also resample the parent models." Only one of these can be true. F18 has 4 trajectories from 2 episodes, and a percentile bootstrap over 4 units has very few distinct resamples, so the 97.5% primary intervals are fragile. The sign test uses episodes (n = 10, p4), while the bootstrap treats the two trajectories of one episode as independent. Finally, X5 split {1,8} appears to reuse the F18 models: the counts of 101 trained and 200 compressed only add up if it does. That makes the "replication" partly non-independent, but the paper never says so. *Fix:* state exactly what is resampled for each interval, say whether models are shared between F18 and X5, and justify treating trajectories rather than episodes as the unit.

6. **The cause of the ensemble's small spread is not shown (bullet 1, p2; Sec. 4.1, p5).** The loss argument, which is smallest at σ = 0, concerns each head's own predicted variance. Disagreement among five heads on one shared trunk is a separate quantity. A shared trunk is a more obvious cause of under-dispersion. Every uncertainty number also comes from the released model on data it "may have trained on". *Fix:* define "spread" (head variance, head disagreement, or both) and weaken "because". A cheap analysis with no new training would settle it: measure calibration of the authors' own held-out five-head ensembles, which are already trained (p3).

7. **Scope is stated in the abstract but dropped elsewhere (title; Discussion heading, p6).** "A checklist for vetting a learned robot simulator" and "the horizon it is used" are general claims. The evidence comes from simulated data, one gait, a velocity command that "appears to change once" per episode, replayed commands and no policy. Also on p4, a constant-velocity extrapolation (0.45 m) beats both learned simulators in position at 2 s. For roboticists this is a central practical finding, and it is buried. *Fix:* call it "a checklist from this case study", restate the scope in the contribution bullets, and add the constant-velocity baseline to Table 1 and Fig. 1.

8. **Novelty relative to prior work is limited and under-positioned.** Rollout training beating teacher forcing is RWM's own ablation and also appears in [2, 3, 14, 15]. The kinematic diagnosis closely follows [26], and the trajectory-based compression sensitivity follows [29]. Missing references:
   - System identification's long-standing distinction between prediction error and simulation error (output error), and simulation-error minimisation: Ljung; Piroddi and Spinelli 2003.
   - Ross and Bagnell 2012 (DAgger-style system identification).
   - Multi-step losses: Asadi et al. 2019.
   - Contraction theory: Lohmiller and Slotine 1998. Control reviewers will expect this given the word "contracting".
   - Learned legged-robot dynamics and actuator models: Hwangbo et al. 2019.

   *Fix:* say precisely what is new beyond [6, 26, 29], namely the explicit comparison of checks and the registration.

9. **Length and readability.** The body runs to 7 pages plus 6 appendix pages, and key evidence (A4, A7) sits in the appendix. For a six-page two-column camera-ready, cut back to the training-objective, pruning and checks results, and drop or shorten the run-time alarms (A6) and the unit-removal material.

## Minor points

1. **The released model's evaluation (p3).** The authors call the one-step-late command feed a "defect" of the release, but the 100-step effect is resolved only on the 4-trajectory subset and in a different metric ("22% [6.3%, 42%], in the release's relative error"), and not on the 20 trajectories. This reads as selective. Also, "raises ... most" is unclear.
2. **Splice count (p3).** Nine episode boundaries × 39 forty-step windows = 351, not 352. Please check.
3. **Table 7, p14: wrong label.** "Q5: λ1 − E1, compressed" gives +0.27. Sec. 4.2 gives this value for the analysis "without the one-step-trained models (88 simulators)", and Table 2's compressed column implies −0.07. The label looks stale.
4. **Table 7: unclassified tests.** There are 13 verdicts but only 2 primary and 7 secondary. Q1 and Q2 at 24 and 368 steps are neither, and get no Holm p-value. State their family and confidence level.
5. **Limitations, p7, contradicts Sec. 4.1, p5.** Limitations says "the 8-bit variant rounds weights only", but Sec. 4.1 reports a registered test of quantising weights and activations.
6. **Table 6 vs "registered on F18 only" (p6).** Table 6 shows a "registered" interval for X5 r̄J, which conflicts with the text.
7. **Table 2 verdict label.** "TRAINING-HORIZON BETTER" should read "8-STEP ERROR BETTER".
8. **"Q4" means two different things.** It is the residual test in Table 7 and the scratch-versus-fine-tune test in Table 9. Rename one.
9. **Table 1.**
   - The "–" at 24 steps in the ensemble row is unexplained.
   - There are no intervals on the error ratios.
   - The fine-tuned row uses seeds 0–2 while the other rows use seeds 0–4.
   - K > 100% reflects fine-tuning itself; report it against the control (99.9%).
10. **Contribution bullet 2 vs Fig. 2 caption.** The bullet says λ1 is "positive for simulators trained one step ahead", but N = 2 is also positive (+0.44; Fig. 2 caption: "one or two steps ahead"). List the N values.
11. **Undefined or late-defined terms.** F18 and X5 are used (p3, line "On F18 ...") before they are defined; the names are cryptic. Also undefined or late-defined:
    - E_h: mean over the window or the value at step h? Fig. 1 plots the value at step k.
    - "floor", "level with", "a rule reads a verdict", "lose their reading", "unresolve"
    - "within band at 6 of 6 horizons" (which band, which horizons?)
    - "the release's relative error", "two-level", "Benettin's method", "Jacobian–vector products", "Holm", "Bonferroni", "Spearman"

    Add a one-line plain-language gloss for each, or a small glossary.
12. **Dense, number-heavy passages:**
    - the abstract
    - the last paragraph of Sec. 3 (p4)
    - the Compression paragraph (p5)
    - "Three exploratory analyses..." (p6)

    Move the intervals into tables and keep one number per claim in the text.
13. **λ1 across different widths.** It is computed "in the network's own coordinates", so comparing it across hidden widths (124, 128, 192, 256) needs justification.
14. **Registration cannot be checked externally.** Registration via "commit times from our own version control" (p4) can be altered. Use a timestamped public registry (e.g. OSF) or publish commit hashes.
15. **Multiplicity outside Table 7.** Holm's correction covers only 7 of many tests (13 in Table 7 plus 12 in Study FT plus many exploratory ones). Say so.
16. **Anonymity.**
    - A2 (p11): "Section 2 of PREREG.md states what the author had read ... through this folder's code". This reveals a single author and the repository layout, and is inconsistent with "author(s)" elsewhere.
    - A7 gives an exact local timestamp, "2026-11-01 22:05". Harmless, but remove it.
    - No names or institutions appear.
    - If [26] is the authors' own work, cite it in the third person.

## 3. Typos and wording

- p1, abstract: "increasingly with horizon" → "by a factor that grows with horizon"; "2.0 and 2.6 times as wrong" → "2.0 and 2.6 times the error".
- p1: "run forward on its own predictions, as image-based world models are" → awkward; rephrase.
- p3: "gradient clipping none" → "no gradient clipping".
- p3: "the 10–50% smallest weights" → "the smallest 10–50% of weights".
- p3: "Held-out arenas" → "Held-out data".
- p4: "a rule reads a verdict only when" → "a rule gives a verdict only when".
- p4: "lose their reading" → "are no longer resolved".
- p7: "unresolve 3 registered verdicts" → "leave 3 registered verdicts unresolved".
- p11: "the author had read" → "the author(s) had read".
- p13: "7 of the 150 copies ... and 7 of the 75 copies" → state that they are the same 7 copies.
- Table 2: "TRAINING-HORIZON BETTER" → "8-STEP BETTER".

**Score: borderline**

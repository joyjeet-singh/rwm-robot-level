# air-claim-checker report (R10, 2026-11-02; saved verbatim)

Given: paper/body.tex, paper/appendix.tex, revision/notes/R5_keys.md, results/stats.json,
results/revision_exploratory.json, prereg/PREREG.md, results/ft_stats.json, prereg/PREREG_FT.md.

Findings, most serious first. Everything not listed passed: Table 1, Table 2, the FT tables, the Q-values and intervals, and the counts all match stats.json, ft_stats.json and revision_exploratory.json.

1. Holm family miscounted. body.tex:161-163 "The secondary verdicts ... 2 of the 7 survive"; appendix.tex:152-153 and the A4 table caption "the 7 secondary tests".
- Evidence: revision_exploratory.json A2c_holm.family has 7 tests (Q3, Q4, Q5 x2 each, Q6). A2a_two_level.registered_tests has 13, i.e. 2 primary plus 11 secondary. Q1 and Q2 at E24 and E368 are secondary at 95% but are not in the Holm family.
- Fix: write "the 7 secondary tests of Q3-Q6" and say the four E24/E368 verdicts were not Holm-corrected.

2. "Held-out" used for the released model. body.tex:105-106 "four held-out trajectories of episodes 1 and 8 (22% ...)", key stale_ho_h100 (results.released.held_out_n4).
- This contradicts body.tex:253 and :288 (training rows unrecorded, "may have trained on") and PREREG.md §3 ("never called held out").
- Fix: say "the four trajectories of episodes 1 and 8, which the release may have trained on".

3. Uncertainty paragraph (body.tex:248-253) and the Table 1 spread-factor row have no label or scope.
- Spread factors 6.6/30.8/41.7 (n8_k_rel_epi_h*) are in the release's relative error, not the paper's measure, and are not descriptive-labelled. PREREG.md §3 says the released model is "descriptive only".
- The "83.5% against 95.45%" tail coverage is horizon 100 only (cov2_perh_h100), and the horizon is not stated.
- Discussion item (4), "Read ensemble spread as a ranking", generalises from one released model on episodes it may have trained on.
- Fix: add "(descriptive; relative error; 100 steps; one model)".

4. Residual verdict is stronger than every reading supports. Abstract (body.tex:17-18), Intro (:59-60) and Table 2 say "registered ... ranked best".
- Evidence: Q4_vs_E8 is secondary, +0.115 [+0.022, +0.233] at uncorrected 95%. Holm p = 0.053 (A2c_holm, Q4_vs_E8, survives = no, near = yes). The abstract's own second pool is level.
- Fix: add "(secondary; not Holm-robust)" to the abstract and Intro, or "best or level".

5. body.tex:230-234 gives "39%" and then "from 38%" two sentences later (A7 explains seeds 0-4 vs 0-2 only in the appendix). Add "(seeds 0-2)" at :233. The abstract's "500 fine-tuning iterations restored it" covers 30% pruning only, 15 parents. Say so.

6. body.tex:236-239: the +0.27/s (50%), +0.40/s (width 128) and "4 and 3 of 25 unstable" figures are unlabelled.
- Q6 covers only 30% (stats.json Q6). These come from lam_shift_X5 and are descriptive.
- "At 10 and 20% the shifts are smaller" (+0.02 [0.00, 0.04]) against 30% (+0.03 [-0.02, +0.07]): the intervals overlap. "Barely resolved" is vague.
- Fix: write "(descriptive)" and "similar", not "smaller".

7. body.tex:244-246. The registered 8-bit dynamic-quantisation test (perc_int8_K = 95.2% [91.8%, 100.9%]) is given without its verdict.
- Table 1's "8-bit weights" 95% at 368 steps is a different quantity (weights-only rounding, D2 int8w = 94.8%).
- Fix: name the label from keys.e2.Q1_int8_dynamic and write "weights-only" in the Table 1 row.

8. Abstract (body.tex:13-15) and Table 1 mix pools. 2.0 comes from 20 trajectories (teacher forcing, D3 B_over_A h100 = 2.05). 2.6 comes from 8 models on 4 trajectories (N=1 against N=8, F18) and has no interval. "Better one step ahead" for N=1 (0.86) has no interval either. Add "(2.6: 4 trajectories)".

9. appendix.tex:282-283 "matched ... within 2x10^-7". The maximum is 2.18e-7 (ft_models.json checks.C-FT4.*.max_rel_per_trajectory). Write "2.2x10^-7".

10. body.tex:226-227 "training a teacher-forced model longer raised its growth rate (+0.18 to +1.5/s)". This is 500 against 10,000 iterations (ng_pathB_lam_500). The paper's own teacher-forced models are +0.68 at 2,500 iterations and +1.53 at 10,000 (stats.json D1_groups_F18, 'B' and 'B 10k'). Give the iteration counts.

11. Figure 1 caption, body.tex:202: "starts within a few per cent of its parent". figure_facts.json gives pruned 0.145 against parent 0.135 at step 1, which is +7%. Say "about 7%".

12. Discussion item (5), body.tex:347 "Re-synchronise on a schedule instead". Appendix A6 only shows a schedule beat the kinematic trigger, not that it helps at all. Also, item (3) says "A positive growth rate flags a simulator that amplifies errors", but the released model has λ1 +0.70 with E100 0.40, against a floor of 0.80 (D5_released). On F18 only 2 of the 12 positive-λ simulators are worse than the floor at 100 steps (D4_F18). Narrow both to "at 368 steps" or "amplifies".

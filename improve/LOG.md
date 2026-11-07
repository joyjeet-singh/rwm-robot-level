# Improve round — log

## I0 status: COMPLETE (2026-11-07 20:30)
Branch improve, tag air2027-improve-start. Build+slack PASS (6/6 pages, abstract 197). Map and facts in improve/notes/.
Key finds: RWM imagines 100 steps/iter (not 24; 24 is num_steps_per_env); iKCE diagnoses one DreamerV3 model, no ranking.
Open: results/build.json and figures/fig_horizon.* are modified by the build/earlier work, left unstaged.

## I1 status: COMPLETE (2026-11-07 22:10)
Wrote scripts/im_within.py (results/im_within.json/.txt, exploratory), im_numbers.py (called from paper_numbers.py), im_facts.py (results/external_facts.json, 14 facts; quotes still unchecked against raw HTML).
Group sizes 48/40/40/97 enforced; group d reproduces Table 2's rho exactly; intervals equal stats.py's helpers. air-stats-reviewer: all points fixed. check_numbers all_agree, check_bib OK.
Headline: among the 48 rollout-trained F18 models rJ (+0.85) and lambda1 (+0.76) rank better than E8 (+0.38); on X5 lambda1 does not carry over (-0.21). Notes in improve/notes/I1.md.
Open: templates untouched (later sessions); slack/claims not rerun (no template change).

## I2 status: COMPLETE (2026-11-07 23:40)
All C1 text changes made in body.template.tex (and appendix A4 table tab:app-within, A6 released-model wording). New count keys in im_numbers.py.
Facts: RWM/RWM-U imagine 100 steps (24 is only an intermediate horizon); 0.13% = 7,991 / 6M via ext_* keys. Abstract 186 words.
Checks: slack PASS (6/6 pages, free slot 0.34), check_claims 127/0 failed, check_numbers all_agree, check_bib OK, submission_checks 0 failures.
Notes: improve/notes/I2.md (I3 instructions: Fig. 1 single 100-step line from external_facts; Tables 1, 3), I2_claims.md (49 lines).
Open: I3 has 0.04 textheight of margin; AI-use statement mismatches listed for the author in I2.md; figures/fig_horizon.* left unstaged for I3.
Commit: f4c63cc (I2: text changes).

## I3 status: COMPLETE (2026-11-07 21:50)
Fig. 1 single 100-step line from external_facts; Tables 1 and 3 regrouped (Table 3 in \scriptsize, DECISIONS.md); FloatBarrier before each appendix section (placeins absent: \clearpage fallback); check_claims 156/0 (+29 for I2_claims).
air-claim-checker: numbers all matched; 8 wording points fixed, rest listed in improve/notes/I3.md for I4.
Pipeline: slack PASS (6/6, free slot 0.30 = zero margin, abstract 186), check_numbers all_agree, check_bib OK, submission_checks 0, supplement 0 hits.
Visual check not possible (no gs; sips black): text-checked. Commit: d2232b8.
Open: I4 must add no main-text/table height. STEP2_PLAN.md untracked, not mine.

## I4 status: COMPLETE (2026-11-07 22:40)
Referee 4/10 (weak reject: novelty vs RWM's ablation, data scale, unequal comparison, overclaim in conclusion); stats reviewer 11 problems in 48 sentences.
Fixed: verdict wording (AR3 pooled not settled, CL5 worse, X5 residual not separated, two of three networks), labels (abstract residual secondary, imagined-return comparison exploratory), conclusion scope, counting sentence split, A3 note on E8/E100 overlap.
Declined with reasons and two Step 2 candidates: improve/DECISIONS.md (I4). Notes: improve/notes/I4.md; reviews: improve/review/.
Pipeline: slack PASS (6/6, free slot 0.30, abstract 185), check_claims 159/0, check_numbers all_agree, check_bib OK, submission_checks 0 failures.
Commit: 5b389d8 (I4: review fixes). Open for author: supplement history names; I3 open items (a), (b).

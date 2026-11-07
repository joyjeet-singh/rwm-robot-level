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

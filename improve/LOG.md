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

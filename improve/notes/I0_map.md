# I0 map (line = template line; words = rough count)
## body.template.tex (374 lines)
- Introduction 29-62 (496 w) sec:intro
- Related Work 63-82 (208) sec:related
- System and Protocol 83-139 (879) sec:setup
- Results 140 (heading only)
  - 4.1 One-step checks reverse training and compression choices 141-221 (947) sec:decisions
  - 4.2 Which check ranks simulators without recorded futures? 222-332 (1127) sec:checks
- Discussion 333-360 (489); Conclusion 361-375 (163)
Floats: tab:decisions (l.152, Table 1, 4.1); fig:horizon (l.174-178, Fig. 1, 4.1, fig_horizon.pdf); tab:checks (l.235, 4.2); tab:verdicts (l.252-258, Table 3, 4.2); fig:checks (l.323-328, Fig. 2, 4.2).
## appendix.template.tex (591 lines)
A1 6-74 (834) app:A1 [tab:app-train l.13-17]; A2 75-88 (208); A3 89-146 (402) [tab:app-horizon 98-102, tab:app-split 128-132]; A4 147-223 (600) [tab:app-two 154-158, tab:app-tests 195-200]; A5 224-295 (626) [tab:app-comp 242-247, tab:app-perc 273-278]; A6 296-321 (382); A7 322-421 (1106) [tab:app-ft 358-364, tab:app-ft-desc 389-394]; A8 422-507 (1106) [tab:app-cl 449-456]; A9 508-578 (794) [tab:app-ar 531-536]; A10 579-591 (213).
## Lines to change
- RWM imagination "24 steps per policy update": body 35 (RWM-U 100-step also l.35-36); Fig. 1 label: scripts/make_figures.py:46 ("24 steps (RWM)"), body 177 caption "imagination horizons of RWM and RWM-U"; table columns "24 steps" body 157; "24 steps" also body 284, 357 and appendix 113,118,209,211,374-375,399 (these are evaluation horizons, not RWM claims).
- Data fraction: {{budget_pct}} body 88, 182 (value 0.13%, from c2_pct).
- "750 times": {{scale_ratio}} body 203 ("about 750"; key scale_ratio, revision_numbers.py:144).
- Model counts: body 109 ({{s1_ar_n}}), 318 ({{s1_cl7_nmodels}}); 101/200/97/255/38 appear as keys (grep results via keys n_P etc.), not literal; body 132 "97.5%" is an interval level, not a count.
- iKCE: refs.bib key schafer2026 (arXiv:2607.05966); cited body 79 only (Related Work, "Kinematic consistency has diagnosed long-horizon failure"); kinematic residual body 18, 58, 122, 289, 298, 365.
- Registrations paragraph: body 126-134 (Statistics and registration); limits body 355; reproducibility 369.
- Imagined return: body 317; appendix 426, 492-495.

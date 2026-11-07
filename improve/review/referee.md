# I4 referee (air-referee), on improve/review/body.txt
Score 4/10 (weak reject).
1 Novelty thin: first results agree with RWM ablation; cut IV-A, drop FT from abstract; residual close to iKCE.
2 Stability may hold only at small data: released model lambda1 +0.70/s; say in abstract. [EXPERIMENT] realistic data size.
3 Unequal comparison: E8 shares trajectories with E100; F18 rho driven by 9 one-step models; within-method lam1 -0.21 on X5 only exploratory. Fix: say lambda1 mainly separates training methods; within-method into main text.
4 Independence/multiplicity: 40 copies share parents; 3 of 13 verdicts unresolved under wider intervals; 10 primaries no correction. Fix: report wider intervals in Table III; state number of primary tests.
5 Title/conclusion overclaim: no policy trained in any scored simulator; CL policy from released model. Limit claim to ranking by 100-step error. [EXPERIMENT] policy training + physics-sim test.
6 Readability: abstract tags; undefined F18, X5, AR, CL, FT1, Q1-Q6, floor, label-free, spread factor, Holm, percentile bootstrap, TRAINING-HORIZON BETTER; dense counting sentence (l.134-139).
7 Inconsistencies: Limitations "Q1 at 24 steps ... Q5 against E1" vs Table III; 39% (Table I) vs "from 38%"; "OBJECTIVE 40 ON X5" vs "40 rollout-trained" -- TF among them?; mixed decimals Table III.
8 Registration emphasis vs "largely predictable" / earlier-rule verdicts; supplement git history may carry names.
Minor: missing spaces (PDF extraction?); residual formula/integration step; uncertainty not held out in text; MLP undefined; difference of correlations can exceed 1; Fig. 2 caption "or with N = 2" vs legend; "judged at 97.5%"; name N values in "falls through +0.44, -0.43, -1.13".

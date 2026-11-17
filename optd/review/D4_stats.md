# D4: air-stats-reviewer on results/d_stats.json vs prereg/PREREG_D.md and the Option D sentences

(a) JSON matches PREREG_D Sec. 10: pools 19/27/16, levels 95/97.5/95, labels, 18 verdicts, 4 third policies, M_s1 skip,
substituted physics return, AUC, best-three; d_stats.txt matches the JSON; every paper number matches after rounding.
1. SHOULD: "registered tests stayed/were unsettled" -> "primary" (secondary registered tests did settle). FIXED.
2. SHOULD: "most tested ... failed" reads as a failure rate; the search set it. -> "13 of 18 searched". FIXED.
3. SHOULD: "ranked walking in reverse" overstates (no registered interval) -> "partly in reverse ... descriptive"; A13 causal wording. FIXED.
4. SHOULD: "released policy's tracking" ambiguous; Sec. 12 confound (amount of adaptation) missing from main text. FIXED.
5. NOTE: A13 walk-test claim should carry the adaptation confound. FIXED.
6. NOTE: A13 "All three registered questions" -> R0 (premise), R1-R2 (primary). FIXED.
7. NOTE: A13 "3-iteration code tests" -> "2- and 3-iteration". FIXED.
8. NOTE: d_stats.txt AUC lines omit dropped draws (15 of 20,000); JSON carries ci97.5 for secondary. Declined: d_stats.py and its results are read-only (registered); the paper quotes only ci95 for secondary.

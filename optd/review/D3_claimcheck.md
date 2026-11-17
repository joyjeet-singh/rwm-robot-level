# D3: independent claim check of the Option D text (air-claim-checker)

Checked: Sec. 4.2 paragraph, Table `ladder`, verdict-table PT rows, abstract/Limitations/Conclusion, appendix A12 root cause, A13
(not covered: walk-test details in A13, A11 tables, prereg hash). All typed counts, the tables and the rule descriptions matched
`results/d_*`, `results/c_*` and `prereg/PREREG_D.md`. Twelve wording findings:

| # | Finding | Action |
|---|---|---|
| 1 | rho of E1, E8 are point estimates, primary pool only (pooled: +0.14) | fixed: "point estimates" in Sec. 4.2 (pooled case already in A13) |
| 2 | secondary variants resolve (R2 label-free better in 4; pooled R0 supported); abstract/conclusion "unsettled" is for the registered primary pool | A13 states them; no room in the body (reserve 0); abstract and conclusion say "registered tests" |
| 3 | "most simulators failed" depends on the tested step counts | fixed: "most tested fine-tuned simulators" (abstract); conclusion "most policies failed" |
| 4 | cause of from-scratch failure not tested | fixed: "probably" (Sec. 4.2 and A13) |
| 5 | first pilot policy fell before Gate A2 existed | fixed in A13 |
| 6 | tracking spread is narrow (83-87%) | already stated in A13 ("only a few percentage points"); body unchanged (no room) |
| 7 | "two physics tests (appendix A12)" omitted A13 | fixed |
| 8 | some saves beat the released model's E1 by about 1% | unchanged: the table in A13 gives every value |
| 9 | "0.59" scope; "0.15" for Wtf at 20 steps | unchanged: failure rule is two policies below 0.15 (mean can reach 0.15) |
| 10 | record's walk test values differ from Gate A2's | fixed: "differ a little" -> "differ" |
| 11 | "only because" too strong | fixed: "partly because" |
| 12 | "at most 0.2%" holds for sampled actions | fixed in A12 root-cause text |

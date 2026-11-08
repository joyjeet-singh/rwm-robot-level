# M6: air-claim-checker on the Step 2 text (2026-11-09)

Checked: Sec. 4.2 run-in paragraph, Table III's physics block, abstract, Limitations, Conclusion, appendices A11 and A12,
against results/s2_*, step2/model/calibration.json and prereg/PREREG_S2.md. All numbers matched. 12 findings, all about
wording, scope or labels. The two-column text has no spare line, so fixes were kept to the same length.

| # | Finding | Disposition |
|---|---|---|
| 1 | Abstract: "how policies walk" wider than the outcome; pool is copies of one simulator | Fixed: "policies' return across 9 copies of one simulator (registered)" |
| 2 | Conclusion: "walking" and no in-sample | Fixed: "physics return ... (in-sample)" |
| 3 | "Training reward" confusable with imagined return; physics return favours standing still; n=9 and unit | Fixed in part: "the MuJoCo reward (which favours standing still)"; 9 simulators in Table III's caption. Bootstrap unit not added (space; in PREREG_S2 and A12) |
| 4 | "Three registrations" now four | Fixed (Sec. 3 and Limitations; R0-R2 listed) |
| 5 | Limitations lacked in-sample and the interval width | Fixed in part: "checks in-sample"; interval width not added (space) |
| 6 | Table III caption did not say what the physics rows correlate with | Fixed in part: "PT: physics test, 9 simulators (R0: correlation, 95%)"; the block's own header names physics return |
| 7-9 | "in reverse", "right sign", "weakly" read as more than the intervals support | Fixed (A12: "in the point estimates"; "in no direction the interval supports") |
| 10 | Selection "only 0.16" | Fixed: "only" dropped; per-check intervals not added |
| 11 | "just below" the two best | Fixed: "below" |
| 12 | "standing still above walking with falls" | Fixed: "above most walking that fell" |

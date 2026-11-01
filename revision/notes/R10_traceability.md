# R10 — traceability: where each review point is handled (revised paper, after R10)

| Id | Handled where (single-column submission) |
|---|---|
| P1 | Intro ¶1 credits RWM's ablation (Sec. 4.3, App. A.4.1); Intro ¶3 and Sec. 4.1 "Training choices" call rows 1–2 an independent confirmation. Abstract: "consistent with RWM's ablation". |
| P1b | Intro ¶2 aims the critique at RAMBO's held-out one-step member selection [10] (MOPO/COMBO); Objective Mismatch [13] in Intro ¶2 and Related Work. |
| P2a | Abstract leads with training choices, pruning, uncertainty, then the residual; the growth rate's failed replication comes last but is stated. Contribution bullets in the same order. |
| P2b | Sec. 3 "Statistics and registration" says what was known (growth rate +0.90 on 57 F18 models; residual +0.93 on 55); appendix A2 in full. "Before the data existed" is gone: "before this study's measurements were run". |
| P3a | R10 also adds to the abstract and bullet 3 that the residual's F18 lead over the 8-step error fails correction for multiple tests. Five-split residual result marked "exploratory" in the abstract, bullet 3, Sec. 4.2, Table 2 "(E)", Discussion (3), Limitations, Conclusion. |
| P3b | Sec. 4.1 "Compression": "Moderate pruning (10–30%)" only; heavier pruning and unit removal raise λ1 (+0.27, +0.40/s), 4 + 3 copies unstable; Discussion (3); appendix A5, Table 8. |
| P3c | Sec. 4.2 (released model +0.70/s yet 100-step error below the floor, ~750× data, may have trained on these trajectories); Discussion (3); bullet 2 and Limitations: the stability account does not cover the released model. |
| P3d | Sec. 3 "Measures": label-free = "the history and the recorded commands but no recorded future state"; Table 2 column "Needs recorded future states". |
| P3e | Sec. 3 "Rebuild": the stale command is not resolved at 100 steps on 20 trajectories, but is on the 4 trajectories of episodes 1 and 8 (22% [6.3%, 42%]); "only" removed. R10: those are no longer called held out for the release, which may have trained on them. |
| P4a | Sec. 4.2 ¶3 (horizon-matched, exploratory); appendix A3, Table 4. |
| P4b | Sec. 4.2 ¶3 split-half (max drop 0.04); appendix A3, Table 5. |
| P4c | Discussion (3) ("tested only with recorded commands, not with commands from a policy"); Limitations; Conclusion ("not tested in closed loop with a policy"). R10 removed the duplicate at the end of Sec. 4.2 for length. |
| P4d | Sec. 3 "Held-out data" now says split {1, 8} reuses F18's models; Sec. 4.2 replication without that split; appendix A4, Table 6. |
| P5 | Discussion heading now "A checklist from this case study". Scope in abstract, Intro ¶3, Sec. 3 "Robot data", Limitations; constant-velocity baseline in Sec. 4.1 (0.45 m vs 0.51/0.84 m). |
| P6a | Sec. 3 last ¶ (two-level intervals, 1.5×, 3 of 13 verdicts lost); Limitations (also: two trajectories per episode treated as independent); appendix A4 (R10 corrected what the registered intervals resample), Tables 6–7. |
| P6b | Sec. 3 last ¶ and Sec. 4.2 (Holm, 2 of 7 survive, exploratory); R10 names the family (Q3–Q6) and says which tests are outside it (A4, Table 7 caption); Q5 rows relabelled. |
| P7a | Table 1 caption no longer counts rows. |
| P7b | Only Table 1's ratio is used; "5.85" no longer appears. |
| P7c | Table 1 has a "Models, trajectories" column naming each row's set; stale-command row moved to Sec. 3. |
| P7d | Sec. 4.3 folded into Discussion (5) and appendix A6; verdict labels only in tables, with a legend in Table 2's caption. |
| P7e | Sec. 3 "Models" (teacher-forced vs N=1 defined; width 124 = capacity-matched control) and training settings; appendix A1, Table 3. |
| P7f | Figure 2 has three panels including the residual. |
| P7g | Sec. 3: earlier registrations and code in the anonymised supplementary material, commit times stated as self-kept. Supplement itself: R11. |
| P7h | Ref. [6]: arXiv + NeurIPS 2025 workshop poster. |
| P7i | "302 versions" replaced by 101 trained simulators, 200 compressed copies and the released model. |
| P7j | AI statement says "author(s)"; R10 replaced the remaining "the author" (and "this folder's code") in appendix A2. Own-voice rewrite: R11 (REPORT). |
| RW | Related Work cites [18, 19] (recalibration), [23, 24] (teacher forcing, chaos), [25] (stable dynamics), [20–22]. |
| N1 | Sec. 4.1 "Compression" (earlier fine-tuned width-192/128 results, from-scratch comparison); appendix A5, Table 9. |
| F5 | Appendix A1–A7 in the single-column submission only (16 pp); two-column check PDF 6 pages after R10. |
| F6 | Study FT: abstract, Sec. 4.1, Table 1 row, Discussion (2), Limitations, appendix A7, Tables 10–11. |

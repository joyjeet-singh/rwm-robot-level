# Rewrite log

## W0 status: COMPLETE (2026-11-07 12:18)
- Branch rewrite from c699388 (tag air2027-rewrite-start); plan committed (ba80afa); tool and notes in 9ef9c46.
- Current paper: single 19 pp, two-column 6 pp, reserve (0.30 slot) 7 pp, abstract 198 words. slack.py: FAIL, free space 0.00 (expected).
- Files: scripts/slack.py, scripts/build_paper.py (--reserve [H]), paper/main_*.tex (\stepslot), rewrite/notes/W0_map.md, W0_evidence.md.
- Open: need to free about 0.30 textheight (one column x 19 lines) in the main text.

## W1 status: COMPLETE (2026-11-07 12:30)
- Wrote rewrite/OUTLINE.md (1,598 words, nine parts): the story, title kept, an abstract plan (draft 192 words), the section budget (target 3,400 words, cap 3,550), moves, Tables 2–3, claims C1–C18 and must-not list, the Step 2 slot at the end of Sec. 4, and the author's-voice paragraphs.
- air-stats-reviewer checked parts 3 and 7: 3 MUST FIX and 9 SHOULD FIX, all fixed; recorded in DECISIONS.md (W1).
- Commit a39cdd9 (outline, decisions).
- Open: the Step 2 abstract sentence will need about 10 words cut (see OUTLINE part 8). Fig. 2 moves to the appendix only as a last resort.

## W2 status: COMPLETE (2026-11-07 13:09)
- body.template.tex rewritten per OUTLINE: abstract 195 words, prose about 3,400 words, \stepslot at the end of Sec. 4; Table 2 restructured (F18 recorded/policy, X5, MLP/RSSM/Transformer), Table 3 new (12 registered verdicts).
- Moved to appendix A1 (models, rebuild, measures and training-choice details), A5 (pruning shifts, width removal), A6 (spread, variance collapse, released model; retitled). No renumbering.
- New scripts/rw_numbers.py (22 rw_t_* format re-prints from s1_stats.json), called after s1_numbers.
- slack.py PASS: two-column 6 pp, reserve 6 pp, free slot 0.50, abstract 195. check_claims/numbers/bib/submission pass.
- Files: rewrite/notes/W2.md, W2_claims.md (67 sentences + 2 tables, N/C marks for W3), DECISIONS.md (W2). Commit ce55843.
- Open: W3 adds/rewords check_claims assertions for rows marked N or C in W2_claims.md.

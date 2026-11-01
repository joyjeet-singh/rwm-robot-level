Last session: R1, COMPLETE (2026-11-01).
Next action: R2 per REVISION_PLAN.md.
Blockers: none for R1. Open point for R2/R3: `./py` imports numpy 2.2.6 from `.pylib` (pulled in with scipy) ahead of the venv's 1.26.4; torch 2.2.2 then raises "Numpy is not available" on any tensor↔array conversion. Any step that trains or scores a model must resolve this first (see DECISIONS.md, R1, for options and the recommendation).

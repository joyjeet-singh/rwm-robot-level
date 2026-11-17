# Option D checkpoint

**Last session:** D2, COMPLETE (see the status line in `optd/LOG.md`).

**Next action:** D3 (claims, checks, supplement, documents). Start from `optd/notes/D2.md`: it lists the new keys and the typed claims that need assertions in `scripts/check_claims.py`.

**Blockers:** none.

**State**
- Registered analysis: `results/d_stats.json`/`.txt` (not overwritten; DECISIONS D2-1). R0–R2 NOT SETTLED.
- Numbers: `scripts/d_numbers.py` (`d_*`, `c_*` keys), called last by `scripts/paper_numbers.py`.
- Paper: `paper/body.template.tex` (Sec. step2 paragraph, `tab:ladder`, verdict-table PT rows, abstract, Limitations, Conclusion, "five registrations"); `paper/appendix.template.tex` (A12 root cause; new A13 "Simulators of a known physics model" with tables `tab:app-drecord`, `tab:app-dladder`, `tab:app-dsec`, `tab:app-dsel`).
- Fit: `slack.py` PASS with no free space (reserve 0). Any added word in the body may need a matching cut; see DECISIONS D2-3 for what was compressed.
- Ladder result files `results/d/` and `results/d_ladder.*` are committed by D2.

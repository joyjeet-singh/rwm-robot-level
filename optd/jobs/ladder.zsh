#!/bin/zsh
# Option D, the ladder job (OPTIOND_PLAN.md, D1 step 10; Stage 2: the ladder). Written by D1; run by the launcher:
#   zsh optd/jobs/ladder.zsh          the job (resumable: the driver resumes from results/d/ladder/; every step skips
#                                     finished work)
# Steps:
#   1. scripts/d/ladder.py run: the driver with two workers (fine-tuning, checks, walk test, policies, measurement,
#      verdicts, the search; one line per measured policy to results/d_ladder.txt). Exit 3 = sanity gate (the driver
#      has written optd/DECISIONS.md and optd/LOG.md);
#   2. scripts/d/ladder.py fill: checks and the walk test for any tested simulator that lacks them (secondary parts
#      never block: a failure here is logged and the job goes on);
#   3. scripts/d/ladder.py final: results/d_ladder.json and the final table appended to results/d_ladder.txt.
# Progress: results/d/progress.json, rewritten by the work itself. Workers use 2 threads each. Never edit this file
# while it runs. Expected: about 40-55 hours (up to 7 runs x 3 tests x 2-3 policies of about 2 hours, 2 workers).
emulate -L zsh
set -uo pipefail
cd "${0:A:h}/../.." || exit 2

mkdir -p .cache/logs results/d/ladder
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 S2_THREADS=1
log() { print "[$(date '+%F %T')] $*" }
prog() { ./py -c "import sys; sys.path.insert(0, 'scripts/d'); import dprogress; dprogress.write('ladder', sys.argv[1])" "$1" }

if [[ -f results/d_ladder.json ]]; then
  log "results/d_ladder.json exists: the ladder job is finished"; exit 0
fi

# ---- 1. the driver ------------------------------------------------------------------------------------------------
prog "ladder driver"
log "ladder driver starts"
./py scripts/d/ladder.py run >> .cache/logs/job_d_ladder.log 2>&1; rc=$?
tail -3 .cache/logs/job_d_ladder.log
if (( rc == 3 )); then log "sanity gate: the job stops BLOCKED (optd/DECISIONS.md)"; exit 3; fi
if (( rc != 0 )); then log "the driver failed (exit $rc; .cache/logs/job_d_ladder.log)"; exit 1; fi

# ---- 2. checks and walk test for any tested simulator that lacks them ----------------------------------------------
prog "ladder: fill missing checks and walk tests"
./py scripts/d/ladder.py fill >> .cache/logs/job_d_ladder.log 2>&1 || log "fill: a check or walk test still fails (see the log); going on"

# ---- 3. the ladder's results --------------------------------------------------------------------------------------
prog "ladder: final table"
./py scripts/d/ladder.py final >> .cache/logs/job_d_ladder.log 2>&1 || { log "final table failed"; exit 1 }
prog "ladder job done"
log "ladder job done: results/d_ladder.json, results/d_ladder.txt"
exit 0

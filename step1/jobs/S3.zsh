#!/bin/zsh
# Job S3 (STEP1_PLAN.md S3 step 7): the measurements of studies CL and AR (prereg/PREREG_S1.md).
# One command, ./py scripts/s1_eval.py, in four phases: 1 checks (tests i, vi, ix), 2 CL scoring (F18, X5, released
# model), 3 training of the 12 new baseline runs (one at a time, 2 threads), 4 AR scoring. The driver is resumable:
# rerunning this job skips every finished model and run. Its full output goes to a log in .cache/logs/; this script
# prints one line per phase start and end and exits non-zero on failure. Only the driver writes
# results/s1/progress.json.
emulate -L zsh
setopt pipe_fail
cd "${0:A:h:h:h}" || exit 1
[[ -f prereg/PREREG_S1.md && -f scripts/s1_eval.py && -x ./py ]] || { print "S3 job: not in AIR 2027"; exit 1; }
mkdir -p .cache/logs results/s1
log=.cache/logs/job_S3_driver_$(date +%Y%m%d-%H%M%S).log
print "S3 job: start $(date '+%Y-%m-%d %H:%M:%S'); driver log $log"

t0=$SECONDS
./py scripts/s1_eval.py 2>&1 | tee -a $log | grep --line-buffered -E '^(== phase|checks:)'
rc=$?
if (( rc != 0 )); then
  print "S3 job: FAILED (exit $rc) after $(( SECONDS - t0 )) s; see $log"
  tail -5 $log
  exit 1
fi
print "S3 job: done in $(( SECONDS - t0 )) s"

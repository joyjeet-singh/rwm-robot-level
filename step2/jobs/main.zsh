#!/bin/zsh
# Step 2 main job (written by M4; run by the launcher, never by a session). Resumable: every step skips finished
# work. Registered in prereg/PREREG_S2.md: the fallback pool (the released model and its eight compressed copies),
# policy seeds 1 and 2, 18 policies.
#   0. the M4 tests (scripts/s2/test_s2.py, about 1-2 minutes);
#   1. scripts/s2_main.py: each policy trained (scripts/s2/train_policy.py --route rebuilt, 500 iterations, 4,096
#      robots), exported and measured in the calibrated MuJoCo model (scripts/s2/evaluate.py), by
#      2 worker(s) of 2 thread(s) each (no session runs during this job; choice: step2/notes/M4.md);
#   2. results/s2_main.json once all 18 are measured. The registered analysis (scripts/s2_stats.py) is run by M5.
# Progress: results/s2/progress.json, rewritten by the training every 25 iterations (and after any iteration more
# than 5 minutes after the last write), by the evaluation every five episodes, and at each policy's start and end.
# Expected time: about 20 hours (18 policies of about 2 hours, two at a time, plus about 3 minutes of measuring each).
emulate -L zsh
set -u
here=${0:A:h:h:h}
cd "$here" || exit 1
mkdir -p results/s2 .cache/logs runs/s2
PY=./py
PYMJ=./py-mj
WORKERS=2
THREADS=2
progress() { print -r -- "{\"job\":\"main\",\"step\":\"$1\",\"time\":\"$(date +%FT%T)\"}" > results/s2/progress.json; }
say() { print -r -- "[$(date +%T)] $*"; }

if [[ -e results/s2_main.json ]]; then
  say "results/s2_main.json exists; nothing to do"
  exit 0
fi
git ls-files --error-unmatch prereg/PREREG_S2.md > /dev/null 2>&1 || { say "prereg/PREREG_S2.md is not committed"; exit 1 }

progress tests
OMP_NUM_THREADS=2 S2_THREADS=2 $PYMJ scripts/s2/test_s2.py > .cache/logs/s2_main_tests.log 2>&1
tail -2 .cache/logs/s2_main_tests.log
grep -q '^ALL PASS' .cache/logs/s2_main_tests.log || { say "M4 tests failed (.cache/logs/s2_main_tests.log)"; exit 1 }

say "main: $WORKERS worker(s) x $THREADS thread(s)"
progress "main start"
$PY scripts/s2_main.py --workers $WORKERS --threads $THREADS > .cache/logs/s2_main_run.log 2>&1
rc=$?
tail -5 .cache/logs/s2_main_run.log
$PY scripts/s2_main.py --plan
(( rc == 0 )) || { say "some policies were not measured (.cache/logs/s2_main_*.log); re-run to retry them"; exit 1 }
progress done
say "main job done: results/s2_main.json"

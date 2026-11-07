#!/bin/zsh
# Step 2 calibration job (written by M2; run by the launcher, never by a session). Resumable: every step skips
# finished work. Order (STEP2_PLAN.md M2):
#   0. the M2 tests (about 10 s; nothing is written);
#   1. calibrate: the 108-point grid on training episodes only, four single-thread workers (results/s2/calib/);
#   2. freeze the constants (step2/model/calibration.json, step2/model/anymal_d_cal.xml, read-only);
#   3. fidelity (a) on the held-out trajectories, uncalibrated then calibrated model;
#   4. (b) the authors' policy on the evaluation protocol (both models) and (c) the recorded reference;
#   5. results/s2_fidelity.json and results/s2_fidelity.txt.
# Progress: results/s2/progress.json, rewritten by the work after every grid point, held-out trajectory and five
# policy episodes.
emulate -L zsh
set -u
here=${0:A:h:h:h}
cd "$here" || exit 1
mkdir -p results/s2 .cache/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 S2_THREADS=2
PY=./py-mj
progress() { print -r -- "{\"job\":\"calib\",\"step\":\"$1\",\"time\":\"$(date +%FT%T)\"}" > results/s2/progress.json; }
say() { print -r -- "[$(date +%T)] $*"; }

progress tests
$PY scripts/s2/test_fid.py > .cache/logs/s2_calib_tests.log 2>&1
tail -6 .cache/logs/s2_calib_tests.log
grep -q '^ALL PASS' .cache/logs/s2_calib_tests.log || { say "M2 tests failed"; exit 1 }

if [[ ! -e step2/model/calibration.json ]]; then
  progress "calibrate grid"
  say "calibration grid: 4 workers"
  pids=()
  for i in 0 1 2 3; do
    $PY scripts/s2/calibrate.py --worker $i --workers 4 >> .cache/logs/s2_calib_w$i.log 2>&1 &
    pids+=($!)
  done
  rc=0
  for p in $pids; do wait $p || rc=1; done
  for i in 0 1 2 3; do tail -2 .cache/logs/s2_calib_w$i.log; done
  (( rc == 0 )) || { say "a calibration worker failed (.cache/logs/s2_calib_w*.log)"; exit 1 }
  say "freeze"
  $PY scripts/s2/calibrate.py --freeze || { say "freeze failed"; exit 1 }
else
  say "calibration already frozen"
fi

for part in a_uncal a_cal b_uncal b_cal c; do
  say "fidelity part $part"
  $PY scripts/s2/fidelity.py --part $part || { say "fidelity part $part failed"; exit 1 }
done
say "report"
$PY scripts/s2/fidelity.py --part report || { say "report failed"; exit 1 }
progress done
say "calib job done: results/s2_fidelity.json, results/s2_fidelity.txt"

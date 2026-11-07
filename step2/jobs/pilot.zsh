#!/bin/zsh
# Step 2 pilot job (written by M3; run by the launcher, never by a session). Resumable: every step skips finished
# work. Order (STEP2_PLAN.md M3, Gate 2):
#   0. the M3 tests (about 40 s);
#   1. train the four pilot policies, seed 0 each, 500 iterations, 4,096 robots (scripts/s2/train_policy.py):
#      two workers of two threads each, run in parallel (4 cores; about 1.5 GB each);
#        worker 0: REL, then A_s0__u50;   worker 1: A_s0, then B_s0;
#   2. measure each exported actor in the calibrated MuJoCo model (scripts/s2/evaluate.py), four single-thread
#      workers in parallel;
#   3. results/s2_pilot.json and results/s2_pilot.txt (scripts/s2/pilot_report.py).
# Progress: results/s2/progress.json, rewritten by the training every 25 iterations (and after any iteration more
# than 5 minutes after the last write), and by the evaluation every five episodes.
# Expected time: about 4-5 hours (two policies of about 2 hours per worker), plus about 10 minutes of evaluation.
emulate -L zsh
set -u
here=${0:A:h:h:h}
cd "$here" || exit 1
mkdir -p results/s2 .cache/logs runs/s2
PY=./py
PYMJ=./py-mj
progress() { print -r -- "{\"job\":\"pilot\",\"step\":\"$1\",\"time\":\"$(date +%FT%T)\"}" > results/s2/progress.json; }
say() { print -r -- "[$(date +%T)] $*"; }

if [[ -e results/s2_pilot.json ]]; then
  say "pilot report exists; nothing to do"
  exit 0
fi

progress tests
OMP_NUM_THREADS=2 S2_THREADS=2 $PYMJ scripts/s2/test_m3.py > .cache/logs/s2_pilot_tests.log 2>&1
tail -2 .cache/logs/s2_pilot_tests.log
grep -q '^ALL PASS' .cache/logs/s2_pilot_tests.log || { say "M3 tests failed (.cache/logs/s2_pilot_tests.log)"; exit 1 }

train_list() {   # $1 = worker id; the rest = simulator keys, trained in order
  local w=$1; shift
  local s
  for s in "$@"; do
    print -r -- "[$(date +%T)] worker $w: train $s" >> .cache/logs/s2_pilot_train_w$w.log
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
      $PY scripts/s2/train_policy.py --sim $s --seed 0 --threads 2 --job pilot \
      > .cache/logs/s2_pilot_train_${s}.log 2>&1 || {
        print -r -- "[$(date +%T)] worker $w: $s FAILED" >> .cache/logs/s2_pilot_train_w$w.log; return 1 }
    tail -1 .cache/logs/s2_pilot_train_${s}.log >> .cache/logs/s2_pilot_train_w$w.log
  done
}

say "training: 2 workers x 2 threads"
progress "train start"
train_list 0 REL A_s0__u50 &
p0=$!
train_list 1 A_s0 B_s0 &
p1=$!
rc=0
wait $p0 || rc=1
wait $p1 || rc=1
tail -3 .cache/logs/s2_pilot_train_w0.log .cache/logs/s2_pilot_train_w1.log
(( rc == 0 )) || { say "a training worker failed (.cache/logs/s2_pilot_train_*.log)"; exit 1 }

say "evaluation: 4 workers x 1 thread"
pids=()
for pol in REL__p0 A_s0__p0 B_s0__p0 A_s0__u50__p0; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 S2_THREADS=1 \
    $PYMJ scripts/s2/evaluate.py --policy $pol --job pilot > .cache/logs/s2_pilot_eval_${pol}.log 2>&1 &
  pids+=($!)
done
rc=0
for p in $pids; do wait $p || rc=1; done
for pol in REL__p0 A_s0__p0 B_s0__p0 A_s0__u50__p0; do tail -1 .cache/logs/s2_pilot_eval_${pol}.log; done
(( rc == 0 )) || { say "an evaluation failed (.cache/logs/s2_pilot_eval_*.log)"; exit 1 }

say "report"
$PY scripts/s2/pilot_report.py > .cache/logs/s2_pilot_report.log 2>&1 || { tail -5 .cache/logs/s2_pilot_report.log; say "report failed"; exit 1 }
cat results/s2_pilot.txt
progress done
say "pilot job done: results/s2_pilot.json, results/s2_pilot.txt"

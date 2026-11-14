#!/bin/zsh
# Option D, the record job (OPTIOND_PLAN.md, D0 step 6; Stage 1: the record). Written by D0; run by go_d.zsh:
#   zsh optd/jobs/record.zsh                 the job (resumable: every step skips finished work)
#   REHEARSAL=1 zsh optd/jobs/record.zsh     the same chain at toy size, everything under .cache/d_rehearsal/
# Steps:
#   1. RWM's released policy measured; sanity gate 1: its walking score equals results/c_authors.json's to 1e-6;
#   2. Step 2's 22 policies measured (two workers); sanity gate 2: REL's two registered policies average >= 0.3;
#   3. checks (primary) and the walk test (secondary) of the 17 simulators (two workers);
#   4. scripts/d/record.py: results/d_record.json and results/d_record.txt.
# A sanity gate that fails writes optd/DECISIONS.md and optd/LOG.md and the job exits 3. Any other failure exits 1.
# Progress: results/d/progress.json, rewritten by the work itself (at least every few minutes).
# Workers use 2 threads each (MuJoCo evaluation is single-threaded). Never edit this file while it runs.
emulate -L zsh
set -uo pipefail
cd "${0:A:h}/../.." || exit 2

R=${REHEARSAL:-0}
SIMS=(REL REL__fp16 REL__u10 REL__u20 REL__u30 REL__u50 REL__int8w REL__s192 REL__s128
      A_s0 B_s0 A_s0__u50 A5_s0 sim_6M_N8_s0 sim_6M_N8_s0__2S sim_10k_N8_s0 sim_10k_N8_s0__2S)
S2=(REL REL__fp16 REL__u10 REL__u20 REL__u30 REL__u50 REL__int8w REL__s192 REL__s128)
POL=()
for s in $S2; do POL+=(${s}__p1 ${s}__p2); done
POL+=(REL__p0 A_s0__p0 B_s0__p0 A_s0__u50__p0)                 # 22: 18 registered, 4 pilot
GATE=()
if (( R )); then
  B=.cache/d_rehearsal
  EVAL=$B/eval CHK=$B/checks WALK=$B/walk OUT=$B/d_record
  export D_PROGRESS=$B/progress.json
  POL=(REL__p1)
  SIMS=(REL)
  GATE=(--rehearsal)
  mkdir -p $EVAL $CHK $WALK
else
  EVAL=results/d/eval CHK=results/d/checks WALK=results/d/walk OUT=results/d_record
fi
mkdir -p .cache/logs results/d
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 S2_THREADS=1
log() { print "[$(date '+%F %T')] $*" }
prog() { ./py -c "import sys; sys.path.insert(0, 'scripts/d'); import dprogress; dprogress.write('record', sys.argv[1])" "$1" }

# ---- 1. RWM's released policy, then sanity gate 1 --------------------------------------------------------------
prog "measure RWM's released policy"
if (( R )); then
  ./py-mj scripts/d/evaluate.py --policy _test_rehearsal_authors --actor ../robotic_world_model_lite/assets/models/pretrain_rnn_ens.pt \
    --schedules 0,1 --starts 1 --no-recorded --out $EVAL/authors.json --job record || { log "authors' policy: evaluation failed"; exit 1 }
else
  ./py-mj scripts/d/evaluate.py --policy authors --job record || { log "authors' policy: evaluation failed"; exit 1 }
fi
./py scripts/d/record.py gate1 --eval-dir $EVAL $GATE; rc=$?
(( rc == 3 )) && exit 3
(( rc != 0 )) && { log "gate 1 could not be computed"; exit 1 }

# ---- 2. Step 2's 22 policies, two workers; then sanity gate 2 ----------------------------------------------------
evaluate_list() {     # $1 worker id, then policy names; one after another
  local w=$1; shift
  local p
  for p in "$@"; do
    if (( R )); then
      ./py-mj scripts/d/evaluate.py --policy _test_rehearsal_$p --actor runs/s2/$p/actor.pt --schedules 0,1 --starts 1 \
        --out $EVAL/$p.json --job record >> .cache/logs/job_d_record_w$w.log 2>&1 || return 1
    else
      ./py-mj scripts/d/evaluate.py --policy $p --job record >> .cache/logs/job_d_record_w$w.log 2>&1 || return 1
    fi
    log "worker $w: measured $p"
  done
}
A=() Bw=()
for (( i = 1; i <= $#POL; i++ )); do
  if (( i % 2 )); then A+=($POL[i]); else Bw+=($POL[i]); fi
done
prog "measure Step 2's policies"
evaluate_list 1 $A & p1=$!
evaluate_list 2 $Bw & p2=$!
wait $p1; r1=$?
wait $p2; r2=$?
if (( r1 || r2 )); then log "a policy evaluation failed (see .cache/logs/job_d_record_w1.log, _w2.log)"; exit 1; fi
if (( R )); then cp $EVAL/REL__p1.json $EVAL/REL__p2.json.tmp && mv $EVAL/REL__p2.json.tmp $EVAL/REL__p2.json; fi
./py scripts/d/record.py gate2 --eval-dir $EVAL $GATE; rc=$?
(( rc == 3 )) && exit 3
(( rc != 0 )) && { log "gate 2 could not be computed"; exit 1 }

# ---- 3. checks (primary) and the walk test (secondary) of the simulators, two workers ------------------------------
sim_list() {          # $1 worker id, then simulator names
  local w=$1; shift
  local s
  for s in "$@"; do
    if (( R )); then
      ./py scripts/d/checks.py --sim $s --max-windows 4 --out $CHK/$s.json --job record >> .cache/logs/job_d_record_w$w.log 2>&1 \
        || { log "checks of $s failed"; return 1 }
      ./py scripts/d/walk_test.py --sim $s --robots 64 --steps 16 --out $WALK/$s.json --job record \
        >> .cache/logs/job_d_record_w$w.log 2>&1 || print -- "- $s" >> $B/walk_failed.txt
    else
      ./py scripts/d/checks.py --sim $s --job record >> .cache/logs/job_d_record_w$w.log 2>&1 \
        || { log "checks of $s failed"; return 1 }
      ./py scripts/d/walk_test.py --sim $s --job record >> .cache/logs/job_d_record_w$w.log 2>&1 \
        || { log "walk test of $s failed (secondary)"; print -- "- $s ($(date '+%F %H:%M'))" >> results/d/walk_failed.txt }
    fi
    log "worker $w: checked $s"
  done
}
A=() Bw=()
for (( i = 1; i <= $#SIMS; i++ )); do
  if (( i % 2 )); then A+=($SIMS[i]); else Bw+=($SIMS[i]); fi
done
prog "checks and walk tests"
sim_list 1 $A & p1=$!
sim_list 2 $Bw & p2=$!
wait $p1; r1=$?
wait $p2; r2=$?
if (( r1 || r2 )); then log "checks failed (see .cache/logs/job_d_record_w1.log, _w2.log)"; exit 1; fi
if (( ! R )) && [[ -s results/d/walk_failed.txt ]]; then
  { print "\n## Record job: walk test left out for some simulators ($(date '+%F %H:%M'))"
    print "The walk test is secondary (Rule 2); these simulators' walk tests failed and are left out of the record:"
    cat results/d/walk_failed.txt } >> optd/DECISIONS.md
fi

# ---- 4. the record ----------------------------------------------------------------------------------------------
prog "record table"
if (( R )); then
  ./py scripts/d/record.py table --eval-dir $EVAL --checks-dir $CHK --walk-dir $WALK --out $OUT --sims REL \
    || { log "record.py failed"; exit 1 }
else
  ./py scripts/d/record.py table --strict || { log "record.py failed"; exit 1 }
  ./py -c "import sys; sys.path.insert(0, 'scripts/d'); import dprogress; dprogress.write('record', 'done', 1, 1)"
  print "\n## Record job status: COMPLETE ($(date '+%F %H:%M'))\nresults/d_record.txt and .json written; gates 1 and 2 passed." >> optd/LOG.md
fi
log "record job finished"
exit 0

#!/bin/zsh
# Option C pilot job (written by C2; run by go_c.zsh, never by a session).
# 1. train the two pilot policies (6M and 10k simulators, rollout-trained seed 0) to 2,000 iterations on two workers, saving
#    at 500, 1,000, 1,500 and 2,000; 2. measure every save and the authors' policy in MuJoCo; 3. apply Gate B and the retry
#    rule (scripts/c/pilot_rule.py); 4. if the rule calls for it, train and measure the two retry policies;
# 5. write results/c_pilot.json and results/c_pilot.txt.
# Every part skips finished work, so this file can be re-run after an interruption. A BLOCKED pilot exits 3 and the launcher
# stops. Progress: results/c/progress.json, from the parts themselves.
set -u
setopt pipefail
cd "${0:A:h:h:h}" || exit 1

say() { print "[$(date '+%F %T')] $*" }

run() {
  say "start: $*"
  "$@"
  local rc=$?
  (( rc == 0 )) && say "ok: $*" || say "FAILED ($rc): $*"
  return $rc
}

both() {                     # two commands at once; fails if either fails
  local -a pids
  local p rc=0
  run ${=1} &
  pids+=($!)
  run ${=2} &
  pids+=($!)
  for p in $pids; do wait $p || rc=1; done
  return $rc
}

TRAIN="./py scripts/c/train_policy.py --seed 0 --threads 2 --job pilot"
EVAL="./py-mj scripts/c/evaluate.py --job pilot"
SAVES=(500 1000 1500 2000)
P6=sim_6M_N8_s0__p0
P10=sim_10k_N8_s0__p0
export S2_THREADS=1

both "$TRAIN --sim sim_6M_N8_s0" "$TRAIN --sim sim_10k_N8_s0" || exit 1

run ${=EVAL} --policy authors || exit 1
for n in $SAVES; do
  both "$EVAL --policy $P6 --iters $n" "$EVAL --policy $P10 --iters $n" || exit 1
done

DECISION=$(./py scripts/c/pilot_rule.py decide | tail -1) || exit 1
say "retry weights: $DECISION"
if [[ $DECISION != none ]]; then
  W=(${=DECISION})
  (( ${#W} == 2 )) || { say "FAILED: expected two retry weights, got: $DECISION"; exit 1 }
  both "$TRAIN --sim sim_6M_N8_s0 --uncertainty-weight $W[1]" "$TRAIN --sim sim_6M_N8_s0 --uncertainty-weight $W[2]" || exit 1
  for n in $SAVES; do
    both "$EVAL --policy sim_6M_N8_s0__w$W[1]__p0 --iters $n" "$EVAL --policy sim_6M_N8_s0__w$W[2]__p0 --iters $n" || exit 1
  done
fi

run ./py scripts/c/pilot_rule.py report
rc=$?
if (( rc == 3 )); then say "pilot BLOCKED: see optc/DECISIONS.md and results/c_pilot.txt"; exit 3; fi
(( rc == 0 )) || exit 1
say "pilot job finished"

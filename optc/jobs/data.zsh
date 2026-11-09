#!/bin/zsh
# Option C data job (written by C1; run by go_c.zsh, never by a session).
# 1. generate the MuJoCo data with two workers; 2. manifest, nested rungs and rung files;
# 3. train the two pilot simulators (6M and 10k, rollout-trained, seed 0); 4. their checks on both test sets.
# S and the training layout come from results/c/probe.json. Every part skips finished work, so this file can be re-run
# after an interruption. Progress: results/c/progress.json, from the parts themselves.
set -u
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

GEN="./py-mj scripts/c/gen_data.py"
both "$GEN --worker 0 --nworkers 2" "$GEN --worker 1 --nworkers 2" || exit 1
run ${=GEN} --finish || exit 1

S=$(./py -c "import json; print(json.load(open('results/c/probe.json'))['S'])") || exit 1
PAR=$(./py -c "import json; print(int(json.load(open('results/c/probe.json'))['parallel_two_trainings']))") || exit 1
say "S = $S, two trainings at once = $PAR"
[[ -n "$S" && -n "$PAR" ]] || exit 1

TRAIN="./py scripts/c/train_sim.py --N 8 --seed 0 --steps $S"
if (( PAR )); then
  both "$TRAIN --rung 6M --threads 2" "$TRAIN --rung 10k --threads 2" || exit 1
else
  run ${=TRAIN} --rung 6M --threads 4 || exit 1
  run ${=TRAIN} --rung 10k --threads 4 || exit 1
fi

run ./py scripts/c/sim_checks.py --sims sim_6M_N8_s0,sim_10k_N8_s0 --out results/c/pilot_sims.json --threads 4 || exit 1
say "data job finished"

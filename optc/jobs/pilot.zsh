#!/bin/zsh
# Option C second pilot job (Amendment 1; written by C2R, run by go_c.zsh, never by a session). It replaces the first
# pilot's job, whose log is .cache/logs/job_c_pilot_first.log.
# 0. Gate A2 (results/c/calibration.json, committed by C2R). If it failed: retrain the two pilot simulators with 2 x S
#    steps (one at a time, 4 threads, as the data job), redo their checks and calibration, and apply Gate A2 again;
#    a second failure is BLOCKED (exit 3).
# 1. Arms P1 and P2 on two workers, then P3 and P4, in the 6M simulator, seed 0, 1,500 iterations, saves at 500, 1,000
#    and 1,500; every save is measured as soon as its arm finishes (an arm stopped early: its saves so far).
# 2. Measure the first pilot's surviving saves and add them to results/c_pilot1.*.
# 3. Gate B' and the iteration rule' (exit 3 if BLOCKED); then the 10k policy with the chosen recipe and iteration count,
#    measured, for Gate C'; 4. results/c_pilot.json and .txt.
# Every part skips finished work, so this file can be re-run after an interruption. Progress: results/c/progress.json,
# from the parts themselves. Never edit this file while it runs.
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

RULE="./py scripts/c/pilot2_rule.py"
EVAL="./py-mj scripts/c/evaluate.py --job pilot"
TRAIN="./py scripts/c/train_policy.py --seed 0 --job pilot"
export S2_THREADS=1

gate_a2() { ${=RULE} a2 | tail -1 }

# ---------------------------------------------------------------- 0. Gate A2
A2=$(gate_a2); rc=$?
if (( rc == 3 )); then say "pilot BLOCKED at Gate A2: see optc/DECISIONS.md"; exit 3; fi
(( rc == 0 )) || { say "FAILED: Gate A2 rule"; exit 1 }
say "Gate A2: $A2"
if [[ $A2 == retrain ]]; then
  S=$(./py -c "import json; print(json.load(open('results/c/probe.json'))['S'])") || exit 1
  [[ -n $S ]] || exit 1
  S2=$(( 2 * S ))
  say "Gate A2 failed: retraining the pilot simulators with 2 x S = $S2 steps"
  TS="./py scripts/c/train_sim.py --N 8 --seed 0 --steps $S2 --threads 4"
  run ${=TS} --rung 6M --name sim_6M_N8_s0__2S || exit 1
  run ${=TS} --rung 10k --name sim_10k_N8_s0__2S || exit 1
  run ./py scripts/c/sim_checks.py --sims sim_6M_N8_s0__2S,sim_10k_N8_s0__2S --out results/c/pilot_sims_2S.json \
      --threads 4 || exit 1
  run ./py scripts/c/calibrate_penalty.py imagine --sims REL,sim_6M_N8_s0__2S,sim_10k_N8_s0__2S \
      --gate-sim sim_6M_N8_s0__2S --out results/c/calibration_2S.json --threads 2 --job pilot
  rc=$?
  (( rc == 0 || rc == 4 )) || exit 1          # 4: Gate A2 failed, decided below
  A2=$(gate_a2); rc=$?
  if (( rc == 3 )); then say "pilot BLOCKED at Gate A2 (after 2 x S): see optc/DECISIONS.md"; exit 3; fi
  (( rc == 0 )) || exit 1
  [[ $A2 != retrain ]] || { say "FAILED: Gate A2 still asks for retraining"; exit 1 }
  say "Gate A2 passed after 2 x S: S becomes 2 x S for every simulator (C3 records it)"
fi
SIMS=(${=A2})
(( ${#SIMS} == 2 )) || { say "FAILED: expected two simulators, got: $A2"; exit 1 }
SIM6=$SIMS[1]
SIM10=$SIMS[2]
say "pilot simulators: $SIM6, $SIM10"

# ---------------------------------------------------------------- 1. the four arms
arm() {                      # arm <penalty> <fall cost> <policy name>: train, then measure every save it kept
  local pen=$1 fall=$2 name=$3 n
  run ${=TRAIN} --threads 2 --sim $SIM6 --penalty $pen --fall-cost $fall --max-iters 1500 || return 1
  local kept
  kept=$(${=RULE} saves $name | tail -1) || return 1
  say "$name: saves kept: ${kept:-none}"
  for n in ${=kept}; do
    run ${=EVAL} --policy $name --iters $n || return 1
  done
}

both "arm cal 0 ${SIM6}__cal__p0" "arm cal 4 ${SIM6}__cal_fall4__p0" || exit 1      # P1, P2
both "arm 0 0 ${SIM6}__w0__p0" "arm 0 4 ${SIM6}__w0_fall4__p0" || exit 1            # P3, P4

# ---------------------------------------------------------------- 2. the first pilot's surviving saves
W2=sim_6M_N8_s0__w2__p0
W4=sim_6M_N8_s0__w4__p0
for n in 500 1000 1500; do
  both "$EVAL --policy $W2 --iters $n" "$EVAL --policy $W4 --iters $n" || exit 1
done
run ${=EVAL} --policy $W4 --iters 2000 || exit 1
run ./py scripts/c/pilot1_record.py --measured || exit 1

# ---------------------------------------------------------------- 3. Gate B', iteration rule', Gate C'
CH=$(${=RULE} choose --sims "$SIM6 $SIM10" | tail -1); rc=$?
if (( rc == 3 )); then say "pilot BLOCKED at Gate B': see optc/DECISIONS.md and results/c_pilot.txt"; exit 3; fi
(( rc == 0 )) || { say "FAILED: Gate B' rule"; exit 1 }
R=(${=CH})
(( ${#R} == 4 )) || { say "FAILED: expected '<penalty> <fall> <iters> <policy>', got: $CH"; exit 1 }
say "Gate B': penalty $R[1], fall cost $R[2], $R[3] iterations; Gate C' policy $R[4]"
run ${=TRAIN} --threads 4 --sim $SIM10 --penalty $R[1] --fall-cost $R[2] --max-iters $R[3] --no-early-stop || exit 1
run ${=EVAL} --policy $R[4] --iters $R[3] || exit 1

run ${=RULE} report --sims "$SIM6 $SIM10"
rc=$?
if (( rc == 3 )); then say "pilot BLOCKED: see optc/DECISIONS.md and results/c_pilot.txt"; exit 3; fi
(( rc == 0 )) || exit 1
say "pilot job finished"

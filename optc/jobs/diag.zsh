#!/bin/zsh
# Option C diagnosis job (written by C0; run by go_c.zsh, never by a session). D1-D4 through scripts/c/diag.py.
# Two workers x two threads. Every part skips finished work, so this file can be re-run after an interruption.
# Progress: results/c/progress.json, from the parts themselves. Output: results/c_diag.json, results/c_diag.txt.
set -u
cd "${0:A:h:h:h}" || exit 1

T="--threads 2"

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

pairs() {                    # the listed commands, two at a time, in order
  while (( $# >= 2 )); do
    both "$1" "$2" || return 1
    shift 2
  done
  if (( $# == 1 )); then run ${=1} || return 1; fi
  return 0
}

MJ="./py-mj scripts/c/diag.py"
PY="./py scripts/c/diag.py"
PILOT=(REL__p0 A_s0__p0 B_s0__p0 A_s0__u50__p0)
D3=(A5_s0__p0 A5_s0__p1)

say "diag job: start"

# D1 physics, and the authors' policy for D2: Step 2's 80 episodes each
tasks=()
for p in $PILOT authors; do tasks+=("$MJ --part phys --policy $p $T"); done
pairs $tasks || exit 1

# D3: two policies in the five-network A5_s0, then Step 2's evaluator
pairs "$PY --part d3train --seed 0 $T" "$PY --part d3train --seed 1 $T" || exit 1
pairs "$MJ --part d3eval --seed 0 $T" "$MJ --part d3eval --seed 1 $T" || exit 1

# D1 imagination (pilot policies and D3's)
tasks=()
for p in $PILOT $D3; do tasks+=("$PY --part imag --policy $p $T"); done
pairs $tasks || exit 1

# D1 real falls and D2 (they need the physics and imagination files above)
tasks=()
for p in $PILOT; do tasks+=("$PY --part falls --policy $p $T"); done
for p in $PILOT; do tasks+=("$PY --part d2 --policy $p $T"); done
pairs $tasks || exit 1

run ${=PY} --part d4 ${=T} || exit 1

# The report is written once; on a re-run it is already there.
if [[ -e results/c_diag.json && -e results/c_diag.txt ]]; then
  say "report exists, not rewritten"
else
  run ${=PY} --part report ${=T} || exit 1
fi

say "diag job: done"

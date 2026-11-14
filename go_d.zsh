#!/bin/zsh
# AIR 2027 Option D launcher (OPTIOND_PLAN.md, early-failure form): the record of what exists, then a searched ladder of adapted simulators.
# Runs from its own folder wherever it is called from.
#   caffeinate -i ./go_d.zsh auto      every remaining session and job, in order, unattended (safe to re-run)
#   ./go_d.zsh status                  sessions, jobs, job progress and the latest ladder results
#   ./go_d.zsh D2                      one session, interactive
#   ./go_d.zsh D2 resume               resume a PARTIAL or BLOCKED session, interactive
#   ./go_d.zsh job ladder              start (or resume) one job in the background and wait for it
# Jobs (record, ladder) run after the session that writes them, one at a time. The launcher sets no time limit (the
# ladder job caps itself, as the plan says); a job is
# stopped only when results/d/progress.json and its log stay unchanged for stall_checks checks in a row (12 x 5 min).
# A session is tried at most `retries` times (3): the plan's sessions stop BLOCKED rather than keep failing.
# A job that exits with code 3 has stopped at one of the plan's sanity gates (BLOCKED): read optd/DECISIONS.md.
# Sessions must leave everything in ~/Downloads/PDM outside "AIR 2027" unchanged (revision/boundary.zsh). Jobs run for
# days, so they check only what their code could change by mistake: PDM's own files and the code folders in jdirs.
# When a session stops at Claude's usage limit, the launcher waits for the reset time the message gives, then resumes.
# Optional overrides in optd/driver.conf (zsh syntax), e.g.  mode=acceptEdits  retries=5  stall_checks=24  deny_flag=0
# Never edit this file while it runs.
emulate -L zsh
set -uo pipefail
cd "${0:A:h}" || exit 2

PLAN=OPTIOND_PLAN.md
D=optd
order=(D0 D1 D2 D3 D4)
typeset -A model effort turns after
model=(D0 claude-opus-5-5  D1 claude-opus-5-5  D2 claude-opus-5-5  D3 claude-sonnet-5-5
       D4 claude-opus-5-5)
effort=(D0 high D1 high D2 high D3 high D4 high)
turns=(D0 200 D1 200 D2 150 D3 120 D4 100)
after=(D0 record D1 ladder)   # the job each session writes, run before the next session
branch=optd        # every session after D0 starts on this branch
base_branch=optc   # D0 starts here and creates the Option D branch
progress=results/d/progress.json
jdirs=(rwm_repro robotic_world_model_lite rsl_rl_rwm rwm_analysis)   # sibling code folders a job reads
mode=auto
retries=3
retry_wait=1800    # seconds to wait before the third and later attempts of a session
poll=300           # seconds between checks on a job
stall_checks=12    # checks in a row with no change in the progress file or the job log: the job counts as stalled
limit_margin=300   # wait this long past a usage limit's reset time
limit_waits=10     # at most this many usage-limit waits per session
name_flag=1
deny_flag=1
deny=("Bash(./go_d.zsh:*)" "Bash(zsh go_d.zsh:*)" "Bash(zsh ./go_d.zsh:*)" "Bash(./go_c.zsh:*)" "Bash(./go_s2.zsh:*)" "Bash(./go_im.zsh:*)"
      "Bash(./go_rw.zsh:*)" "Bash(./go_s1.zsh:*)" "Bash(./go_rev.zsh:*)" "Bash(caffeinate:*)" "Bash(nohup:*)"
      "Bash(zsh optd/jobs:*)" "Bash(zsh ./optd/jobs:*)" "Bash(./optd/jobs:*)" "Bash(bash optd/jobs:*)"
      "Bash(pip:*)" "Bash(pip3:*)" "Bash(git clone:*)" "Bash(git push:*)")
[[ -r $D/driver.conf ]] && source $D/driver.conf
mkdir -p .cache/logs .cache/done .cache/boundary $D/jobs $D/notes $D/review results/d

now() { date '+%F %T' }
last_out=""

epoch_at() {          # $1 "YYYY-MM-DD HH:MM", $2 time zone or "": seconds since 1970 (macOS date, else GNU date)
  if [[ -n ${2:-} ]]; then
    TZ=$2 date -j -f '%Y-%m-%d %H:%M' "$1" +%s 2>/dev/null || TZ=$2 date -d "$1" +%s 2>/dev/null
  else
    date -j -f '%Y-%m-%d %H:%M' "$1" +%s 2>/dev/null || date -d "$1" +%s 2>/dev/null
  fi
}

limit_wait() {        # $1 session log. 0: it stopped at a usage limit and we slept past the reset; 1: no limit; 2: unreadable
  local line h m ap tz day target now_s secs
  grep -qiE "(session|usage|weekly) limit" "$1" 2>/dev/null || return 1
  now_s=$(date +%s)
  line=$(grep -oiE "limit reached\|[0-9]{9,}" "$1" | tail -1)
  if [[ -n $line ]]; then
    target=${line##*|}
  else
    line=$(grep -oiE "resets [0-9]{1,2}(:[0-9]{2})? ?[ap]m( \([^)]+\))?" "$1" | tail -1)
    [[ $line =~ 'resets ([0-9]{1,2})(:([0-9]{2}))? ?([aApP][mM])( \(([^)]+)\))?' ]] || return 2
    h=$match[1]; m=${match[3]:-00}; ap=${(L)match[4]}; tz=${match[6]:-}
    (( h == 12 )) && h=0
    [[ $ap == pm ]] && (( h += 12 ))
    if [[ -n $tz ]]; then day=$(TZ=$tz date +%F); else day=$(date +%F); fi
    target=$(epoch_at "$day ${(l:2::0:)h}:$m" "$tz")
    [[ $target == <-> ]] || return 2
    (( target <= now_s )) && (( target += 86400 ))
  fi
  [[ $target == <-> ]] && (( target < now_s + 7 * 86400 )) || return 2
  secs=$(( target > now_s ? target - now_s + limit_margin : limit_margin ))
  print "[$(now)] Claude's usage limit was reached; waiting $(( (secs + 59) / 60 )) min, until just after the reset, then resuming"
  sleep $secs
  return 0
}

log_status() {        # last status word that session $1 wrote to optd/LOG.md, or "none"
  local line
  line=$(grep -E "^## $1 status: " $D/LOG.md 2>/dev/null | tail -1)
  [[ -z $line ]] && { print none; return }
  line=${line#*status: }
  print -- ${line%% *}
}

jtag() { print -- "job_d_$1" }                   # marker and log name for job $1
job_alive() { local p f=.cache/done/$(jtag $1).pid; [[ -r $f ]] || return 1; p=$(<$f); kill -0 $p 2>/dev/null }

job_state() {         # none | pending | running | done | BLOCKED | STALLED | BOUNDARY | FAILED
  local t=.cache/done/$(jtag $1)
  if [[ -e $t ]]; then print done
  elif job_alive $1; then print running
  elif [[ -e $t.blocked ]]; then print BLOCKED
  elif [[ -e $t.stalled ]]; then print STALLED
  elif [[ -e $t.boundary ]]; then print BOUNDARY
  elif [[ -e $t.failed ]]; then print FAILED
  elif [[ -r $D/jobs/$1.zsh ]]; then print pending
  else print none; fi
}

block() { awk -v b="<!-- BEGIN $1 -->" -v e="<!-- END $1 -->" '$0==b{f=1;next} $0==e{f=0} f' "$PLAN" }

prompt_for() {        # $1 session id, $2 new|resume, $3 headless|interactive
  local sid=$1 head
  head="You are session $sid of AIR 2027 Option D in early-failure form (the record of existing results, then a searched ladder of simulators adapted from the released model to the MuJoCo ANYmal D), working in this folder."
  if [[ $3 == interactive ]]; then
    head+=" The author is present in this terminal and may answer questions; otherwise work as the plan says."
  else
    head+=" You are running unattended: never ask the user anything; follow $PLAN, or stop BLOCKED as its common rules say."
  fi
  head+=" Your instructions are the common rules and your own block, both below; the fixed design is in $PLAN under 'Design fixed by this plan'. If your context is compacted, re-read them with: awk '/BEGIN COMMON/,/END COMMON/' $PLAN ; awk '/BEGIN $sid -->/,/END $sid -->/' $PLAN"
  [[ $2 == resume ]] && head+=$'\n'"This is a resumed attempt: first read $D/CHECKPOINT.md, the last 20 lines of $D/LOG.md, your notes in $D/notes/ and git log --oneline -15; continue from the first unfinished step and never redo a finished one."
  print -r -- "$head"
  print -r -- ""
  block COMMON
  print -r -- ""
  block $sid
}

boundary() { zsh revision/boundary.zsh "$@" }

jsnap() {             # PDM's own files and the code folders a job reads: what a job could change by mistake
  ( cd .. || exit 1
    { find . -maxdepth 1 -type f ! -name .DS_Store -print0
      for d in $jdirs; do
        [[ -d $d ]] && find "$d" \( -name '.venv-*' -o -name .git -o -name __pycache__ -o -name .cache \
          -o -name .DS_Store \) -prune -o -type f -print0
      done } | xargs -0 stat -f '%m %z %N' | LC_ALL=C sort )
}

jboundary() {         # start|end <label>: revision/boundary.zsh's check, over jsnap's files only
  local f=.cache/boundary/$2
  if [[ $1 == start ]]; then jsnap > $f.start; return 0; fi
  jsnap > $f.end
  if cmp -s $f.start $f.end; then rm -f $f.diff; return 0; fi
  diff $f.start $f.end > $f.diff
  print -u2 "boundary: $(grep -c '^[<>]' $f.diff) changed entries in PDM's own files or ${(j:, :)jdirs} (see $f.diff)"
  return 1
}

check_setup() {
  local ok=1 a sid
  command -v claude >/dev/null || { print -u2 "claude (Claude Code CLI) not found on PATH"; ok=0 }
  [[ -r $PLAN ]] || { print -u2 "$PLAN missing"; ok=0 }
  for a in RESULTS_SO_FAR.md $D/REVIEW.md; do
    [[ -r $a ]] || { print -u2 "$a missing (it comes with the plan)"; ok=0 }
  done
  for sid in COMMON $order; do
    [[ -n "$(block $sid 2>/dev/null)" ]] || { print -u2 "$PLAN: block $sid missing"; ok=0 }
  done
  grep -q '^## C2R status: COMPLETE' optc/LOG.md 2>/dev/null || { print -u2 "Option C has not reached C2R (optc/LOG.md)"; ok=0 }
  for a in scripts/c/evaluate.py scripts/c/calibrate_penalty.py scripts/c/train_sim.py scripts/c/gen_data.py; do
    [[ -r $a ]] || { print -u2 "$a missing (Option C wrote it)"; ok=0 }
  done
  for a in scripts/s2/train_policy.py scripts/lite/run_lite.py scripts/compress.py results/s2_main.json results/models.json; do
    [[ -r $a ]] || { print -u2 "$a missing (Step 2 wrote it)"; ok=0 }
  done
  [[ -r revision/boundary.zsh ]] || { print -u2 "revision/boundary.zsh missing"; ok=0 }
  [[ -r .claude/settings.json ]] || { print -u2 ".claude/settings.json missing (permissions and denials; go_rev.zsh installs it)"; ok=0 }
  for a in air-reader air-bib-checker air-stats-reviewer air-claim-checker air-referee; do
    [[ -r .claude/agents/$a.md ]] || { print -u2 ".claude/agents/$a.md missing"; ok=0 }
  done
  [[ -e py && -e py-mj ]] || { print -u2 "py or py-mj missing (Step 2 makes py-mj)"; ok=0 }
  [[ -e py && ! -x py ]] && chmod +x py
  [[ -e py-mj && ! -x py-mj ]] && chmod +x py-mj
  (( ok ))
}

on_branch() {         # every session after D0 must start on the Option D branch; D0 on Option C's or Option D's
  local b
  b=$(git branch --show-current 2>/dev/null)
  [[ $b == $branch ]] && return 0
  [[ ${1:-} == D0 && $b == $base_branch ]] && return 0
  if [[ ${1:-} == D0 ]]; then
    print -u2 "git is on branch '${b:-?}': D0 starts on '$base_branch' (or resumes on '$branch'). Run  git switch $base_branch  then ./go_d.zsh auto"
  else
    print -u2 "git is on branch '${b:-?}', not '$branch': run  git switch $branch  then ./go_d.zsh auto"
  fi
  return 1
}

run_headless() {      # $1 session id, $2 new|resume
  local sid=$1 kind=$2 out=.cache/logs/session_d_${1}_${2}_$(date +%Y%m%d-%H%M%S).log rc
  local -a extra=()
  (( deny_flag )) && extra+=(--disallowedTools "${deny[@]}")
  (( name_flag )) && extra+=(--name "air-d-$sid")
  print "[$(now)] $sid: $kind, ${model[$sid]}, effort ${effort[$sid]}, max ${turns[$sid]} turns, mode $mode -> $out"
  boundary start $sid
  claude -p "$(prompt_for $sid $kind headless)" --model "${model[$sid]}" --effort "${effort[$sid]}" \
    --permission-mode "$mode" --max-turns "${turns[$sid]}" "${extra[@]}" > "$out" 2>&1
  rc=$?
  last_out=$out
  if ! boundary end $sid; then
    print "[$(now)] $sid: STOPPED: files outside AIR 2027 changed during the session (.cache/boundary/$sid.diff). If the change was yours, run ./go_d.zsh auto again."
    exit 1
  fi
  print "[$(now)] $sid: exit $rc, log status $(log_status $sid)"
}

start_job_bg() {      # start job $1 unless it is done or running; jobs skip finished work, so this resumes them
  local j=$1 f=$D/jobs/$1.zsh t=.cache/done/$(jtag $1)
  [[ -e $t ]] && return 0
  [[ -r $f ]] || { print -u2 "[$(now)] $f missing: the session before it should have written it"; return 1 }
  job_alive $j && return 0
  rm -f $t.failed $t.stalled $t.boundary $t.blocked
  ( jboundary start $(jtag $j)
    nice -n 5 caffeinate -i zsh "$f" >> .cache/logs/$(jtag $j).log 2>&1
    rc=$?
    if (( rc == 0 )); then
      if jboundary end $(jtag $j); then touch "$t"; else touch $t.failed $t.boundary; fi
    elif (( rc == 3 )); then touch $t.failed $t.blocked      # the job stopped at one of the plan's gates
    else touch $t.failed; fi ) &
  print $! > $t.pid
  disown
  print "[$(now)] job $j: started, pid $(<$t.pid) (.cache/logs/$(jtag $j).log)"
}

progress_fp() { { cksum < $progress; wc -c < .cache/logs/$(jtag $1).log } 2>/dev/null }

kill_tree() {         # $1 pid: stop it and every process under it
  local c
  for c in $(pgrep -P $1 2>/dev/null); do kill_tree $c; done
  kill -TERM $1 2>/dev/null
}

wait_job() {          # no time limit while the job makes progress; stopped after stall_checks checks without any
  local j=$1 t=.cache/done/$(jtag $1) fp last="" same=0 n=0
  [[ -e $t ]] && return 0
  start_job_bg $j || return 1
  print "[$(now)] waiting for job $j: no time limit; stopped only after $stall_checks checks in a row ($(( poll * stall_checks / 60 )) min) without progress. Watch: tail -f .cache/logs/$(jtag $j).log"
  while job_alive $j && [[ ! -e $t ]]; do
    sleep $poll
    (( n += 1 ))
    fp=$(progress_fp $j)
    if [[ $fp == $last ]]; then (( same += 1 )); else same=0; last=$fp; fi
    if (( same >= stall_checks )) && job_alive $j; then
      print "[$(now)] job $j: no progress in $stall_checks checks in a row; stopping it (see .cache/logs/$(jtag $j).log)"
      kill_tree $(<$t.pid)
      touch $t.failed $t.stalled
      return 1
    fi
    if (( n % 12 == 0 )) && [[ -r $progress ]]; then print "[$(now)] job $j running: $(head -c 200 $progress | tr -d '\n')"; fi
  done
  [[ -e $t ]]
}

run_after() {         # $1 session id: run the job it wrote, if any, and wait for it
  local j=${after[$1]:-} st
  [[ -z $j ]] && return 0
  wait_job $j && return 0
  st=$(job_state $j)
  if [[ $st == BLOCKED ]]; then
    print "[$(now)] job $j stopped at one of the plan's sanity gates (BLOCKED). It did not crash. Read $D/DECISIONS.md and the last lines of .cache/logs/$(jtag $j).log; write your answer in $D/DECISIONS.md, run ./go_d.zsh $1 resume to change the job with Claude, then ./go_d.zsh auto."
  elif [[ $st == BOUNDARY ]]; then
    print "[$(now)] job $j finished its work, but PDM's own files or the code folders changed while it ran (.cache/boundary/$(jtag $j).diff). If the change was yours, run ./go_d.zsh auto again: the job re-checks its finished work in minutes."
  else
    print "[$(now)] job $j $st: see .cache/logs/$(jtag $j).log. Fix the cause if there is one, then ./go_d.zsh auto: the job resumes where it stopped."
  fi
  return 1
}

case ${1:-} in
  auto)
    check_setup || exit 2
    for sid in $order; do
      on_branch $sid || exit 1
      st=$(log_status $sid)
      if [[ $st == COMPLETE ]]; then run_after $sid || exit 1; continue; fi
      if [[ $st == BLOCKED ]]; then
        print "$sid is BLOCKED: answer in $D/DECISIONS.md, run ./go_d.zsh $sid resume, then ./go_d.zsh auto"; exit 1
      fi
      kind=new; [[ $st == PARTIAL ]] && kind=resume
      attempt=0; waits=0
      while (( attempt < retries )); do
        run_headless $sid $kind
        st=$(log_status $sid)
        [[ $st == COMPLETE || $st == BLOCKED ]] && break
        kind=resume
        limit_wait "$last_out"; lw=$?
        if (( lw == 0 && waits < limit_waits )); then (( waits += 1 )); continue; fi
        if (( lw == 2 )); then print "[$(now)] $sid: Claude's usage limit was reached, and its reset time could not be read ($last_out). Run ./go_d.zsh auto again after it resets."; exit 1; fi
        (( attempt += 1 ))
        if (( attempt >= 2 && attempt < retries )); then print "[$(now)] $sid not complete; waiting $retry_wait s before retrying"; sleep $retry_wait; fi
      done
      if [[ $st == BLOCKED ]]; then print "$sid is BLOCKED: see $D/DECISIONS.md and $D/CHECKPOINT.md"; exit 1; fi
      if [[ $st != COMPLETE ]]; then print "$sid did not complete after $retries attempts; see .cache/logs"; exit 1; fi
      run_after $sid || exit 1
    done
    print "[$(now)] Option D complete: see $D/REPORT.md" ;;
  status)
    for sid in $order; do print -- "$sid  ${model[$sid]}  session $(log_status $sid)${after[$sid]:+   then job ${after[$sid]}: $(job_state ${after[$sid]})}"; done
    [[ -r $progress ]] && print -- "job progress: $(head -c 300 $progress | tr -d '\n')"
    [[ -r results/d_ladder.txt ]] && { print -- "latest ladder results (results/d_ladder.txt):"; tail -n 8 results/d_ladder.txt }
    print -- "branch: $(git branch --show-current 2>/dev/null)" ;;
  job)
    j=${2:-}
    [[ $j == (record|ladder) ]] || { print -u2 "usage: ./go_d.zsh job record|ladder"; exit 2 }
    if wait_job $j; then print "[$(now)] job $j: done"
    else print "[$(now)] job $j: $(job_state $j); see .cache/logs/$(jtag $j).log"; exit 1; fi ;;
  D<->)
    sid=$1
    [[ -n ${model[$sid]:-} ]] || { print -u2 "unknown session: $sid"; exit 2 }
    check_setup || exit 2
    on_branch $sid || exit 1
    kind=new; [[ ${2:-} == resume ]] && kind=resume
    typeset -a ix=()
    (( deny_flag )) && ix+=(--disallowedTools "${deny[@]}")
    (( name_flag )) && ix+=(--name "air-d-$sid")
    # the denial list takes several values, so a flag must follow it before the prompt
    exec claude "${ix[@]}" --model "${model[$sid]}" --effort "${effort[$sid]}" "$(prompt_for $sid $kind interactive)" ;;
  *)
    print -u2 "usage: ./go_d.zsh auto | status | job record|ladder | D<n> [resume]"; exit 2 ;;
esac

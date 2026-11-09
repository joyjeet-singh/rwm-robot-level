#!/bin/zsh
# AIR 2027 Option C launcher (OPTIONC_PLAN.md): root cause, mixed MuJoCo data, five-network simulators, policies.
# Runs from its own folder wherever it is called from.
#   caffeinate -i ./go_c.zsh auto      every remaining session and job, in order, unattended (safe to re-run)
#   ./go_c.zsh status                  sessions, jobs and job progress
#   ./go_c.zsh C3                      one session, interactive
#   ./go_c.zsh C3 resume               resume a PARTIAL or BLOCKED session, interactive
#   ./go_c.zsh job pilot               start (or resume) one job in the background and wait for it
# Jobs (diag, data, pilot, main) run after the session that writes them, one at a time. A job has no time limit; it is
# stopped only when results/c/progress.json and its log stay unchanged for stall_checks checks in a row (12 x 5 min).
# Sessions must leave everything in ~/Downloads/PDM outside "AIR 2027" unchanged (revision/boundary.zsh). Jobs run for
# days, so they check only what their code could change by mistake: PDM's own files and the code folders in jdirs.
# When a session stops at Claude's usage limit, the launcher waits for the reset time the message gives, then resumes.
# Optional overrides in optc/driver.conf (zsh syntax), e.g.  mode=acceptEdits  retries=4  stall_checks=24  deny_flag=0
# Never edit this file while it runs.
emulate -L zsh
set -uo pipefail
cd "${0:A:h}" || exit 2

PLAN=OPTIONC_PLAN.md
D=optc
order=(C0 C1 C2 C3 C4 C5 C6)
typeset -A model effort turns after
model=(C0 claude-opus-5-5  C1 claude-opus-5-5  C2 claude-opus-5-5  C3 claude-opus-5-5
       C4 claude-opus-5-5  C5 claude-sonnet-5-5 C6 claude-opus-5-5)
effort=(C0 high C1 high C2 high C3 high C4 high C5 high C6 high)
turns=(C0 150 C1 180 C2 150 C3 150 C4 150 C5 120 C6 100)
after=(C0 diag C1 data C2 pilot C3 main)   # the job each session writes, run before the next session
branch=optc        # every session after C0 starts on this branch
progress=results/c/progress.json
jdirs=(rwm_repro robotic_world_model_lite rsl_rl_rwm rwm_analysis)   # sibling code folders a job reads
mode=auto
retries=6
retry_wait=1800    # seconds to wait before the third and later attempts of a session
poll=300           # seconds between checks on a job
stall_checks=12    # checks in a row with no change in the progress file or the job log: the job counts as stalled
limit_margin=300   # wait this long past a usage limit's reset time
limit_waits=10     # at most this many usage-limit waits per session
name_flag=1
deny_flag=1
deny=("Bash(./go_c.zsh:*)" "Bash(zsh go_c.zsh:*)" "Bash(zsh ./go_c.zsh:*)" "Bash(./go_s2.zsh:*)" "Bash(./go_im.zsh:*)"
      "Bash(./go_rw.zsh:*)" "Bash(./go_s1.zsh:*)" "Bash(./go_rev.zsh:*)" "Bash(caffeinate:*)" "Bash(nohup:*)"
      "Bash(zsh optc/jobs:*)" "Bash(zsh ./optc/jobs:*)" "Bash(./optc/jobs:*)" "Bash(bash optc/jobs:*)"
      "Bash(pip:*)" "Bash(pip3:*)" "Bash(git clone:*)" "Bash(git push:*)")
[[ -r $D/driver.conf ]] && source $D/driver.conf
mkdir -p .cache/logs .cache/done .cache/boundary $D/jobs $D/notes $D/review results/c

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

log_status() {        # last status word that session $1 wrote to optc/LOG.md, or "none"
  local line
  line=$(grep -E "^## $1 status: " $D/LOG.md 2>/dev/null | tail -1)
  [[ -z $line ]] && { print none; return }
  line=${line#*status: }
  print -- ${line%% *}
}

jtag() { print -- "job_c_$1" }                   # marker and log name for job $1
job_alive() { local p f=.cache/done/$(jtag $1).pid; [[ -r $f ]] || return 1; p=$(<$f); kill -0 $p 2>/dev/null }

job_state() {         # none | pending | running | done | STALLED | BOUNDARY | FAILED
  local t=.cache/done/$(jtag $1)
  if [[ -e $t ]]; then print done
  elif job_alive $1; then print running
  elif [[ -e $t.stalled ]]; then print STALLED
  elif [[ -e $t.boundary ]]; then print BOUNDARY
  elif [[ -e $t.failed ]]; then print FAILED
  elif [[ -r $D/jobs/$1.zsh ]]; then print pending
  else print none; fi
}

block() { awk -v b="<!-- BEGIN $1 -->" -v e="<!-- END $1 -->" '$0==b{f=1;next} $0==e{f=0} f' "$PLAN" }

prompt_for() {        # $1 session id, $2 new|resume, $3 headless|interactive
  local sid=$1 head
  head="You are session $sid of AIR 2027 Option C (root cause, then learned simulators of the MuJoCo ANYmal D), working in this folder."
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
  for sid in COMMON $order; do
    [[ -n "$(block $sid 2>/dev/null)" ]] || { print -u2 "$PLAN: block $sid missing"; ok=0 }
  done
  grep -q '^## M7 status: COMPLETE' step2/LOG.md 2>/dev/null || { print -u2 "Step 2 is not complete (step2/LOG.md)"; ok=0 }
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

on_branch() {         # every session after C0 must start on the Option C branch
  local b
  b=$(git branch --show-current 2>/dev/null)
  [[ $b == $branch ]] && return 0
  print -u2 "git is on branch '${b:-?}', not '$branch': run  git switch $branch  then ./go_c.zsh auto"
  return 1
}

run_headless() {      # $1 session id, $2 new|resume
  local sid=$1 kind=$2 out=.cache/logs/session_c_${1}_${2}_$(date +%Y%m%d-%H%M%S).log rc
  local -a extra=()
  (( deny_flag )) && extra+=(--disallowedTools "${deny[@]}")
  (( name_flag )) && extra+=(--name "air-c-$sid")
  print "[$(now)] $sid: $kind, ${model[$sid]}, effort ${effort[$sid]}, max ${turns[$sid]} turns, mode $mode -> $out"
  boundary start $sid
  claude -p "$(prompt_for $sid $kind headless)" --model "${model[$sid]}" --effort "${effort[$sid]}" \
    --permission-mode "$mode" --max-turns "${turns[$sid]}" "${extra[@]}" > "$out" 2>&1
  rc=$?
  last_out=$out
  if ! boundary end $sid; then
    print "[$(now)] $sid: STOPPED: files outside AIR 2027 changed during the session (.cache/boundary/$sid.diff). If the change was yours, run ./go_c.zsh auto again."
    exit 1
  fi
  print "[$(now)] $sid: exit $rc, log status $(log_status $sid)"
}

start_job_bg() {      # start job $1 unless it is done or running; jobs skip finished work, so this resumes them
  local j=$1 f=$D/jobs/$1.zsh t=.cache/done/$(jtag $1)
  [[ -e $t ]] && return 0
  [[ -r $f ]] || { print -u2 "[$(now)] $f missing: the session before it should have written it"; return 1 }
  job_alive $j && return 0
  rm -f $t.failed $t.stalled $t.boundary
  ( jboundary start $(jtag $j)
    if nice -n 5 caffeinate -i zsh "$f" >> .cache/logs/$(jtag $j).log 2>&1; then
      if jboundary end $(jtag $j); then touch "$t"; else touch $t.failed $t.boundary; fi
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
  if [[ $st == BOUNDARY ]]; then
    print "[$(now)] job $j finished its work, but PDM's own files or the code folders changed while it ran (.cache/boundary/$(jtag $j).diff). If the change was yours, run ./go_c.zsh auto again: the job re-checks its finished work in minutes."
  else
    print "[$(now)] job $j $st: see .cache/logs/$(jtag $j).log. Fix the cause if there is one, then ./go_c.zsh auto: the job resumes where it stopped."
  fi
  return 1
}

case ${1:-} in
  auto)
    check_setup || exit 2
    for sid in $order; do
      if [[ $sid != C0 ]]; then on_branch || exit 1; fi
      st=$(log_status $sid)
      if [[ $st == COMPLETE ]]; then run_after $sid || exit 1; continue; fi
      if [[ $st == BLOCKED ]]; then
        print "$sid is BLOCKED: answer in $D/DECISIONS.md, run ./go_c.zsh $sid resume, then ./go_c.zsh auto"; exit 1
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
        if (( lw == 2 )); then print "[$(now)] $sid: Claude's usage limit was reached, and its reset time could not be read ($last_out). Run ./go_c.zsh auto again after it resets."; exit 1; fi
        (( attempt += 1 ))
        if (( attempt >= 2 && attempt < retries )); then print "[$(now)] $sid not complete; waiting $retry_wait s before retrying"; sleep $retry_wait; fi
      done
      if [[ $st == BLOCKED ]]; then print "$sid is BLOCKED: see $D/DECISIONS.md and $D/CHECKPOINT.md"; exit 1; fi
      if [[ $st != COMPLETE ]]; then print "$sid did not complete after $retries attempts; see .cache/logs"; exit 1; fi
      run_after $sid || exit 1
    done
    print "[$(now)] Option C complete: see $D/REPORT.md" ;;
  status)
    for sid in $order; do print -- "$sid  ${model[$sid]}  session $(log_status $sid)${after[$sid]:+   then job ${after[$sid]}: $(job_state ${after[$sid]})}"; done
    [[ -r $progress ]] && print -- "job progress: $(head -c 300 $progress | tr -d '\n')"
    print -- "branch: $(git branch --show-current 2>/dev/null)" ;;
  job)
    j=${2:-}
    [[ $j == (diag|data|pilot|main) ]] || { print -u2 "usage: ./go_c.zsh job diag|data|pilot|main"; exit 2 }
    if wait_job $j; then print "[$(now)] job $j: done"
    else print "[$(now)] job $j: $(job_state $j); see .cache/logs/$(jtag $j).log"; exit 1; fi ;;
  C<->)
    sid=$1
    [[ -n ${model[$sid]:-} ]] || { print -u2 "unknown session: $sid"; exit 2 }
    check_setup || exit 2
    if [[ $sid != C0 ]]; then on_branch || exit 1; fi
    kind=new; [[ ${2:-} == resume ]] && kind=resume
    typeset -a ix=()
    (( deny_flag )) && ix+=(--disallowedTools "${deny[@]}")
    (( name_flag )) && ix+=(--name "air-c-$sid")
    # the denial list takes several values, so a flag must follow it before the prompt
    exec claude "${ix[@]}" --model "${model[$sid]}" --effort "${effort[$sid]}" "$(prompt_for $sid $kind interactive)" ;;
  *)
    print -u2 "usage: ./go_c.zsh auto | status | job diag|data|pilot|main | C<n> [resume]"; exit 2 ;;
esac

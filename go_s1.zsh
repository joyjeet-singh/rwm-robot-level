#!/bin/zsh
# AIR 2027 Step 1 launcher (STEP1_PLAN.md). Runs from its own folder wherever it is called from.
#   caffeinate -i ./go_s1.zsh auto     every remaining session and the job, in order, unattended (safe to re-run)
#   ./go_s1.zsh status                 each session's status and the job's state
#   ./go_s1.zsh S3                     one session, interactive
#   ./go_s1.zsh S6 resume              resume a PARTIAL or BLOCKED session, interactive
#   ./go_s1.zsh job S3                 start (or resume) job S3 in the background and wait for it
# The job has no time limit. It is stopped only when its progress file and its log stay unchanged for
# stall_checks checks in a row (12 checks, 5 minutes apart: one hour). Checks that span a sleep of the Mac
# count once, so a sleeping Mac is not mistaken for a stalled job.
# Optional overrides in step1/driver.conf (zsh syntax), e.g.  mode=acceptEdits  retries=4  stall_checks=24  deny_flag=0
# Never edit this file while it runs.
emulate -L zsh
set -uo pipefail
cd "${0:A:h}" || exit 2

PLAN=STEP1_PLAN.md
D=step1
order=(S0 S1 S2 S3 S4 S5 S6 S7 S8)
typeset -A model effort turns
model=(S0 claude-sonnet-5-5  S1 claude-opus-5-5    S2 claude-opus-5-5    S3 claude-opus-5-5
       S4 claude-sonnet-5-5  S5 claude-sonnet-5-5  S6 claude-opus-5-5    S7 claude-sonnet-5-5
       S8 claude-opus-5-5)
effort=(S0 medium S1 high S2 high S3 high S4 high S5 medium S6 high S7 high S8 high)
turns=(S0 60 S1 150 S2 150 S3 150 S4 120 S5 60 S6 120 S7 80 S8 100)
bg_job=S3          # trains the new baselines and scores every simulator, in the background while S4 and S5 work
needs_job=S6       # waits for bg_job to finish
branch=step1       # every session after S0 starts on this branch
progress=results/s1/progress.json   # the job rewrites this as it works (STEP1_PLAN.md, S3)
mode=auto
retries=6
retry_wait=1800    # seconds to wait before the third and later attempts of a session
poll=300           # seconds between checks on the job
stall_checks=12    # checks in a row with no change in the progress file or the log: the job counts as stalled
limit_margin=300   # when a session stops at Claude's usage limit, wait this long past the reset time it names
limit_waits=10     # at most this many such waits per session
name_flag=1
deny_flag=1        # pass the extra denials below to every session
# Sessions must never start this launcher, the job or caffeinate themselves (on top of .claude/settings.json).
deny=("Bash(./go_s1.zsh:*)" "Bash(zsh go_s1.zsh:*)" "Bash(zsh ./go_s1.zsh:*)" "Bash(caffeinate:*)"
      "Bash(nohup:*)" "Bash(zsh step1/jobs/S3.zsh:*)" "Bash(zsh ./step1/jobs/S3.zsh:*)")
[[ -r $D/driver.conf ]] && source $D/driver.conf
mkdir -p .cache/logs .cache/done $D/jobs $D/notes $D/review

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

log_status() {        # last status word that session $1 wrote to step1/LOG.md, or "none"
  local line
  line=$(grep -E "^## $1 status: " $D/LOG.md 2>/dev/null | tail -1)
  [[ -z $line ]] && { print none; return }
  line=${line#*status: }
  print -- ${line%% *}
}

job_alive() { local p; [[ -r .cache/done/job_$1.pid ]] || return 1; p=$(<.cache/done/job_$1.pid); kill -0 $p 2>/dev/null }

job_state() {         # none | pending | running | done | STALLED | FAILED
  if [[ -e .cache/done/job_$1 ]]; then print done
  elif job_alive $1; then print running
  elif [[ -e .cache/done/job_$1.stalled ]]; then print STALLED
  elif [[ -e .cache/done/job_$1.failed ]]; then print FAILED
  elif [[ -r $D/jobs/$1.zsh ]]; then print pending
  else print none; fi
}

block() { awk -v b="<!-- BEGIN $1 -->" -v e="<!-- END $1 -->" '$0==b{f=1;next} $0==e{f=0} f' "$PLAN" }

prompt_for() {        # $1 session id, $2 new|resume, $3 headless|interactive
  local sid=$1 head
  head="You are session $sid of AIR 2027 Step 1, working in this folder."
  if [[ $3 == interactive ]]; then
    head+=" The author is present in this terminal and may answer questions; otherwise work as the plan says."
  else
    head+=" You are running unattended: never ask the user anything; follow $PLAN, or stop BLOCKED as its common rules say."
  fi
  head+=" Your instructions are the common rules and your own block, both below. If your context is compacted, re-read them with: awk '/BEGIN COMMON/,/END COMMON/' $PLAN ; awk '/BEGIN $sid -->/,/END $sid -->/' $PLAN"
  [[ $2 == resume ]] && head+=$'\n'"This is a resumed attempt: first read $D/CHECKPOINT.md, the last 20 lines of $D/LOG.md, your notes in $D/notes/ and git log --oneline -15; continue from the first unfinished step and never redo a finished one."
  print -r -- "$head"
  print -r -- ""
  block COMMON
  print -r -- ""
  block $sid
}

boundary() { zsh revision/boundary.zsh "$@" }

check_setup() {
  local ok=1 a sid
  command -v claude >/dev/null || { print -u2 "claude (Claude Code CLI) not found on PATH"; ok=0 }
  [[ -r $PLAN ]] || { print -u2 "$PLAN missing"; ok=0 }
  for sid in COMMON $order; do
    [[ -n "$(block $sid 2>/dev/null)" ]] || { print -u2 "$PLAN: block $sid missing"; ok=0 }
  done
  grep -q '^## R11 status: COMPLETE' revision/LOG.md 2>/dev/null || { print -u2 "revision round 1 is not complete (revision/LOG.md)"; ok=0 }
  job_alive R3 && { print -u2 "revision job R3 is still running"; ok=0 }
  [[ -r revision/boundary.zsh ]] || { print -u2 "revision/boundary.zsh missing"; ok=0 }
  [[ -r .claude/settings.json ]] || { print -u2 ".claude/settings.json missing (permissions and denials; go_rev.zsh installs it)"; ok=0 }
  for a in air-reader air-stats-reviewer air-claim-checker air-referee; do
    [[ -r .claude/agents/$a.md ]] || { print -u2 ".claude/agents/$a.md missing"; ok=0 }
  done
  [[ -x py ]] || chmod +x py
  (( ok ))
}

on_branch() {         # every session after S0 must start on the Step 1 branch
  local b
  b=$(git branch --show-current 2>/dev/null)
  [[ $b == $branch ]] && return 0
  print -u2 "git is on branch '${b:-?}', not '$branch': run  git switch $branch  then ./go_s1.zsh auto"
  return 1
}

run_headless() {      # $1 session id, $2 new|resume
  local sid=$1 kind=$2 out=.cache/logs/session_${1}_${2}_$(date +%Y%m%d-%H%M%S).log rc
  local -a extra=()
  (( deny_flag )) && extra+=(--disallowedTools "${deny[@]}")
  (( name_flag )) && extra+=(--name "air-s1-$sid")
  print "[$(now)] $sid: $kind, ${model[$sid]}, effort ${effort[$sid]}, max ${turns[$sid]} turns, mode $mode -> $out"
  boundary start $sid
  claude -p "$(prompt_for $sid $kind headless)" --model "${model[$sid]}" --effort "${effort[$sid]}" \
    --permission-mode "$mode" --max-turns "${turns[$sid]}" "${extra[@]}" > "$out" 2>&1
  rc=$?
  last_out=$out
  if ! boundary end $sid; then
    print "[$(now)] $sid: STOPPED: files outside AIR 2027 changed during the session (.cache/boundary/$sid.diff)"; exit 1
  fi
  print "[$(now)] $sid: exit $rc, log status $(log_status $sid)"
}

start_job_bg() {      # start the job unless it is done or running; the job skips finished work, so this resumes it
  local sid=$1 f=$D/jobs/$1.zsh ok=.cache/done/job_$1
  [[ -e $ok ]] && return 0
  [[ -r $f ]] || { print -u2 "[$(now)] $f missing: session $sid should have written it"; return 1 }
  job_alive $sid && return 0
  rm -f .cache/done/job_$sid.failed .cache/done/job_$sid.stalled
  ( boundary start job_$sid
    if nice -n 5 caffeinate -i zsh "$f" >> .cache/logs/job_$sid.log 2>&1 && boundary end job_$sid; then touch "$ok"
    else touch .cache/done/job_$sid.failed; fi ) &
  print $! > .cache/done/job_$sid.pid
  disown
  print "[$(now)] job $sid: started in the background, pid $(<.cache/done/job_$sid.pid) (.cache/logs/job_$sid.log)"
}

progress_fp() {       # what changes while the job works: its progress file and the size of its log
  { cksum < $progress; wc -c < .cache/logs/job_$1.log } 2>/dev/null
}

kill_tree() {         # $1 pid: stop it and every process under it
  local c
  for c in $(pgrep -P $1 2>/dev/null); do kill_tree $c; done
  kill -TERM $1 2>/dev/null
}

wait_job() {          # no time limit while the job makes progress; stopped after stall_checks checks without any
  local sid=$1 fp last="" same=0 n=0
  [[ -e .cache/done/job_$sid ]] && return 0
  start_job_bg $sid || return 1
  print "[$(now)] waiting for job $sid: no time limit; stopped only after $stall_checks checks in a row ($(( poll * stall_checks / 60 )) min) without progress. Watch: tail -f .cache/logs/job_$sid.log"
  while job_alive $sid && [[ ! -e .cache/done/job_$sid ]]; do
    sleep $poll
    (( n += 1 ))
    fp=$(progress_fp $sid)
    if [[ $fp == $last ]]; then (( same += 1 )); else same=0; last=$fp; fi
    if (( same >= stall_checks )) && job_alive $sid; then
      print "[$(now)] job $sid: no progress in $stall_checks checks in a row; stopping it (see .cache/logs/job_$sid.log)"
      kill_tree $(<.cache/done/job_$sid.pid)
      touch .cache/done/job_$sid.failed .cache/done/job_$sid.stalled
      return 1
    fi
    if (( n % 12 == 0 )) && [[ -r $progress ]]; then print "[$(now)] job $sid running: $(head -c 200 $progress | tr -d '\n')"; fi
  done
  [[ -e .cache/done/job_$sid ]]
}

after_session() {     # $1 session id: start its background job, if it has one
  [[ $1 == $bg_job ]] || return 0
  start_job_bg $1 || exit 1
}

case ${1:-} in
  auto)
    check_setup || exit 2
    for sid in $order; do
      if [[ $sid != S0 ]]; then on_branch || exit 1; fi
      if [[ $sid == $needs_job ]]; then
        wait_job $bg_job || { print "[$(now)] job $bg_job $(job_state $bg_job): see .cache/logs/job_$bg_job.log (and .cache/boundary/job_$bg_job.diff, if present). Fix the cause if there is one, then ./go_s1.zsh auto: the job resumes where it stopped."; exit 1 }
      fi
      st=$(log_status $sid)
      if [[ $st == COMPLETE ]]; then after_session $sid; continue; fi
      if [[ $st == BLOCKED ]]; then
        print "$sid is BLOCKED: answer in $D/DECISIONS.md, run ./go_s1.zsh $sid resume, then ./go_s1.zsh auto"; exit 1
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
        if (( lw == 2 )); then print "[$(now)] $sid: Claude's usage limit was reached, and its reset time could not be read ($last_out). Run ./go_s1.zsh auto again after it resets."; exit 1; fi
        (( attempt += 1 ))
        if (( attempt >= 2 && attempt < retries )); then print "[$(now)] $sid not complete; waiting $retry_wait s before retrying"; sleep $retry_wait; fi
      done
      if [[ $st == BLOCKED ]]; then print "$sid is BLOCKED: see $D/DECISIONS.md and $D/CHECKPOINT.md"; exit 1; fi
      if [[ $st != COMPLETE ]]; then print "$sid did not complete after $retries attempts; see .cache/logs"; exit 1; fi
      after_session $sid
    done
    print "[$(now)] Step 1 complete: see $D/REPORT.md" ;;
  status)
    for sid in $order; do print -- "$sid  ${model[$sid]}  session $(log_status $sid)"; done
    print -- "job $bg_job: $(job_state $bg_job)"
    [[ -r $progress ]] && print -- "job progress: $(head -c 300 $progress | tr -d '\n')"
    print -- "branch: $(git branch --show-current 2>/dev/null)" ;;
  job)
    [[ ${2:-} == $bg_job ]] || { print -u2 "usage: ./go_s1.zsh job $bg_job"; exit 2 }
    if wait_job $2; then print "[$(now)] job $2: done"
    else print "[$(now)] job $2: $(job_state $2); see .cache/logs/job_$2.log"; exit 1; fi ;;
  S<->)
    sid=$1
    [[ -n ${model[$sid]:-} ]] || { print -u2 "unknown session: $sid"; exit 2 }
    check_setup || exit 2
    if [[ $sid != S0 ]]; then on_branch || exit 1; fi
    kind=new; [[ ${2:-} == resume ]] && kind=resume
    typeset -a ix=()
    (( deny_flag )) && ix+=(--disallowedTools "${deny[@]}")
    (( name_flag )) && ix+=(--name "air-s1-$sid")
    # the denial list takes several values, so a flag must follow it before the prompt
    exec claude "${ix[@]}" --model "${model[$sid]}" --effort "${effort[$sid]}" "$(prompt_for $sid $kind interactive)" ;;
  *)
    print -u2 "usage: ./go_s1.zsh auto | status | job $bg_job | S<n> [resume]"; exit 2 ;;
esac

#!/bin/zsh
# AIR 2027 revision launcher (REVISION_PLAN.md). Runs from its own folder wherever it is called from.
#   caffeinate -i ./go_rev.zsh auto     every remaining session and job, in order, unattended (safe to re-run)
#   ./go_rev.zsh status                 each session's status and the job's state
#   ./go_rev.zsh R0                     one session, interactive (recommended for R0)
#   ./go_rev.zsh R6 resume              resume a PARTIAL or BLOCKED session, interactive
#   ./go_rev.zsh job R3                 run (or resume) job R3 in the foreground
# Optional overrides in revision/driver.conf (zsh syntax), e.g.  mode=acceptEdits  retries=4  name_flag=0
# Never edit this file while it runs.
emulate -L zsh
set -uo pipefail
cd "${0:A:h}" || exit 2

PLAN=REVISION_PLAN.md
order=(R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11)
typeset -A model effort turns
model=(R0 claude-sonnet-5-5  R1 claude-opus-5-5   R2 claude-sonnet-5-5  R3 claude-opus-5-5
       R4 claude-sonnet-5-5  R5 claude-sonnet-5-5 R6 claude-opus-5-5    R7 claude-sonnet-5-5
       R8 claude-sonnet-5-5  R9 claude-opus-5-5   R10 claude-opus-5-5   R11 claude-sonnet-5-5)
effort=(R0 medium R1 high R2 medium R3 high R4 medium R5 high R6 high R7 medium
        R8 high R9 high R10 high R11 medium)
turns=(R0 80 R1 100 R2 60 R3 150 R4 80 R5 100 R6 150 R7 100 R8 100 R9 100 R10 120 R11 80)
bg_job=R3          # runs in the background while R4-R8 work
needs_job=R9       # waits for bg_job to finish
mode=auto
retries=6
name_flag=1
[[ -r revision/driver.conf ]] && source revision/driver.conf
mkdir -p .cache/logs .cache/done revision/jobs revision/notes

now() { date '+%F %T' }

log_status() {        # last status word that session $1 wrote to revision/LOG.md, or "none"
  local line
  line=$(grep -E "^## $1 status: " revision/LOG.md 2>/dev/null | tail -1)
  [[ -z $line ]] && { print none; return }
  line=${line#*status: }
  print -- ${line%% *}
}

job_alive() { local p; [[ -r .cache/done/job_$1.pid ]] || return 1; p=$(<.cache/done/job_$1.pid); kill -0 $p 2>/dev/null }

job_state() {         # none | pending | running | done | FAILED
  if [[ ! -r revision/jobs/$1.zsh ]]; then print none
  elif [[ -e .cache/done/job_$1 ]]; then print done
  elif job_alive $1; then print running
  elif [[ -e .cache/done/job_$1.failed ]]; then print FAILED
  else print pending; fi
}

block() { awk -v b="<!-- BEGIN $1 -->" -v e="<!-- END $1 -->" '$0==b{f=1;next} $0==e{f=0} f' "$PLAN" }

prompt_for() {        # $1 session id, $2 new|resume
  local sid=$1 head
  head="You are session $sid of the AIR 2027 revision, working in this folder. You are running unattended: never ask the user anything; follow REVISION_PLAN.md, or stop BLOCKED as its common rules say. Your instructions are the common rules and your own block, both below. If your context is compacted, re-read them with: awk '/BEGIN COMMON/,/END COMMON/' $PLAN ; awk '/BEGIN $sid -->/,/END $sid -->/' $PLAN"
  [[ $2 == resume ]] && head+=$'\n'"This is a resumed attempt: first read revision/CHECKPOINT.md, the last 20 lines of revision/LOG.md, your notes in revision/notes/ and git log --oneline -15; continue from the first unfinished step and never redo a finished one."
  print -r -- "$head"
  print -r -- ""
  block COMMON
  print -r -- ""
  block $sid
}

boundary() { zsh revision/boundary.zsh "$@" }

install_config() {    # copy revision/claude/ into .claude/ where missing; never overwrite an edited copy
  local f dst
  [[ -d revision/claude ]] || return 0
  mkdir -p .claude/agents
  for f in revision/claude/settings.json revision/claude/agents/*.md(N); do
    dst=.claude/${f#revision/claude/}
    if [[ ! -e $dst ]]; then cp "$f" "$dst" && print "[$(now)] installed $dst"
    elif ! cmp -s "$f" "$dst"; then print -u2 "note: $dst differs from $f; keeping $dst"; fi
  done
}

check_setup() {
  local ok=1
  install_config
  command -v claude >/dev/null || { print -u2 "claude (Claude Code CLI) not found on PATH"; ok=0 }
  [[ -r $PLAN ]] || { print -u2 "$PLAN missing"; ok=0 }
  [[ -r .claude/settings.json ]] || { print -u2 ".claude/settings.json missing (permissions and denials)"; ok=0 }
  for a in air-reader air-stats-reviewer air-bib-checker air-claim-checker air-referee; do
    [[ -r .claude/agents/$a.md ]] || { print -u2 ".claude/agents/$a.md missing"; ok=0 }
  done
  [[ -x py ]] || chmod +x py
  (( ok ))
}

run_headless() {      # $1 session id, $2 new|resume
  local sid=$1 kind=$2 out=.cache/logs/session_${1}_${2}_$(date +%Y%m%d-%H%M%S).log rc
  local -a extra=()
  (( name_flag )) && extra=(--name "air-rev-$sid")
  print "[$(now)] $sid: $kind, ${model[$sid]}, effort ${effort[$sid]}, max ${turns[$sid]} turns, mode $mode -> $out"
  boundary start $sid
  claude -p "$(prompt_for $sid $kind)" --model "${model[$sid]}" --effort "${effort[$sid]}" \
    --permission-mode "$mode" --max-turns "${turns[$sid]}" "${extra[@]}" > "$out" 2>&1
  rc=$?
  if ! boundary end $sid; then
    print "[$(now)] $sid: STOPPED: files outside AIR 2027 changed during the session (.cache/boundary/$sid.diff)"; exit 1
  fi
  print "[$(now)] $sid: exit $rc, log status $(log_status $sid)"
}

run_job() {           # foreground; job scripts skip finished work, so re-running resumes
  local sid=$1 f=revision/jobs/$1.zsh ok=.cache/done/job_$1
  [[ -e $ok || ! -r $f ]] && return 0
  rm -f .cache/done/job_$sid.failed
  print "[$(now)] job $sid: start (.cache/logs/job_$sid.log)"
  boundary start job_$sid
  if caffeinate -i zsh "$f" >> .cache/logs/job_$sid.log 2>&1; then touch "$ok"; print "[$(now)] job $sid: done"
  else touch .cache/done/job_$sid.failed; print "[$(now)] job $sid: FAILED (see its log)"; return 1; fi
  boundary end job_$sid || { print "[$(now)] job $sid: files outside AIR 2027 changed"; return 1 }
}

start_job_bg() {
  local sid=$1 f=revision/jobs/$1.zsh ok=.cache/done/job_$1
  [[ -e $ok || ! -r $f ]] && return 0
  job_alive $sid && return 0
  rm -f .cache/done/job_$sid.failed
  ( if caffeinate -i zsh "$f" >> .cache/logs/job_$sid.log 2>&1; then touch "$ok"; else touch .cache/done/job_$sid.failed; fi ) &
  print $! > .cache/done/job_$sid.pid
  disown
  print "[$(now)] job $sid: started in the background, pid $(<.cache/done/job_$sid.pid) (.cache/logs/job_$sid.log)"
}

wait_job() {
  local sid=$1
  [[ -e .cache/done/job_$sid ]] && return 0
  start_job_bg $sid
  print "[$(now)] waiting for job $sid (checks every 5 min; tail -f .cache/logs/job_$sid.log to watch)"
  while job_alive $sid && [[ ! -e .cache/done/job_$sid ]]; do sleep 300; done
  [[ -e .cache/done/job_$sid ]]
}

after_session() {     # $1 session id: run or start its job, if it has one
  if [[ $1 == $bg_job ]]; then start_job_bg $1; else run_job $1 || exit 1; fi
}

case ${1:-} in
  auto)
    check_setup || exit 2
    for sid in $order; do
      if [[ $sid == $needs_job ]]; then
        wait_job $bg_job || { print "[$(now)] job $bg_job FAILED: see .cache/logs/job_$bg_job.log, then ./go_rev.zsh job $bg_job and ./go_rev.zsh auto"; exit 1 }
      fi
      st=$(log_status $sid)
      if [[ $st == COMPLETE ]]; then after_session $sid; continue; fi
      if [[ $st == BLOCKED ]]; then
        print "$sid is BLOCKED: answer in revision/DECISIONS.md, run ./go_rev.zsh $sid resume, then ./go_rev.zsh auto"; exit 1
      fi
      kind=new; [[ $st == PARTIAL ]] && kind=resume
      for attempt in {1..$retries}; do
        run_headless $sid $kind
        st=$(log_status $sid)
        [[ $st == COMPLETE || $st == BLOCKED ]] && break
        kind=resume
        if (( attempt >= 2 && attempt < retries )); then print "[$(now)] $sid not complete; waiting 30 min before retrying"; sleep 1800; fi
      done
      if [[ $st == BLOCKED ]]; then print "$sid is BLOCKED: see revision/DECISIONS.md and revision/CHECKPOINT.md"; exit 1; fi
      if [[ $st != COMPLETE ]]; then print "$sid did not complete after $retries attempts; see .cache/logs"; exit 1; fi
      after_session $sid
    done
    print "[$(now)] revision round 1 complete: see revision/REPORT.md" ;;
  status)
    for sid in $order; do print -- "$sid  ${model[$sid]}  session $(log_status $sid)"; done
    print -- "job $bg_job: $(job_state $bg_job)"
    [[ -r results/ft/progress.json ]] && print -- "study FT progress: $(<results/ft/progress.json)" ;;
  job)
    run_job ${2:?"usage: ./go_rev.zsh job R3"} ;;
  R<->)
    sid=$1
    [[ -n ${model[$sid]:-} ]] || { print -u2 "unknown session: $sid"; exit 2 }
    check_setup || exit 2
    kind=new; [[ ${2:-} == resume ]] && kind=resume
    exec claude --model "${model[$sid]}" --effort "${effort[$sid]}" --name "air-rev-$sid" "$(prompt_for $sid $kind)" ;;
  *)
    print -u2 "usage: ./go_rev.zsh auto | status | job R3 | R<n> [resume]"; exit 2 ;;
esac

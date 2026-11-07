#!/bin/zsh
# AIR 2027 improvement-round launcher (IMPROVE_PLAN.md). Runs from its own folder wherever it is called from.
#   caffeinate -i ./go_im.zsh auto     every remaining session, in order, unattended (safe to re-run)
#   ./go_im.zsh status                 each session's status
#   ./go_im.zsh I2                     one session, interactive
#   ./go_im.zsh I2 resume              resume a PARTIAL or BLOCKED session, interactive
# When a session stops at Claude's usage limit, the launcher waits until the reset time the message gives, then
# resumes; such waits do not use up the session's retries.
# Optional overrides in improve/driver.conf (zsh syntax), e.g.  mode=acceptEdits  retries=4  name_flag=0  deny_flag=0
# Never edit this file while it runs.
emulate -L zsh
set -uo pipefail
cd "${0:A:h}" || exit 2

PLAN=IMPROVE_PLAN.md
D=improve
order=(I0 I1 I2 I3 I4 I5)
typeset -A model effort turns
model=(I0 claude-sonnet-5-5  I1 claude-sonnet-5-5  I2 claude-opus-5-5  I3 claude-sonnet-5-5
       I4 claude-opus-5-5    I5 claude-sonnet-5-5)
effort=(I0 medium I1 high I2 high I3 high I4 high I5 medium)
turns=(I0 80 I1 100 I2 150 I3 150 I4 120 I5 60)
branch=improve     # every session after I0 starts on this branch
mode=auto
retries=6
retry_wait=1800    # seconds to wait before the third and later attempts of a session
limit_margin=300   # when a session stops at Claude's usage limit, wait this long past the reset time it names
limit_waits=10     # at most this many such waits per session
name_flag=1
deny_flag=1        # pass the extra denials below to every session
# Sessions must never start a launcher or caffeinate themselves (on top of .claude/settings.json).
deny=("Bash(./go_im.zsh:*)" "Bash(zsh go_im.zsh:*)" "Bash(zsh ./go_im.zsh:*)" "Bash(./go_rw.zsh:*)"
      "Bash(zsh go_rw.zsh:*)" "Bash(zsh ./go_rw.zsh:*)" "Bash(./go_s1.zsh:*)"
      "Bash(zsh go_s1.zsh:*)" "Bash(zsh ./go_s1.zsh:*)" "Bash(caffeinate:*)" "Bash(nohup:*)")
[[ -r $D/driver.conf ]] && source $D/driver.conf
mkdir -p .cache/logs $D/notes $D/review

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

log_status() {        # last status word that session $1 wrote to improve/LOG.md, or "none"
  local line
  line=$(grep -E "^## $1 status: " $D/LOG.md 2>/dev/null | tail -1)
  [[ -z $line ]] && { print none; return }
  line=${line#*status: }
  print -- ${line%% *}
}

job_alive() { local p; [[ -r .cache/done/job_$1.pid ]] || return 1; p=$(<.cache/done/job_$1.pid); kill -0 $p 2>/dev/null }

block() { awk -v b="<!-- BEGIN $1 -->" -v e="<!-- END $1 -->" '$0==b{f=1;next} $0==e{f=0} f' "$PLAN" }

prompt_for() {        # $1 session id, $2 new|resume, $3 headless|interactive
  local sid=$1 head
  head="You are session $sid of the AIR 2027 improvement round, working in this folder."
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
  grep -q '^## W5 status: COMPLETE' rewrite/LOG.md 2>/dev/null || { print -u2 "the rewrite is not complete (rewrite/LOG.md)"; ok=0 }
  job_alive S3 && { print -u2 "Step 1's job S3 is still running"; ok=0 }
  [[ -r revision/boundary.zsh ]] || { print -u2 "revision/boundary.zsh missing"; ok=0 }
  [[ -r .claude/settings.json ]] || { print -u2 ".claude/settings.json missing (permissions and denials; go_rev.zsh installs it)"; ok=0 }
  for a in air-reader air-bib-checker air-stats-reviewer air-claim-checker air-referee; do
    [[ -r .claude/agents/$a.md ]] || { print -u2 ".claude/agents/$a.md missing"; ok=0 }
  done
  [[ -x py ]] || chmod +x py
  (( ok ))
}

on_branch() {         # every session after I0 must start on the improve branch
  local b
  b=$(git branch --show-current 2>/dev/null)
  [[ $b == $branch ]] && return 0
  print -u2 "git is on branch '${b:-?}', not '$branch': run  git switch $branch  then ./go_im.zsh auto"
  return 1
}

run_headless() {      # $1 session id, $2 new|resume
  local sid=$1 kind=$2 out=.cache/logs/session_${1}_${2}_$(date +%Y%m%d-%H%M%S).log rc
  local -a extra=()
  (( deny_flag )) && extra+=(--disallowedTools "${deny[@]}")
  (( name_flag )) && extra+=(--name "air-im-$sid")
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

case ${1:-} in
  auto)
    check_setup || exit 2
    for sid in $order; do
      if [[ $sid != I0 ]]; then on_branch || exit 1; fi
      st=$(log_status $sid)
      [[ $st == COMPLETE ]] && continue
      if [[ $st == BLOCKED ]]; then
        print "$sid is BLOCKED: answer in $D/DECISIONS.md, run ./go_im.zsh $sid resume, then ./go_im.zsh auto"; exit 1
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
        if (( lw == 2 )); then print "[$(now)] $sid: Claude's usage limit was reached, and its reset time could not be read ($last_out). Run ./go_im.zsh auto again after it resets."; exit 1; fi
        (( attempt += 1 ))
        if (( attempt >= 2 && attempt < retries )); then print "[$(now)] $sid not complete; waiting $retry_wait s before retrying"; sleep $retry_wait; fi
      done
      if [[ $st == BLOCKED ]]; then print "$sid is BLOCKED: see $D/DECISIONS.md and $D/CHECKPOINT.md"; exit 1; fi
      if [[ $st != COMPLETE ]]; then print "$sid did not complete after $retries attempts; see .cache/logs"; exit 1; fi
    done
    print "[$(now)] improvement round complete: see $D/REPORT.md" ;;
  status)
    for sid in $order; do print -- "$sid  ${model[$sid]}  session $(log_status $sid)"; done
    print -- "branch: $(git branch --show-current 2>/dev/null)" ;;
  I<->)
    sid=$1
    [[ -n ${model[$sid]:-} ]] || { print -u2 "unknown session: $sid"; exit 2 }
    check_setup || exit 2
    if [[ $sid != I0 ]]; then on_branch || exit 1; fi
    kind=new; [[ ${2:-} == resume ]] && kind=resume
    typeset -a ix=()
    (( deny_flag )) && ix+=(--disallowedTools "${deny[@]}")
    (( name_flag )) && ix+=(--name "air-im-$sid")
    # the denial list takes several values, so a flag must follow it before the prompt
    exec claude "${ix[@]}" --model "${model[$sid]}" --effort "${effort[$sid]}" "$(prompt_for $sid $kind interactive)" ;;
  *)
    print -u2 "usage: ./go_im.zsh auto | status | W<n> [resume]"; exit 2 ;;
esac

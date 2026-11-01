#!/bin/zsh
# revision/boundary.zsh start|end <label>
# Files under ~/Downloads/PDM outside "AIR 2027" must not change while a session or job runs.
# Snapshot = mtime, size and path of every file, except virtual environments, git internals, caches,
# Claude Code's own folders and Finder metadata. "end" exits 1 and writes <label>.diff if anything changed.
emulate -L zsh
set -uo pipefail
air=${0:A:h:h}
pdm=${air:h}
dir="$air/.cache/boundary"
mkdir -p "$dir"

snap() {
  ( cd "$pdm" && find . \( -path "./${air:t}" -o -name '.venv-*' -o -name '.git' -o -name '__pycache__' \
        -o -name '.cache' -o -name '.claude' -o -name '.DS_Store' \) -prune -o -type f -print0 \
      | xargs -0 stat -f '%m %z %N' | LC_ALL=C sort )
}

case ${1:-} in
  start)
    snap > "$dir/${2:?label}.start" ;;
  end)
    snap > "$dir/${2:?label}.end"
    if cmp -s "$dir/$2.start" "$dir/$2.end"; then rm -f "$dir/$2.diff"; exit 0; fi
    diff "$dir/$2.start" "$dir/$2.end" > "$dir/$2.diff"
    print -u2 "boundary: $(grep -c '^[<>]' "$dir/$2.diff") changed entries outside AIR 2027 (see $dir/$2.diff)"
    exit 1 ;;
  *)
    print -u2 "usage: revision/boundary.zsh start|end <label>"; exit 2 ;;
esac

#!/bin/zsh
# ./py <args>: the PDM interpreter (../.venv-rwm311: Python 3.11, torch 2.2.2) for this folder's scripts.
# Sets PDM_ROOT, puts scripts/ (and .pylib/, if R0 created it) on the path, writes no bytecode, puts LaTeX on PATH.
emulate -L zsh
here=${0:A:h}
export PYTHONDONTWRITEBYTECODE=1
export PDM_ROOT="${here:h}"
export PYTHONPATH="$here/scripts${PYTHONPATH:+:$PYTHONPATH}"
[[ -d "$here/.pylib" ]] && export PYTHONPATH="$PYTHONPATH:$here/.pylib"
export MPLCONFIGDIR="$here/.cache/matplotlib"
export PATH="/Library/TeX/texbin:/opt/homebrew/bin:/usr/local/bin:$PATH"
exec "${here:h}/.venv-rwm311/bin/python" "$@"

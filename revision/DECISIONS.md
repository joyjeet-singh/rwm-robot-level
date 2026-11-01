# Decisions

## R0
- poppler: `brew install poppler` was running ~40 min compiling Python 3.15 from source on Intel/macOS 13; stopped it and used the pypdf route the plan allows when brew is unavailable. Conservative: no further system changes. Author may install poppler later; the scripts then use it automatically.
- Git identity: left as the existing global one (the plan sets neutral values only when none exists).

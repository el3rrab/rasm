#!/bin/zsh
# Rasm: bring this machine up to date with el3rrab/rasm and rebuild the app.
#
# Run by the launchd agent rasm/com.el3rrab.rasm.update.plist every 6 hours, or by hand.
# It only touches the checkout when it is safe: on branch main, with no uncommitted changes,
# and only by fast-forward. Otherwise it logs why it skipped and leaves everything alone.
set -u

REPO="${RASM_REPO:-$HOME/vectorcraft}"
LOG="$HOME/Library/Logs/rasm-update.log"
export PATH="$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"

log() { print -r -- "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"; }

cd "$REPO" 2>/dev/null || { log "skip: $REPO not found"; exit 0; }

branch=$(git symbolic-ref --short -q HEAD)
if [[ "$branch" != "main" ]]; then
  log "skip: on branch '$branch', not main"
  exit 0
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  log "skip: uncommitted changes"
  exit 0
fi

git fetch --quiet origin || { log "error: fetch failed"; exit 1; }
before=$(git rev-parse HEAD)
if ! git merge --ff-only --quiet origin/main; then
  log "skip: main has local commits not on origin (push them first)"
  exit 0
fi
after=$(git rev-parse HEAD)

if [[ "$before" == "$after" ]]; then
  log "up to date at ${after:0:8}"
  exit 0
fi

log "updated ${before:0:8} -> ${after:0:8} ($(git describe --tags --always)); building"
if cargo build --release -p vectorcraft -p vectorcraft-cli >> "$LOG" 2>&1; then
  log "build ok"
else
  log "error: build failed (see above)"
  exit 1
fi

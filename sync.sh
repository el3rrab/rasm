#!/bin/bash
# ONE way to save this project: this Mac -> GitHub (private). No server copy: this app runs on the Mac, so two copies must match.
#   ./sync.sh "what changed"
# 1. takes the other Mac's pushed work first (fast-forward only: stops if the two Macs both changed things)
# 2. commits every change here in the owner's name and pushes the current branch
# 3. prints the Mac and GitHub commits and requires them to be the same. Any failed step stops with ❌.
set -euo pipefail
cd "$(dirname "$0")"
MSG="${1:-update}"
B=$(git branch --show-current)
[ -n "$B" ] || { echo "❌ not on a branch (detached HEAD)"; exit 1; }

git fetch -q origin
if git rev-parse -q --verify "origin/$B" >/dev/null; then
  git merge -q --ff-only "origin/$B" 2>/dev/null || {
    [ -n "$(git status --porcelain)" ] && echo "❌ GitHub has newer work on $B and there are local edits — commit or stash them, then run again" \
                                       || echo "❌ this Mac and GitHub both have new commits on $B — merge by hand first"
    exit 1; }
fi

git add -A
if ! git diff --cached --quiet; then
  git -c user.name="Islam Adel Salem" -c user.email="7ekayatel3rrab@gmail.com" commit -qm "$MSG"
fi
git push -q -u origin "$B"

LOCAL=$(git rev-parse HEAD); REMOTE=$(git ls-remote origin -h "refs/heads/$B" | cut -f1)
echo "Branch  $B"
echo "Mac     $LOCAL"
echo "GitHub  $REMOTE"
[ "$LOCAL" = "$REMOTE" ] && echo "✅ the two copies are identical" || { echo "❌ copies differ"; exit 1; }

#!/usr/bin/env bash
# Ship the current work as one pull request: stage, commit, push, open the PR, arm auto-merge, watch the required checks, and print the failing tests if CI fails.
# Usage: tools/ship.sh "title" body-file
set -euo pipefail
export LC_ALL=C
title=${1:?usage: tools/ship.sh "title" body-file}; body=${2:?usage: tools/ship.sh "title" body-file}
branch=$(git branch --show-current)
if [ -z "$branch" ] || [ "$branch" = main ]; then echo "STOPPED: switch to a work branch first (git switch -c name)"; exit 1; fi
git fetch -q origin
git add -A
if [ -z "$(git diff --cached --name-only)" ]; then echo "STOPPED: nothing is staged"; exit 1; fi
if [ -x tools/check.sh ] && [ -z "${SKIP_CHECK:-}" ]; then tools/check.sh; fi
{ printf '%s\n\n' "$title"; cat "$body"; } | git commit -q -F -
git push -q --force-with-lease -u origin HEAD
url=$(gh pr view --json url -q .url 2>/dev/null || gh pr create --base main --title "$title" --body-file "$body")
echo "PR: $url"
if gh pr merge --auto --merge >/dev/null 2>&1; then echo "auto-merge armed"; else echo "auto-merge not armed: merge by hand when green"; fi
flags=""; if gh pr checks --help 2>&1 | grep -q -- "--required"; then flags="--required"; fi
for i in 1 2 3 4 5 6 7 8; do
  out=$(gh pr checks $flags 2>&1 || true)
  case "$out" in *"no checks reported"*|*"no required checks"*) echo "waiting for GitHub to register the checks ($i/8)"; sleep 15;; *) break;; esac
done
if gh pr checks $flags --watch --interval 30; then echo "CI green"; else echo "CI FAILED: the failing tests follow"; "$(dirname "$0")/ci_failures.sh" || true; exit 1; fi

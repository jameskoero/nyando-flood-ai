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
  case "$out" in *"no checks reported"*|*"no required checks"*) echo "waiting for GitHub to register the checks ($i/8)"; sleep "${SHIP_NAP:-15}";; *) break;; esac
done
# A dropped connection or a pending check is not a failed check: the verdict is read from the check states, and "unknown" is said when GitHub cannot confirm them.
nap=${SHIP_NAP:-15}
verdict() {
  local o rc=0
  o=$(gh pr checks $flags 2>&1) || rc=$?
  if printf '%s\n' "$o" | awk -F'\t' '$2=="fail"{f=1} END{exit !f}'; then echo failed
  elif [ "$rc" -eq 0 ]; then echo green
  elif printf '%s\n' "$o" | awk -F'\t' '$2=="pending"{p=1} END{exit !p}'; then echo pending
  else echo unknown; fi
}
result=unknown
for round in 1 2 3 4 5 6; do
  gh pr checks $flags --watch --interval 30 || true
  result=$(verdict)
  case "$result" in pending|unknown) echo "checks not settled ($result), looking again ($round/6)"; sleep "$nap";; *) break;; esac
done
case "$result" in
  green) echo "CI green";;
  failed) echo "CI FAILED: the failing tests follow"; "$(dirname "$0")/ci_failures.sh" || true; exit 1;;
  *) echo "CI UNKNOWN: GitHub could not confirm the checks (last state: $result); look with: gh pr checks"; exit 2;;
esac

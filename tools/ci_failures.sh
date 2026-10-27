#!/usr/bin/env bash
# Print the failing tests of the latest CI run on this branch, instead of reading the log by hand.
set -euo pipefail
branch=$(git branch --show-current)
id=$(gh run list --branch "$branch" --workflow CI --limit 1 --json databaseId -q '.[0].databaseId')
if [ -z "$id" ]; then echo "no CI run found for $branch"; exit 1; fi
gh run view "$id" --log-failed | sed -E 's/^.*[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.]+Z //' | grep -E '^(FAILED|ERROR) |^E  |short test summary|[0-9]+ (passed|failed)' | head -60

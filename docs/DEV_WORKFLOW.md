# Development workflow

How work ships from a phone. One persistent shell, one login, one command per change.

## One-time setup (Termux)

    pkg update -y && pkg upgrade -y
    pkg install -y git gh python
    git config --global user.name jameskoero
    git config --global user.email 227142491+jameskoero@users.noreply.github.com
    gh auth login -h github.com -p https -w
    gh auth status
    gh repo clone jameskoero/nyando-flood-ai ~/nyando-flood-ai

`gh auth login` shows a one-time code; open github.com/login/device on the phone, enter it and approve. The login carries the repo and workflow scopes, so no personal access token is made, pasted or ticked. `gh auth refresh -s workflow` adds a scope later. `gh auth logout` removes the stored token; it can also be revoked under github.com/settings/applications.

## The loop

1. `cd ~/nyando-flood-ai && git switch main && git pull -q && git switch -c work-name`
2. Apply the change (a pasted script that writes the files).
3. `tools/ship.sh "title" body-file` stages, commits, pushes, opens the PR, arms auto-merge and watches the required checks. If CI fails it prints the failing tests (`tools/ci_failures.sh`).
4. Fix, amend (`git add -A && git commit --amend --no-edit`) and run `tools/ship.sh` again; the same PR updates.

## Rules that came from failures

- CI is the test oracle: it runs Python 3.11 with the repository secrets and sees only tracked files. Stage files (`git add -A`) before running tests anywhere else.
- Heavy runs (Earth Engine, training, scoring) run in GitHub Actions or Colab, never in the shell that ships.
- A token or key is never pasted into chat or a file in the repository. `gh` keeps its own login; other keys go in repository secrets (`gh secret set NAME`, which prompts for the value).

## Before and after a push (tools/check.sh and tools/live_check.py)

tools/ship.sh runs tools/check.sh after staging and stops, before anything is committed, if one of the repository's structural rules is broken: the README Repository layout must name every tracked root file and folder, every register row must be well formed and name a test that exists, staged files must parse, workflows must not contain ': write', and no key-shaped text may be staged. SKIP_CHECK=1 skips it. It replays the rules of tests/test_repository_hygiene.py and tests/test_roadmap_conformance.py with the standard library, so a missing README line stops on the phone and not in CI.

After the push, tools/ship.sh reads the verdict from the check states. A dropped connection or a pending check is not a failed check: it looks again, up to six times, and says CI UNKNOWN (exit 2) when GitHub cannot confirm the checks. CI FAILED (exit 1) is printed only when a check shows fail, followed by the failing tests. tests/test_ship_tools.py runs ship.sh against a scripted GitHub CLI to prove each case.

tools/live_check.py HOSTNAME is the acceptance check of a deployed service: health, the six security headers, the registered model's SHA-256, /docs and /openapi.json not exposed, and, for a real flood row and a real control row, the same score on /v2/score and /v2/advisory, a valid level, the output guard, the limits text and the ward filter. Run it after every deploy.

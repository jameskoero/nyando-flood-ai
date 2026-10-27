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

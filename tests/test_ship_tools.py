"""Shipping tools (docs/DEV_WORKFLOW.md): tools/ship.sh reads the verdict from the check states and never reports a dropped connection as a failed check, tools/check.sh replays the structural rules before a push, and tools/live_check.py is the acceptance check of a deployed service."""
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAKE_GH = r"""#!/usr/bin/env bash
state="${GH_STATE:?}"
if [ "$1 $2" = "pr view" ] || [ "$1 $2" = "pr merge" ] || [ "$1 $2" = "pr create" ]; then echo "https://example.invalid/pr/1"; exit 0; fi
if [ "$1 $2" != "pr checks" ]; then exit 0; fi
case " $* " in *" --help "*) echo "      --required   Only show required checks"; exit 0;; esac
n=$(cat "$state" 2>/dev/null || echo 0); n=$((n+1)); echo "$n" > "$state"
green() { printf 'data-gate\tpass\t25s\thttps://x/1\nmanifest-check\tpass\t5s\thttps://x/2\ntest\tpass\t3m\thttps://x/3\n'; }
dropped() { echo 'Post "https://api.github.com/graphql": read tcp 10.0.0.1:1->140.82.1.5:443: read: connection reset by peer'; }
case "$GH_SCENARIO" in
  green) green; exit 0;;
  failed) printf 'data-gate\tpass\t25s\thttps://x/1\ntest\tfail\t4m\thttps://x/3\n'; exit 1;;
  dropped_then_green) case $n in 2|3) dropped; exit 1;; *) green; exit 0;; esac;;
  pending_then_green) case $n in 2) printf 'test\tpending\t0\thttps://x/3\n'; exit 1;; 3) printf 'test\tpending\t0\thttps://x/3\n'; exit 8;; *) green; exit 0;; esac;;
  always_dropped) if [ "$n" -eq 1 ]; then green; exit 0; fi; dropped; exit 1;;
esac
"""


def env_for(d, **extra):
    base = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.invalid", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.invalid")
    base.update(extra)
    return base


def run(cmd, cwd, env):
    return subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True)


def ship(scenario):
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "bin").mkdir()
        gh = d / "bin" / "gh"
        gh.write_text(FAKE_GH)
        gh.chmod(0o755)
        env = env_for(d, SHIP_NAP="0", GH_SCENARIO=scenario, GH_STATE=str(d / "state"), PATH=str(d / "bin") + os.pathsep + os.environ["PATH"])
        run(["git", "init", "-q", "--bare", str(d / "remote.git")], d, env)
        work = d / "work"
        (work / "tools").mkdir(parents=True)
        for cmd in (["git", "init", "-q"], ["git", "checkout", "-q", "-b", "main"]):
            run(cmd, work, env)
        (work / "a.txt").write_text("a\n")
        for cmd in (["git", "add", "-A"], ["git", "commit", "-q", "-m", "init"], ["git", "remote", "add", "origin", str(d / "remote.git")], ["git", "push", "-q", "origin", "main"], ["git", "checkout", "-q", "-b", "work"]):
            run(cmd, work, env)
        (work / "tools" / "ship.sh").write_text((ROOT / "tools" / "ship.sh").read_text(encoding="utf-8"))
        (work / "tools" / "ci_failures.sh").write_text("#!/usr/bin/env bash\necho FAKE-CI-FAILURES-RAN\n")
        (work / "tools" / "ci_failures.sh").chmod(0o755)
        (work / "new.txt").write_text("new\n")
        (work / "body.txt").write_text("body\n")
        return run(["bash", "tools/ship.sh", "a title", "body.txt"], work, env)


def test_green_checks_end_with_ci_green_and_a_real_failure_prints_the_failing_tests():
    r = ship("green")
    assert r.returncode == 0 and "CI green" in r.stdout and "auto-merge armed" in r.stdout, r.stdout + r.stderr
    r = ship("failed")
    assert r.returncode == 1 and "CI FAILED" in r.stdout and "FAKE-CI-FAILURES-RAN" in r.stdout, r.stdout + r.stderr


def test_a_dropped_connection_or_a_pending_check_is_never_reported_as_a_failed_check():
    r = ship("dropped_then_green")
    assert r.returncode == 0 and "CI green" in r.stdout and "CI FAILED" not in r.stdout and "checks not settled (unknown)" in r.stdout, r.stdout + r.stderr
    r = ship("pending_then_green")
    assert r.returncode == 0 and "CI green" in r.stdout and "checks not settled (pending)" in r.stdout and "CI FAILED" not in r.stdout, r.stdout + r.stderr


def test_checks_that_cannot_be_confirmed_end_with_ci_unknown_and_exit_2():
    r = ship("always_dropped")
    assert r.returncode == 2 and "CI UNKNOWN" in r.stdout and "CI FAILED" not in r.stdout, r.stdout + r.stderr


def repo_for_check(tmp):
    d = Path(tmp)
    env = env_for(d)
    (d / "docs").mkdir()
    (d / "tests").mkdir()
    (d / "README.md").write_text("# T\n\n## Repository layout\n\n```text\nREADME.md  docs/  tests/\n```\n\n## Next\n\nx\n")
    (d / "docs" / "ROADMAP_DEVIATIONS.md").write_text("| D1 | Deviation | a | b | c | tests/t.py::test_a | closed |\n")
    (d / "tests" / "t.py").write_text("def test_a():\n    assert True\n")
    for cmd in (["git", "init", "-q"], ["git", "add", "-A"]):
        run(cmd, d, env)
    return d, env


def check_in(d, env):
    return run(["bash", str(ROOT / "tools" / "check.sh")], d, env)


def test_check_sh_passes_a_consistent_repository_and_names_each_broken_rule():
    with tempfile.TemporaryDirectory() as tmp:
        d, env = repo_for_check(tmp)
        r = check_in(d, env)
        assert r.returncode == 0 and "none broken" in r.stdout, r.stdout + r.stderr
        broken = (
            ("extra.txt", "x\n", "does not mention"),
            ("bad.py", "def f(:\n", "line 1"),
            (".github/workflows/w.yml", "permissions:\n  contents: write\n", "': write'"),
            ("keys.md", "gsk_" + "a" * 40 + "\n", "key-shaped"),
        )
        for name, text, expect in broken:
            f = d / name
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(text)
            run(["git", "add", "-A"], d, env)
            r = check_in(d, env)
            assert r.returncode == 1 and expect in r.stdout, (name, r.stdout + r.stderr)
            run(["git", "rm", "-q", "-f", name], d, env)
        reg = d / "docs" / "ROADMAP_DEVIATIONS.md"
        reg.write_text(reg.read_text() + "| D2 | Deviation | a | b | c | tests/t.py::test_missing | closed |\n| D3 | Nonsense | a | b | c | open: x | open |\n")
        run(["git", "add", "-A"], d, env)
        r = check_in(d, env)
        assert r.returncode == 1 and "missing test test_missing" in r.stdout and "kind 'Nonsense'" in r.stdout, r.stdout + r.stderr


def test_the_tools_parse_are_executable_and_documented():
    for name in ("ship.sh", "check.sh"):
        p = ROOT / "tools" / name
        assert p.read_text(encoding="utf-8").startswith("#!/usr/bin/env bash") and subprocess.run(["bash", "-n", str(p)]).returncode == 0, name
    assert os.access(ROOT / "tools" / "check.sh", os.X_OK)
    compile((ROOT / "tools" / "live_check.py").read_text(encoding="utf-8"), "live_check.py", "exec")
    doc = (ROOT / "docs" / "DEV_WORKFLOW.md").read_text(encoding="utf-8")
    for must in ("tools/check.sh", "tools/live_check.py", "SKIP_CHECK", "CI UNKNOWN"):
        assert must in doc, must

"""Earth Engine gating (docs/ROADMAP_DEVIATIONS.md D36): the session fixture is opt-in, an empty key is not parsed as JSON, only runs that cannot have the key skip, and tests that need no Earth Engine run without credentials."""
import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PURE = "tests/test_baseline.py::test_unseen_land_cover_class_is_ignored"
GATED = "tests/test_ee_gating.py::test_earth_engine_session_is_usable"


def _run(node, **env_over):
    env = dict(os.environ, GITHUB_ACTIONS="true", GEE_SERVICE_ACCOUNT_KEY="", EE_OPTIONAL="true")
    env.update(env_over)
    r = subprocess.run([sys.executable, "-m", "pytest", node, "-q", "-p", "no:cacheprovider", "-W", "ignore", "-rs"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    return r.returncode, r.stdout + r.stderr


@pytest.mark.usefixtures("ee_session")
def test_earth_engine_session_is_usable():
    import ee
    assert ee.Number(1).getInfo() == 1


def test_fork_and_dependabot_runs_skip_the_gated_test_with_a_reason():
    rc, out = _run(GATED, EE_OPTIONAL="true")
    assert rc == 0 and "skipped" in out and "no Earth Engine credentials" in out, out[-600:]


def test_a_trusted_run_without_the_key_fails_clearly_not_with_a_json_error():
    rc, out = _run(GATED, EE_OPTIONAL="false")
    assert rc != 0 and "empty on a trusted GitHub Actions run" in out and "JSONDecodeError" not in out, out[-600:]


def test_a_key_that_is_not_json_is_reported_without_echoing_it():
    rc, out = _run(GATED, EE_OPTIONAL="false", GEE_SERVICE_ACCOUNT_KEY="not-json-SECRETMARKER")
    assert rc != 0 and "not valid JSON" in out and "SECRETMARKER" not in out, out[-600:]


def test_a_test_that_needs_no_earth_engine_passes_without_credentials():
    rc, out = _run(PURE)
    assert rc == 0 and "passed" in out and "failed" not in out, out[-600:]


def test_the_session_fixture_is_not_autouse():
    tree = ast.parse((ROOT / "tests/conftest.py").read_text(encoding="utf-8"))
    fx = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "ee_session"]
    assert fx and not any("autouse" in ast.dump(d) for d in fx[0].decorator_list)


def test_ci_marks_only_secretless_pull_requests_as_optional():
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for s in ("EE_OPTIONAL:", "github.event_name == 'pull_request'", "dependabot[bot]", "head.repo.full_name != github.repository", "::notice::", "pytest -v -rs"):
        assert s in text, s

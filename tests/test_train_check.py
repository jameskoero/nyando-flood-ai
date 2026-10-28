"""train-check (register row D13; roadmap Section 2): the weekly workflow runs the two measured tests on Python 3.11 under the CI pins, with a read-only token and no secret."""
from pathlib import Path
import pytest

WF = (Path(__file__).resolve().parent.parent / ".github" / "workflows" / "train-check.yml").read_text(encoding="utf-8")


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this check never touches Earth Engine."""
    yield


def test_the_workflow_is_weekly_manual_and_off_the_top_of_the_hour():
    assert "schedule:" in WF and "workflow_dispatch:" in WF
    cron = [l for l in WF.splitlines() if l.strip().startswith("- cron:")]
    assert len(cron) == 1
    minute, hour, dom, month, dow = cron[0].split('"')[1].split()
    assert minute not in ("0", "00") and dom == "*" and month == "*" and dow.isdigit()


def test_the_workflow_is_read_only_and_uses_no_secret():
    assert "permissions:\n  contents: read" in WF and ": write" not in WF and "secrets." not in WF
    assert "persist-credentials: false" in WF and "timeout-minutes" in WF and '"3.11"' in WF


def test_the_workflow_runs_the_two_measured_tests_under_the_ci_pins():
    assert "-c constraints-ci.txt" in WF
    assert "python -m pytest -q -p no:cacheprovider tests/test_environment_parity.py tests/test_phase_e_results.py" in WF

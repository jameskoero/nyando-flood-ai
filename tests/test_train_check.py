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


def test_the_register_and_the_readme_record_train_check_and_keep_d13_open():
    root = Path(__file__).resolve().parent.parent
    row = [l for l in (root / "docs" / "ROADMAP_DEVIATIONS.md").read_text(encoding="utf-8").splitlines() if l.startswith("| D13 |")]
    assert len(row) == 1 and "train-check built" in row[0] and "PR #77" in row[0]
    cells = [c.strip() for c in row[0].strip().strip("|").split("|")]
    assert cells[-1] == "open" and cells[-2].startswith("open:") and "drift-monitor and deploy" in cells[-2]
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "- [x] **Gates (train-check):**" in readme and "- [ ] **Gates (remainder):** `drift-monitor` and a gated `deploy` workflow" in readme

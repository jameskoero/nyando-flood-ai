"""The recorded state of the security work (docs/SECURITY_BASELINE.md, README): the per-client rate limit is recorded with its measurement, and the README no longer lists the built train-check as missing."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = (ROOT / "docs" / "SECURITY_BASELINE.md").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_the_baseline_records_the_rate_limit_fix_with_its_measurement():
    items = BASE.split("## Open items", 1)[1]
    for must in ("rate limit", "tools/burst_check.py", "10 admitted, 1 refused", "17 to 21 of 25", "not isolated"):
        assert must in items, must
    assert "the cause of the earlier leak is not established" not in items


def test_the_readme_lists_only_the_workflows_that_are_not_built():
    assert "the scheduled `train-check` and `drift-monitor`" not in README
    line = [l for l in README.splitlines() if l.startswith("- **Not built yet:**")]
    assert len(line) == 1 and "`drift-monitor`" in line[0] and "`train-check` is built" in line[0]

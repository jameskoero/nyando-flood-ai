"""The shipping tools parse and the workflow document names them (docs/DEV_WORKFLOW.md)."""
import subprocess
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this check never touches Earth Engine."""
    yield


def test_the_tools_parse_and_the_workflow_document_names_them():
    for name in ("ship.sh", "ci_failures.sh"):
        p = ROOT / "tools" / name
        assert p.read_text(encoding="utf-8").startswith("#!/usr/bin/env bash"), name
        assert subprocess.run(["bash", "-n", str(p)], capture_output=True).returncode == 0, name
    doc = (ROOT / "docs" / "DEV_WORKFLOW.md").read_text(encoding="utf-8")
    assert "tools/ship.sh" in doc and "tools/ci_failures.sh" in doc and "gh auth login" in doc and "never pasted into chat" in doc

"""CI gate workflow (docs/ROADMAP_DEVIATIONS.md D23): exact pins, a retried install, and pins that cover the gate's imports."""
import ast
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PIN_FILE = ROOT / "requirements-gate.txt"
WORKFLOW = ROOT / ".github" / "workflows" / "data-gate.yml"
DIST = {"sklearn": "scikit-learn"}


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this check never touches Earth Engine."""
    yield


def _pins():
    out = {}
    for line in PIN_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            assert re.fullmatch(r"[A-Za-z0-9_.\-]+==[A-Za-z0-9_.!+\-]+", line), "not an exact pin: " + line
            name, version = line.split("==")
            out[name.lower().replace("_", "-")] = version
    return out


def test_gate_pins_are_exact():
    pins = _pins()
    assert len(pins) >= 5 and {"numpy", "pandas", "scipy", "scikit-learn", "pytest"} <= set(pins)


def test_pins_cover_every_third_party_import_of_the_gate_tests():
    tree = ast.parse((ROOT / "tests" / "test_data_gate.py").read_text(encoding="utf-8"))
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            mods.add(n.module.split(".")[0])
    local = {p.name for p in ROOT.iterdir()} | {p.stem for p in ROOT.glob("*.py")}
    third = {m for m in mods if m not in sys.stdlib_module_names and m not in local}
    pins = _pins()
    missing = {m for m in third if DIST.get(m, m).lower().replace("_", "-") not in pins}
    assert not missing, "the gate tests import packages that requirements-gate.txt does not pin: %s" % sorted(missing)


def test_gate_workflow_installs_pinned_dependencies_with_retries():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "-r requirements-gate.txt" in text and "for attempt in 1 2 3" in text and "--retries" in text
    assert "tests/test_data_gate.py" in text and "permissions:" in text
    assert re.search(r"^\s*pull_request:\s*$", text, re.M), "the gate must run on every pull request"

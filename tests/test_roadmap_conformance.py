"""Roadmap conformance: the deviation register is traceable, no oversampling in V2 model code, decision rules behave as written."""
import ast
import re
from pathlib import Path

import pytest

from src.models.decision import claim_beyond_elevation, decide, decide_frame

ROOT = Path(__file__).resolve().parent.parent
REGISTER = ROOT / "docs" / "ROADMAP_DEVIATIONS.md"


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this check never touches Earth Engine."""
    yield


def _rows():
    out = []
    for line in REGISTER.read_text(encoding="utf-8").splitlines():
        if line.startswith("| D"):
            out.append([c.strip() for c in line.strip().strip("|").split("|")])
    return out


def _functions(path):
    return {n.name for n in ast.parse(path.read_text(encoding="utf-8")).body if isinstance(n, ast.FunctionDef)}


def test_register_rows_are_complete_and_traceable():
    rows = _rows()
    assert len(rows) >= 17
    assert [r[0] for r in rows] == ["D%d" % i for i in range(1, len(rows) + 1)]
    for r in rows:
        assert len(r) == 7 and all(r), r
        _, kind, _, _, _, ver, status = r
        assert kind in {"Deviation", "Improvement", "Open"}, r[0]
        assert status.startswith("open") or status == "closed", r[0]
        if ver.startswith("tests/"):
            rel, name = ver.split("::")
            assert (ROOT / rel).exists(), r[0] + ": missing file " + rel
            assert name in _functions(ROOT / rel), r[0] + ": missing test " + name
        elif ver.startswith("run:"):
            assert re.fullmatch(r"run:[0-9a-f]{32}", ver), r[0]
        else:
            assert ver.startswith("open:") and status.startswith("open"), r[0] + ": a closed row needs a test or a run"


def test_no_oversampling_in_the_v2_model_code():
    files = ["src/models/cv.py", "src/models/baseline.py", "src/models/boosters.py", "src/models/decision.py",
             "scripts/run_baseline.py", "scripts/run_reference.py", "scripts/run_mappable.py", "scripts/run_boosters.py",
             "src/models/constrained.py", "src/models/registration.py", "scripts/run_constrained.py"]
    bad = []
    for rel in files:
        assert (ROOT / rel).exists(), rel
        for n in ast.walk(ast.parse((ROOT / rel).read_text(encoding="utf-8"))):
            if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("imblearn"):
                bad.append(rel)
            if isinstance(n, ast.Import) and any(a.name.startswith("imblearn") for a in n.names):
                bad.append(rel)
            if isinstance(n, ast.Name) and n.id == "SMOTE":
                bad.append(rel)
    assert not bad, "oversampling found in " + ", ".join(sorted(set(bad)))


def _r(h, x, hm, xm, x_vs_h):
    return {"hgb:con": {"beats": h, "mean": hm}, "xgb:con": {"beats": x, "mean": xm}, "xgb_vs_hgb_ci_low": x_vs_h}


def test_decision_rules_follow_section_13_4():
    """Arithmetic test vectors for the decision logic only. They are not data and no result is claimed from them."""
    assert decide_frame(_r(True, True, 0.95, 0.96, 0.002)) == "xgb:con"
    assert decide_frame(_r(True, True, 0.95, 0.96, -0.001)) == "hgb:con"
    assert decide_frame(_r(True, True, 0.96, 0.95, 0.010)) == "hgb:con"
    assert decide_frame(_r(True, False, 0.95, 0.90, 0.0)) == "hgb:con"
    assert decide_frame(_r(False, True, 0.90, 0.95, 0.0)) == "xgb:con"
    assert decide_frame(_r(False, False, 0.90, 0.90, 0.0)) == "logistic"
    only_hgb, only_xgb, neither = _r(True, False, 0.95, 0.90, 0.0), _r(False, True, 0.90, 0.95, 0.0), _r(False, False, 0.90, 0.90, 0.0)
    assert decide(only_hgb, only_hgb) == "hgb:con"
    assert decide(only_hgb, only_xgb) == "unresolved"
    assert decide(neither, neither) == "logistic"


def test_claim_beyond_elevation_needs_a_positive_lower_bound():
    assert claim_beyond_elevation(0.001) is True
    assert claim_beyond_elevation(0.0) is False
    assert claim_beyond_elevation(-0.01) is False

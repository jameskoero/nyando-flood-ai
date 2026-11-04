"""tests/test_legacy_predict_removed.py -- register row D51: the legacy POST /predict route is removed from the code.

The route served a retracted v1 model. It answered 410 Gone since 30 October 2026 and no call was recorded after that day.
These checks keep it gone: no route, no gate, no leftover switch, and nothing in the tests or workflows that still calls it.
Standard library only; no Earth Engine, no network, no secrets.
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAIN = (ROOT / "backend" / "main.py").read_text(encoding="utf-8")


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this test never touches Earth Engine."""
    yield


def _routes():
    out = []
    for node in ast.walk(ast.parse(MAIN)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in node.decorator_list:
                if isinstance(d, ast.Call) and d.args and isinstance(d.args[0], ast.Constant) and isinstance(d.args[0].value, str):
                    out.append(d.args[0].value)
    return out


def test_the_legacy_predict_route_is_gone():
    routes = _routes()
    assert "/predict" not in routes, routes
    assert {"/v2/score", "/v2/advisory", "/v2/metrics", "/health"} <= set(routes), routes
    for gone in ("LEGACY_PREDICT", "legacy_predict_attempt", "_legacy_gate"):
        assert gone not in MAIN, gone


def test_the_v2_score_input_keeps_its_ward_length_limit():
    cls = next(n for n in ast.walk(ast.parse(MAIN)) if isinstance(n, ast.ClassDef) and n.name == "ScoreInput")
    assert "max_length=64" in ast.get_source_segment(MAIN, cls)


def test_no_test_or_workflow_still_calls_the_legacy_route():
    me = Path(__file__).name
    files = list((ROOT / "tests").glob("*.py")) + list((ROOT / ".github" / "workflows").glob("*.yml"))
    offenders = [p.name for p in files if p.name != me and re.search(r"post\(\s*[\"']/predict", p.read_text(encoding="utf-8"))]
    assert not offenders, offenders

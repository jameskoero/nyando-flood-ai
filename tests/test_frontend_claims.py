"""tests/test_frontend_claims.py -- the dashboard must not show retracted or unsupported claims.

The V2 rebuild retracted the v1 metrics and the README says this is not an early-warning system, so
the live dashboard may not contradict either. Hard-coded metric values are also banned (roadmap
Phase F: every number comes from /metrics). A dynamic label such as "AUC" is fine; a literal
"AUC 0.97" is not. Standard library only; no Earth Engine, no network, no secrets.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BANNED_TEXT = ("early warning system", "residents protected", "model accuracy", "immediate alert recommended",
               "conditions stable", "routine monitoring", "early warning broadcast", "undp/usaid/gcf",
               "floods annually", "displacing thousands", "ward-level flood risk predictions", "2,308 gee")
HARDCODED_METRIC = re.compile(r"\b(AUC(-ROC)?|F1|CV)\b\s*[=:]?\s*\d\.\d+")


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this test never touches Earth Engine."""
    yield


def _sources():
    roots = [p.parent / "src" for p in REPO.glob("*/package.json") if (p.parent / "src").is_dir()]
    files = [f for src in roots for ext in ("*.js", "*.jsx", "*.ts", "*.tsx") for f in src.rglob(ext)]
    assert files, "no frontend sources found under any */src folder"
    return {p: p.read_text(encoding="utf-8") for p in files}


def test_dashboard_has_no_banned_claims():
    hits = [(p.name, w) for p, s in _sources().items() for w in BANNED_TEXT if w in s.lower()]
    assert not hits, f"dashboard shows claims the README retracted or denies: {hits}"


def test_dashboard_has_no_hardcoded_model_metrics():
    hits = [(p.name, m.group(0)) for p, s in _sources().items() for m in HARDCODED_METRIC.finditer(s)]
    assert not hits, f"hard-coded model metrics in the dashboard (read them from /metrics): {hits}"


def test_dashboard_says_the_model_is_not_validated():
    n = sum(s.lower().count("not validated") for s in _sources().values())
    assert n >= 2, "the dashboard must say, in the header and in each result message, that the legacy model is not validated"


def test_no_legacy_dashboard_folder():
    assert not (REPO / 'dashboard').exists(), 'legacy dashboard/ held retracted claims; do not restore'

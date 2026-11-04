"""tests/test_dashboard_theme.py -- the dashboard's artwork stays decorative and the colour rules stay readable.

The hero (splashes, rain, droplets, waves) is an illustration. It may not read data, it must say it is an
illustration, and it must be hidden from assistive technology. Crimson is a brand accent; inside the charts, red
is reserved for the HIGH risk class. Motion must respect prefers-reduced-motion. Standard library only.
"""

import re
from pathlib import Path

import pytest

DASH = Path(__file__).resolve().parents[1] / "frontend" / "src" / "dashboard"


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this test never touches Earth Engine."""
    yield


def _read(name):
    return (DASH / name).read_text(encoding="utf-8")


def test_the_hero_is_decorative_and_says_so():
    hero = _read("Hero.jsx")
    assert "illustration only" in hero.lower() and 'aria-hidden="true"' in hero
    for forbidden in ("fetch(", "useEffect", "useState", "evidence", "import "):
        assert forbidden not in hero, "the hero must not read data: " + forbidden


def test_motion_respects_reduced_motion_and_moves_only_transform_and_opacity():
    css = _read("dashboard.css")
    assert "prefers-reduced-motion: reduce" in css
    for block in re.findall(r"@keyframes [\w-]+ \{(.*?)\}\s*(?=@keyframes|@media|\n/\*|\Z)", css, re.S):
        assert not re.search(r"\b(width|height|top|left|margin|padding)\s*:", block), block


def test_red_is_reserved_for_the_high_risk_class_inside_the_charts():
    css = _read("dashboard.css")
    for rule in re.findall(r"^\.(?:bar|whisker[\w. ]*|dot[\w. ]*|track|grid|zero|tick|sub|val|lab)\b[^{]*\{([^}]*)\}", css, re.M):
        assert not re.search(r"--(?:crimson|scarlet|blood|high)\b", rule), rule
    charts = _read("charts.jsx")
    assert charts.count("var(--high)") == 1, "only the risk-class map may use the HIGH colour"


def test_the_hero_motion_starts_paused_and_runs_only_after_the_page_settles():
    css = _read("dashboard.css")
    assert re.search(r"\.nd:not\(\.alive\) \.hero-art \* \{[^}]*animation-play-state: paused", css), "idle until the page has settled"
    dash = _read("Dashboard.jsx")
    assert 'className={"nd" + (alive ? " alive" : "")}' in dash and "setTimeout(wake" in dash
    for event in ("pointerdown", "keydown", "scroll"):
        assert event in dash

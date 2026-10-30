"""Dashboard accessibility and page metadata (README, Phase F; Lighthouse accessibility and SEO audits): every control has an accessible name, the page has a main landmark, muted text meets the WCAG AA contrast on every dashboard background, and the page title, description and robots.txt say what the dashboard is."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = (ROOT / "frontend" / "src" / "App.jsx").read_text(encoding="utf-8")
HTML = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
BACKGROUNDS = ("#060E1A", "#0A1628", "#0D1F30", "#0D2137", "#2D0A0A", "#2D1A00", "#0A2D0F")
MUTED = "#7F9BBB"


def luminance(hex_color):
    def lin(c):
        c = c / 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def contrast(a, b):
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def test_every_control_has_an_accessible_name():
    slider = APP[APP.index("function Slider("):].split("\nfunction ")[0].split("\nexport default")[0]
    assert slider.count("<input") == 1 and "<input aria-label=" in slider
    assert len(re.findall(r"<select\b", APP)) == 2
    assert sorted(re.findall(r'<select aria-label="([^"]+)"', APP)) == ["Land cover class", "Ward"]


def test_the_page_has_a_main_landmark():
    assert 'role="main"' in APP


def test_muted_text_meets_wcag_aa_on_every_dashboard_background():
    assert MUTED in APP and "#5b7a9a" not in APP.lower()
    for bg in BACKGROUNDS:
        assert contrast(MUTED, bg) >= 4.5, bg


def test_the_band_colours_meet_wcag_aa_on_their_own_backgrounds():
    found = re.findall(r'(HIGH|MEDIUM|LOW): \{ color: "(#[0-9A-Fa-f]{6})", bg: "(#[0-9A-Fa-f]{6})"', APP)
    assert len(found) == 3
    for name, fg, bg in found:
        assert contrast(fg, bg) >= 4.5, (name, fg, bg)


def test_the_page_title_description_and_robots_say_what_the_dashboard_is():
    title = re.search(r"<title>([^<]*)</title>", HTML).group(1)
    assert "Susceptibility" in title and "prediction" not in title.lower()
    assert re.search(r'<meta name="description" content="[^"]*not a flood probability[^"]*"', HTML)
    for banned in ("ward-level flood risk prediction", "early warning system"):
        assert banned not in HTML.lower()
    robots = (ROOT / "frontend" / "public" / "robots.txt").read_text(encoding="utf-8")
    assert robots.startswith("User-agent:")

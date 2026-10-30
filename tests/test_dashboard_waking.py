"""Dashboard cold start: after a short wait the dashboard says that the free-tier server may be waking up. File checks only, so these run on the phone too; the build and lint run in CI."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = (ROOT / "frontend" / "src" / "App.jsx").read_text(encoding="utf-8")
WAKE_TEXT = "Waking the free-tier server. After a quiet period the first assessment can take up to a minute; the result appears here as soon as it is ready."


def test_the_waking_status_appears_only_while_loading_and_after_a_short_wait():
    assert "const [waking, setWaking] = useState(false);" in APP
    wake = re.search(r"setTimeout\(\(\) => setWaking\(true\), (\d+)\)", APP)
    limit = re.search(r"setTimeout\(\(\) => ctl\.abort\(\), (\d+)\)", APP)
    assert wake and limit
    assert 1000 <= int(wake.group(1)) <= 10000 and int(wake.group(1)) < int(limit.group(1))
    assert "{loading && waking && (" in APP


def test_the_waking_timer_is_always_cleared():
    tail = APP.split("} finally {", 1)[1].split("};", 1)[0]
    for needle in ("clearTimeout(timer);", "clearTimeout(wake);", "setWaking(false);", "setLoading(false);"):
        assert needle in tail, needle


def test_the_status_is_announced_to_screen_readers_and_keeps_the_muted_colour():
    block = APP.split("{loading && waking && (", 1)[1].split(")}", 1)[0]
    assert 'role="status"' in block and "#7F9BBB" in block and WAKE_TEXT in block


def test_the_waking_text_makes_no_claim_the_claims_guard_forbids():
    for banned in ("probab", "warning", "predict", "alert", "accura", "real-time", "instant"):
        assert banned not in WAKE_TEXT.lower(), banned


def test_the_dashboard_contract_lines_are_untouched():
    assert 'const API_URL = "https://nyando-flood-v2.onrender.com/v2/advisory";' in APP
    assert "/predict" not in APP and "nyando-flood-api" not in APP
    assert "res.status === 429" in APP and "res.status === 503" in APP and 'role="main"' in APP

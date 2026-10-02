"""Registration rules (docs/PHASE_C_PROTOCOL.md, Section 14): the rules as code, on arithmetic test vectors."""
import importlib.util
from pathlib import Path

import pytest

from src.models.registration import CHAMPION, CONSTRAINED_FEATURES, REGISTERED_TRAINING_FRAME, eligible, registered

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this check never touches Earth Engine."""
    yield


def _frame(h, x, hm=0.95, xm=0.95, xh=-0.01):
    return {"hgb:con": {"beats": h, "mean": hm}, "xgb:con": {"beats": x, "mean": xm}, "xgb_vs_hgb_ci_low": xh}


def test_registered_follows_the_rules():
    """Arithmetic test vectors for the decision logic only; they are not data and no result is claimed from them."""
    f, t = _frame(False, False), _frame(True, True, 0.96, 0.97, 0.002)
    assert registered(f, f) == CHAMPION
    assert registered(t, f) == CHAMPION and registered(f, t) == CHAMPION
    assert registered(_frame(True, False), _frame(True, False)) == "hgb:con"
    assert registered(_frame(False, True), _frame(False, True)) == "xgb:con"
    assert registered(t, t) == "xgb:con"
    assert registered(_frame(True, True, 0.96, 0.97, -0.001), t) == "hgb:con"
    assert registered(_frame(True, True, 0.97, 0.96, 0.01), t) == "hgb:con"


def test_eligibility_needs_zero_violations_on_every_constrained_feature():
    ok = {f: 0 for f in CONSTRAINED_FEATURES}
    assert eligible(ok)
    assert not eligible({**ok, "slope": 200})
    assert not eligible({"slope": 0})


def test_registered_training_frame_is_the_mappable_frame():
    assert REGISTERED_TRAINING_FRAME == "mappable"


def test_protocol_section_14_states_the_rules_and_the_candidate():
    text = (ROOT / "docs" / "PHASE_C_PROTOCOL.md").read_text(encoding="utf-8")
    assert "## 14. Registration protocol" in text
    for tag in ("R1.", "R2.", "R3.", "R4.", "R5.", "R6.", "R7."):
        assert tag in text, tag
    assert "logistic:con" in text and "trained on the mappable frame" in text


def test_versus_champion_on_arithmetic_vectors():
    spec = importlib.util.spec_from_file_location("run_constrained", ROOT / "scripts" / "run_constrained.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    champ = {"e%d" % i: 0.90 + 0.001 * i for i in range(12)}
    assert mod.versus_champion(champ, {k: v + 0.05 for k, v in champ.items()})["beats"] is True
    assert mod.versus_champion(champ, {k: v - 0.05 for k, v in champ.items()})["beats"] is False
    assert mod.versus_champion(champ, dict(champ))["beats"] is False

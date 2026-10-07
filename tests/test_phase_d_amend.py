"""Phase D amendments r3 and r4 (docs/PHASE_D_PROTOCOL.md Sections 10 and 11; register row D46): a scorable event has both classes, the early decision follows X3, and the sections state both rules."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.models import phase_d as D
from src.models.phase_d_stats import aucs

TEXT = (Path(__file__).resolve().parent.parent / "docs" / "PHASE_D_PROTOCOL.md").read_text(encoding="utf-8")
S10, S11 = TEXT.split("## 10. ")[1].split("## 11. ")[0], TEXT.split("## 11. ")[1]


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this check never touches Earth Engine."""
    yield


def test_a_scorable_event_has_both_classes_and_one_row_of_each_is_enough():
    df = pd.DataFrame({"event_id": list("aaaaabbbbbcccc"), "flooded": [1, 0, 0, 0, 0] + [0] * 5 + [1, 1, 0, 0]})
    scores = np.array([.9, .1, .2, .3, .4] + [.5] * 5 + [.9, .8, .1, .2])
    assert D.MIN_CLASS_N == 1 and aucs(df, scores) == {"a": 1000000, "c": 1000000}


def test_the_early_decision_follows_x3_on_the_selected_models_violations():
    zero = {"rainfall_3day": 0, "elevation": 0, "distance_river": 0, "slope": 0}
    assert D.x3_ok(zero) and D.early_decision(zero) is None
    assert not D.x3_ok(dict(zero, slope=1)) and D.early_decision(dict(zero, slope=1)) == D.NOT_PROMOTED


def test_the_sections_state_both_rules_and_what_was_seen_before_them():
    assert "MIN_CLASS_N = %d" % D.MIN_CLASS_N in S10 and "withdrawn" in S10 and "0.841302" in S10 and "Disclosure" in S10
    assert "early_decision" in S11 and D.NOT_PROMOTED in S11 and "sealed" in S11 and "counts were seen before" in S11

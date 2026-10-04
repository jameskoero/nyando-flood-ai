"""Run script helpers (docs/ROADMAP_DEVIATIONS.md D28): the functions scripts/run_robustness.py composes work on real rows."""
import importlib.util
from pathlib import Path

import pytest

from src.models.constrained import build_constrained_logistic
from src.models.cv import EVENT, FEATURE_SETS, LABEL, load_training_frame

ROOT = Path(__file__).resolve().parent.parent
SIX = FEATURE_SETS["six"]


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this check never touches Earth Engine."""
    yield


@pytest.fixture(scope="module")
def subset():
    df, _ = load_training_frame()
    g = df.groupby(EVENT)[LABEL].agg(["size", "sum"])
    both = sorted(g[(g["sum"] > 0) & (g["sum"] < g["size"])].index)[:6]
    return df[df[EVENT].isin(both)].reset_index(drop=True)


def _script():
    spec = importlib.util.spec_from_file_location("run_robustness", ROOT / "scripts" / "run_robustness.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_events_for_and_put_stats_work_on_real_rows(subset):
    mod = _script()
    ev = mod.events_for(build_constrained_logistic, subset, SIX)
    assert ev and all(0.0 <= v <= 1.0 for v in ev.values())
    m = mod.put_stats("x", mod.paired(ev, ev))
    assert set(m) == {"x/" + k for k in mod.STAT_KEYS} and all(isinstance(v, float) for v in m.values())

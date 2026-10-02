"""Monotonicity (roadmap Section 1, protocol Section 13.5): constrained boosters must not move against the declared directions."""
import numpy as np
import pandas as pd
import pytest

from src.models.boosters import make_estimator, monotone_violations
from src.models.cv import CONSTRAINTS, FEATURE_SETS, LABEL, load_layer_flags, load_training_frame, restrict_to_mappable

FEATS = FEATURE_SETS["six"]
HEAVIEST = {"hgb": {"max_depth": 4, "max_iter": 300, "min_samples_leaf": 20},
            "xgb": {"max_depth": 4, "n_estimators": 300, "min_child_weight": 1}}


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this check never touches Earth Engine."""
    yield


class _Fake:
    """Arithmetic test vector: a score that falls as elevation rises. Not data."""

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(X["elevation"].to_numpy(float) / 100.0))
        return np.column_stack([1 - p, p])


def test_the_probe_counts_violations_against_a_known_function():
    X = pd.DataFrame({"elevation": np.linspace(1100, 2500, 50), "slope": 1.0})
    assert monotone_violations(_Fake(), X, {"elevation": -1}) == {"elevation": 0}
    assert monotone_violations(_Fake(), X, {"elevation": +1}) == {"elevation": 50}


def test_constrained_boosters_have_no_violations_on_the_mappable_frame():
    df, _ = load_training_frame()
    mapp = restrict_to_mappable(df, load_layer_flags(df))
    X, y = mapp[FEATS], mapp[LABEL].to_numpy()
    for kind in ("hgb", "xgb"):
        model = make_estimator(kind, FEATS, True)(HEAVIEST[kind]).fit(X, y)
        assert monotone_violations(model, X, CONSTRAINTS) == {f: 0 for f in CONSTRAINTS}, kind

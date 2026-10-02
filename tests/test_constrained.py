"""Sign-constrained logistic (docs/PHASE_C_PROTOCOL.md, Section 14): same objective as scikit-learn, bounds hold, optimality holds."""
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from src.models.baseline import build_logistic
from src.models.boosters import monotone_violations
from src.models.constrained import BoundedLogistic, _objective, build_constrained_logistic, constraint_bounds, padded_bounds
from src.models.cv import CONSTRAINTS, FEATURE_SETS, LABEL, load_layer_flags, load_training_frame, restrict_to_mappable

FEATS = FEATURE_SETS["six"]
NUM = [f for f in FEATS if f != "land_cover"]


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this check never touches Earth Engine."""
    yield


@pytest.fixture(scope="module")
def full():
    df, _ = load_training_frame()
    return df


@pytest.fixture(scope="module")
def mappable(full):
    return restrict_to_mappable(full, load_layer_flags(full))


@pytest.fixture(scope="module")
def design(mappable):
    prep = build_logistic(FEATS).named_steps["prep"]
    return prep.fit_transform(mappable[FEATS]), mappable[LABEL].to_numpy().astype(float)


def test_unbounded_fit_equals_scikit_learn(design):
    X, y = design
    ours = BoundedLogistic().fit(X, y)
    ref = LogisticRegression(C=1.0, tol=1e-10, max_iter=20000).fit(X, y)
    assert np.allclose(ours.coef_, ref.coef_[0], atol=2e-3) and abs(ours.intercept_ - ref.intercept_[0]) < 2e-3


def test_bounds_hold_and_the_kkt_conditions_hold(design):
    X, y = design
    lower, upper = constraint_bounds(FEATS)
    m = BoundedLogistic(lower=lower, upper=upper).fit(X, y)
    lo, hi = padded_bounds(lower, upper, X.shape[1])
    assert (m.coef_ >= lo - 1e-12).all() and (m.coef_ <= hi + 1e-12).all()
    _, g = _objective(np.append(m.coef_, m.intercept_), X, y, m.C)
    for j in range(X.shape[1]):
        if np.isfinite(hi[j]) and abs(m.coef_[j] - hi[j]) < 1e-9:
            assert g[j] <= 1e-3, (j, g[j])
        elif np.isfinite(lo[j]) and abs(m.coef_[j] - lo[j]) < 1e-9:
            assert g[j] >= -1e-3, (j, g[j])
        else:
            assert abs(g[j]) < 1e-3, (j, g[j])
    assert abs(g[-1]) < 1e-3


def test_the_logged_unconstrained_signs_reproduce_on_the_full_frame(full):
    pipe = build_logistic(FEATS).fit(full[FEATS], full[LABEL])
    coef = pipe.named_steps["clf"].coef_[0]
    signs = {f: int(np.sign(coef[NUM.index(f)])) for f in CONSTRAINTS}
    assert signs == {"rainfall_3day": 1, "elevation": -1, "distance_river": -1, "slope": 1}


def test_the_constrained_pipeline_has_no_violations_on_real_rows(mappable):
    con = build_constrained_logistic(FEATS).fit(mappable[FEATS], mappable[LABEL].to_numpy())
    assert monotone_violations(con, mappable[FEATS], CONSTRAINTS) == {f: 0 for f in CONSTRAINTS}


def test_the_constrained_pipeline_clones_and_refits_identically(mappable):
    X, y = mappable[FEATS], mappable[LABEL].to_numpy()
    a = build_constrained_logistic(FEATS).fit(X, y)
    b = clone(build_constrained_logistic(FEATS)).fit(X, y)
    assert np.allclose(a.predict_proba(X), b.predict_proba(X), atol=1e-12)


def test_scikit_learn_logistic_regression_has_no_coefficient_bounds_option():
    params = LogisticRegression().get_params()
    assert not {"bounds", "positive", "lower_bounds", "upper_bounds"} & set(params)

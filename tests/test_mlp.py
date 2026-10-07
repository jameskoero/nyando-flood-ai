"""Phase D Stage 1 (register row D46): the torch part of the MLP. It runs where torch is installed (Colab) and is skipped in CI, which has no torch; the scheduled train-check workflow is register row D13."""
from pathlib import Path
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from src.models import phase_d as D
from src.models.cv import FLAGS_PATH, LABEL, load_layer_flags, load_training_frame, loeo_splits, out_of_fold_scores, restrict_to_mappable
from src.models.mlp import FEATURES, PhysicsMLP, build_net, constraint_penalty

ROOT = Path(__file__).resolve().parent.parent
DF, _ = load_training_frame(ROOT)
FRAME = restrict_to_mappable(DF, load_layer_flags(DF, FLAGS_PATH)).reset_index(drop=True)
X, Y = FRAME[list(FEATURES)], FRAME[LABEL].to_numpy()


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this check never touches Earth Engine."""
    yield


def _linear(w_elevation, w_rainfall):
    net = torch.nn.Linear(D.N_INPUTS, 1)
    with torch.no_grad():
        net.weight.zero_()
        net.bias.zero_()
        net.weight[0, 0], net.weight[0, 2] = w_elevation, w_rainfall
    return net


def test_the_networks_have_the_parameter_counts_the_protocol_states():
    assert [sum(p.numel() for p in build_net(h).parameters()) for h in D.HIDDEN] == [D.param_count(h) for h in D.HIDDEN] == [1025, 3073]


def test_the_penalty_is_zero_when_the_constraints_hold_and_positive_when_one_is_violated():
    x = torch.zeros(3, D.N_INPUTS, requires_grad=True)
    assert float(constraint_penalty(_linear(-1.0, 1.0), x)) == 0.0
    assert float(constraint_penalty(_linear(1.0, 1.0), x)) == pytest.approx(0.25, abs=1e-6)


def test_fit_is_deterministic_for_a_seed_and_returns_probabilities():
    a = PhysicsMLP(epochs=2).fit(X, Y).predict_proba(X)
    b = PhysicsMLP(epochs=2).fit(X, Y).predict_proba(X)
    assert a.shape == (len(X), 2) and np.allclose(a, b, atol=1e-6) and np.allclose(a.sum(axis=1), 1) and a.min() >= 0 and a.max() <= 1


def test_both_penalty_paths_fit_and_clone_keeps_the_settings():
    for lam in (0.0, 10.0):
        assert np.isfinite(PhysicsMLP(lam=lam, epochs=1).fit(X, Y).predict_proba(X)).all()
    m = PhysicsMLP(hidden=(64, 32), lam=10.0)
    assert clone(m).get_params() == m.get_params()


def test_it_plugs_into_the_event_grouped_evaluation_and_learns_something():
    split = loeo_splits(FRAME, drop_shared_locations=True)[:1]
    s = out_of_fold_scores(PhysicsMLP(epochs=1), FRAME, list(FEATURES), split)
    assert np.isfinite(s[split[0][2]]).all() and np.isnan(np.delete(s, split[0][2])).all()
    assert roc_auc_score(Y, PhysicsMLP(epochs=20).fit(X, Y).predict_proba(X)[:, 1]) > 0.6

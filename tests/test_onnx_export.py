"""ONNX export of the registered logistic (docs/PHASE_C_PROTOCOL.md, Section 15): exact parity with scikit-learn on real rows."""
import copy
import os

import numpy as np
import pytest

if os.environ.get("GITHUB_ACTIONS") == "true":
    import onnx
    import onnxruntime  # noqa: F401  (in CI a missing or broken install must fail, not skip)
else:
    try:
        import onnx
        import onnxruntime  # noqa: F401
    except Exception as e:  # a protobuf clash raises VersionError, which is not an ImportError
        pytest.skip("onnx or onnxruntime is not usable here: %s" % type(e).__name__, allow_module_level=True)

from src.models.baseline import build_logistic
from src.models.boosters import monotone_violations
from src.models.constrained import build_constrained_logistic
from src.models.cv import CONSTRAINTS, FEATURE_SETS, LABEL, load_layer_flags, load_training_frame, restrict_to_mappable
from src.models.export_onnx import IR_VERSION, OPSET, OUTPUT, OnnxLogistic, UnsupportedPipeline, to_onnx

FEATS = FEATURE_SETS["six"]
TOL = 1e-9


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this check never touches Earth Engine."""
    yield


@pytest.fixture(scope="module")
def mappable():
    df, _ = load_training_frame()
    return restrict_to_mappable(df, load_layer_flags(df))


@pytest.fixture(scope="module")
def fitted(mappable):
    X, y = mappable[FEATS], mappable[LABEL].to_numpy()
    return {"con": build_constrained_logistic(FEATS).fit(X, y), "free": build_logistic(FEATS).fit(X, y)}


def _check_parity(pipe, frame):
    blob = to_onnx(pipe, FEATS, {"model": "test"})
    a, b = pipe.predict_proba(frame[FEATS]), OnnxLogistic(blob, FEATS).predict_proba(frame[FEATS])
    assert a.shape == b.shape == (len(frame), 2) and np.isfinite(b).all()
    assert float(np.abs(a - b).max()) < TOL
    assert np.allclose(b.sum(axis=1), 1.0, atol=1e-12)
    assert ((a[:, 1] > 0.5) == (b[:, 1] > 0.5)).all()


def test_constrained_logistic_matches_on_all_real_rows(fitted, mappable):
    _check_parity(fitted["con"], mappable)


def test_unconstrained_scikit_learn_logistic_matches_on_all_real_rows(fitted, mappable):
    _check_parity(fitted["free"], mappable)


def test_missing_clay_and_an_unseen_land_cover_class_match(fitted, mappable):
    rows = mappable[FEATS].iloc[:200].copy()
    rows.loc[rows.index[:100], "clay_percent"] = np.nan
    rows.loc[rows.index[100:], "land_cover"] = 999
    for pipe in fitted.values():
        a = pipe.predict_proba(rows)
        b = OnnxLogistic(to_onnx(pipe, FEATS), FEATS).predict_proba(rows)
        assert np.isfinite(b).all() and float(np.abs(a - b).max()) < TOL


def test_the_onnx_model_has_no_monotonicity_violations(fitted, mappable):
    model = OnnxLogistic(to_onnx(fitted["con"], FEATS), FEATS)
    assert monotone_violations(model, mappable[FEATS], CONSTRAINTS) == {f: 0 for f in CONSTRAINTS}


def test_model_format_inputs_and_metadata(fitted):
    blob = to_onnx(fitted["con"], FEATS, {"model": "logistic:con", "frame": "mappable"})
    m = onnx.load_from_string(blob)
    onnx.checker.check_model(m)
    assert {o.domain: o.version for o in m.opset_import}[""] == OPSET and m.ir_version == IR_VERSION
    assert [i.name for i in m.graph.input] == FEATS and [o.name for o in m.graph.output] == [OUTPUT]
    kinds = {i.name: i.type.tensor_type.elem_type for i in m.graph.input}
    assert kinds["land_cover"] == onnx.TensorProto.INT64
    assert all(kinds[f] == onnx.TensorProto.DOUBLE for f in FEATS if f != "land_cover")
    assert {p.key: p.value for p in m.metadata_props} == {"model": "logistic:con", "frame": "mappable"}


def test_export_is_deterministic(fitted):
    assert to_onnx(fitted["con"], FEATS, {"a": "1"}) == to_onnx(fitted["con"], FEATS, {"a": "1"})


def test_the_exporter_refuses_structures_it_does_not_reproduce(fitted, mappable):
    feats5 = [f for f in FEATS if f != "clay_percent"]
    pipe5 = build_logistic(feats5).fit(mappable[feats5], mappable[LABEL].to_numpy())
    with pytest.raises(UnsupportedPipeline):
        to_onnx(pipe5, feats5)
    changed = copy.deepcopy(fitted["con"])
    changed.named_steps["prep"].transformers_[0][1].steps[0][1].strategy = "mean"
    with pytest.raises(UnsupportedPipeline):
        to_onnx(changed, FEATS)

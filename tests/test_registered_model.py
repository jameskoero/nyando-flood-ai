"""Registered Phase C model (docs/PHASE_C_PROTOCOL.md, Section 15): manifest entry, provenance, reproducibility and behaviour of the artifact."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

if os.environ.get("GITHUB_ACTIONS") == "true":
    import onnx
    import onnxruntime  # noqa: F401  (in CI a missing or broken install must fail, not skip)
else:
    try:
        import onnx
        import onnxruntime  # noqa: F401
    except Exception as e:
        pytest.skip("onnx or onnxruntime is not usable here: %s" % type(e).__name__, allow_module_level=True)

from src.models.boosters import monotone_violations
from src.models.constrained import build_constrained_logistic
from src.models.cv import CONSTRAINTS, FEATURE_SETS, FLAGS_PATH, LABEL, load_layer_flags, load_training_frame, restrict_to_mappable
from src.models.export_onnx import OnnxLogistic

ROOT = Path(__file__).resolve().parent.parent
FEATS = FEATURE_SETS["six"]
TOL = 1e-5
COEF = [-1.8624, 0.0, 0.6537, -1.0963, 0.7656, 0.0296, 1.1798, -2.5650, -1.2384, 0.9053, -0.6118, -0.1874, 2.0217, 0.4958]
INTERCEPT = -0.2047


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this check never touches Earth Engine."""
    yield


@pytest.fixture(scope="module")
def entry():
    models = json.loads((ROOT / "models" / "MANIFEST.json").read_text(encoding="utf-8"))
    phase_c = {k: v for k, v in models.items() if v.get("algorithm") == "logistic_constrained" and v.get("status") in ("active", "retired")}
    assert len(phase_c) == 1, "exactly one Phase C registered model is expected, found %s" % sorted(phase_c)
    return next(iter(phase_c.items()))


@pytest.fixture(scope="module")
def mappable():
    df, _ = load_training_frame()
    return restrict_to_mappable(df, load_layer_flags(df))


@pytest.fixture(scope="module")
def committed(entry):
    return OnnxLogistic((ROOT / entry[0]).read_bytes(), FEATS)


@pytest.fixture(scope="module")
def refit(mappable):
    return build_constrained_logistic(FEATS).fit(mappable[FEATS], mappable[LABEL].to_numpy())


def test_manifest_entry_matches_the_committed_artifact(entry):
    rel, e = entry
    p = ROOT / rel
    name = Path(rel).name
    assert p.exists() and rel.startswith("models/nyando_logcon_") and name.endswith(".onnx")
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    assert e["sha256"] == digest and name == "nyando_logcon_%s.onnx" % digest[:12]
    assert e["size_bytes"] == p.stat().st_size and e["algorithm"] == "logistic_constrained"
    data = json.loads((ROOT / "data" / "MANIFEST.json").read_text(encoding="utf-8"))
    train = [v["sha256"] for v in data.values() if isinstance(v, dict) and "label_source" in v]
    assert len(train) == 1 and e["trained_on"] == train[0]


def test_embedded_metadata_matches_the_data_and_the_protocol(entry):
    rel, e = entry
    meta = {p.key: p.value for p in onnx.load(str(ROOT / rel)).metadata_props}
    for k in ("model", "training_frame", "features", "training_data_sha256", "layer_flags_sha256", "protocol", "registration_run",
              "input_contract", "claim_limit"):
        assert k in meta, k
    assert meta["model"] == "logistic:con" and meta["training_frame"] == "mappable" and meta["features"] == ",".join(FEATS)
    assert meta["training_data_sha256"] == e["trained_on"]
    assert meta["layer_flags_sha256"] == hashlib.sha256(Path(FLAGS_PATH).read_bytes()).hexdigest()
    assert "Section 15" in meta["protocol"] and "not flood probabilities" in meta["claim_limit"]


def test_the_artifact_reproduces_from_the_committed_data(committed, refit, mappable):
    a, b = refit.predict_proba(mappable[FEATS]), committed.predict_proba(mappable[FEATS])
    assert float(np.abs(a - b).max()) < TOL


def test_the_refit_coefficients_match_the_protocol(refit):
    clf = refit.named_steps["clf"]
    assert np.abs(clf.coef_ - np.array(COEF)).max() < 5e-4 and abs(clf.intercept_ - INTERCEPT) < 5e-4


def test_the_artifact_has_no_monotonicity_violations_and_ignores_slope(committed, mappable):
    assert monotone_violations(committed, mappable[FEATS], CONSTRAINTS) == {f: 0 for f in CONSTRAINTS}
    flat = mappable[FEATS].copy()
    flat["slope"] = 0.0
    assert float(np.abs(committed.predict_proba(mappable[FEATS]) - committed.predict_proba(flat)).max()) < 1e-8


def test_unseen_land_cover_and_missing_clay_are_scored(committed, mappable):
    rows = mappable[FEATS].iloc[:100].copy()
    rows.loc[rows.index[:50], "clay_percent"] = np.nan
    rows.loc[rows.index[50:], "land_cover"] = 999
    p = committed.predict_proba(rows)
    assert p.shape == (100, 2) and np.isfinite(p).all() and np.allclose(p.sum(axis=1), 1.0)


def test_protocol_states_the_artifact_and_its_hash(entry):
    text = (ROOT / "docs" / "PHASE_C_PROTOCOL.md").read_text(encoding="utf-8")
    assert "## 15. ONNX export" in text and Path(entry[0]).name in text
    assert entry[1]["sha256"] in text

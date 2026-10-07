"""Registered hgb:con (docs/PROMOTION_PROTOCOL.md; docs/REGISTRATION.json; register row D43): the manifest entry is the committed artifact (export gate E4), the
artifact reproduces a refit through the production feed (E1, E2) with no monotonicity violation (E3), and the recorded outcome follows the gates."""
import ast
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import pytest

from src.models import promotion as pr
from src.models.boosters import build_hgb, monotone_violations
from src.models.cv import CONSTRAINTS, EVENT, FEATURE_SETS, FLAGS_PATH, LABEL, load_layer_flags, load_training_frame, per_group_auc, restrict_to_mappable
from src.models.export_hgb import CATEGORICAL, FEATURES, OUTPUT

ROOT = Path(__file__).resolve().parent.parent
MODELS = json.loads((ROOT / "models" / "MANIFEST.json").read_text(encoding="utf-8"))
ACTIVE = [(k, v) for k, v in MODELS.items() if v.get("status") == "active"]
REC = json.loads((ROOT / "docs" / "REGISTRATION.json").read_text(encoding="utf-8"))
PROMO = json.loads((ROOT / "docs" / "PROMOTION_RESULTS.json").read_text(encoding="utf-8"))
REL, ENTRY = ACTIVE[0]
BLOB = (ROOT / REL).read_bytes()
META = {p.key: p.value for p in onnx.load_from_string(BLOB).metadata_props}
SESS = ort.InferenceSession(BLOB, providers=["CPUExecutionProvider"])
DF, TSHA = load_training_frame(ROOT)
FRAME = restrict_to_mappable(DF, load_layer_flags(DF, FLAGS_PATH)).reset_index(drop=True)
BLOCK_A = pd.read_csv(ROOT / "data" / "confirmatory" / "nyando_block_a.csv")
SIX = FEATURE_SETS["six"]
REFIT = build_hgb(SIX, True, **pr.FIXED_PARAMS).fit(FRAME[SIX], FRAME[LABEL].to_numpy())


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this check never touches Earth Engine."""
    yield


def onnx_p(X):
    return SESS.run([OUTPUT], {f: X[[f]].to_numpy(np.int64 if f == CATEGORICAL else np.float64) for f in FEATURES})[0][:, 1]


def test_e4_the_manifest_entry_is_the_committed_artifact():
    sha = hashlib.sha256(BLOB).hexdigest()
    assert len(ACTIVE) == 1 and REL == "models/nyando_hgbcon_%s.onnx" % sha[:12] and ENTRY["sha256"] == sha and ENTRY["size_bytes"] == len(BLOB)
    assert ENTRY["algorithm"] == "hgb_constrained" and ENTRY["trained_on"] == TSHA and ENTRY["protocol"] == "docs/PROMOTION_PROTOCOL.md"
    assert ENTRY["registration_run"] == REC["registration_run"] == META["registration_run"]
    old = MODELS[REC["previous"]["file"]]
    assert old["status"] == "retired" and old["sha256"] == REC["previous"]["sha256"] and old["algorithm"] == "logistic_constrained"


def test_the_embedded_metadata_matches_the_data_and_the_contract():
    for k in ("model", "training_frame", "features", "training_data_sha256", "layer_flags_sha256", "protocol", "registration_run", "input_contract", "claim_limit", "land_cover_classes"):
        assert k in META, k
    assert META["model"] == "hgb:con" and META["training_frame"] == "mappable" and META["features"] == ",".join(SIX)
    assert META["training_data_sha256"] == ENTRY["trained_on"] and META["layer_flags_sha256"] == hashlib.sha256(Path(FLAGS_PATH).read_bytes()).hexdigest()
    assert "PROMOTION_PROTOCOL" in META["protocol"] and "not flood probabilities" in META["claim_limit"]
    served = re.search(r"^LAND_COVER_CLASSES = (\(.*?\))", (ROOT / "backend" / "registered.py").read_text(encoding="utf-8"), re.M)
    assert [int(c) for c in META["land_cover_classes"].split(",")] == list(ast.literal_eval(served.group(1)))


def test_e1_e2_the_artifact_reproduces_a_refit_through_the_production_feed():
    for X in (FRAME, BLOCK_A):
        ref, got = REFIT.predict_proba(X[SIX])[:, 1], onnx_p(X)
        assert float(np.abs(got - ref).max()) <= pr.EXPORT_TOLERANCE
        a, b = per_group_auc(X, ref, EVENT, 1), per_group_auc(X, got, EVENT, 1)
        assert max(abs(a[e] - b[e]) for e in a) <= pr.EXPORT_TOLERANCE


def test_e3_the_artifact_has_no_monotonicity_violation():
    class W:
        def predict_proba(self, X):
            q = onnx_p(X)
            return np.column_stack([1 - q, q])
    assert monotone_violations(W(), FRAME[SIX], CONSTRAINTS) == {f: 0 for f in CONSTRAINTS}


def test_the_registration_record_follows_the_gates():
    e = REC["export_gates"]
    e1, e2 = max(e["E1_max_abs_diff"].values()), max(e["E2_per_event_auc_diff"].values())
    ok = pr.export_gates(e1, e2, e["E3_violations"], REC["model"]["sha256"] == ENTRY["sha256"])
    assert ok and e["passed"] is True and REC["statistical_gates"] == PROMO["gates"] and pr.build_exporter(PROMO["gates"])
    assert REC["outcome"] == pr.decision(PROMO["gates"], ok) == pr.PROMOTED
    assert REC["protocol_sha256"] == hashlib.sha256((ROOT / "docs" / "PROMOTION_PROTOCOL.md").read_bytes()).hexdigest() == PROMO["protocol_sha256"]
    assert REC["promotion_results_sha256"] == hashlib.sha256((ROOT / "docs" / "PROMOTION_RESULTS.json").read_bytes()).hexdigest()
    assert REC["metrics"]["model_sha256"] == ENTRY["sha256"] and REC["metrics"]["source"] == "docs/REGISTRATION.json"
    assert re.fullmatch(r"[0-9a-f]{32}", REC["registration_run"])


def test_the_model_card_and_the_register_state_the_registration():
    card = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert Path(REL).name in card and ENTRY["sha256"] in card and REC["registration_run"] in card and pr.PROMOTED in card
    for k in ("pooled/mappable", "pooled/full", "S6/pooled_mappable", "buffered", "temporal"):
        s = PROMO["stats"][k]
        assert "%+.4f [%.4f, %.4f]" % (s["mean_diff"], s["ci_low"], s["ci_high"]) in card, k
    row = [l for l in (ROOT / "docs" / "ROADMAP_DEVIATIONS.md").read_text(encoding="utf-8").splitlines() if l.startswith("| D43 |")]
    assert len(row) == 1 and "tests/test_registered_hgb.py::" in row[0]

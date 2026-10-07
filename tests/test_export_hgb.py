"""Exporter for hgb:con (src/models/export_hgb.py; docs/PROMOTION_PROTOCOL.md export gates E1 to E3; register row D42): the exported graph reproduces
scikit-learn through the production feed on real rows, keeps the column order, the class codes and every monotonic direction, and a mis-ordered graph is caught."""
import ast
import re
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from src.models.boosters import build_hgb, monotone_violations
from src.models.cv import CONSTRAINTS, EVENT, FEATURE_SETS, FLAGS_PATH, LABEL, load_layer_flags, load_training_frame, per_group_auc, restrict_to_mappable
from src.models.export_hgb import CATEGORICAL, FEATURES, OUTPUT, categories, internal_order, to_onnx_hgb
from src.models.promotion import EXPORT_TOLERANCE, FIXED_PARAMS, export_gates

ROOT = Path(__file__).resolve().parent.parent
DF, _ = load_training_frame(ROOT)
FRAME = restrict_to_mappable(DF, load_layer_flags(DF, FLAGS_PATH)).reset_index(drop=True)
BLOCK_A = pd.read_csv(ROOT / "data" / "confirmatory" / "nyando_block_a.csv")
SIX = FEATURE_SETS["six"]
MODEL = build_hgb(SIX, True, **FIXED_PARAMS).fit(FRAME[SIX], FRAME[LABEL].to_numpy())
ONX, INFO = to_onnx_hgb(MODEL)


def session(m):
    return ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])


SESS = session(ONX)


def onnx_p(X, sess=SESS):
    feed = {f: X[[f]].to_numpy(np.int64 if f == CATEGORICAL else np.float64) for f in FEATURES}
    return sess.run([OUTPUT], feed)[0][:, 1]


def test_the_graph_keeps_the_production_contract():
    assert [(i.name, i.type, i.shape) for i in SESS.get_inputs()] == [(f, "tensor(int64)" if f == CATEGORICAL else "tensor(double)", [None, 1]) for f in FEATURES]
    assert [(o.name, o.type, o.shape) for o in SESS.get_outputs()] == [(OUTPUT, "tensor(double)", [None, 2])]
    served = re.search(r"^FEATURES = (\(.*?\))", (ROOT / "backend" / "registered.py").read_text(encoding="utf-8"), re.M)
    assert served and ast.literal_eval(served.group(1)) == FEATURES


def test_e1_e2_parity_on_real_rows_through_the_production_feed():
    for X in (FRAME, BLOCK_A):
        ref, got = MODEL.predict_proba(X[SIX])[:, 1], onnx_p(X)
        d, blank = np.abs(got - ref), X["clay_percent"].isna().to_numpy()
        assert d.max() <= EXPORT_TOLERANCE and blank.any() and d[blank].max() <= EXPORT_TOLERANCE
        e1, e2 = per_group_auc(X, ref, EVENT, 1), per_group_auc(X, got, EVENT, 1)
        assert set(e1) == set(e2) and max(abs(e1[e] - e2[e]) for e in e1) <= EXPORT_TOLERANCE


def test_every_land_cover_class_maps_to_its_code():
    cats = categories(MODEL)
    props = {p.key: p.value for p in ONX.metadata_props}
    assert props["land_cover_classes"] == ",".join(str(int(c)) for c in cats)
    for c in cats:
        X = FRAME[FRAME[CATEGORICAL] == c]
        assert len(X) > 0, c
        assert np.abs(onnx_p(X) - MODEL.predict_proba(X[SIX])[:, 1]).max() <= EXPORT_TOLERANCE, c


def test_the_internal_column_order_is_land_cover_first():
    assert internal_order(MODEL) == [CATEGORICAL] + [f for f in FEATURES if f != CATEGORICAL]
    assert INFO["trees"] == FIXED_PARAMS["max_iter"] and INFO["member_nodes"] > 0


def test_e3_the_artifact_keeps_every_monotonic_direction():
    class W:
        def predict_proba(self, X):
            p = onnx_p(X)
            return np.column_stack([1 - p, p])
    assert monotone_violations(W(), FRAME[SIX], CONSTRAINTS) == {f: 0 for f in CONSTRAINTS}


def test_the_export_gates_pass_and_a_mis_ordered_graph_is_caught():
    worst = max(float(np.abs(onnx_p(X) - MODEL.predict_proba(X[SIX])[:, 1]).max()) for X in (FRAME, BLOCK_A))
    assert export_gates(worst, 0.0, {f: 0 for f in CONSTRAINTS}, True)
    bad = onnx.ModelProto()
    bad.CopyFrom(ONX)
    node = [n for n in bad.graph.node if list(n.output) == ["X"]][0]
    del node.input[:]
    node.input.extend(["lc_double" if f == CATEGORICAL else f for f in FEATURES])
    d = np.abs(onnx_p(FRAME, session(bad)) - MODEL.predict_proba(FRAME[SIX])[:, 1])
    assert d.max() > EXPORT_TOLERANCE


def test_the_guards_fail_loudly():
    rev = list(reversed(SIX))
    with pytest.raises(ValueError):
        to_onnx_hgb(build_hgb(rev, True, **FIXED_PARAMS).fit(FRAME[rev], FRAME[LABEL].to_numpy()))
    with pytest.raises(ValueError):
        to_onnx_hgb(HistGradientBoostingClassifier(max_iter=5, random_state=42).fit(FRAME[SIX], FRAME[LABEL].to_numpy()))

"""scripts/register_hgbcon.py: fit hgb:con (the D20 candidate, fixed settings) on the mappable frame, export it with src/models/export_hgb.py, measure export gates
E1 to E3 on the artifact bytes and write the artifact under models/ named by its SHA-256 (docs/PROMOTION_PROTOCOL.md). Run in CI's versions (Python 3.11,
numpy 1.26.4, scikit-learn 1.6.1) so CI's refit reproduces the same trees. Usage: register_hgbcon.py <mlflow_run_id> <out.json>"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
CLAIM = "scores rank locations inside GFM-mappable areas and are not flood probabilities; the sample is case-control"
CONTRACT = "six [N, 1] inputs: elevation, slope, rainfall_3day, distance_river, clay_percent (double, NaN when missing), land_cover (int64); output probabilities, double [N, 2]"


def main(run_id, out):
    import onnxruntime as ort
    import sklearn
    from src.models.boosters import build_hgb, monotone_violations
    from src.models.cv import CONSTRAINTS, EVENT, FEATURE_SETS, FLAGS_PATH, LABEL, load_layer_flags, load_training_frame, per_group_auc, restrict_to_mappable
    from src.models.export_hgb import CATEGORICAL, FEATURES, OUTPUT, to_onnx_hgb
    from src.models.promotion import FIXED_PARAMS
    df, tsha = load_training_frame(REPO)
    frame = restrict_to_mappable(df, load_layer_flags(df, FLAGS_PATH)).reset_index(drop=True)
    blk = pd.read_csv(REPO / "data" / "confirmatory" / "nyando_block_a.csv")
    six = FEATURE_SETS["six"]
    model = build_hgb(six, True, **FIXED_PARAMS).fit(frame[six], frame[LABEL].to_numpy())
    meta = {"training_frame": "mappable", "features": ",".join(six), "training_data_sha256": tsha, "layer_flags_sha256": hashlib.sha256(Path(FLAGS_PATH).read_bytes()).hexdigest(),
            "protocol": "docs/PROMOTION_PROTOCOL.md (D20 promotion)", "registration_run": run_id, "input_contract": CONTRACT, "claim_limit": CLAIM, "fixed_params": FIXED_PARAMS}
    onx, info = to_onnx_hgb(model, meta)
    blob = onx.SerializeToString()
    sha = hashlib.sha256(blob).hexdigest()
    rel = "models/nyando_hgbcon_%s.onnx" % sha[:12]
    sess = ort.InferenceSession(blob, providers=["CPUExecutionProvider"])

    def p(X):
        return sess.run([OUTPUT], {f: X[[f]].to_numpy(np.int64 if f == CATEGORICAL else np.float64) for f in FEATURES})[0][:, 1]

    class W:
        def predict_proba(self, X):
            q = p(X)
            return np.column_stack([1 - q, q])

    e1, e2 = {}, {}
    for name, X in (("training", frame), ("block_a", blk)):
        ref, got = model.predict_proba(X[six])[:, 1], p(X)
        a, b = per_group_auc(X, ref, EVENT, 1), per_group_auc(X, got, EVENT, 1)
        e1[name], e2[name] = float(np.abs(got - ref).max()), float(max(abs(a[e] - b[e]) for e in a))
    viol = monotone_violations(W(), frame[six], CONSTRAINTS)
    (REPO / rel).write_bytes(blob)
    res = {"file": rel, "sha256": sha, "size_bytes": len(blob), "training_data_sha256": tsha, "layer_flags_sha256": meta["layer_flags_sha256"], "graph": info,
           "E1_max_abs_diff": e1, "E2_per_event_auc_diff": e2, "E3_violations": viol, "fixed_params": FIXED_PARAMS,
           "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "onnxruntime": ort.__version__}}
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

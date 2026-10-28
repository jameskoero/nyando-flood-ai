"""Phase E E1 (docs/PHASE_E_PROTOCOL.md Sections 3, 4 and 8; register row D49): out-of-fold reference scores, fixed cutpoints, one scoring pass over Block A with the registered artifact, the three gates and one MLflow run.
Prints the result as one RESULT_JSON line. Run in GitHub Actions (.github/workflows/e1.yml): python scripts/run_phase_e1.py --log-mlflow"""
import argparse
import base64
import hashlib
import json
import os
import platform
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
TRACK = "https://dagshub.com/jmskoero/nyando-flood-ai.mlflow"


def log_mlflow(tok, res):
    auth = "Basic " + base64.b64encode(("jmskoero:" + tok).encode()).decode()

    def call(path, body=None, query=None):
        url = TRACK + "/api/2.0/mlflow/" + path + (("?" + "&".join("%s=%s" % kv for kv in query.items())) if query else "")
        rq = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None, headers={"Authorization": auth, "Content-Type": "application/json", "User-Agent": "nyando-e1"})
        for i in range(4):
            try:
                with urllib.request.urlopen(rq, timeout=120) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                sys.exit("STOPPED: MLflow %s: HTTP %d" % (path, e.code))
            except Exception:
                time.sleep(5 * (i + 1))
        sys.exit("STOPPED: MLflow %s: no answer" % path)
    m = {"cut_low": res["cutpoints"]["low"], "cut_high": res["cutpoints"]["high"], "reference_event_auc": res["reference"]["mean_event_auc"],
         "block_a_rows_mappable": res["block_a"]["rows_mappable"], "block_a_rows_kept": res["block_a"]["rows_kept"], "block_a_rows_excluded": res["block_a"]["rows_excluded"]}
    for lv, r in res["rates"].items():
        m["rows_" + lv.lower()] = r["rows"]
        if r["share"] is not None:
            m["share_" + lv.lower()] = r["share"]
    b = res["bootstrap"]
    if b["diff"] is not None:
        m.update({"high_minus_low": b["diff"], "high_minus_low_ci_low": b["ci_low"], "high_minus_low_ci_high": b["ci_high"]})
    m.update({"gate_" + k.lower(): float(v) for k, v in res["gates"].items()})
    if "experiment" not in call("experiments/get", query={"experiment_id": "0"}):
        sys.exit("STOPPED: DagsHub did not return experiment 0")
    now = int(time.time() * 1000)
    rid = call("runs/create", {"experiment_id": "0", "start_time": now, "tags": [{"key": "mlflow.runName", "value": "phase_e_e1"}]})["run"]["info"]["run_id"]
    params = {"outcome": res["outcome"], "protocol_sha256": res["protocol_sha256"], "training_data_sha256": res["training_data_sha256"], "model_sha256": res["model_sha256"], "commit": res["commit"]}
    call("runs/log-batch", {"run_id": rid, "metrics": [{"key": k, "value": float(v), "timestamp": now, "step": 0} for k, v in m.items()], "params": [{"key": k, "value": str(v)[:250]} for k, v in params.items()]})
    call("runs/update", {"run_id": rid, "status": "FINISHED", "end_time": int(time.time() * 1000)})
    got = call("runs/get", query={"run_id": rid})["run"]
    if got["info"]["status"] != "FINISHED" or len(got["data"].get("metrics", [])) != len(m):
        sys.exit("STOPPED: the MLflow read-back does not match: %d metrics logged, %d read" % (len(m), len(got["data"].get("metrics", []))))
    return {"run_id": rid, "tracking_uri": TRACK, "metrics_logged": len(m)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-mlflow", action="store_true")
    args = ap.parse_args()
    tok = os.environ.get("DAGSHUB_TOKEN", "").strip()
    if args.log_mlflow and not tok:
        sys.exit("STOPPED: DAGSHUB_TOKEN is not set (store it with: gh secret set DAGSHUB_TOKEN)")
    import numpy as np
    import pandas as pd
    import onnxruntime
    import sklearn
    from backend.registered import FEATURES, load_registered
    from src.advisory import e1
    from src.advisory import rules
    from src.models import phase_d as D
    from src.models.boosters import make_estimator
    from src.models.cv import EVENT, FEATURE_SETS, FLAGS_PATH, LABEL, load_layer_flags, load_training_frame, location_keys, loeo_splits, out_of_fold_scores, restrict_to_mappable
    from src.models.phase_d_stats import aucs, mean

    df, tsha = load_training_frame(REPO)
    M = restrict_to_mappable(df, load_layer_flags(df, FLAGS_PATH)).reset_index(drop=True)
    six = list(FEATURE_SETS["six"])
    params = json.loads((REPO / "docs" / "REGISTRATION.json").read_text(encoding="utf-8"))["fixed_params"]
    ref_scores = out_of_fold_scores(make_estimator("hgb", six, True)(params), M, six, loeo_splits(M, drop_shared_locations=True))
    ref = ref_scores[~np.isnan(ref_scores)]
    repro = mean(aucs(M, ref_scores))
    want, tol = D.REPRODUCE["hgb:con"]
    cuts = rules.cutpoints(ref)
    ref_levels = rules.levels_of(ref, cuts)

    blk = pd.read_csv(REPO / "data" / "confirmatory" / "nyando_block_a.csv")
    flags = load_layer_flags(blk, REPO / "data" / "confirmatory" / "nyando_block_a_layer_flags.csv")
    B = restrict_to_mappable(blk, flags)
    bkeys = location_keys(B)
    keep = np.array(e1.keep_mask(list(bkeys), list(location_keys(M))))
    K = B[keep].reset_index(drop=True)
    reg = load_registered(str(REPO))
    if reg.session is None:
        sys.exit("STOPPED: the registered model did not load: %s" % reg.reason)
    scores = []
    for r in B[list(FEATURES)].to_dict("records"):
        v = {k: (None if pd.isna(r[k]) else r[k]) for k in FEATURES}
        v["land_cover"] = int(v["land_cover"])
        scores.append(reg.score(v))
    sB = np.array(scores)
    sK = sB[keep]
    levels = rules.levels_of(sK, cuts)
    summary = e1.summarize(levels, K[LABEL].to_numpy(), K[EVENT].astype(str).to_numpy())
    res = {"schema": 1, "protocol": "docs/PHASE_E_PROTOCOL.md", "protocol_sha256": hashlib.sha256((REPO / "docs" / "PHASE_E_PROTOCOL.md").read_bytes()).hexdigest(),
           "commit": os.environ.get("GITHUB_SHA", ""), "training_data_sha256": tsha, "model_sha256": reg.sha256, "registered_params": params,
           "reference": {"rows": int(len(ref)), "rows_never_tested": int(np.isnan(ref_scores).sum()), "mean_event_auc": round(float(repro), 6), "stored": want, "tolerance": tol,
                         "reproduces": bool(abs(repro - want) <= tol), "rows_per_level": {lv: ref_levels.count(lv) for lv in rules.LEVELS}},
           "cutpoints": {"low": cuts[0], "high": cuts[1]},
           "block_a": {"rows_total": int(len(blk)), "rows_mappable": int(len(B)), "rows_excluded": int((~keep).sum()), "rows_kept": int(keep.sum()),
                       "locations_mappable": int(bkeys.nunique()), "locations_excluded": int(bkeys[~keep].nunique()), "locations_kept": int(bkeys[keep].nunique()),
                       "artifact_mean_event_auc_mappable": round(float(mean(aucs(B, sB))), 6), "artifact_mean_event_auc_kept": round(float(mean(aucs(K, sK))), 6)},
           "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "onnxruntime": onnxruntime.__version__}}
    res.update(summary)
    print("== outcome:", res["outcome"], "| gates:", res["gates"])
    print("== cutpoints:", res["cutpoints"], "| reference event AUC %.6f (stored %s, reproduces %s)" % (res["reference"]["mean_event_auc"], want, res["reference"]["reproduces"]))
    print("== Block A rows: mappable %d, excluded at training locations %d, kept %d" % (res["block_a"]["rows_mappable"], res["block_a"]["rows_excluded"], res["block_a"]["rows_kept"]))
    for lv in rules.LEVELS:
        print("== %-6s rows %5d flood %5d share %s" % (lv, res["rates"][lv]["rows"], res["rates"][lv]["flood"], res["rates"][lv]["share"]))
    print("== HIGH minus LOW:", res["bootstrap"], flush=True)
    if args.log_mlflow:
        res["mlflow"] = log_mlflow(tok, res)
        print("== MLflow run", res["mlflow"]["run_id"], "logged and read back", flush=True)
    print("RESULT_JSON " + json.dumps(res, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()

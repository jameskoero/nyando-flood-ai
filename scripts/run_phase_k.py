#!/usr/bin/env python3
"""Phase K, step K1 (docs/PHASE_K_PROTOCOL.md): the ward-level fairness slices. It recomputes the E1 reference and the E1 cutpoints, checks that they are the ones in docs/PHASE_E_RESULTS.json, scores the two frames, applies the estimability rule, computes M1 to M4 and the descriptive flags, prints one RESULT_JSON line and logs one MLflow run (read back). It runs only from .github/workflows/k1.yml, by hand."""
import argparse
import base64
import hashlib
import importlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
TRACK = "https://dagshub.com/jmskoero/nyando-flood-ai.mlflow"


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def metrics_of(res):
    m = {}
    for frame, rep in res["frames"].items():
        for ward, e in rep["wards"].items():
            base = "%s_%s_" % (frame, slug(ward))
            for k in ("flood_rows", "control_rows", "events_both"):
                m[base + k] = e[k]
            m[base + "estimable"] = int(e["estimable"])
            if not e["estimable"]:
                continue
            for lv, r in e["rates"].items():
                if r["share"] is not None:
                    m[base + "share_" + lv.lower()] = r["share"]
            for k, v in (("high_minus_low", e["bands"]["diff"]), ("hml_ci_low", e["bands"]["ci_low"]), ("hml_ci_high", e["bands"]["ci_high"]),
                         ("auc_mean", e["auc"]["mean"]), ("auc_ci_low", e["auc"]["ci_low"]), ("auc_ci_high", e["auc"]["ci_high"]),
                         ("gap", e["gap"]["gap"]), ("gap_ci_low", e["gap"]["ci_low"]), ("gap_ci_high", e["gap"]["ci_high"])):
                if v is not None:
                    m[base + k] = v
    return m


def log_mlflow(tok, res):
    auth = "Basic " + base64.b64encode(("jmskoero:" + tok).encode()).decode()

    def call(path, body=None, query=None):
        url = TRACK + "/api/2.0/mlflow/" + path + (("?" + "&".join("%s=%s" % kv for kv in query.items())) if query else "")
        rq = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                    headers={"Authorization": auth, "Content-Type": "application/json", "User-Agent": "nyando-flood-ai/2.0"})
        for i in range(4):
            try:
                with urllib.request.urlopen(rq, timeout=120) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                sys.exit("STOPPED: MLflow %s: HTTP %d" % (path, e.code))
            except Exception:
                time.sleep(5 * (i + 1))
        sys.exit("STOPPED: MLflow %s: no answer" % path)

    m = metrics_of(res)
    if "experiment" not in call("experiments/get", query={"experiment_id": "0"}):
        sys.exit("STOPPED: DagsHub did not return experiment 0")
    now = int(time.time() * 1000)
    rid = call("runs/create", {"experiment_id": "0", "start_time": now, "tags": [{"key": "mlflow.runName", "value": "phase_k_k1"}]})["run"]["info"]["run_id"]
    params = [{"key": "outcome", "value": res["outcome"]}, {"key": "protocol_sha256", "value": res["protocol_sha256"]},
              {"key": "training_data_sha256", "value": res["training_data_sha256"]}, {"key": "model_sha256", "value": str(res["model_sha256"])}]
    call("runs/log-batch", {"run_id": rid, "params": params, "metrics": [{"key": k, "value": float(v), "timestamp": now, "step": 0} for k, v in m.items()]})
    call("runs/update", {"run_id": rid, "status": "FINISHED", "end_time": int(time.time() * 1000)})
    got = call("runs/get", query={"run_id": rid})["run"]
    if got["info"]["status"] != "FINISHED" or len(got["data"].get("metrics", [])) != len(m):
        sys.exit("STOPPED: the MLflow read-back does not match: %d metrics logged, %d read" % (len(m), len(got["data"].get("metrics", []))))
    return {"run_id": rid, "tracking_uri": TRACK, "metrics_logged": len(m)}


def pick(name, *mods):
    for mod in mods:
        if hasattr(mod, name):
            return getattr(mod, name)
    sys.exit("STOPPED: %s was not found in %s" % (name, ", ".join(m.__name__ for m in mods)))


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
    from src.advisory import e1, rules
    from src.models import cv, phase_d as D, phase_d_stats as DS, phase_k as K, phase_k_stats as S
    from src.models.boosters import make_estimator

    EVENT, LABEL, FEATURE_SETS, FLAGS_PATH = cv.EVENT, cv.LABEL, cv.FEATURE_SETS, cv.FLAGS_PATH
    restrict_to_mappable = pick("restrict_to_mappable", cv, DS, D)
    out_of_fold_scores = pick("out_of_fold_scores", cv, DS, D)
    aucs, mean = DS.aucs, DS.mean

    stored = json.loads((REPO / "docs" / "PHASE_E_RESULTS.json").read_text(encoding="utf-8"))
    df, tsha = cv.load_training_frame(REPO)
    if tsha != stored["training_data_sha256"]:
        sys.exit("STOPPED: the training file is not the one E1 used")
    M = restrict_to_mappable(df, cv.load_layer_flags(df, FLAGS_PATH)).reset_index(drop=True)
    six = list(FEATURE_SETS["six"])
    params = json.loads((REPO / "docs" / "REGISTRATION.json").read_text(encoding="utf-8"))["fixed_params"]
    ref_scores = out_of_fold_scores(make_estimator("hgb", six, True)(params), M, six, cv.loeo_splits(M, drop_shared_locations=True))
    finite = ~np.isnan(ref_scores)
    ref = ref_scores[finite]
    repro = mean(aucs(M, ref_scores))
    want, tol = D.REPRODUCE["hgb:con"]
    cuts = rules.cutpoints(ref)
    low, high = (cuts["low"], cuts["high"]) if isinstance(cuts, dict) else (cuts[0], cuts[1])
    if abs(low - stored["cutpoints"]["low"]) > 1e-9 or abs(high - stored["cutpoints"]["high"]) > 1e-9:
        sys.exit("STOPPED: the recomputed cutpoints (%.12f, %.12f) are not E1's (%.12f, %.12f)" % (low, high, stored["cutpoints"]["low"], stored["cutpoints"]["high"]))
    ref_levels = rules.levels_of(ref, cuts)

    blk = pd.read_csv(REPO / "data" / "confirmatory" / "nyando_block_a.csv")
    B = restrict_to_mappable(blk, cv.load_layer_flags(blk, REPO / "data" / "confirmatory" / "nyando_block_a_layer_flags.csv"))
    keep = np.array(e1.keep_mask(list(cv.location_keys(B)), list(cv.location_keys(M))))
    K_A = B[keep].reset_index(drop=True)
    if len(K_A) != stored["block_a"]["rows_kept"]:
        sys.exit("STOPPED: Block A keeps %d rows, E1 kept %d" % (len(K_A), stored["block_a"]["rows_kept"]))
    reg = load_registered(str(REPO))
    if reg.session is None:
        sys.exit("STOPPED: the registered model did not load: %s" % reg.reason)
    scores = []
    for r in B[list(FEATURES)].to_dict("records"):
        v = {k: (None if pd.isna(r[k]) else r[k]) for k in FEATURES}
        v["land_cover"] = int(v["land_cover"])
        scores.append(reg.score(v))
    sK = np.array(scores)[keep]
    levels_K = rules.levels_of(sK, cuts)

    frames = {}
    for name, frame, sc, lv in (("primary", K_A, sK, levels_K), ("secondary", M[finite].reset_index(drop=True), ref, ref_levels)):
        if "ward" not in frame.columns:
            sys.exit("STOPPED: the %s frame has no ward column" % name)
        unknown = sorted(set(frame["ward"].astype(str)) - set(K.WARDS))
        if unknown:
            sys.exit("STOPPED: the %s frame has ward names the protocol does not list: %s" % (name, unknown))
        wards = frame["ward"].astype(str).to_numpy()
        frames[name] = S.frame_report(wards, frame[LABEL].to_numpy(), frame[EVENT].astype(str).to_numpy(), np.asarray(lv),
                                      lambda m, frame=frame, sc=sc: aucs(frame[m].reset_index(drop=True), sc[m]))
    res = {"schema": 1, "step": "K1", "outcome": K.OUTCOME, "protocol": "docs/PHASE_K_PROTOCOL.md",
           "protocol_sha256": hashlib.sha256((REPO / "docs" / "PHASE_K_PROTOCOL.md").read_bytes()).hexdigest(),
           "commit": os.environ.get("GITHUB_SHA"), "training_data_sha256": tsha, "model_sha256": getattr(reg, "sha256", None),
           "e1": {"cutpoints": {"low": float(low), "high": float(high)}, "cutpoints_match": True, "rows_kept_match": True,
                  "reference_mean_event_auc": float(repro), "reference_reproduces": bool(abs(repro - want) <= tol)},
           "frames": frames,
           "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "onnxruntime": onnxruntime.__version__, "sklearn": sklearn.__version__}}
    for name, rep in frames.items():
        print("== frame %s (%d rows)" % (name, rep["rows"]))
        for ward, e in rep["wards"].items():
            print("%-18s flood %4d control %4d events %2d | %s | %s" % (ward, e["flood_rows"], e["control_rows"], e["events_both"],
                  "estimable" if e["estimable"] else "NOT estimable (%s)" % e["reason"], ",".join(e["flags"]) or "no flags"))
    print("RESULT_JSON " + json.dumps(res, sort_keys=True))
    if args.log_mlflow:
        print("MLFLOW_RUN " + json.dumps(log_mlflow(tok, res)))


if __name__ == "__main__":
    main()

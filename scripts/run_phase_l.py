#!/usr/bin/env python3
"""Phase L, step L1 (docs/PHASE_L_PROTOCOL.md): the leakage audit of land_cover (D14) and clay_percent (D55). It trains the registered arm and the two audit arms with the registered fixed parameters on the mappable training rows, checks that the registered arm reproduces the registered artifact on Block A, recomputes the E1 cutpoints, computes the contrasts of the protocol on both frames, prints one RESULT_JSON line and logs one MLflow run (read back). It runs only from .github/workflows/l1.yml, by hand."""
import argparse
import base64
import hashlib
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
PARITY_TOLERANCE = 1e-9


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def flatten(prefix, obj, out):
    if isinstance(obj, bool):
        out[prefix] = int(obj)
    elif isinstance(obj, (int, float)):
        out[prefix] = float(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            flatten(prefix + "_" + slug(k) if prefix else slug(k), v, out)


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

    m = {}
    flatten("", res["frames"], m)
    if "experiment" not in call("experiments/get", query={"experiment_id": "0"}):
        sys.exit("STOPPED: DagsHub did not return experiment 0")
    now = int(time.time() * 1000)
    rid = call("runs/create", {"experiment_id": "0", "start_time": now, "tags": [{"key": "mlflow.runName", "value": "phase_l_l1"}]})["run"]["info"]["run_id"]
    params = [{"key": "outcome", "value": res["outcome"]}, {"key": "protocol_sha256", "value": res["protocol_sha256"]}, {"key": "training_data_sha256", "value": res["training_data_sha256"]},
              {"key": "model_sha256", "value": str(res["model_sha256"])}, {"key": "d14_primary", "value": res["frames"]["primary"]["d14_outcome"]}, {"key": "d55_primary", "value": res["frames"]["primary"]["d55_outcome"]}]
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
    from src.models import cv, phase_d as D, phase_d_stats as DS, phase_k as K, phase_l as L, phase_l_stats as LS
    from src.models.boosters import build_hgb, make_estimator

    LABEL, EVENT, FEATURE_SETS, FLAGS_PATH = cv.LABEL, cv.EVENT, cv.FEATURE_SETS, cv.FLAGS_PATH
    restrict_to_mappable = pick("restrict_to_mappable", cv, DS, D)
    out_of_fold_scores = pick("out_of_fold_scores", cv, DS, D)

    stored = json.loads((REPO / "docs" / "PHASE_E_RESULTS.json").read_text(encoding="utf-8"))
    registration = json.loads((REPO / "docs" / "REGISTRATION.json").read_text(encoding="utf-8"))
    params = registration["fixed_params"]
    model_file = REPO / registration["model"]["file"]
    if not model_file.is_file() or hashlib.sha256(model_file.read_bytes()).hexdigest() != registration["model"]["sha256"]:
        sys.exit("STOPPED: the registered model file is not the artifact named in docs/REGISTRATION.json")
    df, tsha = cv.load_training_frame(REPO)
    if tsha != stored["training_data_sha256"]:
        sys.exit("STOPPED: the training file is not the one E1 used")
    M = restrict_to_mappable(df, cv.load_layer_flags(df, FLAGS_PATH)).reset_index(drop=True)
    six = list(FEATURE_SETS["six"])
    if set(six) != {"elevation", "slope", "rainfall_3day", "distance_river", "clay_percent", "land_cover"}:
        sys.exit("STOPPED: the six features are not the ones the protocol names: %s" % six)
    arms = {"six": six, "no_land_cover": [f for f in six if f != "land_cover"], "no_clay": [f for f in six if f != "clay_percent"]}
    if "six_no_land_cover" in FEATURE_SETS and list(FEATURE_SETS["six_no_land_cover"]) != arms["no_land_cover"]:
        sys.exit("STOPPED: the repository's six_no_land_cover differs from the arm derived from the six features")

    y = M[LABEL].to_numpy()
    models = {a: build_hgb(fs, True, **params).fit(M[fs], y) for a, fs in arms.items()}
    splits = cv.loeo_splits(M, drop_shared_locations=True)
    oof = {a: out_of_fold_scores(make_estimator("hgb", fs, True)(params), M, fs, splits) for a, fs in arms.items()}
    repro = DS.mean(DS.aucs(M, oof["six"]))
    recorded = json.loads((REPO / "docs" / "PHASE_K_RESULTS.json").read_text(encoding="utf-8"))["e1"]["reference_mean_event_auc"]
    if abs(repro - recorded) > 1e-9:
        sys.exit("STOPPED: the E1 reference mean AUC %.10f is not the %.10f recorded by K1" % (repro, recorded))
    finite = ~np.isnan(oof["six"])
    if any(not np.array_equal(finite, ~np.isnan(oof[a])) for a in arms):
        sys.exit("STOPPED: the out-of-fold arms do not cover the same rows")
    cuts = rules.cutpoints(oof["six"][finite])
    low, high = (cuts["low"], cuts["high"]) if isinstance(cuts, dict) else (cuts[0], cuts[1])
    if abs(low - stored["cutpoints"]["low"]) > 1e-9 or abs(high - stored["cutpoints"]["high"]) > 1e-9:
        sys.exit("STOPPED: the recomputed cutpoints (%.12f, %.12f) are not E1's (%.12f, %.12f)" % (low, high, stored["cutpoints"]["low"], stored["cutpoints"]["high"]))

    blk = pd.read_csv(REPO / "data" / "confirmatory" / "nyando_block_a.csv")
    B = restrict_to_mappable(blk, cv.load_layer_flags(blk, REPO / "data" / "confirmatory" / "nyando_block_a_layer_flags.csv"))
    keep = np.array(e1.keep_mask(list(cv.location_keys(B)), list(cv.location_keys(M))))
    K_A = B[keep].reset_index(drop=True)
    if len(K_A) != stored["block_a"]["rows_kept"]:
        sys.exit("STOPPED: Block A keeps %d rows, E1 kept %d" % (len(K_A), stored["block_a"]["rows_kept"]))
    reg = load_registered(str(REPO))
    if reg.session is None:
        sys.exit("STOPPED: the registered model did not load: %s" % reg.reason)
    scored = []
    for r in B[list(FEATURES)].to_dict("records"):
        v = {k: (None if pd.isna(r[k]) else r[k]) for k in FEATURES}
        v["land_cover"] = int(v["land_cover"])
        scored.append(reg.score(v))
    s_reg = np.array(scored)[keep]
    parity = float(np.max(np.abs(models["six"].predict_proba(K_A[six])[:, 1] - s_reg)))
    if parity > PARITY_TOLERANCE:
        sys.exit("STOPPED: the registered arm trained here differs from the registered artifact on Block A by %.3e" % parity)
    primary_scores = {"six": s_reg, "no_land_cover": models["no_land_cover"].predict_proba(K_A[arms["no_land_cover"]])[:, 1], "no_clay": models["no_clay"].predict_proba(K_A[arms["no_clay"]])[:, 1]}
    Ms = M[finite].reset_index(drop=True)
    secondary_scores = {a: oof[a][finite] for a in arms}

    def t_contrast(frame, sa, sb):
        """T(six) minus T(audit): the per-event mean AUC difference with the paired event bootstrap of src/models/cv.py."""
        a = {e: v / 1e6 for e, v in DS.aucs(frame, sb).items()}
        b = {e: v / 1e6 for e, v in DS.aucs(frame, sa).items()}
        r = cv.paired_event_bootstrap(a, b)
        return {k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in r.items()}

    def report(frame, sc):
        classes = frame["land_cover"].astype(int).to_numpy()
        flooded = frame[LABEL].to_numpy().astype(int)
        events = frame[EVENT].astype(str).to_numpy()
        strata = L.class_strata(list(classes), list(flooded), list(events))
        use = [c for c, v in strata.items() if v["estimable"]]
        out = {"rows": int(len(frame)), "classes": {str(c): v for c, v in strata.items()}, "classes_used": [int(c) for c in use]}
        dw = LS.delta_w(sc["six"], sc["no_land_cover"], flooded, classes, events, use) if use else None
        out["delta_w_land_cover"] = dw
        out["delta_t_land_cover"] = t_contrast(frame, sc["six"], sc["no_land_cover"])
        out["d14_outcome"] = L.d14_outcome(dw["ci_low"], dw["ci_high"]) if dw else "not_estimable"
        present = frame["clay_percent"].notna().to_numpy()
        fl, ct, both = K.ward_counts(flooded[present], events[present])
        ok, why = K.estimability(fl, ct, both)
        out["clay"] = {"rows_blank": int((~present).sum()), "rows_observed": int(present.sum()), "observed_flood_rows": int(fl), "observed_control_rows": int(ct), "observed_events_both": int(both), "observed_estimable": bool(ok), "reason": why}
        if ok:
            sub = frame[present].reset_index(drop=True)
            out["delta_t_clay_all"] = t_contrast(frame, sc["six"], sc["no_clay"])
            out["delta_t_clay_present"] = t_contrast(sub, sc["six"][present], sc["no_clay"][present])
            out["d55_outcome"] = L.d55_outcome((out["delta_t_clay_all"]["ci_low"], out["delta_t_clay_all"]["ci_high"]), (out["delta_t_clay_present"]["ci_low"], out["delta_t_clay_present"]["ci_high"]))
        else:
            out["d55_outcome"] = "not_estimable"
        return out

    frames = {"primary": report(K_A, primary_scores), "secondary": report(Ms, secondary_scores)}
    res = {"schema": 1, "step": "L1", "outcome": L.OUTCOME, "protocol": "docs/PHASE_L_PROTOCOL.md",
           "protocol_sha256": hashlib.sha256((REPO / "docs" / "PHASE_L_PROTOCOL.md").read_bytes()).hexdigest(), "commit": os.environ.get("GITHUB_SHA"),
           "training_data_sha256": tsha, "model_sha256": registration["model"]["sha256"],
           "checks": {"reference_mean_auc": float(repro), "reference_recorded_by_k1": float(recorded), "cutpoints_match": True, "rows_kept_match": True, "registered_parity_max_abs_diff": parity, "parity_tolerance": PARITY_TOLERANCE},
           "arms": arms, "fixed_params": params, "frames": frames,
           "claims": {"d14": L.CLAIMS[frames["primary"]["d14_outcome"]], "d55": L.CLAIMS[frames["primary"]["d55_outcome"]], "limits": L.LIMITS},
           "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "onnxruntime": onnxruntime.__version__, "sklearn": sklearn.__version__}}
    for name, rep in frames.items():
        dw, dt = rep["delta_w_land_cover"], rep["delta_t_land_cover"]
        print("== frame %s (%d rows): D14 %s | D55 %s" % (name, rep["rows"], rep["d14_outcome"], rep["d55_outcome"]))
        print("   classes used %s of %d; dW %s; dT land_cover %+.4f [%+.4f, %+.4f]" % (rep["classes_used"], len(rep["classes"]), ("%+.4f [%+.4f, %+.4f]" % (dw["delta"], dw["ci_low"], dw["ci_high"])) if dw else "n/a", dt["mean_diff"], dt["ci_low"], dt["ci_high"]))
        c = rep["clay"]
        print("   clay blank %d, observed %d (%s); dT all %s; dT present %s" % (c["rows_blank"], c["rows_observed"], "estimable" if c["observed_estimable"] else c["reason"],
              ("%+.4f [%+.4f, %+.4f]" % (rep["delta_t_clay_all"]["mean_diff"], rep["delta_t_clay_all"]["ci_low"], rep["delta_t_clay_all"]["ci_high"])) if "delta_t_clay_all" in rep else "n/a",
              ("%+.4f [%+.4f, %+.4f]" % (rep["delta_t_clay_present"]["mean_diff"], rep["delta_t_clay_present"]["ci_low"], rep["delta_t_clay_present"]["ci_high"])) if "delta_t_clay_present" in rep else "n/a"))
    print("PARITY max abs diff vs the registered artifact on Block A: %.3e" % parity)
    print("RESULT_JSON " + json.dumps(res, sort_keys=True))
    if args.log_mlflow:
        print("MLFLOW_RUN " + json.dumps(log_mlflow(tok, res)))


if __name__ == "__main__":
    main()

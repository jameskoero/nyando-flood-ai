"""Log the Phase C booster evaluation (protocol Section 13) to MLflow, one arm after another.

In Colab, from a clean checkout of main: set sys.argv = ["run_boosters.py", FRAME, ARM, ...] and run this file.
FRAME is mappable or full; each ARM is hgb:con, hgb:free, xgb:con or xgb:free. The token is read through a hidden prompt."""
import getpass
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.baseline import build_logistic
from src.models.boosters import (HGB_FIXED, HGB_SPACE, INNER_FOLDS, XGB_FIXED, XGB_SPACE, describe, describe_fixed,
                                 make_estimator, monotone_violations, nested_scores, space_for)
from src.models.cv import (CONSTRAINTS, EVENT, FEATURE_SETS, LABEL, SEED, load_layer_flags, load_training_frame,
                           loeo_splits, out_of_fold_scores, paired_event_bootstrap, per_group_auc, pooled_auc,
                           restrict_to_mappable)
from src.tracking import tracked_run

ARMS = ("hgb:con", "hgb:free", "xgb:con", "xgb:free")


def compare(prefix, ref, pe, m):
    """Paired per-event bootstrap of pe minus ref (97.5% interval); the values are added to m."""
    bs = paired_event_bootstrap(ref, pe, alpha=0.025)
    for k in ("mean_diff", "ci_low", "ci_high", "wins", "losses"):
        m[prefix + "/" + k] = float(bs[k])
    return bs


def main():
    frame, arms = sys.argv[1], sys.argv[2:]
    assert frame in ("mappable", "full") and arms and all(a in ARMS for a in arms), "usage: FRAME ARM... with ARM in " + str(ARMS)
    df, _ = load_training_frame()
    if frame == "mappable":
        df = restrict_to_mappable(df, load_layer_flags(df))
    feats = FEATURE_SETS["six"]
    base = per_group_auc(df, out_of_fold_scores(build_logistic(feats), df, feats, loeo_splits(df, drop_shared_locations=True)), EVENT, 1)
    base_mean = float(np.mean(list(base.values())))
    elev = per_group_auc(df, -df["elevation"].to_numpy(float), EVENT, 1)
    print("frame:", frame, "| rows:", len(df), "| scorable events:", len(base), "| logistic per-event mean: %.4f" % base_mean, flush=True)
    os.environ.setdefault("MLFLOW_TRACKING_USERNAME", "jmskoero")
    if not os.environ.get("MLFLOW_TRACKING_PASSWORD"):
        os.environ["MLFLOW_TRACKING_PASSWORD"] = getpass.getpass("DagsHub token (hidden): ")
    params = {"frame": frame, "arms": ",".join(arms), "hgb_space": describe(HGB_SPACE), "xgb_space": describe(XGB_SPACE),
              "hgb_fixed": describe_fixed(HGB_FIXED), "xgb_fixed": describe_fixed(XGB_FIXED),
              "inner_folds": str(INNER_FOLDS), "seed": str(SEED), "protocol": "docs/PHASE_C_PROTOCOL.md Section 13"}
    tags = {"phase": "C", "arm": "boosters", "model_status": "candidate-not-a-product"}
    pes = {}
    with tracked_run("phase-c-boosters-" + frame, "HistGradientBoosting+XGBoost", params, tags=tags) as (mlflow, run):
        for arm in arms:
            kind, mode = arm.split(":")
            t0 = time.time()
            make = make_estimator(kind, feats, mode == "con")
            scores, chosen = nested_scores(make, space_for(kind), df, feats, n_jobs=2)
            pe = per_group_auc(df, scores, EVENT, 1)
            key = frame + "/" + arm.replace(":", "_")
            m = {key + "/pooled_auc": pooled_auc(df[LABEL], scores), key + "/per_event_mean": float(np.mean(list(pe.values())))}
            for e, a in pe.items():
                m[key + "/event/" + str(e)] = a
            vl = compare(key + "/vs_logistic", base, pe, m)
            ve = compare(key + "/vs_elevation", elev, pe, m)
            beats = bool(vl["ci_low"] > 0 and m[key + "/per_event_mean"] > base_mean)
            m[key + "/beats_logistic"] = float(beats)
            for name, vals in space_for(kind).items():
                for v in vals:
                    m[key + "/chosen/%s_%s" % (name, v)] = float(sum(c[name] == v for c in chosen))
            modal = max(chosen, key=chosen.count)
            model = clone(make(modal)).fit(df[feats], df[LABEL].to_numpy())
            viol = monotone_violations(model, df[feats], CONSTRAINTS)
            for f, n in viol.items():
                m[key + "/violations/" + f] = float(n)
            assert all(np.isfinite(v) for v in m.values()), "non-finite metric"
            mlflow.log_metrics(m)
            pes[arm] = pe
            print("%s %s | per-event mean %.4f, pooled %.4f | vs logistic %+.4f [%.4f, %.4f] wins %d losses %d | "
                  "vs elevation %+.4f [%.4f, %.4f] | beats logistic: %s | modal grid point %s | violations %s | %.0fs"
                  % (frame, arm, m[key + "/per_event_mean"], m[key + "/pooled_auc"], vl["mean_diff"], vl["ci_low"], vl["ci_high"],
                     vl["wins"], vl["losses"], ve["mean_diff"], ve["ci_low"], ve["ci_high"], beats, modal, viol, time.time() - t0),
                  flush=True)
        if "hgb:con" in pes and "xgb:con" in pes:
            m = {}
            bs = compare(frame + "/xgb_con_vs_hgb_con", pes["hgb:con"], pes["xgb:con"], m)
            mlflow.log_metrics(m)
            print("xgb:con minus hgb:con, paired: %+.4f [%.4f, %.4f] wins %d losses %d"
                  % (bs["mean_diff"], bs["ci_low"], bs["ci_high"], bs["wins"], bs["losses"]), flush=True)
        run_id = run.info.run_id
    os.environ.pop("MLFLOW_TRACKING_PASSWORD", None)
    print("run_id:", run_id)


if __name__ == "__main__":
    main()

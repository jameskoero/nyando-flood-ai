"""Log the Phase C frame comparison (full and mappable frames) and the mappable-frame shuffle null (protocol r4).

Run from a clean checkout of main in Colab. The DagsHub token is read through a hidden prompt."""
import getpass
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.baseline import build_logistic
from src.models.cv import (EVENT, FEATURE_SETS, FLAGS_PATH, LABEL, SEED, land_cover_lookup_scores, load_layer_flags,
                           load_training_frame, loeo_splits, null_distribution, null_interval, out_of_fold_scores,
                           per_group_auc, pooled_auc, reference_scores, restrict_to_mappable, selection_metric_rule)
from src.tracking import tracked_run

N_SEEDS = 100


def frame_metrics(prefix, fr, metrics):
    for name, (p, e, _) in reference_scores(fr).items():
        metrics[prefix + "/ref/" + name + "/pooled_auc"] = p
        metrics[prefix + "/ref/" + name + "/per_event_mean"] = e
    lc = land_cover_lookup_scores(fr)
    metrics[prefix + "/ref/land_cover_lookup/pooled_auc"] = pooled_auc(fr[LABEL], lc)
    metrics[prefix + "/ref/land_cover_lookup/per_event_mean"] = float(np.mean(list(per_group_auc(fr, lc, EVENT, 1).values())))
    splits = loeo_splits(fr, drop_shared_locations=True)
    for arm in ("bare4", "six", "six_no_land_cover"):
        feats = FEATURE_SETS[arm]
        sc = out_of_fold_scores(build_logistic(feats), fr, feats, splits)
        pe = per_group_auc(fr, sc, EVENT, 1)
        metrics[prefix + "/" + arm + "/sel/pooled_auc"] = pooled_auc(fr[LABEL], sc)
        metrics[prefix + "/" + arm + "/sel/per_event_mean"] = float(np.mean(list(pe.values())))


def main():
    df, _ = load_training_frame()
    flags = load_layer_flags(df)
    mapp = restrict_to_mappable(df, flags)
    metrics = {}
    frame_metrics("full", df, metrics)
    frame_metrics("mappable", mapp, metrics)
    feats = FEATURE_SETS["six"]
    pooled, per_event = null_distribution(build_logistic(feats), mapp, feats, N_SEEDS, n_jobs=-1)
    for tag, vals in (("pooled", pooled), ("per_event_mean", per_event)):
        iv = null_interval(vals)
        for k in ("mean", "low", "high"):
            metrics["null_mappable/six/" + tag + "/" + k] = iv[k]
    rule = selection_metric_rule(pooled, per_event)

    bad = sorted(k for k, v in metrics.items() if not np.isfinite(v))
    assert not bad, "non-finite metrics: " + str(bad[:5])
    os.environ.setdefault("MLFLOW_TRACKING_USERNAME", "jmskoero")
    if not os.environ.get("MLFLOW_TRACKING_PASSWORD"):
        os.environ["MLFLOW_TRACKING_PASSWORD"] = getpass.getpass("DagsHub token (hidden): ")
    params = {"arm": "frames-and-mappable-null", "n_seeds": str(N_SEEDS), "base_seed": str(SEED),
              "frames": "full=%d,mappable=%d" % (len(df), len(mapp)),
              "layer_flags_sha256": hashlib.sha256(Path(FLAGS_PATH).read_bytes()).hexdigest(),
              "selection_metric_rule_mappable_frame": rule, "protocol": "docs/PHASE_C_PROTOCOL.md Section 12"}
    tags = {"phase": "C", "arm": "frames-and-mappable-null", "model_status": "baseline-not-a-product"}
    with tracked_run("phase-c-frames-and-mappable-null", "LogisticRegression", params, tags=tags) as (mlflow, run):
        mlflow.log_metrics(metrics)
        run_id = run.info.run_id
    os.environ.pop("MLFLOW_TRACKING_PASSWORD", None)
    print("run_id:", run_id)
    print("pooled null (mappable):", {k: round(v, 4) for k, v in null_interval(pooled).items()})
    print("per-event null (mappable):", {k: round(v, 4) for k, v in null_interval(per_event).items()})
    print("selection_metric_rule on the mappable frame:", rule, "| metrics logged:", len(metrics))


if __name__ == "__main__":
    main()

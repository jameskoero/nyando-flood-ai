"""Log the Phase C reference arms, stratified baseline results and the shuffle-null distribution (protocol r3).

Run from a clean checkout of main in Colab. The DagsHub token is read through a hidden prompt."""
import getpass
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.baseline import build_logistic
from src.models.cv import (EVENT, FEATURE_SETS, LABEL, SEED, land_cover_lookup_scores, load_training_frame,
                           loeo_splits, null_distribution, null_interval, out_of_fold_scores, per_group_auc,
                           pooled_auc, reference_scores, selection_metric_rule, stratified_pooled_auc)
from src.tracking import tracked_run

N_SEEDS = 100


def main():
    df, _ = load_training_frame()
    metrics = {}
    for name, (p, e, _) in reference_scores(df).items():
        metrics["ref/" + name + "/pooled_auc"] = p
        metrics["ref/" + name + "/per_event_mean"] = e
    lc = land_cover_lookup_scores(df)
    metrics["ref/land_cover_lookup/pooled_auc"] = pooled_auc(df[LABEL], lc)
    metrics["ref/land_cover_lookup/per_event_mean"] = float(np.mean(list(per_group_auc(df, lc, EVENT, 1).values())))

    splits = loeo_splits(df, drop_shared_locations=True)
    for arm in ("six", "six_no_land_cover"):
        feats = FEATURE_SETS[arm]
        scores = out_of_fold_scores(build_logistic(feats), df, feats, splits)
        for stratum, v in stratified_pooled_auc(df, scores).items():
            metrics["strat/" + arm + "/" + stratum] = v

    feats = FEATURE_SETS["six"]
    pooled, per_event = null_distribution(build_logistic(feats), df, feats, N_SEEDS, n_jobs=-1)
    for tag, vals in (("pooled", pooled), ("per_event_mean", per_event)):
        iv = null_interval(vals)
        for k in ("mean", "low", "high"):
            metrics["null/six/" + tag + "/" + k] = iv[k]
    rule = selection_metric_rule(pooled, per_event)

    bad = sorted(k for k, v in metrics.items() if not np.isfinite(v))
    assert not bad, "non-finite metrics: " + str(bad[:5])
    os.environ.setdefault("MLFLOW_TRACKING_USERNAME", "jmskoero")
    if not os.environ.get("MLFLOW_TRACKING_PASSWORD"):
        os.environ["MLFLOW_TRACKING_PASSWORD"] = getpass.getpass("DagsHub token (hidden): ")
    params = {"arm": "reference-and-null", "n_seeds": str(N_SEEDS), "base_seed": str(SEED),
              "selection_metric_rule": rule, "protocol": "docs/PHASE_C_PROTOCOL.md Section 10"}
    tags = {"phase": "C", "arm": "reference-and-null", "model_status": "baseline-not-a-product"}
    with tracked_run("phase-c-reference-and-null", "LogisticRegression", params, tags=tags) as (mlflow, run):
        mlflow.log_metrics(metrics)
        run_id = run.info.run_id
    os.environ.pop("MLFLOW_TRACKING_PASSWORD", None)
    print("run_id:", run_id)
    print("pooled null:", {k: round(v, 4) for k, v in null_interval(pooled).items()})
    print("per-event null:", {k: round(v, 4) for k, v in null_interval(per_event).items()})
    print("selection_metric_rule:", rule, "| metrics logged:", len(metrics))


if __name__ == "__main__":
    main()

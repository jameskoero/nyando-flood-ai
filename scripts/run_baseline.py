"""Log the Phase C logistic baseline to MLflow (run from a clean checkout of main, in Colab).

The DagsHub token is read through a hidden prompt, never from an argument or a file."""
import getpass
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.baseline import HEADLINE_SET, run_all
from src.models.cv import FEATURE_SETS, SEED, load_training_frame
from src.tracking import tracked_run


def main():
    df, _ = load_training_frame()
    metrics, _ = run_all(df)
    bad = sorted(k for k, v in metrics.items() if not np.isfinite(v))
    assert not bad, "non-finite metrics: " + str(bad[:5])
    os.environ.setdefault("MLFLOW_TRACKING_USERNAME", "jmskoero")
    if not os.environ.get("MLFLOW_TRACKING_PASSWORD"):
        os.environ["MLFLOW_TRACKING_PASSWORD"] = getpass.getpass("DagsHub token (hidden): ")
    params = {
        "arm": "baseline",
        "headline_features": ",".join(FEATURE_SETS[HEADLINE_SET]),
        "feature_sets": ",".join(FEATURE_SETS),
        "splits": "loeo,loeo_drop_shared_locations,lowo,shuffle_null_within_events",
        "scaler": "StandardScaler",
        "clay_policy": "median+indicator (training folds only)",
        "land_cover": "one-hot",
        "C": "1.0 (default, untuned)",
        "smote": "none",
        "seed": str(SEED),
        "protocol": "docs/PHASE_C_PROTOCOL.md",
    }
    tags = {"phase": "C", "arm": "baseline", "model_status": "baseline-not-a-product"}
    with tracked_run("phase-c-baseline-logreg", "LogisticRegression", params, tags=tags) as (mlflow, run):
        mlflow.log_metrics(metrics)
        run_id = run.info.run_id
    os.environ.pop("MLFLOW_TRACKING_PASSWORD", None)
    print("run_id:", run_id)
    for k in ("loeo", "sel", "null_sel"):
        print(HEADLINE_SET, k, "pooled", round(metrics[HEADLINE_SET + "/" + k + "/pooled_auc"], 4),
              "| per-event mean", round(metrics[HEADLINE_SET + "/" + k + "/per_event_mean"], 4))
    print("metrics logged:", len(metrics))


if __name__ == "__main__":
    main()

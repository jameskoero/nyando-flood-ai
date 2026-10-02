"""Phase C baseline: logistic regression, scored with the harness in src/models/cv.py.

Settings are fixed by docs/PHASE_C_PROTOCOL.md (Sections 3 to 5): no tuning, no SMOTE, land_cover
one-hot, clay_percent median-imputed with a missing indicator, both fitted on training folds only.
This module is offline: it imports neither MLflow nor any network client.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.cv import (CATEGORICAL, CONSTRAINTS, FEATURE_SETS, LABEL, loeo_splits, lowo_splits,
                           out_of_fold_scores, shuffle_labels_within_events, summarize_scores)

BLANK_PRONE = "clay_percent"
HEADLINE_SET = "six"


def build_logistic(features):
    """Unfitted pipeline. The missing-value indicator is always built, so its column exists in every fold."""
    cat = [f for f in features if f in CATEGORICAL]
    num = [f for f in features if f not in CATEGORICAL]
    parts = [("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num)]
    if BLANK_PRONE in features:
        parts.append(("blank", MissingIndicator(features="all"), [BLANK_PRONE]))
    if cat:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat))
    return Pipeline([("prep", ColumnTransformer(parts)), ("clf", LogisticRegression(max_iter=3000))])


def coefficient_signs(fitted, features):
    """Sign (+1, -1 or 0) of each constrained numeric feature's coefficient. Reported, never enforced."""
    num = [f for f in features if f not in CATEGORICAL]
    coef = fitted.named_steps["clf"].coef_[0]
    return {f: int(np.sign(coef[num.index(f)])) for f in CONSTRAINTS if f in num}


def _summary(df, est, features, splits):
    return summarize_scores(df, out_of_fold_scores(est, df, features, splits))


def evaluate_arm(df, features):
    """Leave-one-event-out, the selection split, leave-one-ward-out, and the shuffle null (selection split)."""
    est = build_logistic(features)
    out = {
        "loeo": _summary(df, est, features, loeo_splits(df)),
        "sel": _summary(df, est, features, loeo_splits(df, drop_shared_locations=True)),
    }
    wards = lowo_splits(df)
    out["lowo"] = _summary(df, est, features, wards) if wards else None
    null = shuffle_labels_within_events(df)
    out["null_sel"] = _summary(null, est, features, loeo_splits(null, drop_shared_locations=True))
    return out


def sensitivity_frames(df):
    """The protocol's sensitivity runs: without the elevation-floor rows, the zero-distance rows, or blank clay."""
    floor = df["elevation"] == df["elevation"].min()
    return {
        "no_floor_rows": df[~floor].reset_index(drop=True),
        "no_zero_distance": df[df["distance_river"] != 0].reset_index(drop=True),
        "clay_present": df[df["clay_percent"].notna()].reset_index(drop=True),
    }


def _key(name):
    return str(name).replace("/", "-").replace(" ", "_")


def run_all(df):
    """Evaluate every protocol arm. Returns (metrics, details); metrics is a flat dict of floats for logging."""
    metrics, details = {}, {}
    for arm, feats in FEATURE_SETS.items():
        r = evaluate_arm(df, feats)
        details[arm] = r
        for split in ("loeo", "sel", "null_sel"):
            metrics[arm + "/" + split + "/pooled_auc"] = r[split]["pooled_auc"]
            metrics[arm + "/" + split + "/per_event_mean"] = r[split]["per_event_mean"]
        if r["lowo"] is not None:
            metrics[arm + "/lowo/pooled_auc"] = r["lowo"]["pooled_auc"]
            for w, a in r["lowo"]["per_ward"].items():
                metrics[arm + "/lowo/ward/" + _key(w)] = a
    for e, a in details[HEADLINE_SET]["sel"]["per_event"].items():
        metrics[HEADLINE_SET + "/sel/event/" + _key(e)] = a
    feats = FEATURE_SETS[HEADLINE_SET]
    fitted = build_logistic(feats).fit(df[feats], df[LABEL])
    for f, s in coefficient_signs(fitted, feats).items():
        metrics["sign/" + f] = float(s)
    for name, sub in sensitivity_frames(df).items():
        r = _summary(sub, build_logistic(feats), feats, loeo_splits(sub, drop_shared_locations=True))
        details["sensitivity/" + name] = r
        metrics["sens/" + name + "/sel/pooled_auc"] = r["pooled_auc"]
        metrics["sens/" + name + "/sel/per_event_mean"] = r["per_event_mean"]
    return metrics, details

"""Phase C evaluation harness (see docs/PHASE_C_PROTOCOL.md). No model is trained in this module.

It fixes the grouped splits, the metrics and the paired bootstrap that every Phase C model is
scored with, so the protocol exists in code before any result does.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from src.tracking import current_training_entry

ROOT = Path(__file__).resolve().parents[2]
LABEL, EVENT, WARD = "flooded", "event_id", "ward"
SEED = 42
MIN_CLASS_N = 30

BARE_4 = ["elevation", "slope", "rainfall_3day", "distance_river"]
FEATURES_6 = ["elevation", "slope", "rainfall_3day", "distance_river", "clay_percent", "land_cover"]
FEATURES_7 = FEATURES_6 + ["hand"]
CATEGORICAL = ["land_cover"]
CONSTRAINTS = {"rainfall_3day": 1, "elevation": -1, "distance_river": -1, "slope": -1}
FEATURE_SETS = {
    "bare4": BARE_4,
    "six": FEATURES_6,
    "seven": FEATURES_7,
    "six_no_clay": [f for f in FEATURES_6 if f != "clay_percent"],
    "six_no_land_cover": [f for f in FEATURES_6 if f != "land_cover"],
    "six_no_rainfall": [f for f in FEATURES_6 if f != "rainfall_3day"],
}


def load_training_frame(repo_root=None):
    """Return (frame, sha256) for the manifest's training file; fail if it does not match the manifest."""
    rel, entry, real_hash = current_training_entry(repo_root)
    root = Path(repo_root) if repo_root else ROOT
    if entry.get("sha256") != real_hash:
        raise ValueError("training file hash does not match data/MANIFEST.json")
    df = pd.read_csv(root / rel)
    if len(df) != entry.get("row_count"):
        raise ValueError("training file row count does not match data/MANIFEST.json")
    return df, real_hash


def location_keys(df):
    """One string per row identifying its location (lon, lat rounded to 6 decimals)."""
    return df["lon"].round(6).astype(str) + "," + df["lat"].round(6).astype(str)


def loeo_splits(df, drop_shared_locations=False):
    """Leave-one-event-out as a list of (event, train_positions, test_positions).

    With drop_shared_locations, training rows at any location present in the held-out event are removed.
    """
    ev = df[EVENT].to_numpy()
    codes = pd.factorize(location_keys(df))[0]
    out = []
    for e in sorted(set(ev)):
        test = np.flatnonzero(ev == e)
        train = np.flatnonzero(ev != e)
        if drop_shared_locations:
            train = train[~np.isin(codes[train], codes[test])]
        out.append((e, train, test))
    return out


def lowo_splits(df, min_class_n=MIN_CLASS_N):
    """Leave-one-ward-out for wards with at least min_class_n flood and control rows."""
    counts = df.groupby([WARD, LABEL]).size().unstack(fill_value=0)
    wards = sorted(w for w in counts.index
                   if counts.loc[w].get(0, 0) >= min_class_n and counts.loc[w].get(1, 0) >= min_class_n)
    w_arr = df[WARD].to_numpy()
    return [(w, np.flatnonzero(w_arr != w), np.flatnonzero(w_arr == w)) for w in wards]


def out_of_fold_scores(estimator, df, features, splits):
    """Fit a fresh clone on each split's training rows; return scores aligned with df (NaN if never tested)."""
    X = df[features]
    y = df[LABEL].to_numpy()
    scores = np.full(len(df), np.nan)
    for _, train, test in splits:
        model = clone(estimator).fit(X.iloc[train], y[train])
        scores[test] = model.predict_proba(X.iloc[test])[:, 1]
    return scores


def pooled_auc(y, scores):
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    ok = ~np.isnan(scores)
    return float(roc_auc_score(y[ok], scores[ok]))


def per_group_auc(df, scores, group_col, min_class_n=1):
    """AUC inside each group that has at least min_class_n rows of each class; other groups are omitted."""
    y_all = df[LABEL].to_numpy()
    scores = np.asarray(scores, dtype=float)
    out = {}
    for g, idx in df.groupby(group_col).indices.items():
        s, y = scores[idx], y_all[idx]
        ok = ~np.isnan(s)
        s, y = s[ok], y[ok]
        pos = int(y.sum())
        if pos >= min_class_n and len(y) - pos >= min_class_n:
            out[g] = float(roc_auc_score(y, s))
    return out


def summarize_scores(df, scores):
    per_event = per_group_auc(df, scores, EVENT, 1)
    per_ward = per_group_auc(df, scores, WARD, MIN_CLASS_N)
    return {
        "pooled_auc": pooled_auc(df[LABEL], scores),
        "per_event": per_event,
        "per_event_mean": float(np.mean(list(per_event.values()))),
        "per_ward": per_ward,
    }


def paired_event_bootstrap(auc_a, auc_b, n_boot=10000, seed=SEED, alpha=0.05):
    """Paired bootstrap over events of the per-event AUC difference (b minus a)."""
    events = sorted(set(auc_a) & set(auc_b))
    if not events:
        raise ValueError("no events in common")
    d = np.array([auc_b[e] - auc_a[e] for e in events])
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"n_events": len(events), "mean_diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "ties": int((d == 0).sum())}


def shuffle_labels_within_events(df, seed=SEED):
    """Copy of df with the labels permuted inside each event (the null run: expected AUC near 0.5)."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    col = out.columns.get_loc(LABEL)
    labels = df[LABEL].to_numpy()
    for _, idx in df.groupby(EVENT).indices.items():
        out.iloc[idx, col] = rng.permutation(labels[idx])
    return out

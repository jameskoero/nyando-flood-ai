"""Phase C robustness battery (docs/PHASE_C_PROTOCOL.md, Section 16): subsets, splits and statistics. Offline: no MLflow, no network."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.neighbors import BallTree

from src.models.cv import (EVENT, LABEL, SEED, loeo_splits, location_keys, paired_event_bootstrap, per_group_auc, pooled_auc)

FLOOR_ELEVATION = 1130.5
BUFFER_M = 1000.0
EARTH_RADIUS_M = 6371008.8
CUTOFF_YEAR = 2021
MIN_TEST_EVENTS = 5
N_REPEATS = 20
COORDS = (("longitude", "latitude"), ("lon", "lat"), ("lng", "lat"))


def subset_masks(df):
    """Boolean masks (True = kept) of the three sensitivity subsets of Section 16.4."""
    return {"no_floor_rows": ~np.isclose(df["elevation"].to_numpy(float), FLOOR_ELEVATION, atol=1e-6),
            "no_zero_distance": df["distance_river"].to_numpy(float) != 0.0,
            "clay_present": df["clay_percent"].notna().to_numpy()}


def coordinate_columns(df):
    """(longitude, latitude) column names, only if the values are plausible degrees."""
    for lon, lat in COORDS:
        if lon in df.columns and lat in df.columns:
            if np.nanmax(np.abs(df[lon].to_numpy(float))) <= 180 and np.nanmax(np.abs(df[lat].to_numpy(float))) <= 90:
                return lon, lat
    raise KeyError("no longitude and latitude columns in degrees among %s; the columns are %s" % (list(COORDS), list(df.columns)))


def buffered_splits(df, radius_m=BUFFER_M):
    """The selection split (leave one event out, shared locations removed) with, in addition, every training row within radius_m
    (haversine) of any held-out row removed. Same (event, train_idx, test_idx) triples as loeo_splits."""
    lon, lat = coordinate_columns(df)
    pts = np.radians(df[[lat, lon]].to_numpy(float))
    tree = BallTree(pts, metric="haversine")
    y = df[LABEL].to_numpy()
    out = []
    for event, tr, te in loeo_splits(df, drop_shared_locations=True):
        near = np.unique(np.concatenate(list(tree.query_radius(pts[te], r=radius_m / EARTH_RADIUS_M))))
        keep = tr[~np.isin(tr, near)]
        if len(np.unique(y[keep])) < 2:
            raise ValueError("buffering leaves a single-class training set for event %s" % event)
        out.append((event, keep, te))
    return out


def event_year_table(df):
    """Per event: first and last year of the event_date values of its rows (an event can span several scene dates)."""
    year = pd.to_datetime(df["event_date"], errors="raise").dt.year.to_numpy()
    return pd.DataFrame({EVENT: df[EVENT].to_numpy(), "year": year}).groupby(EVENT)["year"].agg(["min", "max"])


def temporal_split(df, cutoff=CUTOFF_YEAR):
    """Training rows from events whose scene dates are all in or before the cutoff year (minus rows at any test location); test rows from
    events whose scene dates are all after it. An event with scene dates on both sides is in neither set (see straddling_events)."""
    g = event_year_table(df)
    ev = df[EVENT].to_numpy()
    train = np.flatnonzero(np.isin(ev, g.index[g["max"] <= cutoff].to_numpy()))
    test = np.flatnonzero(np.isin(ev, g.index[g["min"] > cutoff].to_numpy()))
    keys = pd.factorize(location_keys(df))[0]
    return train[~np.isin(keys[train], np.unique(keys[test]))], test


def straddling_events(df, cutoff=CUTOFF_YEAR):
    """Events with scene dates on both sides of the cutoff year."""
    g = event_year_table(df)
    return sorted(g.index[(g["min"] <= cutoff) & (g["max"] > cutoff)].tolist())


def temporal_holdout(df, arms, cutoff=CUTOFF_YEAR):
    """arms = {name: (builder, features)}. Per-event AUCs on the test events for each arm and for elevation only."""
    train, test = temporal_split(df, cutoff)
    y = df[LABEL].to_numpy()
    held = df.iloc[test].reset_index(drop=True)
    events = {}
    for name, (builder, feats) in arms.items():
        model = builder(feats).fit(df.iloc[train][feats], y[train])
        events[name] = per_group_auc(held, model.predict_proba(held[feats])[:, 1], EVENT, 1)
    events["elevation_only"] = per_group_auc(held, -held["elevation"].to_numpy(float), EVENT, 1)
    return {"cutoff": cutoff, "train_rows": int(len(train)), "test_rows": int(len(test)), "events": events,
            "straddling_events": len(straddling_events(df, cutoff))}


def prior_only_scores(df):
    """Section 16.1: every held-out row scores the training prevalence of its fold (selection split)."""
    y = df[LABEL].to_numpy()
    scores = np.full(len(df), np.nan)
    for _, tr, te in loeo_splits(df, drop_shared_locations=True):
        scores[te] = y[tr].mean()
    return scores


def permutation_audit(builder, df, features, n_repeats=N_REPEATS, seed=SEED):
    """Section 16.2 on the selection split; the model of each fold is fitted once. Returns
    {"base": {event: AUC}, "base_pooled": float, "within_drop": {feature: {event: drop}}, "pooled_drop": {feature: [drop per repeat]}}."""
    y = df[LABEL].to_numpy()
    X = df[list(features)]
    rng = np.random.default_rng(seed)
    folds, base = [], np.full(len(df), np.nan)
    for _, tr, te in loeo_splits(df, drop_shared_locations=True):
        model = clone(builder(features)).fit(X.iloc[tr], y[tr])
        folds.append((te, model))
        base[te] = model.predict_proba(X.iloc[te])[:, 1]
    base_pe = per_group_auc(df, base, EVENT, 1)
    base_pooled = pooled_auc(df[LABEL], base)
    within = {f: {e: [] for e in base_pe} for f in features}
    pooled = {f: [] for f in features}
    for f in features:
        col = X[f].to_numpy()
        for _ in range(n_repeats):
            shuffled = col[rng.permutation(len(col))]
            s_in, s_all = np.full(len(df), np.nan), np.full(len(df), np.nan)
            for te, model in folds:
                Xi, Xa = X.iloc[te].copy(), X.iloc[te].copy()
                Xi[f] = col[te][rng.permutation(len(te))]
                Xa[f] = shuffled[te]
                s_in[te] = model.predict_proba(Xi)[:, 1]
                s_all[te] = model.predict_proba(Xa)[:, 1]
            pe = per_group_auc(df, s_in, EVENT, 1)
            for e in base_pe:
                within[f][e].append(pe[e])
            pooled[f].append(base_pooled - pooled_auc(df[LABEL], s_all))
    drops = {f: {e: base_pe[e] - float(np.mean(within[f][e])) for e in base_pe} for f in features}
    return {"base": base_pe, "base_pooled": base_pooled, "within_drop": drops, "pooled_drop": pooled}


def paired(ref, other):
    """Paired per-event bootstrap of other minus ref (97.5% interval) plus the means over the common events."""
    bs = paired_event_bootstrap(ref, other, alpha=0.025)
    common = sorted(set(ref) & set(other))
    return {**bs, "n": len(common), "ref_mean": float(np.mean([ref[e] for e in common])), "other_mean": float(np.mean([other[e] for e in common]))}


def beats(ref, other):
    """The rule of Section 13.4 item 1: interval lower end above 0 and a higher mean."""
    p = paired(ref, other)
    return bool(p["ci_low"] > 0 and p["other_mean"] > p["ref_mean"]), p


def reversal(champion, boosters):
    """Section 16.4: a subset reverses the verdict if any booster beats the champion there."""
    detail = {}
    for arm, events in boosters.items():
        ok, p = beats(champion, events)
        detail[arm] = {**p, "beats": ok}
    return any(d["beats"] for d in detail.values()), detail

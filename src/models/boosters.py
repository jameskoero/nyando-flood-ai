"""Phase C boosters and their nested tuning (docs/PHASE_C_PROTOCOL.md, Section 13). Offline: no MLflow, no network."""
from itertools import product

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier

from src.models.cv import (CONSTRAINTS, EVENT, LABEL, SEED, loeo_splits, location_keys, out_of_fold_scores,
                           per_group_auc)

LAND_COVER_CLASSES = [10, 20, 30, 40, 50, 60, 80, 90]
INNER_FOLDS = 4
HGB_SPACE = {"max_depth": (2, 4), "max_iter": (100, 300), "min_samples_leaf": (20, 100)}
XGB_SPACE = {"max_depth": (2, 4), "n_estimators": (100, 300), "min_child_weight": (1, 10)}
HGB_FIXED = {"learning_rate": 0.05, "l2_regularization": 1.0, "early_stopping": False}
XGB_FIXED = {"learning_rate": 0.05, "reg_lambda": 1.0, "subsample": 1.0, "tree_method": "hist"}


def expand(space):
    """Grid points in a fixed order, simplest first (the first parameter varies slowest)."""
    keys = list(space)
    return [dict(zip(keys, v)) for v in product(*(space[k] for k in keys))]


def describe(space):
    return "; ".join("%s in {%s}" % (k, ", ".join(str(v) for v in vals)) for k, vals in space.items())


def describe_fixed(fixed):
    return "; ".join("%s = %s" % (k, v) for k, v in fixed.items())


def _constraints(features):
    return {f: d for f, d in CONSTRAINTS.items() if f in features}


def build_hgb(features, constrained, **params):
    cat = [f for f in features if f == "land_cover"]
    return HistGradientBoostingClassifier(categorical_features=cat or None,
                                          monotonic_cst=_constraints(features) if constrained else None,
                                          random_state=SEED, **HGB_FIXED, **params)


class XGBBooster(ClassifierMixin, BaseEstimator):
    """XGBoost with monotone constraints and land_cover as a fixed-category column (clone-friendly)."""

    def __init__(self, features=None, constrained=True, max_depth=2, n_estimators=100, min_child_weight=1):
        self.features = features
        self.constrained = constrained
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.min_child_weight = min_child_weight

    def _prep(self, X):
        X = X[list(self.features)].copy()
        if "land_cover" in X.columns:
            X["land_cover"] = X["land_cover"].astype(pd.CategoricalDtype(categories=LAND_COVER_CLASSES))
        return X

    def fit(self, X, y):
        import xgboost as xgb
        kw = dict(max_depth=self.max_depth, n_estimators=self.n_estimators, min_child_weight=self.min_child_weight,
                  enable_categorical=True, random_state=SEED, n_jobs=1, **XGB_FIXED)
        if self.constrained:
            kw["monotone_constraints"] = tuple(CONSTRAINTS.get(f, 0) for f in self.features)
        self.model_ = xgb.XGBClassifier(**kw).fit(self._prep(X), np.asarray(y))
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(self._prep(X))


def make_estimator(kind, features, constrained):
    """Return f(params) -> unfitted estimator, for kind 'hgb' or 'xgb'."""
    if kind == "hgb":
        return lambda params: build_hgb(features, constrained, **params)
    if kind == "xgb":
        return lambda params: XGBBooster(features=list(features), constrained=constrained, **params)
    raise ValueError(kind)
# ---- part 2: tuning and the monotonicity probe ----


def space_for(kind):
    return HGB_SPACE if kind == "hgb" else XGB_SPACE


def inner_splits(df, train_idx, n_folds=INNER_FOLDS):
    """Event-grouped inner folds over the rows at train_idx (positions in df). Events are dealt round-robin in sorted
    order; training rows at any location present in the inner test fold are dropped."""
    train_idx = np.asarray(train_idx)
    ev = df[EVENT].to_numpy()[train_idx]
    codes = pd.factorize(location_keys(df))[0][train_idx]
    fold_of = {e: i % n_folds for i, e in enumerate(sorted(set(ev)))}
    fold = np.array([fold_of[e] for e in ev])
    out = []
    for k in range(n_folds):
        te = fold == k
        tr = ~te & ~np.isin(codes, codes[te])
        out.append((k, train_idx[tr], train_idx[te]))
    return out


def select_params(make, space, df, features, train_idx):
    """Grid point with the highest inner per-event mean AUC (rounded to 4 decimals); ties keep the earlier point."""
    splits = inner_splits(df, train_idx)
    grid = expand(space)
    best, best_score = grid[0], -np.inf
    for params in grid:
        sc = out_of_fold_scores(make(params), df, features, splits)
        pe = per_group_auc(df, sc, EVENT, 1)
        score = round(float(np.mean(list(pe.values()))), 4) if pe else -np.inf
        if score > best_score:
            best, best_score = params, score
    return best, best_score


def nested_scores(make, space, df, features, n_jobs=1):
    """Outer leave-one-event-out (shared locations removed); each outer fold tunes on its training rows by inner
    event-grouped CV, refits with the chosen point and scores the held-out event. Returns (scores, chosen)."""
    y = df[LABEL].to_numpy()
    X = df[features]

    def one(train, test):
        params, _ = select_params(make, space, df, features, train)
        model = clone(make(params)).fit(X.iloc[train], y[train])
        return test, model.predict_proba(X.iloc[test])[:, 1], params

    outer = loeo_splits(df, drop_shared_locations=True)
    res = Parallel(n_jobs=n_jobs)(delayed(one)(tr, te) for _, tr, te in outer)
    scores = np.full(len(df), np.nan)
    chosen = []
    for te, p, params in res:
        scores[te] = p
        chosen.append(params)
    return scores, chosen


def monotone_violations(model, X, constraints, n_rows=200, n_grid=25, tol=1e-9):
    """Real rows with one constrained feature swept over its observed 1st to 99th percentile, the others fixed.
    Returns {feature: number of rows whose score moves against the declared direction by more than tol}."""
    rows = X.iloc[np.linspace(0, len(X) - 1, min(n_rows, len(X))).astype(int)].reset_index(drop=True)
    out = {}
    for f, d in constraints.items():
        grid = np.linspace(*np.percentile(X[f].dropna(), [1, 99]), n_grid)
        rep = rows.loc[rows.index.repeat(n_grid)].reset_index(drop=True)
        rep[f] = np.tile(grid, len(rows))
        p = model.predict_proba(rep)[:, 1].reshape(len(rows), n_grid)
        out[f] = int(((d * np.diff(p, axis=1)) < -tol).any(axis=1).sum())
    return out

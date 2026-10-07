"""Phase D Stage 2 helper (docs/PHASE_D_PROTOCOL.md Section 9): the nested selection of src/models/boosters.py computed one outer fold at a time and cached, so an interrupted Colab run resumes.
Its scores and chosen points equal boosters.nested_scores (tests/test_phase_d_stage2.py). Returns None when the time budget ran out before every outer fold was done."""
import json
import time

import numpy as np
from sklearn.base import clone

from src.models.boosters import select_params
from src.models.cv import LABEL, loeo_splits


def _plain(p):
    return {k: tuple(v) if isinstance(v, list) else v for k, v in p.items()}


def nested_cached(make, space, df, features, cache, budget_s=None, label=""):
    X, y = df[features], df[LABEL].to_numpy()
    store = json.loads(cache.read_text()) if cache.exists() else {}
    t0, outer = time.time(), loeo_splits(df, drop_shared_locations=True)
    for e, tr, te in outer:
        if str(e) in store:
            continue
        if budget_s is not None and time.time() - t0 > budget_s:
            return None
        params, _ = select_params(make, space, df, features, tr)
        model = clone(make(params)).fit(X.iloc[tr], y[tr])
        store[str(e)] = {"params": params, "p": model.predict_proba(X.iloc[te])[:, 1].tolist()}
        cache.write_text(json.dumps(store))
        print("%s: event %s done (%d of %d, %.0f s)" % (label, e, len(store), len(outer), time.time() - t0), flush=True)
    scores, chosen = np.full(len(df), np.nan), []
    for e, tr, te in outer:
        scores[te] = store[str(e)]["p"]
        chosen.append(_plain(store[str(e)]["params"]))
    return scores, chosen

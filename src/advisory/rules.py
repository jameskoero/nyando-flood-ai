"""Phase E rules as code (docs/PHASE_E_PROTOCOL.md; register row D49): pure functions and constants, no model, no network, no data files."""
import numpy as np

LEVELS = ("LOW", "MEDIUM", "HIGH")
CUT_QUANTILES = (1 / 3, 2 / 3)       # protocol Section 3: terciles of the reference scores, never tuned
MIN_ROWS_PER_LEVEL = 100             # gate V1
N_BOOT, ALPHA, SEED = 10000, 0.025, 42
PROVIDERS = ("gemini", "groq")       # then the deterministic template (protocol Section 5)
PROVIDER_TIMEOUT_S, TOTAL_TIMEOUT_S, CACHE_TTL_S = 8, 12, 3600
VALIDATED, NOT_VALIDATED = "LEVELS VALIDATED", "LEVELS NOT VALIDATED"
CLAIM = "ranking score, not a flood probability"


def cutpoints(reference):
    lo, hi = np.quantile(np.asarray(reference, dtype=float), CUT_QUANTILES)
    return float(lo), float(hi)


def level_of(score, cuts):
    return LEVELS[0] if score < cuts[0] else (LEVELS[1] if score < cuts[1] else LEVELS[2])


def levels_of(scores, cuts):
    return [level_of(float(s), cuts) for s in scores]


def level_rates(levels, flooded):
    out = {}
    for lv in LEVELS:
        idx = [i for i, x in enumerate(levels) if x == lv]
        k = int(sum(int(flooded[i]) for i in idx))
        out[lv] = {"rows": len(idx), "flood": k, "share": (k / len(idx)) if idx else None}
    return out


def cluster_bootstrap_diff(levels, flooded, events, n_boot=N_BOOT, seed=SEED, alpha=ALPHA):
    """HIGH minus LOW flood share, resampling whole events with replacement. Draws in which HIGH or LOW is empty are dropped."""
    ev = sorted(set(events))
    cnt = {lv: np.zeros(len(ev)) for lv in ("LOW", "HIGH")}
    tot = {lv: np.zeros(len(ev)) for lv in ("LOW", "HIGH")}
    pos = {e: i for i, e in enumerate(ev)}
    for lv, y, e in zip(levels, flooded, events):
        if lv in cnt:
            tot[lv][pos[e]] += 1
            cnt[lv][pos[e]] += int(y)
    draws = np.random.default_rng(seed).integers(0, len(ev), size=(n_boot, len(ev)))
    with np.errstate(invalid="ignore", divide="ignore"):
        share = {lv: cnt[lv][draws].sum(axis=1) / tot[lv][draws].sum(axis=1) for lv in cnt}
    d = share["HIGH"] - share["LOW"]
    d = d[~np.isnan(d)]
    point = (cnt["HIGH"].sum() / tot["HIGH"].sum()) - (cnt["LOW"].sum() / tot["LOW"].sum())
    lo, hi = np.percentile(d, [100 * alpha, 100 * (1 - alpha)])
    return {"diff": float(point), "ci_low": float(lo), "ci_high": float(hi), "draws_used": int(len(d))}


def gates(rates, ci_low):
    s = [rates[lv]["share"] for lv in LEVELS]
    v1 = all(rates[lv]["rows"] >= MIN_ROWS_PER_LEVEL for lv in LEVELS)
    v2 = v1 and s[0] < s[1] < s[2]
    return {"V1": v1, "V2": bool(v2), "V3": bool(v1 and ci_low > 0)}


def outcome(g):
    return VALIDATED if all(g.values()) else NOT_VALIDATED

"""Phase L statistics (docs/PHASE_L_PROTOCOL.md Section 4): the within-class concordance W under an event-clustered bootstrap, vectorised and checked against the brute-force definition in src/models/phase_l.py. Intervals use the constants of src/advisory/rules.py (N_BOOT, ALPHA, SEED). Nothing here reads a file or the network."""
import numpy as np

from src.advisory import rules as R

CHUNK = 500


def multiplicities(n_events, n_boot, seed=R.SEED):
    """M[b, e]: how many times event e is drawn in resample b (whole events are resampled with replacement)."""
    draws = np.random.default_rng(seed).integers(0, n_events, size=(n_boot, n_events))
    m = np.zeros((n_boot, n_events), dtype=np.int32)
    for b in range(n_boot):
        m[b] = np.bincount(draws[b], minlength=n_events)
    return m


def _class_terms(score, flood, wt):
    """Concordant weight (ties half) and pair weight for one land-cover class: score (n,), flood (n,) bool, wt (B, n) row weights."""
    order = np.argsort(score, kind="stable")
    s, f, w = score[order], flood[order], wt[:, order]
    first, last = np.searchsorted(s, s, side="left"), np.searchsorted(s, s, side="right")
    cum = np.zeros((w.shape[0], len(s) + 1))
    cum[:, 1:] = np.cumsum(w * (~f), axis=1)
    less, tie = cum[:, first], cum[:, last] - cum[:, first]
    num = (w * f * (less + 0.5 * tie)).sum(axis=1)
    pairs = (w * f).sum(axis=1) * (w * (~f)).sum(axis=1)
    return num, pairs


def w_resamples(scores, flood, classes, events, use, mult):
    """W for every row of mult (one resample per row); NaN where a resample has no pair. events must be sorted(set(events)) in the column order of mult."""
    ev = {e: i for i, e in enumerate(sorted(set(events.tolist())))}
    wt = mult[:, np.array([ev[e] for e in events.tolist()])].astype(float)
    num, pairs = np.zeros(mult.shape[0]), np.zeros(mult.shape[0])
    for c in use:
        m = classes == c
        n_, p_ = _class_terms(scores[m], flood[m], wt[:, m])
        num += n_
        pairs += p_
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / pairs


def pair_count(flood, classes, use):
    return int(sum(int(flood[classes == c].sum()) * int((~flood[classes == c]).sum()) for c in use))


def delta_w(scores_a, scores_b, flooded, classes, events, use, n_boot=R.N_BOOT, seed=R.SEED):
    """W of arm a minus W of arm b: the point estimate and the event-clustered percentile interval (2.5th to 97.5th percentile of N_BOOT resamples)."""
    sa, sb = np.asarray(scores_a, float), np.asarray(scores_b, float)
    flood, classes, events = np.asarray(flooded).astype(bool), np.asarray(classes), np.asarray(events)
    n_ev = len(set(events.tolist()))
    one = np.ones((1, n_ev), dtype=np.int32)
    wa, wb = w_resamples(sa, flood, classes, events, use, one)[0], w_resamples(sb, flood, classes, events, use, one)[0]
    if np.isnan(wa) or np.isnan(wb):
        return None
    mult = multiplicities(n_ev, n_boot, seed)
    d = np.concatenate([w_resamples(sa, flood, classes, events, use, mult[i:i + CHUNK]) - w_resamples(sb, flood, classes, events, use, mult[i:i + CHUNK]) for i in range(0, n_boot, CHUNK)])
    d = d[~np.isnan(d)]
    lo, hi = np.percentile(d, [100 * R.ALPHA, 100 * (1 - R.ALPHA)])
    return {"w_a": float(wa), "w_b": float(wb), "delta": float(wa - wb), "ci_low": float(lo), "ci_high": float(hi), "draws_used": int(len(d)), "pairs": pair_count(flood, classes, use)}

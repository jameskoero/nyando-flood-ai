"""Phase K statistics (docs/PHASE_K_PROTOCOL.md Sections 4 and 5): the per-ward metrics M1 to M4 and the per-frame report. Intervals are event-clustered percentile bootstraps that use N_BOOT, SEED and ALPHA of src/advisory/rules.py (the E1 convention); HIGH minus LOW is the E1 function itself. Nothing here reads a file or the network."""
import numpy as np

from src.advisory import rules as R
from src.models import phase_k as K

NONE_BOOT = {"diff": None, "ci_low": None, "ci_high": None, "draws_used": 0}


def _draws(n):
    return np.random.default_rng(R.SEED).integers(0, n, size=(R.N_BOOT, n))


def _interval(d):
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return None, None
    lo, hi = np.percentile(d, [100 * R.ALPHA, 100 * (1 - R.ALPHA)])
    return float(lo), float(hi)


def mean_ci(values):
    """M3: the mean of the per-event values and its event-clustered percentile interval."""
    v = np.asarray(list(values), dtype=float)
    if len(v) == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "events": 0}
    lo, hi = _interval(v[_draws(len(v))].mean(axis=1))
    return {"mean": float(v.mean()), "ci_low": lo, "ci_high": hi, "events": int(len(v))}


def gap_ci(ward, pooled):
    """M4: the ward's mean per-event AUC minus the pooled one, with the events of the frame resampled jointly. ward and pooled map event to AUC."""
    events = sorted(set(ward) | set(pooled))
    if not events or not ward or not pooled:
        return {"gap": None, "ci_low": None, "ci_high": None}
    wv = np.array([ward.get(e, np.nan) for e in events], dtype=float)
    pv = np.array([pooled.get(e, np.nan) for e in events], dtype=float)
    idx = _draws(len(events))
    with np.errstate(invalid="ignore", divide="ignore"):
        wm = np.where(np.isnan(wv), 0.0, wv)[idx].sum(axis=1) / (~np.isnan(wv))[idx].sum(axis=1)
        pm = np.where(np.isnan(pv), 0.0, pv)[idx].sum(axis=1) / (~np.isnan(pv))[idx].sum(axis=1)
    lo, hi = _interval(wm - pm)
    return {"gap": float(np.nanmean(wv) - np.nanmean(pv)), "ci_low": lo, "ci_high": hi}


def ward_block(levels, flooded, events, ward_aucs, pooled_aucs):
    """M1 to M4 for one estimable ward in one frame. The AUC dictionaries hold millionths, as phase_d_stats.aucs returns them."""
    rates = R.level_rates(levels, flooded)
    bands = dict(NONE_BOOT) if rates["LOW"]["rows"] == 0 or rates["HIGH"]["rows"] == 0 else R.cluster_bootstrap_diff(levels, flooded, events)
    w = {e: v / 1e6 for e, v in ward_aucs.items()}
    p = {e: v / 1e6 for e, v in pooled_aucs.items()}
    return {"rates": rates, "bands": bands, "auc": mean_ci(w.values()), "gap": gap_ci(w, p)}


def frame_report(wards, flooded, events, levels, aucs_fn):
    """The report of one frame. aucs_fn(mask) returns the per-event AUC dictionary of the rows in mask."""
    wards, flooded, events, levels = (np.asarray(x) for x in (wards, flooded, events, levels))
    pooled = aucs_fn(np.ones(len(wards), dtype=bool))
    out = {"rows": int(len(wards)), "wards": {}}
    for ward in K.WARDS:
        m = wards == ward
        fl, ct, both = K.ward_counts(flooded[m], events[m])
        ok, why = K.estimability(fl, ct, both)
        entry = {"flood_rows": int(fl), "control_rows": int(ct), "events_both": int(both), "estimable": bool(ok), "reason": why}
        if ok:
            entry.update(ward_block(list(levels[m]), flooded[m], events[m], aucs_fn(m), pooled))
            gap = (entry["gap"]["ci_low"], entry["gap"]["ci_high"]) if entry["gap"]["ci_low"] is not None else None
            entry["flags"] = K.flags(True, entry["bands"]["ci_low"], entry["auc"]["ci_low"], gap)
        else:
            entry["flags"] = K.flags(False, None, None, None)
        out["wards"][ward] = entry
    return out

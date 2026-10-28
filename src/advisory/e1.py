"""Phase E E1 helpers (docs/PHASE_E_PROTOCOL.md Sections 3, 4 and 8; register row D49): pure functions around the rules, used by scripts/run_phase_e1.py and checked on known values."""
from src.advisory import rules as R


def keep_mask(block_keys, train_keys):
    """Section 8: True for a Block A row whose location does not occur in the training frame."""
    seen = set(train_keys)
    return [k not in seen for k in block_keys]


def event_counts(levels, flooded, events):
    """{event: {level: [rows, flood rows]}}: enough to rebuild the rows and recompute every figure."""
    out = {}
    for lv, y, e in zip(levels, flooded, events):
        c = out.setdefault(str(e), {x: [0, 0] for x in R.LEVELS})[lv]
        c[0] += 1
        c[1] += int(y)
    return out


def rebuild(counts):
    """Inverse of event_counts: row-level lists with the same per-event, per-level counts."""
    lv, y, ev = [], [], []
    for e in sorted(counts):
        for level in R.LEVELS:
            n, k = counts[e][level]
            lv += [level] * n
            y += [1] * k + [0] * (n - k)
            ev += [e] * n
    return lv, y, ev


def summarize(levels, flooded, events):
    """Rates, the HIGH minus LOW clustered bootstrap, the gates and the outcome for the kept Block A rows."""
    rates = R.level_rates(levels, flooded)
    if rates["LOW"]["rows"] == 0 or rates["HIGH"]["rows"] == 0:
        boot, ci_low = {"diff": None, "ci_low": None, "ci_high": None, "draws_used": 0}, float("-inf")
    else:
        boot = R.cluster_bootstrap_diff(levels, flooded, events)
        ci_low = boot["ci_low"]
    g = R.gates(rates, ci_low)
    return {"rates": rates, "bootstrap": boot, "gates": g, "outcome": R.outcome(g), "per_event": event_counts(levels, flooded, events)}

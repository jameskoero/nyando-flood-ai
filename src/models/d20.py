"""D20 confirmatory replication (docs/D20_PROTOCOL.md): the date-selection rule, the sample-size rule and the decision rules as code, so none of them can depend on a model score. Standard library only, except the cluster bootstrap (numpy)."""
import random
from datetime import date

SEED = 42
MIN_GAP_DAYS = 14
MIN_VALID_PX = 997936          # 95% of the 1,050,459 pixels of the ward area
MIN_FLOOD_PX = 100
MAX_NEW_DATES = 60
WINDOW = (date(2020, 1, 1), date(2026, 8, 31))
MIN_CLASS_N = 30
MARGIN = 5
FLOOR_PER_CLASS = 40
MAX_PER_CLASS = 100


def eligible_dates(rows, existing):
    """Scan rows (date, valid_px, flood_px, error as text): dates in the window, fully covered, with enough flood pixels and at least MIN_GAP_DAYS from every existing date."""
    out = []
    for r in rows:
        d = date.fromisoformat(r["date"])
        if (r.get("error") or "").strip() or not WINDOW[0] <= d <= WINDOW[1]:
            continue
        if int(float(r["valid_px"])) >= MIN_VALID_PX and int(float(r["flood_px"])) >= MIN_FLOOD_PX and all(abs((d - e).days) >= MIN_GAP_DAYS for e in existing):
            out.append(d)
    return sorted(out)


def select_dates(eligible, existing, seed=SEED, gap=MIN_GAP_DAYS, cap=MAX_NEW_DATES):
    """Seeded greedy selection in acceptance order: shuffle the ascending list, accept a date when it is at least `gap` days from every existing and accepted date."""
    order = list(eligible)
    random.Random(seed).shuffle(order)
    taken, chosen = list(existing), []
    for d in order:
        if len(chosen) >= cap:
            break
        if all(abs((d - t).days) >= gap for t in taken):
            chosen.append(d)
            taken.append(d)
    return chosen


def split_blocks(chosen):
    """Block A (built and scored first) takes the even acceptance positions; Block B (sealed for Phase D) the odd ones."""
    return chosen[0::2], chosen[1::2]


def per_class_sample_size(retention):
    """Smallest multiple of 10, at least FLOOR_PER_CLASS, for which the 10th percentile (nearest rank) of the control retention keeps MIN_CLASS_N + MARGIN controls."""
    ordered = sorted(retention)
    q10 = ordered[-(-len(ordered) // 10) - 1]
    if q10 <= 0:
        raise ValueError("a date set kept no control outside the exclusion mask")
    n = max(FLOOR_PER_CLASS, 10 * -(-(MIN_CLASS_N + MARGIN) // (10 * q10)))
    n = int(n)
    if n > MAX_PER_CLASS:
        raise ValueError("the sample size would exceed %d per class" % MAX_PER_CLASS)
    return n


def replicates_beyond_elevation(ci_low):
    """R-A: the lower end of the paired interval, champion minus elevation only, on Block A must be above 0."""
    return bool(ci_low > 0)


def challenger_replaces(block_ci_low, block_mean_diff, pooled_ci_low, pooled_mean_diff):
    """R-B: a challenger replaces the champion only if it beats it on Block A and on all events pooled (interval lower end above 0 and a higher mean on both)."""
    return bool(block_ci_low > 0 and block_mean_diff > 0 and pooled_ci_low > 0 and pooled_mean_diff > 0)


def land_cover_survives(ci_low):
    """R-C: the land_cover contribution on Block A is reported as surviving when the lower end is above 0; leakage is never declared excluded."""
    return bool(ci_low > 0)


def month_cluster_interval(diffs, months, n_boot=10000, seed=SEED, alpha=0.025):
    """S-1: bootstrap over calendar-month clusters of per-event differences. Returns (mean, low, high); the interval is the percentile interval from alpha / 2 to 1 - alpha / 2."""
    import numpy as np
    diffs = np.asarray(diffs, dtype=float)
    months = np.asarray(months)
    groups = [np.flatnonzero(months == k) for k in sorted(set(months.tolist()))]
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        means[b] = diffs[np.concatenate([groups[i] for i in pick])].mean()
    return float(diffs.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


MIN_BLOCK_A_EVENTS = 15


def built_block_a(block_a, failed):
    """Section 11: the Block A dates that were built. A failed date is listed and never replaced, least of all from Block B."""
    unknown = set(failed) - set(block_a)
    if unknown:
        raise ValueError("failed dates that are not in Block A: %s" % sorted(unknown))
    return [d for d in block_a if d not in set(failed)]


def block_a_estimable(n_scorable_events):
    """Section 11: with fewer scorable Block A events than the floor, R-A, R-B and R-C are reported as not estimable and nothing is claimed."""
    return bool(n_scorable_events >= MIN_BLOCK_A_EVENTS)


def shared_training_rows(train_keys, block_keys):
    """Section 12.1: for each training row, True when its location also appears in Block A. Those rows are dropped from every arm's training data."""
    block = set(block_keys)
    return [k in block for k in train_keys]


ARMS = ("elevation", "champion", "no_land_cover", "hgb:con", "xgb:con")
CHALLENGERS = ("hgb:con", "xgb:con")
TOLERANCE = 0.003      # Monte Carlo tolerance of a recomputed interval end (10,000 resamples)


def to_micro(d):
    """Per-event AUCs are stored as integers in millionths, so the stored file and the analysis use identical values."""
    return {str(k): int(round(float(v) * 1e6)) for k, v in d.items()}


def from_micro(d):
    return {k: v / 1e6 for k, v in d.items()}


def month_of(event):
    """Calendar month of an event id: 'date-YYYY-MM-DD' gives YYYY-MM; ids such as '2020-04' or '2024-04_05' give their first seven characters."""
    return event[5:12] if event.startswith("date-") else event[:7]


def _stat(pb, a, b, n_boot, seed, alpha):
    s = pb(a, b, n_boot=n_boot, seed=seed, alpha=alpha)
    ev = sorted(set(a) & set(b))
    _, lo, hi = month_cluster_interval([b[e] - a[e] for e in ev], [month_of(e) for e in ev], n_boot=n_boot, seed=seed, alpha=alpha)
    s["month_ci_low"], s["month_ci_high"] = lo, hi
    return s


def analyse(old, blk, violations, n_boot=10000, seed=SEED, alpha=0.025):
    """Section 7 applied to stored per-event AUCs (mappable frame primary). old: {arm: {event: AUC}} for champion, hgb:con and xgb:con on the existing events (selection split);
    blk: {frame: {arm: {event: AUC}}} for Block A; violations: {challenger: {feature: count}}. Intervals: paired per-event bootstrap, 97.5% (alpha 0.025)."""
    from src.models.cv import paired_event_bootstrap as pb
    frames = {}
    for fr in ("mappable", "full"):
        a = blk[fr]
        frames[fr] = {"champion_vs_elevation": _stat(pb, a["elevation"], a["champion"], n_boot, seed, alpha),
                      "land_cover_contribution": _stat(pb, a["no_land_cover"], a["champion"], n_boot, seed, alpha)}
        for c in CHALLENGERS:
            frames[fr][c + "_vs_champion"] = _stat(pb, a["champion"], a[c], n_boot, seed, alpha)
    pooled, rb = {}, {}
    for c in CHALLENGERS:
        if set(old["champion"]) & set(blk["mappable"]["champion"]):
            raise ValueError("an existing event and a Block A event share an id")
        base, other = {**old["champion"], **blk["mappable"]["champion"]}, {**old[c], **blk["mappable"][c]}
        pooled[c] = _stat(pb, base, other, n_boot, seed, alpha)
        f, p = frames["mappable"][c + "_vs_champion"], pooled[c]
        ok = all(int(v) == 0 for v in violations[c].values())
        rb[c] = {"eligible": ok, "block_ci_low": f["ci_low"], "pooled_ci_low": p["ci_low"], "replaces": bool(ok and challenger_replaces(f["ci_low"], f["mean_diff"], p["ci_low"], p["mean_diff"]))}
    ra, rc, n = frames["mappable"]["champion_vs_elevation"], frames["mappable"]["land_cover_contribution"], len(blk["mappable"]["champion"])
    est = block_a_estimable(n)
    v = {"estimable": est, "n_scorable_block_a": n, "R-B": rb,
         "R-A": {"replicates": replicates_beyond_elevation(ra["ci_low"]) if est else None, "ci_low": ra["ci_low"], "month_cluster_ci_low": ra["month_ci_low"]},
         "R-C": {"survives": land_cover_survives(rc["ci_low"]) if est else None, "ci_low": rc["ci_low"], "month_cluster_ci_low": rc["month_ci_low"]},
         "champion_stays": not (est and any(x["replaces"] for x in rb.values()))}
    return {"frames": frames, "pooled": pooled, "verdicts": v}


def sensitivity(blk, artifact, n_boot=10000, seed=SEED, alpha=0.025):
    """Section 12.1 sensitivity: the registered artifact (fit on every mappable row, so it has seen the locations shared with Block A) minus elevation only, and minus the primary champion, on Block A."""
    from src.models.cv import paired_event_bootstrap as pb
    return {fr: {"artifact_vs_elevation": _stat(pb, blk[fr]["elevation"], artifact[fr], n_boot, seed, alpha),
                 "artifact_vs_primary_champion": _stat(pb, blk[fr]["champion"], artifact[fr], n_boot, seed, alpha)} for fr in ("mappable", "full")}

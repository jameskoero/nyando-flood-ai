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

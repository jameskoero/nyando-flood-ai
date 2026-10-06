"""Promotion rules for hgb:con (docs/PROMOTION_PROTOCOL.md) as code, fixed before any promotion gate is computed. Offline: no MLflow, no network."""
import numpy as np

from src.models.registration import CONSTRAINED_FEATURES
from src.models.robustness import beats

CHAMPION, CHALLENGER = "logistic:con", "hgb:con"
FIXED_PARAMS = {"max_depth": 4, "max_iter": 100, "min_samples_leaf": 20}
SUBSETS = ("no_floor_rows", "no_zero_distance", "clay_present")
FLOOR_SUBSET = "no_floor_rows"
REQUIRED_RUNS = tuple("%s/%s" % (s, part) for s in SUBSETS for part in ("new", "pooled")) + ("buffered", "temporal")
STATISTICAL_GATES = ("S1", "S2", "S3", "S4", "S5", "S6")
EXPORT_TOLERANCE = 1e-6
PROMOTED = "D20 PROMOTED"
NOT_PROMOTED = "D20 NOT PROMOTED — EXISTING CHAMPION RETAINED"


def mean_advantage(champion, challenger):
    """Mean over the common events of challenger minus champion."""
    common = sorted(set(champion) & set(challenger))
    if not common:
        raise ValueError("the two arms share no event")
    return float(np.mean([challenger[e] - champion[e] for e in common]))


def r3(pairs):
    """S1 and S2. pairs = {name: (champion_events, challenger_events)}: passes only if the challenger beats the champion in every pair."""
    if not pairs:
        raise ValueError("R3 needs at least one pair")
    detail = {k: beats(c, h)[0] for k, (c, h) in pairs.items()}
    return all(detail.values()), detail


def reverses(champion, challenger):
    """Reading A: a run reverses the win only if the champion beats the challenger there."""
    return beats(challenger, champion)[0]


def run_verdict(name, champion, challenger):
    """One sensitivity run under reading A plus the floor check; readings B and C are reported, not deciding."""
    adv = mean_advantage(champion, challenger)
    rev = reverses(champion, challenger)
    floor_fail = name.split("/")[0] == FLOOR_SUBSET and not adv > 0
    return {"mean_advantage": adv, "reverses": rev, "floor_fail": floor_fail, "unresolved": bool(rev or floor_fail),
            "reading_B_holds": beats(champion, challenger)[0], "reading_C_holds": adv > 0, "limitation": not adv > 0}


def sensitivity(runs):
    """S3 to S5. runs = {name: (champion_events, challenger_events)} naming every run in REQUIRED_RUNS."""
    missing = sorted(set(REQUIRED_RUNS) - set(runs))
    if missing:
        raise ValueError("missing sensitivity runs: %s" % missing)
    detail = {k: run_verdict(k, c, h) for k, (c, h) in runs.items()}
    return not any(d["unresolved"] for d in detail.values()), detail


def r6(elevation, challenger):
    """S6: the challenger beats elevation only."""
    return beats(elevation, challenger)[0]


def export_gates(max_abs_diff, per_event_auc_diff, violations, sha_ok):
    """E1 to E4: at most 1e-6 on probabilities and per-event AUCs, zero violations on every constrained feature, a verified hash."""
    if set(violations) != set(CONSTRAINED_FEATURES):
        raise ValueError("violations must name exactly %s" % (CONSTRAINED_FEATURES,))
    return bool(max_abs_diff <= EXPORT_TOLERANCE and per_event_auc_diff <= EXPORT_TOLERANCE
                and all(int(v) == 0 for v in violations.values()) and sha_ok)


def build_exporter(statistical):
    """The exporter is built only if every statistical gate passed."""
    if set(statistical) != set(STATISTICAL_GATES):
        raise ValueError("statistical results must name exactly %s" % (STATISTICAL_GATES,))
    return all(bool(v) for v in statistical.values())


def decision(statistical, export=None):
    """Exactly one outcome: promoted only if every statistical gate and the export gates passed."""
    return PROMOTED if build_exporter(statistical) and export is True else NOT_PROMOTED

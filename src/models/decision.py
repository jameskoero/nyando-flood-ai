"""Phase C decision rules (docs/PHASE_C_PROTOCOL.md, Section 13.4) as code, so the verdict cannot depend on the scores seen.

Inputs come from the logged runs (scripts/run_boosters.py): per arm, whether it beats the logistic (the lower end of the
97.5% paired per-event interval is above 0 and the per-event mean is higher), its per-event mean, and for the pair the
lower end of the paired interval of xgb:con minus hgb:con."""

PRIMARY = ("hgb:con", "xgb:con")


def beats_baseline(ci_low, mean, base_mean):
    return bool(ci_low > 0 and mean > base_mean)


def decide_frame(res):
    """res = {"hgb:con": {"beats": bool, "mean": float}, "xgb:con": {...}, "xgb_vs_hgb_ci_low": float}.
    Returns "hgb:con", "xgb:con" or "logistic" (rules 2 and 3 of Section 13.4)."""
    h, x = res["hgb:con"], res["xgb:con"]
    if h["beats"] and x["beats"]:
        if x["mean"] > h["mean"] and res["xgb_vs_hgb_ci_low"] > 0:
            return "xgb:con"
        return "hgb:con"
    if h["beats"]:
        return "hgb:con"
    if x["beats"]:
        return "xgb:con"
    return "logistic"


def _verdict(res):
    return (bool(res["hgb:con"]["beats"]), bool(res["xgb:con"]["beats"]))


def decide(mappable, full):
    """Rule 6: if the item-1 verdicts differ between the frames the result is unresolved; otherwise the mappable-frame winner."""
    if _verdict(mappable) != _verdict(full):
        return "unresolved"
    return decide_frame(mappable)


def claim_beyond_elevation(ci_low):
    """Rule 4: the lower end of the paired interval against elevation only must be above 0."""
    return bool(ci_low > 0)

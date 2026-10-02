"""Phase C registration rules (docs/PHASE_C_PROTOCOL.md, Section 14) as code."""
from src.models.decision import PRIMARY

CHAMPION = "logistic:con"
REGISTERED_TRAINING_FRAME = "mappable"
CONSTRAINED_FEATURES = ("rainfall_3day", "elevation", "distance_river", "slope")


def eligible(violations):
    """R1: eligible only with zero monotonicity violations for every constrained feature."""
    return set(violations) == set(CONSTRAINED_FEATURES) and all(int(v) == 0 for v in violations.values())


def qualifies(mappable, full, arm):
    """R3: a booster qualifies only if it beats the champion on both frames."""
    return bool(mappable[arm]["beats"] and full[arm]["beats"])


def registered(mappable, full):
    """R2 to R4. Each frame dict is {"hgb:con": {"beats": bool, "mean": float}, "xgb:con": {...}, "xgb_vs_hgb_ci_low": float},
    with beats measured against the champion. Returns the registered arm."""
    q = {a: qualifies(mappable, full, a) for a in PRIMARY}
    if q["hgb:con"] and q["xgb:con"]:
        x, h = mappable["xgb:con"]["mean"], mappable["hgb:con"]["mean"]
        return "xgb:con" if (x > h and mappable["xgb_vs_hgb_ci_low"] > 0) else "hgb:con"
    if q["hgb:con"]:
        return "hgb:con"
    if q["xgb:con"]:
        return "xgb:con"
    return CHAMPION

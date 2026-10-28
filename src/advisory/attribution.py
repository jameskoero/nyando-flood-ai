"""Phase E attribution by reference substitution (docs/PHASE_E_PROTOCOL.md Section 5, docs/PHASE_E_ADVISORY.md): for each feature, the change in the score when that feature alone is replaced by its reference value. The scorer is injected, so this module needs no model and no data."""

FEATURES = ("elevation", "slope", "rainfall_3day", "distance_river", "clay_percent", "land_cover")


def substitution_attribution(score_fn, values, reference, k=3, eps=1e-12):
    """The k features whose substitution changes the score most. delta > 0 means the feature raises the score compared with a reference location."""
    base = float(score_fn(values))
    out = []
    for f in FEATURES:
        swapped = dict(values)
        swapped[f] = reference[f]
        delta = base - float(score_fn(swapped))
        if abs(delta) > eps:
            out.append({"feature": f, "delta": delta, "value": values.get(f), "reference": reference[f]})
    out.sort(key=lambda d: -abs(d["delta"]))
    return out[:k]

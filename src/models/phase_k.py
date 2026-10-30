"""Phase K rules as code (docs/PHASE_K_PROTOCOL.md): the slices, the estimability rule, the descriptive flags and the outcome string. No model, network or data file is touched here."""

WARDS = ("Ahero", "Awasi/Onjiko", "East Kano/Wawidhi", "Kabonyo/Kanyagwal", "Kobura")
MIN_ROWS_PER_CLASS = 30
MIN_EVENTS_BOTH = 5
CHANCE_AUC = 0.5
OUTCOME = "PHASE K REPORTED"
FLAGS = ("not_estimable", "bands_not_separated", "auc_not_above_chance", "differs_from_pooled")


def ward_counts(flooded, events):
    """Flood rows, control rows and the number of events that contain both classes."""
    flooded = [bool(x) for x in flooded]
    events = [str(e) for e in events]
    assert len(flooded) == len(events), "flooded and events must have the same length"
    with_flood = {e for f, e in zip(flooded, events) if f}
    with_control = {e for f, e in zip(flooded, events) if not f}
    return sum(flooded), len(flooded) - sum(flooded), len(with_flood & with_control)


def estimability(flood_rows, control_rows, events_both):
    """(True, "") when a ward is estimable in a frame, else (False, the first reason)."""
    if flood_rows < MIN_ROWS_PER_CLASS:
        return False, "flood rows %d below %d" % (flood_rows, MIN_ROWS_PER_CLASS)
    if control_rows < MIN_ROWS_PER_CLASS:
        return False, "control rows %d below %d" % (control_rows, MIN_ROWS_PER_CLASS)
    if events_both < MIN_EVENTS_BOTH:
        return False, "events with both classes %d below %d" % (events_both, MIN_EVENTS_BOTH)
    return True, ""


def flags(estimable, band_ci_low, auc_ci_low, gap_ci):
    """The descriptive flags of one ward in one frame; gap_ci is (low, high) of M4."""
    if not estimable:
        return ["not_estimable"]
    out = []
    if band_ci_low is None or band_ci_low <= 0:
        out.append("bands_not_separated")
    if auc_ci_low is None or auc_ci_low <= CHANCE_AUC:
        out.append("auc_not_above_chance")
    if gap_ci is not None and (gap_ci[0] > 0 or gap_ci[1] < 0):
        out.append("differs_from_pooled")
    return out

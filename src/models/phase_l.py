"""Phase L rules as code (docs/PHASE_L_PROTOCOL.md; register rows D14 and D55): the within-class concordance, the estimability of land-cover classes, the two decision rules, the allowed claims and the counts of the training file. No model, network or result is touched here. Standard library only."""
import bisect
import csv
from pathlib import Path

from src.models import phase_k as K

OUTCOME = "PHASE L REPORTED"
ARMS = ("six", "no_land_cover", "no_clay")
D14_OUTCOMES = ("within_class_gain", "between_class_only", "within_class_loss", "not_estimable")
D55_OUTCOMES = ("value_gain", "missingness_only", "no_contribution", "other", "not_estimable")
MISSING = ("", "nan", "none", "null")
CITED = (
    ("MODEL_CARD.md", "+0.1045 [0.0853, 0.1239]"),
    ("MODEL_CARD.md", "+0.0398 [0.0144, 0.0706]"),
    ("MODEL_CARD.md", "A blank clay value is informative in the training data"),
    ("docs/PHASE_C_CLOSURE.md", "+0.0157 [-0.0028, 0.0378]"),
    ("docs/PHASE_C_CLOSURE.md", "(wins 13, losses 13)"),
)
CLAIMS = {
    "within_class_gain": "In Block A the model discriminates better with land_cover than without it between rows of the same land-cover class; this is an association in a case-control sample, not a cause.",
    "between_class_only": "In Block A land_cover does not add discrimination between rows of the same land-cover class; its contribution comes from how land-cover classes differ in the sample, a limit the Model Card states.",
    "within_class_loss": "In Block A land_cover lowers the ranking between rows of the same land-cover class; this audit leaves the registered model unchanged and the finding goes to the owner as a decision.",
    "value_gain": "Where clay is observed, its value adds discrimination in Block A.",
    "missingness_only": "The contribution of clay in Block A comes through whether it is missing, not through its observed value; the Model Card states that the model reads a missing clay value as a signal.",
    "no_contribution": "No contribution of clay is shown in Block A, with or without its missing values.",
    "other": "The two clay contrasts do not fit one summary; both are reported as measured and no summary claim is made.",
    "not_estimable": "No claim is possible: too little of Block A meets the estimability rule.",
}
LIMITS = "The audit models are not registered, promoted or served; no causal reading, no real flood prevalence and no lead-time or early-warning claim follows."
BEGIN = "<!-- phase-l:counts:begin -->"
END = "<!-- phase-l:counts:end -->"


def class_strata(classes, flooded, events):
    """Per land-cover class: flood rows, control rows, events with both classes, and whether the class enters W (the Phase K rule, unchanged)."""
    out = {}
    for c in sorted(set(classes)):
        idx = [i for i, x in enumerate(classes) if x == c]
        fl, ct, both = K.ward_counts([flooded[i] for i in idx], [events[i] for i in idx])
        ok, why = K.estimability(fl, ct, both)
        out[c] = {"flood_rows": fl, "control_rows": ct, "events_both": both, "estimable": ok, "reason": why}
    return out


def within_class_concordance(scores, flooded, classes, use=None):
    """W: the share of (flood row, control row) pairs from the same land-cover class in which the flood row scores higher; a tie counts one half. use limits the classes. Returns (W, pairs), or (None, 0) when there is no pair."""
    num, pairs = 0.0, 0
    for c in sorted(set(classes)):
        if use is not None and c not in use:
            continue
        flood = [s for s, f, k in zip(scores, flooded, classes) if k == c and f]
        ctrl = sorted(s for s, f, k in zip(scores, flooded, classes) if k == c and not f)
        for s in flood:
            lo, hi = bisect.bisect_left(ctrl, s), bisect.bisect_right(ctrl, s)
            num += lo + 0.5 * (hi - lo)
        pairs += len(flood) * len(ctrl)
    return (num / pairs, pairs) if pairs else (None, 0)


def d14_outcome(ci_low, ci_high):
    """The D14 outcome from the interval of delta W (registered arm minus the arm without land_cover)."""
    if ci_low is None or ci_high is None:
        return "not_estimable"
    if ci_low > 0:
        return "within_class_gain"
    if ci_high < 0:
        return "within_class_loss"
    return "between_class_only"


def d55_outcome(all_ci, present_ci):
    """The D55 outcome from the intervals (low, high) of delta T on all rows and on the rows where clay is observed."""
    if all_ci is None or present_ci is None:
        return "not_estimable"
    if present_ci[0] > 0:
        return "value_gain"
    present_zero = present_ci[0] <= 0 <= present_ci[1]
    all_zero = all_ci[0] <= 0 <= all_ci[1]
    if all_ci[0] > 0 and present_zero:
        return "missingness_only"
    if all_zero and present_zero:
        return "no_contribution"
    return "other"


def _land(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def count_tables(path):
    """Row counts of the training file by land-cover class and by whether clay is blank, split into control and flood rows."""
    land, clay, rows = {}, {"blank": [0, 0], "observed": [0, 0]}, [0, 0]
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            flood = 1 if r["flooded"] in ("1", "1.0") else 0
            rows[flood] += 1
            land.setdefault(_land(r["land_cover"]), [0, 0])[flood] += 1
            clay["blank" if str(r["clay_percent"]).strip().lower() in MISSING else "observed"][flood] += 1
    return {"land_cover": land, "clay": clay, "rows": rows}


def render_counts(t):
    n = lambda x: "{:,}".format(x)
    ctrl, flood = t["rows"]
    lines = ["Training file `data/training/nyando_training_v2_multidate.csv`: %s rows (%s flood rows, %s control rows)." % (n(ctrl + flood), n(flood), n(ctrl)), "",
             "| land_cover class | Control rows | Flood rows |", "|---|---|---|"]
    for c in sorted(t["land_cover"], key=lambda k: (k is None, k)):
        a, b = t["land_cover"][c]
        lines.append("| %s | %s | %s |" % ("not a class" if c is None else c, n(a), n(b)))
    lines += ["", "| clay_percent | Control rows | Flood rows |", "|---|---|---|"]
    for k in ("blank", "observed"):
        a, b = t["clay"][k]
        lines.append("| %s | %s | %s |" % (k, n(a), n(b)))
    return "\n".join(lines)

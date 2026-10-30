"""Phase K report text (docs/PHASE_K_PROTOCOL.md Section 8): the Model Card and Datasheet sections are generated from docs/PHASE_K_RESULTS.json, so no number in them is typed by hand. Standard library only."""
from src.models import phase_k as K

BEGIN = "<!-- phase-k:begin -->"
END = "<!-- phase-k:end -->"
FLAG_TEXT = {
    "not_estimable": "not estimable in this frame, so no claim is possible",
    "bands_not_separated": "the interval of HIGH minus LOW reaches 0 or below",
    "auc_not_above_chance": "the lower end of the mean per-event AUC interval is at or below 0.5",
    "differs_from_pooled": "the interval of the ward's AUC minus the pooled AUC excludes 0",
}


def _f(v, spec="%.3f"):
    return "n/a" if v is None else spec % v


def _iv(lo, hi, spec="%.3f"):
    return "[%s, %s]" % (_f(lo, spec), _f(hi, spec))


def _flags(e):
    return ", ".join("`%s`" % x for x in e["flags"]) or "none"


def _table(rep):
    out = ["| Ward | Flood rows | Control rows | Events with both classes | HIGH minus LOW | Mean per-event AUC | AUC minus pooled | Flags |", "|---|---|---|---|---|---|---|---|"]
    for ward, e in rep["wards"].items():
        head = "| %s | %d | %d | %d |" % (ward, e["flood_rows"], e["control_rows"], e["events_both"])
        if not e["estimable"]:
            out.append(head + " not estimable | not estimable | not estimable | %s |" % _flags(e))
            continue
        b, a, g = e["bands"], e["auc"], e["gap"]
        out.append(head + " %s %s | %s %s | %s %s | %s |" % (_f(b["diff"], "%+.3f"), _iv(b["ci_low"], b["ci_high"], "%+.3f"), _f(a["mean"]), _iv(a["ci_low"], a["ci_high"]),
                                                             _f(g["gap"], "%+.3f"), _iv(g["ci_low"], g["ci_high"], "%+.3f"), _flags(e)))
    return out


def _statements(rep):
    out = []
    for ward, e in rep["wards"].items():
        if not e["estimable"]:
            out.append("- %s: no claim possible (%s; the rule needs %d flood rows, %d control rows and %d events with both classes)." % (ward, e["reason"], K.MIN_ROWS_PER_CLASS, K.MIN_ROWS_PER_CLASS, K.MIN_EVENTS_BOTH))
        elif not e["flags"]:
            out.append("- %s: no flag raised." % ward)
        else:
            parts = []
            for fl in e["flags"]:
                t = FLAG_TEXT[fl]
                if fl == "differs_from_pooled":
                    t += "; %s the pooled value" % ("below" if e["gap"]["ci_high"] < 0 else "above")
                parts.append("`%s` (%s)" % (fl, t))
            out.append("- %s: %s." % (ward, "; ".join(parts)))
    return out


def render_card(res):
    p, s, e1, ml = res["frames"]["primary"], res["frames"]["secondary"], res["e1"], res.get("mlflow") or {}
    lines = [
        "### Ward-level slices (Phase K, register rows D52 and D53)", "",
        "Protocol `docs/PHASE_K_PROTOCOL.md` (SHA-256 `%s`); results `docs/PHASE_K_RESULTS.json` (K1 run: GitHub Actions run %s, MLflow run `%s`, commit `%s`). The scores are ranking scores, not flood probabilities, and the result is not an official warning. The training and Block A samples are case-control, so a ward's share of flood rows describes the sampling and is not a flood rate."
        % (res["protocol_sha256"], res.get("github_run_id"), ml.get("run_id"), res["commit"]), "",
        "Intervals are event-clustered percentile bootstrap intervals: the 2.5th to 97.5th percentile of 10,000 resamples of events with seed 42, a two-sided 95% interval. Earlier sections of this card and earlier protocols call the same interval a 97.5% interval; the code takes the 2.5th and 97.5th percentiles (register row D53).", "",
        "**Primary frame** (the %d Block A rows kept after excluding rows at training locations, scored by the registered artifact; this frame decides what the card may say):" % p["rows"], "",
        *_table(p), "", *_statements(p), "",
        "**Secondary frame** (the %d mappable training rows, out-of-fold scores; context only):" % s["rows"], "",
        *_table(s), "", *_statements(s), "",
        "What this does not show: no causal reading, no real flood prevalence, nothing outside the GFM-mappable areas, and no test of lead time, so no early-warning claim follows. The DEM-floor rows of Kabonyo/Kanyagwal and the land-cover question (register row D14) are not separated here. The reference check of K1 reads `reference_reproduces: %s` (mean event AUC %.3f), as in E1; the cutpoints equal E1's, so no band changed."
        % (e1["reference_reproduces"], e1["reference_mean_event_auc"]),
    ]
    return "\n".join(lines)


def render_datasheet(res):
    p, s = res["frames"]["primary"], res["frames"]["secondary"]
    lines = ["### 6.1 Ward-level coverage (Phase K)", "",
             "Counts by ward in the two frames of `docs/PHASE_K_RESULTS.json`. They are counts only: the samples are case-control, so they are not flood rates.", "",
             "| Ward | Primary: flood / control rows | Primary: events with both classes | Secondary: flood / control rows | Secondary: events with both classes | Estimable (primary / secondary) |", "|---|---|---|---|---|---|"]
    for ward in p["wards"]:
        a, b = p["wards"][ward], s["wards"][ward]
        lines.append("| %s | %d / %d | %d | %d / %d | %d | %s / %s |" % (ward, a["flood_rows"], a["control_rows"], a["events_both"], b["flood_rows"], b["control_rows"], b["events_both"],
                                                                      "yes" if a["estimable"] else "no", "yes" if b["estimable"] else "no"))
    gone = [w for w in p["wards"] if not p["wards"][w]["estimable"] and not s["wards"][w]["estimable"]]
    lines += ["", ("%s: not estimable in either frame, so no per-ward statement is made for it. " % ", ".join(gone) if gone else "") +
              "Metrics and flags are in MODEL_CARD.md, section \"Ward-level slices (Phase K, register rows D52 and D53)\"."]
    return "\n".join(lines)

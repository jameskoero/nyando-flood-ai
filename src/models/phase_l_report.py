"""Phase L report text (docs/PHASE_L_PROTOCOL.md Section 7): the Model Card and Datasheet sections are generated from docs/PHASE_L_RESULTS.json, so no number in them is typed by hand. Standard library only."""
from src.models import phase_l as L

BEGIN = "<!-- phase-l:begin -->"
END = "<!-- phase-l:end -->"


def _f(v, spec="%+.4f"):
    return "n/a" if v is None else spec % v


def _iv(lo, hi, spec="%+.4f"):
    return "[%s, %s]" % (_f(lo, spec), _f(hi, spec))


def _classes(frame):
    return sorted(frame["classes"], key=int)


def _table(f):
    dw, dt = f["delta_w_land_cover"], f["delta_t_land_cover"]
    used = ", ".join(str(c) for c in f["classes_used"]) or "none"
    out = ["| Contrast (registered model minus audit arm) | Estimate | Interval |", "|---|---|---|",
           "| W, land_cover: within-class concordance (classes %s of %d) | %s | %s |" % (used, len(f["classes"]), _f(dw["delta"]) if dw else "not estimable", _iv(dw["ci_low"], dw["ci_high"]) if dw else "not estimable"),
           "| T, land_cover: mean per-event AUC | %s | %s |" % (_f(dt["mean_diff"]), _iv(dt["ci_low"], dt["ci_high"]))]
    c = f["clay"]
    if c["observed_estimable"]:
        a, p = f["delta_t_clay_all"], f["delta_t_clay_present"]
        out += ["| T, clay_percent: all rows | %s | %s |" % (_f(a["mean_diff"]), _iv(a["ci_low"], a["ci_high"])),
                "| T, clay_percent: only rows where clay is observed (%d blank rows left out) | %s | %s |" % (c["rows_blank"], _f(p["mean_diff"]), _iv(p["ci_low"], p["ci_high"]))]
    else:
        out.append("| T, clay_percent | not estimable (%s) | not estimable |" % c["reason"])
    return out


def _class_table(f):
    out = ["| land_cover class | Flood rows | Control rows | Events with both classes | Enters W |", "|---|---|---|---|---|"]
    for k in _classes(f):
        e = f["classes"][k]
        out.append("| %s | %d | %d | %d | %s |" % (k, e["flood_rows"], e["control_rows"], e["events_both"], "yes" if e["estimable"] else "no (%s)" % e["reason"]))
    return out


def render_card(res):
    p, s, ml = res["frames"]["primary"], res["frames"]["secondary"], res.get("mlflow") or {}
    same = (p["d14_outcome"], p["d55_outcome"]) == (s["d14_outcome"], s["d55_outcome"])
    lines = [
        "### Leakage audit of land_cover and clay_percent (Phase L, register rows D14 and D55)", "",
        "Protocol `docs/PHASE_L_PROTOCOL.md` (SHA-256 `%s`); results `docs/PHASE_L_RESULTS.json` (L1 run: GitHub Actions run %s, MLflow run `%s`, commit `%s`). The two audit arms are the registered model trained without land_cover and without clay_percent, with the registered fixed parameters, on the mappable training rows; they are not registered, promoted or served, and the registered model is unchanged. The scores are ranking scores, not flood probabilities."
        % (res["protocol_sha256"], res.get("github_run_id"), ml.get("run_id"), res["commit"]), "",
        "Intervals are event-clustered percentile bootstrap intervals: the 2.5th to 97.5th percentile of 10,000 resamples of events with seed 42, a two-sided 95% interval. Every contrast is the registered model minus the audit arm, so a positive value means the feature helps. W is the share of flood and control row pairs of the same land-cover class in which the flood row scores higher (ties count one half); T is the mean per-event AUC.", "",
        "**Primary frame** (the %d Block A rows kept after excluding rows at training locations, scored by the registered artifact; this frame decides what the card may say):" % p["rows"], "",
        *_table(p), "",
        "- D14 (land_cover): `%s`. %s" % (p["d14_outcome"], L.CLAIMS[p["d14_outcome"]]),
        "- D55 (clay_percent): `%s`. %s" % (p["d55_outcome"], L.CLAIMS[p["d55_outcome"]]), "",
        "Only the land-cover classes marked yes below enter W, so the D14 statement covers those classes and no others.", "",
        *_class_table(p), "",
        "**Secondary frame** (the %d mappable training rows, out-of-fold scores; context only):" % s["rows"], "",
        *_table(s), "",
        "The secondary frame reaches the outcomes `%s` (D14) and `%s` (D55)%s." % (s["d14_outcome"], s["d55_outcome"], ", the same as the primary frame" if same else ", which differ from the primary frame; the claim rests on the primary frame"), "",
        "What this does not show: " + L.LIMITS[0].lower() + L.LIMITS[1:] + " The registered arm reproduced the registered artifact on Block A to within %.0e (largest difference %.1e), and the reference check equals the K1 record." % (res["checks"]["parity_tolerance"], res["checks"]["registered_parity_max_abs_diff"]),
    ]
    return "\n".join(lines)


def render_datasheet(res):
    p, s = res["frames"]["primary"], res["frames"]["secondary"]
    lines = ["### 6.2 Land-cover class and clay coverage (Phase L)", "",
             "Counts of the two frames of `docs/PHASE_L_RESULTS.json`. They are counts only: the samples are case-control, so they are not flood rates.", "",
             "| land_cover class | Primary: flood / control rows | Primary: events with both classes | Secondary: flood / control rows | Secondary: events with both classes | Enters W (primary / secondary) |", "|---|---|---|---|---|---|"]
    for k in sorted(set(p["classes"]) | set(s["classes"]), key=int):
        a, b = p["classes"].get(k), s["classes"].get(k)
        cell = lambda e: ("%d / %d | %d" % (e["flood_rows"], e["control_rows"], e["events_both"])) if e else "0 / 0 | 0"
        yn = lambda e: ("yes" if e["estimable"] else "no") if e else "no"
        lines.append("| %s | %s | %s | %s / %s |" % (k, cell(a), cell(b), yn(a), yn(b)))
    lines += ["", "| clay_percent | Primary rows | Secondary rows |", "|---|---|---|",
              "| blank | %d | %d |" % (p["clay"]["rows_blank"], s["clay"]["rows_blank"]), "| observed | %d | %d |" % (p["clay"]["rows_observed"], s["clay"]["rows_observed"]), "",
              "Metrics and outcomes are in MODEL_CARD.md, section \"Leakage audit of land_cover and clay_percent (Phase L, register rows D14 and D55)\"."]
    return "\n".join(lines)

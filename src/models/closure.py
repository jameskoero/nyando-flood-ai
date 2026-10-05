"""Phase C closure (docs/PHASE_C_CLOSURE.md): the readings of protocol Section 16 applied to the stored results, and the closure record
rendered from them. Pure functions of the results dictionary (docs/PHASE_C_RESULTS.json): no network, no MLflow."""
from src.models.cv import FEATURE_SETS
from src.models.registration import CHAMPION, CONSTRAINED_FEATURES, registered
from src.models.robustness import BUFFER_M, CUTOFF_YEAR, MIN_TEST_EVENTS, N_REPEATS

SIX = list(FEATURE_SETS["six"])
SUBSETS = ("no_floor_rows", "no_zero_distance", "clay_present")
SUBSET_TEXT = {"no_floor_rows": "without the 549 elevation-floor rows", "no_zero_distance": "without the 15 zero-distance rows",
               "clay_present": "rows where clay_percent is present"}
SHUFFLE_NULL = {"full": (0.4349, 0.4830), "mappable": (0.4203, 0.4874)}
STAT_KEYS = ("mean_diff", "ci_low", "ci_high", "wins", "losses")


def metric(results, run, key):
    return float(results["runs"][run]["metrics"][key])


def stats(results, run, prefix):
    return {k: metric(results, run, "%s/%s" % (prefix, k)) for k in STAT_KEYS}


def relied_upon(ci_low):
    return bool(ci_low > 0)


def inside_null(frame, pooled):
    lo, hi = SHUFFLE_NULL[frame]
    return bool(lo <= pooled <= hi)


def verdict(reversals):
    return "unresolved" if any(reversals.values()) else "stands"


def frame_result(results, frame):
    """The per-frame input of src/models/registration.py, from the stored values."""
    out = {}
    for arm in ("hgb", "xgb"):
        s = stats(results, "registration", "%s/%s_vs_con" % (frame, arm))
        out[arm + ":con"] = {"beats": bool(s["ci_low"] > 0 and s["mean_diff"] > 0),
                             "mean": metric(results, "boosters_" + frame, "%s/%s_con/per_event_mean" % (frame, arm))}
    out["xgb_vs_hgb_ci_low"] = metric(results, "boosters_" + frame, "%s/xgb_con_vs_hgb_con/ci_low" % frame)
    return out


def registered_arm(results):
    return registered(frame_result(results, "mappable"), frame_result(results, "full"))


def iv(s):
    return "%+.4f [%.4f, %.4f] (wins %d, losses %d)" % (s["mean_diff"], s["ci_low"], s["ci_high"], s["wins"], s["losses"])


def readings(results):
    R = "robustness"
    out = {"prior": {}, "perm": {}, "subsets": {}, "registration": {}}
    for fr in SHUFFLE_NULL:
        pooled = metric(results, R, "prior_only/%s/pooled_auc" % fr)
        out["prior"][fr] = {"pooled": pooled, "per_event_mean": metric(results, R, "prior_only/%s/per_event_mean" % fr), "inside_null": inside_null(fr, pooled)}
    out["perm_base"] = {"pooled": metric(results, R, "perm/base_pooled_auc"), "per_event_mean": metric(results, R, "perm/base_per_event_mean")}
    for f in SIX:
        s = stats(results, R, "perm/%s/within_drop" % f)
        out["perm"][f] = {**s, "relied": relied_upon(s["ci_low"]), "pooled_drop": metric(results, R, "perm/%s/pooled_drop_mean" % f),
                          "pooled_sd": metric(results, R, "perm/%s/pooled_drop_sd" % f)}
    n = int(metric(results, R, "temporal/scorable_test_events"))
    t = {"scorable": n, "train_rows": int(metric(results, R, "temporal/train_rows")), "test_rows": int(metric(results, R, "temporal/test_rows")),
         "straddling": int(metric(results, R, "temporal/straddling_events")), "estimable": n >= MIN_TEST_EVENTS}
    if t["estimable"]:
        t["vs_elevation"] = stats(results, R, "temporal/con_minus_elevation")
        t["beyond_elevation"] = relied_upon(t["vs_elevation"]["ci_low"])
        t["land_cover"] = stats(results, R, "temporal/land_cover_contribution")
        t["land_cover_survives"] = relied_upon(t["land_cover"]["ci_low"])
        for arm in ("con_six", "con_no_land_cover", "elevation_only"):
            t[arm] = metric(results, R, "temporal/%s/per_event_mean" % arm)
    out["temporal"] = t
    b = {"con": metric(results, R, "buffer/con_per_event_mean"), "min_train_rows": int(metric(results, R, "buffer/min_train_rows")),
         "vs_selection": stats(results, R, "buffer/con_minus_selection_split"), "vs_elevation": stats(results, R, "buffer/con_minus_elevation"),
         "vs_unconstrained": stats(results, R, "buffer/con_minus_unconstrained")}
    b["beyond_elevation"] = relied_upon(b["vs_elevation"]["ci_low"])
    out["buffer"] = b
    for s in SUBSETS:
        p = "sens/%s/" % s
        out["subsets"][s] = {"rows": int(metric(results, R, p + "rows")), "events": int(metric(results, R, p + "scorable_events")),
                             "con": metric(results, R, p + "con_per_event_mean"), "vs_elevation": stats(results, R, p + "con_minus_elevation"),
                             "vs_unconstrained": stats(results, R, p + "con_minus_unconstrained"),
                             "hgb": stats(results, R, p + "hgb_con_minus_con"), "xgb": stats(results, R, p + "xgb_con_minus_con"),
                             "reversal": bool(metric(results, R, p + "reversal"))}
    for fr in ("mappable", "full"):
        g = "registration"
        out["registration"][fr] = {"con": metric(results, g, "%s/con/per_event_mean" % fr), "vs_unconstrained": stats(results, g, "%s/con/vs_unconstrained" % fr),
                                   "vs_elevation": stats(results, g, "%s/con/vs_elevation" % fr), "hgb": stats(results, g, "%s/hgb_vs_con" % fr),
                                   "xgb": stats(results, g, "%s/xgb_vs_con" % fr)}
    out["violations"] = {f: int(metric(results, "registration", "violations/con/%s" % f)) for f in CONSTRAINED_FEATURES}
    return out


def render_outcome(results, r, arm):
    model, runs = results["registered_model"], results["runs"]
    reg, rob, rb = runs["registration"], runs["robustness"], results["artifact_readback"]
    won = arm != CHAMPION
    rev = [s for s in SUBSETS if r["subsets"][s]["reversal"]]
    t, v = r["temporal"], r["violations"]
    L = ["# Phase C closure record", "",
         "Date: %s. Rules: [PHASE_C_PROTOCOL.md](PHASE_C_PROTOCOL.md), Sections 1 to 16. Values: [PHASE_C_RESULTS.json](PHASE_C_RESULTS.json), read from MLflow by "
         "`scripts/build_phase_c_results.py`. This record is rendered from those values by `src/models/closure.py`, and `tests/test_phase_c_results.py` fails if the two differ." % results["date"],
         "", "## Outcome", "",
         "- Registered model: `%s`, a sign-constrained logistic regression (`logistic:con`) trained on the mappable frame; SHA-256 `%s`." % (model["file"], model["sha256"])]
    if won:
        L.append("- Exit criterion \"the winner beats the baseline\": **met** by `%s` (rules R1 to R4, protocol Section 14.3)." % arm)
    else:
        L.append("- Exit criterion \"the winner beats the baseline\": **not met**. No booster beat `logistic:con` on both frames (rules R1 to R4, protocol Section 14.3), so the registered "
                 "model is the simplest candidate that satisfies the declared monotonic constraints. No booster is described as better.")
    if rev:
        L.append("- Robustness (protocol Section 16.4): the verdict is **unresolved**. A booster beats `logistic:con` in %d of 3 sensitivity subsets (%s); the registered model is unchanged." % (len(rev), ", ".join(SUBSET_TEXT[s] for s in rev)))
    else:
        L.append("- Robustness (protocol Section 16.4): the verdict **stands** in all three sensitivity subsets; no booster beats `logistic:con` in any of them.")
    L.append("- Claim limit: scores rank locations inside the areas GFM can map. They are not flood probabilities (the sample is case-control) and say nothing about locations inside the GFM exclusion mask or about floods GFM cannot detect.")
    L.append("- Value beyond elevation only (protocol Section 13.4 item 4, lower end of the paired interval above 0): selection split %s; %s m buffered split %s; out of time (events after %d) %s." % (
        "holds" if relied_upon(r["registration"]["mappable"]["vs_elevation"]["ci_low"]) else "not demonstrated", format(int(BUFFER_M), ","),
        "holds" if r["buffer"]["beyond_elevation"] else "not demonstrated", CUTOFF_YEAR,
        ("holds" if t["beyond_elevation"] else "not demonstrated") if t["estimable"] else "not estimable"))
    L += ["", "## Exit criteria (protocol Section 6)", "", "| Criterion | State | Evidence |", "|---|---|---|",
          "| The winner beats the baseline | %s | rules R1 to R4 applied to the logged runs: registered arm `%s` |" % ("Met" if won else "Not met", arm),
          "| The A-Gate and the monotonicity test pass in CI | Met | required checks `data-gate` and `test` on main; `tests/test_monotonicity.py`, `tests/test_onnx_export.py`, `tests/test_registered_model.py`; "
          "violations of the registered model (rows of 200): rainfall_3day %d, elevation %d, distance_river %d, slope %d |" % (v["rainfall_3day"], v["elevation"], v["distance_river"], v["slope"]),
          "| The model-load check passes under the production pins | Met | the CI job `test` installs `requirements.txt` (numpy 1.26.4, scikit-learn 1.6.1) and `requirements-onnx.txt` on Python 3.11 and loads the committed artifact with onnxruntime in `tests/test_registered_model.py` |",
          "| Every reported number is traceable to an MLflow run with a training-data hash | Met | the runs below, all with training-data SHA-256 `%s` |" % reg["training_data_sha256"],
          "| The winner is entered in `models/MANIFEST.json` | Met | `%s`, name rule `nyando_<algorithm>_<hash prefix>` |" % model["file"],
          "| Permutation audit on the six features (a Phase A item) | Done | run `%s`; reliance is not legitimacy, see below |" % rob["id"],
          "| The registered artifact is logged to the registration run (register row D12) | %s | %s |" % ("Done" if rb["present"] else "Open",
              "MLflow artifact listing of run `%s` shows the file" % rb["run"] if rb["present"] else "the upload or the read-back failed; the artifact is committed and hashed in `models/MANIFEST.json`"), ""]
    return L


def render_battery(results, r):
    runs = results["runs"]
    reg, rob = runs["registration"], runs["robustness"]
    t, b = r["temporal"], r["perm_base"]
    L = ["## Registration (MLflow run `%s`, commit `%s`)" % (reg["id"], reg["commit"]), "",
         "Per-event mean AUC under the selection split (leave one event out, shared locations removed). Intervals: paired per-event bootstrap, 97.5%, 10,000 resamples of events, seed 42.", "",
         "| Frame | logistic:con | vs unconstrained logistic | vs elevation only | hgb:con minus logistic:con | xgb:con minus logistic:con |", "|---|---|---|---|---|---|"]
    for fr in ("mappable", "full"):
        x = r["registration"][fr]
        L.append("| %s | %.4f | %s | %s | %s | %s |" % (fr, x["con"], iv(x["vs_unconstrained"]), iv(x["vs_elevation"]), iv(x["hgb"]), iv(x["xgb"])))
    L += ["", "## Robustness battery (MLflow run `%s`, commit `%s`)" % (rob["id"], rob["commit"]), "", "### Prior-only null (16.1)", ""]
    for fr in ("mappable", "full"):
        p = r["prior"][fr]
        lo, hi = SHUFFLE_NULL[fr]
        L.append("- %s frame: pooled AUC %.4f, per-event mean %.4f. The pooled shuffle null interval is %.4f to %.4f, so the mechanism (the training prevalence falls when a flood-heavy event is held out) is **%s**." % (
            fr, p["pooled"], p["per_event_mean"], lo, hi, "supported" if p["inside_null"] else "not supported"))
    L += ["", "### Permutation audit of the registered model (16.2)", "",
          "Unpermuted: pooled AUC %.4f, per-event mean %.4f. %d repeats per feature." % (b["pooled"], b["per_event_mean"], N_REPEATS), "",
          "| Feature | Within-event drop (97.5% interval) | Relied upon | Pooled drop (sd over repeats) |", "|---|---|---|---|"]
    for f in SIX:
        x = r["perm"][f]
        L.append("| %s | %s | %s | %+.4f (%.4f) |" % (f, iv(x), "yes" if x["relied"] else "no", x["pooled_drop"], x["pooled_sd"]))
    L += ["", "The slope drop is 0 because the registered model's slope coefficient is held at 0 by its sign bound. A feature that does not vary inside an event would also have a within-event drop of 0 by construction, so its pooled drop is the column to read. Reliance is not legitimacy: this audit does not show that any feature is free of leaked signal.",
          "", "### Temporal holdout (16.3)", "",
          "Trained on events dated %d or earlier (%d rows), scored on later events (%d rows, %d scorable test events; %d events straddle the cutoff)." % (CUTOFF_YEAR, t["train_rows"], t["test_rows"], t["scorable"], t["straddling"]), ""]
    if t["estimable"]:
        L += ["- Per-event mean: logistic:con %.4f; without land_cover %.4f; elevation only %.4f." % (t["con_six"], t["con_no_land_cover"], t["elevation_only"]),
              "- Six features against elevation only: %s. The value beyond elevation %s out of time." % (iv(t["vs_elevation"]), "holds" if t["beyond_elevation"] else "is not demonstrated"),
              "- Six features against the arm without land_cover: %s. The land_cover contribution %s out of time." % (iv(t["land_cover"]), "survives" if t["land_cover_survives"] else "is not demonstrated"),
              "- Neither reading shows that land_cover is free of leaked signal; register row D14 stays open."]
    else:
        L.append("Not estimable: fewer than %d scorable test events." % MIN_TEST_EVENTS)
    return L


def render_rest(results, r):
    runs, d = results["runs"], r["buffer"]
    L = ["", "### Sensitivity subsets (16.4)", "", "| Subset | Rows | Events | logistic:con | vs elevation only | hgb:con minus con | xgb:con minus con | Reversal |", "|---|---|---|---|---|---|---|---|"]
    for s in SUBSETS:
        x = r["subsets"][s]
        L.append("| %s | %d | %d | %.4f | %s | %s | %s | %s |" % (SUBSET_TEXT[s], x["rows"], x["events"], x["con"], iv(x["vs_elevation"]), iv(x["hgb"]), iv(x["xgb"]), "yes" if x["reversal"] else "no"))
    L += ["", "### Buffered-neighbour split (16.5, %s m)" % format(int(BUFFER_M), ","), "",
          "- logistic:con per-event mean %.4f (minimum training rows in a fold: %d); against its own selection-split result: %s." % (d["con"], d["min_train_rows"], iv(d["vs_selection"])),
          "- Against elevation only: %s. The value beyond elevation %s under the buffered split." % (iv(d["vs_elevation"]), "holds" if d["beyond_elevation"] else "is not demonstrated"),
          "- Against the unconstrained logistic: %s." % iv(d["vs_unconstrained"]),
          "", "## Limitations that stay", "",
          "- The labels come from GFM: 945 of the 2,450 controls (38.6%) lie inside its exclusion mask, so \"control\" means \"not detected as flooded\" (protocol Section 12).",
          "- 23 scorable events give paired intervals about 0.015 wide on each side; an interval that includes 0 means \"not shown\", not \"equal\".",
          "- The boosters' tuning chose the edge of the committed grids; the grids were not widened after seeing scores (protocol Section 13).",
          "- The cause of the land_cover contribution is not established (register row D14).",
          "- The slope coefficient of the registered model is held at 0 by its bound, so the model ignores slope.",
          "", "## Open items and owners", "",
          "- Serving the registered model (D9): a deploy pull request with onnxruntime in the image, a SHA-256 check before loading and a new input contract.",
          "- The OSM check of the 15 zero-distance rows (D11) and the train-check, drift-monitor and deploy workflows (D13).",
          "- Confirmatory replication on new GFM scene dates, champion against challengers (D20).",
          "- Dependency pinning in `ci.yml`, a `permissions:` block and SHA-pinned actions (hardening pull request).",
          "", "## Reproducibility", "", "| Run | MLflow run ID | Commit |", "|---|---|---|"]
    for key, label in (("registration", "logistic:con scoring"), ("robustness", "robustness battery"), ("boosters_mappable", "boosters, mappable frame"), ("boosters_full", "boosters, full frame")):
        L.append("| %s | `%s` | `%s` |" % (label, runs[key]["id"], runs[key]["commit"]))
    L += ["", "Regenerate the results and this record with `python scripts/build_phase_c_results.py` (needs a DagsHub token)."]
    return L


def render(results):
    r, arm = readings(results), registered_arm(results)
    return "\n".join(render_outcome(results, r, arm) + render_battery(results, r) + render_rest(results, r)) + "\n"

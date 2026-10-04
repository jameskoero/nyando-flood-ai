"""Run the Phase C robustness battery (protocol Section 16) and log it to MLflow in one run.

In Colab, from a clean checkout of main: run this file with no arguments. The DagsHub token is read through a hidden prompt."""
import getpass
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.baseline import build_logistic
from src.models.boosters import make_estimator, nested_scores, space_for
from src.models.constrained import build_constrained_logistic
from src.models.cv import (EVENT, FEATURE_SETS, LABEL, SEED, load_layer_flags, load_training_frame, loeo_splits, out_of_fold_scores,
                           per_group_auc, pooled_auc, restrict_to_mappable)
from src.models.robustness import (BUFFER_M, CUTOFF_YEAR, MIN_TEST_EVENTS, N_REPEATS, buffered_splits, paired, permutation_audit,
                                   prior_only_scores, reversal, subset_masks, temporal_holdout)
from src.tracking import DEFAULT_TRACKING_URI, tracked_run

REGISTRATION_RUN = "54afe7e812fd4c4985280283d52f3793"
STAT_KEYS = ("mean_diff", "ci_low", "ci_high", "wins", "losses")


def events_for(builder, frame, feats, splits=None):
    """Per-event AUCs of builder(feats) under the selection split, or under the given splits."""
    splits = splits if splits is not None else loeo_splits(frame, drop_shared_locations=True)
    return per_group_auc(frame, out_of_fold_scores(builder(feats), frame, feats, splits), EVENT, 1)


def put_stats(prefix, p):
    """The paired statistics of p as MLflow metric entries under prefix."""
    return {"%s/%s" % (prefix, k): float(p[k]) for k in STAT_KEYS}


def main():
    t0 = time.time()
    df, _ = load_training_frame()
    mp = restrict_to_mappable(df, load_layer_flags(df))
    frames = {"full": df, "mappable": mp}
    six, no_lc = FEATURE_SETS["six"], FEATURE_SETS["six_no_land_cover"]
    os.environ.setdefault("MLFLOW_TRACKING_USERNAME", "jmskoero")
    if not os.environ.get("MLFLOW_TRACKING_PASSWORD"):
        os.environ["MLFLOW_TRACKING_PASSWORD"] = getpass.getpass("DagsHub token (hidden): ")
    params = {"frame": "mappable (the prior-only null also on full)", "buffer_m": str(BUFFER_M), "cutoff_year": str(CUTOFF_YEAR),
              "n_repeats": str(N_REPEATS), "seed": str(SEED), "registration_run": REGISTRATION_RUN,
              "protocol": "docs/PHASE_C_PROTOCOL.md Section 16"}
    tags = {"phase": "C", "arm": "robustness", "model_status": "candidate-not-a-product"}
    reversals = {}

    def say(text):
        print("[%4.0fs] %s" % (time.time() - t0, text), flush=True)

    with tracked_run("phase-c-robustness", "logistic:con", params, tags=tags) as (mlflow, run):
        def put(m):
            bad = [k for k, v in m.items() if not np.isfinite(v)]
            assert not bad, "non-finite metric: %s" % bad
            mlflow.log_metrics(m)

        for name, fr in frames.items():
            sc = prior_only_scores(fr)
            pe = per_group_auc(fr, sc, EVENT, 1)
            m = {"prior_only/%s/pooled_auc" % name: float(pooled_auc(fr[LABEL], sc)), "prior_only/%s/per_event_mean" % name: float(np.mean(list(pe.values())))}
            put(m)
            say("16.1 prior-only %s: pooled %.4f | per-event mean %.4f" % (name, m["prior_only/%s/pooled_auc" % name], m["prior_only/%s/per_event_mean" % name]))

        res = permutation_audit(build_constrained_logistic, mp, six)
        base = res["base"]
        put({"perm/base_pooled_auc": float(res["base_pooled"]), "perm/base_per_event_mean": float(np.mean(list(base.values())))})
        for f in six:
            perm = {e: base[e] - res["within_drop"][f][e] for e in base}
            p = paired(perm, base)
            pooled = res["pooled_drop"][f]
            m = put_stats("perm/%s/within_drop" % f, p)
            m.update({"perm/%s/pooled_drop_mean" % f: float(np.mean(pooled)), "perm/%s/pooled_drop_sd" % f: float(np.std(pooled))})
            put(m)
            say("16.2 %-14s within-event drop %+.4f [%.4f, %.4f] | pooled drop %+.4f (sd %.4f)" % (f, p["mean_diff"], p["ci_low"], p["ci_high"], np.mean(pooled), np.std(pooled)))

        th = temporal_holdout(mp, {"con_six": (build_constrained_logistic, six), "con_no_land_cover": (build_constrained_logistic, no_lc)})
        ev = th["events"]
        n_test = len(ev["con_six"])
        m = {"temporal/train_rows": float(th["train_rows"]), "temporal/test_rows": float(th["test_rows"]), "temporal/scorable_test_events": float(n_test),
         "temporal/straddling_events": float(th["straddling_events"])}
        for arm, e in ev.items():
            if e:
                m["temporal/%s/per_event_mean" % arm] = float(np.mean(list(e.values())))
        if n_test >= MIN_TEST_EVENTS:
            m.update(put_stats("temporal/con_minus_elevation", paired(ev["elevation_only"], ev["con_six"])))
            m.update(put_stats("temporal/land_cover_contribution", paired(ev["con_no_land_cover"], ev["con_six"])))
        put(m)
        say("16.3 temporal: %d train rows, %d test rows, %d scorable test events" % (th["train_rows"], th["test_rows"], n_test))
        if n_test >= MIN_TEST_EVENTS:
            for label, key in (("six vs elevation only", "con_minus_elevation"), ("six vs without land_cover", "land_cover_contribution")):
                say("16.3 %s: %+.4f [%.4f, %.4f] wins %d losses %d" % ((label,) + tuple(m["temporal/%s/%s" % (key, k)] for k in STAT_KEYS)))
        else:
            say("16.3 not estimable: fewer than %d scorable test events" % MIN_TEST_EVENTS)

        splits = buffered_splits(mp)
        sel = events_for(build_constrained_logistic, mp, six)
        con = events_for(build_constrained_logistic, mp, six, splits)
        unc = events_for(build_logistic, mp, six, splits)
        elev = per_group_auc(mp, -mp["elevation"].to_numpy(float), EVENT, 1)
        m = {"buffer/min_train_rows": float(min(len(tr) for _, tr, _ in splits)), "buffer/con_per_event_mean": float(np.mean(list(con.values())))}
        m.update(put_stats("buffer/con_minus_selection_split", paired(sel, con)))
        m.update(put_stats("buffer/con_minus_elevation", paired(elev, con)))
        m.update(put_stats("buffer/con_minus_unconstrained", paired(unc, con)))
        put(m)
        say("16.5 buffered (%.0f m): con per-event mean %.4f | min training rows %d | vs selection split %+.4f | vs elevation %+.4f [%.4f, %.4f] | vs unconstrained %+.4f" % (
            BUFFER_M, m["buffer/con_per_event_mean"], m["buffer/min_train_rows"], m["buffer/con_minus_selection_split/mean_diff"],
            m["buffer/con_minus_elevation/mean_diff"], m["buffer/con_minus_elevation/ci_low"], m["buffer/con_minus_elevation/ci_high"],
            m["buffer/con_minus_unconstrained/mean_diff"]))

        for name, mask in subset_masks(mp).items():
            sub = mp[mask].reset_index(drop=True)
            con = events_for(build_constrained_logistic, sub, six)
            unc = events_for(build_logistic, sub, six)
            elev = per_group_auc(sub, -sub["elevation"].to_numpy(float), EVENT, 1)
            boost = {}
            for lib in ("hgb", "xgb"):
                sc, _ = nested_scores(make_estimator(lib, six, True), space_for(lib), sub, six, n_jobs=2)
                boost[lib + "_con"] = per_group_auc(sub, sc, EVENT, 1)
                say("16.4 %s: %s:con scored" % (name, lib))
            rev, det = reversal(con, boost)
            reversals[name] = bool(rev)
            r6, r5 = paired(elev, con), paired(unc, con)
            m = {"sens/%s/rows" % name: float(len(sub)), "sens/%s/scorable_events" % name: float(len(con)),
                 "sens/%s/elevation_per_event_mean" % name: r6["ref_mean"], "sens/%s/unconstrained_per_event_mean" % name: r5["ref_mean"],
                 "sens/%s/con_per_event_mean" % name: r6["other_mean"], "sens/%s/reversal" % name: float(rev)}
            m.update(put_stats("sens/%s/con_minus_elevation" % name, r6))
            m.update(put_stats("sens/%s/con_minus_unconstrained" % name, r5))
            for arm, d in det.items():
                m["sens/%s/%s_per_event_mean" % (name, arm)] = d["other_mean"]
                m["sens/%s/%s_beats_con" % (name, arm)] = float(d["beats"])
                m.update(put_stats("sens/%s/%s_minus_con" % (name, arm), d))
            put(m)
            say("16.4 %s (%d rows, %d scorable events): con %.4f | vs elevation %+.4f [%.4f, %.4f] | vs unconstrained %+.4f | hgb:con minus con %+.4f [%.4f, %.4f] | xgb:con minus con %+.4f [%.4f, %.4f] | reversal: %s" % (
                name, len(sub), len(con), r6["other_mean"], r6["mean_diff"], r6["ci_low"], r6["ci_high"], r5["mean_diff"],
                det["hgb_con"]["mean_diff"], det["hgb_con"]["ci_low"], det["hgb_con"]["ci_high"],
                det["xgb_con"]["mean_diff"], det["xgb_con"]["ci_low"], det["xgb_con"]["ci_high"], rev))
        run_id = run.info.run_id

    try:
        import mlflow as ml
        from mlflow.tracking import MlflowClient
        ml.set_tracking_uri(DEFAULT_TRACKING_URI)
        models = json.load(open(ROOT / "models" / "MANIFEST.json"))
        rel = [k for k, v in models.items() if v.get("status") == "active"][0]
        client = MlflowClient()
        client.log_artifact(REGISTRATION_RUN, str(ROOT / rel), artifact_path="registered_model")
        listed = [a.path for a in client.list_artifacts(REGISTRATION_RUN, "registered_model")]
        say("D12 artifact read-back: %s | file present: %s" % (listed, any(p.endswith(Path(rel).name) for p in listed)))
    except Exception as e:
        say("D12 artifact logging failed (%s: %s): D12 stays open" % (type(e).__name__, str(e)[:160]))
    os.environ.pop("MLFLOW_TRACKING_PASSWORD", None)
    print("run_id:", run_id)
    print("verdict reversals by subset:", reversals)


if __name__ == "__main__":
    main()

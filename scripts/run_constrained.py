"""Evaluate logistic:con on both frames, compare it with the logged booster runs, apply the registration rules (protocol
Section 14) and log everything to MLflow. Run from a clean checkout of main in Colab; the token is read through a hidden prompt."""
import getpass
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.baseline import build_logistic
from src.models.boosters import monotone_violations
from src.models.constrained import build_constrained_logistic
from src.models.cv import (CONSTRAINTS, EVENT, FEATURE_SETS, LABEL, SEED, load_layer_flags, load_training_frame, loeo_splits,
                           out_of_fold_scores, paired_event_bootstrap, per_group_auc, pooled_auc, restrict_to_mappable)
from src.models.registration import CHAMPION, REGISTERED_TRAINING_FRAME, eligible, registered
from src.tracking import tracked_run

BOOSTER_RUNS = {"mappable": "620e9d6219b64bc5818ba0fba82d92d7", "full": "b74d0c183b914b298a6b404e3d49718a"}


def frame_events(fr, builder, feats):
    """Selection-split out-of-fold scores: (pooled AUC, {event: AUC})."""
    sc = out_of_fold_scores(builder(feats), fr, feats, loeo_splits(fr, drop_shared_locations=True))
    return pooled_auc(fr[LABEL], sc), per_group_auc(fr, sc, EVENT, 1)


def versus_champion(champion_events, booster_events):
    """A booster's paired per-event interval (97.5%) against the champion, and whether it beats it (rule R3 building block)."""
    bs = paired_event_bootstrap(champion_events, booster_events, alpha=0.025)
    common = sorted(set(champion_events) & set(booster_events))
    champ_mean = float(np.mean([champion_events[e] for e in common]))
    boost_mean = float(np.mean([booster_events[e] for e in common]))
    return {"beats": bool(bs["ci_low"] > 0 and boost_mean > champ_mean), "mean": boost_mean, **bs}


def booster_events(metrics, frame, arm):
    pre = "%s/%s/event/" % (frame, arm.replace(":", "_"))
    return {k[len(pre):]: v for k, v in metrics.items() if k.startswith(pre)}


def main():
    df, _ = load_training_frame()
    frames = {"full": df, "mappable": restrict_to_mappable(df, load_layer_flags(df))}
    feats = FEATURE_SETS["six"]
    os.environ.setdefault("MLFLOW_TRACKING_USERNAME", "jmskoero")
    if not os.environ.get("MLFLOW_TRACKING_PASSWORD"):
        os.environ["MLFLOW_TRACKING_PASSWORD"] = getpass.getpass("DagsHub token (hidden): ")
    params = {"arm": "logistic-con", "frames": "full=%d,mappable=%d" % (len(df), len(frames["mappable"])), "seed": str(SEED),
              "booster_runs": ",".join("%s=%s" % kv for kv in BOOSTER_RUNS.items()),
              "registered_training_frame": REGISTERED_TRAINING_FRAME, "protocol": "docs/PHASE_C_PROTOCOL.md Section 14"}
    tags = {"phase": "C", "arm": "logistic-con", "model_status": "candidate-not-a-product"}
    m, res = {}, {}
    with tracked_run("phase-c-logistic-con", "BoundedLogistic", params, tags=tags) as (mlflow_mod, run):
        client = mlflow_mod.tracking.MlflowClient()
        for name, fr in frames.items():
            booster_metrics = client.get_run(BOOSTER_RUNS[name]).data.metrics
            con_pool, con = frame_events(fr, build_constrained_logistic, feats)
            _, unc = frame_events(fr, build_logistic, feats)
            elev = per_group_auc(fr, -fr["elevation"].to_numpy(float), EVENT, 1)
            m[name + "/con/pooled_auc"], m[name + "/con/per_event_mean"] = con_pool, float(np.mean(list(con.values())))
            for tag, ref in (("vs_unconstrained", unc), ("vs_elevation", elev)):
                bs = paired_event_bootstrap(ref, con, alpha=0.025)
                for k in ("mean_diff", "ci_low", "ci_high", "wins", "losses"):
                    m["%s/con/%s/%s" % (name, tag, k)] = float(bs[k])
            res[name] = {}
            for arm in ("hgb:con", "xgb:con"):
                v = versus_champion(con, booster_events(booster_metrics, name, arm))
                res[name][arm] = {"beats": v["beats"], "mean": v["mean"]}
                for k in ("mean_diff", "ci_low", "ci_high", "wins", "losses"):
                    m["%s/%s_vs_con/%s" % (name, arm.split(":")[0], k)] = float(v[k])
            res[name]["xgb_vs_hgb_ci_low"] = float(booster_metrics[name + "/xgb_con_vs_hgb_con/ci_low"])
        fit_fr = frames[REGISTERED_TRAINING_FRAME]
        model = build_constrained_logistic(feats).fit(fit_fr[feats], fit_fr[LABEL].to_numpy())
        viol = monotone_violations(model, fit_fr[feats], CONSTRAINTS)
        for f, n in viol.items():
            m["violations/con/" + f] = float(n)
        choice = registered(res["mappable"], res["full"])
        m["eligible"], m["registered_is_champion"] = float(eligible(viol)), float(choice == CHAMPION)
        assert all(np.isfinite(v) for v in m.values()), "non-finite metric"
        mlflow_mod.log_metrics(m)
        mlflow_mod.set_tag("registered_arm", choice)
        run_id = run.info.run_id
    os.environ.pop("MLFLOW_TRACKING_PASSWORD", None)
    print("run_id:", run_id)
    for name in frames:
        print(name, "| logistic:con per-event mean %.4f | vs unconstrained %+.4f [%.4f, %.4f] | vs elevation %+.4f [%.4f, %.4f]" % (
            m[name + "/con/per_event_mean"], m[name + "/con/vs_unconstrained/mean_diff"], m[name + "/con/vs_unconstrained/ci_low"],
            m[name + "/con/vs_unconstrained/ci_high"], m[name + "/con/vs_elevation/mean_diff"], m[name + "/con/vs_elevation/ci_low"],
            m[name + "/con/vs_elevation/ci_high"]))
        for arm in ("hgb", "xgb"):
            print("   ", arm + ":con minus logistic:con %+.4f [%.4f, %.4f] wins %d losses %d | beats: %s" % (
                m["%s/%s_vs_con/mean_diff" % (name, arm)], m["%s/%s_vs_con/ci_low" % (name, arm)], m["%s/%s_vs_con/ci_high" % (name, arm)],
                m["%s/%s_vs_con/wins" % (name, arm)], m["%s/%s_vs_con/losses" % (name, arm)], res[name][arm + ":con"]["beats"]))
    print("violations of logistic:con (rows of 200):", {f: int(n) for f, n in viol.items()})
    print("eligible (R1):", eligible(viol), "| registered arm (R2 to R4):", choice)


if __name__ == "__main__":
    main()

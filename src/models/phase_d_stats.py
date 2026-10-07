"""Phase D Stage 2 statistics (docs/PHASE_D_PROTOCOL.md Section 9): per-event AUCs stored in millionths, paired per-event bootstrap of the MLP minus each reference, reproduction checks and seeds."""
from src.models import phase_d as D
from src.models.cv import EVENT, MIN_CLASS_N, SEED, paired_event_bootstrap, per_group_auc


def aucs(df, scores):
    return {str(e): int(round(a * 1e6)) for e, a in per_group_auc(df, scores, EVENT, MIN_CLASS_N).items()}


def mean(a):
    return sum(a.values()) / len(a) / 1e6


def stat(ref, mlp):
    """mlp minus ref, from the stored millionths, so a test can recompute it exactly."""
    return paired_event_bootstrap({e: v / 1e6 for e, v in ref.items()}, {e: v / 1e6 for e, v in mlp.items()}, n_boot=D.N_BOOT, seed=SEED, alpha=D.ALPHA)


def record(R, elev):
    m = {"mlp": R["mlp_nested"]["auc"], "hgb:con": R["hgb_nested"]["auc"], "logistic:con": R["logcon_mappable"], "elevation": elev["mappable"]}
    f = {"mlp": R["mlp_modal_full"], "hgb:con": R["hgb_modal_full"], "logistic:con": R["logcon_full"], "elevation": elev["full"], "control": R["mlp_control_full"]}
    stats = {"mappable": {k: stat(m[k], m["mlp"]) for k in ("hgb:con", "logistic:con", "elevation")},
             "full": {k: stat(f[k], f["mlp"]) for k in ("hgb:con", "logistic:con", "elevation", "control")}}
    stats["mappable"]["control"] = stat(R["mlp_control_mappable"], R["mlp_modal_mappable_s%d" % D.FIXED["seed"]])
    checks = {arm: {"recomputed": round(mean(m[arm]), 6), "stored": D.REPRODUCE[arm][0], "tolerance": D.REPRODUCE[arm][1],
                    "reproduces": abs(mean(m[arm]) - D.REPRODUCE[arm][0]) <= D.REPRODUCE[arm][1]} for arm in ("hgb:con", "logistic:con")}
    checks["scorable_events_mappable"] = len(m["mlp"])
    arms = {"mappable": dict(m, control=R["mlp_control_mappable"], mlp_fixed=R["mlp_modal_mappable_s%d" % D.FIXED["seed"]]), "full": f}
    means = {fr: {k: round(mean(v), 6) for k, v in a.items()} for fr, a in arms.items()}
    seeds = {str(s): round(mean(R["mlp_modal_mappable_s%d" % s]), 6) for s in D.SEEDS_REPORTED}
    return {"auc_micro": arms, "mean_per_event_auc": means, "stats": stats, "checks": checks, "seeds_mean_mappable": seeds}

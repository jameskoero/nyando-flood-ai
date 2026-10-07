"""Phase D Stage 2 jobs (docs/PHASE_D_PROTOCOL.md Section 9): the smoke test, the fixed-point evaluations and the violation check used by phase_d_run."""
import time

from src.models import phase_d as D
from src.models.boosters import expand, monotone_violations
from src.models.constrained import build_constrained_logistic
from src.models.cv import CONSTRAINTS, FEATURE_SETS, LABEL, out_of_fold_scores
from src.models.mlp import PhysicsMLP
from src.models.phase_d_stats import aucs

SIX = list(FEATURE_SETS["six"])


def mlp(p, seed=D.FIXED["seed"]):
    return PhysicsMLP(hidden=tuple(p["hidden"]), lam=p["lam"], seed=seed)


def smoke(M, S, hgb, hspace):
    """One outer fold of every model and one violation check before any long step. fit_s is the time of one full MLP fit."""
    p0, t1 = {"hidden": D.HIDDEN[0], "lam": 1.0}, time.time()
    out_of_fold_scores(mlp(p0), M, SIX, S["mappable"][:1])
    fit_s = time.time() - t1
    for est in (hgb(expand(hspace)[0]), build_constrained_logistic(SIX)):
        out_of_fold_scores(est, M, SIX, S["mappable"][:1])
    monotone_violations(mlp(p0).set_params(epochs=1).fit(M[SIX], M[LABEL].to_numpy()), M[SIX], CONSTRAINTS)
    return {"outer_folds": len(S["mappable"]), "fit_s": round(fit_s, 2)}


def jobs(F, S, pt, hgb):
    def fixed(est, fr):
        return aucs(F[fr], out_of_fold_scores(est, F[fr], SIX, S[fr]))
    ctl = {"hidden": pt["mlp"]["hidden"], "lam": 0.0}
    out = [("mlp_modal_mappable_s%d" % s, lambda s=s: fixed(mlp(pt["mlp"], s), "mappable")) for s in D.SEEDS_REPORTED]
    return out + [("mlp_modal_full", lambda: fixed(mlp(pt["mlp"]), "full")), ("mlp_control_mappable", lambda: fixed(mlp(ctl), "mappable")),
                  ("mlp_control_full", lambda: fixed(mlp(ctl), "full")), ("hgb_modal_full", lambda: fixed(hgb(pt["hgb"]), "full")),
                  ("logcon_mappable", lambda: fixed(build_constrained_logistic(SIX), "mappable")), ("logcon_full", lambda: fixed(build_constrained_logistic(SIX), "full"))]


def violations(M, pt, hgb):
    y, ctl = M[LABEL].to_numpy(), {"hidden": pt["mlp"]["hidden"], "lam": 0.0}
    return {n: monotone_violations(e.fit(M[SIX], y), M[SIX], CONSTRAINTS) for n, e in (("mlp", mlp(pt["mlp"])), ("mlp_control", mlp(ctl)), ("hgb", hgb(pt["hgb"])))}

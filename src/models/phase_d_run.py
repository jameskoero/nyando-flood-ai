"""Phase D Stage 2 run (docs/PHASE_D_PROTOCOL.md Section 9, register row D46): selection and evaluation on the existing events. Resumable: every outer fold and every step is cached in the work
directory, and the run stops when its time budget is used up (run it again to resume). Writes results_raw.json. Nothing here decides promotion: Block B decides."""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import sklearn

from src.models import phase_d as D
from src.models.boosters import expand, make_estimator, space_for
from src.models.cv import FLAGS_PATH, load_layer_flags, load_training_frame, loeo_splits, restrict_to_mappable
from src.models.phase_d_cv import nested_cached
from src.models.phase_d_jobs import SIX, jobs, mlp, smoke, violations
from src.models.phase_d_stats import aucs, record

REPO = Path(__file__).resolve().parents[2]


def run(work="/content/p2", budget_s=1500):
    import torch
    t0, work = time.time(), Path(work)
    work.mkdir(exist_ok=True)
    df, tsha = load_training_frame(REPO)
    F = {"mappable": restrict_to_mappable(df, load_layer_flags(df, FLAGS_PATH)).reset_index(drop=True), "full": df.reset_index(drop=True)}
    M, S = F["mappable"], {k: loeo_splits(v, drop_shared_locations=True) for k, v in F.items()}
    commit = subprocess.run("git rev-parse HEAD", shell=True, capture_output=True, text=True, cwd=REPO).stdout.strip()
    tag, hgb, hspace = "%s_%s" % (tsha[:12], commit[:10]), make_estimator("hgb", SIX, True), space_for("hgb")
    rp = work / ("results_%s.json" % tag)
    R = json.loads(rp.read_text()) if rp.exists() else {}

    def stop():
        rp.write_text(json.dumps(R))
        print("time budget reached: run again to resume", flush=True)
        return False

    def step(key, fn):
        if key not in R:
            if time.time() - t0 > budget_s:
                return stop()
            R[key] = fn()
            rp.write_text(json.dumps(R))
            print("%s done (%.0f s)" % (key, time.time() - t0), flush=True)
        return True

    if not step("smoke", lambda: smoke(M, S, hgb, hspace)):
        return False
    k = R["smoke"]
    print("smoke ok: %d outer folds; the MLP nested selection takes at most %.0f min at %.1f s per fit" % (k["outer_folds"], k["outer_folds"] * (len(D.GRID) * D.INNER_FOLDS + 1) * k["fit_s"] / 60, k["fit_s"]), flush=True)
    for name, make, space in (("mlp_nested", mlp, {"hidden": D.HIDDEN, "lam": D.LAMBDAS}), ("hgb_nested", hgb, hspace)):
        if name not in R:
            out = nested_cached(make, space, M, SIX, work / ("%s_%s.json" % (name, tag)), budget_s - (time.time() - t0), name)
            if out is None:
                return stop()
            R[name] = {"auc": aucs(M, out[0]), "chosen": out[1]}
            rp.write_text(json.dumps(R))
    pt = {"mlp": D.modal_point(R["mlp_nested"]["chosen"], list(D.GRID)), "hgb": D.modal_point(R["hgb_nested"]["chosen"], expand(hspace))}
    if not all(step(key, fn) for key, fn in jobs(F, S, pt, hgb)) or not step("violations", lambda: violations(M, pt, hgb)):
        return False
    elev = {k: aucs(F[k], -F[k]["elevation"].to_numpy()) for k in F}
    out = dict(record(R, elev), schema=1, training_data_sha256=tsha, commit=commit, modal_points=pt, chosen={"mlp": R["mlp_nested"]["chosen"], "hgb": R["hgb_nested"]["chosen"]},
               violations=R["violations"], smoke=R["smoke"], elapsed_s=round(time.time() - t0),
               environment={"python": sys.version.split()[0], "numpy": np.__version__, "scikit-learn": sklearn.__version__, "torch": torch.__version__})
    (work / "results_raw.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    print("\n=== Stage 2 complete: mean per-event AUC", json.dumps(out["mean_per_event_auc"]), flush=True)
    for fr, d in out["stats"].items():
        for ref, s in d.items():
            print("%-8s MLP minus %-12s %+.4f [%.4f, %.4f] wins %d losses %d" % (fr, ref, s["mean_diff"], s["ci_low"], s["ci_high"], s["wins"], s["losses"]))
    print("checks:", json.dumps(out["checks"]), "| selected:", json.dumps(pt), "| violations:", json.dumps(out["violations"]), flush=True)
    return True

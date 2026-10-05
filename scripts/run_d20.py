"""scripts/run_d20.py: score D20 Block A once with the frozen Phase C models (docs/D20_PROTOCOL.md Sections 5 to 7 and 11) and return what docs/D20_RESULTS.json stores. Colab only: scikit-learn, xgboost and a clean venv for ONNX Runtime. Intermediate results are cached in the work directory under a hash of every input, so a re-run never scores twice and never reuses stale results."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
TOL = {"champion": 1e-5, "hgb:con": 1e-3, "xgb:con": 1e-3}
KEY_FILES = ("data/training/nyando_training_v2_multidate.csv", "data/derived/gfm_layer_flags.csv", "data/confirmatory/nyando_block_a.csv", "data/confirmatory/nyando_block_a_layer_flags.csv",
             "data/derived/d20_selection.json", "models/MANIFEST.json", "docs/PHASE_C_RESULTS.json", "src/models/cv.py", "src/models/boosters.py", "src/models/constrained.py")


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    return o.item() if hasattr(o, "item") else o


def cache_key():
    """A hash of every input of this run: a cached intermediate result is reused only when none of them changed."""
    return hashlib.sha256("".join(hashlib.sha256((REPO / p).read_bytes()).hexdigest() for p in KEY_FILES).encode()).hexdigest()


def environment(work):
    import numpy, pandas, scipy, sklearn, xgboost
    ort = subprocess.run([str(work / "venv" / "bin" / "python"), "-c", "import onnxruntime; print(onnxruntime.__version__)"], capture_output=True, text=True).stdout.strip()
    return {"python": sys.version.split()[0], "numpy": numpy.__version__, "pandas": pandas.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__, "onnxruntime": ort}


def stage1(work):
    """The existing events on the mappable frame: the Phase C evaluation recomputed (champion leave-one-event-out, boosters nested) with the Phase C definition of a scorable event (both classes present) and checked against the stored Phase C values, arm by arm, before anything is pooled."""
    import numpy as np
    from src.models.boosters import make_estimator, nested_scores, space_for
    from src.models.constrained import build_constrained_logistic
    from src.models.cv import EVENT, FEATURE_SETS, FLAGS_PATH, load_layer_flags, load_training_frame, loeo_splits, out_of_fold_scores, per_group_auc, restrict_to_mappable
    res = json.loads((REPO / "docs" / "PHASE_C_RESULTS.json").read_text(encoding="utf-8"))["runs"]
    stored = {"champion": res["registration"]["metrics"]["mappable/con/per_event_mean"], "hgb:con": res["boosters_mappable"]["metrics"]["mappable/hgb_con/per_event_mean"],
              "xgb:con": res["boosters_mappable"]["metrics"]["mappable/xgb_con/per_event_mean"]}
    df, _ = load_training_frame(REPO)
    frame, six, out = restrict_to_mappable(df, load_layer_flags(df, FLAGS_PATH)), FEATURE_SETS["six"], {}

    def check(arm):
        d = abs(out[arm]["mean"] - stored[arm])
        print("%s: recomputed per-event mean %.6f | stored in the Phase C results %.6f | difference %.2e" % (arm, out[arm]["mean"], stored[arm], d), flush=True)
        if d > TOL[arm]:
            raise SystemExit("STOPPED: %s does not reproduce the stored Phase C value (difference %.2e, limit %.0e)" % (arm, d, TOL[arm]))

    cf = work / "old_champion.json"
    if cf.exists():
        out["champion"] = json.loads(cf.read_text())
    else:
        sc = out_of_fold_scores(build_constrained_logistic(six), frame, six, loeo_splits(frame, True))
        pe = per_group_auc(frame, sc, EVENT, 1)
        out["champion"] = {"events": _clean(pe), "mean": float(np.mean(list(pe.values())))}
        cf.write_text(json.dumps(out["champion"]))
    check("champion")
    for kind in ("hgb", "xgb"):
        arm, cf = kind + ":con", work / ("old_%s.json" % kind)
        if cf.exists():
            out[arm] = json.loads(cf.read_text())
        else:
            t0 = time.time()
            print("nested selection for %s on the mappable frame (the Phase C run, recomputed) ..." % arm, flush=True)
            scores, chosen = nested_scores(make_estimator(kind, six, True), space_for(kind), frame, six, n_jobs=2)
            pe, modal = per_group_auc(frame, scores, EVENT, 1), max(chosen, key=chosen.count)
            out[arm] = {"events": _clean(pe), "mean": float(np.mean(list(pe.values()))), "modal": _clean(modal)}
            cf.write_text(json.dumps(out[arm]))
            print("  done in %.0f s | modal grid point %s" % (time.time() - t0, modal), flush=True)
        check(arm)
    return out, {a: {"stored": stored[a], "recomputed": out[a]["mean"]} for a in out}


ONNX_RUN = """import csv, sys
import numpy as np
import onnxruntime as ort
s = ort.InferenceSession(open(sys.argv[1], 'rb').read(), providers=['CPUExecutionProvider'])
rows = list(csv.DictReader(open(sys.argv[2])))
F = ('elevation', 'slope', 'rainfall_3day', 'distance_river', 'clay_percent', 'land_cover')
feed = {n: (np.array([[int(float(r[n]))] for r in rows], dtype=np.int64) if n == 'land_cover' else np.array([[float('nan') if r[n] == '' else float(r[n])] for r in rows], dtype=np.float64)) for n in F}
p = s.run(['probabilities'], feed)[0][:, 1]
open(sys.argv[3], 'w').write('\\n'.join(repr(float(x)) for x in p) + '\\n')
"""


def onnx_scores(work, rel, X):
    import numpy as np
    X.to_csv(work / "onnx_in.csv", index=False)
    (work / "onnx_run.py").write_text(ONNX_RUN)
    venv = work / "venv"
    if not (venv / "bin" / "python").exists():
        for c in ("pip -q install uv", "uv venv --clear %s --python %s" % (venv, sys.executable), "uv pip install --python %s/bin/python numpy onnxruntime==1.30.0" % venv):
            r = subprocess.run(c, shell=True, capture_output=True, text=True)
            if r.returncode:
                raise SystemExit("STOPPED: %s failed:\n%s" % (c[:60], (r.stdout + r.stderr)[-500:]))
    r = subprocess.run([str(venv / "bin" / "python"), str(work / "onnx_run.py"), str(REPO / rel), str(work / "onnx_in.csv"), str(work / "onnx_out.txt")], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("STOPPED: ONNX scoring failed:\n" + (r.stdout + r.stderr)[-500:])
    return np.array([float(x) for x in (work / "onnx_out.txt").read_text().split()])


def validate_block_a(blk, sel):
    """The Block A data contract: schema, events inside the committed Block A, finite features, known land cover classes, no duplicate points, per-class counts within the committed sample size."""
    import numpy as np
    from src.models.boosters import LAND_COVER_CLASSES
    cols = ["event_id", "sample_set", "sample_mode", "event_date", "lon", "lat", "flooded", "gfm_item_id", "elevation", "slope", "rainfall_3day", "distance_river", "clay_percent", "land_cover"]
    miss = [c for c in cols if c not in blk.columns]
    if miss:
        raise SystemExit("STOPPED: Block A lacks columns %s" % miss)
    a, b, ev = {"date-" + d for d in sel["block_a"]}, {"date-" + d for d in sel["block_b"]}, set(blk["event_id"])
    if not ev or not ev <= a or ev & b:
        raise SystemExit("STOPPED: the Block A events are not a non-empty subset of the committed Block A dates, or they touch Block B")
    num = ["elevation", "slope", "rainfall_3day", "distance_river", "land_cover"]
    if not np.isfinite(blk[num].to_numpy(dtype=float)).all():
        raise SystemExit("STOPPED: Block A has a missing or infinite value in %s" % num)
    if not set(blk["land_cover"].astype(int)) <= set(LAND_COVER_CLASSES):
        raise SystemExit("STOPPED: Block A has a land_cover class the models were not trained on")
    if blk.duplicated(["lon", "lat", "event_date"]).any():
        raise SystemExit("STOPPED: Block A has duplicate points")
    per = blk.groupby("event_id")["flooded"].agg(["sum", "count"])
    if ((per["sum"] > sel["per_class"]) | ((per["count"] - per["sum"]) > sel["per_class"])).any():
        raise SystemExit("STOPPED: a Block A date has more points of a class than the committed sample size")
    return {"events": len(ev), "rows": len(blk)}


def preflight(work, sel):
    """Everything that can fail cheaply, before the one-shot: the Block A data contract, and the ONNX path on real training rows (never Block A labels) against an independent refit."""
    import numpy as np
    import pandas as pd
    from src.models.constrained import build_constrained_logistic
    from src.models.cv import FEATURE_SETS, FLAGS_PATH, LABEL, load_layer_flags, load_training_frame, restrict_to_mappable
    contract = validate_block_a(pd.read_csv(REPO / "data" / "confirmatory" / "nyando_block_a.csv"), sel)
    rel = [k for k, v in json.loads((REPO / "models" / "MANIFEST.json").read_text(encoding="utf-8")).items() if v.get("status") == "active"][0]
    df, _ = load_training_frame(REPO)
    frame, six = restrict_to_mappable(df, load_layer_flags(df, FLAGS_PATH)), FEATURE_SETS["six"]
    probe = frame.iloc[np.linspace(0, len(frame) - 1, 40).astype(int)]
    onnx = onnx_scores(work, rel, probe[six])
    refit = build_constrained_logistic(six).fit(frame[six], frame[LABEL].to_numpy()).predict_proba(probe[six])[:, 1]
    d = float(np.max(np.abs(onnx - refit)))
    print("preflight: Block A contract ok (%d events, %d rows) | ONNX artifact against an independent refit on 40 training rows: max abs difference %.2e" % (contract["events"], contract["rows"], d), flush=True)
    if d > 1e-3:
        raise SystemExit("STOPPED: the registered ONNX artifact and an independent refit disagree on training rows (%.2e): send me this line" % d)
    return {"contract": contract, "onnx_vs_refit_training_rows": d}


def stage2(work, old):
    """Block A, scored once. Every arm is fit on the mappable training frame without the rows at locations that appear in Block A (protocol Section 12.1): the champion's pipeline refit, the challengers at the recomputed modal grid points, the no-land_cover logistic and elevation only. The registered ONNX artifact, fit on every row, is scored as a sensitivity."""
    import numpy as np
    import pandas as pd
    from sklearn.base import clone
    from src.models import d20
    from src.models.boosters import make_estimator, monotone_violations
    from src.models.constrained import build_constrained_logistic
    from src.models.cv import CONSTRAINTS, EVENT, FEATURE_SETS, FLAGS_PATH, LABEL, MIN_CLASS_N, load_layer_flags, load_training_frame, location_keys, per_group_auc, restrict_to_mappable
    models = json.loads((REPO / "models" / "MANIFEST.json").read_text(encoding="utf-8"))
    rel, entry = [(k, v) for k, v in models.items() if v.get("status") == "active"][0]
    blk = pd.read_csv(REPO / "data" / "confirmatory" / "nyando_block_a.csv")
    fl = pd.read_csv(REPO / "data" / "confirmatory" / "nyando_block_a_layer_flags.csv")
    if len(fl) != len(blk) or list(fl["flooded"]) != list(blk[LABEL]):
        raise SystemExit("STOPPED: the layer flags do not align with the Block A rows")
    keep = ((blk[LABEL] == 1) | (fl["exclusion_mask"] == 0)).to_numpy()
    df, tsha = load_training_frame(REPO)
    frame, six = restrict_to_mappable(df, load_layer_flags(df, FLAGS_PATH)), FEATURE_SETS["six"]
    fkeys, bkeys = list(location_keys(frame)), list(location_keys(blk))
    drop = np.asarray(d20.shared_training_rows(fkeys, bkeys), dtype=bool)
    shared = len(set(fkeys) & set(bkeys))
    train = frame[~drop].reset_index(drop=True)
    ytr = train[LABEL].to_numpy()
    print("%d Block A locations also appear in the training frame: %d of %d training rows are dropped from every arm (protocol Section 12.1); %d rows remain (%d floods, %d controls)" % (shared, int(drop.sum()), len(frame), len(train), int(ytr.sum()), int(len(ytr) - ytr.sum())), flush=True)
    if len(train) < 1000 or ytr.sum() < 200 or len(ytr) - ytr.sum() < 200:
        raise SystemExit("STOPPED: too few training rows remain after dropping the shared locations: send me this line")
    scores, viol = {}, {}
    for kind in ("hgb", "xgb"):
        arm = kind + ":con"
        m = clone(make_estimator(kind, six, True)(old[arm]["modal"])).fit(train[six], ytr)
        viol[arm], scores[arm] = monotone_violations(m, train[six], CONSTRAINTS), m.predict_proba(blk[six])[:, 1]
    nolc = [f for f in six if f != "land_cover"]
    scores["no_land_cover"] = build_constrained_logistic(nolc).fit(train[nolc], ytr).predict_proba(blk[nolc])[:, 1]
    scores["elevation"] = -blk["elevation"].to_numpy(dtype=float)
    scores["champion"] = build_constrained_logistic(six).fit(train[six], ytr).predict_proba(blk[six])[:, 1]
    artifact = onnx_scores(work, rel, blk[six])
    diff = float(np.max(np.abs(artifact - scores["champion"])))
    print("primary champion (refit without the shared locations) against the registered artifact on the Block A features: max abs difference %.2e (recorded)" % diff, flush=True)
    out, art = {}, {}
    for fr, mask in (("mappable", keep), ("full", np.ones(len(blk), dtype=bool))):
        sub = blk[mask].reset_index(drop=True)
        out[fr] = {arm: per_group_auc(sub, np.asarray(s)[mask], EVENT, MIN_CLASS_N) for arm, s in scores.items()}
        art[fr] = per_group_auc(sub, artifact[mask], EVENT, MIN_CLASS_N)
    sha = hashlib.sha256((REPO / "data" / "confirmatory" / "nyando_block_a.csv").read_bytes()).hexdigest()
    meta = {"file": "data/confirmatory/nyando_block_a.csv", "sha256": sha, "rows": len(blk), "scorable_mappable": len(out["mappable"]["champion"]), "scorable_full": len(out["full"]["champion"]),
            "shared_locations": shared, "training_rows_total": len(frame), "training_rows_dropped": int(drop.sum()), "training_rows_used": len(train), "primary_champion_vs_artifact_max_abs_diff": diff}
    mods = {"champion": {"file": rel, "sha256": entry["sha256"]}, "primary_champion": {"pipeline": "build_constrained_logistic(six), fit without the rows at locations shared with Block A", "training_rows": len(train)},
            **{a: {"modal": _clean(old[a]["modal"]), "violations": _clean(viol[a])} for a in viol}}
    return out, art, mods, meta, tsha


def run(work_dir="/content/d20_score"):
    from src.data import block_a
    from src.models.d20 import analyse, from_micro, sensitivity, to_micro
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    key, kf = cache_key(), work / "cache_key.txt"
    if kf.exists() and kf.read_text() != key:
        raise SystemExit("STOPPED: the cached results in %s were computed from different inputs: delete the folder (Cell 0) and start again" % work)
    kf.write_text(key)
    rf = work / "results.json"
    if rf.exists():
        print("results already computed: not scoring again", flush=True)
        return json.loads(rf.read_text())
    t0 = time.time()
    pre = preflight(work, block_a.load_selection())
    old, repro = stage1(work)
    blk, art, mods, meta, tsha = stage2(work, old)
    old_m = {a: to_micro(old[a]["events"]) for a in old}
    blk_m = {fr: {arm: to_micro(ev) for arm, ev in arms.items()} for fr, arms in blk.items()}
    art_m = {fr: to_micro(ev) for fr, ev in art.items()}
    blk_f = {fr: {arm: from_micro(e) for arm, e in arms.items()} for fr, arms in blk_m.items()}
    ana = analyse({a: from_micro(e) for a, e in old_m.items()}, blk_f, {c: mods[c]["violations"] for c in ("hgb:con", "xgb:con")})
    sens = sensitivity(blk_f, {fr: from_micro(e) for fr, e in art_m.items()})
    res = {"schema": 3, "protocol": "docs/D20_PROTOCOL.md Sections 5 to 7, 11 and 12.1", "training_data_sha256": tsha, "block_a": meta, "models": mods, "reproduction": repro, "old_events": old_m,
           "block_a_auc": blk_m, "analysis": _clean(ana), "registered_artifact_sensitivity": {"block_a_auc": art_m, "analysis": _clean(sens)},
           "bootstrap": {"n_boot": 10000, "seed": 42, "alpha": 0.025}, "preflight": pre, "environment": environment(work), "seconds": round(time.time() - t0)}
    rf.write_text(json.dumps(res))
    return res

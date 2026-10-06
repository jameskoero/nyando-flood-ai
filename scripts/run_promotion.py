"""scripts/run_promotion.py: compute the statistical gates S1 to S6 of docs/PROMOTION_PROTOCOL.md (hgb:con against logistic:con) and return what
docs/PROMOTION_RESULTS.json stores. Colab only (scikit-learn; no network). Heavy parts are cached in the work directory under a hash of every input."""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
TOL = {"champion": 1e-6, "hgb:con": 1e-3}
ARMS = ("champion", "hgb:con")
KEY_FILES = ("data/training/nyando_training_v2_multidate.csv", "data/derived/gfm_layer_flags.csv", "data/confirmatory/nyando_block_a.csv",
             "data/confirmatory/nyando_block_a_layer_flags.csv", "docs/D20_RESULTS.json", "docs/PHASE_C_RESULTS.json", "docs/PROMOTION_PROTOCOL.md",
             "src/models/cv.py", "src/models/boosters.py", "src/models/constrained.py", "src/models/robustness.py", "src/models/promotion.py", "src/models/d20.py")


def file_sha(p):
    return hashlib.sha256((REPO / p).read_bytes()).hexdigest()


def cache_key():
    return hashlib.sha256("".join(file_sha(p) for p in KEY_FILES).encode()).hexdigest()


def micro(d):
    return {str(k): int(round(float(v) * 1e6)) for k, v in d.items()}


def unmicro(d):
    return {k: v / 1e6 for k, v in d.items()}


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o.item() if hasattr(o, "item") else o


def cached(work, name, fn):
    f = work / (name.replace("/", "__").replace(":", "_") + ".json")
    if f.exists():
        return json.loads(f.read_text())
    t0 = time.time()
    out = fn()
    f.write_text(json.dumps(out))
    print("  computed %s in %.0f s" % (name, time.time() - t0), flush=True)
    return out


def pool(a, b):
    if set(a) & set(b):
        raise SystemExit("STOPPED: an existing event and a Block A event share an id")
    return {**a, **b}


def builders():
    from src.models.boosters import build_hgb
    from src.models.constrained import build_constrained_logistic
    from src.models.promotion import FIXED_PARAMS
    return {"champion": lambda feats: build_constrained_logistic(feats), "hgb:con": lambda feats: build_hgb(feats, True, **FIXED_PARAMS)}


def load():
    import pandas as pd
    from src.models import d20
    from src.models.cv import FEATURE_SETS, FLAGS_PATH, LABEL, load_layer_flags, load_training_frame, location_keys, restrict_to_mappable
    df, tsha = load_training_frame(REPO)
    df = df.reset_index(drop=True)
    frame = restrict_to_mappable(df, load_layer_flags(df, FLAGS_PATH)).reset_index(drop=True)
    blk = pd.read_csv(REPO / "data" / "confirmatory" / "nyando_block_a.csv")
    fl = pd.read_csv(REPO / "data" / "confirmatory" / "nyando_block_a_layer_flags.csv")
    if len(fl) != len(blk) or list(fl["flooded"]) != list(blk[LABEL]):
        raise SystemExit("STOPPED: the Block A layer flags do not align with its rows")
    drop = np.asarray(d20.shared_training_rows(list(location_keys(frame)), list(location_keys(blk))), dtype=bool)
    return {"df": df, "frame": frame, "blk": blk, "keep": ((blk[LABEL] == 1) | (fl["exclusion_mask"] == 0)).to_numpy(), "train": frame[~drop].reset_index(drop=True),
            "d20": json.loads((REPO / "docs" / "D20_RESULTS.json").read_text(encoding="utf-8")), "tsha": tsha, "six": FEATURE_SETS["six"]}


def loeo_events(data, builder, six):
    from src.models.cv import EVENT, loeo_splits, out_of_fold_scores, per_group_auc
    return per_group_auc(data, out_of_fold_scores(builder(six), data, six, loeo_splits(data, True)), EVENT, 1)


def block_events(S, train, mask, builder):
    from src.models.cv import EVENT, LABEL, MIN_CLASS_N, per_group_auc
    six = S["six"]
    model = builder(six).fit(train[six], train[LABEL].to_numpy())
    sub = S["blk"][mask].reset_index(drop=True)
    return per_group_auc(sub, model.predict_proba(sub[six])[:, 1], EVENT, MIN_CLASS_N)


def check_block_a(work, S, B):
    """Recompute before trust: this script's Block A path must reproduce the per-event AUCs stored by D20."""
    A = S["d20"]["block_a_auc"]["mappable"]
    if len(S["train"]) != S["d20"]["block_a"]["training_rows_used"]:
        raise SystemExit("STOPPED: the Block A training frame has %d rows, D20 used %d" % (len(S["train"]), S["d20"]["block_a"]["training_rows_used"]))
    out = {}
    for arm in ARMS:
        got = cached(work, "check/block_a/" + arm, lambda arm=arm: micro(block_events(S, S["train"], S["keep"], B[arm])))
        if set(got) != set(A[arm]):
            raise SystemExit("STOPPED: the Block A events differ from D20 for %s" % arm)
        out[arm] = max(abs(got[e] - A[arm][e]) for e in got) / 1e6
        print("Block A %s against D20: max per-event difference %.2e" % (arm, out[arm]), flush=True)
        if out[arm] > TOL[arm]:
            raise SystemExit("STOPPED: the Block A path does not reproduce D20 for %s" % arm)
    return out


def full_frame(work, S, B):
    """S2 full frame: the existing events evaluated as in Phase C, checked against the stored Phase C per-event means before pooling."""
    from src.models.boosters import make_estimator, nested_scores, space_for
    from src.models.cv import EVENT, per_group_auc
    df, six = S["df"], S["six"]
    m = json.loads((REPO / "docs" / "PHASE_C_RESULTS.json").read_text(encoding="utf-8"))["runs"]
    try:
        stored = {"champion": m["registration"]["metrics"]["full/con/per_event_mean"], "hgb:con": m["boosters_full"]["metrics"]["full/hgb_con/per_event_mean"]}
    except KeyError as e:
        raise SystemExit("STOPPED: the stored Phase C results lack %s" % e)
    def one(arm):
        pe = loeo_events(df, B["champion"], six) if arm == "champion" else per_group_auc(df, nested_scores(make_estimator("hgb", six, True), space_for("hgb"), df, six, n_jobs=2)[0], EVENT, 1)
        return {"events": micro(pe), "mean": float(np.mean(list(pe.values())))}
    out, rep = {}, {}
    for arm in ARMS:
        out[arm] = cached(work, "full/" + arm, lambda arm=arm: one(arm))
        rep[arm] = {"stored": stored[arm], "recomputed": out[arm]["mean"]}
        print("full frame %s: recomputed per-event mean %.6f | stored %.6f" % (arm, out[arm]["mean"], stored[arm]), flush=True)
        if abs(out[arm]["mean"] - stored[arm]) > TOL[arm]:
            raise SystemExit("STOPPED: %s on the full frame does not reproduce Phase C" % arm)
    return {a: unmicro(out[a]["events"]) for a in ARMS}, rep


def run(work_dir="/content/prom_run"):
    import pandas as pd
    import sklearn
    from src.models import promotion as pr
    from src.models.cv import EVENT, out_of_fold_scores, per_group_auc
    from src.models.robustness import buffered_splits, paired, subset_masks, temporal_holdout
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    key, kf = cache_key(), work / "cache_key.txt"
    if kf.exists() and kf.read_text() != key:
        raise SystemExit("STOPPED: the cache in %s came from different inputs: delete the folder and run again" % work)
    kf.write_text(key)
    t0, S, B = time.time(), load(), builders()
    six, frame, A = S["six"], S["frame"], S["d20"]["block_a_auc"]
    check = check_block_a(work, S, B)
    full, rep = full_frame(work, S, B)
    old = {a: unmicro(S["d20"]["old_events"][a]) for a in ARMS}
    new = {fr: {a: unmicro(A[fr][a]) for a in ARMS + ("elevation",)} for fr in ("mappable", "full")}
    pairs = {"new/mappable": (new["mappable"]["champion"], new["mappable"]["hgb:con"]), "new/full": (new["full"]["champion"], new["full"]["hgb:con"]),
             "pooled/mappable": (pool(old["champion"], new["mappable"]["champion"]), pool(old["hgb:con"], new["mappable"]["hgb:con"])),
             "pooled/full": (pool(full["champion"], new["full"]["champion"]), pool(full["hgb:con"], new["full"]["hgb:con"]))}
    s1, d1 = pr.r3({k: v for k, v in pairs.items() if k.startswith("new/")})
    s2, d2 = pr.r3({k: v for k, v in pairs.items() if k.startswith("pooled/")})
    fm, bm, tm, runs = subset_masks(frame), subset_masks(S["blk"]), subset_masks(S["train"]), {}
    for s in pr.SUBSETS:
        fs, ts = frame[fm[s]].reset_index(drop=True), S["train"][tm[s]].reset_index(drop=True)
        o = {a: unmicro(cached(work, "sub/%s/old/%s" % (s, a), lambda a=a, fs=fs: micro(loeo_events(fs, B[a], six)))) for a in ARMS}
        n = {a: unmicro(cached(work, "sub/%s/new/%s" % (s, a), lambda a=a, ts=ts, s=s: micro(block_events(S, ts, S["keep"] & bm[s], B[a])))) for a in ARMS}
        runs[s + "/new"] = (n["champion"], n["hgb:con"])
        runs[s + "/pooled"] = (pool(o["champion"], n["champion"]), pool(o["hgb:con"], n["hgb:con"]))
    bs = buffered_splits(frame)
    buf = {a: unmicro(cached(work, "buffered/" + a, lambda a=a: micro(per_group_auc(frame, out_of_fold_scores(B[a](six), frame, six, bs), EVENT, 1)))) for a in ARMS}
    runs["buffered"] = (buf["champion"], buf["hgb:con"])
    th = cached(work, "temporal", lambda: {k: micro(v) for k, v in temporal_holdout(frame, {a: (B[a], six) for a in ARMS})["events"].items()})
    runs["temporal"] = (unmicro(th["champion"]), unmicro(th["hgb:con"]))
    _, detail = pr.sensitivity(runs)
    s3 = not any(d["unresolved"] for k, d in detail.items() if k.split("/")[0] in pr.SUBSETS)
    s4, s5 = not detail["buffered"]["unresolved"], not detail["temporal"]["unresolved"]
    elev = unmicro(cached(work, "old/elevation", lambda: micro(per_group_auc(frame, -frame["elevation"].to_numpy(float), EVENT, 1))))
    s6pair = (pool({e: v for e, v in elev.items() if e in old["hgb:con"]}, new["mappable"]["elevation"]), pool(old["hgb:con"], new["mappable"]["hgb:con"]))
    gates = {"S1": s1, "S2": s2, "S3": s3, "S4": s4, "S5": s5, "S6": pr.r6(*s6pair)}
    allp = {**pairs, **runs, "S6/pooled_mappable": s6pair}
    stats = {k: {f: v for f, v in clean(paired(a, b)).items() if f not in ("ref_mean", "other_mean")} for k, (a, b) in allp.items()}
    stop = not pr.build_exporter(gates)
    res = {"schema": 1, "protocol": "docs/PROMOTION_PROTOCOL.md", "protocol_sha256": file_sha("docs/PROMOTION_PROTOCOL.md"),
           "inputs": {p: file_sha(p) for p in KEY_FILES if not p.startswith("src/")}, "training_data_sha256": S["tsha"], "fixed_params": pr.FIXED_PARAMS,
           "reproduction": {"block_a_max_per_event_difference": check, "full_frame": rep}, "gates": gates, "r3_detail": clean({**d1, **d2}),
           "runs": clean(detail), "stats": stats, "events": {k: {"ref": micro(a), "other": micro(b)} for k, (a, b) in allp.items()},
           "readings_reported": {"B_holds_in_every_run": all(d["reading_B_holds"] for d in detail.values()), "C_holds_in_every_run": all(d["reading_C_holds"] for d in detail.values())},
           "limitations": sorted(k for k, d in detail.items() if d["limitation"]), "outcome": pr.NOT_PROMOTED if stop else None,
           "next": "no exporter; the champion stays registered" if stop else "the exporter pull request (export gates E1 to E4)",
           "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__}, "seconds": round(time.time() - t0)}
    print("gates:", gates, "| limitations:", res["limitations"], "| outcome:", res["outcome"] or "statistical gates passed; export gates next", flush=True)
    return clean(res)

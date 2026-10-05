"""Build docs/PHASE_C_RESULTS.json from the logged MLflow runs and render docs/PHASE_C_CLOSURE.md (protocol Section 16.7).
Run from the repository root; the DagsHub token is read through a hidden prompt."""
import base64
import getpass
import json
import sys
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.closure import render

BASE_URL = "https://dagshub.com/jmskoero/nyando-flood-ai.mlflow/api/2.0/mlflow/"
REGISTRATION = "54afe7e812fd4c4985280283d52f3793"
BOOSTERS = {"mappable": "620e9d6219b64bc5818ba0fba82d92d7", "full": "b74d0c183b914b298a6b404e3d49718a"}


def call(path, auth, params=None, body=None):
    url = BASE_URL + path + ("?" + urllib.parse.urlencode(params) if params else "")
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Authorization": auth, "User-Agent": "nyando-results", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)


def run_record(auth, run_id, keep=None):
    r = call("runs/get", auth, {"run_id": run_id})["run"]
    metrics = {m["key"]: float(m["value"]) for m in r["data"].get("metrics", [])}
    if keep is not None:
        missing = sorted(set(keep) - set(metrics))
        if missing:
            raise SystemExit("run %s lacks metrics %s" % (run_id, missing))
        metrics = {k: v for k, v in metrics.items() if k in keep}
    params = {p["key"]: p["value"] for p in r["data"].get("params", [])}
    tags = {t["key"]: t["value"] for t in r["data"].get("tags", [])}
    return {"id": run_id, "status": r["info"]["status"], "commit": params.get("git_commit_sha", ""),
            "training_data_sha256": params.get("training_data_sha256", ""), "registered_arm": tags.get("registered_arm"), "metrics": metrics}


def latest_robustness(auth):
    res = call("runs/search", auth, body={"experiment_ids": ["0"], "filter": "tags.arm = 'robustness'",
                                          "order_by": ["attributes.start_time DESC"], "max_results": 10})
    runs = [r for r in res.get("runs", []) if r["info"]["status"] == "FINISHED"]
    if not runs:
        raise SystemExit("no FINISHED robustness run found in experiment 0: run the battery first")
    return runs[0]["info"]["run_id"]


def artifact_readback(auth, run_id, file_name):
    try:
        listed = [f["path"] for f in call("artifacts/list", auth, {"run_id": run_id, "path": "registered_model"}).get("files", [])]
    except Exception as e:
        return {"run": run_id, "listed": [], "present": False, "error": type(e).__name__}
    return {"run": run_id, "listed": listed, "present": any(p.endswith(file_name) for p in listed)}


def main():
    models = json.loads((ROOT / "models" / "MANIFEST.json").read_text(encoding="utf-8"))
    active = [(k, v) for k, v in models.items() if v.get("status") == "active"]
    assert len(active) == 1, "expected exactly one active model"
    rel, entry = active[0]
    data = json.loads((ROOT / "data" / "MANIFEST.json").read_text(encoding="utf-8"))
    train = [v["sha256"] for v in data.values() if isinstance(v, dict) and "label_source" in v][0]
    tok = getpass.getpass("DagsHub token (hidden): ")
    auth = "Basic " + base64.b64encode(("jmskoero:" + tok).encode()).decode()
    del tok
    rob_id = latest_robustness(auth)

    def keep(fr):
        return {"%s/hgb_con/per_event_mean" % fr, "%s/xgb_con/per_event_mean" % fr, "%s/xgb_con_vs_hgb_con/ci_low" % fr}

    runs = {"registration": run_record(auth, REGISTRATION), "robustness": run_record(auth, rob_id),
            "boosters_mappable": run_record(auth, BOOSTERS["mappable"], keep("mappable")),
            "boosters_full": run_record(auth, BOOSTERS["full"], keep("full"))}
    for name, rec in runs.items():
        if rec["status"] != "FINISHED" or rec["training_data_sha256"] != train:
            raise SystemExit("%s: run %s is not FINISHED with the manifest's training-data hash" % (name, rec["id"]))
    results = {"schema": 1, "date": date.today().isoformat(), "registered_model": {"file": rel, "sha256": entry["sha256"], "arm": "logistic:con"},
               "runs": runs, "artifact_readback": artifact_readback(auth, REGISTRATION, Path(rel).name)}
    (ROOT / "docs" / "PHASE_C_RESULTS.json").write_text(json.dumps(results, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    doc = render(results)
    (ROOT / "docs" / "PHASE_C_CLOSURE.md").write_text(doc, encoding="utf-8")
    print("robustness run:", rob_id, "| artifact read-back present:", results["artifact_readback"]["present"], "\n")
    print("\n".join(doc.splitlines()[:14]))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""scripts/check_manifests.py -- independent manifest verification (Roadmap Section 2, manifest-check.yml).

Standard library only. Recomputes every hash itself; it never trusts a manifest written by the step
that added the file. Checks, over the files tracked by git:
  1. every entry in data/MANIFEST.json points at a real file whose SHA-256 matches;
  2. every data/training/*.csv has an entry;
  3. every .pkl / .onnx has an entry in models/MANIFEST.json, with a matching SHA-256;
  4. a non-legacy model is named <name>_<algo>_<sha256 prefix>.<ext> (no _v1/_v3 labels), records its
     algorithm, and names a training-data hash that exists in data/MANIFEST.json;
  5. two different binaries never share one file name (the bug that started this rebuild).

Run with --init-legacy once to record files that predate V2 as status "legacy" (their hashes are
computed here, never typed by hand).
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_MANIFEST = ROOT / "data" / "MANIFEST.json"
MODEL_MANIFEST = ROOT / "models" / "MANIFEST.json"
MODEL_EXT = (".pkl", ".onnx")
NAME_RE = re.compile(r"^[a-z0-9]+_[a-z0-9]+_([0-9a-f]{6,12})\.(pkl|onnx)$")
VERSION_RE = re.compile(r"_v\d+")
STATUSES = {"active", "legacy", "retired"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tracked_files():
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        return [line for line in out.splitlines() if line]
    except Exception:
        return [p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*")
                if p.is_file() and ".git" not in p.parts]


def load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def verify():
    errors = []
    data, models, files = load(DATA_MANIFEST), load(MODEL_MANIFEST), tracked_files()

    for rel, e in data.items():
        p = ROOT / rel
        if not p.exists():
            errors.append(f"data manifest lists {rel}, but that file does not exist")
        elif sha256(p) != e.get("sha256"):
            errors.append(f"{rel}: SHA-256 differs from data/MANIFEST.json")
    for rel in files:
        if rel.startswith("data/training/") and rel.endswith(".csv") and rel not in data:
            errors.append(f"{rel}: training file has no entry in data/MANIFEST.json")
    for rel, e in data.items():
        for field, needed in (("sets_table", True), ("dropped_log", e.get("dropped_points", 0) > 0)):
            if e.get(field) and needed and e[field] not in data:
                errors.append(f"{rel}: {field} {e[field]} has no entry in data/MANIFEST.json")
    data_hashes = {e.get("sha256") for e in data.values()}

    artifacts = [f for f in files if f.endswith(MODEL_EXT)]
    for rel in artifacts:
        if rel not in models:
            errors.append(f"{rel}: model artifact has no entry in models/MANIFEST.json")
    for rel, e in models.items():
        p = ROOT / rel
        digest = sha256(p) if p.exists() else None
        if digest is None:
            errors.append(f"models manifest lists {rel}, but that file does not exist")
        elif digest != e.get("sha256"):
            errors.append(f"{rel}: SHA-256 differs from models/MANIFEST.json")
        status = e.get("status")
        if status not in STATUSES:
            errors.append(f"{rel}: status must be one of {sorted(STATUSES)}, got {status!r}")
        elif status == "legacy":
            if not e.get("note"):
                errors.append(f"{rel}: a legacy entry needs a note")
        else:
            name = Path(rel).name
            m = NAME_RE.match(name)
            if VERSION_RE.search(name):
                errors.append(f"{rel}: version label in the file name; use <name>_<algo>_<hash prefix>")
            elif not m:
                errors.append(f"{rel}: name must be <name>_<algo>_<6-12 hex>.<pkl|onnx>")
            elif digest and not digest.startswith(m.group(1)):
                errors.append(f"{rel}: the hash in the file name is not a prefix of the file's SHA-256")
            if not e.get("algorithm"):
                errors.append(f"{rel}: algorithm is not recorded")
            if e.get("trained_on") not in data_hashes:
                errors.append(f"{rel}: trained_on is not a SHA-256 listed in data/MANIFEST.json")

    by_name = {}
    for rel, e in models.items():
        by_name.setdefault(Path(rel).name, []).append(e)
    for name, group in by_name.items():
        if len({g.get("sha256") for g in group}) > 1 and any(g.get("status") != "legacy" for g in group):
            errors.append(f"{name}: two different binaries share one file name")

    if errors:
        print(f"manifest-check FAILED ({len(errors)} problem(s)):")
        for line in errors:
            print("  -", line)
        return 1
    print(f"manifest-check OK: {len(data)} data entries, {len(models)} model entries, {len(artifacts)} model artifacts")
    return 0


def init_legacy():
    data, models, files = load(DATA_MANIFEST), load(MODEL_MANIFEST), tracked_files()
    today = datetime.now(timezone.utc).date().isoformat()
    added = []
    companions = {e[f]: k for k, e in data.items() for f in ("sets_table", "dropped_log") if e.get(f)}
    for rel in files:
        p = ROOT / rel
        if rel in companions and rel not in data:
            data[rel] = {"sha256": sha256(p), "date_added": today, "status": "companion", "companion_of": companions[rel]}
            added.append(rel)
        elif rel.startswith("data/training/") and rel.endswith(".csv") and rel not in data:
            data[rel] = {"sha256": sha256(p), "date_added": today, "status": "legacy-discredited",
                         "note": "pre-V2 training file kept for history only; not used for V2 training (see CHANGES.md)"}
            added.append(rel)
        if rel.endswith(MODEL_EXT) and rel not in models:
            models[rel] = {"sha256": sha256(p), "size_bytes": p.stat().st_size, "date_added": today,
                           "status": "legacy", "algorithm": "unverified", "trained_on": None,
                           "note": "pre-V2 artifact with no recorded training-data hash; kept until a V2 model replaces it"}
            added.append(rel)
    save(DATA_MANIFEST, data)
    save(MODEL_MANIFEST, models)
    print("legacy entries added:", added or "none")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-legacy", action="store_true")
    sys.exit(init_legacy() if ap.parse_args().init_legacy else verify())

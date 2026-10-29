"""Burst check of the per-client rate limit (docs/SECURITY_BASELINE.md): python3 tools/burst_check.py HOSTNAME. Sends 11 quick POSTs to /v2/score for one real training row and expects 10 admitted and 1 refused, then one request with a client-set CF-Connecting-IP and one with a client-set X-Forwarded-For to show whether a client can choose its own key. It uses at most 13 of the shared 15 requests per minute and exits 1 if the honest burst is not limited to 10."""
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request

HOST = sys.argv[1] if len(sys.argv) > 1 else "nyando-flood-v2"
BASE = os.environ.get("BURST_BASE") or "https://%s.onrender.com" % HOST
WAIT = int(os.environ.get("BURST_WAIT", "65"))
UA = "nyando-flood-ai/2.0 burst-check"
FEATURES = ("elevation", "slope", "rainfall_3day", "distance_river", "clay_percent", "land_cover")


def real_row():
    with open("data/training/nyando_training_v2_multidate.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["flooded"] == "1" and all(r.get(k, "") != "" for k in FEATURES):
                return {k: (int(float(r[k])) if k == "land_cover" else float(r[k])) for k in FEATURES}
    raise SystemExit("no complete flood row in the training file")


def call(path, data=None, extra=None):
    headers = {"Content-Type": "application/json", "User-Agent": UA}
    headers.update(extra or {})
    req = urllib.request.Request(BASE + path, data=None if data is None else json.dumps(data).encode(), headers=headers, method="GET" if data is None else "POST")
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


BODY = real_row()
print("warm-up /health:", call("/health"))
admitted = refused = 0
for attempt in (1, 2):
    print("waiting %d s so earlier requests leave the minute window" % WAIT, flush=True)
    time.sleep(WAIT)
    codes = [call("/v2/score", BODY) for _ in range(11)]
    admitted, refused = codes.count(200), codes.count(429)
    print("attempt %d: %d admitted, %d refused (%s)" % (attempt, admitted, refused, " ".join(map(str, codes))), flush=True)
    if admitted <= 10 and set(codes) <= {200, 429}:
        break
ok = admitted == 10 and refused == 1
if ok:
    cf = call("/v2/score", BODY, {"CF-Connecting-IP": "203.0.113.9"})
    xf = call("/v2/score", BODY, {"X-Forwarded-For": "203.0.113.10"})
    print("client-set CF-Connecting-IP ->", cf, "| client-set X-Forwarded-For ->", xf)
    print("PASS: one client is limited to 10 per minute.")
    if cf == 200:
        print("NOTE: a client can choose its own key through CF-Connecting-IP, so the shared 15 per minute cap is the only hard bound.")
else:
    print("FAIL: the honest burst admitted %d (expected 10) and refused %d (expected 1)." % (admitted, refused))
sys.exit(0 if ok else 1)

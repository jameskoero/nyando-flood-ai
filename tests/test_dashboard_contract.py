"""The dashboard contract (README, Phase F): the dashboard calls /v2/advisory with exactly the inputs the API accepts, offers only the land-cover classes the model was trained on, takes its defaults and slider ranges from the real training data, and carries no legacy wording, legacy thresholds or hardcoded metrics."""
import ast
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = (ROOT / "frontend" / "src" / "App.jsx").read_text(encoding="utf-8")
FEATURES = ("elevation", "slope", "rainfall_3day", "distance_river", "clay_percent")


def block(start):
    lines = APP.split("\n")
    i = next(k for k, l in enumerate(lines) if l.strip().startswith(start))
    depth, out = 0, []
    for l in lines[i:]:
        out.append(l)
        depth += l.count("{") - l.count("}")
        if depth == 0:
            break
    return "\n".join(out)


def training_rows():
    with open(ROOT / "data" / "training" / "nyando_training_v2_multidate.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def backend_tree(name):
    return ast.parse((ROOT / "backend" / name).read_text(encoding="utf-8"))


def trained_classes():
    for n in backend_tree("registered.py").body:
        if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "LAND_COVER_CLASSES" for t in n.targets):
            return set(ast.literal_eval(n.value))
    raise AssertionError("LAND_COVER_CLASSES not found")


def api_input_fields():
    cls = next(n for n in backend_tree("main.py").body if isinstance(n, ast.ClassDef) and n.name == "ScoreInput")
    return {n.target.id for n in cls.body if isinstance(n, ast.AnnAssign)}


def defaults():
    return {m.group(1): m.group(2).strip('"') for m in re.finditer(r"^\s*(\w+):\s*([\d.]+|\"[^\"]*\"),", block("const DEFAULTS = {"), re.M)}


def ranges():
    out = {}
    for m in re.finditer(r"^\s*(\w+):\s*\{.*?min: ([\d.]+), max: ([\d.]+), step: ([\d.]+)", block("const FEATURE_META = {"), re.M):
        out[m.group(1)] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    return out


def test_the_dashboard_calls_the_v2_advisory_endpoint_and_nothing_legacy():
    assert 'const API_URL = "https://nyando-flood-v2.onrender.com/v2/advisory";' in APP
    assert "nyando-flood-api" not in APP and "/predict" not in APP


def test_no_legacy_wording_thresholds_or_hardcoded_metrics():
    for bad in ("Legacy model", "LEGACY MODEL", "NOT VALIDATED", "not validated", "GradientBoosting", "flood_probability"):
        assert bad not in APP, bad
    assert not re.search(r"(>=|<=|>|<)\s*0\.(35|60|65|80)\b", APP), "a legacy probability cutoff"
    assert not re.search(r"\b(AUC|F1|accuracy|precision|recall)\b", APP, re.I)


def test_the_request_body_has_exactly_the_inputs_the_api_accepts():
    keys = set(re.findall(r"^\s*(\w+):", block("const body = {"), re.M))
    assert keys == api_input_fields(), (keys, api_input_fields())


def test_land_cover_options_are_exactly_the_trained_classes_and_thin_classes_say_so():
    keys = {int(k) for k in re.findall(r"^\s*(\d+):", block("const LAND_COVER_LABELS = {"), re.M)}
    assert keys == trained_classes(), (keys, trained_classes())
    counts = {}
    for r in training_rows():
        counts[int(float(r["land_cover"]))] = counts.get(int(float(r["land_cover"])), 0) + 1
    labels = dict((int(k), v) for k, v in re.findall(r"^\s*(\d+):\s*\"([^\"]*)\"", block("const LAND_COVER_LABELS = {"), re.M))
    for k, n in counts.items():
        assert (n < 30) == ("few training examples" in labels[k]), (k, n, labels[k])


def test_defaults_are_the_reference_medians_and_inside_the_slider_ranges():
    ref = json.loads((ROOT / "docs" / "PHASE_E_REFERENCE.json").read_text(encoding="utf-8"))
    d, r = defaults(), ranges()
    for k in FEATURES:
        med = ref["median"][k]
        assert abs(float(d[k]) - med) <= 0.005 * max(1.0, abs(med)), (k, d[k], med)
        assert r[k][0] <= float(d[k]) <= r[k][1], k
    assert int(d["land_cover"]) == ref["land_cover_mode"]
    assert d["ward"] in re.findall(r"\"([^\"]+)\"", block("const WARDS = ["))


def test_slider_ranges_cover_the_real_training_range():
    rows, r = training_rows(), ranges()
    assert set(r) == set(FEATURES)
    for k in FEATURES:
        values = [float(x[k]) for x in rows if x[k] != ""]
        assert r[k][0] <= min(values) and r[k][1] >= max(values), (k, r[k], min(values), max(values))


def test_the_wards_are_the_real_ward_names():
    wards = set(re.findall(r"\"([^\"]+)\"", APP.split("const WARDS = [", 1)[1].split("]", 1)[0]))
    geo = json.loads((ROOT / "data" / "external" / "nyando_wards.geojson").read_text(encoding="utf-8"))["features"]
    by_key = {}
    for f in geo:
        for k, v in f["properties"].items():
            by_key.setdefault(k, set()).add(v)
    assert any(wards == vals for vals in by_key.values()), wards


def test_errors_and_the_client_rate_guard_follow_the_api():
    assert "res.status === 429" in APP and "res.status === 503" in APP and "CALL_LOG.length >= 5" in APP
    main = (ROOT / "backend" / "main.py").read_text(encoding="utf-8")
    assert re.search(r'@app\.post\("/v2/advisory"\)\s*\n@limiter\.limit\("5/minute"\)', main)


def test_the_applicability_tails_are_the_real_5th_and_95th_percentiles():
    import statistics

    tails = {}
    for m in re.finditer(r"^\s*(\w+):\s*\[([\d.]+),\s*([\d.]+)\],?", block("const TRAINING_TAILS = {"), re.M):
        tails[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    assert set(tails) == set(FEATURES), tails
    rows = training_rows()
    for k in FEATURES:
        q = statistics.quantiles(sorted(float(x[k]) for x in rows if x[k] != ""), n=20)
        for got, want in zip(tails[k], (q[0], q[-1])):
            assert abs(got - want) <= 0.001 * max(1.0, abs(want)), (k, got, want)

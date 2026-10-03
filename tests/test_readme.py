"""tests/test_readme.py -- the README as a tested document (docs as code).

Every number, path and claim in README.md that can drift is checked here against the repo, so an
edit to the data, the manifests or the roadmap that makes the README false turns CI red.
Standard library plus numpy/scikit-learn only; no Earth Engine, no network, no secrets.
"""

import csv
import json
import math
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REQUIRED_SECTIONS = [
    "Status", "What this is and is not", "Data (V2)", "How the dataset was built", "Findings so far",
    "Known limitations", "Integrity controls", "Quick start", "API (legacy model)", "Repository layout",
    "Roadmap", "License and data terms",
]
RETRACTED = ("0.9717", "0.9022", "0.9727")
FORBIDDEN = ("todo", "tbd", "lorem", "fully compliant", "gdpr", "72-hour", "100m resolution", "kes 500m")
PATH_EXT = (".py", ".md", ".json", ".csv", ".yml", ".geojson", ".pkl", ".onnx")


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this test never touches Earth Engine."""
    yield


def _readme():
    return (REPO / "README.md").read_text(encoding="utf-8")


def _headings(text):
    return [m.group(1).strip() for m in re.finditer(r"^#{1,6}\s+(.+?)\s*$", text, re.M)]


def _slug(heading):
    s = re.sub(r"[^\w\s-]", "", heading.lower())
    return re.sub(r"\s", "-", s.strip())


def _entry():
    manifest = json.loads((REPO / "data" / "MANIFEST.json").read_text())
    entries = [(k, v) for k, v in manifest.items() if isinstance(v, dict) and "label_source" in v]
    assert len(entries) == 1, "README describes exactly one current training file"
    return entries[0]


def _rows():
    rel, entry = _entry()
    with open(REPO / rel, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _outside_code_blocks(text):
    return re.sub(r"```.*?```", "", text, flags=re.S)


def test_required_sections_exist_in_order():
    heads = _headings(_readme())
    pos = [heads.index(h) for h in REQUIRED_SECTIONS if h in heads]
    assert len(pos) == len(REQUIRED_SECTIONS), f"missing sections: {[h for h in REQUIRED_SECTIONS if h not in heads]}"
    assert pos == sorted(pos), "sections are out of order"


def test_internal_anchors_resolve():
    text = _outside_code_blocks(_readme())
    slugs = {_slug(h) for h in _headings(text)}
    bad = [a for a in re.findall(r"\]\(#([^)]+)\)", text) if a not in slugs]
    assert not bad, f"anchors with no matching heading: {bad}"


def test_relative_links_and_inline_paths_exist():
    text = _outside_code_blocks(_readme())
    links = [l for l in re.findall(r"\]\(([^)#\s]+)\)", text) if not l.startswith(("http://", "https://", "mailto:"))]
    tokens = [t for t in re.findall(r"`([^`\s]+)`", text)
              if not t.startswith(("/", "%")) and not any(c in t for c in "*=(){}<>")
              and ("/" in t or t.endswith(PATH_EXT))]
    missing = [p for p in links + tokens if not (REPO / p).exists()]
    assert not missing, f"README points at files that do not exist: {sorted(set(missing))}"


def test_external_links_use_https():
    assert not re.findall(r"\]\((http://[^)]+)\)", _readme()), "use https links"


def test_code_blocks_have_a_language():
    fences = re.findall(r"^```(\S*)\s*$", _readme(), re.M)
    assert len(fences) % 2 == 0, "unbalanced code fences"
    opening = fences[0::2]
    assert all(opening), "every code block needs a language tag"


def test_no_placeholders_or_unverifiable_claims():
    low = _readme().lower()
    found = [w for w in FORBIDDEN if w in low]
    assert not found, f"README contains placeholder or unverifiable wording: {found}"


def test_retracted_figures_appear_only_in_the_correction_notice():
    text = _readme()
    lines = text.splitlines()
    idx = next(i for i, l in enumerate(lines) if l.startswith("> **Correction notice"))
    assert idx < 15, "the correction notice must be near the top"
    rest = "\n".join(lines[:idx] + lines[idx + 1:])
    assert not [f for f in RETRACTED if f in rest], "a retracted figure appears outside the notice"


def test_status_matches_the_model_manifest():
    models = json.loads((REPO / "models" / "MANIFEST.json").read_text())
    other = sorted(k for k, e in models.items() if e.get("status") not in ("legacy", "active"))
    assert not other, "model statuses other than legacy and active need a README rule: %s" % other
    active = {k: e for k, e in models.items() if e.get("status") == "active"}
    text = _readme()

    def section(title):
        m = re.search(r"^## %s\s*$" % re.escape(title), text, re.M)
        assert m, title
        nxt = re.search(r"^## ", text[m.end():], re.M)
        return text[m.end(): m.end() + nxt.start()] if nxt else text[m.end():]

    status, api, roadmap = section("Status"), section("API (legacy model)"), section("Roadmap")
    if active:
        assert len(active) == 1, "README describes exactly one registered model"
        rel, e = next(iter(active.items()))
        name = Path(rel).name
        assert e["sha256"].startswith(name.rsplit("_", 1)[1].split(".")[0]), name
        assert name in status, "the README Status must name the registered model " + name
        low = status.lower()
        assert "exclusion mask" in low and "not flood probabilities" in low, "the README Status must carry the claim limit"
        assert "not served" in api.lower(), "the README API section must say the registered model is not served yet"
    c_done = re.search(r"^- \[x\] \*\*C\. ", roadmap, re.M) is not None
    assert c_done == (REPO / "docs" / "PHASE_C_CLOSURE.md").exists(), "Roadmap item C is checked only with docs/PHASE_C_CLOSURE.md"
    b_done = re.search(r"^- \[x\] \*\*B\. ", roadmap, re.M) is not None
    assert b_done == (REPO / "src" / "tracking.py").exists(), "Roadmap item B is checked only with src/tracking.py"
def test_numbers_match_the_data():
    rel, entry = _entry()
    rows = _rows()
    text = _readme()
    flooded = [r for r in rows if r["flooded"] == "1"]
    types = entry["sample_types"]
    blank = sum(1 for r in rows if r["clay_percent"] == "")
    blank_flood = sum(1 for r in flooded if r["clay_percent"] == "") / len(flooded)
    blank_ctrl = sum(1 for r in rows if r["flooded"] == "0" and r["clay_percent"] == "") / (len(rows) - len(flooded))
    floor = min(float(r["elevation"]) for r in rows)
    at_floor = [r for r in rows if float(r["elevation"]) == floor]
    awasi = sum(1 for r in flooded if r["ward"] == "Awasi/Onjiko")
    expect = {
        "rows (table)": f"| Rows | {len(rows):,} |", "rows (status)": f"{len(rows):,} rows from {entry['sample_sets']} Sentinel-1 scene dates",
        "sets": f"| Scene dates (sample sets) | {entry['sample_sets']} (5 flood events + 30 screened dates) |",
        "cases/controls": f"| Flood cases / controls | {types['flood_case']:,} / {types['flood_control']:,} |",
        "blank clay": f"| Rows with blank clay | {blank:,} |",
        "flood share": f"{entry['flood_rate'] * 100:.1f}%", "patches": f"{entry['flood_patches_total']:,}",
        "dem floor": f"{len(at_floor)} rows ({len(at_floor) / len(rows) * 100:.1f}%)",
        "floor elevation": f"({floor:.1f} m)", "floor flooded": f"{round(sum(r['flooded'] == '1' for r in at_floor) / len(at_floor) * 100)}% of them",
        "blank shares": f"blank on {round(blank_flood * 100)}% of flood cases but only {blank_ctrl * 100:.1f}% of controls",
        "awasi": f"Awasi/Onjiko has only {awasi} flood cases",
    }
    missing = {k: v for k, v in expect.items() if v not in text}
    assert not missing, f"README numbers differ from the data: {missing}"


def test_rainfall_finding_matches_the_sets_table():
    _, entry = _entry()
    with open(REPO / entry["sets_table"], newline="", encoding="utf-8") as f:
        sets = [r for r in csv.DictReader(f) if r["sample_mode"] == "date_mosaic"]
    wet = [r for r in sets if float(r["rain_basin_14d"]) >= 90]
    dry = [r for r in sets if float(r["rain_basin_14d"]) < 90]
    fw = sum(float(r["flood_px"]) > 0 for r in wet)
    fd = sum(float(r["flood_px"]) > 0 for r in dry)
    n, k, m = len(sets), fw + fd, len(wet)
    p = sum(math.comb(k, x) * math.comb(n - k, m - x) for x in range(fw, min(k, m) + 1)) / math.comb(n, m)
    text = _readme()
    for needle in (f"{fw} of {len(wet)} dates", f"{fd} of {len(dry)} below 90 mm", f"p = {p:.4f}"):
        assert needle in text, f"README rainfall statement differs from the sets table: {needle!r}"


def test_model_free_statements_recompute():
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rows = _rows()
    y = np.array([int(r["flooded"]) for r in rows])
    col = lambda c: np.array([float(r[c]) for r in rows])
    groups = np.array([r["event_id"] for r in rows])
    x4 = np.column_stack([col("elevation"), col("slope"), col("rainfall_3day"), col("distance_river")])
    p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)), x4, y,
                          cv=LeaveOneGroupOut(), groups=groups, method="predict_proba")[:, 1]
    auc4 = roc_auc_score(y, p)
    a_rain = roc_auc_score(y, col("rainfall_3day"))
    a_elev = roc_auc_score(y, col("elevation"))
    text = _readme()
    assert abs(auc4 - 0.841) < 0.002 and "scores 0.841 leave-one-event-out" in text, f"bare-4 AUC is now {auc4:.3f}"
    assert f"AUC {max(a_rain, 1 - a_rain):.2f}" in text and f"reaches {max(a_elev, 1 - a_elev):.2f}" in text


def test_license_statement_is_consistent():
    text = _readme()
    assert "CC BY-NC 4.0 or ODbL 1.0" in text and "cannot be published under CC BY 4.0" in text
    assert "all outputs" not in text.lower()


def test_quick_start_output_matches_the_gate_file():
    gate = (REPO / "tests" / "test_data_gate.py").read_text(encoding="utf-8")
    total = len(re.findall(r"^def test_", gate, re.M))
    xfail = gate.count("@pytest.mark.xfail")
    assert f"{total - xfail} passed, {xfail} xfailed" in _readme(), "the gate's test count changed: update the Quick start"

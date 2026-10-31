"""D37 (docs/ROADMAP_DEVIATIONS.md): the datasheet and the README state the licence of the data files, the terms and credits of every source, and the register row is closed on this file. Standard library only."""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHEET = (ROOT / "data" / "DATA_SOURCES.md").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")
REGISTER = (ROOT / "docs" / "ROADMAP_DEVIATIONS.md").read_text(encoding="utf-8")
PLACEHOLDERS = ("see the provider's terms", "see the catalogue entry")
SOURCES = {
    "flooded (label)": "Copernicus Emergency Management Service terms",
    "elevation, slope, HAND": "ODbL 1.0",
    "distance_river": "Copernicus Programme",
    "clay_percent": "CC BY 4.0",
    "land_cover": "CC-BY-4.0",
    "rainfall_3day and the rainfall aggregates": "public domain",
    "upstream basin for the rainfall aggregates": "HydroSHEDS version 1 licence agreement",
}


def _section(text, start, end):
    m = re.search(r"^%s.*?(?=^%s)" % (start, end), text, re.M | re.S)
    assert m, start
    return m.group(0)


def _terms():
    rows = {}
    for line in _section(SHEET, r"## 4\. ", r"### 4\.1 ").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and len(cells) == 4 and cells[0] not in ("Quantity", "---"):
            rows[cells[0]] = cells[3]
    return rows


def _years():
    ys = set()
    for rel in ("data/training/nyando_training_v2_multidate.csv", "data/confirmatory/nyando_block_a.csv"):
        with open(ROOT / rel, newline="", encoding="utf-8") as f:
            ys |= {r["event_date"][:4] for r in csv.DictReader(f)}
    return "%s-%s" % (min(ys), max(ys))


def test_the_datasheet_states_the_data_licence_and_every_required_credit():
    s5 = _section(SHEET, r"## 5\. ", r"## 6\. ")
    assert "open decision" not in s5 and "register row D37 (closed)" in s5 and "not legal advice" in s5 and "2026-10-31" in s5
    for must in ("ODbL 1.0", "https://opendatacommons.org/licenses/odbl/1.0/", "nyando_training_v2_multidate.csv", "nyando_block_a", "data/derived/", "CC BY 4.0", "nyando_wards.geojson", "Not covered", "nyando_training_v1.csv", "Code: MIT"):
        assert must in s5, must
    s41 = _section(SHEET, r"### 4\.1 ", r"### 4\.2 ")
    for must in ("Contains modified Copernicus Emergency Management Service information " + _years(), "doi:10.1029/2019WR024873", "doi:10.1038/nature20584", "doi:10.5194/soil-7-217-2021", "doi:10.1038/sdata.2015.66",
                 "This product (the Nyando Flood AI data files) incorporates data from the HydroSHEDS version 1 database", "Lehner, B., Verdin, K., Jarvis, A. (2008)", "no HydroSHEDS layer, geometry or basin identifier"):
        assert must in s41, must
    s42 = _section(SHEET, r"### 4\.2 ", r"## 5\. ")
    for must in ("Benedict Aboki Omare and George Omare", "2017", "https://github.com/benaboki/Kenya-County-Assembly-Boundaries", "https://creativecommons.org/licenses/by/4.0/", "modified subset", "5765ef5", "Kisumu", "Nyando"):
        assert must in s42, must


def test_every_source_row_records_its_terms_and_none_is_a_placeholder():
    rows = _terms()
    assert set(rows) == set(SOURCES), sorted(set(rows) ^ set(SOURCES))
    for name, keyword in SOURCES.items():
        assert keyword in rows[name] and not any(p in rows[name] for p in PLACEHOLDERS), name
    assert "ODbL 1.0 chosen" in rows["elevation, slope, HAND"] and "Section 4.1" in rows["flooded (label)"]


def test_the_copernicus_notice_years_are_the_years_of_the_data():
    ys = _years()
    lo, hi = ys.split("-")
    assert re.fullmatch(r"20\d\d", lo) and re.fullmatch(r"20\d\d", hi) and lo <= hi


def test_the_readme_states_the_data_licence_and_keeps_its_heading():
    assert "## License and data terms" in README
    body = _section(README, r"## License and data terms", r"## Citation, security and contributing")
    for must in ("Code: MIT", "ODbL 1.0", "2026-10-31", "register row D37", "cannot be published under CC BY 4.0", "CC BY 4.0", "legacy v1 files", "data/DATA_SOURCES.md", "not legal advice"):
        assert must in body, must
    assert "still to be chosen" not in body and "treat the data as non-commercial" not in body


def test_the_register_closes_d37_on_this_file():
    rows = [l for l in REGISTER.splitlines() if l.startswith("| D37 |")]
    assert len(rows) == 1
    cells = [c.strip() for c in rows[0].strip().strip("|").split("|")]
    assert len(cells) == 7 and cells[6] == "closed" and cells[5] == "tests/test_data_licence.py::test_the_datasheet_states_the_data_licence_and_every_required_credit"
    assert "ODbL 1.0" in cells[3] and "MERIT" in rows[0] and "not legal advice" in cells[4]

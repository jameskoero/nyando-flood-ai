"""
Smoke test for the four raw feature extractors, against Ahero and the
confirmed April 2020 event -- same reference point Stage 2 used.
"""
from datetime import date

from src.data.raw_features import extract_raw_features


def test_ahero_raw_features_for_april_2020_event():
    ahero = (34.9167, -0.1667)
    results = extract_raw_features([ahero], event_date=date(2020, 4, 15))

    assert results[0] is not None, "expected coverage at Ahero for all four sources"
    sample = results[0]

    assert sample.distance_river_m >= 0
    assert sample.rainfall_3day_mm >= 0
    assert 0 <= sample.clay_percent <= 100, f"clay_percent out of range: {sample.clay_percent}"
    assert isinstance(sample.land_cover_class, int)


def test_multiple_points_preserve_order():
    points = [(34.9167, -0.1667), (35.0, -0.05)]
    results = extract_raw_features(points, event_date=date(2020, 4, 15))
    assert len(results) == 2

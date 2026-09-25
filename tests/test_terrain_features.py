"""
Smoke test for terrain feature extraction. Uses Ahero (a known low-lying
town in Nyando Constituency, near the river) as a physically-meaningful
check point rather than an arbitrary coordinate.
"""
from src.data.terrain_features import extract_terrain_features


def test_ahero_point_has_low_hand():
    ahero = (34.9167, -0.1667)
    results = extract_terrain_features([ahero])

    assert results[0] is not None, "expected MERIT/Hydro coverage at Ahero"
    sample = results[0]

    assert 0 <= sample.hand_m < 50, f"unexpectedly high HAND at a known lowland point: {sample.hand_m}"
    assert sample.elevation_m < 1400, f"unexpectedly high elevation for the Nyando floodplain: {sample.elevation_m}"


def test_multiple_points_preserve_order():
    points = [(34.9167, -0.1667), (35.0, -0.05)]
    results = extract_terrain_features(points)
    assert len(results) == 2

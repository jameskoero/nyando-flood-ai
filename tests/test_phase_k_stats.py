"""Phase K statistics (src/models/phase_k_stats.py): the interval and the gap on values whose answer is known, and the per-frame report on a small synthetic file where one ward cannot be estimated."""
import numpy as np

from src.models import phase_k_stats as S


def test_the_mean_interval_of_constant_values_is_that_value_and_empty_input_gives_none():
    m = S.mean_ci([0.5] * 6)
    assert m == {"mean": 0.5, "ci_low": 0.5, "ci_high": 0.5, "events": 6}
    assert S.mean_ci([])["mean"] is None
    v = S.mean_ci([0.2, 0.4, 0.9, 0.7, 0.6])
    assert v["ci_low"] <= v["mean"] <= v["ci_high"] and v["events"] == 5


def test_the_gap_recovers_a_known_offset_and_is_zero_for_equal_dictionaries():
    events = ["e%d" % i for i in range(8)]
    pooled = {e: 0.6 + 0.02 * i for i, e in enumerate(events)}
    g = S.gap_ci({e: v + 0.2 for e, v in pooled.items()}, pooled)
    assert abs(g["gap"] - 0.2) < 1e-9 and abs(g["ci_low"] - 0.2) < 1e-9 and abs(g["ci_high"] - 0.2) < 1e-9
    z = S.gap_ci(dict(pooled), pooled)
    assert abs(z["gap"]) < 1e-12 and abs(z["ci_low"]) < 1e-12 and abs(z["ci_high"]) < 1e-12
    assert S.gap_ci({}, pooled)["gap"] is None


def _frame(seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for ward, n_events, per in (("Ahero", 9, 60), ("Awasi/Onjiko", 9, 40), ("Kobura", 6, 50)):
        for e in range(n_events):
            for _ in range(per):
                flood = int(rng.random() < (0.5 if ward != "Awasi/Onjiko" else (0.02 if e == 0 else 0.0)))
                strong = 0.6 * flood + 0.4 * rng.random()
                score = strong if ward == "Ahero" else rng.random()
                rows.append((ward, "ev%d" % e, flood, score))
    wards, events, flooded, scores = (np.array([r[i] for r in rows]) for i in range(4))
    levels = np.where(scores.astype(float) < 0.34, "LOW", np.where(scores.astype(float) < 0.67, "MEDIUM", "HIGH"))
    return wards, events, flooded.astype(int), scores.astype(float), levels


def _aucs_fn(events, flooded, scores):
    def fn(mask):
        out = {}
        for e in sorted(set(events[mask])):
            m = mask & (events == e)
            y, s = flooded[m], scores[m]
            if y.sum() >= 1 and (1 - y).sum() >= 1:
                pos, neg = s[y == 1], s[y == 0]
                out[e] = int(round(float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()) * 1e6))
        return out
    return fn


def test_a_frame_report_estimates_the_wards_that_can_be_and_never_the_others():
    wards, events, flooded, scores, levels = _frame()
    rep = S.frame_report(wards, flooded, events, levels, _aucs_fn(events, flooded, scores))
    assert rep["rows"] == len(wards) and list(rep["wards"]) == list(S.K.WARDS)
    assert rep["wards"]["Awasi/Onjiko"]["estimable"] is False
    assert rep["wards"]["Awasi/Onjiko"]["flags"] == ["not_estimable"] and "auc" not in rep["wards"]["Awasi/Onjiko"]
    for ward in ("East Kano/Wawidhi", "Kabonyo/Kanyagwal"):
        assert rep["wards"][ward]["estimable"] is False and rep["wards"][ward]["flood_rows"] == 0
    ahero = rep["wards"]["Ahero"]
    assert ahero["estimable"] is True and ahero["events_both"] == 9
    assert ahero["auc"]["ci_low"] > 0.5 and "auc_not_above_chance" not in ahero["flags"] and "bands_not_separated" not in ahero["flags"]
    assert set(ahero) >= {"rates", "bands", "auc", "gap", "flags"}


def test_a_ward_whose_scores_carry_no_information_is_flagged_and_the_report_is_repeatable():
    wards, events, flooded, scores, levels = _frame()
    fn = _aucs_fn(events, flooded, scores)
    a, b = S.frame_report(wards, flooded, events, levels, fn), S.frame_report(wards, flooded, events, levels, fn)
    assert a == b
    kobura = a["wards"]["Kobura"]
    assert kobura["estimable"] is True and kobura["flags"] and set(kobura["flags"]) <= set(S.K.FLAGS)

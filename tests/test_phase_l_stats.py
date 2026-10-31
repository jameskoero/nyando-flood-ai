"""Phase L statistics (src/models/phase_l_stats.py): the vectorised within-class concordance equals the brute-force definition of src/models/phase_l.py on resampled data, including ties; the interval behaves on constructed cases."""
import numpy as np

from src.models import phase_l as L
from src.models import phase_l_stats as S


def _data(seed=3, n=140):
    rng = np.random.default_rng(seed)
    classes = rng.choice([10, 30, 40], size=n)
    events = np.array(["e%d" % i for i in rng.integers(0, 6, size=n)])
    flood = rng.random(n) < 0.45
    a = np.round(rng.random(n) * 0.6 + flood * 0.25, 1)
    b = np.round(rng.random(n) * 0.6 + flood * 0.10, 1)
    return a, b, flood, classes, events


def test_the_vectorised_w_equals_the_brute_force_definition_on_resampled_rows_with_ties():
    a, b, flood, classes, events = _data()
    ev = sorted(set(events.tolist()))
    mult = S.multiplicities(len(ev), 25, seed=7)
    for scores in (a, b):
        got = S.w_resamples(scores, flood, classes, events, [10, 30, 40], mult)
        for i in range(0, 25, 4):
            idx = np.concatenate([np.repeat(np.where(events == e)[0], mult[i, k]) for k, e in enumerate(ev)])
            want, _ = L.within_class_concordance(list(scores[idx]), [int(x) for x in flood[idx]], list(classes[idx]), use=[10, 30, 40])
            assert abs(got[i] - want) < 1e-12, (i, got[i], want)
    one = np.ones((1, len(ev)), dtype=np.int32)
    point, pairs = L.within_class_concordance(list(a), [int(x) for x in flood], list(classes))
    assert abs(S.w_resamples(a, flood, classes, events, [10, 30, 40], one)[0] - point) < 1e-12 and pairs == S.pair_count(flood, classes, [10, 30, 40])


def test_a_subset_of_classes_is_respected():
    a, b, flood, classes, events = _data(5)
    one = np.ones((1, len(set(events.tolist()))), dtype=np.int32)
    point, _ = L.within_class_concordance(list(a), [int(x) for x in flood], list(classes), use=[30])
    assert abs(S.w_resamples(a, flood, classes, events, [30], one)[0] - point) < 1e-12


def test_identical_arms_give_a_zero_interval_and_a_better_arm_gives_a_positive_one():
    a, b, flood, classes, events = _data(9, n=400)
    same = S.delta_w(a, a, flood, classes, events, [10, 30, 40], n_boot=600)
    assert same["delta"] == 0.0 and same["ci_low"] == 0.0 and same["ci_high"] == 0.0
    assert L.d14_outcome(same["ci_low"], same["ci_high"]) == "between_class_only"
    better = S.delta_w(a, b, flood, classes, events, [10, 30, 40], n_boot=600)
    assert better["delta"] > 0 and better["ci_low"] <= better["delta"] <= better["ci_high"] and better["draws_used"] <= 600
    assert L.d14_outcome(better["ci_low"], better["ci_high"]) in ("within_class_gain", "between_class_only")
    worse = S.delta_w(b, a, flood, classes, events, [10, 30, 40], n_boot=600)
    assert abs(worse["delta"] + better["delta"]) < 1e-12


def test_the_resampling_is_repeatable_and_uses_whole_events():
    m1, m2 = S.multiplicities(6, 50, seed=11), S.multiplicities(6, 50, seed=11)
    assert (m1 == m2).all() and (m1.sum(axis=1) == 6).all()

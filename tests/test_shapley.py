import itertools, os, random, sys
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis", "shap"))
from shapley import shapley, interaction, check_efficiency, ORDER

def subsets():
    return [frozenset(c) for k in range(len(ORDER) + 1) for c in itertools.combinations(ORDER, k)]

def test_additive_game_recovers_weights_and_zero_interaction():
    w = dict(zip(ORDER, [0.4, 0.1, 0.25, 0.05, 0.2]))
    v = {S: sum(w[p] for p in S) for S in subsets()}
    phi = shapley(v, ORDER)
    for p in ORDER:
        assert phi[p] == pytest.approx(w[p], abs=1e-12)
    check_efficiency(v, phi)
    assert interaction(v, ORDER, "char", "setting") == pytest.approx(0, abs=1e-12)

def test_pure_synergy_pair():
    v = {S: 1.0 if {"char", "setting"} <= S else 0.0 for S in subsets()}
    phi = shapley(v, ORDER)
    assert phi["char"] == pytest.approx(0.5) and phi["setting"] == pytest.approx(0.5)
    assert phi["tail"] == pytest.approx(0, abs=1e-12)
    assert interaction(v, ORDER, "char", "setting") == pytest.approx(1.0)

def test_efficiency_on_random_game():
    rng = random.Random(0)
    v = {S: rng.random() for S in subsets()}
    check_efficiency(v, shapley(v, ORDER))

def test_missing_coalition_raises():
    v = {S: 0.0 for S in subsets()}
    del v[frozenset(["char"])]
    with pytest.raises(ValueError):
        shapley(v, ORDER)
def test_records_from_score_py_items():
    from shapley import records_from
    recs = records_from({"items": [{"scene_id": "s09", "coalition": "char", "seed": 1, "checklist": 0.5}]})
    assert recs[0]["coalition"] == "char"

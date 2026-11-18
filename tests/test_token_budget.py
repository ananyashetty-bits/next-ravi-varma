import json, os, sys
import pytest
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from pipeline.token_budget import build, variant_parts, scene_means, VARIANTS, _cfg

SCENES = json.load(open(os.path.join(ROOT, "analysis", "shap", "shap_scenes.json"), encoding="utf-8"))


def test_shap_scenes_design():
    assert len(SCENES) == 8
    tiers = sorted(s["tier"] for s in SCENES)
    assert tiers == [1, 1, 2, 2, 3, 3, 4, 4]
    dev = {s["id"]: s for s in json.load(open(os.path.join(ROOT, "eval", "test_scenes_dev.json"), encoding="utf-8"))}
    for s in SCENES:
        assert dev[s["scene_id"]]["tier"] == s["tier"]
        assert set(s["players"]) == {"char", "action", "setting", "mood", "tail"}
        for phrase in s["players"].values():
            assert len(phrase.split()) <= 14
            for n in s["names"]:
                assert n.lower() not in phrase.lower()          # players are name-free


def test_build_variants():
    cfg = _cfg()
    recs = build(SCENES, cfg)
    assert len(recs) == 8 * len(VARIANTS)
    s = SCENES[0]
    by = {r["coalition"]: r["positive"] for r in recs if r["scene_id"] == s["scene_id"]}
    assert by["full"].startswith(cfg["trigger"]) and by["full"].endswith(s["players"]["tail"])
    assert by["trigger_last"].endswith(cfg["trigger"]) and not by["trigger_last"].startswith(cfg["trigger"])
    assert by["no_tail"] == by["full"][: -len(", " + s["players"]["tail"])]
    assert s["names"][0] in by["with_names"] and s["names"][0] not in by["full"]
    assert by["char"] == f"{cfg['trigger']}, {s['players']['char']}"
    assert all(r["negative"] == cfg["negative"] for r in recs)


def test_scene_means_uneven_seeds_raise():
    items = [{"scene_id": "s09", "coalition": "full", "seed": 1, "checklist": 0.5},
             {"scene_id": "s09", "coalition": "full", "seed": 2, "checklist": 0.7},
             {"scene_id": "s09", "coalition": "char", "seed": 1, "checklist": 0.4}]
    with pytest.raises(ValueError):
        scene_means(items, "checklist")
    m = scene_means(items[:2], "checklist")
    assert m["full"]["s09"] == pytest.approx(0.6)


def test_shap_coalitions_256():
    sys.path.insert(0, os.path.join(ROOT, "analysis", "shap"))
    from coalitions import coalition_prompts
    out = [p for s in SCENES for p in coalition_prompts(s)]
    assert len(out) == 256 and len({(p["scene_id"], p["coalition"]) for p in out}) == 256

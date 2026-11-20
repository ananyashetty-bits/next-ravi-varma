"""CI for run_batch's flag parsing and label override (A, Overview 6.5). CPU-only; loads batch_args by path."""
import importlib.util, json, os
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
P = lambda *a: os.path.join(ROOT, *a)
_spec = importlib.util.spec_from_file_location("batch_args", P("pipeline", "batch_args.py"))
BA = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(BA)

BASE = ["--seeds", "1", "2", "--out", "x"]
parse = lambda *a: BA.build_parser().parse_args(list(a))
fake_profile = lambda name: {"control_scale": 1.0, "guidance_start": 0.0, "guidance_end": 1.0, "lora_scale": None}


def test_condition_is_mandatory():
    with pytest.raises(SystemExit):
        parse("--expansions", "e.json", *BASE)


def test_all_flags_parse():
    a = parse("--expansions", "e.json", "--pairs", "p.json", "--condition", "p2_openpose_tuned", "--control", "openpose",
              "--control_profile", "tuned", "--control_source", "auto", "--control_scale", "0.8",
              "--guidance_start", "0.1", "--guidance_end", "0.7", "--lora_scale", "1.2", *BASE)
    assert (a.control, a.control_profile, a.control_source) == ("openpose", "tuned", "auto")
    assert (a.control_scale, a.guidance_start, a.guidance_end, a.lora_scale) == (0.8, 0.1, 0.7, 1.2)
    assert a.seeds == [1, 2] and a.pairs == "p.json"


def test_defaults_keep_m3_behaviour():
    a = parse("--expansions", "e.json", "--condition", "c1_lora_raw", *BASE)
    assert (a.control, a.control_profile, a.control_source) == ("none", "baseline", "pair")
    assert BA.settings(a, fake_profile) is None                       # no control settings for p0


def test_unknown_control_is_rejected():
    with pytest.raises(SystemExit):
        parse("--expansions", "e.json", "--condition", "x", "--control", "depth", *BASE)


def test_sweep_flags_override_the_profile():
    a = parse("--expansions", "e.json", "--pairs", "p.json", "--condition", "s", "--control", "openpose",
              "--control_scale", "0.6", "--lora_scale", "1.2", *BASE)
    s = BA.settings(a, fake_profile)
    assert s == {"control_scale": 0.6, "guidance_start": 0.0, "guidance_end": 1.0, "lora_scale": 1.2}


def test_validate():
    none_with_scale = parse("--expansions", "e.json", "--condition", "x", "--control_scale", "0.5", *BASE)
    assert BA.validate(none_with_scale)
    no_pairs = parse("--expansions", "e.json", "--condition", "x", "--control", "openpose", *BASE)
    assert BA.validate(no_pairs)
    ok = parse("--expansions", "e.json", "--pairs", "p.json", "--condition", "x", "--control", "openpose", *BASE)
    assert BA.validate(ok) is None
    p0 = parse("--expansions", "e.json", "--pairs", "p.json", "--condition", "p0", "--control", "none", *BASE)
    assert BA.validate(p0) is None


def test_label_overrides_the_records_condition():
    recs = json.load(open(P("fixtures", "expansions.json"), encoding="utf-8"))
    recs = recs if isinstance(recs, list) else recs.get("expansions", recs)
    assert recs, "fixtures/expansions.json is empty"
    items = BA.expansion_items(recs, "p1_openpose", None, "rrv_v1")
    assert len(items) == len(recs)
    assert {it["condition"] for it in items} == {"p1_openpose"}
    assert all(it["negative"] == r["negative"] and it["prompt"] == r["positive"] for it, r in zip(items, recs))


def test_filter_to_pairs():
    items = [{"scene_id": s, "name": s} for s in ("s01", "s02", "s03")]
    assert [i["scene_id"] for i in BA.filter_to_pairs(items, {"s01": {}, "s03": {}})] == ["s01", "s03"]
    with pytest.raises(ValueError):
        BA.filter_to_pairs(items, {"s01": {}, "s99": {}})

"""Unit tests for kb/parser.py (C). Run in CI. Gold dev-scene metrics are checked against H-C1 thresholds."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from kb.parser import parse_scene

def chars(text):
    return [c["canonical"] for c in parse_scene(text)["characters"]]

def test_exact_name():
    assert chars("Rama and Sita in the forest") == ["Rama", "Sita"]

def test_misspellings_resolve_to_canonical():
    assert chars("Arjun and Subhadra in a garden") == ["Arjuna", "Subhadra"]
    assert chars("Seetha waiting by a lake") == ["Sita"]
    assert chars("Duryodhan shouting in the court") == ["Duryodhana"]

def test_mention_keeps_users_spelling():
    spec = parse_scene("Seetha waiting by a lake")
    assert spec["characters"][0]["mention"] == "Seetha"
    assert spec["characters"][0]["canonical"] == "Sita"

def test_rama_not_matched_inside_ramayana():
    spec = parse_scene("A scene from the Ramayana with no named figures")
    assert spec["characters"] == []

def test_ramayana_is_not_an_unknown_entity():
    assert "Ramayana" not in parse_scene("Rama in the Ramayana")["unknown_entities"]

def test_unknown_character_is_reported_not_matched():
    spec = parse_scene("The sage Agastya and Gandhari in the forest")
    assert spec["characters"] == []
    assert "Agastya" in spec["unknown_entities"]

def test_bhima_is_not_bhishma():
    assert "Bhishma" not in chars("Bhima fighting in the forest")

def test_character_without_images_still_matches():
    spec = parse_scene("Karna weeping on the battlefield")
    assert spec["characters"][0]["canonical"] == "Karna"
    assert spec["characters"][0]["coverage"] == "none"

def test_setting_and_mood():
    spec = parse_scene("Sita weeping by a lake")
    assert spec["setting"]["name"] == "lake"
    assert spec["mood"] == "grief"

def test_no_characters_epic_unknown():
    assert parse_scene("A sad king sitting alone")["epic"] == "Unknown"

def test_epic_majority():
    assert parse_scene("Rama, Lakshmana and Sita crossing a river")["epic"] == "Ramayana"

def test_empty_and_odd_input_do_not_crash():
    for t in ["", "   ", "!!!", "x" * 500]:
        spec = parse_scene(t)
        assert spec["characters"] == []

def test_contract_fields_present():
    spec = parse_scene("Rama in the forest", "s99")
    for k in ["id", "raw", "epic", "characters", "setting", "action", "mood", "unknown_entities", "kb_version"]:
        assert k in spec
    assert spec["id"] == "s99" and spec["kb_version"].startswith("kb")

def test_kb_descriptions_contain_no_names():
    import re
    kb = json.load(open(os.path.join(ROOT, "kb", "kb.json"), encoding="utf-8"))
    names = json.load(open(os.path.join(ROOT, "kb", "names.json"), encoding="utf-8"))
    texts = [c["visual"] for c in kb["characters"].values()]
    texts += [c["visual_alt"]["visual"] for c in kb["characters"].values() if "visual_alt" in c]
    texts += [s["visual"] for s in kb["settings"].values()] + list(kb["moods"].values())
    for t in texts:
        assert len(t.split()) <= 14, t
        for n in names:
            assert not re.search(r"\b" + re.escape(n) + r"\b", t, re.I), (n, t)

def test_gold_dev_scenes_meet_hypothesis_thresholds():
    scenes = json.load(open(os.path.join(ROOT, "eval", "test_scenes_dev.json"), encoding="utf-8"))
    kb = json.load(open(os.path.join(ROOT, "kb", "kb.json"), encoding="utf-8"))
    canon = {a.lower(): c["canonical"].lower() for c in kb["characters"].values() for a in [c["canonical"], *c["aliases"]]}
    n = lambda x: canon.get(x.strip().lower(), x.strip().lower())
    tp = fp = fn = ok = tot = 0
    for s in scenes:
        spec = parse_scene(s["text"], s["id"])
        got, exp = {n(c["canonical"]) for c in spec["characters"]}, {n(x) for x in s.get("expected_characters", [])}
        tp += len(got & exp); fp += len(got - exp); fn += len(exp - got)
        if s.get("expected_epic"):
            tot += 1; ok += spec["epic"] == s["expected_epic"]
    assert tp / max(tp + fp, 1) >= 0.90
    assert tp / max(tp + fn, 1) >= 0.90
    assert ok / max(tot, 1) >= 0.85

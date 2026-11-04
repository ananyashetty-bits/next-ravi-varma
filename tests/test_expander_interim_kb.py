"""D's interim KB (expander/interim_kb/kb_d0.json) and the role-replacing stand-in parser."""

import json
import re

import pytest

from expander import _cfg
from expander._shims.stub_parser import parse_scene
from expander.names import NameMatcher, default_matcher, load_names

ROOT = _cfg.repo_root()
KB_PATH = ROOT / "expander" / "interim_kb" / "kb_d0.json"
KB = json.loads(KB_PATH.read_text())
SCENES = ROOT / "eval" / "test_scenes_dev.json"
EPICS = {"Mahabharata", "Ramayana", "Puranic", "Other", "Unknown"}


def _entries():
    for kind in ("characters", "groups"):
        for key, e in KB.get(kind, {}).items():
            yield kind, key, e


def _kb_matcher() -> NameMatcher:
    names = {e["canonical"]: list(e.get("aliases", [])) for _, _, e in _entries()}
    return NameMatcher(names)


def parse(text):
    return parse_scene(text, "t", str(KB_PATH))


def test_kb_version_and_entry_fields():
    assert KB["version"] == "d-kb0"
    for kind, key, e in _entries():
        for field in ("canonical", "aliases", "epic", "role", "visual", "coverage", "source", "status"):
            assert field in e, (kind, key, field)
        assert e["epic"] in EPICS and e["status"] in {"verified", "unverified"}
        src = e["source"]
        assert src["ref"], (key, "every entry cites a source")
        if e["status"] == "verified":
            assert src["type"] == "images" and all(r.startswith("varma_") for r in src["ref"]), key
        else:
            assert src["type"] == "text" and e["coverage"] == "none", key


def test_roles_and_visuals_are_name_free():
    for m in (default_matcher(), _kb_matcher()):
        for _, key, e in _entries():
            assert m.find(e["role"]) == [], (key, e["role"])
            assert m.find(e["visual"]) == [], (key, e["visual"])
        for key, st in KB["settings"].items():
            assert m.find(st["role"]) == [] and m.find(st["visual"]) == [], key


def test_seed_matcher_knows_every_kb_name():
    seed = NameMatcher(load_names(ROOT / "expander" / "names_seed.json"))
    for _, key, e in _entries():
        assert seed.find(e["canonical"]), key


@pytest.mark.skipif(not SCENES.exists(), reason="dev scenes not in this checkout")
def test_every_dev_scene_parses_to_its_expected_characters_with_a_name_free_action():
    m = default_matcher()
    for s in json.loads(SCENES.read_text()):
        spec = parse_scene(s["text"], s["id"], str(KB_PATH))
        got = {c["canonical"] for c in spec["characters"]}
        assert got == set(s["expected_characters"]) | set(s.get("_groups") or []), s["id"]
        assert spec["kb_version"] == "d-kb0"
        assert m.find(spec["action"]) == [], (s["id"], spec["action"])
        assert " 's" not in spec["action"] and "the the " not in spec["action"].lower(), s["id"]


def test_names_become_roles_with_articles_and_possessives():
    a = parse("Hanuman kneeling before Rama and offering him Sita's ring")["action"]
    assert a == ("the monkey-faced devotee kneeling before the blue-skinned prince and offering him "
                 "the young princess's ring")


def test_object_possessive_is_not_a_figure_but_place_possessive_is():
    assert [c["canonical"] for c in parse("Hanuman offering Rama Sita's ring")["characters"]] == ["Hanuman", "Rama"]
    spec = parse("Krishna standing in Duryodhana's royal court")
    assert [c["canonical"] for c in spec["characters"]] == ["Krishna", "Duryodhana"]


def test_no_double_article_after_adjectives():
    assert "a hesitant young warrior" in parse("turning to counsel a hesitant Arjuna")["action"]
    assert parse("old Shabri offering berries")["action"].startswith("old hermit woman")
    assert "little blue-skinned figure" in parse("yashodha scolding little krishn")["action"]


def test_groups_places_and_imperatives():
    spec = parse("Show Bhishma on the field of Kurukshetra while the Pandavas gather")
    assert [c["canonical"] for c in spec["characters"]] == ["Bhishma", "Pandavas"]
    assert spec["setting"]["name"] == "kurukshetra"
    assert spec["action"] == "the old warrior on the field of the battlefield while the five crowned princes gather"


def test_legacy_kb_without_roles_still_cuts_names():
    legacy = str(_cfg.SHIM_DIR / "stub_kb.json")
    a = parse_scene("Hanuman kneeling before Rama", "t", legacy)["action"]
    assert a == "kneeling before"
    assert not re.search(r"\b(the|a)\b", a)

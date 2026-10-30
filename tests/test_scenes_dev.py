"""Checks every dev scene follows the rules in Overview §6.2 (owner: B)."""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "eval", "test_scenes_dev.json")
TYPES = {"count", "figure", "pose", "attire", "object", "setting", "mood"}
EPICS = {"Mahabharata", "Ramayana", "Puranic", "Other", "Unknown"}
KEYS = {"id", "split", "text", "tier", "expected_epic", "expected_characters", "seen_in_training", "contributed_by", "checklist"}


def load():
    return json.load(open(PATH))


def test_ids_unique_and_keys_present():
    S = load()
    ids = [s["id"] for s in S]
    assert len(ids) == len(set(ids)), f"duplicate ids: {sorted(i for i in ids if ids.count(i) > 1)}"
    for s in S:
        missing = KEYS - set(s)
        assert not missing, f"{s.get('id')}: missing {missing}"
        assert s["tier"] in range(1, 6), f"{s['id']}: tier must be 1-5"
        assert s["expected_epic"] in EPICS, f"{s['id']}: epic {s['expected_epic']!r}"
        assert isinstance(s["seen_in_training"], bool), f"{s['id']}: seen_in_training must be true/false"


def test_checklists_follow_rules():
    S = load()
    names = {n.lower() for s in S for n in s["expected_characters"]}
    for s in S:
        cl = s["checklist"]
        assert 3 <= len(cl) <= 6, f"{s['id']}: needs 3-6 checklist questions, has {len(cl)}"
        assert len({c["id"] for c in cl}) == len(cl), f"{s['id']}: duplicate checklist ids"
        for c in cl:
            q = c["q"].lower()
            assert c["type"] in TYPES, f"{s['id']}/{c['id']}: type {c['type']!r}"
            assert 0 < c.get("weight", 1.0) <= 1.0, f"{s['id']}/{c['id']}: weight must be in (0, 1]"
            hit = [n for n in names if re.search(rf"\b{re.escape(n)}\b", q)]
            assert not hit, f"{s['id']}/{c['id']}: checklist question contains a name {hit}"
            assert "style" not in q and "painting" not in q, f"{s['id']}/{c['id']}: no style questions"

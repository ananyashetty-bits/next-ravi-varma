import re

import pytest

from pose import text as pt


def fake_parse(text):
    """Stand-in for kb.parser.parse_scene: every capitalised word from NAMES counts as a resolved character."""
    names = ["Draupadi", "Bhishma", "Duryodhana", "Krishna", "Rama", "Lakshmana", "Hanuman", "Savithri", "Yama",
             "Satyavan", "Sita", "Karna", "Arjuna", "Shabri"]
    chars = [{"mention": m.group(0), "canonical": m.group(0)} for m in re.finditer(r"[A-Za-z]+", text)
             if m.group(0) in names]
    return {"raw": text, "characters": chars, "unknown_entities": ["Nobody"]}


def fam(text):
    return pt.infer(fake_parse(text))[0]


# ---- pure logic: no rendering, no parser, runs in CI ----
@pytest.mark.parametrize("text, expected", [
    ("Draupadi kneeling before Bhishma and Duryodhana", ["kneeling", "standing", "standing"]),
    ("Rama, Lakshmana and Hanuman sitting on rocks", ["seated", "seated", "seated"]),
    ("Sita sitting alone under a tree", ["seated"]),
    ("Savithri pleading with Yama, Satyavan lying lifeless", ["kneeling", "standing", "lying"]),
    ("Rama and Sita kneeling before Hanuman", ["kneeling", "kneeling", "standing"]),
    ("Krishna blessing Arjuna", ["raised_arms", "standing"]),
    ("Karna standing proudly with his bow", ["standing"]),          # "bow" as a weapon is not a bow of respect
    ("Rama who draws his bow", ["standing"]),
    ("Sita bows down before Rama", ["kneeling", "standing"]),
    ("the sun rises over the hills", ["standing"]),                 # no character: one figure, "rises" is not "raise"
])
def test_families(text, expected):
    assert fam(text) == expected


def test_figure_count_rules():
    assert len(fam("Draupadi kneeling before Bhishma and Duryodhana as courtiers look on")) == 3
    assert len(fam("Rama Sita Lakshmana Hanuman Krishna standing")) == 3          # capped at 3
    assert len(fam("a quiet forest")) == 1                                         # none resolved -> 1
    assert len(pt.infer({"raw": "x", "characters": [], "unknown_entities": ["A", "B", "C"]})[0]) == 1  # not from unknowns


def test_unplaceable_name_falls_back_with_warning():
    f, w = pt.infer({"raw": "something sitting", "characters": [{"mention": "Zed", "canonical": "Zed"}]})
    assert f == ["seated"] and any("could not place" in x for x in w)


def test_guess_warning_only_when_guessing():
    _, w = pt.infer(fake_parse("Karna standing proudly"))
    assert w == []
    _, w = pt.infer(fake_parse("Draupadi folding her hands before Krishna"))
    assert len(w) == 2 and all("guessed" in x for x in w)


def test_deterministic():
    t = "Savithri pleading with Yama, Satyavan lying lifeless"
    assert fam(t) == fam(t)


# ---- rendering (needs opencv; skipped in a CI image without it) ----
def test_pose_from_text_returns_control_image():
    pytest.importorskip("cv2")
    c = pt.pose_from_text("Draupadi kneeling before Bhishma and Duryodhana", parse=fake_parse)
    assert c["image"].size == (512, 512) and c["image"].getbbox() is not None
    assert c["type"] == "openpose" and c["n_people"] == 3 and c["source"] == "auto"
    assert c["pose_json"]["families"] == ["kneeling", "standing", "standing"]
    assert len(c["pose_json"]["people"]) == 3


def test_parser_failure_never_crashes():
    pytest.importorskip("cv2")
    c = pt.pose_from_text("anything", parse=lambda t: 1 / 0)
    assert c["n_people"] == 1 and any("parser failed" in w for w in c["warnings"])


def test_accepts_a_parsed_scene_dict():
    pytest.importorskip("cv2")
    c = pt.pose_from_text(fake_parse("Sita sitting alone"))
    assert c["n_people"] == 1 and c["pose_json"]["families"] == ["seated"]

import json
import pathlib

import pytest
from PIL import Image

from pose import library as L
from pose import presets as pr
from pose import skeleton as sk

LIB = pathlib.Path(L.LIB_DIR)
NAMES = sorted(L.LIBRARY)


def test_at_least_14_presets_with_the_four_pairs_asked_for():
    assert len(NAMES) >= 14
    for must in ("embrace", "kneeling_before_standing", "seated_with_attendant", "bowing_pair"):
        assert must in L.LIBRARY and len(L.LIBRARY[must][2]) == 2


@pytest.mark.parametrize("name", NAMES)
def test_every_preset_is_valid_and_clean(name):
    pj = L.library_posejson(name)
    assert sk.validate_posejson(pj) == []
    assert len(pj["people"]) == len(L.LIBRARY[name][2]) <= 3
    assert sk.facing_camera_ok(pj) == []
    assert sk.layout_warnings(pj) == [], sk.layout_warnings(pj)


def test_index_format_for_the_dropdown():
    idx = L.index()
    assert [e["name"] for e in idx] == list(L.LIBRARY)
    for e in idx:
        assert {"name", "label", "n_people", "description"} <= set(e)
        assert e["label"] and e["description"] and e["n_people"] in (1, 2, 3)


def test_committed_library_matches_the_code():
    idx = json.load(open(LIB / "library.json"))
    assert idx == L.index(), "pose/library/library.json is stale; run: python -m pose.library"
    for n in NAMES:
        assert json.load(open(LIB / f"{n}.json")) == L.library_posejson(n), f"{n}.json is stale"
        assert Image.open(LIB / f"{n}.png").size == (256, 256)


def test_make_pose_accepts_library_names():
    pytest.importorskip("cv2")
    c = pr.make_pose("embrace")
    assert c["n_people"] == 2 and c["image"].size == (512, 512) and c["source"] == "preset:embrace"
    assert pr.make_pose("namaste")["n_people"] == 1
    assert pr.make_pose("standing", 2)["n_people"] == 2            # the generic path still works
    with pytest.raises(ValueError):
        pr.make_pose("embrace", 3)
    with pytest.raises(ValueError):
        pr.make_pose("not_a_preset")

import pytest

from pose import presets as pr
from pose import skeleton as sk

NAMES = sorted(pr.POSES)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("n", [1, 2, 3])
def test_presets_are_valid_and_clean(name, n):
    pj = pr.make_posejson(name, n)
    assert sk.validate_posejson(pj) == []
    assert len(pj["people"]) == n
    assert sk.facing_camera_ok(pj) == []
    assert sk.layout_warnings(pj) == [], sk.layout_warnings(pj)


def test_figures_share_a_ground_line():
    pj = pr.make_posejson(["kneeling", "standing", "seated"], 3)
    lows = [max(k[1] for k in p["keypoints"]) for p in pj["people"]]
    assert max(lows) - min(lows) < 1e-3


def test_deterministic_and_bad_input():
    assert pr.make_posejson("standing", 2) == pr.make_posejson("standing", 2)
    for bad in (lambda: pr.make_posejson("nope"), lambda: pr.make_posejson("standing", 4),
                lambda: pr.make_posejson(["standing"], 2), lambda: pr.make_posejson("standing", 2, ["left"])):
        with pytest.raises(ValueError):
            bad()


def test_make_pose_returns_control_image():
    pytest.importorskip("cv2")       # CI image has no opencv; Colab does
    c = pr.make_pose("kneeling", 2)
    assert c["image"].size == (512, 512) and c["image"].getbbox() is not None
    assert c["type"] == "openpose" and c["n_people"] == 2

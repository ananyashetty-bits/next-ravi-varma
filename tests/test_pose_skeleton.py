import copy
import pathlib

import pytest

from pose import skeleton as sk
from pose.stub import _standing_posejson

ROOT = pathlib.Path(__file__).resolve().parent.parent


def good():
    return _standing_posejson("preset:standing")


def moved(pj, dx=0.0, scale=1.0):
    q = copy.deepcopy(pj)
    for k in q["people"][0]["keypoints"]:
        k[0] = 0.5 + (k[0] - 0.5) * scale + dx
        k[1] = 0.5 + (k[1] - 0.5) * scale
    return q


def two(dx):
    a = good()
    a["people"].append(moved(good(), dx=dx)["people"][0])
    return a


# ---- CI: no models, no detector ----
def test_valid_posejson():
    assert sk.validate_posejson(good()) == []


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(version="x"),
    lambda p: p["people"][0]["keypoints"].pop(),
    lambda p: p["people"][0]["keypoints"][3].__setitem__(0, 1.5),
    lambda p: p["people"].extend([copy.deepcopy(p["people"][0]) for _ in range(3)]),
    lambda p: p.pop("source"),
])
def test_invalid_posejson(mutate):
    pj = good()
    mutate(pj)
    assert sk.validate_posejson(pj)


def test_left_right_order():
    assert sk.facing_camera_ok(good()) == []
    swapped = good()
    kp = swapped["people"][0]["keypoints"]
    kp[2], kp[5] = kp[5], kp[2]
    assert sk.facing_camera_ok(swapped) == [0]


def test_layout_warnings():
    assert sk.layout_warnings(good()) == []
    assert any("edge" in w for w in sk.layout_warnings(moved(good(), dx=0.45)))
    assert any("small" in w for w in sk.layout_warnings(moved(good(), scale=0.3)))
    assert sk.layout_warnings(two(0.3)) == []
    assert any("overlap" in w for w in sk.layout_warnings(two(0.05)))


def test_letterbox_keeps_proportions():
    assert sk.letterbox_xy(0.5, 0.5, 200, 100) == (0.5, 0.5)
    assert sk.letterbox_xy(0, 0, 200, 100) == (0.0, 0.25)       # black bars above and below
    assert sk.letterbox_xy(1, 1, 200, 100) == (1.0, 0.75)
    (x0, y0), (x1, y1) = sk.letterbox_xy(0, 0, 100, 200), sk.letterbox_xy(1, 0, 100, 200)
    assert (x0, y0) == (0.25, 0.0) and (x1, y1) == (0.75, 0.0)  # bars left and right


def test_empty_gives_black_control_image():
    c = sk.to_control({"version": "pose0.1", "canvas": 512, "people": [], "source": "auto", "producer": "C"})
    assert c["image"].size == (512, 512) and c["image"].getbbox() is None
    assert c["n_people"] == 0 and any("no person" in w for w in c["warnings"])


def test_to_control_sidecar_and_determinism():
    pytest.importorskip("cv2")        # the CI image may lack opencv; Colab has it
    a, b = sk.to_control(good(), source="preset:standing"), sk.to_control(good(), source="preset:standing")
    assert a["image"].size == (512, 512) and a["image"].mode == "RGB" and a["image"].getbbox() is not None
    assert a["type"] == "openpose" and a["n_people"] == 1 and a["source"] == "preset:standing"
    assert a["image"].tobytes() == b["image"].tobytes()


def test_canvas_bounds_do_not_crash():
    pytest.importorskip("cv2")
    c = sk.to_control(moved(good(), dx=0.45))
    assert c["image"].size == (512, 512) and any("edge" in w for w in c["warnings"])


# ---- slow: needs controlnet_aux (Colab) (paste the output in the PR) ----
@pytest.mark.slow
@pytest.mark.parametrize("name", ["stub", "s18", "s19", "s21", "s22"])
def test_render_is_pixel_identical_to_library_drawing(name):
    """The OpenPose detector does not recognise rendered skeletons, so instead we require that our rendering equals
    controlnet_aux's own draw_bodypose (installed version) bit for bit."""
    pytest.importorskip("controlnet_aux")      # not installed in CI -> skipped there, runs in Colab
    import json
    import numpy as np
    from controlnet_aux.open_pose.body import Keypoint
    from controlnet_aux.open_pose.util import draw_bodypose
    pj = good() if name == "stub" else json.load(open(ROOT / "eval" / "poses" / "dev" / f"{name}.json"))
    canvas = np.zeros((512, 512, 3), dtype=np.uint8)
    for p in pj["people"]:
        kps = [Keypoint(k[0], k[1]) if k[2] > 0 else None for k in p["keypoints"]]
        canvas = draw_bodypose(canvas, kps)
    mine = np.array(sk.render(pj))
    assert mine.shape == canvas.shape and (mine == canvas).all(), "differs from the library's own drawing"

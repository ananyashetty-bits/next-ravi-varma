"""CI tests for eval/pose_metric.compare and eval/pose_io (no detector, no GPU)."""
import json, os, sys
import pytest
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from eval.pose_metric import compare, _match
from eval.pose_io import make_posejson


def person(dx=0.0, dy=0.0, scale=1.0, cx=0.5):
    # a simple standing figure, 18 OpenPose points
    base = [(0.0, -0.30), (0.0, -0.22), (-0.06, -0.22), (-0.08, -0.12), (-0.09, -0.03), (0.06, -0.22), (0.08, -0.12),
            (0.09, -0.03), (-0.04, 0.0), (-0.04, 0.12), (-0.04, 0.24), (0.04, 0.0), (0.04, 0.12), (0.04, 0.24),
            (-0.01, -0.31), (0.01, -0.31), (-0.02, -0.30), (0.02, -0.30)]
    return [[cx + x * scale + dx, 0.5 + y * scale + dy, 1.0] for x, y in base]


def pj(*people):
    return make_posejson(list(people), source="test", producer="B")


def test_identical_is_perfect():
    r = compare(pj(person()), pj(person()))
    assert r["pck"] == 1.0 and r["joint_recall"] == 1.0 and r["ok"] and r["n_detected"] == r["n_expected"] == 1


def test_shift_is_low():
    assert compare(pj(person()), pj(person(dx=0.2)))["pck"] < 0.2


def test_small_jitter_is_tolerated():
    assert compare(pj(person()), pj(person(dx=0.02)))["pck"] == 1.0      # 0.02 < 0.10 x bbox diagonal (~0.56)


def test_missing_person_counts_as_wrong():
    r = compare(pj(person(cx=0.25), person(cx=0.75)), pj(person(cx=0.25)))
    assert r["pck"] == pytest.approx(0.5) and r["n_expected"] == 2 and r["n_detected"] == 1


def test_extra_detected_person_does_not_raise_pck():
    one = compare(pj(person(cx=0.25)), pj(person(cx=0.25)))
    extra = compare(pj(person(cx=0.25)), pj(person(cx=0.25), person(cx=0.75)))
    assert extra["pck"] == one["pck"] == 1.0 and extra["n_detected"] == 2


def test_empty_detection_not_ok():
    r = compare(pj(person()), pj())
    assert r["ok"] is False and r["pck"] == 0.0


def test_missing_joints_count_as_wrong():
    det = person(); det[4] = [0.0, 0.0, 0.0]; det[7] = [0.0, 0.0, 0.0]        # both wrists not detected
    r = compare(pj(person()), pj(det))
    assert r["pck"] == pytest.approx(16 / 18) and r["joint_recall"] == pytest.approx(16 / 18)


def test_matching_follows_position_not_order():
    ctrl = pj(person(cx=0.25), person(cx=0.75))
    det = pj(person(cx=0.75), person(cx=0.25))                                 # detector lists them the other way round
    assert compare(ctrl, det)["pck"] == 1.0
    assert _match([p["keypoints"] for p in ctrl["people"]], [p["keypoints"] for p in det["people"]]) == {0: 1, 1: 0}


def test_make_posejson_rejects_wrong_length():
    with pytest.raises(ValueError):
        make_posejson([[[0.5, 0.5, 1.0]] * 17], source="x", producer="B")


def test_render_draws_on_black():
    pytest.importorskip("cv2"); np = pytest.importorskip("numpy")
    from eval.pose_io import render
    im = np.array(render(pj(person())))
    assert im.shape == (512, 512, 3) and im[0, 0].sum() == 0 and (im.sum(axis=2) > 0).mean() > 0.005


def test_committed_pair_skeletons_match_their_posejson():
    """Every pair PNG must be the renderer's drawing of its PoseJSON (so ControlNet sees what the metric scores)."""
    pytest.importorskip("cv2"); np = pytest.importorskip("numpy")
    from PIL import Image
    from eval.pose_io import render, load
    pairs = json.load(open(os.path.join(ROOT, "eval", "test_pairs_dev.json")))
    for p in pairs:
        if p.get("pose_source") != "photo":
            continue
        a = np.array(Image.open(os.path.join(ROOT, p["pose_file"])).convert("RGB")).astype(int)
        b = np.array(render(load(os.path.join(ROOT, p["pose_json"])))).astype(int)
        assert np.abs(a - b).max() == 0, p["id"]

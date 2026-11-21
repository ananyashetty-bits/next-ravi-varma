"""D's composite pair tool (eval/compose_pair_pose.py): placement and mirroring. Pure maths, no detector."""

from eval.compose_pair_pose import FLOOR, compose, place

PERSON = [[0.50, 0.20, 1], [0.50, 0.30, 1], [0.45, 0.30, 1], [0.42, 0.40, 1], [0.40, 0.50, 1], [0.55, 0.30, 1],
          [0.58, 0.40, 1], [0.60, 0.50, 1], [0.47, 0.55, 1], [0.47, 0.70, 1], [0.47, 0.85, 1], [0.53, 0.55, 1],
          [0.53, 0.70, 1], [0.53, 0.85, 1], [0.49, 0.18, 1], [0.51, 0.18, 1], [0.48, 0.19, 1], [0, 0, 0]]


def _box(kps):
    pts = [k for k in kps if k[2] > 0]
    return min(k[0] for k in pts), min(k[1] for k in pts), max(k[0] for k in pts), max(k[1] for k in pts)


def test_place_scales_and_stands_on_floor():
    out = place(PERSON, x=0.3, height=0.5)
    x0, y0, x1, y1 = _box(out)
    assert abs((y1 - y0) - 0.5) < 1e-3 and abs(y1 - FLOOR) < 1e-3 and abs((x0 + x1) / 2 - 0.3) < 1e-3
    assert out[17] == [0.0, 0.0, 0.0]                       # a missing joint stays missing


def test_flip_mirrors_and_keeps_left_right_anatomy():
    out = place(PERSON, x=0.5, height=0.65, flip=True)
    # facing the camera, the person's right shoulder (2) stays on the image left of the left shoulder (5)
    assert out[2][0] < out[5][0]
    assert out[8][0] < out[11][0] and out[14][0] < out[15][0]   # hips and eyes too


def test_compose_multi_person():
    pj, rep = compose([{"src": "a", "x": 0.25, "height": 0.8}, {"src": "b", "x": 0.75, "height": 0.4}],
                      people_from=[[PERSON], [PERSON]])
    assert len(pj["people"]) == 2 and pj["producer"] == "D" and len(rep) == 2
    assert all(len(p["keypoints"]) == 18 for p in pj["people"])


def test_no_person_is_reported_not_crashed():
    pj, rep = compose([{"src": "a", "x": 0.5, "height": 0.8}], people_from=[[]])
    assert pj["people"] == [] and "NO PERSON" in rep[0]

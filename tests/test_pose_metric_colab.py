"""Colab-only checks (need the pinned controlnet_aux + weights). Run in an eval session, paste the output in the PR:
    python -m pytest -q tests/test_pose_metric_colab.py -s

Note: the OpenPose detector is trained on photos of people. It does NOT read a skeleton drawing back (verified
22 Nov: 0 people found on all 4 rendered pair skeletons), so a "render -> detect" round trip is not a valid test.
What we can and do check here:
1. our vendored render() is pixel-identical to controlnet_aux's own drawing (so ControlNet sees the drawing it was
   trained on);
2. the detector runs and finds nobody in a blank image.
Detection quality on real images is measured on our generated outputs and on the 15 real Varma paintings (B's
detector audit), which is where it matters.
"""
import os, sys
import pytest
pytest.importorskip("torch"); pytest.importorskip("controlnet_aux"); np = pytest.importorskip("numpy")
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from PIL import Image
from eval.pose_io import load, render
from eval.pose_metric import detect


@pytest.mark.parametrize("pid", ["s01", "s02", "s04", "s05", "s09", "s10", "s11", "s15"])
def test_render_matches_controlnet_aux_drawing(pid):
    from controlnet_aux.open_pose import util
    from controlnet_aux.open_pose.body import Keypoint
    pj = load(os.path.join(ROOT, "eval", "poses", "dev", f"{pid}.json"))
    canvas = np.zeros((512, 512, 3), dtype=np.uint8)
    for p in pj["people"]:
        kps = [Keypoint(k[0], k[1]) if k[2] > 0 else None for k in p["keypoints"]]
        canvas = util.draw_bodypose(canvas, kps)
    diff = np.abs(np.array(render(pj)).astype(int) - canvas.astype(int)).max()
    print(pid, "max pixel diff vs controlnet_aux:", diff)
    assert diff == 0


def test_blank_image_has_no_people():
    assert detect(Image.new("RGB", (512, 512)))["people"] == []

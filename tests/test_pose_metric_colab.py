"""Colab-only check (needs the pinned controlnet_aux + weights): the detector recovers our own rendered skeletons,
and finds nobody in a blank image. Run in an eval session and paste the output in the PR:
    python -m pytest -q tests/test_pose_metric_colab.py -s
"""
import os, sys
import pytest
pytest.importorskip("torch"); pytest.importorskip("controlnet_aux"); pytest.importorskip("cv2")
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from PIL import Image
from eval.pose_io import load, render
from eval.pose_metric import detect, compare


@pytest.mark.parametrize("pid", ["s01", "s02", "s04", "s05"])
def test_detector_recovers_rendered_skeleton(pid):
    ctrl = load(os.path.join(ROOT, "eval", "poses", "dev", f"{pid}.json"))
    r = compare(ctrl, detect(render(ctrl)))
    print(pid, r)
    assert r["ok"] and r["pck"] >= 0.8


def test_blank_image_has_no_people():
    assert detect(Image.new("RGB", (512, 512)))["people"] == []

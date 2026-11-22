import pytest
from pose import stub

CALLS = [lambda: stub.make_pose("kneeling", n=2, positions=["left", "right"]),
         lambda: stub.pose_from_text("anything at all"),
         lambda: stub.pose_from_text({"characters": []}),
         lambda: stub.extract_pose(None),
         lambda: stub.extract_edges(None)]


@pytest.mark.parametrize("call", CALLS)
def test_valid_control_image(call):
    c = call()
    assert c["image"].size == (512, 512) and c["image"].mode == "RGB"
    assert c["type"] in ("openpose", "canny")
    assert c["n_people"] == 1 and isinstance(c["warnings"], list)
    pj = c["pose_json"]
    assert pj["version"] == "pose0.1" and len(pj["people"]) == 1
    kps = pj["people"][0]["keypoints"]
    assert len(kps) == 18 and all(0 <= x <= 1 and 0 <= y <= 1 and 0 <= v <= 1 for x, y, v in kps)
    assert c["image"].getbbox() is not None          # not all black


def test_fallback_render_works_without_detector():
    img = stub._fallback_render(stub._standing_posejson("preset:test"))
    assert img.size == (512, 512) and img.getbbox() is not None

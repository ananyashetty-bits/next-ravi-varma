"""post/face_strips.py: head centre from the control skeleton, crops inside the image at any resolution."""

from PIL import Image

from post.face_strips import crop, head_centre, strips


def _pose(nose=(0.5, 0.2), neck=(0.5, 0.3)):
    kps = [[0.0, 0.0, 0.0] for _ in range(18)]
    kps[0] = [nose[0], nose[1], 1.0]
    kps[1] = [neck[0], neck[1], 1.0]
    return {"people": [{"keypoints": kps}]}


def test_head_centre_uses_face_points_then_neck():
    assert head_centre(_pose()) == (0.5, 0.2)
    p = _pose(); p["people"][0]["keypoints"][0] = [0, 0, 0]
    assert head_centre(p) == (0.5, 0.3)


def test_crop_same_region_at_512_and_768_and_clamped():
    a = crop(Image.new("RGB", (512, 512)), (0.5, 0.2))
    b = crop(Image.new("RGB", (768, 768)), (0.5, 0.2))
    assert a.size == b.size == (256, 256)
    edge = crop(Image.new("RGB", (512, 512)), (0.0, 0.0))     # clamped inside, no black border
    assert edge.size == (256, 256)


def test_strips_layout():
    c = Image.new("RGB", (256, 256))
    out = strips([("s09 s1", [c, c, c]), ("s19 s1", [c, c, c])])
    assert out.width > 3 * 256 and out.height > 2 * 256

"""Photo -> ControlImage and Painting -> Canny ControlImage (owner C; Overview §6.5).
Privacy: processes strictly in memory; writes nothing to disk and holds no reference to the input image.
"""
import cv2
import numpy as np
from PIL import Image
from eval.pose_io import make_posejson
from pose.skeleton import letterbox_xy, to_control

_DETECTOR = None


def _get_detector():
    global _DETECTOR
    if _DETECTOR is None:
        from controlnet_aux import OpenposeDetector
        _DETECTOR = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
    return _DETECTOR


def _empty_control(warnings):
    pj = make_posejson([], source="photo", producer="C")
    return to_control(pj, source="photo", warnings=warnings)


def extract_pose(image: Image.Image) -> dict:
    """Extract up to 3 people from a PIL image into a 512x512 ControlImage without distorting proportions.
    Writes nothing to disk and retains no reference to the input image."""
    warnings = []
    if image is None:
        return _empty_control(["no image provided"])

    rgb = image.convert("RGB")
    w, h = rgb.size
    arr = np.array(rgb)
    del rgb  # drop reference immediately

    try:
        det = _get_detector()
        poses = det.detect_poses(arr)
    except Exception as e:
        del arr
        return _empty_control([f"detector error: {e}"])
    del arr

    if not poses:
        return _empty_control(["no person detected in uploaded photo; suggest a clearer full-body photo"])

    # Extract 18-point body keypoints, mapped via letterbox_xy so proportions never distort
    scored_people = []
    for p in poses:
        kps = []
        xs, ys = [], []
        body_kps = p.body.keypoints[:18]
        for kp in body_kps:
            if kp is not None and kp.score > 0.1:
                lx, ly = letterbox_xy(float(kp.x), float(kp.y), w, h)
                lx = max(0.0, min(1.0, round(lx, 4)))
                ly = max(0.0, min(1.0, round(ly, 4)))
                conf = max(0.0, min(1.0, round(float(kp.score), 3)))
                kps.append([lx, ly, conf])
                xs.append(lx)
                ys.append(ly)
            else:
                kps.append([0.0, 0.0, 0.0])

        if len(xs) < 4:
            continue

        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        scored_people.append((area, kps))

    if not scored_people:
        return _empty_control(["detected figure had too few visible joints; try cropping closer"])

    # Keep the 3 largest people by bounding-box area
    scored_people.sort(key=lambda item: item[0], reverse=True)
    if len(scored_people) > 3:
        warnings.append(f"detected {len(scored_people)} people; kept the 3 largest")
        scored_people = scored_people[:3]

    people = []
    for idx, (area, kps) in enumerate(scored_people):
        # Check for cropped legs (ankles are indices 10 and 13)
        if kps[10][2] == 0.0 and kps[13][2] == 0.0:
            warnings.append(f"person {idx} legs/ankles not visible (possibly cropped)")
        # Check low average joint confidence
        valid_confs = [k[2] for k in kps if k[2] > 0]
        if valid_confs and (sum(valid_confs) / len(valid_confs)) < 0.45:
            warnings.append(f"person {idx} has low keypoint confidence")
        people.append(kps)

    pj = make_posejson(people, source="photo", producer="C")
    # to_control automatically adds layout_warnings (small figure, edge clipping, torso overlap)
    return to_control(pj, source="photo", warnings=warnings)


def extract_edges(image: Image.Image, low: int = 100, high: int = 200) -> dict:
    """Extract Canny edges (white on black, 512x512 RGB) for composition study (p4)."""
    from PIL import ImageOps
    padded = ImageOps.pad(image.convert("RGB"), (512, 512), color=(0, 0, 0))
    gray = cv2.cvtColor(np.array(padded), cv2.COLOR_RGB2GRAY)
    del padded
    edges = cv2.Canny(gray, low, high)
    edge_rgb = Image.fromarray(edges).convert("RGB")
    return {
        "image": edge_rgb,
        "type": "canny",
        "source": "painting",
        "n_people": 0,
        "pose_json": None,
        "warnings": [],
    }

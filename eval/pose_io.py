"""PoseJSON helpers shared by eval/ (Overview §6.2). CPU only; no torch.

render(posejson) draws a skeleton exactly as controlnet_aux 0.0.10 does (``open_pose.util.draw_bodypose``, vendored
below so CI can run it without torch; Apache-2.0, https://github.com/huggingface/controlnet_aux). ControlNet was
trained on that drawing, so every skeleton we feed it should come from here or from the detector itself.
"""
import json
import math

BODY_POINTS = ["nose", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder", "l_elbow", "l_wrist", "r_hip",
               "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle", "r_eye", "l_eye", "r_ear", "l_ear"]
_LIMBS = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], [10, 11], [2, 12], [12, 13], [13, 14],
          [2, 1], [1, 15], [15, 17], [1, 16], [16, 18]]
_COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
           [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
           [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


def load(path):
    return json.load(open(path, encoding="utf-8"))


def make_posejson(people, source, producer, canvas=512, **extra):
    """people: list of 18-point keypoint lists [[x, y, conf], ...] (normalised 0-1, conf 0 = missing)."""
    for p in people:
        if len(p) != 18:
            raise ValueError(f"expected 18 keypoints, got {len(p)}")
    out = {"version": "pose0.1", "canvas": canvas, "people": [{"keypoints": p} for p in people],
           "source": source, "producer": producer}
    out.update(extra)
    return out


def present(kp, min_conf=0.3):
    return kp is not None and len(kp) >= 3 and kp[2] > min_conf


def draw_bodypose(canvas, keypoints):
    """controlnet_aux 0.0.10 util.draw_bodypose, unchanged except Keypoint -> (x, y) tuples or None."""
    import cv2
    import numpy as np
    H, W, _ = canvas.shape
    stickwidth = 4
    for (k1, k2), color in zip(_LIMBS, _COLORS):
        p1, p2 = keypoints[k1 - 1], keypoints[k2 - 1]
        if p1 is None or p2 is None:
            continue
        Y = np.array([p1[0], p2[0]]) * float(W)
        X = np.array([p1[1], p2[1]]) * float(H)
        mX, mY = np.mean(X), np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, [int(float(c) * 0.6) for c in color])
    for kp, color in zip(keypoints, _COLORS):
        if kp is None:
            continue
        cv2.circle(canvas, (int(kp[0] * W), int(kp[1] * H)), 4, color, thickness=-1)
    return canvas


def render(posejson, size=512):
    """PoseJSON -> PIL RGB skeleton on black (body only), drawn the way the detector library draws."""
    import numpy as np
    from PIL import Image
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    for person in posejson.get("people", []):
        kps = [(k[0], k[1]) if (k is not None and len(k) >= 3 and k[2] > 0) else None for k in person["keypoints"]]
        canvas = draw_bodypose(canvas, kps)
    return Image.fromarray(canvas)

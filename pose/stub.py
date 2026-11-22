"""Pose-input stubs (owner C; Overview 6.5 and 7). Every function has the real name and signature and returns a
valid ControlImage (512x512 RGB + sidecar dict) showing ONE standing figure, for any input. Lets D's app and A's
`--control_source auto` run before the real pose/ modules land; replaced at S2 by changing one import line.
Never needs a GPU or the detector, and never writes anything."""

from PIL import Image, ImageDraw

_NAMES = ["nose", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder", "l_elbow", "l_wrist", "r_hip",
          "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle", "r_eye", "l_eye", "r_ear", "l_ear"]

# One standing figure, normalised 0-1 (x right, y down), well inside the canvas. Person's right = image left.
_STANDING = {
    "nose": (0.50, 0.16), "neck": (0.50, 0.24),
    "r_shoulder": (0.42, 0.26), "r_elbow": (0.39, 0.40), "r_wrist": (0.38, 0.53),
    "l_shoulder": (0.58, 0.26), "l_elbow": (0.61, 0.40), "l_wrist": (0.62, 0.53),
    "r_hip": (0.45, 0.55), "r_knee": (0.45, 0.73), "r_ankle": (0.45, 0.91),
    "l_hip": (0.55, 0.55), "l_knee": (0.55, 0.73), "l_ankle": (0.55, 0.91),
    "r_eye": (0.48, 0.145), "l_eye": (0.52, 0.145), "r_ear": (0.455, 0.155), "l_ear": (0.545, 0.155),
}
_LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
          (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]


def _standing_posejson(source):
    return {"version": "pose0.1", "canvas": 512,
            "people": [{"keypoints": [[_STANDING[n][0], _STANDING[n][1], 1.0] for n in _NAMES]}],
            "source": source, "producer": "C"}


def _fallback_render(pj):
    """Plain PIL skeleton on black; used only if the repo's real renderer can't be imported (e.g. CPU-only CI)."""
    img = Image.new("RGB", (512, 512), (0, 0, 0))
    d = ImageDraw.Draw(img)
    for person in pj["people"]:
        kp = person["keypoints"]
        for a, b in _LIMBS:
            if kp[a][2] > 0 and kp[b][2] > 0:
                d.line([kp[a][0] * 512, kp[a][1] * 512, kp[b][0] * 512, kp[b][1] * 512],
                       fill=(255, 140, 0), width=4)
        for x, y, c in kp:
            if c > 0:
                d.ellipse([x * 512 - 4, y * 512 - 4, x * 512 + 4, y * 512 + 4], fill=(0, 200, 255))
    return img


def _render(pj):
    try:
        from eval.pose_io import render      # the same renderer make_pair_pose.py uses
        img = render(pj).convert("RGB")
        return img if img.size == (512, 512) else img.resize((512, 512))
    except Exception:
        return _fallback_render(pj)


def _control(source, kind="openpose", extra_warning=None):
    pj = _standing_posejson(source)
    img = _render(pj)
    if kind == "canny":                       # white edges on black
        img = img.convert("L").point(lambda v: 255 if v > 0 else 0).convert("RGB")
    warnings = ["stub: returns a fixed standing figure"] + ([extra_warning] if extra_warning else [])
    return {"image": img, "type": kind, "source": source, "n_people": 1, "pose_json": pj, "warnings": warnings}


def make_pose(name, n=1, positions=None):
    """Stub: always one standing figure, whatever `name`, `n` and `positions` are."""
    return _control(f"preset:{name}")


def pose_from_text(text_or_scene):
    """Stub: always one standing figure, whatever the text is."""
    return _control("auto")


def extract_pose(image):
    """Stub: ignores the image, returns one standing figure. Writes nothing and keeps no reference to the input."""
    return _control("photo")


def extract_edges(image, low=100, high=200):
    """Stub: ignores the image, returns the standing figure as white edges on black (canny)."""
    return _control("painting", kind="canny")

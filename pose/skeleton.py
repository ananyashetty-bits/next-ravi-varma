"""Skeleton renderer for pose inputs (owner C; Overview 6.2, 6.3).

PoseJSON -> ControlImage. Drawing is ALWAYS eval.pose_io.render, which is controlnet_aux 0.0.10's own
draw_bodypose (ControlNet was trained on that exact rendering). Nothing in pose/ may draw skeletons any other way.
Body-only: hands and face exist only for A's photo-based p8. CPU only, writes nothing.
"""
from eval.pose_io import BODY_POINTS, present, render


def validate_posejson(pj):
    """Return a list of problems (empty list = valid PoseJSON, Overview 6.2)."""
    problems = []
    if pj.get("version") != "pose0.1":
        problems.append("version must be 'pose0.1'")
    if pj.get("canvas") != 512:
        problems.append("canvas must be 512")
    for key in ("source", "producer"):
        if key not in pj:
            problems.append(f"missing '{key}'")
    people = pj.get("people")
    if not isinstance(people, list) or not people:
        problems.append("no people")
        return problems
    if len(people) > 3:
        problems.append("more than 3 people")
    for i, p in enumerate(people):
        kps = p.get("keypoints")
        if not isinstance(kps, list) or len(kps) != 18:
            problems.append(f"person {i}: need exactly 18 keypoints")
            continue
        for name, k in zip(BODY_POINTS, kps):
            if len(k) != 3:
                problems.append(f"person {i} {name}: need [x, y, conf]")
            elif k[2] > 0 and not all(0 <= v <= 1 for v in k):
                problems.append(f"person {i} {name}: value outside 0-1")
            elif not 0 <= k[2] <= 1:
                problems.append(f"person {i} {name}: conf outside 0-1")
    return problems


def facing_camera_ok(pj):
    """Indices of people whose left/right look swapped. A front-facing person's right shoulder is on the image LEFT."""
    bad = []
    for i, p in enumerate(pj["people"]):
        rs, ls = p["keypoints"][2], p["keypoints"][5]
        if present(rs, 0) and present(ls, 0) and rs[0] >= ls[0]:
            bad.append(i)
    return bad


def _box(points):
    xs, ys = [x for x, _ in points], [y for _, y in points]
    return min(xs), min(ys), max(xs), max(ys)


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def layout_warnings(pj, margin=0.03, min_height=0.30, max_torso_iou=0.10):
    """Things that make ControlNet silently ignore a skeleton: cut off, too small, torsos overlapping."""
    warnings, torsos = [], []
    for i, p in enumerate(pj["people"]):
        pts = [(k[0], k[1]) for k in p["keypoints"] if present(k, 0)]
        if not pts:
            warnings.append(f"person {i} has no visible joints")
            continue
        x0, y0, x1, y1 = _box(pts)
        if x0 < margin or y0 < margin or x1 > 1 - margin or y1 > 1 - margin:
            warnings.append(f"person {i} touches or leaves the canvas edge")
        if y1 - y0 < min_height:
            warnings.append(f"person {i} is small ({y1 - y0:.0%} of the canvas height)")
        torso = [(p["keypoints"][j][0], p["keypoints"][j][1]) for j in (2, 5, 8, 11) if present(p["keypoints"][j], 0)]
        torsos.append((i, _box(torso)) if len(torso) >= 2 else None)
    torsos = [t for t in torsos if t]
    for a in range(len(torsos)):
        for b in range(a + 1, len(torsos)):
            if _iou(torsos[a][1], torsos[b][1]) > max_torso_iou:
                warnings.append(f"torsos of person {torsos[a][0]} and {torsos[b][0]} overlap")
    return warnings


def to_control(pj, source=None, warnings=None):
    """PoseJSON -> ControlImage (Overview 6.3): 512x512 RGB skeleton on black + sidecar dict."""
    people = pj.get("people", [])
    warn = list(warnings or []) + layout_warnings(pj)
    if not people:
        warn.append("no person: black control image (no pose)")
    return {"image": render(pj).convert("RGB"), "type": "openpose", "source": source or pj.get("source", "preset"),
            "n_people": len(people), "pose_json": pj, "warnings": warn}


def letterbox_xy(x, y, w, h):
    """Map a normalised point in a w x h image to the same point after letterboxing to a square (same scale on both
    axes, image centred, black bars). Used by extract_pose so proportions are never distorted."""
    s = float(max(w, h))
    return (x * w + (s - w) / 2) / s, (y * h + (s - h) / 2) / s

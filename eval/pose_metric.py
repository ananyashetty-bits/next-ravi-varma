"""Pose adherence: does the generated image follow the control skeleton? (Overview §6.5, §10)

compare(control, detected)  pure maths, runs in CI
detect(image)               OpenPose (pinned controlnet_aux) on a generated image -> PoseJSON. Colab only. Cached.
pose_score(control, image)  = compare(control, detect(image))

Method (written in experiments/HYPOTHESES_M4.md section B before anything was scored):
1. people are matched control <-> detected by minimum total distance between body centroids (neck + hips);
2. a control joint counts as correct if the matched person has it (conf > 0.3) within pck_thresh x the diagonal of
   the control person's keypoint bounding box;
3. pck = correct joints / all control joints (missed people and joints count as wrong); ok = anyone detected.
Known weakness: the detector is the family that drew the control skeletons (circular), and painted robes and
stylised bodies reduce detection. Human pose-match ratings are the main evidence; PCK supports them.
"""
import hashlib
import io
import itertools
import json
import os

try:
    from eval.pose_io import present, make_posejson
except ImportError:                                  # run as a script from eval/
    from pose_io import present, make_posejson

CENTROID = (1, 8, 11)          # neck, r_hip, l_hip
MIN_CONF = 0.3


def _centroid(kps):
    pts = [kps[i] for i in CENTROID if present(kps[i], MIN_CONF)] or [k for k in kps if present(k, MIN_CONF)]
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def _diag(kps):
    pts = [k for k in kps if present(k, MIN_CONF)]
    if len(pts) < 2:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5


def _match(ctrl, det):
    """Minimum-cost assignment of control people to detected people (brute force; <= 3 control people)."""
    cc = [_centroid(p) for p in ctrl]
    dc = [_centroid(p) for p in det]
    best, best_cost = {}, float("inf")
    k = min(len(ctrl), len(det))
    for ci in itertools.combinations(range(len(ctrl)), k):
        for di in itertools.permutations(range(len(det)), k):
            cost, pairs = 0.0, {}
            for c, d in zip(ci, di):
                if cc[c] is None or dc[d] is None:
                    cost += 10.0
                else:
                    cost += ((cc[c][0] - dc[d][0]) ** 2 + (cc[c][1] - dc[d][1]) ** 2) ** 0.5
                pairs[c] = d
            if cost < best_cost:
                best, best_cost = pairs, cost
    return best


def compare(control_pose_json, detected_pose_json, pck_thresh=0.10):
    ctrl = [p["keypoints"] for p in control_pose_json.get("people", [])]
    det = [p["keypoints"] for p in (detected_pose_json or {}).get("people", [])]
    n_ctrl_joints = sum(1 for p in ctrl for k in p if present(k, MIN_CONF))
    out = {"pck": None, "n_expected": len(ctrl), "n_detected": len(det), "joint_recall": 0.0, "ok": bool(det)}
    if n_ctrl_joints == 0:
        return out
    if not det:
        out["pck"] = 0.0
        return out
    m = _match(ctrl, det)
    correct = found = 0
    for c, d in m.items():
        diag = _diag(ctrl[c])
        for j, kc in enumerate(ctrl[c]):
            kd = det[d][j]
            if not present(kc, MIN_CONF) or not present(kd, MIN_CONF):
                continue
            found += 1
            dist = ((kc[0] - kd[0]) ** 2 + (kc[1] - kd[1]) ** 2) ** 0.5
            if diag and dist < pck_thresh * diag:
                correct += 1
    out["pck"] = correct / n_ctrl_joints
    out["joint_recall"] = found / n_ctrl_joints
    return out


# ---- detection (Colab: needs the pinned controlnet_aux + its weights) -----------------------------------------------
_DET = None


def _detector():
    global _DET
    if _DET is None:
        from controlnet_aux import OpenposeDetector
        cfg = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "m4.json")))
        _DET = OpenposeDetector.from_pretrained(cfg["pose_detector"]["weights"])
    return _DET


def detector_version():
    try:
        import controlnet_aux
        return f"controlnet_aux-{getattr(controlnet_aux, '__version__', '?')}"
    except ImportError:
        return "none"


def poses_to_posejson(poses, source, producer="B", max_people=None):
    """controlnet_aux PoseResult list (normalised keypoints) -> PoseJSON; keeps the largest people if max_people."""
    people = []
    for p in poses:
        kps = [[float(k.x), float(k.y), 1.0] if k is not None else [0.0, 0.0, 0.0] for k in p.body.keypoints]
        people.append(kps)
    if max_people is not None and len(people) > max_people:
        def area(kps):
            pts = [k for k in kps if k[2] > 0]
            if not pts:
                return 0.0
            return (max(k[0] for k in pts) - min(k[0] for k in pts)) * (max(k[1] for k in pts) - min(k[1] for k in pts))
        people = sorted(people, key=area, reverse=True)[:max_people]
    return make_posejson(people, source=source, producer=producer)


def letterbox(image, size=512):
    from PIL import ImageOps
    return ImageOps.pad(image.convert("RGB"), (size, size), color=(0, 0, 0))


class DetectCache:
    """Per-owner jsonl cache on Drive: {"key": sha256+detector version, "pose": PoseJSON}. Never in git."""
    def __init__(self, path):
        self.path, self.d = path, {}
        if path and os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                r = json.loads(line)
                self.d[r["key"]] = r["pose"]

    def get(self, key):
        return self.d.get(key)

    def put(self, key, pose):
        self.d[key] = pose
        if self.path:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "pose": pose}) + "\n")


def _key(image_bytes):
    return hashlib.sha256(image_bytes).hexdigest() + "|" + detector_version()


def detect(image, cache=None):
    """PIL image or path -> PoseJSON of the people the pinned detector finds (body only)."""
    import numpy as np
    from PIL import Image
    if isinstance(image, str):
        raw = open(image, "rb").read()
        img = Image.open(io.BytesIO(raw))
    else:
        buf = io.BytesIO(); image.save(buf, format="PNG"); raw = buf.getvalue(); img = image
    key = _key(raw)
    if cache is not None and cache.get(key) is not None:
        return cache.get(key)
    poses = _detector().detect_poses(np.array(letterbox(img)), include_hand=False, include_face=False)
    pj = poses_to_posejson(poses, source="detected")
    if cache is not None:
        cache.put(key, pj)
    return pj


def pose_score(control_pose_json, image, cache=None, pck_thresh=0.10):
    return compare(control_pose_json, detect(image, cache), pck_thresh)

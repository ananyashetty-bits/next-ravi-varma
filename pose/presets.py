"""Templated poses (owner C; Overview 6.5; Member C file 1.6). A small body model plus forward kinematics: a pose is
limb angles, never hand-placed points. Angles are in degrees from straight down, positive toward the image right
(a person's right side is on the image LEFT). All presets are respectful, balanced postures; names say the posture,
never a deity. Pure Python; rendering happens in pose.skeleton (the library's own drawing)."""
import math

from eval.pose_io import BODY_POINTS
from pose.skeleton import to_control, validate_posejson

TORSO = 0.22                  # neck-to-hip-centre height as a fraction of the canvas
GROUND = 0.86                 # the lowest point of every figure sits on this line
_L = {"shoulder": 0.41, "hip": 0.20, "uarm": 0.70, "farm": 0.62, "thigh": 0.95, "shin": 0.90, "neck_nose": 0.35}

# angles of each limb (absolute, from straight down), plus optional foreshortening factors "f" (1 = full length)
POSES = {
    "standing": dict(lean=0, r_uarm=-6, r_farm=-3, l_uarm=6, l_farm=3,
                     r_thigh=-3, r_shin=0, l_thigh=3, l_shin=0),
    "seated": dict(lean=0, r_uarm=-8, r_farm=28, l_uarm=8, l_farm=-28,
                   r_thigh=-30, r_shin=-4, l_thigh=30, l_shin=4, f={"r_thigh": 0.5, "l_thigh": 0.5}),
    "kneeling": dict(lean=0, r_uarm=-8, r_farm=28, l_uarm=8, l_farm=-28,
                     r_thigh=-4, r_shin=-75, l_thigh=4, l_shin=75, f={"r_shin": 0.4, "l_shin": 0.4}),
    # arms clearly above the shoulder line and the head; "k" shrinks the figure so the hands stay inside the canvas
    "raised_arms": dict(lean=0, r_uarm=-150, r_farm=-165, l_uarm=150, l_farm=165,
                        r_thigh=-3, r_shin=0, l_thigh=3, l_shin=0, k=0.85),
    # the standing body turned 90 degrees (head to the image left), smaller so a lying figure fits beside others
    "lying": dict(lean=0, r_uarm=-6, r_farm=-3, l_uarm=6, l_farm=3,
                  r_thigh=-3, r_shin=0, l_thigh=3, l_shin=0, rot=-90, k=0.55),
}


def _figure(spec, cx, scale):
    """Forward kinematics -> 18 (x, y) points in BODY_POINTS order, ground-anchored, centred on cx."""
    T, f = TORSO * scale * spec.get("k", 1.0), spec.get("f", {})

    def step(p, key, length):
        th = math.radians(spec[key])
        d = length * f.get(key, 1.0)
        return (p[0] + math.sin(th) * d, p[1] + math.cos(th) * d)

    lean = math.radians(spec["lean"])
    neck = (T * math.sin(lean), -T * math.cos(lean))
    nose = (neck[0] + _L["neck_nose"] * T * math.sin(lean), neck[1] - _L["neck_nose"] * T * math.cos(lean))
    r_sh, l_sh = (neck[0] - _L["shoulder"] * T, neck[1]), (neck[0] + _L["shoulder"] * T, neck[1])
    r_el, l_el = step(r_sh, "r_uarm", _L["uarm"] * T), step(l_sh, "l_uarm", _L["uarm"] * T)
    r_wr, l_wr = step(r_el, "r_farm", _L["farm"] * T), step(l_el, "l_farm", _L["farm"] * T)
    r_hp, l_hp = (-_L["hip"] * T, 0.0), (_L["hip"] * T, 0.0)
    r_kn, l_kn = step(r_hp, "r_thigh", _L["thigh"] * T), step(l_hp, "l_thigh", _L["thigh"] * T)
    r_an, l_an = step(r_kn, "r_shin", _L["shin"] * T), step(l_kn, "l_shin", _L["shin"] * T)
    r_ey, l_ey = (nose[0] - 0.10 * T, nose[1] - 0.12 * T), (nose[0] + 0.10 * T, nose[1] - 0.12 * T)
    r_ea, l_ea = (nose[0] - 0.22 * T, nose[1] - 0.02 * T), (nose[0] + 0.22 * T, nose[1] - 0.02 * T)
    pts = [nose, neck, r_sh, r_el, r_wr, l_sh, l_el, l_wr, r_hp, r_kn, r_an, l_hp, l_kn, l_an, r_ey, l_ey, r_ea, l_ea]
    rot = spec.get("rot", 0)
    if rot:                                           # turn the whole figure about the hip centre
        c, s_ = math.cos(math.radians(rot)), math.sin(math.radians(rot))
        pts = [(x * c - y * s_, x * s_ + y * c) for x, y in pts]
    dy = GROUND - max(y for _, y in pts)
    dx = cx
    if rot:                                           # centre the turned figure on cx, then keep it on the canvas
        xs = [x for x, _ in pts]
        dx = cx - (min(xs) + max(xs)) / 2
        dx = min(max(dx, 0.04 - min(xs)), 0.96 - max(xs))
    return [(x + dx, y + dy) for x, y in pts]


_X = {1: {"center": 0.5}, 2: {"left": 0.28, "right": 0.72}, 3: {"left": 0.2, "center": 0.5, "right": 0.8}}
_DEFAULT = {1: ["center"], 2: ["left", "right"], 3: ["left", "center", "right"]}
_SCALE = {1: 1.0, 2: 1.0, 3: 0.9}


def make_posejson(name, n=1, positions=None):
    names = list(name) if isinstance(name, (list, tuple)) else [name] * n
    if n not in (1, 2, 3):
        raise ValueError("n must be 1, 2 or 3")
    if len(names) != n:
        raise ValueError("need one posture name per figure")
    unknown = [nm for nm in names if nm not in POSES]
    if unknown:
        raise ValueError(f"unknown posture {unknown}; available: {sorted(POSES)}")
    positions = positions or _DEFAULT[n]
    if len(positions) != n:
        raise ValueError("need one position per figure")
    people = []
    for nm, pos in zip(names, positions):
        cx = float(pos) if isinstance(pos, (int, float)) else _X[n][pos]
        pts = _figure(POSES[nm], cx, _SCALE[n])
        people.append({"keypoints": [[round(x, 4), round(y, 4), 1.0] for x, y in pts]})
    pj = {"version": "pose0.1", "canvas": 512, "people": people,
          "source": "preset:" + "+".join(names), "producer": "C"}
    problems = validate_posejson(pj)
    if problems:
        raise ValueError("preset produced an invalid PoseJSON: " + "; ".join(problems))
    return pj


def make_pose(name, n=1, positions=None):
    """name: a posture, or a list with one posture per figure. positions: 'left'/'center'/'right' or x in 0-1."""
    pj = make_posejson(name, n, positions)
    return to_control(pj, source=pj["source"])

"""Preset library (owner C; Overview 6.5; Member C file 2.1). 19 named poses as PoseJSON + 256x256 thumbnails + the
index D's dropdown reads (pose/library/library.json: name, label, n_people, description). Built by `python -m
pose.library` (deterministic, CPU only). make_pose(name) accepts every library name, so the app can call it directly.
All poses are upright, balanced, respectful postures; names say the posture, never a deity.
"""
import json
import os

from pose.presets import POSES, TORSO, _figure, _SCALE
from pose.skeleton import validate_posejson, layout_warnings, facing_camera_ok

HERE = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.join(HERE, "library")

# extra single postures (wide or specialised arms, so they are not part of the generic POSES x n grid)
_STAND = dict(r_thigh=-3, r_shin=0, l_thigh=3, l_shin=0)
_KNEEL = dict(r_thigh=-4, r_shin=-75, l_thigh=4, l_shin=75, f={"r_shin": 0.4, "l_shin": 0.4})
_NAMASTE = dict(r_uarm=-20, r_farm=108, l_uarm=20, l_farm=-108)       # hands joined at the chest
EXTRA = {
    "namaste": dict(lean=0, **_NAMASTE, **_STAND),
    "kneel_namaste": dict(lean=0, **_NAMASTE, **_KNEEL),
    "bow_l": dict(lean=14, **_NAMASTE, **_STAND),                      # leans toward the image right
    "bow_r": dict(lean=-14, **_NAMASTE, **_STAND),                     # leans toward the image left
    "archer": dict(lean=0, r_uarm=-75, r_farm=130, l_uarm=90, l_farm=90, **_STAND),   # string hand drawn to the cheek, other arm out
    "pointing": dict(lean=0, r_uarm=-6, r_farm=-3, l_uarm=75, l_farm=80, **_STAND),
    "walking": dict(lean=0, r_uarm=8, r_farm=4, l_uarm=-8, l_farm=-4,       # one leg forward (shorter), one trailing
                    r_thigh=-9, r_shin=-13, l_thigh=7, l_shin=2, f={"l_thigh": 0.85, "l_shin": 0.85}),
    "hug_l": dict(lean=0, r_uarm=-6, r_farm=-3, l_uarm=50, l_farm=80, **_STAND),    # left figure reaches to the partner
    "hug_r": dict(lean=0, r_uarm=-50, r_farm=-80, l_uarm=6, l_farm=3, **_STAND),    # right figure reaches to the partner
}
SPECS = {**POSES, **EXTRA}

# name -> (label, description, [(posture, x), ...], scale or None)
LIBRARY = {
    "standing": ("Standing", "One figure standing upright, arms relaxed.", [("standing", 0.5)], None),
    "seated": ("Seated", "One figure seated, hands resting.", [("seated", 0.5)], None),
    "kneeling": ("Kneeling", "One figure kneeling upright.", [("kneeling", 0.5)], None),
    "lying": ("Resting on the ground", "One figure lying at rest on the ground.", [("lying", 0.5)], None),
    "raised_arms": ("Arms raised", "One figure standing with both arms raised above the head.", [("raised_arms", 0.5)], None),
    "namaste": ("Hands joined", "One figure standing with hands joined at the chest in greeting.", [("namaste", 0.5)], None),
    "kneel_namaste": ("Kneeling, hands joined", "One figure kneeling with hands joined in appeal.", [("kneel_namaste", 0.5)], None),
    "archer": ("Archer", "One figure standing, one arm extended and the other drawn back, as if holding a bow.", [("archer", 0.5)], None),
    "pointing": ("Pointing", "One figure standing and pointing to one side.", [("pointing", 0.5)], None),
    "walking": ("Walking", "One figure walking, mid-stride.", [("walking", 0.5)], None),
    "embrace": ("Embrace", "Two figures standing close, each with an arm around the other.", [("hug_l", 0.39), ("hug_r", 0.61)], None),
    "kneeling_before_standing": ("Kneeling before a standing figure", "One figure kneels at the left before a standing figure.", [("kneeling", 0.3), ("standing", 0.7)], None),
    "seated_with_attendant": ("Seated, with an attendant", "One figure seated, another standing beside them.", [("seated", 0.32), ("standing", 0.7)], None),
    "bowing_pair": ("Bowing pair", "Two figures facing each other, bowing slightly with hands joined.", [("bow_l", 0.30), ("bow_r", 0.70)], None),
    "standing_pair": ("Two standing", "Two figures standing side by side with space between them.", [("standing", 0.28), ("standing", 0.72)], None),
    "walking_pair": ("Walking together", "Two figures walking side by side.", [("walking", 0.3), ("walking", 0.7)], None),
    "standing_trio": ("Three standing", "Three figures standing in a row.", [("standing", 0.2), ("standing", 0.5), ("standing", 0.8)], None),
    "seated_trio": ("Three seated", "Three figures seated side by side.", [("seated", 0.2), ("seated", 0.5), ("seated", 0.8)], None),
    "kneeling_before_two": ("Kneeling before two", "One figure kneels at the left before two standing figures.", [("kneeling", 0.18), ("standing", 0.58), ("standing", 0.86)], None),
}


def library_posejson(name):
    label, desc, figs, scale = LIBRARY[name]
    n = len(figs)
    sc = scale or _SCALE[n]
    people = []
    for posture, x in figs:
        pts = _figure(SPECS[posture], x, sc)
        people.append({"keypoints": [[round(a, 4), round(b, 4), 1.0] for a, b in pts]})
    pj = {"version": "pose0.1", "canvas": 512, "people": people, "source": "preset:" + name, "producer": "C"}
    problems = validate_posejson(pj)
    if problems:
        raise ValueError(f"library preset {name} invalid: " + "; ".join(problems))
    return pj


def index():
    return [{"name": n, "label": v[0], "n_people": len(v[2]), "description": v[1],
             "pose_json": f"{n}.json", "thumbnail": f"{n}.png", "smoke_tested": False}
            for n, v in LIBRARY.items()]


def build(out_dir=LIB_DIR):
    """Write <name>.json, <name>.png (256x256) and library.json. Needs opencv (the renderer); run in Colab."""
    from eval.pose_io import render
    os.makedirs(out_dir, exist_ok=True)
    for name in LIBRARY:
        pj = library_posejson(name)
        json.dump(pj, open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8"))
        render(pj).resize((256, 256)).save(os.path.join(out_dir, f"{name}.png"))
    json.dump(index(), open(os.path.join(out_dir, "library.json"), "w", encoding="utf-8"), indent=1)
    return len(LIBRARY)


if __name__ == "__main__":
    print("built", build(), "presets in", LIB_DIR)

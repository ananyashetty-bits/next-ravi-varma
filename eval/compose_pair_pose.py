"""Compose a multi-person PoseJSON from single-person stock photos (D's pairs; Colab, eval or gen profile).

Each slot = (photo URL or path, which detected person: "largest" or an index, x-centre 0-1, height 0-1, flip).
The person's 18 body keypoints are scaled so their height is `height` of the canvas, their feet stand on the floor
line, their centre is at `x`, and flip mirrors them (swapping left/right joints so anatomy stays correct).
Only the skeleton is kept; photos are read into memory and never written to disk.
"""
import io
import json
import os
import re
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from eval.pose_io import make_posejson, render          # noqa: E402

FLOOR = 0.93
SWAP = {2: 5, 3: 6, 4: 7, 5: 2, 6: 3, 7: 4, 8: 11, 9: 12, 10: 13, 11: 8, 12: 9, 13: 10, 14: 15, 15: 14, 16: 17, 17: 16}


def _url(u):
    m = re.search(r"pexels\.com/photo/[^/]*?-?(\d+)/?$", u.strip())
    if m:                                     # a Pexels photo PAGE -> its image file
        i = m.group(1)
        return f"https://images.pexels.com/photos/{i}/pexels-photo-{i}.jpeg?auto=compress&w=1280"
    return u


def _load(src):
    from PIL import Image, ImageOps
    if src.startswith("http"):
        req = urllib.request.Request(_url(src), headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=60).read()
        img = Image.open(io.BytesIO(data))
    else:
        img = Image.open(src)
    return ImageOps.exif_transpose(img).convert("RGB")


def detect_people(img):
    """-> list of 18x[x, y, c] in the letterboxed 512 frame, largest first."""
    import numpy as np
    from eval.pose_metric import _detector, letterbox
    sq = letterbox(img)
    poses = _detector().detect_poses(np.array(sq), include_hand=False, include_face=False)
    people = [[[float(k.x), float(k.y), 1.0] if k is not None else [0.0, 0.0, 0.0] for k in p.body.keypoints]
              for p in poses]
    return sorted(people, key=_area, reverse=True)


def _bbox(kps):
    pts = [k for k in kps if k[2] > 0]
    xs, ys = [k[0] for k in pts], [k[1] for k in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _area(kps):
    if not any(k[2] > 0 for k in kps):
        return 0.0
    x0, y0, x1, y1 = _bbox(kps)
    return (x1 - x0) * (y1 - y0)


def place(kps, x, height, flip=False, floor=FLOOR):
    """Scale/translate one person: bbox height -> `height`, bbox bottom -> `floor`, bbox centre x -> `x`."""
    if flip:
        kps = [[1.0 - kps[SWAP.get(i, i)][0], kps[SWAP.get(i, i)][1], kps[SWAP.get(i, i)][2]] for i in range(18)]
    x0, y0, x1, y1 = _bbox(kps)
    s = height / max(y1 - y0, 1e-6)
    cx = (x0 + x1) / 2
    out = []
    for k in kps:
        if k[2] <= 0:
            out.append([0.0, 0.0, 0.0])
            continue
        nx, ny = x + (k[0] - cx) * s, floor - (y1 - k[1]) * s
        out.append([round(nx, 4), round(ny, 4), 1.0] if 0 <= nx <= 1 and 0 <= ny <= 1 else [0.0, 0.0, 0.0])
    return out


def compose(slots, people_from=None):
    """slots: list of dicts {src, pick ("largest"|int), x, height, flip}. Returns (PoseJSON, report lines)."""
    people, report, cache = [], [], {}
    for i, s in enumerate(slots):
        if people_from is not None:                       # tests: keypoints given directly
            found = people_from[i]
        else:
            if s["src"] not in cache:
                img = _load(s["src"])
                cache[s["src"]] = detect_people(img)
                del img
            found = cache[s["src"]]
        if not found:
            report.append(f"slot {i}: NO PERSON FOUND in {s['src']}  <-- pick another photo")
            continue
        pick = 0 if s.get("pick", "largest") == "largest" else int(s["pick"])
        kps = found[pick]
        missing = sum(k[2] == 0 for k in kps)
        placed = place(kps, s["x"], s["height"], s.get("flip", False))
        lost = sum(k[2] == 0 for k in placed) - missing
        people.append(placed)
        report.append(f"slot {i}: {len(found)} people in photo, used #{pick}; {18 - missing}/18 joints"
                      + (f"  WARNING: {missing} missing" if missing else "") + (f"; {lost} fell off canvas" if lost > 0 else ""))
    return make_posejson(people, source="photo", producer="D"), report


def build_pair(pid, slots, out_dir="eval/poses/dev"):
    pj, report = compose(slots)
    pj["stock_photos"] = [s["src"] for s in slots]
    os.makedirs(os.path.join(ROOT, out_dir), exist_ok=True)
    jp, pp = os.path.join(ROOT, out_dir, f"{pid}.json"), os.path.join(ROOT, out_dir, f"{pid}.png")
    json.dump(pj, open(jp, "w"))
    render(pj).save(pp)
    print(f"{pid}: n_people={len(pj['people'])}"); [print("   " + r) for r in report]
    return pp


def preview(candidates, size=256):
    """Show each candidate photo next to the skeleton the detector finds in it, with joint counts."""
    from IPython.display import display
    from PIL import Image
    from eval.pose_metric import letterbox
    for src in candidates:
        try:
            img = _load(src)
        except Exception as e:                                   # noqa: BLE001
            print(f"{src}\n   could not download ({type(e).__name__}); skip it"); continue
        people = detect_people(img)
        sq = letterbox(img).resize((size, size))
        del img
        sk = render(make_posejson(people, source="photo", producer="D")).resize((size, size))
        row = Image.new("RGB", (2 * size, size)); row.paste(sq, (0, 0)); row.paste(sk, (size, 0))
        joints = [18 - sum(k[2] == 0 for k in p) for p in people]
        print(f"{src}\n   people found: {len(people)}; joints per person (largest first): {joints}")
        display(row)
        del sq, row


# D's pairs (Overview §10): slot = one figure. Candidates are free-licence Pexels photos of models
# (team decision 21 Nov: no photos of ourselves). Pick by eye with preview(), then build_pair().
CANDIDATES = {
    "s25 figure (leap, one arm raised)": ["https://www.pexels.com/photo/flexible-male-dancer-jumping-in-studio-4226383/",
                                         "https://www.pexels.com/photo/flexible-dancer-rehearsing-on-shore-of-ocean-4226422/",
                                         "https://www.pexels.com/photo/low-angle-shot-of-ballet-dancer-11668020/"],
    "s26 A (seated, hand raised)": ["https://www.pexels.com/photo/man-in-brown-button-up-shirt-sitting-on-brown-wooden-chair-raising-his-hand-5756576/",
                                    "https://www.pexels.com/photo/a-side-view-of-a-man-raising-his-hands-while-sitting-on-the-chair-7862625/"],
    "s26 B (standing, head down)": ["https://www.pexels.com/photo/man-standing-by-chair-with-head-down-5303044/",
                                    "https://www.pexels.com/photo/man-with-head-down-13801996/"],
    "s27 A (woman bending forward)": ["https://www.pexels.com/photo/a-woman-bending-forward-looking-at-the-camera-8027804/",
                                      "https://www.pexels.com/photo/woman-bending-forward-while-leaning-on-wall-2827801/"],
    "s27 B (child sitting on floor)": ["https://www.pexels.com/photo/full-length-of-boy-sitting-on-floor-315265/",
                                       "https://www.pexels.com/photo/kid-sitting-on-the-floor-while-crying-6223620/",
                                       "https://www.pexels.com/photo/happy-asian-little-girl-sitting-on-floor-5063390/"],
    "s31 A (older woman, hands held out)": ["https://www.pexels.com/photo/woman-raising-her-both-hands-1786244/"],
    "s31 B+C (two men standing)": ["https://www.pexels.com/photo/men-in-suits-standing-by-each-other-17362822/",
                                   "https://www.pexels.com/photo/men-in-black-jacket-standing-by-the-window-5483054/"],
}

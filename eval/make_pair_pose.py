"""Photo -> pose skeleton for a test pair (Overview §6.2, §10). Run in Colab (gen or eval profile).

  python eval/make_pair_pose.py --photo me.jpg --id s09 [--split dev] [--full] [--max_people 3]
  python eval/make_pair_pose.py --from_json eval/poses/dev/s01.json          # re-render the PNG from a PoseJSON

Writes eval/poses/<split>/<id>.png (OpenPose skeleton, 512x512, black) and <id>.json (PoseJSON).
--full also writes <id>_full.png with hand + face keypoints (for p8), if your controlnet_aux supports it.

PRIVACY: this script never copies or saves the photo. DELETE THE PHOTO after running it; only the skeleton goes in git.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:] = [ROOT] + [p for p in sys.path if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]

from eval.pose_io import load, render          # noqa: E402
from eval.pose_metric import letterbox, poses_to_posejson, _detector   # noqa: E402


def from_photo(photo, max_people=3, full=False):
    import numpy as np
    from PIL import Image
    img = letterbox(Image.open(photo))                       # square, proportions kept, black bars
    det = _detector()
    poses = det.detect_poses(np.array(img), include_hand=False, include_face=False)
    pj = poses_to_posejson(poses, source="photo", producer=os.environ.get("M4_MEMBER", "?"), max_people=max_people)
    full_img = None
    if full:
        try:
            full_img = det(img, include_body=True, include_hand=True, include_face=True)
        except TypeError:
            print("WARNING: this controlnet_aux has no include_hand/include_face; no _full.png written")
    del img
    return pj, full_img


def report(pj, n_found=None):
    n = len(pj["people"])
    print(f"n_people = {n}" + (f" (detector found {n_found}; kept the {n} largest)" if n_found and n_found > n else ""))
    for i, p in enumerate(pj["people"]):
        missing = sum(1 for k in p["keypoints"] if k[2] == 0)
        print(f"  person {i}: {18 - missing}/18 joints" + ("   WARNING: missing joints (cropped or hidden?)" if missing else ""))
    if n == 0:
        print("WARNING: no person found; retake the photo (full body, plain background, good light)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photo"); ap.add_argument("--id"); ap.add_argument("--split", default="dev")
    ap.add_argument("--full", action="store_true"); ap.add_argument("--max_people", type=int, default=3)
    ap.add_argument("--from_json", help="re-render <same name>.png from an existing PoseJSON")
    a = ap.parse_args()
    import json
    if a.from_json:
        out_png = os.path.splitext(a.from_json)[0] + ".png"
        render(load(a.from_json)).save(out_png); print("wrote", out_png); return
    if not (a.photo and a.id):
        ap.error("--photo and --id are required (or use --from_json)")
    out_dir = os.path.join(ROOT, "eval", "poses", a.split); os.makedirs(out_dir, exist_ok=True)
    pj, full_img = from_photo(a.photo, a.max_people, a.full)
    report(pj)
    json.dump(pj, open(os.path.join(out_dir, f"{a.id}.json"), "w"))
    render(pj).save(os.path.join(out_dir, f"{a.id}.png"))
    if full_img is not None:
        full_img.resize((512, 512)).save(os.path.join(out_dir, f"{a.id}_full.png"))
    print(f"wrote {out_dir}/{a.id}.png and .json" + (" and _full.png" if full_img is not None else ""))
    print(f"NOW DELETE THE PHOTO: {a.photo}")


if __name__ == "__main__":
    main()

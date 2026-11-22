"""Detector failure audit + NSFW false-positive test on REAL Varma paintings (Overview §9, §10; B's guide §2.4).

python eval/audit_real_paintings.py --out results/audit_real_paintings.json
       [--holdout_dir /content/drive/MyDrive/m2/eval/eval_holdout]   # the 15 held-out Varma paintings (never trained on)
       [--extra <images dir or manifest.json> ...]                   # e.g. our p0/p1 outputs, once they exist
       [--detect_cache <Drive>/m4/cache/detections_B.jsonl] [--nsfw_model Falconsai/nsfw_image_detection]

Detector audit: for each image, how many people the pinned OpenPose detector finds and how many body joints per
person. A real painting of people where the detector finds nobody tells us how much a low pose_pck is the detector's
fault, not the model's.

NSFW test: a lightweight image classifier over the same real paintings. Its flag rate on REAL Varma paintings is the
false-positive rate. D's content policy: more than 1 of 15 flagged -> don't enable the filter; else a soft blur+confirm.
Nothing here is written anywhere except --out (paintings are public-domain artworks; no photos of people involved).
"""
import argparse
import glob
import json
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:] = [ROOT] + [p for p in sys.path if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]
IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")


def list_images(src):
    if src.endswith(".json"):
        m = json.load(open(src))
        return [it["path"] for it in m.get("images", []) if os.path.exists(it["path"])]
    return sorted(p for p in glob.glob(os.path.join(src, "*")) if p.lower().endswith(IMG_EXT)
                  and not p.lower().endswith("_pose.png"))


def detector_stats(paths, cache):
    from eval.pose_metric import detect
    rows = []
    for p in paths:
        pj = detect(p, cache)
        joints = [sum(1 for k in person["keypoints"] if k[2] > 0) for person in pj["people"]]
        rows.append({"image": os.path.basename(p), "n_people": len(joints), "joints_per_person": joints})
    n = len(rows)
    found = [r for r in rows if r["n_people"] > 0]
    jp = [j for r in found for j in r["joints_per_person"]]
    return {"n_images": n, "pct_with_person": (100.0 * len(found) / n) if n else None,
            "mean_people_when_found": (sum(r["n_people"] for r in found) / len(found)) if found else None,
            "mean_joints_per_person_of_18": (sum(jp) / len(jp)) if jp else None, "per_image": rows}


def nsfw_stats(paths, model_id, threshold=0.5):
    from transformers import pipeline
    from PIL import Image
    clf = pipeline("image-classification", model=model_id)
    rows = []
    for p in paths:
        out = clf(Image.open(p).convert("RGB"))
        score = next((o["score"] for o in out if o["label"].lower() in ("nsfw", "porn", "sexy", "hentai")), 0.0)
        rows.append({"image": os.path.basename(p), "nsfw_score": float(score), "flagged": bool(score > threshold)})
    return {"model": model_id, "threshold": threshold, "n_images": len(rows),
            "n_flagged": sum(r["flagged"] for r in rows), "per_image": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout_dir", default="/content/drive/MyDrive/m2/eval/eval_holdout")
    ap.add_argument("--extra", nargs="*", default=[], help="folders or manifests of our outputs (detector + NSFW too)")
    ap.add_argument("--detect_cache", default=None)
    ap.add_argument("--nsfw_model", default="Falconsai/nsfw_image_detection")
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if os.path.exists(a.out) and not a.force:
        sys.exit(f"{a.out} exists; use a fresh name (or --force)")
    from eval.pose_metric import DetectCache, detector_version
    cache = DetectCache(a.detect_cache) if a.detect_cache else None

    real = list_images(a.holdout_dir)
    if not real:
        sys.exit(f"no images in {a.holdout_dir} (is Drive mounted? the M2 folder shortcut present?)")
    print(f"{len(real)} real Varma paintings")
    out = {"run_at": datetime.now().isoformat(), "detector": detector_version(),
           "real_paintings": {"dir": a.holdout_dir, "detector": detector_stats(real, cache),
                              "nsfw": nsfw_stats(real, a.nsfw_model)}, "extra": {}}
    for src in a.extra:
        paths = list_images(src)
        out["extra"][src] = {"detector": detector_stats(paths, cache), "nsfw": nsfw_stats(paths, a.nsfw_model)}

    r = out["real_paintings"]
    print(f"\nREAL PAINTINGS ({r['detector']['n_images']}):")
    print(f"  detector: person found in {r['detector']['pct_with_person']:.0f}% | "
          f"mean joints per person {r['detector']['mean_joints_per_person_of_18'] or 0:.1f}/18")
    print(f"  NSFW ({a.nsfw_model}): {r['nsfw']['n_flagged']} of {r['nsfw']['n_images']} flagged "
          f"-> filter {'NOT enabled (more than 1 flagged)' if r['nsfw']['n_flagged'] > 1 else 'may be enabled as a soft blur+confirm'}")
    for src, e in out["extra"].items():
        print(f"\n{src} ({e['detector']['n_images']}): person found in {e['detector']['pct_with_person']:.0f}%, "
              f"joints {e['detector']['mean_joints_per_person_of_18'] or 0:.1f}/18, NSFW flagged {e['nsfw']['n_flagged']}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()

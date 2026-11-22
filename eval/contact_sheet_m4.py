"""M4 blind contact sheets + rating template (B's guide §2.3, Overview §10).

One PNG per pair: the control skeleton on the left (labelled "pose"), then every condition's seed-1 image,
labelled ONLY with random codes. Post-processed images (p5/p6/p7) are shown from their `_512.png` copy when it
exists, and every tile is resized to the same size, so resolution cannot unblind them.

python eval/contact_sheet_m4.py --manifests <p0 manifest> <p1 manifest> ... --pairs eval/test_pairs_dev.json \
       --out /content/drive/MyDrive/m4/rating/dev --key /content/drive/MyDrive/B_private/blind_key_m4_dev.json \
       [--seed_img 1] [--raters A B C D] [--per_image 2] [--calibration 5] [--sheet_name sheet1]

Writes:
  <out>/<pair_id>.png          the sheet
  <out>/rating_template.csv    one row per (pair, code, rater); the first --calibration (pair, code) rows go to ALL
                               raters (shared practice rows), the rest rotate so each image gets --per_image raters
  <key>                        {pair_id: {code: {condition, image}}}   PRIVATE: never inside the repo
"""
import argparse
import csv
import json
import os
import random
import sys
import textwrap

from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:] = [ROOT] + [p for p in sys.path if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]
from eval.score import load_items       # noqa: E402
from eval.contact_sheet import codes    # noqa: E402

TILE, PAD, HEAD = 320, 10, 70
COLUMNS = ["pair_id", "code", "rater", "calibration", "style_1to5", "pose_match_1to5", "faces_1to5",
           "hands_1to5_or_NV", "failure", "notes"]
FAILURES = ("pose ignored | pose followed but scene wrong | extra/missing limbs | merged/duplicated figures | "
            "face distorted | hands malformed | style weakened (photographic) | stiff/over-constrained | "
            "post-processing artifacts | other")


def rated_copy(path):
    """Post-processed images are rated from their 512 copy (same filter everywhere)."""
    base, ext = os.path.splitext(path)
    c = base + "_512" + ext
    return c if os.path.exists(c) else path


def collect(manifests, seed_img, pair_ids):
    by_pair = {}
    for mp in manifests:
        meta, items = load_items(mp)
        for it in items:
            if it.get("seed") != seed_img or it["scene_id"] not in pair_ids or not os.path.exists(it["path"]):
                continue
            by_pair.setdefault(it["scene_id"], []).append((it.get("condition", meta.get("condition", "?")),
                                                           rated_copy(it["path"])))
    return by_pair


def assign(pair_codes, raters, per_image, calibration):
    """pair_codes: [(pair_id, code)] in sheet order -> rating rows."""
    rows = []
    for i, (pid, c) in enumerate(pair_codes):
        calib = i < calibration
        who = raters if calib else [raters[(i + k) % len(raters)] for k in range(min(per_image, len(raters)))]
        for r in who:
            rows.append({"pair_id": pid, "code": c, "rater": r, "calibration": "yes" if calib else "",
                         **{k: "" for k in COLUMNS[4:]}})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--pairs", default=os.path.join(ROOT, "eval", "test_pairs_dev.json"))
    ap.add_argument("--scenes", default=os.path.join(ROOT, "eval", "test_scenes_dev.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", required=True, help="PRIVATE path for the blind key (outside the repo)")
    ap.add_argument("--seed_img", type=int, default=1)
    ap.add_argument("--raters", nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--per_image", type=int, default=2)
    ap.add_argument("--calibration", type=int, default=5)
    ap.add_argument("--rng", type=int, default=0)
    a = ap.parse_args()
    if os.path.abspath(a.key).startswith(os.path.abspath(ROOT)):
        sys.exit("Refusing to write the blind key inside the repo. Use a private Drive path.")
    if os.path.exists(a.key):
        sys.exit(f"{a.key} exists; never overwrite a blind key (use a new name for a second sheet)")

    pairs = {p["id"]: p for p in json.load(open(a.pairs))}
    scenes = {s["id"]: s for s in json.load(open(a.scenes))} if os.path.exists(a.scenes) else {}
    by_pair = collect(a.manifests, a.seed_img, set(pairs))
    rng = random.Random(a.rng)
    os.makedirs(a.out, exist_ok=True)
    key, order = {}, []
    for pid in sorted(by_pair):
        entries = by_pair[pid]
        rng.shuffle(entries)
        cs = codes(len(entries), rng)
        key[pid] = {c: {"condition": cond, "image": img} for c, (cond, img) in zip(cs, entries)}
        order += [(pid, c) for c in cs]
        n = len(entries) + 1
        W = PAD + n * (TILE + PAD)
        sheet = Image.new("RGB", (W, HEAD + TILE + 40), "white")
        d = ImageDraw.Draw(sheet)
        text = scenes.get(pairs[pid]["scene_id"], {}).get("text", "")
        d.text((PAD, 6), f"{pid}: " + "\n".join(textwrap.wrap(text, width=max(40, W // 7))[:4]), fill="black")
        pose = pairs[pid]["pose_file"]
        pose = pose if os.path.isabs(pose) else os.path.join(ROOT, pose)
        sheet.paste(Image.open(pose).convert("RGB").resize((TILE, TILE)), (PAD, HEAD))
        d.text((PAD + TILE // 2 - 14, HEAD + TILE + 10), "pose", fill="black")
        for j, (c, (_, img)) in enumerate(zip(cs, entries), start=1):
            x = PAD + j * (TILE + PAD)
            sheet.paste(Image.open(img).convert("RGB").resize((TILE, TILE), Image.LANCZOS), (x, HEAD))
            d.text((x + TILE // 2 - 8, HEAD + TILE + 10), c, fill="black")
        sheet.save(os.path.join(a.out, f"{pid}.png"))
    rows = assign(order, a.raters, a.per_image, a.calibration)
    with open(os.path.join(a.out, "rating_template.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS); w.writeheader(); w.writerows(rows)
    os.makedirs(os.path.dirname(os.path.abspath(a.key)), exist_ok=True)
    json.dump(key, open(a.key, "w"), indent=1)
    per = {r: sum(1 for x in rows if x["rater"] == r) for r in a.raters}
    print(f"{len(by_pair)} sheets, {len(order)} images, {len(rows)} rating rows {per} -> {a.out}")
    print(f"key -> {a.key} (PRIVATE). Failure options: {FAILURES}")


if __name__ == "__main__":
    main()

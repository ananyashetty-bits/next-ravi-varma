"""Blind contact sheets + rating template (Member B).

One PNG per scene: every condition's seed-1 image side by side, labelled ONLY with random codes.
The code -> condition key is written to a PRIVATE path (not the repo) and revealed after rating.

python eval/contact_sheet.py --manifests <m1.json> <m2.json> ... --split dev \
       --out /content/drive/MyDrive/m3/rating/dev --key /content/drive/MyDrive/B_private/blind_key_dev.json \
       [--scenes eval/test_scenes_dev.json] [--seed_img 1] [--raters A B C D] [--per_image 2]

Writes:
  <out>/<scene_id>.png         the sheet (codes under each image, scene text on top)
  <out>/rating_template.csv    one row per (scene, code, rater) -> import into Google Sheets
  <key>                        {scene_id: {code: {"condition", "model", "image"}}}  (PRIVATE)
"""
import argparse
import csv
import json
import os
import random
import string
import sys
import textwrap

from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from eval.score import load_items  # noqa: E402

TILE, PAD, HEAD = 320, 10, 70


def codes(n, rng):
    pool = [a + b for a in string.ascii_uppercase for b in "23456789"]
    return rng.sample(pool, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", required=True, help="one run_batch manifest per condition")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", required=True, help="PRIVATE path for the blind key (keep it out of the repo)")
    ap.add_argument("--scenes", default=os.path.join(ROOT, "eval", "test_scenes_dev.json"))
    ap.add_argument("--seed_img", type=int, default=1, help="which seed's image to show")
    ap.add_argument("--raters", nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--per_image", type=int, default=2, help="raters per image (>= 2)")
    ap.add_argument("--rng", type=int, default=0)
    a = ap.parse_args()
    if os.path.abspath(a.key).startswith(ROOT):
        sys.exit("Refusing to write the blind key inside the repo. Use a private Drive path.")

    scenes = {s["id"]: s for s in json.load(open(a.scenes))}
    by_scene = {}                                   # scene -> [(condition, model, path)]
    for mp in a.manifests:
        meta, items = load_items(mp)
        model = meta.get("model", "?")
        for it in items:
            if it.get("seed") == a.seed_img and os.path.exists(it["path"]):
                by_scene.setdefault(it["scene_id"], []).append((it.get("condition", "?"), model, it["path"]))

    rng = random.Random(a.rng)
    os.makedirs(a.out, exist_ok=True)
    key, rows = {}, []
    for i, sid in enumerate(sorted(by_scene)):
        entries = by_scene[sid]
        rng.shuffle(entries)
        cs = codes(len(entries), rng)
        key[sid] = {c: {"condition": e[0], "model": e[1], "image": e[2]} for c, e in zip(cs, entries)}

        W = PAD + len(entries) * (TILE + PAD)
        sheet = Image.new("RGB", (W, HEAD + TILE + 40), "white")
        d = ImageDraw.Draw(sheet)
        text = scenes.get(sid, {}).get("text", "")
        d.text((PAD, 6), f"{sid}: " + "\n".join(textwrap.wrap(text, width=max(40, W // 7))[:4]), fill="black")
        for j, (c, e) in enumerate(zip(cs, entries)):
            x = PAD + j * (TILE + PAD)
            sheet.paste(Image.open(e[2]).convert("RGB").resize((TILE, TILE)), (x, HEAD))
            d.text((x + TILE // 2 - 8, HEAD + TILE + 10), c, fill="black")
        sheet.save(os.path.join(a.out, f"{sid}.png"))

        # rotate raters so each image gets >= per_image raters and load is even
        raters = [a.raters[(i + k) % len(a.raters)] for k in range(min(a.per_image, len(a.raters)))]
        for c in cs:
            for r in raters:
                rows.append({"scene_id": sid, "code": c, "rater": r, "faithfulness_1to5": "", "style_1to5": "",
                             "quality_1to5": "", "failure": "", "notes": ""})

    with open(os.path.join(a.out, "rating_template.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["scene_id"])
        w.writeheader()
        w.writerows(rows)
    os.makedirs(os.path.dirname(os.path.abspath(a.key)), exist_ok=True)
    json.dump(key, open(a.key, "w"), indent=1)
    print(f"{len(by_scene)} sheets, {len(rows)} rating rows -> {a.out}; key -> {a.key} (keep private)")


if __name__ == "__main__":
    main()

"""Rebuild a lost blind key from the contact sheets themselves (Member B).

Each contact sheet PNG holds the seed-1 images as tiles, left to right, in the same order as the codes
appear for that scene in the rating template CSV. This script crops every tile, compares it with every
candidate seed-1 image (resized exactly as contact_sheet.py does), and assigns each code the condition of
the image it matches. It refuses to write a key if any tile has no clear match.

    python eval/recover_key.py --sheets <folder with sXX.png> --template <rating_template.csv> \
        --manifests <c0 manifest> <c1 manifest> <c4 manifest> --key <PRIVATE path>/blind_key_dev.json

Output has the same format as contact_sheet.py's key: {scene_id: {code: {"condition", "model", "image"}}}.
"""
import argparse
import csv
import json
import os
import sys

from PIL import Image, ImageChops, ImageStat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from eval.contact_sheet import HEAD, PAD, TILE  # noqa: E402  (same layout constants)
from eval.score import load_items  # noqa: E402

MATCH_MAX = 3.0      # mean absolute pixel difference (0-255) for a tile to count as the same image
MARGIN = 10.0        # the runner-up must be at least this much worse


def codes_by_scene(template):
    out = {}
    with open(template, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            row = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
            cs = out.setdefault(row["scene_id"], [])
            if row["code"] not in cs:
                cs.append(row["code"])
    return out


def diff(a, b):
    return sum(ImageStat.Stat(ImageChops.difference(a, b)).mean) / 3


def recover(sheets, template, manifests, seed_img=1):
    candidates = {}                                   # scene -> [(condition, model, path, tile-sized image)]
    for mp in manifests:
        meta, items = load_items(mp)
        for it in items:
            if it.get("seed") == seed_img and os.path.exists(it["path"]):
                im = Image.open(it["path"]).convert("RGB").resize((TILE, TILE))
                candidates.setdefault(it["scene_id"], []).append(
                    (it.get("condition", "?"), meta.get("model", "?"), it["path"], im))
    key, report, failures = {}, [], []
    for sid, codes in sorted(codes_by_scene(template).items()):
        path = os.path.join(sheets, f"{sid}.png")
        if not os.path.exists(path):
            failures.append(f"{sid}: no sheet {path}")
            continue
        sheet = Image.open(path).convert("RGB")
        cands = candidates.get(sid, [])
        key[sid], used = {}, set()
        for j, code in enumerate(codes):
            x = PAD + j * (TILE + PAD)
            tile = sheet.crop((x, HEAD, x + TILE, HEAD + TILE))
            scored = sorted((diff(tile, c[3]), i) for i, c in enumerate(cands))
            if not scored:
                failures.append(f"{sid}/{code}: no candidate images for this scene")
                continue
            best, i = scored[0]
            runner = scored[1][0] if len(scored) > 1 else float("inf")
            report.append(f"{sid} {code}: best {best:.2f}, runner-up {runner:.2f}")
            if best > MATCH_MAX or runner - best < MARGIN or i in used:
                failures.append(f"{sid}/{code}: no clear match (best {best:.2f}, runner-up {runner:.2f})")
                continue
            used.add(i)
            cond, model, img, _ = cands[i]
            key[sid][code] = {"condition": cond, "model": model, "image": img}
    return key, report, failures


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheets", required=True, help="folder with the round's sXX.png contact sheets")
    ap.add_argument("--template", required=True, help="the round's ORIGINAL rating_template.csv (code order)")
    ap.add_argument("--manifests", nargs="+", required=True, help="one manifest per condition on the sheets")
    ap.add_argument("--key", required=True, help="PRIVATE output path (outside the repo)")
    ap.add_argument("--seed_img", type=int, default=1)
    a = ap.parse_args()
    if os.path.abspath(a.key).startswith(ROOT):
        sys.exit("Refusing to write the blind key inside the repo. Use a private Drive path.")
    key, report, failures = recover(a.sheets, a.template, a.manifests, a.seed_img)
    n = sum(len(v) for v in key.values())
    print(f"matched {n} codes across {len(key)} scenes; worst best-match "
          f"{max(float(r.split('best ')[1].split(',')[0]) for r in report) if report else float('nan'):.2f}")
    if failures:
        print("NOT written; problems:\n  " + "\n  ".join(failures))
        sys.exit(1)
    os.makedirs(os.path.dirname(os.path.abspath(a.key)), exist_ok=True)
    json.dump(key, open(a.key, "w"), indent=1)
    print("key ->", a.key, "(keep private; codes only, no condition names are printed here)")


if __name__ == "__main__":
    main()

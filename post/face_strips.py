"""Face-crop strips for APP.md (owner D): base (p1) | p5 hi-res | p6 restore, same pair and seed, side by side.

  python post/face_strips.py --p1 $IMG/p1_openpose@rrv_v1/dev --p5 $IMG/p5_hires@rrv_v1/dev \
         --p6 $IMG/p6_restore@rrv_v1/dev --pairs s09 s19 s31 --out docs/figures/post_face_strips.png

The crop is centred on the head of the first figure in the pair's control skeleton (eval/poses/dev/<id>.json), which
the generated figure follows; its side is 24% of the image. Every image is cropped in its own resolution (768 for p5),
then resized to 256 px. NOT blind: this is the side-by-side detail view the Overview asks for (section 10,
same-size rule), reported next to B's blind ratings, never instead of them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from PIL import Image, ImageDraw

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
HEAD = (0, 14, 15, 16, 17, 1)          # nose, eyes, ears, then neck as the last resort
SIDE = 0.24


def head_centre(pose_json: dict, person: int = 0) -> tuple[float, float]:
    kps = pose_json["people"][person]["keypoints"]
    pts = [kps[i] for i in HEAD[:-1] if kps[i][2] > 0] or [kps[1]]
    x = sum(p[0] for p in pts) / len(pts)
    y = sum(p[1] for p in pts) / len(pts)
    return x, y


def crop(image: Image.Image, centre: tuple[float, float], side: float = SIDE, size: int = 256) -> Image.Image:
    w, h = image.size
    s = side * min(w, h)
    cx, cy = centre[0] * w, centre[1] * h
    x0 = min(max(cx - s / 2, 0), w - s)
    y0 = min(max(cy - s / 2, 0), h - s)
    return image.crop((round(x0), round(y0), round(x0 + s), round(y0 + s))).resize((size, size), Image.LANCZOS)


def strips(rows: list[tuple[str, list[Image.Image]]], labels=("p1 base", "p5 hi-res", "p6 restore"), size=256):
    pad, top = 4, 22
    out = Image.new("RGB", (len(labels) * (size + pad) + 70, top + len(rows) * (size + pad)), (255, 255, 255))
    d = ImageDraw.Draw(out)
    for j, lab in enumerate(labels):
        d.text((70 + j * (size + pad) + 4, 4), lab, fill=(0, 0, 0))
    for i, (name, crops) in enumerate(rows):
        y = top + i * (size + pad)
        d.text((4, y + size // 2), name, fill=(0, 0, 0))
        for j, c in enumerate(crops):
            out.paste(c, (70 + j * (size + pad), y))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1", required=True); ap.add_argument("--p5", required=True); ap.add_argument("--p6", required=True)
    ap.add_argument("--pairs", nargs="+", default=["s09", "s19", "s31"]); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="docs/figures/post_face_strips.png")
    a = ap.parse_args(argv)
    rows = []
    for pid in a.pairs:
        pj = json.load(open(os.path.join(ROOT, "eval", "poses", "dev", f"{pid}.json"), encoding="utf-8"))
        c = head_centre(pj)
        name = f"{pid}_s{a.seed}.png"
        imgs = [Image.open(os.path.join(d, name)).convert("RGB") for d in (a.p1, a.p5, a.p6)]
        rows.append((f"{pid} s{a.seed}", [crop(im, c) for im in imgs]))
    os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)), exist_ok=True)
    strips(rows).save(os.path.join(ROOT, a.out))
    print("wrote", a.out)


if __name__ == "__main__":
    main()

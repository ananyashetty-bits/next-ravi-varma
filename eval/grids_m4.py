"""M4 comparison grids and the p1/p2 image-similarity check (Member B, 24 Nov 2026).

Cuts the seed-1 tiles out of the blind M4 contact sheets (eval/contact_sheet_m4.py layout: the pair's skeleton in
column 0, then the round's codes), decodes them with each round's key and lays one scene out as
skeleton | p0 | p1 | p2 | p3 | p5 | p6, each labelled with its two raters' mean style / pose / faces / hands.

Grid:
    python eval/grids_m4.py grid --scene s02 --pairs eval/test_pairs_dev.json --scenes eval/test_scenes_dev.json \
        --round <sheets1>/ <key1.json> <ratings1.csv> --round <sheets2>/ <key2.json> <ratings2.csv> \
        --note "..." --out docs/figures/grid_m4_dev_s02.png
Similarity (mean absolute pixel difference, 0-255, between two conditions' tiles on the same sheet):
    python eval/grids_m4.py similarity --round <sheets1>/ <key1.json> <ratings1.csv> \
        --a p1_openpose --b p2_openpose_tuned --ref p0_m3_default --out results/m4_p1_p2_similarity_dev.json
Keys and rating CSVs are B's private copies; nothing from them except condition labels and means is written.
"""
import argparse
import json
import os
import sys
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from eval import tables_m4 as T            # noqa: E402
from eval.contact_sheet_m4 import HEAD, PAD, TILE  # noqa: E402

ORDER = ["p0_m3_default", "p1_openpose", "p2_openpose_tuned", "p3_pose_auto", "p5_hires", "p6_restore"]
LABEL = {"p0_m3_default": "p0  no pose", "p1_openpose": "p1  photo pose", "p2_openpose_tuned": "p2  tuned window",
         "p3_pose_auto": "p3  pose from text", "p5_hires": "p5  p1 + hi-res", "p6_restore": "p6  p1 + face restore"}
OUT_TILE = 240


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def tiles(sheet_png, key, pid):
    sheet = Image.open(sheet_png).convert("RGB")
    out = {"skeleton": sheet.crop((PAD, HEAD, PAD + TILE, HEAD + TILE))}
    for j, (code, v) in enumerate(key[pid].items(), start=1):          # key order = left-to-right tile order
        x = PAD + j * (TILE + PAD)
        out[v["condition"]] = sheet.crop((x, HEAD, x + TILE, HEAD + TILE))
    return out


def load_rounds(rounds):
    keys = [json.load(open(k)) for _, k, _ in rounds]
    ratings = T.load_ratings([c for _, _, c in rounds], [k for _, k, _ in rounds])
    return keys, ratings


def grid(a):
    keys, ratings = load_rounds(a.round)
    pid = a.scene
    imgs = {}
    for (d, _, _), key in zip(a.round, keys):
        imgs.update(tiles(os.path.join(d, f"{pid}.png"), key, pid))
    conds = [c for c in ORDER if c in imgs]
    means = {(m, c): T.pair_means(ratings, m, c).get(pid) for m in T.SCORES for c in conds}
    text = {s["id"]: s["text"] for s in json.load(open(a.scenes))}[pid]
    cols = ["skeleton"] + conds
    W = PAD + len(cols) * (OUT_TILE + PAD)
    lines = textwrap.wrap(f"{pid}: {text}", width=int(W / 8.2))
    top = 12 + 20 * len(lines) + (20 if a.note else 0) + 6
    H = top + OUT_TILE + 64
    g = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(g)
    y = 10
    for ln in lines:
        d.text((PAD, y), ln, fill="black", font=font(16)); y += 20
    if a.note:
        d.text((PAD, y), a.note, fill=(90, 90, 90), font=font(14))
    for j, c in enumerate(cols):
        x = PAD + j * (OUT_TILE + PAD)
        g.paste(imgs[c].resize((OUT_TILE, OUT_TILE), Image.LANCZOS), (x, top))
        d.text((x, top + OUT_TILE + 6), "control skeleton" if c == "skeleton" else LABEL[c], fill="black", font=font(14))
        if c != "skeleton":
            s = lambda m: "-" if means[(m, c)] is None else f"{means[(m, c)]:.1f}"
            d.text((x, top + OUT_TILE + 24), f"style {s('style')}  pose {s('pose_match')}", fill=(40, 40, 40), font=font(13))
            d.text((x, top + OUT_TILE + 41), f"faces {s('faces')}  hands {s('hands')}", fill=(40, 40, 40), font=font(13))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    g.save(a.out, optimize=True)
    print(a.out, cols)


def similarity(a):
    keys, _ = load_rounds(a.round)
    per = {}
    for (d, _, _), key in zip(a.round, keys):
        for pid in sorted(key):
            t = {k: np.asarray(v, dtype=float) for k, v in tiles(os.path.join(d, f"{pid}.png"), key, pid).items()}
            if a.a in t and a.b in t:
                per[pid] = {f"{a.a} vs {a.b}": round(float(np.abs(t[a.a] - t[a.b]).mean()), 2)}
                if a.ref in t:
                    per[pid][f"{a.ref} vs {a.a}"] = round(float(np.abs(t[a.ref] - t[a.a]).mean()), 2)
    ab = [v[f"{a.a} vs {a.b}"] for v in per.values()]
    out = {"note": "mean absolute pixel difference (0-255) between seed-1 tiles (320x320) on the blind contact sheets",
           "n_pairs": len(per), "median": {f"{a.a} vs {a.b}": float(np.median(ab))}, "max": {f"{a.a} vs {a.b}": max(ab)},
           "per_pair": per}
    ref = [v[f"{a.ref} vs {a.a}"] for v in per.values() if f"{a.ref} vs {a.a}" in v]
    if ref:
        out["median"][f"{a.ref} vs {a.a}"] = float(np.median(ref)); out["min_ref"] = min(ref)
    if os.path.exists(a.out):
        sys.exit(f"{a.out} exists; not overwriting")
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_pair"}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("grid", "similarity"):
        p = sub.add_parser(name)
        p.add_argument("--round", nargs=3, action="append", required=True, metavar=("SHEET_DIR", "KEY", "RATINGS_CSV"))
        p.add_argument("--out", required=True)
    g = sub.choices["grid"]
    g.add_argument("--scene", required=True)
    g.add_argument("--pairs", default=os.path.join(ROOT, "eval", "test_pairs_dev.json"))
    g.add_argument("--scenes", default=os.path.join(ROOT, "eval", "test_scenes_dev.json"))
    g.add_argument("--note", default="")
    s = sub.choices["similarity"]
    s.add_argument("--a", required=True); s.add_argument("--b", required=True); s.add_argument("--ref", default="")
    a = ap.parse_args()
    grid(a) if a.cmd == "grid" else similarity(a)


if __name__ == "__main__":
    main()

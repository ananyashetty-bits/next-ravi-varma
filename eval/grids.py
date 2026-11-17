"""Labelled comparison grids for EVAL_REPORT §5 (Member B).

Cuts the seed-1 tiles out of the blind contact sheets (one or more rounds), decodes each code with that round's
key, and lays the images out in a fixed condition order, each labelled with its condition and its mean human
faithfulness / style / quality (two raters). Identical images (same prompt, same seed) are marked.

    python eval/grids.py --scene s27 --scenes eval/test_scenes_dev.json \
        --sheet <round1>/s27.png eval/ratings/blind_key_dev_round1.json \
        --sheet <round2>/s27.png eval/ratings/blind_key_dev_round2.json \
        --ratings eval/ratings/ratings_dev_round1.csv eval/ratings/blind_key_dev_round1.json \
        --ratings eval/ratings/ratings_dev_round2.csv eval/ratings/blind_key_dev_round2.json \
        --note "largest c3 loss vs c1" --out docs/figures/grid_dev_s27.png
"""
import argparse
import hashlib
import json
import os
import sys
import textwrap

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from eval.contact_sheet import HEAD, PAD, TILE  # noqa: E402
from eval.ratings import load_round, norm_condition, per_image  # noqa: E402

ORDER = ["c0_base_raw", "c1_lora_raw", "c2_template", "c4_llm_nokb", "c3_llm_kb_ckb", "c6_llm_kbllm"]
LABEL = {"c0_base_raw": "c0  base, raw text", "c1_lora_raw": "c1  LoRA, raw text", "c2_template": "c2  template",
         "c4_llm_nokb": "c4  LLM, no KB", "c3_llm_kb_ckb": "c3  LLM + KB", "c6_llm_kbllm": "c6  LLM + LLM-written KB"}
OUT_TILE = 256


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                                  # Pillow < 10.1
        return ImageFont.load_default()


def tiles(sheet_png, key, scene):
    sheet = Image.open(sheet_png).convert("RGB")
    out = {}
    for j, (code, v) in enumerate(key[scene].items()):         # key order = left-to-right tile order
        x = PAD + j * (TILE + PAD)
        out[norm_condition(v["condition"])] = sheet.crop((x, HEAD, x + TILE, HEAD + TILE))
    return out


def build(scene, text, sheets, scores, note, out):
    imgs = {}
    for png, key in sheets:
        imgs.update(tiles(png, json.load(open(key)), scene))
    conds = [c for c in ORDER if c in imgs]
    digest = {c: hashlib.md5(imgs[c].tobytes()).hexdigest() for c in conds}
    same = {c: [o for o in conds if o != c and digest[o] == digest[c]] for c in conds}
    W = PAD + len(conds) * (OUT_TILE + PAD)
    lines = textwrap.wrap(f"{scene}: {text}", width=int(W / 8.2))
    top = 14 + 20 * len(lines) + (20 if note else 0) + 8
    H = top + OUT_TILE + 70
    g = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(g)
    y = 10
    for ln in lines:
        d.text((PAD, y), ln, fill="black", font=font(16))
        y += 20
    if note:
        d.text((PAD, y), note, fill=(90, 90, 90), font=font(14))
    for j, c in enumerate(conds):
        x = PAD + j * (OUT_TILE + PAD)
        g.paste(imgs[c].resize((OUT_TILE, OUT_TILE)), (x, top))
        sc = scores.get((c, scene))
        d.text((x, top + OUT_TILE + 6), LABEL[c], fill="black", font=font(14))
        if sc:
            d.text((x, top + OUT_TILE + 24), f"faith {sc['faithfulness']:.1f}  style {sc['style']:.1f}  qual {sc['quality']:.1f}",
                   fill=(40, 40, 40), font=font(13))
        if same[c]:
            d.text((x, top + OUT_TILE + 42), "= same image as " + ", ".join(o.split("_")[0] for o in same[c]),
                   fill=(160, 60, 0), font=font(12))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    g.save(out, optimize=True)
    return conds, same


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--sheet", nargs=2, action="append", required=True, metavar=("PNG", "KEY"))
    ap.add_argument("--ratings", nargs=2, action="append", default=[], metavar=("CSV", "KEY"))
    ap.add_argument("--note", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    text = {s["id"]: s["text"] for s in json.load(open(a.scenes))}[a.scene]
    rows = []
    for i, (c, k) in enumerate(a.ratings, 1):
        rows += load_round(c, k, i)[0]
    scores = per_image(rows) if rows else {}
    conds, same = build(a.scene, text, a.sheet, scores, a.note, a.out)
    print(a.out, conds, {c: v for c, v in same.items() if v})


if __name__ == "__main__":
    main()

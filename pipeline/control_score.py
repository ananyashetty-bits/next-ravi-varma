"""Score and summarise the stage 1 sweep (A). Needs a GPU: m2kit/metrics loads CLIP-L and DINOv2 on CUDA at import.
Run as its own process AFTER control_sweep.py, so SD and the metric models are never resident together."""
import argparse, glob, json, os, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MD = "/content/drive/MyDrive/m2"                       # same reference folders as eval/score.py
IMG_EXT = (".png", ".jpg", ".jpeg")
METRICS = ["style_margin", "max_train_dino", "clip_text", "pose_pck"]   # clip_text here = vs the FULL c1 prompt, not B's clip_raw


def collect(sweep_dir):
    rows = []
    for mp in sorted(glob.glob(os.path.join(sweep_dir, "*", "manifest.json"))):
        m = json.load(open(mp, encoding="utf-8")); c = m["cell"]
        for im in m["images"]:
            p = im["path"]
            if not os.path.exists(p):
                p = os.path.join(os.path.dirname(mp), os.path.basename(p))     # sweep folder was moved
            if os.path.exists(p):
                rows.append(dict(cell=c["name"], control_scale=c["control_scale"], lora_scale=c["lora_scale"],
                                 guidance_end=c["guidance_end"], fixture=im["fixture"], seed=im["seed"],
                                 path=p, prompt=im["prompt"]))
    return rows


def aggregate(rows):
    by = {}
    for r in rows:
        by.setdefault(r["cell"], []).append(r)
    out = []
    for cell, rs in by.items():
        d = dict(cell=cell, kind="window" if cell.startswith("win_") else "grid", control_scale=rs[0]["control_scale"],
                 lora_scale=rs[0]["lora_scale"], guidance_end=rs[0]["guidance_end"], n=len(rs))
        for k in METRICS:
            v = [r[k] for r in rs if r.get(k) is not None]
            if v:
                d[k] = float(np.mean(v)); d[k + "_std"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        v = [r["max_train_dino"] for r in rs if r.get("max_train_dino") is not None]
        if v:
            d["max_train_dino_max"] = float(max(v))
        out.append(d)
    return out


def make_grids(rows, out_dir, size=256):
    from PIL import Image
    g = [r for r in rows if r["cell"].startswith("c")]
    w = [r for r in rows if r["cell"].startswith("win_")]
    controls = sorted({r["control_scale"] for r in g}); loras = sorted({r["lora_scale"] for r in g})
    idx = {(r["fixture"], r["control_scale"], r["lora_scale"], r["seed"]): r["path"] for r in g}
    for f in sorted({r["fixture"] for r in g}):
        canvas = Image.new("RGB", (len(loras) * size, len(controls) * size), (128, 128, 128))
        for i, c in enumerate(controls):
            for j, l in enumerate(loras):
                p = idx.get((f, c, l, 1))
                if p:
                    canvas.paste(Image.open(p).convert("RGB").resize((size, size)), (j * size, i * size))
        canvas.save(os.path.join(out_dir, f"grid_{f}.png"))
        print(f"grid_{f}.png: rows = control {controls}, columns = LoRA {loras} (seed 1)")
    ends = sorted({r["guidance_end"] for r in w}); fx = sorted({r["fixture"] for r in w})
    widx = {(r["fixture"], r["guidance_end"], r["seed"]): r["path"] for r in w}
    if ends:
        canvas = Image.new("RGB", (len(ends) * size, len(fx) * size), (128, 128, 128))
        for i, f in enumerate(fx):
            for j, e in enumerate(ends):
                p = widx.get((f, e, 1))
                if p:
                    canvas.paste(Image.open(p).convert("RGB").resize((size, size)), (j * size, i * size))
        canvas.save(os.path.join(out_dir, "grid_window.png"))
        print(f"grid_window.png: rows = fixtures {fx}, columns = guidance_end {ends} (control 1.0, LoRA 1.0, seed 1)")


def make_plot(table, out_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = [t for t in table if t["kind"] == "grid"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for l in sorted({t["lora_scale"] for t in g}):
        pts = sorted((t for t in g if t["lora_scale"] == l), key=lambda t: t["control_scale"])
        ax[0].plot([t["control_scale"] for t in pts], [t.get("style_margin", np.nan) for t in pts], marker="o", label=f"LoRA {l}")
        if any("pose_pck" in t for t in pts):
            ax[1].plot([t["control_scale"] for t in pts], [t.get("pose_pck", np.nan) for t in pts], marker="o", label=f"LoRA {l}")
    ax[0].set_xlabel("control scale"); ax[0].set_ylabel("style_margin"); ax[0].legend()
    ax[1].set_xlabel("control scale"); ax[1].set_ylabel("pose_pck")
    if not ax[1].lines:
        ax[1].text(0.5, 0.5, "pose_pck pending\n(needs B's detect)", ha="center", va="center", transform=ax[1].transAxes)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "tradeoff.png"), dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--pose_scores", help="optional JSON {image path: pose_pck} once B's detector exists")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = collect(a.sweep_dir)
    assert rows, f"no images found under {a.sweep_dir}"
    cache_p = os.path.join(a.out, "per_image.json")
    cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
    todo = [r for r in rows if r["path"] not in cache]
    print(f"{len(rows)} images, {len(todo)} to score")
    if todo:
        sys.path.insert(0, os.path.join(ROOT, "m2kit"))
        import metrics as M
        ls = lambda d: sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(IMG_EXT))
        R = M.Refs(ls(f"{MD}/eval/eval_holdout"), ls(f"{MD}/eval/control_art"), ls(f"{MD}/dataset/train_v1"))
        per = M.score([r["path"] for r in todo], [r["prompt"] for r in todo], R)
        for i, r in enumerate(todo):
            cache[r["path"]] = {k: float(per[k][i]) for k in ("style_margin", "max_train_dino", "clip_text")}
        json.dump(cache, open(cache_p, "w"), indent=1)
    pose = json.load(open(a.pose_scores)) if a.pose_scores else {}
    for r in rows:
        r.update(cache.get(r["path"], {}))
        if r["path"] in pose:
            r["pose_pck"] = pose[r["path"]]
    table = aggregate(rows)
    json.dump(table, open(os.path.join(a.out, "cells.json"), "w"), indent=1)
    for t in sorted(table, key=lambda t: (t["kind"] != "window", t["lora_scale"], t["control_scale"], t["guidance_end"])):
        print(f"{t['cell']:>14}  n={t['n']}  style_margin={t.get('style_margin', float('nan')):+.4f}  "
              f"max_train_dino_max={t.get('max_train_dino_max', float('nan')):.3f}")
    make_grids(rows, a.out); make_plot(table, a.out)


if __name__ == "__main__":
    main()

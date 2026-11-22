"""Run a post-processing condition over an existing image manifest (owner D).

python post/run_post.py --manifest $IMG/p1_openpose@rrv_v1/dev/manifest.json --condition p5_hires \
       --method hires_img2img --out $IMG/p5_hires@rrv_v1/dev
Methods: hires_img2img | hires_esrgan | restore_gfpgan | hires_restore (p7: img2img, then GFPGAN on the result).

For every source image it writes the result at full size (<scene>_s<seed>.png) AND a Lanczos 512x512 copy
(<scene>_s<seed>_512.png). B's raters see only the _512 copies (same-size rule); eval/score.py scores the full-size
file. Images are never labelled. Existing outputs are skipped (resumable). The manifest has the same shape as
run_batch.py's and records the method, settings, library versions and config hashes. Prompt and negative are read
from the source manifest, never typed.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

METHODS = ("hires_img2img", "hires_esrgan", "restore_gfpgan", "hires_restore")


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def resolve_path(p, manifest_dir):
    """Same lookup as eval/score.py: absolute, next to the manifest, or relative to the repo."""
    if os.path.isabs(p):
        if os.path.exists(p):
            return p
        q = os.path.join(manifest_dir, os.path.basename(p))
        return q if os.path.exists(q) else p
    cand = os.path.join(manifest_dir, p)
    return cand if os.path.exists(cand) else os.path.join(ROOT, p)


def out_names(out_dir, scene_id, seed):
    base = os.path.join(out_dir, f"{scene_id}_s{seed}")
    return base + ".png", base + "_512.png"


def to_512(image):
    return image.convert("RGB").resize((512, 512), Image.LANCZOS)


def load_source(manifest_path):
    m = json.load(open(manifest_path, encoding="utf-8"))
    items = m if isinstance(m, list) else m.get("images") or m.get("items") or []
    d = os.path.dirname(os.path.abspath(manifest_path))
    return (m if isinstance(m, dict) else {}), [{**it, "path": resolve_path(it["path"], d)} for it in items
                                                if it.get("selected", True) is not False]


def make_apply(method, strength=None, weight=None):
    """-> (fn(image, item) -> (image, extra_dict), settings_dict). Imports are lazy (GPU only)."""
    from post import hires as H
    from post import restore as R
    hs = H.hires_settings()
    if strength is not None:
        hs["strength"] = float(strength)
    w = R.DEFAULT_WEIGHT if weight is None else float(weight)

    def _hi(img, it):
        if method.endswith("esrgan"):
            return H.hires(img, it["prompt"], it["negative"], it["seed"], method="esrgan"), {}
        return H.hires(img, it["prompt"], it["negative"], it["seed"], method="img2img",
                       lora_scale=it.get("lora_scale"), strength=hs["strength"]), {}

    def _rest(img, it):
        out, n = R.restore_faces_info(img, "gfpgan", w)
        return out, {"n_faces_restored": n}

    if method in ("hires_img2img", "hires_esrgan"):
        return _hi, {"method": method, **({"strength": hs["strength"], "size": hs["size"]}
                                          if method == "hires_img2img" else {"scale": 2})}
    if method == "restore_gfpgan":
        return _rest, {"method": method, "weight": w}
    if method == "hires_restore":
        def _both(img, it):
            a, _ = _hi(img, it)
            b, e = _rest(a, it)
            return b, e
        return _both, {"method": method, "strength": hs["strength"], "size": hs["size"], "weight": w}
    raise ValueError(f"unknown method {method}")


def run(src_manifest, condition, out_dir, apply_fn, settings, meta=None):
    """Process every image of `src_manifest`; returns the new manifest. `apply_fn(image, item) -> (image, extra)`."""
    os.makedirs(out_dir, exist_ok=True)
    src_meta, items = load_source(src_manifest)
    mpath = os.path.join(out_dir, "manifest.json")
    manifest = json.load(open(mpath, encoding="utf-8")) if os.path.exists(mpath) else {"images": []}
    manifest.update(meta or {})
    manifest.update(condition=condition, post=settings, source_manifest=os.path.abspath(src_manifest),
                    source_condition=src_meta.get("condition"))
    done = {(i["scene_id"], i["seed"]) for i in manifest["images"]}
    for it in items:
        key = (it["scene_id"], it["seed"])
        full, small = out_names(out_dir, *key)
        if key in done and os.path.exists(full) and os.path.exists(small):
            continue
        t0 = time.time()
        img, extra = apply_fn(Image.open(it["path"]).convert("RGB"), it)
        img.save(full)
        to_512(img).save(small)
        row = {k: v for k, v in it.items() if k not in ("path", "gen_seconds")}
        row.update(condition=condition, path=full, path_512=small, size=list(img.size),
                   post_method=settings["method"], post_seconds=round(time.time() - t0, 2),
                   source_path=it["path"], source_sha256=sha_file(it["path"]), **extra)
        manifest["images"] = [r for r in manifest["images"] if (r["scene_id"], r["seed"]) != key] + [row]
        tmp = mpath + ".tmp"
        json.dump(manifest, open(tmp, "w", encoding="utf-8"), indent=2)
        os.replace(tmp, mpath)
        print(f"done {full} ({row['post_seconds']:.1f}s)")
    return manifest


def run_meta():
    def sh(*a):
        try:
            return subprocess.run(a, capture_output=True, text=True, cwd=ROOT).stdout.strip()
        except Exception:
            return "n/a"
    meta = {"git_sha": sh("git", "rev-parse", "--short", "HEAD"),
            "config_hash": sha_file(os.path.join(ROOT, "config", "m3.json"))[:12],
            "m4_config_hash": sha_file(os.path.join(ROOT, "config", "m4.json"))[:12]}
    v = {}
    from importlib import metadata as md
    for pkg in ("torch", "diffusers", "gfpgan", "basicsr", "realesrgan", "torchvision", "numpy"):
        try:
            v[pkg] = md.version(pkg)
        except Exception:
            v[pkg] = "n/a"
    meta["versions"] = v
    try:
        import torch
        meta["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    except Exception:
        meta["gpu"] = "n/a"
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--condition", required=True, help="the run's label, e.g. p5_hires")
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strength", type=float, help="img2img strength override (default: config/m4.json)")
    ap.add_argument("--weight", type=float, help="GFPGAN blend weight (default: 0.5)")
    a = ap.parse_args()
    fn, settings = make_apply(a.method, a.strength, a.weight)
    run(a.manifest, a.condition, a.out, fn, settings, run_meta())


if __name__ == "__main__":
    main()

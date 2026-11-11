"""Generate images for c6 and its paired baseline directly with M2's generator (m2kit/generate.py), because
pipeline/generate_api.py has no generate()/CFG. Same LoRA, scale, sampler, steps, CFG, size and seeds for both
conditions; only the expansion file differs. Resumable. Writes manifest.json in the format eval/score.py reads."""
import argparse, hashlib, json, os, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "m2kit"))
import torch
import generate as m2

ap = argparse.ArgumentParser()
ap.add_argument("--expansions", nargs="+", required=True)
ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
ap.add_argument("--img_root", default="/content/drive/MyDrive/m3/images")
ap.add_argument("--limit", type=int, default=0, help="only the first N scenes per condition (smoke test)")
a = ap.parse_args()

cfg_path = os.path.join(ROOT, "config", "m3.json")
cfg = json.load(open(cfg_path))
mc = cfg["models"]["rrv_v1"]
sha = hashlib.sha256(open(mc["lora_path"], "rb").read()).hexdigest()
assert sha == mc["sha256"], f"LoRA sha256 mismatch: {sha} != {mc['sha256']}"
print("LoRA sha256 ok, scale", mc["scale"])
pipe = m2.load(os.path.dirname(mc["lora_path"]), mc["scale"])

def git_sha():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"

def records(path):
    d = json.load(open(path))
    return d if isinstance(d, list) else d.get("records", [d])

for path in a.expansions:
    recs = records(path)
    if a.limit:
        recs = recs[:a.limit]
    cond = recs[0]["condition"]
    out = os.path.join(a.img_root, f"{cond}@rrv_v1", "dev")
    os.makedirs(out, exist_ok=True)
    mpath = os.path.join(out, "manifest.json")
    if os.path.exists(mpath):
        man = json.load(open(mpath))
    else:
        man = {"git_sha": git_sha(), "config_hash": hashlib.sha256(open(cfg_path, "rb").read()).hexdigest()[:12],
               "model": "rrv_v1", "model_sha256": sha, "versions": m2.versions(),
               "gpu": torch.cuda.get_device_name(0), "generator": "m2kit/generate.py load() + direct pipe call",
               "expansions_file": os.path.relpath(path, ROOT), "images": []}
    done = {(i["scene_id"], i["seed"]) for i in man["images"] if os.path.exists(i["path"])}
    n_new = 0
    for r in recs:
        for seed in a.seeds:
            if (r["scene_id"], seed) in done:
                continue
            p = os.path.join(out, f"{r['scene_id']}_{cond}_s{seed}.png")
            t = time.time()
            g = torch.Generator("cuda").manual_seed(seed)
            img = pipe(r["positive"], negative_prompt=r.get("negative") or cfg["negative"],
                       num_inference_steps=cfg["steps"], guidance_scale=cfg["cfg"],
                       height=cfg["size"], width=cfg["size"], generator=g).images[0]
            img.save(p)
            man["images"].append({"scene_id": r["scene_id"], "condition": cond, "seed": seed,
                                  "prompt": r["positive"], "path": p, "gen_seconds": round(time.time() - t, 2)})
            json.dump(man, open(mpath, "w"), indent=2)
            n_new += 1
            print(f"[{cond}] {r['scene_id']} seed {seed} -> {p} ({time.time() - t:.1f}s)")
    print(f"[{cond}] done: {n_new} new images, {len(man['images'])} in manifest -> {mpath}")

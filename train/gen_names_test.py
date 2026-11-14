"""Generate images for the names test (12 characters x name/description prompt x {rrv_v1, C_names} x seeds 1-4),
or (--mode expansions) the c3 expansions with C_names. Uses M2's generator. LoRA = checkpoint-800 of C_names.
Resumable; one manifest per condition in the format eval/score.py reads."""
import argparse, hashlib, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "m2kit"))
import torch
import generate as m2

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["names_test", "expansions"], default="names_test")
ap.add_argument("--expansions", default="outputs/expansions/c3_llm_kb_ckb_dev.json")
ap.add_argument("--models", nargs="+", default=["rrv_v1", "C_names"])
ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4])
ap.add_argument("--img_root", default="/content/drive/MyDrive/m3/images")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

cfg = json.load(open(os.path.join(ROOT, "config", "m3.json")))
v1 = cfg["models"]["rrv_v1"]
LORAS = {"rrv_v1": (os.path.dirname(v1["lora_path"]), v1["sha256"], v1["scale"]),
         "C_names": ("/content/drive/MyDrive/m2/runs/C_names/checkpoint-800", None, 1.0)}
TAG = {"C_names": "rrv_names", "rrv_v1": "rrv_v1"}

if a.mode == "names_test":
    recs = json.load(open(os.path.join(ROOT, "train", "names_test_prompts.json")))
else:
    d = json.load(open(os.path.join(ROOT, a.expansions)))
    recs = d if isinstance(d, list) else d.get("records", [d])
if a.limit:
    keep = sorted({r["scene_id"] for r in recs})[:a.limit]; recs = [r for r in recs if r["scene_id"] in keep]

pipe = None
for model in a.models:
    d_, sha_cfg, scale = LORAS[model]
    f = os.path.join(d_, "pytorch_lora_weights.safetensors")
    sha = hashlib.sha256(open(f, "rb").read()).hexdigest()
    if sha_cfg: assert sha == sha_cfg, f"{model}: sha256 mismatch"
    if pipe is None: pipe = m2.load(d_, scale)
    else: m2.swap_lora(pipe, d_, scale)
    print(f"[{model}] loaded {f.split('/')[-2]} sha256 {sha[:12]} scale {scale}")
    groups = {}
    for r in recs:
        if a.mode == "names_test":
            label = f"{r['condition']}@{model}"; out = os.path.join(a.img_root, "names_test", label)
        else:
            label = r["condition"]; out = os.path.join(a.img_root, f"{label}@{TAG[model]}", "dev")
        groups.setdefault((label, out), []).append(r)
    for (label, out), rs in groups.items():
        os.makedirs(out, exist_ok=True); mp = os.path.join(out, "manifest.json")
        man = json.load(open(mp)) if os.path.exists(mp) else {"model": TAG[model], "model_sha256": sha, "lora": f,
              "scale": scale, "generator": "m2kit/generate.py + direct pipe call", "versions": m2.versions(), "images": []}
        done = {(i["scene_id"], i["seed"]) for i in man["images"] if os.path.exists(i["path"])}
        n = 0
        for r in rs:
            for seed in a.seeds:
                if (r["scene_id"], seed) in done: continue
                p = os.path.join(out, f"{r['scene_id']}_{label}_s{seed}.png"); t = time.time()
                g = torch.Generator("cuda").manual_seed(seed)
                img = pipe(r["positive"], negative_prompt=r.get("negative") or cfg["negative"], num_inference_steps=cfg["steps"],
                           guidance_scale=cfg["cfg"], height=cfg["size"], width=cfg["size"], generator=g).images[0]
                img.save(p)
                man["images"].append({"scene_id": r["scene_id"], "condition": label, "seed": seed, "prompt": r["positive"],
                                      "path": p, "gen_seconds": round(time.time() - t, 2)})
                json.dump(man, open(mp, "w"), indent=2); n += 1
        print(f"[{label}] {n} new images, {len(man['images'])} in manifest")

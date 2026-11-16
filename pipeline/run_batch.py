import argparse, hashlib, json, os, subprocess, sys, time
import torch, diffusers, transformers

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pipeline.generate_api import generate, CFG
from common.tokens import n_tokens

RAW_MODEL = {"c0_base_raw": "base", "c1_lora_raw": CFG["default_model"]}

def sh(*cmd):
    try: return subprocess.check_output(cmd).decode().strip()
    except Exception: return "unknown"

def build_items(args):
    items = []
    if args.raw:
        if args.condition not in RAW_MODEL:
            sys.exit(f"--condition must be one of {list(RAW_MODEL)}")
        for s in json.load(open(args.raw, encoding="utf-8")):
            raw = s.get("text", s.get("raw", ""))
            prompt = raw if args.condition == "c0_base_raw" else f"{CFG['trigger']}, {raw}, {CFG['style_tail']}"
            items.append(dict(scene_id=s["id"], name=s["id"], condition=args.condition, prompt=prompt,
                              negative=CFG["negative"], model=RAW_MODEL[args.condition], coalition=None))
    elif args.expansions:
        for e in json.load(open(args.expansions, encoding="utf-8")):
            c = e.get("coalition")
            items.append(dict(scene_id=e["scene_id"], name=f"{e['scene_id']}_{c}" if c else e["scene_id"],
                              condition=e["condition"], prompt=e["positive"], negative=e["negative"],
                              model=args.model or CFG["default_model"], coalition=c))
    else:
        sys.exit("give --raw or --expansions")
    return items

def save_manifest(m, path):
    tmp = path + ".tmp"
    json.dump(m, open(tmp, "w", encoding="utf-8"), indent=2)
    os.replace(tmp, path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expansions"); ap.add_argument("--raw"); ap.add_argument("--condition")
    ap.add_argument("--model", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    mpath = os.path.join(args.out, "manifest.json")
    if os.path.exists(mpath):
        manifest = json.load(open(mpath, encoding="utf-8"))
    else:
        manifest = {
            "git_sha": sh("git", "rev-parse", "--short", "HEAD"),
            "config_hash": hashlib.sha256(open("config/m3.json", "rb").read()).hexdigest()[:12],
            "model_sha256": {k: v.get("sha256") for k, v in CFG["models"].items() if v},
            "versions": {"torch": torch.__version__, "diffusers": diffusers.__version__,
                         "transformers": transformers.__version__},
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "images": [],
        }

    for it in build_items(args):
        for seed in args.seeds:
            path = os.path.join(args.out, f"{it['name']}_s{seed}.png")
            if os.path.exists(path):
                continue
            t0 = time.time()
            img = generate(it["prompt"], it["negative"], seed, model=it["model"])  # errors surface on purpose
            secs = time.time() - t0
            img.save(path)
            manifest["images"].append({
                "scene_id": it["scene_id"], "condition": it["condition"], "model": it["model"],
                "seed": seed, "prompt": it["prompt"], "token_count": n_tokens(it["prompt"]),
                "path": path, "gen_seconds": round(secs, 2),
                **({"coalition": it["coalition"]} if it["coalition"] else {}),
            })
            save_manifest(manifest, mpath)
            print(f"done {path} ({secs:.1f}s)")

if __name__ == "__main__":
    main()
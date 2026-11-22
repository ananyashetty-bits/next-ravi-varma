import hashlib, json, os, subprocess, sys, time
from importlib import metadata as md
import torch, diffusers, transformers

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from pipeline.generate_api import generate, CFG
from pipeline import batch_args as BA
from common.tokens import n_tokens

RAW_MODEL = {"c0_base_raw": "base", "c1_lora_raw": CFG["default_model"]}

def sh(*cmd):
    try: return subprocess.check_output(cmd).decode().strip()
    except Exception: return "unknown"

def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()

def ver(pkg):
    try: return md.version(pkg)
    except Exception: return "n/a"

def build_items(args):
    if args.raw:
        if args.condition not in RAW_MODEL:
            sys.exit(f"--condition must be one of {list(RAW_MODEL)} when using --raw")
        items = []
        for s in json.load(open(args.raw, encoding="utf-8")):
            raw = s.get("text", s.get("raw", ""))
            prompt = raw if args.condition == "c0_base_raw" else f"{CFG['trigger']}, {raw}, {CFG['style_tail']}"
            items.append(dict(scene_id=s["id"], name=s["id"], condition=args.condition, prompt=prompt,
                              negative=CFG["negative"], model=RAW_MODEL[args.condition], coalition=None))
        return items
    if args.expansions:
        return BA.expansion_items(json.load(open(args.expansions, encoding="utf-8")),
                                  args.condition, args.model, CFG["default_model"])
    sys.exit("give --raw or --expansions")

def save_manifest(m, path):
    tmp = path + ".tmp"
    json.dump(m, open(tmp, "w", encoding="utf-8"), indent=2)
    os.replace(tmp, path)

def _abs(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)

def pose_for(args, it, pairs, scene_text):
    """Path of the skeleton PNG used for this item."""
    if args.control_source == "pair":
        pr = pairs[it["scene_id"]]
        f = pr.get("pose_full_file") if args.control == "openpose_full" else pr.get("pose_file")
        if not f:
            sys.exit(f"pair {it['scene_id']} has no {'pose_full_file' if args.control == 'openpose_full' else 'pose_file'}")
        return _abs(f)
    out = os.path.join(args.out, f"{it['name']}_pose.png")        # auto: C's pose_from_text, saved next to the images
    if not os.path.exists(out):
        try: from pose import pose_from_text
        except ImportError: from pose.stub import pose_from_text
        if it["scene_id"] not in scene_text:
            sys.exit(f"no scene text for {it['scene_id']} in eval/test_scenes_dev.json")
        pose_from_text(scene_text[it["scene_id"]])["image"].save(out)
    return out

def main():
    args = BA.build_parser().parse_args()
    err = BA.validate(args)
    if err: sys.exit(err)
    os.makedirs(args.out, exist_ok=True)

    items = build_items(args)
    pairs = BA.load_pairs(args.pairs) if args.pairs else None
    if pairs: items = BA.filter_to_pairs(items, pairs)

    ctl, cn_rev, scene_text = None, None, {}
    if args.control != "none":
        from pipeline import control_api as CA
        ctl = BA.settings(args, CA.profile)                       # profile first, sweep flags on top
        cn_rev = CA.CFG4["controlnets"][CA._CN_KEY[args.control]]["revision"]
        if args.control_source == "auto":
            sc = json.load(open("eval/test_scenes_dev.json", encoding="utf-8"))
            sc = sc if isinstance(sc, list) else sc.get("scenes", sc)
            scene_text = {s["id"]: s["text"] for s in sc}

    mpath = os.path.join(args.out, "manifest.json")
    if os.path.exists(mpath):
        manifest = json.load(open(mpath, encoding="utf-8"))
    else:
        manifest = {"images": []}
    manifest.setdefault("git_sha", sh("git", "rev-parse", "--short", "HEAD"))
    manifest.setdefault("config_hash", sha_file("config/m3.json")[:12])
    manifest.setdefault("m4_config_hash", sha_file("config/m4.json")[:12])
    manifest.setdefault("model_sha256", {k: v.get("sha256") for k, v in CFG["models"].items() if v})
    manifest.setdefault("versions", {"torch": torch.__version__, "diffusers": diffusers.__version__,
                                     "transformers": transformers.__version__, "peft": ver("peft"),
                                     "controlnet_aux": ver("controlnet_aux"), "gradio": ver("gradio")})
    manifest.setdefault("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    manifest.update(condition=args.condition, control=args.control, control_profile=args.control_profile,
                    control_source=args.control_source)

    for it in items:
        pose_path = pose_for(args, it, pairs, scene_text) if ctl is not None else None
        for seed in args.seeds:
            path = os.path.join(args.out, f"{it['name']}_s{seed}.png")
            if os.path.exists(path):
                continue
            t0 = time.time()
            if ctl is None:
                img = generate(it["prompt"], it["negative"], seed, model=it["model"], lora_scale=args.lora_scale)
                lora_used = args.lora_scale
            else:
                from PIL import Image
                from pipeline.control_api import generate_controlled
                img = generate_controlled(it["prompt"], it["negative"], seed, Image.open(pose_path).convert("RGB"),
                                          control_type=args.control, model=it["model"], **ctl)
                lora_used = ctl.get("lora_scale")
            if lora_used is None:
                lora_used = 0.0 if it["model"] == "base" else CFG["models"][it["model"]]["scale"]
            secs = time.time() - t0
            img.save(path)
            manifest["images"].append({
                "scene_id": it["scene_id"], "condition": it["condition"], "model": it["model"],
                "seed": seed, "prompt": it["prompt"], "negative": it["negative"],
                "token_count": n_tokens(it["prompt"]), "path": path, "gen_seconds": round(secs, 2),
                "lora_scale": lora_used, "control": args.control, "control_settings": ctl,
                "control_file": pose_path, "control_sha256": sha_file(pose_path) if pose_path else None,
                "controlnet_revision": cn_rev,
                **({"coalition": it["coalition"]} if it["coalition"] else {}),
            })
            save_manifest(manifest, mpath)
            print(f"done {path} ({secs:.1f}s)")

if __name__ == "__main__":
    main()

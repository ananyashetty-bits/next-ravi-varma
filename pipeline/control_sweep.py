"""Stage 1 of the LoRA x ControlNet sweep (A): 3 fixtures x 2 seeds, one cell at a time, resumable.
Generation only. Scoring is separate (m2kit/metrics.py loads CUDA models at import).
Order: window sweep first, then the grid one LoRA column at a time (0.6 last = the first thing to cut)."""
import argparse, hashlib, json, os, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONTROLS = [0.4, 0.6, 0.8, 1.0, 1.2]
LORAS = [1.0, 0.8, 1.2, 0.6]
WINDOW_ENDS = [0.4, 0.6, 0.8, 1.0]
FIXTURES = ["f01", "f02", "f03"]
SEEDS = [1, 2]


def cells(loras=LORAS, controls=CONTROLS, ends=WINDOW_ENDS):
    out = [dict(name=f"win_ge{ge}", control_scale=1.0, lora_scale=1.0, guidance_end=ge) for ge in ends]
    out += [dict(name=f"c{c}_l{l}", control_scale=c, lora_scale=l, guidance_end=1.0) for l in loras for c in controls]
    return out


def fixture_prompts():
    """f01, f02 from M3's fixtures; f03 = dev scene s01 (M3's fixtures have no 3-figure scene). Negative from the records."""
    sys.path.insert(0, ROOT)
    from kb.parser import parse_scene
    from expander.expand import expand
    fx = json.load(open(os.path.join(ROOT, "fixtures", "test_scenes.json"), encoding="utf-8"))
    fx = fx if isinstance(fx, list) else fx.get("scenes", fx)
    out = {}
    for s in fx:
        if s["id"] in ("f01", "f02"):
            t = s.get("text") or s.get("scene_text")
            e = expand(t, parse_scene(t), "raw")
            out[s["id"]] = (e["positive"], e["negative"])
    dev = {r["scene_id"]: r for r in json.load(open(os.path.join(ROOT, "outputs", "expansions", "c1_lora_raw_dev.json"), encoding="utf-8"))}
    out["f03"] = (dev["s01"]["positive"], dev["s01"]["negative"])
    return out


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--loras", nargs="+", type=float, default=LORAS)       # fallback: drop 0.6 first
    ap.add_argument("--fixtures", nargs="+", default=FIXTURES)             # fallback: cut to 2 fixtures
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = ap.parse_args()
    sys.path.insert(0, ROOT)
    from PIL import Image
    from pipeline.control_api import generate_controlled, CFG4

    prompts = fixture_prompts()
    cn_rev = CFG4["controlnets"]["openpose"]["revision"]
    todo = cells(loras=args.loras)
    n_total, n_new, t_start = len(todo) * len(args.fixtures) * len(args.seeds), 0, time.time()
    print(f"{len(todo)} cells, {n_total} images")
    for ci, cell in enumerate(todo, 1):
        d = os.path.join(args.out, cell["name"]); os.makedirs(d, exist_ok=True)
        man = {"cell": cell, "controlnet_revision": cn_rev, "config_m4_sha256": sha(os.path.join(ROOT, "config", "m4.json"))[:12],
               "config_m3_sha256": sha(os.path.join(ROOT, "config", "m3.json"))[:12], "images": []}
        for fid in args.fixtures:
            pose_path = os.path.join(ROOT, "fixtures", "pairs", fid + ".png")
            pos, neg = prompts[fid]
            for seed in args.seeds:
                path = os.path.join(d, f"{fid}_s{seed}.png")
                secs = None
                if not os.path.exists(path):
                    t0 = time.time()
                    img = generate_controlled(pos, neg, seed, Image.open(pose_path).convert("RGB"),
                                              control_scale=cell["control_scale"], guidance_start=0.0,
                                              guidance_end=cell["guidance_end"], lora_scale=cell["lora_scale"])
                    img.save(path); secs = round(time.time() - t0, 2); n_new += 1
                man["images"].append({"fixture": fid, "seed": seed, "path": path, "prompt": pos, "negative": neg,
                                      "control_sha256": sha(pose_path), "gen_seconds": secs})
                json.dump(man, open(os.path.join(d, "manifest.json"), "w"), indent=1)
        print(f"[{ci}/{len(todo)}] {cell['name']} done | new images so far: {n_new} | elapsed {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()

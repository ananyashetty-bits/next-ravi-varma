import argparse, json, os, time, subprocess
import torch
import diffusers
import sys

# Ensure Python can find the pipeline module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline.generate_api import generate, CFG

def get_git_sha():
    try: 
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except: 
        return "unknown"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expansions", help="Path to expansions JSON")
    parser.add_argument("--raw", help="Path to raw scenes JSON (for c0/c1)")
    parser.add_argument("--condition", help="Condition name (if using --raw)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--seeds", nargs='+', type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, "manifest.json")
    
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    else:
        manifest = {
            "git_sha": get_git_sha(),
            "config_hash": "stub", 
            "model": args.model,
            "model_sha256": CFG["models"].get(args.model, {}).get("sha256", "unknown"),
            "versions": {"torch": torch.__version__, "diffusers": diffusers.__version__},
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "images": []
        }

    items = []
    if args.raw:
        with open(args.raw, "r") as f:
            scenes = json.load(f)
            # Handle if scenes is a list or dict
            scene_list = scenes if isinstance(scenes, list) else [scenes]
            for scene in scene_list:
                prompt = scene.get("text", scene.get("raw", ""))
                if args.condition == "c1_lora_raw":
                    prompt = f"{CFG['trigger']}, {prompt}, {CFG['style_tail']}"
                items.append({"scene_id": scene["id"], "condition": args.condition, "prompt": prompt})
    elif args.expansions:
        with open(args.expansions, "r") as f:
            expansions = json.load(f)
            exp_list = expansions if isinstance(expansions, list) else [expansions]
            for exp in exp_list:
                items.append({"scene_id": exp["scene_id"], "condition": exp["condition"], 
                              "prompt": exp["positive"], "negative": exp["negative"]})

    existing_paths = {img["path"] for img in manifest.get("images", [])}

    for item in items:
        for seed in args.seeds:
            img_name = f"{item['scene_id']}_{item['condition']}_s{seed}.png"
            img_path = os.path.join(args.out, img_name)
            
            if img_path in existing_paths or os.path.exists(img_path):
                print(f"Skipping {img_path}, already exists.")
                continue
            
            print(f"Generating {img_path}...")
            start_t = time.time()
            neg = item.get("negative", CFG.get("negative", "ugly, blurry"))
            
            try:
                # Use generate API (Mocked/Errors if no GPU locally, but safe for Colab)
                image = generate(item["prompt"], neg, seed, model=args.model)
                image.save(img_path)
                gen_seconds = time.time() - start_t
            except Exception as e:
                print(f"Generation failed (Normal if running locally without GPU): {e}")
                gen_seconds = 0
                img_path = "failed"
            
            manifest["images"].append({
                "scene_id": item["scene_id"],
                "condition": item["condition"],
                "seed": seed,
                "prompt": item["prompt"],
                "path": img_path,
                "gen_seconds": gen_seconds
            })
            
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

if __name__ == "__main__":
    main()
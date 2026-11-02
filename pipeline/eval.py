import argparse
import json
import os
import imagehash
from PIL import Image

def evaluate_manifest(manifest_path):
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    images = manifest.get("images", [])
    total_requested = len(images)
    
    hashes = []
    generation_times = []
    
    for img_info in images:
        path = img_info.get("path")
        gen_time = img_info.get("gen_seconds", 0)
        if gen_time > 0:
            generation_times.append(gen_time)
            
        if path and path != "failed" and os.path.exists(path):
            try:
                img = Image.open(path)
                hashes.append(str(imagehash.average_hash(img)))
            except Exception:
                pass

    avg_gen_time = (sum(generation_times) / len(generation_times)) if generation_times else 0.0

    metrics = {
        "manifest": manifest_path,
        "model": manifest.get("model", "unknown"),
        "git_sha": manifest.get("git_sha", "unknown"),
        "total_requested": total_requested,
        "valid_images": len(hashes),
        "avg_gen_seconds": round(avg_gen_time, 3),
        "sample_hashes": hashes[:5]
    }
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate batch generation manifests")
    parser.add_argument("--manifest", required=True, help="Path to manifest.json")
    parser.add_argument("--out", required=True, help="Output path for metrics JSON")
    args = parser.parse_args()

    metrics = evaluate_manifest(args.manifest)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Evaluation complete. Saved metrics to {args.out}")

if __name__ == "__main__":
    main()
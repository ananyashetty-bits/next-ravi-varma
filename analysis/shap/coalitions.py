import itertools, json, os

CFG = json.load(open("config/m3.json"))
ORDER = ["char", "action", "setting", "mood", "tail"]

def coalition_prompts(scene):
    out = []
    for k in range(len(ORDER) + 1):
        for S in itertools.combinations(ORDER, k):
            parts = [CFG.get("trigger", "rrvarma style")]
            for p in ORDER:
                if p in S:
                    parts.append(scene["players"][p])
            
            out.append({
                "scene_id": scene["scene_id"], 
                "coalition": "+".join(S) or "empty",
                "condition": "shap", 
                "positive": ", ".join(parts), 
                "negative": CFG.get("negative", "ugly, blurry")
            })
    return out

if __name__ == "__main__":
    scenes = json.load(open("analysis/shap/shap_scenes.json"))
    all_prompts = []
    for s in scenes:
        all_prompts.extend(coalition_prompts(s))
    
    os.makedirs("outputs/expansions", exist_ok=True)
    out_path = "outputs/expansions/shap_dev.json"
    with open(out_path, "w") as f:
        json.dump(all_prompts, f, indent=2)
    print(f"Generated {len(all_prompts)} prompts across {len(scenes)} scenes to {out_path}")
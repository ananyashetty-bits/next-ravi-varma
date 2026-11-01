"""Score a folder of generated images (Member B owns this; everyone runs it on their own images).

python eval/score.py --manifest <images>/manifest.json --out results/<condition>@<model>_<split>.json
       [--scenes eval/test_scenes_dev.json] [--no_judge] [--no_m2] [--cache <judge cache json>]

Per image:
  checklist      weighted mean P(yes) from the judge over the scene's checklist
  clip_raw       CLIP image-text similarity vs the RAW user text (what the user asked for; truncated at 77 tokens)
  style_margin   M2 metric: closer to real Varma than to non-Varma paintings
  max_train_dino M2 metric: similarity to the nearest training image (memorization)
Resumable: judge answers are cached by (image sha256, model, question).
"""
import argparse
import json
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from eval.stats import boot_ci, scene_means  # noqa: E402

MD = "/content/drive/MyDrive/m2"
IMG_EXT = (".png", ".jpg", ".jpeg")


def load_items(manifest_path):
    m = json.load(open(manifest_path))
    items = m if isinstance(m, list) else m.get("items") or m.get("images") or []
    base = os.path.dirname(os.path.abspath(manifest_path))
    out = []
    for it in items:
        p = it.get("path") or it.get("image")
        if not os.path.isabs(p):
            cand = os.path.join(base, p)
            p = cand if os.path.exists(cand) else os.path.join(ROOT, p)
        if it.get("selected", True) is False:
            continue
        out.append({**it, "path": p})
    return (m if isinstance(m, dict) else {}), out


def m2_scores(paths, texts):
    sys.path.insert(0, os.path.join(ROOT, "m2kit"))
    import metrics as M  # M2 kit, unchanged
    ls = lambda d: sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(IMG_EXT))
    R = M.Refs(ls(f"{MD}/eval/eval_holdout"), ls(f"{MD}/eval/control_art"), ls(f"{MD}/dataset/train_v1"))
    per = M.score(paths, texts, R)
    return {"clip_raw": per["clip_text"], "style_margin": per["style_margin"], "max_train_dino": per["max_train_dino"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", default=os.path.join(ROOT, "eval", "test_scenes_dev.json"))
    ap.add_argument("--no_judge", action="store_true", help="skip the VLM checklist judge")
    ap.add_argument("--no_m2", action="store_true", help="skip CLIP/style/memorization (e.g. fixture test without Drive)")
    ap.add_argument("--cache", default=None, help="judge cache (default: Drive m3/cache if mounted, else /content)")
    a = ap.parse_args()

    scenes = {s["id"]: s for s in json.load(open(a.scenes))}
    meta, items = load_items(a.manifest)
    missing = sorted({it["scene_id"] for it in items} - set(scenes))
    if missing:
        print("WARNING: scenes not in", a.scenes, "->", missing, "(they get no checklist score)")
    items = [it for it in items if os.path.exists(it["path"]) or print("missing image:", it["path"])]
    print(f"{len(items)} images to score")

    rows = [{"scene_id": it["scene_id"], "condition": it.get("condition", meta.get("condition", "?")),
             "seed": it.get("seed"), "image": it["path"], "coalition": it.get("coalition"),
             "prompt": it.get("prompt") or it.get("positive")} for it in items]

    judge_id = None
    if not a.no_judge:
        from eval.judge import Judge
        cache = a.cache or ("/content/drive/MyDrive/m3/cache/judge_cache.json"
                            if os.path.isdir("/content/drive/MyDrive") else "/content/judge_cache.json")
        J = Judge(cache_path=cache)
        judge_id = J.model_id
        for i, r in enumerate(rows):
            sc = scenes.get(r["scene_id"])
            if sc and sc.get("checklist"):
                r["checklist"], r["checklist_detail"] = J.checklist(r["image"], sc["checklist"])
            if (i + 1) % 25 == 0:
                print(f"  judged {i + 1}/{len(rows)}")
        J.save()

    if not a.no_m2 and rows:
        texts = [scenes[r["scene_id"]]["text"] if r["scene_id"] in scenes else (r["prompt"] or "") for r in rows]
        m2 = m2_scores([r["image"] for r in rows], texts)
        for i, r in enumerate(rows):
            for k, v in m2.items():
                r[k] = float(v[i])

    metrics = [k for k in ("checklist", "clip_raw", "style_margin", "max_train_dino") if any(k in r for r in rows)]
    summary = {}
    for cond in sorted({r["condition"] for r in rows}):
        summary[cond] = {}
        for k in metrics:
            sm = scene_means(rows, k, cond)
            if sm:
                summary[cond][k] = {"mean": sum(sm.values()) / len(sm), "ci95": boot_ci(sm.values()), "n_scenes": len(sm)}
    per_scene = {cond: {k: scene_means(rows, k, cond) for k in metrics} for cond in summary}

    out = {"manifest": os.path.abspath(a.manifest), "scenes_file": a.scenes, "scored_at": datetime.now().isoformat(),
           "judge": judge_id, "n_images": len(rows), "summary": summary, "per_scene": per_scene, "items": rows}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()

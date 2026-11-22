"""Score a folder of generated images (Member B owns this; everyone runs it on their own images).

python eval/score.py --manifest <images>/manifest.json --out results/<condition>@<model>_<split>.json
       [--scenes eval/test_scenes_dev.json] [--no_judge] [--no_m2] [--cache <judge cache json>]
       [--pairs eval/test_pairs_dev.json] [--detect_cache <Drive>/m4/cache/detections_<you>.jsonl] [--force]

Per image:
  checklist      weighted mean P(yes) from the judge over the scene's checklist
  clip_raw       CLIP image-text similarity vs the RAW user text (what the user asked for; truncated at 77 tokens)
  style_margin   M2 metric: closer to real Varma than to non-Varma paintings
  max_train_dino M2 metric: similarity to the nearest training image (memorization)
M4 (with --pairs):
  the pair's checklist_override replaces the scene checklist for EVERY condition (p0 included);
  with --detect_cache also pose_pck, pose_detected, n_people_out, pose_joint_recall, scored against the skeleton
  actually used (the manifest's control_pose_json for auto-pose conditions, else the pair's pose_json).
Never overwrites an existing --out unless --force.
Resumable: judge answers are cached by (image sha256, model, question); detections by (image sha256, detector).
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

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
        elif not os.path.exists(p) and os.path.exists(os.path.join(base, os.path.basename(p))):
            p = os.path.join(base, os.path.basename(p))   # manifest written on another Drive: image sits next to it
        if it.get("selected", True) is False:
            continue
        it = dict(it)
        if it.get("condition"):
            it["condition"] = re.sub(r"_\d+$", "", it["condition"])   # run_batch sometimes appends _<idx>
        out.append({**it, "path": p})
    return (m if isinstance(m, dict) else {}), out


def m2_scores(paths, texts):
    sys.path.insert(0, os.path.join(ROOT, "m2kit"))
    import metrics as M  # M2 kit, unchanged
    ls = lambda d: sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(IMG_EXT))
    R = M.Refs(ls(f"{MD}/eval/eval_holdout"), ls(f"{MD}/eval/control_art"), ls(f"{MD}/dataset/train_v1"))
    per = M.score(paths, texts, R)
    return {"clip_raw": per["clip_text"], "style_margin": per["style_margin"], "max_train_dino": per["max_train_dino"]}


def load_pairs(path):
    """scene_id -> pair (eval/test_pairs_dev.json)."""
    d = json.load(open(path, encoding="utf-8"))
    d = d if isinstance(d, list) else d.get("pairs", [])
    return {p["scene_id"]: p for p in d}


def apply_overrides(scenes, pairs):
    """A pair's checklist_override replaces its scene's checklist (copy; M3's scene file is never edited)."""
    out = dict(scenes)
    for sid, p in pairs.items():
        if p.get("checklist_override") and sid in out:
            out[sid] = dict(out[sid], checklist=p["checklist_override"])
    return out


def _local(p, manifest_dir):
    if os.path.exists(p):
        return p
    q = os.path.join(manifest_dir, os.path.basename(p))
    return q if os.path.exists(q) else None


def control_pose_for(row, pairs, manifest_dir, cache=None):
    """The skeleton actually used for this image:
    1. the manifest's sidecar PoseJSON (auto-pose conditions write <name>_pose.json next to <name>_pose.png);
    2. else, if the control PNG is the pair's own skeleton (or there is no control: p0), the pair's PoseJSON;
    3. else (an auto pose without a sidecar) None: the image gets no pose flags and a warning. The detector cannot
       read a skeleton drawing back (it is trained on photos of people), so there is no safe fallback."""
    p = row.get("control_pose_json")
    if p and _local(p, manifest_dir):
        return json.load(open(_local(p, manifest_dir), encoding="utf-8"))
    pair = pairs.get(row["scene_id"])
    cf = row.get("control_file")
    if cf and pair and os.path.basename(cf) != os.path.basename(pair.get("pose_file") or ""):
        sidecar = _local(os.path.splitext(cf)[0] + ".json", manifest_dir)
        if sidecar:
            return json.load(open(sidecar, encoding="utf-8"))
        print(f"WARNING: {row['image']}: auto pose without a PoseJSON sidecar; no pose flags")
        return None
    if pair and pair.get("pose_json"):
        return json.load(open(os.path.join(ROOT, pair["pose_json"]), encoding="utf-8"))
    return None


def add_pose_flags(rows, pairs, cache_path, manifest_dir):
    from eval.pose_metric import DetectCache, pose_score
    m4 = json.load(open(os.path.join(ROOT, "config", "m4.json")))["pose_metric"]
    cache = DetectCache(cache_path)
    for i, r in enumerate(rows):
        ctrl = control_pose_for(r, pairs, manifest_dir, cache)
        if ctrl is None:
            continue
        res = pose_score(ctrl, r["image"], cache=cache, pck_thresh=m4["pck_thresh"])
        r["pose_pck"], r["pose_detected"] = res["pck"], float(res["ok"])
        r["n_people_out"], r["n_people_expected"], r["pose_joint_recall"] = res["n_detected"], res["n_expected"], res["joint_recall"]
        if (i + 1) % 25 == 0:
            print(f"  pose-scored {i + 1}/{len(rows)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", default=os.path.join(ROOT, "eval", "test_scenes_dev.json"))
    ap.add_argument("--no_judge", action="store_true", help="skip the VLM checklist judge")
    ap.add_argument("--no_m2", action="store_true", help="skip CLIP/style/memorization (e.g. fixture test without Drive)")
    ap.add_argument("--cache", default=None, help="judge cache (default: Drive m3/cache if mounted, else /content)")
    ap.add_argument("--pairs", default=None, help="M4 test pairs: checklist_override + control pose per scene")
    ap.add_argument("--detect_cache", default=None, help="per-owner pose-detection cache (jsonl, on Drive); enables pose flags")
    ap.add_argument("--force", action="store_true", help="overwrite an existing --out")
    a = ap.parse_args()
    if os.path.exists(a.out) and not a.force:
        sys.exit(f"{a.out} exists; score.py never overwrites results. Use a fresh --out name (or --force if you mean it).")

    scenes = {s["id"]: s for s in json.load(open(a.scenes))}
    pairs = load_pairs(a.pairs) if a.pairs else {}
    scenes = apply_overrides(scenes, pairs)
    meta, items = load_items(a.manifest)
    missing = sorted({it["scene_id"] for it in items} - set(scenes))
    if missing:
        print("WARNING: scenes not in", a.scenes, "->", missing, "(they get no checklist score)")
    items = [it for it in items if os.path.exists(it["path"]) or print("missing image:", it["path"])]
    print(f"{len(items)} images to score")

    rows = [{"scene_id": it["scene_id"], "condition": it.get("condition", meta.get("condition", "?")),
             "seed": it.get("seed"), "image": it["path"], "coalition": it.get("coalition"),
             "prompt": it.get("prompt") or it.get("positive"),
             "control_pose_json": it.get("control_pose_json"), "control_file": it.get("control_file")} for it in items]

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

    if a.detect_cache and rows:
        add_pose_flags(rows, pairs, a.detect_cache, os.path.dirname(os.path.abspath(a.manifest)))

    from eval.stats import boot_ci, scene_means   # numpy; imported here so load_items works without it
    metrics = [k for k in ("checklist", "clip_raw", "style_margin", "max_train_dino", "pose_pck", "pose_detected")
               if any(r.get(k) is not None for r in rows)]
    summary = {}
    for cond in sorted({r["condition"] for r in rows}):
        summary[cond] = {}
        for k in metrics:
            sm = scene_means(rows, k, cond)
            if sm:
                summary[cond][k] = {"mean": sum(sm.values()) / len(sm), "ci95": boot_ci(sm.values()), "n_scenes": len(sm)}
    per_scene = {cond: {k: scene_means(rows, k, cond) for k in metrics} for cond in summary}

    out = {"manifest": os.path.abspath(a.manifest), "scenes_file": a.scenes, "pairs_file": a.pairs,
           "checklist_overrides": sorted(k for k, p in pairs.items() if p.get("checklist_override")),
           "scored_at": datetime.now().isoformat(),
           "judge": judge_id, "n_images": len(rows), "summary": summary, "per_scene": per_scene, "items": rows}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()

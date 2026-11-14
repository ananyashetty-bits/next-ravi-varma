"""Judge robustness check (Member B; pre-registered in experiments/HYPOTHESES.md section B.4).

Pre-registered: "on 20 images, shuffling question order changes the checklist score by < 0.05 on average."

The judge asks each question in a separate forward pass (it never sees the other questions), so question
order can only matter through GPU numerical noise. This check therefore also works as a re-run determinism
test: every answer is recomputed WITHOUT the cache, in a shuffled order, and compared with the stored
score in results/. A difference here can also come from a different library version than on scoring day.

Optional extra (NOT pre-registered; report as descriptive): --paraphrase re-asks the same questions with a
reworded answer instruction to see whether the P(yes) scale depends on that wording.

    python eval/judge_robustness.py --results results/c0_base_raw@base_dev.json \
        results/c1_lora_raw@rrv_v1_dev.json results/c4_llm_nokb@rrv_v1_dev.json \
        --n 20 --out /content/drive/MyDrive/m3/judge_robustness.json [--paraphrase]
"""
import argparse
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

THRESHOLD = 0.05
PARAPHRASE_SUFFIX = " Reply Yes or No."


def pick(results_paths, n, seed=0):
    items = []
    for p in results_paths:
        d = json.load(open(p))
        items += [it for it in d.get("items", []) if it.get("checklist") is not None and it.get("checklist_detail")]
    by_cond = {}
    for it in sorted(items, key=lambda it: (it["condition"], it["scene_id"], it.get("seed") or 0)):
        by_cond.setdefault(it["condition"], []).append(it)
    rng = random.Random(seed)
    for v in by_cond.values():
        rng.shuffle(v)
    out, conds = [], sorted(by_cond)                  # round-robin over conditions: balanced sample
    while len(out) < n and any(by_cond[c] for c in conds):
        for c in conds:
            if by_cond[c] and len(out) < n:
                out.append(by_cond[c].pop())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--scenes", default=os.path.join(ROOT, "eval", "test_scenes_dev.json"))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--paraphrase", action="store_true", help="extra, not pre-registered")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import eval.judge as JM
    scenes = {s["id"]: s for s in json.load(open(a.scenes))}
    chosen = pick(a.results, a.n, a.seed)
    missing = [it["image"] for it in chosen if not os.path.exists(it["image"])]
    if missing:
        sys.exit(f"{len(missing)} images not found (is Drive mounted? is A's images folder in MyDrive/m3/images?):\n"
                 + "\n".join(missing[:5]))

    J = JM.Judge(cache_path=None)                     # no cache: every answer is recomputed
    rng = random.Random(a.seed)
    rows = []
    for k, it in enumerate(chosen, 1):
        cl = list(scenes[it["scene_id"]]["checklist"])
        rng.shuffle(cl)                               # shuffled question order
        new, detail = J.checklist(it["image"], cl)
        row = {"image": it["image"], "condition": it["condition"], "scene_id": it["scene_id"], "seed": it.get("seed"),
               "stored": it["checklist"], "shuffled": new, "abs_diff": abs(new - it["checklist"]),
               "order": [c["id"] for c in cl],
               "max_q_diff": max(abs(detail[q] - it["checklist_detail"][q]) for q in detail if q in it["checklist_detail"])}
        if a.paraphrase:
            JM.SUFFIX, old = PARAPHRASE_SUFFIX, JM.SUFFIX
            J.cache = {}
            try:
                para, _ = J.checklist(it["image"], cl)
            finally:
                JM.SUFFIX = old
                J.cache = {}
            row.update({"paraphrase": para, "para_abs_diff": abs(para - it["checklist"])})
        rows.append(row)
        print(f"{k:2d}/{len(chosen)} {it['condition']:<14} {it['scene_id']} s{it.get('seed')}: "
              f"stored {it['checklist']:.3f}  shuffled {new:.3f}"
              + (f"  paraphrase {row['paraphrase']:.3f}" if a.paraphrase else ""))

    mean_d = sum(r["abs_diff"] for r in rows) / len(rows)
    summary = {"n_images": len(rows), "judge": J.model_id, "threshold": THRESHOLD,
               "mean_abs_diff": mean_d, "max_abs_diff": max(r["abs_diff"] for r in rows),
               "max_question_diff": max(r["max_q_diff"] for r in rows), "pass": mean_d < THRESHOLD}
    try:
        import transformers
        import torch
        summary["versions"] = {"transformers": transformers.__version__, "torch": torch.__version__}
    except Exception:
        pass
    if a.paraphrase:
        summary["paraphrase_suffix"] = PARAPHRASE_SUFFIX
        summary["paraphrase_mean_abs_diff"] = sum(r["para_abs_diff"] for r in rows) / len(rows)
    print(json.dumps(summary, indent=1))
    print(f"Pre-registered check (mean |change| < {THRESHOLD}): {'PASS' if summary['pass'] else 'FAIL'}")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        json.dump({"summary": summary, "rows": rows}, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()

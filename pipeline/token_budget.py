"""Token-budget experiment (H-A1 to H-A3, plus the names check for H-A6).

Prompts are built from the same 8 scenes and name-free player phrases as SHAP (analysis/shap/shap_scenes.json),
so prompt content is controlled: only length, trigger position, the tail or the names change.

  build:   python pipeline/token_budget.py build   [--out outputs/expansions/token_budget_dev.json]
  analyse: python pipeline/token_budget.py analyse --scores results/token_budget@rrv_v1_dev.json
           [--out results/token_budget_summary_dev.json]

Variants (one ExpandedPrompt per scene each; run_batch names images <scene>_<variant>_s<seed>.png):
  char, char_action, char_action_setting, full   length ladder (trigger first, tail only in `full`)
  no_tail        full without the style tail                      (tail on vs off)
  trigger_last   full with the trigger moved to the end            (trigger first vs last)
  with_names     full with the canonical names after the trigger   (H-A6: do names add anything?)
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [ROOT] + [q for q in sys.path if os.path.abspath(q or ".") != _HERE]   # pipeline/eval.py would shadow eval/

SCENES = os.path.join(ROOT, "analysis", "shap", "shap_scenes.json")
VARIANTS = ["char", "char_action", "char_action_setting", "full", "no_tail", "trigger_last", "with_names"]
LADDER = ["char", "char_action", "char_action_setting", "full"]
METRICS = ["checklist", "clip_raw", "style_margin"]
# (label, a, b): reported as b - a, paired over scenes
CONTRASTS = [
    ("length: full vs char only", "char", "full"),
    ("length: char_action_setting vs char_action", "char_action", "char_action_setting"),
    ("trigger last vs first", "full", "trigger_last"),
    ("tail on vs off", "no_tail", "full"),
    ("names added", "full", "with_names"),
]
# not pre-registered; reported separately as exploratory
EXPLORATORY = [
    ("mood added (no_tail vs char_action_setting)", "char_action_setting", "no_tail"),
    ("mood + tail added (full vs char_action_setting)", "char_action_setting", "full"),
]
EXPANSIONS = os.path.join(ROOT, "outputs", "expansions", "token_budget_dev.json")


def _cfg():
    return json.load(open(os.path.join(ROOT, "config", "m3.json"), encoding="utf-8"))


def join_names(names):
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def variant_parts(scene, variant, trigger):
    p = scene["players"]
    body = {
        "char": [p["char"]],
        "char_action": [p["char"], p["action"]],
        "char_action_setting": [p["char"], p["action"], p["setting"]],
        "full": [p["char"], p["action"], p["setting"], p["mood"], p["tail"]],
        "no_tail": [p["char"], p["action"], p["setting"], p["mood"]],
        "trigger_last": [p["char"], p["action"], p["setting"], p["mood"], p["tail"]],
        "with_names": [join_names(scene["names"]), p["char"], p["action"], p["setting"], p["mood"], p["tail"]],
    }[variant]
    return body + [trigger] if variant == "trigger_last" else [trigger] + body


def build(scenes, cfg):
    out = []
    for sc in scenes:
        for v in VARIANTS:
            out.append({"scene_id": sc["scene_id"], "coalition": v, "condition": "token_budget",
                        "positive": ", ".join(variant_parts(sc, v, cfg["trigger"])), "negative": cfg["negative"]})
    return out


def scene_means(items, metric):
    """-> {variant: {scene_id: mean over seeds}}; raises on missing scores or uneven seeds."""
    acc, seeds = {}, {}
    for r in items:
        val = r.get(metric)
        if val is None or val != val:
            raise ValueError(f"missing {metric} for {r['scene_id']} {r.get('coalition')} s{r.get('seed')}")
        k = (r["coalition"], r["scene_id"])
        acc.setdefault(k, []).append(float(val)); seeds.setdefault(k, set()).add(r["seed"])
    if len({tuple(sorted(s)) for s in seeds.values()}) != 1:
        raise ValueError("variants were generated with different seed sets")
    out = {}
    for (v, sid), vals in acc.items():
        out.setdefault(v, {})[sid] = sum(vals) / len(vals)
    return out


def analyse(items, expansions=None):
    from eval.stats import paired, boot_ci
    variants = sorted({r["coalition"] for r in items})
    if set(variants) != set(VARIANTS):
        raise ValueError(f"expected variants {VARIANTS}, found {variants}")
    tokens = {}
    src = items if all(r.get("token_count") is not None for r in items) else (expansions or [])   # score.py drops token_count
    for r in src:
        if r.get("token_count") is not None:
            tokens.setdefault(r["coalition"], []).append(r["token_count"])
    res = {"n_images": len(items), "n_scenes": len({r["scene_id"] for r in items}),
           "mean_tokens": {v: sum(t) / len(t) for v, t in tokens.items()}, "means": {}, "contrasts": {}}
    for m in METRICS:
        sm = scene_means(items, m)
        res["means"][m] = {v: {"mean": sum(sm[v].values()) / len(sm[v]), "ci95": boot_ci(list(sm[v].values()))}
                           for v in VARIANTS}
        res["contrasts"][m] = {lab: paired(sm[a], sm[b]) for lab, a, b in CONTRASTS + EXPLORATORY}
    return res


def to_markdown(res):
    L = [f"{res['n_scenes']} scenes × 2 seeds, {res['n_images']} images. Means over scenes, 95% bootstrap CI.", "",
         "| Variant | Tokens (mean) | " + " | ".join(METRICS) + " |", "|---|---|" + "---|" * len(METRICS)]
    for v in VARIANTS:
        cells = [f"{res['means'][m][v]['mean']:.3f} [{res['means'][m][v]['ci95'][0]:.3f}, {res['means'][m][v]['ci95'][1]:.3f}]"
                 for m in METRICS]
        L.append(f"| `{v}` | {res['mean_tokens'].get(v, float('nan')):.1f} | " + " | ".join(cells) + " |")
    L += ["", "Paired differences (b − a, scene is the unit; win/tie/loss = scenes where b is higher/equal/lower):", "",
          "| Contrast | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    for lab, _a, _b in CONTRASTS + EXPLORATORY:
        cells = []
        for m in METRICS:
            c = res["contrasts"][m][lab]
            cells.append(f"{c['diff']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] ({c['win']}/{c['tie']}/{c['loss']})")
        L.append(f"| {lab}{' (exploratory)' if (lab, _a, _b) in EXPLORATORY else ''} | " + " | ".join(cells) + " |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--scenes", default=SCENES)
    b.add_argument("--out", default=os.path.join(ROOT, "outputs", "expansions", "token_budget_dev.json"))
    a = sub.add_parser("analyse"); a.add_argument("--scores", default=os.path.join(ROOT, "results", "token_budget@rrv_v1_dev.json"))
    a.add_argument("--out", default=os.path.join(ROOT, "results", "token_budget_summary_dev.json"))
    a.add_argument("--expansions", default=EXPANSIONS, help="token counts, if the scores file has none")
    args = ap.parse_args()
    if args.cmd == "build":
        recs = build(json.load(open(args.scenes, encoding="utf-8")), _cfg())
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(recs, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"wrote {len(recs)} expansions to {args.out}")
    else:
        items = json.load(open(args.scores, encoding="utf-8"))["items"]
        exp = json.load(open(args.expansions, encoding="utf-8")) if os.path.exists(args.expansions) else None
        res = analyse(items, exp)
        json.dump(res, open(args.out, "w", encoding="utf-8"), indent=2)
        print(to_markdown(res)); print("wrote", args.out)


if __name__ == "__main__":
    main()

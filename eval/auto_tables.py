"""Automated-metric tables for EVAL_REPORT.md §2.2 (Member B), computed from results/*_dev.json only.

    python eval/auto_tables.py [--results "results/*_dev.json"] [--scenes eval/test_scenes_dev.json] [--out FILE]

Unit of analysis is the scene (seeds averaged, as stored in each file's per_scene). Paired comparisons use
eval/stats.paired (5000 resamples, seed 0) on the scenes both conditions share, tie band 0.0295 (B.2).
max_train_dino p95 is over images, as in M2 (guardrail: <= 0.85).
"""
import argparse
import glob
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

METRICS = ("checklist", "clip_raw", "style_margin", "max_train_dino")
TIE_AUTO = 0.0295
MEM_P95 = 0.85
# (a, b, label). Short names are matched exactly against condition ids.
PAIRS = [
    ("c0_base_raw", "c1_lora_raw", "LoRA (Overview §8)"),
    ("c1_lora_raw", "c3_llm_kb_ckb", "full system vs raw text (Overview §8)"),
    ("c4_llm_nokb", "c3_llm_kb_ckb", "adding the KB (Overview §8; H-D2)"),
    ("c2_template", "c3_llm_kb_ckb", "LLM vs template (Overview §8)"),
    ("c6_llm_kbllm", "c3_llm_kb_ckb", "grounded vs LLM-written KB (Overview §8; C)"),
    ("c2_template", "c3_llm_kb", "LLM vs template, both on interim KB (H-D1 as run by D)"),
    ("c3_llm_kb", "c3_llm_kb_ckb", "interim KB → C's KB (same LLM and prompt)"),
    ("c1_lora_raw", "c2_template", "template vs raw text"),
    ("c1_lora_raw", "c4_llm_nokb", "LLM without KB vs raw text"),
]


def load(paths):
    conds = {}
    for p in sorted(paths):
        d = json.load(open(p))
        if "per_scene" not in d:
            continue
        for c, per in d["per_scene"].items():
            items = [it for it in d.get("items", []) if it.get("condition") == c]
            conds[c] = {"file": os.path.relpath(p, ROOT), "per_scene": per, "items": items, "judge": d.get("judge")}
    return conds


def p95(x):
    x = sorted(x)
    if not x:
        return float("nan")
    k = (len(x) - 1) * 0.95                     # linear interpolation, as numpy.quantile
    f = math.floor(k)
    return x[f] + (x[min(f + 1, len(x) - 1)] - x[f]) * (k - f)


def f(x, nd=3, sign=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def tables(conds, scenes):
    from eval.stats import boot_ci, paired
    order = sorted(conds)
    common = set.intersection(*(set(conds[c]["per_scene"]["checklist"]) for c in order)) if order else set()
    L = [f"Sources: {', '.join('`' + conds[c]['file'] + '`' for c in order)}. Judge: "
         f"{', '.join(sorted({str(conds[c]['judge']) for c in order}))}. Seeds 1-2 averaged per scene.", ""]

    L += ["#### Per condition (mean over scenes [95% CI])", "",
          "| Condition | n scenes | checklist (judge) | checklist on the " + f"{len(common)} common scenes"
          + " | `clip_raw` | `style_margin` | `max_train_dino` p95 (≤ 0.85) |", "|---|---|---|---|---|---|---|"]
    out = {"common_scenes": sorted(common), "conditions": {}, "paired": []}
    for c in order:
        ps = conds[c]["per_scene"]
        row = {}
        for m in ("checklist", "clip_raw", "style_margin"):
            v = list(ps[m].values())
            row[m] = {"mean": sum(v) / len(v), "ci95": boot_ci(v), "n": len(v)}
        cc = [ps["checklist"][s] for s in common]
        row["checklist_common"] = sum(cc) / len(cc) if cc else float("nan")
        row["dino_p95"] = p95([it["max_train_dino"] for it in conds[c]["items"] if "max_train_dino" in it])
        out["conditions"][c] = row
        L.append(f"| `{c}` | {row['checklist']['n']} | {f(row['checklist']['mean'])} "
                 f"[{f(row['checklist']['ci95'][0])}, {f(row['checklist']['ci95'][1])}] | {f(row['checklist_common'])} | "
                 f"{f(row['clip_raw']['mean'])} | {f(row['style_margin']['mean'])} | {f(row['dino_p95'])}"
                 f"{'' if row['dino_p95'] <= MEM_P95 else ' **over**'} |")

    L += ["", f"#### Paired comparisons (b − a per scene, shared scenes; [95% CI]; W/T/L with tie band {TIE_AUTO})", "",
          "| a → b | What it tests | n | checklist (judge) | `clip_raw` | `style_margin` |", "|---|---|---|---|---|---|"]
    for a, b, label in PAIRS:
        if a not in conds or b not in conds:
            continue
        r = {"a": a, "b": b, "label": label}
        cells = []
        for m in ("checklist", "clip_raw", "style_margin"):
            p = paired(conds[a]["per_scene"][m], conds[b]["per_scene"][m], tie=TIE_AUTO)
            r[m] = p
            bold = p["ci95"][0] > 0 or p["ci95"][1] < 0
            txt = f"{f(p['diff'], sign=True)} [{f(p['ci95'][0], sign=True)}, {f(p['ci95'][1], sign=True)}] {p['win']}/{p['tie']}/{p['loss']}"
            cells.append(f"**{txt}**" if bold else txt)
        r["n"] = r["checklist"]["n"]
        out["paired"].append(r)
        L.append(f"| `{a}` → `{b}` | {label} | {r['n']} | " + " | ".join(cells) + " |")
    L.append("\nBold: the 95% CI excludes 0.")

    def split(fn, title):
        groups = sorted({str(fn(s)) for c in order for s in conds[c]["per_scene"]["checklist"] if fn(s) is not None})
        T = ["", f"#### Checklist (judge) {title} (mean, n scenes)", "", "| Condition | " + " | ".join(groups) + " |",
             "|---" * (len(groups) + 1) + "|"]
        for c in order:
            g = {}
            for s, v in conds[c]["per_scene"]["checklist"].items():
                k = fn(s)
                if k is not None:
                    g.setdefault(str(k), []).append(v)
            T.append(f"| `{c}` | " + " | ".join(f"{f(sum(g[k]) / len(g[k]))} ({len(g[k])})" if k in g else "–"
                                                 for k in groups) + " |")
        return T
    L += split(lambda s: scenes.get(s, {}).get("tier"), "by tier")
    L += split(lambda s: {True: "seen", False: "unseen"}.get(scenes.get(s, {}).get("seen_in_training")),
               "seen vs unseen in training")
    return "\n".join(L) + "\n", out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(ROOT, "results", "*_dev.json"))
    ap.add_argument("--scenes", default=os.path.join(ROOT, "eval", "test_scenes_dev.json"))
    ap.add_argument("--out", default=None, help="write markdown here (and <out>.json)")
    a = ap.parse_args()
    scenes = {s["id"]: s for s in json.load(open(a.scenes))}
    md, data = tables(load(glob.glob(a.results)), scenes)
    print(md)
    if a.out:
        open(a.out, "w").write(md)
        json.dump(data, open(a.out + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()

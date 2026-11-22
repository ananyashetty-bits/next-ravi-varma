"""M4 results tables from blind ratings + automated scores (B's guide §2.6, §3.1; Overview §8, §10).

python eval/tables_m4.py --ratings <filled rating CSV> [<second sheet CSV> ...] --keys <blind key json> [...] \
       --results results/p0_m3_default@rrv_v1_dev_pose.json results/p1_openpose@rrv_v1_dev_pose.json ... \
       --pairs eval/test_pairs_dev.json --out results/m4_tables_dev.json [--md results/m4_tables_dev.md]

Unit = the pair (seed-1 image, raters averaged). Paired comparisons (Overview §8): diff = b - a, 95% bootstrap CI
over pairs (5000), win/tie/loss with the pre-registered bands (ratings 0.25, pose_pck 0.05, style_margin 0.0295).
Faces and hands are always separate. "NV" (hands not visible) is not a score.
"""
import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:] = [ROOT] + [p for p in sys.path if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]

SCORES = {"style": "style_1to5", "pose_match": "pose_match_1to5", "faces": "faces_1to5", "hands": "hands_1to5_or_NV"}
AUTO = ["checklist", "clip_raw", "style_margin", "pose_pck", "pose_detected", "max_train_dino"]
TIE = {"style": 0.25, "pose_match": 0.25, "faces": 0.25, "hands": 0.25, "pose_pck": 0.05, "style_margin": 0.0295,
       "checklist": 0.0295, "clip_raw": 0.0, "pose_detected": 0.0, "max_train_dino": 0.0}
COMPARISONS = [("p0_m3_default", "p1_openpose"), ("p1_openpose", "p2_openpose_tuned"), ("p1_openpose", "p3_pose_auto"),
               ("p1_openpose", "p5_hires"), ("p1_openpose", "p6_restore"), ("p3_pose_auto", "p4_canny_painting"),
               ("p1_openpose", "p8_openpose_full"), ("p0_m3_default", "p2_openpose_tuned")]


def score(v):
    v = (v or "").strip()
    try:
        x = int(float(v))
        return x if 1 <= x <= 5 else None
    except ValueError:
        return None                        # blank, "NV" (hands not visible), anything else


def load_ratings(csvs, keys):
    """-> list of {pair_id, condition, rater, calibration, style, pose_match, faces, hands, failures}"""
    key = {}
    for k in keys:
        for pid, codes in json.load(open(k)).items():
            for c, e in codes.items():
                key[(pid, c)] = e["condition"]
    out, unknown = [], set()
    for path in csvs:
        for r in csv.DictReader(open(path, encoding="utf-8")):
            k = (r["pair_id"].strip(), r["code"].strip())
            if k not in key:
                unknown.add(k); continue
            row = {"pair_id": k[0], "condition": key[k], "rater": r["rater"].strip(),
                   "calibration": r.get("calibration", "").strip().lower() == "yes",
                   "failures": [f.strip().lower() for f in (r.get("failure") or "").replace(";", ",").split(",") if f.strip()]}
            for s, col in SCORES.items():
                row[s] = score(r.get(col))
            out.append(row)
    if unknown:
        print(f"WARNING: {len(unknown)} (pair, code) not in any key, e.g. {sorted(unknown)[:3]}")
    return out


def pair_means(rows, metric, condition):
    d = defaultdict(list)
    for r in rows:
        if r["condition"] == condition and r.get(metric) is not None:
            d[r["pair_id"]].append(r[metric])
    return {p: sum(v) / len(v) for p, v in d.items()}


def auto_rows(results_files):
    rows = []
    for f in results_files:
        for it in json.load(open(f))["items"]:
            rows.append({"pair_id": it["scene_id"], "condition": it["condition"], "seed": it.get("seed"),
                             **{m: it.get(m) for m in AUTO}, "figure_count_ok":
                             (None if it.get("n_people_out") is None else float(it["n_people_out"] == it.get("n_people_expected")))})
    return rows


def spearman(a, b):
    from eval.ratings import spearman as sp
    return sp(a, b)


def agreement(rows):
    """Per score: Spearman and quadratic-weighted kappa between the two raters of each image."""
    from eval.ratings import weighted_kappa
    by_img = defaultdict(dict)
    for r in rows:
        by_img[(r["pair_id"], r["condition"])][r["rater"]] = r
    out = {}
    for s in SCORES:
        a, b = [], []
        for raters in by_img.values():
            vals = [v[s] for v in raters.values() if v[s] is not None]
            if len(vals) >= 2:
                a.append(vals[0]); b.append(vals[1])
        out[s] = {"n_images": len(a), "spearman": spearman(a, b), "weighted_kappa": weighted_kappa(a, b) if a else float("nan")}
    return out


def build(ratings, autos, pairs):
    from eval.stats import boot_ci, paired
    conds = sorted({r["condition"] for r in ratings} | {r["condition"] for r in autos})
    res = {"conditions": conds, "n_pairs": len({r["pair_id"] for r in ratings}), "means": {}, "comparisons": {},
           "by_tier": {}, "by_people": {}, "failures": {}, "validation": {}, "agreement": agreement(ratings)}
    srcs = {**{s: ratings for s in SCORES}, **{m: autos for m in AUTO + ["figure_count_ok"]}}
    for m, rows in srcs.items():
        res["means"][m] = {}
        for c in conds:
            pm = pair_means(rows, m, c)
            if pm:
                v = list(pm.values())
                res["means"][m][c] = {"mean": sum(v) / len(v), "ci95": boot_ci(v), "n_pairs": len(v),
                                      "range": [min(v), max(v)]}
    for a, b in COMPARISONS:
        if a in conds and b in conds:
            res["comparisons"][f"{a} -> {b}"] = {m: paired(pair_means(rows, m, a), pair_means(rows, m, b), tie=TIE.get(m, 0.0))
                                                for m, rows in srcs.items() if pair_means(rows, m, a) and pair_means(rows, m, b)}
    for label, field, out in (("tier", "tier", res["by_tier"]), ("people", "n_people_expected", res["by_people"])):
        groups = defaultdict(set)
        for pid, p in pairs.items():
            groups[str(p[field])].add(pid)
        for g, ids in sorted(groups.items()):
            out[g] = {}
            for a, b in COMPARISONS[:3]:
                if a in conds and b in conds:
                    out[g][f"{a} -> {b}"] = {m: paired({k: v for k, v in pair_means(ratings, m, a).items() if k in ids},
                                                       {k: v for k, v in pair_means(ratings, m, b).items() if k in ids},
                                                       tie=TIE[m]) for m in ("pose_match", "faces", "hands", "style")}
    for c in conds:
        cnt = defaultdict(int)
        for r in ratings:
            if r["condition"] == c:
                for f in r["failures"]:
                    cnt[f] += 1
        res["failures"][c] = dict(sorted(cnt.items(), key=lambda x: -x[1]))
    # metric validation, per image (condition x pair, seed 1, raters averaged)
    img_h = defaultdict(lambda: defaultdict(list))
    for r in ratings:
        for s in SCORES:
            if r[s] is not None:
                img_h[(r["pair_id"], r["condition"])][s].append(r[s])
    img_a = {(r["pair_id"], r["condition"]): r for r in autos if r.get("seed") in (1, None)}
    for auto_m, human in (("pose_pck", "pose_match"), ("style_margin", "style")):
        xs, ys = [], []
        for k, h in img_h.items():
            if k in img_a and img_a[k].get(auto_m) is not None and h.get(human):
                xs.append(img_a[k][auto_m]); ys.append(sum(h[human]) / len(h[human]))
        res["validation"][f"{auto_m} vs human {human}"] = {"n_images": len(xs), "spearman": spearman(xs, ys)}
    return res


def fmt(x, nd=2):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def to_markdown(res):
    L = [f"Pairs rated: {res['n_pairs']}. Means over pairs (seed 1, raters averaged), 95% bootstrap CI.", "",
         "| Condition | style | pose match | faces | hands | pose_pck | figure count ok | style_margin | checklist | max_train_dino |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for c in res["conditions"]:
        cells = []
        for m in ("style", "pose_match", "faces", "hands", "pose_pck", "figure_count_ok", "style_margin", "checklist", "max_train_dino"):
            e = res["means"].get(m, {}).get(c)
            cells.append("–" if not e else f"{e['mean']:.2f} [{e['ci95'][0]:.2f}, {e['ci95'][1]:.2f}]")
        L.append(f"| `{c}` | " + " | ".join(cells) + " |")
    L += ["", "Paired comparisons (b − a; win/tie/loss = pairs where b is higher/equal/lower within the band):", "",
          "| Comparison | style | pose match | faces | hands | pose_pck | style_margin |", "|---|---|---|---|---|---|---|"]
    for lab, ms in res["comparisons"].items():
        cells = []
        for m in ("style", "pose_match", "faces", "hands", "pose_pck", "style_margin"):
            c = ms.get(m)
            cells.append("–" if not c or not c["n"] else f"{c['diff']:+.2f} [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}] ({c['win']}/{c['tie']}/{c['loss']})")
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    L += ["", "Agreement between the two raters of each image:", "", "| Score | n | Spearman | weighted kappa |", "|---|---|---|---|"]
    for s, a in res["agreement"].items():
        L.append(f"| {s} | {a['n_images']} | {fmt(a['spearman'])} | {fmt(a['weighted_kappa'])} |")
    L += ["", "Metric validation (per image):", ""] + [f"- {k}: Spearman {fmt(v['spearman'])} (n={v['n_images']})"
                                                     for k, v in res["validation"].items()]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", nargs="+", required=True); ap.add_argument("--keys", nargs="+", required=True)
    ap.add_argument("--results", nargs="*", default=[])
    ap.add_argument("--pairs", default=os.path.join(ROOT, "eval", "test_pairs_dev.json"))
    ap.add_argument("--out", required=True); ap.add_argument("--md")
    a = ap.parse_args()
    pairs = {p["id"]: p for p in json.load(open(a.pairs))}
    res = build(load_ratings(a.ratings, a.keys), auto_rows(a.results), pairs)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    md = to_markdown(res)
    if a.md:
        open(a.md, "w").write(md)
    print(md); print("wrote", a.out)


if __name__ == "__main__":
    main()

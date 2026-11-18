"""Exact Shapley values + interaction indices for the 5-player prompt game (Member A guide S.4).
Run from the repo root:  python analysis/shap/shapley.py --scores results/shap@rrv_v1_dev.json
"""
import argparse, json, os, re, sys
from collections import defaultdict
from itertools import combinations
from math import factorial
from statistics import fmean

ORDER = ["char", "action", "setting", "mood", "tail"]
PAIRS = [("char", "setting"), ("tail", "mood")]
FAITH_PREF = ("checklist", "clip_raw")          # judge first; CLIP until the judge lands
_NAME = re.compile(r"^(?P<scene>[^_]+)_(?P<coal>.+)_s(?P<seed>\d+)\.png$")

def shapley(v, players):
    """v: dict frozenset -> value (mean over seeds). Exact; raises if any coalition is missing."""
    n = len(players)
    if len(v) != 2 ** n:
        raise ValueError(f"need all {2 ** n} coalitions, got {len(v)}")
    phi = {}
    for i in players:
        rest, s = [p for p in players if p != i], 0.0
        for k in range(n):
            w = factorial(k) * factorial(n - k - 1) / factorial(n)
            for S in combinations(rest, k):
                S = frozenset(S)
                s += w * (v[S | {i}] - v[S])
        phi[i] = s
    return phi

def interaction(v, players, i, j):
    """Shapley interaction index for (i, j): >0 complements, <0 substitutes."""
    n, rest, s = len(players), [p for p in players if p not in (i, j)], 0.0
    for k in range(n - 1):
        w = factorial(k) * factorial(n - k - 2) / factorial(n - 1)
        for S in combinations(rest, k):
            S = frozenset(S)
            s += w * (v[S | {i, j}] - v[S | {i}] - v[S | {j}] + v[S])
    return s

def check_efficiency(v, phi, tol=1e-9):
    full, empty = frozenset(ORDER), frozenset()
    gap = abs(sum(phi.values()) - (v[full] - v[empty]))
    if gap > tol:
        raise AssertionError(f"efficiency check failed: gap {gap:.3e}")

def parse_coalition(c):
    return frozenset() if c == "empty" else frozenset(c.split("+"))

def records_from(data):
    recs = data if isinstance(data, list) else next((data[k] for k in ("items", "images", "results", "scores") if k in data), None)
    if recs is None:
        raise ValueError("cannot find per-image records in the scores file")
    out = []
    for r in recs:
        r = dict(r)
        if not all(k in r for k in ("scene_id", "coalition", "seed")):      # fall back to the filename
            m = _NAME.match(os.path.basename(r.get("path", "")))
            if not m:
                raise ValueError(f"record lacks scene_id/coalition/seed and its path is unparsable: {r}")
            r.setdefault("scene_id", m["scene"]); r.setdefault("coalition", m["coal"]); r.setdefault("seed", int(m["seed"]))
        out.append(r)
    return out

def build_values(records, metric):
    """-> {scene_id: {frozenset: mean over seeds}}; raises on missing scores or uneven seeds."""
    acc, seeds = defaultdict(list), defaultdict(set)
    for r in records:
        val = r.get(metric)
        if val is None or val != val:
            raise ValueError(f"missing {metric} for {r['scene_id']} {r['coalition']} s{r['seed']}")
        key = (r["scene_id"], parse_coalition(r["coalition"]))
        acc[key].append(float(val)); seeds[key].add(r["seed"])
    if len({tuple(sorted(s)) for s in seeds.values()}) != 1:
        raise ValueError("coalitions were generated with different seed sets (common random numbers violated)")
    out = defaultdict(dict)
    for (sid, S), vals in acc.items():
        out[sid][S] = fmean(vals)
    return out

def per_scene(records, metric):
    phi, inter = {}, {}
    for sid, v in build_values(records, metric).items():
        p = shapley(v, ORDER)
        check_efficiency(v, p)
        phi[sid] = p
        inter[sid] = {f"{i}x{j}": interaction(v, ORDER, i, j) for i, j in PAIRS}
    return phi, inter

def mean_ci(x, n_boot=5000, seed=0):
    import numpy as np
    x = np.asarray(x, float)
    idx = np.random.default_rng(seed).integers(0, len(x), size=(n_boot, len(x)))   # resample scenes
    m = x[idx].mean(axis=1)
    return {"mean": float(x.mean()), "lo": float(np.percentile(m, 2.5)), "hi": float(np.percentile(m, 97.5))}

def by_tier(phi, dev_path):
    if not os.path.exists(dev_path):
        return {}
    tier = {s["id"]: s.get("tier") for s in json.load(open(dev_path, encoding="utf-8"))}
    groups = defaultdict(list)
    for sid, p in phi.items():
        if tier.get(sid) is not None:
            groups[tier[sid]].append(p)
    return {str(t): {"n_scenes": len(ps), **{pl: fmean(p[pl] for p in ps) for pl in ORDER}}
            for t, ps in sorted(groups.items())}

def _n_tokens(text):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from common.tokens import n_tokens   # CLIP tokenizer (downloads from the Hub)
    return n_tokens(text)

def value_per_token(mean_phi, scenes_path, scene_ids):
    scenes = json.load(open(scenes_path, encoding="utf-8"))
    scenes = scenes if isinstance(scenes, list) else list(scenes.values())
    toks = defaultdict(list)
    for sc in scenes:
        if sc["scene_id"] in scene_ids:
            for p in ORDER:
                toks[p].append(sc["player_tokens"][p] if "player_tokens" in sc else _n_tokens(sc["players"][p]))
    rows = {p: {"mean_tokens": fmean(toks[p]), "faithfulness_per_token": mean_phi[p] / fmean(toks[p])} for p in ORDER}
    return dict(sorted(rows.items(), key=lambda kv: -kv[1]["faithfulness_per_token"]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/shap@rrv_v1_dev.json")
    ap.add_argument("--scenes", default="analysis/shap/shap_scenes.json")
    ap.add_argument("--dev", default="eval/test_scenes_dev.json")
    ap.add_argument("--out", default="analysis/shap/shap_values_dev.json")
    a = ap.parse_args()

    records = records_from(json.load(open(a.scores, encoding="utf-8")))
    metrics = [m for m in ("checklist", "style_margin", "clip_raw") if all(m in r for r in records)]
    faith = next((m for m in FAITH_PREF if m in metrics), None)
    if faith is None or "style_margin" not in metrics:
        sys.exit(f"scores file needs a faithfulness metric {FAITH_PREF} and style_margin; found {metrics}")
    print("metrics:", metrics, "| faithfulness proxy:", faith)

    result = {"scores_file": a.scores, "faithfulness_metric": faith, "metrics": {}}
    for m in metrics:
        phi, inter = per_scene(records, m)
        result["n_scenes"] = len(phi)
        result["metrics"][m] = {
            "phi": {p: mean_ci([phi[s][p] for s in phi]) for p in ORDER},
            "interactions": {k: mean_ci([inter[s][k] for s in inter]) for k in next(iter(inter.values()))},
            "by_tier": by_tier(phi, a.dev),
            "per_scene": phi,
        }
        print(f"\n{m}  (n_scenes={len(phi)}, efficiency check passed)")
        for p in ORDER:
            s = result["metrics"][m]["phi"][p]
            print(f"  {p:8s} {s['mean']:+.4f}  [{s['lo']:+.4f}, {s['hi']:+.4f}]")
    mean_faith = {p: result["metrics"][faith]["phi"][p]["mean"] for p in ORDER}
    result["value_per_token"] = value_per_token(mean_faith, a.scenes, set(result["metrics"][faith]["per_scene"]))
    json.dump(result, open(a.out, "w", encoding="utf-8"), indent=2)
    print("\nvalue per token (send to D):", json.dumps(result["value_per_token"], indent=1))

if __name__ == "__main__":
    main()
"""H-C2: paired comparison of c3 (kb.json) vs c6 (kb_llm.json) with B's eval/stats.py. Difference is c3 - c6
(positive supports the hypothesis). Seeds averaged within scene first. Writes results/c6_comparison.json."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval"))
from stats import scene_means, paired

TIE = 0.0295
def load(name):
    return json.load(open(os.path.join(ROOT, "results", name)))
c3, c6 = load("c3_llm_kb_ckb@rrv_v1_dev.json"), load("c6_llm_kbllm@rrv_v1_dev.json")
print("row keys:", sorted(c3["items"][0].keys()))

def numeric_rows(res):
    out = []
    for r in res["items"]:
        out.append({k: v for k, v in r.items() if k == "scene_id" or isinstance(v, (int, float)) and not isinstance(v, bool)})
    return out
R3, R6 = numeric_rows(c3), numeric_rows(c6)
keys = [k for k in R3[0] if k not in ("scene_id", "seed")]

kb = json.load(open(os.path.join(ROOT, "kb", "kb.json"), encoding="utf-8"))
cov = {}
for c in kb["characters"].values():
    cov[c["canonical"]] = c["coverage"]
scenes = {s["id"]: s for s in json.load(open(os.path.join(ROOT, "eval", "test_scenes_dev.json")))}
def strong(sid):
    ch = scenes[sid]["expected_characters"]
    return bool(ch) and all(cov.get(n, "none") in ("high", "medium") for n in ch)
S = {sid for sid in scenes if strong(sid)}

out = {"definition": "difference = c3 - c6 (positive = image-grounded KB better)", "tie_band": TIE,
       "n_scenes_strong_coverage": len(S), "metrics": {}}
print(f"\nscenes with all expected characters at medium/high coverage: {len(S)} of {len(scenes)}\n")
print(f"{'metric':16s} {'c3':>7s} {'c6':>7s} {'diff':>8s}  95% CI              W/T/L   | strong-coverage scenes: diff [CI] n")
for k in keys:
    a, b = scene_means(R6, k), scene_means(R3, k)
    p = paired(a, b, tie=TIE)
    ps = paired({s: v for s, v in a.items() if s in S}, {s: v for s, v in b.items() if s in S},
                tie=TIE)
    m3 = sum(b.values()) / max(len(b), 1); m6 = sum(a.values()) / max(len(a), 1)
    out["metrics"][k] = {"c3_mean": m3, "c6_mean": m6, "all": p, "strong_coverage": ps}
    print(f"{k:16s} {m3:7.3f} {m6:7.3f} {p['diff']:+8.3f}  [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]  "
          f"{p['win']}/{p['tie']}/{p['loss']}  | {ps['diff']:+.3f} [{ps['ci95'][0]:+.3f}, {ps['ci95'][1]:+.3f}] n={ps['n']}")
json.dump(out, open(os.path.join(ROOT, "results", "c6_comparison.json"), "w"), indent=1)
print("\nwrote results/c6_comparison.json")

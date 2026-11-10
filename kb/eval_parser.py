"""Gold evaluation of kb/parser.py on eval/test_scenes_dev.json.
Character precision / recall (micro, over all scenes) and epic accuracy. Writes kb/parser_metrics.json."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from kb.parser import parse_scene

kb = json.load(open(os.path.join(ROOT, "kb", "kb.json"), encoding="utf-8"))
canon = {}
for c in kb["characters"].values():
    for a in [c["canonical"], *c["aliases"]]:
        canon[a.lower()] = c["canonical"].lower()
norm = lambda n: canon.get(n.strip().lower(), n.strip().lower())

scenes = json.load(open(os.path.join(ROOT, "eval", "test_scenes_dev.json"), encoding="utf-8"))
tp = fp = fn = 0
epic_ok = epic_n = 0
rows = []
for s in scenes:
    spec = parse_scene(s["text"], s["id"])
    got = {norm(c["canonical"]) for c in spec["characters"]}
    exp = {norm(n) for n in s.get("expected_characters", [])}
    tp += len(got & exp); fp += len(got - exp); fn += len(exp - got)
    if s.get("expected_epic"):
        epic_n += 1; epic_ok += spec["epic"] == s["expected_epic"]
    rows.append({"id": s["id"], "missed": sorted(exp - got), "extra": sorted(got - exp),
                 "epic_expected": s.get("expected_epic"), "epic_got": spec["epic"]})

P = tp / (tp + fp) if tp + fp else 1.0
R = tp / (tp + fn) if tp + fn else 1.0
E = epic_ok / epic_n if epic_n else 1.0
out = {"n_scenes": len(scenes), "precision": round(P, 3), "recall": round(R, 3),
       "epic_accuracy": round(E, 3), "kb_version": kb["version"], "per_scene": rows}
json.dump(out, open(os.path.join(ROOT, "kb", "parser_metrics.json"), "w"), indent=1)
print(f"{len(scenes)} scenes | precision {P:.2f} | recall {R:.2f} | epic accuracy {E:.2f}")
print("\nscenes with problems:")
for r in rows:
    if r["missed"] or r["extra"] or r["epic_expected"] != r["epic_got"]:
        print(f"  {r['id']}: missed={r['missed']} extra={r['extra']} epic {r['epic_expected']} -> {r['epic_got']}")

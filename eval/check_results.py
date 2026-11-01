"""Sanity-check a results file from score.py (Member B). Run it on every results/*.json someone commits.

python eval/check_results.py results/c1_lora_raw@rrv_v1_dev.json [--scenes eval/test_scenes_dev.json] [--seeds 1 2]
Exit code 1 if something is missing, so it can be used in CI later.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check(path, scenes, seeds):
    r = json.load(open(path))
    rows = r.get("items", [])
    problems = []
    want = {(s, k) for s in scenes for k in seeds}
    have = {(x["scene_id"], x.get("seed")) for x in rows}
    missing = sorted(want - have)
    if missing:
        problems.append(f"{len(missing)} (scene, seed) pairs missing, e.g. {missing[:5]}")
    if not r.get("judge"):
        problems.append("no judge (scored with --no_judge): fine for a smoke test, not for results")
    for k in ("checklist", "clip_raw", "style_margin", "max_train_dino"):
        n = sum(1 for x in rows if x.get(k) is not None)
        if n < len(rows):
            problems.append(f"{k}: present on {n}/{len(rows)} images")
    conds = sorted({x.get("condition") for x in rows})
    if len(conds) != 1:
        problems.append(f"expected one condition per file, found {conds}")
    return conds, len(rows), problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--scenes", default=os.path.join(ROOT, "eval", "test_scenes_dev.json"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    a = ap.parse_args()
    scenes = [s["id"] for s in json.load(open(a.scenes))]
    bad = 0
    for p in a.results:
        conds, n, problems = check(p, scenes, a.seeds)
        print(f"{'OK ' if not problems else 'FIX'} {os.path.basename(p)}  {conds}  {n} images")
        for x in problems:
            print("     -", x)
        bad += bool(problems)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

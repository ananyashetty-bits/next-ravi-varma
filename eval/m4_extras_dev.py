"""Two dev numbers for EVAL_REPORT_M4.md that eval/tables_m4.py does not print (B, 24 Nov 2026).

python eval/m4_extras_dev.py --ratings <sheet1.csv> <sheet2.csv> --keys <key1.json> <key2.json> \
       --out results/m4_extras_dev.json

1. Figure count (H-B1, H-C2): per condition, how many of the 16 seed-1 images have n_people_out ==
   n_people_expected, and paired counts (only b right / only a right) for all tiers and tiers 2-4.
2. pose_pck vs human pose match on subsets (post-hoc, labelled as such). The pre-registered number is the
   all-conditions Spearman in results/m4_tables_dev.json; p3's pose_pck is scored against its own auto skeleton,
   not the pair pose the raters compared against, so it is also reported without p3.
The blind keys are read from B's private Drive copy and never written to the output.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from eval import tables_m4 as T  # noqa: E402

CONDS = ["p0_m3_default", "p1_openpose", "p2_openpose_tuned", "p3_pose_auto", "p5_hires", "p6_restore"]
PAIRS_FC = [("p0_m3_default", "p1_openpose"), ("p0_m3_default", "p3_pose_auto"), ("p1_openpose", "p3_pose_auto")]
SUBSETS = {"all six conditions (pre-registered)": CONDS,
           "without p3 (post-hoc)": [c for c in CONDS if c != "p3_pose_auto"],
           "p0 and p1 only (post-hoc)": ["p0_m3_default", "p1_openpose"],
           "controlled photo-pose conditions p1, p2, p5, p6 (post-hoc)": ["p1_openpose", "p2_openpose_tuned",
                                                                          "p5_hires", "p6_restore"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", nargs="+", required=True)
    ap.add_argument("--keys", nargs="+", required=True)
    ap.add_argument("--pairs", default=os.path.join(ROOT, "eval", "test_pairs_dev.json"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f"{a.out} exists; not overwriting")
    pairs = {p["scene_id"]: p for p in json.load(open(a.pairs))}
    autos = T.auto_rows([os.path.join(ROOT, "results", f"{c}@rrv_v1_dev_pose.json") for c in CONDS])
    seed1 = {(r["pair_id"], r["condition"]): r for r in autos if r["seed"] == 1}

    ok = {c: {p: bool(seed1[(p, c)]["figure_count_ok"]) for p in pairs} for c in CONDS}
    fc = {"per_condition_seed1": {c: f"{sum(ok[c].values())}/16" for c in CONDS}, "paired": {}}
    for x, y in PAIRS_FC:
        for label, tiers in (("all tiers", None), ("tiers 2-4", (2, 3, 4))):
            ids = [p for p in pairs if tiers is None or pairs[p]["tier"] in tiers]
            fc["paired"][f"{x} -> {y}, {label}"] = {
                "n_pairs": len(ids), "a_right": sum(ok[x][p] for p in ids), "b_right": sum(ok[y][p] for p in ids),
                "only_b_right": sum(ok[y][p] and not ok[x][p] for p in ids),
                "only_a_right": sum(ok[x][p] and not ok[y][p] for p in ids)}

    ratings = T.load_ratings(a.ratings, a.keys)
    val = {}
    for label, conds in SUBSETS.items():
        xs, ys = [], []
        for c in conds:
            for p, v in T.pair_means(ratings, "pose_match", c).items():
                r = seed1.get((p, c))
                if r and r["pose_pck"] is not None:
                    xs.append(r["pose_pck"]); ys.append(v)
        val[label] = {"n_images": len(xs), "spearman": T.spearman(xs, ys)}

    out = {"note": __doc__.strip().splitlines()[0], "figure_count": fc, "pose_pck_vs_human_pose_match": val}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

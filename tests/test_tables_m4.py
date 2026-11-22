import csv, json, os, random, sys
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
import pytest
pytest.importorskip("numpy")
from eval.tables_m4 import load_ratings, build, to_markdown, score


def test_score_parsing():
    assert score("4") == 4 and score(" 5 ") == 5 and score("NV") is None and score("") is None and score("7") is None


def make(tmp_path):
    rng = random.Random(0)
    pairs = {f"s{i:02d}": {"id": f"s{i:02d}", "tier": 1 + i % 3, "n_people_expected": 1 + i % 3} for i in range(1, 9)}
    key, rows, autos = {}, [], []
    for pid in pairs:
        key[pid] = {"AA": {"condition": "p0_m3_default", "image": "x"}, "BB": {"condition": "p1_openpose", "image": "y"}}
        for code, bump in (("AA", 0), ("BB", 1)):
            for rater in ("A", "B"):
                rows.append({"pair_id": pid, "code": code, "rater": rater, "calibration": "",
                             "style_1to5": 4, "pose_match_1to5": min(5, 2 + bump + rng.randint(0, 1)),
                             "faces_1to5": 3, "hands_1to5_or_NV": "NV" if rater == "B" else 2,
                             "failure": "pose ignored" if bump == 0 else "", "notes": ""})
            for seed in (1, 2):
                autos.append({"scene_id": pid, "condition": key[pid][code]["condition"], "seed": seed,
                              "pose_pck": 0.3 + 0.4 * bump, "style_margin": 0.1, "checklist": 0.5,
                              "n_people_out": pairs[pid]["n_people_expected"] if bump else 0,
                              "n_people_expected": pairs[pid]["n_people_expected"]})
    kp = tmp_path / "key.json"; kp.write_text(json.dumps(key))
    cp = tmp_path / "r.csv"
    with open(cp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    rp = tmp_path / "res.json"; rp.write_text(json.dumps({"items": autos}))
    return pairs, str(cp), str(kp), str(rp)


def test_build_end_to_end(tmp_path):
    from eval.tables_m4 import auto_rows
    pairs, cp, kp, rp = make(tmp_path)
    res = build(load_ratings([cp], [kp]), auto_rows([rp]), pairs)
    c = res["comparisons"]["p0_m3_default -> p1_openpose"]
    assert c["pose_match"]["diff"] > 0.5 and c["faces"]["diff"] == 0 and c["faces"]["tie"] == 8
    assert c["pose_pck"]["diff"] == pytest.approx(0.4) and c["figure_count_ok"]["diff"] == pytest.approx(1.0)
    assert res["means"]["hands"]["p0_m3_default"]["mean"] == 2               # NV ignored, not scored
    assert res["failures"]["p0_m3_default"]["pose ignored"] == 16
    assert set(res["by_people"]) == {"1", "2", "3"}
    assert "p0_m3_default -> p1_openpose" in to_markdown(res)

"""Tests for eval/ratings.py (owner: B). Synthetic sheet + key; no real ratings are used."""
import csv
import json

import pytest

from eval import ratings as R

HEAD = ["scene_id", "code", "rater", "faithfulness_1to5", "style_1to5", "quality_1to5", "failure", "notes"]


def write_round(tmp_path, rows, key, name="r1"):
    c, k = tmp_path / f"{name}.csv", tmp_path / f"{name}_key.json"
    with open(c, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEAD)
        w.writerows(rows)
    k.write_text(json.dumps(key))
    return str(c), str(k)


KEY = {s: {f"{s}x": {"condition": "c1_lora_raw", "model": "rrv_v1", "image": f"/img/c1/{s}_s1.png"},
           f"{s}y": {"condition": "c4_llm_nokb_3", "model": "rrv_v1", "image": f"/img/c4/{s}_s1.png"}}
       for s in ("s01", "s02", "s03", "s04", "s05")}


def test_tags_normalised():
    assert R.norm_tag("generic figure") == ("generic figure (name ignored)", False)
    assert R.norm_tag("Missing figure identity ") == ("wrong/missing figure identity", False)
    assert R.norm_tag("wrong setting") == ("wrong setting", True)
    assert R.norm_tag("blurry") == ("other", False)
    assert R.norm_tag("") == ("", False)


def test_load_round_decodes_and_reports(tmp_path):
    rows = [["s01", "s01x", "b", 4, 3, 2, "weak style", ""],
            ["s01", "s01y", "B", 2, 2, 2, "generic figure", ""],
            ["s01", "s01", "D", 3, 3, 3, "", ""],             # typo'd code, like the real s15 row
            ["s02", "s02x", "C", "", "", "", "", ""],          # not rated yet
            ["s02", "s02y", "B", 6, 3, 3, "", ""],             # out of range
            ["s03", "s03x", "B", 3, "", 3, "", ""]]            # partly filled
    c, k = write_round(tmp_path, rows, KEY)
    got, problems, pending, total = R.load_round(c, k, 1)
    assert [(g["scene_id"], g["code"], g["rater"], g["condition"]) for g in got] == [
        ("s01", "s01x", "B", "c1_lora_raw"), ("s01", "s01y", "B", "c4_llm_nokb")]
    assert got[1]["failure"] == "generic figure (name ignored)"
    assert pending == {"C": 1}
    assert total == {"B": 4, "D": 1, "C": 1}
    text = "\n".join(problems)
    assert "'s01' is not in the blind key" in text
    assert "outside 1-5" in text
    assert "some scores empty" in text
    assert "counted as 'generic figure (name ignored)'" in text


def test_rank_correlations():
    assert R.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert R.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert R.ranks([3, 1, 3, 2]) == [3.5, 1.0, 3.5, 2.0]
    assert R.spearman([1, 1, 1], [1, 2, 3]) != R.spearman([1, 1, 1], [1, 2, 3])   # constant -> nan
    assert R.weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)
    assert R.weighted_kappa([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) < 0


def test_resolve_aliases():
    conds = ["c1_lora_raw", "c3_llm_kb", "c3_llm_kb_ckb"]
    assert R.resolve("c1", conds, {}) == "c1_lora_raw"
    assert R.resolve("c5", conds, {}) is None
    assert R.resolve("c3", conds, {"c3": "c3_llm_kb_ckb"}) == "c3_llm_kb_ckb"
    with pytest.raises(SystemExit):
        R.resolve("c3", conds, {})


def test_scene_coverage(tmp_path):
    kb = {"characters": {"sita": {"canonical": "Sita", "aliases": ["Seeta"], "coverage": "high"},
                         "karna": {"canonical": "Karna", "aliases": [], "coverage": "low"}}}
    p = tmp_path / "kb.json"
    p.write_text(json.dumps(kb))
    scenes = {"a": {"id": "a", "expected_characters": ["seeta"]},
              "b": {"id": "b", "expected_characters": ["Sita", "Karna"]},
              "c": {"id": "c", "expected_characters": ["Nobody"]},
              "d": {"id": "d", "expected_characters": []}}
    assert R.scene_coverage(scenes, str(p)) == {"a": "high", "b": "low", "c": "none", "d": "n/a"}


def test_analyse_end_to_end(tmp_path):
    pytest.importorskip("numpy")                       # eval/stats.py needs numpy (CI installs it only if added)
    # c4 is rated exactly 1 point higher than c1 on faithfulness in every scene, by both raters
    rows = []
    for i, s in enumerate(("s01", "s02", "s03", "s04", "s05")):
        for r in ("B", "D"):
            rows.append([s, f"{s}x", r, 1 + i % 3, 3, 3, "wrong action", ""])
            rows.append([s, f"{s}y", r, 2 + i % 3, 4, 3, "", ""])
    c, k = write_round(tmp_path, rows, KEY)
    got, problems, _, _ = R.load_round(c, k, 1)
    assert not problems
    scenes = {s: {"id": s, "tier": 1 + j % 2, "seen_in_training": j == 0} for j, s in enumerate(KEY)}
    # judge tracks faithfulness perfectly; clip is noise
    results = {}
    for i, s in enumerate(KEY):
        results[("c1_lora_raw", s, 1)] = {"checklist": 0.1 * (1 + i % 3) + 0.01 * i, "clip_raw": 0.3, "style_margin": 0.1}
        results[("c4_llm_nokb", s, 1)] = {"checklist": 0.1 * (2 + i % 3) + 0.01 * i, "clip_raw": 0.3, "style_margin": 0.2}
    rep = R.analyse(got, scenes, results, n_boot=200)

    assert rep["conditions"] == ["c1_lora_raw", "c4_llm_nokb"]
    (p,) = [x for x in rep["paired"] if (x["a"], x["b"]) == ("c1_lora_raw", "c4_llm_nokb")]
    assert p["faithfulness"]["diff"] == pytest.approx(1.0)
    assert (p["faithfulness"]["win"], p["faithfulness"]["loss"], p["n"]) == (5, 0, 5)
    assert rep["agreement"]["n_images"] == 10
    assert rep["agreement"]["faithfulness"]["spearman"] == pytest.approx(1.0)
    assert rep["agreement"]["H-B5_pass"] is True
    assert rep["failures"]["c1_lora_raw"]["wrong action"] == 10
    jv = rep["judge_validation"]
    assert jv["n_images"] == 10 and jv["unmatched"] == 0
    assert jv["judge_vs_faith"]["spearman"] > 0.9
    assert jv["H-B1_pass"] is True
    assert R.verdict(None) == "not enough data" and R.verdict(False) == "not met"
    assert set(rep["by_seen"]["c1_lora_raw"]) == {"seen", "unseen"}
    md = R.to_markdown(rep)
    assert "`c1_lora_raw` → `c4_llm_nokb`" in md and "H-B1" in md


def test_judge_robustness_sample_is_balanced_and_fixed():
    from eval.judge_robustness import pick
    import glob
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = sorted(glob.glob(os.path.join(root, "results", "c[014]_*_dev.json")))
    if len(paths) < 3:
        pytest.skip("c0/c1/c4 results not present")
    a, b = pick(paths, 20), pick(paths, 20)
    assert [x["image"] for x in a] == [x["image"] for x in b]
    assert len({x["image"] for x in a}) == 20
    counts = {}
    for x in a:
        counts[x["condition"]] = counts.get(x["condition"], 0) + 1
    assert max(counts.values()) - min(counts.values()) <= 1

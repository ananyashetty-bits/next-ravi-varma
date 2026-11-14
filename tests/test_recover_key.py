"""eval/recover_key.py rebuilds exactly the key contact_sheet.py wrote (owner: B)."""
import json
import os
import random
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_round(tmp_path):
    from PIL import Image
    rng = random.Random(1)
    scenes = [{"id": f"s{i:02d}", "text": f"scene {i}"} for i in range(1, 5)]
    (tmp_path / "scenes.json").write_text(json.dumps(scenes))
    manifests = []
    for cond, model in (("c0_base_raw", "base"), ("c1_lora_raw", "rrv_v1"), ("c4_llm_nokb", "rrv_v1")):
        d = tmp_path / cond
        d.mkdir()
        items = []
        for s in scenes:
            for seed in (1, 2):
                p = d / f"{s['id']}_s{seed}.png"
                Image.frombytes("RGB", (64, 64), bytes(rng.randrange(256) for _ in range(64 * 64 * 3))).save(p)
                items.append({"scene_id": s["id"], "condition": cond, "seed": seed, "path": str(p)})
        m = d / "manifest.json"
        m.write_text(json.dumps({"model": model, "items": items}))
        manifests.append(str(m))
    return manifests


def test_recovers_original_key(tmp_path):
    pytest.importorskip("PIL")
    manifests = make_round(tmp_path)
    out, key = tmp_path / "sheets", tmp_path / "private" / "key.json"
    subprocess.run([sys.executable, os.path.join(ROOT, "eval", "contact_sheet.py"), "--manifests", *manifests,
                    "--out", str(out), "--key", str(key), "--scenes", str(tmp_path / "scenes.json")],
                   check=True, capture_output=True)
    from eval.recover_key import recover
    # manifests passed in a different order on purpose: matching is by image, not by order
    got, report, failures = recover(str(out), str(out / "rating_template.csv"), manifests[::-1])
    assert not failures, failures
    assert got == json.load(open(key))


def test_refuses_when_an_image_is_missing(tmp_path):
    pytest.importorskip("PIL")
    manifests = make_round(tmp_path)
    out, key = tmp_path / "sheets", tmp_path / "private" / "key.json"
    subprocess.run([sys.executable, os.path.join(ROOT, "eval", "contact_sheet.py"), "--manifests", *manifests,
                    "--out", str(out), "--key", str(key), "--scenes", str(tmp_path / "scenes.json")],
                   check=True, capture_output=True)
    from eval.recover_key import recover
    _, _, failures = recover(str(out), str(out / "rating_template.csv"), manifests[:2])   # c4 not offered
    assert failures and all("no clear match" in f for f in failures)


def test_load_items_finds_images_next_to_a_moved_manifest(tmp_path):
    from eval.score import load_items
    (tmp_path / "s01_s1.png").write_bytes(b"x")
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps({"items": [{"scene_id": "s01", "seed": 1, "condition": "c2_template",
                                        "path": "/content/drive/MyDrive/elsewhere/s01_s1.png"}]}))
    _, items = load_items(str(m))
    assert items[0]["path"] == str(tmp_path / "s01_s1.png")

import json, os, sys
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from eval.contact_sheet_m4 import assign, rated_copy, main
import pytest


def test_calibration_rows_go_to_everyone_rest_rotate():
    order = [("s01", f"C{i}") for i in range(9)]
    rows = assign(order, ["A", "B", "C", "D"], per_image=2, calibration=5)
    per_img = {}
    for r in rows:
        per_img.setdefault(r["code"], []).append(r["rater"])
    assert all(len(per_img[f"C{i}"]) == 4 for i in range(5))
    assert all(len(set(per_img[f"C{i}"])) == 2 for i in range(5, 9))
    assert all(r["calibration"] == "yes" for r in rows if r["code"] in {f"C{i}" for i in range(5)})


def test_rated_copy_prefers_512(tmp_path):
    big = tmp_path / "s01_s1.png"; big.write_bytes(b"x")
    assert rated_copy(str(big)) == str(big)
    (tmp_path / "s01_s1_512.png").write_bytes(b"x")
    assert rated_copy(str(big)) == str(tmp_path / "s01_s1_512.png")


def test_end_to_end_and_key_privacy(tmp_path, monkeypatch):
    pytest.importorskip("PIL")
    from PIL import Image
    pairs = json.load(open(os.path.join(ROOT, "eval", "test_pairs_dev.json")))[:2]
    pf = tmp_path / "pairs.json"; pf.write_text(json.dumps(pairs))
    mans = []
    for cond in ["p0_m3_default", "p1_openpose"]:
        d = tmp_path / cond; d.mkdir()
        imgs = []
        for p in pairs:
            ip = d / f"{p['id']}_s1.png"; Image.new("RGB", (512, 512), (90, 60, 30)).save(ip)
            imgs.append({"scene_id": p["scene_id"], "condition": cond, "seed": 1, "path": str(ip)})
        (d / "manifest.json").write_text(json.dumps({"condition": cond, "images": imgs})); mans.append(str(d / "manifest.json"))
    out, key = tmp_path / "sheets", tmp_path / "private" / "key.json"
    monkeypatch.setattr(sys, "argv", ["x", "--manifests", *mans, "--pairs", str(pf), "--out", str(out), "--key", str(key)])
    main()
    k = json.load(open(key))
    assert set(k) == {p["id"] for p in pairs} and all(len(v) == 2 for v in k.values())
    assert (out / f"{pairs[0]['id']}.png").exists() and (out / "rating_template.csv").exists()
    monkeypatch.setattr(sys, "argv", ["x", "--manifests", *mans, "--pairs", str(pf), "--out", str(out), "--key", str(key)])
    with pytest.raises(SystemExit):
        main()                                          # never overwrite a blind key
    monkeypatch.setattr(sys, "argv", ["x", "--manifests", *mans, "--pairs", str(pf), "--out", str(out),
                                      "--key", os.path.join(ROOT, "eval", "k.json")])
    with pytest.raises(SystemExit):
        main()                                          # never inside the repo

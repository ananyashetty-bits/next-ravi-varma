"""post/ (owner D): config-driven settings, 512 copies, resumable manifest. CPU only; no GPU, torch or gfpgan needed."""
import json
import os
import sys

import pytest
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from post import hires as H  # noqa: E402
from post import restore as R  # noqa: E402
from post import run_post as RP  # noqa: E402


def test_modules_import_without_gpu_libraries():
    # lazy imports: importing post.hires / post.restore / post.run_post above must not need torch or gfpgan
    import post.api as api
    assert callable(api.hires) and callable(api.restore_faces)


def test_hires_settings_come_from_config():
    s = H.hires_settings()
    m4 = json.load(open(os.path.join(ROOT, "config", "m4.json")))["post"]["hires"]
    assert s == {"method": m4["method"], "size": int(m4["size"]), "strength": float(m4["strength"])}


def test_lanczos_up_size_and_mode():
    out = H.lanczos_up(Image.new("L", (512, 512), 128), 768)
    assert out.size == (768, 768) and out.mode == "RGB"


def test_unknown_methods_rejected():
    with pytest.raises(ValueError):
        H.hires(Image.new("RGB", (8, 8)), "p", "n", 1, method="nope")
    with pytest.raises(ValueError):
        R.restore_faces_info(Image.new("RGB", (8, 8)), "nope")


def test_restore_none_is_passthrough():
    img = Image.new("RGB", (8, 8))
    assert R.restore_faces_info(img, "none") == (img, 0)


def test_no_prompt_text_is_hardcoded():
    for f in ("hires.py", "restore.py", "run_post.py"):
        src = open(os.path.join(ROOT, "post", f), encoding="utf-8").read()
        assert "rrvarma style" not in src and "negative_prompt=\"" not in src


def _src_manifest(tmp_path):
    d = tmp_path / "src"
    d.mkdir()
    items = []
    for sid, seed in (("s01", 1), ("s01", 2), ("s02", 1)):
        p = d / f"{sid}_s{seed}.png"
        Image.new("RGB", (512, 512), (10 * seed, 20, 30)).save(p)
        items.append({"scene_id": sid, "seed": seed, "path": str(p), "prompt": f"rrvarma style, {sid}",
                      "negative": "NEG", "condition": "p1_openpose", "lora_scale": 1.0, "model": "rrv_v1"})
    m = d / "manifest.json"
    m.write_text(json.dumps({"condition": "p1_openpose", "images": items}))
    return str(m)


def test_run_writes_full_and_512_copies_and_manifest(tmp_path):
    calls = []

    def fake(img, it):
        calls.append((it["prompt"], it["negative"], it["seed"]))
        return img.resize((768, 768), Image.LANCZOS), {"n_faces_restored": 1}

    out = tmp_path / "out"
    man = RP.run(_src_manifest(tmp_path), "p5_hires", str(out), fake, {"method": "hires_img2img", "strength": 0.35})
    assert len(man["images"]) == 3 and man["condition"] == "p5_hires"
    for r in man["images"]:
        assert Image.open(r["path"]).size == (768, 768)            # full size kept for scoring
        assert Image.open(r["path_512"]).size == (512, 512)        # what raters see
        assert r["path_512"] == r["path"][:-4] + "_512.png"        # B's contact sheet looks for exactly this name
        assert r["negative"] == "NEG" and r["lora_scale"] == 1.0   # inherited from the source manifest
        assert r["condition"] == "p5_hires" and len(r["source_sha256"]) == 64
    assert ("rrvarma style, s01", "NEG", 1) in calls
    assert json.load(open(out / "manifest.json"))["post"]["strength"] == 0.35


def test_run_is_resumable(tmp_path):
    n = []

    def fake(img, it):
        n.append(1)
        return img, {}

    src, out = _src_manifest(tmp_path), str(tmp_path / "out")
    RP.run(src, "p6_restore", out, fake, {"method": "restore_gfpgan"})
    RP.run(src, "p6_restore", out, fake, {"method": "restore_gfpgan"})
    assert len(n) == 3                                              # second pass skipped everything


def test_manifest_loads_with_scores_loader(tmp_path):
    pytest.importorskip("numpy")
    out = str(tmp_path / "out")
    RP.run(_src_manifest(tmp_path), "p5_hires", out, lambda i, it: (i, {}), {"method": "hires_img2img"})
    m = json.load(open(os.path.join(out, "manifest.json")))
    assert all(os.path.exists(r["path"]) for r in m["images"]) and "images" in m

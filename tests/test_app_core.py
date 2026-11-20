"""App logic (app/core.py) and post stubs: config-driven defaults, input checks, label, metadata. CPU only."""

import json

from PIL import Image

from app import core
from post import stub as post_stub

M3 = {"default_model": "rrv_v1", "models": {"rrv_v1": {"scale": 1.0}}}


def test_defaults_come_from_config():
    m4 = {"lora_scale": None, "control": {"baseline": {"scale": 1.0}, "tuned": None},
          "ui": {"style_slider_range": [0.8, 1.0], "pose_slider_range": [0.4, 1.0]}, "post": {"default_on": False}}
    d = core.defaults(M3, m4)
    assert d["style"] == 1.0 and d["style_range"] == [0.8, 1.0]
    assert d["pose"] == 1.0 and d["pose_profile"] == "baseline" and d["post_default_on"] is False


def test_tuned_profile_and_clamping():
    m4 = {"lora_scale": 1.2, "control": {"baseline": {"scale": 1.0}, "tuned": {"scale": 0.7}},
          "ui": {"style_slider_range": [0.8, 1.0], "pose_slider_range": [0.4, 0.6]}}
    d = core.defaults(M3, m4)
    assert d["pose_profile"] == "tuned"
    assert d["pose"] == 0.6          # clamped to the tested range
    assert d["style"] == 1.0         # 1.2 is outside the tested slider range


def test_real_config_defaults():
    m3, m4 = core.load_configs()
    d = core.defaults(m3, m4)
    assert d["style_range"][0] <= d["style"] <= d["style_range"][-1]
    assert d["pose_range"][0] <= d["pose"] <= d["pose_range"][-1]


def test_check_text():
    assert not core.check_text("")["ok"]
    assert not core.check_text("   ")["ok"]
    assert not core.check_text("word " * 300)["ok"]
    assert core.check_text("word " * 60)["warning"]
    r = core.check_text("Sita in the Ashoka grove")
    assert r["ok"] and not r["warning"]


def test_label_and_metadata(tmp_path):
    img = Image.new("RGB", (512, 512), (120, 80, 40))
    out = core.add_label(img)
    assert out.width == 512 and out.height > 512
    assert out.getpixel((10, 10)) == (120, 80, 40)          # the painting itself is untouched
    s = core.settings_record("Sita", {"positive": "p", "negative": "n"}, 3, 1.0, "none", None, None, [], "rrv_v1")
    path = core.save_png(out, tmp_path / "x.png", s)
    info = Image.open(path).info
    assert info["Description"] == core.LABEL
    assert json.loads(info["parameters"])["seed"] == 3


def test_post_stubs_are_passthroughs():
    img = Image.new("RGB", (64, 64))
    assert post_stub.hires(img, "p", "n", 1) is img
    assert post_stub.restore_faces(img) is img

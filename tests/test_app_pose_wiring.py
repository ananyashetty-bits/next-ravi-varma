"""Pose wiring in the app: the active control profile (scale AND guidance window) reaches generate_controlled, an
import failure is reported instead of swallowed, and the settings JSON records the window. CPU only, no GPU."""

import json
import sys
import types

from PIL import Image

from app import app as A
from app import core

M3 = {"default_model": "rrv_v1", "models": {"rrv_v1": {"scale": 1.0}}}
UI = {"style_slider_range": [0.6, 1.2], "pose_slider_range": [0.4, 1.2]}


def test_window_baseline_tuned_and_missing_keys():
    base = {"scale": 1.0, "guidance_start": 0.0, "guidance_end": 1.0}
    d = core.defaults(M3, {"control": {"baseline": base, "tuned": None}, "ui": UI})
    assert d["pose_window"] == [0.0, 1.0] and d["pose_profile"] == "baseline"
    tuned = {"scale": 1.0, "guidance_start": 0.0, "guidance_end": 0.8}
    d = core.defaults(M3, {"control": {"baseline": base, "tuned": tuned}, "ui": UI})
    assert d["pose_window"] == [0.0, 0.8] and d["pose_profile"] == "tuned" and d["pose"] == 1.0
    d = core.defaults(M3, {"control": {"baseline": {"scale": 1.0}, "tuned": None}, "ui": UI})   # old-style config
    assert d["pose_window"] == [0.0, 1.0]


def _engine(control, gen):
    e = object.__new__(A.Engine)          # no model load, no GPU
    e.control, e.gen = control, gen
    return e


def test_engine_passes_scale_and_window_to_generate_controlled(monkeypatch):
    monkeypatch.setitem(A.DEFAULTS, "pose_window", [0.0, 0.8])
    seen = {}

    def generate_controlled(*a, **kw):
        seen.update(kw)
        return "IMG"

    ctrl_mod = types.SimpleNamespace(generate_controlled=generate_controlled)
    img, used = _engine(ctrl_mod, None).generate("pos", "neg", 3, 0.9, {"image": Image.new("RGB", (8, 8))}, 0.7)
    assert (img, used) == ("IMG", True)
    assert seen["lora_scale"] == 0.9 and seen["control_scale"] == 0.7
    assert seen["guidance_start"] == 0.0 and seen["guidance_end"] == 0.8


def test_engine_without_control_falls_back_to_plain_generate():
    gen = types.SimpleNamespace(generate=lambda *a, **kw: "PLAIN")
    img, used = _engine(None, gen).generate("pos", "neg", 3, 1.0, {"image": Image.new("RGB", (8, 8))}, 1.0)
    assert (img, used) == ("PLAIN", False)


def test_import_failure_is_recorded_not_swallowed(capsys):
    assert A._import("no_such_module_xyz") is None
    assert "ModuleNotFoundError" in A.IMPORT_ERRORS["no_such_module_xyz"]
    assert "no_such_module_xyz" in capsys.readouterr().err


def test_settings_record_has_window_only_when_pose_used():
    p = {"positive": "p", "negative": "n"}
    s = core.settings_record("t", p, 1, 1.0, "preset", 1.0, "preset:x", [], "rrv_v1", [0.0, 0.8])
    assert s["control_guidance"] == [0.0, 0.8]
    s = core.settings_record("t", p, 1, 1.0, "none", None, None, [], "rrv_v1")
    assert s["control_guidance"] is None and s["control_scale"] is None


class _PoseEngine(A.Engine):
    """Fake engine whose pose IS applied, to check what the handler records."""
    def __init__(self):
        self.control, self.pose, self.post = None, self, None

    def parse_scene(self, text):
        return {"characters": []}

    def expand(self, text, scene, mode):
        return {"positive": f"rrvarma style, {text}", "negative": "ugly"}

    def make_pose(self, name):
        return {"image": Image.new("RGB", (512, 512)), "n_people": 1, "source": f"preset:{name}", "warnings": []}

    def generate(self, positive, negative, seed, style, ctrl, pose_strength):
        return Image.new("RGB", (512, 512), (90, 60, 30)), ctrl is not None


def test_handler_records_window_and_no_not_switched_on_warning():
    out = A.make_handler(_PoseEngine())("Sita under a tree", "preset", "standing", None, 1.0, 0.8, 5, False, False)
    status, settings = out[5], json.loads(out[3])
    assert "not available" not in status and "not switched on" not in status
    assert settings["control_scale"] == 0.8 and settings["control_guidance"] == A.DEFAULTS["pose_window"]

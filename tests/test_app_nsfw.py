"""NSFW soft blur-and-confirm, the time note, and the UI wiring. CPU only; the classifier is faked."""

import json

import pytest
from PIL import Image

from app import app as A
from app import core, nsfw


def _img():
    im = Image.new("RGB", (256, 256))
    px = im.load()
    for x in range(256):
        for y in range(256):
            px[x, y] = ((x * 7) % 256, (y * 5) % 256, ((x + y) * 3) % 256)      # busy, so a blur visibly changes it
    return im


class _Eng(A.Engine):
    def __init__(self, score):
        self.control, self.pose, self.post, self._score = None, None, None, score

    def parse_scene(self, text):
        return {"characters": []}

    def expand(self, text, scene, mode):
        return {"positive": f"rrvarma style, {text}", "negative": "ugly"}

    def generate(self, positive, negative, seed, style, ctrl, pose_strength):
        return _img(), False

    def nsfw_score(self, image):
        return self._score


def _run(eng, tmp_path, monkeypatch, seed=11):
    monkeypatch.setattr(A, "OUT_DIR", tmp_path)
    return A.make_handler(eng)("Sita under a tree", "none", None, None, 1.0, 0.8, seed, False, False)


def test_flagged_image_is_blurred_with_clear_copy_and_reveal(tmp_path, monkeypatch):
    img, _, _, settings, path, status = _run(_Eng(0.9), tmp_path, monkeypatch)
    s = json.loads(settings)
    assert s["nsfw_flagged"] is True and "blurred" in status and "Show image" in status
    clear = tmp_path / "nrv_seed11_clear.png"
    assert clear.exists()
    shown = img.convert("RGB").crop((0, 0, 256, 256))
    assert list(shown.getdata()) != list(_img().getdata())              # what the user first sees is not the original
    back = A.reveal(settings)
    assert back is not None and back.size[1] > 256                      # clear copy (with the label bar) comes back


def test_unflagged_and_unavailable(tmp_path, monkeypatch):
    stale = tmp_path / "nrv_seed11_clear.png"
    _img().save(stale)                                                  # a stale file from an earlier run
    _, _, _, settings, _, status = _run(_Eng(0.1), tmp_path, monkeypatch)
    assert json.loads(settings)["nsfw_flagged"] is False and "blurred" not in status and not stale.exists()
    _, _, _, settings, _, _ = _run(_Eng(None), tmp_path, monkeypatch, seed=12)       # filter off or unavailable
    assert "nsfw_flagged" not in json.loads(settings)


def test_reveal_never_raises():
    assert A.reveal("not json") is None and A.reveal(json.dumps({"seed": 999999999})) is None


def test_classifier_off_error_and_threshold(monkeypatch):
    assert nsfw.Classifier(enabled=False).score(_img()) is None
    c = nsfw.Classifier(enabled=True)
    monkeypatch.setattr(c, "_load", lambda: (_ for _ in ()).throw(RuntimeError("no model")))
    assert c.score(_img()) is None and "RuntimeError" in c.error and c.score(_img()) is None
    assert nsfw.is_flagged(0.51) and not nsfw.is_flagged(0.5) and not nsfw.is_flagged(None)


def test_blur_changes_the_image_but_not_its_size():
    b = nsfw.blur(_img())
    assert b.size == (256, 256) and list(b.getdata()) != list(_img().getdata())


def test_time_note(tmp_path):
    assert "takes a while" in core.time_note(tmp_path / "missing.json")      # no file: no invented number
    f = tmp_path / "lat.json"
    cfg = {k: {"median_s": v} for k, v in {"text_only": 9.4, "text_pose": 18.2, "text_pose_hires": 36.0}.items()}
    f.write_text(json.dumps({"gpu": "Tesla T4", "configs": cfg}))
    n = core.time_note(f)
    assert "Tesla T4" in n and "9 s" in n and "18 s" in n and "36 s" in n
    f.write_text("{ broken")
    assert "takes a while" in core.time_note(f)


def test_ui_builds_and_has_show_button():
    pytest.importorskip("gradio")
    demo = A.build(None)                                                # --no_model path
    labels = [getattr(b, "value", None) for b in demo.blocks.values()]
    assert "Show image" in labels and "Generate" in labels

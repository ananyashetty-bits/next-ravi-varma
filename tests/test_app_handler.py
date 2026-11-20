"""App handler (app/app.py make_handler) with a fake engine: no GPU, no model, no Gradio needed.

Checks the graceful-error rule: empty, too long, blocked, no person, out-of-memory -> a friendly message, never a
traceback; uploaded photos are deleted; every returned image carries the label."""

import json

import pytest
from PIL import Image

from app import app as A
from app import core


class FakeEngine(A.Engine):
    def __init__(self, fail=None, n_people=1):
        self.fail, self.n_people = fail, n_people
        self.control = None
        self.pose = self
        self.post = None
        self.calls = []

    def parse_scene(self, text):
        return {"characters": [{"canonical": "Sita"}] if "Sita" in text else []}

    def expand(self, text, scene, mode):
        assert mode == "raw"
        return {"positive": f"rrvarma style, {text}, oil painting", "negative": "ugly, blurry, bad anatomy"}

    # pose module API (C's stub signatures)
    def make_pose(self, name):
        return {"image": Image.new("RGB", (512, 512)), "n_people": self.n_people, "source": f"preset:{name}",
                "warnings": []}

    def extract_pose(self, image):
        return {"image": Image.new("RGB", (512, 512)), "n_people": self.n_people, "source": "photo",
                "warnings": ["low confidence"] if self.n_people else []}

    def pose_from_text(self, x):
        return self.make_pose("standing")

    def generate(self, positive, negative, seed, style, ctrl, pose_strength):
        if self.fail == "oom":
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        if self.fail == "boom":
            raise ValueError("unexpected")
        self.calls.append((positive, seed, style, ctrl is not None, pose_strength))
        return Image.new("RGB", (512, 512), (90, 60, 30)), False


def run(engine, text="Sita under a tree", mode="none", preset=None, photo=None, seed=7):
    return A.make_handler(engine)(text, mode, preset, photo, 1.0, 0.8, seed, False, False)


def test_success_is_labelled_and_reproducible():
    eng = FakeEngine()
    img, skel, details, settings, path, status = run(eng)
    assert img.height > 512 and status.startswith("Done")
    s = json.loads(settings)
    assert s["seed"] == 7 and s["label"] == core.LABEL and s["lora_scale"] == 1.0
    assert Image.open(path).info["Description"] == core.LABEL
    assert "Sita" in details


@pytest.mark.parametrize("text", ["", "   ", "word " * 400])
def test_bad_text_gives_a_message(text):
    out = run(FakeEngine(), text=text)
    assert out[0] is None and out[-1] and "Traceback" not in out[-1]


def test_blocked_prompt_never_generates():
    eng = FakeEngine()
    out = run(eng, text="naked Sita in the river")
    assert out[0] is None and "sexualised" in out[-1]
    assert eng.calls == []


def test_upload_is_deleted_and_no_person_falls_back(tmp_path):
    photo = tmp_path / "me.png"
    Image.new("RGB", (64, 64)).save(photo)
    eng = FakeEngine(n_people=0)
    out = run(eng, mode="upload photo", photo=str(photo))
    assert not photo.exists()
    assert out[0] is not None and "No person was found" in out[-1]
    assert eng.calls[-1][3] is False                     # generated without a pose


def test_upload_deleted_even_when_blocked(tmp_path):
    photo = tmp_path / "me.png"
    Image.new("RGB", (64, 64)).save(photo)
    run(FakeEngine(), text="Ganesha as a cartoon", mode="upload photo", photo=str(photo))
    assert not photo.exists()


def test_pose_shown_but_not_applied_is_said():
    out = run(FakeEngine(), mode="preset", preset="kneeling")
    assert out[1] is not None and "not applied" in out[-1]


@pytest.mark.parametrize("fail,needle", [("oom", "ran out of memory"), ("boom", "Something went wrong")])
def test_errors_are_friendly(fail, needle):
    out = run(FakeEngine(fail=fail))
    assert out[0] is None and needle in out[-1] and "Traceback" not in out[-1]


def test_ui_builds_without_a_model():
    pytest.importorskip("gradio")
    demo = A.build(None)
    assert demo is not None

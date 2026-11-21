"""CI tests for the M4 additions to eval/score.py (no judge, no detector)."""
import json, os, subprocess, sys
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from eval.score import apply_overrides, load_pairs, control_pose_for


def test_override_replaces_checklist_without_touching_scene_file(tmp_path):
    scenes = {"s22": {"id": "s22", "checklist": [{"q": "more than three people?"}]}, "s21": {"id": "s21", "checklist": [1]}}
    ov = [{"id": "c1", "q": "exactly three main figures?", "type": "count", "weight": 1.0}] * 3
    pairs = {"s22": {"scene_id": "s22", "checklist_override": ov}, "s21": {"scene_id": "s21", "checklist_override": None}}
    out = apply_overrides(scenes, pairs)
    assert out["s22"]["checklist"] == ov and out["s21"]["checklist"] == [1]
    assert scenes["s22"]["checklist"] == [{"q": "more than three people?"}]          # original untouched


def test_control_pose_defaults_to_pair_pose():
    pairs = load_pairs(os.path.join(ROOT, "eval", "test_pairs_dev.json"))
    sid = next(iter(pairs))
    row = {"scene_id": sid, "control_pose_json": None, "control_file": os.path.join(ROOT, pairs[sid]["pose_file"])}
    pj = control_pose_for(row, pairs, ROOT)
    assert len(pj["people"]) == pairs[sid]["n_people_expected"]


def test_control_pose_prefers_sidecar(tmp_path):
    side = tmp_path / "x_pose.json"
    side.write_text(json.dumps({"people": [{"keypoints": [[0.5, 0.5, 1.0]] * 18}] * 2}))
    row = {"scene_id": "zz", "control_pose_json": str(side), "control_file": None}
    assert len(control_pose_for(row, {}, str(tmp_path))["people"]) == 2


def test_refuses_to_overwrite(tmp_path):
    out = tmp_path / "r.json"; out.write_text("{}")
    man = tmp_path / "manifest.json"; man.write_text(json.dumps({"images": []}))
    r = subprocess.run([sys.executable, os.path.join(ROOT, "eval", "score.py"), "--manifest", str(man), "--out", str(out),
                        "--no_judge", "--no_m2"], capture_output=True, text=True)
    assert r.returncode != 0 and "never overwrites" in (r.stderr + r.stdout)
    assert out.read_text() == "{}"

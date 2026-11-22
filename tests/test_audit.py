import json, os, sys
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from eval.audit_real_paintings import list_images


def test_list_images_folder_skips_skeletons(tmp_path):
    for n in ["a.jpg", "b.png", "s01_pose.png", "notes.txt"]:
        (tmp_path / n).write_bytes(b"x")
    assert [os.path.basename(p) for p in list_images(str(tmp_path))] == ["a.jpg", "b.png"]


def test_list_images_manifest_keeps_existing(tmp_path):
    (tmp_path / "x.png").write_bytes(b"x")
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps({"images": [{"path": str(tmp_path / "x.png")}, {"path": str(tmp_path / "gone.png")}]}))
    assert list_images(str(m)) == [str(tmp_path / "x.png")]

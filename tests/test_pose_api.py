import pose
from pose import presets, stub


def test_public_names_exist():
    for n in ("make_pose", "pose_from_text", "extract_pose", "extract_edges"):
        assert callable(getattr(pose, n))


def test_make_pose_is_the_real_one_not_the_stub():
    assert pose.make_pose is presets.make_pose and pose.make_pose is not stub.make_pose

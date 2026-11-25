"""Public entry point for pose/ (Overview 6.5). Everyone imports from here:
    from pose import make_pose, pose_from_text, extract_pose, extract_edges
All four are real (pose/stub.py is kept for tests)."""
from pose.presets import make_pose                          # real
from pose.text import pose_from_text                        # real
from pose.extract import extract_pose, extract_edges      # real

__all__ = ["make_pose", "pose_from_text", "extract_pose", "extract_edges"]

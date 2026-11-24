"""Public entry point for pose/ (Overview 6.5). Everyone imports from here:
    from pose import make_pose, pose_from_text, extract_pose, extract_edges
make_pose and pose_from_text are real. extract_pose and extract_edges are stubs until pose/extract.py lands; each
replaces its stub import below, so nobody else has to change an import line."""
from pose.presets import make_pose                          # real
from pose.text import pose_from_text                        # real
from pose.extract import extract_pose, extract_edges           # stubs for now

__all__ = ["make_pose", "pose_from_text", "extract_pose", "extract_edges"]

"""Public entry point for pose/ (Overview 6.5). Everyone imports from here:
    from pose import make_pose, pose_from_text, extract_pose, extract_edges
make_pose is real. The other three are stubs until pose/text.py and pose/extract.py land; each replaces its stub
import below, so nobody else has to change an import line."""
from pose.presets import make_pose                                          # real
from pose.stub import pose_from_text, extract_pose, extract_edges           # stubs for now

__all__ = ["make_pose", "pose_from_text", "extract_pose", "extract_edges"]

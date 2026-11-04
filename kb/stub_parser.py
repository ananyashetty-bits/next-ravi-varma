"""Stub parser. Re-exports D's contract-compliant shim until C's real kb/parser.py lands.
Contract (Overview 6.5): parse_scene(text, scene_id="", kb_path=None) -> SceneSpec.
The previous version took only (text) and returned a different shape, which broke expander tests."""
from expander._shims.stub_parser import parse_scene  # noqa: F401

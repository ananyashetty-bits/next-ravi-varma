def parse_scene(text):
    """Stub parser returning a dummy SceneSpec."""
    return {
        "scene_id": "stub_01", 
        "players": {
            "char": text, 
            "action": "standing", 
            "setting": "background", 
            "mood": "dramatic lighting",
            "tail": "highly detailed"
        }
    }
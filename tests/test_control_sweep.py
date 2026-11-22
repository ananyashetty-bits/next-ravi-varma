import importlib.util, os
ROOT = os.path.join(os.path.dirname(__file__), "..")
_s = importlib.util.spec_from_file_location("control_sweep", os.path.join(ROOT, "pipeline", "control_sweep.py"))
CS = importlib.util.module_from_spec(_s); _s.loader.exec_module(CS)
PER_CELL = len(CS.FIXTURES) * len(CS.SEEDS)


def test_stage1_image_counts_match_the_guide():
    c = CS.cells()
    assert len(c) == 24 and len({x["name"] for x in c}) == 24
    assert len(c) * PER_CELL == 144                                  # 120 grid + 24 window
    assert sum(x["name"].startswith("win_") for x in c) * PER_CELL == 24


def test_window_runs_first_and_lora_06_last():
    c = CS.cells()
    assert all(x["name"].startswith("win_") for x in c[:4])
    assert c[-1]["lora_scale"] == 0.6 and c[4]["lora_scale"] == 1.0


def test_dropping_lora_06_is_the_documented_fallback():
    c = CS.cells(loras=[1.0, 0.8, 1.2])
    assert len(c) == 4 + 5 * 3 and all(x["lora_scale"] != 0.6 for x in c)

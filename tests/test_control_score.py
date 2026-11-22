import importlib.util, json, os
ROOT = os.path.join(os.path.dirname(__file__), "..")
_s = importlib.util.spec_from_file_location("control_score", os.path.join(ROOT, "pipeline", "control_score.py"))
CS = importlib.util.module_from_spec(_s); _s.loader.exec_module(CS)


def _cell(tmp_path, name, c, l, ge, vals):
    d = tmp_path / name; d.mkdir()
    imgs = []
    for i, v in enumerate(vals):
        p = d / f"f01_s{i + 1}.png"; p.write_bytes(b"x")
        imgs.append({"fixture": "f01", "seed": i + 1, "path": str(p), "prompt": "p"})
    (d / "manifest.json").write_text(json.dumps({"cell": {"name": name, "control_scale": c, "lora_scale": l, "guidance_end": ge}, "images": imgs}))


def test_collect_and_aggregate(tmp_path):
    _cell(tmp_path, "c0.8_l1.0", 0.8, 1.0, 1.0, [0, 0]); _cell(tmp_path, "win_ge0.6", 1.0, 1.0, 0.6, [0, 0])
    rows = CS.collect(str(tmp_path))
    assert len(rows) == 4
    for r, v in zip(rows, [0.10, 0.20, 0.30, 0.50]):
        r["style_margin"] = v; r["max_train_dino"] = v / 2
    t = {x["cell"]: x for x in CS.aggregate(rows)}
    assert t["c0.8_l1.0"]["kind"] == "grid" and t["win_ge0.6"]["kind"] == "window"
    assert abs(t["c0.8_l1.0"]["style_margin"] - 0.15) < 1e-9 and abs(t["win_ge0.6"]["style_margin"] - 0.40) < 1e-9
    assert abs(t["win_ge0.6"]["max_train_dino_max"] - 0.25) < 1e-9
    assert "pose_pck" not in t["c0.8_l1.0"]

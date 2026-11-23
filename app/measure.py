"""Measurements for APP.md (owner D). Colab T4, gen profile + post/install_post.sh.

  python app/measure.py latency        -> results/app_latency.json        (H-D4)
  python app/measure.py upload_cache   -> results/app_upload_cache.json   (N for the upload note)

latency: cold start (engine load + first text-only image) reported separately; then for each configuration
  (text only; + preset pose; + hi-res; + face restoration) 1 warm-up and 5 timed end-to-end runs through the app's own
  handler (policy, parse, prompt, pose, generation, post, label, PNG save). Median, min, max and peak VRAM
  (torch.cuda.max_memory_allocated, reset per configuration).
upload_cache: launches the real app locally, uploads a test image through the Gradio API exactly as a browser does
  (pose mode "upload photo"), then watches Gradio's temp folder for every file with the upload's sha256 and records
  when the last copy is gone. The test image is a synthetic picture, not a photo of a person.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("APP_POSE_MODULE", "pose")          # C's public entry point (real make_pose)
os.environ.setdefault("APP_POST_MODULE", "post.api")      # D's real hi-res and GFPGAN

TEXT = "Hanuman kneeling before Rama and offering him Sita's ring"
CONFIGS = [  # name, pose mode, preset, hires, restore
    ("text_only", "none", None, False, False),
    ("text_preset_pose", "preset", "kneeling", False, False),
    ("text_pose_hires", "preset", "kneeling", True, False),
    ("text_pose_hires_restore", "preset", "kneeling", True, True),
]


def _gpu():
    import torch
    return torch


def latency(n=5, out="results/app_latency.json"):
    torch = _gpu()
    t0 = time.time()
    from app import app as A
    engine = A.Engine(warm=True)
    run = A.make_handler(engine)
    d = A.DEFAULTS
    r = run(TEXT, "none", None, None, d["style"], d["pose"], 1, False, False)
    cold = time.time() - t0
    assert r[0] is not None, r[-1]
    rows = []
    for name, mode, preset, hires, restore in CONFIGS:
        run(TEXT, mode, preset, None, d["style"], d["pose"], 1, hires, restore)          # warm-up (loads post models)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        times = []
        for seed in range(2, 2 + n):
            s = time.time()
            r = run(TEXT, mode, preset, None, d["style"], d["pose"], seed, hires, restore)
            torch.cuda.synchronize()
            times.append(time.time() - s)
            assert r[0] is not None, (name, r[-1])
        settings = json.loads(r[3])
        rows.append({"config": name, "n": n, "median_s": round(statistics.median(times), 2),
                     "min_s": round(min(times), 2), "max_s": round(max(times), 2),
                     "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                     "pose_applied": settings.get("control_scale") is not None, "post": settings.get("post")})
        print(rows[-1])
    res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "text": TEXT,
           "cold_start_s": round(cold, 1), "cold_start_note": "engine load (weights cached on disk) + first "
           "text-only image; a brand-new runtime adds the model download",
           "pose_module": os.environ["APP_POSE_MODULE"], "post_module": os.environ["APP_POST_MODULE"],
           "defaults": d, "rows": rows,
           "h_d4_text_pose_hires_median_under_60s": next(r["median_s"] for r in rows
                                                        if r["config"] == "text_pose_hires") < 60}
    Path(ROOT / out).write_text(json.dumps(res, indent=1) + "\n")
    print("wrote", out)


def _files_with(root: Path, digest: str) -> list[str]:
    hits = []
    for p in root.rglob("*"):
        if p.is_file():
            try:
                if hashlib.sha256(p.read_bytes()).hexdigest() == digest:
                    hits.append(str(p))
            except OSError:
                pass
    return hits


def upload_cache(max_minutes=12, poll_s=10, out="results/app_upload_cache.json", engine=None):
    from PIL import Image, ImageDraw
    from gradio_client import Client, handle_file
    from app import app as A
    temp = Path(os.environ.get("GRADIO_TEMP_DIR", Path(tempfile.gettempdir()) / "gradio"))
    img = Image.new("RGB", (400, 600), (200, 190, 170))
    ImageDraw.Draw(img).ellipse((150, 60, 250, 160), fill=(90, 60, 40))                  # synthetic, no person
    test = Path(tempfile.mkdtemp()) / "upload_test.png"
    img.save(test)
    digest = hashlib.sha256(test.read_bytes()).hexdigest()
    engine = engine or A.Engine(warm=True)
    demo = A.build(engine)
    demo.launch(prevent_thread_lock=True, server_port=7861, quiet=True)
    client = Client("http://127.0.0.1:7861/", verbose=False)
    t_up = time.time()
    r = client.predict(TEXT, "upload photo", None, handle_file(str(test)), A.DEFAULTS["style"], A.DEFAULTS["pose"],
                       1, False, False, api_name="/on_go")
    t_done = time.time()
    test.unlink()                                                    # our own copy; only Gradio's copies remain
    log, gone_at = [], None
    while time.time() - t_done < max_minutes * 60:
        hits = _files_with(temp, digest)
        log.append({"t_after_response_s": round(time.time() - t_done), "copies": hits})
        print(log[-1])
        if not hits:
            gone_at = time.time() - t_done
            break
        time.sleep(poll_s)
    demo.close()
    res = {"gradio_temp_dir": str(temp), "delete_cache": [A.CACHE_EVERY_S, A.CACHE_MAX_AGE_S],
           "engine": type(engine).__name__, "request_s": round(t_done - t_up, 1), "status": r[5] if isinstance(r, (list, tuple)) and len(r) > 5 else None,
           "all_copies_gone_after_s": None if gone_at is None else round(gone_at),
           "N_minutes": None if gone_at is None else max(1, round(gone_at / 60 + 0.49)),
           "note": "N = minutes until no file with the upload's sha256 remains under the Gradio temp folder, measured "
                   "from the end of the request; polled every %d s." % poll_s, "log": log}
    Path(ROOT / out).write_text(json.dumps(res, indent=1) + "\n")
    print("wrote", out, "| N =", res["N_minutes"], "min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["latency", "upload_cache"])
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    latency(a.n) if a.what == "latency" else upload_cache()

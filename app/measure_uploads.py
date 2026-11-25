"""Which uploads are still on disk when the request ends? (issue 158; owner D; CPU only, needs gradio)

  python app/measure_uploads.py      -> results/app_upload_formats.json

Real Gradio layer and handler (as a browser upload, through gradio_client); the pose detector and image generation are
fakes, and every upload is random noise, not a photo. `left` lists files with the upload's name under a fresh Gradio
temp folder right after the response.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

T = os.environ["GRADIO_TEMP_DIR"] = tempfile.mkdtemp()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OK, BLOCKED = "Sita under a tree", "naked Sita"
CASES = [  # file name, PIL mode, format, EXIF rotation, text, pose mode
    ("rgb.png", "RGB", "PNG", False, OK, "upload photo"), ("rgb.jpg", "RGB", "JPEG", False, OK, "upload photo"),
    ("my photo (1).jpg", "RGB", "JPEG", False, OK, "upload photo"), ("grey.jpg", "L", "JPEG", False, OK, "upload photo"),
    ("cmyk.jpg", "CMYK", "JPEG", False, OK, "upload photo"), ("rgba.png", "RGBA", "PNG", False, OK, "upload photo"),
    ("exif.jpg", "RGB", "JPEG", True, OK, "upload photo"), ("rgb.webp", "RGB", "WEBP", False, OK, "upload photo"),
    ("none.jpg", "RGB", "JPEG", False, OK, "none"), ("preset.jpg", "RGB", "JPEG", False, OK, "preset"),
    ("blocked.jpg", "RGB", "JPEG", False, BLOCKED, "upload photo"), ("empty.jpg", "L", "JPEG", False, "", "upload photo"),
]


def main():
    import gradio
    import numpy as np
    from gradio_client import Client, handle_file
    from PIL import Image
    from app import app as A
    from tests.test_app_handler import FakeEngine
    demo = A.build(FakeEngine())
    demo.launch(prevent_thread_lock=True, server_port=7862, quiet=True)
    client = Client("http://127.0.0.1:7862/", verbose=False)
    rng, rows = np.random.default_rng(0), []
    for name, mode, fmt, rotated, text, pose in CASES:
        src = Path(tempfile.mkdtemp()) / name
        exif = Image.Exif()
        if rotated:
            exif[274] = 6
        Image.fromarray(rng.integers(0, 255, (90, 60, 3), dtype="uint8")).convert(mode).save(src, fmt, exif=exif)
        client.predict(text, pose, None, handle_file(str(src)), 1.0, 0.8, 1, False, False, api_name="/on_go")
        left = [str(p.relative_to(T)) for p in Path(T).rglob("*") if p.is_file() and p.stem == src.stem]
        rows.append({"file": name, "mode": mode, "pose_mode": pose, "exif_rotated": rotated,
                     "prompt": "blocked" if text == BLOCKED else ("ok" if text else "empty"), "left": left})
        print(rows[-1])
    demo.close()
    res = {"gradio": gradio.__version__, "engine": "FakeEngine (CPU)", "n_left": sum(bool(r["left"]) for r in rows),
           "rows": rows}
    (ROOT / "results" / "app_upload_formats.json").write_text(json.dumps(res, indent=1) + "\n")
    print("files left:", res["n_left"], "of", len(rows))


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)

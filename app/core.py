"""Pure logic for the app (owner D): defaults from config, input checks, the content label, PNG metadata.

CPU-only and importable without torch or Gradio, so CI tests it (tests/test_app_core.py). app/app.py is the thin
Gradio layer on top.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo

ROOT = Path(__file__).resolve().parents[1]
APP_VERSION = "m4-app0.1"
LABEL = "AI-generated in the style of Raja Ravi Varma. Not an authentic work."
MAX_WORDS = 150          # longer texts get a friendly refusal; CLIP reads only ~60 words of a prompt anyway
LONG_WORDS = 50          # above this the style tail may fall outside CLIP's 77 tokens: warn, don't refuse


def load_configs() -> tuple[dict, dict]:
    m3 = json.loads((ROOT / "config" / "m3.json").read_text(encoding="utf-8"))
    m4p = ROOT / "config" / "m4.json"
    m4 = json.loads(m4p.read_text(encoding="utf-8")) if m4p.exists() else {}
    return m3, m4


def defaults(m3: dict | None = None, m4: dict | None = None) -> dict:
    """Slider defaults and ranges, read from config only (never hardcoded)."""
    if m3 is None or m4 is None:
        m3, m4 = load_configs()
    ui = m4.get("ui", {})
    style_range = list(ui.get("style_slider_range", [1.0, 1.0]))
    pose_range = list(ui.get("pose_slider_range", [1.0, 1.0]))
    model = m4.get("default_model") or m3.get("default_model", "rrv_v1")
    style = m4.get("lora_scale") if m4.get("lora_scale") is not None else m3["models"][model]["scale"]
    control = m4.get("control", {})
    prof = control.get("tuned") or control.get("baseline") or {"scale": 1.0}
    return {
        "model": model,
        "style": _clamp(style, style_range),
        "style_range": style_range,
        "pose": _clamp(prof.get("scale", 1.0), pose_range),
        "pose_range": pose_range,
        "pose_profile": "tuned" if control.get("tuned") else "baseline",
        "post_default_on": bool(m4.get("post", {}).get("default_on", False)),
    }


def _clamp(x: float, rng: list) -> float:
    lo, hi = float(rng[0]), float(rng[-1])
    return float(min(max(float(x), lo), hi))


def check_text(text: str) -> dict:
    """{"ok": bool, "message": str, "warning": str}. Policy is checked separately (app/policy.py)."""
    words = (text or "").split()
    if not words:
        return {"ok": False, "message": "Please describe a scene first, e.g. \"Sita sitting alone under a tree in "
                                        "the Ashoka grove\".", "warning": ""}
    if len(words) > MAX_WORDS:
        return {"ok": False, "message": f"That description has {len(words)} words. Please shorten it to under "
                                        f"{MAX_WORDS}: the model reads only about the first 60 words, so the "
                                        "rest would be ignored anyway.", "warning": ""}
    warning = ""
    if len(words) > LONG_WORDS:
        warning = ("Long description: the model reads only about the first 60 words, so details near the end may be "
                   "ignored. Put the most important figures and action first.")
    return {"ok": True, "message": "", "warning": warning}


def add_label(image: Image.Image, text: str = LABEL) -> Image.Image:
    """Return a copy with a caption bar under the image (the visible content label)."""
    img = image.convert("RGB")
    bar = max(24, img.height // 20)
    out = Image.new("RGB", (img.width, img.height + bar), (20, 20, 20))
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default(size=max(11, bar // 2))
    except TypeError:                                   # Pillow < 10.1
        font = ImageFont.load_default()
    draw.text((8, img.height + (bar - max(11, bar // 2)) // 2 - 1), text, fill=(235, 235, 235), font=font)
    return out


def metadata(settings: dict) -> dict:
    """What goes into the PNG text chunks: the label, model, seed, app version and every setting."""
    return {"Description": LABEL, "Software": APP_VERSION,
            "parameters": json.dumps(settings, sort_keys=True, ensure_ascii=False)}


def save_png(image: Image.Image, path: str | Path, settings: dict) -> Path:
    info = PngInfo()
    for k, v in metadata(settings).items():
        info.add_text(k, v)
    path = Path(path)
    image.save(path, pnginfo=info)
    return path


def settings_record(text: str, prompt: dict, seed: int, style: float, pose_mode: str, pose_strength: float | None,
                    pose_source: str | None, post: list[str], model: str) -> dict:
    """The "copy settings" JSON: enough to reproduce the image from the CLI."""
    return {"app_version": APP_VERSION, "text": text, "positive": prompt.get("positive"),
            "negative": prompt.get("negative"), "seed": int(seed), "model": model, "lora_scale": float(style),
            "pose_mode": pose_mode, "control_scale": None if pose_strength is None else float(pose_strength),
            "pose_source": pose_source, "post": list(post), "label": LABEL}

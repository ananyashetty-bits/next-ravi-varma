"""Output NSFW backstop: a soft blur-and-confirm, never a block (docs/CONTENT_POLICY.md section 5).

Decision (B's audit, results/audit_real_paintings.json): 1 of 15 real Varma paintings flagged, which is not more
than 1, so the rule enables a soft filter. It runs on app output only, never on evaluation images. It is a backstop:
the input blocklist and the content label do the real work, and the filter will fire on some innocent paintings
(nursing, bathing, drapery). Runs on CPU (about 330 MB, loaded on first use) so it never competes for T4 memory.
Off with APP_NSFW=0. If the model cannot load, the filter stays off and the reason goes to the server log."""

from __future__ import annotations

import os
import sys

from PIL import Image, ImageFilter

MODEL = "Falconsai/nsfw_image_detection"
THRESHOLD = 0.5                                   # same threshold as eval/audit_real_paintings.py
LABELS = ("nsfw", "porn", "sexy", "hentai")


class Classifier:
    def __init__(self, model: str = MODEL, enabled: bool | None = None):
        self.model = model
        self.enabled = (os.environ.get("APP_NSFW", "1") != "0") if enabled is None else enabled
        self._clf = None
        self.error: str | None = None

    def _load(self):
        from transformers import pipeline
        self._clf = pipeline("image-classification", model=self.model, device=-1)

    def score(self, image: Image.Image) -> float | None:
        """NSFW score 0-1, or None when the filter is off or unavailable (never raises)."""
        if not self.enabled or self.error:
            return None
        try:
            if self._clf is None:
                self._load()
            out = self._clf(image.convert("RGB"))
            return float(next((o["score"] for o in out if str(o["label"]).lower() in LABELS), 0.0))
        except Exception as e:                    # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
            print(f"[app] NSFW filter is OFF: {self.error}", file=sys.stderr)
            return None


def is_flagged(score: float | None, threshold: float = THRESHOLD) -> bool:
    return score is not None and score > threshold


def blur(image: Image.Image) -> Image.Image:
    """Heavy blur: the composition stays visible, the detail does not."""
    img = image.convert("RGB")
    return img.filter(ImageFilter.GaussianBlur(radius=max(24, img.width // 12)))

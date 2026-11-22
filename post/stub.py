"""Post-processing stubs (owner D; Overview 6.5). Passthroughs with the real signatures, so the app and other
members can call them before the real post/ functions land. Replaced by post/hires.py and post/restore.py."""

from PIL import Image


def hires(image: Image.Image, positive: str, negative: str, seed: int, method: str = "img2img",
          lora_scale=None, strength=None) -> Image.Image:
    """Stub: returns the input unchanged."""
    return image


def restore_faces(image: Image.Image, method: str = "gfpgan") -> Image.Image:
    """Stub: returns the input unchanged."""
    return image

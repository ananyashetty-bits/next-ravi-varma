"""Face restoration (owner D). Contract: restore_faces(image, method="gfpgan") -> PIL.Image.

GFPGAN v1.3 at upscale 1, so the output keeps the input size and only faces change (pasted back). `weight` blends
the restored face with the original (0 = original, 1 = fully restored); the default 0.5 is recorded in every manifest.
H-D2 expects this to raise face ratings and lower style (photographic faces): that is measured, not assumed.
Needs gfpgan + basicsr (see post/install_post.sh; basicsr does not build on Python 3.13 without a patch).
Heavy imports are lazy, so this module imports on CPU (CI).
"""
import os

from PIL import Image

GFPGAN_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
DEFAULT_WEIGHT = 0.5
_R = {"obj": None}


def _restorer():
    if _R["obj"] is None:
        from gfpgan import GFPGANer
        _R["obj"] = GFPGANer(model_path=GFPGAN_URL, upscale=1, arch="clean", channel_multiplier=2, bg_upsampler=None)
    return _R["obj"]


def restore_faces_info(image: Image.Image, method: str = "gfpgan", weight: float = DEFAULT_WEIGHT):
    """-> (image, n_faces_found). If no face is found the image is returned unchanged (n_faces_found == 0)."""
    if method in (None, "none"):
        return image, 0
    if method != "gfpgan":
        raise ValueError(f"unknown restore method '{method}'; use 'gfpgan' or 'none'")
    import numpy as np
    bgr = np.array(image.convert("RGB"))[:, :, ::-1]
    _, faces, out = _restorer().enhance(bgr, has_aligned=False, only_center_face=False, paste_back=True,
                                        weight=weight)
    if not faces or out is None:
        return image, 0
    res = Image.fromarray(out[:, :, ::-1])
    if res.size != image.size:                      # upscale=1 should keep the size; guard anyway
        res = res.resize(image.size, Image.LANCZOS)
    return res, len(faces)


def restore_faces(image, method="gfpgan", weight=DEFAULT_WEIGHT):
    return restore_faces_info(image, method, weight)[0]

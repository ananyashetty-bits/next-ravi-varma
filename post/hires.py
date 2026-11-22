"""Hi-res post-processing (owner D). Contract: hires(image, positive, negative, seed, method="img2img") -> PIL.Image.

img2img (default): Lanczos upscale 512 -> `post.hires.size`, then SD img2img at `post.hires.strength` with the SAME
weights, LoRA, c1 prompt and negative as the base image (`from_pipe` on M3's pipeline, so the LoRA state is shared).
Never generate directly at 768: SD1.5 duplicates figures. If figures duplicate, lower the strength in
config/m4.json (0.25-0.3).
esrgan (variant): Real-ESRGAN x2, adds detail without redrawing. Untested until run in Colab.

The prompt and negative are passed in (they come from the prompt record); nothing is typed here. Heavy imports are
lazy, so this module imports on CPU (CI).
"""
import json
import os

from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ESRGAN_X2_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"

_I2I = {"base_id": None, "pipe": None}
_ESRGAN = {"up": None}


def load_post_config(m4_path=None) -> dict:
    p = m4_path or os.path.join(ROOT, "config", "m4.json")
    return json.load(open(p, encoding="utf-8")).get("post", {})


def hires_settings(m4_path=None) -> dict:
    """{'method','size','strength'} straight from config/m4.json."""
    h = load_post_config(m4_path).get("hires", {})
    return {"method": h.get("method", "img2img"), "size": int(h.get("size", 768)),
            "strength": float(h.get("strength", 0.35))}


def lanczos_up(image: Image.Image, size: int) -> Image.Image:
    return image.convert("RGB").resize((size, size), Image.LANCZOS)


def _img2img_pipe(lora_scale):
    from diffusers import StableDiffusionImg2ImgPipeline
    from pipeline.generate_api import _ensure
    base = _ensure(None, lora_scale)                 # M3's own pipeline; LoRA scale from m3.json unless overridden
    if _I2I["pipe"] is None or _I2I["base_id"] != id(base):
        _I2I["pipe"] = StableDiffusionImg2ImgPipeline.from_pipe(base)
        _I2I["base_id"] = id(base)
    return _I2I["pipe"]


def _hires_img2img(image, positive, negative, seed, lora_scale=None, strength=None, size=None):
    import torch
    from pipeline.generate_api import CFG
    s = hires_settings()
    strength = s["strength"] if strength is None else float(strength)
    size = s["size"] if size is None else int(size)
    pipe = _img2img_pipe(lora_scale)
    g = torch.Generator("cuda").manual_seed(int(seed))
    return pipe(prompt=positive, negative_prompt=negative, image=lanczos_up(image, size), strength=strength,
                num_inference_steps=CFG["steps"], guidance_scale=CFG["cfg"], generator=g).images[0]


def _hires_esrgan(image):
    import numpy as np
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    if _ESRGAN["up"] is None:
        net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        _ESRGAN["up"] = RealESRGANer(scale=2, model_path=ESRGAN_X2_URL, model=net, half=True)
    out, _ = _ESRGAN["up"].enhance(np.array(image.convert("RGB"))[:, :, ::-1], outscale=2)   # BGR in, BGR out
    return Image.fromarray(out[:, :, ::-1])


def hires(image, positive, negative, seed, method="img2img", lora_scale=None, strength=None):
    """Higher-resolution version of `image`. `lora_scale`/`strength` are optional overrides (None = config)."""
    if method == "img2img":
        return _hires_img2img(image, positive, negative, seed, lora_scale, strength)
    if method == "esrgan":
        return _hires_esrgan(image)
    raise ValueError(f"unknown hires method '{method}'; use 'img2img' or 'esrgan'")

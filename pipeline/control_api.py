"""generate_controlled(...) -> PIL.Image: SD1.5 + M3's LoRA + one ControlNet (Overview 6.5).

The LoRA scale is set ONLY through M3's _ensure(), which uses M2's set_scale (set_adapters).
Never pass cross_attention_kwargs={"scale": ...}: on PEFT-backend diffusers it multiplies.
The base pipeline is M3's own object; ControlNet pipelines are derived with from_pipe (shared UNet + LoRA).
"""
import json, os
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from pipeline import generate_api as G

CFG = G.CFG
CFG4 = json.load(open(os.path.join(G.ROOT, "config", "m4.json"), encoding="utf-8"))

_CN_KEY = {"openpose": "openpose", "openpose_full": "openpose", "canny": "canny"}  # full = same weights, other image
_CN = {}                                                   # controlnet key -> model (one resident at a time)
_CACHE = {"base": None, "key": None, "pipe": None}


def profile(name="baseline"):
    """Settings for run_batch --control_profile. tuned is null in m4.json until the Day 2 sweep winner is set."""
    p = CFG4["control"].get(name)
    if p is None:
        raise ValueError(f"control profile '{name}' is not set in config/m4.json")
    return {"control_scale": p["scale"], "guidance_start": p["guidance_start"], "guidance_end": p["guidance_end"],
            "lora_scale": CFG4["lora_scale"] if name == "tuned" else None}


def _controlnet(key):
    if key not in _CN:
        _CN.clear(); _CACHE.update(base=None, key=None, pipe=None); torch.cuda.empty_cache()
        spec = CFG4["controlnets"][key]
        _CN[key] = ControlNetModel.from_pretrained(
            spec["repo"], revision=spec["revision"], torch_dtype=torch.float16).to("cuda")
    return _CN[key]


def _pipe(base, key):
    if _CACHE["base"] is not base or _CACHE["key"] != key:     # new base after release(), or another ControlNet
        pipe = StableDiffusionControlNetPipeline.from_pipe(base, controlnet=_controlnet(key))
        pipe.set_progress_bar_config(disable=True)
        _CACHE.update(base=base, key=key, pipe=pipe)
    return _CACHE["pipe"]


def generate_controlled(positive, negative, seed, control_image, control_type="openpose", model=None,
                        lora_scale=None, control_scale=None, guidance_start=None, guidance_end=None):
    """None = config baseline profile; lora_scale None = M3's models.<model>.scale.
    control_image: ControlImage dict or a bare PIL image."""
    if control_type not in _CN_KEY:
        raise ValueError(f"control_type must be one of {list(_CN_KEY)}, got {control_type!r}")
    b = profile("baseline")
    cs = b["control_scale"] if control_scale is None else control_scale
    gs = b["guidance_start"] if guidance_start is None else guidance_start
    ge = b["guidance_end"] if guidance_end is None else guidance_end
    img = control_image["image"] if isinstance(control_image, dict) else control_image
    img = img.convert("RGB").resize((CFG["size"], CFG["size"]))
    base = G._ensure(model, lora_scale)                    # sets the LoRA scale (M2 set_scale)
    try:
        pipe = _pipe(base, _CN_KEY[control_type])
        g = torch.Generator("cuda").manual_seed(int(seed))
        return pipe(positive, negative_prompt=negative, image=img, num_inference_steps=CFG["steps"],
                    guidance_scale=CFG["cfg"], width=CFG["size"], height=CFG["size"],
                    controlnet_conditioning_scale=cs, control_guidance_start=gs,
                    control_guidance_end=ge, generator=g).images[0]
    finally:
        if lora_scale is not None:
            G._ensure(model, None)                         # restore M3's scale

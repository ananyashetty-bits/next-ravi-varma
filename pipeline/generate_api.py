"""generate(positive, negative, seed, model=None, lora_scale=None, control_image=None) -> PIL.Image"""
import json, torch, sys
sys.path.insert(0, "m2kit")
from generate import load, swap_lora          # M2 code, unchanged

CFG = json.load(open("config/m3.json"))
_PIPE, _CUR = None, None

def _ensure(model):
    global _PIPE, _CUR
    m = model or CFG["default_model"]
    
    if _PIPE is None:
        lora_path = None if m == "base" else CFG["models"][m]["lora_path"]
        scale = CFG["models"].get(m, {}).get("scale", 0)
        _PIPE = load(lora_path, scale)
    elif m != _CUR and m != "base":
        lora_path = CFG["models"][m]["lora_path"]
        scale = CFG["models"].get(m, {}).get("scale", 0)
        swap_lora(_PIPE, lora_path, scale)
        
    _CUR = m
    return _PIPE

def generate(positive, negative, seed, model=None, lora_scale=None, control_image=None):
    pipe = _ensure(model)
    generator = torch.Generator("cuda").manual_seed(seed)
    
    # Passing the exact settings from the config
    image = pipe(
        prompt=positive,
        negative_prompt=negative,
        num_inference_steps=CFG["steps"],
        guidance_scale=CFG["cfg"],
        height=CFG["size"],
        width=CFG["size"],
        generator=generator
    ).images[0]
    
    return image
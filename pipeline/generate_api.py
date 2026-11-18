"""generate(positive, negative, seed, model=None, lora_scale=None, control_image=None) -> PIL.Image

Wraps the unmodified M2 kit (m2kit/generate.py). We call the pipeline directly instead of
m2kit's gen() only because gen() hardcodes its negative prompt and we need config's.
"""
import json, os, sys
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "m2kit"))
from generate import load, swap_lora, set_scale      # M2 code, unchanged

CFG = json.load(open(os.path.join(ROOT, "config", "m3.json"), encoding="utf-8"))
if CFG["sampler"] != "dpmpp_2m":                      # M2's load() fixes the scheduler
    raise ValueError("m2kit.load() only sets dpmpp_2m; config sampler must match")
SAFETY_CHECKER = CFG.get("safety_checker", False)     # off in evaluation: it false-blocks paintings

_PIPE, _CUR, _SCALE = None, None, None

def _attach_safety_checker(pipe):                     # untested; leave config safety_checker=false
    from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
    from transformers import CLIPImageProcessor
    pipe.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
        "CompVis/stable-diffusion-safety-checker").to("cuda", torch.float16)
    pipe.feature_extractor = CLIPImageProcessor.from_pretrained(CFG["base_model"], subfolder="feature_extractor")
    pipe.register_to_config(requires_safety_checker=True)

def _ensure(model, lora_scale):
    global _PIPE, _CUR, _SCALE
    m = model or CFG["default_model"]
    if m != "base" and not CFG["models"].get(m):
        raise ValueError(f"unknown model '{m}'; known: {list(CFG['models'])}")
    scale = 0.0 if m == "base" else (lora_scale if lora_scale is not None else CFG["models"][m]["scale"])
    lora_dir = None if m == "base" else os.path.dirname(CFG["models"][m]["lora_path"])  # M2 expects the dir
    if _PIPE is None:
        _PIPE = load(lora_dir, scale)
        if SAFETY_CHECKER:
            _attach_safety_checker(_PIPE)
    elif m != _CUR:
        swap_lora(_PIPE, lora_dir, scale)             # lora_dir=None unloads, giving the plain base
    elif m != "base" and scale != _SCALE:
        set_scale(_PIPE, scale)
    _CUR, _SCALE = m, scale
    return _PIPE

def release():
    """Free GPU memory (call before loading the LLM)."""
    global _PIPE, _CUR, _SCALE
    _PIPE = _CUR = _SCALE = None
    torch.cuda.empty_cache()

def generate(positive, negative, seed, model=None, lora_scale=None, control_image=None):
    pipe = _ensure(model, lora_scale)                 # control_image: reserved for M4, ignored
    g = torch.Generator("cuda").manual_seed(int(seed))   # per call, so same inputs give the same image
    return pipe(positive, negative_prompt=negative, num_inference_steps=CFG["steps"],
                guidance_scale=CFG["cfg"], height=CFG["size"], width=CFG["size"],
                generator=g).images[0]

def text_to_image(text, seed, mode="raw", model=None):
    """text -> (image, ExpandedPrompt, SceneSpec). mode "raw" (c1) is the default pipeline
    (docs/TEAM_DECISIONS_M3.md D1); template / llm_kb / llm_nokb / llm_kb_layout use D's expander."""
    sys.path.insert(0, ROOT)
    from kb.parser import parse_scene                 # C's real parser with kb/kb.json
    from expander.expand import expand                # D's real module
    scene = parse_scene(text)
    exp = expand(text, scene, mode)
    close = getattr(sys.modules[expand.__module__], "close", None)
    if callable(close):
        close()                                       # free the LLM before diffusion
    torch.cuda.empty_cache()
    return generate(exp["positive"], exp["negative"], seed, model=model), exp, scene
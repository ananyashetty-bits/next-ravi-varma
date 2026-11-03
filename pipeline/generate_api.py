import json
import torch
import sys
from PIL import Image

# Import stubs dynamically
try:
    from kb.parser import parse_scene
    from expander.expand import expand
except ImportError:
    from kb.stub_parser import parse_scene
    from expander.stub_expand import expand

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def text_to_image(text, seed, mode="llm_kb", model=None):
    """Parses scene, expands prompt, and generates image."""
    scene = parse_scene(text)
    exp = expand(text, scene, mode)
    
    if DEVICE == "cpu":
        print("\n[LOCAL CPU DETECTED] Bypassing 4GB model download.")
        print(f"SceneSpec: {scene}")
        print(f"Prompt: {exp.get('positive', '')}")
        # Instantly create a 512x512 blue placeholder image
        img = Image.new('RGB', (512, 512), color=(73, 109, 137))
    else:
        # This will only run later when you are on Colab's T4 GPU
        sys.path.insert(0, "m2kit")
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", 
            torch_dtype=torch.float16, safety_checker=None
        ).to(DEVICE)
        
        generator = torch.Generator(DEVICE).manual_seed(seed)
        img = pipe(
            prompt=exp["positive"], negative_prompt=exp["negative"], 
            generator=generator, num_inference_steps=30
        ).images[0]

    return img, exp, scene
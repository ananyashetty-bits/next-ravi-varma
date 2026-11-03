# Pipeline Implementation & API Specification

## 1. Architecture Overview
The pipeline acts as the primary entry point for single-image generation, orchestrating three main stages:
1. **Scene Parsing (`kb/parser.py` or stub)**: Converts unstructured text into a structured `SceneSpec` dictionary.
2. **Prompt Expansion (`expander/expand.py` or stub)**: Transforms `SceneSpec` into structured positive and negative prompts based on the selected mode (`llm_kb`, `template`, `raw`).
3. **Diffusion Generation (`pipeline/generate_api.py`)**: Feeds expanded prompts into Stable Diffusion v1.5 with the Raja Ravi Varma LoRA (`rrv_v1`) adapter.

---

## 2. Core Python API

```python
text_to_image(text: str, seed: int = 1, mode: str = "llm_kb", model: str = None) -> tuple[PIL.Image.Image, dict, dict]
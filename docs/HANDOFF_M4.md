# Milestone 4 Integration & Handoff Guide

## 1. Integration Interface
For downstream web application, frontend, and API integration, Milestone 4 services should consume the high-level generation interface exposed by `pipeline/generate_api.py`:

```python
from pipeline.generate_api import text_to_image

# Execution contract
image, expanded_prompt, scene_spec = text_to_image(
    text="Sita in Ashoka grove, sorrowful",
    seed=1,
    mode="llm_kb",
    model="rrv_v1"
)

## API Contract
pipeline/generate_api.py exposes text_to_image() for M4 web integration.
#!/usr/bin/env bash
# Usage: bash pipeline/setup_session.sh gen|llm|eval   (from the repo root, one profile per Colab session)
#   gen  = Stable Diffusion + LoRA (M2 kit requirements)
#   llm  = D's expander (local LLM, 4-bit)
#   eval = B's judge and metrics
set -euo pipefail
PROFILE=${1:-gen}
case "$PROFILE" in
  gen)  pip install -q -r m2kit/requirements_m2.txt rapidfuzz ;;   # rapidfuzz: kb/parser.py, used by text_to_image
  llm)  pip install -q -r expander/requirements_llm.txt ;;
  eval) pip install -q -r eval/requirements_eval.txt ;;
  *)    echo "unknown profile '$PROFILE' (use gen, llm or eval)"; exit 1 ;;
esac
pip install -q -e .
# M4: one pinned set for every profile that generates or scores images (must match config/m4.json)
if [ "$PROFILE" = "gen" ] || [ "$PROFILE" = "eval" ]; then
  pip install -q diffusers==0.40.0 transformers==5.17.0 peft==0.21.0 accelerate==1.15.0 controlnet_aux==0.0.10 gradio==6.31.0
  pip uninstall -y -q torchao || true   # old torchao makes peft refuse to load the LoRA; nothing here uses it
fi
python - <<'PY'
import importlib
for m in ("torch", "diffusers", "transformers", "peft", "accelerate", "controlnet_aux", "gradio", "bitsandbytes", "open_clip"):
    try:
        print(f"{m:13s}", getattr(importlib.import_module(m), "__version__", "?"))
    except Exception:
        print(f"{m:13s} not installed")
PY
echo "Setup complete for profile: $PROFILE"

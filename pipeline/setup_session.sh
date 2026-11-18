#!/usr/bin/env bash
# Usage: bash pipeline/setup_session.sh gen|llm|eval   (from the repo root, one profile per Colab session)
#   gen  = Stable Diffusion + LoRA (M2 kit requirements)
#   llm  = D's expander (local LLM, 4-bit)
#   eval = B's judge and metrics
set -euo pipefail
PROFILE=${1:-gen}
case "$PROFILE" in
  gen)  pip install -q -r m2kit/requirements_m2.txt ;;
  llm)  pip install -q -r expander/requirements_llm.txt ;;
  eval) pip install -q -r eval/requirements_eval.txt ;;
  *)    echo "unknown profile '$PROFILE' (use gen, llm or eval)"; exit 1 ;;
esac
pip install -q -e .
python - <<'PY'
import importlib
for m in ("torch", "diffusers", "transformers", "accelerate", "bitsandbytes", "open_clip"):
    try:
        print(f"{m:13s}", getattr(importlib.import_module(m), "__version__", "?"))
    except Exception:
        print(f"{m:13s} not installed")
PY
echo "Setup complete for profile: $PROFILE"

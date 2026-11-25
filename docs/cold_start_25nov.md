# Cold-start test log (25 Nov 2026, owner C)

- **Date:** 25 Nov 2026
- **Environment:** Fresh Google Colab session (Tesla T4 GPU, Python 3.13, `torch 2.14.1+cu130`, `diffusers 0.40.0`, `gradio 6.31.0`)
- **Setup:** Cloned `main`, mounted Drive, and installed dependencies via `bash pipeline/setup_session.sh gen`.

## 1. CLI batch & seed reproducibility
- **Command:** Ran `pipeline/run_batch.py` on dev scene `s01` (`seed 1`) and compared against the reference run.
- **Outcome:** `s01_s1.png` reproduced deterministically with identical output.

## 2. Gradio web app (`app/app.py`)
- **Command:** `python app/app.py --share --auth user:password`
- **Valid inputs tested:**
  1. Misspelled character name (`"Arjunaah standing in a royal court"`) with `auto from text` mode → parsed Arjuna cleanly and generated painting + skeleton.
  2. Two-figure scene (`"Rama and Lakshmana in the forest"`) with `preset` mode → rendered 2-figure skeleton and painting.
  3. Photo upload (`upload photo` mode with a JPEG photo of a person) → extracted skeleton via `pose/extract.py` and generated painting.
- **Bad inputs tested:**
  1. Empty prompt (`""`) → returned friendly UI message, no Python traceback.
  2. ~300-word prompt → handled gracefully without crashing.
  3. Photo with no person (landscape/empty room) → returned friendly UI warning, no Python traceback.
- **Upload temp file check:**
  - Inspected `/tmp/gradio` immediately after the photo upload request finished; the uploaded JPEG remained in `/tmp/gradio/<session_hash>/` instead of being deleted immediately by the handler (relying on Gradio's 360 s timer). Filed issue #158 (`needs:D`).

## 4. Re-test after #162 (added by B from C's comment on issue #158, 25 Nov 2026 04:57 UTC)
- Same JPEG, current `main` (after #162): `/tmp/gradio` had 0 files right after the request. Issue #158 closed by C. Not re-run by B.

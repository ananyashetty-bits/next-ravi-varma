# Handoff to Milestone 4

What M4 gets from M3, which pipeline to start from, and what we learned about where the remaining failures
are. Every number below comes from a file in `results/` or from `eval/EVAL_REPORT.md`.

## 1. Start from the default pipeline: c1 (raw text + LoRA)
```python
from pipeline.generate_api import text_to_image, generate, release

image, expanded, scene = text_to_image(
    "Hanuman kneeling before Rama and offering him Sita's ring",
    seed=1,
    mode="raw",        # the default pipeline (c1); see docs/TEAM_DECISIONS_M3.md D1
    model="rrv_v1",
)
```
- **What `mode="raw"` sends to the model:** `rrvarma style, <text>, oil painting, canvas texture, classical indian
  art`, with negative `ugly, blurry, bad anatomy`. The model is SD1.5 + `rrv_v1` at LoRA scale 1.0, DPM++ 2M,
  30 steps, CFG 7.0, 512×512, all from `config/m3.json`.
- **Why c1 and not an expanded prompt:**
  - Dev: c1 was more faithful than every expansion on human ratings (c1 → c3 −0.72 [−1.25, −0.25]; c1 → c2 −0.66;
    c1 → c4 −0.59).
  - Held-out: c1 → c3 −0.50 [−0.83, −0.17], and only c4 matched c1 (`eval/EVAL_REPORT.md` §2.1, §4).
  - The LoRA itself is what gives the style: c0 → c1 is +1.38 on dev and +1.54 on held-out.
- **Lower-level call:** `generate(positive, negative, seed, model=None, lora_scale=None, control_image=None)` takes
  your own prompts. `control_image` is a reserved slot for ControlNet and is ignored in M3.
- **`release()`** frees the GPU before another model (an LLM, a judge, ControlNet) is loaded.
- **Determinism:** each call builds its own seeded generator, so the same inputs give the same image on the same
  library versions.

## 2. Where the remaining failures are
Raters tagged one main failure per rated image (`results/human_ratings_dev.json`, `…_heldout.json` → `failures`):

| | Composition (wrong figure count + wrong action) | Identity and setting (wrong/missing identity + generic figure + wrong setting) |
|---|---|---|
| c1, dev | 25% | 50% |
| c1, held-out | 46% | 38% |
| Expanded conditions (c2-c6), dev | 46-53% | 38-52% |
| Expanded conditions (c2-c6), held-out | 42-54% | 33-50% |

- **Composition** (how many figures, who does what, where): longer prompts did not fix it. This is the natural
  place for **ControlNet** (OpenPose from a figure layout, or a sketch). SD1.5 works with ControlNet 1.x, and
  `generate(..., control_image=)` already has the slot.
- **Identity** (the model does not know who "Karna" or "Hanuman" is): two fixes were tried, and neither worked.
  - **Prompt-side:** KB descriptions did not help, and on held-out they hurt (c4 → c3 −0.50 [−0.88, −0.12]).
  - **Model-side:** a LoRA with names in its captions showed no gain (`train/NAMES_MODEL.md`).

  The remaining options are more and better-captioned training images per character, or reference-image
  conditioning (for example IP-Adapter).

## 3. Bridges already built
- **`layout` output:** `expander` mode `llm_kb_layout` emits validated JSON: `n_figures`, and for each figure a
  role, a pose from a fixed 8-word vocabulary and a position from 5. It was not evaluated in M3 (c5 was cut), but
  it is the obvious input for an OpenPose skeleton.
- **Parser and KB:** `kb/parser.py` with `kb/kb.json` has precision 1.00 everywhere. Recall is 1.00 on dev but 0.42
  on held-out, because 9 of the 11 missed characters have no KB entry (`eval/CLAIMS_AUDIT.md` §8). Coverage, not
  the parser, is the limit.
- **Evaluation kit:** checklist judge, scoring, blind contact sheets, rating stats and grids (`eval/`,
  `eval/RUNBOOK_B.md`). The judge agrees with human faithfulness at ρ 0.41 (dev) and 0.65 (held-out), so it is
  usable for comparing conditions, but not for judging single images.

## 4. Known issues to fix early in M4
- **`text_to_image()` defaults to `mode="raw"`** and uses C's real parser for the KB modes (fixed 18 Nov). An LLM
  mode loads and frees a 4B LLM on each call, so for many scenes use `expander/batch_expand.py` +
  `pipeline/run_batch.py` (how every evaluated image was made).
- **Fixed settings:** 512×512 only; one base model; LoRA weights at a Google Drive path set in `config/m3.json`.
- **Negative prompt:** M3 used `ugly, blurry, bad anatomy` throughout. The M2 kit's longer negative is a reasonable
  M4 experiment, but it is untested here.
- **Not completed in M3:** the SHAP attribution and the token-budget experiment have no scored results
  (`eval/CLAIMS_AUDIT.md` §5). `max_prompt_tokens = 70` is a design choice, not a measured optimum.

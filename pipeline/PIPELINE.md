# Pipeline: generation API, batch runner, CLI

How images are generated in M3, with the settings, commands and measured timings. Setup and a first image are in
the repo `README.md`. The copy-paste Colab sessions are in `expander/RUNBOOK_COLAB.md` and `eval/RUNBOOK_B.md`.

## 1. Architecture
```
user text ──► parse_scene (kb/parser.py) ──► SceneSpec ──► expand (expander/expand.py) ──► positive/negative ──► generate ──► image
```
- **Generation** (`pipeline/generate_api.py`) wraps the unmodified M2 kit (`m2kit/generate.py`: `load`,
  `swap_lora`, `set_scale`). It calls the diffusers pipeline directly only because the kit's `gen()` hardcodes its
  negative prompt, and M3 takes the negative from config.
- **Models:** base Stable Diffusion 1.5 (`stable-diffusion-v1-5/stable-diffusion-v1-5`), plus the M2 LoRA `rrv_v1`
  (M2 winner `D_lr2e-4`, step 800, sha256 `067ee19f…`), or `base` for no LoRA.
- **GPU sessions:** the LLM (expander), the diffusion model and the judge each need the T4 to themselves, so they
  run in separate Colab sessions and exchange JSON files on Drive. `release()` frees the diffusion model in-process.

## 2. Settings (`config/m3.json`, the single source)
| Setting | Value |
|---|---|
| sampler | DPM++ 2M (`dpmpp_2m`), 30 steps, CFG 7.0 |
| size | 512×512 |
| trigger / style tail | `rrvarma style` / `oil painting, canvas texture, classical indian art` |
| negative | `ugly, blurry, bad anatomy` (CHANGELOG 1.3) |
| LoRA scale | 1.0 |
| `max_prompt_tokens` | 70 CLIP tokens, excluding start/end (expanded modes) |
| dev / held-out seeds | 1-2 / 1-2 (held-out cut from 1-4) |
| safety checker | off: it false-blocks classical paintings (an M2 finding) |

## 3. Python API
```python
from pipeline.generate_api import generate, text_to_image, release, CFG

img = generate(positive, negative, seed, model=None, lora_scale=None, control_image=None)
img, expanded, scene = text_to_image(text, seed, mode="raw", model=None)
release()
```
- **`generate`:** loads the base model and LoRA once and swaps when `model` changes. `model=None` means
  `CFG["default_model"]` (`rrv_v1`), and `"base"` means no LoRA. Every call uses a fresh
  `torch.Generator("cuda").manual_seed(seed)`, so results do not depend on call order. `control_image` is
  reserved for M4.
- **`text_to_image`:** parse → expand → `close()` the LLM if one was loaded → generate. Modes: `raw` (c1, the
  default pipeline), `template` (c2), `llm_kb` (c3), `llm_nokb` (c4), `llm_kb_layout` (c5). Parsing uses C's `kb/parser.py`
  with `kb/kb.json`; the default mode is `raw`.

## 4. Batch runner (`pipeline/run_batch.py`)
```bash
python pipeline/run_batch.py --raw <scenes.json> --condition c1_lora_raw [--model base] --seeds 1 2 --out <dir>
python pipeline/run_batch.py --expansions <expansions.json> [--model rrv_v1] --seeds 1 2 --out <dir>
```
- **Output:** one PNG per (scene, seed) plus `manifest.json`. The manifest has a header with `git_sha`,
  `config_hash`, `model_sha256`, library versions and GPU, then one row per image with scene, condition, model,
  seed, prompt, token count, path and `gen_seconds`.
- **Resumable:** images that already exist are skipped, and the manifest is rewritten atomically after each one.
- **Raw mode** builds `trigger, text, tail` exactly like `expand(..., "raw")`. Expansion mode generates from each
  record's `positive` and `negative`.

## 5. Measured timings (Colab T4)
| Step | Time | Source |
|---|---|---|
| Generation, per 512×512 image, model loaded | 4.4 s base (c0); 5.0-5.4 s with the LoRA (c4 dev 4.99, c1 dev 5.28, c6 held-out 5.35) | `outputs/eval/c1_lora_raw_metrics.json`, `c4_llm_nokb_metrics.json`; held-out manifests (`gen_seconds`, first image excluded) |
| Diffusion model load (included in the first image's `gen_seconds`) | 53 s for the first run in a session (c0, held-out); about 9 s once cached (c6, held-out); add the download time on a brand-new runtime | held-out manifests, 16 Nov |
| LLM load (expander) | 23-33 s cached; 114-194 s fresh | `expander/EXPANDER.md` §12 |
| LLM expansion per scene | 9.7 s (c3), 13.3 s (c4) | `expander/EXPANDER.md` §12 |
| Judge per question | 0.87 s | `eval/JUDGE_NOTES.md` |

A full condition (24 scenes × 2 seeds) takes about 4-5 minutes, including loading.

## 6. CLI (`pipeline/cli.py`)
| Subcommand | Status |
|---|---|
| `generate "<text>" --seed 1 --out x.png [--mode raw] [--model rrv_v1]` | `text_to_image` in-process; prints the characters, prompt, token count and seconds |
| `batch --raw <scenes> [--condition c1_lora_raw] / --expansions <file> --seeds 1 2 --out <dir>` | runs `pipeline/run_batch.py` |
| `expand --scenes <file> --mode <mode> --out <file> …` | runs `expander/batch_expand.py` with the same options |
| `evaluate --manifest <dir>/manifest.json --out <results file> …` | runs `eval/score.py` with the same options |

All subcommands work from any directory. `pipeline/eval.py` (CLIP/hash metrics from early runs) is no longer used.

## 7. Reproducing an evaluated image
Run the same scene, condition and seed with `run_batch.py`. The manifest records the git sha and config hash of
every evaluated run. The held-out run used `64b7a2f` with config hash `1ced5c5c04bb`. Pixels can differ slightly
across torch/diffusers versions, but the composition should match.

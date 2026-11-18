# The Next Ravi Varma: Milestone 3

Turn a plain-language scene from the Indian epics ("Hanuman kneeling before Rama, offering Sita's ring") into an
image in Raja Ravi Varma's style. M3 adds the text side to M2's style model (Stable Diffusion 1.5 + the `rrv_v1`
LoRA): a scene parser and knowledge base, a prompt expander, and an evaluation of whether any of it helps.

**Result in one line:** on both the dev and the held-out set, raw text with the LoRA (**c1**) was at least as
faithful to the scene as every prompt-expansion pipeline. c1 is the default pipeline (`docs/TEAM_DECISIONS_M3.md`;
evidence in `eval/EVAL_REPORT.md`).

## What runs where
Everything runs on a **free Colab T4**. The LLM, the diffusion model and the judge each need the GPU to themselves,
so they run in **separate Colab sessions** and exchange files on Google Drive.

| Profile | Install | Used for |
|---|---|---|
| `gen` | `bash pipeline/setup_session.sh gen` | image generation (`pipeline/`) |
| `llm` | `pip install -r expander/requirements_llm.txt` | LLM prompt expansion (`expander/`) |
| `eval` | `pip install -r eval/requirements_eval.txt` | judge + metrics (`eval/`) |
| tests | `pip install pytest jsonschema rapidfuzz pyyaml numpy pillow` | `pytest -q tests` (CPU) |

**Drive prerequisites** (paths are set in `config/m3.json`):
- The M2 LoRA must be at `/content/drive/MyDrive/m2/hub/rrvarma_lora_v1/pytorch_lora_weights.safetensors`
  (sha256 `067ee19f…`).
- The base model `stable-diffusion-v1-5/stable-diffusion-v1-5` and the LLM `Qwen/Qwen3-4B-Instruct-2507`
  download from Hugging Face on first use.

## Quick start: one image with the default pipeline (Colab, T4)
```python
from google.colab import drive; drive.mount('/content/drive')
import getpass, subprocess
tok = getpass.getpass("GitHub token (classic, 'repo' scope): ").strip()   # the repo is private
subprocess.run(["git", "clone", f"https://{tok}@github.com/ananyashetty-bits/next-ravi-varma-m3", "/content/m3"], check=True)
del tok
%cd /content/m3
!bash pipeline/setup_session.sh gen
```
```python
from pipeline.generate_api import text_to_image
image, expanded, scene = text_to_image("Sita sitting alone under a tree in the Ashoka grove, sorrowful", seed=1, mode="raw")
image.save("sita_s1.png"); print(expanded["positive"])
```
`mode="raw"` is the default pipeline (c1): `rrvarma style, <your text>, oil painting, canvas texture, classical indian art`
on SD1.5 + `rrv_v1`, DPM++ 2M, 30 steps, CFG 7.0, 512×512, negative `ugly, blurry, bad anatomy`. The same seed
gives the same image (a fresh generator per call). The first call loads the model (about 1 minute in a fresh
session); after that, each image takes about 5 s on a T4 (`pipeline/PIPELINE.md` §5).

**Same thing from the command line:**
```bash
python pipeline/cli.py generate "Sita sitting alone under a tree in the Ashoka grove, sorrowful" --mode raw --seed 1 --out sita_s1.png
```
**Interactive demo:** `pipeline/demo.ipynb`.

## Other pipelines (the M3 ablation)
| `mode` | Condition | What the model sees |
|---|---|---|
| `raw` | c1 (**default**) | trigger + raw text + style tail |
| `template` | c2 | deterministic template from the parsed scene |
| `llm_kb` | c3 | LLM expansion with KB character descriptions |
| `llm_nokb` | c4 | LLM expansion without the KB |
| `llm_kb_layout` | c5 | c3 + layout phrases (built, not evaluated) |
| model `base` + `raw` | c0 | the same, without the LoRA |

**Known limitation:** `text_to_image()` still imports the stub parser (`kb/stub_parser.py`). For the KB modes,
generate the way the evaluated runs did instead: expand with `expander/batch_expand.py` (C's parser and KB), then
generate with `pipeline/run_batch.py --expansions …`. That route is shown below and in the runbooks.

## Batch runs (how all evaluated images were made)
```bash
# expansions (llm session)
python expander/batch_expand.py --scenes eval/test_scenes_dev.json --split dev --mode llm_kb --kb kb/kb.json \
    --condition c3_llm_kb_ckb --out outputs/expansions/c3_llm_kb_ckb_dev.json
python expander/check_expansions.py outputs/expansions/c3_llm_kb_ckb_dev.json --strict
# images (gen session): raw text, or from an expansion file
python pipeline/run_batch.py --raw eval/test_scenes_dev.json --condition c1_lora_raw --seeds 1 2 --out <dir>
python pipeline/run_batch.py --expansions outputs/expansions/c3_llm_kb_ckb_dev.json --seeds 1 2 --out <dir>
# scoring (eval session)
python eval/score.py --manifest <dir>/manifest.json --scenes eval/test_scenes_dev.json --out results/<cond>@<model>_dev.json
```
`run_batch.py` resumes after a disconnect and writes a `manifest.json` that records the git sha, the config hash,
the LoRA sha256, library versions and per-image seconds.

## Runbooks (copy-paste Colab cells)
- `expander/RUNBOOK_COLAB.md`: LLM session, expansions, frozen held-out commands.
- `eval/RUNBOOK_B.md`: scoring, judge, blind contact sheets, rating stats, the held-out run (§7).

## Repository map
| Path | What |
|---|---|
| `config/m3.json` | the single source of settings (model, sampler, trigger, tail, negative, token budget, seeds) |
| `pipeline/` | `generate_api.py` (`generate`, `text_to_image`), `run_batch.py`, `cli.py`, `demo.ipynb`, `PIPELINE.md` |
| `kb/` | parser, knowledge base (`kb.json`, `kb_llm.json`, `names.json`), `KB.md` |
| `expander/` | prompt expander, prompts, interim KB, `EXPANDER.md` |
| `train/` | names-aware LoRA experiment, `NAMES_MODEL.md` |
| `eval/` | test scenes (dev + held-out), judge, scoring, ratings, stats, grids, `EVAL_REPORT.md`, `CLAIMS_AUDIT.md` |
| `results/` | scored results per condition and split; human-rating summaries; `rated/` = the exact files paired with ratings |
| `outputs/expansions/` | every expanded prompt that was generated |
| `experiments/HYPOTHESES.md` | pre-registered hypotheses and dated deviations |
| `docs/` | `M3_REPORT.md`, `TEAM_DECISIONS_M3.md`, `HANDOFF_M4.md`, figures |
| `m2kit/` | M2 code, unchanged |

## Cold-start check (fresh runtime, README only)
1. Run the Quick start in a new Colab runtime. You should get an image and the expanded prompt.
2. Run the CLI line. It should save a PNG and print the timing.
3. Reproduce an evaluated image: `python pipeline/run_batch.py --raw eval/test_scenes_heldout.json --condition c1_lora_raw --seeds 1 --out /content/check`,
   then compare `h01_s1.png` with `M3/heldout_images/c1_lora_raw@rrv_v1/heldout/h01_s1.png` on the evaluator's Drive.
   The pixels can differ slightly across torch versions, but the composition should match.
4. `pytest -q tests` on CPU should pass.

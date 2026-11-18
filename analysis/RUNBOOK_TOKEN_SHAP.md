# Runbook: token budget and SHAP (Colab T4)

Four GPU sessions, in this order:

| Part | Session | Images | Time |
|---|---|---|---|
| 1a | Token budget: generate | 112 | ~12 min |
| 1b | Token budget: score | 112 | ~12 min |
| 2a | SHAP: generate | 512 | ~50 min |
| 2b | SHAP: score | 512 | ~45 min |

- **Resuming:** every step resumes. If Colab disconnects, run the same cells again; finished images and judge
  answers are kept.
- **Where things go:** everything is written to your Drive under `MyDrive/m3/a_experiments/`. Nothing in this
  runbook pushes to GitHub.
- **Inputs:** the prompts are already committed: `outputs/expansions/token_budget_dev.json` (56 prompts) and
  `outputs/expansions/shap_dev.json` (256 prompts). No LLM session is needed.

## Cell 1: setup (first cell of every session)
Runtime → Change runtime type → **T4 GPU**. Paste your classic `ghp_` token (`repo` scope) when asked.
```python
from google.colab import drive; drive.mount('/content/drive')
import getpass, os
tok = getpass.getpass("Paste GitHub token: ")
!rm -rf /content/m3
!git clone -q https://{tok}@github.com/ananyashetty-bits/next-ravi-varma-m3 /content/m3
del tok
%cd /content/m3
assert os.path.exists("pipeline/token_budget.py"), "clone failed or old main: check the token / that the PR is merged"
M3 = "/content/drive/MyDrive/m3"
OUT = f"{M3}/a_experiments"; os.makedirs(f"{OUT}/results", exist_ok=True)
!git rev-parse --short HEAD
```

## Cell 2: install (depends on the session)
**Generation sessions (1a, 2a):**
```python
!bash pipeline/setup_session.sh gen
```
**Scoring sessions (1b, 2b):**
```python
!pip install -q -r eval/requirements_eval.txt
```

## Part 1a: token budget, generate (generation session)
Cell 1, then the generation install, then:
```python
!python pipeline/run_batch.py --expansions outputs/expansions/token_budget_dev.json --seeds 1 2 --out {OUT}/token_budget@rrv_v1/dev
import json; m = json.load(open(f"{OUT}/token_budget@rrv_v1/dev/manifest.json"))
print(len(m["images"]), "images (expect 112); max tokens:", max(i["token_count"] for i in m["images"]), "(expect 75)")
```
Then Runtime → **Disconnect and delete runtime**.

## Part 1b: token budget, score (scoring session)
Cell 1, then the scoring install, then:
```python
!python eval/score.py --manifest {OUT}/token_budget@rrv_v1/dev/manifest.json --out {OUT}/results/token_budget@rrv_v1_dev.json
!python pipeline/token_budget.py analyse --scores {OUT}/results/token_budget@rrv_v1_dev.json --out {OUT}/results/token_budget_summary_dev.json
```
- **Expected:** `112 images to score`, then a table with 7 variants and 5 contrasts.
- **If it worked:** send a message saying "1b done". The results are written up from those two files.
- **If it failed:** paste the last lines of the output.

## Part 2a: SHAP, generate (generation session, ~50 min)
Cell 1, then the generation install, then:
```python
!python pipeline/run_batch.py --expansions outputs/expansions/shap_dev.json --seeds 1 2 --out {OUT}/shap@rrv_v1/dev
import json; m = json.load(open(f"{OUT}/shap@rrv_v1/dev/manifest.json"))
print(len(m["images"]), "images (expect 512)")
```
- **If it disconnects:** run Cell 1, the install and this cell again. It skips the images that already exist.
- **Before scoring:** only move on to 2b when it prints 512.

## Part 2b: SHAP, score and analyse (scoring session, ~45 min)
Cell 1, then the scoring install, then:
```python
!python eval/score.py --manifest {OUT}/shap@rrv_v1/dev/manifest.json --out {OUT}/results/shap@rrv_v1_dev.json
!python analysis/shap/shapley.py --scores {OUT}/results/shap@rrv_v1_dev.json --out {OUT}/results/shap_values_dev.json
```
- **Expected:** `512 images to score`, then `efficiency check passed` for each metric and a value-per-token table.
- **If it worked:** send a message saying "2b done". The chart, SHAP_ANALYSIS.md and the commit are done from the
  result files.

## What gets committed afterwards (not from Colab)
`results/token_budget@rrv_v1_dev.json`, `results/token_budget_summary_dev.json`, `results/shap@rrv_v1_dev.json`,
`analysis/shap/shap_values_dev.json`, `docs/figures/shap_dev.png`, and the filled-in `pipeline/TOKEN_BUDGET.md` and
`analysis/shap/SHAP_ANALYSIS.md`. These are new files, so no original `results/` file is overwritten.

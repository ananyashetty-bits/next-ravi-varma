# Member B runbook: blind key, rating stats, judge robustness (Colab)

Everything here runs in one Colab notebook. Steps 1-4 are CPU only. Step 5 needs a **T4 GPU** runtime
(Runtime → Change runtime type → T4), so pick T4 from the start if you plan to run it.

Outputs that decode codes into conditions go to `MyDrive/B_private/` and **never into the repo** until every
rater has submitted (the scripts refuse to write decoded provisional files inside the repo).

## 0. One-time Drive setup (in the Drive web UI, 1 min)
- `Shared with me → AI PROJECT → dev_contact_sheets`: right-click → *Organize → Add shortcut* → put it in
  `My Drive/m3/`. It then appears in Colab as `/content/drive/MyDrive/m3/dev_contact_sheets`.
- `My Drive/m3/images` must be the shortcut to A's `images` folder (already there since 3 Nov).

## 1. Setup
```python
from google.colab import drive; drive.mount('/content/drive')
!git clone -q https://github.com/ananyashetty-bits/next-ravi-varma-m3 /content/m3
%cd /content/m3
!pip install -q -r eval/requirements_eval.txt
M3, PRIV = "/content/drive/MyDrive/m3", "/content/drive/MyDrive/B_private"
```

## 2. Rebuild the round-1 blind key (only if `B_private/blind_key_dev.json` is missing)
The key was not saved to Drive on 4 Nov. `eval/recover_key.py` rebuilds it by matching every tile on the
contact sheets to the original seed-1 images; it writes nothing unless every tile has a clear match.
```python
!python eval/recover_key.py --sheets {M3}/dev_contact_sheets \
    --template {M3}/dev_contact_sheets/rating_template.csv \
    --manifests {M3}/images/c0_base_raw/dev/manifest.json {M3}/images/c1_lora_raw/dev/manifest.json \
                {M3}/images/c4_llm_nokb/dev/manifest.json \
    --key {PRIV}/blind_key_dev.json
```
Expected: `matched 48 codes across 16 scenes`, worst best-match close to 0. Use the **original**
`rating_template.csv` from the sheets folder (it has the code order); not the live Google Sheet.

## 3. Pull the live rating sheet as CSV
```python
from google.colab import auth; auth.authenticate_user()
import google.auth, gspread, csv
gc = gspread.authorize(google.auth.default()[0])
rows = gc.open_by_key("1J_CnRLZDIRTrlRuGMxTF0EDx_edFATgxjdvsBEmQmLs").sheet1.get_all_values()
with open(f"{PRIV}/ratings_r1.csv", "w", newline="") as f: csv.writer(f).writerows(rows)
print(len(rows) - 1, "rows")
```

## 4. Rating stats (final dev run: 16 Nov)
Download both sheets as CSV (*File → Download → CSV*): the round-1 sheet (tab `rating_template`) and the round-2
sheet (`AI PROJECT/dev_contact_sheets_r2/rating_template`). Put them in `B_private` as `ratings_r1.csv` and
`ratings_r2.csv`.
```python
!python eval/ratings.py --round {PRIV}/ratings_r1.csv {PRIV}/blind_key_dev.json \
    --round {PRIV}/ratings_r2.csv {PRIV}/blind_key_dev_r2.json \
    --results "results/*_dev.json" "results/rated/*.json" --alias c3=c3_llm_kb_ckb --out {PRIV}/ratings_dev_final
```
Every rating is paired with the automated scores of the exact image that was rated (same condition and file
name). Files under `results/rated/` replace the main file for their condition. That is how c4's rated images
(A's 2 Nov set) stay paired after D regenerated c4 in #72. The output also applies the B.3 default-pipeline rule.
The 16 Nov output is committed as `results/human_ratings_dev.json`.

## 5. Judge robustness check (T4 GPU, ~3 min incl. model load)
```python
!python eval/judge_robustness.py --results results/c0_base_raw@base_dev.json \
    results/c1_lora_raw@rrv_v1_dev.json results/c4_llm_nokb@rrv_v1_dev.json \
    --n 20 --paraphrase --out {M3}/judge_robustness_dev.json
```
Pre-registered pass: mean |change| < 0.05 after shuffling question order. Because the judge asks one
question per forward pass, this is effectively a re-run determinism check. `--paraphrase` is a descriptive
extra (not pre-registered): same questions, reworded answer instruction. The output file has no ratings in
it, so it can be committed (`outputs/eval/judge_robustness_dev.json`).

## 6. Round 2 contact sheets (when D's and C's folders are shared)
**Drive first (1 min):** create the folder `My Drive/m3/round2_images`. Then, for each shared folder
(D's `c2_template@rrv_v1`, C's `c3_llm_kb_ckb@rrv_v1` and `c6_llm_kbllm@rrv_v1`, or their `dev` subfolders),
right-click it → *Organize → Add shortcut* → `My Drive/m3/round2_images`.

**Find the manifests and check every seed-1 image is reachable:**
```python
import glob, os
from eval.score import load_items
R2 = f"{M3}/round2_images"
WANT = ["c2_template", "c3_llm_kb_ckb", "c6_llm_kbllm"]
mans = {}
for p in glob.glob(f"{R2}/**/manifest.json", recursive=True):
    for c in WANT:
        if f"{c}@" in p or f"/{c}/" in p:
            mans[c] = p
for c in WANT:
    if c not in mans:
        print("MISSING:", c, "- add its shortcut to", R2); continue
    _, items = load_items(mans[c])
    s1 = [i for i in items if i.get("seed") == 1]
    print(f"{c}: {len(s1)} seed-1 images, {sum(os.path.exists(i['path']) for i in s1)} found  ({mans[c]})")
```
Expected: c2 16/16, c3_llm_kb_ckb 24/24, c6 24/24. Don't go on until every count matches.

**Build the sheets** (`--rng 2` gives round 2 its own codes; the rater rotation puts the same pair of raters
on each of the 16 old scenes as in round 1):
```python
OUT2 = f"{M3}/dev_contact_sheets_r2"
!python eval/contact_sheet.py --manifests {mans['c2_template']} {mans['c3_llm_kb_ckb']} {mans['c6_llm_kbllm']} \
    --split dev --out {OUT2} --key {PRIV}/blind_key_dev_r2.json --rng 2
```
Expected: `24 sheets, 128 rating rows` (16 scenes × 3 codes + 8 scenes × 2 codes, two raters each). The key
goes to `B_private`. **Check it exists there before closing Colab**; round 1's key was lost this way.

**Add a `round2` tab to the rating sheet:**
```python
import csv
from google.colab import auth; auth.authenticate_user()
import google.auth, gspread
gc = gspread.authorize(google.auth.default()[0])
rows = list(csv.reader(open(f"{OUT2}/rating_template.csv")))
ws = gc.open_by_key("1J_CnRLZDIRTrlRuGMxTF0EDx_edFATgxjdvsBEmQmLs").add_worksheet("round2", rows=len(rows) + 5, cols=8)
ws.update(range_name="A1", values=rows)
print(len(rows) - 1, "rows in tab round2;", os.path.exists(f"{PRIV}/blind_key_dev_r2.json") and "key saved")
```
Then, in Drive, move `dev_contact_sheets_r2` into `AI PROJECT` (next to `dev_contact_sheets`) so the team
can open the sheets, and add a section B deviation line recording round 2's composition.

## 7. Held-out run (seeds 1-2; frozen pipeline)
Conditions: c0, c1, c2, `c3_llm_kb_ckb`, c4, c6. That is 12 scenes × 6 × 2 seeds = **144 images**. Each condition
is made exactly the way its rated dev version was made:

| Condition | How (same as dev) |
|---|---|
| c0, c1 | `run_batch.py --raw`, config negative `ugly, blurry, bad anatomy` (#79) |
| c2 | template `tpl2`, D's interim KB `d-kb0`, `--parser shim` (D's frozen command, `EXPANDER.md` §11) |
| c3_llm_kb_ckb | LLM `v1` with **C's KB as of 10 Nov** (`kb0.2`, 21 characters, commit `1b96b6c`), the version the rated c3 used; not today's kb0.4 |
| c4 | LLM `v1`, no KB, `--parser shim` (D's frozen command) |
| c6 | LLM `v1` with `kb/kb_llm.json` (`kb0.2-llm`, unchanged since 10 Nov) |

Everything is written to `B_private` or your own Drive until the reveal. **Never** commit the held-out cache, and never
run these from a session that pushes to the repo.

### 7a. Session 1, LLM (T4): expansions (~15 min)
```python
from google.colab import drive; drive.mount('/content/drive')
import getpass, os
tok = getpass.getpass("Paste GitHub token: ")
!rm -rf /content/m3
!git clone https://{tok}@github.com/ananyashetty-bits/next-ravi-varma-m3 /content/m3
del tok
%cd /content/m3
!pip install -q -r expander/requirements_llm.txt
M3, PRIV = "/content/drive/MyDrive/m3", "/content/drive/MyDrive/B_private"
H, EXP = f"{PRIV}/test_scenes_heldout.json", f"{PRIV}/expansions_heldout"
import shutil; os.makedirs(EXP, exist_ok=True)
shutil.copy(f"{PRIV}/PRIVATE_test_scenes_heldout (1).json", H)      # clean file name
!git show 1b96b6c:kb/kb.json > {PRIV}/kb0.2.json                  # C's KB as used for the rated c3
!git rev-parse --short HEAD
```
```python
C = f"--scenes {H} --split heldout --cache {PRIV}/heldout_cache.json"
!python expander/batch_expand.py {C} --mode template --kb expander/interim_kb/kb_d0.json --parser shim --condition c2_template   --out {EXP}/c2_template_heldout.json
!python expander/batch_expand.py {C} --mode llm_kb   --kb {PRIV}/kb0.2.json                  --condition c3_llm_kb_ckb --out {EXP}/c3_llm_kb_ckb_heldout.json
!python expander/batch_expand.py {C} --mode llm_nokb --parser shim                            --condition c4_llm_nokb   --out {EXP}/c4_llm_nokb_heldout.json
!python expander/batch_expand.py {C} --mode llm_kb   --kb kb/kb_llm.json                      --condition c6_llm_kbllm  --out {EXP}/c6_llm_kbllm_heldout.json
!python expander/check_expansions.py {EXP}/c2_template_heldout.json {EXP}/c3_llm_kb_ckb_heldout.json {EXP}/c4_llm_nokb_heldout.json {EXP}/c6_llm_kbllm_heldout.json --strict
```
Expected: 12 records per file and `check_expansions` passes (trigger first, ≤ 70 tokens, no leaked names). Then
**Runtime → Disconnect and delete runtime** (the LLM and Stable Diffusion don't fit together).

### 7b. Session 2, generation (T4): images (~25 min)
Same first cell as 7a (mount, clone, `%cd`, variables), but install with `!bash pipeline/setup_session.sh gen` instead
of the LLM requirements.
```python
IMG = f"{M3}/heldout_images"
!python pipeline/run_batch.py --raw {H} --condition c0_base_raw --seeds 1 2 --out {IMG}/c0_base_raw@base/heldout
!python pipeline/run_batch.py --raw {H} --condition c1_lora_raw --seeds 1 2 --out {IMG}/c1_lora_raw@rrv_v1/heldout
for c in ["c2_template", "c3_llm_kb_ckb", "c4_llm_nokb", "c6_llm_kbllm"]:
    !python pipeline/run_batch.py --expansions {EXP}/{c}_heldout.json --seeds 1 2 --out {IMG}/{c}@rrv_v1/heldout
```
Expected: 24 images per folder (144 in total). It resumes if Colab disconnects: just run the cell again. Then
disconnect and delete the runtime.

### 7c. Session 3, scoring (T4): judge + metrics (~25 min)
Same first cell, install with `!pip install -q -r eval/requirements_eval.txt`.
```python
RES = f"{PRIV}/results_heldout"; os.makedirs(RES, exist_ok=True)
for c, m in [("c0_base_raw", "base"), ("c1_lora_raw", "rrv_v1"), ("c2_template", "rrv_v1"),
             ("c3_llm_kb_ckb", "rrv_v1"), ("c4_llm_nokb", "rrv_v1"), ("c6_llm_kbllm", "rrv_v1")]:
    !python eval/score.py --manifest {IMG}/{c}@{m}/heldout/manifest.json --scenes {H} --out {RES}/{c}@{m}_heldout.json
!python eval/check_results.py {RES}/*_heldout.json --scenes {H} --seeds 1 2
```

### 7d. Contact sheets and rating sheet (CPU, ~5 min)
```python
mans = " ".join(f"{IMG}/{c}@{m}/heldout/manifest.json" for c, m in [("c0_base_raw", "base"), ("c1_lora_raw", "rrv_v1"),
        ("c2_template", "rrv_v1"), ("c3_llm_kb_ckb", "rrv_v1"), ("c4_llm_nokb", "rrv_v1"), ("c6_llm_kbllm", "rrv_v1")])
!python eval/contact_sheet.py --manifests {mans} --scenes {H} --split heldout --out {M3}/heldout_contact_sheets \
    --key {PRIV}/blind_key_heldout.json --rng 3
print("key saved:", os.path.exists(f"{PRIV}/blind_key_heldout.json"))
```
Expected: `12 sheets, 144 rating rows`, then `key saved: True`. Then, in Drive:
1. Move `heldout_contact_sheets` into `AI PROJECT`.
2. Open its `rating_template.csv` with Google Sheets. That makes the rating sheet, and the team rates there, as in round 2.

After the ratings come in, the stats are run with `ratings.py --scenes {H}` and these results, and the reveal
commits the held-out scenes, keys, ratings and results.

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

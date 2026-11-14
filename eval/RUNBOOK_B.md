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

## 4. Rating stats (provisional until A and C have rated)
```python
!python eval/ratings.py --round {PRIV}/ratings_r1.csv {PRIV}/blind_key_dev.json \
    --results "results/*_dev.json" --out {PRIV}/ratings_dev_provisional
```
Prints a **PROVISIONAL** banner and the sheet problems (wrong codes, non-standard tags). Keep the output to
yourself until everyone has rated. When round 2 exists, add `--round <r2 csv> <r2 key>`; if both
`c3_llm_kb` and `c3_llm_kb_ckb` are rated, say which one is "c3" with `--alias c3=c3_llm_kb_ckb`.

What it computes (section B of `experiments/HYPOTHESES.md`): condition means with CIs and ranges; paired
comparisons from Overview §8 (tie band 0.25); faithfulness by tier, seen/unseen (H-B3) and KB coverage
(H-B4); failure-tag counts; inter-rater agreement (H-B5); judge validation against human faithfulness and
CLIP (H-B1) and `style_margin` against human style (H-B2).

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

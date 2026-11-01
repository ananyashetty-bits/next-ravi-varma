# Checklist judge: setup check (owner B)

**Model:** `Qwen/Qwen2-VL-2B-Instruct`, fp16, `eval/judge.py`. It sees one image and one yes/no question; score = P(yes) from the "Yes"/"No" token logits.

## T4 smoke test, 2026-11-01 (Colab, `transformers==5.21.0`, `torch==2.11.0+cu130`)
| Measure | Value |
|---|---|
| GPU memory after loading | 4.4 GB |
| Speed | 0.87 s per question (≈ 4-5 s per image with 5 questions) |
| First load | ~1 min download (3.8 GB) + weights |

Sanity check on 3 real Varma paintings from `m2/eval/eval_holdout` (P(yes)):

| Painting | woman? | horse? | outdoors? |
|---|---|---|---|
| varma_ahalya_a37 | 0.95 | 0.03 | 0.94 |
| varma_damayanti_a01 | 0.97 | 0.01 | 0.93 |
| varma_draupadi_vastraharan_b09 (court scene) | 0.95 | 0.04 | 0.19 |

All 9 answers match the paintings. On the 64×64 flat-red fixture image the judge is unsure (0.32-0.55); that image is far outside what it will see and only tests the plumbing.

`score.py` ran end to end on `fixtures/manifest.json` with the judge (`--no_m2`).

## Known
- `transformers` warns that `min_pixels`/`max_pixels` are deprecated (removed in v5.23). The pinned 5.21.0 is fine; update `judge.py` before upgrading.
- This is a setup check, not a validation. Whether the judge agrees with human ratings is H-B1 in `experiments/HYPOTHESES.md`, tested on Day 4.

## Rough cost on a T4
| Job | Images | Judge time |
|---|---|---|
| One dev condition (32 scenes × 2 seeds) | 64 | ~5 min |
| SHAP (8 scenes × 32 subsets × 2 seeds) | 512 | ~35-40 min |
| Held-out, one condition (12 × 4 seeds) | 48 | ~4 min |

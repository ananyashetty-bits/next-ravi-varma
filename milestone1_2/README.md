# Milestones 1 and 2 (imported 25 Nov 2026)

This folder holds the Milestone 1 (data) and Milestone 2 (LoRA training and evaluation) work, so the whole project
(M1 to M4) can be read in one repository. The M3 and M4 work is everything outside this folder.

**Source:** `https://github.com/kanuvyas/next-ravi-varma`, commit `50466059de08c8a51395ca7019bf7b6f82fd60e0`
(main, 29 Oct 2026). Files are copied unchanged, with the same folder layout, so the relative links inside the
original README (kept here as `README_original_m1_m2.md`) still work.

## What is here

| Folder | Contents |
|---|---|
| `docs/` | Project proposal and the team decision logs for M1 and M2 |
| `m1_data/` | M1 data work: provenance logs, caption instructions and captions, audit files |
| `m2_training/kit/` | The M2 kit: launch, sweep, evaluation and metric scripts, thresholds, evaluation prompts (the M3 pipeline's `m2kit/` is a copy of this) |
| `m2_eval/` | M2 evaluation tables and logs (text and CSV only) |
| `bundles/` | The four team members' M2 training bundles: configs, logs, sweep tables (text only) |
| `hub/rrvarma_lora_v1/` | The released M2 adapter (`pytorch_lora_weights.safetensors`, 6.4 MB, rank 16, trigger phrase `rrvarma style`), its model card, `EVAL_REPORT_M2.md`, `HANDOFF_M3.md`, inference notes and loss curve |

## What is not here (still in the original repository)

About 650 MB of images and weights were left out on purpose: the training paintings, the sample and comparison
grids, the sweep images, the validation samples, the checkpoints of the sweep runs and the per-person bundle
weights. This repository does not track images or weights (see `.gitignore`), apart from the final adapter and its
loss curve. Everything left out is in the original repository at the commit above.

## Where the M2 result is used later

The M3 pipeline loads this adapter as `rrv_v1` (`config/m3.json`; sha256 recorded there). M3 and M4 documents:
`docs/M3_REPORT.md`, `docs/M4_REPORT.md`, `eval/EVAL_REPORT_M4.md`.

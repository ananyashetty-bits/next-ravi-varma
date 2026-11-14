# NAMES_MODEL.md: the name-aware LoRA `C_names` (Member C)

**Status:** trained, guardrails and names test run (sections 5, 6, 8, 9); the 2x2 is in section 7; registration status in section 10.

## 1. Question
The M2 model ignores character names (`NAMES_TEST.md`) because the text encoder is frozen and no name-to-appearance association was ever trained. Can the UNet learn name -> appearance through the frozen CLIP name tokens if the names are in the captions? The only change against the M2 winner is the captions.

## 2. Coverage audit and GO/NO-GO
Pre-registered in `experiments/HYPOTHESES.md` (Section C, before the audit): GO if at least 60% of the training images have characters filled and at least 6 characters appear in 3 or more images.
Result: **99 of 147 images (67%)** have characters filled; **11 characters** appear in 3 or more images (`train/audit_coverage.py`, `train/coverage_table.json`). **GO.** The LoRA was nevertheless run late, after the Day-3 cutoff, because of the order the work was done in (deviation line in HYPOTHESES.md).

| Character | Training images |
|---|---|
| Sita | 10 |
| Krishna | 9 |
| Rama | 9 |
| Shakuntala | 8 |
| Draupadi | 6 |
| Lakshmana | 5 |
| Radha | 5 |
| Hanuman | 4 |
| Vishwamitra | 4 |
| Ganesha | 3 |
| Vishnu | 3 |

Most characters have a single image, which is the memorisation risk: with one painting per name, the model can learn "name = this exact painting".

## 3. Dataset `train_v3`
`train_v3` = the images of `train_v1` (same 147 files in the same order, asserted in `train/build_v3.py`) with names added to the captions as `rrvarma style, <Name>, <Name2>, <v1 caption body>`. **70 of 147 captions** got names (counted against `train_v1`); 74 distinct names; the trigger appears once at the start; the longest caption is 73 CLIP tokens (limit 75); no name had to be dropped.
Only epic and mythological characters present in an image's `scene_metadata` were named; historical portraits (rulers, nobles, a judge and others) and non-epic figures (e.g. Judith, Adi Shankaracharya, Gopichandra) were excluded. Spelling variants were merged through the alias map of `train/audit_coverage.py`.
sha256 of `metadata.jsonl`: `06bedbccf734074ac6c489c2304018a34fd212137a9f03a1eadd7740210d7e4d` (committed in `train/train_v3_sha256.txt`; the Drive copy matches). I spot-read 15 random captions against their scene descriptions: every name matched a character in the painting.
**Held-out check:** the image list is identical to `train_v1`, which was leak-checked in M2; only the captions changed. I did **not** re-run `m2kit/leak_check.py` on `train_v3`.

## 4. Training
Same recipe as the M2 winner `D_lr2e-4`, stopped at its step 800: rank 16, learning rate 2e-4, constant with 100 warmup steps, batch size 1 with 4 accumulation steps, seed 42, bf16, 800 steps. The learning-rate schedule is constant, so an 800-step run follows the winner's first 800 steps.
Run in four resumed sessions of 200 steps (checkpoints 200, 400, 600, 800 on Drive; each resume loaded model, optimizer, scheduler and sampler state), about 8.3-9.2 s per optimizer step on a T4. The 50-step smoke test of the guide was **skipped**; the first minutes of the full run served as the check (data found: 147 examples, losses finite).
Final launch command:
```
accelerate launch --mixed_precision="bf16" /content/diffusers/examples/text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --train_data_dir="/content/train_v3" --caption_column="text" \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 --gradient_accumulation_steps=4 --gradient_checkpointing \
  --dataloader_num_workers=2 --max_train_steps=800 \
  --learning_rate=2e-4 --lr_scheduler="constant_with_warmup" --lr_warmup_steps=100 \
  --rank=16 --seed=42 --checkpointing_steps=200 \
  --validation_prompt="rrvarma style, a woman in a silk sari standing in a palace hall" \
  --num_validation_images=4 --validation_epochs=8 --report_to="tensorboard" \
  --output_dir="/content/drive/MyDrive/m2/runs/C_names" --resume_from_checkpoint=latest
```
Weights: `m2/runs/C_names/pytorch_lora_weights.safetensors`, sha256 `4ea33e82ab134ec0efa3357e7863bd2a03a373823c80eda0f12918eba3c68f44` (the end-of-run file stores float32 values under the older `lora.down`/`lora.up` names; `m2/runs/C_names/checkpoint-800/pytorch_lora_weights.safetensors`, sha256 `aaf395f39db8d39e7469800b0239f6cd241b3c89a255eb41b2effee22252309a`, stores the same weights as bfloat16 under `lora_A`/`lora_B` names. Compared tensor by tensor after renaming, the maximum absolute difference over all 256 tensors is exactly 0. **The checkpoint-800 file is the one used for evaluation and registration**, because it has the same layout and precision as the registered M2 winner `rrv_v1`).
Training losses are noisy by nature (step losses between 0.002 and 0.45); no conclusion is drawn from them.

## 5. Guardrails against the M2 winner
`m2kit/eval_run.py --run_dir C_names --only_steps 800` (checkpoint-800; plain-SD1.5 baselines and base control images reused from the winner's evaluation). Pass lines (`thresholds.json`): style_bleed <= 0.059, holdout-loss rise <= 0.01, max_train_dino p95 <= 0.85. The formal check uses the step-800 *screen* rows (comparable to the winner's); the *deep* rows are reported alongside.

| Metric | M2 winner, screen | C_names, screen | M2 winner, deep | C_names, deep |
|---|---|---|---|---|
| style_margin | 0.128 | 0.111 | 0.123 | 0.115 |
| clip_text | 0.301 | 0.317 | 0.300 | 0.314 |
| max_train_dino_p95 | 0.819 | 0.816 | 0.846 | 0.826 |
| style_bleed | -0.005 | -0.001 | 0.011 | 0.020 |
| drift | 0.153 | 0.104 | 0.165 | 0.142 |
| holdout_loss | 0.20076 | 0.20081 | 0.20076 | 0.20081 |
| train_probe_loss | 0.188 | 0.188 | 0.188 | 0.188 |

Holdout-loss rise vs the winner: +0.00005. PASS style_bleed <= 0.059; PASS holdout rise <= 0.01; PASS max_train_dino p95 <= 0.85. **Overall: PASS, eligible to be registered as `rrv_names`.**

## 6. Names test
12 characters, name prompt (`rrvarma style, <Name> standing`) and description prompt (the KB visual), on `rrv_v1` and `C_names` (checkpoint-800), seeds 1-4, checklist score by `eval/score.py` (Qwen2-VL-2B judge, a proxy; no human ratings). Checklists were written from the KB descriptions, so description prompts have an advantage by construction; the meaningful comparison is the name prompt on `C_names` vs `rrv_v1`. Design and prediction fixed in HYPOTHESES.md before any `C_names` image existed.

| Group | rrv_v1 name prompt | C_names name prompt | Gain | rrv_v1 description | C_names description |
|---|---|---|---|---|---|
| 3 or more images | 0.600 | 0.595 | -0.005 | 0.714 | 0.721 |
| 1 image | 0.500 | 0.509 | +0.009 | 0.765 | 0.755 |
| 0 images (controls) | 0.369 | 0.364 | -0.006 | 0.799 | 0.709 |

Per character (mean checklist score over seeds 1-4):

| Scene | Character | Training images | v1 name | C_names name | Gain | v1 description | C_names description |
|---|---|---|---|---|---|---|---|
| n01 | Sita | 10 | 0.735 | 0.836 | +0.100 | 0.932 | 0.944 |
| n02 | Krishna | 9 | 0.550 | 0.450 | -0.099 | 0.610 | 0.629 |
| n03 | Shakuntala | 8 | 0.566 | 0.552 | -0.015 | 0.854 | 0.845 |
| n04 | Radha | 5 | 0.635 | 0.643 | +0.008 | 0.660 | 0.692 |
| n05 | Draupadi | 6 | 0.579 | 0.524 | -0.055 | 0.652 | 0.683 |
| n06 | Hanuman | 4 | 0.535 | 0.565 | +0.030 | 0.574 | 0.532 |
| n07 | Bhishma | 1 | 0.413 | 0.420 | +0.007 | 0.905 | 0.880 |
| n08 | Subhadra | 1 | 0.528 | 0.533 | +0.006 | 0.754 | 0.759 |
| n09 | Duryodhana | 1 | 0.340 | 0.336 | -0.004 | 0.577 | 0.570 |
| n10 | Savitri | 1 | 0.719 | 0.746 | +0.027 | 0.822 | 0.810 |
| n11 | Karna | 0 | 0.399 | 0.428 | +0.030 | 0.790 | 0.704 |
| n12 | Shabari | 0 | 0.340 | 0.299 | -0.041 | 0.808 | 0.714 |

Group means are descriptive (n = 6, 4 and 2 characters); no confidence intervals or significance claims.

## 7. 2x2 (model x prompt) on the dev scenes
Not run in the time available. `pipeline/run_batch.py` cannot run (`pipeline/generate_api.py` defines no `generate()`), so c1 was not produced for either model.

## 8. Memorisation
max_train_dino over the 48 name-prompt images: `C_names` p95 = 0.823 (`rrv_v1`, same prompts: 0.840); images at or above 0.85: 0 of 48. **Memorisation flag (p95 >= 0.85): no.** Memorisation sheet: `images/names_test/memorization_nt_name_C_names.png` on Drive. My reading of the sheet: The nine most similar name-prompt images (max_train_dino 0.79 to 0.84, all below the 0.85 flag) are new compositions, not copies of training paintings; six of the nine nearest training images are the same painting of a standing woman in a red sari, so the similarity reflects a shared generic look rather than a copied painting. Several generated figures are women in a red sari, and some have four arms (the training set contains four-armed deities); I did not determine whether this is a name side effect or an anatomy artefact.

## 9. Did it learn names, or memorise paintings?
Pre-registered H-C3 against the data: gain for characters with 3 or more images -0.005 (predicted >= 0.10: not met); 1 image +0.009 (predicted smaller than the 3+ group: not met); controls with no images -0.006 (predicted < 0.10: met). These data do not show that names help the characters with several images. **Straight answer:** by this test it neither learned names (no gain for any group) nor memorised paintings (no image reached the 0.85 flag, and p95 was lower than for `rrv_v1` on the same prompts). Per-character gains range from -0.10 (Krishna) to +0.10 (Sita) and are treated as noise at 4 seeds and 4 to 6 checklist items per character. Why the names did not help is not tested here; possible reasons (frozen text encoder, one image per name for most characters, 800 steps, captions that already describe each image in full) are untested ideas. One further observation, not interpreted: on the two 0-image controls the description prompts scored lower on `C_names` (0.709 vs 0.799, n = 2). This is weak evidence: 6, 4 and 2 characters, a 2B-model judge, checklists written from the KB, and no human ratings. My reading of the memorisation sheet: The nine most similar name-prompt images (max_train_dino 0.79 to 0.84, all below the 0.85 flag) are new compositions, not copies of training paintings; six of the nine nearest training images are the same painting of a standing woman in a red sari, so the similarity reflects a shared generic look rather than a copied painting. Several generated figures are women in a red sari, and some have four arms (the training set contains four-armed deities); I did not determine whether this is a name side effect or an anatomy artefact.

## 10. Registration
Passed the guardrails (section 5) but **not registered** as `rrv_names`: the names test shows no gain from the names (section 9), so registering it would change `config/m3.json` for no measured benefit. The weights stay at `m2/runs/C_names/checkpoint-800` on Drive.

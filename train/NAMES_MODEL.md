# NAMES_MODEL.md: the name-aware LoRA `C_names` (Member C)

**Status:** trained; **guardrails, names test, 2x2 runs and registration are PENDING** (no GPU quota at the time of writing). Nothing below claims that the model learned names. Sections marked PENDING are filled in when those runs exist.

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

## 5. Guardrails against the M2 winner (PENDING)
`m2kit/eval_run.py --run_dir C_names --only_steps 800`. Pass lines from `m2kit/thresholds.json`: style_bleed <= 0.059, holdout-loss rise <= 0.01, max_train_dino p95 <= 0.85.
| Metric | M2 winner, step 800 | C_names, step 800 |
|---|---|---|
| style_margin | 0.128 | PENDING |
| style_bleed | -0.005 | PENDING |
| drift | 0.153 | PENDING |
| max_train_dino p95 | 0.819 | PENDING |
| holdout loss | 0.20076 | PENDING |
`C_names` is eligible to be registered as `rrv_names` only if it passes all three lines (fixed in HYPOTHESES.md, 2026-11-12); otherwise it is reported as a trade-off.

## 6. Names test (design fixed before any `C_names` image was generated; results PENDING)
12 characters (6 with 3 or more training images, 4 with 1, 2 with none as controls): Bhishma, Draupadi, Duryodhana, Hanuman, Karna, Krishna, Radha, Savitri, Shabari, Shakuntala, Sita, Subhadra. Each as a **name prompt** (`rrvarma style, <Name> standing`) and a **description prompt** (`rrvarma style, <KB visual>, standing`), on `rrv_v1` and `C_names`, seeds 1-4, shared name-free checklists (`train/names_test_scenes.json`), scored with `eval/score.py`.
Hypothesis H-C3: on `C_names`, name prompts beat the same prompts on `rrv_v1` by a checklist gain of 0.10 or more for the 3-or-more-image group, by less for the 1-image group, and by less than 0.10 for the 0-image controls. A control that gains 0.10 or more is treated as a leak. Group means are descriptive (n = 6, 4, 2).
**Caveat:** the checklists were written from the KB descriptions, so description prompts have an advantage by construction; the meaningful comparison is the name prompt on `C_names` against the same prompt on `rrv_v1`.
| Group | rrv_v1 name prompt | C_names name prompt | Gain | C_names description prompt |
|---|---|---|---|---|
| 3 or more images | PENDING | PENDING | PENDING | PENDING |
| 1 image | PENDING | PENDING | PENDING | PENDING |
| 0 images (controls) | PENDING | PENDING | PENDING | PENDING |

## 7. 2x2 (model x prompt) on the dev scenes (PENDING)
Only if `rrv_names` is registered: c1 and c3 with `--model rrv_names` (B's table).

## 8. Memorisation sheet (PENDING)
Name-prompt images on `C_names` with max_train_dino p95 >= 0.85 are reported as memorisation of a training painting, not understanding. I will look at the sheet myself.

## 9. Did it learn names, or memorise paintings? (PENDING)
Not answered yet. It will be answered from sections 5, 6 and 8 only.

## 10. Registration (PENDING)
If the guardrails pass: `models.rrv_names` (path, sha256, scale from `m2kit/sweep.py` on the best checkpoint) in `config/m3.json` by PR with a CHANGELOG line and another member tagged. If not registered, the reason is stated here.

# M3 evaluation report (owner: B)

> **Status 2026-11-18: complete (§1-§6). Claims audit of the other documents: `eval/CLAIMS_AUDIT.md`.**
> Rules: every number names its source file. Human-rating numbers come only from `eval/ratings.py` run on the
> final sheet after every rater has submitted. Nothing here is typed from memory or from a chat.
> Markers: `TODO(B)` = B to fill; `TODO(x)` = needs input from member x.

## 1. Setup

### 1.1 Test scenes
**Dev set** (`eval/test_scenes_dev.json`): the **analysed dev set is 24 scenes**, 8 each from B (s09-s16), C (s17-s24)
and D (s25-s32). Every count, rating and comparison in this report refers to these 24 unless it says otherwise.
A's s01-s08 were added on 16 Nov (#77). That was after the dev ratings, the dev results and the default-pipeline
decision. The scenes were also rewritten to use characters the interim KB knows: the first version had Yudhishthira,
Ravana and Damayanti, which it doesn't. Because of the timing and that selection, s01-s08 are **not part of the
pre-registered dev analysis**. Any scores on them are reported separately as descriptive (section B deviation, 2026-11-16).

| | Count |
|---|---|
| Tier 1, one well-known character | 5 |
| Tier 2, two characters interacting | 6 |
| Tier 3, multi-figure | 7 |
| Tier 4, rare / absent-in-training character or spelling variant | 3 |
| Tier 5, stress (vague, non-epic, long, misspelt) | 3 |
| Ramayana / Mahabharata / Puranic / unknown epic | 11 / 10 / 1 / 2 |
| `seen_in_training` true / false | 5 / 19 |

Every contributor has at least one scene per tier, and the team-wide targets of ≥ 10 Ramayana and ≥ 10
Mahabharata scenes are met. Each scene has 5-6 yes/no checklist questions (126 in total; 85 with weight 1.0
and 41 with weight 0.5), typed as pose 27, setting 25, count 23, object 19, figure 16, attire 11, mood 5.
`tests/test_scenes_dev.py` enforces the checklist rules: 3-6 questions, no character names (checked against
the scene's characters and the 190 canonical names and aliases in `expander/names_seed.json`), and no
style questions. Checklists were written before any image was generated.

`seen_in_training` marks scenes whose episode is depicted by an M1 training image (M1 `scene_metadata.scene`).
With only 5 seen scenes, seen-versus-unseen results are descriptive (H-B3).

The **first 16 scenes** (B and D) are the ones every condition up to c4 was run on; C's 8 were added on
10 Nov. Their profile: tiers 1-5 = 3 / 4 / 5 / 2 / 2; 6 Ramayana, 8 Mahabharata; 5 seen, 11 unseen.

**Held-out set** (`eval/test_scenes_heldout.json`, revealed 18 Nov): **12 scenes** (h01-h12) written by B on
1 Nov. They were stored privately (outside the repo and the shared Drive) until after held-out scoring and rating,
and were never used for judge development or any pipeline choice.
- Tiers 1-5: 2 / 2 / 3 / 3 / 2. Epics: Ramayana 5, Mahabharata 4, Puranic 2, unknown 1.
- All 12 are unseen in training.
- 4-5 checklist questions each (49 in total). The scenes pass the same rule tests as the dev scenes.
- Only 7 of their 19 character mentions are in the KB that c3 used (`kb0.2`), so the held-out set mostly tests
  characters the system has no description for.

### 1.2 Conditions
All conditions use the same generator: Stable Diffusion 1.5, DPM++ 2M, 30 steps, CFG 7.0, 512×512
(`config/m3.json`). Everything except c0 adds `rrv_v1`, the M2 winner (`D_lr2e-4` step 800, sha256
`067ee19f…`), at LoRA scale 1.0. Trigger `rrvarma style` first, style tail `oil painting, canvas texture,
classical indian art` last, prompts fitted to 70 CLIP tokens.

| Condition | What the model sees | Scored dev scenes | Results file |
|---|---|---|---|
| `c0_base_raw` | base SD1.5, raw user text | 16 | `results/c0_base_raw@base_dev.json` |
| `c1_lora_raw` | LoRA, trigger + raw text + style tail | 16 | `results/c1_lora_raw@rrv_v1_dev.json` |
| `c2_template` | LoRA, deterministic template from the parsed scene (D's interim KB `d-kb0`) | 24 | `results/c2_template@rrv_v1_dev.json` |
| `c3_llm_kb` | LoRA, LLM expansion with D's interim KB `d-kb0` | 24 | `results/c3_llm_kb@rrv_v1_dev.json` |
| `c3_llm_kb_ckb` | LoRA, LLM expansion with C's image-grounded KB | 24 | `results/c3_llm_kb_ckb@rrv_v1_dev.json` |
| `c4_llm_nokb` | LoRA, LLM expansion without KB | 24 (rated: 16) | `results/c4_llm_nokb@rrv_v1_dev.json` (D, 15 Nov); rated images: `results/rated/c4_llm_nokb@rrv_v1_dev.json` (A, 2 Nov) |
| `c6_llm_kbllm` | as `c3_llm_kb_ckb`, but KB descriptions written by the LLM instead of grounded in paintings | 24 | `results/c6_llm_kbllm@rrv_v1_dev.json` |

The expander LLM is `Qwen/Qwen3-4B-Instruct-2507` (revision pinned in `config/m3.json`), 4-bit, temperature 0,
prompt `v1`. Each dev condition has seeds 1 and 2 (32 or 48 images). Section D treats only real-KB images as
evidence; for human ratings, "c3" means `c3_llm_kb_ckb` unless stated otherwise.
`@rrv_names` was not registered: C's names model passed the guardrails, but the names test showed no gain (H-C3 not supported), so no c1/c3 runs were made on it (`train/NAMES_MODEL.md`).
The optional conditions (`c5`, `c3_7b`, `c3_neg`, `c3_v2`) were not run (cut list; `expander/EXPANDER.md`).

### 1.3 Metrics
**Primary: blind human ratings** (§1.4): faithfulness, style fidelity and visual quality, each 1-5.

**Checklist judge** (`eval/judge.py`): `Qwen/Qwen2-VL-2B-Instruct`, fp16. For each checklist question it
sees only the image (native 512×512) and that one question, followed by "Answer with only Yes or No." The
answer probability is P(yes) = sigmoid(logsumexp of the Yes/yes logits − logsumexp of the No/no logits) at
the first generated token. An image's `checklist` score is the weight-averaged P(yes) over its scene's
questions. Answers are cached by (image sha256, model, question). Setup check (`eval/JUDGE_NOTES.md`):
4.4 GB on a T4, 0.87 s per question, 9/9 sensible answers on three held-out Varma paintings. Whether it
agrees with humans is H-B1 (§3).

**From the M2 kit** (`m2kit/metrics.py`, unchanged):
- `clip_raw`: cosine similarity between the image and the **raw user text** under `openai/clip-vit-large-patch14`.
  CLIP reads at most 77 tokens, so long scene texts (tier 5) are truncated. The generator's text encoder is
  also CLIP (ViT-L/14), which may favour prompts close to the raw text.
- `style_margin`: CLIP image similarity to the centroid of M2's held-out Varma paintings minus similarity to
  the centroid of the control (non-Varma) paintings. Higher means more Varma-like.
- `max_train_dino`: highest DINOv2-base similarity to any `train_v1` image (memorisation). Guardrail: p95 ≤ 0.85
  (`m2kit/thresholds.json`).

### 1.4 Human rating protocol
- **Contact sheets** (`eval/contact_sheet.py`): one PNG per scene, with the scene text on top and each
  condition's seed-1 image below, labelled only by a random two-character code. Images are shown at
  320×320 (downscaled from 512).
- **Blind key**: code → condition, kept privately by B until every rater has submitted. The round-1 key was
  not saved to Drive on 4 Nov. It is rebuilt with `eval/recover_key.py`, which matches each tile on the
  sheets to its source image, so no ratings or results are involved (section B deviation, 2026-11-14).
  On 14 Nov it matched all 48 codes across 16 scenes, each with a pixel difference of 0.00.
- **Scales and failure tags**: `eval/RATING_GUIDE.md`. One main failure per picture from a fixed list:
  wrong figure count · wrong/missing figure identity · wrong setting · wrong action · generic figure (name
  ignored) · weak style · anatomy/face artifacts · other.
- **Assignment**: two raters per image, rotated across A-D so the load is even. Raters rate alone and
  submit before any discussion. All four raters are project members who know the system.
- **Rounds**:
  - Round 1 (sheets 4 Nov): c0, c1, c4 on the first 16 scenes, 48 images, 96 rating rows.
  - Round 2 (sheets 14 Nov): c2, `c3_llm_kb_ckb`, c6 on the first 16 scenes, plus `c3_llm_kb_ckb` and c6 on s17-s24. That is 64
    images and 128 rating rows, with the same two raters per scene as in round 1. `c3_llm_kb` (D's interim KB) is not rated.
  - Held-out round (sheets 16 Nov, rated 17 Nov): all six conditions side by side on one sheet per scene. 12 scenes,
    72 images, 144 rating rows (36 per rater).

  Images rated in different rounds were never side by side on one sheet. Comparisons across rounds (such as
  c1 → c3) therefore rest on absolute 1-5 judgements, not on a direct side-by-side comparison.

### 1.5 Statistics
- **Unit of analysis**: the scene. Automated scores are averaged over seeds within a scene, and ratings over
  raters within an image (one rated image per scene and condition).
- **Paired comparisons** (`eval/stats.paired`): mean per-scene difference b − a over the scenes both
  conditions share, a paired bootstrap 95% CI (5000 resamples, seed 0), and win/tie/loss counts. Tie bands
  (section B.2): 0.25 rating points for ratings, 0.0295 for automated scores. No p-values. A CI that
  includes 0 means "no clear difference".
- **Agreement and validation** (`eval/ratings.py`): Spearman ρ with a bootstrap CI over images, plus
  quadratic-weighted κ for inter-rater agreement.
- **Pre-registration**: section B of `experiments/HYPOTHESES.md` was merged on 1 Nov (#23), before the first
  scored run on 4 Nov. Deviations are listed there as dated lines.

### 1.6 Hardware
Free Colab T4 for generation, scoring and the judge. Expander LLM, diffusion model and judge run in separate
sessions and exchange files. Generation takes ≈ 5 s per image (`outputs/eval/c1_lora_raw_metrics.json`
`avg_gen_seconds` 5.28; c4 5.0). Judge cost: ≈ 5 min per 64-image dev condition (`eval/JUDGE_NOTES.md`).

## 2. Results, dev

### 2.1 Human ratings (primary)
**Source.** Round 1 sheet (tab `rating_template`, 96 rows) and round 2 sheet (`dev_contact_sheets_r2/rating_template`,
128 rows), decoded with the two private blind keys. All 224 rows were usable after the 14-16 Nov cell fixes: 56 per rater,
112 images with two raters each. Produced by:
```
python eval/ratings.py --round ratings_r1.csv blind_key_dev.json --round ratings_r2.csv blind_key_dev_r2.json \
    --results "results/*_dev.json" "results/rated/*.json" --alias c3=c3_llm_kb_ckb
```
The full output is `results/human_ratings_dev.json` (aggregates only; raw sheets and keys are revealed with the
held-out set). Each rating is paired with the judge score of **the exact image that was rated**, matched by file
name. For c4, those are A's 2 Nov images (`results/rated/c4_llm_nokb@rrv_v1_dev.json`). D's 15 Nov regeneration
(#72) has different pixels and is used only in §2.2.

##### Human ratings by condition (mean over scenes [95% CI]; range)

| Condition | n scenes | Faithfulness | Style | Quality |
|---|---|---|---|---|
| `c0_base_raw` | 16 | 2.00 [1.59, 2.44]; 1.0–4.0 | 1.59 [1.34, 1.91]; 1.0–3.0 | 2.09 [1.88, 2.31]; 1.5–3.0 |
| `c1_lora_raw` | 16 | 2.56 [2.19, 2.97]; 1.5–4.5 | 2.97 [2.75, 3.19]; 2.0–4.0 | 2.47 [2.25, 2.69]; 2.0–3.5 |
| `c2_template` | 16 | 1.91 [1.59, 2.31]; 1.0–4.0 | 2.47 [2.22, 2.72]; 1.5–3.5 | 2.09 [1.78, 2.41]; 1.0–3.5 |
| `c3_llm_kb_ckb` | 24 | 1.94 [1.67, 2.21]; 1.0–3.0 | 2.71 [2.48, 2.92]; 1.5–3.5 | 2.56 [2.35, 2.79]; 1.5–4.0 |
| `c4_llm_nokb` | 16 | 1.97 [1.59, 2.41]; 1.0–4.5 | 2.84 [2.62, 3.06]; 2.0–3.5 | 2.56 [2.41, 2.75]; 2.0–3.5 |
| `c6_llm_kbllm` | 24 | 1.81 [1.56, 2.08]; 1.0–3.5 | 2.60 [2.44, 2.79]; 2.0–3.5 | 2.42 [2.25, 2.60]; 1.5–3.5 |

##### Paired comparisons (b − a per scene; tie band 0.25)

| a → b | n | Faithfulness diff [CI] W/T/L | Style diff [CI] | Quality diff [CI] |
|---|---|---|---|---|
| `c0_base_raw` → `c1_lora_raw` | 16 | 0.56 [0.00, 1.09] 12/2/2 | 1.38 [1.06, 1.69] | 0.38 [0.12, 0.62] |
| `c1_lora_raw` → `c3_llm_kb_ckb` | 16 | -0.72 [-1.25, -0.25] 3/2/11 | -0.38 [-0.75, -0.03] | 0.03 [-0.28, 0.31] |
| `c4_llm_nokb` → `c3_llm_kb_ckb` | 16 | -0.12 [-0.62, 0.34] 5/4/7 | -0.25 [-0.56, 0.03] | -0.06 [-0.41, 0.25] |
| `c2_template` → `c3_llm_kb_ckb` | 16 | -0.06 [-0.47, 0.34] 5/4/7 | 0.12 [-0.09, 0.34] | 0.41 [0.03, 0.78] |
| `c6_llm_kbllm` → `c3_llm_kb_ckb` | 24 | 0.12 [-0.23, 0.48] 8/11/5 | 0.10 [-0.10, 0.31] | 0.15 [-0.12, 0.42] |
| `c1_lora_raw` → `c2_template` | 16 | -0.66 [-1.19, -0.19] 3/3/10 | -0.50 [-0.84, -0.16] | -0.38 [-0.72, -0.06] |
| `c1_lora_raw` → `c4_llm_nokb` | 16 | -0.59 [-1.00, -0.19] 3/2/11 | -0.12 [-0.38, 0.12] | 0.09 [-0.12, 0.31] |

##### Faithfulness by tier (mean, n scenes)

| Condition | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| `c4_llm_nokb` | 1.83 (3) | 2.25 (4) | 1.80 (5) | 2.25 (2) | 1.75 (2) |
| `c0_base_raw` | 1.83 (3) | 2.38 (4) | 2.00 (5) | 2.50 (2) | 1.00 (2) |
| `c1_lora_raw` | 3.00 (3) | 2.75 (4) | 2.40 (5) | 2.75 (2) | 1.75 (2) |
| `c3_llm_kb_ckb` | 2.00 (5) | 1.83 (6) | 1.93 (7) | 1.67 (3) | 2.33 (3) |
| `c6_llm_kbllm` | 2.30 (5) | 1.50 (6) | 1.79 (7) | 1.33 (3) | 2.17 (3) |
| `c2_template` | 2.83 (3) | 1.50 (4) | 1.90 (5) | 1.50 (2) | 1.75 (2) |

##### Faithfulness, seen vs unseen (mean, n scenes)

| Condition | seen | unseen |
|---|---|---|
| `c4_llm_nokb` | 2.30 (5) | 1.82 (11) |
| `c0_base_raw` | 2.40 (5) | 1.82 (11) |
| `c1_lora_raw` | 3.00 (5) | 2.36 (11) |
| `c3_llm_kb_ckb` | 1.90 (5) | 1.95 (19) |
| `c6_llm_kbllm` | 2.10 (5) | 1.74 (19) |
| `c2_template` | 2.20 (5) | 1.77 (11) |

##### Faithfulness by KB coverage (worst character in the scene) (mean, n scenes)

| Condition | high/medium | low/none |
|---|---|---|
| `c4_llm_nokb` | 1.75 (8) | 2.29 (7) |
| `c0_base_raw` | 1.94 (8) | 2.21 (7) |
| `c1_lora_raw` | 2.38 (8) | 2.86 (7) |
| `c3_llm_kb_ckb` | 1.79 (14) | 2.06 (8) |
| `c6_llm_kbllm` | 1.71 (14) | 1.88 (8) |
| `c2_template` | 2.00 (8) | 1.93 (7) |

##### Failure tags (count of ratings)

| Condition | wrong figure count | wrong/missing figure identity | wrong setting | wrong action | generic figure (name ignored) | weak style | anatomy/face artifacts | other | none |
|---|---|---|---|---|---|---|---|---|---|
| `c0_base_raw` | 2 | 4 | 6 | 5 | 1 | 13 | 1 | 0 | 0 |
| `c1_lora_raw` | 6 | 6 | 2 | 2 | 8 | 2 | 6 | 0 | 0 |
| `c2_template` | 9 | 8 | 1 | 8 | 3 | 1 | 2 | 0 | 0 |
| `c3_llm_kb_ckb` | 10 | 11 | 8 | 12 | 2 | 0 | 5 | 0 | 0 |
| `c4_llm_nokb` | 9 | 5 | 6 | 8 | 1 | 2 | 1 | 0 | 0 |
| `c6_llm_kbllm` | 12 | 19 | 3 | 10 | 3 | 0 | 1 | 0 | 0 |


**What the human ratings show:**
1. **The LoRA works for style, and probably for faithfulness.** c0 → c1:
   - style +1.38 [+1.06, +1.69];
   - quality +0.38 [+0.12, +0.62];
   - faithfulness +0.56 [+0.00, +1.09], better in 12 of 16 scenes (the CI just reaches 0).
2. **Every prompt-expansion condition is less faithful than raw text with the LoRA (c1).** Each of these CIs excludes 0:
   - c1 → c3 (the full system): −0.72 [−1.25, −0.25], worse in 11 of 16 scenes;
   - c1 → c2 (template): −0.66 [−1.19, −0.19];
   - c1 → c4 (LLM without KB): −0.59 [−1.00, −0.19].

   Expansion also costs style for c2 (−0.50) and c3 (−0.38), and gains nothing on quality.
3. **Among the expansion conditions, nothing separates clearly on faithfulness.** All of these CIs include 0:
   - **H-D1** (template c2 → LLM c3): −0.06 [−0.47, +0.34]. Null, so "no clear benefit of the LLM over a deterministic
     template" (D's wording), at n = 16 instead of the planned 32.
   - **H-D2** (no KB c4 → KB c3): −0.12 [−0.62, +0.34] overall. Null. The tier-4 part rests on 2 scenes (s15: c4 3.0 vs
     c3 1.5; s31: 1.5 vs 2.5), so it cannot be resolved at this n. D applies the D.0 rule and reports.
   - **H-C2** (LLM-written KB c6 → image-grounded KB c3): +0.12 [−0.23, +0.48] on 24 scenes, and +0.07 [−0.18, +0.36] on
     the 14 scenes whose characters have medium or high coverage. Null.
4. **Failure tags:**
   - c3 and c6 fail mostly on figure identity, count and action (c6: 19 identity tags out of 48 ratings).
   - c1's most common tag is "generic figure (name ignored)".
   - c0's most common tag is "weak style".
5. **H-B3 (seen vs unseen; descriptive):**
   - c1 → c3 on the 5 seen scenes: −1.10 [−1.30, −1.00], worse in all 5;
   - on the 11 unseen scenes: −0.55 [−1.32, +0.09].

   The predicted pattern (a gain from expansion that is larger on unseen scenes) did not appear: there is no gain
   anywhere, and the loss is smaller on unseen scenes.
6. **H-B4 (coverage; descriptive):** not supported. Scenes whose worst-covered character has low or no KB coverage score
   no lower than high/medium scenes, and higher in 5 of 6 conditions. The groups are small (7-14 scenes), and coverage is
   tangled with which scenes each member wrote.
7. **Default pipeline (B.3 rule): `c1_lora_raw`.** c1 has the highest human faithfulness of the eligible conditions (2.56
   on the 16 common scenes) and is the simplest candidate, so the tie-break never applies. c2 and `c3_llm_kb_ckb` also
   fail the style eligibility (more than 0.3 below c1), but that does not change the outcome. A records it in
   `docs/TEAM_DECISIONS_M3.md`.

##### Default pipeline (B.3 rule; human ratings on the 16 scenes every candidate was rated on)

| Condition | Faithfulness | Style | `max_train_dino` p95 | Eligible |
|---|---|---|---|---|
| `c1_lora_raw` | 2.56 | 2.97 | 0.805 | yes |
| `c2_template` | 1.91 | 2.47 | 0.789 | no (style) |
| `c4_llm_nokb` | 1.97 | 2.84 | 0.797 | yes |
| `c3_llm_kb_ckb` | 1.84 | 2.59 | 0.799 | no (style) |
| `c6_llm_kbllm` | 1.91 | 2.69 | 0.802 | yes |

1. eligible (style >= c1's 2.97 - 0.3, dino p95 <= 0.85): c1_lora_raw, c4_llm_nokb, c6_llm_kbllm
2. highest mean human faithfulness on the 16 common scenes: c1_lora_raw (2.56)
3. c1_lora_raw is the simplest eligible condition

**Default pipeline by the rule: `c1_lora_raw`**

**Caveat on comparisons across rounds.** c0, c1 and c4 were rated in round 1, and c2, c3 and c6 in round 2. Within a
scene the same two raters rated both rounds, but never side by side. Differences between round-1 and round-2
conditions (for example c1 → c3) therefore rest on absolute 1-5 ratings and could partly reflect a shift between
sessions. Within-round comparisons (c0 → c1, c1 → c4, c2 → c3, c6 → c3) do not have this problem. The automated
judge, which is the same instrument for all conditions, points the same way: c1 is at least as good as every expanded
condition (§2.2).

### 2.2 Automated metrics (supporting)
Generated by `python eval/auto_tables.py` from the files on `main` on 16 Nov, after D's #72 put c2, c3 and c4
on all 24 scenes. Re-run it rather than editing the numbers by hand. c4 here is D's 15 Nov regeneration. The rated
c4 images (A's 2 Nov set, 16 scenes) are in `results/rated/` and are used only in §2.1 and §3.

> **How to read this.** The judge passed its validation bar, but only just (§3.1), so it supports the human
> ratings and does not replace them. Its absolute level depends on the answer wording (§3.3), so only differences
> between conditions mean anything. `clip_raw` compares the image with the **raw** user text, which pushes it up for
> conditions whose prompt *is* the raw text (c0, c1).

Sources: `results/c0_base_raw@base_dev.json`, `results/c1_lora_raw@rrv_v1_dev.json`, `results/c2_template@rrv_v1_dev.json`, `results/c3_llm_kb@rrv_v1_dev.json`, `results/c3_llm_kb_ckb@rrv_v1_dev.json`, `results/c4_llm_nokb@rrv_v1_dev.json`, `results/c6_llm_kbllm@rrv_v1_dev.json`. Judge: Qwen/Qwen2-VL-2B-Instruct. Seeds 1-2 averaged per scene.

##### Per condition (mean over scenes [95% CI])

| Condition | n scenes | checklist (judge) | checklist on the 16 common scenes | `clip_raw` | `style_margin` | `max_train_dino` p95 (≤ 0.85) |
|---|---|---|---|---|---|---|
| `c0_base_raw` | 16 | 0.569 [0.496, 0.643] | 0.569 | 0.268 | 0.025 | 0.800 |
| `c1_lora_raw` | 16 | 0.576 [0.504, 0.649] | 0.576 | 0.264 | 0.093 | 0.805 |
| `c2_template` | 24 | 0.564 [0.491, 0.634] | 0.570 | 0.268 | 0.089 | 0.789 |
| `c3_llm_kb` | 24 | 0.540 [0.468, 0.609] | 0.520 | 0.254 | 0.108 | 0.807 |
| `c3_llm_kb_ckb` | 24 | 0.522 [0.472, 0.578] | 0.502 | 0.249 | 0.110 | 0.799 |
| `c4_llm_nokb` | 24 | 0.545 [0.478, 0.613] | 0.533 | 0.255 | 0.104 | 0.797 |
| `c6_llm_kbllm` | 24 | 0.509 [0.450, 0.568] | 0.514 | 0.241 | 0.111 | 0.802 |

##### Paired comparisons (b − a per scene, shared scenes; [95% CI]; W/T/L with tie band 0.0295)

| a → b | What it tests | n | checklist (judge) | `clip_raw` | `style_margin` |
|---|---|---|---|---|---|
| `c0_base_raw` → `c1_lora_raw` | LoRA (Overview §8) | 16 | +0.007 [-0.060, +0.064] 8/4/4 | -0.004 [-0.018, +0.008] 1/12/3 | **+0.069 [+0.046, +0.095] 13/3/0** |
| `c1_lora_raw` → `c3_llm_kb_ckb` | full system vs raw text (Overview §8) | 16 | -0.074 [-0.148, +0.012] 3/0/13 | **-0.016 [-0.026, -0.006] 0/12/4** | +0.010 [-0.004, +0.027] 3/13/0 |
| `c4_llm_nokb` → `c3_llm_kb_ckb` | adding the KB (Overview §8; H-D2) | 24 | -0.023 [-0.078, +0.037] 7/5/12 | -0.006 [-0.018, +0.006] 2/17/5 | +0.006 [-0.004, +0.017] 4/18/2 |
| `c2_template` → `c3_llm_kb_ckb` | LLM vs template (Overview §8) | 24 | -0.042 [-0.110, +0.030] 9/1/14 | **-0.019 [-0.031, -0.008] 1/16/7** | **+0.021 [+0.010, +0.033] 8/16/0** |
| `c6_llm_kbllm` → `c3_llm_kb_ckb` | grounded vs LLM-written KB (Overview §8; C) | 24 | +0.013 [-0.045, +0.068] 12/3/9 | +0.008 [-0.004, +0.020] 6/14/4 | -0.001 [-0.012, +0.011] 4/16/4 |
| `c2_template` → `c3_llm_kb` | LLM vs template, both on interim KB (H-D1 as run by D) | 24 | -0.023 [-0.074, +0.025] 10/3/11 | **-0.014 [-0.030, -0.001] 0/20/4** | **+0.019 [+0.009, +0.030] 6/18/0** |
| `c3_llm_kb` → `c3_llm_kb_ckb` | interim KB → C's KB (same LLM and prompt) | 24 | -0.019 [-0.076, +0.040] 8/2/14 | -0.005 [-0.020, +0.013] 2/16/6 | +0.002 [-0.007, +0.009] 2/20/2 |
| `c1_lora_raw` → `c2_template` | template vs raw text | 16 | -0.006 [-0.045, +0.033] 5/5/6 | +0.004 [-0.006, +0.013] 2/13/1 | -0.015 [-0.033, +0.004] 1/10/5 |
| `c1_lora_raw` → `c4_llm_nokb` | LLM without KB vs raw text | 16 | -0.042 [-0.095, +0.015] 4/2/10 | -0.009 [-0.025, +0.005] 1/12/3 | +0.005 [-0.013, +0.024] 4/10/2 |

Bold: the 95% CI excludes 0.

##### Checklist (judge) by tier (mean, n scenes)

| Condition | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| `c0_base_raw` | 0.634 (3) | 0.603 (4) | 0.584 (5) | 0.351 (2) | 0.583 (2) |
| `c1_lora_raw` | 0.665 (3) | 0.606 (4) | 0.608 (5) | 0.436 (2) | 0.437 (2) |
| `c2_template` | 0.631 (5) | 0.618 (6) | 0.553 (7) | 0.442 (3) | 0.488 (3) |
| `c3_llm_kb` | 0.631 (5) | 0.578 (6) | 0.528 (7) | 0.326 (3) | 0.555 (3) |
| `c3_llm_kb_ckb` | 0.620 (5) | 0.487 (6) | 0.475 (7) | 0.469 (3) | 0.589 (3) |
| `c4_llm_nokb` | 0.604 (5) | 0.536 (6) | 0.576 (7) | 0.430 (3) | 0.508 (3) |
| `c6_llm_kbllm` | 0.565 (5) | 0.520 (6) | 0.494 (7) | 0.330 (3) | 0.606 (3) |

##### Checklist (judge) seen vs unseen in training (mean, n scenes)

| Condition | seen | unseen |
|---|---|---|
| `c0_base_raw` | 0.617 (5) | 0.547 (11) |
| `c1_lora_raw` | 0.655 (5) | 0.539 (11) |
| `c2_template` | 0.662 (5) | 0.538 (19) |
| `c3_llm_kb` | 0.588 (5) | 0.528 (19) |
| `c3_llm_kb_ckb` | 0.572 (5) | 0.508 (19) |
| `c4_llm_nokb` | 0.645 (5) | 0.519 (19) |
| `c6_llm_kbllm` | 0.564 (5) | 0.494 (19) |

**What the automated metrics show:**
1. **The LoRA changes style, not checklist content.** c0 → c1 `style_margin` +0.069, better in 13 of 16 scenes, worse in none.
2. **No expansion condition scores higher than raw text on the judge.**
   - c1 → `c3_llm_kb_ckb`: −0.074, lower in 13 of 16 scenes (CI includes 0).
   - c1 → c4: −0.042. c1 → c2: −0.006.

   This agrees in direction with the human ratings (§2.1), where the gap is clear.
3. **The LLM buys style over the template.** c2 → c3 `style_margin` is +0.021 with C's KB and +0.019 with the interim
   KB (both CIs exclude 0; no scene worse). The judge shows no clear difference. Humans did not see the style gain
   (§2.1: +0.12, CI includes 0).
4. **Which KB is used makes no measurable difference.** This holds for c6 → c3 and for interim KB → C's KB.
5. **Memorisation guardrail:** every condition passes (`max_train_dino` p95 between 0.789 and 0.807, limit 0.85).
6. **Splits (descriptive):** tier 4 is the lowest tier on the judge in every condition (narrowly for c1 and
   `c3_llm_kb_ckb`), and seen scenes score higher than unseen ones in every condition.

**Consistency check:**
- **D's tables:** c0 and c1 match D's tables in `expander/EXPANDER.md` §9.
- **c2 and c3_llm_kb:** extended by #72, and the 16 original scenes are unchanged image for image.
- **C's file:** the c6 → c3 differences and CIs match `results/c6_comparison.json`, apart from the checklist tie band
  (C used 0 there; B.2 fixes 0.0295).

## 3. Judge validation and rater agreement

### 3.1 Judge vs humans (H-B1, H-B2)
Each of the 112 rated images is paired with its own automated scores (matched by file name; 0 unmatched).

##### Judge validation (112 rated images matched to results/; 0 unmatched)

| Pair | n | Spearman ρ [95% CI] |
|---|---|---|
| checklist judge vs human faithfulness | 112 | 0.41 [0.24, 0.56] |
| clip_raw vs human faithfulness | 112 | 0.24 [0.04, 0.41] |
| style_margin vs human style | 112 | 0.34 [0.16, 0.50] |

H-B1 (judge ρ ≥ 0.4 and ≥ CLIP's): **met**. H-B2 (style_margin ρ > 0): **met**.


- **H-B1 met, but narrowly.** The judge correlates with mean human faithfulness at ρ = 0.41, above the 0.4 bar and
  above CLIP's 0.24. Its CI [0.24, 0.56] still reaches below 0.4. The judge is a usable **secondary** faithfulness
  metric (including for SHAP's faithfulness chart), with humans as the primary evidence. It should not be read
  image by image.
- **H-B2 met.** `style_margin` correlates positively with human style ratings (ρ = 0.34 [0.16, 0.50]).

### 3.2 Inter-rater agreement (H-B5)
Images with two raters, using the first two raters alphabetically.

##### Inter-rater agreement (112 images with two raters)

| Dimension | Spearman ρ [CI] | Weighted κ | Exact | Within 1 |
|---|---|---|---|---|
| faithfulness | 0.43 [0.26, 0.58] | 0.50 | 0.42 | 0.92 |
| style | 0.30 [0.12, 0.46] | 0.30 | 0.45 | 0.89 |
| quality | 0.19 [-0.01, 0.37] | 0.20 | 0.43 | 0.96 |

H-B5 (faithfulness ρ ≥ 0.5): **not met**


- **H-B5 not met.** Agreement on faithfulness is ρ = 0.43 against a bar of 0.5 (weighted κ 0.50). Exact agreement is
  42%, and 92% of pairs are within one point. Style (ρ 0.30) and quality (ρ 0.19) agree less.
- As pre-registered, the human results are therefore **weaker evidence** and are reported with their spread. The
  large differences in §2.1 survive this (c1 vs every expanded condition, c0 → c1 on style). The small ones would
  not be resolved even with perfect agreement at this n.
- The judge agrees with the mean human score (0.41) about as well as two humans agree with each other (0.43).

### 3.3 Judge robustness
Source: `outputs/eval/judge_robustness_dev.json`, made with `eval/judge_robustness.py` on 14 Nov on a Colab T4
(`transformers` 5.21.0, `torch` 2.14.1). It used 20 dev images (7 c0, 7 c1, 6 c4; seeds 1 and 2; fixed sample, seed 0).
For each image, every question was asked again with the cache off and the question order shuffled. The new
score was compared with the score stored in `results/` on 4 Nov (`torch` 2.11.0).

| Check | Mean abs. change in `checklist` | Max | Outcome |
|---|---|---|---|
| Question order shuffled (pre-registered, threshold < 0.05) | 0.0005 | 0.0011 | **passed** |
| Answer instruction reworded, "Answer with only Yes or No." → "Reply Yes or No." (extra, not pre-registered) | 0.033 | 0.066 | lower on 19 of 20 images (mean shift −0.033); rank order kept (Spearman ρ = 0.98 vs stored) |

The judge asks each question in its own forward pass, so question order cannot matter by design. The first
check is therefore mainly a re-run determinism test. It also shows the scores did not change with the newer
`torch`; the largest per-question change was 0.004.

The second check shows that the judge's **absolute** scale depends on the wording of the answer instruction,
while its **ranking** of images barely changes. Comparisons between conditions are unaffected, because every
results file uses the pinned wording in `eval/judge.py`. A checklist value on its own (for example 0.55)
should not be read as an absolute proportion of the scene that is correct, and scores made with different
wordings must not be mixed.

## 4. Results, held-out
**Run.** Done once on 16 Nov with the frozen pipeline (`main` at `64b7a2f`, `eval/RUNBOOK_B.md` §7):
- 12 scenes × c0, c1, c2, `c3_llm_kb_ckb`, c4, c6 × **seeds 1-2** (cut list) = 144 images.
- Each condition was made exactly as its rated dev version: c2 and c4 with D's frozen commands; c3 with C's KB as
  of 10 Nov (`kb0.2`); c6 with `kb_llm.json`; c0 and c1 with the dev negative prompt.
- Expansions: 12 per condition, all valid, trigger first, within budget, **0 names leaked after repair**
  (`outputs/expansions/*_heldout.json`).
- Scored once, with no re-runs (`results/*_heldout.json`).
- Rated blind by all four of us on 17 Nov: 144/144 rows, 72 images × 2 raters
  (`eval/ratings/ratings_heldout.csv`, key `eval/ratings/blind_key_heldout.json`).
- Five failure tags had spacing or spelling slips, such as `generic figure(name ignored)`. `eval/ratings.py`
  mapped them to the intended taxonomy tag and listed them; no scores were changed.
- Output: `results/human_ratings_heldout.json`. The B.3 rule is **not** applied: held-out is for reporting only.

### 4.1 Human ratings
All six conditions were rated on the same sheets, so there is no cross-round caveat here. The coverage table uses
the KB c3 actually used (`kb0.2`); one scene names no character.

##### Human ratings by condition (mean over scenes [95% CI]; range)

| Condition | n scenes | Faithfulness | Style | Quality |
|---|---|---|---|---|
| `c0_base_raw` | 12 | 2.25 [1.88, 2.62]; 1.0–3.0 | 1.67 [1.29, 2.12]; 1.0–3.5 | 2.29 [1.92, 2.75]; 1.5–4.0 |
| `c1_lora_raw` | 12 | 2.54 [2.17, 2.96]; 1.5–4.0 | 3.21 [3.04, 3.38]; 2.5–3.5 | 2.54 [2.33, 2.75]; 2.0–3.0 |
| `c2_template` | 12 | 2.29 [1.96, 2.71]; 1.5–4.0 | 3.00 [2.71, 3.25]; 2.0–3.5 | 2.50 [2.29, 2.71]; 2.0–3.0 |
| `c3_llm_kb_ckb` | 12 | 2.04 [1.62, 2.50]; 1.0–4.0 | 2.92 [2.58, 3.21]; 2.0–3.5 | 2.46 [2.21, 2.71]; 1.5–3.0 |
| `c4_llm_nokb` | 12 | 2.54 [2.21, 2.92]; 2.0–4.0 | 3.25 [3.04, 3.46]; 2.5–4.0 | 2.62 [2.38, 2.88]; 2.0–3.5 |
| `c6_llm_kbllm` | 12 | 2.00 [1.58, 2.46]; 1.0–4.0 | 2.88 [2.62, 3.08]; 2.0–3.5 | 2.46 [2.21, 2.71]; 1.5–3.0 |

##### Paired comparisons (b − a per scene; tie band 0.25)

| a → b | n | Faithfulness diff [CI] W/T/L | Style diff [CI] | Quality diff [CI] |
|---|---|---|---|---|
| `c0_base_raw` → `c1_lora_raw` | 12 | 0.29 [-0.17, 0.75] 7/2/3 | 1.54 [1.08, 1.96] | 0.25 [-0.29, 0.71] |
| `c1_lora_raw` → `c3_llm_kb_ckb` | 12 | -0.50 [-0.83, -0.17] 1/4/7 | -0.29 [-0.62, 0.00] | -0.08 [-0.42, 0.25] |
| `c4_llm_nokb` → `c3_llm_kb_ckb` | 12 | -0.50 [-0.88, -0.12] 1/5/6 | -0.33 [-0.58, -0.08] | -0.17 [-0.38, 0.04] |
| `c2_template` → `c3_llm_kb_ckb` | 12 | -0.25 [-0.62, 0.08] 3/4/5 | -0.08 [-0.42, 0.17] | -0.04 [-0.38, 0.29] |
| `c6_llm_kbllm` → `c3_llm_kb_ckb` | 12 | 0.04 [-0.25, 0.38] 3/6/3 | 0.04 [-0.12, 0.21] | 0.00 [-0.25, 0.21] |
| `c1_lora_raw` → `c2_template` | 12 | -0.25 [-0.54, 0.00] 0/9/3 | -0.21 [-0.42, -0.04] | -0.04 [-0.17, 0.08] |
| `c1_lora_raw` → `c4_llm_nokb` | 12 | 0.00 [-0.42, 0.42] 3/5/4 | 0.04 [-0.17, 0.25] | 0.08 [-0.25, 0.42] |

##### Faithfulness by tier (mean, n scenes)

| Condition | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| `c0_base_raw` | 2.50 (2) | 2.00 (2) | 2.00 (3) | 2.33 (3) | 2.50 (2) |
| `c2_template` | 2.50 (2) | 1.50 (2) | 2.17 (3) | 2.17 (3) | 3.25 (2) |
| `c3_llm_kb_ckb` | 2.25 (2) | 1.50 (2) | 2.00 (3) | 2.00 (3) | 2.50 (2) |
| `c6_llm_kbllm` | 2.00 (2) | 1.75 (2) | 2.17 (3) | 1.67 (3) | 2.50 (2) |
| `c4_llm_nokb` | 2.50 (2) | 3.00 (2) | 2.50 (3) | 2.00 (3) | 3.00 (2) |
| `c1_lora_raw` | 3.25 (2) | 1.75 (2) | 2.50 (3) | 2.17 (3) | 3.25 (2) |

##### Faithfulness by KB coverage (worst character in the scene) (mean, n scenes)

| Condition | high/medium | low/none |
|---|---|---|
| `c0_base_raw` | 2.50 (2) | 2.11 (9) |
| `c2_template` | 2.25 (2) | 2.11 (9) |
| `c3_llm_kb_ckb` | 1.75 (2) | 1.89 (9) |
| `c6_llm_kbllm` | 1.50 (2) | 1.89 (9) |
| `c4_llm_nokb` | 2.00 (2) | 2.50 (9) |
| `c1_lora_raw` | 3.00 (2) | 2.28 (9) |

##### Failure tags (count of ratings)

| Condition | wrong figure count | wrong/missing figure identity | wrong setting | wrong action | generic figure (name ignored) | weak style | anatomy/face artifacts | other | none |
|---|---|---|---|---|---|---|---|---|---|
| `c0_base_raw` | 1 | 4 | 2 | 1 | 2 | 14 | 0 | 0 | 0 |
| `c1_lora_raw` | 6 | 7 | 2 | 5 | 0 | 1 | 3 | 0 | 0 |
| `c2_template` | 6 | 8 | 1 | 7 | 0 | 0 | 2 | 0 | 0 |
| `c3_llm_kb_ckb` | 7 | 8 | 0 | 5 | 0 | 1 | 3 | 0 | 0 |
| `c4_llm_nokb` | 3 | 10 | 0 | 7 | 1 | 1 | 2 | 0 | 0 |
| `c6_llm_kbllm` | 5 | 8 | 3 | 5 | 1 | 0 | 2 | 0 | 0 |


**What held-out shows:**
1. **The main dev finding replicates.** Raw text with the LoRA (c1) is more faithful than the full system:
   c1 → c3 is −0.50 [−0.83, −0.17], worse in 7 of 12 scenes and better in 1. On held-out, every condition was rated on
   the same sheet, so this no longer depends on comparing across rating rounds.
2. **The LoRA again gives a large style gain.** c0 → c1 style +1.54 [+1.08, +1.96]. Faithfulness +0.29 has a CI that
   includes 0, as it did on dev, where it just touched 0.
3. **The LLM without the KB does as well as raw text.** c1 → c4 is 0.00 [−0.42, +0.42]: c4 ties c1 for the highest
   faithfulness (2.54). On dev, c4 was clearly below c1 (−0.59). Different scene sets give different answers here,
   and n is small.
4. **Adding the KB hurts on held-out.** c4 → c3 is −0.50 [−0.88, −0.12] on faithfulness and −0.33 on style, with
   both CIs excluding 0. On dev the same comparison was null (−0.12). Most held-out characters are not in the KB
   (7 of 19 mentions are). The pattern fits the KB's descriptions crowding out the scene when a character is
   unknown or poorly covered, but 12 scenes cannot show that on their own (D's analysis, `EXPANDER.md`).
   Leaving out the two scenes (h08, h11) where c3 and c4 got the **same prompt, and so the same image** (no
   character in the KB), the KB effect is −0.60 [−1.05, −0.15] on the 10 scenes where it changed the prompt.
5. **Template vs LLM, and grounded vs LLM-written KB, are still null.** c2 → c3 −0.25 [−0.62, +0.08];
   c6 → c3 +0.04 [−0.25, +0.38].
6. **Failure tags:** every expansion condition fails mainly on figure identity, count and action. c0 fails mainly
   on style (14 of 24 ratings).
7. **The default pipeline holds up.** c1, chosen on dev, has the highest held-out faithfulness (tied with c4) and
   a style rating within 0.04 of the best.

### 4.2 Automated metrics
Generated by `python eval/auto_tables.py --results "results/*_heldout.json" --scenes eval/test_scenes_heldout.json`.

Sources: `results/c0_base_raw@base_heldout.json`, `results/c1_lora_raw@rrv_v1_heldout.json`, `results/c2_template@rrv_v1_heldout.json`, `results/c3_llm_kb_ckb@rrv_v1_heldout.json`, `results/c4_llm_nokb@rrv_v1_heldout.json`, `results/c6_llm_kbllm@rrv_v1_heldout.json`. Judge: Qwen/Qwen2-VL-2B-Instruct. Seeds 1-2 averaged per scene.

##### Per condition (mean over scenes [95% CI])

| Condition | n scenes | checklist (judge) | checklist on the 12 common scenes | `clip_raw` | `style_margin` | `max_train_dino` p95 (≤ 0.85) |
|---|---|---|---|---|---|---|
| `c0_base_raw` | 12 | 0.623 [0.552, 0.690] | 0.623 | 0.284 | -0.012 | 0.773 |
| `c1_lora_raw` | 12 | 0.667 [0.584, 0.743] | 0.667 | 0.277 | 0.087 | 0.801 |
| `c2_template` | 12 | 0.646 [0.589, 0.707] | 0.646 | 0.279 | 0.092 | 0.801 |
| `c3_llm_kb_ckb` | 12 | 0.598 [0.521, 0.681] | 0.598 | 0.254 | 0.069 | 0.764 |
| `c4_llm_nokb` | 12 | 0.642 [0.563, 0.720] | 0.642 | 0.265 | 0.089 | 0.752 |
| `c6_llm_kbllm` | 12 | 0.534 [0.458, 0.622] | 0.534 | 0.250 | 0.080 | 0.726 |

##### Paired comparisons (b − a per scene, shared scenes; [95% CI]; W/T/L with tie band 0.0295)

| a → b | What it tests | n | checklist (judge) | `clip_raw` | `style_margin` |
|---|---|---|---|---|---|
| `c0_base_raw` → `c1_lora_raw` | LoRA (Overview §8) | 12 | +0.044 [-0.017, +0.104] 7/1/4 | -0.007 [-0.024, +0.009] 1/9/2 | **+0.098 [+0.067, +0.134] 11/1/0** |
| `c1_lora_raw` → `c3_llm_kb_ckb` | full system vs raw text (Overview §8) | 12 | -0.069 [-0.136, +0.001] 3/2/7 | -0.023 [-0.048, +0.001] 2/5/5 | -0.018 [-0.045, +0.010] 3/2/7 |
| `c4_llm_nokb` → `c3_llm_kb_ckb` | adding the KB (Overview §8; H-D2) | 12 | -0.044 [-0.105, +0.011] 2/4/6 | -0.012 [-0.024, +0.000] 0/10/2 | **-0.020 [-0.045, -0.001] 0/10/2** |
| `c2_template` → `c3_llm_kb_ckb` | LLM vs template (Overview §8) | 12 | -0.048 [-0.101, +0.003] 1/6/5 | **-0.026 [-0.047, -0.004] 0/7/5** | **-0.023 [-0.046, -0.002] 1/6/5** |
| `c6_llm_kbllm` → `c3_llm_kb_ckb` | grounded vs LLM-written KB (Overview §8; C) | 12 | **+0.064 [+0.011, +0.122] 7/3/2** | +0.003 [-0.009, +0.016] 2/9/1 | -0.011 [-0.032, +0.008] 1/8/3 |
| `c1_lora_raw` → `c2_template` | template vs raw text | 12 | -0.022 [-0.065, +0.023] 3/4/5 | +0.002 [-0.008, +0.013] 1/10/1 | +0.005 [-0.009, +0.021] 3/9/0 |
| `c1_lora_raw` → `c4_llm_nokb` | LLM without KB vs raw text | 12 | -0.025 [-0.108, +0.056] 6/2/4 | -0.012 [-0.032, +0.007] 1/7/4 | +0.002 [-0.017, +0.020] 2/7/3 |

Bold: the 95% CI excludes 0.

##### Checklist (judge) by tier (mean, n scenes)

| Condition | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| `c0_base_raw` | 0.661 (2) | 0.579 (2) | 0.519 (3) | 0.630 (3) | 0.773 (2) |
| `c1_lora_raw` | 0.744 (2) | 0.473 (2) | 0.580 (3) | 0.731 (3) | 0.820 (2) |
| `c2_template` | 0.696 (2) | 0.515 (2) | 0.564 (3) | 0.684 (3) | 0.792 (2) |
| `c3_llm_kb_ckb` | 0.689 (2) | 0.443 (2) | 0.558 (3) | 0.549 (3) | 0.796 (2) |
| `c4_llm_nokb` | 0.633 (2) | 0.610 (2) | 0.636 (3) | 0.564 (3) | 0.812 (2) |
| `c6_llm_kbllm` | 0.591 (2) | 0.461 (2) | 0.488 (3) | 0.435 (3) | 0.766 (2) |


The judge points the same way as the humans, with smaller effects:
- **c1 → c3:** −0.069, CI just reaching 0.
- **LoRA:** a clear style gain, `style_margin` +0.098, better in 11 of 12 scenes.
- **KB grounding:** the image-grounded KB scores higher on the judge than the LLM-written KB (c6 → c3 +0.064, CI
  excludes 0), while humans see no difference (+0.04).
- **Memorisation guardrail:** every condition passes (p95 ≤ 0.801).

### 4.3 Judge validation and rater agreement on held-out
##### Judge validation (72 rated images matched to results/; 0 unmatched)

| Pair | n | Spearman ρ [95% CI] |
|---|---|---|
| checklist judge vs human faithfulness | 72 | 0.65 [0.46, 0.78] |
| clip_raw vs human faithfulness | 72 | 0.17 [-0.06, 0.41] |
| style_margin vs human style | 72 | 0.38 [0.15, 0.56] |

H-B1 (judge ρ ≥ 0.4 and ≥ CLIP's): **met**. H-B2 (style_margin ρ > 0): **met**.

##### Inter-rater agreement (72 images with two raters)

| Dimension | Spearman ρ [CI] | Weighted κ | Exact | Within 1 |
|---|---|---|---|---|
| faithfulness | 0.65 [0.49, 0.78] | 0.53 | 0.62 | 0.90 |
| style | 0.53 [0.34, 0.68] | 0.60 | 0.51 | 0.99 |
| quality | 0.27 [0.05, 0.47] | 0.30 | 0.50 | 0.99 |

H-B5 (faithfulness ρ ≥ 0.5): **met**


- **H-B1 and H-B2 hold on held-out too**, and more clearly than on dev. The judge agrees with human faithfulness at
  ρ = 0.65 [0.46, 0.78], and CLIP only at 0.17.
- **H-B5 is met on held-out** (ρ 0.65 ≥ 0.5), though it was not on dev (0.43). A plausible reason: all six
  conditions sat on one sheet, which makes relative judgements easier.
- **Identical images as a consistency check.** In three held-out scenes, two or three conditions produced the same
  image because their prompts were the same: c1 = c2 in h07, and c3 = c4 = c6 in h08 and h11. Raters scored
  identical images identically in all 14 within-rater pairs. The duplicates were on the same sheet, so this is
  an easy check, but it shows the scores were entered carefully.

## 5. Comparison grids
Seven scenes, chosen **by rule** rather than by eye. For dev and for held-out separately, the rule picks the scene
where c3 lost most against c1 on human faithfulness, the scene where it gained most, and the median scene. One
held-out tier-4 scene (rare characters) where the two tie is added.

Each grid shows the seed-1 image the raters saw for every condition, decoded from the blind contact sheets with
the keys in `eval/ratings/`. Under each image are the mean human faithfulness, style and quality (two raters).
`= same image` marks conditions whose prompt, and therefore picture, was identical. To rebuild a grid, run
`python eval/grids.py` (usage in the file).

| # | Scene | Why chosen | What it shows |
|---|---|---|---|
| 1 | [dev s27](../docs/figures/grid1_dev_s27.png): Yashoda scolding little Krishna | largest c3 loss (−3.5) | Only c1 keeps the scene: mother, child and the pot on the floor (faithfulness 4.5). c3 and c6 replace it with unrelated single figures (1.0). |
| 2 | [dev s12](../docs/figures/grid2_dev_s12.png): Hanuman offering Rama the ring | largest c3 gain (+1.0) | Every condition gets the kneeling pose, and none draws Hanuman as a vanara ("generic figure"). The base model (c0) is rated most faithful (4.0). |
| 3 | [dev s13](../docs/figures/grid3_dev_s13.png): Krishna as envoy in Duryodhana's court | median (−1.0) | All conditions draw a court. None clearly shows one envoy addressing a king on a throne, and faithfulness ranges only 1.5-3.0. |
| 4 | [held-out h12](../docs/figures/grid4_heldout_h12.png): Hanuman flying to Lanka (tier 5, long and informal) | largest c3 loss (−1.5) | c1 and c2 keep a flying figure over waves under the moon. c3 and c6 lose Hanuman's identity (1.0). |
| 5 | [held-out h07](../docs/figures/grid5_heldout_h07.png): churning of the ocean | largest c3 gain (+0.5) | c1 and c2 are the same image (the template added nothing). No condition shows the mountain or the gods-versus-demons split. All score 1.5-2.5. |
| 6 | [held-out h08](../docs/figures/grid6_heldout_h08.png): waking Kumbhakarna | median (−0.5) | Kumbhakarna is in no KB, so c3, c4 and c6 got the same LLM prompt and image. That image **drops the sleeping giant** and shows standing soldiers (2.0). c1 and c2 keep a huge reclining figure surrounded by people (2.5). |
| 7 | [held-out h09](../docs/figures/grid7_heldout_h09.png): Ekalavya and Dronacharya | tier 4, tie | Every LoRA condition shows a man kneeling before a figure in a forest. In c1-c3, Dronacharya is drawn as a woman in a sari, and no condition shows the thumb (all 2.0). |

**Pattern across the grids:**
- The LoRA always gives the look: c0 is the odd one out in style every time.
- Expansion mostly changes **who and what is in the picture**, often for the worse. The LLM rewrites the scene into
  generic figure descriptions, and the named character or the key object disappears (grids 1, 4, 6).
- Raw text keeps the scene's core more often.
- These are the same failure modes the taxonomy counts in §2.1 and §4.1 (identity, count, action).

## 6. Limitations
Known so far:
- small n (24 dev scenes, 16 for most conditions; 5 seen scenes);
- raters are the project team and know the system, though the codes hide the conditions;
- two raters per image;
- contact sheets show 320 px images;
- CLIP truncates at 77 tokens and shares an encoder family with the generator;
- the judge is a 2B model checked on 3 paintings before validation, and its absolute scale shifts by
  ≈ 0.03 when the answer instruction is reworded (§3.3), so only relative scores are meaningful;
- round 2 and round 1 were rated on separate sheets (§2.1 caveat);
- rater agreement on faithfulness is moderate (ρ 0.43, below the pre-registered 0.5; §3.2);
- the judge clears its validation bar only narrowly (ρ 0.41, CI down to 0.24; §3.1);
- c4 was regenerated by #72; human ratings refer to the 2 Nov images, the automated table to the 15 Nov ones;
- the round-1 key was rebuilt rather than saved;
- held-out is 12 scenes × 2 seeds (cut from 4 seeds), and every held-out scene is unseen. Held-out results for
  c4 vs c3 reverse the dev result, a reminder of how much these small sets vary.

**Said first, as required:** raw text with the LoRA (c1) is **more** faithful than every prompt-expansion condition on
dev, according to the human ratings (§2.1), and at least as good on the judge (§2.2). It is the default pipeline by
the pre-registered rule. On held-out (§4), c1 is again more faithful than the full system (c1 → c3 −0.50), and
only the LLM-without-KB condition (c4) matches it.

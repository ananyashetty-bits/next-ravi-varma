# M3 evaluation report (owner: B)

> **Status 2026-11-14: §1 (setup), §2.2 (automated metrics) and §3.3 (judge robustness) written; §2.1, §3.1-3.2, §4-§6 are placeholders.**
> Rules: every number names its source file. Human-rating numbers come only from `eval/ratings.py` run on the
> final sheet after every rater has submitted. Nothing here is typed from memory or from a chat.
> Markers: `TODO(B)` = B to fill; `TODO(x)` = needs input from member x.

## 1. Setup

### 1.1 Test scenes
**Dev set** (`eval/test_scenes_dev.json`): **24 scenes**, 8 each from B (s09-s16), C (s17-s24) and D (s25-s32).
The plan was 32; A's s01-s08 were never added (section B deviation, 2026-11-14).

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

**Held-out set:** 12 scenes written by B, stored privately (outside the repo and the shared Drive) and not
used for judge development. `TODO(B)`: run with the frozen code, seeds 1-4, then reveal
`eval/test_scenes_heldout.json` with its tier, epic and seen counts. Held-out results are reported and
never used to choose the default pipeline.

### 1.2 Conditions
All conditions use the same generator: Stable Diffusion 1.5, DPM++ 2M, 30 steps, CFG 7.0, 512×512
(`config/m3.json`). Everything except c0 adds `rrv_v1`, the M2 winner (`D_lr2e-4` step 800, sha256
`067ee19f…`), at LoRA scale 1.0. Trigger `rrvarma style` first, style tail `oil painting, canvas texture,
classical indian art` last, prompts fitted to 70 CLIP tokens.

| Condition | What the model sees | Scored dev scenes | Results file |
|---|---|---|---|
| `c0_base_raw` | base SD1.5, raw user text | 16 | `results/c0_base_raw@base_dev.json` |
| `c1_lora_raw` | LoRA, trigger + raw text + style tail | 16 | `results/c1_lora_raw@rrv_v1_dev.json` |
| `c2_template` | LoRA, deterministic template from the parsed scene (D's interim KB `d-kb0`) | 16 | `results/c2_template@rrv_v1_dev.json` |
| `c3_llm_kb` | LoRA, LLM expansion with D's interim KB `d-kb0` | 16 | `results/c3_llm_kb@rrv_v1_dev.json` |
| `c3_llm_kb_ckb` | LoRA, LLM expansion with C's image-grounded KB | 24 | `results/c3_llm_kb_ckb@rrv_v1_dev.json` |
| `c4_llm_nokb` | LoRA, LLM expansion without KB | 16 | `results/c4_llm_nokb@rrv_v1_dev.json` |
| `c6_llm_kbllm` | as `c3_llm_kb_ckb`, but KB descriptions written by the LLM instead of grounded in paintings | 24 | `results/c6_llm_kbllm@rrv_v1_dev.json` |

The expander LLM is `Qwen/Qwen3-4B-Instruct-2507` (revision pinned in `config/m3.json`), 4-bit, temperature 0,
prompt `v1`. Each dev condition has seeds 1 and 2 (32 or 48 images). Section D treats only real-KB images as
evidence; for human ratings, "c3" means `c3_llm_kb_ckb` unless stated otherwise.
`TODO(C)`: `@rrv_names` status per `train/NAMES_MODEL.md` (Section C deviation of 4 Nov says it was not run).
`TODO(D)`: optional conditions (`c5`, `c3_7b`, `c3_neg`, `c3_v2`) if any were run.

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
  - Round 2: c2, `c3_llm_kb_ckb`, c6 on the first 16 scenes, plus `c3_llm_kb_ckb` and c6 on s17-s24. `TODO(B)`: dates and counts.

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
`TODO(B)` from the final `eval/ratings.py` run, after every rater has submitted:
- condition × (faithfulness, style, quality): mean, CI and range;
- paired comparisons (Overview §8);
- per tier, seen/unseen (H-B3), KB coverage (H-B4);
- failure-tag counts per condition;
- default-pipeline choice by the B.3 rule (recorded by A in `docs/TEAM_DECISIONS_M3.md`).

### 2.2 Automated metrics (supporting)
Generated by `python eval/auto_tables.py` from the files on `main` (14 Nov). Re-run it rather than editing
the numbers by hand.

> **How to read this.** The judge's `checklist` counts as faithfulness evidence only to the extent H-B1
> (judge vs humans, §3.1) supports it, and it is not decided yet. Its absolute level depends on the answer
> wording (§3.3), so only differences between conditions mean anything. `clip_raw` compares the image with
> the **raw** user text, which pushes it up for conditions whose prompt *is* the raw text (c0, c1). Human
> ratings (§2.1) are the primary evidence.

Sources: `results/c0_base_raw@base_dev.json`, `results/c1_lora_raw@rrv_v1_dev.json`, `results/c2_template@rrv_v1_dev.json`, `results/c3_llm_kb@rrv_v1_dev.json`, `results/c3_llm_kb_ckb@rrv_v1_dev.json`, `results/c4_llm_nokb@rrv_v1_dev.json`, `results/c6_llm_kbllm@rrv_v1_dev.json`. Judge: Qwen/Qwen2-VL-2B-Instruct. Seeds 1-2 averaged per scene.

##### Per condition (mean over scenes [95% CI])

| Condition | n scenes | checklist (judge) | checklist on the 16 common scenes | `clip_raw` | `style_margin` | `max_train_dino` p95 (≤ 0.85) |
|---|---|---|---|---|---|---|
| `c0_base_raw` | 16 | 0.569 [0.496, 0.643] | 0.569 | 0.268 | 0.025 | 0.800 |
| `c1_lora_raw` | 16 | 0.576 [0.504, 0.649] | 0.576 | 0.264 | 0.093 | 0.805 |
| `c2_template` | 16 | 0.570 [0.481, 0.659] | 0.570 | 0.268 | 0.078 | 0.767 |
| `c3_llm_kb` | 16 | 0.520 [0.426, 0.612] | 0.520 | 0.249 | 0.104 | 0.796 |
| `c3_llm_kb_ckb` | 24 | 0.522 [0.472, 0.578] | 0.502 | 0.249 | 0.110 | 0.799 |
| `c4_llm_nokb` | 16 | 0.535 [0.453, 0.619] | 0.535 | 0.255 | 0.099 | 0.797 |
| `c6_llm_kbllm` | 24 | 0.509 [0.450, 0.568] | 0.514 | 0.241 | 0.111 | 0.802 |

##### Paired comparisons (b − a per scene, shared scenes; [95% CI]; W/T/L with tie band 0.0295)

| a → b | What it tests | n | checklist (judge) | `clip_raw` | `style_margin` |
|---|---|---|---|---|---|
| `c0_base_raw` → `c1_lora_raw` | LoRA (Overview §8) | 16 | +0.007 [-0.060, +0.064] 8/4/4 | -0.004 [-0.018, +0.008] 1/12/3 | **+0.069 [+0.046, +0.095] 13/3/0** |
| `c1_lora_raw` → `c3_llm_kb_ckb` | full system vs raw text (Overview §8) | 16 | -0.074 [-0.148, +0.012] 3/0/13 | **-0.016 [-0.026, -0.006] 0/12/4** | +0.010 [-0.004, +0.027] 3/13/0 |
| `c4_llm_nokb` → `c3_llm_kb_ckb` | adding the KB (Overview §8; H-D2) | 16 | -0.033 [-0.105, +0.047] 3/6/7 | -0.007 [-0.021, +0.008] 1/11/4 | +0.005 [-0.010, +0.020] 3/11/2 |
| `c2_template` → `c3_llm_kb_ckb` | LLM vs template (Overview §8) | 16 | -0.068 [-0.154, +0.031] 4/1/11 | **-0.020 [-0.035, -0.005] 1/10/5** | **+0.025 [+0.011, +0.042] 6/10/0** |
| `c6_llm_kbllm` → `c3_llm_kb_ckb` | grounded vs LLM-written KB (Overview §8; C) | 24 | +0.013 [-0.045, +0.068] 12/3/9 | +0.008 [-0.004, +0.020] 6/14/4 | -0.001 [-0.012, +0.011] 4/16/4 |
| `c2_template` → `c3_llm_kb` | LLM vs template, both on interim KB (H-D1 as run by D) | 16 | -0.050 [-0.116, +0.011] 5/3/8 | **-0.019 [-0.042, -0.001] 0/13/3** | **+0.025 [+0.014, +0.039] 4/12/0** |
| `c3_llm_kb` → `c3_llm_kb_ckb` | interim KB → C's KB (same LLM and prompt) | 16 | -0.018 [-0.098, +0.071] 5/0/11 | -0.001 [-0.020, +0.023] 2/10/4 | +0.000 [-0.010, +0.009] 0/14/2 |
| `c1_lora_raw` → `c2_template` | template vs raw text | 16 | -0.006 [-0.045, +0.033] 5/5/6 | +0.004 [-0.006, +0.013] 2/13/1 | -0.015 [-0.033, +0.004] 1/10/5 |
| `c1_lora_raw` → `c4_llm_nokb` | LLM without KB vs raw text | 16 | -0.041 [-0.095, +0.018] 4/2/10 | -0.009 [-0.026, +0.006] 1/12/3 | +0.005 [-0.013, +0.024] 4/10/2 |

Bold: the 95% CI excludes 0.

##### Checklist (judge) by tier (mean, n scenes)

| Condition | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| `c0_base_raw` | 0.634 (3) | 0.603 (4) | 0.584 (5) | 0.351 (2) | 0.583 (2) |
| `c1_lora_raw` | 0.665 (3) | 0.606 (4) | 0.608 (5) | 0.436 (2) | 0.437 (2) |
| `c2_template` | 0.657 (3) | 0.606 (4) | 0.607 (5) | 0.494 (2) | 0.346 (2) |
| `c3_llm_kb` | 0.600 (3) | 0.587 (4) | 0.535 (5) | 0.276 (2) | 0.474 (2) |
| `c3_llm_kb_ckb` | 0.620 (5) | 0.487 (6) | 0.475 (7) | 0.469 (3) | 0.589 (3) |
| `c4_llm_nokb` | 0.553 (3) | 0.594 (4) | 0.583 (5) | 0.394 (2) | 0.408 (2) |
| `c6_llm_kbllm` | 0.565 (5) | 0.520 (6) | 0.494 (7) | 0.330 (3) | 0.606 (3) |

##### Checklist (judge) seen vs unseen in training (mean, n scenes)

| Condition | seen | unseen |
|---|---|---|
| `c0_base_raw` | 0.617 (5) | 0.547 (11) |
| `c1_lora_raw` | 0.655 (5) | 0.539 (11) |
| `c2_template` | 0.662 (5) | 0.528 (11) |
| `c3_llm_kb` | 0.588 (5) | 0.489 (11) |
| `c3_llm_kb_ckb` | 0.572 (5) | 0.508 (19) |
| `c4_llm_nokb` | 0.658 (5) | 0.479 (11) |
| `c6_llm_kbllm` | 0.564 (5) | 0.494 (19) |

**What the automated metrics show (descriptive, judge not yet validated):**
1. **The LoRA changes style, not content.** c0 → c1 raises `style_margin` clearly (+0.069; better in 13 of 16
   scenes, worse in none). Checklist and `clip_raw` do not move.
2. **No prompt-expansion condition scores higher than raw text on the judge.**
   - c1 → `c3_llm_kb_ckb` is −0.074, lower in 13 of 16 scenes, but the CI includes 0.
   - `clip_raw` is clearly lower for c3. That is partly built into the metric, because c1's prompt is the raw text.
   - Template (c2) and LLM without KB (c4) are also no better than c1.
3. **The LLM expander buys style over the template.** c2 → c3 raises `style_margin` by +0.025 with both
   KBs (CI excludes 0; no scene worse). There is no clear difference on the judge.
4. **Which KB is used makes no measurable difference.**
   - Image-grounded vs LLM-written KB (c6 → c3) shows no clear difference on any automated metric.
   - Neither does D's interim KB vs C's KB (`c3_llm_kb` → `c3_llm_kb_ckb`).
5. **Memorisation guardrail:** every condition passes (`max_train_dino` p95 between 0.767 and 0.805, limit 0.85).
6. **Splits (descriptive, small n):**
   - Tier 4 (rare characters, spelling variants) is the lowest tier for most conditions.
   - Every condition scores higher on the 5 seen scenes than on the unseen ones.
   - The 24-scene conditions include C's s17-s24. For those two conditions, use the "common scenes" column
     when comparing with the 16-scene conditions.

**Consistency check:**
- **D's tables:** the c0-c4 means, guardrail values and paired comparisons are identical to D's tables in
  `expander/EXPANDER.md` §9.
- **C's file:** the c6 → c3 differences and CIs match `results/c6_comparison.json`. That file's checklist
  win/tie/loss (13/1/10) uses a tie band of 0, whereas section B.2 fixes 0.0295 for all automated scores,
  which gives 12/3/9. The difference and CI do not depend on the tie band.

`TODO(B)`: decide H-D1, H-D2 and the Overview §8 comparisons on human faithfulness (§2.1); the table above is
supporting only.

## 3. Judge validation and rater agreement

### 3.1 Judge vs humans (H-B1, H-B2)
`TODO(B)`, from the final `eval/ratings.py` run: judge vs human faithfulness compared with CLIP (H-B1), and
`style_margin` vs human style (H-B2).

### 3.2 Inter-rater agreement (H-B5)
`TODO(B)`, from the final `eval/ratings.py` run.

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
`TODO(B, held-out)`: the same tables on 12 scenes × 4 seeds, run once with the frozen code.

## 5. Comparison grids
`TODO(B)`: 6-8 grids (one scene, all conditions), including failures.

## 6. Limitations
Known so far:
- small n (24 dev scenes, 16 for most conditions; 5 seen scenes);
- raters are the project team and know the system, though the codes hide the conditions;
- two raters per image;
- contact sheets show 320 px images;
- CLIP truncates at 77 tokens and shares an encoder family with the generator;
- the judge is a 2B model checked on 3 paintings before validation, and its absolute scale shifts by
  ≈ 0.03 when the answer instruction is reworded (§3.3), so only relative scores are meaningful;
- round 2 and round 1 were rated on separate sheets;
- the round-1 key was rebuilt rather than saved.

`TODO(B)`: if c1 or c2 is as good as c3, say so first.

# M3 evaluation report (owner: B)

> **Status 2026-11-14: §1 (setup) drafted from the repo as of `b31485c`; §2-§6 are placeholders.**
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
  `TODO(B)`: confirm the rebuild matched all 48 codes.
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
`TODO(B)` from `eval/ratings.py` (final run) and `results/*_dev.json`:
- condition × (faithfulness, style, quality): mean, CI and range;
- condition × (checklist, clip_raw, style_margin, max_train_dino p95);
- paired comparisons (Overview §8): c0→c1, c1→c3, c4→c3, c2→c3, c6→c3;
- per tier, seen/unseen (H-B3), KB coverage (H-B4);
- failure-tag counts per condition;
- default-pipeline choice by the B.3 rule (recorded by A in `docs/TEAM_DECISIONS_M3.md`).

## 3. Judge validation and rater agreement
`TODO(B)`:
- H-B1: judge vs human faithfulness, compared with CLIP;
- H-B2: `style_margin` vs human style;
- H-B5: inter-rater agreement;
- judge robustness: `eval/judge_robustness.py`, 20 images, question order shuffled. The judge asks each
  question in a separate pass, so this mainly tests re-run determinism.

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
- the judge is a 2B model checked on 3 paintings before validation;
- round 2 and round 1 were rated on separate sheets;
- the round-1 key was rebuilt rather than saved.

`TODO(B)`: if c1 or c2 is as good as c3, say so first.

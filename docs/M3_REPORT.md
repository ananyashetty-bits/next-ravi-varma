# Milestone 3 Report: Text Parsing and Semantic Scene Generation

**Team:** A (pipeline, SHAP), B (evaluation), C (KB, parser, name-aware LoRA), D (expander, report assembly). **Assembled by:** D.

> **Rules for this file**
> - **Every number comes from a file in `results/` or from B's tables in `eval/EVAL_REPORT.md`** (ratings sheet included), and each one names its source next to it. "Not tested" is an acceptable entry. No number is typed in from memory or from a chat.
> - **B audits this report line by line on Day 6** (claims audit). Anything not backed by a file is softened or removed.
> - **Timing risk (Day 5):** contributors' sections are due **Day 5 noon**, but B's held-out run and held-out ratings finish only on **Day 5 evening**. The report is therefore assembled in two passes: (1) by Day 5 evening, with dev results and every held-out cell marked `TODO(B, held-out)`; (2) held-out tables added when B posts them, then the text is frozen before B's Day-6 audit. If held-out numbers are late, the report ships with the cells marked "pending" rather than with estimates.
> - Markers: `TODO(A|B|C|D)` = owner must supply; `TODO(results)` = needs a measured number.

---

## 1. Goal and system

**Inputs:** Overview v2 §1-§3 (D). A's `pipeline/PIPELINE.md` for the run instructions and timings. **Feeds from `results/`:** none.

**Goal.** A user types an epic scene in plain words, for example "Yudhishthira weeping over the fallen body of Karna on the battlefield of Kurukshetra". The system produces a painting in Raja Ravi Varma's style that shows that scene. Milestone 2 gave us a LoRA on Stable Diffusion 1.5 (`rrv_v1`, the M2 winner `D_lr2e-4` at step 800) that knows Varma's *style* but not the *stories*. Its training captions had no names and its text encoder was frozen, so "Karna" means nothing to it (`hub/rrvarma_lora_v1/NAMES_TEST.md`). Milestone 3 adds the text side: parse the scene, turn names into visual descriptions, and fit them into a prompt the model can use.

```
user text ──► [C] parse_scene ──► SceneSpec ──► [D] expand ──► ExpandedPrompt ──► [A] generate ──► image
                 (KB lookup,                     (local LLM or template,              (M2 model + LoRA)
                  no LLM)                         token-fitted, names scrubbed)
                                                                     [A] SHAP attribution of prompt parts
                                [B] test set, checklist judge, blind ratings, stats, evaluate the whole chain
                                [C] optional: name-aware LoRA (model-side fix)
```

- **Parser and KB (C):** finds characters (including misspellings and aliases), setting and mood, and attaches name-free visual descriptions grounded in Varma's paintings (`kb/KB.md`).
- **Expander (D):** writes name-free scene phrases with a local instruct LLM (temperature 0, cached), or a deterministic template. The code puts the trigger first and the style tail last, fits the prompt into 70 CLIP tokens and scrubs names (`expander/EXPANDER.md`).
- **Generator (A):** SD1.5 + `rrv_v1`, DPM++ 30 steps, CFG 7.0, 512×512, settings from `config/m3.json` (`pipeline/PIPELINE.md`). One call: `text_to_image(text, seed)`.
- **Constraint:** everything runs on a free Colab T4. The LLM, the diffusion model and the judge run in separate sessions and communicate through files.

Generation takes about 5 s per 512×512 image on a T4 once the model is loaded (mean `gen_seconds` in A's run records: c1 5.3 s, c4 5.0 s; `outputs/eval/c{1,4}_*_metrics.json`). Loading the diffusion model takes 53 s on the first run of a session and about 9 s once cached, plus the download on a brand-new runtime (`pipeline/PIPELINE.md` §5, from the held-out manifests).

**Example (dev scene s09, seed 1; `outputs/expansions/c3_llm_kb_dev.json`, `results/c3_llm_kb@rrv_v1_dev.json`):**
- **Input:** "Sita sitting alone under a tree in the Ashoka grove, sorrowful, the garden quiet at dusk"
- **Expansion (c3, 54 tokens):** "rrvarma style, young woman in red saree, gold border and bangles, layered pearl necklaces, sitting alone under tree, flowering trees in grove, dusk lighting, quiet atmosphere, sorrowful expression, still posture, oil painting, canvas texture, classical indian art"
- **Image:** `s09_c3_llm_kb_s1.png` (D's Drive, `m3/images/c3_llm_kb@rrv_v1/dev/`), judge checklist 0.92. Raw text with the LoRA scores 0.87 on the same scene (`s09_c1_lora_raw_s1.png`).

This is a simple, one-figure scene where nothing is dropped. The typical failures are in §5 and `expander/EXPANDER.md` §11.

## 2. Research question and what is new since the halfway version

**Inputs:** Overview v2 §1, §9 (D). **Feeds from `results/`:** none.

**Research question.** The M2 model knows the style but not the stories. **Which part of the system actually makes the picture match the story: the knowledge base, the LLM, the prompt structure, or the model itself? And at what cost to style?**

**Short answer on dev (human ratings, `eval/EVAL_REPORT.md` §2.1):**
- **The model:** the LoRA adds Varma style (c0 → c1: style +1.38 [+1.06, +1.69]).
- **The prompt side:** none of the prompt-side components helped faithfulness. Raw text with the LoRA (c1) was more faithful than:
  - the template (c2): −0.66 [−1.19, −0.19];
  - the LLM without the KB (c4): −0.59 [−1.00, −0.19];
  - the LLM with the KB (c3): −0.72 [−1.25, −0.25].
- **Among the expansion conditions,** neither the LLM (H-D1) nor the KB (H-D2) made a clear difference.
- **The default pipeline** is therefore `c1_lora_raw` (§4.1).
- **Held-out (12 unseen scenes, rated blind on one sheet, §4.2):**
  - **The main finding replicates:** c1 → c3 is −0.50 [−0.83, −0.17], worse in 7 of 12 scenes.
  - The LLM without the KB (c4) ties c1 (0.00 [−0.42, +0.42]).
  - Adding the KB to the LLM made pictures **less** faithful (c4 → c3: −0.50 [−0.88, −0.12]). Most held-out characters are not in the KB.

**What is new.** The halfway feedback asked for "the simplest thing possible". We kept the pipeline simple (no UI, one base model, one LoRA, local models only) and put the effort into answering the question with evidence:
1. a controlled **ablation ladder** c0-c6 (§3) that adds one component at a time;
2. a **Shapley (SHAP) attribution** of the prompt's parts, which measures which phrases buy faithfulness and which buy style (A);
3. a **model-side fix**: a LoRA retrained with names in the captions, gated by a coverage audit and compared with the prompt-side fix (C);
4. an evaluation we can defend: **blind human ratings** plus a **checklist judge validated against those ratings**, with **hypotheses pre-registered** in `experiments/HYPOTHESES.md` before any scored run (B).

**Model-side fix, outcome:**
- **Audit:** GO. 99 of 147 training images (67%) have characters filled, and 11 characters appear in 3 or more images (`train/NAMES_MODEL.md` §2; C confirmed 11 as the final count on 16 Nov, so the 10 in `kb/KB.md` §8 is superseded).
- **The LoRA:** `C_names` was trained late, passed the M2 guardrails, and showed **no gain from names** on the pre-registered names test. It was not registered as `rrv_names` (§4.5).

## 3. Method

**Inputs:** B's `eval/EVAL_REPORT.md` §1 (setup); `experiments/HYPOTHESES.md` (all sections); A's `pipeline/TOKEN_BUDGET.md`; C's `kb/KB.md` (KB schema, coverage thresholds). **Feeds from `results/`:** none (method only).

**Test sets.** The dev set (`eval/test_scenes_dev.json`) was planned as 32 scenes, 8 written by each member, with at least one per tier. The analysed dev set has **24** (B s09-s16, C s17-s24, D s25-s32). A's 8 (s01-s08) were added on 16 Nov, after the dev results and the B.3 decision, so they are outside the analysis and descriptive only (§6). The tiers are:
1. one well-known character;
2. two characters interacting;
3. multi-figure;
4. rare or absent-in-training character, or a spelling variant;
5. stress: vague, non-epic, 100+ words, misspellings.

The team-wide targets were at least 10 Ramayana and at least 10 Mahabharata scenes. Each scene has 3-6 name-free, style-free yes/no checklist questions, written before any image was generated and reviewed by a second member. B also wrote a held-out set of 12 scenes privately and revealed it only after the S3 freeze. `seen_in_training` marks scenes whose episode appears in M1's `scene_metadata`.

Final dev counts (`eval/EVAL_REPORT.md` §1.1):
- **Tiers 1-5:** 5 / 6 / 7 / 3 / 3.
- **Epics:** Ramayana 11, Mahabharata 10, Puranic 1, unknown 2.
- **Seen / unseen:** 5 / 19.
- **Checklists:** 126 questions; pose 27, setting 25, count 23, object 19, figure 16, attire 11, mood 5.
- **Which conditions cover which scenes:** c0 and c1 were scored on the first 16 scenes (B's and D's). D's c2, c3 and c4, and C's `c3_llm_kb_ckb` and c6, were scored on all 24.
- **Changed scenes:** C's s17 and s23 were changed from Radha and Ahalya to characters in D's interim KB (`kb/KB.md` §6). So the dev set tests the rare-character case less than intended.

**Conditions** (all on `rrv_v1` unless marked `@rrv_names`):

| ID | What the model sees | Owner |
|---|---|---|
| `c0_base_raw` | base SD1.5, raw text, no LoRA | A |
| `c1_lora_raw` | LoRA, trigger + raw text + style tail | A |
| `c2_template` | LoRA, deterministic template from the SceneSpec | D |
| `c3_llm_kb` | LoRA, LLM expansion with KB descriptions (the main system) | D |
| `c4_llm_nokb` | LoRA, LLM expansion without KB | D |
| `c5_llm_kb_layout` (optional) | c3 + figure count / position / pose phrases | D |
| `c6_llm_kbllm` | c3 with LLM-written instead of image-grounded KB descriptions | C |
| `@rrv_names` (gated) | the same prompts on the name-aware LoRA | C |

Two KB versions of c3 exist:
- **`c3_llm_kb`** uses D's interim KB `d-kb0` (`expander/interim_kb/`), written because C's KB was not ready when D's runs were due.
- **`c3_llm_kb_ckb`** is C's re-run on C's image-grounded KB `kb0.2`, made as the baseline for c6.

For **human** ratings, "c3" means `c3_llm_kb_ckb` (Section B deviation, 14 Nov). On the judge, the two c3 versions show no clear difference (§4.3).

**Optional D conditions:** none was run (`c5_llm_kb_layout`, `c3_7b`, `c3_neg`, `c3_v2`). c5, c3_7b and c3_neg are built and runnable (`expander/EXPANDER.md` §10). `c3_v2` needs SHAP values that have no source file yet.

**Frozen expander settings:**
- prompt `v1`; template `tpl2`;
- LLM `Qwen/Qwen3-4B-Instruct-2507` @ `cdbee75f`, 4-bit NF4, temperature 0;
- `config/m3.json`: `style_tail` "oil painting, canvas texture, classical indian art", `negative` "ugly, blurry, bad anatomy", `max_prompt_tokens` 70 (excluding start/end tokens), LoRA scale 1.0.

**Metrics.**
- *Primary:* blind human ratings, 1-5, on scene faithfulness, style fidelity and visual quality. At least 2 raters per image; contact sheets carry random codes only. Raters also tag failures from a fixed taxonomy. Dev ratings use seed 1.
- *Supporting:* checklist judge (vision-language model, P(yes) per question); `clip_raw` (CLIP against the raw user text); `style_margin`; `max_train_dino` (memorisation guardrail).
- *Process:* expansion validity, name leaks and latency.

Judge details:
- **Model:** `Qwen/Qwen2-VL-2B-Instruct`, fp16, one image and one question at a time (`eval/judge.py`).
- **Validation threshold (H-B1):** the judge counts as faithfulness evidence only if its per-image score correlates with mean human faithfulness at Spearman ρ ≥ 0.4, and at least as well as `clip_raw` does (`experiments/HYPOTHESES.md` §B).

**Statistics.** Comparisons are paired per scene. We report win/tie/loss counts and a paired bootstrap 95% CI over scenes, with no significance claims; a CI that includes 0 means "no clear difference". Tie bands are 0.25 for ratings and M2's 0.0295 for automated scores.

**Choosing the default pipeline.** The default for M4 is chosen at S3 **on dev only**, by the rule fixed in Overview §10: highest mean faithfulness, subject to the style and memorisation guardrails; on a CI including 0, the simpler condition wins. **Decision: `c1_lora_raw`** (raw text + LoRA; `text_to_image(text, seed, mode="raw")`), recorded in `docs/TEAM_DECISIONS_M3.md` D1 with the eligibility table (§4.1).

**Hypotheses.** Pre-registered per owner in `experiments/HYPOTHESES.md` (sections A-D). Each section's merge timestamp must precede that owner's first scored run. B's Day-6 audit confirmed this from git history for all four sections (`eval/CLAIMS_AUDIT.md` §1). Section D (expander), merged in #20 before any D condition was scored:
- **H-D1 (core):** `c3_llm_kb` is more faithful than `c2_template` (the LLM adds composition the template lacks).
- **H-D2 (core):** `c3_llm_kb` beats `c4_llm_nokb`, with the gain concentrated on tier 4 (rare characters, spelling variants).
- **H-D3 (core if c5 runs):** `c5_llm_kb_layout` beats `c3_llm_kb` on count and pose checklist items in tiers 2-3.
- **H-D4 (optional):** `c3_7b` shows no clear faithfulness gain over `c3_llm_kb` and costs at least 2x the LLM seconds per scene.
- **H-D5 (optional):** `c3_neg` produces fewer anachronism failures than `c3_llm_kb`, with no loss of faithfulness or style.
- **H-D6 (optional):** prompt v2, ordered by A's SHAP value per token, improves faithfulness on scenes where v1 dropped phrases.
- **D.7 validity targets:** at least 95% valid records and zero names after repair per expansion file.

Hypotheses of the other sections, in one line each (full text in `experiments/HYPOTHESES.md`):
- **A:** the token budget (H-A1-A3) and the SHAP attribution (H-A4-A6).
- **B:** judge validity, ρ ≥ 0.4 and at least as good as `clip_raw` (H-B1); `style_margin` tracks human style (H-B2); seen vs unseen and coverage (H-B3, H-B4); rater agreement ρ ≥ 0.5 (H-B5).
- **C:** parser precision and recall ≥ 0.90 (H-C1); grounded KB beats LLM-written KB (H-C2); names in captions make names work (H-C3).

Deviations (dated lines in `experiments/HYPOTHESES.md`):
- **D, 4 Nov:** c2/c3 use D's interim KB `d-kb0`, because C's KB was not available.
- **D, 4 Nov:** c2/c3 images are made with `generate_api.py` as of `e9b5a5d` (the generator that made c0/c1/c4), because `generate()` is missing on `main`.
- **B:** the dev set has 24 scenes, not 32; LoRA scale 1.0 is assumed for c0/c1; the c4 manifest labels were normalised; the round-1 blind key was rebuilt from the images; ratings come in two rounds; one non-pre-registered robustness extra.
- **C:**
  - c6 was compared with a fresh `c3_llm_kb_ckb`;
  - H-C2 was decided on the judge only;
  - the names LoRA was run late, with 70 of 147 captions named;
  - its guardrails used the step-800 screen rows;
  - the 2×2 was not run, because the names test showed no gain and `run_batch.py` is broken.
- **A, 18 Nov (design notes, before any token-budget or SHAP image):**
  - SHAP and the token budget use 8 dev scenes (2 per tier, 1-4) written as name-free player phrases, at seeds 1-2.
  - The length ladder is about 22/36/46/64 tokens instead of 15/30/50/70.
  - H-A6 is tested with an added `with_names` variant.
  - The earlier s07/s08 SHAP files were discarded.

## 4. Results

**Inputs:**
- B: `eval/EVAL_REPORT.md` §2-§4 (tables, paired comparisons, judge validation, grids);
- A: `analysis/shap/SHAP_ANALYSIS.md` (chart, interactions, value per token), `pipeline/TOKEN_BUDGET.md`;
- C: `train/NAMES_MODEL.md` (2×2) and `kb/KB.md` (c6, parser precision/recall);
- D: `expander/EXPANDER.md` (process metrics, optional experiments).

**Feeds from `results/`:** `results/<condition>@<model>_dev.json` and `results/<condition>@<model>_heldout.json` for c0-c6, the optional D conditions and `@rrv_names`; A's SHAP results; `outputs/expansions/*.json` via `check_expansions.py`.

### 4.1 Ablation, dev
**Human ratings (primary).**
- **Source:** `eval/EVAL_REPORT.md` §2.1 and `results/human_ratings_dev.json`, from B's final `eval/ratings.py` run on rounds 1 and 2.
- **Ratings:** 224 ratings of 112 images (seed 1), two raters per image.
- **Rounds:** round 1 (4 Nov) rated c0, c1 and c4; round 2 (14 Nov) rated c2, `c3_llm_kb_ckb` and c6.
- **c4 images:** the rated ones are A's 2 Nov images (`results/rated/c4_llm_nokb@rrv_v1_dev.json`).
- **Notation:** means over scenes on a 1-5 scale, with 95% CI.

| Condition | n scenes | Faithfulness | Style | Quality |
|---|---|---|---|---|
| `c0_base_raw` | 16 | 2.00 [1.59, 2.44] | 1.59 [1.34, 1.91] | 2.09 [1.88, 2.31] |
| `c1_lora_raw` | 16 | **2.56 [2.19, 2.97]** | **2.97 [2.75, 3.19]** | 2.47 [2.25, 2.69] |
| `c2_template` | 16 | 1.91 [1.59, 2.31] | 2.47 [2.22, 2.72] | 2.09 [1.78, 2.41] |
| `c3_llm_kb_ckb` | 24 | 1.94 [1.67, 2.21] | 2.71 [2.48, 2.92] | 2.56 [2.35, 2.79] |
| `c4_llm_nokb` | 16 | 1.97 [1.59, 2.41] | 2.84 [2.62, 3.06] | 2.56 [2.41, 2.75] |
| `c6_llm_kbllm` | 24 | 1.81 [1.56, 2.08] | 2.60 [2.44, 2.79] | 2.42 [2.25, 2.60] |

Paired over scenes (b − a, tie band 0.25 rating points; faithfulness W/T/L):

| a → b | n | Faithfulness | Style | Quality |
|---|---|---|---|---|
| c0 → c1 | 16 | +0.56 [0.00, +1.09], 12/2/2 | **+1.38 [+1.06, +1.69]** | **+0.38 [+0.12, +0.62]** |
| c1 → c2 | 16 | **−0.66 [−1.19, −0.19]**, 3/3/10 | **−0.50 [−0.84, −0.16]** | **−0.38 [−0.72, −0.06]** |
| c1 → c4 | 16 | **−0.59 [−1.00, −0.19]**, 3/2/11 | −0.12 [−0.38, +0.12] | +0.09 [−0.12, +0.31] |
| c1 → c3 | 16 | **−0.72 [−1.25, −0.25]**, 3/2/11 | **−0.38 [−0.75, −0.03]** | +0.03 [−0.28, +0.31] |
| c2 → c3 (H-D1) | 16 | −0.06 [−0.47, +0.34], 5/4/7 | +0.12 [−0.09, +0.34] | **+0.41 [+0.03, +0.78]** |
| c4 → c3 (H-D2) | 16 | −0.12 [−0.62, +0.34], 5/4/7 | −0.25 [−0.56, +0.03] | −0.06 [−0.41, +0.25] |
| c6 → c3 (H-C2) | 24 | +0.12 [−0.23, +0.48], 8/11/5 | +0.10 [−0.10, +0.31] | +0.15 [−0.12, +0.42] |

Bold: the CI excludes 0. **Every expansion condition is less faithful than raw text with the LoRA**, and the template and c3 also lose style. Among the expansion conditions, nothing separates on faithfulness.

**Default pipeline (B.3 rule, `eval/EVAL_REPORT.md` §2.1): `c1_lora_raw`.**
- **Rule:** the highest human faithfulness among the eligible conditions, ranked on the 16 scenes every candidate was rated on.
- **Eligible conditions:** style at most 0.3 below c1, and `max_train_dino` p95 ≤ 0.85. That leaves c1, c4 and c6.
- **Not eligible:** c2 and c3 miss on style.

**Two caveats:**
- **Cross-round comparisons:** c1 → c2/c3 and c4 → c3 compare conditions from different rating rounds, so they rest on absolute 1-5 ratings, not side-by-side judgement. The judge, the same instrument for every condition, points the same way (below).
- **Rater agreement** missed its bar (H-B5, §4.6), so the human results count as weaker evidence. The large differences survive this; the small ones would not be resolved at this n anyway.

**Automated metrics (supporting).** The full tables are in `eval/EVAL_REPORT.md` §2.2 (16-scene c2/c3/c4) and `expander/EXPANDER.md` §9 (24-scene c2/c3/c4). The headlines on the judge:
- **The LoRA changes style, not content.** c0 → c1 `style_margin` is +0.069 [+0.046, +0.095], better in 13 of 16 scenes. Checklist and `clip_raw` do not move.
- **No expansion condition beats raw text with the LoRA on the judge.** On the 16 shared scenes:
  - c1 → c2: −0.006 [−0.045, +0.033]
  - c1 → c4: −0.042 [−0.095, +0.015]
  - c1 → c3 (D's KB): −0.055 [−0.113, −0.003]
  - c1 → `c3_llm_kb_ckb`: −0.074 [−0.148, +0.012]
- **The LLM buys style over the template:** c2 → c3 `style_margin` is +0.019 [+0.009, +0.030] on 24 scenes, and +0.025 with C's KB on 16 scenes.
- **The KB makes no measurable difference** on any automated metric: c4 → c3, interim KB → C's KB, and grounded → LLM-written KB (c6 → c3) all show no clear difference.
- **Memorisation guardrail:** every condition passes (`max_train_dino` p95 0.767-0.807, limit 0.85).

No c5 or `@rrv_names` dev conditions exist.

### 4.2 Ablation, held-out
**Run.** Source: `eval/EVAL_REPORT.md` §4, `results/*_heldout.json`, `results/human_ratings_heldout.json`.
- **Images:** 12 unseen scenes × c0, c1, c2, `c3_llm_kb_ckb`, c4, c6 × seeds 1-2 = 144 images.
- **Pipeline:** made once on 16 Nov with the frozen pipeline (`main` at `64b7a2f`; seeds cut from 1-4, Section B deviation, 16 Nov). Each condition was made the way its rated dev version was.
- **Ratings:** rated blind by all four members on 17 Nov, two raters per image (seed 1). All six conditions were on the **same sheet**, so the cross-round caveat of dev does not apply here.
- **Use:** reported only; the B.3 rule is not re-applied.

| Condition | Faithfulness | Style | Quality | Judge checklist |
|---|---|---|---|---|
| `c0_base_raw` | 2.25 [1.88, 2.62] | 1.67 [1.29, 2.12] | 2.29 [1.92, 2.75] | 0.623 |
| `c1_lora_raw` | **2.54** [2.17, 2.96] | 3.21 [3.04, 3.38] | 2.54 [2.33, 2.75] | **0.667** |
| `c2_template` | 2.29 [1.96, 2.71] | 3.00 [2.71, 3.25] | 2.50 [2.29, 2.71] | 0.646 |
| `c3_llm_kb_ckb` | 2.04 [1.62, 2.50] | 2.92 [2.58, 3.21] | 2.46 [2.21, 2.71] | 0.598 |
| `c4_llm_nokb` | **2.54** [2.21, 2.92] | 3.25 [3.04, 3.46] | 2.62 [2.38, 2.88] | 0.642 |
| `c6_llm_kbllm` | 2.00 [1.58, 2.46] | 2.88 [2.62, 3.08] | 2.46 [2.21, 2.71] | 0.534 |

| a → b | Faithfulness, b − a [95% CI], W/T/L | Style | Judge checklist |
|---|---|---|---|
| c0 → c1 (LoRA) | +0.29 [−0.17, +0.75], 7/2/3 | **+1.54 [+1.08, +1.96]** | +0.044 [−0.017, +0.104] |
| c1 → c3 (full system vs raw) | **−0.50 [−0.83, −0.17]**, 1/4/7 | −0.29 [−0.62, 0.00] | −0.069 [−0.136, +0.001] |
| c1 → c2 (template vs raw) | −0.25 [−0.54, 0.00], 0/9/3 | **−0.21 [−0.42, −0.04]** | −0.022 [−0.065, +0.023] |
| c1 → c4 (LLM, no KB, vs raw) | 0.00 [−0.42, +0.42], 3/5/4 | +0.04 [−0.17, +0.25] | −0.025 [−0.108, +0.056] |
| c2 → c3 (H-D1) | −0.25 [−0.62, +0.08], 3/4/5 | −0.08 [−0.42, +0.17] | −0.048 [−0.101, +0.003] |
| c4 → c3 (H-D2) | **−0.50 [−0.88, −0.12]**, 1/5/6 | **−0.33 [−0.58, −0.08]** | −0.044 [−0.105, +0.011] |
| c6 → c3 (H-C2) | +0.04 [−0.25, +0.38], 3/6/3 | +0.04 [−0.12, +0.21] | **+0.064 [+0.011, +0.122]** |

Bold: the CI excludes 0.

**What held-out shows:**
- **The dev finding replicates:** raw text with the LoRA is more faithful than the full system (c1 → c3), and this time without the cross-round caveat.
- **The LoRA's style gain replicates** (+1.54).
- **What differs from dev:**
  - The LLM without the KB (c4) ties c1 instead of losing to it (dev: −0.59).
  - Adding the KB to the LLM **hurts** (c4 → c3 −0.50, CI excludes 0; dev: −0.12, null).
- **Where the KB effect comes from:**
  - Only 7 of 19 held-out character mentions are in C's KB `kb0.2`.
  - In h08 and h11 no character is in any KB, so c3, c4 and c6 got identical prompts and identical images. Without those two scenes the c4 → c3 effect is −0.60 [−1.05, −0.15] (B, #83).
  - The likely mechanism, as on dev (`EXPANDER.md` §11): KB appearance phrases take the token budget from the action and the scene. That is a plausible reading of 12 scenes, not a tested one.
- **Guardrail:** every condition passes the memorisation guardrail (`max_train_dino` p95 ≤ 0.801).

### 4.3 Expansion ladder and expander hypotheses
**Status 2026-11-16: decided on dev.**
- **H-D1 and H-D2** are decided by the D.0 rule on their primary measure, human faithfulness (`eval/EVAL_REPORT.md` §2.1; n = 16 scenes, seed 1).
- **The judge columns** are the supporting measure: 24 dev scenes × seeds 1-2 (c0/c1: first 16 scenes).
- **Sources:** `results/c2_template@rrv_v1_dev.json`, `c3_llm_kb@rrv_v1_dev.json`, `c4_llm_nokb@rrv_v1_dev.json` (D); `c0_base_raw@base_dev.json`, `c1_lora_raw@rrv_v1_dev.json` (B, #62); `c3_llm_kb_ckb@rrv_v1_dev.json` (C).
- **Method:** paired over scenes with `eval/stats.paired`; the full tables are in `expander/EXPANDER.md` §9.

| Hypothesis | Human faithfulness, b − a [95% CI] (primary) | Judge checklist, b − a [95% CI] (supporting) | Outcome (D.0 rule) |
|---|---|---|---|
| H-D1: c3 > c2 | c2 → `c3_llm_kb_ckb`: −0.06 [−0.47, +0.34], W/T/L 5/4/7 (n = 16) | c2 → c3: −0.023 [−0.074, +0.025] (n = 24) | **no clear difference**: no clear benefit of the LLM over a deterministic template at n = 16 |
| H-D2: c3 > c4, mostly tier 4 | c4 → `c3_llm_kb_ckb`: −0.12 [−0.62, +0.34], 5/4/7 (n = 16); tier 4 (n = 2): s15 3.0 → 1.5, s31 1.5 → 2.5 | c4 → c3: −0.005 [−0.052, +0.048] (n = 24); tier 4 (n = 3): −0.104 | **no clear difference** overall; the tier-4 part is **not resolvable** at n = 2 |
| *Held-out (reported, not used to decide)* | H-D1 c2 → c3: −0.25 [−0.62, +0.08]; H-D2 c4 → c3: **−0.50 [−0.88, −0.12]** (n = 12) | c2 → c3: −0.048; c4 → c3: −0.044 | H-D1 no clear difference again; on H-D2, held-out goes **against** the prediction: the KB made pictures less faithful |
| H-D3: c5 > c3 on tier 2-3 count/pose | — | — | not tested: cut (Overview §14; `EXPANDER.md` §10) |
| H-D4: 7B not needed | — | — | not tested: cut |
| H-D5: scene negatives | — | — | not tested: cut |
| H-D6: prompt v2 (SHAP order) | — | — | not tested: SHAP values arrived on 18 Nov, after the held-out freeze; their M4 recommendation (action first, no mood, no names) is in §4.4 |

**Secondary measures, as pre-registered.**

H-D1:
- **Style rating:** c2 → c3 +0.12 [−0.09, +0.34], no clear difference (as predicted). On the judge's `style_margin`, c3 is clearly above c2 (+0.019 [+0.009, +0.030]).
- **Quality:** c3 is rated higher than c2 (+0.41 [+0.03, +0.78]).
- **Failure tags** (share of each condition's ratings with that tag; `eval/EVAL_REPORT.md` §2.1):
  - "wrong figure count": 28% in c2 vs 21% in `c3_llm_kb_ckb`;
  - "wrong action": 25% in both.
  - So the predicted drop in c3 shows only for figure count, and only descriptively.
  - c3's ratings include s17-s24, which c2's do not.

H-D2:
- "wrong/missing figure identity" is 16% of c4's ratings against 23% of `c3_llm_kb_ckb`'s.
- "generic figure (name ignored)" is 3% against 4%.
- So the KB block shows no sign of fixing identity.

**Interpretation.** Both core expander hypotheses are null, and both nulls sit inside a larger negative result: every expansion is less faithful than raw text with the LoRA (§4.1). The expander does what it was built to do:
- valid JSON;
- no names;
- 70 tokens;
- more Varma-like phrasing than the template.

But on SD1.5 with this LoRA, that did not translate into pictures that match the story better. Likely reasons are in §5.

Also on the judge:
- c1 → c3 is −0.055 [−0.113, −0.003] (16 scenes), the same direction as the humans.
- D's interim KB vs C's KB under c3 makes no clear difference (−0.019 [−0.076, +0.040]).

**Caveats:**
- **Which c3 was rated.** c2 and D's c3 use D's interim KB `d-kb0` (`expander/interim_kb/`), because C's KB was not available (dated deviation, Section D). For human ratings, "c3" is C's `c3_llm_kb_ckb` (Section B deviation, 14 Nov). So:
  - H-D1 is decided on c2 (`d-kb0` template) vs `c3_llm_kb_ckb` (LLM with C's KB `kb0.2`), i.e. the comparison also changes the KB;
  - H-D2 is decided on c4 vs `c3_llm_kb_ckb`.

  D's own `c3_llm_kb` was not rated. On the judge, the two c3 versions are not clearly different, and c2 → `c3_llm_kb_ckb` (−0.042 [−0.110, +0.030]) agrees with c2 → c3 in direction.
- **Smaller n than planned.** H-D1/H-D2 were pre-registered on 32 scenes; the human comparisons have 16, and the H-D2 tier-4 split has 2 (s15, s31). The pre-registered interaction CI (tier 4 minus tiers 1-3) is not computed: with 2 scenes in a stratum, a bootstrap within it cannot give a meaningful interval.

**Expander validity (D.7), from `check_expansions.py --strict` on the committed files:**

| file | n | valid | leaks before repair | leaks after | fallback | s/scene (excl. load) |
|---|---:|---:|---:|---:|---:|---|
| `c2_template_dev.json` | 24 | 100% | 0% | 0 | 0% | 0.003 (CPU) |
| `c3_llm_kb_dev.json` | 24 | 100% | 8.3% | 0 | 0% | mean 9.7 (first 16 scenes) |
| `c4_llm_nokb_dev.json` | 24 | 100% | 70.8% | 0 | 0% | mean 13.3 (first 16 scenes) |

Both D.7 targets are met (≥ 95% valid, zero names after repair).

### 4.4 SHAP attribution
**Sources:**
- `analysis/shap/SHAP_ANALYSIS.md`, `results/shap@rrv_v1_dev.json`, `analysis/shap/shap_values_dev.json`, chart `docs/figures/shap_dev.png`;
- the token budget: `pipeline/TOKEN_BUDGET.md`, `results/token_budget_summary_dev.json`.

**Run.** 8 dev scenes (2 per tier, 1-4), five name-free players (`char`, `action`, `setting`, `mood`, `tail`), all 32 coalitions × seeds 1-2 = 512 images, scored with the judge. Exact Shapley values per scene; the efficiency check passed for every scene and metric. An earlier version of the SHAP document had numbers with no images behind them (`eval/CLAIMS_AUDIT.md` A1). They were removed, and only the 18 Nov run is cited here.

| Player | Mean tokens | φ faithfulness (`checklist`) | φ style (`style_margin`) | Faithfulness per token |
|---|---|---|---|---|
| `char` | 16.9 | +0.066 [+0.033, +0.100] | **+0.038** [+0.023, +0.054] | +0.0039 |
| `action` | 12.8 | **+0.230** [+0.158, +0.310] | +0.004 [−0.008, +0.016] | **+0.0181** |
| `setting` | 9.0 | +0.061 [+0.004, +0.116] | −0.050 [−0.062, −0.038] | +0.0068 |
| `mood` | 7.3 | −0.007 [−0.026, +0.018] | −0.041 [−0.052, −0.030] | −0.0010 |
| `tail` | 9.0 | −0.043 [−0.072, −0.017] | +0.015 [+0.009, +0.022] | −0.0048 |

- **Interactions:** `char × setting` are complements (faithfulness +0.041 [+0.008, +0.071], style +0.070 [+0.043, +0.098]).
- **Token budget (112 images, same 8 scenes):** faithfulness rises with content up to character + action + setting (0.357 → 0.599 → 0.672, about 46 tokens). Adding mood and tail then lowers it (−0.070 [−0.113, −0.024], exploratory). Adding canonical names lowers it as well (−0.056 [−0.107, −0.012], 6 of 8 scenes). `max_prompt_tokens` stays 70.

| Hypothesis | Result |
|---|---|
| H-A1: style falls with length | not supported (`char` → `full` style −0.000; faithfulness +0.244) |
| H-A2: trigger first gives more style | not supported (+0.002 [−0.008, +0.013]) |
| H-A3: the tail helps style, may cost faithfulness | not supported (both CIs include 0) |
| H-A4: `char` and `action` lead faithfulness | partly supported: `action` is clearly first; `char` is not separable from `setting` |
| H-A5: `tail` > 0 for style, ≈ 0 or < 0 for faithfulness | supported (faithfulness clearly negative) |
| H-A6: names add ≈ 0 once descriptions are present | not supported: names lowered faithfulness |

**How this connects to the ladder.** These are judge-based numbers on 8 dev scenes. They agree with the human result in §4.1-§4.2 and with the expander failures in `expander/EXPANDER.md` §11:
- The **action** phrase carries most of the faithfulness.
- Appearance, mood and tail phrases add little faithfulness or cost it.

The expansions spend much of the 70-token budget on appearance from the KB, and the budget sometimes drops the action. Raw user text is mostly character + action, which is why c1 held up so well.

### 4.5 Model-side fix and KB grounding
Sources: `train/NAMES_MODEL.md`, `kb/KB.md`, `results/names_test_*.json`, `results/c6_comparison.json`, `kb/parser_metrics.json`.

**Parser (H-C1, supported):** precision 1.00, recall 1.00, epic accuracy 1.00 on the 24 dev scenes (thresholds 0.90 / 0.90 / 0.85). C's caveat: the KB was built mainly from the dev-scene characters, so this shows the parser works when the KB has the character. **On held-out** (B's audit, `eval/CLAIMS_AUDIT.md` §8; `kb0.4`, 12 scenes): precision 1.00, **recall 0.42** (8 of 19), epic accuracy 0.58. 9 of the 11 misses are characters not in the KB, and 2 are implied rather than named. So H-C1 holds on dev and fails on held-out: the parser never invents a character, but the KB covers too few characters for unseen scenes.

**KB grounding (H-C2, not supported at this n):**
- c6 (LLM-written descriptions) → `c3_llm_kb_ckb` (image-grounded): checklist +0.013 [−0.045, +0.068], `style_margin` −0.001 [−0.012, +0.011], n = 24.
- **Human faithfulness** (c6 was rated in round 2 and on held-out; `eval/EVAL_REPORT.md` §2.1, §4.1):
  - dev +0.12 [−0.23, +0.48] (n = 24);
  - the 14 strong-coverage scenes +0.07 [−0.18, +0.36];
  - held-out +0.04 [−0.25, +0.38] (n = 12).
- No clear difference anywhere on human ratings. The judge prefers the grounded KB on held-out (+0.064 [+0.011, +0.122]), and humans do not.

**Model-side fix:**
- **Audit:** GO (67% of images with characters filled; 11 characters in 3 or more images).
- **Training:** `C_names` is the M2 winner's recipe (rank 16, LR 2e-4, seed 42, 800 steps) on `train_v3`, i.e. `train_v1` with epic names in 70 of 147 captions.
- **Guardrails:** passed against the winner (style_bleed, holdout-loss rise +0.00005, `max_train_dino` p95 0.816).
- **Names test** (12 characters, name prompt, seeds 1-4, judge), gain of `C_names` over `rrv_v1` by group:

| Group | Gain | Prediction |
|---|---|---|
| characters with 3 or more training images | −0.005 | ≥ +0.10 |
| characters with 1 training image | +0.009 | — |
| controls with no training image | −0.006 | — |

**H-C3 not supported.** C's answer to "did it learn names or memorise paintings?": neither. No group gained, and no image reached the 0.85 memorisation flag (`C_names` p95 0.823 vs `rrv_v1` 0.840 on the same prompts). `C_names` was **not registered** as `rrv_names`. The 2×2 (`c1`/`c3` × `rrv_v1`/`C_names`) was **not run**, because there was no gain and `pipeline/run_batch.py` could not run.

### 4.6 Judge validation
**Judge vs humans (H-B1, H-B2) and rater agreement (H-B5)** (`eval/EVAL_REPORT.md` §3.1-§3.2): 112 rated images, each paired with its own automated scores (matched by file name; 0 unmatched).

| Pair | Spearman ρ [95% CI] | Bar | Outcome |
|---|---|---|---|
| checklist judge vs human faithfulness | 0.41 [0.24, 0.56] | ≥ 0.4 and ≥ `clip_raw` | **H-B1 met, narrowly** |
| `clip_raw` vs human faithfulness | 0.24 [0.04, 0.41] | — | — |
| `style_margin` vs human style | 0.34 [0.16, 0.50] | > 0 | **H-B2 met** |
| rater vs rater, faithfulness | 0.43 [0.26, 0.58] (weighted κ 0.50; 92% within one point) | ≥ 0.5 | **H-B5 not met** |

**What this means:**
- **The judge** is usable as a secondary faithfulness measure, condition by condition, not image by image. That includes SHAP's faithfulness chart.
- **The human ratings** are the primary evidence. Because agreement missed its bar, they are reported as weaker evidence, with their spread.
- **The judge agrees with the mean human score** about as well as two humans agree with each other (0.41 vs 0.43).

**Robustness** (`eval/EVAL_REPORT.md` §3.3, `outputs/eval/judge_robustness_dev.json`, 20 images):
- **Question order shuffled** (pre-registered, threshold < 0.05): mean absolute change 0.0005. **Passed.**
- **Answer instruction reworded** (descriptive extra): scores drop by 0.033 on average while the ranking is kept (Spearman ρ = 0.98). So only differences between conditions are meaningful, not absolute judge values.

## 5. Analysis

**Inputs:**
- B: `EVAL_REPORT.md` (failure taxonomy counts, seen vs unseen, coverage, cost);
- A: `SHAP_ANALYSIS.md` §3-§4;
- C: `KB.md` (coverage table, known gaps);
- D: `EXPANDER.md` §11 (failure modes).

**Feeds from `results/`:** per-tier and seen/unseen splits of the dev and held-out results; `latency_s` from the expansion files; seconds per image from the manifests.

- **What did not help (said first, per B's rule).** **Prompt expansion, in every form we tried, made pictures less faithful to the story than the user's own words with the LoRA:**
  - c1 → c2 −0.66 [−1.19, −0.19];
  - c1 → c4 −0.59 [−1.00, −0.19];
  - c1 → c3 −0.72 [−1.25, −0.25].

  These are human faithfulness on 16 scenes (§4.1), and c3 is worse in 11 of 16 scenes. The judge points the same way (c1 → c3 −0.055 [−0.113, −0.003]). Neither the LLM over the template (H-D1) nor the KB block (H-D2) nor image-grounded descriptions over LLM-written ones (H-C2) made a clear difference. The template and c3 also cost style (−0.50 and −0.38).

  **Held-out confirms it** (§4.2): c1 → c3 −0.50 [−0.83, −0.17] on 12 unseen scenes rated on one sheet. The LLM alone (c4) ties raw text there, and adding the KB on top of it makes pictures less faithful (c4 → c3 −0.50 [−0.88, −0.12]). The most consistent single finding of M3 is that **the user's own short text, with the LoRA, is as good as or better than any rewritten prompt we produced**.
- **What helped:**
  - **The LoRA.** c0 → c1 style +1.38 [+1.06, +1.69] and quality +0.38 [+0.12, +0.62] on human ratings. Faithfulness +0.56 [0.00, +1.09], better in 12 of 16 scenes, with the CI just reaching 0.
  - **The KB block** kept names out of the LLM's phrases: 2 of 24 c3 records needed a names re-ask, against 17 of 24 in c4. This made the expander cleaner, not the pictures better.
  - **The LLM over the template:** better image quality (+0.41 [+0.03, +0.78]) and more Varma-like phrasing on the judge (`style_margin` +0.019 [+0.009, +0.030]), but not better faithfulness.
- **Why expansion may have hurt.** These are hypotheses for M4, not tested here:
  - **Specific words get replaced.** SD1.5's CLIP text encoder responds to concrete, frequent words. The user's text ("Hanuman kneeling before Rama") gives it short, specific tokens. Expansion replaces names with descriptions ("the monkey-faced devotee kneeling before the blue-skinned prince"), adds setting and mood phrases, and appends a style tail. Each phrase competes for the encoder's attention within 77 tokens.
  - **c1's top failure is the gentlest one.** Its most common tag is "generic figure (name ignored)" (8 of 32 ratings). The figure is there, just not recognisably the character.
  - **The expanded conditions fail harder.** Their top tags are "wrong figure count", "wrong action" and "wrong/missing identity" (c3: 33 of 48; c2: 25 of 32; c4: 22 of 32). The longer prompt seems to lose the scene's structure (who is there, doing what) rather than add to it.
  - **The template** also drops 4.5 phrases per record to fit 70 tokens (`EXPANDER.md` §9), so crowded scenes lose their figures and actions.
- **Failure taxonomy** (human tags; one tag per rating, the main failure; `eval/EVAL_REPORT.md` §2.1):

| Condition | ratings | wrong count + wrong action (composition) | identity + generic figure + setting | weak style + anatomy |
|---|---:|---:|---:|---:|
| `c0_base_raw` | 32 | 7 (22%) | 11 (34%) | 14 (44%) |
| `c1_lora_raw` | 32 | 8 (25%) | 16 (50%) | 8 (25%) |
| `c2_template` | 32 | 17 (53%) | 12 (38%) | 3 (9%) |
| `c4_llm_nokb` | 32 | 17 (53%) | 12 (38%) | 3 (9%) |
| `c3_llm_kb_ckb` | 48 | 22 (46%) | 21 (44%) | 5 (10%) |
| `c6_llm_kbllm` | 48 | 22 (46%) | 25 (52%) | 1 (2%) |

  How the failures shift along the ladder:
  - **The LoRA** turns style failures (c0) into identity failures (c1).
  - **Expansion** turns c1's identity failures into composition failures.
  - Example images with decoded codes: `eval/EVAL_REPORT.md` §5 (7 comparison grids chosen by rule, `docs/figures/grid*.png`, #83). Expansion-stage failure modes are in `expander/EXPANDER.md` §11.
- **Coverage, seen vs unseen** (judge, descriptive; `eval/EVAL_REPORT.md` §2.2):
  - Every condition scores higher on the 5 seen scenes than on the unseen ones.
  - Tier 4 (rare characters, spelling variants) is the lowest tier for most conditions.
  - **Human ratings, seen vs unseen (H-B3, descriptive):** c1 → c3 is −1.10 [−1.30, −1.00] on the 5 seen scenes (worse in all 5) and −0.55 [−1.32, +0.09] on the 11 unseen scenes. The predicted pattern (expansion helping more on unseen scenes) did not appear (`eval/EVAL_REPORT.md` §2.1).
  - **Coverage (H-B4, descriptive):** not supported. Scenes whose worst-covered character has low or no KB coverage are rated no lower than high/medium ones, and higher in 5 of 6 conditions. The groups have 7-14 scenes and are tangled with who wrote which scenes.
- **Cost:** generation takes about 5 s per image on a T4 (§1; A's run records). Loading the diffusion model takes 53 s on the first run of a session and about 9 s cached (`pipeline/PIPELINE.md` §5). Expander on a Colab T4 (`EXPANDER.md` §12):
  - LLM load: 23-33 s with cached weights, about 2-3 min on a fresh runtime;
  - about 7-8 s per LLM call; 9.7 s per scene for c3 and 13.3 s for c4, including name re-asks;
  - peak VRAM 2.8 GiB, freed before diffusion;
  - template c2 is instant on CPU.

## 6. Limitations

**Inputs:** all four owners' "limits" sections. **Feeds from `results/`:** sample sizes and the judge-validation numbers, cited where used.

These are known now and hold whatever the results show; owners add their own.
- **Small n.** 24 dev scenes (16 for c0/c1) and 12 held-out scenes; tier subsets have 3-7 scenes, and only 5 dev scenes are seen in training. Most CIs are wide. "No clear difference" is not evidence of no effect.
- **Proxy metrics.** The checklist judge, `clip_raw` and `style_margin` are proxies. CLIP shares an encoder family with the generator's text encoder. The judge is trusted only as far as B's validation shows.
- **Raters knew the project.** All raters are team members who wrote the scenes and built the conditions. Blind codes hide the condition, but not the intent of the project.
- **77 tokens.** CLIP reads at most 77 tokens. Expansions are fitted to 70 (excluding start/end), so long scenes lose detail. The raw text in c0/c1 and in `clip_raw` is truncated, which matters most for tier-5 texts with 100+ words.
- **One model.** One base model (SD1.5), one LoRA (`rrv_v1`, trained with one seed), one sampler setting and one small local LLM family. Nothing here is shown to transfer to other models.
- **Style tail.** The model was **not trained with the style tail** we append: the winner's train_v1 captions have no fixed tail. The measurements:
  - In SHAP, the tail adds a little style (φ +0.015) and costs faithfulness (φ −0.043).
  - In the token budget, switching it on or off makes no clear difference to either (`pipeline/TOKEN_BUDGET.md`).
  - The style comes from the LoRA, not from the prompt.
- **Name scrub.** It removes every listed name, including rivers and mounts that double as setting words (Ganga, Nandi), so a setting phrase can be lost. Measured on dev: **0 phrases scrubbed** in c2, c3 and c4. The one re-ask removed every leaked name first.
- **Interim KB.** D's c2 and c3 dev results use D's interim KB `d-kb0` (20 characters, written from train_v1 captions; 2 entries without a training painting), not C's KB. C's KB version of c3 exists (`c3_llm_kb_ckb`) and is not clearly different on the judge.
- **Interim KB coverage on held-out.** `d-kb0` covers only the dev-scene characters, so held-out scenes with other characters get no description for them in c2 (the parser writes "the figure"). Of the 11 held-out scenes that name characters, 9 include one missing from `d-kb0`, and 5 have none covered. With C's `kb0.2` (used by c3), 9 of 11 also miss one, and 4 have none covered (`eval/CLAIMS_AUDIT.md`, M8; checked against `eval/test_scenes_heldout.json`).
- **Parser in the held-out LLM runs.** `rapidfuzz` is not in `expander/requirements_llm.txt`, so the held-out c3/c6 expansions fell back to D's stand-in parser (exact alias match) instead of C's fuzzy parser (`parser` field in `outputs/expansions/*_heldout.json`). Both find the same characters in 11 of 12 held-out scenes. The exception is h12, where C's parser would also have found a misspelled Hanuman, so h12's c3/c6 prompts lack his description.
- **Smaller dev set.** The dev analysis uses 24 of the planned 32 scenes (s09-s32), and c0/c1/c2/c4 were rated on only 16 of them. A's 8 scenes (s01-s08, #77) arrived on 16 Nov, after the dev results and the B.3 decision, and were rewritten to use characters known to the interim KB. So they are outside the pre-registered analysis and descriptive only (Section B deviation, 16 Nov).
  - Tier subsets have 2-7 scenes.
  - The H-D2 tier-4 test has n = 3 on the judge and n = 2 on human ratings.
- **c3 in the human ratings is C's.** D's `c3_llm_kb` (interim KB) was not rated, so H-D1 on humans compares a `d-kb0` template with an LLM using C's KB. On the judge, the two c3 versions are not clearly different.
- **Self-made benchmark.** Checklists were written by the team, and the dev set was also used to tune prompts and the KB (allowed by our protocol). Only the held-out set is untouched.
- **Judge scale (B).** The 2B judge's absolute scores shift by about 0.03 when the answer instruction is reworded. Only relative scores are meaningful.
- **Rating rounds (B).** Round 1 and round 2 were rated on separate sheets, so cross-round comparisons (e.g. c1 → c3) rest on absolute 1-5 ratings. The round-1 key was rebuilt from the images rather than saved.
- **KB gaps (C):**
  - 41 epic characters with training images have no KB entry (they are scrubbed but not described);
  - Karna, Shabari and the battlefield are not grounded in paintings;
  - several characters' descriptions come from a single painting;
  - some looks overlap (Radha and Yashoda; Draupadi and Savitri).
- **Names LoRA (C).** It was trained late with one seed and 800 steps, and judged only by the 2B judge with checklists written from the KB descriptions. No human ratings.
- **Generator (A).** From 3 Nov to 16 Nov, `pipeline/generate_api.py` on `main` had no `generate()`, and its `text_to_image` did not load the LoRA. All scored dev images were made with the earlier generator: A's runs before 3 Nov, D's runs with the `e9b5a5d` version, C's `kb/gen_c6.py`. #75 (16 Nov) restored `generate()` on top of the unmodified M2 kit. `TODO(A)`: confirm LoRA scale 1.0 for c0/c1.
- **Config changes on 16 Nov.** #75 changed three things in `config/m3.json`:
  - the `negative` prompt;
  - the `base_model` id (`runwayml/…` → `stable-diffusion-v1-5/stable-diffusion-v1-5`);
  - a new `safety_checker` flag.

  #79 restored the negative to "ugly, blurry, bad anatomy" (CHANGELOG 1.3), the value every dev image and committed expansion record used. The held-out run uses that same negative (Section B deviation, 16 Nov), so dev and held-out share the pipeline. A regenerated c0/c1 on 16 Nov with the other negative. The rated 2 Nov files are kept in `results/rated/`, so human ratings stay paired with the images that were rated.
- **SHAP and token budget (A).**
  - 8 dev scenes and 2 seeds, so the per-tier values (2 scenes each) are descriptive only.
  - Faithfulness is the 2B judge, with no human ratings of the SHAP images.
  - Each player is one hand-written phrase in a fixed order, so other wording or another order could change the values.
  - In the token budget, length and content grow together.

## 7. Handoff to M4

**Inputs:** A's `docs/HANDOFF_M4.md`; B's failure taxonomy; D's `layout` field (if c5 was built). **Feeds from `results/`:** failure tag counts by type.

- **Interface for M4:** `text_to_image(text, seed, mode)` returns `(image, ExpandedPrompt, SceneSpec)`, and `generate(..., control_image=None)` already has a slot for ControlNet. SD1.5 is compatible with ControlNet 1.x (OpenPose, Canny). Everything is 512×512 only.
- **Composition failures** (human failure tags, §5; `eval/EVAL_REPORT.md` §2.1):
  - **Pose or composition problems** (wrong figure count, wrong action) are ControlNet territory: 46-53% of the main failures in every expanded condition, against 25% in c1.
  - **Identity and setting problems** (wrong/missing identity, generic figure, wrong setting) are 50% of c1's failures. This is what a names-aware model or better descriptions would have to fix. The prompt-side KB did not fix it on dev (H-D2), and neither did `C_names` (H-C3).
  - **For M4:** start from c1 (raw text + LoRA). Add structure through the image (OpenPose from the `layout` field) rather than through longer prompts.
- **`layout`:** c5 was not run, so whether layout phrases help is untested (H-D3). The expander already emits a validated `layout` JSON in `llm_kb_layout` mode (`n_figures`; per figure a role, a pose from a fixed vocabulary of 8, and a position from 5). That is the bridge to an OpenPose skeleton in M4.
- **Default pipeline and known issues:** the default is `c1_lora_raw` (`docs/TEAM_DECISIONS_M3.md` D1); `text_to_image(text, seed)` now defaults to `mode="raw"` and loads the LoRA (#84, #86). For many scenes, M4 should use `expander/batch_expand.py` + `pipeline/run_batch.py`, which is how every evaluated image was made. The full list is in `docs/HANDOFF_M4.md` §4. For prompt content, SHAP and the token budget point the same way: action first, then setting and character, no mood, no names; the tail is a style option.
- **Cold-start test (D, 18 Nov): passed after a one-line fix; two findings.**
  - **Setup:** a fresh Colab T4 runtime. D followed only `README.md`, opened `pipeline/demo.ipynb` from GitHub on `main` (after #84-#90) and ran all cells.
  - **Finding 1, blocking (A, setup):** section 2 failed with `ModuleNotFoundError: No module named 'rapidfuzz'`.
    - `text_to_image` imports C's `kb/parser.py` in every mode, including `raw`, and the parser needs `rapidfuzz`.
    - `pipeline/setup_session.sh gen` installs `m2kit/requirements_m2.txt`, which does not include it.
    - So a new user cannot make one image by following the README.
    - **Fix:** add `rapidfuzz` to the `gen` (and `llm`) install. With one added cell, `!pip install -q rapidfuzz`, everything after it ran.
  - **Results after the fix:**
    - three scenes, with the parser finding Sita, Hanuman + Rama, and Krishna (first image 57.2 s including download and model load; then 4.5 s each);
    - **same seed → identical image** (max pixel difference 0, `reproducible`);
    - base vs `rrv_v1` on the same seed (generic framed painting vs the Varma look);
    - the CLI saved an image (Yashoda, 31 tokens, 20.3 s including model load, negative `ugly, blurry, bad anatomy`);
    - the template mode ran.
  - **Finding 2, demo only (C/D, template mode):** through `text_to_image`, the template prompt reads "kneeling before and offering him 's ring, reaching out to take it".
    - C's `kb/parser.py` cuts names out of the action instead of replacing them with a role.
    - The evaluated c2 does not have this problem: it used `d-kb0` with D's role-replacing parser (`expander/EXPANDER.md` §11).
    - It affects only the non-default template mode in the demo. **For M4:** make C's parser return the role-replaced action, as D's stand-in does.
  - **Also seen:** in the raw (c1) Hanuman image, Hanuman is drawn as an ordinary man. This is the known c1 failure "generic figure (name ignored)" (§5), not a setup problem.

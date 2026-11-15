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

Generation takes about 5 s per 512×512 image on a T4 once the model is loaded (mean `gen_seconds` in A's run records: c1 5.3 s, c4 5.0 s; `outputs/eval/c{1,4}_*_metrics.json`). `TODO(A)`: load time. `TODO(D)`: an example input → expansion → image triple, chosen after results (seed 1, dev scene).

## 2. Research question and what is new since the halfway version

**Inputs:** Overview v2 §1, §9 (D). **Feeds from `results/`:** none.

**Research question.** The M2 model knows the style but not the stories. **Which part of the system actually makes the picture match the story: the knowledge base, the LLM, the prompt structure, or the model itself? And at what cost to style?**

**What is new.** The halfway feedback asked for "the simplest thing possible". We kept the pipeline simple (no UI, one base model, one LoRA, local models only) and put the effort into answering the question with evidence:
1. a controlled **ablation ladder** c0-c6 (§3) that adds one component at a time;
2. a **Shapley (SHAP) attribution** of the prompt's parts, which measures which phrases buy faithfulness and which buy style (A);
3. a **model-side fix**: a LoRA retrained with names in the captions, gated by a coverage audit and compared with the prompt-side fix (C);
4. an evaluation we can defend: **blind human ratings** plus a **checklist judge validated against those ratings**, with **hypotheses pre-registered** in `experiments/HYPOTHESES.md` before any scored run (B).

**Model-side fix, outcome:**
- **Audit:** GO. 99 of 147 training images (67%) have characters filled, and 11 characters appear in 3 or more images (`train/NAMES_MODEL.md` §2; `kb/KB.md` §8 says 10, `TODO(C)`: reconcile).
- **The LoRA:** `C_names` was trained late, passed the M2 guardrails, and showed **no gain from names** on the pre-registered names test. It was not registered as `rrv_names` (§4.5).

## 3. Method

**Inputs:** B's `eval/EVAL_REPORT.md` §1 (setup); `experiments/HYPOTHESES.md` (all sections); A's `pipeline/TOKEN_BUDGET.md`; C's `kb/KB.md` (KB schema, coverage thresholds). **Feeds from `results/`:** none (method only).

**Test sets.** The dev set (`eval/test_scenes_dev.json`) was planned as 32 scenes, 8 written by each member, with at least one per tier. It has **24** (B s09-s16, C s17-s24, D s25-s32; A's 8 were never added). The tiers are:
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

**Choosing the default pipeline.** The default for M4 is chosen at S3 **on dev only**, by the rule fixed in Overview §10: highest mean faithfulness, subject to the style and memorisation guardrails; on a CI including 0, the simpler condition wins. `TODO(A)`: the decision as recorded in `docs/TEAM_DECISIONS_M3.md`.

**Hypotheses.** Pre-registered per owner in `experiments/HYPOTHESES.md` (sections A-D). Each section's merge timestamp must precede that owner's first scored run; `TODO(B)`: confirm this from git history during the Day-6 audit. Section D (expander), merged in #20 before any D condition was scored:
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
- **A:** `TODO(A)`.

## 4. Results

**Inputs:**
- B: `eval/EVAL_REPORT.md` §2-§4 (tables, paired comparisons, judge validation, grids);
- A: `analysis/shap/SHAP_ANALYSIS.md` (chart, interactions, value per token), `pipeline/TOKEN_BUDGET.md`;
- C: `train/NAMES_MODEL.md` (2×2) and `kb/KB.md` (c6, parser precision/recall);
- D: `expander/EXPANDER.md` (process metrics, optional experiments).

**Feeds from `results/`:** `results/<condition>@<model>_dev.json` and `results/<condition>@<model>_heldout.json` for c0-c6, the optional D conditions and `@rrv_names`; A's SHAP results; `outputs/expansions/*.json` via `check_expansions.py`.

### 4.1 Ablation, dev
**Human ratings (primary):** `TODO(B)`. These come from the final `eval/ratings.py` run on rounds 1 and 2: faithfulness, style and quality per condition, plus the paired comparisons.

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
`TODO(B, held-out)`: the same tables on 12 scenes × 4 seeds. Arrives Day 5 evening (see timing risk above).

### 4.3 Expansion ladder and expander hypotheses
**Status 2026-11-15: provisional, judge only.** Section D decides H-D1/H-D2 on human faithfulness, which is pending (round 2), so every outcome below is on the supporting measure (the checklist judge). Data: 24 dev scenes × seeds 1-2 (c0/c1: first 16 scenes).
- **Sources:** `results/c2_template@rrv_v1_dev.json`, `c3_llm_kb@rrv_v1_dev.json`, `c4_llm_nokb@rrv_v1_dev.json` (D); `c0_base_raw@base_dev.json`, `c1_lora_raw@rrv_v1_dev.json` (B, #62); `c3_llm_kb_ckb@rrv_v1_dev.json` (C).
- **Method:** paired over scenes with `eval/stats.paired`; the full tables are in `expander/EXPANDER.md` §9.

| Hypothesis | Comparison | Judge checklist, b − a [95% CI] | Outcome on the judge | Pre-registered outcome |
|---|---|---|---|---|
| H-D1: c3 > c2 | c2 → c3 (n = 24) | −0.023 [−0.074, +0.025] | no clear difference | `TODO(B)`: human faithfulness |
| H-D2: c3 > c4, mostly tier 4 | c4 → c3 (n = 24) | −0.005 [−0.052, +0.048]; tier 4 (n = 3): −0.104 | no clear difference; tier 4 not resolvable | `TODO(B)`: human faithfulness |
| H-D3: c5 > c3 on tier 2-3 count/pose | — | — | not run yet | — |
| H-D4: 7B not needed | — | — | not run yet | — |
| H-D5: scene negatives | — | — | not run yet | — |
| H-D6: prompt v2 (SHAP order) | — | — | not run: no SHAP values with a source file yet | — |

Also on the judge:
- c1 → c3 is −0.055 [−0.113, −0.003] (16 scenes), so expansion did not improve faithfulness over raw text with the LoRA.
- On `style_margin`, c3 is clearly above c2 (+0.019 [+0.009, +0.030]).
- D's interim KB vs C's KB under c3 makes no clear difference (−0.019 [−0.076, +0.040]).

**Caveat on the KB:** c2 and c3 use D's interim KB `d-kb0` (`expander/interim_kb/`), because C's KB was not available. This is a dated deviation in Section D. For human ratings, B rates C's `c3_llm_kb_ckb` as "c3", so H-D1/H-D2 will be decided on c2 (`d-kb0`) vs `c3_llm_kb_ckb` (C's KB) and c4 vs `c3_llm_kb_ckb`. On the judge, the two c3 versions are not clearly different.

**Expander validity (D.7), from `check_expansions.py --strict` on the committed files:**

| file | n | valid | leaks before repair | leaks after | fallback | s/scene (excl. load) |
|---|---:|---:|---:|---:|---:|---|
| `c2_template_dev.json` | 24 | 100% | 0% | 0 | 0% | 0.003 (CPU) |
| `c3_llm_kb_dev.json` | 24 | 100% | 8.3% | 0 | 0% | mean 9.7 (first 16 scenes) |
| `c4_llm_nokb_dev.json` | 24 | 100% | 70.8% | 0 | 0% | mean 13.3 (first 16 scenes) |

Both D.7 targets are met (≥ 95% valid, zero names after repair).

### 4.4 SHAP attribution
**Pending (A).** `analysis/shap/SHAP_ANALYSIS.md` lists Shapley values per player, but no scores file exists to source them yet:
- the SHAP images exist (128 images, 2 scenes);
- `outputs/eval/shap_metrics.json` holds image counts only;
- `analysis/shap/shapley.py` does not compute values from it yet.

Per this report's rule, the values are not quoted until their scores file is in `results/`. `TODO(A)`: the chart (players × {faithfulness, style}, CI whiskers), the interaction table and the H-A outcomes, from a scored SHAP run.

### 4.5 Model-side fix and KB grounding
Sources: `train/NAMES_MODEL.md`, `kb/KB.md`, `results/names_test_*.json`, `results/c6_comparison.json`, `kb/parser_metrics.json`.

**Parser (H-C1, supported):** precision 1.00, recall 1.00, epic accuracy 1.00 on the 24 dev scenes (thresholds 0.90 / 0.90 / 0.85). C's caveat: the KB was built mainly from the dev-scene characters, so this shows the parser works when the KB has the character. The held-out set is the real test.

**KB grounding (H-C2, not supported at this n):**
- c6 (LLM-written descriptions) → `c3_llm_kb_ckb` (image-grounded): checklist +0.013 [−0.045, +0.068], `style_margin` −0.001 [−0.012, +0.011], n = 24.
- No clear difference, including on the 14 strong-coverage scenes. Judge only; c6 has no human ratings.

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
**Judge vs humans (H-B1) and rater agreement (H-B5):** `TODO(B)`, from the final `eval/ratings.py` run.

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

- **What helped (provisional, judge and automated metrics only):**
  - The LoRA adds Varma style: c0 → c1 `style_margin` +0.069 [+0.046, +0.095].
  - KB descriptions keep names out of the LLM's phrases: 2 of 24 c3 records needed a names re-ask, against 17 of 24 in c4.
  - c3 is the most Varma-like condition on `style_margin`.
- **What did not help (provisional):**
  - On the judge, **raw text with the LoRA (c1) is as good as or better than every expanded condition**: c1 → c2 −0.006, c1 → c4 −0.042 and c1 → c3 −0.055 (16 scenes), the last with a CI just below 0. Per B's rule, this is said first.
  - The template drops many phrases under the 70-token budget (5.7 per record), so crowded and long scenes lose their figures and actions.
- `TODO(D)`: rewrite both bullets from human faithfulness once B's ratings for c2/c3 exist.
- **Failure taxonomy:** `TODO(B)`: counts per condition from the rating sheets, with 2-3 example images (codes decoded). Expansion-stage failure modes are in `expander/EXPANDER.md` §11.
- **Coverage, seen vs unseen** (judge, descriptive; `eval/EVAL_REPORT.md` §2.2):
  - Every condition scores higher on the 5 seen scenes than on the unseen ones.
  - Tier 4 (rare characters, spelling variants) is the lowest tier for most conditions.
  - `TODO(B)`: the same from human ratings (H-B3, H-B4).
- **Cost:** generation takes about 5 s per image on a T4 (§1; A's run records); `TODO(A)`: model load time. Expander on a Colab T4 (`EXPANDER.md` §12):
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
- **Style tail.** The model was **not trained with the style tail** we append: the winner's train_v1 captions have no fixed tail. `TODO(A)`: what the token-budget and SHAP results say about it.
- **Name scrub.** It removes every listed name, including rivers and mounts that double as setting words (Ganga, Nandi), so a setting phrase can be lost. Measured on dev: **0 phrases scrubbed** in c2, c3 and c4. The one re-ask removed every leaked name first.
- **Interim KB.** D's c2 and c3 dev results use D's interim KB `d-kb0` (20 characters, written from train_v1 captions; 2 entries without a training painting), not C's KB. C's KB version of c3 exists (`c3_llm_kb_ckb`) and is not clearly different on the judge.
- **Interim KB coverage on held-out.** `d-kb0` covers only the dev-scene characters, so held-out scenes with other characters get no description for them in c2/c3 (the parser writes "the figure"). `TODO(B, held-out)`: share of held-out scenes affected.
- **Smaller dev set.** 24 of the planned 32 dev scenes (A's 8 were not added; c0/c1 cover only the first 16), so tier subsets have 3-6 scenes. The H-D2 tier-4 test has n = 3.
- **Self-made benchmark.** Checklists were written by the team, and the dev set was also used to tune prompts and the KB (allowed by our protocol). Only the held-out set is untouched.
- **Judge scale (B).** The 2B judge's absolute scores shift by about 0.03 when the answer instruction is reworded. Only relative scores are meaningful.
- **Rating rounds (B).** Round 1 and round 2 were rated on separate sheets, so cross-round comparisons (e.g. c1 → c3) rest on absolute 1-5 ratings. The round-1 key was rebuilt from the images rather than saved.
- **KB gaps (C):**
  - 41 epic characters with training images have no KB entry (they are scrubbed but not described);
  - Karna, Shabari and the battlefield are not grounded in paintings;
  - several characters' descriptions come from a single painting;
  - some looks overlap (Radha and Yashoda; Draupadi and Savitri).
- **Names LoRA (C).** It was trained late with one seed and 800 steps, and judged only by the 2B judge with checklists written from the KB descriptions. No human ratings.
- **Generator (A).** Since 3 Nov, `pipeline/generate_api.py` on `main` has no `generate()`, and its `text_to_image` does not load the LoRA. All scored images were made with the earlier generator (A's runs before 3 Nov, D's runs with the `e9b5a5d` version, C's `kb/gen_c6.py`). `TODO(A)`: fix, and confirm LoRA scale 1.0 for c0/c1.
- `TODO(A)`: SHAP-specific limits (position effects, 2 scenes so far).

## 7. Handoff to M4

**Inputs:** A's `docs/HANDOFF_M4.md`; B's failure taxonomy; D's `layout` field (if c5 was built). **Feeds from `results/`:** failure tag counts by type.

- **Interface for M4:** `text_to_image(text, seed, mode)` returns `(image, ExpandedPrompt, SceneSpec)`, and `generate(..., control_image=None)` already has a slot for ControlNet. SD1.5 is compatible with ControlNet 1.x (OpenPose, Canny). Everything is 512×512 only.
- **Composition failures:** `TODO(B, D)`: which failures are pose or composition problems (wrong figure count, wrong action), i.e. ControlNet territory, and which are identity or setting problems that the KB and prompt can still fix.
- **`layout`:** c5 was not run, so whether layout phrases help is untested (H-D3). The expander already emits a validated `layout` JSON in `llm_kb_layout` mode (`n_figures`; per figure a role, a pose from a fixed vocabulary of 8, and a position from 5). That is the bridge to an OpenPose skeleton in M4.
- **Default pipeline and known failure modes:** `TODO(A)`: the default chosen by the Overview §10 rule (recorded in `docs/TEAM_DECISIONS_M3.md`, which does not exist yet) and A's known failure modes. Note for M4: `text_to_image` must load the LoRA before it can be the M4 entry point (§6, Generator).
- **Cold-start test (Day 6, D):** `TODO(D)`: pass or fail from a fresh runtime using only the README (demo on 3 inputs, CLI once, one image reproduced with the same seed), and the issues filed as `needs:A`.

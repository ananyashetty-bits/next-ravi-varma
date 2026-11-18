# M3 team decisions

The decisions that fix what M3 is and what M4 starts from. Each one gives its date, the rule or reason behind it,
and the file that backs it. Recorded by B on 18 Nov 2026.

## D1. Default pipeline for M4: `c1_lora_raw` (16 Nov)
**Decision:** raw user text with the LoRA. The prompt is `rrvarma style, <text>, oil painting, canvas texture,
classical indian art`, run on SD1.5 + `rrv_v1` with the settings in `config/m3.json`. In code it is
`text_to_image(text, seed, mode="raw")`.

**Rule:** the pre-registered rule from Overview §10, copied into `experiments/HYPOTHESES.md` B.3 before any
rating. It is applied to the **dev** human ratings only:
1. A condition is eligible if its mean style rating is ≥ c1's minus 0.3, and its `max_train_dino` p95 is ≤ 0.85.
2. Eligible conditions are ranked by mean human faithfulness.
3. If the paired CI between the top condition and the next-simplest eligible one includes 0, the simpler one is
   chosen (simplest first: c1 < c2 < c4 < c3 < c5/c6).

**Applied** (source: `results/human_ratings_dev.json` → `default_pipeline`, made by `eval/ratings.py` on 16 Nov).
Candidates are ranked on the 16 scenes every one of them was rated on (Section B deviation, 16 Nov):

| Condition | Faithfulness | Style | `max_train_dino` p95 | Eligible |
|---|---|---|---|---|
| `c1_lora_raw` | **2.56** | 2.97 | 0.805 | yes |
| `c2_template` | 1.91 | 2.47 | 0.789 | no (style) |
| `c4_llm_nokb` | 1.97 | 2.84 | 0.797 | yes |
| `c3_llm_kb_ckb` | 1.84 | 2.59 | 0.799 | no (style) |
| `c6_llm_kbllm` | 1.91 | 2.69 | 0.802 | yes |

c1 has the highest faithfulness and is also the simplest candidate, so the tie-break step never applies.
**Choice: `c1_lora_raw`.**

**Held-out check** (reported, never used to choose): c1 also has the highest held-out faithfulness, tied with c4
at 2.54. c1 → c3 is −0.50 [−0.83, −0.17] (`results/human_ratings_heldout.json`, `eval/EVAL_REPORT.md` §4).

**Frozen code:** the held-out run used `main` at `64b7a2f` (recorded in the held-out manifests), with
`config/m3.json` as of CHANGELOG 1.3.

## D2. Negative prompt stays `ugly, blurry, bad anatomy` (16 Nov)
Every dev image and every committed expansion record used this negative prompt. A config change on 16 Nov (#75)
replaced it with the M2 kit's negative. #79 restored it (CHANGELOG 1.3), so the held-out images were made with
the evaluated pipeline. The M2 negative is listed as an M4 candidate in `docs/HANDOFF_M4.md`.

## D3. Analysed dev set: s09-s32 (24 scenes) (16 Nov)
s01-s08 were added on 16 Nov (#77), after the dev ratings, the dev results and D1. They stay in
`eval/test_scenes_dev.json` but are outside the pre-registered dev analysis and are reported descriptively only
(Section B deviation, 16 Nov).

## D4. "c3" in human ratings is `c3_llm_kb_ckb` (14 Nov)
The LLM expansion with C's image-grounded KB (`kb0.2`). D's interim-KB run `c3_llm_kb` was not rated (Section B
deviation, 14 Nov). On the held-out set, c3 was made with the same `kb0.2` (`eval/RUNBOOK_B.md` §7).

## D5. Held-out seeds 1-2 instead of 1-4 (16 Nov)
This is the last item on the Overview §14 cut list, cut because of the deadline. It was decided before any
held-out image was generated (Section B deviation, 16 Nov).

## D6. What was cut or not registered
- **Optional expander conditions** `c5_llm_kb_layout`, `c3_7b`, `c3_neg` and `c3_v2` were not run (cut list;
  `expander/EXPANDER.md` §10). c5's `layout` output is built and validated, and is the bridge to M4.
- **Names-aware LoRA** `C_names`: it passed the M2 guardrails but showed no gain on the pre-registered names test
  (H-C3 not supported), so it was not registered as `rrv_names` (`train/NAMES_MODEL.md`).
- **SHAP attribution and the token-budget experiment** had no scored results as of the morning of 18 Nov, so no
  document could cite them. *Update, 18 Nov evening:* both were run and scored (112 + 512 images). Results are in
  `pipeline/TOKEN_BUDGET.md` and `analysis/shap/SHAP_ANALYSIS.md`, and only those numbers may be cited.
  `max_prompt_tokens` stays 70 (no config change).

## D7. Rated images are pinned (14-16 Nov)
Human ratings refer to specific images. c0, c1 and c4 were regenerated later, so the files that were actually
rated are kept in `results/rated/`. `eval/ratings.py` pairs each rating with the judge score of the exact image
rated, matched by file name (Section B deviations, 16 Nov).

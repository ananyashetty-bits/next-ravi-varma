# Claims audit (owner: B, 18 Nov 2026)

The plan's Day-6 rule: every claim in the team's documents must be backed by a file in `results/`, the ratings,
or a script. Anything that is not gets softened or removed by its owner. This audit checks every document on
`main` at `caa173c` (after the held-out reveal, #82). Each finding names the owner, its severity and a suggested
fix.

**Severity:** **must fix** = a claim with no data behind it, or one that contradicts the results.
**update** = true when written, but out of date now. **ok** = checked, no issue found.

## 1. Pre-registration (asked for in `docs/M3_REPORT.md` §3)
Each hypothesis section was committed before that owner's first scored result (git history of `main`):

| Section | First committed | Owner's first scored result | OK |
|---|---|---|---|
| A | 30 Oct 16:19 (`4279e78`) | none in `results/` (no token-budget or SHAP results) | ok (nothing scored) |
| B | 1 Nov 16:01 (`094ecbf`, #23) | c0/c1/c4 scored 4 Nov (`scored_at`), committed 14 Nov (#62) | ok |
| C | 4 Nov 12:30 (`c1edf4d`); names test pre-registered 12 Nov 04:07 (`07c3bfb`) | c6 scored 11 Nov (`5ff1969`, `ab11ef9`); names test 14 Nov (`201f864`) | ok |
| D | 1 Nov 08:48 (`576ef1b`) | c2/c3 scored 4 Nov 22:00 (`859ab42`) | ok |

The B.3 default-pipeline rule (Overview §10, copied into Section B) also predates every rating.

## 2. `eval/EVAL_REPORT.md` (B)
**ok.** Every table is pasted from `eval/ratings.py` or `eval/auto_tables.py`, both run on committed files:
- **Dev:** re-running `ratings.py` on the revealed sheets and keys in `eval/ratings/` reproduces
  `results/human_ratings_dev.json` exactly.
- **Held-out:** `results/human_ratings_heldout.json` comes from the same script.
- **Prose:** every number in the prose was checked against those tables.

## 3. `docs/M3_REPORT.md` (assembled by D)

**Checked and correct:**
- the dev tables;
- the failure-tag shares (c2 count 28%, action 25%; c3 21% / 25%; c4 identity 16%; c3 23%; composition 22-53%);
- the §5 counts (c3 33 of 48, c2 25 of 32, c4 22 of 32, c1 8 of 32 generic);
- the §1 example (s09 judge 0.92 for c3, 0.87 for c1);
- the judge validation and robustness numbers.

The SHAP section correctly refuses to quote unbacked values.

| # | Where | Finding | Severity | Fix |
|---|---|---|---|---|
| M1 | §4.2 | Held-out is still `TODO(B, held-out)`. | update | Fill it from `eval/EVAL_REPORT.md` §4: c1 → c3 −0.50 [−0.83, −0.17] (1/4/7); c4 → c3 −0.50 [−0.88, −0.12]; c1 → c4 0.00; c2 → c3 −0.25 [−0.62, +0.08]; c6 → c3 +0.04; judge ρ 0.65; agreement ρ 0.65. |
| M2 | §2 last bullet | "The held-out set has not been run yet." | update | Replace with the held-out headline: c1 → c3 replicates. |
| M3 | §3 Test sets | "A's 8 were never added" contradicts §6, which says they were added on 16 Nov. | update | Say they were added on 16 Nov, after the dev decision, and are outside the analysis. |
| M4 | §4.5 KB grounding | "Judge only; c6 has no human ratings." Wrong: c6 was rated in round 2 and on held-out. §4.1's own table has c6 → c3 +0.12 [−0.23, +0.48]. | **must fix** | Use the human numbers: dev +0.12 [−0.23, +0.48]; strong-coverage scenes +0.07 [−0.18, +0.36]; held-out +0.04 [−0.25, +0.38]. All are null, so the H-C2 outcome is unchanged. |
| M5 | §4.5 Parser | "The held-out set is the real test." That test is now done (§8 below): precision 1.00, **recall 0.42**, epic accuracy 0.58. | update | Add it: H-C1 holds on dev but not on held-out. |
| M6 | §4.6 | Dev only. | update | Add held-out: judge ρ 0.65 [0.46, 0.78] vs CLIP 0.17; agreement ρ 0.65, so H-B5 is met on held-out. |
| M7 | §5 | "Example images with decoded codes: `TODO(B)`." | update | Point to `eval/EVAL_REPORT.md` §5 (7 grids, `docs/figures/grid*.png`). |
| M8 | §6 "Interim KB coverage on held-out" | `TODO(B, held-out)` | update | 9 of the 11 held-out scenes that name characters include one missing from `d-kb0` (c2), and 5 have none covered. With `kb0.2` (c3): also 9 of 11, with 4 having none covered. |
| M9 | §6 Small n / §3 | Seeds for held-out | ok | Already says seeds 1-2. |
| M10 | §3 Hypotheses | `TODO(B)`: confirm the merge timestamps. | update | Confirmed, all four sections ok (§1 above). |
| M11 | §1, §6 | `TODO(A)` items: load time, style tail, LoRA scale. | A | See §5 below. |

## 4. `expander/EXPANDER.md` (D)
**ok** for dev: the §9 tables matched `eval/auto_tables.py` output on 14 and 16 Nov.

| # | Finding | Severity | Fix |
|---|---|---|---|
| E1 | Held-out rows pending. | update | Source: `results/*_heldout.json`, `outputs/expansions/*_heldout.json`. All 48 records are valid with 0 leaks after repair. Names were re-asked in 16.7% (c3), 33.3% (c4) and 41.7% (c6) of records. |
| E2 | Worth adding to §11: in held-out h08 and h11, no character is in any KB, so c3, c4 and c6 got **identical prompts and images**. The c4 → c3 held-out effect without those two scenes is −0.60 [−1.05, −0.15]. | update | Optional, but it explains where the KB effect comes from. |

## 5. A's documents

| # | Doc | Finding | Severity | Fix |
|---|---|---|---|---|
| A1 | `analysis/shap/SHAP_ANALYSIS.md` | No data file, script or run produced the φ table. The only SHAP data, `outputs/eval/shap_metrics.json`, reports `valid_images: 0` of 128 requested (plan: ~512). There is no `results/shap*.json`. The table's values sum to exactly **1.000**, but they are described as CLIP alignment differences. Real CLIP score differences between prompts are in the 0.01-0.05 range (`clip_raw` across conditions: 0.25-0.28), so this looks normalised or invented. There are no CIs. The value functions are not the planned ones (judge faithfulness and `style_margin`). The "(char, setting) synergy" and "tail < 6%" claims have no source. | **must fix** | Run SHAP and score it with `eval/score.py`, which handles the `coalition` field. Otherwise mark the whole document "not completed; values are placeholders", and remove §2-§5's numbers and recommendations. SHAP is on the Overview's "never cut" list, so the final report must say plainly whether it was done. |
| A2 | `pipeline/TOKEN_BUDGET.md` | Every cell is `TBD`; the core experiment was not run. `max_prompt_tokens` = 70 has no evidence behind it in M3. | **must fix** | Run it, or say "not run" and that 70 was carried over from M2/D's design (cite the source). |
| A3 | `docs/HANDOFF_M4.md` | It tells M4 to call `text_to_image(..., mode="llm_kb")`. That is the c3 pipeline, **less faithful** than c1 on dev (−0.72) and held-out (−0.50). The default pipeline by the pre-registered rule is **c1 (raw text + LoRA)**. The code block is also never closed. | **must fix** | Use `mode="raw"` (c1) as the M4 default, and cite `results/human_ratings_dev.json` and `docs/TEAM_DECISIONS_M3.md`. |
| A4 | `docs/TEAM_DECISIONS_M3.md` | Missing. The default pipeline (`c1_lora_raw`, B.3 rule, 16 Nov) is not recorded where the plan says it must be. | **must fix** | Create it: the decision, the rule, the eligibility table (`results/human_ratings_dev.json` → `default_pipeline`), the frozen sha `64b7a2f`. |
| A5 | `pipeline/PIPELINE.md` | 13-line stub with an unclosed code block. No run instructions, no timings, no load time (`M3_REPORT` §1 has `TODO(A)`). | update | Complete it, or point to `expander/RUNBOOK_COLAB.md` and `eval/RUNBOOK_B.md` §7, which document the working commands. |
| A6 | `README.md` | Missing, and D's cold-start test runs from it. | **must fix** | Minimal README: setup per profile, `text_to_image` example, CLI, the three runbooks. |
| A7 | LoRA scale for c0/c1 (2 Nov) | Still assumed to be 1.0 (Section B deviation, 4 Nov). A stated 1.0 for the 16 Nov run (#17), but not for the rated 2 Nov images. | update | One line on #71 or #17 confirming. |

## 6. `kb/KB.md` (C)

| # | Finding | Severity | Fix |
|---|---|---|---|
| K1 | §7 Limits: "no human ratings exist for c6". | update | c6 was rated (dev round 2 and held-out): see M4 above. |
| K2 | §8: "10 characters appear in 3 or more images". `train/NAMES_MODEL.md` and C's final count say 11. | update | Change to 11. |
| K3 | §7 table: checklist W/T/L 13/1/10 is computed with tie band 0, although `results/c6_comparison.json` declares `tie_band: 0.0295` (B.2). With 0.0295 it is 12/3/9. Diff and CI are unaffected. | update | Recompute with 0.0295, or label the column "tie band 0". |
| K4 | Parser precision/recall (H-C1) | ok on dev (1.00 / 1.00 / epic 1.00, `kb/parser_metrics.json`). The held-out result is in §8 of this audit. | update | Add the held-out line. |

## 7. `train/NAMES_MODEL.md` (C)
**ok.** The group gains agree with the per-character table: the 3+-image group means 0.600 → 0.595 equal the
means of the six characters' rows. The guardrail table and memorisation numbers are internally consistent, and
the 2×2 is correctly reported as not run.

## 8. New check: parser on the held-out set (H-C1)
C's documents note that "the held-out set is the real test". Running `kb/parser.py` (`kb0.4`) on the 12 held-out
scenes against their `expected_characters` (the same logic as `kb/eval_parser.py`, without writing
`parser_metrics.json`):

| Metric | Dev (24) | Held-out (12) | Pre-registered bar |
|---|---|---|---|
| Character precision | 1.00 | **1.00** | ≥ 0.90 |
| Character recall | 1.00 | **0.42** (8 of 19) | ≥ 0.90 |
| Epic accuracy | 1.00 | **0.58** | ≥ 0.85 |

Of the 11 missed characters:
- **9 are not in the KB:** Saraswati, Dushyanta, Nala, Shakuni, Vasuki, Kumbhakarna, Drona, Ekalavya, Jatayu.
- **2 are implied, not named:** h05 Hanuman, h06 Duryodhana.

The wrong epics are mostly "Unknown" in scenes where no character was recognised. So the parser never invents a
character, but the KB covers too few characters for unseen scenes: H-C1 holds on dev and fails on held-out.

## 9. Status update (18 Nov)
Resolved in #84:
- **A3:** `HANDOFF_M4.md` now starts M4 from c1 (`mode="raw"`), with the evidence.
- **A4:** `docs/TEAM_DECISIONS_M3.md` created (D1-D7).
- **A5:** `pipeline/PIPELINE.md` completed, with the measured timings (model load 53 s first / ~9 s cached).
- **A6:** `README.md` added.
- **CI:** `numpy` and `pillow` installed, so the stats, key and grid tests run.

- **A2:** token budget run and reported in `pipeline/TOKEN_BUDGET.md` (112 images, `results/token_budget*_dev.json`).

- **A1:** SHAP run, scored and reported in `analysis/shap/SHAP_ANALYSIS.md` (512 images, efficiency check passed); the old unbacked numbers are gone.

Still open: **A7** (LoRA scale for the 2 Nov images), and C's and D's updates.

## 10. Summary for owners

| Owner | Must fix | Updates |
|---|---|---|
| **A** | A1 SHAP_ANALYSIS (unbacked numbers), A2 TOKEN_BUDGET (not run), A3 HANDOFF default mode, A4 TEAM_DECISIONS missing, A6 README missing | A5, A7 |
| **C** | none | K1-K4 |
| **D** | M4 (c6 has human ratings) | M1-M3, M5-M8, M10, E1-E2 |
| **B** | none | none |

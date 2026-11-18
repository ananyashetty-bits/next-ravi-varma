# Token budget and prompt engineering

Tests hypotheses H-A1 to H-A3 and gives the image evidence for H-A6 (`experiments/HYPOTHESES.md` Section A).
The design notes are dated 18 Nov 2026, written before any image was generated.

## Design
- **Scenes:** the 8 SHAP dev scenes, 2 per tier from 1 to 4 (s09, s10, s12, s27, s13, s21, s23, s31). Each scene
  is written as five name-free player phrases of ≤ 14 words each (`analysis/shap/shap_scenes.json`).
- **Controlled content:** every variant is built from the same phrases (`pipeline/token_budget.py build` →
  `outputs/expansions/token_budget_dev.json`). Only length, trigger position, the style tail or the names change.
- **Seeds and model:** seeds 1-2 with `rrv_v1`, using the config negative. That gives 8 × 7 × 2 = **112 images**.
- **Metrics:** `checklist` (judge P(yes), faithfulness), `clip_raw` (CLIP vs the raw user text) and `style_margin`.
  Images are scored with `eval/score.py`.
- **Statistics:** the scene is the unit, and seeds are averaged first. Each contrast is a paired difference over
  the 8 scenes with a 95% bootstrap CI (`eval/stats.paired`, 5000 resamples).

| Variant | Prompt | CLIP tokens (min / mean / max) |
|---|---|---|
| `char` | trigger, char | 18 / 21.9 / 24 |
| `char_action` | trigger, char, action | 29 / 35.6 / 42 |
| `char_action_setting` | trigger, char, action, setting | 39 / 45.6 / 52 |
| `no_tail` | trigger, char, action, setting, mood | 48 / 53.9 / 60 |
| `full` | trigger, char, action, setting, mood, tail | 58 / 63.9 / 70 |
| `trigger_last` | char, action, setting, mood, tail, trigger | 58 / 63.9 / 70 |
| `with_names` | trigger, names, char, action, setting, mood, tail | 62 / 69.2 / 75 |

- **Length (H-A1):** the ladder is `char` → `char_action` → `char_action_setting` → `full`, about 22 → 64 tokens.
  The targets 15/30/50/70 are approximate, because the steps add whole phrases, so the measured counts are
  reported. Length and content grow together, as in a real expansion.
- **Trigger position (H-A2):** `full` vs `trigger_last`, with identical tokens.
- **Style tail (H-A3):** `full` vs `no_tail`.
- **Names (H-A6):** `full` vs `with_names`. Every prompt stays within CLIP's 75-token window, so nothing is
  truncated.

## Results
Pending: the generation and scoring runs are in `analysis/RUNBOOK_TOKEN_SHAP.md`. Tables come from
`python pipeline/token_budget.py analyse` (→ `results/token_budget_summary_dev.json`).

## Recommendation
Pending the results. All M3 evaluated runs used `max_prompt_tokens = 70`, so any different value found here is a
recommendation for M4. The M3 config stays unchanged, so that the evaluated results remain reproducible.

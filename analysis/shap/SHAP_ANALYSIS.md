# SHAP attribution: which parts of a prompt matter

Tests hypotheses H-A4 to H-A6 (`experiments/HYPOTHESES.md` Section A).

**Status, 18 Nov:** pending; the run is set up (`analysis/RUNBOOK_TOKEN_SHAP.md`). This file previously
contained attribution numbers that no scored images back (`eval/CLAIMS_AUDIT.md` §5). Those numbers were
removed, and nothing from them may be cited.

## 1. Game
- **Players:** five prompt parts, `char`, `action`, `setting`, `mood` and `tail`. Each is a name-free phrase of
  ≤ 14 words, written per scene in `analysis/shap/shap_scenes.json`.
- **Coalitions:** a coalition's prompt is the trigger plus its players in fixed order
  (`analysis/shap/coalitions.py`). The empty coalition is the trigger alone.
- **Value v(S):** the score of coalition S's images, averaged over seeds 1-2. The same seeds are used for every
  coalition (common random numbers).
- **Metrics:** `checklist` for faithfulness, `style_margin` for style, and `clip_raw` as a second faithfulness
  measure.
- **Scale:** 8 dev scenes (2 per tier from 1 to 4: s09, s10, s12, s27, s13, s21, s23, s31) × 32 coalitions ×
  2 seeds = **512 images**.
- **Computation:** exact Shapley values per scene (`analysis/shap/shapley.py`). The efficiency check
  (Σφ = v(all) − v(∅)) is asserted for every scene. The mean over scenes gets a 95% bootstrap CI over scenes.

## 2. Shapley values
Pending.

## 3. Interactions
`char×setting` and `tail×mood`, as Shapley interaction indices. Pending.

## 4. Value per token
Mean φ (faithfulness) divided by each player's mean CLIP token count. Pending.

## 5. Hypotheses
| Hypothesis | Test | Result |
|---|---|---|
| H-A4: `char` and `action` have the highest φ for faithfulness | Rank of mean φ (`checklist`), with CIs | pending |
| H-A5: `tail` > 0 for style, ≈ 0 or < 0 for faithfulness | φ_tail CIs on `style_margin` and `checklist` | pending |
| H-A6: names add ≈ 0 once descriptions are present | `with_names` − `full` (`pipeline/TOKEN_BUDGET.md`) | pending |

## 6. Limits
Pending.

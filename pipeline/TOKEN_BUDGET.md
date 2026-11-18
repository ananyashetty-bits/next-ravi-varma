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
All 112 images scored on 18 Nov 2026 with `eval/score.py` (judge `Qwen/Qwen2-VL-2B-Instruct`). Sources:
`results/token_budget@rrv_v1_dev.json` (per image) and `results/token_budget_summary_dev.json` (the tables below,
from `python pipeline/token_budget.py analyse`). The scene is the unit, with seeds averaged first. Means over 8
scenes are shown with 95% bootstrap CIs. **Memorisation guardrail:** `max_train_dino` p95 0.819 (≤ 0.85).

| Variant | Tokens (mean) | `checklist` | `clip_raw` | `style_margin` |
|---|---|---|---|---|
| `char` | 21.9 | 0.357 [0.299, 0.421] | 0.202 [0.188, 0.217] | 0.112 [0.098, 0.128] |
| `char_action` | 35.6 | 0.599 [0.513, 0.683] | 0.246 [0.230, 0.265] | 0.106 [0.086, 0.124] |
| `char_action_setting` | 45.6 | **0.672** [0.601, 0.753] | 0.263 [0.235, 0.291] | 0.094 [0.063, 0.124] |
| `no_tail` | 53.9 | 0.633 [0.555, 0.717] | 0.264 [0.243, 0.284] | 0.104 [0.084, 0.121] |
| `full` | 63.9 | 0.602 [0.494, 0.726] | 0.256 [0.235, 0.276] | 0.112 [0.085, 0.135] |
| `trigger_last` | 63.9 | 0.577 [0.450, 0.721] | 0.261 [0.242, 0.279] | 0.114 [0.094, 0.132] |
| `with_names` | 69.2 | 0.545 [0.417, 0.687] | 0.257 [0.234, 0.280] | 0.105 [0.087, 0.121] |

Paired differences (b − a). Win/tie/loss counts the scenes where b is higher / equal / lower.

| Contrast (a → b) | `checklist` | `clip_raw` | `style_margin` |
|---|---|---|---|
| Length: `char` → `full` | **+0.244** [+0.148, +0.347] (8/0/0) | **+0.054** [+0.031, +0.078] (8/0/0) | −0.000 [−0.028, +0.025] (5/0/3) |
| Length: `char_action` → `char_action_setting` | **+0.072** [+0.019, +0.126] (6/0/2) | +0.017 [−0.003, +0.043] (5/0/3) | −0.013 [−0.043, +0.015] (3/0/5) |
| Trigger: first (`full`) → last | −0.024 [−0.053, +0.003] (3/0/5) | +0.005 [−0.007, +0.015] (6/0/2) | +0.002 [−0.008, +0.013] (4/0/4) |
| Tail: off (`no_tail`) → on (`full`) | −0.031 [−0.096, +0.029] (4/0/4) | −0.008 [−0.016, +0.001] (3/0/5) | +0.008 [−0.005, +0.023] (5/0/3) |
| Names: `full` → `with_names` | **−0.056** [−0.107, −0.012] (2/0/6) | +0.000 [−0.019, +0.017] (4/0/4) | −0.007 [−0.023, +0.007] (3/0/5) |
| *Exploratory:* mood added (`char_action_setting` → `no_tail`) | −0.039 [−0.071, −0.008] (2/0/6) | +0.001 [−0.014, +0.014] | +0.010 [−0.005, +0.025] |
| *Exploratory:* mood + tail added (`char_action_setting` → `full`) | −0.070 [−0.113, −0.024] (2/0/6) | −0.007 [−0.023, +0.008] | +0.018 [+0.002, +0.034] (6/0/2) |

### Hypotheses
The pre-registered rule: a hypothesis is **supported** only if the 95% CI excludes 0 in the predicted direction.
"≈ 0" counts if the CI includes 0 and lies within ±0.0295.

| Hypothesis | Prediction | Result | Verdict |
|---|---|---|---|
| H-A1 (length) | `style_margin` falls slightly as the prompt grows from ~15 to ~70 tokens | `char` → `full`: −0.000 [−0.028, +0.025] | **Not supported.** Style did not change with length, while faithfulness rose by +0.244. |
| H-A2 (trigger position) | Trigger first gives clearly higher style | first → last: +0.002 [−0.008, +0.013] | **Not supported.** No measurable difference either way. |
| H-A3 (style tail) | The tail improves style slightly but may cost faithfulness at 70 tokens | style +0.008 [−0.005, +0.023]; checklist −0.031 [−0.096, +0.029] | **Not supported.** Both CIs include 0. |
| H-A6 (names, image evidence) | Names add ≈ 0 once descriptions are present | checklist −0.056 [−0.107, −0.012], 6 of 8 scenes lower | **Not supported.** Adding names *lowered* faithfulness. |

### What the numbers say
1. **Content drives faithfulness, not length.** Each scene part up to the setting raised the checklist score:
   - adding the action: 0.357 → 0.599
   - adding the setting: 0.599 → 0.672

   The mood phrase and the tail then lowered it: −0.070 from `char_action_setting` to `full` (exploratory). The
   peak is at about **46 tokens**, not at the 70-token limit.
2. **The style comes from the LoRA, not from the prompt.** `style_margin` stayed around 0.09–0.11 for every
   variant, with or without the tail, and with the trigger first or last. The only style gain with a CI above 0 is
   the exploratory +0.018 for adding mood and tail together.
3. **Names hurt.** Putting the canonical names in front of the visual descriptions lowered the checklist score in
   6 of 8 scenes. This agrees with C's names test (`train/NAMES_MODEL.md`): the model does not know the names, so
   they only take attention away from the description.

## Recommendation
- **`max_prompt_tokens`: keep 70. No config change.**
  - Every evaluated M3 run used 70, so changing it now would break reproducibility.
  - Nothing here suggests a larger budget would help, since faithfulness peaked at about 46 tokens.
- **For M4, spend the budget on scene content:**
  - Fill the character, action and setting first.
  - Add mood or tail only if there is room, and treat them as style options, not faithfulness options.
  - Never add names.
- **Keep the trigger first.** It made no measurable difference here, and it is the order every evaluated M3 prompt used.

**Limits:**
- 8 dev scenes and 2 seeds, so the CIs are wide.
- `checklist` is a 2B-model proxy for human judgement (ρ 0.41 dev / 0.65 held-out with humans, `eval/EVAL_REPORT.md`).
- Length and content grow together on the ladder.
- The mood phrases were written by us; a different wording might behave differently.

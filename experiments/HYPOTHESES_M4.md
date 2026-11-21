# Hypotheses M4

## Section A (A: control engine). Written before any sanity check or sweep was scored

### Sanity-check thresholds (pipeline/control_api.py; same session, same GPU)
1. Scale 0: 5 dev prompts x seed 1, M3 generate() vs generate_controlled(..., control_scale=0): mean |dpixel| <= 2/255 AND DINOv2 >= 0.98.
2. Effective LoRA scale: same test at LoRA 0.5 and 1.0 against M3 generate(..., lora_scale=...): same thresholds as (1). LoRA 0 vs LoRA 1.0 must differ: mean |dpixel| >= 10/255 [A's proposed number, fixed before running].
3. Fixture skeleton: at control 1.0 the figures clearly follow it (checked by eye on f01-f03, plus detector pose_pck if available).
4. Black control image at control 1.0: runs, no NaN, image is not blank (pixel std > 0.02) and looks sensible by eye.
5. Determinism: same inputs twice give identical arrays.
Comparisons are always against M3 images generated in the same session, never old M3 files.

### Tie bands
style_margin 0.0295; pose_pck 0.05.

### Default-setting rule (Overview section 10, rule 1, verbatim)
1. **Control scale, guidance window, LoRA scale (A, Day 2 by 12:30).**
   - Run 4-5 stage-2 candidates on all 16 dev pairs, seed 1 (at least 12 pairs if some are late, flagged in `CONTROL.md`), compared with **p0's seed-1 images**.
   - **Eligible** = `max_train_dino` p95 <= 0.85 **and** mean `style_margin` >= p0's - 0.0295.
   - Among eligible candidates, pick the highest mean `pose_pck`. If two candidates' paired CI includes 0, pick the lower control scale.
   - The LoRA scale stays 1.0 unless a candidate with another scale wins by this rule.

### Hypotheses
- H-A1: pose adherence rises with control scale while style_margin falls, with a knee around 0.6-0.8.
- H-A2: releasing control after 60-80% of the steps (guidance_end) keeps more style than lowering the scale, at similar pose adherence.
- H-A3: at strong control, a LoRA scale above 1.0 (e.g. 1.2) partly restores style, at some cost to prompt following.
- H-A4: p1 beats p0 on human pose match and figure count, but not on face or hand quality.
- H-A5 (optional p8): hand and face keypoints do not improve human face/hand ratings much either, because the limit is resolution, not the body-only conditioning.

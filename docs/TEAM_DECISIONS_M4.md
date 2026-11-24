# M4 team decisions

The decisions that fix the M4 defaults. Each one gives the rule (pre-registered in `experiments/HYPOTHESES_M4.md`,
Overview §10), the numbers it was applied to and the file behind them. All three rules are applied to the **dev**
set only; the held-out scores are not opened until this file is merged. Recorded by B on 24 Nov 2026.

Rating data: two blind rounds (sheet 1: p0, p1, p2; sheet 2: p3, p5, p6), 16 dev pairs, seed 1, 512×512 copies,
2 raters per image out of 4 (A, B, C, D), 192 ratings, all filled. Tables: `results/m4_tables_dev.md` and
`results/m4_tables_dev.json` (made by `eval/tables_m4.py`, #144). The blind keys stay in B's private Drive.

## D1. Control settings: `control.tuned` = control 1.0, guidance window 0-0.8, LoRA 1.0 (A, 23 Nov)
**Rule** (Overview §10 rule 1): among stage-2 candidates with `max_train_dino` p95 ≤ 0.85 and mean `style_margin`
≥ p0's − 0.0295, pick the highest mean `pose_pck`; if two candidates' paired CI includes 0, pick the lower control
scale; LoRA stays 1.0 unless another scale wins.

**Applied by A** on 16 dev pairs, seed 1, against p0 seed 1 (p0 `style_margin` 0.1101, floor 0.0806). All five
candidates were eligible. `c1.2_l1.0` had the top `pose_pck` (0.832), but its paired CI includes 0 against the
baseline [−0.016, +0.069] and against `c1.0_ge0.8` [−0.014, +0.066], so the rule takes the lowest control scale
among the three: **control 1.0, window 0-0.8, LoRA 1.0** (`results/control_rule_16pairs.json`,
`docs/CONTROL.md` §2; merged into `config/m4.json`, approved by D).

**What the ratings add** (not part of the rule): nothing. p1 and p2 images are near-identical (median pixel
difference 2.1 of 255 vs 39.1 for p0 vs p1; `results/m4_p1_p2_similarity_dev.json`), so their rating differences
(faces +0.25, others ≤ +0.16) are rater noise. The tuned window costs nothing and gains nothing.

## D2. Post-processing default: hi-res (p5) passes the rating test; face restoration (p6) is opt-in (S3)
**Rule** (Overview §10 rule 2): post-processing is on by default only if, against p1, the mean **face** rating
improves by more than 0.25, **hands** do not drop by more than 0.25 and the mean **style** rating drops by less
than 0.3. D then spot-checks the winning method on 4 of A's p2 images; a visible new artifact makes it opt-in.

**Applied** (paired means over 16 pairs, `results/m4_tables_dev.md`):

| p1 → | faces (> +0.25?) | hands (≥ −0.25?) | style (> −0.3?) | Passes |
|---|---|---|---|---|
| p5 hi-res (img2img 768) | +0.31 [+0.12, +0.53] ✓ | +0.25 [+0.00, +0.50] ✓ | −0.03 [−0.25, +0.22] ✓ | **yes** |
| p6 face restore | +0.19 [+0.00, +0.34] ✗ | +0.25 [+0.09, +0.41] ✓ | +0.12 [−0.06, +0.31] ✓ | no (faces) |

**Outcome:** hi-res is the winning method. **It goes on by default if D's spot check on 4 of A's p2 images shows
no new artifact**; otherwise it stays opt-in. Face restoration stays an opt-in checkbox.
- In the app this is `post.default_on` in `config/m4.json` (it drives the "Higher resolution" box). D records the
  spot-check result and makes the config-only change: `<<pending: D, spot check + config PR>>`.
- p5 and p6 were made from D's own p1 base run (same prompts, seeds and baseline profile). B checked s09 seed 1:
  D's base image is byte-identical to A's rated p1 image (sha256 `8d63f5db…`), so p1 → p5/p6 compares the same base.
- Caveats: faces passed by 0.06, and two raters scoring the near-identical p1/p2 images already differ by 0.25 on
  faces, so the pass is narrow. Hands agreement between raters is low (weighted kappa 0.14), so the hands half of
  the test is weak evidence. p1 and p5 were rated in different rounds. The rule is applied as written.

## D3. Default pose source: a supplied pose (photo or preset), not "auto from text" (S3)
**Rule** (Overview §10 rule 3): the better of p1 (photo skeleton) and p3 (`pose_from_text`) by human pose match;
a tie goes to p3.

**Applied:** p1 3.72 [3.47, 3.94] vs p3 1.78 [1.56, 2.03]; p1 → p3 is −1.94 [−2.25, −1.59], 0/0/16. Not a tie.
**Outcome:** "auto from text" is **not** the app's default pose source. The app starts with no pose ("none") and
the user supplies one (preset or photo).
- Presets are hand-made skeletons and were not rated; only p1 (photo skeletons) was. C's `pose/library/status.json`
  marks 4 presets as ControlNet-tested.
- p3 does get the figure count right more often (15/16 seed-1 images vs p1's 13/16,
  `results/m4_extras_dev.json`), so auto pose remains available as a mode for multi-figure scenes.

## D4. Frozen code and config
- **S2:** A's end-to-end app check passed at `d8a4dd5` (Status #17, 24 Nov). One bug fix after S2, announced
  there: `793c5a7` (#140, scales cast to float; 2 lines in `pipeline/control_api.py`). It is the only code change
  since `d8a4dd5`; everything else merged since is docs, results files or C's `pose/library/status.json`
  (#139, #141-#144).
- **Config hashes** (first 12 hex of sha256, as written into manifests): `config/m4.json` `a41e8a18c928`,
  `config/m3.json` `1ced5c5c04bb`. `config/m4.json` last changed in #137 (slider help text).
- **Held-out run:** at `main` at or after `793c5a7` with these hashes; nothing from held-out changes D1-D3.

## D5. Conditions not run
- **p4** (composition from a painting, canny; optional, C) and **p8** (OpenPose with hands and face; optional, A)
  were not generated. Their comparisons (p3 → p4, p1 → p8) and hypotheses H-A5, H-B2b and H-C3 are reported as
  not run.
- **p7** (hi-res + restore) and **p9** (face inpaint) were optional and not run.

## D6. `pose_pck` is not a headline metric (H-B3 rule)
**Rule** (`experiments/HYPOTHESES_M4.md` §B, H-B3): if `pose_pck` correlates with human pose match at Spearman
< 0.3 per image, it is dropped from the headline. **Applied:** 0.15 over 96 images (all six conditions;
`results/m4_tables_dev.json`). PCK stays as supporting evidence only; human pose match is the headline.
- Post-hoc and labelled as such: p3's `pose_pck` is scored against its own auto skeleton, not the pair pose the
  raters saw. Without p3 the Spearman is 0.58 (80 images); p0 and p1 alone give 0.72 (32 images)
  (`results/m4_extras_dev.json`). This does not change D6, because the rule was fixed on all conditions.

## D7. Held-out check of D1-D3 (24 Nov 2026, after the run; nothing was tuned on it)
Six private pairs, all six conditions at `main` `9ab0765`, rated blind in one round (`eval/EVAL_REPORT_M4.md` §2.7).
- **D1 (control settings):** confirmed. p1 → p2 is within the tie band on every score (pose match −0.17, 0/5/1).
- **D3 (pose source):** confirmed. p1 → p3 pose match −1.58 [−2.58, −0.66], 0/1/5; supplied pose stays the default.
- **D2 (post-processing default): not confirmed.** Dev p1 → p5: faces +0.31, hands +0.25, style −0.03 (passed, narrowly). Held-out
  p1 → p5: faces −0.08 [−0.42, +0.25], hands −0.17, style −0.17 (would not pass). p6: faces −0.33.
  **Decision (B, recommended to the team):** keep `post.default_on: false`. Hi-res stays an opt-in checkbox for
  everyone. This is the conservative outcome of the rule on the full evidence; D's spot check and a config PR are no longer
  needed. The dev pass and the held-out miss are both reported in the report; neither is hidden.


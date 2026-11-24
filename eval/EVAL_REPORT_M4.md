# M4 evaluation report (Member B)

24 Nov 2026. Dev results (§2.1-2.6, 2.8) and the held-out run (§2.7) are final.
Every number below traces to `results/m4_tables_dev.md` / `.json` (#144), `results/m4_extras_dev.json`
or the `results/<condition>@rrv_v1_dev_pose.json` files (#130, #143). Hypotheses: `experiments/HYPOTHESES_M4.md`.

**The short answer.** ControlNet fixes posture, not faces or hands. Adding OpenPose control (p0 → p1) raised human
pose match by +1.59 points (1-5 scale; 95% CI [+1.09, +2.03], better on 14 of 16 pairs), but faces got **worse**
(−0.44 [−0.78, −0.09]) and hands did not improve (−0.25 [−0.56, +0.06]). The tuned setting the app uses (p2)
produces near-identical images to p1 (median pixel difference 2.1 of 255), so it does not change this. Hi-res
post-processing (p5) is the one step that lifted faces past our 0.25 bar (+0.31 [+0.12, +0.53] over p1) without
costing style, but only just: two raters scoring the same p1/p2 images already differ by 0.25 on faces (§2.8).
Face restoration (p6) helped less (+0.19). Hands ratings are the least reliable score we collected (rater
agreement kappa 0.14), so read every hands number as weak.

## 1. Setup

**Pairs.** 16 dev pairs (`eval/test_pairs_dev.json`): each is an M3 dev scene plus an OpenPose skeleton made from a
photo the team took of itself (photos deleted; skeletons only in git). Mix: 5 single-figure, 6 two-figure and
5 three-figure scenes; tiers 1/2/3/4 = 5/6/3/2. s22 has a `checklist_override` (its M3 item needed more than three
people). Held-out: 6 private pairs written by B (2 single, 2 two-figure, 2 three-figure), not opened before the
S3 decisions: all three were applied to dev ratings and recorded in `docs/TEAM_DECISIONS_M4.md` (#145) before the
held-out images were generated or scored; the held-out run is a check, not a tuning set (§2.7).

**Conditions** (16 pairs × seeds 1-2 = 32 images each; SD1.5 + `rrv_v1` LoRA 1.0, M3's c1 raw prompt, 512×512):

| Condition | What changes | Made by |
|---|---|---|
| `p0_m3_default` | M3 default, no control | A |
| `p1_openpose` | + OpenPose ControlNet, control 1.0, window 0-1.0 (baseline) | A |
| `p2_openpose_tuned` | + control 1.0, window 0-0.8 (A's rule winner, the app default) | A |
| `p3_pose_auto` | skeleton from the text (`pose_from_text`), baseline control | C |
| `p5_hires` | p1 + img2img hi-res to 768 (strength 0.35) | D |
| `p6_restore` | p1 + GFPGAN-class face restoration | D |

p4 (composition from a painting) and p8 (OpenPose with hands and face) were optional and not run; p7 and p9 were
not run either (`docs/TEAM_DECISIONS_M4.md` D5).

**Human ratings (main evidence).** Blind contact sheets: the pair's skeleton on the left, then the round's
conditions under random codes (`eval/contact_sheet_m4.py`). Four scores, 1-5, with written anchors
(`eval/RATING_GUIDE_M4.md`): style fidelity, pose match (to the skeleton), faces, hands ("NV" = not visible, not a
score; never used). Plus a failure box from the M4 taxonomy.
- Two rounds: sheet 1 = p0, p1, p2 (built 23 Nov); sheet 2 = p3, p5, p6 (built 24 Nov). Seed 1 only.
- 4 raters (A, B, C, D), 2 per image, each rating alone; 96 rows per round, all filled (192 ratings).
- Every image appears on the sheet at the same 320×320 tile size (downsized with Lanczos; p5's 768 images
  included), so resolution cannot unblind p5.
- **Deviation:** the planned 10-minute calibration (first 5 rows rated by all four, anchors reworded) was dropped
  to save time. The rows it would have produced were removed before rating; nobody rated a calibration row.

**Automated metrics (supporting).** `checklist` (M3's VLM judge, Qwen2-VL-2B; validated in M3, no M4 human
counterpart), `clip_raw`, `style_margin`, `max_train_dino` (M2 kit), and B's pose metric (`eval/pose_metric.py`):
`pose_pck` (PCK@0.10 of the control person's box diagonal, people matched by body centroid), `pose_detected`,
and figure count (`n_people_out` = `n_people_expected`).
- Limits: PCK is circular (the same detector family drew the photo skeletons); painted robes hide joints; for
  p3 PCK is scored against p3's own auto skeleton, not the pair pose; CLIP shares an encoder family with the
  generator; CLIP and DINO see resized images, so they cannot see hi-res gains.

**Statistics.** Unit = the pair (seed-1 image, two raters averaged). Paired differences b − a with 95% bootstrap CIs
over 16 pairs (5000 resamples) and win/tie/loss using the pre-registered tie bands: ratings 0.25, `pose_pck` 0.05,
`style_margin` 0.0295. No significance claims. A hypothesis is supported only if the CI excludes 0 in the
predicted direction.

**Hardware.** Generation and scoring on Colab Tesla T4 (14.6 GB), fp16, pins as in `docs/CONTROL.md`.

## 2. Result (dev)

### 2.1 Means per condition
Pair means (seed 1, raters averaged) with 95% CIs. Figure count = share of images with the right number of people.

| Condition | style | pose match | faces | hands | `pose_pck` | figure count | `style_margin` | checklist |
|---|---|---|---|---|---|---|---|---|
| p0 | 3.66 [3.44, 3.88] | 2.12 [1.78, 2.50] | 3.31 [3.06, 3.53] | 2.97 [2.75, 3.19] | 0.10 | 0.53 | 0.11 | 0.56 |
| p1 | 3.38 [3.19, 3.56] | 3.72 [3.47, 3.94] | 2.88 [2.69, 3.06] | 2.72 [2.50, 2.94] | 0.82 | 0.81 | 0.10 | 0.59 |
| p2 | 3.47 [3.25, 3.69] | 3.88 [3.66, 4.09] | 3.12 [2.81, 3.44] | 2.88 [2.62, 3.12] | 0.82 | 0.84 | 0.10 | 0.59 |
| p3 | 3.00 [2.78, 3.22] | 1.78 [1.56, 2.03] | 2.75 [2.53, 2.97] | 3.00 [2.75, 3.22] | 0.93* | 0.88 | 0.12 | 0.50 |
| p5 | 3.34 [3.12, 3.56] | 3.72 [3.53, 3.88] | 3.19 [2.91, 3.47] | 2.97 [2.78, 3.16] | 0.82 | 0.84 | 0.09 | 0.57 |
| p6 | 3.50 [3.31, 3.69] | 3.81 [3.66, 3.97] | 3.06 [2.84, 3.28] | 2.97 [2.72, 3.22] | 0.81 | 0.81 | 0.10 | 0.59 |

\* p3's `pose_pck` is against its own auto skeleton, so it is not comparable with the other rows.
`max_train_dino` p95 (memorisation flag ≤ 0.85), both seeds: p0 0.807, p1 0.811, p2 0.818, p3 0.771, p5 0.815,
p6 0.810. No condition is flagged.

### 2.2 Paired comparisons
b − a, 95% CI, (win/tie/loss over 16 pairs).

| Comparison | style | pose match | faces | hands | `pose_pck` |
|---|---|---|---|---|---|
| p0 → p1 | −0.28 [−0.53, −0.03] (3/5/8) | **+1.59 [+1.09, +2.03] (14/1/1)** | **−0.44 [−0.78, −0.09] (4/2/10)** | −0.25 [−0.56, +0.06] (5/3/8) | +0.72 (16/0/0) |
| p1 → p2 | +0.09 [−0.16, +0.31] (6/7/3) | +0.16 [−0.03, +0.34] (8/5/3) | +0.25 [+0.03, +0.47] (9/4/3) | +0.16 [−0.06, +0.38] (6/7/3) | −0.00 (0/16/0) |
| p0 → p2 | −0.19 [−0.53, +0.19] (4/3/9) | **+1.75 [+1.25, +2.22] (14/2/0)** | −0.19 [−0.62, +0.19] (6/3/7) | −0.09 [−0.38, +0.22] (4/5/7) | +0.71 (16/0/0) |
| p1 → p3 | −0.38 [−0.75, +0.00] (3/4/9) | **−1.94 [−2.25, −1.59] (0/0/16)** | −0.12 [−0.41, +0.16] (3/6/7) | +0.28 [−0.03, +0.56] (9/2/5) | n/a* |
| p1 → p5 | −0.03 [−0.25, +0.22] (4/7/5) | +0.00 [−0.19, +0.19] (3/9/4) | **+0.31 [+0.12, +0.53] (8/7/1)** | +0.25 [+0.00, +0.50] (6/8/2) | +0.01 (2/13/1) |
| p1 → p6 | +0.12 [−0.06, +0.31] (5/9/2) | +0.09 [−0.12, +0.31] (6/6/4) | +0.19 [+0.00, +0.34] (8/6/2) | +0.25 [+0.09, +0.41] (8/7/1) | −0.00 (0/16/0) |

**p1 → p2 compares near-identical images** (§2.8): the window change alters the last 20% of the steps only, and the
two images differ by a median of 2.1 of 255 per pixel. Its rating differences measure rater noise, not the setting.

**Read with care:** p1 was rated in round 1 and p3/p5/p6 in round 2, on different sheets and days. Comparisons
across rounds (p1 → p3, p5, p6) include any shift in how raters used the scale between sessions; comparisons inside
a round (p0, p1, p2) do not.

### 2.3 By tier and by people count (human ratings)
Pairs per tier: t1 5, t2 6, t3 3, t4 2; by people: 1 = 5, 2 = 6, 3 = 5. CIs on 2-6 pairs are very wide.

| p0 → p1 | pose match | faces | hands | style |
|---|---|---|---|---|
| tier 1 (n=5) | +2.10 [+1.50, +2.70] | −0.30 [−0.70, +0.20] | −0.70 [−1.10, −0.30] | −0.30 [−0.70, +0.10] |
| tier 2 (n=6) | +1.33 [+0.50, +2.00] | −0.75 [−1.17, −0.25] | −0.42 [−0.83, +0.00] | −0.50 [−0.83, −0.17] |
| tier 3 (n=3) | +1.67 [+0.50, +2.50] | +0.33 [+0.00, +0.50] | +0.50 [+0.50, +0.50] | +0.33 [+0.00, +0.50] |
| tier 4 (n=2) | +1.00 [+0.00, +2.00] | −1.00 [−1.50, −0.50] | +0.25 [+0.00, +0.50] | −0.50 [−1.00, +0.00] |
| 1 person (n=5) | +2.10 [+1.50, +2.70] | −0.30 [−0.70, +0.20] | −0.70 [−1.10, −0.30] | −0.30 [−0.70, +0.10] |
| 2 people (n=6) | +1.33 [+0.50, +2.00] | −0.75 [−1.17, −0.25] | −0.42 [−0.83, +0.00] | −0.50 [−0.83, −0.17] |
| 3 people (n=5) | +1.40 [+0.50, +2.20] | −0.20 [−0.90, +0.40] | +0.40 [+0.20, +0.50] | +0.00 [−0.50, +0.40] |

| p1 → p3 pose match | tier 1 | tier 2 | tier 3 | tier 4 |
|---|---|---|---|---|
| | −2.50 (0/0/5) | −1.83 (0/0/6) | −1.33 (0/0/3) | −1.75 (0/0/2) |

The mean pose-match gain is positive at every tier and people count. The face cost is clearest in 2-person scenes
(−0.75 [−1.17, −0.25]); in 3-person scenes faces are level and hands slightly better (+0.40 [+0.20, +0.50]). Per-tier rows for p1 → p2 and the people-count rows
for p1 → p3 are in `results/m4_tables_dev.json`.

### 2.4 Figure count (seed 1, from `results/m4_extras_dev.json`)
Right number of people: p0 9/16, p1 13/16, p2 13/16, p3 15/16, p5 13/16, p6 13/16.
- p0 → p1: only p1 right on 5 pairs, only p0 right on 1. In tiers 2-4 (11 pairs): 8 vs 5 right (4 vs 1).
- p0 → p3: only p3 right on 6 pairs, only p0 right on 0.

### 2.5 Faces and hands, and the failure taxonomy
Faces and hands are separate scores throughout; neither is merged into an "anatomy" number. Failure-box ticks
over 32 ratings per condition (2 raters × 16 pairs; a rating can tick several boxes):

| Failure | p0 | p1 | p2 | p3 | p5 | p6 |
|---|---|---|---|---|---|---|
| pose ignored | 16 | 2 | 2 | 27 | 0 | 0 |
| pose followed but scene wrong | 0 | 5 | 5 | 0 | 8 | 10 |
| face distorted | 0 | 9 | 10 | 7 | 8 | 8 |
| hands malformed | 3 | 5 | 5 | 1 | 1 | 4 |
| merged/duplicated figures | 5 | 4 | 5 | 2 | 4 | 3 |
| extra/missing limbs | 2 | 2 | 2 | 0 | 3 | 3 |
| style weakened (photographic) | 0 | 2 | 1 | 1 | 1 | 2 |
| stiff/over-constrained | 0 | 1 | 1 | 3 | 1 | 1 |
| post-processing artifacts | 0 | 0 | 0 | 0 | 0 | 0 |
| other | 1 | 2 | 1 | 0 | 2 | 1 |

(p3's "pose ignored" includes one entry typed with a double space, merged here.) "Face distorted" appears only once
control is on (0 → 9), matching the face ratings. Nobody ticked "post-processing artifacts" for p5 or p6.

### 2.6 Comparison grids
Seed-1 images cut from the blind contact sheets and labelled after unblinding (`eval/grids_m4.py`), each row:
control skeleton | p0 | p1 | p2 | p3 | p5 | p6, with the two raters' mean style / pose / faces / hands. Failures are
included on purpose.

| Grid | Why it was picked |
|---|---|
| `docs/figures/grid_m4_dev_s02.png` | largest pose-match gain with control (p0 → p1 +3.0) |
| `docs/figures/grid_m4_dev_s11.png` | the one pair where control lost on pose match (−0.5); faces −1.5 with control |
| `docs/figures/grid_m4_dev_s15.png` | 3 figures, tier 4: faces −1.5 with control |
| `docs/figures/grid_m4_dev_s18.png` | largest hi-res face gain (p1 → p5 +1.0); hands −1.5 with control |
| `docs/figures/grid_m4_dev_s26.png` | failure: "pose followed but scene wrong" for p1, p2, p5 and p6 (no chariot) |
| `docs/figures/grid_m4_dev_s31.png` | failure: p3 ignores the pose; control alone gives no pose gain here (0.0) |

### 2.7 Held-out
Six private pairs written by B (m4h01-m4h06: 2 single, 2 two-figure, 2 three-figure; `eval/test_pairs_heldout.json`,
scenes in `eval/test_scenes_heldout_m4.json`), kept private until now. All six conditions were generated at `main`
`9ab0765` (config hashes as in D4), seeds 1-2, on a Colab T4, scored with a private detection cache, and rated blind
in one round of 36 images (6 conditions × 6 pairs, 2 of 4 raters each, no calibration rows). Files: `results/m4_tables_heldout.md`/`.json`,
`results/<condition>@rrv_v1_heldout_pose.json`, `eval/ratings/ratings_m4_heldout.csv`, `eval/ratings/blind_key_m4_heldout.json`.
Means over 6 pairs, seed 1, 95% bootstrap CI (one round, so p1 and p3/p5/p6 were rated together this time):

| Condition | style | pose match | faces | hands |
|---|---|---|---|---|
| `p0_m3_default` | 3.25 [2.83, 3.67] | 2.08 [1.50, 2.75] | 2.83 [2.50, 3.00] | 2.83 [2.50, 3.00] |
| `p1_openpose` | 3.42 [3.00, 3.83] | 3.83 [3.33, 4.33] | 3.00 [2.75, 3.25] | 2.75 [2.33, 3.17] |
| `p2_openpose_tuned` | 3.25 [3.08, 3.42] | 3.67 [3.25, 4.08] | 2.92 [2.50, 3.25] | 2.75 [2.33, 3.17] |
| `p3_pose_auto` | 3.67 [3.33, 4.00] | 2.25 [1.50, 3.08] | 3.25 [2.83, 3.67] | 3.08 [3.00, 3.25] |
| `p5_hires` | 3.25 [2.83, 3.58] | 3.67 [3.08, 4.17] | 2.92 [2.67, 3.17] | 2.58 [2.00, 3.08] |
| `p6_restore` | 3.17 [2.83, 3.42] | 3.75 [3.17, 4.17] | 2.67 [2.17, 3.08] | 2.67 [2.17, 3.08] |

Paired comparisons (b − a, win/tie/loss within the tie bands):

| Comparison | pose match | faces | hands | style |
|---|---|---|---|---|
| p0 → p1 | +1.75 [+0.75, +2.58] (5/0/1) | +0.17 [−0.17, +0.58] (2/3/1) | −0.08 [−0.50, +0.25] | +0.17 [−0.17, +0.50] |
| p1 → p2 | −0.17 [−0.50, +0.00] (0/5/1) | −0.08 [−0.50, +0.25] | +0.00 [−0.25, +0.25] | −0.17 [−0.58, +0.17] |
| p1 → p3 | −1.58 [−2.58, −0.66] (0/1/5) | +0.25 [−0.17, +0.83] | +0.33 [+0.00, +0.67] | +0.25 [−0.17, +0.83] |
| p1 → p5 | −0.17 [−0.33, +0.00] | −0.08 [−0.42, +0.25] (2/1/3) | −0.17 [−0.33, +0.00] | −0.17 [−0.33, +0.00] |
| p1 → p6 | −0.08 [−0.58, +0.50] | −0.33 [−0.83, +0.17] (1/2/3) | −0.08 [−0.42, +0.25] | −0.25 [−0.58, +0.17] |

What held and what did not:
- **Held:** a supplied pose fixes posture (+1.75 pose match, better on 5 of 6 pairs, dev +1.59). Auto-from-text loses to
  a photo skeleton on pose match (−1.58, 0/1/5; dev −1.94). The tuned window changes nothing (p1 → p2 within the
  tie band on every score, as in dev). Figure count right: p0 0.58, p1 0.83, p2 0.83, p3 0.83, p5 0.75, p6 0.83
  (share of pairs, seed 1).
- **Did not hold:** the dev claim that control makes faces worse. Held-out faces and hands barely move with control
  (+0.17 and −0.08, CIs include 0), so with 6 pairs we cannot say control hurts faces; the safest reading is "control
  does not help faces or hands" (H-B2 is still supported in the sense that it does not improve them).
- **Did not hold:** the hi-res face gain. Dev p1 → p5 faces was +0.31; held-out it is −0.08 [−0.42, +0.25], and hands
  −0.17. By the D2 rule (faces > +0.25) hi-res would **not** pass on the held-out pairs. Restoration is lower still
  (faces −0.33). Six pairs is a small test, but the dev pass was already narrow (§3).
- **Pose from text, figure count:** `pose_from_text` drew the right number of figures for 4 of 6 scenes (m4h01, 02, 03
  and 05); it drew 1 figure for m4h04 (two people: Dasharatha and Kaikeyi) and for m4h06 (three: Gandhari,
  Dhritarashtra and a maid). These two scenes are also p3's lowest on automated pose score (m4h04 pck 0.11 / 0.00).
- **Agreement (36 images, 2 raters each):** pose match κw 0.80, hands 0.62, faces 0.53, style 0.38: better than dev
  for hands and faces, from a single round with fewer, simpler scenes.
- **Metric validation:** `pose_pck` vs human pose match 0.46 (n=36), `style_margin` vs human style 0.08. The H-B3 rule
  was applied to dev (0.15) and is not re-decided; held-out PCK sits above 0.3, within the noise of n=36.

### 2.8 Metric validation and rater agreement
**Rater agreement** (the two raters of each image, 96 images):

| Score | Spearman | Weighted kappa |
|---|---|---|
| pose match | 0.73 | 0.75 |
| faces | 0.40 | 0.36 |
| style | 0.33 | 0.31 |
| hands | 0.16 | 0.14 |

**Rating noise, measured on near-identical images.** p1 and p2 differ by a median of 2.1 of 255 per pixel on the
contact sheets (max 2.98; p0 vs p1: median 39.1, min 28.9; `results/m4_p1_p2_similarity_dev.json`). In all 16
pairs the p1 and p2 images were rated by different raters. Their pair-mean ratings still differ by 0.34 (style,
pose match, hands) to 0.44 (faces) on average, and on faces p2 came out +0.25 [+0.03, +0.47] higher. So a
difference of about 0.25 can appear between two raters looking at the same picture; differences near the 0.25
tie band (p1 → p5 faces +0.31, p1 → p6 +0.19) should be read as small.

**`pose_pck` vs human pose match:** Spearman 0.15 per image over all six conditions (pre-registered). By the H-B3
rule (< 0.3), PCK is dropped from the headline. Post-hoc: p3's PCK measures something different (its own
skeleton); without p3 the correlation is 0.58 (80 images), and p0 + p1 alone give 0.72 (32 images). Inside the
controlled photo-pose conditions (p1, p2, p5, p6) it is 0.23, because they all sit near PCK 0.8.
**`style_margin` vs human style:** 0.17. The checklist judge has no M4 human counterpart (validated in M3).

## 3. What it means

- **ControlNet fixes posture, not faces or hands.** Pose match rose 1.6-1.75 points with control and won on 14 of 16
  pairs, and the right figure count went from 9/16 to 13/16. Faces got worse with control (−0.44 for p1; p2's −0.19
  is the same images rated by other raters). This is the result H-B2 predicted (better posture does not give better
  faces), and the raters' failure boxes say the same thing: "face distorted" appears only once control is on.
- **The tuned window (p2) changes nothing we can measure.** Its images are near-identical to p1's (§2.8), so its
  rating differences from p1 are rater noise. It stays the app default because the pre-registered rule picked it
  (D1), and it costs nothing.
- **Post-processing:** hi-res helps faces (+0.31, just above the 0.25 that rater noise alone produced) without
  costing style or pose; face restoration helps faces a
  little less (+0.19) and, unexpectedly, did not cost style (+0.12). Under the pre-registered rule, hi-res passes
  and restoration does not (`docs/TEAM_DECISIONS_M4.md` D2).
- **Pose from text (p3)** gets the head count right (15/16) but not the posture: raters scored it 1.78 on pose
  match, below even p0's 2.12, and ticked "pose ignored" on 27 of 32 ratings. A text-made skeleton fixes who is in
  the picture, not how they stand. The gap to photo poses is large at every tier, not only at tier 3+.
- **The automated pose metric agrees with humans on "is there control at all?"** (0.72 on p0 + p1) **but not on
  finer differences** (0.23 among the controlled conditions). Humans are the evidence for anything finer.

**Hypotheses** (support rule: CI excludes 0 in the predicted direction):

| Hypothesis | Verdict | Evidence |
|---|---|---|
| H-B1 / H-A4: p1 beats p0 on pose match and figure count (tiers 2-4) | **Supported** for pose match; figure count in the predicted direction (8 vs 5 of 11), no CI | +1.59 [+1.09, +2.03]; §2.3, §2.4 |
| H-B2 / H-A4: p1 does not beat p0 on faces or hands | **Supported** (faces worse) | faces −0.44 [−0.78, −0.09]; hands −0.25 [−0.56, +0.06] |
| H-B2b, H-A5 (p8) | Not run | |
| H-B3: PCK vs human pose match ≥ 0.5 | **Not supported** (0.15); PCK dropped from the headline | §2.8 |
| H-B4: p2 lowers style by ≤ 0.3 vs p0 | Mean within (−0.19), but the CI reaches −0.53: **not established** | p0 → p2 style |
| H-C2: p3 below p1 on pose match, above p0 on figure count; gap small at tiers 1-2, large at 3+ | First two **supported** (−1.94 [−2.25, −1.59]; 15 vs 9 of 16, 6/0); the tier pattern **not supported** (largest gap at tier 1) | §2.2-2.4 |
| H-D2: restoration raises faces but lowers style | **Not supported** on either half: faces +0.19 [+0.00, +0.34] touches 0; style +0.12 | p1 → p6 |
| H-D1 (hi-res, duplicated figures) | Raters' "merged/duplicated figures" 4 (p1) vs 4 (p5); D's by-eye count is in `APP.md` | §2.5 |
| H-C1, H-C3, H-D3, H-D4, H-A1-A3 | Owners' docs (`POSE.md`, `APP.md`, `CONTROL.md`) | |

## 4. Limits
- **Small n:** 16 dev pairs (6 held-out); per-tier cells have 2-6 pairs. No significance claims.
- **Raters:** the four team members, who knew the project and photographed the poses (rows were
  assigned at random, so raters could rate pairs made from their own photos). Calibration was skipped. Agreement is good for pose match, fair for faces and style, and poor
  for hands (kappa 0.14).
- **Rater noise is about the size of the tie band:** two raters scoring near-identical images (p1, p2) differ by
  0.25 on faces on average (§2.8).
- **Two rating rounds:** p1 → p3/p5/p6 crosses rounds (§2.2). On sheet 2, p5 and p6 share a base image and often
  look alike, which could hint that two of the three codes are variants of each other (p1 was not on that sheet).
- **Poses** come from 4 people's photos; body-only OpenPose (no hands or face keypoints, p8 not run).
- **One model** (SD1.5 + `rrv_v1`), 512×512 generation, 320-pixel tiles for rating; hi-res detail beyond that was not
  rated blind.
- **Gaze and drapery** were not tested.
- **PCK** is circular and weak on painted figures; for p3 it measures a different target.
- **Held-out is 6 pairs, 36 images, one rating round, two raters per image:** the held-out confidence intervals are
  wide, p1 → p5 and p1 → p6 differences are within rater noise, and pairs are composites/stock photos written by B,
  who also rated. No held-out result changed a decision already recorded in `TEAM_DECISIONS_M4.md`; D2's hi-res default
  is revisited in D7 because the held-out did not replicate the dev gain.

## 5. Hard questions
- **"Is the pose metric trustworthy?"** Only for the coarse question. Per image it correlates 0.15 with human pose
  match over all conditions (pre-registered), which is why it is not a headline number. Most of that gap is p3,
  whose PCK is scored against its own skeleton: on p0 + p1 alone it is 0.72. Among the controlled conditions it
  barely separates anything (0.23). Human pose match is the evidence.
- **"Did you tune on the held-out?"** No. All three default rules were applied to dev ratings and recorded in
  `docs/TEAM_DECISIONS_M4.md` before the held-out scores were opened; the held-out pairs live only in B's private
  Drive until the reveal. Held-out generated and scored 24 Nov 2026 at `main` `9ab0765`; rated 24 Nov; revealed in this PR.
- **"Why only OpenPose body?"** Time and hardware: one ControlNet on a T4. The hands/face variant (p8) was planned
  as optional and not run, so whether hand keypoints would help hands is untested. Our data say the body skeleton
  does not fix faces or hands by itself.
- **"Does ControlNet fix faces and hands?"** No. See the short answer at the top.

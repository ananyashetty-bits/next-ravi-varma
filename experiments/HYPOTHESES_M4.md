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

## Section B (B: evaluation). Written 22 Nov 2026, before any M4 image was generated or scored

### Rating protocol (decided at the contract meeting, from M3's judge validation)
- **4 human scores, each 1-5:** style fidelity, pose match, faces, hands. Hands also has a "not visible" option, which
  is not a score.
- **Human faithfulness is dropped.** The M3 checklist judge passed its pre-registered validation: on dev it reached
  Spearman 0.41 with human faithfulness, against 0.24 for CLIP, and the bar was ≥ CLIP and ≥ 0.4. On held-out it
  reached 0.65 (`eval/EVAL_REPORT.md`). The judge supplies faithfulness.
- **Anchors:**
  - *Style fidelity:* 1 = photographic or modern-looking; 3 = painted but stiff or cluttered; 5 = reads as a dignified,
    balanced Varma-style painting.
  - *Pose match:* 1 = skeleton ignored; 3 = roughly the right figures and posture, limbs wrong; 5 = limbs and placement
    follow the skeleton.
  - *Faces:* 1 = distorted or missing; 3 = smeared but recognisable; 5 = convincing as a painted face.
  - *Hands:* 1 = malformed or merged; 3 = plausible blobs; 5 = convincing.
- **Failure taxonomy** (one or more per image): pose ignored · pose followed but scene wrong · extra/missing limbs ·
  merged/duplicated figures · face distorted · hands malformed · style weakened (photographic) ·
  stiff/over-constrained · post-processing artifacts · other.
- **Raters and sizes:**
  - ≥ 2 raters per image, each rating alone; the first 5 rows are rated by all four as calibration.
  - Dev uses seed 1.
  - Every condition is rated on a 512×512 copy (post-processed images are downsized with Lanczos).
- **LoRA scale check:** the LoRA scale for M3's rated baselines was confirmed as 1.0 by A on 19 Nov
  (`docs/M3_REPORT.md`).

### Pose metric (`eval/pose_metric.py`), fixed before scoring
- **Matching:** detected people are matched to control people by minimum total distance between body centroids
  (neck, right hip, left hip). Joints count only with confidence > 0.3.
- **A correct joint:** within `pck_thresh` = 0.10 × the diagonal of the control person's keypoint bounding box.
- **Scores:**
  - `pose_pck` = correct joints / all control joints; missed people and missed joints count as wrong.
  - `pose_detected` = anyone detected.
  - Also reported: `n_people_out` and joint recall.
- **What it's scored against:** the skeleton actually used, which is the pair's PoseJSON. For auto-pose conditions
  it's the `<name>_pose.json` sidecar. p0 is scored against the pair's pose too, as the "no control" baseline.
- **Known weaknesses:**
  - Circularity: the same detector family made the photo skeletons.
  - Painted robes and stylised bodies reduce detection.
  - The detector can't read skeleton drawings back (22 Nov: 0 people found on all 4 rendered pair skeletons). The
    renderer is instead verified to be pixel-identical to controlnet_aux's own drawing (#101).
  - Detection failures are reported separately, from the detector audit on the 15 real held-out Varma paintings and
    on p0/p1.
- **Precedence:** human pose match is the main evidence. **When PCK and humans disagree, humans win.**

### Tie bands
- Human ratings: **0.25**.
- `pose_pck`: **0.05**.
- `style_margin`: **0.0295** (M2 `tie_band`).
- Memorisation flag: `max_train_dino` p95 ≤ **0.85** (M2 `mem_flag_dino_p95`).

### Statistics
- **Method:** per-pair paired comparisons (seeds averaged first), giving the mean difference, a 95% bootstrap CI
  over pairs (5000 resamples, M3's `eval/stats.py`) and win/tie/loss with the bands above.
- **Breakdowns:** by tier and by people count (1 / 2 / 3).
- **No significance claims** (16 dev and 6 held-out pairs).
- **Support rule:** a hypothesis is **supported** only if the 95% CI excludes 0 in the predicted direction. "Does
  not beat" counts as supported if the CI includes 0 or lies on the predicted side.

### Hypotheses
- **H-B1:** p1 beats p0 on human **pose match** and on **figure count** (`n_people_out` = `n_people_expected`), in
  tiers 2-4.
- **H-B2:** p1 does **not** beat p0 on human **face** or **hand** ratings, because pose control fixes limbs and
  placement, not faces.
  - A positive result would mean better posture also gives the model a cleaner layout to draw faces and hands into.
    We would report that as a finding, not explain it away.
  - **H-B2b (if p8 runs):** hand and face keypoints improve hands more than faces.
- **H-B3:** `pose_pck` correlates with human pose match at Spearman ≥ 0.5 per image, and less well on robed or
  painted figures. If it is < 0.3, PCK is dropped from the headline.
- **H-B4:** at default settings (p2), pose control lowers the mean human style rating by ≤ 0.3 relative to p0.

### Default-setting rules (Overview §10, rules 1-3, verbatim)
1. *Control scale, guidance window, LoRA scale (A, Day 2 by 12:30).*
   - Run 4–5 stage-2 candidates on all 16 dev pairs, seed 1 (at least 12 pairs if some are late, flagged in
     `CONTROL.md`), compared with **p0's seed-1 images**.
   - **Eligible** = `max_train_dino` p95 ≤ 0.85 **and** mean `style_margin` ≥ p0's − 0.0295.
   - Among eligible candidates, pick the highest mean `pose_pck`. If two candidates' paired CI includes 0, pick the
     lower control scale.
   - The LoRA scale stays 1.0 unless a candidate with another scale wins by this rule.
2. *Post-processing in the app (D, S3).* It's on by default only if all three hold:
   - the mean **face** rating improves by more than 0.25;
   - **hands** don't drop by more than 0.25;
   - the mean **style** rating drops by less than 0.3.

   Otherwise it's an opt-in checkbox. Before deciding, D spot-checks the winning method on 4 of A's p2 images; a
   visible new artifact means opt-in.
3. *Default pose source in the app:* the better of p1 and p3 by human pose match; a tie goes to p3 (no upload
   needed).

### Held-out
- **Pairs:** 6 private pairs written by B (2 single, 2 two-figure, 2 three-figure; ≥ 2 on tiers 4-5; ≥ 2 characters
  not in the KB).
- **Prompts:** c1 raw, CPU only.
- **Images:** generated once at the S2 sha, with a private detection cache.
- **Scores:** not opened until the S3 decisions are written. Nothing from held-out changes any default.

## Section C (C: pose inputs, pose-from-text, composition). Written 23 Nov 2026, before any p3 or p4 image was generated or scored

### What was already run, and what was not
- Unscored ControlNet smoke tests (Colab, T4, fp16, baseline control profile, 2 seeds): the four photo pairs s18, s19, s21, s22 and four presets (`standing`, `seated`, `kneeling`, `kneeling`+`standing`). Looked at by eye only; no setting or prompt was tuned from them. They are not results.
- No `pose_from_text` output, no p3 image and no p4 image exists yet.

### Definitions fixed before running
- **Figure count (`pose_from_text`):** the number of people in the PoseJSON that `pose_from_text` returns equals the pair's `n_people_expected`. Measured on the 16 dev pairs, with no image involved.
- **Posture family:** one of standing, seated, kneeling, lying, raised arms. Reference labels are assigned by C from each pair's own skeleton (not from any `pose_from_text` output) and stored in `eval/pose_families_dev.json` before `pose_from_text` is run. A pair is correct when the multiset of families over its figures equals the reference.
- **p3 figure count in images:** `n_people_out` = `n_people_expected`, from B's `pose_metric.py`.
- **Held-out:** B measures `pose_from_text` on the 6 held-out pairs on Day 3. M3's parser recall on held-out was 0.42, so a drop is expected. Nothing from held-out changes any default.

### Tie bands and support rule
As in Section B: human ratings 0.25, `pose_pck` 0.05, memorisation flag `max_train_dino` p95 <= 0.85. A hypothesis is supported only if the 95% CI excludes 0 in the predicted direction; proportions on 16 pairs are reported as k/16 with no significance claim. Humans win when PCK and human pose match disagree.

### Hypotheses
- **H-C1:** `pose_from_text` gets the right figure count on at least 80% of the 16 dev pairs (>= 13/16) and the right posture family on at least 60% (>= 10/16). Expected to be worse on held-out.
- **H-C2:** p3 (text-only pose) is below p1 (photo pose) on human pose match, but above p0 (no control) on figure count (`n_people_out` = `n_people_expected`). The p1-p3 gap is small for tiers 1-2 and large for tier 3 and above (reported by tier).
- **H-C3 (optional p4):** composition from a held-out painting gives high style and pose adherence but also high similarity to its source: DINOv2 similarity to the source painting is > 0.85 for at least half of the p4 images, so p4 needs a copy guardrail before it is offered in the app.

### Known weaknesses
- `pose_from_text` is deterministic and uses M3's `parse_scene`, so it inherits the parser's misses (names, counts and postures that the parser did not extract).
- A skeleton fixes the count and layout of the main figures, not who they are or the extras the prompt asks for: crowd members and gender or costume details come from the prompt and the LoRA. For tier 3 and above scenes, p3 and p1 can therefore lose background figures that the prompt mentions.
- Reference posture labels are assigned by one person (C), from skeletons, in `eval/pose_families_dev.json`. Three are uncertain (s01, s10, s26; labelled standing from a thumbnail), so H-C1's posture accuracy is reported on all 16 pairs and on the 13 certain ones.

## Section D (D: app, post-processing, content policy). Written 22 Nov 2026, before any p5/p6 run was generated or scored

### Conditions and labels
- p5_hires: the p1 images + img2img hi-res (Lanczos 512 -> 768, then img2img with the same weights, LoRA, c1 prompt and negative; strength from `config/m4.json` `post.hires.strength`). ESRGAN x2 is a variant of p5 (optional; img2img is kept if time runs short, Overview section 14).
- p6_restore: the p1 images + face restoration (GFPGAN-class).
- p7_hires_restore (optional): both steps.
- The optional face-detail inpaint is labelled `p9_face_inpaint`, not p7, so that `p7` means one thing (Overview section 8).

### Same-size rule, and what the automated metrics cannot see
- Raters see only the 512x512 Lanczos copies (`_512.png`), so resolution cannot unblind p5/p6.
- CLIP and DINO see a resized image, so hi-res gains are invisible to them. Hi-res detail is judged separately on side-by-side face-crop strips (base | img2img | ESRGAN | restore, 3 pairs) and reported as non-blind.

### Tie band
ratings 0.25.

### Default-setting rule (Overview section 10, rule 2, verbatim)
2. **Post-processing in the app (D, S3).** It's on by default only if all three hold:
   - the mean **face** rating improves by more than 0.25;
   - **hands** don't drop by more than 0.25;
   - the mean **style** rating drops by less than 0.3.

   Otherwise it's an opt-in checkbox. Before deciding, D spot-checks the winning method on 4 of A's p2 images; a visible new artifact means opt-in.

### Hypotheses
- H-D1: img2img hi-res improves detail (non-blind crop comparison) without increasing duplicated figures (counted by eye on all 32 images per condition, against p1). ESRGAN improves sharpness but not faces (face rating change within the 0.25 tie band).
- H-D2: face restoration raises the face rating (B's separate face score) but lowers the style rating (photographic faces). If faces do not improve, that is reported as the result.
- H-D3: both post steps keep `max_train_dino` p95 <= 0.85 (from the results files).
- H-D4: the app's median latency on a T4 is under 60 s for text + pose + hi-res. Measured as the median of 5 warm runs for: text only, text + preset pose, + hi-res, + restore. Peak VRAM and the cold start (model load) are reported separately.

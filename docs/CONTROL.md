# CONTROL: control engine and the style/pose trade-off (Member A)

Final. Numbers trace to `results/`; human ratings are in B's `EVAL_REPORT_M4.md`.

## 1. What I built
`pipeline/control_api.py`: `generate_controlled(...)` runs SD1.5 + M3's Varma LoRA + one OpenPose ControlNet, derived from M3's own pipeline with `from_pipe`. The LoRA scale is set only through M3's `_ensure` (M2's `set_scale`), never `cross_attention_kwargs`. M3's `generate(..., control_image=X)` delegates to it.
`pipeline/run_batch.py` gained `--pairs`, `--control {none,openpose,openpose_full,canny}`, `--control_profile {baseline,tuned}`, `--control_source {pair,auto}` and the sweep-only `--control_scale --guidance_start --guidance_end --lora_scale`. `--condition` is mandatory and overrides the label in the prompt records. Manifests record the control settings, pose-file sha256, ControlNet revision, LoRA scale and negative used, config hashes and library versions.
Sweep: `pipeline/control_sweep.py` (generation) and `pipeline/control_score.py` (scoring, grids, plot).
Reproduce p1: see the pins block in the README (one `run_batch.py --control openpose --control_profile baseline` command per condition).

Pins: diffusers 0.40.0, transformers 5.17.0, peft 0.21.0, accelerate 1.15.0, controlnet_aux 0.0.10, gradio 6.31.0, torch 2.14.1+cu130; OpenPose ControlNet revision 9ae9f970358db89e211b87c915f9535c6686d5ba.
Sanity checks (one T4 session, all passed): scale 0 and LoRA 0.5/1.0 match M3 exactly (0.0/255, DINOv2 1.0); LoRA 0 differs from 1.0 by 41.4/255; a black control image runs; identical inputs give identical images.

## 2. Result
Stage 1 (3 fixtures x 2 seeds, 144 images; `results/control_stage1_cells.json`), LoRA 1.0 unless stated:

| control scale | pose_pck | style_margin |
|---|---|---|
| 0.4 | 0.461 | 0.1042 |
| 0.6 | 0.723 | 0.1188 |
| 0.8 | 0.781 | 0.1142 |
| 1.0 | 0.817 | 0.1145 |
| 1.2 | 0.862 | 0.1141 |
At LoRA 0.6, style_margin falls from 0.0999 (control 0.6) to 0.0894 (control 1.2); no other LoRA scale falls with control.

Window (control 1.0, LoRA 1.0), `guidance_end` 0.4 / 0.6 / 0.8 / 1.0: pose_pck 0.769 / 0.801 / 0.817 / 0.817; style_margin 0.111 / 0.122 / 0.118 / 0.115.
Per fixture, control 0.4 to 1.2: f01 0.25 to 0.89, f02 0.76 to 1.00, f03 0.38 to 0.70. In f03 the bent-over figure renders as a horse at every control scale. The detector found the expected number of people in 97 of 144 images.

Stage 2 (16 dev pairs, seed 1, against p0 seed 1; p0 style_margin 0.1101, eligibility floor 0.0806; all max_train_dino p95 <= 0.817):

| candidate | control / window / LoRA | pose_pck | style_margin |
|---|---|---|---|
| p1 baseline | 1.0 / 0-1.0 / 1.0 | 0.807 | 0.1070 |
| c0.8_l1.0 | 0.8 / 0-1.0 / 1.0 | 0.792 | 0.1038 |
| c1.2_l1.0 | 1.2 / 0-1.0 / 1.0 | 0.832 | 0.1029 |
| c0.8_l1.2 | 0.8 / 0-1.0 / 1.2 | 0.787 | 0.1069 |
| c1.0_ge0.8 | 1.0 / 0-0.8 / 1.0 | 0.809 | 0.1086 |

Rule outcome (Overview section 10): all five are eligible. c1.2_l1.0 has the top pose_pck, but its paired CI includes 0 against the baseline [-0.016, +0.069] and against c1.0_ge0.8 [-0.014, +0.066], so those three tie; the rule picks the lowest control scale among them, giving control.tuned = control 1.0, window 0-0.8, LoRA 1.0 (merged into config/m4.json, approved by D).
Hardware (Tesla T4, 14.6 GB, 512x512, 30 steps, fp16): baseline 6.19 GB, 19.4 s/image; attention slicing 6.19 GB, 23.4 s; model CPU offload 4.94 GB, 29.4 s; sequential offload 0.82 GB, 61.6 s; img2img 768 pass 7.07 GB, 18.5 s. With the ControlNet pipeline and the img2img pass both loaded, peak VRAM is 7.07 GB (measured end to end, 36.7 s per image for both passes), so SD + LoRA + ControlNet + hi-res fit one T4 with about 7.5 GB spare and no offloading.
p0 / p1 / p2 on 16 dev pairs x 2 seeds (automated metrics, quoted from `results/<condition>@rrv_v1_dev_pose.json`; human ratings are in B's report):

| condition | pose_pck | style_margin | checklist | clip_raw | max_train_dino p95 |
|---|---|---|---|---|---|
| p0_m3_default | 0.100 | 0.1082 | 0.565 | 0.277 | 0.807 |
| p1_openpose | 0.816 | 0.1016 | 0.587 | 0.270 | 0.811 |
| p2_openpose_tuned | 0.815 | 0.1026 | 0.594 | 0.271 | 0.818 |

Files behind these numbers (all in `results/`): `control_stage1_cells.json`, `control_stage1_pose_scores.json`, `control_rule_16pairs.json`, `control_stage2_<candidate>.json` (x4), `p0_m3_default@rrv_v1_dev_pose.json`, `p1_openpose@rrv_v1_dev_pose.json`, `p2_openpose_tuned@rrv_v1_dev_pose.json`.

## 3. What it means
- The pose knee is at control 0.6 to 0.8: f01 goes from 0.25 to 0.81 between 0.4 and 0.6.
- style_margin shows no clear cost of control: at LoRA 1.0 it spans 0.015 across control 0.4-1.2, half the 0.0295 tie band. Only LoRA 0.6 loses style as control rises.
- H-A1: the pose half held; the style half is not shown by this metric. H-A2: pose is unchanged at guidance_end 0.8 vs 1.0 (0.817 both) and style differs within the tie band, so neither supported nor refuted. H-A3: not supported; LoRA 1.2 gives no consistent style gain (control 1.0: 0.1098 vs 0.1145; control 1.2: 0.1161 vs 0.1141). H-A4 (p1 beats p0 on human pose match and figure count, but not on faces or hands) is decided on B's blind ratings in `EVAL_REPORT_M4.md`; automated pose_pck rises from 0.100 (p0) to 0.816 (p1). H-A5: p8 not run.
- On 16 pairs the tuned setting is indistinguishable from the baseline (pose_pck 0.809 vs 0.807, style 0.1086 vs 0.1070; identical on 8 of 16 pairs), and the three-way tie has CI edges of -0.016 and -0.014. Tuning gives no gain over the baseline; the 0-0.8 window costs nothing and gains nothing.
- c1.2_l1.0 has the highest pose match on 16 pairs (+0.025 over the baseline, CI includes 0), driven by s02 (+0.235) and s01 (+0.126), and it loses 0.1 or more on s18, s19 and s27, so it is not shown to be better.
- Slider help text (in the app, #137). Style strength (0.6 to 1.2, default 1.0): "How strongly the picture follows Varma's painted look. Lower numbers look more generic and glossy. 1.0 is the default. Going above 1.0 did not add more style in our tests." Pose strength (0.4 to 1.2, default 1.0): "How closely the figures follow your pose. Below about 0.6 they often ignore it. 0.8 to 1.0 follows it well. Above 1.0 adds little for simple poses. Unusual poses, such as bending right over, can be misread (in one test a bent-over figure turned into a horse)."

## 4. Limits
Stage 1 used 3 fixtures (f03 is dev scene s01, because M3's fixtures have no 3-figure scene) and 2 seeds, and only finds the region. Stage 2 used 16 pairs, one seed, bootstrap over 16 pairs. pose_pck uses the detector family that drew the skeletons (circular), and painted figures are hard to detect, so human pose-match ratings are the main evidence. style_margin measures similarity to Varma references and cannot see stiffness, faces or hands. One base model (SD1.5, 512x512), fp16 not quantized, hardware on one T4. Gaze and drapery were not tested.

## 5. For the report's Q&A
- Does ControlNet fix faces? That is answered by B's blind face and hand ratings (`EVAL_REPORT_M4.md`). None of our automated metrics scores faces, and body-only OpenPose carries only coarse head keypoints.
- Why this pose strength? Pose match rises steeply up to control 0.6-0.8 and flattens after 1.0 (stage 1: 0.46 to 0.72 to 0.82 to 0.86 at 0.4, 0.6, 1.0, 1.2). On 16 pairs, 1.2 is not shown to beat 1.0 (paired CI against the baseline [-0.016, +0.069]) and style_margin does not fall at any tested setting, so we keep 1.0 and the rule's 0-0.8 window, which costs nothing.
- Does the style survive? By style_margin, yes (within the tie band at every tested setting). That metric does not measure stiffness, so B's style ratings decide.

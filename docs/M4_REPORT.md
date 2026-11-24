# Milestone 4 Report: Pose Control and a Usable Web App

**Team:** A (control engine, sweep), B (evaluation), C (pose inputs), D (app, post-processing, content policy;
assembles this report). **Assembled by:** D.

> **Rules for this file**
> - Every number comes from a file in `results/`, the rating sheets, or an owner's section file, and names its
>   source next to it. "Not tested" is an acceptable entry. No number is typed in from memory or from a chat.
> - Owners' sections are pasted, not rewritten: `CONTROL.md` (A) → §4, `POSE.md` (C) → §5, `APP.md` +
>   `CONTENT_POLICY.md` (D) → §6, `EVAL_REPORT_M4.md` (B) → §3, §7, §8.
> - `<<pending: owner, file>>` marks text still to come. Where two sections disagree, the owner is asked; nothing
>   is resolved silently.
> - B audits this file line by line before the `m4-v1.0` tag (`eval/CLAIMS_AUDIT_M4.md`).
> - **If pose control did not fix faces and hands, §7 says so in its first paragraph.**

---

## 1. Goal and system

**Goal.** A user types a scene from the Indian epics and can give a pose: a preset, a photo, or nothing. The app
returns a painting in Raja Ravi Varma's style. M4 adds **pose control (ControlNet)** and **a web app** to the M3
pipeline. It also tests the claim made since M2: *"faces and hands are our weak spot, and pose control is what we
plan to do about it."*

```
                          ┌───────────── [C] pose inputs ─────────────┐
 user text ─► M3 expand(text, "raw") ─► prompt    preset | photo | from-text | painting ─► ControlImage
                               │                                                              │
                               └─────────► [A] generate_controlled (SD1.5 + LoRA + ControlNet) ◄┘
                                                        │
                                       [D] post: hi-res / face restoration ─► [D] Gradio app + content policy
 [B] evaluates every stage: pose metric, blind ratings, stats, held-out     [A] sweeps the LoRA × ControlNet trade-off
```

**Where M1–M3 left us** (half a page; sources `docs/M3_REPORT.md`, `docs/HANDOFF_M4.md`):
- **M1–M2, the style model:**
  - a curated set of Raja Ravi Varma paintings (147 training images, 15 held-out paintings never trained on);
  - a LoRA on Stable Diffusion 1.5 (`rrv_v1`, the M2 winner `D_lr2e-4` at step 800) that knows Varma's *style*
    but not the *stories*.
- **M3, the text side:**
  - a parser and knowledge base, a prompt expander (template and local LLM), and a pre-registered evaluation.
  - **Main finding:** the user's own words with the LoRA (**c1**, raw text) were more faithful than every expanded
    prompt. Dev: c1 → full system −0.72 [−1.25, −0.25]. Held-out: −0.50 [−0.83, −0.17]. So c1 is the default
    pipeline, and M4 starts from it.
  - SHAP showed that the **action** phrase carries most of the faithfulness (φ +0.230).
  - Composition (wrong figure count + wrong action) was 25% of c1's main failures on dev and 46% on held-out. Pose
    control targets exactly that.
- **What M4 inherits unchanged:**
  - the c1 prompt `rrvarma style, <text>, oil painting, canvas texture, classical indian art`;
  - negative `ugly, blurry, bad anatomy`;
  - LoRA scale 1.0, DPM++ 2M, 30 steps, CFG 7.0, 512×512, all from `config/m3.json`.
  - There is no LLM anywhere in M4.

## 2. Research question and what is new since M3

**Research question.** *Does pose control fix what is wrong with our images (figure count, limbs, posture, faces,
hands), and what does it cost in style?*

**What is new:**
1. **A controlled ablation p0–p8** on 16 scene + pose pairs (4 per member), plus 6 private held-out pairs (§3, §7).
   Every condition uses the same c1 prompt, so the pose is the only variable.
2. **A LoRA × ControlNet interaction sweep** (A): how pose strength trades against style strength, and the default
   chosen by a pre-registered rule (§4).
3. **A pose-source study** (C): a photo vs a preset vs a pose made from the text alone (§5).
4. **A post-processing study** (D): does hi-res or face restoration help faces, or does it wash out the painted
   look (§6)?
5. **A pose-adherence metric checked against blind human ratings,** with **faces and hands rated separately** (B,
   §8).
6. **A Gradio web app with a content policy:** an input blocklist, a visible "not an authentic work" label, upload
   deletion, and an output filter decided by measurement (§6).

**Proposal coverage, stated honestly:**
- **Posture:** covered by OpenPose body keypoints.
- **Gaze direction and drapery:** **not tested.**
- **Dignity and balance:** enter through the style-rating anchor.
- **Memory:** diffusion runs in **fp16**, not quantized (A's hardware table, §4). The quantization evidence is M3's
  4-bit local LLM.

## 3. Method
`<<pending: B, EVAL_REPORT_M4.md §1>>`: the pairs (tiers, people count, dev vs held-out), the conditions p0–p8, the
metrics and their limits, the rating protocol and anchors, the raters, and the hardware.

Hypotheses are pre-registered per owner in `experiments/HYPOTHESES_M4.md`. Each section's commit predates that
owner's first scored result (B confirms in the audit):
- **A**, H-A1–A5: control vs style knee, guidance window, LoRA above 1.0, p1 vs p0, p8.
- **B**, H-B1–B4: pose match and count, faces and hands, PCK validity, style cost.
- **C**, H-C1–C3: `pose_from_text` accuracy, p3 vs p1, the copy check.
- **D**, H-D1–D4: hi-res, face restoration vs style, the memorisation guardrail, app latency.

## 4. Control engine and the style/pose trade-off (A)
*Pasted from `docs/CONTROL.md` (A, #139) on 24 Nov. Edit it there; this copy is refreshed at assembly.*

Final. Numbers trace to `results/`; human ratings are in B's `EVAL_REPORT_M4.md`.

### 4.1 What I built
`pipeline/control_api.py`: `generate_controlled(...)` runs SD1.5 + M3's Varma LoRA + one OpenPose ControlNet, derived from M3's own pipeline with `from_pipe`. The LoRA scale is set only through M3's `_ensure` (M2's `set_scale`), never `cross_attention_kwargs`. M3's `generate(..., control_image=X)` delegates to it.
`pipeline/run_batch.py` gained `--pairs`, `--control {none,openpose,openpose_full,canny}`, `--control_profile {baseline,tuned}`, `--control_source {pair,auto}` and the sweep-only `--control_scale --guidance_start --guidance_end --lora_scale`. `--condition` is mandatory and overrides the label in the prompt records. Manifests record the control settings, pose-file sha256, ControlNet revision, LoRA scale and negative used, config hashes and library versions.
Sweep: `pipeline/control_sweep.py` (generation) and `pipeline/control_score.py` (scoring, grids, plot).
Reproduce p1: see the pins block in the README (one `run_batch.py --control openpose --control_profile baseline` command per condition).

Pins: diffusers 0.40.0, transformers 5.17.0, peft 0.21.0, accelerate 1.15.0, controlnet_aux 0.0.10, gradio 6.31.0, torch 2.14.1+cu130; OpenPose ControlNet revision 9ae9f970358db89e211b87c915f9535c6686d5ba.
Sanity checks (one T4 session, all passed): scale 0 and LoRA 0.5/1.0 match M3 exactly (0.0/255, DINOv2 1.0); LoRA 0 differs from 1.0 by 41.4/255; a black control image runs; identical inputs give identical images.

### 4.2 Result
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

### 4.3 What it means
- The pose knee is at control 0.6 to 0.8: f01 goes from 0.25 to 0.81 between 0.4 and 0.6.
- style_margin shows no clear cost of control: at LoRA 1.0 it spans 0.015 across control 0.4-1.2, half the 0.0295 tie band. Only LoRA 0.6 loses style as control rises.
- H-A1: the pose half held; the style half is not shown by this metric. H-A2: pose is unchanged at guidance_end 0.8 vs 1.0 (0.817 both) and style differs within the tie band, so neither supported nor refuted. H-A3: not supported; LoRA 1.2 gives no consistent style gain (control 1.0: 0.1098 vs 0.1145; control 1.2: 0.1161 vs 0.1141). H-A4 (p1 beats p0 on human pose match and figure count, but not on faces or hands) is decided on B's blind ratings in `EVAL_REPORT_M4.md`; automated pose_pck rises from 0.100 (p0) to 0.816 (p1). H-A5: p8 not run.
- On 16 pairs the tuned setting is indistinguishable from the baseline (pose_pck 0.809 vs 0.807, style 0.1086 vs 0.1070; identical on 8 of 16 pairs), and the three-way tie has CI edges of -0.016 and -0.014. Tuning gives no gain over the baseline; the 0-0.8 window costs nothing and gains nothing.
- c1.2_l1.0 has the highest pose match on 16 pairs (+0.025 over the baseline, CI includes 0), driven by s02 (+0.235) and s01 (+0.126), and it loses 0.1 or more on s18, s19 and s27, so it is not shown to be better.
- Slider help text (in the app, #137). Style strength (0.6 to 1.2, default 1.0): "How strongly the picture follows Varma's painted look. Lower numbers look more generic and glossy. 1.0 is the default. Going above 1.0 did not add more style in our tests." Pose strength (0.4 to 1.2, default 1.0): "How closely the figures follow your pose. Below about 0.6 they often ignore it. 0.8 to 1.0 follows it well. Above 1.0 adds little for simple poses. Unusual poses, such as bending right over, can be misread (in one test a bent-over figure turned into a horse)."

### 4.4 Limits
Stage 1 used 3 fixtures (f03 is dev scene s01, because M3's fixtures have no 3-figure scene) and 2 seeds, and only finds the region. Stage 2 used 16 pairs, one seed, bootstrap over 16 pairs. pose_pck uses the detector family that drew the skeletons (circular), and painted figures are hard to detect, so human pose-match ratings are the main evidence. style_margin measures similarity to Varma references and cannot see stiffness, faces or hands. One base model (SD1.5, 512x512), fp16 not quantized, hardware on one T4. Gaze and drapery were not tested.

### 4.5 For the report's Q&A
- Does ControlNet fix faces? That is answered by B's blind face and hand ratings (`EVAL_REPORT_M4.md`). None of our automated metrics scores faces, and body-only OpenPose carries only coarse head keypoints.
- Why this pose strength? Pose match rises steeply up to control 0.6-0.8 and flattens after 1.0 (stage 1: 0.46 to 0.72 to 0.82 to 0.86 at 0.4, 0.6, 1.0, 1.2). On 16 pairs, 1.2 is not shown to beat 1.0 (paired CI against the baseline [-0.016, +0.069]) and style_margin does not fall at any tested setting, so we keep 1.0 and the rule's 0-0.8 window, which costs nothing.
- Does the style survive? By style_margin, yes (within the tie band at every tested setting). That metric does not measure stiffness, so B's style ratings decide.

**Figure:** the stage-1 grids and the trade-off plot are on A's Drive (`m4/sweep/stage1_results/`: `grid_f01`, `grid_f02`, `grid_f03`, `grid_window`, `tradeoff.png`). `<<pending: A or D, copy tradeoff.png into docs/figures/>>`.

## 5. Pose inputs and the copy check (C)
`<<pending: C, POSE.md>>`: the renderer, the preset library, `pose_from_text` accuracy (H-C1), photo extraction
robustness, p3 vs p1 (from B's tables), and p4 with the copy check, if it ran.

## 6. App, post-processing and content policy (D)
`<<pending: D, from docs/APP.md and docs/CONTENT_POLICY.md once their measurements are in>>`

## 7. Results (B)
`<<pending: B, EVAL_REPORT_M4.md §2-§3>>`:
- the answer to "does ControlNet fix faces and hands?", in plain words, **first**;
- dev and held-out tables;
- paired comparisons (p0→p1, p1→p2, p1→p3, p1→p5, p1→p6), per tier and per people count;
- faces and hands, kept separate;
- the failure taxonomy;
- 6 comparison grids, including failures.

**Weak-spots table** (updated row by row as results land):

| Weak spot (since M2) | What M4 did | M4 result |
|---|---|---|
| Faces and hands | pose control (p1/p2); hi-res (p5); face restoration (p6) | `<<from B: face and hand ratings p0→p1, p1→p5, p1→p6>>` |
| Figure count and posture | OpenPose from a photo, a preset or the text | `<<from B: pose match, n_people_out = expected, p0→p1, p1→p3>>` |
| No human test of the system | blind ratings of style, pose match, faces and hands; ≥ 2 raters per image | `<<from B: inter-rater agreement>>` |
| Safety checker blocks paintings | M2 checker off; input blocklist + label; output filter by the pre-set rule | 1 of 15 real paintings flagged by the NSFW classifier → soft blur-and-confirm (`results/audit_real_paintings.json`, `docs/CONTENT_POLICY.md` §5) |

## 8. Metric validation and inter-rater agreement (B)
`<<pending: B, EVAL_REPORT_M4.md>>`:
- Spearman of `pose_pck` vs human pose match (H-B3), and of `style_margin` vs the style rating;
- inter-rater agreement per score;
- the detector audit on the 15 real paintings (`results/audit_real_paintings.json`: a person was found in 15/15,
  with 14.4 of 18 joints per person on average).

## 9. Limitations
These hold whatever the results show; owners add their own:
- **Small n:** 16 dev pairs and 6 held-out pairs, 2 seeds. No significance claims.
- **Circular pose metric:** PCK uses the same detector family that drew the control skeletons. Robes and stylised
  proportions reduce detection. Human pose match is the main evidence.
- **Raters knew the project:** all four raters built the conditions. Blind codes hide the condition, not the intent.
- **One model:** SD1.5 at 512×512, one LoRA, one ControlNet family (OpenPose body; Canny optional).
- **Poses from a few sources:** team photos, stock photos of models (D's pairs; team decision, 21 Nov), and
  templated presets. **Body keypoints only,** except the optional p8.
- **Gaze direction and drapery: not tested.**
- **fp16 diffusion, not quantized.**
- **CLIP and DINO see 512 images,** so hi-res gains are judged on non-blind face crops and the blind ratings, not the
  automated metrics.
- **The content policy is a blocklist:** paraphrases, other languages and new names can get through, and there is no
  human moderation (`docs/CONTENT_POLICY.md` §7).

## 10. Future work
- **SDXL or another higher-resolution base,** with the LoRA retrained, so faces and hands get more pixels.
- **Training at a higher resolution** on the Varma set, rather than post-hoc hi-res.
- **Hand-specific control** (hand-pose or depth ControlNets) and face-detail inpainting with the LoRA.
- **Reference-image identity conditioning** (IP-Adapter class) for recognisable characters, which M3 showed text
  alone does not give.
- **A user study** with people outside the team, on the app itself.

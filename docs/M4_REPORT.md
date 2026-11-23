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
`<<pending: A, CONTROL.md>>`. Facts already on `main`, to be replaced by A's section:
- **Default (Overview §10, rule 1; `config/m4.json` `control.tuned`, #123):**
  - control scale 1.0;
  - guidance window 0–0.8;
  - LoRA 1.0.
- **Hardware (A, Status #17, 22–23 Nov):**

  | Setup | Peak VRAM | Time |
  |---|---|---|
  | SD + LoRA + OpenPose ControlNet, fp16 | 6.19 GB | 19.4 s/image |
  | + the img2img 768 pass | 7.07 GB | — |
  | both passes | — | 36.7 s |

  Everything fits one T4 (14.6 GB) without offloading. Model CPU offload would give 4.94 GB / 29.4 s; sequential
  offload 0.82 GB / 61.6 s.

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

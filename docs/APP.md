# APP.md: web app and post-processing (owner D)

Updated 24 Nov 2026 with B's dev ratings. Anything marked `<<pending: ...>>` has not been measured yet; no number in this file is
estimated. Files in `results/` are the source for every metric.

## 1. What I built
**The panel** (`app/app.py`, Gradio): scene text + pose mode (none / preset / upload photo / auto from text) + style
slider (LoRA scale) + pose slider (control scale) + seed + post-processing checkboxes. Outputs: the painting (with the
content label), the skeleton, the prompt and parsed characters, and a settings JSON. The prompt is M3's c1 raw prompt,
so no LLM is loaded. The model is loaded once at launch, one job runs at a time (queue), and every failure (empty or
over-long text, blocked prompt, no person in a photo, out of memory) gives a friendly message, not a traceback.
Defaults and slider ranges come from `config/m3.json` and `config/m4.json` (currently style 0.6–1.2, pose 0.4–1.2, the
ranges A's sweep covered).

**Post-processing** (`post/`):
- *Hi-res (p5):* Lanczos 512→768, then SD img2img at strength 0.35 with the same weights, LoRA, prompt and negative
  (`from_pipe`, so the LoRA state is shared).
- *Face restoration (p6):* GFPGAN 1.3 (`post/restore.py`).
- ESRGAN ×2 is implemented (`--method hires_esrgan`) but **was not run**, so it is not evaluated (cut list, Overview §14).

```
python app/app.py --share --auth user:password
python post/run_post.py --manifest $IMG/p1_openpose@rrv_v1/dev/manifest.json --condition p5_hires --method hires_img2img --out $IMG/p5_hires@rrv_v1/dev
python post/run_post.py --manifest $IMG/p1_openpose@rrv_v1/dev/manifest.json --condition p6_restore --method restore_gfpgan --out $IMG/p6_restore@rrv_v1/dev
```
Versions (Colab T4, Python 3.13): torch 2.14.1+cu130, diffusers 0.40.0, gradio 6.31.0, gfpgan 1.3.8, basicsr 1.4.2
(pins in `post/requirements_post.txt`). GFPGAN does not install on Python 3.13 with plain `pip`; `post/install_post.sh`
builds a patched basicsr from source. **Screenshots** (dev scenes only, 23 Nov, on a phone over mobile data through the public link with `--auth`), on Drive `m4/demo/`: three paintings of one scene (recorded on 23 Nov as text only, style 0.6 and style 1.2); one run with preset mode on but no preset picked, so no pose was used; a blocked prompt; and a failed run (TypeError; probably the scale bug fixed later in #140). **No screenshot in that folder shows a working preset or uploaded-photo pose** (A's S2 check, Status issue 17, 24 Nov, has a preset one), and none shows a photo of a person. The preset and uploaded-photo shots were not retaken. Drive link: https://drive.google.com/drive/folders/1lWFiRzGHe-4JRSgDlchg0xX0xRCYOhSr

## 2. Result
**Post-processing time on a T4** (post step only; excludes the base image). Source: `results/post_timing_dev.json` (all 32 images; from D's p5/p6 manifests on Drive); the end-to-end app numbers below are in `results/app_latency.json`:

| step | median per image | note |
|---|---|---|
| p5 hi-res (img2img 768, 10 steps) | 17.3 s | slowest 48.6 s (probably model load) |
| p6 face restoration (GFPGAN) | 0.84 s | slowest 12.1 s (model load) |

**End-to-end latency and peak VRAM** (`results/app_latency.json`; Tesla T4, torch 2.14.1+cu130; through the app's
own handler, so policy, parsing, prompt, pose, generation, post-processing, label and PNG save are all included; scene
"Hanuman kneeling before Rama and offering him Sita's ring"; tuned pose profile; median of 5 warm runs):

| configuration | median | min-max | peak VRAM |
|---|---|---|---|
| text only (no pose) | 5.95 s | 5.64-6.23 s | 2.62 GB |
| + preset pose (kneeling, ControlNet) | 22.53 s | 22.33-22.67 s | 6.18 GB |
| + hi-res (img2img 768) | **40.34 s** | 40.05-40.63 s | 7.07 GB |
| + hi-res + face restoration | 40.76 s | 40.55-41.51 s | 7.58 GB |

- **Cold start:** 39.0 s. That covers loading the app's models from the disk cache plus the first image; a brand-new
  runtime adds the model download (not measured).
- **Hi-res cost:** about 17.8 s on top of the pose image. Face restoration adds about 0.4 s.
- **Memory:** everything fits one T4 (14.6 GB) at once, with no offloading. This matches A's hardware table.
- **H-D4 (text + pose + hi-res under 60 s): supported** (40.3 s).

**Automated scores, p5 and p6** (M3 metrics; `results/p5_hires@rrv_v1_dev.json`, `results/p6_restore@rrv_v1_dev.json`;
all 16 dev pairs × 2 seeds, rescored 23 Nov with `--pairs`; p1 for reference from `results/p1_openpose@rrv_v1_dev.json`):

| | checklist | style_margin | clip_raw | max_train_dino mean / p95 |
|---|---|---|---|---|
| p1 (reference: `results/p1_openpose@rrv_v1_dev.json`) | 0.587 | 0.102 | 0.270 | 0.688 / — |
| p5 hi-res | 0.571 | 0.094 | 0.272 | 0.696 / 0.815 |
| p6 restore | 0.587 | 0.102 | 0.270 | 0.689 / 0.810 |

p5 and p6 are made from D's own p1 base run (`run_batch.py`, baseline profile, the same prompts and seeds). B checked s09 seed 1: D's base image is byte-identical to A's rated p1 image (sha256 `8d63f5db…`, `docs/TEAM_DECISIONS_M4.md` D2), so p1 → p5/p6 compares the same base (compared with A on one image; the same code, settings and seeds produced the rest). All 32 p5 and 32 p6 rows record a `source_sha256` that matches D's p1 base file (`results/post_timing_dev.json`). Paired comparisons with CIs are in `eval/EVAL_REPORT_M4.md`. Read as means only:
- p6 matches p1 on every automated metric. GFPGAN changes only the face region, which these whole-image metrics
  barely see.
- p5 is slightly lower on checklist and `style_margin`. This is consistent with img2img redrawing details.

H-D3 (`max_train_dino` p95 ≤ 0.85): **holds for both** on 16 pairs (0.815 and 0.810). **CLIP and DINO see a resized image, so they cannot
show resolution gains.** The question "do these steps help faces?" is answered by B's blind face, hand and style ratings
(`eval/EVAL_REPORT_M4.md` §2.2, `results/m4_tables_dev.md`) and by the face-crop strips (`docs/figures/post_face_strips.png`, `post/face_strips.py`: s09, s19 and s31, seed 1, p1 | p5 | p6, crops centred on the control skeleton's head; not blind).

**Blind ratings, p1 → p5 / p6** (paired over 16 pairs, seed 1; `docs/TEAM_DECISIONS_M4.md` D2, `results/m4_tables_dev.md`):

| p1 → | faces (rule: > +0.25) | hands (rule: ≥ −0.25) | style (rule: > −0.3) | passes rule 2 |
|---|---|---|---|---|
| p5 hi-res | +0.31 [+0.12, +0.53] | +0.25 [+0.00, +0.50] | −0.03 [−0.25, +0.22] | **yes** |
| p6 face restore | +0.19 [+0.00, +0.34] | +0.25 [+0.09, +0.41] | +0.12 [−0.06, +0.31] | no (faces) |

**What is on by default** (Overview §10 rule 2, applied at S3):
- **Hi-res** passes the rating test, but not the spot check. D's spot check (24 Nov, T4; s09, s19, s27, s31, seed 1) found visible new artifacts, so by the rule hi-res stays an **opt-in checkbox** and `post.default_on` stays `false`. s09: dark marks on the toes, and the resting hand loses its fingers. s27: specks on the child's chest, and the child's foot becomes hand-like. s19 and s31 showed no clear new artifact (s19's faces improved; s31's bottom-left hand looked softer but was not examined closely). No figure was duplicated. The four p2 images were regenerated with the tuned profile and were not compared with A's files; the check was by eye, not blind, read from the notebook output by Claude, and the opt-in decision was D's. B's held-out run agrees: p1 → p5 faces −0.08 [−0.42, +0.25], which would not pass the rule (`docs/TEAM_DECISIONS_M4.md` D7).
- **Face restoration** stays an opt-in checkbox.

**Caveats (B):** faces passed by 0.06, while two raters scoring the near-identical p1/p2 images already differ by
0.25 on faces. Hands agreement is low (weighted kappa 0.14). p1 and p5/p6 were rated in different rounds.

**Uploads:** deleted when the request finishes, and **within N = 6 minutes** in any case. The Gradio temp path is `/tmp/gradio` (Gradio's default; `GRADIO_TEMP_DIR` overrides it). Normal case: no copy left at the first check after the response. If the handler never deletes it, Gradio's timer removed it after 360 s. (`results/app_upload_cache.json`, `results/app_upload_cache_timer.json`; measured 23 Nov with the app's real Gradio layer and handler on the pinned Gradio 6.31.0, image generation stubbed on CPU).
Photo uploads were re-checked on 25 Nov after C's cold-start test found a file left behind (issue 158): Gradio converted non-RGB images (grayscale, CMYK, transparent PNG) into a second file and the handler deleted only that copy, and an upload sent with pose mode "none" was not deleted. A plain RGB JPEG did not reproduce it. Fixed; 12 upload cases now leave nothing under the temp folder when the request ends (5 left a file before the fix; `results/app_upload_formats.json`, `app/measure_uploads.py`; real Gradio layer, fake pose and generation, CPU; not re-run on a GPU). An image that is uploaded but never submitted is removed only by Gradio's timer.

## 3. What it means
**First look by eye, before any ratings** (face-crop strips for s09, s19 and s31, plus a few full images). This is
not a rating, and B's blind ratings decide:
- img2img changes faces, not just sharpens them: at strength 0.35 some nose, mouth and eye shapes differ from the base.
- GFPGAN keeps the layout but makes faces smoother and brighter, which is the "less painted" effect H-D2 predicts.
- Faces already distorted in the base (half-hidden or asymmetric) stay distorted after both steps.
- In the strips:
  - **Hi-res redraws the face.** s19's mask-like eyes become cleaner and more natural, s09's eyes and nose change shape,
    and s31 gains detail in the crown.
  - **Restoration barely moves the layout** but smooths the skin and sharpens the eyes (most visible on s31).
  - So hi-res can *repair* a face by redrawing it, but it can also change a face that was fine. Restoration is a
    lighter touch that leans photographic.

**What the blind ratings say** (the table in §2):
- **H-D1 (hi-res improves detail without duplicating figures):** faces improve slightly (+0.31), just past the bar, and
  style holds (−0.03). **Duplicated figures did not increase:** "merged/duplicated figures" was ticked 4 times for both p1 and p5 out of 32
  ratings, and nobody ticked "post-processing artifacts" (`eval/EVAL_REPORT_M4.md` §2.5). So H-D1 holds on these
  ratings, narrowly. The eye-ball note above (hi-res can repair a face by redrawing it) is consistent with this small
  gain.
- **H-D2 (restoration raises faces but lowers style):** not supported as predicted. Faces rose only +0.19 (under the
  bar), and style did not drop (+0.12, CI includes 0). "Style weakened (photographic)" was ticked twice for both p1 and p6.
- **Neither step fixes faces or hands in a large way.** The largest face gain (+0.31) is about the size of the
  rater-to-rater noise (0.25), and it did not replicate on the held-out pairs (−0.08).

## 4. Limits
- Native resolution is 512 (SD1.5). Hi-res is upscale-then-refine, not a true 768 generation.
- p5 and p6 were first run on 12 of the 16 dev pairs, because C's four pairs merged later. On 23 Nov the base images
  were completed and all 16 pairs were scored.
- n is small (12–16 pairs, 2 seeds): no significance claims.
- CLIP and DINO cannot see resolution gains; hi-res detail is judged on crops and is not blind.
- ESRGAN was not run. GFPGAN was tested on SD paintings only. A face was found in 31 of 32
  p6 images; `s25_s2` (a leaping figure) had none (`results/post_timing_dev.json`).
- Free Colab: one GPU session, one job at a time, the session can time out, and the public link dies with it.
- **Photo poses are live** (`pose/extract.py`, C, #156 and #157; live run reported in `docs/cold_start_25nov.md`), but they were not rated:
  p1's photo skeletons came from `eval/make_pair_pose.py`, not from this path.
- Uploads pass through Gradio's temporary cache, and the link goes through Gradio's relay.
- The output NSFW filter is a soft blur-and-confirm backstop (`app/nsfw.py`, `docs/CONTENT_POLICY.md` §5). It sits
  at the boundary of B's audit (1 of 15 real paintings flagged, a false positive), so it will blur some innocent
  paintings, and its flag rate on our own outputs is not measured. The pose and the filter were exercised with fakes on
  CPU; the live GPU run is the S2 check.

## 5. For the report's Q&A
**Can a user generate harmful content?** The input blocklist and category lexicon refuse sexualised or nude depictions,
demeaning depictions of deities, hateful content, real named living people and gore, and every image is labelled. It is a
blocklist: paraphrases, misspellings and other languages can get through. The output passes a soft NSFW blur-and-confirm
filter (§4, `docs/CONTENT_POLICY.md` §5). There is no human moderation (`docs/CONTENT_POLICY.md` §7).

**Where do uploads go?** A photo is meant only for extracting a skeleton and is deleted by the app as soon as the skeleton
exists (measured: no copy left when the request finished, after the issue 158 fix, §2). If that ever fails, Gradio's own timer removes it; measured at 360 s, so the app
promises deletion within 6 minutes. We do not claim they
are "not saved".

**Is this a Ravi Varma?** No. The images are *in the style of*, not by, Raja Ravi Varma. They carry that label on the image
and in the file metadata, and are for research and coursework only.

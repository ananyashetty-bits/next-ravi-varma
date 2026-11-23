# APP.md: web app and post-processing (owner D)

Draft of 22 Nov 2026. Anything marked `<<pending: ...>>` has not been measured yet; no number in this file is
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
APP_POST_MODULE=post.api python app/app.py --share --auth user:password
python post/run_post.py --manifest $IMG/p1_openpose@rrv_v1/dev/manifest.json --condition p5_hires --method hires_img2img --out $IMG/p5_hires@rrv_v1/dev
python post/run_post.py --manifest $IMG/p1_openpose@rrv_v1/dev/manifest.json --condition p6_restore --method restore_gfpgan --out $IMG/p6_restore@rrv_v1/dev
```
Versions (Colab T4, Python 3.13): torch 2.14.1+cu130, diffusers 0.40.0, gradio 6.31.0, gfpgan 1.3.8, basicsr 1.4.2
(pins in `post/requirements_post.txt`). GFPGAN does not install on Python 3.13 with plain `pip`; `post/install_post.sh`
builds a patched basicsr from source. Screenshot: `<<pending: D>>`. Demo: `<<pending: D, Drive link>>`.

## 2. Result
**Post-processing time on a T4** (post step only, from the p5/p6 manifests; excludes the base image):

| step | median per image | note |
|---|---|---|
| p5 hi-res (img2img 768, 10 steps) | about 17.2 s | about 1.39 s/step |
| p6 face restoration (GFPGAN) | 0.84 s | first image 8.9 s (model load) |

**End-to-end latency and peak VRAM** (text · +preset pose · +hi-res · +restore; median of 5 warm runs; cold start
separately): `<<pending: D, results/app_latency.json>>`. This is H-D4.

**Automated scores, p5 and p6** (M3 metrics; `results/p5_hires@rrv_v1_dev.json`, `results/p6_restore@rrv_v1_dev.json`;
12 pairs × 2 seeds at the time of the first run, 16 pairs `<<pending: D, rerun>>`):

| | checklist | style_margin | clip_raw | max_train_dino mean / p95 |
|---|---|---|---|---|
| p5 hi-res | 0.589 | 0.095 | 0.274 | 0.691 / 0.812 |
| p6 restore | 0.594 | 0.104 | 0.267 | 0.683 / 0.811 |

H-D3 (`max_train_dino` p95 ≤ 0.85): holds for both on these 12 pairs. **CLIP and DINO see a resized image, so they cannot
show resolution gains.** The question "do these steps help faces?" is answered by B's blind face, hand and style ratings
(`<<pending: B, EVAL_REPORT_M4.md>>`) and by the face-crop strips (`<<pending: D, docs/figures>>`).

**What is on by default:** both post-processing boxes are off (`post.default_on = false` in `config/m4.json`). The S3
rule (Overview §10, rule 2) is applied once B's dev ratings are in: `<<pending: B ratings; D applies the rule>>`.

**Uploads:** deleted when the request finishes, and **within N = 6 minutes** in any case. The Gradio temp path is `/tmp/gradio` (Gradio's default; `GRADIO_TEMP_DIR` overrides it). Normal case: no copy left at the first check after the response. If the handler never deletes it, Gradio's timer removed it after 360 s. (`results/app_upload_cache.json`, `results/app_upload_cache_timer.json`; measured 23 Nov with the app's real Gradio layer and handler on the pinned Gradio 6.31.0, image generation stubbed on CPU).

## 3. What it means
**First look by eye, before any ratings (3 scenes and 2 face crops; not a rating):**
- img2img changes faces, not just sharpens them: at strength 0.35 some nose, mouth and eye shapes differ from the base.
- GFPGAN keeps the layout but makes faces smoother and brighter, which is the "less painted" effect H-D2 predicts.
- Faces already distorted in the base (half-hidden or asymmetric) stay distorted after both steps.

Whether either step is worth turning on depends on B's ratings against the S3 thresholds: face up by more than 0.25,
hands not down by more than 0.25, style down by less than 0.3. `<<pending: B ratings>>`. I report the result as it comes
out, including if post-processing does not help.

## 4. Limits
- Native resolution is 512 (SD1.5). Hi-res is upscale-then-refine, not a true 768 generation.
- p5 and p6 were first run on **12 of the 16 dev pairs**: C's four pairs merged after the run. They cover A's, B's and
  D's pairs only. `<<pending: D, 16-pair rerun>>`.
- n is small (12–16 pairs, 2 seeds): no significance claims.
- CLIP and DINO cannot see resolution gains; hi-res detail is judged on crops and is not blind.
- ESRGAN was not run. GFPGAN was tested on SD paintings only; 23 of 24 images had a face found, and one (`s25_s2`, a
  leaping figure) had none.
- Free Colab: one GPU session, one job at a time, the session can time out, and the public link dies with it.
- Uploads pass through Gradio's temporary cache, and the link goes through Gradio's relay.
- The output NSFW filter is a soft blur-and-confirm backstop (`app/nsfw.py`, `docs/CONTENT_POLICY.md` §5). It sits
  at the boundary of B's audit (1 of 15 real paintings flagged, a false positive), so it will blur some innocent
  paintings, and its flag rate on our own outputs is not measured. The pose and the filter were exercised with fakes on
  CPU; the live GPU run is the S2 check.

## 5. For the report's Q&A
**Can a user generate harmful content?** The input blocklist and category lexicon refuse sexualised or nude depictions,
demeaning depictions of deities, hateful content, real named living people and gore, and every image is labelled. It is a
blocklist: paraphrases, misspellings and other languages can get through, there is no human moderation and the output
is not classified (`docs/CONTENT_POLICY.md` §7).

**Where do uploads go?** A photo is used only to extract a skeleton and is deleted by the app as soon as the skeleton
exists (measured: no copy left when the request finished). If that ever fails, Gradio's own timer removes it; measured at 360 s, so the app
promises deletion within 6 minutes. We do not claim they
are "not saved".

**Is this a Ravi Varma?** No. The images are *in the style of*, not by, Raja Ravi Varma. They carry that label on the image
and in the file metadata, and are for research and coursework only.

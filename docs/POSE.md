# POSE.md: pose conditioning, presets, text-to-pose and photo extraction (owner C)

Updated 25 Nov 2026 with B's dev and held-out evaluation metrics (`results/m4_extras_dev.json`, `eval/EVAL_REPORT_M4.md` §2.7) and cold-start verification log (`docs/cold_start_25nov.md`).

## 1. What I built

- **Canonical `PoseJSON` schema & renderer (`pose/skeleton.py`):** 18-point OpenPose body keypoint format (plus optional face/hand points and figure labels) rendered onto a 512×512 black canvas with standard OpenPose limb colors and joint circles for ControlNet (`lllyasviel/control_v11p_sd15_openpose`).
- **Preset pose library (`pose/presets.py`, `pose/library/`):** 19 curated `PoseJSON` templates covering single-figure, two-figure, and three-figure compositions (`pose/library/status.json`).
  - **ControlNet-tested (`4/19`, baseline profile, 2 seeds, neutral prompt, T4):** `standing`, `seated`, `kneeling`, `kneeling_before_standing`.
  - **Untested (`15/19`, valid `PoseJSON` passing CI layout/bounds checks, not yet run through ControlNet):** `lying`, `raised_arms`, `namaste`, `kneel_namaste`, `archer`, `pointing`, `walking`, `embrace`, `seated_with_attendant`, `bowing_pair`, `standing_pair`, `walking_pair`, `standing_trio`, `seated_trio`, `kneeling_before_two`.
- **Rule-based `pose_from_text` (`pose/text.py`):** Parses scene text via M3's character/action parser (`kb/parser.py`) to count figures (1–3) and map verbs/roles (e.g., kneeling before a standing/seated figure, seated, standing, walking, bowing) to layout presets (`p3_pose_from_text`).
- **Real photo pose extraction (`pose/extract.py`, wired in `pose/__init__.py`):** Extracts OpenPose body skeletons (`extract_pose`) and edge maps (`extract_edges`) from user-uploaded photos using `controlnet_aux` (`OpenposeDetector`), resizing cleanly and returning structured warnings when no person is detected or input dimensions are invalid.

## 2. Result

### `pose_from_text` (`p3`) accuracy and human ratings (from B)
- **Figure count accuracy:**
  - **Dev (`results/m4_extras_dev.json`):** `p3` draws the right number of figures in **15/16** seed-1 images.
  - **Held-out (`eval/EVAL_REPORT_M4.md` §2.7):** `p3` gets **4/6** scenes right; it drew 1 figure for `m4h04` (2 people) and `m4h06` (3 people).
- **Human pose-match rating (`p3` vs photo-extracted `p1`):**
  - **Dev:** `p3` scores **1.78** vs `p1` **3.72** (`eval/EVAL_REPORT_M4.md`).
  - **Held-out:** `p3` scores **2.25** vs `p1` **3.83** (`eval/EVAL_REPORT_M4.md` §2.7).
- **H-C1 (`pose_from_text` figure count & pose match):** Figure count accuracy is high on dev (15/16) and moderate on held-out (4/6), while human pose-match rating for coarse rule-based templates (`1.78` dev / `2.25` held-out) is far below photo-extracted poses (`3.72` dev / `3.83` held-out).

### Photo extraction (`pose/extract.py`) & Cold-start verification
- **Automated tests:** `pytest -q tests` on `main` passes (**452 passed, 1 skipped** in Colab; 435 passed, 10 skipped on CPU without optional packages). **No automated test runs the real `extract_pose`**: the tests that mention it use fakes or the stub (`tests/test_app_pose_wiring.py`, `tests/test_app_handler.py`, `tests/test_pose_stub.py`), and `tests/test_pose_api.py` only checks that the name is exported. What was checked on the real function was done by hand in the cold-start run below (one photo with a person, one with no person, which gave a friendly warning and no traceback). Grayscale, RGBA and non-square inputs were not tested.
- **Cold-start test (`docs/cold_start_25nov.md`, fresh Colab T4 session, 25 Nov 2026):**
  - **CLI seed reproduction:** Verified (`s01_s1.png` reproduced deterministically).
  - **Gradio UI (`app/app.py --share --auth user:password`):** Verified all 6 UI paths (misspelled name `"Arjunaah..."` with `auto from text`, 2-figure `preset`, `upload photo`, empty prompt, 300-word prompt, and no-person photo) complete with friendly UI messages and no Python tracebacks. Filed issue #158 (`needs:D`) regarding uploaded input photos remaining in `/tmp/gradio` until Gradio's 360 s timer when the handler finishes.

## 3. What it means

- **Figure count works on simple scenes, misses harder multi-figure phrasing; fine articulation does not:** `pose_from_text` got the right number of figures on **15/16** dev images, but only **4/6** held-out scenes (drawing `m4h04` [2 people] and `m4h06` [3 people] as a single figure). Even when headcount is right, generic preset poses cannot capture scene-specific limb angles, hand gestures, or interactions—explaining the 1.5–1.9 point gap in human pose-match ratings behind `p1`.
- **Presets are dependable for core postures:** The 4 ControlNet-tested presets (`standing`, `seated`, `kneeling`, `kneeling_before_standing`) give reliable spatial anchoring for common epic compositions without needing a reference photo.
- **A reference photo gives a much better pose than text:** The `p1` skeletons (human pose match 3.72 dev, 3.83 held-out) were made from the reference photos with `eval/make_pair_pose.py` (B's script), not with `pose/extract.py`. `extract_pose` is the same kind of step inside the app (photo in, OpenPose skeleton out), but it has not been scored on its own; the one manual no-person check gave a friendly warning (`docs/cold_start_25nov.md`).

## 4. Limits

- **15 of 19 presets remain untested on GPU:** Only 4 core presets have been verified end-to-end through ControlNet (`pose/library/status.json`); the remaining 15 pass `PoseJSON` schema/bounds checks only.
- **Rule-based parser misses complex multi-person phrasing:** On held-out scenes (`m4h04` with 2 people, `m4h06` with 3 people), `pose_from_text` under-counted figures and fell back to a single skeleton.
- **Body-only OpenPose keypoints:** `PoseJSON` and `extract_pose` use 18-point body skeletons without detailed finger or facial landmarks, so hand mudras and gaze direction are unconstrained.
- **No `p4` style-transfer/copy-check run:** `p4` was not run in the M4 evaluation sweep.

## 5. For the report's Q&A

- **Why is `pose_from_text` (`p3`) rated much lower on pose match than photo poses (`p1`)?** Because `pose_from_text` maps broad verbs to a small library of static templates rather than predicting scene-specific limb geometry. It gets the right headcount on 15/16 dev scenes (and 4/6 held-out), but the pose itself is generic (1.78 vs 3.72 on dev; 2.25 vs 3.83 on held-out).
- **What happens when a user uploads a photo with no person?** `extract_pose` detects zero people and returns a friendly UI warning instead of crashing or hallucinating a skeleton.
- **Are all presets in the dropdown tested with ControlNet?** Four core presets (`standing`, `seated`, `kneeling`, `kneeling_before_standing`) are ControlNet-tested; the other 15 are structurally validated `PoseJSON` files (`pose/library/status.json`).

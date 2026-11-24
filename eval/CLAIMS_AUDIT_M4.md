# M4 claims audit (owner: B), draft of 24 Nov 2026

The rule (Member B guide §3.6): every number in the team's documents must trace to a file in `results/`, the
ratings, or a script, and no sentence may claim more than that. This **draft** checks the documents on `main` at
`9eec04d` (after #145). It is re-run on the final documents before D tags `m4-v1.0`; held-out results, `POSE.md`
and the assembled `M4_REPORT.md` sections are not in yet.

**Severity:** **must fix** = no data behind a claim, or it contradicts the results. **update** = true when written,
out of date now, or not traceable to a committed file. **ok** = checked, no issue found.

## 1. Pre-registration
Each hypothesis section was committed before that owner's first scored result was committed (git history of
`main`):

| Section | First committed | Owner's first scored result in `results/` | OK |
|---|---|---|---|
| A | 21 Nov 15:02 IST (`e442569`, #97) | stage-1 sweep committed 24 Nov (`f36cc1e`, #139); p0 dev scores 23 Nov 16:19 (`885dfdd`) | ok (git shows commit order; the sweep was scored on A's Drive before it was committed) |
| B | 22 Nov 14:11 IST (`8905a5c`, #102) | first `_dev_pose` file 23 Nov 19:31 (`90c4f59`, #130); ratings 23-24 Nov | ok |
| C | 23 Nov 19:20 IST (`dd341f9`) | `c_pose_from_text_dev.json` 24 Nov 09:20 IST (`5351c83`); p3 images 24 Nov | ok |
| D | 22 Nov 20:08 IST (`9722b04`, #111) | p5/p6 scores 22 Nov 22:37 IST (`de5bd43`) | ok |

The three §10 default rules are copied verbatim into Section B (and rule 2 into Section D) before any rating.

## 2. `eval/EVAL_REPORT_M4.md` and `docs/TEAM_DECISIONS_M4.md` (B)
**ok.** Every table comes from `eval/tables_m4.py` → `results/m4_tables_dev.{md,json}`, `eval/m4_extras_dev.py` →
`results/m4_extras_dev.json`, or `eval/grids_m4.py` → `results/m4_p1_p2_similarity_dev.json` and
`docs/figures/grid_m4_dev_*.png`, all re-runnable from the rating sheets and B's private keys.
- **Correction made in this pass (B):** p1 and p2 images are near-identical (median pixel difference 2.1 of 255),
  so the first draft's "the tuned window gives slightly better faces" and "p2 brings faces back to level" were rater
  noise. Both documents now say so (same PR as this draft).
- Held-out sections are `<<pending>>`.

## 3. `docs/CONTROL.md` (A)
| # | Claim | Finding | Severity |
|---|---|---|---|
| A1 | p0/p1/p2 table (pose_pck 0.100/0.816/0.815; style_margin 0.1082/0.1016/0.1026; checklist 0.565/0.587/0.594; clip_raw 0.277/0.270/0.271) | matches `results/*_dev_pose.json` summaries | ok |
| A2 | stage-2 means (e.g. c1.2_l1.0 pose_pck 0.832, style 0.1029; c1.0_ge0.8 0.809 / 0.1086) | match `results/control_stage2_*.json` | ok |
| A3 | paired CIs [−0.016, +0.069] and [−0.014, +0.066]; p0 seed-1 style 0.1101, floor 0.0806; baseline seed-1 0.807 / 0.1070 | **not in any committed file**: `results/control_rule_16pairs.json` records only the winner and the tied set | **update**: add the CIs and seed-1 means to `control_rule_16pairs.json` (or cite the script output that has them) |
| A4 | "H-A4 ... is decided on B's blind ratings" | now decided: pose match supported (+1.59 [+1.09, +2.03]); faces worse (−0.44 [−0.78, −0.09]); hands −0.25 [−0.56, +0.06] (`EVAL_REPORT_M4.md` §2.2) | **update**: replace the forward reference with the result |
| A5 | "the tuned setting is indistinguishable from the baseline" | confirmed and stronger: the images themselves are near-identical (`results/m4_p1_p2_similarity_dev.json`) | ok (could cite it) |
| A6 | slider help text: "Going above 1.0 did not add more style in our tests"; bent-over figure → horse | backed by H-A3 (not supported) and stage-1 f03 (`control_stage1_cells.json`) | ok |
| A7 | "Does the style survive? By style_margin, yes ... B's style ratings decide" | ratings: p0 → p1 style −0.28 [−0.53, −0.03]; p0 → p2 −0.19 [−0.53, +0.19] | **update**: add the rating result; style_margin and human style correlate only 0.17 |

## 4. `docs/APP.md` (D)
| # | Claim | Finding | Severity |
|---|---|---|---|
| D1 | `<<pending: B ...>>` in §2 and §3, and "both post-processing boxes are off" | B's numbers are in: p1 → p5 faces +0.31 [+0.12, +0.53], hands +0.25, style −0.03; p1 → p6 faces +0.19, hands +0.25, style +0.12. Rule 2: hi-res passes, restore does not (`TEAM_DECISIONS_M4.md` D2) | **must fix**: fill the pendings; record the spot check and the default |
| D2 | "p5 and p6 are made from D's p1 base images ..., not from the images behind the p1 results file, so this row is a reference rather than an exact pair" | B checked s09 seed 1: D's base is byte-identical to A's p1 image (sha256 `8d63f5db…`) | **update**: check all 32 with the manifests' `source_sha256`; if they match, say the pair is exact |
| D3 | §5: "there is no human moderation and the output is not classified" | contradicts §4 and `CONTENT_POLICY.md` §5: the soft NSFW filter is implemented (`app/nsfw.py`) | **must fix**: "the output passes a soft NSFW blur (§4); there is no human moderation" |
| D4 | post time per image 17.2 s / 0.84 s (first 8.9 s); "23 of 24 images had a face found" | from manifests on D's Drive, not in `results/` | **update**: commit a small results file (or cite the manifest path) |
| D5 | latency table, cold start 39.0 s, H-D4 supported (40.34 s) | match `results/app_latency.json` | ok |
| D6 | H-D3 holds (p95 0.815, 0.810) | match the p5/p6 results files (B recomputed: 0.815, 0.810) | ok |
| D7 | uploads deleted within N = 6 minutes; no "not saved" | `results/app_upload_cache.json`, `..._timer.json` | ok |
| D8 | screenshots on Drive `m4/demo/` include "an uploaded photo"; the folder is link-shared | rule: no photos of people on the shared Drive. B did not open the screenshots | **check (D)**: confirm the uploaded-photo screenshot shows no identifiable person, or crop it / move it out of the shared folder |
| D9 | §3 eye-ball notes on the face strips ("hi-res can repair a face") | labelled as a first look, not a rating; the blind ratings now back a small face gain for hi-res | ok (add the rating line next to it) |

## 5. `docs/CONTENT_POLICY.md` (D)
| # | Claim | Finding | Severity |
|---|---|---|---|
| P1 | 1 of 15 flagged (score 0.994, nursing mother); the other 14 below 0.01 | `results/audit_real_paintings.json`: 1 flagged at 0.994; the other 14 at most 0.0033 | ok |
| P2 | rule → soft blur-and-confirm | applied as pre-registered | ok |
| P3 | "Painting uploads also stay off if C's p4 ... `<<pending: C's p4>>`" | p4 was not run (`TEAM_DECISIONS_M4.md` D5) | **update**: "p4 was not run, so painting (Canny) uploads stay off" |
| P4 | "Nothing says 'not saved'"; known gaps §7 | consistent with README and APP.md | ok |

## 6. `docs/M4_REPORT.md` (assembled by D)
| # | Claim | Finding | Severity |
|---|---|---|---|
| R1 | §2 "A controlled ablation p0–p8" | p4, p7, p8 and p9 were not run | **must fix**: "p0–p3, p5, p6 (p4, p7, p8 not run)" |
| R2 | §2 "A pose-source study (C): a photo vs a preset vs a pose made from the text alone" | presets were not rated; only photo (p1) vs text (p3) was | **must fix**: drop "a preset" or say presets were not evaluated |
| R3 | §2 "Every condition uses the same c1 prompt, so the pose is the only variable" | true for p0-p3; p5/p6 add post-processing | **update**: "the pose (or the post step) is the only variable" |
| R4 | §1 recap: 147 training images, c1 → c3 −0.72 [−1.25, −0.25] dev / −0.50 [−0.83, −0.17] held-out, φ(action) +0.230, composition 25% dev / 46% held-out | trace to `docs/M3_REPORT.md` (§3, SHAP table) and `docs/HANDOFF_M4.md` §2 | ok ("full system" = c3, the LLM + KB condition; say so) |
| R5 | §3 "Each section's commit predates that owner's first scored result (B confirms in the audit)" | confirmed in §1 above | ok |
| R6 | §8 detector audit: a person in 15/15, 14.4 of 18 joints | `results/audit_real_paintings.json` | ok |
| R7 | §9 limits | add: calibration was skipped; rater noise ≈ the 0.25 tie band (p1/p2); p1 and p3/p5/p6 rated in different rounds; hands agreement kappa 0.14 | **update** |
| R8 | §3, §5, §6, §7, §8 `<<pending>>` | paste from `EVAL_REPORT_M4.md`, `APP.md`, `CONTENT_POLICY.md`, `POSE.md` | pending |
| R9 | §4 figure: `tradeoff.png` on A's Drive, `<<pending: A or D, copy into docs/figures/>>` | not in the repo | **update** |

## 7. `README.md` (D)
| # | Claim | Finding | Severity |
|---|---|---|---|
| M1 | "A's full pins block: `<<pending: A, CONTROL.md>>`" | the pins are in `CONTROL.md` §1 | **update**: paste them |
| M2 | "Photos of people are never committed to the repo" | no `.jpg/.jpeg/.heic/.webp` has ever been added on any branch; all 16 `eval/poses/dev/*.png` are skeleton drawings | ok |
| M3 | uploads deleted within N = 6 minutes | as D7 | ok |

## 8. Not yet audited
- `docs/POSE.md` (C): not on `main` yet. H-C1 numbers will be checked against `results/c_pose_from_text_dev*.json`
  and H-C2 against `EVAL_REPORT_M4.md` (p3 below p1 on pose match: supported; figure count 15/16 vs 9/16; the
  tier pattern not supported).
- Held-out results and the reveal (`eval/test_pairs_heldout.json`, skeletons, key, ratings).
- The final `M4_REPORT.md` once all sections are pasted.

## Re-run, 25 Nov 2026 (after #152-#159; main `7790d75`)

**Closed since the first audit:** CONTROL.md numbers now in `results/control_rule_16pairs.json` (A, #149); APP/CONTENT_POLICY/M4_REPORT
fixes R1-R4, R7, D1-D4 (D, #148, #154, #155); hi-res stays opt-in (D2 + D7); post timings in `results/post_timing_dev.json`
(31 of 32 faces; 32/32 base shas match D's p1); POSE.md and real `extract_pose` merged (C, #156, #157, #159); `needs:D` issue #158 filed by C.

**Must fix before `m4-v1.0`:**
- **P1 (D, privacy claim): uploaded photos are not deleted when the request finishes.** README, `docs/APP.md` §4/§5 and `docs/CONTENT_POLICY.md` say the handler
  deletes the upload as soon as the skeleton is extracted. C's cold-start test (issue #158) found the file still in `/tmp/gradio` after the response. Either fix the handler and re-test, or
  change the text to what is true (file removed by Gradio's timer, within N = 6 min) in all three places.
- **P2 (C): POSE.md names files that do not exist:** `pose/schema.py`, `pose/render.py`, `pose/from_text.py`. The real files are `pose/skeleton.py`,
  `pose/presets.py`, `pose/text.py` (+ `pose/extract.py`).
- **P3 (C): "467 CPU tests" is not reproducible.** `pytest -q tests` on `7790d75` (CPU, Python 3.13): 435 passed, 10 skipped. Cite the actual run or the real count.
- **P4 (C): "reliably allocates the right number of skeletons" (POSE.md §3) overstates.** Dev 15/16 but held-out 4/6 (m4h04 and m4h06 drawn as one figure).
  Say "right on 15/16 dev images, 4/6 held-out scenes; misses multi-person phrasing".
- **P5 (D): stale "photo poses are a stub" text.** `docs/APP.md` lines 122 and 136 and `docs/M4_REPORT.md` lines 335, 349, 409 still say `extract_pose` is a stub; it is real since #157.
  Update, retake the preset and uploaded-photo screenshots, and clear the last `<<pending: D, retake ...>>` marker (APP.md line 28, M4_REPORT line 241).
- **P6 (C): cold-start claims have no committed evidence** (fresh-session results, "all 6 UI paths"). Add a short log or note in `docs/` or `results/`, or label it as "reported by C, not committed".

**Minor (B, this PR):** D7 said D's spot check "is no longer needed"; D had already run it (found artifacts in s09 and s27), so D2 and D7 now agree: both reasons keep hi-res off.

**Verified OK:** POSE.md figure counts (15/16 dev, 4/6 held-out) and pose-match numbers (1.78 / 3.72 dev; 2.25 / 3.83 held-out) match `results/m4_extras_dev.json` and EVAL_REPORT_M4 §2.2/§2.7;
status.json = 4 tested, 15 untested (matches POSE.md); `post.default_on` is `false`; remaining `<<pending: ...>>` markers: 1 (P5).

# Content policy (owner D) — draft v0, 20 Nov

**What this covers:** what the app (`app/app.py`) will and won't make, how that is enforced, and what is still
open.

**Status:** draft. Three things are `<<pending>>` until they are measured: the upload-deletion time N (§4), the
NSFW measurement from B (§5) and the access choice for the public link (§6).

## 1. Allowed
- Scenes from the epics and the Puranas, including **non-graphic** battles.
- Devotional, courtly and domestic subjects in the manner Raja Ravi Varma painted.
- Deities and epic figures shown **respectfully**: dignified, balanced and fully clothed in the period's dress.

## 2. Blocked at input
The checks are deterministic: `app/policy.py` with `app/blocklist.json`.

| Category | Examples refused | Message to the user (shortened) |
|---|---|---|
| Sexual / nude | "nude", "naked", "lingerie", "sexy"; "seductive" or "sensual" next to a deity or epic figure | doesn't make sexualised or nude images of anyone, especially deities |
| Demeaning | "cartoon", "meme", "clown", "drunk", "holding a gun" next to a deity or epic figure | depicts deities respectfully; a respectful version is fine |
| Hateful | hate symbols and slogans; desecration next to a figure | doesn't make hateful content about any group or faith |
| Gore | "decapitated", "disembowelled", "entrails", "pool of blood" | battles are fine, gore isn't |
| Real living people | named politicians, actors, sportspeople, business leaders | doesn't depict real, named living people |

**How the matching works:**
- **Two texts are checked:** the user's text and the final positive prompt. The negative prompt is not checked,
  because M3's negative contains "ugly" on purpose.
- **Matching:** whole words or phrases, case- and accent-insensitive.
- **Words that depend on context** ("cartoon", "seductive") block only within 6 words of a name from M3's
  `kb/names.json` (559 deities and epic figures). So "a cartoon of a mountain" is allowed, and "Ganesha as a
  cartoon" is not.
- **Refusals** are one calm, specific sentence, never a lecture.
- **Tests:** `tests/test_app_policy.py` has 21 allowed and 20 blocked examples, including borderline ones. They run
  in CI.

**Decision: Draupadi's disrobing (*vastraharan*) is allowed.**
- **What is allowed:** the episode as Varma painted it. Draupadi is fully clothed, and Krishna's endless sari
  protects her.
- **Why:** it is a central scene of the Mahabharata and part of Varma's own subject list. Refusing it would refuse
  the epic itself, not harmful content.
- **How:** the episode's name (`allowed_despite_terms` in `app/blocklist.json`) never triggers a block on its own.
  Any sexual or nudity term in the same text still does ("Draupadi's disrobing, naked …" is refused).
- **Related:** "Draupadi humiliated in the dice hall" is allowed for the same reason.

## 3. Labelling
- **On screen and in downloads:** every image the app shows or offers for download has a caption bar: *"AI-generated
  in the style of Raja Ravi Varma. Not an authentic work."*
- **In the file:** the PNG metadata holds the same text (`Description`) plus the app version and every setting:
  model, seed, LoRA scale, pose mode, control scale and post-processing (`parameters`). So any download can be
  reproduced and identified.
- **Evaluation images are never labelled.** They are made by `run_batch.py`, not the app, so the metrics and raters
  see the raw image.

## 4. Uploads
- **Purpose:** a photo is used only to extract a stick-figure skeleton (C's `extract_pose`, which writes nothing).
- **Deletion:** the app deletes the uploaded file in its handler as soon as the skeleton exists, including when the
  prompt is refused. Gradio's own upload cache is cleared on a timer (`delete_cache`).
- **The app tells the user:** "deleted after **N minutes**" next to the upload box. N = `<<pending: D's
  upload-cache test>>`, and the temp path is in `APP.md`. **Nothing says "not saved".**
- **Canny (edge) uploads are not offered for photos of people,** because edges carry a person's silhouette. Painting
  uploads also stay off if C's p4 finds that composition transfer redraws the painting (`<<pending: C's p4>>`).

## 5. Output NSFW filter
- **Measurement (B, `results/audit_real_paintings.json`, run 22 Nov 2026):** classifier
  `Falconsai/nsfw_image_detection`, threshold 0.5, on the 15 real held-out Varma paintings: **1 of 15 flagged**
  (`varma_suckling_child_a52.png`, score 0.994: a mother nursing a child, a classic false positive on classical
  painting). The other 14 scored below 0.01.
- **Rule, fixed in advance:**
  - more than 1 of 15 real paintings flagged → **don't enable** the filter, and rely on the input blocklist and the
    label;
  - otherwise → a soft blur-and-confirm filter, not a hard block.
- **Decision:** 1 is not more than 1, so by the rule the filter is **enabled as a soft blur-and-confirm** on app
  output only: a flagged image is shown blurred with a "Show image" button, never blocked, and never applied to
  evaluation images.
- **Caveats, stated plainly:**
  - the result sits exactly on the boundary, and the one flagged painting is a false positive of the kind the M2
    safety checker also produced, so expect the blur to fire on some innocent images (nursing, bathing, drapery);
  - the measurement on our own generated outputs is not in the audit file (`extra` is empty), so the flag rate on
    our images is not measured;
  - the filter is a backstop, not the main control: the input blocklist and the label do the real work.
- **Status:** `<<pending: implementation. The decision is recorded, but the output filter is not built into the app yet>>`.
- **Background:** the M2 safety checker stays off, because it false-blocked classical paintings.

## 6. Public link
- **Access (decision, 22 Nov):** the public link is launched with `--auth user:password`, and the password is shared
  only with the class and the markers. The link is shared only while demoing or testing, and the Colab session is
  closed afterwards (the link dies with it).
- **Why it matters:** the link goes through Gradio's relay and dies with the Colab session. Anyone who has it can use
  our GPU quota.

## 7. Known gaps
- **A blocklist can't catch everything:** paraphrases, misspellings, other languages and new names get through.
- **There is no human moderation,** and no output classifier unless §5 enables one.
- **The living-people list is short and illustrative,** not complete.
- **The images themselves are not checked for respectful depiction.** That rests on the input rules, the
  respectful presets (C) and the label.

## 8. Model card note (shown in the app's About text)
Research and coursework use only. Outputs are *in the style of*, not *by*, Raja Ravi Varma, and are never to be
passed off as authentic works. Not for demeaning or sexualised depictions of deities or anyone else.

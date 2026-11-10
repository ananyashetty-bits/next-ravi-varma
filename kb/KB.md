# KB.md: knowledge base and parser (Member C)

## 1. What this is
`kb/kb.json` (version kb0.2) turns a name in the user's text into something the model can paint, and `kb/parser.py` turns a sentence into a `SceneSpec`. The M2 model does not know character names (`NAMES_TEST.md`), so every character, place and mood gets a short visual description. The parser is deterministic (no LLM): testable, free, offline, and the fallback when the LLM fails.

## 2. Schema
`characters` (canonical, aliases, epic, `visual`, `coverage_n`, `coverage`, `source`, optional `visual_alt`, `note`), `settings` (aliases, `visual`, `coverage`, `source`), `moods` (label -> lighting phrase). `kb/names.json` is the flat list of every name and alias, built by `kb/build_names.py` as the KB's names plus D's `expander/names_seed.json`, so characters without a KB entry (e.g. Ravana) are still scrubbed from prompts.

## 3. How descriptions were grounded in the paintings
For each character with training images, I opened 2-3 of the paintings (solo ones where they exist) and wrote what Varma painted: garments, colours, jewellery, headgear, props. Only features that repeat across paintings went into the line; features that vary (pose, expression, a sari colour that changes) were left out. Each line is at most 14 words, has no names, and avoids abstract words. Characters that look different in some scenes get a `visual_alt` (Rama, Lakshmana, Krishna, Arjuna). Where two characters could look alike (Rama and Lakshmana, Radha and Yashoda), I looked for a visible difference in the paintings (for example Lakshmana: fair skin, smaller crown) and did not invent one when I could not find it (Shatrughna is a generic brothers' look, marked as such).

## 4. Coverage
Thresholds: **high** >= 5 images, **medium** 2-4, **low** 1, **none** 0. Coverage counts the training images (`train_v1`, 147 images) whose `scene_metadata` lists the character, after merging spelling variants in `train/audit_coverage.py`.

| Character | Epic | Images | Coverage | Source |
|---|---|---|---|---|
| Sita | Ramayana | 10 | high | paintings |
| Krishna | Mahabharata | 9 | high | paintings |
| Rama | Ramayana | 9 | high | paintings |
| Shakuntala | Mahabharata | 8 | high | paintings |
| Draupadi | Mahabharata | 6 | high | paintings |
| Lakshmana | Ramayana | 5 | high | paintings |
| Radha | Puranic | 5 | high | paintings |
| Hanuman | Ramayana | 4 | medium | paintings |
| Arjuna | Mahabharata | 2 | medium | paintings |
| Bharata | Ramayana | 2 | medium | paintings |
| Yashoda | Puranic | 2 | medium | paintings |
| Ahalya | Ramayana | 1 | low | paintings |
| Bhishma | Mahabharata | 1 | low | paintings |
| Duryodhana | Mahabharata | 1 | low | paintings |
| Satyavan | Mahabharata | 1 | low | paintings |
| Savitri | Mahabharata | 1 | low | paintings |
| Shatrughna | Ramayana | 1 | low | paintings |
| Subhadra | Mahabharata | 1 | low | paintings |
| Yama | Mahabharata | 1 | low | paintings |
| Karna | Mahabharata | 0 | none | text only (no painting) |
| Shabari | Ramayana | 0 | none | text only (no painting) |

| Setting | Coverage | Source |
|---|---|---|
| court | grounded | paintings |
| forest | grounded | paintings |
| palace | grounded | paintings |
| river | grounded | paintings |
| garden | grounded | paintings |
| lake | grounded | paintings |
| ocean | grounded | paintings |
| kurukshetra | none | text only |

The battlefield (`kurukshetra`) is **not grounded**: the training set has no battlefield painting (keyword scan found 0), so it is a generic description marked `coverage: none`. Karna and Shabari also have no training painting and are text-only.

## 5. Parser design
`parse_scene(text, scene_id="", kb_path=None)` returns the `SceneSpec` of Overview 6.3. The matching core (word-bounded alias lookup, longest alias first, matched text consumed so "Rama" never matches inside "Ramayana", epic by majority, settings, mood lexicon, unknown entities) is **D's stub shim, copied to `kb/parser_core.py`**. My additions in `kb/parser.py`: fuzzy spelling repair with rapidfuzz (cutoff 88; the first letter must match; words of 6 letters or fewer need 92; a word that is a known name in `names.json` is never "corrected", so Bhima is not turned into Bhishma), extra mood words, a stoplist for epic and place names, and reporting of a known-but-unmatched name in first position as unknown. Misspelled names resolve to the canonical name and the user's own spelling is kept as `mention`.

## 6. Results (H-C1)
Pre-registered thresholds (HYPOTHESES.md C.1): character precision >= 0.90, recall >= 0.90, epic accuracy >= 0.85.
On the 24 dev scenes: **precision 1.00, recall 1.00, epic accuracy 1.00** (`kb/eval_parser.py`, `kb/parser_metrics.json`). 14 unit tests in `kb/test_parser.py` cover misspellings, Rama vs Ramayana, unknown characters, Bhima vs Bhishma, empty input, the contract fields, and that no KB description contains a name.
**Caveat:** the KB was built mainly from the characters in the team's dev scenes, so these scores show the parser works when the KB has the character, not that it generalises. The held-out set is the real test. Dev scenes s17 and s23 were originally written with Radha and Ahalya (both in the KB) but were changed to characters present in D's interim KB so D's test passes; this means the dev set does not test the rare-character case.

## 7. Grounding experiment (c6)
**Not run.** `kb_llm.json` (descriptions written by an LLM without looking at the paintings) was not built because of time, so H-C2 is untested and no claim is made about whether grounding matters.

## 8. Names LoRA
The coverage audit passed the pre-registered gate (67% of the 147 training images have characters filled, and 10 characters appear in 3 or more images; bar: 60% and 6). The LoRA was **not trained**: the Day 3 cutoff for launching training passed before the KB existed (deviation line in HYPOTHESES.md). There is no `rrv_names` model and no 2x2 result.

## 9. Known gaps
- Missing from the KB: Ravana, Bhima, Yudhishthira, Nakula, Sahadeva and most one-image characters (Valli, Devasena, Urvashi and others). They are caught by `names.json` for scrubbing but not described.
- Radha and Ahalya are in the KB but not in D's interim KB, so no dev scene uses them (issue open for D).
- Karna, Shabari and the battlefield are not grounded in paintings.
- Bharata's line is his Nandigrama (ascetic regent) look, the only one painted in the data. Shatrughna could not be told apart from the other brothers.
- Several characters have one or two images, so their lines come from one painting.
- Similar looks remain: Radha and Yashoda (bun with flowers, nose ring), Draupadi and Savitri (loose hair).
- Mood phrases are untested except "dusk" (works) and "oil lamp" / "dramatic cloudy sky" (partly), from M2's VISUAL_VOCAB.md.
- The parser has not been tested on held-out scenes.

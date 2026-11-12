# KB.md: knowledge base and parser (Member C)

## 1. What this is
`kb/kb.json` (version kb0.4; see section 10) turns a name in the user's text into something the model can paint, and `kb/parser.py` turns a sentence into a `SceneSpec`. The M2 model does not know character names (`NAMES_TEST.md`), so every character, place and mood gets a short visual description. The parser is deterministic (no LLM): testable, free, offline, and the fallback when the LLM fails.

## 2. Schema
`characters` (canonical, aliases, epic, `visual`, `coverage_n`, `coverage`, `source`, optional `visual_alt`, `note`), `settings` (aliases, `visual`, `coverage`, `source`), `moods` (label -> lighting phrase). `kb/names.json` is the flat list of every name and alias, built by `kb/build_names.py` as the KB's names plus D's `expander/names_seed.json`, so characters without a KB entry (e.g. Ravana) are still scrubbed from prompts.

## 3. How descriptions were grounded in the paintings
For each character with training images, I opened 2-3 of the paintings (solo ones where they exist) and wrote what Varma painted: garments, colours, jewellery, headgear, props. Only features that repeat across paintings went into the line; features that vary (pose, expression, a sari colour that changes) were left out. Each line is at most 14 words, has no names, and avoids abstract words. Characters that look different in some scenes get a `visual_alt` (Rama, Lakshmana, Krishna, Arjuna, Ravana). Where two characters could look alike (Rama and Lakshmana, Radha and Yashoda), I looked for a visible difference in the paintings (for example Lakshmana: fair skin, smaller crown) and did not invent one when I could not find it (Shatrughna is a generic brothers' look, marked as such).

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
| Vishwamitra | Ramayana | 4 | medium | paintings |
| Ganesha | Puranic | 3 | medium | paintings |
| Vishnu | Puranic | 3 | medium | paintings |
| Arjuna | Mahabharata | 2 | medium | paintings |
| Bharata | Ramayana | 2 | medium | paintings |
| Damayanti | Mahabharata | 2 | medium | paintings |
| Lakshmi | Puranic | 2 | medium | paintings |
| Menaka | Mahabharata | 2 | medium | paintings |
| Mohini | Puranic | 2 | medium | paintings |
| Parvati | Puranic | 2 | medium | paintings |
| Ravana | Ramayana | 2 | medium | paintings |
| Shiva | Puranic | 2 | medium | paintings |
| Yashoda | Puranic | 2 | medium | paintings |
| Ahalya | Ramayana | 1 | low | paintings |
| Bhima | Mahabharata | 1 | low | paintings |
| Bhishma | Mahabharata | 1 | low | paintings |
| Duryodhana | Mahabharata | 1 | low | paintings |
| Nakula | Mahabharata | 1 | low | paintings |
| Sahadeva | Mahabharata | 1 | low | paintings |
| Satyavan | Mahabharata | 1 | low | paintings |
| Savitri | Mahabharata | 1 | low | paintings |
| Shatrughna | Ramayana | 1 | low | paintings |
| Subhadra | Mahabharata | 1 | low | paintings |
| Yama | Mahabharata | 1 | low | paintings |
| Yudhishthira | Mahabharata | 1 | low | paintings |
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

## 7. Grounding experiment (c6, H-C2)
`kb/kb_llm.json` has the same characters, aliases and settings as `kb.json`, but every `visual` was written by the local LLM (Qwen3-4B, temperature 0) without seeing the paintings, using the prompt from the C guide; moods are copied unchanged. Both KBs were expanded with the same LLM and prompt on the 24 dev scenes (`c3_llm_kb_ckb`, `c6_llm_kbllm`; 24/24 valid, 0 names after repair), rendered with `rrv_v1` at scale 1.0, seeds 1-2, and scored with B's `eval/score.py`. Comparison: B's `paired()` from `eval/stats.py`, scene as unit, bootstrap 95% CI (`kb/compare_c6.py`, `results/c6_comparison.json`). Difference = c3 - c6.

| Metric | c3 (kb.json) | c6 (kb_llm.json) | Diff c3-c6 [95% CI] | W/T/L | Diff, 14 strong-coverage scenes [CI] |
|---|---|---|---|---|---|
| Checklist judge | 0.522 | 0.509 | +0.013 [-0.045, +0.068] | 13/1/10 | +0.002 [-0.064, +0.068] |
| clip_raw | 0.249 | 0.241 | +0.008 [-0.004, +0.020] | 6/14/4 | -0.005 [-0.017, +0.007] |
| style_margin | 0.110 | 0.111 | -0.001 [-0.012, +0.011] | 4/16/4 | +0.004 [-0.008, +0.017] |
| max_train_dino | 0.714 | 0.700 | +0.015 [-0.009, +0.041] | 4/16/4 | +0.025 [-0.013, +0.067] |

**Result: no clear difference.** Every CI includes 0, including on the scenes where all characters have medium or high coverage, where the gain was predicted. H-C2 is not supported at this sample size; this does not show that grounding is irrelevant.
**Limits:** the checklist judge is a 2B-model proxy and no human ratings exist for c6; 24 scenes and 2 seeds give wide CIs; the LLM-written descriptions are also shorter and worded differently, so c6 differs from c3 in more than grounding; the expander rewrites the KB text before generation. Observations only: the c6 descriptions are generic and overlap (several women are 'white saree, golden necklet') and miss defining features (Hanuman has no monkey face), and colours in the images follow the KB wording (red sari vs white saree).
Images were generated with `kb/gen_c6.py` (M2 generator) because `pipeline/generate_api.py` defines no `generate()`.

**Version note:** the c3 and c6 expansions, images and scores were produced with kb0.2 (21 characters); `kb/kb_llm.json` (kb0.2-llm) was built for those same 21 characters and was not extended. The KB has since grown to 35 characters (kb0.4); the results stay tied to the kb0.2 commit.

## 8. Names LoRA
The coverage audit passed the pre-registered gate (67% of the 147 training images have characters filled, and 10 characters appear in 3 or more images; bar: 60% and 6). The LoRA was run late, after the Day-3 cutoff: `C_names` is the M2 winner's recipe (rank 16, LR 2e-4, seed 42, 800 steps) trained on `train_v3`, which is `train_v1` with epic character names added to 70 of 147 captions (74 distinct names; historical portraits and non-epic figures excluded). **Status at the time of writing:** training is finished and the weights are saved; the guardrail run, the names test and the 2x2 runs have **not** been run, so no claim is made about whether it learned names. Everything about it is in `train/NAMES_MODEL.md`.

## 9. Known gaps
- Epic characters with training images but no KB entry: 41, mostly with a single image (e.g. Bhoomi Devi (2), Adisesha (1), Ambika (1), Balarama (1), Bali (1), Dattatreya (1), Devaki (1), Devasena (1), Durga (1), Dushyanta (1), Hamsa (1), Hiranyakashipu (1), Indira (1), Janaka (1)). They are caught by `names.json` for scrubbing but not described.
- Riddhi, Siddhi, Sridevi and Bhudevi (one image each) have no entry: the two women in each painting could not be told apart.
- Nakula and Sahadeva are visually identical in the only painting and share one generic twins' look (as Shatrughna's is a generic brothers' look).
- Damayanti's line is her forest-exile look only; Yudhishthira, Bhima, Nakula and Sahadeva come from one crowded court painting.
- Radha and Ahalya are in the KB but not in D's interim KB, so no dev scene uses them (issue open for D).
- Karna, Shabari and the battlefield are not grounded in paintings.
- Bharata's line is his Nandigrama (ascetic regent) look, the only one painted in the data. Shatrughna could not be told apart from the other brothers.
- Several characters have one or two images, so their lines come from one painting.
- Similar looks remain: Radha and Yashoda (bun with flowers, nose ring), Draupadi and Savitri (loose hair).
- Mood phrases are untested except "dusk" (works) and "oil lamp" / "dramatic cloudy sky" (partly), from M2's VISUAL_VOCAB.md.
- The parser has not been tested on held-out scenes.

## 10. KB versions
| Version | Characters | Used for |
|---|---|---|
| kb0.1 | stub (about 6) | stub parser, early integration |
| kb0.2 | 21 | parser v1, dev-scene parser metrics, the c3 and c6 expansions, images and scores |
| kb0.3 | 31 | 10 more characters (Vishwamitra, Ganesha, Vishnu, Ravana, Damayanti, Lakshmi, Menaka, Mohini, Shiva, Parvati) |
| kb0.4 | 35 | Yudhishthira, Bhima, Nakula, Sahadeva; current |
| kb0.2-llm | 21 | `kb/kb_llm.json` (c6), descriptions written by the LLM without images |
Adding characters did not change any dev-scene result (parser metrics stayed at precision 1.00, recall 1.00, epic 1.00 on the 24 dev scenes).

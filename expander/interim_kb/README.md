# Interim KB `d-kb0` (owner D)

**Why it exists:** c2 (`c2_template`) and c3 (`c3_llm_kb`) need character descriptions and a parser that returns name-free action text (Overview §6.3). C's `kb/kb.json`, `kb/parser.py` and `kb/names.json` were not on `main` when D's dev runs were due (#21), so D wrote this small KB for the **18 characters of the 16 dev scenes** on `main`, plus one group (the Pandavas) and five settings. It is **not C's KB**. Every record made with it carries `kb_version: "d-kb0"`. Results made with it are reported as a deviation of section D in `experiments/HYPOTHESES.md`.

**How the descriptions were written:**
- **Wording:** `visual` reuses the **train_v1 caption** wording of the character's training paintings, i.e. what the LoRA was trained on. The paintings come from M1 `scene_metadata.characters`.
- **Sources:** `source.ref` lists those training image stems.
- **Status:** 16 entries are `verified`, backed by at least one training painting. **Karna** and **Shabari** have no training painting: they are `unverified` (`coverage: none`), with a named text reference, and need C's review.
- **Roles:** `role` is a short name-free noun phrase that the stand-in parser puts in place of the name in the action text ("Sita's ring" → "the young princess's ring").
- **Notes:** the `notes` fields record what is not from a painting (e.g. the white beard of the old warrior).
- **Names:** no `visual` or `role` contains a name (`tests/test_expander_interim_kb.py`).

**Additions:** Radha and Ahalya (2026-11-14), entries supplied by C from `kb/kb.json` (#52).

**Use:** pass it explicitly. Nothing picks it up by default. Since C's `kb/parser.py` exists, also pass `--parser shim` to stay on the parser used for the scored d-kb0 runs.
```
python expander/batch_expand.py --scenes eval/test_scenes_dev.json --mode template --kb expander/interim_kb/kb_d0.json --parser shim --out outputs/expansions/c2_template_dev.json
python expander/batch_expand.py --scenes eval/test_scenes_dev.json --mode llm_kb   --kb expander/interim_kb/kb_d0.json --parser shim --out outputs/expansions/c3_llm_kb_dev.json
```

**When C's KB lands:** re-run c2 and c3 without `--kb` (or with C's file) to new output files. The cache keys include `kb_version`, so the `d-kb0` outputs stay separate and comparable. Delete this folder once the `d-kb0` results are no longer cited.

**Deviation line for `experiments/HYPOTHESES.md`, section D (add with the first d-kb0 results):**
> `2026-11-05: c2_template and c3_llm_kb dev runs use D's interim KB (kb_version d-kb0, expander/interim_kb/) and the role-replacing stand-in parser, because C's KB and parser were not available; the dev set has 16 scenes (s09-s16, s25-s32). Affects H-D1, H-D2 (and H-D3/H-D5 if run). Re-run with C's KB when it lands; both sets are reported.`

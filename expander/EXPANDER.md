# Expander (owner D)

`expand(text, scene=None, mode="llm_kb") -> ExpandedPrompt` turns a user's epic-scene sentence into a prompt the M2 model (`rrv_v1`) can render: name-free visual phrases, the trigger first, the style tail last, under the token limit. It runs a **local** instruct LLM (no paid API) with a deterministic template as the fallback.

> **Status (2026-11-14):** the design and all modes are built and tested (`tests/test_expander_*.py`). c2, c3 and c4 are expanded, generated and scored on the first 16 dev scenes (§9); c2 is expanded on all 24; c3/c4 on s17-s24 and human ratings for c2/c3 are pending. **c2 and c3 run on D's interim KB `d-kb0`** (`expander/interim_kb/`), because C's KB and parser were not available (#21); see the Section D deviations. Everything still marked `TODO(results)` waits for numbers from `outputs/expansions/`, `results/` or B's tables; no number goes in by hand. Contracts referred to: Overview v2 §6 (`config/m3.json`, SceneSpec, ExpandedPrompt, signatures, CLI).

## 1. Design principle

```
<trigger>, <scene phrase 1>, <scene phrase 2>, …, <style_tail>
```

The LLM writes **only the scene phrases**, ordered from most to least important. The **code** adds the trigger (always first) and the style tail (always last), scrubs names, and fits the phrases under the budget. The LLM therefore cannot drop the trigger, leak a name into the final prompt, or exceed the budget. Every record carries `prompt_version`, `kb_version` and `llm_model`, so any expansion can be traced and reproduced from the cache.

## 2. Modes and conditions

| Mode (`expand`) | Condition ID | What the model sees | `source` | `prompt_version` | Run by |
|---|---|---|---|---|---|
| `raw` | `c1_lora_raw` | `trigger, <raw text>, style_tail`: no fitting, no scrub (parity with A's c1 string; names recorded only) | `template` (+ `note: "raw"`) | `raw` | A |
| `template` | `c2_template` | SceneSpec → each figure's first `visual` phrase, the `action` clauses, `setting.visual`, the mood phrase, then the remaining visual phrases round-robin (§2.1); no LLM | `template` | `tpl2` | D |
| `llm_kb` | `c3_llm_kb` | LLM phrases, user message **with** the `Reference descriptions` block | `llm` / `fallback` | `v1` | D |
| `llm_nokb` | `c4_llm_nokb` | LLM phrases, user message **without** that block (the only c3/c4 difference) | `llm` / `fallback` | `v1` | D |
| `llm_kb_layout` | `c5_llm_kb_layout` (optional) | c3 plus a leading layout phrase from the reply's `layout` JSON | `llm` / `fallback` | `v1_layout` | D |
| `llm_kb` + `--kb kb/kb_llm.json` | `c6_llm_kbllm` | c3 with LLM-written KB descriptions | `llm` / `fallback` | `v1` | C |
| `llm_kb` + `--llm large` (= `llm.model_large`; or `--llm_model <id>`) | `c3_7b` (optional) | c3 with the larger LLM | `llm` / `fallback` | `v1` | D |
| `llm_kb` + `--scene_negatives` | `c3_neg` (optional) | c3 positive (same cached calls); negative = config negative + 3-6 LLM scene negatives from one extra call | `llm` / `fallback` | `v1` (negatives call: `neg_prompt_version` `v1_neg`) | D |
| `llm_kb` + `--prompt_version v2` | `c3_v2` (optional; **proposed ID**) | c3 with prompt v2 (SHAP value-per-token ordering) | `llm` / `fallback` | `v2` | D |

`batch_expand.py` sets `condition` from `--condition`, otherwise from the `--out` file name (`<condition>_<split>.json`), otherwise from the mode. Hypotheses for these comparisons: `experiments/HYPOTHESES.md`, section D.

### 2.1 Template (c2 and the fallback), version `tpl2`

Deterministic phrase priority, most important first. Token fitting (§5) drops phrases from the end of this list:
1. each character's **first** visual phrase (its KB `visual`, split on commas), in SceneSpec order;
2. the `action`, split into clauses on `, ; : . ! ?` (each clause is one phrase, so a long action loses only the clauses that do not fit, and names are scrubbed clause by clause);
3. the `setting.visual` phrases;
4. the mood lighting phrase (`mood_visual` > KB `moods` > a built-in fallback map);
5. the remaining visual phrases, **round-robin** across characters (the 2nd phrase of each, then the 3rd, ...).

`tpl1` (before 2026-10-30) put all of character 1's phrases before character 2's and before the action. With the real CLIP counter, multi-figure scenes then overflowed the 70-token budget and **lost the action** (fixtures f02/f03), which made c2 a strawman. Records carry `template_version: "tpl2"` (also on `source: "fallback"` records) and template-mode records carry `prompt_version: "tpl2"`, so older outputs remain distinguishable.

## 3. System prompt v1 (verbatim; `expander/prompts/system_v1.txt`)

```
You write scene descriptions for a text-to-image model trained on 19th-century Indian oil paintings.
Return ONLY valid JSON: {"phrases": ["...", "..."]}
Rules:
- 6 to 12 phrases, each 2 to 6 words, ordered from MOST to LEAST important to the scene.
- NEVER use proper names of people, gods or places. The model does not know names. Describe by
  clothing, ornaments, pose, expression and setting instead.
- If reference descriptions are given, use their wording for the matching figures.
- Cover, in roughly this order: the main figures (attire), what they are doing (pose, gesture),
  the setting, the lighting and mood.
- Do NOT mention style, painting, canvas or artist; that is added separately.
- Use concrete visual words (e.g. "kneeling", "gilded armour", "pillared hall").
```

**User message:** the raw text, plus, in `llm_kb` mode only, `Reference descriptions:` followed by one `<mention> -> <visual>` line per character from the SceneSpec. Variants (`v1_layout`, `v1_neg`, `v2`) are separate files with their own `prompt_version`; v1 is never edited (a test pins its sha256 and checks `prompts/system_v1.txt` against the built-in text). **`v1ex`** = v1 + one worked example from dev scene s25 (`prompts/system_v1_example.txt`; `v1ex_layout` uses `system_v1_layout_example.txt`), for small models that answer in prose: `--prompt_version v1ex`. On CPU, Qwen2.5-0.5B-Instruct answered v1 with an invalid `{"phrases": "<the input sentence>"}` and v1ex with 9 valid phrases (a plumbing check only, not a result). With v1ex, s25's own expansion is not independent of the prompt. `TODO(results)`: the variant texts and the change log of v2.

## 4. Fallback chain (the pipeline never crashes because of the LLM)

1. LLM call (cached) → parse JSON `{"phrases": [...]}` (tolerant: strips code fences, takes the first `{…}`).
2. Parse failure → **one** retry with "Return ONLY valid JSON" appended (cached; `json_retry: true`).
3. Second failure **or any exception** (load error, OOM, timeout) → **template mode** on the same SceneSpec, with `source: "fallback"` and `error` set. In `batch_expand.py`, a model that **fails to load** stops the batch instead (records written so far are kept), so a file of fallbacks is never labelled c3/c4; re-running resumes.
4. Name leak → **one** re-ask: "remove proper names; describe by clothing, pose and setting" (cached; `name_repair: true`). Names still present → drop the offending phrases (`phrases_scrubbed`).
5. `assemble` (§5), then a last re-scan of the final `positive`, excluding trigger and tail. Any hit drops that phrase and the prompt is re-assembled.

## 5. Token fitting

- **Budget:** `max_prompt_tokens` = **70**, counted **excluding** the start/end tokens with the team's one counter `common/tokens.py` `n_tokens(text)` (CLIP BPE of the base model's tokenizer). It stays 70 until A's `pipeline/TOKEN_BUDGET.md` says otherwise. The M2 helper `generate.py n_tokens` counts **including** start/end tokens and is never used here.
- **`assemble(trigger, phrases, style_tail, max_tokens, *, counter=None)`:** adds phrases greedily in priority order and skips any phrase that does not fit, so a shorter later phrase can still be kept. Trigger and tail are always kept, and an empty tail adds no trailing comma. `phrases_dropped` lists **budget drops only**; name removals go to `phrases_scrubbed`.
- **Fixed cost (team config since #18, measured with `common/tokens.py` on 2026-11-03):** trigger `rrvarma style` 4 tokens; `"<trigger>, <tail>"` with the team tail `oil painting, canvas texture, classical indian art` 14 tokens, so **56 tokens** remain for scene phrases. (The tail first proposed here, M2's v2 suffix, would have cost 25.)
- **Measured, c4 dev (16 scenes):** `token_count` mean 62.4, min 40, max 70; on average 8.2 phrases kept and 0.75 dropped for the budget per record. `TODO(results)`: the same for c2 and c3.

## 6. Name scrub

- **Why:** the LoRA does not know names (M2 `NAMES_TEST.md`), and A's CI fails if a KB alias appears in any `positive` of c2-c6.
- **Name list:** C's `kb/names.json` when present. Until then, the stub KB names together with `expander/names_seed.json`: 190 canonical names and 350 aliases from M1 `scene_metadata` plus the major epic figures and places.
- **Matcher** (`expander/names.py`, shared with A's CI so the two cannot disagree): text normalised with NFKD to ASCII; whole words (`\b…\b`, multi-word aliases joined by `\s+`, longest first); **case-insensitive by default**.
- **Case policy for English-word aliases:** aliases that are also ordinary words are matched **only when capitalised**: `Ram`, `Lava`, `Kush`/`Kusha`, `Bali`, `Kali`, `Tara`, `Kama`, `Hamsa`, `Uma`, `Hari`, `Agni`, `Surya`, `Usha`, `Matsya`, `Varaha`, … (full list in `names_seed_meta.json → risk.case_sensitive_only`). This keeps "a ram", "kusha grass" and "flowing lava" in the prompt. The LLM capitalises proper names. Honorifics and class nouns (`Devi`, `Sri`, `Rakshasa`, `Apsara`, `Gopi`) and `ashoka grove` are not names. In prep, the list had 0 false positives on 178 name-free M2 texts.
- **Record fields:** `names_leaked` = names found **before** repair (B reports how often the LLM leaks). `names_after_repair` = names left in the final phrases, which **must be `[]`**. `phrases_scrubbed` = phrases removed for names.
- **Known cost:** rivers and mounts that are also names (`Ganga`, `Yamuna`, `Sarayu`, `Nandi`, `Garuda`) are scrubbed, which can remove a setting phrase. Measured in c4 dev: **0 phrases scrubbed** in 16 records, because the one re-ask removed every leaked name. `TODO(results)`: c2/c3.
- **Measured leak rate before repair, c4 dev:** 12 of 16 records (75%). Most frequent: Arjuna 4, Krishna 4, Sita 3, Hanuman 3, Rama 2, Lakshmana 2. **After repair: 0.**
- **Template mode:** there is no LLM to re-ask, so the matched name is deleted from the phrase, dangling connectors are tidied, and the phrase is dropped if fewer than 2 content words remain.

## 7. Cache

- **File:** `expander/cache.json`, **committed for dev**, so anyone can reproduce every expansion without loading an LLM (Overview §13.1).
- **Key:** `sha256("prompt_version|model|kb_version|mode|user message")`, one entry per LLM call, retries and repair calls included. The key separates c3 from c4 (`mode`), c3 from c6 (`kb_version`) and v1 from v2 (`prompt_version`). Entry value: `{reply, created, model, prompt_version}`. Entries are only added, never overwritten. Temperature 0.
- **Held-out:** B runs with `--cache <private path>` outside the repo, so held-out text never enters the shared cache.

## 8. Configuration and open team decisions

D reads `trigger`, `style_tail`, `negative`, `max_prompt_tokens` and `llm.*` from `config/m3.json`; nothing is hardcoded. Keys missing from the team file are filled from `expander/_shims/m3.default.json`, and team values always win.

**As merged (config PR #18, 2026-11-01):**

| Key | Team value | Note |
|---|---|---|
| `trigger` | `rrvarma style` | M2 handoff |
| `style_tail` | `oil painting, canvas texture, classical indian art` | chosen by the team (A); the winner's train_v1 captions have no fixed tail, so any tail is an addition the model was not trained with. A's token-budget and SHAP runs measure it |
| `negative` | `ugly, blurry, bad anatomy` | differs from M2's evaluation negative (`lowres, blurry, deformed, extra limbs, bad anatomy, text, watermark`); flagged in #18 |
| `max_prompt_tokens` | 70 (excluding start/end) | until `TOKEN_BUDGET.md` |
| `llm.model` / `revision` | `Qwen/Qwen3-4B-Instruct-2507` @ `cdbee75f17c01a7cc42f958dc650907174af0554` | 4-bit NF4, fp16 compute; verified on a T4 (§12) |
| `llm.model_large` / `model_large_revision` | `Qwen/Qwen2.5-7B-Instruct` @ `a09a35458c702b33eeacc393d103063234e8bc28` | `c3_7b` only |
| `llm.temperature` / `prompt_version` | `0.0` / `v1` | final version posted at S3 |
| `models.rrv_v1.scale` | 1.0 | **open:** M2's handoff recommends 0.8; A to confirm (#16). c4 images so far were made at 1.0 |

**History, the Day-1 proposal (superseded):**

| Key | Proposed value | Status |
|---|---|---|
| `trigger` | `rrvarma style` | M2 handoff; agreed |
| `style_tail` | `oil painting, classical Indian academic realism, soft skin rendering, richly detailed drapery and jewelry` (M2's frozen v2 suffix, `DATA_FROZEN.json`) | **OPEN.** The winner (`D_lr2e-4` step 800) was trained on **train_v1**, whose captions have **no fixed tail**, so the overview's definition ("the fixed tail used at the end of training captions") is empty for it. This suffix is the tail behind all M2 test-set numbers for the winner. Alternatives: `""`. A's token-budget run and SHAP `tail` player measure it; change by config PR only. It also changes c1 |
| `negative` | `lowres, blurry, deformed, extra limbs, bad anatomy, text, watermark` | **OPEN, recommended.** `generate.py`'s `NEG`, used for every M2 measurement. The handoff's "ugly, blurry, poorly drawn, bad anatomy" is only an example ("such as") |
| `max_prompt_tokens` | 70 (excluding start/end) | until `TOKEN_BUDGET.md` |
| `llm.model` | `Qwen/Qwen3-4B-Instruct-2507` (primary, Apache-2.0), 4-bit NF4, fp16 compute on T4 | **OPEN** until D's Day-1 T4 load and speed test; pin the revision |
| `llm.model_large` | `Qwen/Qwen2.5-7B-Instruct` (for `c3_7b`) | optional experiment |
| `llm.temperature` / `prompt_version` | `0.0` / `v1` | frozen version posted at S3 |

Settled since Day 1: scene IDs `s25`-`s32` (B, #22); B's scene check uses `expander/names_seed.json` for its no-names rule. Still open: A's CI importing `expander.names` (#16); the ID `c3_v2` for prompt v2.

## 9. Ladder results (c2 → c4 → c3)

**Judge results, dev (supporting evidence; 16 scenes × seeds 1-2; `rrv_v1` at scale 1.0).** Means from `results/` on `main`: `c0_base_raw@base_dev.json`, `c1_lora_raw@rrv_v1_dev.json`, `c4_llm_nokb@rrv_v1_dev.json` (B, #62) and `c2_template@rrv_v1_dev.json`, `c3_llm_kb@rrv_v1_dev.json` (D, #38). The 8 scenes C added (s17-s24) are not scored for c2/c3/c4 yet:

| Condition | checklist (judge), 95% CI | `style_margin` | `clip_raw` | `max_train_dino` p95 |
|---|---|---|---|---|
| `c0_base_raw` | 0.569 | 0.025 | 0.268 | 0.800 |
| `c1_lora_raw` | 0.576 | 0.093 | 0.264 | 0.805 |
| `c2_template` (d-kb0) | 0.570 [0.481, 0.659] | 0.078 | 0.268 | 0.767 |
| `c3_llm_kb` (d-kb0) | 0.520 [0.426, 0.612] | 0.104 | 0.249 | 0.796 |
| `c4_llm_nokb` | 0.535 [0.453, 0.619] | 0.099 | 0.255 | 0.797 |

Paired over scenes (`eval/stats.paired`, b − a, 95% bootstrap CI, tie band 0.0295 for win/tie/loss):

| a → b | checklist (judge) | `style_margin` |
|---|---|---|
| c0 → c1 | +0.007 [−0.060, +0.064], W/T/L 8/4/4 | **+0.069 [+0.046, +0.095]**, 13/3/0 |
| c2 → c3 (H-D1) | −0.050 [−0.116, +0.011], 5/3/8 | **+0.025 [+0.014, +0.039]**, 4/12/0 |
| c4 → c3 (H-D2, all scenes) | −0.015 [−0.079, +0.054], 6/3/7 | +0.005 [−0.014, +0.022], 5/9/2 |
| c4 → c3 (H-D2, tier 4: s15, s31) | −0.119, n = 2 (not resolvable) | — |
| c1 → c3 | **−0.055 [−0.113, −0.003]**, 4/4/8 | +0.010 [−0.003, +0.026], 2/13/1 |
| c1 → c2 | −0.006 [−0.045, +0.033], 5/5/6 | −0.015 [−0.033, +0.004], 1/10/5 |
| c1 → c4 | −0.041 [−0.095, +0.018], 4/2/10 | +0.005 [−0.013, +0.024], 4/10/2 |

All five conditions pass the memorisation guardrail (`max_train_dino` p95 ≤ M2's `mem_flag_dino_p95` 0.85).

**Status of the hypotheses:**
- **H-D1, H-D2: not decided.** Their primary measure is human faithfulness, and c2/c3 are not rated yet (requested in #17).
- **On the judge:** no clear difference for H-D1 or H-D2. The H-D2 tier-4 split has 2 scenes.
- **First reading:** on the judge, expansion did **not** improve faithfulness over raw text with the LoRA (c1 → c3 is slightly negative), while c3 is the most Varma-like condition on `style_margin`.

`TODO(results)`: human faithfulness, style and quality per condition from B's ratings; H-D1/H-D2 decided by the D.0 rule; per-tier and seen/unseen splits; held-out rows after B's run.

Process table per file from `check_expansions.py --strict` (real CLIP counter):

| file | n | valid | trigger-first | within budget | leaks before | leaks after | fallback | latency mean / median (s) | tokens mean / max | dropped / rec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `c4_llm_nokb_dev.json` (16 scenes, 2026-11-02) | 16 | 100.0% | 100.0% | 100.0% | 75.0% | 0 | 0.0% | 21.03 / 13.57 | 62.4 / 70 | 0.75 |
| `c2_template_dev.json` (24 scenes, interim KB `d-kb0`; s17-s24 added 2026-11-14) | 24 | 100.0% | 100.0% | 100.0% | 0.0% | 0 | 0.0% | 0.003 / 0.003 | 64.3 / 70 | 4.46 |
| `c3_llm_kb_dev.json` (16 scenes, interim KB `d-kb0`, 2026-11-04) | 16 | 100.0% | 100.0% | 100.0% | 12.5% | 0 | 0.0% | 16.14 / 9.85 | 64.2 / 70 | 1.81 |

The LLM rows' mean latency includes the model load in the first scene's `latency_s` (`model_load_s`: c4 124 s, c3 103 s, fresh runtimes). Without it, per scene: c4 mean 13.3 s, median 13.3 s; c3 mean 9.7 s, median 9.4 s. `json_retry` was false in every record. The second calls (c4: 12, c3: 2) were all name re-asks; no phrase had to be scrubbed in any file. With the KB block, the LLM used names in 2 of 16 records (c3), against 12 of 16 without it (c4).

## 10. Optional experiments

| Condition | Built | Run | Note |
|---|---|---|---|
| `c5_llm_kb_layout` (H-D3) | yes (`--mode llm_kb_layout`) | not yet | c3 is done (#38); runnable on Colab |
| `c3_7b` (H-D4) | yes (`--llm large`) | not yet | c3 is done (#38); runnable on Colab |
| `c3_neg` (H-D5) | yes (`--scene_negatives`, prompt `v1_neg`) | not yet | c3 is done (#38); runnable on Colab |
| `c3_v2` (H-D6) | no | not yet | needs A's SHAP table and B's failure notes |

`TODO(results)`: each with the result, or "not run" and the reason (Overview §14 cut order: negatives → LLM size → c5).

## 11. Failure modes with examples

Seen so far at the **expansion** stage (from records, not from scored images):
- **The LLM reaches for names** (c4: 75% of records before repair). One re-ask fixed all of them; nothing was scrubbed.
- **The stand-in parser breaks the action text** (c2 preview on the 16 dev scenes, with the stand-in parser and KB; not committed, not scored). It cuts known names out of the sentence and leaves fragments: s12 "kneeling before and offering him 's ring"; s29 "lying on a bed of arrows on the field of while the and gather around him". The fix is C's parser returning the action already name-free (#21); the scrub can delete words but cannot rewrite them.
- **Characters missing from the KB get no description or the wrong one** (same preview): s10 has no description for its warrior; s29's old warrior picked up another character's description. KB coverage for the dev characters is requested in #21.
- **Reproducibility note (2026-11-14):** a later stand-in parser fix (an article after a comma: "…, the young prince") changes one existing dev prompt if c2 is re-run from scratch: s28's "young prince" becomes "the young prince". The committed s28 record, and the images scored from it, are kept unchanged.
- **Both are fixed for the dev runs** by the interim KB `d-kb0` and the role-replacing stand-in parser (#37): the c2 file has 0 names before repair, and every action reads as a sentence ("the old warrior lying on a bed of arrows … while the five crowned princes and the blue-skinned figure gather around him").
- **Template c2 loses phrases on crowded or long scenes:** 5.7 phrases dropped per record on average. For example, s28 keeps the coronation but drops the brothers standing and the devotee kneeling; s16 and s32 (tier 5, 100+ words) keep only their first clauses. This is the template working as designed under the 70-token budget.

`TODO(results)`: from B's Day-4 failure notes (`needs:D`) and `phrases_dropped`, for 3-5 bad images: the scene, the expansion, the image code and the diagnosed cause (figure phrase dropped by budget, KB description too vague, tail pushed out, name scrub removed a setting phrase, layout ignored). Include failures, not only successes.

## 12. Measured latency

Colab T4, `Qwen/Qwen3-4B-Instruct-2507` @ `cdbee75f`, 4-bit NF4 (requirements as pinned in `requirements_llm.txt`, torch 2.11.0+cu130):

| Measure | Value | Source |
|---|---|---|
| Load, first run on a fresh runtime (includes the 8 GB download) | 114-194 s | smoke test 2026-11-01; c4 run 2026-11-02 |
| Load, weights already cached | 23-33 s | Day-2 preview runs |
| Seconds per LLM call (warm) | 7.1-8.0 s | smoke test; Day-2 preview; c4 run (28 calls) |
| Seconds per scene, c4 (one call, or two with a name re-ask) | mean 13.3 s, median 13.3 s (`latency_s` − `model_load_s`) | `c4_llm_nokb_dev.json` |
| Peak VRAM | 2.77-2.81 GiB; `close()` frees it to 0.01 GiB | smoke test; batch logs |
| Seconds per scene, c3 (longer user message with the KB block) | mean 9.7 s, median 9.4 s (`latency_s` − `model_load_s`) | `c3_llm_kb_dev.json` |
| Template mode (c2) | 0.004 s per scene, CPU | `c2_template_dev.json` |

`TODO(results)`: `c3_7b`. Expansion runs alone in the `llm` session: expand everything, `close()`, then generate in a new `gen` session.

## 13. Commands

Step-by-step Colab cells, smoke test, hand-off to generation and troubleshooting: `expander/RUNBOOK_COLAB.md`. Library pins: `expander/requirements_llm.txt`.

**From Python (Tier 1, `text_to_image`):** `from expander.expand import expand; rec = expand(text, scene, "llm_kb")` works with the contract signature alone: with no `llm=`, the configured client (`config/m3.json` `llm`) is built once and loads on its first call. Call `import expander; expander.close()` before loading Stable Diffusion, so the LLM and SD never share the T4 (Overview §2.4). With `llm.backend` set to `"none"`, LLM modes use the template instead.

Dev (D, `llm` session):
```
python expander/batch_expand.py --scenes eval/test_scenes_dev.json --mode template --out outputs/expansions/c2_template_dev.json
python expander/batch_expand.py --scenes eval/test_scenes_dev.json --mode llm_kb   --out outputs/expansions/c3_llm_kb_dev.json
python expander/batch_expand.py --scenes eval/test_scenes_dev.json --mode llm_nokb --out outputs/expansions/c4_llm_nokb_dev.json
python expander/check_expansions.py outputs/expansions/c2_template_dev.json outputs/expansions/c3_llm_kb_dev.json outputs/expansions/c4_llm_nokb_dev.json --strict
# optional (Day 4)
python expander/batch_expand.py --scenes eval/test_scenes_dev.json --mode llm_kb --scene_negatives --out outputs/expansions/c3_neg_dev.json
```

Images (A's `gen` profile) and scores (B's `eval/score.py`, `eval` profile): `RUNBOOK_COLAB.md` steps 8 and 8b.

`batch_expand.py` flags: `--scenes --mode --out [--kb] [--cache] [--condition] [--split] [--llm default|large|fake] [--llm_model] [--llm_revision] [--no_4bit] [--prompt_version] [--limit N] [--config] [--overwrite] [--counter auto|clip|approx] [--scene_negatives]`. It resumes an existing `--out` and loads the LLM only on a cache miss. `check_expansions.py FILE... [--max_tokens N] [--json OUT] [--strict] [--names PATH] [--no_recount] [--config]`.

### Freeze (S3) and held-out run

**D's frozen expander:**

| Item | Frozen value |
|---|---|
| prompt | `v1` (`expander/prompts/system_v1.txt`) |
| template | `tpl2` |
| KB | `d-kb0` (`expander/interim_kb/kb_d0.json`), unless C's KB replaces it before the freeze |
| LLM | `Qwen/Qwen3-4B-Instruct-2507` @ `cdbee75f17c01a7cc42f958dc650907174af0554`, 4-bit NF4, temperature 0 |
| parser | the stand-in (`expander/_shims/stub_parser.py`). Since C's `kb/parser.py` landed (#49), `batch_expand.py` picks C's parser by default, so D's runs pass `--parser shim` to stay on the parser behind the scored dev records (each new record has a `parser` field) |

The frozen commit is the `main` merge of the freeze PR; D posts its sha in the Status issue. No changes after it except announced bug fixes.

**Held-out (B, after S3, private scenes and cache; never commit the cache):**
```
git checkout <D's frozen sha>
PRIV=<B's private folder>   # outside the repo and the shared Drive
python expander/batch_expand.py --scenes $PRIV/test_scenes_heldout.json --mode template --split heldout --kb expander/interim_kb/kb_d0.json --parser shim --out outputs/expansions/c2_template_heldout.json
python expander/batch_expand.py --scenes $PRIV/test_scenes_heldout.json --mode llm_kb   --split heldout --kb expander/interim_kb/kb_d0.json --parser shim --cache $PRIV/heldout_cache.json --out outputs/expansions/c3_llm_kb_heldout.json
python expander/batch_expand.py --scenes $PRIV/test_scenes_heldout.json --mode llm_nokb --split heldout --parser shim --cache $PRIV/heldout_cache.json --out outputs/expansions/c4_llm_nokb_heldout.json
python expander/check_expansions.py outputs/expansions/c2_template_heldout.json outputs/expansions/c3_llm_kb_heldout.json outputs/expansions/c4_llm_nokb_heldout.json --strict
```
- **What can be committed:** the expansion files hold only prompts, but they reveal the held-out texts' content. Commit them only when B reveals the held-out set (Day 5).
- **Images:** seeds 1-4 (`config/m3.json` `heldout_seeds`), same generator as dev.
- **Known limit:** `d-kb0` covers the 18 dev-scene characters only. A held-out character outside it gets no description, and the stand-in parser writes "the figure" in its place in the action (never its name). c2/c3 are therefore weaker on held-out scenes with uncovered characters. Report the share of such scenes (`characters` vs `unknown_entities` in the records).

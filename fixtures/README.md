# Fixtures

Small invented inputs for CPU tests and dry runs (Overview §7). Nothing here is evaluation data.

| File | What | Schema |
|---|---|---|
| `test_scenes.json` | 3 invented scenes (`f01`-`f03`, split `fixture`), tiers 1, 2 and 5 | `contracts/test_scene.schema.json` |
| `scene_specs.json` | Their SceneSpecs, from `kb.parser.parse_scene` | `contracts/scene.schema.json` |
| `expansions.json` | Their ExpandedPrompts, template mode (`batch_expand.py --mode template --counter approx`, so token counts are approximate) | `contracts/expansion.schema.json` |
| `stub_scene.json`, `stub_expansion.json` | The Day-1 stubs, kept for older tests | same |
| `img_{red,green,blue}.png`, `manifest.json` | 3 solid-colour images and a run_batch-style manifest pointing at them | |

`tests/test_contracts.py` validates all of them on every CI run.

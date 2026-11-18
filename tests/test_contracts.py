"""Contract tests (Overview §6): every shared file validates against contracts/*.schema.json, and the
evaluated expansions keep their promises (trigger first, within max_prompt_tokens, no name left in a positive)."""
import glob, json, os, re
import pytest

jsonschema = pytest.importorskip("jsonschema")
ROOT = os.path.join(os.path.dirname(__file__), "..")
P = lambda *a: os.path.join(ROOT, *a)
load = lambda p: json.load(open(p, encoding="utf-8"))
CFG = load(P("config", "m3.json"))
# evaluated conditions only; SHAP/token-budget prompt sets are analysis inputs, not ExpandedPrompts
EXPANDED = sorted(glob.glob(P("outputs", "expansions", "c[2-6]*.json")))


def errors(schema, obj):
    v = jsonschema.Draft202012Validator(load(P("contracts", schema)))
    return [e.message for e in v.iter_errors(obj)]


def test_config():
    assert errors("config.schema.json", CFG) == []
    assert CFG["default_model"] in CFG["models"]


@pytest.mark.parametrize("path", ["eval/test_scenes_dev.json", "eval/test_scenes_heldout.json", "fixtures/test_scenes.json"])
def test_test_scenes(path):
    scenes = load(P(path))
    assert len({s["id"] for s in scenes}) == len(scenes)
    for s in scenes:
        assert errors("test_scene.schema.json", s) == [], s["id"]


@pytest.mark.parametrize("path", ["fixtures/scene_specs.json", "fixtures/stub_scene.json"])
def test_scene_specs(path):
    data = load(P(path))
    for s in data if isinstance(data, list) else [data]:
        assert errors("scene.schema.json", s) == [], s["id"]


@pytest.mark.parametrize("path", EXPANDED + [P("fixtures", "expansions.json"), P("fixtures", "stub_expansion.json")],
                         ids=os.path.basename)
def test_expansions_schema(path):
    data = load(path)
    for r in data if isinstance(data, list) else [data]:
        assert errors("expansion.schema.json", r) == [], r["scene_id"]


def test_fixture_manifest_points_at_files():
    m = load(P("fixtures", "manifest.json"))
    assert len(m["images"]) == 3
    for it in m["images"]:
        assert os.path.exists(P(it["path"]))


@pytest.mark.parametrize("path", EXPANDED, ids=os.path.basename)
def test_expanded_prompts_keep_promises(path):
    names_file = P("kb", "names.json")
    names = load(names_file) if os.path.exists(names_file) else []
    names = names if isinstance(names, list) else [a for k, v in names.items() for a in [k, *v]]
    pat = re.compile(r"\b(" + "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True)) + r")\b", re.I) if names else None
    for r in load(path):
        assert r["positive"].startswith(CFG["trigger"]), r["scene_id"]
        assert r["token_count"] <= CFG["max_prompt_tokens"], r["scene_id"]
        if pat:
            assert not pat.search(r["positive"]), (r["scene_id"], pat.search(r["positive"]).group(0))

"""M4 CI checks (Overview §5, A's two checks). CPU-only.
1. config/m4.json must not duplicate anything that lives in config/m3.json.
2. eval/test_pairs_dev.json validates (skipped until B's file is merged)."""
import json, os
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
P = lambda *a: os.path.join(ROOT, *a)
load = lambda p: json.load(open(p, encoding="utf-8"))

M3 = load(P("config", "m3.json"))
M4 = load(P("config", "m4.json"))
PAIRS = P("eval", "test_pairs_dev.json")
CHECKLIST_TYPES = set(load(P("contracts", "test_scene.schema.json"))
                      ["properties"]["checklist"]["items"]["properties"]["type"]["enum"])
M3_ONLY = {"negative", "negative_prompt", "trigger", "style_tail", "tail",
           "sampler", "dev_seeds", "heldout_seeds", "seeds"}


def _keys(o):
    if isinstance(o, dict):
        for k, v in o.items():
            yield k
            yield from _keys(v)
    elif isinstance(o, list):
        for v in o:
            yield from _keys(v)


# ---- check 1: m4.json vs m3.json -------------------------------------------------------------
def test_m4_top_level_keys_not_in_m3():
    assert not set(M4) & set(M3), f"duplicated from m3.json: {set(M4) & set(M3)}"


def test_m4_has_no_m3_only_settings():
    assert not set(_keys(M4)) & M3_ONLY, f"m3-only settings found in m4.json: {set(_keys(M4)) & M3_ONLY}"


def test_m4_baseline_profile_is_fixed():
    assert M4["control"]["baseline"] == {"scale": 1.0, "guidance_start": 0.0, "guidance_end": 1.0}


def test_m4_pins_are_filled_in():
    for name, spec in M4["controlnets"].items():
        assert len(spec["revision"]) == 40, f"{name} revision is not a full commit sha"
    assert "<" not in M4["pose_detector"]["version"]


# ---- check 2: test pairs ---------------------------------------------------------------------
def pair_problems(pair, scene_ids, root=ROOT):
    out, pid = [], pair.get("id")
    n = pair.get("n_people_expected")
    if not isinstance(n, int) or not 1 <= n <= 3:
        out.append(f"{pid}: n_people_expected must be an int 1-3, got {n!r}")
    if pair.get("scene_id") not in scene_ids:
        out.append(f"{pid}: scene_id {pair.get('scene_id')!r} is not an M3 dev scene")
    pj = os.path.join(root, pair.get("pose_json") or "")
    if not os.path.isfile(pj):
        out.append(f"{pid}: pose_json file missing ({pair.get('pose_json')!r})")
    else:
        people = load(pj).get("people", [])
        if len(people) != n:
            out.append(f"{pid}: n_people_expected={n} but PoseJSON has {len(people)} people")
    ov = pair.get("checklist_override")
    if ov is not None:
        if not 3 <= len(ov) <= 6:
            out.append(f"{pid}: checklist_override needs 3-6 items, has {len(ov)}")
        for it in ov:
            if it.get("type") not in CHECKLIST_TYPES:
                out.append(f"{pid}: invalid checklist type {it.get('type')!r}")
    return out


def _scene_ids():
    d = load(P("eval", "test_scenes_dev.json"))
    sc = d if isinstance(d, list) else d.get("scenes", d)
    return {s["id"] for s in sc}


def test_pairs_dev_file_validates():
    if not os.path.exists(PAIRS):
        pytest.skip("eval/test_pairs_dev.json not merged yet (B owns it)")
    pairs = load(PAIRS)
    pairs = pairs if isinstance(pairs, list) else pairs.get("pairs", pairs)
    ids = [p["id"] for p in pairs]
    assert len(ids) == len(set(ids)), "duplicate pair ids"
    sids = _scene_ids()
    problems = [m for p in pairs for m in pair_problems(p, sids)]
    assert not problems, "\n".join(problems)


def _pose(tmp_path, n):
    (tmp_path / "p.json").write_text(json.dumps({"people": [{} for _ in range(n)]}))


def test_pair_problems_accepts_a_valid_pair(tmp_path):
    _pose(tmp_path, 2)
    pair = {"id": "x", "scene_id": "s04", "n_people_expected": 2, "pose_json": "p.json", "checklist_override": None}
    assert pair_problems(pair, {"s04"}, root=str(tmp_path)) == []


def test_pair_problems_flags_each_rule(tmp_path):
    _pose(tmp_path, 2)
    good = {"id": "x", "scene_id": "s04", "n_people_expected": 2, "pose_json": "p.json", "checklist_override": None}
    item = {"q": "?", "type": "pose", "weight": 1.0}
    bad = [
        dict(good, n_people_expected=4),                                  # more than 3 people
        dict(good, n_people_expected=3),                                  # count differs from PoseJSON
        dict(good, scene_id="s99"),                                       # not a dev scene
        dict(good, pose_json="nope.json"),                                # missing file
        dict(good, checklist_override=[item, item]),                      # fewer than 3 items
        dict(good, checklist_override=[item] * 7),                        # more than 6 items
        dict(good, checklist_override=[dict(item, type="banana")] * 3),   # invalid type
    ]
    for b in bad:
        assert pair_problems(b, {"s04"}, root=str(tmp_path)), f"not flagged: {b}"

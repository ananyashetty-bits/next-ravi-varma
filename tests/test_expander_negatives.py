"""c3_neg: scene-specific negatives (D guide Day 4, H-D5). Positive identical to c3; only the negative changes."""

import json
import re

import pytest

from expander import _cfg
from expander import batch_expand as be
from expander.cache import Cache
from expander.expand import NEG_PROMPT_VERSION, expand
from expander.llm_base import FakeLLM, load_system_prompt
from expander.names import SEED_PATH, NameMatcher, load_names

CFG = json.loads((_cfg.SHIM_DIR / "m3.default.json").read_text())
KB_PATH = str(_cfg.SHIM_DIR / "stub_kb.json")
SEED = NameMatcher(load_names(SEED_PATH))
TEXT = "Hanuman kneeling before Rama in the royal hall"
SCENE = {"id": "n1", "raw": TEXT, "epic": "Ramayana",
         "characters": [{"mention": "Rama", "canonical": "Rama", "visual": "blue-skinned crowned prince",
                         "coverage": "medium"}],
         "setting": {"name": "palace", "visual": "pillared royal hall"}, "action": "kneeling in the royal hall",
         "mood": "devotion", "unknown_entities": [], "kb_version": "stub0"}
GOOD = '{"phrases": ["a monkey-faced figure kneeling", "a blue-skinned crowned prince", "pillared royal hall"]}'
NEGS = '{"negatives": ["modern clothing", "western armour", "european castle"]}'


def fake_count(s: str) -> int:
    return sum(-(-len(w) // 6) if w[0].isalnum() else 1 for w in re.findall(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]", s))


def run(replies, **kw):
    llm = FakeLLM(list(replies))
    rec = expand(TEXT, SCENE, kw.pop("mode", "llm_kb"), llm=llm, cfg=CFG, count_fn=fake_count, matcher=SEED,
                 kb_path=KB_PATH, **kw)
    return rec, llm


def test_neg_prompt_file_exists_and_asks_for_json():
    text = load_system_prompt(NEG_PROMPT_VERSION)
    assert '{"negatives"' in text and "NEVER use proper names" in text


def test_positive_identical_to_c3_and_negative_extended():
    c3, _ = run([GOOD])
    neg, llm = run([GOOD, NEGS], scene_negatives=True)
    assert neg["positive"] == c3["positive"]
    assert neg["prompt_version"] == c3["prompt_version"]
    assert neg["negatives_added"] == ["modern clothing", "western armour", "european castle"]
    assert neg["negative"] == CFG["negative"] + ", modern clothing, western armour, european castle"
    assert neg["neg_prompt_version"] == NEG_PROMPT_VERSION and neg["neg_error"] is None
    assert neg["llm_calls"] == 2 and len(llm.calls) == 2
    assert c3.get("negatives_added") is None  # plain c3 records are unchanged


def test_names_and_duplicates_are_dropped_from_negatives():
    base_item = CFG["negative"].split(",")[0].strip()
    reply = json.dumps({"negatives": ["Ravana's palace", "modern clothing", base_item.upper(), "cars"]})
    rec, _ = run([GOOD, reply], scene_negatives=True)
    assert rec["negatives_added"] == ["modern clothing", "cars"]
    assert rec["names_after_repair"] == []


def test_bad_negatives_reply_keeps_config_negative():
    rec, _ = run([GOOD, "sorry, I cannot help"], scene_negatives=True)
    assert rec["negative"] == CFG["negative"] and rec["negatives_added"] == []
    assert rec["neg_error"] and rec["source"] == "llm"


def test_negatives_fit_the_token_budget():
    many = json.dumps({"negatives": [f"very long anachronistic thing number {i}" for i in range(6)]})
    rec, _ = run([GOOD, many], scene_negatives=True)
    assert fake_count(rec["negative"]) <= CFG["max_prompt_tokens"]
    assert len(rec["negatives_added"]) <= 6


def test_negatives_are_cached_separately(tmp_path):
    cache = Cache(str(tmp_path / "cache.json"))
    run([GOOD, NEGS], scene_negatives=True, cache=cache)
    rec, llm = run([], scene_negatives=True, cache=cache)  # every call must now hit the cache
    assert rec["llm_calls"] == 0 and rec["cache_hits"] == 2 and llm.calls == []
    assert rec["negatives_added"] == ["modern clothing", "western armour", "european castle"]


def test_scene_negatives_rejects_non_llm_modes():
    with pytest.raises(ValueError):
        expand(TEXT, SCENE, "template", cfg=CFG, count_fn=fake_count, matcher=SEED, kb_path=KB_PATH,
               scene_negatives=True)
    with pytest.raises(SystemExit):
        be.run(["--scenes", "x.json", "--mode", "template", "--out", "c3_neg_dev.json", "--scene_negatives"])

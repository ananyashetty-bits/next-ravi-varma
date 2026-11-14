"""Names test (pre-registered): 12 characters x {name prompt, description prompt}. Writes
train/names_test_scenes.json (scene format for eval/score.py, shared checklists) and
train/names_test_prompts.json (records for the generator). Checks: no names in checklists, trigger first, token limit."""
import json, os, re, sys
sys.path.insert(0, os.getcwd())
CFG = json.load(open("config/m3.json"))
KB = json.load(open("kb/kb.json", encoding="utf-8"))["characters"]
NAMES = json.load(open("kb/names.json", encoding="utf-8"))
TYPES = {"count", "figure", "pose", "attire", "object", "setting", "mood"}
C = lambda q, t, w=1.0: {"q": q, "type": t, "weight": w}
ONE = C("Is there exactly one figure in the image?", "count")
WOMAN = C("Is the figure a woman?", "figure")
MAN = C("Is the figure a man?", "figure")

SPEC = [
 ("sita", "3plus", [ONE, WOMAN, C("Is she wearing a red sari?", "attire"),
     C("Does her sari have a yellow or gold border?", "attire", 0.5), C("Is she wearing gold bangles or other gold jewellery?", "attire", 0.5)]),
 ("krishna", "3plus", [ONE, C("Does the figure have blue or dark blue skin?", "figure"),
     C("Is the figure wearing a crown with a feather in it?", "attire"), C("Is the figure wearing a yellow lower garment?", "attire", 0.5),
     C("Is the figure wearing a garland of flowers?", "attire", 0.5)]),
 ("shakuntala", "3plus", [ONE, WOMAN, C("Is she wearing a pink sari?", "attire"),
     C("Is she wearing flowers as a garland or jewellery?", "attire")]),
 ("radha", "3plus", [ONE, WOMAN, C("Does her sari have a gold border?", "attire"),
     C("Does she have flowers in her hair bun?", "attire"), C("Is she wearing a nose ring?", "attire", 0.5)]),
 ("draupadi", "3plus", [ONE, WOMAN, C("Is her long black hair worn loose and open?", "figure"),
     C("Is she wearing a sari with a thin border?", "attire", 0.5), C("Is she wearing light, thin jewellery?", "attire", 0.5)]),
 ("hanuman", "3plus", [ONE, C("Does the figure have a monkey face?", "figure"), C("Does the figure have a grey or white beard?", "figure"),
     C("Is the figure wearing a red lower garment?", "attire"), C("Is the figure holding a mace?", "object", 0.5)]),
 ("bhishma", "one", [ONE, C("Does the figure have a moustache?", "figure"), C("Does the figure have long curly hair?", "figure"),
     C("Is the figure wearing a white robe?", "attire"), C("Is there a yellow cloth tied around the waist?", "attire", 0.5)]),
 ("subhadra", "one", [ONE, WOMAN, C("Is she wearing a purple sari?", "attire"),
     C("Does she have flowers in her hair bun?", "attire", 0.5), C("Is she wearing green and white bangles?", "attire", 0.5)]),
 ("duryodhana", "one", [ONE, MAN, C("Is the figure wearing a blue tunic?", "attire"),
     C("Is the figure wearing a red turban with white feathers?", "attire"), C("Is the figure wearing pearl necklaces?", "attire", 0.5)]),
 ("savitri", "one", [ONE, WOMAN, C("Is her long dark curly hair worn open?", "figure"),
     C("Is she wearing a red sari?", "attire"), C("Does she have a red mark on her forehead?", "attire", 0.5)]),
 ("karna", "none", [ONE, MAN, C("Is the figure wearing golden armour?", "attire"), C("Is the figure wearing large gold earrings?", "attire")]),
 ("shabari", "none", [ONE, WOMAN, C("Does the woman look elderly?", "figure"), C("Does she have grey hair?", "figure", 0.5),
     C("Is she holding fruit?", "object"), C("Is she wearing simple, plain cloth?", "attire", 0.5)]),
]
try:
    from common.tokens import n_tokens
except Exception as e:
    n_tokens = None; print("token counter unavailable:", e)

def group_of(n): return "3plus" if n >= 3 else "one" if n == 1 else "none"
name_re = re.compile(r"\b(" + "|".join(sorted(map(re.escape, NAMES), key=len, reverse=True)) + r")\b", re.I)
scenes, prompts = [], []
for i, (key, group, items) in enumerate(SPEC, 1):
    c = KB[key]; sid = f"n{i:02d}"
    assert group_of(c["coverage_n"]) == group, f"{key}: {c['coverage_n']} images is not group {group}"
    cl = []
    for j, it in enumerate(items, 1):
        q = it["q"]
        assert it["type"] in TYPES and 0 < it["weight"] <= 1.0
        assert not name_re.search(q), f"{sid}: name in question: {q}"
        assert "style" not in q.lower() and "painting" not in q.lower(), q
        cl.append({"id": f"c{j}", **it})
    assert 3 <= len(cl) <= 6
    name_p = f"{CFG['trigger']}, {c['canonical']} standing"
    desc_p = f"{CFG['trigger']}, {c['visual']}, standing"
    assert not name_re.search(desc_p.replace(CFG["trigger"], "")), f"{sid}: name leaked into description prompt"
    if n_tokens:
        assert n_tokens(name_p) <= CFG["max_prompt_tokens"] and n_tokens(desc_p) <= CFG["max_prompt_tokens"], sid
    scenes.append({"id": sid, "split": "names_test", "tier": 1, "expected_epic": c["epic"],
                   "expected_characters": [c["canonical"]], "text": f"{c['canonical']} standing",
                   "seen_in_training": c["coverage_n"] > 0, "contributed_by": "C",
                   "group": group, "n_train_images": c["coverage_n"], "checklist": cl})
    for cond, p in (("nt_name", name_p), ("nt_desc", desc_p)):
        prompts.append({"scene_id": sid, "condition": cond, "positive": p, "negative": CFG["negative"]})
json.dump(scenes, open("train/names_test_scenes.json", "w"), indent=1, ensure_ascii=False)
json.dump(prompts, open("train/names_test_prompts.json", "w"), indent=1, ensure_ascii=False)
print(len(scenes), "scenes,", len(prompts), "prompts\n")
for s, pn, pd in zip(scenes, prompts[0::2], prompts[1::2]):
    print(f"{s['id']} {s['group']:5s} ({s['n_train_images']:2d} imgs) | {pn['positive']}\n      | {pd['positive']}")
    for it in s["checklist"]: print(f"        - [{it['type']}] {it['q']}")

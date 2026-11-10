"""Build kb/names.json = every canonical name + alias in kb.json, UNION D's expander/names_seed.json
(characters that have no KB entry, e.g. Ravana, still must never leak into a prompt)."""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
kb = json.load(open(os.path.join(ROOT, "kb", "kb.json"), encoding="utf-8"))
seed = json.load(open(os.path.join(ROOT, "expander", "names_seed.json"), encoding="utf-8"))

names = set()
for c in kb["characters"].values():
    names.add(c["canonical"]); names.update(c["aliases"])
for canon, aliases in seed.items():
    names.add(canon)
    if isinstance(aliases, list):
        names.update(a for a in aliases if isinstance(a, str))
names.update(["Kurukshetra", "Yamuna", "Ganga", "Sarayu", "Ashoka Vatika"])

out = os.path.join(ROOT, "kb", "names.json")
json.dump(sorted(names), open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(len(names), "names written to kb/names.json")

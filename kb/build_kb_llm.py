"""Build kb/kb_llm.json for c6: same characters/settings/aliases as kb.json, but every `visual` is written by
the local LLM WITHOUT looking at the paintings. Temperature 0. Not hand-edited: bad replies are re-asked,
then mechanically cut to 14 words and flagged."""
import copy, json, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from expander._cfg import load_config
from expander.llm import make_client

SYSTEM = ("You describe how figures and places look in classical Indian paintings. Reply with ONE line of at "
          "most 14 words: a comma-separated list of visible features (clothing, ornaments, colours, props). "
          "Never use any person's name or title. No full sentences, no explanation.")
EPIC = {"Mahabharata": "Mahabharata", "Ramayana": "Ramayana", "Puranic": "Hindu Puranic tradition"}
SETTING_LABEL = {"kurukshetra": "battlefield"}

kb = json.load(open(os.path.join(ROOT, "kb", "kb.json"), encoding="utf-8"))
names = json.load(open(os.path.join(ROOT, "kb", "names.json"), encoding="utf-8"))
pat = re.compile(r"\b(" + "|".join(sorted(map(re.escape, names), key=len, reverse=True)) + r")\b", re.I)

cfg = load_config()
client = make_client(cfg)

def clean(r):
    r = r.strip()
    r = r.splitlines()[0] if r else ""
    return r.strip().strip("\"'`").rstrip(".").strip()

def ask(user):
    flags, out = [], ""
    for attempt in range(3):
        u = user if attempt == 0 else user + "\nYour previous answer was not acceptable (" + flags[-1] + "). Answer again: one line, at most 14 words, no names."
        out = clean(client.complete(SYSTEM, u))
        leak, n = pat.search(out), len(out.split())
        if not leak and 0 < n <= 14:
            return out, flags
        flags.append("contained a name" if leak else ("empty" if n == 0 else "too long"))
    if len(out.split()) > 14:
        out = " ".join(out.split()[:14]); flags.append("TRUNCATED")
    if pat.search(out):
        flags.append("STILL HAS A NAME")
    return out, flags

new = copy.deepcopy(kb)
new["version"] = kb["version"] + "-llm"
new["_note"] = ("c6 KB: every character/setting visual written by the local LLM without images, temperature 0. "
                "Aliases, coverage numbers and moods copied unchanged from kb.json; visual_alt removed.")
report = {}
for k, c in new["characters"].items():
    user = (f"Describe the typical appearance of {c['canonical']} from the {EPIC.get(c['epic'], c['epic'])} in at "
            "most 14 words: clothing, ornaments, colours. No names.")
    c["visual"], f = ask(user)
    c.pop("visual_alt", None); c.pop("note", None)
    c["source"] = {"type": "llm", "ref": cfg["llm"]["model"]}
    report[k] = f
for k, s in new["settings"].items():
    user = (f"Describe the typical look of a {SETTING_LABEL.get(k, k)} as a setting in a classical Indian painting "
            "in at most 14 words: architecture, vegetation, colours. No names.")
    s["visual"], f = ask(user)
    s["coverage"] = "llm"
    s["source"] = {"type": "llm", "ref": cfg["llm"]["model"]}
    report["setting:" + k] = f
client.close()

new["llm_flags"] = {k: v for k, v in report.items() if v}
json.dump(new, open(os.path.join(ROOT, "kb", "kb_llm.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)

print(f"{len(new['characters'])} characters, {len(new['settings'])} settings written; flags: {new['llm_flags'] or 'none'}\n")
for k, c in new["characters"].items():
    print(f"{c['canonical']:11s} MINE: {kb['characters'][k]['visual']}\n{'':11s} LLM : {c['visual']}")
for k, s in new["settings"].items():
    print(f"[{k}] MINE: {kb['settings'][k]['visual']}\n{'':11s} LLM : {s['visual']}")

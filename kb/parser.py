"""Deterministic scene parser (no LLM). parse_scene(text, scene_id="", kb_path=None) -> SceneSpec.
Core matching logic = D's stub shim (copied to parser_core.py). Added here: fuzzy spelling repair
(rapidfuzz), extra mood words. Misspelled names are replaced by the canonical name before the core
runs, and the user's original word is restored as `mention`."""
import json, os, re
from rapidfuzz import process, fuzz
try:
    from kb import parser_core as core
except ImportError:  # run as a plain script
    import parser_core as core

DEFAULT_KB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kb.json")
CUTOFF = 88

core.MOOD_LEXICON.update({
    "fear": ("afraid", "fear", "frightened", "terrified", "trembling", "recoils", "cowering"),
    "longing": ("waiting", "longing", "yearning", "pining", "anxiously"),
    "calm": ("calm", "peaceful", "serene", "resting", "quiet"),
    "valour": ("heroic", "valour", "courage", "triumphant", "victorious", "slaying"),
})


def _known_names():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "names.json")
    try:
        names = json.load(open(path, encoding="utf-8"))
    except OSError:
        return set()
    return {t for n in names for t in core._ascii(n).lower().split()}


core._STOP.update({"Ramayana", "Mahabharata", "Mahabharat", "Kaurava", "Kauravas", "Pandava", "Pandavas",
                   "Kurukshetra", "Ayodhya", "Lanka", "Indraprastha", "Hastinapura", "Dwaraka", "Vrindavan"})


def _spelling_index(kb):
    exact, single = _known_names(), {}
    for c in kb.get("characters", {}).values():
        for a in [c["canonical"], *c.get("aliases", [])]:
            toks = core._ascii(a).lower().split()
            exact.update(toks)
            if len(toks) == 1 and len(toks[0]) >= 5:
                single[toks[0]] = c["canonical"]
    for s in kb.get("settings", {}).values():
        for a in s.get("aliases", []) + s.get("names", []):
            exact.update(core._ascii(a).lower().split())
    return exact, single


def _fix_spellings(text, kb):
    exact, single = _spelling_index(kb)
    fixes = {}
    keys = list(single)

    def repl(m):
        w = m.group(0)
        n = core._ascii(w).lower().strip()
        if not n or n in exact:
            return w
        hit = process.extractOne(n, keys, scorer=fuzz.ratio, score_cutoff=CUTOFF)
        if hit and hit[0][0] == n[0] and hit[1] >= (92 if len(n) <= 6 else CUTOFF) and (w[0].isupper() or hit[1] >= 92):
            canon = single[hit[0]]
            fixes[canon.lower()] = w
            return canon
        return w

    return re.sub(r"[^\W\d_]{5,}", repl, text), fixes


def parse_scene(text, scene_id="", kb_path=None):
    path = kb_path or DEFAULT_KB
    kb = core._load_kb(path)
    fixed, fixes = _fix_spellings(text, kb)
    spec = core.parse_scene(fixed, scene_id, path)
    spec["raw"] = text
    for ch in spec["characters"]:
        orig = fixes.get(ch["canonical"].lower())
        if orig:
            ch["mention"] = orig
    # a known name in first position that is not in the KB is still reported as unknown
    first = re.match(r"[^\W\d_]+", text.strip())
    if first:
        w = first.group(0)
        seen = " ".join(c["mention"] for c in spec["characters"]).lower().split()
        if (w.lower() in _known_names() and w not in core._STOP and w.lower() not in seen
                and w not in spec["unknown_entities"]):
            spec["unknown_entities"].insert(0, w)
    return spec

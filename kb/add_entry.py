import json, os
KB_PATH = "/content/m3/kb/kb.json"

def _load():
    if os.path.exists(KB_PATH):
        return json.load(open(KB_PATH, encoding="utf-8"))
    return {"version": "kb0.1", "characters": {}, "settings": {}, "moods": {}}

def _save(kb):
    json.dump(kb, open(KB_PATH, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

def cov(n):
    return "high" if n >= 5 else "medium" if n >= 2 else "low" if n == 1 else "none"

def add_character(key, canonical, aliases, epic, visual, n_images, refs, vocab_status="untested"):
    assert len(visual.split()) <= 14, f"visual too long: {len(visual.split())} words"
    kb = _load()
    kb["characters"][key] = {
        "canonical": canonical, "aliases": aliases, "epic": epic, "visual": visual,
        "vocab_status": vocab_status, "coverage_n": n_images, "coverage": cov(n_images),
        "source": {"type": "images", "ref": refs}}
    _save(kb)
    print("saved", key, "| coverage:", cov(n_images), "| entries now:", len(kb["characters"]))

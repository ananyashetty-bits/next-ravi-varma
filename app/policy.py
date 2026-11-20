"""Content policy enforcement for the app (owner D; rules in docs/CONTENT_POLICY.md).

check(text) -> {"allowed": bool, "category": str | None, "term": str | None, "message": str}

Deterministic, CPU-only, no model. Applied to the user's text AND to the final positive prompt (never to the
negative prompt, which contains words such as "ugly" on purpose).
- "anywhere" terms block on their own (word-boundary, case-insensitive).
- "near_figure" terms block only within `window` words of a name from M3's kb/names.json (deities and epic
  figures), e.g. "Krishna as a cartoon".
- Exempt episode names (e.g. "Draupadi's disrobing") are removed before matching, so the name of the episode alone
  never blocks; every other word is still checked.
A blocklist cannot catch everything (paraphrases, other languages); docs/CONTENT_POLICY.md says so.
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCKLIST = ROOT / "app" / "blocklist.json"
NAMES = ROOT / "kb" / "names.json"

OK = {"allowed": True, "category": None, "term": None, "message": ""}


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    text = re.sub(r"[‘’`]", "'", text)
    return re.sub(r"\s+", " ", text).strip()


def _pattern(term: str) -> re.Pattern:
    words = [re.escape(w) for w in _norm(term).split()]
    return re.compile(r"(?<![a-z0-9])" + r"[\s\-]+".join(words) + r"(?![a-z0-9])")


@lru_cache(maxsize=1)
def _rules():
    data = json.loads(BLOCKLIST.read_text(encoding="utf-8"))
    names = json.loads(NAMES.read_text(encoding="utf-8")) if NAMES.exists() else []
    name_pats = [_pattern(n) for n in sorted(set(names), key=len, reverse=True) if len(n) >= 3]
    cats = []
    for cat, spec in data["categories"].items():
        cats.append((cat, spec["message"],
                     [(t, _pattern(t)) for t in spec.get("anywhere", [])],
                     [(t, _pattern(t)) for t in spec.get("near_figure", [])]))
    exempt = [_pattern(p) for p in data.get("allowed_despite_terms", {}).get("phrases", [])]
    return data.get("window", 6), cats, name_pats, exempt


def _word_index(text: str, pos: int) -> int:
    return len(text[:pos].split())


def check(text: str) -> dict:
    """Return OK, or the first matching block with a calm, specific message."""
    window, cats, name_pats, exempt = _rules()
    t = _norm(text)
    for p in exempt:
        t = p.sub(" ", t)
    figures = [_word_index(t, m.start()) for p in name_pats for m in p.finditer(t)]
    for cat, message, anywhere, near in cats:
        for term, p in anywhere:
            if p.search(t):
                return {"allowed": False, "category": cat, "term": term, "message": message}
        if figures:
            for term, p in near:
                for m in p.finditer(t):
                    w = _word_index(t, m.start())
                    if any(abs(w - f) <= window for f in figures):
                        return {"allowed": False, "category": cat, "term": term, "message": message}
    return dict(OK)


def check_all(*texts: str) -> dict:
    """check() over several texts (user text, final positive prompt); first block wins."""
    for text in texts:
        r = check(text)
        if not r["allowed"]:
            return r
    return dict(OK)

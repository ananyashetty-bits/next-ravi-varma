"""pose_from_text (owner C; Overview 6.5; Member C file 2.2). Deterministic, no LLM, no GPU.

1. Parse with M3's kb.parser.parse_scene (or accept an already parsed scene dict).
2. Figure count = number of RESOLVED characters, 1 if none, capped at 3. unknown_entities is never used.
3. Posture family per figure from a keyword lexicon, read from the words between a name and the next name in the
   raw text (the parser's `action` has the names cut out, so it cannot say who does what).
4. Roles: "X kneeling before Y": Y (after before/to/with/at/toward) is a target and defaults to standing; names
   joined by commas or "and" before one verb ("A, B and C sitting") share that verb. Figures are placed left to
   right in order of mention.
Families are the five of eval/pose_families_dev.json: standing, seated, kneeling, lying, raised_arms.
Deliberately simple: walking, embracing and aiming a bow map to standing (they are standing postures); a verb that
belongs to an object ("a pot lying on the floor") can fool it. Report its accuracy honestly."""
import re

from pose.presets import make_posejson
from pose.skeleton import to_control

# (family, pattern), in priority order; the EARLIEST match inside a window wins, ties go to the earlier entry.
LEXICON = [
    ("kneeling", r"\bkneel\w*|\bbows?\s+(?:down|low|before)\b|\bbow(?:ed|ing)\b|\bpray\w*|\bplead\w*|\bprostrat\w*"
                 r"|\btouch\w*\s+(?:\w+\s+){0,2}feet\b|\bbeg\w*|\bsupplicat\w*"),
    ("lying", r"\blie\b|\blies\b|\blying\b|\bfallen\b|\blifeless\b|\bsleep\w*|\basleep\b|\breclin\w*"),
    ("seated", r"\bsit\b|\bsits\b|\bsitting\b|\bseated\b|\bthrone\b|\bmeditat\w*|\blisten\w*"),
    ("raised_arms", r"\braise\w*|\baloft\b|\boverhead\b|\bbless\w*"),
    # explicit standing postures; walking, embracing and aiming a bow are standing postures (no separate skeleton)
    ("standing", r"\bstand\w*|\bwalk\w*|\bmarch\w*|\bdepart\w*|\bexile\w*|\bslip\w*|\bembrac\w*"
                 r"|\baim\w*|\bshoot\w*|\bdraw\w*|\bpoint\w*"),
]
DEFAULT = "standing"
_PREP_END = re.compile(r"\b(?:before|to|with|at|toward|towards|beside|near)\s*$", re.I)
_EMPTY = re.compile(r"^[\s,;&]*(?:and)?[\s,;&]*$", re.I)


def _family(window):
    best = None
    for rank, (fam, pat) in enumerate(LEXICON):
        m = re.search(pat, window, re.I)
        if m and (best is None or (m.start(), rank) < best[0]):
            best = ((m.start(), rank), fam)
    return best[1] if best else None


def _spans(raw, chars):
    """(start, end) of each character's mention in the raw text, searched left to right; None if not found."""
    spans, cursor = [], 0
    for c in chars:
        hit = None
        for word in (c.get("mention"), c.get("canonical")):
            if not word:
                continue
            m = re.compile(re.escape(word), re.I).search(raw, cursor)
            if m:
                hit = (m.start(), m.end())
                break
        spans.append(hit)
        if hit:
            cursor = hit[1]
    return spans


def infer(scene):
    """scene: a parsed scene dict (kb.parser.parse_scene output). Returns (families, warnings); pure logic."""
    raw = scene.get("raw", "") or ""
    chars = list(scene.get("characters") or [])[:3]
    warnings = []
    if not chars:
        fam = _family(raw)
        warnings.append("no character resolved: one figure" + ("" if fam else ", posture guessed (standing)"))
        return [fam or DEFAULT], warnings
    spans = _spans(raw, chars)
    if any(s is None for s in spans):
        fam = _family(raw)
        warnings.append("could not place every name in the text: one shared posture for all figures")
        return [fam or DEFAULT] * len(chars), warnings
    windows = [raw[spans[i][1]: spans[i + 1][0] if i + 1 < len(spans) else len(raw)] for i in range(len(spans))]
    prefix = raw[: spans[0][0]]
    target = [False] + [bool(_PREP_END.search(raw[spans[i - 1][1]: spans[i][0]])) for i in range(1, len(spans))]
    fams = [_family(w) for w in windows]
    for i in range(len(fams) - 2, -1, -1):             # "A, B and C sitting": a name with no words of its own
        if fams[i] is None and not target[i] and _EMPTY.match(windows[i]):   # takes the next name's verb
            fams[i] = fams[i + 1]
    if fams[0] is None and not target[0]:
        fams[0] = _family(prefix)
    out = []
    for c, f in zip(chars, fams):
        if f is None:
            warnings.append(f"posture guessed (standing) for {c.get('mention') or c.get('canonical')}")
        out.append(f or DEFAULT)
    return out, warnings


def pose_from_text(text_or_scene, parse=None):
    """ControlImage (Overview 6.3) for a scene text or a parsed scene dict. `parse` lets tests inject a parser."""
    warnings = []
    if isinstance(text_or_scene, dict):
        scene = text_or_scene
    else:
        try:
            if parse is None:
                from kb.parser import parse_scene as parse
            scene = parse(str(text_or_scene))
        except Exception as e:                           # never crash the app on a parser problem
            scene = {"raw": str(text_or_scene), "characters": []}
            warnings.append(f"parser failed ({type(e).__name__}): one standing figure")
    families, w = infer(scene)
    n_found = len(scene.get("characters") or [])
    if n_found > 3:
        warnings.append(f"{n_found} characters found; the first 3 are used")
    pj = make_posejson(families, n=len(families))
    pj["source"] = "auto"
    pj["families"] = families
    return to_control(pj, source="auto", warnings=warnings + w)

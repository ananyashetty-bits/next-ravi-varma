"""Human ratings -> tables, paired comparisons, agreement and judge validation (Member B).

Reads the rating sheet (exported as CSV from Google Sheets) together with the PRIVATE blind key that
eval/contact_sheet.py wrote for the same round, decodes codes to conditions, and computes everything in
section B of experiments/HYPOTHESES.md that depends on human ratings.

    python eval/ratings.py \
        --round /content/ratings_r1.csv /content/drive/MyDrive/B_private/blind_key_dev.json \
        [--round <round-2 csv> <round-2 key>] \
        [--results results/*_dev.json] [--kb kb/kb.json] [--alias c3=c3_llm_kb_ckb] \
        [--out /content/drive/MyDrive/B_private/ratings_dev]

The decoded output reveals which code is which condition. Until every rater has submitted, write it only
to a private path (the script prints a PROVISIONAL banner and names who is missing).

Unit of analysis is the scene (B.1): raters are averaged within an image, and each scene has one rated
image per condition (seed --seed_img). Paired comparisons use eval/stats.paired (5000 resamples, seed 0)
with the pre-registered tie band of 0.25 rating points (B.2).
"""
import argparse
import csv
import glob
import json
import math
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DIMS = ("faithfulness", "style", "quality")
COLS = {d: f"{d}_1to5" for d in DIMS}
TIE_RATING = 0.25            # B.2
JUDGE_MIN_RHO = 0.4          # H-B1
AGREE_MIN_RHO = 0.5          # H-B5
TAXONOMY = ("wrong figure count", "wrong/missing figure identity", "wrong setting", "wrong action",
            "generic figure (name ignored)", "weak style", "anatomy/face artifacts", "other")
TAG_ALIASES = {                                    # common shorthand -> taxonomy label
    "generic figure": "generic figure (name ignored)",
    "generic figure (name ignored)": "generic figure (name ignored)",
    "missing figure identity": "wrong/missing figure identity",
    "wrong figure identity": "wrong/missing figure identity",
    "wrong/missing identity": "wrong/missing figure identity",
    "anatomy": "anatomy/face artifacts",
    "face artifacts": "anatomy/face artifacts",
    "anatomy/face artefacts": "anatomy/face artifacts",
    "figure count": "wrong figure count",
    "weak varma style": "weak style",
}
# Comparisons as (a, b): b - a is reported. Short ids are resolved with resolve().
# The first six are Overview §8; c1 -> c2 and c1 -> c4 are descriptive extras (prompt structure / LLM without KB).
DEFAULT_PAIRS = [("c0", "c1"), ("c1", "c3"), ("c4", "c3"), ("c2", "c3"), ("c6", "c3"), ("c3", "c5"),
                 ("c1", "c2"), ("c1", "c4")]
COVERAGE_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3}


# ---------------------------------------------------------------------------------------------- loading
def norm_condition(c):
    return re.sub(r"_\d+$", "", c or "")             # run_batch sometimes appended _<idx> (B deviation, 4 Nov)


def norm_tag(t):
    t = re.sub(r"\s+", " ", (t or "").strip().lower())
    if not t:
        return "", False
    if t in TAXONOMY:
        return t, True
    if t in TAG_ALIASES:
        return TAG_ALIASES[t], False
    return "other", False


def load_round(csv_path, key_path, rnd, seed_img=1):
    """Return (ratings, problems, pending, total): decoded filled rows, sheet problems, and per-rater counts
    of empty rows and of all rows."""
    key = json.load(open(key_path))
    ratings, problems, pending, total = [], [], {}, {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):            # row 1 is the header
            row = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
            sid, code, rater = row.get("scene_id", ""), row.get("code", ""), row.get("rater", "").upper()
            where = f"{os.path.basename(csv_path)} row {i} ({sid}/{code}/{rater})"
            vals = {d: row.get(COLS[d], "") for d in DIMS}
            total[rater] = total.get(rater, 0) + 1
            if not any(vals.values()):
                pending[rater] = pending.get(rater, 0) + 1
                continue
            if sid not in key or code not in key[sid]:
                problems.append(f"{where}: code {code!r} is not in the blind key for {sid}; row skipped")
                continue
            if not all(vals.values()):
                problems.append(f"{where}: some scores empty {vals}; row skipped")
                continue
            try:
                sc = {d: int(float(vals[d])) for d in DIMS}
            except ValueError:
                problems.append(f"{where}: non-numeric score {vals}; row skipped")
                continue
            if any(not 1 <= v <= 5 for v in sc.values()):
                problems.append(f"{where}: score outside 1-5 {sc}; row skipped")
                continue
            tag, exact = norm_tag(row.get("failure", ""))
            if row.get("failure") and not exact:
                problems.append(f"{where}: failure tag {row['failure']!r} counted as {tag!r}")
            k = key[sid][code]
            ratings.append({"round": rnd, "scene_id": sid, "code": code, "rater": rater,
                            "condition": norm_condition(k.get("condition")), "model": k.get("model", "?"),
                            "image": k.get("image"), "seed": seed_img, "failure": tag,
                            "notes": row.get("notes", ""), **sc})
    return ratings, problems, pending, total


def load_results(paths):
    """Automated metrics per image, keyed by (condition, image file name), plus per-condition max_train_dino.

    Ratings are matched to the exact image that was rated (same condition and same file name), never just by
    scene and seed: a re-generated image with the same seed is a different image. If a file under a `rated/`
    folder exists for a condition (the images that were actually put on the contact sheets), it replaces the
    other files for that condition.
    """
    per_file = []
    for p in sorted(paths):
        d = json.load(open(p))
        conds = {norm_condition(it.get("condition")) for it in d.get("items", [])}
        per_file.append((p, d, conds))
    rated = {c for p, _, conds in per_file if os.sep + "rated" + os.sep in p or "/rated/" in p for c in conds}
    out = {"images": {}, "dino": {}, "files": {}}
    for p, d, conds in per_file:
        is_rated = os.sep + "rated" + os.sep in p or "/rated/" in p
        for it in d.get("items", []):
            c = norm_condition(it.get("condition"))
            if c in rated and not is_rated:
                continue
            out["files"].setdefault(c, set()).add(os.path.relpath(p, ROOT) if p.startswith(ROOT) else p)
            m = {k: it[k] for k in ("checklist", "clip_raw", "style_margin", "max_train_dino") if k in it}
            out["images"][(c, os.path.basename(it.get("image") or it.get("path") or ""))] = m
            if "max_train_dino" in it:
                out["dino"].setdefault(c, []).append(it["max_train_dino"])
    return out


def p95(x):
    x = sorted(x)
    if not x:
        return float("nan")
    k = (len(x) - 1) * 0.95
    f_ = math.floor(k)
    return x[f_] + (x[min(f_ + 1, len(x) - 1)] - x[f_]) * (k - f_)


def scene_coverage(scenes, kb_path):
    """Worst KB coverage among a scene's expected characters (H-B4). Unknown characters count as 'none'."""
    if not kb_path or not os.path.exists(kb_path):
        return {}
    chars = json.load(open(kb_path)).get("characters", {})
    lookup = {}
    for c in chars.values():
        for n in [c.get("canonical", "")] + c.get("aliases", []):
            lookup[n.lower()] = c.get("coverage", "none")
    out = {}
    for s in scenes.values():
        names = s.get("expected_characters") or []
        if not names:
            out[s["id"]] = "n/a"
            continue
        covs = [lookup.get(n.lower(), "none") for n in names]
        out[s["id"]] = min(covs, key=lambda c: COVERAGE_ORDER.get(c, 0))
    return out


# ---------------------------------------------------------------------------------------------- maths
def mean(x):
    x = list(x)
    return sum(x) / len(x) if x else float("nan")


def ranks(x):
    """Average ranks (ties share the mean rank), 1-based."""
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def pearson(a, b):
    ma, mb = mean(a), mean(b)
    sa = math.sqrt(sum((v - ma) ** 2 for v in a))
    sb = math.sqrt(sum((v - mb) ** 2 for v in b))
    if sa == 0 or sb == 0:
        return float("nan")
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb)


def spearman(a, b):
    return pearson(ranks(a), ranks(b)) if len(a) >= 3 else float("nan")


def spearman_ci(a, b, n=2000, seed=0):
    """Bootstrap 95% CI of Spearman's rho, resampling items (images)."""
    if len(a) < 5:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    idx = range(len(a))
    vals = []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        r = spearman([a[i] for i in s], [b[i] for i in s])
        if not math.isnan(r):
            vals.append(r)
    if not vals:
        return (float("nan"), float("nan"))
    vals.sort()
    return (vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))])


def weighted_kappa(a, b, k=5):
    """Quadratic-weighted Cohen's kappa for two raters' 1..k scores."""
    if not a:
        return float("nan")
    n = len(a)
    obs = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[x - 1][y - 1] += 1
    ra = [sum(r) for r in obs]
    cb = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    w = [[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)]
    num = sum(w[i][j] * obs[i][j] for i in range(k) for j in range(k)) / n
    den = sum(w[i][j] * ra[i] * cb[j] for i in range(k) for j in range(k)) / (n * n)
    return 1 - num / den if den else float("nan")


# ---------------------------------------------------------------------------------------------- analysis
def per_image(ratings):
    """Average raters within each rated image. Key: (condition, scene_id)."""
    img = {}
    for r in ratings:
        img.setdefault((r["condition"], r["scene_id"]), []).append(r)
    out = {}
    for k, rs in img.items():
        out[k] = {"condition": k[0], "scene_id": k[1], "seed": rs[0]["seed"], "n_raters": len(rs),
                  "image": os.path.basename(rs[0].get("image") or ""),
                  "raters": sorted(r["rater"] for r in rs), **{d: mean(r[d] for r in rs) for d in DIMS}}
    return out


def resolve(short, conditions, aliases):
    """'c3' -> full condition name. Full names pass through. Ambiguous prefixes need --alias."""
    if short in conditions:
        return short
    if short in aliases:
        return aliases[short] if aliases[short] in conditions else None
    hits = sorted(c for c in conditions if c.startswith(short + "_"))
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise SystemExit(f"'{short}' matches {hits}; pass --alias {short}=<one of them>")
    return None


def analyse(ratings, scenes, results=None, coverage=None, pairs=None, aliases=None, n_boot=5000):
    from eval.stats import boot_ci, paired      # numpy; imported here so parsing works without it
    aliases = aliases or {}
    imgs = per_image(ratings)
    conds = sorted({c for c, _ in imgs})
    rep = {"n_ratings": len(ratings), "n_images": len(imgs), "conditions": conds}

    # scene means per condition (one image per scene per condition, so scene mean = image mean)
    sm = {c: {d: {s: v[d] for (cc, s), v in imgs.items() if cc == c} for d in DIMS} for c in conds}
    rep["by_condition"] = {
        c: {d: {"mean": mean(sm[c][d].values()), "ci95": boot_ci(sm[c][d].values(), n=n_boot),
                "min": min(sm[c][d].values()), "max": max(sm[c][d].values()), "n_scenes": len(sm[c][d])}
            for d in DIMS} for c in conds}

    # paired comparisons on human ratings (b - a)
    rep["paired"] = []
    for a, b in pairs or DEFAULT_PAIRS:
        ca, cb = resolve(a, conds, aliases), resolve(b, conds, aliases)
        if not ca or not cb or ca == cb:
            continue
        row = {"a": ca, "b": cb}
        for d in DIMS:
            row[d] = paired(sm[ca][d], sm[cb][d], n=n_boot, tie=TIE_RATING)
        row["n"] = row["faithfulness"]["n"]
        if row["n"]:
            rep["paired"].append(row)

    # tiers, seen/unseen, coverage (faithfulness)
    def group_table(fn):
        t = {}
        for (c, s), v in imgs.items():
            g = fn(s)
            if g is None:
                continue
            t.setdefault(c, {}).setdefault(str(g), []).append(v["faithfulness"])
        return {c: {g: {"mean": mean(x), "n": len(x)} for g, x in sorted(gs.items())} for c, gs in t.items()}
    rep["by_tier"] = group_table(lambda s: scenes.get(s, {}).get("tier"))
    rep["by_seen"] = group_table(lambda s: {True: "seen", False: "unseen"}.get(scenes.get(s, {}).get("seen_in_training")))
    if coverage:
        rep["by_coverage"] = group_table(
            lambda s: {"high": "high/medium", "medium": "high/medium", "low": "low/none", "none": "low/none"}
            .get(coverage.get(s)))

    # failure taxonomy counts (per rating, not per image)
    rep["failures"] = {c: {t: sum(1 for r in ratings if r["condition"] == c and r["failure"] == t)
                           for t in TAXONOMY + ("",)} for c in conds}

    # inter-rater agreement (H-B5): images with >= 2 raters, first two raters alphabetically
    by_img = {}
    for r in ratings:
        by_img.setdefault((r["condition"], r["scene_id"]), {})[r["rater"]] = r
    pairs2 = [tuple(v[x] for x in sorted(v)[:2]) for v in by_img.values() if len(v) >= 2]
    rep["agreement"] = {"n_images": len(pairs2)}
    for d in DIMS:
        a, b = [p[0][d] for p in pairs2], [p[1][d] for p in pairs2]
        rep["agreement"][d] = {"spearman": spearman(a, b), "ci95": spearman_ci(a, b),
                               "kappa_qw": weighted_kappa(a, b),
                               "exact": mean(x == y for x, y in zip(a, b)) if a else float("nan"),
                               "within1": mean(abs(x - y) <= 1 for x, y in zip(a, b)) if a else float("nan")}
    rho = rep["agreement"]["faithfulness"]["spearman"]
    rep["agreement"]["H-B5_pass"] = None if math.isnan(rho) else rho >= AGREE_MIN_RHO

    # judge validation (H-B1, H-B2): per rated image, automated metric of the same image vs mean human score
    if results:
        rows = []
        for (c, s), v in imgs.items():
            m = results["images"].get((c, v["image"]))
            if m:
                rows.append({**v, **m})
        val = {"n_images": len(rows), "unmatched": len(imgs) - len(rows)}
        for name, metric, dim in (("judge_vs_faith", "checklist", "faithfulness"),
                                  ("clip_vs_faith", "clip_raw", "faithfulness"),
                                  ("stylemargin_vs_style", "style_margin", "style")):
            pr = [(r[metric], r[dim]) for r in rows if r.get(metric) is not None]
            a, b = [p[0] for p in pr], [p[1] for p in pr]
            val[name] = {"spearman": spearman(a, b), "ci95": spearman_ci(a, b), "n": len(pr)}
        rj, rc = val["judge_vs_faith"]["spearman"], val["clip_vs_faith"]["spearman"]
        val["H-B1_pass"] = None if math.isnan(rj) else (rj >= JUDGE_MIN_RHO and (math.isnan(rc) or rj >= rc))
        rs = val["stylemargin_vs_style"]["spearman"]
        val["H-B2_pass"] = None if math.isnan(rs) else rs > 0
        rep["judge_validation"] = val

    # default pipeline (B.3), on human ratings; ranking on the scenes every candidate was rated on
    rep["default_pipeline"] = default_pipeline(sm, conds, aliases, (results or {}).get("dino", {}), paired, n_boot)
    return rep


SIMPLEST_FIRST = ["c1", "c2", "c4", "c3", "c5", "c6"]   # B.3: c1 < c2 < c4 < c3 < c5/c6 (c0 is not a candidate)
STYLE_MARGIN_PTS = 0.3
MEM_P95 = 0.85


def default_pipeline(sm, conds, aliases, dino, paired, n_boot=5000):
    """Apply the pre-registered B.3 rule literally and record every step."""
    cands = []
    for short in SIMPLEST_FIRST:
        full = resolve(short, conds, aliases)
        if full and full not in cands:
            cands.append(full)
    out = {"candidates": cands, "steps": []}
    c1 = resolve("c1", conds, aliases)
    if not c1 or len(cands) < 2:
        out["choice"] = None
        out["steps"].append("not enough rated conditions (c1 and at least one other are needed)")
        return out
    common = set.intersection(*(set(sm[c]["faithfulness"]) for c in cands))
    out["common_scenes"] = sorted(common)
    c1_style = mean(sm[c1]["style"][x] for x in common)
    rows = {}
    for c in cands:
        st = mean(sm[c]["style"][x] for x in common)
        fa = mean(sm[c]["faithfulness"][x] for x in common)
        dp = p95(dino.get(c, []))
        ok_style = st >= c1_style - STYLE_MARGIN_PTS
        ok_mem = (not math.isnan(dp)) and dp <= MEM_P95
        rows[c] = {"faithfulness": fa, "style": st, "dino_p95": dp, "eligible": ok_style and ok_mem,
                   "style_ok": ok_style, "mem_ok": ok_mem}
    out["table"] = rows
    elig = [c for c in cands if rows[c]["eligible"]]
    out["steps"].append(f"eligible (style >= c1's {c1_style:.2f} - {STYLE_MARGIN_PTS}, dino p95 <= {MEM_P95}): "
                        + (", ".join(elig) or "none"))
    if not elig:
        out["choice"] = None
        return out
    top = max(elig, key=lambda c: (rows[c]["faithfulness"], -cands.index(c)))
    out["top"] = top
    out["steps"].append(f"highest mean human faithfulness on the {len(common)} common scenes: {top} "
                        f"({rows[top]['faithfulness']:.2f})")
    simpler = [c for c in elig if cands.index(c) < cands.index(top)]
    choice = top
    if simpler:
        nxt = simpler[-1]
        pr = paired({x: sm[nxt]["faithfulness"][x] for x in common}, {x: sm[top]["faithfulness"][x] for x in common},
                    n=n_boot, tie=TIE_RATING)
        out["vs_next_simpler"] = {"a": nxt, "b": top, **pr}
        incl0 = pr["ci95"][0] <= 0 <= pr["ci95"][1]
        out["steps"].append(f"{nxt} -> {top}: diff {pr['diff']:+.2f}, CI [{pr['ci95'][0]:+.2f}, {pr['ci95'][1]:+.2f}]"
                            + (" includes 0, so the simpler one is chosen" if incl0 else " excludes 0, so the top one stays"))
        if incl0:
            choice = nxt
    else:
        out["steps"].append(f"{top} is the simplest eligible condition")
    out["choice"] = choice
    return out


# ---------------------------------------------------------------------------------------------- report
def f(x, nd=2):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def verdict(x):
    return "not enough data" if x is None else ("met" if x else "not met")


def ci(t, nd=2):
    return f"[{f(t[0], nd)}, {f(t[1], nd)}]"


def to_markdown(rep, header=""):
    L = [header] if header else []
    L += ["### Human ratings by condition (mean over scenes [95% CI]; range)", "",
          "| Condition | n scenes | Faithfulness | Style | Quality |", "|---|---|---|---|---|"]
    for c, v in rep["by_condition"].items():
        cells = [f"{f(v[d]['mean'])} {ci(v[d]['ci95'])}; {f(v[d]['min'], 1)}–{f(v[d]['max'], 1)}" for d in DIMS]
        L.append(f"| `{c}` | {v['faithfulness']['n_scenes']} | " + " | ".join(cells) + " |")
    if rep["paired"]:
        L += ["", f"### Paired comparisons (b − a per scene; tie band {TIE_RATING})", "",
              "| a → b | n | Faithfulness diff [CI] W/T/L | Style diff [CI] | Quality diff [CI] |", "|---|---|---|---|---|"]
        for p in rep["paired"]:
            fa = p["faithfulness"]
            L.append(f"| `{p['a']}` → `{p['b']}` | {p['n']} | {f(fa['diff'])} {ci(fa['ci95'])} "
                     f"{fa['win']}/{fa['tie']}/{fa['loss']} | {f(p['style']['diff'])} {ci(p['style']['ci95'])} | "
                     f"{f(p['quality']['diff'])} {ci(p['quality']['ci95'])} |")
    for title, key in (("Faithfulness by tier", "by_tier"), ("Faithfulness, seen vs unseen", "by_seen"),
                       ("Faithfulness by KB coverage (worst character in the scene)", "by_coverage")):
        if not rep.get(key):
            continue
        groups = sorted({g for v in rep[key].values() for g in v})
        L += ["", f"### {title} (mean, n scenes)", "", "| Condition | " + " | ".join(groups) + " |",
              "|---" * (len(groups) + 1) + "|"]
        for c, v in rep[key].items():
            L.append(f"| `{c}` | " + " | ".join(f"{f(v[g]['mean'])} ({v[g]['n']})" if g in v else "–" for g in groups) + " |")
    L += ["", "### Failure tags (count of ratings)", "", "| Condition | " + " | ".join(TAXONOMY) + " | none |",
          "|---" * (len(TAXONOMY) + 2) + "|"]
    for c, v in rep["failures"].items():
        L.append(f"| `{c}` | " + " | ".join(str(v[t]) for t in TAXONOMY) + f" | {v['']} |")
    ag = rep["agreement"]
    L += ["", f"### Inter-rater agreement ({ag['n_images']} images with two raters)", "",
          "| Dimension | Spearman ρ [CI] | Weighted κ | Exact | Within 1 |", "|---|---|---|---|---|"]
    for d in DIMS:
        a = ag[d]
        L.append(f"| {d} | {f(a['spearman'])} {ci(a['ci95'])} | {f(a['kappa_qw'])} | {f(a['exact'])} | {f(a['within1'])} |")
    L.append(f"\nH-B5 (faithfulness ρ ≥ {AGREE_MIN_RHO}): **{verdict(ag['H-B5_pass'])}**")
    if "judge_validation" in rep:
        jv = rep["judge_validation"]
        L += ["", f"### Judge validation ({jv['n_images']} rated images matched to results/; {jv['unmatched']} unmatched)", "",
              "| Pair | n | Spearman ρ [95% CI] |", "|---|---|---|"]
        for name, label in (("judge_vs_faith", "checklist judge vs human faithfulness"),
                            ("clip_vs_faith", "clip_raw vs human faithfulness"),
                            ("stylemargin_vs_style", "style_margin vs human style")):
            L.append(f"| {label} | {jv[name]['n']} | {f(jv[name]['spearman'])} {ci(jv[name]['ci95'])} |")
        L.append(f"\nH-B1 (judge ρ ≥ {JUDGE_MIN_RHO} and ≥ CLIP's): **{verdict(jv['H-B1_pass'])}**. "
                 f"H-B2 (style_margin ρ > 0): **{verdict(jv['H-B2_pass'])}**.")
    dp = rep.get("default_pipeline")
    if dp and dp.get("table"):
        L += ["", f"### Default pipeline (B.3 rule; human ratings on the {len(dp['common_scenes'])} scenes every candidate was rated on)", "",
              "| Condition | Faithfulness | Style | `max_train_dino` p95 | Eligible |", "|---|---|---|---|---|"]
        for c in dp["candidates"]:
            r = dp["table"][c]
            why = "" if r["eligible"] else " (" + ", ".join(x for x, ok in (("style", r["style_ok"]), ("memorisation", r["mem_ok"])) if not ok) + ")"
            L.append(f"| `{c}` | {f(r['faithfulness'])} | {f(r['style'])} | {f(r['dino_p95'], 3)} | {'yes' if r['eligible'] else 'no' + why} |")
        L += [""] + [f"{i}. {t}" for i, t in enumerate(dp["steps"], 1)]
        L.append(f"\n**Default pipeline by the rule: `{dp['choice']}`**")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round", nargs=2, action="append", metavar=("CSV", "KEY"), required=True,
                    help="rating sheet CSV and its blind key; repeat for each round")
    ap.add_argument("--scenes", default=os.path.join(ROOT, "eval", "test_scenes_dev.json"))
    ap.add_argument("--results", nargs="*", default=None, help="results/*.json for judge validation (globs ok)")
    ap.add_argument("--kb", default=os.path.join(ROOT, "kb", "kb.json"), help="for the coverage table (H-B4)")
    ap.add_argument("--alias", action="append", default=[], help="short=full condition, e.g. c3=c3_llm_kb_ckb")
    ap.add_argument("--seed_img", type=int, default=1, help="seed of the images on the contact sheets")
    ap.add_argument("--raters", nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--out", default=None, help="write <out>.json and <out>.md (private path until everyone has rated)")
    a = ap.parse_args()

    scenes = {s["id"]: s for s in json.load(open(a.scenes))}
    ratings, problems, pending, total = [], [], {}, {}
    for i, (c, k) in enumerate(a.round, start=1):
        r, p, pend, tot = load_round(c, k, i, a.seed_img)
        ratings += r
        problems += p
        for x, n in pend.items():
            pending[x] = pending.get(x, 0) + n
        for x, n in tot.items():
            total[x] = total.get(x, 0) + n
    if not ratings:
        sys.exit("no filled rating rows found")
    paths = [p for g in (a.results or []) for p in sorted(glob.glob(g))]
    aliases = dict(x.split("=", 1) for x in a.alias)
    rep = analyse(ratings, scenes, load_results(paths) if paths else None, scene_coverage(scenes, a.kb),
                  aliases=aliases)
    rep["problems"], rep["pending_rows"] = problems, pending

    done = {r: sum(1 for x in ratings if x["rater"] == r) for r in a.raters}
    header = ["## Human ratings (usable rows / rows assigned): "
              + ", ".join(f"{r} {done[r]}/{total.get(r, 0)}" for r in a.raters)]
    if any(pending.values()):
        header.append(f"\n> **PROVISIONAL**: rows still empty for {', '.join(f'{r} ({n})' for r, n in pending.items() if n)}. "
                      "Do not share decoded results until every rater has submitted.")
    if problems:
        header.append("\n**Problems in the sheet:**\n" + "\n".join(f"- {p}" for p in problems))
    md = to_markdown(rep, "\n".join(header) + "\n")
    print(md)
    if a.out:
        if os.path.abspath(a.out).startswith(ROOT) and any(pending.values()):
            sys.exit("Refusing to write decoded provisional results inside the repo. Use a private Drive path.")
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        json.dump(rep, open(a.out + ".json", "w"), indent=1, default=list)
        open(a.out + ".md", "w").write(md)
        print("wrote", a.out + ".json", a.out + ".md")


if __name__ == "__main__":
    main()

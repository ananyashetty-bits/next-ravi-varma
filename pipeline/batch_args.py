"""Torch-free pieces of pipeline/run_batch.py, so CPU-only CI can test them (Overview 6.5)."""
import argparse, json

CONTROLS = ["none", "openpose", "openpose_full", "canny"]


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expansions"); ap.add_argument("--raw")
    ap.add_argument("--condition", required=True)         # the run's label; always overrides the records
    ap.add_argument("--model", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs")
    ap.add_argument("--control", choices=CONTROLS, default="none")          # none = p0
    ap.add_argument("--control_profile", choices=["baseline", "tuned"], default="baseline")
    ap.add_argument("--control_source", choices=["pair", "auto"], default="pair")
    for k in ("control_scale", "guidance_start", "guidance_end", "lora_scale"):   # sweep only
        ap.add_argument("--" + k, type=float, default=None)
    return ap


def validate(args):
    """Returns an error message, or None."""
    if args.control == "none":
        for k in ("control_scale", "guidance_start", "guidance_end"):
            if getattr(args, k) is not None:
                return f"--{k} needs --control openpose|openpose_full|canny"
        return None
    if args.control == "canny" and args.control_source == "pair":
        return "canny is not wired to pairs yet (needs C's extract_edges)"
    if args.control_source == "pair" and not args.pairs:
        return "--control_source pair needs --pairs"
    return None


def load_pairs(path):
    d = json.load(open(path, encoding="utf-8"))
    d = d if isinstance(d, list) else d.get("pairs", d)
    return {p["scene_id"]: p for p in d}


def filter_to_pairs(items, pairs):
    missing = sorted(set(pairs) - {it["scene_id"] for it in items})
    if missing:
        raise ValueError(f"pairs without a prompt record: {missing}")
    return [it for it in items if it["scene_id"] in pairs]


def settings(args, profile_fn):
    """Control settings for this run: the profile first, then explicit sweep flags on top. None for p0."""
    if args.control == "none":
        return None
    s = dict(profile_fn(args.control_profile))
    for k in ("control_scale", "guidance_start", "guidance_end", "lora_scale"):
        if getattr(args, k) is not None:
            s[k] = getattr(args, k)
    return s


def expansion_items(records, condition, model, default_model):
    """One work item per prompt record. `condition` is the run label and always wins over the record's own."""
    items = []
    for e in records:
        c = e.get("coalition")
        items.append(dict(scene_id=e["scene_id"], name=f"{e['scene_id']}_{c}" if c else e["scene_id"],
                          condition=condition, prompt=e["positive"], negative=e["negative"],
                          model=model or default_model, coalition=c))
    return items

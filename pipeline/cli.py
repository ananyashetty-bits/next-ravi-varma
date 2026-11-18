"""M3 pipeline CLI. Run from the repo root:

  python pipeline/cli.py generate "Hanuman kneeling before Rama" --seed 1 --out x.png   [--mode raw] [--model rrv_v1]
  python pipeline/cli.py batch    --raw eval/test_scenes_dev.json --condition c1_lora_raw --seeds 1 2 --out <dir>
  python pipeline/cli.py batch    --expansions outputs/expansions/c4_llm_nokb_dev.json --seeds 1 2 --out <dir>
  python pipeline/cli.py expand   --scenes eval/test_scenes_dev.json --mode llm_nokb --out <file> [any batch_expand.py option]
  python pipeline/cli.py evaluate --manifest <dir>/manifest.json --out results/<condition>@<model>_<split>.json [any score.py option]

`generate` loads the model in-process; `batch`, `expand` and `evaluate` run the scripts that made every evaluated
result (pipeline/run_batch.py, expander/batch_expand.py, eval/score.py), so their options are passed through.
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [ROOT] + [p for p in sys.path if os.path.abspath(p or ".") != _HERE]   # pipeline/eval.py would shadow eval/

MODES = ["raw", "template", "llm_kb", "llm_nokb", "llm_kb_layout"]


def _run(script, args):
    cmd = [sys.executable, os.path.join(ROOT, script), *args]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def run_generate(a):
    from pipeline.generate_api import text_to_image
    from common.tokens import n_tokens
    t0 = time.time()
    image, exp, scene = text_to_image(a.prompt, seed=a.seed, mode=a.mode, model=a.model)
    print("characters:", [c["canonical"] for c in scene.get("characters", [])], "| epic:", scene.get("epic"))
    print("prompt:", exp["positive"])
    print("negative:", exp["negative"])
    print(f"tokens: {n_tokens(exp['positive'])} | {time.time() - t0:.1f} s (includes model load on the first call)")
    image.save(a.out)
    print("saved", a.out)


def run_batch(a, extra):
    args = ["--out", a.out, "--seeds", *map(str, a.seeds)]
    if a.model:
        args += ["--model", a.model]
    if a.raw:
        args += ["--raw", a.raw, "--condition", a.condition or "c1_lora_raw"]
    elif a.expansions:
        args += ["--expansions", a.expansions]
    else:
        sys.exit("batch needs --raw or --expansions")
    _run("pipeline/run_batch.py", args + extra)


def main():
    ap = argparse.ArgumentParser(description="Next Ravi Varma M3 pipeline CLI")
    sub = ap.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="one image from text (text_to_image)")
    g.add_argument("prompt")
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--mode", default="raw", choices=MODES, help="raw = the default pipeline (c1)")
    g.add_argument("--model", default=None, help="rrv_v1 (default) or base")
    g.add_argument("--out", required=True)

    b = sub.add_parser("batch", help="many images (pipeline/run_batch.py)")
    b.add_argument("--raw"); b.add_argument("--expansions")
    b.add_argument("--condition", help="with --raw: c0_base_raw or c1_lora_raw (default)")
    b.add_argument("--model", default=None)
    b.add_argument("--seeds", nargs="+", type=int, required=True)
    b.add_argument("--out", required=True)

    sub.add_parser("expand", help="expand a scene file (expander/batch_expand.py; same options)", add_help=False)
    sub.add_parser("evaluate", help="score images (eval/score.py; same options)", add_help=False)

    a, extra = ap.parse_known_args()
    if a.command == "generate":
        if extra:
            ap.error(f"unrecognized arguments: {' '.join(extra)}")
        run_generate(a)
    elif a.command == "batch":
        run_batch(a, extra)
    elif a.command == "expand":
        _run("expander/batch_expand.py", extra)
    elif a.command == "evaluate":
        _run("eval/score.py", extra)


if __name__ == "__main__":
    main()
